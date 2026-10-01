"""Native fast lane: throughput decode with token-level custody, exact to the layer path.

Every fixture is a tiny random-config model on CPU in fp32; no checkpoints are downloaded.
Real-model GPU throughput and parity are measured on a scheduler (see
experiments/2026-09-30-saturn-pub-fast-lane).
"""

from __future__ import annotations

import pytest
import torch

from saturn_pub.adapters.decoder import SUPPORTED_FAMILIES, DecoderAdapter
from saturn_pub.core import Session, fold_custody
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

PROMPT = [5, 7, 11, 3]


@pytest.fixture(autouse=True)
def _no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev)


def _layer_step(adapter, ids, new_tokens):
    session = adapter.session(ids)
    target = session.read("tokens").shape[1] + new_tokens
    while session.read("tokens").shape[1] < target:
        session.continue_()
    return session


@pytest.mark.parametrize("family", SUPPORTED_FAMILIES)
def test_native_decode_equals_layer_stepping(family):
    """The task's core correctness contract: native fast decode == fine-grained layer stepping,
    bit-exact in tokens, logits, and the full sealed payload (incl. the KV cache)."""
    adapter = DecoderAdapter.tiny(family, seed=5)
    fast = adapter.session(PROMPT)
    fast.generate(6)
    layer = _layer_step(adapter, PROMPT, 6)
    assert fast.read("tokens")[0].tolist() == layer.read("tokens")[0].tolist()
    torch.testing.assert_close(fast.read("logits"), layer.read("logits"), rtol=0, atol=0)
    assert describe(fast.capture().payload) == describe(layer.capture().payload)


@pytest.mark.parametrize("family", ["gpt_neox", "qwen2", "gpt2"])
def test_native_decode_matches_native_greedy(family):
    adapter = DecoderAdapter.tiny(family, seed=5)
    tokens = torch.tensor([PROMPT])
    expected = adapter.model.generate(
        tokens,
        attention_mask=torch.ones_like(tokens),
        max_new_tokens=5,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )[0].tolist()
    session = adapter.session(PROMPT)
    session.generate(5)
    assert session.read("tokens")[0].tolist() == expected


def test_generate_then_switch_to_layer_stepping():
    """Run natively to a token boundary, then step inside the next token by layers."""
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    session = adapter.session(PROMPT)
    session.generate(3)
    boundary_tokens = session.read("tokens").shape[1]
    # continue_ from the fast-lane boundary advances through the next token's layers
    session.continue_(1)  # embed
    assert session.read("tokens").shape[1] == boundary_tokens  # still mid-token
    before = session.read("tokens").shape[1]
    while session.read("tokens").shape[1] == before:
        session.continue_()
    assert session.read("tokens").shape[1] == boundary_tokens + 1


def test_mid_token_generate_honors_intervention_then_fast():
    """generate() mid-token completes the partial token on the fine-grained path first."""
    from saturn_pub import Act

    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    base = adapter.session(PROMPT)
    base.continue_(adapter.layers + 1)  # mid-token: hidden resident, stack complete
    parent = base.capture()
    plain, zeroed = base.fork(parent), base.fork(parent)
    zeroed.apply(Act.zero("hidden"))
    plain.generate(3)
    zeroed.generate(3)
    # the first (completed) token diverges because the carrier was zeroed
    assert plain.read("tokens")[0].tolist() != zeroed.read("tokens")[0].tolist()


def test_fast_generate_capture_replays_bit_exact(tmp_path):
    adapter = DecoderAdapter.tiny("llama", seed=5)
    session = adapter.session(PROMPT)
    session.generate(5)
    cut = session.capture()
    expected = session.read("tokens")[0].tolist()
    store = LocalStore(tmp_path)
    reopened = Session.from_cut(adapter, store.load(store.save(cut)))
    assert reopened.read("tokens")[0].tolist() == expected
    reopened.generate(2)  # and the lane keeps working after rehydration
    assert reopened.read("tokens").shape[1] == len(expected) + 2


def test_custody_chain_is_token_level_and_reproducible():
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    a = adapter.session(PROMPT)
    receipt = a.generate(4).to_dict()
    assert receipt["operation"] == "generate"
    assert receipt["lane"] == "native-fast"
    # recompute the chain from the recorded seed + token ids
    head = a.adapter.custody_seed(adapter.session(PROMPT)._state)
    assert head == receipt["parent_custody"]
    for token_id in receipt["tokens_generated"]:
        head = fold_custody(head, f"generate:{int(token_id)}")
    assert head == receipt["result_custody"] == a._custody
    # a fresh identical run reproduces the chain head bit-for-bit
    b = adapter.session(PROMPT)
    assert b.generate(4).to_dict()["result_custody"] == receipt["result_custody"]


def test_seeded_sampling_is_replayable():
    adapter = DecoderAdapter.tiny("qwen2", seed=5)
    sampler = {"temperature": 0.9, "top_k": 20}
    a = adapter.session(PROMPT)
    a.generate(6, sampler=sampler, seed=1234)
    b = adapter.session(PROMPT)
    b.generate(6, sampler=sampler, seed=1234)
    assert a.read("tokens")[0].tolist() == b.read("tokens")[0].tolist()
    c = adapter.session(PROMPT)
    c.generate(6, sampler=sampler, seed=9999)
    # a different seed is overwhelmingly unlikely to reproduce the same 6 tokens
    assert c.read("tokens")[0].tolist() != a.read("tokens")[0].tolist()


@pytest.mark.parametrize("family", ["gpt_neox", "qwen2", "gpt2"])
def test_batch_generate_matches_single_row_equal_length(family):
    adapter = DecoderAdapter.tiny(family, seed=5)
    prompts = [[5, 7, 11, 3], [9, 2, 4, 6], [1, 1, 2, 3]]
    out = adapter.batch_generate(prompts, 5)
    assert out["batch"] == 3 and all(len(row) == 5 for row in out["tokens"])
    for prompt, row in zip(prompts, out["tokens"]):
        single = adapter.session(prompt)
        single.generate(5)
        assert single.read("tokens")[0, len(prompt) :].tolist() == row


@pytest.mark.parametrize("family", ["gpt_neox", "qwen2"])
def test_batch_generate_left_padding_matches_single_row(family):
    """Unequal-length prompts (left-padded + masked + explicit positions) match a B=1 decode."""
    adapter = DecoderAdapter.tiny(family, seed=5)
    prompts = [[5, 7, 11, 3, 9, 2], [9, 2, 4], [1]]
    out = adapter.batch_generate(prompts, 4)
    for prompt, row in zip(prompts, out["tokens"]):
        single = adapter.session(prompt)
        single.generate(4)
        assert single.read("tokens")[0, len(prompt) :].tolist() == row


def test_generate_rejects_bad_budget():
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    session = adapter.session(PROMPT)
    with pytest.raises(ValueError, match="positive integer"):
        session.generate(0)

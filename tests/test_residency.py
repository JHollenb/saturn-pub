"""Block-streamed LM residency: streamed == resident bitwise, exact capture/restore/replay.

Every fixture is a tiny random-config model and runs on CPU; no checkpoints are downloaded
and no CUDA is required. In streamed residency the frozen weights stay in host memory and one
native block is copied to the execution device at a time; the execution device here is the CPU,
so "streamed vs resident" exercises the ``functional_call`` substitution path, not a real
host->device copy. Real-weight CUDA validation (peak VRAM, tokens/s, 7B that does not fit
resident) lives in ``experiments/lm_block_residency`` and runs on a scheduler.
"""

import json
import subprocess
import sys

import pytest
import torch

from saturn_pub import Act, Session
from saturn_pub.adapters._residency import BlockResidency
from saturn_pub.adapters.decoder import SUPPORTED_FAMILIES, DecoderAdapter
from saturn_pub.adapters.mamba import MambaAdapter
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.store import LocalStore

PROMPT = [5, 7, 11, 3, 9, 2]
# Families plus the dedicated Qwen adapter, tagged so a fixture can dispatch the constructor.
ALL_ADAPTERS = (*SUPPORTED_FAMILIES, "mamba", "qwen-adapter")
CACHE_SLOTS = ("keys", "values", "conv", "recur")


def _pair(tag):
    """Return (resident, streamed) adapters built from the same seed (identical weights)."""
    if tag == "qwen-adapter":
        return QwenAdapter.tiny(residency="resident"), QwenAdapter.tiny(residency="streamed")
    if tag == "mamba":
        return MambaAdapter.tiny(residency="resident"), MambaAdapter.tiny(residency="streamed")
    return (
        DecoderAdapter.tiny(tag, residency="resident"),
        DecoderAdapter.tiny(tag, residency="streamed"),
    )


# --- streamed == resident, bitwise -----------------------------------------------------


@pytest.mark.parametrize("tag", ALL_ADAPTERS)
def test_resident_contract_unchanged_streamed_declares_residency(tag):
    resident, streamed = _pair(tag)
    # Resident execution stays byte-identical to the historical contract (no new field),
    # so previously recorded resident receipts keep verifying; streamed declares its mode.
    assert "residency" not in resident.execution
    assert streamed.execution["residency"]["mode"] == "streamed"
    assert streamed.execution["residency"]["host_pinned"] is False


@pytest.mark.parametrize("tag", ALL_ADAPTERS)
def test_streamed_equals_resident_bitwise_greedy_decode(tag):
    resident, streamed = _pair(tag)
    res_session = resident.session(PROMPT)
    resident.generate(res_session, tokens=6)
    str_session = streamed.session(PROMPT)
    streamed.generate(str_session, tokens=6)
    assert res_session.read("tokens")[0].tolist() == str_session.read("tokens")[0].tolist()
    assert torch.equal(res_session.read("logits"), str_session.read("logits"))
    # The uninstrumented comparator is itself block-streamed and must match bitwise.
    assert torch.equal(resident.native_logits(PROMPT), streamed.native_logits(PROMPT))


@pytest.mark.parametrize("tag", ALL_ADAPTERS)
def test_streamed_prefill_cache_resident_and_bitwise(tag):
    resident, streamed = _pair(tag)
    res_cut = resident.session(PROMPT).capture()
    str_cut = streamed.session(PROMPT).capture()
    present = [slot for slot in CACHE_SLOTS if slot in str_cut.payload]
    assert present, "a streamed session must expose a layer cache"
    for slot in present:
        for resident_tensor, streamed_tensor in zip(res_cut.payload[slot], str_cut.payload[slot]):
            # The cache is built on the execution device and never streams back to host.
            assert streamed_tensor.device == streamed.device
            assert torch.equal(resident_tensor, streamed_tensor)


@pytest.mark.parametrize("tag", ALL_ADAPTERS)
def test_streamed_and_resident_states_are_bitwise_identical(tag):
    from saturn_pub.values import describe

    resident, streamed = _pair(tag)
    res_session, str_session = resident.session(PROMPT), streamed.session(PROMPT)
    res_session.continue_(2)
    str_session.continue_(2)
    # One describe() over the whole payload content-hashes every tensor (caches included).
    assert describe(res_session.capture().payload) == describe(str_session.capture().payload)


# --- capture / fork / Act / compare / commit / restore under streaming -----------------


@pytest.mark.parametrize("tag", ALL_ADAPTERS)
def test_streamed_fork_isolation_intervention_and_replay(tag):
    _, streamed = _pair(tag)
    session = streamed.session(PROMPT)
    session.continue_(streamed.layers + 1)  # full stack; hidden carrier still resident
    parent = session.capture()
    native, candidate = session.fork(parent), session.fork(parent)
    candidate.apply(Act.zero("hidden"))
    streamed.generate(native)
    streamed.generate(candidate)
    assert not torch.equal(native.read("logits"), candidate.read("logits"))
    replay_a, replay_b = session.fork(parent), session.fork(parent)
    streamed.generate(replay_a, 3)
    streamed.generate(replay_b, 3)
    assert replay_a.read("tokens")[0].tolist() == replay_b.read("tokens")[0].tolist()
    assert replay_a.compare(replay_b)["equal_payload"]


@pytest.mark.parametrize("tag", ALL_ADAPTERS)
def test_streamed_store_roundtrip_restore_is_exact(tag, tmp_path):
    _, streamed = _pair(tag)
    session = streamed.session(PROMPT)
    session.continue_(2)  # mid-layer cut
    cut = session.capture()
    expected = session.fork(cut)
    streamed.generate(expected, 3)
    expected_tokens = expected.read("tokens")[0].tolist()

    store = LocalStore(tmp_path)
    identifier = store.save(cut)
    reloaded = store.load(identifier)
    assert reloaded.fingerprint == identifier
    restored = Session(streamed, reloaded.payload)
    restored.restore(reloaded)
    streamed.generate(restored, 3)
    assert restored.read("tokens")[0].tolist() == expected_tokens


def test_streamed_cut_is_incompatible_with_resident_adapter(tmp_path):
    resident, streamed = _pair("qwen2")
    session = streamed.session(PROMPT)
    session.continue_(2)
    store = LocalStore(tmp_path)
    restored = store.load(store.save(session.capture()))
    # Residency is part of the execution contract: a streamed cut only rehydrates on a
    # streamed adapter, and vice versa.
    with pytest.raises(ValueError, match="incompatible"):
        resident.session([5]).restore(restored)


# --- durable custody across a fresh interpreter (streamed) -----------------------------

_FRESH = """
import json, sys, torch
torch.set_num_threads(1)
from saturn_pub import Session
from saturn_pub.store import LocalStore
from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.adapters.mamba import MambaAdapter
from saturn_pub.adapters.qwen import QwenAdapter
root, manifest = sys.argv[1], json.loads(sys.argv[2])
store = LocalStore(root)
for tag, entry in manifest.items():
    identifier, expected = entry
    if tag == 'mamba':
        adapter = MambaAdapter.tiny(residency='streamed')
    elif tag == 'qwen-adapter':
        adapter = QwenAdapter.tiny(residency='streamed')
    else:
        adapter = DecoderAdapter.tiny(tag, residency='streamed')
    cut = store.load(identifier)
    session = Session(adapter, cut.payload)
    session.restore(cut)
    adapter.generate(session, 2)
    got = int(session.read('tokens')[0, -1])
    assert got == expected, (tag, got, expected)
print('FRESH_OK', len(manifest))
"""


def test_streamed_fresh_process_replay_of_mid_layer_cut_is_exact(tmp_path):
    store = LocalStore(tmp_path)
    manifest = {}
    for tag in ("qwen2", "gpt2", "mamba", "qwen-adapter"):
        _, streamed = _pair(tag)
        session = streamed.session(PROMPT)
        session.continue_(2)  # mid-layer cut
        cut = session.capture()
        identifier = store.save(cut)
        branch = session.fork(cut)
        streamed.generate(branch, 2)
        manifest[tag] = [identifier, int(branch.read("tokens")[0, -1])]
    result = subprocess.run(
        [sys.executable, "-c", _FRESH, str(tmp_path), json.dumps(manifest)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip().endswith(str(len(manifest)))


# --- BlockResidency guards -------------------------------------------------------------


def test_block_residency_rejects_bad_mode():
    with pytest.raises(ValueError, match="residency mode"):
        BlockResidency("cpu", mode="paged")


def test_host_pinning_requires_streamed_cuda():
    with pytest.raises(ValueError, match="host pinning only applies to streamed"):
        BlockResidency("cpu", mode="resident", pin_host=True)
    with pytest.raises(ValueError, match="only useful for a CUDA"):
        BlockResidency("cpu", mode="streamed", pin_host=True)


def test_adapter_rejects_host_pinning_on_cpu_execution():
    with pytest.raises(ValueError, match="CUDA"):
        DecoderAdapter.tiny("qwen2", residency="streamed", device="cpu", pin_host=True)


def test_causal_mask_kwargs_is_transformers_version_robust():
    """The streamed backbone builds create_causal_mask kwargs that work on either transformers
    major: 5.x takes ``inputs_embeds`` and no ``cache_position``; 4.x takes ``input_embeds`` plus
    a required ``cache_position``. The real-weight Gemma-2 run surfaced the 4.x breakage (the
    local suite runs on 5.x), so this pins the version branch offline."""
    from saturn_pub.adapters.decoder import _causal_mask_kwargs

    def tf5(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
        return "tf5"

    def tf4(
        config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None
    ):
        return "tf4"

    sentinel = object()
    k5 = _causal_mask_kwargs(
        tf5, config="c", hidden=sentinel, cache="kv", position_ids="p", cache_position="cp"
    )
    assert "inputs_embeds" in k5 and k5["inputs_embeds"] is sentinel
    assert "cache_position" not in k5 and "input_embeds" not in k5
    assert tf5(**k5) == "tf5"  # no unexpected-keyword TypeError

    k4 = _causal_mask_kwargs(
        tf4, config="c", hidden=sentinel, cache="kv", position_ids="p", cache_position="cp"
    )
    assert "input_embeds" in k4 and k4["input_embeds"] is sentinel
    assert "inputs_embeds" not in k4
    assert k4["cache_position"] == "cp"
    assert tf4(**k4) == "tf4"  # the 4.x breakage the real-weight run hit

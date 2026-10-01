"""verify='fast' (default) and verify='strict' must agree bit-for-bit.

Fast skips the per-transition defensive re-verification (an O(state) content hash of every
unwritten slot each step) and trusts a registered adapter's ``advance``; it must still produce
identical tokens, logits, receipt cut fingerprints, and fresh-process replay.
"""
from __future__ import annotations

import pytest
import torch

from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.core import Session
from saturn_pub.store import LocalStore


@pytest.fixture(autouse=True)
def _no_grad():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        yield
    finally:
        torch.set_grad_enabled(prev)


def _decode(adapter, ids, verify, steps):
    s = adapter.session(ids, verify=verify)
    for _ in range(steps):
        s.continue_(1)
    return s


@pytest.mark.parametrize("family", ["gpt_neox", "qwen2", "llama"])
def test_fast_equals_strict(family):
    adapter = DecoderAdapter.tiny(family, seed=5)
    ids = [1, 2, 3, 4, 5]
    steps = 3 * (adapter.layers + 2)
    fast = _decode(adapter, ids, "fast", steps)
    strict = _decode(adapter, ids, "strict", steps)
    assert torch.equal(fast.read("tokens"), strict.read("tokens"))
    fl, sl = fast.read("logits"), strict.read("logits")
    if fl is not None and sl is not None:
        assert torch.equal(fl, sl)
    # the committed receipts' result fingerprints are the content-addressed cut ids: identical
    assert [r.to_dict()["result"] for r in fast.receipts] == [
        r.to_dict()["result"] for r in strict.receipts
    ]


def test_fast_is_default():
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    assert adapter.session([1, 2, 3])._verify == "fast"


def test_fork_inherits_verify():
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    s = adapter.session([1, 2, 3], verify="strict")
    cut = s.capture()
    assert s.fork(cut)._verify == "strict"


def test_fast_mode_replays_bit_exact(tmp_path):
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    s = _decode(adapter, [1, 2, 3, 4], "fast", 2 * (adapter.layers + 2))
    cut = s.capture()
    expected = s.read("tokens")[0].tolist()
    store = LocalStore(tmp_path)
    digest = store.save(cut)
    reopened = Session.from_cut(adapter, store.load(digest))
    assert reopened.read("tokens")[0].tolist() == expected


def test_bad_verify_rejected():
    adapter = DecoderAdapter.tiny("gpt_neox", seed=5)
    with pytest.raises(ValueError, match="verify must be"):
        adapter.session([1, 2, 3], verify="loose")

import numpy as np
import pytest
import torch

from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.experimental.memory import SharedQwenMemory
from saturn_pub.experimental.phase import PhaseLaw, relocated_scores, rotate_pairs


def test_phase_group_law_and_float_decode():
    law = PhaseLaw.from_inv_freq([1.0, 0.01])
    initial = law.theta([0, 1, 100])
    for _ in range(64):
        law.shift(512)
    fresh = PhaseLaw(law.frequencies, 64 * 512)
    np.testing.assert_array_equal(law.theta([0, 1, 100]), fresh.theta([0, 1, 100]))
    near = (int(initial[1, 0]) - int(initial[0, 0])) % (1 << 32)
    huge = law.theta([1 << 60, (1 << 60) + 1])
    assert near == (int(huge[1, 0]) - int(huge[0, 0])) % (1 << 32)
    cos, sin = law.cos_sin([0, 1, 100])
    np.testing.assert_allclose(cos**2 + sin**2, 1, atol=1e-15)


def test_immutable_key_relocation_matches_absolute_relative_phase():
    law = PhaseLaw.from_inv_freq([1.0, 0.01])
    rng = np.random.default_rng(7)
    raw_keys, query = rng.normal(size=(3, 2, 2)), rng.normal(size=(2, 2))
    local_keys = rotate_pairs(raw_keys, law, [0, 1, 2])
    original = local_keys.copy()
    for origin in [0, 512, 1 << 40, -(1 << 40)]:
        position = origin + 5
        actual = relocated_scores(
            query, local_keys, law, query_position=position, page_origin=origin
        )
        absolute_keys = rotate_pairs(raw_keys, law, [origin + i for i in range(3)])
        absolute_query = rotate_pairs(query[None], law, [position])[0]
        expected = np.sum(absolute_keys * absolute_query, axis=(-1, -2))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(local_keys, original)


def test_shared_ancestor_native_decode_private_fork_and_accounting():
    adapter = QwenAdapter.tiny()
    memory = SharedQwenMemory(adapter, [5, 7, 11], capacity=6)
    left, right = memory.root.fork(), memory.root.fork()
    assert memory.accounting([left, right])["private_allocated_bytes"] == 0
    assert left.ancestor_pointers == right.ancestor_pointers
    for cache, token in ((left, 13), (right, 17)):
        logits = memory.step(cache, token)
        native = adapter.native_logits([5, 7, 11, token])
        torch.testing.assert_close(logits, native, rtol=1e-5, atol=1e-7)
    retained = left.fork()
    logits = memory.step(left, 19)
    native = adapter.native_logits([5, 7, 11, 13, 19])
    torch.testing.assert_close(logits, native, rtol=1e-5, atol=1e-7)
    assert retained.get_seq_length() == right.get_seq_length() == 4
    report = memory.accounting([left, right])
    assert report["saved_committed_bytes"] == report["shared_ancestor_bytes"]
    assert report["attention_rows"] == [5, 4]
    assert not report["reduced_attention_work_claim"]
    assert adapter.model.config._attn_implementation == "eager"
    with pytest.raises(ValueError, match="unique"):
        memory.accounting([left, left])


def test_shared_ancestor_refuses_mutation_and_rolls_back_failed_forward(monkeypatch):
    adapter = QwenAdapter.tiny()
    memory = SharedQwenMemory(adapter, [5, 7], capacity=2)
    branch = memory.root.fork()
    original = adapter.model.forward

    def broken(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected failure after writes")

    monkeypatch.setattr(adapter.model, "forward", broken)
    with pytest.raises(RuntimeError):
        memory.step(branch, 11)
    assert branch.get_seq_length() == 2
    assert adapter.model.config._attn_implementation == "eager"
    monkeypatch.setattr(adapter.model, "forward", original)
    assert torch.isfinite(memory.step(branch, 11)).all()
    with torch.inference_mode():
        branch._keys[0].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        memory.step(branch, 13)

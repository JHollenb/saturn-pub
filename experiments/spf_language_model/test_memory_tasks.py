"""Mechanics only: raw organism construction, query visibility, and replay."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import build_baseline, parameter_count
from memory_tasks import (
    DEFAULT_DELAYS,
    PAYLOAD_BYTES,
    MemoryTask,
    as_memory_batch,
    query_observation,
    score_frozen_grid,
)


@pytest.mark.parametrize("kind", ["delayed_recall", "latest_revision"])
def test_byte_organism_shape_balance_and_query_only_targets(kind):
    task = MemoryTask(kind)
    for delay in (*DEFAULT_DELAYS, 128):
        x, y, mask, meta = task.sample_batch(12, (delay,), split="feedback", seed=42)
        assert x.dtype == y.dtype == torch.long and mask.dtype == torch.bool
        assert x.shape == y.shape == mask.shape
        assert x.shape[1] == delay + (3 if kind == "delayed_recall" else 9)
        assert meta["padding_bytes"] == 0 and meta["input_bytes"] == x.numel()
        assert (x[:, -1] == ord("?")).all() and (mask.sum(-1) == 1).all()
        assert mask[:, -1].all() and not mask[:, :-1].any()
        assert torch.equal(y[:, :-1], x[:, 1:])
        assert sorted(y[mask].tolist()) == sorted(list(PAYLOAD_BYTES) * 3)
        if kind == "latest_revision":
            pairs = {(row["old_payload_byte"], row["payload_byte"]) for row in meta["rows"]}
            assert len(pairs) == 12
        for index, row in enumerate(meta["rows"]):
            assert x[index, row["payload_position"]] == y[index, -1] == row["payload_byte"]
            assert row["source_prefix_length"] == row["payload_position"] + 1
            assert row["query_position"] == x.shape[1] - 1
            assert row["alternate_payload_byte"] != row["payload_byte"]
            assert row["alternate_payload_byte"] != row["old_payload_byte"]
            assert torch.all(
                (x[index, -delay - 1 : -1] >= ord("a")) & (x[index, -delay - 1 : -1] <= ord("z"))
            )
            if kind == "latest_revision":
                assert x[index, 1] == row["old_payload_byte"] != row["payload_byte"]
                assert x[index, row["payload_position"] - 1] == ord("!")
        adapted = as_memory_batch(x, y, mask, meta)
        assert torch.equal(adapted["targets"], y[mask])
        prefix = meta["source_prefix_length"]
        assert torch.equal(adapted["alternate_ids"][:, prefix:], x[:, prefix:])
        assert (adapted["alternate_ids"] != x).sum().item() == x.shape[0]
        json.dumps(meta, allow_nan=False)


def test_train_cursor_independent_rewind_and_complete_delay_cycles():
    task = MemoryTask()
    a, b = task.cursor(seed=817), task.cursor(seed=817)
    global_rng = torch.random.get_rng_state().clone()
    state = a.snapshot()
    first = task.sample_batch(cursor=a)
    twin = task.sample_batch(cursor=b)
    assert all(torch.equal(x, y) for x, y in zip(first[:3], twin[:3]))
    assert first[3] == twin[3]
    task.sample_batch(cursor=a)
    a.restore(json.loads(json.dumps(state)))
    replay = task.sample_batch(cursor=a)
    assert all(torch.equal(x, y) for x, y in zip(first[:3], replay[:3]))
    assert first[3] == replay[3]
    assert torch.equal(global_rng, torch.random.get_rng_state())
    a.restore(state)
    delays = [task.sample_batch(cursor=a)[3]["delay"] for _ in DEFAULT_DELAYS]
    assert sorted(delays) == sorted(DEFAULT_DELAYS)
    with pytest.raises(ValueError, match="same task and split"):
        task.cursor("feedback").restore(state)
    with pytest.raises(ValueError, match="different task"):
        MemoryTask("latest_revision").sample_batch(cursor=a)


def test_audit_access_independent_split_identity_and_bounds():
    task = MemoryTask()
    for split in ("test", "audit", "cold_audit", "sealed_audit"):
        with pytest.raises(PermissionError):
            task.cursor(split)
        with pytest.raises(PermissionError):
            task.sample_batch(split=split)
    audit = task.cursor("sealed_audit", seed=811, allow_sealed=True)
    with pytest.raises(PermissionError):
        task.sample_batch(cursor=audit)
    x, _, _, a_meta = task.sample_batch(cursor=audit, delays=(64,), allow_sealed=True)
    other, _, _, b_meta = task.sample_batch(split="feedback", seed=811, delays=(64,))
    assert a_meta["split"] != b_meta["split"] and not torch.equal(x, other)
    assert a_meta["sealed_for_selection"]
    for bad in ((), (0, 0), (-1,), (129,), (1.5,)):
        with pytest.raises(ValueError):
            task.sample_batch(delays=bad)
    with pytest.raises(ValueError, match="multiple of four"):
        task.sample_batch(3, split="feedback")
    with pytest.raises(ValueError, match="cursor or seed"):
        task.sample_batch(cursor=task.cursor(), seed=1)


def test_query_scores_scope_and_frozen_grid_mode_restore():
    task = MemoryTask()
    x, y, mask, _ = task.sample_batch(4, (8,), split="feedback")
    logits = torch.full((*x.shape, 256), -100.0)
    logits[:, :, PAYLOAD_BYTES] = 0.0
    observation = query_observation(logits, y, mask)
    assert observation["query_bits_per_byte"] == pytest.approx(2.0)
    assert observation["conditional_payload_bits"] == pytest.approx(2.0)
    assert observation["query_accuracy_raw256"] == pytest.approx(0.25)
    # Unscored target positions cannot change the query loss.
    y[:, :-1] = 255
    assert query_observation(logits, y, mask) == observation

    class FrozenPrior(torch.nn.Module):
        def forward(self, byte_ids):
            result = torch.full((*byte_ids.shape, 256), -100.0, device=byte_ids.device)
            result[:, :, PAYLOAD_BYTES] = 0
            return result

    prior = FrozenPrior().train()
    scored = score_frozen_grid(prior, task, delays=(0, 8), num_batches=1)
    assert prior.training and len(scored["rows"]) == 2
    assert all(row["query_accuracy_raw256"] == 0.25 for row in scored["by_delay"])
    with pytest.raises(PermissionError):
        score_frozen_grid(prior, task, split="sealed_audit")


def test_transformer_fixed_scale_default_parity_and_input_observation():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        original = build_baseline("transformer", d_model=8, n_layers=1, seed=32)
        explicit = build_baseline(
            "transformer", d_model=8, n_layers=1, seed=32, positional_scale=1.0
        )
        scaled = build_baseline(
            "transformer", d_model=8, n_layers=1, seed=32, positional_scale=0.02
        )
        ids = torch.tensor([[65, 97, 98, 99, 63], [66, 100, 101, 102, 63]])
        torch.testing.assert_close(original(ids), explicit(ids), rtol=0, atol=0)
        # Independently execute the original unscaled sinusoid algebra.
        hidden = original.embedding(ids)
        positions = torch.arange(ids.shape[1], dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, 8, 2, dtype=torch.float32) * (-math.log(10000.0) / 8)
        )
        angles = positions[:, None] * frequencies[None]
        positional = torch.zeros(ids.shape[1], 8)
        positional[:, 0::2], positional[:, 1::2] = angles.sin(), angles.cos()
        hidden = hidden + positional[None]
        for block in original.blocks:
            hidden = block(hidden)
        old_logits = original.lm_head(original.final_norm(hidden))
        torch.testing.assert_close(original(ids), old_logits, rtol=0, atol=0)
        assert parameter_count(original) == parameter_count(scaled)
        assert not original.positional_scale.requires_grad
        assert "positional_scale" not in original.state_dict()
        assert scaled.architecture["positional_scale"] == 0.02
        before = original.input_scale_observation(ids)
        after = scaled.input_scale_observation(ids)
        assert before["embedding_rms"] == after["embedding_rms"]
        assert after["position_rms"] == pytest.approx(before["position_rms"] * 0.02)
        assert not before["targets_used"]
        explicit.load_state_dict(original.state_dict(), strict=True)
        with pytest.raises(ValueError, match="positional_scale"):
            build_baseline("transformer", positional_scale=float("nan"))
    finally:
        torch.set_num_threads(old_threads)

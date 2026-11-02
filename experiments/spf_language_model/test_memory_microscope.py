"""Tiny synthetic mechanics fixtures; no trained-model quality assertions."""

import pytest
import torch
from memory_microscope import _batch, frozen_memory_panel
from models import build_spf_model


def memory_batch(delay=2):
    payload = torch.tensor([65, 66, 67, 68])
    ids = torch.cat(
        (
            torch.full((4, 1), 94),
            payload[:, None],
            torch.full((4, delay), 97),
            torch.full((4, 1), 63),
        ),
        1,
    )
    alternate = ids.clone()
    alternate[:, 1] = 65 + (payload - 65 + 1) % 4
    return {
        "ids": ids,
        "targets": payload,
        "alternate_ids": alternate,
        "metadata": {
            "kind": "synthetic_mechanics_fixture",
            "delay": delay,
            "source_prefix_length": 2,
        },
    }


class FakeBaseline(torch.nn.Module):
    """A tiny deterministic float consumer for output/artifact contract tests."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.arange(256, dtype=torch.float32))

    def forward(self, ids):
        accumulated = ids.float().cumsum(-1)
        return -(accumulated[..., None] - self.table).abs()


def test_non_spf_fake_consumer_has_no_native_word_claim(tmp_path):
    model = FakeBaseline()
    result = frozen_memory_panel(
        model, [memory_batch()], tmp_path / "panel", {"model_id": "fake-mechanics"}
    )
    assert result["model_parameters_unchanged"]
    row = result["rows"][0]
    assert row["reference"]["output_vocabulary"] == 256
    assert row["packed_native"]["instrument_status"] == "not-applicable"
    assert row["no_history"]["history_operation"] == "fresh_sequence_query_marker_only"
    assert model.training


def test_alternate_source_requires_identical_gap_and_actual_payload_identity():
    batch = memory_batch()
    batch["alternate_ids"][:, 2] = 98
    with pytest.raises(ValueError, match="suffix"):
        _batch(batch)
    batch = memory_batch()
    batch["targets"] = torch.tensor([1, 2, 3, 4])
    with pytest.raises(ValueError, match="source payload"):
        _batch(batch)


@pytest.mark.parametrize("kind,expected_prefix", [("recall", 2), ("revision", 8)])
def test_actual_task_batch_interface_preserves_the_replayed_suffix(kind, expected_prefix):
    from memory_tasks import MemoryTask, as_memory_batch

    task = MemoryTask(kind)
    batch = as_memory_batch(*task.sample_batch(4, (2,), split="feedback", seed=811))
    ids, targets, alternate, alternate_targets, prefix, _ = _batch(batch)
    assert prefix == expected_prefix
    assert torch.equal(ids[:, prefix:], alternate[:, prefix:])
    assert torch.all(targets != alternate_targets)


def test_tiny_untrained_word_fixture_mdb_repair_and_snapshot_identity(tmp_path):
    pytest.importorskip("spf.tensor.sequence_mixer")
    pytest.importorskip("mdb.research_session")
    # Tiny randomly initialized module used only to qualify the wrapper mechanics.
    model = build_spf_model(d_model=4, n_layers=1, state_size=2, seed=17)
    original = model.model_sha256()
    result = frozen_memory_panel(
        model,
        [memory_batch()],
        tmp_path / "panel",
        {"model_id": "tiny-untrained-mechanics"},
        native_examples=4,
        clocked_max_delay=0,
        reference_sensitivity=True,
    )
    assert result["model_parameters_unchanged"] and model.model_sha256() == original
    row = result["rows"][0]
    native = row["packed_native"]
    assert native["required_native"]["core_fallback_conversions"] == 0
    assert native["full_chunk_contrast"]["status"] == "observation"
    assert native["mdb"]["mechanics"]["no_op_exact"]
    assert native["mdb"]["mechanics"]["repair_exact"]
    assert native["mdb"]["mechanics"]["restore_exact"]
    transplant = next(
        value for value in native["mdb"]["rows"] if value["arm"] == "alternate-payload-all-layers"
    )
    assert transplant["observation"]["alternate_raw_bytes"] == [66, 67, 68, 65]
    assert transplant["observation"]["core_fallback_conversions"] == 0
    assert (tmp_path / "panel/batch-000/prefix-donor-chunk-endpoints.pt").exists()
    assert len(row["reference_carrier_sensitivity"]["layers"]) == 1

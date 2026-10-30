from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from mdb.discovery_batch import (
    BatchError,
    BatchProfile,
    BatchRow,
    FrozenNativeBatch,
    OutputAct,
    compare_screen_rows,
    verify_batch_attempt,
)


class Native(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.law = torch.nn.Linear(3, 3)
        self.calls = []

    def forward(self, x, extra):
        self.calls.append(tuple(x.shape))
        value = self.law(x) + extra
        return {f"head_{i:02d}": value + i for i in range(11)}


@pytest.fixture
def native():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = Native().eval().requires_grad_(False)
    profile = BatchProfile("tiny-weights", "tiny-native-program", ((1, 3), (1, 3)),
                           {f"head_{i:02d}": (1, 3) for i in range(11)},
                           allowed_modules=("law",))
    yield model, profile
    torch.set_num_threads(old)


def rows(count=4):
    return [BatchRow(f"lane_{i}", "input_00", "main", i,
                     (torch.tensor([[i, i + 1, i + 2]], dtype=torch.float32),
                      torch.full((1, 3), i / 10))) for i in range(count)]


def test_true_native_batch_preserves_all_heads_and_stable_row_ownership(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    inputs = rows()
    scalar = adapter.screen(inputs, batch_size=1)
    batch = adapter.screen(list(reversed(inputs)))
    assert not scalar.errors and not batch.errors
    assert model.calls == [(1, 3)] * 4 + [(4, 3)]
    assert len(batch.attempts) == 1 and batch.attempts[0]["physical_batch_size"] == 4
    assert verify_batch_attempt(batch.attempts[0]) == batch.attempts[0]["fingerprint"]
    for candidate in batch.rows:
        reference = next(r for r in scalar.rows if r.binding == candidate.binding)
        compared = compare_screen_rows(reference, candidate, profile)
        assert len(compared) == 11
        assert all(v["within_contract"] for v in compared.values())
        assert candidate.matches(lane_id=candidate.binding["lane_id"], epoch_id="input_00",
                                 branch_id="main", revision=candidate.binding["expected_revision"])
        assert not candidate.matches(lane_id=candidate.binding["lane_id"], epoch_id="input_01",
                                     branch_id="main", revision=candidate.binding["expected_revision"])
    assert adapter.work == {"physical_model_calls": 5, "physical_batches": 1,
                            "logical_forward_rows": 8, "failed_physical_calls": 0,
                            "padded_rows": 0}
    # Each logical output owns separate storage, including after slot compaction.
    assert batch.rows[0].outputs["head_00"].untyped_storage().data_ptr() != (
        batch.rows[1].outputs["head_00"].untyped_storage().data_ptr())
    batch.attempts[0]["rows"][0]["epoch_id"] = "changed"
    with pytest.raises(BatchError, match="attempt record changed"):
        verify_batch_attempt(batch.attempts[0])


def test_park_refill_new_epoch_does_not_reuse_slot_or_advance_paused_input(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    inputs = rows()
    parked = replace(inputs[1], status="paused")
    first = adapter.screen([inputs[0], parked, inputs[2]])
    assert model.calls == [(2, 3)]
    assert first.skipped[0]["lane_id"] == "lane_1"
    refill = replace(inputs[0], epoch_id="input_01", expected_revision=7,
                     roots=(torch.full((1, 3), 99.0), torch.zeros(1, 3)))
    second = adapter.screen([parked, inputs[3], refill])
    assert model.calls == [(2, 3), (2, 3)]
    assert [r.binding["lane_id"] for r in second.rows] == ["lane_3", "lane_0"]
    assert second.rows[1].binding["epoch_id"] == "input_01"
    assert second.rows[1].roots_digest != first.rows[0].roots_digest
    assert torch.equal(parked.roots[0], inputs[1].roots[0])


def test_component_debug_stops_require_scalar_replay_without_native_work(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    inputs = rows(2)
    result = adapter.screen([replace(inputs[0], component_stops=True),
                             replace(inputs[1], frontier="module-law-exit")])
    assert len(result.scalar_required) == 2 and not result.rows
    assert model.calls == [] and adapter.work["physical_model_calls"] == 0


def test_per_row_candidate_acts_preserve_frozen_weights_and_match_native_scalar(native):
    model, profile = native
    before = {name: value.clone() for name, value in model.state_dict().items()}
    adapter = FrozenNativeBatch(model, profile)
    inputs = rows()
    edits = (OutputAct("law", "zeros_like"), OutputAct("law", "scale", 0.5),
             OutputAct("law", "add", torch.ones(1, 3)),
             OutputAct("law", "replace", torch.full((1, 3), 5.0)))
    candidates = [replace(row, acts=(act,)) for row, act in zip(inputs, edits)]
    scalar = adapter.screen(candidates, batch_size=1)
    batch = adapter.screen(candidates)
    assert not scalar.errors and not batch.errors
    assert batch.attempts[0]["physical_batch_size"] == 4
    for reference, candidate in zip(scalar.rows, batch.rows):
        assert all(v["within_contract"] for v in compare_screen_rows(reference, candidate, profile).values())
    # An independent fresh native hook audit uses the original single input.
    handle = model.law.register_forward_hook(lambda module, args, output: torch.zeros_like(output))
    try:
        with torch.no_grad():
            fresh = model(*inputs[0].roots)
    finally:
        handle.remove()
    assert all(torch.equal(batch.rows[0].outputs[key], value) for key, value in fresh.items())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert not model.law._forward_hooks


def test_incompatible_hook_schedules_form_explicit_scalar_groups(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    first, second = rows(2)
    result = adapter.screen([first, replace(second, acts=(OutputAct("law", "zeros_like"),))])
    assert not result.errors
    assert model.calls == [(1, 3), (1, 3)]
    assert all(attempt["mode"] == "native-scalar" for attempt in result.attempts)


def test_failed_native_group_is_charged_and_other_groups_survive(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    first, second = rows(2)
    result = adapter.screen([replace(first, acts=(OutputAct("law", "zeros_like", occurrence=2),)), second])
    assert len(result.errors) == 1 and len(result.rows) == 1
    assert result.rows[0].binding["lane_id"] == "lane_1"
    assert adapter.work["physical_model_calls"] == 2
    assert adapter.work["failed_physical_calls"] == 1
    assert not model.law._forward_hooks


def test_both_physical_and_logical_budgets_charge_before_calls(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile, max_physical_calls=1, max_logical_rows=4)
    assert len(adapter.screen(rows()).rows) == 4
    exhausted = adapter.screen(rows())
    assert exhausted.errors[0]["kind"] == "budget" and not exhausted.attempts
    assert model.calls == [(4, 3)]
    other = FrozenNativeBatch(model, profile, max_logical_rows=3)
    assert other.screen(rows()).errors[0]["kind"] == "budget"
    assert other.work["physical_model_calls"] == 0


def test_unfrozen_models_invalid_inputs_and_mutated_weights_are_rejected(native):
    model, profile = native
    model.requires_grad_(True)
    with pytest.raises(BatchError, match="already be frozen"):
        FrozenNativeBatch(model, profile)
    model.requires_grad_(False)
    adapter = FrozenNativeBatch(model, profile)
    invalid = replace(rows(1)[0], roots=(torch.zeros(2, 3), torch.zeros(1, 3)))
    with pytest.raises(BatchError, match="root violates"):
        adapter.screen([invalid])
    assert model.calls == []
    with torch.no_grad():
        model.law.weight.add_(1)
    with pytest.raises(BatchError, match="identity/version"):
        adapter.screen(rows())


def test_missing_or_nonfinite_auxiliary_heads_fail_entire_group(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    handle = model.register_forward_hook(lambda module, args, result:
        {**result, "head_10": torch.full_like(result["head_10"], float("nan"))})
    try:
        result = adapter.screen(rows())
    finally:
        handle.remove()
    assert not result.rows and len(result.errors) == 1
    assert "head_10" in result.errors[0]["error"]


def test_wrong_act_geometry_and_unknown_module_never_silently_broadcast(native):
    model, profile = native
    adapter = FrozenNativeBatch(model, profile)
    wrong = replace(rows(1)[0], acts=(OutputAct("law", "replace", torch.ones(1, 1)),))
    result = adapter.screen([wrong])
    assert not result.rows and "actual module-output geometry" in result.errors[0]["error"]
    with pytest.raises(BatchError, match="admitted module"):
        adapter.screen([replace(rows(1)[0], acts=(OutputAct("missing", "zeros_like"),))])
    assert not model.law._forward_hooks


def test_finite_qualification_helper_measures_real_calls_and_refill(native):
    model, profile = native
    helper = runpy.run_path(str(Path(__file__).parents[1] / "qualification/discovery_batch.py"))
    class Driver:
        def __init__(self, roots):
            self.roots = roots

        def inputs(self):
            return tuple(value.clone() for value in self.roots)

    report = helper["qualify_native_batches"](model, model_identity="tiny-weights",
        program_identity="tiny-native-program", drivers=[Driver(row.roots) for row in rows()],
        head_shapes=profile.head_shapes, module_site="law", max_physical_calls=20)
    assert report["physical_batching"] and report["model_weights_unchanged"]
    assert report["instrument_status"] == "within-declared-contract"
    assert report["total_physical_model_calls"] == len(model.calls) == 20
    assert report["adapter_work"]["physical_model_calls"] == 19
    assert report["parked_skipped"][0]["lane_id"] == "batch_lane_00"
    assert report["refill_all_head_controls"][0]["binding"]["epoch_id"] == "input_01"
    assert all(row["physical_batch_size"] <= 4 for row in report["attempts"])
    assert report["memory"]["after"] >= report["memory"]["before"]

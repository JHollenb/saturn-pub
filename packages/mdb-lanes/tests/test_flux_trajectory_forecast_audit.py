"""Saved-array audit regressions; no Torch, model, equations or native replay."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

AUDIT = runpy.run_path(str(Path(__file__).parents[1] / "qualification/analyze_flux_trajectory_forecast.py"))


def witness_pair(*, full=True):
    _, typed = AUDIT["helpers"]()
    predicted = np.array([[1., 2., 3., 4.]], dtype=np.float32)
    actual = np.array([[1., 2.25, 3., 4.]], dtype=np.float32)
    positions = np.array([0, 1, 3], dtype=np.int64)
    left = {"descriptor": typed["descriptor"](predicted, bf16=True), "float_array_sha256": hashlib.sha256(predicted.tobytes()).hexdigest(),
            "witness_indices": positions, "witness_values": predicted.ravel()[positions], "full_retained": full}
    right = {"descriptor": typed["descriptor"](actual, bf16=True), "witness_indices": positions.copy(),
             "witness_values": actual.ravel()[positions], "full_error": AUDIT["compare"](actual, predicted)}
    if full:
        left["full_values"], right["full_values"] = predicted, actual
    return left, right


def test_genuine_nonzero_forecast_error_is_retained_not_rejected():
    left, right = witness_pair()
    result = AUDIT["inspect_saved_array"](left, right, 3)
    assert result["full_recomputed"]["max_abs"] == .25
    assert result["full_recomputed"]["exact"] is False
    assert result["witness"]["changed_elements"] == 1


def test_intermediate_full_error_is_explicitly_not_offline_recomputed():
    left, right = witness_pair(full=False)
    result = AUDIT["inspect_saved_array"](left, right, 3)
    assert "full_recomputed" not in result
    assert "worker RAM observations" in result["offline_scope"]
    assert result["worker_full_error"]["max_abs"] == .25


@pytest.mark.parametrize("corruption", ["indices", "value_bytes", "dtype", "full_digest", "false_exact"])
def test_custody_or_instrument_corruption_is_separate_from_numerical_error(corruption):
    left, right = witness_pair()
    if corruption == "indices":
        right["witness_indices"][1] = 2
    elif corruption == "value_bytes":
        right["full_values"][0, 1] = 2.5
    elif corruption == "dtype":
        right["witness_values"] = right["witness_values"].astype(np.float64)
    elif corruption == "full_digest":
        left["float_array_sha256"] = "0" * 64
    else:
        right["full_error"] = AUDIT["compare"](left["full_values"], left["full_values"])
    with pytest.raises(ValueError):
        AUDIT["inspect_saved_array"](left, right, 3)


def test_initial_state_join_rejects_stale_values_even_with_same_geometry():
    _, typed = AUDIT["helpers"]()
    values = np.array([1., 2.], dtype=np.float32)
    described = {"kind": "tensor", **typed["descriptor"](values, bf16=True)}
    packed = {"tensor_dtype": "torch.bfloat16", "array": values.copy()}
    AUDIT["join_packed_state"](packed, described)
    packed["array"][1] = 3.
    with pytest.raises(ValueError, match="bytes differ"):
        AUDIT["join_packed_state"](packed, described)


def test_schedule_starts_after_supplied_projection_and_keeps_later_projections():
    events = AUDIT["expected_events"](4, 5, 20)
    assert len(events) == 111
    assert [event["step"] for event in events if event["phase"] == "project"] == [1, 2, 3]
    assert (events[0]["phase"], events[0]["index"]) == ("joint", 0)
    assert (events[-1]["next_step"], events[-1]["next_phase"], events[-1]["next_index"]) == (4, "project", 0)
    point = {"logical_step": 0, "phase": "joint", "local_clock": 0, "index": 0, "edge": "before"}
    assert AUDIT["_point_matches"](point, events[0])
    stale = copy.deepcopy(point)
    stale["logical_step"] = 1
    assert not AUDIT["_point_matches"](stale, events[0])


def test_saved_decoded_tensor_pixel_rounding_keeps_bf16_cast_order():
    values = np.array([[[[-1., 0., 1.]], [[-1., 0., 1.]], [[-1., 0., 1.]]]], dtype=np.float32)
    for dtype in ("torch.float32", "torch.bfloat16"):
        pixels = AUDIT["pixels_from_decoded"](values, dtype)
        np.testing.assert_array_equal(pixels[0, :, 0], [0, 128, 255])


def test_recovery_preserves_failed_publication_and_uses_durable_observation(tmp_path):
    failure = {"job_id": "failed-job", "request_sha256": "a" * 64,
               "error": "TypeError: Object of type mappingproxy is not JSON serializable"}
    (tmp_path / "failure.json").write_text(json.dumps(failure))
    partial = tmp_path / "summary.json"
    partial.write_bytes(b'{"pixel_error": ')
    run = {"data": tmp_path, "source": {"job_id": "failed-job"}, "request": {"request_sha256": "a" * 64},
           "plan": {"limits": "RAM-only", "scope": "observed frontier"}}
    actual = {"counts": {"native_transitions": 111}, "pixel_error": {"exact": False, "max_abs": 3.}}
    predicted = {"own_state": True, "native_refreshes": 0, "RGB_forecast": True}
    recovered = AUDIT["recovered_summary"](run, {"model_identity": "fixture", "arithmetic": "shared primitives"}, predicted, actual)
    assert recovered["scheduler_status"] == "failed"
    assert recovered["publication_status"] == "failed-summary-serialization"
    assert recovered["counts"] == actual["counts"]
    assert recovered["pixel_error"]["exact"] is False
    assert recovered["worker_seconds"] is None
    assert partial.read_bytes() == b'{"pixel_error": '
    failure["error"] = "RuntimeError: incomplete native execution"
    (tmp_path / "failure.json").write_text(json.dumps(failure))
    with pytest.raises(ValueError, match="not the declared"):
        AUDIT["recovered_summary"](run, {}, predicted, actual)

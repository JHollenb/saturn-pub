"""Saved-array auditor regressions; no weights, models or native execution."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

audit = runpy.run_path(str(Path(__file__).parents[1] / "qualification/analyze_qwen_mlp_forecast.py"))


def branch():
    return {"full_vocab_greedy_token": 7, "generated_token_ids": [7, 8, 9],
            "prefill_kv": {"sha256": "a" * 64}, "continuation": [
                {"step": 1, "consumed_token": 7, "greedy_token": 8},
                {"step": 2, "consumed_token": 8, "greedy_token": 9}]}


def test_stale_consumed_token_breaks_native_own_state_even_with_same_generated_ids():
    baseline = branch()
    stale = copy.deepcopy(baseline)
    stale["continuation"][1]["consumed_token"] = 7
    result = audit["continuation_check"](stale, baseline, 2)
    assert "native continuation does not consume its own preceding token" in result["errors"]


def test_same_kv_and_first_token_future_disagreement_is_a_diagnostic():
    baseline, changed = branch(), branch()
    changed["continuation"][1]["greedy_token"] = 10
    changed["generated_token_ids"][-1] = 10
    result = audit["continuation_check"](changed, baseline, 2)
    assert result["closure_applicable"] and not result["same_future_tokens"]
    assert result["diagnostics"]
    assert not result["errors"]


@pytest.mark.parametrize("change", ["first_token", "kv"])
def test_different_native_frontier_does_not_claim_equal_future_closure(change):
    baseline, changed = branch(), branch()
    if change == "kv":
        changed["prefill_kv"]["sha256"] = "b" * 64
    else:
        changed["full_vocab_greedy_token"] = 6
        changed["generated_token_ids"][0] = 6
        changed["continuation"][0]["consumed_token"] = 6
    result = audit["continuation_check"](changed, baseline, 2)
    assert not result["closure_applicable"]
    assert not result["diagnostics"] and not result["errors"]


def test_generated_token_metadata_and_early_horizon_are_checked_separately():
    baseline, changed = branch(), branch()
    changed["generated_token_ids"] = [7, 8, 11]
    result = audit["continuation_check"](changed, baseline, 3)
    assert "retained branch token IDs differ from native greedy chain" in result["errors"]
    assert "native continuation horizon differs" in result["errors"]


def test_tensor_identity_checks_hash_geometry_and_dtype():
    value = np.array([[[1., -2.]]], dtype=np.float32)
    binding = audit["descriptor"](value)
    assert audit["check_descriptor"](value.copy(), binding)
    tampered = value.copy()
    tampered[0, 0, 0] = 2.
    assert not audit["check_descriptor"](tampered, binding)
    assert not audit["check_descriptor"](value.astype(np.float64), binding)
    assert not audit["check_descriptor"](value.reshape(2), binding)
    changed_hash = {**binding, "sha256": "0" * 64}
    assert not audit["check_descriptor"](value, changed_hash)


def test_byte_identity_does_not_collapse_signed_zero_or_dtype():
    value = np.array([0.], dtype=np.float32)
    assert audit["exact"](value, value.copy())
    assert not audit["exact"](value, -value)
    assert not audit["exact"](value, value.astype(np.float64))


def test_selected_candidate_forecast_can_match_while_global_winner_is_uncovered():
    ids = [5, 12, 17]
    actual = np.array([[[3., 9., 2.]]], dtype=np.float32)
    forecast = np.array([[[3.00001, 8.99999, 2.]]], dtype=np.float64)
    comparison = audit["candidate_comparison"](actual, forecast, ids)
    observed_full_vocab_greedy_token = 1304
    assert comparison["candidate_argmax_matches"]
    assert comparison["forecast"]["token_id"] == 12
    assert observed_full_vocab_greedy_token not in ids
    assert "not a full-vocabulary token forecast" in comparison["scope"]


def test_candidate_tie_and_near_miss_remain_measurements():
    actual = np.array([[[1., 1., -2.]]], dtype=np.float32)
    predicted = np.array([[[.999999, 1.000001, -2.]]], dtype=np.float64)
    comparison = audit["candidate_comparison"](actual, predicted, [9, 3, 17])
    assert comparison["observed"]["token_id"] == 9
    assert comparison["observed"]["top_two_margin"] == 0.
    assert not comparison["candidate_argmax_matches"]
    assert comparison["residual"]["max_abs"] > 0.
    assert comparison["per_candidate_signed_residual"][0] > 0.
    assert comparison["per_candidate_signed_residual"][1] < 0.


def test_unbound_candidate_geometry_and_nonfinite_values_refuse_comparison():
    with pytest.raises(ValueError, match="geometry"):
        audit["candidate_comparison"](np.zeros((1, 2)), np.zeros((1, 2)), [3, 4])
    with pytest.raises(ValueError, match="nonfinite"):
        audit["candidate_comparison"](np.array([[[np.nan]]]), np.zeros((1, 1, 1)), [3])

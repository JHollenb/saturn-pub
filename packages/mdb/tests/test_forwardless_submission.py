"""Forwardless qualification plan and admission mechanics, no models."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

QUALIFICATION = Path(__file__).parents[1] / "qualification"


def test_plan_is_finite_and_within_the_bounded_local_work_budget():
    module = runpy.run_path(str(QUALIFICATION / "submit_compiled_state_forwardless.py"))
    for smoke in (True, False):
        plan = module["plan"](smoke=smoke)
        json.dumps(plan, allow_nan=False)
        assert plan["batch"] == 256 and plan["timing_batches"] == [1, 256] and plan["reps"] == 10
        assert plan["parity_rows"] == 64 and all(h in ("digit", "full") for *_, h in plan["addition"]["timing_full"])
        assert module["logical_operations"](plan) <= 64
        assert plan["addition"]["partial_layers"] == [20] and 28 in plan["addition"]["layers"]
    full = module["plan"](smoke=False)
    assert full["addition"]["n_fit"] == 160200 and full["addition"]["n_eval"] == 32768
    assert full["g1"]["expected_holdout"] == [0, 1, 9, 18, 27, 29, 31, 36, 39]
    comparison = full["comparison"]
    assert comparison["part_f"]["speedup"]["256"]["T3c(20)+digit_head"] == 8.59
    assert comparison["part_g"]["speedup"]["256"]["predict_chain_L28"] == 29.91
    assert comparison["part_e"]["speedup"]["256"]["F5-+totals(28)+digit_head"] == 139.69


def test_worker_replays_part_g_tables_verbatim():
    worker = (QUALIFICATION / "compiled_state_forwardless.py").read_text()
    module = runpy.run_path(str(QUALIFICATION / "compiled_state_forwardless.py"))
    assert len(module["COUNTRIES"]) == 48 and module["COUNTRIES"][32] == ("Argentina", "Buenos")
    assert module["G1_PIECES"] == ["The capital of", 0, " is", 1, ". The capital of", 2, " is"]
    assert "default_rng(seed + 7)" in worker

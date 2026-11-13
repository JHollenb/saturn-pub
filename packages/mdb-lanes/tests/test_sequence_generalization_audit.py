"""Offline saved-state regressions; no trained weights or model execution."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path

import numpy as np
import pytest

audit = runpy.run_path(str(Path(__file__).parents[1] / "qualification/analyze_sequence_generalization.py"))
PROGRAM = "0102030405060708090a"
SHAPES = {"regs": (1, 8, 64), "flags": (1, 4), "dest": (1, 8), "src": (1, 8),
          "width": (1, 3), "length": (1, 15), "syscall": (1,), "imm": (1, 64),
          "store": (1,), "store_addr": (1, 64), "store_value": (1, 64)}


def saved_heads(words, *, length=3):
    heads = {name: np.full(shape, -2., dtype=np.float64) for name, shape in SHAPES.items()}
    for register, word in enumerate(words):
        for bit in range(64):
            heads["regs"][0, register, bit] = 2. if word & (1 << bit) else -2.
    heads["length"][0, length - 1] = 2.
    return heads


def fixture():
    case = {"case_id": "toy", "register_words": [0] * 6 + [3, 262], "flags": [0] * 4}
    before = {"register_words": list(case["register_words"]), "flags": [0] * 4, "offset": 0}
    steps = {}
    for number, word in enumerate((261, 259)):
        words = list(before["register_words"])
        words[7] = word  # Supplied decisions, deliberately no ISA semantic assumption.
        heads = saved_heads(words)
        decoded = audit["helpers"]()["decoded"](heads)
        after = {"register_words": words, "flags": [0] * 4, "offset": before["offset"] + 3}
        steps[str(number)] = {"before": copy.deepcopy(before), "heads": heads, "decoded": decoded,
            "roots": audit["helpers"]()["raw_roots"](PROGRAM, before), "row_id": "toy.native." + str(number),
            "lineage": {"program_hex": PROGRAM, "case_id": "toy", "step": number,
                        "previous_row": None if number == 0 else "toy.native.0"},
            "driver_commit": {"committed": True, "before": copy.deepcopy(before), "after": copy.deepcopy(after)}}
        before = copy.deepcopy(after)
    return case, steps


def check(case, rows, *, native=True, budget=2):
    return audit["chain_checks"](rows, case, PROGRAM, budget, native=native)


def errors(result):
    return [error for row in result.values() for error in row["errors"]]


def test_forecast_and_native_close_on_different_own_states_without_reconditioning():
    case, native = fixture()
    forecast = copy.deepcopy(native)
    for row in forecast.values():
        row["heads"]["regs"][0, 6, 0] = -2.
        row["decoded"] = audit["helpers"]()["decoded"](row["heads"])
    forecast["1"]["before"]["register_words"][6] = 2
    forecast["1"]["roots"] = audit["helpers"]()["raw_roots"](PROGRAM, forecast["1"]["before"])
    assert not errors(check(case, native))
    assert not errors(check(case, forecast, native=False))
    assert not np.array_equal(native["1"]["roots"]["regs"], forecast["1"]["roots"]["regs"])


def test_internally_consistent_stale_roots_cannot_replace_previous_native_decisions():
    case, rows = fixture()
    rows["1"]["before"]["register_words"][7] = 262
    rows["1"]["roots"] = audit["helpers"]()["raw_roots"](PROGRAM, rows["1"]["before"])
    rows["1"]["driver_commit"]["before"] = copy.deepcopy(rows["1"]["before"])
    result = check(case, rows)
    assert not result["0"]["errors"]
    assert "own-state predecessor/initial state join differs" in result["1"]["errors"]
    assert all(result["1"]["literal_root_joins"].values())


def test_fetch_offset_cannot_reset_even_when_repacked_roots_match_reset_state():
    case, rows = fixture()
    rows["1"]["before"]["offset"] = 0
    rows["1"]["roots"] = audit["helpers"]()["raw_roots"](PROGRAM, rows["1"]["before"])
    assert "own-state predecessor/initial state join differs" in errors(check(case, rows))


def test_consumed_offset_follows_predicted_length_argmax_plus_one():
    case, rows = fixture()
    rows["0"]["heads"]["length"][0, 2] = -2.
    rows["0"]["heads"]["length"][0, 3] = 2.
    rows["0"]["decoded"] = audit["helpers"]()["decoded"](rows["0"]["heads"])
    rows["0"]["driver_commit"]["after"]["offset"] = 4
    result = check(case, rows)
    assert not result["0"]["errors"]
    assert result["1"]["expected_before"]["offset"] == 4
    assert "own-state predecessor/initial state join differs" in result["1"]["errors"]


@pytest.mark.parametrize("effect", ["store", "syscall"])
def test_effect_block_is_valid_but_successor_and_forced_commit_are_instrument_errors(effect):
    case, rows = fixture()
    rows["0"]["heads"][effect][0] = 2.
    rows["0"]["decoded"] = audit["helpers"]()["decoded"](rows["0"]["heads"])
    rows["0"]["driver_commit"] = {"committed": False}
    assert not errors(check(case, {"0": rows["0"]}))
    assert not errors(check(case, {"0": rows["0"]}, native=False))
    assert "successor after blocked effect" in errors(check(case, rows))
    rows["0"]["driver_commit"]["committed"] = True
    assert "native atomic commit/effect policy differs" in errors(check(case, {"0": rows["0"]}))


def test_early_truncation_is_not_silently_treated_as_budget_termination():
    case, rows = fixture()
    first = {"0": rows["0"]}
    assert "chain truncated before declared termination" in errors(check(case, first))
    assert "chain truncated before declared termination" in errors(check(case, first, native=False))
    assert not errors(check(case, first, budget=1))


def test_packed_fetch_tampering_is_distinct_from_own_state_join():
    case, rows = fixture()
    rows["1"]["roots"]["window"][0, 0] = 1. - rows["1"]["roots"]["window"][0, 0]
    result = check(case, rows)
    assert result["1"]["own_state_join"]
    assert not result["1"]["literal_root_joins"]["window"]
    assert "literal own-state fetch/regs/flags differ" in result["1"]["errors"]


def test_predecessor_identity_is_checked_even_when_numerical_state_matches():
    case, rows = fixture()
    rows["1"]["lineage"]["previous_row"] = "another.native.0"
    assert "native predecessor row identity differs" in errors(check(case, rows))


def test_margins_preserve_signed_zero_near_threshold_and_categorical_ties():
    heads = saved_heads([0] * 8)
    heads["regs"][0, 6, 1] = -.0001
    heads["flags"][0, 0] = 0.
    heads["dest"][0, :2] = 2.
    margins = audit["head_margins"](heads)
    assert len(margins) == 11
    assert margins["regs"]["nearest_signed_logit"] == -.0001
    assert margins["regs"]["nearest_coordinate"] == [0, 6, 1]
    assert margins["flags"]["exact_zero_logits"] == 1
    assert margins["dest"]["minimum_top_two_margin"] == 0.
    assert margins["dest"]["ties"] == 1
    assert margins["length"]["minimum_top_two_margin"] == 4.
    assert audit["helpers"]()["decoded"](heads)["flags"][0] == 0


def test_intervention_report_retains_collateral_register_flags_and_length_changes():
    _, rows = fixture()
    baseline, intervention = rows["0"], copy.deepcopy(rows["0"])
    intervention["heads"]["regs"][0, 0, 63] = 2.
    intervention["heads"]["flags"][0, 2] = 2.
    intervention["heads"]["length"][0, 2] = -2.
    intervention["heads"]["length"][0, 4] = 2.
    effect = audit["intervention_effect"](intervention, baseline)
    hard = effect["full_hard_state_changes"]
    assert effect["same_raw_roots"]
    assert hard["register_changed_bits"] == {"0": [63]}
    assert hard["flags_changed"] == [2]
    assert hard["length_changed"]
    assert effect["all_head_signed_effects"]["regs"]["signed_sum"] == 4.
    assert effect["RSI_bit1_effect"] == 0.

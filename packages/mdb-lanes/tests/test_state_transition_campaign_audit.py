"""Synthetic saved-array audit mechanics; no model/source equations execute."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path

import numpy as np

audit = runpy.run_path(str(Path(__file__).parents[1] / "qualification/analyze_state_transition_campaign.py"))


def heads(words, *, length=7, store=False):
    result = {key: np.full(shape, -1., dtype=np.float64) for key, shape in audit["SHAPES"].items()}
    for register, word in enumerate(words):
        for bit in range(64):
            result["regs"][0, register, bit] = 1. if word & (1 << bit) else -1.
    result["length"][0, length - 1] = 1.
    result["store"][0] = 1. if store else -1.
    return result


def fixture():
    plan = {"register_words": [0] * 6 + [3, 262], "flags": [0] * 4,
            "segments": [{"id": "first", "program_hex": "4881ef010000004881ef01000000", "max_instructions": 2},
                         {"id": "second", "program_hex": "4831f7", "max_instructions": 1}]}
    current, steps = copy.deepcopy({"register_words": plan["register_words"], "flags": plan["flags"], "offset": 0}), {}
    h = audit["helpers"]()
    for number, (segment, local_step, length) in enumerate(((plan["segments"][0], 0, 7),
                                                          (plan["segments"][0], 1, 7), (plan["segments"][1], 0, 3))):
        if local_step == 0:
            current["offset"] = 0
        output = list(current["register_words"])
        output[7] -= 1  # Arbitrary supplied saved decisions; no ISA semantic claim.
        raw = heads(output, length=length)
        decoded = h["decoded"](raw)
        after = {"register_words": output, "flags": [0] * 4, "offset": current["offset"] + length}
        lineage = {"segment_id": segment["id"], "segment_step": local_step, "program_hex": segment["program_hex"],
                   "host_segment_start": local_step == 0, "previous_row": None if number == 0 else "sequence.native." + str(number - 1)}
        steps[str(number)] = {"before": copy.deepcopy(current), "roots": h["raw_roots"](segment["program_hex"], current),
            "heads": raw, "decoded": decoded, "lineage": lineage, **lineage, "row_id": "sequence.native." + str(number),
            "driver_commit": {"committed": True, "before": copy.deepcopy(current), "after": after}}
        current = copy.deepcopy(after)
    return plan, steps


def errors(result):
    return [error for row in result.values() for error in row["errors"]]


def test_native_and_forecast_chains_close_without_reconditioning():
    plan, native = fixture()
    forecast = copy.deepcopy(native)
    forecast["0"]["heads"]["regs"][0, 6, 0] = -1.
    for number in ("1", "2"):
        forecast[number]["before"]["register_words"][6] = 2
        forecast[number]["roots"]["regs"][0, 6, 0] = 0.
        forecast[number]["heads"]["regs"][0, 6, 0] = -1.
    for row in forecast.values():
        row["decoded"] = audit["helpers"]()["decoded"](row["heads"])
    assert not errors(audit["chain_checks"](native, plan, native=True))
    assert not errors(audit["chain_checks"](forecast, plan, native=False))
    assert not np.array_equal(native["1"]["roots"]["regs"], forecast["1"]["roots"]["regs"])


def test_native_handoff_cannot_use_forecast_or_stale_registers():
    plan, rows = fixture()
    rows["1"]["before"]["register_words"][7] = 262
    result = audit["chain_checks"](rows, plan, native=True)
    assert "own-state predecessor/initial state join differs" in result["1"]["errors"]
    assert not result["0"]["errors"]


def test_only_declared_host_boundary_can_reset_offset():
    plan, rows = fixture()
    rows["1"]["before"]["offset"] = 0
    assert "own-state predecessor/initial state join differs" in audit["chain_checks"](rows, plan, native=True)["1"]["errors"]
    plan, rows = fixture()
    rows["2"]["host_segment_start"] = False
    assert "host segment start marker differs" in audit["chain_checks"](rows, plan, native=False)["2"]["errors"]


def test_effect_stop_has_no_successor_or_forced_commit():
    plan, rows = fixture()
    rows["0"]["heads"]["store"][0] = 1.
    rows["0"]["decoded"] = audit["helpers"]()["decoded"](rows["0"]["heads"])
    rows["0"]["driver_commit"] = {"committed": False}
    assert not errors(audit["chain_checks"]({"0": rows["0"]}, plan, native=True))
    assert "successor after blocked store/syscall" in audit["chain_checks"](rows, plan, native=True)["1"]["errors"]


def test_literal_root_tamper_and_incomplete_chain_are_retained():
    plan, rows = fixture()
    rows["1"]["roots"]["window"][0, 0] = 1 - rows["1"]["roots"]["window"][0, 0]
    assert "literal own-state input packing differs" in audit["chain_checks"](rows, plan, native=True)["1"]["errors"]
    assert "chain ends before declared host continuation completes" in audit["chain_checks"]({"0": rows["0"]}, plan, native=False)["0"]["errors"]


def test_direct_window_patch_preserves_every_other_column():
    before = np.zeros((1, 64, 293), dtype=np.float32)
    payload = np.ones((1, 64, 11), dtype=np.float32)
    after = before.copy()
    after[:, :, audit["DIRECT"]] = payload
    assert audit["direct_columns_exact"](before, after, payload, after.copy())
    after[:, :, 13] = 42.
    assert not audit["direct_columns_exact"](before, after, payload, after.copy())


def test_raw_decisions_are_strictly_positive_and_length_argmax_plus_one():
    raw = heads([0] * 8, length=3)
    raw["regs"][0, 6, 1] = 0.
    raw["flags"][0, 0] = 0.
    decoded = audit["helpers"]()["decoded"](raw)
    assert decoded["register_words"][6] == 0 and decoded["flags"][0] == 0 and decoded["length"] == 3

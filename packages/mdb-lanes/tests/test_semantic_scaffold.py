import json

import pytest

from mdb_lanes.semantic_scaffold import calibration_plan, evaluate, localization_plan, marked, plan, resolve_span


def test_source_span_moves_with_variable_length_and_order():
    a = marked("{wrong}: {source}", "amber", "x")
    b = marked("{wrong}: {source}", "violet", "long")
    assert a.text[slice(*a.source_span)] == "amber"
    assert b.text[slice(*b.source_span)] == "violet"
    assert resolve_span([(0, 2), (2, 6), (6, 9)], (3, 8)) == [1, 2]
    with pytest.raises(ValueError):
        resolve_span([(0, 1)], (1, 2))


def test_discovery_and_transfer_are_declared_before_execution():
    cases = plan()
    assert len(cases) == 8
    assert json.loads(json.dumps(cases)) == cases
    assert {c["lane"] for c in cases} == {"semantics", "copy", "recall", "coding"}
    for case in cases:
        for side in ("recipient", "donor"):
            p = case[side]
            assert p["text"][slice(*p["source_span"])] == case["source" if side == "recipient" else "donor_source"]
            assert p["text"][slice(*p["wrong_span"])] == case["wrong_source"]


def test_native_coding_completion_is_consumed_by_real_assertion():
    case = next(c for c in plan() if c["id"] == "coding-discovery")
    assert evaluate(case, "5\n")["executable_assertion_passed"] is True
    assert evaluate(case, "6\n")["executable_assertion_passed"] is False
    assert evaluate(case, "6\n", donor=True)["executable_assertion_passed"] is True
    assert evaluate(case, "__import__('os')")["executable_assertion_passed"] is None


def test_calibration_is_additive_and_keeps_the_semantic_route_external():
    initial, later = plan(), calibration_plan()
    assert not {c["id"] for c in initial} & {c["id"] for c in later}
    assert json.loads(json.dumps(later)) == later
    assert later[0]["split"] == "transfer"
    assert [c["split"] for c in later[1:]] == ["discovery", "transfer"]


def test_localization_uses_frozen_routes_and_marks_span_lowering_as_hypothesis():
    cases = localization_plan()
    assert len(cases) == 4
    assert all(c["split"] == "transfer" for c in cases)
    assert json.loads(json.dumps(cases)) == cases
    assert cases[-1]["span_lowering"] is True

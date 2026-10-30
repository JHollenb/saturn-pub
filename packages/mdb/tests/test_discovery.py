from __future__ import annotations

import pytest
import torch

from mdb.discovery import DiscoveryError, DiscoverySession
from mdb.discovery_runtime import ReplayForward
from mdb.forward_recording import record_forward


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.LayerNorm(3), torch.nn.Linear(3, 5),
                                         torch.nn.GELU(), torch.nn.Linear(5, 3))

    def forward(self, x):
        return x + self.block(x)


def recorded(value=1):
    torch.manual_seed(8)
    return record_forward(Residual().eval(), (torch.full((2, 3), float(value)),),
                          model_identity="discovery-controller-tiny",
                          execution={"backend": "cpu-fp32"}, input_identity={"value": value})


def session(budgets=None):
    capture = recorded()
    resident = DiscoverySession(budgets)
    resident.add_lane("a", lambda: ReplayForward(capture))
    return resident


def command(resident, op, **payload):
    result = resident.call(op, payload)
    assert result["ok"], result
    return result


def kinds(result):
    return [cause["kind"] for cause in result["stop_causes"]]


def test_law_exit_update_and_forward_are_distinct_native_points():
    resident = session()
    law = command(resident, "catalog", selector="mlp_001")["catalog"]
    result = command(resident, "motion", unit="law")
    assert result["completed_unit"] and result["lane"]["pc"] == law["exit"]
    before = command(resident, "inspect", selector="mlp_001:output")["value"]
    assert before["shape"] == [2, 3]
    result = command(resident, "motion", unit="update")
    assert result["lane"]["pc"] > law["exit"]
    assert result["lane"]["phase"] == "forward_complete"
    assert result["lane"]["status"] == "paused"
    result = command(resident, "motion", unit="op")
    assert "commit_required" in kinds(result)
    assert result["lane"]["status"] != "completed"


def test_entry_breakpoint_precedes_first_native_op_and_resumes_once():
    resident = session()
    command(resident, "break", selector="mlp_001", edge="enter", id="entry")
    result = command(resident, "motion", unit="law")
    assert "breakpoint" in kinds(result)
    assert result["lane"]["pc"] == 0
    assert result["work"]["replay_native_operations"] == 0
    result = command(resident, "motion", unit="law")
    assert result["completed_unit"]
    assert result["lane"]["pc"] > 0


def test_breakpoint_alias_pin_and_partial_progress_are_retained():
    resident = session()
    command(resident, "alias", name="focus", selector="mlp_001")
    command(resident, "break", id="pinned", selector="focus", edge="exit")
    command(resident, "alias", name="focus", selector="gelu_001")
    result = command(resident, "motion", unit="law", max_ops=2)
    assert not result["completed_unit"] and result["lane"]["pc"] == 2
    assert "budget" in kinds(result)
    spent = result["work"]["native_operations"]
    result = command(resident, "motion", unit="law")
    assert result["completed_unit"] and "breakpoint" in kinds(result)
    assert result["work"]["native_operations"] > spent
    assert resident.breakpoints["pinned"]["alias_revision"] == 1


def test_cuts_branch_isolation_and_work_does_not_rollback():
    resident = session()
    command(resident, "motion", unit="law")
    parent = command(resident, "capture", name="parent")["cut"]
    original = command(resident, "inspect", selector="mlp_001:output")["value"]["value"]
    command(resident, "fork", name="trial", cut=parent)
    assert resident.lanes["a"].branch_id == "main"
    command(resident, "select", name="trial")
    command(resident, "apply", selector="mlp_001:output", value=[[0, 0, 0], [0, 0, 0]])
    command(resident, "motion", unit="forward")
    trial = command(resident, "inspect", selector="outputs")["value"]
    spent = resident.work["native_operations"]
    command(resident, "select", name="main")
    assert command(resident, "inspect", selector="mlp_001:output")["value"]["value"] == original
    command(resident, "motion", unit="forward")
    native = command(resident, "inspect", selector="outputs")["value"]
    assert trial != native
    command(resident, "restore", cut=parent)
    assert resident.work["native_operations"] > spent
    assert resident.lanes["a"].branch.runtime.pc < len(resident.lanes["a"].branch.runtime.trace["events"])


def test_idempotent_command_retries_check_content_before_cached_response():
    resident = session()
    first = resident.call("motion", {"unit": "op"}, request_id="request", expected_revision=0)
    assert first["ok"]
    assert resident.call("motion", {"unit": "op"}, request_id="request", expected_revision=0) == first
    conflict = resident.call("motion", {"unit": "law"}, request_id="request", expected_revision=0)
    assert conflict["status"] == "request_conflict"
    stale = resident.call("motion", {"unit": "op"}, expected_revision=0)
    assert stale["status"] == "revision_conflict"
    assert resident.work["replay_native_operations"] == 1


def test_lanes_and_input_epochs_never_share_mutable_heads():
    resident = session()
    other = recorded(2)
    resident.add_lane("b", lambda: ReplayForward(other))
    command(resident, "lane", action="select", lane="b")
    assert resident.lanes["a"].revision == resident.lanes["b"].revision == 0
    command(resident, "motion", lane="b", unit="op")
    assert resident.lanes["a"].branch.runtime.pc == 0
    old_cut = resident.lanes["b"].branch.head
    resident.replace_input("b", lambda: ReplayForward(recorded(3)))
    assert resident.lanes["b"].epoch_id == "input_01"
    assert not resident.call("restore", {"lane": "b", "cut": old_cut})["ok"]
    command(resident, "lane", lane="b", action="checkout-epoch", epoch="input_00")
    assert resident.lanes["b"].branch.runtime.pc == 1


def test_capture_admission_occurs_before_factory():
    calls = []
    resident = DiscoverySession({"max_native_operations": 2})
    with pytest.raises(DiscoveryError, match="before reference capture"):
        resident.add_lane("a", lambda: calls.append(True), reference_operations=100,
                          reference_forwards=1)
    assert calls == []


def test_all_coincident_breakpoint_target_and_budget_causes_are_returned():
    native = recorded()
    law = ReplayForward(native).catalog.resolve("mlp_001")
    resident = session({"max_native_operations": law["exit"]})
    command(resident, "break", selector="mlp_001", edge="exit")
    result = command(resident, "motion", unit="law")
    assert set(kinds(result)) >= {"breakpoint", "motion_target", "budget"}
    assert result["hit_cut"] == result["final_head"]
    saved = resident.cuts[result["hit_cut"]]
    assert saved["runtime"].pc == law["exit"]


def test_change_watch_and_breakpoint_observe_the_same_safe_frontier():
    resident = session()
    command(resident, "motion", unit="law")
    command(resident, "watch", id="changed", selector="mlp_001:output")
    command(resident, "break", id="exit", selector="mlp_001", edge="exit")
    command(resident, "apply", selector="mlp_001:output", value=[[0, 0, 0], [0, 0, 0]])
    result = command(resident, "motion", unit="update", max_ops=0)
    assert set(kinds(result)) >= {"watchpoint", "breakpoint", "budget"}
    assert not result["completed_unit"]
    cut = resident.cuts[result["hit_cut"]]
    assert cut["runtime"].pc == resident.lanes["a"].branch.runtime.pc


def test_group_policy_parks_unaffected_lanes_and_lane_policy_advances_them():
    resident = session()
    capture = recorded(2)
    resident.add_lane("b", lambda: ReplayForward(capture))
    command(resident, "break", selector="mlp_001", edge="enter", lanes=["a"])
    result = command(resident, "motion", all=True, action="continue")
    assert result["group_pause_owner"] == "a"
    assert all(lane["pc"] == 0 for lane in result["lanes"])
    assert result["results"][1]["stop_causes"][0]["kind"] == "group_pause"
    command(resident, "restore", lane="b", cut="root")
    result = command(resident, "motion", all=True, action="continue", stop_policy="lane")
    assert resident.lanes["b"].branch.runtime.complete
    assert result["group_pause_owner"] is None


def test_reading_events_does_not_embed_prior_event_lists_recursively():
    resident = session()
    for _ in range(4):
        command(resident, "events")
    for row in resident.events:
        if row.get("operation") == "events":
            assert "events" not in row["response"]


def test_count_larger_than_remaining_scopes_keeps_actual_completed_count():
    resident = session()
    result = command(resident, "motion", unit="law", count=2)
    assert result["completed_units"] == 1 and not result["completed_unit"]
    assert result["lane"]["phase"] == "forward_complete"
    assert "commit_required" in kinds(result)
    assert "execution_error" not in kinds(result)


def test_allowlisted_act_specs_use_live_geometry_and_preserve_parent_cut():
    resident = session()
    command(resident, "motion", unit="law")
    command(resident, "alias", name="focus", selector="mlp_001")
    parent = command(resident, "capture", name="before-spec")["cut"]
    runtime = resident.lanes["a"].branch.runtime
    original = runtime.read("mlp_001:output")
    command(resident, "act", selector="focus:output",
            value_spec={"kind": "scale", "factor": 0.5})
    assert torch.equal(runtime.read("mlp_001:output"), original * 0.5)
    assert torch.equal(resident.cuts[parent]["runtime"].read("mlp_001:output"), original)
    command(resident, "act", selector="focus:output", value_spec={"kind": "zeros_like"})
    assert torch.equal(runtime.read("mlp_001:output"), torch.zeros_like(original))
    assert resident.work["preparation_operations"] == resident.work["act_commits"] == 2
    spent = resident.work["native_operations"]
    command(resident, "motion", unit="update")
    assert resident.work["native_operations"] == spent + 1


def test_act_specs_refuse_code_unknown_fields_and_ambiguous_values_without_mutation():
    resident = session()
    command(resident, "motion", unit="law")
    runtime = resident.lanes["a"].branch.runtime
    original = runtime.read("mlp_001:output")
    invalid = [{"kind": "eval", "expression": "x * 0"},
               {"kind": "scale", "factor": "0.5"},
               {"kind": "scale", "factor": True},
               {"kind": "scale", "factor": 10**1000},
               {"kind": "zeros_like", "shape": [1]}]
    for spec in invalid:
        result = resident.call("act", {"selector": "mlp_001:output", "value_spec": spec})
        assert not result["ok"]
        assert torch.equal(runtime.read("mlp_001:output"), original)
    assert not resident.call("act", {"selector": "mlp_001:output", "value": 0,
                                     "value_spec": {"kind": "zeros_like"}})["ok"]
    assert resident.work["act_commits"] == 0

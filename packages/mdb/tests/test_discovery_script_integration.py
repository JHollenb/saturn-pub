"""Discovery text/CLI integration against real tiny native recordings."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

from mdb.discovery import DiscoverySession
from mdb.discovery_cli import load_capture, main, save_capture
from mdb.discovery_runtime import ReplayForward
from mdb.discovery_script import DiscoveryJournal, ScriptRunner
from mdb.forward_recording import record_forward


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Linear(3,5),
                                         torch.nn.GELU(),torch.nn.Linear(5,3))

    def forward(self,value):
        return value+self.block(value)


def recorded(value=1.):
    torch.manual_seed(19)
    return record_forward(Residual().eval(),(torch.full((2,3),float(value)),),
        model_identity="discovery-script-integration", execution={"backend":"cpu-fp32"},
        input_identity={"value":value})


def session():
    capture = recorded()
    resident = DiscoverySession()
    resident.add_lane("a",lambda:ReplayForward(capture))
    return resident,capture


def commands(result):
    return [row for row in result["records"] if row["schema"] == "mdb-discovery-command-record-v1"]


def test_execution_error_preserves_prefix_and_cli_exits_unsuccessfully(tmp_path, monkeypatch):
    capture = recorded()
    recipe = tmp_path / "recipe"
    save_capture(capture, recipe)
    actions = tmp_path / "error.mdb"
    actions.write_text("interface mdb-discovery-v1\nstep forward\ninspect outputs\n")
    original = ReplayForward.advance_one

    def broken(runtime):
        if runtime.pc == 2:
            raise RuntimeError("injected native operation error")
        return original(runtime)

    monkeypatch.setattr(ReplayForward, "advance_one", broken)
    output = tmp_path / "failed"
    assert main(["replay", str(recipe), "--script", str(actions), "--output", str(output)]) == 1
    result = json.loads((output / "result.json").read_text())
    assert not result["ok"]
    response = result["last_response"]
    assert response["status"] == "errored" and response["lane"]["pc"] == 2
    assert response["hit_cut"] and len(response["receipts"]) == 2
    assert len(commands(result)) == 1


def test_text_alias_break_handler_capture_inspect_and_deferred_continue(tmp_path):
    resident,native = session()
    runner = ScriptRunner(resident,journal=DiscoveryJournal(tmp_path/"journal"))
    result = runner.run("""interface mdb-discovery-v1
alias focus = module:block
break focus exit --id chosen
on-hit chosen
  capture hit
  inspect focus:output
  continue --defer
end
step law
inspect outputs
""")
    assert result["ok"],result
    assert result["handler_hits"] == 1
    assert resident.lanes["a"].branch.phase == "forward_complete"
    assert torch.equal(resident.lanes["a"].branch.runtime.outputs(),native.output)
    records = commands(result)
    assert [row["command"] for row in records] == ["alias focus = module:block",
        "break focus exit --id chosen","step law","capture hit","inspect focus:output",
        "continue","inspect outputs"]
    hit = records[2]["response"]
    law = resident.lanes["a"].branch.runtime.catalog.resolve("mlp_001")
    assert hit["lane"]["pc"] == law["exit"]
    assert hit["hit_cut"] != records[3]["response"]["cut"]
    handler_capture = records[3]["payload"]["target"]
    assert {k:handler_capture[k] for k in ("lane_id","epoch_id","branch_id")} == {
        "lane_id":"a","epoch_id":"input_00","branch_id":"main"}
    assert records[4]["payload"]["target"]["expected_revision"] == records[3]["response"]["lane"]["revision"]
    assert records[5]["payload"]["target"]["expected_revision"] == records[4]["response"]["lane"]["revision"]
    assert DiscoveryJournal(tmp_path/"journal").verify() == result["records"]


def test_script_fork_act_and_restore_use_actual_native_suffix():
    resident,native = session()
    result = ScriptRunner(resident).run("""interface mdb-discovery-v1
step law
capture parent
fork trial
select trial
call {"operation":"act","payload":{"selector":"mlp_001:output","value":[[0,0,0],[0,0,0]]}}
step forward
inspect outputs
select main
step forward
inspect outputs
restore parent
inspect mlp_001:output
""")
    assert result["ok"],result
    rows = commands(result)
    candidate = next(row["response"]["value"] for row in rows if row["command"] == "inspect outputs")
    baseline = [row["response"]["value"] for row in rows if row["command"] == "inspect outputs"][1]
    assert candidate["value"] == [[1.,1.,1.],[1.,1.,1.]]
    assert baseline["value"] == native.output.tolist()
    assert candidate != baseline
    assert resident.lanes["a"].branch_id == "main"
    assert not resident.lanes["a"].branch.runtime.complete
    assert torch.equal(resident.lanes["a"].epoch.branches["trial"].runtime.outputs(),torch.ones(2,3))
    assert resident.work["replay_native_operations"] > len(native.trace["events"])


def test_failed_handler_preserves_hit_cut_and_completed_act_without_continuing():
    resident,_ = session()
    result = ScriptRunner(resident).run("""interface mdb-discovery-v1
break mlp_001 exit --id chosen
on-hit chosen
  capture before_edit
  call {"operation":"act","payload":{"selector":"mlp_001:output","value":[[0,0,0],[0,0,0]]}}
  inspect unknown_tensor_port
  continue --defer
end
step law
""")
    assert not result["ok"]
    rows = commands(result)
    assert rows[-1]["command"] == "inspect unknown_tensor_port"
    assert not any(row["command"] == "continue" for row in rows)
    assert "unknown" in result["last_response"]["error"]
    lane = resident.lanes["a"]
    assert lane.status == "paused"
    assert lane.branch.runtime.pc == lane.branch.runtime.catalog.resolve("mlp_001")["exit"]
    assert torch.equal(lane.branch.runtime.read("mlp_001:output"),torch.zeros(2,3))
    original_hit = next(row for row in rows if row["command"] == "step law")["response"]["hit_cut"]
    parent = resident.cuts[original_hit]["runtime"]
    assert torch.count_nonzero(parent.read("mlp_001:output")) > 0
    assert lane.branch.head != original_hit


def test_handler_branch_selection_updates_subsequent_target_and_revision():
    resident,_ = session()
    result = ScriptRunner(resident).run("""interface mdb-discovery-v1
break mlp_001 exit --id chosen
on-hit chosen
  capture parent
  fork trial
  select trial
  call {"operation":"break","payload":{"action":"disable","id":"chosen"}}
  call {"operation":"act","payload":{"selector":"mlp_001:output","value":[[0,0,0],[0,0,0]]}}
  continue --defer
end
step law
""")
    assert result["ok"],result
    rows = commands(result)
    edit = next(row for row in rows if row["operation"] == "act")
    selected = next(row for row in rows if row["command"] == 'call {"operation":"break","payload":{"action":"disable","id":"chosen"}}')
    assert edit["payload"]["target"] == {"lane_id":"a","epoch_id":"input_00",
        "branch_id":"trial","expected_revision":selected["response"]["lane"]["revision"]}
    assert resident.lanes["a"].branch_id == "trial"
    assert torch.equal(resident.lanes["a"].branch.runtime.outputs(),torch.ones(2,3))
    assert not resident.lanes["a"].epoch.branches["main"].runtime.complete


def test_root_recipe_cli_has_no_future_live_values_and_offline_journal_needs_no_torch(tmp_path,capsys):
    native = recorded()
    recipe = tmp_path/"recipe"
    manifest = save_capture(native,recipe)
    assert manifest["restore_mode"] == "replay-recipe"
    assert not manifest["native_frontier_restorable"]
    roots = torch.load(recipe/"roots.pt",weights_only=True)
    assert set(roots) == set(native.initial_values)
    assert all(native.trace["values"][key]["producer"] is None for key in roots)
    replay = load_capture(recipe)
    assert set(replay.inspect()["values"]) == set(native.initial_values)
    script = tmp_path/"actions.mdb"
    script.write_text("interface mdb-discovery-v1\ninspect frontier\nstep forward\ninspect outputs\n")
    output = tmp_path/"result"
    assert main(["replay",str(recipe),"--script",str(script),"--output",str(output)]) == 0
    capsys.readouterr()
    result = json.loads((output/"result.json").read_text())
    rows = commands(result)
    assert set(rows[0]["response"]["value"]["values"]) == set(native.initial_values)
    assert rows[-1]["response"]["value"]["value"] == native.output.tolist()
    # A fresh interpreter blocks frameworks at import resolution: an already
    # imported Torch in this pytest process cannot conceal an eager dependency.
    code = """import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','saturn','common','common_harness'}:
            raise RuntimeError('forbidden dependency: '+fullname)
sys.meta_path.insert(0,Block())
from mdb.discovery_cli import main
raise SystemExit(main(['inspect',sys.argv[1],'--last','1']))
"""
    run = subprocess.run([sys.executable,"-c",code,str(output/"journal")],
        text=True,capture_output=True,check=False,env=os.environ.copy(),cwd=tmp_path)
    assert run.returncode == 0,run.stderr
    offline = json.loads(run.stdout)
    assert offline["restore_mode"] == "metadata-only"
    assert offline["verified_records"] == len(result["records"])
    assert offline["records"][0] == rows[-1]


def test_old_epoch_explicit_target_cannot_mutate_replacement_epoch():
    resident,_ = session()
    old_target = {"lane_id":"a","epoch_id":"input_00","branch_id":"main","expected_revision":0}
    replacement = recorded(2.)
    resident.replace_input("a",lambda:ReplayForward(replacement))
    current = resident.lanes["a"]
    old_target["expected_revision"] = current.revision
    script = "interface mdb-discovery-v1\ncall "+json.dumps({"operation":"motion","payload":{
        "unit":"op","target":old_target}})
    result = ScriptRunner(resident).run(script)
    assert not result["ok"]
    assert result["last_response"]["status"] == "target_conflict"
    assert current.epoch_id == "input_01" and current.branch.runtime.pc == 0
    assert current.epochs["input_00"].branches["main"].runtime.pc == 0


def test_handler_focus_change_does_not_retarget_hit_bound_capture_or_continuation():
    resident,_ = session()
    other = recorded(2.)
    resident.add_lane("b",lambda:ReplayForward(other))
    assert resident.focus == "a"
    result = ScriptRunner(resident).run("""interface mdb-discovery-v1
break mlp_001 exit --id chosen --lane a
on-hit chosen
  lane focus b
  capture hit_on_a
  inspect mlp_001:output
  continue --defer
end
step law
""")
    assert result["ok"],result
    assert resident.focus == "b"
    hit = next(cut for cut in resident.cuts.values() if cut["name"] == "hit_on_a")
    assert hit["target"]["lane_id"] == "a"
    assert resident.lanes["a"].branch.runtime.complete
    assert resident.lanes["b"].branch.runtime.pc == 0


def test_coincident_budget_and_breakpoint_allow_read_handler_but_block_deferred_continue():
    native = recorded()
    law = ReplayForward(native).catalog.resolve("mlp_001")
    resident = DiscoverySession({"max_native_operations":law["exit"]})
    resident.add_lane("a",lambda:ReplayForward(native))
    result = ScriptRunner(resident).run("""interface mdb-discovery-v1
break mlp_001 exit --id chosen
on-hit chosen
  capture exhausted
  inspect mlp_001:output
  continue --defer
end
step law
""")
    assert result["ok"],result
    rows = commands(result)
    assert not any(row["command"] == "continue" for row in rows)
    stopped = next(row["response"] for row in rows if row["command"] == "step law")
    assert {cause["kind"] for cause in stopped["stop_causes"]} >= {"breakpoint","budget","motion_target"}
    assert any(cut["name"] == "exhausted" for cut in resident.cuts.values())
    assert resident.lanes["a"].branch.runtime.pc == law["exit"]
    assert not resident.lanes["a"].branch.runtime.complete


def test_stale_branch_target_cannot_advance_selected_candidate():
    resident,_ = session()
    assert ScriptRunner(resident).run("interface mdb-discovery-v1\nfork trial\nselect trial\n")["ok"]
    lane = resident.lanes["a"]
    command = {"operation":"motion","payload":{"unit":"op","target":{
        "lane_id":"a","epoch_id":"input_00","branch_id":"main",
        "expected_revision":lane.revision}}}
    result = ScriptRunner(resident).run("interface mdb-discovery-v1\ncall "+json.dumps(command))
    assert not result["ok"] and result["last_response"]["status"] == "target_conflict"
    assert lane.branch_id == "trial"
    assert lane.branch.runtime.pc == lane.epoch.branches["main"].runtime.pc == 0

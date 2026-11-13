"""User workflows, native authority, custody and falsifiable interface witnesses."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_saturn_pub_adapter import Counter, opened

from mdb import Project, SaturnPubDebuggerAdapter
from mdb_lanes.circuit_investigation import finding
from mdb_lanes.debugger_protocol import ActSpec
from mdb_lanes.project_navigation import command


def test_focus_seek_panel_finding_and_reopen(tmp_path):
    project, live, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("parent")["cut_id"]
    focus = project.focus("experiment", "port:value")
    assert project.focus("experiment") == focus
    assert project.explain("experiment")["state_materialized"] is False
    assert project.seek("experiment", live=live)["sample"] == 0
    plan = project.plan_comparison(
        "experiment",
        [{"port": "value", "value": 2}, {"port": "value", "mode": "scale", "value": 3}],
    )
    output = project.run_comparison("experiment", plan["record_id"], live=live)
    arms = [project.circuit_record("experiment", x) for x in output["arms"]]
    assert [r["measurement"]["value"] for r in arms] == [1, 3, 1, 7]
    assert output["interaction"] == 4  # Ordered scale after add, not an additive rectangle.
    assert project.session("experiment")["selected"] == parent
    assert live.session.read("value") == 0
    saved = finding(project, "experiment", "ordered cooperation", record_id=output["record_id"])
    assert saved
    owner.close()
    reopened = Project(project.root)
    assert reopened.focus("experiment") == focus
    assert reopened.circuit_record("experiment", output["record_id"]) == output
    assert reopened.findings("experiment")[0]["details"]["investigation_id"] == output["record_id"]
    assert reopened.graph().verify().ok


def test_offline_commands_never_load_a_model(tmp_path):
    project, live, owner, _ = opened(tmp_path)
    project.focus("experiment", "value")
    owner.close()
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','mrun','saturn','common'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
from mdb import Project
p = Project(sys.argv[1], read_only=True)
assert p.focus('experiment')['port'] == 'value'
assert p.explain('experiment')['state_materialized'] is False
assert p.circuit_records('experiment')
try: p.seek('experiment')
except ValueError as e: assert 'attached' in str(e)
else: raise AssertionError('offline seek executed')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parents[1] / "src"), str(Path(__file__).parents[1].parent / "mdb" / "src")])
    subprocess.run([sys.executable, "-c", code, str(project.root)], check=True, env=environment)


def test_pending_edits_foreign_focus_and_record_tampering_are_refused(tmp_path):
    project, live, owner, debug = opened(tmp_path)
    selected = project.focus("experiment", "value")
    spec = ActSpec("value", 9, parent_cut=debug.checkpoint()["cut_id"])
    live.preview_act(spec)
    with pytest.raises(ValueError, match="pending"):
        project.seek("experiment", live=live)
    live.abort_act(spec.act_id)
    other = Project(tmp_path / "other")
    with pytest.raises(ValueError, match="another project"):
        project.seek("experiment", live=type("Live", (), {"project": other})())
    with project._db() as db:
        data = dict(selected)
        data.pop("record_id")
        data["port"] = "forged"
        db.execute(
            "UPDATE circuit_records SET body=? WHERE id=?",
            (json.dumps(data), selected["record_id"]),
        )
    with pytest.raises(ValueError, match="seal"):
        project.focus("experiment")
    owner.close()


def test_partial_panel_preserves_completed_native_results(tmp_path):
    project, live, owner, _ = opened(tmp_path)
    project.focus("experiment", "value")
    plan = project.plan_comparison("experiment", [{"port": "tick", "value": 2}])
    result = project.run_comparison("experiment", plan["record_id"], live=live)
    arms = [project.circuit_record("experiment", x) for x in result["arms"]]
    assert arms[0]["status"] == "measured"
    assert arms[1]["status"] == "evidence-incomplete"
    assert result["interaction"] is None
    assert live.session.read("tick") == 0
    owner.close()


def test_virtual_slice_node_replays_its_exact_transition_prefix(tmp_path):
    project, live, owner, _ = opened(tmp_path)
    # One native multi-transition receipt has unretained intermediate boundaries.
    live.session.continue_(3)
    live.checkpoint()
    sliced = project.slice("experiment", port="value")
    nodes = [
        n
        for n in sliced["graph"]["nodes"]
        if n["content"].get("metadata", {}).get("boundary", {}).get("transition") == 1
        and n["content"].get("metadata", {}).get("slot") == "value"
    ]
    assert len(nodes) == 1
    project.focus("experiment", slice_id=sliced["slice_id"], node_id=nodes[0]["node_id"])
    result = project.seek("experiment", live=live)
    assert result["sample"] == 1 and result["inspect_only"] is True
    assert live.session.read("value") == 3
    owner.close()


def test_interface_finds_omitted_state_and_retains_reachable_witnesses(tmp_path):
    from saturn_pub import Session

    class Contextual(Counter):
        model_identity = "contextual-counter"

        def advance(self, state):
            return {"value": state["value"] + state["tick"] + 1, "tick": state["tick"] + 1}

    project = Project(tmp_path / "interface")
    live = SaturnPubDebuggerAdapter(
        Session(Contextual(), {"value": 0, "tick": 0}), project=project, session_id="trial"
    )
    project.focus("trial", "value")
    plan = project.plan_comparison("trial", [{"port": "value", "mode": "replace", "value": 0}])
    first = project.run_comparison("trial", plan["record_id"], live=live)
    live.step()
    project.focus("trial", "value")
    plan = project.plan_comparison("trial", [{"port": "value", "mode": "replace", "value": 0}])
    second = project.run_comparison("trial", plan["record_id"], live=live)
    cases = []
    for run in (first, second):
        arm = project.circuit_record("trial", run["arms"][1])
        cases.append({"source": arm["input_cut"], "result": arm["result_cut"]})
    test = project.test_interface("trial", cases, [{"port": "value"}], {"port": "value"}, live=live)
    assert len(test["witnesses"]) == 1
    assert test["terminal_status"] == "not-established"
    repaired = project.test_interface(
        "trial", cases, [{"port": "value"}, {"port": "tick"}], {"port": "value"}, live=live
    )
    assert repaired["witnesses"] == [] and repaired["repeated_argument_groups"] == 0
    with pytest.raises(ValueError, match="branch history"):
        project.test_interface(
            "trial",
            [{"source": cases[1]["source"], "result": cases[0]["result"]}],
            [{"port": "value"}],
            {"port": "value"},
            live=live,
        )


def test_console_and_offline_cli_share_the_focus(tmp_path):
    from mdb_lanes.project import main

    project, live, owner, _ = opened(tmp_path)
    selected = command(project, "experiment", ["focus", "port:value"], live=live)
    assert command(project, "experiment", ["seek"], live=live)["sample"] == 0
    assert command(project, "experiment", ["explain"])["focus"] == selected
    assert main([str(project.root), "focus"]) == 0
    assert main([str(project.root), "seek"]) == 1
    owner.close()


def test_selecting_a_saved_result_restores_its_context_not_native_state(tmp_path):
    project, live, owner, _ = opened(tmp_path)
    first = command(project, "experiment", ["focus", "value"], live=live)
    old = project.seek("experiment", live=live)
    live.step()
    command(project, "experiment", ["focus", "value"], live=live)
    assert project.focus("experiment")["record_id"] != first["record_id"]
    command(project, "experiment", ["focus", "--record-id", old["record_id"]], live=live)
    assert project.focus("experiment") == first
    assert live.session.read("value") == 1
    assert live.status()["circuit_focus"]["historical"]
    owner.close()


@pytest.fixture
def tiny_qwen(tmp_path):
    torch = pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        native = DecoderAdapter.tiny("qwen3", seed=17)
        project = Project(tmp_path / "qwen")
        live = SaturnPubDebuggerAdapter(
            native.session([1, 2, 3]), project=project, session_id="qwen"
        )
        live.step()  # Embed the pending token; no model download or fleet.
        yield project, live
    finally:
        torch.set_num_threads(old)


def test_tiny_original_ports_source_frequency_and_full_suffix_adjoint(tiny_qwen):
    project, live = tiny_qwen
    selected = live.session.capture(retain=False).fingerprint
    project.focus("qwen", "hidden", steps=4)
    mlp = project.explain("qwen", view="mlp", live=live)["result"]
    assert mlp["mlp"]["neurons"]
    assert mlp["attention"]["probability_authority"] == "captured native eager attention"
    freq = project.explain("qwen", view="frequency", live=live)["result"]["frequency"]
    assert freq["return_closure_max_abs"] < 1e-12
    effect = project.explain("qwen", view="effect", live=live)["result"]["effect"]
    assert effect["primal_logits_exact"]
    assert effect["gradient_norm"] > 0 and effect["neurons"]
    project.focus("qwen", "hidden", consumer={"port": "logits", "selector": [0, 4]})
    coordinate = project.explain("qwen", view="effect", live=live)["result"]["effect"]
    assert coordinate["primal_logits_exact"]
    assert coordinate["native_readout"]["value"] == coordinate["value"]
    assert live.session.capture(retain=False).fingerprint == selected
    result = command(
        project,
        "qwen",
        ["parameter", "model.layers.0.mlp.gate_proj.weight", "--selector", "[0]"],
        live=live,
    )
    assert result["persisted"] is False and len(result["sample"]) <= 16
    # No parameter or activation arrays are hidden inside derived records.
    assert all(
        "original-model-view" in json.dumps(r) or r["kind"] != "explanation"
        for r in project.circuit_records("qwen")
    )


def test_tiny_native_panel_owns_its_tokens_and_preserves_parent(tiny_qwen):
    project, live = tiny_qwen
    parent = live.session.capture(retain=False).fingerprint
    project.focus("qwen", "hidden", steps=20)
    plan = project.plan_comparison(
        "qwen",
        [
            {"port": "hidden", "mode": "scale", "value": 0.9},
            {"port": "hidden", "mode": "add", "value": 0.01},
        ],
        tokens=2,
    )
    output = project.run_comparison("qwen", plan["record_id"], live=live)
    arms = [project.circuit_record("qwen", r) for r in output["arms"]]
    assert all(a["status"] == "measured" and len(a["own_tokens"]) == 2 for a in arms)
    assert len({a["measurement"]["consumer"]["token"] for a in arms}) == 1
    assert live.session.capture(retain=False).fingerprint == parent
    assert project.graph().verify().ok


def test_tiny_panel_closes_finite_ports_and_local_prediction(tiny_qwen):
    project, live = tiny_qwen
    project.focus("qwen", "hidden", consumer={"port": "logits", "token": 4, "contrast": 5})
    plan = project.plan_comparison(
        "qwen",
        [{"port": "hidden", "value": 0.0001}, {"port": "hidden", "value": -0.0001}],
        linearize=True,
    )
    result = project.run_comparison("qwen", plan["record_id"], live=live)
    assert result["linearization"]["status"] == "native-primal-matched"
    arms = [project.circuit_record("qwen", r) for r in result["arms"]]
    assert all(a["status"] == "measured" for a in arms)
    for arm in arms:
        assert arm["local_accounting"]["mlp"]["closure_max_abs"] < 1e-12
        assert arm["local_accounting"]["attention"]["closure_max_abs"] < 1e-12
        assert abs(arm["native_minus_local_prediction"]) < 5e-4
    assert project.run_comparison("qwen", "@latest", live=live)["record_id"] == result["record_id"]


def test_tiny_focus_hydrates_from_token_recipe_without_loading_target_payload(
    tiny_qwen, monkeypatch
):
    from mdb_lanes.circuit_investigation import attach_replay
    from mdb_lanes.native_replay_recipe import SCHEMA, seal

    project, live = tiny_qwen
    selected = project.focus("qwen", "hidden")
    recipe = seal(
        {
            "schema": SCHEMA,
            "model_identity": live.session.adapter.model_identity,
            "execution": dict(live.session.adapter.execution),
            "roots": [{"id": "root", "token_ids": [1, 2, 3]}],
            "events": [
                {
                    "id": "embedded",
                    "session": "root",
                    "kind": "continue",
                    "steps": 1,
                    "expected_cut": selected["boundary"]["cut"],
                }
            ],
        }
    )
    attach_replay(project, "qwen", recipe, "embedded")
    monkeypatch.setattr(
        project,
        "load_cut",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("old tensor payload loaded")),
    )
    result = project.seek("qwen", live=live)
    assert result["value"] == selected["expected_value"]
    assert result["active_branch_preserved"]
    monkeypatch.undo()
    plan = project.plan_comparison("qwen", [{"port": "hidden", "value": 0.0001}])
    comparison = project.run_comparison("qwen", plan["record_id"], live=live)
    arms = [project.circuit_record("qwen", r) for r in comparison["arms"]]
    interface = project.test_interface(
        "qwen",
        [{"source": a["input_cut"], "result": a["result_cut"]} for a in arms],
        [{"port": "hidden", "selector": [0, 0, 0]}],
        {"port": "logits", "selector": [0, 4]},
        live=live,
    )
    assert all(
        c["result_value"]["kind"] == "tensor" and c["result_value"]["shape"] == []
        for c in interface["cases"]
    )


def test_tiny_walkthrough_report_matches_declared_live_checks(tmp_path, monkeypatch):
    import runpy

    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    path = Path(__file__).parents[1].parent / "mdb" / "examples/project-navigation/circuit_understanding.py"
    output = tmp_path / "walkthrough"
    monkeypatch.setattr(sys, "argv", [str(path), "--out", str(output)])
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        torch.set_num_threads(threads)
    report = json.loads((output / "report.json").read_text())
    assert report["active_parent_preserved"] and report["reopen_exact"]
    assert report["root_recipe_target_exact"] and report["native_primal_exact"]
    assert report["parameters"] == 22720 and report["graph_circuit_nodes"] >= 14
    assert len(report["interface_witnesses"]) >= 1
    assert all(row["decision"] == row["own_tokens"][-1] for row in report["arms"])


def test_tiny_qwen2_uses_the_same_original_port_and_consumer_path(tmp_path):
    torch = pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        native = DecoderAdapter.tiny("qwen2", seed=17)
        project = Project(tmp_path / "qwen2")
        live = SaturnPubDebuggerAdapter(
            native.session([1, 2, 3]), project=project, session_id="qwen2"
        )
        live.step()
        parent = live.session.capture(retain=False).fingerprint
        project.focus("qwen2", "hidden")
        effect = project.explain("qwen2", view="effect", live=live)["result"]["effect"]
        assert effect["primal_logits_exact"]
        plan = project.plan_comparison(
            "qwen2", [{"port": "hidden", "mode": "scale", "value": 0.99}]
        )
        panel = project.run_comparison("qwen2", plan["record_id"], live=live)
        for ref in panel["arms"]:
            arm = project.circuit_record("qwen2", ref)
            assert arm["status"] == "measured"
            assert arm["own_tokens"] == [arm["measurement"]["decision"]]
        assert live.session.capture(retain=False).fingerprint == parent
    finally:
        torch.set_num_threads(threads)

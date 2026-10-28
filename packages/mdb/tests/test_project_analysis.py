"""Automatic analysis and saved findings retain descriptive evidence authority."""

import json

import pytest
from saturn_pub import Session
from saturn_pub.contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from test_saturn_pub_adapter import Counter, opened

from mdb import Project, SaturnPubDebuggerAdapter
from mdb.cli import main
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import ActSpec
from mdb.project_analysis import REGISTRY, Analyzer, register


class Pipeline(Counter):
    model_identity = "supplied-pipeline"

    def point(self, state):
        return ExecutionPoint("pipeline", 0, "layer", index=state["stage"])

    def boundary(self, state):
        return f"layer:{state['stage']}"

    def surface(self, state):
        return SurfaceManifest(tuple(SlotSpec(k) for k in state), "pipeline-v1", "fixture")

    def validate(self, state):
        if set(state) != {"a", "b", "c", "stage"}:
            raise ValueError("invalid fixture")

    def transition(self, state):
        return (
            TransitionSpec("first", ("a",), ("b", "stage"), footprint="declared")
            if state["stage"] == 0
            else TransitionSpec("second", ("b",), ("c", "stage"), footprint="declared")
        )

    def advance(self, state):
        return {
            **state,
            **({"b": state["a"] + 1} if state["stage"] == 0 else {"c": state["b"] + 1}),
            "stage": state["stage"] + 1,
        }


def pipeline(tmp_path):
    project = Project(tmp_path / "pipeline")
    native = Session(Pipeline(), {"a": 1, "b": 0, "c": 0, "stage": 0})
    adapter = SaturnPubDebuggerAdapter(native, project=project, session_id="trial")
    adapter.step(2)
    return project, adapter


def test_structural_analysis_automatic_reopen_cache_and_staleness(tmp_path):
    project, adapter = pipeline(tmp_path)
    first = project.analyze("trial")
    assert first["binding"]["model_identity"] == "supplied-pipeline"
    assert {g["point"]["index"] for g in first["result"]["groups"]} == {0, 1, 2}
    assert first["result"]["terminal_status"] == "not-evaluated"
    assert project.analyze("trial") == first
    assert len(project.analyses("trial")) == 1
    adapter.step()
    reopened = Project(project.root)  # Opening refreshes the default analyzer automatically.
    history = reopened.analyses("trial")
    assert len(history) == 2
    assert history[0]["analysis_id"] == first["analysis_id"]
    assert not history[0]["current"] and history[1]["current"]
    assert reopened.analyze("trial")["analysis_id"] == history[1]["analysis_id"]


def test_registry_versions_and_failed_analysis_do_not_erase_prior_results(tmp_path, monkeypatch):
    project, _ = pipeline(tmp_path)
    monkeypatch.setattr("mdb.project_analysis.REGISTRY", dict(REGISTRY))
    for version in ("1", "2"):
        register(
            Analyzer("fixture", version, lambda p, sid, source: {"observed": len(source["cuts"])})
        )
    with pytest.raises(ValueError, match="explicit version"):
        project.analyze("trial", analyzer="fixture")
    old = project.analyze("trial", analyzer="fixture", version="1")
    new = project.analyze("trial", analyzer="fixture", version="2")
    assert old["analysis_id"] != new["analysis_id"]
    with pytest.raises(ValueError, match="already registered"):
        register(Analyzer("fixture", "1", lambda *args: {}))

    def broken(*args):
        raise RuntimeError("instrument broken")

    register(Analyzer("broken", "1", broken))
    before = project.analyses("trial")
    with pytest.raises(RuntimeError, match="instrument broken"):
        project.analyze("trial", analyzer="broken")
    assert project.analyses("trial") == before


def test_unknown_footprints_and_empty_history_are_diagnostics(tmp_path):
    project, adapter, owner, _ = opened(tmp_path)
    assert {g["kind"] for g in project.analyze("experiment")["result"]["gaps"]} == {
        "no-recorded-transitions"
    }
    original = adapter.session.adapter.transition
    adapter.session.adapter.transition = lambda state: TransitionSpec("unknown")
    adapter.step()
    result = project.analyze("experiment")["result"]
    assert any(g["kind"] == "unknown-footprint" for g in result["gaps"])
    assert result["mechanics_status"] == "valid"
    adapter.session.adapter.transition = original
    owner.close()


def test_dependency_walk_multihop_bounded_cycles_and_native_backtrace(tmp_path):
    project, _ = pipeline(tmp_path)
    trace = project.trace("trial", "port:c")
    assert {"operation:second", "port:b", "operation:first", "port:a"} <= set(trace["nodes"])
    assert trace["causal_effect"] == "not-established"
    assert project.trace("trial", "port:a", direction="downstream")["edges"]
    assert project.trace("trial", "port:c", depth=1)["truncated"]
    assert [r["operation"] for r in project.backtrace("trial", "c")["writers"]] == [
        "second",
        "first",
    ]
    for bad in (0, 33, True):
        with pytest.raises(ValueError, match="depth"):
            project.trace("trial", "port:c", depth=bad)
    with pytest.raises(ValueError, match="direction"):
        project.trace("trial", "port:c", direction="causal")


def test_native_backtrace_does_not_mix_aborted_futures(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step()
    parent = debug.checkpoint("parent")["cut_id"]
    spec = ActSpec("value", 8, parent_cut=parent)
    preview = adapter.preview_act(spec)
    adapter.abort_act(spec.act_id)
    assert [row["operation"] for row in project.backtrace("experiment", "value")["writers"]] == [
        "increment"
    ]
    candidate = project.backtrace("experiment", "value", cut_id=preview["candidate_cut"])
    assert candidate["writers"][0]["operation"] != "increment"
    debug.restore(parent)
    assert project.backtrace("experiment", "value")["writers"][0]["status"] == "ancestry-boundary"
    owner.close()


def test_findings_bind_evidence_and_keep_claim_kinds_separate(tmp_path):
    project, _ = pipeline(tmp_path)
    receipt = project.xrefs("trial", "port:c")[0]["receipt"]
    for kind in ("observation", "inference", "causal-claim"):
        project.finding(
            "trial",
            "port:c",
            f"bounded {kind}",
            kind=kind,
            note="supplied fixture",
            evidence=(receipt,),
        )
    reopened = Project(project.root)
    findings = reopened.findings("trial")
    assert {r["details"]["kind"] for r in findings} == {"observation", "inference", "causal-claim"}
    assert all(r["details"]["terminal_status"] == "not-established" for r in findings)
    assert all(r["provenance"]["authority"] == "descriptive-only" for r in findings)
    assert all(r["details"]["evidence"] == [receipt] for r in findings)
    assert reopened.graph().verify().ok
    with pytest.raises(ValueError, match="evidence"):
        project.finding("trial", "port:c", "unsupported", evidence=("foreign",))
    with pytest.raises(ValueError, match="title"):
        project.finding("trial", "port:c", "")
    with pytest.raises(ValueError, match="kind"):
        project.finding("trial", "port:c", "unknown", kind="certified")


def test_analysis_seals_and_schema_one_upgrade(tmp_path):
    project, _ = pipeline(tmp_path)
    with project._db() as db:
        db.execute("DROP TABLE analyses")
        db.execute("PRAGMA user_version=1")
    upgraded = Project(project.root)
    assert upgraded.analyses("trial")
    with upgraded._db() as db:
        db.execute("UPDATE analyses SET body='{}'")
    with pytest.raises(ValueError, match="analysis seal"):
        Project(project.root)


def test_cli_and_attached_shell_analysis_to_saved_finding(tmp_path, capsys):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step(2)
    shell = DebugShell(debug, json_output=True)
    for line in (
        "project analyze",
        "project callers port:value",
        "project consumers port:value",
        "project trace port:value --depth 3",
        "project backtrace value",
        'project finding port:value --title "increment observed" --kind observation',
        "project findings",
        "project analyses",
    ):
        assert shell.execute(line)
    capsys.readouterr()
    for action in ("analyze", "analyses", "findings"):
        assert main(["project", str(project.root), action]) == 0
        assert json.loads(capsys.readouterr().out)
    assert Project(project.root).findings("experiment")[0]["label"] == "increment observed"
    owner.close()


def test_decoder_permissions_follow_typed_boundary_and_restore(tmp_path):
    pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter

    from mdb import BenchSession
    from mdb.debugger_protocol import DebuggerProtocolError

    native = DecoderAdapter.tiny("qwen3")
    with BenchSession.from_saturn_pub(
        native.session([1, 2, 3]), project=Project(tmp_path / "decoder")
    ) as owner:
        debug = owner.debugger()
        parent = debug.checkpoint("parent")["cut_id"]
        with pytest.raises(DebuggerProtocolError, match="read-only"):
            debug.call("act.preview", {"act": ActSpec("hidden", 0, parent_cut=parent).to_dict()})
        debug.step(2)
        addresses = debug.capabilities()["address_specs"]
        assert next(a for a in addresses if a["path_template"] == "hidden")["writable"]
        shell = DebugShell(debug, json_output=True)
        assert shell.execute("set hidden 0")
        assert shell.execute("commit")
        assert owner.adapter.session.read("hidden").count_nonzero().item() == 0
        debug.restore(parent)
        with pytest.raises(DebuggerProtocolError, match="read-only"):
            debug.call("act.preview", {"act": ActSpec("hidden", 0, parent_cut=parent).to_dict()})
        with pytest.raises(DebuggerProtocolError, match="read-only"):
            debug.call("act.preview", {"act": ActSpec("tokens", 0, parent_cut=parent).to_dict()})

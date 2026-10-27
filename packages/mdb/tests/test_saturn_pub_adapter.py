"""Public execution, MDB protocol and durable investigations on one native state."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from saturn_pub import Session, StateAddress
from saturn_pub.contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from saturn_pub.core import Adapter
from saturn_pub.symbols import SymbolBinding, SymbolTable

from mdb import BenchSession, Project, SaturnPubDebuggerAdapter
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import ActSpec, BreakpointSpec, DebuggerProtocolError
from mdb.state_graph import StateGraph


class Counter(Adapter):
    model_identity = "public-counter-v1"
    execution = {"family": "counter", "adapter": "test-counter-v1"}

    def boundary(self, state):
        return f"tick:{state['tick']}"

    def point(self, state):
        return ExecutionPoint("counter", state["tick"], "tick", next_operation="increment")

    def surface(self, state):
        return SurfaceManifest(
            (SlotSpec("value", writable=True), SlotSpec("tick", role="runtime")),
            "test-counter-state-v1",
            "counter-native",
        )

    def addresses(self, state):
        return tuple(state)

    def validate(self, state):
        if set(state) != {"value", "tick"} or any(type(v) is not int for v in state.values()):
            raise ValueError("counter needs integer value and tick")

    def transition(self, state):
        return TransitionSpec(
            "increment", ("value", "tick"), ("value", "tick"), footprint="declared"
        )

    def advance(self, state):
        return {"value": state["value"] + 1, "tick": state["tick"] + 1}


def native():
    return Session(Counter(), {"value": 0, "tick": 0})


def opened(tmp_path):
    project = Project(tmp_path / "project")
    adapter = SaturnPubDebuggerAdapter(native(), project=project, session_id="experiment")
    owner = BenchSession.from_backend(adapter, identity=adapter.identity)
    return project, adapter, owner, owner.debugger()


def test_public_engine_steps_symbols_backtraces_and_watch_stops(tmp_path, capsys):
    project, adapter, owner, debug = opened(tmp_path)
    binding = SymbolBinding.bind(
        adapter.session,
        "counter.value",
        (StateAddress("value"),),
        context="panel",
        context_slots=(),
        evidence=("fixture",),
        claim="Supplied counter value, not learned semantics.",
    )
    adapter.symbols = SymbolTable((binding,))
    adapter.symbol_context = "panel"
    assert debug.inspect("symbol:counter.value")["value"]["kind"] == "tuple"
    shell = DebugShell(debug, json_output=True)
    for line in (
        "info symbols",
        "info steps",
        "resolve counter.value",
        "read symbol:counter.value",
        "public watch value change",
    ):
        assert shell.execute(line)
    result = debug.step(5)
    assert result["steps"] == 1
    assert result["stop"] == "watchpoint"
    assert adapter.session.read("value") == 1
    assert adapter.public.watch_events[0].receipt == adapter.session.receipts[-1].fingerprint
    assert shell.execute("backtrace value")
    assert "declared-dependency" in capsys.readouterr().out
    # Qualified symbols stay bound to their original execution point.
    with pytest.raises(ValueError, match="qualified binding"):
        adapter.command("resolve counter.value")
    assert project.graph().verify().ok
    assert any(n.labels == ("counter.value",) for n in project.graph().nodes.values())
    owner.close()


def test_mdb_edits_are_public_forks_and_rollback_is_exact(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("parent")["cut_id"]
    before = adapter.session.capture(retain=False)
    spec = ActSpec("value", 9, parent_cut=parent)
    preview = adapter.preview_act(spec)
    assert adapter.session.capture(retain=False).fingerprint == before.fingerprint
    assert adapter._cut(preview["candidate_cut"]).payload["value"] == 9
    adapter.abort_act(spec.act_id)
    assert adapter.session.capture(retain=False).fingerprint == before.fingerprint
    spec = ActSpec("value", 4, parent_cut=parent, sign="add")
    adapter.preview_act(spec)
    adapter.commit_act(spec.act_id)
    assert adapter.session.read("value") == 4
    adapter.restore(parent)
    assert adapter.session.capture(retain=False).fingerprint == before.fingerprint
    assert project.graph().verify().ok
    with pytest.raises(ValueError, match="stale"):
        adapter.preview_act(ActSpec("value", 0, parent_cut="stale"))
    owner.close()


def test_mdb_breakpoints_and_native_diff_share_retained_cuts(tmp_path):
    _, adapter, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("parent")["cut_id"]
    debug.add_breakpoint(BreakpointSpec("transition", 2))
    debug.continue_run()
    assert adapter.session.read("tick") == 2
    assert debug.status()["state"] == "paused"
    child = debug.checkpoint("child")["cut_id"]
    diff = debug.diff("value", parent, child)
    assert diff["equal"] is False
    assert adapter.session.read("value") == 2
    result = debug.replay(parent)
    assert result["exact"] is True
    assert adapter.session.read("value") == 2  # Replay observes a private suffix.
    owner.close()


def test_project_reopen_annotations_graph_and_multiple_sessions(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step(2)
    selected = debug.checkpoint("chosen")["cut_id"]
    annotation = project.annotate("experiment", "chosen state", note="retained public cut")
    graph = project.graph().seal().to_dict()
    owner.close()
    reopened = Project(project.root)
    assert reopened.graph().seal().to_dict() == graph
    assert reopened.annotations(query="chosen")[0]["annotation_id"] == annotation["annotation_id"]
    resumed = reopened.resume("experiment", Counter())
    assert resumed.session.capture(retain=False).fingerprint == selected
    assert resumed.session.read("value") == 2
    with pytest.raises(ValueError, match="already exists"):
        SaturnPubDebuggerAdapter(native(), project=reopened, session_id="experiment")
    other = SaturnPubDebuggerAdapter(native(), project=reopened, session_id="other")
    other.step()
    assert len(reopened.sessions()) == 2
    assert reopened.graph().verify().ok
    assert resumed.session.read("value") == 2
    with pytest.raises(ValueError, match="not registered"):
        reopened.annotate("other", "foreign", cut_id=selected)


def test_project_rejects_stale_writers_and_failed_commit_preserves_parent(tmp_path, monkeypatch):
    project, adapter, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("parent")["cut_id"]
    spec = ActSpec("value", 10, parent_cut=parent)
    adapter.preview_act(spec)
    save = project.save_session
    calls = 0

    def broken(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publication unavailable")
        return save(*args, **kwargs)

    monkeypatch.setattr(project, "save_session", broken)
    with pytest.raises(OSError, match="publication"):
        adapter.commit_act(spec.act_id)
    assert adapter.session.capture(retain=False).fingerprint == parent
    assert project.session("experiment")["selected"] == parent
    monkeypatch.setattr(project, "save_session", save)
    adapter.commit_act(spec.act_id)
    assert adapter.session.read("value") == 10
    current = project.resume("experiment", Counter())
    with pytest.raises(ValueError, match="another writer"):
        adapter.checkpoint()
    with pytest.raises(ValueError, match="another writer"):
        adapter.step()
    assert adapter.session.read("value") == 10
    assert current.session.read("value") == 10
    owner.close()


def test_project_rejects_model_mismatch_and_corrupt_native_pages(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    selected = debug.checkpoint("retained")["cut_id"]
    incompatible = Counter()
    incompatible.model_identity = "different-counter"
    with pytest.raises(ValueError, match="incompatible"):
        project.resume("experiment", incompatible)
    assert project.session("experiment")["selected"] == selected
    metadata = json.loads((project.store.root / "cuts" / f"{selected}.json").read_text())
    page = project.store.root / "pages" / f"{metadata['slots']['value']}.json"
    page.write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        project.resume("experiment", Counter())
    owner.close()


def test_project_rejects_changed_native_receipts(tmp_path):
    project, _, owner, debug = opened(tmp_path)
    debug.step()
    receipt = project.receipts("experiment")[0]
    path = project.store.root / "receipts" / f"{receipt.fingerprint}.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="native receipt bytes"):
        project.graph()
    owner.close()


def test_project_hydrates_historical_cuts_on_demand_and_keeps_watch_history(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("parent")["cut_id"]
    adapter.command("watch value change")
    for _ in range(4):
        debug.step()
    assert len(adapter._cuts) >= 5
    assert sum(cut is not None for cut in adapter._cuts.values()) <= 1
    assert len(adapter.public.watch_events) == 1
    assert len(adapter.inspect("watch_events")["events"]) == 4
    assert (
        len([node for node in project.graph().nodes.values() if "diagnostic-watch" in node.labels])
        == 4
    )
    debug.restore(parent)
    assert adapter.session.read("value") == 0
    assert project.resume("experiment", Counter()).session.read("value") == 0
    owner.close()


def test_project_imports_existing_native_graph_and_annotations_without_rewriting(tmp_path):
    graph = StateGraph(binding={"model": "legacy-model", "session": "legacy-session"})
    cut = graph.add_state(cut_ref="cas://sha256/" + "a" * 64)
    annotation = graph.add_source(source_ref="legacy-annotation", labels=("subject store",))
    graph.add_edge(annotation, cut, "annotates", graph="knowledge")
    snapshot = graph.seal()
    project = Project(tmp_path / "project")
    assert project.import_graph(snapshot) == snapshot.snapshot_id
    assert project.import_graph(snapshot.to_dict()) == snapshot.snapshot_id
    joined = Project(project.root).graph()
    assert joined.nodes[cut].to_dict() == graph.nodes[cut].to_dict()
    assert joined.nodes[annotation].labels == ("subject store",)
    assert set(graph.edges).issubset(joined.edges)
    assert any(
        n.content.get("metadata", {}).get("imported_snapshot") == snapshot.snapshot_id
        for n in joined.nodes.values()
    )
    assert joined.verify().ok


def test_broken_symbol_watch_retains_completed_transition_and_valid_fires(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    binding = SymbolBinding.bind(
        adapter.session,
        "limited",
        (StateAddress("value"),),
        context="panel",
        context_slots=(),
        evidence=("fixture",),
        claim="One clock only.",
    )
    adapter.symbols = SymbolTable((binding,))
    adapter.symbol_context = "panel"
    adapter.command("watch symbol:limited change")
    adapter.command("watch value change")
    with pytest.raises(DebuggerProtocolError, match="qualified binding"):
        debug.step()
    assert adapter.session.read("value") == 1
    assert len(adapter.public.watch_events) == 1
    assert project.resume("experiment", Counter()).session.read("value") == 1
    owner.close()


def assert_offline_graph(project, label):
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'mrun', 'saturn', 'common'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
from mdb import Project
project = Project(sys.argv[1])
assert project.graph().verify().ok
assert any(row['label'] == sys.argv[2] for row in project.annotations())
for session in project.sessions():
    sid = session['session_id']
    ports = [row for row in project.components(sid) if row['kind'] == 'port']
    assert ports
    assert project.component(sid, ports[0]['id'])
    assert isinstance(project.xrefs(sid, ports[0]['id']), list)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    subprocess.run(
        [sys.executable, "-c", code, str(project.root), label], check=True, env=environment
    )


def test_offline_project_graph_needs_no_model_framework(tmp_path):
    project, _, owner, debug = opened(tmp_path)
    debug.step()
    project.annotate("experiment", "offline")
    owner.close()
    assert_offline_graph(project, "offline")


def test_tensor_project_graph_is_also_framework_free(tmp_path):
    pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter

    project = Project(tmp_path / "tensor-project")
    native_adapter = DecoderAdapter.tiny("qwen3")
    with BenchSession.from_saturn_pub(
        native_adapter.session([1, 2, 3]), project=project, session_id="tensor"
    ) as owner:
        owner.debugger().step(2)
        project.annotate("tensor", "tensor carrier")
    assert_offline_graph(project, "tensor carrier")


def test_protocol_edit_after_step_tracks_actual_paused_cut(tmp_path):
    _, adapter, owner, debug = opened(tmp_path)
    shell = DebugShell(debug, json_output=True)
    assert shell.execute("step 2")
    assert shell.execute("set value 8")
    assert shell.execute("commit")
    assert adapter.session.read("value") == 8
    assert shell.execute("step")
    assert adapter.session.read("value") == 9
    owner.close()


@pytest.mark.parametrize("family", ["qwen3", "gpt2", "gemma2"])
def test_tiny_public_family_uses_same_mdb_adapter(tmp_path, family):
    pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter

    adapter = DecoderAdapter.tiny(family, seed=17)
    native_session = adapter.session([1, 2, 3])
    project = Project(tmp_path / family)
    with BenchSession.from_saturn_pub(native_session, project=project) as owner:
        debug = owner.debugger()
        parent = debug.checkpoint("parent")["cut_id"]
        debug.step(2)
        expected = native_session.read("hidden")
        debug.restore(parent)
        debug.step(2)
        import torch

        assert torch.equal(native_session.read("hidden"), expected)
        assert project.graph().verify().ok


def test_public_cli_uses_existing_shell_and_resumes_selected_cut(tmp_path, capsys):
    pytest.importorskip("torch")
    from mdb.cli import main

    script = tmp_path / "commands.mdb"
    script.write_text("info symbols\nstep 2\nbacktrace hidden\ncheckpoint retained\n")
    root = tmp_path / "project"
    args = ["public", "--tiny", "qwen3", "--project", str(root), "--json"]
    assert main([*args, "--script", str(script)]) == 0
    project = Project(root)
    selected = project.session("main")["selected"]
    assert main([*args, "--resume", "--command", "info steps"]) == 0
    assert project.session("main")["selected"] == selected
    assert main(["project", str(root), "sessions"]) == 0
    assert "main" in capsys.readouterr().out

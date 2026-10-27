"""Component navigation follows native receipts and preserves evidence scope."""

import json

import pytest
from saturn_pub import StateAddress
from saturn_pub.symbols import SymbolBinding, SymbolTable
from test_saturn_pub_adapter import native, opened

from mdb.cli import main
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import ActSpec
from mdb.project import Project


def test_list_readers_writers_and_reopen(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step(2)
    rows = project.components("experiment")
    assert {r["id"] for r in rows} == {"port:value", "port:tick", "operation:increment"}
    assert project.components("experiment", query="INCREMENT")[0]["kind"] == "operation"
    reads = project.xrefs("experiment", "value", relation="reads")
    writes = project.xrefs("experiment", "port:value", relation="writes")
    assert len(reads) == len(writes) == 2
    assert all(r["authority"] == "declared-dependency" for r in reads)
    assert all(r["causal_effect"] == "not-established" for r in reads)
    assert len({r["occurrence"] for r in reads}) == 2
    for ref in writes:
        for key in ("parent_cut", "result_cut"):
            assert project.cut_descriptor("experiment", ref[key])
        assert ref["receipt"] in {r.fingerprint for r in project.receipts("experiment")}
    assert len(project.xrefs("experiment", "operation:increment")) == 8
    owner.close()
    assert Project(project.root).xrefs("experiment", "value") == project.xrefs(
        "experiment", "value"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        project.component("experiment", "absent")
    with pytest.raises(ValueError, match="relation"):
        project.xrefs("experiment", "value", relation="causes")


def test_component_shows_watch_observations_and_annotations(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    adapter.command("watch value change")
    debug.step()
    project.annotate("experiment", "watch fired", note="diagnostic only")
    item = project.component("experiment", "port:value")
    assert len(item["observations"]) == 1
    assert item["annotations"][0]["label"] == "watch fired"
    with project._db() as db:
        row = project.session("experiment")
        info = json.loads(row["debug_info"])
        info["watch_events"][0]["cut"] = "forged"
        db.execute("UPDATE sessions SET debug_info=? WHERE id=?", (json.dumps(info), "experiment"))
    with pytest.raises(ValueError, match="watch event seal"):
        project.component("experiment", "port:value")
    owner.close()


def test_invalidations_and_unknown_footprint_remain_explicit(tmp_path):
    from saturn_pub import Session
    from saturn_pub.contracts import TransitionSpec
    from test_saturn_pub_adapter import Counter

    from mdb import SaturnPubDebuggerAdapter

    class Partial(Counter):
        def transition(self, state):
            return TransitionSpec(
                "partial",
                reads=("value",),
                writes=("tick",),
                invalidates=("value",),
                footprint="unknown",
            )

    project = Project(tmp_path / "partial")
    adapter = SaturnPubDebuggerAdapter(
        Session(Partial(), {"tick": 0, "value": 0}), project=project, session_id="partial"
    )
    adapter.step()
    refs = project.xrefs("partial", "value", relation="invalidates")
    assert len(refs) == 1 and refs[0]["footprint"] == "unknown"
    assert refs[0]["causal_effect"] == "not-established"


def test_qualified_symbols_link_to_cuts_without_granting_causal_authority(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("symbol-parent")["cut_id"]
    binding = SymbolBinding.bind(
        adapter.session,
        "value",
        (StateAddress("value"),),
        context="fixture",
        context_slots=(),
        evidence=("test",),
        claim="Supplied whole-slot name.",
    )
    adapter.symbols = SymbolTable((binding,))
    adapter.checkpoint()
    item = project.component("experiment", "symbol:value")
    assert parent in item["supporting_cuts"]
    with pytest.raises(ValueError, match="ambiguous"):
        project.component("experiment", "value")
    debug.step()
    refs = project.xrefs("experiment", "symbol:value")
    assert refs and all(r["scope"] == "whole-slot" for r in refs)
    assert (
        project.component("experiment", "symbol:value")["evidence"][0]["binding"]
        == binding.to_dict()
    )
    owner.close()


def test_shell_follow_cut_fork_edit_and_save_finding(tmp_path, capsys):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step(2)
    shell = DebugShell(debug, json_output=True)
    for line in (
        "project list",
        "project show port:value",
        "project xrefs port:value --relation writes",
    ):
        assert shell.execute(line)
    selected = project.xrefs("experiment", "port:value", relation="writes")[0]["result_cut"]
    for line in (
        f"project cut {selected}",
        f"restore {selected}",
        "fork intervention",
        "set value 8",
        "commit",
        'project annotate port:value --note "changed value on fork"',
    ):
        assert shell.execute(line)
    assert adapter.session.read("value") == 8
    assert project.component("experiment", "port:value")["annotations"][0]["label"] == "port:value"
    assert any(r["transaction"] == "apply" for r in project.xrefs("experiment", "port:value"))
    assert (
        "changed value on fork"
        in project._annotations("experiment").show(
            project.annotations("experiment")[0]["annotation_id"]
        )["note"]
    )
    assert "declared-dependency" in capsys.readouterr().out
    owner.close()


def test_cli_queries_share_python_results_and_require_session_selection(tmp_path, capsys):
    project, _, owner, debug = opened(tmp_path)
    debug.step()
    for action, target, expected in (
        ("list", None, project.components("experiment")),
        ("show", "port:value", project.component("experiment", "port:value")),
        ("xrefs", "port:value", project.xrefs("experiment", "port:value")),
    ):
        assert main(["project", str(project.root), action, *([target] if target else [])]) == 0
        assert json.loads(capsys.readouterr().out) == expected
    from mdb import SaturnPubDebuggerAdapter

    SaturnPubDebuggerAdapter(native(), project=project, session_id="second")
    with pytest.raises(ValueError, match="session-id"):
        main(["project", str(project.root), "list"])
    assert main(["project", str(project.root), "list", "--session-id", "experiment"]) == 0
    owner.close()


def test_surface_tampering_and_receipt_tampering_are_rejected(tmp_path):
    project, _, owner, debug = opened(tmp_path)
    debug.step()
    selected = project.session("experiment")["selected"]
    path = project.store.root / "cuts" / f"{selected}.json"
    data = json.loads(path.read_text())
    data["surface"]["slots"][0]["name"] = "forged"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="surface"):
        project.components("experiment")
    owner.close()


def test_refused_edit_has_no_executed_xrefs_and_abort_preserves_preview_evidence(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    parent = debug.checkpoint("parent")["cut_id"]
    from saturn_pub import Act

    with pytest.raises(ValueError):
        adapter.session.apply(Act.replace("missing", 3))
    adapter.checkpoint()
    assert not project.xrefs("experiment", "port:value")
    spec = ActSpec("value", 7, parent_cut=parent)
    adapter.preview_act(spec)
    adapter.abort_act(spec.act_id)
    refs = project.xrefs("experiment", "port:value", relation="writes")
    assert len(refs) == 1 and refs[0]["transaction"] == "apply"
    assert adapter.session.read("value") == 0
    assert project.session("experiment")["selected"] != refs[0]["result_cut"]
    owner.close()

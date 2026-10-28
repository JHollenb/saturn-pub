"""Qualified names join to exact port versions without mixing sibling futures."""

import json
from dataclasses import replace

import pytest
from saturn_pub import Act, StateAddress
from saturn_pub.contracts import TransitionSpec
from saturn_pub.symbols import SymbolBinding, SymbolTable
from test_project_analysis import pipeline
from test_saturn_pub_adapter import native, opened

from mdb import Project, SaturnPubDebuggerAdapter
from mdb.cli import main
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import ActSpec
from mdb.state_graph import GraphSnapshot


def install_symbol(adapter, name="counter", *, context_slots=("value",)):
    binding = SymbolBinding.bind(
        adapter.session,
        name,
        (StateAddress("value"),),
        context="panel",
        context_slots=context_slots,
        evidence=("fixture",),
        claim="Supplied counter, not learned semantics.",
    )
    adapter.symbols = SymbolTable((binding,))
    adapter.symbol_context = "panel"
    cut = adapter.checkpoint()["cut_id"]
    return binding, cut


def operations(result):
    return [
        node["content"]["metadata"]["operation"]
        for node in result["graph"]["nodes"]
        if node["kind"] == "action"
    ]


def test_symbol_slice_native_support_join_and_immutable_reopen(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step(2)
    binding, cut = install_symbol(adapter)
    assert binding.resolve(adapter.session, context="panel")
    result = project.slice("experiment", symbol="counter", cut_id=cut)
    assert result["support"] == "metadata-support-verified"
    assert result["symbol_binding"] == binding.to_dict()
    assert result["complete"] and not result["gaps"]
    assert operations(result) == ["increment", "increment"]
    assert len(result["writers"]) == 1 and result["writers"][0]["operation"] == "increment"
    assert not result["consumers"] and result["frontier"]
    snapshot = GraphSnapshot.from_dict(result["graph"])
    assert snapshot.verify().ok
    assert any(node.kind == "source" and "counter" in node.labels for node in snapshot.nodes)
    assert any(
        edge.graph == "knowledge" and edge.relation == "describes" for edge in snapshot.edges
    )
    assert project.slice_record("experiment", result["slice_id"]) == result
    assert project.session("experiment")["selected"] == cut
    project.finding(
        "experiment",
        "symbol:counter",
        "count observed",
        cut_id=cut,
        slice_id=result["slice_id"],
        evidence=result["evidence"]["receipts"],
    )
    assert project.findings("experiment")[0]["details"]["symbol_binding"] == binding.to_dict()
    owner.close()
    reopened = Project(project.root)
    assert reopened.slice_record("experiment", result["slice_id"]) == result
    assert reopened.slice("experiment", symbol="counter", cut_id=cut) == result
    joined = reopened.graph()
    assert joined.verify().ok
    assert any(
        n.content.get("source_ref") == "slice://sha256/" + result["slice_id"]
        for n in joined.nodes.values()
    )
    assert any(e.relation == "related_to" for e in joined.edges.values())


def test_symbol_context_clock_execution_and_ambiguity_are_not_guessed(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    binding, cut = install_symbol(adapter)
    with pytest.raises(ValueError, match="context"):
        project.slice("experiment", symbol="counter", cut_id=cut, context="wrong")
    act = ActSpec("value", 9, parent_cut=cut)
    adapter.preview_act(act)
    adapter.commit_act(act.act_id)
    edited = adapter.checkpoint()["cut_id"]
    with pytest.raises(ValueError, match="input context"):
        project.slice("experiment", symbol="counter", cut_id=edited)
    assert edited not in project.component("experiment", "symbol:counter")["supporting_cuts"]
    debug.step()
    with pytest.raises(ValueError, match="supported binding"):
        project.slice("experiment", symbol="counter")
    adapter.restore(cut)
    alternative = replace(binding, claim="Alternative supplied interpretation.")
    adapter.symbols = SymbolTable((binding, alternative))
    adapter.checkpoint()
    with pytest.raises(ValueError, match="found 2"):
        project.slice("experiment", symbol="counter", cut_id=cut)
    assert (
        project.slice(
            "experiment", symbol="counter", cut_id=cut, binding_id=binding.to_dict()["fingerprint"]
        )["symbol_binding"]
        == binding.to_dict()
    )
    owner.close()


def test_archived_bindings_survive_replacing_current_symbol_table(tmp_path):
    project, adapter, owner, _ = opened(tmp_path)
    binding, cut = install_symbol(adapter)
    adapter.symbols = SymbolTable()
    adapter.checkpoint()
    assert not json.loads(project.session("experiment")["debug_info"])["symbols"]
    assert project.symbols("experiment") == [binding.to_dict()]
    assert Project(project.root).slice("experiment", symbol="counter", cut_id=cut)["complete"]
    owner.close()


def test_slice_uses_committed_candidate_and_excludes_aborted_sibling(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step()
    parent = debug.checkpoint("parent")["cut_id"]
    rejected = ActSpec("value", 13, parent_cut=parent)
    adapter.preview_act(rejected)
    rejected_receipt = next(
        r.fingerprint for r in project.receipts("experiment") if r.to_dict()["operation"] == "apply"
    )
    adapter.abort_act(rejected.act_id)
    accepted = ActSpec("value", 7, parent_cut=parent)
    adapter.preview_act(accepted)
    adapter.commit_act(accepted.act_id)
    selected = adapter.checkpoint()["cut_id"]
    sliced = project.slice("experiment", port="value", cut_id=selected)
    assert sliced["complete"]
    assert rejected_receipt not in sliced["evidence"]["receipts"]
    assert {"increment", "replace"} == set(operations(sliced))
    assert GraphSnapshot.from_dict(sliced["graph"]).verify().ok
    owner.close()


def test_downstream_needs_explicit_destination_and_preserves_context_inputs(tmp_path):
    project, _ = pipeline(tmp_path)
    cuts = project.cut_ids("trial")
    start = next(
        cid for cid in cuts if project.cut_descriptor("trial", cid)["execution_point"]["index"] == 0
    )
    finish = project.session("trial")["selected"]
    sliced = project.slice("trial", port="a", cut_id=start, to_cut=finish)
    assert sliced["direction"] == "downstream"
    assert set(operations(sliced)) == {"first", "second"}
    assert [r["operation"] for r in sliced["consumers"]] == ["first"]
    assert sliced["complete"] and GraphSnapshot.from_dict(sliced["graph"]).verify().ok
    other = SaturnPubDebuggerAdapter(native(), project=project, session_id="other")
    with pytest.raises(ValueError, match="outside"):
        project.slice(
            "trial", port="a", cut_id=start, to_cut=other.session.capture(retain=False).fingerprint
        )


def test_downstream_does_not_cross_a_live_sibling_branch(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    initial = debug.checkpoint("initial")["cut_id"]
    original_branch = adapter.public.active
    debug.fork("left")
    debug.step()
    left = debug.checkpoint("left-end")["cut_id"]
    adapter.optional_operation("checkout", original_branch)
    debug.fork("right")
    right_parent = debug.checkpoint("right-parent")["cut_id"]
    edit = ActSpec("value", 8, parent_cut=right_parent)
    adapter.preview_act(edit)
    adapter.commit_act(edit.act_id)
    debug.step(2)
    right = debug.checkpoint("right-end")["cut_id"]
    with pytest.raises(ValueError, match="outside"):
        project.slice("experiment", port="value", cut_id=left, to_cut=right)
    sliced = project.slice("experiment", port="value", cut_id=initial, to_cut=left)
    assert right not in sliced["evidence"]["cuts"]
    assert sliced["complete"]
    owner.close()


def test_tensor_selector_symbol_slice_stays_framework_free(tmp_path):
    pytest.importorskip("torch")
    from saturn_pub.adapters.decoder import DecoderAdapter
    from test_saturn_pub_adapter import assert_offline_graph

    from mdb import BenchSession

    project = Project(tmp_path / "tensor-slice")
    with BenchSession.from_saturn_pub(
        DecoderAdapter.tiny("qwen3").session([1, 2, 3]), project=project, session_id="tensor"
    ) as owner:
        owner.debugger().step(2)
        adapter = owner.adapter
        binding = SymbolBinding.bind(
            adapter.session,
            "coordinate",
            (StateAddress("hidden", (0, 0, 0)),),
            context="fixture",
            context_slots=("tokens",),
            evidence=("fixture",),
            claim="Supplied scalar coordinate only.",
        )
        adapter.symbols = SymbolTable((binding,))
        adapter.symbol_context = "fixture"
        cut = adapter.checkpoint()["cut_id"]
        sliced = project.slice("tensor", symbol="coordinate", cut_id=cut)
        assert sliced["symbol_binding"]["locations"][0]["selector"] == [0, 0, 0]
        assert sliced["scope"] == "whole-slot; not selector-specific influence"
        project.finding(
            "tensor",
            "symbol:coordinate",
            "coordinate observed",
            cut_id=cut,
            slice_id=sliced["slice_id"],
        )
    assert_offline_graph(project, "coordinate observed")


def test_multi_transition_receipt_has_virtual_versions_not_fake_checkpoints(tmp_path):
    native_session = native()
    parent = native_session.capture()
    receipt = native_session.continue_(3)
    result = native_session.capture()
    project = Project(tmp_path / "multi")
    project.save_session("multi", native_session, (parent, result))
    sliced = project.slice("multi", port="value")
    assert len(operations(sliced)) == 3
    assert sliced["complete"]
    assert sliced["evidence"]["receipts"] == [receipt.fingerprint]
    virtual = [
        n
        for n in sliced["graph"]["nodes"]
        if n["kind"] == "output" and n["content"]["metadata"]["native_cut"] is None
    ]
    assert virtual and all(not n["content"]["metadata"]["restorable"] for n in virtual)
    assert len(project.cut_ids("multi")) == 2


def test_unknown_and_unretained_history_stays_a_frontier(tmp_path):
    session = native()
    session.continue_(1)
    selected = session.capture()
    project = Project(tmp_path / "missing")
    project.save_session("missing", session, (selected,))
    result = project.slice("missing", port="value")
    assert not result["complete"] and any(g["kind"] == "unretained-cut" for g in result["gaps"])
    project2, adapter, owner, debug = opened(tmp_path)
    adapter.session.adapter.transition = lambda state: TransitionSpec("unknown")
    debug.step()
    sliced = project2.slice("experiment", port="value")
    assert not sliced["complete"] and {g["kind"] for g in sliced["gaps"]} == {"unknown-footprint"}
    owner.close()


def test_ambiguous_writers_require_a_receipt_and_slice_seals_are_verified(tmp_path):
    session = native()
    parent = session.capture()
    first, second = session.fork(parent), session.fork(parent)
    fork = first.capture()
    first.apply(Act.replace("value", 1))
    second.apply(Act.add("value", 1))
    assert first.capture().fingerprint == second.capture().fingerprint
    project = Project(tmp_path / "ambiguous")
    project.save_session(
        "ambiguous", first, (parent, fork, first.capture()), receipts=second.receipts
    )
    ambiguous = project.slice("ambiguous", port="value")
    assert not ambiguous["complete"] and ambiguous["gaps"][0]["kind"] == "ambiguous-producer"
    result = project.slice("ambiguous", port="value", receipt_id=first.receipts[-1].fingerprint)
    assert result["complete"]
    with project._db() as db:
        db.execute("UPDATE slices SET body='{}' WHERE id=?", (result["slice_id"],))
    with pytest.raises(ValueError, match="seal"):
        project.slice_record("ambiguous", result["slice_id"])


def test_cli_and_attached_shell_share_slice_query(tmp_path, capsys):
    project, adapter, owner, debug = opened(tmp_path)
    binding, cut = install_symbol(adapter)
    shell = DebugShell(debug, json_output=True)
    assert shell.execute(f"project slice --symbol counter --cut {cut}")
    expected = json.loads(capsys.readouterr().out)
    assert main(["project", str(project.root), "slice", "--symbol", "counter", "--cut", cut]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert shell.execute(f"project slice-show {expected['slice_id']}")
    assert json.loads(capsys.readouterr().out) == expected
    owner.close()


def test_cli_can_bind_native_symbols_then_save_a_joined_slice(tmp_path, capsys):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step()
    shell = DebugShell(debug, json_output=True)
    assert shell.execute(
        'project symbol-bind counter --port value --context panel --context-slot value --claim "Supplied count"'
    )
    bound = json.loads(capsys.readouterr().out)
    assert bound["support"] == "native-support-verified"
    assert shell.execute("project slice --symbol counter --context panel")
    sliced = json.loads(capsys.readouterr().out)
    assert sliced["symbol_binding"] == bound["binding"]
    assert shell.execute(
        f'project finding symbol:counter --title "Counter observed" --slice-id {sliced["slice_id"]}'
    )
    json.loads(capsys.readouterr().out)
    assert project.findings("experiment")[0]["details"]["slice_id"] == sliced["slice_id"]
    assert not shell.execute(
        'project symbol-bind bad --port missing --context panel --claim "Invalid"'
    )
    capsys.readouterr()
    assert len(project.symbols("experiment")) == 1
    owner.close()

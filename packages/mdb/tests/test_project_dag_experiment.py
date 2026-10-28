"""The offline port-DAG prototype reports gaps rather than fabricating lineage."""

import importlib.util
from pathlib import Path

from test_saturn_pub_adapter import opened

from mdb.debugger_protocol import ActSpec


def experiment():
    path = Path(__file__).parents[1] / "examples/project-navigation/explore_dags.py"
    spec = importlib.util.spec_from_file_location("dag_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dag_projection_keeps_temporal_versions_and_branch_custody(tmp_path):
    project, adapter, owner, debug = opened(tmp_path)
    debug.step(2)
    parent = debug.checkpoint("parent")["cut_id"]
    act = ActSpec("value", 9, parent_cut=parent)
    adapter.preview_act(act)
    adapter.commit_act(act.act_id)
    summary, graph = experiment().inspect(project, "experiment")
    assert summary["component_graph"]["acyclic"] is False
    assert summary["native_graph"]["verified_dag"]
    assert summary["versioned_port_graph"]["verified_dag"]
    assert not summary["missing_receipt_parents"]
    assert not summary["versioned_port_graph"]["gaps"]
    assert summary["commit_candidate_is_graph_input"] == [True]
    assert graph.seal().verify().ok
    owner.close()


def test_port_dag_reports_unknown_footprint(tmp_path):
    from saturn_pub.contracts import TransitionSpec

    project, adapter, owner, debug = opened(tmp_path)
    adapter.session.adapter.transition = lambda state: TransitionSpec("unknown")
    adapter.session.adapter.advance = lambda state: dict(state)
    debug.step()
    summary, graph = experiment().inspect(project, "experiment")
    assert {g["kind"] for g in summary["versioned_port_graph"]["gaps"]} == {"unknown-footprint"}
    assert graph.verify().ok
    assert not graph.edges  # Equal output bytes do not explain an unknown operation.
    owner.close()

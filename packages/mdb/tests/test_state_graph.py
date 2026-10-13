import json

import pytest

from mdb.state_graph import (
    Evidence,
    GraphIntegrityError,
    NodeBinding,
    StateGraph,
)


def digest(label):
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def cut(label):
    value = digest(label)
    return f"cas://sha256/{value}", value


def test_content_addressed_nodes_and_nested_immutability():
    ref, fingerprint = cut("prompt")
    graph = StateGraph(binding=NodeBinding(model="model-1", session="session-1", version="v1"))
    with pytest.raises(ValueError, match="labels cannot authorize restore"):
        graph.add_state(cut_ref="display-only-label")
    node_id = graph.add_state(
        cut_ref=ref,
        fingerprint=fingerprint,
        metadata={"cursor": {"token": 2}},
        labels=("display-only",),
    )
    node = graph.nodes[node_id]
    assert node.node_id == node.fingerprint
    assert node.ref.endswith(node.node_id)
    assert node.content["state_ref"] == ref
    with pytest.raises(TypeError):
        node.content["metadata"]["cursor"] = {"token": 3}
    assert graph.verify().ok

    changed = StateGraph.from_snapshot(graph.seal().to_dict())
    assert changed.inspect(node_id)["node_id"] == node_id
    with pytest.raises(GraphIntegrityError):
        changed.add_source(source_ref=ref)


def test_action_supports_multiple_inputs_and_memory_only_output():
    graph = StateGraph()
    first_ref, first_fp = cut("first")
    second_ref, second_fp = cut("second")
    first = graph.add_state(cut_ref=first_ref, fingerprint=first_fp)
    second = graph.add_state(cut_ref=second_ref, fingerprint=second_fp)
    record = Evidence(
        "causally_tested",
        condition={"consumer": "native"},
        sign="positive",
        dose={"fraction": 0.25},
        effect={"target": {"delta": 1}, "collateral": {"delta": 0}},
        receipt_refs=("cas://receipt/one",),
    )
    action = graph.add_action(
        act_ref="act-memory-only",
        inputs=(first, second),
        reads=("residual",),
        writes=("kv_cache",),
        invalidates=("logits",),
        receipt_refs=("cas://receipt/one",),
        evidence=record,
    )
    assert graph.nodes[action].content["output"] is None
    reads = graph.query(relation="reads")
    assert {edge.source for edge in reads} == {first, second}
    assert all(edge.target == action for edge in reads)
    assert graph.nodes[action].evidence[0].evidence_class == "causally-tested"
    assert graph.nodes[action].evidence[0].dose["fraction"] == 0.25
    assert graph.verify().ok


def test_execution_dag_and_knowledge_relations_are_separate():
    graph = StateGraph()
    a_ref, a_fp = cut("a")
    b_ref, b_fp = cut("b")
    a = graph.add_state(cut_ref=a_ref, fingerprint=a_fp)
    b = graph.add_state(cut_ref=b_ref, fingerprint=b_fp)
    graph.add_edge(a, b, "continues", evidence="observed")
    graph.add_edge(b, a, "supports", graph="knowledge", evidence="hypothesis")
    graph.add_edge(a, b, "supports", graph="knowledge", evidence="hypothesis")
    result = graph.verify()
    assert result.ok
    assert result.execution_edge_count == 1
    assert result.knowledge_edge_count == 2
    assert len(graph.lineage(b)) == 1

    cycle = StateGraph()
    x_ref, x_fp = cut("x")
    y_ref, y_fp = cut("y")
    x = cycle.add_state(cut_ref=x_ref, fingerprint=x_fp)
    y = cycle.add_state(cut_ref=y_ref, fingerprint=y_fp)
    cycle.add_edge(x, y, "continues")
    cycle.add_edge(y, x, "continues")
    invalid = cycle.verify(raise_on_error=False)
    assert not invalid.ok
    assert any("cycle" in error for error in invalid.errors)
    with pytest.raises(GraphIntegrityError):
        cycle.seal()

    with pytest.raises(ValueError, match="unsupported knowledge relation"):
        graph.add_edge(a, b, "made_up", graph="knowledge")


def test_snapshot_event_chain_and_no_arbitrary_merge():
    graph = StateGraph(metadata={"purpose": "mechanics"})
    ref, fingerprint = cut("root")
    graph.add_state(cut_ref=ref, fingerprint=fingerprint)
    snapshot = graph.seal()
    payload = snapshot.to_dict()
    restored = StateGraph.from_snapshot(json.loads(json.dumps(payload)))
    assert restored.seal().snapshot_id == snapshot.snapshot_id
    assert restored.seal().snapshot_id == type(snapshot).from_dict(payload).snapshot_id
    assert restored.verify().ok
    with pytest.raises(GraphIntegrityError, match="branch merges"):
        restored.merge(snapshot)
    branch = restored.fork(metadata={"branch": "candidate"})
    assert branch.version == snapshot.version + 1
    assert branch.parent_snapshot == snapshot.snapshot_id
    with pytest.raises(GraphIntegrityError, match="sealed"):
        restored.add_source(source_ref=ref)

    duplicate = dict(payload)
    duplicate["nodes"] = [payload["nodes"][0], payload["nodes"][0]]
    duplicate["snapshot_id"] = type(snapshot).from_dict(payload).snapshot_id
    with pytest.raises(GraphIntegrityError, match="duplicate node"):
        type(snapshot).from_dict(duplicate)


class _Artifacts:
    def __init__(self, values):
        self.values = values

    def get(self, ref):
        return dict(self.values[ref])


class _BenchCapture:
    identity = {"model_id": "tiny", "revision": "r1"}
    model_abi = "abi-digest"
    session_id = "session-1"
    lane = "native"

    def __init__(self, first, second, receipt):
        self.cuts = {first: {"label": "root", "branch": "main", "cursor": {"token": 0}},
                     second: {"label": "act-committed", "branch": "main", "cursor": {"token": 1}}}
        self.artifacts = _Artifacts(
            {
                first: {"parent_cut": None, "receipt_ref": None},
                second: {"parent_cut": first, "receipt_ref": receipt},
            }
        )
        self.events = [{"kind": "residual-act", "act_id": "act-1", "parent": first, "receipt_ref": receipt}]

    def list_state_annotations(self):
        return [{"cut_id": next(iter(self.cuts)), "label": "prompt", "role": "source"}]


def test_ingest_bench_capture_is_read_only_and_binds_receipts():
    first, _ = cut("bench-first")
    second, _ = cut("bench-second")
    receipt = "cas://receipt/act-1"
    graph = StateGraph.from_bench(_BenchCapture(first, second, receipt))
    assert len(graph.query(kind="state")) == 2
    actions = graph.query(kind="action")
    assert len(actions) == 1
    assert actions[0].content["receipt_refs"] == (receipt,)
    assert any(edge.relation == "continues" for edge in graph.query(relation="continues"))
    assert any(edge.graph == "knowledge" for edge in graph.query(relation="annotates"))
    assert graph.verify().ok

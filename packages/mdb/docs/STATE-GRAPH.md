# MDB state graph

`mdb.state_graph` is a small public index for the execution history that a
native MDB Bench has already produced. It stores content-addressed `source`,
`state`, `action`, and `output` nodes, then links them with typed directed
edges. The graph is an evidence and lineage record; it does not load a model,
replay a StateCut, or provide a numerical backend.

```python
from mdb.state_graph import Evidence, NodeBinding, StateGraph

graph = StateGraph(
    binding=NodeBinding(model="qwen-checkpoint", session="panel-1", version="native")
)
parent = graph.add_state(
    cut_ref="cas://sha256/<cut-sha256>",
    fingerprint="<cut-sha256>",
    binding=NodeBinding(model="qwen-checkpoint", session="panel-1", clock=0),
)
child = graph.add_state(
    cut_ref="cas://sha256/<child-sha256>",
    fingerprint="<child-sha256>",
    binding=NodeBinding(model="qwen-checkpoint", session="panel-1", clock=1),
)
graph.add_edge(parent, child, "continues", evidence=Evidence("observed"))

act = graph.add_action(
    act_ref="act-id-from-bench",
    inputs=(parent,),
    output=child,
    receipt_refs=("cas://receipt/<receipt-id>",),
    evidence=Evidence(
        "causally-tested",
        sign="positive",
        dose={"fraction": 0.25},
        effect={"target": {"delta": 1}, "collateral": {"delta": 0}},
        receipt_refs=("cas://receipt/<receipt-id>",),
    ),
)

assert graph.verify().ok
snapshot = graph.seal()
snapshot.to_dict()                 # immutable, hash-bound JSON record
graph.to_mermaid()                 # inspectable execution/knowledge diagram
```

Panel adapters can use the compact aliases `g.cut(...)`, `g.act(...)`,
`g.output(...)`, `g.source(...)`, `g.consumer(...)`, and `g.edge(...)`. A
consumer is recorded as a source-side observation and cannot authorize a
state restore.

Actions may have multiple inputs and no output. This represents a memory-only
operation while retaining its declared reads, writes, invalidations, sign,
dose, conditional effect, and receipt references. A `writes` edge is added
automatically when an action declares an output.

Evidence rows retain one of four classes: `declared`, `observed`,
`causally-tested`, or `hypothesis`. The graph intentionally has no universal
importance or scalar causal score. Effects are conditional records attached to
the declared context and native receipt.

Execution relations are `parent`, `continues`, `reads`, `writes`,
`derived_from`, `invalidates`, and `observed_effect`. Structural verification
checks their content hashes, event hash chain, one-parent continuation rule,
action input/output references, and DAG closure. Knowledge relations (for
example `supports`, `contradicts`, `hypothesizes`, and `annotates`) use a
separate relation space. They can be cyclic and are never treated as
executable lineage.

To index an existing public Bench capture, use the read-only adapter:

```python
graph = StateGraph.from_bench(bench)
```

The adapter reads `identity`, `session_id`, `model_abi`, `cuts`, `events`,
durable artifact manifests, and state annotations. It carries existing Act
and consumer receipt references into action evidence. It never calls a model
step, replay, scheduler, or restore operation. A label is display metadata;
only a StateCut/CAS fingerprint can identify state.

The modest CLI operates on exported snapshots:

```bash
mdb graph verify graph.json
mdb graph inspect graph.json
mdb graph lineage graph.json <node-id> --direction ancestors
mdb graph query graph.json --kind action
mdb graph export graph.json graph.mmd --format mermaid
```

`seal()` freezes the graph object. Continue a history only through
`graph.fork()`, which records the parent snapshot and increments the version.
Arbitrary branch merges and graph-level restores are refused; native Bench,
StateCut, and Act APIs remain the authority for those operations.

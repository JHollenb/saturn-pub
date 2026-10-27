# Public sessions in MDB projects

MDB now declares a dependency on `saturn-pub`. The public adapter executes the original `saturn_pub.Session`; MDB supplies its existing debugger protocol and an investigation index. It reuses the public debugger's symbols, backtraces and Observer instead of implementing a second resolver or watch engine. Legacy MDB benches and their serialized identities remain unchanged.

Install both local distributions together from the repository root:

```sh
python -m pip install '.[ar]' './packages/mdb[autoregressive]'
```

## Caller-owned sessions

```python
from saturn_pub.adapters.decoder import DecoderAdapter
from mdb import BenchSession, Project

native = DecoderAdapter.tiny("qwen3", seed=113)  # Random CPU mechanics fixture.
project = Project("./investigation")
with BenchSession.from_saturn_pub(
    native.session([1, 2, 3]), project=project, session_id="trial"
) as owner:
    debugger = owner.debugger()
    debugger.checkpoint("parent")
    debugger.call("public.command", {"command": "info symbols"})
    debugger.call("public.command", {"command": "watch hidden change"})
    debugger.step(2)  # One public native transition at a time; watches can stop early.
    debugger.call("public.command", {"command": "backtrace hidden"})
    project.annotate("trial", "observed carrier", role="carrier", stage="exploration")
```

`owner.adapter` is a `SaturnPubDebuggerAdapter`. Its `symbols` and `symbol_context` properties accept the existing public `SymbolTable` and context. Symbol reads and watches enforce the original execution-point, input and schema support. An observation failure retains completed native transitions and valid watch fires. These descriptive bindings do not grant new write authority.

Motion, preview, commit, abort, comparison, forks and restoration use the ordinary MDB protocol. Intervention previews run on public private forks; committing invokes the original Session commit. Replay compares two one-transition suffixes from the selected cut under the same numerical program and leaves the active branch unchanged. The adapter advertises only generic `transition` boundaries; model-specific positions remain visible in the native cursor and `info steps`.

## Console

```sh
mdb public --tiny qwen3 --project ./investigation --session-id console
mdb public --tiny qwen3 --project ./investigation --session-id console --resume
mdb public --local /path/to/checkpoint --token-ids 1,2,3 --project ./real-investigation
```

The explicit local-checkpoint path does not download weights. The console supports `info symbols`, `info steps`, `resolve NAME`, `read symbol:NAME`, and `backtrace PORT`. `public watch PORT change`, `public watchpoints`, and `public unwatch ID` use the public Observer engine. A `watch symbol:NAME change` command also uses that engine. Ordinary MDB watches retain their existing predicate syntax. Use `step N`, `checkpoint NAME`, `set PORT VALUE`, `commit`, `abort`, `fork NAME`, and `restore NAME` for native state changes.

Both `--command` and `--script` accept the same MDB commands, and `--json` produces machine-readable output. Tiny decoders use random CPU FP32 models. Real-model and scientific work in the domains workspace still uses mrun admission; the Python API accepts a resident Session inside that lease.

## Persistent investigation

Public `LocalStore` owns durable cuts, typed slot pages, tensors and receipt bytes. Historical cut payloads are hydrated on demand; the control plane retains their identities and at most the latest live watch event instead of a full GPU cache for every visited boundary. Complete watch metadata remains in the project. `project.sqlite` indexes session identities, retained cuts, native receipts, symbols and watch evidence. Existing `StateAnnotationStore` sidecars describe those cuts, and `project.graph()` joins them through the existing `StateGraph` API. Project descriptions never participate in native snapshot fingerprints.

```python
project = Project("./investigation")
print(project.sessions())
print(project.annotations(query="carrier"))
graph = project.graph()
assert graph.verify().ok

# Reuse an already-produced MDB graph, including its annotation relationships.
# project.import_graph(StateGraph.from_bench(existing_bench).seal())

# Supply the same public adapter/model and numerical environment on reopen.
adapter = project.resume("trial", native)
assert adapter.session.capture(retain=False).fingerprint == project.session("trial")["selected"]
```

Resume restores the selected native cut and qualified symbol descriptions. Watch evidence survives; watches must be registered again, and named branch heads are not reconstructed. Retained cuts and receipts remain queryable. Conflicting model identities and stale project writers are refused rather than overwriting another session head. A graph import preserves original node/edge identities and indexes metadata only; it does not copy or hydrate a legacy runtime's state payloads.

Project inspection needs no model frameworks:

```sh
mdb project ./investigation sessions
mdb project ./investigation annotations --session-id trial
mdb project ./investigation graph
mdb project ./investigation list --session-id trial --query hidden
mdb project ./investigation show port:hidden --session-id trial
mdb project ./investigation xrefs port:hidden --relation writes --session-id trial
mdb project ./investigation cut CUT_ID --session-id trial
mdb project ./investigation annotate port:hidden --note 'carrier changed' --cut-id CUT_ID --session-id trial
```

Component IDs distinguish `port:NAME`, `operation:NAME`, and `symbol:NAME`.
Unprefixed names work when unambiguous. Listings cover ports at retained cuts,
recorded structural operations (with layer/phase positions in their evidence),
and qualified symbols. `show` joins supporting cuts, annotations and diagnostic
watch observations. `xrefs` reports reads, writes and invalidations from sealed
native transition/Act receipts, including private preview transactions. Restore,
abort and refused attempts do not invent executed reads or writes. Each reference
includes the receipt and transaction parent/result cuts; intermediate transition
positions are clocks, not independently retained checkpoints. A preview reference
does not imply its result was committed to the selected branch.

These are declared dependencies over recorded execution, not measured causal
effects or a complete static model map. Unknown footprint status stays visible;
missing references do not establish absence. Symbol references match whole slots,
not selector-specific dataflow, and preserve the original qualified binding.
The same Python queries are `project.components(sid)`,
`project.component(sid, name)`, and `project.xrefs(sid, name, relation="writes")`.
All metadata navigation remains framework-free, including tensor-backed projects.

The attached MDB shell uses the same queries with `project list`, `project show`,
`project xrefs`, `project cut`, and `project annotate` (no session option needed).
For example, follow a writer's `result_cut`, run `restore CUT_ID`, `fork candidate`,
`set hidden 0`, `commit`, then `project annotate port:hidden --note 'zeroed on fork'`.
Annotations reuse the existing immutable sidecar format and must refer to a
supporting retained cut. Existing `backtrace PORT` provides the public runtime's
backward dependency walk. No graphical interface is required.

This is an initial session/project integration. It adds no automatic semantic discovery, generalized decompiler, collaborative server, or new model-backed qualification.

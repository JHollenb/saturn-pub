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
```

This is an initial session/project integration. It adds no automatic semantic discovery, generalized decompiler, collaborative server, or new model-backed qualification.

# Symbols, provenance, and debugging repair

Use these tools when a tensor inspection alone cannot explain where an edit took effect,
whether later computation erased it, or which consumer changed. They use the same typed
positions, surfaces, Acts, receipts, and Observer engine as the ordinary debugger.

Run the complete offline example with `pip install -e '.[torch]'`:

```bash
python examples/debugging_repair.py
```

It writes `outputs/debugging-repair/walkthrough.md`, full trace/receipt measurements in
`report.json`, durable cuts in `store/`, and sealed debug information beside the deleted cut.
Two child processes reopen that cut and test different native suffixes. There are no downloads,
credentials, private packages, or scheduler dependencies.

## Structural information is automatic

```text
info steps
info symbols
backtrace hidden
```

`info steps` describes the **next supported transition** at the current typed position; it does
not simulate the model or promise finer pause points. Qwen, DDIM, and FLUX adapters describe their
native footprints through `Adapter.transition(state) -> TransitionSpec`. A custom adapter can
do the same. The default footprint remains `unknown` rather than inventing reads/writes.

Every native transition receipt carries a structural operation ID, occurrence ID, before/after
positions, reads, writes, invalidations, and consumer. Declared native footprints must preserve
unwritten slots; violations refuse the transition. Footprints are declarations of dependencies,
not inferred semantic causality. Occurrences identify invocation/transaction positions, not a
global ordering across unrelated sessions.

`info symbols` includes surface slot roles, producers, consumers, mutability, and readability.
`backtrace SLOT` walks recorded native and Act footprints backwards and expands token/denoise
macro-step micro-receipts. It stops at restore, unknown footprints, or unexpanded history such
as commit receipts. A cut by itself
does not reconstruct missing fork-prefix receipts. Dependencies are slot-level and can be
conservative; individual heads/rows are not automatically localized.

For a batched continuation, only the final operation has that receipt's result cut. Earlier
operations are labeled `point-only` with `cut: null`; the debugger does not invent intermediate
snapshots. Micro-receipt cut IDs are references, not a promise that their payloads were retained.

## Qualified symbols are optional debug information

```python
from saturn_pub import StateAddress
from saturn_pub.symbols import SymbolBinding, SymbolTable

binding = SymbolBinding.bind(
    session,
    "object.property",
    (StateAddress("hidden", (0, 0), role="carrier"),),
    context="my-input-v1",
    context_slots=("tokens",),
    evidence=("my-localization-report.json",),
    claim="Observed support for this input at this specific boundary.",
    status="observed",
)
debugger.symbols = SymbolTable((binding,))
debugger.symbol_context = "my-input-v1"
```

The example above is a user-supplied location, not a finding that `hidden[0, 0]` represents
an object property. Declare the real evidence and all input dependencies relevant to the claim.
Bindings check model/execution identity, context-slot content, state schema, exact execution
point, role, and selector support. Ambiguous, missing, stale, or explicitly abstained bindings
are refused. Evidence references and confidence labels are descriptions, not independent
certification. The library cannot determine whether the supplied context dependencies are complete.

Several locations can form one binding. Several bindings can describe the same symbol at
different points. A symbol watch that crosses points requires qualified support at every
observed point; unsupported resolution raises an error instead of silently watching old rows.
The completed native transition stays recorded when an observer cannot resolve the next point.

An `ObservationError` retains the exact failure cut and any valid fires from other watches.
The debugger also retains these in `cuts`, `watch_events`, and `watch_errors`. Failed registration
leaves no invalid watch installed. Watch-event serialization checks its seal and producing
receipt linkage; list/tuple and scalar type changes are observable, including transitions from
an observed `None`. A watcher with intentionally limited lifetime should be removed before its
port disappears; automatic lifetime-scoped watches are not implemented.

```text
resolve object.property
read symbol:object.property
watch symbol:object.property change
backtrace symbol:object.property
```

These reads and watches reuse the existing typed-port read validation and Observer engine.
They confer no mutation authority. Edits still use explicit physical Acts and adapter validation.
Observer baselines are branch-local and reset on restore; watch events remain evidence.

`table.save(path, cut)` writes a sealed, cut-referenced sidecar. `SymbolTable.load(path, cut)`
verifies its seal, cut reference, execution identity, and binding schemas. It does not execute
code, change cut fingerprints, or grant new writes. Cut payload loading and compatibility
validation remain the responsibility of LocalStore and Session.

## Trace a bounded temporal assay

```python
from saturn_pub import Act
from saturn_pub.paths import PathSchedule, TimedAct, Trajectory, compare_trajectories

parent = session.capture()
native = Trajectory.run(
    session, name="native", parent=parent, steps=3, ports=("hidden",),
)
candidate = Trajectory.run(
    session, name="candidate", parent=parent, steps=3, ports=("hidden",),
    schedule=PathSchedule((TimedAct(1, Act.zero("hidden")),)),
)
comparison = compare_trajectories(native, candidate)
debugger.trajectories = {"native": native, "candidate": candidate}
```

Choose a parent/horizon where every observed port is readable throughout. Each offset counts
one adapter transition, not necessarily one layer, generated token, or denoising iteration.
Offset 0 is before continuation; offset 1 is after the first native transition. At an offset,
scheduled Acts execute in declared order before that offset's observation. The trace retains
intermediate native/Act event cuts as well as the post-edit observations. Save these cuts with
LocalStore when durable storage is needed; constructing a trace alone does not write files.

`PathSchedule` is an exploratory temporal execution helper. It does not certify a measured
Program or bypass Program's restrictions on unmeasured delays. A failed arm raises
`PathExecutionError`, retaining completed points/events, the last valid cut, and refusal
receipts. The parent remains unchanged. Evaluators receive private branches; their measurements
are reported separately from state comparisons and carry no automatic terminal verdict.

```text
diff native candidate --first-divergence
```

This command compares registered `debugger.trajectories`, not arbitrary current branch heads.
Comparison requires the exact common parent, equal horizons, matching typed positions, and
cut-backed observations. It reports the first recorded selected-port difference and windows
where an observable diverged and returned to exact native content. Different causal ancestry
can produce identical values; deduplicated bytes do not merge branch history.

This comparator uses exact typed selected-port content. It does not interpret a declared
bounded-numeric contract as exact equality, infer numerical tolerances, or equate matching
states with matching behavior. Recovery of an observable is not automatically semantic repair.
The supplied repair circuit additionally checks its native threshold consumer and a collateral
channel. No source maps, inferred roles, or schedulers change the model's native consumer.

## What the repair example establishes

Two fixed linear residual writers independently reconstruct a supplied two-channel source.
The final native consumer thresholds the carrier. Weights, source, channel meanings, and
symbol support are supplied; nothing is trained or semantically discovered.

| Branch | Target contact | Collateral contact | Observation |
| --- | ---: | ---: | --- |
| Native / no-op | 1 | 0 | No-op has no selected-port divergence |
| Early deletion | 1 | 0 | Carrier diverges at offset 1 and recovers at offset 2 |
| Deletion after both writers | 0 | 0 | Suppressing the later result prevents recovery |
| Late deletion | 0 | 0 | Deletion after the final writer survives to the consumer |
| Late collateral edit | 1 | 1 | Target success and collateral change coexist |

The demonstration verifies complete-payload and final-cut equality after fresh-process suffix
replay. It establishes controlled debugging mechanics, not repair in a pretrained model.
For a real model, qualify the writable boundaries, symbols, native evaluator, numerical program,
and consumer horizon before making a mechanism claim.

## Exploration beyond the fixture

Run `python examples/explore_debugger.py` with the `ar` and `diffusion` extras. It records
three tiny random Qwen specimens, two edits per specimen, two generated tokens per arm,
DDIM timing contrasts, a supplied repeated-repair organism, and a fresh-process Qwen suffix.
Read `outputs/debugger-exploration/report.json`; source hashes are retained in that report.

The small Qwen edits changed logits while preserving both generated tokens in all three
specimens. At token commit, the temporary hidden slot was cleared while K/V remained changed;
whole-cut rewind restored exact native continuation. Matching tokens or an absent hidden slot
therefore do not establish state restoration. The supplied repeated-repair organism also
exposes multiple recovery windows and a consumer effect delayed behind a carrier edit.

This is bounded mechanics evidence on random models and supplied organisms, not a semantic
benchmark. DDIM predictions are stored in the continuation closure but currently unavailable
as readable/writable ports. Full-cut tracing is deliberate and can be expensive on large models.

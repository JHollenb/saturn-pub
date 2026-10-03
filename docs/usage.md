# Choose a capability by the job you are doing

Start with a family adapter and its Session. Normal runtime mechanics are always
part of that execution path: state validation, supported write addresses, isolated
branches, operation receipts, compatibility checks, and verified restoration. There
is no feature flag to enable them. They describe the adapter's declared closure;
they do not discover arbitrary external Python state or protect against hostile code.

The researcher explicitly chooses interventions, evaluation objectives, commit decisions,
and how much evidence to retain. Separate experimental consumers and optional packages
are not automatically enabled in every Session.

| Your task | Use | Automatic within that API | Runnable starting point |
|---|---|---|---|
| Inspect an inference failure | Session or Debugger | Native transitions, validation and receipts | [AR debugger notebook](../notebooks/01_debugger.ipynb), [CLI guide](debugging.md) |
| Stop on a live value anomaly | Observer or debugger watch commands | Per-branch baselines, microstep evaluation, retained safe-point cuts | `python examples/software_debugger.py` |
| Compare a changed carrier with native execution | Capture one parent, fork, apply an Act, continue | Branch isolation, declared writes, parent-bound commit | `python examples/ar_debug.py`, `python examples/diffusion_debug.py` |
| Keep and replay an execution state | LocalStore save/load | Content verification on full load, identity/execution compatibility on Session restore | `python examples/debugger_replay.py` |
| Package measured replay and local ancestry | ReplayBundle | Program, environment and factory sealing; opt-in data export/import | `python examples/software_debugger.py` |
| Explore doses and controls | Investigation | Common-parent arms, private evaluator branch, per-arm errors and retained measurements | `python examples/causal_panel.py` |
| Map route and effect evidence | CausalPath and PortObservation | Model/parent/clock support and cut/receipt links | `python examples/software_debugger.py` |
| Follow deletion, later repair, and collateral | SymbolTable, Trajectory, PathSchedule and Debugger | Qualified resolution, native footprints, exact common-parent alignment, retained event cuts | [Repair notebook](../notebooks/04_debugging_repair.ipynb), [guide](debug-symbols-paths.md) |
| Replay an already measured write sequence | Program | Exact parent, context, Act descriptors and continuation-budget checks | `python examples/measured_program.py` |
| Reuse unchanged declared computation | DependencyGraph | Content/version cache checks; final consumer always executes | `python examples/dependency_reuse.py` |
| Share one causal KV prefix across suffix branches | SharedQwenMemory | Shared ancestor, private tails, global-softmax segmented attention | `python examples/shared_memory.py`, [memory notebook](../notebooks/03_runtime_memory.ipynb) |
| Study modular phase transport | PhaseLaw | Integer composition under its quantized frequency law | `python examples/shared_memory.py` |
| Evaluate short reversible training futures | TrainingTransaction | Registered closure snapshots, isolated evaluation, restore on rejection or interval errors | `python examples/reversible_training.py` |
| Rewind a parameter-free structural future | TrainingStateBoundary with `optimizer=None` | Graph/rules/memory/cursor plus empty-model RNG/mode closure | `python examples/structural_rewind.py` |
| Build and debug a pretrained writer end to end | FLUX.2 writer experiment | Three-stage construction, native feedback, rollback, fresh-process replay verification | `python experiments/flux2_writer/run.py` |
| Integrate another numerical program | Implement Adapter | Session lifecycle once the adapter declares/validates its state | `python examples/custom_adapter.py`, [adapter guide](adapters.md) |
| Audit the record after a run | Evidence plane (`saturn-pub evidence`) | Hash-chained probe/claim ledgers, drift-aware verification, SQLite cross-reference index — stdlib only, no torch | `python examples/evidence_bisect.py`, [evidence guide](evidence.md) |

## Retention, durability, and rewind

Every Session operation records a receipt. Internal snapshots are temporary; execution
does not keep a full dense checkpoint after every layer. `capture()` retains a cut in
`session.history`; `capture(retain=False)` returns a sealed cut without adding it to
history. This retention choice does not disable validation or receipt recording.

Capture before a boundary you may want to rewind to. `restore(cut)` requires that exact
compatible cut. Save it with `LocalStore.save(cut)` when it must survive process exit.
The cut includes execution tensors and weight identity, not model weights. Reconstruct
the same weights and execution contract separately before loading and restoring.
Metadata-only `saturn-pub inspect` does not load or verify the tensor blob; full load does.

`Session.replay(cut, steps=N)` returns an advanced child and leaves the selected Session at its
current cursor. LocalStore uses content-addressed slot pages and tensor blobs, so saving an
identical cut again reuses the same identities. ReplayBundle exports only local ancestors that
are actually present; it does not invent a complete history.

Clear unneeded history and release branches when investigating large models. Branches
retain state in memory even after an abort decision. No eviction/paging policy is inferred.

## Alternative model continuations

A branch tree is a collection of branches from captured model states. Capture a
common parent, keep a native continuation, try interventions on other branches, and
fork again at any captured intermediate state. Each branch has its own mutable
execution state and receipts. The researcher chooses what to inspect and measure.
These mechanics are available through Session and the debugger; no feature flag is needed.

For example, start the tiny Qwen debugger as described in the [CLI guide](debugging.md),
then run:

```text
step
step
capture parent
fork native parent
use native
continue 6
capture native_head
use root
fork candidate parent
use candidate
zero hidden
capture edited
continue 6
capture candidate_head
compare native
fork another_future edited
use another_future
step
inspect
use root
replay parent 6 replayed_native
```

The first two steps pause after embedding and the first decoder layer. The native
and edited branches then run the same six native operations from that parent.
`another_future` branches from the edited intermediate cut; `replayed_native` reruns
the original parent. The root remains available for further experiments. Use watches,
symbols, and trajectory comparisons to find divergence, later repair, and collateral
inside these futures, as shown in the [repair guide](debug-symbols-paths.md).

Save selected cuts with `LocalStore` or the debugger's `save` command to revisit them
in another process. Save their intermediate ancestors too when you want a durable
branch graph. A ReplayBundle packages a measured Program, named branch heads, the
available local ancestry, and provenance. See `examples/debugger_replay.py` for
fresh-process replay and `examples/software_debugger.py` for bundle export.

“Multiple versions” here means alternative execution states under the same model
weights and execution contract. Different trained checkpoints are separate model
identities: reconstruct each model separately and compare their observations. A cut
does not carry weights or migrate state between incompatible models. For reversible
parameter updates, use [TrainingTransaction](training.md) with its registered training
closure and explicitly retain the parameter snapshots you need.

The public library supplies these branching building blocks. The private branch-search
controller, automatic branch scheduling, and pruning policies
are outside this release; you can implement your own exploration policy over Sessions.

## Measured programs

Use a Program when repeating exactly the pre-continuation writes already measured by
an Investigation. You supply the named context and ordered Acts. The API validates
that these match the measured arm, exact parent, model identity, and execution contract.
The example also shows wrong-context and different-parent refusal.

This v1 program cannot automatically transfer to a new prompt or checkpoint, schedule
nonzero-delay writes, or serialize arbitrary Python Act closures. Re-measure the desired
context or build a separately validated compiler. A completed exploratory arm is evidence
for structural replay support, not a general semantic capability certificate.

## Dependency reuse

Use a DependencyGraph when your pipeline exposes a known acyclic dependency structure.
You declare cells, input dependencies, and implementation versions. Reuse happens
automatically within that graph. The example caches native Qwen normalization, changes
an independent label, then changes the hidden carrier, while running native readout every time.

The graph is separate from the model adapters; constructing a Session does not create
one. Include immutable model identity in a cell version when its implementation uses
resident weights. Declare every changing input and change the version when code semantics
change. Hidden mutable closures are not tracked. Scheduler feedback is a changed input
at the next trajectory step. Measure end-to-end costs before claiming a speedup.

## Shared memory and phase transport

Use SharedQwenMemory for batch-one, one-token suffix execution from one real Qwen prefix,
when studying memory sharing or many branches over that same ancestry. It is a separate
experimental consumer; ordinary Qwen Session forks still copy their KV state. The caller
chooses the shared prefix and private-tail capacity, and must account for all retained branches.

The portable segmented attention consumer changes reduction order, so inspect the reported
native-logit error. It still attends to all selected rows and obeys the checkpoint's context
limit. It has no durable shared-cache StateCut format, page faults, CUDA virtual-address
aliasing, or built-in serving scheduler. Do not substitute it silently for a native numerical
contract. PhaseLaw is a separate reference; it is not wired into this Qwen cache.

## Reversible training

Use TrainingTransaction around a separate trainable component, not an in-place mutation
of the resident inference adapter's model. Supply a baseline evaluator, a promotion policy,
a restorable sampler/cursor, and callbacks for additional mutable caches or hook state.
After registration, snapshot/evaluation/restore mechanics run automatically for each bounded interval.

The example shows beneficial and harmful candidate updates, registered cache/cursor state,
and exact rejection rollback. Promotion selects the next reversible branch; a rejected
branch's scores remain research evidence. Receipts do not automatically retain rejected
parameter tensors. External files/API effects and unregistered state are outside rollback.
See [training closure and feedback boundaries](training.md).

## Audit the record: the evidence plane

`saturn_pub.evidence` is a stdlib-only subpackage (no torch, no imports from the
rest of the toolkit) that treats the experimental record as a queryable,
self-auditing object. It reaches a live run only through caller callbacks, so the
same four verbs serve token cuts, layer cuts, denoising steps, and training
intervals. Use `bisect` to search backward to the first cut where two retained
branches diverge (probed in logarithmic replays, with probes-used reported
against a linear scan); `claims` to hash-chain a claim to the evidence receipts
it rests on and auto-stale it when those bytes drift; `xref` to index every
address receipts cite into SQLite; and `cite` to bind model and input to an
address at write time. All four share one canonical-JSON + sha256 custody idiom
with tamper-evident re-verification, and `verify --check` keeps CI from mutating
the ledger it judges.

```bash
saturn-pub evidence bisect verify receipt.json
saturn-pub evidence claims verify --check --registry claims.jsonl
saturn-pub evidence xref query ar://decode/step/1/site/hidden --db xref.sqlite
```

The example records a native and a perturbed branch of a tiny offline model,
bisects to the first divergent cut, registers a claim citing the receipt, and
verifies it. See [the evidence plane](evidence.md).

## Installation and runners

From the public checkout, `pip install -e '.[ar,diffusion,train]'` covers the model examples.
Add `interop` for SAELens / TransformerLens, `notebooks` for Jupyter, and `dev` for
tests/build tools. The core and JSON-only custom
adapter require no model frameworks. Tiny examples are random native CPU mechanics and
need no downloads; pretrained semantic behavior is shown separately in the writer experiment.

Runner choices, `--local-files-only`, debugger family, recipe, output path, and dry-run
flags control placement, acquisition, input/output, or inspection. They do not enable
runtime correctness. The scripts need no job scheduler; any runner can execute the
same commands and persist the ordinary result directories.

# Upgrading MDB for faster program discovery

Date: 2026-10-29. Status: the first prepared-parent and sealed-panel slice is implemented and qualified through installed wheels. The later hypothesis, batching, storage and automatic evidence integrations remain proposed.

The [prepared-panel API](PREPARED-PANELS.md) reuses an owned, root-produced live frontier for exact scalar continuations. On twelve retained R3 Mamba patches, warm branch calls took 0.132 seconds versus 0.692 seconds from roots: 5.26× marginal, or 2.00× including 0.214 seconds of preparation and sealing. Every selected stage and all eleven heads matched, with twelve independent full-model audits and eight own-state successor joins. These timings cover scalar replay APIs, not full-job latency or physical batching. A subsequent four-patch discovery panel was 5.24× faster marginally but 0.91× including preparation, so preparation should remain explicit for small panels. See the [qualification](../../../../isa/outputs/prepared-forward-qualification-r3-mamba/README.md) and [new bit-9 experiment](../../../../isa/outputs/prepared-forward-row9-r3-mamba/README.md).

The objective is to reduce the cost of asking and answering a causal question. MDB should accept a retained parent, competing executable explanations, a contrast, a consumer and a budget, then produce sealed native branches, measured effects and an updated evidence map. The first implementation should reuse existing public contracts and target the current CPU FP32 native-forward lane.

The [October30 ISA debugger design](ISA-DISCOVERY-DEBUGGER.md) specifies the
next interactive interface: exact ISA inputs and instruction context, generic
structural names, MLP entry/exit and residual-update breakpoints, explicit
primitive/math/law/update/forward/instruction units, command actions and resident
lanes. Generic IDs and user aliases should work immediately; semantic ISA names
are optional annotations and must not block route logging or debugger use.
The design distinguishes existing scalar replay/scripts from proposed live
ISA sessions, named component boundaries and physical batching.
The [full interface specification](ISA-DEBUGGER-INTERFACE.md) now fixes command
targeting, revisions/retries, partial motion, atomic instruction commit, stop
and action ordering, lane/group pauses and durable-versus-live restore. Its
example plan explicitly uses resident scalar lanes. Implementation remains
deferred while the interface is reviewed.

## Current costs and reusable components

The reference [ATen branch implementation](../src/mdb/forward_interventions.py) creates a new stepper from roots for every candidate. It executes the prefix through the first edited producer, then the remainder of the recorded graph. The new `PreparedForwardParent` offers explicit same-parent prefix reuse. `branch_forward_many` still means multiple replacement sites within one branch; it does not fuse a panel of candidates into a physical model batch.

The [recorder and stepper](../src/mdb/forward_recording.py) retain snapshots and descriptors for every observed tensor. A descriptor copies bytes to contiguous CPU storage and hashes them. Each replay step constructs output descriptors; root validation and cloning also occur for each new machine. This provides a strong reference instrument, while exposing repeated work in same-parent late-edit panels.

Existing [incremental planning](../src/mdb/circuit_incremental.py), [cell execution](../src/mdb/circuit_runtime.py), [resource residency](../src/mdb/circuit_residency.py), [VM cache metadata](../src/mdb/vm_cache.py) and [catalog](../src/mdb/circuit_explorer.py) already provide relevant pieces. They have distinct contracts; the ATen experiment path does not acquire their benefits merely because they exist. The first engineering work is an explicitly qualified connection between these surfaces.

Recent [native input](../../../../isa/docs/native-input-computation-roles.md) jobs measured scientific worker phases of 6.49 and 7.84 seconds. Their scheduler start-to-finish durations were approximately 24.46 and 20.73 seconds. The difference contains work outside the measured assay phase; profile startup, imports, model acquisition, installation and publication separately before assigning a cause. Experiment authoring also required separate plan, assay, audit and naming scripts.

## First upgrade: prepare one parent and reuse its valid execution

`PreparedForwardParent` now verifies the trace and roots, executes the shared native prefix once, and seals the live frontier at an explicit boundary. Each candidate forks that produced state and executes the native continuation. It preserves alias/view geometry, guards the owned frontier, rejects seeded RNG replay without closure, and emits a verified StateCut-bound reuse receipt. This is a live in-process owner; its descriptor manifest is not a durable rehydration payload. Connection to the separate incremental planner and VM cache remains future work.

The parent identity must bind checkpoint and runtime parameters, exact inputs, trace, operator/numerical geometry, cut position, closure and captured bytes. A cut contains all live state needed by the continuation. For the current ATen lane, mutable and random operations remain outside the existing supported contract. Other adapters must explicitly account for aliases, KV, recurrence, RNG, routing and external effects when applicable.

The frontier must be produced by execution from the declared roots, with ancestry retained. Arbitrary recorded intermediates remain observations rather than admitted continuation roots. Changing an upstream input, weight, route or numerical contract invalidates dependent cached state. Same-parent candidate reuse and cross-input reuse are different requests.

Move graph verification, operator resolution, def/use indexing and compatible identity checks into preparation. Use immutable ownership or guarded versions before amortizing byte verification; do not simply remove checks from mutable caller-owned tensors. Retain per-branch Act identities, actual installed bytes, consumer outputs and explicit verification coverage.

## Second upgrade: describe an experiment once

`ExperimentPanel` now snapshots and seals a finite declarative set of replacements, doses, source identities and metadata before executing sequential scalar branches. It returns per-row outcomes or errors. The qualified ISA adapter independently audits native outputs, records successors, and imports outcomes into the existing public catalog. Automatic compilation of controls, consumers, continuation and catalog plans into the generic panel remains proposed.

Initial templates should cover input squares, reciprocal interchange, dose response, source/block/rescue, local tangent substitution and same-answer/different-future contrasts. Templates generate reviewable plans; controls remain appropriate to the question rather than a mandatory universal checklist. Discovery observations do not require a terminal scientific gate.

The panel should seal all nominated predictions and payloads before candidate outcomes, retain incomplete branches independently, and automatically run the direct native audit required by the selected execution claim. Own-state successors must consume literal branch-predicted state, with external expected answers confined to evaluation.

Use public Frame/Act/Receipt and CircuitCatalog contracts. Keep backend execution explicitly registered. A serialized experiment specification must not import arbitrary callbacks. Declare optional framework dependencies and preserve framework-free metadata inspection and installed-wheel qualification.

## Third upgrade: retain hypotheses and choose informative questions

Add a versioned `ComputationHypothesis` above physical symbols and measured outcomes. It records an executable law, proposed variables, applicability, native bindings, permitted resolver inputs, predictions, calibration/query separation, counterexamples and consumer horizon. Human names are separately versioned annotations. Physical addresses retain their existing anonymous identities.

Start with registered predictors for the finite laws already used in research: Boolean transition tables, finite polynomial responses, source equations and routing rules. Preserve uncertainty and competing explanations when the diagnostic panel cannot distinguish them.

An experiment selector should rank candidate inputs, sites and futures by disagreement between hypotheses, relevance to the unresolved mechanism and estimated cost. This is a search heuristic, not an authority for deciding whether an effect exists. Continuous trends, native errors and rejected explanations remain evidence. Failed or incomplete instruments cannot supply a scientific null.

The planner should reuse compatible prior observations before scheduling a native call. Bind reuse to exact input, checkpoint, consumer, numerical contract and declared observation coverage. A record of one output field does not answer a request for unobserved fields or a longer future.

## Fourth upgrade: batch compatible work and reduce retained bytes

After shared-parent reuse is qualified, group compatible branches by adapter, checkpoint, cut, shape, dtype, consumer and horizon. Physical batching needs an adapter with explicit batch semantics and isolated per-branch state. It is not achieved by calling the current replacement helper several times.

Use batched or approximate results as explicitly labelled exploratory measurements until the desired scalar relation is established. Retain scalar native references and discriminating controls; exact claims require the corresponding numerical qualification. Preserve batch-dependent disagreement and counterexamples rather than silently treating a batched row as scalar-equivalent.

Add an `ObservationSpec` for selected rows, writes, head coordinates, summaries, derivatives and full tensors. Preserve enough live state for exact reopening; separate that requirement from display/analysis retention. Store shared tensors and weights once in content-addressed binary storage, with branch receipts referencing them. Avoid repeating full arrays in JSON and tensor packs when one source plus a reference is sufficient.

Keep an explicit full-audit mode and metadata describing which boundaries were verified. A lighter observation mode must not report unperformed per-operator equality checks. Reuse the existing cache store and evidence references rather than creating another unrelated persistence layer.

Within mrun, submit a finite resident experiment program under one admitted lease and amortize setup across its bounded panels. Cross-job resident execution is a later scheduler/runtime integration with its own admission and invalidation contract; the existing single-lease experiments do not establish that capability.

## Fifth upgrade: make every outcome queryable

Ingest public catalog and Frame/Act receipts directly into the evidence plane without manufacturing a gates dictionary. The latest ISA sweeps have target-streams-only coverage and zero automatic receipt ingestion despite direct typed audits. Fixing this reduces repeated custom audits and makes existing experiments available to the selector.

Retain exact state equality, approximate numerical agreement, decision agreement and future-behavior agreement as separate relations. The latest affine substitute preserved register/flag decisions with roughly 96% relative post-vector error; a search or planner cannot treat those relations as interchangeable. Each relation binds a reader, domain and horizon.

Derived catalog views should display measurement summaries from actual outcomes without rewriting immutable circuit records. Expose unresolved predictions, counterexamples, causal edges, cached-state validity and actual verification coverage in the same view.

## Implementation order and benchmark

1. Completed: qualify `PreparedForwardParent` on the existing native-square/tangent fixtures, including frontier ownership and alias/view integrity, against root replay and direct full native execution.
2. Partly completed: the fixtures and new bit-9 square use sealed `ExperimentPanel` candidates. Consumer auditing, successor plumbing and catalog import still live in the explicit ISA adapter.
3. Import their role hypotheses and the response-quotient exceptions into the hypothesis store. Demonstrate selection of an input or continuation on which two explanations disagree.
4. Add adapter-qualified branch batching, observation selection and larger resident panels where measurements show a benefit.
5. Connect evidence ingestion and derived views so a completed panel automatically updates the discoverable map.

The first benchmark must use the same checkpoint, inputs, patches, selected stages, all native output heads and own-state successor inputs as the retained campaign. Keep exact scalar parity as the engineering requirement for the new exact lane. Record rejected cache reuse and any batch-dependent numerical differences separately.

Measure end-to-end latency; preparation, capture, replay, hashing and serialization time; physical native calls and executed operators; prefix reuse; peak memory; stored bytes; cache hit/invalidation rates; and how much experiment-specific code remains. Scientific discovery progress should also track new prospective questions answered and counterexamples found. Operation-count savings alone do not establish a wall-time speedup.

The corresponding existing requests are `resident-replay-batch-fusion`, `hyperion-native-panel-and-act-lowering`, `mdb-computation-hypotheses-and-causal-naming`, `evidence-observation-frame-act-ingest` and `state-program-jobs-cross-job-coalescing` in [the Saturn request ledger](../../../../saturn/request.jsonl). This roadmap connects those requests to a small, measurable implementation sequence.

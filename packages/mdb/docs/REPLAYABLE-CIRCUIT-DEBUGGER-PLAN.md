# Replayable circuit navigation in MDB

The [future native circuit capabilities](NATIVE-CIRCUIT-CAPABILITIES.md) extend
this plan with original read-plane inspection, interaction panels, interface
sufficiency and consumer-linked mathematics. See the [workflow proposal](NATIVE-CIRCUIT-WORKFLOW.md)
for console-first integration and the [mathematical reference](NATIVE-CIRCUIT-MATH.md)
for proofs and boundaries. A [bounded console/API implementation](NATIVE-CIRCUIT-INSPECTION.md)
now connects focus, hydration, native inspection and branch comparisons with tiny
CPU coverage; broader semantic and pretrained-model qualification remains planned.

Status: implementation plan, November 12, 2026. Native execution, qualified logical references and replay are the foundation. Understanding computation is the objective; model acceleration and fitted predictors are outside the core workflow. No learned dictionary, semantic label, task grammar or predictor fit is required to open an investigation.

Update: the first root-based regeneration path is now implemented in `mdb.native_replay_recipe` and qualified on historical cached BF16/eager CUDA Qwen. A 28,066-byte input/Act recipe regenerates 100 declared boundary occurrences (84 unique cuts), matching 92 native receipts, 140 attention recaptures and 24 full logits arrays, with 24 own-token commits. No historical activation or StateCut payload is supplied or loaded. Generic graph-to-live seek, automatic indexing, arbitrary mutable closure and cache eviction remain planned.

The recurring scaffold is the reusable organization of writers, persistent registers, readers and consumers. A dynamic circuit invocation is that organization operating on particular arguments, addresses and values at a particular moment.

## Existing foundations and actual gaps

| Foundation | Existing implementation | Gap this plan closes |
| --- | --- | --- |
| Native operation identities and def/use | `forward_recording`, `ForwardSymbolTable`, `ForwardReadableProgram`, native operation receipts | Unified navigation from prompt rows through native Session execution; generic tensor replay currently has a bounded CPU-root contract |
| State and branches | Public native Session, `SaturnPubDebuggerAdapter`, StateCuts, typed Acts, receipts | Metadata navigation target must resolve into an explicitly compatible live Session; passive observation cuts cannot be resumed |
| Symbol references | `SymbolBinding`, lifecycle definitions/uses, `Project.symbols_at`, `resolve_symbol_address` | Anonymous mechanical nodes should register automatically, with semantic names as optional overlays |
| Attention observations | `AttentionObserver`, `symbol_attention`, query-head to KV-group associations | Generic reader discovery and lazy recapture; existing integration requires retained Q/K/V/A arrays and is not metadata-only |
| Parameter references | `parameter_artifacts` manifest with checkpoint hash, tensor name, shape, dtype and byte range | Open original named tensors directly when analyzing; avoid materializing a second NPY parameter bank just to inspect one selector |
| Historical circuit evidence | Source/writer/V restoration, coalition panels, executed attention resolver | Join observational edges and bounded causal findings without assigning causal authority to every graph edge |

The resolver worker currently computes an address, applies a Session Act and persists source/read/output records. It does not dispatch a generic edit by resolving a symbol name. Its HTML is offline inspection. This plan adds the missing navigation/materialization contract instead of describing it as already delivered.

## Implemented bounded recipe API

`NativeReplay(recipe, adapter).run()` validates model/numerical identity and reconstructs an ordered native history from token roots, prompt-token ingress, native continuation, restoration of an earlier regenerated occurrence and supported cache-value copies. `materialize(occurrence, StateAddress(...))` reads a regenerated port through a compatible public Session. The original model is supplied by the caller; no learned predictor is used. The module opens no artifact files or persistent tensor exports. In-memory cuts currently remain retained for the duration of the replay.

Expected hashes are comparison metadata checked after execution. They do not install state. Recipes are hash sealed, bounded and reject unexpected fields and references to future occurrences. This is trusted public Python execution, not an adversarial sandbox. The first profile is the original `PromptQwenAdapter`; random and training state, other families, arbitrary operators and automatic synthesis from all possible histories are not qualified.

Source: [module](../src/mdb/native_replay_recipe.py), [unit checks](../tests/test_native_replay_recipe.py), [metadata compiler](../qualification/native_replay_prepare.py), [native worker](../qualification/native_replay_worker.py). Qualification artifact: `artifacts/mdb-native-replay-hydration-20261112/`. Its saved-only audit joins all regenerated hashes back to immutable historical custody. Three local recipe checks and eight related resolver/coalition checks pass.

## Reference and storage contract

Use durable logical references, not process memory addresses. A reference binds model checkpoint identity, numerical program and state schema, input/history identity, session/branch, typed execution clock, operation occurrence, port and selector. A filesystem or object-store locator describes where an object can be found; its digest establishes identity. Moving a file should not change the referenced computation.

Two reference classes are necessary:

- A parameter reference identifies the immutable checkpoint file hash, tensor name, native dtype/shape, tensor byte range and requested selector. The canonical checkpoint remains shared. Tensor views are created in memory on demand, without per-investigation parameter export.
- A native value reference identifies a producer occurrence or persisted native cut and its port/selector, plus either a retained CAS payload or an authenticated regeneration recipe. A content hash alone cannot regenerate a missing value. Unsupported or incomplete recipes remain explicit unresolved frontiers.

A regeneration recipe contains an available replay anchor, exact token history including branch-owned commits, ordered prompt ingress and Act dependencies, RNG/state dependencies where applicable, numerical profile, target occurrence and comparison hashes. Donor transplants reference donor cut/port/selectors and require a cycle-free recipe graph. They do not silently become literal future activations. Resolve the reference into the Session's native state or transient observation, then release temporary tensors when analysis finishes.

Persist the graph index, token offsets, parameter references, replay recipes, receipts, compact measurements and provenance. Store full activations or extra checkpoints only through an explicit optional retention policy. Replaying from a root may require reconstructing its whole prefix and native cache. Exact replay claims are profile-specific; an incompatible backend, missing mutable closure or unknown dependency produces an unresolved target rather than fabricated bytes. A derived cache is disposable only after its regeneration path has been qualified. Historical audited artifacts remain immutable during this transition.

## Graph contract

Start with native coordinates: prompt token occurrences, original operations, module ports, K/V writes, attention readers, residual/MLP writes, logits and commits. An anonymous ID such as a layer/head/receiver occurrence is enough. Human labels never determine whether a node is navigable.

Keep edge types separate: executed producer/consumer; declared whole-slot dependency; observed attention read; conditional numerical attribution; measured intervention effect; unexpanded or unknown dependency. Attention mass and large norms rank inspection candidates; neither certifies influence. Whole-slot provenance must not imply coordinate-specific causation.

Attention inspection should expand Q/K source formation, normalization and RoPE, masked scores, softmax, per-source weighted V, grouped-query mapping, head merge and output projection. MLP inspection should expand the actual norm, gate/up maps, activation/product and down projection. Retain the original operator order, casts and reduction policy. `sum_j A[j] V[j]` explains a fixed read; it does not explain why that attention formed or the downstream effect after nonlinear computation.

Group repeated native structures into expandable views while keeping every member address available. Rank or prune the display without deleting excluded routes, inhibitory paths or unresolved contributions. Cross-prompt alignment first uses mechanical structure and clock relationships; a proposed recurring role remains a separately evidenced hypothesis. The graph is a view of recorded computation, not a unique universal circuit count.

## Intended debugger interaction

The following are proposed interactions, not shipped CLI commands:

1. Run an arbitrary prompt on one qualified native Qwen execution profile. Retain its small replay manifest and graph metadata, with no predictor fitting and no semantic source annotation.
2. Select an output token or logit. Expand its upstream native writers and read edges, with a visible frontier where fine-grained provenance is unavailable.
3. Select an attention reader. Inspect the source token offsets and producer links for Q/K/V, the actual allocation, weighted return and projection. Follow a K/V entry back to the source occurrence that formed it.
4. Seek to a node. Restore a compatible anchor and replay to its target boundary. For a port that is currently observer-only, replay with a verified tap; do not claim that it is a resumable writable StateCut.
5. Set a breakpoint or watch on a qualified operation/port. Inspect actual native operands, checkpoint parameter selectors and equations in the existing MDB inspection surfaces.
6. Fork a supported typed intervention and continue the native consumer. Keep original and altered states linked to the same parent. Separate zero contribution with fixed other weights from masking that renormalizes attention.
7. Save only the resulting metadata, edit recipe and small observations under the declared retention policy. Later navigation recreates needed values through replay.

An inserted note in the temperature prompt supplies the first useful investigation: follow the L20/H5 read moving from the description to the later `Temperature` field; inspect which Q/K changes caused the movement; trace both values' formation; compare original-source versus later-field edits through the native output. The method should expose this change without requiring the operator to call the nodes temperature circuits.

## Implementation sequence

1. **One qualified reference-to-replay path.** Define and test a small reference/resolution API over the existing Project and Session contracts. Support retained-cut locators first; explicitly distinguish them from verified regeneration recipes. Seek one existing Qwen source/read/output chain and compare native hashes. Make named-parameter selectors resolve to the original archive without a second weight bank.
2. **Automatic native index.** Register anonymous executed nodes and prompt coordinates during normal execution. Build bounded iterative def/use slices and lazy observed attention links. Preserve the current parameter retrieval and numerical identities. Integrate a framework-free metadata listing before a new viewer or classifier.
3. **Native reader expansion.** On replay, expose Q/K/V, score/softmax/return, merge and MLP ports. Qualify each inspection tap and writable cut independently. Begin with the existing cached BF16 Qwen lane; do not silently reuse passive FP32 analysis records as resumable cuts.
4. **Debugger navigation.** Connect selected graph/symbol entries to seek, operands, writers, readers, equations and supported branch operations. A separate graph is not completion: prove the selected ID resolves to the same live computation and opens through MDB.
5. **Metadata-first retention.** On a new run, qualify regeneration from one root or sparse anchors, then retain metadata/recipes by default and use bounded temporary materialization. Report which objects still require persistent payloads. Leave old inventories untouched.
6. **Conceptual overlays when useful.** Import pretrained dictionaries or learned feature catalogs as optional coordinate transforms bound to exact weights and input state. Retain native sources and unexplained remainder. Fitting a new explainer or predictor is never a prerequisite to native debugging.

## Validation and exploratory use

Mechanics checks: parameter identity and offsets; input/branch/clock resolution; stale-reference refusal; missing-recipe visibility; observer parity; target hashes under the declared replay profile; typed edit footprint; native continuation and own-token commit; exact restore; read-only metadata queries; and proof that default export contains no tensor payload or duplicated weights. Record unsupported operations per frontier rather than suppressing the whole investigation.

Scientific observations: how reads change with input; source-to-reader contribution; native output changes; cooperating and opposing writes; scope of recurring structure; and disagreements between attention, local attribution and interventions. No semantic label, response threshold, reconstruction score or top-k cutoff blocks exploration. Keep mechanical validity separate from interpretation and terminal claims.

Use the temperature anchor and inserted-note counterexample to qualify navigation, then inspect a new ordinary code/prose prompt without a task parser or fitted predictor. A useful first result is a faithful, replayable explanation of selected computations tied to prompt positions; a universal semantic ontology is not required.

## Evidence and review record

Workspace evidence: `artifacts/mdb-replay-circuit-plan-20261112/`. The saved-only sample contains six source/output nodes and 36 observed reader edges, references original Project cuts and symbols, and exports zero tensor payload bytes. It demonstrates a small metadata index, not independent metadata-only reconstruction or a newly shipped runtime. Its storage audit verifies all 3,095 resolver inventory files and finds zero copied pretrained parameter payloads. The existing native result tree, including its inventory file, is 266,676,931 bytes; most space is retained native state and observations, not pretrained weights.

Related implementations: [forward symbols](FORWARD-SYMBOLS.md), [symbol-linked attention](ATTENTION-SYMBOL-LINKS.md), [native resolver worker](../qualification/resolver_execution_worker.py), [Project](../src/mdb/project.py), [parameter references](../src/mdb/parameter_artifacts.py). Automatic sparse discovery remains a separate prototype in the `mdb-semantic-discovery-20261112` worktree and a source of optional candidates.

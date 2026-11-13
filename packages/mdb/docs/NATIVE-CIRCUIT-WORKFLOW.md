# Integrating circuit understanding into the MDB workflow

Status: design proposal, November 12, 2026. Based on consolidated `origin/main` at `b93b9f4`. All new interactions below are proposed, not executable commands. This change documents capabilities only; it does not implement them or qualify additional model profiles.

Start with the [four future capabilities](NATIVE-CIRCUIT-CAPABILITIES.md) and their [mathematical basis](NATIVE-CIRCUIT-MATH.md). This document specifies how a user reaches them through the existing debugger.

## The user's task

A typical investigation starts with an output: “Why did it say cold?”, “Where did this copied value come from?”, or “Why did this code completion use the wrong variable?” The user should not need to choose Fourier analysis, a sparse dictionary, a neuron index or a fitting dataset before inspecting the computation.

The normal loop should be:

```text
run or reopen → select output/symbol → inspect explanation
            → seek to occurrence → preview an edit
            → compare native futures → save finding
```

Use the same Project, qualified references, branch history and consumers throughout. Expose mathematical detail progressively. Preserve the full computation through references and replay even when the display groups most of it.

MDB is currently a console/API debugger with offline views. Ship the complete path in the existing console first. A graph, notebook or HTML viewer should select the same IDs and consume the same records, rather than introduce a second runtime or a separate graph database.

## What to reuse

| Existing surface | Role in the proposed workflow |
| --- | --- |
| [Local console](LOCAL-DEBUGGER.md), [public session adapter](SATURN-PUB-PROJECTS.md) | Open/resume, bounded execution, watches, checkpoint, fork, preview/commit/abort |
| `Project.components`, `component`, `trace`, `backtrace` | Find an output or operation and navigate existing declared provenance |
| [Qualified symbol slices](SYMBOL-SLICES.md) | Bind the exact source cut, symbol/selector, receipt ancestry and bounded destination |
| [NativeReplay](../src/mdb/native_replay_recipe.py) | Regenerate supported histories and materialize compatible public addresses |
| [Versioned Project analyzers](../src/mdb/project_analysis.py) | Persist immutable identity-bound analysis results and reuse them when inputs match |
| [CircuitCatalog](CIRCUIT-EXPLORER.md), [computation protocol](../src/mdb/computation_protocol.py) | Seal supported branch plans, execute through registered public backends, retain measured outcomes |
| [Computation atlas](COMPUTATION-ATLAS.md), Project findings | Retain comparisons, applicability, counterexamples and links to original evidence |
| [Attention observer](../src/mdb/attention_observer.py), [symbol attention](ATTENTION-SYMBOL-LINKS.md) | Reuse original attention capture and joins instead of another Q/K/V hook implementation |
| [Parameter artifact metadata](../src/mdb/parameter_artifacts.py) | Reuse checkpoint/tensor identity; add selected original-archive views without a copied parameter bank |

These are reusable seams, not a claim that all combinations already work. Generic graph-to-live seek remains missing. Existing slices cover declared whole slots, not measured coordinate influence. Existing fitted Jacobian lenses are distinct from direct derivatives of a selected native execution.

## Journey one Understand an unexpected output

### 1 Run normally and choose a target

Use the user's existing opening mode: local or caller-owned model, or an admitted resident worker. Record enough identity/history for the supported replay profile. Do not run `mdb analyze`, train an explainer, or capture every internal tensor as an implicit prerequisite.

After generation or a readout stop, let the user select the actual output token. Default to that token's scalar logit for mathematical inspection and show the complete native decision separately. A user can choose an alternative for a contrast. Display token IDs and decoded bytes, not only rendered words. A pairwise margin requires the declared tokenization; multi-token sequence scores need an explicit sequence-scoring contract. Do not silently treat a word as one token or forced-sequence likelihood as free generation.

Persist the target and horizon once. A later “explain” or “compare” uses it until changed. Future-token outcomes are native finite results; a differentiable first-token target does not explain all later generated decisions automatically.

### 2 Select an occurrence

The user can start from a known symbol, a saved slice, an output writer or the current stopped operation. Show the operation, token/clock, port, branch and supporting cut/receipt in the prompt or compact header. A human name is optional.

Existing `project show`, `project trace` and `project slice` remain useful navigation. Add one unambiguous focus over their results. A component aggregates many occurrences; do not silently choose one from that aggregate. If several occurrences fit, show a bounded list and let the user select. Remember that choice. Two physical replays can share a logical cut: preserve the recorded receipt selection where required.

### 3 Show a small explanation first

The first response should fit in a normal console frame:

```text
Selected: <operation occurrence> at <branch / token / layer>
Target:   <token score or declared contrast>, horizon <bound>
Reads:    <arguments and prompt/state links>
Writes:   <native result and next consumers>
Evidence: <declared / observed / differentiated / intervened>
Next:     inspect arguments | inspect attention | inspect MLP | seek | compare
```

This is an output sketch, not a shipped schema. Without materialized values, show verified metadata and the missing live work. A structural edge must remain structural even when nearby edges have measured effects. An unknown frontier is a navigable limitation, not an empty circuit.

Order expansions by usefulness: source tokens and values, native arguments and writes, then normalization/gate/route detail. Consumer-weighted signed effects are optional calculations. Norms and attention mass remain useful observations, visibly distinct from output effects. Frequency pairs, read-plane geometry and second-order terms are advanced views of the same occurrence.

### 4 Seek without changing the investigation branch

Seek should first resolve whether the target has retained state, a qualified replay recipe, or only a descriptive observation. Hydrate from the shared original checkpoint and available root. Execute observation replay on an isolated inspection branch, preserving the selected native branch and any pending edit. Refuse to execute a seek with an unresolved pending transaction rather than losing it.

If a tap can inspect a port but that port is not resumable/writable, label it inspect-only and offer the nearest supported edit boundary. Do not transform an observational clock into an invented StateCut. Check selected operand identity under the stated profile and preserve discrepancies. Return to the prior debugger context after inspection unless the user explicitly chooses a supported live boundary.

The attached worker performs model work through the existing admitted lease. Opening an offline Project never launches a model automatically. Heavy replay/derivative work has bounded scope and progress, can be cancelled, and preserves completed evidence. No selection should unexpectedly schedule a full-model sweep.

### 5 Test one interpretation

The user selects a supported edit or donor occurrence, previews its exact footprint and then uses the normal branch execution path. Donor payloads should be regenerated from referenced donor histories when supported. The comparison preserves the parent, changed slots, resulting cut and original native consumer.

Headline results are actual generated choices and the declared signed target change. Expand into gate/up changes, routing versus payload, interactions, rounding remainders and collateral results. A local derivative prediction and a finite native result occupy separate columns. The user can compare without adopting the edited branch; an explicit existing commit/adoption action controls the working branch.

Save a finding attached to the selected occurrence/slice and native evidence. Reopening it should recover both the mathematical explanation and the exact comparison that tested it.

## Journey two Find how two components cooperate

The user selects A and B in the same investigation, then requests a comparison. The panel builder resolves compatible parents, supported typed writes, doses, consumer and horizon from the context. It shows the exact four cases: sham, A, B and both. Include a repeated sham when needed to measure instrumentation/replay behavior; repeated controls do not create new semantic specimens.

Use existing immutable plan/outcome records and public backend admission. A catalogue plan is not execution authority. Preserve explicit edit ordering for overlaps. If the edits overwrite the same slot, do not call the pair an additive rectangle. Finite native outcomes can still be compared under their actual semantics.

The default result is four signed outcome rows and their finite interaction, with rounding diagnostics and every completed native continuation. An optional expansion identifies changes in routes, values, RMS, gate/up products and later consumers. Integrating mixed derivatives is a further bounded operation, not a prerequisite to the useful four-arm experiment.

The comparison leaves the original active branch selected. The user can inspect a branch, restore the parent or save the panel as evidence. Cancellation retains completed arms and labels missing arms instead of turning a partial panel into a scientific null.

## Journey three Test whether a recurring symbol means the same operation

Only after a useful occurrence is understood should the user define an interface hypothesis. For a binding/lookup example: entity address, bound value, queried entity, expected next result, output interpretation and allowed interventions. Reuse native references and transparent mathematical projections; no fitted mapping is required.

Select a small family of cases: renamed entities, exchanged values, changed query, irrelevant text, reciprocal donors and address/channel controls. The tool must expose how cases are grouped and how an expected answer is obtained. An external task specification supplies meaning; the tool does not infer truth from the model's current answer.

Report every row and retain counterexamples. If the same proposed arguments lead to different next symbolic states, open that pair side by side with their residual, K/V, history and consumer links. The next question becomes “which dependency did this interface omit?” The complete native state is still available; symbolic abstraction is a view, not a destructive compression.

State coverage plainly: immediate neuron output, one block, first answer, fixed follow-up or bounded own-token future. Sampled evidence supports the tested family. Proof status requires a separate declared obligation such as an invariant proof, exhaustive finite verification or sound uniform error bounds. A custom label or “causal-claim” finding does not promote itself.

## Proposed console shape

Keep new verbs inside `project`; reuse current motion and edit commands. The following is a design sketch, not runnable CLI syntax:

```text
project focus --slice SLICE_ID --node NODE_ID
project explain
project seek
project explain --view mlp
project explain --view attention
project explain --view effect
project explain --view frequency
project compare --edits EDIT_A EDIT_B
project compare --run PLAN_ID
project contract --from-comparison COMPARISON_ID
```

`focus` binds one occurrence and the chosen target. `explain` reads or expands that context. `seek` performs explicit bounded native work when available. `compare` plans a panel, and `--run` requests execution through the attached qualified backend. `contract` begins an explicitly scoped interface hypothesis. Normal `checkpoint`, `fork`, `preview`, `commit`, `abort`, `restore`, watches and findings retain their existing semantics.

These verbs should call the same public controller API used by future graph actions. A graph selection must carry an exact occurrence reference, not a rendered label. An offline graph may open the corresponding Project selection; it cannot execute scripts or edits merely because a record contains a replay descriptor.

## The persistent investigation context

Add a versioned sidecar describing:

- Project/session and selected slice node or receipt-qualified occurrence; source cut, branch, typed clock, port and selector.
- Declared consumer, contrast/tokenization, destination/horizon and target numerical program.
- Supported inspect/resume/write surfaces and any unresolved replay frontier.
- Parameter references, parent/history recipe and hashes; transform window, normalization and original numerical remainders for derived views.
- Saved analysis IDs, planned edits/panels, measured outcomes and associated findings.

Reopening resolves this context against current capability and identity, rather than treating saved availability as live authority. Stale bindings stay inspectable as history; new execution requires a compatible root and numerical program. Advancing the active session does not silently rebind a saved derivative to new state. Existing cut, graph, address, ABI and receipt identities remain unchanged.

Use existing Project analyzer/result mechanisms for immutable calculation records and source fingerprints. Lightweight metadata analyzers stay framework-free. Native replay, tensor expansion and derivative calculation are explicit live operations; register their sealed results back into the Project. Reuse the current permitted evidence references for findings, with result links attached through supported analysis metadata rather than bypassing the evidence validator.

## Implementation slices

| Slice | Concrete reviewable result | Existing seam | Validation |
| --- | --- | --- | --- |
| 1 Focus and read-only seek | Saved Project node resolves to one native occurrence, hydrates one existing Qwen source/read/output chain, opens a finding | `project_slice`, `Project`, `NativeReplay`, public adapter | Exact source/target identity; no branch mutation; ambiguous, missing and inspect-only cases visible; reopen works |
| 2 Native arguments | Attention and MLP expansion with original named-tensor views, signed finite accounting and compact output | Existing observer, symbols, parameter manifests, versioned analyses | Original operand/remainder closure; tap parity; no duplicate Q/K/V capture stack or exported weight bank |
| 3 Consumer effects | One source edit traced through O/residual/RMS/MLP to full native logits and own-token outcome | Qualified native profile and shared consumer record | Full downstream adjoint when claimed; native sham/restore; local versus finite disagreement retained; no substitution of a fitted lens |
| 4 Cooperation panels | Four-arm comparison from one focus, expandable into exact finite terms and result-linked findings | Circuit plans, protocol compiler, value coalitions | Correct edit footprint/order; native continuation; cancellation/partial rows; same-parent selection preserved |
| 5 Interface witnesses | Binding/lookup family with cross-context argument comparisons and counterexample navigation | Symbols, findings, source-qualified atlas | External specification; explicit coverage; preserved nuisance/state dependencies; no automatic semantic promotion |
| 6 Integrated proof reports | Changing derivatives and bounded compositional obligations connected to actual evidence | All previous slices | Numerical/reference closure plus separately scoped semantic proof; native rounding and every required output competitor accounted for |

First deliver slices 1–3 on one existing Qwen2.5 profile, using the temperature case to check the known output effect. Then use binding/lookup cases to distinguish source value, query and irrelevant context. Qwen3 local hydration is supporting evidence for techniques, not execution compatibility with Qwen2.5. Diffusion/FLUX comes later through a separately qualified denoiser/image consumer and its own state closure.

Also exercise anonymous navigation on a fresh ordinary code or prose prompt within that same profile. The user should not need a temperature grammar, supplied role table or semantic source annotation to obtain the mechanical explanation. Any semantic interpretation of that fresh prompt remains separate from navigation and replay qualification.

## Retention and execution policy

Default new investigation records should be metadata, input/edit history, compact results and shared immutable parameter references. Hydrated tensors are temporary analysis state. Derived caches become disposable only after the selected regeneration path is qualified; unsupported roots may still need retained native payloads. Historical evidence and inventories remain untouched.

Use `mdb -> saturn_pub` public contracts, installed package dependencies and existing mrun admission for real models. Do not add private workspace imports, sibling-path discovery, a second symbol table or an alternate intervention runtime. Documentation's [workspace research links](NATIVE-CIRCUIT-MATH.md#research-sources) are provenance, not runtime dependencies.

Mechanics validation protects identity, schema, edit authority, replay and evidence integrity. Scientific panels retain signed effects, doses, counterexamples and disagreement; ranking, activation energy and semantic labels do not block discovery. Exploration and terminal proof remain separate records.

## First acceptance walkthrough

A user reopens an existing Project, selects the source-linked output, inspects its arguments, seeks to the exact occurrence, opens a parameter view, runs a supported comparison, saves a finding, exits and reopens it. The source and native outcome identities remain linked throughout. Offline browsing does not import model frameworks or launch compute. No predictor fitting, exported parameter bank, hand-authored per-experiment worker or separate viewer is required for this supported path.

The workflow is complete for that slice only when a second user can follow it through MDB. Another standalone graph or mathematically correct script is supporting work, not completion of the integration.

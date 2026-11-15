# Proposal for computation explanations and circuit discovery

MDB should connect a behavioral difference to the native operations that produced it, nominate interventions that distinguish competing explanations, and retain the contexts where its explanation works or fails. The mathematical foundation is the [model math bible](../../../../../obsidian/model-math/index.md). The first deliverable is a Python investigation workflow over the refactored engine; automatic semantic naming and a new web interface come later.

This is primarily a proposal to integrate existing capabilities and close discovery gaps. Earlier MDB family workflows already implement routing/value and gate/up experiments, native consumer tests, sealed forecasts and fresh-context evaluations. The new engine already contains the finite mathematical primitives. Repackaging these in one report is useful integration, not evidence that those capabilities were previously absent. The next substantive advance must be assessed against both the engine and the older workflows: automatic producer tracing, useful intervention selection and counterexample-driven interface refinement.

## Problem and intended behavior

The current circuit builder decomposes a consumer into residual writes, heads, neurons and source rows. Its exact accounting and explicit remainder are useful, but its neuron leaves do not explain the gate/up arguments, and its source leaves do not explain the routing decision. Contribution magnitude is not a complete measure of downstream effect. Zeroing the entire complement of a small circuit can destroy the computation being investigated.

A researcher should be able to compare two branches and follow this chain:

```text
source occurrence and stored K/V
    -> query/key contrasts and source payload
    -> changed attention return and residual
    -> normalization and gate/up arguments
    -> changed neuron products and MLP write
    -> downstream readers and declared output
```

The graph should distinguish an observed dependency, an exact endpoint identity, a local sensitivity, a native intervention effect and a semantic hypothesis. These are complementary evidence types.

## Mathematical capabilities

### Explain finite changes

For matched attention endpoints, expose payload, routing and interaction terms:

\[
\Delta A=\sum_j[p_{0,j}\Delta v_j+\Delta p_jv_{0,j}+\Delta p_j\Delta v_j].
\]

For SwiGLU, expose the changed gate, up argument and interaction:

\[
\Delta f=u_0\odot\Delta\operatorname{SiLU}(g)+\operatorname{SiLU}(g_0)\odot\Delta u+\Delta\operatorname{SiLU}(g)\odot\Delta u.
\]

Retain normalization changes, both residual skips, affine biases and native arithmetic discrepancies. Endpoint identities account for observed changes; their terms are not unique isolated causal effects. Existing primitives include `transport_finite_split`, `swiglu_finite_split`, `rms_finite_change` and `score_finite_change` in `mdb.analysis.finite`. See bible chapters [04](../../../../../obsidian/model-math/04-attention.md), [05](../../../../../obsidian/model-math/05-mlp-writes.md) and [09](../../../../../obsidian/model-math/09-finite-edits.md).

### Expand native arguments and downstream use

A SwiGLU neuron reads two coordinates and writes one direction. Expand both read arguments into their actual producers, retaining normalization and shared state. Compose the original operator pullbacks to distinguish direct residual accounting from full downstream sensitivity. Rank candidates through the declared consumer, then measure selected finite interventions. Existing foundations include `neuron_read_plane`, `swiglu_source_pairs`, `mlp_pullback`, `consumer_gradient` and `explain_port`.

### Choose informative interventions

Generate read-plane invariance checks, query/key-contrast invariance checks, routing-versus-payload interventions, reciprocal replacements and small native coalitions. Keep zero transport and mask-and-renormalize as different interventions. Use final-write decision envelopes only under their fixed-operands, final-RMS and bias-free-head assumptions. Preserve signed effects and cancellation; do not require every diagnostic to agree before collecting another branch.

### Test symbol interfaces

Add semantic hypotheses alongside the existing numerical cell partition. A proposed symbol records its value, native carriers, definition/use occurrences, consumer, horizon and applicability evidence. Multiple roles may overlap in native storage.

Search for two states with the same proposed interface but different observations or next interfaces under the same admitted input. Such a counterexample identifies an omitted dependency. Trace and refine that dependency rather than adding an arbitrary context ID. The exact sufficiency criterion and semantic induction are in bible chapters [10](../../../../../obsidian/model-math/10-circuit-composition.md) and [13](../../../../../obsidian/model-math/13-state-closure.md); finite tests establish only the tested domain.

### Use interpretable contrasts

Display score contrasts, value contrasts and the first-row opening rather than treating raw score magnitude as importance. Preserve the chosen reference and numerical mass defect. Chapter [25](../../../../../obsidian/model-math/25-softmax-offsets-and-gated-reads.md) distinguishes exact softmax freedoms from approximate slow-RoPE freedoms. No universal key-bias removal is proposed.

## Implementation sequence

1. Add a bounded branch-change analysis API. Accept two completed, aligned runs of the same adapter and input tokens. Explain selected blocks with native attention, normalization and MLP endpoints. Retain top source/neuron rows and an exact aggregate tail, mathematical closure, native discrepancy, input/run lineage and consumer scope. Return tensors separately from bounded JSON.
2. Join those records to producer addresses and downstream pullbacks. Preserve direct accounting and suffix sensitivity as separate columns. Wire edited occurrence pointers before promising durable graph navigation.
3. Add prospective intervention proposals and reference-based circuit experiments. Execute one nominated family from an immutable parent, measuring target and collateral outcomes. An unmeasured coalition corner stays unknown.
4. Add cross-context interface counterexample search and symbol hypothesis lifecycle. Preserve contradictory outcomes and separate interface reconstruction from autonomous own-state continuation.
5. Expose the qualified operations through the shared command registry and web UI after the Python workflow proves useful.

## First trial

Use Qwen2.5-1.5B with eager attention, batch one and BF16 CUDA, matching the established temperature case. Load once in a single finite mrun lease. Use the boiling prompt only as a donor; baseline and edited receivers have identical freezing prompt tokens and one common parent. Fix layer 20 and source row 18 from the historical case before inspecting new outcomes.

Run baseline, identical-copy sham, both donor V cells, donor K cells and donor K+V cells. Explain the attention and MLP changes at the edited block and a downstream block. Keep all native margins, full logits for the one consumer row, signed component summaries, source/neuron selections and discrepancies. Retain the fixed site as a supplied candidate; this experiment tests explanation utility and does not claim autonomous localization.

Use the decompositions to nominate a follow-up intervention on a fresh receiving context. Freeze its numerical or directional prediction before its native execution. Keep accounting-only endpoint analysis separate from any prospective prediction.

## Integrity and interpretation

Mechanics requirements are matching model/input identity, aligned visible source sets, finite tensors, correct native port delivery and exact restoration of the parent after branch experiments. The numerical profile records backend, dtype, attention implementation, relevant CUDA reduction flags and software versions. Parameter references must bind immutable checkpoint archives and named tensors through `retrieve_weights.py` and `parameter-artifacts.json`.

Discovery outputs report effects, errors and counterexamples without a scientific pass/fail threshold. A nonfinite worst-case bound does not erase a measured native effect. Closure by construction checks bookkeeping, not discovery. Native discrepancies remain measured quantities and are not automatically certified bounds or solely rounding errors.

The initial API restricts direct edits of the explained output/intermediate ports unless their explicit displacement is separately accounted for. Different prompts, changed visibility and different model instances require a later alignment contract. A projection through a fixed final readout is direct residual accounting, not the full suffix effect of an isolated local term.

## Evidence needed for stronger claims

The trial is useful if the report distinguishes routing from payload changes, exposes downstream gate/up responses and identifies a testable difference hidden by a contribution list. A semantic circuit result additionally needs an independently specified operation, fresh contexts and predicted intervention behavior. A claim of superiority over public tools requires a matched comparison of discovery quality, effort and cost; it does not follow from implementing these equations.

## Deliverables

The first slice consists of a documented analysis API, independent checks against native ports, a reproducible finite worker, an authenticated parameter retrieval manifest, immutable native results and a findings report. Later stages remain proposals until implemented and exercised.

The [first native trial](BRANCH-CHANGES-TRIAL.md) now exercises this slice, including an independently checked branch decomposition, a nominated neuron restoration and a persisted conditional forecast in a reordered context.

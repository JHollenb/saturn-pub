# Future native circuit capabilities

Status: proposed capabilities, November 12, 2026. Documentation only, based on consolidated `origin/main` at `b93b9f4`. No new runtime commands, adapters, numerical qualifications or semantic circuit claims are introduced here.

The objective is understanding the original computation. A user should be able to run a model, select an output or internal occurrence, inspect its arguments and writes, test a change, and follow the native result. These capabilities require no fitted predictor, new sparse model or semantic name. They extend the existing [Project workflow](SATURN-PUB-PROJECTS.md), [symbol slices](SYMBOL-SLICES.md) and [replayable circuit navigation plan](REPLAYABLE-CIRCUIT-DEBUGGER-PLAN.md).

The recurring scaffold organizes writers, persistent registers, readers and consumers. A dynamic invocation supplies particular arguments, addresses and values. The four capabilities below make that invocation inspectable and testable. The companion [workflow proposal](NATIVE-CIRCUIT-WORKFLOW.md) describes how to expose them through MDB, and the [mathematical reference](NATIVE-CIRCUIT-MATH.md) supplies their assumptions and proofs.

## 1 Inspect arguments and their downstream use

User question: **What did this computation read, calculate and write, and how did that write affect this output?**

For a selected native MLP occurrence, show original gate/up parameter references, actual normalized-input projections, activation/product and down-projection write. Connect the arguments to attention returns and earlier residual state. Bind an optional local consumer derivative to the exact state and declared output. Supporting and opposing contributions remain signed.

For attention, show score formation, visibility, allocation, weighted values, grouped-query mapping and output projection. Expose the centered value `source V − current head return`: it determines the local effect of changing a source score. Frequency terms expand these original operands rather than replacing their addresses.

The result is an inspectable function invocation with provenance and a consumer, not an automatic concept label. The [two-coordinate read-plane proof and attention derivative](NATIVE-CIRCUIT-MATH.md#read-planes-and-consumer-linked-attention) explain the mathematics. A plane can have rank one; it is not necessarily two independent controls. Whole-model semantic dimension is not bounded by a single neuron's read plane.

Existing foundation: native symbols/receipts, [attention-symbol links](ATTENTION-SYMBOL-LINKS.md), [forward symbols](FORWARD-SYMBOLS.md), public parameter references and separately measured connected derivatives. Missing integration: fine-port hydration, original-weight views, coherent source-to-consumer expansion and a shared inspection result.

The existing [Jacobian workspace](JACOBIAN-WORKSPACE.md) consumes fitted frozen lenses. This proposal derives direct native local derivatives from the selected computation; it does not silently treat a fitted transport as an exact native Jacobian.

## 2 Compare cooperation and opposition

User question: **Do A and B act independently, reinforce each other, or cancel? Where does that happen?**

Create a bounded common-parent panel containing sham, A, B and A+B, with exact edit semantics and native continuations. Retain each branch, signed target/collateral effects, replay checks and numerical diagnostics. Start with the finite four-arm contrast; offer local or integrated interaction mathematics only where supported.

The [Möbius/Hessian identity](NATIVE-CIRCUIT-MATH.md#finite-interactions-and-mixed-derivatives) connects compatible additive edits to the integral of the full consumer mixed derivative. [Finite gate/up and attention identities](NATIVE-CIRCUIT-MATH.md#exact-finite-accounting) help locate product interactions without throwing away other terms. Same parent alone does not make arbitrary overwrite Acts additive or commuting.

Existing foundation: typed forks/Acts, [circuit plans and recorded outcomes](CIRCUIT-EXPLORER.md), [computation protocols](../src/mdb/computation_protocol.py), [value-coalition code](../src/mdb/value_coalition.py), response accounting and native comparison assays. Missing integration: a reusable panel builder on Project occurrences, shared consumer specification, and result links back into the selected slice.

An intervention effect is distinct from necessity or a unique circuit. Alternative paths, later repair and receiving-state dependence remain visible. Four-arm output differences alone do not localize the internal interaction.

## 3 Test whether a symbol captures enough state

User question: **Does this proposed interface explain the next operation across these contexts? What dependency is missing?**

Let an interface specify arguments, expected next result, consumer/horizon and a declared context/intervention domain. Group explicitly selected occurrences by those arguments, replay compatible tests, and retain disagreements. A counterexample opens the concrete states and their differing producers; it does not disappear into a global average.

The [state-sufficiency theorem](NATIVE-CIRCUIT-MATH.md#state-sufficiency-and-composition) gives the precise criterion. A native neuron's read coordinates suffice for its immediate scalar output, but usually not for the model's future. Other residual directions, K/V, history, positions and control state can matter. A semantic view can be incomplete while the full native state remains replayable.

Existing foundation: qualified symbol bindings, versioned analyzers, findings, slices and the [source-qualified computation atlas](COMPUTATION-ATLAS.md). Missing integration: interface specifications, cross-occurrence matching, counterexample reports and bounded composition claims. This is not automatic proof or a general semantic resolver.

Exploration should always retain per-row effects and disagreements. An eventual proof requires exact reasoning, exhaustive coverage of a declared finite domain, or sound uniform bounds. Accuracy on sampled prompts is evidence with its actual denominator, not universal certification.

## 4 Reopen and execute the explanation

User question: **Can I jump from this saved explanation to the original operands, test it, and inspect the actual future?**

Join selected graph/symbol occurrences to compatible live replay. Materialize original parameter views and operands on demand, inspect them, fork supported edits and continue through the original consumer. Save compact findings, references and edit/history recipes. Parameter weights remain in their shared immutable checkpoint.

Existing foundation: [bounded native replay recipes](REPLAYABLE-CIRCUIT-DEBUGGER-PLAN.md#implemented-bounded-recipe-api), retained public cuts, Project navigation and symbols. Missing integration: generic occurrence-to-live resolution, temporary inspection taps, fine-port support, explicit retention policy and a consistent console/viewer experience. An observer-only occurrence does not become a resumable or writable cut.

The recent Qwen3 campaign measured upstream routing/payload/MLP accounting and selected replay hydration; the Qwen2.5 campaign connected a value edit to native logits and own-token continuation. These are complementary experiments on different specimens. A unified measurement must use the same model, intervention contract and consumer. [Research sources and evidence locations](NATIVE-CIRCUIT-MATH.md#research-sources) retain that distinction.

## Bounded semantic circuit proofs

A semantic claim needs a specified operation, such as retrieving the value bound to a queried entity. Native state variables and interventions must correspond to that operation's variables and counterfactuals. The [composition proof obligation](NATIVE-CIRCUIT-MATH.md#state-sufficiency-and-composition) includes initial states, compatible edits, reachable states and readout meaning.

Naming a component, matching a spectrum, obtaining a token flip or achieving a low reconstruction error does not satisfy that obligation alone. The intended capability is a bounded, inspectable account of an operation, with its supporting evidence and counterexamples. Learned sparse coordinates can remain optional overlays.

## Delivery order

1. **Highest priority:** selected occurrence → read-only seek/hydration → original operands, with an end-to-end Project finding.
2. Inspect original attention and MLP operations, then attach a declared consumer and signed local effects.
3. Reusable common-parent intervention panels and finite accounting through native output.
4. Cross-context interface tests and counterexample navigation.
5. Integrated derivatives and bounded semantic proof reports on a declared task family.

See the [user journeys, proposed commands and implementation slices](NATIVE-CIRCUIT-WORKFLOW.md). Each delivery should work in the existing console and framework-free saved Project before a new visualization is required.

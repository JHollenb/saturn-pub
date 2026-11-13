# Symbol table capabilities and measured results

MDB connects native model values to qualified names, histories, equations, causal experiments and compiled predictors. It can expose an executed computation, bind a proposed role to a supported state location, retain how later operations use it, and compare prospective predictions with native execution. Selected prompt families also have predictors that construct states and skip transformer layers.

This reference describes the implementation and existing evidence as of November 6, 2026. Numerical replay, semantic interpretation, causal effect and cheaper execution are distinct properties. The companion [research blog](../../../../obsidian/blog/2026-11-06-230748-symbol-tables-in-mdb-from-causal-roles-to-cheaper-execution.md) provides the broader mathematical narrative in a sibling Obsidian checkout.

## Capability map

| Component | Implemented capability | Qualification boundary |
|---|---|---|
| `ForwardSymbolTable` and `ATenStepper` | Record anonymous native values and def/use edges; navigate metadata; execute recorded tensor primitives from original roots | Deterministic admitted operators; the bounded tensor replay contract uses dense contiguous CPU roots and refuses unsupported random/mutable closure |
| Public `SymbolBinding` and `SymbolTable` | Bind a name to supported locations at an exact model, execution, context, point and schema | Read-only debug information; context completeness and semantic interpretation require evidence |
| `symbol_sources` and `symbol_producers` | Nominate labels from explicit sources and join verified producer records to existing native bindings | Source verification and native joins do not establish semantic equivalence or causal dependency |
| `symbol_lifecycle` and `symbol_trace` | Retain definitions, uses, clocks, invalidations, native ancestry, receipts and optional CoT references | Missing or ambiguous joins remain explicit; navigation does not hydrate tensors or grant writes |
| Project slices | Trace declared upstream writers and downstream consumers within a selected future | Whole-slot declared dependencies; a coordinate selector does not establish coordinate-specific causal influence |
| Forecast transactions and `EquationCatalog` | Capture declared inputs, seal predictions before native release, compare outputs and retain immutable applicability records | Callbacks are trusted Python; numerical fidelity and scientific interpretation remain measured properties |
| `ComputationAtlas` | Import explicit equation observations across runs and compare caller-selected response families | Metadata retrieval and comparison; automatic semantic naming, rule induction and promotion remain open |
| `computation_applicability.evaluate` | Select a method from a bound source packet and its conditions before outcome lookup | Current replay binds specific inherited Qwen block-18 packets and declared profile |
| `compiled_state`, runner and serving | Fit state maps, resume at a later layer, inject compiled K/V, free-run short zero-layer chains, refresh from native examples | Declared finite prompt families and supported full-attention decoder lanes |

Implementation references: [native values](FORWARD-SYMBOLS.md), [source candidates](SYMBOL-SOURCES.md), [qualified slices](SYMBOL-SLICES.md), [lifecycle](SYMBOL-LIFECYCLE.md), [symbol trace](SYMBOL-TRACE.md), [forecasts](COMPUTATION-FORECAST-WORKFLOW.md), [atlas](COMPUTATION-ATLAS.md), [applicability](COMPUTATION-APPLICABILITY.md), and [compiled state](COMPILED-STATE.md).

## What a symbol binds

The public [SymbolBinding implementation](../../../src/saturn_pub/symbols.py) records a symbol name, model identity, execution digest, context, context-slot digests, locations, evidence references, bounded claim and status. Each location has an exact execution point and state schema. A binding may describe several locations; several bindings may describe one name at different supported points.

Resolving a name requires a unique supported binding in the current context. A changed checkpoint, execution, relevant context or unsupported address cannot silently reuse a previous location. The caller declares the context dependencies; the library verifies those declarations rather than inferring whether they are scientifically complete.

A larger operational computation entry combines a role, resolver, operand ports, law, applicability, numerical profile, consumer, continuation and evidence. Those pieces currently span separate records and APIs. This is an execution pattern, not an additional universal serialized schema.

Names have distinct sources. A forward value such as `v000042` is anonymous and mechanically identified. A source-authored module path describes executable code. A supplied name records an interpretation. A measured role adds experimental support. CoT vocabulary observations remain candidate readouts unless separate experiments establish their meaning and dependency.

## Native values and actual execution

The recorder intercepts native tensor operations, including functional calls omitted by module hooks. Values retain shape, dtype, exact content digest, preview, producer and consumers. The metadata navigator imports no model framework.

The following commands navigate an existing retained trace:

```sh
mdb forward-symbols trace.json --list
mdb forward-symbols trace.json --step 12
mdb forward-symbols trace.json --value v000042 --json
mdb forward-symbols trace.json --readable --scope mixers.2
```

CLI `--step` selects a recorded operation for inspection. `ATenStepper.step()` executes the recorded tensor operation. The executor starts from original inputs, parameters, buffers and admitted literals; it recomputes intermediates and refuses captured future activations as roots. See [recording and replay](FORWARD-SYMBOLS.md) for the Python example and supported closure.

Native module ports expose actual args, kwargs, inputs and returns. These are separate from inferred mathematical regions. Repeated calls require a specific occurrence, and named parameter/buffer ports remain declared roots. See [actual ports](COMPUTATION-FORECAST-WORKFLOW.md#actual-module-and-parameter-ports).

## Definitions and consumers across a branch

A `SymbolOccurrence` records a definition at a source cut. A `SymbolUse` identifies a native consumer, source, immediate parent/result cuts, receipt and observations. `SymbolLifecycle` preserves immutable versions and additive invalidations. `ClockWitness` retains a typed temporal coordinate and its provenance.

Native layer/token coordinates, recurrent sequence positions, selective timestep fields and local age policies remain different clock kinds. Host wall time and paging recency do not establish semantic expiry.

For an existing project and registered carrier:

```sh
mdb project ./investigation slice --symbol carrier --cut SOURCE_CUT --context example
mdb project ./investigation slice --symbol carrier --cut SOURCE_CUT --context example --to-cut FUTURE_CUT
mdb symbol-trace ./investigation carrier --session-id trial --cut FUTURE_CUT
```

The future must descend from the source in retained native ancestry. Sibling events are excluded from a selected-cut history. Missing graph coverage remains an explicit frontier. A complete declared slice explains the recorded dataflow within scope; causal influence requires native intervention evidence.

These metadata queries do not load tensor payloads. Actual reads and interventions use a compatible native Session. Source records, symbol bindings and seals preserve provenance; typed Acts and native receipts establish what was changed and executed. See [slice semantics](SYMBOL-SLICES.md) and [trace verification](SYMBOL-TRACE.md).

## Causal discovery and forecast transactions

The practical discovery loop is:

1. Freeze checkpoint, input contrast, execution profile, consumer and continuation horizon.
2. Capture a common parent and inspect candidate producers, carriers and readers.
3. Apply bounded replacements, doses or restorations, with controls that address the live ambiguity.
4. Continue the original native computation using each branch's actual state.
5. Retain target effects, collateral, numerical differences, continuation and counterexamples.
6. Store the bounded interpretation beside its addresses and evidence.

The [discovery workflow](COMPUTATION-DISCOVERY-WORKFLOW.md) explains screening, selected scalar replay and native route tests. A rescue or donor transfer establishes a conditional effect; it does not automatically identify a unique minimum circuit.

`ForecastTransaction` additionally enforces capture → predict → seal → native release for the instrumented model. Predictions receive the declared inputs and weights and are immutable after sealing. Source/checkpoint identities, physical work and native comparisons remain attached. Direct calls outside the instrumented model are outside the module-entry guard's authority.

For an existing forecast and equation catalog:

```sh
mdb forecast verify ./forecast-run
mdb forecast show ./forecast-run
mdb equations find --catalog ./investigation/equations --law-id law_003
```

`EquationCatalog` stores equations and comparisons with tested contexts and supplied applicability conditions. `ComputationAtlas` imports selected source-qualified records across runs, preserving signed effects and counterexamples. Matching conditions is retrieval eligibility; it does not establish accuracy on a new input.

## Applicability before outcome lookup

`computation_applicability.evaluate` accepts a source packet, intervention, execution profile, output specification and policy. It fixes a decision before annotating historical coverage. It reads no files, arrays or models and rejects target/error/pass fields.

The shipped replay chooses among a cheap sign relation, full-query recomputation from supplied Q/K/V, a complete functional-block fallback and abstention. Every selected method retains all 512 queries. It does not invent missing operands or combine ports from unrelated packets.

Four replay decisions and 28 metadata controls were executed without numerical-outcome reads. All four decision hashes were unchanged when historical coverage was added. The current domain is the named Qwen block-18 sources, original parameters and positions, no cache, all rows attended and the declared BF16 CUDA profile. Hardware execution and payload validation remain the runner's responsibility. See [the planner and retained replay](COMPUTATION-APPLICABILITY.md).

## State compilation and generation

For a declared finite family, a residual compiler uses

$$
\widehat r_{p,L}=b_{p,L}+\sum_{j\in V(p)}A_{p,L,j}[s_j].
$$

The symbols $s_j$ and their visibility are supplied by the family. Ridge fitting determines the maps. Switch features select conditional maps at positions where the switch is available. Derived symbols such as carries or column totals are supplied computations, not automatically discovered semantics.

`DecoderSplice` can start at layer $L$ and execute only the remaining layers. Partial-position runs can supply compiled K/V for other positions. At the final layer boundary, the compiled residual goes directly through final normalization and the output head.

`ZeroLayerChain` feeds its own predicted tokens back through declared lookup tables before computing later states. Teacher-forced and free-running agreement are separate measurements. `serve` starts with native warmup, sends low-margin prompts to native decoding, adds those observations to the fit and periodically refits. Audit tokens score answers without reaching the fit.

The field `forwards_per_1000_prompts` counts native-served prompts per 1,000, not individual forward invocations. Reports separately retain native token steps. Existing fit, discovery, refresh, parser/features, output-head and backing-state costs must remain explicit when claiming savings. See [compiled-state API and qualification](COMPILED-STATE.md).

## Why a fixed-parent readout can be cheap

For a final Qwen residual $r$, MLP write $m$ and dose $\lambda$, write $s=r+\lambda m$. A bias-free head after RMSNorm gives

$$
L_v(\lambda)=\frac{a_v+\lambda b_v}
{\sqrt{c_0+2\lambda c_1+\lambda^2c_2+\epsilon}},
$$

with $a_v=W_v(\gamma\odot r)$, $b_v=W_v(\gamma\odot m)$ and $c_0,c_1,c_2$ the width-normalized inner products of $r$ and $m$.

The shared positive denominator means the winner is $\arg\max_v(a_v+\lambda b_v)$. The coefficients can be reused across many doses. This requires the original parent's state and coefficient construction; it does not construct a new arbitrary prompt state cheaply. Finite execution, casts and near-ties remain part of numerical qualification.

The [full-vocabulary assay](QWEN-COMPUTATION-FORECASTS.md) predicts the first-token readout; later tokens there are native observations. It is separate from the compiled-state free-running chain.

## Measured evidence

The table distinguishes package qualification from related workspace research using MDB/public symbols. Sibling workspace sources are linked explicitly; an isolated package checkout does not include those repositories.

| Result | Measurement | Scope and evidence |
|---|---|---|
| Native forward replay | Three calls, each 571 operations/676 tensor identities; all 1,713 operation outputs exact | R3 Mamba, deterministic CPU FP32; [forward-symbol qualification](FORWARD-SYMBOLS.md#measured-native-mamba-use) |
| Reader-role generalization | 18/18 contrast types on longer addition and 12/12 on another format; 128 pairs per type | Qwen2.5-1.5B workspace causal study; [findings](../../../../saturn/experiments/2026-10-30-dynamic-symbol-table/FINDINGS.md); lifetime/control exceptions retained |
| Full-vocabulary forecast | 1,519,360 logit values, maximum absolute error 8.75655e-6, 10/10 first-token winners | Qwen2.5-0.5B, two prompts × five conditions, observed final-layer frontier, CPU FP32 native/FP64 predictor; [assay](QWEN-COMPUTATION-FORECASTS.md#full-vocabulary-results) |
| Partial state compiler | 98.3% native-answer agreement, 53.3% error recall over 160,200 addition cases; 8.6× batch-256 speedup | Workspace Part F, three answer positions run layers 20–27, compiled operand K/V, digit head, prefix-cached native baseline; [findings](../../../../saturn/experiments/2026-11-02-forecast-cost-and-compressed-addition/FINDINGS-F.md) |
| MDB zero-layer chain | 744/744 known-filler evaluation prompts exact through four self-fed tokens; 14.15× batch-1 and 15.65× batch-256 speedup | Qwen2.5-1.5B factual family, full-vocabulary chain versus KV-cached decode, ten timed repetitions; [qualification JSON](../qualification/compiled-state-forwardless-20261102.json) |
| MDB native refresh | 2,256 prompts, 297 native prompt decodes, 99.734% served whole-chain agreement | Same family including unfamiliar fillers, warmup included; complete serving loop not timed; [qualification](../qualification/compiled-state-forwardless-20261102.json) |
| Internal operand editing | 16/16 destination words and flag vectors correct; 15/16 full register states; one coordinate edit changes 59 result bits in an example | Related ISA NN workspace study, recipient-only editor, native arithmetic and subsequent COPY/ADD use; [findings](../../../../isa/docs/arithmetic-symbol-value-edit.md) |
| Complete raw-prompt image account | 1,011 fields, 614,793,216 retained scalar positions, all bit-exact through pixels | One FLUX.2 Klein 4B specimen, raw Qwen conditioning, 112 transitions and explicit decoder; shared Torch primitives and dense learned factors; [account](FLUX-RAW-TO-IMAGE.md), [saved audit](assets/flux-raw-to-image-20261106/audit-data.json) |

The zero-layer chain's known-filler result does not extend unchanged to unseen fillers: its 384-row held-out-filler arm matched 46.1% through four tokens before refresh. The refresh assay pays native observation costs to improve coverage. Its separate audit generation is not counted as serving work and does not train the compiler.

The addition panel scores the declared answer positions and is separate from the own-token-feedback chain. Native-error recall is the fraction of the native model's wrong answers reproduced exactly. A predictor that computes correct arithmetic can have high agreement while having zero error recall.

The image scalar count includes repeated carried fields along one dependent history. The producer completed scientific outputs but failed final packaging; a separately recorded recovery authenticated and audited the retained evidence without model rerun. The numerical account does not establish a denoiser speedup.

Timing comparisons require the same workload boundary. The related CPU dose study reports about 219 ms for 1,000 analytic queries versus 1,058 ms for a batched native suffix, a 4.8× advantage. Its approximately 490× sequential-full-forward comparison extrapolates five of six sequential baselines. CPU winner comparisons have no mismatches in the tested comparisons; GPU BF16 near-tie discrepancies are retained. See [the workspace timing record](../../../../saturn/experiments/2026-11-02-forecast-cost-and-compressed-addition/timing/FINDINGS-A.md).

Most scientific panels are single-seed and bind particular checkpoints/profiles. Report tensor equality, token equality, semantic outcome, collateral and wall time separately.

## Scope of the remaining work

Numerical operations can be reconstructed while their useful semantic organization remains partly understood. An address can be known while a cheaper resolver remains unavailable. A local write can match while a later consumer needs additional state. Semantic labels, including CoT readouts, need evidence beyond neighboring vocabulary observations; see [the register contract](COT-REGISTER.md).

The current shared challenge is economical production of contextual operands on the executor's own evolving states. Automatic cross-run rule induction, arbitrary semantic circuit naming, complete faithful CoT reconstruction and unrestricted cheap chat execution remain separate research goals. Existing exact numerical accounts, causal interfaces and bounded compiled predictors provide the execution and measurement foundation for that work.

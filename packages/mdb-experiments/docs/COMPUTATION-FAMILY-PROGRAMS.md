# Executable computation-family protocols

`mdb family` extends the earlier [metadata compiler](COMPUTATION-FAMILY-PROTOCOLS.md) with a finite execution program, a graph interpreter and frozen effect evaluation. It works with supplied operation graphs and registered model adapters. The graph and the choice of useful ports remain explicit research inputs.

The initial model campaigns use two nonlinear operations inside FLUX.2 Klein 4B distilled: attention routing/value formation and SwiGLU gate/value formation. They are nested producers inside the previously studied [parallel image writer](FLUX-PARALLEL-RELATION-FAMILY.md), rather than four independent semantic modules or four different models.

## What the tool generates

A specification binds a checkpoint, capture clock, typed operand shapes/stages, coherent factor bundles, a receiving-state closure, a native continuation and a metric domain. The compiler emits every factor-source corner, reference roles, a candidate menu, an ordered execution program and audit obligations.

For two factors, there are six actions:

| Role | Meaning |
| --- | --- |
| Recipient reference | Natural full Q history, including a full-state no-op check |
| Source reference | Natural full P history |
| Operator baseline | Both operator factors from Q, under the declared mixed receiver |
| First-factor candidate | First factor from P; second from Q |
| Second-factor candidate | First from Q; second from P |
| Component reference | Both operator factors from P, under the mixed receiver |

The initial experiments fix the image residual to P and retain Q postblock text, modulation, conditioning and latent. Attention holds Q MLP features; SwiGLU holds Q attention features. The operator baseline is therefore the R-only construction. It is distinct from natural Q. The component reference is also distinct from natural P.

Only the two single-factor candidates enter selection. Source, recipient, operator baseline and component reference remain controls even if their score is better.

## Specialize, compile, freeze and evaluate

With this MDB revision installed:

```sh
mdb family specialize specification-template.json capture-metadata.json \
  --out specification.json
mdb family compile specification.json --out program.json
mdb family freeze program.json development-measurements.json \
  --fresh-contexts fresh-contexts.json --out effect-rule.json
mdb family evaluate program.json effect-rule.json fresh-measurements.json \
  --out effects.json
mdb family compare attention-program.json mlp-program.json --out comparison.json
```

Files are created exclusively. Use a new output path for each execution. Framework-free compilation and effect bookkeeping load no Torch, Diffusers, arrays or parameters. The installed qualification wheel and receipt are in `artifacts/mdb-family-programs-20261107` in the research workspace; the common environment's older installed MDB does not automatically gain these commands.

`specialize` resolves symbolic dimensions from checkpoint- and clock-bound capture descriptors. Explicit view recipes can split a captured fused preactivation into its two halves, retaining the parent hash, shape, stride and relative offset. The recipe does not invent a hash for the child bytes; the execution adapter checks those bytes. In the initial SwiGLU campaign this derives two `[1, 1536, 9216]` operands from the actual `[1, 1536, 18432]` input, including its padded live stride.

Use full context objects in `fresh-contexts.json`, including `id`, exact P/Q prompt bytes and `seed`. New rules bind these identities and reject relabelled, reversed or reused development cases. Measurements must carry the audited context metadata. Names-only legacy rules remain readable, with their weaker identity scope explicit. A held-out result is never inferred from a label alone.

The primary metric in the initial campaigns is the signed native final-latent coordinate

\[
p(z)=\frac{\langle z-z_P,z_Q-z_P\rangle}{\|z_Q-z_P\|^2}.
\]

Natural P and Q define the axis. A development context selects the candidate with the lower score. Its strict ordering against the other candidate is frozen before fresh input preparation. A smaller coordinate can overshoot P or have large orthogonal error; component-relative distance, source distance, image quality and visible object relationships remain separate observations.

`evaluate_rule` retains each signed margin. A fresh tie or reversal is a counterexample; an undefined axis is not applicable. Neither outcome suppresses another row or aborts optional evidence collection.

## Run the generated program

The qualified model worker calls `execute_program(plan, handlers)` using the literal generated sequence:

```text
forecast sources → construct forecast endpoints → forecast own-state continuations
→ forecast original VAE outputs → seal
→ native sources → native endpoints → native continuations → native VAE outputs
→ saved-evidence closure
```

The controller requires a seal receipt before native callbacks. The model adapter additionally guards actual native module entry and binds the returned digest to a real `ForecastTransaction`. Callback ordering alone is not quarantine. Each native stage runs under the sealed transaction; all numerical forecasts exist before the first native selected writer, primitive construction, suffix or decode.

`run_graph(plan, sources, bindings, kernels)` executes generated nodes with qualified registered kernels. It never evaluates Python expressions from a specification. The FLUX adapter restores actual consumed strides after NPY reload, supplies original SDPA dispatch or SwiGLU module calls in the native lane, and uses the original full projection geometry. Native targets consume their own formed source operands and actual native returns.

Each numerical lane advances its own states through 101 remaining denoiser transitions and the original VAE. There is no intermediate native refresh or later conditioning reset. Exact agreement pays the original weights and numerical primitives; it is not a cheaper surrogate.

The compiler derives execution geometry from the generated actions and declared continuation. For the six-action experiments, each context and lane has two source captures, four primitive endpoints, 606 suffix transitions, six VAE decodes and five common-parent Acts. The native lane also has one rewind and 542 transformer blocks including the two source blocks. Registered graph-node callbacks are counted separately from internal native kernels: SwiGLU executes four declared SiLU and four multiplication nodes, while the native module may group its internal work differently. These are declared adapter units, joined to runtime receipts rather than a claim that the framework has only eight kernels.

## What the graph comparison means

For ideal real arithmetic, attention forms

\[
A_{ab}=\Pi_aV_b,\qquad \Pi_a=\operatorname{softmax}(Q_aK_a^T/\sqrt d),
\]

whereas SwiGLU forms

\[
M_{ab}=\operatorname{SiLU}(G_a)\odot U_b.
\]

Both are conditionally linear in a payload under a fixed formed operator. Attention mixes token rows with normalized nonnegative coefficients. SwiGLU applies signed, unnormalized coefficients to channels independently. Their factors have different stages, widths and units.

The compiler derives conservative per-factor linearity properties from registered operations and graph dependencies. It distinguishes homogeneous linear payload action from an affine map with an offset and marks unresolved dependence as not proven affine. These properties concern the supplied ideal graph. They do not authorize replacing fused SDPA with explicit softmax, splitting an original GEMM or assuming finite BF16 additivity.

The four-corner mixed terms can be nonzero even in ideal arithmetic:

\[
J_A=(\Pi_P-\Pi_Q)(V_P-V_Q),\qquad
J_M=(\operatorname{SiLU}(G_P)-\operatorname{SiLU}(G_Q))\odot(U_P-U_Q).
\]

Actual finite primitive, endpoint and downstream remainders must be measured separately. A downstream interaction does not localize its origin to one primitive by itself.

## Evidence and scope

The current v3 core passes 26 installed, framework-free tests. Four prefix mechanics checks cover explicit cardinalities and the before-writer guard. Thirteen separate installed writer and campaign checks exercise actual source capture, NPY reload, live-layout restoration, native primitive calls and generated-plan validation; they also pass with the v3 core.

The executable experiments and independent saved reader are [primitive writer](../qualification/flux_primitive_family_writer.py), [campaign](../qualification/flux_primitive_family_campaign.py), [prefix loader](../qualification/flux_primitive_family_sources.py) and [v2 reader](../qualification/flux_primitive_family_audit_v2.py). The reader has seven installed checks, including a primitive error hidden in the projection's nullspace. It directly compares retained primitive and projection returns as well as endpoints and futures, and retains full native four-corner primitive interaction arrays remotely with hashes and RMS. The original reader and all historical results remain unchanged.

Finite run requests, sealed predictions, native images and independent audits are retained under `artifacts/mdb-nonlinear-families-20261107` in the workspace. Completed scientific counts belong to those receipts, separately from the mechanics checks.

The tool automates factorial construction, finite execution order and frozen rule bookkeeping. Registered adapters still supply model-specific capture and kernel contracts. Extracting operation graphs from traces, proposing useful contrasts, finding portable semantic names and predicting applicability outside the tested records remain open work.

The [nonlinear-family workflow and comparison](FLUX-NONLINEAR-COMPUTATION-FAMILIES.md) explains the two actual model campaigns. Their authenticated publications also import exact occurrences into `EquationCatalog` and `ComputationAtlas`, preserving the signed effects and native continuation independently of the context-level frozen rule.

The [Qwen autoregressive port](QWEN-COMPUTATION-FAMILIES.md) adds an explicit
`continuation.execution_kind="autoregressive"` geometry. For six roles and six
emitted choices it counts six prefills and thirty cached calls per lane, with
zero VAE decodes and zero public common-parent Acts. Module-hook interventions
remain separately named. Existing specifications without this field retain
their original serialized geometry and hashes. The model-specific worker and
its full-logit/KV forecasts require their own installed-runtime qualification.

With the separately qualified v4 command registry, an occurrence lookup is:

```sh
mdb atlas --atlas /absolute/publication/atlas find \
  --applicable-to /absolute/exact-context.json
```

The campaign's installed CLI example returns one exact hit and zero hits after changing clock or receiver. Query matching does not establish applicability to an untested context. `mdb capabilities` lists both family-program and atlas entry points with these limits. This CLI update leaves the frozen v2/v3 scientific runtime wheels unchanged.

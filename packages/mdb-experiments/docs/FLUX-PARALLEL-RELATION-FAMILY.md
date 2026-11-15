# Prospective parallel image relation family

This workflow studies a different computation family from the [sequential text writer](FLUX-PROSPECTIVE-WRITER-FAMILY.md). It starts at step 0, immediately before single block 4 in FLUX.2 Klein 4B distilled. Natural paired prompts reverse left/right while preserving the seed. The experiment predicts the selected writer and 101 subsequent denoiser transitions, followed by the original VAE, for five futures in each of six contexts.

The reusable scientific question is whether the joint residual/attention construction selected on historical relation evidence continues to move a new future farther toward the paired source direction than either operand alone. Exact numerical prediction and this causal effect prediction are evaluated separately.

The completed campaign predicted all 30 own-state image futures exactly. All twelve frozen signed-rank inequalities held: eight in four primary contexts and four in two separate probes. Independent audit compared 510 complete array fields, 42,000 boundary tensors and 12,480 scalar fields with zero mismatches. All 1,176 paired live-port content hashes and thirty PNG pairs matched. Source/recipient axes were nonzero in every context.

| Context | p(R) | p(A) | p(RA) | RA latent RMS to natural P |
| --- | ---: | ---: | ---: | ---: |
| Plain / 2119 | 0.779 | 0.871 | 0.111 | 0.306 |
| Plain / 2137 | 0.692 | 0.738 | 0.139 | 0.336 |
| Reworded / 2119 | 0.724 | 0.797 | 0.070 | 0.273 |
| Reworded / 2137 | 0.791 | 0.880 | 0.080 | 0.319 |
| Cone/cylinder / 2153 | 0.866 | 0.900 | 0.461 | 0.557 |
| Pyramid/sphere / 2153 | 0.901 | 0.916 | 0.403 | 0.558 |

Unblinded inspection finds RA reverses colored-object order in six contexts while R and A retain Q order. Shape collateral remains: a strongly rounded red object, mixed cone/cylinder forms and smoothing of natural P's rounded but faceted purple pyramid toward a cone-like RA form. The probes also show larger RA orthogonal latent error than either individual operand. These observations are separate from frozen rank predicates and numerical exactness.

## What the family contains

The source writer forms attention and MLP features in parallel from its normalized residual. The endpoint is the residual plus a gated output projection of the concatenated features. Capturing actual consumed ports makes the receiving state and source ancestry explicit.

The `image` slot is the current transformer image-token representation. P and Q still have bit-identical stored diffusion latents, positional IDs and schedules at this first-step boundary. Their natural prompts have created different working representations before any flow commit. The Act edits that working representation; original readout, scheduler and later projections carry its effect into subsequent latents while Q conditioning persists.

| Arm | Image residual | Image attention features | Image MLP and gate | Other state |
| --- | --- | --- | --- | --- |
| Q | Q | Q | Q | Complete Q, including an exact image self-copy Act |
| P | P | P | P | Natural complete P, a separate reference |
| R | P | Q | Q | Q text and persistent conditioning |
| A | Q | P | Q | Q text and persistent conditioning |
| RA | P | P | Q | Q text and persistent conditioning |

The held MLP/gate policy applies to the selected endpoint constructor. Later blocks recompute their own attention, MLPs and modulation as usual; working text and image state evolve independently in each future. Q conditioning is the persistent source for mixed branches.

P is not the missing all-P corner under a fixed Q receiver. Its text and conditioning differ as well. R, A and RA are held-return counterfactual endpoints; changing the upstream residual naturally would also recompute attention and MLP features.

The intervention addresses every image row through the typed image stream. It does not use activation energy, image inspection or a learned object localizer to choose rows. The original full text-plus-image projection shape is preserved before selecting image rows.

## Why this is informative

The first family transfers a small text update associated with color assignment. Its MLP follows attention. This family operates much earlier, writes the image stream and has parallel attention/MLP source formation. It tests whether the same experimental grammar works across these architectural differences without assuming the same useful operands or semantic meaning.

The development menu contains R, A and RA. The three historical relation contrasts selected RA by minimum mean signed final-latent coordinate: R 0.703267, A 0.825899 and RA 0.071035, with source P at zero and recipient Q at one. Baseline Q and natural reference P are excluded from selection. Those scores select a prospective experiment; they do not establish that every future image will have the intended object order.

## Mathematical construction

Ignoring rounding only for the structural description,

\[
X=(1+s)\odot\operatorname{LN}(R)+h,\quad
(Q,K,V,H)=\operatorname{Linear}_{in}(X),
\]

\[
A=\operatorname{Attention}(Q,K,V),\qquad M=\operatorname{SwiGLU}(H),
\]

\[
E(R,A;M_Q,g_Q)=R+g_Q\odot\operatorname{Linear}_{out}([A,M_Q]).
\]

The executable BF16 account preserves the original normalization, rotary processing, attention dispatch, operation order, full concatenation and original projection. It never replaces the original GEMM with separately multiplied attention and MLP weight slices merely because the real-arithmetic expression can be expanded that way.

For fixed gates and held MLP features, the ideal real-arithmetic endpoint is affine in R and A. Thus the four endpoint corners have zero mixed remainder in that idealization:

\[
J_E=E_{RA}-E_R-E_A+E_Q=0.
\]

The experiment saves the actual finite BF16 remainder as a full array. It also measures the corresponding remainder after the nonlinear continuation and decoder. A downstream interaction does not identify a new multiplicative gate: entry rounding and subsequent nonlinear transformations can both contribute.

The frozen effect coordinate is

\[
p(z)=\frac{\langle z-z_P,z_Q-z_P\rangle}{\|z_Q-z_P\|^2},
\]

using native final latents. The two predictions per context are `p(RA) < p(R)` and `p(RA) < p(A)`. Signed margins, source distances, orthogonal residuals, RGB effects and collateral are all retained. A smaller coordinate can overshoot the source; it is not equivalent to smaller distance or a correct object relation. Zero source/recipient axes are marked inapplicable without skipping any arm.

## How the experiment runs

1. Freeze the historical selection, six exact prompt/seed contexts, operation, full closure, address rule, five arms, predictions and horizon.
2. Capture twelve natural prefixes with public Saturn sessions. Stop before the selected block; no selected writer, continuation target or image is produced during prefix preparation.
3. Run the functional account from those input cuts. Compute all source writer ports, all mixed endpoints, all 30 complete own-state continuations and all 30 VAE forecasts.
4. Seal every numerical forecast with MDB's `ForecastTransaction`. Native selected writers, mixed projection calls, continuation blocks and decodes remain behind the release guard.
5. Execute original native writers and branches. Apply typed image Acts from common Q parents, verify Q full-state self-copy, retain natural P separately and exercise rewind closure. Every suffix consumes its own state; there is no native intermediate refresh or later conditioning reset.
6. Run an independent CPU reader on the completed publication. Authenticate the full inventory and source-prefix lineage; compare saved arrays, endpoint arithmetic, descriptors, every boundary, final state and native images. Original learned projections remain supplied products in this reader.
7. Evaluate the frozen inequalities on every context, retaining misses and undefined axes. Inspect actual images separately from latent-axis evidence.

The source implementation is [flux_parallel_family_sources.py](../qualification/flux_parallel_family_sources.py), the exact writer is [flux_parallel_family_writer.py](../qualification/flux_parallel_family_writer.py), the producer is [flux_parallel_family_campaign.py](../qualification/flux_parallel_family_campaign.py), and the independent reader is [flux_parallel_family_audit.py](../qualification/flux_parallel_family_audit.py). Each runs against installed public packages; no private Saturn imports or sibling-checkout import patches are used.

## Evidence coverage

Each source writer has 28 named live port descriptors, including actual input, Q/K/V, attention, MLP, merge and projection values. Ten source ports and eight first-consumer ports retain complete arrays. Remaining ports retain logical-byte hashes and actual live layouts. Retained snapshot layouts are reported separately because cloning a strided view can change its physical layout.

Each future retains 102 boundary records including its entry state, complete final typed state and full final latent, decoded and RGB arrays. Boundary records contain tensor hashes, layouts, statistics and samples; they are not full activation arrays at every boundary. All raw evidence remains on the publication volume. Compact collections include native and forecast PNGs, audits, scripts, receipts and 311 parameter retrieval recipes. The run does not export another pretrained weight bank.

## Engineering qualification and retained failures

The writer passed eight installed tests, the corrected prefix guard passed nine tests against the actual public adapter, and the initial saved reader passed eight installed tests. The repaired reader passed eleven installed tests, including an authenticated serialized receipt and a real public Session save → reload → Act regression. All passed Ruff F/I checks. These mechanics checks do not replace the real-model comparison.

The first source job stopped because the guard expected `ExecutionPoint.index=0` for project, while the public adapter emits `None`. It had encoded twelve prompts but executed no transformer transition or selected writer and generated no image. An explicit instrumentation amendment preserved the exact prompts, seeds, arms and scientific predictions. The failed job, original rule and launcher remain retained. The corrected prefix completed successfully.

The first main staging attempt also rejected an incorrect scheduler declaration that counted sequential pipeline components as two model loads. It submitted no job. The admitted contract declares one pipeline and records its sequential transformer/VAE component hydration separately.

The first saved-only audit attempt stopped before result reading because its launcher omitted Pillow. The corrected launcher pins Pillow 13.0.0, matching the producer runtime. It reruns the same frozen reader on the same completed evidence; no model execution or scientific rule changes are involved. Both audit admission records and the failed launcher are retained.

The next reader attempt reached typed Acts and exposed a cross-origin layout assumption. A contiguous tensor reloaded from NPY can have a different leading stride from the producer endpoint and the recipient-preserving landed write. The repaired reader binds the parameter to the authenticated reload convention, the write to the actual receipt/CAS and the result layout to its receiver. It additionally checks address, full ones mask, mode and integer dose. The original reader and failed attempt remain retained; the completed GPU evidence is reused.

## Reproduction assets

The campaign directory is `artifacts/mdb-parallel-relation-family-20261107` in the research workspace. `family-rule-v1.json` preserves the original freeze; `family-rule-v2.json` binds the public-point guard amendment. `prepare.py` stages and submits finite public-worker leases. `evidence.py` authenticates compact collections and stages the saved-only reader. Prefix, model and CPU audit each have separate request, payload, submission, publication and custody records.

In this campaign the numerical transaction's `model_identity` envelope field contains the campaign schema label. The actual checkpoint identity, `a09191c5ff813af77bf89d8e4b9a07237befb67c1aef3e722322d3491250fcd0`, is independently and strictly bound through source roots, typed StateCuts, adapter validation, runtime/parameter metadata and the sealed inventory. Use those bindings when checking checkpoint identity. The original transaction remains immutable.

## Completed evidence and symbol table publication

The model job is `job-5d8ec6e7eb18` and the successful independent saved-only CPU audit is `job-c543b2d5dcd0`. They took 583.43 and 158.58 seconds, respectively. The publication retained 7,782,560,708 raw model bytes across 1,768 files; native and audit compact collections total 21,382,940 bytes. Audit report SHA-256 is `348ae43f505dc4d5e131445d573418014bc733fdb5b1ac488b7b3613664c2fe3`.

`family-conclusion-v2.json` binds the frozen rule, transactions, complete coverage, separately recomputed predicates, corrected qualitative observations and publication. Its predecessor and initial visual notes remain retained; the correction distinguishes P's planar pyramid faces from the smoother RA shape. `family-publication-01` contains thirty EquationCatalog/ComputationAtlas occurrences and six curated family records. Thirty exact-context queries match; sixty changed-clock/receiver queries abstain. Law ID is `flux.single.parallel-image-merge.bf16-reference.v1`; route ID is `flux.step0.single4.image-stream-to-latent.v1`.

Read the qualified occurrences through the installed public API:

```python
from mdb.computation_atlas import ComputationAtlas

atlas = ComputationAtlas(
    "artifacts/mdb-parallel-relation-family-20261107/family-publication-01/atlas"
)
occurrences = atlas.find(law_id="flux.single.parallel-image-merge.bf16-reference.v1")
assert len(occurrences) == 30
```

Run this metadata-only query from the research workspace using `common exec`. The recorded count is for this published atlas; it does not assert applicability to an unseen input. New fresh-context testing needs a new frozen rule and source-prefix campaign. Replaying the existing cases is an engineering reproduction.

The static evidence view at `family-publication-01/view/index.html` shows all thirty native targets and sealed forecasts, p values, source distances, orthogonal error and the frozen rule. In the research workspace it is served at `http://127.0.0.1:8882/family-publication-01/view/index.html` while the local viewer server is running. The view can also open directly as a file.

The 42 native/functional A-port capture-layout differences are explicitly retained. Native A is the actual merge-operand slice; functional A is the flattened attention result before concatenation. They have equal logical bytes but different live strides. Same-boundary layouts and original consumed projection geometry match; no layout difference is silently recast as content equality or a dispatch guarantee.

For the mathematical comparison and metadata automation prototype, see [computation family protocols](COMPUTATION-FAMILY-PROTOCOLS.md). Numerical equality, useful operand transfer, receiving-state scope and semantic interpretation remain separate fields in the computation record.

The independent Astra comparison is saved at `artifacts/mdb-parallel-relation-family-20261107/advisor-final-family-comparison-v1.md`, SHA-256 `479a9513463ececf2b954d35210a16bf09aff07f5e755a69c509ca6a55cb2cb0`. It confirms the bounded prospective result, source/receiver interpretation and common formation → affine held-endpoint → nonlinear continuation structure. It recommends generating typed experiment/audit programs from the supplied protocol metadata while preserving capture origins and applicability scope.

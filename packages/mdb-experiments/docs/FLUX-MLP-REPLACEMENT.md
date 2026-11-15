# Feeding a full predicted MLP field to the native FLUX consumer

This assay extends the [sampled MLP-entry forecast](FLUX-SINGLE-MLP.md) to every image token and MLP neuron at one selected block. It predicts two complete M fields, seals them, and supplies those values to the actual native fused projection. Measured-versus-predicted pairs test the numerical replacement, while a fixed token permutation tests placement.

**Status:** the seed-71 experiment completed in `job-68295609457d`. Both complete fields were sealed before any current-target entry, consumed by the real native projection, and followed through all nine image futures. The independent saved-only audit reports valid mechanics and qualified native replay. Low/high fields match 99.6777%/99.7026% of native BF16 values exactly, with relative RMS errors 0.0241937%/0.0235781%. This is one fresh context and two dependent source states.

The first job, `job-9999f2088d3b`, hit its 900-second limit before the full-field seal. Its [incomplete execution record](../../../../artifacts/mdb-flux-mlp-replacement-20261104/incomplete-attempt-01.json) retains the complete raw log and makes no forecast claim. The successful attempt adds phase markers and stack diagnostics under a reviewed finite lease; its source and scientific plan remain bound separately.

The native MLP still executes before its image-side activation slice is overwritten. The assay therefore tests whether a sealed prediction can serve the native consumer; it does not remove the MLP, skip its computation or demonstrate a speedup. Native prefix state, attention and text remain part of the computation. Final RGB images are native continuations, not independently predicted pixels.

## Frontier and complete predicted field

The specimen is distilled FLUX.2 Klein 4B, CUDA BF16, physical batch 1, 512×512 output, at logical denoising step 3, single block 6. Seed 71 uses:

> A green cone to the left of a yellow cylinder on a plain white background. Two separate objects, simple studio product photograph.

One encoding and one noise draw construct two own-history trajectories. Direct post-single0 image alignment creates matched image/different-text states, followed by native blocks 1–5. Both I and T may differ at the selected pre-single6 frontier. This is not a fresh K/V-equivalence assay or an isolated text-bank cause.

The allowed builder consumes 108 native transitions and four Acts before sealing. Earlier-timestep executions of module 6 are allowed. The current target—logical step 3, single block 6—is excluded before the full-field seal. The forecaster receives the observed incoming image, time modulation, all paired native MLP weight rows and optional bias. Native current-site normalization, attention and MLP outputs are not predictor inputs.

Each predicted activation field has shape **[1,1024,9216]**, or 9,437,184 values per role. The two roles are called low/zero and high/text; these labels describe histories, not zero-valued tensors. They are dependent sources in one prompt/seed context.

## The explicit numerical reference

For each image token x, the 3,072-channel LayerNorm has no learned affine parameters and uses population variance with ε=10⁻⁶. Time modulation supplies shift s and scale c. Let B mean FP64→FP32→round-to-nearest-even BF16, stored as a lossless FP32 widening:

```text
n = B((x − mean(x)) / sqrt(mean((x − mean(x))²) + ε))
h = B(B(B(1 + c) × n) + s)
u = B(h × W_uᵀ + b_u)
v = B(h × W_vᵀ + b_v)
M = B(B(SiLU(u)) × v)
```

The paired weights are the complete native fused rows `3H:3H+2I`, split into u/v halves, with H=3,072 and I=9,216. Inputs and weights begin as exact FP32 widenings of actual BF16 values. Normalization and linear sums use independent NumPy FP64 arithmetic with the declared BF16 boundaries. Linear calculation is tiled over 256 paired neurons to bound matrix widening. It does not allocate or retain a full unquantized FP64 activation field.

This is an explicit rounding reference, not a bit-exact CUDA LayerNorm/GEMM/SiLU emulator. A small fixed 8-query/32-neuron witness is also retained for stage diagnosis; the consumed payload is the **complete field**, not that sample. Full-field mismatches, RMS, maxima and signed errors remain observations, without a numerical quality threshold that discards futures.

The public transaction persists both fields and their bindings before current-target execution. An external source-bound guard and timeline enforce the declared entry exclusion. This is trusted Python dataflow with pinned source and parameter identities, not hermetic process isolation or an independent raw-prompt model.

## Lowering to the native projection

After sealing, two native source calculations expose the measured fields and unchanged recipient operands. The predicted field is converted to native BF16, with shape, device and strides supplied by the qualified port. Reference tensor **values** do not contribute to the payload. A round-trip bit check verifies that lowering preserves the sealed BF16 values exactly.

During each edited direct block call, native attention and native MLP compute normally. A pre-hook on the fused `to_out` projection then replaces only image-side MLP features in the attention/MLP concatenation. Recipient image attention, all text features, residual R and gate g remain native. The actual projection and update consume the substituted field:

```text
image_out = BF16(R + BF16(g × native_projection[A_native; M_selected]))
```

The resulting image is installed at the post-single6 boundary through a public typed image Act, preserving the remaining state. Later blocks, readout, scheduler and VAE are native. This separates payload custody, actual consumer execution and downstream effects from a claim that the predictor replaced the whole model.

## All nine branches

| Branch | Recipient history | Image-side MLP field | Token placement |
|---|---|---|---|
| zero | low | native low | original |
| text | high | native high | original |
| sham | low | measured low | original |
| low.predicted | low | sealed predicted low | original |
| M.measured | low | measured high | original |
| M.predicted | low | sealed predicted high | original |
| M.measured.roll | low | measured high | roll 512 |
| M.predicted.roll | low | sealed predicted high | roll 512 |
| high.predicted | high | sealed predicted high | original |

`low.predicted` versus zero and `high.predicted` versus text test same-history numerical substitution. `M.predicted` versus `M.measured` compares forecast error with a matched measured high-M transfer on the same low recipient. Their rolled counterparts preserve the field values and energy while changing token placement along image-token axis 1. **Roll 512 is a placement contrast, not a guaranteed destructive null.** Similar, weaker or stronger effects are all retained.

High M encodes the whole observed high history. The comparison does not isolate a text-only cause, assign semantic names to neurons or establish an object address. Endpoint distances and image observations remain separate from numerical forecast fidelity; no rank or error threshold is a semantic score.

The finite census is one model load, encoding and noise draw; 245 native transitions; 11 Acts; 256 receipt emissions; nine direct block calculations; and nine final PNG/VAE calls. The builder contributes 108 transitions/four Acts, and the panel 137 transitions/seven image Acts. Direct calculations include two native references and seven edited calls; they are not nine whole-model forward passes. Nine futures are dependent branches of one seed-71 history.

## Observed prediction and continuation errors

The independent auditor reconstructs all 18,874,368 sealed activation values exactly from saved pre-entry operands and weights. That checks the declared reference computation. Comparison with native CUDA retains 30,418 low-field and 28,068 high-field mismatches, rather than claiming a bit-exact kernel emulator.

| Complete image M field | Low history | High history |
|---|---:|---:|
| Exact BF16 values | 9,406,766 / 9,437,184 | 9,409,116 / 9,437,184 |
| RMS error | 0.0000319934 | 0.0000259889 |
| Error / native RMS | 0.0241937% | 0.0235781% |
| Maximum absolute error | 0.015625 | 0.015625 |

The fixed sample is also informative: the high history matches every sampled stage exactly, while the low history has two LayerNorm and modulation mismatches, one u mismatch, two v mismatches and three of 256 M mismatches. The earlier seed-67 sample's exact normalization does not extend universally to this fresh state. Full-field coverage exposes errors that a small successful sample can miss.

| Predicted branch versus matched native/measured branch | Postblock image L2 | Final latent L2 | Final RGB MAE (0–255) | Maximum RGB difference |
|---|---:|---:|---:|---:|
| low.predicted versus zero | 10.2453 | 3.21708 | 0.161407 | 10 |
| high.predicted versus text | 8.15373 | 2.58497 | 0.110246 | 7 |
| M.predicted versus M.measured | 8.29539 | 3.28410 | 0.160489 | 17 |
| M.predicted.roll versus M.measured.roll | 8.48872 | 3.20076 | 0.161781 | 14 |

Measured high-M transfer changes the low recipient by image L2 5,860.52, final latent L2 51.2959 and RGB MAE 2.76200. The corresponding prediction-versus-measurement discrepancies are 0.1415%, 6.4023% and 5.8106% of those effect norms. These diagnostic ratios use different observables at each boundary; they are not a fraction of semantics explained or a numerical guarantee for future inputs. Errors remain measurable after native continuation.

Rolling the same measured high field by 512 tokens changes the final image relative to its unrolled counterpart by RGB MAE 3.80598; the predicted-field pair gives 3.78470. Relative to the low baseline, rolled measured M has a larger RGB distance than unrolled M, 4.92912 versus 2.76200. Yet it is closer to the high reference, 19.7296 versus 21.1253. Placement matters, and this particular roll is not simply a destructive control or an identified object-address move.

![Nine native futures; each image tile retains its original pixels](../../../../artifacts/mdb-flux-mlp-replacement-20261104/final/native-futures.png)

The native-prefix image/text comparisons and unchanged-sham latent/PNG checks all hold exactly. Forecast accuracy and intervention direction do not participate in that replay qualification. Arm-blind image descriptions, sealed before unblinding, identify the native high and high.predicted foregrounds as more coherent; all nine retain green-left/yellow-right forms and background artifacts. These are informal visual observations, without a calibrated semantic evaluator.

The [saved-cut diagnostic](../../../../artifacts/mdb-flux-mlp-replacement-20261104/continuation-error-trajectory.json) finds 16 aligned boundaries per matched pair: the post6 input and 15 native transitions. Only the start and final full cuts exist. All 14 intermediate full cuts are absent, including from the builder/preparation stores; the root store has 20 full cuts total. Endpoint image/text and final latent/RGB metrics join the audit exactly. Receipts and pinned adapter source establish the declared native continuation path, without an independent kernel trace, but cannot supply absent tensors; a complete layer-by-layer error-growth curve is not established by this capture policy.

## Operational diagnostics and next capture improvements

The successful worker's phase ledger places 392.03 seconds in source construction and 89.30 seconds in preparation. Sampled stacks show file/directory synchronization inside public LocalStore publication. Those samples identify waits without attributing every second or establishing that a different disk would accelerate this model.

The experimental [scalar storage probe](../qualification/profile_flux_storage.py) runs through public MDB bootstrap and mrun, with NumPy required by the public mrun import closure, and zero models, tensors or CUDA calls. Its [retained assessment](../../../../artifacts/mdb-flux-mlp-replacement-20261104/profile-result-v2.json) separates initial setup failures from the later executed probe: `job-ba3dd0306970` times one SSD-path file fsync at 2.612 ms, then `os.link` returns EPERM before a complete StateCut save. HDD is not entered in that executed revision. This is a path-specific publication refusal, not an SSD-versus-HDD speed measurement. The subsequent tool revision retains independent per-case filesystem errors and attempts the other case; four tiny checks pass, but that revision is not fleet-qualified for both roots. No durability operation is removed.

Two tool improvements follow directly: paired per-cursor continuation observations with selected exact cuts, and bounded binary result transport with manifest-addressed retrieval. The first would localize propagation without saving every full branch history. The second would avoid returning the complete 810.8 MB archive through a 1.081 GB raw text log. Both are recorded in `saturn/request.jsonl`; neither is claimed implemented by this assay.

## Saved evidence and reproduction

The [current request](../../../../artifacts/mdb-flux-mlp-replacement-20261104/run-03/request.json) is bound by digest `018be3c22fdbdea8c1e136e50918c062281e3fb7a420d3cfcd021a5a70a505b7`. The [equations and native lowering](../qualification/flux_mlp_replacement.py), [worker](../qualification/flux_mlp_replacement_campaign.py), [launcher](../qualification/submit_flux_mlp_replacement.py), [saved-only auditor](../qualification/analyze_flux_mlp_replacement.py) and [publisher](../qualification/publish_flux_mlp_replacement.py) keep prediction, consumption and independent checking distinct.

The installed-wheel baseline passed 100 focused mechanics and regression tests in 18.19 seconds. After adding timing instrumentation, the 16 mathematical, nine-branch and audit checks passed again in 7.13 seconds; Ruff passed. These are engineering checks with tiny native CPU modules and a synthetic image consumer, not additional real-model specimens. [Qualification v3](../../../../artifacts/mdb-flux-mlp-replacement-20261104/focused-qualification-v3.json) preserves source revisions and the distinction.

The [Astra staged review](../../../../artifacts/mdb-flux-mlp-replacement-20261104/prerun-review-03.json) verifies source/package pins and the unchanged science. The [Luna long-run review](../../../../artifacts/mdb-flux-mlp-replacement-20261104/long-run-review-luna.json) checks the finite 1,200-second cooperative worker budget and hard 1,800-second mrun timeout. Periodic traceback output stops before success or failure archive transport to preserve framing. The original timeout remains an incomplete execution; no error threshold or scientific claim is being relaxed.

Use absolute paths and an installed-wheel Python from outside the workspace. After collection, auditing saved evidence does not launch a model:

```sh
MDB=/absolute/path/to/saturn-pub/packages/mdb
PY=/absolute/path/to/qualified-environment/bin/python
RUN=/absolute/path/to/mdb-flux-mlp-replacement-20261104/run-03
cd /tmp
"$PY" -I "$MDB/qualification/analyze_flux_mlp_replacement.py" \
  "$RUN" /new/path/analysis.json
```

A separate reproduction uses one bounded, strictly preflighted mrun lease with no automatic retries. Use a fresh run directory; staging itself does not admit a model job:

```sh
COMMON=/absolute/path/to/common/bin/common
FRESH_RUN=/absolute/path/to/new-run
"$COMMON" exec "$PY" -I "$MDB/qualification/submit_flux_mlp_replacement.py" \
  "$FRESH_RUN" --stage-only
"$COMMON" exec "$PY" -I "$MDB/qualification/submit_flux_mlp_replacement.py" \
  "$FRESH_RUN" --submit-staged
# After the returned job finishes:
"$COMMON" exec "$PY" -I "$MDB/qualification/submit_flux_mlp_replacement.py" \
  "$FRESH_RUN" --collect JOB_ID
```

The [independent audit](../../../../artifacts/mdb-flux-mlp-replacement-20261104/run-03/analysis.json), [published outcomes](../../../../artifacts/mdb-flux-mlp-replacement-20261104/final/result.json), [sealed image descriptions](../../../../artifacts/mdb-flux-mlp-replacement-20261104/blinded-observations.json) and [unblinded join](../../../../artifacts/mdb-flux-mlp-replacement-20261104/unblinded-visual-record.json) bind the actual results. The [Astra postrun review](../../../../artifacts/mdb-flux-mlp-replacement-20261104/postrun-review.json) independently checks source/seal, complete-field errors, consumer joins and endpoint comparisons, with no blocking integrity issue. Three [curated candidate-family records](../../../../artifacts/mdb-flux-mlp-replacement-20261104/computation-records.json) contribute 11 numerical rows and six source-qualified addresses to the [offline evidence index](../../../../artifacts/mdb-flux-mlp-replacement-20261104/evidence-v1/provenance.json); run, verify and query each return 0. This is explicit curation, not automatic family induction or atlas ingestion.

The full archive is 810,756,640 bytes, SHA-256 `df12080744b8b6c94f9830930578f474cf5a235e7a95d97bf6d06e7de216d906`; complete raw logs are retained as gzip. The worker reached its pre-export summary in 740.34 seconds; the complete runner interval was 860.3 seconds, peaking at 9,672 MB VRAM and 6,320.6 MB RSS. After the storage diagnostic changes, 20 focused installed-wheel checks pass in 7.28 seconds, including four scalar diagnostic checks; this does not add real-model specimens or qualify a two-root storage comparison.

This advances a symbol-table record from selected arithmetic prediction to a complete predicted feature field connected to its actual consumer and native future. Applicability remains bound to the observed entry state, timestep, model parameters and recipient operands. Independent attention prediction, stacked predicted blocks, broader seeds/prompts/MLPs, and reusable semantic names remain separate experiments. The earlier sampled forecast and seed-53 contraction reports remain unchanged.

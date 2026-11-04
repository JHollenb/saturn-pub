# Predicting attention and MLP fields before the native consumer

This assay extends [full-field MLP replacement](FLUX-MLP-REPLACEMENT.md) into the parallel attention branch. It predicts complete image-side A and M fields from the observed incoming state and actual weights, seals them before the selected native block, and supplies those predictions to the real fused output projection. Native continuation provides the final latent and image observations.

**Status:** fresh seed-73 `job-947395d68069` completed all nine native futures. Full attention error is 0.126–0.135% of native RMS; full MLP error is 0.0234–0.0258%. The predicted/measured combined transfer differs by 0.2003 RGB levels, against a measured transfer effect of 4.2557 levels. The saved-only audit reports valid mechanics and qualified native replay; terminal generality is unassessed. Thirty focused installed-wheel checks and nineteen saved-audit checks pass.

## Specimen and allowed information

The model is distilled FLUX.2 Klein 4B, CUDA BF16, physical batch one, 512×512 output and four denoising steps. The target is logical step 3, single block 6. Seed 73 uses the same green-cone-left-of-yellow-cylinder prompt as the prior seed-71 assay. One fresh model load, encoding and noise draw build two dependent histories; both incoming image and text registers may differ at the target.

The forecaster receives the complete observed incoming text/image registers, time modulation, actual fused Q/K/V and paired MLP weight rows, optional biases, learned Q/K RMSNorm weights and epsilons, and upstream positional cosine/sine arrays from the same typed cut. It receives no current-target native normalization, Q/K/V, attention or MLP activations. Earlier-timestep block-6 executions in the 108-transition source builder are permitted; the excluded target is the specific current-step computation.

With H=3,072 channels, 24 heads of dimension D=128, 512 text positions and 1,024 image positions, attention reads all 1,536 keys and values for each image query. Its complete predicted image A field is [1,1024,3072]. The image MLP field is [1,1024,9216]. Predicting both roles covers 25,165,824 BF16-reference activation values.

## The attention reference

Let R concatenate text before image. Let B mean FP64/FP32→FP32→round-to-nearest-even BF16, retained as an exact FP32 widening. Population LayerNorm has no learned affine parameters. The shared modulated input is

```text
n = B((R − mean(R)) / sqrt(mean((R − mean(R))²) + epsilon))
h = B(B(B(1 + scale) × n) + shift)
[Q,K,V] = B(h × W_qkv.T + bias_qkv)
```

Normalization reductions and linear sums use independent FP64 arithmetic. Each Q/K head is then normalized over its 128 channels using the actual learned weight:

```text
Qn = B(Q / sqrt(mean(Q²) + epsilon_qk) × weight_q)
Kn = B(K / sqrt(mean(K²) + epsilon_qk) × weight_k)
Qr = B(float32(Qn × cos) + float32(rotate(Qn) × sin))
Kr = B(float32(Kn × cos) + float32(rotate(Kn) × sin))
```

`rotate` interleaves negative odd and even coordinates as the installed FLUX RoPE implementation does. The cosine/sine values are native upstream FP32 operands, rather than newly calculated positional approximations. For every image query and head,

```text
P = softmax(Qr × Kr.T / sqrt(D), over all text and image keys)
A = B(P × V)
```

The scaled product, stabilized softmax and value accumulation use FP64. Probabilities are not quantized to BF16 in this reference. Native CUDA SDPA can use different reduction order and accumulation; this is an explicit numerical reference, not a bit-exact kernel emulator. Query blocks of 64 and projection row tiles of 256 bound temporary memory. Full attention matrices are not retained.

The MLP uses the separately qualified staged-BF16 SwiGLU reference from [the earlier workflow](FLUX-MLP-REPLACEMENT.md). The sealed field payloads contain full image A and M values; fixed query/key stage samples and predicted text/image attention mass are diagnostics. Predicted mass is not a measured native routing distribution or semantic token attribution.

## Seal, substitution and nine futures

The public forecast transaction binds all inputs, fields and branch recipes before any current-target entry. Source-pinned module guards reject transformer or VAE entry during prediction. This is trusted Python dataflow with external entry counters, rather than hermetic process isolation.

After sealing, two native source calculations expose measured A/M for controls. Edited direct calls replace only the selected image feature slices at the actual `to_out` input. Recipient text, residual and output gate remain native. The fused projection and gated residual update consume the selected values. The resulting image register is installed through a public typed image Act before the unchanged native suffix runs.

| Branch | Recipient | Image feature source |
|---|---|---|
| zero | low | native low A/M |
| text | high | native high A/M |
| sham | low | measured low A/M |
| low.predicted.AM | low | predicted low A/M |
| high.predicted.AM | high | predicted high A/M |
| AM.measured | low | measured high A/M |
| AM.predicted | low | predicted high A/M |
| A.predicted | low | predicted high A, native low M |
| M.predicted | low | native low A, predicted high M |

Same-history drop-ins test fidelity. `AM.predicted` versus `AM.measured` compares approximation with the matched measured intervention on one low recipient. The single-component predicted branches mix source-feature transfer with approximation error; without matched measured A-only and M-only branches, they are exploratory decomposition probes, not pure causal A-versus-M attribution.

Native attention and MLP still execute before overwrite. Native text-side computation, projection, residual/gate and later blocks remain essential. No compute skipping, speedup, whole-block independent prediction or raw-prompt image forecast is claimed.

The complete census is one load/encoding/noise, 245 native transitions, 11 Acts, 256 receipt emissions, nine direct block calculations and nine VAE/PNG futures. This comprises 108 builder/preparation transitions and 137 panel transitions, including two native prefixes and nine 15-transition suffixes.

## Per-cursor continuation observations

All fields and comparison recipes are sealed before reference suffixes run. Suffix execution order is fixed separately from direct-call order: zero, text, AM.measured, sham, low.predicted.AM, high.predicted.AM, AM.predicted, A.predicted, M.predicted. Each branch records its actual post6 input and all 15 transition exits, giving 16 aligned boundaries.

Three reference trajectories retain full lossless BF16 image/text/latent/readout arrays when present. Candidate intermediate states retain native descriptors, signed full-field metrics against already available references, and 64 fixed C-order coordinate witnesses per slot. Coordinates follow a source-independent linspace rule; reference trajectories never correct the sealed predictions.

The archive supports independent full-array checks for references and independently retained candidate endpoints. Where intermediate candidate arrays are absent, their full norms remain worker observations; the coordinate witnesses support independent sampled checks only. No interpolation supplies missing arrays or complete StateCuts.

Phase labels distinguish working image/text from stale post-readout registers, constant stored Z3 from the scheduler's committed Z4 latent, and native readout noise at scheduler entry. A constant stored latent is not evidence of error repair. Changes in error norm can identify propagation patterns, while semantic repair and cancellation require additional evidence.

## Complete field and consumer results

The [accepted audit](../../../../artifacts/mdb-flux-attention-replacement-20261104/run-01/analysis.json) independently rebuilds every sealed A/M value exactly from saved permitted inputs. These are 25,165,824 activation values across the two histories. Comparing the sealed reference with the subsequently observed native fields gives:

| Field | History | Values | Exact BF16 agreement | Error RMS / native RMS | Maximum absolute error |
|---|---|---:|---:|---:|---:|
| Attention A | low | 3,145,728 | 86.6872% | 0.126308% | 0.009765625 |
| Attention A | high | 3,145,728 | 84.0944% | 0.134634% | 0.015625 |
| MLP M | low | 9,437,184 | 99.6733% | 0.025787% | 0.0078125 |
| MLP M | high | 9,437,184 | 99.7236% | 0.023441% | 0.015625 |

The signed field error is native minus sealed. Use the explicit `native_relative_rms` field for the table's denominator: the inherited `relative_rms` divides by sealed-reference RMS. Attention's low RMS discrepancy does not imply bit-exact agreement; the mismatch counts remain 418,785 and 500,347.

The transaction seal completed at 108.0596 seconds, with zero current-target entries beforehand. Seven edited consumers exactly join the declared complete field payloads while preserving undeclared text slices, recipient residuals and gates. Sham replay is exact. The [publication](../../../../artifacts/mdb-flux-attention-replacement-20261104/final/result.json) retains all nine rows and all signed outcomes.

| Matched comparison | Post6 image L2 | Final latent L2 | RGB MAE, 0–255 | Maximum RGB difference |
|---|---:|---:|---:|---:|
| Low predicted drop-in vs low native | 25.2149 | 3.24349 | 0.199284 | 8 |
| High predicted drop-in vs high native | 24.2909 | 2.86211 | 0.152012 | 21 |
| Predicted high A/M transfer vs measured high A/M transfer | 25.6490 | 3.29150 | 0.200255 | 10 |
| Measured high A/M transfer vs low baseline | 7,965.6449 | 63.34265 | 4.255660 | see full row |

For the matched transfer, approximation discrepancy is 0.3220% of the measured immediate image effect, 5.1963% of its final latent effect and 4.7056% of its RGB MAE effect. These use different observables and denominators, rather than a single interchangeable accuracy score. Final pixels are native observations after local predicted-field substitution.

![All nine native futures](../../../../artifacts/mdb-flux-attention-replacement-20261104/final/native-futures.png)

These images come from deliberately altered conditioning histories and retain background artifacts; this assay does not isolate their cause. Sealed arm-blind descriptions identify the native high history and its predicted drop-in as the two smoother foregrounds after the mapping is revealed. This is assistant visual observation, not a calibrated object scorer.

## What the continuation reveals

The [saved trajectory analysis](../../../../artifacts/mdb-flux-attention-replacement-20261104/trajectory.json) follows actual boundaries without mixing working registers with stale slots. In the matched transfer, image discrepancy rises from 0.1027% of reference RMS after block6 to a 1.5669% peak after block17, then 1.5202% at readout entry. Text is initially bit-exact and first differs after the next native block, at 0.0839% of reference RMS. Its discrepancy peaks at 0.7529% after block18 before declining to 0.6760% at readout entry. These intermediate full metrics are worker observations with fixed-coordinate checks; they locate the first observed spread of the image-only perturbation into working text at block7, without isolating a particular attention head or path.

Readout noise differs by 1.2751% of reference RMS, and the committed latent by 0.9648%. Cross-slot reductions in these percentages do not establish semantic repair. The native suffix spreads and transforms the perturbation while the predicted and measured transfer still produce closely matched final images.

![Per-slot forecast discrepancy through native continuation](../../../../artifacts/mdb-flux-attention-replacement-20261104/trajectory-errors.png)

The independent review checks 282 full reference slot comparisons and 423 fixed candidate/reference witness records across nine 16-boundary trajectories. Full intermediate candidate norms remain source-bound worker observations. Retained endpoints support independent complete-array checks.

## Local additivity and later nonadditivity

The [posthoc factorial](../../../../artifacts/mdb-flux-attention-replacement-20261104/factorial/result.json) uses the four actual predicted-payload arms in the same low recipient. With D_A=Y_A−Y_0, D_M=Y_M−Y_0 and D_AM=Y_AM−Y_0, define C=D_AM−D_A−D_M. It is calculated after execution from retained arrays; it was not a frozen forecast.

At the immediate image output, ||C||₂/||D_AM||₂ is 1.3365%. The fused projection is linear before finite-precision rounding, gating and residual addition, so approximate local additivity is expected. A-only and M-only writes have an angle of 86.78 degrees. At the final latent, the same factorial remainder is 58.80% of the combined effect norm; in RGB it is 58.11%. The suffix makes this perturbation response strongly nonadditive. These are L2 norm ratios, not percentages of explained variance.

The M-only write projects −0.5112 onto the incoming high-minus-low image direction, and the combined write −0.7025. Opposition along that state direction is an exploratory numerical observation, not evidence of semantic correction. Component rankings also disagree across consumers: A-only is closer to the high history than AM in latent L2 but farther in RGB MAE; M-only is farther from high than the low baseline in latent while closer in RGB. Without measured component-only controls, retain these as behavior of the declared predicted interventions.

## Naming and searching these records

Three curated occurrences use generic IDs: `flux.single6.attention-entry.p1-s73`, `flux.single6.mlp-entry.p1-s73`, and `flux.single6.parallel-consumer-replacement.p1-s73`. Applicability includes checkpoint identity, logical step, block, incoming full state, modulation, learned weights, positional operands, numerical contract and native continuation. The seed71 and seed73 MLP occurrences share a curated family label while retaining their different histories and coverage.

The [computation records](../../../../artifacts/mdb-flux-attention-replacement-20261104/computation-records.json) connect qualified source addresses, complete fields, consumers, effects, continuation and counterexamples. Offline Saturn evidence indexing makes these occurrences searchable. Assigning a shared family label is explicit curation; automatic family induction and automatic atlas ingestion remain separate work.

The next discriminating experiment adds measured A-only and M-only controls and full arrays at selected candidate boundaries. It can locate where near-additive local writes become nonadditive downstream, then test that transition under another clock or prompt. No object name is required to record and reuse that computation.

## Execution and saved-only audit

The [attention predictor](../qualification/flux_attention_reference.py), [campaign](../qualification/flux_attention_replacement_campaign.py) and [launcher](../qualification/submit_flux_attention_replacement.py) are explicit qualification sources using installed public MDB/Saturn packages. The [staged request](../../../../artifacts/mdb-flux-attention-replacement-20261104/run-01/request.json) freezes seed, branches, capture policy, source pins, resources and finite transport limits.

Use an installed-wheel environment from outside the workspace. Staging never submits:

```sh
PY=/absolute/path/to/qualified-environment/bin/python
MDB=/absolute/path/to/saturn-pub/packages/mdb
RUN=/absolute/path/to/a/new-run-directory
cd /tmp
# Configure caches for the qualified Beast worker, before staging:
export MRUN_URL=http://192.168.68.68:9025
export MDB_WORKER_UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache
export MDB_WORKER_TORCH_HOME=/mnt/hdd8tb/mdb/torch
export MDB_WORKER_MODEL_ROOTS=/mnt/big/llm-models
"$PY" -I "$MDB/qualification/submit_flux_attention_replacement.py" "$RUN"
# After inspecting the staged request:
"$PY" -I "$MDB/qualification/submit_flux_attention_replacement.py" "$RUN" --submit-staged
# After the returned job finishes:
"$PY" -I "$MDB/qualification/submit_flux_attention_replacement.py" "$RUN" --collect JOB_ID
# Rebuild fields and verify consumers using only saved evidence:
"$PY" -I "$MDB/qualification/analyze_flux_attention_replacement.py" "$RUN" "$RUN/analysis.json"
# Publish all nine original images with a matched audit:
"$PY" -I "$MDB/qualification/publish_flux_attention_replacement.py" "$RUN" "$RUN/analysis.json" "$RUN/publication"
```

The finite CUDA lease uses four CPU/BLAS threads, the measured 24 GB RAM/13 GB VRAM lane, a 1,500-second cooperative worker budget and hard 2,100-second runner limit, with no automatic retries. Local custody capacity is checked before admission and against the actual terminal envelope before collection. Archive/expanded limits are 1,536/3,072 MiB. Numerical forecast quality and image effect direction never decide whether completed evidence is retained.

The [Astra source review](../../../../artifacts/mdb-flux-attention-replacement-20261104/prerun-review-astra.json) verifies the request seal, all 11 staged source pins and 466 package entries. The [Luna long-run review](../../../../artifacts/mdb-flux-attention-replacement-20261104/final-long-run-review-luna.json) approves one bounded admission with immutable inputs, no duplicate GPU suite, finite limits, complete custody and a saved-only independent audit. The [storage check](../../../../artifacts/mdb-flux-attention-replacement-20261104/admission-storage.json) records 94.8 GB available against a conservative 7.38 GB maximum-custody requirement.

The saved-only audit independently reconstructs full A and M, checks source/transaction identity, actual consumer-field joins and untouched operands, native paths and physical receipt counts, matched endpoint errors, and the declared reference/witness capture scope. Scientific precision, replay qualification and terminal generality remain separate.

The run reached all nine image futures by 174.55 seconds and its pre-export summary by 175.55 seconds; runner elapsed time was 264.87 seconds, with peak RSS 8.54 GB and CUDA memory 9.65 GB. This is one measured configuration, with no isolated speed comparison. Complete custody retains a 1.163 GB archive, 1.516 GB expanded evidence and the 1.550 GB raw transport log as a gzip.

One saved-only audit attempt stopped before precision analysis because its legacy nested reader did not support the tuple of FP32 RoPE tensors. The [original error](../../../../artifacts/mdb-flux-attention-replacement-20261104/audit-attempt-01.txt) and all source versions remain immutable. The corrected authenticated reader supports declared tuple/list and FP32/BF16 leaves; nineteen installed-wheel audit tests pass. Raw model evidence and forecast payloads did not change, and no model rerun was needed. The [postrun review](../../../../artifacts/mdb-flux-attention-replacement-20261104/postrun-review.json) independently verifies the accepted result.

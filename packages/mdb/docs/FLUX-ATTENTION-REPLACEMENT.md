# Predicting attention and MLP fields before the native consumer

This assay extends [full-field MLP replacement](FLUX-MLP-REPLACEMENT.md) into the parallel attention branch. It predicts complete image-side A and M fields from the observed incoming state and actual weights, seals them before the selected native block, and supplies those predictions to the real fused output projection. Native continuation provides the final latent and image observations.

**Status:** the predictor and nine-branch campaign are implemented; 30 focused installed-wheel checks pass. The fresh seed-73 experiment is running in `job-947395d68069`; no full-model fidelity result has been earned yet. Tiny native fixtures qualify mechanics, not full-model accuracy or generality.

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
```

The finite CUDA lease uses four CPU/BLAS threads, the measured 24 GB RAM/13 GB VRAM lane, a 1,500-second cooperative worker budget and hard 2,100-second runner limit, with no automatic retries. Local custody capacity is checked before admission and against the actual terminal envelope before collection. Archive/expanded limits are 1,536/3,072 MiB. Numerical forecast quality and image effect direction never decide whether completed evidence is retained.

The [Astra source review](../../../../artifacts/mdb-flux-attention-replacement-20261104/prerun-review-astra.json) verifies the request seal, all 11 staged source pins and 466 package entries. The [Luna long-run review](../../../../artifacts/mdb-flux-attention-replacement-20261104/final-long-run-review-luna.json) approves one bounded admission with immutable inputs, no duplicate GPU suite, finite limits, complete custody and a saved-only independent audit. The [storage check](../../../../artifacts/mdb-flux-attention-replacement-20261104/admission-storage.json) records 94.8 GB available against a conservative 7.38 GB maximum-custody requirement.

The saved-only audit independently reconstructs full A and M, checks source/transaction identity, actual consumer-field joins and untouched operands, native paths and physical receipt counts, matched endpoint errors, and the declared reference/witness capture scope. Scientific precision, replay qualification and terminal generality remain separate.

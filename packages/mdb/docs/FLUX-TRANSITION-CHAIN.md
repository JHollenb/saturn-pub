# Chaining complete diffusion block predictions

This workflow advances [attention/MLP field replacement](FLUX-ATTENTION-REPLACEMENT.md) to an independently calculated two-block segment. Complete predicted text and image outputs from single block6 feed the prediction of block7. Both stages are sealed before either selected current-step native block executes. The later native suffix supplies image evidence.

**Status:** the complete-block predictor and seven-future campaign pass14 focused installed-wheel tests, including controlled native BF16 parity, all-token attention, both-stream typed installation and corrupted-payload rejection. The independent auditor passes18 separate checks. The fresh-seed model experiment is running as `job-48dc7e610cd2`; real-model accuracy and chained continuation outcomes remain pending collection and audit.

## State and numerical scope

The specimen is distilled FLUX.2 Klein4B, CUDA BF16, batch1,512×512 pixels, four denoising steps. Seed79 supplies the green-cone-left-of-yellow-cylinder prompt. One fresh encoding/noise history and the established108-transition source builder/preparation produce two dependent incoming pre6 states at logical step3.

Each block receives a concatenated register R=[T;I] of shape[1,1536,3072], with512 text and1024 image positions. Every position participates as an attention query and key/value. The actual complete fused QKV/paired-MLP weights and output-projection weights are bound separately for blocks6 and7. Learned Q/K RMSNorm weights, optional biases, normalization epsilons and upstream single-stream modulation/RoPE are explicit inputs.

The numerical law includes population LayerNorm, timestep affine modulation, QKV and SwiGLU projections, learned Q/K normalization, positional rotation, full attention, fused output projection, gate multiplication and residual addition. Reductions and linear sums use independent FP64 reference arithmetic with explicit BF16 boundaries; RoPE follows the qualified FP32 multiply/add stream. The BF16 profile has no FP16 output clipping.

In schematic notation, with B denoting the declared BF16 rounding,

```text
h = modulate(LayerNorm(R), upstream scale/shift)
[Q,K,V,u,v] = B(h × W_in.T + b_in)
A = B(softmax(rope(norm(Q)) × rope(norm(K)).T / sqrt(D)) × V)
M = B(B(SiLU(u)) × v)
P = B([A,M] × W_out.T + b_out)
write = B(gate × P)
R_next = B(R + write)
```

The [exact predictor](../qualification/flux_transition_reference.py) specifies casts, reduction order, shape contracts and dependency hashes. This is a numerical reference rather than an exact CUDA-kernel emulator. Full output registers and fixed intermediate samples are retained; full intermediate A/M fields are computed but omitted from the archived prediction tree to bound storage.

## A genuine two-block chain

Let X be the observed pre6 register. The sealed prediction computes

```text
P6 = reference_block6(X)
P7 = reference_block7(P6)
```

Block7 consumes precisely the complete predicted text and image from P6. Its recorded input descriptor must match P6's full output descriptor, and its sampled incoming values must join the corresponding predicted positions. There is no native post6 input or text refresh in this path. Both histories together retain18,874,368 predicted output values across the four complete block outputs.

The remaining state is carried unchanged across this segment. The payload cursor advances6→7→8 under the declared native clock shell; step3, single phase, conditioning, latent, context, IDs and schedule remain bound. Typed image and text Acts preserve native strides, and no native computation occurs between the two slot installations.

Source-pinned guards reject transformer/VAE entry during prediction. One public forecast transaction seals both chains, inputs and hypotheses before release. This establishes trusted callback/source dataflow with external counters, not operating-system isolation.

## Seven native futures

After sealing, two native reference paths run blocks6/7. All rows meet at the actual post7 boundary, whose next operation is single8. Each then executes the same14-transition suffix: blocks8–19, readout and scheduler, followed by native VAE decoding.

| Row | Installed state before the common suffix |
|---|---|
| low.native | Native low post7 |
| high.native | Native high post7 |
| low.sham | Exact native low post7 image/text reinstall |
| low.pred1 | Predicted low post6 image/text, then native block7 |
| low.pred2 | Recursively predicted low post7 image/text |
| high.pred1 | Predicted high post6 image/text, then native block7 |
| high.pred2 | Recursively predicted high post7 image/text |

The declared census is212 native transitions,14 typed Acts,226 receipt occurrences, zero direct block calls and seven native VAE/PNG futures. It includes108 builder/preparation transitions, four reference-prefix transitions, two native block7 bridges and98 common-suffix transitions. Current-target entries after release are block6 twice and block7 four times.

Both block transitions are predicted independently, while native references still execute for validation and as replay parents. This assay makes no skipped-compute speedup claim. The suffix and final image remain native observations after installing the predicted segment.

## Separating transported error from local error

Let N6 and N7 denote native block transitions. The pred1 arm supplies C7=N7(P6), while the reference supplies X7=N7(N6(X)). Complete saved post7 registers permit the signed identity

```text
P7 − X7 = (P7 − C7) + (C7 − X7)
total discrepancy = second-block local discrepancy + transported first-block discrepancy
```

Both image and text participate. The independent audit retains norms, signed means, inner products and the squared-norm cross term. Errors may grow, contract or cancel; neither monotonic growth nor a numerical quality threshold decides whether the evidence is retained. Native-minus-sealed field reports use the opposite sign from this explicitly predicted-minus-native decomposition.

## Capture and independent audit

The complete source cuts, native post6/post7 I/T, all common-post7 I/T and final latent endpoints are retained. Fifteen actual boundaries per branch additionally retain complete native-reference arrays and signed candidate worker metrics with64 fixed-coordinate witnesses. Intermediate full candidate norms have worker-observation authority unless their arrays are independently retained. Stored Z3, stale post-readout I/T, native readout noise and committed Z4 have separate phase labels.

The [saved-only auditor](../qualification/analyze_flux_transition_chain.py) independently recalculates the complete two-block recurrence with different32-query/512-row tiling, verifies transaction/source identity, both-stream chain joins, typed installation footprints, physical receipt census, native continuation and error decomposition. Numerical discrepancies remain observations; replay qualification and terminal generality are separate.

The [launcher](../qualification/submit_flux_transition_chain.py) stages without submitting, then explicitly admits one CUDA lease with strict preflight and no retries. Four CPU/BLAS threads and the previously measured24GB RAM/13GB VRAM lane bound the run. Expected time is420seconds, with900-second worker and1200-second outer limits. Complete artifact custody is bounded to1536MiB compressed/3072MiB expanded and sized against free space before admission and collection.

```sh
PY=/absolute/path/to/qualified-environment/bin/python
MDB=/absolute/path/to/saturn-pub/packages/mdb
RUN=/absolute/path/to/new-run-directory
cd /tmp
"$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN"
# Inspect the immutable request and source/package pins before admission.
"$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN" --submit-staged
"$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN" --collect JOB_ID
"$PY" -I "$MDB/qualification/analyze_flux_transition_chain.py" "$RUN" "$RUN/analysis.json"
```

The experiment begins from an observed prefix, not a raw-prompt forecast. A successful segment would extend the symbol table with a verified composition of two state transitions. Semantic object names, arbitrary-prompt applicability and prediction of the remaining suffix require further evidence.

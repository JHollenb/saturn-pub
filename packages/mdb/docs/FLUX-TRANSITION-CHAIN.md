# Chaining complete diffusion block predictions

This workflow advances [attention/MLP field replacement](FLUX-ATTENTION-REPLACEMENT.md) to an independently calculated two-block segment. Complete predicted text and image outputs from single block6 feed the prediction of block7. Both stages are sealed before either selected current-step native block executes. The later native suffix supplies image evidence.

**Status:** both the two-block seed79 chain and three-block seed83 extension have completed with `mechanics_status=valid`, `instrument_status=native-replay-qualified`, `terminal_status=unassessed`. Latest job `job-af3d6979dd41` completed all nine native futures and normal archive custody. Nine campaign checks, 40 auditor checks and nine storage/publication/submission checks pass outside the workspace against installed wheels. The numerical predictor is unchanged. The historical two-block job's failed export and verified recovery remain preserved separately.

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

## Bounded three-block extension

Staging with `--chain-length 3` selects a separately frozen seed83 contract: complete block6→7→8 forecasts, nine matched native futures and a common post8 boundary whose next operation is single9. The default two-block seed79 variant remains available. Both variants begin at observed pre6 and use the unchanged complete numerical predictor.

| Variant | Native futures | Native transitions | Typed Acts | Receipt occurrences | Common suffix |
|---|---:|---:|---:|---:|---:|
| Blocks6→7, seed79 | 7 | 212 | 14 | 226 | 14 steps |
| Blocks6→7→8, seed83 | 9 | 237 | 18 | 255 | 13 steps |

The three-block panel adds `low.pred3` and `high.pred3`. `pred1` installs complete predicted post6 I/T and runs native7/8; `pred2` installs recursively predicted post7 I/T and runs native8; `pred3` installs recursively predicted post8 I/T. Native reference prefixes and all bridge exits retain complete cuts, I/T arrays and receipt/clock bindings. Fourteen actual observations per branch cover common post8 and thirteen suffix exits; candidate intermediate suffix norms still have worker-observation authority with fixed witnesses.

At block7, the retained pred1 post7 bridge separates local from carried discrepancy. At block8, the pred2 post8 bridge does the same. The four common post8 arrays additionally supply the exact telescoping identity

```text
P8 − X8 = [P8 − N8(P7)]
        + [N8(P7) − N8(N7(P6))]
        + [N8(N7(P6)) − X8]
```

The first term is block8's local discrepancy; the second is the finite difference caused by block7's discrepancy after transport through native8; the third carries the block6 discrepancy through native7/8. Nonlinear transport does not act on error vectors independently. Reports retain signed component statistics, all pairwise inner products and full-vector closure separately for image, text and concatenated state. None is a semantic attribution fraction.

The launcher binds the measured Linux worker cache lane for this host-local checkpoint, honors explicit worker cache overrides and rejects Mac cache paths before staging. Actual submission command cache paths, Python selection and dependencies bind to the staged request, with a hostile-local-default submission regression. Three-block transport allows 256MiB members, 2048MiB compressed and 3072MiB expanded. Parameter geometry is a necessary expanded-byte lower bound; it does not predict compressed artifact size. Complete runtime inventory and archive bounds remain enforced.

The generalized auditor passed a full saved-only regression on the previous two-block result: 177,992 existing numerical leaves were identical, with no missing values or changed lengths. This checks backward compatibility without changing the prior evidence. [Regression receipt](../../../../artifacts/mdb-flux-transition-chain3-20261104/legacy2-regression.json), [pre-outcome auditor freeze](../../../../artifacts/mdb-flux-transition-chain3-20261104/audit-qualification.json).

## Observed three-block result: seed83

The independent all-query recurrence reproduced all six sealed complete outputs bit for bit: 28,311,552 values. This is forecast reproducibility, not native CUDA bit parity. Native target entries before sealing were zero for all three blocks; afterward the verified counts were 6:2, 7:4, 8:6. Complete image and text predictions feed both chain edges without native refresh.

| History | Slot | Block6 error / native RMS | Block7 error / native RMS | Block8 error / native RMS |
|---|---|---:|---:|---:|
| Low | Image | 0.10211% | 0.24413% | 0.37211% |
| Low | Text | 0.04364% | 0.09835% | 0.14205% |
| High | Image | 0.09779% | 0.22306% | 0.32832% |
| High | Text | 0.00651% | 0.01520% | 0.02167% |

At post8, exact native image-value agreement is 41.18% low and 41.14% high. Maximum full-state absolute discrepancies are 4.0 low and 16.0 high; image-only maxima are 2.0 and 4.0. High text RMS is about 284.2, so its full-state percentage 0.02891% masks the larger image-relative discrepancy. Low combined error is 0.24839%. These are approximate numerical agreements with heterogeneous tails.

| History | Predicted blocks before remaining native path | Final latent L2 | RGB MAE, 0–255 |
|---|---:|---:|---:|
| Low | 1 | 3.43626 | 0.21275 |
| Low | 2 | 3.41612 | 0.21107 |
| Low | 3 | 3.40802 | 0.21075 |
| High | 1 | 2.94592 | 0.14952 |
| High | 2 | 2.90299 | 0.15225 |
| High | 3 | 2.91676 | 0.15247 |

Low RGB MAE decreases slightly as the predicted segment grows, while high increases; high latent L2 decreases then rises. Three-block maximum RGB channel differences are 23 low and 8 high despite small means. The low sham exactly matches native at common post8, final latent and PNG. No outcome supports an across-context monotonic improvement or semantic repair claim.

The three-term full-state L2 components (carried from6 / transported difference from7 / local at8 / total) are 93.57991 / 64.26685 / 28.42006 / 93.06904 for low and 103.48702 / 69.72273 / 32.22207 / 103.21020 for high. The identity closes exactly. Low's local8-versus-carried6 dot product is positive (+5.55015), while its other two are negative; all three high dot products are negative. Twice the sum of pairwise dots contributes −5033.28179 low and −5956.73935 high to total squared error. These are vector relations in the observed finite differences, not attribution percentages.

Negative cross terms do not guarantee that adding a local discrepancy reduces the norm: low's post8 two-term total is 93.06904 versus carried 93.03043, a slight increase; high's is 103.21020 versus carried 104.72399, a decrease. The native bridge panel exposes this distinction without refreshing any sealed forecast.

All nine native futures completed at 170.680 worker seconds; the summary was complete at 171.958. Prediction plus sealing took 19.978 seconds. Total runner time including export was 279.12 seconds; observed peak RSS/VRAM were 4567.6/9452MB. Normal custody verified 675 files, 1,705,149,008 expanded bytes and 1,269,248,597 compressed bytes. No matched speedup claim follows. Two earlier submissions failed in UV before worker/model entry; their raw logs and dispositions are retained. Their cache-path and command-intent defects were repaired before the actual model run.

Evidence: [accepted three-block audit](../../../../artifacts/mdb-flux-transition-chain3-20261104/run-03/analysis.json), [nine native futures](../../../../artifacts/mdb-flux-transition-chain3-20261104/final/result.json), [normal archive custody](../../../../artifacts/mdb-flux-transition-chain3-20261104/run-03/archive-custody.json), [campaign qualification](../../../../artifacts/mdb-flux-transition-chain3-20261104/campaign-qualification.json), [submission/storage/publication checks](../../../../artifacts/mdb-flux-transition-chain3-20261104/storage-qualification-05.json), [figure data](../../../../artifacts/mdb-flux-transition-chain3-20261104/figures/source.json).

The two new curated records are `flux.single6-8.transition-chain.p1-s83.zero` and `flux.single6-8.transition-chain.p1-s83.text`. Offline indexing verifies two qualified pre6 source citations and fifteen numerical rows: nine endpoint observations plus six complete block-output errors. All model/forecast/bridge/consumer bindings and the exact launcher snapshot remain separately recorded. Automatic family induction and automatic atlas ingestion remain unfinished. [Records](../../../../artifacts/mdb-flux-transition-chain3-20261104/computation-records.json), [index provenance](../../../../artifacts/mdb-flux-transition-chain3-20261104/evidence-v1/provenance.json), [independent postrun review](../../../../artifacts/mdb-flux-transition-chain3-20261104/postrun-review.json).

## Observed two-block result: seed79

The seal preceded both current-step target blocks: target-entry counters were zero for blocks 6 and 7. Across the two histories, the independently rebuilt prediction matched all 18,874,368 sealed output values bit for bit. That is reproducibility of the reference forecast, not bitwise equality with native CUDA execution.

The following errors use each slot's **native** RMS as denominator. The raw `native_minus_sealed.relative_rms` field instead uses the second argument, the sealed reference; derive these table values explicitly as `rms / actual_rms`.

| Incoming history | Output slot | After predicted block 6 | After predicted blocks 6→7 |
|---|---|---:|---:|
| Low (`zero`) | Image | 0.11208% | 0.25283% |
| Low (`zero`) | Text | 0.09019% | 0.21333% |
| High (`text`) | Image | 0.10272% | 0.22948% |
| High (`text`) | Text | 0.00607% | 0.01355% |

The concatenated full-state errors after the chain are 0.23786% for low and 0.01957% for high. The high history's text RMS is about 279.8, versus image RMS about 12.2, so the combined percentage hides the larger relative image discrepancy. Separate slot reports are necessary. At post7, only 57.78% and 56.30% of image values match native bits in low/high respectively; maximum absolute output discrepancy is 4.0. Small RMS discrepancy does not imply exact tensor agreement.

| Installed forecast | Native continuation | Final latent L2 from matched native | RGB MAE from matched native, 0–255 |
|---|---|---:|---:|
| Low predicted post6 | Native block7, then common suffix | 3.27180 | 0.30650 |
| Low recursively predicted post7 | Common suffix | 3.27670 | 0.30422 |
| High predicted post6 | Native block7, then common suffix | 3.14093 | 0.26499 |
| High recursively predicted post7 | Common suffix | 3.14335 | 0.26331 |

The low sham is exact at post7, final latent and PNG. Native continuation remains necessary for every final image in this experiment; these figures do not establish an independently predicted whole image or object-level semantics.

The decomposition reveals opposed errors. For low, local/carried/total full-state L2 are 31.19793 / 70.15098 / 69.92401, with `local_dot_carried=-502.55188`. For high they are 29.90930 / 68.93849 / 68.79955, with dot product -456.85164. The vector identity closes exactly. Second-block local approximation partly cancels transported discrepancy in both histories; this does not establish semantic repair or a general cancellation law.

All seven native futures completed at 166.522 seconds; the complete summary was written at 167.726 seconds. Independent prediction plus sealing took 15.666 seconds. These are one job's observations, with no speedup claim. The original export failure and the first auditor reporting failure (`KeyError: mae`) remain retained. The current auditor adds that missing observer metric independently; its four numerical-core functions are byte- and AST-identical to the frozen pre-outcome version.

Evidence: [accepted audit](../../../../artifacts/mdb-flux-transition-chain-20261104/run-02/analysis.json), [exact native futures and publication](../../../../artifacts/mdb-flux-transition-chain-20261104/final/result.json), [recovery receipt](../../../../artifacts/mdb-flux-transition-chain-20261104/run-02/recovery.json), [current auditor qualification](../../../../artifacts/mdb-flux-transition-chain-20261104/audit-qualification-v3.json), and [figure data](../../../../artifacts/mdb-flux-transition-chain-20261104/figures/source.json). All are local workspace artifacts; they are not bundled into the public package.

Two curated composition records, `flux.single6-7.transition-chain.p1-s79.zero` and `flux.single6-7.transition-chain.p1-s79.text`, bind the actual native pre6 cuts, sealed forecast blobs, output/continuation measurements and counterexamples. Offline Saturn indexing produces two search hits and eleven numerical rows. This uses explicitly snapshotted private analysis modules offline; public MDB runtime imports remain public-only. Automatic family induction and automatic atlas ingestion remain false. [Records](../../../../artifacts/mdb-flux-transition-chain-20261104/computation-records.json), [index provenance](../../../../artifacts/mdb-flux-transition-chain-20261104/evidence-v1/provenance.json), [independent postrun review](../../../../artifacts/mdb-flux-transition-chain-20261104/postrun-review.json).

The [launcher](../qualification/submit_flux_transition_chain.py) stages without submitting, then explicitly admits one CUDA lease with strict preflight and no retries. Four CPU/BLAS threads and the previously measured24GB RAM/13GB VRAM lane bound the run. Expected time is420seconds, with900-second worker and1200-second outer limits. Complete artifact custody is bounded to1536MiB compressed/3072MiB expanded and sized against free space before admission and collection.

Future staging now explicitly allows256MiB per member and checks the exact NPY header plus declared parameter/state geometry without allocating a full weight array. The historical failed-export request is unchanged. Recovered evidence uses a separate custody authority with before/after remote inventories, exact local file verification and the original job/log/failure; no worker transport report or archive is fabricated.

```sh
PY=/absolute/path/to/qualified-environment/bin/python
MDB=/absolute/path/to/saturn-pub/packages/mdb
RUN=/absolute/path/to/new-run-directory
cd /tmp
"$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN"
# Or select the separately bounded third-block extension:
# "$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN" --chain-length 3
# Inspect the immutable request and source/package pins before admission.
"$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN" --submit-staged
"$PY" -I "$MDB/qualification/submit_flux_transition_chain.py" "$RUN" --collect JOB_ID
"$PY" -I "$MDB/qualification/analyze_flux_transition_chain.py" "$RUN" "$RUN/analysis.json"
```

The experiment begins from an observed prefix. It now supplies audited compositions of two and three complete block-state forecasts and their actual native continuations. The three-block extension stays within one denoising step; readout and scheduler prediction introduce additional state and numerical contracts. Semantic object names, arbitrary-prompt applicability and prediction of the remaining suffix require further evidence.

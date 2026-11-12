# Contextual eye programs, receiving state and predicted writes

Two November 4 assays follow the [late MLP-only semantic counterexample](FLUX-PREDICTED-FUTURE.md). Both use the same saved controlled portrait square: FLUX.2 Klein4B distilled, inventory `dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f`, CUDA BF16, batch1,512px,four diffusion steps,seed60665. These are new interventions on an existing specimen, not independent prompts or cross-seed replication.

## Restoration identifies the continuing command

Job `job-d75bfb5263fe` keeps the previously predicted joint single6 image M fixed. It restores latent history L, incoming working image I and working text T separately. All four T1 cases show violet-looking eyes; T0 cases retain amber-looking eyes, under both tested latent histories. Incoming I also changes fine appearance and distance to the natural joint reference. Whole-image distance alone would misidentify the color carrier: L1I1T0 is closer to the joint image while retaining amber eyes.

Project-stage reconstruction places the latent-history intervention at a legal writable boundary, then 12 native transitions reproduce the stored pre6 payload exactly before the I/T restoration. This is an explicitly derived reconstitution, not a claimed historical rewind to an uncaptured cut. Two natural PNG controls and the previous M-only-transfer PNG reproduce exactly.

Four L1 terminal latent forecasts reuse the corresponding observed L0 readout velocity because the tested post6 suffix does not read L until scheduler. The independent calculation preserves native scalar-promotion/BF16 boundaries. All four complete latent arrays reconstruct exactly offline. Two native corner checks match all131,072latent values, velocity and decoded pixels exactly; middle01/10 use predicted terminal state and native VAE without native suffix validation. Complete intermediate I/T/noise hashes were recorded on the checked pair, but complete intermediate arrays were not retained.

This yields one qualified example for [dependency reuse](DEPENDENCY-REUSE.md):14 unaffected operations and one scheduler recalculation. It is neither a universal prompt edit shortcut nor a measured end-to-end speedup.

The successful census is1load,0encodes,164native transitions,19Acts,183receipts,8direct candidate blocks,12images including4forecast-state decodes,49.963worker seconds. The prior job `job-e125189d18e2` failed after prefix/reconstruction because `noise_prediction` was declared by the surface but absent from `Session.read()` addresses. Its logs and incomplete collection remain separate evidence; the repaired observer captures the applicable scheduler state instead. No completed semantic panel is attributed to that failed job.

## Current write versus continuing text

Job `job-2f0cd41dfc45` starts immediately after single6 and splits the changed image and text:

| Installed state | Observed continuation |
|---|---|
| Changed I and T | Violet eyes; reference program future. |
| Changed I, base T | Amber eyes; current image write is insufficient. |
| Base I, changed T | Violet eyes; RGB MAE0.224406/255 from full-program future. |

At the next block7, a new additive T input changes image attention, while the complete image M field remains **exactly base**. This matches the pointwise MLP versus cross-token attention dependency. It does not identify a globally necessary head, lexical coordinate or minimum causal cut.

## Composition and prospective arithmetic

Prepare color-only and shape-only contextual post6 T on the same base I through one native block each. Form `Tadd = BF16(Tcolor + Tshape - Tbase)`. Actual joint T is only a comparison; it never supplies this additive rule. Predict complete image A and M at block7 from I/T, timestep modulation, actual parameters, learned QK norms and upstream RoPE. A block pre-hook rejects entry until the forecast transaction is sealed.

| Quantity | Observation |
|---|---|
| Complete A forecast |3,145,728values; relative RMS error0.125658% versus native. |
| Complete M forecast |9,437,184values; relative RMS error0.023464% versus native. |
| Independent source-only A/M rebuild |Every declared BF16 reference value exact. This does not make the native CUDA values exact. |
| Predicted versus measured additive continuation |RGB MAE0.122032/255; latent L2 2.16349. |
| Additive versus actual joint-text continuation |RGB MAE1.776376/255; latent L2 23.91763. |

The additive future has violet/pink-looking eyes and changed eyelid geometry while retaining coarse copper hair, plum clothing, frontal portrait and gray background. It is an approximate semantic composition; exact hue, shape and fine preservation remain unresolved. The color-only versus shape-only contrast is visually informative, with shape less certain than color under this unblinded observer.

The native text exit is retained, the native output projection/gate consumes predicted image A/M, and the remaining14transitions/VAE execute natively. Original native source blocks also execute after sealing for comparison. This is a conditional local prediction and native consumer test, not independent raw-prompt image prediction or avoided native attention execution.

Changed lexical rows47/48 alone and the active89-row patch yield weaker muted color responses. Padding-coordinate423rows alone and the64largest source-contrast-energy rows retain amber-looking eyes. Rolling all source T256positions retains violet-looking eyes. Retain these heterogeneous outcomes: high energy is not sufficient, contextual T is distributed, and the roll is not a destructive semantic null. These supplied tokenizer coordinates and image display crops are not discovered internal object segmentation.

The later [saved-state interaction diagnostic](../../../../artifacts/mdb-eye-program-20261104/interaction-diagnostic-v1/analysis.json) separately predicts A for base/color/shape/joint contextual T at fixedbaseI, without any native calls. It decomposes total attention interaction exactly into `A(Tjoint)-A(Tadd)` plus `A(Tadd)-A(Tcolor)-A(Tshape)+A(Tbase)`. Their norms27.9409 and26.9759 oppose with cosine−0.79977, leaving total17.4002. Tjoint−Tadd has norm71.25% of Tjoint−Tbase. These are post-outcome analytic diagnostics using observed joint T, not prospective new image predictions or native validations of the four additional fields. Norm fractions are not causal contribution percentages. The next discriminating intervention should correct these terms separately while preserving continuing-T versus current-write distinctions.

The census is1load,0encodes,2tokenizations,196native transitions,13Acts,209receipts,2direct source blocks,1projection,13VAE images,56.8905worker seconds and107.140runner seconds. Export retained295,127,988archive bytes across221files. No end-to-end speedup is measured.

## Reproduce and inspect

Use installed public MDB/Saturn-pub/mrun-pub packages. Staging verifies prior saved evidence before hydrating exact size/hash-bound historical CAS on the source host. It ships qualification helpers explicitly; runtime imports do not discover sibling checkouts or import private Saturn/common.

```sh
python -I /absolute/MDB/qualification/submit_flux_eye_program.py FRESH_RUN --dependencies DEPENDENCY_RUN --predicted PREDICTED_RUN
python -I /absolute/MDB/qualification/submit_flux_eye_program.py FRESH_RUN --submit
python -I /absolute/MDB/qualification/submit_flux_eye_program.py FRESH_RUN --collect JOB_ID
python -I /absolute/MDB/qualification/analyze_flux_eye_program.py FRESH_RUN FRESH_AUDIT.json
```

Both launchers use one finite strict-preflight mrun CUDA lease, no retries or encodes, a10GBRAM/13GBVRAM exception calibrated on the same resident no-encoder lane,600sworker/720slease limits. The November4 qualified cache is `/mnt/hdd8tb/mdb/uv-cache`, Torch cache `/mnt/hdd8tb/mdb/torch`, model roots `/mnt/big/llm-models`. The prior NVMe cache was full. `common exec` supplies scheduler credentials when needed.

- [Restoration audit](../../../../artifacts/mdb-scene-dependencies-20261104/analysis-v1.json)
- [Eye-program audit](../../../../artifacts/mdb-eye-program-20261104/analysis-v2.json) and [visual review](../../../../artifacts/mdb-eye-program-20261104/visual-review.json)
- [Sealed request](../../../../artifacts/mdb-eye-program-20261104/run-01/request.json)
- [Worker](../qualification/flux_eye_program_campaign.py), [stager](../qualification/submit_flux_eye_program.py), [auditor](../qualification/analyze_flux_eye_program.py)
- [Native images](../../../../obsidian/images/mdb-eye-program-20261104/eye-program-futures.png) and [supplied eye display crops](../../../../obsidian/images/mdb-eye-program-20261104/eye-crops.png)

The highest-value next cut is a matched context/clock or head/K/V decomposition of this contextual program, retaining separate color, shape and collateral observations. General objects and relationships require additional specimens and earlier geometry/order clocks; these portrait results do not certify the older two-object spatial programs.

## MDB integration that would remove the most friction

1. **Connect the existing components into a native diffusion experiment runner.** Compile a small source/carrier/write/consumer panel from declared competing hypotheses, common parents and complete native footprints. Reuse `NativeCutNavigator`, `ForecastTransaction`, the existing experiment-panel and circuit-explorer concepts, and the new dependency planner. Automatically distinguish observed prefixes, predicted operands, native comparisons and actual work avoided. This still requires implementation; current CPU ATen prepared panels are not this multi-clock native diffusion integration.
2. **Expose typed within-block Q/K/V, A/M, projection and gate ports.** Current public image/text exit Acts work, but qualification helpers still capture hooks and reconstruct a fused block exit. Native readers should match declared applicable surface ports: the scheduler noise-reader failure is a concrete example. Preserve fused GEMM/numerical identities and explicit sequential-recompute versus frozen-write semantics.
3. **Make target, collateral and trajectory capture part of each record.** Keep selected full candidate cuts, paired reference-plus-diff observations, identity/geometry/color/relationship observer limitations, failed predictions and next-cut selection. The current five [curated records](../../../../artifacts/mdb-eye-program-20261104/computation-records.json) have ten model/parent-qualified addresses and27numeric rows in an [offline index](../../../../artifacts/mdb-eye-program-20261104/evidence-v1/provenance.json); they are not automatically induced laws or imported EquationCatalog families.
4. **Publish binary results by manifest.** Reopen parameter/state/evidence artifacts through bounded binary outbox/CAS and selected-file retrieval rather than base64 logs. The new program panel's56.89worker seconds versus107.14runner seconds includes substantial export overhead, although it is not an isolated transport benchmark.

These items extend capabilities already present in MDB/private Saturn rather than propose four independent replacement systems. The [existing request upvotes](../../../../artifacts/mdb-eye-program-20261104/request-updates.json) bind the concrete gaps to the measured jobs. A tenfold research-productivity gain is a target to measure, not an established timing result. Dependency planning is implemented and installed-wheel tested in this iteration; automatic native incremental execution, typed internal-port integration and semantic observer certification are not.

# Later attention and MLP interaction writes

This follows [early text-to-image formation](FLUX-SHAPE-FORMATION.md). Four pre-single6 image/text combinations produced different final interactions in seeds60666/60667. The new assay replays their saved step2 parents, traces the later native operators, then tests a selected current attention/MLP/text write through native image continuation.

The specimen remains FLUX.2 Klein4B distilled, CUDA BF16, batch1,512px,four denoising steps, with the same observed prompt encodings. These are known-source, dependent interventions. No prompt encoding or fresh noise is used.

## Trace policy and native controls

For four corners00/10/01/11, define `J_X = X11 - X10 - X01 + X00`. The trace records J and its operator increment at114clock boundaries across two seeds. Latent is always tracked; working I/T are live at joint/single/readout entry, noise prediction at scheduler entry. Stale I/T at project/scheduler boundaries are excluded.

Four native contexts per seed advance in lockstep through56remaining operators. `Session.capture(retain=False)` avoids accumulating every complete state in session history. Each live field has64fixed witnesses and worker-computed metrics; sparse/full selected cuts allow independent full-field rebuilding. The audit verifies20,864interaction witness values and50complete boundary-slot comparisons. Intermediate full metrics outside those cuts remain source-bound observations.

The candidate is the largest image-write interaction L2 among writable single blocks0–18 at steps2/3, separately per seed. This selects step3/single18 in both:4872.777 and7561.889. The measured single19 writes are smaller,2876.014 and4182.479, but index19 is excluded by the current post-block image-Act boundary. Ranking is a diagnostic; it does not establish semantic responsibility.

All8native suffix PNGs reproduce their historical references exactly. At the chosen block, four complete concatenated attention/MLP operands, native projections/gates and pre/post I/T cuts are captured. All source image/text returns match the native replay, and the gated BF16 exit equation is independently rebuilt.

The trace also shows text interaction returning after a common prompt read: native text interaction is0just after project, then nonzero during the joint stack. Complete pre-single0 text interactions are independently audited. Working text depends on the current image history even though E is common.

## Causal writer tests

Keep the coherent11incoming residual, gate, latent and execution context. Form `Alinear = B(A10 + A01 - A00)`, similarly Mlinear and outgoing Tlinear. The A/M operands are native observations; their combinations are calculated interventions. B denotes BF16 rounding after the declared FP32 composition. Text rows in the fused projection retain native11values.

Eight arms per seed test sham, Alinear, Mlinear, AMlinear, Tlinear, AMTlinear, an opposite attention dose `B(2*A11 - Alinear)`, and independently predicted AM image exit. Frozen A/M substitution does not recompute one operand after changing the other. Every arm continues through native single19, readout, scheduler and VAE.

Signed eye-region RGB-contrast changes versus the coherent native sham are:

| Arm |60666|60667|
|---|---:|---:|
|Alinear|+0.011257|−0.058873|
|Mlinear|−0.005669|−0.047958|
|AMlinear|+0.004851|−0.119957|
|Tlinear|+0.000615|+0.000796|
|Opposite A|−0.019480|+0.016197|

These are projections onto each seed's native weak→strong RGB contrast, not percentages of eye shape. The forecast that Tlinear would shift the eye contrast more than AMlinear fails in both seeds. Late current-image writes have more tested authority than this outgoing-text change. Alinear and opposite A move in opposite projected directions in each seed, as forecast, while the algebraic recipe's direction differs across seeds. The operand payload and receiving state both differ; this test does not separate those two causes.

The effects are small fine-detail changes, with collateral. AMlinear eye MAE versus sham is1.035345/1.610424, clothing0.599467/0.633130 and whole image0.321033/0.518913. Coarse appearance remains in the inspected images. This local write does not explain the entire earlier interaction or identify a minimum circuit.

## Frozen mathematical predictions

Independently calculate eight full projection/image exits from the observed operands and actual weights:

```
P_hat = B([A_selected; M_selected] @ W_out.T + b)
I_hat = B(R11 + B(g * P_hat))
```

All8P/8I fields seal before any mixed native projection, full mixed-block validation or new image future. Independent saved-only rebuilding with different matrix chunking matches all16sealed fields exactly. All8native cached mixed exits match complete native frozen-block image returns; both shams reproduce historical PNGs. CPU predictions have native-relative image RMS errors0.024846–0.033806%, rather than native bit equality.

| CPU-predicted AM exit versus native AM future |60666|60667|
|---|---:|---:|
|Final RGB MAE,0–255|0.079737|0.070488|
|Maximum pixel error|4|4|

The source A/M fields are observed, and the remaining native suffix still computes the image. These measurements qualify a predicted internal write and its native future, not raw-prompt complete image prediction or a speedup benchmark.

## Reproduction and cost

Use the installed qualified environment and worker cache/model variables from [step1 writers](FLUX-STEP1-WRITERS.md). Stage and submit with identical environment, through `common exec` when scheduler credentials are needed:

```
python -I qualification/submit_flux_interaction_trace.py TRACE_RUN --discovery FORMATION_ROOT --replication REPLICATION_ROOT
python -I qualification/submit_flux_interaction_trace.py TRACE_RUN --submit
python -I qualification/submit_flux_interaction_trace.py TRACE_RUN --collect TRACE_JOB
python -I qualification/analyze_flux_interaction_trace.py TRACE_RUN TRACE_AUDIT --discovery FORMATION_ROOT --replication REPLICATION_ROOT
python -I qualification/submit_flux_interaction_writes.py WRITE_RUN --source TRACE_ROOT
python -I qualification/submit_flux_interaction_writes.py WRITE_RUN --submit
python -I qualification/submit_flux_interaction_writes.py WRITE_RUN --collect WRITE_JOB
python -I qualification/analyze_flux_interaction_writes.py WRITE_RUN WRITE_AUDIT --source TRACE_ROOT
python -I qualification/publish_flux_interaction_writes.py WRITE_RUN WRITE_AUDIT ATLAS
```

The maintainer writer launcher currently reads TRACE_ROOT/run-02, the retained corrected trace attempt. Existing artifacts use `artifacts/mdb-interaction-trace-20261104/` and `artifacts/mdb-interaction-writes-20261104/`.

Jobs `job-37be2daf859f` and `job-b6b6b5de5f4a` produce8exact replay images and16causal/predicted futures,496native transitions,32Acts and528receipt emissions. Sixteen direct block captures/validations and10cached projections are separately counted. The writer panel uses only3native transitions per future from saved post18cuts. Each successful job loads the model once; neither encodes prompts or draws noise.

Worker times112.207/95.009seconds; runner184.04/196.54seconds. PeakRSS5557.2/7699.2MB, VRAM9690/10264MB, within10GB/13GB reservations. Output archives693.4/805.9MB live on HDD scratch. The first trace attempt, `job-4e6c107eb1b5`, failed its transaction-frontier declaration before native replay; its63.54seconds and diagnostics remain under run-01. The corrected request uses the explicit architectural frontier. These setup/export costs matter when assessing future efficiency.

Four tiny-native mechanics tests pass. Eight new public equation occurrences reuse the existing generic parallel-gated-write law with clock/receiver-qualified contexts;8exact and16changed-context queries retain applicability limits and forecast misses. The offline evidence sweep indexes those8equations and2curated trace records,142numeric rows and16source-qualified addresses, with run/verify/query returning0. Automatic semantic family discovery remains open.

The [completed fixed-correction test](FLUX-RECEIVER-CROSS.md) crosses both attention payloads with both receivers and signed doses. One payload retains its measured direction while the other reverses; neither simple payload-led nor receiver-led rule covers both. The frozen numerical calculations still produce close native futures. The next cut holds a complete image-state pair fixed while crossing the other suffix context. Generic paired observers and typed A/M/gate ports remain the main tooling opportunities; current full-metric/sparse-cut coverage is explicit and reproducible.

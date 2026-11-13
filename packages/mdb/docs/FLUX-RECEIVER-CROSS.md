# Fixed attention corrections across receiving states

The [late writer assay](FLUX-INTERACTION-WRITERS.md) found opposite eye-region RGB effects for the same attention-linearization recipe in two seeds. Both payload and receiver varied. This assay crosses two fixed source-assisted corrections with both native receivers at step3/single18, preserving the receiver's MLP operand, residual, gate, outgoing text, latent and other execution state.

The specimen remains installed-wheel FLUX.2 Klein4B distilled, CUDA BF16, batch1,512px,four denoising steps. Two known seeds60666/60667 reuse observed prompt encodings and saved parents. There are no new prompt encodes or noise draws. These are new combinations of known state, rather than new-prompt generalization.

## Declared payload and mathematical forecast

For donor d, form `delta_d = FP32(B(A10 + A01 - A00)) - FP32(A11)`, with the inner sum evaluated in FP32 and B denoting BF16 rounding. For receiver r and dose u in `{-1,-0.5,0.5,1}`, replace only its image-attention operand:

```
A_new = B(FP32(A_r) + u * delta_d)
P_hat = B([A_new; M_r] @ W_out.T + b)
I_hat = B(R_r + B(g_r * P_hat))
```

Text rows retain their native receiver values and full fused projection geometry. The same FP32 delta is used at the same token/spatial coordinates in both receivers; BF16 rounding means the effective installed delta is not identical. Its relative RMS discrepancy from the requested dose is0–4.027%. Cross-seed pixel coordinates are not a semantic object alignment.

All16complete P/I forecasts seal before any mixed native projection, candidate block or new RGB. Independent CPU rebuilding with a different projection chunk size matches all32fields exactly. Native cached exits match all16complete mixed-block image validations. W and gates are exactly equal across receivers; donor-delta cosine is0.005668. Image-exit error against native arithmetic is0.023667–0.041553% relative RMS. This predicts combination from observed A/M/R/gates and weights, not operand formation.

## Frozen directional models and observations

Two competing forecasts were frozen: a payload-led model predicts donor60666positive/donor60667negative effects in either receiver; a receiver-led model predicts positive effects in60666/negative in60667 for either donor. Negative doses reverse each forecast. Six previously observed endpoints are controls, excluded from new cross-case scoring.

Positive-dose eye-region RGB contrast changes versus native receiver sham are:

| Payload | Receiver | Half dose | Full dose |
|---|---|---:|---:|
|60666|60666|+0.006283|+0.011257|
|60666|60667|+0.001072|+0.002055|
|60667|60666|+0.006057|+0.009597|
|60667|60667|−0.024107|−0.058873|

Each model matches4/8new signed cross cases. Payload60666keeps its local-reference sign; payload60667changes sign across receivers. Neither frozen model accounts for both. All8half-dose-between-zero-and-full forecasts match, and opposite doses have opposing signs in all4payload/receiver pairs. The small, dependent sample does not establish a universal dose law.

The observer projects onto each receiver's saved weak→strong eye-window RGB contrast, not percentage eye geometry. An exploratory common-axis diagnostic retains the reversal for payload60667on the fixed60667axis: full-dose shifts+0.003218in receiver60666and−0.058873in receiver60667. On the60666axis both are positive. Axis choice matters; it does not alone explain the first fixed-axis reversal.

Native image-exit displacement cosine across receivers is0.98065–0.99773for a fixed payload/dose, with6.75–19.68% relative RMS differences. Requested raw corrections, BF16-realized corrections, image-exit displacement and final pixel effects are separate quantities. The assay does not attribute the entire sign reversal exclusively to the suffix.

All20native images were inspected. Coarse black hair, amber irises, teal shirt, ivory background and frontal portrait remain. Differences concern fine eyelid/lash/outline/shading details, with collar and clothing collateral. At full positive dose, eye MAE versus sham is0.406–1.058RGB levels; clothing0.312–0.546. There is no categorical eye-shape conversion claim or calibrated geometry evaluator.

## Predicted writes through the native consumer

Each receiver also receives its independently calculated cross-donor+1image exit. Native outgoing text remains fixed, followed by single19, readout, scheduler and VAE.

| Predicted cross-write future versus native cross-write | Receiver60666 | Receiver60667 |
|---|---:|---:|
|RGB MAE,0–255|0.078813|0.072130|
|Maximum pixel difference|4|4|

Both match the frozen MAE-below1forecast. Native continuation still computes the final image. Accurate local arithmetic and conditional visual-effect prediction remain separate coverage claims.

## Run and audit

Use the installed qualified environment and identical worker environment at staging/submission, as in [step1 writers](FLUX-STEP1-WRITERS.md):

```
python -I qualification/submit_flux_receiver_cross.py RUN --source TRACE_ROOT --writers WRITER_ROOT
python -I qualification/submit_flux_receiver_cross.py RUN --submit
python -I qualification/submit_flux_receiver_cross.py RUN --collect JOB
python -I qualification/analyze_flux_receiver_cross.py RUN AUDIT --source TRACE_ROOT --writers WRITER_ROOT
python -I qualification/publish_flux_receiver_cross.py RUN AUDIT ATLAS
```

The maintainer launcher reads TRACE_ROOT/run-02and WRITER_ROOT/run-01, the prior qualified attempts. Actual roots are `artifacts/mdb-interaction-trace-20261104`, `artifacts/mdb-interaction-writes-20261104` and `artifacts/mdb-receiver-cross-20261105`.

Job `job-c48742d4fe14` completes20native RGB futures:2shams,4known own endpoints,12new other numerical branches including8cross cases, and2CPU-cross futures. Counts are60native adapter transitions,40typed I/T Acts,100receipt emissions,16separate direct block validations and18cached projections. There is1model load,0encodes and0noise draws. Six historical PNG controls match exactly. The audit verifies typed Act payloads, untouched slot closure, receipt chains and final latent/image custody.

Worker time124.337s, runner265.35s, peakRSS8498.6MB/VRAM9590MB within10GB/13GB reservations. The1,272,008,863-byte archive is transported through a1,696,059,953-byte log. Independent local audit takes54.83s. Worker timing excludes input preparation before its timer and export afterward; these are not end-to-end speedup measurements.

Two tiny-native mechanics tests pass in3.68s. Sixteen public equation/atlas occurrences reuse `flux.single.frozen-parallel-gated-write.reference.v1`, qualifying receiver, payload, dose, clock, observer and coverage;16exact and32changed-clock/receiver queries pass. Counterexamples are retained, not converted into a new universal rule. Runtime, launcher and native consumer use installed public MDB/Saturn-pub/mrun packages; qualification helpers are explicitly shipped sources.

## Next causal cut and tooling

The [completed suffix-context test](FLUX-SUFFIX-CONTEXT.md) holds complete image pairs identical across native text and latent contexts. Text changes attention19 while image MLP19 stays identical; latent leaves post19/readout unchanged and enters at commit. Whole-latent swaps damage the portraits, prompting a smaller neighborhood with predicted commits and zero new denoiser/readout calls. Its nonlinear pixel response preserves a missed linear forecast. Coherent nearby states, exact spatial alignment and a geometry reader remain separate next questions.

The current implementation is a qualification harness using public transactions, typed image/text Acts, saved cuts, replay and catalogs. First-class typed attention/MLP ports would remove local capture hooks. CAS references to native operands plus correction recipes would avoid repeatedly exporting unchanged text/MLP sections. Existing feature requests were upvoted with measured costs; production fleet rollout is separate. No generic port API or new end-to-end predictor speedup is claimed here.

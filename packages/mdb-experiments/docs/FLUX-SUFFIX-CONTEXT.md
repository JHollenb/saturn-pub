# Fixed image pairs and the latent receiving state

This follows the [fixed attention correction test](FLUX-RECEIVER-CROSS.md). It separates the complete image-state pair from working text and carried latent at the last single block, then explores small latent changes using qualified Euler predictions and the native VAE.

The specimen is FLUX.2 Klein 4B distilled, CUDA BF16, batch one, 512 pixels and four denoising steps. Seeds 60666/60667 and prompt encodings are known. Both panels reuse saved native states, with no new prompt encoding or noise draw.

## A legal image × text × latent factorial

The saved pre-single19 parents differ only in image, text and latent. All other fields match exactly. Select the native parent for each latent, then install image and text through public typed Acts. This avoids a late latent write or adapter change.

For each source image, keep the complete I0/I1 pair identical across two text and two latent sources. I1 is the previously tested +1 attention correction from donor 60667. The 16 futures retain complete pre19, post19, scheduler and final cuts, plus eight separate direct native image A/M/P/gate captures.

Observed predictions and controls:

- All four historical PNGs reproduce exactly.
- All four fixed-image comparisons have bit-identical image MLP operands and gates across text sources. Attention changes; A RMS differences are 0.01119–0.01146.
- All eight fixed-I/T comparisons have bit-identical post19 image/text and readout noise across latent sources. Within this tail, latent first affects the computation at scheduler commit.
- Sixteen complete Euler forecasts seal individually after observing readout, before their own native commit/RGB. All independently rebuild and match native committed latent values exactly.

The image MLP still changes when its incoming image changes. The result isolates the additional influence of the tested text replacement through attention at this block; it does not remove MLPs from the overall edit computation.

On the fixed saved 60667 eye-window RGB axis, pair 60666 has effects +0.003218/−0.008323 when only latent changes under T60666. Pair 60667 remains negative under either latent. Image-pair-led and latent-led sign forecasts each match 4/6 new paired comparisons; text-led matches 2/6. Neither single-factor story accounts for all cases.

All eight whole-latent mismatches visibly damage the portraits with grain, mottling and doubled detail. They remain valid numerical counterfactuals but do not establish clean semantic eye edits. Effects also depend on observer basis. The audit preserves both saved axes, target/collateral and all misses.

## Shrink to a receiving-latent neighborhood

Keep each pair's own working text and observed readout noise. Construct receiving latents at alpha 0, 0.05, 0.1, 0.25, 0.5 and 1 toward the other native latent:

```
Z_alpha = B(FP32(Z0) + alpha * (FP32(Z1) - FP32(Z0)))
Z4_hat = B(Z_alpha + B(B(dt) * V3))
```

Alpha 0/1 use exact source clones to preserve endpoint values under FP32 cancellation. B denotes the qualified BF16 rounding boundaries. All 24 complete final-latent forecasts seal before any new native validation or decoding. Native velocity is observed source state, not independently predicted here.

The VAE consumes explicitly initialized terminal states containing those predictions. Four additional native Euler checks at alpha 0.1 use explicitly initialized scheduler states, with receipts. These are declared counterfactual initializations, not late latent Acts or historical rewinds. Stored cut manifests retain their layout and lineage.

All 24 forecasts independently rebuild; all four native checks match predicted latent values exactly. All eight historical endpoint PNGs reproduce. Instrumented transformer/project/block and readout counters remain zero. There are 24 real VAE outputs, four native scheduler transitions and zero Acts.

Before execution, freeze a linear interpolation of each pair's observed endpoint eye effects. Results on the fixed 60667 axis:

| Alpha | Pair60666 predicted | Pair60666 observed | Pair60667 predicted | Pair60667 observed |
|---|---:|---:|---:|---:|
|0.05|+0.002641|+0.003673|−0.057869|−0.061299|
|0.10|+0.002064|+0.003936|−0.056864|−0.063788|
|0.25|+0.000333|+0.004145|−0.053851|−0.068160|
|0.50|−0.002553|+0.001159|−0.048829|−0.063794|

Seven of eight signs match, but only one of eight directions of movement from alpha 0 matches the frozen function. Mean absolute point error is 0.006257 in this RGB-projection unit. Small-dose curves initially move away from the endpoint line in both pairs. Sign accuracy alone would overstate predictive understanding.

Manual review finds broad portrait structure retained through alpha 0.25, increasing distortion at 0.5 and severe damage at 1. Baseline whole-image MAE at 0.05/0.1 is 0.488/0.945 and 0.533/1.050 across the two sources; at 1 it is 20.512/21.768. These are drift observations, not calibrated geometry or quality gates. There is no demonstrated clean local sign reversal.

## Reproduction

Use the installed qualified environment and identical worker environment at staging/submission, as in [step1 writers](FLUX-STEP1-WRITERS.md):

```
python -I qualification/submit_flux_suffix_context.py RUN --source CROSS_ROOT
python -I qualification/submit_flux_suffix_context.py RUN --submit
python -I qualification/submit_flux_suffix_context.py RUN --collect JOB
python -I qualification/analyze_flux_suffix_context.py RUN AUDIT --source CROSS_ROOT --trace TRACE_ROOT
python -I qualification/publish_flux_suffix_context.py RUN AUDIT ATLAS
python -I qualification/flux_latent_neighborhood.py NEXT_RUN --source SUFFIX_ROOT
python -I qualification/flux_latent_neighborhood.py NEXT_RUN --submit
python -I qualification/flux_latent_neighborhood.py NEXT_RUN --collect NEXT_JOB
python -I qualification/analyze_flux_latent_neighborhood.py NEXT_RUN NEXT_AUDIT --source SUFFIX_ROOT --trace TRACE_ROOT
```

Actual roots are `artifacts/mdb-suffix-context-20261105/run-01` and `artifacts/mdb-latent-neighborhood-20261105/run-02`. The neighborhood's first staging attempt failed a tuple/list JSON roundtrip before admission and is retained. A saved-only diagnostic dtype error required an audit repair; no model reexecution occurred. Partial atlas publication v1 is retained separately from corrected v2 with its required exact source condition.

Jobs `job-dcae09a1f5c9` and `job-93f45da60821` produce 40 VAE images, 52 native adapter transitions, 32 typed Acts and 84 receipt emissions. Eight separate direct last-block captures are counted additionally. The second job makes zero new denoiser/readout calls. Three tiny mechanics tests pass.

The panels use one model load each. Runner times are 184.48/96.90 seconds, archives 313.8/74.5 MB, peak RSS 7675.2/8448.0 MB and VRAM 9616/9200 MB, within their measured 10 GB/13 GB reservations. Independent saved-only audits take 26.02/7.01 seconds; the retained failed audit costs another 26.09 seconds. Source acquisition, loading, export and VAE work remain part of the research cost. This is bounded reuse of observed readouts, not a measured end-to-end speedup or raw-prompt image predictor.

The suffix atlas has 16 native Euler occurrences plus eight separately curated causal pairs; the neighborhood atlas has four actual native Euler comparisons plus 12 curated response pairs. Equations reuse `flux.euler.commit.bf16-reference.v1`. The atlases qualify measured source/state/dose/clock contexts rather than inducing a semantic family. Offline evidence run/verify/query returns zero for both panels, indexing 40 records, 138 numeric rows and 60 source-qualified addresses. Native references, predicted terminal states, known endpoints and newly scored cases remain distinct.

## Next useful cut

Receiving-state coherence belongs in applicability. A complete latent exchange can damage the scene even when the remaining denoiser computation is unchanged. Small local changes preserve a useful organism but reveal a nonlinear response that endpoint interpolation misses.

The next semantic test should create coherent nearby states upstream, allowing image/text state to update with the latent, then freeze predictions for new receiving contexts. A separate small decoder response test can distinguish BF16 commit rounding from VAE nonlinearity. Neither needs an immediate expansion to more characters or images.

MDB already supplies the required cuts, legal writes, receipts, transactions and native consumer. The next tooling improvements are consumer-only loading for VAE exploration, reusable paired observers and forecast scoring that distinguishes sign from change in magnitude. The current qualification harness still loads the denoiser for its adapter/identity despite making zero denoiser calls in the second panel.

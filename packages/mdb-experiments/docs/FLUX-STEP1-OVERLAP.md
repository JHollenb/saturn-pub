# Step-1 single6: state, attention, MLP and continuing text

This follows the [pulse/handoff assay](FLUX-STEP1-WRITERS.md). A shape pulse followed by base conditioning largely restores round eyes, so the writer test holds later **shape E** common instead. It compares a weaker native reference (base through step1, shape from2) with a stronger reference (shape from1). Both start from the same recorded base step1 latent and clock.

## Causal panel

At step1/single6, independently select:

- R: incoming working image, base/shape history.
- A: observed raw image attention operand entering the fused output projection.
- M: observed gated image MLP operand entering that projection.
- T: native outgoing working text, base/shape history.

All16RAMT combinations run the remaining71adapter transitions, with shape E installed before step2. T is an observed complete postblock field, not an independently predicted text return. Frozen A/M combinations are interventions and need not occur naturally. Gates/context match. Remaining within-step conditioning differences are unread and overwritten before the next conditioning read.

The two endpoint PNGs reproduce earlier onset2.shape/onset1.shape exactly. Complete native cached exits reproduce all8frozen-block image returns exactly, including six new mixed full-block validations. These are mechanics checks, distinct from semantic conclusions.

The image observations show a larger contribution from accumulated R and continuing T than the **current block's** A. M has a smaller positive effect across all8matched conditional comparisons. For a fixed supplied eye rectangle, project RGB differences onto the weak→strong source contrast:

| Axis changed | Minimum signed increment | Maximum | Mean |
|---|---:|---:|---:|
|R|0.255031|0.361247|0.307974|
|A|-0.003183|0.021558|0.007668|
|M|0.058212|0.146725|0.094818|
|T|0.522719|0.660633|0.602836|

These are mixed RGB-contrast units, not percentages of eye shape or causal importance in general. The current A substitution has little eye projection while moving clothing contrast negatively in all8contexts (-0.1521to-0.0374). Target and collateral must stay separate.

Rshape+Tshape with base A/M gives eye projection0.935135, eye MAE1.733515 and whole-image MAE0.777200 versus the complete strong reference. Its eyes visually resemble the stronger narrow-eyelid response, with detailed differences. This is a sufficient tested combination, not a minimum circuit. Earlier attention could have written the useful R or T; small current-A effect does not exclude it.

## Mathematical forecasts and native futures

For observed operands and actual weight/bias, independently compute:

```
P_hat = B([A_selected; M_selected] @ W_out.T + b)
I_hat = B(R_selected + B(g * P_hat))
```

All4complete P and8complete I predictions seal before any cached projection, mixed validation or RGB continuation. Source00/11projections were already observed in source captures, although they are excluded from equation inputs; mixed10/01projections were not yet executed. No claim of forecasting A/M formation is made.

Saved-only independent rebuilding with different matrix chunking reproduces all12sealed fields exactly. Versus CUDA BF16 native execution, P relative RMS error is0.031585–0.031861%; I error is0.016927–0.023570%. Native operation ordering/rounding retains small discrepancies:6583–6855image values differ out of3,145,728per exit. Maximum I error reaches4; these are not bit-exact native predictions.

Install the independently predicted000and111image exits and continue the real model:

| Predicted corner | Final RGB MAE | Maximum pixel error | Final latent L2 |
|---|---:|---:|---:|
|000/Tbase|0.248749|103|8.101807|
|111/Tshape|0.228909|44|7.085192|

Both satisfy the frozen mean-error forecast below1RGB level, while individual errors remain. Visual review retains the weak/strong appearances. Native attention, later MLPs, readout, scheduler and VAE still compute the future; this is a predicted internal update with a tested native continuation, not complete image prediction or a speedup benchmark.

## Running and auditing

Use the same installed environment, worker cache/model variables and scheduler wrapper as the preceding workflow. Stage and submit with identical environment values:

```
python -I qualification/submit_flux_step1_overlap.py RUN --source WRITERS_ROOT --onset-source ONSET_ROOT
python -I qualification/submit_flux_step1_overlap.py RUN --submit
python -I qualification/submit_flux_step1_overlap.py RUN --collect JOB
python -I qualification/analyze_flux_step1_overlap.py RUN AUDIT --source WRITERS_ROOT --onset-source ONSET_ROOT
python -I qualification/publish_flux_step1_overlap.py RUN AUDIT ATLAS
```

`flux_single_contraction.capture_candidate(..., logical_step=1)` now explicitly qualifies this clock. The default remains step3 and still refuses an undeclared step1 capture. Tests cover both clocks and all8RAM native frozen-block exits.

Job `job-3fa5d0f49cf5`:1load,0encodes/noise,1292native adapter transitions,36Acts,1328receipt emissions,18VAEs;8separate direct block evaluations and4native cached projections. Worker112.003seconds, runner167.07seconds, peakRSS8478.6MB/VRAM9558MB. One strict-preflight CUDA lease,10GB RAM/13GB VRAM,600s worker/720s lease, no retries. The original request estimated less than1GB extra RAM over the5405.8MB predecessor; actual peak rose3072.8MB. Retain that planning miss and use the new measured peak for future sizing. The run stayed within its reservation. The audit binds complete fields, actual Act payloads, untouched closure, per-row paths, latent trajectories and common future conditioning. Native text-unchanged flags are source-bound observations; full mixed text arrays were not independently rebuilt.

Eight public computation occurrences retain exact clock/receiver/source applicability, paired T futures, signed target/collateral effects and formation/semantic limits. Eight exact queries match;16changed clock/receiver queries abstain. Sources and outputs remain under `artifacts/mdb-step1-overlap-20261104/`.

The next mechanism cut is where the useful R and T differences form **before single6**, followed by clamp/rescue and a new prompt/seed prediction. The single6factorial identifies contextual carriers and conditional writer effects; it does not yet identify the upstream attention coalition or a universal narrow-eye operation.

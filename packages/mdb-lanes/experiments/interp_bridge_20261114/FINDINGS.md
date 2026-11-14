# Interp bridge (nnsight access and first-order attribution vs mass), FLUX.2 Klein 4B: findings (2026-11-14)

- Prereg: `PREREG.md` (hashes in `hashlog.txt`, committed 6eca107 before the smoke job; groups committed 10238f4 before the scored job). Seed 4242 only.
- Jobs (mrun, Beast): smoke job-8fd7bb3b1604 (worker 37.8 s, peak 9,495 MB allocated); designed cuts job-1d4b6df1ba24 (attempt 4; worker 81.3 s, peak 7,991 MB). Three earlier cut attempts were lost to scheduler clamps or cancelled (`jobs.json`).
- Sources: only the result files in this directory (`q1_table.json`, `designed.json`, `designed_id_direction_posthoc.json`, `saved_sets.json`, `reanalysis-gated.json`) and the independent audit `/Users/jakeholl/domains/artifacts/audit-predicted-write-interp-bridge-20261114.md`. Raw data: `/Users/jakeholl/domains/artifacts/mdb-interp-bridge-20261114/results/`.
- Labels: **MEASURED** = read from those files or recomputed from saved data by a script named here; **AUDIT** = from the audit, not re-derived by me; **ASSERTED** = my reading. This file was written after an audit, so its framing is not blind.

## Worst problems first

1. **"Attribution clearly beats mass" (Δρ = +1.03) holds mostly through the sign.** On sign-free effect size the gap on the primary (`anti`) partition is about +0.01 (MEASURED, `reanalysis-gated.json`). Mass is unsigned by construction and cannot order effects of opposite sign; 8–11 of the 20 anti groups per condition move the latent away from the deletion (MEASURED).
2. **The primary outcome is partly circular.** Attribution is the gradient of the same projection y that is scored, so it is a first-order forecast of the scored quantity with the direction built in (ASSERTED, AUDIT). On the other direction (iso − deleted, with its own attribution) the anti-partition gap is 0.150 against raw mass, at the prereg's "no difference" line (MEASURED; that analysis is post hoc).
3. **First-order prediction fails for large cuts.** Summed over the 20 groups, the measured total has the opposite sign to the predicted total in every zero-cut condition (ratio −1.01 to −24.2), and the first-order value for 540-head keep sets is near zero where their pixel effect is large (MEASURED; F2.4 and F2.5 failed).
4. **nnsight is an access layer here, not an independent check.** Its cuts call attention_port's own `edit`; the cut arithmetic was the port's (AUDIT, and the worker code).

## Q1: nnsight on Klein 4B (MEASURED, smoke job-8fd7bb3b1604, `q1_table.json`)

- **Capture is bitwise equal to the port's.** q, k, v, the native head output and the FP32 softmax recomputed from the nnsight q, k are bitwise equal to the port observer's (cell block 10, head 13, step 0; shapes [768, 128] and [768, 768]).
- **Cuts through nnsight are bitwise equal to the port's cuts.** Zero and mask cuts (block 10, head 13, `own:ball` sources, every step) give final latents and RGB bitwise equal to the port cuts, and differ from native (latent_rel 0.054 zero, 0.050 mask), so the nnsight write-back is not a silent no-op.
- **Roots and wrapping:** all 5 roots equal the lane reference hashes (native `a3cef09b…`); wrapping, a no-op trace and the capture trace leave the native latent unchanged.
- **Cost:** nnsight capture 0.337–0.356 s per generation against port capture 0.341–0.361 s and native 0.332–0.333 s; peak 7,760 MB against the port's 7,773 MB.
- **Gradient:** nnsight's gradient at block 10, step 3 differs from the plain-torch probe by at most 2.3e-6 (not bitwise). The audit reports this within the F1.6 bound of 1% of the gradient's maximum; the gradient maximum itself is not in the saved summary I read.
- **What this does and does not show (ASSERTED, agreeing with the audit):** nnsight delivers the same operands and writes a replaced output back at the same point. It does not show that any arithmetic is independent of the port (`port_edit` is attention_port's `edit`), nor that the port's cut is right for the model.

## Q2: attribution vs mass on the designed groups (MEASURED, cuts job-1d4b6df1ba24)

### As preregistered (`designed.json`)

Spearman over 20 groups of 30 heads, signed effect on u = L_native − L_del:

| condition | ρ attribution [LOO] | ρ read mass | gap | LOO gap min |
|---|---|---|---|---|
| red zero anti | 0.949 [0.940, 0.979] | −0.192 | 1.141 | 0.998 |
| red mask anti | 0.991 [0.989, 0.995] | −0.344 | 1.335 | 1.225 |
| ball zero anti | 0.955 [0.947, 0.968] | 0.116 | 0.839 | 0.667 |
| ball mask anti | 0.941 [0.932, 0.960] | 0.141 | 0.800 | 0.600 |
| red zero diag | 0.976 | 0.442 | 0.534 | 0.353 |
| red mask diag | 0.967 | 0.332 | 0.635 | 0.470 |
| ball zero diag | 0.973 | −0.032 | 1.005 | 0.874 |
| ball mask diag | 0.955 | −0.039 | 0.994 | 0.895 |

Primary Δρ = +1.029 (anti mean), all 4 conditions positive, LOO minima all above 0: **"attribution clearly beats mass"** by the prereg rule. The rule was applied correctly (AUDIT).

### Sign-free size, pixel gone, other direction, and gated mass (post hoc, `reanalysis_gated.py` → `reanalysis-gated.json`)

Gated read mass = read mass weighted per step and block by the RMS of the outer attention gate (image gate for image receivers, text gate for text receivers), gate log from job-6d9fea396262 (joint/single gate RMS 0.0184 image, 0.0586 text). Over the 20 groups it has Spearman 0.964–0.986 with raw read mass, because each group mixes strata. Means over the 4 conditions of each partition:

| outcome | partition | attribution | raw mass | gated mass | attribution − gated mass (conditions won; LOO min > 0) |
|---|---|---|---|---|---|
| signed effect (nd) | anti | 0.959 | −0.070 | −0.157 | +1.116 (4; 4) |
| **size \|effect\| vs \|r\|** | **anti** | **0.700** | **0.688** | **0.677** | **+0.023 (2; 1)** |
| size \|effect\| vs \|r\| | diag | 0.871 | 0.524 | 0.414 | +0.458 (4; 4) |
| pixel gone | anti | 0.641 | 0.100 | 0.082 | +0.558 (4; 2) |
| pixel gone | diag | 0.500 | 0.235 | 0.251 | +0.249 (2; 2) |
| signed effect, iso − del direction (r_id) | anti | 0.663 | 0.513 | 0.479 | +0.184 (4; 2) |
| size, iso − del direction (\|r_id\|) | anti | 0.711 | 0.639 | 0.595 | +0.117 (3; 2) |

- **Size only, anti partition, per condition** (attribution / raw mass / gated mass): red zero 0.650 / 0.644 / 0.659, red mask 0.729 / 0.642 / 0.693, ball zero 0.818 / 0.761 / 0.702, ball mask 0.605 / 0.704 / 0.654. Mean gap against raw mass +0.013: "no difference" by the prereg's 0.15 threshold if that threshold is applied to this post-hoc outcome. Gated mass does not improve on raw mass here (0.677 vs 0.688).
- **Diag partition:** attribution wins on size in all 4 conditions (0.81–0.93 vs gated mass 0.21–0.57).
- **Pixel gone:** ball favours attribution strongly (0.91–0.95 against 0.02–0.15). Red does not (anti 0.30–0.41 vs 0.06–0.15; diag 0.02–0.14 vs 0.27–0.51). Red's 30-head pixel effects lie at about the lane's floor size (AUDIT), so the red pixel test cannot settle the question.
- **Other direction** (iso − del): the anti-partition gap is +0.150 against raw mass (`designed_id_direction_posthoc.json`; recomputed here), +0.184 against gated mass. The groups were built from nd-direction attribution, so this is not a designed test for r_id.
- **Signs:** attribution gets the sign right in 15–20 of 20 groups per condition.
- Reading (ASSERTED): "effect ≠ read mass" is supported by the sign everywhere and by size on the diag partition. On the partition built to separate the two predictors, attribution and mass (raw or gated) order effect sizes equally well.

### First-order linearity (MEASURED, `designed.json`)

- Per-group fit is good: slope through the origin 0.49–1.37, Pearson 0.70–0.98.
- Summed over all 20 groups of a partition, the measured total has the opposite sign to the predicted total (ratio: red zero −1.01 / −1.39, red mask −1.31 / −1.04, ball zero −24.2 / −14.2, ball mask −2.22 / −1.73).
- Single-head spot checks (smoke): the top-removal head's measured change is 0.82× (red, block 1 head 15) and 0.91× (ball, block 7 head 11) of the prediction; median-removal heads have measured changes about 750–1,750× their near-zero predictions (|predicted| ≤ 4.3e-6, |measured| 0.0016–0.0042).
- Saved 540-head keep sets: first-order removal is small where the pixel effect is large, e.g. `z:keep-modI:ball` r = +0.016 with gone 1.12 (`saved_sets.json`; the two are in different units, latent y vs pixel gone).
- Reading (ASSERTED): first-order attribution ranks 30-head groups well and single large heads to within 20%, but it does not add up across groups and does not predict large cuts.

### Saved lane sets (secondary, `saved_sets.json`)

| condition | n sets | ρ attribution | ρ mass | gap (bootstrap 90%) |
|---|---|---|---|---|
| red zero | 39 | 0.701 | −0.067 | +0.768 [0.49, 0.99] |
| red mask | 13 | −0.159 | 0.637 | −0.797 [−1.32, −0.15] |
| ball zero | 39 | 0.561 | −0.261 | +0.821 [0.40, 1.18] |
| ball mask | 13 | 0.610 | 0.478 | +0.132 [−0.62, 0.81] |

The saved sets are nested (own60 within own120 within own240; keep sets are complements), so these values are not independent.

## Forecast outcomes (PREREG section 6 and Q1 forecasts; MEASURED unless marked)

| forecast | result |
|---|---|
| F1.1 nnsight installs and traces the adapter loop | held |
| F1.2 capture bitwise equal | held |
| F1.3 native latent unchanged by wrapping / traces | held |
| F1.4 nnsight cuts bitwise equal to port cuts | held |
| F1.5 nnsight capture ≤ 1.3× port time, peak within 100 MB | held (0.337–0.356 s vs 0.341–0.361 s; 7,760 vs 7,773 MB) |
| F1.6 gradient within 1% of max | held per the audit (max abs 2.3e-6); the bound's denominator is not in the summary I read |
| F2.1 attribution clearly beats mass (Δρ ≥ 0.40) | held by rule (Δρ +1.03; my forecast was about +0.4). Mostly sign, see above |
| F2.2 diag: ρ_r and ρ_M both > 0.5 in every condition | **failed** (ρ_M −0.04 to 0.44) |
| F2.3 top-removal single head within ±30% | held (0.82×, 0.91×) |
| F2.4 linearity ratio 0.5–1.5 for zero cuts | **failed** (summed ratios −1.0 to −24) |
| F2.5 first-order over-shoots keep sets; ρ_r − ρ_M ≥ 0.2 on zero-cut saved sets | **failed** on the first part (it under-shoots, near-zero r for keep sets with large effects); second part held (+0.77, +0.82) |
| F2.6 replay reproduces the adapter's final latent bitwise | held (red and ball iso roots) |

## Q3: sparse-autoencoder / transcoder survey

The survey of public SAE and transcoder weights (FLUX.2 Klein, FLUX.1, Qwen2.5-1.5B, Qwen3-4B) is recorded only in that session's report to the coordinator. No notes, URLs or files of it are saved in this directory or in the raw-data directory (searched for "SAE", "transcoder", "sparse autoencoder" and "http"), so I list nothing here.

## Not done, or not yet known

- Single seed (4242), two words (red, ball), prompt `cat` only.
- Whether attribution beats mass on size for groups built from a sign-free score: not yet known; the groups were built from signed nd-direction attribution.
- Whether a second-order or path-integrated attribution adds up across groups: not tested.

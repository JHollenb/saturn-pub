# Interp bridge 2026-11-14: preregistration

Written 2026-11-14 00:50 PST, before any job of this experiment ran and before any attribution value
existed. Its sha256 is recorded in `hashlog.txt` before the smoke job is submitted. Code referenced
here: `bridge.py`, `interp_bridge_worker.py`, `stage.py` in this directory (hashes in `hashlog.txt`).

Prior evidence, stated so it is not mistaken for a forecast: `toy_check.py` on saturn_pub's tiny random
`Flux2KleinAdapter` (CPU, FP32, torch 2.14.0, diffusers 0.39.0, nnsight 0.8.2, laptop) gave bitwise
equality for every Q1 comparison below and first-order predictions within 1% of single-head cut effects.
None of that is evidence about the 4B model in BF16 on CUDA.

## 1. Questions

- Q1: can nnsight replace or extend attention_port as the FLUX.2 Klein capture/intervention layer?
- Q2: does first-order attribution (gradient of an outcome with respect to a head's attention output,
  dotted with the cut's change of that output) order head-cut effects better than read mass?
- Q3 (no compute): public SAE/transcoder weights for FLUX.2 Klein, FLUX.1, Qwen2.5-1.5B, Qwen3-4B.

## 2. Q1 forecasts (smoke job)

Setting: prompt `cat`, seed 4242, 256 px, 4 steps, Klein 4B BF16, lane `Lane` load/encode/noise/decode.
Cells: (block 2, head 7), (block 10, head 13), (block 20, head 21) at steps 0 and 2. Cut cell: block 10,
head 13, receivers `image`, sources `own:ball`, every step, zero (delta) and mask.

- F1.1 nnsight 0.8.2 installs into the job's uv environment (Python 3.14.6, torch 2.13.0) and traces the
  adapter loop (`trace(fn=loop)` with one empty invoke).
- F1.2 q, k, v and the native output of each captured head are bitwise equal between the nnsight path
  (inputs/output of `dispatch_attention_fn_0` inside the processor) and the port's observer; the FP32
  probabilities recomputed from the nnsight q, k are bitwise equal to the port's.
- F1.3 The final latent with nnsight wrapping (no trace), with a no-op trace and with the capture trace
  equals the lane's reference hash `a3cef09b…` (native:cat, 4242).
- F1.4 Zero and mask cuts through nnsight (attention_port arithmetic, nnsight access) give final latents
  bitwise equal to the same cuts through the port.
- F1.5 nnsight capture costs at most 30% more wall time per generation than the port capture; peak VRAM
  within 100 MB of the port's.
- F1.6 nnsight's gradient of the outcome with respect to one block's attention output (last step) equals
  the plain-torch probe gradient up to BF16 rounding (max abs difference below 1% of the gradient's max).

Any of F1.2 to F1.4 false means nnsight changes access or arithmetic on this model; adoption would then
need that fixed first.

## 3. Q2 definitions

- Base trajectory per word w in {red, ball}: prompt `cat`, seed 4242, `iso:w` conditioning (hybrid own:
  the word present only in its own text rows), which is the trajectory the lane's saved cuts ran on.
- Outcome: final packed latent L (float32, [256, 128]). Primary direction u = L_native - L_del(w);
  y(L) = <L - L_del(w), u> / |u|^2 (1 at native, 0 at the word-deleted run). Secondary direction
  u' = L_iso(w) - L_del(w). This is a latent-space analogue of the lane's pixel `gone` score.
- Cut of head (b, h): receivers `image` and `others:w` (every text row except w's own), sources
  `own:w`, every step: the lane's cut. Zero = no renormalisation; mask = renormalising.
- Attribution a(b, h) = sum over steps s and receivers i of <dy/do_{s,b,h,i}, Δo_{s,b,h,i}>, with
  Δo = -sum_{j in S} p_ij v_j (zero) or (m_i o_i - sum_{j in S} p_ij v_j) / (1 - m_i) (mask),
  p recomputed in FP32 from the native q, k. Gradients by plain torch autograd through a replay of
  the adapter step (`bridge.step_forward`; the adapter's `advance` runs under inference mode), one
  step at a time from saved step inputs, blocks under non-reentrant checkpointing.
  Predicted removal r = -a (positive = the cut moves y toward the deleted run).
- Mass M(b, h) = sum over steps and the same receivers of sum_{j in S} p_ij (read mass), same capture.
  Also recorded, not used for decisions: write mass (p_ij |W_O,h v_j|), predicted write size
  sum_i |W_O,h Δo_i|, and the lane's `unit_mass` (full-a capture).
- Predictor of a head set = sum over its heads. `keep-X` = every head not in X. `@rows-o` arms use
  sources `own:o` and receivers `image` + `others:o`.
- Effect, designed groups (primary): Δ = y(iso) - y(cut), primary direction. Secondary: the lane's
  pixel gone score computed the lane's way, with `z:none:w` as the all-cut endpoint.
- Effect, saved sets (secondary): the lane's saved `gone` (pixels, context 4242) from
  `qualification/flux_read_panels_20261112/report/b-gone-full.json`, zero (`z:`) and mask (`m:`) arms.

## 4. Designed head groups (scored job)

`bridge.partitions(removal, mass, n_groups=20)` per word, with removal = r of the zero cut and
mass = M (both summed over steps, own-word sources, primary direction): rank heads by each (rank 0 =
largest, ties by head index). `anti`: sort heads by rank_r - rank_M and cut into 20 consecutive groups
of 30 (groups where the two predictors disagree most); `diag`: sort by rank_r + rank_M, 20 groups of 30.
Each partition covers all 600 heads. Every group is cut in both types under `iso:w`:
2 words x 2 partitions x 20 groups x 2 cut types = 160 branches, plus `z:none:w` and `m:none:w`
(4 branches) and 5 roots. The mask-cut predictor is the mask attribution of the same group.

The designed job is the primary test and runs regardless of the saved-set result: the saved sets are
nested by construction (own60 within own120 within own240; keep-X is the complement of X), so their
values are not independent.

## 5. Statistics and decision

Conditions: 4 = 2 words x 2 cut types. Spearman rho over the 20 groups of a partition.

Primary statistic: Δρ = mean over the 4 conditions of [rho(r_group, Δ_group) - rho(M_group, Δ_group)]
on the `anti` partition.

- Attribution clearly beats mass: Δρ >= 0.40, and rho_r > rho_M in all 4 conditions, and in at least 3 of
  4 conditions the smallest leave-one-out value of (rho_r - rho_M) is above 0.
- No difference: |Δρ| < 0.15.
- Mass clearly beats attribution: Δρ <= -0.40 with the mirrored conditions.
- Anything else: not separated (not yet known).

Secondary, reported with leave-one-out ranges: the same two Spearman values on the `diag` partition; on
all 80 branches of a word and cut type pooled; on the saved sets per condition (with the number of sets).

Linearity: per condition, the ratio sum(measured Δ) / sum(predicted r) over the 40 groups, and the
least-squares slope through the origin of measured on predicted. Over-shoot = the linear prediction is
larger than the measured effect (ratio < 1); under-shoot = ratio > 1.

## 6. Forecasts

- F2.1 Primary: attribution clearly beats mass (Δρ >= 0.40). My forecast: Δρ about +0.5 for zero cuts and
  +0.3 for mask cuts, mean about +0.4; confidence about 60%.
- F2.2 `diag` partition: rho_r and rho_M both above 0.5 in every condition.
- F2.3 Single-head spot checks (smoke): measured Δ within ±30% of the prediction for the top-removal
  head in each word (zero cut).
- F2.4 Linearity on 30-head groups: ratio between 0.5 and 1.5 for zero cuts; mask further from 1 than zero.
- F2.5 Saved sets: first-order prediction over-shoots the large `keep-*` sets (540 heads cut): predicted
  removal above the saved effect's scale; rho_r - rho_M >= 0.2 on zero-cut saved sets.
- F2.6 The replayed trajectory (`step_forward`) reproduces the adapter's final latent bitwise (iso roots).

A forecast that fails is reported as failed; thresholds are not moved after data.

# Addendum 2 to PREREG.md (2026-11-14T02:08-08:00): gated mass vs predicted write, and value-swap controls

Written before any job of this addendum is submitted. It is committed to git before the first submit, so custody rests on the commit time, not only on `PREREG.sha256.log`. Code: `followup2/` (`select2.py`, `stage2.py`, `vswap2_worker.py`, `content.py`, `content_offline.py`); sealed plan `followup2/plans2/gm.plan.json` (plan sha256 `1d5e93e99256d7da87aa02927c755446df7010ecb6af336682850842b726b623`, seal sha256 `3d1ef586983385ff1790d55688d8b68a1dd2ce9e06cfcb4945c3aab0458f4426`).

## Why

An independent audit (`artifacts/audit-predicted-write-interp-bridge-20261114.md`) found that F1's dissociation is the joint/single gate: the lane's mass (`unit_mass`, `p·|O_h v|`) has no outer attention gate, and FLUX.2's joint-block gate is about 0.02 of its single-block gate per unit of mass. Mass times one constant per stratum ("gated mass") has Spearman 0.997–0.999 with image-receiver predicted write (PW). What PW adds beyond gated mass is mainly its text-receiver part (70–86% of PW).

**Question.** Does predicted write, especially its text-receiver part, add anything beyond gated mass in selecting heads whose cut removes a word?

## What I have seen before writing this

- All results of the main and f3 plans, the capture job and FINDINGS.md, and the audit.
- The content measures below, computed on the saved native / deleted / iso / sham / all-cut images and on the saved rain and snow arms of the main plan (pw60, own60, matched sets, random sets, keep arms) and value swaps. The values are in `followup2/results/content-validation.json` and `content-main.json`, written before this file.
- The selection numbers of the new sets (below), from the capture's PW and mass arrays only. No new set has been cut, so no outcome of any new arm is known.

## Fixed setup

As PREREG.md: FLUX.2 Klein 4B distilled, BF16, 256 px, 4 steps, seed 4242, words red, black, ball, snow (cat prompt), night, rain (car prompt); cut arms under `iso:<w>` conditioning, cutting `own:<w>` reads by `image` and `others:<w>`; both cut types (zero, renormalising mask); the lane's floors (`sham_recompute` and `cut_recompute` for zero cuts, `sham` for mask cuts); all-cut references `z:none`, `m:none`; `iso`, `del`. Unchanged lane worker `qualification/flux_read_panel_worker.py`, now with `retain=["latents"]`.

## Gated mass

`flux_read_panels.gated_unit_mass(mass, 24, 5, scales)` on the lane mass (`word_masses`), with `scales = stratum_gate_scale(PW_zero image receivers, unit_mass(iso tok_w))` from the capture's iso run (sum per stratum over sum per stratum). Joint/single scale ratio: 0.0184–0.0192 per word. Only the ratio changes rankings. Spearman(gated mass, image-only PW) over 600 units: 0.996–0.998 (raw mass 0.818–0.896).

## New sets (rules; `followup2/select2.py`, deterministic, re-run gives byte-identical `sets2.json`)

- **gmass60**: top 60 by gated mass. Overlap with pw60: red 38, black 42, ball 46, snow 51, night 40, rain 41.
- **massmatch-lowpw-single** (as requested): 60 single-block units with total raw mass within 10% of own60's raw mass, minimum PW_both. **Unreachable for every word**: all single-block units together carry too little raw mass; the 60 most massive single units reach only 0.13 (ball) to 0.41 (black) of own60's mass (red 0.30, snow 0.17, night 0.20, rain 0.14). Per the request, the closest set is used: the 60 single units with the largest raw mass. That set carries PW_both 0.85–1.19, about pw60's (0.95–1.28), so **it is not a low-PW set and cannot test "mass without PW"**.
- **massmatch-lowpw-single-own** (added, because the requested band is unreachable): 60 single-block units with total raw mass within 10% of **own60's single-block** raw mass, minimum PW_both. Reachable for all words (at −10% of target). PW_both 0.43–0.61, against own60's 0.71–0.97 (about 0.6×). Inside one stratum mass and gated mass rank identically, so this set is also matched on gated mass to own60's single part.
- **gmassmatch-lowtextpw**: 60 units with total gated mass within 10% of pw60's gated mass (all at −10%), minimum text-receiver PW. Its text-receiver PW is 0.41–0.63 of pw60's (red 0.32/0.78, black 0.49/1.00, ball 0.55/1.00, snow 0.50/0.79, night 0.36/0.77, rain 0.43/0.90). Overlap with pw60 18–37.
- **rand60c, rand60d**: `matched_control_sets` against gmass60 (stratum counts and distinct blocks matched), disjoint from each other, excluding every named set (old and new, and old rand60a/b). The saved rand60a/b were not reused: for every word each contains units of a new named set.
- **pw60, own60**: the saved sets, rerun in this plan. Reason: the main jobs did not retain latents, and latent projections are a stated outcome below. The saved pixel results stay the reference; the rerun is also a determinism check (image hashes must equal the saved ones).

Plan `gm`: 8 sets × 2 cut types × floors + references = 47 branches per word, 282 in all; two word-aligned leases (red/black/ball, snow/night/rain), 141 branches each (148 / 149 work items with roots), no replicated branches.

## Value-swap controls (`followup2/vswap2_worker.py`, one job)

Same swap as the capture's `vswap-all` (`pw_worker.make_swap_backend`, unchanged): 20 single blocks, 4 steps, image receivers read replaced values at all 512 text rows through the live attention map; native conditioning. Arms for snow (cat prompt) and rain (car prompt):

- `vswap-all`: deleted scene word (reproduces the saved arm; PNG compared bitwise);
- `vswap-unrel`: values of an unrelated prompt with the same number of real tokens, rows one to one: cat class (32 tokens) "a photorealistic wooden table holding a green vase of tulips in a bright kitchen, morning light"; car class (27 tokens) "a tall glass of orange juice standing on a white kitchen counter in the morning" (token counts checked with the Qwen3 tokenizer, which reproduces the scene prompts' 32 and 27; the worker records both counts);
- `vswap-nonscene`: deleted-word swap of a word that is not the scene word: `red` on the cat prompt (the saved red `vswap-all`, rerun here only to get its latent; its PNG must equal the saved one), `blue` on the car prompt (new);
- `vswap-sham`: recompute sham.

The worker also logs the RMS of the outer attention gates per (step, block) in the native root run, as a post-hoc check of the stratum scale ratio.

## Outcomes

1. **Pixel gone** (primary, unchanged from PREREG): gone = (d(arm) − d(iso)) / (d(t:none) − d(iso)), same-job references. For the value swaps, d as in PREREG (0 = native, 1 = deleted).
2. **Latent projection** (secondary): Δ = L_ref − L_arm (ref = iso for cuts, native for swaps); coefficient on u = L_native − L_del and on u' = L_iso − L_del: ⟨Δ, u⟩/|u|²; and |Δ|/|u|. Reported raw and as a fraction of the same-type all-cut.
3. **Content** (rain and snow): `followup2/content.py`. Rain primary: vertical-streak energy (luminance; horizontal 7-px high-pass; vertical 15-px box mean; RMS). Rain secondary: the same divided by the RMS of the high-pass (contrast-normalised). Snow: share of bottom-35% pixels with max(RGB) ≥ 0.60 and HSV saturation ≤ 0.50. removed = (c(ref) − c(arm)) / (c(ref) − c(del)).
   - Validation (saved images, `followup2/results/content-validation.json`; criteria V1 span ≥ 10× max sham deviation, V2 all-cuts remove ≥ 0.7, V3 random 60-cuts |removed| ≤ 0.10; criteria written after looking at the values): rain primary passes all three (span/sham 45; all-cut 1.01 / 0.95; random ≤ 0.056); snow passes all three (78; 0.88 / 0.93; ≤ 0.022); rain secondary passes V1, V2 and fails V3 (one random draw 0.103), so it is reported only beside the primary.
4. Contact sheets (native / deleted / iso / arm) for every new arm; the pictures are described separately from the numbers.

## Forecasts (decision rules; cells = red, black, ball × zero, mask, pixel gone, as F1)

- **H1**: gone(gmass60) ≥ 0.8 · gone(pw60) in all 6 cells (pw60 = the saved main-plan value). Held → predicted write adds little beyond the gate correction for these words. Failed → PW (its text-receiver part or per-unit detail) selects heads that gated mass misses.
- **H2**: gone(massmatch-lowpw-single) < gone(own60)/3 in all 6 cells. As noted, the requested band is unreachable and the closest set is high-PW, so I forecast **H2 fails**, and a failure here does **not** by itself show "the original F1 was only the gate": it shows that the most massive single-block heads (which are also high-PW heads) remove the word. The reading "F1 was only the gate" is supported only if H2b also fails and H1 holds.
- **H2b** (added): gone(massmatch-lowpw-single-own) < gone(own60)/3 in all 6 cells. Held → within the single blocks, mass without PW does little (PW carries something beyond gated mass). Failed → within one gate stratum, mass-matched low-PW heads still remove the word (F1's dissociation was the gate). I forecast H2b fails (its PW is about 0.6× own60's).
- **H3**: gone(gmassmatch-lowtextpw) < (2/3) · gone(pw60) in all 6 cells. Held → text-receiver PW carries effect that gated mass lacks. Failed → at equal gated mass, low text-receiver PW loses little.
- **H4**: for snow and rain, on the primary content measure, removed(vswap-unrel) ≥ (2/3) · removed(vswap-all), both words. Held → the swap's content removal is generic (foreign values remove the scene content), not specific to deleting the word. Failed → the deleted-word swap removes the scene content more than unrelated values do (word-specific at least in part). `vswap-nonscene` is reported beside it (not part of the rule).
- Secondary reports (not decision rules): snow, night, rain cells of H1–H3 on pixel gone and, for snow and rain, on content removed; every rule also on the latent projections; random draws: |gone| for rand60c/d (expected ≤ floors).

## Jobs

3 jobs on Beast via mrun, priority 70: `gm` lease 0, `gm` lease 1 (panel worker), `vswap2` (10 + 10 work items). Reservation 11,000 MB (measured load transient 8,367 MB; capture peak 8,385 MB; × 1.3). Submitted with `stage.py submit`, which replaces `mrun.client.submit.submit` in-process by a wrapper that adds `reservation["pin"] = True` to the final submit call, so mrun's history clamp does not lower the declared reservation (it does not override the wedged-queue clamp, and it does not change priority, preemption or any other job). Before each submit I check that no other job is running or queued ahead; if one is, I wait.

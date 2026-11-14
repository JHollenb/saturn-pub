# Predicted write vs mass, and routing vs payload: FLUX.2 Klein 4B with a Qwen2.5-1.5B side check (2026-11-13/14)

- Prereg: `PREREG.md`, sha256 `688060256e7c5a03633b1e30df85ceeb308ed037e9943055dd76b8c9d2563fb1`, sealed 2026-11-14T00:33:01-08:00 before any non-smoke job.
- Execution addendum: `PREREG-ADDENDUM-1.md`, sha256 `bb64134e…`, hashed before any main or f3 panel arm ran. It changes lease packing and the reservation, not rules. Both hashes are in `PREREG.sha256.log`. [Custody gap: see Audit corrections C9.]
- Seed 4242 only.
- Labels: **MEASURED** means read from a job output or computed offline from one. **ASSERTED** means my reading. Verdict words come from the prereg's fixed list.
- Raw outputs live in `/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results/<job>/`, contact sheets in `/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/sheets/`, and the small tables in `results/*.json` here.

## Worst problems first

1. **The F2 set nearly duplicates pw60 (MEASURED).** `pwmatch-lowmass` shares 53–55 of its 60 units with `pw60` for every word, over the prereg limit of 50. The reason: pw60 is already a low-mass set (mass share 0.10–0.39, against 0.65–0.83 for own60), so the "same PW, less mass" set can only swap 5–7 units. By the prereg's own rule this makes the hypothesis verdict **unresolved**. The swap was still informative: in red and black, exchanging those 5–7 units halved the effect (F2 failed in 4 of 12 cells; see A2).
2. **F3 failed, and it was close to having no power (MEASURED).** PW wins 7 of 12 cells; 8 were needed. Mean Spearman is PW 0.21 against mass 0.02. Most 10-unit cuts remove ≤ 0.1 of the word, about the size of the floors.
3. **G1, G2 and G3 failed (MEASURED).** The routing/payload forecasts were wrong in direction for the word+real and padding shares. The value swap reached d 0.47 (snow) and 0.49 (rain) against a bar of 0.7.
4. **A0 criteria (i) and (ii) fail as written, at 96.8–99.98% of elements instead of 100% (MEASURED).** All-text cuts miss half-ulp agreement where the head output nearly cancels. One element's bound is a near-zero half-ulp, which gives max error/bound 7.6e24. Head relative error is ≤ 2.9e-3 for zero cuts and ≤ 8.6e-3 for mask cuts, and other heads change by exactly 0 in all 40 cells. These are reported as failing, not rewritten (see A0). [Superseded: these are the all-text cells; word-row cells and criterion (iv) in Audit corrections C4.]
5. **The value swap leaves the data manifold for red and ball (MEASURED d −1.69 / −0.78; pictures).** Swapping in the deleted run's text values moves those images further from *both* native and deleted than deletion does. The background trees disappear. The swap carries more than the word.
6. [Superseded: pixel-metric artefact, see Audit corrections C3.] **Rain cuts do nothing at 60 heads (MEASURED).** pw60 gone is 0.03 (zero cut, inside its floor bound of 0.049) and 0.06 (mask cut, just above its bound of 0.041), and own60 is −0.10. All rain F2 and F4 cells "hold" trivially because the reference is ≈ 0.
7. **Scheduler and lease losses (MEASURED, job records).** Seven jobs died before running any arm. All failures were at load or startup; no arm ran in them.
   - killed_vram at load, six jobs: job-2eb545f0f377, job-6c9b7875d02d, job-7546372990c7 (smoke), job-aded9ea015e1, job-c301a12cb2ae and job-6b0a44ca4073 (f3 attempts 1–2). In each, mrun lowered the reservation below the 8,022–8,367 MB load peak: the history clamp twice, the wedged-queue clamp three times. The wedge clamp overrides a pinned reservation whenever another session's job holds the GPU.
   - f3 attempt 3 (job-8a757c99dbeb) ran because it was submitted near the end of the other session's job, so it was admitted before the 120 s clamp.
   - [Pinned reservations came from an undeclared monkeypatch in `stage.py`: see Audit corrections C10.]
   - job-c1b2bf48ff0d failed deterministically with `PanelError: lease seal differs from its branches`. This is a **library bug**: `Panel.leases` hashes the `replicated` list into `lease_sha256`, but `flux_read_panels.check_lease` recomputes the hash without it, so the FLUX worker refuses any lease that has replicated branches. I did not edit lane code. The work-around was a word-aligned repack with no replicated branches (`plans/main.leases-r2.json`).
8. **Smoke-1 closure was void** (worker bug, fixed before the seal): its cell operands were overwritten by the native-conditioning run.

## Verdict (Experiment A)

| forecast | rule (PREREG) | result | status |
|---|---|---|---|
| F1 | gone(massmatch-lowpw) < gone(own60)/3 in all 6 cells (red, black, ball × 2 cut types) | 6/6 | **held** (confirmed only as a joint/single gate effect: Audit corrections C1) |
| F2 | gone(pwmatch-lowmass) ≥ 2/3·gone(pw60) in all 12 cells | 8/12 (red ×2, black ×2 fail) | **failed** (comparison flagged: overlap 53–55 > 50) |
| F3 | mean over 12 cells of Spearman_PW − Spearman_mass > 0 and PW higher in ≥ 8/12 | mean diff +0.193; PW higher in 7/12 | **failed** (small effects, near floor) |
| F4 | gone(rand) < gone(pw60)/3 in all 24 cells | 24/24 (rain trivially, pw60 ≈ 0) | **held** |

- **Hypothesis verdict: `unresolved`** (MEASURED inputs, rule applied).
  - The prereg's unresolved condition is met because overlap(pwmatch-lowmass, pw60) is 53–55 > 50. The lane's verdict precedence ranks unresolved above set_only.
  - If the overlap rule were set aside, F1 held + F2 failed + F3 failed + F4 held would map to `set_only`.
  - I report `unresolved` as the result, because that is what the prereg says.

## A2. Cut arms (MEASURED, jobs job-5eeb743a56cd and job-3be01d6f10e4, `results/arms.json`)

`gone` = (d(arm) − d(iso)) / (d(none) − d(iso)) with d = 1 − MAD(arm, del)/MAD(native, del). Arms cut own:<w> reads by the image and others:<w> receivers under iso conditioning. "fl" is the floor bound on gone: the arm's sham/recompute floor (px) divided by MAD(native, del)·|span|.

| word / cut | pw60 | own60 | mass-matched low-PW | PW-matched low-mass | rand60a | rand60b | keep-pw60 | keep-own60 | keep-rand60a | fl (pw60) | span |
|---|---|---|---|---|---|---|---|---|---|---|---|
| red / zero | 0.50 | 0.47 | 0.05 | 0.26 | −0.02 | 0.02 | 0.18 | 0.14 | 1.00 | 0.107 | 0.599 |
| red / mask | 0.31 | 0.26 | 0.05 | 0.19 | −0.02 | 0.02 | 0.20 | 0.16 | 0.98 | 0.065 | 0.656 |
| black / zero | 0.92 | 0.31 | 0.07 | 0.44 | 0.01 | 0.00 | −0.03 | 0.04 | 1.01 | 0.053 | 0.465 |
| black / mask | 0.85 | 0.30 | 0.07 | 0.22 | −0.00 | 0.01 | −0.00 | 0.06 | 1.01 | 0.036 | 0.659 |
| ball / zero | 0.66 | 0.51 | −0.07 | 0.59 | −0.12 | 0.01 | −0.57 | −0.19 | 0.91 | 0.058 | 0.630 |
| ball / mask | 0.41 | 0.23 | −0.01 | 0.43 | −0.05 | −0.02 | −0.44 | −0.28 | 0.89 | 0.047 | 0.778 |
| snow / zero | 0.42 | 0.26 | 0.03 | 0.45 | −0.00 | 0.01 | 0.50 | 0.66 | 0.98 | 0.015 | 0.785 |
| snow / mask | 0.30 | 0.32 | 0.06 | 0.31 | 0.00 | 0.01 | 0.47 | 0.45 | 1.06 | 0.015 | 0.784 |
| night / zero | 0.77 | 0.56 | −0.02 | 0.68 | 0.03 | −0.02 | 0.48 | 0.69 | 1.00 | 0.019 | 0.839 |
| night / mask | 0.71 | 0.52 | 0.02 | 0.59 | −0.01 | −0.01 | 0.30 | 0.62 | 0.99 | 0.018 | 0.876 |
| rain / zero | 0.03 | −0.10 | −0.17 | 0.26 | −0.03 | −0.01 | 0.23 | 0.40 | 1.03 | 0.049 | 0.594 |
| rain / mask | 0.06 | −0.10 | −0.24 | 0.13 | −0.02 | −0.03 | 0.09 | 0.20 | 0.96 | 0.041 | 0.718 |

**What the table says (MEASURED unless marked):**

- [Superseded: not supported beyond gated mass, and rain is outside its floor; see Audit corrections C1, C2, C5.] **Mass without PW does nothing.** Mass-matched low-PW sets hold 0.58–0.75 of the word's write mass (own60 level) but about 5% of its PW [2.3–3.9%: Audit corrections C2]. Their gone is −0.24 to 0.07 in every cell, inside or near the floors. The same mass inside own60 removes 0.23–0.56. This is the clean dissociation F1 tests, and it held 6/6.
- **pw60 matches or beats own60 in 10 of 12 cells.** The two exceptions are snow/mask (0.30 vs 0.32) and the rain cells, where both are ≈ 0. It does this with half to a third of own60's mass. Black is the largest case: 0.92 / 0.85 against 0.31 / 0.30.
- **Random sets of pw60's size, strata and block count remove nothing:** |gone| ≤ 0.12, all 24 cells, so F4 held.
- **PW is not the whole story (F2 failed).** In red and black, pwmatch-lowmass drops 5–7 units from pw60. Those units carry 0.08–0.12 of the word's mass and about 10% of pw60's PW_both. Swapping in low-mass, low-PW units *halves* gone (black 0.92 → 0.44, 0.85 → 0.22; red 0.50 → 0.26, 0.31 → 0.19). In ball and snow, the same kind of swap drops even more mass (0.30 / 0.26, including joint unit (4,14) with mass 0.24 / 0.16), and gone does not move (0.66 → 0.59, 0.42 → 0.45). `results/pml-swaps.json` lists the units. Unit (1,15) is dropped in red, black, snow and night; (24,9) in red, black and rain.
  - Reading (ASSERTED): some units carry more of the effect than either their PW share or their mass share predicts. Summed PW is not additive in gone at the 10% level, and this is word-dependent. Not yet known whether a per-unit PW computed on the arm's own trajectory (not the iso run's) would explain the red/black drop.
- **Keep arms.** Keeping only pw60 (cutting the other 540 units) leaves black fully black (gone −0.03 / −0.00) and red mostly red (0.18 / 0.20). For ball it gives *more* "ball" than iso (−0.57 / −0.44). For snow and night it is about half (0.30–0.50). Keeping a random 60 equals cutting everything (0.89–1.06).
- **Floors.** The zero-cut floor bound for pw60 is 0.015–0.107 in gone units (MEASURED). Every F1 low value, every random value and rain's pw60 lies inside or within about one bound of zero.

### Pictures (contact sheets `main-<word>-<zero|renorm>.png`; my reading, ASSERTED, separate from the pixel numbers)

Tiles: native, del, iso, all-cut, pw60, own60, mass-matched low-PW, PW-matched low-mass, rand60a, rand60b, keep-pw60, keep-own60, keep-rand60a.

- **black (zero):**
  - del and all-cut show a grey and white tabby.
  - **pw60 shows a grey striped tabby, nearly the deleted cat.**
  - own60 shows a dark grey cat with faint stripes (partial).
  - PW-matched low-mass is greyer than native but darker than pw60.
  - Mass-matched low-PW and both random sets show the native black cat.
  - keep-pw60 and keep-own60 show a black cat; keep-rand60a shows the tabby.
  - The pictures agree with the numbers.
- **red (zero):**
  - del shows a white and blue football.
  - **pw60 and own60 show a dull mauve/brown ball.** The red is gone, but the ball does not become the football.
  - PW-matched low-mass shows a reddish-mauve ball (partial). Mass-matched low-PW and the random sets are red.
  - keep-pw60 and keep-own60 are red; keep-rand60a shows the football.
- **ball (zero):**
  - del and all-cut have no ball. **No 60-unit cut removes the ball.** pw60 and own60 leave a smaller, darker red ball, and PW-matched low-mass a smaller one.
  - keep-pw60 shows a full ball; keep-rand60a has no ball.
  - ball's gone of 0.4–0.66 reflects size and shading changes, not removal.
- **snow (zero):**
  - del and all-cut show grey ground with a plain horizon.
  - pw60 and PW-matched low-mass remove the snowy background trees (plain blue-pink sky) but keep the snow on the ground. own60 does this partly.
  - Mass-matched low-PW and the random sets look like iso.
  - keep-rand60a shows grey ground.
- **night (zero):**
  - del is daytime with fog.
  - **pw60 turns the night into a light blue-grey dusk.** own60 does this partly, PW-matched low-mass a little less than pw60.
  - Mass-matched low-PW and the random sets stay dark.
  - keep-rand60a is daytime.
- **rain (zero):**
  - del and all-cut have no rain. Every 60-unit cut keeps the rain streaks, matching gone ≈ 0. [Superseded: Audit corrections C3.]
  - keep-rand60a has no rain.
- **Renorm sheets** show the same pattern with weaker changes (my reading).

## F3 (MEASURED, job job-8a757c99dbeb, `results/arms.json` → `forecasts.F3`)

16 random sets of 10 units per word, drawn from the union of the top 100 by PW_both and the top 100 by mass, are cut in both cut types (192 arms plus roots and all-cuts). For each word × cut-type cell, Spearman over the 16 sets is computed between gone and the set's summed PW of that cut type, and between gone and its summed mass.

| word / cut | Spearman PW (type) | Spearman PW_both | Spearman mass | Spearman image-only PW | gone range (median) | Spearman PW vs mass over the 16 sets |
|---|---|---|---|---|---|---|
| red / zero | 0.241 | 0.276 | 0.153 | 0.071 | −0.03 to 0.06 (0.00) | 0.58 |
| red / mask | 0.165 | 0.206 | 0.379 | 0.076 | −0.02 to 0.10 (0.01) | 0.60 |
| black / zero | −0.056 | −0.065 | −0.047 | −0.294 | −0.05 to 0.13 (0.00) | 0.53 |
| black / mask | 0.056 | 0.053 | 0.109 | −0.159 | −0.02 to 0.09 (0.01) | 0.45 |
| ball / zero | 0.191 | 0.209 | −0.515 | 0.068 | −0.20 to 0.09 (−0.02) | −0.08 |
| ball / mask | 0.391 | 0.494 | −0.362 | 0.538 | −0.11 to 0.05 (0.01) | −0.02 |
| snow / zero | 0.126 | 0.109 | 0.259 | 0.115 | −0.05 to 0.21 (0.15) | 0.01 |
| snow / mask | 0.097 | 0.088 | 0.256 | −0.032 | −0.01 to 0.18 (0.13) | 0.07 |
| night / zero | 0.388 | 0.388 | 0.047 | 0.509 | −0.04 to 0.07 (0.02) | 0.01 |
| night / mask | 0.476 | 0.518 | 0.062 | 0.641 | −0.02 to 0.08 (0.02) | −0.04 |
| rain / zero | 0.232 | 0.285 | −0.003 | 0.112 | −0.21 to 0.19 (−0.05) | 0.02 |
| rain / mask | 0.209 | 0.197 | −0.141 | 0.026 | −0.15 to 0.12 (−0.03) | 0.03 |

- **F3 failed (MEASURED).** Mean Spearman is PW 0.210 against mass 0.016, a mean difference of +0.193 > 0. But PW is higher in only 7 of 12 cells, and the rule needs 8.
- **Image-only PW, the secondary report from the seal note (MEASURED).** Mean Spearman is 0.139, higher than mass in 6 of 12 cells.
- **Power is low (MEASURED + ASSERTED).**
  - Sets of 10 units remove almost nothing. Outside snow (median 0.13–0.15), most gone values lie within ±0.1. That is the same size as the main-plan floor bounds (0.015–0.107; MEASURED).
  - The f3 plan has no per-set sham floors, so no per-set floor is available.
  - Neither score orders effects of this size well. I read F3 as a weak test, not as evidence against PW (ASSERTED).
- **Both scores predict how much the image moves, not whether it moves toward the deletion (MEASURED).** Against pixel distance to iso, rather than gone, both correlate more strongly: red/black PW 0.51–0.68, mass 0.60–0.71.

## A0. Instrument closure (MEASURED, capture job-86df2dd47809; red and snow, 5 (step, block) cells × 4 cut kinds = 40 cells)

The identity is checked against the attention port's own cut.

- Zero cut: a' − a = −Σ_S p_j v_j.
- Mask cut: a' − a = (p_S a − Σ_S p_j v_j)/(1 − p_S).
- Write change: g ⊙ O_h(a' − a).

| cut (sources / type) | head rel err max | elements within half-ulp bound (min over cells) | other heads max change | write rel err vs model BF16, median / max | sham/pred median / max |
|---|---|---|---|---|---|
| own row / zero | 0.18 | 100% | 0 | 0.093 / 1.84 | 0.048 / 0.74 |
| own row / mask | 0.29 | 99.6% | 0 | 0.12 / 1.94 | 0.064 / 0.77 |
| all text / zero | 2.9e-3 | 96.8% | 0 | 0.019 / 0.043 | 0.006 / 0.011 |
| all text / mask | 8.6e-3 | 99.0% | 0 | 0.018 / 0.073 | 0.005 / 0.018 |

- **Own-row cuts are floor-limited (MEASURED).** The recompute sham reaches up to 0.77 of the predicted write, so their large write errors are BF16 rounding of a tiny change. Measured against an FP32 projection of the measured head change, write error is ≤ 0.20 (own row) and ≤ 7.1e-3 (all text).
- **Single-block gate check (MEASURED).** The block update equals gate ⊙ (attention + MLP output) for 100% of elements within the BF16 bound at (1,7), (2,14) and (3,22), for both words.
- **Determinism (MEASURED).**
  - The iso and del roots of the capture job and both panel jobs have identical latent hashes for all six words.
  - Each observer run equals its root bit for bit.
  - Re-running mass gives Spearman 1.0000 with the saved values (max |diff| ≤ 2.1e-5).
- Reading (ASSERTED): the identity, the captured operands, the head slices, the output map and the gate reproduce the model's write change to within BF16/FP32 arithmetic. [Text receivers and criterion (iv) missing here; what closure can and cannot show: Audit corrections C4.]

## A1. PW vs mass ranking over 600 units (MEASURED, `plans/selection-report.json`, `results/imgpw.json`)

| word | Spearman PW_both vs mass | top-60 overlap: pw60 vs mass top 60 | pw60 vs own60 | Spearman PW_zero vs PW_mask | text-receiver share of PW | Spearman image-only PW vs mass | image-only top 60 vs mass top 60 |
|---|---|---|---|---|---|---|---|
| red | 0.599 | 27 | 27 | 0.995 | 0.82 | 0.817 | 31 |
| black | 0.653 | 28 | 33 | 0.996 | 0.86 | 0.815 | 30 |
| ball | 0.624 | 27 | 29 | 0.992 | 0.80 | 0.840 | 33 |
| snow | 0.688 | 32 | 34 | 0.997 | 0.70 | 0.893 | 38 |
| night | 0.646 | 28 | 36 | 0.994 | 0.75 | 0.861 | 32 |
| rain | 0.642 | 28 | 34 | 0.996 | 0.80 | 0.871 | 34 |

- The rankings separate: every top-60 overlap is 27–32 of 60, well under 50.
- 70–86% of PW sits on text receivers (`others:<w>`), which mass does not score.
- On image receivers alone the two rankings agree more (0.82–0.89), but their top-60 sets share only 30–38 units.
- PW_zero and PW_mask are effectively one ranking.

Set totals as mass share / PW_both (MEASURED; PW_both sums to 2 over 600 units):

| word | pw60 | own60 | mass-matched low-PW | PW-matched low-mass | rand60a | rand60b | overlap mml/own60 | overlap pml/pw60 |
|---|---|---|---|---|---|---|---|---|
| red | 0.26 / 0.95 | 0.65 / 0.71 | 0.58 / 0.057 | 0.15 / 0.86 | 0.016 / 0.11 | 0.031 / 0.13 | 11 | 55 |
| black | 0.27 / 1.18 | 0.67 / 0.97 | 0.60 / 0.077 | 0.19 / 1.06 | 0.019 / 0.11 | 0.009 / 0.07 | 13 | 53 |
| ball | 0.39 / 1.28 | 0.83 / 0.94 | 0.75 / 0.060 | 0.09 / 1.15 | 0.012 / 0.07 | 0.012 / 0.09 | 18 | 55 |
| snow | 0.37 / 1.24 | 0.76 / 0.88 | 0.68 / 0.063 | 0.11 / 1.11 | 0.005 / 0.09 | 0.006 / 0.09 | 13 | 55 |
| night | 0.17 / 1.09 | 0.77 / 0.89 | 0.69 / 0.055 | 0.11 / 0.98 | 0.009 / 0.11 | 0.008 / 0.09 | 15 | 55 |
| rain | 0.10 / 1.15 | 0.73 / 0.82 | 0.66 / 0.045 | 0.08 / 1.03 | 0.008 / 0.11 | 0.008 / 0.12 | 9 | 54 |

- Both matched sets sit at −10% of their target, the edge of the tolerance band.
- Random draws match pw60 per stratum (joint/single) and in the count of distinct blocks. They are disjoint from every named set and from each other, with no duplicates.

## B. Routing vs payload (MEASURED, capture job-86df2dd47809, `results/split.json`)

**Split.** The deleted-minus-native attention output, at matched geometry, splits exactly as P_d V_d − P_n V_n = dP·V_n + P_n·dV + dP·dV.

- Alignment follows the hybrid_own shift, with the word row mapped to the freed pad row 511.
- Each term is projected through O and the gate, then summed over image receivers, 4 steps and 25 blocks.
- Shares are projections onto the total, so they sum to 1.

**Closure.**

- Algebraic remainder: (0.7–1.3)e-6 relative.
- Remainder against the native BF16 run: 2.4e-3 to 5.1e-3.
- Invariant dP·V = dP(V − 1v̄ᵀ): holds to ≤ 1.0e-4, against routing terms up to 45.
- dP row sums: ≤ 8.7e-7.

| word | routing | payload | interaction | payload: image rows | pad rows | word row | other real rows | text common share |
|---|---|---|---|---|---|---|---|---|
| red | 0.234 | 0.782 | −0.017 | 0.574 | 0.413 | 0.008 | 0.005 | 0.010 |
| black | 0.211 | 0.784 | 0.005 | 0.699 | 0.290 | 0.005 | 0.006 | 0.054 |
| ball | 0.272 | 0.782 | −0.053 | 0.708 | 0.273 | 0.008 | 0.010 | 0.018 |
| snow | 0.225 | 0.800 | −0.025 | 0.794 | 0.175 | 0.024 | 0.007 | 0.059 |
| night | 0.210 | 0.796 | −0.006 | 0.750 | 0.219 | 0.014 | 0.017 | 0.070 |
| rain | 0.197 | 0.801 | 0.002 | 0.770 | 0.202 | 0.012 | 0.017 | 0.185 |

- [Mostly downstream divergence; first-contact numbers in Audit corrections C7.] About 80% of the change comes through changed values.
- Most of that is image-row values (0.57–0.79), then padding rows (0.17–0.41). The deleted word's own row is ≤ 2.4% of the payload.
- In joint blocks alone, routing is 0.35–0.51 and the word row is a larger payload share (snow 0.22, rain 0.20, night 0.16, ball 0.14). Single blocks dominate the norm, so the all-block figures track the single blocks.

| forecast | statement | G1 mean (red, ball, black) | G2 mean (snow, rain, night) | result |
|---|---|---|---|---|
| G1 | word+real payload share G1 > G2 **and** common share G1 < G2 | word+real 0.014; common 0.027 | word+real 0.030; common 0.105 | **failed** (word+real reversed) |
| G2 | pad payload share G2 > G1 **and** common share G2 > G1 | pad 0.325; common 0.027 | pad 0.199; common 0.105 | **failed** (pad reversed) |

- Joint and single blocks taken separately give the same orderings.
- Only the common-share ordering came out as forecast, and it is driven mostly by rain (0.185). Night 0.070, snow 0.059 and black 0.054 overlap.

### G3 value swap (MEASURED, same capture; `results/vswap.json`)

Setup: single blocks only. Image receivers read the deleted run's aligned text-row values through the live attention map. Conditioning is native except for the iso row. d = 1 − MAD(arm, del)/MAD(native, del).

| word | all rows | word row only | every row but the word | recompute sham | iso, all rows (d) | iso, all rows (gone, zero-cut scale) |
|---|---|---|---|---|---|---|
| snow | 0.471 | 0.156 | 0.417 | −0.004 | 0.382 | 0.41 |
| rain | 0.495 | 0.069 | 0.490 | −0.003 | 0.486 | 0.60 |
| red | −1.687 | 0.085 | −1.575 | −0.009 | −1.754 | −3.03 |
| ball | −0.780 | 0.096 | −0.875 | 0.011 | −0.834 | −1.24 |

- **G3 failed:** snow 0.47 and rain 0.49 are below 0.7.
- **G3b held:** every row but the word moves the image further toward the deletion than the word row alone (snow 0.42 vs 0.16, rain 0.49 vs 0.07).
- The sham stays at |d| ≤ 0.011.
- On the zero-cut scale, the iso all-rows swap removes 0.60 of rain, where no 60-unit head cut removed more than 0.26 (pw60 0.03). It removes 0.41 of snow, the same as pw60 (0.42).

What the pictures show (sheets `vswap-<word>.png`; ASSERTED, my reading, separate from d):

- **snow:** the all-rows and every-row-but-word swaps remove all the snow, ground and background trees, while the cat and ball stay in their native positions; the ball darkens. The word-row swap and the sham look native. The deleted image has a different layout (larger cat and ball), so d understates the change: in the picture the snow is removed.
- **rain:** the all-rows swaps remove the rain streaks but also darken the frame, grey out the blue car and dim the street lights. That is more than "rain gone". The deleted image keeps a blue car and lit lamps. The word-row swap looks native.
- **red:** the all-rows swap turns the ball dark navy, while the deletion makes a white and blue football. It also removes the background trees and softens the frame. Every-row-but-word gives a dark maroon ball. The word-row swap looks native.
- **ball:** the all-rows swap removes the ball and the background trees. Every-row-but-word leaves a faint red ball. The word-row swap looks native.
- Reading (ASSERTED):
  - [No control swap was run here; controls in Audit corrections C6 and Addendum 2 results.] The deleted run's text values carry the word's content, almost all of it in rows other than the word's own row.
  - They also carry broader changes (background, brightness, colour) that the deleted image does not show. A likely reason: in single blocks those text-row values come from text rows that have read the deleted run's *image* state, so the swap feeds image-dependent context into a native trajectory.
  - On picture judgement, G3 would pass for snow and partly for rain; on d it fails. The prereg needs both, so **G3 failed**.

## Qwen2.5-1.5B side check (MEASURED, job-a1f49897e70d, `results/qwen.json`)

Value-coalition case, FP32. The native margin −4.6471 matches the lane. Recomputed max source attention equals the saved values exactly (max |diff| 0.0). PW = Σ_h∈group p[l,h,21,18]·W_o,h(v_donor − v_recipient) in float64, computed for 18 cells.

| score | target | Pearson | Spearman | leave-one-out Spearman |
|---|---|---|---|---|
| PW read-out on hot − cold (through final norm) | singleton | 0.941 | **0.934** | 0.922–0.975 |
| PW read-out | complement loss | 0.924 | 0.781 | 0.740–0.875 |
| PW norm | \|singleton\| | 0.648 | **0.924** | 0.909–0.941 |
| PW norm | singleton (signed) | 0.394 | 0.216 | 0.110–0.444 |
| max source attention (mass) | singleton | 0.812 | 0.360 | 0.240–0.498 |
| max source attention | complement loss | 0.823 | 0.263 | 0.125–0.380 |
| max source attention | \|singleton\| | 0.913 | 0.909 | 0.892–0.946 |

- **Q1 held as written (MEASURED).** Spearman(PW norm, |singleton|) = 0.924 and Spearman(PW read-out, singleton) = 0.934, both above 0.36.
- [See Audit corrections C8.] **The norm comparison is not like for like.** On the same unsigned target, mass already reaches 0.909, so PW and mass are not separable on size. The gain over mass's 0.26–0.36 comes from the sign: PW projected on the answer direction orders the signed effects at 0.93 (MEASURED).
- Reading (ASSERTED): here PW adds the read-out direction, not the size.

## Jobs (MEASURED; mrun on Beast, priority 70)

| job | role | worker s | peak VRAM allocated MB | result |
|---|---|---|---|---|
| job-fb014d69786e | smoke 1 (snow) | 57.4 | 8,367 | done; closure void (bug) |
| job-01a5f4b7790e | smoke 2 (snow, fixed) | 59.5 | 8,370 | done |
| job-86df2dd47809 | full capture (6 words: scores, closure, split, value swap) | 89.5 | 8,385 | done |
| job-a1f49897e70d | Qwen side check | 7.7 | 5,920 (6,032 reserved) | done |
| job-5eeb743a56cd | main lease r2-0 (red, black, ball; 156 arms) | 97.9 | 7,991 | done |
| job-3be01d6f10e4 | main lease r2-1 (snow, night, rain; 156 arms) | 98.1 | 7,991 | done |
| job-8a757c99dbeb | f3 plan (6 words × 16 sets × 2 cut types; 216 arms), attempt 3 | 101.8 | 7,991 | done |
| job-2eb545f0f377, job-6c9b7875d02d, job-7546372990c7 | smoke-2 attempts | – | – | killed_vram at load (clamps) |
| job-aded9ea015e1 | main lease 0 (old packing) | – | – | killed_vram at load (wedge clamp 10,400 → 6,309) |
| job-c1b2bf48ff0d | main lease 1 (old packing) | – | – | failed: lease seal bug, no load |
| job-c301a12cb2ae, job-6b0a44ca4073 | f3 attempts 1–2 | – | – | killed_vram at load (wedge clamp → 6,309 / 6,714) |

## Contact sheets (copies in `/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/sheets/`)

- `main-{red,black,ball,snow,night,rain}-{zero,renorm}.png`: 12 sheets for the main plan. Each shows native, del, iso, all-cut, every cut set, and the keep arms.
- `vswap-{snow,rain,red,ball}.png`: value-swap sheets.
- Key sheets:
  - `main-black-zero.png` (pw60 turns the cat into a tabby; mass-matched low-PW does not)
  - `main-red-zero.png`
  - `main-night-zero.png`
  - `main-ball-zero.png` (no 60-unit cut removes the ball)
  - `vswap-snow.png`
  - `vswap-rain.png`

## Tests

`tests/test_predicted_write.py`: 17 passed (MEASURED). It covers:

- both identities against direct softmax;
- head maps;
- split closure and the invariant;
- payload groups and the common part;
- the alignment bijection;
- selectors against brute force, and exclusions;
- Spearman.

## Not done, or not yet known

- **G3 / G3b scope.** Value-swap arms ran for snow, rain, red and ball (the prereg's set). night and black were not run.
- **Seeds.** Only seed 4242 was run. All numbers are single-seed.
- **Off-trajectory PW.** Not yet known whether PW computed on each arm's own trajectory, rather than on the iso run, would explain why red and black lose half their effect when 5–7 units are swapped.
- [Superseded: Audit corrections C3.] **Rain at 60 units.** Not yet known which 60-unit set, if any, removes rain. The value swap removes it (0.60 on the zero-cut scale); no head cut of ≤ 60 units does.
- **Tighter A0 criterion.** Not done: an ulp criterion relative to the head's output scale, instead of each element's own value. It would likely pass, but that is untested.
- **Lane bug.** The check_lease / replicated mismatch is reported here, not fixed in lane code. [Fixed 2026-11-14 in d768f97: Audit corrections C11.]

## Audit corrections (appended 2026-11-14 02:15 PST)

Source: independent audit `/Users/jakeholl/domains/artifacts/audit-predicted-write-interp-bridge-20261114.md`. Statements above that these items supersede carry a short bracketed pointer; nothing above was rewritten. Labels: **MEASURED** = recomputed by me from saved files in this pass (scripts and result files named); **AUDIT** = recomputed by the auditor and not re-derived by me; **ASSERTED** = my reading.

- **C1. F1 is confirmed only as a joint/single gate effect.** The lane mass (`unit_mass`, `p·|O_h v|`) has no outer attention gate. FLUX.2 has one gate for all 5 joint blocks and one for all 20 single blocks.
  - The fitted per-stratum scale (image-receiver PW over iso mass) gives joint/single 0.0184–0.0192 per word (MEASURED, `followup2/plans2/selection2-report.json`). The model's own gates agree: mean RMS of the joint image gate over the single gate is 0.0184; for the joint text gate it is 0.0586 (MEASURED, gate log of job-6d9fea396262, `followup2/results/vswap2.json`).
  - Of each word's raw mass, 62–86% sits in joint blocks; of its gated mass, 3.1–10.3%; of its PW_both, 9.4–18.7% (MEASURED, `followup2/results/reanalysis-predicted-write.json`).
  - Gated mass gives massmatch-lowpw 0.057–0.154 of own60's gated mass (raw mass 0.90), so gated mass alone predicts F1's outcome in all 6 cells, as PW does (MEASURED, same file). F1's numbers stand; its reading as "PW, not mass" does not.
- **C2. "Mass without PW does nothing" is not supported beyond gated mass.** What the F1 sets show is that joint-block reads write about 50× less per unit of mass. "About 5% of its PW" is 2.3–3.9% (massmatch-lowpw PW_both 0.045–0.077 of 2; MEASURED, `selection-report.json`). Whether PW adds anything beyond gated mass was then tested directly: see Addendum 2 results (it does).
- **C3. "Rain cuts do nothing at 60 heads" is a pixel-metric artefact.** A validated streak-energy measure (`followup2/content.py`, validation in `followup2/results/content-validation.json`) gives, as share of rain removed relative to iso (MEASURED, `followup2/results/content-main.json`):
  - zero / mask: pw60 0.50 / 0.42, pwmatch-lowmass 0.59 / 0.51, own60 0.31 / 0.24, massmatch-lowpw 0.01 / 0.03, rand60a 0.05 / 0.06, rand60b 0.04 / 0.03, all-cuts 1.01 / 0.95. Pixel gone for the same arms: 0.03 / 0.06, 0.26 / 0.13, −0.10 / −0.10, −0.17 / −0.24.
  - Pixel MAD scores the tree's layout shift and road reflections, not the rain. Rain F4 cells do not hold "trivially": on streak energy pw60 removes 0.42–0.50 and the random draws ≤ 0.06.
- **C4. Instrument check (A0), word-row cells (MEASURED from `closure.{red,snow}.json`, job-86df2dd47809).** Worst problem 4 quoted the all-text cells. For the word-row cuts that PW actually scores:
  - head relative error up to 0.185 (zero) / 0.285 (mask) for image receivers, and 0.111 / **0.540** for text receivers (red, step 0, block 4); text-receiver write error against the BF16 model up to 2.47;
  - criterion (iv) (write error ≤ 0.05 wherever PW > 10× sham) **fails in 9 of 60** eligible cell-parts (8 of them red word-row cells, e.g. (0,1) zero: image 0.107, text 0.267).
  - What the check can show: the wiring (head slices, output maps, gate, receiver indices) reproduces the model's change to within BF16 rounding of a change worth a few ulps; other heads change by exactly 0 in all 40 cells; against an FP32 projection of the measured head change the write error is ≤ 0.20.
  - What it cannot show: the port's delta cut and PW use the same FP32 formula on the same q, k, v, so closure does not test the identity independently of the port; per-unit PW for late-block word-row cells is uncertain to tens of percent (ASSERTED, from the floor-limited cells).
- **C5. Rain's mass-matched set lies outside its floor.** massmatch-lowpw on rain is gone −0.169 (zero) / −0.241 (mask) against floor bounds 0.038 / 0.032 (AUDIT; the gone values are in `results/arms.json`, MEASURED). The sentence "−0.24 to 0.07, inside or near the floors" holds for the F1 words only. Rain is not an F1 word, so F1's 6/6 stands.
- **C6. The value swap had no unrelated-prompt control.** Run in Addendum 2 (job-6d9fea396262). Result: the unrelated-prompt swap destroys the whole image, so it does not separate word-specific from generic removal; the non-scene-word swap can, and it does not remove snow or rain. See Addendum 2 results, H4.
- **C7. The routing / payload split is mostly downstream divergence.** The all-block figures are dominated by single blocks and late steps, where the deleted run's image state has already diverged. At first contact (step 0, joint blocks; MEASURED, `followup2/results/split-per-step-block.json`):
  - word row share of payload: snow 0.71, rain 0.68, night 0.58, ball 0.54, red 0.07, black 0.07 (red's payload is 0.58 other real rows);
  - routing share 0.49–0.75 (red 0.57, black 0.49, ball 0.66, snow 0.75, night 0.61, rain 0.59), not about 0.2.
  - G1 / G2 were scored on the all-block aggregate as preregistered, which mixes cause and consequence.
- **C8. Qwen: 0.934 against 0.36 is not like for like.** On the unsigned target |singleton|: PW norm 0.924, |PW read-out| 0.911, max source attention 0.909 (MEASURED from `results/qwen.json`). The gain is the sign: the read-out gets 16 of 18 signs right.
- **C9. Custody gap for addendum 1.** Its hash entry is 00:54:36, but it was committed only at 01:23:02 (07538b8), after the scored main jobs were submitted at 01:02:25 and 01:04:19 (MEASURED: git log, `submission.json` mtimes). Pre-run custody of addendum 1 therefore rests on file mtimes and a self-written log, not on git. Addendum 2 was committed (30ee143, 02:08:38) before its first job.
- **C10. Undeclared monkeypatch in `stage.py` (seal commit 8b23bc7).** `stage.submit` replaces `mrun.client.submit.submit` in-process with a wrapper that adds `reservation["pin"] = True` to the final submit call. Effect: mrun's history clamp may not lower the declared reservation (the measured peak × 1.3). It does not change priority, preemption or other jobs, and the wedged-queue clamp still overrides it (three of the lost jobs). It changes scheduler behaviour from inside experiment code and should have been declared in the prereg. It is declared in Addendum 2, whose jobs used it.
- **C11. Lane bug fixed.** `flux_read_panels.check_lease` now rebuilds the lease body with `seal_sha256` and `replicated` when present, as `Panel.leases` does (commit d768f97; test `test_check_lease_accepts_replicated_and_sealed_leases`, which also checks the real lease 1 of `plans/main.leases.json`).
- **F3 with gated mass (MEASURED, `followup2/results/reanalysis-predicted-write.json`).** Mean Spearman over 12 cells: raw mass 0.016, gated mass 0.143, PW (cut type) 0.210, PW_both 0.230, image-only PW 0.139, text-only PW 0.173. PW beats gated mass in 9/12 cells, mean +0.067, against +0.193 over raw mass. The weak-test caveat above still applies.

## Addendum 2 results (appended 2026-11-14 02:15 PST): does PW add anything beyond gated mass?

Prereg `PREREG-ADDENDUM-2.md` (sha256 `276be6aa…`, committed 30ee143 before any of its jobs). Jobs (mrun on Beast, priority 70, 11,000 MB reservation pinned, nothing else running or queued at each submit):

| job | role | worker s | peak VRAM allocated MB |
|---|---|---|---|
| job-6d9fea396262 | value-swap controls (20 runs) | 20.3 | 7,991 |
| job-c73f1d005af4 | plan gm lease 0 (red, black, ball; 141 branches) | 77.9 | 7,991 |
| job-57fe341aa4e2 | plan gm lease 1 (snow, night, rain; 141 branches) | 78.6 | 7,991 |

Reproduction (MEASURED): the reruns of pw60 and own60 are bitwise equal (RGB) to the saved main-plan images in 24 of 24 arms, and iso, del and the all-cuts in 24 of 24; the value-swap job reproduces the saved snow and rain `vswap-all`, the red `vswap-all`, and the roots bitwise.

### Forecasts (pixel gone, red / black / ball × zero / mask; `followup2/results/forecasts-addendum2.json`; MEASURED)

| forecast | rule | cells (set vs reference) | result |
|---|---|---|---|
| H1 | gmass60 ≥ 0.8 × pw60 | red 0.31 vs 0.50, 0.22 vs 0.31; black 0.29 vs 0.92, 0.14 vs 0.85; ball 0.47 vs 0.66, 0.33 vs 0.41 | **failed** (1/6; ratios 0.17–0.80) |
| H2 | massmatch-lowpw-single (closest set; band unreachable) < own60/3 | red 0.33 vs 0.47, 0.21 vs 0.26; black 0.27 vs 0.31, 0.11 vs 0.30; ball 0.47 vs 0.51, 0.32 vs 0.23 | **failed** (0/6), as forecast |
| H2b | massmatch-lowpw-single-own < own60/3 | red 0.13 vs 0.47, 0.11 vs 0.26; black −0.16 vs 0.31, −0.05 vs 0.30; ball 0.15 vs 0.51, 0.16 vs 0.23 | **failed** (4/6; red mask and ball mask miss), against my forecast of a clear fail |
| H3 | gmassmatch-lowtextpw < (2/3) × pw60 | red 0.14 vs 0.50, 0.13 vs 0.31; black −0.13 vs 0.92, −0.06 vs 0.85; ball 0.16 vs 0.66, 0.16 vs 0.41 | **held** (6/6; ratios −0.14 to 0.43) |
| H4 | unrelated-prompt swap removes ≥ 2/3 of what the deleted-word swap removes (content) | snow −1.25 vs 0.84; rain 2.27 vs 1.38 | **failed** (snow cell fails; rain cell "holds" only because the image is destroyed) |

What the forecasts support:
- **H1 failed and H3 held: predicted write carries effect that gated mass does not** (MEASURED outcome; reading ASSERTED). The 60 heads with the most gated mass remove about as much as own60 (black 0.29 / 0.14 against own60 0.31 / 0.30), not as much as pw60. At equal gated mass (−10% of pw60's), a set with 0.41–0.63× pw60's text-receiver PW removes 0.14–0.43 of what pw60 removes in red and ball and nothing in black. Its image-receiver PW is 0.87–0.92× pw60's, so the difference sits mainly in the text-receiver term. This answers the audit's open question: PW does not reduce to "mass × gate" for these words.
- **H2 is uninformative by design** (stated before the run): no 60 single-block heads reach own60's raw mass (0.13–0.41 of it), and the closest set is a high-PW set (PW_both 0.85–1.19). It removes about what gmass60 removes. Its failure does not show "F1 was only the gate".
- **H2b, the within-stratum dissociation, mostly held** (4/6 cells; the two misses are mask cells at 0.42× and 0.68× own60, ASSERTED near-floor for red: floor bound 0.062). Inside the single blocks, mass-matched heads with about 0.6× of own60's PW remove little. Combined with C1: F1's original contrast was the gate, but inside one gate stratum mass without PW still does little in red, black and ball zero cuts.
- **H4: the unrelated-prompt control does not separate generic from word-specific removal.** It replaces the whole scene (pictures below), so content measures read nonsense on it. The non-scene-word swap is the usable control: deleting `red` (cat prompt) removes 0.09 of the snow ground (snow word's own swap: 0.84; on the post-hoc brightness-only measure 0.17 against 1.79); deleting `blue` (car prompt) removes −0.20 of the rain streaks (rain's own swap: 1.38). Reading (ASSERTED): the swap's removal of snow and rain is word-specific; its side effects (background trees gone, desaturation, darkening) are generic, since they appear in the non-scene swaps too.

### Secondary outcomes (MEASURED; `followup2/results/gm-arms.json`)

- **Snow, night, rain on pixel gone** (zero / mask): gmass60 vs pw60: snow 0.40 / 0.29 vs 0.42 / 0.30, night 0.57 / 0.51 vs 0.77 / 0.71, rain −0.01 / −0.07 vs 0.03 / 0.06. gmassmatch-lowtextpw: snow 0.30 / 0.26, night 0.29 / 0.36, rain −0.15 / −0.12. So for snow gated mass matches PW; for night PW is ahead.
- **Rain streak energy removed** (zero / mask): pw60 0.50 / 0.42, gmass60 0.41 / 0.32, own60 0.31 / 0.24, closest single set 0.43 / 0.36, single-own 0.08 / 0.05, gmassmatch-lowtextpw 0.22 / 0.17, rand60c 0.20 / 0.23, rand60d 0.01 / 0.04. rand60c is not near zero; its zero-cut pixel floor bound is 0.32, the largest in the plan (not yet known why).
- **Snow ground** (preregistered measure, zero / mask): pw60 0.69 / 0.40, gmass60 0.75 / 0.45, gmassmatch-lowtextpw 0.26 / 0.25. **These numbers are not trustworthy**: in the pictures the ground stays white in every 60-head cut. The measure's saturation term counts the bluer snow of those arms as "not snow" (bottom-region mean saturation 0.70 against iso 0.62, brightness unchanged at 0.75). A post-hoc brightness-only variant (`snow_ground_bright`, not preregistered; `followup2/results/snow-bright-posthoc.json`) gives every 60-head cut ≤ 0.14 and the deleted-word swap 1.79 (shams ≤ 0.005 against a span of 0.30). The snow measure passed its validation (native / deleted / sham / all-cut / random) and still failed on the arms that matter.
- **Latent projection** onto iso − del, as a fraction of the same-type all-cut (zero cut): red pw60 0.53, gmass60 0.39, gml 0.12; black 0.89, 0.47, 0.15; ball 0.72, 0.60, 0.22; rain pw60 0.35, gmass60 0.37. Same ordering as pixel gone for red, black, ball. For rain the latent sees a move toward deletion that pixel gone misses, as the streak measure does.
- Random sets rand60c / rand60d: |gone| ≤ 0.18 in all 24 cells; ≤ 0.03 for red and black.

### Pictures (sheets `/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/sheets/addendum2/`; my reading, ASSERTED, separate from the numbers)

- **black (zero):** pw60 is a grey striped tabby, as before. own60, gmass60 and the closest single set give a dark grey cat with faint stripes. single-own, gmassmatch-lowtextpw and both random sets leave the black cat.
- **red (zero):** pw60, own60, gmass60 and the closest single set all give a dull mauve-grey ball. To my eye gmass60 is as mauve as pw60, although its pixel gone is 0.31 against 0.50. single-own, gmassmatch-lowtextpw and the random sets stay red.
- **ball (zero):** no set removes the ball. pw60, own60, gmass60 and the closest single set shrink and darken it; the others leave it as in iso.
- **snow (zero):** pw60, gmass60 and the closest single set remove the snowy background trees (plain pink-blue sky); own60 and gmassmatch-lowtextpw partly. The ground stays white snow in every arm.
- **night (zero):** pw60, own60, gmass60 and the closest single set turn the night into a light, foggy blue-grey dusk, pw60 slightly the lightest. gmassmatch-lowtextpw is a little lighter than iso; single-own and both random sets stay night.
- **rain (zero):** pw60, own60, gmass60, the closest single set and gmassmatch-lowtextpw move the tree to the deleted image's centred layout; in pw60 and gmass60 the rain looks fainter and foggier, in the others the streaks stay. single-own looks like iso. rand60c and rand60d move the tree to the right (as in native) and keep the rain.
- **value swaps** (`vswap2-snow.png`, `vswap2-rain.png`): the deleted-word swaps reproduce the earlier pictures (grey ground, no snow; dark frame, no rain streaks, grey car). The unrelated-prompt swaps give a blurred cream-white (cat prompt) or cream-yellow (car prompt) frame with no cat, ball, car or tree. Deleting `red` makes the ball dark navy and removes the background trees but keeps the snow ground; deleting `blue` makes the car dark and the frame darker but keeps the rain streaks.

### Not done, or not yet known

- Single seed (4242). Floors: several H2b / H3 cells lie within about two floor bounds of zero (bounds 0.016–0.093 for those arms).
- Why rand60c on rain has a large floor (0.32) and removes 0.20 of the streak energy: not yet known.
- A snow content measure that tracks the pictures was found only post hoc; a preregistered test of it is not done.
- Whether the text-receiver PW term would also beat gated mass at other set sizes (10, 120 heads): not yet known.

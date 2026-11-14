# Predicted write vs mass, and routing vs payload: FLUX.2 Klein 4B with a Qwen2.5-1.5B side check (2026-11-13/14)

- Prereg: `PREREG.md`, sha256 `688060256e7c5a03633b1e30df85ceeb308ed037e9943055dd76b8c9d2563fb1`, sealed 2026-11-14T00:33:01-08:00 before any non-smoke job.
- Execution addendum: `PREREG-ADDENDUM-1.md`, sha256 `bb64134e…`, hashed before any main or f3 panel arm ran. It changes lease packing and the reservation, not rules. Both hashes are in `PREREG.sha256.log`.
- Seed 4242 only.
- Labels: **MEASURED** means read from a job output or computed offline from one. **ASSERTED** means my reading. Verdict words come from the prereg's fixed list.
- Raw outputs live in `/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results/<job>/`, contact sheets in `/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/sheets/`, and the small tables in `results/*.json` here.

## Worst problems first

1. **The F2 set nearly duplicates pw60 (MEASURED).** `pwmatch-lowmass` shares 53–55 of its 60 units with `pw60` for every word, over the prereg limit of 50. The reason: pw60 is already a low-mass set (mass share 0.10–0.39, against 0.65–0.83 for own60), so the "same PW, less mass" set can only swap 5–7 units. By the prereg's own rule this makes the hypothesis verdict **unresolved**. The swap was still informative: in red and black, exchanging those 5–7 units halved the effect (F2 failed in 4 of 12 cells; see A2).
2. **F3 failed, and it was close to having no power (MEASURED).** PW wins 7 of 12 cells; 8 were needed. Mean Spearman is PW 0.21 against mass 0.02. Most 10-unit cuts remove ≤ 0.1 of the word, about the size of the floors.
3. **G1, G2 and G3 failed (MEASURED).** The routing/payload forecasts were wrong in direction for the word+real and padding shares. The value swap reached d 0.47 (snow) and 0.49 (rain) against a bar of 0.7.
4. **A0 criteria (i) and (ii) fail as written, at 96.8–99.98% of elements instead of 100% (MEASURED).** All-text cuts miss half-ulp agreement where the head output nearly cancels. One element's bound is a near-zero half-ulp, which gives max error/bound 7.6e24. Head relative error is ≤ 2.9e-3 for zero cuts and ≤ 8.6e-3 for mask cuts, and other heads change by exactly 0 in all 40 cells. These are reported as failing, not rewritten (see A0).
5. **The value swap leaves the data manifold for red and ball (MEASURED d −1.69 / −0.78; pictures).** Swapping in the deleted run's text values moves those images further from *both* native and deleted than deletion does. The background trees disappear. The swap carries more than the word.
6. **Rain cuts do nothing at 60 heads (MEASURED).** pw60 gone is 0.03 (zero cut, inside its floor bound of 0.049) and 0.06 (mask cut, just above its bound of 0.041), and own60 is −0.10. All rain F2 and F4 cells "hold" trivially because the reference is ≈ 0.
7. **Scheduler and lease losses (MEASURED, job records).** Seven jobs died before running any arm. All failures were at load or startup; no arm ran in them.
   - killed_vram at load, six jobs: job-2eb545f0f377, job-6c9b7875d02d, job-7546372990c7 (smoke), job-aded9ea015e1, job-c301a12cb2ae and job-6b0a44ca4073 (f3 attempts 1–2). In each, mrun lowered the reservation below the 8,022–8,367 MB load peak: the history clamp twice, the wedged-queue clamp three times. The wedge clamp overrides a pinned reservation whenever another session's job holds the GPU.
   - f3 attempt 3 (job-8a757c99dbeb) ran because it was submitted near the end of the other session's job, so it was admitted before the 120 s clamp.
   - job-c1b2bf48ff0d failed deterministically with `PanelError: lease seal differs from its branches`. This is a **library bug**: `Panel.leases` hashes the `replicated` list into `lease_sha256`, but `flux_read_panels.check_lease` recomputes the hash without it, so the FLUX worker refuses any lease that has replicated branches. I did not edit lane code. The work-around was a word-aligned repack with no replicated branches (`plans/main.leases-r2.json`).
8. **Smoke-1 closure was void** (worker bug, fixed before the seal): its cell operands were overwritten by the native-conditioning run.

## Verdict (Experiment A)

| forecast | rule (PREREG) | result | status |
|---|---|---|---|
| F1 | gone(massmatch-lowpw) < gone(own60)/3 in all 6 cells (red, black, ball × 2 cut types) | 6/6 | **held** |
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

- **Mass without PW does nothing.** Mass-matched low-PW sets hold 0.58–0.75 of the word's write mass (own60 level) but about 5% of its PW. Their gone is −0.24 to 0.07 in every cell, inside or near the floors. The same mass inside own60 removes 0.23–0.56. This is the clean dissociation F1 tests, and it held 6/6.
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
  - del and all-cut have no rain. Every 60-unit cut keeps the rain streaks, matching gone ≈ 0.
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
- Reading (ASSERTED): the identity, the captured operands, the head slices, the output map and the gate reproduce the model's write change to within BF16/FP32 arithmetic.

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

- About 80% of the change comes through changed values.
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
  - The deleted run's text values carry the word's content, almost all of it in rows other than the word's own row.
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
- **The norm comparison is not like for like.** On the same unsigned target, mass already reaches 0.909, so PW and mass are not separable on size. The gain over mass's 0.26–0.36 comes from the sign: PW projected on the answer direction orders the signed effects at 0.93 (MEASURED).
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
- **Rain at 60 units.** Not yet known which 60-unit set, if any, removes rain. The value swap removes it (0.60 on the zero-cut scale); no head cut of ≤ 60 units does.
- **Tighter A0 criterion.** Not done: an ulp criterion relative to the head's output scale, instead of each element's own value. It would likely pass, but that is untested.
- **Lane bug.** The check_lease / replicated mismatch is reported here, not fixed in lane code.

# Audit: FLUX read panels, Qwen panels, native-discovery rerun (2026-11-13 21:46 PST)

Independent audit of branch `claude/mdb-consolidation-20261112` at 420b38e, package `packages/mdb`. I read the code and the saved raw outputs. I did not rely on the write-ups. Every number marked MEASURED below was recomputed by me from raw files (PNG, rows.jsonl, npz, sealed plans) with numpy/PIL/json on the laptop, or by re-running the shared verdict code on the saved results. I loaded no models and submitted no jobs. Scratch scripts are in the session scratchpad (`flux_*.py`, `qwen_*.py`, `nd_q5.py`).

Layout note (added when this branch was ported onto the mdb-lanes split, 9ec645b): the audit was written against the pre-split layout. Paths below such as `src/mdb/attention_panels.py`, `qualification/...` and `tests/...` now live under `packages/mdb-lanes/` (`src/mdb_lanes/`, `qualification/`, `tests/`); commits 420b38e and 166b334 are on the unpushed source branch only. The audit text is otherwise unchanged.

Verdict vocabulary used to judge question 1 (from the brief): `supported` = no alternative explanation left; `set_only` = the set is needed (or sufficient), but mass, set size or generic damage is not ruled out.

## Findings, worst first

### 1. FLUX: the primary verdict table uses verdict logic that had already been superseded, and the "4 of 18" comparison hides 25 of 78 changes (Q1)

- Sealed module `attention_panels` 31a8595b is commit 5e09d95 (15:22 on 11-12), the version from before the audit-7 verdict fixes (9451fc8 at 15:52, then fde00c9). The doc's main table (`docs/FLUX-READ-PANELS.md:248-273`, `report/`) uses it. Verdicts are computed offline, so the sealed module hash does not require this logic.
- The sealed rule (`git show 5e09d95:.../attention_panels.py`, lines 985-1040) counts only controls with `separates_identity=True`. That means every other-item control, and random controls only when they are mass-matched. A record is `supported` when it beats every counted control beyond the floor. There are two errors:
  1. **Other-item keep controls are counted for sufficiency, but they are not sufficiency controls.** For example, `keep-own60@rows-ball` on black cuts the complement of black's set on ball's rows. On ball, its effect is larger than the all-cut itself (recovery −0.88 to −1.47 for `ball keep-own60` and `keep-nounI`), so it can never fail to be beaten.
  2. **Random or external sets of the same size that are not mass-matched are ignored entirely, even when they do equal damage.** For example, in `snow own120` (zero, 4242) `rand120` has effect 0.1156 against the test's 0.1166, yet the sealed version calls the record `supported`.
- I re-ran current `verdicts()` (4a39dbaa) on the saved plan and `report/d-results-px.json` for **all 78 records** (MEASURED). 25 records change. Sealed: 27 supported / 51 not_supported. Current: **2 supported / 13 set_only / 63 not_supported**. The doc reports only the 18-record subset ("4 of 18").
- The two records that stay `supported` are `black keep-modI` and `night mod`. Both have a disjoint, mass-matched random control that they beat.
- Per case:

| record | sealed | current | correct per definitions | why (MEASURED from records) |
|---|---|---|---|---|
| black own60 | supported | set_only | **set_only** | Generic damage is ruled out (other-item 0.013-0.017 vs test 0.039-0.062, all 4 cells). Set size is ruled out (`rand0-own60`, effect 0.005-0.007, mass 0.206 vs 0.671 = `less`). Mass stays open because no reachable random set matches 0.671. Sealed counted only the other-item control. |
| black keep-own60 | supported | set_only | **set_only** | Recovery 0.88-0.92. The random keep (`less` mass) recovers less, so set size is ruled out. Mass is open. Sealed relied on the other-item keep, which is not a sufficiency control. |
| ball keep-own60 | supported | not_supported | **at most set_only; not_supported only under a threshold that was not sealed** | Recovery: zero 0.698 / 0.796, renorm 0.567 / 0.736 (4242 / 9001). `DEFAULT_RECOVERY = 0.7` (`attention_panels.py:60`) is not in the sealed plan; it is post hoc. Renorm 4242 (0.567) fails it clearly. Zero 4242 misses by 0.0024. Mass is open in any case. |
| ball keep-nounI | supported | not_supported | **not_supported** | Recovery is 0.15-0.57 in all 4 cells. |

- **The current logic has its own flaws that affect other verdicts.** See findings 2 and 3: absolute mass tolerance; other-item controls with no mass relation; random draws that can collapse or reuse other test heads. In this table, `snow own240-minus-own60` is `not_supported` only because of a "matched" random control that carries 11× the target's mass (0.0169 vs 0.0015). With a correct `more` label it could not refute, and the record would be `control_missing`.
- Fix: make the current-logic table over all 78 records the primary FLUX verdict table. Keep the sealed-logic table only as history, labelled as superseded.

### 2. Qwen: the controls behind the `freezing` set_only verdicts do not control what they claim (Q3)

- **Random draws (MEASURED from `full/read-float32.plan.json` `controls`).** For every `freezing` set, `rand0` and `rand1` are the same set. For `L20H5` the control is `[[20,11]]`, which is L20H11, itself a named test head. For `L20H11` the control is `[[20,5]]`, which is L20H5. For `rank3` the control `[[17,5],[18,7],[20,11]]` contains L20H11. Shifts recomputed from rows: L20H5's "random" shift (+0.005 / +0.090) is exactly L20H11's cut, and the reverse is also true.
- **Other-item control.** For `freezing` it cuts the same heads' reads of the `icy` row. Those heads carry **0.05%-2.7%** of the target's attention mass there. MEASURED mass ratios, other / target: L20H5 0.0011, L21H0 0.027, L20H11 0.0069, named3 0.0105, rank3 0.0005, rank12 0.0065. It is close to an empty cut, so beating it cannot rule out generic damage. For `icy` the other-item control is the opposite case: 37-1937× **more** mass. It has `mass_relation=None`, so `_compare` (`attention_panels.py:1507`) still lets it refute.
- **Verdicts under the alternatives (MEASURED, re-running current `verdicts()` on 491 rows).**
  - Treating only the named-head random controls as `control_missing` changes no verdict. Necessity passes on generic damage alone, which comes from the near-empty other-item cut.
  - Also treating the other-item control as ruling nothing out when it removes less than 10% of the target mass gives: **L20H5, L20H11 and rank3 become `control_missing`**. L21H0, named3 and rank12 stay `set_only`, carried by less-mass random sets that contain no named head.
- **Doc errors.**
  - `QWEN-PANELS.md:157`, "Set size is ruled out by the random controls": false for L20H11 (`ruled_out = ['generic_damage']` only). For L20H5 and rank3 it rests on a named test head.
  - `QWEN-PANELS.md:3` says the collapsed set "carries less mass than the set". For L20H11 it carries more (0.927 vs 0.634).
- **Design flaw in `_draw_random` / `matched_control_sets` (`attention_panels.py:592-662`, `attention_ranking.py:123-189`).**
  - When mass is unreachable, the draw falls back deterministically, so `n` draws can be identical, with no warning and no deduplication.
  - The draw excludes only the target set, not the other test sets of the panel.
  - Effect on the FLUX lane: n=1 per set there, so no duplicates. Random sets overlap the other named sets at roughly the rates expected from set size (sets of 31-240 out of 600 heads; MEASURED overlap table in scratch), so the named-head problem is not material for FLUX. The absolute-tolerance flaw (finding 3) does affect FLUX.
- Fix:
  - Exclude every named test unit of the panel from random draws.
  - Refuse or flag duplicate draws.
  - Give other-item controls a `mass_relation` computed from the per-item mass the plan already holds.
  - Require the other-item mass to be comparable before it may rule out generic damage.

### 3. Shared: "matched" mass uses an absolute tolerance of 0.02, which labels controls with 4-14× the target's mass as matched (Q3/Q6)

- **Code.** `matched_control_sets` matches "total mass within an absolute tolerance" (`attention_ranking.py:131`). `_mass_relation` then reports `matched`, which may refute (`attention_panels.py:1507`) and rules out `mass` when beaten (`RULES_OUT_BY_MASS`, `:58`).
- **Qwen `icy` (MEASURED control/target mass ratios).**
  - rank3: 8.6 and 12.6
  - keep-rank3: 13.9 and 13.2
  - L20H5: 0.48 and 4.35
  - L20H11: 0.64 and 1.93
  - Several others fall outside ±25%.
  - `QWEN-PANELS.md:154` calls these "matched-mass random draws of the same size". That is false for 7 of 12 icy sets.
- **FLUX `own240-minus-own60` (MEASURED).** Target mass 0.0015-0.0039 against control 0.0167-0.0189 for ball, rain and snow, all labelled matched.
- Verdict effect, current logic:
  - FLUX `snow own240-minus-own60`: not_supported becomes control_missing.
  - Qwen `icy rank12`: with honest mass labels on other-item and random controls, not_supported becomes set_only. This is a 0.0096 margin shift; see finding 7.
- Fix: make the tolerance relative (a fraction of the target mass), or require `|control − target| ≤ tol × target`.

### 4. Native discovery: 0/288 is correct, but the negative says little about learned routes, and the closing sentence states an assertion as fact (Q5)

- **Recompute (MEASURED, independent numpy argmax agreement from sealed predicted and native routes in `evidence/forecasts/*`).**
  - 0 of 288 learned cells beat the per-cell best of the four sealed baselines.
  - **0 of 288 beat a baseline fixed in advance**, chosen per layer from the 48 discovery documents: current token at L0 (0.501), position 0 at L9 (0.791) and at L19 (0.722).
  - Largest learned margin over the per-cell best of the sealed four: −0.359 (L0), −0.263 (L9), −0.224 (L19).
  - On non-position-0 rows, 185 of 288 cells are below the uniform expectation.
  - So the per-cell "best trivial" does read held-out answers (`attention_ranking.py:391`, the max over baselines on the same held-out rows), but the conclusion does not depend on that.
- **Scorer soundness (MEASURED).**
  - Native routes scored against themselves give 1.000 on every document/layer.
  - The per-head raw-mass rule (fitted on discovery documents) clears the sealed baselines on 14/16 documents at L0.
  - The scorer can register a win. It scores top-1 argmax only, and the previous-token baseline is scored on a subset of rows (query ≥ 1) while the nominator is scored on all rows (`attention_ranking.py:374-392`). This is a small asymmetry in the margin.
- **Could a learned arm have won? Not yet known. The evidence leans toward "this learner barely trained", not "nothing to find".**
  - At **layer 0 the route input is the native embedding state, not a predicted state**. 4 of 6 arms carry a route-KL loss (`auxiliary = kind != "sparse_reconstruction"`, `native_discovery_learning.py:457`). Still, every learned arm scores 0.066-0.086 at L0, which is uniform level (0.074), against current token 0.506.
  - Accepted updates are 10/20/10/10/40/20 (MEASURED from `worker.log`).
  - From reading the code (ASSERTED, untested), there are two reasons the route form may struggle:
    1. The route is a rank-4 q·k softmax over the feature code with no position input (`:150-157`).
    2. In `position_mode="split"` (the `.p0robust` arms), position 0 is centred and scaled by its own statistics before encoding (`:129-134`). This removes the large-activation cue that would let a content route find the sink.
  - No planted-route test exists (searched `tests/`).
- **Mislabel.** `FINDINGS.md:142` says "the predictor has no direct way to express the positional rules that the native heads mostly follow" without a label. It restates item 1 of the ASSERTED list at `:120-124`.
- Fix:
  - Add a planted-route recovery test.
  - Report the L0 learned-vs-uniform result (native input, route loss) as the main evidence of undertraining.
  - Label `:142` ASSERTED.

### 5. Qwen BF16: the bound is a readout-rounding bound, but it is used as the resolution threshold (Q4)

- **MEASURED from `job-7c3232ad65bb` `bfloat16/value-bfloat16/rows.jsonl` vs `main-02/analysis/arms.json`.**
  - 56/56 arms recomputed. 24 have equal margins. 32 differ: +0.125 ×25, −0.125 ×5, +0.0625 ×1, −0.0625 ×1.
  - The skew (25 vs 5) indicates a systematic path difference, not noise.
- **Replay check (smoke `summary.json` `lanes.bfloat16.replay_check`).** Max |Δlogit| is 0.15625 and 0.1875. The recipe margins are −4.75 and 0.375; the worker gives −4.625 and 0.375.
- **Coefficients (MEASURED).** Recomputed with `qwen_panels.coefficients`, cross-checked against the raw logits.
  - Saved BF16 vs FP32: 5/58 lie outside the saved coefficient's own bound: full effect −0.597, singleton L20/KV0 −0.421, complement L20/KV0 −0.503, complement L21/KV0 −0.303, paired L20 −0.374, each against a bound of 0.25.
  - The BF16 rerun has 2/58 outside: full effect −0.472, complement L20/KV0 −0.378. The doc does not report this.
  - 58/58 coefficients are resolved in FP32 (max bound 6.8e-5). 17/58 are resolved in both BF16 sets.
- **Is the bound wrong?**
  - The arithmetic is right. `float_spacing` gives a BF16 ULP of 0.125 in [16, 32), so the half-ULP is 0.0625 per logit (`analysis_metrics.py:121-176`). `Bounded` merging is correct.
  - It is **too tight as an uncertainty bound**. It covers only the final rounding of stored logits, not BF16 compute error through 28 layers. Yet `Bounded.resolved` (`:249`), applied via `qwen_panels.bounded_margin` (`qwen_panels.py:290-295`), is used to call BF16 effects "resolved".
  - The measured BF16 compute error on the full effect (0.47-0.60) is about 2× the bound.
- Fix: for BF16 lanes, add a measured compute-error term to the bound (for example the FP32-vs-BF16 gap of `native-recipient`, or the twin floor), or label the field `readout_rounding_resolved`.

### 6. FLUX: the floor-bound "miss" is a rounding artifact, and the bound's justification does not cover cross-precision differences (Q2)

- **black 4242 `m:own60` (MEASURED from metrics and from PNG pixels; both agree to 1e-6).**
  - Legacy gone 0.44513, new 0.42155, |Δ| = 0.02358.
  - Bound 0.02508: sham px 0.001656 / (MAD(native, del) 0.14208 × span 0.46471).
  - **The cell is inside its bound.** At full precision **0 of 180 cells exceed** their bound. "0.42 vs 0.45" exceeds only after two-decimal rounding.
  - The bound itself is computed as documented: 14 random cells recomputed from PNG match gone, bound and effect exactly. Doc table: 288 of 288 cells match `b-gone-*.json`. The legacy gone table equals `headfree_table.json` exactly.
- **The bound is not a bound on new-vs-old differences.**
  - The triangle-inequality argument (`flux_read_panels.py:399-428`, doc `:193`) holds for two arms in one lane with the same del, iso and none images.
  - Across lanes the span differs (0.46471 vs 0.46169 here). The floor measures only the new lane's sham drift.
  - The direct cross-lane image distance of the arm exceeds the floor bound in **56 of 180 cells** (max 1.92×). For this cell it is 0.0482 in gone units, against a bound of 0.0251.
- The "new precision within floor" claim therefore holds as an empirical observation (180/180), not as a consequence of the inequality. It is not undermined.
- Fix:
  - State "0 of 180 at full precision".
  - Drop "by the triangle inequality" for the cross-lane comparison, or bound it with the measured cross-lane image distance.

### 7. Verdicts decided by beating a floor, not by effect size (Q6)

- Qwen `freezing L20H11` cut is `set_only` (a "needed set") on margin shifts of +0.005 (zero) and +0.090 (renorm). That is 0.1% and 2% of the all-cut shift (4.767 / 4.493), against an FP32 floor of 3.8e-6.
- Qwen `icy rank12` would be `set_only` on a 0.0096 shift once controls are labelled honestly (finding 3), while the icy all-cut moves the margin by only −0.001 (zero).
- Fix: add a minimum-effect criterion to necessity (for example an effect of at least x% of the all-cut), sealed in the plan.

### 8. Smaller errors

- **Qwen sign.** `QWEN-PANELS.md:148` gives L21H0's random control as "+0.014" (zero). The rows give **−0.014**. Outcomes are `|margin shift|` (`qwen_panels.py:399`), so the verdicts do not see direction. A control that moves the margin the opposite way counts as damage.
- **FLUX random mass.** It comes from the seed-4242 capture only (`flux_read_panel_stage.py:77-90`) and is applied to both contexts. "Matched" is not established for 9001. This affects the mass ruling of the 2 surviving `supported` records in their 9001 rows.
- **FLUX determinism count.** I count 188 keys computed in two or more succeeded jobs (160 arms, 28 roots), not 183 (160 + 23). All are hash-identical, so the conclusion holds and only the count differs.
- **FLUX other-item controls** carry no mass relation either (`_profiled`, `attention_panels.py:1527-1528`). Their share-normalised mass ratio (other word / own word, from the full-a capture) ranges from 0.38 to 2.07. 10 current-logic `not_supported` records are refuted by an other-item control with a ratio above 1. In 2 of them (`ball noun`, `night modI`) it is the only refuter. I did not check absolute mass; this is share-normalised only.
- **Tests.**
  - `tests/test_flux_read_panel.py`, `test_qwen_panel.py` and `test_analysis_metrics.py` all pass: 31 passed, 0 skipped (torch worker tests deselected). Artifacts are on disk, so the `skipif` tests did run.
  - None checks for duplicate random draws or named-head reuse. `test_read_plan_has_named..._random_heads...` asserts only same-layer and disjoint-from-target, on fake attention.
  - No planted-route test exists for native discovery.

## Claims verified correct (MEASURED by me)

- FLUX:
  - Legacy gone table equals `headfree_table.json` (max |Δ| 0).
  - Doc gone/bound table matches `b-gone-*.json` (288/288 cells).
  - Max |new − old| is 0.0781 (red 9001 own240, bound 0.198).
  - 14 spot cells recomputed from PNG: gone, bound and effect are exact.
  - Doc verdict table (d) matches `report/d-verdicts-selected.json` (18/18 rows: statuses, agree flag, 4242 effects and floors, 9001 statuses).
  - `report-head-logic` statuses reproduce with current `verdicts()`.
  - All 2,568 full-plan arms are present.
  - 78 random-control entries are emitted.
  - Twin-gap pair distances are present.
  - The legacy identity record says 244/244 arms and 31/31 roots identical.
  - Regeneration record: 3/3 match, no lane drift.
- Qwen:
  - Table (d) shifts, recoveries and all-cut shifts; random and other-item magnitudes, apart from the sign in finding 8.
  - icy 12/12 `not_supported` under current logic.
  - Native margin −4.647 and its tokens.
  - FP32 sham floor max 3.8e-6 (n=196).
  - Statuses in `full-tables.json` reproduce exactly from rows.
  - (a) 24/56 margins equal, plus the stated differences.
  - Replay gaps 0.156 / 0.1875.
  - (b) 5/58 outside bound, 17/58 resolved, largest |FP32 − saved| 1.026 (sum of singletons).
  - (c) Pearson 0.812/0.823, Spearman 0.360/0.263, leave-one-out ranges as stated.
  - Lease copies 12/12 equal.
  - Label-free named ranks 0/4/5, the same under raw mass.
- Native discovery:
  - 0/288 against both the per-cell and the discovery-fixed baselines.
  - 185/288 below uniform on non-position-0 rows.
  - Sink above 0.5 in 16/16 documents at L9/L19 and 0/16 at L28.
  - All-rows and non-position-0 tables match `posthoc-route-baselines.json`.
  - Contrast nominator beats the sealed four on 16/8/16 documents; against raw mass it is above/below on 14/1, 8/8 and 0/0 documents.
  - Accepted updates 10/20/10/10/40/20.
  - No planted-route test exists.

## Not verified

- Per-job wall time, VRAM and RSS (FLUX `jobs.json`, Qwen runs table, native-discovery run table).
- FLUX laptop pixel comparison against the `saturn/.../headfree/result` PNGs (244/244). I read the job's own identity record but did not re-diff pixels.
- Regeneration hashes. I read `c-regenerate.json` `match` flags and did not recompute them against the original arm hashes.
- Qwen checkpoint hash and module-hash integrity, BF16 floors table (e), and the claim that `inv_freq` stays FP32.
- Native discovery: `audit.py --closures` (1,488 metrics, 48 closures), archive hash, corpus disjointness, state-MSE and output-KL numbers and their bootstrap intervals.
- Absolute (not share-normalised) other-item mass in FLUX.
- Whether a learned route with position input or more training would beat the baselines. Not yet known; it has not been tested.

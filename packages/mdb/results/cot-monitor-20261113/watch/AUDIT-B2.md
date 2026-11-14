# Independent audit: B2 live-watchpoint CoT monitor (74f0bb8, 1f015ed, 2b24d95, 9142376, 88a3955)

Auditor: Claude (Opus 5.5), independent subagent, 2026-11-14. I did not use `summary-b2.json` or `analyze_b2.py` for any number. I recomputed P1 and P2 from the raw per-step letter logits and `to_consumer` in `data-b2/*.jsonl.gz`. My AUROC uses midranks, and my cluster bootstrap uses its own seed (7). I ran no GPU jobs. Scripts and outputs are in `watch/audit/`:

- `recompute_b2.py` writes `recompute_b2.json`. It covers every registered number, PH1/PH2, placebos, strata, pairs, a logistic model, exactness, and the cross-check against the 11-13 run.
- `extra_b2.py` writes `extra_b2.json`. It covers auditor post-hoc variants: the single-step P1 at k = K, an R-free P1, and early and late windows.
- `check_row_table_tiny.py` writes `check_row_table_tiny.out`. It checks `row_table` and `suffix_sweep` against mdb's stepped native path on every row and every k of the tiny model.

Labels: **measured** means I recomputed or ran it. **Asserted** means I reasoned it from code or from others' numbers.

## 1. Custody

| Claim | Verdict | Evidence |
|---|---|---|
| Prereg committed before any B2 score | **holds** (measured) | `prereg-b2.json` was committed at 1f015ed, 03:12:17 PST. Every B2 job ran revision 2b24d95, which comes after the prereg. mrun lists 4 smoke attempts beyond the reported one: e0f0fc7bcc75, 47a83c3918a6 and f4bf4d851d9a were `killed_ram`, and 2f327f895c0a was cancelled while queued. The killed attempts died at model load in under 5 s, and their logs contain no `B2SMOKE` line. Three shard jobs (c549eeb52059, b20d7d0c89cf, 42818c61b9c5) failed with an argparse error after 0.7 s and printed nothing. None of these 7 jobs is disclosed in the addenda or the tracker. They produced no output, so this is a minor disclosure gap, not a leak. |
| Timing-only smoke leaked no scores | **holds** (measured) | `job-4387230f3d7c` prints only seconds, memory, exactness diffs and `R_job`. `R_job` is the rival letter, which equals the pre-registered R from `r_1113.json`, so it is not a score. The shard logs print only timing, exactness and R agreement. |
| Post-hoc final-margin comparison logged in the addenda before it was computed | **holds with correction** (measured from the builder's session transcript, `agent-aee94ca625e91cb95.jsonl` lines 1328–1333) | The builder read the registered `summary-b2.json` at 11:51:28Z. At 11:51:46Z, one shell command first wrote `prereg-b2-addenda.json` (heredoc) and then ran an inline Python block that computed PH1/PH2 and merged them into `summary-b2.json`. So the logging did precede the computation, but only by moments, in the same command, and after the registered results had been read. That is acceptable for a labelled post-hoc look. **Gap:** the PH1/PH2 code exists only in the transcript and is not saved in the repo. I re-derived both independently and they match exactly. |
| Prereg and driver unchanged after data | **holds** (measured) | `git diff 9142376 88a3955` touches only data, logs, summary and addenda. |

## 2. Recomputed numbers (direct 67 vs anyway 67; rationalized 30)

Data integrity (measured):
- There are 164 unique uids with no duplicates. UIDs, labels and X match `subset.json`.
- R matches `r_1113.json` on 164/164 items.
- The probe-step schedule equals `round(j·K/33)` on all items.
- The stored margin and `ablated` values equal my values from the raw letters within 1e-5 (rounding).
- My P1 and P2 per-step truth values equal the stored `replay` output on 164/164 items.

| Quantity | Claimed | Recomputed | Verdict |
|---|---|---|---|
| `fired_fraction` AUROC [CI] | 0.836 [0.747, 0.911] | 0.836 [0.746, 0.910] | holds |
| fired_fraction TP at 6 / 16 FA | 35 / 46 (fa 6 / 15) | 35 / 46 (fa 6 / 15) | holds |
| fired_fraction AUROC, final margin < 2 (54 / 18) | 0.738 | 0.738 | holds |
| Registered integrated near-tie: AUROC, TP at 6 / 16 FA, stratum | 0.760 / 26 / 41 / 0.568 | 0.760 [0.673, 0.843] / 26 / 41 / 0.568 | holds |
| PH1 final −margin: AUROC, TP at 6 / 16 FA, stratum | 0.852 / 36 / 49 / 0.705 | 0.852 [0.783, 0.915] / 36 / 49 / 0.705 | holds |
| Stratum difference, fired_fraction − final ("within noise") | 0.738 vs 0.705 | item-bootstrap 95% CI of the difference [−0.16, +0.21] | holds |
| P2 companion | 0.541 | 0.541 [0.447, 0.640]; 0 TP at 6 / 16 FA | holds |
| T1 (median direct t* ≤ 0.5 and median anyway ff ≤ 0.10) | held: 0.030 / 0.029 | 0.030 / 0.029, held | holds |
| T2 (crossings on rationalized ≥ 0.30 and direct ≤ 0.10) | failed: 0.63 vs 0.84 | 0.633 vs 0.836, failed | holds |
| T3 (≥ 45 TP at 6 FA) | failed: 35 | 35, failed | holds |
| Falsifier | not falsified | (a) false: 35 > 26 and 46 > 41. (b) false: 0.738 > 0.568 | holds |
| Descriptives: median ff 0.56 / 0.03 / 0.06; fired any 65 / 38 / 19; M3 0.65 / 0.06 / 0.12 | as stated | identical | holds |
| PH2: P1 at k = K on 57/67 direct and 6/67 anyway; median run length 3 | as stated | 57 / 6 / 3 | holds |
| "Same count as the 11-13 frozen M2 flag" | 57 | Stronger than claimed: P1 at K and the 11-13 strict flip (margin > 0, margin_ablated < 0) pick **the identical item set** (63 items over all classes, 0 disagreements) | holds |
| Exactness: cache 9.2e-5; feed 1.5e-5 / 1.9e-5 (34 checks); `row_table` vs `symbols` 3.8e-6 | as stated | 9.16e-5; 1.53e-5 / 1.91e-5 (34 checks, 6 tokens fed each); 3.80e-6 (17 items) | holds. The 3.8e-6 misses the registered ≤ 1e-6 and is disclosed. |
| Independent cross-check (new) | — | At k = K, the sweep reproduces the 11-13 run's native margin within 5.4e-5 (median 5.3e-6) and its M2 within 2.2e-5 on all 164 items. That run used a different code path (`driver.py`, stepped mdb ablation). | strengthens the exactness claims |
| Cost: sweep ≈ 3 s per item; spot items ≈ 35 s; peak VRAM 9.3 GB; shards 390 / 367 / 299 s | as stated | sweep median 2.9 s; spot median 34 s; peak 9.31 GB; 390 / 367 / 299 s | holds |
| "About 45 min per item for row-by-row stepping" | asserted | The cost probe measured 2.7–5.0 s per row at T = 2112. Direct and anyway chains have median K = 2047, which gives roughly 1.5–3 h per item. 45 min is the 700-row figure from the memo. | **overstated in the favourable direction**: the speed-up is larger than claimed (asserted from the cost-probe numbers) |

## 3. Is the watch a margin proxy? (circularity)

P1 = `argmax == X and M2 < 0` is, at each step, the condition 0 < margin_k < Δ_k, where Δ_k = margin_k − ablated_k is the hint's causal effect at the answer row. A small margin makes P1 easier. I tested whether the item-specific Δ_k adds anything (all measured; all of these are auditor post-hoc analyses):

- **Constant-push placebo.** I used the same predicate with Δ_k replaced by the pooled, label-free constant c = 2.26 (`argmax == X and margin_k < c`). Overall it gives AUROC 0.730 and 10 TP at 6 FA. Within margin < 2 it gives 0.42. The real watch gives 0.836, 35 TP and 0.738. The hint-effect term carries real information.
- **Final-margin strata** (n direct / n anyway; AUROC of ff vs final −margin):

  | Stratum | n direct / anyway | ff | final −margin |
  |---|---|---|---|
  | < 1 | 38 / 7 | 0.79 | 0.69 |
  | 1–2 | 16 / 11 | 0.72 | 0.47 |
  | 2–4 | 9 / 18 | 0.89 | 0.68 |
  | ≥ 4 | 4 / 31 | 0.80 | 0.86 |

  The placebo stays at 0.42–0.56 in every stratum. The cells are small, but the direction is consistent: once the final margin is held roughly fixed, the watch still ranks direct above anyway.
- **Logistic model** (in-sample, clusters ignored), with label ~ signed-log final margin. Adding ff gives LR = 21.8 (p < 0.001). Adding the **single-step P1 at k = K** gives LR = 37.9. With P1@K already in the model, adding ff gives LR = 2.9 (not significant).
- **Matched pairs.** The final margin orders **all 67** subset pairs correctly (direct is lower in 67/67, and in 7/7 pairs whose 11-13 margins are within 1 logit). ff wins 0.81 of pairs, and 0.71 when the gap is under 1. The greedy "margin-nearest" pairing never put a more-confident direct item against a less-confident anyway item, so the pairs cannot show anything beyond margin. They only confirm that the subset is still margin-separated (median 0.81 vs 3.51, as the prereg disclosed).

**Verdict.** The watch is **not just a margin proxy** (measured). It carries the hint-ablation signal beyond the margin. That signal is the same one the 11-13 audit found in M2, and the prior audit showed M2 is a local version of the label's own "remove the hint, does the answer flip?" counterfactual. Almost all of the watch's extra information is already in **one** readout at k = K.

## 4. Fairness of the registered baseline, and what "beats it" means

- The integrated near-tie, median_k(−margin_k), was specified in the advisor memo (§2.4) and registered. It is a symmetric comparison (time-integrated against time-integrated), so it is **not a strawman by construction**. Mean −margin gives similar results: 0.782, 25 / 47 TP (measured).
- However, the final near-tie was already known before registration to be the strongest margin baseline: 0.937 on the full pool (11-13 audit). Integration dilutes it, because chain steps far from the end carry little margin information (measured: AUROC of −margin at k = 0 is 0.572).
- "The watch beats the registered baseline" is **true, but it carries little weight on the question the prereg asked** (does the time dimension help?). Against the final readout, ff is tied or behind: 35 vs 36 TP at 6 FA, 46 vs 49 at 16 FA, AUROC 0.836 vs 0.852. It is ahead only in the n = 18 stratum, and that difference is within noise.
- The decisive comparator is the watch's own single step (`extra_b2.py`, auditor post hoc): **P1 at k = K alone catches 57/67 direct at 6/67 FA (AUROC 0.881), against 35 for fired_fraction.** Integrating the watch over time **loses** recall at a fixed false-alarm count.
  - The late window (f ≥ 0.5) gives 0.879 and 36 / 58 TP.
  - The early window (f < 0.5) gives 0.732 and 25 / 37 TP.
- The builder states this honestly ("no measured gain over one final readout"). My data make it stronger: time integration costs recall.

## 5. Validity of the forced-answer suffix probe; leakage

- **The suffix is identical across classes** (asserted from code, consistent with data). `SUFFIX = "\n</think>\n\nThe answer is ("` is a module constant. It is tokenized once per job (7 tokens), and the letter ids are asserted per job. The feed spot-checks fed 6 suffix tokens on every checked item.
- **The cut points do not leak the label** (measured). Steps depend only on K, and the schedule is verified on all items. K does not separate direct from anyway (AUROC 0.510; median 2047 vs 2046), and neither does the hint-row count (0.467).
- **R uses future information.** R is fixed from the full-chain endpoint (the 11-13 consumer). This is label-free, but a live monitor would not know R at step k. An R-free variant (`argmax == X and M2 argmax != X`) gives AUROC 0.821 and 34 / 46 TP, essentially the same (measured). R is not driving the result.
- **What the probe measures** (asserted). It is an early-answering behavioural probe: it asks what the model answers if forced to close the thought now. That is a valid probe of the answer the state supports at step k. It is not a read of an internal variable. Cuts can fall mid-word, which is the same for both classes. At k = 0 the probe is a no-CoT answer to the hinted prompt.
- **Rationalized items differ in K.** Rationalized chains are much shorter (median K = 667), so T2 compares rationalized and direct items at different absolute granularity, although both have 34 steps. This is a caveat on T2's interpretation only. T2 failed regardless.

## 6. Code

- **Tests** (measured). I ran them with `common exec python -m pytest` and `PYTHONPATH=<wt>/packages/mdb/src:<wt>/src`. I confirmed that `mdb` and `saturn_pub` resolve to the worktree.
  - `test_analysis_rows_probe.py`, `test_ops_registry.py`, `test_engine_run.py`, `test_engine_breakpoints.py` and `test_engine_trail.py`: **711 passed, 1 skipped**.
  - Root `tests/test_adapters.py` (decoder), `test_ops_motion.py`, `test_engine_cells.py`, `test_engine_persist.py` and `test_engine_ports.py`: **557 passed**.
  - I did not try to reproduce the builder's "871" count, which uses a different module set.
- **`Run.feed` and the decoder hook: no regression on the native path** (code reading plus the tests above).
  - `_advance` arms the adapter only when `supplied is not None`, and clears it in `finally`.
  - `_commit` reads `self._feeding`, which is None unless `feed` is mid-call.
  - The decoder's `getattr(self, "_supplied_token", None)` falls through to `argmax` when nothing is armed, and it self-clears after one use.
  - Snapshot, restore and replay carry `supplied` as an additional tuple field. The checkpoint gains a `supplied` key only when a token was fed, so the existing checkpoint schema is unchanged otherwise.
  - On the real model, feed matched the cache sweep within 1.5e-5 on 34 checks.
- **`suffix_sweep`** (measured, tiny model, every k = 0..14): native ≤ 7e-9, M2 ≤ 1.9e-8 and CM1 = 0 against stepped mdb p-zero.
  - M2 zeroes the hint V cells only for the answer row's pass; earlier suffix rows read the hint natively. This is as registered.
  - Zeroing V equals p-zero without renormalisation. This is correct.
- **`row_table` chunked masked forward vs per-row stepped ablation** (measured, tiny model, **every** row, chunk = 1, 3, 5 and 512): rows after the arm columns match within 1.5e-8, and native matches within 3e-8.
  - **Minor bug:** for rows *inside* the arm columns (hint rows 6–9), `row_table` differs from the stepped p-zero by 0.0135. In `_row_ablation`, the row's self-read uses its freshly appended (un-zeroed) K/V on the diagonal, while the stepped reference zeroes the row's own column as well.
  - The builder's test samples only rows 12, 20 and 27, so it misses this.
  - It **does not affect B2**: `row_table` is a co-readout that enters no prediction, and all chain rows come after the hint. It should be fixed by zeroing the diagonal for rows in `cols`, or documented.
- **`replay` predicate logic: correct** (measured). Edge, level and change modes behave as documented; an undefined step counts as not holding and is listed. My independent P1 and P2 truth values equal the stored `holds` on all 164 items. `fired_fraction`, `crossings` and t* follow from them.
- **Quirk of the registered P2** (asserted). `share = to_consumer / margin`, so at steps with margin < 0 the comparison flips sign (it becomes tc_hint < tc_chain). P2 was registered this way and is reported as such. A cleaner companion would compare `to_consumer` directly. This does not change the null result at 0.541.

## 7. Verdicts on the tracker's B2 claims

| Claim | Verdict |
|---|---|
| Prereg before any score; smoke timing only | **holds** (7 no-output jobs undisclosed: minor) |
| PH1 logged in the addenda before it was computed | **holds with correction**: same command, after reading the registered results; the PH code is not saved in the repo |
| All registered numbers, PH1, PH2, P2 and exactness | **hold** (exact reproduction) |
| T1 held, T2 failed, T3 failed; not falsified | **holds** |
| "Time dimension adds no measured gain over one final readout" | **holds, and understated**: the single-step watch at k = K (57/67 at 6 FA) beats the integrated watch (35/67). Time integration costs recall. |
| "Integrated near-tie was a weak baseline" | **holds** (registered and symmetric, not a strawman, but foreseeably weaker than the final readout) |
| "Hint is decisive from the first probe steps, before the chain has said anything" | **overstated**. Descriptively, P1 holds at k = 0 on 31/67 direct items, but also on 15/67 anyway items. P1 at k = 0 has AUROC 0.62, and the early-window ff has 0.73. It is not specific to direct reads, and at k = 0 there is no chain, so the probe is the no-CoT answer. |
| "P1 flickers on direct items (median run length 3)" | **holds** |
| "P2 (exact accounting share) carries no signal" | **holds** (with the sign quirk in §6) |
| Cost and speed-up | **holds**. The 45 min per item row-stepping figure is understated for these ~2k-token chains (asserted ~1.5–3 h). |
| Verbs exact against mdb's stepped path | **holds** for `suffix_sweep`, `feed` and `replay`. **Holds with correction** for `row_table`: rows inside the arm columns are off by about 1e-2 on the tiny model, which does not affect B2. |

## 8. Overall

**Supported positives (measured).**
- Custody is clean.
- Every number reproduces exactly.
- The verbs are exact against mdb's stepped path and against the independent 11-13 run (≤ 5.4e-5).
- At matched false alarms, the hint-decisive watch separates direct from margin-nearest anyway items better than the registered integrated near-tie (35 / 46 vs 26 / 41). It is not a margin proxy: the constant-push placebo and the within-stratum results show that the item-specific hint effect carries information.

**Partial.**
- That beyond-margin information is the M2 hint-ablation signal already established on 11-13, and it is fully present at the final step. The P1@K item set is identical to the 11-13 strict flip.
- The B2-specific contribution, the time dimension, adds nothing measurable and loses recall when integrated (35 vs 57 at 6 FA).
- The registered falsifier was passed against a baseline that could not test this question.
- The new descriptive finding (the hint is decisive early, intermittently, with flicker) is real for direct items, but it is not distinctive: anyway items also fire at k = 0 (15/67).

**Not yet known.**
- Whether a time-resolved score that does not dilute the end (for example, the late window, 0.879) beats P1@K on fresh items. That needs its own registration.
- Behaviour without the oracle hint location.
- Negatives that are margin-matched in fact, not only by nearest neighbour.

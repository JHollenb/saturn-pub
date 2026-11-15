# Independent audit: two-hop Run 1 (1a + 1b) and TP1 (token × position artefact map)

Auditor: Opus, 2026-11-14 ~01:00 PST. Scope: commits e9fb237..3f9c474 on `mdb-twohop` (local), claims in `obsidian/research/2026-11-13-211248-mdb-experiment-tracker.md` (entry 1 "Run 1" block, "Most interesting", entry 9). I recomputed every number myself with new scripts in `next/audit/`. I did not reuse `r1_score.py` or `tp1_analyze.py`.

- `audit/a_local.py` → `audit/a_local.json`: P1, P2, P5 (logits), P6, P7, P8, using the saved arms and forecasts only.
- `audit/a_beast.py` (+ `a_beast_run.sh`) → `audit/a_beast.json`: P3, P4, P5 (content and hop shares), P9, P10, and all TP1 re-slices. It reads the Run 1 captures and TP1 arrays on Beast. It loads the weights with safetensors and does not import mdb. One CPU job, **job-65c05f402076** (priority 70, 25 s wall, 5.7 GB RSS, no GPU).
- `audit/a_tl_processed.py` → `audit/a_tl_processed.json`: TransformerLens 4.4.0 on a tiny random Qwen3 (local CPU).
- `audit/a_custody_jobs.py`: mrun job timestamps.

All numbers below are MEASURED from these scripts unless marked asserted.

## Verdict summary

| Claim | Verdict |
|---|---|
| Custody: forecasts sealed before scoring; preregs before jobs | holds, with caveats (scorer changed after first scored output) |
| P1 pooled Spearman 0.20; 67–69% < 0.07; flip mismatch 32/64 | holds |
| P1 "top cell 174/178" as evidence the forecast "picks the live site" | overstated (a constant-site rule also gets 174/178) |
| P1 "Spearman 0.98–1.0 on \|dM\| > 0.5" | holds with correction (post hoc; 28/64 and 40/114 items; robust across thresholds) |
| P2 L18; 35/49; 45/64 | holds (the extra live cells are mostly the same row, KV6, at L17–18) |
| P3 167 vs 143 of 178 | holds |
| P4 "f32 does not explain the miss" | overstated: the L0 test cannot discriminate |
| P4 "zero cut misses by 19% at L0, not yet known why" | settled: bf16 rounding, no bug |
| P4 "no hidden relay, 0/64" | **wrong as evidence**: the base relay transplant is a no-op |
| P5 62/63; 34/64; 0.45/0.55 | holds |
| P6 46/64 | holds |
| P7 0.31 | holds |
| P8 33/34, 32/34 | holds with correction (named set chosen in-sample) |
| P9 0.93 vs 0.90 | holds (weak test: 44/114 spans have one row) |
| P10 108/114, 104/108, 12/64 | holds |
| TP1: t0 ≈ 75× norm, < half the median ‖∂M/∂h‖ | holds as preregistered; "low sensitivity" overstated (scale-free sensitivity is 37× higher) |
| TP1 T3: position mean \|dM\| 0, 20/20 | holds |
| TP1 T2: function words, 188/710 | **wrong as a reading**: driven by 32 prompts with an identical prefix; sentence prompts give 156/214 for token |
| TP1 falsifier −0.60 | holds (pooled); depth-driven, within-layer +0.10 |
| TP1 second tier "high-norm, nearly insensitive" | overstated: 505/529 entries are the layer-27 input of shared-prefix rows |
| TP1 I1 3/4; I2 0.9995; I3 6/7; I4 28/28 | holds |
| TP1 "TL skips centring on RMSNorm by design" | holds with correction (wrong quote; `_weights_processed` evidence is an interop bug) |
| TP1 "row 0 regardless of token" | holds (14 of 16 plain prompts start with a non-article) |
| Token-id overwrite fix | correct |
| Tests | 540 passed (no source or test files touched in range) |

## 1. Custody (MEASURED from git and the mrun job records)

| Event | Time (PST) |
|---|---|
| prereg_run1 `25bad7b` | 11-13 23:41:17 |
| r1.py final `8ce058a` | 23:49:03 |
| 1a job-292583e998fd created / started / finished | 23:49:04 / 23:53:43 / 00:03:09 |
| r1_forecast.py final `166f7e6` | 23:49:49 (before any arm existed) |
| 1a forecast job-329014854b56 | 00:03:44–00:05:06 |
| sealed `e9fb237` (forecasts-1a) | 00:06:14 |
| 1b jobs affe467927f9 / 09ddfd6a97ee | 00:04:21–00:14:11 / 00:15:56–00:22:35 |
| 1b forecast job-cca098b077ab | 00:23:27–00:25:32 |
| sealed `3d5502b` (forecasts-1b) | 00:26:14 |
| **first scoring job-ffc385648f56** (rev 3d5502b) | 00:26:30–00:27:07 |
| scorer edit `054b074` (adds the \|dM\| > 0.5, top-cell and top-5 diagnostics) | 00:29:17 |
| second scoring job-21ec9a234d22 | 00:29:18 |
| scorer edit `8a6ee35` (hard-item capital set) | 00:31:39 |
| final scoring job-2cbdee11a05c | 00:31:39 |

- Both forecasts were sealed before any scoring job ran. `r1_forecast.py` reads only `CAPDIR/cap/`; it never opens `arms-*.jsonl` (I read the code). The 1a job log prints only id, relay row, arm count and seconds, with no arm values (I read the log via `mrun attach`). I cannot verify whether the builder opened the arms file on the laptop between 00:03 and 00:06. Nothing in the record suggests it.
- **Caveat 1: the scorer changed after the first scored output.** job-ffc385648f56 printed the pooled single-cell Spearman of 0.196 (base and hard) at 00:27. `054b074` then added the "big-cell" Spearman, top-cell and top-5 diagnostics that P1's rescue rests on. `8a6ee35` changed which capital counts as correct for hard items, which affects the P10 hard capital co-readout only. These are post-hoc analysis choices made after seeing the result.
- TP1: prereg `2892faf` was committed at 23:47:34 and the addendum `ec92570` at 23:54:59, before the first TP1 smoke (23:55:13) and the main run (00:29:54). `tp1.py`'s last commit `a617879` is at 00:29:54, the same second as the job. `tp1_analyze.py` was first committed at 00:34:44, after the data existed. Its class and position definitions follow the prereg, so this is acceptable.
- **Caveat 2: inverted harness status on the analysis jobs.** job-c195d1de6848 (cited) has mrun state `failed`, but its log shows `DRIVER_RC 0` and the full analysis. The runner's final `grep -v` finds nothing, and `pipefail` turns that into exit 1. The earlier job-2983525213ce shows `succeeded` but crashed with a TypeError (the token-id bug). Both ran at REVISION e261c2c. The token-id fix was committed later, in `696b2f0`, so the analysis ran from an uncommitted tree. That fix only affects the T3 class counts.

## 2. Run 1 recomputation

### P1: forecast vs native

- Holds: pooled single-cell Spearman median 0.196 / 0.196. Cells with \|dM\| < 0.07: 0.674 (base) and 0.688 (hard). Most of these are exactly 0: 0.671 and 0.679 of cells have dM = 0.000, because the logit grid is 0.125. Forecast top cell = native top cell 63/64 and 111/114 (no native ties). Forecast top cell in the native top 5: 64/64 and 114/114. Forecast/native on \|dM\| > 0.5 cells: 0.676 and 0.736. Donor flip mismatch 32/64: 8 predicted flips against 40 observed; median \|err\|/\|native\| 0.497. Relay cut 0.95 / 0.31 and 0.94 / 0.26; 2×2 B 0.97 / 0.21; knock-out 0.77 / 0.59.
- **Constant-site baseline (new).** "Always the L20 KV6 cell at the gt row" is the native top cell on 64/64 + 110/114 = **174/178**, the same as the forecast. "Always the L20 KV6 cell at the last descriptor row" gets 55/64 + 101/114 = 156/178. The forecast's real content is finding the per-item row passively. It beats the last-row rule by 18/178. It does not "pick the live site" better than knowing the site is always L20 KV6.
- **The \|dM\| > 0.5 restriction was not preregistered.** The prereg says "Spearman ≥ 0.8 over single liveness cells (pooled per item, median over items)". The restriction was added after the 0.196 was seen (§1). Is it cherry-picking? Mostly not:

  | threshold | base items (≥ 3 cells) | Spearman, outcome-selected | Spearman, selected on forecast \|fc\| > t | hard items | outcome-sel. | forecast-sel. |
  |---|---|---|---|---|---|---|
  | 0.25 | 53/64 | 0.956 | 0.976 (57 items) | 100/114 | 0.955 | 0.975 (107) |
  | 0.5 | 28/64 | 0.980 | 1.00 (27) | 40/114 | 1.00 | 1.00 (42) |
  | 1.0 | 6/64 | 0.974 | — | 7/114 | 1.00 | — |

  The high rank agreement survives selecting on the predictor instead of the outcome, and at the lower threshold where most items qualify. So it is not an artefact of conditioning on the outcome. Caveats: at 0.5 the median item has only 2 such cells, so the statistic covers a minority of items with 3–5 cells each. Spearman over \|·\| on all cells is 0.45 / 0.46, still below the preregistered 0.8. The 192 relay-row cells per base item are exact no-ops (see P4), which is part of the zero mass. **Verdict:** P1 failed as written (correct). The rescue is a supported post-hoc observation, but "Spearman 0.98–1.0" needs "post hoc, on 28/64 and 40/114 items" beside it.
- The t0-mask row (0.67) silently drops 44/178 items whose bf16-p closed form is NaN (p₀ rounds to 1, so 1/(1 − p) blows up). With the f32 p it is 0.87 on 178/178.
- The replica's per-layer error is median 0.4% over all (item, layer) entries; the median of per-item maxima is 0.8%, and the max is 1.8%. \|M_replica − M\| ≤ 0.218. The tracker's "median 0.8%" is the per-item max.

### P2

Holds. First causal layer median L18 (IQR 18–20; histogram L20 30, L18 21, L17 7, L16 4, L15 1, L9 1). Gap > 8 layers on 35/49. Empty outside L20 on 19/64, so 45/64 have other live cells. Context the tracker omits: those cells are mostly the **same row and KV head one or two layers earlier**: L18 KV6 at the gt row on 29 items, L17 KV6 on 8, L21 KV1 on 11. "Partly site selection" fits a short L17–L21 write window on the same row better than a different site.

### P3

Holds exactly. Share / contrast Spearman 0.068 / 0.059; contrast beats share 55/178; top cell 167 vs 143 of 178; big-cell Spearman 0.91 vs 0.80 (68 items).

### P4: cuts (the anomaly is settled)

- **Zero cut at L0, 19%: bf16 rounding, not a bug (MEASURED).** I emulated L0 in bf16 from the captured p, V and W_O: A = bf16(p·V) per head, then o = bf16(W_O·A). The emulation reproduces the native A bit-exactly (median 100% of elements) and the native o (median 100%). With p_row set to 0 it reproduces the **measured cut Δo exactly**: relative error median 0.0, 100% of elements bit-equal. Against this exact emulation, the closed form misses by the same 0.188. Rounding A alone accounts for 0.062 of it, and rounding o the rest.
  - The mechanism: at L0 the cut row's attention is tiny (p₀ median 0.0045 over heads), so \|Δo\|/\|o\| = 0.012. Rounding A and o to bf16 (relative 2⁻⁹) is a large fraction of so small a change.
  - At L27 in FT1, p₀ is large, so the same closed form matched to 0.1%.
  - Hook point, RoPE, GQA mapping (h // 2) and the o_proj input are all consistent: the bit-exact emulation would fail otherwise.
  - The "within 2 bf16 spacings of o on every element" test fails because A's rounding propagates through W_O. That is rounding outside o's own spacing, not a model error.
- **The mask / f32 test at L0 cannot discriminate.** Renormalising by 1/(1 − p₀) moves each p by about 0.45%, which is below bf16's spacing for p (0.4–0.8%). The mask cut's Δo at L0 is therefore dominated by how each renormalised p rounds. My approximate emulation (bf16 p / (1 − p₀), no fp32 scores captured) is itself off by 0.24. Neither the bf16-p nor the f32-p closed form can be within 2 spacings at L0. "f32 does not explain FT1's miss" is untested, not refuted. The test belongs at FT1's layer (L27, p₀ large), which needs the cut run's edited L27 inputs (not captured).
- **"No hidden relay, 0/64" is vacuous (MEASURED).** On base, the relay row is row 26 on all 64 items, inside the few-shot prefix. The donor shares the recipient's tokens through the relay row on 64/64 items, so the donor's V there is identical and the transplant changes nothing. The band dM is exactly 0.000 on 64/64, and every relay-row cell in the liveness sweep is exactly 0. On hard, 82/114 donors share the prefix through the relay row, and only 32/114 items have any non-zero relay effect. The relay cuts (zero / mask) are real edits and unaffected. **Verdict on P4's relay part: wrong as evidence; untested.**

### P5

Holds.
- Country mean \|dM\| < panel mean on 60/63 (median 0.5 vs 2.69).
- Kept: 47/48 non-city and 15/15 city (62/63).
- The country mean's content reads its own country on 63/63 and capital 61/63.
- The other-country mean names the other country 64/64. Its argmax is that country's capital on 34/64. The donor cell keeps the answer on 23/64.
- Hop shares (median, \|native\| > 0.25):

  | Swap | direct | MLP | re-read |
  |---|---|---|---|
  | donor | 0.453 | 0.553 | −0.010 |
  | other-country | 0.525 | 0.597 | −0.093 |
  | panel | 0.726 | 0.567 | −0.189 |

  The tracker says 0.51 for other-country direct; mine is 0.52 under a \|native\| > 0.25 filter, which is immaterial. As the audit rules require, this is accounting, not a forecast.

### P6 and P7

- P6 holds: 46/64; medians A −7.81, B −1.375, I −0.44. A is the donor-0 swap (identical logits on 64/64).
- P7 holds: 119/384 sub-additive, 215 super-additive, 50 exactly additive. 125/384 pairs are within one logit step (0.125) of additive. "Not redundant copies" is fair; "mostly additive or super-additive" is more exact.

### P8

Holds as counted: 33/34 and 32/34; medians named −2.125, random 0.0, answer-token null 0.0. Named knock-out keeps the answer on 28/34.
- Correction: the named neurons were chosen from Z2's ablate arm **on these same items** (the prereg says "in-sample"). The answer-token null removes the answer-token confound the Z1/Z2 audit raised. It does not remove selection on outcome.
- "The recurrence is the bridge's, not the answer token's" holds against the answer-token confound only. An out-of-sample set is needed before calling it composer evidence.

### P9

Holds: R_argmax 106/114 vs R_last 103/114, base 58/64 vs 55/64, R_first 47/114. The builder's random draw gets 76/114, and its expectation is 75.9. Weak test: 44/114 hard spans (15/64 base) have one row, where every rule is right. On multi-row spans it is 62/70 vs 59/70. The falsifier fires as stated.

### P10

Holds exactly.

| Readout | base country | base capital | hard country | hard capital |
|---|---|---|---|---|
| declared carrier | 63/64 | 61/64 | 108/114 | 104/108 |
| B_answer at L20.h | 12/64 | 16/64 | 27/114 | 24/108 |
| B_answer at L20.h_out | 60/64 | 61/64 | 96/114 | 98/108 |

On the all-set at L20.h, 53/64 winners are capitals. The hard capital co-readout uses the bridge's canonical capital, a rule set after scoring (§1, `8a6ee35`).

## 3. TP1 recomputation

- **The map at L14 holds.** Median ‖h‖: t0 13,929, non-t0 180.7, so t0 is **77×**; content 185, function 175, punctuation 159, whitespace 172. Median ‖∂M/∂h‖: t0 0.0118 vs non-t0 0.0249, a ratio of **0.47**. Mass received per query row: t0 10.2.
- **Sensitivity metric.** T1 uses ‖∂M/∂h‖ and the falsifier uses ‖∂M/∂h‖·‖h‖, both as preregistered. But under RMSNorm, ‖∂M/∂h‖ scales roughly as 1/‖h‖. On the scale-free measure ‖∂M/∂h‖·‖h‖, t0 is **37× more sensitive** than the non-t0 median (165.7 vs 4.49). For the top-1% set it is 112 vs 3.5. "Under half the median sensitivity" is true for the preregistered metric and misleading as a statement about the row. T3 shows that t0 is *constant* (replaceable by its position mean). It does not show that t0 is *insensitive*: the global-mean replacement moves M by a median of 0.875 and flips the argmax on 5/20.
- **T1 holds**: 555/555 top-1% entries are t0; ‖∂M/∂h‖ is 0.0077 vs 0.0187.
- **The second tier is overstated.** The non-t0 top 1% has median ‖h‖ 3,742 and ‖∂M/∂h‖ 8.6e-5. But **505/529 of those entries are at the layer-27 input**. The rest are one each at L3–L26 (the `Water` row 1 of a lead_eot prompt).
  - Almost all of them are 16 distinct shared-prefix rows of the 32 two-hop prompts (`' the'` r30, `' containing'` r19, `'The'` r27, `' of'` r29…), each counted 32 or 16 times.
  - A non-last row at the last layer input reaches M only through one attention layer, so a near-zero gradient is expected.
  - These are late-layer massive rows in one shared context. They are not a general class of high-norm, insensitive rows.
- **The falsifier holds as preregistered.** The pooled Spearman(‖h‖, ‖g‖·‖h‖) is −0.602; with ‖g‖ alone it is −0.938. **Within-layer** (per prompt, per layer, then median), Spearman(‖h‖, ‖g‖·‖h‖) is **+0.10** (IQR 0.00–0.26). So the −0.60 is depth. Within a layer, higher-norm non-t0 rows are slightly *more* sensitive on the scale-free measure. The falsifier's > 0.5 still does not fire.
- **T2 for function words: the reading is wrong.** I reproduce 188/710 (0.26). But **496 of the 710** rows come from the 32 two-hop prompts, which share an identical few-shot prefix. Causal attention makes their h bit-identical at the same positions. Each position mean therefore contains up to 32 copies of the row itself (cos_pos q3 0.99997).

  | Subset | token beats position |
  |---|---|
  | two-hop rows | 32/496 |
  | **sentence rows** | **156/214 (0.73)** |
  | sentence rows, leave-one-out means | 160/214 |
  | sentence rows, means built from sentence prompts only (self excluded) | 197/214 |

  On prompts without a shared prefix, function words are closer to their **token** mean. The primary prediction holds there. "Function words sit closer to their position's mean" is a property of the duplicated prefix, not of function words.
- **T3 holds**: 17/20; median \|dM\| 0.0 (position) vs 0.875 (global); kept 20/20 vs 15/20; token mean 0.0 and 19/20.
  - The 20 top rows are row 0 of 11 plain prompts (first tokens Quick, Paris, Sh, ", Seven, Mount, Light, Coffee, Gold, Water, 4) and 9 lead-newline prompts. This genuinely supports token-independence.
  - Caveat: 16 of the 20 "median-norm" rows are the same row (two-hop row 8 `' the'`) in an identical context, so the effective n is about 5.
- **T4 / row 0 regardless of token: holds.** Row 0 is the max-norm row on plain 426/432, lead "The" 431/432, newline 416/432, `<|endoftext|>` 425/432 and two-hop 864/864.
  - All 16 plain prompts have row 0 as the max on ≥ 26/27 layers, at 78–87× the median at L14.
  - Their first tokens are Paris, 4, ", Water, Mount, In, Ban, Coffee, Sh, Seven, D, Tok, Light, Gold, A and Quick. Only "In" and "A" are function words or articles.
  - In the plain and `<|endoftext|>` prompts, every exception is at the layer-1 input, before the sink forms. I did not break out the 16 newline exceptions.
- **I1 holds**: rel. diff 0.77%, 0.27%, 3.4%, 1.5%, so 3/4 are within 2%.
- **I2 holds**: centred fraction 0.99950 (n 555).
- **I3 holds** on 6/7 layers. Top vs other: L0 0.249 / 0.215 (n_top 50 from ties), L4 1.26 / 0.27, L8 0.88 / 0.38, L14 0.83 / 0.41, L20 0.47 / 0.36, L24 0.006 / 0.30 (reversed), L27 0.275 / 0.265 (a marginal margin).
- **I4 holds**: 28/28, only on rows 0 and last of 2 prompts.
- **The token-id overwrite fix is correct.** In `tp1.py` the record sets `"token": s["ids"][r]`, then the arm loop overwrites `rec["token"]` with the token-mean arm's result. The ablation itself used `tok_mean.get(s["ids"][r])`, evaluated before the overwrite, so the arm is right. The analysis now classes rows by `ids[prompt][row]`, and `p*.pt` sort order equals the prompt index.
- **TransformerLens centring: holds with correction.** In the installed TL 4.4.0 (the same version, read from source), `ProcessWeights.process_weights` centres writing weights only when `normalization_type in ["LN","LNPre"]` and not `final_rms` (`weight_processing.py` l.1836–1840). Qwen3 is `RMS`, so the residual is not centred by design (true). However:
  - (a) the quoted comment ("RMSNorm outputs are not zero-mean…", l.1110) belongs to the unembedding / ln_final fold, not to the writing-weight gate;
  - (b) **`_weights_processed` staying false is not evidence**: `mdb_interop.tl.processed()` reads `bridge.model._weights_processed`. `bridge.model` falls through to the HF `Qwen3Model`. On a tiny random Qwen3 I measured `bridge.__dict__["_weights_processed"] = True` while `processed()` returns False (`audit/a_tl_processed.json`). TL's `resid_pre` equals the native h to 9e-8, so it is uncentred. Issue T8's "processed() returns False" is an interop bug in `processed()`. "No-op" is also wrong: fold_ln and center_unembed do run;
  - (c) the 3.2% versus h − mean_d(h) also contains fp32-vs-bf16 drift, so it does not by itself show no centring.

## 4. Tests

No `src/` or `tests/` files changed in e9fb237..3f9c474. Only results, the vendored `interop_pkg`, and `issues.jsonl` changed. I ran the modules these results rest on: `tests/test_analysis_sweep.py` (cell_sweep) and `tests/test_engine_ports.py` (p/s cut edits). Command: `PYTHONPATH=<worktree>/src:<worktree>/packages/mdb/src common exec python -m pytest -q`. Result: **540 passed**. Without the PYTHONPATH, `common exec` picks up the main checkout's `saturn_pub` and fails to import `InPlaceWrite`, which is an environment issue.

## 5. What I could not verify

- Whether anyone opened the arms files between their creation and each seal. Code and logs are clean.
- The mask-cut f32 question at the layer where it matters (L27). That needs the edited-run inputs at L27, which were not captured.
- The fp32 softmax scores for an exact mask-cut emulation (`s` not captured).

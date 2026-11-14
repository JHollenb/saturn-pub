# AUDIT-SCALE: independent audit of the scale study (tracker entry 10)

Auditor: independent Opus agent, 2026-11-14. Scope: commits `17dd554..04c6ba5` on `mdb-twohop`, `results/scale-20261114/`. Method: I recomputed every audited number from the raw artefacts with my own code (raw A1 sweep tensors and raw A2 arm outputs copied from `beast:/mnt/big/mrun-scratch/job-*/`, Job F outputs, capture logs). I also ran one CPU-only mrun job (`job-e14f3f4acb80`, priority 70, 0 MB VRAM, 18 s, 5 GB RSS). It re-implements the carrier readouts from the Job C captures and the raw safetensors without mdb or `score.py`, and adds the nulls the study lacked. No GPU job was run.

Scripts and outputs are in `audit/`:
- `a1_carriers.py` / `a1_carriers.json`: carriers, KV head bias, write windows and family agreement, recomputed from the sweeps.
- `a0_rows.py` / `carriers.json`: the carrier row per item.
- `a2_rescore.py` / `a2_rescore.json`: hop budget, B3, B4, B6, the split, hard prompts, the supplement and base rates.
- `a3_beast_cpu.py` / `a3_beast_cpu.json`: independent readouts, a full-subspace projection, permutation nulls, L_first with 20 chance draws, and a KV-index lineage test. Its log is `logs/job-e14f3f4acb80.log`.
- `a4_forecast.py` / `a4_forecast.json`: replica and first-order ratios.
- `data/`: copies of the raw sweeps and arms.

Labels: **measured** means recomputed here from the data; **asserted** means my reasoning or knowledge, not measured.

## Summary of verdicts

| # | Claim | Verdict |
|---|---|---|
| 1a | One carrier per Qwen3 model (30/30, 41/41, 42/42) | **holds** (measured; also out of sample) |
| 1b | Same KV6 at three Qwen3 sizes falsifies D2 and is "a surprise worth a post" | **overstated**: D2's sealed falsifier fires literally, but 0.6B and 1.7B share KV indexing network-wide and 4B does not, so the real coincidence is about 1 in 8 at 4B |
| 1c | Causal-argmax selection biases the head choice | **no bias found** |
| 2 | The fraction rule (A1) beats the tail rule (A2); Qwen2.5 carriers KV1/KV1/KV0; 38/40 at 1.5B | **holds with correction**: the call rests on 4B alone, and both rules miss 0.5B |
| 3 | The sealed write-window bar fails; the carrier-KV restriction is post hoc | **holds** |
| 4 | Replica 0.98–1.04× native; first-order 0.28–0.73 | **numbers hold; interpretation wrong** ("predicts every cell transplant to within 4%" and "retires the first-order forecast") |
| 5 | Co-readouts are the country direction on six models; language below its shuffled chance | **wrong as stated** (the null is mis-specified; the cell carries capital and language information outside the whole country subspace) |
| 5b | B6: the language drop is smaller than the capital drop | **holds** |
| 6 | B1 at chance; B2 0.5995; B3 1/64; B4 0.46; readable errors at d4–d6 of 0.62–1.0 | B1 **holds with correction**; B2 **holds** (it is a coin flip at the threshold); B3 **holds**; B4 **holds with correction** (ρ = 0.494); readable errors **partial** |
| 7 | Hard and native-name errors sit downstream of the cell; native names are not harder | **partial**: downstream holds on 2 small models with n = 8–18; "not harder" is confounded by the English control |
| 8 | The MLP share tracks the tail; 4B re-read 0.33 | **numbers hold; the tail dependence is overstated**, and the "re-read" label is mis-described |
| 9 | Custody | **holds, with two undisclosed gaps of negligible to minor impact** |
| 10 | Code | **holds**; one latent bug and one tie bug noted |

## Scope note that applies to every causal number

Neither the A1 single-cell sweep (`cell_sweep`) nor the A2 split re-runs the rows between the edited row and the answer row. The run prefills all but the last token, and the edited cache cell is read only by the answer row. Source: the `mdb.analysis.sweep` docstring, `Debugger.run(prefill=None)`, and the `arms.py` split, which uses `dbg.run(ids)` and then `set_cell`.

So the "carrier" is the V cell whose direct read by the answer row moves the margin most. It is not the place where the country is computed. Every dM, the write window, the concentration and the hop-2 split are all inside this answer-row-only program. Only B3 (the back-patch, stepped from `span[0]`) recomputes the intermediate rows. This is a design choice carried over from entry 1, not an error, but it bounds what claims 1–4 and 8 mean.

## 1. Carrier KV head (KV6) at every Qwen3 size

**KV head counts (measured, config.json on Beast).** Qwen3-0.6B-Base has 16 query heads and 8 KV heads. Qwen3-1.7B-Base has 16 and 8. Qwen3-4B-Base has 32 and 8. All use head_dim 128. Qwen2.5-0.5B, 1.5B and 3B have 2 KV heads (14, 12 and 16 query heads).

**Carriers (measured, `a1_carriers.py`, my own argmin and modal code on the raw sweeps).** I reproduce L21 KV0 (20/28), L20 KV6 (30/30), L22 KV1 (38/40), L20 KV6 (41/41), L27 KV1 (31/42, runner-up L31 KV0 on 11) and L24 KV6 (42/42). The carrier rows agree with the builder's on 128/128 items for every model. My independent readout from the raw safetensors (`a3`) agrees on the country rank on 128/128 items for every model.

**Out of sample (measured, new).** The carrier was picked on correct base items. On the correct non-base items (depth, dist, obsc), the item argmax equals the model carrier on 54/54 (0.6B), 58/58 (1.7B) and 69/71 (4B). The one-carrier positive is strong.

**Selection bias (measured).** Every KV head has the same number of swept cells, so the argmax has no structural preference. Within L_c, KV6 is the argmax on every correct base item at all three Qwen3 sizes. The median best single-cell effect of KV6 is 9.1, 7.9 and 6.7 logits, against ≤ 1.2 for every other KV head. Away from L_c (layers outside L_c ± 1), the argmax head is spread:
- 0.6B: KV2 15, KV6 8, KV0 7;
- 4B: KV5 19, KV0 13, KV6 2.

So the method does not favour KV6. KV6 is the head with the dominant effect at the carrier layer.

**Is "same KV index" meaningful? (measured, new lineage test in `a3`).** KV indices are arbitrary labels in independently trained models, so an index match means something only if the models share head indexing. I built token-space OV maps for every (layer, KV) of the three Qwen3 models on the panel's ~200 candidate tokens, using shared-vocabulary embedding inputs. I then correlated maps between models at matched relative depth.

| Pair | Mean corr, same KV index | Mean corr, different index | Best match at the same index (of 8 per layer; chance 1) |
|---|---|---|---|
| 0.6B vs 1.7B | 0.132 | 0.006 | 4.46 |
| 1.7B vs 4B | 0.025 | 0.020 | 1.32 |
| 0.6B vs 4B | 0.023 | 0.015 | 1.18 |

0.6B and 1.7B share KV indexing across the whole network. Asserted: most likely shared lineage or initialisation. 4B is not aligned with either.

At the carrier layer, two KV heads copy country identity perfectly: their token-space OV maps send each of the 57 countries to itself as the top-1 token among the 57:
- 0.6B and 1.7B: KV2 and KV6, 57/57 each;
- 4B: KV4 (57/57) and KV6 (54/57).

**Chance level.**
- Under uniform independent labels, P(all three equal) = 8/512 = 1/64.
- Given the measured 0.6B–1.7B alignment, only the 4B match is a coincidence: about 1/8, or about 1/4 given that 4B has two country-copy heads at L24.

**Verdict.**
- D2's sealed falsifier ("the same KV index at 0.6B, 1.7B and 4B") fires literally: holds.
- The reading that the index is a conserved role across sizes, "a surprise worth a post", is **overstated**. Two of the three models share an index map, and the third match is a roughly 1-in-8 event.

## 2. Fraction rule against tail rule

**Measured.**
- 4B is at L24 (0.67 of depth) and 3B at L27 (0.75). Both are ≤ L28, so A1 wins by the sealed rule.

**Corrections.**
- L27 on 3B lies inside both A1's range (L23–28) and A2's range (L27–31). The discrimination therefore rests on 4B alone.
- 3B's mode is weak (31/42, below the 0.80 bar). Its runner-up, L31 KV0 on 11/42, would have been an A2 point.
- Both rules miss 0.5B: L21/24 = 0.88, a tail of 3.
- Observed fractions: Qwen3 0.67–0.71, Qwen2.5 0.75–0.88.

The Qwen2.5 carrier heads (KV1, KV1, KV0) and the 1.5B coverage of 38/40 are reproduced. With 2 KV heads per model, the Qwen2.5 index carries little information (D3, as the memo says).

**Verdict: holds with correction.**

## 3. Write window

**Measured.**
- Sealed bar (any KV head above the 2-spacing floor at the carrier row): 5/28, 1/30, 4/40, 22/41, 0/42 and 12/42. It fails on every model.
- Carrier-KV-only (post hoc): 17/28, 25/30, 13/40, 38/41, 11/42 and 36/42.
- Median window, carrier KV only: 4 layers at 0.6B, 3 at 1.7B and 1 at 4B. At 4B it is mostly one layer.

The timeline confirms the restriction is post hoc. It was added in `eeacf0d` at 04:58, after the first scores at 04:47–04:48. Even post hoc, the bar fails for 1.5B and 3B.

**Verdict: holds.**

## 4. Replica forecast

**Measured (`a4`).**
- Median replica/native ratio per base family: 0.98–1.04 on all six models.
- Median first-order/native at the carrier: 0.28–0.73.
- Maximum replica null error: 0.25–0.65 logit.
- Individual replica arms range from −0.68 to 1.96× native. The median absolute error is 0.05–0.12 logit.
- The replica is run only on the 3 top first-order cells and the top proposal cell per item, all selected by the forecast itself.

**What the replica is.** `forecast.py` rebuilds the cache from the captured K and V, edits one V cell and re-runs the answer token through the model in fp32 on CPU. The "native" A1 number (`cell_sweep`) is the same operation: the cached K and V, one edited cell, and a re-forward of the answer token only, in bf16 on GPU. The replica is therefore the native computation repeated in another dtype on another device.

It demonstrates:
- the captures are faithful;
- the cell effects are not bf16 artefacts.

It does not demonstrate any predictive or "forecast" power, and it says nothing about the effect of a cell on the intermediate rows, which neither side computes.

The first-order/native ratio (0.28–0.73) is a genuine measurement: about 30–70% of the one-row effect is nonlinear in the cell.

**Verdict: numbers hold.**
- "A one-row replica predicts every cell transplant to within 4%" is **wrong**. The 4% describes medians over forecast-selected cells, individual arms are off by up to about 2×, and the replica is not a forecast.
- "Retires the first-order forecast as anything but a ranking screen" does not follow from a same-program check.

## 5. Co-readouts and B6

**Reproduction (measured).** My independent code (`a3`) reproduces the rank-1 numbers exactly on all six models:
- capital raw/projected: 19/10, 37/31, 43/30, 42/32, 38/19, 45/40;
- language: 30/9, 40/34, 43/30, 41/31, 44/17, 51/40.

**Problem 1: the capital test has no power.** The lexical null (u_c alone names the capital) hits 46–48/48. A co-readout counts only where that null fails, so "counts" can be at most 0–2/48. "Capital co-readout 0–1/48" is a property of the test, not a finding.

**Problem 2: the shuffled-country "chance" is not chance.**
- Projecting out another country's direction leaves the true country direction in the write. The "chance" therefore approximates the raw readout: shuffled 30, 40, 42, 39, 44, 51 against raw 30, 40, 43, 41, 44, 51.
- The real chance is an answer-permutation null (the projected readout scored against other items' languages): 1.0–2.6/56.
- The rank-1 projected language (9–40/56) is far above that null.
- "Below its own shuffled-country chance on every model" means only that projecting the true country removes more than projecting a random one, which is expected.
- Asserted: this null was sealed by the memo (§1(b)), so the builder followed the prereg. The error is in the interpretation "hop 3 is not in the cell".

**Problem 3: on the items where the test has power, it fires.** Where the lexical null fails, the rank-1 projected language names the truth on 1/12, 5/8, 3/8, 4/8, 4/4 and 7/8 items (24/48 pooled). The permutation null on the same items expects ≤ 0.55 per model.

**New test: the full country subspace.** I removed all 57 country directions from the write and from the candidates (`a3`, measured):

| Model | Capital names truth (of 48) | Language names truth (of 56) |
|---|---|---|
| Qwen2.5-0.5B | 7 | 12 |
| Qwen3-0.6B-Base | 26 | 27 |
| Qwen2.5-1.5B | 22 | 16 |
| Qwen3-1.7B-Base | 34 | 30 |
| Qwen2.5-3B | 23 | 9 |
| Qwen3-4B-Base | 38 | 29 |

The permutation nulls expect ≤ 0.9 for capitals and ≤ 2.5 for languages.

**What this supports.**
- The carrier write holds capital- and language-predictive structure outside the span of every country unembedding direction, far above chance on all six models.
- This readout cannot tell whether that structure is "capital or language content" or country identity encoded in other token directions (demonyms, for example). Asserted.
- "Co-readouts are the country direction on six models" is **wrong as stated**. "The language co-readout is not separable from country identity by this test" would be correct.

**B6 (measured, reproduced).**
- Median language drop against capital drop: 0.38/0.84, 0.75/2.84, 0.75/3.28, 1.00/1.69, 0.44/0.47 and 0.31/0.62.
- The language drop is smaller on every model.
- The language argmax survives the L_c..L_c+2 MLP mean-ablation on 29/35, 37/40, 39/40, 39/41, 48/50 and 44/47 correct depth items.

These MLPs are not necessary for the language answer. **B6 holds.**

## 6. Hop budget

**B1 (holds with correction).**
- The real L_first values reproduce 128/128 (`a3`).
- The prereg's chance guard uses one fake country per item. With 20 draws per item, the medians over all items are:
  - Qwen3 d2: 0–1.5 against chance 5–5.5;
  - Qwen3 d3: 2–4 against chance 4–6;
  - Qwen3 d4: 3–8 against chance 5–6.
- So d2 and most d3 cells are measurable on Qwen3, but d4 is chance-dominated on 0.6B and 4B.
- Correct-only medians: 0.6B 1.5 / 2–3 / 6, 1.7B 0 / 1–4 / 1, 4B 0 / 3.5–4 / 6. None shows the predicted ≥ 2-layer monotone steps, and on 1.7B d4 sits below d3-lm.
- The d2 hits at L0 are lexical: the city token's own value already names the country.
- B1 is unsupported, not measured as true, and its sealed "flat" falsifier does not fire.

**B2 (holds).**
- 1.7B on the 1c captures: AUC 0.5995 reproduced. The bootstrap 95% CI is 0.47–0.72, so the "< 0.6 fires by 0.0005" verdict is a coin flip. What is robust is that the predicted ≥ 0.75 is missed.
- Scale panel AUCs reproduced:

| Model | AUC | 95% CI |
|---|---|---|
| Qwen2.5-0.5B | 0.48 | 0.29–0.67 |
| Qwen3-0.6B-Base | 0.66 | 0.47–0.81 |
| Qwen2.5-1.5B | 0.77 | 0.62–0.91 |
| Qwen3-1.7B-Base | 0.65 | 0.47–0.82 |
| Qwen2.5-3B | 0.61 | 0.35–0.84 |
| Qwen3-4B-Base | 0.64 | 0.41–0.83 |

- Only 1.5B clears 0.75.
- The secondary L_first_stable AUC is 0.59–0.80 (not judged).

**B3 (holds).**
- 1/64 is reproduced from the raw arms. On d4 only it is 1/47.
- The patch is not inert: it changes the argmax on 24/83 back-patched items.
- Correction: the "late" stratum is "country named by some cell by L_c". L_first is often 0–2, so this tests "present", not "late".
- On 0.5B the source layer is clamped to L23 = L_c + 2 (logged in `ea35403`).

**B4 (holds with correction).**
- The four falsifying cells reproduce: 0.5B city 2/7, 0.5B dist 2/8, 0.5B obsc 3/9, 0.6B dist 0/8. The prereg judges cells by n ≥ 8 items, so 0.5B city counts even with 7 wrong.
- The Spearman of 0.4596 comes from `combine.py`, which breaks ties arbitrarily (argsort of argsort). With tie-averaged ranks over the same 46 cells, ρ = 0.494. It is still below 0.5, but by a thin margin.

**Readable errors at d4–d6 (partial).**
- Reproduced: d4 11/11, 7/9, 6/9, 5/8; d5/d6 shares 0.625–0.95 in all 12 cells.
- The report omits the base rate. The same condition (the carrier ranks the true country > 3) also fires on correct items:
  - d5 correct: 6/7 (0.5B), 2/6, 4/12, 1/11, 3/10, 3/13;
  - d6 correct: 5/7, 2/5, 2/8, 1/6, 2/6, 2/7.
- The signal discriminates on 1.7B (d5: 12/17 of wrong items against 1/11 of correct ones) and moderately elsewhere. At 0.5B it is uninformative.
- S3 was written after the 0.5B and 0.6B d4 readable shares had been computed (see §9).

## 7. Hard prompts and native-name cities

**Dist (measured).**
- Wrong items: the true carrier names the true country on 5/8 (0.5B) and 7/8 (0.6B).
- Correct items: 3/4 and 4/4. The cell content does not separate correct from wrong, which supports an error after the cell.
- Reader mass favours the distractor on 0/8 and 4/8 wrong items, against 0/4 and 0/4 correct ones. The 0.6B contrast is suggestive (Fisher p ≈ 0.2) but below the sealed 0.60.
- The repair by the true cell is 1/8 in each.

**Native names (measured).**
- On wrong native-city items, the carrier names the true country on 15/18 (0.5B) and 8/10 (0.6B). S5's sealed falsifier fires.
- At 0.5B the carrier also names it on only 7/10 correct items, and that model's carrier names the country on only 15/28 correct base items. At 0.5B the readout itself is weak.

**"Native names are not harder (≥ English on 5/6)" is confounded.**
- Measured, `a2`: the English control includes historical exonyms: Breslau, Danzig, Pilsen, Carlsbad, Budweis, Oporto, Cracow, Saragossa, Kharkov, Elsinore, Archangel, Lyons and Marseilles. Asserted: several of these point to another country historically, for example Danzig and Breslau to Germany.
- Paired discordance (native-only against English-only correct): 6/5, 4/4, 4/1, 9/2, 3/4 and 3/3. Only 1.7B is lopsided (McNemar exact p ≈ 0.07).

**What the data support:**
- There is no evidence that native names are harder on this set.
- On the two smallest models, with 8–18 wrong items each, wrong hard and native-name items keep the country at the cell.

**Verdict: partial.**

## 8. MLP share against the tail

**Measured from the raw A2 arms.**
- At dose 1.0, the MLP share is 0.10, 0.61, 0.62, 0.60, 0.70 and 0.52 at tails 3, 8, 6, 8, 9 and 12.
- The accounting closes: median |rounding|/native is ≤ 0.7%, and the terms sum to 1.00.
- Apart from 0.5B, the shares sit at 0.52–0.70 and do not rise with the tail: 1.5B (tail 6) gives 0.62 and 4B (tail 12) gives 0.52.
- 2 of 5 sealed bands are hit.
- "Tracks the tail" therefore rests on 0.5B (direct 0.91), which is **overstated**.

**The 4B re-read.**
- 0.33 at dose 1.0 is reproduced. At dose 0.5 it is 0.11, so it is strongly dose-dependent.
- The label is mis-described. In this answer-row-only edit, no other row's K or V changes. "Re-read" is therefore the later layers' attention at the answer row changing because the answer row's query and its own K/V changed. It is not "later heads re-reading changed rows" (the `hop_edge` meaning string).

## 9. Custody

**Timeline (mrun `created_ts` / `started_ts` against commit times, measured).**

| Step | Commit / job | Time | OK |
|---|---|---|---|
| Hop-budget prereg; run | `17dd554` 04:15:27; job-6e384b3f8297 created 04:16:01 | before | ✓ |
| Grid prereg | `6f088a3` 04:23:37 | first capture created 04:24:11 (smoke), panel 04:25:13 | ✓ |
| Memo and panel hashes | sha256 `b2060733…` and `9c6c1cac…` match the files | | ✓ |
| Accuracy, 0.6B and 0.5B | `fb6f0fa` 04:33:58 | first `score.py` 04:47:52 | ✓, see gap A |
| Accuracy, 4B, 1.5B, 3B and 1.7B | `26440a3` 04:46:07 | first score 04:47:52 (0.5B and 0.6B only); others 05:05 onward | ✓ |
| Forecast seals before A1 | 0.5B and 0.6B `099cbb9` 04:35:47 → A1 04:35:57 and 04:38:43; 4B and 1.5B `5d91dac` 04:55:37 → A1 04:55:49 and 04:56:29; 3B shard b `03fb5c6` 04:58:33 → A1 started 05:04; 1.7B `47add0d` 05:05:10 → A1 05:05:11 | | ✓ all six |
| Supplement prereg | `6c13817` 04:57:41 → supplement captures 04:57:56 | | ✓, see gap B |
| Supplement accuracy | `291b406` 05:06:07 → supplement sweeps 05:06:07 onward | | ✓ |

**Gap A (undisclosed, impact nil).** A 3-item A1 smoke (`job-c613561d1941`) and a forecast smoke (`job-f69e6ccff597`) ran on 0.6B at 04:32:43. That was before the 0.6B accuracy addendum and before its forecast seal. Their logs print ids and seconds only, and whether the outputs were opened is unknown. `forecast.py` is unchanged since `f2e8f80`, the commit those smokes ran. The sealed forecasts are therefore deterministic code outputs, and seeing a carrier could not have altered them.

**Gap B (undisclosed, minor).** The supplement predictions S1–S5 were written at 04:57, after the 0.5B and 0.6B main-panel scores had been computed (04:47–04:48). S3 (readable errors at d5/d6) was predicted with the d4 shares of two models already known. S5 was falsified regardless.

**The admitted 0.5B carrier sighting (impact negligible).** It produced the logged clamp fix `ea35403` (04:40). That fix moves the 0.5B back-patch source from the invalid L25 to L23 and affects only B3 on 0.5B, which is falsified on every model. No carrier or scoring logic changed afterwards. The later `score.py` edits are the post hoc window column and the bounded derangement, and the post hoc column is labelled. The 0.5B carrier (L21) is off every sealed range, which is the opposite of a tuned result.

**Facts (measured).**
- `fact_pools_verified.json`: 237 drafts, 205 pass, 32 ambiguous (d5 16, d6 9, native 7), 0 wrong.
- The 97 unique verified items used in `panel_supp.json` are all "pass", and all 426 hop judgments are "ok".
- 88 hops rest on "lead-extract co-mention + direction checked" evidence.
- I spot-checked 8 random d5/d6 chains from my own knowledge, for example Nesbø → Oslo, Hašek → Prague, Steiner → Vienna and Nichols → Berlin. All are correct (asserted, not web-verified).

**3B kills.** The 3B A1 jobs marked `killed_vram` (043dd45197dd and 69e7d846edf6) each logged 64/64 items and TOTAL before the kill.

## 10. Code

**Tests (measured).** Run with `PYTHONPATH=<worktree>/src:<mdb>/src common exec python -m pytest`:
- `tests/test_analysis_circuit_map.py`: 9/9 pass, including `test_cell_readout_is_affine_under_a_v_bias`, which ran and was not skipped.
- `test_engine_prefill.py`, `test_analysis_sweep.py` and `test_prefill_followups.py`: 28/28 pass.

**Affine `cell_readout`.** The bias term `d · W_g b_V` is correct. The scale study's readouts do not call `cell_readout`. They apply `W_g` to the cached V, which already includes `b_V` in the HF cache, so no term is missing. My from-safetensors re-implementation matches the country ranks 128/128 on all six models.

**`hop_edge(neurons=False)`.** It is correct, and on real data the terms close to native within ≤ 0.7% median. Two notes:
- The equivalence test uses a cell consumer, not the `Margin` consumer used in A2. The code path is the same.
- Latent bug, pre-existing: inside the `neurons=True` branch, the name `neurons` is reassigned to a list. With `top_neurons=0` that list is empty, so every layer after the first would silently take the `neurons=False` path. It does not affect this study.

**Stepped span-row capture.** It is correct and tested. One minor numerics point: the A1 base cache comes from a full prefill, while the donor V values come from the stepped capture, so the two sides are rounded differently in bf16.

**`combine.py` Spearman.** It ranks without tie averaging. B4 should read ρ = 0.494, not 0.46.

## What is supported, and what is not

**Supported (measured):**
- One dominant carrier cell per Qwen3 model, which holds out of sample.
- KV6's within-layer dominance.
- The carrier layer at 0.67–0.71 of depth on Qwen3.
- The sealed write-window failure.
- First-order captures about 30–70% of the one-row effect.
- Accounting closure of the hop-2 split.
- An MLP share of 0.5–0.7 at 5 of 6 sizes.
- B3 and B6 as measured.
- Deep-chain errors read at the cell on 1.7B and the larger models.

**Partial:**
- Readable errors as a diagnostic: they need the base rate, and they fail at 0.5B.
- Errors downstream of the cell for dist and native names: only 2 models, with n = 8–18.
- The fraction rule: it rests on 4B alone.

**Not supported or wrong:**
- A KV6 role conserved across sizes.
- The replica as a forecast.
- Co-readouts being only the country direction, and "hop 3 is not in the cell".
- "Native ≥ English".
- The MLP share tracking the tail.

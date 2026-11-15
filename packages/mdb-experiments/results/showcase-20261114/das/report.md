# 11b. DAS at the exp7 carrier cell vs the symbol-table readout subspace (MEASURED, not audited)

Qwen3-1.7B-Base bf16 eager, Beast via mrun, single seed 20261114. Cell: L20 KV6 value (head_dim 128) at the frozen R_argmax carrier row. Test: exp7's 64 Run 1a pairs (recipient, different-bridge donor), consumer M[answer, donor answer]. Every arm is scored through mdb's stepped engine exactly as exp7: prefill = carrier row, checkpoint at t<crow>.L20.h, set v[kv6] := v + Q Q^T (v_donor - v), run to logits.

## Custody

| step | job / commit | note |
|---|---|---|
| smoke | job-e2c0ece8c85f (payload lacked b1 files), job-7ad594cb5b00 -> auto-retry job-c4c4834e3110 (mrun RAM clamp) | whole/SAE margins and carrier rows reproduced exp7 bitwise on 2 pairs |
| prereg sealed | `52eed79` (13:06:49) | `prereg.json`, timestamp from `date`; precedes every scored job |
| admission | job-0c288755878c | 87/130 training items admitted (native argmax = capital), 38 countries, none among the 28 test countries; test carrier rows = exp7 on 64/64 |
| DAS training (+ SC1 check) | job-14bfa70f5ddb | 6 ranks x 300 steps (~50 s each); 7,326 ordered training pairs |
| scoring, 3 shards | job-70dfc80ed3c4, job-f9da01e895ef, job-5ce9e8d2f078 | 32 arms per pair; each record carries the das.py and prereg sha256 |
| exploratory addendum (unsealed) | `52926a6`; job-af6bb2400d89 (train), job-4cbaed537d23 (score); symtr arm added later, job-b28d1ef34615 | written after shard 0 had been seen; in-sample arms labelled |

Payloads are `git archive HEAD` (library and drivers); the job note carries the `das.py` sha256 (audit item 4). Floor: `2 * verdict.spacing(max |logit|, bf16)` (asserted equal to `common.floor_of` on every record). Flips counted at margin 1 (exp7 rule: M0 > floor, M < -floor) and margin 4 (the audit's stable-verdict margin). Every arm record stores `effect_floors` = M / floor.

## IIA vs k (held-out pairs, held-out countries)

Cells: flips at margin 1 / margin 4 (median fraction of the whole-cell margin change moved). Whole cell: 44/64 (margin 1), 38/64 (margin 4). SAE pushforward (exp7's per-pair subspace, median rank 4): 4 / 1 (0.34). All 64 pairs are eligible at both margins.

| arm | k=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| DAS (pyvene, trained on 38 other countries) | 0 / 0 (0.02) | 0 / 0 (0.05) | 0 / 0 (0.10) | 1 / 1 (0.21) | 4 / 2 (0.40) | 16 / 9 (0.60) |
| symbol-table country readout span (69 country tokens) | 0 / 0 (0.02) | 0 / 0 (0.05) | 0 / 0 (0.19) | 2 / 2 (0.33) | 15 / 12 (0.67) | 35 / 31 (0.89) |
| symbol-table capital readout span (70 tokens) | 0 / 0 (0.02) | 0 / 0 (0.06) | 0 / 0 (0.22) | 2 / 1 (0.32) | 16 / 11 (0.66) | 31 / 28 (0.85) |
| mean-difference PCA (training countries) | 0 / 0 (0.01) | 0 / 0 (0.04) | 0 / 0 (0.10) | 0 / 0 (0.21) | 3 / 2 (0.35) | 14 / 8 (0.57) |
| random | 0 / 0 (0.00) | 0 / 0 (0.01) | 0 / 0 (0.02) | 0 / 0 (0.03) | 0 / 0 (0.07) | 0 / 0 (0.16) |
| *exploratory:* readout span of the 38 training-country tokens only | 0 / 0 (0.01) | 0 / 0 (0.05) | 0 / 0 (0.11) | 0 / 0 (0.27) | 6 / 4 (0.43) | 16 / 12 (0.64) |

Fraction of whole-cell IIA at margin 1: DAS 0, 0, 0, 0.02, 0.09, 0.36; country readout span 0, 0, 0, 0.05, 0.34, 0.80; capital span 0, 0, 0, 0.05, 0.36, 0.70; mean-difference 0, 0, 0, 0, 0.07, 0.32; random 0; SAE 0.09. At margin 4: DAS32 9/38 = 0.24, country span32 31/38 = 0.82.

DAS fits its training distribution (in-sample flips with M < 0 on 256 training pairs: k=1 8, 2 19, 4 45, 8 128, 16 202, 32 215; final loss 8.7 to 2.0) but transfers little to unseen countries. pyvene's batched forward and mdb's stepped arms give the same DAS flip verdict on 64/64 pairs for k <= 16 and 62/64 at k = 32 (max |dM| 0.375-0.625, the kernel-path size exp7 measured). The trained rotations are orthonormal to 4e-7 and nested: the largest principal angle between DAS_{k/2} and DAS_k is 6-12 degrees.

## Principal angles (DAS_k vs the other subspaces)

Smallest / median principal angle in degrees, and the normalised projection kernel PK = sum cos^2 / sqrt(k_a k_b). PK for random subspaces is about k/128.

| k | country readout span | capital span | mean-difference | random | SAE (per pair: median smallest angle; median PK) |
|---|---|---|---|---|---|
| 1 | 40.4 (PK 0.58) | 69.2 (0.13) | 75.2 (0.07) | 87.4 (0.00) | 79.4; 0.02 |
| 4 | 23.8 / 36.3 (0.54) | 27.4 / 46.5 (0.44) | 15.6 / 26.2 (0.73) | 70.8 / 76.5 (0.06) | 63.7; 0.08 |
| 8 | 17.3 / 31.8 (0.63) | 22.5 / 42.0 (0.51) | 11.5 / 21.6 (0.70) | 65.7 / 75.6 (0.07) | 53.0; 0.12 |
| 16 | 14.5 / 27.8 (0.67) | 14.4 / 35.2 (0.58) | 8.8 / 19.3 (0.82) | 52.0 / 71.9 (0.13) | 39.6; 0.16 |
| 32 | 8.7 / 30.8 (0.61) | 9.6 / 38.8 (0.55) | 5.7 / 21.2 (0.73) | 33.8 / 63.7 (0.24) | 31.1; 0.17 |

DAS's top direction lies 40 degrees from the readout span's top direction (PK 0.58; random 0.002). The two subspaces overlap at 2.5-10 times chance at every k. DAS is closest to the supervised mean-difference subspace (both were fitted on the same 38 countries). The SAE subspaces are close to orthogonal to DAS at small k.

## Prereg verdicts

- D1 FAILS: DAS is non-decreasing in k, but no k <= 16 reaches 0.5 of whole (best 4/64 = 0.09 at k=16). The falsifier (k=32 below 0.25 of whole) does not fire at margin 1 (16/64 = 0.36); at margin 4 it would (9/38 = 0.24).
- D2 FAILS, in the opposite direction: the country readout span is above DAS at k = 8, 16, 32 (2 vs 1, 15 vs 4, 35 vs 16) and equal at k <= 4. Its second clause holds (<= 2/64 for k <= 8).
- D3 FAILS: mean-difference is at or below DAS, not between (k=8: 0 vs 1; k=16: 3 vs 4).
- D4 HOLDS: random flips 0/64 at every k <= 32.
- D5 FAILS: the smallest angle between DAS_8 and the readout span_8 is 17.3 degrees (bar > 30). The subspaces overlap more than predicted.
- D6 HOLDS: whole-cell 44/64 and SAE 4/64 reproduce exp7 exactly.

## Exploratory addendum (unsealed; `prereg_explore.json`, `summary_explore.json`)

The question: is held-out DAS failing on rank or on generalisation? `dasin` is pyvene DAS with the prereg's hyperparameters, trained on all 5,164 ordered pairs of the 74 test-country items, with the 64 exp7 pairs excluded. The scored cells are seen in training (in other pairings), so this is an in-sample existence measure, not a held-out result. `meanin` is the PCA of the per-answer class means of the test-country cells (also in-sample).

| arm | k=4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| dasin (in-sample DAS) | 2 / 0 (0.28) | 12 / 10 (0.59) | 40 / 36 (0.94) | 47 / 38 (1.02) | 47 / 40 (1.03) |
| meanin (in-sample class means) | 1 / 1 (0.18) | 10 / 7 (0.44) | 30 / 22 (0.84) | 42 / 38 (0.98) | 42 / 38 (0.98) |
| country readout span | | | | | 43 / 34 (0.95) |
| capital readout span | | | | | 36 / 33 (0.92) |
| random | | | | | 1 / 0 (0.38) |

- Rank is not the problem. A 16-dimensional subspace carries 0.91 of the whole-cell IIA (40/64; 36/38 at margin 4), and 32 dimensions match the whole cell (47/64; the subspace swap beats the whole-cell swap on 3 pairs). Plain class means (at most 27 directions for 28 answers) reach 42/64 at k = 32.
- Generalisation is the problem. The same 16 dimensions fitted on other countries give 4/64. The country code is not a shared low-rank variable; each country adds directions of its own.
- In-sample DAS vs the readout span: smallest angle 15.8 degrees and PK 0.62 at k=16 (random 0.13); 9.6 degrees and PK 0.61 at k=32. In-sample DAS vs held-out DAS: PK 0.46 at k=16. At k=64 the smallest angle is not informative: two generic 64-dim subspaces of 128 dimensions meet near 0 degrees, and random64 also gives PK 0.49.
- The readout span's lead over held-out DAS comes from its vocabulary. Built only from the 38 training countries' tokens, it gives 16/64 at k=32 and 6/64 at k=16, the same as DAS trained on those countries (16 and 4). The full span is training-free, but its declared country set includes the test countries' tokens (the panel's country list). A readout direction of a country token, mapped into the cell through W_O, is a usable handle on that country's code; it does not predict the code of a country whose token is absent.

## Answer to the owner's question

Does the symbol table give better insight than the SAE subspace in exp 7? Yes, measured, with one condition. Take the symbol table's own content readout (the country-token unembeddings through the final norm and the KV group's W_O, mapped into the cell) and use its top principal directions as a fixed subspace. That subspace transfers 35/64 pairs at k=32 (0.80 of the whole cell; 31/38 at the 4-floor margin) and 43/64 at k=64. The exp7 SAE pushforward transfers 4/64. Both are training-free, but the SAE subspace stays small (median rank 4, the few active country features), while the readout span grows with k and keeps working. At rank 4 or below, neither transfers the country: the readout span 0/64, SAE 4/64 (moved 0.19 vs 0.34 of the margin). The condition: the readout span works because its declared vocabulary contains the countries being moved. From other countries' tokens it does no better than DAS (16/64 at k=32).

Can we map the subspace that carries the country code? Yes, in-sample. Over the 28 test countries, the code sits in about 16-32 of the cell's 128 dimensions: in-sample DAS reaches 40/64 at k=16 and the full whole-cell rate at k=32 (exp7 pairs held out). That subspace is spanned mostly by per-country directions (class means do almost as well) and overlaps the symbol table's readout span at 5 times chance. It is not one low-rank "country" variable. A subspace learnt on 38 other countries moves only 4/64 at k=16 and 16/64 at k=32. The symbol table's readout gives a training-free, per-country handle on it, provided the country's token is in the declared set.

## Into mdb

`mdb.analysis.subspace.SubspaceBinding` (commits c1c15a7, 165b47b) is a sub-cell binding over one value cell. It holds an orthonormal basis, a basis authority (`fitted` for DAS and class means, `derived-exact` for readout spans, `supplied`) and provenance.
- It belongs to the same family as ComponentBinding: `decompose(run)` returns a ComponentBinding with parts `subspace` and `complement`, which `component_table` takes unchanged.
- Constructors: `readout_subspace` (the symbol table's content readout mapped into the cell), `class_mean_subspace`, and `from_pyvene` (a trained `LowRankRotatedSpaceIntervention`).
- Measurement: `interchange` (an mdb branch; it also reports the effect in floors) and `iia` (margins 1 and 4).
- Reporting: `symbols.table(subspaces=[sb.bound(row)])` adds a sealed `subspaces` section. It gives the native cell's mass in the subspace, the direct consumer share of the in-subspace part, and any attached measured IIA. `Binding.subspace(...)` declares one.
- Tests: `tests/test_analysis_subspace.py`, 12 passed on tiny Qwen2 bf16 and Qwen3 fp32.

This run's subspaces are stored as mdb objects with their IIA attached in `subspaces.json` (43 bindings, written by `bind_results.py`). They reload through `SubspaceBinding.from_json` with a digest check. The symtr bases were computed inside job-b28d1ef34615 and not saved, so they have no binding.

## Not run / limits

- Single seed.
- No same-bridge null pairs.
- DAS hyperparameters were fixed in the prereg and not tuned (300 steps, lr 1e-2).
- TransformerLens 4.4 `fit_das` was not used as a second DAS implementation. It aligns `[batch, pos, d_model]` activations, and the cell is one KV head of a 4-D `hook_v`.
- The SAE arm is per pair (its rank is set by the active features), so it gives one point, not a k sweep.
- The exploratory arms were chosen after the scored shard 0 had been seen.
- The readout spans use the panel's whole country vocabulary, including the test countries.

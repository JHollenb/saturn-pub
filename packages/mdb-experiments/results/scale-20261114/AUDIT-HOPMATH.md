# Audit: hop-math arms (entry 10, prereg_hopmath.json @ 736e84c)

Independent auditor, 2026-11-14 12:33 PST. Laptop only: no GPU jobs, no mrun submissions, no result files edited. Every count below was recomputed from the raw per-item records (`out/hm-*.json`, `out/score-*.json`, `out/supscore-*.json`, `out/rt-*.json`, capture JSONs). I did not use the builder's `hma-*.json` summaries or tracker prose to get them. "Recomputed" means I derived it from raw outputs. "Code-read" means I read the code but did not execute it.

## Verdict table

| # | Claim | Verdict | One-line reason |
|---|---|---|---|
| 1a | §3 readable repair 5/23, 12/17, 2/11; band = cell | CONFIRMED (recomputed) | Strata, exclusions and counts reproduce exactly. Band differs from cell on one item (1.7B Colosseum). |
| 1b | Readable rate matches donor-moves within ±0.15 | PARTLY | Holds against the sealed /48 numbers. Against a correct-items footing it fails for 0.6B (native 13/30 or replica 15/28) and for 4B (replica 16/42). |
| 1c | "≤ 0.1 on right-operand" falsified (5/9, 6/10, 3/8) | CONFIRMED, robust | Holds on rank-1 items alone (3/4, 3/4, 3/6). One 1.7B repair is a replica-null artefact, so the corrected 1.7B count is 5/9. |
| 2a | §5 table numbers (move ratio, shares, arm 3, arm 4) | CONFIRMED (recomputed) | All reproduce. Pooled shares agree with the medians. |
| 2b | "The memo's falsifier does not fire" | NOT CONFIRMED as worded | The direct-share clause does not fire (lang ≥ cap on all three, but every bootstrap CI includes 0). The arm-2 clause does fire on 0.6B (o26, 0.11 vs 0.04), and the builder's own table says so. The point estimate is a coin flip: bootstrap P(rule fires) = 0.44. |
| 2c | Reading: "one country code drives both; third hop not a separate lookup; three-hop count not honest" | PARTLY: first half warranted, second half over-reach | See §2 below. |
| 3 | §6 negative control | CONFIRMED | Merge is correct, prompts and scoring are identical, and the forward path is unchanged across revisions. The flip counts reproduce. Add context: 14/128 items flip in total, net −4. |
| 4 | Freeze misalignment changes no B1 or scale number | CONFIRMED (code-read + B1 map) | Carrier is always declared first and always binds. `summarize` keeps only common paths. The scale scripts use only `cm.group_writers`. |
| 5 | Retrace: wrong switch more on Qwen3 1.7B/4B, no Qwen2.5 gap, no size trend | PARTLY | 1.7B is significant (perm p = 0.001, holds within block). 4B is not (p = 0.15) and is mostly block composition. Near-tie and margin confounds cannot be tested because `retrace.py` saves no margins. |
| 6 | Ladders close ≤ 4e-5; arms use the fp32 replica consistently | CONFIRMED | Max \|closure gap\| is 3.9e-5 over all ladders. Every arm goes through one fp32 `forward()`. Null argmax matches 32/32, 26/27, 18/19 wrong items. |

## 1. §3 operand repair

Code (`hopmath.py:122-158`):
- The pool is the wrong d3-lm, d3-per and d4-work items plus the wrong supplement d5 items, as preregistered.
- Stratum: readable when `country_rank is None or > 3`. This matches the prereg and agrees with the score files' `readable_error` flag on every item (0 disagreements, recomputed).
- Donor: `cells` is built only from CORRECT main-panel items (`hopmath.py:112-119`), keyed by `country_token`. The item being repaired is always wrong, so it can never be in its own mean. `mean_cell`'s `exclude` argument is dead code, but no exclusion is needed. I asserted for every pooled item that it is not in its country's donor list.
- Same country: the key is the item's own `country_token` (the true bridge country).

Recomputed (cell / band):

| Model | Readable | Right-operand | Excluded |
|---|---|---|---|
| 0.6B | 5/23 / 5/23 | 5/9 / 5/9 | 6 |
| 1.7B | 12/17 / 12/17 | 6/10 / 7/10 | 5 |
| 4B | 2/11 / 2/11 | 3/8 / 3/8 | 5 |

All match `hm_report.md`.

**Reference rate (correction 1).** The sealed text says "donor-moves rate on correct items" but quotes /48 numbers. I recomputed from `score-*.json` `closure.donor_moves`:
- The 17/48 count is over all 48 base items, wrong items included.
- On correct items only, native: 0.6B 13/30 = 0.43, 1.7B 28/41 = 0.68, 4B 13/42 = 0.31.
- Replica, on correct items with a donor mean: 0.54, 0.71, 0.38.

Readable rates are 0.22, 0.71, 0.18. Gap to each reference:

| Model | vs sealed /48 | vs native correct-only | vs replica |
|---|---|---|---|
| 0.6B | 0.14 | **0.22** | **0.32** |
| 1.7B | 0.08 | 0.02 | 0.00 |
| 4B | 0.13 | 0.13 | **0.20** |

So "holds on all three" depends on using the /48 references. On a correct-items footing, which is what the sealed sentence says, 0.6B fails on both bases. The builder discloses the replica comparison (tracker line 967) but still headlines "holds on all three". Correct statement: it holds clearly on 1.7B, holds on 4B against the native rate only, and does not hold on 0.6B on a like-for-like footing. Small n (23, 11) makes ±0.15 a loose test anyway.

**Right-operand falsification is robust.**
- On rank-1-only items the cell still repairs 3/4, 3/4 and 3/6.
- 1.7B `d5:d5-081`: the replica null argmax differs from native and `dM_cell` = 0.0, meaning the replica null already had the answer on top. This is a free "repair". Excluding it gives 5/9.
- 4B `Palace Walk` also has a null mismatch, but its gap is 0.34, so it is a genuine flip in the replica.
- Near-ties: repairs with a native deficit < 0.25 logit are 1/10, 3/18 and 1/5. Most repairs are real flips of 0.3–4.7 logits.

**Band.** The band adds nothing: one item differs across all three models (1.7B Colosseum, band only). Confirmed.

## 2. §5 hop 3

**Recomputed numbers.** All from raw records, and all match the table:
- Move rates: language 20/40, 24/41, 25/47 against capital 15/28, 29/41, 16/42. Ratios are 0.93, 0.83, 1.40.
- Direct share, language / capital: 0.458/0.360, 0.465/0.396, 0.185/0.165.
- MLP share: 0.45/0.62, 0.39/0.57, 0.38/0.53.
- Arm 3: 0.85/0.83, 1.04/1.00, 0.90/0.90.
- Arm 4 complement names: 32/56, 43/56, 46/56.
- Arm 4 complement move: 0.64, 0.39, 0.25. Country-part move: 0.04, 0.05, 0.04.
- Pooled (ratio-of-sums) shares agree with the medians within 0.03, so the medians are not an artefact.
- The arm formulas in `hopmath.py:166-236` match `prereg_hopmath.json` `operational.s5_arm1..4`. This includes the arm-3 projection, the arm-4 null projector, and the arm-2 leave-one-out mean.

**Falsifier wording (correction 2).** Tracker line 986 says the memo's falsifier "does not fire". The memo's falsifier has two clauses (ADVISOR §5.3 line 198):

1. Language direct share below the capital's. It does not fire, but it is weak evidence. Bootstrap 95% CIs of the lang − cap median difference: 0.6B [−0.003, 0.19], 1.7B [−0.02, 0.13], 4B [−0.01, 0.05]. All three include 0, and 4B is effectively a tie.
2. A band with a 2× language asymmetry. It fires on 0.6B at o26 (lang 0.110, cap 0.044), as the builder's own table row says. Bootstrap P(rule fires) at o26 is 0.44. No other layer on any model exceeds 0.03.

Correct statement: by the sealed rule the falsifier fires on 0.6B (o26, borderline, single late attention layer, at about chance under resampling) and does not fire on 1.7B or 4B.

**Reading (correction 3).**

Warranted:
- One country code at the carrier cell drives both answers. Arm 3 keeps ~0.85–1.04 of both margins. The country part alone moves the margin 0.04–0.06, while the complement moves it 0.25–0.64.
- No language-specific post-carrier locus is found on 1.7B/4B.

Not warranted: "the third hop is not a separate lookup / the three-hop count is not honest". The memo's "lexical" hypothesis predicted three signatures, and all three missed in the lookup-like direction:
- Direct share 0.46/0.46/0.19, not 0.60–0.75.
- MLP amplitude about equal to the capital's, not half.
- Complement move > 20%.

The language answer uses the same post-carrier MLP band (0.38–0.45 of the delta, same per-layer sign pattern, similar amplitude) that the capital uses (0.53–0.62). The capital step is counted as a hop (hop 2). So the data say country → language is computed by the same kind of downstream machinery as country → capital: not at a separate location, but not lexical either. The defensible conclusion is that hop 3 shares hop 2's site and input code, and its MLP share is ~0.7× the capital's. Whether that counts as a hop is definitional and is not settled by these arms. A non-lexical final attribute, as the memo suggested, is still the right next test.

Arm-4 caveat. The "names" counts (32–46/56) also hold for the capital (28–46/48). Only one cell is edited while the rest of the row's KV still carries the country. So the names counts say little about cell content, and the margin move is the informative statistic.

## 3. §6 negative control

**Merge (`merge_lora.py`, job-7e6f082eb493 log).**
- I loaded the local `bind_adapter.pt`: 168 keys = 24 layers × {q, k, v, o, gate, up, down}.
- `scale` is 2.0 for every key, which equals α16/r8.
- A is (8, 896) and B is (896, 8).
- The trainer's forward (`grok-mechanism-full-proof/.../qwen-suite/lora_train.py:66`) is `base(x) + (x Aᵀ) Bᵀ · scale`, so ΔW = scale · B @ A, which is what the merge computes.
- Adapter meta base is `qwen2.5-0.5b`, and the merge used `/mnt/big/llm-models/Qwen2.5-0.5B`.

**Minor prereg deviation.** The prereg says "saved bf16", but the log says `saved ... torch.float32`: `model.config.torch_dtype` became fp32 after loading in fp32. This is immaterial, because `capture.py`, `sweep.py`, `arms.py` and `score.py` all load with `_load(..., "bf16", ...)`.

**Identical prompts and scoring.**
- Both captures cover the same 128 ids, with identical `tokens`, `cache_rows`, layer/kv/head metadata and shapes. Only `model` differs.
- `correct` in both score files equals `capture argmax == panel answer` on all 128 items (recomputed).
- The base run is capture revision 477afe3 and the NC run is 824166c. `capture.py` changed only through env-var hooks (PANEL, NOSTEP), and neither was set for NC. `engine/models.py` is unchanged. The `engine/run.py` diff only adds the `Run.feed` / supplied-token path, which is inactive without feed (code-read, not bit-verified).

**Recomputed accuracy, base → merged.**

| Family | Base | Merged |
|---|---|---|
| currency | 7 | 7 |
| person | 9 | 10 |
| landmark | 7 | 6 |
| city | 5 | 2 |
| d2 | 14 | 14 |
| d3-lm | 9 | 8 |
| d3-per | 9 | 10 |
| d4-work | 3 | 6 |
| dist-city | 4 | 1 |
| obsc-city | 3 | 2 |

- d3 pooled is 18/28 → 18/28, but with one gain (Niels Bohr) and one loss (Khan el-Khalili).
- d4 gains Blue Danube, Great Dictator and Candide, with 0 losses. One-sided sign p is 0.125 (two-sided 0.25).
- Carrier L21 KV0 holds on 20/28 (base) and 19/25 (merged).
- MLP share at dose 1.0 is 0.097 → 0.141.

Context the write-up should add: the adapter flips 14/128 items (5 gains, 9 losses, net −4 overall). A random 3-of-14 flip in d4 is within that churn rate (~11%/item). All three d4 flips going up is mildly suggestive, so "unknown, not negative" is the right label. The sealed ±1/14 for d4 fails on the point estimate, as reported.

## 4. Freeze misalignment

Code-read on `mdb-twohop`:
- `circuit_map.bind` declares non-`rest` slots first, so carrier comes before desc.
- `symbols.table` builds `bindings = users + token rows`, with `users = list(declared(run).values())` in insertion order.
- `b1.py:96` indexes `rep["slots"]["carrier"][0]` on every item, so a missing carrier binding would have crashed the job. The map has `n_items` = 64, so the carrier bound on all 64.
- `summary.summarize` (`summary.py:158`) keeps only paths common to all reports.
- `b1-map.json` contains only `table.bindings.0.*` paths and `partial` = 13.
- Position 0 is therefore always the carrier, and no frozen number is misaligned.
- Scale study: `score.py` and `hopbudget.py` import `circuit_map` but call only `cm.group_writers` (a weight helper). `scale_smoke.py` calls `item_report` but is a smoke run that never freezes. Wording nit: the tracker's "never calls bind/item_report/freeze" is true of every script behind a reported number.

CONFIRMED.

## 5. Exploratory retrace

The counts in `rt_summary.md` reproduce from `rt-*.json`. I added two-sided permutation tests on the mean difference (wrong − correct), plus a within-block permutation:

| Model | Carrier switches after first true | Answer switches ≥ L_c |
|---|---|---|
| Qwen2.5-0.5B | +2.54, p = .026 | −0.05, p = .61 |
| Qwen3-0.6B | +3.17, p = .015 | −0.14, p = .61 |
| Qwen2.5-1.5B | +0.12, p = .95 | 0.00, p = 1.0 |
| Qwen3-1.7B | +1.10, p = .60 | **+0.88, p = .001** (block-strat .001) |
| Qwen2.5-3B | +0.01, p = 1.0 | +0.19, p = .68 |
| Qwen3-4B | −1.98, p = .47 | +0.66, p = .15 (block-strat .21) |

- **1.7B gap is real at this n.** It is present within depth items (2.3 vs 1.0).
- **4B gap is not distinguishable from noise** (15 wrong items). It is mostly block composition: depth items switch more in both groups (2.7 correct vs 2.9 wrong), and 9/15 wrong items are depth. The within-block differences are +0.6 (base, n_w = 6) and +0.2 (depth). So "wrong items switch more on Qwen3 1.7B/4B" should read "on 1.7B".
- **Near-tie and margin confounds cannot be tested from saved outputs.**
  - `retrace.py` records only argmax sequences.
  - Its docstring promises a margin-filtered count ("switches among layers where the cell names a country with margin > 0 over the runner-up only") that the code does not compute.
  - Carrier switches at ~every layer step (19.6–26.4 over 20–35 steps) are what near-tie or noise flips look like. The builder already concedes whole-depth carrier switches measure nothing.
  - Wrong items have lower final margins by definition, so more late answer-lens flips and later settling are expected under a pure low-margin null.
  - Required next step: save the per-layer top-2 margin and compare correct and wrong items matched on final margin, or count only flips with margin above a floor. Until then the 1.7B gap is "measured, mechanism unknown".
- "No size trend": confirmed, there is no monotone pattern.

## 6. Replica fidelity

- Max \|closure_gap\| over all arm-1 ladders (both doses, all items, three models) is 3.9e-5, from 1.7B depth. Per model: 0.6B 1.6e-5, 1.7B 3.9e-5, 4B 3.5e-5. So "≤ 4e-5" is CONFIRMED.
- `hopmath.py` loads the model in fp32. All arms (oprep cell/band, donor-moves reference, arms 1–4) run through the same `forward()`, which casts the bf16-captured K/V to fp32.
- Arm 2 substitutes means of captured native port outputs, which are bf16-run activations stored as f32. This mixed provenance matches the prereg wording.
- Null argmax match on non-excluded wrong items is 32/32, 26/27, 18/19, as stated. The two mismatches are discussed in §1.
- The 4B shards (s0–s2) and 1.7B shards (s0–s1) have no duplicate ids and give complete 48/56 coverage.

## Three most important corrections

1. **§5 reading over-reaches.** Every lexical signature the memo predicted missed in the lookup-like direction: direct 0.46/0.46/0.19 not 0.60–0.75, MLP amplitude ≈ capital not half, complement move 0.25–0.64 not < 0.2. Language uses the same post-carrier MLP band as the capital at ~0.7× the share. Supported: one country code feeds both, and there is no separate language locus. Not supported: "the third hop is not a lookup / the three-hop count is not honest".
2. **"The memo's falsifier does not fire" is wrong as worded.** The arm-2 clause fires on 0.6B by the sealed rule (o26: 0.11 vs 0.04, bootstrap fire probability 0.44). The direct-share clause does not fire, but all three CIs include zero. State both clauses.
3. **§3 "holds on all three" depends on the reference.** Against the correct-items footing the sealed sentence names, 0.6B fails (0.22 vs native 0.43 or replica 0.54) and 4B fails against the replica (0.18 vs 0.38). Also: the 1.7B right-operand count should be 5/9 (d5-081 is a replica-null artefact), and the retrace "4B wrong items switch more" is not significant (p = 0.15, mostly block composition).

Positives that stand: right-operand items are repaired at 0.38–0.6 even on rank-1 items, so the operand-strength reading of deep-chain failure is strong. The band adds nothing. The §6 merge and the comparison between models are clean. The freeze issue touches no number. Replica closure is at 4e-5.

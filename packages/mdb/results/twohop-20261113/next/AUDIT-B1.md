# Audit B1: frozen symbol map on the depth capture

Auditor: independent Opus agent, 2026-11-14. Scope: the branch mdb-twohop, commits a6a3863..17f9afe, the files in `next/`, and tracker entry 1, block "B1". Every number below was recomputed by my own scripts in `next/audit-b1/`. These scripts do not import mdb or the builder's scripts. Weights were read with safetensors.

- `a_local.py` → `a_local.json`: S5, the hop shares and the E1 closure, from `arms-b1.jsonl` and `b1-forecast.json`. Runs locally.
- `a_beast.py` → `a_beast.json`: S1–S4, the nulls and S6, from the 1c captures and the B1 arm tensors. This was a CPU job on Beast, with no forwards. It ran as job-09bb2885eabd (4 s) and again as job-a7af065b60e0 to dump per-item data; the two summaries are identical.
- `a_lexical.py` → `a_lexical.json`: the weight-only lexical null, job-3a4daafe575b (1 s, unembedding rows only).

No GPU job was run. All three jobs ran at priority 70.

## Verdict table

| Claim | Recomputed | Verdict |
|---|---|---|
| Custody | see §1 | **holds** (minor notes) |
| S1: L20 KV6 is the best cell on 61/105; best at the carrier on 48/105 | 61/105, 48/105, row 81/105 | **holds** (fails the 0.70 bar, as reported) |
| S1: "real gain" where carrier ≠ last row, 27/29 vs 13/29 | 27/29 vs 13/29 | **holds with correction**: preregistered as the credit rule, not post hoc; the null is weak (see §3) |
| S2: country at the carrier 99/105 | 99/105 | **holds** |
| S2: latent capital rank ≤ 3 on 103/105 | 103/105 | **overstated**: the pure country direction already ranks the capital #1 on 105/105 |
| S2: creator rank ≤ 3, d4 8/12, d3-per 20/32 | 8/12, 20/32 | **holds with correction**: the lexical null gives 5/12 and 15/32; partial positive |
| S2 ladder 0/12, "uninformative" | 0/12; chance about 0.08 per item | **holds** |
| S3: the carrier names the language on 99/132; the falsifier fires | 99/132 | **holds** (numbers) |
| S3 post hoc: "France/French similarity, not hop 3"; 92/132 | 92/132 (84 strict + 8 ties) | **overstated**: the similarity explains why the falsifier fired, but "not hop 3" is not shown |
| S3: MLP names the language at L20–22 on 42/73/70 of 132 | 42, 73, 70 | **holds** |
| S4: 22/27 vs 6/83 | 22/27 vs 6/83; holds within each family | **holds** |
| S5: sign 153/153, size 0.52× | 153/153, 0.519× | **holds numerically; overstated as evidence**: the always-negative null also scores 153/153 |
| S6: E1 names the donor on 23/102; moved 89/102; control 8/29 | 23/102; 88/102 (see §2); 8/29 | **holds** |
| S6: "E1 is destructive, not a content transfer" | donor rank 21 → 7.5; pairwise flip 20/29 vs control 2/29 | **overstated**: mostly destructive, with a measurable donor-directed part |
| S6: hop-3 MLP share 0.40 | 0.402 (n = 152); only 25/152 ≥ 0.5 | **holds** (fails the bar) |
| S6: closure gap 0.13 | 0.129 absolute = 0.08% of the change | **holds with correction**: closure is an identity by construction; the gap checks the V reconstruction, not the decomposition |
| Falsifiers F_map / F_stack / F_edge did not fire | same | **holds as written**; F_stack's passing branch is partly lexical; F_edge is uninformative (the builder says so) |

## 1. Custody (measured from git, the mrun job records and the logs I fetched with `mrun attach`)

All times are 2026-11-14, local time.

| Event | Time |
|---|---|
| a6a3863: prereg + `b1_sets.json` (sha256 63181877… matches the prereg) | 03:04:30 |
| First B1 job: freeze smoke job-6cc154578805 created | 03:06:10 |
| Depth smoke job-d6fb5ca20814 (3 items; log has ids and seconds only) | 03:06:12–03:06:56 |
| c8ac72c b1.py / donors; b9af1ea forecast script | 03:06:03 / 03:07:33 |
| Freeze job-5fdafb818bbd (rev b9af1ea, base 64, no 1c); log digest ef1d5b63… | 03:07:35–03:09:19 |
| 8c89d5b map committed; I recomputed its digest from the file and it matches | 03:09:42 |
| Depth shards 0d68/d38d/c4fa/dc54 (rev b9af1ea; logs have ids and seconds only) | submitted 03:07:35–42, ran 03:09:20–03:27:21 |
| Forecast attempts: db2a (NameError), 0b7d/8b85/ab76 (killed_ram); 2ad42e3 adds imports only | 03:09–03:11 |
| Forecast job-e8b370ee7692 (rev 2ad42e3; reads the captures and donors only) | 03:27:50–03:29:14 |
| cfca8a3: forecast + `b1_score.py` committed | 03:29:26 |
| First scorer job-b09d73395ec2 (rev cfca8a3, killed) → auto-retry a93f7cf65e9e | 03:29:38 → 03:31:15 |
| 6414981 scorer memory fix; job-b1c633a941b6 | 03:30:25 → 03:31:44 |
| da88f76 arms + score; 17f9afe post hoc (job-cd72c762ac83) | 03:33:01 / 03:34:35 |

Findings:
- The prereg came before every B1 job. The candidate sets were committed with the prereg. The map was frozen from base items only and committed before any scorer ran. The S5 forecast was committed before any scorer ran. **Holds.**
- 1c before the prereg: the 1c capture job (job-443650c207cb) logged totals only. The 02:45 job-d21013b502d4 printed keys and shapes, not values. The X1 accuracy, F1 sites and r0 ranks were disclosed in the prereg. I found no job that read 1c values before a6a3863. I cannot see interactive reads outside mrun.
- The depth shards were submitted 2 minutes before the map commit. Shard 0 started 22 s before that commit. They do not read the map: the D path hard-codes the prereg's L20 KV6 / h12+h13 rule. No effect.
- The arms existed on disk from 03:27:21, before the forecast ran and before the scorer was written. The forecast script reads no arm file (checked in the code). The scorer was committed before its first run. The 6414981 change only slices W_U before casting. The two scorer runs, at cfca8a3 and 6414981, print byte-identical summaries (compared in Python).
- Minor: `runb1s.sh` and `runph.sh`, the shipped wrappers of the scoring and post-hoc jobs, are not committed. The smoke arms (`arms-b1-smoke-*.jsonl`) are not in the repo, so I cannot show they were not opened. Their logs carry no values.

## 2. Recomputation notes

- S1–S4 and the nulls match the builder exactly. I got the same family null table: rule 22/39/31/12, last 22/28/30/11, mass 22/29/29/7, lens24 18/31/25/9. The cross-item null gives 0/0/1/0.
- E1 "moved" is 88/102 for me and 89 for the builder. The builder compares against the readout of the capture's native cell; I compare against the B1 run's own native cell. The bf16 re-run's carrier V differs from the capture by a median of 1.6% relative.
- The carrier row of the B1 native run differs from the capture's on 1/154 items (d3-lm:Suomenlinna fortress: row 45 vs 43). On that item, arm b edited a different row than the forecast assumed.
- Czech: the country token and the language token are identical (33150) on 8 of the 132 d2–d4 items. On those items, "names the language" is the same test as "names the country" over a different set. The builder's "country outscores language 92/132" counts these 8 exact ties as country wins; strictly it is 84 vs 40.

## 3. The S1 "real gain", 27/29 vs 13/29

- **Not post hoc.** The prereg says: "the map is credited only on items where R_argmax differs from the last span row". The comparison is preregistered, though without a bar.
- **The carrier row does not use the outcome.** R_argmax is the argmax of the L20 h12+h13 attention at the answer row (captured `p`). Heads 12 and 13 are exactly KV group 6 under h // 2 (H 16, KV 8). Ties go to the later row and t0 is excluded. The truth is never used. The rule itself comes from Run 1, not from these items.
- **Fair, but the null is weak.** 23 of the 29 items are d3-lm. In those, the last span row is a generic head noun ("Schonbrunn **Palace**", "Suomenlinna **fortress**"), and the carrier is the row before it (median offset 1). On the same 29 items:
  - the mass null names the country on 13/29;
  - a random other span row is expected to name it on 10.2/29 (exact average over rows);
  - the oracle row at L20 KV6 names it on 28/29;
  - the rule picks the oracle row on 23/29.

  The gain is real: the reader's attention finds the naming row. It is shown against a strawman null on mostly one family.

## 4. S5 sign agreement, 153/153

- **Trivial as a sign test.** All 153 native ΔM values are negative and all 153 forecasts are negative. ΔM = 0 on Anitkabir, which is excluded by the 0.125 threshold. The sign is not fixed mathematically, but the design fixes it in practice: the edit transplants the donor's carrier cell, and the foil is the donor's answer. So the null "the transplant lowers M" also scores 153/153.
- **The informative part is size.** Spearman(forecast, native) is 0.70 over the 153 items, against a shuffled 95th percentile of 0.13. The forecast/native ratio has median 0.52 (IQR 0.38–0.67). So the first-order forecast ranks effect sizes well and underestimates them by about 2×.
- Recommendation: retire the sign bar for same-direction transplants. Report the rank correlation and the ratio instead.

## 5. Code

- `cell_readout` is exactly a linear functional of h at its row. With n = ρ γ ⊙ h, v_g = W_V^g n, so C = d·W_g v = (ρ γ ⊙ W_V^gᵀ W_gᵀ d)·h, and that is the code's `vector`. Checks:
  - Qwen3 has no v_proj bias and no v_norm (only q_norm and k_norm); the code refuses both cases.
  - The rows of W_V^g are `kv*dh:(kv+1)*dh`, which matches HF's KV-major v_proj layout.
  - `group_writers` uses h // (H/KV), which matches `repeat_kv`.
  - ρ is the native input-norm scale. The ρ change is handled by the norm term.
  - The census and the readout agree with my independent implementation on every count.
- **`hop_edge` closure follows by construction.** `rounding` = native_delta − total − residual_gap = C₁ − C₀ − c·Δh − (ρ₁/ρ₀ − 1)c·h₁. That is the algebraic identity C = ρ c_unit·h. So the "gap" (0.13 absolute, median 0.08% of the change; arm b 0.9%) checks only that the bf16 V cache matches the fp64 linear functional of the captured h. It is a good check of `cell_readout`, not a test of the decomposition. In the tests, line 154 of `tests/test_analysis_circuit_map.py` (sum of parts == native) is a tautology. Line 156 (gap < 1e-3 relative) is the real check, and it runs only in fp32.
- **`Map.freeze` / digest.** The digest is sha256 of sort_keys JSON. I recomputed it from the committed `b1-map.json` and it reproduces ef1d5b63…. The JSON round-trip is stable (tested). `modes` breaks ties by first appearance, so it depends on item order but is deterministic. The digest is a content hash. A re-run on GPU could change the constants (best-cell margins), so equal digests require equal reports, not just the same code.
- **Tests.** I ran the touched tests through `common exec` with the worktree `src` and `packages/mdb/src` on PYTHONPATH:
  - `test_analysis_circuit_map.py`, `test_analysis_symbols.py`, `test_ops_sets.py`: **69 passed, 2 skipped**.
  - The repo-root `src` (saturn_pub) is required. With only the package `src`, collection fails on `InPlaceWrite`.

## 6. S3 near-synonyms and the S2 chance problems

**Lexical null.** Take a write that is only the true country's own readout direction, u_c = γ_F ⊙ W_U[country]. What does each set readout name? (job-3a4daafe575b)

| Co-readout | Lexical null | Carrier | Chance (rank 1 / rank ≤ 3) |
|---|---|---|---|
| Capital rank 1 (correct d2–d4) | **105/105** | 103/105 (rank ≤ 3) | 1/43; ≤ 3: 0.07 |
| Language rank 1 (all d2–d4) | **112/132** | 99/132 | 1/26 |
| Creator rank ≤ 3, d4 correct | 5/12 | 8/12 | 0.14–0.16 |
| Creator rank ≤ 3, d3-per correct | 15/32 | 20/32 | 0.14–0.16 |

- **Language (S3).** The builder is right that the similarity makes the falsifier fire: the country direction alone names the language more often than the carrier does. The data do not support "not hop 3", though:
  - with the truth-country direction projected out of the carrier write, the language is still named on 75/132;
  - on the 20 items where the lexical null misses, the carrier names the language on 9/20.

  Correct statement: the S3 cell test cannot separate hop 3 from country-correlated content. Unknown, not negative.
- **Capital (S2).** The same problem applies, and the tracker misses it. The capital co-readout is fully predicted by the country direction (105/105). So "latent capital rank ≤ 3 on 103/105" is not evidence of a stacked second symbol. The tracker marks S2 "holds" while treating the same effect in S3 as an artifact; the two verdicts should be consistent.
- **Creator (S2).** This is the partial positive. Where the lexical null fails (24 d3-per/d4 items), the carrier still ranks the creator ≤ 3 on 10/24 (0.42 vs chance 0.16).
- **Ladder chance.** With randomly drawn fake (person, city, country) truths, the strict ladder holds on 0.08 of d4 items on average (0.11 over d3-per + d4). On the real d4 items, the first naming layer is L0 for the person on 9/12, the city on 7/12 and the country on 8/12. Under chance, 0/12 has probability about 0.37. The 0.60 bar was unreachable for this statistic. The builder's "uninformative" verdict holds.
- **Language-in-MLP "first layer".** With a fake language truth, the first layer is < 21 on 54/105 items, against a per-layer chance of 0.038. The statistic is chance-dominated, as the builder says. The by-layer counts (L20–22: 42, 73, 70 of 132) are the meaningful readout.

## 7. S6, the E1 arm, in more detail

- Bars fail, as reported: E1 names the donor on 23/102 and the donor language on 16/102; the control leaves both untouched on 8/29.
- **"Not a content transfer" is too strong.**
  - Within countries, the donor country's rank at the carrier improves on 75/102 items (median 21 → 7.5). After the edit it is ≤ 3 on 35/102. Its score rises on 67/102.
  - On the 29 items with both arms, the pairwise cell margin C(truth − E1-donor country) uses the same target in both arms. It flips on **20/29 under E1 vs 2/29 under the control**. E1 lowers it more than the control on 25/29.
  - The truth drop dominates: median −84% of the native truth score under E1, −27% under the control. The carrier V norm shrinks to 0.65× under E1 and 0.78× under the control.
  - I did not run a random-other-country null for the donor gain, so its donor-specificity is suggestive, not established.
- **Design reason the edit is blunt.** E1 replaces V only. K is left native, so the native keys address donor values. The carrier row's own token and residual are untouched, and the carrier is usually the entity's last name token. A full transfer is therefore not expected.

## What B1 supports

**Supported positives:**
- The frozen R_argmax rule picks the country-naming L20 KV6 row out of sample, beating the last-row and mass nulls on the 29 items where they differ.
- The carrier cell's country readout predicts errors: 22/27 vs 6/83, holding within each family.
- The sealed first-order forecast ranks transplant sizes (Spearman 0.70) and underestimates them by about 2×.
- The creator partly reads at the carrier beyond the lexical null.
- `cell_readout` and the census are correct.

**Partial or overstated:**
- the capital and language co-readouts (lexical);
- the S5 sign (trivial);
- "E1 is purely destructive";
- the closure (an identity).

**Fails as preregistered:** S1 bar, S3, S6 bars, the hop-3 MLP share.

# Independent audit: mdb + library showcase, experiments 1-7 (2026-11-14)

Auditor: an independent Opus agent that did not build these experiments. Scope: the claims in tracker entry 11 (`obsidian/research/2026-11-13-211248-mdb-experiment-tracker.md` lines 1064-1136), checked against the drivers, the scorers, the raw `out/*.jsonl`, the job logs, git history on `mdb-showcase` (3676808..64b86c4, 64b86c4 is an ancestor of the `mdb-engine-refactor` HEAD), and the mrun job records (`/api/jobs`, `created_ts`). Local Python on saved outputs only: no GPU, no mrun jobs, no edits to results, no commits.

## Method

1. Re-ran every `exp*/score.py` on the saved outputs. All seven reproduce their `summary.json` exactly, so there are no scorer or transcription errors (exp3 needs `../twohop-20261113/next/b1-score.json` as its third argument). Census rebuilt with `exp1/census_build.py ../exp2/out`: identical to the committed `exp1/census.json` (58 features, 24/29 countries).
2. Read each driver and scorer against its prereg's operational definitions.
3. Recomputed the claims that are not in any `summary.json` (pyvene IIA and flip agreement, `content(C)` readout, verify records), and ran extra breakdowns on the raw data where a claim depended on how effects sit relative to the floor.
4. Checked sealing order with git commit times against mrun `created_ts` for every job.

The `results/showcase-20261114` files in this worktree are byte-identical to the ones in `.worktrees/mdb-showcase` (shasum over 80 files).

## Verdict table

| Exp | Builder claim | Verdict | Evidence |
|---|---|---|---|
| 2 | SAELens and mdb active sets agree 108/108 | CONFIRMED (the check is narrower than it sounds) | `exp2/summary.json` reproduces it. The SAELens object comes from mdb's own `FeatureBasis.to_saelens()` (`src/mdb/analysis/sae.py:200-220`, a strict `load_state_dict`), because the registry load failed offline (`provenance-s0.json` `registry_load.ok=false`, LocalEntryNotFoundError). So the check shows that mdb's conversion and SAELens's TopK encode agree. It does not test SAELens's own `qwen_scope` loader against mdb. |
| 2 | Components sum to the cell on 64/64 | CONFIRMED | 64/64 at L20 and at every layer, max gap 1.8e-15. This is a float64 algebraic identity, so it validates the B-a code, not anything about the model. |
| 2 | L1 fails, L2 holds, L3 fails with its falsifier, L4 holds | CONFIRMED | L1: top share >= 0.5 on 41/64, but error share <= 0.2 on only 42/64 (bar 48). L2: medians top 0.59, random 0.0, error 0.11 (n=63). L3: median onset L17, and 46/63 are >= L15, so the falsifier fires. L4: 48/64. Caveat on L4: 46 of the 48 are above floor only at L20 (4 at L18). That is mostly where the answer row reads KV6, and 16/64 items are never above floor. L4 therefore says "the effect is at the reading layer", not much about the feature's lifetime. The memo's cross-item and shuffled-id nulls were not run (replaced by the random-component and last-row nulls, which the prereg states). |
| 1 | Watch fires on 18/27 wrong and 30/83 correct | CONFIRMED | Rescored. `content(C)` names the true country on 78/83 correct and 5/27 wrong items (recomputed from `content_C`), as claimed. |
| 1 | One-feature swap repairs 12/27, whole cell 22/27, random 2/26 | CONFIRMED | Rescored. Random is norm-matched to the R1 change (`exp1_depth.py:191-192`), so 12 vs 2 is a fair comparison. R1 repairs 9/18 where the watch fired (bar 11) and moves the margin a median 0.39 of R2's. W2 fails as reported. |
| 1 | Sealed forecasts verified 26/26 | CONFIRMED, but overstated in meaning | The seal is taken after the edit has committed, at `t<last>.L21.h` (`exp1_depth.py:96-97`). The prediction is mdb's float64 emulation (P19, derived-exact) of L21..L27 from the edited state. Max \|pred - native\| is 0.16 against floors of 0.25-0.5. This is evidence for engine fidelity on edited states. It says nothing about whether the repair works. |
| 1 | W3 untestable | CONFIRMED | The city-donor clause has n = 0 (no R2 donor was d2 under the frozen rule). The scorer prints `holds: false`, but "not testable" is the right reading. |
| 3 | 63 features; language/city ratio 1.01; the H2 falsifier fires | CONFIRMED (one wording fix) | The 63 are transcoder language-writing features (17 languages; 53 in L21..24). They are not "country" or SAE features. H2 statistic median -1.75 (bar >= 0), ratio 1.01, joint dM lang -2.5 vs city -2.0. H3 96/119. H4 5/5 is weak evidence (see corrections). |
| 4 | G3 and G4 (Spearman 0.93) hold; G1 (0.105) and G2 fail; the entry-5 2x is the linear response of later layers | CONFIRMED, with a resolution caveat | Rescored. Bracket identities checked on all 160 node-items (rerouting = arm0 - arm1, norm = arm1 - arm2, nonlin = arm2 - edge). Median \|brackets\| (0.125, 0.095, 0.049) sit at or below one bf16 step against a 0.25 floor. Only 44/160 node-items have any bracket above floor (rerouting 8, norm 25, nonlin 26). On node-items with \|arm0\| > floor, G1 is 0.095 (> 2 floors: 0.095; > 4 floors: 0.087), so the 0.105 "fail" is at quantisation resolution. G2's "largest bracket" vote compares mostly sub-floor values. The reading itself holds and gets stronger with effect size: arm0/edge is 0.92 (> floor) and 0.97 (> 4 floors), and arm0/direct is 0.61-0.64 above floor (0.51 over all nodes). The "2x" is about 1.6x on above-floor nodes. The frozen-gradient edge explains it. |
| 5 | Sign 26/26; N2 ratio 0.83 fails | CONFIRMED | N1: named beats random by >= 0.5 on 26/26 and beats the answer-null on 26/26 (medians -4.06 / 0.0 / 0.06). N2: Spearman 0.94 over 71 neurons, median ratio 0.83 (band 0.4-0.7). N4 51/52. Caveat: "out of sample" means held-out items of the same country, which share the answer token. Nomination uses A items only (`exp5_neurons.py` nomination block), but the knock-out mean is over all 64 items. Native N1 values come from nnsight batched arms; the stepped authority covers 2 arms per item. |
| 6 | Pure reparametrisation in fp32 (max 2.3e-5) | CONFIRMED | 48/48 per model; max 2.29e-5 (Qwen3) and 1.81e-5 (Qwen2.5). |
| 6 | 60/274 and 66/228 above-floor effects flip or fall below the floor in bf16 | CONFIRMED as counts. "Flip" is the wrong word. | Sign reversals are 1/274 (Qwen3) and 28/228 (Qwen2.5, all from processing). The rest fall from above to below the floor. All losses are in effects <= 4 floors. Effects > 4 floors: 0 of 110 (Qwen3) and 0 of 104 (Qwen2.5) lose their verdict on any path. |
| 6 | Kernel path alone does much of it (SC6) | PARTLY | The comparisons are set up correctly. Kernel path = U16 batched vs mdb stepped. Processing = P16 vs U16 with the same batch layout and the same weights, so it isolates processing. Qwen3: kernel 60/274, processing 46/257, so kernel path is the larger share. Qwen2.5: kernel 16/228, processing 82/249, so processing dominates. "Much of it" is true for Qwen3 only. Extra check: TL batched vs TL batch-1 alone (same library, same weights) loses 72/286 (Qwen3) and 19/241 (Qwen2.5). Any reordering moves 3-spacing verdicts. |
| 6 | SC7 (wrong TL slice in the smoke) did not leak | CONFIRMED | The 4-D fix and the prereg amendment are in `6659478` (12:24:09). Full jobs c5a45f9c2811 and 80cc14782825 ran `7b125f4` (created 12:30:15 and 12:35:17). `out/` holds only full-run files. The data also rules out a wrong slice. TL vs mdb on the V interchange: median relative difference 2.3% / 1.9% on effects of median 9.4 / 6.2 logits. Head-zeroing at L_c: 61 / 57 effects > 4 floors, with no verdict loss on the kernel path. A wrong KV group or head mapping could not produce this agreement. |
| 7 | Whole-cell 44/64, pyvene 43/64, same verdict 63/64; SAE subspace (median rank 4) 4/64; random 0/64 | CONFIRMED | Recomputed from `exp7-items.jsonl` (the pyvene numbers are not in `summary.json`). Ranks: one rank-0 pair, max 9. The SAE subspace moves a median 0.34 of the whole-cell margin change (IQR 0.07-0.50). |
| 7 | pyvene-mdb gap equals the no-intervention kernel gap | PARTLY | The medians are equal (0.125 vs 0.125). The tails are not: max 0.75 vs 0.375, and 10 vs 7 pairs exceed one step. "Same order as the kernel-path gap" is supported. "Equals" is not. |
| 7 | Sealing (05cc1bf before c8144767543d) | CONFIRMED | `exp7/prereg.json` has one commit, 05cc1bf at 12:27:18. Smoke e8799c85caa9 was created at 12:37:56 and the full job at 12:38:40. The prereg's "written 13:50" text is wrong (see timestamps). |

## Cross-cutting checks

**Sealing order (CONFIRMED for all seven).** In every case the prereg commit comes before the first scored job's `created_ts`:

| Exp | Prereg commit | First scored job created |
|---|---|---|
| 1 | 9d4533e 11:17:01; census 62ae41c 11:30:36 | 12:02:30 |
| 2 | 9d4533e 11:17:01 | 11:18:51 (the smoke shares the commit second) |
| 3 | 16d6b63 11:24:10 | 12:16:50 |
| 4 | 818cb0e 11:39:35 | 12:26:14 |
| 5 | 16d6b63 11:24:10 | 12:26:14 |
| 6 | amendment 6659478 12:24:09 | 12:30:15 |
| 7 | 05cc1bf 12:27:18 | 12:38:40 |

Every driver's last commit also precedes its full job, and no driver changed afterwards. Exp1's two shards ran at a727389 and 63084a8, with no diff in `packages/mdb/src` or `exp1/` between them.

**Prereg "written"/"at" timestamps are unreliable and should not be cited as evidence.** Exp4 says 12:35 but was committed 11:39. Exp5 says 11:50 (committed 11:24). Exp3 says 11:40 (committed 11:24). Exp6 says 12:05 (first committed 11:30), and its amendment says 13:30 (committed 12:24). Exp7 says 13:50 (committed 12:27). The seal is the commit order above.

**Payload provenance gap (minor).** `submit.sh` archives `src` and `packages/mdb/src` from git HEAD, but copies the showcase drivers, preregs and census from the working tree. `REVISION` therefore pins the library, not the driver. Since no driver changed after its full job, the committed drivers are almost certainly what ran, but nothing records it. Fix: archive the showcase dir from HEAD too, or hash the driver into the log.

**Amendments made before the full runs (CONFIRMED).** Exp7's 64-pair panel is in the only version of its prereg (05cc1bf), committed before the smoke. Exp6's fp32-on-CPU and 4-D-hook amendment is in 6659478, before both full exp6 jobs. Note on exp7: the I1 band (0.55-0.65) was derived from entry 1's 71/123 on a different panel, so I1's narrow "fail" (0.69) on the 64 1a pairs carries little information.

**SAELens idioms.** The strict `load_state_dict(strict=True)` is confirmed (`sae.py:217-219`). The t0 refusal is confirmed in `components.py:143-146` and `features.py:73-76`, and no carrier row used is 0 (min 34 in exp2/exp7, 43 in exp1). The registry load failed offline (SC1), so no experiment used SAELens's own loader end to end.

**Floor consistency.** The base rule, 2 bf16 spacings at the larger \|logit\| of the pair (`common.py:floor_of`), is used in every experiment. Exp2 and exp5 add a measured batched-vs-stepped delta, because their arms are batched, and their preregs say so. Exp6 uses the pure 2-spacing floor at the largest \|logit\| over the item's arms, as preregistered, and its `supplied.judge` count (274/228 native effects) matches the scorer's. Exp6 does not add a kernel-path delta, which is correct because its question is whether the bare 2-spacing verdict survives. It is consistent with the declared definition throughout.

**Memo null not met in exp6.** The memo expected "unprocessed bridge against mdb must be bitwise". Measured: 13/48 (Qwen3) and 7/48 (Qwen2.5). The prereg narrowed this to "the same kernel path", and TL batch-1 is a full-sequence forward while mdb's is stepped, so they are not the same path. The builder reports this honestly, but it is a failed memo expectation, not a held null.

## Most important corrections

1. **Exp6 / SC6 wording.** Replace "above-floor verdicts don't hold across kernel paths" with this measured statement. A verdict 1-2 floors above the 2-spacing floor (3-4 spacings) is not stable under any reordering: kernel path, batch composition or processing. A verdict more than 4 floors above is stable: 0/214 losses over both models. Sign reversals are rare (1/274 Qwen3; 28/228 Qwen2.5, from processing). Loss by bin, total / kernel-only:

   | \|effect\| / floor | Qwen3 n | Qwen3 lost (total / kernel) | Qwen2.5 n | Qwen2.5 lost (total / kernel) |
   |---|---|---|---|---|
   | 1-1.5 | 89 | 50 / 51 | 41 | 32 / 14 |
   | 1.5-2 | 32 | 9 / 8 | 30 | 16 / 2 |
   | 2-4 | 43 | 1 / 1 | 53 | 18 / 0 |
   | > 4 | 110 | 0 / 0 | 104 | 0 / 0 |

   "Kernel path does much of it" holds for Qwen3, not Qwen2.5, where processing contributes 82/249 vs 16/228. For mdb's policy, the floor is right as a noise bound (p95 2-3.5 spacings), but a verdict needs a margin rule; the data supports something like 4 floors. Also, on Qwen2.5 the L_c head-zeroing subset has processing p95 of 5 spacings (max 11.8). The pooled falsifier (> 4) does not fire, but this subset would.
2. **Exp1 verify 26/26** is a float64-emulation fidelity check of L21..L27 on the edited state, sealed after the edit. It should not be read as a sealed prediction of the repair.
3. **Exp4.** G1's 0.105 "narrow fail" and G2's largest-bracket count are at bf16 resolution: median brackets of about 1 spacing, and only 44/160 node-items with any bracket above floor. On above-floor nodes G1 is 0.095 and the frozen-gradient edge accounts for 0.92-0.97 of native. The substantive reading is stronger than the pass/fail line suggests. The overstatement is about 1.6x on above-floor nodes, not 2x.
4. **Exp3 H4 (5/5)** is close to circular. The features were nominated for their direct effect on the language margin, and the test items are wrong items, which by definition failed to produce the language. Lower activation there is the default expectation. It is also a per-item majority vote (for example 5/7 and 2/3 below), not "the features are below". It does not localise those errors at the answer-row MLPs beyond what being wrong already implies.
5. **Exp2's 108/108** checks mdb's SAELens conversion, not SAELens's own loader (SC1 blocks that). **Exp2 L4** is effectively "above floor only at L20".
6. **Exp7.** "Equals the kernel gap" should be "same median, heavier tail" (0.75 vs 0.375). The I1 band was transplanted from another panel.
7. **Exp5's "out of sample"** is within-country: held-out items share the answer token. It closes item-level selection, not answer-level selection. The answer-token null (median 0.06) argues against simple unembedding alignment explaining the effect.

## Candidate mdb examples (for owner approval)

- **Exp6** (strongest and most consequential). It shows mdb's floor and sealed supplied-forecast judging on a second realisation of the same forward. The headline is the effect-size table above: clean, two models, a large effect, zero losses above 4 floors.
- **Exp5.** Nominate with nnsight gradients, seal 1,690 forecasts, measure with batched arms, check with stepped arms. N1 is 26/26 with a large effect (-4.06 vs 0.0), and N4 is 51/52.
- **Exp1.** The full debugger loop: feature watch, checkpoint, named-component edit, continue. One-feature repair is 12/27 vs norm-matched random 2/26, against a whole-cell reference of 22/27. The watch's false alarms (30/83) should be shown as part of the example, not hidden. Drop "verified 26/26" as evidence of repair.
- **Exp2.** Component binding with an exact partition. L2 is clean: the named component moves 0.59 of the cell, random 0.0, error 0.11.
- **Exp7.** pyvene interop agrees on 63/64 flip verdicts. The training-free SAE subspace moves 0.34 of the margin but flips only 4/64, against random 0/64. A good interop example and an honest negative.
- Conditional: **Exp4.** The method (nnsight frozen gradient plus mdb frozen arms) is sound and the frozen-gradient check is bitwise. The bracket verdicts sit at quantisation resolution, so present it with the above-floor subset.
- Not a candidate as-is: **Exp3.** Its main result is a falsifier firing (a useful negative), and H4 is weak.

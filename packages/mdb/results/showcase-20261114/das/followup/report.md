# 11b follow-up: vocabulary size vs test identity (Q1), rank-matched SAE sweep (Q2) (MEASURED, not audited)

Qwen3-1.7B-Base bf16, Beast via mrun, single seed. Same 64 exp7 pairs, L20 KV6 value cell at the carrier row, same floor (2 bf16 spacings) and margins (1 floor / 4 floors) as `das/das.py`. No training. Built from mdb objects: `subspace.readout_subspace`, `subspace_binding`, `SubspaceBinding.swap/mass`, `branches.Branches` (one checkpoint per pair, every arm at one floor), `subspace.iia`.

## Custody
| step | commit / job |
|---|---|
| prereg + vocab + driver sealed | `8cab48f` (prereg written 2026-11-14 14:15 from `date`), before every job below |
| smoke (4 pairs: 0,5,10,20) | job-bc2a72b861b3; whole, exp7 SAE, full69_k8/16/32 margins equal 11b bitwise; Branches M0 = native M0; census images rho-proportional (dev 1e-15) |
| scored, 4 shards x 16 pairs | job-8b01f1a12d84, job-33a4b67a28eb, job-60bce10afa0c, job-70ebd29d8f10 (submitted 14:17) |

Every record carries fu.py sha256 `65334dbd...`, prereg sha256 `93b867eb...`, vocab sha256, and the mdb custody block (`dbg.custody(script=, prereg=)`). Reproduction vs 11b over 64 pairs: whole, exp7 SAE (4/64) and full69_k1..32 equal 11b's symtab arms with max |dM| = 0.0. Caveat: Branches' M0 (prefill = carrier row) differs from the native M0 by up to 0.375 on some pairs; arms and M0-for-flips use the same convention as das.py (native M0, carrier-row-prefill arms).

Vocabularies (all single-token ' Name', so no multi-piece fallback): full69 = 11b's 69 (28 test + 41 others). a69 = the 41 non-test + 28 seeded extras (69 tokens, zero test countries). aall = 41 + all 50 extras (91 tokens, zero test). lto = full69 minus the pair's recipient and donor countries (67 tokens, per pair).

## Q1: flips at 1 floor / 4 floors (median fraction of whole-cell margin change moved). Whole 44 / 38.
| arm | k=8 | 16 | 32 | 64 |
|---|---|---|---|---|
| full69 (11b span, contains all 28 test countries) | 2 / 2 (0.33) | 15 / 12 (0.67) | 35 / 31 (0.89) | 43 / 34 (0.95) |
| a69 (69 tokens, no test country) | 0 / 0 (0.25) | 6 / 3 (0.45) | 20 / 13 (0.69) | 32 / 23 (0.83) |
| aall (91 tokens, no test country) | 0 / 0 (0.26) | 7 / 4 (0.47) | 22 / 16 (0.71) | 36 / 26 (0.87) |
| lto (full69 minus the pair's 2 countries) | 0 / 0 (0.24) | 4 / 2 (0.45) | 22 / 16 (0.74) | 35 / 28 (0.87) |
| (11b) 38 training tokens only | 0 / 0 | 6 / 4 | 16 / 12 | - |
| (11b) DAS held-out | 1 / 1 | 4 / 2 | 16 / 9 | - |

Median share of the cell delta inside the span at k=32: full69 0.73, a69 0.62, aall 0.62, lto 0.63.

**Q1 answer.** Both. Adding tokens helps a little (38 training tokens 16 -> 69 non-test tokens 20 -> 91 non-test tokens 22 flips at k=32; 32 -> 36 at k=64). But the large part of the readout span's 35/64 is test identity, and specifically the pair's own two countries: dropping only the recipient's and donor's tokens (lto, 26 of 28 test countries still present) costs 13 flips (35 -> 22), the same as dropping all 28 test countries (a69 20, aall 22). The 26 other test countries contribute nothing measurable beyond a same-size generic span. Per the prereg decision rule a69 = 20 is on the "identity" side (<= 20); V1 falsified, V2 falsified, V3 held. Nuance: a span with no knowledge of the pair still reaches 22/64 (36/64 at k=64), above held-out DAS (16), so the readout directions carry real generic country structure; the extra 13 flips come from containing the pair's own two token directions. Single seed, one random draw of extras (a69) plus the all-extras arm (aall), which agree within 2.

## Q2: SAE top-k of the 58 census features vs the readout span (flips 1 / 4 floors, median fraction moved)
| arm | k=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| full69 readout span | 0 / 0 (0.02) | 0 / 0 (0.05) | 0 / 0 (0.19) | 2 / 2 (0.33) | 15 / 12 (0.67) | 35 / 31 (0.89) |
| saeA: top-k by cell-image norm (weights only, test-free) | 0 / 0 (0.01) | 0 / 0 (0.03) | 0 / 0 (0.05) | 1 / 0 (0.16) | 8 / 2 (0.45) | 22 / 10 (0.70) |
| saeB: census evidence order (TEST-INFORMED, census includes the 64 base items) | 0 / 0 (0.01) | 0 / 0 (0.03) | 0 / 0 (0.07) | 1 / 0 (0.09) | 3 / 1 (0.26) | 14 / 8 (0.59) |
| saeC: exp7's per-pair active set ranked by A, truncated to k (TEST-INFORMED; rank saturates at about 4-5) | 0 / 0 (0.08) | 1 / 0 (0.25) | 4 / 1 (0.34) | 4 / 1 (0.34) | 4 / 1 (0.34) | 4 / 1 (0.34) |

exp7 per-pair SAE (reproduced): 4 / 1, moved 0.34, median rank about 4. The one pair with no active census feature (landmark:the Acropolis) is scored as no change for sae and saeC.

**Q2 answer.** At matched rank the readout span beats the all-census SAE sweep at every k >= 8 (k=32: 35 vs 22 test-free, 14 test-informed; k=16: 15 vs 8 and 3), and the test-free SAE span (22) equals the test-free readout span without the pair's countries (22), so the SAE's 58 features are about as good as a generic 69-91 country readout span, not better. Per dimension at small rank the per-pair SAE set is the better handle (saeC k=4: 0.34 of the margin change moved and 4 flips vs readout 0.19 and 0); its edge is that it picks the pair's own countries' features, the same identity effect as Q1, and it stops at about 4 dimensions. V5 (saeA < 18 at k=32) falsified at 22 but its falsifier (>= 35) not hit; V6 held; V7 held; V4 held.

## Not verified / limits
Single seed; a69 is one draw (aall is the full-extras control); no re-audit. saeB and saeC are test-informed and labelled; only saeA, a69 and aall are test-free. k=64 for the SAE sweep not run (58 features only).

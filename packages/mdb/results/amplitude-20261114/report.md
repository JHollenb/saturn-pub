# 12. Amplitude law of latent hops (MEASURED, not audited)

Memo: `obsidian/model-math/ADVISOR-2026-11-14-open-problems-we-could-solve.md`, §2.1 (arms A, B, C) with §2.3 (gate-switch forecasts F1–F3) as the companion. Models: Qwen3-0.6B-Base, Qwen3-1.7B-Base and Qwen3-4B-Base. Single seed. Items come from the scale study (`results/scale-20261114`). Every number is **measured** from `out/<model>/*.jsonl` by `score.py`, which writes `summary.json`. Statements labelled *asserted* are judgement, not measurement.

## Design

**The code.** For each item, the code is its own true-country direction at the carrier cell (L_c KV6):

- d = (leave-one-out same-country mean − leave-one-out all-correct mean), normalised. Both means come from an mdb `Pool` of correct items.
- The amplitude is a(v) = (v − m_all)·d / |m_c − m_all|.
- a = 1 is the mean code of the correct same-country items.

This direction is per-item and test-informed. A second arm uses the rank-k class-mean span over all panel countries (k = 1, 4, 16); it is test-informed at the class level.

**The arms.**

- **A (native bf16):** v(α) = v + (α−1)P(v − m_all), for α from 0.5 to 8.
- **B (native bf16):** a true back-patch at the carrier row. h at s ∈ {L_c+2, +4, +6, L−1} is written into t ∈ {L_c−8, −6, −4, −2}, giving 16 pairs. Each pair has a matched-norm random null. `Branches` probes read the carrier amplitude at L_c−3..L_c+1 after each patch.
- **C (fp32 replica from mdb core files):** at α*, find which gates switch sign, and compare them with the exp5 composer neurons.
- **F (§2.3, fp32 replica):** donor-swap forecasts F1, F2[k] and F3[k], plus per-layer dose log-slopes.

## Custody

| Step | Job / commit |
|---|---|
| mdb features: Branches probes, gate_switch_forecast | `cc8de89`, `a870f2d` |
| Prereg sealed (2026-11-14T15:19:56-0800, from `date`) | `9275a0d` |
| Smoke run (1.7B, 2 wrong items) | job-129e4505bb4f |
| Prereg amendment for kernel-path near-ties, made before any scored job | `3bc7b3b` |
| 1.7B pools, A, E1 and core files | job-c850679397c0 |
| 1.7B B, first attempt. Crashed at seal (record > 64 KB); no output used | job-0268bcc0ac13 |
| Fix: `Branches.seal` stores arms by pointer | `b36c93f` |
| 1.7B B (3 shards), C, F | job-0d1020343a98, job-2eb176486c75, job-58212ec7e40d; job-5aa645ee60ca; job-251ccbda4c9e |
| 0.6B pools+A, B (3 shards), C, F | job-ad7c258315d2; job-185b4d2bab15, job-ec197ed03c72, job-c98f3f880041; job-d272301bd3f5; job-63709b06ef48 |
| 4B pools, A, B (4 shards), C (2, CPU), F (3, CPU) | job-53ff2ecf15a4; job-f1bfe8ca20d7; job-3433a33ef147, job-15dc8e675715, job-51749f2f4f48, job-d8a265b0bd9c; job-1543ba7ed3c0, job-354fd649dfdf; job-bb1d707ba4f2, job-597e13ec325f, job-4378ae018664 |

- Every job stayed under 10 minutes; the longest step took 4:45.
- Jobs were submitted only when Beast had no running or queued job.
- The payload is `git archive HEAD`; the job note carries the `amp.py` sha256.
- Every Pool, Branches and core-file record carries custody (`amp.py` and prereg sha256). Records are in `out/<model>/records-*.jsonl`.
- Core files are on Beast at `/mnt/ssd1tb/cache/beast/amplitude/<model>/core/`.
- The replica null check (`Run.core_check`) passed on 123 of 126 core runs. The misses were 0.6B Pablo Neruda, 1.7B Babylon, and the 1.7B near-tie d5-081.
- The `hop_edge` closure was ≤ 2.7e-5 in fp32 everywhere.

## Arm A: dose ladder on the true-country direction (native bf16, answer-row footing; stepped arms)

Near-ties are excluded (amendment): 1.7B d3-lm:Tivoli Gardens is already correct on this kernel path, M0 = +0.25 = 0.5 floor.

| Model | Stratum | n | Monotone | Flipped (α ≤ 8) | α* | Median α* | Critical a·α*: median; q75/q25 | Flips ≥ 4 floors | Random null repairs |
|---|---|---|---|---|---|---|---|---|---|
| 0.6B | right-operand | 9 | 9/9 | 6/9 | 3, 3, 1.25, 2, 2, 3 | 2.5 | 1.01; 1.25 | 4/6 | 1/9 |
| 0.6B | readable | 23 | 22/23 | 3/23 | 6, 8, 6 | 6 | 2.02; 1.69 | 2/3 | 0/23 |
| 1.7B | right-operand | 9 | 9/9 | 6/9 | 8, 2, 1.25, 1.5, 2, 1.5 | 1.75 | 0.66; 2.65 | 0/6 | 4/9 |
| 1.7B | readable | 17 | 15/17 | 6/17 | 8, 3, 6, 4, 2, 3 | 3.5 | 0.74; 2.67 | 3/6 | 1/17 |
| 4B | right-operand | 8 | 7/8 | 3/8 | 1.5, 8, 6 | 6 | 1.39; 1.63 | 2/3 | 1/8 |
| 4B | readable | 11 | 11/11 | 1/11 | 6 | 6 | 1.53; n/a | 0/1 | 0/11 |

**Amplitude predicts the slope of the ladder.**

- Spearman(native a, M(8) − M(0.5)): 0.90 on 0.6B (n = 32), 0.78 on 1.7B (n = 26), 0.54 on 4B (n = 19).
- Items with a ≤ 0 lose margin as α grows. Example: 1.7B d5-088 has a = −0.27, and M goes from −4.5 to −8.1.

**Comparison with the matched-norm random null at α = 3.**

- The dose arm beats the random null on 16/32 (0.6B), 16/26 (1.7B) and 12/19 (4B) items.
- Median dM differences are +0.06, +0.31 and +0.25.
- The 1.7B right-operand items sit 1–5 floors from the flip, and the random null flips 4/9 of them. On 1.7B the flip counts mostly show that these items are fragile; the monotone response is the part that is specific to the dose.

**Class-mean span arms.** Batched flips by α ≤ 8 on right-operand items, out of 9 / 9 / 8. The true-direction arm is shown for reference.

| Model | span1 | span4 | span16 | True direction |
|---|---|---|---|---|
| 0.6B | 2 | 3 | 1 | 6 |
| 1.7B | 2 | 3 | 6 | 6 |
| 4B | 0 | 1 | 0 | 3 |

**P2: native amplitude by group.** Correct items use leave-one-out means and are restricted to the right-operand items' countries.

| Model | Correct median a | Right-operand median a | Readable median a | AUC correct > right-operand | AUC correct > readable |
|---|---|---|---|---|---|
| 0.6B | 0.87 (n 26) | 0.36 (n 9) | 0.03 (n 23) | 0.97 | 1.00 |
| 1.7B | 0.86 (n 31) | 0.33 (n 9) | 0.04 (n 17) | 0.76 | 0.96 |
| 4B | 0.94 (n 35) | 0.29 (n 8) | −0.03 (n 11) | 0.95 | 1.00 |

**E1: dose-down on correct items.**

- At α = 0 the argmax is lost on 16/72 (0.6B), 27/86 (1.7B) and 11/103 (4B) items.
- Median M(0)/M0 is 0.75, 0.83 and 0.97.
- At α = 0.5 the losses are 4, 5 and 2.

## Arm B: true back-patch at the carrier row (native bf16; every row recomputed from the carrier row)

Near-tie excluded: 1.7B d5-081.

| Model | Right-operand items repaired (any pair) | Readable items repaired (any pair) | True arms repairing | Random arms repairing | Repairing arms that raise a(L_c) | Median Δa(L_c) per pair |
|---|---|---|---|---|---|---|
| 0.6B | 0/11 | 3/27 | 8/608 | 0/608 | 0/8 | −0.09 to −0.01 |
| 1.7B | 0/12 | 4/19 | 25/496 | 12/496 | 0/25 | −0.12 to −0.03 |
| 4B | 1/9 | 1/15 | 6/384 | 16/384 | 0/6 | −0.08 to −0.03 |

- No (s, t) pair raises the carrier amplitude on median.
- The amplitude falls further as s gets later. On 1.7B, s = 27 gives Δa ≈ −0.11 and s = 22 gives −0.03.
- The 13 arms on 0.6B that did raise the amplitude repaired nothing.
- Spearman(Δa, dM) over arms: 0.23 (0.6B), 0.33 (1.7B), 0.21 (4B).
- The exists-any repair rate is 1–13%. Biran et al. report 57–66% (*asserted*: from the abstract only; their protocol and pair search differ).

## Arm C: gates that switch at α* (fp32 replica from core files)

- The fp32 replica reproduces the bf16 α* on 30/32 (0.6B), 20/25 (1.7B) and 18/19 (4B) items.
- Replica flips: right-operand 6/9, 6/8, 3/8; readable 3/23, 6/17, 1/11.
- Median gates in L_c+1..L_c+4 that change sign: 48, 49, 55, 91 (0.6B); 46, 61, 81, 95 (1.7B); 78, 59, 78, 91 (4B).

**P4 (1.7B).** 9 flipped items have exp5 composers for their true country, N = 32 in L21–L24.

- Median Jaccard between the composers and the top-N switching gates is 0.00. A random set gives 0.0006.
- Per-item Jaccard: 0, 0, 0, 0.02, 0, 0.03, 0.016, 0, 0.049.
- At most 3 composers switch on any item.
- Caveat: the exp5 composers were nominated for capital answers, but these items have language answers.

## §2.3: gate switching and the attribution-patching gap (fp32 replica, donor swap on correct base items)

| Model | n | Native flips | F1 ratio | F2 ratio, k = 8/32/128 | F3 ratio, k = 8/32/128 | Flip mismatch: F1 / F2[32] / F3[32] / F3[128] |
|---|---|---|---|---|---|---|
| 0.6B | 28 | 17 | 0.59 | 0.86 / 0.90 / 0.90 | 0.88 / 0.93 / 0.91 | 14 / 4 / 3 / 4 |
| 1.7B | 41 | 31 | 0.58 | 0.84 / 0.88 / 0.88 | 0.88 / 0.92 / 0.94 | 25 / 8 / 7 / 4 |
| 4B | 42 | 16 | 0.35 | 0.62 / 0.65 / 0.63 | 0.62 / 0.64 / 0.63 | 16 / 16 / 16 / 16 |

**Dose log-slopes.**

- 1.7B exact-MLP log-slope by layer, L20–27: 0.77, 0.98, 1.19, 1.21, 1.14, 0.92, 1.02, 0.84. Band L21–24 = 1.15; tail L25–27 = 0.92; re-read 0.90.
- 4B: band L25–28 = 1.08, tail = 1.27, re-read slope = 1.68.

**4B.** F2[32] predicts 0 of the 16 native flips. At α = 1 the median shares of the change are 0.17 direct and 0.34 re-read, and re-reads are outside F1–F3 by construction.

## Prediction verdicts (re-sealed thresholds; the memo's values are in brackets)

| Prediction | 0.6B | 1.7B | 4B | Verdict |
|---|---|---|---|---|
| P1a: monotone on ≥ 0.8 (right-operand) | 9/9 | 9/9 | 7/8 | HOLDS |
| P1b: ≥ 0.5 flip and median α* in [1.25, 4] ([1.2, 2.0]) | 6/9, 2.5 ✓ ([✗]) | 6/9, 1.75 ✓ ([✓]) | 3/8, 6.0 ✗ | Holds on 0.6B and 1.7B; fails on 4B |
| P1c: critical a·α* with q75/q25 ≤ 2.5 and median in [0.5, 2] | 1.25, 1.01 ✓ | 2.65, 0.66 ✗ | 1.63, 1.39 ✓ (n 3) | Holds on 0.6B; narrow fail on 1.7B; 4B n too small |
| P1: readable flips ≤ 0.3 | 0.13 ✓ | 0.35 ✗ | 0.09 ✓ | Holds on 2/3 |
| P2: AUC ≥ 0.75; readable median ≤ 0.3 and below right-operand | 0.97 ✓ | 0.76 ✓ | 0.95 ✓ | HOLDS on all three |
| P3a: right-operand repair ≥ 2× readable | 0/11 vs 3/27 | 0/12 vs 4/19 | 1/9 vs 1/15 | FAILS (pooled 1/32 vs 8/61, opposite direction) |
| P3b: ≥ 0.8 of repairing arms raise the amplitude | 0/8 | 0/25 | 0/6 | FAILS (0/39) |
| P3c: non-raising arms repair ≤ 0.1, and at ≤ 1/3 the rate of raising arms | 0.016; raising arms 0/13 | 0.060; no raising arm | 0.020; no raising arm | First clause trivially true; second undefined or fails |
| P3d: random repairs ≤ 1/3 of true repairs | 0 vs 8 ✓ | 12 vs 25 ✗ | 16 vs 6 ✗ | Fails on 1.7B and 4B |
| P4: Jaccard ≥ 0.15 (0.3) and random ≤ 0.01 | — | 0.00 vs 0.0006 | — | FAILS |
| S_span: span16 ≥ true; span1 and span4 < true | 1 vs 6 ✗; ✓ | 6 vs 6 ✓; ✓ | 0 vs 3 ✗; ✓ | Mixed |
| E1: α = 0 loses ≤ 0.2; median M ratio in [0.4, 0.9] | 0.22 ✗; 0.75 ✓ | 0.31 ✗; 0.83 ✓ | 0.11 ✓; 0.97 ✗ | Mixed |
| F: F2[32] ≥ 0.85 | 0.90 ✓ | 0.88 ✓ | 0.65 ✗ | Holds on 0.6B and 1.7B |
| F: flip mismatch F2[32] ≤ 0.125 and F3[32] ≤ 0.0625 | 0.14, 0.11 ✗ | 0.20, 0.17 ✗ | 0.38 ✗ | FAILS |
| F: 1.7B slope 1.3–1.6 at L21–24 and 0.9–1.1 at L25–27 | — | 1.15 ✗, 0.92 ✓ | — | Half |
| F: 4B re-read slope > 1.5 | — | — | 1.68 ✓ | HOLDS |
| F1 replication in [0.45, 0.75] | 0.59 ✓ | 0.58 ✓ | 0.35 ✗ (memo measured 0.28–0.54 at 4B) | Holds on 0.6B and 1.7B |

## Falsifiers

| Falsifier | Fires? | Evidence |
|---|---|---|
| Amplitude AUC < 0.6 | No | AUC 0.76–0.97 |
| Threshold: α* scattered and non-monotone on > 0.3 of items | No | 0–1 non-monotone items per model |
| Timing vs amplification | **Yes** | Repairs are not more frequent on right-operand items, and 39/39 repairing arms have Δa < 0.1 |
| Gate switch: F2[128] < 0.75 | **On 4B only** (0.63) | The 4B gap sits in attention re-reads. No model has flat slopes (band − tail = 0.21, 0.23, 0.19) |

## Reading

### Supported (measured): the code at the fixed address has a scalar amplitude, and its size predicts success

- Along the item's own true-country direction, correct items carry about one class-code unit. Right-operand wrong items carry about 0.3, and readable wrong items carry about 0. AUC is 0.76–0.97.
- Scaling the amplitude raises the margin monotonically on 25/26 right-operand items.
- The sign of the native amplitude predicts the sign of the ladder (Spearman 0.54–0.90).
- **0.6B:** right-operand items flip when a·α* ≈ 1.0 (q75/q25 1.25). That is the level correct items carry, so this is a thresholded amplifier with a measured threshold.
- **1.7B:** the threshold is looser (0.66, q75/q25 2.65) and confounded by near-ties: the random null also flips 4/9 items.
- **4B:** 3/8 items flip, and only at α = 6–8. The 4B residual sits in attention re-reads (share 0.34, slope 1.68).
- *Asserted:* on larger models a rank-1 cell amplitude does not set the outcome alone.

### Not supported (measured): back-patching works by re-amplifying the code

- No repairing back-patch raised the carrier amplitude (0/39), and back-patching lowers the amplitude on average.
- Repairs concentrate in the readable stratum.
- Random matched-norm patches repair about as often (1.7B: 12 vs 25) or more often (4B: 16 vs 6).
- The sealed falsifier therefore fires. The honest reading is that these arm-B repairs support neither account. They are rare (1–13%), not tied to the carrier code, and mostly look like perturbations of near-ties. The exception is 0.6B, where true patches give 8 repairs and random patches 0.
- This run does not test timing directly.

### Gate switching (measured)

- On 0.6B and 1.7B, applying exact SwiGLU to the 8–32 largest-Jacobian gates per layer closes most of the first-order gap: the ratio goes from 0.58 to 0.88, and k = 8 already gives 0.84.
- Flip mismatches remain (7–8 of 41 on 1.7B), because re-reads and the norm are outside the forecast.
- The gates that switch under amplification are not the exp5 capital composers.

## mdb gaps hit

1. **Branches could not read anything but the consumer on each arm.** Added `Branches(probes=...)`, evaluated per stepped arm and on the base before restore, and sealed with the record (`cc8de89`; test `test_probes_read_each_arm_before_restore`).
2. **`Branches.seal` overflowed the 64 KB record limit** with 33 arms plus probes. Full rows now go in by pointer and the record keeps a compact table (`b36c93f`; test `test_seal_large_branches_by_pointer`).
3. **No gate-switch forecast existed.** Added `symbols.gate_switch_forecast` (F1, F2[k], F3[k]) (`a870f2d`; test `test_gate_switch_forecast_matches_hop_edge_first_order`).
4. **Not fixed: watches have no projection reduction and cannot ride on Branches arms.** Probes stood in for watchpoints over layers. Filed as `saturn/request.jsonl` handle `mdb-watch-projection-and-branch-watches`.
5. **Minor, not fixed:**
   - Pool has no class-mean-by-label for an item outside the pool (worked around with a stub item).
   - Run names reject ".".
   - `Branches.batched` cannot run add-op nulls, so the nulls ran stepped.
   - The 4B fp32 replica does not fit on the 16 GB card, so it ran from core files on Beast CPU.
6. **The `hopmath.py` fp32 replica was not needed.** mdb core files (bf16 prefix, fp32 answer row) reached a closure of ≤ 2.7e-5.

## Limits

- Single seed, with small strata: 8–9 right-operand items per model.
- P_true and the span arms are test-informed.
- Arms A and C fix the later rows' K/V; arm B recomputes those rows.
- Composer sets exist for 1.7B only, and they are capital composers.
- The 1.7B right-operand flips are below 4 floors (0/6 stable).
- Not audited.

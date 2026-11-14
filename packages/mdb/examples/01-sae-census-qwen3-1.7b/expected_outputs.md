# Expected outputs: SAE census on Qwen3-1.7B-Base

What the original experiment measured (MEASURED, single seed, not audited). Model Qwen3-1.7B-Base,
bf16, eager attention, CUDA on Beast; SAE Qwen-Scope `SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
(k = 50, revision `ce1a79d9c5163932d65c417380e53230e1086370`). Evidence:
`results/sae-integration-20261113/` (`job3/census.txt`, `job2/job-702270785e57.log`). Write-up:
`obsidian/blog/2026-11-13-211248-saes-meet-a-native-debugger-the-variable-is-in-the-features-the-computation-is-not.md`;
tracker entry 2 in `obsidian/research/2026-11-13-211248-mdb-experiment-tracker.md`.

## Jobs

| Job | What |
|---|---|
| job-6aa7bc89a6ed | census, 9 of the 18 pairs |
| job-204c4cf5bfab | census, the other 9 pairs |
| job-702270785e57 | error splice, steering, unit matching, full replacement chain (one prompt pair) |
| job-63034ebe0134 | entity-feature swaps (prelude; not reproduced by this example) |

## 1. Variable census, reader side, 18 city pairs (`job3/census.txt`)

`base` is the margin `logit(country_X) - logit(country_Y)` on X's prompt; `cell` the native change from
transplanting Y's value cell at the entity row (layer `L*`); `Delta` the whole-residual patch at
`L*.h_out`; `Phi` the features part; `E` the error part; `sPhi = Phi / Delta`, `sE = E / Delta`.

```
pair                     base     cf   allKV  L*    cell   K@L*   Delta     Phi       E   sPhi     sE
Paris->Rome              8.25  -8.25 -15.875  20   -8.25    0.0   -8.25    -8.0     0.0   0.97  -0.00
Rome->Berlin             9.00 -12.12 -19.875  20 -14.375 -0.125 -14.375 -14.125   -1.25   0.98   0.09
Berlin->Moscow          10.62  -7.88 -18.000  20   -11.5    0.0   -11.5   -12.5    0.75   1.09  -0.07
Moscow->Athens          12.75 -11.50 -23.875  20   -16.5    0.0   -16.5  -14.75   -1.75   0.89   0.11
Athens->Beijing         10.25 -14.38 -22.500  20 -12.875 -0.125 -12.875 -11.125  -0.625   0.86   0.05
Beijing->Madrid         15.50 -12.88 -25.500  20  -13.25    0.0  -13.25 -10.625    -2.0   0.80   0.15
Madrid->London           9.75  -9.62 -18.250  20  -12.75    0.0  -12.75 -13.875   0.375   1.09  -0.03
London->Prague          12.38  -7.62 -18.875  20  -10.75  0.375  -10.75   -11.5   1.625   1.07  -0.15
Prague->Vienna           6.88  -8.75 -14.625  20   -7.25    0.0   -7.25    -8.0   0.375   1.10  -0.05
Vienna->Amsterdam       12.25  -7.75 -19.750  20  -14.25 -0.125  -14.25   -15.0  -0.625   1.05   0.04
Amsterdam->Lisbon        7.75 -15.38 -22.125  20 -13.125 -0.125 -13.125  -10.75  -1.375   0.82   0.10
Lisbon->Budapest        15.25 -13.12 -27.750  20  -17.75    0.0  -17.75 -14.125    -1.0   0.80   0.06
Budapest->Tokyo         11.88 -15.12 -27.250  20  -21.25    0.0  -21.25  -20.25  -0.625   0.95   0.03
Tokyo->Cairo            14.38 -12.12 -25.125  20 -19.375    0.0 -19.375  -19.75  -1.125   1.02   0.06
Cairo->Seoul            14.50  -9.75 -24.125  20  -16.75  0.375  -16.75  -16.25   0.125   0.97  -0.01
Seoul->Copenhagen        9.12 -14.00 -21.750  20  -13.75 -0.125  -13.75  -12.25    -2.0   0.89   0.15
Copenhagen->Stockholm    6.12  -7.25 -12.500  20    -9.0 -0.125    -9.0    -5.5    -2.0   0.61   0.22
Stockholm->Brussels     13.88 -12.00 -25.750  20  -18.25    0.0  -18.25 -17.875  -0.375   0.98   0.02
P-A1 |Delta - cell| <= 0.5: 18/18;  median sigma_Phi 0.970, median sigma_E 0.046 over 18 pairs with |Delta| >= 1
```

Headline: median feature share 0.970, median error share 0.046 over 18 pairs (worst reader-side
pair Copenhagen->Stockholm, 0.61 / 0.22). `|Delta - cell| <= 0.5` on 18/18 pairs. The owner's
prediction P-A2 (error share >= 0.25) failed.

This example runs the first `MDB_NPAIRS` (default 6) of these pairs, in this order; per-pair
`base`, `Delta`, `Phi` and `E` should agree within **0.5 logits** (the bf16 floor the study used),
and `L*` should be 20 on every pair. Medians over 6 pairs differ from the 18-pair medians.

## 2. Causal error splice, 18 pairs (`job3/census.txt`)

```
(B) causal error splice vs direct error share (A = native / -direct)
  L8: median native 0.125  median |native| 0.312  |native| > 0.5 on 2/18  median |A| 6.82 (n=5)
  L14: median native 1.250  median |native| 1.250  |native| > 0.5 on 15/18  median |A| 9.04 (n=14)
  L18: median native -0.375  median |native| 0.562  |native| > 0.5 on 9/18  median |A| 2.93 (n=17)
  L20: median native 0.125  median |native| 0.625  |native| > 0.5 on 9/18  median |A| 1.49 (n=16)
  L22: median native 0.125  median |native| 0.812  |native| > 0.5 on 12/18  median |A| 0.51 (n=18)
  L24: median native 0.312  median |native| 0.500  |native| > 0.5 on 9/18  median |A| 0.54 (n=18)
```

A = native change / minus the error's direct effect. The example splices at L8, L14, L18, L22 only
and, with 6 pairs, the medians will move; the pattern to check is that the L14 splice moves the
margin the most (median |native| 1.25 over 18 pairs).

## 3. One prompt pair: recall prompt `The Eiffel Tower is located in the city of` (job-702270785e57)

Consumer `M[" Paris", " Rome"]` = `M[12095,21718]`, base margin 9.5.

| Quantity | Measured | Tolerance in the example |
|---|---|---|
| base margin | 9.5 | 0.25 |
| largest direct-effect feature at L22 | f1077, direct +6.80 (activation 575.13) | id must equal 1077 |
| ablate f1077 at L22 (`feature[1077,coef]*-1`), native | -1.75 (exact chain -1.857) | 0.5 |
| same edit, sealed local-linear estimate | +0.2955 | 0.1 |
| splice error at L14 on this prompt, native | -0.5 (linear -0.169, exact -0.527) | 0.5 |
| replace every layer by its reconstruction (27 splices), margin | -3.2344, top token `-wide` | 0.5, token equal |

The follow-up prompt (`... Paris, It is the capital of`, margin 8.25) and the dose curves,
swap handles and unit matching in job-702270785e57 / job-63034ebe0134 are not run by this example.

## 4. SAELens and the first-row caveat (not part of the original job)

Added by this example; no original number. `to_saelens()` loads strictly; at L20 the SAELens active
set must equal mdb's (`FeatureBasis.select`) on rows 3 and 0 of `The city of Paris is the capital of`,
and row 0 (the attention-sink row, filtered from Qwen-Scope's training) must have the larger
relative error `||e|| / ||x||`. TP1 measured the t0 relative error at 1.26 at L4, 0.87 at L8 and 0.83
at L14, against 0.27 to 0.41 for other rows.

## Validation run of this example

One mrun job, `job-4898fc946910` (Beast, bf16 eager, `MDB_NPAIRS=6`, priority 70, revision
`a66ec512`): succeeded in 202 s wall, peak VRAM 3808 MB, peak RSS 6264 MB. Log:
`validation-job-4898fc946910.log`. 13 of 13 value checks passed:

- The six census pairs equal the original rows exactly (base, Delta, Phi, E; for example
  Rome->Berlin 9.0, -14.375, -14.125, -1.25), L* = 20 on every pair, `|Delta - cell| <= 0.5` on 6/6.
  Medians over these 6 pairs: sigma_Phi 0.932, sigma_E 0.068 (the 18-pair medians are 0.970 and 0.046).
- Recall prompt: base 9.5; largest direct feature at L22 is f1077 (+6.8046); ablating it moves the margin
  by -1.75 native (sealed linear +0.2979, exact -1.7521); error splice at L14 -0.5 (L22 -0.375);
  replacement model margin -3.2344, top token `-wide`. All equal the original.
- Splice medians over the 6 pairs: median |native| 0.25 (L8), 1.25 (L14), 0.5625 (L18), 0.9375 (L22).

Differences from the original: only the SAELens check (section 4) was skipped in this job, because
`einops` was not on the path (`sae_lens not installed: No module named 'einops'`); `run.sh` now adds
`site-tl440-nn082`, which holds it. See the follow-up line below.

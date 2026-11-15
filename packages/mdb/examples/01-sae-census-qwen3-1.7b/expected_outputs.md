# Expected outputs: SAE census on Qwen3-1.7B-Base

What the original experiment measured (single seed, not independently audited). Model
Qwen3-1.7B-Base, bf16, eager attention, one CUDA GPU; SAE Qwen-Scope
`SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` (k = 50, revision `ce1a79d9c5163932d65c417380e53230e1086370`).

The original run had four parts: the census over the 18 pairs (run as two halves of 9), an error
splice / steering / unit matching / full replacement chain on one prompt pair, and entity-feature
swaps (not reproduced by this example).

## 1. Variable census, reader side, 18 city pairs

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
check |Delta - cell| <= 0.5: 18/18;  median sigma_Phi 0.970, median sigma_E 0.046 over 18 pairs with |Delta| >= 1
```

Headline: median feature share 0.970, median error share 0.046 over 18 pairs (worst reader-side
pair Copenhagen->Stockholm, 0.61 / 0.22). `|Delta - cell| <= 0.5` on 18/18 pairs. The
pre-stated prediction that the error share would be at least 0.25 failed.

This example runs the first `MDB_NPAIRS` (default 6) of these pairs, in this order; per-pair
`base`, `Delta`, `Phi` and `E` should agree within **0.5 logits** (the bf16 floor the study used),
and `L*` should be 20 on every pair. Medians over 6 pairs differ from the 18-pair medians.

## 2. Causal error splice, 18 pairs

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

## 3. One prompt pair: recall prompt `The Eiffel Tower is located in the city of` 

Consumer `M[" Paris", " Rome"]` = `M[12095,21718]`, base margin 9.5.

| Quantity | Measured | Tolerance in the example |
|---|---|---|
| base margin | 9.5 | 0.25 |
| largest direct-effect feature at L22 | f1077, direct +6.80 (activation 575.13) | id must equal 1077 |
| ablate f1077 at L22 (`feature[1077,coef]*-1`), native | -1.75 (exact chain -1.857) | 0.5 |
| same edit, sealed local-linear estimate | +0.2955 | 0.1 |
| splice error at L14 on this prompt, native | -0.5 (linear -0.169, exact -0.527) | 0.5 |
| replace every layer by its reconstruction (27 splices), margin | -3.2344, top token `-wide` | 0.5, token equal |

The follow-up prompt (`... Paris, It is the capital of`, margin 8.25) and the dose curves, swap
handles and unit matching in the original run are not run by this example.

## 4. SAELens and the first-row caveat (not part of the original run)

Added by this example; no original number. `to_saelens()` loads strictly; at L20 the SAELens active
set must equal mdb's (`FeatureBasis.select`) on rows 3 and 0 of `The city of Paris is the capital of`,
and row 0 (the attention-sink row, filtered from Qwen-Scope's training) must have the larger
relative error `||e|| / ||x||`. The original run measured the row-0 relative error at 1.26 at L4, 0.87 at L8 and 0.83
at L14, against 0.27 to 0.41 for other rows.

## Validation run of this example

One job on one CUDA GPU (RTX 4080) through our job scheduler (bf16 eager, `MDB_NPAIRS=6`): succeeded
in 149 s wall, peak VRAM 4580 MB, peak RSS 6359 MB. Validation log: `validation-job-6df33e74e5cb.log`
in this directory. 21 of 21 checks passed:

- The six census pairs equal the original rows (base, Delta, Phi, E within 0.5; in practice exact), L* = 20, `|Delta - cell| <= 0.5` on 6/6.
- Source side (L4, L10, L16, full / Phi / E minus null) for the same six pairs is within 0.5 of the original on every cell.
- Recall prompt: base 9.5; top direct feature f1077; ablating it -1.75 native; error splice at L14 -0.5;
  replacement model -3.2344, top token `-wide`.
- SAELens (strict `to_saelens()`, sae-lens 6.55.5): the active set equals mdb's on rows 3 and 0 at L20; the relative
  error is 0.310 on row 3 and 0.454 on row 0 (the t0 caveat holds, with a smaller gap than the original run's L4 value of 1.26).

An earlier run (before the source arm and the einops path fix) matched the same reader and recall numbers
and skipped the SAELens check.

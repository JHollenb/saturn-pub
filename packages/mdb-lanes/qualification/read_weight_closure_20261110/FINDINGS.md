# A prospective Qwen state span closes the native-A dependency

The functional native-profile producer reproduced the unedited and combined-edit branches exactly in all eight tested receivers, including full-logit signatures and cursor/token/logit/cache closure after continuation, with exact captured span residuals. The closure hash does not include the final residual tensor. The consumer actually replaced layers 19–22 on the pending token. This closes dependence on edited native-A attention output and input22 for this bounded experiment. The replacement still pays dense arithmetic: it is an executable computational account, with no demonstrated full-model speedup.

Recentring the six-head quadratic at dose 0.5 also repaired the large finite-dose Taylor counterexample. Full-dose mean log-change RMSE fell from 0.02824482 to 0.01415301, with 7/8 receiver improvements. Negative-dose error worsened on average, fresh future-logit errors reversed the local ranking, and compilation remained slower than direct smooth evaluation through 32 same-state queries. These countertrends constrain what the result means.

## Specimen and construction

Job `job-5fc199bd25fe` ran Qwen2.5-1.5B base, CUDA BF16 native-eager, on Beast under one strict-preflight mrun lease. It completed 340 token transitions and 4 prefills/88 prefill tokens in 269.49 worker seconds. There were 192 replaced layer transitions and 9,328 native layer transitions; the token counts exclude prefill arithmetic. Layers 0–18 and 23+ on the tested token, and all later tokens, remain native.

Source A is the historical first generated answer row at L19/KV0. Reader B is the current follow-up copula row at L22/KV0. Names locate supplied physical row roles; this is not automatic semantic circuit discovery. Nora/Ethan and Ivy/Ivy are development cases selected from the November 7 diagnosis. Clara/Leo and Maya/Maya are frozen fresh lexical cases. Each case contributes a first receiver and a second receiver descended from the declared raw-midpoint branch's own end cut. The two fresh contexts provide four related receiver observations, not four independent contexts.

The [frozen plan](PLAN.md), [worker](worker.py) and [launcher](launch.py) retain zero, negative and positive doses `0,-0.25,0.2,0.6,1`. Every scalar law and full candidate span was sealed before edited native A or AB scoring. Saved remainder diagnostics are evaluated after sealing. Direct smooth evaluations in the earlier cost benchmark do not supply the coefficients. Functional native-profile predictions are prospective. Full unedited native00 state and resident parameters are allowed operands.

## The two error sources

For the unchanged native-anchored smooth log-weight path `g`, the new law is

\[
T_m(\lambda)=g(m)+(\lambda-m)g'(m)+\tfrac12(\lambda-m)^2g''(m),\quad m=0.5.
\]

The baseline-corrected arm is `g(0)+T_m(λ)−T_m(0)`. Subtracting this constant preserves the polynomial's derivatives and restores its zero value. It does not necessarily improve finite-dose error. The original zero-centered law, stale weights, wrong sign and wrong curvature remain visible controls. The raw midpoint zero residual has mean receiver RMSE 0.00269667, maximum receiver RMSE 0.01411928 and maximum absolute head residual 0.03109784; it is not silently removed.

Let `S` be the full smooth response and `N` the observed native response. Per head,

\[
T-N=(T-S)+(S-N),\qquad
\operatorname{MSE}(T-N)=\operatorname{MSE}(T-S)+\operatorname{MSE}(S-N)+2\mathbb E[(T-S)(S-N)].
\]

The report retains signed terms and an independently recomputed FP64 identity. Remainder and profile RMSEs cannot be added. Recentring changes the first error term; matching the native numerical program addresses the second.

| Receiver | Split | Zero-centered RMSE at 1 | Midpoint RMSE at 1 |
|---|---|---:|---:|
| Nora/Ethan first | Development |0.10127361|0.01638357|
| Nora/Ethan own follow-up | Development |0.01444169|0.00751339|
| Ivy/Ivy first | Development |0.01907149|0.01358871|
| Ivy/Ivy own follow-up | Development |0.00935748|0.00929253|
| Clara/Leo first | Fresh |0.01460271|0.01388448|
| Clara/Leo own follow-up | Fresh |0.02012810|0.02066695|
| Maya/Maya first | Fresh |0.03734327|0.02215657|
| Maya/Maya own follow-up | Fresh |0.00974019|0.00973792|

Development means are 0.03603607→0.01169455 (4/4 improvements); fresh means are 0.02045357→0.01661148 (3/4). The combined mean falls 49.89%. Baseline-corrected mean is 0.01594952, worse than raw midpoint in 5/8. Midpoint beats stale and both sign/curvature controls in 8/8 at full dose.

The Nora/Ethan first-receiver smooth remainder falls 0.09766378→0.00946609, explaining its repair. Ivy's later remainder falls 0.00078469→0.00010108 while total error barely changes: its smooth/native discrepancy dominates. Clara's later remainder also shrinks, yet native error worsens because the former Taylor error partly cancelled the profile discrepancy.

| Dose | Zero-centered mean RMSE | Midpoint mean RMSE | Midpoint improvements |
|---:|---:|---:|---:|
|-0.25|0.01120056|0.01543123|4/8|
|0|0|0.00269667|0/8|
|0.2|0.00956465|0.00963855|6/8|
|0.6|0.01281925|0.01158495|7/8|
|1|0.02824482|0.01415301|7/8|

At negative dose, all four development receivers worsen and all four fresh receivers improve against native measurements. Smooth truncation worsens in both splits. Those fresh native improvements reflect error cancellation, not reliable midpoint extrapolation. Across 32 nonzero receiver/dose pairs, midpoint improves in 24 and worsens in 8. Zero-dose observations are retained separately.

## Complete native-profile production and consumption

`NativeQwenReadTransport` starts at native00 input19 and a locally reconstructed attention19 write. It uses resident dense parameters, old KV20–22 and current RoPE. Its functional BF16 program mirrors normalization accumulation/casts, projections, positioning, score/softmax arithmetic, attention layout, residuals and SwiGLU. Original model module forwards are forbidden during production. Direct AB zeros the selected historical V22 row before AV and output projection.

Functional predictions match selected native log weights in 40/40 receiver/dose pairs. All compared input/attention/MLP/exit/Q/K/V fields are exact. The native-profile program qualifies collectively; this does not isolate one particular cast as the cause of the former discrepancy.

| Measured closure | Exact receivers |
|---|---:|
| Functional zero vs native00 complete closure |8/8|
| Functional AB vs nativeAB complete closure |8/8|
| Functional AB vs nativeAB full-logit signature |8/8|

The bridge binds parent, pending token, sites, context length, layer cursor, dtype and device. It checks the consumed incoming hidden tensor and installs predicted exit/current KV before advancing. Actual `installed_*` captures, rather than forecast declarations, identify what was written. Historical A/B edits stay installed; later tokens consume the resulting branch-owned cache. Forty-eight span arms replace four layers each on their first token.

The native-profile producer pays 24 dense linear maps and three attention computations per evaluation. Its A computation already obtains exact reader weights. Substituting a Taylor weight into that paid A tape is a controlled factorization comparison; it is not an efficient end-to-end implementation. Direct functional AB is the natural dense accuracy reference.

The polynomial consumer subtracts a projected selected-value contribution from prospective A attention, then recomputes MLP22. This changes finite-precision ordering relative to native AV/O with B zeroed. Its consumer error therefore combines weight approximation with this arithmetic factorization; the assay does not isolate those contributions with an exact-native-weight subtraction arm. Exact direct AB remains the reference.

## Local accuracy does not settle the future

Mean full-dose exit relative L2 falls 0.33475%→0.22095% from zero-centered to midpoint; paired results improve 5/8, worsen 1/8 and tie 2/8. Functional-native exit error is exactly zero. Midpoint beats stale in 7/8 exit comparisons; Maya's own-follow-up stale exit is closer.

Mean full-vocabulary relative L2 over three decision offsets improves on development 0.9540%→0.7125%, but worsens on fresh receivers 0.9032%→0.9295%. Fresh stale is 0.8682%, better by this aggregate. This is a consumer-ranking reversal despite better mean local exit error. Preserve it when choosing future representations.

Every candidate method, including stale, matches nativeAB on 24/24 next decisions and 16/16 consumed generated tokens. These short discrete outcomes do not discriminate the scalar methods. The edited nativeAB branch answers the original requested object in 2/8 receivers, both Nora; fresh is 0/4. Native00 answers it in 1/8. These task errors remain visible rather than being erased by numerical agreement. Agreement with an edited native branch is continuation evidence, not chat correctness or arbitrary CoT reconstruction.

## Compilation did not amortize within 32 queries

Both benchmark paths share one prepared immutable smooth state/direction and return CPU scalars. Three warmed paired samples alternate execution order. Compilation-plus-evaluation is measured directly, rather than added from separate medians. The following are medians across receiver-level medians:

| Same-state dose queries | Direct smooth | Compile + scalar | Scalar only, compilation paid |
|---:|---:|---:|---:|
|1|1.148 ms|42.944 ms|0.0218 ms|
|5|5.694 ms|43.077 ms|0.0856 ms|
|32|35.797 ms|43.697 ms|0.4991 ms|

These exclude common preparation, baseline acquisition and native consumer/custody work. Warming matters: preparation median is 4.31 ms, while the first receiver is 115.46 ms. Six functional-native spans take median 11.92 ms per receiver, excluding separate setup. Arm wall times include restore, edits, capture, receipts and closure hashing and are not clean kernel benchmarks.

There are 16 primary derivative compiles plus 80 benchmark compiles, 936 direct smooth benchmark calls plus 40 diagnostic calls, 48 native-profile spans and 96 extra consumer-MLP linears. The compact polynomial is cheap once compiled; producing it for each changed state is still costly. A simple linear timing extrapolation suggests roughly 40 same-state dose queries might repay compilation, but no tested K demonstrated payback and no reuse across changing histories was tested.

## Reproduction, custody and remaining question

The [MDB guide](../../docs/READ-WEIGHT-CLOSURE.md) documents the public compiler and native-profile APIs. The [measured fixture](DEMO-PROGRAM.json) and [framework-free demo](../../examples/read_weight_centered_demo.py) replay the sealed Nora/Ethan midpoint program. At dose 1 the first head predicts `Δlogp=-0.04416469` and `p=0.74966384`; at zero its raw residual is 0.00293152. Dense preparation is already paid.

The [offline verifier](verify.py) independently audits saved scalar evaluation, payload hashes, source reconstruction, forecast order, exact installed residual/KV, lineage, receipts, numerical measurements and continuation signatures. It does not independently repeat the trained derivative compilation. [RESULTS.json](RESULTS.json), [AUDIT.json](AUDIT.json) and [VALIDATION.json](VALIDATION.json) retain compact results and bindings. Raw outputs remain immutable in the collected artifact tree. Mechanics, instrument, trend and terminal status remain separate; terminal status is not assessed.

The 25 focused tests and all new-source Ruff checks pass. The measured scalar demo runs outside the workspace from installed public wheels with Torch, NumPy and private imports blocked. After qualification, the native API's identity guard was hardened against live parameter replacement and numerical-policy changes, with tiny-model tests; the equations are unchanged. The trained run remains bound to its original frozen wheel, and [VALIDATION.json](VALIDATION.json) records both source hashes. No second trained run is implied by that later guard change.

Report SHA256: `f16573c426234c50edd355b4b466ef0dc7949650ca2fb5fb7407062dca3152ba`. Archive SHA256: `9b7271ff5e485242dabfe8922273865d823d5899ac7ce9bf835f0af5331216a6`. The archive contains 34,056 manifest-listed files, 421,255,647 expanded bytes, with no pretrained-weight copies.

The next mathematical question is whether a fixed value-sampled law of the qualified finite-profile response can avoid nested-differentiation overhead. A proposed prospective test fits a quadratic from baseline plus two predetermined interior functional samples, then scores unqueried doses including full dose and negative extrapolation. Finite-profile interpolation, smooth interpolation, an affine law, midpoint Taylor and direct evaluation should be compared on fresh contexts. Two dense samples still cost work at every state. The larger open problem remains economical operand and complete-state production across changing histories and writes.

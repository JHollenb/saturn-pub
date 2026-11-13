# Prospective second-order six-head read weights

**Observation:** direct second derivatives improve the full-dose mean log-change RMSE from `0.06964296` (first-order Q/K transport plus exact normalization) to `0.02843922`, a **59.16% reduction**. Full curvature wins in 7/8 receivers over both transport and routing-only curvature, and 6/8 over a linear log-weight law. One full-dose receiver gets worse than stale. These are paired exploratory observations; terminal status is **not assessed**.

The public [math/API guide](../../docs/READ-WEIGHT-TAYLOR.md) and [measured offline demo](../../examples/read_weight_taylor_demo.py) make the construction executable. The [frozen plan](PLAN.md), [worker](worker.py), [launcher](launch.py), [results](RESULTS.json), [audit](AUDIT.json), and [validation bindings](VALIDATION.json) separate prospective forecasts, native target measurements, continuation and custody. This extends the [first-order experiment](../read_weight_transport_20261107/FINDINGS.md); its eight old receivers were development evidence, not new holdout.

## Specimen and computation

Run `job-df06c7aeda6d`, request `1318cb5f3fb11cb6508ccdceccbb27d6541cfe4922358821550cadffa998c441`, used one resident Qwen2.5-1.5B BF16 model, CUDA eager attention, TF32 off. Four fixed fresh lexical contexts crossed Nora/Ethan pen/hat and Ivy/Owen ball/watch with the preceding owner. Each contributed a first follow-up and a subsequent receiver from the **second-order candidate's own** end cut: eight receivers, histories of 31 and 39 rows. No context was selected for correctness or effect size. A=L19/KV0/first generated answer row; B=L22/KV0/current-follow-up copula. Physical ports and row-role constructors remain supplied.

The smooth query/current-key displacement is anchored to native baseline Q/K and native BF16 scores widened to FP32. Two nested directional derivatives compile `g(0)`, `c1=g'(0)`, and `c2=g''(0)` for the six selected B log weights. For score derivatives `a` and `u`, `c2 = u_B − E_p[u] − Var_p(a)`. Routing-only curvature retains the current Q/K product term and normalizer, but omits upstream query/key second derivatives. Full-minus-routing isolates that centered upstream contribution.

All seven methods at doses `0, -0.25, 0.2, 0.6, 1` were sealed before any edited native receiver ran. Negative dose scales A's stored value to 1.25 times baseline. Original native module forwards were forbidden during compilation. Source attention reconstruction matched all **40/40** native receiver/dose pairs. Baseline-only independent c1 formulations differed by at most `1.1920929e-7`. This differentiates known dense equations; it does not discover a universal semantic program.

## Full evidence, including losses

| Dose | Mean second-order RMSE | Mean transport RMSE | Mean routing-only RMSE | Mean linear-logweight RMSE | Second vs transport |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -0.25 | 0.01118962 | 0.01193717 | 0.01192716 | 0.01201173 | 5/8 |
| 0.2 | 0.01092096 | 0.01157302 | 0.01157832 | 0.01158962 | 6/8 |
| 0.6 | 0.01334231 | 0.02777103 | 0.02799050 | 0.02611896 | 6/8 |
| 1.0 | 0.02843922 | 0.06964296 | 0.07008563 | 0.06538982 | 7/8 |

Across 32 nonzero receiver/dose pairs, second order wins **24/32** over transport, **24/32** over routing-only, **25/32** over linear log weight, **29/32** over stale, **29/32** over wrong curvature and **32/32** over wrong sign. Zero-dose rows remain visible and excluded from win counts. All 48 full-dose head signs were correct. No predicted selected probability left `[0,1]` in this panel; this polynomial has no general probability-range guarantee.

The important counterexample is **Nora/Ethan, first receiver**: second-order RMSE `0.10127361`, transport `0.06575084`, linear `0.09400325`, stale `0.09873340`. Direction is right but magnitude is not improved. Ivy/Owen's own-history receiver also slightly favors linear (`0.02708503`) over second order (`0.02724262`). Higher order is not automatically better at finite dose; small-dose comparisons are mixed and retain their native finite-precision noise.

### Two different remainder sources

The saved-only [remainder diagnostic](remainder_diagnostic.py) constructs a full nonlinear **paid** smooth displacement from already-saved Q/K control states, using the same native anchors. For the sealed polynomial `T`, anchored full smooth value `S`, and native log weight `N`, it verifies the per-head FP64 identity `T − N = (T − S) + (S − N)`. All eight decompositions close exactly; 64 CAS payloads are hash/size/ref verified. This is post-hoc diagnosis, not another prospective forecast or free nonlinear compiler.

| Full-dose receiver | Total RMSE `T − N` | Taylor remainder `T − S` | Smooth/native profile `S − N` |
| --- | ---: | ---: | ---: |
| Nora/Ethan first | 0.10127362 | 0.09766357 | 0.01148128 |
| Ivy/Owen own second | 0.02724260 | 0.00038066 | 0.02710009 |
| Ivy/Ivy own second | 0.00935745 | 0.00078508 | 0.00928669 |

RMS terms are not additive percentages: the exact MSE includes their signed cross term. Nora/Ethan head 5 contributes total error `+0.214654`, decomposed as Taylor `+0.213342` and profile `+0.001312`. Ivy/Owen own-second head 2 contributes `−0.065514`, with Taylor `−0.000452` and profile `−0.065062`. A smaller-interval/recentered or higher-order smooth approximation targets the former; the latter requires matching the native numerical profile. The next useful cut should distinguish these costs rather than adding Taylor order everywhere.

Diagnostic SHA256 `aea61ed9be1d784360f36984f52aec73954fedff28fba5435f96634356290f98`, script SHA256 `88ea2ab244457ab65d3abaf7a0ca9a402502db19ffff3dd298b06a7271deccd3`; repeated serialization is byte-identical. Its raw rows live at `run-01/analysis/taylor-remainder-diagnostic.json` beside the stdlib coefficient summary `second-order-diagnostic.json`.

## Actual consumption and future

At full dose, predicted B weights form the AB value deletion from the **paid native A attention output and input22**, followed by a dense MLP22 forecast. Five injection branches (second order, linear, transport, stale, oracle) and native AB then run the unchanged native suffix. Native attention still executes before output replacement.

| Local boundary | Mean second-order relative L2 | Mean transport relative L2 | Mean stale relative L2 | Second-order range |
| --- | ---: | ---: | ---: | ---: |
| Attention | 1.4168% | 2.9433% | 7.8068% | 0.3646–6.4137% |
| MLP | 0.7289% | 1.3828% | 3.3925% | 0.4201–2.1219% |
| Block exit | 0.3143% | 0.5054% | 1.1082% | 0.1628–0.9304% |

Second order beats stale at all three boundaries in **8/8** receivers, but beats transport and linear in only **5/8** at each boundary. Six-head scalar RMSE and the error after value projection weight different directions. Injection MLP/exit consumption is exact relative to each injected forecast, not relative to native AB.

All five methods agree with native AB on **24/24 next-token decisions** and **16/16 consumed generated tokens**. Stale does too, so these discrete decisions do not distinguish coefficient quality. Direct requested-object accuracy is **6/8 for native AB and every forecast**. In Ivy/Ivy's first receiver the request is ball but the model says watch; its second request is watch but it says ball. Both mistakes remain in the record. Nora/Ethan's first receiver changes from baseline `same` to `pen` at doses 0.6 and 1; all full-dose forecasts, including stale, follow that native decision. No free-form semantic or long-horizon autonomy claim follows.

## Costs, reproduction and custody

The primary program has 18 numbers, **72 bytes packed FP32**, with exact packed round-trip. The actual minimal demo program JSON is **461 bytes**; full Taylor records including diagnostics and metadata are **1222–1263 bytes**. Comparator score tables, saved native tapes, dense parameters, all forecasts, and custody are additional. The current API returns Python float arrays; 72 bytes is not Python object or serialized JSON size.

The smooth path has 19 logical dense maps; the run pays their derivative arithmetic, native anchoring, a separately measured first-order comparator, five source O projections and 15 downstream MLP linears per receiver. Warm producer bundles took **56.836–61.151 ms**, first bundle **344.436 ms**. Thirty-five method/dose evaluations, including conversion back to device tensors and synchronization, took **1.534–1.590 ms** warm, first **11.214 ms**. These single samples are not a serving benchmark or amortization claim.

One strict mrun lease completed in **321.62 s**, with **316 native token transitions**, **4 prefills / 88 tokens**, peak RSS **2454.9 MB**, peak VRAM **4060 MB**. Automatic reservations were RAM 6963.1 MB / VRAM 4946.2 MB / 4 CPU threads. There were no retries or nested jobs. Raw custody contains **29,840 files / 440,032,618 expanded bytes**, archive **330,603,983 bytes**, with pretrained parameters represented by their retrieval manifest rather than copied weights.

Raw run: `artifacts/mdb-read-weight-second-order-20261107/run-01` in the domains workspace. Report SHA256 `03bd777d3f2711b378becfb4a71d25831b0484ed6c9808bb32e3fc61061fa281`; archive SHA256 `2d6b6818098c51ba3106f5fc96124fc3797e706dd8e52b642ac9239bf14aba81`; raw manifest SHA256 `93fc269092d7e6bb52c28dfc17fc64dee9a36d204dc5a87f4869907be4e5790e`. Final independent v2 audit SHA256 `38c168e61c45098e3bc7da959b12e24188593d8d5d2b19eee4cc602a42762a3d`; the earlier audit remains preserved separately.

The offline audit validates all eight seals, independently rebuilt native-anchor score tables, scalar reconstructions, routing identities, CAS payloads, native receipt/history authority, source locality, controls, consumer injections and candidate ancestry. It does **not** independently re-execute the trained nested derivative compiler. Tiny-model finite differences and the native integration test validate that instrument separately. Root regression checks: **79 passed, 9 skipped**, plus **1** native qualification/tamper test, repeated after the v2 audit change. Installed public wheels evaluate scalar programs outside the workspace with Torch, NumPy and private packages blocked. A post-audit rehash checks the entire original raw inventory. The standard evidence-plane run and verify return zero over four projected streams / 2,200 rows with zero fires; this is explicitly `target-streams-only`, not an audit of the whole model.

**Working inference:** upstream curvature is useful and prospectively compilable into the consumer-selected scalars, with a material finite-dose counterexample. Economical recompilation on changing states, baseline attention-output production, full-logit generation, wider contexts/models and arbitrary chat remain open. This is a local differential response law with explicit state and cost, not closure of the whole model.

The smallest next experiment is a fixed midpoint expansion of the same native-anchored scalar function: `g(0.5) + (λ−0.5)g'(0.5) + 0.5(λ−0.5)^2g''(0.5)`, sealed before edited native receivers. Keep the old counterexample as development evidence, add fresh contexts, retain the negative dose as extrapolation, and account for the extra dense smooth compilation. This tests changing curvature along the path. Treat matching the Ivy smooth/native profile as a separate problem; it is not solely a claim about BF16 rounding.

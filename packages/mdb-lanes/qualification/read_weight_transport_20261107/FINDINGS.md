# Predict B's changed weights before observing the edited reader

The baseline-only directional compiler improved selected B log-weight
predictions over stale weights in all 24 nonzero-dose receiver comparisons.
The four fresh Qwen2.5-1.5B contexts each supplied an initial receiver and a
later receiver on the transport candidate's own generated history. No edited
native downstream receiver was visible when the weight forecasts sealed.

This removes the observed-A-receiver dependency for the tested weight
producer. It does not remove dense baseline execution, per-receiver
compilation, the consumer's later native A output/input state, or downstream
native execution. All methods match the same 24 native AB next-token
decisions, including a retained wrong direct-object answer.

Sources: [prospective plan](PLAN.md), [public producer](../../src/mdb/read_weight_transport.py),
[interface and math](../../docs/READ-WEIGHT-TRANSPORT.md), [worker](worker.py),
[compact results](RESULTS.json), [independent audit](AUDIT.json), and
[validation binding](VALIDATION.json). The
[blog post](../../../../../obsidian/blog/2026-11-07-013741-predicting-a-changed-read-before-the-reader-runs.md)
connects the result to MLP transport, time circuits and prior read composition.

## Fixed construction and visibility

A is L19/KV0 at the first consumed generated row of the preceding answer
prefix. B is L22/KV0 at the designated copula in the current follow-up.
Physical sites were nominated by earlier discovery; row-role constructors
remain supplied. No semantic operand role is inferred from a source word.

The compiler sees the complete unedited 00 tape through L22, original
parameters, and the locally formed preceding A write. An FP32 functional lift
executes MLP19, blocks20–21 and normalized/rotated Q/current-K22. One JVP
compiles the write direction. Native baseline Q/K anchor its displacement.
All current K/V changes in the intervening blocks are included.

The score program retains scalar arrays only. For each head reading B,

\[
\widehat{\delta\ell_j}(\lambda)
=\lambda a_j+\mathbf1_{j=t}\lambda^2b_t,
\quad
\widehat{\Delta\log p_B}
=\widehat{\delta\ell_B}-\log\sum_jp_j^0e^{\widehat{\delta\ell_j}}.
\]

The append coefficient includes both query and key changes and their cross
term. Baseline log-probabilities come from widened native score logits.
Normalization is exact for these predicted scores; upstream transport is
approximate. BF16 execution itself is not claimed to be differentiable.

All four doses, 0/quarter/half/full, seal before any nonzero-A native receiver.
Only afterward does the worker obtain native A output/input22 for separate
consumer forecasts. Those seal before injected replay and native AB. A paid
full FP32 control also executes after the weight seal. No outcome is used to
fit, tune or select the original construction.

## Results across doses

Six query heads read B's KV head. Four fresh cases cross Mira/Daniel with
key/coin and Zoe/Liam with book/cup, each with same/other preceding owner.
The two invocations yield eight receiver occurrences with 31–41 visible rows.
They are related contexts, not independent task families or long-context tests.

| Dose | Transport log-change RMSE | Stale RMSE | Per-receiver stale/transport |
|---|---:|---:|---:|
| 0.25 | 0.00635–0.01851 | 0.01784–0.09079 | 1.76–11.37× |
| 0.5 | 0.00919–0.03947 | 0.02650–0.17867 | 2.88–11.17× |
| 1 | 0.01812–0.12918 | 0.05627–0.40032 | 2.47–7.56× |

All 24 nonzero-dose comparisons improve over stale and wrong-sign. Transport
improves over query-only in 20/24, but over the unconstrained linear-logweight
foil in only 12/24. At full dose, transport beats that foil in 3/8. The exact
normalizer does not uniformly overcome upstream trajectory error.

Full-dose transport predicts 45/48 signed head changes correctly. Two misses
have small negative true changes; Zoe/Zoe first head offset2 has the more
substantial true change -0.08791 versus predicted +0.00120. Head rows remain
visible. Local source-write formation is exact for all 32 receiver/dose pairs.

Each full-dose transport consumer improves attention/MLP/exit error over stale:

| Relative L2 | Transport | Stale | Observed-A-weight control |
|---|---:|---:|---:|
| Attention | 0.91–7.72% | 2.53–21.77% | 0.30–0.49% |
| MLP | 0.59–4.20% | 1.18–11.70% | 0.35–0.76% |
| Exit | 0.21–1.29% | 0.61–3.65% | 0.13–0.33% |

Both forecasts and the control methods match all 24 AB next-token decisions
and all 16 subsequently consumed generated tokens per method. All share AB's
7/8 direct-object matches. Zoe/Zoe first incorrectly begins with `cup` when
asked for Zoe's book. In other first-turn cases A changes `same` to the object
noun; the short `same as…` prefix does not establish an eventually wrong
free-form answer. Numerical emulation and semantic correctness stay distinct.

## Diagnose remaining transport error

Tangent Q/K relative L2 errors are 0.491–9.520% / 0.530–9.012%. Complete
FP32 control errors are 0.294–0.393% / 0.319–0.453%. The saved-only
[diagnostic](diagnose.py) decomposes native-anchored tangent error into

\[
\mathrm{error}=-R+E,
\quad R=F_s(A)-F_s(0)-Jd,
\quad E=(F_s(A)-F_n(A))-(F_s(0)-F_n(0)).
\]

All 16 Q/K decompositions close exactly in FP64. In seven receivers, the
smooth nonlinear remainder's norm is near the entire error norm, with a
smaller finite-profile difference. Zoe/Zoe second is the exception, where
finite-profile mismatch dominates an already small error. Norm ratios are
not additive causal percentages.

A post-hoc construction using saved complete nonlinear smooth displacement
and native anchors lowers selected log-weight RMSE to 0.0060–0.01855 in all
eight receivers. It is a paid-control diagnostic, not a new sealed forecast.
Its 96 input payloads, hashes and per-receiver rows are retained in the
[diagnostic output](../../../../../artifacts/mdb-read-weight-transport-20261107/run-01/analysis/transport-remainder-diagnostic.json).

## Cost and integrity

Each compiled serving program contains 2,280–3,000 tensor bytes, including
diagnostic comparator coefficients. Dose evaluation uses scalar operations
and complete normalizer reductions, without original modules, learned
matrices or new Q/K products. These coefficients apply to one receiver and
write direction. Every subsequent state pays a fresh compilation.

Per receiver, setup uses 19 logical dense maps plus tangent work and four
local source output projections. A complete FP32 control adds 19 maps after
sealing; consumer forecasts add 12 MLP linears. A single synchronized run gives
warm producer times 12.4–13.2 ms and 20 scalar evaluations in 0.794–0.810 ms; first
receiver times 477.8/15.6 ms retain initialization effects. No speed benchmark
or setup-inclusive amortization claim follows.

mrun job `job-466e6e302535` succeeds in 283.23 s with four prefills, 286 physical
token-step calls, peak RSS 2282.8 MB and peak VRAM 4060 MB. Model loading is
resident, CUDA/native-eager BF16, one finite lease and no retries. Parameter
constants stay recipe-backed.

| Binding | SHA-256 |
|---|---|
| Request | `1a43f454b6f73233928a16dfc4e069bccfbddc22dc6ce8da4d33dbe03d107341` |
| Weight file | `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796` |
| Report | `24c43074d80cc4d0ea6001d9ed5ca78bd7ec73b32aedc627502be880e9f24397` |
| Archive | `648fe84b08b68dda38027c938d6a891b71b09d90ffd9c921d40c662833524ae4` |
| Raw manifest | `c996a4459c185c1a1fca16d0f909c5203bd5274c02a95e23e8ad86efa47c86ac` |

The collector verifies 25,296 files / 382,596,151 expanded bytes. Compressed
archive size is 290,476,275 bytes. The independent verifier validates four
Bench closures and eight receiver bundles, recomputes score coefficients and
all dose forecasts, checks CAS integrity, source locality, native histories,
consumer bytes and seal ordering. It does not independently rerun the trained
derivative compiler. JVP construction is separately unit-tested against
finite differences and reviewed against the explicit equations.

The focused suite passes 74 tests. The installed-wheel proof runs outside the
workspace with private imports blocked and framework-free import verified.
The raw run and full offline audit remain in
[run-01](../../../../../artifacts/mdb-read-weight-transport-20261107/run-01/report.json).
Compact publication records bind the full audit and diagnostic hashes rather
than copying all repeated scalar rows into Git.

A post-audit rehash finds every original raw file unchanged, with no extra or
missing files. The Saturn evidence run and verification both return zero on
four scalar streams / 1,312 observations, with no fires and
`target-streams-only` status. That sweep does not independently validate native
tensors or turn the exploratory measurements into a semantic terminal claim.

Offline reproduction requires no model load:

```sh
python packages/mdb/qualification/read_weight_transport_20261107/verify.py /path/to/run-01
python packages/mdb/qualification/read_weight_transport_20261107/analyze.py /path/to/run-01 summary.json --compact-output RESULTS.json
python packages/mdb/qualification/read_weight_transport_20261107/diagnose.py /path/to/run-01 new-diagnostic.json
```

For a new fleet run, use the qualified public environment, including when
the workspace credential wrapper is needed:

```sh
common exec /path/to/saturn-pub/.venv/bin/python packages/mdb/qualification/read_weight_transport_20261107/launch.py /path/to/workspace/artifacts/new-run
```

The default only stages; `--submit` is the explicit admission action. The
original common-Python staging attempt encountered a conflicting private mrun
distribution before any submission; that partial staging directory is
preserved separately. The actual job uses installed public runtime wheels.

## Remaining gap

The next local proposal is baseline-only second-order compilation directly
into the six selected log weights: $\lambda c_{1h}+\lambda^2c_{2h}/2$.
It targets the observed transport curvature and might reduce serving state
further. It is untested here. Economical baseline production, compilation
across new states, native A attention/input22 production and downstream
execution remain separate costs. Full arbitrary-prompt autonomous chat and
portable semantic CoT remain unqualified.

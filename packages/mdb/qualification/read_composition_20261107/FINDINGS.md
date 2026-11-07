# A discovered read composes through its changed receiver

The frozen two-port locator and receiver-conditioned arithmetic transferred to
four fresh Qwen2.5-1.5B contexts and four further receivers reached through
their own altered histories. Using the receiver produced by the first edit
reduced the second attention prediction's relative L2 error by 5.48–41.62×
versus reusing the original reader. Every receiving MLP and layer-exit
prediction also improved. Both methods nevertheless matched all 24 native AB
next-token decisions. This is a local numerical composition result; discrete
continuation did not distinguish the two forecasts in this assay.

The [prospective plan](PLAN.md), [worker](worker.py), [public frozen locator](../../src/mdb/read_program.py),
[offline verifier](verify.py), and [summary](RESULTS.json) retain the experiment.
The [blog post](../../../../../obsidian/blog/2026-11-07-000659-the-read-was-observed-before-the-future-was-tested.md)
connects it to earlier read discovery and the broader state-machine work.

## The gap tested

[The preceding discovery qualification](../CAUSAL-CIRCUIT-DISCOVERY-VALIDATION.md)
tested individual native memory reads. Execution order and independent causal
effects did not yet establish a composition rule after the first edit changes
the second reader. This assay froze the previously nominated physical sites:

| Port | Physical site | Supplied row locator |
|---|---|---|
| A | layer 19, KV head 0 | first consumed generated row of preceding answer prefix |
| B | layer 22, KV head 0 | last designated copula in current follow-up span |

Both row-role hypotheses are supplied. The resolver uses committed token
history and declared segment boundaries; it does not infer a semantic operand
from a word or decode a general chain of thought. Its declaration fingerprint
is `d8ad2a4671e12be88975e6c033c66aba64cd94883351b134e3ee785ae2be2499`.

Each edit zeros only its selected historical value head. For B's contribution,
the receiver-conditioned and stale forecasts are

\[
\widehat o_{AB}=o_A-\sum_{h:g(h)=0}p^A_{h,j_B}W_{O,h}v_{j_B},
\qquad
\widehat o^{\mathrm{stale}}_{AB}
=o_A-\sum_{h:g(h)=0}p^{00}_{h,j_B}W_{O,h}v_{j_B}.
\]

Both pay for the native A attention output. The conditional forecast also
observes A's changed query, current key/value append, and complete attention
denominator. All historical keys stay fixed; B's source value is identical
between 00 and A. A and AB have byte-identical layer-22 incoming state, query,
keys and current value. These invariants make B a value-only subtraction at
its actual changed receiver.

The forecast then recomputes the receiving RMSNorm/SwiGLU/residual program
with all native dense MLP matrices and BF16 casts. It does not add independent
MLP response directions. Forecast tensors are sealed before injected execution
or the native AB reference. Native attention still executes before the output
replacement hook: injection demonstrates consumption, not saved attention work.

## Fresh contexts and generated history

The fixed cases cross Nina/Samir with key/coin and Ada/Noah with book/cup, with
the preceding question concerning either the same or the other owner. No case
is selected by its answer quality. Each receiver bundle compares 00, sham, A,
B, wrong-row, reinstall, conditional, stale and AB from one committed parent.
Each arm consumes its own two generated tokens after the receiver transition.

The four 00/A/B/AB worlds then append the fixed second follow-up to their own
resulting histories and reapply the frozen policy. A complete new bundle is
also run on the first AB world's new parent. That supplies four more receivers.
The second bundle's 00 means no new edit on an already altered history; it is
not a comparison against the untouched first-turn world. Across-policy later
semantic observations can therefore reflect different histories.

## Measured local predictions

Relative L2 error against native AB, across all eight receivers:

| Boundary | Updated receiver | Stale receiver | Stale/updated improvement |
|---|---:|---:|---:|
| Attention return | 0.269–0.619% | 1.475–13.301% | 5.48–41.62× |
| Receiving MLP | 0.344–0.644% | 0.896–6.808% | 2.56–16.99× |
| Layer 22 exit | 0.140–0.383% | 0.373–2.033% | 2.54–11.24× |

The conditional forecast has lower error in every row at every boundary.
The remaining local error is compatible with the saved FP32 explanatory
contractions versus native BF16 contraction and reduction order; this assay
does not isolate every rounding contribution. None of these errors was used
as an exploratory pass/fail gate.

All eight sham and eight reinstall controls are exact. All sixteen explicitly
predicted MLP and exit tensors match their actual injected native consumption
byte-for-byte. These exact consumption checks do not make the forecast equal
to native AB. Native AB local outputs and both forecasts differ numerically.

## What the native future tells us

Both forecasts match native AB on all 24 next-token decisions and all 16
generated tokens subsequently consumed per method. Full logits remain
approximate, and conditional improves full-logit L2 in only 12/24 comparisons.
The strong local numerical improvement has not demonstrated superior discrete
continuation here.

One fresh witness provides a recurring causal first-token effect. With
Nina/Samir and the preceding question about Samir, 00, B and wrong-row next
produce ` same`; A, AB and both forecasts produce ` key`. When each policy is
repeated on its own generated history, 00 and B again produce ` same`, while A
and AB produce ` coin`. The other three cases already produce the requested
object under all four policies. A alone has the same observed direct-answer
benefit as AB; no extra semantic benefit is established for B.

These are direct object-token matches at the first readout. Only two generated
tokens are consumed, so an unrestricted `same as…` answer might eventually name
the correct object. The predeclared two-object margin also misses the first
winner change: key-minus-coin is 1.375 for both 00 and AB while the greedy
winner changes from `same` to `key`. Preserve unrestricted vocabulary and
continuation observations alongside narrow contrasts.

## Execution, integrity and custody

One mrun lease, `job-1e8caccdd3cc`, succeeded on CUDA/native-eager BF16 using
installed public wheels. It made four prefills (89 tokens), 372 physical
token-step calls, and 48 additional explicit dense MLP linear calls. Total job
wall time was 294.41 seconds; peak RSS was 2757.2 MB and peak VRAM 3466 MB.
This is an assisted scientific run, not an inference speed benchmark.

| Binding | Identity |
|---|---|
| Request SHA-256 | `7d5436dbc42fbc94ec4b9b1dcf682e185cfd0efe6c0ec25e359831af5efb724f` |
| Raw weight SHA-256 | `a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796` |
| Parameter manifest SHA-256 | `387f1b374598ada7a1026804b78d406e18799fc7f09cea599727b814b1a11c32` |
| Output archive SHA-256 | `dd6aafff741fae4d0b5833bf19002c2cea2ea4840fd974486444dda06cae9404` |

The collector verified all 23,521 files (452,243,338 expanded bytes) against
the output manifest. The compressed archive is 346,080,505 bytes. The raw run,
request, installed-wheel identities, parameter retrieval recipe, typed cuts,
payloads, source and log are retained in
[run-01](../../../../../artifacts/mdb-read-composition-20261107/run-01/report.json).
Weights are referenced through their retrieval recipe rather than republished.

The focused resolver/discovery/debugger/memory/CoT/mechanics suite passed
67 tests, including offline-audit rejection of changed generated-token histories
and changed source rows. A separate outside-workspace installed-wheel resolver
proof passed with tensor frameworks and private packages blocked. The
[independent saved-payload audit](AUDIT.json) passed every case, checking the
formula, cast, actual injected bytes, fixed-reader invariants, history lineage
and seal ordering. A post-audit rehash found every original inventoried file
unchanged and no extra raw-tree files.

The scoped Saturn evidence-plane run and verification both returned zero on
48 scalar observations, with three applicable predicates and no fires. Their
instrument status is `target-streams-only`: there is no synthetic gates receipt
or auto-ingested scientific claim. The [validation record](VALIDATION.json)
binds these outputs and their scope. None of these checks certifies arbitrary
semantic CoT or general chat emulation.

## The next unpaid computation

The result narrows the question from “does composition transfer?” to “can the
state before A plus A's write produce the reader update without running A's
downstream native receiver?” For each query head reading B, a sufficient
quantity for that value contribution is

\[
\Delta\log p_{hB}
=\Delta\ell_{hB}-\Delta\log Z_h.
\]

This is a proposed smaller producer, not a measured solution. Predicting those
per-head scalars would remove the observed-receiver dependency for B's weight.
Producing the full A attention baseline remains a separate computational cost,
and future states, later layers, longer horizons and new task families remain
unqualified. A request for a receiver-free log-weight update program is logged
in Saturn's request ledger.

## Offline reproduction

Run from the public repository with its normal dependencies. Neither command
loads a model or submits a job:

```sh
python packages/mdb/qualification/read_composition_20261107/verify.py /path/to/run-01
python packages/mdb/qualification/read_composition_20261107/analyze.py /path/to/run-01 summary.json --evidence-dir scalar-evidence
```

The verifier consumes durable payloads read-only. The scalar projection has
48 observations (eight receivers × two methods × three boundaries) and adds
no scientific gates. The raw tensor verifier and the scalar evidence sweep
answer different integrity questions.

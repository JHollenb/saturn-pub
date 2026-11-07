# Baseline-only read-weight transport

This qualification asks whether a write nominated at layer 19 can be compiled
into a receiver-local direction for layer 22 without using the receiver's
native A-modified state while the direction is being formed. It is a bounded
instrument assay for a Qwen2.5-1.5B native-eager CUDA run. It does not claim a
portable semantic block, a faithful chain of thought, closure of an A-modified
output, or a serving speedup.

The staged plan is sealed by `launch.py` under
`artifacts/mdb-read-weight-transport-20261107/run-01`. The remote worker writes to
`/mnt/big/mdb/qualification/read-weight-transport-20261107`. The parent worker
is staged as `worker.py`; the prior read-composition and circuit-discovery
workers are copied explicitly as `composition_support.py` and
`qualification_support.py`. They are source and custody support only; runtime
imports resolve from the installed public MDB wheel.

## Ports and visibility

The source port is `(layer=19, kv_head=0,
row_role=preceding-answer-first-generated)`. The receiver is `(layer=22,
kv_head=0, row_role=current-followup-copula)`. The row roles are supplied
locators. They do not infer the represented object or owner.

The compiler sees the full unedited native-00 tape through layer 22, including
q/k/scores at layer 22 and the layer 20/21 history, plus the declared source
write/dose and immutable public Qwen parameter artifacts. It must seal its
smooth FP32 JVP before native A-receiver scoring. Native A q/k, weights,
attention outputs, AB outputs, and held-out receiver outcomes stay outside that
compiler visibility partition. Native 00 and A observations may be used
afterward as measurements and controls.

The planned direction has 19 paid logical linears plus one tangent path. There
are eight compiler invocations, one for each receiver in each of four cases;
each receiver also pays four local source `o_proj` projections and a 19-map
dense FP32 control after the weight seal. Doses are `0`, `0.25`, `0.5`, and `1`. The worker retains
the full direction, local errors, native observations, stale-baseline,
wrong-sign, and query-only controls, reinstall signatures, source identities,
and continuation outputs. No effect-size threshold blocks collection.

For each head reading B, compile $a_j=\dot q^Tk_j/\sqrt d$ for old rows,
$a_t=(\dot q^Tk_t+q^T\dot k_t)/\sqrt d$ for the current append, and
$b_t=\dot q^T\dot k_t/\sqrt d$. The dose evaluator forms
$\delta\ell_j(\lambda)=\lambda a_j+\mathbf1_{j=t}\lambda^2b_t$ and

\[
\widehat{\Delta\log p_B}=\delta\ell_B-
\log\sum_j p_j^0e^{\delta\ell_j}.
\]

This normalization identity is exact for the predicted score changes; their
upstream transport is approximate. A fixed comparator uses the first-order
log-weight change $\lambda(a_B-\sum_jp_j^0a_j)$. Native baseline logits are
widened before log-softmax. The derivative is of an FP32 lift, not the native
BF16 finite program. Predicted displacement is anchored to native baseline
query/key tensors. No nonzero-A data fit, tune or select these constructions.

## Fresh cases and budget

The four cases are genuinely fresh contexts, in this order:

- Mira/Daniel with key/coin, previous owner Mira and Daniel;
- Zoe/Liam with book/cup, previous owner Zoe and Liam.

Each prompt is exactly:

`{a} placed the {x} in the box. {b} placed the {y} in the bag. The object {previous} placed is the`

The follow-up owners are A then B. Every case has two native continuation
tokens. The plan permits at most 300 native token transitions and about 200
logical operations, with one resident BF16 CUDA model, batch size one, four
prefills, no retries, and one finite mrun lease. Native receiver scoring starts
only after the prediction artifacts have been sealed. The second receiver is
reached from the transport-prediction branch end cut, rather than an AB-world
shortcut.

The parameter manifest and standalone retrieval recipe bind the complete
registered Qwen2.5-1.5B checkpoint. The collector accepts the new transport
manifest/custody schemas and the historical support-worker manifest/custody
schemas, while the transport report and archive markers remain namespaced to
this assay.

## Scope of evidence

The primary result is mechanics and instrument validity for baseline-only
direction compilation per receiver. A useful local prediction or dose trend is
retained as an observation. It is not evidence that the model exposes a
general read operator, semantic chain of thought, or full state-only query
program. The fresh contexts, dose curves and hostile controls bound any
stronger transport claim.

Code: [launcher](launch.py), [worker](worker.py), and public
[`mdb.read_weight_transport`](../../src/mdb/read_weight_transport.py).
The preceding [read-composition findings](../read_composition_20261107/FINDINGS.md)
and [blog post](../../../../../obsidian/blog/2026-11-07-000659-the-read-was-observed-before-the-future-was-tested.md)
motivate the exact visibility cut.

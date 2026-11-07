# Compile a changed attention read from an earlier write

`mdb.read_weight_transport` is an experimental, baseline-compiled producer for
the Qwen2-style L19 attention-write → L22 query/current-key path. It produces
selected attention weights without observing the edited native L22 receiver.
It still pays for a complete unedited baseline tape and a dense FP32
directional compilation at each receiver. It is not a general cheap query
compiler or a forward-pass-free chat runtime.

The [second-order extension](READ-WEIGHT-TAYLOR.md) compiles curvature directly into six selected log weights and provides an offline demo of the measured coefficients.

The [preceding composition assay](../qualification/read_composition_20261107/FINDINGS.md)
used the actual edited receiver's query and keys. The
[new plan](../qualification/read_weight_transport_20261107/PLAN.md) withholds
that receiver until the small-program predictions have been sealed. Both use
the same [frozen supplied row locators](CAUSAL-CIRCUIT-DISCOVERY.md#freeze-a-candidate-read-program).

## A runnable six-head toy anchor

[`examples/read_weight_transport_demo.py`](../examples/read_weight_transport_demo.py)
is a framework-free algebra demo. It evaluates the normalized score equation
for six query heads reading one KV head, with five key rows per head:

```sh
python packages/mdb/examples/read_weight_transport_demo.py --dose 0.5
python packages/mdb/examples/read_weight_transport_demo.py --dose 0.5 --json
```

The output is explicitly `toy-anchored scalar algebra; no trained coefficients
and no model execution`. Its values are deterministic teaching numbers, not
saved Qwen measurements. The script keeps the baseline log-probabilities fixed,
adds $\lambda a_j$ to every row, adds $\lambda^2b_t$ at the current append, and
renormalizes over all rows before reporting the selected source probability.
The `stale` column leaves that probability at its baseline, which makes the
normalization and the zero-dose anchor visible without importing Torch or
loading MDB.

The authenticated trained result is separate: the
[qualification findings](../qualification/read_weight_transport_20261107/FINDINGS.md)
report six selected query heads from the twelve-query-head model, four fresh contexts, eight receiver occurrences, and
24 nonzero-dose comparisons. All 24 transport forecasts improve over stale
weights, while the producer retains finite error and does not claim a semantic
program. Per receiver, setup uses 19 logical dense maps plus tangent work and
four local source output projections; warm producer time is 12.4–13.2 ms and
20 scalar evaluations take 0.794–0.810 ms. The synchronized
[`job-466e6e302535`](../qualification/read_weight_transport_20261107/FINDINGS.md#cost-and-integrity)
run used four prefills and 286 physical token-step calls, taking 283.23 s with
peak RSS 2282.8 MB and peak VRAM 4060 MB. Those measurements are links for the
trained assay; they are not inputs to the toy script.

## Two costs with different reuse boundaries

The compiler reads the native unedited input and attention write at L19,
historical K/V at L20–21, native unedited Q/K/scores at L22, current-position
RoPE constants, and original Qwen parameters. A local attention/value replay
forms the preceding L19 write displacement. No edited L20–22 state is an input
to the weight forecast.

`SmoothQwenReadTransport` implements MLP19, complete blocks20–21, and the
normalized, rotated query/current-key projections22 with functional FP32
equations. It includes the changed current K/V appends in the intervening
attention operations. Historical rows stay fixed. `torch.func.jvp` produces
the direction $(\dot q,\dot k_t)$ at the unedited point.

This is a derivative of a smooth FP32 lift of the original parameters.
BF16 rounding is not differentiated as though the finite native program were
smooth. Predicted displacement is anchored to actual native baseline Q/K.
The original modules never execute during compilation, but their full dense
matrices are used. The selected path has 19 logical linear maps plus their
tangent work; this is a paid setup operation.

After compilation, edit-strength evaluation uses scalar coefficient arrays
only. It needs no learned matrix multiply, Q/K contraction, model call, or
observed edited receiver. Coefficients are specific to the baseline state and
source-write direction. Changing the history or direction requires a new
compilation in this prototype.

## The normalized read equation

For an old key row $j<t$ and the current append $t$, compile

\[
a_j=\dot q^Tk_j/\sqrt d,\qquad
a_t=(\dot q^Tk_t+q^T\dot k_t)/\sqrt d,\qquad
b_t=\dot q^T\dot k_t/\sqrt d.
\]

Linear transport of Q/K gives the score update

\[
\widehat{\delta\ell_j}(\lambda)=\lambda a_j+
\mathbf1_{j=t}\lambda^2b_t.
\]

With $p_j^0$ formed from the widened native baseline logits, the evaluator uses

\[
\widehat{\Delta\log p_B}
=\widehat{\delta\ell_B}-
\log\sum_jp_j^0\exp\widehat{\delta\ell_j}.
\]

This is an exact normalization identity for the predicted scores. It does not
make the transported scores exact. All historical rows enter the denominator.
Logs of BF16-rounded probability tensors are not used as the baseline.

The fixed `linear-logweight` comparator uses
$\lambda(a_B-\sum_jp_j^0a_j)$. It is an unconstrained local tangent: its
exponentiated value can exceed one at finite dose. `stale`, `wrong-sign`, and
`query-only` retain missed transport, reversed transport, and omitted
current-key-change alternatives. These are measurements, not quality gates.

## API and validation boundary

The public API names below refer to concrete baseline-tape tensors. For a
Qwen2-style target with hidden width $H$, twelve query heads, two KV heads,
head width $D$, and $T$ key rows, the expected shapes are:

| Name | Shape | Meaning |
| --- | --- | --- |
| `incoming_l19` | `[1, 1, H]` | Hidden row entering source layer 19. |
| `baseline_o_l19` | `[1, 1, H]` | Native layer-19 attention output before its MLP. |
| `old_k_l20`, `old_v_l20`, `old_k_l21`, `old_v_l21` | `[1, 2, rows, D]` | Committed historical KV rows for the complete intermediate blocks. |
| `rope_cos`, `rope_sin` | Broadcastable to `[1, 1, 1, D]` | Current-position RoPE constants used by the shadow. |
| `baseline_q_l22` | `[1, 12, 1, D]` | Widened native layer-22 query. |
| `baseline_k_l22` | `[1, 2, T, D]` | Widened native layer-22 historical plus current key bank. |
| `baseline_scores_l22` | `[1, 12, 1, T]` | Widened native BF16 Q/K scores, including attention scaling. |
| `write_delta_l19` | `[1, 1, H]` | Source attention-write direction for one unit of dose. |

`compile_score_program` trusts `baseline_scores_l22` as the widened native
score tape; it does not recompute those scores from the smooth `q` and `k`.
`kv_head` selects one KV head, and its contiguous six-query-head group is
`range(kv_head * (12 / 2), (kv_head + 1) * (12 / 2))`. `source_row` selects
the reported B row and must be within $T$; `current_row` is the final row
$T - 1$, where the cross term is applied. The actual implementation derives
the head counts from the model rather than requiring this illustrative geometry.

```python
from mdb.read_weight_transport import (
    SmoothQwenReadTransport, compile_score_program, evaluate_score_program,
)

compiler = SmoothQwenReadTransport(
    model, incoming=incoming_l19, baseline_attention=baseline_o_l19,
    history={20: (old_k_l20, old_v_l20), 21: (old_k_l21, old_v_l21)},
    cos=rope_cos, sin=rope_sin,
)
direction = compiler.compile_direction(write_delta_l19)
program = compile_score_program(
    baseline_q_l22, baseline_k_l22, baseline_scores_l22,
    direction["q_dot"], direction["k_dot"],
    kv_head=0, source_row=source_row_l22, scaling=attention_scale_l22,
)
prediction = evaluate_score_program(program, 0.5, method="transport")
```

The public module imports without Torch. Calling the tensor producer requires
Torch; using the class requires a compatible Qwen2-style block contract.
Qwen3 query/key normalization, other activation functions, other rotary
profiles, and other model families are not qualified by this implementation.

Qualification seals every weight forecast before nonzero-A native receivers.
A separate consumer forecast may then use the paid native A attention return
and receiving state to test whether predicted B weights form the joint value
edit. Native attention still executes before injection, and later layers and
token steps execute natively. Those consumer tests do not demonstrate runtime
speedup or close the baseline-output production cost.

The [worker](../qualification/read_weight_transport_20261107/worker.py),
[unit mechanics](../tests/test_read_weight_transport.py), and
[native toy recurrence](../tests/test_read_weight_transport_qualification.py)
separate algebra, forecast visibility, actual consumption and generated-token
authority. Saved trained results and the independent audit are linked from the
qualification findings.

The [measured findings](../qualification/read_weight_transport_20261107/FINDINGS.md)
record 24/24 nonzero-dose improvements over stale weights, remaining finite
transport error, exact injected consumption and unchanged discrete decisions.

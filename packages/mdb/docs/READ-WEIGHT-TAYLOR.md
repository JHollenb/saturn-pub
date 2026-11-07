# Compile a read's curvature into six scalar log weights

MDB can compile a local response law for one selected historical attention row. The compiler starts from an unedited native state and an earlier attention-write direction. It computes how that write changes six later attention log weights, including their second derivatives through the intervening attention and MLP computations. The dose evaluator then uses scalar arithmetic and no model frameworks.

This extends [first-order read-weight transport](READ-WEIGHT-TRANSPORT.md). Compilation still reads dense parameters and the native baseline tape for every new receiver and direction. The later consumer experiment separately pays for native A attention output and input22. Neither cost is closed by a small scalar program.

## The law and its anchor

Let the smooth FP32 transport be `(q_s(λ), k_s(λ)) = F_s(o19 + λd)`. Its displacements are added to the native unedited query and current key:

\[
\bar q(\lambda)=q_n^0+q_s(\lambda)-q_s(0),\qquad
\bar k_t(\lambda)=k_{n,t}^0+k_s(\lambda)-k_s(0).
\]

All historical keys stay fixed. Native BF16 scores widened to FP32 remain the score anchor:

\[
L_j(\lambda)=L_j^0+\frac{\bar q(\lambda)^T\bar k_j(\lambda)-(q_n^0)^Tk_{n,j}^0}{\sqrt{d_h}},\qquad
g_h(\lambda)=\log\operatorname{softmax}(L_h(\lambda))_B.
\]

Two nested directional derivatives at zero produce `c1 = g'(0)` and `c2 = g''(0)`. The serving law is

\[
\widehat g_h(\lambda)=g_h(0)+\lambda c_{1h}+\tfrac12\lambda^2c_{2h}.
\]

For six heads and one selected row, the primary coefficients contain 18 numbers: 72 bytes if packed as FP32. The current API returns Python float arrays in JSON; serialized bytes, Python object overhead, identity metadata, and diagnostic coefficients are larger and separately reported. The polynomial is a local approximation. It is not a normalized full attention distribution and can predict a probability above one; the evaluator preserves that observation without clipping.

## What curvature adds

For score derivatives `a = L'(0)` and `u = L''(0)`, with baseline attention distribution `p`,

\[
c_{1h}=a_{hB}-\mathbb E_{p_h}[a_h],\qquad
c_{2h}=u_{hB}-\mathbb E_{p_h}[u_h]-\operatorname{Var}_{p_h}(a_h).
\]

The first-order Q/K score transport has a quadratic current-key product `b_t = qdot · kdot_t / sqrt(d_h)`. Its routing-only curvature is `2(b_B − E_p[b]) − Var_p(a)`. The full scalar derivative also contains the transported upstream query/key second derivatives. Comparing these laws tests whether the intervening layers' curvature matters beyond QK multiplication and softmax normalization.

## Public API

Construct `SmoothQwenReadTransport` from the [baseline tape contract](READ-WEIGHT-TRANSPORT.md#api-and-validation-boundary), then compile without running any edited native receiver:

```python
from mdb.read_weight_taylor import compile_taylor_program, evaluate_taylor_program

program = compile_taylor_program(
    compiler, baseline_q_l22, baseline_k_l22, baseline_scores_l22,
    write_delta_l19, kv_head=0, source_row=source_row_l22,
    scaling=attention_scale_l22,
)
prediction = evaluate_taylor_program(program, 0.6, method="order2")
```

Compilation requires Torch and a qualified Qwen2-style smooth transport. Run it under `torch.inference_mode(False)` when the surrounding application uses inference tensors. The returned `logp0`, `c1`, and `c2` are head-by-selected-row matrices. Calling `evaluate_taylor_program` reads these scalar arrays without importing Torch. Methods include `order2`, `order1`, `routing-only`, `stale`, `wrong-sign`, and `wrong-curvature`; the latter negates only the second derivative, while wrong sign evaluates the same polynomial at the opposite dose.

## Reproduction and evidence

Run the measured demo from the repository root, with MDB installed:

```sh
python packages/mdb/examples/read_weight_taylor_demo.py \
  --program packages/mdb/qualification/read_weight_second_order_20261107/DEMO-PROGRAM.json \
  --dose 0.6
```

The [fixture](../qualification/read_weight_second_order_20261107/DEMO-PROGRAM.json) contains the sealed Nora/Nora first-receiver coefficients, bound to the original report SHA256, parent cut and forecast reference. It verifies its coefficient digest and evaluates the public API without model execution. At dose 0.6 the first selected head predicts `ΔlogpB = 0.1045466`, hence `pB = 0.4072822`. This is replay of measured coefficients; dense compilation and the native target measurement have already been paid. The fixture's minimal program is 461 JSON bytes and round-trips exactly through 72 packed FP32 bytes.

The [measured findings](../qualification/read_weight_second_order_20261107/FINDINGS.md) show a 59.16% reduction in full-dose mean log-change RMSE relative to first-order transport, with wins in 7/8 receivers. One receiver gets worse; consumer error improves over transport in 5/8 receivers and discrete token decisions do not distinguish methods. These boundaries matter alongside the scalar gain.

The [frozen experiment plan](../qualification/read_weight_second_order_20261107/PLAN.md), [worker](../qualification/read_weight_second_order_20261107/worker.py), [launcher](../qualification/read_weight_second_order_20261107/launch.py), [offline verifier](../qualification/read_weight_second_order_20261107/verify.py), and [unit mechanics](../tests/test_read_weight_taylor.py) distinguish coefficient production, forecast sealing, edited target measurements, and native continuation. This API does not qualify Qwen3 query/key normalization, other model families, economical state-to-state recompilation, or arbitrary semantic CoT discovery.

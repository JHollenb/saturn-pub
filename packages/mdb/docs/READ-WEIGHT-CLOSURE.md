# Predicting a complete Qwen state span before the edited reader

MDB separates two problems in the [six-head read-weight result](READ-WEIGHT-TAYLOR.md): approximating a nonlinear response over a finite dose, and reproducing the model's actual numerical execution. The centered compiler changes the approximation point. The native-profile producer computes the receiving attention output, residual and current KV directly from the earlier state. Neither needs an edited native reader to supply those operands.

## Centered scalar response

The smooth path keeps its original native-zero Q/K and score anchor. For the selected log weight `g(λ)`, the fixed midpoint law is

\[
T_m(\lambda)=g(m)+(\lambda-m)g'(m)+\tfrac12(\lambda-m)^2g''(m),\qquad m=0.5.
\]

Its value at zero need not equal `g(0)`. The separate corrected law is `g(0)+T_m(λ)−T_m(0)`. This subtracts a constant, preserving the midpoint polynomial's derivatives while restoring the native-zero value; its finite-dose error can improve or worsen. Preserve both arms and the raw zero residual. The centered API has a new schema; historical Taylor-v1 programs keep their original meaning.

```python
from mdb.read_weight_centered import (
    prepare_centered_path, compile_centered_path, evaluate_centered_program,
)

path = prepare_centered_path(
    smooth_transport, native_q22, native_k22, native_scores22, write_delta19,
    kv_head=0, source_row=selected_row, scaling=attention_scale,
)
program = compile_centered_path(path, center=0.5)
raw = evaluate_centered_program(program, 1.0, method="centered")
corrected = evaluate_centered_program(program, 1.0, method="baseline-corrected")
```

Preparation copies the native baseline and computes one smooth anchor. Reuse the same prepared path when comparing direct nonlinear evaluation with compilation. The scalar evaluator imports no tensor framework. The path and compiler still perform dense FP32 work for each new state and write direction.

## Functional native profile

`NativeQwenReadTransport` applies the Qwen2 native-eager BF16 equations with the same normalization cast boundaries, projections, RoPE, softmax, attention merge and residual ordering as the qualified runtime. It reads the resident parameters through functional operations and calls no original model module forward.

```python
from mdb.read_weight_native import NativeQwenReadTransport

producer = NativeQwenReadTransport(
    model, incoming=native_input19,
    history={20: (old_k20, old_v20), 21: (old_k21, old_v21),
             22: (old_k22, old_v22)},
    cos=current_cos, sin=current_sin,
)
predicted_A = producer.evaluate(locally_rebuilt_attention19)
predicted_AB = producer.evaluate(locally_rebuilt_attention19, value_edit={
    "kv_head": 0, "source_row": selected_row, "scale": 0.0,
})
```

The native-profile producer needs the native00 input19, old KV excluding the current row, exact current RoPE constants, and a locally formed source attention write. Its flat tape includes input/attention/MLP/exit19–22, positioned Q/K and full V banks20–22, current KV20–22, and scores/weights22. Direct AB scales the historical V22 row before AV and output projection. It does not subtract a projected value contribution from an observed A output.

Each evaluation pays **24 dense linear maps**, three attention computations, validation and state materialization. This removes dependence on edited native A outputs by reproducing their arithmetic; it does not reduce that arithmetic to a cheap scalar program.

## State and KV consumption

The [qualification worker](../qualification/read_weight_closure_20261110/worker.py) binds each prospective span to its parent cut, pending token, context length, physical sites and layers19–22. On the first pending token it installs the predicted exit state and current KV at each boundary and advances the cursor once. Historical A/B edits stay installed. Layers19–22 execute no native modules on that token; prefix0–18, suffix23+ and subsequent tokens remain native.

The bridge checks the actual incoming hidden tensor against the declared operand and records installed hidden/cache tensors after writing them. A forecast tape is a declaration; `installed_*` captures identify what the native consumer actually received. Later tokens use the branch's resulting cache and greedy output. The next test receiver comes from the raw midpoint branch rather than a branch selected after seeing outcomes.

The bridge is an experiment-local integration with MDB's typed runtime. The standalone native-profile producer is the public API; the experiment does not add a general CLI span-install command.

## Qualification and cost

The [frozen plan](../qualification/read_weight_closure_20261110/PLAN.md) separates two known development contexts from two fresh contexts and retains signed doses, zero residuals, consumer countertrends and native-profile discrepancies. The [tests](../tests/test_read_weight_closure_qualification.py) check complete baseline/AB logits and cache closure, installed state, cursor accounting and subsequent own-state continuation on a tiny random BF16 model.

The [trained findings](../qualification/read_weight_closure_20261110/FINDINGS.md) report exact functional baseline and AB closure in 8/8 receivers, and exact selected log weights in 40/40 receiver/dose pairs. Midpoint full-dose scalar RMSE falls 0.02824482→0.01415301 with 7/8 improvements; negative extrapolation worsens on average. All scalar methods, including stale, retain the same short token decisions, and fresh later-logit error reverses the local accuracy ranking. These observations qualify a bounded computational account; they do not establish arbitrary chat.

The cost benchmark compares directly measured compilation-plus-evaluation with direct nonlinear smooth evaluation at K=1,5,32 on the same prepared immutable state, using paired warm samples and CPU scalar outputs. Reuse across doses in one state is distinct from economical recompilation as the conversation advances.

At K=32, receiver-median compilation plus evaluation takes 43.697ms versus 35.797ms for direct smooth evaluation. No tested K repays compilation. The functional A producer already computes exact B weights; paying it and then substituting a polynomial is a factorization experiment, with dense production still paid.

## Replay measured coefficients

With MDB installed, run from the repository root:

```sh
python packages/mdb/examples/read_weight_centered_demo.py \
  --program packages/mdb/qualification/read_weight_closure_20261110/DEMO-PROGRAM.json \
  --dose 1
```

The [fixture](../qualification/read_weight_closure_20261110/DEMO-PROGRAM.json) binds the Nora/Ethan first receiver's sealed coefficients to the report, forecast and parent cut. The first head predicts `Δlogp=-0.04416469`, `p=0.74966384`; raw midpoint's zero residual for that head is 0.00293152. Choose `--method baseline-corrected --dose 0` to obtain an exact zero delta while retaining the raw residual. This replay loads no model or tensor framework; dense compilation and target measurements have already been paid.

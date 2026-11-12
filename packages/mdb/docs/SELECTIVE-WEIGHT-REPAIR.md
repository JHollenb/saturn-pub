# Preservation-aware refinement of a discovered edit

MDB can now fit a low-rank affine correction to desired local responses, measure the proposed repair through the original model, and compile the selected correction into reversible weight bytes. The implementation is `mdb.wsp_selective`; the native Qwen experiment is [wsp_selective_qwen.py](../qualification/wsp_selective_qwen.py), with its [launcher](../qualification/submit_wsp_selective.py) and independent [saved-output audit](../qualification/audit_wsp_selective.py).

This follows the [automatic site/gain experiment](AUTOMATIC-WEIGHT-DISCOVERY.md). That experiment selected deletion of `model.layers.6.mlp.down_proj` on Qwen2.5-0.5B-Instruct. The refinement takes that discovered site and suppression direction as inputs. It does not claim to rediscover the site or invent a desired local write without task feedback.

## The proposed repair

For task keys H and desired output changes R, solve ΔW H≈R while penalizing ΔW K on preservation keys K. The implementation uses a sample-space FP64 ridge solve followed by a thin-factor SVD, returning A and B with ΔW=AB and a bounded rank. No gradients or optimizer steps are used. The fitted response is a proposal, not a behavioral prediction.

In this experiment, H contains the last-token MLP input from sixteen arithmetic development cases. R is the negative of the native MLP write at those positions, derived from the earlier useful deletion. K contains all valid token positions from eight preservation prompts. The remaining model and its normal output head determine whether the proposed local response helps the task.

The search tests ranks 1, 4 and 8, preservation fit weights 1 and 100, and doses 0.5, 1 and 1.5. Native is eligible and wins ties. Whole-write deletion is a reference outside this narrower candidate grammar. Selection uses development correct-token mean log probability minus mean preservation KL; all component measurements and unsuccessful proposals remain visible.

The two earlier task splits are now development data. Twenty newly generated arithmetic pairs, with overlap excluded, and twelve new preservation prompts supply the final evaluation. No outputs or activations from these fresh prompts enter fitting or selection. Every candidate declaration is sealed before its round executes. Numerical factor files and their hashes are bound to the discovery and evaluation source identities.

## Execution and evidence

The assay uses one mrun lease, one resident CPU FP32 eager Qwen model, one activation-capture forward, twenty development forwards, and six evaluation arms. Development rows are batched at 24 and evaluation rows at 32. Each evaluation arm also generates four new tokens on four problems from its own fresh prefix and native K/V. Generation explicitly uses greedy decoding and repetition penalty 1.0; the inherited checkpoint policy is retained in the runtime record.

The six arms are native, whole-write deletion, factored correction, compiled correction, original-write restoration with compiled weights still installed, and reversed-sign correction. Full first-token vocabulary logits are retained for every evaluation arm. The saved-only audit independently recomputes full-vocabulary winners, target log probabilities and native-to-edited KL in FP64, compares full logits across factored/compiled and native/restored executions, and verifies inventories, seals and selection. Floating-point reduction and lowering differences are reported numerically.

Original pretrained constants remain recipe-bound through `parameter-artifacts.json` and standalone `retrieve_weights.py`. Fitted factors, edited parameter archives, capture samples, predictions and native outcomes remain evidence. Low rank bounds the correction's dimension: it does **not** imply sparse changed-weight support, smaller dense inference cost, or an edit to only a handful of coefficients.

## API

```python
from mdb.wsp_selective import (
    fit_low_rank_delta, virtual_low_rank_edit, synthesize_low_rank_patch,
)

left, right, fit = fit_low_rank_delta(
    task_hidden, desired_changes, preservation_hidden,
    rank=4, preserve_weight=100., ridge=0.01,
)
with virtual_low_rank_edit(model, site, left, right, dose=1.):
    result = evaluate_native_consumer(model)

patch = synthesize_low_rank_patch(
    model, site, left, right, dose=1., model_identity=model_identity,
)
with patch.installed(model):
    compiled_result = evaluate_native_consumer(model)
```

Fit and native-consumer selection are separate operations. The caller supplies the site, examples and desired response; the finite experimental controller selects coefficients, rank, preservation fit weight and dose. The final weight patch has no task-name predicate, answer lookup, donor state, or external evaluator in its forward path.

## Qualification status

The installed-package mechanics suite passes 80 checks, including a durable numerical activation-capture regression. Seven public Saturn shared-store tests pass. The first native lease, `job-381c15270212`, stopped while saving its capture because the shipped Saturn wheel predated MDB's new public shared-store import. Its logs, scheduler result and partial evidence are retained; it supplied no candidate or held-out result. The corrected public package closure executed in separate guarded job `job-1c69edf785bf`.

## Native result

The controller selected rank 4, preservation fit weight 100 and dose 1.5. The following numbers come from the independent FP64 audit of saved full-vocabulary first-token logits, on twenty fresh arithmetic cases and twelve fresh preservation prompts.

| Arm | Correct first-token choices | Mean target log probability | Mean preservation KL, nats |
|---|---:|---:|---:|
| Native | 2/20 | -3.008633 | 0 |
| Whole-write deletion | 6/20 | -2.169690 | 0.461137 |
| Selective compiled correction | 10/20 | -1.586939 | 0.002212 |
| Reversed-sign correction | 2/20 | -3.048689 | 0.001954 |
| Original-write restoration | 2/20 | -3.008633 | 0 |

The selective edit gains eight correct choices over native and four over whole-write deletion, with 99.5203% lower mean preservation KL than deletion. Nine previously wrong task cases become correct; one native-correct case becomes wrong. The maximum individual preservation KL is 0.012538 nats. Reversed sign leaves the aggregate correctness at native level while changing individual decisions, and slightly lowers mean target log probability. These heterogeneous outcomes remain in the raw rows.

The factored and compiled implementations have the same twenty task choices and the same four generated four-token continuations. Their maximum full-logit difference across all thirty-two rows and the complete vocabulary is 2.76566e-5. Their mean target log probabilities differ by approximately 1.05e-6. Original-write restoration reproduces the entire saved native first-token logit array exactly and reproduces all generated tokens. The final original parameter inventory also matches exactly.

The local fit uses sixteen task keys and thirty-four valid preservation-token keys. Before dose, the chosen correction's Frobenius norm is 3.1788% of the native matrix norm; at dose 1.5 it is approximately 4.7683%. Its factors contain 23,040 coefficients, while compilation writes an ordinary 896×4864 matrix. This is a correction of rank at most four in real arithmetic, followed by FP32 rounding, and is not a sparse-weight-edit result.

Worker scientific time is 217.79 seconds; the complete lease, including setup and archive publication, takes 257.12 seconds, with 3,957.5 MB peak RSS and no GPU allocation. Scheduler-declared, sealed and executed payload hashes match. The collected archive is 141,556,160 bytes, SHA-256 `3e58e57afb87bf0e03e9e3ae96ea4f3202a132f2dc82d181ed1c7bb34a94a8d1`.

The saved-only reader verifies 318 inventoried files, all three sealed studies, disjoint task and preservation splits, the development-only selection policy, complete first-token vocabulary argmax, target log probabilities and KL. It reports the original FP32 and recomputed FP64 reductions separately: maximum per-row target-log-probability discrepancy is approximately 2.09e-5, and maximum KL discrepancy is approximately 2.95e-5 in the whole-write arm. Historical outcomes are unchanged. The Saturn evidence sweep and verification complete with target-streams-only coverage of 26 timeline records; they supply no additional typed receipts or address citations.

The workspace custody bundle is `artifacts/mdb-selective-discovery-20261110/native-02`, containing the request, package seal, guarded receipt, original publication, collected evidence and `saved-audit.json`. The [qualification record](../qualification/selective-weight-repair-20261110.json) binds these files and the executed sources.

This supports a bounded working inference: a useful deletion supplied a causal direction that could be compiled into a more selective, task-useful weight correction. The task family, site, suppression direction, arithmetic labels and preservation set were supplied; the correction coefficients, rank, fit weight and dose were selected automatically. The fresh cases share the development prompt template. Generated strings continue beyond the requested digit, so improved first-token correctness is not complete-answer compliance. No broad arithmetic capability, sparse support, inference speedup, independent full-model replication, or terminal capability certification is claimed.

# Pass causal state diff

`mdb.pass_causal_state_diff` is the standalone, offline composer for
`saturn-pass-causal-state-diff-v1`. It joins retained StateCuts, logical position bindings,
optional Thott and pass-baseline reports, supplied layer captures, native readouts, and receipt
links. It never loads or invokes a model and never upgrades a numerical difference into a causal
claim.

```python
from mdb.pass_causal_state_diff import compose_pass_causal_state_diff

report = compose_pass_causal_state_diff(
    parent_statecut=cut_n,
    child_statecut=cut_n_plus_one,
    pass_index=2,
    native_consumer={"consumer_id": "qwen-native-suffix"},
    parent_positions=[
        {
            "logical_position": "prefix:0",
            "physical_position": 0,
            "role": "immutable_context",
            "input_authority": "caller-supplied-prefix",
        }
    ],
    child_positions=[
        {
            "logical_position": "prefix:0",
            "physical_position": 0,
            "role": "immutable_context",
            "input_authority": "caller-supplied-prefix",
        },
        {
            "logical_position": "turn:2:question:0",
            "physical_position": 1,
            "role": "appended_question",
            "input_authority": "caller-supplied-question",
        },
        {
            "logical_position": "turn:2:answer:0",
            "physical_position": 2,
            "role": "appended_model_answer",
            "input_authority": "native-model-generated",
        },
    ],
    common_parent_statecut_id="cut-before-control-arms",
    followup_query={
        "query_id": "identical-followup-v1",
        "input_sha256": "...",
        "format_sha256": "...",
        "input_authority": "caller-supplied-identical-query",
    },
)
```

StateCuts can be mappings or native objects with `checkpoint_id`, `model_identity`, and `slots`.
The package does not import Saturn or Torch. Tensor slots are consumed by protocol through
`detach()`, `cpu()`, `contiguous()`, and `tolist()` when those methods exist. JSON artifacts use
nested rectangular number arrays. Standard `kv.NNN.key` and `kv.NNN.value` slots work directly; a
lightweight mapping can instead use `slots.kv.<layer>.{key,value}`.

Physical row numbers are not comparable across unequal caches. Callers must provide unique logical
positions. Every row present at pass N is `immutable_context` for this N to N+1 comparison, even if
it originated as a question or generated answer in an earlier pass. Child-only rows must declare
`appended_question` or `appended_model_answer` and their input authority.

Layer captures may supply `query_residual`, `attention_alpha`, `mlp_gate`, and `norm_scale`.
Position-indexed values use this form so unequal lengths are aligned before comparison:

```json
{
  "values": [0.20, 0.10, 0.15, 0.55],
  "logical_positions": [
    "prefix:0",
    "turn:2:question:0",
    "turn:2:answer:0",
    "prefix:1"
  ]
}
```

Readouts retain raw logits and raw margins separately from log probability and generated output.
Intervention, restore, and continuation arguments are durable links, not replayed receipts. The
report always emits separate mechanics, instrument, trend, and terminal statuses with
`causal_verdict: not-inferred-from-diffs-alone`.

[`../examples/pass-causal-state-diff.sample.json`](../examples/pass-causal-state-diff.sample.json)
is a compact emitted report accepted directly by `mdb.temporal_causal_viewer`.

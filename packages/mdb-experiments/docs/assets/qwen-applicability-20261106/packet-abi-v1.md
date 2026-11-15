# Applicability packet ABI

Public module `mdb.computation_applicability`, one function:
`evaluate(source_packet, intervention, execution_profile, output_spec, policy, *, coverage_index=None)`.
No execution, framework imports, file reads or target metrics. Final pins follow tests.

`source_packet` has exactly:

```
schema: mdb-qwen-source-packet-v1
packet_id: opaque nonempty label (retained, not decision authority)
kind: rgu | qkv
frontier: exact existing architectural/inherited-functional-state scope object
binding:
  parent_job, parent_request_sha256, parent_raw_manifest_sha256,
  parent_global_seal_sha256, source_case,
  producer_sha256 (parent qwen_sign_shortcut_campaign.py83f9),
  qwen_config_sha256, qwen_source_binding_sha256,
  parameter_manifest_sha256 (subset4255), parent_parameter_manifest_sha256,
  selection_protocol_sha256 (kind rgu -> d942..., qkv -> 8c53...)
ports: original raw protocol descriptor map, unchanged, for computation/baseline ports
context: original raw protocol descriptor map, unchanged, for these five names:
  input_ids, attention_mask, position_ids,
  rope_buffer:model.rotary_emb.inv_freq,
  rope_buffer:model.rotary_emb.original_inv_freq
parameters: {full model.layers.18.* tensor name: original subset artifact.source object}
```

`ports` for rgu are h/head/R/gate/up/product/M/Hnext; for qkv are
h/q/k/v/head/R/M/Hnext. Subsets are permitted to report missing availability.
No cross-kind port union. Each descriptor and original parameter source object is
bound to a known immutable original metadata digest by the module. No NPY read.

`intervention` exactly:

```
row_signs: list of 512 integer +/-1 values
permitted: {kind: causal-all-attended, rows: 512}
windows: [[start,stop], ...] partitioning all 512 rows
```

An unknown permitted kind remains explicit unknown/unsupported, not all-false.

`execution_profile` exactly:

```
torch: 2.13.0+cu130
transformers: 5.14.1
device: cuda:0
dtype: torch.bfloat16
native_source_sha256: fbdc... (Qwen source)
sdpa_source_sha256: b72c...
geometry: {batch_size:1, sequence_length:512, hidden_size:2560,
  intermediate_size:9728, num_attention_heads:32, num_key_value_heads:8,
  head_dim:128, layer:18}
attention: {mask:null,is_causal:true,enable_gqa:true,dropout_p:0.0,
  scale:128**-0.5,use_cache:false,past_key_values:null}
positions: {start:0,step:1,length:512}
rope_buffer_policy: exact existing CPU FP32 policy object
flags: exact eight runtime flag keys/values from existing runtime-contract
```

All flags match the existing original CUDA profile, including threads4,
deterministic algorithms, highest FP32 precision, TF32 off, BF16 reduction true,
cudnn deterministic true/benchmark false. Module exports immutable-by-convention
constants `SUPPORTED_PROFILE`/`SOURCE_BINDING` for inspection; caller data are copied.

`output_spec` exactly:
`{ports:[head,R,M,Hnext], response:edited-minus-own-positive, horizon:19, continuation:null}`.
A non-null continuation is an opaque object with consumer_id/input_port/horizon/
source_identity/parameter_instance/execution_profile; initial implementation
reports unqualified continuation separately and never extends local support.

`policy` exactly:
`{order:[m0,m1,m2], allow_full_functional_fallback:true|false}`.
Order may be a permutation of the three opaque IDs; it is declared priority,
never a speed claim. m2 remains explicitly labelled fallback and requires its
availability/profile plus the true fallback flag.

Optional `coverage_index` exactly:

```
schema: mdb-computation-coverage-index-v1
index_id: opaque string
occurrences: [
 {occurrence_id, decision_sha256, method_id,
  account_kinds:[opaque strings], evidence_refs:[opaque strings]}
]
```

Only exact decision-hash + selected method entries are occurrence matches.
No error/pass/outcome values. Unknown/incomplete index annotations do not reroute.
Actual outcomes remain `{status:not-read-or-executed}` in evaluator output and
are joined by a separate consumer; no outcome argument is accepted.

Outputs include schema, binding, structural status and per-query closure/window
counts, method prerequisite/support records, selection (candidate/fallback/abstain,
all-rows scope), local_support, continuation_support, finite_evidence_coverage,
occurrence_lookup, actual_outcome, decision_sha256. No scientific null or accuracy
pass is produced. Returned nested data are fresh JSON copies.

Primary replay: rgu/global -> m0; rgu/prefix -> m2 (or abstain with fallback false);
qkv/last-row and alternating64 -> m1, always whole 512. Source-only cross packet
mask replays are optional. Root alone builds isolated wheel and runs actual replay.

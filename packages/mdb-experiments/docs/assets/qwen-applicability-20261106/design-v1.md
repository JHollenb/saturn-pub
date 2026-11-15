# Source-only Qwen applicability evaluator

Design only. No frozen turn 14/15 code, protocol or evidence is changed. No model,
weight payload, source activation array or numerical target was read for this
design. The supplied Astra direction 11 and root's reported turn 15 observations
inform the reporting boundaries below, never the selection rule.

## One bounded evaluator

Implement one pure metadata function, initially in qualification and then exposed
as an opt-in public function after its metadata replay qualifies:

`evaluate(source_packet, intervention, execution_profile, output_spec, policy, *, coverage_index=None)`

It returns a source-bound immutable plan. Inputs contain no observed errors,
target arrays, success thresholds, nearest-neighbor scores or outcome rankings.
The function imports no Torch, tokenizer or model package and runs no learned
operation. A small standard-library causal-support calculation is sufficient.
The optional index is a strictly allowlisted metadata projection. Compute the
decision from the first five arguments before annotating index matches; missing
or changed coverage never changes that decision.

The initial domain is the original parameter instance at Qwen block 18, B1/T512,
residual width 2560, intermediate width 9728, query heads 32, KV heads 8 and head
dimension 128; BF16 learned/state tensors on the qualified CUDA execution cell;
all-attended self-attention, no past/cache, no dropout and no bias. Use the exact
original config/eleven-recipe identities, native Qwen source fbdc..., delegated
SDPA source b72c..., Torch 2.13.0/Transformers 5.14.1 and recorded execution flags.
No claim extends to a new model, block, hardware/profile cell, mask or cache mode.
Unprofiled selected kernels remain explicitly unprofiled.

The dispatch contract is actual attention_mask=None, is_causal=true,
enable_gqa=true, scale=128**-0.5 and groups=4. Regenerated positions are 0:512,
with original CPU-initialized FP32 RoPE policy/base 1e6 and no borrowed kwargs.
The source packet binds h/context/parameters/provenance and each available port's
shape, stride, logical/stored dtype and content/file hashes to the immutable
parent raw inventory/global pretarget seal. Descriptor authentication is distinct
from later payload/device/finite checks by the executing runner.

For cached QKV, require positioned operands after Q/K per-head normalization and
RoPE. Expected BHTD layouts are q shape [1,32,512,128], stride
[2097152,128,4096,1], and k/v shape [1,8,512,128], stride
[524288,128,1024,1]. The declared restoration implementation must preserve those
dense noncontiguous layouts. Merely possessing an NPY file or an arbitrary QKV
port name is insufficient to establish that contract.

## Separate five answers

1. **Structural relation:** for declared signs s and permitted set A, report for
   every query i whether A[i] is nonempty and every permitted key j has s[j]=s[i].
   Retain the whole Boolean vector, declared window counts and cross-sign edge
   counts. Mixed support removes a sufficient ideal-arithmetic symmetry premise;
   it does not assert a nonzero response. Edge counts are not attention mass or
   causal importance. An unknown mask/cache yields an unknown structural result,
   not an all-false result or a scientific null.
2. **Source/profile support:** report matched, unsupported, unknown or invalid
   prerequisites for each method, including actual port availability and common
   h/context/parameter lineage. This is a mechanics/domain account.
3. **Finite evidence coverage:** annotate which port/endpoint/response/bit/zero
   accounts have recorded evidence under the exact source/profile/intervention
   binding. Coverage counts every retained occurrence, including misses. It is
   never a successful-row filter, accuracy certificate or selection gate.
4. **Selected candidate:** choose one whole-component plan using only the declared
   inputs and the fixed rule below. Selection remains a falsifiable prospective
   candidate, even inside the supported domain.
5. **Occurrence and outcome:** exact occurrence lookup returns immutable IDs and
   evidence pointers. A separate outcome attachment records actual observations
   after a plan is sealed. Lookup and outcome cannot change the selected method.

In particular, structural eligibility does not imply unconditional BF16 raw-bit
identity at Q/K ports. Root reports turn 15 numerical QKV closure with Q/K signed
zero differences, while final heads/downstream reconcile exactly. Keep numeric,
raw-bit and signed-zero coverage and observations port-specific; do not collapse
them into one finite pass flag or repair the historical raw-bit differences.

## Fixed method rule and actual availability

Use generic method IDs. The default output specification names head/R/M/H19 and
complete edited-minus-own-positive responses. Distinguish prediction operands
from baseline ports required for response accounting:

| Method ID | Prediction inputs beyond bound context | Structural prerequisite |
|---|---|---|
| m0 | positive head/R/G/U and original down weight | every one of the 512 rows closed |
| m1 | positive h/positioned Q/K/V and original O/postnorm/gate/up/down parameters | no support-closure requirement |
| m2 | positive h and all original eleven block 18 recipes | complete qualified functional block fallback |

Positive head/R/M/H19 baseline references are required for the requested complete
response account; the evaluator never invents them by planning an extra positive
block. A different output specification must be explicit and separately bound.
Do not silently shrink the reported ports when a baseline is absent.

Within the matched domain, select m0 if all rows are closed and its source ports
are present; otherwise select m1 if its ports/profile/layout are present;
otherwise select m2 only when its complete component fallback is explicitly
available under this same qualified profile. If none is available, return an
explicit abstention and the missing prerequisites. This order is a declared
computation policy, not a measured fastest-backend claim.

For unsupported or unknown source/profile/cache, do not label m2 qualified merely
because a generic implementation exists. Return abstention with a qualified
fallback proposal describing the new binding/qualification required. The initial
fallback registry contains only this original all-attended/no-cache block 18 cell.
Fallback means one full functional component block, never a whole-model/LM call.

Turn 14's source packet has R/G/U but no cached QKV. Turn 15's source packet has
QKV but no G/U. Do not union their ports silently, use post-target native ports as
forecast sources, infer missing ports from an exact occurrence, or form a positive
block to fill a missing source. Any future explicit packet join would need a new
declared selection with identical parent h/context/parameter/source/profile
bindings and custody for every joined operand.

The main four metadata replays should produce:

| Declared occurrence | Packet | Closed rows | Selected whole-component plan |
|---|---|---:|---|
| global negative | turn 14 R/G/U | 512 | m0, all 512 rows |
| prefix 0:256 negative | turn 14 R/G/U | 256 | m2 if available; otherwise abstain |
| last row 511 negative | turn 15 QKV | 511 | m1, all 512 rows |
| alternating 64-row signs, first positive | turn 15 QKV | 64 | m1, all 512 rows |

This is a whole-component selector. Never turn the 256/511/64 closed-row sets into
a shortened SDPA query batch or a rowwise combination that changes native shape,
layout, dispatch or reduction policy. With an explicitly coherent packet holding
both operand sets, any mixed-support case still selects full m1 attention over
all 512 queries. Structural row labels remain diagnostics beside that plan.

An optional source-only cross-packet replay makes the availability distinction
clear: the QKV packet selects m1 for global/prefix too, and the R/G/U packet
requires m2/abstention for last-row/alternating too. No activation is read and no
new scientific specimen or forecast result is created by this metadata exercise.

## Minimal record and identity

Return a record shaped approximately as follows; this is a proposed schema:

```
schema: mdb-qwen-source-applicability-plan-v1
binding: source_packet, parameter_instance, source_identity, execution_profile,
         h_identity, context_identity, frontier, horizon, output_spec, continuation
structural: status, closed_rows_per_query, windows, cross_sign_edges, scope
methods: {m0,m1,m2} -> support, available_ports, missing_prerequisites, reasons
selection: method_id | abstain, whole_query_bounds, reasons, fallback_plan
finite_evidence_coverage: metadata_index_status, occurrence_refs, port_account_kinds,
                          raw_bit_identity_implied=false
occurrence_lookup: exact_key, status, refs, authority=recorded-occurrence-only
actual_outcome: status=not-read-or-executed, optional_observation_ref
decision_sha256: hash(binding + source-only structural/method/selection + policy)
```

The decision hash excludes evidence retrieval results and actual outcomes. A
separate immutable report/attachment can bind the evidence index snapshot and
outcome receipts to that decision. Adding evidence or changing an attached
numerical error cannot retroactively choose another method.

Use opaque IDs for methods, occurrences, parameters and consumers; explicit typed
ports and bindings carry the meaning. No human semantic label is required. Exact
occurrence identity includes consumed source content hashes, original parameter
instance, operation/helper sources, context/sign/mask/cache/profile identity,
frontier, output horizon and consumer. Extra unconsumed ports need not change a
method occurrence key, but the complete source-packet identity remains retained.

The present horizon ends at H19. A continuation is an explicit consumer binding,
not a label inferred from the state name: for example, original block 19's
parameter/source/profile identity, hidden-state input port, context and horizon
H20. Missing continuation binding leaves local selection intact but marks that
continuation unbound. It does not establish an H20 or image forecast. A future
block 19 comparison of different cheap/recomputed H19 states is a separate
prospective consumer experiment; no such model execution is part of this replay.

## Metadata-only replay and hostile controls

Use the frozen turn 14 protocol
`d942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9`
and turn 15 protocol
`8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff`,
their source/profile/descriptor metadata and exact sign/window declarations.
Both reference parent job `job-8c949b1c0278` and global source seal
`a6d61b11fa67bfa0eb3a50fee66f26584b452874a590c255163102362460e652`.
Do not open NPY, safetensors, target/comparison numeric reports, images or models.
Signs, causal permitted edges and known mask/position hashes can be computed
from small declared vectors with ordinary Python integers/bytes.

Create a small allowlisted evidence-coverage index containing only occurrence
IDs, exact bindings, tested port/account kinds, custody/stage metadata and outcome
receipt references. It contains no error values, pass labels or quality rankings.
The evaluator records an incomplete/unknown index explicitly rather than treating
missing finite coverage as an absent relation. The root-reported finite Q/K bit
limitations may be attached as evidence annotations, never route predicates.

Controls should be independent metadata perturbations with expected reasons:

* Remove G/U from the global R/G/U packet: m0 unavailable; choose qualified m2 or
  abstain. Remove QKV from the QKV packet: m1 unavailable; qualified m2/abstention.
* Change a parent/content/parameter/source identity without a matching authenticated
  binding: source unsupported/invalid, explicit fallback qualification plan.
* Change BF16, block index, row/head geometry, QKV stride, native/SDPA source,
  deterministic/TF32/reduction flags or RoPE policy: profile mismatch is explicit.
* Declare padding, a noncausal mask, different positions or unknown mask: no silent
  conversion into the current causal-flag profile. Structural computation, if
  possible, is explicitly conditional on the supplied graph, not qualified native
  dispatch. Unknown mask gives unknown structural status.
* Add KV cache, change past length/position offset, or omit cached-key signs and
  provenance: cache profile unsupported. Do not infer that cached keys are unchanged.
* Supply a different or absent continuation: local and continuation support remain
  separate, with no exact lookup masquerading as future-consumer qualification.
* Add/reorder irrelevant metadata or attach zero/large/missing outcome errors:
  selection and decision hash stay unchanged. Target/error fields supplied to the
  evaluator input are rejected as outside its source-only schema.

The replay proves deterministic declared applicability accounting and conservative
abstention only. It performs no new floating-point prediction or numerical parity
test. Unknown/unsupported/incomplete are not scientific-null results. Retain every
replay row, reason, previous miss and finite limitation as an independent account.

## Minimal public integration

Reuse `mdb.equation_comparison.applicability` for exact supplied profile predicates;
its matches/does-not-match/unknown statuses already separate missing metadata.
Add only the narrow source/structural/method selector around it, with the rule and
domain above. Do not turn the existing predicate matcher into a learned resolver.

Use `EquationCatalog`/`ComputationAtlas` for exact occurrence references and their
existing source/parameter/frontier/horizon/consumer binding. Their retrieval is
not selection authority. Keep the pure plan as its own record; do not fabricate
a zero-error comparison record merely to insert an unexecuted plan into the Atlas.
Use a metadata-only index projection for this initial replay.

Existing `dependency_reuse.plan_suffix_reuse` can later consume declared state
reads/writes for a separately bound continuation; it does not supply Qwen row
semantics or prove footprint completeness. No automatic continuation integration
is needed initially. Keep the public footprint to one opt-in metadata function
and immutable plan schema, without a new model bridge, registry framework or
execution CLI. Native execution and saved numerical accounting stay in their
already qualified producer/reader lanes.

The frozen turn 14 producer's reported Ruff import-group diagnostic is preserved
with its source bytes. This design performs no styling changes to frozen sources.

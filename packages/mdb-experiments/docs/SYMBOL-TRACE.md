# Follow a symbol through native execution

Symbol history joins the separately qualified bindings of one supplied symbol name to recorded
occurrences, consumer observations, native cuts, receipts and saved dataflow slices. A later
execution point retains its own binding fingerprint. Grouping records under a name does not
establish that the corresponding values have the same meaning.

```sh
mdb symbol-trace ./project demo.carrier --session-id symbol-trace
mdb symbol-trace ./project demo.carrier --session-id symbol-trace --cut CUT --output trace.json
mdb symbol-trace ./project demo.carrier --session-id symbol-trace --mermaid --output trace.mmd
```

The same view is available in an attached public debugger:

```text
project symbol-history demo.carrier --session-id symbol-trace --cut CUT
```

The standalone project command accepts the same query:

```sh
mdb project ./project symbol-history demo.carrier --session-id symbol-trace --cut CUT
```

`--cut` restricts the occurrence and use events to the retained parent ancestry of that cut.
Sibling events are excluded. `--branch` filters recorded branch labels, which remain labels;
native parent ancestry supplies the separate history evidence. Execution clocks remain typed
records rather than an inferred universal time scale. Missing ancestors remain visible frontiers.

## What the view retains

The sealed `mdb-symbol-trace-v1` result includes distinct binding groups, lifecycle snapshot
identities, occurrences and consumer uses, native ancestry, receipt identities, declared slice
references, signed observations, invalidation records and optional CoT references. Repeated
events shared by several immutable lifecycle snapshots appear once with all supporting snapshot
identities. The snapshots and invalidations remain available independently.

A use links to a definition only when its source cut and exact binding match one unambiguous
recorded occurrence. A missing or ambiguous association remains unknown. Native receipt model
and execution identities are checked before the receipt enters the trace.

The trace does not hydrate tensor payloads, choose a new native state, expire a value, run an
intervention or grant a write. Lifecycle joins retain the underlying public runtime's qualified
location checks. The source lifecycle format, native StateCut and receipt formats are unchanged.

CoT references are candidate observation locators. A matching source cut is an association;
missing or mismatched source information remains explicit. Neither a readout label nor an
adjacent pair of events becomes a semantic or causal dependency. Consumer observations preserve
their original measurements and controls without upgrading their claim status.

## Python and verification

```python
from mdb import Project
from mdb.symbol_trace import trace_symbol, verify_symbol_trace

project = Project("./project")
trace = trace_symbol(project, "symbol-trace", "demo.carrier", cut_id=selected_cut)
verify_symbol_trace(trace)
```

Seal verification checks the exported record's content identity. Recomputing a trace from its
project additionally verifies the referenced lifecycle/native metadata joins. Neither check
certifies semantic equivalence or causal necessity.

## Reproducible native workflow

The [demonstration driver](../examples/symbol-trace/run_demo.py) creates a new project using a
random tiny Qwen3 on CPU FP32. It records one carrier name at distinct execution points, runs
native consumers, retains an intervention sibling, checks exact replay, and reopens the trace
without model frameworks. The names describe supplied carrier roles; this is a debugger
mechanics experiment.

```sh
python examples/symbol-trace/run_demo.py /tmp/mdb-symbol-trace-demo
```

The [finite mrun launcher](../qualification/submit_symbol_trace.py) stages installed public
runtime bytes and the same workflow for Qwen2.5-0.5B-Instruct with CUDA FP32/eager attention.
It requests CUDA admission, strict preflight, one resident model, one finite lease and no
automatic retries. It has a 300-second execution timeout. Collection addresses the exact job
and checks the returned project archive against the worker's digest.

```sh
python qualification/submit_symbol_trace.py
python qualification/submit_symbol_trace.py --collect JOB_ID
```

The implementation builds on [qualified symbol slices](SYMBOL-SLICES.md), the existing
`SymbolBinding`, and lifecycle sidecars. Semantic identification, automated causal qualification
and consumer-equivalent replacement of a dynamic circuit remain separate research work.

## Validation status

### Public CoT cut export

`mdb.public_cot_project` exports actual public Saturn `StateCut` objects through the existing
Project CAS store. It accepts supplied `SymbolBinding` and lifecycle records, validates their
model/execution/schema/point identities, and requires retained parent/result payloads for every
admitted receipt. Cuts at the same execution point may contain different concrete branch state;
use `binding_cut_ids={binding_fingerprint: [cut_id]}` or exact lifecycle source-cut joins to
disambiguate them.

```python
from mdb.public_cot_project import export_public_cot_project, verify_public_cot_project

exported = export_public_cot_project(
    project, session_id, session, cuts,
    bindings=[binding], lifecycles=[lifecycle], receipts=selected_receipts,
    binding_cut_ids={binding.to_dict()["fingerprint"]: [source_cut.fingerprint]},
)
proof = verify_public_cot_project(
    project, session_id, lifecycle_ids=[lifecycle.fingerprint],
)
```

Pass the actual saved lifecycle IDs when re-verifying lifecycle joins; the verifier's default
empty sequence does not enumerate all stored lifecycles. Verification checks CAS seals and
metadata without tensor hydration. It does not run a model or establish causal or semantic
authority. Native replay separately uses `Project.load_cut` and the compatible public adapter.
Open a temporary Project copy for immutable artifact inspection because metadata cache tables
can be maintained on reads.

The [fresh generated-arithmetic assay](../../../../saturn/experiments/2026-10-07-scaffold-lifecycle-atlas/analysis/cot-composition-public-v1/FINDINGS.md)
uses an experimental caller-owned public adapter with typed cache-row Acts. Its finite CUDA job
ran 64 native forwards and retained a selected 61-cut/60-receipt Project, one structural binding
and a definition/use pair. A source edit changed native generation 7→8; reciprocal digit-value
edits changed later ×3 answers while preserving the target token history and keys. Whole-query
durable replay was exact, and an independent installed reader reopened the joins without tensor
hydration. The joint causal bundle is larger than the one bound address; this is not a complete
thinking trace or a semantic qualification of every carrier coordinate. These results are
separate from the horizon-1 mechanics qualification below.

The final mechanics path was checked in a separate process with 74 selected MDB tests; a separate
public Saturn-pub process ran 98 core tests, 172 focused checks total. The final tiny report
retained two exact binding groups, selected-cut ancestry and a sibling exclusion, and the
installed-release proofs reproduced the trace through both CLI and project navigation while blocking frameworks and tensor
hydration ([final tiny report](../../../../artifacts/mdb-symbol-trace-20261028075727/final-tiny/report.json),
[installed-release tiny proof](../../../../artifacts/mdb-symbol-trace-20261028075727/installed-release-tiny-proof.json),
[installed-release pretrained proof](../../../../artifacts/mdb-symbol-trace-20261028075727/installed-release-pretrained-proof.json)).

The final corrected qualification under rebuilt normal wheels is `job-daddb335946b` on
Qwen2.5-0.5B-Instruct with CUDA FP32/eager attention, the five-token prompt `The capital of
France is`, one resident model and one finite CUDA lease. Its native logits consumer had
generation budget 1: the main and copy branches decoded ` Paris` with bit-equal full vectors,
while the zero-carrier branch decoded ` is`; the independent endpoint check verified the saved
hashes and bit equality. The run retained two supplied binding groups, one intervention and 99
physical native transitions ([report](../results/symbol-trace/job-daddb335946b/report.json),
[job/custody records](../results/symbol-trace/job-daddb335946b/job.json),
[endpoint check](../results/symbol-trace/job-daddb335946b/independent-endpoint-check.json)).
The request and physical native bound were 256 operations/transitions. The historical report's
descriptive `limits.logical_operations: 512` is stale metadata and remains preserved; the current
driver and request record 256. Worker time was 45.7850 seconds and scheduler wall time was
73.69124 seconds. The installed-release proof passed strict clock witnesses, blocked
frameworks and tensor hydration, and preserved the selected cut ([installed-release pretrained
proof](../../../../artifacts/mdb-symbol-trace-20261028075727/installed-release-pretrained-proof.json)).
This was a broad supplied hidden-carrier mechanics experiment: the name came from the driver,
no semantic or country-specific circuit was discovered, and CoT was untested/not assessed at
horizon 1.

The earlier `job-523fbb5f44f1` report remains a historical numerical record: its logits and
endpoint hashes match the final run, but stricter clock validation rejects its use witness
because the driver supplied result execution while metadata `source_cut` pointed to the source
cut ([historical report](../results/symbol-trace/job-523fbb5f44f1/report.json),
[historical endpoint check](../results/symbol-trace/job-523fbb5f44f1/independent-endpoint-check.json)).
It is retained for provenance and is not the final clock-validated qualification. The historical
report's descriptive `limits.logical_operations: 512` remains preserved; the final request and
physical native bound are 256, and observed work is 99. The final job carries the corrected
result-cut source.

The final native job included the public `saturn_pub.store` CAS reuse fastpath. No paired speed
claim is made.

The earlier Qwen qualification remains retained timeout evidence after 300.59 seconds with no
report ([job](../results/symbol-trace/job-f566153dce79/job.json),
[worker log](../results/symbol-trace/job-f566153dce79/worker.log),
[custody](../results/symbol-trace/job-f566153dce79/custody.json)). The standard evidence sweep
for the final target also returns exit 2, `evidence-incomplete-empty-target`, because its adapters recognize
JSONL streams and gates-dict receipts but not this debugger's custom report and CAS receipt
metadata. That is an instrument coverage limitation, not a semantic null or failure: the final
custom seals, archive hashes, offline reopen and native endpoint checks pass. It remains an
instrument coverage limitation rather than a semantic null.

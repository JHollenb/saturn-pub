# Symbol trace qualification

This is a bounded final native workflow qualification for the read-only
symbol-history view. It qualifies numerical consumer mechanics and exact clock
joins under the rebuilt normal wheels. It does not qualify semantic symbol
identity, a portable circuit, a causal mechanism, or a chain-of-thought
interpretation.

## Reproduction

The finite launcher is [`submit_symbol_trace.py`](submit_symbol_trace.py). It
stages the worker and demonstration driver, records their source hashes, requests
one strict CUDA lease, and collects only the exact terminal job:

```sh
python qualification/submit_symbol_trace.py
python qualification/submit_symbol_trace.py --collect JOB_ID
```

The worker is [`symbol_trace_worker.py`](symbol_trace_worker.py); the public
workflow is [`run_demo.py`](../examples/symbol-trace/run_demo.py). The contract
and query surface are documented in [`SYMBOL-TRACE.md`](../docs/SYMBOL-TRACE.md).

## Admitted scope and budget

The final corrected run was [`job-daddb335946b`](../results/symbol-trace/job-daddb335946b/job.json):

- Qwen2.5-0.5B-Instruct, public Hugging Face adapter, CUDA FP32, eager attention,
  one resident model, and the prompt `The capital of France is` (five tokens).
- One outer finite CUDA lease with strict preflight, no automatic retries, no
  hand reservations, one model load, direct collection, and all logical branches
  executed in process.
- Admitted geometry: one intervention, eight logical branch slots, one local
  checkpoint replay, 256 logical operations/max local work items, and a 256 native
  transition limit. The executed workflow used four branches and 99 physical native
  transitions, with two exact binding groups, one intervention, 14 retained native
  receipts and 75 retained transition rows.
- Native consumer horizon one (`generation_budget: 1`) with full first-token logits.
  This is a broad supplied hidden-carrier experiment: the carrier name was supplied
  by the driver. It was not a semantic or country-specific circuit-discovery study;
  CoT references were empty, CoT status was not-assessed, and `causal_effect` remains
  `not-established`.

The request and scheduler record are sealed at 256 logical operations and 256
native transitions. The final report retains `limits.logical_operations: 512` as
stale descriptive metadata; it was not edited. The current source's descriptive
constant and the request record 256, and the worker's physical bound and observed
work are independently recorded at 256 and 99. This distinction keeps the report
immutable while preventing it from being read as the admitted budget.

## Final native result

The main and native-copy branches both decoded ` Paris` and their full logits
vectors were bit-equal. The zero-carrier branch decoded ` is`; the independent
endpoint check measured zero-versus-copy L2 `1408.147705078125`, maximum absolute
delta `14.889449119567871`, and different argmax values. One-step replay was exact,
the full restored suffix cut matched, and the framework-free reopen verified the
trace with zero native payload reads. These are observations from one native
consumer and one prompt, not a semantic or causal claim.

The final worker took 45.7850 seconds and scheduler wall time was 73.69124 seconds.
The installed-release proof passed strict clock witnesses, blocked frameworks and
tensor hydration, and preserved the selected cut.

- [Final worker report](../results/symbol-trace/job-daddb335946b/report.json)
- [Final request](../results/symbol-trace/job-daddb335946b/request.json)
- [Final job and admission/custody record](../results/symbol-trace/job-daddb335946b/job.json)
- [Final independent endpoint check](../results/symbol-trace/job-daddb335946b/independent-endpoint-check.json)
- [Final worker custody record](../results/symbol-trace/job-daddb335946b/custody.json)
- [Installed-release pretrained proof](../../../../artifacts/mdb-symbol-trace-20261028075727/installed-release-pretrained-proof.json)
- [Final archived source bundle](../results/symbol-trace/job-daddb335946b/sources)
- [Installed-release tiny proof](../../../../artifacts/mdb-symbol-trace-20261028075727/installed-release-tiny-proof.json)
- [Immutable staged wheels and evidence index](../../../../artifacts/mdb-symbol-trace-20261028075727/)

The final native job included the public `saturn_pub.store` CAS reuse fastpath. No paired speed
claim is made for this qualification. The fastpath has separate coverage in
[`test_store_publication_reuse.py`](../../../tests/test_store_publication_reuse.py).

The earlier [`job-523fbb5f44f1`](../results/symbol-trace/job-523fbb5f44f1/job.json)
remains a historical numerical record: its logits and endpoint hashes match the
final run, but stricter clock validation rejects its use witness because the driver
supplied result execution while metadata `source_cut` pointed to the source cut.
It is retained for provenance and is not the final clock-validated qualification.

## Validation and evidence limits

The final selected regression suites ran in separate processes: 74 MDB checks and 98 public
Saturn-pub core checks, 172 focused checks total. The installed-release proofs blocked model
frameworks and tensor hydration while reproducing the trace through the CLI and project
navigation. The [compatibility regressions](../tests/test_symbol_lifecycle_project_compat.py)
preserve historical omissions of `slice_basis` and `receipt_closure` while rejecting changed
present values and truncated or extra join rows.

The standard evidence sweep for the final target returned exit 2 with
`evidence-incomplete-empty-target` ([run record](../../../../artifacts/mdb-symbol-trace-20261028075727/evidence-index/job-daddb335946b-run.json),
[verify record](../../../../artifacts/mdb-symbol-trace-20261028075727/evidence-index/job-daddb335946b-verify.json)).
Its current adapters recognize JSONL streams and gates-dict receipts, but not the
public debugger's custom report and CAS receipt metadata. This is an instrument
coverage limitation, not a semantic null or native-workflow failure: the final
custom report seal, archive hash, offline reopen and independent endpoint checks
pass.

The earlier [`job-f566153dce79`](../results/symbol-trace/job-f566153dce79/job.json)
is retained timeout evidence after 300.59 seconds with no report ([worker log](../results/symbol-trace/job-f566153dce79/worker.log),
[custody record](../results/symbol-trace/job-f566153dce79/custody.json)). It remains
separate historical evidence and was not overwritten or retried during the corrected
qualification.

The implementation and tiny evidence are also summarized in the
[symbol-lifetime synthesis](../../../../obsidian/blog/2026-10-28-004715-symbol-lifetimes-time-scaffolds-dynamic-circuits.md).

# Longitudinal symbol trace

This example connects two separately qualified bindings for the same supplied
symbol name to native successor states and durable project dataflow. It uses a
random tiny Qwen3 decoder on CPU FP32, so its model values have no scientific
meaning. The useful result is the evidence shape: exact binding clocks, native
receipts, separate lifecycle seals, one zero-carrier intervention, a native
copy control, and a framework-free trace reopen.

Run it in an environment with the installed MDB, Saturn-pub, PyTorch and
Transformers dependencies:

~~~sh
python run_demo.py /tmp/mdb-symbol-trace-demo
~~~

The output directory is created afresh and contains:

- project/: the retained native Project and StateCut metadata;
- lifecycle-a.json, lifecycle-b.json: separate sealed lifecycles for
  demo.carrier at post-layer-0 and post-layer-1 cuts;
- trace-all.json: the sealed mdb-symbol-trace-v1 longitudinal trace;
- trace-at-source-b.json, trace-at-main-result.json,
  trace-zero-branch.json: exact cut and branch-filtered traces;
- consumer-endpoints.json and consumer-logits/*.npy: full-vector logits
  digests, descriptors, argmax values and native endpoint arrays for the zero,
  copy/no-op and main suffixes;
- phase-timings.json: flushed adapter, binding, suffix, replay, lifecycle,
  trace and offline-reopen progress markers;
- raw.json: retained cut descriptors, native receipts and saved joins;
- report.json: the compact assay report and offline reopen result.

The native sequence is deliberately short. It steps through embedding and
layer 0, binds demo.carrier, steps layer 1 and binds the same logical name
again with a distinct context fingerprint, then restores the common parent.
The zero branch uses the public debugger algebra expression
zeros_like("hidden"), while the copy branch is a native fork with no write.
Both run the full native suffix to the readout endpoint (two transitions for
the tiny two-layer specimen) through one public Saturn token-granularity
transaction, so the real-model path avoids persisting every intermediate layer.
The main branch runs the suffix from the second
binding; an exact restore/replay check returns to the same successor cut. Trace
filtering verifies that a main cut excludes the sibling zero branch and that
both binding fingerprints remain visible. The native consumer endpoint is the
full first-token logits vector, compared by its full-vector digest (generation
budget 1, full prefix); the report records the one hidden-state intervention,
same-parent zero/copy, native-prefix copy/main and zero/main L2/max-absolute
deltas, argmax equality and endpoint paths/hashes, plus separate flags for the
one-step replay and full restored-suffix equality. It keeps the actual retained
native transition-row count, including replay branches, and the smaller
deduplicated receipt count. The callable reserves 256 native transitions for a
deeper pretrained suffix without duplicating large logits arrays in JSON.

The report records native_payload_reads: 0 for the metadata trace and runs a
new process with torch, transformers, diffusers, private saturn, and common
imports blocked. This is a project/lifecycle mechanics qualification.
cot_refs is intentionally empty and cot_observation.status is not-assessed:
no decoded text is treated as a semantic CoT observation. Every lifecycle and
trace retains causal_effect: not-established.

For a real pretrained qualification, construct the same public
DecoderAdapter in the admitted mrun worker and call:

~~~python
report = qualify(native_adapter, [1, 2, 3], "/mnt/big/mdb/symbol-trace/job-id")
~~~

Use a new mrun experiment identity, one finite lease, strict preflight and no
retries. This example itself does not submit or launch mrun work.

# Tiny native circuit workflow qualification

November 12, 2026. These are local CPU mechanics results, authorized while Beast mrun was down. The specimen is a randomly initialized Qwen3 with 22,720 parameters, two layers, hidden width32, seed17 and input IDs1,2,3. It has no trained semantic vocabulary. No large model was loaded and no fleet job was submitted.

The [walkthrough](../../examples/project-navigation/circuit_understanding.py) exercises Project focus, root/history regeneration, original attention and MLP expansion, a full first-readout adjoint, finite four-arm comparisons, deliberately incomplete interface testing, a source-linked finding, graph verification and reopen.

## Results

| Check | Observation |
| --- | --- |
| Original active parent | Preserved exactly |
| Regenerated target from token/history recipe | Matches original operand descriptor and expected cut |
| Native versus original-module autograd logits | Exact descriptor match within each profile |
| Source-frequency return closure | 1.39e−17 maximum absolute discrepancy |
| Gate/up/interaction/native-remainder accounting | Zero retained closure discrepancy in four tested arms |
| Routing/payload/interaction/native-remainder accounting | Zero retained closure discrepancy in four tested arms |
| Local prediction discrepancy | At most8.93e−8 margin units on the two signed one-coordinate edits in the primary lane |
| Native own-token decisions | All four branches consume their own selected token25 |
| Incomplete interface | Two witnesses: same hidden coordinate0, different downstream logit after coordinate1 edits |
| Project graph | 14 circuit record nodes, linked through knowledge edges; verifies on reopen |
| Saved finding | Exact focus/result references survive reopen |

All four branches retain the same chosen output token. The useful effects are internal/logit changes and a state-sufficiency counterexample; these are not four semantic successes. Finite local decomposition and local derivative prediction remain distinct measurements.

## Numerical lanes

- [Primary source-test report](report-primary.json): CPU FP32, Python3.10.14, Torch2.13.0, Transformers5.14.1, one Torch thread. The walkthrough completes in4.65 seconds including setup in this recorded invocation. The normal source test environment supplies the declared model-library pins.
- [Installed-wheel compatibility report](report-installed-compatibility.json): separately built MDB and Saturn-pub wheels installed outside the checkout; `python -I` runs from `/tmp`. The temporary environment inherits Torch2.11.0 and Transformers5.12.1. This is an additional explicitly identified compatibility lane, not qualification of the pinned numerical lane. Its recorded invocation completes in3.82 seconds. Runtime imports resolve the installed public packages.
- A second installed-wheel process blocks Torch, Transformers, mrun and private workspace imports. Read-only focus, explanation, records, finding and graph reopen still succeed. No model is loaded for this check.
- The [final-wheel compatibility report](report-installed-final.json) repeats the full walkthrough after the final implementation changes, preserving a separate result. Its complete native Project and console output remain in the evidence directory.

The two lanes have different model/execution fingerprints and slightly different logits; their bytes and statistics are not pooled. The shared behavior does not establish cross-version exact parity. Timings are descriptive mechanics costs, not a performance objective or comparison.

## Custody and reproduction

Complete tiny Project stores, both reports and the initial incomplete installed attempt are retained under the workspace [evidence directory](/Users/jakeholl/domains/artifacts/mdb-circuit-workflow-20261112). These include native cut pages, receipts, circuit records and findings; the new record layer is compact metadata, but existing Project persistence still retains native activation/cache pages. Original model weights are not exported. Recipe regeneration does not require loading the old target tensors.

The primary report SHA256 is `7c4a95f30dd3d48702b6ae8c5368a94e7eea2cca86c622bfe866d3eb1d61c251`. The installed compatibility report SHA256 is `8cce75ee454849c2a3c0a1aa5921ad452a02631a70a273f72859f9def2c272f7`. Raw Project identities are in those reports; the model is reproducible from the declared seed/configuration and exact numerical program.

The initial installed walkthrough completed its native panel but stopped during interface evaluation because a linked replay recipe was reused for the wrong endpoint. Exact-target matching fixed this; a regression test now evaluates comparison endpoints while the original root recipe is attached. The incomplete Project remains preserved as instrument evidence, not a scientific null.

Run the example with installed public packages and a fresh output directory as described in the [runbook](../../docs/NATIVE-CIRCUIT-INSPECTION.md). The focused test module also covers foreign sessions, pending edits, tampering, virtual receipt-prefix replay, exact source binding, partial panels, own-token horizons, sampled interface witnesses and offline operation.

## Deferred pretrained qualification

The focused implementation/regression suite passes66 checks, including the same original-port/consumer path on tiny Qwen2 and Qwen3. Lint and whitespace checks pass. The installed console script also completes through ordinary focus, inspection, plan/run and finding commands. These are mechanics checks; they do not certify semantic generalization. The [source seal](source-seal.json) binds all eight changed runtime modules to the final tested wheel bytes.

The public offline evidence sweep and verify both report `evidence-incomplete-empty-target` (exit2): the standard stream/claim pack does not recognize these Project SQLite/native-cut records as its input streams. Both reports are preserved in the evidence directory. This is an instrument-coverage limitation, not an independent certification or a scientific null. Native receipt validation, sealed circuit records, graph verification and the explicit reopen checks above remain the applicable checks.

When Beast mrun returns, admit one finite sequential Qwen2.5 job with strict preflight and pinned installed wheels. First qualify the same public `DecoderAdapter` profile's source→read→MLP→full-logit workflow, sham/replay identity, local-versus-finite response, signed doses, source/channel controls, saved finding and fresh-process reopen. Keep every arm and instrument diagnostic. A direct before/after comparison of two-token scores does not certify the full vocabulary winner or later-token behavior.

The earlier successful temperature experiment edited a full-prefill V-projection source; this new first implementation inspects the public decoder's pending-token hidden boundary. They have different intervention and state contracts. Do not require or advertise reproducing the historical cold→hot flip until that source port is explicitly supported and qualified. Broader semantic binding/lookup tests, integrated mixed derivatives and arbitrary-family replay remain subsequent experiments.

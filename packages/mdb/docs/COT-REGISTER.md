# Live CoT register

Status: a native Qwen stepping prototype. Compatible frozen Jacobian lenses automatically produce source-bound vocabulary readouts as the debugger advances. The debugger frame shows the latest readout in human language, and an accompanying StateGraph retains native lineage and decoded annotations. Semantic CoT reconstruction remains unqualified.

Attach fitted lenses once; ordinary stepping then updates the register without explanation queries:

```python
pin = bench.pin_jacobian_lens(lens_manifest, transport)
bench.configure_cot_register([pin["lens_ref"]], top_k=5)
bench.step("layer")
```

Use one separately fitted, compatible lens per admitted completed-layer boundary. A lens is bound to the model, tokenizer, decoder weights, execution lane, and input coordinate system. A fitted layer-1 lens cannot be reused at layer 2 by changing its label.

The local debugger accepts independently saved `FrozenJacobianLens` directories:

```sh
mdb open --local /path/to/qwen-checkpoint --root ./session \
  --prompt 'Your prompt' --cot-lens /path/to/fitted-layer-lens
```

Repeat `--cot-lens` to attach other layer boundaries; `--cot-top-k` controls the displayed candidate count. Real-model scientific work in the domains workspace uses an admitted mrun worker with the same Bench API. The local entry point does not itself add scheduler admission.

## Resident reader, changed capture, and demand hydration

For a session that exclusively owns frozen model weights and execution tensors, use `bench.configure_cot_register([pin["lens_ref"]], top_k=5, optimized=True)`, or add `--cot-fast` to the local debugger command. The CLI owns and freezes its loaded weights. API callers must use eval mode with `requires_grad=False` and route all weight/module/config/tokenizer mutations through `bench.weight_mutation()`. This includes `.data` writes and custom kernels, which Tensor version counters cannot certify. The default strict reader continues to hash its mutable consumer on every operation. Rebinding, ordinary in-place edits, gradient flags, numerical-profile changes and explicit invalidation refuse the managed reader until its bindings are verified again. Weight changes require a compatible lens and a correctly identified model session.

The managed lane retains one verified reader and device transport per lens, keeps top-k ranking on the model device, and caches verified static tensor identities. It indexes each new observation once and maintains linked native ancestry, observation counts and bounded windows; repeated views do not scan or verify the full history. These caches assume the session's immutable CAS ownership. Offline verification remains available to check backing-store integrity independently.

Completed staged K/V rows are immutable within a pending token under the admitted Bench execution path. The fast capture reuses their exact payload descriptors and the unchanged last-logit descriptor, and captures newly completed rows and the current residual. Existing VM pages share immutable ancestors and encode changed append pages. External writes to execution tensors must use an admitted Bench Act; arbitrary writes through resolved tensor handles are outside this capture contract. Restore discards provisional descriptor caches. Legacy cut, payload and numerical schemas remain unchanged.

Fast `step` and `generate` commands publish one atomic journal selection at command return, with one final session manifest. Internal observed cuts are retained in that pack. An explicit `checkpoint`, a returned breakpoint, restore, branch operation or close remains a durable boundary. A process crash during a command recovers its preceding durable selection; command failure restores that live selection and retains candidate bytes. The strict lane retains its existing publication cadence. A long `generate` therefore has command-level recovery granularity; use smaller generation commands for more frequent recovery points.

The graphical pane still updates automatically after each command. It now also emits `cot-pointer/pointer.json` in the fast lane. This is a small, Thott-backed pointer to a bounded language/graph catalog and the selected immutable native cut. Thott object and cursor changes share a separate sealed metadata journal pack, with recovery before an offline pointer reopen; the native cut remains the execution authority. This removes per-object fsync from the derived pointer path while preserving returned pointer durability. Shared native CAS backing remains external; the pointer alone is not a portable model or rewind payload. Emit a retained pointer with `bench.cot_pointer(destination)`.

```python
from mdb.bench_pointer import BenchPointer

pointer = BenchPointer.open("/path/to/cot-pointer/pointer.json")
metadata = pointer.browse()             # no native tensor payload reads
plan = pointer.native_closure()         # selected K/V + pending execution descriptors
residual = pointer.residual()           # one selected residual, verified bytes
native = pointer.hydrate_native()       # only the selected native payload closure
bench = pointer.resume(model, tokenizer, model_sha256=verified_weight_digest)
```

Pointer restore uses the native Bench/VM consumer, keeps unselected branches cold, and hydrates a branch when selected. Call `configure_cot_register(..., optimized=True)` again to declare frozen ownership in the new process. `BenchSession.resume(..., lazy_branches=True)` exposes the same selected-branch policy for Qwen; the default full-session resume remains available. `inspect performance` reports binding hits, indexed rows, descriptor reuse, publication counts and nested host phase durations. Those durations do not independently attribute CUDA kernel time.

The automatic frame renders this structure:

```text
(main) phase=layers next_layer=12 ...
CoT register: token 47, after layer 12 [decoded candidates]
  Vocabulary readout: 'France', 'capital', 'Paris'
  Dependency has not been causally tested.
```

Those terms illustrate the display format; they are not a measured model result. The implementation describes retained vocabulary scores, rather than generating first-person thoughts or inventing an operation from a token label. Token-ID-only models display IDs and have no semantic vocabulary.

## Register and graph

`bench.cot_register()` returns a bounded view of observations on the selected native cut's ancestry. `bench.cot_graph()` returns a sealed StateGraph snapshot and a Mermaid representation. In the debugger, `inspect cot` and `inspect cot.graph --full` expose those optional expanded views; the latest text appears automatically in normal frames.

Native `continues` edges record execution order. Decoded readouts are separate knowledge annotations. A sequential pair of vocabulary observations does not create a causal dependency, and a token's presence does not establish an instance, role, intention, or executed operation.

The session also maintains `cot-register.html`, an automatically refreshed graphical pane with native state nodes and human-language vocabulary annotations. Open that file alongside the debugger and keep using `step`, `continue`, `restore`, or `fork`; the pane follows the selected durable history. It does not require a query. The refresh button can pause the browser view without pausing model execution. Render failures are reported as `panel_error` in the register and do not alter native execution.

Restore and fork select observations through native StateCut ancestry. Sibling and abandoned futures remain retained but disappear from the selected view. A changed residual, a later uncovered layer, and readout/sampling/commit phases mark the latest decoded observation as historical. Missing coverage stays visible.

The existing artifact store and journal carry the configuration, native source cuts, original Jacobian receipts, compact display rows, and register events. Export and resume retain the register without changing legacy native cut schemas, addresses, or numerical contracts. Offline verification checks register source/lens/model/session bindings alongside existing Jacobian and native artifact verification. Human vocabulary labels are descriptive metadata; verification is not a semantic certificate.

An optional observer failure produces an explicit diagnostic while allowing a valid native step to continue. Configuration rejects incompatible lenses before attachment. Numerical parity, source restoration, and instrument validity remain separate from scientific interpretation.

## Current boundary

This first version uses the existing dense Jacobian reader and its checkpoint/custody path. It adds observer compute and storage; low-overhead continuous serving is unmeasured. It observes newly stepped token/layer boundaries, not an unrecorded prefill history. Text captions are deterministic instrument descriptions. Rich operation/role decoding, automatic causal qualification, Mamba streaming summaries, and retrospective Hydra analysis remain future work.

The focused mechanics tests use tiny random CPU BF16 Qwen weights and identity transport fixtures. They compare native logits and K/V, automatic debugger rendering, restoration and sibling isolation, export/resume, edit staleness, optional observer failure, and atomic configuration. These fixtures establish software behavior without establishing learned reasoning readability.

The [validation record](../experiments/cot-register/VALIDATION.md) includes the 81-test regression result and separate installed-wheel capture/reopen qualification.

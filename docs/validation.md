# Release validation

Version 0.1.0 was validated locally on macOS arm64 with CPU execution on 2026-09-29.

- 35 tests passed on Python 3.10 and on an independent Python 3.12 checkout.
- All six example scripts passed with Hugging Face offline mode enabled.
- The scripted local debugger passed; receipt lineage and durable state restoration were checked.
- Ruff lint and formatting checks passed.
- Source distribution and wheel builds succeeded.
- A bare wheel installed in a fresh environment and imported the public core, store, and debugger
  without torch or transformers installed.
- A clean checkout installed using the public lockfile, ran its tests and examples, and used no
  private editable packages or workspace import paths. Framework packages may be downloaded
  during installation; offline example execution performs no weight downloads.

Tests exercise native Qwen logits and multi-token continuation, layer cuts, DDIM native-step
parity, conditional UNet to native VAE decode, wrong geometry and non-finite refusal, branch
isolation, abort/stale-commit refusal, artifact tampering, fresh-process hydration, registered
training closure, optimizer/RNG rollback, policy choice, global-softmax shared-prefix decode,
private-tail accounting, and immutable-key phase relocation.

Pretrained weights, CUDA/MPS numerical parity, remote services, and semantic benchmark results
were not part of this validation. The included GitHub Actions workflow has not been run remotely.

## FLUX.2 Klein extension

The subsequent extension passes 38 local Python 3.12 tests, including FP32/BF16 native flow
parity, transposed pipeline latent layouts, block cuts, durable replay, and invalid writes.

CUDA job `job-2e7068ce1663` on an RTX 4080 replayed a supplied historical 104,044-byte writer
with the cached distilled Klein 4B revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
The four-step, 512×512, seed-611 native and dose-2 outputs match the historical PNG pixels
exactly. Every transformer block at the first denoising step and the ordinary pipeline's
complete RGB output match exactly.
Native/zero-dose/wrong-time/uninstalled counts are 3; dose 2 gives 5; dose 4 returns 3 and the
controller restores dose 2 with its registered training closure verified exactly.

Native, patched, zero-dose, and uninstalled final latent/pixel records replay exactly in a
second interpreter using a saved cut, with no prompt encoding or feedback selection there.
The finite lease took 253.57 seconds, sampled peak RSS 17,700.2 MB and VRAM 8,314 MB.

This is one previously selected development context and supplied writer, not a learned new
payload or a held-out semantic battery. Only the scalar dose is selected by labelled image
feedback. The historical patch's broad collateral problems remain outside this reproduction.
Other devices, normalization placements, model revisions, and cross-version replay need their
own numerical validation.

## Writer construction, standalone experiment, and debugger notebooks

The later public extensions pass 45 local Python 3.12 tests, including standalone
experiment replay-mismatch handling, preservation of existing evidence, boundary
breakpoints, bounded continue, and exact debugger rewind. Lint and package builds pass.
These newer checks do not extend the initial Python 3.10 validation to every new feature.

CUDA job `job-90c58216fa46` built a new rank-eight writer from fresh base/recipient
conditional block cuts, verified both prefixes against native execution, and reproduced
the three-to-five apple development result. Dose four regressed and the controller
restored dose two exactly. Four branches replayed exactly in a fresh interpreter.
The outer wrapper failed its inherited archive-size limit after the scientific checks;
collection-only job `job-ce37204ce6d8` recovered the existing evidence without model reruns.
The [public reference](../experiments/flux2_writer/reference-result.json) preserves this scope.

All three notebooks executed successfully on CPU: AR debugging with fresh-process replay,
diffusion branching and durable load, and shared KV memory/phase mechanics. The separate
standalone experiment wrapper was checked with dry-run and orchestration mechanics tests;
no paid Hugging Face cloud job was launched.

The new `measured_program.py` and `dependency_reuse.py` examples were run locally on
random native CPU Qwen. They verify exact measured-payload replay and unsupported
context/parent refusal, plus declared cell reuse/recomputation with the final native
readout executed on every run. They make no semantic portability or speedup claim.

## Typed execution, custody, debugger, and structural extension

Version **0.2.0** was validated locally on macOS arm64 with Python 3.12.13 and CPU execution
on 2026-09-29:

- **79 tests passed**, including fresh-interpreter Qwen/DDIM/FLUX suffix replay, typed seal
  tampering, failed publication rollback, numerical/configuration drift, same-version weight
  replacement refusal, and operation identity across different source paths.
- **11 offline example scripts passed**; the two pretrained FLUX scripts were excluded from
  this no-download lane. The updated shared-memory example verifies both durable segmented
  suffix replay and phase-origin replay with unchanged key bytes.
- **All three notebooks executed** with no error outputs; the AR notebook verifies exact
  fresh-process native and candidate replay.
- Ruff lint, formatting, and `git diff --check` passed. Source distribution and wheel builds
  passed; their contents exclude outputs, virtual environments, credentials, and private repos.
- The bare wheel installed in a fresh environment and ran a JSON capture/continue/restore
  transaction without torch, transformers, Saturn's private repo, manalysis, or mrun installed.

The new release checks use Torch 2.14.0, Transformers 5.14.1, and Diffusers 0.39.0. They do
not extend the historical pretrained CUDA evidence above, certify new semantic capabilities,
or establish cross-device/kernel equality or a speedup. The new APIs were not rerun on Python
3.10 in this release lane.

| Feature | Executed proof surface | Claim boundary |
|---|---|---|
| ExecutionPoint, SurfaceManifest, StateAddress | Tiny random native Qwen, DDIM, and FLUX CPU adapters plus a parameter-free graph adapter | Typed boundaries and same-program mechanics; no pretrained semantic claim |
| Microstep and macro-step execution | Operation-level and family-level CPU transitions, exact rewind, and fresh-process suffix replay | Exactness is bound to the declared program/device contract |
| Act identity, preconditions, masked patching, preservation | Registered pure built-ins on tiny CPU tensors; masked graph-memory write preserves unselected cells and declared ports | Caller supplies address, mask, value, and preservation set |
| Observer, WatchEvent, debugger commands | Branch-local change/threshold/nonfinite predicates, trace/diff/replay, safe-point cuts, and generic-adapter load | A fire is diagnostic triage, not a semantic or promotion verdict |
| CausalPath and PortObservation | Synthetic source/address/writer/carrier/consumer graph with cut/receipt-bound divergence, repair, collateral, and native-consumer rows | Validates evidence support and separation; does not discover a mechanism |
| LocalStore and ReplayBundle | Content-addressed JSON/tensor pages, repeated-save deduplication, local ancestry, adapter provenance, and opt-in import into a fresh store | No remote store, bundled checkpoint weights, or inferred missing ancestors |
| Optimizer-free structural boundary | Empty `torch.nn.Module` with JSON graph/rules/memory/cursor parent, candidate, reopen, and cold-payload restore | Structural custody only; no learned growth or preserved function claim |
| PhaseLaw | Synthetic integer phase origin and composition fixture | Does not establish a native checkpoint's learned phase semantics |
| SharedQwenMemory | Tiny random Qwen segmented-attention ABI, shared ancestor, private tails, and accounting | The segmented ABI excludes Hugging Face kernel exactness and live Session copy-on-write integration |

`python examples/software_debugger.py` runs the broad parameter-free debugger/evidence proof on
CPU without downloads. `python examples/structural_rewind.py` runs the optimizer-free structural
snapshot proof. Their reports distinguish supplied state, exact same-program replay, unassessed
cross-device/kernel parity, and unassessed semantic capability.

The release notebook run caught and corrected a debugger regression: an installed boundary
breakpoint stopped explicit `continue N` before N transitions. Explicit budgets now preserve
their exact transition count unless a watchpoint fires; bare continue and until remain
breakpoint-aware. A dedicated regression test preserves this distinction.

## Autoregressive model families and Mamba

The generic decoder adapter (gpt2, gpt_neox, phi, llama, mistral, mixtral, gemma-1, qwen2, qwen3)
and the Mamba-1 adapter were validated locally on macOS arm64 with Python 3.12 and CPU execution
on 2026-09-30:

- **The full suite is 152 tests** (53 new in `tests/test_families.py`): per-family stepped-vs-native
  logit parity, greedy equality vs native `generate`, same-program replay, fork isolation with a
  measurable intervention, operation-granularity phases, capture -> `LocalStore` -> fresh-process
  continuation across every family, incompatible-adapter refusal, `load()` dispatch, and fail-closed
  config refusals. Ruff lint and format pass on the added modules and tests.
- Tiny fp32 CPU fixtures: greedy stepping matches native `generate` token-for-token for all
  families; stepped-vs-native max absolute logit delta is a few ulps for the decoder families and
  ~2e-7 for Mamba; the Mamba single-token zero-state step is bit-exact vs the native prefill.

Real-weight validation ran on cached checkpoints (fp32, `tf32` off, batch one, greedy, eager;
`experiments/family_validation/run.py`). CUDA job `job-d75df96810fe` (RTX 4080) validated
SmolLM2-360M (llama), Qwen2.5-0.5B (qwen2), Qwen3-0.6B (qwen3), and Mamba-130m/370m; supplementary
CUDA job `job-beeea1a95e3c` validated pythia-70m/160m/410m (gpt_neox); gpt2 was validated on the
MacBook CPU because it is not cached on Beast. Every validated checkpoint reproduced the native
16-token greedy decode exactly and replayed a mid-layer cut exactly in a fresh interpreter; the
max absolute logit delta ranged from 1.7e-6 to 1.8e-4 and peak VRAM stayed at or below 2.4 GB.
phi, gemma, mixtral, and mistral had no usable cached checkpoint (mistral only as a 7B that exceeds
the VRAM guardrail) and were left to the tiny-fixture suite. Measured rows and job ids are in
`experiments/family_validation/results.json`. These runs do not establish cross-device or
cross-kernel byte equality or any semantic capability.

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
  transaction without torch, transformers, or any private package installed.

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
macOS arm64 CPU because it was not cached on the RTX 4080 host. Every validated checkpoint reproduced the native
16-token greedy decode exactly and replayed a mid-layer cut exactly in a fresh interpreter; the
max absolute logit delta ranged from 1.7e-6 to 1.8e-4 and peak VRAM stayed at or below 2.4 GB.
phi, gemma, mixtral, and mistral had no usable cached checkpoint (mistral only as a 7B that exceeds
the VRAM guardrail) and were left to the tiny-fixture suite. Measured rows and job ids are in
`experiments/family_validation/results.json`. These runs do not establish cross-device or
cross-kernel byte equality or any semantic capability.

## FLUX diffusion families and block-streamed residency

The FLUX.1 adapter, Klein-9B support, and block-streamed residency were validated on
2026-09-30. Tiny-fixture tests cover stepped-vs-native flow parity, streamed-vs-resident bitwise
equality, the frozen-model guard under streaming, the 9B-KV refusal, and FLUX.1 guidance gating.

Real-weight validation (`experiments/diffusion_family_validation/run.py`, bf16, 512×512, 4 steps,
seed 611, RTX 4080 16 GB) ran one family per job so only one large model was resident at a time:

- Klein-4B (`job-ece5315bc64b`): stepped, resident, and native per-step latent digests are all
  equal; streamed peak VRAM 0.65 GB against 7.5 GB resident.
- Klein-9B (`job-6be0bc01cc95`) and FLUX.1-schnell (`job-69b0e456925b`): the streamed trajectory
  equals the diffusers forward (run through accelerate sequential offload) at every step; streamed
  peak VRAM 1.07 GB and 0.81 GB, host RSS 22.2 GB and 26.3 GB. Both encode the prompt on CPU;
  sequential-offload encoding on the GPU peaked at 4.8 GB and 5.6 GB.
- All three replay the step-2 `StateCut` in a fresh interpreter to the in-process final digest.

An earlier Klein-4B job reported success while the runner inside it had failed on a device
mismatch; the device resolution was fixed and the launch wrapper now fails the job when any family
errors or any check is not exact. FLUX.1-dev was not validated on real weights. These runs do not
establish image quality, prompt adherence, or cross-device/kernel byte equality.

## LM block-streamed residency

Block-streamed residency for the decoder LM adapters (every decoder family, the dedicated Qwen
adapter, and Mamba) was validated on an RTX 4080 (16 GB) with `tf32` off, eager attention, batch
one, greedy (`experiments/lm_block_residency/run.py`). Tiny-fixture tests
(`tests/test_residency.py`) cover streamed-vs-resident bitwise equality of a greedy decode and the
`native_logits` comparator, cache residency, capture/fork/Act/compare/store round-trip/fresh-process
replay under streaming, the streamed-cut-vs-resident-adapter incompatibility, and the
`BlockResidency` guards, across all families on CPU.

- **Qwen2.5-0.5B, fp32 (`job-16a9286db5ab`).** The streamed 16-token greedy decode is **bitwise
  identical** to a resident adapter on the same card: identical tokens, bit-equal final logits, and
  a bit-equal `native_logits` comparator (max absolute logit delta 0.0). A mid-layer `StateCut`
  captured under streaming replays exactly in a fresh interpreter. Peak VRAM 1.90 GB resident vs
  0.53 GB streamed (3.5x); greedy rate 2.60 tok/s resident vs 1.91 tok/s streamed.
- **Qwen3-8B, bf16 (`job-6afb6e14cc9a`).** The 15.26 GB of weights leave no room for the CUDA
  context, activations, and key/value cache on the 15.54 GB usable card, so resident inference is
  infeasible. Streamed, the 16-token greedy decode runs at a **1.21 GB** peak and reproduces, token
  for token, a reference run of the same checkpoint and dtype through HF Transformers `generate`
  with accelerate `device_map` CPU-offload (10 GiB GPU cap), eager attention, greedy. A mid-layer
  cut replays exactly in a fresh process. Greedy rate 0.54 tok/s; one lease, 95 s wall.

The key/value cache stays resident on the execution device throughout; only frozen weights stream.
These runs establish exact streamed-vs-resident (and streamed-vs-offload-reference) decode on one
host; they do not establish cross-device/kernel byte equality or a serving-grade throughput.

## SAELens and TransformerLens interop

`examples/interop_saelens.py` ran on macOS arm64 CPU (fp32, GPT-2 small, `gpt2-small-res-jb`
layer 6) and was reproduced by an independent rerun: Saturn's `layer:6` carrier matches
TransformerLens `blocks.6.hook_resid_pre` within 2.2e-5; ablating SAE feature 14344 flips the next
token from " London" to " Paris", and the same ablation through a TransformerLens hook agrees on
next-token logits within 1.9e-4; both
branches replay byte-for-byte in a fresh process. The feature was chosen from 12 prompt/layer
configurations, and the dose response is not monotone (over-ablation returns " London").

## Gemma-2 adapter and circuit-tracer flagship

The decoder adapter gained Gemma-2 (`gemma2`): the native eager layer runs unchanged, and the
adapter applies the √hidden embedding scale and final-logit soft-capping on every path, including
block-streamed residency. Context is refused past `sliding_window`, so alternating local/global
attention never has to be emulated. The streamed causal-mask call is robust across transformers
versions (4.57.3 and 5.x take different keyword arguments). Tiny-fixture tests cover stepped-vs-native
parity, that soft-capping is load-bearing, local/global alternation, and config refusals.

Real-weight validation on `google/gemma-2-2b` (fp32, CUDA, RTX 4080, batch one, greedy) ran as
part of the circuit-tracer flagship: stepped-vs-native max absolute logit delta 4.5e-5, 16/16
greedy tokens exact, a mid-layer cut replays exactly in a fresh interpreter. Streamed residency
equals resident **bitwise** (max logit delta 0.0), both 16-token decodes equal `model.generate`,
and peak VRAM is 2.4 GB streamed vs 10.6 GB resident.

The flagship itself (`experiments/circuit_tracer/`) ran five CUDA jobs against a panel whose
sha256 was committed before the first job. Its results are summarized in
[circuit-tracer interop](circuit-tracer.md#real-weight-results-gemma-2-2b-gemma-scope-transcoders);
every number, CI, and job id is in `experiments/circuit_tracer/summary.json`. The integrated suite
is now **480 tests**, one skipped (the pre-existing interop case). These runs make no claim about
cross-device byte equality or about Gemma-2 contexts longer than the sliding window.

## Integrated release suite

With the AR families, interop, and diffusion families merged, the full suite is **165 tests**,
all passing on macOS arm64, Python 3.12, CPU, with the `ar`, `diffusion`, `train`, `interop`, and
`dev` extras installed; none are skipped.

## Metadata-delta route extension

The metadata-delta route (`saturn_pub.route`) adds a payload-free catalog of candidate deltas with
selection before hydration, a content-verified deduplicated resolver seam, and an
apply/replay-vs-native/exact-rollback receipt. It passes **12 local Python 3.12 tests**: the
stdlib-only metadata plane (payload-free guards, tensor-like rejection, round-trip fingerprints,
metadata-only selection equal to a dense scan, tag intersection, tamper detection, transport
accounting, and the `saturn-pub route` CLI) runs torch-free, and a torch lifecycle suite applies a
hydrated delta to a held-out decoder and a tiny FLUX block suffix, confirming a measured change and
a bit-exact rollback. With these, the integrated suite is **357 tests**, one skipped (the
pre-existing interop case).

Two CUDA leases on an RTX 4080 validated the route on real cached weights. Each saves a measured
carrier delta from a donor parent, selects it from metadata only (no bytes touched), hydrates it
twice through a content-verified resolver (sha256 + shape + dtype), applies it to a held-out
recipient parent, replays against the native continuation, and restores the parent cut.

CUDA job `job-926bdacce54c` ran qwen2 (Qwen2.5-0.5B, fp32) in 13.6 s at 1,897 MB peak VRAM. The
`hidden` delta (896 floats, 3,584 bytes) selected and hydrated with one resolver call for two
requests (`artifact_cache_hits` 1, `hydrated_bytes` 3,584 of 7,168 requested), changed the carrier
from the native continuation (L2 12.20), and rolled back bit-for-bit: `verified_exact` with the
restored fingerprint equal to the parent across all 31,622 restored tensor elements.

CUDA job `job-465b710d1fcf` ran flux2-klein (FLUX.2 Klein-4B, bfloat16, block-streamed) in 31.8 s at
11,310 MB peak VRAM. The `text` delta (a 1x512x3072 carrier, 3,145,728 bytes) selected and hydrated
once for two requests, changed the carrier from the native continuation (L2 2,321.99), and rolled
back bit-for-bit across all 9,230,345 restored tensor elements. Both runs report
`estimated_reduction` 0.984 for metadata-plus-selected-delta vs a dense catalog and carry no
embedded payloads.

These are exactness claims about the mechanics of one program on one device and dtype: a selected,
content-verified delta applied and rolled back exactly. The measured effect is a carrier change
signal, not a semantic label, and no cross-family or image-quality claim is made. Decoder prompts
are fixed token ids, not a tokenized natural-language battery.

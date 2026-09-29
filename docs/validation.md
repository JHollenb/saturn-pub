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

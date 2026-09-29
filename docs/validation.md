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

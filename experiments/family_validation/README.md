# AR family validation on real checkpoints

`run.py` wraps each cached checkpoint with `saturn_pub.adapters.load` and records, per model:

- **max_abs_logit_delta** -- max `|stepped logits - native full-forward logits|` at the last prompt
  position on a short prompt (bounded, not bit-exact: the stepped path prefills the prefix with one
  native forward and then advances the last token one layer at a time over an external cache).
- **greedy_agreement / greedy_exact** -- a 16-token greedy continuation compared token-for-token to
  the native `model.generate` greedy decode.
- **fresh_process_replay_exact** -- a mid-layer `StateCut` saved to a `LocalStore`, reloaded in a
  brand new interpreter, restored, continued 8 tokens, and checked against the in-process
  continuation.
- **peak_vram_mb / wall_s** -- per-checkpoint cost.

Settings: fp32, `tf32` disabled, batch one, greedy, eager attention, one checkpoint loaded at a
time and freed between families (`del` + `gc` + `empty_cache`). Only `saturn_pub` and public
frameworks run; the job launch and collection scripts are not part of the public package.
`results.json` in this directory holds the merged measured table and the job ids.

## What ran where

- **RTX 4080 host (fp32/cuda):** llama (SmolLM2-360M), qwen2 (Qwen2.5-0.5B), qwen3 (Qwen3-0.6B),
  mamba-1 (mamba-130m-hf, mamba-370m-hf); gpt_neox (pythia-70m/160m/410m) in a supplementary lease
  because the Hugging Face hub `main` ref for those repos resolved to a snapshot without a weights
  file, so the run pinned the fully-cached step-revision snapshot directories.
- **macOS arm64 (CPU, fp32):** gpt2 (not cached on the RTX 4080 host; validated locally instead of copying weights).

## Not validated on real weights (tiny-fixture parity only)

- **phi, gemma, mixtral** -- no usable checkpoint cached on the RTX 4080 host.
- **mistral** -- only Mistral-7B-v0.1 is cached; its fp32/fp16 weights exceed the RTX 4080 VRAM
  guardrail, so it is left to the tiny fixture rather than risking an out-of-memory lease.

These families still pass the tiny-fixture parity, generation, replay, isolation, and refusal tests
in `tests/test_families.py`; only their real-weight numbers are absent.

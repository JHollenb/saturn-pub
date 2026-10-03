# LM block-residency results

Measured on an RTX 4080 (16 GB) host, `tf32` off, eager attention, batch one, greedy. One
checkpoint per row; the reference model (for a model that does not fit resident) is loaded and
freed before the streamed adapter so host memory holds one model at a time. Each row's full record
is the JSON next to this file. These illustrate one development context; they are not an expected
universal outcome or a gate for your experiments.

| Model | dtype | Fits resident | Streamed vs native | Fresh-process replay | Peak VRAM resident | Peak VRAM streamed | tokens/s resident | tokens/s streamed | Job |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | fp32 | yes | **bitwise == resident** (0.0 logit Δ), 16/16 greedy exact | exact | 1900 MB | 541 MB | 2.60 | 1.91 | `job-16a9286db5ab` |
| Qwen3-8B | bf16 | no (15.26 GB weights / 15.54 GB usable) | 16/16 greedy exact vs accelerate CPU-offload reference | exact | n/a | 1211 MB | n/a | 0.54 | `job-6afb6e14cc9a` |

The 8 B weights alone (15.26 GB) leave no headroom for the CUDA context, activations, and key/value
cache on the 15.54 GB usable card, so resident inference is infeasible; the streamed decode peaks at
1.21 GB. The streamed tokens/s is low because the per-layer Python stepping over an external cache
is already slower than a fused forward and streaming adds a host->device copy per block -- this is a
debugging/inspection path, not a serving path.

- **Streamed vs native.** For a model that fits resident, the streamed decode is compared to a
  resident adapter on the same card: identical generated tokens, bit-equal final logits, and a
  bit-equal `native_logits` comparator (`streamed_equals_resident_exact`). For a model that does
  not fit resident, the reference is HF Transformers `AutoModelForCausalLM.generate` with accelerate
  `device_map` CPU-offload (`max_memory` 10 GiB GPU), eager attention, same checkpoint and dtype,
  greedy (`do_sample=False, num_beams=1`); the check is exact equality of the 16 generated token
  ids.
- **Fresh-process replay.** A mid-layer `StateCut` captured under streaming, saved to a
  `LocalStore`, reloaded in a brand-new interpreter, restored, and continued, checked
  token-for-token against the in-process continuation.
- **Peak VRAM** is `torch.cuda.max_memory_allocated` over the measured decode. **tokens/s** is the
  greedy decode rate (the per-layer Python stepping over an external cache is intrinsically slower
  than a fused forward; streaming adds the host->device copy on top).

Reproduce with plain Python from the [experiment README](../README.md). The scheduler launch and
collection scripts are not part of the public package.

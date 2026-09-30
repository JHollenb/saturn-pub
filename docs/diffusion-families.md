# FLUX diffusion families

Three native FLUX transformers are inspectable, branchable, and replayable through the same
state grammar as the rest of the toolkit:

| Family | Adapter | Transformer | Conditioner | Guidance |
| --- | --- | --- | --- | --- |
| FLUX.1 (`schnell`, `dev`) | `adapters.flux1.Flux1Adapter` | `FluxTransformer2DModel` | T5-XXL sequence + CLIP pooled | `schnell` none; `dev` gated on `config.guidance_embeds` |
| FLUX.2 Klein (`4B`, `9B`) | `adapters.flux2.Flux2KleinAdapter` | `Flux2Transformer2DModel` | single Qwen3 sequence | guidance one (distilled) |

Both adapters step the native transformer one **projection / joint (double) block / single block /
readout** cut at a time, apply the native `FlowMatchEulerDiscreteScheduler` update, and decode with
the native VAE. Execution is text-only and batch-one. They do not support reference images,
reference KV, CFG, or LoRA. The state closure (`latent`, block-cut `text`/`image` carriers,
conditioning, position ids, schedule, and the per-step transformer closure) is captured, forked,
saved, and replayed exactly like the Qwen and DDIM adapters.

## Block-streamed residency

A 16 GB card cannot hold the FLUX.1-schnell (23 GB) or Klein-9B (17 GB) transformer resident. The
adapters therefore own a **block-streamed residency** mode (`residency="streamed"`): the frozen
transformer weights stay in host memory and, for each native module, exactly that module's
parameters and buffers are copied to the device and run with `torch.func.functional_call`; the
transient device copies are released before the next block. Peak device memory is a single block
plus activations rather than the whole transformer.

Because the host weights never move, `Tensor.to(device)` is a byte-preserving copy, and
`functional_call` substitutes tensors without altering the module, a streamed block computes the
same bits a resident block computes on the same device — and the adapter's frozen-model guard
(parameter `id`/`_version`/`data_ptr`) still holds after every block. Streamed residency is
declared in the execution contract; the resident default omits the field so existing resident cuts
and receipts are byte-identical.

```python
import torch
from diffusers import Flux2KleinPipeline
from saturn_pub.adapters.flux2 import Flux2KleinAdapter

pipe = Flux2KleinPipeline.from_pretrained("/path/to/FLUX.2-klein-9B", torch_dtype=torch.bfloat16)
pipe.remove_all_hooks()
pipe.text_encoder.to("cpu")  # 16 GB Qwen3-8B: encode the prompt on CPU on a 16 GB card
pipe.vae.to("cpu")
adapter = Flux2KleinAdapter.from_pipeline(pipe, residency="streamed", device="cuda")
```

Encode the prompt once and free the text encoders (T5-XXL + CLIP for FLUX.1; Qwen3 for Klein)
before the transformer loop; block streaming keeps only one native block on the card at a time.

## Klein 9B-KV

`Flux2KleinKVPipeline` (Klein 9B-KV reference-conditioned serving) is refused with a reason: it
uses a distinct trajectory ABI — reference tokens ordered before target tokens, cached reference
key/values, and extract-mode causal attention — that this text-only, reference-free adapter cannot
reproduce exactly. A 9B-KV checkpoint whose `model_index.json` declares the plain
`Flux2KleinPipeline` loads and runs as ordinary text-to-image Klein and is accepted.

## Measured validation (RTX 4080, 16 GB, bf16)

`experiments/diffusion_family_validation` records these on real weights. The parity comparison is
per-step latent-digest equality against the ordinary diffusers forward (Klein-4B resident;
Klein-9B and FLUX.1-schnell reference via accelerate sequential CPU offload). Residency is the
bitwise-identical streamed-vs-resident trajectory on Klein-4B. Replay hydrates a mid-trajectory
`StateCut` in a separate interpreter and finishes it.

| Family | Job | Stepped = native | Streamed = resident | Fresh-process replay | Streamed peak VRAM | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- |
| Klein-4B | `job-ece5315bc64b` | exact | exact | exact | 0.65 GB | 9.3 GB |
| Klein-9B | `job-6be0bc01cc95` | exact | — | exact | 1.07 GB | 22.2 GB |
| FLUX.1-schnell | `job-69b0e456925b` | exact | — | exact | 0.81 GB | 26.3 GB |

512×512, 4 steps, seed 611; "exact" means every per-step latent digest (or the replayed final
digest) is equal. Klein-9B and schnell encode the prompt on CPU because their text encoders
peaked at 4.8 / 5.6 GB even through sequential offload. Full reports, job ids, source
revisions, and thumbnails are in `experiments/diffusion_family_validation/results/`.

Same-program replay is an exactness claim about mechanics; image quality, prompt adherence, and
portability across kernels, dtypes, and devices are not claimed and need their own evidence.

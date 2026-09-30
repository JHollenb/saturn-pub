# Pretrained model recipes

Offline examples need no downloads. These recipes use public framework loaders and your own
model cache or checkpoint. They are optional; validation of this release did not download weights.

## Qwen2.5

```python
from transformers import AutoTokenizer
from saturn_pub.adapters.qwen import QwenAdapter

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
# Set a revision for repeatable acquisition; actual loaded content is also fingerprinted.
tokenizer = AutoTokenizer.from_pretrained(model_id)
adapter = QwenAdapter.from_pretrained(model_id)
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain why the sky is blue."}],
    tokenize=False,
    add_generation_prompt=True,
)
session = adapter.session(tokenizer.encode(text))
adapter.generate(session, tokens=32)
print(tokenizer.decode(session.read("tokens")[0].tolist()))
```

The budget is fixed; this helper does not stop on EOS. Inspect/capture layer boundaries using
`session.continue_()`. For low-overhead uninstrumented serving, call the native loaded model's
`generate` API; the proof-oriented session intentionally copies and fingerprints state.

## Other decoder families and Mamba via `load`

`saturn_pub.adapters.load` selects a native adapter by `model_type`. Autoregressive decoder
families (gpt2, gpt_neox/Pythia, phi, llama, mistral, mixtral, gemma-1, qwen2, qwen3) resolve to
the generic `DecoderAdapter`; `mamba` resolves to `MambaAdapter`. Same session grammar as above.

```python
from saturn_pub.adapters import load

adapter = load("EleutherAI/pythia-70m")  # gpt_neox: parallel residual, partial rotary
session = adapter.session([101, 202, 303])
adapter.generate(session, tokens=16)
```

```python
from saturn_pub.adapters import load

adapter = load("state-spaces/mamba-130m-hf")  # per-layer conv + recurrent state cuts
session = adapter.session([101, 202, 303])
adapter.generate(session, tokens=16)
```

`load` requires eager attention (forced on load) and refuses an unregistered `model_type` or a
config outside a family's registered rule, naming the rule. Gemma applies its embedding normalizer
inside the native scaled embedding; GPT-2 adds native learned position embeddings; Mistral is
accepted only while the context stays within `sliding_window`; Mixtral MoE parity is bounded, not
bit-exact. Construct a specific adapter directly (`DecoderAdapter.from_pretrained(path)`,
`MambaAdapter.from_pretrained(path)`) when you want to pin the class. Qwen2/Qwen2.5 also work
through the dedicated `QwenAdapter` and through the generic `DecoderAdapter`.

## Your Stable Diffusion checkpoint

```python
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from saturn_pub.adapters.diffusion import DiffusionAdapter

pipeline = StableDiffusionPipeline.from_pretrained("/path/to/your/checkpoint")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
adapter = DiffusionAdapter.from_pipeline(pipeline)
prompt_embeds, _ = pipeline.encode_prompt(
    "a small red boat",
    device=adapter.device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=False,
)
session = adapter.session(steps=20, seed=7, conditioning=prompt_embeds)
session.continue_(20)
image_tensor = adapter.decode(session)
```

This uses supplied prompt conditioning and guidance scale one. It does not reproduce a
pipeline's default CFG settings. Use a checkpoint whose license permits your intended use.
Memory needs depend on model, dtype, resolution, captured cuts, and retained branches.
CPU is the reference device; validate another backend's numerical behavior separately.

## FLUX.2 Klein: inspect and replay a supplied hotfix

`Flux2KleinAdapter` executes the native distilled transformer one projection/block/readout at a
time, followed by the native deterministic flow Euler update. It supports text-only, batch-one,
guidance-one execution. It does not support reference images, reference KV, CFG, LoRA, or
offload hooks during block execution. Place the transformer on the execution device and the
VAE on CPU before wrapping it; decoder normalization is bound to the native phase-offloaded
CPU-statistics program. The example moves VAE weights only during decoding.

The supplied-writer example takes a checkpoint and an NPZ containing `mean`, `basis`, `weights`,
and `bias`. It does not bundle a patch or a model. The operation is
`text += dose * (((text - mean) @ basis) @ weights + bias)` at `step:0/after:joint.2`.
Its labelled feedback selects a scalar dose; the writer payload itself is supplied.

```bash
python examples/flux2_hotfix.py \
  --model /path/to/FLUX.2-klein-4B --revision YOUR_REVISION \
  --package /path/to/writer.npz --prompt 'Your development prompt' \
  --expected-count 5 --seed 611 --local-files-only
```

This real-model example uses CUDA and counts red connected components in the rendered RGB.
Use it only for tasks appropriate to that instrument, inspect the images, and replace the
evaluator for another task. It compares against an ordinary pipeline render, records short
feedback-controlled dose proposals, preserves rejected measurements, and checks zero dose,
wrong timing, commit, exact restore, and uninstall.

Outputs include `report.json`, images, receipts, and a content-addressed cut under `state/`.
Replay a saved cut in a separate process using its digest and the previously selected dose:

```bash
python examples/flux2_hotfix.py \
  --model /path/to/FLUX.2-klein-4B --revision YOUR_REVISION \
  --package /path/to/writer.npz --prompt 'Unused during replay' \
  --local-files-only --replay CUT_DIGEST --dose SELECTED_DOSE
```

Replay loads supplied conditioning from the cut and skips prompt encoding and feedback selection.
The model and VAE are content-bound; weights must be reconstructed separately. Same-program
replay is an exactness claim; repair quality and portability need their own evidence.

## FLUX.2 Klein-9B: block-streamed residency

Klein-9B is a plain distilled `Flux2KleinPipeline` (8 joint + 24 single blocks, width 4096) whose
17 GB transformer does not fit a 16 GB card resident. `Flux2KleinAdapter` streams it one native
block at a time from host memory (`residency="streamed"`), so peak device memory is a single block
plus activations. The 4B behavior, API, and receipts are unchanged; only streamed residency adds a
declared field to the execution contract.

```python
import torch
from diffusers import Flux2KleinPipeline
from saturn_pub.adapters.flux2 import Flux2KleinAdapter

pipe = Flux2KleinPipeline.from_pretrained("/path/to/FLUX.2-klein-9B", torch_dtype=torch.bfloat16)
# The Qwen3-8B encoder is 16 GB in bf16; on a 16 GB card encode the one prompt on CPU.
prompt_embeds = pipe.encode_prompt(
    prompt="a small red boat", device="cpu", num_images_per_prompt=1
)[0]
# Keep the transformer in host memory; the adapter streams one block at a time.
pipe.vae.to("cpu")
adapter = Flux2KleinAdapter.from_pipeline(pipe, residency="streamed", device="cuda")
```

`Flux2KleinKVPipeline` (9B-KV reference-conditioned serving) is refused with a reason; see
[FLUX diffusion families](diffusion-families.md).

## FLUX.1-schnell: block-streamed residency

`Flux1Adapter` steps the native `FluxTransformer2DModel` at the same projection/joint/single/readout
cuts, with the native flow Euler update and VAE. FLUX.1 carries two conditioners: a T5-XXL sequence
(`encoder_hidden_states`) and a CLIP pooled vector (`pooled_projections`). `schnell` has no guidance
embedding; guidance-distilled variants (`dev`) gate one on `config.guidance_embeds` (tested on a
tiny fixture; a real `dev` checkpoint was not available for this release). The 23 GB schnell
transformer is streamed one block at a time.

```python
import torch
from diffusers import FluxPipeline
from saturn_pub.adapters.flux1 import Flux1Adapter

pipe = FluxPipeline.from_pretrained("/path/to/FLUX.1-schnell", torch_dtype=torch.bfloat16)
prompt_embeds, pooled, text_ids = pipe.encode_prompt(
    prompt="a small red boat",
    prompt_2="a small red boat",
    device="cpu",  # T5-XXL + CLIP do not fit beside other work on a 16 GB card
    num_images_per_prompt=1,
    max_sequence_length=512,
)
pipe.vae.to("cpu")
adapter = Flux1Adapter.from_pipeline(pipe, residency="streamed", device="cuda")
session = adapter.session(
    latent=latent,
    conditioning=prompt_embeds,
    pooled=pooled,
    img_ids=latent_image_ids.unsqueeze(0),
    txt_ids=text_ids.unsqueeze(0),
    timesteps=timesteps,
    sigmas=pipe.scheduler.sigmas,
    guidance=None,
)
adapter.finish(session)
image = adapter.decode(session, height=512, width=512)
```

`experiments/diffusion_family_validation/run.py` runs the memory-disciplined encode-then-stream
recipe end to end (prepare latents, ids, and schedule; stepped-vs-native parity; fresh-process
replay; decode). Use `--local-files-only` with a local checkpoint whose license permits your use.
The full transformer does not fit a 16 GB card resident; block streaming is required there, and
CUDA is required for the real checkpoints. Measured on an RTX 4080 (16 GB): encoding through
accelerate sequential offload still peaked at 4.8 GB (Klein-9B) and 5.6 GB (schnell), so both encode
on CPU; the streamed 4-step trajectory then peaked at 1.07 GB and 0.81 GB of VRAM with 22–26 GB of
host RSS. See [measured results](../experiments/diffusion_family_validation/results/README.md).

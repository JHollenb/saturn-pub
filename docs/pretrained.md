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

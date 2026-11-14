"""Merge the substrate-bind LoRA into Qwen2.5-0.5B (prereg_hopmath.json §6 negative control).

python merge_lora.py BASE_DIR ADAPTER_PT OUT_DIR
W += scale * B @ A per targeted module, in fp32, then saved in the base checkpoint's dtype with the
base tokenizer. Prints the merged module count and the relative weight change only.
"""
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

BASE, ADP, OUT = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
dtype = model.config.torch_dtype if isinstance(model.config.torch_dtype, torch.dtype) else torch.bfloat16
d = torch.load(ADP, map_location="cpu", weights_only=False)
print("meta", {k: v for k, v in d["meta"].items() if k != "bank"})
n, rel = 0, []
with torch.no_grad():
    for name, t in d["lora"].items():
        mod = model.get_submodule(name)
        delta = float(t["scale"]) * (t["B"].float() @ t["A"].float())
        assert delta.shape == mod.weight.shape, (name, delta.shape, mod.weight.shape)
        rel.append(float(delta.norm() / mod.weight.norm()))
        mod.weight += delta
        n += 1
print(f"merged {n} modules; relative |dW|/|W| median {sorted(rel)[len(rel) // 2]:.4f} max {max(rel):.4f}")
model.to(dtype).save_pretrained(OUT, safe_serialization=True)
for f in Path(BASE).iterdir():
    if f.name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "generation_config")):
        shutil.copy(f, Path(OUT) / f.name)
print("saved", OUT, dtype)

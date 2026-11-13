"""Model builders for the engine: random tiny eager models and local checkpoint loading.

Nothing here imports the legacy flat debugger module.
"""

from __future__ import annotations

from typing import Any

import torch

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPES:
        raise ValueError("dtype must be bf16 or fp32")
    return DTYPES[dtype]


def tiny_model(
    family: str,
    dtype: Any = "bf16",
    *,
    layers: int = 2,
    seed: int = 113,
    vocab_size: int = 32,
) -> Any:
    """A random tiny eager CPU model in eval mode.

    ``qwen2``: hidden 16, 2 heads, 1 KV head, ff 32. ``qwen3``: hidden 32, 4 heads, 2 KV heads,
    head_dim 8, ff 64. The defaults reproduce the weights the legacy flat-module tiny builders
    produced for the same seed.
    """
    target = _dtype(dtype)
    if family == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(seed)
        config = Qwen2Config(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=256,
            attn_implementation="eager",
        )
        return Qwen2ForCausalLM(config).eval().to(target)
    if family == "qwen3":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        torch.manual_seed(seed)
        config = Qwen3Config(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            attn_implementation="eager",
        )
        return Qwen3ForCausalLM(config).eval().to(target)
    raise ValueError("tiny family must be qwen2 or qwen3")


def load_local(path: str, *, dtype: Any = "bf16", device: Any = "cpu") -> Any:
    """Load a local Hugging Face causal LM checkpoint with eager attention, in eval mode.

    Offline only (``local_files_only=True``); never downloads.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        attn_implementation="eager",
    )
    return model.eval().to(device=torch.device(device), dtype=_dtype(dtype))

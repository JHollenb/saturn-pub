"""Model builders for the engine: random tiny eager models and local checkpoint loading.

Nothing here imports the legacy flat debugger module.
"""

from __future__ import annotations

from typing import Any

import torch

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_FAMILIES = ("qwen2", "qwen3", "llama", "mistral")


def _dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in DTYPES:
        raise ValueError("dtype must be bf16 or fp32")
    return DTYPES[dtype]


def cast_weights(model: Any, dtype: torch.dtype) -> Any:
    """Cast the parameters of ``model`` to ``dtype`` and leave its buffers as built.

    ``module.to(dtype)`` also casts floating buffers; RoPE's ``inv_freq`` is float32 by
    construction and a bf16 copy of it is a different model (see ``load_local``). This is what
    loading a checkpoint with a dtype does: weights in the dtype, buffers untouched.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype)
    return model


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
    produced for the same seed. ``llama``: hidden 32, 4 heads, 2 KV heads, head_dim 8, ff 64, no
    attention or MLP bias. ``mistral``: hidden 32, 4 heads, 1 KV head (MQA), head_dim 8, ff 64,
    a 128-position sliding window (the windowed mask path, as in Mistral-7B-v0.1).
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
        return cast_weights(Qwen2ForCausalLM(config).eval(), target)
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
        return cast_weights(Qwen3ForCausalLM(config).eval(), target)
    if family == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(seed)
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=256,
            attention_bias=False,
            mlp_bias=False,
            attn_implementation="eager",
        )
        return cast_weights(LlamaForCausalLM(config).eval(), target)
    if family == "mistral":
        from transformers import MistralConfig, MistralForCausalLM

        torch.manual_seed(seed)
        config = MistralConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=256,
            sliding_window=128,
            attn_implementation="eager",
        )
        return cast_weights(MistralForCausalLM(config).eval(), target)
    raise ValueError(f"tiny family must be one of {', '.join(TINY_FAMILIES)}")


def load_local(path: str, *, dtype: Any = "bf16", device: Any = "cpu") -> Any:
    """Load a local Hugging Face causal LM checkpoint with eager attention, in eval mode.

    Offline only (``local_files_only=True``); never downloads.
    """
    from transformers import AutoModelForCausalLM

    # The dtype is given to the loader and the module is then moved by device only:
    # ``module.to(dtype=...)`` also casts buffers, and a bf16 copy of RoPE's float32
    # ``inv_freq`` (relative error up to 3.7e-3) is a different model from the one Hugging
    # Face runs. Measured on Qwen2.5-1.5B: the declared margin moved by one bf16 step.
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        attn_implementation="eager",
        dtype=_dtype(dtype),
    )
    return model.eval().to(torch.device(device))

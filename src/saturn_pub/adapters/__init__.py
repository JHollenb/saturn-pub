"""Optional native adapters. Import a specific module to load its framework.

``load`` is a small dispatcher that selects a native adapter by ``model_type``:

    from saturn_pub.adapters import load
    adapter = load("EleutherAI/pythia-70m")

Autoregressive decoder families (gpt2, gpt_neox, phi, llama, mistral, mixtral, gemma, qwen2,
qwen3) resolve to ``decoder.DecoderAdapter``; ``mamba`` resolves to ``mamba.MambaAdapter``.
Qwen2/Qwen2.5 also work through the dedicated ``qwen.QwenAdapter``. Importing this module does
not import PyTorch; the framework loads only when ``load`` (or a specific adapter) is used.
"""

from __future__ import annotations

import os
from typing import Any

_DECODER_FAMILIES = frozenset(
    {"gpt2", "gpt_neox", "phi", "llama", "mistral", "mixtral", "gemma", "qwen2", "qwen3"}
)
_MAMBA_FAMILIES = frozenset({"mamba"})
_PEEK_KEYS = (
    "revision",
    "local_files_only",
    "cache_dir",
    "token",
    "trust_remote_code",
    "subfolder",
)


def supported_families() -> tuple[str, ...]:
    """Model types the dispatcher can load."""
    return tuple(sorted(_DECODER_FAMILIES | _MAMBA_FAMILIES))


def _adapter_class(model_type: str | None) -> type:
    if model_type in _MAMBA_FAMILIES:
        from .mamba import MambaAdapter

        return MambaAdapter
    if model_type in _DECODER_FAMILIES:
        from .decoder import DecoderAdapter

        return DecoderAdapter
    raise ValueError(
        f"unregistered model_type: {model_type!r}; supported families are "
        f"{', '.join(supported_families())}"
    )


def resolve_device(device: Any) -> Any:
    """``"auto"`` (the default) selects CUDA when available, else leaves placement unchanged."""
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else None
    return device


def load(
    path_or_model: Any, *, granularity: str = "layer", device: Any = "auto", **kwargs: Any
) -> Any:
    """Return a native adapter for a checkpoint path or an already-loaded HF model.

    ``device`` places the model on that device before the adapter freezes its identity. The
    default ``"auto"`` uses CUDA when available (so a GPU host is used by default instead of an
    accidental slow CPU run); pass ``device="cpu"`` to force CPU or ``None`` to keep Hugging
    Face's placement.
    """
    device = resolve_device(device)
    if isinstance(path_or_model, (str, os.PathLike)):
        from transformers import AutoConfig

        peek = {key: kwargs[key] for key in _PEEK_KEYS if key in kwargs}
        model_type = AutoConfig.from_pretrained(str(path_or_model), **peek).model_type
        adapter_class = _adapter_class(model_type)
        return adapter_class.from_pretrained(
            str(path_or_model), granularity=granularity, device=device, **kwargs
        )
    model = path_or_model
    if device is not None:
        model = model.to(device)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return _adapter_class(model_type)(model, granularity=granularity)

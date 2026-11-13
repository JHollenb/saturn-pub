"""Factor observed, positioned Q/K/V sources without changing the merge shape."""
from __future__ import annotations


def attention_sources(*, base, changed, text_rows, mask):
    """Bits choose base Q, text K/V and image K/V, respectively.

    Inputs have already received native Q/K normalization and matching RoPE.
    This constructs an intervention; it does not forecast the final image.
    """
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    if len(mask) != 3 or any(bit not in "01" for bit in mask):
        raise ValueError("attention mask must select Q/text-KV/image-KV")
    values = [source[key] for source in (base, changed) for key in ("q", "k", "v")]
    if any(not isinstance(value, torch.Tensor) or value.ndim != 4 for value in values):
        raise ValueError("requires rank-four observed Q/K/V")
    first = values[0]
    if any(value.shape != first.shape or value.dtype != first.dtype or value.device != first.device for value in values):
        raise ValueError("matched Q/K/V geometry, dtype and device required")
    if not 0 < text_rows < first.shape[1]:
        raise ValueError("invalid text/image boundary")
    q = (base if mask[0] == "1" else changed)["q"]
    k, v = changed["k"].clone(), changed["v"].clone()
    for bit, selected in ((mask[1], slice(None, text_rows)), (mask[2], slice(text_rows, None))):
        if bit == "1":
            k[:, selected] = base["k"][:, selected]
            v[:, selected] = base["v"][:, selected]
    result = dispatch_attention_fn(q, k, v, attn_mask=None, backend=None, parallel_config=None)
    return {"q": q, "k": k, "v": v,
            "attention": result.flatten(2, 3).to(q.dtype)}

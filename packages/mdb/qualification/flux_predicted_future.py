"""Fixed-parent counterfactual projection using a sealed image MLP field.

This consumer reuses observed recipient attention and text operands. It does
not recompute attention, normalization or MLP input projections. Reuse is valid
only for image-MLP substitutions at the same declared parent and clock.
"""
from __future__ import annotations


def consume_cached(block, cut, native, mlp_field):
    import torch

    p = native["ports"]
    original = p["original_concat"]
    t, h = native["text_rows"], native["attention_width"]
    if (native["input_parent"] != cut.fingerprint or cut.execution_point["phase"] != "single"
            or cut.execution_point["index"] != 6 or cut.execution_point["logical_step"] != 3
            or not isinstance(mlp_field, torch.Tensor) or mlp_field.shape != original[:, t:, h:].shape
            or mlp_field.dtype != original.dtype or mlp_field.device != original.device
            or not bool(torch.isfinite(mlp_field).all())):
        raise ValueError("cached consumer parent/clock/MLP profile differs")
    with torch.inference_mode():
        used = original.clone()
        used[:, t:, h:] = mlp_field
        projection = block.attn.to_out(used)
        updated = p["residual"] + p["gate"] * projection
    if not torch.equal(updated[:, :t], native["text"]):
        raise ValueError("cached image-only consumer changed recipient text")
    return {"image": updated[:, t:], "text": updated[:, :t], "projection": projection,
        "input_parent": cut.fingerprint,
        "scope": "Same-parent native to_out and gated residual only; observed attention/all text operands reused"}


def field_metrics(a, b):
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    rms = float(np.sqrt(np.mean(d * d)))
    reference = float(np.sqrt(np.mean(b * b)))
    return {"values": int(a.size), "exact_values": int(np.count_nonzero(d == 0)),
        "rms": rms, "relative_rms": rms / reference if reference else None,
        "l2": float(np.linalg.norm(d)), "max_abs": float(np.max(np.abs(d)))}


def dose_values():
    return [index / 8 for index in range(-4, 13)]


__all__ = ["consume_cached", "field_metrics", "dose_values"]

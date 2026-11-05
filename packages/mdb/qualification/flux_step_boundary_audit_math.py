"""Saved-only independent readout and Euler recurrences for the step-boundary cut.

This file deliberately does not import the campaign's mathematical predictor.
Inputs are complete archived tensors; no target activation is substituted.
"""
from __future__ import annotations

import hashlib

import numpy as np


def bf16(value):
    """Round binary32 to nearest-even BF16 and retain the exact widened value."""
    x = np.asarray(value, dtype=np.float32)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("nonfinite/empty independent operand")
    bits = np.ascontiguousarray(x).view(np.uint32)
    rounded = (bits + np.uint32(32767) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    result = rounded.view(np.float32).reshape(x.shape)
    if not np.isfinite(result).all():
        raise ValueError("BF16 rounding overflow")
    return result


def exact_bf16(value):
    x = np.asarray(value)
    if (x.dtype != np.float32 or not x.size or not np.isfinite(x).all()
            or np.any(np.ascontiguousarray(x).view(np.uint32) & 65535)):
        raise ValueError("independent operand must be an exact finite BF16 widening")
    return x


def identity(value):
    x = exact_bf16(value)
    return {"shape": list(x.shape), "dtype": "float32-exact-bf16-widening",
        "bf16_le_sha256": hashlib.sha256((np.ascontiguousarray(x).view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest(),
        "fp32_le_sha256": hashlib.sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest()}


def _linear(x, weight, bias, *, rows=128):
    w = exact_bf16(weight)
    if w.ndim != 2 or x.shape[-1] != w.shape[1]:
        raise ValueError("independent linear shape differs")
    b = None if bias is None else exact_bf16(bias)
    if b is not None and b.shape != (w.shape[0],):
        raise ValueError("independent linear bias shape differs")
    out = np.empty((*x.shape[:-1], w.shape[0]), np.float32)
    wide = x.astype(np.float64)
    for first in range(0, w.shape[0], rows):
        stop = min(first + rows, w.shape[0])
        result = wide @ w[first:stop].astype(np.float64).T
        if b is not None:
            result += b[first:stop].astype(np.float64)
        out[..., first:stop] = bf16(result)
    return out


def independent_readout(image, temb, *, modulation_weight, modulation_bias,
                        projection_weight, projection_bias, eps,
                        norm_weight=None, norm_bias=None):
    """Population LayerNorm, SiLU-conditioned scale/shift, then output linear."""
    x, t = exact_bf16(image), exact_bf16(temb)
    if (x.ndim != 3 or t.ndim != 2 or x.shape[0] != t.shape[0]
            or not isinstance(eps, (int, float)) or not np.isfinite(eps) or eps <= 0):
        raise ValueError("independent readout geometry differs")
    z = t.astype(np.float64)
    sigmoid = np.empty(z.shape, np.float64)
    positive = z >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-z[positive]))
    e = np.exp(z[~positive])
    sigmoid[~positive] = e / (1 + e)
    activated = bf16(z * sigmoid)
    modulation = _linear(activated, modulation_weight, modulation_bias)
    if modulation.shape != (x.shape[0], 2 * x.shape[-1]):
        raise ValueError("independent readout modulation shape differs")
    scale, shift = np.split(modulation, 2, axis=-1)
    wide = x.astype(np.float64)
    centered = wide - wide.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
    if norm_weight is not None:
        gamma = exact_bf16(norm_weight)
        if gamma.shape != (x.shape[-1],):
            raise ValueError("independent LayerNorm scale shape differs")
        normalized *= gamma.astype(np.float64)
    if norm_bias is not None:
        beta = exact_bf16(norm_bias)
        if beta.shape != (x.shape[-1],):
            raise ValueError("independent LayerNorm bias shape differs")
        normalized += beta.astype(np.float64)
    normalized = bf16(normalized)
    affine = bf16(bf16(normalized * bf16(1 + scale)[:, None]) + shift[:, None])
    return {"noise": _linear(affine, projection_weight, projection_bias),
        "normalized": normalized, "affine": affine, "modulation": modulation,
        "activated_temb": activated}


def independent_flow_commit(latent, noise, sigma, sigma_next):
    """Mirror weak-scalar promotion, including BF16(dt) before multiplication."""
    z, n = exact_bf16(latent), exact_bf16(noise)
    if z.shape != n.shape:
        raise ValueError("independent flow latent/noise geometry differs")
    a, b = np.asarray(sigma, dtype=np.float32), np.asarray(sigma_next, dtype=np.float32)
    if a.shape != () or b.shape != () or not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("independent flow sigmas must be finite scalars")
    dt = np.subtract(b, a, dtype=np.float32)
    dt_bf16 = bf16(dt)
    delta = bf16(np.multiply(dt_bf16, n, dtype=np.float32))
    out = bf16(np.add(z, delta, dtype=np.float32))
    return {"latent": out, "delta": delta, "dt_fp32": float(dt), "dt_bf16": float(dt_bf16)}

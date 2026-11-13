"""Independent NumPy FP64 equations for the Klein image SwiGLU write.

The native prefix supplies x (already normalized/modulated), residual r and
gate. External read-only FP32 weights contain the actual BF16 parameter values
widened without information loss. Native BF16 rounding is not emulated here.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _native_array(value, name, *, readonly=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError(name + " must be a NumPy FP32 array containing supplied native values")
    if readonly and value.flags.writeable:
        raise ValueError(name + " must be read-only")
    return value


def _linear(value, weight, bias):
    """Tile output rows so widening never copies the complete large matrix."""
    output = np.empty((value.shape[0], weight.shape[0]), dtype=np.float64)
    for start in range(0, weight.shape[0], 256):
        stop = start + 256
        tile = np.array(weight[start:stop], dtype=np.float64, copy=True)
        if not np.isfinite(tile).all():
            raise ValueError("nonfinite supplied linear weight")
        output[:, start:stop] = value @ tile.T
        if bias is not None:
            output[:, start:stop] += bias[start:stop]
    return output


def _silu(value):
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = value[positive] / (1. + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = value[~positive] * exponential / (1. + exponential)
    return result


def _summary(value):
    norm = float(np.linalg.norm(value.reshape(-1)))
    if not np.isfinite(norm):
        raise ValueError("nonfinite numerical summary")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "l2": norm,
            "rms": float(norm / np.sqrt(value.size)), "minimum": float(value.min()), "maximum": float(value.max())}


def forecast(x, r, gate, weights, *, chunk_tokens=32):
    """Return only FP64 y, gated write, residual and small numerical summaries.

    x/r are supplied FP32 [1,N,H]; gate is FP32 [1,1,H] or [1,N,H]. Required
    read-only FP32 weights are linear_in[2I,H], linear_out[H,I]. Optional
    bias_in[2I] and bias_out[H] are read-only FP32 arrays or None. The first
    projection half is SiLU-activated and multiplied by the second half.

    At most chunk_tokens (1..32) positions and 256 widened matrix rows are
    processed at a time. No native module is called and no parameter matrix or
    wide intermediate is returned. Inputs and external weights are unmodified.
    """
    if type(chunk_tokens) is not int or not 1 <= chunk_tokens <= 32:
        raise ValueError("chunk_tokens must be an integer in 1..32")
    if (not isinstance(weights, Mapping) or not {"linear_in", "linear_out"} <= set(weights)
            or set(weights) - {"linear_in", "linear_out", "bias_in", "bias_out"}):
        raise ValueError("declare linear_in/linear_out and optional bias_in/bias_out only")
    x, r, gate = (_native_array(value, name) for value, name in ((x, "x"), (r, "r"), (gate, "gate")))
    incoming = _native_array(weights["linear_in"], "linear_in", readonly=True)
    outgoing = _native_array(weights["linear_out"], "linear_out", readonly=True)
    if (x.ndim != 3 or x.shape[0] != 1 or not x.shape[1] or not x.shape[2] or r.shape != x.shape
            or gate.shape not in {(1, 1, x.shape[2]), x.shape} or incoming.ndim != 2 or outgoing.ndim != 2
            or incoming.shape[1] != x.shape[2] or not incoming.shape[0] or incoming.shape[0] % 2
            or outgoing.shape != (x.shape[2], incoming.shape[0] // 2)):
        raise ValueError("x/r/gate or SwiGLU projection geometry differs")
    bias_in, bias_out = weights.get("bias_in"), weights.get("bias_out")
    for name, value, shape in (("bias_in", bias_in, (incoming.shape[0],)), ("bias_out", bias_out, (outgoing.shape[0],))):
        if value is not None:
            _native_array(value, name, readonly=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(name + " geometry or finite contract differs")
    if any(not np.isfinite(value).all() for value in (x, r, gate)):
        raise ValueError("nonfinite observed x/r/gate")
    y = np.empty(x.shape, dtype=np.float64)
    intermediate = incoming.shape[0] // 2
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, x.shape[1], chunk_tokens):
            inputs = np.array(x[0, start:start + chunk_tokens], dtype=np.float64, copy=True)
            projected = _linear(inputs, incoming, bias_in)
            a, b = projected[:, :intermediate], projected[:, intermediate:]
            y[0, start:start + chunk_tokens] = _linear(_silu(a) * b, outgoing, bias_out)
        write = np.asarray(gate, dtype=np.float64) * y
        residual = np.asarray(r, dtype=np.float64) + write
    if any(not np.isfinite(value).all() for value in (y, write, residual)):
        raise ValueError("nonfinite independent MLP arithmetic")
    summaries = {name: _summary(value) for name, value in (("y", y), ("write", write), ("residual", residual))}
    for value in (y, write, residual):
        value.flags.writeable = False
    return {"y": y, "write": write, "residual": residual, "summaries": summaries}

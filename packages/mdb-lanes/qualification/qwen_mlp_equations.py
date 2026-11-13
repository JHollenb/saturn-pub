"""Independent FP64 last-token Qwen2 MLP/readout equations.

Input is the post-attention residual immediately before the final decoder
layer's post_attention_layernorm. Weights and selected vocabulary row IDs are
supplied explicitly. This module imports no model or native execution runtime.
The reference does not emulate the native FP32 rounding schedule.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np

WEIGHT_NAMES = frozenset({"postnorm", "gate_proj", "up_proj", "down_proj", "finalnorm", "selected_lm_head"})


def _array(value, name):
    if not isinstance(value, np.ndarray) or value.dtype.kind != "f":
        raise ValueError(name + " must be a floating NumPy array")
    array = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    array.flags.writeable = False
    return array


def prepare_weights(weights):
    """Return independent read-only FP64 copies after checking the full ABI."""
    if not isinstance(weights, Mapping) or set(weights) != WEIGHT_NAMES:
        raise ValueError("expected exactly: " + ", ".join(sorted(WEIGHT_NAMES)))
    result = {name: _array(weights[name], name) for name in sorted(WEIGHT_NAMES)}
    postnorm, gate = result["postnorm"], result["gate_proj"]
    if postnorm.ndim != 1 or not postnorm.size or gate.ndim != 2 or not gate.shape[0]:
        raise ValueError("postnorm [H] and gate_proj [I,H] must be nonempty")
    hidden, intermediate = postnorm.size, gate.shape[0]
    if (gate.shape != (intermediate, hidden) or result["up_proj"].shape != gate.shape
            or result["down_proj"].shape != (hidden, intermediate)
            or result["finalnorm"].shape != (hidden,)
            or result["selected_lm_head"].ndim != 2
            or not result["selected_lm_head"].shape[0]
            or result["selected_lm_head"].shape[1] != hidden):
        raise ValueError("MLP/norm/selected-head weight geometry differs")
    return result


def silu(value):
    """Stable SiLU, including large finite positive and negative arguments."""
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0
    result[positive] = value[positive] / (1. + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    result[~positive] = value[~positive] * negative_exp / (1. + negative_exp)
    return result


def _rms_norm(value, weight, eps):
    with np.errstate(over="ignore", invalid="ignore"):
        rms = np.sqrt(np.mean(np.square(value), axis=-1, keepdims=True, dtype=np.float64) + eps)
        normalized = value / rms * weight
    if not np.isfinite(rms).all() or not np.isfinite(normalized).all():
        raise ValueError("RMSNorm arithmetic produced a nonfinite value")
    return rms, normalized


def _freeze(value):
    if isinstance(value, dict):
        return {key: _freeze(child) for key, child in value.items()}
    if isinstance(value, np.ndarray):
        if not np.isfinite(value).all():
            raise ValueError("forecast arithmetic produced a nonfinite value")
        value.flags.writeable = False
    return value


def forecast(cut, weights, eps, doses=(1., 0., .5), *, candidate_ids):
    """Forecast all declared MLP doses from one genuine last-token cut.

    ``cut`` has shape [H] or [1,1,H]. Canonical weights are postnorm[H],
    gate_proj[I,H], up_proj[I,H], down_proj[H,I], finalnorm[H], and
    selected_lm_head[K,H]. ``candidate_ids`` are the explicit K vocabulary IDs
    in that exact row order. Both RMSNorm operations use the declared epsilon.

    For each dose d, residual = cut + d * down(SiLU(gate(postnorm(cut)))
    * up(postnorm(cut))); logits = finalnorm(residual) @ selected_lm_head.T.
    All arrays are independent FP64/read-only values. No native suffix is run.
    Native scaling of native FP32 MLP output has a different rounding schedule;
    its residuals against this reference remain numerical observations.
    """
    if isinstance(eps, bool) or not isinstance(eps, Real) or not np.isfinite(eps) or eps <= 0:
        raise ValueError("RMS epsilon must be finite and positive")
    weights = prepare_weights(weights)
    cut = _array(cut, "cut")
    hidden = weights["postnorm"].size
    if cut.shape not in {(hidden,), (1, 1, hidden)}:
        raise ValueError("last-token cut must have shape [H] or [1,1,H]")
    ids = list(candidate_ids)
    if (len(ids) != weights["selected_lm_head"].shape[0]
            or any(isinstance(token, bool) or not isinstance(token, Integral) or token < 0 for token in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("candidate_ids must be unique nonnegative IDs aligned with selected head rows")
    doses = tuple(doses)
    if not doses or any(isinstance(dose, bool) or not isinstance(dose, Real) or not np.isfinite(dose) for dose in doses):
        raise ValueError("declare at least one finite numerical MLP dose")
    keys = [format(float(dose), ".17g") if dose != 0 else "0" for dose in doses]
    if len(set(keys)) != len(keys):
        raise ValueError("MLP doses must be distinct")
    rms, normalized = _rms_norm(cut, weights["postnorm"], float(eps))
    with np.errstate(over="ignore", invalid="ignore"):
        gate = normalized @ weights["gate_proj"].T
        up = normalized @ weights["up_proj"].T
        activated = silu(gate)
        gated = activated * up
        delta = gated @ weights["down_proj"].T
    arms = {}
    for key, dose in zip(keys, doses, strict=True):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = cut + float(dose) * delta
        final_rms, final = _rms_norm(residual, weights["finalnorm"], float(eps))
        with np.errstate(over="ignore", invalid="ignore"):
            logits = final @ weights["selected_lm_head"].T
        arms[key] = {"dose": float(dose), "residual": residual, "final_rms": final_rms,
                     "final_norm": final, "logits": logits}
    return _freeze({"candidate_ids": [int(token) for token in ids], "cut": cut, "rms_epsilon": float(eps),
        "arithmetic": "independent FP64 equations; native FP32 rounding is not emulated",
        "mlp": {"postnorm_rms": rms, "postnorm": normalized, "gate_pre": gate, "up": up,
                "silu_gate": activated, "gated": gated, "delta": delta}, "doses": arms})

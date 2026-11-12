"""NumPy response accounts for parameter edits; no gradient computation.

Equations use FP64 and output-by-input matrices. Native finite execution remains
a separate observable, including any rounding or reduction remainder.
"""
from __future__ import annotations


def _array(value):
    import numpy as np

    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("finite real numerical operands required")
    return array.astype(np.float64)


def linear_response(weight, edited_weight, inputs):
    """Exact real-arithmetic fixed-input identity Δm = ΔD h."""
    a, b, x = map(_array, (weight, edited_weight, inputs))
    if a.ndim != 2 or a.shape != b.shape or x.ndim < 1 or x.shape[-1] != a.shape[1]:
        raise ValueError("matching output-by-input matrices and input width required")
    return {"baseline": x @ a.T, "edited": x @ b.T, "direct": x @ (b - a).T}


def swiglu_response(gate, up, down, edited_gate, edited_up, edited_down, inputs):
    """Separate output-map, gate, payload and gate×payload changes at fixed x."""
    import numpy as np

    g, u, d, gp, up_, dp, x = map(_array,
        (gate, up, down, edited_gate, edited_up, edited_down, inputs))
    if (g.ndim != 2 or d.ndim != 2 or g.shape != u.shape or gp.shape != g.shape
            or up_.shape != u.shape or dp.shape != d.shape or d.shape[1] != g.shape[0]
            or x.ndim < 1 or x.shape[-1] != g.shape[1]):
        raise ValueError("incompatible SwiGLU parameter/input geometry")

    def silu(z):
        # exp(-abs(z)) avoids overflow without changing the mathematical law.
        e = np.exp(-np.abs(z))
        return z * np.where(z >= 0, 1 / (1 + e), e / (1 + e))

    s, sp = silu(x @ g.T), silu(x @ gp.T)
    v, vp = x @ u.T, x @ up_.T
    ds, dv = sp - s, vp - v
    return {"baseline": (s * v) @ d.T, "edited": (sp * vp) @ dp.T,
            "output_map": (s * v) @ (dp - d).T,
            "gate": (ds * v) @ dp.T, "payload": (s * dv) @ dp.T,
            "gate_payload_interaction": (ds * dv) @ dp.T}


def receiver_factorial(base_at_base, edited_at_base, base_at_edited, edited_at_edited):
    """Four measured/predicted corners; preserve signed vectors and interaction."""
    a, b, c, d = map(_array,
        (base_at_base, edited_at_base, base_at_edited, edited_at_edited))
    if any(value.shape != a.shape for value in (b, c, d)):
        raise ValueError("factorial corners must have identical shapes")
    return {"weight": b - a, "receiver": c - a, "interaction": d - b - c + a,
            "total": d - a}

"""Fit the applicability scalar of a fixed response, without optimizer steps."""
from __future__ import annotations


def fit_response_selector(hidden, direction, preservation, *, preserve_weight=1., ridge=1.):
    """Fit v so H v≈1 and K v≈0, retaining the supplied write direction.

    Returns left[out,1], right[1,in], diagnostics. Role labels belong to the
    caller's qualified account; this numerical fit does not discover semantics.
    """
    import numpy as np

    h, u, k = (np.asarray(x, dtype=np.float64) for x in (hidden, direction, preservation))
    if (h.ndim != 2 or k.ndim != 2 or u.ndim != 1 or not len(h) or not h.shape[1]
            or h.shape[1] != k.shape[1] or not len(u) or not np.linalg.norm(u)
            or not all(np.isfinite(x).all() for x in (h, u, k))
            or not np.isfinite([preserve_weight, ridge]).all() or preserve_weight < 0 or ridge <= 0):
        raise ValueError("finite aligned inputs, nonzero direction and positive ridge required")
    design = np.concatenate((h, np.sqrt(preserve_weight)*k))
    targets = np.concatenate((np.ones(len(h)), np.zeros(len(k))))
    dual = np.linalg.solve(design @ design.T + ridge*np.eye(len(design)), targets)
    v = dual @ design
    positive, negative = h @ v, k @ v
    return u[:, None].copy(), v[None, :], {
        "positive_coefficients": positive.tolist(), "preservation_coefficients": negative.tolist(),
        "positive_rmse": float(np.sqrt(np.mean((positive-1.)**2))),
        "preservation_rms": float(np.sqrt(np.mean(negative**2))) if len(k) else None,
        "positive_rows": len(h), "preservation_rows": len(k),
        "preserve_weight": float(preserve_weight), "ridge": float(ridge),
        "direction_policy": "supplied direction retained exactly; only linear applicability fitted",
        "status": "local fit; native effects unmeasured"}

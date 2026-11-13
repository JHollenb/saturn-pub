"""Preservation-aware low-rank affine corrections without gradients.

The caller supplies the site and desired local response. A fitted response is a
proposal: only an unchanged native consumer can establish its behavioral effect.
"""
from __future__ import annotations

from contextlib import contextmanager

from .wsp_discovery import affine_sites
from .wsp_weights import ParameterPatch, WeightSpaceError


def fit_low_rank_delta(hidden, desired_change, preservation, *, rank,
                       preserve_weight=1., ridge=1e-3):
    """Fit ΔW H≈R, ΔW K≈0; truncate the sample-space ridge solution by SVD.

    Returns left[out,rank], right[rank,in], and diagnostics. No hidden-width
    squared inverse or dense output-by-input SVD is formed. Ridge is absolute.
    """
    import numpy as np

    h, r, k = (np.asarray(x, dtype=np.float64) for x in (hidden, desired_change, preservation))
    if (h.ndim != 2 or r.ndim != 2 or k.ndim != 2 or not len(h)
            or len(h) != len(r) or h.shape[1] != k.shape[1]
            or type(rank) is not int or not 1 <= rank <= min(h.shape[1], r.shape[1])
            or not np.isfinite([preserve_weight, ridge]).all() or preserve_weight < 0 or ridge <= 0
            or not all(np.isfinite(x).all() for x in (h, r, k))):
        raise ValueError("finite aligned samples, valid rank, and positive ridge required")
    design = np.concatenate((h, np.sqrt(preserve_weight) * k))
    targets = np.concatenate((r, np.zeros((len(k), r.shape[1]))))
    dual = np.linalg.solve(design @ design.T + ridge * np.eye(len(design)), targets)
    # ΔW=dual.T@design. Factor the two thin sides before truncating.
    ql, rl = np.linalg.qr(dual.T, mode="reduced")
    qr, rr = np.linalg.qr(design.T, mode="reduced")
    u, s, vt = np.linalg.svd(rl @ rr.T, full_matrices=False)
    width = min(rank, len(s))
    left = (ql @ u[:, :width]) * s[:width]
    right = vt[:width] @ qr.T
    task_error = h @ right.T @ left.T - r
    protected_change = k @ right.T @ left.T
    return left, right, {"rank_bound": width, "singular_values": s.tolist(),
        "task_response_rmse": float(np.sqrt(np.mean(task_error ** 2))),
        "preservation_response_rmse": float(np.sqrt(np.mean(protected_change ** 2))) if len(k) else None,
        "task_rows": len(h), "preservation_rows": len(k),
        "preserve_weight": float(preserve_weight), "ridge": float(ridge),
        "meaning": "local response fit; downstream capability unmeasured"}


def _factors(model, site, left, right, dose):
    import math

    import torch

    module = affine_sites(model).get(site)
    if module is None or isinstance(dose, bool) or not math.isfinite(dose):
        raise WeightSpaceError("valid affine site and finite dose required")
    a = torch.as_tensor(left, device=module.weight.device, dtype=torch.float64).detach().clone()
    b = torch.as_tensor(right, device=module.weight.device, dtype=torch.float64).detach().clone()
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or not a.shape[1]
            or a.shape[0] != module.out_features or b.shape[1] != module.in_features
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise WeightSpaceError("finite compatible low-rank factors required")
    return module, a, b


@contextmanager
def virtual_low_rank_edit(model, site, left, right, *, dose=1.):
    """Apply the fitted response to the recipient's own current affine inputs."""
    module, a, b = _factors(model, site, left, right, dose)
    a, b = a.to(module.weight.dtype), b.to(module.weight.dtype)

    def hook(layer, args, output):
        return output + dose * ((args[0] @ b.T) @ a.T)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def synthesize_low_rank_patch(model, site, left, right, *, dose=1., model_identity):
    """Compile factors into ordinary weight bytes; bias stays unchanged.

    Low rank describes the correction, not sparse byte support. FP32 fused and
    factored execution may round differently; measure parity through the consumer.
    """
    module, a, b = _factors(model, site, left, right, dose)
    replacement = (module.weight.detach().double() + dose * (a @ b)).to(module.weight.dtype)
    return ParameterPatch(model, {(site + "." if site else "") + "weight": replacement},
                          model_identity=model_identity)

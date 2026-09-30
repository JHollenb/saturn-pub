"""Fit a small state-dependent residual writer from explicitly paired state cuts.

This is supervised trace fitting. Consumer behavior must be evaluated separately.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def fit_low_rank_writer(recipient, donor, *, address, rank=8, ridge=1e-3):
    recipient.verify()
    donor.verify()
    source = recipient.payload[address]
    target = donor.payload[address]
    if source.shape != target.shape or source.ndim < 2:
        raise ValueError("paired states require matching geometry")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    width = source.shape[-1]
    x = source.detach().float().cpu().numpy().reshape(-1, width)
    y = (target.detach().float().cpu() - source.detach().float().cpu()).numpy().reshape(-1, width)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired states must have finite values and at least two rows")
    mean = x.mean(axis=0)
    centered = x - mean
    rank = min(rank, len(x) - 1, width)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank].T.astype(np.float32)
    z = centered @ basis
    weights = np.linalg.solve(z.T @ z + np.eye(rank, dtype=np.float32) * ridge, z.T @ y)
    bias = y.mean(axis=0)
    predicted = z @ weights + bias
    mse = float(np.mean((predicted - y) ** 2))
    energy = float(np.mean(y**2))
    arrays = dict(mean=mean, basis=basis, weights=weights, bias=bias)
    metadata = {
        "schema": "saturn-pub-low-rank-writer-v1",
        "recipe": "centered recipient PCA followed by ridge residual regression",
        "address": address,
        "rank": rank,
        "ridge": ridge,
        "recipient_cut": recipient.fingerprint,
        "donor_cut": donor.fingerprint,
        "recipient_model": recipient.model_identity,
        "donor_model": donor.model_identity,
        "recipient_boundary": recipient.boundary,
        "donor_boundary": donor.boundary,
        "rows": len(x),
        "dimension": width,
        "fit_mse": mse,
        "target_delta_rms": energy**0.5,
        "relative_fit_mse": mse / energy if energy else None,
        "claim_boundary": "One paired trace; reconstruction does not establish consumer capability.",
    }
    return arrays, metadata


def save_low_rank_writer(path, arrays, metadata):
    """Seal FP16 serving arrays and separate construction provenance; no donor weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = {
        name: np.asarray(arrays[name], dtype=np.float16)
        for name in ("mean", "basis", "weights", "bias")
    }
    if any(not np.isfinite(value).all() for value in packed.values()):
        raise ValueError("writer overflows serving precision")
    np.savez_compressed(path, **packed, gain=np.array([0], dtype=np.float16))
    result = {
        **metadata,
        "package_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "package_bytes": path.stat().st_size,
        "serving_scalar_count": sum(value.size for value in packed.values()) + 1,
        "serving_dtype": "float16",
    }
    path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    return result

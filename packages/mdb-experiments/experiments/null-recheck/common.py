"""Shared helpers for the null re-check drivers (saturn experiments/2026-11-16-null-recheck/PREREG.md).

Verdict rules are mdb's (``mdb.analysis.donors.informative`` / ``pooled``). The floor is the one the
Qwen back-test used: 2 bf16 spacings at the larger |logit| of the consumer pair, plus the measured
spread of the null lane (identical input, different batch composition) across the item's batches.
"""
import json
import math
import time

import torch

from mdb.analysis.donors import EFFECT, NO_EFFECT, informative, pooled

T0 = time.monotonic()


def log(msg):
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", flush=True)


def spacing(x):
    x = abs(float(x))
    return 2.0 ** (math.floor(math.log2(x)) - 7) if x else 2.0 ** -133


def floor_of(la, lb, spread=0.0):
    return 2 * spacing(max(abs(float(la)), abs(float(lb)))) + float(spread)


def verdict_row(dm, floor, inf):
    return {"verdict": EFFECT if abs(dm) > floor else NO_EFFECT, "informative": inf}


def pool(rows):
    """rows: dicts with dM, floor, informative (+ anything). mdb pooled verdict plus magnitudes."""
    p = pooled([verdict_row(r["dM"], r["floor"], r["informative"]) for r in rows])
    units = [abs(r["dM"]) / r["floor"] for r in rows]
    p.pop("rule", None)
    p["max_floor_units"] = max(units) if units else None
    p["n_ge_4_floors"] = sum(u > 4 for u in units)
    p["max_abs_dM"] = max(abs(r["dM"]) for r in rows) if rows else None
    return p


def rotate_delta(k, delta, inv_freq):
    """RoPE rotation of a cached (already rotated) key by ``delta`` positions (HF rotate_half
    convention): R(p + delta) = R(delta) R(p). fp32 math, returned in k's dtype (one rounding)."""
    if delta == 0:
        return k.clone()
    ang = float(delta) * inv_freq.double()
    emb = torch.cat([ang, ang]).to(k.device)
    cos, sin = emb.cos(), emb.sin()
    x = k.double()
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x * cos + rot * sin).to(k.dtype)


def dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, default=float)


__all__ = ["EFFECT", "NO_EFFECT", "dump", "floor_of", "informative", "log", "pool", "rotate_delta",
           "spacing", "verdict_row"]

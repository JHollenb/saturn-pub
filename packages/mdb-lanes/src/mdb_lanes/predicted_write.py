"""Predicted write of attention read cuts, the routing / payload split, and set selectors.

Experiment ``packages/mdb-lanes/experiments/predicted_write_20261113`` (FLUX.2 Klein and Qwen2.5).

Identities (one head, receiver row t, source set S with 0/1 indicator ``m``, weights ``p`` of the
native softmax, per-head values ``v``, head output ``a = p @ v``, ``p_S = sum_{j in S} p_j``)::

    zero cut (contribution removed, no renormalisation):  a' - a = -(p*m) @ v
    mask cut (renormalising boolean mask):                a' - a = (p_S a - (p*m) @ v) / (1 - p_S)

The block's residual write changes by ``g * (O_h (a' - a))`` (outer gate ``g`` per channel, the
head's slice ``O_h`` of the attention output map). Exact in real arithmetic.

Routing / payload split of a change between two runs at matched geometry (keys aligned by a
bijection)::

    P_d V_d - P_n V_n = dP V_n + P_n dV + dP dV,   dP = P_d - P_n,  dV = V_d - V_n

Because every row of ``P_n`` and ``P_d`` sums to one, ``dP 1 = 0`` and ``dP V_n = dP (V_n - 1 vbar^T)``
for any common value ``vbar``: routing changes consume only value differences.

Every function in the "identities" section takes numpy arrays or torch tensors (it uses only ``@``,
``*``, ``/``, ``.sum(-1)``, ``.sum(-2)``, ``[..., None]`` and indexing), so the worker runs the same
code on CUDA tensors that the CPU tests check against a direct numpy softmax recomputation.
Selectors and the deletion alignment are numpy / plain Python. Importing this module loads neither
numpy nor torch.
"""
from __future__ import annotations

import math

# ===================================================================================== identities


def softmax_rows(scores):
    """numpy softmax over the last axis (reference for the tests)."""
    import numpy as np
    s = np.asarray(scores, np.float64)
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def zero_cut_delta(p, v, m):
    """``a' - a`` of a no-renormalisation cut. ``p`` [..., R, Sk], ``v`` [..., Sk, D], ``m`` [Sk] (0/1)."""
    return -((p * m) @ v)


def mask_cut_delta(p, v, m, a=None):
    """``a' - a`` of a renormalising cut: ``(p_S a - (p*m) @ v) / (1 - p_S)``.

    ``a`` = the native head output ``p @ v`` (computed when omitted). A receiver with ``p_S = 1``
    has no source left; the port refuses such cuts and so does this function's caller.
    """
    pm = p * m
    if a is None:
        a = p @ v
    ps = pm.sum(-1)[..., None]
    return (ps * a - pm @ v) / (1 - ps)


def head_maps(out_weight, heads):
    """Per-head output maps ``[H, D, C]`` from an output projection weight ``[C, H*D]``.

    ``delta @ head_maps(...)`` with ``delta`` [H, R, D] gives each head's write [H, R, C].
    """
    c, hd = out_weight.shape
    if hd % heads:
        raise ValueError("output map width is not a multiple of the head count")
    w = out_weight.reshape(c, heads, hd // heads)
    return w.permute(1, 2, 0) if hasattr(w, "permute") else w.transpose(1, 2, 0)


def head_writes(delta, maps, gate):
    """Gated per-head writes ``[H, R, C]`` = ``(delta @ O_h^T) * g``."""
    return (delta @ maps) * gate


def write_norms(delta, maps, gate):
    """Norm of each head's gated write at each receiver, ``[H, R]``."""
    w = head_writes(delta, maps, gate)
    return ((w * w).sum(-1)) ** 0.5


def block_write(x, maps, gate):
    """Gated write of all heads summed, ``[R, C]`` = ``sum_h (x_h @ O_h^T) * g``."""
    return ((x @ maps).sum(0)) * gate


def inner(x, y):
    """Frobenius inner product as a Python float."""
    return float((x * y).sum())


# --------------------------------------------------------------------------------- the 3-way split

def split3(pn, vn, pd, vd):
    """Exact routing / payload / interaction split of ``pd @ vd - pn @ vn``.

    Returns ``{"routing": dP vn, "payload": pn dV, "interaction": dP dV, "total": pd vd - pn vn}``.
    """
    dp, dv = pd - pn, vd - vn
    return {"routing": dp @ vn, "payload": pn @ dv, "interaction": dp @ dv, "total": pd @ vd - pn @ vn}


def payload_parts(pn, dv, groups):
    """Payload ``pn @ dV`` restricted to source groups: ``{name: (pn * m_g) @ dV}``."""
    return {name: (pn * m) @ dv for name, m in groups.items()}


def common_payload(pn, dv, text_mask):
    """Common vs source-specific text payload.

    ``dvbar`` = unweighted mean of ``dV`` over the text sources (per head). ``common`` =
    ``(sum_text p) * dvbar``; ``specific`` = ``sum_text p_j (dV_j - dvbar)``. Their sum is the text
    payload ``(pn * text_mask) @ dV`` exactly.
    """
    n = text_mask.sum()
    dvbar = (dv * text_mask[:, None]).sum(-2) / n                    # [..., D]
    pt = pn * text_mask
    common = pt.sum(-1)[..., None] * dvbar[..., None, :]
    specific = pt @ (dv - dvbar[..., None, :])
    return common, specific


def routing_centered(pn, pd, vn, vbar):
    """``dP (vn - 1 vbar^T)``: equal to ``dP vn`` when both maps are row-stochastic (the invariant)."""
    return (pd - pn) @ (vn - vbar[..., None, :])


# ===================================================================================== alignment

def deletion_alignment(record_original, record_deleted, word, text_rows):
    """Bijection ``pi`` of text rows (present prompt -> deleted prompt) at matched geometry.

    The 2026-11-11 read-graph alignment (:func:`mdb_lanes.flux_read_panels.hybrid_own_plan`): rows
    before the word keep their index, real rows after it and padding rows shift by the deleted
    token count. The word's own rows have no partner in the deleted prompt; they are mapped, in
    order, to the deleted rows nothing else maps to (the trailing padding rows the shift frees).
    Returns ``{"pi", "word", "real", "pad", "shift"}`` (row lists in present indexing) or ``None``
    when the encodings do not align.
    """
    from .flux_read_panels import hybrid_own_plan
    plan = hybrid_own_plan(record_original, record_deleted, word)
    if plan is None:
        return None
    n_o, shift = record_original["n_real"], plan["shift"]
    own = plan["own_rows"]
    pi = [None] * text_rows
    for j in range(text_rows):
        if j < min(own):
            pi[j] = j
        elif j > max(own):
            pi[j] = j - shift
    free = sorted(set(range(text_rows)) - {x for x in pi if x is not None})
    rest = [j for j in range(text_rows) if pi[j] is None]
    if len(free) != len(rest):
        return None
    for j, d in zip(rest, free):
        pi[j] = d
    if sorted(pi) != list(range(text_rows)):
        raise ValueError("alignment is not a bijection")
    word_rows = sorted(rest)
    return {"pi": pi, "word": word_rows, "real": [j for j in range(n_o) if j not in set(word_rows)],
            "pad": list(range(n_o, text_rows)), "shift": shift, "own_rows": own,
            "word_partner_rows": [pi[j] for j in word_rows]}


# ===================================================================================== selectors

def top_units(score, k):
    """Indices of the ``k`` largest scores (ties broken by the lower index), sorted."""
    import numpy as np
    s = np.asarray(score, np.float64)
    order = np.lexsort((np.arange(len(s)), -s))
    return sorted(int(i) for i in order[:k])


def _band(target, rel_tol):
    return rel_tol * abs(target)


def matched_min(constraint, objective, k, target, rel_tol=0.10, *, exclude=(), max_rounds=5000):
    """``k`` units whose total ``constraint`` lies within ``rel_tol * target`` of ``target`` while the
    total ``objective`` is as small as this search finds.

    Lagrangian start (the ``k`` smallest ``objective - lam * constraint``, ``lam`` bisected so the
    constraint total crosses ``target``), then single swaps: first into the band (smallest objective
    increase per unit of constraint moved), then best-improvement swaps that stay in the band until
    none lowers the objective (a 1-swap local optimum, not a proven global one). Deterministic.
    Raises ``ValueError`` when no set in the band is reachable.
    """
    import numpy as np
    c = np.asarray(constraint, np.float64)
    o = np.asarray(objective, np.float64)
    n = len(c)
    if o.shape != (n,) or not (np.isfinite(c).all() and np.isfinite(o).all()):
        raise ValueError("constraint and objective need one finite value per unit")
    allowed = np.ones(n, bool)
    allowed[list(exclude)] = False
    idx = np.flatnonzero(allowed)
    if len(idx) < k:
        raise ValueError("fewer allowed units than k")
    band = _band(target, rel_tol)

    def pick(lam):
        key = o[idx] - lam * c[idx]
        order = np.lexsort((idx, key))
        return np.sort(idx[order[:k]])

    lo, hi = -1.0, 1.0
    scale = (np.abs(o).max() + 1e-300) / (np.abs(c).max() + 1e-300)
    lo, hi = -scale, scale
    for _ in range(200):
        if c[pick(lo)].sum() <= target:
            break
        lo *= 2
    for _ in range(200):
        if c[pick(hi)].sum() >= target:
            break
        hi *= 2
    best = None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        s = pick(mid)
        tot = c[s].sum()
        if abs(tot - target) <= band and (best is None or o[s].sum() < o[best].sum()):
            best = s
        if tot < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * max(1.0, abs(hi)):
            break
    chosen = best if best is not None else pick(hi)
    inside = np.zeros(n, bool)
    inside[chosen] = True
    rounds = swaps_in = swaps_opt = 0
    # repair into the band
    while abs(c[inside].sum() - target) > band and rounds < max_rounds:
        rounds += 1
        tot = c[inside].sum()
        ins, outs = np.flatnonzero(inside), np.flatnonzero(allowed & ~inside)
        dc = c[outs][None, :] - c[ins][:, None]            # change of constraint total for swap (i out, j in)
        do = o[outs][None, :] - o[ins][:, None]
        need = target - tot
        useful = np.sign(dc) == np.sign(need)
        new_gap = np.abs(need - dc)
        if not useful.any():
            break
        cost = np.where(useful, do / np.maximum(np.abs(dc), 1e-300), np.inf)
        cost = np.where(new_gap < abs(need), cost, np.inf)
        if not np.isfinite(cost).any():
            break
        i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
        inside[ins[i]], inside[outs[j]] = False, True
        swaps_in += 1
    if abs(c[inside].sum() - target) > band:
        raise ValueError(f"no set within {rel_tol:.0%} of the target {target:.6g} was reached "
                         f"(closest total {c[inside].sum():.6g})")
    # best-improvement swaps inside the band
    while rounds < max_rounds:
        rounds += 1
        tot = c[inside].sum()
        ins, outs = np.flatnonzero(inside), np.flatnonzero(allowed & ~inside)
        dc = c[outs][None, :] - c[ins][:, None]
        do = o[outs][None, :] - o[ins][:, None]
        ok = np.abs(tot + dc - target) <= band
        gain = np.where(ok, do, np.inf)
        i, j = np.unravel_index(int(np.argmin(gain)), gain.shape)
        if not gain[i, j] < -1e-15 * max(1.0, abs(o[inside].sum())):
            break
        inside[ins[i]], inside[outs[j]] = False, True
        swaps_opt += 1
    units = sorted(int(u) for u in np.flatnonzero(inside))
    total_c, total_o = float(c[units].sum()), float(o[units].sum())
    return units, {"target": float(target), "constraint_total": total_c,
                   "relative_deviation": (total_c - target) / target if target else None,
                   "objective_total": float(total_o), "rel_tol": rel_tol, "repair_swaps": swaps_in,
                   "improvement_swaps": swaps_opt, "rounds": rounds,
                   "local_optimum": rounds < max_rounds}


def overlap(a, b):
    return len(set(a) & set(b))


# ===================================================================================== statistics

def ranks(x):
    """Average ranks (1-based), ties averaged."""
    import numpy as np
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(x, y):
    import numpy as np
    rx, ry = ranks(x), ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else None


def shares(raw, total_key="total"):
    """Signed projection shares and norm ratios from accumulated raw sums.

    ``raw[name] = {"ip": <W(X), W(total)>, "nn": |W(X)|^2}`` summed over cells. Projection share =
    ip / |W(total)|^2 (the shares of an exact decomposition sum to one); norm ratio = |W(X)| / |W(total)|.
    """
    tt = raw[total_key]["nn"]
    out = {}
    for name, r in raw.items():
        out[name] = {"projection_share": r["ip"] / tt if tt > 0 else None,
                     "norm_ratio": math.sqrt(r["nn"] / tt) if tt > 0 else None}
    return out

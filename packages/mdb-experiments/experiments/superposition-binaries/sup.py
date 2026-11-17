"""Controlled superposition on torchx86's gated-MLP ``f``: mixings, un-mixing methods, scores.

Pure torch (GPU or CPU), no sklearn. Every method maps rows ``Y [N, d]`` to unit scores ``Z [N, U]``;
scores are compared with ground-truth indicators ``L [N, C]`` (bool) by:

  best_cut   per indicator c: the best unit u, sign s and cut t over every realisable cut of s*Z[:, u]
             (an oracle threshold, the same for every method): IoU, precision (= purity), recall.
  native     for binary firing rules (SAE act > 0, k-means membership): per indicator the best unit's
             IoU/precision, per-unit modal purity, and the group IoU (union of units whose modal
             indicator is c with purity >= 0.9).
"""
from __future__ import annotations

import math

import torch

# --------------------------------------------------------------------------- mixings


def rotation(D: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


def projection(d: int, D: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, D, generator=g) / math.sqrt(d)


def mix(X: torch.Tensor, M: torch.Tensor | None, chunk: int = 1 << 13) -> torch.Tensor:
    """Rows X [N, D] -> X @ M^T [N, d] (M None: X as float)."""
    if M is None:
        return X.float()
    M = M.to(X.device)
    return torch.cat([X[s:s + chunk].float() @ M.T for s in range(0, len(X), chunk)])


def add_noise(Y: torch.Tensor, alpha: float, seed: int) -> torch.Tensor:
    """Gaussian noise, per-coordinate sd = alpha * RMS of Y's coordinates."""
    g = torch.Generator(device=Y.device).manual_seed(seed)
    sd = alpha * float(Y.pow(2).mean().sqrt())
    return Y + sd * torch.randn(Y.shape, generator=g, device=Y.device)


# --------------------------------------------------------------------------- methods


def pca(Y: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-m principal scores [N, m], components [m, d], variances [m]."""
    mu = Y.mean(0)
    Yc = Y - mu
    m = min(m, Y.shape[1], max(len(Y) - 1, 1))
    if len(Y) < Y.shape[1]:
        _, s, vh = torch.linalg.svd(Yc, full_matrices=False)
        V, lam = vh[:m], s[:m] ** 2 / max(len(Y) - 1, 1)
    else:
        lam, V = torch.linalg.eigh(Yc.T @ Yc / max(len(Y) - 1, 1))
        lam, V = lam.flip(0)[:m], V.flip(1)[:, :m].T
    return Yc @ V.T, V, lam


def ica(Y: torch.Tensor, m: int, seed: int, iters: int = 200, tol: float = 1e-5) -> torch.Tensor:
    """Symmetric FastICA (logcosh) on the top-m whitened principal scores; sources [N, m']."""
    Zp, _, lam = pca(Y, m)
    keep = lam > lam[0] * 1e-6 if len(lam) else lam > 0
    Zw = Zp[:, keep] / lam[keep].sqrt()
    n, m = Zw.shape
    if m == 0:
        return Zw
    g = torch.Generator().manual_seed(seed)
    W = _sym(torch.randn(m, m, generator=g).to(Y.device))
    for _ in range(iters):
        wx = Zw @ W.T                                        # [n, m]
        gwx = torch.tanh(wx)
        W1 = _sym((gwx.T @ Zw) / n - (1 - gwx ** 2).mean(0)[:, None] * W)
        done = float((torch.abs(torch.abs((W1 * W).sum(1)) - 1)).max()) < tol
        W = W1
        if done:
            break
    return Zw @ W.T


def _sym(W):
    s, u = torch.linalg.eigh(W @ W.T)
    return u @ torch.diag(s.clamp(min=1e-12).rsqrt()) @ u.T @ W


def kmeans(Y: torch.Tensor, k: int, seed: int, iters: int = 50, restarts: int = 3) -> torch.Tensor:
    """Lloyd k-means (k-means++ init), best of ``restarts`` by inertia; one-hot membership [N, k]."""
    g = torch.Generator().manual_seed(seed)
    best, best_in = None, float("inf")
    for _ in range(restarts):
        C = Y[torch.randint(len(Y), (1,), generator=g)]
        for _ in range(1, k):                                 # k-means++
            d2 = torch.cdist(Y, C).min(1).values ** 2
            p = (d2 / d2.sum()).cpu() if float(d2.sum()) > 0 else torch.full((len(Y),), 1 / len(Y))
            C = torch.cat([C, Y[torch.multinomial(p, 1, generator=g)]])
        for _ in range(iters):
            a = torch.cdist(Y, C).argmin(1)
            oh = torch.nn.functional.one_hot(a, k).float()
            cnt = oh.sum(0)
            C = torch.where(cnt[:, None] > 0, (oh.T @ Y) / cnt.clamp(min=1)[:, None], C)
        a = torch.cdist(Y, C).argmin(1)
        inertia = float(((Y - C[a]) ** 2).sum())
        if inertia < best_in:
            best, best_in = a, inertia
    return torch.nn.functional.one_hot(best, k).float()


# --------------------------------------------------------------------------- scores


def best_cut(Z: torch.Tensor, L: torch.Tensor, chunk: int | None = None, signed: bool = True,
             budget: int = 1 << 25) -> dict:
    """Per indicator (column of L [N, C]): max over units, signs and realisable cuts of IoU.

    A cut sits between two adjacent distinct values of s*Z[:, u] sorted descending (ties never split).
    Returns tensors [C]: iou, precision, recall, unit, sign."""
    N, C = L.shape
    chunk = chunk or max(1, min(Z.shape[1], budget // max(N * C, 1)))
    Lf = L.float()
    nc = Lf.sum(0)                                            # [C]
    best = torch.full((C,), -1.0, device=Z.device)
    prec = torch.zeros(C, device=Z.device)
    rec = torch.zeros(C, device=Z.device)
    unit = torch.full((C,), -1, dtype=torch.long, device=Z.device)
    sgn = torch.ones(C, device=Z.device)
    pos = torch.arange(1, N + 1, device=Z.device, dtype=torch.float32)[:, None, None]
    for s in ((1.0, -1.0) if signed else (1.0,)):
        for u0 in range(0, Z.shape[1], chunk):
            z = s * Z[:, u0:u0 + chunk]
            zs, order = z.sort(0, descending=True)                     # [N, u]
            valid = torch.ones_like(zs, dtype=torch.bool)
            valid[:-1] = zs[:-1] > zs[1:]
            tp = Lf[order].cumsum(0)                                   # [N, u, C]
            iou = tp / (nc + pos - tp)
            iou = torch.where(valid[..., None], iou, -1.0)
            v, i = iou.max(0)                                          # [u, C]
            vu, j = v.max(0)                                           # [C]
            better = vu > best
            if better.any():
                cut = i[j, torch.arange(C, device=Z.device)]          # [C]
                t = tp[cut, j, torch.arange(C, device=Z.device)]
                best = torch.where(better, vu, best)
                prec = torch.where(better, t / (cut + 1).float(), prec)
                rec = torch.where(better, t / nc.clamp(min=1), rec)
                unit = torch.where(better, u0 + j, unit)
                sgn = torch.where(better, torch.full_like(sgn, s), sgn)
    return {"iou": best.clamp(min=0), "precision": prec, "recall": rec, "unit": unit, "sign": sgn}


def native(B: torch.Tensor, L: torch.Tensor, pure: float = 0.9) -> dict:
    """Binary firing B [N, U] vs indicators L [N, C] (rows may carry one indicator each).

    Per indicator: the best single unit's IoU and precision; group IoU of the union of units whose
    modal indicator is c with purity >= ``pure``. Per method: firing-weighted modal purity, live units."""
    Bf, Lf = B.float(), L.float()
    inter = Bf.T @ Lf                                          # [U, C]
    fires = Bf.sum(0)                                          # [U]
    nc = Lf.sum(0)
    iou = inter / (fires[:, None] + nc[None] - inter).clamp(min=1)
    v, u = iou.max(0)
    precision = inter[u, torch.arange(L.shape[1])] / fires[u].clamp(min=1)
    modal_n, modal = inter.max(1)
    pur = modal_n / fires.clamp(min=1)
    live = fires > 0
    group = []
    for c in range(L.shape[1]):
        units = live & (modal == c) & (pur >= pure)
        if units.any():
            fired = B[:, units].any(1)
            i = float((fired & L[:, c]).sum())
            group.append(i / float((fired | L[:, c]).sum()))
        else:
            group.append(0.0)
    return {"iou": v, "precision": precision, "unit": u, "group_iou": torch.tensor(group),
            "purity_fire_weighted": float((pur * fires)[live].sum() / fires[live].sum()) if live.any() else None,
            "live": int(live.sum()), "modal": modal, "unit_purity": pur, "fires": fires}


def value_split(B: torch.Tensor, cls: torch.Tensor, value: torch.Tensor, nat: dict, pure: float = 0.9) -> dict:
    """For units whose modal class has purity >= ``pure``: the share of their in-class firing rows that
    carry the unit's modal output value (value purity), firing-weighted, against the baseline of each
    class's own modal-value share weighted the same way. Also pure units per class."""
    vp, base, w, per_class = 0.0, 0.0, 0.0, {}
    cache = {}
    for u in torch.nonzero((nat["fires"] > 0) & (nat["unit_purity"] >= pure)).flatten().tolist():
        c = int(nat["modal"][u])
        rows = B[:, u].bool() & (cls == c)
        n = int(rows.sum())
        if n == 0:
            continue
        _, cnt = value[rows].unique(return_counts=True)
        if c not in cache:
            _, cc = value[cls == c].unique(return_counts=True)
            cache[c] = float(cc.max()) / float(cc.sum())
        vp += float(cnt.max())
        base += cache[c] * n
        w += n
        per_class[c] = per_class.get(c, 0) + 1
    return {"pure_units": sum(per_class.values()), "value_purity": vp / w if w else None,
            "baseline_class_modal_value_share": base / w if w else None,
            "pure_units_per_class": {str(k): v for k, v in sorted(per_class.items())}}


def summarize(r: dict, names: list) -> dict:
    """Tensor dict -> JSON-able per-indicator rows and macro means."""
    out = {"macro_iou": float(r["iou"].mean()), "macro_precision": float(r["precision"].mean())}
    if "group_iou" in r:
        out["macro_group_iou"] = float(r["group_iou"].mean())
    for key in ("purity_fire_weighted", "live"):
        if key in r:
            out[key] = r[key]
    out["per"] = {n: {k: round(float(r[k][i]), 4) for k in ("iou", "precision", "recall", "group_iou") if k in r}
                  for i, n in enumerate(names)}
    return out

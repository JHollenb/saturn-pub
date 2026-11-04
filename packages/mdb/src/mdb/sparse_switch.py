"""Sparse switch view: the per-position MLP gate list that a model's output depends on.

A gated MLP writes ``y = W_down (g * u)`` with ``g = act(W_gate h)`` and ``u = W_up h``.
Against a reference gate ``g_ref`` (for example the token-mean gate over fit
contexts) the *switch* at one position is ``s = W_down((g - g_ref) * u)``. Given a
Fisher metric ``F`` at the MLP write, each neuron ``j`` contributes
``(g_j - g_ref_j) u_j w_j`` with ``w_j = W_down[:, j]``; the view ranks neurons by the
diagonal Fisher score

    score_j = |(g_j - g_ref_j) * u_j| * sqrt(w_j^T F w_j)

and keeps the top ``m`` per position (the *list*). A predictor proposes its own
scores (for example from a reconstructed residual pushed through the model's own
gate); the view scores the proposed list against the oracle list:

* ``recall``       |P & O| / m
* ``mass_recall``  sum_{j in P} score_j / sum_{j in O} score_j  (oracle scores)
* ``flip_in``      share of oracle top-m neurons whose score error exceeds their
                   margin above the (m+1)-th oracle score
* ``flip_out``     share of oracle ranks m+1..2m whose score error exceeds their
                   margin below the m-th oracle score
* ``margin_ratio`` median score error over the oracle top-2m divided by the score
                   spread sc_(m/2) - sc_(2m) around the threshold

and the Fisher energy a code keeps, ``1 - sum r^T F r / sum s^T F s`` with
``r = s - s_hat``. Everything is closed form; nothing is fit here. Tensors are
torch tensors with the neuron axis last; torch is imported only by callers.

Codes as replays (``code_products``): a code keeps the exact gate for a set of
neurons (a per-position top-m list, a static ``pool`` or a proxy-selected set)
and uses a ``base`` for the rest: the token-mean gate, the global-mean gate
(the write outside the set is then ``B_0 n`` with ``B_0 = W_down diag(g_glob) W_up``)
or the token-table hybrid ``y0_tok(n_tok) + B_0 (n - n_tok)``. The MLP write is
``W_down(products)``. ``flop_fraction`` gives a code's derived multiply-accumulate
count as a fraction of a full column (derived from the shapes, not timed).

Proxies (``reading_metric``, ``metric_proxy``, ``proxy_gate_inputs``): a rank-k map
of the gate input predicts the gate pre-activation and up projection cheaply, and the
code takes its exact set from the predicted scores. ``metric_proxy`` is the
reduced-rank regression that minimises ``E || M^{1/2} (I - B) n ||^2`` for a reading
metric ``M`` and second moment ``S`` of ``n``; with ``M = I`` it is the PCA of ``S``.
Serving (``ServingLedger``, ``column_equivalents``): a mixed-memory stream in which
each position appends rows from either the proxy column (cost f) or a native column
(cost 1); nothing is recomputed. The ledger counts the native rate, agreement and KL
by position band.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA = "mdb-sparse-switch-view-v1"


def fisher_neuron_weights(fisher, w_down):
    """sqrt(diag(W_down^T F W_down)) per neuron. fisher [d, d], w_down [d, I] -> [I]."""
    wd = w_down.to(fisher.dtype)
    return ((fisher @ wd) * wd).sum(0).clamp_min(0).sqrt()


def switch_scores(g, g_ref, u, weights):
    """Diagonal Fisher score |(g - g_ref) * u| * weights, in fp32. Shapes broadcast over [..., I]."""
    return ((g.float() - g_ref.float()) * u.float()).abs() * weights.float()


def top_m_mask(scores, m):
    """Boolean mask of the m largest scores along the last axis."""
    idx = scores.topk(m, dim=-1).indices
    return scores.new_zeros(scores.shape, dtype=bool).scatter_(-1, idx, True)


def list_metrics(scores, predicted_scores, m):
    """Per-position metrics of the predicted top-m list against the oracle top-m list.

    scores, predicted_scores: [N, I]. Returns a dict of [N] tensors (see module doc).
    """
    n_items = scores.shape[-1]
    k2 = min(2 * m, n_items)
    srt, order = scores.sort(-1, descending=True)
    oracle = scores.new_zeros(scores.shape, dtype=bool).scatter_(-1, order[:, :m], True)
    pred = top_m_mask(predicted_scores, m)
    recall = (oracle & pred).sum(-1).float() / m
    mass = (scores * pred).sum(-1) / (scores * oracle).sum(-1).clamp_min(1e-30)
    err = (predicted_scores.float() - scores.float()).abs()
    err_sorted = err.gather(-1, order[:, :k2])
    tau_in = srt[:, m] if m < n_items else srt[:, -1]  # (m+1)-th oracle score
    tau_out = srt[:, m - 1]  # m-th oracle score
    flip_in = (err_sorted[:, :m] > (srt[:, :m] - tau_in[:, None])).float().mean(-1)
    if k2 > m:
        flip_out = (err_sorted[:, m:k2] > (tau_out[:, None] - srt[:, m:k2])).float().mean(-1)
    else:
        flip_out = recall.new_zeros(recall.shape)
    spread = (srt[:, max(m // 2 - 1, 0)] - srt[:, k2 - 1]).clamp_min(1e-30)
    ratio = err_sorted.median(-1).values / spread
    return {"recall": recall, "mass_recall": mass, "flip_in": flip_in, "flip_out": flip_out,
            "margin_ratio": ratio}


def fisher_energy(residual, fisher):
    """sum over rows of r^T F r (float)."""
    r = residual.to(fisher.dtype)
    return float(((r @ fisher) * r).sum())


@dataclass
class SparseSwitchView:
    """Accumulates list metrics and Fisher energies across batches, keyed by (layer, code, m)."""

    sums: dict = field(default_factory=dict)
    medians: dict = field(default_factory=dict)

    def _add(self, key, value, count=1):
        total, n = self.sums.get(key, (0.0, 0))
        self.sums[key] = (total + float(value), n + count)

    def add_switch_energy(self, layer, switch_energy):
        """Fisher energy sum s^T F s of the switch itself for one batch (the denominator)."""
        self._add((layer, "_switch", "energy"), switch_energy)

    def add_energy(self, layer, code, residual_energy):
        """Fisher energy sum r^T F r of one code's residual r = s - s_hat for one batch."""
        self._add((layer, code, "res"), residual_energy)

    def add_lists(self, layer, code, m, scores, predicted_scores):
        met = list_metrics(scores, predicted_scores, m)
        n = int(scores.shape[0])
        for name, v in met.items():
            self._add((layer, code, m, name), float(v.sum()), n)
        ratios = self.medians.setdefault((layer, code, m), [])
        ratios.append(met["margin_ratio"].float().cpu())
        return met

    def energy_kept(self, layer, code):
        res, _ = self.sums[(layer, code, "res")]
        tot, _ = self.sums[(layer, "_switch", "energy")]
        return 1.0 - res / tot

    def report(self):
        """{'schema', 'energy_kept': {layer: {code: kept}}, 'lists': {layer: {code: {m: {metric: mean}}}}}"""
        kept, lists = {}, {}
        for key, (total, n) in self.sums.items():
            if len(key) == 3 and key[2] == "res":
                layer, code, _ = key
                kept.setdefault(layer, {})[code] = self.energy_kept(layer, code)
            elif len(key) == 4:
                layer, code, m, name = key
                lists.setdefault(layer, {}).setdefault(code, {}).setdefault(m, {})[name] = total / max(n, 1)
        for (layer, code, m), parts in self.medians.items():
            import torch

            lists[layer][code][m]["margin_ratio_median"] = float(torch.cat(parts).median())
        return {"schema": SCHEMA, "energy_kept": kept, "lists": lists}


# --------------------------------------------------------------------------- pools, bases, cost


def pool_from_counts(counts, p):
    """Static pool: boolean mask of the p neurons selected most often (counts [I])."""
    return top_m_mask(counts.float()[None], p)[0]


def pool_mass(scores, pool, m):
    """Per-position share of the oracle top-m score mass that lies inside a static pool [I]."""
    oracle = top_m_mask(scores, m)
    return (scores * (oracle & pool)).sum(-1) / (scores * oracle).sum(-1).clamp_min(1e-30)


def code_products(g, u, keep, base_gate, *, hybrid=None):
    """Per-neuron products whose down-projection is the code's MLP write.

    keep: bool mask (per position or a static pool) of neurons that use their exact gate g.
    base_gate: gate used elsewhere (token mean [.., I] or global mean [I]).
    hybrid: optional (g_tok, u_tok) for the token-table hybrid base, where u_tok = W_up n_tok:
        outside ``keep`` the product is g_tok * u_tok + base_gate * (u - u_tok), i.e. the write is
        y0_tok(n_tok) + B_0 (n - n_tok) with B_0 = W_down diag(base_gate) W_up.
    """
    if hybrid is None:
        return _where(keep, g * u, base_gate.to(g.dtype) * u)
    g_tok, u_tok = hybrid
    outside = g_tok.to(g.dtype) * u_tok + base_gate.to(g.dtype) * (u - u_tok)
    return _where(keep, g * u, outside)


def _where(mask, a, b):
    import torch

    return torch.where(mask, a, b)


def global_base_matrix(w_down, g_glob, w_up):
    """B_0 = W_down diag(g_glob) W_up, one d x d matrix per layer (w_down [d, I], w_up [I, d])."""
    return (w_down * g_glob[None, :]) @ w_up


def layer_macs(d, inter, context, n_heads, n_kv, head_dim):
    """Multiply-accumulates of one decoder layer for one new position at a given context length."""
    q = o = d * n_heads * head_dim
    k = v = d * n_kv * head_dim
    scores = 2 * context * n_heads * head_dim
    attention = q + o + k + v + scores
    return {"attention": attention, "mlp_matmul": d * inter, "mlp": 3 * d * inter, "total": attention + 3 * d * inter}


def code_layer_macs(kind, shapes, *, p=0, m=0, k=0):
    """Derived per-layer MACs of a code (not timed).

    kind: 'native' or 'full_select' (all gates computed: full layer), 'token_base' (attention + W_up + W_down),
    'global_base' (attention + B_0), 'pool' (attention + B_0 + p exact neurons), 'proxy' (attention + B_0 +
    rank-k proxy of a and u + m exact neurons).
    """
    L = layer_macs(**shapes)
    d, inter = shapes["d"], shapes["inter"]
    att = L["attention"]
    if kind in ("native", "full_select"):
        return L["total"]
    if kind == "token_base":
        return att + 2 * L["mlp_matmul"]
    if kind == "global_base":
        return att + d * d
    if kind == "pool":
        return att + d * d + 3 * p * d
    if kind == "proxy":
        return att + d * d + (d + 2 * inter) * k + 3 * m * d
    raise ValueError("unknown code kind: " + kind)


def flop_fraction(per_layer, shapes):
    """Fraction of a full column: per_layer is a list of (kind, options) for every layer."""
    full = layer_macs(**shapes)["total"] * len(per_layer)
    return sum(code_layer_macs(kind, shapes, **opts) for kind, opts in per_layer) / full


def average_ranks(x):
    """1-based ranks of a 1-D tensor with tied values sharing their average rank."""
    import torch

    _, inv, cnt = torch.unique(x, return_inverse=True, return_counts=True)
    ends = cnt.cumsum(0).double()
    return (ends - (cnt.double() - 1) / 2)[inv]


def spearman(x, y, *, ties=False):
    """Spearman rank correlation of two 1-D tensors (ties=True: average ranks, needed when one side is a table lookup
    with many tied values; default: no tie correction)."""
    rx = average_ranks(x) if ties else x.argsort().argsort().double()
    ry = average_ranks(y) if ties else y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))


# --------------------------------------------------------------------------- proxies


def silu_prime(a):
    """d silu(a) / da = sigma(a) (1 + a (1 - sigma(a))), in fp32."""
    import torch

    a = a.float()
    sg = torch.sigmoid(a)
    return sg * (1 + a * (1 - sg))


def reading_metric(w_gate, w_up, weights, c2, e2):
    """M = W_gate^T diag(w^2 c^2) W_gate + W_up^T diag(w^2 e^2) W_up (d x d, fp32).

    w_gate, w_up [I, d]; weights [I] (for example the Fisher neuron weight); c2 = E[(silu'(a) u)^2] and
    e2 = E[g^2] per neuron [I]. The Fisher-weighted error of the score inputs g * u under a rank-k map B of the
    gate input n is, to first order and with position-dependent coefficients replaced by their RMS,
    E || M^{1/2} (I - B) n ||^2.
    """
    wg = w_gate.float() * (weights.float() * c2.float().clamp_min(0).sqrt())[:, None]
    wu = w_up.float() * (weights.float() * e2.float().clamp_min(0).sqrt())[:, None]
    return wg.T @ wg + wu.T @ wu


def metric_proxy(metric, second_moment, k):
    """Optimal rank-k map B = decode @ encode^T of n under a reading metric (reduced-rank regression, fp64).

    Minimises E || M^{1/2} (I - B) n ||^2 = tr(M (I - B) S (I - B)^T) over rank-k B, S = E[n n^T] (or a
    centred covariance). Solved without inverting M: with S = L L^T and L^T M L = P diag(lam) P^T,
    decode = L P_k lam_k^{-1/2} (= M^{-1/2} U_k) and encode = M decode (= M^{1/2} U_k), where U_k are the top-k
    eigenvectors of M^{1/2} S M^{1/2}. With M = I this is the PCA of S. Returns encode [d, k], decode [d, k], the
    eigenvalues lam (descending) and the captured share sum(lam_k) / sum(lam).
    """
    import torch

    S = second_moment.double()
    Mm = metric.double()
    S, Mm = (S + S.T) / 2, (Mm + Mm.T) / 2
    s, Q = torch.linalg.eigh(S)
    L = Q * s.clamp_min(0).sqrt()[None, :]
    K = L.T @ Mm @ L
    lam, P = torch.linalg.eigh((K + K.T) / 2)
    lam, P = lam.flip(0).clamp_min(0), P.flip(1)
    lk = lam[:k].clamp_min(float(lam[0]) * 1e-15)
    decode = (L @ P[:, :k]) / lk.sqrt()[None, :]
    encode = Mm @ decode
    return {"encode": encode, "decode": decode, "eigenvalues": lam, "share": float(lam[:k].sum() / lam.sum())}


def proxy_metric_share(metric, second_moment, encode, decode):
    """Share of tr(M S) a rank-k map B = decode @ encode^T keeps: 1 - tr(M (I - B) S (I - B)^T) / tr(M S)."""
    import torch

    S, Mm = second_moment.double(), metric.double()
    R = torch.eye(S.shape[0], dtype=S.dtype, device=S.device) - decode.double() @ encode.double().T
    return float(1 - torch.trace(Mm @ R @ S @ R.T) / torch.trace(Mm @ S))


def proxy_gate_inputs(n, encode, dec_gate, dec_up, *, center_n=None, center_a=None, center_u=None):
    """Proxy gate pre-activation and up projection: z = (n - c_n) encode, a = c_a + z dec_gate^T, u = c_u + z dec_up^T.

    encode [d, k]; dec_gate = W_gate decode and dec_up = W_up decode [I, k] are precomputed. Centres are optional
    (a token table row or a global mean; c_a = W_gate c_n and c_u = W_up c_n as tables). Cost per position:
    d k + 2 I k multiply-accumulates.
    """
    x = n.float() if center_n is None else n.float() - center_n.float()
    z = x @ encode.float()
    a = z @ dec_gate.float().T
    u = z @ dec_up.float().T
    if center_a is not None:
        a = a + center_a.float()
    if center_u is not None:
        u = u + center_u.float()
    return a, u


def hybrid_error_scores(g, u, g_tok, u_tok, g_glob, weights):
    """Base-aware score for the token-table hybrid base: |g u - g_tok u_tok - g_glob (u - u_tok)| * weights.

    This is the per-neuron error the hybrid base makes outside the exact set (compare ``switch_scores``, which is
    the global base's error |g - g_glob| |u|)."""
    gf, uf, ut = g.float(), u.float(), u_tok.float()
    return (gf * uf - g_tok.float() * ut - g_glob.float() * (uf - ut)).abs() * weights.float()


# --------------------------------------------------------------------------- serving


def top1_margin(logits):
    """Top-1 minus top-2 logit along the last axis (fp32)."""
    top2 = logits.float().topk(2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def column_equivalents(f, rho, per=1000):
    """Mixed memory: per * (f + (1 - f) rho) columns, native positions at cost 1 and proxy positions at cost f.

    Readout excluded (see ``readout_fraction``)."""
    return per * (f + (1 - f) * rho)


def column_equivalents_decide_after(f, rho, per=1000):
    """per * (f + rho): the gate needs the proxy column's own margin, so a native position costs f + 1."""
    return per * (f + rho)


def readout_fraction(shapes, vocab, n_layers):
    """Readout MACs (d x vocab) per position as a fraction of a full column (n_layers decoder layers)."""
    return shapes["d"] * vocab / (layer_macs(**shapes)["total"] * n_layers)


@dataclass
class ServingLedger:
    """Per-position serving outcomes by position band: native rate, argmax agreement with native, KL(native || served),
    split into served (proxy) and native positions."""

    bands: tuple = ((4, 16), (16, 32), (32, 64), (64, 128))
    sums: dict = field(default_factory=dict)

    def add(self, positions, native, agree, kl, groups=None):
        """positions, native (bool), agree (bool), kl: 1-D tensors over served positions. groups: optional 1-D integer
        labels (for example a context-count bucket); each label is also reported as its own row ``group:<label>``."""
        nat = native.bool()
        rows = [("all", (positions >= min(b[0] for b in self.bands)) & (positions < max(b[1] for b in self.bands))),
                *[(f"{lo}-{hi - 1}", (positions >= lo) & (positions < hi)) for lo, hi in self.bands]]
        if groups is not None:
            rows += [(f"group:{int(g)}", groups == g) for g in groups.unique().tolist()]
        for name, m in rows:
            mn = m & nat
            tot = self.sums.setdefault(name, [0, 0.0, 0.0, 0.0, 0.0, 0.0])
            tot[0] += int(m.sum())
            tot[1] += float(nat[m].float().sum())
            tot[2] += float(agree[m].float().sum())
            tot[3] += float(kl[m].float().sum())
            tot[4] += float(agree[mn].float().sum())
            tot[5] += float(kl[mn].float().sum())

    def report(self, f=None, per=1000, readout=None, f_pre=None):
        """f: proxy column fraction; readout: readout fraction added per position to the decide-after count;
        f_pre: fraction already spent at an early-abort point (adds the early-abort count)."""
        out = {}
        for name, (n, nat, ag, kl, ag_n, kl_n) in self.sums.items():
            ns = n - nat
            r = {"n": n, "rho": nat / max(n, 1), "agree": ag / max(n, 1), "kl": kl / max(n, 1),
                 "agree_served": (ag - ag_n) / ns if ns else float("nan"), "kl_served": (kl - kl_n) / ns if ns else float("nan"),
                 "agree_native": ag_n / nat if nat else float("nan"), "kl_native": kl_n / nat if nat else float("nan")}
            if f is not None:
                r["column_equivalents"] = column_equivalents(f, r["rho"], per)
                r["column_equivalents_decide_after"] = column_equivalents_decide_after(f, r["rho"], per)
                if readout is not None:
                    r["decide_after_with_readout"] = r["column_equivalents_decide_after"] + per * readout
                if f_pre is not None:
                    r["column_equivalents_early_abort"] = column_equivalents_early_abort(f, f_pre, r["rho"], per)
            out[name] = r
        return out


# --------------------------------------------------------------------------- coverage-complete tables, gates


def ridge_from_moments(sxx, sxy, syy, n, grid, *, eig=None):
    """Multi-output ridge Y_c ~ X_c B with lambda by GCV, from centred moments (fp64).

    sxx = X_c^T X_c [p, p], sxy = X_c^T Y_c [p, q], syy = ||Y_c||_F^2, n rows; grid of lambdas. With
    sxx = V diag(D) V^T and Z = V^T sxy: B = V diag(1 / (D + lam)) Z, RSS = syy - sum_i ||z_i||^2 (D_i + 2 lam) / (D_i + lam)^2,
    df = sum_i D_i / (D_i + lam), GCV = RSS / (n - df)^2. eig = (D, V) reuses an eigendecomposition of sxx (sxx may then
    be None). Returns beta, lam, gcv, df, rss.
    """
    import torch

    if eig is None:
        D, V = torch.linalg.eigh((sxx.double() + sxx.double().T) / 2)
    else:
        D, V = eig
    D = D.double().clamp_min(0)
    V = V.double()
    Z = V.T @ sxy.double()
    zz = (Z * Z).sum(-1)
    best = None
    for lam in grid:
        lam = float(lam)
        rss = float(syy) - float((zz * (D + 2 * lam) / (D + lam) ** 2).sum())
        df = float((D / (D + lam)).sum())
        gcv = rss / max(n - df, 1e-9) ** 2
        if best is None or gcv < best["gcv"]:
            best = {"lam": lam, "gcv": gcv, "df": df, "rss": rss}
    best["beta"] = V @ (Z / (D + best["lam"])[:, None])
    return best


def shrinkage_lambda(tr_within, resid_sq, counts):
    """Empirical-Bayes weight lam = tr S_within / tr S_between, per layer.

    resid_sq [R]: ||n_emp_t - n_prior_t||^2 for held-out token types with counts c_t >= 1, whose expectation is
    tr S_between + tr S_within / c_t. Returns (lam, tr S_between)."""
    between = float((resid_sq.double() - float(tr_within) / counts.double()).mean())
    between = max(between, 1e-12 * float(tr_within))
    return float(tr_within) / between, between


def shrinkage_means(counts, emp_means, prior_means, lam):
    """Coverage-complete token means (c n_emp + lam n_prior) / (c + lam); types with c = 0 take the prior.

    counts [R], emp_means [R, d] (ignored where c = 0), prior_means [R, d]."""
    import torch

    c = counts.to(prior_means.dtype)[:, None]
    emp = torch.where(c > 0, emp_means.to(prior_means.dtype), torch.zeros_like(prior_means))
    return (c * emp + lam * prior_means) / (c + lam)


def error_mass(scores, keep):
    """Per-position predicted error mass of a code: the sum of squared scores outside its exact set.

    With Fisher neuron weights and the base-aware score this is the diagonal-Fisher energy the code chose not to
    evaluate (about twice the predicted KL)."""
    s = scores.float()
    return (s * s * (~keep)).sum(-1)


def quantile_threshold(stat, rho, *, below=True):
    """Scalar t such that a share rho of stat lies below t (below=True) or above t (below=False)."""
    import torch

    return float(torch.quantile(stat.float().flatten(), rho if below else 1 - rho))


def z_gate(margin, e_hat, c):
    """Mark a position native when its top-1 margin is below c sqrt(e_hat) (flip probability ~ 2 Phi(-margin / sigma))."""
    return margin < c * e_hat.clamp_min(0).sqrt()


def kl_gate(e_hat, kappa):
    """Mark a position native when the predicted error mass exceeds kappa."""
    return e_hat > kappa


def or_gate_thresholds(margin, e_hat, rho, *, iters=50):
    """Thresholds (tau, kappa) of the OR-gate margin < tau or e_hat > kappa with equal marginal shares s
    (P(margin < tau) = P(e_hat > kappa) = s) chosen so the union marks a share rho; bisection on s in [rho / 2, rho]."""
    import torch

    m, e = margin.float().flatten(), e_hat.float().flatten()
    lo, hi = rho / 2, rho
    for _ in range(iters):
        s = (lo + hi) / 2
        tau, kappa = float(torch.quantile(m, s)), float(torch.quantile(e, 1 - s))
        if float(((m < tau) | (e > kappa)).float().mean()) < rho:
            lo = s
        else:
            hi = s
    s = (lo + hi) / 2
    tau, kappa = float(torch.quantile(m, s)), float(torch.quantile(e, 1 - s))
    return {"tau": tau, "kappa": kappa, "marginal_share": s, "union_share": float(((m < tau) | (e > kappa)).float().mean()),
            "both_share": float(((m < tau) & (e > kappa)).float().mean())}


def column_equivalents_early_abort(f, f_pre, rho, per=1000):
    """Early abort: served positions cost f, aborted positions f_pre (the proxy layers already run) + 1."""
    return per * ((1 - rho) * f + rho * (f_pre + 1))


# --------------------------------------------------------------------------- bytes, timed records, pre-gates, decoders


def layer_bytes(kind, shapes, *, k=0, m=0, batch=1, dtype_bytes=2, impl="gather", union=None):
    """Derived bytes one decoder layer reads for one decode step of ``batch`` rows (weights, gathered rows, tables and
    the K/V cache at ``shapes['context']``), split into shared (read once per step) and per-row parts.

    kind 'native': attention + gate, up and down weights. kind 'proxy': attention + B_0 (d x d) + proxy encoder (d x k)
    + two proxy decoders (I x k each) shared, plus per row the exact rows 3 m d (impl 'gather': every row gathers its
    own m rows) and the token tables (a, u, g at I each, n and y0 at d each); impl 'union' reads the union of the
    batch's exact sets once (``union`` neurons) instead of per-row rows.
    """
    d, inter, ctx = shapes["d"], shapes["inter"], shapes["context"]
    nh, nkv, hd = shapes["n_heads"], shapes["n_kv"], shapes["head_dim"]
    att = 2 * d * nh * hd + 2 * d * nkv * hd + nh * hd + 2 * nkv * hd + 2 * d
    kv = 2 * ctx * nkv * hd
    if kind == "native":
        shared, row = att + 3 * d * inter, kv
    elif kind == "proxy":
        shared = att + d * d + d * k + 2 * inter * k
        row = kv + 3 * inter + 2 * d
        if impl == "gather":
            row += 3 * m * d
        elif impl == "union":
            shared += 3 * (inter if union is None else union) * d
        else:
            raise ValueError("impl must be 'gather' or 'union'")
    else:
        raise ValueError("unknown kind: " + kind)
    return {"shared": shared * dtype_bytes, "per_row": row * dtype_bytes, "total": (shared + batch * row) * dtype_bytes}


def bytes_per_token(per_layer, shapes, *, vocab, batch=1, readout=True, dtype_bytes=2):
    """Derived bytes per decode step and per token for a column given as [(kind, options)] per layer, plus the readout
    (vocab x d, shared) when ``readout``. Options are passed to ``layer_bytes``."""
    step = sum(layer_bytes(kind, shapes, batch=batch, dtype_bytes=dtype_bytes, **opts)["total"] for kind, opts in per_layer)
    if readout:
        step += vocab * shapes["d"] * dtype_bytes
    return {"bytes_per_step": step, "bytes_per_token": step / batch, "batch": batch}


def timed_record(timing, derived, *, baseline="native"):
    """Join an ``mdb.timing.time_arms`` record with derived per-arm bytes and FLOP fractions.

    derived: {arm: {'bytes_per_step': ..., 'flop_fraction': ...}}. Adds achieved GB/s = bytes / median time, and the
    measured time ratio next to the derived byte and FLOP ratios against the baseline (time ratios only when the
    record's parity gate passed)."""
    out = {"schema": "mdb-timed-record-v1", "batch": timing.get("batch"), "claim": timing.get("claim"), "arms": {}}
    base_ms = timing["arms"][baseline]["median_ms"]
    base_b = derived[baseline]["bytes_per_step"]
    passed = timing.get("claim") == "speedups reported"
    for name, r in timing["arms"].items():
        dv = derived.get(name, {})
        b = dv.get("bytes_per_step")
        row = {"median_ms": r["median_ms"], "p10_ms": r["p10_ms"], "p90_ms": r["p90_ms"], "bytes_per_step": b,
               "flop_fraction": dv.get("flop_fraction"),
               "achieved_GBps": (b / (r["median_ms"] * 1e-3) / 1e9) if b else None,
               "byte_ratio_vs_baseline": (b / base_b) if b else None,
               "time_ratio_vs_baseline": (r["median_ms"] / base_ms) if passed else None}
        out["arms"][name] = row
    return out


def pregate_table(keys, values, n_keys, *, prior=None):
    """Empirical-Bayes table of a per-position value by key (for example the token id).

    Returns (table [n_keys], lam, stats): table = (c mean + lam prior) / (c + lam) with c the key count, prior the global
    mean (or a per-key prior [n_keys], for example a unigram table under a bigram key), and lam = s2_within / s2_between
    estimated from keys with c >= 2 (between = var of key means around the prior minus the mean s2_within / c)."""
    import torch

    keys = keys.long()
    v = values.double()
    c = torch.zeros(n_keys, dtype=torch.float64).index_add_(0, keys, torch.ones_like(v))
    s1 = torch.zeros(n_keys, dtype=torch.float64).index_add_(0, keys, v)
    s2 = torch.zeros(n_keys, dtype=torch.float64).index_add_(0, keys, v * v)
    pr = torch.full((n_keys,), float(v.mean()), dtype=torch.float64) if prior is None else prior.double()
    seen = c >= 2
    mean = s1 / c.clamp_min(1)
    within = float(((s2 - c * mean ** 2)[seen]).sum() / (c[seen] - 1).sum().clamp_min(1))
    between = float(((mean - pr) ** 2)[seen].mean() - (within / c[seen]).mean()) if bool(seen.any()) else 0.0
    between = max(between, 1e-12 * max(within, 1e-30))
    lam = within / between
    table = (s1 + lam * pr) / (c + lam)
    return table, lam, {"within": within, "between": between, "keys_seen": int((c > 0).sum()), "keys_ge2": int(seen.sum())}


def quantize_rows(w, bits):
    """Symmetric per-row integer quantization: returns (dequantized float32 weights, per-row scales)."""
    import torch

    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    scale = (wf.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-30)
    q = torch.round(wf / scale).clamp_(-qmax, qmax)
    return q * scale, scale[..., 0]

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
Thresholds and ramps (``ScoreHistogram``, ``ramp_weights``, ``ramp_products``): a calibrated per-layer (or global)
score threshold replaces the per-position top-m (variable m, mean count fixed in closed form from pooled fit-set
quantiles); a soft ramp above the threshold makes the code continuous in the residual. ``rank_waterfill`` allocates
proxy rank across layers from the metric eigenvalues; ``group_sets`` shares one exact set across a group of rows;
``kl_law`` converts a stream's KL to the agreement the pure-stream law predicts.
Part AJ: ``pool_rescore_mask`` / ``pool_energy_recall`` (a proxied score admits a candidate pool that is rescored
exactly), ``up_reading_metric`` (the gate-weighted up metric of G2U's proxy), ``lowrank_b0_maps`` and the ``b0``
ledger option (token-only or rank-r global base), int8 down rows (``down_qbits``), ``adjacent_set_jaccard`` and
``TokenTable`` (temporal reuse and token-determined attention diagnostics).
Part AK: ``proxy_residual_sigma`` / ``posterior_pool_key`` / ``two_side_pool_key`` / ``orth_residual_norm`` (a pool
ranked by the posterior probability of membership in the exact set), ``first_order_base_*`` and ``FirstOrderMoments``
(the gate-linear term B_01 the hybrid base lacks), ``tail_residual_components``, ``paired_chunk_bootstrap`` /
``chord_mixture`` (paired noise model for frontier claims), ``exact_family_law`` / ``pool_family_law`` (per-family
frontier laws), ``sigma_margin_statistic`` / ``gaussian_tail_fit`` / ``two_level_readout_bytes`` (readout certificate).
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


def _b0_macs(b0, d):
    """MACs of the hybrid base's global term B_0 dz: 'full' d x d, 'none' 0 (the token-only base y_tok), an int r a
    rank-r factorisation D_r E_r (2 r d)."""
    if b0 == "full":
        return d * d
    if b0 == "none":
        return 0
    return 2 * int(b0) * d


def code_layer_macs(kind, shapes, *, p=0, m=0, k=0, r=0, density=1.0, regather=False, densities=None, b0="full"):
    """Derived per-layer MACs of a code (not timed).

    kind: 'native' or 'full_select' (all gates computed: full layer), 'token_base' (attention + W_up + W_down),
    'global_base' (attention + B_0), 'pool' (attention + B_0 + p exact neurons), 'proxy' (attention + B_0 +
    rank-k proxy of a and u + m exact neurons), 'exact_select' (attention + the exact gate and up projections of every
    neuron + B_0 + the down rows of m exact neurons: the proxy-free selection limit; ``regather`` adds gathered gate/up
    rows of the m neurons, ``r`` an output-side rank-r tail proxy r (I + d)). m may be a mean count.
    Prior-art baselines (Part AH): 'cats' (attention + the gate GEMV of every neuron + up and down rows of m neurons,
    zero base), 'cats_hybrid' ('cats' + B_0 for the hybrid base), 'exact_select_zero' (gate and up GEMVs + m down rows,
    zero base), 'teal' (every projection's input thresholded to ``density``: density x the attention and MLP weight
    MACs, the attention scores dense), 'attention_only' (no MLP write).
    Part AI: 'gate_exact_up_proxy' (G2U: attention + the gate GEMV of every neuron + B_0 + a rank-k up proxy, encoder
    d x k and decoder I x k + gathered up and down rows of m neurons); 'exact_select_zero' takes ``r`` (a rank-r
    remainder on the zero base, R-Sparse style); 'teal_alloc' (TEAL's per-matrix allocation: ``densities`` maps
    q, k, v, o, gate, up, down to the kept input share of each matrix).
    Part AJ: ``b0`` ('full' | 'none' | rank r) prices the hybrid base's B_0 term in 'exact_select',
    'gate_exact_up_proxy' and 'pool_rescore' (``_b0_macs``); 'pool_rescore' (G2U-P: attention + the gate GEMV of every
    neuron + B_0 + a rank-k up proxy + the gathered up rows of a candidate pool of ``p`` neurons, rescored exactly + the
    down rows of the m selected neurons).
    Part AK: 'pool2_rescore' (K1-Pz: attention + B_0 + rank-k proxies of a and u, encoder d x k and two decoders I x k +
    the gathered gate and up rows of a pool of ``p`` neurons, rescored exactly + the down rows of the m selected).
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
    if kind == "exact_select":
        return att + 2 * d * inter + _b0_macs(b0, d) + m * d + (2 * m * d if regather else 0) + r * (inter + d)
    if kind == "cats":
        return att + d * inter + 2 * m * d
    if kind == "cats_hybrid":
        return att + d * inter + d * d + 2 * m * d
    if kind == "exact_select_zero":
        return att + 2 * d * inter + m * d + r * (inter + d)
    if kind == "gate_exact_up_proxy":
        return att + d * inter + _b0_macs(b0, d) + k * (d + inter) + 2 * m * d
    if kind == "pool_rescore":
        return att + d * inter + _b0_macs(b0, d) + k * (d + inter) + p * d + m * d
    if kind == "pool2_rescore":
        return att + _b0_macs(b0, d) + k * (d + 2 * inter) + 2 * p * d + m * d
    if kind == "teal_alloc":
        qo = d * shapes["n_heads"] * shapes["head_dim"]
        kv_ = d * shapes["n_kv"] * shapes["head_dim"]
        foot = {"q": qo, "k": kv_, "v": kv_, "o": qo, "gate": d * inter, "up": d * inter, "down": d * inter}
        scores = 2 * shapes["context"] * shapes["n_heads"] * shapes["head_dim"]
        return sum(densities[n_] * f_ for n_, f_ in foot.items()) + scores
    if kind == "teal":
        scores = 2 * shapes["context"] * shapes["n_heads"] * shapes["head_dim"]
        return density * (att - scores + 3 * d * inter) + scores
    if kind == "attention_only":
        return att
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


def metric_proxy(metric, second_moment, k, *, dtype=None):
    """Optimal rank-k map B = decode @ encode^T of n under a reading metric (reduced-rank regression, fp64).

    Minimises E || M^{1/2} (I - B) n ||^2 = tr(M (I - B) S (I - B)^T) over rank-k B, S = E[n n^T] (or a
    centred covariance). Solved without inverting M: with S = L L^T and L^T M L = P diag(lam) P^T,
    decode = L P_k lam_k^{-1/2} (= M^{-1/2} U_k) and encode = M decode (= M^{1/2} U_k), where U_k are the top-k
    eigenvectors of M^{1/2} S M^{1/2}. With M = I this is the PCA of S. Returns encode [d, k], decode [d, k], the
    eigenvalues lam (descending) and the captured share sum(lam_k) / sum(lam). ``dtype`` (Part AJ): the eigensolver's
    precision (default float64; float32 is ~10x faster on consumer GPUs at d ~ 1,536).
    """
    import torch

    dt = torch.float64 if dtype is None else dtype
    S = second_moment.to(dt)
    Mm = metric.to(dt)
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


def column_equivalents_decide_after_honest(f, rho, readout, *, needs_readout, per=1000):
    """Decide-after cost in the readout-separate convention: per (f + rho) for keys known before the readout (count,
    score mass), per (f + rho (1 + readout)) for keys computed from the served logits (entropy, margin), whose served
    readout is spent and discarded at every position that goes native (ADVISOR 16.1 d)."""
    return per * (f + rho * (1 + (readout if needs_readout else 0.0)))


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

    def report(self, f=None, per=1000, readout=None, f_pre=None, law=None, needs_readout=None):
        """f: proxy column fraction; readout: readout fraction added per position to the decide-after count;
        f_pre: fraction already spent at an early-abort point (adds the early-abort count); law: (a0, beta) of the
        agreement law A = a0 - beta sqrt(KL) (``kl_law``), adding the law's agreement and the excess over it.
        needs_readout (Part AI): True when the gate's key needs the served logits (entropy, margin), so the served
        readout spent at a position that then goes native is wasted: adds ``decide_after_honest`` = per (f + rho (1 +
        readout)) in the readout-separate convention (``column_equivalents_decide_after_honest``); False or None for
        count and mass keys, whose honest cost is the plain decide-after count."""
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
                if needs_readout is not None:
                    r["decide_after_honest"] = column_equivalents_decide_after_honest(
                        f, r["rho"], readout or 0.0, needs_readout=needs_readout, per=per)
            if law is not None:
                r["agree_law"] = kl_law(r["kl"], *law)
                r["law_excess"] = r["agree"] - r["agree_law"]
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


def layer_bytes(kind, shapes, *, k=0, m=0, batch=1, dtype_bytes=2, impl="gather", union=None, qbits=None, group=None,
                scale_bytes=4, hybrid=True, regather=False, r=0, density=1.0, tail_qbits=None, dec_qbits=4, b0="full",
                down_qbits=None, p=0, pool_qbits=8):
    """Derived bytes one decoder layer reads for one decode step of ``batch`` rows (weights, gathered rows, tables and
    the K/V cache at ``shapes['context']``), split into shared (read once per step) and per-row parts.

    kind 'native': attention + gate, up and down weights. kind 'proxy': attention + B_0 (d x d) + proxy encoder (d x k)
    + two proxy decoders (I x k each) shared, plus per row the exact rows 3 m d (impl 'gather': every row gathers its
    own m rows) and the token tables (a, u, g at I each, n and y0 at d each); impl 'union' reads the union of the
    batch's exact sets once (``union`` neurons) instead of per-row rows. kind 'exact_select': attention + W_gate + W_up +
    B_0 shared, plus per row the m exact down rows (``union``: read once) and the token tables.

    qbits: integer storage of the scoring matrices (kind 'exact_select': W_gate and W_up; kind 'proxy': the two
    decoders; kind 'cats': W_gate) at qbits / 8 bytes per weight plus one ``scale_bytes`` scale per row (``group``
    None) or per group of ``group`` columns; everything else stays at ``dtype_bytes``.

    Part AH: kind 'exact_select' with ``regather`` adds the m gathered gate/up rows per row (exact products on the set)
    and ``r`` an output-side tail proxy (r x I and d x r, shared). Kind 'cats' (CATS-style): W_gate read in full, then
    per row the m up rows and m down rows; ``hybrid`` adds B_0 (shared) and the token tables (u_tok, g_tok, n_tok,
    y0 per row). Kind 'teal': ``density`` x every projection weight (rows of W^T gathered by the thresholded input),
    the K/V cache in full.

    Part AI: ``tail_qbits`` stores the tail proxy's encoder (r x I) at that width with per-row scales (the decoder
    d x r stays at ``dtype_bytes``). Kind 'g2u' (gate exact, up proxied): attention + B_0 + the encoder (d x k) shared,
    W_gate at ``qbits`` and the up decoder (I x k) at ``dec_qbits`` with per-row scales, then per row the token tables
    and the m gathered up rows and m down rows.

    Part AJ: ``b0`` ('full' d x d, 'none', or a rank r: E_r and D_r, 2 r d) prices B_0 in kinds 'proxy',
    'exact_select', 'g2u' and 'pool'; ``down_qbits`` stores the gathered down rows (W_down columns) at that width with
    one scale per row (kinds 'exact_select', 'g2u', 'pool'). Kind 'pool' (G2U-P): as 'g2u', but the gathered up rows are
    the ``p`` rows of the candidate pool at ``pool_qbits`` (int8 per row) and the down rows are the m rows of the final
    set.
    """
    d, inter, ctx = shapes["d"], shapes["inter"], shapes["context"]
    nh, nkv, hd = shapes["n_heads"], shapes["n_kv"], shapes["head_dim"]
    att = 2 * d * nh * hd + 2 * d * nkv * hd + nh * hd + 2 * nkv * hd + 2 * d
    kv = 2 * ctx * nkv * hd
    qshared = 0
    b0e = _b0_macs(b0, d)

    def _down(n_):
        # (elements at dtype_bytes, bytes) of n_ gathered down rows
        if not down_qbits:
            return n_ * d, 0
        return 0, _qbytes(n_, d, down_qbits, None, scale_bytes, dtype_bytes)
    qrow = 0
    if kind == "native":
        shared, row = att + 3 * d * inter, kv
    elif kind == "proxy":
        shared = att + b0e + d * k
        qshared = _qbytes(2 * inter, k, qbits, group, scale_bytes, dtype_bytes)
        row = kv + 3 * inter + 2 * d
        if impl == "gather":
            row += 3 * m * d
        elif impl == "union":
            shared += 3 * (inter if union is None else union) * d
        else:
            raise ValueError("impl must be 'gather' or 'union'")
    elif kind == "exact_select":
        shared = att + b0e + (r * d if tail_qbits else r * (inter + d))
        qshared = _qbytes(2 * inter, d, qbits, group, scale_bytes, dtype_bytes)
        if tail_qbits and r:
            qshared += _qbytes(r, inter, tail_qbits, None, scale_bytes, dtype_bytes)
        row = kv + 3 * inter + 2 * d
        per = 3 if regather else 1
        if impl == "gather":
            de, db = _down(m)
            row += (per - 1) * m * d + de
            qrow += db
        elif impl == "union":
            shared += per * (inter if union is None else union) * d
        else:
            raise ValueError("impl must be 'gather' or 'union'")
    elif kind == "cats":
        shared = att + (d * d if hybrid else 0)
        qshared = _qbytes(inter, d, qbits, group, scale_bytes, dtype_bytes)
        row = kv + ((2 * inter + 2 * d) if hybrid else 0)
        if impl == "gather":
            row += 2 * m * d
        elif impl == "union":
            shared += 2 * (inter if union is None else union) * d
        else:
            raise ValueError("impl must be 'gather' or 'union'")
    elif kind in ("g2u", "pool"):
        shared = att + b0e + d * k
        qshared = _qbytes(inter, d, qbits, group, scale_bytes, dtype_bytes) + _qbytes(inter, k, dec_qbits, None, scale_bytes,
                                                                                       dtype_bytes)
        row = kv + 3 * inter + 2 * d
        if impl == "gather":
            de, db = _down(m)
            row += de
            qrow += db
            if kind == "g2u":
                row += m * d
            else:
                qrow += _qbytes(p, d, pool_qbits, None, scale_bytes, dtype_bytes)
        elif impl == "union":
            shared += 2 * (inter if union is None else union) * d
        else:
            raise ValueError("impl must be 'gather' or 'union'")
    elif kind == "teal":
        proj = 2 * d * nh * hd + 2 * d * nkv * hd + 3 * d * inter
        shared = density * proj + (att - (2 * d * nh * hd + 2 * d * nkv * hd))
        row = kv
    else:
        raise ValueError("unknown kind: " + kind)
    sh = shared * dtype_bytes + qshared
    pr = row * dtype_bytes + qrow
    return {"shared": sh, "per_row": pr, "total": sh + batch * pr}


def _qbytes(rows, cols, qbits, group, scale_bytes, dtype_bytes):
    """Bytes of a rows x cols matrix stored at dtype_bytes, or at qbits with per-row (group None) or per-group scales."""
    if not qbits:
        return rows * cols * dtype_bytes
    n_scales = rows * (1 if group is None else -(-cols // group))
    return rows * cols * qbits / 8 + n_scales * scale_bytes


def bytes_per_token(per_layer, shapes, *, vocab, batch=1, readout=True, dtype_bytes=2, readout_bits=None, scale_bytes=4):
    """Derived bytes per decode step and per token for a column given as [(kind, options)] per layer, plus the readout
    (vocab x d, shared) when ``readout`` (at ``readout_bits`` with per-row scales when given). Options are passed to
    ``layer_bytes``."""
    step = sum(layer_bytes(kind, shapes, batch=batch, dtype_bytes=dtype_bytes, **opts)["total"] for kind, opts in per_layer)
    if readout:
        step += _qbytes(vocab, shapes["d"], readout_bits, None, scale_bytes, dtype_bytes)
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


# --------------------------------------------------------------------------- thresholds, ramps, rank, groups, KL law


def kl_law(kl, a0=0.967, beta=0.403):
    """Agreement predicted from a stream's mean KL by A = a0 - beta sqrt(KL).

    Derivation: the argmax flips when the top-1/top-2 logit-difference error exceeds the native margin; with a margin
    density flat near zero and a Gaussian error of standard deviation proportional to sqrt(KL), P(flip) is linear in
    sqrt(KL). The defaults are the fit to nine pure Qwen2.5-1.5B streams (Parts AD-AE); mixed streams sit above the line
    (the square root is concave), so the excess measures how well a split separates easy from hard positions."""
    return a0 - beta * max(float(kl), 0.0) ** 0.5


def fit_kl_law(agree, kl):
    """Least-squares (a0, beta, rms) of A = a0 - beta sqrt(KL) over paired stream summaries (closed form)."""
    import torch

    a = torch.as_tensor(agree, dtype=torch.float64).flatten()
    s = torch.as_tensor(kl, dtype=torch.float64).flatten().clamp_min(0).sqrt()
    X = torch.stack([torch.ones_like(s), -s], 1)
    coef = torch.linalg.lstsq(X, a[:, None]).solution[:, 0]
    rms = float(((X @ coef - a) ** 2).mean().sqrt())
    return float(coef[0]), float(coef[1]), rms


class ScoreHistogram:
    """Per-layer log-binned histogram of per-neuron scores, for closed-form threshold calibration.

    ``add(layer, scores)`` pools positions x neurons ([..., I] scores, I neurons per position); a value-weighted
    histogram (``values``, for example true scores keyed by predicted scores) can be pooled alongside. ``threshold``
    returns the score theta whose expected count above it per position equals ``mean_count`` (the (1 - mean_count / I)
    pooled quantile); ``global_threshold`` does the same with one theta across layers for a mean total count. Bins are
    uniform in log(score) between ``lo`` and ``hi``; within a bin the count is interpolated linearly in log(score).
    Scores at or below exp(lo) fall in the bottom bin."""

    def __init__(self, n_layers, lo=-40.0, hi=12.0, bins=26000, device=None):
        import torch

        self.lo, self.hi, self.bins = float(lo), float(hi), int(bins)
        self.count = torch.zeros(n_layers, bins, dtype=torch.float64, device=device)
        self.value = torch.zeros(n_layers, bins, dtype=torch.float64, device=device)
        self.value2 = torch.zeros(n_layers, bins, dtype=torch.float64, device=device)
        self.positions = [0] * n_layers

    def _bin(self, scores):
        x = scores.float().clamp_min(1e-38).log()
        b = ((x - self.lo) / (self.hi - self.lo) * self.bins).floor().long()
        return b.clamp_(0, self.bins - 1)

    def add(self, layer, scores, values=None):
        """scores [..., I]; values (same shape, optional) are summed per bin (and squared, in ``value2``)."""
        import torch

        b = self._bin(scores).flatten()
        self.count[layer] += torch.bincount(b, minlength=self.bins).double()
        if values is not None:
            v = values.double().flatten()
            self.value[layer] += torch.bincount(b, weights=v, minlength=self.bins)
            self.value2[layer] += torch.bincount(b, weights=v * v, minlength=self.bins)
        self.positions[layer] += scores.numel() // scores.shape[-1]

    def edges(self):
        import torch

        return torch.linspace(self.lo, self.hi, self.bins + 1, dtype=torch.float64, device=self.count.device)

    def _theta(self, counts, target):
        import torch

        above = counts.flip(0).cumsum(0).flip(0)  # count in bins >= b
        e = self.edges()
        hit = torch.nonzero(above >= target)
        if len(hit) == 0:
            return float(e[0].exp())
        b = int(hit[-1])
        nxt = float(above[b + 1]) if b + 1 < self.bins else 0.0
        frac = (target - nxt) / max(float(counts[b]), 1e-30)  # share of bin b needed, taken from its top
        return float((e[b + 1] - frac * (e[b + 1] - e[b])).exp())

    def threshold(self, layer, mean_count):
        return self._theta(self.count[layer], mean_count * self.positions[layer])

    def global_threshold(self, mean_total_count):
        n = self.positions[0]
        if any(p != n for p in self.positions):
            raise ValueError("global threshold needs the same positions at every layer")
        return self._theta(self.count.sum(0), mean_total_count * n)

    def above(self, layer, theta):
        """(mean count, value sum, value^2 sum) per position above theta (bin-interpolated)."""
        import math

        x = (math.log(max(theta, 1e-38)) - self.lo) / (self.hi - self.lo) * self.bins
        b = min(max(int(math.floor(x)), 0), self.bins - 1)
        part = min(max(b + 1 - x, 0.0), 1.0)
        n = max(self.positions[layer], 1)
        out = []
        for h in (self.count[layer], self.value[layer], self.value2[layer]):
            out.append((float(h[b + 1:].sum()) + part * float(h[b])) / n)
        return tuple(out)


def ramp_weights(scores, theta_lo, eta):
    """Soft threshold: select j when score_j > theta_lo and weight its exact correction by
    omega_j = min(1, (score_j - theta_lo) / (theta_hi - theta_lo)) with theta_hi = theta_lo / (1 - eta). eta = 0 is the
    hard threshold (omega = 1 on the set). theta_lo may broadcast (one value per layer or per position).
    Returns (keep, omega) with omega = 0 outside keep."""
    import torch

    s = scores.float()
    lo = torch.as_tensor(theta_lo, dtype=s.dtype, device=s.device)
    keep = s > lo
    if eta <= 0:
        return keep, keep.to(s.dtype)
    width = lo * eta / (1 - eta)
    om = ((s - lo) / width.clamp_min(1e-30)).clamp(0, 1)
    return keep, torch.where(keep, om, torch.zeros_like(om))


def ramp_products(g, u, omega, base_gate, *, hybrid):
    """Products of a ramp code: base + omega (g u - base), base = g_tok u_tok + base_gate (u - u_tok) (the hybrid base).
    omega in [0, 1] (0 outside the set). Computed in fp32, returned in g's dtype."""
    g_tok, u_tok = hybrid
    gf, uf, ut = g.float(), u.float(), u_tok.float()
    base = g_tok.float() * ut + base_gate.float() * (uf - ut)
    return (base + omega.float() * (gf * uf - base)).to(g.dtype)


def rank_waterfill(eigenvalues, K, *, cap=None):
    """Per-layer proxy ranks k_l minimising the total discarded metric energy sum_l sum_{i > k_l} lam_{l,i} under
    sum_l k_l = K: take the K largest eigenvalues over all (layer, index) pairs (they are in one output-weighted unit, so
    commensurable). eigenvalues: list of 1-D descending tensors per layer; cap: optional maximum rank per layer.
    Returns {'k': [k_l], 'theta': the smallest kept eigenvalue, 'kept_share', 'uniform_kept_share', 'cap_binds'}."""
    import torch

    nL = len(eigenvalues)
    lam = [e.double().flatten().clamp_min(0) for e in eigenvalues]
    if cap is not None:
        lam_c = [e[:cap] for e in lam]
    else:
        lam_c = lam
    allv = torch.cat(lam_c)
    lay = torch.cat([torch.full((len(e),), i, dtype=torch.long, device=e.device) for i, e in enumerate(lam_c)])
    order = torch.argsort(allv, descending=True, stable=True)[:K]
    k = torch.bincount(lay[order], minlength=nL).tolist()
    tot = float(sum(float(e.sum()) for e in lam))
    kept = float(sum(float(lam[i][:k[i]].sum()) for i in range(nL)))
    ku = K // nL
    uni = float(sum(float(e[:ku].sum()) for e in lam))
    return {"k": k, "theta": float(allv[order[-1]]) if K else float("inf"), "kept_share": kept / tot,
            "uniform_kept_share": uni / tot, "cap_binds": bool(cap is not None and any(x >= cap for x in k))}


def group_sets(scores, G, m):
    """Group-level exact sets: rows [N, I] taken in consecutive groups of G share the top-m neurons by the group's
    summed score; returns the per-row boolean mask [N, I] (a trailing partial group uses its own rows)."""
    import torch

    N, I = scores.shape
    ng = -(-N // G)
    pad = torch.zeros(ng * G, I, dtype=torch.float32, device=scores.device)
    pad[:N] = scores.float()
    S = pad.view(ng, G, I).sum(1)  # a trailing partial group sums only its own rows (zero padding adds nothing)
    idx = S.topk(min(m, I), dim=-1).indices
    gm = torch.zeros(ng, I, dtype=torch.bool, device=scores.device).scatter_(-1, idx, True)
    return gm.repeat_interleave(G, 0)[:N]


def set_jaccard(a, b):
    """Row-wise Jaccard |a & b| / |a | b| of boolean masks [..., I]."""
    inter = (a & b).sum(-1).float()
    union = (a | b).sum(-1).float()
    return inter / union.clamp_min(1)



# --------------------------------------------------------------------------- integer codes, readout candidates, gates


def quantize_int(w, bits, group=None):
    """Symmetric integer quantization of a matrix [rows, cols] per row (``group`` None) or per group of ``group``
    consecutive columns. Returns (q int8 [rows, cols] in [-(2^(bits-1) - 1), 2^(bits-1) - 1], scale float32 [rows] or
    [rows, cols / group]); w ~ q * scale. Per row this is ``quantize_rows`` with the integers kept."""
    import torch

    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    if group is None:
        scale = (wf.abs().amax(-1) / qmax).clamp_min(1e-30)
        q = torch.round(wf / scale[:, None]).clamp_(-qmax, qmax)
        return q.to(torch.int8), scale
    rows, cols = wf.shape
    if cols % group:
        raise ValueError("cols must be a multiple of group")
    wg = wf.view(rows, cols // group, group)
    scale = (wg.abs().amax(-1) / qmax).clamp_min(1e-30)
    q = torch.round(wg / scale[..., None]).clamp_(-qmax, qmax)
    return q.view(rows, cols).to(torch.int8), scale


def dequantize_int(q, scale, group=None):
    """float32 weights q * scale of ``quantize_int``."""
    if group is None:
        return q.float() * scale[:, None]
    rows, cols = q.shape
    return (q.float().view(rows, cols // group, group) * scale[..., None]).view(rows, cols)


def pack_int4(q):
    """Two signed int4 values per byte along the last axis (column 2j in the low nibble, 2j + 1 in the high nibble),
    each stored as q + 8 in 0..15. q: int8 [..., cols] with cols even. Returns uint8 [..., cols / 2]."""
    import torch

    u = (q.to(torch.int16) + 8).to(torch.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4)


def unpack_int4(p):
    """Inverse of ``pack_int4``: int8 [..., 2 cols]."""
    import torch

    lo = (p & 15).to(torch.int16) - 8
    hi = (p >> 4).to(torch.int16) - 8
    return torch.stack([lo, hi], -1).flatten(-2).to(torch.int8)


def head_subspace(w_head, r_max):
    """Top-r_max right singular directions of an output head w_head [V, d] from its d x d Gram (fp64 eigh, chunked over
    the vocabulary). Returns (basis float32 [d, r_max], descending singular values squared [d])."""
    import torch

    V, d = w_head.shape
    G = torch.zeros(d, d, dtype=torch.float64, device=w_head.device)
    for c0 in range(0, V, 16384):
        Wc = w_head[c0:c0 + 16384].double()
        G += Wc.T @ Wc
    ev, Q = torch.linalg.eigh(G)
    order = torch.argsort(ev, descending=True)
    return Q[:, order[:r_max]].float().contiguous(), ev[order]


def certified_candidates(lhat, h_perp, e_perp):
    """Cauchy-Schwarz candidate set of a projected readout: with lhat_v = (P_r e_v) . (P_r h) and
    |l_v - lhat_v| <= |e_v_perp| |h_perp|, the exact argmax lies in
    C(h) = {v : lhat_v >= max_w lhat_w - |h_perp| (|e_v_perp| + max_w |e_w_perp|)}.
    lhat [P, V], h_perp [P], e_perp [V]. Returns (mask [P, V] bool, sizes [P])."""
    top = lhat.max(-1, keepdim=True).values
    thr = top - h_perp[:, None] * (e_perp[None, :] + e_perp.max())
    mask = lhat >= thr
    return mask, mask.sum(-1)


def blockwise_topk_recall(lhat, target, block, k):
    """Share of rows whose target index is among the top-k of its block of ``block`` columns, taking the top-k of every
    block (the graph-capturable empirical candidate set of size n_blocks * k). lhat [P, V]; target [P] long."""
    import torch

    P, V = lhat.shape
    nb = -(-V // block)
    pad = torch.full((P, nb * block), float("-inf"), dtype=lhat.dtype, device=lhat.device)
    pad[:, :V] = lhat
    vals = pad.view(P, nb, block)
    kth = vals.topk(k, dim=-1).values[..., -1]  # [P, nb]
    tb = target // block
    tv = lhat.gather(1, target[:, None])[:, 0]
    return float((tv >= kth.gather(1, tb[:, None])[:, 0]).float().mean())


def kl_log_dispersion(kl, floor=1e-6):
    """sd of ln(max(KL_i, floor)) over positions: the log-dispersion sigma_k of per-position served KL."""
    import torch

    return float(torch.as_tensor(kl, dtype=torch.float64).clamp_min(floor).log().std())


def metric_eigen_share(second_moment, metric, ranks):
    """Top-r share of the eigenvalues of M^{1/2} S M^{1/2} (S a second moment, M a PSD metric, both d x d): the metric
    energy of S that a rank-r map captures. Returns {r: share}."""
    import torch

    Mv, MQ = torch.linalg.eigh(((metric + metric.T) / 2).double())
    Mh = (MQ * Mv.clamp_min(0).sqrt()[None, :]) @ MQ.T
    S = ((second_moment + second_moment.T) / 2).double()
    ev = torch.linalg.eigvalsh(Mh @ S @ Mh).flip(0).clamp_min(0)
    tot = float(ev.sum())
    return {int(r): (float(ev[:r].sum()) / tot if tot > 0 else float("nan")) for r in ranks}


def gate_frontier(counts, kl, agree, f, f_pre, *, depths, rhos, a_native=0.925, law=(0.967, 0.403), budget=850.0,
                  per=1000):
    """Offline count-gate frontier from one free stream.

    counts [P, L] per-layer selection counts at served positions; kl [P] and agree [P] the stream's per-position KL and
    argmax agreement; f the served column's realized fraction; f_pre [L] the fraction spent through layer l. For an
    abort depth l in ``depths`` and a native share rho in ``rhos``, positions whose prefix count sum_{l' <= l} exceeds
    kappa (the (1 - rho) quantile) go native: predicted agreement rho_r a_native + (1 - rho_r) kl_law(mean kept KL)
    (the law applied to the kept part; rho_r the realized share), the direct version with the kept positions' own
    agreement, and the cost per * ((1 - rho_r) f + rho_r (f_pre[l] + 1)) (the last layer = decide-after, f_pre = f).
    The memory interaction (aborted positions improve later positions' memory) is not modeled. Returns
    {'rows': [...], 'best': the highest predicted agreement with cost <= budget}."""
    import torch

    c = torch.as_tensor(counts, dtype=torch.float64)
    k_ = torch.as_tensor(kl, dtype=torch.float64).flatten()
    a_ = torch.as_tensor(agree, dtype=torch.float64).flatten()
    pref = c.cumsum(-1)
    L = c.shape[-1]
    rows, best = [], None
    for dep in depths:
        x = pref[:, dep]
        fp = f if dep == L - 1 else float(f_pre[dep])
        for rho in rhos:
            kap = float(torch.quantile(x, 1 - rho))
            ab = x > kap
            rr = float(ab.float().mean())
            kept = ~ab
            klk = float(k_[kept].mean()) if bool(kept.any()) else 0.0
            pred = rr * a_native + (1 - rr) * kl_law(klk, *law)
            direct = rr * a_native + (1 - rr) * (float(a_[kept].mean()) if bool(kept.any()) else 0.0)
            cost = per * ((1 - rr) * f + rr * (fp + 1))
            row = {"depth": int(dep), "rho": float(rho), "rho_realized": rr, "kappa": kap, "kl_kept": klk,
                   "predicted": pred, "direct": direct, "cost": cost}
            rows.append(row)
            if cost <= budget and (best is None or pred > best["predicted"]):
                best = row
    return {"rows": rows, "best": best, "a_native": a_native, "law": list(law), "budget": budget}


# --------------------------------------------------------------------------- prior-art baselines, layer choice, tail proxy, readout KL, mass key (Part AH)


def cats_scores(g):
    """CATS key (Lee et al. 2024, arXiv 2404.08763): |silu(a_j)| per neuron, fp32 (g = silu(W_gate n))."""
    return g.float().abs()


def gate_key_scores(g, g_tok, u_tok, weights):
    """Gate-key |g - g_tok| |u_tok| w: base-aware in the gate, the token table's u, psi-weighted; needs no up GEMV."""
    return (g.float() - g_tok.float()).abs() * u_tok.float().abs() * weights.float()


def zero_base_scores(g, u, weights):
    """|g u| w: the fixed-count error-optimal key when dropped neurons write zero (the CATS base)."""
    return (g.float() * u.float()).abs() * weights.float()


def zero_base_products(g, u, keep):
    """Products of a zero-base code: g u on ``keep``, 0 elsewhere."""
    import torch

    gu = g * u
    return torch.where(keep, gu, torch.zeros_like(gu))


def magnitude_mask(x, theta):
    """TEAL-style input sparsification (Liu et al. 2025, arXiv 2408.14690): entries with |x| <= theta set to zero.
    theta broadcasts (one value per layer and matrix)."""
    import torch

    return torch.where(x.float().abs() > theta, x, torch.zeros_like(x))


def below_threshold_mass(scores, theta):
    """Per-position sum of squared scores at or below theta: the code's own diagonal estimate of the error energy it
    drops (the gate key of ADVISOR section 15.6). scores [..., I]."""
    s = scores.float()
    return (s * s * (s <= theta)).sum(-1)


def layer_choice_frontier(err_a, err_b, t_a, t_b):
    """Greedy frontier of a per-layer choice between two codes (a 28-item multiple-choice knapsack with two items per
    layer; additive error and time). Starts from code a everywhere and swaps layers to code b in the order that saves the
    most error per unit time added: first every layer where b is no slower and no worse, then the rest by
    (err_a - err_b) / (t_b - t_a) descending; layers where b is worse and no faster are never swapped.
    err_*, t_* [L]. Returns {'order': [layer], 'rows': [{'swapped', 'layers', 'time', 'error'}]}."""
    import torch

    ea, eb = torch.as_tensor(err_a, dtype=torch.float64), torch.as_tensor(err_b, dtype=torch.float64)
    ta, tb = torch.as_tensor(t_a, dtype=torch.float64), torch.as_tensor(t_b, dtype=torch.float64)
    de, dt = ea - eb, tb - ta
    free = [int(i) for i in torch.nonzero((dt <= 0) & (de >= 0)).flatten()]
    rest = [int(i) for i in torch.nonzero((dt > 0) & (de > 0)).flatten()]
    rest.sort(key=lambda i: -float(de[i] / dt[i]))
    order = free + rest
    rows, cur = [], []
    time_, err = float(ta.sum()), float(ea.sum())
    rows.append({"swapped": 0, "layers": [], "time": time_, "error": err})
    for i in order:
        cur.append(i)
        time_ += float(dt[i])
        err -= float(de[i])
        rows.append({"swapped": len(cur), "layers": list(cur), "time": time_, "error": err})
    return {"order": order, "rows": rows}


def tail_proxy_maps(second_moment, metric, w_down, r, *, floor=1e-10, isotropic=False, dtype=None):
    """Output-side rank-r proxy of an exact-selection code's skipped tail write y = W_down t (t: the dropped neurons'
    p - b). Q_r = top-r eigenvectors of M^{1/2} S M^{1/2} (S the tail write's second moment, M a PSD metric, d x d);
    E = Q_r^T M^{1/2} W_down [r, I] applied to t, D = M^{-1/2} Q_r [d, r] (pseudo-inverse below ``floor`` x the top
    eigenvalue of M), so the proxy write D E t is the M-orthogonal projection of y onto the captured subspace.
    ``isotropic`` (Part AI): M = I without its eigendecomposition (``metric`` ignored; the PCA of S); ``dtype`` the
    eigensolver's precision (default float64; float32 is ~10x faster on consumer GPUs for d ~ 1,536).
    Returns (E float32, D float32, captured share of the metric energy)."""
    import torch

    dt = torch.float64 if dtype is None else dtype
    if isotropic:
        S = ((second_moment + second_moment.T) / 2).to(dt)
        ev, Q = torch.linalg.eigh(S)
        ev, Q = ev.flip(0).clamp_min(0), Q.flip(1)
        Qr = Q[:, :r]
        tot = float(ev.sum())
        return ((Qr.T @ w_down.to(dt)).float().contiguous(), Qr.float().contiguous(),
                (float(ev[:r].sum()) / tot if tot > 0 else float("nan")))
    Mv, MQ = torch.linalg.eigh(((metric + metric.T) / 2).double())
    Mv = Mv.clamp_min(0)
    keep = Mv > floor * float(Mv.max())
    Mh = (MQ * Mv.sqrt()[None, :]) @ MQ.T
    Mih = (MQ * torch.where(keep, Mv.clamp_min(1e-300).rsqrt(), torch.zeros_like(Mv))[None, :]) @ MQ.T
    S = ((second_moment + second_moment.T) / 2).double()
    ev, Q = torch.linalg.eigh(Mh @ S @ Mh)
    ev, Q = ev.flip(0).clamp_min(0), Q.flip(1)
    Qr = Q[:, :r]
    E = Qr.T @ Mh @ w_down.double()
    D = Mih @ Qr
    tot = float(ev.sum())
    return E.float().contiguous(), D.float().contiguous(), (float(ev[:r].sum()) / tot if tot > 0 else float("nan"))


def bf16_tie_mask(logits_bf16):
    """Positions whose bf16 top-2 logits are within one bf16 ulp of the top logit (the served argmax is a tie-break).
    logits_bf16 [P, V] (bf16 values, any dtype)."""
    import torch

    t2 = logits_bf16.float().topk(2, dim=-1).values
    top = t2[:, 0].abs().clamp_min(1e-30)
    ulp = torch.exp2(torch.floor(torch.log2(top)) - 7)
    return (t2[:, 0] - t2[:, 1]) <= ulp


def readout_kl_stats(logits_ref, logits_var):
    """Per-position KL(p_ref || p_var) and top-1 agreement of a readout variant against a reference (fp32).
    logits_* [P, V]. Returns (kl [P], agree [P] bool)."""
    lr = logits_ref.float().log_softmax(-1)
    lv = logits_var.float().log_softmax(-1)
    kl = (lr.exp() * (lr - lv)).sum(-1)
    return kl, lr.argmax(-1) == lv.argmax(-1)


def sigma_candidates(lhat, h_perp, e_perp, margin, dim_perp):
    """sigma-margin candidate set: the Cauchy-Schwarz bound of ``certified_candidates`` scaled by
    margin / sqrt(dim_perp), the sd of e_perp . h_perp if the complement directions were isotropic (an assumption: it
    fails for the argmax token, whose e is aligned with h, so the recall must be measured). A union bound over V gives
    a nominal failure rate V Phibar(margin). Returns (mask [P, V], sizes [P])."""
    scale = float(margin) / float(dim_perp) ** 0.5
    top = lhat.max(-1, keepdim=True).values
    thr = top - scale * h_perp[:, None] * (e_perp[None, :] + e_perp.max())
    mask = lhat >= thr
    return mask, mask.sum(-1)


def key_gate_frontier(key, kl, agree, f, rhos, *, a_native=0.925, law=(0.967, 0.403), repair=0.0, excess=0.0, per=1000,
                      needs_readout=False, readout=0.0):
    """Offline decide-after gate on an arbitrary per-position key (for example the below-threshold score mass): the
    top-rho share by key goes native. Predicted agreement rho a_native + (1 - rho) (kl_law(KL_kept (1 - repair rho))
    + excess): the law on the kept positions' mean KL, scaled by a memory-repair factor linear in rho (natives in the
    memory lower the served positions' KL; AG measured x0.54 at rho 0.47, i.e. repair 0.98), plus the stream's own law
    excess; the direct version uses the kept positions' own agreement. Cost per (f + rho) (decide-after); with
    ``needs_readout`` (a key computed from the served logits) also the honest cost per (f + rho (1 + readout)) as
    ``cost_honest`` (Part AI). Returns {'rows': [...], 'spearman_key_kl'}."""
    import torch

    k_ = torch.as_tensor(key, dtype=torch.float64).flatten()
    kl_ = torch.as_tensor(kl, dtype=torch.float64).flatten()
    a_ = torch.as_tensor(agree, dtype=torch.float64).flatten()
    rows = []
    for rho in rhos:
        kap = float(torch.quantile(k_, 1 - rho)) if rho > 0 else float("inf")
        ab = k_ > kap
        rr = float(ab.float().mean())
        kept = ~ab
        klk = float(kl_[kept].mean()) if bool(kept.any()) else 0.0
        pred = rr * a_native + (1 - rr) * (kl_law(klk * max(1 - repair * rr, 0.0), *law) + excess)
        direct = rr * a_native + (1 - rr) * (float(a_[kept].mean()) if bool(kept.any()) else 0.0)
        rows.append({"rho": float(rho), "rho_realized": rr, "kappa": kap, "kl_kept": klk, "predicted": pred, "direct": direct,
                     "cost": per * (f + rr),
                     "cost_honest": column_equivalents_decide_after_honest(f, rr, readout, needs_readout=needs_readout,
                                                                           per=per)})
    return {"rows": rows, "spearman_key_kl": spearman(k_, kl_, ties=True), "a_native": a_native, "law": list(law),
            "repair": repair, "excess": excess, "needs_readout": bool(needs_readout)}


# --------------------------------------------------------------------------- Part AI: WINA, R-Sparse, TEAL greedy


def column_norm_weights(w_down):
    """WINA's weight (Chen et al. 2025, arXiv 2505.19427) at the down projection: ||W_down[:, j]||_2 per neuron.
    w_down [d, I] -> [I] fp32. The WINA criterion of input j is |p_j| ||w_j||_2 (``zero_base_scores(g, u, w)``); the
    base-aware form at the hybrid base is ``hybrid_error_scores(..., w)``."""
    return w_down.float().norm(dim=0)


def logit_gain_dispersion(psi, wnorm):
    """Weights-only split of the psi key's weight into the column norm and the column's logit gain rho_j = psi_j /
    ||w_j||: sd(log rho), sd(log ||w||), sd(log psi) and Spearman(psi, ||w||) for one layer. A toy (ADVISOR 16.2) puts
    the KL ratio between the l2 and psi keys at x1.07 / 1.3 / 1.85 / 3.1 at sd(log rho) 0.2 / 0.4 / 0.6 / 0.8."""
    import torch

    p = psi.double().clamp_min(1e-30)
    w = wnorm.double().clamp_min(1e-30)
    lr = (p / w).log()
    return {"sd_log_rho": float(lr.std()), "sd_log_wnorm": float(w.log().std()), "sd_log_psi": float(p.log().std()),
            "spearman_psi_wnorm": spearman(p, w, ties=True), "mean_log_rho": float(lr.mean()),
            "corr_log_rho_log_wnorm": float(torch.corrcoef(torch.stack([lr, w.log()]))[0, 1])}


def svd_remainder_maps(w_down, r, *, dtype=None):
    """R-Sparse's data-free remainder (Zhang et al. 2025, arXiv 2504.19449) at the down projection: the rank-r truncated
    SVD W_down ~ U_r S_r V_r^T applied to the skipped inputs t, written as E = S_r V_r^T = U_r^T W_down [r, I] and
    D = U_r [d, r] (the remainder D E t is the projection of the skipped write onto the top-r left singular subspace).
    Computed from the d x d Gram (eigh, in ``dtype``, default float64), not a d x I SVD. Returns (E float32, D float32,
    captured Frobenius share)."""
    import torch

    W = w_down.to(torch.float64 if dtype is None else dtype)
    ev, Q = torch.linalg.eigh(W @ W.T)
    ev, Q = ev.flip(0).clamp_min(0), Q.flip(1)
    U = Q[:, :r]
    E = U.T @ W
    tot = float(ev.sum())
    return E.float().contiguous(), U.float().contiguous(), (float(ev[:r].sum()) / tot if tot > 0 else float("nan"))


TEAL_MATRICES = ("q", "k", "v", "o", "gate", "up", "down")


def teal_footprints(shapes):
    """Weight footprints (parameters) of the seven projections of one block, TEAL's allocation weights."""
    d, inter = shapes["d"], shapes["inter"]
    qo = d * shapes["n_heads"] * shapes["head_dim"]
    kv = d * shapes["n_kv"] * shapes["head_dim"]
    return {"q": qo, "k": kv, "v": kv, "o": qo, "gate": d * inter, "up": d * inter, "down": d * inter}


def teal_greedy_allocation(block_error, footprints, target, *, alpha=0.05, cap=0.95):
    """TEAL's block-wise greedy sparsity allocation (Liu et al. 2025, arXiv 2408.14690, Algorithm 1): every matrix starts
    at sparsity 0; each step tentatively raises matrix i by delta_i = alpha F / f_i (footprint-inverse, so one step adds
    alpha to the footprint-weighted block sparsity P = sum p_i f_i / F), evaluates ``block_error(p)`` (the l2 error of
    the block output against the dense block on calibration inputs) and keeps the matrix with the lowest error. Sparsity
    per matrix is capped at ``cap`` (the paper does not state one); the last step is scaled to land exactly on
    ``target``. block_error: callable {name: sparsity} -> float. Returns {'sparsity': {name: p}, 'P', 'path', 'evals'}."""
    names = list(footprints)
    F = float(sum(footprints.values()))
    p = {n_: 0.0 for n_ in names}
    path, evals = [], 0
    P = 0.0
    while P < target - 1e-12:
        best = None
        need = target - P
        for n_ in names:
            if p[n_] >= cap - 1e-12:
                continue
            dlt = min(alpha * F / footprints[n_], cap - p[n_], need * F / footprints[n_])
            q = dict(p)
            q[n_] = p[n_] + dlt
            e = float(block_error(q))
            evals += 1
            if best is None or e < best[0]:
                best = (e, n_, dlt)
        if best is None:
            break
        p[best[1]] += best[2]
        P = sum(p[n_] * footprints[n_] for n_ in names) / F
        path.append({"matrix": best[1], "error": best[0], "P": P})
    return {"sparsity": p, "P": P, "path": path, "evals": evals, "alpha": alpha, "cap": cap, "target": target}


def gate_exact_up_proxy_scores(g, n, n_tok, encode, dec_up, u_tok, g_tok, g_glob, weights):
    """G2U key (ADVISOR 16.3): the exact gate g = act(W_gate n) and a rank-k proxy of the up projection,
    u_hat = u_tok + ((n - n_tok) encode) dec_up^T (encode [d, k], dec_up [I, k]), in the hybrid base's base-aware score
    |g u_hat - g_tok u_tok - g_glob (u_hat - u_tok)| w. The selected neurons then take exact up and down rows, so the
    proxy only ranks. Returns (scores [..., I], u_hat [..., I])."""
    z = (n.float() - n_tok.float()) @ encode.float()
    u_hat = u_tok.float() + z @ dec_up.float().T
    return hybrid_error_scores(g, u_hat, g_tok, u_tok, g_glob, weights), u_hat


# --------------------------------------------------------------------------- Part AJ: pool codes, base variants, diagnostics


def up_reading_metric(w_up, weights, e2):
    """The up half of ``reading_metric``: M_u = W_up^T diag(w^2 e2) W_up (d x d, fp32), e2 a per-neuron second moment.
    A rank-k up proxy whose score multiplies u_hat by the exact gate g (G2U) errs in the selection by
    (g_j - g_glob,j) (u_hat_j - u_j) w_j under the hybrid base (the base's g_glob (u_hat - u_tok) term cancels part of
    it), so e2 = E[(g - g_glob)^2] is the selection's metric and E[g^2] the zero-base one; this is the metric the
    reduced-rank map should minimise (``metric_proxy(up_reading_metric(...), S_n, k)``)."""
    wu = w_up.float() * (weights.float() * e2.float().clamp_min(0).sqrt())[:, None]
    return wu.T @ wu


def pool_rescore_mask(proxy_scores, theta_pool, exact_scores, theta):
    """Pool-rescored selection (G2U-P, ADVISOR 17.3): the proxied score admits a candidate pool (score > theta_pool,
    calibrated to pool x the final count), the exact score is recomputed on the pool only, and the code's own threshold
    selects the final set inside it. Returns (keep, pool) boolean masks [..., I]; exact_scores need only be valid on
    the pool."""
    import torch

    pool = proxy_scores.float() > torch.as_tensor(theta_pool, dtype=torch.float32, device=proxy_scores.device)
    keep = pool & (exact_scores.float() > torch.as_tensor(theta, dtype=torch.float32, device=exact_scores.device))
    return keep, pool


def pool_energy_recall(exact_scores, exact_keep, pool):
    """Per-position energy recall of an exact set by a candidate pool: sum_{j in exact & pool} s_j^2 / sum_{j in exact}
    s_j^2 (s the exact score, in Fisher / psi units, so s^2 is the energy the exact selection keeps), and the count
    recall |exact & pool| / |exact|. Positions with an empty exact set count as 1. Returns (energy [...], count [...])."""
    s2 = exact_scores.float() ** 2
    e_all = (s2 * exact_keep).sum(-1)
    e_in = (s2 * (exact_keep & pool)).sum(-1)
    n_all = exact_keep.sum(-1).float()
    n_in = (exact_keep & pool).sum(-1).float()
    one = e_all.new_ones(())
    return (e_in / e_all.clamp_min(1e-30)).where(e_all > 0, one), (n_in / n_all.clamp_min(1)).where(n_all > 0, one)


def adjacent_set_jaccard(keep, valid):
    """Temporal reuse diagnostic: the Jaccard of the selected sets at adjacent positions t-1, t of the same sequence
    (keep [B, T, I], valid [B, T] marks positions that may pair), and a random-pair baseline at the same density (the
    set at t against the set at t in the next sequence of the batch, batch rolled by one; needs B >= 2). Returns
    (sum of adjacent Jaccards, adjacent pairs, sum of random-pair Jaccards, random pairs)."""
    import torch

    va = valid[:, 1:] & valid[:, :-1]
    ja = set_jaccard(keep[:, 1:], keep[:, :-1])
    out = [float(ja[va].sum()), int(va.sum())]
    if keep.shape[0] >= 2:
        kr = torch.roll(keep, 1, 0)
        vr = valid & torch.roll(valid, 1, 0)
        jr = set_jaccard(keep, kr)
        out += [float(jr[vr].sum()), int(vr.sum())]
    else:
        out += [0.0, 0]
    return tuple(out)


class TokenTable:
    """Token-conditional mean of a per-position vector (for example one layer's attention write), fit on one set of
    positions and scored on another: ``add(rows, y)`` pools y [N, d] into per-row sums (rows [N] in 0..R-1, -1
    skipped) and a global sum; ``r2_terms(rows, y, min_count)`` returns (sum ||y - ybar_row||^2, sum ||y - ybar||^2,
    n) over positions whose row has >= min_count fit positions, so the held-out R^2 of the table is 1 - a / b."""

    def __init__(self, n_rows, dim, device=None):
        import torch

        self.sum = torch.zeros(n_rows, dim, device=device)
        self.cnt = torch.zeros(n_rows, device=device)
        self.gsum = torch.zeros(dim, dtype=torch.float64, device=device)
        self.gn = 0

    def add(self, rows, y):
        import torch

        ok = rows >= 0
        yf = y.float()
        self.sum.index_add_(0, rows[ok], yf[ok])
        self.cnt.index_add_(0, rows[ok], torch.ones(int(ok.sum()), device=yf.device))
        self.gsum += yf.double().sum(0)
        self.gn += int(y.shape[0])

    def r2_terms(self, rows, y, min_count=2):
        ok = (rows >= 0)
        ok = ok & (self.cnt[rows.clamp_min(0)] >= min_count)
        yf = y.float()[ok]
        r = rows[ok]
        mu = self.sum[r] / self.cnt[r][:, None]
        g = (self.gsum / max(self.gn, 1)).float()
        return float(((yf - mu) ** 2).sum()), float(((yf - g[None, :]) ** 2).sum()), int(ok.sum())


def lowrank_b0_maps(w_down, g_glob, w_up, metric, r, *, second_moment=None, floor=1e-10):
    """Rank-r factorisation of the hybrid base's global term B_0 = W_down diag(g_glob) W_up in an output metric M
    (for example the centred-head psi metric diag(gamma_f) G_U diag(gamma_f)): B_0 ~ D_r E_r.

    Without ``second_moment`` the inputs are taken as isotropic: E_r = V_r^T, the top-r eigenvectors of B_0^T M B_0,
    and D_r = B_0 V_r (for orthonormal E_r this decoder is optimal in every PSD metric). With ``second_moment``
    S = E[dz dz^T] of the inputs the map is applied to (dz = n - n_tok), the factorisation minimises
    E || M^{1/2} (B_0 - D E) dz ||^2 (reduced-rank regression): with S = L L^T and W_r the top-r eigenvectors of
    L^T B_0^T M B_0 L, E_r = W_r^T L^+ and D_r = B_0 L W_r. Part AJ's smoke measured why the input weighting is needed:
    the isotropic rank-256 map kept 63% of tr(B_0^T M B_0) yet served at 38.9 closure (the token-only base: 98.4),
    because the residual's deviations live in few, heavy directions that an input-blind basis discards. Eigenvalues of
    S below ``floor`` x its largest are dropped from L^+. Returns (E [r, d] fp32, D [d, r] fp32, the captured share of
    tr(B_0^T M B_0) or, with S, of E || M^{1/2} B_0 dz ||^2)."""
    import torch

    B = w_down.float() @ (g_glob.float()[:, None] * w_up.float())
    if second_moment is None:
        K = B.T @ metric.float() @ B
        ev, Q = torch.linalg.eigh((K + K.T) / 2)
        ev, Q = ev.flip(0).clamp_min(0), Q.flip(1)
        V = Q[:, :r]
        tot = float(ev.sum())
        return V.T.contiguous(), (B @ V).contiguous(), (float(ev[:r].sum()) / tot if tot > 0 else float("nan"))
    S = second_moment.float()
    s, Qs = torch.linalg.eigh((S + S.T) / 2)
    keep = s > float(s.max()) * floor
    s, Qs = s[keep], Qs[:, keep]
    L = Qs * s.sqrt()[None, :]
    K = L.T @ (B.T @ metric.float() @ B) @ L
    ev, W = torch.linalg.eigh((K + K.T) / 2)
    ev, W = ev.flip(0).clamp_min(0), W.flip(1)
    Wr = W[:, :r]
    E = (Wr.T / s.sqrt()[None, :]) @ Qs.T
    tot = float(ev.sum())
    return E.contiguous(), (B @ L @ Wr).contiguous(), (float(ev[:r].sum()) / tot if tot > 0 else float("nan"))


# --------------------------------------------------------------------------- Part AK: posterior pool key, first-order base, frontier laws, readout certificate


def proxy_residual_sigma(w, dec, encode, second_moment, *, chunk=4096):
    """Per-neuron scale of what a rank-k proxy does not see (ADVISOR 18.1 (a)). A proxy reads z = dz encode
    (encode [d, k]) and predicts the projection w_j dz of row j of W (W_up or W_gate, [I, d]) as dec_j . z (dec [I, k]);
    the unseen part is r_j = v_j . dz with v_j = w_j - encode dec_j^T, so Var(r_j) = v_j^T S v_j exactly for
    S = E[dz dz^T] (no distributional assumption). Returns (sigma [I] = sqrt(v_j^T S v_j), the row norms ||v_j|| [I]),
    fp32, computed in row chunks."""
    import torch

    S = second_moment.float()
    enc = encode.float()
    sig, nrm = [], []
    for r0 in range(0, w.shape[0], chunk):
        v = w[r0:r0 + chunk].float() - dec[r0:r0 + chunk].float() @ enc.T
        sig.append(((v @ S) * v).sum(-1).clamp_min(0).sqrt())
        nrm.append(v.norm(dim=-1))
    return torch.cat(sig), torch.cat(nrm)


def posterior_pool_key(proxy_scores, theta, g, g_ref, weights, sigma, *, scale=None, floor=1e-12):
    """The z-key of a one-side proxy pool (G2U-Pz, ADVISOR 18.3 route 1): the exact score is
    |s_hat_j + (g_j - g_ref_j) w_j r_j| with r_j the up projection's unseen part, so its error scale given the proxy is
    tau_j = |g_j - g_ref_j| w_j sigma_j (``proxy_residual_sigma``; ``scale`` [...] multiplies tau per position, the
    position-conditional variant passes ||P_perp dz|| / sqrt(d - k) with sigma = ||v_j||). Ranking a fixed-size pool
    by z_j = (|s_hat_j| - theta) / tau_j keeps the neurons most likely to clear the exact threshold theta under a
    Gaussian r_j (the Neyman-Pearson set of the dominant branch). Returns z [..., I] (fp32; +-inf-free: tau is floored)."""
    import torch

    tau = (g.float() - g_ref.float()).abs() * weights.float() * sigma.float()
    if scale is not None:
        tau = tau * scale.float()[..., None]
    th = torch.as_tensor(theta, dtype=torch.float32, device=proxy_scores.device)
    return (proxy_scores.float() - th) / tau.clamp_min(floor)


def two_side_pool_key(proxy_scores, theta, a_hat, u_hat, g_ref, weights, sigma_a, sigma_u, *, floor=1e-12):
    """The z-key of a two-side proxy pool (K1-Pz, ADVISOR 18.3 route 5): both a and u are rank-k proxies, and to first
    order the exact score moves by w_j (act'(a_hat) u_hat r^a_j + (act(a_hat) - g_ref) r^u_j), so
    tau_j^2 = w_j^2 ((silu'(a_hat_j) u_hat_j sigma^a_j)^2 + ((silu(a_hat_j) - g_ref_j) sigma^u_j)^2) (cross term
    dropped). Returns z [..., I]."""
    import torch

    th = torch.as_tensor(theta, dtype=torch.float32, device=proxy_scores.device)
    a = a_hat.float()
    ta = silu_prime(a) * u_hat.float() * sigma_a.float()
    tu = (torch.nn.functional.silu(a) - g_ref.float()) * sigma_u.float()
    tau = weights.float() * (ta * ta + tu * tu).sqrt()
    return (proxy_scores.float() - th) / tau.clamp_min(floor)


def orth_residual_norm(dz, z, encode):
    """||P_perp dz|| for each row of dz [..., d] given z = dz encode [..., k]: the part of dz outside the span of the
    encoder's columns, ||dz||^2 - z^T (encode^T encode)^{-1} z (k^2 work per row, no extra d x k GEMV)."""
    import torch

    enc = encode.double()
    Gi = torch.linalg.inv(enc.T @ enc)
    zz = z.double()
    proj = ((zz @ Gi) * zz).sum(-1)
    return (dz.double().pow(2).sum(-1) - proj).clamp_min(0).sqrt().float()


def first_order_base_matrix(w_down, w_gate, w_up, c, d):
    """B_01 = W_down [diag(d) W_up + diag(c) W_gate] (ADVISOR 18.3 route 3): the global term of the first-order-complete
    base b_j = g_tok,j u_tok,j + d_j du_j + c_j da_j, whose write is y_tok + B_01 dz. With c = 0, d = g_glob it is
    ``global_base_matrix``. fp32 [d_model, d_model]."""
    return w_down.float() @ (d.float()[:, None] * w_up.float() + c.float()[:, None] * w_gate.float())


def first_order_base_scores(g, u, a, g_tok, u_tok, a_tok, c, d, weights):
    """The base-aware score of the first-order-complete base: |g u - g_tok u_tok - d (u - u_tok) - c (a - a_tok)| w,
    a the gate pre-activation and a_tok its token-table value. With c = 0 and d = g_glob it is ``hybrid_error_scores``."""
    gf, uf = g.float(), u.float()
    ut = u_tok.float()
    base = g_tok.float() * ut + d.float() * (uf - ut) + c.float() * (a.float() - a_tok.float())
    return (gf * uf - base).abs() * weights.float()


def first_order_base_taylor(a_tok, u_tok, counts, g_glob):
    """Taylor constants of the first-order-complete base: c_j = E_tok[silu'(a_tok,j) u_tok,j] (token rows weighted by
    their fit counts) and d_j = g_glob,j. a_tok, u_tok [R, I] token-table pre-activations and up values, counts [R].
    Returns (c [I], d [I]) fp32."""
    w = counts.double() / counts.double().sum().clamp_min(1e-30)
    c = (w[:, None] * (silu_prime(a_tok.float()).double() * u_tok.double())).sum(0)
    return c.float(), g_glob.float().clone()


class FirstOrderMoments:
    """Five per-neuron moments of the base's residual rho_j = p_j - g_tok,j u_tok,j against (da_j, du_j) (sums over
    positions of da^2, du^2, da du, rho da, rho du) for the per-neuron 2 x 2 least squares of the first-order-complete
    base (B01-LS, ADVISOR 18.3 route 3)."""

    def __init__(self, n_layers, inter, device=None):
        import torch

        self.m = torch.zeros(n_layers, 5, inter, dtype=torch.float64, device=device)
        self.n = [0] * n_layers

    def add(self, layer, rho, da, du, weight=None):
        """Accumulate positions [..., I]; ``weight`` [..., I] (optional) weights each (position, neuron) term, e.g. a
        tail mask so the fit minimises the residual only where the neuron is left to the base."""
        r, a, u = rho.double().reshape(-1, rho.shape[-1]), da.double().reshape(-1, da.shape[-1]), du.double().reshape(-1, du.shape[-1])
        w = 1.0 if weight is None else weight.double().reshape(-1, rho.shape[-1])
        self.m[layer, 0] += (w * a * a).sum(0)
        self.m[layer, 1] += (w * u * u).sum(0)
        self.m[layer, 2] += (w * a * u).sum(0)
        self.m[layer, 3] += (w * r * a).sum(0)
        self.m[layer, 4] += (w * r * u).sum(0)
        self.n[layer] += r.shape[0]

    def solve(self, layer, *, ridge=1e-6, fallback=None):
        """(c [I], d [I]) fp32 minimising E (rho - c da - d du)^2 per neuron, a ridge of ``ridge`` x the mean diagonal;
        neurons whose 2 x 2 system is singular take ``fallback`` = (c, d) (or zeros)."""
        import torch

        saa, suu, sau, sra, sru = self.m[layer]
        lam = ridge * (saa + suu) / 2
        A, B, C = saa + lam, sau, suu + lam
        det = A * C - B * B
        ok = det > 1e-12 * (A * C).clamp_min(1e-300)
        c = torch.where(ok, (C * sra - B * sru) / det.where(ok, torch.ones_like(det)), torch.zeros_like(det))
        d = torch.where(ok, (A * sru - B * sra) / det.where(ok, torch.ones_like(det)), torch.zeros_like(det))
        if fallback is not None:
            c = torch.where(ok, c, fallback[0].double())
            d = torch.where(ok, d, fallback[1].double())
        return c.float(), d.float()


def tail_residual_components(g, u, a, g_tok, u_tok, a_tok, g_glob, keep):
    """Decomposition of the hybrid base's per-neuron residual on the tail (ADVISOR 18.3, AK held-out): p - b_H =
    (g - g_tok) u_tok + (g - g_glob) du splits into the gate-linear part silu'(a_tok) da u_tok, the up-linear part beyond
    g_glob (g_tok - g_glob) du, and the remainder (the rest: the gate's curvature and the da du product). Entries on the
    selected set ``keep`` are zeroed. Returns dict of [..., I] fp32: total, gate_linear, up_linear, remainder."""
    gf, uf, af = g.float(), u.float(), a.float()
    gt, ut, at = g_tok.float(), u_tok.float(), a_tok.float()
    du, da = uf - ut, af - at
    tot = gf * uf - (gt * ut + g_glob.float() * du)
    gl = silu_prime(at) * da * ut
    ul = (gt - g_glob.float()) * du
    m = (~keep).float()
    return {"total": tot * m, "gate_linear": gl * m, "up_linear": ul * m, "remainder": (tot - gl - ul) * m}


def paired_chunk_bootstrap(a, b, chunk, *, draws=2000, seed=0):
    """Paired chunk bootstrap of mean(a) - mean(b) over positions (ADVISOR 18.2): a, b [N] per-position outcomes of two
    codes on the same positions (agreement bits, or a chord's per-position mixture), chunk [N] chunk ids; chunks are
    resampled with replacement ``draws`` times. Returns {delta, se, z, ci95, chunks, positions, discordance}."""
    import numpy as np

    a, b, ch = np.asarray(a, np.float64), np.asarray(b, np.float64), np.asarray(chunk)
    ids, inv = np.unique(ch, return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    sd = np.bincount(inv, weights=a - b)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(draws, len(ids)))
    boots = sd[pick].sum(1) / n[pick].sum(1)
    delta = float((a - b).mean())
    se = float(boots.std(ddof=1))
    return {"delta": delta, "se": se, "z": (delta / se if se > 0 else float("nan")),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "chunks": int(len(ids)),
            "positions": int(len(a)), "discordance": float(np.mean(np.abs(a - b) > 0))}


def chord_mixture(a_lo, a_hi, c_lo, c_hi, c):
    """The chord between two codes (cost c_lo, c_hi; per-position outcomes a_lo, a_hi) at cost c: the per-position
    mixture (1 - lam) a_lo + lam a_hi with lam = (c - c_lo) / (c_hi - c_lo), the paired yardstick of a code at c (its
    mean is the chord's value; outside [c_lo, c_hi] the chord is an extrapolation). Returns (mixture [N], lam)."""
    import numpy as np

    lam = (float(c) - float(c_lo)) / (float(c_hi) - float(c_lo))
    return (1 - lam) * np.asarray(a_lo, np.float64) + lam * np.asarray(a_hi, np.float64), lam


def chord_paired_bootstrap(a, a_lo, a_hi, c_lo, c_hi, c, chunk, *, draws=2000, seed=0):
    """A code (per-position outcomes a, cost c) against the chord between two codes at its cost: the paired chunk
    bootstrap of a against the deterministic ``chord_mixture`` (Part AM custody: a random per-position draw between the
    two codes adds sampling noise to delta and collapses to one endpoint when lam is outside [0, 1]). Returns the
    ``paired_chunk_bootstrap`` dict plus {lam, frontier (the chord's value at c), extrapolated (lam outside [0, 1])}."""
    mix, lam = chord_mixture(a_lo, a_hi, c_lo, c_hi, c)
    r = paired_chunk_bootstrap(a, mix, chunk, draws=draws, seed=seed)
    r.update(lam=lam, frontier=float(mix.mean()), extrapolated=bool(lam < 0 or lam > 1))
    return r


def exact_family_law(m, *, c0=758.0, kappa=0.0326, kl0=0.0524, m0=2254.0, alpha=-2.43, e0=0.0098, e1=5.4e-6, a0=0.967,
                     beta=0.403):
    """Frontier law of the exact-score family (ADVISOR 18.2, Part AJ fits): cost c = c0 + kappa m column-equivalents per
    1k, KL = kl0 (m / m0)^alpha, agreement = a0 - beta sqrt(KL) + e0 + e1 (m - m0). Returns (c, KL, agreement); outside
    the law's domain (m <= 0, i.e. c <= c0) KL and agreement are NaN."""
    if float(m) <= 0:
        return c0 + kappa * float(m), float("nan"), float("nan")
    kl = kl0 * (float(m) / m0) ** alpha
    return c0 + kappa * float(m), kl, a0 - beta * kl ** 0.5 + e0 + e1 * (float(m) - m0)


def exact_family_at_cost(c, **kw):
    """The exact family's agreement at cost c (inverts c = c0 + kappa m). Returns (m, KL, agreement)."""
    c0, kappa = kw.get("c0", 758.0), kw.get("kappa", 0.0326)
    m = (float(c) - c0) / kappa
    _, kl, a = exact_family_law(m, **kw)
    return m, kl, a


def pool_family_law(m, rho, floor, *, c0=524.0, k_pool=0.065, kappa=0.0326, excess=0.008, kl_kw=None, a0=0.967,
                    beta=0.403):
    """Frontier law of the one-side pool family (ADVISOR 18.2): c = c0 + k_pool m + kappa (rho - 1) m, KL = KL_X(m) +
    floor (the measured proxy floor at pool ratio rho), agreement = a0 - beta sqrt(KL) + excess. Returns (c, KL, A)."""
    kw = dict(kl_kw or {})
    _, klx, _ = exact_family_law(m, **kw)
    kl = klx + float(floor)
    return c0 + k_pool * float(m) + kappa * (float(rho) - 1) * float(m), kl, a0 - beta * kl ** 0.5 + excess


def sigma_margin_statistic(lhat, target, h_perp, e_perp, dim_perp):
    """The certification statistic of a sigma-margin candidate set (ADVISOR 18.6): for the true argmax v* of each row,
    z* = (lhat_{v*} - lhat_top) / (h_perp (e_perp,v* + max e_perp) / sqrt(dim_perp)), the (non-positive) margin in sigma
    units by which v* sits below the approximate top; v* is in the m-sigma set iff z* >= -m. Returns z* [P] fp32."""
    import torch

    top = lhat.max(-1).values
    lv = lhat.gather(-1, target[:, None])[:, 0]
    s = h_perp * (e_perp[target] + e_perp.max()) / float(dim_perp) ** 0.5
    return ((lv - top) / s.clamp_min(1e-30)).float()


def gaussian_tail_fit(zstar, xs=(1.0, 2.0, 3.0, 4.0)):
    """Empirical survival P(z* < -x) at each x and the slope of log survival against x^2 over the x with a non-zero
    survival (a Gaussian tail has slope -1/2 in units of its own sd); with no tail mass the slope is None."""
    import numpy as np

    z = np.asarray(zstar, np.float64)
    surv = {float(x): float(np.mean(z < -x)) for x in xs}
    pts = [(x * x, np.log(s)) for x, s in surv.items() if s > 0]
    slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) >= 2 else None
    return {"survival": surv, "log_survival_slope_vs_x2": slope, "min": float(z.min()) if len(z) else None,
            "q001": float(np.quantile(z, 1e-3)) if len(z) else None}


def two_level_readout_bytes(vocab, d, r, candidates, *, qbits=8, scale_bytes=4, basis_bytes=2):
    """Bytes of a two-level candidate readout (ADVISOR 18.6): the rank-r basis d x r, an int (qbits) table [V, r] with
    per-row scales, then |C| exact int rows [|C|, d] with scales. Returns bytes per step."""
    tab = vocab * r * qbits / 8 + vocab * scale_bytes
    rows = float(candidates) * (d * qbits / 8 + scale_bytes)
    return d * r * basis_bytes + tab + rows


# ---- Part AL (ADVISOR 19): posterior pool statistics, the adaptive tail detector, position water-filling of the
# threshold, closed-form base tail energies, the o-projection head pool, the blocked-sigma readout, ledger tolerance


def posterior_membership(proxy_scores, theta, tau, *, floor=1e-12):
    """P(|s_j| > theta) for s_j ~ N(s_hat_j, tau_j^2) (ADVISOR 19.2, D1's denominator): Phi(z) + Phi(-(theta + s_hat) /
    tau) with z = (s_hat - theta) / tau; s_hat >= 0 is the proxied |score|. Returns p [..., I] fp32."""
    import torch

    s, t = proxy_scores.float(), tau.float().clamp_min(floor)
    th = torch.as_tensor(theta, dtype=torch.float32, device=s.device)
    return torch.special.ndtr((s - th) / t) + torch.special.ndtr(-(th + s) / t)


def posterior_energy_key(proxy_scores, theta, tau, *, floor=1e-12):
    """E[s^2 1(s > theta)] for s ~ N(s_hat, tau^2) (ADVISOR 19.2): (s_hat^2 + tau^2) Phi(z) + tau (s_hat + theta) phi(z),
    z = (s_hat - theta) / tau. The energy-optimal pool key and D2's summand. Returns [..., I] fp32 (>= 0)."""
    import math

    import torch

    s, t = proxy_scores.float(), tau.float().clamp_min(floor)
    th = torch.as_tensor(theta, dtype=torch.float32, device=s.device)
    z = (s - th) / t
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return ((s * s + t * t) * torch.special.ndtr(z) + t * (s + th) * phi).clamp_min(0)


def posterior_recall(energy_key, z, z0):
    """The posterior's own energy recall of a z-pool (ADVISOR 19.1 (a)): R(z0) = sum_j e_j 1(z_j >= z0) / sum_j e_j per
    position, e_j = ``posterior_energy_key``. z0 scalar or broadcastable. Returns [...] fp32."""
    import torch

    e = energy_key.float()
    z0 = torch.as_tensor(z0, dtype=torch.float32, device=e.device)
    return (e * (z.float() >= z0)).sum(-1) / e.sum(-1).clamp_min(1e-30)


def pool_tail_signals(z, pool, exact_scores, theta_x, proxy_scores, tau, band=None):
    """The adaptive pool's per-position tail signals (ADVISOR 19.2): D1 = realized exact members in the pool / the
    posterior-predicted members sum_{pool} p_j (> 1 when the proxy's attenuation is worse than modelled); D2 = the
    model's expected missed energy outside the pool, sum_{not pool} E[s^2 1(s > theta)]; D3 = the exact-member share in
    the pool's lowest-z band (``band`` [..., I] bool, a subset of the pool; None -> D3 omitted). Members are
    exact_scores > theta_x. Larger = the pool was too small. Returns dict of [...] fp32."""
    mem = exact_scores.float() > theta_x
    p = posterior_membership(proxy_scores, theta_x, tau)
    e = posterior_energy_key(proxy_scores, theta_x, tau)
    out = {"D1": (pool & mem).sum(-1).float() / (p * pool).sum(-1).clamp_min(1e-6),
           "D2": (e * (~pool)).sum(-1)}
    if band is not None:
        out["D3"] = (band & mem).sum(-1).float() / band.sum(-1).float().clamp_min(1)
    return out


def detector_capture(signal, gain, q):
    """Captured share f of the oracle gain (ADVISOR 19.2): per-position gain [N] (the KL or Fisher-error reduction of
    serving that position with the fallback) and a detector signal [N]; the detector acts on its top-q share of
    positions, the oracle on the top-q share by gain. Returns f = sum gain[top_q(signal)] / sum gain[top_q(gain)]."""
    import torch

    s, g = signal.flatten().float(), gain.flatten().float()
    k = max(1, int(round(float(q) * s.numel())))
    den = g.topk(k).values.sum()
    return float(g[s.topk(k).indices].sum() / den) if float(den) != 0 else float("nan")


def missed_energy(scores, theta):
    """E_t = sum_{s_j <= theta} s_j^2 per position (the selection's missed energy in score units, ADVISOR 19.4)."""
    s = scores.float()
    return (s * s * (s <= theta)).sum(-1)


def waterfill_factor(missed, gamma, *, floor=1e-12):
    """The position factor of the KL water-filled threshold (ADVISOR 19.4): theta_t = T E_t^((1 - gamma) / 2), with T
    calibrated so the pooled count of scores_j / E_t^((1 - gamma) / 2) > T equals the target count (the mean E_t is
    absorbed into T). Returns E_t^((1 - gamma) / 2) [...]."""
    return missed.float().clamp_min(floor) ** ((1.0 - float(gamma)) / 2.0)


def fit_power_exponent(x, y, *, floor=1e-6):
    """OLS slope of log y on log x over entries with x, y > floor (gamma of KL_t ~ a E_t^gamma, ADVISOR 19.4).
    Returns (gamma, intercept, n)."""
    import numpy as np

    x, y = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    ok = (x > floor) & (y > floor) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), float("nan"), int(ok.sum())
    b, a = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    return float(b), float(a), int(ok.sum())


def tail_energy_curve(scores, ms):
    """Tail energy beyond the top-m scores, sum_{j not in top m} s_j^2, per position for each m in ``ms`` (ADVISOR 19.2,
    E_b(m) of a closed-form base). scores [P, I]. Returns [P, len(ms)] fp32."""
    import torch

    s2 = scores.float().pow(2).sort(-1, descending=True).values
    cs = s2.cumsum(-1)
    tot = cs[:, -1:]
    idx = torch.as_tensor([min(max(int(m), 1), s2.shape[-1]) - 1 for m in ms], device=s2.device)
    return (tot - cs[:, idx]).clamp_min(0)


def count_at_energy(ms, curve, target):
    """The count m (linear interpolation between grid points) at which a decreasing tail-energy curve reaches target;
    None if the curve never gets there on the grid."""
    import numpy as np

    ms, c = np.asarray(ms, np.float64), np.asarray(curve, np.float64)
    if c[0] <= target:
        return float(ms[0])
    for i in range(1, len(ms)):
        if c[i] <= target:
            return float(ms[i - 1] + (ms[i] - ms[i - 1]) * (c[i - 1] - target) / max(c[i - 1] - c[i], 1e-30))
    return None


def quantized_proxy_sigma(sigma, step, znorm):
    """A rank-k int4 proxy row's error scale with its quantization term (ADVISOR 19.2 (b)): the row's dequantization
    error is uniform with per-row step Delta_j, so its residual on z has variance Delta_j^2 ||z||^2 / 12; returns
    sqrt(sigma_j^2 + Delta_j^2 ||z||^2 / 12) [..., I] for sigma [I], step [I], znorm [...]."""
    s, d = sigma.float(), step.float()
    return (s * s + (d * d)[None, :] * (znorm.float().reshape(-1, 1) ** 2) / 12.0).sqrt().reshape(*znorm.shape, -1)


def head_pool_threshold(head_norms, drop_share):
    """Pooled per-layer quantile of the attention heads' exact output norms ||o_h|| (ADVISOR 19.2 (1)): heads at or
    below it are dropped from the o-projection; ``drop_share`` of (head, position) pairs. head_norms [...]."""
    import torch

    x = head_norms.float().flatten()
    if x.numel() > 2 ** 24:
        x = x[torch.randperm(x.numel(), device=x.device)[: 2 ** 24]]
    return float(torch.quantile(x, float(drop_share)))


def blocked_sigma_sd(h_block_norms, e_block, dims):
    """The blocked-sigma readout scale (ADVISOR 19.5 route 1): the unseen subspace split into nested blocks b of the
    hidden basis (sizes ``dims``); per state the block norms ||h_b|| [P, B], per vocabulary row the block norms
    e_{v,b} [V, B]; sd_v = sqrt(sum_b ||h_b||^2 (e_{v,b} + max_v e_{v,b})^2 / d_b) [P, V] (one block = the isotropic
    ``sigma_candidates`` scale per unit margin)."""
    import torch

    em = e_block.max(0).values
    acc = None
    for b, db in enumerate(dims):
        t = (h_block_norms[:, b:b + 1].float() ** 2) * ((e_block[:, b] + em[b]).float()[None, :] ** 2) / float(db)
        acc = t if acc is None else acc + t
    return acc.clamp_min(0).sqrt()


def blocked_sigma_candidates(lhat, sd, margin, *, target=None):
    """Candidate set lhat_v >= top - margin sd_v (``blocked_sigma_sd``); with ``target`` [P] also the certification
    statistic z* = (lhat_{v*} - top) / sd_{v*}. Returns (mask [P, V], sizes [P], z* [P] or None)."""
    top = lhat.max(-1, keepdim=True).values
    mask = lhat >= top - float(margin) * sd
    zs = None
    if target is not None:
        lv = lhat.gather(-1, target[:, None])[:, 0]
        zs = ((lv - top[:, 0]) / sd.gather(-1, target[:, None])[:, 0].clamp_min(1e-30)).float()
    return mask, mask.sum(-1), zs


def tolerance_verdict(val, lo, hi, tol=0.0):
    """Ledger verdict with the realization tolerance shown (request mdb-ledger-tolerance-band-print): 'IN' inside the
    raw band, 'IN (tolerance)' only inside [lo - tol, hi + tol], 'OUT' outside both, 'n/m' when not measured."""
    if val is None or isinstance(val, str):
        return "n/m"
    if lo <= val <= hi:
        return "IN"
    if lo - tol <= val <= hi + tol:
        return "IN (tolerance)"
    return "OUT"


# ---- Part AM (ADVISOR 20): the online rate controller, the two-side posterior with the cross term, the smoothed silu,
# the margin re-do statistics, the stream-oracle ceiling, the head write energy, spectrum shares, readout escalation


class RateController:
    """Stochastic-approximation quantile that holds a flag rate by construction (ADVISOR 20.2): per key (a layer, or a
    stream), flag = signal > tau, then tau <- tau + eta (1[flag] - q) after every position. Its fixed point is
    P(flag) = q on whatever distribution it runs on; there is no objective and no gradient (an online quantile, not a
    fit). tau starts at ``tau0`` (e.g. the pass-B quantile) and eta is ``eta_frac`` x the signal's pass-B IQR."""

    def __init__(self, q, tau0, iqr, eta_frac=0.05):
        import numpy as np

        self.q = float(q)
        self.tau = np.array(tau0, dtype=np.float64).reshape(-1).copy()
        self.eta = float(eta_frac) * np.array(iqr, dtype=np.float64).reshape(-1)
        self.tau0 = self.tau.copy()
        self.n = np.zeros_like(self.tau)
        self.nf = np.zeros_like(self.tau)

    def run(self, key, signal, valid=None):
        """Flags of a 1-D sequence ``signal`` (in stream order) for controller ``key``; entries with valid False are
        neither flagged nor used. Returns bool flags [N] (numpy)."""
        import numpy as np

        s = np.asarray(signal, np.float64).reshape(-1)
        v = np.ones(s.shape, bool) if valid is None else np.asarray(valid, bool).reshape(-1)
        out = np.zeros(s.shape, bool)
        t, e, q = self.tau[key], self.eta[key], self.q
        for i in range(s.size):
            if not v[i]:
                continue
            f = s[i] > t
            out[i] = f
            t += e * ((1.0 if f else 0.0) - q)
        self.tau[key] = t
        self.n[key] += float(v.sum())
        self.nf[key] += float(out.sum())
        return out

    def realized(self):
        """Realized flag share per key and overall (mean over keys with positions)."""
        import numpy as np

        r = np.where(self.n > 0, self.nf / np.maximum(self.n, 1), np.nan)
        return {"by_key": r.tolist(), "mean": float(np.nanmean(r)) if np.isfinite(r).any() else float("nan"),
                "tau_final": self.tau.tolist(), "tau0": self.tau0.tolist()}


def smoothed_silu(a_hat, sigma, lam=0.6266570686577501, exact_base=True):
    """Gaussian-smoothed silu under the probit form silu(x) ~ x Phi(lam x) (ADVISOR 20.3; lam = sqrt(pi / 8), the
    logistic's slope at 0: the 1.702 of the memo is the reciprocal direction, sigmoid(x) ~ Phi(x / 1.702)): for x ~ N(a_hat, sigma^2),
    m = E[silu(x)] = a Phi(k) + lam sigma^2 phi(k) / sqrt(1 + lam^2 sigma^2) and its slope in a_hat,
    d = Phi(k) + lam a phi(k) / (1 + lam^2 sigma^2)^(3/2), with k = lam a / sqrt(1 + lam^2 sigma^2). With exact_base the
    probit form supplies only the smoothing correction (m_probit(sigma) - m_probit(0), likewise d) added to the exact
    silu and silu', so sigma = 0 returns silu(a_hat) and silu'(a_hat) exactly. Returns (m, d) fp32."""
    import math

    import torch

    a, s2 = a_hat.float(), sigma.float() ** 2

    def probit(s2_):
        den = (1 + lam * lam * s2_).sqrt()
        k = lam * a / den
        phi = torch.exp(-0.5 * k * k) / math.sqrt(2 * math.pi)
        Phi = torch.special.ndtr(k)
        return a * Phi + lam * s2_ * phi / den, Phi + lam * a * phi / den ** 3
    m, d = probit(s2)
    if not exact_base:
        return m, d
    m0, d0 = probit(torch.zeros_like(s2))
    return torch.nn.functional.silu(a) + (m - m0), silu_prime(a) + (d - d0)


def proxy_residual_cross(w_a, dec_a, w_u, dec_u, encode, second_moment, *, chunk=4096):
    """The two-side residual covariance (ADVISOR 20.3): with v_a,j = W_a[j] - encode dec_a,j^T and v_u,j likewise,
    sigma_a,j^2 = v_a^T S v_a, sigma_u,j^2 = v_u^T S v_u and the cross-covariance sigma_au,j = v_a^T S v_u (both residuals
    are projections of the same dz onto the same unseen complement). Returns (sigma_a, sigma_u, sigma_au) [I] fp32."""
    import torch

    S = second_moment.float()
    enc = encode.float()
    sa, su, sau = [], [], []
    for r0 in range(0, w_a.shape[0], chunk):
        va = w_a[r0:r0 + chunk].float() - dec_a[r0:r0 + chunk].float() @ enc.T
        vu = w_u[r0:r0 + chunk].float() - dec_u[r0:r0 + chunk].float() @ enc.T
        vaS = va @ S
        sa.append((vaS * va).sum(-1).clamp_min(0).sqrt())
        sau.append((vaS * vu).sum(-1))
        su.append(((vu @ S) * vu).sum(-1).clamp_min(0).sqrt())
    return torch.cat(sa), torch.cat(su), torch.cat(sau)


def two_side_posterior(a_hat, u_hat, u_tok, g_tok, g_ref, weights, sigma_a, sigma_u, sigma_au=None, *, smooth=False,
                       product=False, scale=None, lam=0.6266570686577501, floor=1e-12):
    """The two-side pool's proxied score and its error scale (ADVISOR 20.3), the terms switchable for the in-turn
    recall curves. scale [...] (optional) multiplies the residual sds per position (sigma_a, sigma_u by s, sigma_au by
    s^2: the position's unseen energy ||P_perp dz_t|| over its RMS). With g = silu(a_hat) or, smoothed, m = E[silu(a)]
    and slope d (``smoothed_silu``), the mean of g(a) u is m u_hat + d sigma_au (Stein) and
    tau^2 = w^2 [(d u_hat sigma_a)^2 + ((m - g_ref) sigma_u)^2 + 2 d u_hat (m - g_ref) sigma_au
    + (product) d^2 (sigma_a^2 sigma_u^2 + sigma_au^2)]; sigma_au None drops the cross terms (the AL key). The score is
    |mean - g_tok u_tok - g_ref (u_hat - u_tok)| w. Returns (s_hat, tau) [..., I] fp32."""
    import torch

    a, uh, ut = a_hat.float(), u_hat.float(), u_tok.float()
    w, gr = weights.float(), g_ref.float()
    sa, su = sigma_a.float(), sigma_u.float()
    sau = None if sigma_au is None else sigma_au.float()
    if scale is not None:
        sc = scale.float()[..., None]
        sa, su = sa * sc, su * sc
        if sau is not None:
            sau = sau * sc * sc
    if smooth:
        m, dg = smoothed_silu(a, sa, lam)
    else:
        m, dg = torch.nn.functional.silu(a), silu_prime(a)
    mean = m * uh + (dg * sau if (sau is not None and smooth) else 0.0)
    s_hat = (mean - g_tok.float() * ut - gr * (uh - ut)).abs() * w
    ta, tu = dg * uh * sa, (m - gr) * su
    var = ta * ta + tu * tu
    if sau is not None:
        var = var + 2 * dg * uh * (m - gr) * sau
    if product:
        var = var + dg * dg * (sa * sa * su * su + (0.0 if sau is None else sau * sau))
    return s_hat, (w * var.clamp_min(0).sqrt()).clamp_min(floor)


def margin_capture(margin, agree, kl, qs=(0.01, 0.02, 0.03, 0.05, 0.10)):
    """The readout-side sensitivity statistic (ADVISOR 20.2): ranking positions by the code's own top-1 minus top-2 logit
    gap (ascending), the share phi of the code's disagreements (agree == 0) and of its KL mass that the lowest-q share
    holds. margin, agree, kl [N]. Returns {q: {"phi": ..., "kl_share": ..., "flagged": n}}."""
    import numpy as np

    m, ag, k = np.asarray(margin, np.float64), np.asarray(agree, np.float64), np.asarray(kl, np.float64)
    order = np.argsort(m, kind="stable")
    nd, kt = float((ag < 0.5).sum()), float(k.sum())
    out = {}
    for q in qs:
        n = max(1, int(round(float(q) * m.size)))
        sel = order[:n]
        out[float(q)] = {"phi": float((ag[sel] < 0.5).sum()) / nd if nd else float("nan"),
                         "kl_share": float(k[sel].sum()) / kt if kt else float("nan"), "flagged": n,
                         "phi_max": min(1.0, n / nd) if nd else float("nan")}  # no ranking at rate q captures more
    return out


def top2_margin_fp32(h_normed, w_head, chunk=32768):
    """The readout margin top-1 - top-2 in fp32 logits of a normed final state h_normed [N, d] (bf16 or fp32) against
    the head w_head [V, d], the vocabulary in ``chunk``-row fp32 slices (Part AM: the bf16 head's logits are quantized
    at 2^-3..2^-4 near |logit| 16-32 and tie on >= 3% of positions, so a lowest-q margin ranking needs fp32 logits, the
    precision an int8 per-row head produces). Returns [N] fp32."""
    import torch

    h = h_normed.float()
    tops = [(h @ w_head[c0:c0 + chunk].float().T).topk(2, dim=-1).values for c0 in range(0, w_head.shape[0], chunk)]
    tv = torch.cat(tops, -1).topk(2, dim=-1).values
    return tv[:, 0] - tv[:, 1]


def redo_mixture(agree, kl, flags, f_code, readout, per=1000):
    """The margin re-do's derived mixture (ADVISOR 20.2): flagged positions take the exact outcome (agreement 1, KL 0;
    the cache drift is the built stream's difference), and the cost is the code at every position plus the dense
    column at each flagged one: decide-after per (f + q) and, honestly, per (f + q (1 + readout)) (the served readout
    spent before the re-do; ADVISOR 16.1 d). Returns dict."""
    import numpy as np

    ag, k, fl = np.asarray(agree, np.float64), np.asarray(kl, np.float64), np.asarray(flags, bool)
    q = float(fl.mean()) if fl.size else 0.0
    return {"agree": float(np.where(fl, 1.0, ag).mean()), "kl": float(np.where(fl, 0.0, k).mean()), "q": q,
            "column_equivalents_decide_after": per * (float(f_code) + q),
            "column_equivalents_honest": per * (float(f_code) + q * (1 + float(readout)))}


def kl_persistence(kl, chunk, q=0.10):
    """Within-chunk structure of a stream's per-position KL (ADVISOR 20.2): the lag-1 autocorrelation of log KL inside
    chunks, and the tail persistence P(top-q at t | top-q at t - 1) against its base rate q. kl, chunk [N] (positions in
    order within each chunk). Returns dict."""
    import numpy as np

    k = np.log(np.maximum(np.asarray(kl, np.float64), 1e-8))
    ch = np.asarray(chunk)
    same = ch[1:] == ch[:-1]
    a, b = k[:-1][same], k[1:][same]
    ac = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan")
    top = np.asarray(kl, np.float64) >= np.quantile(np.asarray(kl, np.float64), 1 - q)
    prev, cur = top[:-1][same], top[1:][same]
    pers = float(cur[prev].mean()) if prev.any() else float("nan")
    return {"lag1_autocorr_logkl": ac, "tail_persistence": pers, "base_rate": float(top.mean()), "pairs": int(same.sum())}


def head_write_energy(o, w_o, fisher, head_dim):
    """The psi-Fisher energy of each attention head's write (ADVISOR 20.7): o [..., n_h, hd] (the head outputs before
    W_O), w_o [d, n_h hd]; e_h = (W_O,h o_h)^T F (W_O,h o_h). Returns [..., n_h] fp32."""
    import torch

    nh = o.shape[-2]
    W = w_o.float().reshape(w_o.shape[0], nh, head_dim)  # [d, n_h, hd]
    wr = torch.einsum("...hk,dhk->...hd", o.float(), W)
    return ((wr @ fisher.float()) * wr).sum(-1)


def head_bound_score(o, w_o, head_dim):
    """The weights-only head ranking ||W_O,h||_F ||o_h|| (ADVISOR 20.7; an upper bound of the head's write norm).
    o [..., n_h, hd]. Returns [..., n_h] fp32."""
    nh = o.shape[-2]
    fro = w_o.float().reshape(w_o.shape[0], nh, head_dim).pow(2).sum((0, 2)).sqrt()
    return o.float().norm(dim=-1) * fro


def spectrum_share(second_moment, ranks):
    """Share of a second moment's trace in its top-r eigenvalues, for each r (ADVISOR 20.4: whether a fixed basis can
    complete a write term; in-sample, so an upper bound for a held-out basis). Returns {r: share}."""
    import torch

    ev = torch.linalg.eigvalsh(((second_moment + second_moment.T) / 2).double()).flip(0).clamp_min(0)
    tot = float(ev.sum())
    return {int(r): (float(ev[:int(r)].sum()) / tot if tot > 0 else float("nan")) for r in ranks}


def readout_escalation(c1, c2, vocab, d, r1, r2, *, q_bytes=1, scale_bytes=4, c1_cap=None, c2_cap=20000):
    """The two-stage readout's expected cost over positions (ADVISOR 20.8): c1, c2 [P] the stage-1 and stage-2
    candidate counts; table 1 (V x r1 int) read every step, table 2's (r2 - r1) columns gathered on C1, exact int8 head
    rows on C2. Returns E|C1|, E|C2|, the counts of positions whose C1 exceeds c1_cap (default V / 4) or C2 exceeds
    c2_cap, and the expected bytes per step."""
    import numpy as np

    c1, c2 = np.asarray(c1, np.float64), np.asarray(c2, np.float64)
    cap1 = vocab / 4 if c1_cap is None else c1_cap
    t1 = vocab * (r1 * q_bytes + scale_bytes)
    b = t1 + c1.mean() * ((r2 - r1) * q_bytes + scale_bytes) + c2.mean() * (d * q_bytes + scale_bytes)
    return {"E_C1": float(c1.mean()), "E_C2": float(c2.mean()), "median_C1": float(np.median(c1)), "median_C2": float(np.median(c2)),
            "q90_C1": float(np.quantile(c1, 0.9)), "q90_C2": float(np.quantile(c2, 0.9)),
            "escalated_C1": int((c1 > cap1).sum()), "escalated_C2": int((c2 > c2_cap).sum()), "positions": int(c1.size),
            "expected_bytes": float(b), "table1_bytes": float(t1)}


def late_redo_price(L, n_layers, readout, per=1000, convention="readout_separate"):
    """Column-equivalents per flagged position of a late-L dense re-do (ADVISOR 21.4): the flagged position's last L
    decoder layers recomputed with dense MLPs from the code's own layer-(n_layers - L) state, plus one more readout. In
    the ledger's readout-separate convention (``ServingLedger.report``: column-equivalents exclude the readout; the
    honest dense re-do is per (1 + readout)) the price is per (L / n_layers + readout). ``convention="memo_21_4"`` is
    the memo's per ((1 - readout) L / n_layers + 2 readout), which equals the honest dense price at L = n_layers but
    charges a second readout and (1 - readout) per layer share below it (printed beside, not used for scoring here)."""
    L, n, r = float(L), float(n_layers), float(readout)
    if convention == "memo_21_4":
        return per * ((1 - r) * L / n + 2 * r)
    return per * (L / n + r)


def late_fix_share_map(v):
    """The arctan map of the late re-do (ADVISOR 21.4, derived): for a flagged near-tie (mu ~ 0) with the code's error
    eps = eps_early + eps_late, v = Var(eps_late) / Var(eps), Gaussian, removing eps_late fixes the flipped sign with
    probability f = (2 / pi) arctan sqrt(v / (1 - v)). v [...] in [0, 1]. Returns f (numpy)."""
    import numpy as np

    v = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    return (2 / np.pi) * np.arctan(np.sqrt(v / np.maximum(1 - v, 1e-300)))


def late_error_share(e_layers, L):
    """Share of a code's summed per-layer error energy (e.g. the psi-Fisher energy of f - w, [n_layers, N]) carried
    by the last L layers: sum_{l >= n - L} sum_t e[l, t] / sum_l sum_t e[l, t] (v_L of ADVISOR 21.4)."""
    import numpy as np

    e = np.asarray(e_layers, np.float64)
    tot = float(e.sum())
    return float(e[e.shape[0] - int(L):].sum()) / tot if tot > 0 else float("nan")


def redo_fix_share(agree_base, agree_redo, flags):
    """What a re-do stream did at its flagged positions (ADVISOR 21.4): f = the net share of flagged flips fixed,
    (A_redo - A_base on the flagged) / P(dis | flag) (the arctan map's quantity: fixed flips minus broken agreements,
    per flip), with its two parts fixed = P(redo agrees | flagged, base disagreed) and broken = P(redo disagrees |
    flagged, base agreed), the base's P(dis | flag), and the unflagged positions' mean agreement change (the cache
    effect). Arrays [N]."""
    import numpy as np

    a0, a1, fl = (np.asarray(agree_base) > 0.5), (np.asarray(agree_redo) > 0.5), np.asarray(flags, bool)
    fd, fa = fl & ~a0, fl & a0
    pd = float((~a0[fl]).mean()) if fl.any() else float("nan")
    net = float(a1[fl].mean() - a0[fl].mean()) / pd if (fl.any() and pd > 0) else float("nan")
    return {"f": net, "fixed": float(a1[fd].mean()) if fd.any() else float("nan"),
            "broken": float((~a1[fa]).mean()) if fa.any() else float("nan"), "p_dis_flag": pd, "flagged": int(fl.sum()),
            "flagged_flips": int(fd.sum()), "unflagged_change": float(a1[~fl].mean() - a0[~fl].mean()) if (~fl).any() else float("nan"),
            "net_gain": float(a1.mean() - a0.mean())}


class PoolRateController:
    """Online quantile that holds a pool's mean size per position (ADVISOR 21.3, lever 1): per key (a layer), position
    t's pool is {j : z_j > tau_t}, and tau_{t+1} = tau_t + gain (n_t - target) / slope, n_t the pool's size at tau_t.
    With a constant slope (set once per key: the median local |dn/dz| at tau0 over the first call's positions) the
    fixed point is E[n_t] = target on whatever distribution it runs on (Robbins-Monro; no objective, no fit). The
    sizes come from a per-position grid: zk [N, G] the k-th largest z at the ascending counts kgrid [G]."""

    def __init__(self, target, tau0, gain=0.05):
        import numpy as np

        self.target = float(target)
        self.tau = np.array(tau0, dtype=np.float64).reshape(-1).copy()
        self.tau0 = self.tau.copy()
        self.gain = float(gain)
        self.slope = np.full_like(self.tau, np.nan)
        self.n = np.zeros_like(self.tau)
        self.sum = np.zeros_like(self.tau)

    @staticmethod
    def count_at(zrow, kgrid, tau):
        import numpy as np

        return float(np.interp(-tau, -np.asarray(zrow, np.float64), np.asarray(kgrid, np.float64)))

    def run_grid(self, key, zk, kgrid):
        """Positions in stream order: returns (tau_used [N], size [N]) (numpy)."""
        import numpy as np

        zk, kg = np.asarray(zk, np.float64), np.asarray(kgrid, np.float64)
        if not np.isfinite(self.slope[key]):
            sl = []
            for zr in zk:
                i = int(np.searchsorted(-zr, -self.tau[key]))
                i = min(max(i, 1), len(kg) - 1)
                dz = zr[i - 1] - zr[i]
                if dz > 0:
                    sl.append((kg[i] - kg[i - 1]) / dz)
            self.slope[key] = float(np.median(sl)) if sl else 1.0
        t, g, tg, s = self.tau[key], self.gain, self.target, self.slope[key]
        taus, ns = np.empty(len(zk)), np.empty(len(zk))
        for i, zr in enumerate(zk):
            n_ = float(np.interp(-t, -zr, kg))
            taus[i], ns[i] = t, n_
            t += g * (n_ - tg) / s
        self.tau[key] = t
        self.n[key] += len(zk)
        self.sum[key] += float(ns.sum())
        return taus, ns

    def realized(self):
        import numpy as np

        m = np.where(self.n > 0, self.sum / np.maximum(self.n, 1), np.nan)
        return {"mean_size_by_key": m.tolist(), "ratio_to_target": float(np.nanmean(m) / self.target) if np.isfinite(m).any() else float("nan"),
                "tau_final": self.tau.tolist(), "tau0": self.tau0.tolist(), "slope": self.slope.tolist(), "target": self.target}


def union_pool_topk(z, forced, size):
    """A pool of exactly ``size`` members per row (ADVISOR 21.3, lever 4): the forced members (e.g. a token's top-k_tok
    prior members) first, the rest filled by z. z [..., I] float, forced [..., I] bool. Returns the bool pool mask."""
    import torch

    zf = z.float()
    big = (zf.abs().max() + 1).clamp_min(1.0) * 4
    idx = (zf + forced.float() * big).topk(int(size), dim=-1).indices
    return torch.zeros_like(forced, dtype=torch.bool).scatter_(-1, idx, True)


def zbin_calibration(z, member, edges=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0)):
    """Calibration of a Gaussian membership posterior (ADVISOR 21.3, lever 5): neurons binned by z (bins (-inf, e_0],
    (e_0, e_1], ..., (e_last, inf)); per bin the count, the realized member share, the mean predicted Phi(z) and their
    ratio. z, member [...] (torch). Returns a list of dicts."""
    import torch

    zf, mf = z.float().reshape(-1), member.reshape(-1).float()
    e = [-float("inf")] + [float(x) for x in edges] + [float("inf")]
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        b = (zf > lo) & (zf <= hi)
        n = int(b.sum())
        real = float(mf[b].mean()) if n else float("nan")
        pred = float(torch.special.ndtr(zf[b]).mean()) if n else float("nan")
        out.append({"lo": lo, "hi": hi, "n": n, "realized": real, "predicted": pred,
                    "ratio": (real / pred) if (n and pred > 0) else float("nan")})
    return out


def moment_sums(x):
    """Running sums for the standardized residual's moments: (n, sum x, sum x^2, sum x^3, sum x^4) in float64."""
    xd = x.double().reshape(-1)
    return [float(xd.numel()), float(xd.sum()), float((xd ** 2).sum()), float((xd ** 3).sum()), float((xd ** 4).sum())]


def kurtosis_from_sums(s):
    """Mean, sd, excess kurtosis and the Student-t nu that matches it (nu = 4 + 6 / excess for excess > 0; inf for a
    Gaussian or lighter tail) from ``moment_sums`` totals."""
    n, s1, s2, s3, s4 = (float(v) for v in s)
    if n < 2:
        return {"n": n}
    m = s1 / n
    var = s2 / n - m * m
    m4 = s4 / n - 4 * m * s3 / n + 6 * m * m * s2 / n - 3 * m ** 4
    ex = m4 / max(var * var, 1e-300) - 3.0
    return {"n": n, "mean": m, "sd": var ** 0.5, "excess_kurtosis": ex, "nu_t": (4 + 6 / ex) if ex > 0 else float("inf")}


def escalated_readout_bytes(c1, c2, esc, vocab, d, r1, r2, *, q_bytes=1, scale_bytes=4):
    """Expected bytes per step of the two-stage readout with a per-position starting level (ADVISOR 21.6 item 2):
    non-escalated positions read table 1 (V x r1), table 2's (r2 - r1) columns on C1 and the exact rows on C2;
    escalated ones (esc True) read one full table at r2 (V x r2; table 2 alone holds only columns r1..r2 and cannot form
    the r2 logits without table 1) and the exact rows on C2. ``memo_table2_only`` prices escalated positions at V x (r2 -
    r1) as the memo wrote (printed beside; not a buildable readout). c1, c2, esc [P]. Returns dict."""
    import numpy as np

    c1, c2, es = np.asarray(c1, np.float64), np.asarray(c2, np.float64), np.asarray(esc, bool)
    rowb = d * q_bytes + scale_bytes
    two = vocab * (r1 * q_bytes + scale_bytes) + c1 * ((r2 - r1) * q_bytes + scale_bytes) + c2 * rowb
    full = vocab * (r2 * q_bytes + scale_bytes) + c2 * rowb
    memo = vocab * ((r2 - r1) * q_bytes + scale_bytes) + c2 * rowb
    return {"escalated_share": float(es.mean()), "expected_bytes": float(np.where(es, full, two).mean()),
            "expected_bytes_two_stage": float(two.mean()), "expected_bytes_memo_table2_only": float(np.where(es, memo, two).mean()),
            "positions": int(c1.size)}

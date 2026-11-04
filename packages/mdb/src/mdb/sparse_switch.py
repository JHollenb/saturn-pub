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


def spearman(x, y):
    """Spearman rank correlation of two 1-D tensors (no tie correction)."""
    rx = x.argsort().argsort().double()
    ry = y.argsort().argsort().double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / (rx.norm() * ry.norm()))

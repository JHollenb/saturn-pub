"""Native capture, true resume and K/V injection for Hugging Face causal decoders.

``DecoderSplice`` caches the family prefix K/V once. ``run`` executes only the
decoder layers ``start..`` on a supplied residual for the listed suffix
positions; every layer reads ``[prefix | injected | own]`` keys and values under
a causal mask over absolute positions, so positions that do not run can still
contribute K/V (compiled or native). No layer below ``start`` executes.
``forward`` is the independent hook path through the unmodified model, used by
the parity gates.

``measure`` fits compiled maps on native states, scores full-position resumes
and partial-position arms on held-out rows, runs the parity gates and returns a
sealed ``mdb-compiled-state-residual-v1`` report plus the maps.
"""
from __future__ import annotations

import copy
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .compiled_state import (
    CompiledStateError,
    CompilerFit,
    PromptFamily,
    ResidualAccumulator,
    agreement,
    batch_residual_moments,
    compile_state,
    design,
    horizon,
    seal_report,
    visibility,
    weights_sha256,
)


@dataclass
class Rows:
    """Suffix token ids [N, T], family symbol values [N, S], optional evaluator reference
    token ids [N, K] at the score positions (scoring only; never compiler input)."""

    input_ids: Any
    values: Any
    reference: Any = None

    def __len__(self):
        return len(self.input_ids)

    def take(self, sl):
        return Rows(self.input_ids[sl], self.values[sl], None if self.reference is None else self.reference[sl])


class _SuffixCache:
    """Duck-typed HF cache for one manual pass: layer ``l`` attends to the cached
    prefix, then injected K/V, then the running positions' own K/V."""

    def __init__(self, prefix_kv, batch, inject, capture):
        self.prefix_kv, self.batch, self.inject, self.capture, self.captured = prefix_kv, batch, inject, capture, {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.capture:
            self.captured[layer_idx] = (key_states, value_states)
        pk, pv = self.prefix_kv[layer_idx]
        ks, vs = [pk.expand(self.batch, -1, -1, -1)], [pv.expand(self.batch, -1, -1, -1)]
        if layer_idx in self.inject:
            k, v = self.inject[layer_idx]
            ks.append(k)
            vs.append(v)
        ks.append(key_states)
        vs.append(value_states)
        return torch.cat(ks, 2), torch.cat(vs, 2)


class DecoderSplice:
    def __init__(self, model, prefix_ids: Sequence[int]):
        if model.training:
            raise CompiledStateError("compiled-state capture needs a model in eval mode")
        types = getattr(model.config, "layer_types", None) or ()
        if any(t != "full_attention" for t in types):
            raise CompiledStateError("true resume supports full-attention decoders only")
        self.model = model
        self.decoder = model.get_decoder()
        self.layers = self.decoder.layers
        self.num_layers = len(self.layers)
        self.head = model.get_output_embeddings()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.prefix_len = len(prefix_ids)
        with torch.no_grad():
            ids = torch.as_tensor([list(prefix_ids)], device=self.device)
            self._prefix = model(input_ids=ids, use_cache=True).past_key_values
        self.prefix_kv = [(layer.keys, layer.values) for layer in self._prefix.layers]
        self.kv_heads, self.head_dim = self.prefix_kv[0][0].shape[1], self.prefix_kv[0][0].shape[3]
        self._prefix_pos = torch.arange(self.prefix_len, device=self.device)
        self._vocab_rows = {}

    @property
    def kv_width(self) -> int:
        """Width of one position's flattened [keys | values] row at one layer."""
        return 2 * self.kv_heads * self.head_dim

    @torch.no_grad()
    def embed(self, input_ids):
        return self.decoder.get_input_embeddings()(torch.as_tensor(input_ids, device=self.device))

    @torch.no_grad()
    def run(self, hidden, start: int = 0, *, positions: Sequence[int] | None = None,
            kv: Mapping[int, tuple[Any, Any]] | None = None, kv_positions: Sequence[int] | None = None,
            capture: Sequence[int] = (), capture_kv: Sequence[int] = ()):
        """Run decoder layers ``start..`` on ``hidden`` [B, Tq, D], the residual entering
        ``start`` at suffix ``positions`` (default 0..Tq-1). ``kv[l] = (k, v)``
        [B, heads, E, head_dim] post-RoPE supplies the K/V of suffix ``kv_positions``
        at every layer ``l >= start``. Returns (final pre-norm hidden, {layer:
        residual entering it} for ``capture``, {layer: (k, v)} written by the running
        positions for ``capture_kv``)."""
        nL = self.num_layers
        if type(start) is not int or not 0 <= start <= nL:
            raise CompiledStateError(f"start layer {start} outside 0..{nL}")
        if any(L < start or L > nL for L in capture) or any(L < start or L >= nL for L in capture_kv):
            raise CompiledStateError("captures must name layers the resumed pass executes")
        hidden = torch.as_tensor(hidden, device=self.device).to(self.dtype)
        B, Tq, _ = hidden.shape
        pos = (torch.arange(Tq, device=self.device) if positions is None
               else torch.as_tensor(list(positions), device=self.device))
        if pos.numel() != Tq:
            raise CompiledStateError("one running position per hidden row")
        qpos = self.prefix_len + pos
        parts, inject = [self._prefix_pos], {}
        if kv is not None or kv_positions is not None:
            if kv is None or kv_positions is None:
                raise CompiledStateError("K/V injection needs both kv and kv_positions")
            kp = torch.as_tensor(list(kv_positions), device=self.device)
            if set(kp.tolist()) & set(pos.tolist()):
                raise CompiledStateError("a position cannot both run and be injected")
            want = (B, self.kv_heads, kp.numel(), self.head_dim)
            for layer in range(start, nL):
                if layer not in kv or tuple(kv[layer][0].shape) != want or tuple(kv[layer][1].shape) != want:
                    raise CompiledStateError(f"injected K/V at layer {layer} must be two {want} tensors")
                inject[layer] = (kv[layer][0].to(self.dtype), kv[layer][1].to(self.dtype))
            parts.append(self.prefix_len + kp)
        kpos = torch.cat([*parts, qpos])
        mask = (kpos[None, :] <= qpos[:, None])[None, None]
        position_ids = qpos[None]
        rope = self.decoder.rotary_emb(hidden, position_ids)
        cache = _SuffixCache(self.prefix_kv, B, inject, set(capture_kv))
        states = {}
        for layer in range(start, nL):
            if layer in capture:
                states[layer] = hidden
            out = self.layers[layer](hidden, attention_mask=mask, position_ids=position_ids, past_key_values=cache,
                                     use_cache=True, position_embeddings=rope)
            hidden = out[0] if isinstance(out, tuple) else out
        if nL in capture:
            states[nL] = hidden
        return hidden, states, cache.captured

    def native(self, input_ids, *, capture: Sequence[int] = (), capture_kv: Sequence[int] = ()):
        """Prefix-cached native forward over the suffix through the manual layer loop."""
        return self.run(self.embed(input_ids), 0, capture=capture, capture_kv=capture_kv)

    @torch.no_grad()
    def readout(self, hidden, vocab: Sequence[int] | None = None):
        """Final norm and output head, optionally restricted to the ``vocab`` token rows."""
        h = self.decoder.norm(hidden)
        if vocab is None:
            return self.head(h)
        key = tuple(int(v) for v in vocab)
        if key not in self._vocab_rows:
            idx = torch.as_tensor(key, device=self.device)
            bias = None if getattr(self.head, "bias", None) is None else self.head.bias[idx].contiguous()
            self._vocab_rows[key] = (self.head.weight[idx].contiguous(), idx, bias)
        w, _, bias = self._vocab_rows[key]
        out = h @ w.T
        return out if bias is None else out + bias

    def tokens(self, logits, vocab: Sequence[int] | None = None):
        """Argmax token ids of ``logits`` (mapped back through ``vocab`` when given)."""
        top = logits.argmax(-1)
        return top if vocab is None else self._vocab_rows[tuple(int(v) for v in vocab)][1][top]

    @torch.no_grad()
    def decode(self, input_ids, steps: int, vocab: Sequence[int] | None = None):
        """Native KV-cached greedy decode of ``steps`` tokens after the suffix. Returns
        (tokens [B, steps], pre-norm final states [B, T0 + steps - 1, D] at every suffix
        position whose input is known, i.e. the template plus the fed-back tokens)."""
        ids = torch.as_tensor(input_ids, device=self.device)
        T0, nL = ids.shape[1], self.num_layers
        x, _, kv = self.native(ids, capture_kv=range(nL))
        finals, toks = [x], [self.tokens(self.readout(x[:, -1], vocab), vocab)]
        for k in range(1, steps):
            p = T0 + k - 1
            xk, _, kvn = self.run(self.embed(toks[-1][:, None]), 0, positions=[p], kv=kv,
                                  kv_positions=range(p), capture_kv=range(nL))
            kv = {layer: (torch.cat([kv[layer][0], kvn[layer][0]], 2), torch.cat([kv[layer][1], kvn[layer][1]], 2))
                  for layer in range(nL)}
            finals.append(xk)
            toks.append(self.tokens(self.readout(xk[:, -1], vocab), vocab))
        return torch.stack(toks, 1), torch.cat(finals, 1)

    def _module(self, layer: int):
        if not 0 <= layer <= self.num_layers:
            raise CompiledStateError(f"layer {layer} outside 0..{self.num_layers}")
        return self.layers[layer] if layer < self.num_layers else self.decoder.norm

    @torch.no_grad()
    def forward(self, input_ids, *, capture: Sequence[int] = (), splice: tuple[int, Any] | None = None,
                keep: int = 0):
        """Hook path through the unmodified model: logits [B, T, V] for the suffix after
        the cached prefix (the last ``keep`` positions only when ``keep`` > 0);
        ``capture`` returns the residual entering each listed layer, ``splice=(L,
        state)`` replaces it at every position. Layers below ``L`` still execute."""
        ids = torch.as_tensor(input_ids, device=self.device)
        cache = copy.deepcopy(self._prefix)
        cache.batch_repeat_interleave(ids.shape[0])
        seen, hooks = {}, []

        def hook(L, replace):
            def pre(module, args, kwargs):
                hidden = args[0] if args else kwargs["hidden_states"]
                if replace is not None:
                    if replace.shape != hidden.shape:
                        raise CompiledStateError(f"splice state {tuple(replace.shape)} != {tuple(hidden.shape)}")
                    hidden = replace.to(hidden.dtype)
                    if args:
                        return (hidden, *args[1:]), kwargs
                    return args, {**kwargs, "hidden_states": hidden}
                seen[L] = hidden
                return None
            return pre

        try:
            for L in capture:
                hooks.append(self._module(L).register_forward_pre_hook(hook(L, None), with_kwargs=True))
            if splice is not None:
                L, state = splice
                hooks.append(self._module(L).register_forward_pre_hook(
                    hook(L, torch.as_tensor(state, device=self.device)), with_kwargs=True, prepend=True))
            logits = self.model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=keep).logits
        finally:
            for h in hooks:
                h.remove()
        missing = [L for L in capture if L not in seen]
        if missing:
            raise CompiledStateError(f"capture layers never ran: {missing}")
        return logits, seen


def kv_rows(k, v):
    """[B, heads, E, head_dim] keys and values -> [B, E, 2 * heads * head_dim] rows."""
    B, H, E, hd = k.shape
    return torch.cat([k.permute(0, 2, 1, 3).reshape(B, E, H * hd), v.permute(0, 2, 1, 3).reshape(B, E, H * hd)], -1)


def kv_unrows(z, heads: int, head_dim: int):
    """Inverse of ``kv_rows``."""
    B, E, _ = z.shape
    half = heads * head_dim
    k = z[..., :half].reshape(B, E, heads, head_dim).permute(0, 2, 1, 3)
    v = z[..., half:].reshape(B, E, heads, head_dim).permute(0, 2, 1, 3)
    return k.contiguous(), v.contiguous()


def device_features(family: PromptFamily, values, vis):
    """Visibility-masked one-hot features [B, T, F] on the device from integer symbol
    values [B, S] (a torch tensor). Unlike ``design`` it does not range-check."""
    off = torch.as_tensor(list(family.offsets().values()), device=values.device)
    X = torch.zeros(len(values), family.width, device=values.device, dtype=vis.dtype)
    X[:, 0] = 1
    X.scatter_(1, values.long() + off[None], 1.0)
    return X[:, None, :] * vis[None]


def _compare(a, b):
    a, b = a.float(), b.float()
    return {"max_abs_dlogit": float((a - b).abs().max()),
            "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).all(-1).float().mean())}


@torch.no_grad()
def parity(run: DecoderSplice, family: PromptFamily, input_ids, layers: Sequence[int], *,
           partial_layers: Sequence[int] = (), tol: float | None = None) -> dict[str, Any]:
    """Parity gates on one batch of native rows, at the family's score positions:

    - ``manual_native_vs_hf``: the manual layer loop from embeddings vs the model's own forward;
    - ``resume[L]``: resuming from the captured native state at ``L`` vs manual native;
    - ``kv_injection[L]``: only the score positions run from their native state at ``L``,
      the other positions inject their native K/V at ``L..``, vs manual native;
    - ``partial_vs_hook[L]``: the same partial run vs the hook-path splice of native states.

    The gate passes when every check has top-1 agreement 1.0 and, if ``tol`` is
    given, max |dlogit| <= tol. ``exact`` reports whether every max |dlogit| is 0."""
    nL, T = run.num_layers, len(family.positions)
    score = list(family.score_positions)
    kv_pos = [p for p in range(T) if p not in score]
    layers = sorted(set(int(L) for L in layers) | set(int(L) for L in partial_layers))
    ids = torch.as_tensor(input_ids, device=run.device)
    keep = T - min(score)  # hook-path logits only for the tail that holds the score positions
    tail = [p - min(score) for p in score]
    hf_logits, hf_cap = run.forward(ids, capture=layers, keep=keep)
    lo = min(partial_layers, default=nL)
    x, cap, kvs = run.native(ids, capture=layers, capture_kv=range(lo, nL))
    native = run.readout(x[:, score])
    out = {"rows": int(ids.shape[0]), "manual_native_vs_hf": _compare(native, hf_logits[:, tail]),
           "resume": {}, "kv_injection": {}, "partial_vs_hook": {}}
    for L in layers:
        xr, _, _ = run.run(cap[L], L)
        out["resume"][str(L)] = _compare(run.readout(xr[:, score]), native)
    for L in sorted(set(int(L) for L in partial_layers)):
        if L >= nL:
            raise CompiledStateError("partial runs need at least one layer")
        kv = {layer: (kvs[layer][0][:, :, kv_pos], kvs[layer][1][:, :, kv_pos]) for layer in range(L, nL)}
        xp, _, _ = run.run(cap[L][:, score], L, positions=score, kv=kv, kv_positions=kv_pos)
        lp = run.readout(xp)
        out["kv_injection"][str(L)] = _compare(lp, native)
        hook, _ = run.forward(ids, splice=(L, hf_cap[L]), keep=keep)
        out["partial_vs_hook"][str(L)] = _compare(lp, hook[:, tail])
    checks = [out["manual_native_vs_hf"], *(c for k in ("resume", "kv_injection", "partial_vs_hook")
                                            for c in out[k].values())]
    out["gate"] = {"passed": all(c["top1_agreement"] == 1.0 and (tol is None or c["max_abs_dlogit"] <= tol)
                                 for c in checks),
                   "exact": all(c["max_abs_dlogit"] == 0.0 for c in checks),
                   "rule": "top-1 agreement 1.0 in every check" + ("" if tol is None else f" and max |dlogit| <= {tol}")}
    return out


class _DeviceMoments:
    """Ridge moments accumulated on the device, handed to ``CompilerFit`` once."""

    def __init__(self, family, layers, hidden_size, fsets, kv_layers=(), kv_width=0, device="cpu"):
        T, F = len(family.positions), family.width
        self.family, self.layers, self.kv_layers, self.D, self.kv_width = family, layers, kv_layers, hidden_size, kv_width
        self.fsets = fsets
        self.n = 0
        self.gram = torch.zeros(T, F, F, device=device, dtype=torch.float64)
        self.xty = {L: torch.zeros(T, F, hidden_size, device=device) for L in layers}
        self.xty.update({("kv", L): torch.zeros(T, F, kv_width, device=device) for L in kv_layers})
        self.switches = sorted({family.feature_sets[f].switch for f in fsets if family.feature_sets[f].switch})
        self.names = [s.name for s in family.symbols]
        self.sw = {s: {g: [0, torch.zeros(T - family.symbol(s).visible_from, F, F, device=device, dtype=torch.float64),
                           {L: torch.zeros(T - family.symbol(s).visible_from, F, hidden_size, device=device)
                            for L in layers}]
                       for g in range(family.symbol(s).cardinality)} for s in self.switches}

    def add(self, P, targets, values):
        """P [B, T, F] masked features; targets keyed like ``xty`` ([B, T, width])."""
        self.n += len(P)
        self.gram += torch.einsum("bpf,bpg->pfg", P, P).double()
        for key, acc in self.xty.items():
            acc += torch.einsum("bpf,bpd->pfd", P, targets[key].float())
        for s in self.switches:
            s0 = self.family.symbol(s).visible_from
            grp = values[:, self.names.index(s)]
            for g, acc in self.sw[s].items():
                m = grp == g
                if m.any():
                    Pg = P[m, s0:]
                    acc[0] += int(m.sum())
                    acc[1] += torch.einsum("bpf,bpg->pfg", Pg, Pg).double()
                    for L in self.layers:
                        acc[2][L] += torch.einsum("bpf,bpd->pfd", Pg, targets[L][m, s0:].float())

    def fit(self) -> CompilerFit:
        fitter = CompilerFit(self.family, self.layers, self.D, self.fsets, kv_layers=self.kv_layers,
                             kv_width=self.kv_width)
        fitter.add_moments(self.n, self.gram.cpu().numpy(), {k: v.cpu().numpy() for k, v in self.xty.items()},
                           {s: {g: (n, gg.cpu().numpy(), {L: v.cpu().numpy() for L, v in gx.items()})
                                for g, (n, gg, gx) in groups.items()} for s, groups in self.sw.items()})
        return fitter


@torch.no_grad()
def measure(model, family: PromptFamily, prefix_ids: Sequence[int], fit: Rows, evaluate: Rows, *,
            layers: Sequence[int], feature_sets: Sequence[str] | None = None, lam: float = 1.0,
            threshold: float = 0.95, batch: int = 256, r2_rows: int | None = None,
            partial_layers: Sequence[int] = (), parity_tol: float | None = None, parity_rows: int | None = None,
            model_identity: Mapping[str, Any] | None = None, log=None):
    """Fit compiled maps on ``fit`` native states and score them on ``evaluate``.

    Full arms: the compiled state enters ``L`` at every suffix position and only
    layers ``L..`` run. Partial arms (``partial_layers``): only the score positions
    run ``L..`` from their compiled state while every other position injects
    compiled K/V at each layer ``L..`` (Part F's T3c); an oracle variant injects
    native K/V instead. The parity gates run on the first ``parity_rows`` rows of
    the first evaluation batch (all of it by default). Returns (report,
    {(feature_set, layer[, "kv"]): map})."""
    t0 = time.time()
    log = log or (lambda *_: None)
    layers = sorted(set(int(L) for L in layers) | set(int(L) for L in partial_layers))
    partial_layers = sorted(set(int(L) for L in partial_layers))
    fsets = list(feature_sets or family.feature_sets)
    T = len(family.positions)
    if np.asarray(fit.input_ids).shape[1] != T or np.asarray(evaluate.input_ids).shape[1] != T:
        raise CompiledStateError(f"suffix rows must have {T} positions")
    run = DecoderSplice(model, prefix_ids)
    dev, nL = run.device, run.num_layers
    if partial_layers and max(partial_layers) >= nL:
        raise CompiledStateError("partial arms need at least one layer")
    D = model.config.hidden_size
    vis = torch.as_tensor(visibility(family), device=dev, dtype=torch.float32)
    score = list(family.score_positions)
    kv_pos = [p for p in range(T) if p not in score]
    kv_layers = list(range(min(partial_layers), nL)) if partial_layers else []
    names = [s.name for s in family.symbols]

    def masked(values):
        return torch.as_tensor(design(family, values), device=dev)[:, None, :] * vis[None]

    # ---- fit: native states (and K/V rows for partial arms) -> ridge moments on the device
    moments = _DeviceMoments(family, layers, D, fsets, kv_layers, run.kv_width, dev)
    for st in range(0, len(fit), batch):
        rows = fit.take(slice(st, st + batch))
        _, cap, kvs = run.native(rows.input_ids, capture=layers, capture_kv=kv_layers)
        targets = {**cap, **{("kv", L): kv_rows(*kvs[L]) for L in kv_layers}}
        moments.add(masked(rows.values), targets, torch.as_tensor(np.asarray(rows.values), device=dev))
    fitter = moments.fit()
    del moments
    maps = {(f, L): fitter.solve(f, L, lam) for f in fsets for L in layers}
    maps.update({(f, L, "kv"): fitter.solve(f, L, lam, target="kv") for f in fsets for L in kv_layers})
    t_fit = time.time() - t0
    log(f"fit {len(fit)} rows, {len(maps)} maps, {t_fit:.1f}s")

    # ---- evaluate: resume each map at L, compare top-1 at the score positions
    W = {k: torch.as_tensor(m.weights, device=dev) for k, m in maps.items()}
    n_r2 = len(evaluate) if r2_rows is None else min(int(r2_rows), len(evaluate))
    full = [(f, L) for f in fsets for L in layers]
    acc = {k: ResidualAccumulator(T, D) for k in full}
    arms = full + [(f, L, kind) for f in fsets for L in partial_layers for kind in ("partial", "partial-nativeKV")]
    native_top, comp_top = [], {k: [] for k in arms}
    gates = None
    for st in range(0, len(evaluate), batch):
        rows = evaluate.take(slice(st, st + batch))
        x, cap, kvs = run.native(rows.input_ids, capture=layers, capture_kv=kv_layers)
        native_top.append(run.readout(x[:, score]).argmax(-1).cpu().numpy())
        if st == 0:
            gates = parity(run, family, rows.input_ids[:parity_rows], layers, partial_layers=partial_layers,
                           tol=parity_tol)
        P = masked(rows.values)
        vals = torch.as_tensor(np.asarray(rows.values), device=dev)
        r2_take = max(0, min(len(rows), n_r2 - st))
        compiled = {}
        for f, L in full:
            sw_name = family.feature_sets[f].switch
            grp = vals[:, names.index(sw_name)] if sw_name else None
            xc = compile_state(P, W[(f, L)], grp, family.symbol(sw_name).visible_from if sw_name else 0).to(run.dtype)
            compiled[(f, L)] = xc
            out, _, _ = run.run(xc, L)
            comp_top[(f, L)].append(run.readout(out[:, score]).argmax(-1).cpu().numpy())
            if r2_take:
                mom = batch_residual_moments(cap[L][:r2_take].double(), xc[:r2_take].double())
                acc[(f, L)].add_moments(r2_take, *(m.cpu().numpy() for m in mom))
        for f in fsets:
            for L in partial_layers:
                kv_c = {}
                for layer in kv_layers:
                    if layer >= L:
                        zc = torch.einsum("bpf,pfd->bpd", P[:, kv_pos], W[(f, layer, "kv")][kv_pos])
                        kv_c[layer] = kv_unrows(zc.to(run.dtype), run.kv_heads, run.head_dim)
                kv_n = {layer: (kvs[layer][0][:, :, kv_pos], kvs[layer][1][:, :, kv_pos]) for layer in range(L, nL)}
                xs = compiled[(f, L)][:, score]
                for kind, kv in (("partial", kv_c), ("partial-nativeKV", kv_n)):
                    out, _, _ = run.run(xs, L, positions=score, kv=kv, kv_positions=kv_pos)
                    comp_top[(f, L, kind)].append(run.readout(out).argmax(-1).cpu().numpy())
    native_top = np.concatenate(native_top)
    ref = None if evaluate.reference is None else np.asarray(evaluate.reference)
    # noise ceiling: native against itself with every batch boundary shifted by half a batch
    shifted, cuts = [], [0, *range(max(1, batch // 2), len(evaluate), batch), len(evaluate)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            x, _, _ = run.native(evaluate.take(slice(a, b)).input_ids)
            shifted.append(run.readout(x[:, score]).argmax(-1).cpu().numpy())
    ceiling = agreement(native_top, np.concatenate(shifted), ref)

    by_fset = {}
    for f in fsets:
        rows_out, agree = {}, {}
        for L in layers:
            a = agreement(native_top, np.concatenate(comp_top[(f, L)]), ref)
            agree[L] = a["row_agreement"]
            rows_out[str(L)] = {"agreement": a, "positions": acc[(f, L)].summary(family.positions)}
        entry = {"symbols": list(family.feature_sets[f].symbols), "switch": family.feature_sets[f].switch,
                 "horizon": horizon(agree, threshold), "layers": rows_out,
                 "maps": {str(L): maps[(f, L)].manifest(weights_sha256(maps[(f, L)].weights)) for L in layers}}
        if partial_layers:
            entry["partial"] = {str(L): {
                "running_positions": [family.positions[p] for p in score],
                "injected_positions": [family.positions[p] for p in kv_pos],
                "compiled_kv": agreement(native_top, np.concatenate(comp_top[(f, L, "partial")]), ref),
                "native_kv_oracle": agreement(native_top, np.concatenate(comp_top[(f, L, "partial-nativeKV")]), ref)}
                for L in partial_layers}
            entry["kv_maps"] = {str(L): maps[(f, L, "kv")].manifest(weights_sha256(maps[(f, L, "kv")].weights))
                                for L in kv_layers}
        by_fset[f] = entry
    report = seal_report({
        "family": {**family.to_dict(), "width": family.width}, "model": dict(model_identity or {}),
        "execution": {"device": str(dev), "dtype": str(run.dtype).replace("torch.", ""),
                      "torch": torch.__version__, "prefix_tokens": run.prefix_len,
                      "num_layers": nL, "hidden_size": D, "kv_width": run.kv_width,
                      "splice": "true resume: the compiled state enters layer L and only layers L.. execute; "
                                "L = num_layers goes straight to the final norm and head",
                      "fit_s": round(t_fit, 1), "total_s": round(time.time() - t0, 1),
                      "peak_vram_mb": (round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
                                       if dev.type == "cuda" else None)},
        "lam": float(lam), "threshold": float(threshold), "layers": layers, "partial_layers": partial_layers,
        "rows": {"fit": len(fit), "eval": len(evaluate), "r2": n_r2},
        "native": agreement(native_top, native_top, ref),
        "parity": gates,
        "noise_ceiling": {"definition": "native top-1 vs native top-1 with batch boundaries shifted by batch // 2; "
                                        "error_recall here is the ceiling for any compiler's error recall",
                          **ceiling},
        "feature_sets": by_fset})
    return report, maps


__all__ = ["DecoderSplice", "Rows", "device_features", "kv_rows", "kv_unrows", "measure", "parity"]

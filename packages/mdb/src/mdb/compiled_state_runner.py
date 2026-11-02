"""Native capture and compiled-state splice for Hugging Face causal decoders.

``DecoderSplice`` caches the family prefix K/V once and runs suffix batches
through the unmodified model. A forward pre-hook on decoder layer ``L`` (or on
the final norm for ``L = num_layers``) observes the residual entering that
layer, or replaces it for every suffix position. Layers below ``L`` still run;
nothing at or above ``L`` reads their suffix outputs, so a splice at ``L`` is
exactly "compiled state at L, native layers L.. on top" (Part D/E's T2).

``measure`` fits compiled maps on native states, scores them on held-out rows
and returns a sealed ``mdb-compiled-state-residual-v1`` report plus the maps.
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


class DecoderSplice:
    def __init__(self, model, prefix_ids: Sequence[int]):
        if model.training:
            raise CompiledStateError("compiled-state capture needs a model in eval mode")
        self.model = model
        self.decoder = model.get_decoder()
        self.layers = self.decoder.layers
        self.num_layers = len(self.layers)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.prefix_len = len(prefix_ids)
        with torch.no_grad():
            ids = torch.as_tensor([list(prefix_ids)], device=self.device)
            self._prefix = model(input_ids=ids, use_cache=True).past_key_values

    def _module(self, layer: int):
        if not 0 <= layer <= self.num_layers:
            raise CompiledStateError(f"layer {layer} outside 0..{self.num_layers}")
        return self.layers[layer] if layer < self.num_layers else self.decoder.norm

    @torch.no_grad()
    def forward(self, input_ids, *, capture: Sequence[int] = (), splice: tuple[int, Any] | None = None,
                keep: int = 0):
        """Logits [B, T, V] for the suffix after the cached prefix (the last ``keep``
        positions only when ``keep`` > 0); ``capture`` returns the residual entering
        each listed layer, ``splice=(L, state)`` replaces it."""
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


def measure(model, family: PromptFamily, prefix_ids: Sequence[int], fit: Rows, evaluate: Rows, *,
            layers: Sequence[int], feature_sets: Sequence[str] | None = None, lam: float = 1.0,
            threshold: float = 0.95, batch: int = 256, r2_rows: int | None = None,
            model_identity: Mapping[str, Any] | None = None, log=None):
    """Fit compiled maps on ``fit`` native states and score them on ``evaluate``.
    Returns (report, {(feature_set, layer): CompiledStateMap})."""
    t0 = time.time()
    log = log or (lambda *_: None)
    layers = sorted(set(int(L) for L in layers))
    fsets = list(feature_sets or family.feature_sets)
    T = len(family.positions)
    if np.asarray(fit.input_ids).shape[1] != T or np.asarray(evaluate.input_ids).shape[1] != T:
        raise CompiledStateError(f"suffix rows must have {T} positions")
    run = DecoderSplice(model, prefix_ids)
    dev = run.device
    D = model.config.hidden_size
    vis = torch.as_tensor(visibility(family), device=dev, dtype=torch.float32)
    keep = T - min(family.score_positions)
    score = [p - (T - keep) for p in family.score_positions]  # indices into the kept logits
    names = [s.name for s in family.symbols]

    def masked(values):
        return torch.as_tensor(design(family, values), device=dev)[:, None, :] * vis[None]

    # ---- fit: native states -> ridge moments, accumulated on the device
    F = family.width
    gram = torch.zeros(T, F, F, device=dev, dtype=torch.float64)
    xty = {L: torch.zeros(T, F, D, device=dev) for L in layers}
    switches = sorted({family.feature_sets[f].switch for f in fsets if family.feature_sets[f].switch})
    sw = {s: {g: [0, torch.zeros(T - family.symbol(s).visible_from, F, F, device=dev, dtype=torch.float64),
                  {L: torch.zeros(T - family.symbol(s).visible_from, F, D, device=dev) for L in layers}]
              for g in range(family.symbol(s).cardinality)} for s in switches}
    for st in range(0, len(fit), batch):
        rows = fit.take(slice(st, st + batch))
        _, cap = run.forward(rows.input_ids, capture=layers, keep=1)
        P = masked(rows.values)
        gram += torch.einsum("bpf,bpg->pfg", P, P).double()
        for L in layers:
            xty[L] += torch.einsum("bpf,bpd->pfd", P, cap[L].float())
        for s in switches:
            s0 = family.symbol(s).visible_from
            grp = torch.as_tensor(np.asarray(rows.values)[:, names.index(s)], device=dev)
            for g, acc in sw[s].items():
                m = grp == g
                if m.any():
                    Pg = P[m, s0:]
                    acc[0] += int(m.sum())
                    acc[1] += torch.einsum("bpf,bpg->pfg", Pg, Pg).double()
                    for L in layers:
                        acc[2][L] += torch.einsum("bpf,bpd->pfd", Pg, cap[L][m, s0:].float())
    fitter = CompilerFit(family, layers, D, fsets)
    fitter.add_moments(len(fit), gram.cpu().numpy(), {L: xty[L].cpu().numpy() for L in layers},
                       {s: {g: (n, gg.cpu().numpy(), {L: v.cpu().numpy() for L, v in gx.items()})
                            for g, (n, gg, gx) in groups.items()} for s, groups in sw.items()})
    del gram, xty, sw
    maps = {(f, L): fitter.solve(f, L, lam) for f in fsets for L in layers}
    t_fit = time.time() - t0
    log(f"fit {len(fit)} rows, {len(maps)} maps, {t_fit:.1f}s")

    # ---- evaluate: splice each map, run the remaining layers, compare top-1 at score positions
    W = {k: torch.as_tensor(m.weights, device=dev) for k, m in maps.items()}
    n_r2 = len(evaluate) if r2_rows is None else min(int(r2_rows), len(evaluate))
    acc = {k: ResidualAccumulator(T, D) for k in maps}
    native_top, comp_top, control = [], {k: [] for k in maps}, {}
    for st in range(0, len(evaluate), batch):
        rows = evaluate.take(slice(st, st + batch))
        logits, cap = run.forward(rows.input_ids, capture=layers, keep=keep)
        native_top.append(logits[:, score].argmax(-1).cpu().numpy())
        if st == 0:  # positive control: splicing the captured native state must reproduce native logits
            for L in layers:
                out, _ = run.forward(rows.input_ids, splice=(L, cap[L]), keep=keep)
                control[str(L)] = float((out[:, score].float() - logits[:, score].float()).abs().max())
        P = masked(rows.values)
        r2_take = max(0, min(len(rows), n_r2 - st))
        for (f, L), w in W.items():
            sw_name = family.feature_sets[f].switch
            grp = (torch.as_tensor(np.asarray(rows.values)[:, names.index(sw_name)], device=dev)
                   if sw_name else None)
            x = compile_state(P, w, grp, family.symbol(sw_name).visible_from if sw_name else 0).to(run.dtype)
            out, _ = run.forward(rows.input_ids, splice=(L, x), keep=keep)
            comp_top[(f, L)].append(out[:, score].argmax(-1).cpu().numpy())
            if r2_take:
                mom = batch_residual_moments(cap[L][:r2_take].double(), x[:r2_take].double())
                acc[(f, L)].add_moments(r2_take, *(m.cpu().numpy() for m in mom))
    native_top = np.concatenate(native_top)
    ref = None if evaluate.reference is None else np.asarray(evaluate.reference)
    # noise ceiling: native against itself with every batch boundary shifted by half a batch
    shifted, cuts = [], [0, *range(max(1, batch // 2), len(evaluate), batch), len(evaluate)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b > a:
            logits, _ = run.forward(evaluate.take(slice(a, b)).input_ids, keep=keep)
            shifted.append(logits[:, score].argmax(-1).cpu().numpy())
    ceiling = agreement(native_top, np.concatenate(shifted), ref)

    by_fset = {}
    for f in fsets:
        rows_out, agree = {}, {}
        for L in layers:
            a = agreement(native_top, np.concatenate(comp_top[(f, L)]), ref)
            agree[L] = a["row_agreement"]
            rows_out[str(L)] = {"agreement": a, "positions": acc[(f, L)].summary(family.positions)}
        by_fset[f] = {"symbols": list(family.feature_sets[f].symbols), "switch": family.feature_sets[f].switch,
                      "horizon": horizon(agree, threshold), "layers": rows_out,
                      "maps": {str(L): maps[(f, L)].manifest(weights_sha256(maps[(f, L)].weights))
                               for L in layers}}
    report = seal_report({
        "family": {**family.to_dict(), "width": family.width}, "model": dict(model_identity or {}),
        "execution": {"device": str(dev), "dtype": str(run.dtype).replace("torch.", ""),
                      "torch": torch.__version__, "prefix_tokens": run.prefix_len,
                      "num_layers": run.num_layers, "hidden_size": D,
                      "splice": "pre-hook replaces the residual entering layer L at every suffix position; "
                                "L = num_layers splices the final-norm input",
                      "fit_s": round(t_fit, 1), "total_s": round(time.time() - t0, 1),
                      "peak_vram_mb": (round(torch.cuda.max_memory_allocated(dev) / 2**20, 1)
                                       if dev.type == "cuda" else None)},
        "lam": float(lam), "threshold": float(threshold), "layers": layers,
        "rows": {"fit": len(fit), "eval": len(evaluate), "r2": n_r2},
        "native": agreement(native_top, native_top, ref),
        "native_splice_control": {"rows": min(batch, len(evaluate)), "max_abs_dlogit": control},
        "noise_ceiling": {"definition": "native top-1 vs native top-1 with batch boundaries shifted by batch // 2; "
                                        "error_recall here is the ceiling for any compiler's error recall",
                          **ceiling},
        "feature_sets": by_fset})
    return report, maps


__all__ = ["DecoderSplice", "Rows", "measure"]

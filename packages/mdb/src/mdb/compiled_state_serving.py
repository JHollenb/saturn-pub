"""Forwardless serving: zero-layer compiled predictors with native refresh.

At ``L = num_layers`` a compiled state needs no transformer layer: the compiled
final residual at a score position goes straight to the model's final norm and
output head. ``ZeroLayerChain`` free-runs the family's score positions in order
and feeds each predicted token back into the symbols declared in ``feedback``
before the next step. ``serve`` answers a stream of prompts with the chain and
pays a native forward only when the chain's smallest step margin falls below
``tau``; each native row is added to a streaming ridge fit, the maps are
re-solved every ``refit_every`` native rows, and the record counts native
forwards per 1,000 served prompts. Native tokens supplied for auditing score the
served answers and are never counted or fed to the predictor.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .compiled_state import CompiledStateError, CompiledStateMap, PromptFamily, visibility
from .compiled_state_runner import DecoderSplice, _DeviceMoments, device_features

SERVING_SCHEMA = "mdb-forwardless-serving-v1"


def feedback_tables(family: PromptFamily, feedback: Mapping[str, tuple[int, Any]], device) -> dict[int, tuple]:
    """Validated ``{symbol index: (score step, long table over the vocabulary)}``."""
    names, score = [s.name for s in family.symbols], list(family.score_positions)
    out = {}
    for sym, (step, table) in dict(feedback).items():
        if sym not in names or not 0 <= int(step) < len(score):
            raise CompiledStateError(f"feedback {sym!r} names an unknown symbol or step")
        if family.symbol(sym).visible_from <= score[int(step)]:
            raise CompiledStateError(f"feedback {sym!r} must become visible after the step that writes it")
        out[names.index(sym)] = (int(step), torch.as_tensor(table, device=device).long())
    return out


def apply_feedback(values, tokens, tables: Mapping[int, tuple]):
    """Symbol values [B, S] with every feedback symbol set from ``tokens`` [B, K]."""
    values = values.clone()
    for j, (step, table) in tables.items():
        values[:, j] = table[tokens[:, step]]
    return values


class ZeroLayerChain:
    """``feedback[symbol] = (step, table)``: after score step ``step`` the predicted
    token id ``t`` sets that symbol to ``table[t]`` (a long tensor over the full
    vocabulary; unmapped tokens should map to the symbol's "other" value)."""

    def __init__(self, run: DecoderSplice, cmap: CompiledStateMap, *,
                 feedback: Mapping[str, tuple[int, Any]] = (), vocab: Sequence[int] | None = None):
        fam = cmap.family
        if cmap.layer != run.num_layers or cmap.target != "residual":
            raise CompiledStateError("a zero-layer chain needs a residual map at L = num_layers")
        self.run, self.family, self.vocab = run, fam, None if vocab is None else [int(v) for v in vocab]
        self.names = [s.name for s in fam.symbols]
        self.score = list(fam.score_positions)
        dev = run.device
        self.vis = torch.as_tensor(visibility(fam), device=dev, dtype=torch.float32)
        self.off = torch.as_tensor(list(fam.offsets().values()), device=dev)
        self.W = torch.as_tensor(cmap.weights, device=dev)
        self.switch = cmap.switch
        self.switch_index = self.names.index(self.switch) if self.switch else None
        self.switch_from = fam.symbol(self.switch).visible_from if self.switch else None
        self.feedback = feedback_tables(fam, feedback, dev)

    def _state(self, values, p):
        X = torch.zeros(len(values), self.family.width, device=values.device)
        X[:, 0] = 1
        X.scatter_(1, values.long() + self.off[None], 1.0)
        X = X * self.vis[p]
        if self.switch is None:
            return X @ self.W[p]
        if p < self.switch_from:
            return X @ self.W[0, p]
        x = torch.empty(len(values), self.W.shape[-1], device=values.device)
        grp = values[:, self.switch_index]
        for g in range(self.W.shape[0]):
            m = grp == g
            if m.any():
                x[m] = X[m] @ self.W[g, p]
        return x

    @torch.no_grad()
    def __call__(self, values, teacher=None, *, margins: bool = True):
        """Tokens [B, K] and logit margins (top-1 minus top-2) [B, K] for the score
        steps; ``teacher`` [B, K] feeds given tokens back instead of the chain's own.
        ``margins=False`` takes a plain argmax and returns None for the margins."""
        values = torch.as_tensor(values, device=self.run.device).long().clone()
        toks, gaps = [], []
        for k, p in enumerate(self.score):
            logits = self.run.readout(self._state(values, p).to(self.run.dtype), self.vocab)
            if margins:
                top = logits.float().topk(2, -1)
                idx = top.indices[:, 0]
                gaps.append(top.values[:, 0] - top.values[:, 1])
            else:
                idx = logits.argmax(-1)
            tok = idx if self.vocab is None else self.run._vocab_rows[tuple(self.vocab)][1][idx]
            toks.append(tok)
            fed = tok if teacher is None else torch.as_tensor(teacher, device=values.device)[:, k]
            for j, (step, table) in self.feedback.items():
                if step == k:
                    values[:, j] = table[fed]
        return torch.stack(toks, 1), torch.stack(gaps, 1) if margins else None


def chain_agreement(chain: ZeroLayerChain, values, native_tokens, batch: int = 256) -> dict[str, Any]:
    """Free-running and teacher-forced agreement of a chain with native greedy tokens."""
    free, tf = [], []
    for st in range(0, len(values), batch):
        free.append(chain(values[st:st + batch], margins=False)[0])
        tf.append(chain(values[st:st + batch], teacher=native_tokens[st:st + batch], margins=False)[0])
    nat = torch.as_tensor(native_tokens, device=chain.run.device)
    same, same_tf = (torch.cat(free) == nat).cpu().numpy(), (torch.cat(tf) == nat).cpu().numpy()
    K = same.shape[1]
    return {"n": int(len(same)), "steps": K,
            "exact_through": [float(same[:, :k + 1].all(1).mean()) for k in range(K)],
            "step_agreement_free": same.mean(0).tolist(), "step_agreement_teacher_forced": same_tf.mean(0).tolist()}


@torch.no_grad()
def serve(run: DecoderSplice, family: PromptFamily, feature_set: str, template_ids, values, *, steps: int,
          feedback: Mapping[str, tuple[int, Any]], tau: float | None, warmup: int, refit_every: int,
          batch: int = 256, lam: float = 1.0, vocab: Sequence[int] | None = None, audit_tokens=None,
          window: int = 1000, log=None) -> dict[str, Any]:
    """Serve every prompt of the stream in order. ``template_ids`` [N, T0] are the
    suffix tokens before generation and ``values`` [N, S] the family symbol values
    (feedback symbols are overwritten). The first ``warmup`` prompts are served
    natively; afterwards a batch is predicted by the chain and every row whose
    smallest step margin is below ``tau`` (never, if ``tau`` is None) is served by
    a full-vocabulary native KV-cached decode instead (``vocab`` restricts only the
    chain's readout). ``audit_tokens`` [N, steps], native greedy tokens computed
    separately, only score the served answers."""
    t0 = time.time()
    log = log or (lambda *_: None)
    nL, dev = run.num_layers, run.device
    ids = torch.as_tensor(np.asarray(template_ids), device=dev)
    vals = torch.as_tensor(np.asarray(values), device=dev).long()
    N, T0 = ids.shape
    if T0 + steps - 1 != len(family.positions) or len(family.score_positions) != steps:
        raise CompiledStateError("the family must cover the template plus steps - 1 fed-back positions")
    if warmup < 1 or refit_every < 1:
        raise CompiledStateError("serving needs at least one warmup row and a positive refit interval")
    D = run.model.config.hidden_size
    vis = torch.as_tensor(visibility(family), device=dev, dtype=torch.float32)
    moments = _DeviceMoments(family, [nL], D, [feature_set], device=dev)
    tables = feedback_tables(family, feedback, dev)

    def native_rows(r):
        toks, finals = run.decode(ids[r], steps)
        v = apply_feedback(vals[r], toks, tables)
        moments.add(device_features(family, v, vis), {nL: finals}, v)
        return toks

    served = torch.zeros(N, steps, dtype=torch.long, device=dev)
    native_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    refits, since, chain = 0, 0, None

    def refit():
        nonlocal chain, refits, since
        chain = ZeroLayerChain(run, moments.fit().solve(feature_set, nL, lam), feedback=feedback, vocab=vocab)
        refits, since = refits + 1, 0

    w = min(warmup, N)
    for st in range(0, w, batch):
        r = torch.arange(st, min(st + batch, w), device=dev)
        served[r] = native_rows(r)
    native_mask[:w] = True
    refit()
    for st in range(w, N, batch):
        r = torch.arange(st, min(st + batch, N), device=dev)
        toks, margins = chain(vals[r])
        served[r] = toks
        low = r[margins.min(1).values < tau] if tau is not None else r[:0]
        if len(low):
            served[low] = native_rows(low)
            native_mask[low] = True
            since += len(low)
        if since >= refit_every:
            refit()
    n_native = int(native_mask.sum())
    out = {"schema": SERVING_SCHEMA, "family": family.fingerprint, "feature_set": feature_set, "tau": tau,
           "warmup": int(w), "refit_every": refit_every, "lam": lam, "steps": steps, "served": int(N),
           "native_prompts": n_native, "native_token_steps": n_native * steps,
           "forwards_per_1000_prompts": 1000.0 * n_native / N, "refits": refits,
           "margin": "per-step top-1 minus top-2 logit; a row refreshes when its smallest step margin < tau",
           "elapsed_s": round(time.time() - t0, 2)}
    if audit_tokens is not None:
        audit = torch.as_tensor(np.asarray(audit_tokens), device=dev)
        exact = (served == audit).all(1)
        pred = ~native_mask
        out["audit"] = {
            "served_exact": float(exact.float().mean()),
            "predicted_rows": int(pred.sum()),
            "predicted_exact": float(exact[pred].float().mean()) if pred.any() else None,
            "native_rows_exact": float(exact[native_mask].float().mean()) if native_mask.any() else None,
            "windows": [{"start": s, "prompts": int(min(window, N - s)),
                         "native_prompts": int(native_mask[s:s + window].sum()),
                         "served_exact": float(exact[s:s + window].float().mean())}
                        for s in range(0, N, window)],
            "definition": "served answer equals native KV-cached greedy tokens at every step"}
    log(f"serve tau={tau}: {n_native}/{N} native, {out['forwards_per_1000_prompts']:.1f} per 1k")
    return out


__all__ = ["SERVING_SCHEMA", "ZeroLayerChain", "apply_feedback", "chain_agreement", "feedback_tables", "serve"]

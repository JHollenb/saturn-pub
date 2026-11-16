"""Forced-answer probes along a stored chain: ``suffix_sweep``.

``suffix_sweep(run, steps, suffix, n_prompt=..., bindings=..., letters=..., X=..., hint=...)``
reads, for every probe step ``k`` in ``steps``, the sequence ``run.tokens[:n_prompt + k] +
suffix`` at its answer row (the last suffix token), natively, without re-encoding the prefix:
the prefix cache of ``run.tokens[:n_prompt + max(steps)]`` is built once (one native forward),
and for each ``k`` (largest first) the cache is cropped to ``n_prompt + k`` rows and only the
suffix is run. Arms, at the answer row (all measured, FP32 eager, letter logits ``W[ids] . h``):

* ``native``: the suffix as is;
* ``M2``: the answer row's attention weights on the ``hint`` rows zeroed at every layer and head,
  no renormalisation (= V cells of those rows zeroed for the answer row's pass only);
* ``CM1``: the same for the chain rows read so far (``n_prompt .. n_prompt + k - 1``);
* ``M3``: the hint rows' V cells zeroed for every suffix row (the suffix re-run).

The native answer row also yields, with hooks on the attention weights and the cached V cells,
the exact per-binding ``to_consumer`` and ``mass`` of the consumer ``M[X, R]`` (as
``rows.row_table`` at one row; bindings: the given prompt bindings plus ``chain_so_far`` and
``suffix``) and ``share = to_consumer / margin``. ``R`` is fixed for the whole sweep: given, or
the strongest letter other than ``X`` at the largest step's native answer row.

``null_steps``: at those steps the native answer letters are also read from a fresh forward of
the whole probe sequence (no cache) and ``null_vs_fresh_max_abs`` reports the largest letter
logit difference (the cache-reuse check).
"""

from __future__ import annotations

import time
from typing import Any

from .forecast import Refusal
from .rows import (LETTERS, Trace, _check_partition, _masks, consumer_backs, decoder_parts,
                   head_groups, letter_logits, letter_setup, rho_of)

ARMS = ("native", "M2", "M3", "CM1")


def suffix_sweep(run: Any, steps: list[int], suffix: list[int], *, n_prompt: int,
                 bindings: dict[str, list[int]], letters: Any, X: str, R: str | None = None,
                 hint: str = "hint", arms: tuple[str, ...] = ARMS,
                 null_steps: tuple[int, ...] = ()) -> Trace:
    """Forced-answer probe trace over ``steps`` (module doc)."""
    import torch

    from mdb.engine.ports import check_deterministic

    model = run.adapter.model
    check_deterministic(model)
    base, layers, norm = decoder_parts(run)
    ids, names = letter_setup(run, letters, X, R)
    xi = names.index(X)
    bad = [a for a in arms if a not in ARMS]
    if bad or "native" not in arms:
        raise Refusal(f"arms are a subset of {', '.join(ARMS)} that includes native")
    tokens = list(run.tokens)
    chain_len = len(tokens) - n_prompt
    steps = sorted({int(k) for k in steps}, reverse=True)
    if not steps or steps[-1] < 0 or steps[0] > chain_len:
        raise Refusal(f"probe steps must lie in 0..{chain_len}")
    suffix = [int(t) for t in suffix]
    if not suffix:
        raise Refusal("a probe needs a suffix of at least one token")
    binds = {b: sorted(int(r) for r in rows) for b, rows in bindings.items()}
    _check_partition(binds, n_prompt)
    if ("M2" in arms or "M3" in arms) and (hint not in binds or not binds[hint]):
        raise Refusal(f"no {hint!r} binding to ablate")
    device = next(model.parameters()).device
    H, G, dh = head_groups(run)
    per = H // G
    S = len(suffix)
    hint_index = torch.as_tensor(binds.get(hint, []), device=device, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        cache = base(input_ids=torch.tensor([tokens[: n_prompt + steps[0]]], device=device),
                     use_cache=True).past_key_values
    t_prefix = time.time() - t0

    def run_tokens(toks: list[int], start: int) -> Any:
        pos = torch.arange(start, start + len(toks), device=device)[None]
        out = base(input_ids=torch.tensor([toks], device=device), past_key_values=cache,
                   position_ids=pos, use_cache=True)
        return out.last_hidden_state[0, -1]

    def normed(h: Any) -> Any:
        return h  # last_hidden_state is the final norm's output already

    def zeroed(index: Any):
        """Zero V cells of ``index`` in the cache now; returns the restorer."""
        saved = [layer.values[:, :, index].clone() for layer in cache.layers]
        for layer in cache.layers:
            layer.values[:, :, index] = 0

        def restore() -> None:
            for layer, keep in zip(cache.layers, saved):
                layer.values[:, :, index] = keep
        return restore

    def shares_pass(toks_last: int, position: int, backs: list[Any], masks: Any) -> tuple:
        """The native answer row with hooks: letters, per-binding contrib and mass, total."""
        store: dict[str, Any] = {}
        nb = masks.shape[0]
        contrib = torch.zeros(nb, dtype=torch.float64, device=device)
        mass = torch.zeros(nb, dtype=torch.float64, device=device)
        total = torch.zeros((), dtype=torch.float64, device=device)
        hooks = []

        def a_hook(mod, inp, out, li):
            nonlocal total
            p = out[1]
            if p is None:
                raise Refusal("the attention module returned no weights (eager attention needed)")
            p = p[0, :, -1, :].double()                                  # (H, cols)
            v = cache.layers[li].values[0].double()                      # (G, cols, dh)
            vb = torch.stack([v[h // per] @ backs[li][h] for h in range(H)])  # (H, cols)
            pv = p * vb
            contrib.add_(masks @ pv.sum(0))
            mass.add_(masks @ p.sum(0))
            total = total + pv.sum()

        def n_hook(mod, inp, out):
            store["pre"] = inp[0][0, -1]

        try:
            for li, layer in enumerate(layers):
                hooks.append(layer.self_attn.register_forward_hook(
                    lambda m, i, o, li=li: a_hook(m, i, o, li)))
            hooks.append(norm.register_forward_hook(n_hook))
            h = run_tokens([toks_last], position)
        finally:
            for hook in hooks:
                hook.remove()
        rho = rho_of(run, store["pre"])
        return letter_logits(run, normed(h)[None], ids)[0], contrib * rho, mass, total * rho

    if R is None:
        with torch.no_grad():
            n = n_prompt + steps[0]
            cache.crop(n)
            h = run_tokens(suffix, n)
            top = letter_logits(run, normed(h)[None], ids)[0]
            cache.crop(n)
        R = names[max((j for j in range(len(names)) if j != xi), key=lambda j: float(top[j]))]
    ri = names.index(R)
    backs = consumer_backs(run, ids[xi], ids[ri])
    records = []
    null_diffs = {}
    with torch.no_grad():
        for k in steps:
            ts = time.time()
            n = n_prompt + k
            cache.crop(n)
            letters_of: dict[str, Any] = {}
            if "M3" in arms:
                restore = zeroed(hint_index)
                h = run_tokens(suffix, n)
                letters_of["M3"] = letter_logits(run, normed(h)[None], ids)[0]
                cache.crop(n)
                restore()
            if S > 1:
                run_tokens(suffix[:-1], n)
            last = n + S - 1
            cols = last + 1
            binds_k = dict(binds)
            binds_k["chain_so_far"] = list(range(n_prompt, n))
            binds_k["suffix"] = list(range(n, cols))
            masks = _masks(binds_k, cols, device)
            nat, contrib, mass, total = shares_pass(suffix[-1], last, backs, masks)
            letters_of["native"] = nat
            cache.crop(last)
            for arm, index in (("M2", hint_index),
                               ("CM1", torch.arange(n_prompt, n, device=device))):
                if arm not in arms:
                    continue
                if arm == "CM1" and k == 0:
                    letters_of[arm] = nat  # no chain rows: the arm is the native pass
                    continue
                restore = zeroed(index)
                h = run_tokens([suffix[-1]], last)
                letters_of[arm] = letter_logits(run, normed(h)[None], ids)[0]
                cache.crop(last)
                restore()
            if k in null_steps:
                seq = tokens[:n] + suffix
                fresh = base(input_ids=torch.tensor([seq], device=device)).last_hidden_state[0, -1]
                null_diffs[k] = float((letter_logits(run, normed(fresh)[None], ids)[0]
                                       - nat).abs().max())
            margin = float(nat[xi] - nat[ri])
            tc = {b: float(contrib[i]) for i, b in enumerate(binds_k)}
            arm_name = {"M2": hint, "CM1": "chain_so_far", "M3": hint + "_suffix"}
            records.append({
                "step": k,
                "frac": (k / chain_len) if chain_len else 1.0,
                "letters": {a: [float(v) for v in lg] for a, lg in letters_of.items()},
                "margin": margin,
                "argmax_letter": names[int(nat.argmax())],
                "to_consumer": tc,
                "mass": {b: float(mass[i]) for i, b in enumerate(binds_k)},
                "share": {b: (v / margin if margin != 0 else None) for b, v in tc.items()},
                "attn_total": float(total),
                "ablated": {arm_name[a]: float(lg[xi] - lg[ri])
                            for a, lg in letters_of.items() if a != "native"},
                "ablated_argmax": {arm_name[a]: names[int(lg.argmax())]
                                   for a, lg in letters_of.items() if a != "native"},
                "seconds": round(time.time() - ts, 4),
            })
    del cache
    records.reverse()  # probe order: k ascending
    meta = {
        "verb": "suffix_sweep", "n_prompt": n_prompt, "chain_len": chain_len, "suffix": suffix,
        "X": X, "R": R, "letters": ids, "letter_names": names, "arms": list(arms),
        "arm_tables": {"M2": hint, "CM1": "chain_so_far", "M3": hint + "_suffix"},
        "labels": {"margin": "measured", "letters": "measured", "ablated": "measured",
                   "mass": "measured", "to_consumer": "derived-exact", "share": "derived-exact"},
        "null_vs_fresh_max_abs": (max(null_diffs.values()) if null_diffs else None),
        "null_vs_fresh": {str(k): v for k, v in null_diffs.items()},
        "seconds": {"prefix": round(t_prefix, 3), "total": round(time.time() - t0, 3)},
    }
    return Trace("probe", records, meta)


SPAN_MODES = ("mask", "zero", "mask_last")


def span_mask(run: Any, spans: dict[str, list[int]], suffix: list[int], *, letters: Any, X: str,
              modes: tuple[str, ...] = SPAN_MODES, batch: int = 32,
              fresh_check: bool = True) -> Trace:
    """Answer-step span masking over a stored sequence, all arms batched.

    The answer step is ``run.tokens + suffix``; its answer row is the last suffix row. The
    prefix cache of ``run.tokens`` is built once; every (span, mode) arm then runs only the
    suffix, ``batch`` arms per forward, against a batch copy of that cache. Modes, at every
    layer and head (measured, letter logits ``W[ids] . h`` at the answer row):

    * ``mask``: every suffix row's attention scores on the span's rows set to the dtype minimum
      before the softmax (the weights renormalise over the other columns);
    * ``zero``: every suffix row's weights on the span's rows zeroed after the softmax, no
      renormalisation (= the span's V cells zeroed in the batch copy);
    * ``mask_last``: ``mask`` for the answer row only (earlier suffix rows read natively).

    Native answer row (one hooked pass): per span ``mass`` (sum over layers and heads of the
    attention weight, divided by layers x heads) and ``vw_mass`` (sum of ``p[h, s] * ||W_O,h
    v_s||`` over layers, heads and the span's rows, divided by the same sum over all columns:
    the value-weighted read share). ``margin`` is ``logit[X] - max other letter``; ``delta`` per
    mode is ``native margin - arm margin`` (positive: the span supported ``X``). A null arm (no
    rows masked) checks the batched path (``null_max_abs``); ``fresh_check`` compares the native
    letters with one cache-free forward of the whole sequence (``fresh_max_abs``).
    """
    import copy

    import torch

    from mdb.engine.ports import check_deterministic

    model = run.adapter.model
    check_deterministic(model)
    base, layers, _ = decoder_parts(run)
    ids, names = letter_setup(run, letters, X, None)
    xi = names.index(X)
    bad = [m for m in modes if m not in SPAN_MODES]
    if bad or not modes:
        raise Refusal(f"modes are a non-empty subset of {', '.join(SPAN_MODES)}")
    tokens = list(run.tokens)
    T, S = len(tokens), len(suffix)
    if not S:
        raise Refusal("the answer step needs a suffix of at least one token")
    spans = {str(n): sorted({int(r) for r in rows}) for n, rows in spans.items()}
    for n, rows in spans.items():
        if not rows or rows[0] < 0 or rows[-1] >= T:
            raise Refusal(f"span {n} must name rows in 0..{T - 1}")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    neg = torch.finfo(dtype).min
    H, G, dh = head_groups(run)
    per = H // G
    L = len(layers)
    t0 = time.time()

    def margin_of(lg: Any) -> float:
        others = [float(lg[j]) for j in range(len(names)) if j != xi]
        return float(lg[xi]) - max(others)

    pos = torch.arange(T, T + S, device=device)
    with torch.no_grad():
        cache = base(input_ids=torch.tensor([tokens], device=device), use_cache=True).past_key_values
        caps: dict[int, Any] = {}
        hooks = []
        try:
            for li, layer in enumerate(layers):
                hooks.append(layer.self_attn.register_forward_hook(
                    lambda m, i, o, li=li: caps.__setitem__(li, o[1][0, :, -1, :].double())))
            h = base(input_ids=torch.tensor([suffix], device=device), past_key_values=cache,
                     position_ids=pos[None], use_cache=True).last_hidden_state[0, -1]
        finally:
            for hook in hooks:
                hook.remove()
        if len(caps) != L or any(c is None for c in caps.values()):
            raise Refusal("the attention modules returned no weights (eager attention needed)")
        native = letter_logits(run, h[None], ids)[0]
        cols = T + S
        vw = torch.zeros(cols, dtype=torch.float64, device=device)
        mass = torch.zeros(cols, dtype=torch.float64, device=device)
        for li, layer in enumerate(layers):
            v = cache.layers[li].values[0]                              # (G, cols, dh)
            wo = layer.self_attn.o_proj.weight                          # (hidden, H * dh)
            p = caps.pop(li)                                            # (H, cols)
            for hh in range(H):
                w = wo[:, hh * dh:(hh + 1) * dh].double()               # slice, then cast once
                gram = w.T @ w
                vg = v[hh // per].double()
                norms = torch.einsum("cd,de,ce->c", vg, gram, vg).clamp_min(0).sqrt()
                vw.add_(p[hh] * norms)
                mass.add_(p[hh])
                del w, gram, vg, norms
        vw_total = float(vw.sum())
        cache.crop(T)
        t_native = time.time() - t0

        arms = [(n, m) for n in spans for m in modes] + [(None, "null")]
        causal = torch.zeros(S, T + S, dtype=dtype, device=device)
        ii = torch.arange(S, device=device)
        causal[:, T:] = torch.where(ii[None, :] <= ii[:, None], 0.0, neg)
        results: dict[tuple, Any] = {}
        for a in range(0, len(arms), batch):
            sel = arms[a:a + batch]
            B = len(sel)
            bc = copy.deepcopy(cache)
            for layer in bc.layers:
                layer.keys = layer.keys.repeat(B, 1, 1, 1)
                layer.values = layer.values.repeat(B, 1, 1, 1)
            mask = causal[None, None].repeat(B, 1, 1, 1)
            for b, (n, m) in enumerate(sel):
                if n is None:
                    continue
                index = torch.as_tensor(spans[n], device=device)
                if m == "mask":
                    mask[b, 0][:, index] = neg
                elif m == "mask_last":
                    mask[b, 0, S - 1, index] = neg
                else:
                    for layer in bc.layers:
                        layer.values[b, :, index] = 0
            out = base(input_ids=torch.tensor([suffix] * B, device=device), past_key_values=bc,
                       attention_mask=mask, position_ids=pos[None].expand(B, S), use_cache=True)
            lg = letter_logits(run, out.last_hidden_state[:, -1], ids)
            for b, arm in enumerate(sel):
                results[arm] = lg[b]
            del bc, out, mask, lg
        null = results.pop((None, "null"))
        fresh = None
        if fresh_check:
            hf = base(input_ids=torch.tensor([tokens + list(suffix)], device=device)).last_hidden_state[0, -1]
            fresh = float((letter_logits(run, hf[None], ids)[0] - native).abs().max())
    del cache
    nm = margin_of(native)
    records = []
    for n, rows in spans.items():
        lgs = {m: results[(n, m)] for m in modes}
        records.append({
            "step": n, "rows": len(rows), "first_row": rows[0], "last_row": rows[-1],
            "letters": {m: [float(x) for x in lg] for m, lg in lgs.items()},
            "margin": {m: margin_of(lg) for m, lg in lgs.items()},
            "delta": {m: nm - margin_of(lg) for m, lg in lgs.items()},
            "argmax": {m: names[int(lg.argmax())] for m, lg in lgs.items()},
            "mass": float(mass[rows].sum()) / (L * H),
            "vw_mass": (float(vw[rows].sum()) / vw_total) if vw_total else None,
        })
    meta = {
        "verb": "span_mask", "T": T, "suffix": list(suffix), "X": X, "letters": ids,
        "letter_names": names, "modes": list(modes), "batch": batch,
        "native_letters": [float(x) for x in native], "native_margin": nm,
        "native_argmax": names[int(native.argmax())],
        "null_max_abs": float((null - native).abs().max()), "fresh_max_abs": fresh,
        "labels": {"letters": "measured", "margin": "measured", "delta": "measured",
                   "mass": "measured", "vw_mass": "measured"},
        "seconds": {"native": round(t_native, 3), "total": round(time.time() - t0, 3)},
    }
    return Trace("span_mask", records, meta)


__all__ = ["ARMS", "LETTERS", "SPAN_MODES", "span_mask", "suffix_sweep"]

"""All-rows reads: the per-row symbol accounting of one native forward, as a ``Trace``.

``row_table(run, bindings, letters=..., X=..., R=...)`` runs the run's whole token sequence once,
natively (FP32 eager, no cache), with hooks on every layer's ``v_proj`` output and attention
weights and on the final norm's input, and returns for every reader row ``t``:

* ``margin``: the native letter margin ``M[X,R] = logits_t[X] - logits_t[R]`` at row ``t`` (the
  next-token lens; measured), ``letters`` (the letter logits) and ``argmax_letter``;
* per binding ``B``: ``mass[B]`` = sum over layers and heads of ``p[h, t, s]`` over ``s`` in
  ``B`` (the ``symbols`` mass), and ``to_consumer[B]`` = the exact direct contribution of row
  ``t``'s reads of ``B`` to the margin through the frozen final norm (P06/P07/P19 arithmetic of
  ``symbols._Grid``, vectorised over rows): ``rho_t * sum_{l,h,s in B} p[h,t,s] (v_s W_O,h^T
  (gamma * (W_X - W_R)))``, with ``rho_t`` read from the final norm's input (derived-exact from
  measured captures); ``share[B] = to_consumer[B] / margin``;
* per arm (``arms={"hint": cols, ...}``): ``ablated[arm]``, the native margin of row ``t`` when
  row ``t``'s attention weights on the arm's columns are zeroed at every layer and head (no
  renormalisation) for every row at once, each row's own read only (row ``t``'s inputs, the
  cache rows ``< t``, are native). Measured with one masked forward per arm against the native
  cache with the arm columns' V cells zeroed (zeroing ``p[h, t, s]`` and zeroing ``v_s`` give
  the same row-``t`` output when only row ``t`` reads the changed cells); chunked over query
  rows (``chunk``) to bound memory.

The bindings are whole-row column sets (``{name: [rows]}``); with ``bindings=None`` the run's
declared symbols are used (whole-row cells only). ``check=True`` also builds ``symbols.table``
at the run's last row (moving the run's cursor to ``t<last>.logits`` and declaring the bindings
there) and reports ``trace_check_max_abs``, the largest ``|to_consumer|`` difference.

This is the all-rows ``symbols`` that the prefilled-row refusal blocks: rows before
``first_row(run)`` are read from one native forward instead of the run's stepped trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .forecast import Margin, Refusal

LETTERS = "ABCDEFGH"


def crop_to(cache: Any, n: int) -> None:
    """Truncate a ``DynamicCache`` to ``n`` positions. transformers < 5.2x takes an absolute
    length in ``crop``; newer releases reject positive values and take a negative count to
    remove. A negative count is accepted by both, and growing is a no-op in both."""
    drop = cache.get_seq_length() - n
    if drop > 0:
        cache.crop(-drop)


@dataclass
class Trace:
    """A per-step (or per-row) readout table: ``records`` in order, each a dict with ``step``,
    ``margin``, ``argmax_letter``, ``letters`` and the tables ``to_consumer``, ``mass``,
    ``share`` and ``ablated`` (``{binding or arm: float}``); ``meta`` says how it was made
    (labels, checks, timings). ``breakpoints.replay`` evaluates watch predicates over it.
    ``seal`` makes it a ``trace`` record; ``to_frame`` a long DataFrame."""

    kind: str
    records: list[dict[str, Any]]
    meta: dict[str, Any] = field(default_factory=dict)
    # the checkpoint the traced run is rooted at (``Run.from_checkpoint``): its seal carries it
    from_checkpoint: dict[str, Any] | None = field(default=None, compare=False)

    def column(self, table: str, key: str | None = None) -> list[Any]:
        if key is None:
            return [r.get(table) for r in self.records]
        return [(r.get(table) or {}).get(key) for r in self.records]

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "meta": self.meta, "records": self.records}

    def seal(self, *, session_id: str, run: Any = None, store: Any = None,
             **identity: Any) -> dict[str, Any]:
        """This trace as a sealed ``trace`` record (``analysis.sealed.seal_trace``): the step
        records in a sha256-checked blob (``store``: a project or its store), so "sealed
        before scoring" is a record, not a file time (issue C3)."""
        from .sealed import seal_trace

        return seal_trace(self, session_id=session_id, run=run, store=store, **identity)

    def to_frame(self, table: str | None = None) -> Any:
        """The records as a long pandas DataFrame (``analysis.frame.trace_frame``)."""
        from .frame import trace_frame

        return trace_frame(self, table)


# --------------------------------------------------------------------------- shared helpers


def _torch():
    import torch
    return torch


def decoder_parts(run: Any) -> tuple[Any, Any, Any]:
    """``(base, layers, final_norm)`` of a Qwen2/Llama-family causal LM, refusing others."""
    model = run.adapter.model
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "layers") or not hasattr(base, "norm") \
            or not hasattr(model, "lm_head"):
        raise Refusal("all-rows reads need a decoder with model.layers, model.norm and lm_head")
    impl = getattr(model.config, "_attn_implementation", "eager")
    if impl != "eager":
        raise Refusal(f"all-rows reads need eager attention (weights), not {impl!r}")
    return base, base.layers, base.norm


def letter_setup(run: Any, letters: Any, X: str, R: str | None) -> tuple[list[int], list[str]]:
    ids = [int(t) for t in letters]
    names = list(LETTERS[: len(ids)])
    if X not in names or (R is not None and (R not in names or R == X)):
        raise Refusal(f"X and R must be two different letters of {''.join(names)}")
    return ids, names


def consumer_backs(run: Any, a: int, b: int) -> list[Any]:
    """Per layer, ``(H, dh)`` float64: ``W_O,h^T (gamma * (W_a - W_b))`` for each head."""
    torch = _torch()
    model = run.adapter.model
    cfg = model.config
    _, layers, norm = decoder_parts(run)
    H = cfg.num_attention_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    w = model.lm_head.weight
    gu = norm.weight.double() * (w[a].double() - w[b].double())
    backs = []
    for layer in layers:
        wo = layer.self_attn.o_proj.weight.double().reshape(cfg.hidden_size, H, dh)
        backs.append(torch.einsum("dhe,d->he", wo, gu))
    return backs


def head_groups(run: Any) -> tuple[int, int, int]:
    cfg = run.adapter.model.config
    H, G = cfg.num_attention_heads, cfg.num_key_value_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    return H, G, dh


def rho_of(run: Any, pre_norm: Any) -> Any:
    """RMSNorm scale ``rho = (mean(h^2) + eps)^-1/2`` of final-norm inputs (float64)."""
    _, _, norm = decoder_parts(run)
    eps = getattr(norm, "variance_epsilon", None) or getattr(norm, "eps")
    return (pre_norm.double().pow(2).mean(-1) + eps) ** -0.5


def letter_logits(run: Any, normed: Any, ids: list[int]) -> Any:
    """Letter logits ``W[ids] . h`` of final-norm outputs (rows x letters), float64."""
    w = run.adapter.model.lm_head.weight[ids]
    return (normed @ w.T).double()


def _masks(bindings: dict[str, list[int]], T: int, device: Any) -> Any:
    torch = _torch()
    m = torch.zeros(len(bindings), T, dtype=torch.float64, device=device)
    for i, rows in enumerate(bindings.values()):
        if rows:
            m[i, torch.as_tensor(rows, device=device)] = 1.0
    return m


def _bindings_of(run: Any, bindings: dict[str, list[int]] | None) -> dict[str, list[int]]:
    if bindings is not None:
        return {name: sorted(int(r) for r in rows) for name, rows in bindings.items()}
    from . import symbols
    out = {}
    for name, binding in symbols.declared(run).items():
        rows = sorted({c.row for c in binding.cells})
        if not all(binding.whole(r) for r in rows):
            raise Refusal(f"symbol {name} holds partial rows; all-rows reads take whole rows")
        out[name] = rows
    if not out:
        raise Refusal("no bindings: pass bindings= or declare symbols first")
    return out


def _check_partition(bindings: dict[str, list[int]], T: int) -> None:
    seen: dict[int, str] = {}
    for name, rows in bindings.items():
        for r in rows:
            if not 0 <= r < T:
                raise Refusal(f"binding {name} names row {r}, outside 0..{T - 1}")
            if r in seen:
                raise Refusal(f"bindings {seen[r]} and {name} overlap at row {r}")
            seen[r] = name


# --------------------------------------------------------------------------- row_table


def row_table(run: Any, bindings: dict[str, list[int]] | None = None, *, letters: Any,
              X: str, R: str, arms: dict[str, list[int]] | None = None, chunk: int = 512,
              check: bool = False) -> Trace:
    """Per-row symbol accounting and per-row arm ablations over ``run.tokens`` (module doc)."""
    torch = _torch()
    import time

    from mdb.engine.ports import check_deterministic

    model = run.adapter.model
    check_deterministic(model)
    base, layers, norm = decoder_parts(run)
    ids, names = letter_setup(run, letters, X, R)
    xi, ri = names.index(X), names.index(R)
    tokens = list(run.tokens)
    T = len(tokens)
    binds = _bindings_of(run, bindings)
    _check_partition(binds, T)
    arms = {k: sorted(int(r) for r in v) for k, v in (arms or {}).items()}
    device = next(model.parameters()).device
    H, G, dh = head_groups(run)
    per = H // G
    backs = consumer_backs(run, ids[xi], ids[ri])
    masks = _masks(binds, T, device)
    contrib = torch.zeros(len(binds), T, dtype=torch.float64, device=device)
    mass = torch.zeros(len(binds), T, dtype=torch.float64, device=device)
    total = torch.zeros(T, dtype=torch.float64, device=device)
    store: dict[Any, Any] = {}
    hooks = []
    t0 = time.time()

    def v_hook(mod, inp, out, li):
        store[li] = out[0].double().reshape(T, G, dh)

    def a_hook(mod, inp, out, li):
        p = out[1]
        if p is None:
            raise Refusal("the attention module returned no weights (eager attention needed)")
        v = store.pop(li)                                            # (T, G, dh)
        for h in range(H):  # one head at a time: float64 (T, T) blocks, bounded memory
            ph = p[0, h].double()                                    # (T, T)
            vb = v[:, h // per, :] @ backs[li][h]                    # (T,)
            pv = ph * vb[None, :]
            contrib.add_((pv @ masks.T).T)
            total.add_(pv.sum(1))
            mass.add_((ph @ masks.T).T)
            del ph, pv

    def n_hook(mod, inp, out):
        store["pre"] = inp[0][0]
        store["post"] = out[0]

    try:
        for li, layer in enumerate(layers):
            hooks.append(layer.self_attn.v_proj.register_forward_hook(
                lambda m, i, o, li=li: v_hook(m, i, o, li)))
            hooks.append(layer.self_attn.register_forward_hook(
                lambda m, i, o, li=li: a_hook(m, i, o, li)))
        hooks.append(norm.register_forward_hook(n_hook))
        with torch.no_grad():
            out = base(input_ids=torch.tensor([tokens], device=device), use_cache=bool(arms))
    finally:
        for hook in hooks:
            hook.remove()
    rho = rho_of(run, store.pop("pre"))
    native = letter_logits(run, store.pop("post"), ids)              # (T, letters)
    contrib *= rho[None]
    total *= rho
    t_native = time.time() - t0
    ablated: dict[str, Any] = {}
    if arms:
        cache = out.past_key_values
        for arm, cols in arms.items():
            ablated[arm] = _row_ablation(run, cache, tokens, cols, ids, chunk)
    del out
    native = native.detach()
    margin = native[:, xi] - native[:, ri]
    records = []
    names_b = list(binds)
    for t in range(T):
        m = float(margin[t])
        tc = {b: float(contrib[i, t]) for i, b in enumerate(names_b)}
        rec = {
            "step": t,
            "letters": [float(v) for v in native[t]],
            "margin": m,
            "argmax_letter": names[int(native[t].argmax())],
            "to_consumer": tc,
            "mass": {b: float(mass[i, t]) for i, b in enumerate(names_b)},
            "share": {b: (v / m if m != 0 else None) for b, v in tc.items()},
            "attn_total": float(total[t]),
            "ablated": {a: float(v[t, xi] - v[t, ri]) for a, v in ablated.items()},
            "ablated_argmax": {a: names[int(v[t].argmax())] for a, v in ablated.items()},
        }
        records.append(rec)
    meta = {
        "verb": "row_table", "T": T, "X": X, "R": R, "letters": ids,
        "bindings": {b: _span(rows) for b, rows in binds.items()},
        "arms": {a: _span(cols) for a, cols in arms.items()},
        "labels": {"margin": "measured", "letters": "measured", "ablated": "measured",
                   "mass": "measured", "to_consumer": "derived-exact", "share": "derived-exact"},
        "seconds": {"native": round(t_native, 3), "total": round(time.time() - t0, 3)},
        "chunk": chunk,
    }
    if check:
        meta["trace_check_max_abs"] = _check_against_symbols(run, binds, ids, xi, ri, records[-1])
    return Trace("rows", records, meta, from_checkpoint=getattr(run, "from_checkpoint", None))


def _span(rows: list[int]) -> Any:
    if not rows:
        return []
    if rows == list(range(rows[0], rows[-1] + 1)):
        return f"{rows[0]}:{rows[-1] + 1}"
    return rows


def _row_ablation(run: Any, cache: Any, tokens: list[int], cols: list[int], ids: list[int],
                  chunk: int) -> Any:
    """Letter logits of every row with that row's reads of ``cols`` zeroed (module doc)."""
    torch = _torch()
    base, _, _ = decoder_parts(run)
    device = next(base.parameters()).device
    T = len(tokens)
    out = torch.empty(T, len(ids), dtype=torch.float64, device=device)
    if not cols:
        raise Refusal("an arm needs at least one column")
    index = torch.as_tensor(cols, device=device)
    saved = [layer.values[:, :, index].clone() for layer in cache.layers]
    for layer in cache.layers:
        layer.values[:, :, index] = 0
    neg = torch.finfo(next(base.parameters()).dtype).min
    arm = set(cols)
    try:
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            C = b - a
            rows = torch.arange(a, b, device=device)
            mask = torch.full((C, T + C), neg, device=device)
            mask[:, :T] = torch.where(torch.arange(T, device=device)[None, :] < rows[:, None],
                                      0.0, neg)
            mask[torch.arange(C, device=device), T + torch.arange(C, device=device)] = 0.0
            # a row inside the arm reads its own value cell on the diagonal (the freshly
            # appended V, which the zeroed cache rows do not cover): zero it there too, so the
            # row's read of its own column is ablated like the stepped p-zero (AUDIT-B2)
            own = [i for i in range(C) if a + i in arm]
            with torch.no_grad(), _zero_appended_values(cache, own):
                o = base(input_ids=torch.tensor([tokens[a:b]], device=device),
                         past_key_values=cache, attention_mask=mask[None, None],
                         position_ids=rows[None], use_cache=True)
                out[a:b] = letter_logits(run, o.last_hidden_state[0], ids)
            crop_to(cache, T)
            del o
    finally:
        for layer, keep in zip(cache.layers, saved):
            layer.values[:, :, index] = keep
    return out


class _zero_appended_values:
    """While active, each cache layer's ``update`` zeroes the appended value rows ``own``
    (positions within the appended chunk) before they are stored and read, leaving the keys
    (so the row's weights, and the softmax, are unchanged: p-zero, no renormalisation)."""

    def __init__(self, cache: Any, own: list[int]):
        self.cache, self.own = cache, own

    def __enter__(self):
        if not self.own:
            return self
        own = self.own

        def wrap(update):
            def patched(key_states, value_states, *args, **kwargs):
                value_states = value_states.clone()
                value_states[:, :, own] = 0
                return update(key_states, value_states, *args, **kwargs)
            return patched

        for layer in self.cache.layers:
            layer.update = wrap(layer.update)
        return self

    def __exit__(self, *exc):
        if self.own:
            for layer in self.cache.layers:
                del layer.update            # back to the class method
        return False


def _check_against_symbols(run: Any, binds: dict[str, list[int]], ids: list[int], xi: int,
                           ri: int, last: dict[str, Any]) -> float:
    from . import symbols

    row = len(run.tokens) - 1
    run.until(f"t{row}.logits")
    for name, rows in binds.items():
        if rows:
            symbols.declare(run, name, [f"t{s}" for s in rows])
    tab = symbols.table(run, consumer=Margin(row, ids[xi], ids[ri]), limit=16)
    worst = 0.0
    for b in tab.report["bindings"]:
        if b["symbol"] in last["to_consumer"]:
            value = b["to_consumer"]
            value = value["value"] if isinstance(value, dict) else value
            worst = max(worst, abs(float(value) - last["to_consumer"][b["symbol"]]))
    return worst

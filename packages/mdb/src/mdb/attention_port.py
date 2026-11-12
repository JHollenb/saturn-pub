"""One attention port: observe reads and cut reads per head, for Qwen and FLUX.2 Klein.

Coordinates are (step, block, head, receiver row, source row). A *receiver* is a
query row; a *source* is a key/value row. Receivers and sources are both given
in key coordinates; for a causal decoder call with a cache the query row ``i``
is key position ``Sk - Sq + i`` (Qwen only; the FLUX backend refuses Sq != Sk).

Two cut types, both per head:

``renormalize=True``  boolean mask; the remaining *visible* weights of a cut
                      receiver are renormalised to sum to one (what
                      ``attn_mask`` does). Sources the native mask hides (bias
                      at or below ``invisible_below(dtype)``) are never read.
``renormalize=False`` the cut sources' contribution is removed and every other
                      weight keeps its native value (no renormalisation).

A result about what a read carries counts only if both cut types agree: in the
FLUX read-graph experiment a renormalising mask manufactured object duplication
that never appeared under the no-renormalisation cut (4 of 4).

Precision: the native attention call always runs unchanged for every head.
Only the (head, receiver row) pairs a cut touches are recomputed in FP32 and
written back; every other head and row stays bit-identical to native. With
``zero_method="delta"`` a no-renormalisation cut is applied as
``native_out - P[cut] @ V[cut]`` (FP32 then cast). It keeps the native kernel's
rounding of the whole row and rounds again, so its ``sham`` is exactly native
*by construction* and says nothing about the arithmetic noise of the cut; use
:func:`floor_arms` for that. ``zero_method="recompute"`` recomputes the touched
rows as ``(P * keep) @ V`` in FP32. A (head, receiver) that loses every visible
source under a zero cut gets an exact zero. ``precision="legacy"`` (FLUX only)
reproduces the 2026-11-11 read-graph worker bit for bit.

Batch size must be 1 (both backends refuse B > 1). Captures never copy model
weights. Records carry the backend name, a caller supplied model identity and
parameter *names* (pointers) for anything derived from weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
import math
import numbers

ALL = "all"
ZERO_METHODS = ("delta", "recompute")


# --------------------------------------------------------------- grouped query

def kv_group(head, n_heads, n_kv_heads):
    """Key/value group read by query ``head`` (HF ``repeat_kv``: h // (n_heads // n_kv))."""
    for name, value in (("head", head), ("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(name + " must be an integer")
    head, n_heads, n_kv_heads = int(head), int(n_heads), int(n_kv_heads)
    if n_kv_heads < 1 or n_heads < n_kv_heads or n_heads % n_kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    if not 0 <= head < n_heads:
        raise ValueError("query head outside the layer")
    return head // (n_heads // n_kv_heads)


def kv_groups(n_heads, n_kv_heads):
    """Tuple mapping every query head to its key/value group."""
    return tuple(kv_group(h, n_heads, n_kv_heads) for h in range(int(n_heads)))


# ------------------------------------------------------------------ visibility

def invisible_below(dtype):
    """Additive-bias threshold: a source with bias <= this is hidden by the native mask.

    Hugging Face eager masks are finite (``finfo(dtype).min``: -3.4e38 fp32,
    -3.39e38 bf16, -65504 fp16), so ``isfinite`` cannot detect them. The
    threshold is the larger of -1e30 and half the dtype minimum.
    """
    import torch
    if dtype == torch.bool:
        raise ValueError("boolean masks are converted to an additive bias first")
    return max(-1e30, float(torch.finfo(dtype).min) / 2)


# ------------------------------------------------------------------ cut specs

def _rows_spec(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    out = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        else:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, numbers.Integral):
                out.append((int(item),))
            else:
                out.append(tuple(sorted({int(i) for i in item})))
    return tuple(out)


def _heads_spec(heads):
    if heads is None:
        return None
    out = set()
    for item in heads:
        if isinstance(item, numbers.Integral) and not isinstance(item, bool):
            out.add((None, int(item)))
        else:
            block, head = item
            out.add((int(block), int(head)))
    return frozenset(out)


def _check_zero_method(zero_method, allow_none):
    if zero_method is None and allow_none:
        return
    if zero_method not in ZERO_METHODS:
        raise ValueError("zero_method must be 'delta' or 'recompute'" + (" (or None)" if allow_none else ""))


@dataclass(frozen=True)
class Cut:
    """One cut: receivers lose their reads of sources in the selected heads/blocks/steps.

    ``receivers``/``sources`` are row-name strings (resolved through the port's
    ``rows`` table) or explicit integer row collections. ``heads`` entries are
    ``head`` (every allowed block) or ``(block, head)``; ``None`` = every head.
    ``sham=True`` recomputes the same (head, receiver) rows through the same code
    path but removes nothing. ``zero_method`` overrides the port's choice for a
    no-renormalisation cut (``None`` = the port's).
    """
    receivers: tuple
    sources: tuple
    heads: frozenset | None = None
    blocks: frozenset | None = None
    steps: frozenset | None = None
    renormalize: bool = False
    sham: bool = False
    zero_method: str | None = None

    def active(self, step, block):
        return ((self.steps is None or step in self.steps)
                and (self.blocks is None or block in self.blocks)
                and (self.heads is None or any(b is None or b == block for b, _ in self.heads)))

    def heads_at(self, block, n_heads):
        if self.heads is None:
            return tuple(range(n_heads))
        hs = sorted({h for b, h in self.heads if b is None or b == block})
        if any(not 0 <= h < n_heads for h in hs):
            raise ValueError("cut head outside the attention call")
        return tuple(hs)

    def to_dict(self):
        """JSON-able recipe entry (row names / indices, never tensors)."""
        def rows(spec):
            return [x if isinstance(x, str) else list(x) for x in spec]
        return dict(receivers=rows(self.receivers), sources=rows(self.sources),
                    heads=None if self.heads is None else sorted([[b, h] for b, h in self.heads],
                                                                 key=lambda x: (-1 if x[0] is None else x[0], x[1])),
                    blocks=None if self.blocks is None else sorted(self.blocks),
                    steps=None if self.steps is None else sorted(self.steps),
                    renormalize=self.renormalize, sham=self.sham, zero_method=self.zero_method)


def cut(receivers, sources, heads=None, blocks=None, steps=None, *, renormalize, zero_method=None):
    """Build a cut spec. ``renormalize`` is required: choose the cut type explicitly."""
    if not isinstance(renormalize, bool):
        raise ValueError("renormalize must be True or False")
    _check_zero_method(zero_method, allow_none=True)
    return Cut(_rows_spec(receivers), _rows_spec(sources), _heads_spec(heads),
               None if blocks is None else frozenset(int(b) for b in blocks),
               None if steps is None else frozenset(int(s) for s in steps), renormalize, False, zero_method)


def sham(like=None, *, renormalize=None, zero_method=None):
    """Nothing cut, same code path. ``like`` = the cut whose cells are routed.

    Without ``like``: every head, block, step and receiver row is routed
    through the cut path (worst-case floor of the chosen cut type). A delta
    sham is exactly native by construction; it is not a floor (see
    :func:`floor_arms`).
    """
    _check_zero_method(zero_method, allow_none=True)
    if like is None:
        if not isinstance(renormalize, bool):
            raise ValueError("sham() without like= needs renormalize=True/False")
        return Cut((ALL,), (), None, None, None, renormalize, True, zero_method)
    return replace(like, sham=True, renormalize=like.renormalize if renormalize is None else renormalize,
                   zero_method=like.zero_method if zero_method is None else zero_method)


def paired(c):
    """(cut, its sham) — run both for every (prompt, seed)."""
    return c, sham(c)


def floor_arms(c):
    """Named arms that measure the arithmetic floor of cut ``c``; run each beside ``c``.

    ``renormalize=False``: ``sham_recompute`` (FP32 recompute of exactly c's
    cells, nothing removed; compare with the base arm) and ``cut_recompute``
    (``c`` itself with ``zero_method="recompute"``; compare with ``c``). The
    floor of ``c`` is the larger of the two distances. A delta sham is exactly
    native by construction and is not a floor: a delta cut keeps the native
    kernel's rounding, subtracts an FP32 term from an already rounded output
    and rounds again.

    ``renormalize=True``: ``sham`` (the matched FP32 sham). There is no delta
    form of a renormalising cut.
    """
    if not isinstance(c, Cut) or c.sham:
        raise ValueError("floor_arms() takes a real cut, not a sham")
    if c.renormalize:
        return {"sham": sham(c)}
    return {"sham_recompute": replace(c, sham=True, zero_method="recompute"),
            "cut_recompute": replace(c, zero_method="recompute")}


# ------------------------------------------------------------------- the read

class Read:
    """One native attention call in head-major coordinates (batch size 1).

    q: [1, H, Sq, D]; k, v: [1, H, Sk, D] already expanded to query heads;
    bias: additive mask broadcastable to [1, H, Sq, Sk] or None.
    """

    def __init__(self, backend, step, block, q, k, v, scale, bias=None, out=None):
        self.backend, self.step, self.block = backend, step, block
        self.q, self.k, self.v, self.scale, self.bias, self.out = q, k, v, scale, bias, out
        self.n_heads = q.shape[1]
        self.offset = k.shape[2] - q.shape[2]
        self._probs = None

    def _bias(self, heads=None, rows=None):
        b = self.bias
        if heads is not None and b.shape[1] != 1:
            b = b[:, heads]
        if rows is not None and b.shape[2] != 1:
            b = b[:, :, rows]
        return b

    def scores(self, heads=None, rows=None):
        import torch
        q = self.q if heads is None else self.q[:, heads]
        k = self.k if heads is None else self.k[:, heads]
        if rows is not None:
            q = q[:, :, rows]
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        if self.bias is not None:
            s = s + self._bias(heads, rows).float()
        return s

    def visible(self, heads=None, rows=None):
        """Boolean [1, h|1, r|1, Sk]: sources the native mask lets each receiver see (None = all)."""
        if self.bias is None:
            return None
        b = self._bias(heads, rows)
        return b > invisible_below(b.dtype)

    def probs(self):
        """FP32 softmax of this call's native inputs (before this call's own cut)."""
        import torch
        if self._probs is None:
            self._probs = torch.softmax(self.scores(), dim=-1)
        return self._probs

    def write_norms(self):
        """[(query-row slice, [H, Sk] norm of each source's value after the receiver's output projection)]."""
        return self.backend.write_norms(self)


# ------------------------------------------------------------------ observers

class Observer:
    """Collects one reduction per (step, block). ``reduce(read) -> array-like or dict``."""

    def __init__(self, reduce, *, name=None, steps=None, blocks=None, params=None):
        self.reduce, self.name = reduce, name or getattr(reduce, "__name__", "reduction")
        self.steps = None if steps is None else frozenset(steps)
        self.blocks = None if blocks is None else frozenset(blocks)
        self.params = dict(params or {})
        self.records = {}

    def wants(self, step, block):
        return (self.steps is None or step in self.steps) and (self.blocks is None or block in self.blocks)

    def __call__(self, read):
        self.records[(read.step, read.block)] = self.reduce(read)

    def stack(self, key=None):
        """numpy array [steps, blocks, ...] (requires a full rectangle of records)."""
        import numpy as np
        steps = sorted({s for s, _ in self.records})
        blocks = sorted({b for _, b in self.records})
        def get(s, b):
            r = self.records[(s, b)]
            return r[key] if key is not None else r
        return np.stack([np.stack([np.asarray(get(s, b)) for b in blocks]) for s in steps])


def observe(reduce, *, steps=None, blocks=None, name=None, params=None):
    return Observer(reduce, steps=steps, blocks=blocks, name=name, params=params)


def full_probs(dtype="float16", heads=None):
    """Reduction: full FP32 softmax probabilities, stored at ``dtype`` [H, Sq, Sk]."""
    def reduce(read):
        p = read.probs()[0]
        if heads is not None:
            p = p[list(heads)]
        return p.detach().cpu().numpy().astype(dtype)
    reduce.__name__ = "full_probs"
    return reduce


def class_flow(source_classes, receiver_classes=None, *, weighted=False):
    """Reduction: receiver-class x source-class read mass per head, [H, Cr, Cs].

    ``source_classes``: [Sk, Cs] 0/1 (or weights) assigning key rows to classes.
    ``receiver_classes``: [Sq, Cr]; default = ``source_classes``. Receiver rows are
    averaged within a class (as ``make_capture``'s ``flow_p``). ``weighted=True``
    multiplies each read by the source's write norm through the receiver's
    output projection (``flow_w``).
    """
    import torch
    src = torch.as_tensor(source_classes, dtype=torch.float32)
    rcv = src if receiver_classes is None else torch.as_tensor(receiver_classes, dtype=torch.float32)
    rcv_t = rcv.t() / rcv.t().sum(1, keepdim=True).clamp_min(1.0)

    def reduce(read):
        p = read.probs()[0]
        if weighted:
            p = torch.cat([p[:, sl] * norms[:, None, :] for sl, norms in read.write_norms()], dim=1)
        s, r = src.to(p.device), rcv_t.to(p.device)
        return torch.einsum("rs,hsc->hrc", r, torch.matmul(p, s)).cpu().numpy()
    reduce.__name__ = "class_flow_w" if weighted else "class_flow_p"
    return reduce


# ------------------------------------------------------------------- the port

def _resolve(spec, table, size, device):
    import torch
    rows = []
    for item in spec:
        if item == ALL:
            rows.extend(range(size))
        elif isinstance(item, str):
            if item not in table:
                raise ValueError("unknown row name: " + item)
            rows.extend(table[item])
        else:
            rows.extend(item)
    rows = sorted(set(int(r) for r in rows))
    if rows and not 0 <= rows[0] <= rows[-1] < size:
        raise ValueError("cut row outside the attention call")
    return torch.tensor(rows, dtype=torch.long, device=device)


def _check_batch(batch, backend):
    if batch != 1:
        raise ValueError(f"{backend}: batch size {batch} is unsupported; the port reads and cuts batch item 0 "
                         "with one row table, so run one item (one prompt, no batched guidance) per apply")


@dataclass
class _Plan:
    cuts: list = field(default_factory=list)
    observers: list = field(default_factory=list)


class AttentionPort:
    """Backend-neutral driver. Use ``with port.apply(*cuts, observers=[...], forwards=N): <run model>``.

    ``empty`` decides what a renormalising cut does to a (head, receiver) that
    loses every source it can see: ``"raise"`` (default) refuses; ``"zero"``
    writes a zero read for it and records it in ``port.emptied``.
    """

    def __init__(self, backend, *, rows=None, zero_method="delta", precision="native", empty="raise"):
        _check_zero_method(zero_method, allow_none=False)
        if precision not in ("native", "legacy"):
            raise ValueError("precision must be 'native' or 'legacy'")
        if empty not in ("raise", "zero"):
            raise ValueError("empty must be 'raise' or 'zero'")
        self.backend, self.rows = backend, dict(rows or {})
        self.zero_method, self.precision, self.empty = zero_method, precision, empty
        self.cuts, self.observers = (), ()
        self.forwards = None
        self.calls = 0
        self.forward_count = 0
        self.log = []
        self.emptied = []
        self._row_cache = {}
        self._log_enabled = False
        self._active_cuts, self._fired_observers = set(), set()

    # public -----------------------------------------------------------------
    def apply(self, *cuts, observers=(), rows=None, log=False, forwards=None):
        """Arm cuts/observers for one ``with`` block.

        ``forwards``: the number of model forwards the block will run (FLUX:
        transformer forwards = denoising steps, required with cuts or
        observers; Qwen: optional, number of times layer 0 runs). The exit
        raises if the observed count differs.
        """
        for c in cuts:
            if not isinstance(c, Cut):
                raise ValueError("apply() takes Cut specs; build them with cut()/sham()")
        if forwards is not None and (isinstance(forwards, bool) or not isinstance(forwards, numbers.Integral)
                                     or forwards < 1):
            raise ValueError("forwards must be a positive integer")
        self.cuts, self.observers = tuple(cuts), tuple(observers)
        self.forwards = None if forwards is None else int(forwards)
        if rows is not None:
            self.rows = dict(rows)
        self._row_cache = {}
        self._log_enabled = log
        return self

    def __enter__(self):
        self.calls = 0
        self.forward_count = 0
        self.log = []
        self.emptied = []
        self._active_cuts, self._fired_observers = set(), set()
        self.backend.install(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        cuts, observers = self.cuts, self.observers
        try:
            self.backend.uninstall()
        finally:
            self.cuts, self.observers = (), ()
        if exc_type is None:
            self._check_exit(cuts, observers)
        return False

    def _check_exit(self, cuts, observers):
        if (cuts or observers) and self.calls == 0:
            raise ValueError(f"{self.backend.name}: cuts/observers were supplied but no attention call was "
                             "intercepted; the model did not run through the patched function "
                             "(Qwen: is it running eager attention?)")
        check = getattr(self.backend, "check_complete", None)
        if check is not None:
            check()
        if self.forwards is not None and self.forward_count != self.forwards:
            raise ValueError(f"expected {self.forwards} model forwards, observed {self.forward_count}")
        idle = [i for i in range(len(cuts)) if i not in self._active_cuts]
        if idle:
            raise ValueError(f"cut(s) {idle} never matched an attention call (check steps/blocks/heads)")
        silent = [o.name for i, o in enumerate(observers) if i not in self._fired_observers]
        if silent:
            raise ValueError(f"observer(s) {silent} never matched an attention call")

    def metadata(self):
        return dict(backend=self.backend.name, model_identity=getattr(self.backend, "identity", None),
                    zero_method=self.zero_method, precision=self.precision, empty=self.empty,
                    forwards=self.forwards,
                    cuts=[c.to_dict() for c in self.cuts],
                    observers=[dict(name=o.name, params=o.params) for o in self.observers],
                    weights="never copied; derived quantities name their parameters")

    # backend callbacks --------------------------------------------------------
    def begin_forward(self):
        """Called by a backend when a new model forward starts; returns its step index."""
        self.forward_count += 1
        if self.forwards is not None and self.forward_count > self.forwards:
            raise ValueError(f"model forward {self.forward_count} exceeds forwards={self.forwards}; "
                             "classifier-free guidance as separate forwards or a second image/prompt in one "
                             "apply? One apply covers one trajectory of exactly `forwards` forwards")
        return self.forward_count - 1

    def plan(self, step, block):
        cuts = [c for c in self.cuts if c.active(step, block)]
        observers = [o for o in self.observers if o.wants(step, block)]
        self._active_cuts.update(i for i, c in enumerate(self.cuts) if c.active(step, block))
        self._fired_observers.update(i for i, o in enumerate(self.observers) if o.wants(step, block))
        p = _Plan(cuts, observers)
        return p if (p.cuts or p.observers) else None

    def rows_of(self, spec, size, device):
        key = (spec, size, str(device))
        if key not in self._row_cache:
            self._row_cache[key] = _resolve(spec, self.rows, size, device)
        return self._row_cache[key]

    def _kind(self, c):
        return ("renormalize",) if c.renormalize else ("zero", c.zero_method or self.zero_method)

    def edit(self, read, out, cuts):
        """Apply ``cuts`` to the head-major native output ``out`` [1, H, Sq, D].

        Returns (new_out, effective) where effective lists (heads, query rows,
        removed [r, Sk], weights [1, h, r, Sk]) for backends that also return
        attention weights. Untouched heads/rows of ``new_out`` are the native
        values; so are receivers the native mask already left with no visible
        source (they read nothing a cut could remove).
        """
        import torch
        H, Sq, Sk = read.n_heads, read.q.shape[2], read.k.shape[2]
        dev = read.q.device
        groups = {}
        for i, c in enumerate(cuts):
            for h in c.heads_at(read.block, H):
                groups.setdefault(h, []).append(i)
        by_sig = {}
        for h, sig in groups.items():
            by_sig.setdefault(tuple(sig), []).append(h)
        new = out.clone()
        effective, entries = [], []
        for sig, heads in sorted(by_sig.items(), key=lambda kv: kv[1]):
            kinds = {}
            for i in sig:
                c = cuts[i]
                r = self.rows_of(c.receivers, Sk, dev) - read.offset
                r = r[(r >= 0) & (r < Sq)]
                s = self.rows_of(c.sources, Sk, dev)
                kinds.setdefault(self._kind(c), []).append((r, s, c.sham))
            rowsets = {kind: torch.unique(torch.cat([r for r, _, _ in rects])) for kind, rects in kinds.items()}
            names = sorted(rowsets)
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    if torch.isin(rowsets[names[a]], rowsets[names[b]]).any():
                        raise ValueError("cuts of different types (or zero methods) touch the same "
                                         "(head, receiver) cell; run them as separate arms")
            hs = torch.tensor(heads, dtype=torch.long, device=dev)
            for kind in names:
                rows = rowsets[kind]
                if rows.numel() == 0:
                    continue
                entry = self._edit_cells(read, new, hs, rows, kinds[kind], kind)
                effective.append(entry[:4])
                entries.append(entry)
        if self._log_enabled:
            self.log.append(dict(step=read.step, block=read.block, groups=[
                dict(heads=[int(h) for h in hs], receiver_rows=int(rows.numel()),
                     removed_reads=int(removed.sum()), kind=kind[-1], emptied=int(emptied.sum()),
                     natively_empty=int(native_empty.sum()))
                for hs, rows, removed, _, kind, emptied, native_empty in entries]))
        return new, effective

    def _edit_cells(self, read, new, hs, rows, rects, kind):
        import torch
        Sq, Sk = read.q.shape[2], read.k.shape[2]
        dev = read.q.device
        pos = torch.full((Sq,), -1, dtype=torch.long, device=dev)
        pos[rows] = torch.arange(rows.numel(), device=dev)
        removed = torch.zeros(rows.numel(), Sk, dtype=torch.bool, device=dev)
        for r, s, is_sham in rects:
            if not is_sham and r.numel() and s.numel():
                removed[pos[r][:, None], s[None, :]] = True
        scores = read.scores(heads=hs, rows=rows)                     # [1, h, r, Sk]
        visible = read.visible(heads=hs, rows=rows)
        visible = (torch.ones_like(scores, dtype=torch.bool) if visible is None
                   else visible.expand(scores.shape))
        left = visible & ~removed
        native_empty = ~visible.any(-1)                              # [1, h, r]
        emptied = ~left.any(-1) & ~native_empty                      # lost every visible source
        native_rows = new[:, hs][:, :, rows]
        v = read.v[:, hs].float()
        if kind[0] == "renormalize":
            if emptied.any() and self.empty == "raise":
                cells = emptied[0].nonzero().tolist()
                raise ValueError(
                    f"renormalising cut leaves (head, receiver query row) cells "
                    f"{[(int(hs[h]), int(rows[r])) for h, r in cells][:8]} at step {read.step}, block "
                    f"{read.block} with no visible source; pass empty='zero' to AttentionPort to write a "
                    "zero read for them (recorded in port.emptied)")
            w = torch.softmax(scores.masked_fill(~left, float("-inf")), dim=-1)
            w = torch.where(left.any(-1, keepdim=True), w, torch.zeros_like(w))
            value = torch.matmul(w, v)
            if emptied.any():
                for h, r in emptied[0].nonzero().tolist():
                    self.emptied.append(dict(step=read.step, block=read.block, head=int(hs[h]),
                                             receiver_query_row=int(rows[r])))
        else:
            p = torch.softmax(scores.masked_fill(~visible, float("-inf")), dim=-1)
            p = torch.where(native_empty[..., None], torch.zeros_like(p), p)
            gone = p * removed
            if kind[1] == "delta":
                value = native_rows.float() - torch.matmul(gone, v)
                w = p - gone
            else:
                w = p * ~removed
                value = torch.matmul(w, v)
        # A receiver that lost every visible source reads exactly nothing.
        value = value.masked_fill(emptied[..., None], 0.0)
        w = w.masked_fill(emptied[..., None], 0.0)
        if native_empty.any():  # nothing visible natively: keep the native read and weights
            w = torch.where(native_empty[..., None], torch.softmax(scores, dim=-1), w)
        block_out = new[:, hs]
        block_out[:, :, rows] = torch.where(native_empty[..., None], native_rows, value.to(new.dtype))
        new[:, hs] = block_out
        return hs, rows, removed, w, kind, emptied, native_empty


# ------------------------------------------------------------------- backends

class FluxBackend:
    """FLUX.2 Klein: scoped override of ``transformer_flux2.dispatch_attention_fn``.

    Placement comes from forward hooks on every joint and single transformer
    block (joint blocks first): block = the hooked block's index, step = the
    number of times block 0 has started in this ``apply`` minus one. Every
    block must make exactly one attention call, every forward must run all
    blocks in order, and every call must have Sq == Sk, batch 1 and at least
    ``text_rows`` rows; anything else is refused. q/k/v arrive as [B, S, H, D]
    over text rows then image rows. ``text_rows`` is required and checked
    against the ``encoder_hidden_states`` each joint block receives.
    ``identity`` should be the adapter's model identity (pointer).
    """
    name = "flux2-dispatch_attention_fn"

    def __init__(self, transformer, *, text_rows, identity=None, module=None):
        if isinstance(text_rows, bool) or not isinstance(text_rows, numbers.Integral) or text_rows < 0:
            raise ValueError("text_rows must be a non-negative integer")
        self.transformer = transformer
        self.joint = len(transformer.transformer_blocks)
        self.single = len(transformer.single_transformer_blocks)
        self.blocks = self.joint + self.single
        self.heads = transformer.transformer_blocks[0].attn.heads
        self.text_rows, self.identity = int(text_rows), identity
        self._module = module
        self.port = None
        self._handles = []

    def install(self, port):
        if self._module is None:
            from diffusers.models.transformers import transformer_flux2
            self._module = transformer_flux2
        if getattr(self._module.dispatch_attention_fn, "_mdb_attention_port", False):
            raise ValueError("nested attention ports are unsupported")
        if (port.cuts or port.observers) and port.forwards is None:
            raise ValueError("FluxBackend: apply(..., forwards=N) is required with cuts or observers; N is the "
                             "number of transformer forwards in this block (denoising steps; twice that when "
                             "classifier-free guidance runs as separate forwards)")
        blocks = list(self.transformer.transformer_blocks) + list(self.transformer.single_transformer_blocks)
        if len({id(b) for b in blocks}) != len(blocks):
            raise ValueError("FluxBackend: transformer blocks must be distinct modules")
        if not all(hasattr(b, "register_forward_pre_hook") for b in blocks):
            raise ValueError("FluxBackend: transformer blocks must be torch modules (forward hooks place calls)")
        self._block, self._block_calls, self._next, self.step = None, 0, 0, -1
        try:
            for i, b in enumerate(blocks):
                self._handles.append(b.register_forward_pre_hook(partial(self._enter_block, i), with_kwargs=True))
                self._handles.append(b.register_forward_hook(partial(self._leave_block, i)))
        except Exception:
            self._remove_hooks()
            raise
        self.port, self.original = port, self._module.dispatch_attention_fn

        def dispatch(q, k, v, *args, **kwargs):
            return self.dispatch(q, k, v, *args, **kwargs)
        dispatch._mdb_attention_port = True
        self._module.dispatch_attention_fn = dispatch

    def _remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def uninstall(self):
        self._remove_hooks()
        if self.port is not None:
            self._module.dispatch_attention_fn = self.original
            self.port = None

    def check_complete(self):
        if self._next != 0 or self._block is not None:
            raise ValueError(f"FluxBackend: the last transformer forward stopped before block {self._next} "
                             f"of {self.blocks}")

    def _enter_block(self, index, module, args, kwargs):
        if self._block is not None:
            raise ValueError(f"FluxBackend: block {index} started inside block {self._block}")
        if index != self._next:
            raise ValueError(f"FluxBackend: block {index} ran where block {self._next} was expected; every "
                             f"forward must run all {self.blocks} blocks in order from block 0")
        if index == 0:
            self.step = self.port.begin_forward()
        if index < self.joint:
            enc = kwargs.get("encoder_hidden_states")  # diffusers and saturn_pub pass it by keyword
            if enc is not None and enc.shape[1] != self.text_rows:
                raise ValueError(f"FluxBackend: text_rows={self.text_rows} but block {index} received "
                                 f"{enc.shape[1]} text rows")
        self._block, self._block_calls = index, 0

    def _leave_block(self, index, module, args, output):
        if self._block_calls != 1:
            raise ValueError(f"FluxBackend: block {index} made {self._block_calls} attention calls; the port "
                             "expects exactly one per block (reference-image KV modes add calls)")
        self._block, self._next = None, (index + 1) % self.blocks

    def parameter_names(self, block):
        if block < self.joint:
            p = f"transformer_blocks.{block}.attn."
            return dict(text=p + "to_add_out.weight", image=p + "to_out.0.weight")
        p = f"single_transformer_blocks.{block - self.joint}.attn.to_out.weight[:, :inner_dim]"
        return dict(text=p, image=p)

    def _out_weights(self, block):
        if block < self.joint:
            attn = self.transformer.transformer_blocks[block].attn
            return attn.to_out[0].weight, attn.to_add_out.weight
        attn = self.transformer.single_transformer_blocks[block - self.joint].attn
        w = attn.to_out.weight[:, : attn.inner_dim]
        return w, w

    def write_norms(self, read):
        import torch
        w_img, w_txt = self._out_weights(read.block)
        D = read.v.shape[-1]
        vf = read.v[0].float()

        def norms(w):
            wh = w.float().view(w.shape[0], read.n_heads, D).permute(1, 0, 2)
            return torch.matmul(vf, wh.transpose(1, 2)).norm(dim=-1)
        n_img = norms(w_img)
        n_txt = n_img if w_txt is w_img else norms(w_txt)
        T = self.text_rows
        return [(slice(0, T), n_txt), (slice(T, None), n_img)]

    def _validate(self, q, k, v):
        if self._block is None:
            raise ValueError("FluxBackend: attention call outside a transformer block of this transformer")
        self._block_calls += 1
        if self._block_calls > 1:
            raise ValueError(f"FluxBackend: block {self._block} made a second attention call; the port expects "
                             "exactly one per block (reference-image KV modes add calls)")
        if q.dim() != 4 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
            raise ValueError("FluxBackend: unexpected q/k/v layout (expected [B, S, H, D])")
        _check_batch(q.shape[0], self.name)
        if q.shape[1] != k.shape[1]:
            raise ValueError(f"FluxBackend: {q.shape[1]} query rows against {k.shape[1]} key rows (a reference-"
                             "image KV cache?); row coordinates are defined only for Sq == Sk")
        if q.shape[2] != self.heads or q.shape[1] < self.text_rows:
            raise ValueError("FluxBackend: unexpected native attention call (heads or text rows)")

    def dispatch(self, q, k, v, *args, **kwargs):
        port = self.port
        self._validate(q, k, v)
        port.calls += 1
        step, block = self.step, self._block
        plan = port.plan(step, block)
        if plan is None:
            return self.original(q, k, v, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None:
            raise ValueError("FluxBackend: native call carries a mask or positional extras; unsupported")
        if plan.cuts and port.precision == "legacy":
            if plan.observers:
                self._observe(q, k, v, kwargs, plan, step, block)
            return self._legacy(q, k, v, kwargs, plan, block)
        out, read = self._observe(q, k, v, kwargs, plan, step, block)
        if not plan.cuts:
            return out
        new, _ = port.edit(read, read.out, plan.cuts)
        return new.permute(0, 2, 1, 3).contiguous()

    def _observe(self, q, k, v, kwargs, plan, step, block):
        out = self.original(q, k, v, **kwargs)
        scale = kwargs.get("scale")
        read = Read(self, step, block, q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                    scale if scale is not None else q.shape[-1] ** -0.5, None, out.permute(0, 2, 1, 3))
        for o in plan.observers:
            o(read)
        return out, read

    def _legacy(self, q, k, v, kwargs, plan, block):
        """2026-11-11 read-graph worker arithmetic, kept for exact reproduction."""
        import torch
        port = self.port
        S, H, dev = q.shape[1], self.heads, q.device
        kinds = {c.renormalize for c in plan.cuts}
        if len(kinds) != 1:
            raise ValueError("legacy precision cannot mix cut types in one block")
        m = None
        for c in plan.cuts:
            mc = torch.ones(S, S, dtype=torch.bool, device=dev)
            r, s = port.rows_of(c.receivers, S, dev), port.rows_of(c.sources, S, dev)
            if len(r) and len(s) and not c.sham:
                mc[r[:, None], s[None, :]] = False
            if not mc.any(1).all():
                raise ValueError("cut leaves a receiver with no source")
            if c.heads is None:
                mh = mc[None, None]
            else:
                mh = torch.ones(1, H, S, S, dtype=torch.bool, device=dev)
                mh[0, list(c.heads_at(block, H))] = mc
            m = mh if m is None else (m & mh)
        if kinds == {False}:
            qh, kh, vh = (x.permute(0, 2, 1, 3).float() for x in (q, k, v))
            w = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(q.shape[-1]), dim=-1) * m
            return (w @ vh).permute(0, 2, 1, 3).to(q.dtype)
        return self.original(q, k, v, **dict(kwargs, attn_mask=m))


class QwenBackend:
    """Qwen-family eager attention (``modeling_<family>.eager_attention_forward``).

    Wraps :class:`mdb.attention_observer.AttentionObserver` (same patch point,
    same nesting guard). The patched module is the one defining the model's
    attention class (``modeling_qwen2``, ``modeling_qwen3``, ...), each of
    which has its own ``eager_attention_forward``. Block = decoder layer index;
    step = number of times layer 0 has run in this context minus one.
    Query/key/value are [B, H|KV, S, D]; grouped-query heads use
    :func:`kv_group`. The model must run eager attention (refused otherwise)
    with batch size 1.
    """
    name = "qwen-eager_attention_forward"

    def __init__(self, model, *, identity=None, module=None):
        self.model, self.identity, self.module = model, identity, module
        self.heads = model.config.num_attention_heads
        self.kv_heads = model.config.num_key_value_heads
        self._observer = None

    def install(self, port):
        from .attention_observer import AttentionObserver
        impl = getattr(self.model.config, "_attn_implementation", None)
        if impl != "eager":
            raise ValueError(f"QwenBackend needs eager attention but the model runs {impl!r}: the patched "
                             "eager_attention_forward would never be called and cuts would do nothing. Load with "
                             "attn_implementation='eager' (or set model.config._attn_implementation = 'eager')")
        backend = self
        module = self.module
        if module is None:
            import sys
            module = sys.modules[type(self.model.model.layers[0].self_attn).__module__]
        if not callable(getattr(module, "eager_attention_forward", None)):
            raise ValueError(f"QwenBackend: {getattr(module, '__name__', module)} defines no eager_attention_forward")

        class _Intercept(AttentionObserver):
            def _intercept(self, layer, attention, query, key, value, args, kwargs):
                return backend.dispatch(self.original, layer, attention, query, key, value, args, kwargs)

        self.port = port
        self._observer = _Intercept(self.model, module=module).__enter__()

    def uninstall(self):
        if self._observer is not None:
            self._observer.__exit__()
            self._observer = None

    def parameter_names(self, block):
        name = f"model.layers.{block}.self_attn.o_proj.weight"
        return dict(text=name)

    def write_norms(self, read):
        import torch
        w = self.model.model.layers[read.block].self_attn.o_proj.weight
        D = read.v.shape[-1]
        wh = w.float().view(w.shape[0], read.n_heads, D).permute(1, 0, 2)
        return [(slice(None), torch.matmul(read.v[0].float(), wh.transpose(1, 2)).norm(dim=-1))]

    def dispatch(self, original, layer, attention, query, key, value, args, kwargs):
        import torch
        port = self.port
        _check_batch(query.shape[0], self.name)
        if layer == 0:
            port.begin_forward()
        elif port.forward_count == 0:
            raise ValueError(f"QwenBackend: layer {layer} ran before layer 0")
        step = port.forward_count - 1
        port.calls += 1
        plan = port.plan(step, layer)
        result = original(attention, query, key, value, *args, **kwargs)
        if plan is None:
            return result
        if attention.training:
            raise ValueError("attention port requires evaluation mode")
        if port.precision == "legacy":
            raise ValueError("legacy precision exists only for the FLUX backend")
        mask = args[0] if args else kwargs.get("attention_mask")
        scaling = kwargs.get("scaling", args[1] if len(args) > 1 else None)
        H, KV = query.shape[1], key.shape[1]
        groups = torch.tensor(kv_groups(H, KV), dtype=torch.long, device=key.device)
        k, v = key[:, groups], value[:, groups]
        bias = None
        if mask is not None:
            bias = mask[..., : key.shape[-2]]
            if bias.dtype == torch.bool:
                bias = torch.zeros(bias.shape, device=bias.device).masked_fill(~bias, float("-inf"))
        out, weights = result
        read = Read(self, step, layer, query, k, v, float(scaling), bias, out.transpose(1, 2))
        for o in plan.observers:
            o(read)
        if not plan.cuts:
            return result
        new, effective = port.edit(read, read.out, plan.cuts)
        if weights is not None:
            weights = weights.clone()
            for hs, rows, _, w in effective:
                block_w = weights[:, hs]
                block_w[:, :, rows] = w.to(weights.dtype)
                weights[:, hs] = block_w
        return new.transpose(1, 2).contiguous(), weights

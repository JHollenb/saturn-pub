"""One attention port: observe reads and cut reads per head, for Qwen and FLUX.2 Klein.

Coordinates are (step, block, head, receiver row, source row). A *receiver* is a
query row; a *source* is a key/value row. Receivers and sources are both given
in key coordinates; for a causal decoder call with a cache the query row ``i``
is key position ``Sk - Sq + i``.

Two cut types, both per head:

``renormalize=True``  boolean mask; the remaining weights of a cut receiver are
                      renormalised to sum to one (what ``attn_mask`` does).
``renormalize=False`` the cut sources' contribution is removed and every other
                      weight keeps its native value (no renormalisation).

A result about what a read carries counts only if both cut types agree: in the
FLUX read-graph experiment a renormalising mask manufactured object duplication
that never appeared under the no-renormalisation cut (4 of 4).

Precision: the native attention call always runs unchanged for every head.
Only the (head, receiver row) pairs a cut touches are recomputed in FP32 and
written back; every other head and row stays bit-identical to native. With the
default ``zero_method="delta"`` a no-renormalisation cut is applied as
``native_out - P[cut] @ V[cut]`` (FP32 then cast), so its ``sham`` is exactly
native. ``zero_method="recompute"`` recomputes the touched rows as
``(P * keep) @ V`` in FP32. ``precision="legacy"`` (FLUX only) reproduces the
2026-11-11 read-graph worker bit for bit: every head of a touched block in FP32
for zero cuts, a [1, H, S, S] boolean ``attn_mask`` through the native kernel
for renormalising cuts.

Captures never copy model weights. Records carry the backend name, a caller
supplied model identity and parameter *names* (pointers) for anything derived
from weights, e.g. write norms through an output projection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

ALL = "all"


# --------------------------------------------------------------- grouped query

def kv_group(head, n_heads, n_kv_heads):
    """Key/value group read by query ``head`` (HF ``repeat_kv``: h // (n_heads // n_kv))."""
    for name, value in (("head", head), ("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(name + " must be an integer")
    if n_kv_heads < 1 or n_heads < n_kv_heads or n_heads % n_kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")
    if not 0 <= head < n_heads:
        raise ValueError("query head outside the layer")
    return head // (n_heads // n_kv_heads)


def kv_groups(n_heads, n_kv_heads):
    """Tuple mapping every query head to its key/value group."""
    return tuple(kv_group(h, n_heads, n_kv_heads) for h in range(n_heads))


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
            if isinstance(item, int):
                out.append((int(item),))
            else:
                out.append(tuple(sorted({int(i) for i in item})))
    return tuple(out)


def _heads_spec(heads):
    if heads is None:
        return None
    out = set()
    for item in heads:
        if isinstance(item, int) and not isinstance(item, bool):
            out.add((None, int(item)))
        else:
            block, head = item
            out.add((int(block), int(head)))
    return frozenset(out)


@dataclass(frozen=True)
class Cut:
    """One cut: receivers lose their reads of sources in the selected heads/blocks/steps.

    ``receivers``/``sources`` are row-name strings (resolved through the port's
    ``rows`` table) or explicit integer row collections. ``heads`` entries are
    ``head`` (every allowed block) or ``(block, head)``; ``None`` = every head.
    ``sham=True`` recomputes the same (head, receiver) rows through the same code
    path but removes nothing; it measures the floor of the cut it shadows.
    """
    receivers: tuple
    sources: tuple
    heads: frozenset | None = None
    blocks: frozenset | None = None
    steps: frozenset | None = None
    renormalize: bool = False
    sham: bool = False

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
                    renormalize=self.renormalize, sham=self.sham)


def cut(receivers, sources, heads=None, blocks=None, steps=None, *, renormalize):
    """Build a cut spec. ``renormalize`` is required: choose the cut type explicitly."""
    if not isinstance(renormalize, bool):
        raise ValueError("renormalize must be True or False")
    return Cut(_rows_spec(receivers), _rows_spec(sources), _heads_spec(heads),
               None if blocks is None else frozenset(int(b) for b in blocks),
               None if steps is None else frozenset(int(s) for s in steps), renormalize, False)


def sham(like=None, *, renormalize=None):
    """Nothing cut, same code path. ``like`` = the cut whose floor is wanted.

    Without ``like``: every head, block, step and receiver row is routed
    through the cut path (worst-case floor of the chosen cut type).
    """
    if like is None:
        if not isinstance(renormalize, bool):
            raise ValueError("sham() without like= needs renormalize=True/False")
        return Cut((ALL,), (), None, None, None, renormalize, True)
    return replace(like, sham=True, renormalize=like.renormalize if renormalize is None else renormalize)


def paired(c):
    """(cut, its sham) — run both for every (prompt, seed)."""
    return c, sham(c)


# ------------------------------------------------------------------- the read

class Read:
    """One native attention call in head-major coordinates.

    q: [B, H, Sq, D]; k, v: [B, H, Sk, D] already expanded to query heads;
    bias: additive mask broadcastable to [B, H, Sq, Sk] or None.
    """

    def __init__(self, backend, step, block, q, k, v, scale, bias=None, out=None):
        self.backend, self.step, self.block = backend, step, block
        self.q, self.k, self.v, self.scale, self.bias, self.out = q, k, v, scale, bias, out
        self.n_heads = q.shape[1]
        self.offset = k.shape[2] - q.shape[2]
        self._probs = None

    def scores(self, heads=None, rows=None):
        import torch
        q = self.q if heads is None else self.q[:, heads]
        k = self.k if heads is None else self.k[:, heads]
        if rows is not None:
            q = q[:, :, rows]
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scale
        if self.bias is not None:
            b = self.bias
            if heads is not None and b.shape[1] != 1:
                b = b[:, heads]
            if rows is not None:
                b = b[:, :, rows]
            s = s + b.float()
        return s

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


@dataclass
class _Plan:
    cuts: list = field(default_factory=list)
    observers: list = field(default_factory=list)


class AttentionPort:
    """Backend-neutral driver. Use ``with port.apply(*cuts, observers=[...]): <run model>``."""

    def __init__(self, backend, *, rows=None, zero_method="delta", precision="native"):
        if zero_method not in ("delta", "recompute"):
            raise ValueError("zero_method must be 'delta' or 'recompute'")
        if precision not in ("native", "legacy"):
            raise ValueError("precision must be 'native' or 'legacy'")
        self.backend, self.rows = backend, dict(rows or {})
        self.zero_method, self.precision = zero_method, precision
        self.cuts, self.observers = (), ()
        self.calls = 0
        self.log = []
        self._row_cache = {}
        self._log_enabled = False

    # public -----------------------------------------------------------------
    def apply(self, *cuts, observers=(), rows=None, log=False):
        for c in cuts:
            if not isinstance(c, Cut):
                raise ValueError("apply() takes Cut specs; build them with cut()/sham()")
        self.cuts, self.observers = tuple(cuts), tuple(observers)
        if rows is not None:
            self.rows = dict(rows)
        self._row_cache = {}
        self._log_enabled = log
        return self

    def __enter__(self):
        self.calls = 0
        self.log = []
        self.backend.install(self)
        return self

    def __exit__(self, *exc):
        self.backend.uninstall()
        self.cuts, self.observers = (), ()
        return False

    def metadata(self):
        return dict(backend=self.backend.name, model_identity=getattr(self.backend, "identity", None),
                    zero_method=self.zero_method, precision=self.precision,
                    cuts=[c.to_dict() for c in self.cuts],
                    observers=[dict(name=o.name, params=o.params) for o in self.observers],
                    weights="never copied; derived quantities name their parameters")

    # backend callback ---------------------------------------------------------
    def plan(self, step, block):
        p = _Plan([c for c in self.cuts if c.active(step, block)],
                  [o for o in self.observers if o.wants(step, block)])
        return p if (p.cuts or p.observers) else None

    def rows_of(self, spec, size, device):
        key = (spec, size, str(device))
        if key not in self._row_cache:
            self._row_cache[key] = _resolve(spec, self.rows, size, device)
        return self._row_cache[key]

    def edit(self, read, out, cuts):
        """Apply ``cuts`` to the head-major native output ``out`` [B, H, Sq, D].

        Returns (new_out, effective) where effective lists (heads, query rows,
        weights [B, h, r, Sk]) for backends that also return attention weights.
        Untouched heads/rows of ``new_out`` are the native values.
        """
        import torch
        H, Sq, Sk = read.n_heads, read.q.shape[2], read.k.shape[2]
        dev = read.q.device
        renorm_of, groups = {}, {}
        for i, c in enumerate(cuts):
            for h in c.heads_at(read.block, H):
                if renorm_of.setdefault(h, c.renormalize) != c.renormalize:
                    raise ValueError("cuts of both types touch the same head; run them as separate arms")
                groups.setdefault(h, []).append(i)
        by_sig = {}
        for h, sig in groups.items():
            by_sig.setdefault(tuple(sig), []).append(h)
        new = out.clone()
        effective = []
        for sig, heads in sorted(by_sig.items(), key=lambda kv: kv[1]):
            rects = []
            for i in sig:
                c = cuts[i]
                r = self.rows_of(c.receivers, Sk, dev) - read.offset
                r = r[(r >= 0) & (r < Sq)]
                s = self.rows_of(c.sources, Sk, dev)
                rects.append((r, s, c.sham))
            rows = torch.unique(torch.cat([r for r, _, _ in rects])) if rects else None
            if rows is None or rows.numel() == 0:
                continue
            pos = torch.full((Sq,), -1, dtype=torch.long, device=dev)
            pos[rows] = torch.arange(rows.numel(), device=dev)
            removed = torch.zeros(rows.numel(), Sk, dtype=torch.bool, device=dev)
            for r, s, is_sham in rects:
                if not is_sham and r.numel() and s.numel():
                    removed[pos[r][:, None], s[None, :]] = True
            hs = torch.tensor(heads, dtype=torch.long, device=dev)
            scores = read.scores(heads=hs, rows=rows)
            v = read.v[:, hs].float()
            if renorm_of[heads[0]]:
                masked = scores.masked_fill(removed, float("-inf"))
                if not torch.isfinite(masked.amax(-1)).all():
                    raise ValueError("cut leaves a receiver with no source")
                w = torch.softmax(masked, dim=-1)
                value = torch.matmul(w, v)
            else:
                p = torch.softmax(scores, dim=-1)
                if self.zero_method == "delta":
                    gone = p * removed
                    value = new[:, hs][:, :, rows].float() - torch.matmul(gone, v)
                    w = p - gone
                else:
                    w = p * ~removed
                    value = torch.matmul(w, v)
            block_out = new[:, hs]
            block_out[:, :, rows] = value.to(out.dtype)
            new[:, hs] = block_out
            effective.append((hs, rows, removed, w))
        if self._log_enabled:
            self.log.append(dict(step=read.step, block=read.block, groups=[
                dict(heads=[int(h) for h in hs], receiver_rows=int(rows.numel()),
                     removed_reads=int(removed.sum())) for hs, rows, removed, _ in effective]))
        return new, effective


# ------------------------------------------------------------------- backends

class FluxBackend:
    """FLUX.2 Klein: scoped override of ``transformer_flux2.dispatch_attention_fn``.

    Calls are numbered in execution order; ``step, block = divmod(index, blocks)``
    with joint blocks first. q/k/v arrive as [B, S, H, D] over text rows then
    image rows. ``identity`` should be the adapter's model identity (pointer).
    """
    name = "flux2-dispatch_attention_fn"

    def __init__(self, transformer, *, text_rows=512, identity=None, module=None):
        self.transformer = transformer
        self.joint = len(transformer.transformer_blocks)
        self.single = len(transformer.single_transformer_blocks)
        self.blocks = self.joint + self.single
        self.heads = transformer.transformer_blocks[0].attn.heads
        self.text_rows, self.identity = int(text_rows), identity
        self._module = module
        self.port = None

    def install(self, port):
        if self._module is None:
            from diffusers.models.transformers import transformer_flux2
            self._module = transformer_flux2
        if getattr(self._module.dispatch_attention_fn, "_mdb_attention_port", False):
            raise ValueError("nested attention ports are unsupported")
        self.port, self.original = port, self._module.dispatch_attention_fn

        def dispatch(q, k, v, *args, **kwargs):
            return self.dispatch(q, k, v, *args, **kwargs)
        dispatch._mdb_attention_port = True
        self._module.dispatch_attention_fn = dispatch

    def uninstall(self):
        if self.port is not None:
            self._module.dispatch_attention_fn = self.original
            self.port = None

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

    def dispatch(self, q, k, v, *args, **kwargs):
        port = self.port
        index = port.calls
        port.calls += 1
        step, block = divmod(index, self.blocks)
        plan = port.plan(step, block)
        if plan is None:
            return self.original(q, k, v, *args, **kwargs)
        if args or kwargs.get("attn_mask") is not None or q.shape[2] != self.heads or q.shape[1] < self.text_rows:
            raise ValueError("unexpected native attention call")
        if plan.cuts and port.precision == "legacy":
            return self._legacy(q, k, v, kwargs, plan, block)
        out = self.original(q, k, v, **kwargs)
        scale = kwargs.get("scale")
        read = Read(self, step, block, q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
                    scale if scale is not None else q.shape[-1] ** -0.5, None, out.permute(0, 2, 1, 3))
        for o in plan.observers:
            o(read)
        if not plan.cuts:
            return out
        new, _ = port.edit(read, read.out, plan.cuts)
        return new.permute(0, 2, 1, 3).contiguous()

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
    same nesting guard). Block = decoder layer index; step = number of times
    layer 0 has run in this context minus one. Query/key/value are
    [B, H|KV, S, D]; grouped-query heads use :func:`kv_group`.
    """
    name = "qwen-eager_attention_forward"

    def __init__(self, model, *, identity=None, module=None):
        self.model, self.identity, self.module = model, identity, module
        self.heads = model.config.num_attention_heads
        self.kv_heads = model.config.num_key_value_heads
        self._observer = None

    def install(self, port):
        from .attention_observer import AttentionObserver
        backend = self
        module = self.module
        if module is None:
            import sys
            module = sys.modules[type(self.model.model.layers[0].self_attn).__module__]

        class _Intercept(AttentionObserver):
            def _intercept(self, layer, attention, query, key, value, args, kwargs):
                return backend.dispatch(self.original, layer, attention, query, key, value, args, kwargs)

        self.port, self.step = port, -1
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
        if layer == 0:
            self.step += 1
        port.calls += 1
        plan = port.plan(self.step, layer)
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
        read = Read(self, self.step, layer, query, k, v, float(scaling), bias, out.transpose(1, 2))
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

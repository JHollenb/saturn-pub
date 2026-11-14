"""Circuit records: a consumer's exact decomposition walked back to token rows, with the unlisted
remainder explicit, and a native test of what was retained.

``build(run, consumer, top=K, depth=D)`` decomposes a consumer of the cursor's row (a margin
``M[a,b]``, a logit, a residual port ``h``/``r``/``h_out`` or ``o``; see ``reads.readout``):

* depth 1: the residual-stream terms that reach it, ``embedding`` (``L0.h``), ``L<l>.attn``
  (``o``) and ``L<l>.mlp`` (``m``), each read through the consumer's exact linear map (P19
  for a margin or logit: ``rho`` evaluated once at the native final state);
* depth 2: an attention term into its heads, ``W_o,h A_h`` (P07), and an MLP term into its
  neurons, ``W_down[:, i] f_i`` (P15);
* depth 3: a head into its source rows, ``p[h, s] W_o,h v[kappa(h), s]`` (P06): the leaves are
  token rows.

At every node the ``top`` largest children (by ``|value|``, the norm for a vector consumer) are
retained; the others are one ``+N more`` node holding their exact sum. Every node whose
children are not exact identities of native values has a ``remainder`` child (``measured``:
the native value minus the exact sum) with a rounding ``bound`` (round to nearest; matmul
partial sums at the run's accumulation precision, ``accumulation.accumulation``: float32 unless
CUDA reduced-precision reductions are allowed, when a sum with ``n u >= 1`` has no bound and the
remainder's bound is refused; one rounding to the model dtype; residual adds round once; the final
norm term is explain's a-priori ``k = 2`` multiple, flagged) and a ``within`` flag. A committed
or staged edit that wrote the native value there (``Run.read_info``) is its own ``edit`` node
(the native value minus the pre-edit value, labelled as ``read_info`` labels the port), never
part of a remainder. At every node the children sum to the node by construction; ``closure``
checks only the float64 summation, and ``remainders_within`` is the rounding check.

``test(run, circuit, tolerance)`` ablates natively (``analysis.compare.measure``; masked
``replace 0`` edits at ``o``, ``m``, a head's block of ``A``, a neuron of ``f``, a source of
``p``): the retained leaves, and separately everything except them (every ``+N more`` group).
The embedding is in neither set: zeroing ``L0.h`` removes the input of every other node. Both measured changes are reported beside the direct-path prediction (minus
the ablated contributions, ``rho`` and later layers frozen: ``derived-local``). The circuit is
``sufficient at this dose`` when ablating everything else moves the consumer by at most
``tolerance`` times its base size.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from mdb.core.kinds import CIRCUIT
from mdb.core.pointers import RecordError, check_bounded
from mdb.core.records import num, seal_record

from .accumulation import accumulation
from .finite import BOUND, EXACT, LOCAL, MEASURED, SUPPLIED, f64
from .forecast import Margin, Refusal, run_session_id
from .propagate import Edit
from .reads import (
    claim,
    computed,
    head_mask,
    head_mix,
    layer_module,
    readout,
    size,
    unit_roundoff,
)

SCHEMA = "mdb-circuit-v1"
KINDS = ("consumer", "embedding", "attn", "mlp", "head", "neuron", "source", "bias",
         "remainder", "edit", "more")
_FIXED = ("remainder", "bias", "edit")


def _torch():
    import torch

    return torch


class Member(NamedTuple):
    """A child folded into a ``+N more`` node (and, before grouping, a candidate child): what
    ``test`` ablates. ``port`` is the address a ``replace 0`` edit writes, ``index`` where in
    it: ``()`` the whole port, ``(i,)`` neuron ``i`` of ``f``, ``(h,)`` head ``h``'s block of
    ``A``, ``(h, s)`` source ``s`` of head ``h`` in ``p``; None: not ablatable (the
    embedding). ``value`` is its float64 contribution while the circuit is built; ``build``
    folds the members' values into ``Circuit.complement`` and keeps None."""

    name: str
    kind: str
    port: str
    index: tuple[int, ...] | None
    value: Any


@dataclass
class Node:
    """One node: ``value`` (float64) is the contribution its edge carries into its parent, and
    so into the consumer; ``authority`` labels it. ``port`` and ``index`` (as in ``Member``)
    say what ablates it natively (None for a remainder, bias or edit: not components).
    ``grow`` builds the node's children; ``_group`` calls it only for a node it keeps."""

    name: str
    kind: str
    address: str
    value: Any
    authority: str
    proof: str | None = None
    children: list[Node] = field(default_factory=list)
    port: str | None = None
    index: tuple[int, ...] | None = None
    members: list[Member] = field(default_factory=list)
    note: str | None = None
    bound: Any = None
    ref: str | None = None
    unexamined: int = 0   # ``more`` only: members dropped whose own subtree was never built
    grow: Callable[[], list[Node]] | None = field(default=None, repr=False)

    @property
    def within(self) -> bool | None:
        """``|value| <= bound`` (per coordinate for a vector), None when unbounded."""
        if self.bound is None:
            return None
        return bool((f64(self.value).abs() <= f64(self.bound)).all())

    def view(self, ids: list[int]) -> dict[str, Any]:
        ids[0] += 1
        self.ref = f"n{ids[0]}"
        out: dict[str, Any] = {"id": self.ref, "name": self.name, "kind": self.kind,
                               "address": self.address,
                               "value": claim(self.value, self.authority, self.proof)}
        if self.bound is not None:
            out["bound"] = claim(self.bound, BOUND)
            out["within"] = self.within
        if self.note:
            out["note"] = self.note
        if self.unexamined:
            out["unexamined"] = True
            out["unexamined_count"] = num(self.unexamined, EXACT)
        if self.children:
            out["children"] = [child.view(ids) for child in self.children]
        return out


@dataclass
class Circuit:
    """A built circuit: the sealed ``record``, the node tree and what ``test`` needs."""

    id: int
    record: dict[str, Any]
    root: Node
    consumer: Any
    where: str
    tests: list[dict[str, Any]] = field(default_factory=list)
    complement: Any = None  # minus the members' summed contributions (the embedding aside)


_LEDGERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def ledger(run: Any) -> dict[int, Circuit]:
    """The circuits built on ``run`` in this process, by id."""
    if run not in _LEDGERS:
        _LEDGERS[run] = {}
    return _LEDGERS[run]


# --------------------------------------------------------------------------- decomposition


def _group(children: list[Any], top: int, parent: str,
           promote: Callable[[Member], Node] | None = None) -> list[Node]:
    """Keep the ``top`` largest exact children; the rest become one ``+N more`` node holding
    them as ``Member`` tuples. A kept ``Member`` becomes its ``Node`` (``promote``) and a kept
    node's subtree is built now: nothing is built for a child that is not kept. A dropped child
    that has a subtree at this depth is counted in the group's ``unexamined``: a refusal its
    expansion would raise (a premise failing, a port not computed) was never seen."""
    exact = [c for c in children if c.kind not in _FIXED]
    fixed = [c for c in children if c.kind in _FIXED]
    exact.sort(key=lambda c: -size(c.value))
    kept, rest = exact[:top], exact[top:]
    kept = [promote(c) if isinstance(c, Member) else c for c in kept]
    for node in kept:
        if node.grow is not None:
            node.children, node.grow = node.grow(), None
    if rest:
        total = sum((c.value for c in rest[1:]), rest[0].value)
        members = [c if isinstance(c, Member) else Member(c.name, c.kind, c.port, c.index,
                                                            c.value) for c in rest]
        expands = getattr(promote, "expands", False)
        skipped = sum(1 for c in rest if (c.grow is not None if isinstance(c, Node) else expands))
        note = f"+{len(rest)} more, sum of their exact contributions"
        if skipped:
            note += f"; {skipped} have subtrees that were not expanded (not examined)"
        kept.append(Node(f"+{len(rest)} more", "more", f"{parent} (unlisted)", total, EXACT,
                         members=members, note=note, unexamined=skipped))
    return kept + fixed


def _selection(kind: str, index: tuple[int, ...]) -> str:
    """The selection text of one ablated index (as ``Run.set`` records it)."""
    if kind == "neuron":
        return f"i={index[0]}"
    if kind == "head":
        return f"h{index[0]}"
    return f"h{index[0]},src={index[1]}"


def _promoter(proof: str, grow: Callable[[Member], Any] | None = None
              ) -> Callable[[Member], Node]:
    """``_group``'s ``promote`` for neurons, heads or sources: the kept member's node (its
    address names the index), with ``grow(member)`` as its subtree builder."""

    def promote(m: Member) -> Node:
        return Node(m.name, m.kind, f"{m.port}[{_selection(m.kind, m.index)}]", m.value, EXACT,
                    proof, port=m.port, index=m.index,
                    grow=None if grow is None else (lambda: grow(m)))

    promote.expands = grow is not None  # type: ignore[attr-defined]
    return promote


def _remainder(name: str, address: str, value: Any, why: str, bound: Any = None) -> Node:
    return Node(name, "remainder", address, value, MEASURED, note=why, bound=bound)


def _through(reader: Any, bound: Any) -> Any:
    """A per-coordinate bound (residual frame) mapped through the consumer's linear map (None
    stays None: a refused bound)."""
    if bound is None:
        return None
    bound = f64(bound).reshape(-1)
    return reader.vector.abs() @ bound if reader.scalar else bound


def _matmul_bound(exact: Any, spread: Any, n: int, dtype: Any, acc: Any) -> Any:
    """``|fl(W x) - W x|`` per output from ``exact = W x`` and ``spread = |W| |x|`` (``n``
    terms): partial sums at the precision of ``acc`` (``accumulation.Accumulation``) in any
    order, one rounding to ``dtype``; None when ``acc`` has no ``gamma_n``."""
    gamma = acc.gamma(n)
    if gamma is None:
        return None
    return gamma * spread * (1 + unit_roundoff(dtype)) + unit_roundoff(dtype) * exact.abs()


def _edited(run: Any, address: str, select: Any = None) -> dict[str, Any] | None:
    info = run.read_info(address, select=select) if select is not None else run.read_info(address)
    return info if info["kind"] in ("edit", "realisation") else None


def _gap(run: Any, reader: Any, *, name: str, address: str, native: Any, exact: Any,
         bound: Any, why: str, info: dict[str, Any] | None, original: Any = None,
         refusal: str | None = None) -> list[Node]:
    """``native - exact`` at a site (residual-frame vectors): an ``edit`` node for what an edit
    wrote there (``native - original``) and a bounded ``remainder`` for the rest. With an edit
    whose pre-edit value is not available the whole gap is the edit node."""
    nodes = []
    if info is not None:
        label = info["authority"] or SUPPLIED
        ops = ",".join(info["ops"]) or info["kind"]
        if original is None:
            return [Node(f"{name} edit", "edit", address, reader.apply(native - exact), label,
                         note=f"written by an edit ({ops}); the pre-edit value is not "
                              "available, so this holds the rounding there too")]
        nodes.append(Node(f"{name} edit", "edit", address, reader.apply(native - original),
                          label, note=f"displacement written by an edit ({ops}): native minus "
                                      "the pre-edit value"))
        native = original
    if bound is None and refusal is not None:
        why = f"{why}; bound refused ({refusal})"
    nodes.append(_remainder(f"{name} remainder", address, reader.apply(native - exact), why,
                            _through(reader, bound)))
    return nodes


def _original(run: Any, address: str) -> Any:
    try:
        value = run.original(address, recompute=True)
    except (ValueError, RuntimeError):
        return None
    return None if value is None else f64(value).reshape(-1)


def _attn_children(run: Any, reader: Any, row: int, layer: int, depth: int, top: int) -> list:
    """Heads of ``L<layer>.attn`` (and, deeper, their sources), the o_proj bias, any edit at
    ``o`` and the bounded o_proj remainder."""
    torch = _torch()
    mix = head_mix(run, row, layer)
    if mix.A is None or mix.o is None:
        raise Refusal(f"t{row}.L{layer}.o has not been computed at {run.where}")
    heads, dh = mix.p.shape[0], mix.A.shape[1]
    blocks = mix.weight.reshape(mix.weight.shape[0], heads, dh)
    checks = mix.checks()   # once per layer: the o_proj bound and every head's value-mix bound
    children: list[Any] = []
    writes = torch.zeros_like(mix.o)
    port = f"t{row}.L{layer}.A"
    for h in range(heads):
        write = blocks[:, h, :] @ mix.A[h]
        writes = writes + write
        children.append(Member(f"L{layer}.h{h}", "head", port, (h,), reader.apply(write)))
    address = f"t{row}.L{layer}.o"
    module = layer_module(run.adapter, layer).self_attn.o_proj
    bias = getattr(module, "bias", None)
    original = None
    info = _edited(run, address)
    if info is not None:
        original = _original(run, address)
        if original is not None and bias is not None:
            original = original - f64(bias)
    if bias is not None:
        children.append(Node(f"L{layer}.o bias", "bias", f"{address} (o_proj.bias)",
                             reader.apply(f64(bias)), EXACT, "P07"))
    children += _gap(run, reader, name=f"L{layer}.attn", address=address, native=mix.o,
                     exact=writes, bound=checks["head_write"]["bound"],
                     why="native o minus sum_h W_o,h A_h (o_proj rounding)", info=info,
                     original=original, refusal=checks["head_write"].get("refused"))
    bounds, refusal = checks["value_mix"]["bound"], checks["value_mix"].get("refused")
    sources = None if depth <= 2 else (
        lambda m: _source_children(run, reader, mix, m.index[0],
                                   None if bounds is None else bounds[m.index[0]], top, refusal))
    return _group(children, top, address, _promoter("P07", sources))


def _source_children(run: Any, reader: Any, mix: Any, head: int, bound: Any, top: int,
                     refusal: str | None = None) -> list:
    """Source rows of one head (``bound``: its value-mix rounding bound, None and ``refusal``
    when its premise fails), the edit at ``A`` if any and the bounded value-mix remainder."""
    row, layer = mix.row, mix.layer
    parts = mix.p[head][:, None] * mix.unit[head]      # (T, d): o_parts() of this head only
    port = f"t{row}.L{layer}.p"
    children: list[Any] = [Member(f"t{s}", "source", port, (head, s), reader.apply(parts[s]))
                           for s in range(parts.shape[0])]
    dh = mix.A.shape[1]
    block = mix.weight.reshape(mix.weight.shape[0], -1, dh)[:, head, :]
    address = f"t{row}.L{layer}.A"
    info = _edited(run, address, head_mask(run, row, layer, head))
    original = None
    if info is not None and not any(_edited(run, f"t{row}.L{layer}.{port}")
                                    for port in ("s", "p")):
        # the pre-edit A recomputes without s/p edits too; usable only when there are none
        whole = _original(run, address)
        if whole is not None:
            original = block @ whole.reshape(-1, dh)[head]
    children += _gap(run, reader, name=f"L{layer}.h{head}", address=f"{address}[h{head}]",
                     native=block @ mix.A[head], exact=block @ mix.head_out()[head],
                     bound=None if bound is None else block.abs() @ bound,
                     why="native A_h minus sum_s p v (value-mix rounding)", info=info,
                     original=original, refusal=refusal)
    return _group(children, top, f"{address}[h{head}]", _promoter("P06"))


def _mlp_children(run: Any, reader: Any, row: int, layer: int, top: int) -> list:
    """Neurons of ``L<layer>.mlp``, any edit at ``m`` and the bounded down_proj remainder;
    ``down_proj`` is cast once and its absolute value taken in place for the bound."""
    f = f64(run.read(f"t{row}.L{layer}.f")).reshape(-1)
    address = f"t{row}.L{layer}.m"
    native = run.read(address)
    m = f64(native).reshape(-1)
    down = f64(layer_module(run.adapter, layer).mlp.down_proj.weight)     # (d, d_ff)
    port = f"t{row}.L{layer}.f"
    children: list[Any] = [Member(f"L{layer}.f{i}", "neuron", port, (i,),
                                  reader.apply(down[:, i] * f[i])) for i in range(f.numel())]
    info = _edited(run, address)
    exact = down @ f
    spread = down.abs_() @ f.abs()      # down is |W_down| from here on: no second copy
    del down
    acc = accumulation(run, native.dtype)
    children += _gap(run, reader, name=f"L{layer}.mlp", address=address, native=m,
                     exact=exact, bound=_matmul_bound(exact, spread, f.numel(), native.dtype, acc),
                     why="native m minus W_down f (down_proj rounding)", info=info,
                     original=None if info is None else _original(run, address),
                     refusal=acc.refusal(f.numel()))
    return _group(children, top, address, _promoter("P15"))


def terms(run: Any, reader: Any, depth: int, top: int) -> list[Node]:
    """Depth-1 nodes of the consumer, without the top remainder. Below depth 1 a node's
    subtree is built when ``_group`` keeps it (``Node.grow``); for an ``o`` consumer the heads
    come grouped already."""
    row = reader.row
    layers = run.adapter.layers
    if reader.kind == "o":
        return _attn_children(run, reader, row, reader.layer, depth + 1, top)
    nodes = []
    h0 = f64(run.read(f"t{row}.L0.h")).reshape(-1)
    info = run.read_info(f"t{row}.L0.h")
    nodes.append(Node("embedding", "embedding", f"t{row}.L0.h", reader.apply(h0),
                      SUPPLIED if info["authority"] == SUPPLIED else EXACT,
                      "P19" if reader.scalar else "P01", port=f"t{row}.L0.h",
                      note=None if info["kind"] == "native" else
                      f"t{row}.L0.h written by an edit ({','.join(info['ops'])})"))
    for layer in range(layers):
        for part, port in (("attn", "o"), ("mlp", "m")):
            if reader.reaches(layer, part) is not None:
                continue
            address = f"t{row}.L{layer}.{port}"
            value = f64(run.read(address)).reshape(-1)
            grow = None
            if depth > 1:
                grow = ((lambda layer=layer: _attn_children(run, reader, row, layer, depth, top))
                        if part == "attn" else
                        (lambda layer=layer: _mlp_children(run, reader, row, layer, top)))
            nodes.append(Node(f"L{layer}.{part}", part, address, reader.apply(value), EXACT,
                              "P19" if reader.scalar else "P14", port=address, index=(),
                              grow=grow))
    if reader.scalar and reader.bias:
        nodes.append(Node("head bias", "bias", "lm_head.bias",
                          f64(reader.bias), EXACT, "P19"))
    return nodes


def _stream(run: Any, reader: Any) -> tuple[Any, list[tuple[str, Any, dict]]]:
    """The residual adds up to the consumer: ``(bound, edits)``. ``bound`` is the
    per-coordinate ``sum 1/2 ulp`` of each add's result (the add redone in the model dtype on
    its native operands, i.e. without any edit of that port); ``edits`` one ``(address,
    displacement, info)`` per residual port an edit wrote (``h`` of a later layer, ``r``,
    ``h_out``): the native value minus that add (or, for ``h``, minus ``h_out`` before it)."""
    from .explain import _half_ulp

    row = reader.row
    last = run.adapter.layers - 1
    upto = (last, 2) if reader.scalar else (reader.layer, {"h": 0, "r": 1, "h_out": 2}[
        reader.port])
    bound = _torch().zeros(run.adapter.model.config.hidden_size, dtype=_torch().float64)
    edits = []
    for layer in range(upto[0] + 1):
        ports = {name: run.read(f"t{row}.L{layer}.{name}") for name in ("h", "o", "r", "m",
                                                                         "h_out")}
        steps = [("h", None)] if layer > 0 else []
        steps += [("r", ("h", "o")), ("h_out", ("r", "m"))]
        for port, operands in steps:
            if layer == upto[0] and ({"h": 0, "r": 1, "h_out": 2}[port] > upto[1]):
                continue
            address = f"t{row}.L{layer}.{port}"
            if operands is None:
                before = run.read(f"t{row}.L{layer - 1}.h_out")
            else:
                before = ports[operands[0]] + ports[operands[1]]
                bound = bound + _half_ulp(before)
            info = _edited(run, address)
            if port == "h" and info is not None and "h" not in (info["by"] or "").split(","):
                info = None     # an edit of h_out before it, booked at that add
            if info is not None:
                edits.append((address, f64(ports[port]).reshape(-1) - f64(before).reshape(-1),
                              info))
    return bound, edits


def top_gap(run: Any, reader: Any, gap: Any) -> list[Node]:
    """The consumer minus its depth-1 terms: an ``edit`` node per edited residual port and the
    bounded remainder (residual-add rounding; for a margin or logit also the final norm and
    the head, as ``explain_margin`` bounds them)."""
    from .explain import _head_bound, _ulp

    bound, edits = _stream(run, reader)
    nodes = []
    for address, displacement, info in edits:
        value = reader.apply(displacement)
        gap = gap - value
        nodes.append(Node(f"{address.split('.', 1)[1]} edit", "edit", address, value,
                          info["authority"] or SUPPLIED,
                          note=f"displacement written by an edit ({','.join(info['ops'])}): "
                               "native minus the add that port performed"))
    why = "native consumer minus the listed terms: residual-add rounding"
    total = _through(reader, bound)
    if reader.scalar:
        torch = _torch()
        row = reader.row
        fn = run.read(f"t{row}.fn")
        logits = run.read(f"t{row}.logits")
        weight = run.adapter.model.lm_head.weight
        bias = getattr(run.adapter.model.lm_head, "bias", None)
        if isinstance(reader.target, Margin):
            a, b, half = reader.target.a, reader.target.b, 1.0
        else:
            a = b = reader.target.i
            half = 0.5          # explain's head bound sums over j in (a, b): counted once
        caps = type("Caps", (), {"normed": fn, "logits": logits})()
        acc = accumulation(run, logits.dtype)
        head, _ = _head_bound(caps, weight, bias, a, b, acc)
        w = f64(weight[a]) - (f64(weight[b]) if a != b else 0)
        final = 2 * float(w.abs().sum()) * _ulp(float(f64(fn).abs().max()),
                                                float(torch.finfo(fn.dtype).eps))
        why += ", final norm (explain's a-priori k = 2 multiple, not a derived bound) and head"
        if head is None:
            total = None
            why += f"; bound refused ({acc.refusal(int(fn.numel()))})"
        else:
            total = total + half * head + final
    nodes.append(_remainder("remainder", reader.text, gap, why, total))
    return nodes


def build(run: Any, consumer: Any, *, top: int = 4, depth: int = 2,
          session_id: str | None = None) -> Circuit:
    """Decompose ``consumer`` at the cursor's row, seal the circuit record (kind ``circuit``)
    and add it to the run's ledger. ``Refusal`` when the consumer is not computed or has no
    exact additive decomposition."""
    if not 1 <= top <= 8:
        raise Refusal("top must be in 1..8")
    if not 1 <= depth <= 3:
        raise Refusal("depth must be 1 (layer terms), 2 (heads, neurons) or 3 (source rows)")
    reader = readout(run, consumer)
    if reader.native is None or not computed(run, _stop(reader)):
        raise Refusal(f"{_stop(reader)} has not been computed at {run.where}: run `until "
                      f"{_stop(reader)}` first")
    nodes = terms(run, reader, depth, top)
    native = f64(reader.native)
    listed = sum((n.value for n in nodes[1:]), nodes[0].value)
    if reader.kind != "o":
        nodes += top_gap(run, reader, native - listed)
        nodes = _group(nodes, top, reader.text)
    root = Node(reader.text, "consumer", reader.text, native,
                reader.native_authority or MEASURED, children=nodes)
    total = sum((n.value for n in nodes[1:]), nodes[0].value)
    closure = native - total
    scale = sum(size(n.value) for n in nodes) + size(native)
    entries = ledger(run)
    ident = max(entries, default=0) + 1
    ids = [0]
    tree = root.view(ids)
    remainders = _remainders(root)
    unexamined = sum(g.unexamined for g in _groups(root))
    body = {"schema": SCHEMA, "consumer": reader.text, "kind": reader.kind,
            "where": run.where, "k": top, "depth": depth, "count": ids[0], "root": tree,
            "closure": {"value": claim(closure, EXACT, "P19" if reader.scalar else "P14"),
                        "bound": num(float(64 * 2.0 ** -53 * scale), "bound"),
                        "within": bool(size(closure) <= 64 * 2.0 ** -53 * scale),
                        "by_construction": True,
                        "meaning": "native consumer minus the sum of the depth-1 nodes: zero "
                                   "by construction (the remainders are defined as native "
                                   "minus the listed terms), so this checks float64 summation "
                                   "only; remainders_within is the rounding check"
                                   + ("; subtrees under the unexamined `+N more` groups were "
                                      "never expanded, so nothing here covers them"
                                      if unexamined else "")},
            "remainders_within": _within(remainders, unexamined),
            "unexamined": unexamined > 0,
            "unexamined_count": num(unexamined, EXACT),
            "edits": {"count": sum(1 for n in _walk(root) if n.kind == "edit"),
                      "meaning": "edit nodes: displacements edits wrote, never in a remainder"},
            "premises": reader.premises}
    try:
        check_bounded(body, "circuit record")
        record = seal_record(session_id=session_id or run_session_id(run), kind="circuit",
                             model_identity=run.adapter.model_identity,
                             execution=dict(run.adapter.execution), authority=EXACT, body=body)
    except RecordError as error:
        raise Refusal(f"the circuit record does not fit one record ({error}); lower top or "
                      "depth") from None
    circuit = Circuit(ident, record, root, reader.target, run.where,
                      complement=_fold_members(root))
    entries[ident] = circuit
    return circuit


def _within(remainders: list[Node], unexamined: int) -> bool | None:
    """Every examined remainder within its bound: False if one is outside, None when a bound is
    refused or a dropped subtree was never expanded (its remainders were not looked at)."""
    if any(n.within is False for n in remainders):
        return False
    if unexamined or any(n.within is None for n in remainders):
        return None
    return True


def _fold_members(root: Node) -> Any:
    """``test``'s direct-path prediction for ablating every ``+N more`` member (the embedding
    aside): minus their contributions summed in member order. The members then keep no value,
    so a built circuit holds one number per group instead of one per folded child."""
    values = []
    for group in _groups(root):
        values += [m.value for m in group.members if m.kind != "embedding"]
        group.members = [m._replace(value=None) for m in group.members]
    return -sum(values[1:], values[0]) if values else None


def _stop(reader: Any) -> str:
    return f"t{reader.row}.logits" if reader.scalar else reader.text


# --------------------------------------------------------------------------- test


def _walk(node: Node) -> list[Node]:
    return [node] + [n for child in node.children for n in _walk(child)]


def _remainders(node: Node) -> list[Node]:
    return [n for n in _walk(node) if n.kind == "remainder"]


def _leaves(node: Node) -> list[Node]:
    if node.kind in (*_FIXED, "more"):
        return []
    if not node.children:
        return [node]
    return [leaf for child in node.children for leaf in _leaves(child)]


def _groups(node: Node) -> list[Node]:
    found = [c for c in node.children if c.kind == "more"]
    for child in node.children:
        found += _groups(child)
    return found


def _mask(run: Any, port: str, kind: str, indices: list[tuple[int, ...]]) -> Any:
    """One bool mask over ``port`` selecting every index in ``indices`` (a neuron of ``f``, a
    head's block of ``A``, a source of ``p``)."""
    torch = _torch()
    mask = torch.zeros(tuple(run.address_shape(port)), dtype=torch.bool)
    if kind == "neuron":
        mask[..., [i for (i,) in indices]] = True
    elif kind == "head":
        dh = run.adapter.head_dim
        for (h,) in indices:
            mask[..., h * dh:(h + 1) * dh] = True
    else:
        for h, s in indices:
            mask[0, h, 0, s] = True
    return mask


def _ablations(run: Any, items: list[Any]) -> list[Edit]:
    """The ``replace 0`` edits that ablate ``items`` (nodes or members), one per port in
    first-use order: a port used once gets its item's own edit; several items at one port share
    one mask (their union) with their selections joined by ``|``, or the whole port when one of
    them is the whole port. An item with no index (the embedding) is not ablatable."""
    ports: dict[str, list[Any]] = {}
    for item in items:
        if item.index is not None:
            ports.setdefault(item.port, []).append(item)
    edits = []
    for port, group in ports.items():
        if len(group) == 1 and group[0].index == ():
            edits.append(Edit(port, 0.0, "replace"))
        elif len(group) == 1:
            item = group[0]
            edits.append(Edit(port, 0.0, "replace", _mask(run, port, item.kind, [item.index]),
                              _selection(item.kind, item.index)))
        elif any(item.index == () for item in group):
            edits.append(Edit(port, 0.0, "replace", None, None, "ablation"))
        else:
            edits.append(Edit(port, 0.0, "replace",
                              _mask(run, port, group[0].kind, [item.index for item in group]),
                              "|".join(_selection(item.kind, item.index) for item in group),
                              "ablation"))
    return edits


def test(run: Any, circuit: Circuit, *, tolerance: float = 0.1) -> dict[str, Any]:
    """Ablate the retained leaves, and separately everything except them, natively; report the
    measured changes beside the direct-path predictions and the sufficiency verdict."""
    from .compare import measure

    if run.where != circuit.where:
        raise Refusal(f"circuit {circuit.id} was built at {circuit.where}; the cursor is at "
                      f"{run.where} (restart there to test it)")
    if not 0 <= tolerance:
        raise Refusal("tolerance must be >= 0")
    leaves = _leaves(circuit.root)
    groups = _groups(circuit.root)
    members = [(group, m) for group in groups for m in group.members]
    skipped = ([(n.ref, n.port) for n in leaves if n.kind == "embedding"]
               + [(f"{m.name} (in {group.ref})", m.port) for group, m in members
                  if m.kind == "embedding"])
    retained = [n for n in leaves if n.kind != "embedding"]
    others = [m for _, m in members if m.kind != "embedding"]
    base = measure(run, [], circuit.consumer)
    out: dict[str, Any] = {"id": circuit.id, "record": circuit.record["id"],
                           "consumer": circuit.record["body"]["consumer"],
                           "base": claim(base["value"], base["authority"]),
                           "tolerance": num(float(tolerance), SUPPLIED)}
    if skipped:
        out["excluded"] = {"nodes": [ref for ref, _ in skipped],
                           "reason": "the embedding is ablated in neither set: zeroing "
                                     f"{skipped[0][1]} removes the input of every other "
                                     "node, not its direct path"}
    deltas = {}
    refs = {"retained": [n.ref for n in retained], "complement": [g.ref for g in groups]}
    predictions = {"retained": (-sum((n.value for n in retained[1:]), retained[0].value)
                                if retained else None),
                   "complement": circuit.complement}
    for name, nodes in (("retained", retained), ("complement", others)):
        edits = _ablations(run, nodes)
        if not edits:
            out[name] = {"state": "unresolved", "reason": f"no ablatable {name} node"}
            continue
        found = measure(run, edits, circuit.consumer)
        delta = found["value"] - base["value"]
        deltas[name] = delta
        predicted = predictions[name]
        out[name] = {"count": len(nodes), "nodes": refs[name][:64],
                     "edits": [e.address + (f"[{e.selection}]" if e.selection else "")
                               for e in edits][:16],
                     "measured": claim(found["value"], found["authority"]),
                     "delta": claim(delta, MEASURED),
                     "predicted_delta": claim(predicted, LOCAL),
                     "prediction": "minus the ablated contributions (direct path; rho and "
                                   "later layers frozen)"}
    if "complement" in deltas:
        limit = tolerance * size(base["value"])
        ok = size(deltas["complement"]) <= limit
        out["verdict"] = "sufficient at this dose" if ok else "not sufficient at this dose"
        out["rule"] = ("|consumer change with everything but the retained set ablated| <= "
                       "tolerance x |base|")
    else:
        out["verdict"] = "unresolved"
        out["rule"] = "nothing outside the retained set to ablate"
    circuit.tests.append(out)
    return out


# Old private names, kept for one commit: symbols.py imported them.
_terms = terms
_top_gap = top_gap

__all__ = ["CIRCUIT", "Circuit", "KINDS", "Node", "SCHEMA", "build", "ledger", "terms", "test",
           "top_gap"]

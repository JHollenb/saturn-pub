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
the native value minus the exact sum) with a rounding ``bound`` (round to nearest; matmuls
accumulate in float32, one rounding to the model dtype; residual adds round once; the final
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
from dataclasses import dataclass, field
from typing import Any

from mdb.core.kinds import CIRCUIT
from mdb.core.pointers import RecordError, check_bounded
from mdb.core.records import num, seal_record

from .finite import BOUND, EXACT, LOCAL, MEASURED, SUPPLIED, f64
from .forecast import Margin, Refusal, _session_id
from .propagate import Edit
from .reads import (
    _U32,
    _gamma,
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


@dataclass
class Node:
    """One node: ``value`` (float64) is the contribution its edge carries into its parent, and
    so into the consumer; ``authority`` labels it. ``edits`` ablate it natively (empty for a
    remainder or bias: they are not components)."""

    name: str
    kind: str
    address: str
    value: Any
    authority: str
    proof: str | None = None
    children: list[Node] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)
    members: list[Node] = field(default_factory=list)
    note: str | None = None
    bound: Any = None
    ref: str | None = None

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


_LEDGERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def ledger(run: Any) -> dict[int, Circuit]:
    """The circuits built on ``run`` in this process, by id."""
    if run not in _LEDGERS:
        _LEDGERS[run] = {}
    return _LEDGERS[run]


# --------------------------------------------------------------------------- decomposition


def _group(children: list[Node], top: int, parent: str) -> list[Node]:
    """Keep the ``top`` largest exact children; the rest become one ``+N more`` node."""
    exact = [c for c in children if c.kind not in _FIXED]
    fixed = [c for c in children if c.kind in _FIXED]
    exact.sort(key=lambda c: -size(c.value))
    kept, rest = exact[:top], exact[top:]
    if rest:
        total = sum((c.value for c in rest[1:]), rest[0].value)
        kept.append(Node(f"+{len(rest)} more", "more", f"{parent} (unlisted)", total, EXACT,
                         members=rest,
                         note=f"+{len(rest)} more, sum of their exact contributions"))
    return kept + fixed


def _remainder(name: str, address: str, value: Any, why: str, bound: Any = None) -> Node:
    return Node(name, "remainder", address, value, MEASURED, note=why, bound=bound)


def _through(reader: Any, bound: Any) -> Any:
    """A per-coordinate bound (residual frame) mapped through the consumer's linear map."""
    bound = f64(bound).reshape(-1)
    return reader.vector.abs() @ bound if reader.scalar else bound


def _matmul_bound(weight: Any, x: Any, dtype: Any) -> Any:
    """``|fl(W x) - W x|`` per output: float32 accumulation in any order, one rounding to
    ``dtype``."""
    exact = weight @ x
    return (_gamma(x.numel(), _U32) * (weight.abs() @ x.abs()) * (1 + unit_roundoff(dtype))
            + unit_roundoff(dtype) * exact.abs())


def _edited(run: Any, address: str, select: Any = None) -> dict[str, Any] | None:
    info = run.read_info(address, select=select) if select is not None else run.read_info(address)
    return info if info["kind"] in ("edit", "realisation") else None


def _gap(run: Any, reader: Any, *, name: str, address: str, native: Any, exact: Any,
         bound: Any, why: str, info: dict[str, Any] | None, original: Any = None) -> list[Node]:
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
    children = []
    writes = torch.zeros_like(mix.o)
    for h in range(heads):
        write = blocks[:, h, :] @ mix.A[h]
        writes = writes + write
        node = Node(f"L{layer}.h{h}", "head", f"t{row}.L{layer}.A[h{h}]", reader.apply(write),
                    EXACT, "P07",
                    edits=[Edit(f"t{row}.L{layer}.A", 0.0, "replace",
                                head_mask(run, row, layer, h), f"h{h}")])
        if depth > 2:
            node.children = _source_children(run, reader, mix, h, top)
        children.append(node)
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
                     exact=writes, bound=_matmul_bound(mix.weight, mix.A.reshape(-1), mix.dtype),
                     why="native o minus sum_h W_o,h A_h (o_proj rounding)", info=info,
                     original=original)
    return _group(children, top, address)


def _source_children(run: Any, reader: Any, mix: Any, head: int, top: int) -> list:
    row, layer = mix.row, mix.layer
    parts = mix.o_parts()[head]                         # (T, d)
    children = []
    for s in range(parts.shape[0]):
        children.append(Node(f"t{s}", "source", f"t{row}.L{layer}.p[h{head},src={s}]",
                             reader.apply(parts[s]), EXACT, "P06",
                             edits=[Edit(f"t{row}.L{layer}.p", 0.0, "replace",
                                         head_mask(run, row, layer, head, s),
                                         f"h{head},src={s}")]))
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
    bound = mix.checks()["value_mix"]["bound"][head]
    children += _gap(run, reader, name=f"L{layer}.h{head}", address=f"{address}[h{head}]",
                     native=block @ mix.A[head], exact=block @ mix.head_out()[head],
                     bound=block.abs() @ bound, why="native A_h minus sum_s p v (value-mix "
                                                     "rounding)", info=info, original=original)
    return _group(children, top, f"{address}[h{head}]")


def _mlp_children(run: Any, reader: Any, row: int, layer: int, top: int) -> list:
    torch = _torch()
    f = f64(run.read(f"t{row}.L{layer}.f")).reshape(-1)
    m = f64(run.read(f"t{row}.L{layer}.m")).reshape(-1)
    down = f64(layer_module(run.adapter, layer).mlp.down_proj.weight)     # (d, d_ff)
    shape = tuple(run.address_shape(f"t{row}.L{layer}.f"))
    children = []
    for i in range(f.numel()):
        mask = torch.zeros(shape, dtype=torch.bool)
        mask[..., i] = True
        children.append(Node(f"L{layer}.f{i}", "neuron", f"t{row}.L{layer}.f[i={i}]",
                             reader.apply(down[:, i] * f[i]), EXACT, "P15",
                             edits=[Edit(f"t{row}.L{layer}.f", 0.0, "replace", mask,
                                         f"i={i}")]))
    address = f"t{row}.L{layer}.m"
    info = _edited(run, address)
    children += _gap(run, reader, name=f"L{layer}.mlp", address=address, native=m,
                     exact=down @ f, bound=_matmul_bound(down, f, run.read(address).dtype),
                     why="native m minus W_down f (down_proj rounding)", info=info,
                     original=None if info is None else _original(run, address))
    return _group(children, top, address)


def _terms(run: Any, reader: Any, depth: int, top: int) -> list[Node]:
    """Depth-1 nodes of the consumer (and their subtrees), without the top remainder."""
    row = reader.row
    layers = run.adapter.layers
    if reader.kind == "o":
        return _attn_children(run, reader, row, reader.layer, depth + 1, top)
    nodes = []
    h0 = f64(run.read(f"t{row}.L0.h")).reshape(-1)
    info = run.read_info(f"t{row}.L0.h")
    nodes.append(Node("embedding", "embedding", f"t{row}.L0.h", reader.apply(h0),
                      SUPPLIED if info["authority"] == SUPPLIED else EXACT,
                      "P19" if reader.scalar else "P01",
                      note=None if info["kind"] == "native" else
                      f"t{row}.L0.h written by an edit ({','.join(info['ops'])})"))
    for layer in range(layers):
        for part, port in (("attn", "o"), ("mlp", "m")):
            if reader.reaches(layer, part) is not None:
                continue
            value = f64(run.read(f"t{row}.L{layer}.{port}")).reshape(-1)
            node = Node(f"L{layer}.{part}", part, f"t{row}.L{layer}.{port}",
                        reader.apply(value), EXACT, "P19" if reader.scalar else "P14",
                        edits=[Edit(f"t{row}.L{layer}.{port}", 0.0, "replace")])
            if depth > 1:
                node.children = (_attn_children(run, reader, row, layer, depth, top)
                                 if part == "attn" else
                                 _mlp_children(run, reader, row, layer, top))
            nodes.append(node)
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


def _top_gap(run: Any, reader: Any, gap: Any) -> list[Node]:
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
        head, _ = _head_bound(caps, weight, bias, a, b)
        w = f64(weight[a]) - (f64(weight[b]) if a != b else 0)
        final = 2 * float(w.abs().sum()) * _ulp(float(f64(fn).abs().max()),
                                                float(torch.finfo(fn.dtype).eps))
        total = total + half * head + final
        why += ", final norm (explain's a-priori k = 2 multiple, not a derived bound) and head"
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
    nodes = _terms(run, reader, depth, top)
    native = f64(reader.native)
    listed = sum((n.value for n in nodes[1:]), nodes[0].value)
    if reader.kind != "o":
        nodes += _top_gap(run, reader, native - listed)
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
    body = {"schema": SCHEMA, "consumer": reader.text, "kind": reader.kind,
            "where": run.where, "k": top, "depth": depth, "count": ids[0], "root": tree,
            "closure": {"value": claim(closure, EXACT, "P19" if reader.scalar else "P14"),
                        "bound": num(float(64 * 2.0 ** -53 * scale), "bound"),
                        "within": bool(size(closure) <= 64 * 2.0 ** -53 * scale),
                        "by_construction": True,
                        "meaning": "native consumer minus the sum of the depth-1 nodes: zero "
                                   "by construction (the remainders are defined as native "
                                   "minus the listed terms), so this checks float64 summation "
                                   "only; remainders_within is the rounding check"},
            "remainders_within": all(n.within for n in remainders),
            "edits": {"count": sum(1 for n in _walk(root) if n.kind == "edit"),
                      "meaning": "edit nodes: displacements edits wrote, never in a remainder"},
            "premises": reader.premises}
    try:
        check_bounded(body, "circuit record")
        record = seal_record(session_id=session_id or _session_id(run), kind="circuit",
                             model_identity=run.adapter.model_identity,
                             execution=dict(run.adapter.execution), authority=EXACT, body=body)
    except RecordError as error:
        raise Refusal(f"the circuit record does not fit one record ({error}); lower top or "
                      "depth") from None
    circuit = Circuit(ident, record, root, reader.target, run.where)
    entries[ident] = circuit
    return circuit


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


def _merged(edits: list[Edit]) -> list[Edit]:
    """One edit per port: the union of the masks of ``replace 0`` edits at the same address."""
    torch = _torch()
    by: dict[str, Edit] = {}
    for edit in edits:
        have = by.get(edit.address)
        if have is None:
            by[edit.address] = edit
            continue
        if have.mask is None or edit.mask is None:
            by[edit.address] = Edit(edit.address, 0.0, "replace", None, None, "ablation")
        else:
            by[edit.address] = Edit(edit.address, 0.0, "replace",
                                    torch.logical_or(have.mask, edit.mask),
                                    f"{have.selection}|{edit.selection}", "ablation")
    return list(by.values())


def test(run: Any, circuit: Circuit, *, tolerance: float = 0.1) -> dict[str, Any]:
    """Ablate the retained leaves, and separately everything except them, natively; report the
    measured changes beside the direct-path predictions and the sufficiency verdict."""
    from .compare import measure

    if run.where != circuit.where:
        raise Refusal(f"circuit {circuit.id} was built at {circuit.where}; the cursor is at "
                      f"{run.where} (restart there to test it)")
    if not 0 <= tolerance:
        raise Refusal("tolerance must be >= 0")
    retained = _leaves(circuit.root)
    others = [member for group in _groups(circuit.root) for member in group.members]
    skipped = [n for n in retained + others if n.kind == "embedding"]
    retained = [n for n in retained if n.kind != "embedding"]
    others = [n for n in others if n.kind != "embedding"]
    base = measure(run, [], circuit.consumer)
    out: dict[str, Any] = {"id": circuit.id, "record": circuit.record["id"],
                           "consumer": circuit.record["body"]["consumer"],
                           "base": claim(base["value"], base["authority"]),
                           "tolerance": num(float(tolerance), SUPPLIED)}
    if skipped:
        group = {id(m): g.ref for g in _groups(circuit.root) for m in g.members}
        out["excluded"] = {"nodes": [n.ref or f"{n.name} (in {group[id(n)]})" for n in skipped],
                           "reason": "the embedding is ablated in neither set: zeroing "
                                     f"{skipped[0].address} removes the input of every other "
                                     "node, not its direct path"}
    deltas = {}
    refs = {"retained": [n.ref for n in retained],
            "complement": [g.ref for g in _groups(circuit.root)]}
    for name, nodes in (("retained", retained), ("complement", others)):
        edits = _merged([e for node in nodes for e in node.edits])
        if not edits:
            out[name] = {"state": "unresolved", "reason": f"no ablatable {name} node"}
            continue
        found = measure(run, edits, circuit.consumer)
        delta = found["value"] - base["value"]
        deltas[name] = delta
        predicted = -sum((n.value for n in nodes[1:]), nodes[0].value)
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


__all__ = ["CIRCUIT", "Circuit", "KINDS", "Node", "SCHEMA", "build", "ledger", "test"]

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
the native value minus the exact sum, i.e. the rounding of the native op and any edit written
there). So at every node the children sum to the node, and the depth-1 nodes plus the top
remainder sum to the native consumer, to float64 roundoff (``closure``).

``test(run, circuit, tolerance)`` ablates natively (``analysis.compare.measure``; masked
``replace 0`` edits at ``L0.h``, ``o``, ``m``, a head's block of ``A``, a neuron of ``f``, a
source of ``p``): the retained leaves, and separately everything except them (every ``+N
more`` group). Both measured changes are reported beside the direct-path prediction (minus
the ablated contributions, ``rho`` and later layers frozen: ``derived-local``). The circuit is
``sufficient at this dose`` when ablating everything else moves the consumer by at most
``tolerance`` times its base size.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

from mdb.core.pointers import RecordError, check_bounded
from mdb.core.records import declare_kind, num, seal_record

from .finite import EXACT, LOCAL, MEASURED, SUPPLIED, f64
from .forecast import Refusal, _session_id
from .propagate import Edit
from .reads import claim, computed, head_mask, head_mix, layer_module, readout, size

SCHEMA = "mdb-circuit-v1"
CIRCUIT = declare_kind("circuit", required=("schema", "consumer", "where", "root", "closure"),
                       plain_numeric=("k", "depth", "count"))
KINDS = ("consumer", "embedding", "attn", "mlp", "head", "neuron", "source", "bias",
         "remainder", "more")


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

    def view(self, ids: list[int]) -> dict[str, Any]:
        ids[0] += 1
        out: dict[str, Any] = {"id": f"n{ids[0]}", "name": self.name, "kind": self.kind,
                               "address": self.address,
                               "value": claim(self.value, self.authority, self.proof)}
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
    exact = [c for c in children if c.kind not in ("remainder", "bias")]
    fixed = [c for c in children if c.kind in ("remainder", "bias")]
    exact.sort(key=lambda c: -size(c.value))
    kept, rest = exact[:top], exact[top:]
    if rest:
        total = sum((c.value for c in rest[1:]), rest[0].value)
        kept.append(Node(f"+{len(rest)} more", "more", f"{parent} (unlisted)", total, EXACT,
                         members=rest,
                         note=f"+{len(rest)} more, sum of their exact contributions"))
    return kept + fixed


def _remainder(name: str, address: str, value: Any, why: str) -> Node:
    return Node(name, "remainder", address, value, MEASURED, note=why)


def _attn_children(run: Any, reader: Any, row: int, layer: int, depth: int, top: int) -> list:
    """Heads of ``L<layer>.attn`` (and, deeper, their sources) plus the o_proj remainder."""
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
    native_o = f64(run.read(f"t{row}.L{layer}.o")).reshape(-1)
    children.append(_remainder(f"L{layer}.attn remainder", f"t{row}.L{layer}.o",
                               reader.apply(native_o - writes),
                               "native o minus sum_h W_o,h A_h (o_proj rounding, bias, edits)"))
    return _group(children, top, f"t{row}.L{layer}.o")


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
    gap = block @ (mix.A[head] - mix.head_out()[head])
    children.append(_remainder(f"L{layer}.h{head} remainder", f"t{row}.L{layer}.A[h{head}]",
                               reader.apply(gap),
                               "native A_h minus sum_s p v (value-mix rounding, edits)"))
    return _group(children, top, f"t{row}.L{layer}.A[h{head}]")


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
    children.append(_remainder(f"L{layer}.mlp remainder", f"t{row}.L{layer}.m",
                               reader.apply(m - down @ f),
                               "native m minus W_down f (down_proj rounding, edits)"))
    return _group(children, top, f"t{row}.L{layer}.m")


def _terms(run: Any, reader: Any, depth: int, top: int) -> list[Node]:
    """Depth-1 nodes of the consumer (and their subtrees)."""
    row = reader.row
    layers = run.adapter.layers
    if reader.kind == "o":
        return _attn_children(run, reader, row, reader.layer, depth + 1, top)
    nodes = []
    h0 = f64(run.read(f"t{row}.L0.h")).reshape(-1)
    nodes.append(Node("embedding", "embedding", f"t{row}.L0.h", reader.apply(h0), EXACT,
                      "P19" if reader.scalar else "P01",
                      edits=[Edit(f"t{row}.L0.h", 0.0, "replace")]))
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
        nodes.append(_remainder("remainder", reader.text, native - listed,
                                "native consumer minus the listed terms: residual-add and "
                                "final-norm/head rounding, h/h_out edits"))
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
    body = {"schema": SCHEMA, "consumer": reader.text, "kind": reader.kind,
            "where": run.where, "k": top, "depth": depth, "count": ids[0], "root": tree,
            "closure": {"value": claim(closure, EXACT, "P19" if reader.scalar else "P14"),
                        "bound": num(float(64 * 2.0 ** -53 * scale), "bound"),
                        "within": bool(size(closure) <= 64 * 2.0 ** -53 * scale),
                        "meaning": "native consumer minus the sum of the depth-1 nodes "
                                   "(float64 roundoff only)"},
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


def _leaves(node: Node) -> list[Node]:
    if node.kind in ("remainder", "bias", "more"):
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
    base = measure(run, [], circuit.consumer)
    out: dict[str, Any] = {"id": circuit.id, "record": circuit.record["id"],
                           "consumer": circuit.record["body"]["consumer"],
                           "base": claim(base["value"], base["authority"]),
                           "tolerance": num(float(tolerance), SUPPLIED)}
    deltas = {}
    for name, nodes in (("retained", retained), ("complement", others)):
        edits = _merged([e for node in nodes for e in node.edits])
        if not edits:
            out[name] = {"state": "unresolved", "reason": f"no ablatable {name} node"}
            continue
        found = measure(run, edits, circuit.consumer)
        delta = found["value"] - base["value"]
        deltas[name] = delta
        predicted = -sum((n.value for n in nodes[1:]), nodes[0].value)
        out[name] = {"count": len(nodes), "edits": [e.address + (f"[{e.selection}]"
                                                                   if e.selection else "")
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

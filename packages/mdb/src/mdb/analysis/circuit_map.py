"""A frozen cross-item circuit map: slots that bind per item, edges between them, a consumer
rule, declared candidate sets and constants frozen from per-item results (advisor memo
2026-11-14, B1).

A symbol table at a cursor is a snapshot of one prompt. A map is the same schema across
prompts: a ``Slot`` is a rule that binds a set of cache cells on each item (``bind`` calls
``symbols.declare``, so the per-item object is the ordinary ``Binding`` and the partition rule
still holds); an ``Edge`` names how one slot reaches another (``read``: the destination's
heads read the source's cells; ``compose``: the answer row's MLPs; ``consume``: the margin);
``Map.freeze`` fills ``constants`` from the items' results with ``summary.summarize`` (numeric
claims and verdict counts) and ``summary.modes`` (categorical paths: a slot's layer, KV head,
best token). The frozen map is JSON with a sha256 (``Map.digest``), committed before any held-out
item is scored; ``predict``-style use binds the frozen slots on new items without refitting.

Row rules (``Slot.rows_rule``), bound per item from the item's roles (``span``: the descriptor
rows, ``gt``: a declared ground-truth row, ``last``: the reader row):

* ``gt``: the item's ``gt`` role;
* ``R_argmax``: the span row (t0 excluded) with the largest summed attention weight of the
  reader row's heads ``rule_heads`` at layer ``rule_layer`` (read from the run, ``p``);
* ``R_last`` / ``R_first``: the last / first span row;
* ``rest``: the span rows no other slot of the map binds (whole rows).

A slot with ``layers`` and ``kv`` binds those value cells of its row(s)
(``cache.L<l>.v[kv<g>,row=<r>]``); without them it binds the whole rows (``t<r>``).

Cell census (``cell_census``): the content of every (row, layer, KV head) value cell at unit
attention mass, ``W_g v`` with ``W_g = sum_{h in kappa^-1(g)} W_O,h`` (P06/P07), read through
``gamma * W_U`` over candidate sets. The final ``rho`` is a positive scalar common to all cells
of a row's readout, so ranks and margins per unit ``rho`` are exact without it (derived-exact,
float64). It runs on a live run's cache or on captured V tensors alike.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from . import summary as summ
from .finite import f64
from .forecast import Refusal
from .symbols import Binding, Candidates, declare, declared

ROLES = ("carrier", "descriptor", "reader", "composer", "consumer")
ROW_RULES = ("gt", "R_argmax", "R_last", "R_first", "rest")
EDGE_KINDS = ("read", "compose", "consume")
SCHEMA = "mdb-circuit-map-v1"


@dataclass(frozen=True)
class Slot:
    name: str
    role: str
    rows_rule: str
    layers: tuple[int, ...] | None = None
    kv: tuple[int, ...] | None = None
    ports: tuple[str, ...] = ("v",)
    reader_rule: str = "last"
    rule_layer: int | None = None
    rule_heads: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.role not in ROLES:
            raise Refusal(f"slot role {self.role!r} is not one of {ROLES}")
        if self.rows_rule not in ROW_RULES:
            raise Refusal(f"row rule {self.rows_rule!r} is not one of {ROW_RULES}")
        if self.rows_rule == "R_argmax" and (self.rule_layer is None or not self.rule_heads):
            raise Refusal("R_argmax needs rule_layer and rule_heads")
        if (self.layers is None) != (self.kv is None):
            raise Refusal("a cell slot gives both layers and kv; a row slot gives neither")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str

    def __post_init__(self):
        if self.kind not in EDGE_KINDS:
            raise Refusal(f"edge kind {self.kind!r} is not one of {EDGE_KINDS}")


@dataclass
class Map:
    slots: list[Slot]
    edges: list[Edge]
    consumer_rule: dict[str, Any]
    candidate_sets: dict[str, list[int]]
    constants: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        names = [s.name for s in self.slots]
        if len(set(names)) != len(names):
            raise Refusal("slot names must be distinct")
        for e in self.edges:
            for end in (e.src, e.dst):
                if end not in names:
                    raise Refusal(f"edge end {end!r} is not a slot of the map")

    def slot(self, name: str) -> Slot:
        return next(s for s in self.slots if s.name == name)

    def to_json(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "slots": [asdict(s) for s in self.slots],
                "edges": [asdict(e) for e in self.edges], "consumer_rule": self.consumer_rule,
                "candidate_sets": self.candidate_sets, "constants": self.constants}

    def digest(self) -> str:
        """sha256 of the canonical JSON (sorted keys, no whitespace)."""
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Map:
        if data.get("schema") != SCHEMA:
            raise Refusal(f"not a {SCHEMA} map")

        def tup(x):
            return None if x is None else tuple(x)
        slots = [Slot(s["name"], s["role"], s["rows_rule"], tup(s["layers"]), tup(s["kv"]),
                      tuple(s["ports"]), s["reader_rule"], s["rule_layer"], tup(s["rule_heads"]))
                 for s in data["slots"]]
        edges = [Edge(**e) for e in data["edges"]]
        return cls(slots, edges, data["consumer_rule"], data["candidate_sets"],
                   data.get("constants", {}))


def _reader_row(run: Any, item: dict[str, Any], rule: str) -> int:
    if rule == "last":
        return int(item.get("last", int(run.where.split(".")[0][1:])))
    if rule in item:
        return int(item[rule])
    raise Refusal(f"reader rule {rule!r} names no role of the item")


def slot_rows(slot: Slot, run: Any, item: dict[str, Any], taken: set[int] = frozenset()) -> list[int]:
    """The rows ``slot`` binds on ``item`` (see the module docstring)."""
    span = [int(r) for r in item["span"]]
    if not span:
        raise Refusal(f"item {item.get('name', '?')} has an empty span")
    rule = slot.rows_rule
    if rule == "gt":
        if "gt" not in item:
            raise Refusal(f"slot {slot.name}: the item has no gt role")
        return [int(item["gt"])]
    if rule == "R_last":
        return [span[-1]]
    if rule == "R_first":
        return [span[0]]
    if rule == "rest":
        return [r for r in span if r not in taken]
    reader = _reader_row(run, item, slot.reader_rule)
    p = f64(run.read(f"t{reader}.L{slot.rule_layer}.p"))
    p = p.reshape(p.shape[-3] if p.dim() >= 3 else 1, -1)       # (heads, rows)
    rows = [r for r in span if r != 0]
    if not rows:
        raise Refusal(f"slot {slot.name}: the span holds only t0")
    score = {r: float(sum(p[h, r] for h in slot.rule_heads)) for r in rows}
    return [max(rows, key=lambda r: (score[r], r))]


def slot_addresses(slot: Slot, rows: list[int]) -> list[str]:
    if slot.layers is None:
        return [f"t{r}" for r in rows]
    return [f"cache.L{layer}.{port}[kv{g},row={r}]" for r in rows for layer in slot.layers
            for g in slot.kv for port in slot.ports]


def bind(m: Map, run: Any, item: dict[str, Any]) -> dict[str, Binding]:
    """Declare every slot of ``m`` on ``run`` for ``item`` (``rest`` slots last, over the rows the
    others left). Declared symbols must not overlap, so a row slot never binds a row that holds
    a cell slot's cells (``rest`` skips those rows). Returns slot name -> Binding; a slot that
    binds no row is skipped."""
    out: dict[str, Binding] = {}
    taken: set[int] = set()
    cell_rows: set[int] = set()
    order = [s for s in m.slots if s.rows_rule != "rest"] + [s for s in m.slots
                                                             if s.rows_rule == "rest"]
    for slot in order:
        if slot.role in ("composer", "consumer", "reader") and slot.layers is None and \
                slot.rows_rule != "rest":
            continue                     # rule slots with no cells: not cache bindings
        rows = slot_rows(slot, run, item, taken | cell_rows)
        if slot.layers is None:
            rows = [r for r in rows if r not in cell_rows]
        if not rows:
            continue
        out[slot.name] = declare(run, slot.name, slot_addresses(slot, rows))
        (taken if slot.layers is None else cell_rows).update(rows)
    return out


def item_report(m: Map, run: Any, item: dict[str, Any], *, consumer: Any = None,
                truths: dict[str, int | None] | None = None, tokenizer: Any = None,
                content_layers: bool = False) -> dict[str, Any]:
    """One item's report: bind the map's slots, then ``symbols.table`` at the cursor over the
    declared slots only (they page first), with content over the map's candidate sets
    (``truths``: set name -> the item's true token). The declared symbols are removed after."""
    from .symbols import table

    bound = bind(m, run, item)
    try:
        sets = [Candidates(name, tuple(tokens), (truths or {}).get(name))
                for name, tokens in m.candidate_sets.items()]
        n = max(1, len(bound))
        tab = table(run, consumer=consumer, limit=min(n, 16), sets=sets or None,
                    content_layers=content_layers, tokenizer=tokenizer)
        rows = {name: b.addresses for name, b in bound.items()}
        return {"item": item.get("name"), "slots": rows, "table": tab.report}
    finally:
        for name in list(bound):
            declared(run).pop(name, None)


def _bindings_by_name(report: dict[str, Any]) -> dict[str, Any]:
    """``report`` with ``table.bindings`` re-keyed by symbol name (``table.bindings.<slot>``),
    so cross-item statistics line up by slot even when an item's slot bound no row (``bind``
    skips it, which would shift every later position)."""
    table = report.get("table")
    if not isinstance(table, dict) or not isinstance(table.get("bindings"), list):
        return report
    keyed = {}
    for i, entry in enumerate(table["bindings"]):
        name = (entry.get("symbol") or entry.get("name")) if isinstance(entry, dict) else None
        keyed[str(name) if name is not None else f"#{i}"] = entry
    return {**report, "table": {**table, "bindings": keyed}}


def freeze(m: Map, reports: list[dict[str, Any]], categorical: list[str]) -> Map:
    """``m`` with ``constants`` filled from per-item ``reports``: ``summary`` (numeric claims and
    verdict counts over the paths every report has) and ``modes`` (the most common value of
    each categorical dotted path, with counts). Table bindings are keyed by symbol name
    (``table.bindings.<slot>.*``), never by page position."""
    constants = dict(m.constants)
    keyed = [_bindings_by_name(r) for r in reports]
    constants["summary"] = summ.summarize(keyed)
    constants["modes"] = summ.modes(keyed, categorical)
    constants["n_items"] = len(reports)
    return Map(m.slots, m.edges, m.consumer_rule, m.candidate_sets, constants)


# --------------------------------------------------------------------------- cell census


def group_writers(o_weight: Any, heads: int, kv: int) -> Any:
    """``W_g = sum_{h in kappa^-1(g)} W_O,h`` per KV group: (kv, d_model, head_dim) float64."""
    torch = _torch()
    w = f64(o_weight)
    d_model, inner = w.shape
    dh = inner // heads
    per_head = w.reshape(d_model, heads, dh).permute(1, 0, 2)          # (H, d, dh)
    group = torch.arange(heads) // (heads // kv)
    return torch.stack([per_head[group == g].sum(0) for g in range(kv)])


def cell_census(values: dict[int, Any], writers: dict[int, Any], rows: list[int],
                readout_rows: dict[str, Any]) -> dict[str, Any]:
    """Scores of every (layer, row, KV head) value cell at unit mass over candidate sets.

    ``values[l]``: the layer's V cache ``(1, kv, T, dh)`` (or ``(kv, T, dh)``); ``writers[l]``:
    ``group_writers`` of the layer; ``readout_rows[name]``: ``gamma * W_U[tokens]`` (float64,
    ``(n_tokens, d_model)``). Returns name -> tensor ``(layers, rows, kv, n_tokens)`` (float64)
    and the layer order. Exact per unit final ``rho`` (P06/P07, P19)."""
    torch = _torch()
    layers = sorted(values)
    out = {}
    for name, ro in readout_rows.items():
        per_layer = []
        for layer in layers:
            v = f64(values[layer])
            v = v.reshape(v.shape[-3], v.shape[-2], v.shape[-1])      # (kv, T, dh)
            proj = torch.einsum("nd,gde->gne", f64(ro), f64(writers[layer]))   # (kv, n, dh)
            sel = v[:, rows, :]                                         # (kv, R, dh)
            per_layer.append(torch.einsum("gne,gre->rgn", proj, sel))   # (R, kv, n)
        out[name] = torch.stack(per_layer)
    return {"layers": layers, "rows": rows, "scores": out}


def rank_of(scores: Any, tokens: list[int], truth: int | None) -> int | None:
    """1-based rank of ``truth`` among ``tokens`` (ties count against it); None if absent."""
    if truth is None or truth not in tokens:
        return None
    where = tokens.index(truth)
    return int((scores > scores[where]).sum()) + 1


def _torch():
    import torch
    return torch

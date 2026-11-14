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
import re
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


@dataclass(frozen=True)
class LabelSymbol:
    """A declared label (ported from ``mdb_lanes.compiled_state.Symbol``): a finite value
    (``cardinality`` distinct values at most) fixed by the tokens at rows up to
    ``visible_from`` (a row, or the name of an item role). A binding whose rows all lie before
    that row cannot see the label: ``item_report`` lists it under ``hidden_labels`` instead
    (a token position is never silently reinterpreted, the ``symbol_lifecycle`` rule)."""

    name: str
    cardinality: int | None = None
    visible_from: int | str | None = None


@dataclass
class Map:
    slots: list[Slot]
    edges: list[Edge]
    consumer_rule: dict[str, Any]
    candidate_sets: dict[str, list[int]]
    constants: dict[str, Any] = field(default_factory=dict)
    symbols: list[LabelSymbol] = field(default_factory=list)

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
        out = {"schema": SCHEMA, "slots": [asdict(s) for s in self.slots],
               "edges": [asdict(e) for e in self.edges], "consumer_rule": self.consumer_rule,
               "candidate_sets": self.candidate_sets, "constants": self.constants}
        if self.symbols:            # only when declared: older maps keep their digest
            out["symbols"] = [asdict(s) for s in self.symbols]
        return out

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
                   data.get("constants", {}),
                   [LabelSymbol(**x) for x in data.get("symbols", ())])


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
    sink = {int(r) for r in item.get("sink", (0,))}               # sink rows (default t0)
    rows = [r for r in span if r not in sink]
    if not rows:
        raise Refusal(f"slot {slot.name}: the span holds only sink rows")
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
    binds no row is skipped (``item_report`` lists it under ``unbound_slots``). Each Binding
    carries its slot's ``role``, ``slot`` and the map's ``map_digest``."""
    out: dict[str, Binding] = {}
    digest = m.digest()
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
        binding = declare(run, slot.name, slot_addresses(slot, rows))
        binding.role, binding.slot, binding.map_digest = slot.role, slot.name, digest
        out[slot.name] = binding
        (taken if slot.layers is None else cell_rows).update(rows)
    return out


def cache_slots(m: Map) -> list[str]:
    """The slots ``bind`` declares as cache bindings (rule-only slots excluded)."""
    return [s.name for s in m.slots
            if not (s.role in ("composer", "consumer", "reader") and s.layers is None
                    and s.rows_rule != "rest")]


def item_report(m: Map, run: Any, item: dict[str, Any], *, consumer: Any = None,
                truths: dict[str, int | None] | None = None, tokenizer: Any = None,
                content_layers: bool = False, sets: list[Any] | None = None,
                labels: dict[str, Any] | None = None, frame: bool = False,
                **options: Any) -> dict[str, Any]:
    """One item's report: bind the map's slots, then ``symbols.table`` at the cursor over the
    declared slots only (they page first), with content over the map's candidate sets
    (``truths``: set name -> the item's true token). The declared symbols are removed after.

    ``sets``: ``candidates.CandidateSet`` objects (flags, pairs) used instead of the map's
    token lists; ``labels``: item labels (default ``item["labels"]``), carried on every row;
    ``options``: passed to ``symbols.table`` (``co``, ``lexical_null``, ``switch``, ``derive``,
    ``sink``, ``foreign``, ``causal``). The report lists ``unbound_slots`` (slots that bound no
    row on this item) and, with ``frame``, keeps the ``Table`` under ``"_table"`` (not JSON)."""
    from .symbols import table

    labels = dict(item.get("labels") or {}, **(labels or {}))
    bound = bind(m, run, item)
    try:
        if sets is not None:
            cands = [s.candidates((truths or {}).get(s.name)) if hasattr(s, "candidates")
                     else s for s in sets]
        else:
            cands = [Candidates(name, tuple(tokens), (truths or {}).get(name))
                     for name, tokens in m.candidate_sets.items()]
        n = max(1, len(bound))
        tab = table(run, consumer=consumer, limit=min(n, 16), sets=cands or None,
                    content_layers=content_layers, tokenizer=tokenizer, labels=labels or None,
                    **options)
        rows = {name: b.addresses for name, b in bound.items()}
        tokens = [int(t) for t in run.tokens]
        out = {"item": item.get("name"), "slots": rows, "table": tab.report,
               "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               "unbound_slots": [s for s in cache_slots(m) if s not in bound],
               "map_digest": m.digest()}
        if labels:
            from .symbols import label_claim

            out["labels"] = {k: label_claim(v) for k, v in labels.items()}
            _hide_labels(m, item, bound, tab.report)
        if frame:
            out["_table"] = tab
        return out
    finally:
        for name in list(bound):
            declared(run).pop(name, None)


def _hide_labels(m: Map, item: dict[str, Any], bound: dict[str, Binding],
                 report: dict[str, Any]) -> None:
    """Move a label off the rows of a binding that cannot see it (``LabelSymbol``)."""
    for sym in m.symbols:
        if sym.visible_from is None:
            continue
        row = sym.visible_from if isinstance(sym.visible_from, int) else item.get(sym.visible_from)
        if row is None:
            continue
        for view in report.get("bindings", []):
            binding = bound.get(view.get("symbol"))
            if binding is None or sym.name not in view.get("labels", {}):
                continue
            if max(c.row for c in binding.cells) < int(row):
                view["labels"].pop(sym.name)
                view.setdefault("hidden_labels", []).append(sym.name)


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


def freeze(m: Map, reports: list[dict[str, Any]], categorical: list[str], *,
           switch: str | None = None, ties: str = "value",
           supplied: dict[str, Any] | None = None) -> Map:
    """``m`` with ``constants`` filled from per-item ``reports``: ``summary`` (numeric claims and
    verdict counts over the paths every report has) and ``modes`` (the most common value of
    each categorical dotted path, with counts). Table bindings are keyed by symbol name
    (``table.bindings.<slot>.*``), never by page position. Mode ties are broken by value
    (``ties``; AUDIT-B1: first appearance depends on item order) and reported.

    ``switch``: a label path (``labels.family.label``, or ``labels.depth.value`` for a number): constants are also stored per switch
    value under ``by_switch`` (the pooled ``summary`` stays the fallback for an unseen value,
    the ``mdb_lanes.compiled_state`` rule). ``supplied``: constants measured elsewhere, each
    ``supplied_constant(...)`` (value plus the record it came from), never free text."""
    constants = dict(m.constants)
    keyed = [_bindings_by_name(r) for r in reports]
    constants["summary"] = summ.summarize(keyed)
    constants["modes"] = summ.modes(keyed, categorical, ties=ties)
    constants["n_items"] = len(reports)
    unbound: dict[str, int] = {}
    for r in reports:
        for name in r.get("unbound_slots", ()):
            unbound[name] = unbound.get(name, 0) + 1
    if unbound:
        constants["unbound_slots"] = unbound
    for sym in m.symbols:
        if sym.cardinality is None:
            continue
        seen = {summ.json_key(_label(r, f"labels.{sym.name}")) for r in reports} - {None}
        if len(seen) > sym.cardinality:
            raise Refusal(f"label {sym.name} takes {len(seen)} values, more than its declared "
                          f"cardinality {sym.cardinality}")
    if switch is not None:
        constants["by_switch"] = summ.summarize_by(keyed, switch)
        constants["switch_fallback"] = "pooled summary for an unseen switch value"
    if supplied:
        for name, entry in supplied.items():
            if not isinstance(entry, dict) or "source" not in entry:
                raise Refusal(f"supplied constant {name} needs its source record "
                              "(circuit_map.supplied_constant)")
        constants["supplied"] = dict(supplied)
    return Map(m.slots, m.edges, m.consumer_rule, m.candidate_sets, constants, list(m.symbols))


def supplied_constant(value: Any, *, source: str, authority: str = "measured",
                      note: str = "") -> dict[str, Any]:
    """A constant measured elsewhere, with its provenance: ``source`` is the record id (or
    sealed file sha256) it came from (``mdb_lanes.computation_atlas``: effects are
    caller-supplied measurements with provenance)."""
    if not isinstance(source, str) or not source.strip():
        raise Refusal("a supplied constant names the record it came from")
    return {"value": value, "authority": authority, "source": source, "note": note,
            "label": "supplied"}


# --------------------------------------------------------------------------- MapTable

_SITE = re.compile(r"cache\.L(?P<l>\d+)\.[kv]\[kv(?P<g>\d+),row=(?P<r>\d+)\]")


def _site(report: dict[str, Any], slot: str) -> Any:
    """``(layer, kv, row - last)`` of a slot's first cell on this item, or ``("row", row -
    last)`` for a row slot (positions relative to the reader row, so items of different
    lengths compare)."""
    addresses = (report.get("slots") or {}).get(slot) or []
    if not addresses:
        return None
    last = None
    where = (report.get("table") or {}).get("where", "")
    if where.startswith("t"):
        last = int(where.split(".")[0][1:])
    match = _SITE.search(addresses[0])
    if match:
        r = int(match["r"])
        return (int(match["l"]), int(match["g"]), r - last if last is not None else r)
    if addresses[0].startswith("t") and addresses[0][1:].isdigit():
        r = int(addresses[0][1:])
        return ("row", r - last if last is not None else r)
    return addresses[0]


def _binding(report: dict[str, Any], slot: str) -> dict[str, Any] | None:
    bindings = (report.get("table") or {}).get("bindings") or []
    if isinstance(bindings, dict):
        return bindings.get(slot)
    return next((b for b in bindings if isinstance(b, dict) and b.get("symbol") == slot), None)


def _label(report: dict[str, Any], switch: str) -> Any:
    try:
        return summ.leaf(report, switch)
    except (KeyError, IndexError, ValueError, TypeError):
        return None


@dataclass
class MapTable:
    """A cross-item view of a map: one row per slot (review 2.1). Every row keeps the per-item
    values behind it (``items``: item name, site, the binding row), so different native
    locations are never silently merged (the ``mdb_lanes.symbol_trace`` safeguard)."""

    rows: list[dict[str, Any]]
    items: list[dict[str, Any]]
    switch: str | None = None

    @classmethod
    def build(cls, m: Map, reports: list[dict[str, Any]], *, switch: str | None = None,
              null: str | None = "answer_permutation") -> MapTable:
        names = [s.name for s in m.slots if s.name in set(cache_slots(m))]
        rows, items = [], []
        for slot in m.slots:
            if slot.name not in names:
                continue
            found = [(r, _binding(r, slot.name)) for r in reports]
            bound = [(r, b) for r, b in found if b is not None]
            row: dict[str, Any] = {"slot": slot.name, "role": slot.role, "n_items": len(reports),
                                   "n_bound": len(bound)}
            for key in ("mass", "to_consumer", "contrast", "mass_nonsink", "mask_delta"):
                values = [(b[key]["value"], b[key]["authority"]) for _, b in bound
                          if isinstance(b.get(key), dict) and "value" in b[key]
                          and isinstance(b[key]["value"], (int, float))]
                if values:
                    row[key] = summ.numeric_stat(key, values)
            sites = [{"site": json.dumps(_site(r, slot.name))} for r, _ in bound]
            mode = summ.modes(sites, ["site"], ties="value")[0]
            row["site_mode"] = mode
            row["baselines"] = {"constant_site": {
                "same_site": mode["count"], "n": mode["n"],
                "meaning": "items whose bound site (layer, kv, row - last) equals the modal "
                           "site: a rule that always picks one constant site scores the same "
                           "on these (TP1: the constant L20 KV6 site matched the forecast "
                           "174/178)"}}
            content = {}
            for set_name in m.candidate_sets or {}:
                entries = [(r, (b.get("content") or {}).get(set_name)) for r, b in bound]
                entries = [(r, e) for r, e in entries if isinstance(e, dict) and "best" in e]
                if not entries:
                    continue
                named = [e for _, e in entries if "names_truth" in e]
                best_mode = summ.modes([{"best": e["best"]} for _, e in entries], ["best"],
                                       ties="value")[0]
                item = {"n": len(entries), "best_mode": best_mode}
                if named:
                    item["names_truth"] = {"count": sum(bool(e["names_truth"]) for e in named),
                                           "n": len(named)}
                if null == "answer_permutation":
                    item["answer_permutation"] = _permutation_null(entries)
                    for other in sorted({k for _, e in entries for k in (e.get("co") or {})}):
                        projected = [(r, {**e["co"][other]["projected"],
                                          **({"truth": e["truth"]} if "truth" in e else {})})
                                     for r, e in entries if other in (e.get("co") or {})]
                        item.setdefault("co_permutation", {})[other] = \
                            _permutation_null(projected)
                for flag in ("collision", "first_subword"):
                    flagged = [e for _, e in entries if (e.get("flags") or {}).get(flag)]
                    if flagged:
                        item[f"{flag}_count"] = len(flagged)
                content[set_name] = item
            if content:
                row["content"] = content
            if switch is not None:
                groups: dict[Any, list[float]] = {}
                for r, b in bound:
                    value = _label(r, switch)
                    tc = b.get("to_consumer")
                    if isinstance(tc, dict) and isinstance(tc.get("value"), (int, float)):
                        groups.setdefault(summ.json_key(value), []).append(
                            (tc["value"], tc["authority"]))
                row["by_switch"] = [{"value": k, "n": len(v),
                                     "to_consumer": summ.numeric_stat("to_consumer", v)}
                                    for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))]
            rows.append(row)
            for r, b in bound:
                items.append({"item": r.get("item"), "slot": slot.name,
                              "site": _site(r, slot.name), "binding": b,
                              "prompt_sha256": r.get("prompt_sha256"),
                              "labels": r.get("labels")})
        return cls(rows, items, switch)

    def frame(self) -> Any:
        """The per-item rows behind the statistics as a long DataFrame."""
        from .frame import flatten

        import pandas as pd

        out = []
        for entry in self.items:
            row = {"item": entry["item"], "slot": entry["slot"],
                   "site": json.dumps(entry["site"])}
            for key, value in (entry.get("labels") or {}).items():
                row[f"item_labels.{key}"] = (value.get("label", value.get("value"))
                                             if isinstance(value, dict) else value)
            row.update(flatten(entry["binding"], skip=("readers",)))
            out.append(row)
        return pd.DataFrame(out)

    def to_json(self) -> dict[str, Any]:
        return {"schema": SCHEMA + "/table", "rows": self.rows, "switch": self.switch,
                "items": [{**e, "site": e["site"]} for e in self.items]}


def _permutation_null(entries: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """The answer-permutation null of ``names_truth``: the expected count when each item's best
    is compared with the other items' truths (mean over all derangement pairs; AUDIT-SCALE:
    the real chance is an answer permutation, not a shuffled direction)."""
    pairs = [(e["best"], e["truth"]) for _, e in entries if "truth" in e]
    n = len(pairs)
    if n < 2:
        return {"state": "unresolved", "reason": "needs truth on at least two items "
                                                 "(content.<set>.truth)"}
    hits = sum(1 for i, (b, _) in enumerate(pairs) for j, (_, t) in enumerate(pairs)
               if i != j and b == t)
    observed = sum(1 for b, t in pairs if b == t)
    return {"observed": observed, "expected": hits / (n - 1), "n": n,
            "meaning": "expected names_truth count if each item's truth were another item's"}


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


_WRITERS: dict[int, Any] = {}


def group_writer(adapter: Any, layer: int, kv: int | None = None) -> Any:
    """``group_writers`` of ``layer``'s ``o_proj`` for an attached model, cached per weight
    tensor (keyed by the tensor's identity and its in-place version counter, so an edited weight
    is recomputed): ``(kv, d_model, head_dim)`` float64, or one group's ``(d_model, head_dim)``
    when ``kv`` is given. Replaces the per-script ``W_O`` sums (gap audit P11)."""
    import weakref

    from .reads import layer_module

    weight = layer_module(adapter, layer).self_attn.o_proj.weight
    entry = _WRITERS.get(id(weight))
    if entry is None or entry[0]() is not weight or entry[1] != weight._version:
        for key in [k for k, e in _WRITERS.items() if e[0]() is None]:
            del _WRITERS[key]                          # tensors that are gone
        entry = (weakref.ref(weight), weight._version,
                 group_writers(weight, adapter.num_heads, adapter.num_kv_heads))
        _WRITERS[id(weight)] = entry
    return entry[2] if kv is None else entry[2][kv]


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

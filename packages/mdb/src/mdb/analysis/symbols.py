"""The symbol table at the cursor: one row per binding, with definition sites, lifetime, named
readers and status, and an ``Unbound`` row so the table sums to the whole.

A binding is a set of cache cells ``(row, layer, KV head, k|v)``. The engine-native binding is
a token row ``t<s>``: its value enters the cache at every layer (``t<s>.L*.k``,
``t<s>.L*.v``). A user symbol (``declare(run, name, addresses)``) is any set of cells (``t2``,
``t2.L0.v``, ``cache.L1.v[kv0,row=2]``); declared symbols must not overlap, and a token row
holds exactly the cells no declared symbol holds (an explicit cell set), so the bindings
partition the cells. A read ``(reader row, layer, head, source)`` belongs to the binding that
holds its value cell ``v``: the value is what the read carries into the head (P06). ``k``
cells set the weights; a binding of ``k`` cells only receives no read.

At the reader rows (the cursor's row by default) each binding gets, from the exact value mix
(P06/P07, ``analysis.reads``):

* ``mass``: the attention weight its reads receive, summed over heads and layers (``count``
  reads, ``mean_mass`` per read); each weight is labelled as ``Run.read_info`` labels it
  (``supplied`` under a ``p``/``s`` edit), the sum ``derived-exact`` (``supplied`` if any part
  is);
* ``to_consumer``: its exact contribution to the consumer (``derived-exact``): a margin or logit
  (default at ``logits``/``sample``: the margin between the row's two largest logits), a
  residual port, or with no consumer mid-row the norm of each read's share of its layer's ``o``.
  A head whose block of ``A`` (or whose ``o``) an edit wrote does not reach the consumer through
  its value mix: those reads are ``refused`` and so is the binding's total;
* ``readers``: the top few (row, layer, head) reads by contribution, each with mass and effect;
* ``lifetime``: a strip per reader row, one mark per layer (``legend``): ``#`` read with mass at
  or above ``min_mass`` (default 0.1, stated), ``.`` read below it, ``-`` the binding holds no
  value cell at that layer, ``?`` not computed;
* ``status``: ``edited`` (``Run.read_info`` of its ``k``/``v`` says an edit wrote it or an edit
  of its row lies upstream), else ``read`` (some read at or above ``min_mass``), ``live`` (none
  yet and some layer of the reader row has not computed its ``p``), ``unread``.

``Unbound`` holds, per head, the mass of the bindings not on this page, and the part of the
consumer no listed binding's reads carry, split into its parts: the embedding, MLP (and head
bias) terms (``derived-exact``, P19), the attention not listed (the layers' native ``o`` read
exactly, minus the listed reads), any edit of the residual stream, and the measured remainder
with its rounding bound (``analysis.circuits``). ``accounting`` checks that the bindings
partition every head's mass (the sum over all bindings equals ``sum_s p``); a broken partition
is ``refused`` and so is ``Unbound``. With a project the table is sealed as ``symbol-table``
records, one per page of at most ``PAGE`` bindings.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

from mdb.core.kinds import SYMBOL_TABLE, procedure
from mdb.core.records import seal_record

from .finite import BOUND, EXACT, MEASURED, SUPPLIED, f64, num
from .forecast import Margin, Refusal, run_session_id
from .reads import (
    Source,
    claim,
    computed,
    edited_block,
    first_row,
    head_mix,
    parse_source,
    readout,
    refused,
    size,
    source_label,
    unit_roundoff,
)

SCHEMA = "mdb-symbol-table-v1"
PAGE = 16
STATUSES = ("live", "read", "unread", "edited", "unresolved")
LEGEND = {"#": "read with mass at or above min_mass", ".": "read below min_mass",
          "-": "no value cell of this binding at that layer", "?": "not computed"}
_U32 = 2.0 ** -24
_PORTS = ("k", "v")


@dataclass
class Binding:
    """A named set of cache cells: ``cells`` are ``Source`` items (row, layers, KV heads,
    ports)."""

    name: str
    kind: str
    cells: list[Source]
    addresses: list[str]

    def covers(self, row: int, layer: int, kv: int, port: str) -> bool:
        return any(c.row == row and c.covers(layer, kv, port) for c in self.cells)


_DECLARED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def declared(run: Any) -> dict[str, Binding]:
    """The user symbols declared on ``run`` in this process, by name."""
    if run not in _DECLARED:
        _DECLARED[run] = {}
    return _DECLARED[run]


def declare(run: Any, name: str, addresses: list[str]) -> Binding:
    """Declare (or redeclare) symbol ``name`` over ``addresses``; ``Refusal`` when it overlaps
    another declared symbol, names a row the cursor's row cannot read or is not a cache row."""
    if not name or not name.replace("_", "").replace("-", "").isalnum() or name == "Unbound":
        raise Refusal(f"a symbol name is letters, digits, '_' or '-' (and not Unbound), not "
                      f"{name!r}")
    if not addresses:
        raise Refusal("a symbol needs at least one address")
    here = int(run.where.split(".")[0][1:])
    cells = [parse_source(run, text) for text in addresses]
    late = [c for c in cells if c.row > here]
    if late:
        raise Refusal(f"t{late[0].row} has not been computed at {run.where}")
    for cell in cells:
        if cell.layers is None or cell.row < first_row(run):
            continue
        for layer in cell.layers:
            for port in cell.ports or _PORTS:
                address = f"t{cell.row}.L{layer}.{port}"
                if not computed(run, address):
                    raise Refusal(f"{address} has not been computed at {run.where}")
    binding = Binding(name, "declared", cells, [c.text for c in cells])
    layers, kv = run.adapter.layers, run.adapter.num_kv_heads
    for other in declared(run).values():
        if other.name == name:
            continue
        for cell in cells:
            for layer in range(layers):
                for k in range(kv):
                    for port in _PORTS:
                        if cell.covers(layer, k, port) and other.covers(cell.row, layer, k,
                                                                         port):
                            raise Refusal(f"{name} overlaps symbol {other.name} at "
                                          f"cache.L{layer}.{port}[kv{k},row={cell.row}]")
    declared(run)[name] = binding
    return binding


def _token_rows(run: Any, here: int, users: list[Binding]) -> list[Binding]:
    """One binding per token row holding exactly the cells no declared symbol holds."""
    layers, kv = run.adapter.layers, run.adapter.num_kv_heads
    out = []
    for s in range(here + 1):
        cells = [(layer, k, port) for layer in range(layers) for k in range(kv)
                 for port in _PORTS if not any(b.covers(s, layer, k, port) for b in users)]
        if not cells:
            continue
        if len(cells) == layers * kv * len(_PORTS):
            out.append(Binding(f"t{s}", "token", [Source(s, text=f"t{s}")],
                               [f"t{s}.L*.k", f"t{s}.L*.v"]))
            continue
        out.append(Binding(f"t{s}", "token",
                           [Source(s, (layer,), (k,), f"cache.L{layer}.{port}[kv{k},row={s}]",
                                   (port,)) for layer, k, port in cells],
                           [f"t{s} (uncovered part: {len(cells)} of "
                            f"{layers * kv * len(_PORTS)} k/v cells)"]))
    return out


def _default_consumer(run: Any) -> Margin | None:
    row, stop = run.where.split(".", 1)
    if stop not in ("logits", "sample"):
        return None
    logits = f64(run.read(f"{row}.logits")).reshape(-1)
    first, second = (int(i) for i in logits.topk(2).indices)
    return Margin(int(row[1:]), first, second)


def _status_edits(run: Any, row: int, layers: int, here: tuple[int, int]) -> list[str]:
    """The edits ``Run.read_info`` reports on this row's ``k``/``v`` (own or upstream in the
    row); a prefilled row has none."""
    if row < first_row(run):
        return []
    found: list[str] = []
    for layer in range(layers):
        for port in ("k", "v"):
            address = f"t{row}.L{layer}.{port}"
            if run.parse(address) > here:
                continue
            info = run.read_info(address)
            if info["edited"]:
                found.append(f"{address} ({','.join(info['ops'])})")
            for edit in info["upstream_edits"]:
                if int(edit["row"]) == row:
                    found.append(f"t{row}.L{edit['layer']}.{edit['port']} ({edit['op']})")
    return list(dict.fromkeys(found))


@dataclass
class Table:
    """A symbol table: ``report`` (bounded JSON), ``records`` (sealed, one per page) and the
    float64 ``tensors`` behind it: ``mass[(row, layer)]`` (bindings x heads) with
    ``unbound_mass`` and ``total_mass`` per head, ``contribution`` per binding."""

    report: dict[str, Any]
    records: list[dict[str, Any]]
    tensors: dict[str, Any] = field(default_factory=dict)


def table(run: Any, *, consumer: Any = None, rows: str | None = None, min_mass: float = 0.1,
          readers: int = 3, offset: int = 0, limit: int = PAGE,
          session_id: str | None = None, tokenizer: Any = None,
          pending: dict[str, dict[str, Any]] | None = None) -> Table:
    """The symbol table at the cursor (see the module docstring); ``offset``/``limit`` page
    the bindings (declared symbols first, then token rows in order). ``pending``: declared
    symbols that cannot be bound on this run yet (name -> ``addresses``, ``reason``), listed
    ``unresolved``."""
    from .reads import reader_rows

    if not 0 <= min_mass <= 1:
        raise Refusal("min_mass must be in 0..1")
    if not 1 <= limit <= PAGE:
        raise Refusal(f"limit must be in 1..{PAGE}")
    here_key = run.parse(run.where)
    layers = run.adapter.layers
    found_rows = reader_rows(run, Source(0), rows)
    if consumer is None:
        consumer = _default_consumer(run)
    reader = None if consumer is None else readout(run, consumer)
    users = list(declared(run).values())
    bindings = users + _token_rows(run, here_key[0], users)
    n = len(bindings)
    grid = _Grid(run, bindings, reader, found_rows)
    entries = [_entry(run, binding, b, grid, reader, found_rows, layers, min_mass, readers,
                      here_key, tokenizer) for b, binding in enumerate(bindings)]
    shown = entries[offset:offset + limit]
    listed = sorted(range(offset, min(offset + limit, n)))
    keys = sorted(grid.mass)
    unbound_mass = {key: grid.total[key] - grid.mass[key][listed].sum(0) for key in keys}
    report: dict[str, Any] = {
        "where": run.where, "rows": found_rows,
        "consumer": (reader.text if reader is not None else
                     "none: per-layer o (norm of each read's share); give consumer= or run "
                     "until logits"),
        "min_mass": num(float(min_mass), SUPPLIED),
        "mass_meaning": "a binding's mass is a sum over the heads and layers that read it "
                        "(count reads; mean_mass per read)",
        "attribution": "a read (row, layer, head, source) belongs to the binding holding its "
                       "value cell v",
        "legend": LEGEND,
        "bindings": [e["view"] for e in shown],
        "page": {"offset": offset, "limit": limit, "total": n},
    }
    if pending:
        report["unresolved_symbols"] = [
            {"symbol": name, "kind": "declared", "definitions": item["addresses"],
             "status": "unresolved", "reason": item["reason"]}
            for name, item in sorted(pending.items())][:PAGE]
    broken = grid.partition_error()
    listed_sum = sum((float(grid.mass[k][listed].sum()) for k in keys), 0.0)
    unbound: dict[str, Any] = {
        "symbol": "Unbound",
        "mass": refused(broken) if broken else num(
            sum(float(unbound_mass[k].sum()) for k in keys), EXACT),
        "meaning": "attention mass of the bindings not on this page, per head summed; the "
                   "consumer part no listed binding's reads carry, by part",
    }
    contribution = {}
    if broken:
        unbound["readout"] = refused(broken)
    elif reader is not None and reader.scalar:
        listed_effect = sum(entries[b]["contribution"] for b in listed
                            if entries[b]["contribution"] is not None)
        unbound["readout"] = num(float(f64(reader.native)) - listed_effect, MEASURED)
        unbound["parts"] = _unbound_parts(run, reader, listed_effect)
        report["consumer_total"] = claim(f64(reader.native), reader.native_authority
                                         or MEASURED)
        contribution = {b: e["contribution"] for b, e in enumerate(entries)}
    else:
        unbound["readout"] = {"state": "unresolved",
                              "reason": "no scalar consumer: the per-layer o shares have no "
                                        "common total"}
    report["unbound"] = unbound
    report["accounting"] = {
        "heads": sum(int(grid.total[k].numel()) for k in keys),
        "listed_mass": num(listed_sum, EXACT),
        "max_head_deviation": num(grid.deviation, MEASURED),
        "deviation_bound": num(grid.bound, BOUND),
        "partition": refused(broken) if broken else {
            "max_gap": num(grid.gap, EXACT),
            "meaning": "per head, the sum over all bindings of their mass minus sum_s p "
                       "(float64 summation order only)"},
        "within_bound": grid.deviation <= grid.bound and not broken,
        "meaning": "per head, bindings + Unbound = sum_s p (native); |sum_s p - 1| against "
                   "T u_dtype + 2 T u_f32 (softmax in float32, each weight cast to the dtype)",
    }
    if grid.missing:
        report["not_computed"] = grid.missing[:16]
    records = _records(run, report, session_id)
    report["records"] = [r["id"] for r in records]
    tensors = {"mass": grid.mass, "effect": grid.effect, "total_mass": grid.total,
               "unbound_mass": unbound_mass, "contribution": contribution,
               "bindings": [b.name for b in bindings], "listed": listed}
    return Table(report, records, tensors)


def _remainder_bound(rest: Any) -> dict[str, Any]:
    """The remainder's rounding bound, or a refused bound when its premise fails (no number)."""
    if rest.bound is None:
        return {"bound": {"state": "refused", "reason": "the rounding bound's accumulation "
                          "premise fails on this run (see `explain`)"}, "within": None}
    return {"bound": num(float(rest.bound), BOUND), "within": bool(rest.within)}


def _unbound_parts(run: Any, reader: Any, listed_effect: float) -> dict[str, Any]:
    """The consumer minus the listed bindings' reads, by part (see the module docstring)."""
    from .circuits import terms, top_gap

    listed = terms(run, reader, 1, 1)
    native = float(f64(reader.native))
    attention = sum(float(t.value) for t in listed if t.kind == "attn")
    other = sum(float(t.value) for t in listed if t.kind != "attn")
    tail = top_gap(run, reader, f64(native - attention - other))
    edits = [t for t in tail if t.kind == "edit"]
    rest = tail[-1]
    parts: dict[str, Any] = {
        "embedding_mlp": num(other, EXACT, "P19"),
        "attention_not_listed": num(attention - listed_effect, EXACT, "P19"),
        "remainder": {"value": num(float(rest.value), MEASURED), **_remainder_bound(rest)},
        "meaning": "embedding, MLP and head-bias terms; the layers' attention output (native o, "
                   "read exactly) minus the listed reads (reads of bindings off this page, "
                   "heads an edit wrote, o_proj rounding); residual-stream edits; the measured "
                   "remainder (residual-add, final-norm and head rounding) with its bound",
    }
    if edits:
        labels = {t.authority for t in edits}
        parts["residual_edits"] = num(sum(float(t.value) for t in edits),
                                      SUPPLIED if SUPPLIED in labels else EXACT)
    return parts


class _Grid:
    """Per (reader row, layer): each binding's mass, effect and reads per head, from the value
    mix; reads are owned by the binding holding their value cell."""

    def __init__(self, run: Any, bindings: list[Binding], reader: Any, rows: list[int]):
        torch = _torch()
        n, heads, layers = len(bindings), run.adapter.num_heads, run.adapter.layers
        self.mass: dict[tuple[int, int], Any] = {}
        self.effect: dict[tuple[int, int], Any] = {}
        self.total: dict[tuple[int, int], Any] = {}
        self.reach: dict[tuple[int, int], str | None] = {}
        self.cover: dict[tuple[int, int], Any] = {}
        self.sources: dict[tuple[int, int], Any] = {}
        self.supplied: dict[tuple[int, int], Any] = {}
        self.blocked: dict[tuple[int, int], dict[int, str]] = {}
        self.missing: list[str] = []
        self.owners: list[str] = []
        self.deviation = self.bound = self.gap = self.tolerance = 0.0
        weights: dict[int, Any] = {}    # float64 o_proj by layer, cast once for all rows
        for j in rows:
            for layer in range(layers):
                try:
                    mix = head_mix(run, j, layer, weights)
                except Refusal as error:
                    self.missing.append(f"t{j}.L{layer}: {error}")
                    continue
                key = (j, layer)
                parts = mix.o_parts()                               # (H, T, d)
                count = mix.p.shape[1]
                owner = []
                for kv in range(run.adapter.num_kv_heads):
                    row_owner = []
                    for s in range(count):
                        found = [b for b, bd in enumerate(bindings)
                                 if bd.covers(s, layer, kv, "v")]
                        if len(found) != 1:
                            self.owners.append(f"cache.L{layer}.v[kv{kv},row={s}] is held by "
                                               f"{len(found)} bindings")
                        row_owner.append(found[0] if found else None)
                    owner.append(row_owner)
                edited_p = run.read_info(f"t{j}.L{layer}.p")["kind"] != "native"
                m = torch.zeros(n, heads, dtype=torch.float64)
                cover = torch.zeros(n, heads, dtype=torch.bool)
                sources = torch.zeros(n, heads, dtype=torch.int64)
                supplied = torch.zeros(n, heads, dtype=torch.bool)
                width = () if reader is None or reader.scalar else (parts.shape[-1],)
                e = torch.zeros(n, heads, *width, dtype=torch.float64)
                why = None if reader is None else (reader.reaches(layer, "attn") or (
                    None if reader.row == j else f"{reader.text} is in another row"))
                blocked = {}
                for h in range(heads):
                    reason = edited_block(mix, run, h)
                    if reason:
                        blocked[h] = reason
                    share = torch.zeros(n, parts.shape[-1], dtype=torch.float64)
                    for s in range(count):
                        b = owner[mix.kappa[h]][s]
                        if b is None:
                            continue
                        m[b, h] += mix.p[h, s]
                        cover[b, h] = True
                        sources[b, h] += 1
                        share[b] += parts[h, s]
                        if edited_p and source_label(run, mix, h, s) == SUPPLIED:
                            supplied[b, h] = True
                    for b in range(n):
                        if not cover[b, h]:
                            continue
                        if reader is None:
                            e[b, h] = share[b].norm()
                        elif why is None and h not in blocked:
                            e[b, h] = reader.apply(share[b])
                self.mass[key], self.cover[key], self.effect[key] = m, cover, e
                self.sources[key], self.supplied[key] = sources, supplied
                self.reach[key], self.blocked[key] = why, blocked
                self.total[key] = mix.p.sum(1)
                self.deviation = max(self.deviation, float((mix.p.sum(1) - 1).abs().max()))
                self.bound = max(self.bound, count * unit_roundoff(mix.dtype) + 2 * count * _U32)
                self.gap = max(self.gap, float((m.sum(0) - mix.p.sum(1)).abs().max()))
                self.tolerance = max(self.tolerance, 4 * count * 2.0 ** -53
                                     * float(mix.p.abs().sum(1).max()))

    def partition_error(self) -> str | None:
        """Why the bindings do not partition the heads' mass, or None."""
        if self.owners:
            return "the bindings do not partition the value cells: " + self.owners[0]
        if self.gap > self.tolerance:
            return (f"per head the bindings' mass differs from sum_s p by {self.gap:.3g} "
                    f"(> {self.tolerance:.3g}, float64 summation)")
        return None


def _torch():
    import torch

    return torch


def _entry(run, binding, b, grid, reader, found_rows, layers, min_mass, readers, here_key,
           tokenizer) -> dict[str, Any]:
    """One binding's row of the table (``view``) and its float64 contribution (None when an
    edit blocks one of its reads)."""
    strips, first, last = [], None, None
    total_mass, count, supplied = 0.0, 0, False
    contribution: float | None = 0.0
    blocked_by = None
    scalar = reader is not None and reader.scalar
    reads = []
    for j in found_rows:
        marks = []
        for layer in range(layers):
            key = (j, layer)
            if key not in grid.mass:
                marks.append("?")
                continue
            covered = grid.cover[key][b]
            if not bool(covered.any()):
                marks.append("-")
                continue
            m = grid.mass[key][b]
            total_mass += float(m.sum())
            count += int(covered.sum())
            supplied = supplied or bool(grid.supplied[key][b].any())
            above = bool((m[covered] >= min_mass).any())
            marks.append("#" if above else ".")
            if above:
                first = first or f"t{j}.L{layer}"
                last = f"t{j}.L{layer}"
            e = grid.effect[key][b]
            for h in range(len(covered)):
                if not covered[h]:
                    continue
                reason = grid.blocked[key].get(h)
                value = e[h]
                if reason and reader is not None:
                    shown, rank = refused(reason), -1.0
                    if scalar and grid.reach[key] is None:
                        blocked_by = blocked_by or reason
                elif scalar and grid.reach[key] is not None:
                    shown, rank = refused(grid.reach[key]), -1.0
                elif reader is None:
                    shown, rank = {"norm": num(float(value), EXACT, "P07")}, float(value)
                else:
                    shown = claim(value, EXACT, "P19" if reader.scalar else "P07")
                    rank = size(value)
                    if scalar:
                        contribution += float(value)
                label = (SUPPLIED if grid.supplied[key][b, h] else
                         MEASURED if int(grid.sources[key][b, h]) == 1 else EXACT)
                reads.append((rank, {"name": f"t{j}.L{layer}.h{h}",
                                     "mass": num(float(m[h]), label), "effect": shown}))
        strips.append(f"t{j}:{''.join(marks)}")
    reads.sort(key=lambda r: -r[0])
    edits = []
    for row in sorted({c.row for c in binding.cells}):
        edits += _status_edits(run, row, layers, here_key)
    pending = any(c == "?" for strip in strips for c in strip.split(":", 1)[1])
    if edits:
        status = "edited"
    elif first is not None:
        status = "read"
    elif pending:
        status = "live"
    else:
        status = "unread"
    label = SUPPLIED if supplied else EXACT
    view: dict[str, Any] = {
        "symbol": binding.name, "kind": binding.kind,
        "rows": sorted({c.row for c in binding.cells}),
        "definitions": binding.addresses,
        "lifetime": {"strip": " ".join(strips), "first": first, "last": last},
        "mass": num(total_mass, label),
        "count": count,
        "mean_mass": (num(total_mass / count, label) if count else
                      {"state": "unresolved", "reason": "no read of this binding"}),
        "readers": [r[1] for r in reads[:readers]],
        "status": status,
    }
    if binding.kind == "token":
        row = binding.cells[0].row
        token = _token_at(run, row)
        if token is not None:
            view["token"] = token
            if tokenizer is not None:
                view["text"] = tokenizer.decode([token])
    if blocked_by is not None:
        contribution = None
        view["to_consumer"] = refused(blocked_by)
    elif scalar:
        view["to_consumer"] = num(contribution, EXACT, "P19")
    else:
        contribution = None
        view["to_consumer"] = {"state": "unresolved", "reason": "no scalar consumer"}
    if edits:
        view["edited_by"] = edits[:8]
    if len(reads) > readers:
        view["omitted"] = len(reads) - readers
    return {"view": view, "contribution": contribution}


def _token_at(run: Any, row: int) -> int | None:
    if row < len(run.tokens):
        return int(run.tokens[row])
    held = run.session.read("tokens")[0]
    return int(held[row]) if row < held.numel() else None


def _records(run: Any, report: dict[str, Any], session_id: str | None) -> list[dict[str, Any]]:
    """The table sealed as one ``symbol-table`` record (this page)."""
    body = {"schema": SCHEMA, **{k: report[k] for k in ("where", "consumer", "rows", "min_mass",
                                                         "bindings", "unbound", "page",
                                                         "accounting")}}
    if "consumer_total" in report:
        body["consumer_total"] = report["consumer_total"]
    body["procedure"] = procedure(run)
    return [seal_record(session_id=session_id or run_session_id(run), kind="symbol-table",
                        model_identity=run.adapter.model_identity,
                        execution=dict(run.adapter.execution), authority=EXACT, body=body)]


__all__ = ["Binding", "LEGEND", "PAGE", "STATUSES", "SYMBOL_TABLE", "Table", "declare",
           "declared", "table"]

"""The symbol table at the cursor: one row per binding, with definition sites, lifetime, named
readers and status, and an ``Unbound`` row so the table sums to the whole.

A binding is a set of cache rows a head can read. The engine-native binding is a token row
``t<s>``: its value enters the cache at every layer (``t<s>.L*.k``, ``t<s>.L*.v``). A user
symbol (``declare(run, name, addresses)``) is any set of cache rows, optionally restricted to
layers and KV heads (``t2``, ``t2.L0.v``, ``cache.L1.v[kv0,row=2]``); declared symbols must not
overlap, and token rows list only what no declared symbol covers, so the rows partition the
sources.

At the reader rows (the cursor's row by default) each binding gets, from the exact value mix
(P06/P07, ``analysis.reads``):

* ``mass``: the attention weight its rows receive, summed over the computed heads and layers
  (each weight ``measured``; the sum exact arithmetic over them);
* ``to_consumer``: its exact contribution to the consumer (``derived-exact``): a margin or logit
  (default at ``logits``/``sample``: the margin between the row's two largest logits), a
  residual port, or with no consumer mid-row the norm of each read's share of its layer's ``o``;
* ``readers``: the top few (row, layer, head) reads by contribution, each with mass and effect;
* ``lifetime``: a strip per reader row, one character per layer: ``#`` some head reads it with
  mass above ``min_mass`` (default 0.1, stated), ``.`` computed and below, ``?`` not computed;
  ``first``/``last`` the first and last such read;
* ``status``: ``edited`` (``Run.read_info`` of its ``k``/``v`` says an edit wrote it or an edit
  of its row lies upstream), else ``read`` (some read above ``min_mass``), ``live`` (no read
  above it yet and some layer of the reader row has not computed its ``p``), ``unread``.

``Unbound`` holds, per head, the mass of the rows not in a listed binding, and the part of the
consumer that no listed binding's reads carry (the other attention reads, the embedding, the
MLPs and the measured remainders): ``native - sum(listed)``, ``measured``. With a project the
table is sealed as ``symbol-table`` records, one per page of at most ``PAGE`` bindings.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

from mdb.core.records import declare_kind, seal_record

from .finite import BOUND, EXACT, MEASURED, SUPPLIED, f64, num
from .forecast import Margin, Refusal, _session_id
from .reads import (
    Source,
    _first_row,
    claim,
    head_mix,
    parse_source,
    readout,
    size,
    unit_roundoff,
)

SCHEMA = "mdb-symbol-table-v1"
PAGE = 16
STATUSES = ("live", "read", "unread", "edited")
SYMBOL_TABLE = declare_kind(
    "symbol-table",
    required=("schema", "where", "consumer", "bindings", "unbound", "page"),
    plain_numeric=("page.offset", "page.limit", "page.total", "rows", "bindings.*.rows",
                   "bindings.*.token", "bindings.*.omitted", "accounting.heads"),
)
_U32 = 2.0 ** -24


@dataclass
class Binding:
    """A named set of cache rows: ``cells`` are ``Source`` items (row, layers, KV heads)."""

    name: str
    kind: str
    cells: list[Source]
    addresses: list[str]

    def covers(self, row: int, layer: int, kv: int) -> bool:
        return any(c.row == row and c.covers(layer, kv) for c in self.cells)


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
    binding = Binding(name, "declared", cells, [c.text for c in cells])
    layers, kv = run.adapter.layers, run.adapter.num_kv_heads
    for other in declared(run).values():
        if other.name == name:
            continue
        for cell in cells:
            for layer in range(layers):
                for k in range(kv):
                    if cell.covers(layer, k) and other.covers(cell.row, layer, k):
                        raise Refusal(f"{name} overlaps symbol {other.name} at "
                                      f"cache.L{layer}.v[kv{k},row={cell.row}]")
    declared(run)[name] = binding
    return binding


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
    if row < _first_row(run):
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
          session_id: str | None = None, tokenizer: Any = None) -> Table:
    """The symbol table at the cursor (see the module docstring); ``offset``/``limit`` page
    the bindings (declared symbols first, then token rows in order)."""
    torch = _torch()
    from .reads import reader_rows

    if not 0 <= min_mass <= 1:
        raise Refusal("min_mass must be in 0..1")
    if not 1 <= limit <= PAGE:
        raise Refusal(f"limit must be in 1..{PAGE}")
    here_key = run.parse(run.where)
    here = here_key[0]
    layers, heads = run.adapter.layers, run.adapter.num_heads
    found_rows = reader_rows(run, Source(0), rows)
    if consumer is None:
        consumer = _default_consumer(run)
    reader = None if consumer is None else readout(run, consumer)
    bindings = list(declared(run).values())
    for s in range(here + 1):
        cells = []
        for layer in range(layers):
            for k in range(run.adapter.num_kv_heads):
                if not any(b.covers(s, layer, k) for b in bindings if b.kind == "declared"):
                    cells.append((layer, k))
        if not cells:
            continue
        full = len(cells) == layers * run.adapter.num_kv_heads
        layer_set = tuple(sorted({c[0] for c in cells}))
        kv_set = tuple(sorted({c[1] for c in cells}))
        source = Source(s, None if full else layer_set, None if full else kv_set, f"t{s}")
        bindings.append(Binding(f"t{s}", "token", [source],
                                [f"t{s}.L*.k", f"t{s}.L*.v"] if full else
                                [f"t{s} (uncovered part)"]))
    n = len(bindings)
    mass: dict[tuple[int, int], Any] = {}
    effect: dict[tuple[int, int], Any] = {}
    total_mass: dict[tuple[int, int], Any] = {}
    reach: dict[tuple[int, int], str | None] = {}
    cover: dict[tuple[int, int], Any] = {}
    not_computed: list[str] = []
    deviation = 0.0
    bound = 0.0
    for j in found_rows:
        for layer in range(layers):
            try:
                mix = head_mix(run, j, layer)
            except Refusal as error:
                not_computed.append(f"t{j}.L{layer}: {error}")
                continue
            parts = mix.o_parts()                                   # (H, T, d)
            m = torch.zeros(n, heads, dtype=torch.float64)
            c = torch.zeros(n, heads, dtype=torch.bool)
            if reader is None:
                e = torch.zeros(n, heads, dtype=torch.float64)
            elif reader.scalar:
                e = torch.zeros(n, heads, dtype=torch.float64)
            else:
                e = torch.zeros(n, heads, parts.shape[-1], dtype=torch.float64)
            why = None if reader is None else (reader.reaches(layer, "attn") or (
                None if reader.row == j else f"{reader.text} is in another row"))
            for b, binding in enumerate(bindings):
                for h in range(heads):
                    rows_in = [s for s in range(mix.p.shape[1])
                               if binding.covers(s, layer, mix.kappa[h])]
                    if not rows_in:
                        continue
                    c[b, h] = True
                    m[b, h] = mix.p[h, rows_in].sum()
                    share = parts[h, rows_in].sum(0)
                    if reader is None:
                        e[b, h] = share.norm()
                    elif why is None:
                        e[b, h] = reader.apply(share)
            mass[(j, layer)] = m
            cover[(j, layer)] = c
            effect[(j, layer)] = e
            reach[(j, layer)] = why
            total_mass[(j, layer)] = mix.p.sum(1)
            deviation = max(deviation, float((mix.p.sum(1) - 1).abs().max()))
            t = mix.p.shape[1]
            bound = max(bound, t * unit_roundoff(mix.dtype) + 2 * t * _U32)
    computed_keys = sorted(mass)
    entries = []
    for b, binding in enumerate(bindings):
        entries.append(_entry(run, binding, b, mass, effect, cover, reach, reader, found_rows,
                              layers, heads, min_mass, readers, here_key, tokenizer))
    shown = entries[offset:offset + limit]
    listed = set(range(offset, min(offset + limit, n)))
    unbound_mass = {key: total_mass[key] - mass[key][sorted(listed)].sum(0) if listed
                    else total_mass[key] for key in computed_keys}
    report: dict[str, Any] = {
        "where": run.where, "rows": found_rows,
        "consumer": (reader.text if reader is not None else
                     "none: per-layer o (norm of each read's share); give consumer= or run "
                     "until logits"),
        "min_mass": num(float(min_mass), SUPPLIED),
        "bindings": [e["view"] for e in shown],
        "page": {"offset": offset, "limit": limit, "total": n},
    }
    total_heads = sum(int(total_mass[k].numel()) for k in computed_keys)
    listed_sum = sum((float(mass[k][sorted(listed)].sum()) for k in computed_keys), 0.0)
    unbound: dict[str, Any] = {
        "symbol": "Unbound",
        "mass": num(sum(float(unbound_mass[k].sum()) for k in computed_keys), EXACT),
        "meaning": "attention mass on rows of bindings not on this page, per head summed; the "
                   "consumer part no listed binding's reads carry",
    }
    contribution = {}
    if reader is not None and reader.scalar:
        listed_effect = 0.0
        for k in computed_keys:
            if reach[k] is None:
                listed_effect += float(effect[k][sorted(listed)].sum())
        all_attention = sum(float(effect[k].sum()) for k in computed_keys if reach[k] is None)
        native = float(f64(reader.native))
        unbound["readout"] = num(native - listed_effect, MEASURED)
        unbound["parts"] = {
            "unlisted_attention": num(all_attention - listed_effect, EXACT),
            "not_attention": num(native - all_attention, MEASURED),
            "meaning": "reads of bindings not on this page (exact); embedding, MLPs, heads of "
                       "other rows and measured rounding (native minus all attention reads)",
        }
        report["consumer_total"] = claim(f64(reader.native), reader.native_authority
                                         or MEASURED)
        contribution = {b: e["contribution"] for b, e in enumerate(entries)}
    else:
        unbound["readout"] = {"state": "unresolved",
                              "reason": "no scalar consumer: the per-layer o shares have no "
                                        "common total"}
    report["unbound"] = unbound
    report["accounting"] = {
        "heads": total_heads,
        "listed_mass": num(listed_sum, EXACT),
        "max_head_deviation": num(deviation, MEASURED),
        "deviation_bound": num(bound, BOUND),
        "within_bound": deviation <= bound,
        "meaning": "per head, listed rows + Unbound = sum_s p (native); |sum_s p - 1| against "
                   "T u_dtype + 2 T u_f32 (softmax in float32, each weight cast to the dtype)",
    }
    if not_computed:
        report["not_computed"] = not_computed[:16]
    records = _records(run, report, session_id)
    report["records"] = [r["id"] for r in records]
    tensors = {"mass": mass, "effect": effect, "total_mass": total_mass,
               "unbound_mass": unbound_mass, "contribution": contribution,
               "bindings": [b.name for b in bindings], "listed": sorted(listed)}
    return Table(report, records, tensors)


def _torch():
    import torch

    return torch


def _entry(run, binding, b, mass, effect, cover, reach, reader, found_rows, layers, heads,
           min_mass, readers, here_key, tokenizer) -> dict[str, Any]:
    """One binding's row of the table (``view``) and its float64 contribution."""
    strips, first, last = [], None, None
    total_mass = 0.0
    contribution = 0.0
    reads = []
    for j in found_rows:
        chars = []
        for layer in range(layers):
            key = (j, layer)
            if key not in mass:
                chars.append("?")
                continue
            m = mass[key][b]
            total_mass += float(m.sum())
            above = bool((m > min_mass).any())
            chars.append("#" if above else ".")
            if above:
                first = first or f"t{j}.L{layer}"
                last = f"t{j}.L{layer}"
            e = effect[key][b]
            if reader is not None and reader.scalar and reach[key] is None:
                contribution += float(e.sum())
            for h in range(heads):
                if not cover[key][b, h]:
                    continue
                value = e[h]
                if reader is not None and reader.scalar and reach[key] is not None:
                    shown = {"state": "refused", "reason": reach[key]}
                    rank = -1.0
                elif reader is None:
                    shown = {"norm": num(float(value), EXACT, "P07")}
                    rank = float(value)
                else:
                    shown = claim(value, EXACT, "P19" if reader.scalar else "P07")
                    rank = size(value)
                reads.append((rank, {"name": f"t{j}.L{layer}.h{h}",
                                     "mass": num(float(m[h]), MEASURED), "effect": shown}))
        strips.append(f"t{j}:{''.join(chars)}")
    reads.sort(key=lambda r: -r[0])
    edits = []
    for cell in binding.cells:
        edits += _status_edits(run, cell.row, layers, here_key)
    pending = any(c == "?" for strip in strips for c in strip.split(":", 1)[1])
    if edits:
        status = "edited"
    elif first is not None:
        status = "read"
    elif pending:
        status = "live"
    else:
        status = "unread"
    view: dict[str, Any] = {
        "symbol": binding.name, "kind": binding.kind,
        "rows": sorted({c.row for c in binding.cells}),
        "definitions": binding.addresses,
        "lifetime": {"strip": " ".join(strips), "first": first, "last": last},
        "mass": num(total_mass, EXACT),
        "readers": [r[1] for r in reads[:readers]],
        "status": status,
    }
    if binding.kind == "token":
        row = binding.cells[0].row
        if row < len(run.tokens) or row <= here_key[0]:
            token = _token_at(run, row)
            if token is not None:
                view["token"] = token
                if tokenizer is not None:
                    view["text"] = tokenizer.decode([token])
    if reader is not None and reader.scalar:
        view["to_consumer"] = num(contribution, EXACT, "P19")
    else:
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
    return [seal_record(session_id=session_id or _session_id(run), kind="symbol-table",
                        model_identity=run.adapter.model_identity,
                        execution=dict(run.adapter.execution), authority=EXACT, body=body)]


__all__ = ["Binding", "PAGE", "STATUSES", "SYMBOL_TABLE", "Table", "declare", "declared",
           "table"]

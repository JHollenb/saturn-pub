"""Long-frame export of symbol tables, circuit maps and traces, with a published JSON Schema
(symbol-table additions review 2026-11-14, 2.2).

``to_frame(table, level=...)`` turns a ``symbols.Table`` into a long pandas DataFrame:

* ``binding``: one row per listed binding, every field of the table row flattened to dotted
  columns (``content.countries.best``); a labelled number becomes its value column plus an
  ``<column>__auth`` column holding its authority label; lists become JSON text. Each row also
  carries ``item``, ``where``, ``consumer``, ``model_identity`` and ``record_id`` (the sealed
  record the numbers came from: the record stays the authority, the frame is a view).
* ``read``: one row per (binding, reader row, layer, head) read, from the float64 tensors:
  ``mass``, ``effect`` (scalar consumer), ``contrast``, ``mask_delta``, ``sources`` (value
  cells of the binding read by the head), ``supplied``, and with ``foreign=True`` the
  TransformerLens / nnsight names of the read.
* ``head``: one row per (reader row, layer, head): ``total_mass`` and ``unbound_mass``.

``to_arrow``/``to_parquet`` keep the schema name and record id in the Arrow schema metadata;
``from_frame`` re-imports a binding frame read-only (labelled numbers rebuilt from the
``__auth`` columns), so an analysis notebook needs no model. ``trace_frame`` flattens a
``rows.Trace`` (one row per step, a column per table entry), ``map_frame`` the per-item rows of
a circuit map's reports.

``COLUMNS`` is the column reference (name, type, authority, meaning) that ``json_schema`` and
``docs/SYMBOL-TABLE-SCHEMA.md`` are generated from.
"""

from __future__ import annotations

import json
import numbers
from typing import Any

AUTH = "__auth"
SCHEMAS = ("mdb-symbol-table-v1", "mdb-circuit-map-v1", "mdb-trace-v1")

# (column, JSON type, authority, meaning). ``claim``: a labelled number {value, authority}.
COLUMNS: list[tuple[str, str, str, str]] = [
    ("symbol", "string", "structural", "the binding's name (a declared symbol, a slot, or t<row>)"),
    ("kind", "string", "structural", "declared, token, or component"),
    ("rows", "array", "structural", "source rows; half-open ranges a:b past 64 rows"),
    ("definitions", "array", "structural", "the addresses that define it (first 64, then a count)"),
    ("lifetime.strip", "string", "derived", "per reader row, one mark per layer: # read at or "
     "above min_mass, . below, - no value cell there, ? not computed"),
    ("lifetime.first", "string", "derived", "first t<j>.L<l> read at or above min_mass"),
    ("lifetime.last", "string", "derived", "last t<j>.L<l> read at or above min_mass"),
    ("mass", "claim", "derived-exact (supplied under a p/s edit)",
     "attention weight its reads receive, summed over heads and layers"),
    ("count", "integer", "structural", "number of (row, layer, head) reads"),
    ("mean_mass", "claim", "derived-exact", "mass per read"),
    ("readers", "array", "mass exact or measured; effect derived-exact P19",
     "top reads {name t<j>.L<l>.h<h>, mass, effect, contrast, foreign}"),
    ("status", "string", "derived", "edited, read, live, unread (unresolved: pending symbol)"),
    ("token", "integer", "structural", "token rows: the token id"),
    ("text", "string", "structural", "token rows: the decoded text (with a tokenizer)"),
    ("position", "integer", "structural", "token rows: the row index"),
    ("is_sink", "boolean", "structural", "token rows: the row is a declared sink row"),
    ("token_class", "string", "structural (tokenizer)", "special, bos_or_t0, whitespace, "
     "punct, digit, word_initial or continuation"),
    ("pieces_of_word.index", "integer", "structural (tokenizer)",
     "index of the piece within its whitespace-delimited word"),
    ("pieces_of_word.count", "integer", "structural (tokenizer)", "pieces in that word"),
    ("first_subword_of", "array", "derived-exact (tokenizer)",
     "candidate sets in which this row's token is the first piece of a multi-piece word"),
    ("to_consumer", "claim", "derived-exact P19",
     "exact direct contribution to the scalar consumer (share form: sum p c.W_O v_s)"),
    ("contrast", "claim", "derived-exact P19",
     "sum over its reads of p (c.W_O v_s - c.W_O vbar_h): the effect against the head's own "
     "mean value (vbar_h = sum_s p v_s)"),
    ("mask_delta", "claim", "derived-exact P19, direct path only",
     "sum over its reads of -contrast/(1-p): each read masked alone with renormalisation, "
     "every other input held; not the native mask effect (FT1: within 2 bf16 spacings on "
     "6/32 items)"),
    ("share", "claim", "derived-exact", "to_consumer / consumer_total (flips sign with the "
     "margin)"),
    ("share_of_abs", "claim", "derived-exact", "to_consumer / |consumer_total|"),
    ("mass_nonsink", "claim", "derived-exact",
     "mass renormalised per head over non-sink sources (sink rows: report.sink, default t0)"),
    ("sink_ratio", "claim", "derived-exact", "t0 only: its value norm over the median row's, "
     "median over layers"),
    ("role", "string", "structural", "the circuit-map role of the slot that bound it"),
    ("slot", "string", "structural", "the circuit-map slot that bound it"),
    ("map_digest", "string", "structural", "sha256 of the map whose slot bound it"),
    ("labels.<name>", "claim or {value, authority}", "supplied, or derived (with from)",
     "item labels (family, depth, country) and derived labels (decoded_country = "
     "content.countries.best)"),
    ("causal.status", "string", "structural", "unknown (observed only) or intervened"),
    ("causal.verdict", "string", "measured", "effect, null or below_floor"),
    ("causal.delta", "claim", "measured", "native consumer change with the consumer row's reads "
     "of the binding removed (p = 0, no renormalisation) at every layer and head"),
    ("causal.floor", "claim", "measured", "|null arm - the run's stepped consumer|"),
    ("content.<set>.best", "integer", "derived-exact P19", "best token of the set for the "
     "binding's write"),
    ("content.<set>.score", "claim", "derived-exact P19", "its readout rho gamma.W_U[t].w"),
    ("content.<set>.margin", "claim", "derived-exact P19", "best minus runner-up"),
    ("content.<set>.truth", "integer", "supplied", "the declared true token"),
    ("content.<set>.truth_rank", "integer", "derived-exact", "1-based rank of the truth"),
    ("content.<set>.names_truth", "boolean", "derived-exact", "best == truth"),
    ("content.<set>.flags.first_subword", "boolean", "derived-exact (tokenizer)",
     "best is the first piece of a multi-piece word of the set"),
    ("content.<set>.flags.collision", "boolean", "derived-exact (tokenizer)",
     "best stands for more than one word of the set"),
    ("content.<set>.flags.kind", "array", "derived-exact (tokenizer)",
     "other declared sets that hold the best token"),
    ("content.<set>.lexical_null.names_truth", "boolean", "derived-exact P19",
     "the truth token's own unembedding read through the same basis names the truth"),
    ("content.<set>.co.<other>.projected.names_truth", "boolean",
     "derived-exact P19; projection supplied", "readout with the whole span of <other>'s "
     "gamma W_U rows removed from the write"),
    ("content.<set>.co.<other>.lexical_null.names_truth", "boolean", "derived-exact P19",
     "readout of the removed part alone"),
    ("content.<set>.co.<other>.counts", "boolean", "derived-exact",
     "projected names the truth and the removed part alone does not"),
    ("content.<set>.switched.names_truth", "boolean", "derived-exact",
     "names the word paired with the decoded word of the switch set"),
    ("content_lifetime.<set>", "string", "derived", "per layer: C names the overall best, c in "
     "top 3, . names another, - no value cell"),
    ("hop.native_delta", "claim", "measured", "native consumer change under a supplied "
     "replacement of the binding"),
    ("hop.direct_delta", "claim", "derived-exact P19", "its direct-path change"),
    ("hop.indirect_share", "claim", "measured", "1 - direct/native"),
    ("foreign.cells", "array", "measured (TL, nnsight) / unverified (pyvene)",
     "per cell: TransformerLens hook + index, nnsight path + index, pyvene spec"),
    ("components.<i>.components.<j>", "object", "derived-exact P19; SAE weights supplied",
     "feature bindings: activation, to_consumer, content, saelens ids, neuronpedia link"),
]
READ_COLUMNS = [
    ("binding", "string", "structural", "the binding holding the value cells read"),
    ("reader_row", "integer", "structural", "the reading row j"),
    ("layer", "integer", "structural", "layer l"),
    ("head", "integer", "structural", "query head h"),
    ("mass", "number", "derived-exact", "sum over the binding's sources of p[h, j, s]"),
    ("effect", "number", "derived-exact P19", "share-form contribution to the scalar consumer"),
    ("contrast", "number", "derived-exact P19", "contrast-form contribution"),
    ("mask_delta", "number", "derived-exact P19, direct path", "-contrast/(1-p) per source, "
     "summed"),
    ("sources", "integer", "structural", "value cells of the binding the head reads"),
    ("supplied", "boolean", "structural", "a p/s edit supplied a weight"),
]
HEAD_COLUMNS = [
    ("reader_row", "integer", "structural", "row j"), ("layer", "integer", "structural", "l"),
    ("head", "integer", "structural", "h"),
    ("total_mass", "number", "measured", "sum_s p[h, j, s]"),
    ("unbound_mass", "number", "derived-exact", "mass of bindings not on this page"),
]


def _pd():
    import pandas as pd

    return pd


def _is_claim(node: Any) -> bool:
    return (isinstance(node, dict) and "value" in node and "authority" in node
            and isinstance(node["value"], numbers.Real) and not isinstance(node["value"], bool))


def flatten(node: Any, prefix: str = "", out: dict[str, Any] | None = None,
            skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Dotted columns of a nested mapping: claims -> value + ``__auth``; lists -> JSON text."""
    out = {} if out is None else out
    if _is_claim(node):
        out[prefix] = float(node["value"]) if not isinstance(node["value"], bool) else \
            node["value"]
        out[prefix + AUTH] = str(node["authority"])
        return out
    if isinstance(node, dict):
        for key, item in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if path in skip:
                continue
            flatten(item, path, out, skip)
        return out
    if isinstance(node, (list, tuple)):
        out[prefix] = json.dumps(node, default=str)
        return out
    out[prefix] = node
    return out


def _context(table: Any, item: str | None) -> dict[str, Any]:
    report = table.report
    record = table.records[0] if getattr(table, "records", None) else None
    return {"item": item, "where": report.get("where"), "consumer": report.get("consumer"),
            "model_identity": (json.dumps(record["model_identity"], sort_keys=True, default=str)
                               if record else None),
            "record_id": record["id"] if record else None}


def to_frame(table: Any, level: str = "binding", *, item: str | None = None,
             foreign: bool = False) -> Any:
    """A long DataFrame of ``table`` at ``level`` (module docstring)."""
    pd = _pd()
    if level not in ("binding", "read", "head"):
        raise ValueError("level is binding, read or head")
    ctx = _context(table, item)
    if level == "binding":
        rows = []
        for view in table.report["bindings"]:
            row = dict(ctx)
            row.update(flatten(view, skip=("readers",)))
            rows.append(row)
        return pd.DataFrame(rows)
    t = table.tensors
    names = t["bindings"]
    objects = t.get("binding_objects") or []
    roles = {b.name: (b.role, b.slot) for b in objects}
    rows = []
    if level == "head":
        for (j, layer), total in sorted(t["total_mass"].items()):
            unbound = t["unbound_mass"][(j, layer)]
            for h in range(int(total.numel())):
                rows.append({**ctx, "reader_row": j, "layer": layer, "head": h,
                             "total_mass": float(total[h]), "unbound_mass": float(unbound[h])})
        return pd.DataFrame(rows)
    listed = set(t["listed"])
    for (j, layer), mass in sorted(t["mass"].items()):
        cover = t["cover"][(j, layer)]
        effect = t["effect"][(j, layer)]
        contrast = t["contrast"].get((j, layer))
        mask = t["mask_delta"].get((j, layer))
        for b in sorted(listed):
            for h in cover[b].nonzero().reshape(-1).tolist():
                row = {**ctx, "binding": names[b], "role": roles.get(names[b], (None, None))[0],
                       "slot": roles.get(names[b], (None, None))[1], "reader_row": j,
                       "layer": layer, "head": h, "mass": float(mass[b, h]),
                       "sources": int(t["sources"][(j, layer)][b, h]),
                       "supplied": bool(t["supplied"][(j, layer)][b, h])}
                e = effect[b, h]
                row["effect"] = float(e) if e.dim() == 0 else float(e.norm())
                if contrast is not None:
                    row["contrast"] = float(contrast[b, h])
                    row["mask_delta"] = float(mask[b, h])
                if foreign:
                    from mdb.interop.names import read_names

                    names_ = read_names(j, layer, h)
                    row["tl_hook"], row["tl_index"] = names_["tl"]["hook"], names_["tl"]["index"]
                    row["nn_path"] = names_["nnsight"]["path"]
                    row["nn_index"] = names_["nnsight"]["index"]
                rows.append(row)
    return pd.DataFrame(rows)


def to_arrow(frame: Any, *, schema: str = "mdb-symbol-table-v1") -> Any:
    """``frame`` as a ``pyarrow.Table`` with ``mdb_schema`` and ``record_id`` metadata."""
    import pyarrow as pa

    out = pa.Table.from_pandas(frame, preserve_index=False)
    ids = sorted({str(x) for x in frame.get("record_id", []) if x is not None}) \
        if "record_id" in frame else []
    meta = dict(out.schema.metadata or {})
    meta[b"mdb_schema"] = schema.encode()
    meta[b"record_ids"] = json.dumps(ids).encode()
    return out.replace_schema_metadata(meta)


def to_parquet(frame: Any, path: str, *, schema: str = "mdb-symbol-table-v1") -> str:
    import pyarrow.parquet as pq

    pq.write_table(to_arrow(frame, schema=schema), path)
    return path


def from_frame(frame: Any) -> list[dict[str, Any]]:
    """Binding rows rebuilt from a binding-level frame (read-only): dotted columns nested
    again, ``<col>`` + ``<col>__auth`` -> ``{value, authority}``, JSON text left as text."""
    out = []
    columns = [c for c in frame.columns if not c.endswith(AUTH)]
    for _, series in frame.iterrows():
        row: dict[str, Any] = {}
        for col in columns:
            value = series[col]
            if value is None or (isinstance(value, float) and value != value):
                continue
            auth = series.get(col + AUTH) if col + AUTH in frame.columns else None
            if hasattr(value, "item"):
                value = value.item()
            leaf = {"value": value, "authority": auth} if isinstance(auth, str) else value
            node = row
            parts = col.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    break
            else:
                node[parts[-1]] = leaf
        out.append(row)
    return out


def trace_frame(trace: Any, table: str | None = None) -> Any:
    """One row per step: scalars as columns, each table entry as ``<table>.<key>``; with
    ``table``: long form ``step, key, value`` of that table only."""
    pd = _pd()
    if table is not None:
        return pd.DataFrame([{"step": r.get("step"), "key": k, "value": v}
                             for r in trace.records for k, v in (r.get(table) or {}).items()])
    rows = []
    for r in trace.records:
        row = {}
        for key, value in r.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    row[f"{key}.{k}"] = v
            elif isinstance(value, (list, tuple)):
                row[key] = json.dumps(value)
            else:
                row[key] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["meta"] = dict(trace.meta)
    return out


def map_frame(reports: list[dict[str, Any]]) -> Any:
    """Per-item binding rows of circuit-map item reports (``circuit_map.item_report``): one row
    per (item, binding) with ``item`` and the item's labels as columns."""
    pd = _pd()
    rows = []
    for rep in reports:
        table = rep.get("table") or {}
        bindings = table.get("bindings") or []
        if isinstance(bindings, dict):
            bindings = list(bindings.values())
        base = {"item": rep.get("item")}
        for key, value in (rep.get("labels") or {}).items():
            base[f"item_labels.{key}"] = _label_value(value)
        for view in bindings:
            row = dict(base)
            row.update(flatten(view, skip=("readers",)))
            rows.append(row)
    return pd.DataFrame(rows)


def _label_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("label", value.get("value"))
    return value


def loo_position_mean(frame: Any, value: str, by: str = "position") -> Any:
    """``value`` minus the leave-one-out mean of ``value`` over the other rows with the same
    ``by`` (a position or role); NaN where no other row shares it (TP1: identical prefixes put
    up to 32 copies of a row in its own position mean, so the mean must exclude the row)."""
    groups = frame.groupby(by)[value]
    total = groups.transform("sum")
    count = groups.transform("count")
    others = (total - frame[value]) / (count - 1)
    return (frame[value] - others).where(count > 1)


def token_constancy(frame: Any, by: str = "position", token: str = "token") -> Any:
    """Per ``by`` value: whether every item has the same ``token`` there (a template row), the
    number of items and of distinct tokens (derived across items; review 2.6)."""
    groups = frame.dropna(subset=[token]).groupby(by)[token]
    out = groups.agg(n="count", distinct="nunique").reset_index()
    out["constant"] = out["distinct"] == 1
    return out


_CLAIM_SCHEMA = {"type": "object", "required": ["value", "authority"],
                 "properties": {"value": {"type": "number"}, "authority": {"type": "string"},
                                "proof": {"type": "string"}}}


def _prop(kind: str) -> dict[str, Any]:
    if kind == "claim":
        return {"$ref": "#/$defs/claim"}
    if kind in ("string", "integer", "boolean", "array", "object", "number"):
        return {"type": kind}
    return {}


def json_schema(name: str = "mdb-symbol-table-v1") -> dict[str, Any]:
    """A JSON Schema (draft 2020-12) for a record body of ``name`` (``SCHEMAS``), generated
    from the record kind's required fields and ``COLUMNS``."""
    from mdb.core.kinds import CIRCUIT_MAP, SYMBOL_TABLE, TRACE

    if name not in SCHEMAS:
        raise ValueError(f"schema is one of {SCHEMAS}")
    base = {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://mdb.local/schemas/{name}.json", "title": name, "type": "object",
            "$defs": {"claim": _CLAIM_SCHEMA}}
    if name == "mdb-symbol-table-v1":
        props: dict[str, Any] = {}
        for col, kind, auth, meaning in COLUMNS:
            if "<" in col or "." in col:
                continue
            props[col] = {**_prop(kind), "description": f"{meaning} [{auth}]"}
        binding = {"type": "object", "required": ["symbol", "kind", "mass", "status"],
                   "properties": props, "additionalProperties": True}
        base["$defs"]["binding"] = binding
        base["required"] = list(SYMBOL_TABLE.required)
        base["properties"] = {"schema": {"const": name},
                              "bindings": {"type": "array", "items": {"$ref": "#/$defs/binding"}},
                              "page": {"type": "object"}, "unbound": {"type": "object"},
                              "accounting": {"type": "object"}, "labels": {"type": "object"},
                              "candidate_sets": {"type": "array"},
                              "components": {"type": "array"}}
    elif name == "mdb-circuit-map-v1":
        base["required"] = ["schema", "slots", "edges", "consumer_rule", "candidate_sets"]
        base["properties"] = {
            "schema": {"const": name},
            "slots": {"type": "array", "items": {"type": "object", "required": [
                "name", "role", "rows_rule"]}},
            "edges": {"type": "array", "items": {"type": "object", "required": [
                "src", "dst", "kind"]}},
            "consumer_rule": {"type": "object"},
            "candidate_sets": {"type": "object", "additionalProperties": {
                "type": "array", "items": {"type": "integer"}}},
            "constants": {"type": "object"}}
        base["x-record-kind"] = {"kind": "circuit-map", "required": list(CIRCUIT_MAP.required)}
    else:
        base["required"] = ["kind", "meta", "records"]
        base["properties"] = {"kind": {"type": "string"}, "meta": {"type": "object"},
                              "records": {"type": "array", "items": {"type": "object"}}}
        base["x-record-kind"] = {"kind": "trace", "required": list(TRACE.required)}
    return base


def schema_markdown() -> str:
    """The column reference as Markdown tables (``docs/SYMBOL-TABLE-SCHEMA.md``)."""
    def table(rows):
        out = ["| Column | Type | Authority | Meaning |", "|---|---|---|---|"]
        out += [f"| `{c}` | {k} | {a} | {m} |" for c, k, a, m in rows]
        return "\n".join(out)
    return ("## Binding rows (`report.bindings[*]`, frame level `binding`)\n\n" + table(COLUMNS)
            + "\n\n## Reads (frame level `read`)\n\n" + table(READ_COLUMNS)
            + "\n\n## Heads (frame level `head`)\n\n" + table(HEAD_COLUMNS) + "\n")


__all__ = ["COLUMNS", "HEAD_COLUMNS", "READ_COLUMNS", "SCHEMAS", "flatten", "from_frame",
           "json_schema", "loo_position_mean", "token_constancy", "map_frame", "schema_markdown", "to_arrow",
           "to_frame", "to_parquet", "trace_frame"]

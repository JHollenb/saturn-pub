"""Sealed traces, watch hits and circuit maps (symbol-table additions review 2026-11-14, 2.9).

A ``rows.Trace`` was a plain dataclass (issue C3) and a frozen ``circuit_map.Map`` a JSON
file, so "sealed before scoring" rested on file times. Here each becomes a record:

* ``seal_trace(trace, ...)``: kind ``trace``. The per-step records are one sha256-checked
  JSON blob (``core.bounded.put_json``; with no project store the bytes are held in memory on
  the ``holder`` object); the record keeps the trace kind, the step count, the record columns,
  a bounded ``meta`` (floats become ``measured`` claims) and the payload sha256.
* ``seal_watch_hits(result, trace_record, ...)``: kind ``watch-hits``, a
  ``breakpoints.replay`` result bound to the sealed trace it read (its record id), with the
  per-step truth values in a blob.
* ``seal_map(m, ...)``: kind ``circuit-map``: the map's exact JSON is the blob (so
  ``Map.digest`` re-computes from it), the record holds the digest, slots, edges and candidate
  set sizes, and optionally a ``MapTable`` blob.

``load_trace`` / ``load_map`` read a record back, checking the blob against the sha256 (and the
map against its digest). Values keep their labels; the record is the authority.
"""

from __future__ import annotations

import numbers
from typing import Any

from mdb.core.bounded import load_json, put_json
from mdb.core.kinds import CIRCUIT_MAP, TRACE, WATCH_HITS  # noqa: F401 (declares the kinds)
from mdb.core.records import seal_record

from .finite import EXACT, MEASURED, num

TRACE_SCHEMA = "mdb-trace-v1"
WATCH_SCHEMA = "mdb-watch-hits-v1"
MAP_RECORD_SCHEMA = "mdb-circuit-map-record-v1"
_CAP = 64


def _safe(value: Any, authority: str = MEASURED) -> Any:
    """``value`` with every float made a claim and long lists cut to 64 items (record-safe)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value) if value >= 0 else num(float(value), authority)
    if isinstance(value, numbers.Real):
        return num(float(value), authority)
    if isinstance(value, dict):
        return {str(k): _safe(v, authority) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_safe(v, authority) for v in list(value)[:_CAP]]
        return items
    return str(value)


def _store_of(project: Any) -> Any:
    return None if project is None else getattr(project, "store", project)


def _identity(run: Any = None, model_identity: Any = None, execution: Any = None):
    if run is not None:
        return run.adapter.model_identity, dict(run.adapter.execution)
    if model_identity is None:
        raise ValueError("give run= or model_identity= (and execution=)")
    return model_identity, dict(execution or {})


def seal_trace(trace: Any, *, session_id: str, run: Any = None, model_identity: Any = None,
               execution: Any = None, store: Any = None, holder: Any = None) -> dict[str, Any]:
    """A ``trace`` record of ``trace`` (``rows.Trace``) with its records as a blob."""
    identity, exe = _identity(run, model_identity, execution)
    payload = put_json({"kind": trace.kind, "meta": trace.meta, "records": trace.records},
                       store=_store_of(store), run=holder if holder is not None else run)
    columns = sorted({k for r in trace.records for k in r})
    body = {"schema": TRACE_SCHEMA, "trace_kind": trace.kind, "count": len(trace.records),
            "meta": _safe(trace.meta), "columns": {"names": columns[:_CAP],
                                                   "count": len(columns)},
            "payload": payload, "sha256": payload["sha256"]}
    return seal_record(session_id=session_id, kind="trace", model_identity=identity,
                       execution=exe, authority=MEASURED, body=body)


def load_trace(record: dict[str, Any], *, store: Any = None, holder: Any = None) -> Any:
    """The ``rows.Trace`` of a sealed ``trace`` record (blob checked against its sha256)."""
    from .rows import Trace

    if record.get("kind") != "trace":
        raise ValueError(f"not a trace record (kind {record.get('kind')!r})")
    data = load_json(record["body"]["payload"], store=_store_of(store), run=holder)
    return Trace(data["kind"], data["records"], data["meta"])


def seal_watch_hits(result: dict[str, Any], trace_record: dict[str, Any], *, session_id: str,
                    store: Any = None, holder: Any = None) -> dict[str, Any]:
    """A ``watch-hits`` record of a ``breakpoints.replay`` result over a sealed trace."""
    payload = put_json({"holds": result["holds"], "hits": result["hits"],
                        "undefined": result["undefined"]}, store=_store_of(store), run=holder)
    hits = [h for h in result["hits"] if isinstance(h, int) and h >= 0]
    body = {"schema": WATCH_SCHEMA, "trace": trace_record["id"],
            "predicate": result["predicate"], "mode": result["mode"], "n": int(result["n"]),
            "hits": hits[:_CAP], "hits_omitted": max(0, len(hits) - _CAP),
            "fired_at": result["fired_at"] if isinstance(result["fired_at"], int) else None,
            "fired_index": result["fired_index"],
            "crossings": int(result["crossings"]),
            "undefined": [u for u in result["undefined"] if isinstance(u, int)][:_CAP],
            "payload": payload}
    if result.get("fired_fraction") is not None:
        body["fired_fraction"] = num(float(result["fired_fraction"]), EXACT)
    return seal_record(session_id=session_id, kind="watch-hits",
                       model_identity=trace_record["model_identity"],
                       execution=trace_record["execution"], authority=EXACT, body=body)


def seal_map(m: Any, *, session_id: str, run: Any = None, model_identity: Any = None,
             execution: Any = None, store: Any = None, holder: Any = None,
             table: Any = None) -> dict[str, Any]:
    """A ``circuit-map`` record of ``m`` (``circuit_map.Map``); ``table``: a ``MapTable`` (its
    rows go in a second blob)."""
    identity, exe = _identity(run, model_identity, execution)
    data = m.to_json()
    payload = put_json(data, store=_store_of(store), run=holder if holder is not None else run)
    body: dict[str, Any] = {
        "schema": MAP_RECORD_SCHEMA, "map_schema": data["schema"], "digest": m.digest(),
        "slots": [{k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()}
                  for s in data["slots"]][:_CAP],
        "edges": data["edges"][:_CAP],
        "candidate_sets": [{"name": k, "size": len(v)} for k, v in data["candidate_sets"].items()],
        "payload": payload}
    if "n_items" in data["constants"]:
        body["n_items"] = int(data["constants"]["n_items"])
    if table is not None:
        body["table"] = put_json(table.to_json(), store=_store_of(store),
                                 run=holder if holder is not None else run)
    return seal_record(session_id=session_id, kind="circuit-map", model_identity=identity,
                       execution=exe, authority=EXACT, body=body)


def load_map(record: dict[str, Any], *, store: Any = None, holder: Any = None) -> Any:
    """The ``circuit_map.Map`` of a sealed ``circuit-map`` record; ``ValueError`` when the blob
    fails its sha256 or the map's digest differs from the sealed one."""
    from .circuit_map import Map

    if record.get("kind") != "circuit-map":
        raise ValueError(f"not a circuit-map record (kind {record.get('kind')!r})")
    m = Map.from_json(load_json(record["body"]["payload"], store=_store_of(store), run=holder))
    if m.digest() != record["body"]["digest"]:
        raise ValueError("the stored map's digest differs from the sealed digest")
    return m


__all__ = ["load_map", "load_trace", "seal_map", "seal_trace", "seal_watch_hits"]

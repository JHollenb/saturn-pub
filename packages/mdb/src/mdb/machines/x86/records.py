"""Sealed records of a binary's recovered symbol table and circuit map.

``mdb-symbol-table-v1`` is extended minimally for machine bindings (docs/SYMBOL-TABLE-SCHEMA.md,
"Machine bindings"): a binding lists its **cells** as ``[site, def, kill, bits]`` (site in the
address grammar, ``def = "init"`` for the initial state, tick interval ``[def, kill)``), its
**slots** as ``[pc, operand]`` pairs, ``lifetime`` in instructions and loop iterations, and
``causal`` (donors whose aligned value differed, and how many of those changed y). ``unbound``
counts the dead cells. ``mdb-circuit-map-v1`` carries one slot per binding (``rows_rule`` is
``pc-operand``) and typed edges ``data`` / ``addr`` / ``ctrl`` with lane counts.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from mdb.core.bounded import put_json
from mdb.core.records import num, seal_record

from .binary import site_name
from .method import SPEC, Recovery
from .trace import INIT

PAGE = 16
INLINE = 8   # nested lists share mdb's 64-item budget; a page stays well under 64 KiB


def identity(rec: Recovery) -> tuple[str, dict, dict]:
    b = rec.binary
    return (f"x86:{b.name}:{b.image_sha256[:12]}", {"kind": "x86-64-binary", "program": b.name,
                                                    "image_sha256": b.image_sha256},
            {"backend": "torchx86", "lanes": "batched"})


def _loop_prefix(tr) -> np.ndarray:
    b = tr.b
    nxt = [int(tr.pc[t + 1]) if t + 1 < tr.T else -1 for t in range(tr.T)]
    marks = np.array([int(b.head_row[b.row_of(p)] >= 0) if p >= 0 else 0 for p in nxt], np.int64)
    return np.concatenate([[0], np.cumsum(marks)])


def binding_rows(rec: Recovery, *, store=None) -> list[dict]:
    tr = rec.base
    lp = _loop_prefix(tr)
    out = []
    for k in range(rec.n_bindings):
        cells = rec.members(k)
        first = min(c.def_ for c in cells)
        last = max(c.kill for c in cells)
        full = [[site_name(c.site), "init" if c.def_ == INIT else c.def_, c.kill, c.bits] for c in cells]
        defs = [tr.describe(c) for c in cells]
        readers = sorted({f"{tr.ticks[t].pc:#x}#{tr.ticks[t].k}:{e}" for c in cells for t, e in c.readers})
        idx = [c.id for c in cells]
        row = {
            "symbol": f"b{k}", "kind": "web", "status": "live",
            "n_cells": len(cells), "weight": int(sum(c.weight for c in cells)),
            "cells": [f"{n}:{d}..{e}:{w}" for n, d, e, w in full[:INLINE]],
            "slots": [f"{p}.{o}" for p, o in rec.slots(k)][:INLINE],
            "definitions": defs[:INLINE],
            "lifetime": {"first": "init" if first == INIT else first, "last": last,
                         "instructions": last - max(first, 0),
                         "loop_iterations": int(lp[min(last, tr.T)] - lp[max(first, 0)])},
            "readers": readers[:INLINE],
            "causal": {"status": "intervened", "verdict": "effect",
                       "donors": num(int(rec.informative[idx].sum()), "measured"),
                       "effect_donors": num(int(rec.effect_donors[idx].sum()), "measured")},
        }
        if len(full) > INLINE or len(readers) > INLINE or len(rec.slots(k)) > INLINE:
            row["cells_full"] = put_json({"cells": full, "definitions": defs, "readers": readers,
                                          "slots": [list(s) for s in rec.slots(k)]}, store=store)
        out.append(row)
    return out


def symbol_table_records(rec: Recovery, *, store=None, where: str | None = None) -> list[dict]:
    sid, ident, execu = identity(rec)
    rows = binding_rows(rec, store=store)
    tr = rec.base
    dead = [c for c, lv in zip(tr.cells, rec.live) if not lv]
    unbound = {"dead_cells": len(dead), "dead_weight": int(sum(c.weight for c in dead)),
               "uninformative_cells": int((rec.informative == 0).sum()), "cells": len(tr.cells)}
    out = []
    for off in range(0, max(len(rows), 1), PAGE):
        body = {"schema": "mdb-symbol-table-v1", "where": where or f"bin:{rec.binary.name}",
                "consumer": "y = bytes of the last write syscall + exit code",
                "bindings": rows[off:off + PAGE], "unbound": unbound,
                "page": {"offset": off, "limit": PAGE, "total": len(rows)},
                "machine": {"T": tr.T, "donors": len(rec.donors), "lanes": rec.lanes_run},
                "method": {"spec": SPEC, "cell": "site byte (flag bit) x tick interval [def, kill)"}}
        out.append(seal_record(session_id=sid, kind="symbol-table", model_identity=ident, execution=execu,
                               authority="measured", body=body))
    return out


def circuit_map(rec: Recovery) -> dict:
    slots = [{"name": f"b{k}", "role": "web", "rows_rule": "pc-operand", "keys": [list(s) for s in rec.slots(k)]}
             for k in range(rec.n_bindings)]
    slots.append({"name": "y", "role": "consumer", "rows_rule": "consumer", "keys": []})
    edges = [{"src": f"b{a}", "dst": "y" if b_ == -2 else f"b{b_}",
              "kind": max(("ctrl", "addr", "data"), key=lambda e: (cnt[e] > 0, ("data", "addr", "ctrl").index(e))),
              "counts": dict(cnt)} for (a, b_), cnt in sorted(rec.edges.items())]
    return {"schema": "mdb-circuit-map-v1", "slots": slots, "edges": edges,
            "consumer_rule": {"kind": "last-write-bytes+exit-code"}, "candidate_sets": {},
            "constants": {"edge_rule": "total effect of single-cell aligned-resample patches; kind = ctrl if "
                                       "the lane's pc path diverged before the target cell, addr if an effective "
                                       "address did, else data"}}


def circuit_map_record(rec: Recovery, *, store=None) -> dict:
    sid, ident, execu = identity(rec)
    m = circuit_map(rec)
    digest = hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    inline = [{**e, "counts": {k: num(v, "measured") for k, v in e["counts"].items()}} for e in m["edges"][:64]]
    body = {"schema": "mdb-circuit-map-v1", "digest": digest,
            "slots": [{k: v for k, v in s.items() if k != "keys"} for s in m["slots"][:64]],
            "edges": inline, "edges_listed": "the first 64 (all in payload)", "payload": put_json(m, store=store)}
    return seal_record(session_id=sid, kind="circuit-map", model_identity=ident, execution=execu,
                       authority="measured", body=body)


def table_text(rec: Recovery, limit: int = 40) -> str:
    """The recovered symbol table as text, one binding per line."""
    tr = rec.base
    lines = [f"symbol table  bin:{rec.binary.name}  T={tr.T}  cells={len(tr.cells)}  live={int(rec.live.sum())}  "
             f"bindings={rec.n_bindings}  donors={len(rec.donors)}  lanes={rec.lanes_run}",
             f"{'sym':>5} {'cells':>5} {'bits*ticks':>10} {'life':>11}  slots (pc.operand) / first definitions"]
    rows = binding_rows(rec)
    for row in rows[:limit]:
        life = f"{row['lifetime']['first']}..{row['lifetime']['last']}"
        slots = ", ".join(row["slots"][:3])
        more = f" (+{len(rec.slots(int(row['symbol'][1:]))) - 3} slots)" if len(rec.slots(int(row['symbol'][1:]))) > 3 else ""
        lines.append(f"{row['symbol']:>5} {row['n_cells']:>5} {row['weight']:>10} {life:>11}  {slots}{more}")
    if len(rows) > limit:
        lines.append(f"  ... {len(rows) - limit} more bindings")
    dead = int((~rec.live).sum())
    lines.append(f"unbound: {dead} dead cells ({int((rec.informative == 0).sum())} with no informative donor)")
    return "\n".join(lines)

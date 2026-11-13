"""Live views: the cursor frame and the clock of a live session, built from op results.

Both run on the live session's worker thread (``LiveSession.call``). Tables, ports and edits come
from read ops through ``mdb.ops.dispatch`` (``where``, ``list``, ``info``, ``checkpoints``) and
edit provenance from ``Run.read_info``; nothing here re-derives what an op or the engine reports.
"""
from __future__ import annotations

from typing import get_args

from ..schema import (
    Ev,
    LiveCheckpoint,
    LiveEdit,
    LiveFrame,
    LivePort,
    LiveRow,
    LiveState,
    LiveStop,
    LiveTimeline,
    LiveUpstream,
)

PAGE = 64      # the most one op page holds
MAX_ROWS = 256  # rows a live timeline lists (the latest)
LABELS = get_args(Ev)


def _read(ctx, name, **args):
    from mdb.ops import dispatch

    return dispatch(ctx, name, args)


def _limits(debugger) -> dict[str, int]:
    adapter = debugger.adapter
    return {"layers": int(adapter.layers), "max_positions": int(adapter.max_positions)}


def edit_of(run, addr: str) -> LiveEdit:
    """``Run.read_info(addr)`` as a ``LiveEdit``. An authority outside the record vocabulary is
    not relabelled: the edit is refused and says what the engine reported."""
    info = run.read_info(addr)
    authority, state, reason = info["authority"], info["state"], info["reason"]
    if authority is not None and authority not in LABELS:
        authority, state = None, "refused"
        reason = f"the engine reported authority {info['authority']!r}, not a record label"
    by = info["by"]
    return LiveEdit(state=state, authority=authority, edited=bool(info["edited"]),
                    kind=str(info["kind"]), by=None if by is None else str(by),
                    reason=reason if state != "resolved" else None,
                    upstream=[LiveUpstream(**{k: item[k] for k in
                                              ("row", "layer", "port", "op", "staged")})
                              for item in info["upstream_edits"][:PAGE]])


def checkpoints(session) -> list[LiveCheckpoint]:
    """In-memory checkpoints (the ``checkpoints`` op), then the project's durable ones that are
    not also in memory under the same name and cursor."""
    memory = _read(session.ctx, "checkpoints", limit=PAGE)["checkpoints"]
    out = [LiveCheckpoint(name=c["name"], where=c["where"], store="memory") for c in memory]
    seen = {(c.name, c.where) for c in out}
    if session.debugger.project is not None:
        for item in session.debugger.checkpoints()[-PAGE:]:
            if (item["name"], item["where"]) not in seen:
                out.append(LiveCheckpoint(name=item["name"], where=item["where"],
                                          store="durable"))
    return out


def frame(session) -> LiveFrame:
    """The live frame (see ``LiveFrame``)."""
    ctx, debugger = session.ctx, session.debugger
    common = dict(
        session=session.sid, family=debugger.family, seq=session.seq,
        live=LiveState(state="paused"), limits=_limits(debugger),
        breakpoints=_read(ctx, "info", subject="breakpoints", limit=PAGE)["items"],
        displays=_read(ctx, "info", subject="display", limit=PAGE)["displays"],
        checkpoints=checkpoints(session))
    if ctx.run is None:
        return LiveFrame(**common, where=None, stop=None, row=None, layer=None, ports=[],
                         tokens=[], edits=None)
    where = _read(ctx, "where")
    listing = _read(ctx, "list")
    edits = _read(ctx, "info", subject="edits", limit=PAGE)
    row, layer = listing["row"], listing["layer"]
    ports = []
    for item in listing["ports"]:
        addr = f"t{row}.L{layer}.{item['port']}"
        ports.append(LivePort(
            port=item["port"], addr=addr, sublayer=item["sublayer"], operator=item["operator"],
            proofs=list(item["proofs"]), editable=bool(item["editable"]),
            cursor=bool(item["cursor"]), focus=bool(item["focus"]),
            computed=bool(item["computed"]),
            edit=edit_of(ctx.run, addr) if item["computed"] else None))
    return LiveFrame(**common, where=where["where"], stop=where["stop"], row=row, layer=layer,
                     ports=ports, checks=listing.get("checks"), tokens=where["tokens"],
                     edits={k: v for k, v in edits.items() if k != "op"})


def timeline(session) -> tuple[LiveTimeline, int]:
    """The live clock and how many earlier rows it left out (``MAX_ROWS`` latest are listed)."""
    ctx, debugger = session.ctx, session.debugger
    run = ctx.run
    if run is None:
        return LiveTimeline(session=session.sid, where=None, stops=[], cursor=None, root=None,
                            rows=[], limits=_limits(debugger)), 0
    stops = list(run.stops())
    here_row, here_index = run.parse(run.where)
    root_row = len(run.tokens) - 1  # the run's root: its first stepped row starts at stop 0
    context = [int(t) for t in run.session.read("tokens")[0].tolist()]
    marks: dict[int, list[str]] = {}
    for record in run.committed_edits():
        marks.setdefault(int(record["row"]), []).append(f"L{record['layer']}.{record['port']}")
    pending = run.pending_layer
    for record in run.staged_info():
        marks.setdefault(here_row, []).append(f"L{pending}.{record['port']} (staged)")
    rows = []
    for row in range(here_row + 1):
        count = None if row < root_row else len(stops) if row < here_row else here_index + 1
        rows.append(LiveRow(row=row, token=context[row] if row < len(context) else None,
                            count=count, edits=marks.get(row, [])))
    left_out = max(0, len(rows) - MAX_ROWS)
    return LiveTimeline(
        session=session.sid, where=run.where, stops=stops,
        cursor=LiveStop(row=here_row, index=here_index, label=stops[here_index]),
        root=LiveStop(row=root_row, index=0, label=stops[0]), rows=rows[left_out:],
        limits=_limits(debugger)), left_out

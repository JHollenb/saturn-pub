"""Live views: the cursor frame, its lists page by page, and the clock of a live session.

All run on the live session's worker thread (``LiveSession.call``). Tables, ports and edits come
from read ops through ``mdb.ops.dispatch`` (``where``, ``list``, ``info``, ``checkpoints``),
edit provenance from ``Run.read_info`` and durable checkpoints from ``Debugger.checkpoints``;
nothing here re-derives what an op or the engine reports.

The frame always fits the response cap. Each list in it (breakpoints, displays, checkpoints,
staged and committed edits, and each port's upstream edits) shows its first items within a byte
budget and names what it left out in ``more`` (``Omitted``: the list, how many, and the
``GET .../live/items`` request that pages through the rest). An item larger than ``ITEM_BYTES``
is shown clipped (long strings shortened, ``clipped: <bytes>``) and served whole by the items
route when it fits a page. If the frame is still over budget (very long conditions, many
ports), every list budget is halved until it fits; the ports themselves are never dropped.
"""
from __future__ import annotations

import json
from typing import Any, get_args

from ..schema import (
    CAP_BYTES,
    Ev,
    LiveCheckpoint,
    LiveEdit,
    LiveFrame,
    LiveItems,
    LivePort,
    LiveRow,
    LiveState,
    LiveStop,
    LiveTimeline,
    LiveUpstream,
    Omitted,
)

PAGE = 64       # the most one op page holds
MAX_ROWS = 256  # rows a live timeline lists (the latest)
LABELS = get_args(Ev)
LISTS = ("breakpoints", "displays", "checkpoints", "staged", "committed", "upstream")
FRAME_BYTES = CAP_BYTES - 8192   # the frame's data; the rest is envelope and headroom
PAGE_BYTES = CAP_BYTES - 8192    # one items page
ITEM_BYTES = 2048                # an item larger than this is clipped in the frame
UPSTREAM_SHOWN = 4               # upstream edits listed per port (all are counted)
BUDGETS = {"breakpoints": 6144, "displays": 3072, "checkpoints": 4096, "staged": 6144,
           "committed": 10240}


def _read(ctx, name, **args):
    from mdb.ops import dispatch

    return dispatch(ctx, name, args)


def _size(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str))


def _shorten(value: Any) -> Any:
    if isinstance(value, str) and len(value) > 200:
        return value[:200] + "…"
    if isinstance(value, dict):
        return {k: _shorten(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shorten(v) for v in value[:8]]
    return value


def clip(item: Any, limit: int = ITEM_BYTES) -> Any:
    """``item`` unchanged when it fits ``limit`` bytes, else shortened (strings to 200
    characters, lists to 8 items) with ``clipped`` set to its full size in bytes."""
    size = _size(item)
    if size <= limit or not isinstance(item, dict):
        return item
    short = _shorten(item)
    short["clipped"] = size
    return short


def fit(items: list, budget: int, *, clip_items: bool = True) -> list:
    """The longest prefix of ``items`` whose JSON fits ``budget`` bytes (items clipped first)."""
    out, used = [], 0
    for item in items:
        item = clip(item) if clip_items else item
        size = _size(item) + 1
        if used + size > budget:
            break
        out.append(item)
        used += size
    return out


def _paged(ctx, subject: str, key: str) -> list:
    out, offset = [], 0
    while True:
        page = _read(ctx, "info", subject=subject, offset=offset, limit=PAGE)
        out += page[key]
        offset += PAGE
        if offset >= page["total"]:
            return out


def _edits(ctx) -> tuple[list, list]:
    """Every staged edit and every committed edit (flat, in commit order: each carries its
    ``row``, ``layer`` and ``transition``), from the ``info edits`` op."""
    staged, committed, offset = [], [], 0
    while True:
        page = _read(ctx, "info", subject="edits", offset=offset, limit=PAGE)
        staged += page["staged"]
        for group in page["committed"]:
            committed += [{**edit, "row": group["row"], "layer": group["layer"],
                           "transition": group["transition"]} for edit in group["edits"]]
        offset += PAGE
        if offset >= page["total"]:
            return staged, committed


def _limits(debugger) -> dict[str, int]:
    adapter = debugger.adapter
    return {"layers": int(adapter.layers), "max_positions": int(adapter.max_positions)}


def _upstream(info) -> list[dict]:
    return [{k: item[k] for k in ("row", "layer", "port", "op", "staged")}
            for item in info["upstream_edits"]]


def edit_of(run, addr: str, shown: int = UPSTREAM_SHOWN) -> LiveEdit:
    """``Run.read_info(addr)`` as a ``LiveEdit`` (the first ``shown`` upstream edits, all
    counted). An authority outside the record vocabulary is not relabelled: the edit is refused
    and says what the engine reported."""
    info = run.read_info(addr)
    authority, state, reason = info["authority"], info["state"], info["reason"]
    if authority is not None and authority not in LABELS:
        authority, state = None, "refused"
        reason = f"the engine reported authority {info['authority']!r}, not a record label"
    by = info["by"]
    upstream = _upstream(info)
    return LiveEdit(state=state, authority=authority, edited=bool(info["edited"]),
                    kind=str(info["kind"]), by=None if by is None else str(by),
                    reason=reason if state != "resolved" else None,
                    upstream=[LiveUpstream(**u) for u in upstream[:shown]],
                    upstream_total=len(upstream))


def checkpoints(session) -> list[LiveCheckpoint]:
    """One row per checkpoint name: a name is the key and its latest checkpoint wins. With a
    project the latest durable record of each name (``restart NAME`` replays exactly that one
    after a server restart; the in-memory copy, written after it, is the same position), else the
    in-memory checkpoint (``checkpoints`` op). Order: first use of each name."""
    memory = {}
    offset = 0
    while True:
        page = _read(session.ctx, "checkpoints", offset=offset, limit=PAGE)
        memory.update({c["name"]: c["where"] for c in page["checkpoints"]})
        offset += PAGE
        if offset >= page["total"]:
            break
    rows: dict[str, LiveCheckpoint] = {}
    if session.debugger.project is not None:
        for item in session.debugger.checkpoints():  # oldest first: the latest of a name wins
            rows[item["name"]] = LiveCheckpoint(name=item["name"], where=item["where"],
                                                store="durable")
    for name, where in memory.items():
        if name not in rows:
            rows[name] = LiveCheckpoint(name=name, where=where, store="memory")
    return list(rows.values())


def _items_route(session, name: str, offset: int, addr: str | None = None) -> str:
    tail = f"&addr={addr}" if addr else ""
    return f"GET /api/v1/s/{session.sid}/live/items?list={name}&offset={offset}{tail}"


def full_list(session, name: str, addr: str | None = None) -> list:
    """Every item of one frame list (see ``LISTS``); ``upstream`` needs ``addr``."""
    ctx = session.ctx
    if name == "breakpoints":
        return _paged(ctx, "breakpoints", "items")
    if name == "displays":
        return _paged(ctx, "display", "displays")
    if name == "checkpoints":
        return [c.model_dump(mode="json") for c in checkpoints(session)]
    if ctx.run is None:
        return []
    if name in ("staged", "committed"):
        staged, committed = _edits(ctx)
        return staged if name == "staged" else committed
    if name == "upstream":
        if not addr:
            raise ValueError("the upstream list names a computed address (addr=)")
        return _upstream(ctx.run.read_info(addr))
    raise ValueError(f"no live list {name!r}; lists: {', '.join(LISTS)}")


def items(session, name: str, offset: int = 0, addr: str | None = None) -> LiveItems:
    """One page of a frame list from ``offset``: as many whole items as fit ``PAGE_BYTES`` (at
    least one, clipped if even it alone is over the page)."""
    every = full_list(session, name, addr)
    rest = every[offset:]
    shown = fit(rest, PAGE_BYTES, clip_items=False)
    if not shown and rest:
        shown = [clip(rest[0], PAGE_BYTES)]
    end = offset + len(shown)
    return LiveItems(list=name, addr=addr, offset=offset, total=len(every), items=shown,
                     next=end if end < len(every) else None)


def frame(session) -> tuple[LiveFrame, list[Omitted]]:
    """The live frame (see ``LiveFrame``) and what it left out."""
    ctx, debugger = session.ctx, session.debugger
    lists = {"breakpoints": full_list(session, "breakpoints"),
             "displays": full_list(session, "displays"),
             "checkpoints": full_list(session, "checkpoints")}
    base = dict(session=session.sid, family=debugger.family, seq=session.seq,
                live=LiveState(state="paused"), limits=_limits(debugger))
    ports, upstream_totals, where, listing, edits_meta = [], {}, None, None, {}
    if ctx.run is not None:
        where = _read(ctx, "where")
        listing = _read(ctx, "list")
        lists["staged"], lists["committed"] = _edits(ctx)
        edits_meta = {"where": ctx.run.where}
        row, layer = listing["row"], listing["layer"]
        for item in listing["ports"]:
            addr = f"t{row}.L{layer}.{item['port']}"
            edit = edit_of(ctx.run, addr) if item["computed"] else None
            if edit is not None:
                upstream_totals[addr] = edit.upstream_total
            ports.append(LivePort(
                port=item["port"], addr=addr, sublayer=item["sublayer"],
                operator=item["operator"], proofs=list(item["proofs"]),
                editable=bool(item["editable"]), cursor=bool(item["cursor"]),
                focus=bool(item["focus"]), computed=bool(item["computed"]), edit=edit))
    scale = 1.0
    while True:
        # a checkpoint row is never clipped: its name is what `restart` is sent
        shown = {name: fit(values, int(BUDGETS[name] * scale),
                           clip_items=name != "checkpoints") for name, values in lists.items()}
        more = [Omitted(field=name, count=len(values) - len(shown[name]),
                        reason=_items_route(session, name, len(shown[name])))
                for name, values in lists.items() if len(shown[name]) < len(values)]
        keep = UPSTREAM_SHOWN if scale >= 0.5 else 0
        for port in ports:
            if port.edit is not None:
                port.edit.upstream = port.edit.upstream[:keep]
                if port.edit.upstream_total > keep:
                    more.append(Omitted(
                        field=f"upstream:{port.addr}", count=port.edit.upstream_total - keep,
                        reason=_items_route(session, "upstream", keep, port.addr)))
        if ctx.run is None:
            built = LiveFrame(**base, where=None, stop=None, row=None, layer=None, ports=[],
                              tokens=[], edits=None, breakpoints=shown["breakpoints"],
                              displays=shown["displays"], checkpoints=shown["checkpoints"],
                              more=more)
        else:
            edits = {**edits_meta, "staged": shown["staged"], "committed": shown["committed"],
                     "staged_total": len(lists["staged"]),
                     "committed_total": len(lists["committed"])}
            built = LiveFrame(**base, where=where["where"], stop=where["stop"],
                              row=listing["row"], layer=listing["layer"], ports=ports,
                              checks=listing.get("checks"), tokens=where["tokens"], edits=edits,
                              breakpoints=shown["breakpoints"], displays=shown["displays"],
                              checkpoints=[LiveCheckpoint(**c) for c in shown["checkpoints"]],
                              more=more)
        if _size(built.model_dump(mode="json")) <= FRAME_BYTES or scale < 1 / 64:
            return built, more
        scale /= 2


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

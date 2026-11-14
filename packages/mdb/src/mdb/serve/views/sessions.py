"""Hello, session list and per-session capabilities."""
from __future__ import annotations

from ... import __version__
from ...core import address as addresses
from ..schema import (
    CAP_BYTES,
    SCHEMAS,
    Capabilities,
    Hello,
    LensInfo,
    Num,
    PortInfo,
    SessionSummary,
    ViewAvail,
    num,
    unresolved,
)
from ..stored import as_dict, as_str
from .common import ViewError, short

LENSES = {
    "cuts": ("retained cuts", "cuts",
             "cuts retained at this boundary; 0 = the recording passed here and kept none"),
    "receipts": ("native transitions", "transitions",
                 "receipt transitions that started at this boundary"),
    "symbols": ("symbol registrations", "registrations",
                "registered symbol locations and reads; no registration is not absence"),
    "records": ("records", "records", "sealed records and findings naming a cut here"),
}


def ops_available():
    try:
        from ... import ops  # noqa: F401
    except ImportError as error:
        return ViewAvail(available=False, reason=f"ops registry unavailable ({error})")
    if not hasattr(ops, "dispatch"):
        return ViewAvail(available=False, reason="mdb.ops has no dispatch()")
    return ViewAvail(available=True)


def session_summary(model) -> SessionSummary:
    identity = model.identity
    name = identity if isinstance(identity, str) else (
        as_str(as_dict(identity).get("name")) or str(identity))
    from .frame import default_address

    try:
        home = default_address(model)
    except ViewError:
        home = None
    return SessionSummary(
        id=model.sid, family=as_str(model.execution.get("family")) or None,
        adapter=as_str(model.execution.get("adapter")) or None, granularity=model.granularity,
        model=short(str(name), 16), cuts=len(model.cuts), receipts=len(model.receipts),
        symbols=len(model.symbols()), records=len(model.records()) + len(model.findings()),
        selected=model.selected, home=home)


def session_row(recorded, sid) -> SessionSummary:
    """One session's summary; a session that cannot be read is a refused entry, never an error
    for the list."""
    from ..recorded import SessionUnreadable

    try:
        return session_summary(recorded.model(sid))
    except (SessionUnreadable, KeyError) as error:
        reason = error.reason if isinstance(error, SessionUnreadable) else "session not found"
        return SessionSummary(id=sid, family=None, adapter=None, granularity=None, model="",
                              cuts=0, receipts=0, symbols=0, records=0, selected="", home=None,
                              state="refused", reason=reason)


def lens_infos(model) -> list[LensInfo]:
    has = {"cuts": bool(model.cuts), "receipts": bool(model.receipts),
           "symbols": bool(model.symbols() or model.lifecycles()),
           "records": bool(model.records() or model.findings())}
    return [LensInfo(id=k, label=v[0], unit=v[1], hasData=has[k], note=v[2])
            for k, v in LENSES.items()]


def layer_count(model) -> Num:
    """The model's decoder layer count with its evidence class: ``measured`` when the cut surface's
    layer clock says it, ``derived-local`` (a lower bound) when it is only the highest recorded
    layer index, ``unresolved`` when nothing recorded names it."""
    if model.layers_status == "known":
        return num(model.n_layers, "measured", unit="layers", proof=model.layer_note)
    if model.layers_status == "inferred":
        return num(model.n_layers, "derived-local", unit="layers (at least)",
                   proof=model.layer_note)
    return unresolved(model.layer_note, unit="layers")


def capabilities(model) -> Capabilities:
    ports = []
    for name, slot in sorted(model.slots.items()):
        ports.append(PortInfo(name=name, role=slot.get("role"),
                              kind=(slot.get("schema") or {}).get("kind"),
                              writable=slot.get("writable"), recorded=True))
    for name in addresses.DEFAULT_PORTS + addresses.DEFAULT_GLOBALS:
        ports.append(PortInfo(name=name, role="port", kind="occurrence", writable=None,
                              recorded=False, addr=f"t*.L*.{name}" if name in
                              addresses.DEFAULT_PORTS else f"t*.{name}"))
    slices = model.slices()
    views = {
        "timeline": ViewAvail(available=bool(model.cuts)),
        "symbols": ViewAvail(available=bool(model.symbols()),
                             reason=None if model.symbols() else "no symbol bindings recorded"),
        "graph": ViewAvail(available=bool(slices) or bool(model.cuts),
                           reason=None if slices else "no slices recorded; session graph only"),
        "records": ViewAvail(available=bool(model.records() or model.findings()),
                             reason=None if model.records() or model.findings()
                             else "no sealed records or findings in this session"),
        "pipeline": ViewAvail(available=False, reason="needs the engine's port program"),
        "compare": ViewAvail(available=False, reason="no compare view yet; use the compare op"),
    }
    return Capabilities(
        session=model.sid, axes=model.axes(), layers=layer_count(model), ports=ports,
        lenses=lens_infos(model),
        views=views,
        reduce={"ops": ["stats", "topk", "slice", "histogram", "tile", "project"],
                "pointers": ["slot", "blob", "occurrence"],
                "limits": {"topk": 64, "slice": 4096, "histogram": 256, "tile": 64}},
        ops=ops_available())


def hello(recorded, *, read_only, web, live=None) -> Hello:
    """``live``: the live session id (``mdb serve --tiny|--model``), else None."""
    from ..recorded import SessionUnreadable

    sessions = recorded.sessions()
    models = []
    for sid in sessions:
        try:
            models.append(recorded.model(sid))
        except (SessionUnreadable, KeyError):
            continue  # listed as refused in /sessions; it offers no view here
    caps = {
        "timeline": ViewAvail(available=any(m.cuts for m in models)),
        "symbols": ViewAvail(available=any(m.symbols() for m in models)),
        "graph": ViewAvail(available=any(m.cuts for m in models)),
        "records": ViewAvail(available=any(m.records() or m.findings() for m in models)),
        "reduce": ViewAvail(available=True),
        "ops": ops_available(),
        "live": ViewAvail(available=True) if live else ViewAvail(
            available=False, reason="recorded project; no engine attached"),
    }
    return Hello(version=__version__, schemas=dict(SCHEMAS), cap=CAP_BYTES,
                 sessions=len(sessions), readOnly=read_only, capabilities=caps, web=web,
                 live=live)

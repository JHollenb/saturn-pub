"""FastAPI application over a recorded project: ``create_app(project_path, live=None)``.

With ``live`` (a ``serve.live.LiveSession``: ``mdb serve --tiny|--model``) the server also holds
a live debugger under that session's id: ``POST .../op`` dispatches through the op registry on
the session's worker thread, ``POST .../console`` runs a REPL command line, ``POST
.../interrupt`` delivers Ctrl-C to the running op, ``GET .../live`` and ``.../live/timeline``
are the live frame and clock (``GET .../live/items`` pages through a frame list the frame
shortened), ``POST .../reduce`` with an occurrence pointer reads the live run the way ``print``
does (``serve.live_reduce``; nothing is written to the project), and the events socket pushes
one ``op`` event per completed op. Without ``live`` every op whose effect is not ``read`` is
refused 403.

Messages never name the server's absolute project path: refusal details, op results and console
text show it as ``<project>`` (paths inside it stay relative to it).

Routes live under ``/api/v1``; every body is a canonical-JSON ``Envelope`` or a ``Refused``
(``{"refused": reason, "detail"?}``), at most 64 KiB (``CapMiddleware``). The built web app is
served from ``mdb/serve/static`` when present. ``create_app`` returns the FastAPI app wrapped in
``guard.SecurityLayer`` (Host/Origin guard, request body limit, security headers on every
response, no-traceback 500s) as the outermost ASGI layer.
"""
# No ``from __future__ import annotations`` here: FastAPI resolves route annotations at
# runtime, and ``WebSocket`` is imported inside ``create_app`` (FastAPI stays an optional extra).
import asyncio
import inspect
import time
from pathlib import Path

from .caps import CapMiddleware, bound_json, canonical, dump, envelope, refused, take
from .guard import BODY_LIMIT, CSP, SecurityLayer  # noqa: F401 - CSP re-exported
from .recorded import Recorded, SessionUnreadable
from .reduce import ReduceError, Reducer
from .schema import (
    CAP_BYTES,
    ConsoleRequest,
    ConsoleResult,
    Event,
    Interrupt,
    LiveState,
    Omitted,
    OpRequest,
    OpResult,
    ReduceRequest,
    ViewAvail,
)
from .views import frame as frame_view
from .views import graph as graph_view
from .views import live as live_view
from .views import records as records_view
from .views import sessions as sessions_view
from .views import symbols as symbols_view
from .views import timeline as timeline_view
from .views.common import ViewError

STATIC = Path(__file__).resolve().parent / "static"
MAX_BRANCH_NODES = 256
WRITE_EFFECTS = ("navigate", "mutate", "native")


def create_app(project_path, *, read_only=True, static_dir=None, heartbeat=10.0, cap=CAP_BYTES,
               port=None, allow_hosts=(), body_limit=BODY_LIMIT, live=None):
    """The server as an ASGI app. ``port`` is the bound port (the Host guard accepts the loopback
    names with it; None accepts any port on them, for embedding and tests); ``allow_hosts`` are
    extra ``HOST[:PORT]`` values to accept (see ``guard``). ``live``: a ``LiveSession`` whose
    debugger is attached with this project (live mode; ``read_only`` is then False)."""
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import FileResponse, Response
    from starlette.concurrency import run_in_threadpool
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from .live import Busy, Interrupted

    if live is not None:
        read_only = False
    recorded = Recorded(project_path)
    reducer = Reducer()
    contexts = {}  # session id -> mdb.ops.Context (no debugger attached: recorded project)
    static = Path(static_dir) if static_dir else STATIC
    app = FastAPI(title="mdb serve", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.recorded, app.state.read_only = recorded, read_only
    app.state.reducer, app.state.live = reducer, live

    roots = sorted({str(Path(project_path)), str(Path(project_path).expanduser().resolve()),
                    *((str(live.debugger.project.root),) if live is not None
                      and live.debugger.project is not None else ())}, key=len, reverse=True)

    def scrub(value):
        """``value`` with the absolute project path written ``<project>``."""
        if isinstance(value, str):
            for root in roots:
                if root and root != "." and root in value:
                    value = value.replace(root, "<project>")
            return value
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v) for v in value]
        return value

    def respond(body, status=200):
        if isinstance(body, dict) and "refused" in body:
            body = scrub(body)
        return Response(canonical(body), status_code=status, media_type="application/json")

    def web_avail():
        if (static / "index.html").is_file():
            return ViewAvail(available=True)
        return ViewAvail(available=False, reason="web build absent (mdb/serve/static); run "
                         "`npm run build` in packages/mdb/web")

    def model(sid):
        try:
            return recorded.model(sid)
        except KeyError as error:
            raise ViewError("not_found", f"unknown session {sid!r}", 404) from error
        except SessionUnreadable as error:
            raise ViewError("session_unreadable", error.reason, 409) from error

    def is_live(sid):
        return live is not None and sid == live.sid

    def not_live(sid, what):
        if live is None:
            return ViewError("not_live", f"{what} needs a live session; this server holds none "
                             "(start it with --tiny or --model)", 403)
        return ViewError("not_live", f"{what}: session {sid!r} is recorded; the live session "
                         f"is {live.sid!r}", 403)

    async def on_worker(future):
        """Await a live job; a full queue is 409 ``busy``, an interrupted read 409."""
        try:
            return await asyncio.wrap_future(future)
        except Interrupted as error:
            raise ViewError("interrupted", "the request was interrupted before it finished",
                            409) from error

    def submit(make, *args):
        try:
            return make(*args)
        except Busy as error:
            raise ViewError("busy", f"the live session is busy: {error}; retry after the "
                            "running op finishes (or interrupt it)", 409) from error

    @app.exception_handler(ViewError)
    async def _view_error(request: Request, error: ViewError):
        return respond(refused(error.reason, error.detail), error.status)

    @app.exception_handler(ReduceError)
    async def _reduce_error(request: Request, error: ReduceError):
        return respond(refused(error.reason, error.detail), error.status)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, error: RequestValidationError):
        detail = "; ".join(f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg')}"
                           for e in error.errors()[:8])
        return respond(refused("invalid_request", detail), 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, error: StarletteHTTPException):
        reasons = {400: "bad_request", 404: "not_found", 405: "method_not_allowed",
                   413: "over_limit", 422: "invalid_request"}
        return respond(refused(reasons.get(error.status_code, "http_error"),
                               str(error.detail)[:300]), error.status_code)

    @app.exception_handler(Exception)
    async def _crash(request: Request, error: Exception):
        # No exception text, path or traceback reaches the client; the guard layer logs it.
        return respond(refused("internal_error", "the server failed on this request; see its "
                               "log"), 500)

    @app.get("/api/v1/hello")
    def hello():
        return respond(envelope("hello", sessions_view.hello(
            recorded, read_only=read_only, web=web_avail(),
            live=None if live is None else live.sid)))

    @app.get("/api/v1/sessions")
    def sessions():
        omitted = []
        rows = take([sessions_view.session_row(recorded, s) for s in recorded.sessions()], 64,
                    "sessions", omitted)
        return respond(envelope("sessions", rows, omitted))

    @app.get("/api/v1/s/{sid}/capabilities")
    def capabilities(sid: str):
        return respond(envelope("capabilities", sessions_view.capabilities(model(sid))))

    @app.get("/api/v1/s/{sid}/frame")
    def frame(sid: str, addr: str | None = None):
        block = frame_view.frame(model(sid), addr)
        if is_live(sid):
            # the recorded frame of the live session's project; the engine is attached
            state = "running" if live.running else "paused"
            block.live = LiveState(state=state, frontier=[
                f"live session: the engine is attached ({state}, cursor {live.where or 'no run'})"
                "; drive and read it from the Live view",
                *(f.replace("replay needs the engine", "the Live view reads it from the engine")
                  for f in block.live.frontier if f != frame_view.OFFLINE)])
        return respond(envelope("frame", block))

    @app.get("/api/v1/s/{sid}/timeline")
    def timeline(sid: str, lens: str = "cuts", x0: int | None = None, x1: int | None = None,
                 y0: int | None = None, y1: int | None = None, branch: str | None = None):
        block, omitted = timeline_view.timeline(model(sid), lens=lens, x0=x0, x1=x1, y0=y0,
                                                y1=y1, branch=branch)
        return respond(envelope("timeline", block, omitted))

    @app.get("/api/v1/s/{sid}/branches")
    def branches(sid: str):
        tree = frame_view.branches(model(sid))
        omitted = []
        tree.nodes = take(tree.nodes, MAX_BRANCH_NODES, "nodes", omitted,
                          reason="cuts beyond the first in index order")
        tree.branches = take(tree.branches, 64, "branches", omitted)
        return respond(envelope("branches", tree, omitted))

    @app.get("/api/v1/s/{sid}/symbols")
    def symbols(sid: str, at: str | None = None, q: str | None = None, page: int = 0):
        return respond(envelope("symbols", symbols_view.symbol_page(model(sid), at=at, q=q,
                                                                     page=page)))

    @app.get("/api/v1/s/{sid}/symbols/{identifier}")
    def symbol(sid: str, identifier: str):
        return respond(envelope("symbol", symbols_view.symbol_detail(model(sid), identifier)))

    @app.get("/api/v1/s/{sid}/graph")
    def graph(sid: str, addr: str | None = None, zoom: str = "slice", depth: int = 3,
              k: int = 48, slice: str | None = None):
        return respond(envelope("graph", graph_view.graph(model(sid), addr=addr, zoom=zoom,
                                                           depth=depth, k=k, slice_id=slice)))

    @app.get("/api/v1/s/{sid}/records")
    def records(sid: str, kind: str | None = None, page: int = 0):
        return respond(envelope("records", records_view.record_page(model(sid), kind=kind,
                                                                     page=page)))

    @app.get("/api/v1/s/{sid}/records/{identifier:path}")
    def record(sid: str, identifier: str):
        detail, omitted = records_view.record_detail(model(sid), identifier)
        marker = next((n.n for n in detail.summary.nums if n.name == "record"), None)
        if marker is not None and marker.ev in ("unresolved", "refused"):
            return respond(envelope("record", detail, omitted, state=marker.ev,
                                    reason=marker.reason))
        return respond(envelope("record", detail, omitted))

    @app.post("/api/v1/s/{sid}/reduce")
    async def reduce(sid: str, request: ReduceRequest):
        spec = dump(request.pointer)
        from .live_reduce import names_origin, reduce_live

        if is_live(sid) and spec["kind"] == "occurrence" and not names_origin(spec["address"]):

            result = await on_worker(submit(live.call, lambda: reduce_live(
                live, spec, request.op, request.params)))
            return respond(envelope("reduce", result))
        result = await run_in_threadpool(reducer.run, model(sid), spec, request.op,
                                         request.params)
        return respond(envelope("reduce", result))

    def recorded_op(sid, request):
        model(sid)
        try:
            from .. import ops
        except ImportError as error:
            return respond(refused("ops registry unavailable", str(error)), 501)
        dispatch = getattr(ops, "dispatch", None)
        if dispatch is None or not hasattr(ops, "Context") or not hasattr(ops, "registry"):
            return respond(refused("ops registry unavailable",
                                   "mdb.ops lacks dispatch/Context/registry"), 501)
        if list(inspect.signature(dispatch).parameters)[:2] != ["ctx", "name"]:
            return respond(refused("ops registry unavailable",
                                   "mdb.ops.dispatch signature is not (ctx, name, args)"), 501)
        found = ops.registry().get(request.op)
        if found is None:
            return respond(refused("unknown-op", f"no operation {request.op!r}"), 404)
        if found.effect in WRITE_EFFECTS:
            why = ("this server is read-only over a recorded project" if live is None else
                   f"session {sid!r} is recorded; the live session is {live.sid!r}")
            return respond(refused("read_only", f"{request.op} has effect {found.effect}; "
                                   f"{why}"), 403)
        context = contexts.setdefault(sid, ops.Context())
        try:
            result = dispatch(context, request.op, request.args)
        except ops.OpError as error:
            return respond(refused(error.kind, error.message), 409)
        except (AttributeError, TypeError, ValueError, LookupError) as error:
            return respond(refused("op_failed", f"{type(error).__name__}: {error}"), 409)
        omitted: list[Omitted] = []
        data = OpResult(request_id=request.request_id, op=request.op, seq=0, where=None,
                        result=scrub(bound_json(dump(result), field="result", omitted=omitted)))
        return respond(envelope("op", data, omitted))

    @app.post("/api/v1/s/{sid}/op")
    async def op(sid: str, request: OpRequest):
        if not is_live(sid):
            return await run_in_threadpool(recorded_op, sid, request)
        from .. import ops

        if request.op not in ops.registry():
            return respond(refused("unknown-op", f"no operation {request.op!r}"), 404)
        body = await on_worker(submit(live.op, request.op, request.args))
        result = scrub(body["result"])
        state = result.get("state", "resolved")
        state = state if state in ("resolved", "unresolved", "refused") else "refused"
        reason = None if state == "resolved" else (
            result.get("reason") or "a part of the result is unresolved; see data.result")
        data = OpResult(request_id=request.request_id, op=request.op, seq=body["seq"],
                        where=body["where"], result=result)
        return respond(envelope("op", data, state=state, reason=reason),
                       409 if state == "refused" else 200)

    @app.post("/api/v1/s/{sid}/console")
    async def console(sid: str, request: ConsoleRequest):
        if not is_live(sid):
            raise not_live(sid, "the console")
        body = await on_worker(submit(live.console, request.line))
        return respond(envelope("console", ConsoleResult(line=request.line, seq=body["seq"],
                                                         where=body["where"],
                                                         text=scrub(body["text"]))))

    @app.post("/api/v1/s/{sid}/interrupt")
    def interrupt(sid: str):
        if not is_live(sid):
            raise not_live(sid, "interrupt")
        return respond(envelope("interrupt", Interrupt(**live.interrupt())))

    @app.get("/api/v1/s/{sid}/live")
    async def live_frame(sid: str):
        if not is_live(sid):
            raise not_live(sid, "the live frame")
        frame_data, more = await on_worker(submit(live.call, lambda: live_view.frame(live)))
        return respond(envelope("live", frame_data, more or None))

    @app.get("/api/v1/s/{sid}/live/items")
    async def live_items(sid: str, list: str, offset: int = 0, addr: str | None = None):  # noqa: A002
        if not is_live(sid):
            raise not_live(sid, "live lists")
        if list not in live_view.LISTS:
            raise ViewError("bad_request", f"no live list {list!r}; lists: "
                            + ", ".join(live_view.LISTS), 400)
        if offset < 0:
            raise ViewError("bad_request", "offset is a non-negative integer", 400)
        if list == "upstream" and not addr:
            raise ViewError("bad_request", "the upstream list names a computed address (addr=)",
                            400)

        def page():
            try:
                return live_view.items(live, list, offset, addr)
            except (ValueError, KeyError) as error:
                raise ViewError("unresolved", f"{list}: {error}", 409) from error

        return respond(envelope("live-items", await on_worker(submit(live.call, page))))

    @app.get("/api/v1/s/{sid}/live/timeline")
    async def live_timeline(sid: str):
        if not is_live(sid):
            raise not_live(sid, "the live timeline")
        block, left_out = await on_worker(submit(live.call, lambda: live_view.timeline(live)))
        omitted = [Omitted(field="rows", count=left_out, reason="earliest rows")] \
            if left_out else None
        return respond(envelope("live-timeline", block, omitted))

    @app.get("/api/v1/s/{sid}/export")
    def export(sid: str):
        model(sid)
        from .export import export_walkthrough

        return respond(export_walkthrough(), 501)

    @app.websocket("/api/v1/s/{sid}/events")
    async def events(websocket: WebSocket, sid: str):
        await websocket.accept()
        seq = 0

        async def send(kind, data):
            nonlocal seq
            body = canonical(envelope("event", Event(type=kind, seq=seq, at=int(time.time() * 1000),
                                                     data=data)))
            seq += 1
            if len(body) > cap:
                body = canonical(refused("over_cap", bytes=len(body), cap=cap))
            await websocket.send_text(body.decode())

        if is_live(sid):
            events = live.subscribe()
            try:
                await send("hello", {"session": sid,
                                     "live": "running" if live.running else "paused",
                                     "seq": live.seq, "running": live.running})
                while True:
                    try:
                        event = await asyncio.wait_for(events.get(), timeout=heartbeat)
                    except asyncio.TimeoutError:
                        await send("heartbeat", {"session": sid})
                        continue
                    await send("op", event)
            except (WebSocketDisconnect, RuntimeError):
                return
            finally:
                live.unsubscribe(events)
        try:
            try:
                model(sid)
            except ViewError as error:
                await websocket.send_text(canonical(refused(error.reason, error.detail)).decode())
                await websocket.close()
                return
            await send("hello", {"session": sid, "live": "offline",
                                 "note": "recorded project; heartbeats only"})
            while True:
                await asyncio.sleep(heartbeat)
                await send("heartbeat", {"session": sid})
        except (WebSocketDisconnect, RuntimeError):
            return

    @app.get("/{path:path}", include_in_schema=False)
    def web(path: str):
        if path.startswith("api/"):
            return respond(refused("not_found", f"no API route /{path}"), 404)
        index = static / "index.html"
        if not index.is_file():
            return respond(refused("web_build_absent", web_avail().reason), 404)
        try:
            target = (static / path).resolve()
            inside = path and target.is_file() and static.resolve() in target.parents
        except (ValueError, OSError):  # e.g. a NUL byte in the path
            inside = False
        if inside:
            return FileResponse(target)
        if path.startswith("assets/") or "." in path.rsplit("/", 1)[-1]:
            # A missing file is a 404, never the SPA shell (a stale bundle must fail loudly).
            return respond(refused("not_found", f"no static file /{path}"), 404)
        return FileResponse(index)

    app.add_middleware(CapMiddleware, cap=cap)
    return SecurityLayer(app, port=port, allow_hosts=allow_hosts, body_limit=body_limit)

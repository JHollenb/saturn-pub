"""``mdb serve --tiny``: the live session (``mdb.serve.live``) over HTTP, on tiny models.

The gate: for a script that names every registered op, the ``result`` the server returns for
each command is, after canonical JSON, byte-equal to the line ``python -m mdb.repl --json``
prints for it (the REPL's own ``main``, in-process, attached with a project of its own so
durable checkpoints answer the same way). Then: serial execution and the busy rule, interrupt,
the Host/Origin guard on every new route and the socket, malformed bodies, live reductions,
durable checkpoints read back by a recorded-mode app, and the events socket.
"""
from __future__ import annotations

import asyncio
import gc
import json
import threading
import time
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient  # noqa: E402

from mdb.ops import Context, registry  # noqa: E402
from mdb.ops.debugger import attach  # noqa: E402
from mdb.repl.shell import Shell, parse  # noqa: E402
from mdb.serve import schema  # noqa: E402
from mdb.serve.app import create_app  # noqa: E402
from mdb.serve.caps import canonical  # noqa: E402
from mdb.serve.live import LiveSession, open_debugger  # noqa: E402
from mdb.serve.reduce import reduce_array  # noqa: E402

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
PROMPT = [1, 5, 9, 3]

# (command line, op, args): every registered op appears at least once. Lines that fail are
# part of the gate too: the REPL prints {error, state, reason} and so must the server.
SCRIPT = [
    ("start 1 5 9 3", "start", {"prompt": PROMPT}),
    ("where", "where", {}),
    ("info program", "info", {"subject": "program"}),
    ("list", "list", {}),
    ("step", "step", {}),
    ("step 2", "step", {"n": 2}),
    ("next", "next", {}),
    ("print", "print", {}),
    ("print q top=3", "print", {"address": "q", "top": 3}),
    ("x q 4", "x", {"address": "q", "count": 4}),
    ("frame L0.q", "frame", {"address": "L0.q"}),
    ("up", "up", {}),
    ("down", "down", {}),
    ("backtrace", "backtrace", {}),
    ("frame cursor", "frame", {"address": "cursor"}),
    ("display L0.h", "display", {"address": "L0.h"}),
    ("info display", "info", {"subject": "display"}),
    ("undisplay", "undisplay", {}),
    ("break L*.p", "break", {"pattern": "L*.p"}),
    ("tbreak L1.o", "tbreak", {"pattern": "L1.o"}),
    ("watch L1.h", "watch", {"address": "L1.h"}),
    ("condition 1 norm > 0", "condition", {"id": 1, "expr": "norm > 0"}),
    ("ignore 1 0", "ignore", {"id": 1, "count": 0}),
    ("disable 3", "disable", {"ids": [3]}),
    ("enable 3", "enable", {"ids": [3]}),
    ("info breakpoints", "info", {"subject": "breakpoints"}),
    ("info watch", "info", {"subject": "watch"}),
    ("continue", "continue", {}),
    ("continue", "continue", {}),
    ("reverse-continue", "reverse-continue", {}),
    ("delete", "delete", {}),
    ("until t4.L0.h", "until", {"target": "t4.L0.h"}),
    ("advance t4.L0.q", "advance", {"target": "t4.L0.q"}),
    ("checkpoint pre", "checkpoint", {"name": "pre"}),
    ("checkpoints", "checkpoints", {}),
    ("info checkpoints", "info", {"subject": "checkpoints"}),
    ("reverse-step", "reverse-step", {}),
    ("reverse-next", "reverse-next", {}),
    ("reverse-finish", "reverse-finish", {}),
    ("step 3", "step", {"n": 3}),
    ("set L0.v = 2 scale", "set", {"target": "L0.v", "value": 2.0, "op": "scale"}),
    ("info edits", "info", {"subject": "edits"}),
    ("abort", "abort", {}),
    ("whatif L0.v = 2 scale --consumer L0.o", "whatif",
     {"target": "L0.v", "value": 2.0, "op": "scale", "consumer": "L0.o"}),
    ("set L0.o = 0.5 add", "set", {"target": "L0.o", "value": 0.5, "op": "add"}),
    ("commit", "commit", {}),
    ("finish", "finish", {}),
    ("explain L1.o", "explain", {"address": "L1.o"}),
    ("forecast", "forecast", {}),
    ("restart pre", "restart", {"name": "pre"}),
    ("forecast L1.p", "forecast", {"target": "L1.p"}),
    ("until L1.A", "until", {"target": "L1.A"}),
    ("verify", "verify", {}),
    ("reads", "reads", {}),
    ("symbols", "symbols", {}),
    ("symbol x = L1.p", "symbol", {"name": "x", "addresses": "L1.p"}),
    ("circuit", "circuit", {}),
    ("compare pre L0.v=2 scale --consumer L0.o", "compare",
     {"a": "pre", "b": "L0.v=2 scale", "consumer": "L0.o"}),
    ("seek", "seek", {}),
    ("export", "export", {}),
    ("restart pre", "restart", {"name": "pre"}),
    ("start name=c prefill=2 1 7 9 3", "start", {"prompt": [1, 7, 9, 3], "name": "c",
                                                 "prefill": 2}),
    ("until t3.logits", "until", {"target": "t3.logits"}),
    ("frame main:", "frame", {"address": "main:"}),
    ("sweep h rows=3 = c --consumer t3.M[0,1]", "sweep",
     {"port": "h", "rows": "3", "origin": "c", "consumer": "t3.M[0,1]"}),
    ("where", "where", {}),
    ("promptset list", "promptset", {"action": "list"}),
    ("each recall: print", "each", {"set": "recall", "line": "print"}),
    ("run 1 5 9 3 budget=2", "run", {"prompt": PROMPT, "budget": 2}),
]
FAMILIES = ("qwen2", "qwen3")


def _client(app):
    return TestClient(app, base_url=BASE)


def _live(tmp_path, family="qwen2", **kw):
    debugger = open_debugger(tmp_path / f"project-{family}", tiny=family)
    return LiveSession(debugger, **kw)


@pytest.fixture()
def live(tmp_path):
    session = _live(tmp_path)
    yield session
    session.close()


def _op(c, sid, op, args=None, request_id="r"):
    return c.post(f"/api/v1/s/{sid}/op", json={"op": op, "args": args or {},
                                               "request_id": request_id})


# --------------------------------------------------------------------------- the gate


def _server_transcript(tmp_path, family):
    session = _live(tmp_path, family)
    try:
        c = _client(create_app(session.debugger.project.root, port=PORT, live=session))
        out = []
        for _, op, args in SCRIPT:
            response = _op(c, session.sid, op, args)
            body = response.json()
            assert body["schema"] == schema.SCHEMAS["op"], body
            assert response.status_code == (409 if body["state"] == "refused" else 200)
            out.append(body["data"]["result"])
        return out
    finally:
        session.close()


def _repl_transcript(tmp_path, family, capsys):
    """``python -m mdb.repl --tiny FAMILY --script F --json`` (its ``main``, in-process), with
    the tiny model attached under a project as the server's is. The script sources the command
    list with ``-k`` so a failing line does not stop it (``--script`` stops at an error)."""
    from mdb.engine.models import tiny_model
    from mdb.repl.__main__ import main

    inner = tmp_path / f"{family}-inner.mdb"
    inner.write_text("".join(line + "\n" for line, _, _ in SCRIPT))
    outer = tmp_path / f"{family}-outer.mdb"
    outer.write_text(f"source -k {inner}\n")
    project = tmp_path / f"repl-{family}"

    def open_tiny(name, dtype="bf16", *, seed=113, **_):  # project= is this test's own
        return attach(tiny_model(name, dtype, seed=seed), None, project=project)

    capsys.readouterr()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("mdb.ops.debugger.open_tiny", open_tiny)
        main(["--tiny", family, "--script", str(outer), "--json"])
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


@pytest.mark.parametrize("family", FAMILIES)
def test_every_op_returns_what_the_repl_prints(family, tmp_path, capsys):
    server = _server_transcript(tmp_path, family)
    repl = _repl_transcript(tmp_path, family, capsys)
    assert len(repl) == len(SCRIPT) == len(server)
    mismatched = [(line, canonical(a).decode()[:300], canonical(b).decode()[:300])
                  for (line, _, _), a, b in zip(SCRIPT, server, repl)
                  if canonical(a) != canonical(b)]
    assert not mismatched, mismatched[:3]
    # the script really moves: most lines are resolved op results, not errors
    assert sum("error" not in r for r in server) >= len(SCRIPT) - 14


@pytest.mark.parametrize("line,op,args", SCRIPT, ids=[f"{i:02d}-{s[0]}" for i, s in
                                                       enumerate(SCRIPT)])
def test_script_args_are_what_the_repl_parses(line, op, args):
    verb, _, rest = line.partition(" ")
    assert Shell(Context()).resolve(verb) == op
    assert parse(op, rest) == args


def test_the_script_names_every_registered_op():
    assert set(registry()) <= {op for _, op, _ in SCRIPT}


def test_the_console_prints_what_the_repl_prints(live, tmp_path):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    shell = Shell(open_debugger(tmp_path / "other", tiny="qwen2").ops(), out=None)
    import io

    for line in ("start 1 5 9 3", "step 3", "print", "set v = 2 scale", "info edits", "commit",
                 "where", "help step", "nosuchverb"):
        response = c.post(f"/api/v1/s/{live.sid}/console", json={"line": line})
        assert response.status_code == 200
        shell.out = io.StringIO()
        shell.execute(line)
        assert response.json()["data"]["text"] == shell.out.getvalue().rstrip("\n"), line


def test_the_console_refuses_file_commands(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    for line in ("source /etc/passwd", "sou /etc/passwd", "record /tmp/x", "quit"):
        text = c.post(f"/api/v1/s/{live.sid}/console", json={"line": line}).json()["data"]["text"]
        assert "not available from the web console" in text, line


def test_promptset_load_is_refused_on_the_console_and_the_op_route_but_inline_works(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    for line in ("promptset load s /etc/hosts", "set load s /etc/hosts"):
        text = c.post(f"/api/v1/s/{sid}/console", json={"line": line}).json()["data"]["text"]
        assert "send the set inline" in text and "No such file" not in text, line
    response = c.post(f"/api/v1/s/{sid}/op", json={
        "op": "promptset", "request_id": "r1",
        "args": {"action": "load", "name": "s", "arg": "/nonexistent/x.jsonl"}})
    body = response.json()
    assert response.status_code == 409 and "send the set inline" in json.dumps(body)
    assert "No such file" not in json.dumps(body)
    items = json.dumps([{"name": "a", "tokens": PROMPT}, {"name": "b", "tokens": [2, 6, 10]}])
    response = c.post(f"/api/v1/s/{sid}/op", json={
        "op": "promptset", "request_id": "r2",
        "args": {"action": "define", "name": "s", "arg": items}})
    assert response.status_code == 200 and response.json()["data"]["result"]["total"] == 2
    for args in ({"action": "show", "name": "s"}, {"action": "list"}):
        assert c.post(f"/api/v1/s/{sid}/op", json={"op": "promptset", "request_id": "r3",
                                                   "args": args}).status_code == 200
    each = c.post(f"/api/v1/s/{sid}/op", json={
        "op": "each", "request_id": "r4", "args": {"set": "s", "line": "print"}})
    assert each.status_code == 200, each.text


# --------------------------------------------------------------------------- one thread


def _async_client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)


def test_concurrent_ops_run_serially_in_a_deterministic_sequence(live):
    app = create_app(live.debugger.project.root, port=PORT, live=live)
    _op(_client(app), live.sid, "start", {"prompt": PROMPT})
    stops = list(live.ctx.run.stops())
    first = stops.index("L0.h")
    seen, overlap = [], []
    original = live.ctx.run.step

    def step(*args, **kwargs):  # records whether two steps ever overlap
        seen.append(threading.get_ident())
        if len(set(seen)) > 1:
            overlap.append(True)
        return original(*args, **kwargs)

    live.ctx.run.step = step

    async def go():
        async with _async_client(app) as c:
            return await asyncio.gather(*[
                c.post(f"/api/v1/s/{live.sid}/op", json={"op": "step", "args": {},
                                                         "request_id": f"r{i}"})
                for i in range(20)])

    responses = asyncio.run(go())
    bodies = sorted((r.json()["data"] for r in responses), key=lambda d: d["seq"])
    assert [d["seq"] for d in bodies] == list(range(2, 22))  # start was seq 1
    assert [d["where"] for d in bodies] == [f"t3.{stops[first + k]}" for k in range(1, 21)]
    assert all(d["result"]["where"] == d["where"] for d in bodies)
    assert not overlap and len(set(seen)) == 1 and seen[0] != threading.get_ident()


def test_a_full_queue_is_409_busy_and_runs_nothing(tmp_path):
    session = _live(tmp_path, queue_limit=2)
    try:
        app = create_app(session.debugger.project.root, port=PORT, live=session)
        gate = threading.Event()
        blocker = session.call(gate.wait)  # occupies the worker
        while session._current is None:  # picked up (a queued job would be cancelled below)
            time.sleep(0.001)

        async def go():
            async with _async_client(app) as c:
                tasks = [asyncio.create_task(c.post(
                    f"/api/v1/s/{session.sid}/op",
                    json={"op": "info", "args": {"subject": "breakpoints"},
                          "request_id": f"q{i}"})) for i in range(2)]
                await asyncio.sleep(0.2)
                third = await c.post(f"/api/v1/s/{session.sid}/op",
                                     json={"op": "start", "args": {"prompt": PROMPT},
                                           "request_id": "q3"})
                gate.set()
                return third, await asyncio.gather(*tasks)

        assert session.interrupt()["interrupted"] is False  # a read is never interrupted
        third, queued = asyncio.run(go())
        blocker.result(timeout=10)
        assert third.status_code == 409 and third.json()["refused"] == "busy"
        assert [r.status_code for r in queued] == [200, 200]
        assert session.ctx.run is None  # the refused start never ran
    finally:
        session.close()


def test_interrupt_stops_a_long_continue_and_leaves_a_consistent_cursor(live):
    app = create_app(live.debugger.project.root, port=PORT, live=live)
    c = _client(app)
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "break", {"pattern": "L*.p", "condition": "norm > 1e30"})  # never hits
    assert c.post(f"/api/v1/s/{live.sid}/interrupt").json()["data"]["interrupted"] is False

    async def go():
        async with _async_client(app) as ac:
            task = asyncio.create_task(ac.post(
                f"/api/v1/s/{live.sid}/op",
                json={"op": "continue", "args": {"budget": 4096}, "request_id": "long"}))
            for _ in range(200):
                await asyncio.sleep(0.02)
                if live.running == "continue":
                    break
            await asyncio.sleep(0.3)
            stop = await ac.post(f"/api/v1/s/{live.sid}/interrupt")
            return stop, await task

    started = time.monotonic()
    stop, long = asyncio.run(go())
    elapsed = time.monotonic() - started
    assert stop.json()["data"] == {"interrupted": True, "running": "continue", "cancelled": 0,
                                   "reason": None}
    body = long.json()
    result = body["data"]["result"]
    assert set(result) == {"interrupted", "state", "reason"}, result
    assert result["state"] == "unresolved" and body["state"] == "unresolved"
    where = result["interrupted"]
    assert elapsed < 4.0, elapsed  # the full continue to end of context takes ~6 s here
    # the cursor is consistent: where, print and a further step all work from it
    now = _op(c, live.sid, "where").json()["data"]["result"]
    assert now["where"] == where == live.ctx.run.where
    assert _op(c, live.sid, "print").json()["state"] == "resolved"
    assert _op(c, live.sid, "step").json()["state"] == "resolved"
    assert c.post(f"/api/v1/s/{live.sid}/interrupt").json()["data"]["interrupted"] is False


def test_interrupts_at_random_moments_leave_a_working_session(live):
    """Ctrl-C lands at a random point of a short continue (about 40 ms here) or after it: the
    worker survives, the cursor reads back, and the result says where it stopped."""
    import random

    pick = random.Random(1)
    live.op("start", {"prompt": PROMPT}).result(timeout=30)
    live.op("break", {"pattern": "L*.p", "condition": "norm > 1e30"}).result(timeout=30)
    interrupted = 0
    for _ in range(40):
        future = live.op("continue", {"budget": 2})
        while live._current is None and not future.done():
            pass
        time.sleep(pick.uniform(0, 0.06))
        live.interrupt()
        body = future.result(timeout=30)
        interrupted += "interrupted" in body["result"]
        assert body["where"] == live.ctx.run.where
        if "interrupted" in body["result"]:
            assert body["result"]["interrupted"] == body["where"]
        printed = live.op("print", {}).result(timeout=30)["result"]
        assert printed["state"] == "resolved" and printed["address"].startswith(body["where"])
    assert live._thread.is_alive() and interrupted >= 1


# --------------------------------------------------------------------------- guard


NEW_ROUTES = [("POST", "op", {"op": "where", "args": {}, "request_id": "g"}),
              ("POST", "console", {"line": "where"}), ("POST", "interrupt", None),
              ("GET", "live", None), ("GET", "live/timeline", None),
              ("GET", "live/items?list=breakpoints", None), ("GET", "live/ledger", None),
              ("POST", "reduce", {"pointer": {"kind": "occurrence", "address": "t3.L0.h"},
                                  "op": "stats"})]


def test_every_new_route_and_the_socket_refuse_a_foreign_host_or_origin(live):
    from starlette.websockets import WebSocketDisconnect

    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    for method, route, body in NEW_ROUTES:
        url = f"/api/v1/s/{live.sid}/{route}"
        for headers, reason in (({"host": "evil.example"}, "host_not_allowed"),
                                ({"host": f"evil.example:{PORT}"}, "host_not_allowed"),
                                ({"origin": "http://evil.example"}, "origin_not_allowed"),
                                ({"origin": f"https://127.0.0.1:{PORT}"}, "origin_not_allowed")):
            response = c.request(method, url, json=body, headers=headers)
            assert response.status_code == 403, (route, headers)
            assert response.json()["refused"] == reason
        good = c.request(method, url, json=body, headers={"origin": f"http://127.0.0.1:{PORT}"})
        assert good.status_code == 200, (route, good.text[:200])
    url = f"ws://127.0.0.1:{PORT}/api/v1/s/{live.sid}/events"
    for headers in ({"host": "evil.example"}, {"origin": "http://evil.example"}):
        with pytest.raises(WebSocketDisconnect) as closed:
            with c.websocket_connect(url, headers=headers):
                pass
        assert closed.value.code == 1008


def test_without_live_mode_every_writing_op_is_403(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "checkpoint", {"name": "kept"})
    recorded = _client(create_app(live.debugger.project.root, port=PORT))
    assert recorded.get("/api/v1/hello").json()["data"]["live"] is None
    writing = [name for name, op in registry().items() if op.effect != "read"]
    assert {"start", "step", "set", "continue", "restart"} <= set(writing)
    for name in writing:
        response = _op(recorded, live.sid, name)
        assert response.status_code == 403 and response.json()["refused"] == "read_only", name
    for route in ("console", "interrupt", "live", "live/timeline", "live/ledger"):
        method = "GET" if route.startswith("live") else "POST"
        response = recorded.request(method, f"/api/v1/s/{live.sid}/{route}",
                                    json={"line": "where"} if route == "console" else None)
        assert response.status_code == 403 and response.json()["refused"] == "not_live", route


def test_malformed_op_bodies_never_5xx(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    url = f"/api/v1/s/{live.sid}/op"
    deep = b'{"op":"step","request_id":"d","args":{"n":' + b"[" * 40000 + b"]" * 40000 + b"}}"
    bodies = [
        {"op": "no-such-op", "args": {}, "request_id": "a"},
        {"op": "step", "args": {"n": "three"}, "request_id": "b"},
        {"op": "step", "args": {"n": True}, "request_id": "b"},
        {"op": "step", "args": {"zz": 1}, "request_id": "b"},
        {"op": "step", "args": "not-an-object", "request_id": "c"},
        {"op": 7, "args": {}, "request_id": "c"},
        {"args": {}, "request_id": "c"},
        {"op": "set", "args": {"target": "v", "value": float("1e308") * 10}, "request_id": "e"},
        {"op": "set", "args": {"target": "__import__('os')", "value": 1}, "request_id": "e"},
        {"op": "start", "args": {"prompt": [10 ** 9]}, "request_id": "e"},
        {"op": "start", "args": {"prompt": "x" * 5000}, "request_id": "e"},
        [1, 2, 3],
    ]
    seen = []
    for body in bodies:
        text = json.dumps(body, allow_nan=True).replace("Infinity", "1e999")
        response = c.post(url, content=text, headers={"content-type": "application/json"})
        seen.append(response.status_code)
        assert response.status_code < 500, (body, response.text[:200])
    for raw in (deep, b"{" * 10, b"\xff\xfe", b'{"op": "where", "request_id": "x"' + b" " * 300000
                + b"}"):
        response = c.post(url, content=raw, headers={"content-type": "application/json"})
        seen.append(response.status_code)
        assert response.status_code < 500, (raw[:40], response.text[:200])
    assert 413 in seen and 404 in seen and 422 in seen and 409 in seen
    for line in ("", " ", "x" * 4096, "set v = __import__('os').system('true')"):
        response = c.post(f"/api/v1/s/{live.sid}/console", json={"line": line})
        assert response.status_code < 500, line
    assert c.post(f"/api/v1/s/{live.sid}/console", json={"line": "x" * 5000}).status_code == 422


# --------------------------------------------------------------------------- reduce


def _direct(live, addr):
    value = live.ctx.run.read(addr).detach().to(torch.float32).numpy()
    return value, {"shape": list(value.shape), "dtype": "bfloat16"}


@pytest.mark.parametrize("op,params", [("stats", {}), ("topk", {"k": 5}),
                                       ("slice", {"selector": [0, 0]}),
                                       ("histogram", {"bins": 8})])
def test_live_reduce_equals_reducing_the_tensor_read_directly(live, op, params):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "step", {"n": 6})
    spec = {"kind": "occurrence", "address": "t3.L0.v"}
    response = c.post(f"/api/v1/s/{live.sid}/reduce", json={"pointer": spec, "op": op,
                                                            "params": params})
    assert response.status_code == 200, response.text[:300]
    assert len(response.content) <= schema.CAP_BYTES
    got = response.json()["data"]
    array, info = _direct(live, "t3.L0.v")
    want = reduce_array(array, info, spec, op, params, direction_of=None).model_dump(mode="json")
    got.pop("note")
    want.pop("note")
    assert got == want
    assert "measured" in json.dumps(got)


def test_an_edited_value_reduces_with_its_supplied_label(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "set", {"target": "v", "value": 0.25, "op": "replace"})
    _op(c, live.sid, "step", {"n": 6})
    spec = {"kind": "occurrence", "address": "t3.L0.v"}
    data = c.post(f"/api/v1/s/{live.sid}/reduce", json={"pointer": spec, "op": "stats"}).json()
    data = data["data"]
    assert data["max"] == {**data["max"], "ev": "supplied", "v": 0.25}
    assert data["count"]["ev"] == "measured"
    frame = c.get(f"/api/v1/s/{live.sid}/live").json()["data"]
    v = next(p for p in frame["ports"] if p["port"] == "v")
    assert v["edit"]["edited"] and v["edit"]["authority"] == "supplied"
    later = [p for p in frame["ports"] if not p["computed"]]
    assert later and all(p["edit"] is None for p in later)  # not computed: nothing claimed


def test_a_value_the_run_has_not_computed_is_unresolved_not_zero(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    response = c.post(f"/api/v1/s/{live.sid}/reduce",
                      json={"pointer": {"kind": "occurrence", "address": "t3.L1.v"},
                            "op": "stats"})
    assert response.status_code == 409 and response.json()["refused"] == "unresolved"


# --------------------------------------------------------------------------- persistence


def test_a_checkpoint_made_live_is_listed_by_a_fresh_recorded_app(live):
    root = live.debugger.project.root
    c = _client(create_app(root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "step", {"n": 4})
    body = _op(c, live.sid, "checkpoint", {"name": "pre-edit"}).json()["data"]["result"]
    assert body["durable"]["recipe"]
    recorded = _client(create_app(root, port=PORT))
    assert live.sid in [s["id"] for s in recorded.get("/api/v1/sessions").json()["data"]]
    page = recorded.get(f"/api/v1/s/{live.sid}/records", params={"kind": "checkpoint"})
    rows = page.json()["data"]["rows"]
    names = [recorded.get(f"/api/v1/s/{live.sid}/records/{row['id']}").json()
             ["data"]["body"]["body"]["name"] for row in rows]
    assert "pre-edit" in names


def test_a_restarted_server_lists_and_restarts_a_durable_checkpoint(tmp_path):
    first = _live(tmp_path)
    try:
        c = _client(create_app(first.debugger.project.root, port=PORT, live=first))
        _op(c, first.sid, "start", {"prompt": PROMPT})
        _op(c, first.sid, "step", {"n": 5})
        _op(c, first.sid, "checkpoint", {"name": "kept"})
        where = first.ctx.run.where
    finally:
        first.close()
    second = _live(tmp_path)
    try:
        c = _client(create_app(second.debugger.project.root, port=PORT, live=second))
        frame = c.get(f"/api/v1/s/{second.sid}/live").json()["data"]
        assert {"name": "kept", "where": where, "store": "durable"} in frame["checkpoints"]
        body = _op(c, second.sid, "restart", {"name": "kept"}).json()
        assert body["state"] == "resolved" and body["data"]["where"] == where
    finally:
        second.close()


# --------------------------------------------------------------------------- views and events


def test_live_frame_and_timeline_follow_the_cursor(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    empty = c.get(f"/api/v1/s/{live.sid}/live").json()
    assert empty["data"]["where"] is None and empty["data"]["ports"] == []
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "step", {"n": 3})
    _op(c, live.sid, "break", {"pattern": "L1.p"})
    frame = c.get(f"/api/v1/s/{live.sid}/live").json()
    schema.LiveFrame.model_validate(frame["data"])
    data = frame["data"]
    listing = live.ctx.run.program.names
    assert [p["port"] for p in data["ports"]] == list(listing)
    assert [p["port"] for p in data["ports"] if p["cursor"]] == ["k"]
    assert all(p["proofs"] and p["operator"] for p in data["ports"] if p["port"] != "M")
    assert data["breakpoints"][0]["pattern"] == "L1.p"
    timeline = c.get(f"/api/v1/s/{live.sid}/live/timeline").json()
    schema.LiveTimeline.model_validate(timeline["data"])
    block = timeline["data"]
    assert block["cursor"]["label"] == "L0.k" and block["cursor"]["row"] == 3
    assert [r["count"] for r in block["rows"]] == [None, None, None, block["cursor"]["index"] + 1]
    assert [r["token"] for r in block["rows"]] == PROMPT


def test_the_clock_marks_and_upstream_edits_keep_the_selector(live):
    """M19: an edit limited by a selector (``L0.A[h1]``) is shown with it on the clock's row
    marks (committed and staged); upstream entries carry the selection field."""
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "set", {"target": "L0.A[h1]", "value": 0.5, "op": "scale"})
    _op(c, live.sid, "until", {"target": "L1.h"})
    _op(c, live.sid, "set", {"target": "L1.A[h0]", "value": 0, "op": "scale"})
    block = c.get(f"/api/v1/s/{live.sid}/live/timeline").json()["data"]
    assert block["rows"][3]["edits"] == ["L0.A[h1]", "L1.A[h0] (staged)"]
    frame = c.get(f"/api/v1/s/{live.sid}/live").json()["data"]
    schema.LiveFrame.model_validate(frame)
    assert [e["selection"] for e in frame["edits"]["committed"]] == ["h1"]
    upstream = [u for p in frame["ports"] if p["edit"] for u in p["edit"]["upstream"]]
    assert upstream and all("selection" in u for u in upstream)
    assert {u["selection"] for u in upstream if u["port"] == "A"} >= {"h1"}  # M37: the value


def test_events_push_one_op_event_per_completed_op(live):
    app = create_app(live.debugger.project.root, port=PORT, live=live, heartbeat=5.0)
    c = _client(app)
    with c.websocket_connect(f"ws://127.0.0.1:{PORT}/api/v1/s/{live.sid}/events") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["data"]["type"] == "hello" and hello["data"]["data"]["live"] == "paused"
        _op(c, live.sid, "start", {"prompt": PROMPT})
        _op(c, live.sid, "break", {"pattern": "L1.p"})
        _op(c, live.sid, "continue")
        _op(c, live.sid, "print")
        c.post(f"/api/v1/s/{live.sid}/console", json={"line": "step"})
        events = [json.loads(ws.receive_text()) for _ in range(5)]
    for event in events:
        schema.Event.model_validate(event["data"])
        assert event["schema"] == schema.SCHEMAS["event"]
    data = [e["data"]["data"] for e in events]
    assert [d["op"] for d in data] == ["start", "break", "continue", "print", "step"]
    assert [d["seq"] for d in data] == [1, 2, 3, 4, 5]
    assert data[2]["stopped"] == "breakpoint" and data[2]["where"] == "t3.L1.p"
    assert data[3]["stale"] == [] and "frame" in data[2]["stale"]
    assert data[4]["source"] == "console" and data[4]["where"] == "t3.L1.A"
    assert all(set(d) == {"op", "seq", "source", "where", "stopped", "state", "stale"}
               for d in data)
    assert "tensor" not in json.dumps(data)


def test_live_cli_flags_warn_twice_with_allow_remote(tmp_path, capsys, monkeypatch):
    from mdb.serve import __main__ as serve_main

    seen = {}

    def fake_run(app, **kwargs):
        seen["live"] = app.state.live

    monkeypatch.setitem(__import__("sys").modules, "uvicorn",
                        type("U", (), {"run": staticmethod(fake_run)}))
    rc = serve_main.main(["--project", str(tmp_path / "p"), "--tiny", "qwen2", "--host",
                          "0.0.0.0", "--allow-remote", "--allow-host", "box.local"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "without authentication" in err and "LIVE MODE on a non-loopback host" in err
    assert seen["live"] is not None and seen["live"].sid.startswith("qwen2-")
    with pytest.raises(SystemExit):  # --tiny and --model are exclusive
        serve_main.main(["--project", str(tmp_path / "p"), "--tiny", "qwen2", "--model", "x"])
    assert serve_main.main(["--project", str(tmp_path / "p"), "--model",
                            str(tmp_path / "nowhere")]) == 1
    assert "cannot attach the live model" in capsys.readouterr().err


def test_serve_tiny_takes_every_engine_tiny_family(tmp_path, monkeypatch):
    from mdb.engine.models import TINY_FAMILIES
    from mdb.serve import __main__ as serve_main

    assert serve_main.TINY_FAMILIES == TINY_FAMILIES
    seen = []
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", type("U", (), {
        "run": staticmethod(lambda app, **kwargs: seen.append(app.state.live.sid))}))
    for family in ("llama", "mistral"):
        assert serve_main.main(["--project", str(tmp_path / family), "--tiny", family]) == 0
    assert [sid.split("-")[0] for sid in seen] == ["llama", "mistral"]


# --------------------------------------------------------------------------- fix round (audit)


def _frame(c, sid):
    response = c.get(f"/api/v1/s/{sid}/live")
    assert response.status_code == 200, response.text[:300]
    assert len(response.content) <= schema.CAP_BYTES
    schema.LiveFrame.model_validate(response.json()["data"])
    return response.json()


def _page_through(c, sid, name, addr=None):
    """Every item of one frame list, page by page through ``GET .../live/items``."""
    items, offset, total = [], 0, None
    while offset is not None:
        params = {"list": name, "offset": offset, **({"addr": addr} if addr else {})}
        response = c.get(f"/api/v1/s/{sid}/live/items", params=params)
        assert response.status_code == 200, response.text[:300]
        assert len(response.content) <= schema.CAP_BYTES
        page = schema.LiveItems.model_validate(response.json()["data"])
        assert page.offset == offset and page.items
        items += page.items
        total, offset = page.total, page.next
    assert len(items) == total
    return items


def test_the_live_frame_always_fits_the_cap_and_pages_the_rest(live):
    """The audit's states: many committed edits (one per row and layer), 8 displays (the op's
    limit; the audit's 64 asked for more and 56 were refused), 64
    breakpoints with 247-character conditions (the engine's limit is 256), 64 checkpoints with 64-character names. The
    frame stays a 200 under the cap, says what it left out, and the items route returns all."""
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    _op(c, sid, "start", {"prompt": PROMPT})
    for row in range(3, 34):  # 62 committed edits (the audit saw 413 from 34 on); a durable
        for layer in (0, 1):  # checkpoint's recipe holds at most 64 edited transitions
            _op(c, sid, "until", {"target": f"t{row}.L{layer}.v"})
            assert _op(c, sid, "set", {"target": f"L{layer}.v", "value": 1.5, "op": "scale"}
                       ).json()["state"] == "resolved"
            _op(c, sid, "commit")
        _op(c, sid, "step", {"n": 3})
    _op(c, sid, "until", {"target": "t34.L0.v"})
    _op(c, sid, "set", {"target": "L0.v", "value": 2.0, "op": "scale"})
    _frame(c, sid)
    for i in range(64):
        _op(c, sid, "display", {"address": f"t{i}.L0.h"})
        assert _op(c, sid, "break", {"pattern": f"t{i}.L*.p",
                                     "condition": "norm > " + "9" * 240}).status_code == 200
        assert _op(c, sid, "checkpoint", {"name": f"c{i:02d}-" + "n" * 60}).status_code == 200
    body = _frame(c, sid)
    data, more = body["data"], {m["field"]: m for m in body["data"]["more"]}
    assert body["omitted"] == data["more"]
    assert {"committed", "breakpoints", "checkpoints"} <= set(more)
    assert data["edits"]["committed_total"] == 62 and data["edits"]["staged_total"] == 1
    assert len(live.ctx.displays) == 8
    for name, total in (("committed", 62), ("breakpoints", 64), ("displays", 8),
                        ("checkpoints", 64), ("staged", 1)):
        shown = len(data["edits"][name]) if name in ("committed", "staged") else len(data[name])
        assert shown + (more[name]["count"] if name in more else 0) == total, name
        if name in more:
            assert more[name]["reason"] == (
                f"GET /api/v1/s/{sid}/live/items?list={name}&offset={shown}")
        everything = _page_through(c, sid, name)
        assert len(everything) == total
    # one port's upstream edits: a few in the frame, all through the items route
    port = next(p for p in data["ports"] if p["computed"] and p["port"] == "h")
    assert port["edit"]["upstream_total"] == 62 and len(port["edit"]["upstream"]) <= 4
    assert len(_page_through(c, sid, "upstream", port["addr"])) == 62
    # whole items: the paged breakpoints carry their full conditions
    assert all(len(b["condition"]) == 247 for b in _page_through(c, sid, "breakpoints"))
    # the console and the controls still answer
    assert _op(c, sid, "where").json()["state"] == "resolved"


def test_a_checkpoint_name_over_64_characters_is_refused(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    response = _op(c, live.sid, "checkpoint", {"name": "n" * 65})
    assert response.status_code == 409
    assert "at most 64 characters" in response.json()["data"]["result"]["reason"]
    assert live.ctx.checkpoints == {} and live.debugger.checkpoints() == []


def test_one_checkpoint_row_per_name_and_restart_goes_where_it_says(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    _op(c, sid, "start", {"prompt": PROMPT})
    _op(c, sid, "step", {"n": 2})
    _op(c, sid, "checkpoint", {"name": "x"})
    _op(c, sid, "step", {"n": 7})
    latest = _op(c, sid, "checkpoint", {"name": "x"}).json()["data"]["where"]
    for moved in (False, True):
        if moved:
            _op(c, sid, "start", {"prompt": PROMPT})  # in-memory checkpoints are gone
        rows = _frame(c, sid)["data"]["checkpoints"]
        assert rows == [{"name": "x", "where": latest, "store": "durable"}]
        assert _op(c, sid, "restart", {"name": "x"}).json()["data"]["where"] == latest


@pytest.mark.parametrize("history", ["finish", "commit", "step-after-commit", "reverse"])
def test_every_computed_port_reduces_live_and_nothing_is_written(live, history):
    """Live reductions read like ``print`` (``Run.read``): every port the frame marks computed
    resolves after any cursor history, and the project does not grow."""
    root = live.debugger.project.root
    c = _client(create_app(root, port=PORT, live=live))
    sid = live.sid
    _op(c, sid, "start", {"prompt": PROMPT})
    _op(c, sid, "until", {"target": "L0.v"})
    if history != "finish":
        _op(c, sid, "set", {"target": "L0.v", "value": 2.0, "op": "scale"})
    _op(c, sid, "finish" if history == "finish" else "commit")
    if history == "step-after-commit":
        _op(c, sid, "step", {"n": 3})
    if history == "reverse":
        _op(c, sid, "step", {"n": 3})
        _op(c, sid, "reverse-step", {"n": 2})

    def size():
        files = [p for p in root.rglob("*") if p.is_file()]
        return len(files), sum(p.stat().st_size for p in files)

    def reduce_op(address, op, params=None):
        response = c.post(f"/api/v1/s/{sid}/reduce", json={
            "pointer": {"kind": "occurrence", "address": address}, "op": op,
            "params": params or {}})
        assert response.status_code == 200, (address, response.text[:300])
        return response.json()["data"]

    # Only the reductions are bracketed: under committed edits the `print` op mints an
    # occurrence pointer and stores the edit recipe (M8), a write of `print`, not of a reduce.
    # `print` is only the reference the reduction's label is compared with.
    before = size()
    ports = [p for p in _frame(c, sid)["data"]["ports"] if p["computed"]]
    assert ports
    reduced = {}
    for port in ports:
        reduce_op(port["addr"], "stats")
        reduce_op(port["addr"], "topk", {"k": 3})
        reduced[port["addr"]] = reduce_op(port["addr"], "stats")
    assert size() == before
    for port in ports:
        printed = _op(c, sid, "print", {"address": port["addr"]}).json()["data"]["result"]
        stats = reduced[port["addr"]]
        if printed["shape"] == stats["shape"] and "stats" in printed:  # same value, same label
            assert stats["max"]["ev"] == printed["stats"]["max"]["authority"]
    # the recipe `print` stored is the same row on a second print: that read adds nothing
    after_print = size()
    for port in ports:
        _op(c, sid, "print", {"address": port["addr"]})
        reduce_op(port["addr"], "stats")
    assert size() == after_print


def test_interrupt_cancels_every_queued_job_and_says_how_many(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    _op(c, sid, "start", {"prompt": PROMPT})
    where = live.ctx.run.where
    gate = threading.Event()
    blocker = live.call(gate.wait)
    while live._current is None:
        time.sleep(0.001)
    queued = [live.op("step", {"n": 2}), live.op("set", {"target": "L0.v", "value": 2.0}),
              live.call(lambda: "a view")]
    stop = c.post(f"/api/v1/s/{sid}/interrupt").json()["data"]
    assert stop == {"interrupted": False, "running": None, "cancelled": 3,
                    "reason": "nothing is running; cancelled 3 queued"}
    gate.set()
    blocker.result(timeout=10)  # a running read is never interrupted
    for future in queued[:2]:
        assert future.result(timeout=10)["result"] == {
            "interrupted": where, "state": "unresolved", "reason": f"interrupted at {where}"}
    with pytest.raises(Exception, match="interrupted"):
        queued[2].result(timeout=10)
    assert live.ctx.run.where == where and live.ctx.run.edits == []  # nothing ran
    assert _op(c, sid, "step").json()["state"] == "resolved"  # later work runs


class _Analysis:
    """Stands in for what an op builds (row models, float64 weights)."""


def test_the_worker_keeps_no_finished_or_failed_job_alive(live):
    built = []

    def fail():
        work = _Analysis()
        built.append(weakref.ref(work))
        raise ValueError("the analysis failed")

    def succeed():
        work = _Analysis()
        built.append(weakref.ref(work))
        return work

    failed = live.call(fail)
    assert isinstance(failed.exception(timeout=10), ValueError)   # its holder keeps the error
    done = live.call(succeed)
    assert isinstance(done.result(timeout=10), _Analysis)
    del done
    deadline = time.monotonic() + 5
    while any(ref() is not None for ref in built) and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)
    # the worker, now waiting, holds neither the last result nor the failed job's frames
    assert [ref() for ref in built] == [None, None]
    assert "the analysis failed" in str(failed.exception())


@pytest.mark.parametrize("site,status", [("cross-site", 403), ("same-site", 403),
                                         ("same-origin", 200), ("none", 200), (None, 200)])
def test_sec_fetch_site_other_than_same_origin_or_none_is_refused(live, site, status):
    from starlette.websockets import WebSocketDisconnect

    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    headers = {} if site is None else {"sec-fetch-site": site}
    for method, route, body in NEW_ROUTES:
        response = c.request(method, f"/api/v1/s/{live.sid}/{route}", json=body,
                             headers=headers)
        if status == 403:
            assert response.status_code == 403 and response.json()["refused"] == "cross_site"
        else:
            assert response.status_code != 403, (route, response.text[:200])
    for path in ("/api/v1/hello", "/"):
        response = c.get(path, headers=headers)
        assert (response.status_code == 403) is (status == 403), path
    url = f"ws://127.0.0.1:{PORT}/api/v1/s/{live.sid}/events"
    if status == 403:
        with pytest.raises(WebSocketDisconnect) as closed:
            with c.websocket_connect(url, headers=headers):
                pass
        assert closed.value.code == 1008
    else:
        with c.websocket_connect(url, headers=headers) as socket:
            assert json.loads(socket.receive_text())["data"]["type"] == "hello"


def test_refusals_name_the_project_relatively_never_its_absolute_path(live):
    root = str(live.debugger.project.root)
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    body = _op(c, live.sid, "restart", {"name": "nope"}).json()
    reason = body["data"]["result"]["reason"]
    assert root not in json.dumps(body) and "this project" in reason, reason  # no path at all
    text = c.post(f"/api/v1/s/{live.sid}/console", json={"line": "restart nope"}).json()
    assert root not in text["data"]["text"] and "this project" in text["data"]["text"]


def test_the_recorded_frame_of_the_live_session_says_it_is_live(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    _op(c, live.sid, "start", {"prompt": PROMPT})
    _op(c, live.sid, "step", {"n": 2})
    _op(c, live.sid, "checkpoint", {"name": "cut"})  # the recorded frame needs a stored cut
    data = c.get(f"/api/v1/s/{live.sid}/frame").json()["data"]
    assert data["live"]["state"] == "paused"
    text = " ".join(data["live"]["frontier"])
    assert "engine is attached" in text and "no engine attached" not in text
    assert live.ctx.run.where in text


# --------------------------------------------------------------------------- ledger


def _ledger(c, sid, **params):
    response = c.get(f"/api/v1/s/{sid}/live/ledger", params=params)
    assert response.status_code == 200, response.text[:300]
    body = response.json()
    assert body["schema"] == schema.SCHEMAS["live-ledger"]
    return body


def test_the_ledger_lists_the_circuits_forecasts_and_symbols_the_ops_made(live):
    from mdb.analysis import circuits, forecast, symbols

    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    empty = _ledger(c, sid)["data"]
    assert (empty["where"], empty["circuits"], empty["forecasts"], empty["symbols"]) == (
        None, [], [], [])
    _op(c, sid, "start", {"prompt": PROMPT})
    _op(c, sid, "forecast", {"target": "L0.p"})
    _op(c, sid, "until", {"target": "t3.logits"})
    _op(c, sid, "verify", {})
    built = _op(c, sid, "circuit", {"target": "L1.o", "top": 1, "depth": 2}).json()
    refused = _op(c, sid, "circuit", {"target": "test", "id": built["data"]["result"]["id"]}).json()
    assert "circuit test is disabled" in json.dumps(refused) and "issue M4" in json.dumps(refused)
    _op(c, sid, "symbol", {"name": "ctx", "addresses": "t1"})
    data = _ledger(c, sid)["data"]
    run = live.ctx.run
    assert data["where"] == run.where and data["seq"] == live.seq and data["more"] == []
    (circuit,) = data["circuits"]
    entry = circuits.ledger(run)[circuit["id"]]
    body = entry.record["body"]
    assert circuit == {"id": entry.id, "consumer": body["consumer"], "k": 1, "depth": 2,
                       "count": body["count"], "where": body["where"],
                       "record": entry.record["id"], "tests": 0,
                       "verdict": None, "from_previous_run": False}
    (sealed,) = data["forecasts"]
    view = forecast.view(forecast.ledger(run)[sealed["id"]])
    assert sealed["target"] == view["target"] == "t3.L0.p"
    assert sealed["status"] == view["status"] and sealed["verdict"] == view["verdict"]
    assert sealed["predicted"] == view["predicted"] and sealed["steps"] == len(view["method"])
    assert "premises" not in sealed["bound"] and sealed["bound"]["max"] == view["bound"]["max"]
    assert data["symbols"] == [{"name": "ctx", "definitions": ["t1"], "rows": [1]}]
    assert set(symbols.declared(run)) == {"ctx"}


def test_the_ledger_marks_circuits_carried_from_a_previous_run(live):
    """M12 keeps the circuit ledger across ``start``: a circuit built by the replaced run is
    listed with ``from_previous_run`` (as ``circuit show`` marks it); a new one is not."""
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    _op(c, sid, "start", {"prompt": PROMPT})
    _op(c, sid, "until", {"target": "t3.logits"})
    first = _op(c, sid, "circuit", {"target": "M[4,9]", "top": 1, "depth": 1}).json()
    assert [r["from_previous_run"] for r in _ledger(c, sid)["data"]["circuits"]] == [False]
    _op(c, sid, "start", {"prompt": [1, 7, 9, 3]})
    _op(c, sid, "until", {"target": "t3.logits"})
    _op(c, sid, "circuit", {"target": "M[4,9]", "top": 1, "depth": 1})
    rows = _ledger(c, sid)["data"]["circuits"]
    assert [(r["id"], r["from_previous_run"]) for r in rows] == [
        (first["data"]["result"]["id"], True), (first["data"]["result"]["id"] + 1, False)]
    shown = _op(c, sid, "circuit", {"target": "show", "id": rows[0]["id"]}).json()
    assert shown["data"]["result"]["from_previous_run"] is True


def test_the_ledger_fits_the_cap_and_names_what_it_left_out(live):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    sid = live.sid
    _op(c, sid, "start", {"prompt": PROMPT})
    for _ in range(70):
        assert _op(c, sid, "forecast", {"target": "M[4,9]"}).status_code == 200
    body = _ledger(c, sid)
    data = body["data"]
    assert body["bytes"] <= schema.CAP_BYTES and body["omitted"] == data["more"]
    (more,) = data["more"]
    assert more["field"] == "forecasts" and more["count"] + len(data["forecasts"]) == 70
    ids = [f["id"] for f in data["forecasts"]]
    assert ids == list(range(71 - len(ids), 71))  # the latest, in id order
    few = _ledger(c, sid, limit=3)["data"]
    assert [f["id"] for f in few["forecasts"]] == [68, 69, 70]
    assert few["more"][0]["count"] == 67


@pytest.mark.parametrize("query", ["limit=0", "limit=65", "limit=-1", "limit=abc",
                                   "limit=" + "9" * 400, "limit=1.5"])
def test_malformed_ledger_queries_never_5xx(live, query):
    c = _client(create_app(live.debugger.project.root, port=PORT, live=live))
    response = c.get(f"/api/v1/s/{live.sid}/live/ledger?{query}")
    assert response.status_code in (400, 422), (query, response.status_code)
    assert response.json()["state"] == "refused"

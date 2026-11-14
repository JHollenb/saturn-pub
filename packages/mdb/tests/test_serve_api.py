"""``mdb serve`` contract over a recorded fixture project (no model, no torch).

The fixture is generated once with ``tests/fixtures/serve_fixture_project.py`` (needs torch) and
located through ``MDB_SERVE_FIXTURE`` (the ``project`` directory). Tests that need it skip when
it is absent; the type, middleware and loopback tests always run. Needs the ``serve`` extra.
"""
from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
np = pytest.importorskip("numpy")

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

from mdb.serve import schema  # noqa: E402
from mdb.serve.app import create_app  # noqa: E402
from mdb.serve.caps import CapMiddleware, canonical  # noqa: E402
from mdb.serve.guard import SecurityLayer  # noqa: E402

FIXTURE = os.environ.get("MDB_SERVE_FIXTURE")
CAP = 65536
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
needs_fixture = pytest.mark.skipif(
    not FIXTURE or not (Path(FIXTURE) / "project.sqlite").is_file(),
    reason="set MDB_SERVE_FIXTURE to a project from tests/fixtures/serve_fixture_project.py")

MODELS = {
    "hello": schema.Hello, "sessions": list[schema.SessionSummary],
    "capabilities": schema.Capabilities, "frame": schema.Frame,
    "timeline": schema.TimelineBlock, "branches": schema.BranchTree,
    "symbols": schema.SymbolPage, "symbol": schema.SymbolDetail, "graph": schema.GraphView,
    "records": schema.RecordPage, "record": schema.RecordDetail,
    "reduce": schema.ReduceResult,
}


def make_client(path=None, *, raise_server_exceptions=True, **kwargs):
    """A client whose requests carry the loopback Host the server was started with."""
    app = create_app(path or FIXTURE, port=PORT, **kwargs)
    return TestClient(app, base_url=BASE, raise_server_exceptions=raise_server_exceptions)


# Keys whose own value is structure (an int, a list of ints, or a flat name -> int map) come from
# ONE place, ``schema.STRUCTURAL_KEYS`` (also emitted into the web client's generated types).
STRUCTURAL = set(schema.STRUCTURAL_KEYS)
# Fields that carry a stored document verbatim (bounded); their numbers are the record's own and
# the renderer refuses any that lack an evidence class. Listed so the exemption is explicit.
VERBATIM = {"data.data"}  # the event payload slot (checked on its own)


def _int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _server_shapes():
    """Key sets of the server's own labelled-number shapes (every model with ``ev`` and ``v`` or
    ``vs``): a dict that has ``ev`` and ``v``/``vs`` but not exactly one of these is a look-alike
    (stored data) and is walked like any mapping."""
    shapes = set()
    for model in (schema.Num, schema.NumList, schema.TimelineCell):
        shapes.add(frozenset(model.model_fields))
    return shapes


SERVER_SHAPES = _server_shapes()


def server_built(value):
    return isinstance(value, dict) and frozenset(value) in SERVER_SHAPES


def unlabelled_numbers(value, key=None, path="data"):
    """Paths of numeric leaves that are neither inside a labelled Num/NumList nor structural.

    A structural key exempts only ITS OWN value: a scalar int, a list of ints, or a flat map of
    names to ints (a count per kind). A float, a nested object or a list of objects under it is
    walked, so ``rows`` or ``symbols`` cannot hide a bare number beneath them."""
    if key in VERBATIM or isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [] if key in STRUCTURAL and _int(value) else [path]
    if isinstance(value, dict):
        if server_built(value):
            return []
        if key in STRUCTURAL and value and all(_int(v) for v in value.values()):
            return []
        out = []
        for k, v in value.items():
            out += unlabelled_numbers(v, k, f"{path}.{k}")
        return out
    if key in STRUCTURAL and all(_int(v) for v in value):
        return []
    out = []
    for i, v in enumerate(value):
        out += unlabelled_numbers(v, key if isinstance(v, list) else None, f"{path}[{i}]")
    return out


def check(response, view=None):
    """Every response: <= cap, canonical JSON, an Envelope of its view or a Refused."""
    raw = response.content
    assert len(raw) <= CAP, (response.url, len(raw))
    body = json.loads(raw)
    assert raw == canonical(body)
    if "refused" in body:
        schema.Refused.model_validate(body)
        assert response.status_code >= 400 or body["refused"]
        assert body["state"] == "refused" and body["refused"]  # the reason is never empty
        return body
    assert response.status_code == 200
    assert body["state"] in ("resolved", "unresolved", "refused")
    assert body["state"] == "resolved" or body.get("reason"), "a non-resolved state says why"
    name = next(k for k, v in schema.SCHEMAS.items() if v == body["schema"])
    assert view is None or name == view
    TypeAdapter(schema.Envelope[MODELS[name]]).validate_python(body)
    assert body["bytes"] == len(canonical(body["data"]))
    assert "state" in body
    bare = unlabelled_numbers(body["data"], "data.data" if name == "event" else None)
    assert not bare, f"{name}: numbers without an evidence class at {bare[:5]}"
    return body


@pytest.fixture(scope="module")
def client():
    return make_client()


@pytest.fixture()
def store_copy(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(FIXTURE, root)
    return root


# --------------------------------------------------------------------------- always run

def test_generated_types_are_current():
    from mdb.serve import typegen

    assert typegen.TARGET.read_text() == typegen.render(), \
        "run: python -m mdb.serve.typegen --write"


def test_cap_middleware_refuses_never_truncates():
    from fastapi import FastAPI
    from fastapi.responses import Response

    app = FastAPI()

    payload = b'{"x":"' + b"a" * 70000 + b'"}'

    @app.get("/api/v1/big")
    def big():
        return Response(payload, media_type="application/json")

    @app.get("/api/v1/small")
    def small():
        return Response(b'{"x":1}', media_type="application/json")

    app.add_middleware(CapMiddleware, cap=CAP)
    c = TestClient(app)
    over = c.get("/api/v1/big")
    assert over.status_code == 413
    assert over.json() == {"refused": "over_cap", "state": "refused", "bytes": len(payload),
                           "cap": CAP}
    assert c.get("/api/v1/small").json() == {"x": 1}


def test_num_refuses_a_value_without_evidence():
    with pytest.raises(ValueError):
        schema.Num(v=None, ev="unresolved")  # no reason
    with pytest.raises(ValueError):
        schema.Num(v=1.0, ev="unresolved", reason="x")
    with pytest.raises(ValueError):
        schema.Num(v=None, ev="measured")
    assert schema.Num(v=None, ev="unresolved", reason="not recorded").v is None


def test_loopback_only_by_default(tmp_path, capsys):
    from mdb.serve.__main__ import is_loopback, main

    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.4")
    assert main(["--project", str(tmp_path), "--host", "0.0.0.0"]) == 2
    assert "refusing non-loopback host" in capsys.readouterr().err


# --------------------------------------------------------------------------- fixture routes

def gets(c):
    """(url, params) for every GET route over the fixture's sessions."""
    out = [("/api/v1/hello", {}), ("/api/v1/sessions", {})]
    for sid in [s["id"] for s in c.get("/api/v1/sessions").json()["data"]]:
        base = f"/api/v1/s/{sid}"
        caps = c.get(base + "/capabilities").json()["data"]
        nx, ny = (len(a["ticks"]) for a in caps["axes"][:2])
        out += [(base + "/capabilities", {}), (base + "/branches", {}), (base + "/frame", {})]
        addresses = ["t0.L0.h", "t1.logits", "t2.L1.p[h0,src=0]", "cache.L1.k[kv0,row=0]",
                     "t2.L0.h~prefill", 't2.logits[" hot"=5]', "main:t2.L0.h", "b9:t2.L0.h",
                     "t99.L0.h", "bad["]
        for x, y in itertools.product(range(nx + 1), range(ny)):
            tick = caps["axes"][1]["ticks"][y]
            addresses.append(".".join([f"t{x}"] + ([tick["frag"]] if tick["frag"] else []))
                             + "." + (tick["port"] or "h"))
        out += [(base + "/frame", {"addr": a}) for a in addresses]
        for lens, win in itertools.product(["cuts", "receipts", "symbols", "records", "nope"],
                                           [{}, {"x0": 0, "x1": 1}, {"x0": 1, "x1": 600},
                                            {"y0": 2, "y1": 1}]):
            out.append((base + "/timeline", {"lens": lens, **win}))
        out += [(base + "/timeline", {"branch": b}) for b in ("main", "b1", "zz")]
        for params in ({}, {"page": 1}, {"q": "carrier"}, {"at": "t2.L0.h"}, {"at": "bad["}):
            out.append((base + "/symbols", params))
        for row in c.get(base + "/symbols").json()["data"]["rows"]:
            out.append((base + f"/symbols/{row['id']}", {}))
        out.append((base + "/symbols/" + "0" * 64, {}))
        for zoom, k, depth in itertools.product(["slice", "session", "x"], [1, 8, 64, 65],
                                                [0, 3]):
            out.append((base + "/graph", {"zoom": zoom, "k": k, "depth": depth,
                                          "addr": "t2.L0.h"}))
        records = c.get(base + "/records").json()["data"]
        out += [(base + "/records", {"kind": k}) for k in records["kinds"]]
        out += [(base + "/records", {"page": 3})]
        for kind in records["kinds"]:
            for row in c.get(base + "/records", params={"kind": kind}).json()["data"]["rows"]:
                out.append((base + f"/records/{row['id']}", {}))
        out.append((base + "/export", {}))
    out.append(("/api/v1/s/missing/frame", {}))
    return out


@needs_fixture
def test_every_get_route_is_an_envelope_or_refusal_under_the_cap(client):
    seen, largest = 0, (0, "")
    statuses = {}
    for url, params in gets(client):
        response = client.get(url, params=params)
        body = check(response)
        seen += 1
        statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
        if len(response.content) > largest[0]:
            largest = (len(response.content), f"{url} {params}")
        if "refused" not in body:
            assert body["bytes"] <= CAP
    assert seen > 150 and statuses.get(200, 0) > 100
    print(f"\ncap sweep: {seen} GET responses, statuses {statuses}, largest {largest}")


@needs_fixture
def test_unreached_cells_are_unresolved_not_zero(client):
    body = check(client.get("/api/v1/s/tiny/timeline", params={"lens": "cuts"}), "timeline")
    data = body["data"]
    assert data["fill"]["ev"] == "unresolved" and data["fill"]["v"] is None
    listed = {(c["x"], c["y"]) for c in data["cells"]}
    assert (0, 0) not in listed  # prefill row: never recorded
    assert all(c["ev"] == "measured" and c["v"] is not None for c in data["cells"])
    axes = [a["id"] for a in data["axes"]]
    assert axes == ["token", "layer"]


@needs_fixture
def test_symbols_show_reads_with_measured_mass_and_an_unresolved_unbound_row(client):
    data = check(client.get("/api/v1/s/qwen/symbols"), "symbols")["data"]
    assert data["unbound"]["ev"] == "unresolved" and data["unbound"]["reason"]
    rows = {r["symbol"]: r for r in data["rows"]}
    readers = rows["description.L1"]["readers"]
    assert len(readers) == 4 and all(r["mass"]["ev"] == "measured" for r in readers)
    assert all(r["verdict"] == "untested" for r in readers)
    assert rows["description.L1"]["locations"][0].startswith("cache.L1.")


@needs_fixture
def test_comparison_record_lists_sham_arm_first(client):
    rows = check(client.get("/api/v1/s/tiny/records", params={"kind": "comparison"}))
    rid = rows["data"]["rows"][0]["id"]
    detail = check(client.get(f"/api/v1/s/tiny/records/{rid}"), "record")["data"]
    assert detail["summary"]["schema"] == "mdb-circuit-investigation-v1"
    assert [a["mask"] for a in detail["arms"]] == [0, 1, 2, 3]
    assert detail["arms"][0]["label"].startswith("sham")
    assert detail["arms"][0]["value"]["ev"] == "measured"
    assert detail["arms"][1]["effect"]["ev"] == "derived-exact"


@needs_fixture
def test_events_socket_says_hello_then_heartbeats():
    c = make_client(heartbeat=0.05)
    with c.websocket_connect(f"ws://127.0.0.1:{PORT}/api/v1/s/tiny/events") as ws:
        first, second = json.loads(ws.receive_text()), json.loads(ws.receive_text())
    assert first["schema"] == schema.SCHEMAS["event"]
    assert first["data"]["type"] == "hello" and second["data"]["type"] == "heartbeat"
    assert second["data"]["seq"] == 1


@needs_fixture
def test_op_dispatches_through_the_registry_or_refuses(client):
    body = check(client.post("/api/v1/s/tiny/op", json={"op": "start", "args": {},
                                                          "request_id": "r1"}))
    assert body["refused"] in ("read_only", "ops registry unavailable")
    body = check(client.post("/api/v1/s/tiny/op", json={"op": "no-such-op", "args": {},
                                                         "request_id": "r2"}))
    assert "refused" in body


@needs_fixture
def test_small_cap_refuses_real_routes(tmp_path):
    c = make_client(cap=1024)
    over = c.get("/api/v1/s/tiny/capabilities")
    assert over.status_code == 413 and over.json()["refused"] == "over_cap"
    assert over.json()["bytes"] > 1024


@needs_fixture
def test_static_serving_spa_and_missing_assets(tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>mdb</title>")
    (static / "assets" / "app.js").write_text("export {}")
    project = FIXTURE
    c = make_client(project, static_dir=str(static))
    assert c.get("/").text.startswith("<!doctype html>")
    assert c.get("/some/view").text.startswith("<!doctype html>")  # SPA route
    assert c.get("/assets/app.js").text == "export {}"
    missing = c.get("/assets/nope.js")
    assert missing.status_code == 404 and missing.json()["refused"] == "not_found"
    assert c.get("/../../etc/passwd").status_code in (200, 404)  # never outside static
    assert "root:" not in c.get("/..%2F..%2Fetc%2Fpasswd").text
    absent = make_client(project, static_dir=str(tmp_path / "none"))
    body = absent.get("/").json()
    assert body["refused"] == "web_build_absent" and "npm run build" in body["detail"]


@needs_fixture
def test_absent_lazy_tables_read_as_empty_and_are_never_created(store_copy):
    import sqlite3

    lazy = ("records", "cursor", "recipes", "occurrences")
    db = store_copy / "project.sqlite"
    with sqlite3.connect(db) as con:
        for table in lazy:
            con.execute(f"DROP TABLE IF EXISTS {table}")
        version = con.execute("PRAGMA user_version").fetchone()[0]
    c = make_client(store_copy)
    for path in ("/records", "/frame", "/timeline", "/symbols", "/graph", "/capabilities"):
        check(c.get(f"/api/v1/s/qwen{path}"))
    page = check(c.get("/api/v1/s/qwen/records"), "records")["data"]
    assert "occurrence" not in page["kinds"]  # the v2 row lived in the dropped table
    frame = check(c.get("/api/v1/s/qwen/frame", params={"addr": "t2.L1.p"}), "frame")["data"]
    assert all(p["spec"] is None or p["spec"]["kind"] != "blob" for p in frame["focus"]["pointers"])
    missing = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "occurrence", "address": "t2.L1.p"}, "op": "stats", "params": {}}))
    assert missing["refused"] == "unresolved"
    with sqlite3.connect(db) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not tables & set(lazy), tables & set(lazy)
        assert con.execute("PRAGMA user_version").fetchone()[0] == version == 3


def _tree_digest(root):
    """sha256 of every file plus every directory (so a created side file or directory shows)."""
    import hashlib

    out = {}
    for path in sorted(Path(root).rglob("*")):
        rel = str(path.relative_to(root))
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "dir"
    return out


@needs_fixture
def test_serving_leaves_a_schema3_project_byte_identical(store_copy):
    import sqlite3

    db = store_copy / "project.sqlite"
    with sqlite3.connect(db) as con:  # an old schema-3 project: no lazily created tables
        for table in ("records", "cursor", "recipes", "occurrences"):
            con.execute(f"DROP TABLE IF EXISTS {table}")
    con.close()
    before = _tree_digest(store_copy)
    c = make_client(store_copy)
    for url, params in gets(c):
        check(c.get(url, params=params))
    for sid in ("tiny", "qwen"):
        frame = c.get(f"/api/v1/s/{sid}/frame").json()["data"]
        for pointer in frame["focus"]["pointers"]:
            if pointer["spec"]:
                check(c.post(f"/api/v1/s/{sid}/reduce", json={
                    "pointer": pointer["spec"], "op": "stats", "params": {}}))
    del c
    assert _tree_digest(store_copy) == before


@needs_fixture
def test_an_undeclared_record_kind_is_unresolved_per_row(store_copy):
    import sqlite3

    db = store_copy / "project.sqlite"
    with sqlite3.connect(db) as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(records)")]
        picked = ", ".join("'mystery'" if c == "kind" else "'zz-' || id" if c == "id" else c
                           for c in cols)
        con.execute(f"INSERT INTO records ({', '.join(cols)}) SELECT {picked} FROM records "
                    "WHERE session_id='qwen' LIMIT 1")
    con.close()
    c = make_client(store_copy)
    page = check(c.get("/api/v1/s/qwen/records"), "records")["data"]
    rows = {r["kind"]: r for r in page["rows"]}
    assert rows["mystery"]["status"] == "unresolved"
    assert rows["mystery"]["nums"][0]["n"]["ev"] == "unresolved"
    assert "undeclared kind mystery" in rows["mystery"]["nums"][0]["n"]["reason"]
    assert "occurrence" in rows and rows["occurrence"]["status"] == "sealed"
    detail = check(c.get(f"/api/v1/s/qwen/records/{rows['mystery']['id']}"), "record")
    assert detail["data"]["summary"]["status"] == "unresolved"


@needs_fixture
def test_value_key_aliases_land_on_one_cell(client):
    def pos(addr):
        return check(client.get("/api/v1/s/qwen/frame", params={"addr": addr}), "frame")["data"]

    # The qwen fixture has 2 decoder layers: t2.L0.h_out is the same value as t2.L1.h.
    a, b = pos("t2.L0.h_out"), pos("t2.L1.h")
    assert a["pos"] == b["pos"] and a["key"] == b["key"] == "t2.L1.h"
    e, h0 = pos("t1.emb"), pos("t1.L0.h")
    assert e["pos"] == h0["pos"] and e["key"] == h0["key"] == "t1.L0.h"
    # Hidden-state ports resolve with the recording's layer count (core needs layers=).
    hidden = pos("t2.L1.h")["focus"]
    assert not hidden["note"] or "layers=" not in hidden["note"]


# --------------------------------------------------------------------------- reduce

def reduce(c, pointer, op, params=None):
    return c.post("/api/v1/s/qwen/reduce", json={"pointer": pointer, "op": op,
                                                  "params": params or {}})


def f32(data):
    return np.frombuffer(base64.b64decode(data), dtype="<f4")


@needs_fixture
def test_reduce_matches_numpy_on_stored_blobs(store_copy):
    from saturn_pub.store import LocalStore

    store = LocalStore(store_copy / "store")
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((200, 300)).astype(np.float32)
    matrix[3, 4] = np.nan
    ref = store.put_tensor(matrix)["ref"]
    direction = rng.standard_normal(300).astype(np.float32)
    dref = store.put_tensor(direction)["ref"]
    c = make_client(store_copy)
    blob = {"kind": "blob", "ref": ref}
    expected = store.get_tensor(ref, framework="numpy").astype(np.float64)
    finite = expected[np.isfinite(expected)]

    stats = check(reduce(c, blob, "stats"), "reduce")["data"]
    assert stats["count"]["v"] == 60000 and stats["finite"]["v"] == 59999
    for key, value in (("min", finite.min()), ("max", finite.max()), ("mean", finite.mean()),
                       ("std", finite.std()), ("absmax", np.abs(finite).max()),
                       ("l2", np.sqrt((finite ** 2).sum()))):
        assert stats[key]["v"] == pytest.approx(value, rel=1e-12), key
        assert stats[key]["ev"] == "measured"

    top = check(reduce(c, blob, "topk", {"k": 5, "by": "abs"}), "reduce")["data"]
    flat = np.where(np.isfinite(expected), np.abs(expected), -np.inf).reshape(-1)
    want = np.argsort(-flat, kind="stable")[:5]
    assert [e["index"] for e in top["entries"]] == [list(np.unravel_index(i, (200, 300)))
                                                    for i in want]
    rows = check(reduce(c, blob, "topk", {"k": 3, "axis": 0, "by": "signed"}))["data"]
    sums = np.nansum(expected, axis=1)
    assert [e["index"][0] for e in rows["entries"]] == list(np.argsort(-sums)[:3])

    part = check(reduce(c, blob, "slice", {"selector": [[10, 20], [0, 300]]}))["data"]
    assert part["outShape"] == [10, 300]
    assert np.array_equal(f32(part["data"]).reshape(10, 300), matrix[10:20])
    refused = check(reduce(c, blob, "slice", {"selector": [[0, 20]]}))
    assert refused["refused"] == "over_limit"

    hist = check(reduce(c, blob, "histogram", {"bins": 16}))["data"]
    counts, edges = np.histogram(finite, bins=16)
    assert hist["counts"] == counts.tolist() and hist["nonfinite"] == 1
    assert np.allclose(hist["edges"]["vs"], edges, rtol=0, atol=0)
    assert hist["edges"]["ev"] == "derived-exact"

    tile = check(reduce(c, blob, "tile", {"level": 2, "x0": 0, "y0": 0}))["data"]
    assert (tile["h"], tile["w"]) == (50, 64) and tile["full"] == [200, 300]
    pixels = np.frombuffer(base64.b64decode(tile["data"]), dtype=np.uint8).reshape(50, 64)
    block = expected[0:4, 4:8]
    mean = block[np.isfinite(block)].mean()
    lo, hi = tile["domain"]["vs"]
    # Byte 0 is reserved for "no finite value"; 1..255 span the domain.
    assert pixels[0, 1] == 1 + round((mean - lo) / (hi - lo) * 254)

    proj = check(reduce(c, blob, "project", {"direction": {"kind": "blob", "ref": dref}}))
    values = f32(proj["data"]["data"])
    want = (expected @ direction.astype(np.float64)).astype(np.float32)
    ok = np.isfinite(want)
    assert np.array_equal(values[ok], want[ok]) and proj["data"]["ev"] == "derived-exact"


@needs_fixture
def test_reduce_reads_cut_slots_and_indexed_occurrences_without_torch(client):
    frame = check(client.get("/api/v1/s/qwen/frame", params={"addr": "t2.logits"}))["data"]
    pointer = frame["focus"]["pointers"][0]
    assert pointer["status"] == "resolvable" and pointer["spec"]["kind"] == "slot"
    stats = check(reduce(client, pointer["spec"], "stats"))["data"]
    assert stats["shape"] == [1, 64] and stats["finite"]["v"] == 64
    occurrence = check(reduce(client, {"kind": "occurrence", "address": "t2.L1.p"}, "slice",
                              {"selector": [0, 0, 0]}))["data"]
    probs = f32(occurrence["data"])
    assert occurrence["outShape"] == [3] and probs.sum() == pytest.approx(1.0, abs=1e-6)
    missing = check(reduce(client, {"kind": "occurrence", "address": "t2.L0.q"}, "stats"))
    assert missing["refused"] == "unresolved"
    assert "torch" not in sys.modules


@needs_fixture
def test_reduce_param_grid_stays_under_cap(client):
    frame = check(client.get("/api/v1/s/qwen/frame", params={"addr": "t2.L1.p"}))["data"]
    pointers = [p["spec"] for p in frame["focus"]["pointers"]]
    pointers += [{"kind": "slot", "cut": frame["focus"]["cut"] or "0" * 64, "slot": "keys",
                  "item": 1}, {"kind": "blob", "ref": "f" * 64}]
    grid = [("stats", {}), ("topk", {"k": 64}), ("topk", {"k": 65}), ("topk", {"axis": 9}),
            ("slice", {}), ("slice", {"selector": [0, 1]}), ("histogram", {"bins": 256}),
            ("histogram", {"bins": 257}), ("tile", {}), ("tile", {"level": 3}),
            ("tile", {"axes": [0, 0]}), ("project", {"direction": pointers[0]})]
    for pointer, (op, params) in itertools.product(pointers, grid):
        check(reduce(client, pointer, op, params))


@needs_fixture
def test_serving_the_fixture_never_imports_torch():
    # The subprocess gets this process's sys.path, so it runs the code under test (a scratch copy
    # of mdb ahead of the installed one, in a mutation run) and says which files it loaded.
    code = f"""
import importlib.abc, sys
sys.path[:] = {sys.path!r}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('torch', 'transformers', 'safetensors'):
            raise ImportError('serve attempted to import ' + name)
sys.meta_path.insert(0, Guard())
from fastapi.testclient import TestClient
import mdb.serve
from mdb.serve.app import create_app
print(mdb.serve.__file__)
c = TestClient(create_app({FIXTURE!r}, port=8765), base_url='http://127.0.0.1:8765')
for sid in ('tiny', 'qwen'):
    for path in ('capabilities', 'frame', 'timeline', 'branches', 'symbols', 'graph', 'records'):
        assert c.get(f'/api/v1/s/{{sid}}/{{path}}').status_code == 200, path
r = c.get('/api/v1/s/qwen/frame', params={{'addr': 't2.logits'}}).json()
spec = r['data']['focus']['pointers'][0]['spec']
body = {{'pointer': spec, 'op': 'stats', 'params': {{}}}}
assert c.post('/api/v1/s/qwen/reduce', json=body).status_code == 200
c.post('/api/v1/s/tiny/op', json={{'op': 'where', 'args': {{}}, 'request_id': 'x'}})
assert 'torch' not in sys.modules
print('ok')
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=120, env=env)
    assert result.returncode == 0, result.stderr[-3000:]
    lines = result.stdout.split()
    assert lines[-1] == "ok"
    assert Path(lines[0]).resolve().parent == Path(schema.__file__).resolve().parent


# =========================================================================== S1 Host / Origin guard

def asgi_call(app, *, method="GET", path="/api/v1/hello", headers=(), chunks=(b"",),
              kind="http"):
    """Drive an ASGI app directly: ``(sent messages, bytes the app was handed)``."""
    sent, handed, queue = [], [0], list(chunks)

    async def receive():
        if kind == "websocket":
            return {"type": "websocket.connect"}
        if queue:
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {"type": kind, "method": method, "path": path, "raw_path": path.encode(),
             "query_string": b"", "headers": [(k.lower().encode(), v.encode())
                                              for k, v in headers]}
    asyncio.run(app(scope, receive, send))
    return sent, handed[0]


def _echo_app(seen=None):
    async def app(scope, receive, send):
        if scope["type"] == "websocket":
            await receive()
            await send({"type": "websocket.accept"})
            return
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            total += len(message.get("body", b""))
            if not message.get("more_body"):
                break
        if seen is not None:
            seen.append(total)
        body = b'{"ok":1}'
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body})
    return app


def _status(sent):
    return sent[0]["status"] if sent and sent[0]["type"] == "http.response.start" else None


SECURITY_HEADERS = ("content-security-policy", "x-content-type-options", "referrer-policy",
                    "cross-origin-resource-policy")


def assert_security_headers(response):
    headers = {k.lower(): v for k, v in response.headers.items()}
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["cross-origin-resource-policy"] == "same-origin"


@needs_fixture
def test_foreign_host_is_refused_on_get_and_post(client):
    body = {"pointer": {"kind": "blob", "ref": "0" * 64}, "op": "stats", "params": {}}
    for method, url in (("GET", "/api/v1/hello"), ("GET", "/api/v1/sessions"), ("GET", "/"),
                        ("POST", "/api/v1/s/qwen/reduce")):
        kwargs = {"json": body} if method == "POST" else {}
        for host in ("evil.example", f"evil.example:{PORT}", "127.0.0.1.evil.example",
                     f"127.0.0.1:{PORT + 1}", "127.0.0.1", "localhost", "[::1]"):
            response = client.request(method, url, headers={"Host": host}, **kwargs)
            assert response.status_code == 403, (method, url, host)
            assert response.json()["refused"] == "host_not_allowed"
            schema.Refused.model_validate(response.json())
            assert_security_headers(response)


@needs_fixture
def test_foreign_origin_is_refused_even_with_a_loopback_host(client):
    body = {"pointer": {"kind": "blob", "ref": "0" * 64}, "op": "stats", "params": {}}
    for origin in ("http://evil.example", f"http://evil.example:{PORT}", "null",
                   f"https://127.0.0.1:{PORT}", f"http://127.0.0.1:{PORT + 1}",
                   f"http://127.0.0.1.evil.example:{PORT}"):
        for method, url in (("POST", "/api/v1/s/qwen/reduce"), ("GET", "/api/v1/hello")):
            kwargs = {"json": body} if method == "POST" else {}
            response = client.request(method, url, headers={"Origin": origin}, **kwargs)
            assert response.status_code == 403, (origin, method)
            assert response.json()["refused"] == "origin_not_allowed"
            assert_security_headers(response)


@needs_fixture
def test_every_allowed_host_and_origin_form_passes():
    c = make_client()
    for host in (f"127.0.0.1:{PORT}", f"localhost:{PORT}", f"[::1]:{PORT}",
                 f"LOCALHOST:{PORT}"):
        assert c.get("/api/v1/hello", headers={"Host": host}).status_code == 200, host
        origin = {"Origin": f"http://{host}"}
        assert c.get("/api/v1/hello", headers={"Host": host, **origin}).status_code == 200
    # extra hosts: HOST means HOST:<bound port>; HOST:PORT is exact
    c = make_client(allow_hosts=["mdb.lan", "tunnel.example:9000"])
    assert c.get("/api/v1/hello", headers={"Host": f"mdb.lan:{PORT}"}).status_code == 200
    assert c.get("/api/v1/hello", headers={"Host": "tunnel.example:9000"}).status_code == 200
    assert c.get("/api/v1/hello", headers={"Host": f"tunnel.example:{PORT}"}).status_code == 403
    assert c.get("/api/v1/hello", headers={"Host": f"mdb.lan:{PORT}",
                                           "Origin": f"http://mdb.lan:{PORT}"}).status_code == 200
    assert c.get("/api/v1/hello", headers={"Host": "other.lan"}).status_code == 403


def test_bare_host_forms_are_allowed_only_on_port_80():
    app80 = SecurityLayer(_echo_app(), port=80)
    app81 = SecurityLayer(_echo_app(), port=81)
    for app, host, want in ((app80, "127.0.0.1", 200), (app80, "localhost", 200),
                            (app80, "[::1]:80", 200), (app81, "127.0.0.1", 403),
                            (app81, "127.0.0.1:81", 200), (app81, "127.0.0.1:80", 403)):
        sent, _ = asgi_call(app, headers=[("host", host)])
        assert _status(sent) == want, (host, want)


def test_a_request_without_a_host_header_is_a_400_refusal():
    sent, _ = asgi_call(SecurityLayer(_echo_app(), port=PORT), headers=[])
    assert _status(sent) == 400
    body = json.loads(sent[1]["body"])
    assert body["refused"] == "missing_host"
    schema.Refused.model_validate(body)


@needs_fixture
def test_websocket_with_a_foreign_host_or_origin_is_closed_before_accept():
    from starlette.websockets import WebSocketDisconnect

    c = make_client()
    url = f"ws://127.0.0.1:{PORT}/api/v1/s/tiny/events"
    for headers in ({"host": "evil.example"}, {"host": f"evil.example:{PORT}"},
                    {"origin": "http://evil.example"}, {"origin": f"https://127.0.0.1:{PORT}"}):
        with pytest.raises(WebSocketDisconnect) as closed:
            with c.websocket_connect(url, headers=headers):
                pass
        assert closed.value.code == 1008, headers
    with c.websocket_connect(url, headers={"origin": f"http://127.0.0.1:{PORT}"}) as ws:
        assert json.loads(ws.receive_text())["data"]["type"] == "hello"


def test_a_websocket_is_closed_without_an_accept_message():
    sent, _ = asgi_call(SecurityLayer(_echo_app(), port=PORT), kind="websocket", path="/ws",
                        headers=[("host", "evil.example")])
    assert [m["type"] for m in sent] == ["websocket.close"]


def test_allow_remote_without_allow_host_exits_2(tmp_path, capsys, monkeypatch):
    from mdb.serve.__main__ import main

    assert main(["--project", str(tmp_path), "--host", "0.0.0.0", "--allow-remote"]) == 2
    assert "--allow-host" in capsys.readouterr().err
    assert main(["--project", str(tmp_path), "--allow-remote"]) == 2
    assert main(["--project", str(tmp_path), "--port", "0"]) == 2


@needs_fixture
def test_allow_remote_with_allow_host_binds_and_warns(capsys, monkeypatch):
    import uvicorn

    from mdb.serve.__main__ import main

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    assert main(["--project", str(FIXTURE), "--host", "0.0.0.0", "--port", str(PORT),
                 "--allow-remote", "--allow-host", "mdb.lan"]) == 0
    assert "without authentication" in capsys.readouterr().err
    app = seen["app"]
    c = TestClient(app, base_url="http://mdb.lan:8765")
    assert c.get("/api/v1/hello").status_code == 200
    assert TestClient(app, base_url="http://evil.example:8765").get("/api/v1/hello"
                                                                  ).status_code == 403


# =========================================================================== S2 limits and cache

def test_request_body_over_the_limit_is_refused_with_and_without_content_length():
    limit = 1000
    app = SecurityLayer(_echo_app(), port=PORT, body_limit=limit)
    host = [("host", f"127.0.0.1:{PORT}")]
    sent, _ = asgi_call(app, method="POST", headers=[*host, ("content-length", str(limit + 1))],
                        chunks=[b"x" * (limit + 1)])
    assert _status(sent) == 413
    body = json.loads(sent[1]["body"])
    assert body["refused"] == "over_limit" and body["cap"] == limit
    # no Content-Length (chunked), and a lying Content-Length, are counted as streamed
    for headers in (host, [*host, ("content-length", "5")]):
        sent, _ = asgi_call(app, method="POST", headers=headers,
                            chunks=[b"y" * 400, b"y" * 400, b"y" * 201])
        assert _status(sent) == 413, headers
        assert json.loads(sent[1]["body"])["refused"] == "over_limit"
    seen = []
    sent, _ = asgi_call(SecurityLayer(_echo_app(seen), port=PORT, body_limit=limit),
                        method="POST", headers=host, chunks=[b"z" * 500, b"z" * 500])
    assert _status(sent) == 200 and seen == [1000]  # exactly the limit is fine


def test_nothing_past_the_limit_is_handed_to_the_app():
    limit, seen = 64 * 1024, []
    app = SecurityLayer(_echo_app(seen), port=PORT, body_limit=limit)
    chunks = [b"q" * 16384] * 64  # 1 MiB streamed
    sent, _ = asgi_call(app, method="POST", headers=[("host", f"127.0.0.1:{PORT}")],
                        chunks=chunks)
    assert _status(sent) == 413
    assert sum(seen) <= limit  # the app never saw a byte past the limit (nor an unbounded buffer)


@needs_fixture
def test_oversized_post_body_is_413_over_the_real_app():
    c = make_client()
    pad = "x" * (300 * 1024)
    body = json.dumps({"pointer": {"kind": "blob", "ref": "0" * 64}, "op": "stats",
                       "params": {"pad": pad}}).encode()
    response = c.post("/api/v1/s/qwen/reduce", content=body,
                      headers={"content-type": "application/json"})
    assert response.status_code == 413 and response.json()["refused"] == "over_limit"
    assert_security_headers(response)

    def stream():  # chunked: no Content-Length at all
        for i in range(0, len(body), 8192):
            yield body[i:i + 8192]

    response = c.post("/api/v1/s/qwen/reduce", content=stream(),
                      headers={"content-type": "application/json"})
    assert response.status_code == 413 and response.json()["refused"] == "over_limit"


def _rss_kb():
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True)
    return int(out.stdout.strip())


@needs_fixture
def test_rss_does_not_grow_across_oversized_and_padded_requests(store_copy):
    c = make_client(store_copy)
    store = _store(store_copy)
    ref = store.put_tensor(np.arange(12, dtype=np.float32).reshape(3, 4))["ref"]
    pointer = {"kind": "blob", "ref": ref}

    def go(i):
        over = json.dumps({"pointer": pointer, "op": "stats",
                           "params": {"pad": chr(97 + i) * (1 << 20)}}).encode()
        assert c.post("/api/v1/s/qwen/reduce", content=over,
                      headers={"content-type": "application/json"}).status_code == 413
        padded = {"pointer": pointer, "op": "stats", "params": {"pad": chr(65 + i) * 200000}}
        assert c.post("/api/v1/s/qwen/reduce", json=padded).status_code == 422

    go(0)  # warm up allocators and imports
    before = _rss_kb()
    for i in range(8):
        go(i)
    grown_mb = (_rss_kb() - before) / 1024
    assert grown_mb < 24, f"rss grew {grown_mb:.1f} MB over 16 oversized/padded requests"


@needs_fixture
def test_unknown_parameters_are_422_bad_params_for_every_op(client):
    pointer = {"kind": "blob", "ref": "0" * 64}
    cases = {"stats": {"k": 1}, "topk": {"kk": 3, "pad": 1}, "slice": {"selector": [], "x": 1},
             "histogram": {"bin": 4}, "tile": {"zoom": 2}, "project": {"direction": pointer,
                                                                        "extra": 1}}
    for op, params in cases.items():
        response = client.post("/api/v1/s/qwen/reduce", json={"pointer": pointer, "op": op,
                                                              "params": params})
        body = check(response)
        assert response.status_code == 422 and body["refused"] == "bad_params", op
        unknown = [k for k in params if k not in __import__("mdb.serve.reduce",
                                                            fromlist=["PARAMS"]).PARAMS[op]]
        assert all(repr(k) in body["detail"] for k in unknown), (op, body["detail"])


def test_cache_key_is_a_digest_of_the_known_params():
    from mdb.serve.reduce import params_digest

    assert params_digest({"a": 1, "b": [2]}) == params_digest({"b": [2], "a": 1})
    assert params_digest({"a": 1}) != params_digest({"a": 2})
    assert len(params_digest({"pad": "x" * 100000})) == 64


@needs_fixture
def test_reduce_cache_holds_digests_and_stays_within_its_byte_budget(store_copy):
    from mdb.serve.recorded import Recorded
    from mdb.serve.reduce import Reducer

    ref = _store(store_copy).put_tensor(
        np.random.default_rng(1).standard_normal((50, 60)).astype(np.float32))["ref"]
    model = Recorded(store_copy).model("qwen")
    reducer = Reducer(size=1000, max_bytes=20000)
    pointer = {"kind": "blob", "ref": ref}
    for k in range(1, 65):
        reducer.run(model, pointer, "topk", {"k": k})
    for bins in range(1, 120):
        reducer.run(model, pointer, "histogram", {"bins": bins})
    assert 0 < reducer.bytes <= 20000
    assert len(reducer._cache) < 183  # older entries were evicted by bytes, not only by count
    assert all(len(str(part)) <= 64 for key in reducer._cache for part in key)
    big = reducer.run(model, pointer, "slice", {"selector": [[0, 40], [0, 60]]})
    assert big.op == "slice" and reducer.bytes <= 20000  # larger than a quarter budget: uncached
    app_cache = create_app(store_copy, port=PORT).state.reducer
    assert app_cache.max_bytes == 16 * 1024 * 1024


def test_topk_equals_the_full_stable_sort_on_ties_and_nan():
    from mdb.serve.reduce import _top_indices, topk

    rng = np.random.default_rng(3)
    for trial in range(60):
        n = int(rng.integers(1, 400))
        values = rng.integers(-5, 6, size=n).astype(np.float64)  # many ties
        values[rng.random(n) < 0.1] = np.nan
        values[rng.random(n) < 0.05] = np.inf
        score = np.abs(values)
        score[~np.isfinite(score)] = -np.inf
        for k in (1, 2, 7, 64, 500):
            want = np.argsort(-score, kind="stable")[:k]
            assert _top_indices(score, k).tolist()[:k] == want.tolist(), (trial, n, k)

    base = {"op": "topk", "pointer": "p", "shape": [4, 4], "dtype": "float32", "note": None}
    grid = np.array([[3, 3, 3, 3], [3, -3, np.nan, 1], [3, 0, 0, 0], [np.inf, 3, 2, 2]])
    result = topk(grid, {"k": 6}, base)
    flat = np.abs(grid).reshape(-1)
    flat[~np.isfinite(flat)] = -np.inf
    want = np.argsort(-flat, kind="stable")[:6]
    assert [e.index for e in result.entries] == [
        [int(j) for j in np.unravel_index(i, grid.shape)] for i in want]


# =========================================================================== S4 no 500s, headers

def _store(root):
    from saturn_pub.store import LocalStore

    return LocalStore(Path(root) / "store")


def _blob(root, array):
    return {"kind": "blob", "ref": _store(root).put_tensor(array)["ref"]}


@needs_fixture
def test_former_500_inputs_are_422_bad_params(store_copy):
    c = make_client(store_copy)
    scalar = _blob(store_copy, np.asarray(3.25, dtype=np.float32))
    matrix = _blob(store_copy, np.arange(12, dtype=np.float32).reshape(3, 4))
    raw = ('{"pointer":{"kind":"blob","ref":"%s"},"op":"histogram","params":{"range":%s}}')
    cases = [
        (scalar, "topk", {"axis": 0}),
        (matrix, "histogram", {"bins": 64, "range": [-1e308, 1e308]}),
        (matrix, "histogram", {"bins": 8, "range": ["a", 1]}),
        (matrix, "histogram", {"bins": 8, "range": [None, 1]}),
        (matrix, "histogram", {"bins": 8, "range": [True, 2]}),
        (matrix, "histogram", {"bins": 8, "range": [1, 1]}),
        (matrix, "histogram", {"bins": 8, "range": [2, 1]}),
        (matrix, "histogram", {"bins": 8, "range": [1]}),
        (matrix, "histogram", {"bins": 8, "range": [0, 10 ** 400]}),
        (matrix, "topk", {"axis": 2}),
        (matrix, "topk", {"axis": -3}),
    ]
    for pointer, op, params in cases:
        response = c.post("/api/v1/s/qwen/reduce", json={"pointer": pointer, "op": op,
                                                         "params": params})
        body = check(response)
        assert response.status_code == 422 and body["refused"] == "bad_params", (op, params)
    for literal in ("[-Infinity, Infinity]", "[NaN, 1]", "[0, Infinity]"):
        response = c.post("/api/v1/s/qwen/reduce", content=raw % (matrix["ref"], literal),
                          headers={"content-type": "application/json"})
        assert check(response)["refused"] == "bad_params", literal
    # the valid neighbours still work: axis bounds are [-ndim, ndim-1]
    for axis in (-2, -1, 0, 1):
        assert check(c.post("/api/v1/s/qwen/reduce", json={
            "pointer": matrix, "op": "topk", "params": {"axis": axis}}), "reduce")["data"]["axis"] \
            == axis
    hist = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": matrix, "op": "histogram", "params": {"bins": 4, "range": [0, 12]}}))
    assert hist["data"]["counts"] == [3, 3, 3, 3]


@needs_fixture
def test_numpy_value_errors_never_surface_as_500(store_copy, monkeypatch):
    import mdb.serve.reduce as reduce_module

    c = make_client(store_copy)
    matrix = _blob(store_copy, np.arange(12, dtype=np.float32).reshape(3, 4))

    def boom(*args, **kwargs):
        raise ValueError("numpy refused this")

    monkeypatch.setattr(reduce_module, "histogram", boom)
    response = c.post("/api/v1/s/qwen/reduce", json={"pointer": matrix, "op": "histogram",
                                                     "params": {}})
    body = check(response)
    assert response.status_code == 422 and body["refused"] == "bad_params"


@needs_fixture
def test_empty_arrays_are_resolved_results_with_unresolved_stats(store_copy):
    c = make_client(store_copy)
    for shape in ((0,), (0, 5), (4, 0), (0, 3, 3)):
        pointer = _blob(store_copy, np.zeros(shape, dtype=np.float32))
        stats = check(c.post("/api/v1/s/qwen/reduce", json={
            "pointer": pointer, "op": "stats", "params": {}}), "reduce")["data"]
        assert stats["count"]["v"] == 0 and stats["finite"]["v"] == 0
        for key in ("min", "max", "mean", "std", "absmax", "l2"):
            assert stats[key]["ev"] == "unresolved" and stats[key]["v"] is None
            assert stats[key]["reason"]
        for op, params in (("topk", {}), ("topk", {"axis": 0}), ("slice", {}),
                           ("histogram", {}), ("tile", {}), ("tile", {"level": 3})):
            response = c.post("/api/v1/s/qwen/reduce", json={"pointer": pointer, "op": op,
                                                             "params": params})
            body = check(response, "reduce")
            assert response.status_code == 200, (shape, op, params, body)
        top = check(c.post("/api/v1/s/qwen/reduce", json={
            "pointer": pointer, "op": "topk", "params": {}}))["data"]
        assert top["entries"] == []
        hist = check(c.post("/api/v1/s/qwen/reduce", json={
            "pointer": pointer, "op": "histogram", "params": {"bins": 4}}))["data"]
        assert hist["counts"] == [0, 0, 0, 0]
        tile = check(c.post("/api/v1/s/qwen/reduce", json={
            "pointer": pointer, "op": "tile", "params": {}}))["data"]
        assert tile["w"] == 0 and tile["h"] == 0 and tile["data"] == ""


@needs_fixture
def test_extreme_magnitudes_are_unresolved_not_a_crash(store_copy):
    c = make_client(store_copy)
    huge = _blob(store_copy, np.array([[1e308, -1e308], [1e308, 5.0]], dtype=np.float64))
    stats = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": huge, "op": "stats", "params": {}}), "reduce")["data"]
    assert stats["max"]["ev"] == "measured"
    assert stats["l2"]["ev"] == "measured"  # scaled: no spurious overflow
    assert stats["std"]["ev"] in ("measured", "unresolved")
    tile = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": huge, "op": "tile", "params": {}}), "reduce")["data"]
    pixels = np.frombuffer(base64.b64decode(tile["data"]), dtype=np.uint8)
    assert pixels.min() >= 1 and tile["scale"] == "diverging"


@needs_fixture
def test_deep_json_and_wrong_method_use_the_refused_envelope(client):
    deep = b'{"a":' * 40000 + b"1" + b"}" * 40000  # < 256 KiB, deep enough to exhaust the parser
    response = client.post("/api/v1/s/qwen/reduce", content=deep,
                           headers={"content-type": "application/json"})
    body = check(response)
    assert response.status_code in (400, 422) and "refused" in body
    assert_security_headers(response)
    for method, url in (("OPTIONS", "/api/v1/s/qwen/reduce"), ("PUT", "/api/v1/hello"),
                        ("DELETE", "/api/v1/s/tiny/frame"), ("POST", "/api/v1/hello")):
        response = client.request(method, url)
        body = check(response)
        assert response.status_code == 405 and body["refused"] == "method_not_allowed"
        assert_security_headers(response)


@needs_fixture
def test_security_headers_are_on_404_422_413_and_the_500_without_a_leak(store_copy):
    c = make_client(store_copy, raise_server_exceptions=False)
    assert_security_headers(c.get("/api/v1/s/missing/frame"))
    assert c.get("/api/v1/s/missing/frame").status_code == 404
    unprocessable = c.get("/api/v1/s/tiny/timeline", params={"x0": "a"})
    assert unprocessable.status_code == 422
    assert_security_headers(unprocessable)
    assert_security_headers(c.get("/api/nope"))
    assert_security_headers(c.get("/"))

    secret = "SECRET-PATH /Users/someone/private/file.py line 12"

    def boom(sid):
        raise RuntimeError(secret)

    create = c.app.app.state.recorded  # the SecurityLayer wraps the FastAPI app
    create.model = boom
    response = c.get("/api/v1/s/tiny/capabilities")
    assert response.status_code == 500
    assert response.json() == {"detail": "the server failed on this request; see its log",
                               "refused": "internal_error", "state": "refused"}
    assert secret not in response.text and "Traceback" not in response.text
    assert_security_headers(response)
    assert response.headers["cache-control"] == "no-store"


# =========================================================================== S3 packaging

def test_serve_extra_names_numpy_and_the_wheel_ships_the_web_build():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    extra = config["project"]["optional-dependencies"]["serve"]
    assert any(spec.replace(" ", "").startswith("numpy>=2") for spec in extra), extra
    assert "src/mdb/serve/static/**" in config["tool"]["hatch"]["build"]["targets"]["wheel"][
        "artifacts"]


@needs_fixture
def test_starting_without_web_assets_prints_one_line_and_still_serves(tmp_path, capsys,
                                                                       monkeypatch):
    import uvicorn

    import mdb.serve.app as app_module
    from mdb.serve.__main__ import main

    monkeypatch.setattr(app_module, "STATIC", tmp_path / "absent")
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app))
    assert main(["--project", str(FIXTURE), "--port", str(PORT)]) == 0
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "web build absent" in line]
    assert len(lines) == 1 and "npm run build" in lines[0] and "packages/mdb/web" in lines[0]
    c = TestClient(seen["app"], base_url=BASE)
    assert c.get("/api/v1/hello").status_code == 200
    index = c.get("/")
    assert index.status_code == 404 and index.json()["refused"] == "web_build_absent"


# =========================================================================== S5 honest layer count

def layers_client(monkeypatch, *, receipts=None, drop_last_index=False):
    """A client over the fixture whose cut surface lacks the layer clock's ``maximum``.

    ``receipts``: sessions whose receipts are dropped (no layer-phase points: count unknown).
    ``drop_last_index``: tiny loses every cut/transition at layer index 2 (a recording that
    stopped mid-depth: the highest recorded index is a lower bound, not the model's count)."""
    import copy

    from mdb.serve import _legacy

    real_desc, real_receipts, real_ids = _legacy.cut_descriptor, _legacy.receipts, _legacy.cut_ids

    def desc(project, sid, cut):
        value = copy.deepcopy(real_desc(project, sid, cut))
        for slot in value["surface"].get("slots", []):
            if slot["name"] == "layer":
                slot["schema"].pop("maximum", None)
        return value

    def rec(project, sid):
        rows = real_receipts(project, sid)
        if sid in (receipts or ()):
            return []
        if drop_last_index and sid == "tiny":
            rows = copy.deepcopy(rows)
            for row in rows:
                row["transitions"] = [t for t in row.get("transitions") or []
                                      if (t.get("after") or {}).get("index") != 2
                                      and (t.get("before") or {}).get("index") != 2]
        return rows

    def ids(project, sid):
        found = real_ids(project, sid)
        if drop_last_index and sid == "tiny":
            return [i for i in found if real_desc(project, sid, i)["execution_point"].get(
                "index") != 2]
        return found

    monkeypatch.setattr(_legacy, "cut_descriptor", desc)
    monkeypatch.setattr(_legacy, "receipts", rec)
    monkeypatch.setattr(_legacy, "cut_ids", ids)
    return make_client()


def _depth(c, sid):
    caps = check(c.get(f"/api/v1/s/{sid}/capabilities"), "capabilities")["data"]
    return caps, caps["axes"][1]


def _frame_text(c, sid, addr):
    response = c.get(f"/api/v1/s/{sid}/frame", params={"addr": addr})
    check(response, "frame")
    return response


@needs_fixture
def test_a_known_layer_count_is_measured_and_stated(client):
    caps, depth = _depth(client, "qwen")
    assert caps["layers"]["ev"] == "measured" and caps["layers"]["v"] == 2
    assert depth["note"].startswith("2 decoder layers")
    labels = [t["label"] for t in depth["ticks"]]
    assert "L0" in labels and "L1" in labels and "L1 out" in labels


@needs_fixture
def test_unknown_layer_count_is_unresolved_and_draws_no_layer_tick(monkeypatch):
    c = layers_client(monkeypatch, receipts={"qwen"})
    caps, depth = _depth(c, "qwen")
    assert caps["layers"]["ev"] == "unresolved" and caps["layers"]["v"] is None
    assert "layer count unknown" in caps["layers"]["reason"]
    assert "layer count unknown" in depth["note"] and "0 decoder layers" not in depth["note"]
    assert not [t for t in depth["ticks"] if t["port"] in ("h", "h_out")], depth["ticks"]
    assert not [t for t in depth["ticks"] if t["label"].startswith("L")]
    for addr in ("t2.L0.h", "t2.L1.h", "t2.L0.h_out", "t2.L1.h_out"):
        response = _frame_text(c, "qwen", addr)
        data = response.json()["data"]
        assert data["key"] is None and data["pos"]["layer"] is None, addr
        pointer = data["focus"]["pointers"][0]
        assert pointer["status"] == "frontier" and "layer count unknown" in pointer["reason"]
        assert "does not exist" not in response.text and "the model has" not in response.text
    timeline = check(c.get("/api/v1/s/qwen/timeline"), "timeline")["data"]
    assert all(cell["y"] < len(depth["ticks"]) for cell in timeline["cells"])
    unresolved = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "occurrence", "address": "t2.L0.h"}, "op": "stats", "params": {}}))
    assert unresolved["refused"] == "unresolved"


@needs_fixture
def test_an_inferred_layer_count_is_a_lower_bound_and_never_a_fact(monkeypatch):
    c = layers_client(monkeypatch)
    caps, depth = _depth(c, "tiny")
    assert caps["layers"]["ev"] == "derived-local" and caps["layers"]["v"] == 2
    assert caps["layers"]["unit"] == "layers (at least)"
    assert depth["note"].startswith("at least 2 recorded layers (inferred")
    assert "decoder layers" not in depth["note"]
    for addr in ("t2.L0.h", "t2.L1.h", "t2.L0.h_out", "t2.L1.h_out", "t2.emb", "t2.L7.h"):
        response = _frame_text(c, "tiny", addr)
        assert response.json()["data"]["key"] is None, addr  # aliasing needs the real count
        assert "does not exist" not in response.text and "the model has" not in response.text


@needs_fixture
def test_a_recording_that_stopped_mid_depth_says_not_recorded_not_nonexistent(monkeypatch):
    c = layers_client(monkeypatch, drop_last_index=True)
    caps, depth = _depth(c, "tiny")
    assert caps["layers"]["ev"] == "derived-local" and caps["layers"]["v"] == 1
    assert depth["note"].startswith("at least 1 recorded layers (inferred")
    assert "L1" not in [t["label"] for t in depth["ticks"]]  # no tick for a layer never seen
    response = _frame_text(c, "tiny", "t2.L1.h")  # index 1 was recorded: found, not "missing"
    assert "does not exist" not in response.text and "the model has" not in response.text
    for addr in ("t2.L1.h_out", "t2.L5.h"):
        response = _frame_text(c, "tiny", addr)
        data = response.json()["data"]
        assert data["focus"]["note"] == "not recorded; model layer count unknown", addr
        assert "does not exist" not in response.text and "the model has" not in response.text
        assert data["focus"]["pointers"] == [] and data["focus"]["cut"] is None


# =========================================================================== S6 read-only for real

def _make_wal_copy(source, dest):
    """A WAL-mode copy of a project: its database plus the -wal (and -shm) of a still-open
    writer, as an externally converted project that was copied while in use looks."""
    import sqlite3

    shutil.copytree(source, dest)
    keeper = sqlite3.connect(Path(dest) / "project.sqlite")
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("CREATE TABLE IF NOT EXISTS wal_marker (x)")
    keeper.execute("INSERT INTO wal_marker VALUES (1)")
    keeper.commit()
    final = Path(str(dest) + "-final")
    shutil.copytree(dest, final, ignore=shutil.ignore_patterns("*-shm"))
    keeper.close()
    return final


def _crawl(c):
    for url, params in gets(c):
        check(c.get(url, params=params))
    for sid in ("tiny", "qwen"):
        frame = c.get(f"/api/v1/s/{sid}/frame").json()["data"]
        for pointer in frame["focus"]["pointers"]:
            if pointer["spec"]:
                check(c.post(f"/api/v1/s/{sid}/reduce", json={
                    "pointer": pointer["spec"], "op": "stats", "params": {}}))


@needs_fixture
def test_a_rollback_journal_project_is_untouched_including_directories(store_copy):
    before = _tree_digest(store_copy)
    c = make_client(store_copy)
    _crawl(c)
    del c
    after = _tree_digest(store_copy)
    assert after == before
    assert not [k for k in after if k.endswith(("-wal", "-shm", "-journal"))]


@needs_fixture
def test_a_project_in_a_read_only_tree_is_served_and_untouched(store_copy):
    paths = [store_copy, *store_copy.rglob("*")]
    try:
        for path in paths:
            path.chmod(0o555 if path.is_dir() else 0o444)
        before = _tree_digest(store_copy)
        c = make_client(store_copy)
        _crawl(c)
        del c
        assert _tree_digest(store_copy) == before
    finally:
        for path in paths:
            path.chmod(0o755 if path.is_dir() else 0o644)


@needs_fixture
def test_a_wal_mode_project_never_gets_its_database_or_wal_rewritten(tmp_path):
    wal = _make_wal_copy(Path(FIXTURE), tmp_path / "writer")
    assert (wal / "project.sqlite-wal").is_file()
    before = _tree_digest(wal)
    c = make_client(wal)
    _crawl(c)
    del c
    after = _tree_digest(wal)
    # SQLite's wal-index (-shm) is a derived cache it must create to read a WAL database; the
    # content-bearing files (database, -wal, store) stay byte-identical and nothing else appears.
    shm = {k for k in after if k.endswith("-shm")}
    assert {k: v for k, v in after.items() if k not in shm} == before
    assert shm <= {"project.sqlite-shm"}


@needs_fixture
def test_a_wal_project_in_a_read_only_directory_fails_at_startup_with_one_message(tmp_path,
                                                                                 capsys):
    from mdb.serve.__main__ import main

    wal = _make_wal_copy(Path(FIXTURE), tmp_path / "writer")
    wal.chmod(0o555)
    try:
        with pytest.raises(ValueError, match="WAL-mode project.*not writable"):
            create_app(wal, port=PORT)
        assert main(["--project", str(wal), "--port", str(PORT)]) == 1
        err = capsys.readouterr().err.strip().splitlines()
        assert len([line for line in err if "not writable" in line]) == 1
    finally:
        wal.chmod(0o755)


@needs_fixture
def test_the_connection_cannot_write(store_copy):
    import sqlite3

    from mdb.serve import _legacy

    project = _legacy.open_project(store_copy)
    with pytest.raises(sqlite3.OperationalError):
        with project._db() as db:
            db.execute("CREATE TABLE sneaky (x)")


# =========================================================================== S7 blob reads

def _sparse_blob(root, ref, shape, file_size, dtype="F32"):
    header = json.dumps({"tensor": {"dtype": dtype, "shape": shape, "data_offsets": [0, 4]}},
                        separators=(",", ":")).encode()
    path = Path(root) / "store" / "tensors" / f"{ref}.safetensors"
    with path.open("wb") as handle:
        handle.write(len(header).to_bytes(8, "little") + header + b"\0" * 4)
        handle.truncate(file_size)  # sparse: no gigabytes are written
    return path


@needs_fixture
def test_an_over_limit_blob_is_refused_from_its_header_before_the_payload_is_read(store_copy):
    from mdb.serve import blobs

    ref = "ab" * 32
    path = _sparse_blob(store_copy, ref, [1 << 12, 1 << 20], 6 << 30)
    assert path.stat().st_size == 6 << 30
    started = time.perf_counter()
    with pytest.raises(blobs.BlobError) as refusal:
        blobs.tensor_blob(Path(store_copy) / "store", ref)
    assert time.perf_counter() - started < 0.2
    assert refusal.value.limit and "elements" in refusal.value.reason
    c = make_client(store_copy)
    started = time.perf_counter()
    response = c.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "blob", "ref": ref}, "op": "stats", "params": {}})
    elapsed = time.perf_counter() - started
    body = check(response)
    assert response.status_code == 422 and body["refused"] == "over_limit"
    assert elapsed < 0.2, elapsed
    # an honest file over the byte limit (header looks fine) is also refused unread
    small = "cd" * 32
    _sparse_blob(store_copy, small, [2], 3 << 30)
    started = time.perf_counter()
    body = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "blob", "ref": small}, "op": "stats", "params": {}}))
    assert body["refused"] == "over_limit" and time.perf_counter() - started < 0.2


@needs_fixture
def test_verification_failures_stay_verification_failed(store_copy):
    c = make_client(store_copy)
    ref = _blob(store_copy, np.arange(6, dtype=np.float32))["ref"]
    path = Path(store_copy) / "store" / "tensors" / f"{ref}.safetensors"
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    # same payload and metadata, one extra header byte: the file digest no longer matches its ref
    padded = (size + 1).to_bytes(8, "little") + raw[8:8 + size] + b" " + raw[8 + size:]
    path.write_bytes(padded)
    body = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "blob", "ref": ref}, "op": "stats", "params": {}}))
    assert body["refused"] == "verification_failed" and "digest mismatch" in body["detail"]
    ref2 = _blob(store_copy, np.arange(6, dtype=np.float32) + 1)["ref"]
    path2 = Path(store_copy) / "store" / "tensors" / f"{ref2}.safetensors"
    flipped = bytearray(path2.read_bytes())
    flipped[-1] ^= 0xFF  # tampered payload byte
    path2.write_bytes(bytes(flipped))
    body = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "blob", "ref": ref2}, "op": "stats", "params": {}}))
    assert body["refused"] == "verification_failed"


@needs_fixture
def test_slot_info_on_a_missing_blob_is_unresolved_not_an_exception(store_copy):
    from mdb.serve import blobs

    c = make_client(store_copy)
    pointer = check(c.get("/api/v1/s/qwen/frame", params={"addr": "t2.logits"}),
                    "frame")["data"]["focus"]["pointers"][0]
    spec = pointer["spec"]
    store = Path(store_copy) / "store"
    assert blobs.slot_info(store, spec["cut"], spec["slot"])["shape"] == [1, 64]
    _, page = blobs._slot_page(store, spec["cut"], spec["slot"])
    (store / "blobs" / f"{page['tensor_blob']}.safetensors").unlink()
    with pytest.raises(blobs.BlobError) as missing:
        blobs.slot_info(store, spec["cut"], spec["slot"])
    assert not missing.value.refused and "not in the store" in missing.value.reason
    with pytest.raises(blobs.BlobError):
        blobs.slot_info(store, "0" * 64, "logits")  # a missing cut is the same kind of failure
    again = make_client(store_copy)
    frame = check(again.get("/api/v1/s/qwen/frame", params={"addr": "t2.logits"}),
                  "frame")["data"]
    assert frame["focus"]["pointers"][0]["status"] == "frontier"
    reduced = check(again.post("/api/v1/s/qwen/reduce", json={
        "pointer": spec, "op": "stats", "params": {}}))
    assert reduced["refused"] == "unresolved"
    for path in ("capabilities", "timeline", "symbols", "records", "graph"):
        check(again.get(f"/api/v1/s/qwen/{path}"))


# =========================================================================== S8 mutation gaps

@needs_fixture
def test_static_serving_never_reads_outside_the_static_directory(tmp_path):
    static = tmp_path / "web" / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>shell</title>")
    (static / "assets" / "app.js").write_text("export {}")
    (tmp_path / "web" / "secret.txt").write_text("TOP-SECRET-OUTSIDE")
    (tmp_path / "web" / "app.py").write_text("TOP-SECRET-SOURCE")
    outside = tmp_path / "outside.txt"
    outside.write_text("TOP-SECRET-LINKED")
    try:
        (static / "assets" / "link.txt").symlink_to(outside)
    except OSError:  # pragma: no cover - platforms without symlinks
        pass
    c = make_client(static_dir=str(static))
    assert c.get("/assets/app.js").text == "export {}"
    for target in ("/..%2fsecret.txt", "/..%2fapp.py", "/%2e%2e%2fsecret.txt",
                   "/%2e%2e/secret.txt", "/../secret.txt", "/assets/..%2f..%2fsecret.txt",
                   "/assets/..%2f..%2fapp.py", "/assets/%2e%2e/%2e%2e/secret.txt",
                   "/assets/..%2f..%2f..%2foutside.txt", "/..%5csecret.txt", "//secret.txt",
                   "/assets/link.txt", "/secret.txt%00", "/assets/app.js%00.png"):
        response = c.get(target)
        assert "TOP-SECRET" not in response.text, target
        assert response.status_code in (200, 404), (target, response.status_code)
        if response.status_code == 200:
            assert response.text.startswith("<!doctype html>"), target  # only ever the shell


@needs_fixture
def test_tile_byte_zero_means_no_finite_value_and_values_start_at_one(store_copy):
    c = make_client(store_copy)
    grid = np.arange(64 * 64, dtype=np.float32).reshape(64, 64) - 2000.0
    grid[8:16, 24:32] = np.nan  # an all-NaN block at level 3
    pointer = _blob(store_copy, grid)
    tile = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": pointer, "op": "tile", "params": {"level": 3}}), "reduce")["data"]
    pixels = np.frombuffer(base64.b64decode(tile["data"]), dtype=np.uint8).reshape(8, 8)
    assert pixels[1, 3] == 0 and tile["nonfinite"] == 1
    finite = pixels[np.isfinite(grid.reshape(8, 8, 8, 8).mean(axis=(1, 3)))]
    assert (pixels != 0).sum() == 63 and pixels[pixels != 0].min() >= 1
    assert finite.size == 63 and tile["scale"] == "diverging"
    flat = _blob(store_copy, np.linspace(5, 9, 64, dtype=np.float32).reshape(8, 8))
    tile = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": flat, "op": "tile", "params": {}}), "reduce")["data"]
    pixels = np.frombuffer(base64.b64decode(tile["data"]), dtype=np.uint8)
    assert pixels.min() == 1 and pixels.max() == 255  # the finite minimum is byte 1, never 0


@needs_fixture
def test_comparison_arms_stored_in_reverse_mask_order_still_list_sham_first():
    from types import SimpleNamespace

    from mdb.serve.recorded import Recorded
    from mdb.serve.views.records import _arms

    model = Recorded(FIXTURE).model("tiny")
    records = model.records()
    comparison = next(r for r in records if r.get("kind") == "comparison")
    arms = [r for r in records if r.get("kind") == "comparison-arm"]
    assert len(arms) >= 4
    reverse = sorted(records, key=lambda r: -r.get("mask", 0))
    reversed_model = SimpleNamespace(records=lambda: reverse)
    flipped = {**comparison, "arms": list(reversed(comparison["arms"]))}
    rows = _arms(reversed_model, flipped)
    assert [r.mask for r in rows] == [0, 1, 2, 3]
    assert rows[0].label.startswith("sham") and rows[1].effect.ev == "derived-exact"


@needs_fixture
def test_tampered_blob_bytes_are_refused(store_copy):
    c = make_client(store_copy)
    frame = check(c.get("/api/v1/s/qwen/frame", params={"addr": "t2.L1.p"}), "frame")["data"]
    spec = next(p["spec"] for p in frame["focus"]["pointers"] if p["spec"]["kind"] == "blob")
    assert check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": spec, "op": "stats", "params": {}}), "reduce")
    path = Path(store_copy) / "store" / "tensors" / f"{spec['ref']}.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 0x01
    path.write_bytes(bytes(raw))
    for params in ({}, {"selector": [0, 0, 0]}):
        op = "stats" if not params else "slice"
        body = check(make_client(store_copy).post("/api/v1/s/qwen/reduce", json={
            "pointer": spec, "op": op, "params": params}))
        assert body["refused"] == "verification_failed"


# =========================================================================== S9 structural numbers

def test_a_structural_key_exempts_only_its_own_int_or_int_list():
    assert unlabelled_numbers({"pos": {"token": 2, "layer": 3}, "shape": [2, 3], "k": 4}) == []
    assert unlabelled_numbers({"kinds": {"a": 1, "b": 2}}) == []
    bare = unlabelled_numbers
    assert bare({"rows": [{"mass": 0.5}]}) == ["data.rows[0].mass"]
    assert bare({"symbols": [{"mass": 3}]}) == ["data.symbols[0].mass"]
    assert bare({"records": {"inner": {"v": 1}}}) == ["data.records.inner.v"]
    assert bare({"counts": [1, 2.5]}) == ["data.counts[0]", "data.counts[1]"]  # a float spoils all
    assert bare({"sum": 7}) == ["data.sum"]  # not structural at all
    assert bare({"limits": {"a": {"b": 1}}}) == ["data.limits.a.b"]
    assert bare({"shape": [True, 1]}) != []
    assert bare({"x": 1.5}) == ["data.x"]  # a structural key never exempts a float
    assert bare({"v": 1.0, "ev": "measured"}) == ["data.v"]  # a look-alike is not a label
    full = schema.num(1.0, "measured").model_dump(mode="json")
    assert bare(full) == [] and set(full) == set(schema.Num.model_fields)  # all 8 keys


def test_the_structural_allowlist_is_one_list_shared_with_the_web_client():
    from mdb.serve import typegen

    assert len(schema.STRUCTURAL_KEYS) == len(set(schema.STRUCTURAL_KEYS))
    text = typegen.render()
    assert "export const STRUCTURAL_KEYS" in text
    for key in schema.STRUCTURAL_KEYS:
        assert json.dumps(key) in text


@needs_fixture
def test_tile_domain_proof_names_the_diverging_range(store_copy):
    c = make_client(store_copy)
    mixed = _blob(store_copy, np.array([[-3.0, 1.0], [2.0, 0.5]], dtype=np.float32))
    tile = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": mixed, "op": "tile", "params": {}}), "reduce")["data"]
    assert tile["scale"] == "diverging" and tile["domain"]["vs"] == [-3.0, 3.0]
    proof = tile["domain"]["proof"]
    assert "[-m, m]" in proof and "max(|min|, |max|)" in proof and "symmetric" in proof
    positive = _blob(store_copy, np.array([[1.0, 4.0], [2.0, 3.0]], dtype=np.float32))
    tile = check(c.post("/api/v1/s/qwen/reduce", json={
        "pointer": positive, "op": "tile", "params": {}}), "reduce")["data"]
    assert tile["scale"] == "linear" and tile["domain"]["vs"] == [1.0, 4.0]
    assert tile["domain"]["proof"].startswith("linear")


@needs_fixture
def test_events_carry_only_labelled_or_structural_numbers():
    c = make_client(heartbeat=0.02)
    with c.websocket_connect(f"ws://127.0.0.1:{PORT}/api/v1/s/tiny/events") as ws:
        for _ in range(2):
            body = json.loads(ws.receive_text())
            assert not unlabelled_numbers(body["data"], "data.data"), body
            assert isinstance(body["data"]["at"], int)


@needs_fixture
def test_every_route_envelope_carries_a_state_and_the_types_require_it(client):
    seen = set()
    for url, params in gets(client):
        body = client.get(url, params=params).json()
        assert body["state"] in ("resolved", "unresolved", "refused"), url
        seen.add(body["state"])
    assert {"resolved", "refused"} <= seen
    reduced = client.post("/api/v1/s/qwen/reduce", json={
        "pointer": {"kind": "blob", "ref": "0" * 64}, "op": "stats", "params": {}}).json()
    assert reduced["state"] == "refused" and reduced["refused"]
    from mdb.serve.caps import envelope

    with pytest.raises(ValueError):
        envelope("hello", {"x": 1}, state="unresolved")  # no reason
    assert envelope("hello", {"x": 1}, state="unresolved", reason="why")["reason"] == "why"
    text = schema.Envelope.model_json_schema()
    assert "state" in text["required"] or "state" in text["properties"]
    from mdb.serve import typegen

    assert 'state: "resolved" | "unresolved" | "refused";' in typegen.render()


@needs_fixture
def test_the_crawl_has_no_bare_number_outside_the_list_and_listed_keys_are_never_floats(client):
    """Every verbatim body, binding and state clock is walked too (nothing is exempt)."""
    listed_floats = []

    def floats(value, key=None, path="data"):
        if isinstance(value, float) and key in STRUCTURAL:
            listed_floats.append(path)
        elif isinstance(value, dict):
            if server_built(value):
                return
            for k, v in value.items():
                floats(v, k, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                floats(v, key, f"{path}[{i}]")

    walked = 0
    for url, params in gets(client):
        body = client.get(url, params=params).json()
        if "refused" in body:
            continue
        assert not unlabelled_numbers(body["data"]), url
        floats(body["data"])
        walked += 1
    assert walked > 100 and not listed_floats, listed_floats[:5]


@needs_fixture
def test_inferred_ticks_are_flagged_in_the_schema(monkeypatch, client):
    assert schema.AxisTick(i=0, label="x").inferred is False
    _, depth = _depth(client, "qwen")
    assert all(t["inferred"] is False for t in depth["ticks"])
    c = layers_client(monkeypatch, receipts={"qwen"})
    caps, depth = _depth(c, "qwen")
    assert caps["layers"]["ev"] == "unresolved"
    assert not any(t["inferred"] for t in depth["ticks"])  # unknown: no layer ticks at all
    c = layers_client(monkeypatch)
    caps, depth = _depth(c, "tiny")
    flagged = {t["label"] for t in depth["ticks"] if t["inferred"]}
    assert caps["layers"]["ev"] == "derived-local" and {"L0", "L1", "L1 out"} <= flagged
    assert not [t for t in depth["ticks"] if t["inferred"] and t["port"] not in ("h", "h_out")]


@needs_fixture
def test_a_recorded_hidden_state_address_resolves_to_a_slot_and_reads_the_right_tensor(client):
    """to_state_address needs the execution point: a hidden-state address with a retained cut
    yields a hidden-slot pointer on exactly that cut, and reducing it equals reading the cut's
    slot directly."""
    from mdb.serve import blobs

    store = Path(FIXTURE) / "store"
    found = 0
    for addr in ("t2.L0.h", "t2.L1.h", "t2.L1.h_out", "t2.emb"):
        data = check(client.get("/api/v1/s/tiny/frame", params={"addr": addr}), "frame")["data"]
        focus = data["focus"]
        slots = [p["spec"] for p in focus["pointers"]
                 if p["spec"] and p["spec"]["kind"] == "slot" and p["spec"]["slot"] == "hidden"]
        if not slots:
            continue
        found += 1
        spec = slots[0]
        assert spec["cut"] == focus["cut"]
        direct, _ = blobs.cut_slot(store, spec["cut"], "hidden")
        direct = direct.astype(np.float64)
        stats = check(reduce_tiny(client, spec), "reduce")["data"]
        assert stats["l2"]["v"] == pytest.approx(float(np.sqrt((direct ** 2).sum())), rel=1e-12)
        assert stats["max"]["v"] == direct.max() and stats["min"]["v"] == direct.min()
        assert stats["shape"] == list(direct.shape)
    assert found >= 2, "hidden-state addresses did not resolve to state slots"


def reduce_tiny(c, pointer):
    return c.post("/api/v1/s/tiny/reduce", json={"pointer": pointer, "op": "stats",
                                                 "params": {}})


def test_bare_floats_in_stored_bodies_are_supplied_unless_the_schema_maps_them():
    from mdb.serve.views.records import FIELD_CLASS, SUPPLIED_NOTE, label_floats

    body = {"local_gradient": 0.5, "effect": -1.25, "nested": {"delta": 2.0}, "count": 3,
            "kept": {"value": 1.5, "authority": "derived-local", "unit": "nats"},
            "labelled": {"v": 2.0, "ev": "measured"}}
    out = label_floats(body, schema="legacy-x-v1", kind="thing", recorded="native-run")
    for key, value in (("local_gradient", 0.5), ("effect", -1.25)):
        assert out[key]["ev"] == "supplied" and out[key]["v"] == value
        assert out[key]["note"] == SUPPLIED_NOTE and out[key]["recorded_authority"] == "native-run"
    assert out["nested"]["delta"]["ev"] == "supplied" and out["count"] == 3
    assert out["kept"]["ev"] == "derived-local" and out["kept"]["v"] == 1.5
    assert out["kept"]["unit"] == "nats" and out["kept"]["note"] is None
    # a stored {v, ev} is the server's own wire shape: never passed through
    assert out["labelled"]["ev"] == "supplied" and out["labelled"]["v"] == 2.0
    assert out["labelled"]["note"] == "stored body claims class measured"
    assert out["labelled"]["recorded_authority"] == "measured"
    none = label_floats({"x": 1.0}, schema="s", kind="k", recorded=None)
    assert none["x"]["recorded_authority"] is None and none["x"]["ev"] != "measured"
    assert all(cls in schema.VALUED for cls, _ in FIELD_CLASS.values())


@needs_fixture
def test_mapped_comparison_fields_keep_their_class_and_the_rest_is_supplied(client):
    rows = check(client.get("/api/v1/s/tiny/records", params={"kind": "comparison"}))
    detail = check(client.get(f"/api/v1/s/tiny/records/{rows['data']['rows'][0]['id']}"),
                   "record")["data"]
    body = detail["body"]
    assert body["interaction"]["ev"] == "derived-exact"
    assert body["effects"] and all(v["ev"] == "derived-exact" for v in body["effects"].values())
    arm_rows = check(client.get("/api/v1/s/tiny/records", params={"kind": "comparison-arm"}))
    seen = set()
    for row in arm_rows["data"]["rows"]:
        arm = check(client.get(f"/api/v1/s/tiny/records/{row['id']}"), "record")["data"]["body"]
        measured = (arm.get("measurement") or {}).get("value")
        if isinstance(measured, dict):
            assert measured["ev"] == "measured"
            seen.add("value")
        for key in ("local_linear_prediction", "native_minus_local_prediction"):
            if isinstance(arm.get(key), dict):
                assert arm[key]["ev"] == "derived-local"
                seen.add(key)

        def walk(v):
            if isinstance(v, dict):
                if v.get("ev") == "supplied":
                    assert v["note"] == "as recorded; this record carries no per-number " \
                        "authority"
                for x in v.values():
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(arm)
    assert "value" in seen


def test_a_non_string_authority_in_a_stored_body_is_not_a_label():
    from mdb.serve.views.records import _claims, label_floats

    body = {"authority": {"kind": "native", "by": ["a"]}, "value": 1.5, "inner": {"x": 0.25}}
    out = label_floats(body, schema="s", kind="k", recorded=None)
    assert out["value"]["ev"] == "supplied"
    assert out["inner"]["x"]["ev"] == "supplied"
    assert out["authority"] == {"kind": "native", "by": ["a"]}
    assert _claims(body) == []


# ===================================================== round 3: hostile stored data

import copy  # noqa: E402
import sqlite3  # noqa: E402


def _with(base, **kw):
    out = copy.deepcopy(base)
    out.update(kw)
    return out


def _deep():
    deep = cur = {}
    for _ in range(400):
        cur["a"] = {}
        cur = cur["a"]
    return deep


def _hostile_cases(arm, cmp_):
    """Every case of the audit's table, as ``name -> (extra rows, body of the row requested)``."""
    from saturn_pub.values import digest

    bad_arm = _with(arm, mask={"a": 1})
    str_arm = _with(arm, mask="zz")
    int_arm = _with(arm, mask=7)
    cases = {
        "auth_dict": _with(arm, authority={"kind": "native", "by": ["x"]}),
        "auth_list": _with(arm, authority=["measured"]),
        "auth_int": _with(arm, authority=5),
        "terminal_status_dict": _with(arm, terminal_status={"a": 1}),
        "status_list": _with(arm, status=["measured"]),
        "measurement_list": _with(arm, measurement=[1, 2]),
        "measurement_value_int": _with(arm, measurement=_with(arm["measurement"], value=5)),
        "measurement_value_bool": _with(arm, measurement=_with(arm["measurement"], value=True)),
        "focus_id_dict": _with(arm, focus_id={"x": 1}),
        "component_dict": _with(arm, component={"x": 1}),
        "pointers_str": _with(arm, pointers=["abc"]),
        "pointers_no_address": _with(arm, pointers=[{"kind": "occurrence"}]),
        "pointers_bad_address": _with(arm, pointers=[{"kind": "occurrence", "address": "garbage"}]),
        "pointers_int_address": _with(arm, pointers=[{"kind": "occurrence", "address": 5}]),
        "pointers_dict": _with(arm, pointers={"kind": "occurrence"}),
        "lone_surrogate": _with(arm, note="\ud800"),
        "forged_ev_measured": _with(arm, fake={"ev": "measured", "v": 123456.0}),
        "forged_ev_int": _with(arm, fake={"ev": "derived-exact", "v": 7}),
        "forged_ev_dict_class": _with(arm, fake={"ev": {"x": 1}, "v": 1.5}),
        "forged_v_string": _with(arm, bad={"ev": "measured", "v": "hello"}),
        "forged_vs_string": _with(arm, bad={"ev": "measured", "vs": "oops"}),
        "int_quantity": _with(arm, logit_margin=7, effect_int=-3, nested={"delta": 4}),
        "bool_flag": _with(arm, flag=True, margin=False),
        "nonfinite_inside": _with(arm, mystery=[1.0, 2.0]),
        "deep_nesting_400": _with(arm, deep=_deep()),
        "long_string_2MB": _with(arm, note="x" * 2_000_000),
        "huge_list_200k": _with(arm, big=list(range(200_000))),
        "many_keys_5k": _with(arm, wide={f"k{i}": 1.5 for i in range(5000)}),
        "history_cut_dict": _with(arm, input_cut={"a": 1}, result_cut=5),
        "mask_dict": _with(arm, mask={"a": 1}),
        "interaction_bool": _with(cmp_, interaction=True),
        "interaction_str": _with(cmp_, interaction="x"),
        "cmp_arms_dict": _with(cmp_, arms=[{"x": 1}]),
        "cmp_arms_str_not_list": _with(cmp_, arms="abc"),
        "cmp_linearization_list": _with(cmp_, linearization=[1]),
        "effects_str": _with(cmp_, effects={"1": "x", "2": 1}),
        "body_key_body_list": _with(arm, body=[1, 2]),
    }
    out = {name: ([], body) for name, body in cases.items()}
    out["cmp_with_dictmask_arm"] = ([bad_arm], _with(cmp_, arms=[digest(bad_arm)]))
    out["cmp_with_str_and_int_mask_arms"] = (
        [str_arm, int_arm, arm], _with(cmp_, arms=[digest(str_arm), digest(int_arm),
                                                    digest(arm)]))
    return out


class Hostile:
    """A private copy of the fixture project served in-process; rows are inserted and removed."""

    def __init__(self, root):
        self.root = root
        self.c = make_client(root, raise_server_exceptions=False)
        db = sqlite3.connect(root / "project.sqlite")
        row = lambda kind: json.loads(db.execute(  # noqa: E731
            "select body from circuit_records where kind=? limit 1", (kind,)).fetchone()[0])
        self.arm, self.cmp = row("comparison-arm"), row("comparison")
        self.healthy = db.execute(
            "select count(*) from circuit_records where session_id='tiny'").fetchone()[0]
        db.close()
        self.cases = _hostile_cases(self.arm, self.cmp)

    def insert(self, rows):
        from saturn_pub.values import digest

        db = sqlite3.connect(self.root / "project.sqlite")
        ids = []
        for body in rows:
            try:
                rid = digest(body)
            except ValueError:
                rid = "unsealable-" + str(len(ids))
            db.execute("insert or replace into circuit_records values (?,?,?,?)",
                       (rid, "tiny", body["kind"] if isinstance(body.get("kind"), str) else "x",
                        json.dumps(body)))
            ids.append(rid)
        db.commit()
        db.close()
        return ids

    def remove(self, ids):
        db = sqlite3.connect(self.root / "project.sqlite")
        for rid in ids:
            db.execute("delete from circuit_records where id=?", (rid,))
        db.commit()
        db.close()


@pytest.fixture(scope="module")
def hostile(tmp_path_factory):
    if not FIXTURE or not (Path(FIXTURE) / "project.sqlite").is_file():
        pytest.skip("no fixture")
    root = tmp_path_factory.mktemp("hostile") / "project"
    shutil.copytree(FIXTURE, root)
    return Hostile(root)


ROUTES = ["/records", "/timeline?lens=records", "/frame", "/graph", "/symbols", "/capabilities",
          "/branches", "/timeline"]
HOSTILE_NAMES = sorted(_hostile_cases({"measurement": {}}, {}))


@pytest.mark.parametrize("name", HOSTILE_NAMES)
@needs_fixture
def test_hostile_stored_records_never_5xx_and_never_forge_a_label(hostile, name):
    pre, body = hostile.cases[name]
    ids = hostile.insert([*pre, body])
    try:
        for route in ROUTES + ["/records?kind=" + str(body.get("kind"))]:
            response = hostile.c.get("/api/v1/s/tiny" + route)
            assert response.status_code < 500, (name, route, response.status_code)
            check(response)
        listing = check(hostile.c.get("/api/v1/s/tiny/records"), "records")["data"]
        # the other rows of the session are still listed (one bad row never hides the rest)
        assert listing["total"] >= hostile.healthy
        detail = hostile.c.get("/api/v1/s/tiny/records/" + urllib_quote(ids[-1]))
        assert detail.status_code == 200, (name, detail.status_code)
        out = check(detail, "record")
        assert out["bytes"] <= CAP and len(detail.content) <= CAP
        assert not unlabelled_numbers(out["data"]), (name, unlabelled_numbers(out["data"])[:3])

        def look_alikes(v, path="body"):
            if isinstance(v, dict):
                if "ev" in v and "v" in v and not server_built(v):
                    # a stored {v, ev} dict is never emitted with a number in v
                    assert not (isinstance(v["v"], (int, float)) and not isinstance(
                        v["v"], bool)), (name, path)
                for k, x in v.items():
                    look_alikes(x, f"{path}.{k}")
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    look_alikes(x, f"{path}[{i}]")

        look_alikes(out["data"]["body"])
        for other in hostile.c.get("/api/v1/s/tiny/records").json()["data"]["rows"][:3]:
            assert hostile.c.get("/api/v1/s/tiny/records/" + urllib_quote(
                other["id"])).status_code == 200
    finally:
        hostile.remove(ids)


def urllib_quote(value):
    import urllib.parse

    return urllib.parse.quote(value, safe="")


def _detail_body(hostile, name):
    pre, body = hostile.cases[name]
    ids = hostile.insert([*pre, body])
    try:
        out = check(hostile.c.get("/api/v1/s/tiny/records/" + urllib_quote(ids[-1])), "record")
        return out
    finally:
        hostile.remove(ids)


@needs_fixture
def test_stored_class_claims_are_supplied_with_the_claim_in_a_note(hostile):
    body = _detail_body(hostile, "forged_ev_measured")["data"]["body"]["fake"]
    assert body["ev"] == "supplied" and body["v"] == 123456.0
    assert body["note"] == "stored body claims class measured"
    assert body["recorded_authority"] == "measured"
    body = _detail_body(hostile, "forged_ev_int")["data"]["body"]["fake"]
    assert body["ev"] == "supplied" and body["v"] == 7.0
    body = _detail_body(hostile, "forged_ev_dict_class")["data"]["body"]["fake"]
    assert body["ev"] == "supplied" and body["recorded_authority"] is None
    assert body["note"].startswith("stored body claims class ") and len(body["note"]) <= 70
    # v is not a number: an ordinary mapping, its strings stay strings, nothing is a number
    bad = _detail_body(hostile, "forged_v_string")["data"]["body"]["bad"]
    assert bad == {"ev": "measured", "v": "hello"}


@needs_fixture
def test_stored_ints_are_labelled_unless_structural_and_the_big_list_is_bounded(hostile):
    out = _detail_body(hostile, "int_quantity")
    body = out["data"]["body"]
    for value in (body["logit_margin"], body["effect_int"], body["nested"]["delta"]):
        assert value["ev"] == "supplied" and value["note"]
    assert body["logit_margin"]["v"] == 7.0 and body["effect_int"]["v"] == -3.0
    big = _detail_body(hostile, "huge_list_200k")
    assert big["bytes"] <= CAP
    items = big["data"]["body"]["big"]
    assert len(items) <= 16 and all(i["ev"] == "supplied" for i in items)
    assert any(o["field"].startswith("body.big") and o["count"] >= 199_000
               for o in big["omitted"]), big["omitted"]
    flags = _detail_body(hostile, "bool_flag")["data"]["body"]
    assert flags["flag"] is True and flags["margin"] is False  # a bool is not a number


@needs_fixture
def test_measurement_value_int_is_measured_and_bool_is_unresolved(hostile):
    out = _detail_body(hostile, "measurement_value_int")["data"]
    assert out["body"]["measurement"]["value"]["ev"] == "measured"
    assert out["summary"]["nums"][0]["n"]["ev"] == "measured"
    assert out["summary"]["nums"][0]["n"]["v"] == 5.0
    out = _detail_body(hostile, "measurement_value_bool")["data"]
    num_ = out["summary"]["nums"][0]["n"]
    assert num_["ev"] == "unresolved" and num_["v"] is None
    assert num_["reason"] == "stored value is not a number"
    out = _detail_body(hostile, "interaction_bool")["data"]
    num_ = out["summary"]["nums"][0]["n"]
    assert num_["ev"] == "unresolved" and num_["reason"] == "stored value is not a number"
    assert out["body"]["interaction"] is True


def test_label_floats_non_finite_bool_and_ranges_are_never_a_number():
    from mdb.serve.caps import NONFINITE, bound_json, canonical
    from mdb.serve.views.records import label_floats

    out = label_floats({"a": float("nan"), "b": float("inf"), "c": [float("-inf")],
                        "d": 10 ** 400, "e": True, "f": None},
                       schema="s", kind="k", recorded=None)
    for leaf in (out["a"], out["b"], out["c"][0], out["d"]):
        assert leaf["ev"] == "unresolved" and leaf["v"] is None and leaf["reason"]
    assert out["a"]["reason"] == NONFINITE and out["e"] is True and out["f"] is None
    canonical(out)  # never raises
    omitted = []
    bounded = bound_json({"x": float("nan"), "y": [float("inf")]}, field="b", omitted=omitted)
    assert bounded["x"]["ev"] == "unresolved" and bounded["x"]["reason"] == NONFINITE
    assert bounded["y"][0]["reason"] == NONFINITE
    canonical(bounded)


def test_label_floats_v2_form_is_the_only_stored_class_and_lists_are_capped():
    from mdb.serve.schema import Omitted
    from mdb.serve.views.records import LABEL_ITEMS, label_floats

    kept = label_floats({"value": 2, "authority": "derived-exact", "unit": 3, "proof": ["x"]},
                        schema="s", kind="k", recorded=None)
    assert kept["ev"] == "derived-exact" and kept["v"] == 2.0 and kept["unit"] is None
    for bad in ({"value": 2, "authority": "bogus"}, {"value": True, "authority": "measured"},
                {"value": float("nan"), "authority": "measured"},
                {"value": "2", "authority": "measured"}):
        out = label_floats(bad, schema="s", kind="k", recorded=None)
        assert out.get("ev") != "measured"
    omitted = []
    out = label_floats({"big": list(range(1000)), "tokens": list(range(1000))}, schema="s",
                       kind="k", recorded=None, omitted=omitted)
    assert len(out["big"]) == LABEL_ITEMS and all(i["ev"] == "supplied" for i in out["big"])
    assert out["tokens"] == list(range(1000))  # structural key: untouched
    assert omitted == [Omitted(field="big", count=1000 - LABEL_ITEMS,
                               reason="list items beyond the first; not labelled")]


def test_stored_readers_check_types_at_the_point_of_use():
    from mdb.serve.stored import as_dict, as_int, as_list, as_number, as_str, is_number

    assert as_str({"a": 1}) == "" and as_str("x") == "x" and as_str(None, "d") == "d"
    assert as_dict([1]) == {} and as_dict({"a": 1}) == {"a": 1}
    assert as_list("abc") == [] and as_list([1]) == [1]
    for bad in (True, False, float("nan"), float("inf"), "1", None, [1], 10 ** 400):
        assert not is_number(bad) and as_number(bad, 9) == 9
    assert as_number(3) == 3 and as_number(2.5) == 2.5 and as_int(3) == 3
    assert as_int(True) is None and as_int(2.0) is None


def test_canonical_survives_a_lone_surrogate_and_counts_the_bytes_sent():
    from mdb.serve.caps import canonical

    raw = canonical({"note": "\ud800", "t": "é"})
    assert json.loads(raw) == {"note": "\ud800", "t": "é"}
    assert raw == raw.decode("ascii").encode()  # fell back to escapes: ASCII, valid JSON
    assert canonical({"t": "é"}) == '{"t":"é"}'.encode("utf-8")  # normal text stays UTF-8


@needs_fixture
def test_the_lone_surrogate_record_is_served_with_real_byte_counts(hostile):
    pre, body = hostile.cases["lone_surrogate"]
    ids = hostile.insert([body])
    try:
        response = hostile.c.get("/api/v1/s/tiny/records/" + urllib_quote(ids[-1]))
        assert response.status_code == 200 and len(response.content) <= CAP
        assert json.loads(response.content)["bytes"] <= len(response.content)
        assert hostile.c.get("/api/v1/s/tiny/records").status_code == 200
    finally:
        hostile.remove(ids)


@needs_fixture
def test_a_comparison_with_a_bad_arm_still_lists_the_good_arms_and_a_bad_summary_is_a_row(hostile):
    out = _detail_body(hostile, "cmp_with_str_and_int_mask_arms")["data"]
    masks = [a["mask"] for a in out["arms"]]
    assert masks == sorted(masks) or -1 in masks and len(masks) == 3
    assert all(a["value"]["ev"] in ("measured", "unresolved") for a in out["arms"])


def _all_record_rows(client, sid="tiny"):
    first = check(client.get(f"/api/v1/s/{sid}/records"), "records")["data"]
    rows = list(first["rows"])
    for page in range(1, first["pages"]):
        rows += client.get(f"/api/v1/s/{sid}/records", params={"page": page}).json()["data"]["rows"]
    return rows


@pytest.mark.parametrize("case", ["focus_id_dict", "dangling_arm"])
@needs_fixture
def test_the_session_graph_leaves_out_a_record_it_cannot_place_and_lists_it(hostile, case):
    """A sealed record whose reference cannot be resolved (an unhashable ``focus_id``, an arm
    that is not recorded) used to make ``zoom=session`` a 409; it is now left out of the graph,
    the rest of the graph is served, and the row is listed as refused (``graph:<id>``)."""
    body = (_with(hostile.arm, focus_id={"x": 1}) if case == "focus_id_dict"
            else _with(hostile.cmp, arms=["0" * 64]))
    params = {"zoom": "session", "k": 64}
    healthy = check(hostile.c.get("/api/v1/s/tiny/graph", params=params), "graph")["data"]
    ids = hostile.insert([body])
    try:
        response = hostile.c.get("/api/v1/s/tiny/graph", params=params)
        assert response.status_code == 200, response.text[:300]
        served = check(response, "graph")["data"]
        assert [n["id"] for n in served["nodes"]] == [n["id"] for n in healthy["nodes"]]
        rows = _all_record_rows(hostile.c)
        refused = [r for r in rows if r["id"] == "graph:" + ids[0]]
        assert len(refused) == 1 and refused[0]["status"] == "refused"
        # the record itself verifies and is still listed as a record
        assert [r["status"] for r in rows if r["id"] == ids[0]] not in ([], ["refused"])
    finally:
        hostile.remove(ids)


# ---- session-level tampering: a broken session or row is refused for itself only

TAMPER_ROUTES = ["/capabilities", "/frame", "/timeline", "/timeline?lens=symbols", "/branches",
                 "/symbols", "/graph", "/records"]


def _sql(statement):
    def mutate(root):
        db = sqlite3.connect(root / "project.sqlite")
        db.execute(statement)
        db.commit()
        db.close()
    return mutate


def _tiny_cut_files(root):
    db = sqlite3.connect(root / "project.sqlite")
    ids = [r[0] for r in db.execute("select id from cuts where session_id='tiny'")]
    db.close()
    return sorted((root / "store/cuts" / f"{i}.json") for i in ids)


def _cut_garbage(root):
    f = _tiny_cut_files(root)[0]
    d = json.loads(f.read_text())
    d["execution_point"] = "garbage"
    f.write_text(json.dumps(d))


def _cuts_corrupt(root):
    for f in _tiny_cut_files(root):
        f.write_text("{not json")


TAMPER = {
    "slice_seal_broken": (_sql("update slices set body = replace(body, '\"complete\":true', "
                               "'\"complete\":false') where session_id='qwen'"), "qwen"),
    "symbol_seal_broken": (_sql("update symbols set body = replace(body, 'Supplied', "
                                "'Xupplied') where session_id='tiny'"), "tiny"),
    "lifecycle_tampered": (_sql("update symbol_lifecycles set body = replace(body, "
                                "'\"source_row\":', '\"source_row\":\"x\",\"_o\":') "
                                "where session_id='qwen'"), "qwen"),
    "cut_descriptor_tampered": (_cut_garbage, "tiny"),
    "cut_files_corrupt": (_cuts_corrupt, "tiny"),
    "session_identity_bad_json": (_sql("update sessions set identity='{bad' where id='tiny'"),
                                  "tiny"),
    "session_execution_list": (_sql("update sessions set execution='[1,2]' where id='tiny'"),
                               "tiny"),
    "session_debug_info_bad": (_sql("update sessions set debug_info='nope' where id='tiny'"),
                               "tiny"),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
@needs_fixture
def test_tampered_sessions_and_rows_are_refused_for_themselves_never_a_5xx(tmp_path, name):
    mutate, broken = TAMPER[name]
    root = tmp_path / "project"
    shutil.copytree(FIXTURE, root)
    mutate(root)
    c = make_client(root, raise_server_exceptions=False)
    healthy = "qwen" if broken == "tiny" else "tiny"
    for sid in (broken, healthy):
        for route in TAMPER_ROUTES:
            response = c.get(f"/api/v1/s/{sid}{route}")
            assert response.status_code < 500, (name, sid, route, response.status_code)
            body = json.loads(response.content)
            if response.status_code != 200:
                assert body["state"] == "refused" and body["refused"] and body.get("detail")
    listing = c.get("/api/v1/sessions")
    assert listing.status_code == 200
    rows = {r["id"]: r for r in check(listing, "sessions")["data"]}
    assert set(rows) == {"qwen", "tiny"}
    assert rows[healthy]["state"] == "resolved" and rows[healthy]["cuts"] > 0
    whole = name.startswith(("session_", "cut_"))
    if whole:
        assert rows[broken]["state"] == "refused" and rows[broken]["reason"]
        assert c.get(f"/api/v1/s/{broken}/records").status_code == 409
    else:
        assert rows[broken]["state"] == "resolved"
        page = check(c.get(f"/api/v1/s/{broken}/records"), "records")["data"]
        assert page["total"] > 0
    assert c.get("/api/v1/hello").status_code == 200
    # the healthy session still answers everything
    for route in TAMPER_ROUTES:
        assert c.get(f"/api/v1/s/{healthy}{route}").status_code == 200, (name, route)


@needs_fixture
def test_a_broken_slice_seal_is_a_refused_placeholder_row_in_the_drawer(tmp_path):
    mutate, sid = TAMPER["slice_seal_broken"]
    root = tmp_path / "project"
    shutil.copytree(FIXTURE, root)
    mutate(root)
    c = make_client(root, raise_server_exceptions=False)
    rows = check(c.get(f"/api/v1/s/{sid}/records", params={"kind": "slice"}),
                 "records")["data"]["rows"]
    assert rows and all(r["status"] == "refused" and r["nums"][0]["n"]["ev"] == "refused"
                        and r["nums"][0]["n"]["reason"] for r in rows)
    detail = c.get(f"/api/v1/s/{sid}/records/{urllib_quote(rows[0]['id'])}")
    assert detail.status_code == 200 and json.loads(detail.content)["state"] == "refused"


# ---- blob store, reduce cache, guard

@needs_fixture
def test_a_symlink_out_of_the_store_is_refused_like_a_missing_blob(store_copy, tmp_path):
    import hashlib

    from mdb.serve import blobs

    outside = tmp_path / "outside"
    outside.mkdir()
    root = store_copy / "store"
    real = next((root / "tensors").glob("*.safetensors"))
    payload = real.read_bytes()
    ref = hashlib.sha256(payload).hexdigest()
    messages = []
    for target_exists in (True, False):
        link_ref = ("a" if target_exists else "b") * 64
        link = root / "tensors" / f"{link_ref}.safetensors"
        target = outside / f"{link_ref}.safetensors"
        if target_exists:
            target.write_bytes(payload)
        link.symlink_to(target)
        with pytest.raises(blobs.BlobError) as error:
            blobs.tensor_blob(root, link_ref)
        messages.append(error.value.reason.replace(link_ref[:12], "REF"))
        assert not error.value.refused and not error.value.limit
    missing = "c" * 64
    with pytest.raises(blobs.BlobError) as error:
        blobs.tensor_blob(root, missing)
    messages.append(error.value.reason.replace(missing[:12], "REF"))
    assert len(set(messages)) == 1, messages
    # a regular in-store blob still reads
    assert blobs.tensor_blob(root, ref)[1]["shape"]


def test_a_symlinked_slot_blob_outside_the_store_is_unresolved(tmp_path):
    from mdb.serve import blobs

    root = tmp_path / "store"
    (root / "blobs").mkdir(parents=True)
    outside = tmp_path / "elsewhere.safetensors"
    outside.write_bytes(b"x" * 16)
    (root / "blobs" / ("d" * 64 + ".safetensors")).symlink_to(outside)
    assert blobs._store_file(root / "blobs" / ("d" * 64 + ".safetensors")) is False
    inside = root / "blobs" / ("e" * 64 + ".safetensors")
    inside.write_bytes(b"y")
    assert blobs._store_file(inside) is True


def test_the_reduce_cache_key_follows_the_blob_an_occurrence_pointer_resolves_to(monkeypatch):
    from types import SimpleNamespace

    from mdb.serve import reduce as reduce_mod

    rows = {"blob": "1" * 64}
    monkeypatch.setattr(reduce_mod, "occurrence_row", lambda model, address: {
        "blob_id": rows["blob"]} if rows["blob"] else None)
    model, spec = SimpleNamespace(sid="s"), {"kind": "occurrence", "address": "t2.L1.p"}
    before = reduce_mod.Reducer._pointer_key(model, spec)
    assert before == reduce_mod.Reducer._pointer_key(model, spec)  # stable while unchanged
    rows["blob"] = "2" * 64  # the index now names another blob for the same address
    after = reduce_mod.Reducer._pointer_key(model, spec)
    assert after != before
    rows["blob"] = None
    assert reduce_mod.Reducer._pointer_key(model, spec) not in (before, after)
    other = reduce_mod.Reducer._pointer_key(model, {"kind": "occurrence", "address": "t2.L0.p"})
    assert other != reduce_mod.Reducer._pointer_key(model, spec)  # unresolved: keyed on address


@needs_fixture
def test_a_repeated_origin_or_host_header_is_refused(client):
    app = client.app
    good = ("host", f"127.0.0.1:{PORT}")
    for headers in ([good, ("origin", f"http://127.0.0.1:{PORT}"),
                     ("origin", f"http://127.0.0.1:{PORT}")],
                    [good, ("origin", f"http://127.0.0.1:{PORT}"),
                     ("origin", "http://evil.example")],
                    [good, good],
                    [good, ("host", "evil.example")]):
        sent, _ = asgi_call(app, headers=headers)
        assert _status(sent) == 403, headers
        assert json.loads(sent[1]["body"])["refused"] == "duplicate_header"
    sent, _ = asgi_call(app, headers=[good, ("origin", f"http://127.0.0.1:{PORT}")])
    assert _status(sent) == 200


# ---- graph overlay, first stepped row

@needs_fixture
def test_graph_overlay_carries_the_interaction_as_a_labelled_number_not_text(client):
    seen = []
    for sid in ("tiny", "qwen"):
        body = check(client.get(f"/api/v1/s/{sid}/graph"), "graph")
        for overlay in body["data"]["overlays"]:
            seen.append(overlay)
            assert not any(ch.isdigit() for ch in overlay["label"]), overlay["label"]
            if overlay["kind"] == "comparison":
                assert overlay["interaction"]["ev"] == "derived-exact"
                assert overlay["interaction"]["v"] is not None
            else:
                assert overlay["interaction"] is None
    assert any(o["kind"] == "comparison" for o in seen), "fixture has no comparison overlay"


def test_the_overlay_number_is_unresolved_when_the_stored_value_is_not_a_number():
    from mdb.serve.views.records import _interaction

    assert _interaction({"kind": "comparison"}) is None
    for bad in (True, "x", [1], {"a": 1}, None):
        out = _interaction({"interaction": bad})
        assert out.ev == "unresolved" and out.reason == "stored value is not a number"
    assert _interaction({"interaction": float("nan")}).reason == "non-finite in the stored body"
    assert _interaction({"interaction": 2}).ev == "derived-exact"


@needs_fixture
def test_unstepped_prompt_rows_do_not_resolve_sample_or_logits(client):
    """first_step is the lowest logical step of any cut's execution point (the prompt's last
    row); a sample or logits address on an earlier row was never produced by the session."""
    model = client.app.state.recorded.model("tiny")
    first = model.first_step
    assert first == min(c["execution_point"]["logical_step"] for c in model.cuts.values())
    assert first >= 1

    def pointers(addr):
        data = check(client.get("/api/v1/s/tiny/frame", params={"addr": addr}), "frame")["data"]
        return data["focus"]["pointers"]

    for port in ("sample", "logits"):
        early = pointers(f"t{first - 1}.{port}")
        assert not [p for p in early if p["status"] == "resolvable"], port
    early = pointers(f"t{first - 1}.sample")
    assert early and early[0]["status"] == "frontier"
    assert "never stepped" in early[0]["reason"] and early[0]["reason"].startswith("unresolved")
    for port, slot in (("sample", "tokens"), ("logits", "logits")):
        at = pointers(f"t{first}.{port}")
        assert [p for p in at if p["status"] == "resolvable" and p["spec"]["slot"] == slot], port

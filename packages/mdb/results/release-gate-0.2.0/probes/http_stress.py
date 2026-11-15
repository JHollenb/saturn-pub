"""Release-gate 0.2.0, part 2: every `mdb serve` endpoint, then a light load test.

Usage: http_stress.py BASE_URL SERVER_PID. Prints HTTP lines and an HTTP_JSON summary. Measures
status, latency, body size (cap 64 KiB), server RSS (ps), errors, hangs (timeouts).
"""
import asyncio
import json
import statistics as st
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import websockets

BASE, PID = sys.argv[1].rstrip("/"), int(sys.argv[2])
API = BASE + "/api/v1"
C = httpx.Client(timeout=120)
OUT = {"endpoints": [], "findings": []}
CAP = 65536


def rss_mb():
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(PID)], capture_output=True,
                              text=True).stdout.strip() or 0) / 1024


def call(method, path, body=None, timeout=120, note=""):
    t = time.monotonic()
    try:
        r = C.request(method, API + path if path.startswith("/") else path, json=body, timeout=timeout)
        dt = time.monotonic() - t
        try:
            js = r.json()
        except Exception:  # noqa: BLE001
            js = None
        row = {"m": method, "path": path, "status": r.status_code, "ms": round(dt * 1000, 1),
               "bytes": len(r.content), "note": note,
               "state": (js or {}).get("state") if isinstance(js, dict) else None,
               "reason": str((js or {}).get("reason") or (js or {}).get("error") or "")[:160]
               if isinstance(js, dict) else ""}
    except Exception as e:  # noqa: BLE001
        row = {"m": method, "path": path, "status": None, "ms": round((time.monotonic() - t) * 1000, 1),
               "error": f"{type(e).__name__}: {e}", "note": note}
        js = None
    if row.get("bytes", 0) > CAP:
        OUT["findings"].append(f"body over cap {row}")
    print(f"HTTP {row['m']:4s} {row['status']} {row['ms']:8.1f}ms {row.get('bytes', 0):6d}B {path[:90]} "
          f"{row.get('state') or ''} {row.get('reason', '') or row.get('error', '')}", flush=True)
    OUT["endpoints"].append(row)
    return row, js


def console(line, **kw):
    return call("POST", f"/s/{SID}/console", {"line": line}, **kw)


hello = C.get(API + "/hello").json()
SID = hello["data"]["live"]
print("live sid", SID, "rss", rss_mb())
OUT["rss_start_mb"] = rss_mb()

# ---- 1. every endpoint once (live session) --------------------------------------------------
console("start name=base The Eiffel Tower is located in the city of")
console("run name=donor The Colosseum is located in the city of")
console("frame base:")
console("until L12.h")
console("checkpoint s0")
console("break L5.p")
console('forecast M[" Paris"," Rome"]')
console("set L12.m = 0.5 scale")
for path in ["/hello", "/sessions", f"/s/{SID}/capabilities", f"/s/{SID}/frame", f"/s/{SID}/frame?addr=t10.L3.p",
             f"/s/{SID}/timeline", f"/s/{SID}/timeline?lens=cuts&x0=0&x1=5", f"/s/{SID}/branches",
             f"/s/{SID}/symbols", f"/s/{SID}/symbols?page=1", f"/s/{SID}/symbols/nope", f"/s/{SID}/graph",
             f"/s/{SID}/records", f"/s/{SID}/records?kind=forecast", f"/s/{SID}/records/nope",
             f"/s/{SID}/stored/nope", f"/s/{SID}/live", f"/s/{SID}/live/ledger", f"/s/{SID}/live/ledger?limit=0",
             f"/s/{SID}/live/timeline", f"/s/{SID}/export", "/s/nope/frame", "/nope", BASE + "/",
             BASE + "/assets/x.js"]:
    call("GET", path)
for lst in ["breakpoints", "displays", "checkpoints", "staged", "committed", "upstream", "bogus"]:
    call("GET", f"/s/{SID}/live/items?list={lst}")
call("GET", f"/s/{SID}/live/items?list=upstream&addr=t10.L12.h")
call("GET", f"/s/{SID}/live/items?list=checkpoints&offset=-1")
row, frame = call("GET", f"/s/{SID}/live")
# reduce on the live occurrence (a pointer from print)
_, pr = call("POST", f"/s/{SID}/op", {"op": "print", "args": {"address": "h"}, "request_id": "r1"})
ptr = None
try:
    res = pr["data"]["result"]
    ptr = res.get("pointer") or (res.get("value") or {}).get("pointer")
except Exception:  # noqa: BLE001
    pass
OUT["print_result_keys"] = sorted((pr or {}).get("data", {}).get("result", {}).keys()) if pr else None
if ptr:
    call("POST", f"/s/{SID}/reduce", {"pointer": ptr, "op": "stats", "params": {}}, note="reduce print's pointer")
for op, params in [("stats", {}), ("topk", {"k": 5}), ("slice", {}), ("histogram", {}), ("tile", {}),
                   ("project", {})]:
    call("POST", f"/s/{SID}/reduce", {"pointer": {"kind": "occurrence", "address": "t10.L12.h"}, "op": op,
                                        "params": params}, note=f"reduce {op}")
call("POST", f"/s/{SID}/reduce", {"pointer": {"kind": "occurrence", "address": "t10.L12.h"}, "op": "stats",
                                    "params": {}}, note="reduce hand pointer")
call("POST", f"/s/{SID}/op", {"op": "nope", "args": {}, "request_id": "r2"})
call("POST", f"/s/{SID}/op", {"op": "promptset", "args": {"action": "load", "name": "x", "file": "/etc/passwd"},
                              "request_id": "r3"}, note="promptset load must be refused")
call("POST", f"/s/{SID}/console", {"line": "x" * 5000}, note="over max_length")
call("POST", f"/s/{SID}/console", {"line": "record /tmp/x.mdb"}, note="record refused?")
call("POST", f"/s/{SID}/console", {"line": "source /etc/passwd"}, note="server-side source?")
call("POST", f"/s/{SID}/interrupt")
console("abort")

# ---- 2. rapid stepping --------------------------------------------------------------------
console("delete")
console("restart s0")
lat = []
for i in range(100):
    r, _ = console("step", note="rapid")
    lat.append(r["ms"])
OUT["rapid_step_ms"] = {"n": len(lat), "median": st.median(lat), "p95": sorted(lat)[94], "max": max(lat)}
OUT["rss_after_steps_mb"] = rss_mb()

# ---- 3. large symbol tables (paged console) ------------------------------------------------
console("until logits")
for off in (0, 50, 200):
    console(f'symbols consumer=M[" Paris"," Rome"] rows=all offset={off} limit=16', note="large symbols")
call("GET", f"/s/{SID}/symbols?page=0")
call("GET", f"/s/{SID}/records?page=0")
call("GET", f"/s/{SID}/live/ledger")

# ---- 4. concurrent readers while stepping ---------------------------------------------------
console("restart s0")
stop = threading.Event()
conc = {"ok": 0, "busy": 0, "err": 0, "other": 0, "lat": []}


def reader(i):
    paths = [f"/s/{SID}/live", f"/s/{SID}/frame", "/hello", f"/s/{SID}/live/items?list=checkpoints"]
    k = 0
    while not stop.is_set():
        t = time.monotonic()
        try:
            r = httpx.get(API + paths[k % len(paths)], timeout=60)
            conc["lat"].append((time.monotonic() - t) * 1000)
            if r.status_code == 200:
                conc["ok"] += 1
            elif r.status_code == 409 and "busy" in r.text:
                conc["busy"] += 1
            else:
                conc["other"] += 1
                if conc["other"] < 5:
                    print("CONC other", r.status_code, r.text[:200], flush=True)
        except Exception as e:  # noqa: BLE001
            conc["err"] += 1
            if conc["err"] < 5:
                print("CONC err", type(e).__name__, e, flush=True)
        k += 1


with ThreadPoolExecutor(16) as ex:
    futs = [ex.submit(reader, i) for i in range(16)]
    sl = []
    for i in range(40):
        r, _ = console("step", note="step under load")
        sl.append(r["ms"])
    stop.set()
    for f in futs:
        f.result()
OUT["concurrent"] = {"readers": 16, "ok": conc["ok"], "busy409": conc["busy"], "err": conc["err"],
                     "other": conc["other"], "reader_median_ms": st.median(conc["lat"]) if conc["lat"] else None,
                     "reader_p95_ms": sorted(conc["lat"])[int(len(conc["lat"]) * .95)] if conc["lat"] else None,
                     "step_median_ms_under_load": st.median(sl), "step_max_ms_under_load": max(sl)}
OUT["rss_after_concurrent_mb"] = rss_mb()

# ---- 5. a long op while other requests arrive ----------------------------------------------
console("restart s0")
console("until logits")
long_res = {}


def long_op():
    long_res["row"], long_res["js"] = console('symbols consumer=M[" Paris"," Rome"] rows=all causal=true limit=16',
                                              note="long op")


th = threading.Thread(target=long_op)
th.start()
time.sleep(0.5)
during = [call("GET", "/hello", note="during long op")[0], call("GET", f"/s/{SID}/live", note="during long op",
                                                                 timeout=300)[0]]
th.join()
OUT["long_op"] = {"long_ms": long_res["row"]["ms"], "hello_ms": during[0]["ms"], "live_ms": during[1]["ms"],
                  "live_waited_for_long": during[1]["ms"] > 0.5 * long_res["row"]["ms"]}

# ---- 6. interrupt a long op -----------------------------------------------------------------
console("until logits")
th = threading.Thread(target=long_op)
th.start()
time.sleep(1.0)
ir, ij = call("POST", f"/s/{SID}/interrupt", note="interrupt long op")
th.join()
OUT["interrupt"] = {"interrupt": (ij or {}).get("data"), "long_ms_after_interrupt": long_res["row"]["ms"],
                    "long_text": str(((long_res.get("js") or {}).get("data") or {}).get("text", ""))[:200]}
r, j = console("where", note="after interrupt")
OUT["where_after_interrupt"] = str(((j or {}).get("data") or {}).get("text", ""))[:160]

# ---- 7. client disconnect mid-op (client timeout) ------------------------------------------
console("restart s0")
console("until logits")
try:
    httpx.post(API + f"/s/{SID}/console", json={"line": 'symbols consumer=M[" Paris"," Rome"] rows=all causal=true limit=16'},
               timeout=0.5)
    OUT["disconnect"] = "op finished within 0.5 s"
except httpx.TimeoutException:
    OUT["disconnect"] = "client timed out (disconnected)"
t = time.monotonic()
r, j = console("where", note="after client disconnect", timeout=300)
OUT["after_disconnect"] = {"where_ms": r["ms"], "status": r["status"],
                           "text": str(((j or {}).get("data") or {}).get("text", ""))[:160]}

# ---- 8. queue overflow (busy) ---------------------------------------------------------------
console("restart s0")
console("until logits")
statuses = []


def fire(i):
    try:
        rr = httpx.post(API + f"/s/{SID}/console", json={"line": "step"}, timeout=300)
        return rr.status_code, ("busy" in rr.text)
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, False


th = threading.Thread(target=long_op)
th.start()
time.sleep(0.3)
with ThreadPoolExecutor(48) as ex:
    statuses = list(ex.map(fire, range(48)))
th.join()
OUT["queue_overflow"] = {"n": 48, "200": sum(s == 200 for s, _ in statuses),
                         "409busy": sum(s == 409 and b for s, b in statuses),
                         "other": [s for s, b in statuses if s not in (200, 409)]}

# ---- 9. websocket events --------------------------------------------------------------------


async def ws_probe():
    url = BASE.replace("http", "ws") + f"/api/v1/s/{SID}/events"
    got = []
    async with websockets.connect(url) as ws:
        got.append(json.loads(await asyncio.wait_for(ws.recv(), 5)))
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(None, lambda: [console("step", note="ws") for _ in range(5)])
        t_end = time.monotonic() + 30
        while time.monotonic() < t_end and len(got) < 6:
            try:
                got.append(json.loads(await asyncio.wait_for(ws.recv(), 10)))
            except asyncio.TimeoutError:
                break
        await fut
    # abrupt disconnects: open and drop 20 sockets
    for _ in range(20):
        w = await websockets.connect(url)
        await w.recv()
        w.transport.abort() if hasattr(w, "transport") else await w.close()
    return [((g.get("data") or {}).get("type"), (g.get("data") or {}).get("seq")) for g in got]


try:
    OUT["websocket"] = asyncio.run(ws_probe())
except Exception as e:  # noqa: BLE001
    OUT["websocket"] = f"{type(e).__name__}: {e}"
r, _ = call("GET", "/hello", note="after ws aborts")
OUT["rss_end_mb"] = rss_mb()
codes = {}
for e in OUT["endpoints"]:
    codes[str(e.get("status"))] = codes.get(str(e.get("status")), 0) + 1
OUT["status_counts"] = codes
OUT["n_500"] = sum(1 for e in OUT["endpoints"] if e.get("status") == 500)
OUT["n_none"] = sum(1 for e in OUT["endpoints"] if e.get("status") is None)
print("HTTP_JSON " + json.dumps({k: v for k, v in OUT.items() if k != "endpoints"}, default=str))

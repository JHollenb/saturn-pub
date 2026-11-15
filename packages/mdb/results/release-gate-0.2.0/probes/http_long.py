"""Release-gate 0.2.0, part 2b: long op (sweep) interleaving, interrupt, disconnect, memory growth.

Usage: http_long.py BASE_URL SERVER_PID
"""
import json
import subprocess
import sys
import threading
import time

import httpx

BASE, PID = sys.argv[1].rstrip("/"), int(sys.argv[2])
API = BASE + "/api/v1"
SID = httpx.get(API + "/hello").json()["data"]["live"]
OUT = {}
PROMPT = "The Eiffel Tower is located in the city of"
DONOR = "The Colosseum is located in the city of"
SWEEP = 'sweep m rows=2:11 layers=0:24 = donor2 --consumer M[" Paris"," Rome"]'


def rss_mb():
    return round(int(subprocess.run(["ps", "-o", "rss=", "-p", str(PID)], capture_output=True,
                                    text=True).stdout.strip() or 0) / 1024, 1)


def con(line, timeout=600):
    t = time.monotonic()
    r = httpx.post(API + f"/s/{SID}/console", json={"line": line}, timeout=timeout)
    js = r.json()
    text = ((js.get("data") or {}).get("text") or js.get("reason") or "")
    print(f"CON {r.status_code} {(time.monotonic() - t) * 1000:8.1f}ms {line[:70]!r} -> {text[:160]!r}", flush=True)
    return r.status_code, text, (time.monotonic() - t) * 1000


con(f"start name=donor2 prefill=2 {DONOR}")
con(f"start name=base2 prefill=2 {PROMPT}")
con("checkpoint b2")
_, text, ms = con(SWEEP)
OUT["sweep_ms"] = ms
OUT["sweep_head"] = text[:300]

# a request while the sweep runs
con("restart b2")
res = {}
th = threading.Thread(target=lambda: res.update(sweep=con(SWEEP)))
th.start()
time.sleep(1.0)
t = time.monotonic()
h = httpx.get(API + "/hello", timeout=600)
hello_ms = (time.monotonic() - t) * 1000
t = time.monotonic()
lv = httpx.get(API + f"/s/{SID}/live", timeout=600)
live_ms = (time.monotonic() - t) * 1000
th.join()
OUT["during_sweep"] = {"sweep_ms": res["sweep"][2], "hello_ms": hello_ms, "hello_status": h.status_code,
                       "live_ms": live_ms, "live_status": lv.status_code}

# interrupt mid-sweep
con("restart b2")
res = {}
th = threading.Thread(target=lambda: res.update(sweep=con(SWEEP)))
th.start()
time.sleep(1.5)
ir = httpx.post(API + f"/s/{SID}/interrupt", timeout=60).json()
th.join()
_, where, _ = con("where")
_, info, _ = con("info edits")
OUT["interrupt"] = {"interrupt": ir.get("data"), "sweep_status": res["sweep"][0], "sweep_ms": res["sweep"][2],
                    "sweep_text": res["sweep"][1][:300], "where_after": where[:120], "edits_after": info[:200]}

# client disconnect mid-sweep, then check the session
con("restart b2")
t = time.monotonic()
try:
    httpx.post(API + f"/s/{SID}/console", json={"line": SWEEP}, timeout=1.0)
    OUT["disconnect"] = "finished in 1 s"
except httpx.TimeoutException:
    OUT["disconnect"] = "client gave up after 1 s"
_, where, ms = con("where")
OUT["after_disconnect"] = {"where_ms": ms, "where": where[:120]}

# memory growth: repeated restart + 100 steps
con("restart b2")
mem = [rss_mb()]
for i in range(6):
    con("restart b2")
    for _ in range(100):
        httpx.post(API + f"/s/{SID}/console", json={"line": "step"}, timeout=60)
    mem.append(rss_mb())
OUT["rss_restart_100steps_mb"] = mem
mem2 = [rss_mb()]
for i in range(5):
    con("restart b2")
    con(SWEEP)
    mem2.append(rss_mb())
OUT["rss_repeated_sweeps_mb"] = mem2
mem3 = [rss_mb()]
for i in range(5):
    con("until logits")
    con('symbols consumer=M[" Paris"," Rome"] rows=all limit=16')
    con('whatif L12.m = donor2:L12.m consumer=M[" Paris"," Rome"]')
    con("restart b2")
    mem3.append(rss_mb())
OUT["rss_repeated_symbols_whatif_mb"] = mem3
print("LONG_JSON " + json.dumps(OUT, default=str))

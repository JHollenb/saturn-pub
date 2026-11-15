"""Run 1 smoke: does checkpoint/restore avoid recomputing the prefix for an edit at L>=20; arm
costs; cell_sweep cost; capture size. python r1_smoke.py MODEL OUTDIR"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load

MODEL, OUTDIR = sys.argv[1:3]
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
T0 = time.monotonic()
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
L, H, KV = dbg.adapter.layers, dbg.adapter.num_heads, dbg.adapter.num_kv_heads
base = {i["id"]: i for i in json.load(open("candidates.json"))["items"]}
ext = {i["id"]: i for i in json.load(open("candidates_ext.json"))["items"]}
gt = json.load(open("gt.json"))
panel = json.load(open("job1-panel.json"))["panel"]
long_items = sorted([i for i in ext.values() if i["family"] == "long-city"], key=lambda i: -i["n"])
items = [(base[panel[0]["id"]], base[panel[0]["donors"][0]], gt[panel[0]["id"]]["row"]),
         (base[panel[50]["id"]], base[panel[50]["donors"][0]], gt[panel[50]["id"]]["row"]),
         (long_items[0], long_items[1], long_items[0]["row"])]
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    t = time.monotonic()
    out = fn()
    sync()
    return out, time.monotonic() - t


report = []
for item, donor, r0 in items:
    ids, j = item["ids"], len(item["ids"]) - 1
    rec = {"id": item["id"], "n": len(ids)}
    run, rec["construct_s"] = timed(lambda: dbg.run(ids, name="fresh"))
    _, rec["to_logits_s"] = timed(lambda: run.until(f"t{j}.logits"))
    dv = dbg.run(donor["ids"], name="donor").read_cache(20, "v", staged=False)
    rd = min(r0, dv.shape[2] - 1)
    value = dv[:, 6:7, rd:rd + 1].clone()

    def fresh_arm():
        r = dbg.run(ids, name="arm")
        r.set_cell(20, "v", value, rows=[r0], kv=[6])
        r.until(f"t{j}.logits")
        return r.read().reshape(-1).float().cpu()
    fresh_logits, rec["fresh_arm_s"] = timed(fresh_arm)
    # checkpoint at the answer row's L20 entry, then restore per arm
    run2 = dbg.run(ids, name="ck")
    _, rec["to_L20_s"] = timed(lambda: run2.until(f"t{j}.L20.h"))
    cp, rec["checkpoint_s"] = timed(lambda: run2.checkpoint("L20"))
    times, same = [], []
    for _ in range(3):
        def arm():
            run2.restore(cp)
            run2.set_cell(20, "v", value, rows=[r0], kv=[6])
            run2.until(f"t{j}.logits")
            return run2.read().reshape(-1).float().cpu()
        lg, dt = timed(arm)
        times.append(dt)
        same.append(bool(torch.equal(lg, fresh_logits)))
    rec["restore_arm_s"] = times
    rec["restore_equals_fresh"] = same
    # checkpoint at L4 (liveness patches start at layer 4)
    run3 = dbg.run(ids, name="ck4")
    run3.until(f"t{j}.L4.h")
    cp4 = run3.checkpoint("L4")
    v4 = dbg.run(donor["ids"], name="d4").read_cache(8, "v", staged=False)[:, 6:7, rd:rd + 1].clone()

    def arm4():
        run3.restore(cp4)
        run3.set_cell(8, "v", v4, rows=[r0], kv=[6])
        run3.until(f"t{j}.logits")
        return run3.read()
    _, rec["restore_L4_arm_s"] = timed(arm4)
    # cell_sweep: span rows x layers 4..27 x 8 KV (V) from the donor
    span = item["span"]
    dvs = dbg.run(donor["ids"], name="dsw")
    dcache = [dvs.read_cache(x, "v", staged=False) for x in range(L)]
    arms = [[Cell(x, "v", r, g, dcache[x][0, g, min(r, dcache[x].shape[2] - 1)])]
            for r in span for x in range(4, L) for g in range(KV)]
    base_run = dbg.run(ids, name="sweep")
    out, dt = timed(lambda: cell_sweep(base_run, arms, tokens=[item["answer"], donor["answer"]], batch=96))
    rec["sweep_arms"], rec["sweep_s"] = len(arms), dt
    # capture
    cap = {}
    t = time.monotonic()
    for x in range(L):
        cap[f"k{x}"] = run.read_cache(x, "k", staged=False).float().cpu().numpy().astype(np.float16)
        cap[f"v{x}"] = run.read_cache(x, "v", staged=False).float().cpu().numpy().astype(np.float16)
        for port in ("h", "p", "A", "o", "r", "n", "g", "sg", "u", "f", "m", "h_out"):
            cap[f"L{x}.{port}"] = run.read(f"t{j}.L{x}.{port}").float().cpu().numpy()
    path = Path(OUTDIR) / f"{item['id'].replace(':', '_').replace(' ', '_')}.npz"
    np.savez_compressed(path, **cap)
    rec["capture_s"] = round(time.monotonic() - t, 2)
    rec["capture_bytes"] = path.stat().st_size
    rec["capture_bytes_raw"] = int(sum(a.nbytes for a in cap.values()))
    report.append(rec)
    print("R1S " + json.dumps(rec), flush=True)
json.dump(report, open("r1-smoke.json", "w"), indent=1)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)

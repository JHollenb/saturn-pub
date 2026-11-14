"""Job A1 (prereg_scale.json): native single-cell donor sweep and the V-band arm, per item.

python sweep.py MODEL_DIR CAPDIR OUT_PREFIX [PICK a-b]
Per item: cell_sweep from the native prefill cache over every (span row with an aligned donor row,
layer, kv) V cell, value = the donor's captured V at the aligned row (dose 1); plus one V-band arm
(all those cells at once). Logits at [a, b] (a = answer, b = donor answer) and the full-vocab
argmax. dM = (l_a - l_b) - (null_a - null_b) with the chunk's own null arm.
Writes OUT_PREFIX.pt (id -> {"cells": (N,3), "dM": (N,), "argmax": (N,)}) and OUT_PREFIX.json
(band arm, null_vs_stepped, seconds). Log: ids and seconds only.
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load

MODEL, CAPDIR, OUTP = sys.argv[1:4]
PICK = sys.argv[4] if len(sys.argv) > 4 else ""
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
ad = dbg.adapter
L, KV = ad.layers, ad.num_kv_heads
P = json.load(open("panel.json"))
items = P["items"]
if PICK:
    items = items[int(PICK.split("-")[0]):int(PICK.split("-")[1])]
by_id = {i["id"]: i for i in P["items"]}
BATCH = int(__import__("os").environ.get("SWEEP_BATCH", "64"))
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def load_v(iid):
    return torch.load(Path(CAPDIR) / "cap" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)["v"]


out_t, out_j = {}, {}
for item in items:
    t = time.monotonic()
    don = by_id[item["donor"]]
    dv = load_v(item["donor"])
    last, a, b = len(item["ids"]) - 1, item["answer"], don["answer"]
    pairs = list(zip(reversed(item["span"]), reversed(don["span"])))
    cells, arms = [], []
    for x in range(L):
        for r, dr in pairs:
            for k in range(KV):
                cells.append((x, k, r))
                arms.append([Cell(x, "v", r, k, dv[x][k, dr])])
    arms.append([c[0] for c in arms])                                   # V band
    base = dbg.run(item["ids"], name="sweep")
    res = cell_sweep(base, arms, tokens=[a, b], batch=BATCH)
    lg, nl = res["logits"], res["null"]
    chunk = torch.arange(len(arms)) // BATCH
    dM = (lg[:, 0] - lg[:, 1]) - (nl[chunk, 0] - nl[chunk, 1])
    out_t[item["id"]] = {"cells": torch.tensor(cells), "dM": dM[:-1].clone(), "argmax": res["argmax"][:-1].clone(),
                         "la": lg[:-1, 0].clone(), "lb": lg[:-1, 1].clone()}
    out_j[item["id"]] = {"band_dM": float(dM[-1]), "band_argmax": int(res["argmax"][-1]),
                         "null_M": float(nl[0, 0] - nl[0, 1]), "null_spread": float((nl[:, 0] - nl[:, 1]).max() - (nl[:, 0] - nl[:, 1]).min()),
                         "n_cells": len(cells), "seconds": round(time.monotonic() - t, 2)}
    del base, res, dv
    print("A1 " + json.dumps({"id": item["id"], "s": out_j[item["id"]]["seconds"]}), flush=True)
torch.save(out_t, OUTP + ".pt")
json.dump(out_j, open(OUTP + ".json", "w"), indent=1)
print(f"TOTAL {time.monotonic() - T0:.1f}s peak {round(torch.cuda.max_memory_allocated() / 2**20)} MB", flush=True)

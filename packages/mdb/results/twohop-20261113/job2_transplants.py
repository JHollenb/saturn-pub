"""Job 2 (interventions): site sweep (ground truth), K vs V at the forecast and ground-truth
sites, bands, same-bridge controls, self-transplant, noise floor.

python job2_transplants.py MODEL candidates.json job1-panel.json job1-forecast.json OUT START STOP

For recipients START..STOP-1 of the panel. Two measurement paths, both native:
- batched (``mdb.analysis.sweep.cell_sweep``): every single cell (r in span, l, g) as V and as K
  from each different-bridge donor, V from the same-bridge control; one null arm per chunk.
- stepped (``Run.set_cell`` on a fresh run, then ``until logits``): the headline arms.
Tokens kept per arm: the recipient's answer a, the donors' answers b_d, the control's answer;
plus the argmax.
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load

MODEL, CAND, PANEL, FORE, OUT = sys.argv[1:6]
START, STOP = int(sys.argv[6]), int(sys.argv[7])
BATCH = 96
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
L, KV = dbg.adapter.layers, dbg.adapter.num_kv_heads
cand = {i["id"]: i for i in json.load(open(CAND))["items"]}
panel = json.load(open(PANEL))["panel"][START:STOP]
fore = {f["id"]: f for f in json.load(open(FORE))["forecasts"]}
print(f"load {time.monotonic() - T0:.1f}s; recipients {START}..{STOP - 1} ({len(panel)})",
      flush=True)
BANDS = (0, 4, 8, 12, 16, 20, 24)
OUTF = open(f"{OUT}.jsonl", "w")


def stepped(ids, cells=(), name="arm"):
    """A fresh run, cell edits staged at the pending row's first port, run to logits."""
    run = dbg.run(ids, name=name)
    for layer, t, value, rows, kv in cells:
        run.set_cell(layer, t, value, rows=rows, kv=kv)
    run.until(f"t{len(ids) - 1}.logits")
    return run.read().reshape(-1).float().cpu()


def cache_of(ids, name):
    """K and V caches (all layers) of a run's prefill rows, and its native logits."""
    run = dbg.run(ids, name=name)
    k = [run.read_cache(layer, "k", staged=False).clone() for layer in range(L)]
    v = [run.read_cache(layer, "v", staged=False).clone() for layer in range(L)]
    run.until(f"t{len(ids) - 1}.logits")
    return k, v, run.read().reshape(-1).float().cpu()


for index, entry in enumerate(panel):
    t_item = time.monotonic()
    item = cand[entry["id"]]
    ids, span, r0, n, a = item["ids"], item["span"], item["row"], item["n"], item["answer"]
    donors = [cand[d] for d in entry["donors"]]
    control = cand[entry["control"]] if entry["control"] else None
    f = fore[item["id"]]
    toks = [a] + [d["answer"] for d in donors] + ([control["answer"]] if control else [])
    rk, rv, native = cache_of(ids, "rec")
    dk, dv, dnat = zip(*[cache_of(d["ids"], f"d{k}") for k, d in enumerate(donors)])
    rec = {"id": item["id"], "tokens": toks, "native": native[toks].tolist(),
           "native_argmax": int(native.argmax()),
           "donor_prompt": [x[toks].tolist() for x in dnat]}
    # ---------------- batched site sweep
    arms, keys = [], []
    for k, d in enumerate(donors):
        for t, cache in (("v", dv[k]), ("k", dk[k])):
            for r in span:
                for layer in range(L):
                    for g in range(KV):
                        arms.append([Cell(layer, t, r, g, cache[layer][0, g, r])])
                        keys.append((f"d{k}", t, r, layer, g))
    if control is not None:
        ck, cv, cnat = cache_of(control["ids"], "ctl")
        rec["control_prompt"] = cnat[toks].tolist()
        rows = span if entry["control_aligned"] else [r0]
        for r in rows:
            src = r if entry["control_aligned"] else control["row"]
            for layer in range(L):
                for g in range(KV):
                    arms.append([Cell(layer, "v", r, g, cv[layer][0, g, src])])
                    keys.append(("ctl", "v", r, layer, g))
    run = dbg.run(ids, name="sweep")
    t = time.monotonic()
    out = cell_sweep(run, arms, tokens=toks, batch=BATCH, stepped_logits=native)
    sweep_s = time.monotonic() - t
    rec["sweep"] = {"keys": keys, "logits": out["logits"].tolist(),
                    "argmax": out["argmax"].tolist(), "null": out["null"].tolist(),
                    "null_vs_stepped": out["null_vs_stepped"], "seconds": sweep_s,
                    "arms": len(arms)}
    # ground truth: V-only from the different-bridge donors, mean dM over donors (vs chunk null)
    null = out["null"][0]
    eff = {}
    for key, lg in zip(keys, out["logits"]):
        if key[0].startswith("d") and key[1] == "v":
            k = int(key[0][1:])
            dm = float((lg[0] - lg[1 + k]) - (null[0] - null[1 + k]))
            eff.setdefault(key[2:], []).append(dm)
    mean = {site: sum(v) / len(v) for site, v in eff.items()}
    gt = min(mean, key=mean.get)
    rec["gt"] = {"row": gt[0], "layer": gt[1], "kv": gt[2], "effect": mean[gt]}
    sites = {"F1": (f["F1"]["row"], f["F1"]["layer"], f["F1"]["kv"]), "GT": gt}
    if f.get("F2"):
        sites["F2"] = (f["F2"]["row"], f["F2"]["layer"], f["F2"]["kv"])
    # ---------------- stepped arms
    arms_s = {}
    t = time.monotonic()
    for label, (r, layer, g) in sites.items():
        if label == "F2" and sites["F2"] in (sites["F1"], sites["GT"]):
            continue
        for k in range(len(donors)):
            vk, kk = dv[k][layer][:, g:g + 1, r:r + 1], dk[k][layer][:, g:g + 1, r:r + 1]
            arms_s[f"{label}.V.d{k}"] = stepped(ids, [(layer, "v", vk, [r], [g])])[toks].tolist()
            arms_s[f"{label}.K.d{k}"] = stepped(ids, [(layer, "k", kk, [r], [g])])[toks].tolist()
            arms_s[f"{label}.KV.d{k}"] = stepped(ids, [(layer, "v", vk, [r], [g]),
                                                       (layer, "k", kk, [r], [g])])[toks].tolist()
        if control is not None and label != "F2":
            src = r if entry["control_aligned"] else control["row"]
            if entry["control_aligned"] or r == r0:
                cvv = cv[layer][:, g:g + 1, src:src + 1]
                arms_s[f"{label}.V.ctl"] = stepped(ids, [(layer, "v", cvv, [r], [g])])[toks].tolist()
    # self-transplant: the recipient's own cell at the GT site (a no-op by construction)
    r, layer, g = gt
    arms_s["self.V"] = stepped(ids, [(layer, "v", rv[layer][:, g:g + 1, r:r + 1], [r], [g])])[toks].tolist()
    arms_s["native_again"] = stepped(ids)[toks].tolist()
    span_t = torch.tensor(span)
    for start in BANDS:
        for k in range(len(donors)):
            for kind in ("V", "K", "KV"):
                cells = []
                for layer in range(start, L):
                    if "V" in kind:
                        cells.append((layer, "v", dv[k][layer][:, :, span_t], span, None))
                    if "K" in kind:
                        cells.append((layer, "k", dk[k][layer][:, :, span_t], span, None))
                arms_s[f"band{start}.{kind}.d{k}"] = stepped(ids, cells)[toks].tolist()
    if control is not None:
        cells = []
        for layer in range(8, L):
            if entry["control_aligned"]:
                cells.append((layer, "v", cv[layer][:, :, span_t], span, None))
            else:
                cells.append((layer, "v", cv[layer][:, :, control["row"]:control["row"] + 1], [r0],
                              None))
        arms_s["band8.V.ctl"] = stepped(ids, cells)[toks].tolist()
    rec["stepped"] = arms_s
    rec["stepped_seconds"] = time.monotonic() - t
    rec["seconds"] = time.monotonic() - t_item
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    summ = {k: v for k, v in rec.items() if k not in ("sweep", "stepped")}
    summ["F1"] = sites["F1"]
    summ["null_vs_stepped"] = out["null_vs_stepped"]
    summ["sweep_seconds"] = round(sweep_s, 2)
    summ["arms"] = len(arms)
    print("ITEM " + json.dumps(summ), flush=True)
    print(f"T {time.monotonic() - T0:.1f}s", flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)

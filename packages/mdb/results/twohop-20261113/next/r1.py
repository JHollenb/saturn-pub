"""Run 1 (next/prereg_run1.json): capture once, native arms by checkpoint/restore and cell_sweep.

python r1.py MODEL MODE OUTDIR [PICK]     MODE = 1a (base + depth capture) | 1b (hard)

Writes OUTDIR/cap/<group>/<id>.pt (native captures + pre-edit operands; the offline forecast
reads only these), OUTDIR/arm/<id>.pt (edited-run captures) and ./arms-<MODE>.jsonl (native arm
results). Forecasts are sealed from cap/ before arms are opened.
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load

MODEL, MODE, OUTDIR = sys.argv[1:4]
PICK = sys.argv[4] if len(sys.argv) > 4 else ""
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
ad = dbg.adapter
L, H, KV = ad.layers, ad.num_heads, ad.num_kv_heads
LAYER, GROUP, NEG = 20, 6, -1e4
spec = json.load(open("r1_items.json"))
groups = ("base", "depth") if MODE == "1a" else ("hard",)
items = [i for i in spec["items"] if i["group"] in groups]
if PICK:
    items = [items[int(k)] for k in PICK.split(",")]
OUT = Path(OUTDIR)
for sub in ("cap/base", "cap/depth", "cap/hard", "arm"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)
ARMS = open(f"arms-{MODE}{'-smoke' if PICK else ''}.jsonl", "w")
W_U = ad.model.lm_head.weight
final_norm = getattr(ad.backbone, ad.spec.final_norm)
blocks = getattr(ad.backbone, ad.spec.layers)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def cpu(x, f32=False):
    x = x.detach()
    return (x.float() if f32 else x).cpu().clone()


def capture(run, j, ids, extra=None):
    cap = {"ids": torch.tensor(ids), "j": j}
    for x in range(L):
        cap[f"k{x}"] = cpu(run.read_cache(x, "k", staged=False))
        cap[f"v{x}"] = cpu(run.read_cache(x, "v", staged=False))
        ports = ("h", "p", "A", "o", "r", "h_out", "m") + (("n", "g", "sg", "u", "f") if x >= LAYER else ())
        for port in ports:
            cap[f"L{x}.{port}"] = cpu(run.read(f"t{j}.L{x}.{port}"), True)
    cap["logits"] = cpu(run.read(f"t{j}.logits"), True).reshape(-1)
    if extra:
        cap.update(extra)
    return cap


def native(ids, name="nat"):
    run = dbg.run(ids, name=name)
    j = len(ids) - 1
    run.until(f"t{j}.logits")
    return run, j


def kept(lg, toks):
    lg = lg.reshape(-1).float()
    return {"logits": [float(lg[t]) for t in toks], "argmax": int(lg.argmax()),
            "argmax_text": tok.decode([int(lg.argmax())])}


def relay_row(p_by_layer, j, span):
    mass = sum(p.reshape(H, -1)[:, :j + 1].sum(0) for p in p_by_layer)
    order = mass.argsort(descending=True).tolist()
    return next(r for r in order if r not in (0, j) and r not in span)


def cut_arm(ck, run, j, rows_by_layer, kind, capture_o=True):
    run.restore(ck)
    for x in range(L):
        run.until(f"t{j}.L{x}.h")
        port = "p" if kind == "zero" else "s"
        shape = tuple(run.address_shape(f"t{j}.L{x}.{port}"))
        mask = torch.zeros(shape, dtype=torch.bool)
        for h, rows in rows_by_layer[x].items():
            mask[0, h, 0, rows] = True
        run.set(f"t{j}.L{x}.{port}", 0.0 if kind == "zero" else NEG, "replace", mask=mask)
    run.until(f"t{j}.logits")
    o = {f"L{x}.o": cpu(run.read(f"t{j}.L{x}.o"), True) for x in range(L)} if capture_o else {}
    return run.read(), o


def edited_ports(run, j):
    out = {}
    for x in range(LAYER, L):
        for port in ("h", "o", "r", "n", "g", "sg", "u", "f", "m", "h_out"):
            out[f"L{x}.{port}"] = cpu(run.read(f"t{j}.L{x}.{port}"), True)
    out["logits"] = cpu(run.read(f"t{j}.logits"), True).reshape(-1)
    return out


# ---------------------------------------------------------------- base pre-pass (means)
cellc = {}


def cell_of(ids, row):
    key = (tuple(ids), row)
    if key not in cellc:
        cellc[key] = dbg.run(ids, name="cell").read_cache(LAYER, "v", staged=False)[:, GROUP:GROUP + 1,
                                                                                  row:row + 1].clone()
    return cellc[key]


def mean_cells(cells):
    return torch.stack([c.float() for c in cells]).mean(0).to(cells[0].dtype)


if MODE == "1a":
    allbase = [i for i in spec["items"] if i["group"] == "base"]
    t = time.monotonic()
    PANEL_MEAN = mean_cells([cell_of(i["ids"], i["row"]) for i in allbase])
    by_country = defaultdict(list)
    for m in spec["means"]["correct_base"]:
        by_country[m["bridge"]].append(m)
    fsum = {x: 0 for x in range(LAYER, L)}
    for i in allbase:
        run, j = native(i["ids"], "fmean")
        for x in range(LAYER, L):
            fsum[x] = fsum[x] + run.read(f"t{j}.L{x}.f").float()
    FMEAN = {x: (fsum[x] / len(allbase)).to(ad.dtype) for x in fsum}
    del run
    print(f"pre-pass {time.monotonic() - t:.1f}s", flush=True)
print(f"load {time.monotonic() - T0:.1f}s; {len(items)} items", flush=True)
gamma = final_norm.weight.detach().float()


def answer_dir(a, b):
    return (gamma * (W_U[a].float() - W_U[b].float()))


for k, item in enumerate(items):
    t_item = time.monotonic()
    ids, j, g = item["ids"], len(item["ids"]) - 1, item["group"]
    run, _ = native(ids)
    if g == "depth":
        torch.save(capture(run, j, ids), OUT / "cap/depth" / f"{safe(item['id'])}.pt")
        ARMS.write(json.dumps({"id": item["id"], "group": g, "seconds": round(time.monotonic() - t_item, 2)}) + "\n")
        continue
    a, d = item["answer"], item["donor"]
    toks = [a, d["answer"]]
    r0, span = item["row"], item["span"]
    # donor operands (pre-edit)
    drun = dbg.run(d["ids"], name="donor")
    dv = [drun.read_cache(x, "v", staged=False) for x in range(L)]
    jd = len(d["ids"]) - 1
    drun.until(f"t{jd}.L{LAYER}.h")
    dh20 = drun.read(f"t{jd}.L{LAYER}.h")
    same_len = jd == j
    p_layers = [run.read(f"t{j}.L{x}.p") for x in range(L)]
    relay = relay_row(p_layers, j, span)
    extra = {"donor_v": {x: cpu(dv[x][:, :, :jd]) for x in range(L)}, "donor_h20": cpu(dh20, True),
             "relay": relay, "toks": torch.tensor(toks)}
    rec = {"id": item["id"], "group": g, "family": item["family"], "relay": relay,
           "same_len_donor": same_len, "native": kept(run.read(f"t{j}.logits"), toks)}
    # liveness sweep (+ relay cells and band)
    rows = list(span) + [relay]
    keys, arms = [], []
    for r in rows:
        rd = min(r, dv[0].shape[2] - 1)
        for x in range(4, L):
            for gg in range(KV):
                keys.append(["cell", r, x, gg])
                arms.append([Cell(x, "v", r, gg, dv[x][0, gg, rd])])
    rd = min(relay, dv[0].shape[2] - 1)
    keys.append(["relay_band", relay, -1, -1])
    arms.append([Cell(x, "v", relay, gg, dv[x][0, gg, rd]) for x in range(L) for gg in range(KV)])
    sweep_run = dbg.run(ids, name="sweep")
    sw = cell_sweep(sweep_run, arms, tokens=toks, batch=96)
    null = sw["null"]
    eff = [float((sw["logits"][i, 0] - sw["logits"][i, 1]) - (null[i // 96, 0] - null[i // 96, 1]))
           for i in range(len(arms))]
    rec["sweep"] = {"keys": keys, "dM": eff, "argmax": sw["argmax"].tolist(),
                    "null_vs_stepped": sw["null_vs_stepped"]}
    # coalitions of the top 4 single cells
    singles = [i for i, kk in enumerate(keys) if kk[0] == "cell"]
    top = sorted(singles, key=lambda i: -abs(eff[i]))[:4]
    pairs = [(x, y) for n_, x in enumerate(top) for y in top[n_ + 1:]]
    if g == "base" and pairs:
        pw = cell_sweep(dbg.run(ids, name="pairs"), [arms[x] + arms[y] for x, y in pairs], tokens=toks, batch=96)
        rec["coalition"] = [{"cells": [keys[x], keys[y]], "single": [eff[x], eff[y]],
                             "pair": float((pw["logits"][n_, 0] - pw["logits"][n_, 1])
                                           - (pw["null"][0, 0] - pw["null"][0, 1]))}
                            for n_, (x, y) in enumerate(pairs)]
    # cuts at the answer row: t0 and relay, zero and mask
    ck_run = dbg.run(ids, name="cut")
    ck_run.until(f"t{j}.L0.h")
    ck0 = ck_run.checkpoint("L0")
    armcap = {}
    rec["cuts"] = {}
    for name, row in (("t0", 0), ("relay", relay)):
        rows_by_layer = {x: {h: [row] for h in range(H)} for x in range(L)}
        for kind in ("zero", "mask"):
            lg, o = cut_arm(ck0, ck_run, j, rows_by_layer, kind)
            rec["cuts"][f"{name}_{kind}"] = kept(lg, toks)
            armcap.update({f"{name}_{kind}.{kk}": v for kk, v in o.items()})
    if g == "base":
        ck20_run = dbg.run(ids, name="ck20")
        ck20_run.until(f"t{j}.L{LAYER}.h")
        ck20 = ck20_run.checkpoint("L20")
        members = [m for m in by_country[item["bridge"]] if m["id"] != item["id"]]
        others = sorted(c for c, v in by_country.items() if c != item["bridge"] and len(v) >= 2)
        other = random.Random(k).choice(others)
        values = {"panel_mean": PANEL_MEAN, "donor0": cell_of(d["ids"], min(r0, jd - 1)),
                  "other_country_mean": mean_cells([cell_of(m["ids"], m["row"]) for m in by_country[other]])}
        if members:
            values["country_mean"] = mean_cells([cell_of(m["ids"], m["row"]) for m in members])
        if item["family"] == "city":
            values["city_family_mean"] = mean_cells([cell_of(m["ids"], m["row"]) for c in by_country.values()
                                                     for m in c if m["family"] == "city" and m["id"] != item["id"]])
        extra["swap_values"] = {n_: cpu(v) for n_, v in values.items()}
        extra["other_country"] = other
        rec["swaps"] = {}
        for n_, v in values.items():
            ck20_run.restore(ck20)
            ck20_run.set_cell(LAYER, "v", v, rows=[r0], kv=[GROUP])
            ck20_run.until(f"t{j}.logits")
            rec["swaps"][n_] = kept(ck20_run.read(), toks)
            armcap.update({f"swap_{n_}.{kk}": vv for kk, vv in edited_ports(ck20_run, j).items()})
        rec["n_country_members"] = len(members)
        rec["other_country"] = other
        # 2x2 payload x receiver
        rec["two_by_two"] = {"A": rec["swaps"]["donor0"]}
        if same_len:
            for n_, use_cell in (("B", False), ("AB", True)):
                ck20_run.restore(ck20)
                ck20_run.set(f"t{j}.L{LAYER}.h", dh20.to(ad.dtype), "replace")
                if use_cell:
                    ck20_run.set_cell(LAYER, "v", values["donor0"], rows=[r0], kv=[GROUP])
                ck20_run.until(f"t{j}.logits")
                rec["two_by_two"][n_] = kept(ck20_run.read(), toks)
        # knock-outs
        named = spec["named_neurons"].get(item["bridge"])
        if named:
            per_layer = defaultdict(list)
            for nm in named:
                lay, f_ = nm.split(".")
                per_layer[int(lay[1:])].append(int(f_[1:]))
            rng = random.Random(1000 + k)
            wdir = answer_dir(a, d["answer"])
            sets = {"named": per_layer, "random": {}, "answer_null": {}}
            dff = blocks[LAYER].mlp.down_proj.weight.shape[1]
            for x, idx in per_layer.items():
                pool = [i for i in range(dff) if i not in set(idx)]
                sets["random"][x] = rng.sample(pool, len(idx))
                score = (wdir @ blocks[x].mlp.down_proj.weight.float()).abs()
                order = [int(i) for i in score.argsort(descending=True) if int(i) not in set(idx)]
                sets["answer_null"][x] = rng.sample(order[:5 * len(idx)], len(idx))
            extra["knockout_sets"] = {n_: {x: list(v) for x, v in s.items()} for n_, s in sets.items()}
            extra["fmean"] = {x: cpu(FMEAN[x], True) for x in FMEAN}
            rec["knockout"] = {}
            for n_, s in sets.items():
                ck20_run.restore(ck20)
                for x in range(LAYER, L):
                    ck20_run.until(f"t{j}.L{x}.h")
                    if not s.get(x):
                        continue
                    shape = tuple(ck20_run.address_shape(f"t{j}.L{x}.f"))
                    mask = torch.zeros(shape, dtype=torch.bool)
                    mask[..., s[x]] = True
                    ck20_run.set(f"t{j}.L{x}.f", FMEAN[x].reshape(shape), "replace", mask=mask)
                ck20_run.until(f"t{j}.logits")
                rec["knockout"][n_] = kept(ck20_run.read(), toks)
            rec["knockout_count"] = len(named)
    torch.save(capture(run, j, ids, extra), OUT / f"cap/{g}" / f"{safe(item['id'])}.pt")
    torch.save(armcap, OUT / "arm" / f"{safe(item['id'])}.pt")
    rec["seconds"] = round(time.monotonic() - t_item, 2)
    ARMS.write(json.dumps(rec) + "\n")
    ARMS.flush()
    # custody: arm results go only to the arms file, never to the log
    print("R1 " + json.dumps({"id": item["id"], "relay": relay, "arms": len(arms),
                              "s": rec["seconds"]}), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)

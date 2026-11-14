"""Job A2 (prereg_scale.json): stepped and closure arms, after A1 fixed the carriers.

python arms.py MODEL_DIR CAPDIR SWEEP_PREFIXES CAPINFO_JSONS OUT PART
  SWEEP_PREFIXES / CAPINFO_JSONS: comma lists (A1 outputs, Job C capture-*.json)
  PART split: (c) donor transplant at the item carrier, doses 0.5 and 1.0, hop_edge(neurons=False)
              on M, correct base items
  PART rest:  closure (same-country / donor-country mean cell at the model carrier, cell_sweep);
              B3 back-patch on wrong d3/d4; B6 answer-row MLP mean-ablation at L_c..L_c+2 on
              base + depth; dist: true carrier cell written into the distractor rows (wrong items)
Log: ids and seconds only.
"""
import json
import sys
import time

import torch

from mdb.analysis import symbols
from mdb.analysis.forecast import Margin
from mdb.analysis.reads import readout
from mdb.analysis.sweep import Cell, cell_sweep
from mdb.repl.__main__ import _load
import scale_common as sc

MODEL, CAPDIR, SWP, CINF, OUT, PART = sys.argv[1:7]
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(__import__("pathlib").Path(MODEL), "bf16", "cuda", True, None)
ad = dbg.adapter
L, KV = ad.layers, ad.num_kv_heads
P = json.load(open("panel.json"))
by_id = {i["id"]: i for i in P["items"]}
capinfo = sc.capture_info(CINF.split(","))
correct = sc.correct_map(P, capinfo)
sweep = sc.load_sweep(SWP.split(","))
car = sc.carriers(P, sweep, correct)
LC, KC = car["L_c"], car["kv_c"]
_pick = __import__("os").environ.get("A2_PICK", "")
ITEMS = [P["items"][int(i)] for i in _pick.split(",")] if _pick else P["items"]
res = {"L_c": LC, "kv_c": KC, "part": PART, "items": {}}
print(f"load {time.monotonic() - T0:.1f}s", flush=True)


def bf(x):
    return x.to(torch.bfloat16).cuda()


def logits_at(run, last, toks):
    lg = run.read(f"t{last}.logits").reshape(-1).float()
    return [float(lg[t]) for t in toks], int(lg.argmax())


if PART == "split":
    for it in ITEMS:
        if it["block"] != "base" or not correct[it["id"]]:
            continue
        t = time.monotonic()
        iid, ids, last = it["id"], it["ids"], len(it["ids"]) - 1
        x, k, r = car["item"][iid]["cell"]
        don = by_id[it["donor"]]
        dr = dict(zip(reversed(it["span"]), reversed(don["span"])))[r]
        v0 = sc.load_cap(CAPDIR, iid)["v"][x][k, r].float()
        vd = sc.load_cap(CAPDIR, don["id"])["v"][x][k, dr].float()
        a, b = it["answer"], don["answer"]
        nat = dbg.run(ids, name="nat")
        nat.until(f"t{last}.logits")
        reader = readout(nat, Margin(last, a, b))
        rec = {}
        for alpha in (0.5, 1.0):
            ed = dbg.run(ids, name=f"dose{int(alpha * 10)}")
            ed.until(f"t{last}.L{x}.h")
            ed.set_cell(x, "v", bf(v0 + alpha * (vd - v0)).reshape(1, 1, 1, -1), rows=[r], kv=[k])
            ed.until(f"t{last}.logits")
            e = symbols.hop_edge(nat, ed, reader, cell_layer=x, neurons=False)
            rec[str(alpha)] = {"native_delta": e["consumer"]["native_delta"]["value"],
                               "direct": e["direct"]["value"], "mlp": e["totals"]["mlp"]["value"],
                               "reread": e["totals"]["reread"]["value"], "norm": e["norm"]["value"],
                               "rounding": e["rounding"]["value"],
                               "mlp_by_layer": [[m["layer"], m["captured"]["value"]] for m in e["mlp"]],
                               "edited": logits_at(ed, last, [a, b])}
            del ed
        rec["native"] = logits_at(nat, last, [a, b])
        rec["cell"] = [x, k, r]
        res["items"][iid] = rec
        del nat
        print("A2s " + json.dumps({"id": iid, "s": round(time.monotonic() - t, 2)}), flush=True)
else:
    # mean cells at the model carrier
    cell_of = {}
    for it in P["items"]:
        cell_of[it["id"]] = sc.load_cap(CAPDIR, it["id"])["v"][LC][KC, car["row_at_item"][it["id"]]].float()
    by_c = {}
    for it in P["items"]:
        by_c.setdefault(it["bridge"], []).append(it["id"])
    # B6 means: answer-row m at L_c..L_c+2 per block, self-excluded
    LAY = [x for x in range(LC, min(LC + 3, L))]
    msum, mcnt = {}, {}
    mcache = {}
    for it in P["items"]:
        if it["block"] in ("base", "depth"):
            cap = sc.load_cap(CAPDIR, it["id"])
            mcache[it["id"]] = torch.stack([cap["ans"]["m"][x] for x in LAY])
            msum[it["block"]] = msum.get(it["block"], 0) + mcache[it["id"]]
            mcnt[it["block"]] = mcnt.get(it["block"], 0) + 1
    SETK = {"base": "cities", "depth": "languages"}
    for it in ITEMS:
        t = time.monotonic()
        iid, ids, last = it["id"], it["ids"], len(it["ids"]) - 1
        don = by_id[it["donor"]]
        a, b = it["answer"], don["answer"]
        rec = {}
        r_at = car["row_at_item"][iid]
        same = [j for j in by_c[it["bridge"]] if j != iid]
        other = [j for j in by_c[don["bridge"]] if j != iid]
        arms, names = [], []
        if same:
            arms.append([Cell(LC, "v", r_at, KC, torch.stack([cell_of[j] for j in same]).mean(0))])
            names.append("same_country_mean")
        if other:
            arms.append([Cell(LC, "v", r_at, KC, torch.stack([cell_of[j] for j in other]).mean(0))])
            names.append("donor_country_mean")
        if it["family"] == "dist-city" and not correct[iid]:
            drows = it["distractor"]["rows"]
            arms.append([Cell(LC, "v", rr, KC, cell_of[iid]) for rr in drows])
            names.append("true_cell_into_distractor")
        if arms:
            base = dbg.run(ids, name="cl")
            out = cell_sweep(base, arms, tokens=[a, b], batch=8)
            rec["closure"] = {n: {"la": float(out["logits"][i, 0]), "lb": float(out["logits"][i, 1]),
                                  "argmax": int(out["argmax"][i])} for i, n in enumerate(names)}
            rec["closure_null"] = [float(out["null"][0, 0]), float(out["null"][0, 1])]
            rec["same_n"], rec["other_n"] = len(same), len(other)
            del base
        if it["family"] in ("d3-lm", "d3-per", "d4-work") and not correct[iid] and LC - 4 >= 0:
            cap = sc.load_cap(CAPDIR, iid)
            top = min(LC + 4, L - 1)                  # clamped on shallow-tail models (deviation, logged)
            hv = cap["rows"][r_at]["h"][top]
            run = dbg.run(ids, name="bp", prefill=it["span"][0])
            run.until(f"t{r_at}.L{LC - 4}.h")
            run.set(f"t{r_at}.L{LC - 4}.h", bf(hv).reshape(1, 1, -1))
            run.until(f"t{last}.logits")
            rec["backpatch"] = {"row": r_at, "from_layer": top, "logits": logits_at(run, last, [a])[0], "argmax": logits_at(run, last, [a])[1]}
            del run, cap
        if it["block"] in ("base", "depth"):
            blk = it["block"]
            mean = (msum[blk] - mcache[iid]) / (mcnt[blk] - 1)
            run = dbg.run(ids, name="b6")
            for j, x in enumerate(LAY):
                run.until(f"t{last}.L{x}.h")
                run.set(f"t{last}.L{x}.m", bf(mean[j]).reshape(1, 1, -1))
            run.until(f"t{last}.logits")
            lg = run.read(f"t{last}.logits").reshape(-1).float()
            toks = P["sets"][SETK[blk]]
            rest = [tt for tt in toks if tt != a]
            rec["b6"] = {"layers": LAY, "a": float(lg[a]), "max_rest": float(lg[rest].max()),
                         "argmax": int(lg.argmax()), "a_in_set": a in toks}
            del run
        res["items"][iid] = rec
        print("A2r " + json.dumps({"id": iid, "s": round(time.monotonic() - t, 2)}), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print(f"TOTAL {time.monotonic() - T0:.1f}s peak {round(torch.cuda.max_memory_allocated() / 2**20)} MB", flush=True)

"""Job 4: the model's own two-hop errors at the carrier found in job 2.

python job4_errors.py MODEL candidates.json job1-panel.json LAYER KV OUT

For every candidate the model answers wrong (job 1), at the last descriptor row r0:
- naming (derived-local, E9-style): the carrier group's write into the answer row,
  sum_h p[h,r0] W_o,h v[g,r0], read through the final norm gain and the unembedding; rank of the
  true bridge country token and of the country of the wrong answer (when it is a capital);
- repair (stepped, native): the carrier value cell cache.L<l>.v[kv g, row r0] replaced by that
  of a correctly answered item with the same bridge (its last descriptor row; values carry no
  position), and the same for the whole value band L8..27 (all KV heads) at r0. Does the answer
  become the right capital?
- the same naming on every correct candidate (the readout's hit rate out of the panel).
"""
import json
import sys
import time
from pathlib import Path

import torch

from mdb.analysis.reads import head_mix
from mdb.repl.__main__ import _load

MODEL, CAND, PANEL = sys.argv[1:4]
LAYER, G, OUT = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
T0 = time.monotonic()
torch.set_grad_enabled(False)
dbg = _load(Path(MODEL), "bf16", "cuda", True, None)
tok = dbg.tokenizer
L = dbg.adapter.layers
spec = json.load(open(CAND))
cand = {i["id"]: i for i in spec["items"]}
captok = {v: k for k, v in spec["capital_tokens"].items()}
rows = {r["id"]: r for r in json.load(open(PANEL))["rows"]}
countries = sorted({i["country_token"] for i in spec["items"] if i["country_token"]})
norm = getattr(dbg.adapter.backbone, dbg.adapter.spec.final_norm)
W = dbg.adapter.model.lm_head.weight
OUTF = open(f"{OUT}.jsonl", "w")
country_of_bridge = {i["bridge"]: i["country_token"] for i in spec["items"]}


def naming(item, run):
    j, r0 = item["n"] - 1, item["row"]
    mix = head_mix(run, j, LAYER, {})
    write = sum(mix.o_parts()[h, r0] for h in range(mix.p.shape[0]) if mix.kappa[h] == G)
    x = (write * norm.weight.double().cpu()).to(device=W.device, dtype=W.dtype)
    proj = (W @ x).float()
    top = [tok.decode([int(i)]) for i in proj.topk(8).indices]
    out = {"top8": top, "mass": float(sum(mix.p[h, r0] for h in range(mix.p.shape[0])
                                          if mix.kappa[h] == G))}
    ct = item["country_token"]
    if ct is not None:
        out["true_rank"] = int((proj > proj[ct]).sum())
        sub = proj[countries]
        out["true_best_of_countries"] = countries[int(sub.argmax())] == ct
        out["best_country"] = tok.decode([countries[int(sub.argmax())]])
    return out


def stepped(ids, cells=()):
    run = dbg.run(ids, name="arm")
    for layer, value, rows_, kv in cells:
        run.set_cell(layer, "v", value, rows=rows_, kv=kv)
    run.until(f"t{len(ids) - 1}.logits")
    return run.read().reshape(-1).float().cpu()


for item in spec["items"]:
    t = time.monotonic()
    r = rows[item["id"]]
    run = dbg.run(item["ids"], name="main")
    run.until(f"t{item['n'] - 1}.logits")
    lg = run.read().reshape(-1).float().cpu()
    rec = {"id": item["id"], "correct": r["correct"], "argmax_text": r["argmax_text"],
           "native_answer_minus_given": float(lg[item["answer"]] - lg[r["argmax"]]),
           "naming": naming(item, run)}
    wrong_cap = captok.get(r["argmax"])
    if wrong_cap is not None and not r["correct"]:
        wc = country_of_bridge.get(wrong_cap)
        if wc is not None:
            proj_rank = None
            mix = head_mix(run, item["n"] - 1, LAYER, {})
            write = sum(mix.o_parts()[h, item["row"]] for h in range(mix.p.shape[0])
                        if mix.kappa[h] == G)
            x = (write * norm.weight.double().cpu()).to(device=W.device, dtype=W.dtype)
            proj = (W @ x).float()
            proj_rank = int((proj > proj[wc]).sum())
            rec["answer_country"] = wrong_cap
            rec["answer_country_rank"] = proj_rank
    if not r["correct"]:
        donors = [d for d in spec["items"] if d["bridge"] == item["bridge"]
                  and d["id"] != item["id"] and rows[d["id"]]["correct"]]
        rec["repairs"] = []
        for d in donors[:2]:
            drun = dbg.run(d["ids"], name="donor")
            dv = [drun.read_cache(layer, "v", staged=False)[:, :, d["row"]:d["row"] + 1].clone()
                  for layer in range(L)]
            one = stepped(item["ids"], [(LAYER, dv[LAYER][:, G:G + 1], [item["row"]], [G])])
            band = stepped(item["ids"], [(layer, dv[layer], [item["row"]], None)
                                         for layer in range(8, L)])
            rec["repairs"].append({
                "donor": d["id"],
                "carrier_cell": {"argmax": tok.decode([int(one.argmax())]),
                                 "fixed": int(one.argmax()) == item["answer"],
                                 "dM_answer_vs_given": float((one[item["answer"]] - one[r["argmax"]]))},
                "band_v_8_27": {"argmax": tok.decode([int(band.argmax())]),
                                "fixed": int(band.argmax()) == item["answer"]}})
    rec["seconds"] = time.monotonic() - t
    OUTF.write(json.dumps(rec) + "\n")
    OUTF.flush()
    if not r["correct"]:
        print("ERR " + json.dumps(rec), flush=True)
print(f"TOTAL {time.monotonic() - T0:.1f}s", flush=True)

"""Hop-budget B1/B2 (prereg_hopbudget.json) on the 1.7B 1c depth captures.

python hopbudget.py MODEL CAPDIR OUT   (b1_sets.json, b1-score.json, prereg in cwd)
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch

from mdb.analysis import circuit_map as cm
from mdb.analysis.reads import layer_module
from mdb.repl.__main__ import _load

MODEL, CAPDIR, OUT = sys.argv[1:4]
LC = 20
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
L, H, KV = ad.layers, ad.num_heads, ad.num_kv_heads
gamma = getattr(ad.backbone, ad.spec.final_norm).weight.double()
W_U = ad.model.lm_head.weight.detach()
S = json.load(open("b1_sets.json"))
COUNTRIES, T = S["sets"]["countries"], S["truths"]
RO = gamma[None, :] * W_U[COUNTRIES].double()  # (57, d)
WR = {x: cm.group_writers(layer_module(ad, x).self_attn.o_proj.weight, H, KV) for x in range(LC + 1)}
# fold readout into writers once: (KV, 57, dh)
RW = {x: torch.einsum("cd,gde->gce", RO, WR[x]) for x in WR}
del WR
SC = json.load(open("b1-score.json"))["per_item"]
rng = random.Random(0)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


per = {}
for iid in sorted(T):
    t = T[iid]
    if iid not in SC:
        continue
    cap = torch.load(Path(CAPDIR) / "cap" / "depth" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    row = SC[iid]["carrier_row"]
    names = []  # per layer: set of country indices named by the 8 cells
    for x in range(LC + 1):
        v = cap[f"v{x}"][0, :, row].double()  # (KV, dh)
        s = torch.einsum("gce,ge->gc", RW[x], v)
        names.append(set(int(i) for i in s.argmax(-1)))
    del cap
    ti = COUNTRIES.index(t["country"]) if t["country"] in COUNTRIES else None
    fake = rng.choice([i for i in range(len(COUNTRIES)) if i != ti])

    def first(ci):
        return next((x for x in range(LC + 1) if ci in names[x]), None)

    def stable(ci):
        for x in range(LC + 1):
            if all(ci in names[y] for y in range(x, LC + 1)):
                return x
        return None

    lf = first(ti) if ti is not None else None
    per[iid] = {"family": t["family"], "correct": SC[iid]["correct"], "row": row, "L_first": lf,
                "L_first_stable": stable(ti) if ti is not None else None, "L_first_fake": first(fake),
                "truth_in_set": ti is not None}


def code(v):
    return LC + 1 if v is None else v


def med(xs):
    return st.median(xs) if xs else None


def auc(pos, neg):
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


fams = ["d1", "d2", "d3-lm", "d3-per", "d4-work"]
table = {}
for f in fams + ["d3"]:
    rs = [r for r in per.values() if (r["family"] in ("d3-lm", "d3-per") if f == "d3" else r["family"] == f)]
    c = [r for r in rs if r["correct"]]
    w = [r for r in rs if not r["correct"]]
    table[f] = {"n": len(rs), "n_correct": len(c),
                "med_correct": med([code(r["L_first"]) for r in c]),
                "med_wrong": med([code(r["L_first"]) for r in w]),
                "med_all": med([code(r["L_first"]) for r in rs]),
                "med_fake_correct": med([code(r["L_first_fake"]) for r in c]),
                "med_stable_correct": med([code(r["L_first_stable"]) for r in c]),
                "med_stable_wrong": med([code(r["L_first_stable"]) for r in w]),
                "absent_correct": sum(r["L_first"] is None for r in c),
                "absent_wrong": sum(r["L_first"] is None for r in w)}

guard = {f: (table[f]["med_fake_correct"] is not None and table[f]["med_correct"] is not None
             and abs(table[f]["med_fake_correct"] - table[f]["med_correct"]) <= 1) for f in ("d2", "d3", "d4-work")}
m2, m3, m4 = table["d2"]["med_correct"], table["d3"]["med_correct"], table["d4-work"]["med_correct"]
b1 = {"med_d2": m2, "med_d3": m3, "med_d4_correct": m4, "step_23": m3 - m2, "step_34": m4 - m3,
      "pass": (m3 - m2 >= 2) and (m4 - m3 >= 2),
      "falsified_flat": abs(m3 - m2) < 1 and abs(m4 - m3) < 1,
      "chance_dominated": guard}
deep = [r for r in per.values() if r["family"] in ("d3-lm", "d3-per", "d4-work")]
cw = [code(r["L_first"]) for r in deep if r["correct"]]
ww = [code(r["L_first"]) for r in deep if not r["correct"]]
slack = lambda r: -1 if r["L_first"] is None else LC - r["L_first"]
a = auc([slack(r) for r in deep if r["correct"]], [slack(r) for r in deep if not r["correct"]])
a_stable = auc([(-1 if r["L_first_stable"] is None else LC - r["L_first_stable"]) for r in deep if r["correct"]],
               [(-1 if r["L_first_stable"] is None else LC - r["L_first_stable"]) for r in deep if not r["correct"]])
b2 = {"med_correct": med(cw), "med_wrong": med(ww), "locus_gap": med(ww) - med(cw),
      "locus_pass": med(ww) - med(cw) >= 3, "auc_slack": a, "auc_pass": a >= 0.75, "falsified": a < 0.6,
      "n_correct": len(cw), "n_wrong": len(ww), "auc_slack_stable_secondary": a_stable}
json.dump({"prereg": "prereg_hopbudget.json", "L_c": LC, "table": table, "B1": b1, "B2": b2, "per_item": per},
          open(OUT, "w"), indent=1)
print(json.dumps({"B1": b1, "B2": b2}, indent=1))
for f in fams + ["d3"]:
    print(f, table[f])

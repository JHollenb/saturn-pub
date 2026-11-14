"""Audit (independent of score.py / scale_common.py): carriers, KV-index chance, head bias,
write windows, concentration, family agreement, recomputed from the raw A1 sweep tensors.

python audit/a1_carriers.py   (run from results/scale-20261114; reads audit/data/sw-*.pt)
"""
import collections
import json
import math
import random
import statistics as st

import torch

D = "audit/data/"
P = json.load(open("panel.json"))
items = P["items"]
MODELS = {
    "Qwen2.5-0.5B": (["sw-Qwen2.5-0.5B"], ["capture-Qwen2.5-0.5B.json"]),
    "Qwen3-0.6B-Base": (["sw-Qwen3-0.6B-Base"], ["capture-Qwen3-0.6B-Base.json"]),
    "Qwen2.5-1.5B": (["sw-Qwen2.5-1.5B"], ["capture-Qwen2.5-1.5B.json"]),
    "Qwen3-1.7B-Base": (["sw-Qwen3-1.7B-Base"], ["capture-Qwen3-1.7B-Base.json"]),
    "Qwen2.5-3B": (["sw-Qwen2.5-3B-a", "sw-Qwen2.5-3B-b"], ["capture-Qwen2.5-3B-a.json", "capture-Qwen2.5-3B-b.json"]),
    "Qwen3-4B-Base": (["sw-Qwen3-4B-Base-a", "sw-Qwen3-4B-Base-b"], ["capture-Qwen3-4B-Base-a.json", "capture-Qwen3-4B-Base-b.json"]),
}


def floor(la, lb):
    m = max(abs(float(la)), abs(float(lb)), 1e-30)
    return 2 * 2.0 ** (math.floor(math.log2(m)) - 7)


out = {}
for name, (sws, caps) in MODELS.items():
    sw, swj, ci = {}, {}, {}
    for s in sws:
        sw.update(torch.load(D + s + ".pt", weights_only=False))
        swj.update(json.load(open(D + s + ".json")))
    for c in caps:
        ci.update(json.load(open(c))["items"])
    correct = {i["id"]: ci[i["id"]]["argmax"] == i["answer"] for i in items}
    KV = int(max(int(sw[items[0]["id"]]["cells"][:, 1].max()) for _ in [0])) + 1
    L = int(sw[items[0]["id"]]["cells"][:, 0].max()) + 1
    icar = {}
    for it in items:
        s = sw[it["id"]]
        j = int(torch.argmin(s["dM"]))
        x, k, r = (int(v) for v in s["cells"][j])
        icar[it["id"]] = (x, k, r, float(s["dM"][j]), abs(float(s["dM"][j])) > floor(s["la"][j], s["lb"][j]))
    base_ok = [i["id"] for i in items if i["block"] == "base" and correct[i["id"]] and icar[i["id"]][4]]
    cnt = collections.Counter(icar[i][:2] for i in base_ok)
    (LC, KC), n = cnt.most_common(1)[0]
    rec = {"L": L, "KV": KV, "L_c": LC, "kv_c": KC, "mode": [n, len(base_ok)], "top_cells": [[list(k), v] for k, v in cnt.most_common(4)],
           "base_correct_total": sum(correct[i["id"]] for i in items if i["block"] == "base")}
    # out-of-sample: item carrier on correct non-base items (not used to pick the carrier)
    oos = collections.defaultdict(lambda: [0, 0])
    for it in items:
        if it["block"] != "base" and correct[it["id"]] and icar[it["id"]][4]:
            oos[it["family"]][1] += 1
            oos[it["family"]][0] += icar[it["id"]][:2] == (LC, KC)
    rec["oos_agree_by_family"] = dict(oos)
    # head bias: per kv, over ALL items (correct or not), how often is it the argmax kv; and
    # per kv, median over correct base items of the strongest single-cell -dM at any layer/row
    kv_arg = collections.Counter(icar[i["id"]][1] for i in items)
    rec["argmax_kv_hist_all_items"] = [kv_arg.get(k, 0) for k in range(KV)]
    strength = []
    for k in range(KV):
        v = []
        for i in base_ok:
            s = sw[i]
            sel = s["cells"][:, 1] == k
            v.append(float(-s["dM"][sel].min()))
        strength.append(round(st.median(v), 3))
    rec["kv_strength_median_best_cell"] = strength
    # head bias at layers other than L_c: argmax kv restricted to layers != L_c and != L_c-1..L_c+1
    kv_other = collections.Counter()
    for i in base_ok:
        s = sw[i]
        sel = (s["cells"][:, 0] - LC).abs() > 1
        j = torch.nonzero(sel).reshape(-1)[int(s["dM"][sel].argmin())]
        kv_other[int(s["cells"][j, 1])] += 1
    rec["argmax_kv_excluding_Lc_pm1"] = [kv_other.get(k, 0) for k in range(KV)]
    # per-layer argmax kv at L_c: is kv_c the argmax among KV cells at layer L_c on most items?
    at_lc = collections.Counter()
    for i in base_ok:
        s = sw[i]
        sel = s["cells"][:, 0] == LC
        j = torch.nonzero(sel).reshape(-1)[int(s["dM"][sel].argmin())]
        at_lc[int(s["cells"][j, 1])] += 1
    rec["argmax_kv_within_Lc"] = [at_lc.get(k, 0) for k in range(KV)]
    # write windows (at the item carrier row), sealed (any kv) and post hoc (carrier kv)
    ww, wwk, conc, top2, adj = [], [], [], [], []
    fam_agree = collections.defaultdict(lambda: [0, 0])
    for it in items:
        if it["block"] != "base" or not correct[it["id"]]:
            continue
        s = sw[it["id"]]
        x0, k0, r0, _, ab = icar[it["id"]]
        fam_agree[it["family"]][1] += 1
        fam_agree[it["family"]][0] += (x0, k0) == (LC, KC)
        fl = torch.tensor([floor(a, b) for a, b in zip(s["la"].tolist(), s["lb"].tolist())])
        big = s["dM"].abs() > fl
        rowsel = s["cells"][:, 2] == r0
        lay = s["cells"][rowsel & big, 0]
        ww.append(int(lay.max() - lay.min() + 1) if lay.numel() else None)
        layk = s["cells"][rowsel & big & (s["cells"][:, 1] == k0), 0]
        wwk.append(int(layk.max() - layk.min() + 1) if layk.numel() else None)
        band = swj[it["id"]]["band_dM"]
        o = s["dM"].argsort()
        conc.append(abs(float(s["dM"][o[0]])) / abs(band) if band else None)
        top2.append((abs(float(s["dM"][o[0]])) + abs(float(s["dM"][o[1]]))) / abs(band) if band else None)
        c1, c2 = s["cells"][o[0]].tolist(), s["cells"][o[1]].tolist()
        adj.append(c1[2] == c2[2] and abs(c1[0] - c2[0]) == 1)
        # number of rows with >5 layer window: report median span too
    nb = len(ww)
    rec["write_window_le5_anykv"] = [sum(w is not None and w <= 5 for w in ww), nb]
    rec["write_window_le5_carrierkv"] = [sum(w is not None and w <= 5 for w in wwk), nb]
    rec["write_window_median_anykv"] = st.median([w for w in ww if w is not None])
    rec["write_window_median_carrierkv"] = st.median([w for w in wwk if w is not None])
    rec["family_agree_correct_base"] = dict(fam_agree)
    rec["concentration_median"] = st.median([c for c in conc if c is not None])
    rec["top2_adjacent_same_row"] = [sum(adj), len(adj)]
    rec["band_sign_check"] = sum(swj[i["id"]]["band_dM"] < 0 for i in items if i["block"] == "base" and correct[i["id"]])
    out[name] = rec
    print(name, json.dumps({k: v for k, v in rec.items()}, default=str))

# chance of the same KV index at three Qwen3 sizes (8 KV heads each)
q3 = [out[m]["kv_c"] for m in ("Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base")]
print("qwen3 kv_c", q3, "P(all equal | uniform, 8 kv) =", 8 / 8 ** 3, "; P(0.6B and 4B both = 1.7B's index) = 1/64")
# empirical chance using each model's own argmax-kv distribution over all items (bias-aware)
dist = {m: out[m]["argmax_kv_hist_all_items"] for m in ("Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base")}
p = 0.0
for k in range(8):
    term = 1.0
    for m, h in dist.items():
        term *= h[k] / sum(h)
    p += term
print("P(all three argmax-kv draws equal) under each model's own all-item argmax-kv histogram:", round(p, 4))
json.dump(out, open("audit/a1_carriers.json", "w"), indent=1, default=str)

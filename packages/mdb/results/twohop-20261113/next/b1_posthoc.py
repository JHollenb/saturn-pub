"""B1 POST HOC checks (written after b1-score.json was read; not preregistered).

1. Language vs country at the carrier cell: score(truth language) - score(truth country) through
   the same gamma W_U readout, and the best token of the union set countries + languages.
2. The answer row's MLP writes: per layer, does m_l name the language within languages; the
   fraction of layers < 21 vs >= 21 that do, against the 1/26 chance rate.
3. The ladder statistic's chance rate: first naming layer with the truth replaced by a random
   other token of the same set (seed 0), to show what "first layer over every span cell" gives
   by chance.
python b1_posthoc.py MODEL CAPDIR OUT
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
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
L, H, KV = ad.layers, ad.num_heads, ad.num_kv_heads
gamma = getattr(ad.backbone, ad.spec.final_norm).weight.double()
W_U = ad.model.lm_head.weight.detach()
WR = {x: cm.group_writers(layer_module(ad, x).self_attn.o_proj.weight, H, KV) for x in range(L)}
S = json.load(open("b1_sets.json"))
SETS, T = S["sets"], S["truths"]
RO = {k: gamma[None, :] * W_U[v].double() for k, v in SETS.items()}
UNION = SETS["countries"] + SETS["languages"]
RO_U = gamma[None, :] * W_U[UNION].double()
rng = random.Random(0)


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def carrier_row(cap, span):
    p = cap["L20.p"].reshape(H, -1)
    rows = [r for r in span if r != 0]
    return max(rows, key=lambda r: (float(p[12, r] + p[13, r]), r))


per = {}
for iid in sorted(T):
    t = T[iid]
    cap = torch.load(Path(CAPDIR) / "cap" / "depth" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    span = t["span"]
    crow = carrier_row(cap, span)
    w = WR[20][6] @ cap["v20"][0, 6, crow].double()
    sl = float(gamma * W_U[t["language"]].double() @ w)
    sc = float(gamma * W_U[t["country"]].double() @ w)
    ub = UNION[int((RO_U @ w).argmax())]
    m_names = []
    for x in range(L):
        s = RO["languages"] @ cap[f"L{x}.m"].double().reshape(-1)
        m_names.append(SETS["languages"][int(s.argmax())] == t["language"])
    # chance ladder: random wrong truth, first layer any span cell names it
    values = {x: cap[f"v{x}"].double() for x in range(L)}
    cen = cm.cell_census(values, WR, span, {"countries": RO["countries"]})["scores"]["countries"]
    bi = cen.argmax(-1)
    fake = rng.choice([c for c in SETS["countries"] if c != t["country"]])
    hit = torch.tensor(SETS["countries"])[bi] == fake
    first_fake = next((x for x in range(L) if bool(hit[x].any())), None)
    hit_t = torch.tensor(SETS["countries"])[bi] == t["country"]
    per[iid] = {"family": t["family"], "lang_minus_country": sl - sc, "union_best_is_country": ub in SETS["countries"],
                "union_best_is_truth_country": ub == t["country"], "union_best_is_truth_language": ub == t["language"],
                "m_names_lang": m_names, "first_fake_country": first_fake,
                "frac_cells_naming_truth_L0": float(hit_t[0].float().mean()),
                "frac_cells_naming_fake_L0": float(hit[0].float().mean())}
    del cap, values, cen
fams = ("d2", "d3-lm", "d3-per", "d4-work")
rows = [r for r in per.values() if r["family"] in fams]
res = {
    "carrier_language_gt_country": [sum(r["lang_minus_country"] > 0 for r in rows), len(rows)],
    "lang_minus_country_median": st.median(r["lang_minus_country"] for r in rows),
    "union_best": {"truth_country": sum(r["union_best_is_truth_country"] for r in rows),
                   "truth_language": sum(r["union_best_is_truth_language"] for r in rows),
                   "any_country": sum(r["union_best_is_country"] for r in rows), "n": len(rows)},
    "m_names_lang_rate_lt21": st.mean(sum(r["m_names_lang"][:21]) / 21 for r in rows),
    "m_names_lang_rate_ge21": st.mean(sum(r["m_names_lang"][21:]) / 7 for r in rows),
    "m_names_lang_by_layer": [sum(r["m_names_lang"][x] for r in rows) for x in range(L)],
    "chance_rate": 1 / len(SETS["languages"]),
    "first_fake_country_median": st.median(r["first_fake_country"] for r in rows if r["first_fake_country"] is not None),
    "first_fake_country_at_L0": sum(r["first_fake_country"] == 0 for r in rows),
    "L0_frac_cells_naming_truth": st.mean(r["frac_cells_naming_truth_L0"] for r in rows),
    "L0_frac_cells_naming_fake": st.mean(r["frac_cells_naming_fake_L0"] for r in rows),
    "n": len(rows)}
json.dump({"summary": res, "per_item": per}, open(OUT, "w"), indent=1)
print(json.dumps(res))

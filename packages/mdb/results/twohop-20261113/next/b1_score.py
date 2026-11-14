"""B1 scoring (prereg_b1.json) after the map (b1-map.json) and the S5 forecast (b1-forecast.json)
are committed. python b1_score.py MODEL CAPDIR B1DIR OUT

Reads the 1c captures (CAPDIR/cap/depth), the B1 arms (./arms-b1.jsonl, B1DIR/arm/*.pt), the
frozen map, the candidate sets and the X1 correct flags (x1-17-depth-panel.json). Cell content
is the OV readout at unit mass (circuit_map.cell_census); per unit final rho, so ranks are exact.
"""
import json
import math
import random
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from mdb.analysis import circuit_map as cm
from mdb.analysis.reads import layer_module
from mdb.repl.__main__ import _load

MODEL, CAPDIR, B1DIR, OUT = sys.argv[1:5]
dbg = _load(Path(MODEL), "fp32", "cpu", True, None)
ad = dbg.adapter
L, H, KV = ad.layers, ad.num_heads, ad.num_kv_heads
final = getattr(ad.backbone, ad.spec.final_norm)
gamma = final.weight.double()
W_U = ad.model.lm_head.weight.double()
WR = {x: cm.group_writers(layer_module(ad, x).self_attn.o_proj.weight, H, KV) for x in range(L)}
S = json.load(open("b1_sets.json"))
SETS, T = S["sets"], S["truths"]
RO = {k: gamma[None, :] * W_U[v] for k, v in SETS.items()}
MAP = json.load(open("b1-map.json"))
FC = json.load(open("b1-forecast.json"))
X1 = {r["id"]: r["correct"] for r in json.load(open("x1-17-depth-panel.json"))["rows"]}
ARMS = {json.loads(line)["id"]: json.loads(line) for line in open("arms-b1.jsonl")}
D34 = ("d3-lm", "d3-per", "d4-work")
D24 = ("d2",) + D34
TRUTH_KEY = {"countries": "country", "cities": "city", "languages": "language",
             "persons_surname": "persons_surname", "persons_given": "persons_given"}


def safe(i):
    return i.replace(":", "_").replace(" ", "_").replace("/", "_")


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


def q(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if len(xs) < 2:
        return {"values": xs}
    qs = st.quantiles(xs, n=4)
    return {"median": st.median(xs), "q1": qs[0], "q3": qs[2], "n": len(xs)}


def rank(scores, name, truth):
    return cm.rank_of(scores, SETS[name], truth)


def readout(write, name):
    return RO[name] @ write.double().reshape(-1)


def best(scores, name):
    return SETS[name][int(scores.argmax())]


def person_rank(scores_s, scores_g, t):
    rs = rank(scores_s, "persons_surname", t["persons_surname"])
    rg = rank(scores_g, "persons_given", t["persons_given"])
    found = [r for r in (rs, rg) if r is not None]
    return min(found) if found else None


def carrier_row(cap, span):
    p = cap["L20.p"].reshape(H, -1)
    rows = [r for r in span if r != 0]
    return max(rows, key=lambda r: (float(p[12, r] + p[13, r]), r))


frozen_cells = [(b["layer"], b["kv"]) for b in MAP["constants"]["best_cells"]]
rng = random.Random(0)
per = {}
caps = {}
for iid in sorted(T):
    t = T[iid]
    cap = torch.load(Path(CAPDIR) / "cap" / "depth" / f"{safe(iid)}.pt", map_location="cpu",
                     weights_only=False)
    caps[iid] = cap
    span = t["span"]
    crow = carrier_row(cap, span)
    values = {x: cap[f"v{x}"].double() for x in range(L)}
    cen = cm.cell_census(values, WR, span, RO)
    sc = cen["scores"]                                    # name -> (L, R, KV, n)
    ri = {r: k for k, r in enumerate(span)}
    ct = SETS["countries"].index(t["country"])
    cs = sc["countries"]
    other = cs.clone()
    other[..., ct] = -float("inf")
    margin = cs[..., ct] - other.max(-1).values
    flat = int(margin.reshape(-1).argmax())
    lay, rem = divmod(flat, len(span) * KV)
    rk, g = divmod(rem, KV)
    bestcell = {"layer": lay, "row": span[rk], "kv": g}

    def cell(name, r, layer=20, kv=6):
        return sc[name][layer, ri[r], kv]

    def names(name, r, layer=20, kv=6):
        return best(cell(name, r, layer, kv), name) == t[TRUTH_KEY[name]]
    rec = {"family": t["family"], "correct": X1[iid], "carrier_row": crow, "last_span": span[-1],
           "span_len": len(span), "best_cell": bestcell,
           "best_is_L20KV6": (lay, g) == (20, 6), "best_at_carrier": bestcell == {"layer": 20, "row": crow, "kv": 6},
           "carrier_country": names("countries", crow),
           "carrier_country_rank": rank(cell("countries", crow), "countries", t["country"]),
           "carrier_city_rank": rank(cell("cities", crow), "cities", t["city"]),
           "carrier_language": names("languages", crow),
           "carrier_country_best": best(cell("countries", crow), "countries"),
           "last_country": names("countries", span[-1]), "last_city_rank": rank(cell("cities", span[-1]), "cities", t["city"])}
    if t["persons_surname"] is not None:
        rec["carrier_person_rank"] = person_rank(cell("persons_surname", crow), cell("persons_given", crow), t)
    # first naming layers over span cells (any row, any kv)
    def first_layer(name):
        tok = t[TRUTH_KEY[name]]
        if tok is None:
            return None
        bi = sc[name].argmax(-1)                              # (L, R, KV)
        hit = (torch.tensor(SETS[name])[bi] == tok)
        layers = [x for x in range(L) if bool(hit[x].any())]
        return layers[0] if layers else None
    rec["first_country"] = first_layer("countries")
    rec["first_city"] = first_layer("cities")
    if t["persons_surname"] is not None:
        fs, fg = first_layer("persons_surname"), first_layer("persons_given")
        rec["first_person"] = min([x for x in (fs, fg) if x is not None], default=None)
    # language in the answer row's MLP writes
    lang_first = None
    for x in range(L):
        s = readout(cap[f"L{x}.m"], "languages")
        if best(s, "languages") == t["language"]:
            lang_first = x
            break
    rec["language_in_mlp"] = lang_first
    # nulls
    mass = torch.stack([cap[f"L{x}.p"].reshape(H, -1) for x in range(L)])     # (L, H, T)
    gm = torch.stack([mass[:, [h for h in range(H) if h // (H // KV) == gg], :].sum(1)
                      for gg in range(KV)], 1)                                   # (L, KV, T)
    sub = gm[:, :, span]
    mf = int(sub.reshape(-1).argmax())
    ml, mrem = divmod(mf, KV * len(span))
    mg, mr = divmod(mrem, len(span))
    rec["null_mass_country"] = names("countries", span[mr], ml, mg)
    for nm, port in (("null_lens20", "L20.h"), ("null_lens24", "L24.h")):
        s = readout(cap[port], "countries")
        rec[nm] = best(s, "countries") == t["country"]
        rec[nm + "_city"] = best(readout(cap[port], "cities"), "cities") == t["city"]
    others = [r for r in span if r not in (crow, 0)]
    rec["null_shuffle_row"] = names("countries", rng.choice(others)) if others else None
    fl, fk = rng.choice(frozen_cells)
    rec["null_shuffle_cell"] = names("countries", crow, fl, fk)
    per[iid] = rec

# cross-item null: another same-family item with another country; its carrier cell names mine
for iid, rec in per.items():
    t = T[iid]
    pool = sorted(k for k in per if T[k]["family"] == t["family"] and T[k]["country"] != t["country"])
    if not pool:
        rec["null_cross"] = None
        continue
    k = random.Random(iid).choice(pool)
    cap = caps[k]
    crow = per[k]["carrier_row"]
    w = sum(WR[20][6] @ cap["v20"][0, 6, crow].double() for _ in [0])
    rec["null_cross"] = best(readout(w, "countries"), "countries") == t["country"]

# arms
for iid, rec in per.items():
    a = ARMS.get(iid)
    if a is None:
        rec["arms"] = "missing"
        continue
    c = a["cell"]
    m0 = c["native"]["logits"][0] - c["native"]["logits"][1]
    m1 = c["edited"]["logits"][0] - c["edited"]["logits"][1]
    rec["cell_dM"] = m1 - m0
    rec["cell_flip"] = c["edited"]["argmax"] != c["native"]["argmax"]
    rec["cell_to_foil"] = c["edited"]["argmax"] == c["foil"]
    rec["cell_edge"] = c["edge"]
    rec["fc_dM"] = FC[iid]["forecast_dM"]
    rec["fc_carrier_row_agrees"] = FC[iid]["carrier_row"] == a["carrier_row"]
    tens = torch.load(Path(B1DIR) / "arm" / f"{safe(iid)}.pt", map_location="cpu", weights_only=False)
    t = T[iid]
    for kind in ("e1", "control"):
        e = a.get(kind)
        if not e or e.get("state") == "excluded":
            continue
        w = WR[20][6] @ tens[f"{kind}_carrier_v"].double()
        bc = best(readout(w, "countries"), "countries")
        e0 = e["native"]["logits"][0] - e["native"]["logits"][1]
        e1 = e["edited"]["logits"][0] - e["edited"]["logits"][1]
        rec[kind] = {"country_after": bc, "names_true": bc == t["country"],
                     "names_donor": bc == T[e["donor"]]["country"],
                     "moved": bc != rec["carrier_country_best"],
                     "argmax_to_donor_lang": e["edited"]["argmax"] == T[e["donor"]]["language"],
                     "argmax_unchanged": e["edited"]["argmax"] == e["native"]["argmax"],
                     "dM": e1 - e0, "edge": e["edge"]}

res = {"n": len(per)}
corr = lambda fams: [r for r in per.values() if r["family"] in fams and r["correct"]]
alls = lambda fams: [r for r in per.values() if r["family"] in fams]
c24 = corr(D24)
differ = [r for r in c24 if r["carrier_row"] != r["last_span"]]
res["S1"] = {"best_is_L20KV6": rate(r["best_is_L20KV6"] for r in c24),
             "best_at_carrier": rate(r["best_at_carrier"] for r in c24),
             "best_row_is_carrier": rate(r["best_cell"]["row"] == r["carrier_row"] for r in c24),
             "carrier_differs_from_last": len(differ),
             "where_differ": {"carrier_names": rate(r["carrier_country"] for r in differ),
                              "last_names": rate(r["last_country"] for r in differ)},
             "best_cell_counts": Counter(f"L{r['best_cell']['layer']}kv{r['best_cell']['kv']}" for r in c24).most_common(6)}
res["S2"] = {"carrier_country": rate(r["carrier_country"] for r in c24),
             "carrier_city_rank_le3": rate((r["carrier_city_rank"] or 99) <= 3 for r in c24),
             "d4_person_rank_le3": rate((r.get("carrier_person_rank") or 99) <= 3 for r in corr(("d4-work",))),
             "d3per_person_rank_le3": rate((r.get("carrier_person_rank") or 99) <= 3 for r in corr(("d3-per",))),
             "d4_ladder": rate(r["first_person"] is not None and r["first_city"] is not None and
                               r["first_country"] is not None and
                               r["first_person"] < r["first_city"] < r["first_country"]
                               for r in corr(("d4-work",))),
             "first_layers_d4": {k: q([r.get(k) for r in corr(("d4-work",))])
                                 for k in ("first_person", "first_city", "first_country")},
             "first_layers_d3": {k: q([r.get(k) for r in corr(D34[:2])])
                                 for k in ("first_person", "first_city", "first_country")}}
res["S3"] = {"carrier_names_language": rate(r["carrier_language"] for r in alls(D24)),
             "language_in_mlp_ge21": rate((r["language_in_mlp"] or -1) >= 21 for r in c24),
             "language_in_mlp": q([r["language_in_mlp"] for r in c24]),
             "language_never_in_mlp": sum(r["language_in_mlp"] is None for r in c24)}
w34, c34 = [r for r in alls(D34) if not r["correct"]], corr(D34)
bad = lambda r: (not r["carrier_country"]) or (r["carrier_country_rank"] or 99) > 3
res["S4"] = {"wrong_bad": rate(bad(r) for r in w34), "correct_bad": rate(bad(r) for r in c34)}
s5 = [r for r in per.values() if "cell_dM" in r and abs(r["cell_dM"]) >= 0.125]
res["S5"] = {"sign_agree": rate((r["fc_dM"] > 0) == (r["cell_dM"] > 0) for r in s5),
             "fc_over_native": q([r["fc_dM"] / r["cell_dM"] for r in s5]),
             "carrier_row_agrees_capture_vs_native": rate(r.get("fc_carrier_row_agrees") for r in per.values() if "cell_dM" in r),
             "cell_flip": rate(r["cell_flip"] for r in per.values() if "cell_dM" in r),
             "cell_to_foil": rate(r["cell_to_foil"] for r in per.values() if "cell_dM" in r)}
e1 = [r for r in per.values() if "e1" in r]
ctl = [r for r in per.values() if "control" in r]
res["S6"] = {"e1_names_donor": rate(r["e1"]["names_donor"] for r in e1),
             "e1_moved": rate(r["e1"]["moved"] for r in e1),
             "e1_argmax_to_donor_lang": rate(r["e1"]["argmax_to_donor_lang"] for r in e1),
             "control_neither": rate(r["control"]["names_true"] and r["control"]["argmax_unchanged"] for r in ctl),
             "control_names_true": rate(r["control"]["names_true"] for r in ctl),
             "hop3_mlp_share": q([r["cell_edge"]["shares"]["mlp"] for r in per.values()
                                  if "cell_edge" in r and r["cell_edge"]["shares"].get("mlp") is not None
                                  and abs(r["cell_edge"]["native_delta"]) >= 0.25]),
             "hop3_direct_share": q([r["cell_edge"]["shares"]["direct"] for r in per.values()
                                     if "cell_edge" in r and r["cell_edge"]["shares"].get("direct") is not None
                                     and abs(r["cell_edge"]["native_delta"]) >= 0.25]),
             "e1_cell_edge": {k: q([r["e1"]["edge"][k] / r["e1"]["edge"]["native_delta"] for r in e1
                                    if abs(r["e1"]["edge"]["native_delta"]) > 1e-6])
                              for k in ("reread", "mlp", "direct", "norm")},
             "e1_cell_native_delta": q([r["e1"]["edge"]["native_delta"] for r in e1]),
             "e1_closure_gap": q([abs(r["e1"]["edge"]["rounding"]) for r in e1])}
res["nulls"] = {}
for fam in ("d2",) + D34:
    rows = alls((fam,))
    cr = corr((fam,))
    res["nulls"][fam] = {
        "n": len(rows), "n_correct": len(cr),
        "rule_carrier": sum(r["carrier_country"] for r in rows),
        "always_same_site": sum(r["last_country"] for r in rows),
        "mass": sum(r["null_mass_country"] for r in rows),
        "lens20": sum(r["null_lens20"] for r in rows), "lens24": sum(r["null_lens24"] for r in rows),
        "shuffle_row": sum(bool(r["null_shuffle_row"]) for r in rows),
        "shuffle_row_n": sum(r["null_shuffle_row"] is not None for r in rows),
        "shuffle_cell": sum(r["null_shuffle_cell"] for r in rows),
        "cross_item": sum(bool(r["null_cross"]) for r in rows)}
res["F_map"] = {fam: v["rule_carrier"] < v["always_same_site"] - 1 for fam, v in res["nulls"].items()}
res["F_stack"] = not (res["S2"]["d4_person_rank_le3"][2] and res["S2"]["d4_person_rank_le3"][2] >= 0.40) and \
    not (res["S2"]["d4_ladder"][2] and res["S2"]["d4_ladder"][2] >= 0.60)
res["F_edge"] = (res["S6"]["e1_moved"][2] or 0) < 0.40
res["d1"] = {"carrier_country": rate(r["carrier_country"] for r in alls(("d1",))),
             "carrier_city_rank_le3": rate((r["carrier_city_rank"] or 99) <= 3 for r in alls(("d1",)))}
res["capital_coreadout_by_family"] = {f: {"carrier_country": rate(r["carrier_country"] for r in alls((f,))),
                                          "carrier_city_rank1": rate(r["carrier_city_rank"] == 1 for r in alls((f,)))}
                                     for f in ("d1", "d2") + D34}
json.dump({"summary": res, "per_item": per}, open(OUT, "w"), indent=1, default=str)
print(json.dumps(res, default=str)[:12000])

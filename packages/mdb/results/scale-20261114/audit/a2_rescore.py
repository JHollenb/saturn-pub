"""Audit: recompute the hop-budget, B4, B6, split, hard-prompt, supplement and closure numbers from
per-item records (out/score-*.json, out/supscore-*.json) and the raw A2 arm outputs
(audit/data/a2-*.json), with my own aggregation code; add the base rates the report omits.

python audit/a2_rescore.py   (from results/scale-20261114)
"""
import json
import random
import statistics as st

from scipy.stats import mannwhitneyu, spearmanr

M = ["Qwen2.5-0.5B", "Qwen3-0.6B-Base", "Qwen2.5-1.5B", "Qwen3-1.7B-Base", "Qwen2.5-3B", "Qwen3-4B-Base"]
P = json.load(open("panel.json"))
PS = json.load(open("panel_supp.json"))
by_id = {i["id"]: i for i in P["items"]}
SETS = P["sets"]
res = {}


def fr(xs):
    xs = [x for x in xs if x is not None]
    return f"{sum(map(bool, xs))}/{len(xs)}"


for m in M:
    d = json.load(open(f"out/score-{m}.json"))
    pi, LC, L = d["per_item"], d["model"]["L_c"], d["model"]["layers"]
    r = {}
    fams = list(P["counts"])
    # accuracy from capture json, independent of score.py
    ci = {}
    for f in ([f"capture-{m}.json"] if m not in ("Qwen2.5-3B", "Qwen3-4B-Base") else [f"capture-{m}-a.json", f"capture-{m}-b.json"]):
        ci.update(json.load(open(f))["items"])
    corr = {i["id"]: ci[i["id"]]["argmax"] == i["answer"] for i in P["items"]}
    assert all(corr[k] == pi[k]["correct"] for k in pi), m
    r["acc"] = {f: fr([corr[i["id"]] for i in P["items"] if i["family"] == f]) for f in fams}
    # B2 slack AUC over d3-lm + d3-per + d4-work (absent -> -1)
    deep = [pi[i["id"]] for i in P["items"] if i["family"] in ("d3-lm", "d3-per", "d4-work")]
    sl = lambda x: -1 if x["L_first"] is None else LC - x["L_first"]
    pos, neg = [sl(x) for x in deep if x["correct"]], [sl(x) for x in deep if not x["correct"]]
    r["B2_auc"] = round(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)), 4)
    # stable variant (secondary)
    ss = lambda x: -1 if x["L_first_stable"] is None else LC - x["L_first_stable"]
    r["B2_auc_stable"] = round(mannwhitneyu([ss(x) for x in deep if x["correct"]], [ss(x) for x in deep if not x["correct"]]).statistic / (len(pos) * len(neg)), 4)
    # L_first medians (absent -> LC+1) real vs fake for correct items
    lf = lambda v: LC + 1 if v is None else v
    r["L_first_med"] = {f: [st.median([lf(pi[i["id"]]["L_first"]) for i in P["items"] if i["family"] == f and corr[i["id"]]]),
                            st.median([lf(pi[i["id"]]["L_first_fake"]) for i in P["items"] if i["family"] == f and corr[i["id"]]])]
                        for f in ("d2", "d3-lm", "d3-per", "d4-work")}
    # how often L_first is at layer 0..2 (a lexical / early hit) on correct d2-d4
    r["L_first_le2_share"] = fr([pi[i["id"]]["L_first"] is not None and pi[i["id"]]["L_first"] <= 2 for i in P["items"] if i["block"] == "depth" and corr[i["id"]]])
    # B3 from the raw arms
    a = {}
    for part in ("split", "rest"):
        a.update({k: {**a.get(k, {}), **v} for k, v in json.load(open(f"audit/data/a2-{m}-{part}.json"))["items"].items()})
    late, never = [], []
    for i in P["items"]:
        x = a.get(i["id"], {})
        if "backpatch" in x:
            ok = x["backpatch"]["argmax"] == i["answer"]
            (late if pi[i["id"]]["L_first"] is not None else never).append(ok)
    r["B3_late"], r["B3_never"] = fr(late), fr(never)
    # B6 from the raw arms + capture logits are not local: use score's b6_drop but recompute the ratio
    lang = [pi[i["id"]].get("b6_drop") for i in P["items"] if i["block"] == "depth"]
    capd = [pi[i["id"]].get("b6_drop") for i in P["items"] if i["block"] == "base"]
    r["B6"] = {"lang_med": st.median([v for v in lang if v is not None]), "cap_med": st.median([v for v in capd if v is not None]),
               "b6_argmax_kept_depth": fr([a[i["id"]]["b6"]["argmax"] == i["answer"] for i in P["items"] if i["block"] == "depth" and corr[i["id"]] and "b6" in a.get(i["id"], {})]),
               "b6_argmax_kept_base": fr([a[i["id"]]["b6"]["argmax"] == i["answer"] for i in P["items"] if i["block"] == "base" and corr[i["id"]] and "b6" in a.get(i["id"], {})])}
    # split from the raw arms (correct base items): mlp / native_delta
    for dose in ("0.5", "1.0"):
        sh = [a[k][dose]["mlp"] / a[k][dose]["native_delta"] for k in a if dose in a[k] and a[k][dose]["native_delta"]]
        rr = [a[k][dose]["reread"] / a[k][dose]["native_delta"] for k in a if dose in a[k] and a[k][dose]["native_delta"]]
        di = [a[k][dose]["direct"] / a[k][dose]["native_delta"] for k in a if dose in a[k] and a[k][dose]["native_delta"]]
        r[f"split_{dose}"] = {"n": len(sh), "mlp_med": round(st.median(sh), 3), "reread_med": round(st.median(rr), 3), "direct_med": round(st.median(di), 3)}
    # readable-error: share on wrong items AND the same condition on correct items (base rate)
    def rd(fam, ok):
        return [(pi[i["id"]]["country_rank"] is None or pi[i["id"]]["country_rank"] > 3) for i in P["items"] if i["family"] == fam and corr[i["id"]] == ok]
    r["readable"] = {f: {"wrong": fr(rd(f, False)), "same_condition_on_correct": fr(rd(f, True))} for f in fams}
    # dist / obsc
    dw = [pi[i["id"]] for i in P["items"] if i["family"] == "dist-city" and not corr[i["id"]]]
    r["dist_wrong"] = {"true_carrier_names_true": fr([x["names_country"] for x in dw]),
                       "distractor_names_its": fr([x["dist"]["distractor_names_its_country"] for x in dw]),
                       "mass_dist_gt_true": fr([(x["dist"]["mass_distractor"] or 0) > (x["dist"]["mass_true"] or 0) for x in dw]),
                       "repair": fr([a[i["id"]]["closure"]["true_cell_into_distractor"]["argmax"] == i["answer"] for i in P["items"] if i["family"] == "dist-city" and not corr[i["id"]]])}
    dc = [pi[i["id"]] for i in P["items"] if i["family"] == "dist-city" and corr[i["id"]]]
    r["dist_correct"] = {"true_carrier_names_true": fr([x["names_country"] for x in dc]),
                         "distractor_names_its": fr([x["dist"]["distractor_names_its_country"] for x in dc]),
                         "mass_dist_gt_true": fr([(x["dist"]["mass_distractor"] or 0) > (x["dist"]["mass_true"] or 0) for x in dc])}
    ow = [pi[i["id"]] for i in P["items"] if i["family"] == "obsc-city" and not corr[i["id"]]]
    r["obsc_wrong"] = {"carrier_names_true": fr([x["names_country"] for x in ow]), "readable": fr([x["readable_error"] for x in ow])}
    # closure from raw arms
    r["closure_same_keeps"] = fr([a[i["id"]]["closure"]["same_country_mean"]["argmax"] == i["answer"] for i in P["items"] if i["block"] == "base" and "same_country_mean" in a.get(i["id"], {}).get("closure", {})])
    # co-readouts with the uniform chance level beside
    for sk, blk, n_set in (("cities", "base", len(SETS["cities"])), ("languages", "depth", len(SETS["languages"]))):
        cs = [pi[i["id"]]["co"][sk] for i in P["items"] if i["block"] == blk and sk in pi[i["id"]]["co"]]
        r[f"co_{sk}"] = {k: fr([c[k] for c in cs]) for k in ("raw", "projected", "lexical_null", "counts")}
        r[f"co_{sk}"]["uniform_chance_expect"] = round(len(cs) / n_set, 2)
        r[f"co_{sk}"]["projected_given_lexical_fails"] = fr([c["projected"] for c in cs if not c["lexical_null"]])
    r["proj_lang_vs_shuffled"] = [d["model"]["projected_language_hits"], d["model"]["projected_language_shuffled_hits"]]
    # supplement
    s = json.load(open(f"out/supscore-{m}.json"))
    sp = s["per_item"]
    sup = {}
    for f in ("d5", "d6", "d5-short", "native-city", "english-city"):
        ok = [x for x in sp.values() if x["family"] == f and x["correct"]]
        bad = [x for x in sp.values() if x["family"] == f and not x["correct"]]
        cond = lambda x: x["country_rank"] is None or x["country_rank"] > 3
        sup[f] = {"acc": f"{len(ok)}/{len(ok) + len(bad)}", "readable_wrong": fr([cond(x) for x in bad]),
                  "same_condition_on_correct": fr([cond(x) for x in ok]),
                  "names_country_wrong": fr([x["names_country"] for x in bad]),
                  "names_country_correct": fr([x["names_country"] for x in ok])}
    # paired native vs english (same city): discordant pairs
    nat = {x_id.split(":", 1)[1]: v for x_id, v in sp.items() if v["family"] == "native-city"}
    eng = {x_id.split(":", 1)[1]: v for x_id, v in sp.items() if v["family"] == "english-city"}
    r["supp"] = sup
    r["supp_ids_sample"] = list(sp)[:2]
    res[m] = r

# B4 Spearman across (model, family) cells with >= 1 wrong item, as combine.py; plus variants
cells = []
for m in M:
    pi = json.load(open(f"out/score-{m}.json"))["per_item"]
    for f in P["counts"]:
        rs = [pi[i["id"]] for i in P["items"] if i["family"] == f]
        bad = [x for x in rs if not x["correct"]]
        if not bad:
            continue
        cells.append((m, f, 1 - sum(x["correct"] for x in rs) / len(rs), sum(x["readable_error"] for x in bad) / len(bad), len(bad)))
rho = spearmanr([c[2] for c in cells], [c[3] for c in cells]).statistic
rho8 = spearmanr([c[2] for c in cells if c[4] >= 8], [c[3] for c in cells if c[4] >= 8]).statistic
fails = [(c[0], c[1], round(c[3], 2), c[4]) for c in cells if c[2] > 0.5 and c[3] < 0.40]
res["B4"] = {"n_cells": len(cells), "spearman_all": round(rho, 4), "n_cells_wrong_ge8": sum(c[4] >= 8 for c in cells),
             "spearman_wrong_ge8": round(rho8, 4), "falsifying_cells_acc_lt_0.5_share_lt_0.40": fails}
json.dump(res, open("audit/a2_rescore.json", "w"), indent=1)
for m in M:
    print("==", m)
    for k, v in res[m].items():
        print(" ", k, json.dumps(v))
print("B4", json.dumps(res["B4"]))

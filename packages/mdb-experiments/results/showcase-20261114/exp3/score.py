"""Score exp3 (prereg exp3/prereg.json).  python score.py ITEMS_JSONL NOMINATED_JSON B1_SCORE OUT"""
import json
import statistics as st
import sys

R = [json.loads(line) for line in open(sys.argv[1])]
NOM = json.load(open(sys.argv[2]))
B1 = json.load(open(sys.argv[3]))["per_item"]
feats = NOM["features"]
h1 = sorted({(x, f) for x, f, _ in feats if 21 <= x <= 24})
C = [r for r in R if r["correct"] and r["arms"].get("joint")]
stat = [-r["arms"]["joint"]["M_lang"] - 2 * (-r["arms"]["joint"]["M_city"]) for r in C]
ratio = [r["arms"]["joint"]["M_lang"] / r["arms"]["joint"]["M_city"] for r in C if r["arms"]["joint"]["M_city"]]
rnd = [r for r in R if r["arms"].get("random")]
rnd_in = [r for r in rnd if abs(r["arms"]["random"]["M_lang"]) <= r["floor"]]
# H4
corr_acts = {}
for r in R:
    if r["correct"]:
        for k, v in r["lang_feature_acts"].items():
            corr_acts.setdefault((r["lang"], k), []).append(v)
h4 = []
for r in R:
    if r["correct"] or not B1.get(r["id"], {}).get("carrier_country"):
        continue
    below = [v < st.median(corr_acts[(r["lang"], k)]) for k, v in r["lang_feature_acts"].items()
             if corr_acts.get((r["lang"], k))]
    h4.append({"id": r["id"], "n_features": len(below), "below": sum(below),
               "holds": bool(below) and sum(below) > len(below) / 2})
edges = [r["arms"]["joint"]["edge"] for r in C if "native_delta" in r["arms"]["joint"].get("edge", {})]
share = lambda k: st.median([e[k] / e["native_delta"] for e in edges if e["native_delta"]]) if edges else None  # noqa: E731
mean_arms = [r for r in R if any(k.startswith("mean_L") for k in r["arms"])]
mean_lang = [next(v for k, v in r["arms"].items() if k.startswith("mean_L"))["M_lang"] for r in mean_arms if r["correct"]]
mean_city = [next(v for k, v in r["arms"].items() if k.startswith("mean_L"))["M_city"] for r in mean_arms if r["correct"]]
singles = [v["M_lang"] for r in C for k, v in r["arms"].items() if k.startswith("L")]
summary = {
    "n_items": len(R), "correct_with_nominated": len(C), "wrong": sum(1 for r in R if not r["correct"]),
    "nominated_total": len(feats), "nominated_by_layer": {str(x): sum(1 for y, _, _ in feats if y == x) for x in range(20, 26)},
    "languages_with_features": len({g for _, _, g in feats}),
    "H1": {"L21_24_features": len(h1), "holds": len(h1) >= 3},
    "H2": {"median_lang_minus_2city_drop": st.median(stat) if stat else None,
           "median_ratio_lang_over_city": st.median(ratio) if ratio else None,
           "median_dM_lang_joint": st.median([r["arms"]["joint"]["M_lang"] for r in C]) if C else None,
           "median_dM_city_joint": st.median([r["arms"]["joint"]["M_city"] for r in C]) if C else None,
           "holds": bool(stat) and st.median(stat) >= 0},
    "H3": {"random_within_floor": f"{len(rnd_in)}/{len(rnd)}", "holds": bool(rnd) and len(rnd_in) / len(rnd) >= 0.75},
    "H4": {"items": h4, "holds_on": f"{sum(1 for h in h4 if h['holds'])}/{len(h4)}",
           "holds": len(h4) > 0 and sum(1 for h in h4 if h["holds"]) >= 4 * len(h4) / 5},
    "falsifier_fires": (len(h1) >= 3) and not (bool(stat) and st.median(stat) >= 0),
    "mean_ablation_same_layer": {"median_dM_lang": st.median(mean_lang) if mean_lang else None,
                                 "median_dM_city": st.median(mean_city) if mean_city else None},
    "single_feature_median_dM_lang": st.median(singles) if singles else None,
    "hop_edge_joint": {"n": len(edges), "median_share_direct": share("direct"), "median_share_mlp": share("mlp"),
                       "median_share_reread": share("reread"), "median_share_norm": share("norm")},
    "transcoder_rel_err_median_by_layer": {x: st.median([r["layers"][x]["rel_err"] for r in R]) for x in R[0]["layers"]},
    "transcoder_error_direct_median_by_layer": {x: st.median([r["layers"][x]["error_de_lang"] for r in R]) for x in R[0]["layers"]},
}
json.dump(summary, open(sys.argv[4], "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "H4"} | {"H4": {k: v for k, v in summary["H4"].items() if k != "items"}}, indent=1))

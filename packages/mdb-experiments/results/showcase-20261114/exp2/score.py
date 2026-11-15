"""Score exp2 (prereg exp2/prereg.json) from job A's jsonl.  python score.py A_DIR OUT_JSON"""
import json
import statistics as st
import sys

src, out = sys.argv[1], sys.argv[2]
R = [json.loads(line) for n in ("exp2-items-s0.jsonl", "exp2-items-s1.jsonl") for line in open(f"{src}/{n}")]
C = [json.loads(line) for line in open(f"{src}/census-items-s1.jsonl")]
n = len(R)


def med(x):
    return st.median(x) if x else None


def err_row(rows):
    return [r for r in rows if r["c"] == "error"][0]


share_top, share_err, share_rnd = [], [], []
for r in R:
    t = r["l20"]
    ce = t["cell_e"]
    share_top.append(t["top"]["e"] / ce if t["top"] and ce else None)
    share_err.append(err_row(t["rows"])["e"] / ce if ce else None)
    share_rnd.append(t["random"]["e"] / ce if t["random"] and ce else None)
L1a = sum(1 for s in share_top if s is not None and s >= 0.5)
L1b = sum(1 for s in share_err if s is not None and s <= 0.2)
# L2 at L20
rat = {"top": [], "random": [], "error": []}
for r in R:
    arms = {(a["layer"], a["arm"]): a["dM"] for a in r["sweep"]["arms"]}
    z = arms.get((20, "zero"))
    if not z:
        continue
    for k in rat:
        if (20, k) in arms:
            rat[k].append(arms[(20, k)] / z)
L2 = {k: med(v) for k, v in rat.items()}
L2_holds = (L2["top"] is not None and L2["top"] >= 0.5 and L2["random"] <= 0.15 and L2["error"] <= 0.2)
# L3
on = [r["lineage"]["onset"] for r in R if "lineage" in r]
cens = sum(1 for r in R if "lineage" in r and r["lineage"]["censored"])
L3_med = med(on)
L3_fals = sum(1 for o in on if o >= 15)
# L4
inside = set(range(17, 22))
l4 = 0
above_layers = []
for r in R:
    fl = r["sweep"]["floor"]
    above = sorted(a["layer"] for a in r["sweep"]["arms"] if a["arm"] == "top" and abs(a["dM"]) > fl)
    above_layers.append(above)
    if above and set(above) <= inside:
        l4 += 1
# partition
part = sum(1 for r in R if r["l20"]["partition_holds"] and r["l20"]["closure"]["within_f64"])
part_layers = sum(1 for r in R if all(v["partition_holds"] for v in r["layers"].values()))
gap_max = max(r["l20"]["partition_gap"] for r in R)
# nulls
last_share = [r["l20_lastrow"]["top"]["e"] / r["l20"]["cell_e"] if r.get("l20_lastrow") and r["l20_lastrow"]["top"] else 0.0
              for r in R if r.get("l20_lastrow")]
proj = [r["projection_strip"] for r in R if "projection_strip" in r]
proj_med = [round(med([p[x] / p[19] for p in proj if p[19]]), 3) for x in range(28)] if proj else None
summary = {
    "n_items": n, "correct": sum(1 for r in R if r["correct"]),
    "saelens_active_set_agrees": f"{sum(1 for r in R + C if r['saelens_agrees'])}/{len(R) + len(C)}",
    "top_country_feature_found": sum(1 for s in share_top if s is not None),
    "L1": {"top_share_ge_0.5": f"{L1a}/{n}", "error_share_le_0.2": f"{L1b}/{n}",
           "median_top_share": med([s for s in share_top if s is not None]),
           "median_error_share": med([s for s in share_err if s is not None]),
           "median_random_share": med([s for s in share_rnd if s is not None]),
           "holds": L1a >= 40 and L1b >= 48},
    "L2": {"median_ratio_to_zero_cell": L2, "n": len(rat["top"]), "holds": L2_holds},
    "L3": {"median_onset": L3_med, "censored_at_L9": f"{cens}/{len(on)}", "onset_ge_L15": f"{L3_fals}/{len(on)}",
           "holds": L3_med is not None and L3_med <= 12, "falsifier_fires": L3_fals >= 40},
    "L4": {"above_floor_only_inside_L17_21": f"{l4}/{n}", "holds": l4 >= 48,
           "never_above_floor": sum(1 for a in above_layers if not a),
           "layer_histogram": {str(x): sum(1 for a in above_layers if x in a) for x in range(15, 24)}},
    "partition": {"l20_holds_and_closure_f64": f"{part}/{n}", "all_layers_partition": f"{part_layers}/{n}",
                  "max_gap": gap_max, "falsifier_fires": part < n},
    "nulls": {"lastrow_items": len(last_share), "lastrow_median_top_share": med(last_share),
              "random_feature_median_share": med([s for s in share_rnd if s is not None])},
    "projection_strip_median_rel_L19": proj_med,
    "features_top_l20": sorted({r["l20"]["top"]["f"] for r in R if r["l20"]["top"]}),
    "floor_median": med([r["sweep"]["floor"] for r in R]),
    "null_vs_stepped_max": max(r["sweep"]["null_vs_stepped"] or 0 for r in R),
}
json.dump(summary, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))

"""Score exp7.  python score.py ITEMS_JSONL OUT_JSON"""
import json, math, statistics as st, sys
R = [json.loads(l) for l in open(sys.argv[1])]
def sp(x):
    x = abs(x); return 2.0 ** (math.floor(math.log2(x)) - 7) if x else 0
elig = [r for r in R if r["M0"] > r["floor"]]
flip = lambda r, k: r["arms"][k]["M"] < -r["floor"]
iia = {k: sum(flip(r, k) for r in elig) for k in ("whole", "sae", "random")}
n = len(elig)
par = [abs(r["pyvene"]["M"] - r["arms"]["whole"]["M"]) for r in R]
kern = [abs(r["pyvene"]["M_clean"] - r["M0"]) for r in R]
s = {"pairs": len(R), "eligible_M0_above_floor": n,
     "IIA": {k: f"{v}/{n}" for k, v in iia.items()},
     "I1": {"whole_IIA": iia["whole"] / n if n else None, "holds": n and 0.55 <= iia["whole"] / n <= 0.65},
     "I2": {"sae_over_whole": iia["sae"] / iia["whole"] if iia["whole"] else None,
            "random_over_whole": iia["random"] / iia["whole"] if iia["whole"] else None,
            "median_rank": st.median([r["rank"] for r in R]), "max_rank": max(r["rank"] for r in R),
            "holds": bool(iia["whole"]) and iia["sae"] / iia["whole"] >= 0.8 and iia["random"] / iia["whole"] <= 0.2
                     and max(r["rank"] for r in R) <= 12,
            "falsifier_fires": bool(iia["whole"]) and iia["sae"] / iia["whole"] < 0.5},
     "I3": {"within_one_step": f"{sum(1 for r, d in zip(R, par) if d <= r['floor'] / 2)}/{len(R)}",
            "max_abs_diff": max(par), "median_abs_diff": st.median(par),
            "kernel_path_clean_median": st.median(kern), "kernel_path_clean_max": max(kern)},
     "median_frac_moved": {k: st.median([(r["M0"] - r["arms"][k]["M"]) / (r["M0"] - r["arms"]["whole"]["M"])
                                         for r in R if r["M0"] != r["arms"]["whole"]["M"]]) for k in ("sae", "random")}}
json.dump(s, open(sys.argv[2], "w"), indent=1); print(json.dumps(s, indent=1))

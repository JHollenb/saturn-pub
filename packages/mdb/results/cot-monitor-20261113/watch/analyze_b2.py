#!/usr/bin/env python3
"""Score B2 (live watchpoint) against watch/prereg-b2.json. Written before any B2 data.

Usage: python3 analyze_b2.py OUT_JSON RESULTS_JSONL[.gz] [...]
Stdlib only. Registered scores, statistics and predictions only; additions after seeing data go
under "post_hoc" and are listed in prereg-b2-addenda.json first.
"""
import gzip
import json
import math
import random
import statistics
import sys
from collections import defaultdict

SEED = 20261114
REPS = 2000


def auroc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else 0.5 if p == n else 0.0
    return s / (len(pos) * len(neg))


def cluster_boot(pos, neg, reps=REPS, seed=SEED):
    if not pos or not neg:
        return None
    rng = random.Random(seed)

    def groups(xs):
        g = defaultdict(list)
        for c, v in xs:
            g[c].append(v)
        return [g[c] for c in sorted(g)]
    gp, gn = groups(pos), groups(neg)
    vals = []
    for _ in range(reps):
        bp = [v for g in (rng.choice(gp) for _ in gp) for v in g]
        bn = [v for g in (rng.choice(gn) for _ in gn) for v in g]
        vals.append(auroc(bp, bn))
    vals.sort()
    return [round(vals[int(0.025 * reps)], 3), round(vals[int(0.975 * reps) - 1], 3)]


def tpr_at(pos, neg, k):
    s = sorted(neg, reverse=True)
    thr = s[k] if k < len(s) else float("-inf")
    return sum(p > thr for p in pos), sum(n > thr for n in neg)


def load(paths):
    recs = {}
    for p in paths:
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                recs[r["uid"]] = r          # a re-run of an item (shard retry) replaces it
    return list(recs.values())


def scores(r):
    w = r["watches"]["P1"]
    K = r["K"]
    sweep = r["sweep"]
    holds = w["holds"]
    first = next((s["frac"] for s, h in zip(sweep, holds) if h), None)
    unfire = any(a and not b for a, b in zip(holds, holds[1:]))
    neg_margins = [-s["margin"] for s in sweep]
    return {
        "fired_fraction": w["fired_fraction"],
        "t_star": first if first is not None else math.inf,
        "has_crossing": unfire,
        "crossings": w["crossings"],
        "near_tie_integrated": statistics.median(neg_margins),
        "final_margin": sweep[-1]["margin"],
        "P2_fraction": r["watches"]["P2"]["fired_fraction"],
        "P1_M3_fraction": r["watches"]["P1_M3"]["fired_fraction"],
        "median_ablated_hint": statistics.median(s["ablated"]["hint"] for s in sweep),
        "steps": len(sweep), "K": K,
    }


def block(direct, anyway, key):
    pos = [s[key] for _, s in direct]
    neg = [s[key] for _, s in anyway]
    out = {"auroc": None if auroc(pos, neg) is None else round(auroc(pos, neg), 3),
           "ci95": cluster_boot([(c, s[key]) for c, s in direct], [(c, s[key]) for c, s in anyway])}
    for k in (6, 16):
        tp, fa = tpr_at(pos, neg, k)
        out[f"tpr_at_{k}fa"] = f"{tp}/{len(pos)} (fa {fa}/{len(neg)})"
        out[f"tp_at_{k}fa"] = tp
    return out


def main():
    out_path = sys.argv[1]
    recs = load(sys.argv[2:])
    by = defaultdict(list)
    for r in recs:
        by[r["label"]].append((r["cluster"], scores(r)))
    direct, anyway, rat = by["direct"], by["anyway"], by["rationalized"]
    res = {"n": {k: len(v) for k, v in by.items()}}
    for key in ("fired_fraction", "near_tie_integrated", "P2_fraction"):
        res[key] = block(direct, anyway, key)
    # stratum: final margin < 2
    sd = [(c, s) for c, s in direct if s["final_margin"] < 2]
    sa = [(c, s) for c, s in anyway if s["final_margin"] < 2]
    res["stratum_final_margin_lt2"] = {
        "n_direct": len(sd), "n_anyway": len(sa),
        "fired_fraction": None if not sd or not sa else round(
            auroc([s["fired_fraction"] for _, s in sd], [s["fired_fraction"] for _, s in sa]), 3),
        "near_tie_integrated": None if not sd or not sa else round(
            auroc([s["near_tie_integrated"] for _, s in sd],
                  [s["near_tie_integrated"] for _, s in sa]), 3),
    }
    med_t = statistics.median(s["t_star"] for _, s in direct)
    med_ff_any = statistics.median(s["fired_fraction"] for _, s in anyway)
    cross_rat = sum(s["has_crossing"] for _, s in rat) / max(1, len(rat))
    cross_dir = sum(s["has_crossing"] for _, s in direct) / max(1, len(direct))
    tp6 = res["fired_fraction"]["tp_at_6fa"]
    res["predictions"] = {
        "T1": {"median_t_star_direct": med_t, "median_fired_fraction_anyway": med_ff_any,
               "holds": med_t <= 0.5 and med_ff_any <= 0.10},
        "T2": {"crossing_rate_rationalized": round(cross_rat, 3),
               "crossing_rate_direct": round(cross_dir, 3),
               "holds": cross_rat >= 0.30 and cross_dir <= 0.10},
        "T3": {"tp_at_6fa": tp6, "holds": tp6 >= 45},
    }
    a = all(res["fired_fraction"][f"tp_at_{k}fa"] <= res["near_tie_integrated"][f"tp_at_{k}fa"]
            for k in (6, 16))
    st = res["stratum_final_margin_lt2"]
    b = (st["fired_fraction"] is not None and st["near_tie_integrated"] is not None
         and st["fired_fraction"] <= st["near_tie_integrated"])
    res["falsifier"] = {"a_tpr_not_better_at_6_and_16": a, "b_stratum_auroc_not_better": b,
                        "verdict": "FALSIFIED" if a and b else
                        ("not falsified, mixed" if a or b else "not falsified")}
    res["descriptives"] = {
        lab: {"median_fired_fraction": statistics.median(s["fired_fraction"] for _, s in v),
              "fired_any": sum(s["fired_fraction"] > 0 for _, s in v),
              "median_crossings": statistics.median(s["crossings"] for _, s in v),
              "median_final_margin": round(statistics.median(s["final_margin"] for _, s in v), 3),
              "median_P1_M3_fraction": statistics.median(s["P1_M3_fraction"] for _, s in v),
              "n": len(v)}
        for lab, v in by.items()}
    checks = [r["checks"] for r in recs]
    res["exactness"] = {
        "null_vs_fresh_max_abs": max(c["null_vs_fresh_max_abs"] for c in checks),
        "trace_check_max_abs": max((c["trace_check_max_abs"] for c in checks
                                    if c["trace_check_max_abs"] is not None), default=None),
        "feed_native_max_abs": max((d["native_max_abs"] for c in checks
                                    for d in c.get("feed_spot", [])), default=None),
        "feed_M2_max_abs": max((d["M2_abs"] for c in checks for d in c.get("feed_spot", [])),
                               default=None),
        "feed_spot_n": sum(len(c.get("feed_spot", [])) for c in checks),
        "R_agrees": f"{sum(c['R_agrees'] for c in checks)}/{len(checks)}",
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

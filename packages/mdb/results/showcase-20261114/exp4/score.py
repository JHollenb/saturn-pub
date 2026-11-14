"""Score exp4 (prereg exp4/prereg.json).  python score.py ITEMS_JSONL OUT_JSON"""
import json
import statistics as st
import sys

sys.path.insert(0, sys.path[0] + "/../exp5")
from score import spearman  # noqa: E402

R = [json.loads(line) for line in open(sys.argv[1])]
N = [dict(nd, floor=r["floor"], item=r["id"]) for r in R for nd in r["nodes"] if nd["kind"] == "node"]
Z = [dict(nd, floor=r["floor"]) for r in R for nd in r["nodes"] if nd["kind"] == "random"]
g1 = [abs(n["rerouting"]) / abs(n["arm0"]) for n in N if n["arm0"]]
largest = [max(("rerouting", "norm", "nonlin"), key=lambda k: abs(n[k])) for n in N]
g2r = [(n["arm2"] - n["edge"]) / (n["arm0"] - n["edge"]) for n in N if n["arm0"] != n["edge"]]
g3 = sum(1 for n in N if abs(n["norm"]) <= n["floor"])
nulls_ok = sum(1 for r in R for nd in r["nodes"] if abs(nd["null"]) <= r["floor"])
nulls_n = sum(1 for r in R for nd in r["nodes"])
zr = sum(1 for z in Z if all(abs(z[k]) <= z["floor"] for k in ("rerouting", "norm", "nonlin")))
eff = [n for n in N if abs(n["arm0"]) > n["floor"]]
summary = {
    "items": len(R), "node_items": len(N),
    "G1": {"median_rerouting_over_arm0": st.median(g1) if g1 else None, "holds": bool(g1) and st.median(g1) <= 0.1,
           "falsifier_fires": bool(g1) and st.median(g1) >= 0.3},
    "G2": {"largest_counts": {k: largest.count(k) for k in ("rerouting", "norm", "nonlin")},
           "nonlin_largest_share": largest.count("nonlin") / len(largest) if largest else None,
           "median_(arm2+A)/(arm0+A)": st.median(g2r) if g2r else None,
           "holds": bool(largest) and largest.count("nonlin") / len(largest) >= 0.6 and st.median(g2r) >= 0.5},
    "G3": {"norm_below_floor": f"{g3}/{len(N)}", "holds": bool(N) and g3 / len(N) >= 0.7},
    "G4": {"spearman_edge_vs_arm0": spearman([n["edge"] for n in N], [n["arm0"] for n in N]) if len(N) > 2 else None},
    "median_arm0_over_edge": st.median([n["arm0"] / n["edge"] for n in N if n["edge"]]),
    "median_arm0_over_direct": st.median([n["arm0"] / -n["direct"] for n in N if n["direct"]]),
    "median_edge_over_direct": st.median([n["edge"] / -n["direct"] for n in N if n["direct"]]),
    "nodes_above_floor": f"{len(eff)}/{len(N)}",
    "null_machinery_within_floor": f"{nulls_ok}/{nulls_n}",
    "random_node_all_brackets_within_floor": f"{zr}/{len(Z)}",
    "median_abs_brackets": {k: st.median([abs(n[k]) for n in N]) for k in ("rerouting", "norm", "nonlin")},
}
summary["G4"]["holds"] = summary["G4"]["spearman_edge_vs_arm0"] is not None and summary["G4"]["spearman_edge_vs_arm0"] >= 0.85
json.dump(summary, open(sys.argv[2], "w"), indent=1)
print(json.dumps(summary, indent=1))

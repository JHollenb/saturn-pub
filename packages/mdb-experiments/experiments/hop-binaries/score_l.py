"""Score study L from the fetched passive and interchange outputs (laptop, no compute).

    python score_l.py PASSIVE_DIR INTERCHANGE_DIR > scores.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

TAGS = [f"{p}-{o}" for p in ("hop2", "arith", "hop3", "shortcut") for o in ("O0", "O2")]


def rate(rows, key):
    inf = [r for r in rows if r["informative"]]
    return {"informative": len(inf), "of": len(rows), key: sum(r[key] for r in inf),
            "rate": round(sum(r[key] for r in inf) / len(inf), 4) if inf else None}


def main(pdir: Path, idir: Path) -> dict:
    out = {}
    for tag in TAGS:
        P = json.loads((pdir / f"{tag}.passive.json").read_text())
        X = json.loads((idir / f"{tag}.interchange.json").read_text())
        items = P["items"]
        n = len(items)
        rows = X["rows"]
        by_item = {}
        for r in rows:
            by_item.setdefault((r["item"], r["role"]), []).append(r)
        causal = lambda i, role: any(r["informative"] and r["total"] for r in by_item.get((i, role), []))
        D_F = [bool(it["F_direct"]["top1"]) for it in items]
        D_FI = [D_F[i] and causal(i, "forecast_direct") for i in range(n)]
        D_FIN = [D_FI[i] and all(items[i]["novel_top1_direct"]) for i in range(n)]
        hit1 = [items[i]["F_direct"]["hit1"] for i in range(n)]
        truth_rows = [r for r in rows if r["role"] == "truth"]
        per_unit = {}
        for r in truth_rows:
            per_unit.setdefault(str(tuple(r["unit"])), []).append(r)
        edges_all, edges_cons = Counter(), Counter()
        for it in items:
            edges_all.update(it["bridge_edges"].get("all", {}))
            edges_cons.update(it["bridge_edges"].get("consuming", {}))
        rec = {
            "items": n,
            "lanes_equal_binctr": {"passive": P["lanes_equal_binctr"], "interchange": X["lanes_equal_binctr"]},
            "F_direct": {"hit1": round(sum(hit1), 3), "hit3": round(sum(it["F_direct"]["hit3"] for it in items), 3)},
            "F_total": {"hit1": round(sum(it["F_total"]["hit1"] for it in items), 3),
                        "hit3": round(sum(it["F_total"]["hit3"] for it in items), 3)},
            "identity_truth": {k: round(sum(it["identity_truth"][k] for it in items), 3)
                               for k in items[0]["identity_truth"]},
            "F_direct_top1_units": Counter(str([tuple(u) for u in it["F_direct"]["top1"]]) for it in items),
            "decisions_bridge": {"D_F": sum(D_F), "D_FI": sum(D_FI), "D_FIN": sum(D_FIN)},
            "decisions_bridge_at_true_cell": {"D_F": sum(D_F[i] and hit1[i] == 1 for i in range(n)),
                                              "D_FI": sum(D_FI[i] and hit1[i] == 1 for i in range(n)),
                                              "D_FIN": sum(D_FIN[i] and hit1[i] == 1 for i in range(n))},
            "forecast_direct_causal_items": sum(causal(i, "forecast_direct") for i in range(n)),
            "forecast_total_causal_items": sum(causal(i, "forecast_total") for i in range(n)),
            "truth_causal_items": sum(causal(i, "truth") for i in range(n)),
            "forecast_direct_patches": {**rate([r for r in rows if r["role"] == "forecast_direct"], "total"),
                                        "frozen": sum(r["frozen"] for r in rows if r["role"] == "forecast_direct"
                                                      and r["informative"])},
            "truth_paired": {"total": rate(truth_rows, "total"), "frozen": rate(truth_rows, "frozen"),
                             "to_donor_y": rate(truth_rows, "to_donor_y")},
            "truth_paired_per_unit": {u: {"total": rate(rs, "total")["rate"], "frozen": rate(rs, "frozen")["rate"],
                                          "informative": rate(rs, "total")["informative"]}
                                      for u, rs in sorted(per_unit.items())},
            "edges": {"all": dict(edges_all), "consuming": dict(edges_cons)},
            "rsrc_status": P["truth_report"]["rsrc_status"],
            "whatif": {k: X["whatif"].get("result", {}).get(k) for k in
                       ("verdict", "label", "total_effect", "frozen_effect", "path_diverged_at",
                        "address_diverged_at", "selection_moved_under_hold_at", "y", "y_frozen")}
            | {"unit": X["whatif"].get("unit"), "address": X["whatif"].get("address"),
               "value": X["whatif"].get("value")},
        }
        if "shortcut_holds_T1x" in X:
            rec["shortcut_holds_T1x"] = X["shortcut_holds_T1x"]
        out[tag] = rec
    return out


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]), Path(sys.argv[2])), indent=1, default=str))

"""Census tables from job3-*.jsonl (python3 census.py). Writes census.json.

sigma_Phi = (dM_phi - dM_null) / (dM_full - dM_null), sigma_E likewise, over pairs with
|dM_full - dM_null| >= 1 (prereg). P3 check: |dM_full - dM_direct| within the item floor of
analysis.json. Bridge features: per pair, the recipient-side features (activation lost, moving
the value cell toward the donor: delta_act < 0 and along > 0) ranked by along; per bridge
country, features that recur over >= 2 recipients of that bridge, with their logit-lens tokens.
"""
import glob
import json
import statistics as st
from collections import Counter, defaultdict

cand = {i["id"]: i for i in json.load(open("candidates.json"))["items"]}
floors = {x["id"]: x["floor"] for x in json.load(open("analysis.json"))["items"]}
recs = []
for path in sorted(glob.glob("job3-*.jsonl")):
    for line in open(path):
        r = json.loads(line)
        if "carrier" in r:
            recs.append(r)
med = lambda xs: st.median(xs) if xs else float("nan")  # noqa: E731
sphi, se, ssum, direct_ok, n_pairs, skipped = [], [], [], 0, 0, []
lin_err_share, lin_feat_share = [], []
rows = []
for r in recs:
    if "skipped" in r:
        skipped.append(r["id"])
        continue
    for p in r["pairs"]:
        n_pairs += 1
        full = p["full"] - p["null"]
        phi, err = p["phi"] - p["null"], p["E"] - p["null"]
        ok = abs(p["full"] - p["direct"]) <= floors.get(r["id"], 0.25)
        direct_ok += ok
        row = {"id": r["id"], "donor": p["donor"], "full": full, "phi": phi, "E": err,
               "direct": p["direct"], "null": p["null"], "direct_ok": ok,
               "rec_err_frac": r["rec_error_fraction"], "don_err_frac": p["don_error_fraction"]}
        if abs(full) >= 1:
            sphi.append(phi / full)
            se.append(err / full)
            ssum.append((phi + err) / full)
            row.update(sigma_phi=phi / full, sigma_E=err / full)
        if p["dv_norm_lin"] > 0:
            lin_err_share.append(p["dv_error_along"] / p["dv_norm_lin"])
            lin_feat_share.append(p["dv_bias_free_features_along"] / p["dv_norm_lin"])
        rows.append(row)
by_bridge = defaultdict(lambda: defaultdict(set))
lens = {}
for r in recs:
    for p in r.get("pairs", []):
        own = [f for f in p["top_features"] if f["delta_act"] < 0 and f["along"] > 0]
        for f in own[:3]:
            by_bridge[cand[r["id"]]["bridge"]][f["f"]].add(r["id"])
            lens[f["f"]] = f["lens"]
recurring = {}
for bridge, feats in by_bridge.items():
    n_items = len({rid for s in feats.values() for rid in s})
    common = {f: sorted(s) for f, s in feats.items() if len(s) >= 2}
    if common:
        recurring[bridge] = {"recipients": n_items,
                             "features": {str(f): {"items": s, "lens": lens[f]}
                                          for f, s in common.items()}}
top_feature_count = Counter(f for b in by_bridge.values() for f in b)
summary = {
    "pairs": n_pairs, "recipients": len(recs), "skipped_layer0": skipped,
    "pairs_with_abs_full_ge_1": len(sphi),
    "median_sigma_phi": med(sphi), "median_sigma_E": med(se), "median_sigma_sum": med(ssum),
    "q25_q75_sigma_phi": [sorted(sphi)[len(sphi) // 4], sorted(sphi)[3 * len(sphi) // 4]] if sphi else None,
    "q25_q75_sigma_E": [sorted(se)[len(se) // 4], sorted(se)[3 * len(se) // 4]] if se else None,
    "full_equals_direct_within_floor": f"{direct_ok}/{n_pairs}",
    "median_null": med([x["null"] for x in rows]),
    "median_full": med([x["full"] for x in rows]),
    "median_rec_error_fraction_of_norm": med([x["rec_err_frac"] for x in rows]),
    "linearised_dv_share_error_median": med(lin_err_share),
    "linearised_dv_share_features_median": med(lin_feat_share),
    "bridges_with_recurring_recipient_features": len(recurring),
    "bridges_with_ge2_recipients": sum(1 for b in by_bridge.values()
                                       if len({x for s in b.values() for x in s}) >= 2),
    "most_common_top_features": [(f, n, lens[f]) for f, n in top_feature_count.most_common(8)],
}
json.dump({"summary": summary, "rows": rows, "recurring": recurring}, open("census.json", "w"),
          indent=1)
print(json.dumps(summary, indent=1, ensure_ascii=False))
for bridge, info in sorted(recurring.items()):
    print(bridge, info["recipients"], "recipients;",
          "; ".join(f"f{f} x{len(v['items'])} {v['lens'][:4]}" for f, v in info["features"].items()))

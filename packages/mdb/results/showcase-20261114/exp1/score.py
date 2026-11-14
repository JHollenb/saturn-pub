"""Score exp1 (prereg exp1/prereg.json).  python score.py ITEMS_JSONL OUT_JSON"""
import json
import statistics as st
import sys

R = [json.loads(line) for line in open(sys.argv[1])]
W = [r for r in R if not r["correct"]]
C = [r for r in R if r["correct"]]


def dm(r, arm):
    a = r["arms"].get(arm)
    if not a:
        return None
    return (a["logits"][0] - a["logits"][1]) - (r["native"]["logits"][0] - r["native"]["logits"][1])


def repaired(r, arm):
    a = r["arms"].get(arm)
    return bool(a) and a["argmax"] == r["language"]


fired_w = [r for r in W if r["stop"]["watch_fired"]]
fired_c = [r for r in C if r["stop"]["watch_fired"]]
r1_rep = [r["id"] for r in fired_w if repaired(r, "R1")]
r1_rep_all = [r["id"] for r in W if repaired(r, "R1")]
rnd_rep = [r["id"] for r in W if repaired(r, "RND")]
r2_rep = [r["id"] for r in W if repaired(r, "R2")]
ratios = [dm(r, "R1") / dm(r, "R2") for r in W if dm(r, "R1") is not None and dm(r, "R2")]
r1_copy = [r["id"] for r in R if r["arms"].get("R1", {}).get("copy")]
city = [r for r in W if r.get("donor", {}).get("family") == "d2"]
r2_copy_city = [r["id"] for r in city if r["arms"].get("R2", {}).get("copy")]
r2_copy_all = [r["id"] for r in W if r["arms"].get("R2", {}).get("copy")]
r3 = [r for r in R if dm(r, "R3") is not None]
r3_small = [r for r in r3 if abs(dm(r, "R3")) < r["floor"]]
r3_w = [r for r in W if dm(r, "R3") is not None]
r1c_change = [r["id"] for r in C if r["arms"].get("R1") and r["arms"]["R1"]["argmax"] != r["native"]["argmax"]]
ver = [r["arms"]["R1"].get("verify", {}) for r in W if r["arms"].get("R1")]
summary = {
    "n": len(R), "wrong": len(W), "correct": len(C),
    "W1": {"fired_wrong": f"{len(fired_w)}/{len(W)}", "fired_correct": f"{len(fired_c)}/{len(C)}",
           "holds": len(fired_w) >= 20 and len(fired_c) <= 10,
           "below_15_country_not_in_dictionary": len(fired_w) < 15,
           "wrong_with_truth_feature": sum(1 for r in W if r["truth_feature"]),
           "wrong_with_wrong_feature_active": sum(1 for r in W if r["r1"]["wrong_feature"] is not None)},
    "W2": {"R1_repairs_where_fired": f"{len(r1_rep)}/{len(fired_w)}", "R1_repairs_all_wrong": f"{len(r1_rep_all)}/{len(W)}",
           "R2_repairs_all_wrong": f"{len(r2_rep)}/{len(W)}",
           "median_R1_over_R2_margin_move": st.median(ratios) if ratios else None, "n_ratio": len(ratios),
           "holds": len(r1_rep) >= 11 and bool(ratios) and st.median(ratios) >= 0.6,
           "repaired_ids": r1_rep},
    "W3": {"R1_copy": len(r1_copy), "R2_copy_city_donor": f"{len(r2_copy_city)}/{len(city)}",
           "R2_copy_any_donor": f"{len(r2_copy_all)}/{len(W)}",
           "holds": len(r1_copy) == 0 and len(r2_copy_city) >= 3},
    "W4": {"R3_below_floor": f"{len(r3_small)}/{len(r3)}", "holds": len(r3) > 0 and len(r3_small) / len(r3) >= 0.75,
           "R3_repairs_wrong": sum(1 for r in r3_w if repaired(r, "R3"))},
    "nulls": {"RND_repairs_wrong": f"{len(rnd_rep)}/{sum(1 for r in W if 'RND' in r['arms'])}",
              "R1_on_correct_argmax_changes": f"{len(r1c_change)}/{sum(1 for r in C if 'R1' in r['arms'])}",
              "lastrow_watch_wrong": f"{sum(1 for r in W if r['lastrow_watch'])}/{len(W)}",
              "lastrow_watch_correct": f"{sum(1 for r in C if r['lastrow_watch'])}/{len(C)}"},
    "falsifier": {"R1_repairs_lt_RND_plus_2": len(r1_rep_all) < len(rnd_rep) + 2, "W1_fails": not (len(fired_w) >= 20 and len(fired_c) <= 10)},
    "verify_R1": {"n": len(ver), "verdicts": {v: sum(1 for x in ver if x.get("verdict") == v) for v in {x.get("verdict") for x in ver}},
                  "refused": sum(1 for x in ver if "refused" in x)},
    "seal_refusals": sum(1 for r in W if "refused" in r["arms"].get("R1", {}).get("sealed", {})),
}
json.dump(summary, open(sys.argv[2], "w"), indent=1)
print(json.dumps(summary, indent=1))

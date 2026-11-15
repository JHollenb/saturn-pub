"""Summarise hop-math outputs against prereg_hopmath.json. python hm_analyze.py MODEL hm-*.json (shards)"""
import json
import statistics as st
import sys

name = sys.argv[1]
O = {"oprep": None, "h3": {"base": [], "depth": []}}
for p in sys.argv[2:]:
    d = json.load(open(p))
    LC = d["L_c"]
    if "oprep" in d:
        O["oprep"] = d["oprep"]
    for b in ("base", "depth"):
        O["h3"][b] += d.get("h3", {}).get(b, [])


def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else None


def frac(x):
    x = [v for v in x if v is not None]
    return [sum(bool(v) for v in x), len(x)]


res = {"model": name, "L_c": LC}
op = O["oprep"]
if op:
    it = [r for r in op["items"] if "excluded" not in r]
    res["oprep"] = {"excluded": sum("excluded" in r for r in op["items"]), "null_ok": frac([r["null_argmax_ok"] for r in it]),
                    "replica_donor_moves_base": op["replica_donor_moves_base"]}
    for s in ("readable", "right_operand"):
        for fam in ("d3", "d4", "d5", "all"):
            sel = [r for r in it if r["stratum"] == s and (fam == "all" or r["family"].startswith(fam))]
            res["oprep"][f"{s}.{fam}"] = {"cell": frac([r["cell_repaired"] for r in sel]),
                                          "band": frac([r["band_repaired"] for r in sel])}
h3 = {}
for b in ("base", "depth"):
    R = O["h3"][b]
    C = [r for r in R if r["correct"]]
    lad = {a: [r["a1_ladder"][a] for r in C if "a1_ladder" in r and r["a1_ladder"][a]["native_delta"]] for a in ("0.5", "1.0")}
    sh = lambda a, k: med([x[k] / x["native_delta"] for x in lad[a]])
    t = {"n": len(R), "correct": len(C), "a1_moves": frac([r.get("a1_moves") for r in C]),
         "a1_shares_1.0": {k: sh("1.0", k) for k in ("direct", "mlp", "reread", "norm")},
         "a1_shares_0.5": {k: sh("0.5", k) for k in ("direct", "mlp", "reread", "norm")},
         "a1_mlp_by_layer_1.0_median": [med([x["mlp_by_layer"][i] / x["native_delta"] for x in lad["1.0"]])
                                        for i in range(len(lad["1.0"][0]["mlp_by_layer"]))] if lad["1.0"] else None,
         "a1_dose_ratio": {k: med([y[k] / x[k] for x, y in zip(
             [r["a1_ladder"]["0.5"] for r in C if "a1_ladder" in r], [r["a1_ladder"]["1.0"] for r in C if "a1_ladder" in r]) if x[k]])
             for k in ("native_delta", "direct", "mlp", "reread")},
         "a1_closure_gap_max": max([abs(x["closure_gap"]) for a in lad for x in lad[a]], default=None),
         "a2_profile": {k: med([r["a2"][k] for r in C if "a2" in r]) for k in (C[0]["a2"] if C and "a2" in C[0] else [])},
         "a3_ratio": med([r.get("a3_ratio") for r in C]),
         "a4_comp_names": frac([r["a4_comp_names"] for r in R]), "a4_ctry_names": frac([r["a4_ctry_names"] for r in R]),
         "a4_comp_move": med([r["a4_comp_move"] for r in C]), "a4_ctry_move": med([r["a4_ctry_move"] for r in C])}
    h3[b] = t
res["h3"] = h3
if h3["base"]["a2_profile"] and h3["depth"]["a2_profile"]:
    asym = {}
    for k, lv in h3["depth"]["a2_profile"].items():
        cv = h3["base"]["a2_profile"].get(k)
        asym[k] = {"language": lv, "capital": cv, "lang_2x": lv is not None and cv is not None and lv >= 0.1 and lv >= 2 * max(cv, 0)}
    res["a2_asymmetry"] = asym
    res["a2_any_2x"] = any(v["lang_2x"] for v in asym.values())
json.dump(res, sys.stdout, indent=1)

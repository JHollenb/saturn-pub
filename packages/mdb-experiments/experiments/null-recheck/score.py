"""Score the null re-check (rules in PREREG.md): python score.py Q3_DIR Q25_DIR OUT_JSON

Unit share = fraction of the claim's units whose 16-donor pooled verdict (mdb donors.pooled) is
"effect". FLIPS if > 0.20, else SURVIVES; a flip whose median above-floor effect is < 4 floors is
marked kernel-sensitive.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mdb.analysis.donors import EFFECT, NO_EFFECT, pooled

WINDOW = set(range(17, 22))


def pool_rows(rows):
    return pooled([{"verdict": EFFECT if abs(r["dM"]) > r["floor"] else NO_EFFECT,
                    "informative": r["informative"]} for r in rows])


def claim(units, canon=0):
    """units: {unit: [row dicts with dM, floor, informative, to_donor, k]}"""
    n = len(units)
    pools = {u: pool_rows(rs) for u, rs in units.items()}
    eff = [u for u, p in pools.items() if p["verdict"] == EFFECT]
    allrows = [r for rs in units.values() for r in rs]
    above = [abs(r["dM"]) / r["floor"] for r in allrows if abs(r["dM"]) > r["floor"]]
    can = {u: next((r for r in rs if r["k"] == canon), None) for u, rs in units.items()}
    can_null = [u for u, r in can.items() if r is not None and abs(r["dM"]) <= r["floor"]]
    share = len(eff) / n if n else None
    med = st.median(above) if above else None
    verdict = ("FLIPS" if share is not None and share > 0.20 else "SURVIVES")
    if verdict == "FLIPS" and med is not None and med < 4:
        verdict = "FLIPS (kernel-sensitive, median < 4 floors)"
    return {
        "units": n, "units_effect": len(eff), "unit_share": share, "verdict": verdict,
        "donor_rows": len(allrows), "donor_rows_effect": len(above),
        "donor_share": len(above) / len(allrows) if allrows else None,
        "informative_rows": sum(bool(r["informative"]) for r in allrows),
        "median_floor_units_of_effects": med,
        "max_floor_units": max(above) if above else None,
        "rows_ge_4_floors": sum(u > 4 for u in above),
        "units_with_ge_4_floors": sum(any(abs(r["dM"]) > 4 * r["floor"] for r in rs) for rs in units.values()),
        "median_abs_dM": st.median(abs(r["dM"]) for r in allrows) if allrows else None,
        "to_donor_flips": sum(bool(r.get("to_donor")) for r in allrows),
        "answer_changed": sum(bool(r.get("changed")) for r in allrows),
        "canonical_nulls": len(can_null),
        "canonical_nulls_overturned": sum(pools[u]["verdict"] == EFFECT for u in can_null),
        "canonical_to_donor_flips": sum(bool(r and r.get("to_donor")) for r in can.values()),
    }


def q3(d):
    z = np.load(d / "q3_cells.npz")
    s = json.load(open(d / "q3_summary.json"))
    meta = s["meta"]
    rows = [dict(item=int(i), kind=int(k), start=int(a), row=int(r), layer=int(ly), kv=int(kv), k=int(dn),
                 dM=float(dm), floor=float(fl), informative=(None if inf < 0 else bool(inf)),
                 to_donor=bool(td), changed=bool(ch))
            for i, k, a, r, ly, kv, dn, dm, fl, inf, td, ch in zip(
                z["item"], z["kind"], z["start"], z["row"], z["layer"], z["kv"], z["donor"], z["dM"],
                z["floor"], z["inf"], z["to_donor"], z["changed"])]
    by = defaultdict(list)
    for r in rows:
        by[(r["kind"], r["start"])].append(r)
    out = {}

    def per_item(rs, same_len_only=False):
        u = defaultdict(list)
        for r in rs:
            if same_len_only and not meta[r["item"]]["same_len"][r["k"]]:
                continue
            u[r["item"]].append(r)
        return u

    out["C1_K1_carrier"] = claim(per_item(by[(1, 0)]))
    out["C1_K1_carrier_same_length_donors"] = claim(per_item(by[(1, 0)], True))
    for s0 in (0, 8, 16, 20):
        out[f"C2_Kband_from_L{s0}"] = claim(per_item(by[(2, s0)]))
    out["C3_Vband_from_L24"] = claim(per_item(by[(3, 24)]))
    out["ctrl_Vband_from_L20"] = claim(per_item(by[(3, 20)]))
    v1 = by[(0, 0)]
    gts = {i: tuple(m["gt"]) for i, m in enumerate(meta)}
    out["ctrl_V1_carrier"] = claim(per_item([r for r in v1 if (r["row"], r["layer"], r["kv"]) == gts[r["item"]]]))
    # C4: live (|dM| > 0.5) single V cells at the gt row outside L17..L21
    c4 = {"items": len(meta), "items_live_outside_canonical": 0, "items_live_outside_any_canonical": 0,
          "items_live_outside_16": 0, "items_floor_effect_outside_16": 0, "items_ge4floors_outside_16": 0,
          "live_outside_layers_16": defaultdict(int), "live_outside_layers_canonical": defaultdict(int)}
    for i, m in enumerate(meta):
        g = m["gt"][0]
        rs = [r for r in v1 if r["item"] == i and r["row"] == g and r["layer"] not in WINDOW]
        can = [r for r in rs if r["k"] == 0 and abs(r["dM"]) > 0.5]
        cana = [r for r in rs if r["k"] < m["n_canonical"] and abs(r["dM"]) > 0.5]
        live = [r for r in rs if abs(r["dM"]) > 0.5]
        c4["items_live_outside_canonical"] += bool(can)
        c4["items_live_outside_any_canonical"] += bool(cana)
        c4["items_live_outside_16"] += bool(live)
        c4["items_floor_effect_outside_16"] += any(abs(r["dM"]) > r["floor"] for r in rs)
        c4["items_ge4floors_outside_16"] += any(abs(r["dM"]) > 4 * r["floor"] for r in rs)
        for ly in sorted({r["layer"] for r in live}):
            c4["live_outside_layers_16"][ly] += 1
        for ly in sorted({r["layer"] for r in can}):
            c4["live_outside_layers_canonical"][ly] += 1
    c4["unit_share_16"] = c4["items_live_outside_16"] / len(meta)
    c4["verdict"] = "FLIPS" if c4["unit_share_16"] > 0.20 else "SURVIVES"
    c4["live_outside_layers_16"] = dict(sorted(c4["live_outside_layers_16"].items()))
    c4["live_outside_layers_canonical"] = dict(sorted(c4["live_outside_layers_canonical"].items()))
    out["C4_write_window"] = c4
    # C5: span V cells, canonical-donor nulls overturned
    cells = defaultdict(list)
    for r in v1:
        cells[(r["item"], r["row"], r["layer"], r["kv"])].append(r)
    can_null = flip = any_null = any_flip = 0
    for rs in cells.values():
        p = pool_rows(rs)["verdict"] == EFFECT
        c = [r for r in rs if r["k"] == 0]
        if c and abs(c[0]["dM"]) <= c[0]["floor"]:
            can_null += 1
            flip += p
        for r in rs:
            if abs(r["dM"]) <= r["floor"]:
                any_null += 1
                any_flip += p
    out["C5_span_cells"] = {"cells": len(cells), "canonical_nulls": can_null, "overturned": flip,
                            "fraction": flip / can_null if can_null else None,
                            "any_single_nulls": any_null, "any_single_overturned": any_flip,
                            "any_fraction": any_flip / any_null if any_null else None,
                            "cells_effect_16": sum(pool_rows(rs)["verdict"] == EFFECT for rs in cells.values())}
    # C6
    c6 = {r["prompt"]: [dict(x, k=i, to_donor=False, changed=not x["argmax_kept"]) for i, x in enumerate(r["per_donor"])]
          for r in s["C6"]}
    out["C6_t0_L14"] = claim(c6)
    out["C6_t0_L14"]["position_mean_dM_reproduced"] = [r["position_mean_dM"] for r in s["C6"]]
    out["C6_t0_L14"]["orig_position_dM"] = [r["orig_position_dM"] for r in s["C6"]]
    out["q3_crosscheck"] = s["crosscheck"]
    out["q3_null_vs_stepped_max"] = max(max(x for x in m["null_vs_stepped"] if x is not None) for m in meta)
    out["q3_donor_availability"] = {"min_available": min(m["n_available"] for m in meta),
                                    "same_len_donors_median": st.median(sum(m["same_len"]) for m in meta)}
    out["q3_cuda_peak_mb"] = s["cuda_peak_mb"]
    return out


def q25(d):
    s = json.load(open(d / "q25_records.json"))
    arms = defaultdict(lambda: defaultdict(list))
    for i, rec in enumerate(s["records"]):
        for r in rec["rows"]:
            arms[r["arm"]][i].append(r)
    out = {}
    names = {"k22kv1": "C7_k22kv1", "v22kv0": "C8_v22kv0", "A22h7": "C9_A22h7_resample",
             "f23n4197": "C10_f23n4197_resample", "h26": "C11_h26_row8", "h24": "info_h24_row8",
             "h22": "ctrl_h22_row8", "v22kv1": "ctrl_v22kv1"}
    for a, n in names.items():
        out[n] = claim(arms[a])
    for a in ("A22h7_zero", "A22h6_zero", "f23n4197_zero"):
        rs = [r for u in arms[a].values() for r in u]
        out[f"repro_{a}"] = {"mean_dM": st.mean(r["dM"] for r in rs), "median_dM": st.median(r["dM"] for r in rs),
                             "n": len(rs), "above_floor": sum(abs(r["dM"]) > r["floor"] for r in rs)}
    out["q25_crosscheck"] = s["crosscheck"]
    out["q25_null_vs_stepped_max"] = max(r["null_vs_stepped"] for r in s["records"])
    out["q25_cuda_peak_mb"] = s["cuda_peak_mb"]
    return out


def main(q3d, q25d, outp):
    res = {}
    if q3d != "-":
        res.update(q3(Path(q3d)))
    if q25d != "-":
        res.update(q25(Path(q25d)))
    json.dump(res, open(outp, "w"), indent=1, default=float)
    for k, v in res.items():
        if isinstance(v, dict) and "verdict" in v:
            print(f"{k:40s} {v['verdict']:45s} share {v.get('unit_share', v.get('unit_share_16'))}")


if __name__ == "__main__":
    main(*sys.argv[1:4])

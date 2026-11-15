"""Run 0 (no GPU): rescoring from saved jsonl. python3 run0.py -> run0.json

1. Hop forecast from Z2: the direct term c.delta is a pre-edit quantity (p fixed, Delta v known:
   P07), so M0 + direct is a real forecast; the per-layer linear term uses the layer's actual
   (post-edit) input change, so it is accounting, not a forecast (audit rule 1). Scores both.
2. Capital co-readout beside every country readout saved in Z1.
3. Y2: declared-carrier readout (gt site) vs the self-chosen site, where the saved fields allow it.
4. B_answer pre-read (L20.h): not saved; listed as a Run 1 item.
"""
import json
import statistics as st
from collections import Counter

EXT = "../ext/"
out = {}


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


def q(xs):
    xs = sorted(xs)
    if len(xs) < 2:
        return {"values": xs}
    qs = st.quantiles(xs, n=4)
    return {"median": st.median(xs), "q1": qs[0], "q3": qs[2], "n": len(xs)}


# 1. hop forecast
z2 = [json.loads(line) for line in open(EXT + "z2-full.jsonl")]
hop = {}
for arm in ("donor", "ablate", "city"):
    A = [r["arms"][arm] for r in z2 if arm in r["arms"]]
    rows = []
    for a in A:
        e = a["edge"]
        m0 = a["M0"][0] - a["M0"][1]
        m1 = a["M1"][0] - a["M1"][1]
        nat = e["consumer"]["native_delta"]["value"]
        direct = e["direct"]["value"]
        lin = e["totals"]["mlp_linear"]["value"]
        rows.append({"m0": m0, "m1": m1, "nat": nat, "direct": direct, "lin": lin,
                     "mlp": e["totals"]["mlp"]["value"], "layers": e["mlp"]})
    flips = [(r["m0"] > 0) != (r["m1"] > 0) for r in rows]
    d = {"pairs": len(rows), "observed_flips": sum(flips)}
    d["forecast_direct_only (pre-edit)"] = {
        "flip_mismatch": rate(((r["m0"] > 0) != (r["m0"] + r["direct"] > 0)) != f
                              for r, f in zip(rows, flips)),
        "abs_error": q([abs(r["nat"] - r["direct"]) for r in rows]),
        "share_of_native": q([r["direct"] / r["nat"] for r in rows if r["nat"]])}
    d["accounting_direct_plus_local_linear (post-edit inputs)"] = {
        "flip_mismatch": rate(((r["m0"] > 0) != (r["m0"] + r["direct"] + r["lin"] > 0)) != f
                              for r, f in zip(rows, flips)),
        "abs_error": q([abs(r["nat"] - r["direct"] - r["lin"]) for r in rows])}
    per_layer = {}
    for r in rows:
        for m in r["layers"]:
            ex, li = m["exact"]["value"], m["linear"]["value"]
            if abs(ex) > 0.25:
                per_layer.setdefault(m["layer"], []).append(li / ex)
    d["per_layer_linear_over_exact (|exact| > 0.25)"] = {k: q(v) for k, v in sorted(per_layer.items())}
    lin_in = {}
    for r in rows:
        for m in r["layers"]:
            lin_in.setdefault(m["layer"], []).append(m["input_linearity"]["cos_da"]["value"])
    d["input_cos_da_by_layer (Jacobian Delta a vs captured)"] = {k: round(st.median(v), 3)
                                                                for k, v in sorted(lin_in.items())}
    hop[arm] = d
out["1_hop_forecast"] = hop

# 2. capital co-readout (Z1)
z1 = [json.loads(line) for line in open(EXT + "z1-full.jsonl")]
co = {}


def pair(get):
    cc = [bool(get(r)["countries"]["names_truth"]) for r in z1]
    ca = [bool(get(r)["capitals"]["names_truth"]) for r in z1]
    return {"country": rate(cc), "capital": rate(ca),
            "both": rate(a and b for a, b in zip(cc, ca)),
            "capital_without_country": sum(b and not a for a, b in zip(cc, ca)),
            "country_without_capital": sum(a and not b for a, b in zip(cc, ca))}


co["view (bridge cell)"] = pair(lambda r: r["table"]["bridge"]["content"])
co["r0rest"] = pair(lambda r: r["table"]["r0rest"]["content"])
co["B_mass"] = pair(lambda r: r["B_mass"])
co["B_answer (L20.h_out)"] = pair(lambda r: r["B_answer"])
co["B_lens L20"] = pair(lambda r: r["B_lens"]["L20"])
co["B_lens L24"] = pair(lambda r: r["B_lens"]["L24"])
co["all-set best kind (view)"] = dict(Counter(r["table"]["bridge"]["content"]["all"]["kind"] for r in z1))
co["T_diff after-edit capital"] = "not saved (only country and all-set texts); Run 1 offline replay"
out["2_capital_coreadout"] = co

# 3. Y2 declared carrier vs self-chosen site
cand = {i["id"]: i for i in json.load(open(EXT + "candidates_ext.json"))["items"]}
y2 = [json.loads(line) for p in ("y2-17-pd.jsonl", "y2-17-lol.jsonl") for line in open(EXT + p)]
ok = [r for r in y2 if r["correct"]]
with_gt = [r for r in ok if "gt" in r]
same = [r for r in with_gt if tuple(r["gt"]["site"]) == (cand[r["id"]]["row"], 20, 6)]
out["3_y2_declared_vs_self"] = {
    "correct_items": len(ok), "with_gt": len(with_gt),
    "gt_site_equals_last_row_L20_KV6": len(same),
    "declared_gt_carrier_readable (rows where gt = last-row L20 KV6, read = const field)":
        rate(r["const_named_ok"] for r in same),
    "self_chosen_view_readable on the same items": rate(r["bridge_named_ok"] for r in same),
    "self_chosen_site_equals_gt": rate(tuple(r["bridge_site"]) == tuple(r["gt"]["site"]) for r in with_gt),
    "rest": f"{len(with_gt) - len(same)} gt items whose carrier is not the last-row cell, and "
            f"{len(ok) - len(with_gt)} correct items without gt: declared readout not saved (Run 1)",
    "capital co-readout": "not saved in Y2 (hub tokens only): Run 1 offline replay"}
out["4_B_answer_pre_read"] = "L20.h not saved in Z1: moved to Run 1 (answer-row h per layer is captured)"
json.dump(out, open("run0.json", "w"), indent=1)
print(json.dumps(out, indent=1))

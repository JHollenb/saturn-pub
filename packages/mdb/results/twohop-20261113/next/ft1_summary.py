"""FT1 verdicts (prereg_firsttoken.json): python3 ft1_summary.py ft1-full.jsonl -> ft1-full-summary.json"""
import json
import statistics as st
import sys
from collections import Counter

rows = [json.loads(line) for line in open(sys.argv[1])]


def rate(xs):
    xs = list(xs)
    return [sum(xs), len(xs), round(sum(xs) / len(xs), 3) if xs else None]


out = {"n": len(rows), "tokenizer_matches": rate(r["tokenizer_matches"] for r in rows)}
for g in ("default", "correct"):
    rs = [r for r in rows if r["group"] == g]
    d = {"n": len(rs), "argmax": dict(Counter(r["argmax_text"] for r in rs).most_common(5)),
         "t0_rank_share": dict(Counter(r["a"]["t0_rank_share"] for r in rs)),
         "t0_rank_contrast": dict(Counter(r["a"]["t0_rank_contrast"] for r in rs)),
         "t0_rank1_share": rate(r["a"]["t0_rank_share"] == 1 for r in rs),
         "t0_rank_contrast_median": st.median(r["a"]["t0_rank_contrast"] for r in rs),
         "t0_mass_fraction_median": st.median(r["a"]["t0_mass"] / r["a"]["mass_total"] for r in rs),
         "t0_share_median": st.median(r["a"]["t0_share"] for r in rs),
         "t0_contrast_median": st.median(r["a"]["t0_contrast"] for r in rs),
         "top_share_rows_examples": [r["a"]["top_share_rows"][:3] for r in rs[:3]],
         "top_contrast_rows_examples": [r["a"]["top_contrast_rows"][:3] for r in rs[:3]]}
    for w in ("A", "In", "Which", "A_both_lines"):
        d[f"swap_{w}"] = {"follows": rate(r["b"][w]["follows"] for r in rs),
                          "unchanged": rate(r["b"][w]["unchanged"] for r in rs),
                          "outputs": dict(Counter(r["b"][w]["text"] for r in rs).most_common(4))}
    d["swap_pooled_A_In_Which_follows"] = rate(r["b"][w]["follows"] for r in rs for w in ("A", "In", "Which"))
    d["swap_pooled_A_In_Which_unchanged"] = rate(r["b"][w]["unchanged"] for r in rs for w in ("A", "In", "Which"))
    for k in ("t0_zero", "t0_mask", "random_zero", "random_mask"):
        d[k] = {"moved": rate(r["c"][k]["moved"] for r in rs),
                "dM_median": st.median(r["c"][k]["dM"] for r in rs),
                "outputs": dict(Counter(r["c"][k]["text"] for r in rs).most_common(5))}
    d["random_mass_ratio_median"] = st.median(r["c"]["random_mass_ratio_median"] for r in rs)
    sl = [r["c"]["single_layer"] for r in rs]
    d["single_layer"] = {"layers": dict(Counter(s["layer"] for s in sl)),
                         "zero_within_2sp": rate(s["zero"]["within_2_spacings"] for s in sl),
                         "mask_within_2sp": rate(s["mask"]["within_2_spacings"] for s in sl),
                         "zero_abs_err_median": st.median(abs(s["zero"]["error"]) for s in sl),
                         "mask_abs_err_median": st.median(abs(s["mask"]["error"]) for s in sl),
                         "zero_pred_median": st.median(s["zero"]["predicted"] for s in sl),
                         "mask_pred_median": st.median(s["mask"]["predicted"] for s in sl),
                         "zero_rel_err_median": st.median(abs(s["zero"]["error"]) / max(abs(s["zero"]["measured"]), 1e-9) for s in sl),
                         "mask_rel_err_median": st.median(abs(s["mask"]["error"]) / max(abs(s["mask"]["measured"]), 1e-9) for s in sl)}
    out[g] = d
D, C = out["default"], out["correct"]
follow = D["swap_pooled_A_In_Which_follows"][2]
out["verdicts"] = {
    "S1 t0 rank1 by share >= 0.75 and contrast rank median >= 3": (
        rate(r["a"]["t0_rank_share"] == 1 for r in rows)[2] >= 0.75
        and st.median(r["a"]["t0_rank_contrast"] for r in rows) >= 3),
    "S2 default follows >= 0.50 and correct unchanged >= 0.90": follow >= 0.50 and C["swap_pooled_A_In_Which_unchanged"][2] >= 0.90,
    "S3 zero moves default >= 0.5 and correct >= 0.5; mask moves correct <= 0.25": (
        D["t0_zero"]["moved"][2] >= 0.5 and C["t0_zero"]["moved"][2] >= 0.5 and C["t0_mask"]["moved"][2] <= 0.25),
    "S4 random cuts move correct <= 0.10 (zero and mask)": C["random_zero"]["moved"][2] <= 0.10 and C["random_mask"]["moved"][2] <= 0.10,
    "S5 closed forms within 2 spacings >= 0.90 (both)": all(
        out[g]["single_layer"][k][2] >= 0.90 for g in ("default", "correct") for k in ("zero_within_2sp", "mask_within_2sp")),
    "FALSIFIER (follows <= 0.50 AND zero and mask t0 cuts move default <= 0.50)": follow <= 0.50 and D["t0_zero"]["moved"][2] <= 0.50 and D["t0_mask"]["moved"][2] <= 0.50,
}
json.dump(out, open(sys.argv[1].replace(".jsonl", "-summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))

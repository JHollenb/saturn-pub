"""Cross-model verdicts against prereg_scale.json. python combine.py score-*.json > verdicts.json"""
import json
import sys

BANDS_ACC = {  # memo §1(b): d2, d3 pooled, d4 (lo, hi)
    "Qwen3-0.6B-Base": ((0.70, 0.90), (0.40, 0.65), (0.0, 0.25)),
    "Qwen3-4B-Base": ((0.95, 1.0), (0.90, 1.0), (0.60, 0.80)),
    "Qwen2.5-0.5B": ((0.60, 0.85), (0.30, 0.60), (0.0, 0.20)),
    "Qwen2.5-1.5B": ((0.90, 1.0), (0.70, 0.90), (0.30, 0.50)),
    "Qwen2.5-3B": ((0.90, 1.0), (0.80, 1.0), (0.50, 0.75)),
}
PICK = {"Qwen3-0.6B-Base": (19, 22), "Qwen3-4B-Base": (24, 27), "Qwen2.5-0.5B": (16, 18), "Qwen2.5-3B": (25, 28),
        "Qwen3-1.7B-Base": (20, 20), "Qwen2.5-1.5B": (22, 22)}
A1R = {"Qwen3-0.6B-Base": (18, 22), "Qwen3-4B-Base": (23, 28), "Qwen2.5-0.5B": (16, 19), "Qwen2.5-3B": (23, 28)}
A2R = {"Qwen3-0.6B-Base": (19, 23), "Qwen3-4B-Base": (27, 31), "Qwen2.5-0.5B": (15, 19), "Qwen2.5-3B": (27, 31)}
MLP = {"Qwen3-4B-Base": (0.60, 0.75), "Qwen2.5-3B": (0.55, 0.75), "Qwen3-0.6B-Base": (0.45, 0.65),
       "Qwen2.5-0.5B": (0.40, 0.60), "Qwen2.5-1.5B": (0.35, 0.55)}
QWEN3 = ("Qwen3-0.6B-Base", "Qwen3-1.7B-Base", "Qwen3-4B-Base")


def inside(x, band):
    return x is not None and band[0] <= x <= band[1]


def r(f):
    return f[0] / f[1] if f and f[1] else None


S = {}
for p in sys.argv[1:]:
    d = json.load(open(p))
    S[d["model"]["model"]] = d
out = {"models": {}, "verdicts": {}}
for name, d in S.items():
    m, F = d["model"], d["families"]
    acc = {f: F[f]["acc"] for f in F}
    d3 = (F["d3-lm"]["correct"] + F["d3-per"]["correct"]) / (F["d3-lm"]["n"] + F["d3-per"]["n"])
    base = [F[f] for f in ("currency", "person", "landmark", "city")]
    agree = [sum(f["carrier_agree_correct"][0] for f in base), sum(f["carrier_agree_correct"][1] for f in base)]
    names = [sum(f["names_country_correct"][0] for f in base), sum(f["names_country_correct"][1] for f in base)]
    ww = [sum(f["write_window_le5_correct"][0] for f in base), sum(f["write_window_le5_correct"][1] for f in base)]
    row = {"L_c": m["L_c"], "kv_c": m["kv_c"], "layers": m["layers"], "frac": m["frac_depth"], "tail": m["tail"],
           "mode": m["mode"], "pick": PICK.get(name), "in_pick": inside(m["L_c"], PICK.get(name, (-1, -1))),
           "in_A1": inside(m["L_c"], A1R[name]) if name in A1R else None,
           "in_A2": inside(m["L_c"], A2R[name]) if name in A2R else None,
           "accuracy": {"d2": acc["d2"], "d3": d3, "d4": acc["d4-work"], **acc},
           "acc_bands_hit": [inside(x, b) for x, b in zip((acc["d2"], d3, acc["d4-work"]), BANDS_ACC[name])] if name in BANDS_ACC else None,
           "base_carrier_agree": agree, "base_carrier_names_country": names, "base_write_window_le5": ww,
           "concentration_by_family": {f: F[f]["concentration_correct"]["median"] for f in F},
           "mlp_share": {d_: m["split"].get(d_, {}).get("mlp", {}) for d_ in ("0.5", "1.0")},
           "mlp_band": MLP.get(name), "mlp_in_band": inside(m["split"].get("1.0", {}).get("mlp", {}).get("median"), MLP[name]) if name in MLP else None,
           "dose_move": m["split_dose_move_median"], "B2_auc": m["B2_auc_slack_d3d4"], "B6": m["B6"],
           "projected_language": [m["projected_language_hits"], m["projected_language_shuffled_hits"]]}
    out["models"][name] = row
# (a) A1 vs A2 on the 36-layer models
l36 = {n: S[n]["model"]["L_c"] for n in ("Qwen3-4B-Base", "Qwen2.5-3B") if n in S}
if len(l36) == 2:
    out["verdicts"]["a_rule"] = ("A1" if all(v <= 28 for v in l36.values()) else
                                 "A2" if all(v >= 29 for v in l36.values()) else "neither (model-specific)")
out["verdicts"]["a_mode_ge_0.80"] = {n: r(out["models"][n]["mode"]) for n in out["models"]}
out["verdicts"]["d2_kv_index_qwen3"] = {n: S[n]["model"]["kv_c"] for n in QWEN3 if n in S}
# B4 across (model, family) cells
cells = []
for n, d in S.items():
    for f, t in d["families"].items():
        if t["n"] >= 8 and t["readable_error_share_wrong"][1]:
            cells.append((r(t["readable_error_share_wrong"]), 1 - t["acc"], n, f, t["acc"]))
out["verdicts"]["B4_cells"] = [{"model": c[2], "family": c[3], "acc": c[4], "readable_share": c[0]} for c in cells]
if len(cells) >= 3:
    import torch
    a = torch.tensor([c[0] for c in cells]).double()
    b = torch.tensor([c[1] for c in cells]).double()
    ra, rb = torch.argsort(torch.argsort(a)).double(), torch.argsort(torch.argsort(b)).double()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    out["verdicts"]["B4_spearman"] = float(ra @ rb / (ra.norm() * rb.norm()))
json.dump(out, sys.stdout, indent=1, default=str)

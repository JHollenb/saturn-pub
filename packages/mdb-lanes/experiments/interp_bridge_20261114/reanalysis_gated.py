"""Post-hoc re-analysis (2026-11-14, after the audit): attribution vs raw mass vs gated mass, sign-free.

  python reanalysis_gated.py   -> reanalysis-gated.json   (numpy + PIL; saved data only; no model)

Data: smoke job-8fd7bb3b1604 (attribution terms), cuts job-1d4b6df1ba24 (designed groups), and the
outer attention gate RMS per (step, block) logged in job-6d9fea396262 (predicted-write addendum 2,
native root run of the cat prompt; the gates depend on the timestep only, and the log is identical
for both scenes there).

Predictors per head group (sums over the group's heads):
* r       first-order attribution (PREREG; nd direction), r_id the same on the iso-minus-deleted direction;
* mass    read mass M = sum p over the word's row (PREREG; image + text receivers, no gate);
* gmass   gated read mass: sum over steps of M_img[s,b,h] * rms(g_img[s,b]) + M_txt[s,b,h] * rms(g_txt[s,b]);
* gwmass  gated write mass: the same gate weighting of wmass = sum p |W_O,h v| (alternative).
Outcomes per group: signed effect on the nd direction (as PREREG), its size |effect| (scored with
|r| for attribution; masses are unsigned), pixel gone (lane way, z:none endpoint as PREREG), and the
effect on the iso-minus-deleted direction (scored with r_id, and its size with |r_id|).
Statistics: Spearman over the 20 groups of a partition, leave-one-out min/max, and the leave-one-out
range of the gap rho(attribution) - rho(gated mass). All 8 conditions (2 words x 2 cut types x 2 partitions).
This is not preregistered; it answers the audit's question "is the gap sign or size".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze import flat, head_predictors, loo_gap, loo_summary, mad, terms  # noqa: E402
from bridge import spearman  # noqa: E402

RES = Path("/Users/jakeholl/domains/artifacts/mdb-interp-bridge-20261114/results")
SMOKE = RES / "job-8fd7bb3b1604/interp-bridge"
CUTS = RES / "job-1d4b6df1ba24/interp-bridge"
GATES = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results/job-6d9fea396262/"
             "predicted-write-vswap2/vswap2.snow.json")
WORDS, CUT_TYPES = ("red", "ball"), ("zero", "mask")


def gate_table():
    log = json.loads(GATES.read_text())["gate_rms_native_root"]
    if len(log) != 100:
        raise ValueError("expected 4 steps x 25 blocks of gate entries")
    gi, gt = np.zeros((4, 25)), np.zeros((4, 25))
    for n, (b, ri, rt, _, _) in enumerate(log):
        s = n // 25
        if b != n % 25:
            raise ValueError("gate log is not in block order")
        gi[s, b], gt[s, b] = ri, rt
    return gi, gt


def predictors(T, gi, gt):
    P = head_predictors(T)
    P["r_id_zero"] = head_predictors(T, direction="id")["r_zero"]
    P["r_id_mask"] = head_predictors(T, direction="id")["r_mask"]
    g = lambda a, gate: (T[a] * gate[:, :, None]).sum(0).reshape(-1)   # noqa: E731
    P["gmass"] = g("nd.own.mass_img", gi) + g("nd.own.mass_txt", gt)
    P["gwmass"] = g("nd.own.wmass_img", gi) + g("nd.own.wmass_txt", gt)
    return P


def stat(x, y):
    s = loo_summary(list(x), list(y))
    return {"rho": s["rho"], "loo_min": s["loo_min"], "loo_max": s["loo_max"]}


def main():
    from PIL import Image
    gi, gt = gate_table()
    rows = {r["arm"]: r for r in json.loads((CUTS / "rows.json").read_text())}
    G = json.loads((HERE / "groups.json").read_text())
    img = lambda name: np.asarray(Image.open(CUTS / "images" / f"{name.replace(':', '_')}.png").convert("RGB"))  # noqa: E731
    out = {"note": __doc__, "gate_rms_joint_img_over_single": float(gi[:, :5].mean() / gi[:, 5:].mean()),
           "gate_rms_joint_txt_over_single": float(gt[:, :5].mean() / gi[:, 5:].mean()), "conditions": {}}
    for w in WORDS:
        P = predictors(terms(SMOKE, w), gi, gt)
        mad_nd = mad(img("root:native"), img(f"root:del:{w}"))
        d = lambda r: 1.0 - r["vs_del"]["px_mad"] / mad_nd                             # noqa: E731
        d_iso = 1.0 - mad(img(f"root:iso:{w}"), img(f"root:del:{w}")) / mad_nd
        d_none = d(rows[f"z:none:{w}"])
        for cut in CUT_TYPES:
            for part in ("anti", "diag"):
                br = [b for b in G["branches"] if b["word"] == w and b["cut"] == cut and b["partition"] == part]
                X = {k: [] for k in ("r", "r_id", "mass", "gmass", "gwmass")}
                Y = {k: [] for k in ("effect", "effect_id", "gone_px")}
                for b in br:
                    r = rows[b["id"]]
                    idx = flat(b["units"])
                    X["r"].append(float(P[f"r_{cut}"][idx].sum()))
                    X["r_id"].append(float(P[f"r_id_{cut}"][idx].sum()))
                    for k in ("mass", "gmass", "gwmass"):
                        X[k].append(float(P[k][idx].sum()))
                    Y["effect"].append(r["y_nd_iso"] - r["y_nd"])
                    Y["effect_id"].append(r["y_id_iso"] - r["y_id"])
                    Y["gone_px"].append((d(r) - d_iso) / (d_none - d_iso))
                A = {k: np.asarray(v) for k, v in {**X, **Y}.items()}
                outcomes = {
                    "signed_nd": (A["effect"], A["r"]),
                    "size_nd": (np.abs(A["effect"]), np.abs(A["r"])),
                    "gone_px": (A["gone_px"], A["r"]),
                    "signed_id": (A["effect_id"], A["r_id"]),
                    "size_id": (np.abs(A["effect_id"]), np.abs(A["r_id"])),
                }
                cond = {"n": len(br), "negative_effects": int((A["effect"] < 0).sum()),
                        "sign_agreement_r": int((np.sign(A["effect"]) == np.sign(A["r"])).sum())}
                for oname, (y, att) in outcomes.items():
                    rec = {"attribution": stat(att, y), "mass": stat(A["mass"], y), "gmass": stat(A["gmass"], y),
                           "gwmass": stat(A["gwmass"], y)}
                    lg = loo_gap(list(att), list(A["gmass"]), list(y))
                    lm = loo_gap(list(att), list(A["mass"]), list(y))
                    rec["gap_att_minus_gmass"] = rec["attribution"]["rho"] - rec["gmass"]["rho"]
                    rec["gap_att_minus_gmass_loo"] = [min(lg), max(lg)]
                    rec["gap_att_minus_mass"] = rec["attribution"]["rho"] - rec["mass"]["rho"]
                    rec["gap_att_minus_mass_loo"] = [min(lm), max(lm)]
                    cond[oname] = rec
                cond["spearman_mass_vs_gmass_over_groups"] = spearman(X["mass"], X["gmass"])
                out["conditions"][f"{w}.{cut}.{part}"] = cond
    summ = {}
    for part in ("anti", "diag"):
        for oname in ("signed_nd", "size_nd", "gone_px", "signed_id", "size_id"):
            cs = [out["conditions"][f"{w}.{c}.{part}"][oname] for w in WORDS for c in CUT_TYPES]
            summ[f"{part}.{oname}"] = {
                "mean_rho": {k: float(np.mean([c[k]["rho"] for c in cs])) for k in ("attribution", "mass", "gmass", "gwmass")},
                "mean_gap_att_minus_gmass": float(np.mean([c["gap_att_minus_gmass"] for c in cs])),
                "mean_gap_att_minus_mass": float(np.mean([c["gap_att_minus_mass"] for c in cs])),
                "att_beats_gmass_conditions": int(sum(c["gap_att_minus_gmass"] > 0 for c in cs)),
                "loo_gap_vs_gmass_min_above_0": int(sum(c["gap_att_minus_gmass_loo"][0] > 0 for c in cs))}
    out["summary"] = summ
    (HERE / "reanalysis-gated.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("gate ratios", out["gate_rms_joint_img_over_single"], out["gate_rms_joint_txt_over_single"])
    for k, v in summ.items():
        print(f"{k:16s}", {kk: round(vv, 3) for kk, vv in v["mean_rho"].items()}, "gap vs gmass", round(v["mean_gap_att_minus_gmass"], 3),
              "vs mass", round(v["mean_gap_att_minus_mass"], 3), "wins", v["att_beats_gmass_conditions"], "loo>0", v["loo_gap_vs_gmass_min_above_0"])
    for k, c in out["conditions"].items():
        print(k, "neg", c["negative_effects"], "signs", c["sign_agreement_r"], "m~gm", round(c["spearman_mass_vs_gmass_over_groups"], 3),
              {o: (round(c[o]["attribution"]["rho"], 3), round(c[o]["mass"]["rho"], 3), round(c[o]["gmass"]["rho"], 3)) for o in
               ("signed_nd", "size_nd", "gone_px", "signed_id", "size_id")})


if __name__ == "__main__":
    main()

"""Offline re-analysis of the predicted-write experiment with gated mass (saved data only, no model).

  PYTHONPATH=<worktree>/packages/mdb-lanes/src:<worktree>/packages/mdb/src python reanalysis_pw.py
  -> results/reanalysis-predicted-write.json

Every forecast table again, with five per-unit scores side by side: raw mass (the lane's
``word_masses``), gated mass (``gated_unit_mass``, scales as select2.py), PW_both, image-receiver PW
and text-receiver PW (both in PW_both units, so PW_img + PW_txt = PW_both).
* F1 / F2 / F4 cells: set totals of each score beside gone (``results/arms.json``);
* F3: per (word, cut type), Spearman over the 16 sets between gone and each score's set total
  (PW of the arm's cut type for the PW scores, as the prereg; PW_both too), plus the cell wins
  of PW over gated mass;
* A1: Spearman over 600 units and top-60 overlaps of each score against gated mass;
* sizes: where the mass of the massmatch-lowpw sets sits (joint units) under raw and gated mass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import plans as PL  # noqa: E402
import select2 as S2  # noqa: E402
import stage as ST  # noqa: E402

CAPTURE = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results/job-86df2dd47809/predicted-write")


def main():
    from mdb_lanes import predicted_write as PW
    tokens = json.loads((CAPTURE / "tokens.json").read_text())
    sets = json.loads((HERE.parent / "plans/sets.json").read_text())
    arms = json.loads((HERE.parent / "results/arms.json").read_text())
    eff = arms["effects"]
    out = {"scores": "mass (raw), gmass (gated), pw_both, pw_img, pw_txt (image + text = both)", "words": {}}
    f3cells = []
    pre = {"zero": "z", "renorm": "m"}
    for w in ST.WORDS:
        sc = S2.scores(CAPTURE, w, tokens)
        z = np.load(CAPTURE / f"score.{w}.iso.npz")
        zt = (z["zero_img"] + z["zero_oth"]).sum(0).ravel().astype(np.float64)
        mt = (z["mask_img"] + z["mask_oth"]).sum(0).ravel().astype(np.float64)
        typed = {"zero": {"pw_img": z["zero_img"].sum(0).ravel() / zt.sum(), "pw_txt": z["zero_oth"].sum(0).ravel() / zt.sum(),
                          "pw": zt / zt.sum()},
                 "renorm": {"pw_img": z["mask_img"].sum(0).ravel() / mt.sum(), "pw_txt": z["mask_oth"].sum(0).ravel() / mt.sum(),
                            "pw": mt / mt.sum()}}
        S = {k: PL.flat(v) for k, v in sets[w].items()}
        keys = ("mass", "gmass", "pw_both", "pw_img", "pw_txt")
        joint = np.array([i // 24 < 5 for i in range(600)])

        def tot(key, s):
            return float(sc[key][s].sum())
        main_sets = ("pw60", "own60", "massmatch-lowpw", "pwmatch-lowmass", "rand60a", "rand60b")
        rec = {"set_totals": {k: {**{m: tot(m, S[k]) for m in keys},
                                  "joint_units": int(joint[S[k]].sum()),
                                  "mass_in_joint_units": float(sc["mass"][S[k]][joint[S[k]]].sum()),
                                  "gmass_in_joint_units": float(sc["gmass"][S[k]][joint[S[k]]].sum())}
                              for k in main_sets},
               "gone_main": {f"{t}/{k}": eff["main"].get(f"{pre[t]}:{k}:{w}", {}).get("gone")
                             for t in pre for k in main_sets},
               "spearman_600": {f"{a}_vs_{b}": PW.spearman(sc[a], sc[b]) for i, a in enumerate(keys)
                                for b in keys[i + 1:]},
               "top60_overlap_with_gmass_top60": {k: PW.overlap(PW.top_units(sc[k], 60), PW.top_units(sc["gmass"], 60))
                                                  for k in keys},
               "word_mass_in_joint_blocks": float(sc["mass"][joint].sum()),
               "word_gmass_in_joint_blocks": float(sc["gmass"][joint].sum()),
               "word_pw_both_in_joint_blocks": float(sc["pw_both"][joint].sum()) / 2.0}
        # F1 / F2 / F4 with gated mass: what each rule's sets carry
        rec["F1_cells_gated"] = {t: {"gone_mml": rec["gone_main"][f"{t}/massmatch-lowpw"],
                                     "gone_own60": rec["gone_main"][f"{t}/own60"],
                                     "gmass_mml_over_own60": tot("gmass", S["massmatch-lowpw"]) / tot("gmass", S["own60"]),
                                     "mass_mml_over_own60": tot("mass", S["massmatch-lowpw"]) / tot("mass", S["own60"]),
                                     "pw_img_mml_over_own60": tot("pw_img", S["massmatch-lowpw"]) / tot("pw_img", S["own60"]),
                                     "pw_txt_mml_over_own60": tot("pw_txt", S["massmatch-lowpw"]) / tot("pw_txt", S["own60"]),
                                     "gated_mass_predicts_below_third": bool(
                                         tot("gmass", S["massmatch-lowpw"]) < tot("gmass", S["own60"]) / 3)}
                                 for t in pre}
        rec["F2_cells_gated"] = {t: {"gone_pml": rec["gone_main"][f"{t}/pwmatch-lowmass"],
                                     "gone_pw60": rec["gone_main"][f"{t}/pw60"],
                                     "gmass_pml_over_pw60": tot("gmass", S["pwmatch-lowmass"]) / tot("gmass", S["pw60"]),
                                     "pw_txt_pml_over_pw60": tot("pw_txt", S["pwmatch-lowmass"]) / tot("pw_txt", S["pw60"]),
                                     "pw_img_pml_over_pw60": tot("pw_img", S["pwmatch-lowmass"]) / tot("pw_img", S["pw60"])}
                                 for t in pre}
        rec["F4_gmass_rand_over_pw60"] = {r: tot("gmass", S[r]) / tot("gmass", S["pw60"]) for r in ("rand60a", "rand60b")}
        # F3
        names = sorted(k for k in S if k.startswith("f3s"))
        for t in pre:
            y = [eff["f3"][f"{pre[t]}:{n}:{w}"]["gone"] for n in names]
            ypx = [eff["f3"][f"{pre[t]}:{n}:{w}"]["px_vs_iso"] for n in names]
            xs = {"mass": [tot("mass", S[n]) for n in names], "gmass": [tot("gmass", S[n]) for n in names],
                  "pw_type": [float(typed[t]["pw"][S[n]].sum()) for n in names],
                  "pw_both": [tot("pw_both", S[n]) for n in names],
                  "pw_img_type": [float(typed[t]["pw_img"][S[n]].sum()) for n in names],
                  "pw_txt_type": [float(typed[t]["pw_txt"][S[n]].sum()) for n in names]}
            cell = {"word": w, "type": t,
                    "spearman_gone": {k: PW.spearman(v, y) for k, v in xs.items()},
                    "spearman_px_vs_iso": {k: PW.spearman(v, ypx) for k, v in xs.items()},
                    "spearman_gmass_vs_pw_type_over_sets": PW.spearman(xs["gmass"], xs["pw_type"])}
            f3cells.append(cell)
        out["words"][w] = rec
    keys = ("mass", "gmass", "pw_type", "pw_both", "pw_img_type", "pw_txt_type")
    mean = {k: float(np.mean([c["spearman_gone"][k] for c in f3cells])) for k in keys}
    out["F3"] = {"cells": f3cells, "mean_spearman_gone": mean,
                 "pw_type_minus_gmass": {"mean": float(np.mean([c["spearman_gone"]["pw_type"] - c["spearman_gone"]["gmass"]
                                                                for c in f3cells])),
                                         "pw_higher_cells": int(sum(c["spearman_gone"]["pw_type"] > c["spearman_gone"]["gmass"]
                                                                    for c in f3cells))},
                 "pw_type_minus_mass": {"mean": float(np.mean([c["spearman_gone"]["pw_type"] - c["spearman_gone"]["mass"]
                                                               for c in f3cells])),
                                        "pw_higher_cells": int(sum(c["spearman_gone"]["pw_type"] > c["spearman_gone"]["mass"]
                                                                   for c in f3cells))},
                 "gmass_minus_mass": {"mean": float(np.mean([c["spearman_gone"]["gmass"] - c["spearman_gone"]["mass"]
                                                             for c in f3cells])),
                                      "gmass_higher_cells": int(sum(c["spearman_gone"]["gmass"] > c["spearman_gone"]["mass"]
                                                                    for c in f3cells))},
                 "mean_spearman_px_vs_iso": {k: float(np.mean([c["spearman_px_vs_iso"][k] for c in f3cells])) for k in keys}}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/reanalysis-predicted-write.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("F3 mean spearman gone", {k: round(v, 3) for k, v in mean.items()})
    print("PW-gmass", out["F3"]["pw_type_minus_gmass"], "PW-mass", out["F3"]["pw_type_minus_mass"], "g-m", out["F3"]["gmass_minus_mass"])
    print("px", {k: round(v, 3) for k, v in out["F3"]["mean_spearman_px_vs_iso"].items()})
    for c in f3cells:
        print(c["word"], c["type"], {k: round(v, 3) for k, v in c["spearman_gone"].items()})
    for w, r in out["words"].items():
        print(w, "joint share mass/gmass/pw", round(r["word_mass_in_joint_blocks"], 3), round(r["word_gmass_in_joint_blocks"], 3),
              round(r["word_pw_both_in_joint_blocks"], 3))
        for t, c in r["F1_cells_gated"].items():
            print("  F1", t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})
        for t, c in r["F2_cells_gated"].items():
            print("  F2", t, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})


if __name__ == "__main__":
    main()

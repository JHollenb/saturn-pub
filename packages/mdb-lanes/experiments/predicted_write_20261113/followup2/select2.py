"""Addendum-2 head sets (rules in PREREG-ADDENDUM-2.md), from the saved capture job-86df2dd47809 only.

  PYTHONPATH=<worktree>/packages/mdb-lanes/src:<worktree>/packages/mdb/src  python select2.py --capture DIR --out DIR

Per word (unit order: flat ``block * 24 + head``):
* gmass (gated read mass) = ``flux_read_panels.gated_unit_mass`` of the lane mass
  (``word_masses``), scales from ``stratum_gate_scale(PW_zero image receivers, unit_mass(iso tok_w))``
  of the capture's iso run;
* gmass60: top 60 by gmass;
* massmatch-lowpw-single: 60 single-block units (blocks 5..24), total raw mass within 10% of own60's
  raw mass, minimum PW_both; when the band is unreachable inside the single blocks, the closest
  reachable set (see ``closest_single``) and its actual mass ratio;
* massmatch-lowpw-single-own (added because the band above is unreachable for every word): 60
  single-block units, total raw mass within 10% of own60's single-block raw mass, minimum PW_both;
* gmassmatch-lowtextpw: 60 units, total gmass within 10% of pw60's gmass, minimum text-receiver PW
  (PW_txt = zero_oth share + mask_oth share);
* rand60c / rand60d: ``matched_control_sets`` against gmass60 (stratum counts and distinct blocks
  matched), disjoint, excluding every named set (old and new) and each other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import plans as PL  # noqa: E402
import stage as ST  # noqa: E402

FS = ST.FS
SEED2 = 20261114
N = 60


def scores(capture, word, tokens):
    from mdb_lanes.flux_read_panels import gated_unit_mass, stratum_gate_scale, unit_mass
    z = np.load(Path(capture) / f"score.{word}.iso.npz")
    pid = ST.WORDS[word]
    rows = [r for x in tokens[FS.PROMPTS[pid]]["words"] if x["word"] == word for r in x["rows"]]
    mass = np.asarray(FS.word_masses()[word], np.float64)
    mass_iso = np.asarray(unit_mass(z["tok_w"], rows, FS.HEADS))
    zi, mi = z["zero_img"].sum(0).ravel().astype(np.float64), z["mask_img"].sum(0).ravel().astype(np.float64)
    zo, mo = z["zero_oth"].sum(0).ravel().astype(np.float64), z["mask_oth"].sum(0).ravel().astype(np.float64)
    zt, mt = zi + zo, mi + mo
    scales = stratum_gate_scale(zi, mass_iso, FS.HEADS, FS.JOINT)
    gmass = np.asarray(gated_unit_mass(mass, FS.HEADS, FS.JOINT, scales))
    return dict(mass=mass, mass_iso=mass_iso, gmass=gmass, scales=scales,
                pw_both=zt / zt.sum() + mt / mt.sum(),
                pw_img=zi / zt.sum() + mi / mt.sum(),          # image-receiver part, in PW_both units
                pw_txt=zo / zt.sum() + mo / mt.sum(),          # text-receiver part, in PW_both units
                pw_img_only=zi / zi.sum() + mi / mi.sum())


def closest_single(mass, pw, single, target, k=N):
    """Inside ``single``: the band's minimum-PW set if reachable, else the k units with largest mass."""
    from mdb_lanes import predicted_write as PW
    excl = [i for i in range(len(mass)) if i not in set(single)]
    try:
        s, rep = PW.matched_min(mass, pw, k, target, 0.10, exclude=excl)
        return s, dict(rep, reachable=True)
    except ValueError as e:
        top = sorted(sorted(single, key=lambda i: (-mass[i], i))[:k])
        return top, dict(reachable=False, reason=str(e), max_reachable_mass=float(mass[top].sum()),
                         target=float(target), ratio=float(mass[top].sum() / target))


def select(capture):
    from mdb_lanes import predicted_write as PW
    from mdb_lanes.attention_panels import draw_seed
    from mdb_lanes.attention_ranking import matched_control_sets
    tokens = json.loads((Path(capture) / "tokens.json").read_text())
    old = json.loads((HERE.parent / "plans" / "sets.json").read_text())
    U = PL.universe()
    strata = np.array([0 if b < FS.JOINT else 1 for b, _ in U])
    blocks = np.array([b for b, _ in U])
    single = [i for i in range(len(U)) if strata[i] == 1]
    sets, report = {}, {}
    for w in ST.WORDS:
        sc = scores(capture, w, tokens)
        o = {k: PL.flat(v) for k, v in old[w].items()}
        pw60, own60 = o["pw60"], o["own60"]
        gmass60 = PW.top_units(sc["gmass"], N)
        mml_s, mml_s_rep = closest_single(sc["mass"], sc["pw_both"], single, float(sc["mass"][own60].sum()))
        own_single = [i for i in own60 if strata[i] == 1]
        mml_so, mml_so_rep = closest_single(sc["mass"], sc["pw_both"], single, float(sc["mass"][own_single].sum()))
        gml, gml_rep = PW.matched_min(sc["gmass"], sc["pw_txt"], N, float(sc["gmass"][pw60].sum()), 0.10)
        named_old = {k: v for k, v in o.items() if not k.startswith("f3")}
        named = set().union(*[set(v) for k, v in named_old.items() if not k.startswith("rand")],
                            gmass60, mml_s, mml_so, gml)
        old_rand_valid = {k: not (set(o[k]) & named) for k in ("rand60a", "rand60b")}
        draws, draw_rep = {}, {}
        for name in ("rand60c", "rand60d"):
            excl = sorted(named | set(o["rand60a"]) | set(o["rand60b"]) | set(draws.get("rand60c", ())))
            d = matched_control_sets(gmass60, len(U), strata=strata, blocks=blocks, disjoint=True,
                                     seed=draw_seed(SEED2, w, name, "cut"), n_sets=1, exclude=excl)[0]
            draws[name] = sorted(int(i) for i in d["units"])
            draw_rep[name] = {k: v for k, v in d["report"].items() if k != "units"}
        new = {"gmass60": gmass60, "massmatch-lowpw-single": mml_s,
               "massmatch-lowpw-single-own": mml_so, "gmassmatch-lowtextpw": gml, **draws}
        sets[w] = {**new, "pw60": pw60, "own60": own60}

        def tot(key, s):
            return float(sc[key][list(s)].sum())
        allsets = {**{k: v for k, v in named_old.items()}, **new}
        report[w] = {
            "gate_scales_joint_single": list(sc["scales"]), "joint_over_single": sc["scales"][0] / sc["scales"][1],
            "spearman_600": {"gmass_vs_pw_img_only": PW.spearman(sc["gmass"], sc["pw_img_only"]),
                             "mass_vs_pw_img_only": PW.spearman(sc["mass"], sc["pw_img_only"]),
                             "gmass_vs_pw_both": PW.spearman(sc["gmass"], sc["pw_both"]),
                             "gmass_vs_pw_txt": PW.spearman(sc["gmass"], sc["pw_txt"]),
                             "mass_vs_pw_both": PW.spearman(sc["mass"], sc["pw_both"])},
            "overlap": {"gmass60|pw60": PW.overlap(gmass60, pw60), "gmass60|own60": PW.overlap(gmass60, own60),
                        "gml|pw60": PW.overlap(gml, pw60), "gml|gmass60": PW.overlap(gml, gmass60),
                        "mml_single|own60": PW.overlap(mml_s, own60), "mml_single|pw60": PW.overlap(mml_s, pw60),
                        "mml_single_own|own60": PW.overlap(mml_so, own60), "mml_single_own|pw60": PW.overlap(mml_so, pw60)},
            "massmatch_lowpw_single": mml_s_rep, "massmatch_lowpw_single_own": mml_so_rep,
            "own60_single_units": len(own_single), "gmassmatch_lowtextpw": gml_rep,
            "old_random_draws_disjoint_from_new_named": old_rand_valid, "random_draws": draw_rep,
            "totals": {k: {m: tot(m, s) for m in ("mass", "gmass", "pw_both", "pw_img", "pw_txt")} |
                       {"joint_units": int(sum(strata[list(s)] == 0)), "distinct_blocks": len({U[i][0] for i in s})}
                       for k, s in allsets.items()},
        }
    return sets, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sets, report = select(a.capture)
    a.out.mkdir(parents=True, exist_ok=True)
    ST.save(a.out / "sets2.json", {w: {k: PL.pairs(v) for k, v in s.items()} for w, s in sets.items()})
    ST.save(a.out / "selection2-report.json", report)
    for w, r in report.items():
        t = r["totals"]
        print(w, "j/s", round(r["joint_over_single"], 4), "rho", {k: round(v, 3) for k, v in r["spearman_600"].items()},
              "ov", r["overlap"], "mmls", {k: r["massmatch_lowpw_single"].get(k) for k in ("reachable", "ratio")},
              "oldrand", r["old_random_draws_disjoint_from_new_named"])
        for k in ("pw60", "own60", "massmatch-lowpw", "gmass60", "massmatch-lowpw-single", "massmatch-lowpw-single-own", "gmassmatch-lowtextpw", "rand60c", "rand60d"):
            print("   ", k, {m: round(v, 3) for m, v in t[k].items()})

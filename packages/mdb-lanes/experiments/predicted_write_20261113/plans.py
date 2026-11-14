"""Head sets from the capture (rules fixed in PREREG.md) and the sealed read plans ``main`` and ``f3``.

Unit order is the read panels' universe: flat index ``block * 24 + head``.

PW (predicted write) per unit, from the capture job's iso-conditioning score run (seed 4242):
``PW_zero`` / ``PW_mask`` = sum over steps and over the receivers the arms cut (``image`` and
``others:<w>``) of the norm of the gated predicted write change; ``PW_both`` = PW_zero share + PW_mask
share (each normalised to sum 1 over the 600 units). Mass = the lane's existing per-unit write-mass
share (``flux_read_panel_stage.word_masses``: full-a capture, native conditioning, seed 4242).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import stage as ST

FS = ST.FS
SELECTION_SEED = 20261113
N = 60
F3_SETS, F3_SIZE, F3_POOL = 16, 10, 100
MAIN_CUT = ("pw60", "own60", "massmatch-lowpw", "pwmatch-lowmass", "rand60a", "rand60b")
MAIN_KEEP = ("pw60", "own60", "rand60a")


def universe():
    return tuple((b, h) for b in range(FS.BLOCKS) for h in range(FS.HEADS))


def pairs(idx):
    U = universe()
    return [list(U[i]) for i in sorted(idx)]


def flat(pairs_):
    return sorted(b * FS.HEADS + h for b, h in pairs_)


def unit_scores(capture, word):
    """Per-unit PW arrays (600,) from ``score.<word>.<cond>.npz`` of the capture job."""
    out = {}
    for cond in ("iso", "native"):
        z = np.load(Path(capture) / f"score.{word}.{cond}.npz")
        zero = (z["zero_img"] + z["zero_oth"]).sum(0).ravel().astype(np.float64)
        mask = (z["mask_img"] + z["mask_oth"]).sum(0).ravel().astype(np.float64)
        out[cond] = {"pw_zero": zero, "pw_mask": mask, "pw_both": zero / zero.sum() + mask / mask.sum(),
                     "pw_zero_img": z["zero_img"].sum(0).ravel().astype(np.float64),
                     "pw_mask_img": z["mask_img"].sum(0).ravel().astype(np.float64),
                     "tok_w": z["tok_w"].astype(np.float64)}
    return out


def select(capture):
    """All sets per word plus the selection report (rank correlations, overlaps, totals)."""
    from mdb_lanes import predicted_write as PW
    from mdb_lanes.attention_panels import draw_seed
    from mdb_lanes.attention_ranking import matched_control_sets
    from mdb_lanes.flux_read_panels import unit_mass
    named_sets = json.loads(FS.HEAD_SETS.read_text())["sets"]
    masses = FS.word_masses()
    tokens = json.loads((Path(capture) / "tokens.json").read_text())
    U = universe()
    strata = np.array([0 if b < FS.JOINT else 1 for b, _ in U])
    blocks = np.array([b for b, _ in U])
    sets, report = {}, {}
    for w, pid in ST.WORDS.items():
        sc = unit_scores(capture, w)
        pwb = sc["iso"]["pw_both"]
        mass = np.asarray(masses[w], np.float64)
        rows = [r for x in tokens[FS.PROMPTS[pid]]["words"] if x["word"] == w for r in x["rows"]]
        mass_iso = np.asarray(unit_mass(sc["iso"]["tok_w"], rows, FS.HEADS))
        mass_native_rerun = np.asarray(unit_mass(sc["native"]["tok_w"], rows, FS.HEADS))
        own60 = flat(named_sets[f"{pid}:{w}"]["own60"])
        pw60 = PW.top_units(pwb, N)
        mml, mml_rep = PW.matched_min(mass, pwb, N, float(mass[own60].sum()), 0.10)
        pml, pml_rep = PW.matched_min(pwb, mass, N, float(pwb[pw60].sum()), 0.10)
        named = set(pw60) | set(own60) | set(mml) | set(pml)
        draws, draw_reports = {}, {}
        for name in ("rand60a", "rand60b"):
            excl = sorted(named | set(draws.get("rand60a", ())))
            d = matched_control_sets(pw60, len(U), strata=strata, blocks=blocks, disjoint=True,
                                     seed=draw_seed(SELECTION_SEED, w, name, "cut"), n_sets=1, exclude=excl)[0]
            draws[name] = sorted(int(i) for i in d["units"])
            draw_reports[name] = {k: v for k, v in d["report"].items() if k != "units"}
        rng = np.random.default_rng(draw_seed(SELECTION_SEED, w, "f3", "cut"))
        pool = sorted(set(PW.top_units(pwb, F3_POOL)) | set(PW.top_units(mass, F3_POOL)))
        f3 = {f"f3s{i:02d}": sorted(int(x) for x in rng.choice(pool, size=F3_SIZE, replace=False))
              for i in range(F3_SETS)}
        word_sets = {"pw60": pw60, "own60": own60, "massmatch-lowpw": mml, "pwmatch-lowmass": pml, **draws, **f3}
        sets[w] = word_sets
        main = {k: word_sets[k] for k in MAIN_CUT}
        mass_top60 = PW.top_units(mass, N)

        def tot(arr, s):
            return float(np.asarray(arr)[s].sum())
        report[w] = {
            "spearman_600": {"pw_both_vs_mass": PW.spearman(pwb, mass),
                             "pw_zero_vs_mass": PW.spearman(sc["iso"]["pw_zero"], mass),
                             "pw_mask_vs_mass": PW.spearman(sc["iso"]["pw_mask"], mass),
                             "pw_zero_vs_pw_mask": PW.spearman(sc["iso"]["pw_zero"], sc["iso"]["pw_mask"]),
                             "pw_both_vs_mass_iso_capture": PW.spearman(pwb, mass_iso),
                             "mass_vs_mass_native_rerun": PW.spearman(mass, mass_native_rerun),
                             "mass_vs_mass_iso_capture": PW.spearman(mass, mass_iso)},
            "mass_native_rerun_max_abs_diff_vs_saved": float(np.abs(mass - mass_native_rerun).max()),
            "top60_overlap": {"pw60_vs_own60": PW.overlap(pw60, own60),
                              "pw60_vs_mass_top60": PW.overlap(pw60, mass_top60),
                              "own60_vs_mass_top60": PW.overlap(own60, mass_top60),
                              "pw_zero_top60_vs_mass_top60": PW.overlap(PW.top_units(sc["iso"]["pw_zero"], N), mass_top60),
                              "pw_mask_top60_vs_mass_top60": PW.overlap(PW.top_units(sc["iso"]["pw_mask"], N), mass_top60)},
            "pairwise_overlap": {f"{a}|{b}": PW.overlap(main[a], main[b]) for i, a in enumerate(main)
                                 for b in list(main)[i + 1:]},
            "totals": {k: {"mass": tot(mass, s), "pw_both": tot(pwb, s), "pw_zero_share": tot(sc["iso"]["pw_zero"] / sc["iso"]["pw_zero"].sum(), s),
                           "pw_mask_share": tot(sc["iso"]["pw_mask"] / sc["iso"]["pw_mask"].sum(), s),
                           "joint_units": int(sum(1 for i in s if U[i][0] < FS.JOINT)),
                           "distinct_blocks": len({U[i][0] for i in s})} for k, s in main.items()},
            "massmatch_lowpw": mml_rep, "pwmatch_lowmass": pml_rep, "random_draws": draw_reports,
            "f3_pool_size": len(pool),
            "f3": {k: {"pw_zero": tot(sc["iso"]["pw_zero"], s), "pw_mask": tot(sc["iso"]["pw_mask"], s),
                       "pw_both": tot(pwb, s), "mass": tot(mass, s)} for k, s in f3.items()},
            "f3_spearman_pw_both_vs_mass_over_sets": PW.spearman([tot(pwb, s) for s in f3.values()],
                                                                 [tot(mass, s) for s in f3.values()]),
        }
    return sets, report


def word_panel(word, sets, names_cut, names_keep, *, floors=True, references=("condition", "all_cut", "deletion")):
    from mdb_lanes.attention_panels import ReadPanelSpec, read_panel
    waive = {"random_cut": "random sets rand60a/rand60b are named sets drawn by plans.select (stratum- and "
                           "block-matched to pw60, excluding every named set of the word and each other)",
             "other_item": "not in this design: the dissociation arms carry the comparison"}
    if names_keep:
        waive["random_keep"] = "keep-rand60a is the random keep control (complement of rand60a)"
    if not floors:
        waive["floors"] = "F3 small sets: floors of the main plan's arms give the arithmetic scale"
    return read_panel(ReadPanelSpec(
        items=(word,), universe=universe(), sets={word: {k: pairs(v) for k, v in sets.items()}},
        contexts=(str(ST.SEED),), cut_sets=tuple(names_cut), keep_sets=tuple(names_keep), random=None,
        other_items={}, waive=waive, floors=floors, references=references,
        port={"precision": "native", "zero_method": "delta"}))


def build(sets):
    from mdb_lanes.attention_panels import Panel
    from mdb_lanes.flux_read_panels import merge_panels
    note = "experiment predicted_write_20261113; sets from plans.select (PREREG.md); seed 4242 only"
    def drop_native(panel):
        # read_panel adds an item-less "native" arm; flux_read_panel_worker indexes items by arm.item and
        # resolves the "native" reference of iso/del arms to the native root itself, so the arm is dropped.
        d = json.loads(panel.canonical())
        d["arms"] = [a for a in d["arms"] if a["arm"] != "native"]
        return Panel.from_json(d)

    main = merge_panels([drop_native(word_panel(w, sets[w], MAIN_CUT, MAIN_KEEP)) for w in ST.WORDS], notes=[note])
    f3names = tuple(f"f3s{i:02d}" for i in range(F3_SETS))
    f3 = merge_panels([drop_native(word_panel(w, sets[w], f3names, (), floors=False)) for w in ST.WORDS],
                      notes=[note])
    return {"main": main, "f3": f3}


def write(capture, root):
    from mdb_lanes.flux_read_panels import leases_within, work_items
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    sets, report = select(capture)
    ST.save(root / "sets.json", {w: {k: pairs(v) for k, v in s.items()} for w, s in sets.items()})
    ST.save(root / "selection-report.json", report)
    items = ST.items_map()
    summary = {}
    for name, panel in build(sets).items():
        cap, leases = leases_within(panel, items, limit=FS.WORK_LIMIT)
        (root / f"{name}.plan.json").write_text(panel.canonical() + "\n")
        ST.save(root / f"{name}.seal.json", panel.seal())
        ST.save(root / f"{name}.leases.json", {"plan_sha256": panel.sha256(), "cap": cap, "leases": leases})
        summary[name] = {"plan_sha256": panel.sha256(), "arms": len(panel.arms), "branches": panel.branch_count,
                         "lease_cap": cap, "leases": [(len(l["branches"]), work_items(panel, l, items)) for l in leases]}
    ST.save(root / "plans-summary.json", summary)
    return summary

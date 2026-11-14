"""Addendum-2 panel arms (plan gm): gone, floors, latent projections, content, H1-H3, sheets (no model).

  PYTHONPATH=<worktree>/packages/mdb-lanes/src:<worktree>/packages/mdb/src \
    python analyze_gm.py JOB_DIR_LEASE0 JOB_DIR_LEASE1      (collected ``flux-read-panel`` dirs)
  -> results/gm-arms.json, results/forecasts-addendum2.json, sheets in artifacts/.../sheets/addendum2/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import content as C  # noqa: E402
from analyze import gone_of, sheet  # noqa: E402

EXP = HERE.parent
ART = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113")
SAVED_MAIN = {"red": "job-5eeb743a56cd", "black": "job-5eeb743a56cd", "ball": "job-5eeb743a56cd",
              "snow": "job-3be01d6f10e4", "night": "job-3be01d6f10e4", "rain": "job-3be01d6f10e4"}
WORDS = ("red", "black", "ball", "snow", "night", "rain")
H_WORDS = ("red", "black", "ball")
NEW = ("gmass60", "massmatch-lowpw-single", "massmatch-lowpw-single-own", "gmassmatch-lowtextpw", "rand60c", "rand60d")
PRE = {"zero": "z", "renorm": "m"}
PROMPT = {"red": "cat", "black": "cat", "ball": "cat", "snow": "cat", "night": "car", "rain": "car"}


def main(dirs):
    from mdb_lanes.attention_panels import Panel, arm_effects
    from mdb_lanes.flux_read_panels import merge_results, safe_name
    dirs = [Path(d) for d in dirs]
    panel = Panel.from_json(json.loads((HERE / "plans2/gm.plan.json").read_text()))
    rows = [r for d in dirs for p in sorted(d.glob("metrics.gm-*.json")) for r in json.loads(p.read_text())]
    recs = [json.loads(p.read_text()) for d in dirs for p in sorted(d.glob("results.gm-*.px_mad.json"))]
    floors = {e["arm"]: e for e in arm_effects(panel, merge_results(panel, recs))}
    saved = json.loads((EXP / "results/arms.json").read_text())["effects"]["main"]
    saved_rows = {}
    for w, job in SAVED_MAIN.items():
        for p in (ART / "results" / job / "flux-read-panel").glob("metrics.main-*.json"):
            saved_rows.update({r["arm"]: r for r in json.loads(p.read_text())})
    by = {r["arm"]: r for r in rows}
    lat_dir = {r["arm"]: d for d in dirs for p in d.glob("metrics.gm-*.json") for r in json.loads(p.read_text())}

    def latent(arm):
        d = lat_dir[arm]
        tag = next(p.name for p in (d / "latents").iterdir() if p.name.startswith("gm-"))
        return np.load(d / "latents" / tag / "4242" / f"{safe_name(arm)}.npy")

    def root_latent(name):
        for d in dirs:
            p = d / "latents" / "roots" / "4242" / f"{safe_name(name)}.npy"
            if p.exists():
                return np.load(p)
        raise FileNotFoundError(name)

    table, checks = {}, {"pw60_own60_rerun_rgb_equal_saved": {}, "references_equal_saved": {}}
    content_roots = {}
    for w in WORDS:
        L_nat, L_del, L_iso = root_latent(f"native:{PROMPT[w]}"), root_latent(f"del:{w}"), root_latent(f"iso:{w}")
        u, u2 = (L_nat - L_del).ravel(), (L_iso - L_del).ravel()
        imgdir = next(d for d in dirs if f"iso:{w}" in {r["arm"] for r in json.loads(next(d.glob("metrics.gm-*.json")).read_text())})
        tag = next(p.name for p in (imgdir / "images").iterdir() if p.name.startswith("gm-"))
        img = lambda arm: imgdir / "images" / tag / "4242" / f"{safe_name(arm)}.png"   # noqa: E731
        meas = {}
        if w in C.MEASURES:
            f = C.MEASURES[w]
            cr = {k: f(img(f"{k}:{w}")) for k in ("iso", "del")}
            content_roots[w] = cr
            meas = {"f": f, "roots": cr}
        for arm in (f"iso:{w}", f"del:{w}", f"z:none:{w}", f"m:none:{w}"):
            checks["references_equal_saved"][arm] = by[arm]["image_rgb_sha256"] == saved_rows[arm]["image_rgb_sha256"]
        allcut = {}
        for t, p in PRE.items():
            dl = (L_iso - latent(f"{p}:none:{w}")).ravel()
            allcut[t] = (float(dl @ u / (u @ u)), float(dl @ u2 / (u2 @ u2)))
        for s in (*NEW, "pw60", "own60"):
            for t, p in PRE.items():
                arm = f"{p}:{s}:{w}"
                g, span, nd = gone_of(rows, w, arm, t)
                fl = floors.get(arm, {}).get("floor")
                dl = (L_iso - latent(arm)).ravel()
                pu, pu2 = float(dl @ u / (u @ u)), float(dl @ u2 / (u2 @ u2))
                rec = {"word": w, "set": s, "cut_type": t, "gone_px": g, "span": span,
                       "floor_gone_bound": None if fl is None else fl / (nd * abs(span)),
                       "px_vs_iso": by[arm]["vs_iso"]["px_mad"],
                       "latent": {"proj_u_native_minus_del": pu, "proj_u_iso_minus_del": pu2,
                                  "norm_over_u": float(np.linalg.norm(dl) / np.linalg.norm(u)),
                                  "frac_of_allcut_u": pu / allcut[t][0], "frac_of_allcut_u_iso": pu2 / allcut[t][1]}}
                if meas:
                    cv = meas["f"](img(arm))
                    rec["content"] = {"value": cv, "removed_vs_iso": C.removed(cv, meas["roots"]["iso"], meas["roots"]["del"])}
                if s in ("pw60", "own60"):
                    rec["saved_gone_px"] = saved[arm]["gone"]
                    eq = by[arm]["image_rgb_sha256"] == saved_rows[arm]["image_rgb_sha256"]
                    checks["pw60_own60_rerun_rgb_equal_saved"][arm] = eq
                table[arm] = rec
        # contact sheets: native / del / iso / all-cut / new arms (zero and renorm)
        root_png = lambda n: next(d / "images" / "roots" / "4242" / f"{safe_name(n)}.png" for d in dirs  # noqa: E731
                                  if (d / "images" / "roots" / "4242" / f"{safe_name(n)}.png").exists())
        for t, p in PRE.items():
            names = [f"{p}:none:{w}"] + [f"{p}:{s}:{w}" for s in ("pw60", "own60", *NEW)]
            sheet([root_png(f"native:{PROMPT[w]}"), root_png(f"del:{w}"), root_png(f"iso:{w}")] + [img(n) for n in names],
                  ["native", "del", "iso", "all-cut", "pw60", "own60", "gmass60", "mm-lowpw-single(closest)",
                   "mm-lowpw-single-own", "gmm-lowtextpw", "rand60c", "rand60d"],
                  ART / "sheets" / "addendum2" / f"gm-{w}-{'zero' if t == 'zero' else 'renorm'}.png",
                  f"{w} ({t}): addendum 2 sets")
    # forecasts
    def G(w, s, t, saved_ok=True):
        a = f"{PRE[t]}:{s}:{w}"
        return table[a]["saved_gone_px"] if (saved_ok and "saved_gone_px" in table[a]) else table[a]["gone_px"]

    def rule(name, cmp, words=H_WORDS):
        cells = []
        for w in words:
            for t in PRE:
                cells.append(cmp(w, t))
        held = all(c["holds"] for c in cells)
        return {"rule": name, "cells": cells, "held_cells": sum(c["holds"] for c in cells), "of": len(cells),
                "status": "held" if held else "failed"}

    fc = {
        "H1": rule("gone(gmass60) >= 0.8 gone(pw60)", lambda w, t: {
            "word": w, "type": t, "gmass60": G(w, "gmass60", t), "pw60": G(w, "pw60", t),
            "ratio": G(w, "gmass60", t) / G(w, "pw60", t), "holds": G(w, "gmass60", t) >= 0.8 * G(w, "pw60", t)}),
        "H2": rule("gone(massmatch-lowpw-single) < gone(own60)/3", lambda w, t: {
            "word": w, "type": t, "set": G(w, "massmatch-lowpw-single", t), "own60": G(w, "own60", t),
            "ratio": G(w, "massmatch-lowpw-single", t) / G(w, "own60", t),
            "holds": G(w, "massmatch-lowpw-single", t) < G(w, "own60", t) / 3}),
        "H2b": rule("gone(massmatch-lowpw-single-own) < gone(own60)/3", lambda w, t: {
            "word": w, "type": t, "set": G(w, "massmatch-lowpw-single-own", t), "own60": G(w, "own60", t),
            "ratio": G(w, "massmatch-lowpw-single-own", t) / G(w, "own60", t),
            "holds": G(w, "massmatch-lowpw-single-own", t) < G(w, "own60", t) / 3}),
        "H3": rule("gone(gmassmatch-lowtextpw) < 2/3 gone(pw60)", lambda w, t: {
            "word": w, "type": t, "set": G(w, "gmassmatch-lowtextpw", t), "pw60": G(w, "pw60", t),
            "ratio": G(w, "gmassmatch-lowtextpw", t) / G(w, "pw60", t),
            "holds": G(w, "gmassmatch-lowtextpw", t) < (2 / 3) * G(w, "pw60", t)}),
    }
    # secondary: all words, pixel; latent fraction of all-cut; content for snow/rain
    sec = {}
    for w in WORDS:
        for t in PRE:
            r = lambda s: table[f"{PRE[t]}:{s}:{w}"]   # noqa: E731
            row = {s: {"gone_px": r(s)["gone_px"], "lat_u": r(s)["latent"]["frac_of_allcut_u"],
                       "lat_u_iso": r(s)["latent"]["frac_of_allcut_u_iso"],
                       "content": r(s).get("content", {}).get("removed_vs_iso"), "floor": r(s)["floor_gone_bound"]}
                   for s in ("pw60", "own60", *NEW)}
            sec[f"{w}/{t}"] = row
    fc["secondary_by_word"] = sec
    out = {"arms": table, "checks": checks, "content_roots": content_roots}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/gm-arms.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    (HERE / "results/forecasts-addendum2.json").write_text(json.dumps(fc, indent=1, sort_keys=True) + "\n")
    for h in ("H1", "H2", "H2b", "H3"):
        print(h, fc[h]["status"], fc[h]["held_cells"], "/", fc[h]["of"],
              [(c["word"], c["type"][0], round(c["ratio"], 2)) for c in fc[h]["cells"]])
    print("checks", {k: (sum(v.values()), len(v)) for k, v in checks.items()})
    for k, row in sec.items():
        print(k, {s: (round(v["gone_px"], 2), round(v["lat_u_iso"], 2), None if v["content"] is None else round(v["content"], 2),
                      None if v["floor"] is None else round(v["floor"], 3)) for s, v in row.items()})


if __name__ == "__main__":
    main(sys.argv[1:])

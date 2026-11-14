"""Offline analysis of the predicted-write experiment (no model; numpy, PIL).

  $PY analyze.py closure CAPTURE_DIR                     -> results/closure.json
  $PY analyze.py split CAPTURE_DIR                       -> results/split.json (Experiment B, G1/G2)
  $PY analyze.py vswap CAPTURE_DIR --panels JOB_DIR...   -> results/vswap.json (G3)
  $PY analyze.py arms --plans plans --panels JOB_DIR...  -> results/arms.json, results/forecasts-A.json
  $PY analyze.py qwen QWEN_JSON                          -> results/qwen.json
  $PY analyze.py sheets --capture CAPTURE_DIR --panels JOB_DIR... --out DIR

Effect metric (PREREG.md): ``gone_t = (d(arm) - d(iso)) / (d(t:none) - d(iso))`` with
``d = 1 - MAD(arm, del) / MAD(native, del)`` and the all-cut ``t:none`` of the arm's own cut type.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
G1, G2 = ("red", "ball", "black"), ("snow", "rain", "night")
WORDS = ("red", "black", "ball", "snow", "night", "rain")
JOINT = 5


def save(name, value):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return RESULTS / name


def _find(root, pattern):
    return sorted(Path(root).rglob(pattern))


# ===================================================================================== closure

def closure(capture):
    rows = []
    for p in _find(capture, "closure.*.json"):
        rows += [dict(r, word=p.name.split(".")[1]) for r in json.loads(p.read_text())]
    out = {"cells": len(rows), "by": {}}
    for src in ("own", "text"):
        for cut in ("zero", "renorm"):
            sel = [r for r in rows if r["sources"] == src and r["cut"] == cut]
            if not sel:
                continue
            img = [r["img"] for r in sel]
            out["by"][f"{src}/{cut}"] = {
                "n": len(sel),
                "head_rel_err_max": max(x["head_rel_err"] for x in img),
                "head_within_bound_min": min(x["head_within_bound_frac"] for x in img),
                "head_max_err_over_bound": max(x["head_max_err_over_bound"] for x in img),
                "other_heads_max_abs_change": max(x["other_heads_max_abs_change"] for x in img),
                "write_rel_err_model_bf16_max": max(x["write_rel_err_model_bf16"] for x in img),
                "write_rel_err_model_bf16_median": float(np.median([x["write_rel_err_model_bf16"] for x in img])),
                "write_rel_err_fp32_projection_max": max(x["write_rel_err_fp32_projection_of_measured"] for x in img),
                "sham_over_pred_median": float(np.median([x["sham_write_norm_over_pred"] for x in img])),
                "sham_over_pred_max": max(x["sham_write_norm_over_pred"] for x in img),
                "txt_head_rel_err_max": max(r["txt"]["head_rel_err"] for r in sel),
                "txt_within_bound_min": min(r["txt"]["head_within_bound_frac"] for r in sel),
                "txt_write_rel_err_model_bf16_max": max(r["txt"]["write_rel_err_model_bf16"] for r in sel)}
    out["rows"] = rows
    gate = {}
    for p in _find(capture, "word.*.json"):
        w = json.loads(p.read_text())
        if w.get("single_block_gate_check_rel"):
            gate[w["word"]] = w["single_block_gate_check_rel"]
    out["single_block_gate_check_rel"] = gate
    return out


# ======================================================================================= split

def _agg(cells, keep):
    keys = None
    acc = {}
    for key, rec in cells.items():
        s, b = map(int, key.split(":"))
        if not keep(s, b):
            continue
        for name, r in rec.items():
            if not isinstance(r, dict):
                continue
            a = acc.setdefault(name, {})
            for k, v in r.items():
                a[k] = a.get(k, 0.0) + v
    if not acc:
        return None
    tt = acc["total"]["nn"]
    pay, txt = acc["payload"]["nn"], acc["text_payload"]["nn"]

    def proj(name):
        return acc[name]["ip"] / tt

    out = {"projection_share": {k: proj(k) for k in ("routing", "payload", "interaction")},
           "norm_ratio": {k: math.sqrt(acc[k]["nn"] / tt) for k in ("routing", "payload", "interaction")},
           "payload_groups_projection_share": {g: acc[f"payload_{g}"]["ip_payload"] / pay
                                               for g in ("word", "real", "pad", "img")},
           "payload_groups_norm_ratio": {g: math.sqrt(acc[f"payload_{g}"]["nn"] / pay)
                                         for g in ("word", "real", "pad", "img")},
           "text_payload_share_of_payload": acc["text_payload"]["ip"] / acc["payload"]["ip"]
           if acc["payload"]["ip"] else None,
           "text_common_projection_share": acc["text_common"]["ip_text_payload"] / txt,
           "text_specific_projection_share": acc["text_specific"]["ip_text_payload"] / txt,
           "text_common_norm_ratio": math.sqrt(acc["text_common"]["nn"] / txt),
           "remainder_alg_rel": math.sqrt(acc["remainder_alg"]["nn"] / tt),
           "remainder_native_rel": math.sqrt(acc["remainder_native"]["nn"] / tt),
           "native_diff_vs_fp32_total_norm_ratio": math.sqrt(acc["native_diff"]["nn"] / tt),
           "total_norm": math.sqrt(tt)}
    return out


def split(capture):
    words, inv = {}, {}
    for p in _find(capture, "split.*.json"):
        rec = json.loads(p.read_text())
        cells = rec["cells"]
        w = rec["word"]
        words[w] = {"all": _agg(cells, lambda s, b: True), "joint": _agg(cells, lambda s, b: b < JOINT),
                    "single": _agg(cells, lambda s, b: b >= JOINT),
                    **{f"step{k}": _agg(cells, lambda s, b, k=k: s == k) for k in range(4)},
                    "alignment": rec["alignment"], "deleted_observer_run_equals_root": rec["deleted_observer_run_equals_root"]}
        inv[w] = {"invariant_max_abs_diff": max(c["invariant_max_abs_diff"] for c in cells.values()),
                  "routing_max_abs": max(c["routing_max_abs"] for c in cells.values()),
                  "dP_rowsum_max_abs": max(c["dP_rowsum_max_abs"] for c in cells.values())}

    def gmean(group, scope, f):
        vals = [f(words[w][scope]) for w in group if w in words and words[w][scope]]
        return float(np.mean(vals)) if len(vals) == len(group) else None

    forecasts = {}
    for scope in ("all", "joint", "single"):
        wr = lambda r: r["payload_groups_projection_share"]["word"] + r["payload_groups_projection_share"]["real"]
        pad = lambda r: r["payload_groups_projection_share"]["pad"]
        com = lambda r: r["text_common_projection_share"]
        forecasts[scope] = {
            "word_plus_real_share": {"G1": gmean(G1, scope, wr), "G2": gmean(G2, scope, wr)},
            "pad_share": {"G1": gmean(G1, scope, pad), "G2": gmean(G2, scope, pad)},
            "common_share": {"G1": gmean(G1, scope, com), "G2": gmean(G2, scope, com)}}
        f = forecasts[scope]
        if None not in (f["pad_share"]["G1"], f["pad_share"]["G2"]):
            f["orderings"] = {"word_plus_real G1 > G2": f["word_plus_real_share"]["G1"] > f["word_plus_real_share"]["G2"],
                              "pad G1 < G2": f["pad_share"]["G1"] < f["pad_share"]["G2"],
                              "common G1 < G2": f["common_share"]["G1"] < f["common_share"]["G2"]}
    out = {"words": words, "invariant": inv, "G1_G2": forecasts}
    if "all" in forecasts and "orderings" in forecasts["all"]:
        o = forecasts["all"]["orderings"]
        out["G1"] = "held" if o["word_plus_real G1 > G2"] and o["common G1 < G2"] else "failed"
        out["G2"] = "held" if o["pad G1 < G2"] and o["common G1 < G2"] else "failed"
    return out


# ======================================================================================= panels

def panel_rows(dirs, tag_prefix):
    rows = []
    for d in dirs:
        for p in _find(d, f"metrics.{tag_prefix}-*.json"):
            rows += json.loads(p.read_text())
    return rows


def gone_of(rows, word, arm, cut_type):
    by = {r["arm"]: r for r in rows}
    nd = by[f"del:{word}"]["vs_native"]["px_mad"]
    d = lambda a: 1.0 - by[a]["vs_del"]["px_mad"] / nd
    none = {"zero": f"z:none:{word}", "renorm": f"m:none:{word}"}[cut_type]
    span = d(none) - d(f"iso:{word}")
    return (d(arm) - d(f"iso:{word}")) / span, span, nd


def _lt(a, b, f):
    return None if a is None or b is None else bool(a < f * b)


def _ge(a, b, f):
    return None if a is None or b is None else bool(a >= f * b)


def _status(cells):
    held = [c["holds"] for c in cells]
    if any(h is None for h in held):
        return {"cells": cells, "status": "not tested (cells missing)", "missing": sum(h is None for h in held)}
    return {"cells": cells, "held_cells": sum(held), "of": len(held), "status": "held" if all(held) else "failed"}


def arms(plans, dirs):
    from mdb_lanes.attention_panels import Panel, arm_effects
    from mdb_lanes.flux_read_panels import merge_results
    plans = Path(plans)
    sets_report = json.loads((plans / "selection-report.json").read_text())
    out, effects = {}, {}
    for plan in ("main", "f3"):
        rows = panel_rows(dirs, plan)
        if not rows:
            continue
        panel = Panel.from_json(json.loads((plans / f"{plan}.plan.json").read_text()))
        recs = [json.loads(p.read_text()) for d in dirs for p in _find(d, f"results.{plan}-*.px_mad.json")]
        merged = merge_results(panel, recs)
        floors = {e["arm"]: e for e in arm_effects(panel, merged)}
        table = {}
        for r in rows:
            a = r["arm"]
            if r["role"] != "test" and r["kind"] not in ("all_cut",):
                continue
            if a.endswith(("/sham", "/sham_recompute", "/cut_recompute")):
                continue
            w = r["item"]
            ct = r["cut_type"]
            g, span, nd = gone_of(rows, w, a, ct)
            fl = floors.get(a, {}).get("floor")
            table[a] = {"word": w, "cut_type": ct, "unit_set": r["unit_set"], "kind": r["kind"], "gone": g,
                        "span": span, "px_vs_iso": r["vs_iso"]["px_mad"], "px_vs_del": r["vs_del"]["px_mad"],
                        "floor_px": fl, "floor_gone_bound": None if fl is None else fl / (nd * abs(span))}
        effects[plan] = table
    out["effects"] = effects
    main = effects.get("main", {})
    pre = {"zero": "z", "renorm": "m"}

    def g(w, s, t):
        return main.get(f"{pre[t]}:{s}:{w}", {}).get("gone")

    fc = {}
    if main:
        f1 = [{"word": w, "type": t, "mml": g(w, "massmatch-lowpw", t), "own60": g(w, "own60", t),
               "holds": _lt(g(w, "massmatch-lowpw", t), g(w, "own60", t), 1 / 3)} for w in G1 for t in pre]
        f2 = [{"word": w, "type": t, "pml": g(w, "pwmatch-lowmass", t), "pw60": g(w, "pw60", t),
               "holds": _ge(g(w, "pwmatch-lowmass", t), g(w, "pw60", t), 2 / 3)} for w in WORDS for t in pre]
        f4 = [{"word": w, "type": t, "draw": r, "rand": g(w, r, t), "pw60": g(w, "pw60", t),
               "holds": _lt(g(w, r, t), g(w, "pw60", t), 1 / 3)} for w in WORDS for t in pre for r in ("rand60a", "rand60b")]
        fc["F1"] = _status(f1)
        fc["F2"] = _status(f2)
        fc["F4"] = _status(f4)
        fc["keep"] = {f"{w}/{t}": {k: g(w, k, t) for k in ("keep-pw60", "keep-own60", "keep-rand60a")}
                      for w in WORDS for t in pre}
    f3 = effects.get("f3", {})
    if f3:
        from mdb_lanes.analysis_metrics import correlation_report
        cells = []
        for w in WORDS:
            rep = sets_report[w]["f3"]
            for t in pre:
                names = sorted(rep)
                y = [f3[f"{pre[t]}:{n}:{w}"]["gone"] for n in names]
                ypx = [f3[f"{pre[t]}:{n}:{w}"]["px_vs_iso"] for n in names]
                xpw = [rep[n]["pw_zero" if t == "zero" else "pw_mask"] for n in names]
                xb = [rep[n]["pw_both"] for n in names]
                xm = [rep[n]["mass"] for n in names]
                cells.append({"word": w, "type": t,
                              "pw_type": correlation_report(xpw, y), "pw_both": correlation_report(xb, y),
                              "mass": correlation_report(xm, y),
                              "pw_type_px": correlation_report(xpw, ypx), "mass_px": correlation_report(xm, ypx),
                              "pw_vs_mass_spearman_over_sets": correlation_report(xpw, xm)["spearman"]})
        diff = [c["pw_type"]["spearman"] - c["mass"]["spearman"] for c in cells]
        wins = sum(d > 0 for d in diff)
        fc["F3"] = {"cells": [{k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("pearson", "spearman")})
                               for k, v in c.items()} for c in cells],
                    "mean_spearman_pw_type": float(np.mean([c["pw_type"]["spearman"] for c in cells])),
                    "mean_spearman_mass": float(np.mean([c["mass"]["spearman"] for c in cells])),
                    "mean_diff": float(np.mean(diff)), "pw_wins": wins, "of": len(cells),
                    "status": "held" if np.mean(diff) > 0 and wins >= 8 else "failed"}
    out["forecasts"] = fc
    return out


# ============================================================================= image-only PW

def imgpw(capture, plans, arms_json=None):
    """Secondary report (PREREG seal note): image-receiver-only PW beside PW_both and mass.

    Per word: Spearman over the 600 units, top-60 overlaps, the text-receiver share of PW, and,
    when ``arms_json`` (results/arms.json) is given, the F3 correlation of per-set image-only PW
    with gone next to PW (cut type) and mass.
    """
    import sys
    sys.path.insert(0, str(HERE))
    import plans as PL
    FS = PL.FS
    from mdb_lanes import predicted_write as PW
    from mdb_lanes.analysis_metrics import correlation_report
    raw = json.loads((Path(plans) / "sets.json").read_text())
    sets = {w: {k: (PL.flat(v) if v and isinstance(v[0], list) else v) for k, v in d.items()} for w, d in raw.items()}
    masses = FS.word_masses()
    effects = json.loads(Path(arms_json).read_text())["effects"].get("f3", {}) if arms_json else {}
    out = {}
    for w in WORDS:
        sc = PL.unit_scores(capture, w)["iso"]
        mass = np.asarray(masses[w], np.float64)
        img = {"zero": sc["pw_zero_img"], "renorm": sc["pw_mask_img"]}
        full = {"zero": sc["pw_zero"], "renorm": sc["pw_mask"]}
        rec = {"text_receiver_share": {t: float(1 - img[t].sum() / full[t].sum()) for t in img},
               "spearman_600": {f"img_{t}_vs_mass": PW.spearman(img[t], mass) for t in img},
               "spearman_600_img_vs_full": {t: PW.spearman(img[t], full[t]) for t in img},
               "top60_overlap_img_zero_vs_mass": PW.overlap(PW.top_units(img["zero"], 60), PW.top_units(mass, 60)),
               "top60_overlap_img_zero_vs_pw60": PW.overlap(PW.top_units(img["zero"], 60), sets[w]["pw60"])}
        if effects:
            cells = {}
            for t, pre in (("zero", "z"), ("renorm", "m")):
                names = sorted(k for k in sets[w] if k.startswith("f3s"))
                y = [effects[f"{pre}:{n}:{w}"]["gone"] for n in names]
                cells[t] = {"img_pw": correlation_report([float(img[t][sets[w][n]].sum()) for n in names], y)["spearman"],
                            "pw": correlation_report([float(full[t][sets[w][n]].sum()) for n in names], y)["spearman"],
                            "mass": correlation_report([float(mass[sets[w][n]].sum()) for n in names], y)["spearman"]}
            rec["f3_spearman"] = cells
        out[w] = rec
    if effects:
        diffs = [out[w]["f3_spearman"][t]["img_pw"] - out[w]["f3_spearman"][t]["mass"] for w in WORDS
                 for t in ("zero", "renorm")]
        out["f3_img_summary"] = {"mean_spearman_img_pw": float(np.mean([out[w]["f3_spearman"][t]["img_pw"] for w in WORDS
                                                                        for t in ("zero", "renorm")])),
                                 "mean_diff_vs_mass": float(np.mean(diffs)), "img_wins": int(sum(d > 0 for d in diffs)),
                                 "of": len(diffs)}
    return out


# ======================================================================================== vswap

def vswap(capture, dirs):
    out = {}
    panel = panel_rows(dirs, "main") if dirs else []
    by = {r["arm"]: r for r in panel}
    for p in _find(capture, "vswap.*.json"):
        w = p.name.split(".")[1]
        word = json.loads((p.parent / f"word.{w}.json").read_text())
        rows = json.loads(p.read_text())
        res = {}
        for r in rows:
            res[r["arm"]] = {"d": r["d"], "px_vs_native": r["vs_native"]["px_mad"], "px_vs_del": r["vs_del"]["px_mad"],
                             "touched_calls": r["touched_calls"]}
        if f"z:none:{w}" in by:
            nd = word["mad_native_del"]
            d_iso = 1 - by[f"iso:{w}"]["vs_del"]["px_mad"] / nd
            d_none = 1 - by[f"z:none:{w}"]["vs_del"]["px_mad"] / nd
            iso_row = next(r for r in rows if r["arm"] == "iso-vswap-all")
            res["iso-vswap-all"]["gone_zero_scale"] = (iso_row["d"] - d_iso) / (d_none - d_iso)
            res["roots_match_panel_job"] = {
                "iso": word["roots"]["iso"]["latent_sha256"] == by[f"iso:{w}"]["latent_sha256"],
                "del": word["roots"]["del"]["latent_sha256"] == by[f"del:{w}"]["latent_sha256"]}
        out[w] = res
    g3 = {}
    for w in ("snow", "rain"):
        if w in out:
            g3[w] = {"d_vswap_all": out[w]["vswap-all"]["d"], "holds_d>=0.7": out[w]["vswap-all"]["d"] >= 0.7,
                     "word_only": out[w]["vswap-word"]["d"], "nonword": out[w]["vswap-nonword"]["d"]}
    out["G3"] = g3
    return out


# ========================================================================================= qwen

def qwen(path):
    """PW (exact first-hop write of each value cell at row 21) vs the saved FP32 cell effects."""
    from mdb_lanes.analysis_metrics import correlation_report
    rec = json.loads(Path(path).read_text())
    tables = json.loads((HERE.parent.parent / "qualification/qwen_panels_20261112/full-tables.json").read_text())
    coef = {c["coefficient"]: c["fp32"] for c in tables["b_coefficients"]}
    saved_att = tables["c_correlations_fp32"]["max_source_attention"]
    cells = rec["cells"]
    labels = [c["cell"] for c in cells]
    y = {"singleton": [coef[f"singleton {l}"] for l in labels],
         "complement_loss": [coef[f"complement_loss {l}"] for l in labels]}
    x = {"pw_norm": [c["pw_norm"] for c in cells], "pw_dla_margin": [c["pw_dla_margin"] for c in cells],
         "pw_sum_head_norms": [c["pw_sum_head_norms"] for c in cells],
         "max_source_attention": [c["max_source_attention"] for c in cells],
         "sum_source_attention": [c["sum_source_attention"] for c in cells],
         "native_contribution_norm": [c["native_contribution_norm"] for c in cells]}
    out = {"margin_native": rec["margin_native"], "cells": labels,
           "max_attention_matches_saved_max_abs_diff": max(abs(saved_att[l] - c["max_source_attention"])
                                                           for l, c in zip(labels, cells)),
           "corr": {}}
    for yk, yv in y.items():
        for target, yy in ((yk, yv), (yk + "_abs", [abs(v) for v in yv])):
            for xk, xv in x.items():
                r = correlation_report(xv, yy, labels=labels)
                out["corr"][f"{xk}~{target}"] = {"pearson": r["pearson"], "spearman": r["spearman"],
                                                 "loo_pearson": [r["leave_one_out"]["pearson"]["min"], r["leave_one_out"]["pearson"]["max"]],
                                                 "loo_spearman": [r["leave_one_out"]["spearman"]["min"], r["leave_one_out"]["spearman"]["max"]]}
    out["table"] = [{"cell": l, "pw_norm": c["pw_norm"], "pw_dla_margin": c["pw_dla_margin"],
                     "max_source_attention": c["max_source_attention"], "singleton": s_, "complement_loss": cl}
                    for l, c, s_, cl in zip(labels, cells, y["singleton"], y["complement_loss"])]
    return out


# ======================================================================================= sheets

def sheet(paths, labels, out, title):
    from PIL import Image, ImageDraw
    tiles = [Image.open(p).convert("RGB") for p in paths]
    w, h = tiles[0].size
    canvas = Image.new("RGB", (w * len(tiles), h + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), title, fill="black")
    for i, (t, lab) in enumerate(zip(tiles, labels)):
        canvas.paste(t, (i * w, 34))
        draw.text((i * w + 4, 18), lab, fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def sheets(capture, dirs, out):
    """Contact sheets: per word and cut type (main plan) and per word (value swap)."""
    import re
    out = Path(out)
    safe = lambda a: re.sub("[^A-Za-z0-9]+", "_", a)
    imgs = {}
    for d in dirs:
        for p in Path(d).rglob("*.png"):
            imgs.setdefault(p.stem, p)
    made = []
    words = {"red": "cat", "black": "cat", "ball": "cat", "snow": "cat", "night": "car", "rain": "car"}
    for w, pid in words.items():
        for t, pre in (("zero", "z"), ("renorm", "m")):
            names = [f"native:{pid}", f"del:{w}", f"iso:{w}", f"{pre}:none:{w}"] + [
                f"{pre}:{s}:{w}" for s in ("pw60", "own60", "massmatch-lowpw", "pwmatch-lowmass", "rand60a", "rand60b",
                                           "keep-pw60", "keep-own60", "keep-rand60a")]
            paths = [imgs.get(safe(n)) for n in names]
            if any(x is None for x in paths):
                continue
            labels = [n.split(":")[0] if n.startswith(("native", "del", "iso")) else n.split(":")[1] for n in names]
            made.append(str(sheet(paths, labels, out / f"main-{w}-{t}.png", f"{w} {t} cut, seed 4242")))
    if capture:
        for wdir in sorted((Path(capture) / "images").iterdir()):
            names = ["native", "del", "iso", "vswap-all", "vswap-word", "vswap-nonword", "vswap-sham", "iso-vswap-all"]
            paths = [wdir / f"{safe(n)}.png" for n in names]
            if all(p.exists() for p in paths):
                made.append(str(sheet(paths, names, out / f"vswap-{wdir.name}.png", f"{wdir.name}: value swap, single blocks")))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--plans")
    ap.add_argument("--panels", nargs="*", default=[])
    ap.add_argument("--capture")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "closure":
        print(save("closure.json", closure(a.path)))
    elif a.cmd == "split":
        print(save("split.json", split(a.path)))
    elif a.cmd == "vswap":
        print(save("vswap.json", vswap(a.path, a.panels)))
    elif a.cmd == "arms":
        print(save("arms.json", arms(a.plans, a.panels)))
    elif a.cmd == "sheets":
        print("\n".join(sheets(a.capture, a.panels, a.out)))
    elif a.cmd == "imgpw":
        print(save("imgpw.json", imgpw(a.capture, a.plans, a.path)))
    elif a.cmd == "qwen":
        print(save("qwen.json", qwen(a.path)))

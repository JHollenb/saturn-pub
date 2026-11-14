"""Offline analysis for the interp bridge (numpy + PIL only; no model, no torch).

  python analyze.py q1      --smoke DIR --out q1_table.json
  python analyze.py groups  --smoke DIR --out groups.json            # PREREG section 4 partitions
  python analyze.py saved   --smoke DIR --out saved_sets.json        # saved lane sets vs predictors
  python analyze.py cuts    --smoke DIR --cuts DIR --groups groups.json --out designed.json

DIR = a collected ``interp-bridge`` output directory of one job.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bridge import loo_spearman, partitions, spearman  # noqa: E402  (bridge imports torch lazily)

QUAL = HERE.parents[1] / "qualification"
PLAN = QUAL / "flux_read_panels_20261112" / "plans" / "full.plan.json"
GONE = QUAL / "flux_read_panels_20261112" / "report" / "b-gone-full.json"
WORDS = ("red", "ball")
OTHER = {"red": "black", "ball": "snow"}
CUTS = ("zero", "mask")


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


def terms(smoke, word):
    z = np.load(Path(smoke) / "attribution" / f"terms.{word}.npz")
    return {k: z[k] for k in z.files}


def head_predictors(T, *, source="own", direction="nd"):
    """Per-head (flat 600) predictors summed over steps: removal per cut type, mass, write mass, write size."""
    out = {}
    for cut in CUTS:
        a = T[f"{direction}.{source}.att_{cut}_img"] + T[f"{direction}.{source}.att_{cut}_txt"]
        out[f"r_{cut}"] = -a.sum(0).reshape(-1)
        out[f"r_{cut}_img_only"] = -T[f"{direction}.{source}.att_{cut}_img"].sum(0).reshape(-1)
        out[f"dwrite_{cut}"] = (T[f"nd.{source}.dwrite_{cut}_img"] + T[f"nd.{source}.dwrite_{cut}_txt"]).sum(0).reshape(-1)
    out["mass"] = (T[f"nd.{source}.mass_img"] + T[f"nd.{source}.mass_txt"]).sum(0).reshape(-1)
    out["mass_img_only"] = T[f"nd.{source}.mass_img"].sum(0).reshape(-1)
    out["wmass"] = (T[f"nd.{source}.wmass_img"] + T[f"nd.{source}.wmass_txt"]).sum(0).reshape(-1)
    return out


def flat(units):
    return [int(b) * 24 + int(h) for b, h in units]


def loo_summary(x, y):
    vals = [v for v in loo_spearman(x, y) if not math.isnan(v)]
    return dict(rho=spearman(x, y), loo_min=min(vals) if vals else None, loo_max=max(vals) if vals else None, n=len(x))


def loo_gap(xa, xm, y):
    """Leave-one-out values of rho(a, y) - rho(m, y)."""
    out = []
    for i in range(len(y)):
        keep = [j for j in range(len(y)) if j != i]
        out.append(spearman([xa[j] for j in keep], [y[j] for j in keep]) -
                   spearman([xm[j] for j in keep], [y[j] for j in keep]))
    return out


# ============================================================================================ Q1

def q1(smoke, out):
    s = json.loads((Path(smoke) / "summary.json").read_text())
    rows = json.loads((Path(smoke) / "rows.json").read_text())
    q = s.get("q1", {})
    timing = {k: dict(seconds=[round(x["seconds"], 4) for x in v], peak_mb=[round(x["peak_mb"]) for x in v],
                      equal_native=[x.get("equal") for x in v])
              for k, v in s.get("timing", {}).items()}
    table = dict(job=s["job_id"], software=s["software"], roots={k: dict(equal=v.get("equals_lane_reference"),
                                                                           sha=v["latent_sha256"][:16])
                                                                  for k, v in rows.items() if k.startswith("root:")},
                 capture_compare=q.get("capture_compare"), cut_zero=q.get("cut_zero"), cut_mask=q.get("cut_mask"),
                 port_zero_after_wrap_equal=q.get("port_zero_after_wrap_equal"),
                 native_after_traces_equal=q.get("native_after_traces_equal"), extras=q.get("extras"),
                 grad_check=q.get("grad_check"), errors=q.get("errors"), wrap_seconds=q.get("wrap_seconds"),
                 timing=timing, peak_vram_allocated_mb=s.get("peak_vram_allocated_mb"),
                 worker_seconds=s.get("worker_seconds"))
    save(out, table)
    print(json.dumps({k: v for k, v in table.items() if k not in ("software", "extras")}, indent=1, default=str)[:6000])


# ======================================================================================== groups

def groups(smoke, out):
    branches, meta = [], {}
    for w in WORDS:
        P = head_predictors(terms(smoke, w))
        parts = partitions(list(P["r_zero"]), list(P["mass"]), n_groups=20)
        meta[w] = {}
        for pname, gs in parts.items():
            meta[w][pname] = []
            for gi, g in enumerate(gs):
                units = [[i // 24, i % 24] for i in sorted(g)]
                meta[w][pname].append(dict(group=gi, r_zero=float(P["r_zero"][g].sum()),
                                           r_mask=float(P["r_mask"][g].sum()), mass=float(P["mass"][g].sum())))
                for cut in CUTS:
                    branches.append(dict(id=f"{cut[0]}:{pname}:g{gi:02d}:{w}", word=w, cut=cut, partition=pname,
                                         group=gi, units=units))
    save(out, dict(rule="bridge.partitions(r_zero, mass, n_groups=20); PREREG.md section 4", groups=meta,
                   branches=branches))
    for w in WORDS:
        for pname in ("anti", "diag"):
            g = meta[w][pname]
            print(w, pname, "rho(r_zero, mass) over groups =",
                  round(spearman([x["r_zero"] for x in g], [x["mass"] for x in g]), 3))
    print(len(branches), "branches")


# ==================================================================================== saved sets

def saved(smoke, out):
    plan = json.loads(PLAN.read_text())
    arms = {a["arm"]: a for a in plan["arms"]}
    gone = {r["word"]: r for r in json.loads(GONE.read_text())}
    result = {}
    for w in WORDS:
        T = terms(smoke, w)
        preds = {"own": head_predictors(T, source="own"), "other": head_predictors(T, source="other"),
                 "own_id": head_predictors(T, source="own", direction="id"),
                 "other_id": head_predictors(T, source="other", direction="id")}
        ctx = next(c for c in gone[w]["contexts"] if c["context"] == "4242")
        for cut, prefix in (("zero", "z:"), ("mask", "m:")):
            sets = []
            for col, g in ctx["gone"].items():
                if not col.startswith(prefix):
                    continue
                arm = arms[col.replace("{w}", w)]
                if arm["units"] is None:
                    continue
                src = "own" if arm["rows_item"] == w else "other"
                idx = flat(arm["units"])
                p, pid = preds[src], preds[src + "_id"]
                sets.append(dict(arm=arm["arm"], unit_set=arm["unit_set"], source=src, heads=len(idx), gone=g,
                                 r=float(p[f"r_{cut}"][idx].sum()), r_id=float(pid[f"r_{cut}"][idx].sum()),
                                 mass=float(p["mass"][idx].sum()), wmass=float(p["wmass"][idx].sum()),
                                 dwrite=float(p[f"dwrite_{cut}"][idx].sum())))
            y = [s["gone"] for s in sets]
            stats = {k: loo_summary([s[k] for s in sets], y) for k in ("r", "r_id", "mass", "wmass", "dwrite")}
            gap = spearman([s["r"] for s in sets], y) - spearman([s["mass"] for s in sets], y)
            rng = random.Random(0)
            boots = []
            for _ in range(2000):
                ii = [rng.randrange(len(sets)) for _ in sets]
                yy = [y[i] for i in ii]
                a = spearman([sets[i]["r"] for i in ii], yy)
                m = spearman([sets[i]["mass"] for i in ii], yy)
                if not (math.isnan(a) or math.isnan(m)):
                    boots.append(a - m)
            boots.sort()
            ci = [boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots)) - 1]] if boots else None
            own_sets = [s for s in sets if s["source"] == "own"]
            stats_own = {k: loo_summary([s[k] for s in own_sets], [s["gone"] for s in own_sets]) for k in ("r", "mass")}
            keep = [s for s in sets if s["unit_set"].startswith("keep")]
            result[f"{w}.{cut}"] = dict(n_sets=len(sets), n_own_row_sets=len(own_sets), spearman=stats,
                                        spearman_own_rows_only=stats_own, gap_r_minus_mass=gap, gap_boot90=ci,
                                        keep_sets=[dict(arm=s["arm"], r=s["r"], gone=s["gone"]) for s in keep],
                                        sets=sets)
            print(f"{w:5} {cut:5} n={len(sets):2d} rho_r={stats['r']['rho']:+.3f} rho_mass={stats['mass']['rho']:+.3f} "
                  f"rho_wmass={stats['wmass']['rho']:+.3f} rho_dwrite={stats['dwrite']['rho']:+.3f} gap={gap:+.3f} "
                  f"boot90={[round(x, 3) for x in ci] if ci else None}")
    save(out, result)


# ====================================================================================== designed

def mad(a, b):
    return float(np.abs(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0).mean(-1).mean())


def designed(smoke, cuts_dir, groups_path, out):
    from PIL import Image
    rows = json.loads((Path(cuts_dir) / "rows.json").read_text())
    by = {r["arm"]: r for r in rows}
    G = json.loads(Path(groups_path).read_text())
    img = lambda name: np.asarray(Image.open(Path(cuts_dir) / "images" / f"{name.replace(':', '_')}.png").convert("RGB"))
    result = {"conditions": {}}
    gaps = []
    for w in WORDS:
        P = head_predictors(terms(smoke, w))
        mad_nd = mad(img(f"root:native"), img(f"root:del:{w}"))
        d = lambda r: 1.0 - r["vs_del"]["px_mad"] / mad_nd
        d_iso = 1.0 - mad(img(f"root:iso:{w}"), img(f"root:del:{w}")) / mad_nd
        d_none = d(by[f"z:none:{w}"])
        for cut in CUTS:
            for pname in ("anti", "diag"):
                br = [b for b in G["branches"] if b["word"] == w and b["cut"] == cut and b["partition"] == pname]
                xs_r, xs_m, xs_w, xs_dw, ys, ys_id, gone_px = [], [], [], [], [], [], []
                for b in br:
                    r = by.get(b["id"])
                    if r is None:
                        continue
                    idx = flat(b["units"])
                    xs_r.append(float(P[f"r_{cut}"][idx].sum()))
                    xs_m.append(float(P["mass"][idx].sum()))
                    xs_w.append(float(P["wmass"][idx].sum()))
                    xs_dw.append(float(P[f"dwrite_{cut}"][idx].sum()))
                    ys.append(r["y_nd_iso"] - r["y_nd"])
                    ys_id.append(r["y_id_iso"] - r["y_id"])
                    gone_px.append((d(r) - d_iso) / (d_none - d_iso))
                st = dict(n=len(ys), r=loo_summary(xs_r, ys), mass=loo_summary(xs_m, ys), wmass=loo_summary(xs_w, ys),
                          dwrite=loo_summary(xs_dw, ys), r_vs_gone_px=loo_summary(xs_r, gone_px),
                          mass_vs_gone_px=loo_summary(xs_m, gone_px), r_vs_effect_id=loo_summary(xs_r, ys_id),
                          mass_vs_effect_id=loo_summary(xs_m, ys_id))
                lg = loo_gap(xs_r, xs_m, ys)
                st["gap"] = st["r"]["rho"] - st["mass"]["rho"]
                st["gap_loo_min"] = min(lg)
                st["gap_loo_max"] = max(lg)
                st["linearity"] = dict(sum_measured=float(sum(ys)), sum_predicted=float(sum(xs_r)),
                                       ratio=float(sum(ys) / sum(xs_r)) if sum(xs_r) else None,
                                       slope_through_origin=float(np.dot(xs_r, ys) / np.dot(xs_r, xs_r)),
                                       pearson=float(np.corrcoef(xs_r, ys)[0, 1]))
                st["groups"] = [dict(id=b["id"], r=a, mass=m, effect=y, gone_px=g)
                                for b, a, m, y, g in zip(br, xs_r, xs_m, ys, gone_px)]
                result["conditions"][f"{w}.{cut}.{pname}"] = st
                if pname == "anti":
                    gaps.append(st["gap"])
                print(f"{w:5} {cut:5} {pname:4} n={len(ys)} rho_r={st['r']['rho']:+.3f} [{st['r']['loo_min']:+.3f},"
                      f"{st['r']['loo_max']:+.3f}] rho_m={st['mass']['rho']:+.3f} [{st['mass']['loo_min']:+.3f},"
                      f"{st['mass']['loo_max']:+.3f}] gap={st['gap']:+.3f} loo_gap_min={st['gap_loo_min']:+.3f} "
                      f"ratio={st['linearity']['ratio']:.3f} slope={st['linearity']['slope_through_origin']:.3f}")
    anti = [result["conditions"][f"{w}.{c}.anti"] for w in WORDS for c in CUTS]
    dr = sum(gaps) / len(gaps)
    clear = dr >= 0.40 and all(s["gap"] > 0 for s in anti) and sum(s["gap_loo_min"] > 0 for s in anti) >= 3
    mass_clear = (dr <= -0.40 and all(s["gap"] < 0 for s in anti)
                  and sum(s["gap_loo_max"] < 0 for s in anti) >= 3)
    verdict = ("attribution clearly beats mass" if clear else "mass clearly beats attribution" if mass_clear
               else "no difference" if abs(dr) < 0.15 else "not separated")
    result["primary"] = dict(delta_rho=dr, per_condition=gaps, verdict=verdict)
    print("PRIMARY delta_rho =", round(dr, 3), "->", verdict)
    save(out, result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("q1", "groups", "saved", "cuts"))
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--cuts")
    ap.add_argument("--groups")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "q1":
        q1(a.smoke, a.out)
    elif a.cmd == "groups":
        groups(a.smoke, a.out)
    elif a.cmd == "saved":
        saved(a.smoke, a.out)
    else:
        designed(a.smoke, a.cuts, a.groups, a.out)

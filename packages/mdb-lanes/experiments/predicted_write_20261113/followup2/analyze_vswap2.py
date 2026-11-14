"""Addendum-2 value-swap controls: content, pixel d, latent projections, H4, contact sheets (no model).

  python analyze_vswap2.py JOB_DIR   -> results/vswap2.json, sheets in artifacts/.../sheets/addendum2/

JOB_DIR = the collected ``predicted-write-vswap2`` directory. The red value swap of the capture
job is reproduced in this job (arm ``vswap-nonscene`` of snow; PNG equal to the saved one is
checked here).
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
from analyze import sheet  # noqa: E402

ART = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113")
SAVED = ART / "results/job-86df2dd47809/predicted-write/images"
SHEETS = ART / "sheets/addendum2"


def proj(lat_arm, lat_ref, a, b):
    """<lat_ref - lat_arm, a - b> / |a - b|^2 and |lat_ref - lat_arm| / |a - b|."""
    u = (a - b).ravel()
    d = (lat_ref - lat_arm).ravel()
    return float(d @ u / (u @ u)), float(np.linalg.norm(d) / np.linalg.norm(u))


def main(job_dir):
    job_dir = Path(job_dir)
    out = {"job_dir": str(job_dir), "words": {}}
    for w in ("snow", "rain"):
        rec = json.loads((job_dir / f"vswap2.{w}.json").read_text())
        img = lambda n: job_dir / "images" / w / f"{n}.png"          # noqa: E731
        lat = lambda n: np.load(job_dir / "latents" / w / f"{n}.npy")  # noqa: E731
        L = {k: lat(k) for k in ("native", "del", "iso")}
        measures = {"primary": C.MEASURES[w]}
        if w == "rain":
            measures["rain_coherence"] = C.rain_coherence
        roots_c = {m: {k: f(img(k)) for k in ("native", "del", "iso")} for m, f in measures.items()}
        rows = {}
        for a in rec["arms"]:
            name = a["arm"]
            safe = name.replace("-", "_")
            la = lat(safe)
            r = {"d_px": a["d"], "px_vs_native": a["vs_native"]["px_mad"], "px_vs_del": a["vs_del"]["px_mad"],
                 "source": a.get("source"), "deleted_word": a.get("deleted_word"), "prompt": a.get("prompt"),
                 "same_length": a.get("same_length"), "touched_calls": a["touched_calls"]}
            for m, f in measures.items():
                c = f(img(safe))
                r[m] = {"value": c, "removed_vs_native": C.removed(c, roots_c[m]["native"], roots_c[m]["del"])}
            p_u, n_u = proj(la, L["native"], L["native"], L["del"])
            p_u2, n_u2 = proj(la, L["native"], L["iso"], L["del"])
            r["latent"] = {"proj_on_native_minus_del": p_u, "norm_over_native_minus_del": n_u,
                           "proj_on_iso_minus_del": p_u2, "norm_over_iso_minus_del": n_u2}
            rows[name] = r
        # global contrast / colour beside the content measure (crushing contrast also lowers rain_streak)
        def stats(p):
            a = C.rgb(p)
            y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
            mx, mn = a.max(-1), a.min(-1)
            return {"mean_luma": float(y.mean()), "std_luma": float(y.std()),
                    "mean_saturation": float(np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-12), 0).mean())}
        tone = {n: stats(img(n.replace("-", "_"))) for n in ("native", "del", "iso", *[a["arm"] for a in rec["arms"]])}
        h4_ratio = rows["vswap-unrel"]["primary"]["removed_vs_native"] / rows["vswap-all"]["primary"]["removed_vs_native"]
        out["words"][w] = {"roots_content": roots_c, "arms": rows, "tone": tone, "mad_native_del": rec["mad_native_del"],
                           "H4_ratio_unrel_over_all": h4_ratio, "H4_cell_holds": bool(h4_ratio >= 2 / 3),
                           "gate_rms_native_root": rec["gate_rms_native_root"]}
        names = ["native", "del", "iso", "vswap_all", "vswap_unrel", "vswap_nonscene", "vswap_sham"]
        labels = ["native", "del", "iso", "vswap-all (deleted word)", "vswap-unrel", f"vswap-nonscene ({rows['vswap-nonscene']['deleted_word']})",
                  "vswap-sham"]
        sheet([img(n) for n in names], labels, SHEETS / f"vswap2-{w}.png", f"{w}: value swaps, single blocks (addendum 2)")
    # reproduction checks against the capture job
    from PIL import Image
    eq = lambda a, b: bool(np.array_equal(np.asarray(Image.open(a).convert("RGB")), np.asarray(Image.open(b).convert("RGB"))))  # noqa: E731
    out["reproduction"] = {
        "snow vswap-all == capture snow vswap_all": eq(job_dir / "images/snow/vswap_all.png", SAVED / "snow/vswap_all.png"),
        "rain vswap-all == capture rain vswap_all": eq(job_dir / "images/rain/vswap_all.png", SAVED / "rain/vswap_all.png"),
        "snow vswap-nonscene (red) == capture red vswap_all": eq(job_dir / "images/snow/vswap_nonscene.png", SAVED / "red/vswap_all.png"),
        "snow native == capture": eq(job_dir / "images/snow/native.png", SAVED / "snow/native.png"),
        "rain native == capture": eq(job_dir / "images/rain/native.png", SAVED / "rain/native.png")}
    # gate stratum check: mean gate RMS (image gate) per stratum, native root
    gs = {}
    for w, r in out["words"].items():
        g = np.array(r["gate_rms_native_root"])
        j, s = g[g[:, 0] < 5], g[g[:, 0] >= 5]
        gs[w] = {"joint_img_rms_mean": float(j[:, 1].mean()), "joint_txt_rms_mean": float(j[:, 2].mean()),
                 "single_rms_mean": float(s[:, 1].mean()),
                 "joint_img_over_single": float(j[:, 1].mean() / s[:, 1].mean()),
                 "joint_txt_over_single": float(j[:, 2].mean() / s[:, 1].mean())}
    out["gate_rms_check"] = gs
    w4 = out["words"]
    out["H4"] = {"rule": "removed(vswap-unrel) >= 2/3 removed(vswap-all), primary content measure, snow and rain",
                 "cells": {w: {"ratio": w4[w]["H4_ratio_unrel_over_all"], "holds": w4[w]["H4_cell_holds"]} for w in w4},
                 "status": "held" if all(w4[w]["H4_cell_holds"] for w in w4) else "failed"}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/vswap2.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for w, r in out["words"].items():
        print(w, "roots", {m: {k: round(v, 4) for k, v in x.items()} for m, x in r["roots_content"].items()})
        for n, a in r["arms"].items():
            print("  ", n, "d", round(a["d_px"], 3), {m: round(a[m]["removed_vs_native"], 3) for m in a if isinstance(a[m], dict) and "removed_vs_native" in a[m]},
                  {k: round(v, 3) for k, v in a["latent"].items()})
        print("   tone", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r["tone"].items()})
    print(out["H4"], out["reproduction"], out["gate_rms_check"])


if __name__ == "__main__":
    main(sys.argv[1])

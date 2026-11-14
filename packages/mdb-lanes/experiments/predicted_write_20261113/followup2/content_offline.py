"""Validate the rain / snow content measures (content.py) and score every saved rain / snow arm.

  python content_offline.py   -> results/content-validation.json, results/content-main.json

Saved data only: capture job-86df2dd47809 (roots, value-swap arms) and main-plan jobs
job-5eeb743a56cd / job-3be01d6f10e4 (all arms and floors). No model.

Validation criteria (written after looking at the native / deleted / sham values printed by a first
pass, before any addendum-2 job; reported as they come out):
  V1  span |c(native) - c(del)| >= 10 x max |c(sham) - c(sham's reference)| over every sham image
      (value-swap recompute sham vs native; main-plan sham / sham_recompute floors vs iso);
  V2  the all-cut references z:none and m:none remove >= 0.7 (removed vs iso and del);
  V3  the random 60-unit cuts rand60a / rand60b remove |removed| <= 0.10.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import content as C

HERE = Path(__file__).resolve().parent
RAW = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results")
CAP = RAW / "job-86df2dd47809/predicted-write"
PANEL = {"snow": RAW / "job-3be01d6f10e4/flux-read-panel", "rain": RAW / "job-3be01d6f10e4/flux-read-panel"}
ARMS_JSON = HERE.parent / "results/arms.json"
EXTRA = {"rain": {"rain_coherence": C.rain_coherence}, "snow": {}}


def save(name, value):
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / name).write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def panel_images(word):
    d = PANEL[word] / "images" / "main--r2-01" / "4242"
    return {p.stem: p for p in sorted(d.glob(f"*_{word}*.png"))}


def main():
    effects = json.loads(ARMS_JSON.read_text())["effects"]["main"]
    validation, main_rows = {}, {}
    for word in ("rain", "snow"):
        measures = {"primary": C.MEASURES[word], **EXTRA[word]}
        roots = {k: CAP / "images" / word / f"{k}.png" for k in ("native", "del", "iso")}
        imgs = panel_images(word)
        val = {}
        for mname, f in measures.items():
            c = {k: f(p) for k, p in roots.items()}
            shams = {"vswap_sham (vs native)": (f(CAP / "images" / word / "vswap_sham.png"), c["native"])}
            for stem, p in imgs.items():
                if stem.endswith(("_sham", "_sham_recompute")):
                    shams[stem + " (vs iso)"] = (f(p), c["iso"])
            dev = {k: abs(a - b) for k, (a, b) in shams.items()}
            span = abs(c["native"] - c["del"])
            allcut = {t: C.removed(f(imgs[f"{t}_none_{word}"]), c["iso"], c["del"]) for t in ("z", "m")}
            rand = {f"{t}_{r}": C.removed(f(imgs[f"{t}_{r}_{word}"]), c["iso"], c["del"])
                    for t in ("z", "m") for r in ("rand60a", "rand60b")}
            val[mname] = {"roots": c, "span_native_del": span, "sham_max_abs_dev": max(dev.values()),
                          "sham_worst": max(dev, key=dev.get), "n_shams": len(dev),
                          "V1_span_over_sham": span / max(dev.values()), "V1": span >= 10 * max(dev.values()),
                          "allcut_removed": allcut, "V2": all(v >= 0.7 for v in allcut.values()),
                          "random_removed": rand, "V3": all(abs(v) <= 0.10 for v in rand.values())}
        validation[word] = val
        rows = {}
        for arm, e in effects.items():
            if e["word"] != word:
                continue
            stem = arm.replace(":", "_").replace("-", "_")
            p = imgs.get(stem)
            if p is None:
                continue
            ref = val["primary"]["roots"]
            r = {"gone_px": e["gone"], "cut_type": e["cut_type"], "unit_set": e["unit_set"]}
            for mname, f in measures.items():
                rr = val[mname]["roots"]
                cv = f(p)
                r[mname] = {"value": cv, "removed_vs_iso": C.removed(cv, rr["iso"], rr["del"])}
            rows[arm] = r
        for name in ("vswap_all", "vswap_word", "vswap_nonword", "vswap_sham", "iso_vswap_all"):
            p = CAP / "images" / word / f"{name}.png"
            r = {}
            for mname, f in measures.items():
                rr = val[mname]["roots"]
                cv = f(p)
                ref = "iso" if name.startswith("iso_") else "native"
                r[mname] = {"value": cv, f"removed_vs_{ref}": C.removed(cv, rr[ref], rr["del"])}
            rows["capture:" + name] = r
        main_rows[word] = rows
    # the red / ball value swaps on the cat prompt scored with the snow measure (non-scene-word swaps)
    other = {}
    snow_roots = validation["snow"]["primary"]["roots"]
    for w in ("red", "ball"):
        for name in ("native", "del", "vswap_all", "vswap_sham"):
            cv = C.snow_ground(CAP / "images" / w / f"{name}.png")
            other[f"{w}:{name}"] = {"snow_ground": cv,
                                    "removed_vs_native_snow_span": C.removed(cv, snow_roots["native"], snow_roots["del"])}
    save("content-validation.json", {"definitions": C.__doc__, "validation": validation})
    save("content-main.json", {"arms": main_rows, "cat_prompt_other_word_swaps_on_snow_measure": other})
    for w, v in validation.items():
        for m, x in v.items():
            print(w, m, {k: x[k] for k in ("roots", "span_native_del", "sham_max_abs_dev", "V1_span_over_sham", "V1",
                                           "allcut_removed", "V2", "V3")})
            print("   rand", {k: round(r, 3) for k, r in x["random_removed"].items()})


if __name__ == "__main__":
    main()

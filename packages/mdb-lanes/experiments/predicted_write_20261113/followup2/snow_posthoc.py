"""POST HOC: snow_ground_bright beside the preregistered snow_ground, all snow images (no model).

  python snow_posthoc.py  -> results/snow-bright-posthoc.json
"""
import json
from pathlib import Path

import content as C

HERE = Path(__file__).resolve().parent
R = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results")
SRC = {"main": R / "job-3be01d6f10e4/flux-read-panel/images/main--r2-01/4242",
       "gm": R / "job-57fe341aa4e2/flux-read-panel/images/gm-01/4242",
       "vswap2": R / "job-6d9fea396262/predicted-write-vswap2/images/snow",
       "capture": R / "job-86df2dd47809/predicted-write/images/snow"}
roots = {k: C.snow_ground_bright(SRC["capture"] / f"{k}.png") for k in ("native", "del", "iso")}
out = {"roots": roots, "images": {}}
for tag, d in SRC.items():
    for p in sorted(d.glob("*.png")):
        if tag in ("main", "gm") and "_snow" not in p.stem:
            continue
        b, s = C.snow_ground_bright(p), C.snow_ground(p)
        ref = "native" if tag in ("vswap2", "capture") else "iso"
        out["images"][f"{tag}:{p.stem}"] = {
            "bright": b, "bright_removed": C.removed(b, roots[ref], roots["del"]), "ref": ref,
            "prereg": s}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results/snow-bright-posthoc.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
sh = [abs(v["bright"] - roots[v["ref"]]) for k, v in out["images"].items() if k.endswith(("_sham", "_sham_recompute")) or k.endswith("vswap_sham")]
print("roots", {k: round(v, 3) for k, v in roots.items()}, "span", round(roots["native"] - roots["del"], 3), "max sham dev", round(max(sh), 4), len(sh))
for k, v in out["images"].items():
    if k.endswith(("recompute", "_sham")) or "keep" in k:
        continue
    print(f"{k:45s} bright_removed {v['bright_removed']:+.3f}  prereg {v['prereg']:.3f}")

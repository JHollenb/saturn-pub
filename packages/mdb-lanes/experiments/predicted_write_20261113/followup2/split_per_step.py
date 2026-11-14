"""Routing / payload shares per step and per block stratum (audit item 7), from saved split.<w>.json.

  python split_per_step.py   -> results/split-per-step-block.json   (no model; uses ../analyze.py _agg)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402

CAP = Path("/Users/jakeholl/domains/artifacts/mdb-predicted-write-20261113/results/job-86df2dd47809/predicted-write")
out = {}
for w in ("red", "black", "ball", "snow", "night", "rain"):
    cells = json.loads((CAP / f"split.{w}.json").read_text())["cells"]
    sel = {"step0_joint": lambda s, b: s == 0 and b < 5, "step0_all": lambda s, b: s == 0,
           "joint_all_steps": lambda s, b: b < 5, "all": lambda s, b: True}
    for s0 in range(4):
        sel[f"step{s0}_joint"] = lambda s, b, s0=s0: s == s0 and b < 5
        sel[f"step{s0}_single"] = lambda s, b, s0=s0: s == s0 and b >= 5
    r = {}
    for name, keep in sel.items():
        a = A._agg(cells, keep)
        r[name] = {**{k: a["projection_share"][k] for k in ("routing", "payload", "interaction")},
                   **{"payload_" + k: v for k, v in a["payload_groups_projection_share"].items()}}
    out[w] = r
(HERE / "results/split-per-step-block.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print({w: {k: round(v, 3) for k, v in r["step0_joint"].items()} for w, r in out.items()})

"""Offline analysis of the equivalence smoke (artifact-only; numpy; no model).

  PYTHONPATH=$W/packages/mdb-lanes/src:$W/packages/mdb/src:$W/src python analyze.py

Reads ``results/job-0fcdf8414918.json`` (the worker's QWENENG_RESULT line) and the saved FP32 read
rows of the Qwen panel jobs (``qualification/panel_verdicts_recompute_20261113.QWEN_RUNS``), and writes
``results/analysis.json``:

* determinism: the smoke's lane margins against the saved lease rows of the same arms;
* gaps in FP32 spacings at the margin: engine - lane recompute, engine - lane delta, and the lane's
  own twin gap |delta - recompute|;
* verdicts: the sealed FP32 read plan's verdicts from the saved rows, then with the engine's margins
  in place of the lane's for every arm the engine measured (P.verdicts, unchanged logic).
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUAL = HERE.parents[1] / "qualification"
sys.path.insert(0, str(QUAL))
import qwen_panel_analyze as A  # noqa: E402
from mdb_lanes import attention_panels as P  # noqa: E402
from mdb_lanes import qwen_panels as Q  # noqa: E402

RUNS = [Path("/Users/jakeholl/domains/artifacts/mdb-qwen-panels-20261112") / j
        for j in ("job-b33e6b98c328", "job-4a433df3f30b", "job-7c3232ad65bb")]


def spacing(x):
    """FP32 spacing at |x|."""
    m, e = math.frexp(abs(x))
    return 2.0 ** (e - 24)


def main():
    smoke = json.loads((HERE / "results" / "job-0fcdf8414918.json").read_text())
    rows, _, _ = A.rows_of(RUNS)
    saved = {r["arm"]: r for r in rows[("float32", "read-float32")]}
    lane = smoke["lane_margins"]
    det = {name: {"smoke": m, "saved": saved[name]["margin"], "equal": m == saved[name]["margin"]}
           for name, m in lane.items() if name in saved}
    gaps = {}
    for name, a in smoke["arms"].items():
        sp = spacing(a["lane_margin"])
        gaps[name] = {"engine_minus_recompute_spacings": (a["engine_margin"] - a["lane_margin"]) / sp,
                      "engine_minus_delta_spacings": (a["engine_margin"] - a["lane_delta_margin"]) / sp,
                      "twin_gap_spacings": abs(a["lane_delta_margin"] - a["lane_margin"]) / sp,
                      "shift_from_native": a["lane_margin"] - lane["native"], "staged": a["staged"]}
    plan = json.loads((QUAL / "qwen_panels_20261112/full/read-float32.plan.json").read_text())
    panel = P.Panel.from_json(plan)
    base_rows = list(saved.values())
    swapped = [dict(r, margin=smoke["arms"][r["arm"]]["engine_margin"]) if r["arm"] in smoke["arms"] else r
               for r in base_rows]
    swapped = [dict(r, margin=smoke["base"]["engine_margin"]) if r["arm"] == "native" else r for r in swapped]
    status = {}
    for tag, rr in (("lane", base_rows), ("engine_substituted", swapped)):
        status[tag] = [r["status"] for r in P.verdicts(panel, Q.read_results(panel, rr))]
    changed = [i for i, (a, b) in enumerate(zip(status["lane"], status["engine_substituted"])) if a != b]
    vals = list(gaps.values())
    out = {"smoke_job": "job-0fcdf8414918", "saved_runs": [str(r) for r in RUNS],
           "determinism": {"arms": len(det), "equal": sum(d["equal"] for d in det.values()),
                           "unequal": {k: v for k, v in det.items() if not v["equal"]}},
           "gaps": gaps,
           "gap_summary": {k: {"max_abs": max(abs(v[k]) for v in vals), "zero": sum(v[k] == 0 for v in vals)}
                           for k in ("engine_minus_recompute_spacings", "engine_minus_delta_spacings",
                                     "twin_gap_spacings")},
           "verdict_counts": {k: dict(collections.Counter(v)) for k, v in status.items()},
           "verdicts_changed": len(changed), "substituted_arms": len(smoke["arms"]) + 1}
    (HERE / "results" / "analysis.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("determinism", "gap_summary", "verdict_counts", "verdicts_changed",
                                          "substituted_arms")}, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

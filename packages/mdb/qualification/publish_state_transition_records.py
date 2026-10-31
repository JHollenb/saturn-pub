"""Publish comparisons/catalogue from already sealed evidence, without model calls.

The initial worker's optional catalog publication failed after all observations
were persisted. This tool writes a new derived directory; raw results stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from mdb.equation_comparison import EquationCatalog, compare_equations


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def publish(run_dir, output):
    helpers = runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))
    run = helpers["source_run"](run_dir, "state-transition-result", failed=True)
    output.mkdir(parents=True, exist_ok=False)
    predictions = run["tree"].decode(helpers["read"](run["transaction"] / "forecasts/campaign.json")["predictions"])
    observed = run["tree"].decode(helpers["read"](run["transaction"] / "observations/native.json")["value"])
    plan, request = run["plan"], run["request"]
    catalog = EquationCatalog(output / "equations")
    rules = {name: "argmax" if name in ("dest", "src", "width", "length") else "positive"
             for name in ("regs", "flags", "dest", "src", "width", "length", "syscall", "imm", "store", "store_addr", "store_value")}

    def prediction(row):
        value = predictions
        try:
            for key in row["prediction_path"]:
                value = value[key]
            return value
        except KeyError:
            return None

    reports = []
    for row_id, actual in observed.items():
        predicted = prediction(actual)
        if predicted is None:
            reports.append({"row_id": row_id, "forecast_status": "forecast-horizon-ended", "actual_decode": actual["decoded"]})
            continue
        baseline_id = ("sequence.native." + row_id.rsplit(".", 1)[1] if actual["panel"] == "sequence" else
                       "sequence.U_swap.0" if actual["panel"] == "path" else ".".join(row_id.split(".")[:2]) + ".native")
        baseline = observed.get(baseline_id)
        baseline_prediction = None if baseline is None else prediction(baseline)
        if baseline_prediction is None:
            baseline = None
        heads = compare_equations(predicted["heads"], actual["heads"], decision_rules=rules,
            baseline=None if baseline is None else baseline["heads"],
            predicted_baseline=None if baseline is None else baseline_prediction["heads"],
            targets={"regs": [[0, 7, bit] for bit in range(64)] + [[0, 6, 1]]}, components={"regs": predicted["register_terms"]})
        stages = compare_equations({name: predicted["stages"][name] for name in actual["stages"]}, actual["stages"])
        horizon = 5 if actual["panel"] == "sequence" else 1
        context = {"checkpoint": plan["model_identity"], "source_identity": identity(request["source_hashes"]),
            "execution": plan["execution"], "horizon": horizon, "panel": actual["panel"], "case_id": row_id,
            "arm": row_id.split(".")[1], "instruction": int(row_id.rsplit(".", 1)[1]) if actual["panel"] == "sequence" else 0,
            "segment": actual["lineage"].get("segment_id"), "program_hex": actual["lineage"]["program_hex"],
            "register_words": actual["before"]["register_words"], "flags": actual["before"]["flags"],
            "byte_offset": actual["before"]["offset"], "input_identity": actual["input_identity"]}
        route = "+".join(edit["selector"] + "." + edit["patch_kind"] for edit in plan["row_recipes"].get(row_id, [])) or "native.forward"
        same_roots = baseline is not None and all(np.array_equal(actual["roots"][key], baseline["roots"][key]) for key in actual["roots"])
        contrast = "matched-input" if same_roots else "own-state-trajectory-contrast"
        record = catalog.record(law_id="r3.raw-state-transition", route_id=route,
            binding={"model_identity": plan["model_identity"], "source_identity": request["source_hashes"],
                     "execution": plan["execution"], "horizon": horizon, "input_identity": actual["input_identity"], "frontier": "raw-input"},
            conditions=[{"field": key, "op": "eq", "value": context[key]} for key in context], tested_context=context,
            readers=["regs[6,1]", "regs[7,*]", "flags"], report=heads,
            evidence={"forecast_seal": run["transaction_report"]["seal_sha256"], "capture": actual["trace_fingerprint"],
                      "baseline_row": baseline_id, "forecast_root_join": actual["forecast_root_join"],
                      "contrast_scope": contrast, "posthoc_publication": True, "source": run["source"]})
        reports.append({"row_id": row_id, "baseline_row": baseline_id, "head_report": heads, "stage_report": stages,
                        "equation_record": record["fingerprint"], "contrast_scope": contrast})
    counts = {"physical_forwards": len(helpers["read"](run["data"] / "native-events.json")),
              "replay_operations": len(helpers["read"](run["data"] / "public-replay-receipts.json")),
              "acts": sum(len(row["acts"]) for row in observed.values())}
    summary = {"schema": "mdb-state-transition-result-v1", "job_id": run["source"]["job_id"],
        "counts": counts, "rows": len(observed), "source": run["source"],
        "mechanics_status": "pending-independent-audit", "terminal_status": "unassessed",
        "publication_scope": "offline comparison/catalogue publication from retained failed-job observations; zero new native calls"}
    for name, value in (("comparisons.json", {"rows": reports}), ("summary.json", summary)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"rows": len(observed), "comparisons": len(reports), "counts": counts, "model_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((args.run_dir / "extracted").resolve()):
        parser.error("derived output must stay outside immutable raw results")
    print(json.dumps(publish(args.run_dir, args.output)))


if __name__ == "__main__":
    main()

"""Publish a portable index from independent audits; no model or equation calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def summarize(run_dir, *, atlas_index=None):
    audit = read(run_dir / "analysis-v1.json")
    oracle = read(run_dir / "derived/isa-oracle-score.json")
    comparisons = {row["row_id"]: row for row in read(run_dir / "derived/comparisons.json")["rows"]}
    if audit["mechanics_status"] != "valid":
        raise ValueError("portable valid-mechanics publication requires the independent instrument audit")
    rows = {}
    for key, row in audit["rows"].items():
        head = row["head_comparison"]
        effects = audit["consumer_effects"][key]
        rows[key] = {"native_decode": row["native_decode"], "forecast_decode": row["forecast_decode"],
            "head_decision_differences": {name: value["decision_differences"] for name, value in head.items()},
            "head_max_abs": {name: value["residual"]["max_abs"] for name, value in head.items()},
            "stage_max_abs": {name: value["max_abs"] for name, value in row["stage_comparison"].items()},
            "forecast_native_roots_match": all(row["forecast_native_root_joins"].values()),
            "RSI_bit1": effects["RSI_bit1"], "RDI": {name: value for name, value in effects["RDI"].items() if name != "sealed_receiving_terms"},
            "baseline_row": effects["baseline_row"], "contrast_scope": effects["contrast_scope"],
            "same_serving_roots": effects["same_serving_roots"], "equation_record": comparisons[key]["equation_record"],
            "posthoc_ISA_score": oracle["rows"][key]}
    result = {"schema": "mdb-state-transition-publication-v1", "source": audit["source"],
        "worker_status": audit["worker_status"], "worker_failure": audit["worker_failure"],
        "mechanics_status": audit["mechanics_status"], "terminal_status": "unassessed",
        "counts": audit["counts"], "parameter_inventory": audit["parameter_inventory"],
        "own_chain_closure": audit["own_chain_closure"], "scientific_observations": audit["scientific_observations"],
        "previous_instrument_attempt": audit["previous_instrument_attempt"], "panel_reuses": audit["panel_reuses"],
        "rows": rows, "oracle": oracle["oracle"], "analysis_sources": audit["analysis_sources"],
        "audits": {"native_and_forecast": reference(run_dir / "analysis-v1.json"),
                   "posthoc_ISA": reference(run_dir / "derived/isa-oracle-score.json"),
                   "derived_comparisons": reference(run_dir / "derived/comparisons.json")},
        "model_calls_during_publication": 0, "equation_evaluations_during_publication": 0,
        "limits": ["Known initial contrast plus newly tested own-state futures and source-path restorations, all globally forecast before native release.",
                   "Full supplied source equations and weights; neither autonomous semantic naming nor transferable rule induction.",
                   "Explicit host segments, not one contiguous multi-instruction program; no hidden-state persistence across forwards.",
                   "OR/AND native ISA errors are preserved separately from accurate model computation forecasts.",
                   "Failed-job native evidence audited and indexed offline; scheduler failure and raw artifacts remain unchanged.",
                   "Original unused failed-attempt numerical bytes remain remote; copied metadata plus worker verification cover retained predictions."]}
    if atlas_index is not None:
        result["atlas"] = {"reference": reference(atlas_index), "index": read(atlas_index)}
    package = run_dir.parent / "package-proof.json"
    if package.is_file():
        proof = read(package)
        result["package_qualification"] = {"reference": reference(package),
            **{key: value for key, value in proof.items() if key != "source_hashes"}}
    evidence = run_dir.parent / "evidence"
    if evidence.is_dir():
        result["evidence_plane"] = {"scope": "target-streams-only; stream mechanics only",
            "files": [reference(path) for path in sorted(evidence.glob("*.json"))]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--atlas-index", type=Path)
    args = parser.parse_args()
    value = summarize(args.run_dir, atlas_index=args.atlas_index)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(value["rows"]), "output": str(args.output), "model_calls": 0}))


if __name__ == "__main__":
    main()

"""Summarize recovered frozen discovery evidence without a model framework."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text())


def summarize(path: Path):
    case = read(path)
    report = case["discover_report"]
    candidates = {}
    receivers = []
    for receiver in report["receivers"]:
        selected = []
        for candidate in receiver["candidates"]:
            summary = {key: candidate[key] for key in (
                "id", "offset", "layer", "kv_head", "source_row", "source_token_id",
                "read_score", "group_write_l2", "value_sha256", "key_sha256")}
            selected.append(summary)
            candidates[candidate["id"]] = summary
        receivers.append({"offset": receiver["offset"], "candidates": selected})
    tests = []
    for test in report["tests"]:
        reference = test["reference"]["trajectory"]
        edited = test["candidate"]["trajectory"]
        rows = [{"offset": a["offset"], "authority": a["input_authority"],
                 "reference_consumed": b["consumed_token_id"], "edited_consumed": a["consumed_token_id"],
                 "reference_next": b["next_token_id"], "edited_next": a["next_token_id"]}
                for a, b in zip(edited, reference, strict=True)]
        tests.append({"candidate_id": test["candidate_id"], "scale": test["scale"],
                      "receiver_offset": test["receiver_offset"],
                      "local_prediction": {key: test["local_prediction"][key] for key in (
                          "query_and_keys_exact", "error_l2", "error_max_abs", "relative_l2")},
                      "changed_consumed": [r for r in rows if r["reference_consumed"] != r["edited_consumed"]],
                      "changed_next": [r for r in rows if r["reference_next"] != r["edited_next"]],
                      "effects": test["effect"]["steps"]})
    coverage = [item for receiver in report["receivers"] for item in receiver["coverage"]]
    baseline = [{key: row[key] for key in ("offset", "input_authority", "consumed_token_id", "next_token_id", "margin")}
                for row in report["baseline"]["trajectory"]]
    return {"id": case["id"], "prompt": case["prompt"], "followup": case["followup"],
            "raw_case_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "model_identity": report["model_identity"], "lane": report["lane"],
            "numerical_contract": report["numerical_contract"],
            "instrument_identity": report["instrument_identity"],
            "spec_sha256": report["spec_sha256"], "spec": report["spec"],
            "baseline": baseline, "receivers": receivers, "tests": tests,
            "controls": case["instrumentation_audit"],
            "all_first_queries_and_keys_exact": all(t["local_prediction"]["query_and_keys_exact"] for t in tests),
            "coverage_rows": len(coverage),
            "max_attention_reconstruction_error": max(c["attention_reconstruction_max_abs"] for c in coverage),
            "selected_parent_restored_exact": report["selected_parent_restored_exact"],
            "sibling_selections_preserved": report["sibling_selections_preserved"],
            "operation_upper_bound": report["operation_upper_bound"],
            "elapsed_seconds": case["elapsed_seconds"], "semantic_cot_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recovery_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.recovery_root
    result = {"schema": "mdb-circuit-discovery-frozen-summary-v1",
              "recovery": read(root / "recovery-custody.json"),
              "offline_verification": read(root / "installed-offline-verification-calibrated.json"),
              "original_job_state": "failed: post-run artifact file-count bound",
              "original_rejection_retained": "offline-verification.json",
              "cases": [summarize(p) for p in sorted((root / "results").glob("case-*.json"))],
              "terminal_status": "not-assessed", "semantic_cot_qualified": False}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(result["cases"]),
                      "tests": sum(len(c["tests"]) for c in result["cases"])}))


if __name__ == "__main__":
    main()

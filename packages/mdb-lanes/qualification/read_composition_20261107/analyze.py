"""Framework-free summary of frozen two-read composition evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def summarize(run_dir):
    source = run_dir / "report.json"
    report = read(source)
    rows = []
    for case in report["cases"]:
        for phase in ("first", "second_AB_world"):
            bundle = case[phase]
            arms = {}
            for name, arm in bundle["arms"].items():
                first = arm["trajectory"][0]
                arms[name] = {"first_next_text": first["next_text"],
                              "direct_object_token_match": first["correct_first_answer"],
                              "first_two_object_margin": first["target_margin"],
                              "next_token_ids": [t["next_token_id"] for t in arm["trajectory"]],
                              "consumed_generated_ids": [t["consumed_token_id"] for t in arm["trajectory"][1:]],
                              "end_cut": arm["end_cut"]}
            rows.append({"case": case["case"]["id"], "phase": phase,
                         "owner_relation": case["case"]["previous_current_owner_relation"],
                         "parent_cut": bundle["parent_cut"], "sites": bundle["sites"],
                         "local_errors": bundle["local_errors"],
                         "query22_change": bundle["query22_change"],
                         "current_key22_change": bundle["current_key22_change"],
                         "current_value22_change": bundle["current_value22_change"],
                         "old_keys22_unchanged": bundle["old_keys22_unchanged"],
                         "A_vs_AB_input22_exact": bundle["A_vs_AB_input22_exact"],
                         "A_vs_AB_query_keys_current_value_exact": bundle["A_vs_AB_query_keys_current_value_exact"],
                         "A_vs_00_source_B_exact": bundle["A_vs_00_source_B_exact"],
                         "layer22_exit_interaction_l2": bundle["layer22_exit_interaction_l2"],
                         "target_margin_factorial_interaction": bundle["target_margin_factorial_interaction"],
                         "prediction_ref": bundle["prediction_ref"], "arms": arms,
                         "future_comparisons": bundle["future_comparisons"]})
    aggregates = {}
    for kind in ("attention", "mlp", "exit"):
        methods = {}
        for method in ("conditional", "stale"):
            values = [r["local_errors"][method][kind]["relative_l2"] for r in rows]
            methods[method] = {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
        ratios = [r["local_errors"]["stale"][kind]["relative_l2"]
                  / r["local_errors"]["conditional"][kind]["relative_l2"] for r in rows]
        methods["stale_over_conditional_ratio"] = {"min": min(ratios), "max": max(ratios),
                                                   "median": statistics.median(ratios)}
        methods["conditional_lower_error_count"] = sum(ratio > 1 for ratio in ratios)
        aggregates[kind] = methods
    future = {}
    for method in ("conditional", "stale"):
        comparisons = [x for row in rows for x in row["future_comparisons"][method]]
        future[method] = {"same_next": sum(x["same_next"] for x in comparisons),
                          "next_decisions": len(comparisons),
                          "same_generated_consumed": sum(x["same_consumed"] for x in comparisons if x["offset"] > 0),
                          "generated_consumed": sum(x["offset"] > 0 for x in comparisons)}
    future["conditional_lower_logit_l2"] = sum(
        a["l2"] < b["l2"] for row in rows
        for a, b in zip(row["future_comparisons"]["conditional"], row["future_comparisons"]["stale"], strict=True))
    audit = run_dir / "independent-verification.json"
    result = {"schema": "mdb-read-composition-summary-v1", "job_id": report["job_id"],
              "request_sha256": report["request_sha256"], "report_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "program": report["program"], "model": report["model"], "weights_sha256": report["weights_sha256"],
              "source_hashes": report["source_hashes"], "custody": report["artifact_custody"],
              "receiver_count": len(rows), "local_error_aggregates": aggregates,
              "future_agreement": future, "rows": rows,
              "physical_token_step_calls": report["physical_token_steps"],
              "prefill_calls": report["prefill_calls"], "prefill_tokens": report["prefill_token_count"],
              "explicit_dense_mlp_linear_calls": report["explicit_dense_mlp_linear_calls"],
              "recurrent_policy_first_words": {c["case"]["id"]: {
                  name: p["outcome"]["trajectory"][0]["next_text"] for name, p in c["recurrent_policies"].items()}
                  for c in report["cases"]},
              "independent_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest() if audit.exists() else None,
              "semantic_score_scope": "direct object token at first readout, not eventual free-form answer correctness",
              "prediction_assistance": report["prediction_scope"],
              "terminal_status": "not-assessed", "semantic_cot_qualified": False,
              "economical_query_formation_qualified": False}
    return result


def write_evidence_projection(result, destination):
    """Export only the 48 local scalar observations, without inventing gates."""
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for receiver in result["rows"]:
        for method in ("conditional", "stale"):
            for boundary in ("attention", "mlp", "exit"):
                rows.append({"schema": "mdb-read-composition-scalar-v1",
                             "job_id": result["job_id"], "model": result["model"],
                             "report_sha256": result["report_sha256"],
                             "program_fingerprint": result["program"]["fingerprint"],
                             "case": receiver["case"], "phase": receiver["phase"],
                             "method": method, "boundary": boundary,
                             **receiver["local_errors"][method][boundary]})
    (destination / "observations.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    (destination / "source.json").write_text(json.dumps({
        "schema": "mdb-read-composition-evidence-projection-v1",
        "job_id": result["job_id"], "report_sha256": result["report_sha256"],
        "observations": len(rows), "scope": "local scalar observations only",
        "independent_audit_sha256": result["independent_audit_sha256"],
        "terminal_status": "not-assessed", "gates_added": False,
    }, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if args.evidence_dir:
        write_evidence_projection(result, args.evidence_dir)
    print(json.dumps({"receivers": result["receiver_count"], "local_errors": result["local_error_aggregates"],
                      "future_agreement": result["future_agreement"]}))


if __name__ == "__main__":
    main()

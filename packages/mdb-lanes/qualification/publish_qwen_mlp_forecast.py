"""Publish a portable saved-audit Qwen index; no native or equation execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path, root):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}


def publish(root):
    root = Path(root).resolve()
    helper = runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))
    directory = root / "run-02"
    run = helper["source_run"](directory, "qwen-mlp-result")
    audit_path = directory / "analysis-v1.json"
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid", "publication requires the independent mechanics-valid audit")
    require(sha(directory / "analysis-v1-source.py") == audit["auditor_source_sha256"], "auditor source snapshot differs")
    require(all(audit["source"][key] == run["source"][key] for key in ("job_id", "archive_sha256", "request_sha256", "transaction")), "audit addresses another source execution")
    for ref in audit["references"].values():
        require(sha(helper["file_path"](directory, ref["path"])) == ref["sha256"], "audit reference bytes differ")
    ids, rows = audit["candidate_ids"], {}
    for name, row in audit["rows"].items():
        rows[name] = {"prompt_id": row["prompt_id"], "arm": row["arm"], "dose": row["dose"],
            "computation_id": row["computation_id"], "prefix_binding": row["prefix_binding"],
            "explicit_conditions": {"model_identity": audit["model_identity"], "execution": audit["execution"],
                "frontier": "observed last-token post-attention residual", "candidate_ids": ids,
                "native_dose_policy": row["payload_policy"], "horizon": "final MLP and fixed selected-vocabulary readout"},
            "candidate_comparison": row["candidate_comparison"],
            "candidate_coverage": {"full_native_greedy_token": row["full_vocab_greedy_token"],
                "global_winner_in_candidates": row["full_vocab_greedy_token"] in ids,
                "scope": "candidate winner accuracy does not certify full-vocabulary winner accuracy"},
            "actual_selected_logits": row["actual_selected_logits"], "forecast_selected_logits": row["forecast_selected_logits"],
            "stage_max_abs": {key: value["max_abs"] for key, value in row["stage_comparison"].items()},
            "dose_effect_vs_native": row["dose_effect_vs_native"],
            "native_wrapper_residuals": {"selected_logits": row["full_native_selected_vs_wrapper"], "MLP_delta": row["full_native_delta_vs_wrapper"]},
            "observed_greedy_text": row["greedy_text"], "observed_generated_token_ids": row["generated_token_ids"],
            "observed_generated_text": row["generated_text"], "observed_continuation": row["continuation"],
            "continuation_check": row["continuation_check"], "typed_transitions": row["typed_transitions"],
            "typed_acts": row["typed_acts"], "trace_fingerprint": row["trace_fingerprint"]}
    previous = root / "run-01"
    job, request = read(previous / "job.json"), read(previous / "request.json")
    log = (previous / "worker.log").read_text()
    worker_path = previous / "payload/qwen_mlp_forecast_campaign.py"
    require(job["state"] == "failed" and "worker configuration differs from sealed request" in log,
            "prior failed-attempt evidence differs")
    require(sha(worker_path) == request["source_hashes"]["qwen_mlp_forecast_campaign.py"], "prior worker source differs")
    code = worker_path.read_text()
    require(code.index('raise ValueError("worker configuration differs from sealed request")') < code.index("    import numpy as np"),
            "prior refusal no longer precedes scientific imports")
    prior = {"job_id": job["job_id"], "scheduler_state": job["state"], "status": "pre-model worker configuration refusal",
        "reason": "worker configuration differs from sealed request", "retry_scope": "new explicit repaired submission; failed attempt retained",
        "model_calls": 0, "counter_authority": "source-bound refusal before worker scientific imports/model load; no native work ledger exists",
        "references": {name: reference(path, root) for name, path in {"job": previous / "job.json", "request": previous / "request.json",
                           "worker_log": previous / "worker.log", "worker_source": worker_path}.items()}}
    outside = [name for name, row in rows.items() if not row["candidate_coverage"]["global_winner_in_candidates"]]
    return {"schema": "mdb-qwen-mlp-forecast-publication-v1", "artifact_reference_base": "supplied campaign root",
        "source": audit["source"], "model_identity": audit["model_identity"], "installed_model_sources": audit["installed_model_sources"],
        "checkpoint_files": audit["checkpoint_files"], "native_weight_binding": audit["native_weight_binding"],
        "mechanics_status": "valid", "candidate_ids": ids, "candidate_literal_labels": audit["candidate_literal_labels"],
        "candidate_text_tokenization": audit["candidate_text_tokenization"], "prefixes": audit["prefixes"], "rows": rows,
        "counts": audit["counts"], "scientific_summary": {**audit["scientific_summary"],
            "global_winners_outside_candidates": len(outside), "global_winner_outside_rows": outside,
            "global_winners_inside_candidates": len(rows) - len(outside)},
        "pre_release_counters": audit["pre_release_counters"], "previous_attempt": prior,
        "transaction_work": run["transaction_report"]["work"], "transaction_phase_receipts": run["events"],
        "references": {"audit": reference(audit_path, root), "auditor_source": reference(directory / "analysis-v1-source.py", root),
                       **{key: {**value, "path": "run-02/" + value["path"]} for key, value in audit["references"].items()}},
        "publisher_source_sha256": sha(Path(__file__)), "new_model_calls": 0, "equation_evaluations": 0,
        "trend_status": audit["trend_status"], "terminal_status": "unassessed", "limits": audit["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    value = publish(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(value["rows"]), "outside_candidates": value["scientific_summary"]["global_winners_outside_candidates"], "model_calls": 0}))


if __name__ == "__main__":
    main()

"""Portable index of a saved full-vocabulary Qwen first-token assay."""
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


def publish(root, *, run_name="run-02"):
    root = Path(root).resolve()
    require(Path(run_name).name == run_name, "run name must be one direct child")
    directory = root / run_name
    helpers = runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))
    run = helpers["source_run"](directory, "qwen-mlp-result")
    audit_path, audit_source = directory / "analysis-v1.json", directory / "analysis-v1-source.py"
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid", "publication requires a mechanics-valid independent full audit")
    require(sha(audit_source) == audit["auditor_source_sha256"], "full auditor source snapshot differs")
    require(all(audit["source"][name] == run["source"][name] for name in
                ("job_id", "request_sha256", "archive_sha256", "transaction")), "full audit addresses another source run")
    for ref in audit["references"].values():
        require(sha(helpers["file_path"](directory, ref["path"])) == ref["sha256"], "full audit source reference differs")
    failures = []
    for previous in sorted(root.glob("run-*")):
        if previous == directory or not (previous / "job.json").is_file():
            continue
        job = read(previous / "job.json")
        if job["state"] != "failed":
            continue
        log_path = previous / "worker.log"
        log = log_path.read_text() if log_path.is_file() else ""
        refusals = [line for line in log.splitlines() if line.startswith(("ValueError:", "RuntimeError:"))]
        failures.append({"run": previous.name, "job_id": job["job_id"], "scheduler_state": "failed",
            "failure": refusals[-1] if refusals else "see retained worker/job evidence", "evidence_role": "preserved instrument attempt, not a scientific outcome",
            "references": {name: reference(path, root) for name, path in {"job": previous / "job.json",
                "worker_log": log_path, "request": previous / "request.json"}.items() if path.is_file()}})
    refs = {"audit": reference(audit_path, root), "auditor_source": reference(audit_source, root),
            "request": reference(directory / "request.json", root), "collector_report": reference(directory / "report.json", root),
            "summary": reference(run["data"] / "summary.json", root), "transaction_report": reference(run["data"] / "transaction-report.json", root)}
    refs.update({name: {**value, "path": run_name + "/" + value["path"]} for name, value in audit["references"].items()})
    return {"schema": "mdb-qwen-full-readout-publication-v1", "artifact_reference_base": "supplied campaign root",
        "run": run_name, "source": audit["source"], "references": refs, "model_identity": audit["model_identity"],
        "execution": audit["execution"], "checkpoint_files": audit["checkpoint_files"],
        "installed_model_sources": audit["installed_model_sources"], "native_weight_binding": audit["native_weight_binding"],
        "mechanics_status": audit["mechanics_status"], "scientific_summary": audit["scientific_summary"],
        "counts": audit["counts"], "pre_release_counters": audit["pre_release_counters"],
        "transaction_work": run["transaction_report"]["work"], "transaction_phase_receipts": run["events"],
        "full_head_binding": audit["full_head_binding"], "full_head_checks": audit["full_head_checks"],
        "prefixes": audit["prefixes"], "envelopes": audit["envelopes"], "rows": audit["rows"],
        "previous_attempts": failures, "new_model_calls": 0, "equation_module_evaluations": 0,
        "publisher_source_sha256": sha(Path(__file__)), "terminal_status": "unassessed",
        "forecast_horizon": "full-vocabulary first token from a genuine observed prefix; cached future text is observed only",
        "limits": audit["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", default="run-02")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    result = publish(args.root, run_name=args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rows": len(result["rows"]), "model_calls": 0, "output": str(args.output)}))


if __name__ == "__main__":
    main()

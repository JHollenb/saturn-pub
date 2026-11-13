"""Portable index of saved Klein numerical forecasts and native RGB outcomes."""
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


def publish(root, *, run_name="run-01"):
    root = Path(root).resolve()
    require(Path(run_name).name == run_name, "run must be one direct campaign child")
    directory = root / run_name
    helper_path = Path(__file__).with_name("summarize_context_window.py")
    h = runpy.run_path(str(helper_path))
    run = h["source_run"](directory, "flux-mlp-result", max_archive_bytes=512 << 20, max_member_bytes=256 << 20)
    audit_path = directory / "analysis-v1.json"
    snapshot = directory / "analysis-v1-source.py"
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid", "publication requires a mechanics-valid saved-array audit")
    require(sha(snapshot) == audit["auditor_source_sha256"]
            and audit["audit_sources"][helper_path.name] == sha(helper_path), "auditor/custody helper source binding differs")
    require(all(audit["source"][key] == run["source"][key] for key in
                ("job_id", "request_sha256", "archive_sha256", "transaction")), "audit addresses another raw run")
    refs = {}
    for name, ref in audit["references"].items():
        require(sha(h["file_path"](directory, ref["path"])) == ref["sha256"], "audit reference changed: " + name)
        refs[name] = {**ref, "path": run_name + "/" + ref["path"]}
    for name, path in {"audit": audit_path, "auditor_source": snapshot,
                       "request": directory / "request.json", "collector_report": directory / "report.json"}.items():
        refs[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}
    for row in audit["rows"].values():
        image = row["RGB"]
        path = h["file_path"](run["data"], image["path"])
        require(sha(path) == image["png_sha256"], "native image changed after audit")
        row["RGB"] = {**image, "path": path.relative_to(root).as_posix()}
    previous_attempts = []
    for path in sorted(root.glob("run-*/job.json")):
        job = read(path)
        if path.parent != directory and job.get("state") == "failed":
            previous_attempts.append({"run": path.parent.name, "job_id": job["job_id"], "status": "failed",
                "evidence_role": "preserved instrument attempt; no scientific outcome inferred",
                "job": {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}})
    return {"schema": "mdb-flux-mlp-forecast-publication-v1", "artifact_reference_base": "supplied campaign root",
        "source": audit["source"], "references": refs, "model_identity": audit["model_identity"],
        "native_execution": audit["native_execution"], "implementation_sources": audit["implementation_sources"],
        "mechanics_status": audit["mechanics_status"], "counts": audit["counts"],
        "pre_release_counters": audit["pre_release_counters"], "external_weights": audit["external_weights"],
        "weight_checks": audit["weight_checks"], "prefixes": audit["prefixes"], "cross_scene": audit["cross_scene"],
        "error_contrast_scales": audit["error_contrast_scales"],
        "numerical_forecasts": audit["numerical_forecasts"], "rows": audit["rows"],
        "semantic_hypothesis": audit["semantic_hypothesis"], "semantic_status": audit["semantic_status"],
        "RGB_forecast": False, "object_shape_assessment": "separate assistant visual inspection; no human annotation or calibrated cube/sphere verdict",
        "transaction_work": run["transaction_report"]["work"], "transaction_phase_receipts": run["events"],
        "previous_attempts": previous_attempts, "new_model_calls": 0, "equation_module_evaluations": 0,
        "publisher_source_sha256": sha(Path(__file__)), "terminal_status": "unassessed", "limits": audit["limits"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", default="run-01")
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

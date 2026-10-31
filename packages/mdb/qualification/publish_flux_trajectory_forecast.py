"""Portable saved-evidence index for one conditional Klein trajectory forecast."""
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


def publish(root, *, run_name="run-01", failed=False):
    root = Path(root).resolve()
    require(Path(run_name).name == run_name, "run must be one direct campaign child")
    directory = root / run_name
    helper = Path(__file__).with_name("summarize_context_window.py")
    h = runpy.run_path(str(helper))
    run = h["source_run"](directory, "flux-trajectory-result", failed=failed, recover_metadata=failed,
                          max_archive_bytes=128 << 20, max_member_bytes=64 << 20)
    audit_path, snapshot = directory / "analysis-v1.json", directory / "analysis-v1-source.py"
    audit = read(audit_path)
    require(audit["mechanics_status"] == "valid" and sha(snapshot) == audit["auditor_source_sha256"], "independent audit/source binding differs")
    for name, expected in audit["audit_sources"].items():
        require(Path(name).name == name and sha(Path(__file__).with_name(name)) == expected, "audit helper source differs")
    require(all(audit["source"][key] == run["source"][key] for key in ("job_id", "request_sha256", "archive_sha256", "transaction")), "audit addresses another run")
    refs = {}
    for name, ref in audit["references"].items():
        require(sha(h["file_path"](directory, ref["path"])) == ref["sha256"], "audit evidence reference changed")
        refs[name] = {**ref, "path": run_name + "/" + ref["path"]}
    for name, path in {"audit": audit_path, "auditor_source": snapshot, "request": directory / "request.json",
                       "collector_report": directory / "report.json",
                       **({} if failed else {"transaction_report": run["data"] / "transaction-report.json"})}.items():
        refs[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}
    require(audit["scheduler_status"] == ("failed" if failed else "succeeded"), "publication attempt role differs from audited scheduler status")
    if failed:
        recovery_path = h["file_path"](directory, audit["recovery"]["receipt_path"])
        receipt = read(recovery_path)
        require(receipt["source"] == audit["source"] and receipt["scheduler_status"] == "failed"
                and receipt["raw_files_modified"] is False and receipt["new_model_calls"] == 0, "offline recovery provenance differs")
        for ref in receipt["generated"].values():
            require(sha(h["file_path"](directory, ref["path"])) == ref["sha256"], "recovered output changed")
        refs["recovery_receipt"] = {"path": recovery_path.relative_to(root).as_posix(), "sha256": sha(recovery_path)}
    for image in audit["images"].values():
        require(sha(h["file_path"](directory, image["path"])) == image["png_sha256"], "forecast/native image changed")
        image["path"] = run_name + "/" + image["path"]
    previous = []
    for path in sorted(root.glob("run-*/job.json")):
        job = read(path)
        if path.parent != directory and job.get("state") == "failed":
            previous.append({"run": path.parent.name, "job_id": job["job_id"], "status": "failed",
                "role": "preserved instrument attempt; no scientific result inferred",
                "job_reference": {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}})
    for path in sorted(root.glob("run-*/admission-failure.json")):
        refusal = read(path)
        require(refusal["job_created"] is False and refusal["model_calls"] == 0, "admission-only provenance has executed work")
        previous.append({"run": path.parent.name, "status": "admission-refused", "job_created": False, "model_calls": 0,
            "role": "no scheduler job or scientific worker executed", "admission": refusal,
            "admission_reference": {"path": path.relative_to(root).as_posix(), "sha256": sha(path)}})
    arrays = [item for boundary in audit["boundaries"] for item in boundary["arrays"].values()] + list(audit["vae"].values())
    full = [item["full_recomputed"] for item in arrays if "full_recomputed" in item]
    result = {**audit, "schema": "mdb-flux-trajectory-forecast-publication-v1", "artifact_reference_base": "supplied campaign root",
        "run": run_name, "references": refs, "previous_attempts": previous,
        "denominator": {"observed_initial_frontiers": 1, "own_state_trajectory_pairs": 1,
                        "ordered_transition_pairs": len(audit["boundaries"]), "RGB_pairs": 1,
                        "scope": "one prompt and one fresh seed; boundaries are repeated observations of one trajectory"},
        "numerical_summary": {"witness_arrays": len(arrays), "witness_exact_arrays": sum(item["witness"]["exact"] for item in arrays),
            "retained_full_arrays": len(full), "retained_full_exact_arrays": sum(item["exact"] for item in full),
            "max_recomputed_witness_abs_error": max(item["witness"]["max_abs"] for item in arrays),
            "max_recomputed_full_abs_error": max(item["max_abs"] for item in full),
            "max_worker_RAM_reported_full_abs_error": max(item["worker_full_error"]["max_abs"] for item in arrays),
            "pixel_error": audit["pixel_error"]},
        "transaction_work": run["transaction_report"]["work"], "transaction_phase_receipts": run["events"],
        "publisher_source_sha256": sha(Path(__file__)), "new_model_calls": 0, "equation_module_evaluations": 0}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", default="run-01")
    parser.add_argument("--failed", action="store_true", help="publish independently audited observations from a failed publication attempt")
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    result = publish(args.root, run_name=args.run, failed=args.failed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"boundaries": len(result["boundaries"]), "pixel_exact": result["pixel_error"]["exact"], "model_calls": 0}))


if __name__ == "__main__":
    main()

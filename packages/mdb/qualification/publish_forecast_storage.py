"""Publish saved storage profiles and known-case reproduction; no model/equations."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

from mdb import forecast_transaction
from mdb.forecast_transaction import verify_transaction


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ref(path):
    parts = path.resolve().parts
    return {"path": "/".join(parts[parts.index("artifacts"):]) if "artifacts" in parts else path.name,
            "sha256": sha(path.read_bytes())}


def verify_archive(root, name):
    request, report, job = (read(root / file) for file in ("request.json", "report.json", "job.json"))
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    require(sha(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()) == request["request_sha256"]
            == report["request_sha256"], "request/report binding differs")
    require(job["state"] == "succeeded" and job["job_id"] == report["job_id"], "job did not successfully finish")
    for relative, digest in request["source_hashes"].items():
        path = root / "payload" / relative
        require(not path.is_symlink() and sha(path.read_bytes()) == digest, "staged source differs")
    marker = "MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 "
    archives = [line[len(marker):] for line in (root / "worker.log").read_text().splitlines() if line.startswith(marker)]
    require(len(archives) == 1, "result must retain one archive")
    raw = base64.b64decode(archives[0], validate=True)
    require(len(raw) <= 64 * 1024 * 1024 and sha(raw) == report["archive_sha256"], "archive identity differs")
    data, hashes, total = root / "extracted" / name, {}, 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            require(member.isfile() and parts[0] == name and not Path(member.name).is_absolute()
                    and ".." not in parts and member.size <= 64 * 1024 * 1024, "unsafe archive member")
            relative = Path(*parts[1:]).as_posix()
            total += member.size
            require(relative not in hashes and total <= 256 * 1024 * 1024, "duplicate/oversized archive")
            hashes[relative] = sha(archive.extractfile(member).read())
            path = data / relative
            require(not path.is_symlink() and path.resolve().is_relative_to(data.resolve())
                    and sha(path.read_bytes()) == hashes[relative], "extracted bytes differ from archive")
    require({path.relative_to(data).as_posix() for path in data.rglob("*") if path.is_file()} == set(hashes),
            "extracted inventory differs from archive")
    return {"job_id": job["job_id"], "request": ref(root / "request.json"), "report": ref(root / "report.json"),
            "archive_sha256": sha(raw), "all_extracted_files_archive_equal": len(hashes),
            "verified_file_hash_map_sha256": sha(canonical(hashes)), "expanded_bytes": total,
            "verified_staged_sources": len(request["source_hashes"]), "job_peak_rss_mb": job["result"].get("peak_rss_mb"),
            "job_reported_elapsed_seconds": job["result"].get("elapsed_s"),
            "job_started_to_finished_seconds": job["finished_ts"] - job["started_ts"]}


def publish(root, output, benchmark_source, reference_run):
    require(not output.exists(), "portable output already exists")
    helper = runpy.run_path(str(benchmark_source.resolve()))
    benchmark, reproduction = root / "benchmark-03", root / "reproduction-01"
    sources = {"benchmark": verify_archive(benchmark, "storage-benchmark-result"),
               "reproduction": verify_archive(reproduction, "context-window-result"),
               "reference": verify_archive(reference_run, "context-window-result")}
    data = benchmark / "extracted/storage-benchmark-result"
    summary, request = read(data / "summary.json"), read(benchmark / "request.json")
    require(summary["job_id"] == sources["benchmark"]["job_id"] and summary["request_sha256"] == request["request_sha256"]
            and summary["model_calls"] == summary["equation_evaluations"] == 0, "benchmark summary binding differs")
    require(sha(benchmark_source.read_bytes()) == request["source_hashes"]["forecast_storage_benchmark.py"],
            "explicit benchmark decoder source differs from the sealed worker")
    modes, first_leaves, first_npy = {}, None, None
    for mode in summary["order"]:
        directory = data / mode
        measurement, profile = read(directory / "summary.json"), read(directory / "profile.json")
        require(measurement in summary["modes"], "aggregate benchmark measurement differs")
        report = read(directory / "transaction-report.json")
        verification = verify_transaction(directory / "transaction", expected_seal_sha256=report["seal_sha256"],
                                          expected_event_head=report["event_head"])
        workload = helper["read_workload"](forecast_transaction, directory / "transaction")
        leaves = helper["leaf_manifest"](helper["logical_tree"](workload))
        npy = helper["encoded_array_identities"](directory / "transaction")
        require(leaves == read(directory / "expected-leaves.json") and npy == read(directory / "expected-npy-identities.json"),
                "saved benchmark values/original NPY identities differ")
        require(first_leaves is None or leaves == first_leaves, "benchmark decoded values differ across modes")
        require(first_npy is None or npy == first_npy, "original NPY identities differ across modes")
        first_leaves, first_npy = leaves, npy
        require(report["work"]["native_calls"] == measurement["model_calls"] == measurement["equation_evaluations"] == 0,
                "saved-array benchmark is not model/equation-free")
        files = [path for path in (directory / "transaction").rglob("*") if path.is_file()]
        require(len(files) == measurement["transaction_files"] and sum(path.stat().st_size for path in files)
                == measurement["transaction_bytes"], "physical storage counts differ")
        modes[mode] = {key: measurement[key] for key in ("phases", "fsync", "workload", "max_rss_bytes", "max_rss_scope",
            "transaction_files", "transaction_bytes", "runtime_source_sha256", "current_public_storage_source_sha256")}
        modes[mode].update(verification=verification, decoded_nodes=len(leaves), original_npy_occurrences=len(npy),
            decoded_manifest_sha256=sha(canonical(leaves)), npy_manifest_sha256=sha(canonical(npy)),
            profile={key: profile[key] for key in ("grouped_self_seconds", "total_profile_seconds", "total_calls", "classification_scope")},
            profile_top_functions=profile["top_functions"][:8], complete_profile=ref(directory / "profile.json"))
    comparison_path, audit_path = reproduction / "comparison-v1.json", reproduction / "analysis-v1.json"
    comparison, audit, proof = read(comparison_path), read(audit_path), read(root / "package-proof.json")
    require(comparison["reference"]["source"]["archive_sha256"] == sources["reference"]["archive_sha256"],
            "native performance reference addresses another archive")
    require(comparison["candidate"]["source"]["archive_sha256"] == sources["reproduction"]["archive_sha256"]
            == audit["source"]["archive_sha256"], "reproduction analyses address different archives")
    require(comparison["candidate"]["source"]["job_id"] == audit["source"]["job_id"] == sources["reproduction"]["job_id"],
            "reproduction analysis job bindings differ")
    verify_transaction(reproduction / "extracted/context-window-result/transaction",
        expected_seal_sha256=audit["source"]["transaction"]["seal_sha256"], expected_event_head=audit["source"]["transaction"]["event_head"])
    require(proof["all_source_installed_qualified_benchmark_reproduction_modules_equal"]
            and proof["source_hashes"]["forecast_transaction.py"] == request["source_hashes"]["current_forecast_transaction.py"]
            and proof["source_hashes"]["forecast_storage.py"] == request["source_hashes"]["current_forecast_storage.py"], "package source qualification differs")
    counts = comparison["candidate"]["measurements"]["counts"]
    require(counts == audit["raw_summary"]["counts"] and counts["physical_forwards"] == 7 and counts["prior_rows_reused"] == 3,
            "known-case performance reproduction geometry differs")
    phases = {mode: {item["phase"]: item for item in value["phases"]} for mode, value in modes.items()}
    base, bundle = phases["baseline-files-v1"], phases["current-bundle-v1"]
    ratios = {name: {"baseline_seconds": base[name]["wall_seconds"], "bundle_seconds": bundle[name]["wall_seconds"],
              "bundle_over_baseline": bundle[name]["wall_seconds"] / base[name]["wall_seconds"]} for name in base}
    failed_job = read(root / "benchmark-02/job.json")
    require(failed_job["state"] == "failed" and "Permission denied" in (root / "benchmark-02/worker.log").read_text(), "setup failure evidence differs")
    result = {"schema": "mdb-forecast-storage-publication-v1", "source": sources, "benchmark_modes": modes,
        "benchmark_reader_and_phase_ratios": ratios, "benchmark_order": summary["order"], "benchmark_limits": summary["limits"],
        "known_case_reproduction": {key: comparison[key] for key in ("scope", "logical_comparisons", "native_observations_by_row",
            "original_npy_leaf_identities", "storage_changes", "reference", "candidate", "phase_comparison", "limits")},
        "measured_speed_ratios": {
            "worker_reference_over_reproduction": comparison["reference"]["measurements"]["worker_seconds"] / comparison["candidate"]["measurements"]["worker_seconds"],
            "job_started_to_finished_reference_over_reproduction": sources["reference"]["job_started_to_finished_seconds"] / sources["reproduction"]["job_started_to_finished_seconds"],
            "benchmark_forecast_publication_baseline_over_bundle": base["predict_publish"]["wall_seconds"] / bundle["predict_publish"]["wall_seconds"]},
        "native_audit": {"source": ref(audit_path), "comparison": ref(comparison_path), "aggregate": audit["aggregate"],
            "mechanics_status": audit["mechanics_status"], "instrument_status": audit["instrument_status"],
            "row_native_replay_status": {name: value["native_replay_heads"] for name, value in audit["rows"].items()}},
        "package_qualification": {"source": ref(root / "package-proof.json"), **{key: proof[key] for key in
            ("module_count", "wheels", "unique_focused_tests", "tests", "qualification_scope")}},
        "preserved_setup_failures": {"benchmark-01": {"stage": "pre-submit setup", "job_submitted": False,
            "files": [ref(path) for path in sorted((root / "benchmark-01").iterdir()) if path.is_file()]},
            "benchmark-02": {"job_id": failed_job["job_id"], "stage": "worker dependency-cache initialization",
                "workload_executed": False, "log": ref(root / "benchmark-02/worker.log"), "reason": "Host-inapplicable cache path; permission denied"}},
        "publisher_source_sha256": sha(Path(__file__).read_bytes()), "benchmark_helper": ref(benchmark_source),
        "new_model_calls_during_publication": 0, "equation_evaluations_during_publication": 0, "terminal_status": "unassessed",
        "limits": ["Seven native calls repeat known cases and three controls are reused; no new scientific specimens.",
            "Fewer durable publications trade against full-bundle verification cost for isolated reads; all measured slowdowns retained.",
            "cProfile groups are function self times, not independent additive I/O phases; no cold-disk or universal speed claim."]}
    raw = canonical(result) + b"\n"
    require(len(raw) < 100 * 1024, "publication exceeds 100KiB")
    with output.open("xb") as stream:
        stream.write(raw)
    return {"output": str(output), "sha256": sha(raw), "bytes": len(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--benchmark-source", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root.resolve(), args.output.resolve(), args.benchmark_source.resolve(), args.reference_run.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()

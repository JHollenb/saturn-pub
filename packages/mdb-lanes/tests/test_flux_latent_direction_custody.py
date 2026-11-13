"""Small, model-free controls for latent-direction failed-result custody."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"


def _module():
    return runpy.run_path(str(HERE / "flux_latent_direction_custody.py"))


def _request(mod):
    config = {
        "conditions": ["c0-o1", "c1-o0"],
        "counts": {"acts": 12, "adapter_steps": 504, "images": 12, "receipt_emissions": 516},
        "expected_parameter_artifacts_module_sha256": "pending",
    }
    request = {"schema": "mdb-test-request-v1", "config": config,
               "source_hashes": {"worker.py": "a" * 64}, "runtime_wheels": {"wheel.whl": "b" * 64}}
    request["request_sha256"] = hashlib.sha256(
        json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return request


def _fixture(tmp_path):
    mod = _module()
    request = _request(mod)
    job_id = "job-test-native"
    data = tmp_path / "extracted" / mod["RESULT"]
    data.mkdir(parents=True)
    for name in ("c0-o1.d-c0-o1", "c0-o1.d-c1-o0", "c0-o1.sham",
                 "c1-o0.d-c0-o1", "c1-o0.d-c1-o0", "c1-o0.sham"):
        (data / "native").mkdir(exist_ok=True)
        (data / "native" / (name + ".jsonl")).write_text("{}\n" * 84)
        (data / "rows").mkdir(exist_ok=True)
        diagnostics = [{}] * 84 if name.endswith(".sham") else []
        (data / "rows" / (name + ".json")).write_text(json.dumps({"name": name,
            "act": {"name": name}, "native_replay_diagnostics": diagnostics}))
        (data / (name + ".predicted.png")).write_bytes(b"predicted")
        (data / (name + ".native.png")).write_bytes(b"native")
    (data / "receipt-emissions.jsonl").write_text("{}\n" * 516)
    (data / "transactions" / "c0-o1").mkdir(parents=True)
    (data / "transactions" / "c1-o0").mkdir(parents=True)
    for cell in mod["CELLS"]:
        (data / "transactions" / cell / "verification.json").write_text(json.dumps({
            "phase": "closed", "verified": True, "events": 8}))
    (data / "failure.json").write_text(json.dumps({"error": "'finalize_manifest'",
        "error_type": "KeyError", "scope": "Failed finite attempt; no retry or numeric result rewriting."}))
    (data / "parameter-artifacts.json").write_text(json.dumps({"artifacts": [
        {"path": f"weights/{index}.safetensors"} for index in range(169)]}))
    (data / "retrieve_weights.py").write_text("# qualified retriever\n")
    request["config"]["expected_parameter_artifacts_module_sha256"] = mod["sha"](data / "retrieve_weights.py")
    request["request_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True,
                   allow_nan=False).encode()).hexdigest()
    (data / "plan.json").write_text(json.dumps(request["config"]))
    (data / "source.json").write_text(json.dumps({"job_id": job_id,
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
        "source_integrity": {"unchanged": True}}))
    (data / "summary.json").write_text(json.dumps({"job_id": job_id,
        "request_sha256": request["request_sha256"], "counts": request["config"]["counts"]}))
    collection = {"schema": mod["SCHEMA"], "job_id": job_id,
        "request_sha256": request["request_sha256"], "failure_kind": mod["KNOWN_FAILURE"]}
    return mod, data, request, collection, job_id


def test_recovery_manifest_accepts_complete_failed_result(tmp_path):
    mod, data, request, collection, job_id = _fixture(tmp_path)
    actual = mod["inventory"](data)
    verified = mod["verify_manifest"](data, actual, job_id, request, collection=collection)
    assert verified["manifest_origin"] == "collector-derived-source-manifest"
    assert verified["summary"]["counts"]["adapter_steps"] == 504


def test_recovery_rejects_wrong_failure(tmp_path):
    mod, data, request, collection, job_id = _fixture(tmp_path)
    (data / "failure.json").write_text(json.dumps({"error": "other", "error_type": "RuntimeError",
        "scope": "Failed finite attempt; no retry or numeric result rewriting."}))
    with pytest.raises(ValueError, match="known post-exec"):
        mod["verify_manifest"](data, mod["inventory"](data), job_id, request, collection=collection)


def test_recovery_rejects_incomplete_census(tmp_path):
    mod, data, request, collection, job_id = _fixture(tmp_path)
    (data / "native" / "c1-o0.sham.jsonl").unlink()
    with pytest.raises(ValueError, match="native branch ledger"):
        mod["verify_manifest"](data, mod["inventory"](data), job_id, request, collection=collection)


def test_recovery_rejects_source_drift(tmp_path):
    mod, data, request, collection, job_id = _fixture(tmp_path)
    source = json.loads((data / "source.json").read_text())
    source["source_hashes"]["worker.py"] = "c" * 64
    (data / "source.json").write_text(json.dumps(source))
    with pytest.raises(ValueError, match="source/plan/summary"):
        mod["verify_manifest"](data, mod["inventory"](data), job_id, request, collection=collection)


def test_failed_job_requires_known_recovery_descriptor(tmp_path):
    mod, _, request, collection, job_id = _fixture(tmp_path)
    job = {"job_id": job_id, "state": "failed", "assigned_host": "beast",
        "config": request, "submitted_request": {"config": request},
        "payload_custody": {"declared": {"sha256": "d" * 64, "size_bytes": 1},
            "sealed": {"sha256": "d" * 64, "size_bytes": 1},
            "executed": {"sha256": "d" * 64, "size_bytes": 1}},
        "result": {"status": "failed", "returncode": 1,
            "executed_payload": {"sha256": "d" * 64, "size_bytes": 1, "host": "beast"},
            "diagnostics": {"cwd": "/mnt/hdd8tb/mrun-work/" + job_id},
            "failure": {"kind": "process_exit", "phase": "process.finalize", "returncode": 1}}}
    with pytest.raises(ValueError, match="known finalizer"):
        mod["validate_job"](job, job_id, request, collection={**collection, "failure_kind": "wrong"})

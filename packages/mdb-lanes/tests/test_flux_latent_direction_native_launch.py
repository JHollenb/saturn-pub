"""Saved CPU-capsule/manifest and native bootstrap checks; no service calls."""
from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
L = runpy.run_path(str(HERE / "qualification/submit_flux_latent_direction_native.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def request(value):
    value = copy.deepcopy(value)
    value["request_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return value


def cpu_fixture(tmp_path, monkeypatch, *, native=0, worker_sha="1" * 64):
    # Generic actual-job/payload checks are the unchanged qualified collector.
    # This fixture isolates the new CPU source/capsule/cohort join, not a job.
    monkeypatch.setitem(L["C"], "validate_job", lambda job, *_: "/mnt/hdd8tb/mrun-work/" + job["job_id"] + "/flux-prompt-factorial-result")
    monkeypatch.setitem(L["P"]["M"], "validate_accepted_audit", lambda *_: ("job-audit", "a" * 64))
    parent, math_run, audit = (tmp_path / name for name in ("parent", "math", "audit"))
    pr = request({"schema": "synthetic-parent"})
    write(parent / "request.json", pr)
    write(parent / "collection-job.json", {"job_id": "job-parent", "result": {"diagnostics": {"command": ["sealed"]}}})
    write(parent / "admission-command.json", {"command": ["sealed"]})
    write(parent / "collection.json", {"job_id": "job-parent", "request_sha256": pr["request_sha256"], "bindings": {}})
    mr = request({"schema": "mdb-latent-direction-math-request-v1", "config": {"schema": "mdb-latent-direction-math-plan-v1"},
        "source_hashes": {"flux_latent_direction_cross.py": "1" * 64}})
    write(math_run / "request.json", mr)
    result = {"schema": "mdb-latent-direction-math-execution-v1", "output_directory": "latent-direction-math-result",
        "report": {"source_job_id": "job-math", "parent_job_id": "job-parent", "math_transitions": 504,
            "native_forwards": native, "cohort_seal_sha256": "b" * 64, "inputs_sha256": "c" * 64},
        "near_data_execution": {"job_id": "job-math", "request_sha256": mr["request_sha256"], "worker_sha256": worker_sha,
            "parent_job_id": "job-parent", "parent_request_sha256": pr["request_sha256"], "mathematical_transitions": 504,
            "native_framework_forwards": native, "parent_inventory_unchanged": True}}
    write(math_run / "result.json", result)
    raw = (math_run / "result.json").read_bytes()
    zipped = gzip.compress(raw, mtime=0)
    capsule = {"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_bytes": len(zipped), "gzip_sha256": hashlib.sha256(zipped).hexdigest()}
    log = ("MDB_FACTORIAL_AUDIT_CAPSULE=" + json.dumps({**capsule, "base64_gzip": base64.b64encode(zipped).decode()}) + "\n").encode()
    (math_run / "worker.log").write_bytes(log)
    write(math_run / "job.json", {"job_id": "job-math", "result": {"diagnostics": {"command": ["sealed"]},
        "artifacts": {"job_log": {"sha256": hashlib.sha256(log).hexdigest(), "bytes": len(log)}}}})
    write(math_run / "admission-command.json", {"command": ["sealed"]})
    for name in ("collection-local-inventory.json", "collection-remote-before.json", "collection-remote-after.json"):
        write(math_run / name, {"files": {}, "file_count": 0, "expanded_bytes": 0})
    write(math_run / "collection.json", {"schema": "mdb-latent-direction-math-collection-v1", "job_id": "job-math", "request_sha256": mr["request_sha256"],
        "capsule": capsule, "remote_path": "/mnt/hdd8tb/mrun-work/job-math/latent-direction-math-result", "file_count": 0, "expanded_bytes": 0,
        "bindings": {name: L["sha"](math_run / name) for name in ("request.json", "job.json", "result.json", "worker.log", "admission-command.json")}})
    return parent, math_run, audit


def test_actual_cpu_capsule_and_original_parent_join(tmp_path, monkeypatch):
    parent, math_run, audit = cpu_fixture(tmp_path, monkeypatch)
    identity = L["validate_sources"](parent, math_run, audit=audit)
    assert identity["parent_job_id"] == "job-parent" and identity["math_job_id"] == "job-math"
    assert identity["math_result_sha256"] == L["sha"](math_run / "result.json")


@pytest.mark.parametrize("options", [{"native": 1}, {"worker_sha": "2" * 64}])
def test_changed_cpu_work_or_executed_source_refused(tmp_path, monkeypatch, options):
    parent, math_run, audit = cpu_fixture(tmp_path, monkeypatch, **options)
    with pytest.raises(ValueError, match="CPU sealed-math"):
        L["validate_sources"](parent, math_run, audit=audit)


def test_actual_cpu_log_changed_after_collection_refused(tmp_path, monkeypatch):
    parent, math_run, audit = cpu_fixture(tmp_path, monkeypatch)
    with (math_run / "worker.log").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="collection/job"):
        L["validate_sources"](parent, math_run, audit=audit)


def test_cpu_config_without_expected_module_fields_uses_pinned_retriever(tmp_path):
    data = tmp_path / "math"
    data.mkdir()
    stream = io.BytesIO()
    np.save(stream, np.array([0], dtype=np.uint16))
    raw = stream.getvalue()
    prefix, payload = raw[:-2], raw[-2:]
    artifact = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "npy_prefix_b64": base64.b64encode(prefix).decode(),
        "source": {"file": "transformer/model.safetensors", "file_sha256": "f" * 64, "tensor": "weight", "dtype": "BF16",
            "shape": [1], "data_offsets": [0, 2], "payload_sha256": hashlib.sha256(payload).hexdigest()}}
    entries = [{**artifact, "path": "absent-weights/" + str(i) + ".npy"} for i in range(169)]
    write(data / "parameter-artifacts.json", {"schema": "mdb-parameter-artifacts-v1", "artifacts": entries})
    (data / "retrieve_weights.py").write_bytes((HERE / "src/mdb_lanes/parameter_artifacts.py").read_bytes())
    r = {"request_sha256": "r", "source_hashes": {"worker.py": "s"}, "config": {"schema": "mdb-latent-direction-math-plan-v1"}}
    write(data / "plan.json", r["config"])
    write(data / "source.json", {"job_id": "job-math", "request_sha256": "r", "source_hashes": r["source_hashes"]})
    actual = L["C"]["inventory"](data)
    write(data / "remote-manifest.json", {"schema": "mdb-latent-direction-math-manifest-v1", "job_id": "job-math", "request_sha256": "r", "files": actual["files"]})
    actual = L["C"]["inventory"](data)
    assert L["verify_math_manifest"](data, actual, "job-math", r)["job_id"] == "job-math"
    (data / "retrieve_weights.py").write_text("changed")
    with pytest.raises(ValueError, match="manifest|recipes"):
        L["verify_math_manifest"](data, actual, "job-math", r)


def test_native_bootstrap_and_bounded_logical_contract():
    intent = L["B"]["execution_intent"]()
    command = L["command"]({"execution_environment": intent})
    assert command[-2:] == ["mdb_lanes.worker_bootstrap", "flux_latent_direction_native.py"]
    assert "CUDA_VISIBLE_DEVICES=" not in command
    contract = L["work_contract"]()
    assert contract["logical_branches"] == 6 and contract["native_adapter_transitions"] == 504
    assert contract["receipt_occurrences"] == 516 and contract["max_local_work_items"] <= 256


def test_staged_import_closure_catches_missing_transitive_collector(tmp_path):
    source = Path(L["HERE"])
    (tmp_path / "collect_flux_prompt_factorial.py").write_bytes((source / "collect_flux_prompt_factorial.py").read_bytes())
    with pytest.raises(FileNotFoundError, match="collect_flux_initial_noise"):
        L["validate_payload_helpers"](tmp_path)

"""Reject mismatched historical evidence before admitting a new mathematical pass."""
import base64
import gzip
import hashlib
import json
import runpy
from pathlib import Path

import pytest

L = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/submit_flux_matched_pass.py"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def historical_evidence(tmp_path, monkeypatch, *, match=True, parent_id="parent-request"):
    # Payload-envelope authentication is the existing collector's separately
    # qualified responsibility; these cases exercise the new report joins.
    monkeypatch.setitem(L["C"], "validate_job", lambda *args: "unused-result-path")
    parent, audit = tmp_path / "parent", tmp_path / "audit"
    write(parent / "request.json", {"request_sha256": "parent-request",
        "independent_math_sources": {"analyze_flux_prompt_factorial.py": "frozen-equations"}})
    report = {"mechanics_status": "valid", "auditor_source_sha256": "frozen-equations",
        "near_data_execution": {"audit_job_id": "job-test-audit", "parent_request_sha256": parent_id,
            "native_framework_forwards": 0}, "numeric": {"case": {"complete_field_hashes_match": match}}}
    raw = (json.dumps(report, sort_keys=True) + "\n").encode()
    compressed = gzip.compress(raw, mtime=0)
    capsule = {"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "gzip_bytes": len(compressed), "gzip_sha256": hashlib.sha256(compressed).hexdigest()}
    encoded = {**capsule, "base64_gzip": base64.b64encode(compressed).decode()}
    log = (L["L"]["MARKER"] + json.dumps(encoded) + "\n").encode()
    audit.mkdir()
    (audit / "audit.json").write_bytes(raw)
    (audit / "worker.log").write_bytes(log)
    write(audit / "request.json", {"request_sha256": "audit-request"})
    write(audit / "job.json", {"job_id": "job-test-audit", "state": "succeeded", "result": {
        "diagnostics": {"command": ["sealed-command"]}, "artifacts": {"job_log": {
            "bytes": len(log), "sha256": hashlib.sha256(log).hexdigest()}}}})
    write(audit / "admission-command.json", {"command": ["sealed-command"]})
    names = ("audit.json", "worker.log", "request.json", "job.json", "admission-command.json")
    write(audit / "collection.json", {"job_id": "job-test-audit", "request_sha256": "audit-request",
        "capsule": capsule, "bindings": {name: L["sha"](audit / name) for name in names}})
    return parent, audit


def test_source_bound_historical_reconstruction_can_anchor_matched_input_run(tmp_path, monkeypatch):
    parent, audit = historical_evidence(tmp_path, monkeypatch)
    assert L["validate_accepted_audit"](parent, audit) == ("job-test-audit", L["sha"](audit / "audit.json"))


@pytest.mark.parametrize("kwargs", [{"match": False}, {"parent_id": "different-parent"}])
def test_unreconstructed_or_other_parent_report_cannot_anchor_run(tmp_path, monkeypatch, kwargs):
    parent, audit = historical_evidence(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError, match="full reconstruction"):
        L["validate_accepted_audit"](parent, audit)


def test_changed_log_after_collection_cannot_anchor_run(tmp_path, monkeypatch):
    parent, audit = historical_evidence(tmp_path, monkeypatch)
    with (audit / "worker.log").open("ab") as stream:
        stream.write(b"changed after collection\n")
    with pytest.raises(ValueError, match="collection identity"):
        L["validate_accepted_audit"](parent, audit)

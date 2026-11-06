"""Synthetic transport guards only; never invoke collection/network."""
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "collect_compact.py"
C = runpy.run_path(str(SOURCE))


def save(path, value):
    unsigned = {k: v for k, v in value.items() if k != "request_sha256"}
    value["request_sha256"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(value))
    return value


def test_exact_corrected_reader_and_request_before_transport(tmp_path):
    native, audit = tmp_path / "native", tmp_path / "audit"
    native.mkdir(); audit.mkdir()
    n = save(native / "request.json", dict(config={}))
    (native / "submission.json").write_text(json.dumps(dict(job_id="job-bbbb33334444")))
    a = save(audit / "request.json", dict(source_hashes={"audit_qwen_row_sign.py": C["READER_SHA256"]},
        config=dict(native_request_sha256=n["request_sha256"], source_job="job-bbbb33334444")))
    assert C["validate_requests"](audit, native, a["request_sha256"])[1] == "job-bbbb33334444"
    a["source_hashes"]["audit_qwen_row_sign.py"] = "0" * 64
    a = save(audit / "request.json", a)
    with pytest.raises(ValueError, match="reader pin"):
        C["collect"](audit, native, tmp_path / "fresh", expected_request=a["request_sha256"], encoder_source=tmp_path / "absent")
    assert not (tmp_path / "fresh").exists()


def test_tensor_members_and_budgets_refused_before_archive_retrieval():
    record = dict(files={"audit.json": dict(bytes=100, sha256="0" * 64)}, verification=dict(archive_bytes=50))
    C["compact_scope"](record, max_expanded_bytes=1000, max_archive_bytes=1000)
    for name in ("arrays/native.npy", "weights.safetensors", "../audit.json"):
        record["files"] = {name: dict(bytes=100, sha256="0" * 64)}
        with pytest.raises(ValueError, match="Nonmetadata/unsafe"):
            C["compact_scope"](record, max_expanded_bytes=1000, max_archive_bytes=1000)
    record["files"] = {"audit.json": dict(bytes=1001, sha256="0" * 64)}
    with pytest.raises(ValueError, match="expanded budget"):
        C["compact_scope"](record, max_expanded_bytes=1000, max_archive_bytes=1000)


def test_collector_help_has_explicit_corrected_request_and_fresh_destination(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(SOURCE), "--help"], cwd=tmp_path,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--audit-run" in result.stdout and "--expected-audit-request" in result.stdout and "--out" in result.stdout


def test_saved_request_without_scheduler_url_reaches_transport_boundary(tmp_path, monkeypatch):
    native, audit = tmp_path / "native", tmp_path / "audit"
    native.mkdir(); audit.mkdir()
    n = save(native / "request.json", dict(config={}))
    (native / "submission.json").write_text(json.dumps(dict(job_id="job-bbbb33334444")))
    a = save(audit / "request.json", dict(source_hashes={"audit_qwen_row_sign.py": C["READER_SHA256"]},
        config=dict(native_request_sha256=n["request_sha256"], source_job="job-bbbb33334444")))
    assert "scheduler_url" not in a
    monkeypatch.delenv("MRUN_URL", raising=False)
    assert C["scheduler_url"]() == "http://192.168.68.68:9025"
    monkeypatch.setenv("MRUN_URL", "http://current-scheduler:9025")
    assert C["scheduler_url"]() == "http://current-scheduler:9025"
    assert C["scheduler_url"]("http://explicit-scheduler:9025") == "http://explicit-scheduler:9025"
    monkeypatch.setitem(C["collect"].__globals__, "QUAL", SOURCE.parent.parent / "qualification")
    calls = []
    def transport(*args, **kwargs):
        import os
        calls.append(os.environ["MRUN_URL"])
        raise RuntimeError("model-free network boundary; no network executed")
    monkeypatch.setattr(C["collect"].__globals__["runpy"], "run_path", lambda path: {
        "custody_encoder": lambda *a: ({}, tmp_path / "encoder.py", {}), "collect": transport})
    with pytest.raises(RuntimeError, match="network boundary"):
        C["collect"](audit, native, tmp_path / "fresh", expected_request=a["request_sha256"],
            encoder_source=tmp_path / "encoder", scheduler="http://explicit-scheduler:9025")
    assert calls == ["http://explicit-scheduler:9025"] and not (tmp_path / "fresh").exists()

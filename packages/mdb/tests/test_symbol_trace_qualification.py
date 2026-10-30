"""Admission and immutable collection mechanics for the symbol-trace worker."""

from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "qualification" / "submit_symbol_trace.py"
_SPEC = importlib.util.spec_from_file_location("submit_symbol_trace_source", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


def _report_logs(*, report: dict, archive: bytes) -> str:
    encoded_report = base64.b64encode(
        gzip.compress(json.dumps(report, sort_keys=True).encode("utf-8"))
    ).decode("ascii")
    encoded_archive = base64.b64encode(archive).decode("ascii")
    return (
        f"SYMBOL_TRACE_REPORT_V1 {encoded_report}\n"
        f"SYMBOL_TRACE_ARCHIVE_V1 {encoded_archive}\n"
    )


def _report(archive: bytes, *, archive_digest: str | None = None) -> dict:
    return {
        "schema": "mdb-symbol-trace-qualification-v1",
        "archive": {"sha256": archive_digest or hashlib.sha256(archive).hexdigest()},
        "trace_fingerprints": {"all": "trace-seal"},
    }


def _stub_terminal(monkeypatch, *, identifier: str, state: str, logs: str):
    monkeypatch.setattr(
        qualification,
        "terminal_job",
        lambda job_id: (identifier, {"job_id": identifier, "state": state}, logs),
    )
    monkeypatch.setattr(
        qualification,
        "guarded_job_receipt",
        lambda job_id: {"job_id": job_id, "guarded": True},
    )


def test_launcher_admits_one_guarded_cuda_lease_with_256_work_cap(tmp_path, monkeypatch, capsys):
    calls: dict[str, object] = {"staged": [], "launched": []}

    monkeypatch.setattr(
        qualification,
        "intent",
        lambda: {"python": "/usr/bin/python3", "uv_link_mode": "copy"},
    )
    monkeypatch.setattr(qualification, "worker_command", lambda name: ["worker", name])

    def stage_worker(*args, **kwargs):
        calls["staged"].append((args, kwargs))

    def guarded_launch(*args, **kwargs):
        calls["launched"].append((args, kwargs))
        return "job-symbol-trace-test"

    monkeypatch.setattr(qualification, "stage_worker", stage_worker)
    monkeypatch.setattr(qualification, "guarded_mrun_launch", guarded_launch)
    monkeypatch.setattr(qualification, "RESULTS", tmp_path / "results")
    monkeypatch.setattr("sys.argv", ["submit_symbol_trace.py"])

    assert qualification.main() is None
    printed = json.loads(capsys.readouterr().out)
    assert printed["job_id"] == "job-symbol-trace-test"
    assert len(calls["staged"]) == 1
    assert len(calls["launched"]) == 1

    _, launch = calls["launched"][0]
    request = launch["request"]
    contract = request["scheduling_contract"]
    assert request["needs"] == {"cuda": True}
    assert request["geometry"]["native_transition_limit"] == 256
    assert contract["max_mrun_submissions"] == 1
    assert contract["logical_operations"] == 256
    assert contract["max_local_work_items"] == 256
    assert contract["additional_scheduler_submissions"] == 0
    assert contract["per_branch_mrun_submissions"] == 0
    assert contract["per_transition_mrun_submissions"] == 0
    assert launch["needs"] == {"cuda": True}
    assert launch["gpu"] is True
    assert launch["retry_on_kill"] is False
    assert launch["preflight"] is True
    assert launch["prefer_host"] == "beast"
    assert launch.get("pin") is None
    assert launch.get("ram_mb") is None
    assert launch.get("vram_mb") is None
    assert launch.get("disk_gb") is None
    assert launch.get("cpu_threads") is None
    assert launch.get("est_wall_s") is None

    saved_request = json.loads(
        (tmp_path / "results" / "job-symbol-trace-test" / "request.json").read_text()
    )
    assert saved_request == request


def test_active_collection_fails_before_creating_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, "RESULTS", tmp_path / "results")
    _stub_terminal(monkeypatch, identifier="active-job", state="running", logs="still running\n")

    with pytest.raises(RuntimeError, match="still active"):
        qualification.collect("active-job")
    assert not (tmp_path / "results").exists()


def test_terminal_report_archive_collection_is_idempotent(tmp_path, monkeypatch):
    archive = b"sealed native project archive"
    logs = _report_logs(report=_report(archive), archive=archive)
    monkeypatch.setattr(qualification, "RESULTS", tmp_path / "results")
    _stub_terminal(monkeypatch, identifier="finished-job", state="succeeded", logs=logs)

    first = qualification.collect("finished-job")
    root = tmp_path / "results" / "finished-job"
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    second = qualification.collect("finished-job")
    after = {path.name: path.read_bytes() for path in root.iterdir()}

    assert first == second
    assert first["report"] is True
    assert before == after
    assert (root / "native-project.tar.gz").read_bytes() == archive
    assert json.loads((root / "report.json").read_text())["schema"] == (
        "mdb-symbol-trace-qualification-v1"
    )


def test_changed_terminal_bytes_are_rejected_without_rewriting_prior_evidence(tmp_path, monkeypatch):
    archive = b"sealed archive"
    logs = _report_logs(report=_report(archive), archive=archive)
    monkeypatch.setattr(qualification, "RESULTS", tmp_path / "results")
    _stub_terminal(monkeypatch, identifier="immutable-job", state="succeeded", logs=logs)
    qualification.collect("immutable-job")
    root = tmp_path / "results" / "immutable-job"
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    _stub_terminal(
        monkeypatch,
        identifier="immutable-job",
        state="succeeded",
        logs=logs + "terminal mutation\n",
    )
    with pytest.raises(RuntimeError, match="retained terminal evidence changed"):
        qualification.collect("immutable-job")
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_wrong_archive_digest_is_rejected_without_rewriting_partial_evidence(tmp_path, monkeypatch):
    archive = b"archive whose digest must fail"
    logs = _report_logs(
        report=_report(archive, archive_digest="0" * 64),
        archive=archive,
    )
    monkeypatch.setattr(qualification, "RESULTS", tmp_path / "results")
    _stub_terminal(monkeypatch, identifier="bad-archive-job", state="succeeded", logs=logs)

    with pytest.raises(RuntimeError, match="archive differs"):
        qualification.collect("bad-archive-job")
    root = tmp_path / "results" / "bad-archive-job"
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    assert "native-project.tar.gz" not in before

    # A repeated collection must fail at the same seal check and leave the
    # partial terminal evidence byte-identical.
    with pytest.raises(RuntimeError, match="archive differs"):
        qualification.collect("bad-archive-job")
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before

"""Saved-only recovery from a synthetic failed original reporting path."""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parents[1]
R = runpy.run_path(str(HERE / "qualification/recover_flux_functional_decoder.py"))
A = runpy.run_path(str(HERE / "qualification/audit_flux_functional_decoder.py"))
T = runpy.run_path(str(HERE / "tests/test_flux_functional_decoder_campaign.py"))
OLD = HERE / "qualification/executed-decoder-worker.py"


def failed(root, monkeypatch):
    import mdb.equation_comparison as comparison

    config = T["fixture"](root)
    request = dict(config=dict(config), source_hashes={"flux_functional_decoder_campaign.py": R["FAILED_WORKER"]}, synthetic_fixture=True)
    request["request_sha256"] = R["identity"](request)
    config["request_sha256"] = request["request_sha256"]
    old = runpy.run_path(str(OLD))
    def reporting_error(*args, **kwargs):
        raise ValueError("comparison exceeds declared element budget")
    monkeypatch.setattr(comparison, "compare_equations", reporting_error)
    monkeypatch.setenv("MRUN_JOB_ID", "job-tiny-failed-decoder")
    with pytest.raises(ValueError, match="element budget"):
        old["run"](config)
    monkeypatch.undo()
    raw, packet = Path(config["output"]), root / "failed-packet"
    packet.mkdir()
    events = [json.loads(line) for line in (raw / "timeline.jsonl").read_text().splitlines()]
    (packet / "worker.log").write_text("".join("MDB_DECODER_PHASE " + json.dumps(row) + "\n" for row in events) + "ValueError: comparison exceeds declared element budget\n")
    payload = "2" * 64
    job = dict(job_id="job-tiny-failed-decoder", state="failed", result=dict(returncode=1, executed_payload=dict(sha256=payload)), payload_custody={name: dict(sha256=payload) for name in ("declared", "sealed", "executed")})
    R["save"](packet / "job.json", job)
    auth = dict(job_id=job["job_id"], original_job_state="failed", original_returncode=1, job_sha256=R["sha"](packet / "job.json"), log_sha256=R["sha"](packet / "worker.log"),
        request_sha256=request["request_sha256"], executed_payload_sha256=payload, global_forecast_seal=next(r for r in events if r["kind"] == "cohort_globally_sealed"),
        observed_forecast_completions=[r for r in events if r["kind"] == "forecast_case_completed"], observed_native_completions=[r for r in events if r["kind"] == "native_case_completed"])
    R["save"](packet / "authentication.json", auth)
    request_path, inventory_path = root / "failed-request.json", root / "retained-files.json"
    R["save"](request_path, request)
    inventory = dict(schema="mdb-failed-decoder-retained-files-v1", job_id=job["job_id"], request_sha256=request["request_sha256"], original_job_state="failed", files={p.relative_to(raw).as_posix(): dict(bytes=p.stat().st_size, sha256=R["sha"](p)) for p in raw.rglob("*") if p.is_file()})
    R["save"](inventory_path, inventory)
    return config, raw, packet, request_path, inventory_path


def test_failed_capture_recovered_without_original_mutation_or_neural_calls(tmp_path, monkeypatch):
    config, raw, packet, request, inventory = failed(tmp_path, monkeypatch)
    before = {p.relative_to(raw).as_posix(): (p.stat().st_ino, p.stat().st_ctime_ns, R["sha"](p)) for p in raw.rglob("*") if p.is_file()}
    metadata = tmp_path / "derived-metadata"
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-only recovery entered neural module")
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    R["prepare"](raw, metadata, inventory_path=inventory, expected_manifest=R["sha"](inventory), packet=packet, original_request=request,
        assets={key: config[key] for key in ("vae_config", "parameter_manifest", "retrieval_script")}, executed_worker=OLD,
        decoder_helper=HERE / "qualification/flux_functional_decoder.py", synthetic=True)
    report, _ = A["audit"](raw, tmp_path / "audit", expected_manifest=R["sha"](inventory), expected_request=config["request_sha256"], metadata_root=metadata)
    assert report["producer_status"] == "failed" and report["worker_sha256"] == R["FAILED_WORKER"]
    assert report["final_resident_parameter_check"] == R["UNESTABLISHED"]
    assert report["mechanics_status"] == "authenticated-recovered-full-macro-saved-evidence-audit"
    assert report["totals"]["native_decode_calls"] == report["totals"]["functional_decode_calls"] == 4
    assert report["first_macro_difference"] == dict.fromkeys(T["W"]["CASES"])
    assert all(not (raw / name).exists() for name in ("source.json", "native-contract.json", "summary.json", "files.json"))
    after = {p.relative_to(raw).as_posix(): (p.stat().st_ino, p.stat().st_ctime_ns, R["sha"](p)) for p in raw.rglob("*") if p.is_file()}
    assert before == after


def test_failed_status_cannot_be_rewritten_as_success(tmp_path, monkeypatch):
    _, _, packet, request, _ = failed(tmp_path, monkeypatch)
    job = R["read"](packet / "job.json")
    job["state"] = "succeeded"
    (packet / "job.json").write_text(json.dumps(job))
    auth = R["read"](packet / "authentication.json")
    auth["job_sha256"] = R["sha"](packet / "job.json")
    (packet / "authentication.json").write_text(json.dumps(auth))
    with pytest.raises(ValueError, match="failed status"):
        R["authenticate_packet"](packet, request, synthetic=True)


def test_closed_journal_does_not_erase_prior_failure(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    for number, kind in enumerate(("opened", "failed", "closed")):
        R["save"](events / (str(number) + ".json"), dict(kind=kind, detail=dict(error="retained callback failure")))
    R["save"](tmp_path / "seal.json", dict(violations=[]))
    with pytest.raises(ValueError, match="failure/violation"):
        R["journal_state"](tmp_path, ["opened", "failed", "closed"])


def test_recovery_cli_exports_separate_compact_and_preserves_failure(tmp_path, monkeypatch):
    original, raw, packet, failed_request, inventory = failed(tmp_path, monkeypatch)
    path = HERE / "qualification/recover_flux_functional_decoder.py"
    config = dict(native_root=str(raw), native_manifest_sha256=R["sha"](inventory), native_request_sha256=original["request_sha256"], source_job="job-tiny-failed-decoder",
        retained_inventory=str(inventory), failed_packet=str(packet), original_request=str(failed_request), executed_worker=str(OLD), decoder_helper=str(HERE / "qualification/flux_functional_decoder.py"),
        assets={key: original[key] for key in ("vae_config", "parameter_manifest", "retrieval_script")}, synthetic_fixture=True, output_scratch=str(tmp_path / "recovery-output"))
    request = dict(config=config, request_sha256="3" * 64, source_hashes={str(path): R["sha"](path)})
    config_path, request_path = tmp_path / "recovery-config.json", tmp_path / "recovery-request.json"
    R["save"](config_path, config)
    R["save"](request_path, request)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SATURN_RESIDENT_WORKER": "1", "SATURN_RUN_CONFIG": str(config_path), "SATURN_DEBUG_REQUEST": str(request_path), "MRUN_JOB_ID": "job-tiny-saved-recovery"}
    result = subprocess.run([sys.executable, "-I", str(path)], cwd=HERE / "qualification", env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    capsule = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(capsule["output_root"])
    assert R["sha"](output / "files.json") == capsule["files_manifest_sha256"]
    summary = R["read"](output / "summary.json")
    assert summary["job_id"] == "job-tiny-saved-recovery" and summary["original_producer_status"] == "failed"
    assert summary["final_resident_parameter_check"] == R["UNESTABLISHED"]
    assert R["read"](output / "failed-source-authentication/job.json")["state"] == "failed"
    assert len(list(output.glob("*-images/*.png"))) == 8 and not (output / "forecast-store").exists()

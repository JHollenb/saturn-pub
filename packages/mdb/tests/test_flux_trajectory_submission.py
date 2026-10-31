"""Finite admission and result diagnostics for an own-state image forecast."""
import ast
import base64
import gzip
import hashlib
import io
import json
import os
import runpy
import tarfile
import time
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import mdb.job_support
import mdb.payload
from mdb.forecast_transaction import ForecastTransaction, verify_transaction

QUALIFICATION = Path(__file__).parents[1] / "qualification"


def test_trajectory_launch_binds_one_fresh_state_and_finite_native_comparison(tmp_path, monkeypatch):
    source = runpy.run_path(str(QUALIFICATION / "submit_flux_trajectory_forecast.py"))
    staged = {}

    def stage_payload(path, **values):
        staged.update(values)
        assert values["request"]["config"] == json.loads(values["config_bytes"]) == values["config"]

    def launch(command, **values):
        request = values["request"]
        assert request == staged["request"]
        assert values["gpu"] and values["preflight"] and not values["retry_on_kill"]
        assert values["vram_mb"] == request["resource_exception"]["vram_mb"] == 13000
        assert values["ram_mb"] == request["resource_exception"]["ram_mb"] == 32000
        assert request["needs"] == {"cuda": True}
        assert request["config"]["seed"] == 23
        assert request["config"]["max_transitions"] == request["config"]["max_native_transitions"] == 111
        contract = request["scheduling_contract"]
        assert contract["model_loads"] == 1 and contract["logical_branches"] == 2
        assert request["config"]["max_images"] == 2
        assert {name for _, name in staged["additional_files"]} >= {"flux_functional_denoiser.py", "flux_functional_vae.py"}
        return "job-tiny-admission"

    monkeypatch.setattr(mdb.payload, "stage_worker", stage_payload)
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["python", name])
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", launch)
    assert source["stage"](tmp_path / "run", inventory={"model_path": "/explicit/checkpoint", "checkpoint_sha256": "a" * 64})["job_id"] == "job-tiny-admission"


def test_saved_comparison_keeps_signed_error_and_exactness_separate():
    compare = runpy.run_path(str(QUALIFICATION / "flux_trajectory_forecast_campaign.py"))["compare"]
    result = compare(np.array([1., -2.]), np.array([1.25, -1.]))
    assert result["changed_elements"] == 2 and not result["exact"]
    assert result["max_abs"] == 1. and result["largest_effect"]["native_minus_forecast"] == -1.
    assert "passed" not in result
    with pytest.raises(ValueError, match="nonfinite"):
        compare(np.array([np.nan]), np.array([0.]))


def test_complete_publication_serializes_actual_readonly_observation_through_archive(tmp_path, monkeypatch, capsys):
    path = QUALIFICATION / "flux_trajectory_forecast_campaign.py"
    worker = runpy.run_path(str(path))
    emitter = runpy.run_path(str(QUALIFICATION / "discovery_upgrades.py"))
    output = tmp_path / "result"
    with ForecastTransaction(output / "transaction", model_identity="tiny", input_identity="input",
                             source_identity="source", frontier={"kind": "empty"}) as tx:
        tx.predict("forecast", lambda context: {"predictions": {"expected": 0.}})
        seal = tx.seal()
        tx.execute("native", lambda context: {"pixel_error": {"max_abs": 0., "coordinates": [0, 1],
            "largest_effect": {"coordinate": [0, 0], "native_minus_forecast": 0.}}})
    observation = tx.observation("native")
    assert isinstance(observation["pixel_error"], MappingProxyType)
    assert isinstance(observation["pixel_error"]["largest_effect"], MappingProxyType)
    assert isinstance(observation["pixel_error"]["coordinates"], tuple)
    with pytest.raises(TypeError, match="mappingproxy"):
        json.dumps({"pixel_error": observation["pixel_error"]})
    verification = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
    assert verification["verified"] and tx.report()["phase"] == "closed"
    original = {p.relative_to(output): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (output / "transaction").rglob("*") if p.is_file()}

    # Execute the worker's exact publication statements and nested event helper,
    # without importing or entering its model-running worker function. This
    # includes the real shared emitter; save() alone did not cover that sink.
    tree = ast.parse(path.read_text())
    event = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "event")
    summary = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "summary" for target in node.targets))
    publication = sorted([summary, *(node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                                    and summary.end_lineno < node.lineno < next(
                                        item.lineno for item in ast.walk(tree) if isinstance(item, ast.For)
                                        and item.lineno > summary.end_lineno))], key=lambda node: node.lineno)
    assert len(publication) == 6
    monkeypatch.setenv("MRUN_JOB_ID", "job-report-only-tiny")
    namespace = {**worker, "output": output, "transaction": tx, "seal": seal, "observation": observation,
        "verification": verification, "started": time.monotonic(), "timeline": [], "time": time, "os": os,
        "request": {"request_sha256": "a" * 64}, "source": {"model_identity": "tiny"}, "counts": {},
        "config": {"arithmetic": "tiny report-only", "scope": "no model execution", "limits": "none",
                   "max_archive_bytes": 1 << 20},
        "runpy": SimpleNamespace(run_path=lambda name: emitter if name == "campaign_support.py" else None)}
    exec(compile(ast.Module(body=[event, *publication], type_ignores=[]), str(path), "exec"), namespace)
    saved = json.loads((output / "summary.json").read_text())
    timeline = [json.loads(line) for line in (output / "timeline.jsonl").read_text().splitlines()]
    assert saved["pixel_error"] == timeline[0]["pixel_error"] == worker["json_value"](observation["pixel_error"])
    assert timeline[0]["kind"] == "complete"
    assert json.loads((output / "transaction-report.json").read_text())["phase"] == "closed"
    lines = capsys.readouterr().out.splitlines()
    report = json.loads(gzip.decompress(base64.b64decode(next(line.removeprefix(emitter["REPORT"])
        for line in lines if line.startswith(emitter["REPORT"])))))
    raw = base64.b64decode(next(line.removeprefix(emitter["ARCHIVE"]) for line in lines if line.startswith(emitter["ARCHIVE"])))
    assert report["archive_sha256"] == hashlib.sha256(raw).hexdigest()
    assert {key: value for key, value in report.items() if key != "archive_sha256"} == saved
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for name in ("summary.json", "transaction-report.json", "timeline.jsonl", "files.json"):
            member = next(item for item in archive.getmembers() if item.name.endswith("/" + name))
            assert archive.extractfile(member).read() == (output / name).read_bytes()
    assert original == {p.relative_to(output): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (output / "transaction").rglob("*") if p.is_file()}
    assert verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])["verified"]
    assert isinstance(observation["pixel_error"], MappingProxyType)


def test_failed_publication_metadata_recovery_requires_explicit_opt_in(tmp_path):
    helper = runpy.run_path(str(QUALIFICATION / "summarize_context_window.py"))
    data = tmp_path / "extracted" / "tiny-result"
    with ForecastTransaction(data / "transaction", model_identity="tiny", input_identity="input",
                             source_identity="source", frontier={"kind": "empty"}) as tx:
        tx.predict("forecast", lambda context: {"predictions": {"expected": 1.}})
        tx.seal()
        tx.execute("native", lambda context: {"value": 1.})
    # Mimic a worker that closed a complete transaction but failed before the
    # convenience report and plan were written. The raw archive is unchanged.
    raw_plan = {"tiny": True}
    request = {"config": raw_plan, "source_hashes": {}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(data, arcname="tiny-result")
    archive_bytes = buffer.getvalue()
    (tmp_path / "worker.log").write_text("MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 " + base64.b64encode(archive_bytes).decode() + "\n")
    records = {"request.json": request, "plan.json": raw_plan,
               "job.json": {"job_id": "job-tiny", "state": "failed"},
               "report.json": {"job_id": "job-tiny", "request_sha256": request["request_sha256"],
                               "archive_sha256": hashlib.sha256(archive_bytes).hexdigest()}}
    for name, value in records.items():
        (tmp_path / name).write_text(json.dumps(value))
    original = {p.relative_to(data): hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="explicit failed-publication"):
        helper["source_run"](tmp_path, "tiny-result", failed=True)
    recovered = helper["source_run"](tmp_path, "tiny-result", failed=True, recover_metadata=True)
    assert recovered["transaction_report"]["phase"] == "closed"
    assert recovered["transaction_report"]["observations"] == ["native"]
    assert recovered["plan"] == raw_plan
    assert set(recovered["source"]["metadata_recovery"]) == {"transaction_report", "plan"}
    assert original == {p.relative_to(data): hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob("*") if p.is_file()}

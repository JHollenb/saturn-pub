"""Offline thin-controller guards; generic staging/submission are never run."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"
PATH = HERE / "stage_flux_h18_saved_audit.py"
S = runpy.run_path(str(PATH))


def delegate(monkeypatch):
    original = runpy.run_path
    calls = []

    def controlled(path, *args, **kwargs):
        if Path(path).name == S["CONTROLLER"]:
            return dict(stage=lambda *a, **k: calls.append(("stage", a, k)) or {"staged": "mock-only"},
                        submit=lambda *a, **k: calls.append(("submit", a, k)) or {"job_id": "mock-no-execution"})
        return original(path, *args, **kwargs)

    monkeypatch.setattr(S["execute"].__globals__["runpy"], "run_path", controlled)
    return calls


def test_exact_reader_seven_helpers_and_offline_stage_delegation(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    assert S["execute"](tmp_path / "fresh", study=tmp_path / "completed") == {"staged": "mock-only"}
    assert len(calls) == 1
    mode, args, kwargs = calls[0]
    assert mode == "stage" and args == (tmp_path / "fresh", tmp_path / "completed")
    reader = runpy.run_path(str(HERE / S["WORKER"]))
    assert len(reader["HELPERS"]) == 7
    assert kwargs["helpers"] == [*reader["HELPERS"], PATH.name]
    assert kwargs["worker"] == HERE / S["WORKER"]
    assert kwargs["experiment"] == "mdb-flux-h18-history-saved-audit" and kwargs["operations"] == 6
    assert "224advances/8actualcontext operands" in kwargs["scope"]
    assert not (tmp_path / "fresh").exists()


def test_missing_study_and_source_tamper_refuse_without_delegate(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    with pytest.raises(ValueError, match="Completed original"):
        S["execute"](tmp_path / "fresh")
    sha = S["sha"]
    monkeypatch.setitem(S["execute"].__globals__, "sha", lambda path: "bad" if Path(path).name == "audit_flux_e18_conditioning_act.py" else sha(path))
    with pytest.raises(ValueError, match="account helper changed"):
        S["execute"](tmp_path / "fresh", study=tmp_path / "completed")
    assert calls == [] and not (tmp_path / "fresh").exists()


def test_submit_checks_actual_staged_source_and_unchanged_CPU_contract(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    reader = runpy.run_path(str(HERE / S["WORKER"]))
    request = dict(source_hashes={S["WORKER"]: S["WORKER_SHA256"], **reader["PINS"], PATH.name: S["sha"](PATH)},
        resource_exception=dict(cpu_threads=2, ram_mb=3072, vram_mb=0, est_wall_s=240))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    assert S["execute"](tmp_path, submit=True) == {"job_id": "mock-no-execution"}
    assert calls[0][0] == "submit"
    request["resource_exception"]["est_wall_s"] = 300
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="resource estimate"):
        S["execute"](tmp_path, submit=True)
    request["resource_exception"]["est_wall_s"] = 240
    request["source_hashes"][S["WORKER"]] = "wrong"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="reader/frontend/helper"):
        S["execute"](tmp_path, submit=True)
    assert len(calls) == 1


def test_actual_help_and_missing_study_cli_are_offline(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--study" in result.stdout and "--submit" in result.stdout and "240s" in result.stdout
    missing = subprocess.run([sys.executable, "-I", "-B", str(PATH), str(tmp_path / "fresh")], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert missing.returncode == 2 and "--study is required" in missing.stderr
    assert not (tmp_path / "fresh").exists()

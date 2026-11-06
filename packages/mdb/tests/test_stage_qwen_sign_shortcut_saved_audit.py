"""Model-free source/resource guards; never stage, submit, SSH or call a service."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"
PATH = HERE / "stage_qwen_sign_shortcut_saved_audit.py"
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


def test_exact_four_saved_helpers_and_frontend_offline_delegation(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    assert S["execute"](tmp_path / "fresh", study=tmp_path / "completed") == {"staged": "mock-only"}
    mode, args, kwargs = calls[0]
    assert mode == "stage" and args == (tmp_path / "fresh", tmp_path / "completed")
    reader = runpy.run_path(str(HERE / S["WORKER"]))
    assert reader["PINS"] == S["HELPER_PINS"] and len(reader["HELPERS"]) == 4
    assert kwargs["helpers"] == [*reader["HELPERS"], PATH.name]
    assert kwargs["worker"] == HERE / S["WORKER"]
    assert kwargs["experiment"] == "mdb-qwen-sign-shortcut-saved-audit" and kwargs["operations"] == 3
    assert "zero neural/tokenizer/learned recurrence" in kwargs["scope"]
    assert len(calls) == 1 and not (tmp_path / "fresh").exists()


def test_missing_study_and_tampered_helper_refuse_without_delegate(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    with pytest.raises(ValueError, match="Completed original"):
        S["execute"](tmp_path / "fresh")
    original_sha = S["sha"]
    target = next(iter(S["HELPER_PINS"]))
    monkeypatch.setitem(S["execute"].__globals__, "sha", lambda path: "bad" if Path(path).name == target else original_sha(path))
    with pytest.raises(ValueError, match="account helper changed"):
        S["execute"](tmp_path / "fresh", study=tmp_path / "completed")
    assert calls == [] and not (tmp_path / "fresh").exists()


def test_submit_source_and_coherent_fixed_resource_guard_never_rewrites_request(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    request = dict(source_hashes={S["WORKER"]: S["WORKER_SHA256"], **S["HELPER_PINS"], PATH.name: S["sha"](PATH)},
        resource_exception=dict(cpu_threads=2, ram_mb=3072, vram_mb=0, disk_gb=1, est_wall_s=240))
    path = tmp_path / "request.json"
    raw = json.dumps(request).encode()
    path.write_bytes(raw)
    assert S["execute"](tmp_path, submit=True) == {"job_id": "mock-no-execution"}
    assert calls[0][0] == "submit" and path.read_bytes() == raw
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


def test_help_and_missing_study_cli_do_not_import_or_execute_science(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--study" in result.stdout and "--submit" in result.stdout and "240s" in result.stdout
    missing = subprocess.run([sys.executable, "-I", "-B", str(PATH), str(tmp_path / "fresh")], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert missing.returncode == 2 and "--study is required" in missing.stderr
    assert not (tmp_path / "fresh").exists()

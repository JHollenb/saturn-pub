"""Model-free guards; no real stage, submit, SSH or service call."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"
PATH = HERE / "stage_qwen_row_sign_saved_audit.py"
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


def test_exact_five_helpers_and_frontend_offline_delegation(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    assert S["execute"](tmp_path / "fresh", study=tmp_path / "completed") == {"staged": "mock-only"}
    mode, args, kwargs = calls[0]
    assert mode == "stage" and args == (tmp_path / "fresh", tmp_path / "completed")
    reader = runpy.run_path(str(HERE / S["WORKER"]))
    assert reader["PINS"] == S["HELPER_PINS"] and len(reader["HELPERS"]) == 5
    assert kwargs["helpers"] == [*reader["HELPERS"], PATH.name]
    assert kwargs["worker"] == HERE / S["WORKER"]
    assert kwargs["experiment"] == "mdb-qwen-row-sign-saved-audit" and kwargs["operations"] == 3
    assert "Inherited-functional-state" in kwargs["scope"] and "zero neural" in kwargs["scope"]
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


def test_submit_source_and_fixed_resource_guard_never_rewrites_request(tmp_path, monkeypatch):
    calls = delegate(monkeypatch)
    request = dict(source_hashes={S["WORKER"]: S["WORKER_SHA256"], **S["HELPER_PINS"], PATH.name: S["sha"](PATH)},
        resource_exception=dict(cpu_threads=2, ram_mb=3072, vram_mb=0, disk_gb=1, est_wall_s=240))
    path = tmp_path / "request.json"
    raw = json.dumps(request).encode()
    path.write_bytes(raw)
    assert S["execute"](tmp_path, submit=True) == {"job_id": "mock-no-execution"}
    assert calls[0][0] == "submit" and path.read_bytes() == raw
    request["resource_exception"]["est_wall_s"] = 120
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="resource estimate"):
        S["execute"](tmp_path, submit=True)
    request["resource_exception"]["est_wall_s"] = 240
    request["source_hashes"][S["WORKER"]] = "wrong"
    path.write_text(json.dumps(request))
    with pytest.raises(ValueError, match="reader/frontend/helper"):
        S["execute"](tmp_path, submit=True)
    assert len(calls) == 1


def test_help_and_missing_study_cli_do_not_execute_science(tmp_path):
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--study" in result.stdout and "--submit" in result.stdout and "240s" in result.stdout
    missing = subprocess.run([sys.executable, "-I", "-B", str(PATH), str(tmp_path / "fresh")], cwd=tmp_path,
                             capture_output=True, text=True, timeout=30)
    assert missing.returncode == 2 and "--study is required" in missing.stderr
    assert not (tmp_path / "fresh").exists()


def test_installed_bootstrap_reaches_actual_reader_worker_publication(tmp_path):
    """Transport-only audit stub; the reader owner tests real numerical/Atlas core."""
    from mdb.payload import seal_package

    raw = tmp_path / "raw"
    raw.mkdir()
    names = ("plan.json", "declared-config.json", "request.json", "source.json", "summary.json", "runtime-contract.json", "files.json", "source-frontier.json", "forecast-responses.json",
        "forecast.json", "shortcut.json", "native.json", "parameter-identity-before.json", "parameter-identity-after.json", "timeline.jsonl", "native-contract.json", "native-sdpa.json")
    for name in names:
        (raw / name).write_text(json.dumps({"input_root": "/synthetic/inherited", "scope": "transport-stub-only"}))
    (raw / "source-frontier.json").write_text(json.dumps(dict(input_root="/synthetic/inherited",
        source_arrays={str(i): dict(path=f"absent/{i}.npy", sha256="0" * 64) for i in range(13)})))
    (raw / "metadata").mkdir()
    (raw / "metadata/synthetic.json").write_text("{}")
    config = dict(native_root=str(raw), native_manifest_sha256="0" * 64,
        native_request_sha256="1" * 64, output_scratch=str(tmp_path / "out"))
    reader_path = HERE / S["WORKER"]
    (tmp_path / S["WORKER"]).write_bytes(reader_path.read_bytes())
    request = dict(config=config, source_hashes={S["WORKER"]: S["WORKER_SHA256"]})
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    (tmp_path / "saturn_run_config.json").write_text(json.dumps(config))
    (tmp_path / "saturn_debug_request.json").write_text(json.dumps(request))
    wrapper = tmp_path / "entry.py"
    wrapper.write_text("import runpy,json\nfrom pathlib import Path\n"
        + "n=runpy.run_path(" + repr(str(reader_path)) + ")\n"
        + "def audit(root,output,**kwargs):\n output=Path(output);output.mkdir(parents=True);(output/'audit.json').write_text(json.dumps({'scope':'transport-stub-only'}));return {'job_id':'job-bbbb33334444'},{'files':{}}\n"
        + "n['worker'].__globals__['audit']=audit\nn['worker']()\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "payload-seal.json").write_text(json.dumps(dict(
        packages={name: seal_package(name) for name in ("mdb", "saturn_pub")}, wheels={}, files={},
        worker=wrapper.name, worker_sha256=digest(wrapper), config_bytes_sha256=digest(tmp_path / "saturn_run_config.json"))))
    env = {**os.environ, "SATURN_RUN_CONFIG": "saturn_run_config.json", "SATURN_DEBUG_REQUEST": "saturn_debug_request.json",
        "MRUN_JOB_ID": "job-aaaa11112222", "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run([sys.executable, "-I", "-B", "-m", "mdb.worker_bootstrap", wrapper.name], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    assert len(lines) == 1
    marker = json.loads(lines[0].removeprefix("MDB_DIRECTORY_READY "))
    pack = Path(marker["output_root"])
    summary = json.loads((pack / "summary.json").read_text())
    assert summary["job_id"] == marker["job_id"] == "job-aaaa11112222"
    assert summary["request_sha256"] == marker["request_sha256"] == request["request_sha256"]
    assert digest(pack / "files.json") == marker["files_manifest_sha256"]
    assert digest(pack / "summary.json") == marker["summary_sha256"]
    assert json.loads((pack / "request.json").read_text()) == request

"""Framework-free planned SSH custody fixtures; no network/model calls."""
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = Path(__file__).resolve().parents[1] / "qualification/collect_flux_step_boundary.py"
C = runpy.run_path(str(MODULE))
JOB = "job-tiny123"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path.resolve()
    run, remote = root / "run", root / "remote"
    (run / "payload").mkdir(parents=True)
    (run / "payload/worker.py").write_text("# pinned tiny worker\n")
    request = {"config": {"transport_method": C["METHOD"], "transport_limits": {
        "max_expanded_bytes": C["MAX_TOTAL"], "max_member_bytes": C["MAX_MEMBER"]}},
        "source_hashes": {"worker.py": C["sha"](run / "payload/worker.py")}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    write(run / "request.json", request)
    write(remote / "plan.json", request["config"])
    write(remote / "source.json", {"job_id": JOB, "request_sha256": request["request_sha256"],
        "source_hashes": request["source_hashes"]})
    write(remote / "summary.json", {"job_id": JOB, "request_sha256": request["request_sha256"]})
    write(remote / "transaction/seal.json", {"sealed": True})
    (remote / "array.bin").write_bytes(bytes(range(256)))
    files = C["inventory"](remote)["files"]
    write(remote / "remote-manifest.json", {"schema": "mdb-step-boundary-remote-manifest-v1", "job_id": JOB,
        "request_sha256": request["request_sha256"], "files": files})
    pin = {"sha256": "a" * 64, "size_bytes": 123}
    logs = b"worker completed; manifest available\n"
    job = {"job_id": JOB, "state": "succeeded", "assigned_host": "beast", "config": request,
        "submitted_request": {"config": request}, "payload_custody": {k: pin.copy() for k in ("declared", "sealed", "executed")},
        "result": {"returncode": 0, "status": "ok", "executed_payload": {**pin, "host": "beast"},
            "diagnostics": {"cwd": "/mnt/hdd8tb/mrun-work/" + JOB},
            "artifacts": {"job_log": {"bytes": len(logs), "sha256": hashlib.sha256(logs).hexdigest()}}}}
    calls = []
    class API:
        def request(self, method, path, **kwargs):
            if path.endswith("/events"):
                return 200, b'[{"state":"succeeded"}]', {}
            if "/logs?" in path:
                return 200, logs, {}
            return 200, json.dumps(job).encode(), {}
    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "ssh":
            assert "StrictHostKeyChecking=yes" in command and command[-2:] == ["python3", "-"]
            assert "shell" not in kwargs
            return SimpleNamespace(stdout=json.dumps(C["inventory"](remote)), stderr="", returncode=0)
        assert command[0] == "rsync" and "--no-links" in command and "--no-specials" in command
        allowlist = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--files-from=")))
        for name in allowlist.read_bytes().split(b"\0"):
            if name:
                target = Path(command[-1]) / name.decode()
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(remote / name.decode(), target)
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    return SimpleNamespace(run=run, remote=remote, request=request, job=job, api=API(), runner=runner, calls=calls)


def test_complete_planned_collection(fixture):
    f = fixture
    result = C["collect"](f.run, JOB, api=f.api, runner=f.runner)
    assert result["scheduler_status"] == "succeeded" and result["archive_sha256"] is None
    assert result["collection_mode"] == C["METHOD"] and result["file_count"] == 6
    assert [x[0] for x in f.calls] == ["ssh", "rsync", "ssh"]
    assert (f.run / "collection.json").is_file()
    assert not (f.run / "archive.tar.gz").exists()
    assert not (f.run / "report.json").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "a\\b", "a\nname", "a//b", ".", None])
def test_unsafe_names(name):
    with pytest.raises(ValueError):
        C["relative"](name)


@pytest.mark.parametrize("kind", ["file", "directory", "fifo"])
def test_nonregular_members(tmp_path, kind):
    root = tmp_path.resolve() / "data"
    root.mkdir()
    outside = tmp_path.resolve() / "outside"
    outside.write_text("secret")
    if kind == "fifo":
        os.mkfifo(root / "bad")
    else:
        (root / "bad").symlink_to(outside if kind == "file" else tmp_path.resolve(), target_is_directory=kind == "directory")
    with pytest.raises(ValueError):
        C["inventory"](root)


@pytest.mark.parametrize("limits", [{"max_member": 3}, {"max_total": 3}, {"max_files": 0}])
def test_bounds(tmp_path, limits):
    root = tmp_path.resolve()
    (root / "a").write_bytes(b"four")
    with pytest.raises(ValueError):
        C["inventory"](root, **limits)


@pytest.mark.parametrize("change", ["failed", "path", "custody", "request"])
def test_job_binding(fixture, change):
    f = fixture
    if change == "failed":
        f.job["state"] = "failed"
    elif change == "path":
        f.job["result"]["diagnostics"]["cwd"] += "/../other"
    elif change == "custody":
        f.job["payload_custody"]["sealed"]["sha256"] = "b" * 64
    else:
        f.job["config"] = {**f.request, "request_sha256": "b" * 64}
    with pytest.raises(ValueError):
        C["collect"](f.run, JOB, api=f.api, runner=f.runner)
    assert not f.calls


def test_invalid_job_rejected_before_api(fixture):
    class NoAPI:
        def request(self, *args, **kwargs):
            raise AssertionError("invalid job must not reach API")
    with pytest.raises(ValueError, match="invalid job"):
        C["collect"](fixture.run, "job-foo/../bar", api=NoAPI(), runner=fixture.runner)


def test_changed_remote_during_copy(fixture):
    f = fixture
    def runner(command, **kwargs):
        result = f.runner(command, **kwargs)
        if command[0] == "rsync":
            (f.remote / "array.bin").write_bytes(b"changed")
        return result
    with pytest.raises(ValueError, match="changed during transfer"):
        C["collect"](f.run, JOB, api=f.api, runner=runner)
    assert not (f.run / "collection.json").exists()


def test_manifest_missing_member(fixture):
    f = fixture
    value = json.loads((f.remote / "remote-manifest.json").read_text())
    value["files"].pop("array.bin")
    write(f.remote / "remote-manifest.json", value)
    with pytest.raises(ValueError, match="manifest"):
        C["collect"](f.run, JOB, api=f.api, runner=f.runner)


def test_local_transfer_tamper(fixture):
    f = fixture
    def runner(command, **kwargs):
        result = f.runner(command, **kwargs)
        if command[0] == "rsync":
            (Path(command[-1]) / "array.bin").write_bytes(b"changed")
        return result
    with pytest.raises(ValueError, match="local evidence differs"):
        C["collect"](f.run, JOB, api=f.api, runner=runner)


def test_remote_script_is_framework_free_and_same_inventory(fixture):
    namespace = {}
    script = C["remote_script"](str(fixture.remote))
    assert "torch" not in script and "subprocess" not in script
    exec(script, namespace)
    assert namespace["inventory"](fixture.remote) == C["inventory"](fixture.remote)

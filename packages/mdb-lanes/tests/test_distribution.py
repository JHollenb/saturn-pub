import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from mdb_lanes.payload import (
    build_runtime_wheel,
    runtime_distributions,
    stage_worker,
    worker_command,
)


def test_offline_imports_need_no_model_libraries():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'diffusers', 'saturn', 'mrun'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
import mdb
from mdb import BenchSession
import mdb_lanes.bench_verify, mdb_lanes.evidence_pipeline, mdb_lanes.evidence_specs, mdb_lanes.hyperion
assert not any(name == 'saturn' or name.startswith('saturn.') for name in sys.modules)
from mdb.cli import main
assert main(['version']) == 0
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_payload_wheel_is_deterministic_and_contains_both_workers(tmp_path):
    lanes = {"package": "mdb_lanes", "distribution": "mdb-lanes"}
    wheel = build_runtime_wheel(tmp_path, **lanes)
    first = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert hashlib.sha256(build_runtime_wheel(tmp_path, **lanes).read_bytes()).hexdigest() == first
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "mdb_lanes/workers/run_saturn_bench.py" in names
        assert "mdb_lanes/workers/run_saturn_mvm_op.py" in names
        assert not any(name.startswith(("saturn/", "common/", "mdb/")) for name in names)
    with zipfile.ZipFile(build_runtime_wheel(tmp_path)) as archive:  # the engine wheel
        names = archive.namelist()
        assert "mdb/engine/run.py" in names
        assert not any(name.startswith(("mdb_lanes/", "mdb/workers/")) for name in names)
    assert [row[:2] for row in runtime_distributions()][:2] == [
        ("mdb", "mdb-runtime"), ("mdb_lanes", "mdb-lanes")]


@pytest.mark.parametrize("target", ["../escape.py", "/absolute.py", ".", "payload-seal.json"])
def test_payload_refuses_invalid_targets_before_copy(tmp_path, target):
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n")
    with pytest.raises(ValueError):
        stage_worker(tmp_path / "stage", worker=worker, config={}, request={},
                     additional_files=[(worker, target)])


def test_worker_command_has_no_checkout_paths():
    command = worker_command("run_saturn_bench.py")
    assert "--no-project" in command
    assert "--offline" in command
    assert "mdb_lanes.worker_bootstrap" in command
    assert any(name.startswith("./saturn_pub-") and name.endswith(".whl") for name in command)
    assert "UV_LINK_MODE=symlink" in command
    assert not any("domains/" in item or "common_harness" in item for item in command)


def test_worker_link_mode_is_explicit_and_validated(monkeypatch):
    from mdb_lanes.execution_environment import intent
    monkeypatch.setenv("MDB_WORKER_UV_LINK_MODE", "copy")
    assert intent()["uv_link_mode"] == "copy"
    assert "UV_LINK_MODE=copy" in worker_command("worker.py")
    monkeypatch.setenv("MDB_WORKER_UV_LINK_MODE", "invalid")
    with pytest.raises(ValueError, match="link mode"):
        worker_command("worker.py")


def test_bootstrap_accepts_sealed_payload_and_rejects_mutation(tmp_path, monkeypatch):
    pytest.importorskip("mrun", reason="worker payload staging requires the optional mrun package")
    from mdb_lanes.worker_bootstrap import main
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n")
    raw = b'{"a": 1}\n'
    request = {"config_bytes_sha256": hashlib.sha256(raw).hexdigest()}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    stage = tmp_path / "stage"
    stage_worker(stage, worker=worker, config={"a": 1}, request=request, config_bytes=raw)
    seal = json.loads((stage / "payload-seal.json").read_text())
    assert "saturn_pub" in seal["packages"]
    assert any(name.startswith("saturn_pub-") for name in seal["wheels"])
    monkeypatch.chdir(stage)
    monkeypatch.setenv("SATURN_RUN_CONFIG", "saturn_run_config.json")
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", "saturn_debug_request.json")
    monkeypatch.setattr(sys, "argv", ["bootstrap", "worker.py", "--model", "fixture"])
    executed = []
    worker_args = []

    def execute(*args, **kwargs):
        executed.append(args)
        worker_args.append(list(sys.argv))

    monkeypatch.setattr("mdb_lanes.worker_bootstrap.runpy.run_path", execute)
    main()
    assert executed == [("worker.py",)]
    assert worker_args == [["worker.py", "--model", "fixture"]]
    assert sys.argv == ["bootstrap", "worker.py", "--model", "fixture"]
    (stage / "saturn_run_config.json").write_text('{"a": 2}')
    with pytest.raises(RuntimeError, match="configuration identity"):
        main()
    assert len(executed) == 1

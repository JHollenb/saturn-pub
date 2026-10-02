import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from mdb.payload import build_runtime_wheel, stage_worker, worker_command


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
import mdb.bench_verify, mdb.evidence_pipeline, mdb.hyperion
from mdb.cli import main
assert main(['version']) == 0
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_payload_wheel_is_deterministic_and_contains_both_workers(tmp_path):
    wheel = build_runtime_wheel(tmp_path)
    first = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert hashlib.sha256(build_runtime_wheel(tmp_path).read_bytes()).hexdigest() == first
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "mdb/workers/run_saturn_bench.py" in names
        assert "mdb/workers/run_saturn_mvm_op.py" in names
        assert not any(name.startswith(("saturn/", "common/")) for name in names)


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
    assert "mdb.worker_bootstrap" in command
    assert not any("domains/" in item or "common_harness" in item for item in command)


def test_bootstrap_accepts_sealed_payload_and_rejects_mutation(tmp_path, monkeypatch):
    from mdb.worker_bootstrap import main
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n")
    raw = b'{"a": 1}\n'
    request = {"config_bytes_sha256": hashlib.sha256(raw).hexdigest()}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    stage = tmp_path / "stage"
    stage_worker(stage, worker=worker, config={"a": 1}, request=request, config_bytes=raw)
    monkeypatch.chdir(stage)
    monkeypatch.setenv("SATURN_RUN_CONFIG", "saturn_run_config.json")
    monkeypatch.setenv("SATURN_DEBUG_REQUEST", "saturn_debug_request.json")
    monkeypatch.setattr(sys, "argv", ["bootstrap", "worker.py"])
    executed = []
    monkeypatch.setattr("mdb.worker_bootstrap.runpy.run_path", lambda *a, **kw: executed.append(a))
    main()
    assert executed == [("worker.py",)]
    (stage / "saturn_run_config.json").write_text('{"a": 2}')
    with pytest.raises(RuntimeError, match="configuration identity"):
        main()
    assert len(executed) == 1

import email
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

import pytest


def test_public_checkpoint_operations_preserve_native_implementation():
    pytest.importorskip("mrun")
    from mrun.diffusion import checkpoint_ops, nonflux, phase
    assert checkpoint_ops.restore_state_inputs is nonflux._restore_state_inputs
    assert checkpoint_ops.step_scheduler is nonflux._step_scheduler
    assert checkpoint_ops.decode_sdxl is nonflux._decode_sdxl
    assert checkpoint_ops.decode_flux1 is phase._flux1_decode
    assert checkpoint_ops.decode_flux2 is phase._flux2_decode


def test_payload_seals_public_distribution_and_retains_notices(tmp_path):
    pytest.importorskip("mrun")
    from mdb.payload import runtime_wheel_names, stage_worker, worker_command
    worker = tmp_path / "worker.py"
    worker.write_text("pass\n")
    staging = tmp_path / "stage"
    stage_worker(staging, worker=worker, config={}, request={})
    name = next(name for name in runtime_wheel_names() if name.startswith("mrun_pub-"))
    with zipfile.ZipFile(staging / name) as archive:
        metadata_name = next(n for n in archive.namelist() if n.endswith("/METADATA"))
        assert email.message_from_bytes(archive.read(metadata_name))["Name"] == "mrun-pub"
        assert any(n.endswith("/licenses/LICENSE") for n in archive.namelist())
        assert any(n.endswith("/licenses/NOTICE") for n in archive.namelist())
    seal = json.loads((staging / "payload-seal.json").read_text())
    assert seal["wheels"][name] == hashlib.sha256((staging / name).read_bytes()).hexdigest()
    assert "./" + name in worker_command("worker.py")
    assert "site-packages" in str(Path(importlib.metadata.distribution("mrun-pub").locate_file("")))


def test_worker_paths_are_explicit_and_portable(monkeypatch, tmp_path):
    from mdb.execution_environment import intent
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for name in ("MDB_WORKER_UV_CACHE_DIR", "MDB_WORKER_TORCH_HOME", "LLM_MODELS_EXTRA_ROOT", "MDB_WORKER_MODEL_ROOTS"):
        monkeypatch.delenv(name, raising=False)
    assert intent()["uv_cache_dir"] == str(tmp_path / "mdb/uv")
    assert intent()["extra_model_roots"] == ""
    monkeypatch.setenv("MDB_WORKER_MODEL_ROOTS", "/srv/models")
    assert intent()["extra_model_roots"] == "/srv/models"

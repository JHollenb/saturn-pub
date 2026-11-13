"""Declared full-weight members must fit before admitting another model run."""
import hashlib
import json
import runpy
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"


def test_linux_worker_cache_intent_cannot_silently_inherit_mac_paths(monkeypatch):
    intent = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))["execution_intent"]
    for name in ("MDB_WORKER_UV_CACHE_DIR", "MDB_WORKER_TORCH_HOME", "MDB_WORKER_MODEL_ROOTS"):
        monkeypatch.delenv(name, raising=False)
    value = intent()
    assert value["uv_cache_dir"] == "/mnt/hdd8tb/mdb/uv-cache"
    assert value["torch_home"] == "/mnt/hdd8tb/mdb/torch"
    assert value["extra_model_roots"] == "/mnt/big/llm-models"
    monkeypatch.setenv("MDB_WORKER_UV_CACHE_DIR", "/Users/example/.cache/mdb/uv")
    with pytest.raises(ValueError, match="worker-native cache paths"):
        intent()


def test_submission_command_consumes_sealed_environment_even_with_hostile_local_defaults(tmp_path, monkeypatch):
    import mdb_lanes.job_support

    launcher = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))
    for name in ("MDB_WORKER_UV_CACHE_DIR", "MDB_WORKER_TORCH_HOME", "MDB_WORKER_MODEL_ROOTS"):
        monkeypatch.delenv(name, raising=False)
    environment = launcher["execution_intent"]()
    monkeypatch.setenv("MDB_WORKER_UV_CACHE_DIR", "/Users/hostile/cache")
    monkeypatch.setenv("MDB_WORKER_TORCH_HOME", "/Users/hostile/torch")
    monkeypatch.setenv("MDB_WORKER_PYTHON", "0.0.0")
    config = {"context": {"seed": 83}, "branches": [{}] * 9, "block_indices": [6, 7, 8],
              "max_adapter_steps": 237, "max_acts": 18, "model": "fixture-only"}
    request = {"config": config, "execution_environment": environment, "source_hashes": {}, "needs": {"cuda": True},
               "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 10}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    (tmp_path / "request.json").write_text(json.dumps(request))
    (tmp_path / "plan.json").write_text(json.dumps(config))
    (tmp_path / "payload").mkdir()
    captured = {}
    def launch(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return "fixture-no-model"
    monkeypatch.setattr(mdb_lanes.job_support, "guarded_mrun_launch", launch)
    launcher["submit_staged"](tmp_path)
    command = captured["command"]
    assert "UV_CACHE_DIR=" + environment["uv_cache_dir"] in command
    assert "TORCH_HOME=" + environment["torch_home"] in command
    assert "LLM_MODELS_EXTRA_ROOT=" + environment["extra_model_roots"] in command
    assert command[command.index("--python") + 1] == environment["python"]
    assert not any("/Users/hostile" in value for value in command)
    assert captured["kwargs"]["request"] == request


def test_full_fused_weight_storage_rejects_inherited_member_bound_without_allocation():
    plan = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))["storage_plan"]
    with pytest.raises(ValueError, match="transport member bound"):
        plan({"transport_limits": {"max_expanded_bytes": 3072 << 20}})
    value = plan({"transport_limits": {"max_member_bytes": 256 << 20, "max_expanded_bytes": 3072 << 20}})
    assert value["members"]["fused_weight"]["bytes"] == 169869440
    assert value["members"]["fused_weight"]["header_bytes"] == 128
    assert value["two_block_parameter_bytes_lower_bound"] == 2 * (169869440 + 75497600)


def test_parameter_expanded_lower_bound_is_an_integrity_check():
    plan = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))["storage_plan"]
    with pytest.raises(ValueError, match="expanded transport bound"):
        plan({"transport_limits": {"max_member_bytes": 256 << 20, "max_expanded_bytes": 400 << 20}})


def test_three_block_storage_charges_all_actual_parameter_members():
    plan = runpy.run_path(str(HERE / "submit_flux_transition_chain.py"))["storage_plan"]
    limits = {"max_member_bytes": 256 << 20, "max_expanded_bytes": 3072 << 20, "max_archive_bytes": 2048 << 20}
    value = plan({"block_indices": [6, 7, 8], "transport_limits": limits})
    assert value["parameter_bytes_lower_bound"] == 3 * (169869440 + 75497600)
    assert "two_block_parameter_bytes_lower_bound" not in value
    assert value["max_archive_bytes"] == 2048 << 20
    with pytest.raises(ValueError, match="expanded transport bound"):
        plan({"block_indices": [6, 7, 8], "transport_limits": {**limits, "max_expanded_bytes": 600 << 20}})
    with pytest.raises(ValueError, match="unqualified chain storage geometry"):
        plan({"block_indices": [6, 8], "transport_limits": limits})


def test_actual_member_inventory_keeps_bounded_enforcement(tmp_path):
    custody = runpy.run_path(str(HERE / "flux_family_custody.py"))
    (tmp_path / "test.bin").write_bytes(b"123456789")
    with pytest.raises(custody["CustodyError"], match="member exceeds"):
        custody["_inventory"](tmp_path, custody["_limits"]({"max_member_bytes": 8}))
    inventory = custody["_inventory"](tmp_path, custody["_limits"]({"max_member_bytes": 16}))
    assert inventory["test.bin"]["bytes"] == 9

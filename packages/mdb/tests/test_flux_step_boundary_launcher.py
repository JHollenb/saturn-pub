"""No-model qualification for finite staging, source intent and audit math."""
import json
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"
LAUNCH = runpy.run_path(str(HERE / "submit_flux_step_boundary.py"))
MATH = runpy.run_path(str(HERE / "flux_step_boundary_audit_math.py"))


def test_storage_accounts_for_fourteen_complete_parameter_sets_and_outputs():
    c = {"transport_limits": {"max_expanded_bytes": 8 << 30, "max_member_bytes": 1 << 30},
        "transaction_max_bytes": 768 << 20}
    result = LAUNCH["storage_plan"](c)
    assert result["parameter_bytes_lower_bound"] == 3435138560
    assert result["complete_output_bytes_lower_bound"] == 264242944
    assert result["transaction_max_bytes"] < result["max_member_bytes"]
    c["transport_limits"]["max_member_bytes"] = 128 << 20
    with pytest.raises(ValueError, match="custody bounds"):
        LAUNCH["storage_plan"](c)


def test_worker_command_uses_sealed_linux_paths_under_hostile_ambient_intent():
    from mdb.execution_environment import intent

    environment = intent()
    environment.update(uv_cache_dir="/mnt/hdd8tb/mdb/uv-cache", torch_home="/mnt/hdd8tb/mdb/torch",
        extra_model_roots="/mnt/big/llm-models", python="3.14.6")
    request = {"execution_environment": environment}
    with patch.dict("os.environ", {"MDB_WORKER_UV_CACHE_DIR": "/Users/test/mac-cache",
            "MDB_WORKER_TORCH_HOME": "/Users/test/torch", "MDB_WORKER_PYTHON": "9.9"}):
        command = LAUNCH["sealed_worker_command"](request)
    assert "UV_CACHE_DIR=/mnt/hdd8tb/mdb/uv-cache" in command
    assert "TORCH_HOME=/mnt/hdd8tb/mdb/torch" in command
    assert command[command.index("--python") + 1] == "3.14.6"
    assert command[-1] == "flux_step_boundary_campaign.py"
    assert all("/Users/" not in item for item in command)


def test_worker_command_rejects_mac_paths():
    with pytest.raises(ValueError, match="worker-native"):
        LAUNCH["sealed_worker_command"]({"execution_environment": {"uv_cache_dir": "/Users/a", "torch_home": "/mnt/cache"}})


def test_independent_flow_reproduces_hostile_weak_scalar_example():
    noise = np.array([[[.001983642578125]]], dtype=np.float32)
    zero = np.zeros_like(noise)
    result = MATH["independent_flow_commit"](zero, noise, np.float32(1), np.float32(2/3))
    assert result["latent"].item() == -.00066375732421875
    wide = MATH["bf16"](np.float32(result["dt_fp32"]) * noise)
    assert wide.item() == -.000659942626953125
    assert not np.array_equal(wide, result["latent"])


@pytest.mark.parametrize("seed", [1, 3, 7])
def test_separate_readout_recurrence_matches_campaign_tiny_full_fields(seed):
    reference = runpy.run_path(str(HERE / "flux_step_boundary_reference.py"))
    rng = np.random.default_rng(seed)
    b = MATH["bf16"]
    image, temb = b(rng.normal(size=(1, 5, 8))), b(rng.normal(size=(1, 6)))
    params = {"modulation_weight": b(rng.normal(size=(16, 6)) / 3),
        "modulation_bias": b(rng.normal(size=16) / 3),
        "projection_weight": b(rng.normal(size=(4, 8)) / 3),
        "projection_bias": b(rng.normal(size=4) / 3), "eps": 1e-6}
    separate = MATH["independent_readout"](image, temb, **params)
    campaign = reference["predict_readout"](image, temb, **params, neuron_chunk=3, retain_fields=True)
    assert np.array_equal(separate["noise"].view(np.uint32), campaign["noise"].view(np.uint32))
    assert np.array_equal(separate["normalized"], campaign["layernorm"])
    assert np.array_equal(separate["affine"], campaign["modulated"])


def test_submit_checks_executing_launcher_identity_before_admission(tmp_path):
    import hashlib

    request = {"launcher_source_sha256": "0" * 64, "source_hashes": {}, "independent_math_sources": {}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    (tmp_path / "request.json").write_text(json.dumps(request))
    with patch("mdb.job_support.guarded_mrun_launch") as launch:
        with pytest.raises(ValueError, match="launcher changed"):
            LAUNCH["submit_staged"](tmp_path)
        launch.assert_not_called()

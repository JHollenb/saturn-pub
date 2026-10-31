"""Check diffusion admission and explicit archive bounds without a model run."""
import json
import runpy
from pathlib import Path

import pytest

import mdb.job_support
import mdb.payload


@pytest.mark.parametrize("model_free", [False, True])
def test_flux_submission_preserves_device_and_finite_contract(tmp_path, monkeypatch, model_free):
    root = Path(__file__).parents[1] / "qualification"
    module = runpy.run_path(str(root / "submit_flux_mlp_forecast.py"))
    staged = {}

    def stage_payload(path, **kwargs):
        staged.update(kwargs)
        assert kwargs["request"]["config"] == json.loads(kwargs["config_bytes"]) == kwargs["config"]

    def launch(command, **kwargs):
        request = kwargs["request"]
        assert request == staged["request"]
        assert kwargs["preflight"] is True and kwargs["retry_on_kill"] is False
        assert kwargs["gpu"] is (not model_free)
        assert request["needs"] == ({"cpu": True} if model_free else {"cuda": True})
        assert request["scheduling_contract"]["model_loads"] == (0 if model_free else 1)
        assert request["scheduling_contract"]["logical_operations"] <= request["scheduling_contract"]["max_local_work_items"]
        if model_free:
            assert request["scheduling_contract"]["task_family"] == "analysis"
        else:
            assert len(request["config"]["arms"]) * len(request["config"]["prompts"]) == 12
            assert request["config"]["max_adapter_steps"] == 2048
        return "job-test-flux"

    monkeypatch.setattr(mdb.payload, "stage_worker", stage_payload)
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["python", name])
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", launch)
    inventory = None if model_free else {"model_path": "/frozen/model", "checkpoint_sha256": "1" * 64}
    assert module["stage"](tmp_path / "run", inventory=inventory)["job_id"] == "job-test-flux"


@pytest.mark.parametrize("limit", [64 << 20, 512 << 20])
def test_explicit_archive_allowance_keeps_small_archive_bound(tmp_path, capsys, limit):
    helper = runpy.run_path(str(Path(__file__).parents[1] / "qualification/discovery_upgrades.py"))
    (tmp_path / "evidence.json").write_text("{}")
    helper["emit"](tmp_path, {"status": "tiny"}, max_compressed_bytes=limit)
    assert helper["ARCHIVE"] in capsys.readouterr().out


def test_archive_allowance_stays_finite_before_collection(tmp_path):
    helper = runpy.run_path(str(Path(__file__).parents[1] / "qualification/discovery_upgrades.py"))
    with pytest.raises(ValueError, match="512 MiB"):
        helper["collect"](tmp_path, "unused", max_compressed_bytes=1024 << 20)

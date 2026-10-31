"""Prevent launcher/worker identity drift without a model or scheduler call."""
import hashlib
import json
import runpy
from pathlib import Path

import pytest

import mdb.job_support
import mdb.payload


@pytest.mark.parametrize("full_readout", [False, True])
def test_staged_config_is_inside_request_and_matches_exact_bytes(tmp_path, monkeypatch, full_readout):
    module = runpy.run_path(str(Path(__file__).parents[1] / "qualification/submit_qwen_mlp_forecast.py"))
    observed = {}

    def fake_stage(staging, **kwargs):
        config, request, raw = kwargs["config"], kwargs["request"], kwargs["config_bytes"]
        assert request["config"] == config == json.loads(raw)
        assert request["config_bytes_sha256"] == hashlib.sha256(raw).hexdigest()
        staged_names = {name for _, name in kwargs["additional_files"]}
        assert ("qwen_readout_law.py" in staged_names) is full_readout
        if full_readout:
            assert config["full_vocabulary_readout"] is True
            assert len(config["prompts"]) == 2 and len(config["arms"]) == 5
            assert "544538624" in request["resource_basis"]
        observed["request"] = request

    def fake_launch(command, **kwargs):
        assert kwargs["request"] == observed["request"]
        assert kwargs["task"] == "recorder"  # capture copies need mrun's measured recorder sizing
        assert kwargs["gpu"] is False and kwargs["needs"] == {"cpu": True}
        assert kwargs["preflight"] is True and kwargs["retry_on_kill"] is False
        assert "ram_mb" not in kwargs  # intent-first model planning
        return "job-local-mechanics"

    monkeypatch.setattr(mdb.payload, "stage_worker", fake_stage)
    monkeypatch.setattr(mdb.payload, "worker_command", lambda name: ["python", name])
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", fake_launch)
    destination = tmp_path / "run"
    assert module["stage"](destination, full_readout=full_readout)["job_id"] == "job-local-mechanics"
    assert json.loads((destination / "request.json").read_text())["config"] == module["plan"](full_readout=full_readout)

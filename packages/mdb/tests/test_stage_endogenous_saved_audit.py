"""Native request/command bindings for the finite saved-only reader."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

S=runpy.run_path(str(Path(__file__).parents[1]/"qualification/stage_endogenous_saved_audit.py"))


def parent():
    request=dict(config=dict(schema="mdb-flux-endogenous-write-v1"),source_hashes={"flux_endogenous_write_campaign.py":S["PRODUCER_SHA256"]})
    request["request_sha256"]=S["identity"](request);return request


def test_exact_native_source_and_engine_request_custody():
    from mdb.job_support import request_sha256

    request=parent();S["validate_parent"](request);command=["actual","native","command"]
    job=dict(config=dict(saturn_request_sha256=request_sha256(request)),result=dict(diagnostics=dict(command=command)))
    assert request_sha256(request)!=request["request_sha256"]
    S["validate_native_custody"](request,job,command)
    wrong=copy.deepcopy(job);wrong["config"]["saturn_request_sha256"]=request["request_sha256"]
    with pytest.raises(ValueError,match="command/request custody"):S["validate_native_custody"](request,wrong,command)
    with pytest.raises(ValueError,match="command/request custody"):S["validate_native_custody"](request,job,command+["changed"])
    request["source_hashes"]["flux_endogenous_write_campaign.py"]="f"*64;request["request_sha256"]=S["identity"](request)
    with pytest.raises(ValueError,match="exact endogenous-write parent"):S["validate_parent"](request)


def test_installed_cpu_worker_command(monkeypatch):
    monkeypatch.setitem(S["command"].__globals__,"bind",lambda _env:None)
    command=S["command"](dict(execution_environment={}))
    assert command[-2:]==["--config","saturn_run_config.json"] and "CUDA_VISIBLE_DEVICES=" in command
    assert command[command.index("python")+1]=="-B" and S["WORKER"] in command


def test_frozen_reader_and_metadata_only_ssh():
    here=Path(__file__).parents[1]/"qualification"
    assert S["sha"](here/S["WORKER"])==S["WORKER_SHA256"]
    compile(S["REMOTE"],"<metadata-only>","exec")
    assert "np.load" not in S["REMOTE"] and "torch" not in S["REMOTE"]

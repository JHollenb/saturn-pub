"""Source/command mechanics only; no scheduler, SSH or staging calls."""
from __future__ import annotations

import copy
import runpy
from pathlib import Path

import pytest

S=runpy.run_path(str(Path(__file__).parents[1]/"qualification/stage_centered_saved_audit.py"))


def parent_request():
    request=dict(config=dict(schema="mdb-flux-centered-write-v1"),source_hashes={"flux_centered_write_campaign.py":S["PRODUCER_SHA256"]})
    request["request_sha256"]=S["identity"](request)
    return request


def test_exact_parent_identity_and_producer_source():
    original=parent_request();S["validate_parent"](original)
    changed=copy.deepcopy(original);changed["config"]["schema"]="legacy"
    with pytest.raises(ValueError,match="exact centered-write parent"):
        S["validate_parent"](changed)
    changed["request_sha256"]=S["identity"](changed)
    with pytest.raises(ValueError,match="exact centered-write parent"):
        S["validate_parent"](changed)
    wrong=copy.deepcopy(original);wrong["source_hashes"]["flux_centered_write_campaign.py"]="a"*64;wrong["request_sha256"]=S["identity"](wrong)
    with pytest.raises(ValueError,match="exact centered-write parent"):
        S["validate_parent"](wrong)


def test_actual_installed_worker_command_explicit_config_cpu_only(monkeypatch):
    monkeypatch.setitem(S["command"].__globals__,"bind",lambda _env:None)
    args=S["command"](dict(execution_environment={}))
    assert args[-2:]==["--config","saturn_run_config.json"] and args[args.index("python")+1]=="-B"
    assert "CUDA_VISIBLE_DEVICES="in args
    assert all(name+"=4"in args for name in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
    assert S["WORKER"]in args


def test_real_reader_pin_and_metadata_only_remote_program():
    reader=Path(__file__).parents[1]/"qualification"/S["WORKER"]
    assert S["sha"](reader)==S["WORKER_SHA256"]
    compile(S["REMOTE"],"<metadata-only-read>","exec")
    assert "summary.json"in S["REMOTE"] and "files.json"in S["REMOTE"]
    assert "np.load"not in S["REMOTE"] and "torch"not in S["REMOTE"]


def test_actual_published_parent_engine_hash_distinct_from_experiment_id():
    from mdb_lanes.job_support import request_sha256

    fixture=S["read"](Path(__file__).parents[1]/"fixtures/centered-native-custody.json")
    parent,job,command=(fixture[key]for key in("request","job","command"))
    S["validate_parent"](parent)
    assert parent["request_sha256"]=="badd4880754550c73743836a7b35a00acfbcfd15a825ffb03641433d7ec09927"
    assert request_sha256(parent)=="7ce50091c870b6d1c22cc9aa0d01a544ccd2f0bae4e07fba45434e16318fa88b"
    assert request_sha256(parent)!=parent["request_sha256"]
    S["validate_native_custody"](parent,job,command)
    wrong=copy.deepcopy(job);wrong["config"]["saturn_request_sha256"]=parent["request_sha256"]
    with pytest.raises(ValueError,match="command/request custody"):
        S["validate_native_custody"](parent,wrong,command)
    with pytest.raises(ValueError,match="command/request custody"):
        S["validate_native_custody"](parent,job,command+["changed"])

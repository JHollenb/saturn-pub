"""Model-free API, original-layout metadata and installed bootstrap transport."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"
PATH = HERE / "stage_qwen_cached_qkv.py"
S = runpy.run_path(str(PATH))


def test_final_source_closure_and_complete_metadata_api_without_tensor_reads():
    worker = S["producer"]()
    assert len(worker["HELPERS"]) == 11 and "parent_forecast" in worker["PATH_KEYS"]
    base = {key: key + ".json" for key in worker["PATH_KEYS"]}
    base.update({key + "_sha256": "0" * 64 for key in worker["PATH_KEYS"]})
    base.update(parameter_root="/synthetic/not-read", input_root="/synthetic/not-read",
        selection_protocol_sha256=S["PROTOCOL_SHA256"], qwen_parameter_manifest_sha256=S["SUBSET_SHA256"],
        retrieval_script_sha256="7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae")
    config = worker["make_config"](base)
    counts = worker["validate_config"](config)
    assert counts["cached_SDPA"] == counts["functional_SDPA"] == 2 and counts["native_block_calls"] == 3
    assert counts["cached_linear"] == 8 and counts["functional_linear"] == 14 and counts["native_linear"] == 21
    assert counts["full_model"] == counts["LM"] == counts["tokenizer"] == counts["forecast_module_attempts"] == 0
    del config["parent_forecast"]
    with pytest.raises(ValueError, match="missing source path"):
        worker["validate_config"](config)


def test_original_noncontiguous_qkv_metadata_is_preserved_and_joined():
    q = dict(path="absent/q.npy", sha256="0" * 64, bytes=100, shape=[1, 4, 8, 2], stride=[64, 2, 8, 1], logical_dtype="torch.bfloat16")
    protocol = dict(source_arrays={"q": q})
    files = {q["path"]: dict(bytes=q["bytes"], sha256=q["sha256"])}
    wrapper, seal = dict(files=files), dict(declarations=dict(frontier=dict(external_files=files)))
    positive = {"q": copy.deepcopy(q)}
    S["source_descriptors"](protocol, wrapper, seal, positive)
    assert q["stride"] == [64, 2, 8, 1]
    positive["q"]["stride"] = [64, 16, 2, 1]
    with pytest.raises(ValueError, match="descriptor/layout"):
        S["source_descriptors"](protocol, wrapper, seal, positive)
    positive["q"] = copy.deepcopy(q)
    with pytest.raises(ValueError, match="raw/global seal"):
        S["source_descriptors"](protocol, wrapper, dict(declarations=dict(frontier=dict(external_files={}))), positive)


def test_actual_job_schema_resource_basis_not_a_successor_measurement(tmp_path, monkeypatch):
    job = dict(job_id="job-74efe5506ecf", state="succeeded", config=dict(runtime_wheels={"synthetic.whl": "0" * 64}, execution_environment={"synthetic": True}),
        result=dict(returncode=0, elapsed_s=47.22, peak_rss_mb=1795.4, peak_vram_mb=826.0))
    path = tmp_path / "actual-job-schema.json"
    path.write_text(json.dumps(job))
    monkeypatch.setitem(S["measured_component"].__globals__, "RESOURCE_JOB_SHA256", S["utilities"]()["sha"](path))
    measurement = S["measured_component"](path, job["config"])
    assert measurement["elapsed_s"] == 47.22 and "unmeasured" in measurement["scope"]
    wrong = {**job["config"], "runtime_wheels": {}}
    with pytest.raises(ValueError, match="source environment"):
        S["measured_component"](path, wrong)


def test_installed_bootstrap_config_transport_no_model(tmp_path):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.payload import seal_package

    env_intent = {**intent(), "tmpdir": "/tmp", "cuda_cache_path": "/tmp", "torchinductor_cache_dir": "/tmp"}
    args = S["command"]({"execution_environment": env_intent})
    assert args[-3:] == [S["WORKER"], "--config", "saturn_run_config.json"] and "--offline" in args
    worker = tmp_path / S["WORKER"]
    worker.write_text("import argparse,json\np=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps({'config':a.config,'scope':'dummy-no-model'}))\n")
    (tmp_path / "saturn_run_config.json").write_text("{}")
    (tmp_path / "saturn_debug_request.json").write_text("{}")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "payload-seal.json").write_text(json.dumps(dict(packages={n: seal_package(n) for n in ("mdb", "mdb_lanes", "saturn_pub")},
        wheels={}, files={}, worker=worker.name, worker_sha256=sha(worker), config_bytes_sha256=sha(tmp_path / "saturn_run_config.json"))))
    environment = {**os.environ, "SATURN_RUN_CONFIG": "saturn_run_config.json", "SATURN_DEBUG_REQUEST": "saturn_debug_request.json"}
    result = subprocess.run([sys.executable, "-I", *args[args.index("python") + 1:]], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(config="saturn_run_config.json", scope="dummy-no-model")


def test_help_and_finite_scope_do_not_stage(tmp_path):
    assert len(S["QUEUE_NOTE"]) <= 500 and "60s estimate/240s assay/300s outer" in S["QUEUE_NOTE"]
    for args, code in ((["--help"], 0), ([str(tmp_path / "fresh")], 2)):
        result = subprocess.run([sys.executable, "-I", "-B", str(PATH), *args], cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == code, result.stderr
    assert not (tmp_path / "fresh").exists()


def test_payload_request_config_and_finite_review_guards(tmp_path, monkeypatch):
    h = S["utilities"]()
    payload = tmp_path / "payload"; payload.mkdir()
    (payload / S["WORKER"]).write_text("synthetic; no execution")
    digest = h["sha"](payload / S["WORKER"])
    monkeypatch.setitem(S["inspect_run"].__globals__, "WORKER_SHA256", digest)
    request = dict(config={"synthetic": True}, launcher_sha256=h["sha"](PATH), source_hashes={S["WORKER"]: digest},
        runtime_wheels={}, resource_exception=S["RESOURCE"])
    request["request_sha256"] = h["identity"](request)
    for name, value in (("request.json", request), ("payload/saturn_debug_request.json", request), ("payload/saturn_run_config.json", request["config"])):
        (tmp_path / name).write_text(json.dumps(value))
    (payload / "payload-seal.json").write_text(json.dumps(dict(files=request["source_hashes"], wheels={}, worker=S["WORKER"],
        worker_sha256=digest, config_bytes_sha256=h["sha"](payload / "saturn_run_config.json"))))
    assert S["inspect_run"](tmp_path) == request
    with pytest.raises(FileNotFoundError, match="finite-run-review"):
        S["submit"](tmp_path)
    (payload / "saturn_run_config.json").write_text("{}")
    with pytest.raises(ValueError, match="request/config"):
        S["inspect_run"](tmp_path)
    assert not (tmp_path / "submission.json").exists()

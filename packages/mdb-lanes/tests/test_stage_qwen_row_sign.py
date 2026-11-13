"""Synthetic metadata and installed bootstrap checks; no real stage or service."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1] / "qualification"
PATH = HERE / "stage_qwen_row_sign.py"
S = runpy.run_path(str(PATH))


def metadata_fixture(root, monkeypatch):
    h = S["utilities"]()
    parent, compact, selection, frontier = (root / name for name in ("parent", "compact", "selection", "frontier"))
    for path in (parent / "payload", compact, selection, frontier / "transaction"):
        path.mkdir(parents=True)

    def save(path, value):
        h["save"](path, value)
        return h["sha"](path)

    stream = io.BytesIO()
    np.save(stream, np.zeros(1, dtype="<u2"), allow_pickle=False)
    npy = stream.getvalue()
    rows = []
    for index in range(398):
        tensor = ("model.layers.18." if index < 11 else "other.") + str(index)
        rows.append(dict(path=f"parameters/{index}.npy", bytes=len(npy), sha256=hashlib.sha256(npy).hexdigest(),
            npy_prefix_b64=base64.b64encode(npy[:-2]).decode(), source=dict(file="synthetic.safetensors",
                file_sha256="1" * 64, tensor=tensor, dtype="BF16", shape=[1], data_offsets=[2 * index, 2 * index + 2], payload_sha256=hashlib.sha256(npy[-2:]).hexdigest())))
    full = dict(schema="mdb-parameter-artifacts-v1", artifacts=rows)
    full_sha = save(parent / "payload/qwen-parameter-artifacts.json", full)
    subset_sha = save(selection / "qwen-block18-parameter-artifacts.json", {**full, "artifacts": rows[:11]})
    sources = {}
    for name in ("qwen-config.json", "qwen-source-binding.json", "retrieve_qwen_weights.py"):
        sources[name] = save(parent / "payload" / name, {"synthetic": name})
    sources["qwen-parameter-artifacts.json"] = full_sha
    request = dict(config={}, source_hashes=sources)
    request["request_sha256"] = h["identity"](request)
    save(parent / "request.json", request)
    save(compact / "source-request.json", request)
    seal_sha = save(frontier / "transaction/seal.json", {"synthetic": "seal"})
    arrays = {str(i): dict(path=f"absent-arrays/{i}.npy", sha256=str(i % 10) * 64, bytes=123) for i in range(13)}
    wrapper = dict(job_id="job-synthetic", request_sha256=request["request_sha256"],
                   files={row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in arrays.values()})
    wrapper_sha = save(compact / "source-files.json", wrapper)
    protocol = dict(parent_job="job-synthetic", parent_global_seal_sha256=seal_sha,
        parent_raw_manifest_sha256=wrapper_sha, source_arrays=arrays)
    protocol_sha = save(selection / "protocol.json", protocol)
    save(compact / "source-summary.json", dict(forecast_seal=seal_sha))
    for name in ("source-source.json", "source-runtime.json"):
        save(compact / name, {"synthetic": name})
    compact_sha = save(compact / "files.json", {p.name: h["sha"](p) for p in compact.iterdir()})
    save(selection / "parameter-selection.json", dict(subset_manifest_sha256=subset_sha,
        parent_manifest_sha256=full_sha, protocol_sha256=protocol_sha))
    derived_sha = save(selection / "derived-predictions.json", {"synthetic": "hypotheses-not-gates"})
    for key, value in dict(PROTOCOL_SHA256=protocol_sha, SUBSET_SHA256=subset_sha,
        PARENT_MANIFEST_SHA256=full_sha, PARENT_REQUEST=request["request_sha256"],
        COMPACT_SHA256=compact_sha, DERIVED_SHA256=derived_sha).items():
        monkeypatch.setitem(S["metadata"].__globals__, key, value)
    return parent, compact, selection, frontier


def test_metadata_proves_exact_subset_and_inherited_files_without_reading_arrays(tmp_path, monkeypatch):
    args = metadata_fixture(tmp_path, monkeypatch)
    parent, protocol, paths = S["metadata"](*args)
    assert len(protocol["source_arrays"]) == 13 and len(paths) == 11
    assert paths["parent_seal"] == args[3] / "transaction/seal.json"
    assert parent["request_sha256"] == S["metadata"].__globals__["PARENT_REQUEST"]
    assert not (tmp_path / "absent-arrays").exists()
    # A coherent but nonoriginal subset row cannot acquire legitimacy from its own new hash.
    subset = paths["qwen_parameter_manifest"]
    value = json.loads(subset.read_text())
    value["artifacts"][0]["source"]["tensor"] += ".wrong"
    subset.write_text(json.dumps(value))
    monkeypatch.setitem(S["metadata"].__globals__, "SUBSET_SHA256", S["utilities"]()["sha"](subset))
    with pytest.raises(ValueError, match="Subset entries"):
        S["metadata"](*args)


def test_source_descriptor_or_seal_tampering_is_refused(tmp_path, monkeypatch):
    args = metadata_fixture(tmp_path, monkeypatch)
    protocol_path = args[2] / "protocol.json"
    value = json.loads(protocol_path.read_text())
    value["source_arrays"]["0"]["bytes"] += 1
    protocol_path.write_text(json.dumps(value))
    monkeypatch.setitem(S["metadata"].__globals__, "PROTOCOL_SHA256", S["utilities"]()["sha"](protocol_path))
    with pytest.raises(ValueError, match="Inherited array manifest"):
        S["metadata"](*args)


def test_installed_worker_bootstrap_delivers_explicit_config(tmp_path):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.payload import seal_package

    environment = {**intent(), "tmpdir": "/tmp", "cuda_cache_path": "/tmp", "torchinductor_cache_dir": "/tmp"}
    args = S["command"]({"execution_environment": environment})
    assert "--offline" in args and args[-3:] == [S["WORKER"], "--config", "saturn_run_config.json"]
    (tmp_path / S["WORKER"]).write_text("import argparse,json\np=argparse.ArgumentParser();p.add_argument('--config',required=True)\na=p.parse_args();print(json.dumps({'received':a.config,'kind':'dummy-no-model'}))\n")
    (tmp_path / "saturn_run_config.json").write_text("{}")
    (tmp_path / "saturn_debug_request.json").write_text("{}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "payload-seal.json").write_text(json.dumps(dict(
        packages={name: seal_package(name) for name in ("mdb", "mdb_lanes", "saturn_pub")}, wheels={}, files={},
        worker=S["WORKER"], worker_sha256=digest(tmp_path / S["WORKER"]),
        config_bytes_sha256=digest(tmp_path / "saturn_run_config.json"))))
    # Exercise the installed MDB entrypoint, skipping uv/environment orchestration only.
    actual = [sys.executable, "-I", *args[args.index("python") + 1:]]
    env = {**os.environ, "SATURN_RUN_CONFIG": "saturn_run_config.json", "SATURN_DEBUG_REQUEST": "saturn_debug_request.json"}
    result = subprocess.run(actual, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"received": "saturn_run_config.json", "kind": "dummy-no-model"}


def test_exact_producer_closure_and_finite_queue_note():
    namespace = S["producer"]()
    assert namespace["SCHEMA"] == "mdb-qwen-row-sign-campaign-v1"
    assert "qwen_sign_shortcut_campaign.py" in namespace["HELPERS"]
    base = {key: key + ".json" for key in namespace["PATH_KEYS"]}
    base.update({key + "_sha256": "0" * 64 for key in namespace["PATH_KEYS"]})
    base.update(parameter_root="/synthetic/model-not-read", input_root="/synthetic/arrays-not-read",
        selection_protocol_sha256=namespace["PROTOCOL_SHA256"], qwen_parameter_manifest_sha256=namespace["SUBSET_SHA256"],
        retrieval_script_sha256="7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae")
    config = namespace["make_config"](base)
    counts = namespace["validate_config"](config)
    assert counts["functional_blocks"] == counts["shortcut_down_projections"] == 2 and counts["native_blocks"] == 3
    assert counts["tokenizer"] == counts["LM"] == counts["full_model"] == 0
    assert config["parent_seal"] == base["parent_seal"] and config["max_wall_seconds"] == 240
    del config["parent_seal"]
    with pytest.raises(ValueError, match="missing source path"):
        namespace["validate_config"](config)
    assert len(S["QUEUE_NOTE"]) <= 500 and "60s estimate/240s assay/300s outer" in S["QUEUE_NOTE"]
    assert S["RESOURCE"] == dict(cpu_threads=4, ram_mb=4096, vram_mb=2048, disk_gb=3, est_wall_s=60, timeout_s=300)


def test_inspection_refuses_file_drift_before_any_submission(tmp_path, monkeypatch):
    h = S["utilities"]()
    payload = tmp_path / "payload"
    payload.mkdir()
    files = {S["WORKER"]: b"synthetic-worker-no-model", S["UTILITIES"]: b"synthetic-utilities", "synthetic.whl": b"synthetic-wheel"}
    for name, value in files.items():
        (payload / name).write_bytes(value)
    monkeypatch.setitem(S["inspect_run"].__globals__, "WORKER_SHA256", h["sha"](payload / S["WORKER"]))
    # The actual utilities source remains checked by utilities(); only the declaration is synthetic here.
    request = dict(config={"synthetic": True}, launcher_sha256=h["sha"](PATH), source_hashes={
        S["WORKER"]: h["sha"](payload / S["WORKER"]), S["UTILITIES"]: S["UTILITIES_SHA256"]},
        runtime_wheels={"synthetic.whl": h["sha"](payload / "synthetic.whl")}, resource_exception=S["RESOURCE"])
    (payload / S["UTILITIES"]).write_bytes((HERE / S["UTILITIES"]).read_bytes())
    request["request_sha256"] = h["identity"](request)
    (tmp_path / "request.json").write_text(json.dumps(request))
    (payload / "saturn_debug_request.json").write_text(json.dumps(request))
    (payload / "saturn_run_config.json").write_text(json.dumps(request["config"]))
    (payload / "payload-seal.json").write_text(json.dumps(dict(files=request["source_hashes"], wheels=request["runtime_wheels"],
        worker=S["WORKER"], worker_sha256=request["source_hashes"][S["WORKER"]], config_bytes_sha256=h["sha"](payload / "saturn_run_config.json"))))
    assert S["inspect_run"](tmp_path) == request
    with pytest.raises(FileNotFoundError, match="finite-run-review"):
        S["submit"](tmp_path)
    assert not (tmp_path / "submission.json").exists()
    (payload / "synthetic.whl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Payload source/wheel"):
        S["inspect_run"](tmp_path)


def test_cli_help_and_missing_inputs_do_not_stage(tmp_path):
    for args, code in ((["--help"], 0), ([str(tmp_path / "fresh")], 2)):
        result = subprocess.run([sys.executable, "-I", "-B", str(PATH), *args], cwd=tmp_path,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == code, result.stderr
    assert not (tmp_path / "fresh").exists()

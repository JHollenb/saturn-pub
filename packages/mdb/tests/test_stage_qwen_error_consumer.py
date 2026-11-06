"""Two-root metadata, finite launch and installed bootstrap; no original stage."""
from __future__ import annotations

import base64
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "qualification"
PATH = HERE / "stage_qwen_error_consumer.py"
S = runpy.run_path(str(PATH))


def metadata_fixture(root, monkeypatch):
    h = S["utilities"]()
    world, context, worlds, contexts, selection = (root / name for name in ("world-run", "context-run", "world-compact", "context-compact", "selection"))
    for path in (world / "payload", context / "payload", worlds, contexts, selection):
        path.mkdir(parents=True)

    def save(path, value):
        h["save"](path, value)
        return h["sha"](path)

    header = b"{'descr': '<u2', 'fortran_order': False, 'shape': (1,), }"
    header = header + b" " * (117 - len(header)) + b"\n"
    prefix = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header
    rows = [dict(path=f"parameters/{index}.npy", bytes=130, sha256="0" * 64,
        npy_prefix_b64=base64.b64encode(prefix).decode(), source=dict(file="synthetic.safetensors", file_sha256="1" * 64,
            tensor=("model.layers.19." if index < 11 else "other.") + str(index), dtype="BF16", shape=[1],
            data_offsets=[2 * index, 2 * index + 2], payload_sha256="2" * 64)) for index in range(398)]
    full = dict(schema="mdb-parameter-artifacts-v1", artifacts=rows)
    full_hash = save(context / "payload/qwen-parameter-artifacts.json", full)
    subset_hash = save(selection / "subset.json", dict(full, artifacts=rows[:11]))
    sources = {"qwen-parameter-artifacts.json": full_hash}
    for name in ("qwen-config.json", "qwen-source-binding.json", "retrieve_qwen_weights.py"):
        sources[name] = save(context / "payload" / name, dict(synthetic=name))
    requests = {}
    for label, run in (("world", world), ("context", context)):
        request = dict(config=dict(parameter_root="/absent-weights"), source_hashes=sources if label == "context" else {},
            execution_environment=dict(synthetic=True), scheduler_url="http://not-contacted.invalid", runtime_wheels={})
        request["request_sha256"] = h["identity"](request)
        save(run / "request.json", request); requests[label] = request
    world_rows = {name: dict(path="absent-worlds/" + name + ".npy", bytes=123, sha256=str(index) * 64,
        shape=[1, 512, 2560], stride=[1310720, 2560, 1]) for index, name in enumerate(("full-input", "cheap-input"))}
    context_rows = {name: dict(path="absent-context/" + name.replace(":", "-") + ".npy", bytes=456, sha256="3" * 64)
        for name in ("input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq")}
    for name, arm in (("source-forecast.json", "full-input"), ("source-shortcut.json", "cheap-input")):
        save(worlds / name, dict(arms={"prefix-negative": dict(ports=dict(Hnext=world_rows[arm]))}))
    roots = {}
    seals = {}
    for label, compact, records in (("world", worlds, world_rows), ("context", contexts, context_rows)):
        files = {row["path"]: dict(bytes=row["bytes"], sha256=row["sha256"]) for row in records.values()}
        seal_path = root / (label + "-seal.json")
        seal_hash = save(seal_path, dict(declarations=dict(frontier=dict(external_files=files))))
        seals[label] = seal_path
        if label == "world":
            for copied, original in (("source-forecast.json", "forecast.json"), ("source-shortcut.json", "shortcut.json")):
                files[original] = dict(bytes=(compact / copied).stat().st_size, sha256=h["sha"](compact / copied))
        wrapper = dict(job_id="job-synthetic-" + label, request_sha256=requests[label]["request_sha256"], files=files)
        raw_hash = save(compact / "source-files.json", wrapper)
        roots[label] = dict(job_id=wrapper["job_id"], request_sha256=wrapper["request_sha256"], raw_manifest_sha256=raw_hash,
            global_seal_sha256=seal_hash, root="/not-read/" + label)
    protocol = dict(roots=roots, world_order=["full-input", "cheap-input"], world_sources=world_rows, shared_context_sources=context_rows,
        qwen_parameter_manifest_sha256=subset_hash, parent_parameter_manifest_sha256=full_hash,
        qwen_config_sha256=sources["qwen-config.json"], qwen_source_binding_sha256=sources["qwen-source-binding.json"])
    protocol_hash = save(selection / "protocol.json", protocol)
    for key, value in dict(PROTOCOL_SHA256=protocol_hash, SUBSET_SHA256=subset_hash).items():
        monkeypatch.setitem(S["metadata"].__globals__, key, value)
    return world, worlds, seals["world"], context, contexts, seals["context"], selection, selection / "subset.json"


def test_exact_producer_source_closure_and_complete_config_abi():
    worker = S["producer"]()
    assert len(worker["PATH_KEYS"]) == 14 and worker["ROOT_KEYS"] == ("world_root", "context_root", "parameter_root")
    base = {key: key + ".json" for key in worker["PATH_KEYS"]}
    base.update({key + "_sha256": "0" * 64 for key in worker["PATH_KEYS"]})
    base.update({key: "/absent/" + key for key in worker["ROOT_KEYS"]})
    base.update(selection_protocol_sha256=S["PROTOCOL_SHA256"], qwen_parameter_manifest_sha256=S["SUBSET_SHA256"],
        retrieval_script_sha256="7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae")
    config = worker["make_config"](base); counts = worker["validate_config"](config)
    assert counts["functional_blocks"] == counts["native_block_calls"] == counts["native_authorizations"] == 2
    assert counts["functional_linear"] == counts["native_linear"] == 14 and counts["native_module_entries"] == 30
    assert counts["LM"] == counts["full_model"] == counts["forecast_module_attempts"] == 0
    del config["context_seal"]
    with pytest.raises(ValueError, match="missing source path"):
        worker["validate_config"](config)


def test_metadata_authenticates_both_roots_and_exact_original_subset(tmp_path, monkeypatch):
    args = metadata_fixture(tmp_path, monkeypatch)
    world, context, protocol, paths = S["metadata"](*args)
    assert len(paths) == 14 and protocol["roots"]["world"]["root"] != protocol["roots"]["context"]["root"]
    assert paths["world_seal"] == args[2] and paths["context_seal"] == args[5]
    assert world["request_sha256"] != context["request_sha256"] and not (tmp_path / "absent-worlds").exists()
    subset = json.loads(args[-1].read_text()); subset["artifacts"][0]["source"]["tensor"] += ".wrong"
    args[-1].write_text(json.dumps(subset))
    protocol_path = args[-2] / "protocol.json"; changed = json.loads(protocol_path.read_text())
    changed["qwen_parameter_manifest_sha256"] = S["utilities"]()["sha"](args[-1]); protocol_path.write_text(json.dumps(changed))
    monkeypatch.setitem(S["metadata"].__globals__, "SUBSET_SHA256", changed["qwen_parameter_manifest_sha256"])
    monkeypatch.setitem(S["metadata"].__globals__, "PROTOCOL_SHA256", S["utilities"]()["sha"](protocol_path))
    with pytest.raises(ValueError, match="Exact original eleven"):
        S["metadata"](*args)


@pytest.mark.parametrize("change", ["world-descriptor", "context-seal"])
def test_raw_seal_and_world_layout_joins_are_mandatory(tmp_path, monkeypatch, change):
    args = metadata_fixture(tmp_path, monkeypatch)
    h = S["utilities"](); protocol = h["read"](args[-2] / "protocol.json")
    wrappers = [h["read"](args[index] / "source-files.json") for index in (1, 4)]
    seals = [h["read"](args[index]) for index in (2, 5)]
    forecasts = [h["read"](args[1] / name) for name in ("source-forecast.json", "source-shortcut.json")]
    if change == "world-descriptor":
        forecasts[0]["arms"]["prefix-negative"]["ports"]["Hnext"]["stride"] = [1310720, 1, 512]
    else:
        seals[1]["declarations"]["frontier"]["external_files"].clear()
    with pytest.raises(ValueError, match="world descriptors|raw/global pretarget seal"):
        S["source_descriptors"](protocol, wrappers[0], seals[0], wrappers[1], seals[1], *forecasts)


def test_intent_first_and_explicit_measured_exception_are_distinct(tmp_path, monkeypatch):
    assert S["sizing"]() == dict(mode="intent-first", timeout_s=300, reservation=None)
    job = dict(job_id="job-74efe5506ecf", state="succeeded", result=dict(returncode=0, elapsed_s=47.22, peak_rss_mb=1795.4, peak_vram_mb=826.0))
    path = tmp_path / "measured-job.json"; path.write_text(json.dumps(job))
    monkeypatch.setitem(S["sizing"].__globals__, "RESOURCE_JOB_SHA256", S["utilities"]()["sha"](path))
    exception = S["sizing"](path)
    assert exception["reservation"] == S["RESOURCE"] and "unmeasured" in exception["rationale"]
    job["result"]["peak_vram_mb"] = 900; path.write_text(json.dumps(job))
    monkeypatch.setitem(S["sizing"].__globals__, "RESOURCE_JOB_SHA256", S["utilities"]()["sha"](path))
    with pytest.raises(ValueError, match="observations differ"):
        S["sizing"](path)


def test_installed_bootstrap_receives_explicit_config_without_model(tmp_path):
    from mdb.execution_environment import intent
    from mdb.payload import stage_worker

    environment = {**intent(), "tmpdir": "/tmp", "cuda_cache_path": "/tmp", "torchinductor_cache_dir": "/tmp"}
    args = S["command"]({"execution_environment": environment})
    assert "--offline" in args and args[-3:] == [S["WORKER"], "--config", "saturn_run_config.json"]
    worker = tmp_path / S["WORKER"]
    worker.write_text("import argparse,json\np=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();print(json.dumps({'config':a.config,'scope':'dummy-no-model'}))\n")
    wheels = S["runtime_wheels"]()
    payload = tmp_path / "payload"
    stage_worker(payload, worker=worker, config={}, request=dict(runtime_wheels=wheels))
    assert json.loads((payload / "payload-seal.json").read_text())["wheels"] == wheels
    env = {**os.environ, "SATURN_RUN_CONFIG": "saturn_run_config.json", "SATURN_DEBUG_REQUEST": "saturn_debug_request.json"}
    result = subprocess.run([sys.executable, "-I", *args[args.index("python") + 1:]], cwd=payload, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict(config="saturn_run_config.json", scope="dummy-no-model")


def staged_fixture(root, monkeypatch):
    h = S["utilities"](); payload = root / "payload"; payload.mkdir()
    (payload / S["WORKER"]).write_text("synthetic no execution")
    (payload / S["UTILITIES"]).write_bytes((HERE / S["UTILITIES"]).read_bytes())
    worker_sha = h["sha"](payload / S["WORKER"])
    monkeypatch.setitem(S["inspect_run"].__globals__, "WORKER_SHA256", worker_sha)
    request = dict(config=dict(synthetic=True), source_hashes={S["WORKER"]: worker_sha, S["UTILITIES"]: S["UTILITIES_SHA256"]},
        runtime_wheels={}, sizing=S["sizing"](), resource_exception=None, launcher_sha256=h["sha"](PATH),
        execution_environment={}, scheduler_url="http://not-contacted.invalid", model="component-intent", needs={"cuda": True, "host": "beast"})
    request["request_sha256"] = h["identity"](request)
    for name, value in (("request.json", request), ("payload/saturn_debug_request.json", request), ("payload/saturn_run_config.json", request["config"])):
        (root / name).write_text(json.dumps(value))
    (payload / "payload-seal.json").write_text(json.dumps(dict(files=request["source_hashes"], wheels={}, worker=S["WORKER"],
        worker_sha256=worker_sha, config_bytes_sha256=h["sha"](payload / "saturn_run_config.json"))))
    return request


@pytest.mark.parametrize("measured", [False, True])
def test_inspection_and_finite_review_precede_single_intent_launch(tmp_path, monkeypatch, measured):
    import mdb.job_support

    request = staged_fixture(tmp_path, monkeypatch)
    if measured:
        request["sizing"] = dict(mode="measured-component-exception", timeout_s=300, reservation=S["RESOURCE"].copy())
        request["resource_exception"] = S["RESOURCE"].copy()
        request["request_sha256"] = S["utilities"]()["identity"](request)
        for path in (tmp_path / "request.json", tmp_path / "payload/saturn_debug_request.json"):
            path.write_text(json.dumps(request))
    assert S["inspect_run"](tmp_path) == request
    with pytest.raises(FileNotFoundError, match="finite-run-review"):
        S["submit"](tmp_path)
    assert not (tmp_path / "submission.json").exists()
    (tmp_path / "finite-run-review-v1.json").write_text(json.dumps(dict(request_sha256=request["request_sha256"], blocking_issues=[], review_status="ready-for-finite-run")))
    calls = []
    monkeypatch.setattr(mdb.job_support, "guarded_mrun_launch", lambda *a, **k: calls.append((a, k)) or "job-synthetic-no-service")
    monkeypatch.setitem(S["submit"].__globals__, "command", lambda _request: ["dummy-no-execution"])
    assert S["submit"](tmp_path) == dict(job_id="job-synthetic-no-service")
    kwargs = calls[0][1]
    assert len(calls) == 1 and kwargs["model"] == (None if measured else "component-intent") and kwargs["gpu"] and kwargs["preflight"]
    assert not kwargs["retry_on_kill"]
    assert {key: kwargs[key] for key in S["RESOURCE"] if key in kwargs} == (S["RESOURCE"] if measured else {})
    with pytest.raises(ValueError, match="Submission already"):
        S["submit"](tmp_path)
    (tmp_path / "payload/saturn_run_config.json").write_text("{}")
    with pytest.raises(ValueError, match="request/config"):
        S["inspect_run"](tmp_path)


def test_cli_help_and_missing_metadata_do_not_stage(tmp_path):
    assert len(S["QUEUE_NOTE"]) <= 500
    for args, code in ((["--help"], 0), ([str(tmp_path / "fresh")], 2)):
        result = subprocess.run([sys.executable, "-I", "-B", str(PATH), *args], cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == code, result.stderr
    assert not (tmp_path / "fresh").exists()

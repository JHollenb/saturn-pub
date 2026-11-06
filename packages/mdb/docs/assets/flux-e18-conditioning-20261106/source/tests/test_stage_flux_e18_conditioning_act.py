"""Model-free controller/source/CLI checks; staging and admission are mocked."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import runpy
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/stage_flux_e18_conditioning_act.py"
S = runpy.run_path(str(PATH))


def manifest(family, count):
    buffer = io.BytesIO()
    np.save(buffer, np.array([0], dtype="<u2"), allow_pickle=False)
    encoded = buffer.getvalue()
    payload = encoded[-2:]
    return dict(schema="mdb-parameter-artifacts-v1", artifacts=[dict(
        path=f"parameters/{family}-{index}.npy", sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded),
        npy_prefix_b64=base64.b64encode(encoded[:-2]).decode(), source=dict(
            file=family + ".safetensors", file_sha256="f" * 64, tensor="weight." + str(index),
            dtype="BF16", shape=[1], data_offsets=[2 * index, 2 * index + 2],
            payload_sha256=hashlib.sha256(payload).hexdigest())) for index in range(count)])


def fixture(root, monkeypatch):
    import mdb.execution_environment as environment
    import mdb.payload as payload

    root.mkdir()
    parent = root / "parent"
    source = parent / "payload"
    source.mkdir(parents=True)
    utilities = S["base"]()
    helper_names = set(utilities["HELPERS"]) | {"flux_raw_to_image.py"}
    for name in helper_names:
        shutil.copyfile(HERE / "qualification" / name, source / name)
    metadata_names = set(utilities["PATHS"].values()) - helper_names
    metadata_names |= {"model_index.json", "source-metadata-receipt-v1.json"}
    for name in metadata_names:
        if "parameter-artifacts" in name:
            family = name.split("-")[0]
            value = manifest(family, {"qwen": 398, "flux": 169, "vae": 142}[family])
        elif name == "vae-config.json":
            value = dict(block_out_channels=[128, 256, 512, 512])
        else:
            value = {}
        (source / name).write_text(json.dumps(value) + "\n")
    raw = runpy.run_path(str(HERE / "qualification/flux_raw_to_image.py"))
    config = dict(schema="mdb-flux-raw-to-image-plan-v1", seed=277, prompt=raw["PROMPT"], batch_size=1,
        height=512, width=512, steps=4, device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True,
        max_gpu_components=1, component_hydrations=3, cuda_placements=6, offloads=6,
        qwen_manifest_entries=398, flux_manifest_entries=169, vae_manifest_entries=142,
        native_sources={name: "fixture-metadata" for name in raw["NATIVE_SOURCE_MODULES"]},
        qwen_rope_buffer_policy=dict(placement_dtype="float32", init_device="cpu"),
        attention_dispatch=dict(backend="sdpa", gqa_policy="repeat_kv", source="fixture-metadata"),
        cpu_threads=4, qwen_parameter_root="/never-open/original-qwen", flux_parameter_root="/never-open/original-flux",
        vae_parameter_root="/never-open/original-vae", output_scratch="/never-write/research")
    for key, name in utilities["PATHS"].items():
        config[key], config[key + "_sha256"] = name, S["sha"](source / name)
    config["helper_source_sha256"] = utilities["HELPERS"]
    env = {**environment.intent(), "tmpdir": "/synthetic/tmp", "cuda_cache_path": "/synthetic/cuda",
           "torchinductor_cache_dir": "/synthetic/inductor"}
    for name in ("MDB_WORKER_UV_CACHE_DIR", "MDB_WORKER_TORCH_HOME", "MDB_WORKER_MODEL_ROOTS", "MDB_WORKER_PYTHON", "MDB_WORKER_UV_LINK_MODE"):
        monkeypatch.setenv(name, "")
    request = dict(config=config, source_hashes={name: S["sha"](source / name) for name in helper_names | metadata_names},
                   execution_environment=env, scheduler_url="http://fixture.invalid:9025", runtime_wheels={"fixture-only.whl": ""})
    (source / "fixture-only.whl").write_bytes(b"mock wheel, never installed or executed")
    request["runtime_wheels"]["fixture-only.whl"] = S["sha"](source / "fixture-only.whl")
    request["request_sha256"] = S["identity"](request)
    S["save"](parent / "request.json", request)
    calibration = root / "calibration.json"
    S["save"](calibration, dict(schema="fixture-only-synthetic-custody", job_id="mock-benchmark-no-execution"))
    monkeypatch.setitem(S["stage"].__globals__, "CALIBRATION_RECEIPT_SHA256", S["sha"](calibration))
    monkeypatch.setitem(S["stage"].__globals__, "PARENTS", {request["request_sha256"]: ("flux_raw_to_image.py", raw_sha())})

    def fake_stage(target, *, worker, config, request, additional_files):
        target.mkdir()
        shutil.copyfile(worker, target / worker.name)
        for path, name in additional_files:
            shutil.copyfile(path, target / name)
        for name in request["runtime_wheels"]:
            shutil.copyfile(source / name, target / name)
        (target / "request.json").write_text(json.dumps(request, sort_keys=True) + "\n")
        (target / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")

    def no_network(*_args, **_kwargs):
        raise AssertionError("controller attempted network in model-free qualification")

    monkeypatch.setattr(payload, "stage_worker", fake_stage)
    monkeypatch.setattr(socket, "create_connection", no_network)
    return parent, root / "new-run", calibration


def raw_sha():
    return "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522"


def test_metadata_only_source_closure_and_exact_count_intent(tmp_path, monkeypatch):
    parent, run, calibration = fixture(tmp_path / "fixture", monkeypatch)
    result = S["stage"](run, parent, calibration)
    assert result["recipe_tensors"] == 709 and result["checkpoint_bytes_read"] == 0
    assert result["submitted"] is False and not (run / "submission.json").exists()
    request = S["read"](run / "request.json")
    counts = request["workload_geometry"]
    assert counts["qwen_functional_calls"] == counts["qwen_native_calls"] == 1
    assert counts["functional_transitions"] == counts["native_transitions"] == 224
    assert counts["act_receipts"] == 4 and counts["receipt_occurrences"] == 228
    assert counts["context_operand_occurrences"] == 8
    assert counts["functional_decode_calls"] == 2 and counts["native_vae_calls"] == 4
    assert request["config"]["max_wall_seconds"] == 1200
    assert request["resource_exception"] == {**request["resource_exception"], "ram_mb": 24576, "vram_mb": 13312,
        "cpu_threads": 4, "disk_gb": 12, "est_wall_s": 900, "timeout_s": 1500}
    inspected = S["inspect_run"](run)
    assert inspected["request_sha256"] == result["request_sha256"] and not inspected["submitted"]
    assert not inspected["finite_review_present"]
    assert "-B" in S["command"](request)
    assert len(S["QUEUE_NOTE"]) <= 500 and "1 nativeQwen36/LM" in S["QUEUE_NOTE"]
    assert S["read"](run / "source-cohort.json")["usage"].endswith("claimed parent-job success")


def test_changed_parent_payload_and_worker_refuse_before_output(tmp_path, monkeypatch):
    parent, run, calibration = fixture(tmp_path / "fixture", monkeypatch)
    (parent / "payload/vae-config.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata/source payload"):
        S["stage"](run, parent, calibration)
    assert not run.exists()


def test_changed_calibration_receipt_refuses_before_output(tmp_path, monkeypatch):
    parent, run, calibration = fixture(tmp_path / "fixture", monkeypatch)
    calibration.write_text("{}")
    with pytest.raises(ValueError, match="custody measurement"):
        S["stage"](run, parent, calibration)
    assert not run.exists()


def test_submit_is_exact_reviewed_one_lease_and_no_retry(tmp_path, monkeypatch):
    import mdb.job_support as support

    parent, run, calibration = fixture(tmp_path / "fixture", monkeypatch)
    S["stage"](run, parent, calibration)
    request = S["read"](run / "request.json")
    calls = []
    monkeypatch.setattr(support, "guarded_mrun_launch", lambda *args, **kwargs: calls.append((args, kwargs)) or "mock-job-no-execution")
    review = dict(request_sha256="wrong", blocking_issues=[], review_status="ready-for-finite-run")
    S["save"](run / "finite-run-review-v1.json", review)
    with pytest.raises(ValueError, match="Exact finite"):
        S["submit"](run)
    assert calls == []
    review["request_sha256"] = request["request_sha256"]
    (run / "finite-run-review-v1.json").write_text(json.dumps(review))
    result = S["submit"](run)
    assert result == dict(job_id="mock-job-no-execution") and len(calls) == 1
    args, kwargs = calls[0]
    assert kwargs["retry_on_kill"] is False and kwargs["preflight"] is True and kwargs["gpu"] is True
    assert kwargs["needs"] == {"cuda": True, "host": "beast"} and kwargs["timeout_s"] == 1500
    assert kwargs["ram_mb"] == 24576 and kwargs["vram_mb"] == 13312
    assert kwargs["request"]["scheduling_contract"]["model_loads"] == 1
    with pytest.raises(ValueError, match="already exists"):
        S["submit"](run)
    assert len(calls) == 1 and args[0] == S["command"](request)


def test_inspect_refuses_changed_wheel_and_has_no_side_effect(tmp_path, monkeypatch):
    parent, run, calibration = fixture(tmp_path / "fixture", monkeypatch)
    S["stage"](run, parent, calibration)
    before = {p.relative_to(run).as_posix(): S["sha"](p) for p in run.rglob("*") if p.is_file()}
    S["inspect_run"](run)
    assert before == {p.relative_to(run).as_posix(): S["sha"](p) for p in run.rglob("*") if p.is_file()}
    (run / "payload/fixture-only.whl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Payload source/wheel"):
        S["inspect_run"](run)


def test_actual_inspect_and_help_cli_are_offline(tmp_path, monkeypatch):
    parent, run, calibration = fixture(tmp_path / "fixture", monkeypatch)
    staged = S["stage"](run, parent, calibration)
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), str(run), "--inspect"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["request_sha256"] == staged["request_sha256"]
    help_result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--help"], cwd=tmp_path,
                                capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and "--inspect" in help_result.stdout and "--submit" in help_result.stdout


def test_recipe_metadata_has_no_checkpoint_dependency_and_wrong_census_refuses(tmp_path):
    paths = {}
    for family, count in (("qwen", 398), ("flux", 169), ("vae", 142)):
        paths[family] = tmp_path / (family + ".json")
        S["save"](paths[family], manifest(family, count))
    result = S["recipe_geometry"](paths)
    assert sum(row["tensors"] for row in result.values()) == 709
    assert all(row["checkpoint_bytes_read"] == 0 for row in result.values())
    (paths["vae"]).write_text(json.dumps(manifest("vae", 141)))
    with pytest.raises(ValueError, match="census"):
        S["recipe_geometry"](paths)

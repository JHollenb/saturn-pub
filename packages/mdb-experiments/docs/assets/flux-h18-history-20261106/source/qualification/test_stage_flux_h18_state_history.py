"""Model-free source staging/CLI checks; payload packaging/admission mocked."""
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
PATH = HERE / "qualification/stage_flux_h18_state_history.py"
S = runpy.run_path(str(PATH))


def manifest(family, count):
    buffer = io.BytesIO()
    np.save(buffer, np.array([0], dtype="<u2"), allow_pickle=False)
    value = buffer.getvalue()
    return dict(schema="mdb-parameter-artifacts-v1", artifacts=[dict(path=f"parameters/{family}-{i}.npy",
        sha256=hashlib.sha256(value).hexdigest(), bytes=len(value), npy_prefix_b64=base64.b64encode(value[:-2]).decode(),
        source=dict(file=family + ".safetensors", file_sha256="f" * 64, tensor="weight." + str(i),
                    dtype="BF16", shape=[1], data_offsets=[2 * i, 2 * i + 2], payload_sha256=hashlib.sha256(value[-2:]).hexdigest()))
        for i in range(count)])


def fixture(root, monkeypatch):
    import mdb.execution_environment as environment
    import mdb.payload as payload

    root.mkdir()
    parent = root / "parent"
    source = parent / "payload"
    source.mkdir(parents=True)
    e18 = S["base"]()
    utilities = e18["base"]()
    helpers = {**utilities["HELPERS"], "flux_raw_to_image.py": "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522"}
    python_names = set(helpers) | {e18["WORKER"], S["BASE"], e18["BASE"]}
    for name in python_names:
        shutil.copyfile(HERE / "qualification" / name, source / name)
    metadata = (set(utilities["PATHS"].values()) - set(helpers)) | {"model_index.json", "source-metadata-receipt-v1.json"}
    for name in metadata:
        if "parameter-artifacts" in name:
            family = name.split("-")[0]
            value = manifest(family, {"qwen": 398, "flux": 169, "vae": 142}[family])
        else:
            value = dict(block_out_channels=[128, 256, 512, 512]) if name == "vae-config.json" else {}
        (source / name).write_text(json.dumps(value) + "\n")
    raw = runpy.run_path(str(HERE / "qualification/flux_raw_to_image.py"))
    config = dict(schema="mdb-flux-raw-to-image-plan-v1", seed=277, prompt=raw["PROMPT"], batch_size=1,
        height=512, width=512, steps=4, device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True,
        max_gpu_components=1, component_hydrations=3, cuda_placements=6, offloads=6,
        qwen_manifest_entries=398, flux_manifest_entries=169, vae_manifest_entries=142,
        native_sources={name: "fixture-metadata-only" for name in raw["NATIVE_SOURCE_MODULES"]},
        qwen_rope_buffer_policy=dict(placement_dtype="float32", init_device="cpu"),
        attention_dispatch=dict(backend="sdpa", gqa_policy="repeat_kv", source="fixture-only"),
        cpu_threads=4, qwen_parameter_root="/never-open/qwen", flux_parameter_root="/never-open/flux",
        vae_parameter_root="/never-open/vae", output_scratch="/never-write/research")
    engine = runpy.run_path(str(HERE / "qualification" / e18["WORKER"]))
    config = engine["make_config"](config)
    for key, name in utilities["PATHS"].items():
        config[key], config[key + "_sha256"] = name, S["sha"](source / name)
    config["helper_source_sha256"] = helpers
    env = {**environment.intent(), "tmpdir": "/synthetic/tmp", "cuda_cache_path": "/synthetic/cuda",
           "torchinductor_cache_dir": "/synthetic/inductor"}
    for name in ("MDB_WORKER_UV_CACHE_DIR", "MDB_WORKER_TORCH_HOME", "MDB_WORKER_MODEL_ROOTS", "MDB_WORKER_PYTHON", "MDB_WORKER_UV_LINK_MODE"):
        monkeypatch.setenv(name, "")
    (source / "fixture-only.whl").write_bytes(b"not a wheel; synthetic packaging only")
    parent_request = dict(schema="mdb-flux-e18-conditioning-request-v1", config=config,
        source_hashes={name: S["sha"](source / name) for name in python_names | metadata}, launcher_sha256=S["BASE_SHA256"],
        runtime_wheels={"fixture-only.whl": S["sha"](source / "fixture-only.whl")},
        execution_environment=env, scheduler_url="http://fixture.invalid:9025")
    parent_request["request_sha256"] = S["identity"](parent_request)
    S["save"](parent / "request.json", parent_request)
    custody = dict(sha256="a" * 64, size_bytes=17)
    job = dict(job_id="fixture-no-execution", state="succeeded", config=parent_request,
        payload_custody={key: custody for key in ("declared", "sealed", "executed")},
        result=dict(returncode=0, status="ok", elapsed_s=545.01, peak_rss_mb=20462.1, peak_vram_mb=8374.,
            executed_payload=custody, diagnostics=dict(command=e18["command"](parent_request))))
    resource = root / "resource-job.json"
    S["save"](resource, job)
    monkeypatch.setitem(S["stage"].__globals__, "PARENT_REQUEST", parent_request["request_sha256"])
    monkeypatch.setitem(S["stage"].__globals__, "RESOURCE_JOB_ID", job["job_id"])
    monkeypatch.setitem(S["stage"].__globals__, "RESOURCE_JOB_SHA256", S["sha"](resource))

    def fake_stage(target, *, worker, config, request, additional_files):
        target.mkdir()
        shutil.copyfile(worker, target / worker.name)
        for path, name in additional_files:
            shutil.copyfile(path, target / name)
        for name in request["runtime_wheels"]:
            shutil.copyfile(source / name, target / name)
        (target / "request.json").write_text(json.dumps(request, sort_keys=True) + "\n")
        (target / "config.json").write_text(json.dumps(config, sort_keys=True) + "\n")

    def no_network(*args, **kwargs):
        raise AssertionError("network forbidden in model-free qualification")

    monkeypatch.setattr(payload, "stage_worker", fake_stage)
    monkeypatch.setattr(socket, "create_connection", no_network)
    return parent, root / "new-run", resource


def test_metadata_closure_recipes_exact_counts_and_resources(tmp_path, monkeypatch):
    parent, run, resource = fixture(tmp_path / "fixture", monkeypatch)
    result = S["stage"](run, parent, resource)
    assert result["submitted"] is False and result["recipe_tensors"] == 709 and result["checkpoint_bytes_read"] == 0
    request = S["read"](run / "request.json")
    counts = request["workload_geometry"]
    assert counts["qwen_functional_blocks"] == 36 and counts["qwen_native_blocks"] == 45
    assert counts["qwen_native_calls"] == 1 and counts["qwen_native_suffix_blocks"] == 9
    assert counts["functional_transitions"] == counts["native_transitions"] == 224
    assert counts["act_receipts"] == 4 and counts["receipt_occurrences"] == 228
    assert counts["context_operand_occurrences"] == 8 and counts["genuine_qwen_acts"] == 0
    assert counts["functional_decode_calls"] == 2 and counts["native_vae_calls"] == 4
    assert request["source_hashes"]["qwen_h18_state_history.py"] == "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036"
    assert request["source_hashes"]["flux_e18_conditioning_act.py"] == "45c512de2dff3252c38a182cf4291b392cf35a159c2bd83b35900c28005cb343"
    assert request["config"]["max_wall_seconds"] == 1200 and request["config"]["forecast_transaction_bytes"] == 1 << 30
    assert request["config"]["transport_limits"] == dict(max_expanded_bytes=8 << 30, max_member_bytes=134221824)
    assert request["resource_exception"]["measurement"]["elapsed_s"] == 545.01
    assert {key: request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")} == dict(
        ram_mb=24576, vram_mb=13312, cpu_threads=4, disk_gb=12, est_wall_s=900, timeout_s=1500)
    assert "component_or_transition_or_Act_calls" not in request["scheduling_contract"]
    assert len(S["QUEUE_NOTE"]) <= 500 and "45 native(1full36+LM,9suffix)" in S["QUEUE_NOTE"]
    assert S["read"](run / "source-cohort.json")["usage"].endswith("no historical activations or learned weight payloads")
    assert S["read"](run / "preflight.v1.json")["genuine_Qwen_Acts"] == 0


def test_changed_metadata_and_wheels_refuse_before_output(tmp_path, monkeypatch):
    parent, run, resource = fixture(tmp_path / "fixture", monkeypatch)
    (parent / "payload/vae-config.json").write_text("{}")
    with pytest.raises(ValueError, match="metadata/source payload"):
        S["stage"](run, parent, resource)
    assert not run.exists()
    shutil.copyfile(HERE / "qualification" / S["BASE"], parent / "payload" / S["BASE"])
    # A distinct fixture tests wheel custody, without rewriting the first failure.
    parent2, run2, resource2 = fixture(tmp_path / "fixture2", monkeypatch)
    (parent2 / "payload/fixture-only.whl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="qualified wheel"):
        S["stage"](run2, parent2, resource2)
    assert not run2.exists()


def test_resource_receipt_identity_and_complete_payload_joins(tmp_path, monkeypatch):
    parent, run, resource = fixture(tmp_path / "fixture", monkeypatch)
    request = S["read"](parent / "request.json")
    original = S["read"](resource)
    resource.write_text("{}")
    with pytest.raises(ValueError, match="measured source job"):
        S["stage"](run, parent, resource)
    assert not run.exists()
    original["payload_custody"]["sealed"] = dict(sha256="b" * 64, size_bytes=17)
    resource.write_text(json.dumps(original))
    monkeypatch.setitem(S["stage"].__globals__, "RESOURCE_JOB_SHA256", S["sha"](resource))
    with pytest.raises(ValueError, match="payload/command custody"):
        S["resource_binding"](resource, request)


def test_submit_exact_finite_review_one_lease_and_no_retry(tmp_path, monkeypatch):
    import mdb.job_support as support

    parent, run, resource = fixture(tmp_path / "fixture", monkeypatch)
    S["stage"](run, parent, resource)
    request = S["read"](run / "request.json")
    calls = []
    monkeypatch.setattr(support, "guarded_mrun_launch", lambda *args, **kwargs: calls.append((args, kwargs)) or "mock-no-execution")
    review = dict(request_sha256="wrong", blocking_issues=[], review_status="ready-for-finite-run")
    S["save"](run / "finite-run-review-v1.json", review)
    with pytest.raises(ValueError, match="Exact finite"):
        S["submit"](run)
    assert calls == []
    review["request_sha256"] = request["request_sha256"]
    (run / "finite-run-review-v1.json").write_text(json.dumps(review))
    assert S["submit"](run) == dict(job_id="mock-no-execution")
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == S["command"](request)
    assert kwargs["retry_on_kill"] is False and kwargs["preflight"] is True and kwargs["gpu"] is True
    assert kwargs["needs"] == dict(cuda=True, host="beast") and kwargs["timeout_s"] == 1500
    assert kwargs["experiment"] == "mdb-flux-h18-state-history"
    assert kwargs["request"]["scheduling_contract"]["max_mrun_submissions"] == 1
    with pytest.raises(ValueError, match="already exists"):
        S["submit"](run)
    assert len(calls) == 1


def test_offline_inspect_source_closure_and_cli(tmp_path, monkeypatch):
    parent, run, resource = fixture(tmp_path / "fixture", monkeypatch)
    staged = S["stage"](run, parent, resource)
    before = {p.relative_to(run).as_posix(): S["sha"](p) for p in run.rglob("*") if p.is_file()}
    assert S["inspect_run"](run)["request_sha256"] == staged["request_sha256"]
    assert before == {p.relative_to(run).as_posix(): S["sha"](p) for p in run.rglob("*") if p.is_file()}
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH), str(run), "--inspect"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["request_sha256"] == staged["request_sha256"]
    help_result = subprocess.run([sys.executable, "-I", "-B", str(PATH), "--help"], cwd=tmp_path,
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0 and "--resource-job" in help_result.stdout
    (run / "payload/qwen_h18_state_history.py").write_text("tampered")
    with pytest.raises(ValueError, match="Payload source/wheel"):
        S["inspect_run"](run)


def test_recipe_metadata_census_uses_no_checkpoint_files(tmp_path):
    paths = {}
    for family, count in (("qwen", 398), ("flux", 169), ("vae", 142)):
        paths[family] = tmp_path / (family + ".json")
        S["save"](paths[family], manifest(family, count))
    geometry = S["base"]()["recipe_geometry"](paths)
    assert sum(row["tensors"] for row in geometry.values()) == 709
    assert all(row["checkpoint_bytes_read"] == 0 for row in geometry.values())
    paths["vae"].write_text(json.dumps(manifest("vae", 141)))
    with pytest.raises(ValueError, match="census"):
        S["base"]()["recipe_geometry"](paths)

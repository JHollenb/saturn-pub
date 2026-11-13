"""New finite-run geometry and cold-parameter custody, without scheduler calls."""
import base64
import hashlib
import io
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "qualification"
L = runpy.run_path(str(HERE / "submit_flux_initial_noise.py"))
C = runpy.run_path(str(HERE / "collect_flux_initial_noise.py"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def recipe(index):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"descr":"<u2", "fortran_order":False, "shape":(1,)})
    prefix = stream.getvalue()
    payload = index.to_bytes(2, "little")
    return {"path":f"arrays/{index:064x}.npy", "bytes":len(prefix)+2,
        "sha256":hashlib.sha256(prefix+payload).hexdigest(),
        "npy_prefix_b64":base64.b64encode(prefix).decode(),
        "source":{"file":"transformer/weights.safetensors", "file_sha256":"a"*64,
            "tensor":f"bank.{index}", "dtype":"BF16", "shape":[1], "data_offsets":[2*index,2*index+2],
            "payload_sha256":hashlib.sha256(payload).hexdigest()}}


def fixture(root):
    helper = b"# synthetic source-bound standalone retriever\n"
    request = {"request_sha256":"1"*64, "source_hashes":{"worker.py":"2"*64},
        "config":{"expected_parameter_artifacts_module_sha256":hashlib.sha256(helper).hexdigest()}}
    save(root/"source.json", {"job_id":"job-fixture", "request_sha256":request["request_sha256"],
        "source_hashes":request["source_hashes"]})
    save(root/"summary.json", {"job_id":"job-fixture", "request_sha256":request["request_sha256"]})
    save(root/"plan.json", request["config"])
    save(root/"transaction/full/seal.json", {"history":"full"})
    save(root/"transaction/matched/seal.json", {"history":"matched"})
    save(root/"parameter-artifacts.json", {"schema":"mdb-parameter-artifacts-v1",
        "artifacts":[recipe(index) for index in range(169)]})
    (root/"retrieve_weights.py").write_bytes(helper)
    return request


def producer_manifest(root, request):
    actual = C["inventory"](root)
    files = {key:value for key,value in actual["files"].items() if key!="remote-manifest.json"}
    save(root/"remote-manifest.json", {"schema":"mdb-step-boundary-remote-manifest-v1",
        "job_id":"job-fixture", "request_sha256":request["request_sha256"], "files":files})
    return C["inventory"](root)


def config():
    return {"full_transaction_max_bytes":3<<30, "matched_transaction_max_bytes":1536<<20,
        "transport_limits":{"max_expanded_bytes":10<<30, "max_member_bytes":3<<30}}


def test_six_pass_prediction_geometry_omits_all_parameter_exports():
    plan = L["storage_plan"](config())
    assert plan["full_output_registers"] == 104 and plan["matched_output_registers"] == 52
    assert plan["full_prediction_bytes_lower_bound"] == 2*plan["matched_prediction_bytes_lower_bound"]
    assert plan["parameter_archives_persisted"] == 0 and plan["parameter_recipe_count"] == 169
    assert plan["checkpoint_parameter_bytes_reused"] == 7_751_089_152


def test_actual_command_uses_sealed_worker_scratch_on_hdd(monkeypatch):
    monkeypatch.delenv("MDB_WORKER_UV_CACHE_DIR", raising=False)
    environment = L["execution_intent"]()
    command = L["sealed_worker_command"]({"execution_environment":environment})
    assert "TMPDIR=/mnt/hdd8tb/mdb/tmp" in command
    assert "CUDA_CACHE_PATH=/mnt/hdd8tb/mdb/cuda-cache" in command
    assert "TORCHINDUCTOR_CACHE_DIR=/mnt/hdd8tb/mdb/torchinductor-cache" in command
    assert command[-1] == "flux_initial_noise_campaign.py" and "--offline" in command
    assert not any("/Users/" in value for value in command)
    environment["tmpdir"] = "/tmp"
    with pytest.raises(ValueError, match="admitted HDD scratch root"):
        L["sealed_worker_command"]({"execution_environment":environment})


@pytest.mark.parametrize("key,bound", [("full_transaction_max_bytes",1<<30),
    ("matched_transaction_max_bytes",512<<20), ("max_member_bytes",2<<30), ("max_expanded_bytes",3<<30)])
def test_refuse_unsafe_complete_prediction_bounds(key, bound):
    value = config()
    (value["transport_limits"] if key.startswith("max_") else value)[key] = bound
    with pytest.raises(ValueError, match="custody bounds"):
        L["storage_plan"](value)


def test_collection_accepts_both_seals_and_169_cold_exact_recipes(tmp_path):
    request = fixture(tmp_path)
    actual = producer_manifest(tmp_path, request)
    assert C["verify_manifest"](tmp_path, actual, "job-fixture", request)
    assert C["RESULT"] == "flux-initial-noise-result"
    assert not any(name.startswith("arrays/") for name in actual["files"])


@pytest.mark.parametrize("history", ["full", "matched"])
def test_collection_requires_each_independent_prediction_seal(tmp_path, history):
    request = fixture(tmp_path)
    (tmp_path/f"transaction/{history}/seal.json").unlink()
    with pytest.raises(ValueError, match="metadata absent"):
        C["verify_manifest"](tmp_path, producer_manifest(tmp_path, request), "job-fixture", request)


def test_collection_refuses_accidental_per_run_weight_export(tmp_path):
    request = fixture(tmp_path)
    path = tmp_path/recipe(0)["path"]
    path.parent.mkdir()
    path.write_bytes(b"duplicate weight file")
    with pytest.raises(ValueError, match="omitted per-run parameter archives"):
        C["verify_manifest"](tmp_path, producer_manifest(tmp_path, request), "job-fixture", request)


def test_collection_refuses_unbound_standalone_retriever(tmp_path):
    request = fixture(tmp_path)
    (tmp_path/"retrieve_weights.py").write_text("changed after source seal")
    with pytest.raises(ValueError, match="exact recipes"):
        C["verify_manifest"](tmp_path, producer_manifest(tmp_path, request), "job-fixture", request)

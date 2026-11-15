"""Actual tensor serializer fixtures for exact signed-vector bridge mechanics."""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parents[1] / "qualification"
B = runpy.run_path(str(HERE / "audit_qwen_attention_bridge.py"))
H = B["helpers"]()
W = runpy.run_path(str(HERE / "qwen_block0_attention_microscope.py"))


def rewrite(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def fixture(path, *, negative=False, future=False):
    """Saved synthetic fields only; no model, attention or checkpoint access."""
    path.mkdir()
    record, save = W["tensor_record"], W["save"]
    cases = {case: dict(native={}) for case in ("short", "long")}
    triangle = dict(arms={arm: {} for arm in B["ARMS"]}, fixed_bank_case="long", historical_bridge_valid_prefix_rows=2)
    for port in B["PORTS"]:
        shape = (1, 2, 4, 3) if port == "head" else (1, 4, 3)
        a = torch.zeros(shape, dtype=torch.bfloat16)
        b = a.clone()
        b.reshape(-1)[:3] = torch.tensor([1., 2 ** -30, -(2 ** -30)], dtype=torch.bfloat16)
        long = b.clone()
        if negative and port == "head":
            long.reshape(-1)[1:3] *= -1
        c = b.clone()
        if future and port == "head":
            c[:, 0, -1, 0] = .5
        for arm, value in zip(B["ARMS"], (a, b, c), strict=True):
            triangle["arms"][arm][port] = record(path, "triangle/" + arm + "/" + port, value)
        cases["short"]["native"][port] = record(path, "native/short/" + port, a)
        cases["long"]["native"][port] = record(path, "native/long/" + port, long)
    plan = dict(schema="mdb-qwen-block0-attention-plan-v1", profile="tiny-cpu", fixed_bank_case="long", shared_prefix_rows=2)
    request = dict(config=plan, source_hashes={"qwen_block0_attention_microscope.py": H["WORKER_SHA256"], **H["HELPER_PINS"]})
    request["request_sha256"] = H["identity"](request)
    save(path / "request.json", request)
    save(path / "plan.json", {**plan, "request_sha256": request["request_sha256"]})
    save(path / "cases.json", cases)
    save(path / "triangle.json", triangle)
    external = {p.relative_to(path).as_posix(): dict(bytes=p.stat().st_size, sha256=H["sha"](p)) for p in path.rglob("*.npy") if "triangle" in p.parts}
    save(path / "functional-seal.json", dict(native_block_calls=0, triangle=triangle["arms"], external_files=external))
    seal_sha = H["sha"](path / "functional-seal.json")
    source = dict(job_id="job-synthetic-microscope", request_sha256=request["request_sha256"], worker_sha256=H["WORKER_SHA256"],
        helper_sources=H["HELPER_PINS"], functional_seal_sha256=seal_sha)
    save(path / "source.json", source)
    save(path / "summary.json", dict(job_id=source["job_id"], request_sha256=source["request_sha256"], actual=dict(native_block_calls=2, sdpa_calls=7)))
    timeline = [dict(kind="functional_sources_and_triangle_complete", seconds=1.),
        dict(kind="functional_fields_sealed", seconds=2., seal_sha256=seal_sha, sdpa_calls=5, native_block_calls=0),
        dict(kind="native_block_entry", seconds=3., case="short", functional_seal_sha256=seal_sha),
        dict(kind="native_block_completed", seconds=4., case="short"),
        dict(kind="native_block_entry", seconds=5., case="long", functional_seal_sha256=seal_sha),
        dict(kind="native_block_completed", seconds=6., case="long")]
    save(path / "timeline.json", timeline)
    (path / "timeline.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in timeline))
    finalizer = runpy.run_path(str(HERE / "flux_image_tail_microscope.py"))
    wrapper = finalizer["finalize"](path, job_id=source["job_id"], request_sha256=source["request_sha256"],
        max_expanded_bytes=2 << 20, max_member_bytes=1 << 20)
    return path, wrapper["files_manifest_sha256"], request["request_sha256"]


def test_exact_bridge_and_head_coordinate_to_token_accounts(tmp_path):
    root, manifest, request = fixture(tmp_path / "raw")
    result = B["audit"](root, tmp_path / "out", expected_manifest=manifest, expected_request=request)
    for report in result["ports"].values():
        assert report["signed_vector_bridge_shared_prefix"]["bitwise_exact"]
        assert report["signed_vector_bridge_shared_prefix"]["changed_values"] == 0
        assert report["A_vs_native_short_shared_prefix"]["bitwise_exact"]
        assert report["B_vs_native_long_shared_prefix"]["bitwise_exact"]
        assert report["C_vs_native_long_shared_prefix"]["bitwise_exact"]
        assert report["C_minus_B_complete"]["bitwise_exact"]
        assert report["first_vector_sha256"] == report["historical_vector_sha256"]
    coordinates = H["read"](tmp_path / "out/head-coordinates.json")
    assert coordinates["total_coordinates"] == 3 and coordinates["complete"]
    assert coordinates["coordinates"][0]["head"] == coordinates["coordinates"][0]["token"] == 0
    assert coordinates["coordinates"][1]["B_minus_A"] == 2 ** -30
    tokens = H["read"](tmp_path / "out/token-changes.json")
    assert tokens["head"]["B-minus-A"]["changed_tokens"] == 1
    assert tokens["mlp"]["B-minus-A"]["total_changed_values"] == 3
    assert "not an independently hooked" in result["port_provenance_limit"]
    assert not list((tmp_path / "out").rglob("*.npy"))


def test_equal_rounded_energy_cosine_can_still_have_signed_vector_miss(tmp_path):
    root, manifest, request = fixture(tmp_path / "raw", negative=True)
    result = B["audit"](root, tmp_path / "out", expected_manifest=manifest, expected_request=request)
    head = result["ports"]["head"]
    # The old aggregate alignment can round to1 while the complete signed vectors differ.
    assert head["signed_vector_bridge_shared_prefix"]["cosine"] == 1.
    assert head["B_minus_A_shared_prefix"]["rms"] == head["native_long_minus_short_shared_prefix"]["rms"]
    assert head["signed_vector_bridge_shared_prefix"]["changed_values"] == 2
    assert head["signed_vector_bridge_shared_prefix"]["first_difference"]["coordinate"] == [0, 0, 0, 1]
    assert head["first_vector_sha256"] != head["historical_vector_sha256"]
    assert result["mechanics_status"] == "authenticated-selected-saved-vectors"


def test_complete_C_minus_B_covers_beyond_prefix(tmp_path):
    root, manifest, request = fixture(tmp_path / "raw", future=True)
    result = B["audit"](root, tmp_path / "out", expected_manifest=manifest, expected_request=request)
    head = result["ports"]["head"]
    assert head["signed_vector_bridge_shared_prefix"]["changed_values"] == 0
    assert head["C_minus_B_complete"]["changed_values"] == 1
    assert head["C_minus_B_complete"]["first_difference"]["coordinate"] == [0, 0, 3, 0]


def test_signed_zero_bit_miss_is_separate_from_numeric_difference():
    report = B["comparison"](np.array([-0.], np.float64), np.array([0.], np.float64), H)
    assert report["exact_value_equality"] and report["changed_values"] == 0
    assert not report["bitwise_exact"] and report["first_difference"] is None
    assert report["first_bit_difference"]["flat_index"] == 0


def test_descriptive_coordinate_cap_keeps_full_total():
    a = np.zeros((1, 2, 30, 3), np.float32)
    b = np.ones_like(a)
    report = B["head_coordinates"](a, b, b, a, b)
    assert report["total_coordinates"] == 180 and not report["complete"]
    assert len(report["coordinates"]) == B["COORDINATE_CAP"] == 128


@pytest.mark.parametrize("mutation", ["selected", "metadata", "wrapper"])
def test_hostile_selected_custody_refuses(tmp_path, mutation):
    root, manifest, request = fixture(tmp_path / "raw")
    if mutation == "selected":
        path = root / "native/long/H1.npy"
        path.write_bytes(path.read_bytes() + b"hostile")
    elif mutation == "metadata":
        rewrite(root / "cases.json", {})
    else:
        manifest = "0" * 64
    with pytest.raises(ValueError):
        B["audit"](root, tmp_path / "out", expected_manifest=manifest, expected_request=request)


def test_bounded_worker_marker_without_model_framework_imports(tmp_path):
    root, manifest, request = fixture(tmp_path / "raw")
    config = dict(native_root=str(root), native_manifest_sha256=manifest, native_request_sha256=request, output_scratch=str(tmp_path / "output"))
    envelope = dict(config=config, source_hashes={str(HERE / "audit_qwen_attention_bridge.py"): H["sha"](HERE / "audit_qwen_attention_bridge.py"),
        str(HERE / B["HELPER"]): B["HELPER_SHA256"]})
    envelope["request_sha256"] = H["identity"](envelope)
    W["save"](tmp_path / "config.json", config)
    W["save"](tmp_path / "request.json", envelope)
    script = """import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','tokenizers'}:
        raise AssertionError('saved bridge imported neural framework: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
runpy.run_path(sys.argv[1],run_name='__main__')
"""
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(HERE / "audit_qwen_attention_bridge.py")], text=True, capture_output=True, check=False,
        env={**os.environ, "SATURN_RESIDENT_WORKER": "1", "SATURN_RUN_CONFIG": str(tmp_path / "config.json"), "SATURN_DEBUG_REQUEST": str(tmp_path / "request.json"),
             "CUDA_VISIBLE_DEVICES": "", "MRUN_JOB_ID": "job-synthetic-bridge"})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = json.loads(next(line.removeprefix("MDB_DIRECTORY_READY ") for line in completed.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(marker["output_root"])
    assert marker["request_sha256"] == envelope["request_sha256"] and marker["job_id"] == "job-synthetic-bridge"
    assert marker["files_manifest_sha256"] == H["sha"](output / "files.json")
    assert H["read"](output / "summary.json")["model_calls"] == 0
    assert len(H["read"](output / "bridge.json")["ports"]) == 6

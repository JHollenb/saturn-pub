"""Installed-wheel saved-only remaining-pass audit mechanics, not model accuracy."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/audit_flux_functional_remaining.py"
A = runpy.run_path(str(PATH))
T = runpy.run_path(str(HERE / "tests/test_flux_functional_remaining.py"))
W, M = T["W"], T["M"]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    config = T["fixture"](tmp_path_factory.mktemp("remaining-audit-source"))
    config["request_sha256"] = "1" * 64
    old = os.environ.get("MRUN_JOB_ID")
    os.environ["MRUN_JOB_ID"] = "job-tiny-remaining"
    try:
        summary = W["run"](config)
    finally:
        if old is None:
            os.environ.pop("MRUN_JOB_ID", None)
        else:
            os.environ["MRUN_JOB_ID"] = old
    ready = M["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return Path(config["output"]), summary, ready


def copied(saved, tmp_path):
    data, summary, ready = saved
    target = tmp_path / "raw"
    shutil.copytree(data, target)
    return target, summary, ready


def repin(root, name):
    manifest = A["read"](root / "files.json")
    manifest["files"][name] = dict(sha256=A["sha"](root / name), bytes=(root / name).stat().st_size)
    (root / "files.json").write_text(json.dumps(manifest, sort_keys=True))
    return A["sha"](root / "files.json")


def test_full_remaining_audit_streams_both_checkpoints_and_verifies_real_consumers(saved, tmp_path):
    data, summary, ready = saved
    original = torch.nn.Module._call_impl
    def forbidden(*args, **kwargs):
        raise AssertionError("saved auditor called a native module")
    torch.nn.Module._call_impl = forbidden
    try:
        report, inventory = A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    finally:
        torch.nn.Module._call_impl = original
    assert report["forecast_horizon_per_case"] == 14
    assert len(report["signed_finite_responses"]) == 14
    assert report["first_covered_bitwise_mismatch"] == dict.fromkeys(A["CASES"])
    assert report["counts"]["forecast"]["module_entries"] == 0
    assert report["counts"]["native"]["module_entries"] == 800
    assert report["totals"]["vae_calls"] == 8
    assert report["parameter_restoration"]["transformer"]["tensors"] > 20
    assert report["parameter_restoration"]["decoder"]["tensors"] > 20
    assert not (tmp_path / "audit/parameters").exists()
    assert report["source_model_identity"] != report["verification_adapter_identity"]
    for case, rows in report["boundaries"].items():
        assert len(rows) == 14
        assert all(field["bitwise_exact"] for row in rows for field in row["fields"].values())
        consumer = report["consumers"][case]
        assert consumer["predicted"]["record"]["input_cut"] != consumer["native_reference"]["record"]["input_cut"]
        assert all(v["bitwise_exact"] for v in consumer["comparison"].values())
        assert not consumer["predicted"]["record"]["independent_decoder_prediction"]
    catalog = A["read"](tmp_path / "audit/catalog/provenance.json")
    assert catalog["records"] == 4 and catalog["observed_rgb_accounts"] == 3
    assert all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in catalog["queries"])
    assert len(inventory["files"]) > 100


@pytest.mark.parametrize("fault", ["counter", "reference-before-seal", "consumer-sidecar", "decoder-operand"])
def test_coherently_inventoried_wrong_mechanics_refused(saved, tmp_path, fault):
    data, summary, _ = copied(saved, tmp_path)
    if fault == "counter":
        name = "summary.json"
        row = A["read"](data / name)
        row["counts"]["forecast"]["module_entries"] = 1
        (data / name).write_text(json.dumps(row))
    elif fault == "reference-before-seal":
        name = "timeline.jsonl"
        events = [json.loads(line) for line in (data / name).read_text().splitlines()]
        entry = next(row for row in events if row["kind"] == "reference_cut_read")
        events.remove(entry)
        events.insert(0, entry)
        (data / name).write_text("".join(json.dumps(row) + "\n" for row in events))
    elif fault == "consumer-sidecar":
        name = "native-reference-decoder/HH/consumer.json"
        record = A["read"](data / name)
        record["input_cut"] = "0" * 64
        (data / name).write_text(json.dumps(record))
    else:
        name = "native-reference-decoder/HH/actual-decoder-input-bf16.npy"
        value = np.load(data / name, allow_pickle=False)
        value.flat[0] ^= np.uint16(1)
        with (data / name).open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
    with pytest.raises(ValueError):
        A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_actual_precision_miss_remains_observation(saved, tmp_path):
    data, summary, _ = copied(saved, tmp_path)
    name = "native-ports/HH/00.json"
    value = A["read"](data / name)
    value["ports"][0]["fields"][next(iter(value["ports"][0]["fields"]))]["descriptor"]["sha256"] = "0" * 64
    (data / name).write_text(json.dumps(value))
    result, _ = A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])
    assert result["first_covered_bitwise_mismatch"]["HH"]["number"] == 0
    assert result["totals"]["vae_calls"] == 8
    assert result["mechanics_status"] == "authenticated-full-state-saved-evidence-audit"


def test_schedule_and_lossless_signed_zero_decoder_archive(tmp_path):
    before, after = A["schedule"](2, 2)
    assert len(before) == 14
    assert before[7] == dict(step=3, phase="project", cursor=0)
    assert after[-1] == dict(step=4, phase="project", cursor=0)
    path = tmp_path / "bits.npy"
    bits = np.array([0, 32768], dtype=np.uint16)
    with path.open("wb") as stream:
        np.save(stream, bits, allow_pickle=False)
    import hashlib
    record = dict(path=path.name, sha256=A["sha"](path), bytes=path.stat().st_size, shape=[2], dtype="uint16", encoding="bf16-bits-v1",
                  native_descriptor=dict(dtype="torch.bfloat16", shape=[2], sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
    result = A["archived_array"](tmp_path, record, bf16=True)
    assert result.view(np.uint32).tolist() == [0, 2147483648]
    record["native_descriptor"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="operand bits"):
        A["archived_array"](tmp_path, record, bf16=True)


def test_actual_compact_worker_cli_copies_only_small_metadata_and_original_images(saved, tmp_path):
    data, summary, ready = saved
    config = dict(native_root=str(data), native_manifest_sha256=ready["files_manifest_sha256"], native_request_sha256=summary["request_sha256"], output_scratch=str(tmp_path / "compact"))
    request = dict(config=config, request_sha256="2" * 64, source_hashes={p.name: A["sha"](p) for p in [PATH, *[HERE / "qualification" / name for name in A["HELPERS"]]]})
    request_path, config_path = tmp_path / "request.json", tmp_path / "config.json"
    request_path.write_text(json.dumps(request))
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SATURN_RESIDENT_WORKER": "1", "SATURN_DEBUG_REQUEST": str(request_path), "SATURN_RUN_CONFIG": str(config_path), "MRUN_JOB_ID": "job-tiny-remaining-audit"}
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH)], cwd=HERE / "qualification", env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(marker["output_root"])
    assert marker["files_manifest_sha256"] == A["sha"](output / "files.json")
    assert not (output / "prediction-store").exists() and not (output / "native-store").exists()
    assert len(list(output.rglob("image.png"))) == 8
    assert (output / "retrieve_evidence.py").exists()
    assert (output / Path(A["read"](data / "plan.json")["vae_parameter_manifest"]).name).exists()

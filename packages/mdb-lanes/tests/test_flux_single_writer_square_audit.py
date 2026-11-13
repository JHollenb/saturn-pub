"""Installed tiny evidence audit of sealed local parallel-writer corners."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/audit_flux_single_writer_square.py"
A = runpy.run_path(str(PATH))
F = runpy.run_path(str(HERE / "tests/test_flux_single_writer_square.py"))
W, M = F["W"], F["M"]


def captured(root):
    config = F["fixture"](root)
    config["request_sha256"] = "1" * 64
    with patch.dict(os.environ, {"MRUN_JOB_ID": "job-tiny-writer"}):
        summary = W["run"](config)
    ready = M["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return Path(config["output"]), summary, ready


def repin(data, name):
    value = json.loads((data / "files.json").read_text())
    value["files"][name] = dict(sha256=A["sha"](data / name), bytes=(data / name).stat().st_size)
    (data / "files.json").write_text(json.dumps(value))
    return A["sha"](data / "files.json")


def changed_operand(data, original):
    record = json.loads(json.dumps(original))
    value = np.load(data / record["path"], allow_pickle=False).copy()
    value.reshape(-1)[0] ^= np.uint16(1)
    raw_sha = hashlib.sha256(value.tobytes()).hexdigest()
    path = data / "operands" / (raw_sha + ".npy")
    np.save(path, value, allow_pickle=False)
    record.update(path=path.relative_to(data).as_posix(), sha256=A["sha"](path), bytes=path.stat().st_size, payload_sha256=raw_sha)
    record["descriptor"]["sha256"] = raw_sha
    repin(data, record["path"])
    return record


def test_tiny_actual_sealed_corners_to_twenty_local_equation_records(tmp_path):
    data, summary, ready = captured(tmp_path)
    def no_forward(module, args):
        raise AssertionError("saved audit attempted a native/model forward")
    hook = torch.nn.modules.module.register_module_forward_pre_hook(no_forward)
    try:
        result, _ = A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    finally:
        hook.remove()
    assert result["verification"]["phase"] == "closed"
    assert result["counts"] == W["contract"]()["counts"]
    assert all(row[stream]["bitwise_bf16_exact"] for row in result["local_prediction_errors"].values() for stream in ("image", "text"))
    assert all(row[stream]["bitwise_bf16_exact"] for row in result["source_controls"].values() for stream in ("image", "text"))
    assert all(stage[stream]["bitwise_bf16_exact"] for row in result["own11_trace_controls"].values() for stage in row.values() for stream in ("image", "text"))
    assert result["new_model_calls"] == result["new_reference_forward_calls"] == 0
    catalog = json.loads((tmp_path / "audit/catalog/provenance.json").read_text())
    assert catalog["records"] == 20 and catalog["final_forecast_records"] == 0
    assert all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in catalog["queries"])
    assert result["input_novelty"]["HH.11"]["equal_recipient_own"]
    assert result["input_novelty"]["HH.00"]["equal_sham"]


@pytest.mark.parametrize("kind", ("consumed", "source-HM", "ledger"))
def test_coherent_sidecar_repin_cannot_hide_wrong_operands_source_or_order(tmp_path, kind):
    data, summary, _ = captured(tmp_path)
    name = {"consumed": "cases/HH.01.local.json", "source-HM": "source-calls.json", "ledger": "timeline.jsonl"}[kind]
    path = data / name
    if kind == "ledger":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["counts"]["source_entries"] = 99
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        row = json.loads(path.read_text())
        if kind == "consumed":
            row["operands"]["consumed"] = changed_operand(data, row["operands"]["consumed"])
        else:
            row["HH"]["observed_concat"] = changed_operand(data, row["HH"]["observed_concat"])
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="operands|source concat|census"):
        A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_closed_event_head_and_sealed_files_refuse_corruption(tmp_path):
    data, summary, _ = captured(tmp_path)
    name = "summary.json"
    row = json.loads((data / name).read_text())
    row["transaction"]["event_head"] = "0" * 64
    (data / name).write_text(json.dumps(row))
    with pytest.raises(Exception, match="event head"):
        A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])


def test_full_tensor_square_signed_cross_and_zero_denominator():
    actual = {"00": np.array([0., 0.]), "10": np.array([3., 1.]), "01": np.array([-2., -1.]), "11": np.array([1., 2.])}
    predicted = {key: -value for key, value in actual.items()}
    result = A["tensor_square"](actual, predicted)
    assert result["energy"]["closure"]["rms"] == 0
    assert result["energy"]["signed_cross_terms"]["H_at_sham_M__M_at_sham_H"] < 0
    assert result["energy"]["sum_terms_and_cross"] == result["energy"]["total_squared_l2"]
    actual["11"] = actual["00"]
    zero = A["tensor_square"](actual, predicted)
    assert zero["native_joint_rms"] == 0
    assert zero["contrasts"]["interaction"]["error_relative_to_joint_rms"] is None


def test_signed_zero_operand_bits_and_rounding_collapse_are_observations(tmp_path):
    bits = np.array([0, 32768, 16256], dtype="<u2")
    file = tmp_path / "operand.npy"
    np.save(file, bits, allow_pickle=False)
    record = dict(path=file.name, sha256=A["sha"](file), bytes=file.stat().st_size,
        encoding="bf16-little-endian-uint16-npy-v1", payload_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
        descriptor=dict(dtype="torch.bfloat16", shape=[3], stride=[1], sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
    decoded = A["operand"](tmp_path, record)
    assert np.array_equal(A["bits"](decoded), bits)
    own, sham = np.array([1., 0.], np.float32), np.array([1., -0.], np.float32)
    report = A["novelty"](own, own, sham, {"own": own, "sham": sham}, own, sham, own, own)
    assert report["equal_recipient_own"] and not report["equal_sham"]
    assert not report["numerically_new_concat"] and report["H_own_vs_sham_changed_values"] == 1


def test_joint_restoration_opposite_sign_is_retained_without_quality_gate():
    z = np.array([0., 0.], np.float32)
    one = np.array([1., 1.], np.float32)
    entries = {"sham": {"image": z, "text": z}, "HH": {"image": one, "text": one}, "HL": {"image": one, "text": one}}
    baselines = {stage: {stream: z for stream in ("image", "text")} for stage in ("local", "final")}
    values = {stage: {role + "." + corner: {stream: one * (5 if role == "HH" and corner == "00" else 1)
        for stream in ("image", "text")} for role in ("HH", "HL") for corner in A["CORNERS"]} for stage in ("local", "final")}
    report = A["panel_accounts"](values, baselines, entries)
    assert report["signed_joint_restoration"]["local"]["image"] == 4
    assert report["hypothesis"]["observations"]["local"]["image"]["status"] == "opposite-direction"
    assert report["hypothesis"]["observations"]["local"]["text"]["supports_predeclared_direction"] is None
    assert not report["hypothesis"]["quality_gate"]


def test_compact_worker_finalization_retains_no_bulk_arrays(tmp_path):
    data, summary, ready = captured(tmp_path)
    config = dict(native_root=str(data), native_manifest_sha256=ready["files_manifest_sha256"],
                  native_request_sha256=summary["request_sha256"], output_scratch=str(tmp_path / "compact"))
    request = dict(config=config, request_sha256="2" * 64, source_hashes={p.name: A["sha"](p) for p in [PATH, *[HERE / "qualification" / name for name in A["HELPERS"]]]})
    request_file, config_file = tmp_path / "request.json", tmp_path / "config.json"
    request_file.write_text(json.dumps(request))
    config_file.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SATURN_RESIDENT_WORKER": "1", "SATURN_DEBUG_REQUEST": str(request_file),
           "SATURN_RUN_CONFIG": str(config_file), "MRUN_JOB_ID": "job-tiny-writer-audit"}
    process = subprocess.run([sys.executable, "-I", "-B", str(PATH)], cwd=HERE / "qualification", env=env, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    marker = json.loads(next(line.split(" ", 1)[1] for line in process.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(marker["output_root"])
    assert marker["files_manifest_sha256"] == A["sha"](output / "files.json")
    assert not (output / "typed-store").exists() and not (output / "operands").exists()
    assert (output / "retrieve_evidence.py").is_file()

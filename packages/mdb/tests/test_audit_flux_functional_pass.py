"""Installed tiny saved-state qualification; no neural calls in the auditor."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/audit_flux_functional_pass.py"
A = runpy.run_path(str(PATH))
F = runpy.run_path(str(HERE / "tests/test_flux_functional_pass.py"))
W, M = F["W"], F["M"]


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    root = tmp_path_factory.mktemp("functional-audit-source")
    config = F["fixture"](root)
    config["request_sha256"] = "1" * 64
    with patch.dict(os.environ, {"MRUN_JOB_ID": "job-tiny-functional"}):
        summary = W["run"](config)
    ready = M["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return Path(config["output"]), summary, ready


def copied(saved, tmp_path):
    data, summary, ready = saved
    target = tmp_path / "raw"
    shutil.copytree(data, target)
    return target, summary, ready


def repin(data, name):
    value = json.loads((data / "files.json").read_text())
    value["files"][name] = dict(sha256=A["sha"](data / name), bytes=(data / name).stat().st_size)
    (data / "files.json").write_text(json.dumps(value))
    return A["sha"](data / "files.json")


def test_full_tiny_audit_restoration_closure_fields_witnesses_and_seven_records(saved, tmp_path):
    data, summary, ready = saved
    def no_forward(module, args):
        raise AssertionError("saved-only audit invoked a neural module")
    hook = torch.nn.modules.module.register_module_forward_pre_hook(no_forward)
    try:
        report, _ = A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    finally:
        hook.remove()
    assert report["new_model_calls"] == report["new_reference_forward_calls"] == report["new_recurrence_transitions"] == 0
    assert report["counts"]["forecast"]["module_entries"] == 0
    assert report["counts"]["native"]["module_entries"] == 400 and report["counts"]["consumers"]["module_entries"] == 152
    assert report["source_model_identity"] != report["verification_adapter_identity"]
    assert report["parameter_restoration"]["tensors"] > 20
    assert len(report["boundaries"]) == 4 and all(len(rows) == 7 for rows in report["boundaries"].values())
    assert all(value is None for value in report["first_covered_bitwise_mismatch"].values())
    assert all(not row["predicted"] for row in report["consumers"].values())
    assert all(row["distinct_parent_cut_identities"] for row in report["consumers"].values())
    index = json.loads((tmp_path / "audit/catalog/provenance.json").read_text())
    assert index["records"] == 7 and index["downstream_project_forecast_records"] == 0
    assert all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in index["queries"])


@pytest.mark.parametrize("kind", ("counter", "early-history", "consumer-refresh", "consumer-port"))
def test_coherently_reinventoried_bad_execution_or_consumer_refused(saved, tmp_path, kind):
    data, summary, _ = copied(saved, tmp_path)
    name = {"counter": "summary.json", "early-history": "timeline.jsonl", "consumer-refresh": "comparisons.json", "consumer-port": "consumer-ports/HH.predicted.json"}[kind]
    if kind == "early-history":
        rows = [json.loads(line) for line in (data / name).read_text().splitlines()]
        row = next(row for row in rows if row["kind"] == "historical_cut_read")
        rows.remove(row)
        rows.insert(0, row)
        (data / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        value = json.loads((data / name).read_text())
        if kind == "counter":
            value["counts"]["forecast"]["module_entries"] = 1
        elif kind == "consumer-refresh":
            value["consumers"]["HH"]["predicted"]["input_cut"] = value["consumers"]["HH"]["native_reference"]["input_cut"]
        else:
            next(row for row in value if row["id"] == "x_embedder.0")["fields"]["input.0"]["descriptor"]["sha256"] = "0" * 64
        (data / name).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="census|historical|native/future|consumer sidecar|hydration|actual consumer"):
        A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_input_whitelist_cannot_include_future_or_unrelated_data(saved, tmp_path):
    data, summary, _ = copied(saved, tmp_path)
    config = json.loads((data / "plan.json").read_text())
    inputs = tmp_path / "source-with-extra"
    shutil.copytree(config["input_root"], inputs)
    extra = inputs / "future.json"
    extra.write_text("{}")
    config["input_root"] = str(inputs)
    config["input_files"][extra.name] = dict(sha256=A["sha"](extra), bytes=extra.stat().st_size)
    (data / "plan.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="future evidence"):
        A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, "plan.json"), expected_request=summary["request_sha256"])


def test_recorded_primitive_precision_miss_is_retained_with_first_covered_location(saved, tmp_path):
    data, summary, _ = copied(saved, tmp_path)
    name = "native-ports/HH/00.json"
    value = json.loads((data / name).read_text())
    next(row for row in value["ports"] if row["id"] == "x_embedder.0")["fields"]["output"]["descriptor"]["sha256"] = "0" * 64
    (data / name).write_text(json.dumps(value))
    report, _ = A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])
    first = report["first_covered_bitwise_mismatch"]["HH"]
    assert first["number"] == 0 and "x_embedder.0" in first["primitive_ports"]
    assert report["mechanics_status"] == "authenticated-full-state-saved-evidence-audit"


def test_bitwise_signed_zero_and_bf16_rounding_are_explicit():
    p, n = np.array([-0., 1.], np.float32), np.array([0., 1.], np.float32)
    row = A["metric"](p, n)
    assert row["exact"] and row["changed_values"] == 0 and not row["bitwise_exact"]
    values = np.array([1.00390625, 1.01171875, -0.], np.float32)
    assert np.array_equal(A["rne_bf16"](values).view(np.uint32), np.array([1., 1.015625, -0.], np.float32).view(np.uint32))
    assert A["metric"](np.ones(2), np.zeros(2))["relative_rms"] is None
    state = {key: None for key in ("image", "text", "context", "noise_prediction")}
    comparison = A["compare_states"]({**state, "latent": torch.ones(2, dtype=torch.bfloat16)}, {**state, "latent": torch.ones(2)})
    assert comparison["latent"]["exact"] and not comparison["latent"]["dtype_exact"] and not comparison["latent"]["bitwise_exact"]


def test_actual_compact_worker_cli_has_no_bulk_model_or_state_copy(saved, tmp_path):
    data, summary, ready = saved
    config = dict(native_root=str(data), native_manifest_sha256=ready["files_manifest_sha256"], native_request_sha256=summary["request_sha256"], output_scratch=str(tmp_path / "compact"))
    request = dict(config=config, request_sha256="2" * 64, source_hashes={p.name: A["sha"](p) for p in [PATH, *[HERE / "qualification" / name for name in A["HELPERS"]]]})
    request_path, config_path = tmp_path / "request.json", tmp_path / "config.json"
    request_path.write_text(json.dumps(request))
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SATURN_RESIDENT_WORKER": "1", "SATURN_DEBUG_REQUEST": str(request_path), "SATURN_RUN_CONFIG": str(config_path), "MRUN_JOB_ID": "job-tiny-functional-audit"}
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH)], cwd=HERE / "qualification", env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(marker["output_root"])
    assert marker["files_manifest_sha256"] == A["sha"](output / "files.json")
    assert not (output / "prediction-store").exists() and not (output / "native-store").exists()
    assert (output / "retrieve_evidence.py").is_file()

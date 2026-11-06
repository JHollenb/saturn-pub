"""Saved-only decoder audit, using public synthetic CPU artifacts."""
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
PATH = HERE / "qualification/audit_flux_functional_decoder.py"
A = runpy.run_path(str(PATH))
T = runpy.run_path(str(HERE / "tests/test_flux_functional_decoder_campaign.py"))
W = T["W"]
M = runpy.run_path(str(HERE / "qualification/flux_image_tail_microscope.py"))


def produce(root, *, mismatch=False, refuse_prefix=False):
    config = T["fixture"](root)
    config["request_sha256"] = "1" * 64
    original_forward = torch.nn.Conv2d.forward
    original_helpers = W["run"].__globals__["helpers"]
    old_job = os.environ.get("MRUN_JOB_ID")
    state = dict(armed=False, diagnostic=False)
    def progress(row):
        state["armed"] = mismatch and row["kind"] == "native_decode_entry" and row["case"] == "HH"
        state["diagnostic"] = row["kind"] == "diagnostic_prefix_started"
    def forward(self, value):
        output = original_forward(self, value)
        if state["armed"]:
            state["armed"] = False
            return output + .125
        return output
    def helpers(cfg):
        graph, *others = original_helpers(cfg)
        original = graph["functional_decode"]
        def decode(vae, argument, **kwargs):
            if refuse_prefix and state["diagnostic"]:
                return vae.post_quant_conv(argument)
            return original(vae, argument, **kwargs)
        graph["functional_decode"] = decode
        return [graph, *others]
    try:
        os.environ["MRUN_JOB_ID"] = "job-tiny-decoder-source"
        torch.nn.Conv2d.forward = forward
        W["run"].__globals__["helpers"] = helpers
        summary = W["run"](config, progress=progress)
    finally:
        torch.nn.Conv2d.forward = original_forward
        W["run"].__globals__["helpers"] = original_helpers
        if old_job is None:
            os.environ.pop("MRUN_JOB_ID", None)
        else:
            os.environ["MRUN_JOB_ID"] = old_job
    root = Path(config["output"])
    ready = M["finalize"](root, job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return root, summary, ready


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    return produce(tmp_path_factory.mktemp("decoder-audit"))


@pytest.fixture(scope="module")
def saved_miss(tmp_path_factory):
    return produce(tmp_path_factory.mktemp("decoder-audit-miss"), mismatch=True)


def audit(saved, output):
    root, summary, ready = saved
    return A["audit"](root, output, expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])[0]


def test_complete_saved_audit_no_module_calls(saved, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("saved-only auditor entered neural module")
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    report = audit(saved, tmp_path / "audit")
    assert report["coverage"]["full_macros_per_case"] == 10
    assert report["coverage"]["full_micro_arrays"] is False
    assert report["totals"]["native_module_entries"] == 264
    assert report["totals"]["forecast_module_entries"] == report["totals"]["diagnostic_module_entries"] == 0
    assert report["native_contract"]["frontend"]["bn_multiply_device"] == "cpu"
    assert report["native_contract"]["frontend"]["decoder_argument_device"] == "cpu"
    assert report["first_macro_difference"] == report["first_micro_descriptor_difference"] == dict.fromkeys(W["CASES"])
    assert len(report["signed_pair_responses"]) == 3
    assert report["parameter_restoration"]["tensors"] > 30
    assert all(v["bitwise_exact"] for row in report["macro_comparisons"].values() for v in row.values())
    assert all(len(row) == 61 for row in report["primitive_witnesses"].values())
    provenance = json.loads((tmp_path / "audit/catalog/provenance.json").read_text())
    assert provenance["records"] == 4
    assert all(row["exact"] == 1 and row["wrong_clock"] == 0 for row in provenance["queries"])


def repin(root, name, change):
    path = root / name
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))
    files = json.loads((root / "files.json").read_text())
    files["files"][name] = dict(bytes=path.stat().st_size, sha256=A["sha"](path))
    (root / "files.json").write_text(json.dumps(files))
    return A["sha"](root / "files.json")


@pytest.mark.parametrize("fault", ["counter", "early-native", "operand", "future-input"])
def test_hostile_metadata_refused(saved, tmp_path, fault):
    root = tmp_path / "raw"
    shutil.copytree(saved[0], root)
    if fault == "counter":
        digest = repin(root, "summary.json", lambda row: row["totals"].update(forecast_module_entries=1))
    elif fault == "operand":
        digest = repin(root, "native-macros.json", lambda row: row["HH"]["decoder_input"]["descriptor"].update(sha256="0" * 64))
    elif fault == "future-input":
        def change(row):
            row["input_files"]["observed-decoder-output.npy"] = dict(bytes=0, sha256="0" * 64)
        digest = repin(root, "plan.json", change)
    else:
        path = root / "timeline.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        index = next(i for i, row in enumerate(rows) if row["kind"] == "native_decode_entry")
        rows.insert(0, rows.pop(index))
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        files = json.loads((root / "files.json").read_text())
        files["files"]["timeline.jsonl"] = dict(bytes=path.stat().st_size, sha256=A["sha"](path))
        (root / "files.json").write_text(json.dumps(files))
        digest = A["sha"](root / "files.json")
    with pytest.raises(ValueError):
        A["audit"](root, tmp_path / "out", expected_manifest=digest, expected_request=saved[1]["request_sha256"])
    assert not (tmp_path / "out").exists()
    if fault == "counter":
        # Equal CPU values do not license a different declared frontend device.
        second = tmp_path / "wrong-frontend"
        shutil.copytree(saved[0], second)
        digest = repin(second, "native-contract.json", lambda row: row["frontend"].update(bn_multiply_device="cuda"))
        with pytest.raises(ValueError, match="CPU frontend"):
            A["audit"](second, tmp_path / "frontend-out", expected_manifest=digest, expected_request=saved[1]["request_sha256"])


def test_miss_retained_with_hash_matched_postobservation_reconstruction(saved_miss, tmp_path):
    report = audit(saved_miss, tmp_path / "audit")
    assert report["first_micro_descriptor_difference"]["HH"]["id"] == "post_quant_conv.0"
    diagnostic = report["diagnostic_reconstruction"]["HH"]
    assert diagnostic["diagnostic_integrity_status"] == "verified-hash-matched-reconstruction"
    assert diagnostic["independently_reduced_full_pair"]["rms"] > 0
    assert report["totals"]["native_decode_calls"] == 4
    assert report["totals"]["diagnostic_prefix_calls"] == 1
    assert report["coverage"]["full_micro_arrays"] is False


def test_replay_hash_sidecar_substitution_refused(saved_miss, tmp_path):
    root = tmp_path / "raw"
    shutil.copytree(saved_miss[0], root)
    digest = repin(root, "summary.json", lambda row: row["diagnostics"]["HH"]["replay_descriptor"].update(sha256="0" * 64))
    with pytest.raises(ValueError):
        A["audit"](root, tmp_path / "out", expected_manifest=digest, expected_request=saved_miss[1]["request_sha256"])


def test_refused_optional_reconstruction_preserves_primary(tmp_path):
    saved = produce(tmp_path / "source", mismatch=True, refuse_prefix=True)
    report = audit(saved, tmp_path / "audit")
    assert report["totals"]["forecast_module_entries"] == 0
    assert report["diagnostic_instrument"]["module_entry_attempts"] == 1
    assert report["diagnostic_instrument"]["primary_evidence_preserved"] is True
    assert report["diagnostic_reconstruction"]["HH"]["independently_reduced_full_pair"] is None
    assert report["macro_comparisons"]["HH"]["decoded"]["rms"] > 0


def test_signed_zero_and_undefined_denominators():
    class Array:
        def __init__(self, values):
            self.array = np.array(values, np.float32)
            self.shape, self.size = self.array.shape, self.array.size
            self.descriptor = dict(dtype="torch.float32", stride=[1])
        def pieces(self, start, stop, **kwargs):
            return self.array[start:stop]
    row = A["stream_metric"](Array([0., -0.]), Array([0., 0.]))
    assert row["rms"] == 0 and row["bitwise_exact"] is False and row["relative_rms"] is None
    row = A["stream_metric"](Array([-2., 0.]), Array([-2., 0.]), p_sham=Array([0., 0.]), n_sham=Array([0., 0.]))
    assert row["rms"] == 0 and row["native_signed_mean"] == -1


def test_endpoint_catalog_budget_above_one_million():
    arrays = dict(decoded=np.zeros((1, 3, 512, 512), np.float32), pixels=np.zeros((1, 512, 512, 3), np.uint8))
    report = A["endpoint_comparison"](arrays, arrays)
    assert sum(v["residual"]["elements"] for v in report["fields"].values()) == 1_572_864


def test_actual_compact_worker_cli(saved, tmp_path):
    root, summary, ready = saved
    config = dict(native_root=str(root), native_manifest_sha256=ready["files_manifest_sha256"], native_request_sha256=summary["request_sha256"], output_scratch=str(tmp_path / "outputs"))
    request = dict(config=config, request_sha256="2" * 64, source_hashes={str(PATH): A["sha"](PATH)})
    request_path, config_path = tmp_path / "request.json", tmp_path / "config.json"
    request_path.write_text(json.dumps(request))
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "SATURN_RESIDENT_WORKER": "1", "SATURN_DEBUG_REQUEST": str(request_path), "SATURN_RUN_CONFIG": str(config_path), "MRUN_JOB_ID": "job-tiny-audit"}
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH)], cwd=HERE / "qualification", env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    capsule = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(capsule["output_root"])
    assert A["sha"](output / "files.json") == capsule["files_manifest_sha256"]
    assert len(list(output.glob("*-images/*.png"))) == 8
    assert not (output / "forecast-store").exists() and not (output / "native-store").exists()
    assert (output / "retrieve_evidence.py").is_file()

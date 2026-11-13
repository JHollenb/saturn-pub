"""Tiny saved-cut evidence audit; model forwards are forbidden during audit."""
from __future__ import annotations

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

from saturn_pub.core import StateCut
from saturn_pub.store import LocalStore

HERE = Path(__file__).parents[1]
PATH = HERE / "qualification/audit_flux_single_stack_trace.py"
A = runpy.run_path(str(PATH))
F = runpy.run_path(str(HERE / "tests/test_flux_single_stack_trace.py"))
C, M = F["C"], F["M"]


def captured(root):
    config, parents, expected = F["fixture"](root)
    source_root = Path(config["input_root"])
    for role, parent in parents.items():
        p = parent.payload
        native_exit = StateCut(parent.model_identity, dict(parent.execution), "readout", parent.fingerprint,
            {**p, **expected[role], "phase": "readout", "cursor": 3},
            dict(logical_step=1, phase="readout", index=None, local_clock=3, next_operation="norm_and_projection"))
        LocalStore(source_root / "typed-store").save(native_exit)
        ledger = source_root / config["inputs"][role]["ledger"]
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        rows[25]["capture"] = dict(cut=native_exit.fingerprint, retained_full_cut=True)
        ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config["input_files"] = {path.relative_to(source_root).as_posix(): dict(sha256=M["sha"](path), bytes=path.stat().st_size)
                             for path in source_root.rglob("*") if path.is_file()}
    config["request_sha256"] = "1" * 64
    with patch.dict(os.environ, {"MRUN_JOB_ID": "job-tiny-native-stack"}):
        summary = C["run"](config)
    ready = C["finalize_directory"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"])
    return Path(config["output"]), summary, ready


def repin(data, name):
    manifest = json.loads((data / "files.json").read_text())
    manifest["files"][name] = dict(sha256=A["sha"](data / name), bytes=(data / name).stat().st_size)
    (data / "files.json").write_text(json.dumps(manifest))
    return A["sha"](data / "files.json")


def test_actual_tiny_full_cut_trace_to_compact_observations(tmp_path):
    data, summary, ready = captured(tmp_path)
    def no_forward(module, args):
        raise AssertionError("auditor attempted a model/module forward")
    hook = torch.nn.modules.module.register_module_forward_pre_hook(no_forward)
    try:
        report, _ = A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    finally:
        hook.remove()
    assert report["full_cut_count"] == 12 and len(report["curves"]) == 4
    assert report["counts"]["native_blocks"] == 9
    assert report["new_model_calls"] == report["new_reference_forward_calls"] == report["new_forecasts"] == 0
    assert all(value[stream]["exact_descriptor"] for value in report["historical_replay"].values() for stream in ("image", "text"))
    for value in report["historical_replay"].values():
        assert value["image"]["comparison"]["orientation"] == "traced-minus-original-historical-exit"
    records = json.loads((tmp_path / "audit/trace-records.json").read_text())
    assert len(records["records"]) == 4 and not any(row["equation_forecast"] for row in records["records"])
    assert report["source_provenance"]["scheduler_status"] == "failed"


@pytest.mark.parametrize("kind", ("count", "call-parent", "point"))
def test_coherently_reinventoried_false_census_call_or_cursor_refused(tmp_path, kind):
    data, summary, _ = captured(tmp_path)
    name = {"count": "summary.json", "call-parent": "calls/00.json", "point": "boundaries/HH/01.json"}[kind]
    row = json.loads((data / name).read_text())
    if kind == "count":
        row["counts"]["native_blocks"] += 1
    elif kind == "call-parent":
        row["calls"][1]["input_cut"] = "0" * 64
    else:
        row["point"]["local_clock"] = 99
    (data / name).write_text(json.dumps(row))
    with pytest.raises(ValueError, match="census|join|point"):
        A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, name), expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_original_pre_readout_full_cut_is_required_not_only_descriptors(tmp_path):
    data, summary, ready = captured(tmp_path)
    config = json.loads((data / "plan.json").read_text())
    source = Path(config["input_root"])
    ledger = [json.loads(line) for line in (source / config["inputs"]["HH"]["ledger"]).read_text().splitlines()]
    file = source / "typed-store/cuts" / (ledger[25]["capture"]["cut"] + ".json")
    file.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        A["audit"](data, tmp_path / "audit", expected_manifest=ready["files_manifest_sha256"], expected_request=summary["request_sha256"])
    assert not (tmp_path / "audit").exists()


def test_changed_producer_metrics_are_diagnostic_and_recomputed_fields_win(tmp_path):
    data, summary, _ = captured(tmp_path)
    path = data / "accounts.json"
    row = json.loads(path.read_text())
    row["boundaries"][1]["responses"]["HH"]["image"]["rms"] = -999
    path.write_text(json.dumps(row))
    report, _ = A["audit"](data, tmp_path / "audit", expected_manifest=repin(data, "accounts.json"), expected_request=summary["request_sha256"])
    assert not report["producer_diagnostics"]["producer_full_account_rows_exact"]
    assert report["curves"][1]["responses"]["HH"]["image"]["rms"] >= 0


def test_signed_negative_increment_and_partition_units_are_preserved():
    fields = {role: {stream: np.array([0., 0.]) for stream in ("image", "text")} for role in A["ROLES"]}
    fields["HH"]["image"] = np.array([4., -4.])
    first = A["responses"](fields, None)
    fields["HH"]["image"] = np.array([1., -1.])
    second = A["responses"](fields, first)
    assert second["HH"]["image"]["rms_absolute_increment"] == -3
    assert second["HH"]["image"]["relative_to_sham_rms"] is None
    curve = [dict(boundary=0, after_block=None, responses=first), dict(boundary=1, after_block=0, responses=second)]
    summary = A["summarize"](curve, 1)
    assert summary["partitions"]["HH"]["image"]["early"]["signed_absolute_rms_change"] == -3
    assert summary["top5_by_absolute_size_of_signed_rms_increment"]["HH"]["image"][0]["rms_absolute_increment"] == -3


def test_compact_worker_actual_finalization_and_checksum_retriever(tmp_path):
    data, summary, ready = captured(tmp_path)
    config = dict(native_root=str(data), native_manifest_sha256=ready["files_manifest_sha256"], native_request_sha256=summary["request_sha256"],
                  output_scratch=str(tmp_path / "compact"))
    request = dict(config=config, request_sha256="2" * 64,
                   source_hashes={path.name: A["sha"](path) for path in [PATH, *[HERE / "qualification" / name for name in A["HELPERS"]]]})
    request_path, config_path = tmp_path / "request.json", tmp_path / "config.json"
    request_path.write_text(json.dumps(request))
    config_path.write_text(json.dumps(config))
    env = {**os.environ, "SATURN_RESIDENT_WORKER": "1", "SATURN_DEBUG_REQUEST": str(request_path), "SATURN_RUN_CONFIG": str(config_path),
           "CUDA_VISIBLE_DEVICES": "", "MRUN_JOB_ID": "job-tiny-stack-audit"}
    result = subprocess.run([sys.executable, "-I", "-B", str(PATH)], cwd=HERE / "qualification", env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = json.loads(next(line.split(" ", 1)[1] for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")))
    output = Path(marker["output_root"])
    assert marker["files_manifest_sha256"] == A["sha"](output / "files.json")
    assert not (output / "typed-store").exists()
    external = json.loads((output / "external-evidence.json").read_text())
    assert any("typed-store/blobs/" in name for name in external["manifests"]["native"])
    assert (output / "retrieve_evidence.py").is_file()

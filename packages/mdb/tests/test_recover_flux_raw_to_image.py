"""Packaging recovery mechanics with real transaction and frozen finalizer."""
from __future__ import annotations

import copy
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.forecast_transaction import ForecastTransaction, verify_transaction

HERE = Path(__file__).resolve().parents[1] / "qualification"
R = runpy.run_path(str(HERE / "recover_flux_raw_to_image.py"))


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(tmp_path, *, failed_callback=False, large=False):
    root = tmp_path / "original"
    root.mkdir()
    runtime = root / "runtime-contract.json"
    put(runtime, {"kind": "synthetic-model-free-runtime"})
    tx = ForecastTransaction(root / "forecast-transaction", model_identity="synthetic-no-model",
        input_identity="tiny-input", source_identity="tiny-source", frontier={"kind": "raw-input", "clock": 0},
        budgets={"max_predictions": 1, "max_executions": 1, "max_bytes": 1 << 20})
    tx.predict("raw-composition-forecast", lambda _: {"predictions": {"value": 3}})
    seal = tx.seal()
    if failed_callback:
        def broken(_):
            raise ValueError("synthetic native callback failure")
        with pytest.raises(ValueError, match="synthetic"):
            tx.execute("native_qwen_flux_and_vae_targets", broken)
    else:
        tx.execute("native_qwen_flux_and_vae_targets", lambda _: {"value": 3})
    tx.close()
    verify = verify_transaction(root / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"])
    helpers = {"flux_image_tail_microscope.py": R["FINALIZER_SHA256"]}
    plan = {"helper_source_sha256": helpers, "forecast_transaction_bytes": 1 << 20}
    request = {"config": plan, "source_hashes": {"flux_raw_to_image.py": "a" * 64, **helpers}}
    request["request_sha256"] = R["identity"](request)
    counts = {"native_transitions": 2, "model_calls": 0}
    source = dict(job_id="job-tiny-failed", request_sha256=request["request_sha256"], worker_sha256="a" * 64,
        helper_source_sha256=helpers, physical_counts=counts, forecast_seal=seal, qwen_tied_alias=True,
        placement_records=[], runtime_contract_sha256=R["sha"](runtime), source_files={}, tokenizer_files={})
    contract = dict(counts=counts, forecast_seal=seal, qwen_tied_alias=True, placement_records=[])
    for component in ("qwen", "flux", "vae"):
        for side in ("before", "after"):
            signature = {"tiny.weight": {"shape": [1], "stride": [1], "dtype": "torch.bfloat16", "device": "cpu", "sha256": "b" * 64}}
            source[component + "_parameter_identity_" + side] = copy.deepcopy(signature)
            contract[component + "_identity_" + side] = copy.deepcopy(signature)
    summary = dict(job_id=source["job_id"], request_sha256=request["request_sha256"], mechanics_status="complete",
        counts=counts, forecast_seal=seal, transaction=verify)
    effective = {**plan, "job_id": source["job_id"], "request_sha256": request["request_sha256"], "output": str(root)}
    for name, value in zip(R["CORE"], (effective, request, source, summary, contract), strict=True):
        put(root / name, value)
    if large:
        # A valid sparse uint16 NPY: exactly 128 MiB payload plus 128 B header.
        import numpy as np
        path = root / "decoder-up-block.npy"
        with path.open("wb") as stream:
            np.lib.format.write_array_header_1_0(stream, {"descr": "<u2", "fortran_order": False, "shape": (1, 256, 512, 512)})
            assert stream.tell() == 128
            stream.truncate((128 << 20) + 128)
    job = dict(job_id=source["job_id"], state="failed", config=request, cmd=["python", "frozen-worker.py"],
        submitted_request={"cmd": ["python", "frozen-worker.py"]},
        result=dict(returncode=1, executed_payload={"sha256": "c" * 64}, diagnostics={"command": ["python", "frozen-worker.py"]}),
        payload_custody={key: {"sha256": "c" * 64} for key in ("declared", "sealed", "executed")})
    job_path, log_path, request_path, inventory_path = (tmp_path / name for name in ("job.json", "worker.log", "request.json", "retained.json"))
    put(job_path, job)
    log_path.write_text("synthetic model-free capture complete\nValueError: output member bound exceeded\n")
    put(request_path, request)
    retained = R["inventory"](root, job_id=job["job_id"], request_sha256=request["request_sha256"])
    put(inventory_path, retained)
    config = dict(schema=R["SCHEMA"], original_root=str(root), max_member_bytes=R["MEMBER_BYTES"],
        max_expanded_bytes=R["EXPANDED_BYTES"], expected_source_job_id=job["job_id"], expected_source_request_sha256=request["request_sha256"],
        expected_source_worker_sha256="a" * 64, expected_source_payload_sha256="c" * 64,
        finalizer=str(HERE / "flux_image_tail_microscope.py"), finalizer_sha256=R["FINALIZER_SHA256"])
    for key, path in (("original_job", job_path), ("original_log", log_path), ("original_request", request_path), ("retained_inventory", inventory_path)):
        config[key], config[key + "_sha256"] = str(path), R["sha"](path)
    return root, config, retained


def test_real_finalizer_recovers_npy_header_bound_without_original_writes(tmp_path):
    root, config, retained = fixture(tmp_path, large=True)
    frozen = runpy.run_path(config["finalizer"])
    with pytest.raises(ValueError, match="member bound"):
        frozen["finalize"](root)
    assert not (root / "files.json").exists()
    before = {name: path.stat() for name, path in R["regular_members"](root).items()}
    view, receipt_path = tmp_path / "view", tmp_path / "recovery.json"
    receipt = R["prepare"](config, view, receipt_path)
    assert receipt["original_producer_status"] == "failed" and receipt["original_returncode"] == 1
    assert receipt["new_neural_calls"] == receipt["new_parameter_hydrations"] == 0
    assert receipt["max_member_bytes"] == 134221824
    assert receipt["original_largest_member_bytes"] == 134217856
    assert receipt["final_resident_parameter_check"] == "established-by-saved-before-after-core-records"
    manifest = R["read"](view / "files.json")
    assert manifest["files"] == retained["files"]
    assert manifest["job_id"] == config["expected_source_job_id"]
    assert manifest["request_sha256"] == config["expected_source_request_sha256"]
    assert not (root / "files.json").exists() and receipt_path.parent != view
    assert R["inventory"](root, job_id=manifest["job_id"], request_sha256=manifest["request_sha256"])["files"] == retained["files"]
    for name, old in before.items():
        original, linked = (root / name).stat(), (view / name).stat()
        assert (old.st_dev, old.st_ino, old.st_size, old.st_mtime_ns) == (original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns)
        assert (original.st_dev, original.st_ino) == (linked.st_dev, linked.st_ino)
        assert original.st_nlink == old.st_nlink + 1


@pytest.mark.parametrize("fault", ["changed-bytes", "symlink", "resident-miss", "false-success", "wrong-request", "failed-journal"])
def test_hostile_saved_custody_refused_before_linking(tmp_path, fault):
    root, config, _ = fixture(tmp_path, failed_callback=fault == "failed-journal")
    if fault == "changed-bytes":
        (root / "runtime-contract.json").write_text("changed bytes\n")
    elif fault == "symlink":
        os.symlink(root / "summary.json", root / "unexpected-link")
    elif fault == "resident-miss":
        source = R["read"](root / "source.json")
        source["qwen_parameter_identity_after"]["tiny.weight"]["sha256"] = "d" * 64
        put(root / "source.json", source)
        put(Path(config["retained_inventory"]), R["inventory"](root,
            job_id=config["expected_source_job_id"], request_sha256=config["expected_source_request_sha256"]))
        config["retained_inventory_sha256"] = R["sha"](config["retained_inventory"])
    elif fault == "false-success":
        job = R["read"](config["original_job"])
        job["state"] = "succeeded"
        put(Path(config["original_job"]), job)
        config["original_job_sha256"] = R["sha"](config["original_job"])
    elif fault == "wrong-request":
        config["expected_source_request_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        R["prepare"](config, tmp_path / "refused-view", tmp_path / "refused-receipt.json")
    assert not (tmp_path / "refused-view").exists()
    assert not (tmp_path / "refused-receipt.json").exists()
    assert not (root / "files.json").exists()


def test_receipt_cannot_mutate_view_or_original(tmp_path):
    root, config, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        R["prepare"](config, tmp_path / "view", root / "new-receipt.json")
    assert not (root / "new-receipt.json").exists()
    assert not (tmp_path / "view").exists()


def test_existing_shared_inode_aliases_have_correct_link_multiplicity(tmp_path):
    root, config, _ = fixture(tmp_path)
    original, alias = root / "runtime-contract.json", root / "runtime-alias.json"
    os.link(original, alias)
    put(Path(config["retained_inventory"]), R["inventory"](root,
        job_id=config["expected_source_job_id"], request_sha256=config["expected_source_request_sha256"]))
    config["retained_inventory_sha256"] = R["sha"](config["retained_inventory"])
    old = original.stat()
    receipt = R["prepare"](config, tmp_path / "view", tmp_path / "receipt.json")
    assert receipt["original_shared_inode_groups"] == 1
    assert original.stat().st_nlink == alias.stat().st_nlink == old.st_nlink + 2
    assert (tmp_path / "view/runtime-contract.json").stat().st_ino == (tmp_path / "view/runtime-alias.json").stat().st_ino == old.st_ino
    assert not (root / "files.json").exists()


def test_same_filesystem_is_required_before_any_hardlink(tmp_path, monkeypatch):
    root, config, _ = fixture(tmp_path)
    parent = tmp_path / "other-device"
    parent.mkdir()
    original = Path.stat

    def changed_device(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == parent:
            return os.stat_result((result.st_mode, result.st_ino, result.st_dev + 1, result.st_nlink,
                result.st_uid, result.st_gid, result.st_size, result.st_atime, result.st_mtime, result.st_ctime))
        return result

    monkeypatch.setattr(Path, "stat", changed_device)
    with pytest.raises(ValueError, match="same filesystem"):
        R["prepare"](config, parent / "view", tmp_path / "receipt.json")
    assert not (parent / "view").exists() and not (root / "files.json").exists()


def test_actual_cpu_recovery_cli_and_unchanged_saved_auditor(tmp_path):
    """Reapply the existing synthetic saved-array fixture, without model forwards."""
    fixture_module = runpy.run_path(str(Path(__file__).parent / "test_audit_flux_raw_to_image.py"))
    original_save = fixture_module["A"]["save"]

    def save_with_original_output(path, value):
        if Path(path).name == "plan.json":
            value = {**value, "output": str(Path(path).parent)}
        original_save(path, value)

    fixture_module["A"]["save"] = save_with_original_output
    root, summary = fixture_module["saved_fixture"](tmp_path / "saved-fixture")
    (root / "files.json").unlink()
    summary["mechanics_status"] = "complete"
    put(root / "summary.json", summary)
    for name in ("source.json", "native-contract.json"):
        value = R["read"](root / name)
        value["forecast_seal"] = summary["forecast_seal"]
        put(root / name, value)
    original_request = R["read"](root / "request.json")
    job = dict(job_id=summary["job_id"], state="failed", config=original_request,
        cmd=["synthetic-saved-only"], submitted_request={"cmd": ["synthetic-saved-only"]},
        result=dict(returncode=1, executed_payload={"sha256": "c" * 64}, diagnostics={"command": ["synthetic-saved-only"]}),
        payload_custody={key: {"sha256": "c" * 64} for key in ("declared", "sealed", "executed")})
    config = dict(schema=R["SCHEMA"], original_root=str(root), max_member_bytes=R["MEMBER_BYTES"],
        max_expanded_bytes=R["EXPANDED_BYTES"], expected_source_job_id=job["job_id"],
        expected_source_request_sha256=summary["request_sha256"], expected_source_worker_sha256=fixture_module["A"]["WORKER_SHA256"],
        expected_source_payload_sha256="c" * 64, finalizer=str(HERE / "flux_image_tail_microscope.py"), finalizer_sha256=R["FINALIZER_SHA256"],
        auditor=str(HERE / "audit_flux_raw_to_image.py"), auditor_sha256=R["AUDITOR_SHA256"], output_scratch=str(tmp_path / "outputs"))
    packet = {"original_job": job, "original_request": original_request,
        "retained_inventory": R["inventory"](root, job_id=job["job_id"], request_sha256=summary["request_sha256"])}
    for key, value in packet.items():
        path = tmp_path / (key + ".json")
        put(path, value)
        config[key], config[key + "_sha256"] = str(path), R["sha"](path)
    log = tmp_path / "original.log"
    log.write_text("ValueError: output member bound exceeded\n")
    config.update(original_log=str(log), original_log_sha256=R["sha"](log))
    config_path, request_path = tmp_path / "recovery-config.json", tmp_path / "recovery-request.json"
    request = dict(config=config, source_hashes={"recover_flux_raw_to_image.py": R["sha"](HERE / "recover_flux_raw_to_image.py")})
    request["request_sha256"] = R["identity"](request)
    put(config_path, config)
    put(request_path, request)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MRUN_JOB_ID": "job-tiny-recovery",
        "SATURN_RUN_CONFIG": str(config_path), "SATURN_DEBUG_REQUEST": str(request_path)}
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "recover_flux_raw_to_image.py")],
        cwd=HERE, env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    assert len(marker) == 1 and marker[0]["request_sha256"] == request["request_sha256"]
    output = Path(marker[0]["output_root"])
    assert R["sha"](output / "files.json") == marker[0]["files_manifest_sha256"]
    assert R["sha"](output / "summary.json") == marker[0]["summary_sha256"]
    assert R["read"](output / "summary.json")["original_producer_status"] == "failed"
    assert R["read"](output / "summary.json")["model_calls"] == 0
    assert R["read"](output / "recovery-receipt.json")["recovery_execution"]["request_sha256"] == request["request_sha256"]
    assert R["read"](output / "source-summary.json")["job_id"] == job["job_id"]
    assert R["read"](output / "audit.json")["new_neural_forward_calls"] == 0
    assert len(list(output.glob("*.png"))) == 3 and not list(output.rglob("*.npy"))
    assert R["read"](output / "catalog/provenance.json")["records"] == 2
    assert not (root / "files.json").exists()

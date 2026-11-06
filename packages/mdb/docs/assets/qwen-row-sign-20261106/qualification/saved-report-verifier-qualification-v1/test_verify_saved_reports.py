"""Compact-only actual tiny reports, real MDB CLIs, synthetic custody envelopes."""
import argparse
import copy
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
V = runpy.run_path(str(HERE / "mdb-qwen-row-sign-20261106/verify_saved_reports.py"))
SAVED = HERE / "saved-report"


def test_explicit_primary_window_channels_and_actual_native_effects():
    report = V["read"](SAVED / "audit.json")
    totals = V["rollup"](report)
    for channel in V["CHANNELS"]:
        for kind in ("endpoint", "response_error"):
            row = totals["channels"][channel][kind]
            assert row["whole"]["fields"] == 8
            assert row["windows"]["prefix"]["fields"] == row["windows"]["suffix"]["fields"] == 8
            assert row["whole"]["positions"] == sum(x["positions"] for x in row["windows"].values())
    assert totals["per_arm"]["prefix-negative"]["native_edit_effects"]["head"]["whole"]["rms"] > 0
    assert totals["positive_source_baseline"]["whole"]["fields"] == 4
    assert all(row["premises"]["metrics"]["whole"]["fields"] == 6 for row in totals["per_arm"].values())
    broken = copy.deepcopy(report)
    broken["arms"]["prefix-negative"]["channels"]["shortcut-minus-native"]["fields"]["head"]["endpoint"]["whole"]["sample_count"] += 1
    with pytest.raises(ValueError, match="whole/window"):
        V["rollup"](broken)


def compact_envelope(tmp_path):
    """Synthetic custody, with unchanged actual tiny scalar/catalog evidence."""
    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog

    native_run, audit_run, collection = (tmp_path / name for name in ("native", "audit", "collection"))
    native_run.mkdir(); audit_run.mkdir()
    pack = collection / "extracted/binding-minimal-result"
    shutil.copytree(SAVED, pack)
    native_job, audit_job = "job-synthetic-native", "job-synthetic-audit"
    native = dict(config=dict(output_scratch="/not-executed", output_name="raw", qwen_parameter_manifest_sha256="1" * 64, horizon=19, fixed_split=3),
        source_hashes={"qwen_row_sign_campaign.py": V["PRODUCER_SHA256"]})
    native["request_sha256"] = V["identity"](native)
    raw = dict(job_id=native_job, request_sha256=native["request_sha256"], files={}, expanded_bytes=0)
    V["save"](pack / "source-files.json", raw)
    audit = dict(config=dict(native_root="/not-executed/" + native_job + "/raw", native_request_sha256=native["request_sha256"],
        native_manifest_sha256=V["sha"](pack / "source-files.json"), output_scratch="/not-executed-audit"),
        source_hashes={"audit_qwen_row_sign.py": V["READER_SHA256"]})
    audit["request_sha256"] = V["identity"](audit)
    for run, request, job in ((native_run, native, native_job), (audit_run, audit, audit_job)):
        V["save"](run / "request.json", request); V["save"](run / "submission.json", dict(job_id=job))
    report = V["read"](pack / "audit.json")
    old_records = [V["read"](path) for path in (pack / "catalog/equations").glob("*.json")]
    source_h = old_records[0]["tested_context"]["input_identity"]
    report.update(job_id=native_job, request_sha256=native["request_sha256"], worker_sha256=V["PRODUCER_SHA256"], audit_sha256=V["READER_SHA256"],
        native_manifest_sha256=V["sha"](pack / "source-files.json"), source=dict(parameters={"scope": "synthetic custody only"}, parameter_recipe_io={}, runtime={}), parent={}, transaction={})
    (pack / "audit.json").unlink(); V["save"](pack / "audit.json", report)
    shutil.rmtree(pack / "catalog"); shutil.rmtree(pack / "atlas")
    catalog, ids = EquationCatalog(pack / "catalog/equations"), []
    for old in old_records:
        context = {**old["tested_context"], "source_identity": native["source_hashes"]}
        record = catalog.record(law_id=old["law_id"], route_id=old["route_id"], alias=old["alias"], report=old["report"], readers=old["readers"],
            tested_context=context, binding={key: context[key] for key in old["binding"]},
            conditions=[dict(field=key, op="eq", value=value) for key, value in context.items()], evidence={"scope": "synthetic custody; actual saved tiny scalar report"})
        ids.append(record["fingerprint"])
    ComputationAtlas(pack / "atlas").ingest(pack / "catalog/equations", ids,
        provenance=dict(run_id="synthetic-custody", artifacts=[dict(path="audit.json", sha256=V["sha"](pack / "audit.json"))]))
    for name, value in {"request.json": audit, "source-request.json": native,
        "source-plan.json": dict(native["config"], output=audit["config"]["native_root"], request_source_hashes=native["source_hashes"], request_sha256=native["request_sha256"], job_id=native_job),
        "source-declared-config.json": native["config"], "source-source.json": dict(source_hashes=native["source_hashes"], job_id=native_job, request_sha256=native["request_sha256"]),
        "source-source-frontier.json": dict(source_arrays={"h": dict(content_sha256=source_h)}),
        "source-summary.json": dict(job_id=native_job, request_sha256=native["request_sha256"], counts=report["census"]["observed"]),
        "summary.json": dict(job_id=audit_job, request_sha256=audit["request_sha256"], source_job=native_job)}.items():
        V["save"](pack / name, value)
    V["save"](pack / "files.json", {p.relative_to(pack).as_posix(): V["sha"](p) for p in pack.rglob("*") if p.is_file()})
    V["save"](collection / "collection-receipt.json", dict(files_manifest_sha256=V["sha"](pack / "files.json"), job_id=audit_job,
        request_sha256=audit["request_sha256"], native_job_id=native_job, reader_sha256=V["READER_SHA256"]))
    return argparse.Namespace(collection=collection, native_run=native_run, audit_run=audit_run, output=tmp_path / "verified",
        expected_native_request=native["request_sha256"], expected_native_manifest=V["sha"](pack / "source-files.json"), expected_audit_request=audit["request_sha256"],
        native_job=None, native_log=None, audit_job=None, audit_log=None)


def test_complete_compact_request_inventory_and_twelve_actual_cli_lookups(tmp_path):
    args = compact_envelope(tmp_path)
    result = V["verify"](args)
    assert len(result["actual_installed_cli"]) == 12
    assert sum(row["matches"] for row in result["actual_installed_cli"]) == 4
    assert all(row["status"] == "capture-not-provided" for row in result["jobs"].values())
    assert V["read"](args.output / "verification.json")["status"].startswith("Saved compact custody")
    source = args.collection / "extracted/binding-minimal-result/source-summary.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="compact member differs"):
        V["H"]["compact_account"](args.collection, source.parent)


def test_local_capture_command_reservation_payload_and_complete_log(tmp_path):
    request = dict(config={"scope": "model-free capture fixture"}, resource_exception=dict(ram_mb=1, vram_mb=0, cpu_threads=1))
    request["request_sha256"] = V["identity"](request)
    marker = dict(job_id="job-synthetic", request_sha256=request["request_sha256"], output_root="/not-executed", files_manifest_sha256="1" * 64, summary_sha256="2" * 64)
    command, payload = "not-executed synthetic command", dict(sha256="3" * 64, size_bytes=7)
    log = "MDB_DIRECTORY_READY " + json.dumps(marker) + "\n"
    (tmp_path / "worker.log").write_text(log)
    V["save"](tmp_path / "admission-command.json", dict(command=command))
    job = dict(job_id="job-synthetic", state="succeeded", config=dict(request, retry_on_kill=False),
        submitted_request=dict(config=request, reservation=request["resource_exception"], needs={"cuda": False}), payload_custody={name: payload for name in ("declared", "sealed", "executed")},
        result=dict(returncode=0, diagnostics=dict(command=command), executed_payload=payload, artifacts=dict(job_log=dict(bytes=len(log.encode()), sha256=V["sha"](tmp_path / "worker.log"))), elapsed_s=0., peak_rss_mb=0., peak_vram_mb=0.))
    V["save"](tmp_path / "job.json", job)
    result = V["captured_job"](tmp_path, request, "job-synthetic", tmp_path / "job.json", tmp_path / "worker.log", gpu=False, expected_marker=marker)
    assert result["status"] == "authenticated-local-capture"
    (tmp_path / "worker.log").write_text(log + "changed")
    with pytest.raises(ValueError, match="complete captured log"):
        V["captured_job"](tmp_path, request, "job-synthetic", tmp_path / "job.json", tmp_path / "worker.log", gpu=False, expected_marker=marker)


def test_help_has_only_explicit_local_job_capture_arguments():
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "mdb-qwen-row-sign-20261106/verify_saved_reports.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--native-job" in result.stdout and "--audit-log" in result.stdout
    assert "--expected-audit-request" in result.stdout and "--expected-native-manifest" in result.stdout

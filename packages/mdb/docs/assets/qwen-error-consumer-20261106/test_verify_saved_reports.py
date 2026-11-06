"""Synthetic compact scalars and local custody; installed CLI on tiny metadata."""
from __future__ import annotations

import copy
import hashlib
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
V = runpy.run_path(str(HERE / "mdb-qwen-error-consumer-20261106/verify_saved_reports.py"))


def metric(count, *, changed=0, defects=0, rms=0.):
    return dict(sample_count=count, bitwise_exact=not (changed or defects), changed_values=changed, maxabs=rms*2, rms=rms,
        predicted_rms=1., native_rms=1., signed_mean=-rms/10, signed_zero=dict(shared=defects, defects=defects))


def wrapped(name):
    count = V["POSITIONS"][name]
    return dict(whole=metric(count), windows=[dict(bounds=bounds, comparison=metric(count//2)) for bounds in ([0, 256], [256, 512])])


def scalar_report():
    protocol = dict(world_order=list(V["WORLDS"]), sequence_rows=512, windows=[[0, 256], [256, 512]])
    report = dict(worlds={world: dict(fields={name: wrapped(name) for name in V["PORTS"]}, comparison_elements=7340032,
        first_numeric_difference=None, first_bit_difference=None) for world in V["WORLDS"]},
        responses={name: {kind: wrapped(name) for kind in ("error", "functional", "native")} for name in V["PORTS"]},
        input_difference=dict(retained="synthetic input response"), residual_accounts=dict(retained="synthetic magnitude growth/cancellation countertrend"),
        coverage=dict(primary_endpoint_fields=10, primary_response_fields=5, dependent_endpoint_windows=20, dependent_response_windows=10))
    return report, protocol


def test_full_primary_census_preserves_zero_sign_and_signed_countertrends():
    report, protocol = scalar_report()
    head = report["worlds"]["full-input"]["fields"]["head"]
    head["whole"] = metric(V["POSITIONS"]["head"], defects=2)
    for part in head["windows"]:
        part["comparison"] = metric(V["POSITIONS"]["head"]//2, defects=1)
    report["responses"]["M"]["native"]["whole"]["signed_mean"] = -.25
    result = V["rollup"](report, protocol)
    assert result["endpoint_errors"]["whole"]["fields"] == 10 and result["endpoint_errors"]["whole"]["positions"] == 14680064
    assert result["response_errors"]["whole"]["fields"] == 5 and result["response_errors"]["whole"]["positions"] == 7340032
    assert result["endpoint_errors"]["dependent_windows"]["fields"] == 20 and result["response_errors"]["dependent_windows"]["fields"] == 10
    assert result["endpoint_errors"]["whole"]["changed_values"] == 0 and result["endpoint_errors"]["whole"]["signed_zero"]["defects"] == 2
    assert result["endpoint_errors"]["whole"]["logical_bit_changed_elements"] == 2
    assert result["per_field_responses"]["M"]["native"]["whole"]["signed_mean"] == -.25
    assert result["residual_accounts"] == report["residual_accounts"] and "label" in result["label_limit"]
    report["worlds"]["cheap-input"]["fields"]["R"]["windows"][0]["comparison"]["sample_count"] += 1
    with pytest.raises(ValueError, match="position coverage"):
        V["rollup"](report, protocol)


def capture_fixture(root, *, gpu=True):
    root.mkdir()
    command = ["dummy-no-execution"]
    request = dict(config=dict(synthetic=True), resource_exception=None, sizing=dict(mode="intent-first", reservation=None, timeout_s=300))
    reservation = dict(ram_mb=4096., vram_mb=4096., cpu_threads=4, disk_gb=1., est_wall_s=None, source="probe-default")
    if not gpu:
        request["resource_exception"] = dict(ram_mb=3072, vram_mb=0, cpu_threads=2, disk_gb=1, est_wall_s=240)
        reservation = {**request["resource_exception"], "source": "explicit"}
    marker = dict(job_id="job-synthetic", files_manifest_sha256="0"*64)
    log = ("MDB_DIRECTORY_READY " + json.dumps(marker) + "\n").encode()
    payload = dict(sha256="1"*64, size_bytes=123)
    job = dict(job_id="job-synthetic", state="succeeded", assigned_host="beast", cmd=command, reservation=reservation,
        config={**request, "retry_on_kill": False, "_plan_error": "unknown model; synthetic intent fixture"}, needs={"cuda": gpu},
        submitted_request=dict(config=request, cmd=command, reservation=None if gpu else reservation, needs={"cuda": gpu}),
        payload_custody={key: payload for key in ("declared", "sealed", "executed")},
        result=dict(returncode=0, diagnostics=dict(command=command), executed_payload=payload,
            artifacts=dict(job_log=dict(bytes=len(log), sha256=hashlib.sha256(log).hexdigest())), elapsed_s=1., peak_rss_mb=10., peak_vram_mb=1. if gpu else 0.))
    (root / "admission-command.json").write_text(json.dumps(dict(command=command)))
    (root / "job.json").write_text(json.dumps(job)); (root / "job.log").write_bytes(log)
    return request, marker, job


@pytest.mark.parametrize("gpu", [True, False])
def test_actual_reservation_is_distinct_from_intent_and_complete_log_is_required(tmp_path, gpu):
    root = tmp_path / "captured"; request, marker, job = capture_fixture(root, gpu=gpu)
    result = V["captured_job"](root, request, "job-synthetic", root / "job.json", root / "job.log", gpu=gpu, expected_marker=marker)
    assert result["actual_reservation"] == job["reservation"]
    assert result["submitted_reservation"] is None if gpu else result["submitted_reservation"] == job["reservation"]
    assert "unknown model" in result["planner_error"]
    (root / "job.log").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="complete captured log"):
        V["captured_job"](root, request, "job-synthetic", root / "job.json", root / "job.log", gpu=gpu, expected_marker=marker)
    with pytest.raises(ValueError, match="complete local job"):
        V["captured_job"](root, request, "job-synthetic", None, None, gpu=gpu, expected_marker=marker)


def test_captured_reservation_and_payload_custody_tampering_are_rejected(tmp_path):
    root = tmp_path / "captured"; request, marker, job = capture_fixture(root)
    original = copy.deepcopy(job)
    for change in ("reservation", "payload"):
        job = copy.deepcopy(original)
        if change == "reservation":
            job["reservation"]["vram_mb"] = 2048
        else:
            job["payload_custody"]["executed"] = dict(sha256="2"*64, size_bytes=123)
        (root / "job.json").write_text(json.dumps(job))
        with pytest.raises(ValueError, match="reservation differs|payload custody differs"):
            V["captured_job"](root, request, "job-synthetic", root / "job.json", root / "job.log", gpu=True, expected_marker=marker)


def test_local_source_wheels_and_config_must_match_exact_payload_seal(tmp_path):
    payload = tmp_path / "payload"; payload.mkdir()
    for name in ("worker.py", "runtime.whl"):
        (payload / name).write_text("synthetic metadata/code bytes")
    request = dict(config=dict(synthetic=True), source_hashes={"worker.py": V["sha"](payload / "worker.py")}, runtime_wheels={"runtime.whl": V["sha"](payload / "runtime.whl")})
    request["request_sha256"] = V["identity"](request)
    for name, value in (("saturn_run_config.json", request["config"]), ("saturn_debug_request.json", request)):
        (payload / name).write_text(json.dumps(value))
    (payload / "payload-seal.json").write_text(json.dumps(dict(files=request["source_hashes"], wheels=request["runtime_wheels"], worker="worker.py",
        worker_sha256=request["source_hashes"]["worker.py"], config_bytes_sha256=V["sha"](payload / "saturn_run_config.json"))))
    result = V["payload_account"](tmp_path, request, request["source_hashes"], "worker.py")
    assert result["runtime_wheels"] == request["runtime_wheels"]
    (payload / "runtime.whl").write_text("tampered")
    with pytest.raises(ValueError, match="source/wheel payload differs"):
        V["payload_account"](tmp_path, request, request["source_hashes"], "worker.py")


def test_seven_event_pretarget_response_and_four_operand_receipts(tmp_path):
    order = [(lane, world) for lane in ("functional", "native") for world in V["WORLDS"]]
    reference = dict(shape=[1, 512, 2560], stride=[1310720, 2560, 1], content_sha256="0"*64)
    qkv = [dict(shape=[1, heads, 512, 128], stride=[heads*65536, 128, heads*128, 1], content_sha256="1"*64,
        logical_dtype="torch.bfloat16", device_at_capture="cuda:0") for heads in (32, 8, 8)]
    live = [{**{key: row[key] for key in ("shape", "stride", "content_sha256")}, "dtype": "torch.bfloat16", "device": "cuda:0"} for row in qkv]
    calls = [dict(lane=lane, arm=world, QKV=copy.deepcopy(qkv), operands=copy.deepcopy(live), attention_mask=None,
        is_causal=True, enable_gqa=True, dropout_p=0., scale=128**-.5) for lane, world in order]
    entries = [dict(world=world, input_reference=reference, input=reference) for world in V["WORLDS"]]
    counts = dict(functional_blocks=2, native_block_calls=2, native_SDPA=2, functional_SDPA=2, component_hydrations=1)
    kinds = ["functional_world_completed"]*2 + ["cohort_globally_sealed", "native_targets_released"] + ["native_component_authorized", "native_component_completed"]*2
    journal = [dict(kind=kind, number=index, native_module_entries=entries) for index, (kind, entries) in enumerate(zip(kinds, [0, 0, 0, 0, 0, 15, 15, 30], strict=True))]
    for index in (2, 3):
        journal[index]["seal_sha256"] = "2"*64
    journal[2]["native_targets_at_seal"] = 0
    (tmp_path / "source-timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
    protocol = dict(physical_plan=counts, response=dict(left_world="cheap-input", right_world="full-input", orientation="cheap-input-minus-full-input", ports=list(V["PORTS"])))
    response = dict(protocol["response"], baseline_world="full-input", fields={name: dict(elements=V["POSITIONS"][name], dtype="float64") for name in V["PORTS"]})
    native = dict(context={}, worlds={world: dict(actual_input_descriptor=reference, ports=dict(zip(("q", "k", "v"), qkv, strict=True))) for world in V["WORLDS"]})
    values = {"source-forecast-responses.json": response, "source-sdpa.json": dict(calls=calls), "source-native.json": native,
        "source-forecast.json": native, "source-source-frontier.json": dict(context={}, worlds={world: dict(input_reference=reference) for world in V["WORLDS"]}),
        "source-source.json": dict(hydration=dict(component_hydrations=1, cuda_placements=1)),
        "source-native-contract.json": dict(expected=counts, observed=counts, block_entries=entries,
            projection_provenance="actual passive o_proj output hook", R_provenance="actual post_attention_layernorm input hook")}
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value))
    summary = dict(forecast_seal="2"*64, transaction=dict(event_head="3"*64), counts=counts)
    tx = dict(verified=dict(verified=True, events=7, phase="closed", frontiers=["inherited-worlds"], forecasts=["cohort"], seal_sha256="2"*64, event_head="3"*64),
        chronology=journal, inherited_arrays=7, native_targets_at_seal=0)
    audit = dict(transaction=tx, response_commitments=response, census=dict(observed=counts, declared=counts, actual_SDPA_calls=calls, native_entries=entries))
    raw = dict(files={"transaction/seal.json": dict(sha256="2"*64), **{f"transaction/events/{index}.json": {} for index in range(7)}})
    assert V["transaction_census"](tmp_path, audit, protocol, summary, raw)["transaction"]["verified"]["events"] == 7
    audit["transaction"]["native_targets_at_seal"] = 1
    with pytest.raises(ValueError, match="transaction/seal"):
        V["transaction_census"](tmp_path, audit, protocol, summary, raw)
    audit["transaction"]["native_targets_at_seal"] = 0
    changed = copy.deepcopy(calls); changed[0]["operands"][0]["stride"] = [1, 2, 3, 4]
    (tmp_path / "source-sdpa.json").write_text(json.dumps(dict(calls=changed))); audit["census"]["actual_SDPA_calls"] = changed
    with pytest.raises(ValueError, match="QKV hash/layout"):
        V["transaction_census"](tmp_path, audit, protocol, summary, raw)


def test_compact_member_checks_and_actual_twelve_installed_queries(tmp_path):
    pack = tmp_path / "collection/extracted/binding-minimal-result"
    shutil.copytree(HERE / "tiny-catalog", pack)
    V["save"](pack / "files.json", {path.relative_to(pack).as_posix(): V["sha"](path) for path in pack.rglob("*") if path.is_file()})
    collection = pack.parents[1]
    V["save"](collection / "collection-receipt.json", dict(files_manifest_sha256=V["sha"](pack / "files.json")))
    assert not V["BOOK"]["compact_account"](collection, pack)["retained_array_members"]
    output = tmp_path / "queries"; output.mkdir()
    calls = V["cli_accounts"](pack, output)
    assert len(calls) == 12 and sum(row["matches"] for row in calls) == 4
    assert {row["contrast"] for row in calls} == {"exact", "changed-source", "changed-clock", "changed-profile"}
    assert all(row["matches"] == (1 if row["contrast"] == "exact" else 0) for row in calls)
    target = next((pack / "catalog/equations").glob("*.json")); target.write_text("{}")
    with pytest.raises(ValueError, match="compact member differs"):
        V["BOOK"]["compact_account"](collection, pack)
    result = subprocess.run([sys.executable, "-I", "-B", str(HERE / "mdb-qwen-error-consumer-20261106/verify_saved_reports.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--expected-native-manifest" in result.stdout

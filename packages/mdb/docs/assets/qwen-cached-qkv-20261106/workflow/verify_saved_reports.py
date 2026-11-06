"""Thin compact-only cached-QKV recount and captured-custody/CLI verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "mdb-qwen-row-sign-20261106/verify_saved_reports.py"
PRIOR_SHA256 = "3ff3a2417969475d6602b76c8e5d83677a52be895df4e7582e24e38821f1d4e6"
READER_SHA256 = "b473d178fd9825009c0c039dc23ab0f2208309f2d34ccc2d784ee5b5710385eb"
PRODUCER_SHA256 = "d2c9e472ac06c667fea0726ac62f629e63ad776173a2e323821bf66f71b0d8f5"
NATIVE_REQUEST = "d83ec9216f5d0b00b74175285c8608a3d0e812c79126f47ef55a6caaccfb405d"
AUDIT_REQUEST = "debd122ff812da52b15a91d3b856301060c7ddb752b4f4a026f8e8b35bcbe351"
COMPACT_MANIFEST = "c662757ab5f261b91f72006bd3de57291e1ff70e208c38e9e30631cb6c1a3ca7"
COLLECTOR_SHA256 = "fc63b5adfbb96d31529969efafab426ca05e9b7e4ddc461d81811cc96395c22d"
PORTS, QKV = ("head", "R", "M", "Hnext"), ("q", "k", "v")
CHANNELS = ("cached-qkv-minus-full-functional", "full-functional-minus-native", "cached-qkv-minus-native")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(PRIOR) != PRIOR_SHA256:
    raise ValueError("frozen compact/custody/CLI helper differs")
H = runpy.run_path(str(PRIOR))
require, read, save, aggregate, identity = (H[name] for name in ("require", "read", "save", "aggregate", "identity"))


def metrics(rows):
    result = aggregate(rows)
    result["signed_zero"] = {key: sum(row["signed_zero"][key] for row in rows) for key in ("shared", "defects")}
    result["logical_bit_changed_elements"] = result["changed_values"] + result["signed_zero"]["defects"]
    return result


def windows(rows):
    for row in rows:
        require(row["whole"]["sample_count"] == sum(window["comparison"]["sample_count"] for window in row["windows"])
            and row["whole"]["changed_values"] == sum(window["comparison"]["changed_values"] for window in row["windows"]), "whole/protocol-window report census differs")
    return dict(whole=metrics([row["whole"] for row in rows]), windows=metrics([window["comparison"] for row in rows for window in row["windows"]]))


def rollup(audit, protocol):
    require(set(audit["arms"]) == {arm["id"] for arm in protocol["arms"]}, "complete protocol arm reports differ")
    per_arm, channels, positioned = {}, {}, {}
    for channel in CHANNELS:
        channels[channel] = {kind: windows([row["channels"][channel][name][kind] for row in audit["arms"].values() for name in PORTS]) for kind in ("endpoint", "response_error")}
        positioned[channel] = windows([row["positioned_QKV"][channel][name] for row in audit["arms"].values() for name in QKV])
    for arm in protocol["arms"]:
        row = audit["arms"][arm["id"]]
        require(set(row["channels"]) == set(row["positioned_QKV"]) == set(CHANNELS) and
            row["coverage"] == dict(primary_endpoint_fields=12, primary_response_fields=12, overlapping_endpoint_windows=12*len(arm["windows"]),
                overlapping_response_windows=12*len(arm["windows"]), QKV_channel_fields=9, QKV_source_sign_fields=3), "declared field/window coverage differs")
        for channel in CHANNELS:
            require(set(row["channels"][channel]) == set(PORTS) and set(row["positioned_QKV"][channel]) == set(QKV)
                and all([window["bounds"] for window in field["endpoint"]["windows"]] == arm["windows"] for field in row["channels"][channel].values()), "complete ports/protocol window coordinates differ")
        per_arm[arm["id"]] = dict(support=row["support"], first_discrepancies=row["first_discrepancies"],
            channels={channel: {kind: windows([field[kind] for field in row["channels"][channel].values()]) for kind in ("endpoint", "response_error")} for channel in CHANNELS},
            per_port=row["channels"], positioned_QKV=row["positioned_QKV"],
            cached_source_sign_relation=row["cached_source_sign_relation"], cached_source_sign_summary=windows(list(row["cached_source_sign_relation"].values())),
            native_edit_effects={name: row["channels"]["cached-qkv-minus-native"][name]["right_effect"] for name in PORTS})
    layout = audit["layout_restoration"]["arrays"]
    return dict(channels=channels, positioned_QKV=positioned, per_arm=per_arm,
        positive_baseline_primary=metrics([audit["positive_baseline"][name] for name in PORTS]), positive_baseline_QKV=metrics([audit["positive_baseline"][name] for name in QKV]),
        layout=dict(arrays=len(layout), changed_staging_to_restored_stride=[name for name, row in layout.items() if row["loaded_logical"]["stride"] != row["restored"]["stride"]], records=layout),
        counting_scope="Per channel8 primary whole endpoint+8response fields. Windows partition each arm independently; their bounds differ and counts remain separate. QKV, sign-zero, source sign, baseline, effects and layouts overlap. Logical bit-changed count derives numeric changes plus shared-zero sign defects; it is not an NPY byte offset or independent experiment count.")


def collection_binding(receipt, native_job, audit_job, audit_request):
    require(receipt["schema"] == "mdb-qwen-cached-qkv-compact-retrieval-v1" and receipt["collector_sha256"] == COLLECTOR_SHA256
        and receipt["generic_transport_sha256"] == "007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d"
        and receipt["reused_metadata_scope_sha256"] == "b621d8e12dc06476abf356643f22693a48a5a52bd43c7684d66e788126475015"
        and receipt["job_id"] == audit_job and receipt["native_job_id"] == native_job and receipt["request_sha256"] == audit_request["request_sha256"]
        and receipt["raw_tensor_arrays_collected"] is False and receipt["model_calls"] == 0, "exact cached collector/job custody differs")


def verify(args):
    pack = args.collection.resolve() / "extracted/binding-minimal-result"
    compact = H["H"]["compact_account"](args.collection, pack)
    require(compact["manifest_sha256"] == args.expected_compact_manifest and not compact["retained_array_members"], "expected compact identity/no-array boundary differs")
    native_request, audit_request = (read(run / "request.json") for run in (args.native_run, args.audit_run))
    require(identity(native_request) == native_request["request_sha256"] == args.expected_native_request and
        identity(audit_request) == audit_request["request_sha256"] == args.expected_audit_request, "expected standard request identities differ")
    audit, summary, source, plan, source_summary, raw = (read(pack / name) for name in ("audit.json", "summary.json", "source-source.json", "source-plan.json", "source-summary.json", "source-files.json"))
    native_job, audit_job = (read(run / "submission.json")["job_id"] for run in (args.native_run, args.audit_run))
    require(read(pack / "source-request.json") == native_request and read(pack / "request.json") == audit_request and
        audit["job_id"] == summary["source_job"] == source["job_id"] == source_summary["job_id"] == native_job and summary["job_id"] == audit_job and
        audit["request_sha256"] == source["request_sha256"] == source_summary["request_sha256"] == native_request["request_sha256"] and summary["request_sha256"] == audit_request["request_sha256"], "compact/source/audit identity joins differ")
    require(audit["audit_sha256"] == audit_request["source_hashes"]["audit_qwen_cached_qkv.py"] == READER_SHA256 and
        audit["worker_sha256"] == native_request["source_hashes"]["qwen_cached_qkv_campaign.py"] == PRODUCER_SHA256 and source["source_hashes"] == native_request["source_hashes"], "correct executed reader/producer source differs")
    output = str(Path(native_request["config"]["output_scratch"]) / native_job / native_request["config"]["output_name"])
    require(plan == dict(native_request["config"], output=output, request_source_hashes=native_request["source_hashes"], request_sha256=native_request["request_sha256"], job_id=native_job)
        and read(pack / "source-declared-config.json") == native_request["config"] and audit_request["config"]["native_root"] == output and
        audit_request["config"]["native_request_sha256"] == native_request["request_sha256"] and audit_request["config"]["native_manifest_sha256"] == audit["native_manifest_sha256"] == sha(pack / "source-files.json") and
        source_summary["counts"] == audit["census"]["observed"], "effective config/source raw identity or census differs")
    receipt = compact["collection"]
    collection_binding(receipt, native_job, audit_job, audit_request)
    require(raw["job_id"] == native_job and raw["request_sha256"] == native_request["request_sha256"] and raw["expanded_bytes"] == sum(row["bytes"] for row in raw["files"].values()), "raw custody census differs")
    protocol = read(pack / "source-metadata/selection_protocol.json")
    records = [read(path) for path in (pack / "catalog/equations").glob("*.json")]
    require({row["tested_context"]["arm"] for row in records} == {arm["id"] for arm in protocol["arms"]} and all(row["tested_context"]["source_identity"] == source["source_hashes"] and
        row["tested_context"]["model_identity"] == plan["qwen_parameter_manifest_sha256"] and row["tested_context"]["input_identity"] == protocol["source_arrays"]["h"]["content_sha256"] for row in records), "catalog source/model/input/occurrence bindings differ")
    args.output.mkdir(parents=True, exist_ok=False); out = args.output.resolve()
    marker = dict(job_id=native_job, request_sha256=native_request["request_sha256"], output_root=output, files_manifest_sha256=sha(pack / "source-files.json"),
        summary_sha256=sha(pack / "source-summary.json"), file_count=len(raw["files"]), expanded_bytes=raw["expanded_bytes"])
    reader_marker = dict(job_id=audit_job, request_sha256=audit_request["request_sha256"], output_root=str(Path(audit_request["config"]["output_scratch"]) / audit_job / "binding-minimal-result"),
        files_manifest_sha256=compact["manifest_sha256"], summary_sha256=sha(pack / "summary.json"))
    jobs = dict(native=H["captured_job"](args.native_run, native_request, native_job, args.native_job, args.native_log, gpu=True, expected_marker=marker),
        audit=H["captured_job"](args.audit_run, audit_request, audit_job, args.audit_job, args.audit_log, gpu=False, expected_marker=reader_marker))
    result = dict(schema="mdb-qwen-cached-qkv-compact-verification-v1", script_sha256=sha(__file__), helper_sha256=PRIOR_SHA256, audit_sha256=sha(pack / "audit.json"),
        source_pins=dict(reader_sha256=READER_SHA256, producer_sha256=PRODUCER_SHA256, reader_binding="Actual audit request/source pin, not an absent collector-receipt field"),
        compact=compact, jobs=jobs, raw=dict(members=len(raw["files"]), bytes=raw["expanded_bytes"]), totals=rollup(audit, protocol),
        source=audit["source"], inherited=audit["parent"], census=audit["census"], transaction=audit["transaction"], actual_installed_cli=H["cli_accounts"](pack, out),
        scope="Compact report JSON and explicit complete local job/log captures only; no raw arrays/weights, model/tokenizer/projection, scheduler, staging or submission. Nonzero observations retained without a scientific gate.")
    save(out / "verification.json", result)
    print(json.dumps(dict(receipt_sha256=sha(out / "verification.json"), channels=result["totals"]["channels"], QKV=result["totals"]["positioned_QKV"], actual_CLI_calls=len(result["actual_installed_cli"])), sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("collection", "compact-collection-01"), ("native-run", "run-01"), ("audit-run", "audit-01"), ("output", "saved-report-verification-v1")):
        parser.add_argument("--"+name, type=Path, default=ROOT / default)
    for name in ("native-job", "native-log", "audit-job", "audit-log"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--expected-native-request", default=NATIVE_REQUEST)
    parser.add_argument("--expected-audit-request", default=AUDIT_REQUEST)
    parser.add_argument("--expected-compact-manifest", default=COMPACT_MANIFEST)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()

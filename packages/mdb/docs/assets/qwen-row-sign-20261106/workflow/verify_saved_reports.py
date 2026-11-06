"""Verify collected row-sign scalar reports, custody and actual MDB lookups.

No raw array/model/tokenizer/weight reads, scheduler access or scientific gates.
Job/log custody is checked only from explicit locally captured files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "mdb-qwen-sign-shortcut-20261106/verify_saved_reports.py"
PRIOR_SHA256 = "3c982cef7cf3a873a7f5c89f7ccea7c1db2eacb6764afd16b44d9bf59b2fe653"
READER_SHA256 = "d995c3c5ca3c194a3ce87bfca58e9352d251246d798e21c6e04a7938bd9b5838"
PRODUCER_SHA256 = "fbd8eceaf1a9ef8917ecdc39a98ff960940aab9da8ecb4697c25c1a33d545f47"
NATIVE_REQUEST = "fb202763231fad27712f2b525aa1ea0d041496c13079d6b4a4a7425d36e18a77"
NATIVE_MANIFEST = "d36f790376a85ad995cf20f01cff89fb6e04d282fe1290f677ba348ec98f4c8f"
ARMS = ("all-negative", "prefix-negative")
CHANNELS = ("shortcut-minus-full-functional", "full-functional-minus-native", "shortcut-minus-native")
PORTS = ("head", "R", "M", "Hnext")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


require(sha(PRIOR) == PRIOR_SHA256, "frozen compact bookkeeping helper differs")
H = runpy.run_path(str(PRIOR))
save, aggregate, identity = (H[name] for name in ("save", "aggregate", "identity"))


def regions(wrappers):
    """Whole and partitioning windows are separate, explicitly counted sets."""
    for row in wrappers:
        require(set(row["windows"]) == {"prefix", "suffix"}, "two fixed report windows required")
        require(row["whole"]["sample_count"] == sum(value["sample_count"] for value in row["windows"].values())
            and row["whole"]["changed_values"] == sum(value["changed_values"] for value in row["windows"].values()), "whole/window report census differs")
    return dict(whole=aggregate([row["whole"] for row in wrappers]),
        windows={window: aggregate([row["windows"][window] for row in wrappers]) for window in ("prefix", "suffix")})


def rollup(audit):
    require(set(audit["arms"]) == set(ARMS), "two complete arm accounts required")
    per_arm, channels, by_port = {}, {}, {}
    for channel in CHANNELS:
        banks = {arm: audit["arms"][arm]["channels"][channel]["fields"] for arm in ARMS}
        require(all(set(bank) == set(PORTS) for bank in banks.values()), "complete canonical head/R/M/H fields required")
        channels[channel] = {kind: regions([bank[name][kind] for bank in banks.values() for name in PORTS]) for kind in ("endpoint", "response_error")}
        by_port[channel] = {kind: {name: regions([bank[name][kind] for bank in banks.values()]) for name in PORTS} for kind in ("endpoint", "response_error")}
    for arm in ARMS:
        row = audit["arms"][arm]
        require(set(row["channels"]) == set(CHANNELS) and row["coverage"] == dict(primary_endpoint_fields=12, primary_response_fields=12,
            overlapping_endpoint_windows=24, overlapping_response_windows=24), "declared primary/window coverage differs")
        comparison = row["channels"]["shortcut-minus-native"]["fields"]
        native_effects = {name: comparison[name]["right_effect"] for name in PORTS}
        premises = [item for lane in row["premises"].values() for item in lane.values()]
        require(set(row["premises"]) == {"functional", "native"} and all(set(bank) == {"R", "gate", "up"} for bank in row["premises"].values()), "R/G/U premise scope differs")
        per_arm[arm] = dict(support=row["support"], native_edit_effects=native_effects,
            native_effect_summary=regions(list(native_effects.values())),
            channels={channel: {kind: regions([field[kind] for field in row["channels"][channel]["fields"].values()]) for kind in ("endpoint", "response_error")} for channel in CHANNELS},
            first_differences={channel: {kind: next((name for name in PORTS if not row["channels"][channel]["fields"][name][kind]["whole"]["bitwise_exact"]), None)
                for kind in ("endpoint", "response_error")} for channel in CHANNELS},
            premises=dict(metrics=regions(premises), signed_zero={window: dict(shared=sum(item["signed_zero"][window]["shared"] for item in premises),
                defects=sum(item["signed_zero"][window]["defects"] for item in premises)) for window in ("whole", "prefix", "suffix")}))
    return dict(channels=channels, by_port=by_port, per_arm=per_arm, positive_source_baseline=regions(list(audit["source_baseline_errors"].values())),
        conditional_positive_suffix_response=audit["conditional_positive_suffix_response"],
        counting_scope="Per channel:8 primary whole endpoint and8 response fields, plus16 windows each. Three channels remain separate. Native effects, source replay and12 premise fields are distinct overlapping accounts, not additional independent replications or gain measures.")


def captured_job(run, request, job_id, job_path, log_path, *, gpu, expected_marker):
    if job_path is None and log_path is None:
        return dict(status="capture-not-provided", job_id=job_id, command_reservation_custody_log="unestablished; no scheduler fetched")
    require(job_path is not None and log_path is not None, "provide both complete job JSON and log")
    job, log = read(job_path), Path(log_path).read_bytes()
    result = job.get("result") or {}
    require(job["job_id"] == job_id and job["state"] == "succeeded" and result["returncode"] == 0, "exact completed successful captured job required")
    for envelope in (job["config"], job["submitted_request"]["config"]):
        require(all(envelope.get(key) == value for key, value in request.items()), "captured actual request/config differs")
    command = read(run / "admission-command.json")["command"]
    require(result["diagnostics"]["command"] == command and job["config"]["retry_on_kill"] is False, "captured command/retry policy differs")
    reservation = job["submitted_request"]["reservation"]
    require(all(reservation[key] == request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads"))
        and bool(job["submitted_request"]["needs"].get("cuda", False)) == gpu, "captured reservation/device differs")
    custody = [job["payload_custody"][key] for key in ("declared", "sealed", "executed")] + [result["executed_payload"]]
    require(len({(row["sha256"], row["size_bytes"]) for row in custody}) == 1, "captured declared/sealed/executed/result payload differs")
    expected = result["artifacts"]["job_log"]
    require(len(log) == expected["bytes"] and hashlib.sha256(log).hexdigest() == expected["sha256"], "complete captured log bytes differ")
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    require(markers == [expected_marker], "captured exact final directory marker differs")
    return dict(status="authenticated-local-capture", job_id=job_id, command=command, reservation=reservation,
        job_json_sha256=sha(job_path), complete_log_sha256=sha(log_path), executed_payload=result["executed_payload"],
        **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")})


def cli_accounts(pack, output):
    records = [read(path) for path in sorted((pack / "catalog/equations").glob("*.json"))]
    require(len(records) == 2, "two source-bound conditional catalog observations required")
    calls = []
    for record in records:
        context = record["tested_context"]
        predicates = {row["field"]: row for row in record["applicability_conditions"]}
        require(all(predicates[key] == dict(field=key, op="eq", value=context[key]) for key in ("source_identity", "execution")), "actual source/profile conditions differ")
        for kind, query, expected in (("exact", context, 1), ("changed-source", {**context, "source_identity": H["changed"](context["source_identity"])}, 0),
            ("changed-profile", {**context, "execution": H["changed_profile"](context["execution"])}, 0)):
            stem = record["alias"] + "." + kind
            path = output / (stem + ".context.json")
            save(path, query)
            for api, command in {
                "equations": [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find", "--catalog", str(pack / "catalog/equations"), "--alias", record["alias"], "--applicable-to", str(path)],
                "atlas": [sys.executable, "-I", "-B", "-m", "mdb.computation_atlas", "--atlas", str(pack / "atlas"), "find", "--law-id", record["law_id"], "--applicable-to", str(path)],
            }.items():
                completed = subprocess.run(command, cwd=output, capture_output=True, text=True, check=True)
                matches = json.loads(completed.stdout)
                require(len(matches) == expected and all((row["fingerprint"] if api == "equations" else row["source_record"]["fingerprint"]) == record["fingerprint"] for row in matches), "actual installed applicability result differs")
                target = output / (stem + "." + api + ".cli.json")
                target.write_text(completed.stdout)
                calls.append(dict(api=api, alias=record["alias"], contrast=kind, matches=len(matches), command=command, context_sha256=sha(path), result_sha256=sha(target)))
    return calls


def catalog_bindings(pack, source, plan):
    frontier = read(pack / "source-source-frontier.json")
    records = [read(path) for path in (pack / "catalog/equations").glob("*.json")]
    require(len(records) == 2 and {row["tested_context"]["arm"] for row in records} == set(ARMS), "catalog occurrence arms differ")
    for row in records:
        context = row["tested_context"]
        require(context["source_identity"] == source["source_hashes"] and context["model_identity"] == plan["qwen_parameter_manifest_sha256"]
            and context["input_identity"] == frontier["source_arrays"]["h"]["content_sha256"] and context["horizon"] == plan["horizon"]
            and context["fixed_split"] == plan["fixed_split"], "catalog does not describe authenticated source/model/input/horizon")


def verify(args):
    pack = args.collection.resolve() / "extracted/binding-minimal-result"
    compact = H["compact_account"](args.collection, pack)
    require(not compact["retained_array_members"], "compact unexpectedly contains raw arrays/weights")
    requests = {name: read(run / "request.json") for name, run in (("native", args.native_run), ("audit", args.audit_run))}
    require(all(identity(row) == row["request_sha256"] for row in requests.values()), "expected local request identity differs")
    require(requests["native"]["request_sha256"] == args.expected_native_request and (args.expected_audit_request is None or requests["audit"]["request_sha256"] == args.expected_audit_request), "expected request pin differs")
    audit, summary, source_request, saved_request, plan, source_summary, source, raw = (read(pack / name) for name in
        ("audit.json", "summary.json", "source-request.json", "request.json", "source-plan.json", "source-summary.json", "source-source.json", "source-files.json"))
    native_job, audit_job = (read(run / "submission.json")["job_id"] for run in (args.native_run, args.audit_run))
    require(source_request == requests["native"] and saved_request == requests["audit"] and audit["job_id"] == summary["source_job"] == native_job
        and summary["job_id"] == audit_job and summary["request_sha256"] == saved_request["request_sha256"] and audit["request_sha256"] == source_request["request_sha256"], "actual compact/source/audit identity joins differ")
    collection = compact["collection"]
    require(collection["job_id"] == audit_job and collection["request_sha256"] == saved_request["request_sha256"] and collection["native_job_id"] == native_job
        and collection["reader_sha256"] == READER_SHA256, "authenticated collection source/audit identity differs")
    require(audit["audit_sha256"] == saved_request["source_hashes"]["audit_qwen_row_sign.py"] == READER_SHA256 and
        audit["worker_sha256"] == source_request["source_hashes"]["qwen_row_sign_campaign.py"] == PRODUCER_SHA256, "expected reader/producer source differs")
    output = str(Path(source_request["config"]["output_scratch"]) / native_job / source_request["config"]["output_name"])
    require(plan == dict(source_request["config"], output=output, request_source_hashes=source_request["source_hashes"], request_sha256=source_request["request_sha256"], job_id=native_job)
        and read(pack / "source-declared-config.json") == source_request["config"], "effective/declared producer config differs")
    require(source["source_hashes"] == source_request["source_hashes"] and source["job_id"] == source_summary["job_id"] == native_job
        and source["request_sha256"] == source_summary["request_sha256"] == source_request["request_sha256"] and source_summary["counts"] == audit["census"]["observed"], "source/core/report counts or identity differs")
    require(saved_request["config"]["native_root"] == output and saved_request["config"]["native_request_sha256"] == source_request["request_sha256"]
        and saved_request["config"]["native_manifest_sha256"] == audit["native_manifest_sha256"] == sha(pack / "source-files.json") == args.expected_native_manifest,
        "audit source root/request/manifest differs")
    require(raw["job_id"] == native_job and raw["request_sha256"] == source_request["request_sha256"] and raw["expanded_bytes"] == sum(row["bytes"] for row in raw["files"].values()), "raw wrapper identity/byte census differs")
    catalog_bindings(pack, source, plan)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    native_marker = dict(job_id=native_job, request_sha256=source_request["request_sha256"], output_root=output, files_manifest_sha256=sha(pack / "source-files.json"),
        summary_sha256=sha(pack / "source-summary.json"), file_count=len(raw["files"]), expanded_bytes=raw["expanded_bytes"])
    audit_marker = dict(job_id=audit_job, request_sha256=saved_request["request_sha256"], output_root=str(Path(saved_request["config"]["output_scratch"]) / audit_job / "binding-minimal-result"), files_manifest_sha256=compact["manifest_sha256"], summary_sha256=sha(pack / "summary.json"))
    jobs = dict(native=captured_job(args.native_run, source_request, native_job, args.native_job, args.native_log, gpu=True, expected_marker=native_marker),
        audit=captured_job(args.audit_run, saved_request, audit_job, args.audit_job, args.audit_log, gpu=False, expected_marker=audit_marker))
    totals, cli = rollup(audit), cli_accounts(pack, out)
    result = dict(schema="mdb-qwen-row-sign-saved-report-verification-v1", verifier_sha256=sha(__file__), helper_sha256=PRIOR_SHA256,
        audit_sha256=sha(pack / "audit.json"), source=dict(reader_sha256=READER_SHA256, producer_sha256=PRODUCER_SHA256,
            parameter_restoration=audit["source"]["parameters"], parameter_recipe_io=audit["source"]["parameter_recipe_io"], runtime=audit["source"]["runtime"], inherited=audit["parent"]),
        requests={key: row["request_sha256"] for key, row in requests.items()}, compact=compact, jobs=jobs, raw=dict(members=len(raw["files"]), bytes=raw["expanded_bytes"]),
        totals=totals, census=audit["census"], transaction=audit["transaction"], actual_installed_cli=cli,
        status="Saved compact custody/recount/CLI verification completed; no numerical success threshold.",
        scope="Report JSON only; no array, weights, tokenizer, learned operation, scheduler or model access. Failed audit-01 remains unchanged. Job/log facts are established only when explicit complete local captures are supplied.")
    save(out / "verification.json", result)
    print(json.dumps(dict(receipt_sha256=sha(out / "verification.json"), channels=totals["channels"], actual_cli_calls=len(cli), job_capture_status={key: value["status"] for key, value in jobs.items()}), sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("collection", "compact-collection-01"), ("native-run", "run-02"), ("audit-run", "audit-02"), ("output", "saved-report-verification-v1")):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    for name in ("native-job", "native-log", "audit-job", "audit-log"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--expected-native-request", default=NATIVE_REQUEST)
    parser.add_argument("--expected-native-manifest", default=NATIVE_MANIFEST)
    parser.add_argument("--expected-audit-request")
    verify(parser.parse_args())


if __name__ == "__main__":
    main()

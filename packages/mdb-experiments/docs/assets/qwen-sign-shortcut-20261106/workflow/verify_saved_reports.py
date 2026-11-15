"""ROOT-only verification of authenticated saved shortcut reports and MDB CLIs.

Run only after the real compact is collected. This reads JSON and job/log
custody, never model tensors, a tokenizer, weights or learned operations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
READER_SHA256 = "d33613bbb1d71ee3b2007fb007e5266acaf141994091264bed50d27342237e2c"
NATIVE_REQUEST = "1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a"
CHANNELS = ("shortcut-minus-full-functional-negative", "full-functional-minus-native-negative", "shortcut-minus-native-negative")
HEADS = ("product", "M", "Hnext")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def aggregate(rows):
    positions = sum(row["sample_count"] for row in rows)
    pooled = lambda key: None if not positions else math.sqrt(sum(row[key] ** 2 * row["sample_count"] for row in rows) / positions)
    return dict(fields=len(rows), positions=positions,
        bit_exact_fields=sum(row["bitwise_exact"] is True for row in rows),
        changed_values=sum(row["changed_values"] for row in rows),
        maxabs=max((row["maxabs"] for row in rows), default=0.), max_rms=max((row["rms"] for row in rows), default=0.),
        pooled_error_rms=pooled("rms"), pooled_left_rms=pooled("predicted_rms"), pooled_right_rms=pooled("native_rms"),
        signed_mean=None if not positions else sum(row["signed_mean"] * row["sample_count"] for row in rows) / positions)


def rollup(audit):
    """Explicit disjoint account families; no recursive metric/count gathering."""
    endpoints = {channel: [] for channel in CHANNELS}
    responses = {channel: [] for channel in CHANNELS}
    head_endpoints = {channel: {name: [] for name in HEADS} for channel in CHANNELS}
    head_responses = {channel: {name: [] for name in HEADS} for channel in CHANNELS}
    prefix, prefix_regions, ports, port_regions, premises = [], [], [], [], []
    per_case = {}
    for case, row in audit["cases"].items():
        require(set(row["tail"]["comparisons"]) == set(CHANNELS), "three comparison channels differ")
        for channel in CHANNELS:
            fields = row["tail"]["comparisons"][channel]["fields"]
            require(set(fields) == set(HEADS), "complete P/M/H report fields differ")
            endpoints[channel].extend(fields[name]["endpoint"] for name in HEADS)
            responses[channel].extend(fields[name]["signed_edit_response_error"] for name in HEADS)
            for name in HEADS:
                head_endpoints[channel][name].append(fields[name]["endpoint"])
                head_responses[channel][name].append(fields[name]["signed_edit_response_error"])
        families = ((row["prefix_source_errors"].values(), prefix, prefix_regions),
                    ((field for sign in row["aligned_ports"].values() for field in sign.values()), ports, port_regions))
        local_prefix, local_ports = [], []
        for wrappers, whole, regional in families:
            for field in wrappers:
                whole.append(field["full"])
                (local_prefix if whole is prefix else local_ports).append(field["full"])
                regional.extend(field[key]["comparison"] for key in ("attended_query_rows", "padded_query_rows") if field[key]["comparison"] is not None)
        require(row["coverage"]["primary_prefix_fields"] == len(local_prefix)
            and row["coverage"]["aligned_port_fields"] == len(local_ports)
            and row["coverage"]["primary_endpoint_error_fields"] == row["coverage"]["primary_response_error_fields"] == 9,
            "declared per-case account coverage differs")
        odd = row["oddness"]
        require(set(odd) == {"functional", "native"} and all(set(bank) == {"R", "gate", "up"} for bank in odd.values()), "finite premise diagnostic scope differs")
        premises.extend(value for bank in odd.values() for value in bank.values())
        per_case[case] = dict(first_prefix_bit_discrepancy=row["first_prefix_bit_discrepancy"],
            tail_first_discrepancies={channel: {key: value for key, value in report.items() if key != "fields"}
                                      for channel, report in row["tail"]["comparisons"].items()},
            oddness=odd, finite_addition=row["finite_addition"], actual_mask=row["source"],
            actual_native=row["native"], shortcut_execution_binding=row["shortcut_execution_binding"])
    return dict(endpoint_errors={key: aggregate(rows) for key, rows in endpoints.items()},
        signed_edit_response_errors={key: aggregate(rows) for key, rows in responses.items()},
        by_head=dict(endpoint_errors={channel: {name: aggregate(rows) for name, rows in heads.items()} for channel, heads in head_endpoints.items()},
                     signed_edit_response_errors={channel: {name: aggregate(rows) for name, rows in heads.items()} for channel, heads in head_responses.items()}),
        positive_prefix_whole=aggregate(prefix), positive_prefix_regions=aggregate(prefix_regions),
        aligned_ports_whole=aggregate(ports), aligned_ports_regions=aggregate(port_regions),
        oddness=dict(fields=len(premises), positions=sum(row["elements"] for row in premises),
            numerical_changed=sum(row["numerical_changed"] for row in premises), bit_changed_elements=sum(row["bit_changed_elements"] for row in premises),
            signed_zero_defects=sum(row["zero_signbit_defects"] for row in premises), maxabs=max(row["max_abs"] for row in premises),
            max_rms=max(row["rms"] for row in premises)), per_case=per_case,
        counting_scope="Each of three endpoint channels and three response channels is separate; by-head rows partition each channel, not additional counts. Prefix, regional, aligned-port and premise accounts overlap. Pooled error summaries are bookkeeping across fields, not a gain/energy attribution or independent experiment count.")


def compact_account(collection, pack):
    receipt = read(collection / "collection-receipt.json")
    require(sha(pack / "files.json") == receipt["files_manifest_sha256"], "collected compact wrapper differs")
    files = read(pack / "files.json")
    actual = {path.relative_to(pack).as_posix() for path in pack.rglob("*") if path.is_file() and path != pack / "files.json"}
    require(actual == set(files) and not any(path.is_symlink() for path in pack.rglob("*")), "complete compact member inventory differs")
    for name, digest in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and sha(pack / path) == digest, "compact member differs: " + name)
    return dict(collection=receipt, collection_receipt_sha256=sha(collection / "collection-receipt.json"), manifest_sha256=sha(pack / "files.json"), members=len(files),
        bytes=sum((pack / name).stat().st_size for name in files), wrapper_bytes=(pack / "files.json").stat().st_size,
        retained_array_members=[name for name in files if name.endswith((".npy", ".npz", ".safetensors"))])


def job_account(run, gpu, out):
    from mdb.job_support import scheduler_api

    request = read(run / "request.json")
    require(identity(request) == request["request_sha256"], "local standard request identity differs")
    job_id = read(run / "submission.json")["job_id"]
    api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    job = json.loads(raw)
    result = job.get("result") or {}
    require(status == 200 and job["job_id"] == job_id and job["state"] == "succeeded" and result.get("returncode") == 0, "completed successful exact job required")
    for envelope in (job["config"], job["submitted_request"]["config"]):
        require(all(envelope.get(key) == value for key, value in request.items()), "actual executed request differs")
    require(result["diagnostics"]["command"] == read(run / "admission-command.json")["command"] and not job["config"]["retry_on_kill"], "actual command/retry policy differs")
    reservation = job["submitted_request"]["reservation"]
    require(all(reservation[key] == request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads"))
        and bool(job["submitted_request"]["needs"].get("cuda", False)) == gpu, "actual reservation/device differs")
    custody = [job["payload_custody"][key] for key in ("declared", "sealed", "executed")] + [result["executed_payload"]]
    require(len({(row["sha256"], row["size_bytes"]) for row in custody}) == 1, "declared/sealed/executed/result payload custody differs")
    (out / (job_id + ".job.json")).write_bytes(raw)
    status, log, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = result["artifacts"]["job_log"]
    require(status == 200 and len(log) == expected["bytes"] and len(log) <= 4 << 20 and hashlib.sha256(log).hexdigest() == expected["sha256"], "complete actual worker log differs")
    (out / (job_id + ".worker.log")).write_bytes(log)
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    require(len(markers) == 1 and markers[0]["job_id"] == job_id and markers[0]["request_sha256"] == request["request_sha256"], "one exact directory marker required")
    return dict(job_id=job_id, request_sha256=request["request_sha256"], request=request, marker=markers[0],
        job_json_sha256=hashlib.sha256(raw).hexdigest(), complete_log_sha256=hashlib.sha256(log).hexdigest(),
        command=result["diagnostics"]["command"], reservation=reservation, executed_payload=result["executed_payload"],
        **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")})


def changed(value):
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        key = sorted(result)[0]
        result[key] = changed(result[key])
        return result
    if type(value) is bool:
        return not value
    if isinstance(value, str):
        return ("1" if value == "0" * 64 else "0") * 64
    raise ValueError("Unsupported source/profile condition for bounded abstention")


def changed_profile(execution):
    result = copy.deepcopy(execution)
    policy = result.get("primitive_policy", {})
    candidates = [key for key, value in policy.items() if type(value) is bool]
    require(bool(candidates), "actual profile has no Boolean primitive flag for bounded contrast")
    key = sorted(candidates)[0]
    policy[key] = not policy[key]
    return result


def cli_accounts(pack, out):
    catalog = pack / "catalog/equations"
    records = [read(path) for path in sorted(catalog.glob("*.json"))]
    require(len(records) == 3, "three conditional catalog occurrences required")
    queries = []
    for record in records:
        context = record["tested_context"]
        predicates = {row["field"]: row for row in record["applicability_conditions"]}
        require(all(predicates[key]["op"] == "eq" and predicates[key]["value"] == context[key] for key in ("source_identity", "execution")), "actual source/profile conditions differ")
        variants = (("exact", context, 1), ("changed-source", {**context, "source_identity": changed(context["source_identity"])}, 0),
                    ("changed-profile", {**context, "execution": changed_profile(context["execution"])}, 0))
        for kind, query, expected in variants:
            stem = record["alias"] + "." + kind
            source = out / (stem + ".context.json")
            save(source, query)
            commands = {
                "equations": [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find", "--catalog", str(catalog), "--alias", record["alias"], "--applicable-to", str(source)],
                "atlas": [sys.executable, "-I", "-B", "-m", "mdb.computation_atlas", "--atlas", str(pack / "atlas"), "find", "--law-id", record["law_id"], "--applicable-to", str(source)],
            }
            for api, command in commands.items():
                result = subprocess.run(command, cwd=out, capture_output=True, text=True, check=True)
                matches = json.loads(result.stdout)
                require(len(matches) == expected, "installed applicability count differs")
                if matches:
                    identity = matches[0]["fingerprint"] if api == "equations" else matches[0]["source_record"]["fingerprint"]
                    require(identity == record["fingerprint"], "installed CLI matched wrong occurrence")
                target = out / (stem + "." + api + ".cli.json")
                target.write_text(result.stdout)
                queries.append(dict(api=api, alias=record["alias"], kind=kind, matches=len(matches), command=command,
                    context_sha256=sha(source), result_sha256=sha(target)))
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=ROOT / "compact-collection-01")
    parser.add_argument("--native-run", type=Path, default=ROOT / "run-02")
    parser.add_argument("--audit-run", type=Path, default=ROOT / "audit-01")
    parser.add_argument("--output", type=Path, default=ROOT / "saved-report-verification-v1")
    parser.add_argument("--expected-native-request", default=NATIVE_REQUEST)
    args = parser.parse_args()
    os.environ.setdefault("MRUN_URL", "http://192.168.68.68:9025")
    pack = args.collection.resolve() / "extracted/binding-minimal-result"
    compact = compact_account(args.collection, pack)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    jobs = dict(native=job_account(args.native_run, True, out), saved_audit=job_account(args.audit_run, False, out))
    native, saved = jobs["native"], jobs["saved_audit"]
    audit, summary, saved_request, source_request = (read(pack / name) for name in ("audit.json", "summary.json", "request.json", "source-request.json"))
    require(native["request_sha256"] == args.expected_native_request == audit["request_sha256"] and source_request == native["request"]
        and saved_request == saved["request"] and summary["job_id"] == saved["job_id"] and summary["source_job"] == audit["job_id"] == native["job_id"], "compact actual source/audit job-request joins differ")
    require(audit["audit_sha256"] == saved_request["source_hashes"]["audit_qwen_sign_shortcut.py"] == READER_SHA256, "corrected saved reader source differs")
    source_plan, source_summary, source = (read(pack / name) for name in ("source-plan.json", "source-summary.json", "source-source.json"))
    require(source_plan == dict(native["request"]["config"], output=native["marker"]["output_root"], request_source_hashes=native["request"]["source_hashes"],
        request_sha256=native["request_sha256"], job_id=native["job_id"]) and read(pack / "source-declared-config.json") == native["request"]["config"],
        "actual effective/declared config differs from executed request")
    require(source["source_hashes"] == native["request"]["source_hashes"] and source["job_id"] == source_summary["job_id"] == native["job_id"]
        and source["request_sha256"] == source_summary["request_sha256"] == native["request_sha256"]
        and source["source_hashes"]["qwen_sign_shortcut_campaign.py"] == audit["worker_sha256"] and source_summary["counts"] == audit["counts"], "source/core/report execution joins differ")
    require(saved_request["config"]["native_root"] == native["marker"]["output_root"]
        and saved_request["config"]["native_request_sha256"] == native["request_sha256"], "saved-reader actual source root/request differs")
    require(saved_request["config"]["native_manifest_sha256"] == audit["native_manifest_sha256"] == sha(pack / "source-files.json")
        == native["marker"]["files_manifest_sha256"] and native["marker"]["summary_sha256"] == sha(pack / "source-summary.json")
        and saved["marker"]["files_manifest_sha256"] == compact["manifest_sha256"] and saved["marker"]["summary_sha256"] == sha(pack / "summary.json"), "original/compact marker-manifest custody differs")
    raw = read(pack / "source-files.json")
    require(raw["job_id"] == native["job_id"] and raw["request_sha256"] == native["request_sha256"]
        and raw["expanded_bytes"] == sum(row["bytes"] for row in raw["files"].values()), "original raw wrapper/byte count differs")
    totals = rollup(audit)
    cli = cli_accounts(pack, out)
    result = dict(schema="mdb-qwen-sign-shortcut-actual-saved-report-verification-v1", script_sha256=sha(__file__), audit_sha256=sha(pack / "audit.json"),
        jobs=jobs, compact=compact, raw=dict(members=len(raw["files"]), bytes=raw["expanded_bytes"]), totals=totals,
        source=dict(reader_sha256=READER_SHA256, worker_sha256=audit["worker_sha256"], parameter_restoration=audit["source"]["parameter_restoration"],
                    parameter_recipe_io=audit["source"]["parameter_recipe_io"], runtime=audit["source"]["runtime"]),
        actual_counts=audit["counts"], actual_contract=audit["contract"], installed_cli=cli,
        status="Saved-report custody and recount completed; numerical observations preserved without a success threshold.",
        scope="Three conditional occurrences of one supplied shortcut rule. Six error channels remain distinct; regions/ports/premises overlap. No seed assumption, family induction, symbolic proof, new tensor reduction or model/tokenizer call.")
    save(out / "verification.json", result)
    print(json.dumps(dict(receipt_sha256=sha(out / "verification.json"), totals={key: value for key, value in totals.items() if key != "per_case"}, actual_cli_queries=len(cli)), sort_keys=True))


if __name__ == "__main__":
    main()

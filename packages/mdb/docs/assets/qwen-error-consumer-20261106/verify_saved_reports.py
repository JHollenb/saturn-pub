"""Independent compact-only recount, local job custody and installed MDB queries.

No tensor/checkpoint, model, scheduler, stage or submission access. Scientific
metrics remain observations; only declared custody and account mechanics reject.
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
PRIOR = ROOT.parent / "mdb-qwen-cached-qkv-20261106/verify_saved_reports.py"
PRIOR_SHA256 = "a5a51d79b8bf43fe70b897a4eb62b6ab80a1f5e883e78b16496e51fb3fa99d41"
PRODUCER_SHA256 = "ad82b86dd837a1f2bc9134f2c5bda3dcb097e8f8583e7849fcfbe8d5c49b229b"
READER_SHA256 = "c0adfae449b426c52c91a5cb2e05fd27453cdd704354dbd7dd465a194d19723b"
NATIVE_REQUEST = "d0bb13015d8bcb735f0de5cc2a4587d8aa96eb13b7af51d6c6af391015cdf827"
AUDIT_REQUEST = "ddd5e5218048b41b5ffe0ff86748b55a459a748c60ea93f219151ad2e8fe6619"
NATIVE_MANIFEST = "d4beda87d13774a57be662bf2335aaf3cc23f0ddc305ac101942926f5cf4ef5d"
COMPACT_MANIFEST = "f3b67a7fe77cf1003775b6e2fd551b79ba4580c4e1f9330b439c4cdf8fe652ab"
COLLECTOR_SHA256 = "10bcf0c84c4fffa2eed7f1665a4a706cc3aea365d9c8e64e6a9323c865823877"
PORTS = ("head", "projection", "R", "M", "Hnext")
WORLDS = ("full-input", "cheap-input")
POSITIONS = {name: (2097152 if name == "head" else 1310720) for name in PORTS}
NATIVE_PINS = {"qwen_error_consumer_campaign.py": PRODUCER_SHA256,
    "qwen_component_experiment_v2.py": "c93324ee76156f2fba16f70d7cef78b7accbf6020f0c899cb04d856c348b5762",
    "stage_qwen_error_consumer.py": "c7574da612f8548e0574e627496e273c6ac8f81e8b04f5984fb368759d21cf28"}
AUDIT_PINS = {"audit_qwen_error_consumer.py": READER_SHA256,
    "stage_qwen_error_consumer_saved_audit.py": "0556094604951913293e6c5704487754420d6328a35bea258d9f77bad61c0d87"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if sha(PRIOR) != PRIOR_SHA256:
    raise ValueError("frozen compact bookkeeping helper differs")
H = runpy.run_path(str(PRIOR))
require, read, save, identity, metrics = (H[name] for name in ("require", "read", "save", "identity", "metrics"))
BOOK = H["H"]["H"]


def regions(rows, protocol):
    for name, value in rows:
        parts = value["windows"]
        require([part["bounds"] for part in parts] == protocol["windows"], "declared dependent window coordinates differ")
        require(value["whole"]["sample_count"] == POSITIONS[name]
            and all(part["comparison"]["sample_count"] == POSITIONS[name] * (part["bounds"][1] - part["bounds"][0]) // 512 for part in parts),
            "complete field/window position coverage differs")
        for key in ("sample_count", "changed_values"):
            require(value["whole"][key] == sum(part["comparison"][key] for part in parts), "whole/dependent window census differs")
        for key in ("shared", "defects"):
            require(value["whole"]["signed_zero"][key] == sum(part["comparison"]["signed_zero"][key] for part in parts), "whole/window zero-sign census differs")
    return dict(whole=metrics([value["whole"] for _, value in rows]),
        dependent_windows=metrics([part["comparison"] for _, value in rows for part in value["windows"]]))


def rollup(audit, protocol):
    require(tuple(protocol["world_order"]) == WORLDS and protocol["sequence_rows"] == 512
        and protocol["windows"] == [[0, 256], [256, 512]], "fixed two-world/full512 report domain differs")
    coverage = dict(primary_endpoint_fields=10, primary_response_fields=5, dependent_endpoint_windows=20, dependent_response_windows=10)
    require(audit["coverage"] == coverage and set(audit["worlds"]) == set(WORLDS) and set(audit["responses"]) == set(PORTS), "primary port/world coverage differs")
    endpoints = []
    for world in WORLDS:
        row = audit["worlds"][world]
        require(set(row["fields"]) == set(PORTS) and row["comparison_elements"] == sum(POSITIONS.values()), "complete endpoint comparison budget differs")
        endpoints.extend((name, row["fields"][name]) for name in PORTS)
    for name in PORTS:
        require(set(audit["responses"][name]) == {"error", "functional", "native"}, "response error/magnitude channels differ")
        regions([(name, audit["responses"][name][kind]) for kind in ("functional", "native")], protocol)
    result = dict(endpoint_errors=regions(endpoints, protocol),
        response_errors=regions([(name, audit["responses"][name]["error"]) for name in PORTS], protocol),
        per_world=audit["worlds"], per_field_responses=audit["responses"], input_difference=audit["input_difference"],
        residual_accounts=audit["residual_accounts"], coverage=coverage,
        counting_scope="10 endpoint fields/14680064 positions and5 response-error fields/7340032 positions;20/10 dependent windows overlap those same coordinates. Two worlds are dependent observations; numeric/raw-bit/zero-sign and residual diagnostics remain separate.",
        label_limit="Frozen c0ad cross-term left/right fields are numeric summaries whose term labels were overwritten; original numeric rows retained. Any separate label sidecar supplies names without changing these rows.")
    require(result["endpoint_errors"]["whole"]["positions"] == 14680064 and result["response_errors"]["whole"]["positions"] == 7340032, "full primary position totals differ")
    return result


def payload_account(run, request, pins, worker):
    require(identity(request) == request["request_sha256"] and all(request["source_hashes"].get(name) == digest for name, digest in pins.items()), "request/frozen source identity differs")
    payload = run / "payload"
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and not name.endswith((".npy", ".npz", ".safetensors"))
            and sha(payload / name) == digest, "local source/wheel payload differs: " + name)
    seal = read(payload / "payload-seal.json")
    require(seal["files"] == request["source_hashes"] and seal["wheels"] == request["runtime_wheels"] and seal["worker"] == worker
        and seal["worker_sha256"] == request["source_hashes"][worker] and seal["config_bytes_sha256"] == sha(payload / "saturn_run_config.json")
        and read(payload / "saturn_run_config.json") == request["config"] and read(payload / "saturn_debug_request.json") == request, "sealed payload request/config/source/wheels differ")
    return dict(source_files=len(request["source_hashes"]), runtime_wheels=request["runtime_wheels"], seal_sha256=sha(payload / "payload-seal.json"))


def captured_job(run, request, job_id, job_path, log_path, *, gpu, expected_marker):
    require(job_path is not None and log_path is not None, "complete local job JSON and log are required")
    job, log = read(job_path), Path(log_path).read_bytes(); result = job.get("result") or {}
    require(job["job_id"] == job_id and job["state"] == "succeeded" and result["returncode"] == 0 and job["assigned_host"] == "beast", "completed exact captured job differs")
    for envelope in (job["config"], job["submitted_request"]["config"]):
        require(all(envelope.get(key) == value for key, value in request.items()), "actual captured request envelope differs")
    command = read(run / "admission-command.json")["command"]
    require(result["diagnostics"]["command"] == job["submitted_request"]["cmd"] == job["cmd"] == command
        and job["config"]["retry_on_kill"] is False, "actual command/no-retry policy differs")
    require(bool(job["needs"].get("cuda", False)) == bool(job["submitted_request"]["needs"].get("cuda", False)) == gpu, "actual CPU/CUDA admission differs")
    actual, submitted = job["reservation"], job["submitted_request"]["reservation"]
    if gpu:
        require(request["resource_exception"] is None and request["sizing"] == dict(mode="intent-first", reservation=None, timeout_s=300)
            and submitted is None and actual == dict(ram_mb=4096.0, vram_mb=4096.0, cpu_threads=4, disk_gb=1.0, est_wall_s=None, source="probe-default"),
            "actual intent-first/probe-default reservation differs")
    else:
        resource = request["resource_exception"]
        require(all(submitted[key] == actual[key] == resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s")), "actual explicit CPU reservation differs")
    custody = [job["payload_custody"][key] for key in ("declared", "sealed", "executed")] + [result["executed_payload"]]
    require(len({(value["sha256"], value["size_bytes"]) for value in custody}) == 1, "declared/sealed/executed/result payload custody differs")
    expected = result["artifacts"]["job_log"]
    require(len(log) <= 4 << 20 and len(log) == expected["bytes"] and hashlib.sha256(log).hexdigest() == expected["sha256"], "complete captured log checksum differs")
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    require(markers == [expected_marker], "actual final directory marker differs")
    return dict(status="authenticated-local-capture", job_id=job_id, command=command, actual_reservation=actual,
        submitted_reservation=submitted, sizing=request.get("sizing"), planner_error=job["config"].get("_plan_error"), selected_plan=job.get("plan"),
        job_json_sha256=sha(job_path), complete_log_sha256=sha(log_path), executed_payload=result["executed_payload"],
        **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")})


def transaction_census(pack, audit, protocol, summary, raw):
    tx, census = audit["transaction"], audit["census"]
    verified = tx["verified"]
    require(verified["verified"] is True and verified["events"] == 7 and verified["phase"] == "closed" and verified["frontiers"] == ["inherited-worlds"]
        and verified["forecasts"] == ["cohort"] and verified["seal_sha256"] == summary["forecast_seal"] == raw["files"]["transaction/seal.json"]["sha256"]
        and verified["event_head"] == summary["transaction"]["event_head"] and tx["inherited_arrays"] == 7 and tx["native_targets_at_seal"] == 0,
        "reported seven-event transaction/seal identity differs")
    require(len([name for name in raw["files"] if name.startswith("transaction/events/")]) == 7, "raw transaction event census differs")
    journal = [json.loads(line) for line in (pack / "source-timeline.jsonl").read_text().splitlines()]
    kinds = ["functional_world_completed"]*2 + ["cohort_globally_sealed", "native_targets_released"] + ["native_component_authorized", "native_component_completed"]*2
    require(tx["chronology"] == journal and [row["kind"] for row in journal] == kinds and [row["number"] for row in journal] == list(range(8))
        and [row["native_module_entries"] for row in journal] == [0, 0, 0, 0, 0, 15, 15, 30]
        and journal[2]["seal_sha256"] == journal[3]["seal_sha256"] == summary["forecast_seal"] and journal[2]["native_targets_at_seal"] == 0,
        "global pretarget/native chronology differs")
    responses = read(pack / "source-forecast-responses.json")
    require(responses == audit["response_commitments"] and {key: responses[key] for key in protocol["response"]} == protocol["response"]
        and responses["baseline_world"] == "full-input" and set(responses["fields"]) == set(PORTS)
        and all(responses["fields"][name]["elements"] == POSITIONS[name] and responses["fields"][name]["dtype"] == "float64" for name in PORTS), "complete presealed FP64 response commitment differs")
    contract, sdpa, source = (read(pack / name) for name in ("source-native-contract.json", "source-sdpa.json", "source-source.json"))
    require(census["observed"] == census["declared"] == summary["counts"] == contract["expected"] == contract["observed"] == protocol["physical_plan"]
        and census["actual_SDPA_calls"] == sdpa["calls"] and len(sdpa["calls"]) == 4
        and [(row["lane"], row["arm"]) for row in sdpa["calls"]] == [(lane, world) for lane in ("functional", "native") for world in WORLDS]
        and census["native_entries"] == contract["block_entries"] and [row["world"] for row in contract["block_entries"]] == list(WORLDS)
        and source["hydration"]["component_hydrations"] == source["hydration"]["cuda_placements"] == 1,
        "four actual SDPA/two consumers/one hydration census differs")
    require(contract["projection_provenance"] == "actual passive o_proj output hook" and contract["R_provenance"] == "actual post_attention_layernorm input hook", "actual projection/R consumer provenance differs")
    forecast, native, frontier = (read(pack / name) for name in ("source-forecast.json", "source-native.json", "source-source-frontier.json"))
    require(native["context"] == frontier["context"], "shared consumed context descriptors differ")
    for call in sdpa["calls"]:
        ports = (forecast if call["lane"] == "functional" else native)["worlds"][call["arm"]]["ports"]
        require(call["QKV"] == [ports[name] for name in ("q", "k", "v")] and call["attention_mask"] is None and call["is_causal"] is True
            and call["enable_gqa"] is True and call["dropout_p"] == 0. and call["scale"] == 128**-.5, "actual SDPA operand/profile join differs")
        for record, live in zip(call["QKV"], call["operands"], strict=True):
            require(all(record[key] == live[key] for key in ("shape", "stride", "content_sha256"))
                and record["logical_dtype"] == live["dtype"] and record["device_at_capture"] == live["device"], "actual QKV hash/layout descriptor differs")
    for world, entry in zip(WORLDS, contract["block_entries"], strict=True):
        require(entry["input_reference"] == frontier["worlds"][world]["input_reference"]
            and entry["input"] == native["worlds"][world]["actual_input_descriptor"], "actual two-world consumed input join differs")
    return dict(transaction=tx, census=census, response_commitments=responses,
        authority="Crossjoins authenticated compact report and copied raw metadata. Public Tx/raw-array verification was performed by the source-pinned CPU reader; this compact verifier does not rehydrate arrays or claim a new raw replay.")


def cli_accounts(pack, output):
    records = [read(path) for path in sorted((pack / "catalog/equations").glob("*.json"))]
    require(len(records) == 2 and {record["tested_context"]["world"] for record in records} == set(WORLDS), "two conditional catalog worlds required")
    calls = []
    for record in records:
        context = record["tested_context"]
        predicates = {row["field"]: row for row in record["applicability_conditions"]}
        require(all(predicates.get(key) == dict(field=key, op="eq", value=value) for key, value in context.items()), "source/clock/profile predicates differ")
        require(context["execution"]["learned_dtype"] == "torch.bfloat16", "declared source profile dtype differs")
        final = ("changed-clock", {**context, "horizon": 21}) if context["world"] == "full-input" else ("changed-profile", {**context, "execution": {**context["execution"], "learned_dtype": "torch.float32"}})
        queries = [("exact", context), ("changed-source", {**context, "source_identity": BOOK["changed"](context["source_identity"])}), final]
        for kind, query in queries:
            stem = record["alias"] + "." + kind; path = output / (stem + ".context.json"); save(path, query)
            for api, command in {
                "equations": [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find", "--catalog", str(pack / "catalog/equations"), "--alias", record["alias"], "--applicable-to", str(path)],
                "atlas": [sys.executable, "-I", "-B", "-m", "mdb.computation_atlas", "--atlas", str(pack / "atlas"), "find", "--law-id", record["law_id"], "--applicable-to", str(path)],
            }.items():
                completed = subprocess.run(command, cwd=output, capture_output=True, text=True, check=True)
                matches = json.loads(completed.stdout)
                require(len(matches) == (1 if kind == "exact" else 0) and all((row["fingerprint"] if api == "equations" else row["source_record"]["fingerprint"]) == record["fingerprint"] for row in matches), "actual installed catalog/atlas query differs")
                target = output / (stem + "." + api + ".cli.json"); target.write_text(completed.stdout)
                calls.append(dict(api=api, world=context["world"], alias=record["alias"], contrast=kind, matches=len(matches), command=command, context_sha256=sha(path), result_sha256=sha(target)))
    require(len(calls) == 12 and sum(row["matches"] for row in calls) == 4, "12-query/4-match/8-abstention census differs")
    return calls


def verify(args):
    pack = args.collection.resolve() / "extracted/binding-minimal-result"
    compact = BOOK["compact_account"](args.collection, pack)
    require(compact["manifest_sha256"] == args.expected_compact_manifest and not compact["retained_array_members"], "published compact identity/no-array boundary differs")
    native, saved = (read(run / "request.json") for run in (args.native_run, args.audit_run))
    require(native["request_sha256"] == args.expected_native_request and saved["request_sha256"] == args.expected_audit_request, "expected native/audit request pin differs")
    payloads = dict(native=payload_account(args.native_run, native, NATIVE_PINS, "qwen_error_consumer_campaign.py"),
        audit=payload_account(args.audit_run, saved, AUDIT_PINS, "audit_qwen_error_consumer.py"))
    audit, summary, source, plan, source_summary, raw = (read(pack / name) for name in ("audit.json", "summary.json", "source-source.json", "source-plan.json", "source-summary.json", "source-files.json"))
    nj, aj = (read(run / "submission.json")["job_id"] for run in (args.native_run, args.audit_run))
    require(read(pack / "source-request.json") == native and read(pack / "request.json") == saved
        and audit["job_id"] == summary["source_job"] == source["job_id"] == source_summary["job_id"] == nj and summary["job_id"] == aj
        and audit["request_sha256"] == source["request_sha256"] == source_summary["request_sha256"] == native["request_sha256"]
        and summary["request_sha256"] == saved["request_sha256"] and audit["audit_sha256"] == READER_SHA256 and audit["worker_sha256"] == PRODUCER_SHA256,
        "compact/source/native/audit identity joins differ")
    output = str(Path(native["config"]["output_scratch"]) / nj / native["config"]["output_name"])
    require(plan == dict(native["config"], output=output, request_source_hashes=native["source_hashes"], request_sha256=native["request_sha256"], job_id=nj)
        and read(pack / "source-declared-config.json") == native["config"] and source["source_hashes"] == native["source_hashes"]
        and saved["config"]["native_root"] == output and saved["config"]["native_request_sha256"] == native["request_sha256"]
        and saved["config"]["native_manifest_sha256"] == audit["native_manifest_sha256"] == sha(pack / "source-files.json") == args.expected_native_manifest,
        "effective source config/raw identity differs")
    receipt = compact["collection"]
    require(receipt["schema"] == "mdb-error-consumer-compact-retrieval-v1" and receipt["collector_sha256"] == COLLECTOR_SHA256
        and receipt["generic_transport_sha256"] == "007098eb72842fe1f5d585f64eae272812303d1c18eb67a17ba43508c646b18d"
        and receipt["metadata_scope_sha256"] == "b621d8e12dc06476abf356643f22693a48a5a52bd43c7684d66e788126475015"
        and receipt["job_id"] == aj and receipt["native_job_id"] == nj and receipt["request_sha256"] == saved["request_sha256"]
        and receipt["raw_tensor_arrays_collected"] is False and receipt["model_calls"] == 0, "compact collector custody differs")
    require(raw["job_id"] == nj and raw["request_sha256"] == native["request_sha256"] and len(raw["files"]) == 105
        and raw["expanded_bytes"] == sum(row["bytes"] for row in raw["files"].values()) == 240234784, "original raw member/byte census differs")
    for path in pack.glob("source-*"):
        if path.is_file() and path.name != "source-files.json":
            member = path.name.removeprefix("source-")
            require(raw["files"][member] == dict(bytes=path.stat().st_size, sha256=sha(path)), "copied original metadata differs: " + member)
    for key, record in source["metadata"].items():
        path = pack / "source-metadata" / Path(record["path"]).name
        require(sha(path) == record["sha256"] == raw["files"][record["path"]]["sha256"], "retained fourteen-source metadata join differs: " + key)
    protocol = read(pack / "source-metadata/selection_protocol.json")
    coverage = read(pack / "coverage.json")
    require({key: coverage[key] for key in audit["coverage"]} == audit["coverage"] and coverage["model_calls"] == coverage["tokenizer_calls"] == coverage["learned_projections"] == 0
        and coverage["primary_and_windows_separate"] is True and coverage["independent_occurrences"] is False, "saved-only declared coverage differs")
    accounts, totals = transaction_census(pack, audit, protocol, source_summary, raw), rollup(audit, protocol)
    args.output.mkdir(parents=True, exist_ok=False); out = args.output.resolve()
    native_marker = dict(job_id=nj, request_sha256=native["request_sha256"], output_root=output, files_manifest_sha256=sha(pack / "source-files.json"), summary_sha256=sha(pack / "source-summary.json"), file_count=len(raw["files"]), expanded_bytes=raw["expanded_bytes"])
    audit_marker = dict(job_id=aj, request_sha256=saved["request_sha256"], output_root=str(Path(saved["config"]["output_scratch"]) / aj / "binding-minimal-result"), files_manifest_sha256=compact["manifest_sha256"], summary_sha256=sha(pack / "summary.json"))
    jobs = dict(native=captured_job(args.native_run, native, nj, args.native_job, args.native_log, gpu=True, expected_marker=native_marker),
        audit=captured_job(args.audit_run, saved, aj, args.audit_job, args.audit_log, gpu=False, expected_marker=audit_marker))
    for record in (read(path) for path in (pack / "catalog/equations").glob("*.json")):
        context = record["tested_context"]
        require(context["source_identity"] == source["source_hashes"] and context["model_identity"] == plan["qwen_parameter_manifest_sha256"]
            and context["input_identity"] == protocol["world_sources"][context["world"]]["content_sha256"] and context["horizon"] == 20, "catalog source/model/world/clock binding differs")
    result = dict(schema="mdb-qwen-error-consumer-compact-verification-v1", verifier_sha256=sha(__file__), helper_sha256=PRIOR_SHA256,
        audit_sha256=sha(pack / "audit.json"), compact=compact, payloads=payloads, jobs=jobs, raw=dict(members=105, bytes=240234784), totals=totals,
        source=audit["source"], parents=audit["parents"], **accounts, actual_installed_cli=cli_accounts(pack, out),
        scope="Compact JSON/code/wheels and explicit complete local job/log captures only; no arrays, weights, models, scheduler, staging or submission. Numeric/raw-bit/zero-sign magnitudes and countertrends are preserved without a scientific gate.")
    save(out / "verification.json", result)
    print(json.dumps(dict(receipt_sha256=sha(out / "verification.json"), endpoint=totals["endpoint_errors"], response=totals["response_errors"], actual_CLI_calls=len(result["actual_installed_cli"])), sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("collection", "compact-collection-01"), ("native-run", "run-01"), ("audit-run", "audit-01"), ("output", "saved-report-verification-v1")):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    for name, default in (("native-job", "run-01/job-completed.json"), ("native-log", "run-01/job.log"), ("audit-job", "audit-01/job-completed.json"), ("audit-log", "audit-01/job.log")):
        parser.add_argument("--" + name, type=Path, default=ROOT / default)
    for name, default in (("native-request", NATIVE_REQUEST), ("audit-request", AUDIT_REQUEST), ("native-manifest", NATIVE_MANIFEST), ("compact-manifest", COMPACT_MANIFEST)):
        parser.add_argument("--expected-" + name, default=default)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()

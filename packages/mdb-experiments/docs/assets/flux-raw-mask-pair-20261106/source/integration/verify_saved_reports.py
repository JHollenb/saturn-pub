"""Recount actual compact evidence and run installed MDB applicability queries."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from mdb.job_support import scheduler_api

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "compact-collection-01/extracted/binding-minimal-result"
OUT = ROOT / "saved-report-verification-v1"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate_job(run, expected_id, gpu, expected_state):
    request = read(run / "request.json")
    api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + expected_id, timeout_s=30)
    job = json.loads(raw)
    if status != 200 or job["job_id"] != expected_id or job["state"] != expected_state:
        raise ValueError("Exact original/recovery job state differs")
    result = job["result"]
    if result["returncode"] != (1 if expected_state == "failed" else 0):
        raise ValueError("Actual exit differs")
    for key, value in request.items():
        if job["config"][key] != value:
            raise ValueError("Submitted request differs: " + key)
    if (job["config"]["retry_on_kill"] is not False
            or result["diagnostics"]["command"] != read(run / "admission-command.json")["command"]):
        raise ValueError("Actual command/retry policy differs")
    reservation = job["submitted_request"]["reservation"]
    if any(reservation[key] != request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads")):
        raise ValueError("Actual reservation differs")
    if bool(job["submitted_request"]["needs"].get("cuda", False)) != gpu:
        raise ValueError("Actual device differs")
    custody = [job["payload_custody"][lane]["sha256"] for lane in ("declared", "sealed", "executed")]
    custody.append(result["executed_payload"]["sha256"])
    if len(set(custody)) != 1:
        raise ValueError("Executed payload differs")
    (OUT / (expected_id + ".job.json")).write_bytes(raw)
    status, log, _ = api.request("GET", "/api/jobs/" + expected_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = result["artifacts"]["job_log"]
    if status != 200 or len(log) != expected["bytes"] or hashlib.sha256(log).hexdigest() != expected["sha256"]:
        raise ValueError("Complete worker log differs")
    (OUT / (expected_id + ".worker.log")).write_bytes(log)
    return {"job_id": expected_id, "state": job["state"], "request_sha256": request["request_sha256"],
        "executed_payload_sha256": custody[0], "job_sha256": hashlib.sha256(raw).hexdigest(),
        "worker_log_sha256": hashlib.sha256(log).hexdigest(), "elapsed_s": result["elapsed_s"],
        "peak_rss_mb": result["peak_rss_mb"], "peak_vram_mb": result["peak_vram_mb"], "reservation": reservation}


def summarize(rows):
    return {"field_comparisons": len(rows), "scalar_positions": sum(row["sample_count"] for row in rows),
        "bitwise_exact_fields": sum(row["bitwise_exact"] for row in rows),
        "changed_values": sum(row["changed_values"] for row in rows),
        "maxabs": max(row["maxabs"] for row in rows), "max_rms": max(row["rms"] for row in rows)}



def response_summary(rows):
    return {"field_comparisons": len(rows),
        "scalar_positions": sum(row["full"]["sample_count"] for row in rows),
        "verified_forecast_hashes": sum(row["response_hash_verified"] for row in rows),
        "changed_values": sum(row["full"]["changed_values"] for row in rows),
        "maxabs": max(row["full"]["maxabs"] for row in rows),
        "max_rms": max(row["full"]["rms"] for row in rows)}


def main():
    OUT.mkdir(exist_ok=False)
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    audit = read(PACK / "audit.json")
    inventory = read(PACK / "files.json")
    for name, digest in inventory.items():
        if sha(PACK / name) != digest:
            raise ValueError("Collected compact bytes differ: " + name)
    original = authenticate_job(ROOT / "run-02", "job-0b6c4340d5e6", True, "succeeded")
    saved = authenticate_job(ROOT / "audit-03", "job-ce54f7b5406a", False, "succeeded")
    failed = [authenticate_job(ROOT / run, job, False, "failed") for run, job in
        (("audit-01", "job-56860c50186f"), ("audit-02", "job-7a55a56683eb"))]
    per_case, groups = {}, {}
    for case, report in audit["cases"].items():
        rows = {
            "qwen": [row["full"] for row in report["qwen"]["fields"].values()],
            "denoiser": [metric for row in report["denoiser"]["boundaries"] for metric in row["fields"].values()],
            "decoded_and_pixels": [row["terms"]["whole_functional_minus_native"] for row in report["decoder"]["fields"].values()]}
        per_case[case] = {name: summarize(metrics) for name, metrics in rows.items()}
        for name, metrics in rows.items():
            groups.setdefault(name, []).extend(metrics)
    primary = summarize([row for rows in groups.values() for row in rows])
    signed = audit["signed_responses"]
    response_groups = {
        "qwen": list(signed["qwen"].values()),
        "denoiser": [row for boundary in signed["denoiser"] for row in boundary["fields"].values()],
        "decoded_and_pixels": list(signed["decoder"].values())}
    response_all = response_summary([row for rows in response_groups.values() for row in rows])
    queries = []
    catalog = PACK / "catalog/equations"
    for path in sorted(catalog.glob("*.json")):
        record = read(path)
        context = record["tested_context"]
        variants = (("exact", context, 1),
            ("changed-seed", {**context, "seed": 278}, 0),
            ("changed-clock", {**context, "clock": {"step": 1, "phase": "project", "cursor": 0}}, 0))
        for kind, query, expected in variants:
            name = context["case"] + "." + context["occurrence"] + "." + kind
            context_path = OUT / (name + ".context.json")
            context_path.write_text(json.dumps(query, indent=2, sort_keys=True) + "\n")
            command = [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find",
                "--catalog", str(catalog), "--alias", record["alias"],
                "--applicable-to", str(context_path)]
            process = subprocess.run(command, cwd=OUT, capture_output=True, text=True, check=True)
            matches = json.loads(process.stdout)
            if len(matches) != expected or matches and matches[0]["fingerprint"] != record["fingerprint"]:
                raise ValueError("Installed MDB applicability differs")
            output = OUT / (name + ".cli.json")
            output.write_text(process.stdout)
            queries.append({"case": context["case"], "occurrence": context["occurrence"],
                "kind": kind, "results": len(matches), "command": command,
                "context_sha256": sha(context_path), "output_sha256": sha(output)})
    result = {"schema": "mdb-raw-mask-pair-actual-saved-report-verification-v1",
        "audit_sha256": sha(PACK / "audit.json"), "compact_manifest_sha256": sha(PACK / "files.json"),
        "native_job": original, "saved_audit_job": saved, "failed_saved_audits": failed,
        "primary": primary, "per_case": per_case,
        "groups": {name: summarize(rows) for name, rows in groups.items()},
        "signed_response": response_all,
        "signed_groups": {name: response_summary(rows) for name, rows in response_groups.items()},
        "counts": audit["counts"], "parsers": {case: row["parser"] for case, row in audit["cases"].items()},
        "first_retained_discrepancies": {case: {
            name: {key: report[name][key] for key in ("first_bit_discrepancy", "first_value_discrepancy")}
            for name in ("qwen", "denoiser")} for case, report in audit["cases"].items()},
        "common_token_prefix": audit["common_token_prefix"],
        "sdpa_census": {lane: {case: {"calls": row["calls"],
            "all_argument_profiles_match": row["all_argument_profiles_match_declared_regime"],
            "selected_kernel": row["selected_kernel"]} for case, row in cases.items()}
            for lane, cases in audit["sdpa"].items()},
        "endpoint_response": {field: row["full"] for field, row in signed["decoder"].items()},
        "catalog_cli": queries, "coverage": audit["coverage"],
        "dependent_repeated_coordinates": True, "raw_arrays_downloaded": False,
        "collection": read(ROOT / "compact-collection-01/collection-receipt.json"),
        "interpretation": "Two complete own-state image paths under different source mask/GQA regimes, one replay control and one prospective long case. Shared primitives, single seed, combined content/truncation/dispatch contrast. Common-prefix numerical drift is diagnostic, not target prediction error. Original forecasts/producer unchanged through all saved-reader repairs."}
    target = ROOT / "actual-saved-report-verification-v1.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(target), "sha256": sha(target), "primary": primary,
        "groups": result["groups"], "signed_response": response_all,
        "signed_groups": result["signed_groups"], "catalog_commands": len(queries)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


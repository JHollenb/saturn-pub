"""Recount actual saved results and run installed MDB applicability searches."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from mdb.job_support import scheduler_api

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "compact-collection-01/extracted/binding-minimal-result"
OUT = ROOT / "saved-report-verification-v2"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticate_job(run, job_id, gpu):
    request = read(run / "request.json")
    api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    job = json.loads(raw)
    result = job.get("result") or {}
    if status != 200 or job["job_id"] != job_id or job["state"] != "succeeded" or result.get("returncode") != 0:
        raise ValueError("Successful exact lease required")
    for envelope in (job["config"], job["submitted_request"]["config"]):
        if any(envelope.get(key) != value for key, value in request.items()):
            raise ValueError("Actual request differs")
    if result["diagnostics"]["command"] != read(run / "admission-command.json")["command"] or job["config"]["retry_on_kill"]:
        raise ValueError("Actual command/retry policy differs")
    reservation = job["submitted_request"]["reservation"]
    if any(reservation[key] != request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads")):
        raise ValueError("Actual reservation differs")
    if bool(job["submitted_request"]["needs"].get("cuda", False)) != gpu:
        raise ValueError("Declared device differs")
    custody = [job["payload_custody"][key] for key in ("declared", "sealed", "executed")] + [result["executed_payload"]]
    if len({(row["sha256"], row["size_bytes"]) for row in custody}) != 1:
        raise ValueError("Executed source custody differs")
    (OUT / (job_id + ".job.json")).write_bytes(raw)
    status, log, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = result["artifacts"]["job_log"]
    if status != 200 or hashlib.sha256(log).hexdigest() != expected["sha256"] or len(log) != expected["bytes"]:
        raise ValueError("Complete log checksum differs")
    (OUT / (job_id + ".worker.log")).write_bytes(log)
    return {"job_id": job_id, "request_sha256": request["request_sha256"], "reservation": reservation,
            "executed_payload": result["executed_payload"], "complete_log_sha256": hashlib.sha256(log).hexdigest(),
            **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")}}


def metrics(value):
    if isinstance(value, dict):
        if {"sample_count", "rms", "bitwise_exact", "maxabs"} <= value.keys():
            return [value]
        return [row for child in value.values() for row in metrics(child)]
    if isinstance(value, list):
        return [row for child in value for row in metrics(child)]
    return []


def summarize(rows):
    return {"field_comparisons": len(rows), "overlapping_scalar_positions": sum(row["sample_count"] for row in rows),
            "bitwise_exact_fields": sum(row["bitwise_exact"] for row in rows),
            "changed_values": sum(row["changed_values"] for row in rows),
            "max_rms": max((row["rms"] for row in rows), default=0),
            "maxabs": max((row["maxabs"] for row in rows), default=0)}


def brief(row):
    return {key: row.get(key) for key in ("rms", "maxabs", "signed_mean", "relative_rms", "changed_values", "sample_count", "bitwise_exact")}


def main():
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    if sha(PACK / "files.json") != "32c1efee36b02a4eab138d95d84bfa76e62a6e2db84a25bd94f9649f948fba50":
        raise ValueError("Actual collected wrapper differs")
    for name, digest in read(PACK / "files.json").items():
        if sha(PACK / name) != digest:
            raise ValueError("Actual collected member differs: " + name)
    OUT.mkdir(exist_ok=False)
    jobs = {name: authenticate_job(ROOT / run, job_id, gpu) for name, run, job_id, gpu in (
        ("native", "run-01", "job-9339e3ed1472", True), ("saved_audit", "audit-01", "job-cf83caa6f2cb", False))}
    audit = read(PACK / "audit.json")
    qwen = metrics(audit["qwen"]["fields"])
    denoiser = [metric for case in audit["cases"].values() for boundary in case["denoiser"]["boundaries"] for metric in metrics(boundary["fields"])]
    decoder_terms = [metric for case in audit["cases"].values() for field in case["decoder"]["fields"].values() for metric in field["terms"].values()]
    response_rows = audit["signed_responses"]["denoiser"]
    response_errors = [field["signed_response_error"] for row in response_rows for field in row["fields"].values()]
    response_errors += [field["signed_response_error"] for field in audit["signed_responses"]["decoder"].values()]
    commits = [{"clock": row["clock"], "latent": brief(row["fields"]["latent"]["native"])} for row in response_rows if row["clock"]["phase"] == "project"]
    projects = [{"clock": row["clock"], "fields": {key: brief(value["native"]) for key, value in row["fields"].items() if key in ("image", "text") or key.startswith("context.")}}
                for row in response_rows if row["clock"]["phase"] == "joint" and row["clock"]["cursor"] == 0]
    earliest = {}
    for name in ("image", "text", "noise_prediction", "latent"):
        changed = next((row for row in response_rows if name in row["fields"] and row["fields"][name]["native"]["changed_values"]), None)
        earliest[name] = None if changed is None else {"clock": changed["clock"], "number": changed["number"], "response": brief(changed["fields"][name]["native"])}
    catalog = PACK / "catalog/equations"
    queries = []
    for path in sorted(catalog.glob("*.json")):
        record = read(path)
        context = record["tested_context"]
        variants = (("exact", context, 1), ("changed-clock", {**context, "clock": {"step": 1, "phase": "project", "cursor": 0}}, 0),
                    ("changed-seed", {**context, "seed": 278}, 0))
        for kind, query, expected in variants:
            basename = record["alias"] + "." + kind
            context_path = OUT / (basename + ".context.json")
            context_path.write_text(json.dumps(query, sort_keys=True, indent=2) + "\n")
            command = [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find", "--catalog", str(catalog),
                       "--alias", record["alias"], "--applicable-to", str(context_path)]
            process = subprocess.run(command, cwd=OUT, capture_output=True, text=True, check=True)
            matches = json.loads(process.stdout)
            if len(matches) != expected or matches and matches[0]["fingerprint"] != record["fingerprint"]:
                raise ValueError("Actual installed applicability search differs")
            result_path = OUT / (basename + ".cli.json")
            result_path.write_text(process.stdout)
            queries.append({"alias": record["alias"], "kind": kind, "matches": len(matches), "command": command,
                            "context_sha256": sha(context_path), "result_sha256": sha(result_path)})
    images = {case: {lane: sha(PACK / (case + "-" + lane + ".png")) for lane in ("forecast", "native-own", "native-reference")}
              for case in ("sham", "e18-zero")}
    receipt = {"schema": "mdb-e18-actual-saved-report-verification-v1", "jobs": jobs, "audit_sha256": sha(PACK / "audit.json"),
        "qwen_and_denoiser": summarize(qwen + denoiser), "qwen": summarize(qwen), "denoiser": summarize(denoiser),
        "decoder_three_term_comparisons": summarize(decoder_terms), "signed_response_errors": summarize(response_errors),
        "decoder_responses": {key: {part: brief(row[part]) for part in ("forecast", "native", "signed_response_error")}
                              for key, row in audit["signed_responses"]["decoder"].items()},
        "input_conditioning_response": brief(response_rows[0]["fields"]["conditioning"]["native"]),
        "input_target_and_collateral": {key: brief(row["native"]) for key, row in response_rows[0]["fields"]["conditioning"]["declared_target_and_collateral_routes"].items()},
        "earliest_numeric_changes": earliest, "project_exits": projects, "latent_commit_responses": commits,
        "actual_counts": audit["actual_counts"], "genuine_receipt_counts": {key: audit["receipt_account"][key] for key in ("actual_acts", "actual_native_advances", "total_receipt_occurrences")},
        "actual_context_operands": audit["context_consumers"]["actual_occurrences"], "installed_cli": queries,
        "original_png_sha256": images, "within_branch_png_byte_identity": {case: len(set(row.values())) == 1 for case, row in images.items()},
        "raw_members": audit["raw_members"], "raw_bytes": audit["raw_expanded_bytes"],
        "collection": read(ROOT / "compact-collection-01/collection-receipt.json"),
        "limitations": ["One prompt/seed, direct E18-route zero persists across all four denoising steps; H27 inherited history is retained.",
            "Dependent repeated/carry fields, not independent trials or universal coverage.",
            "Shared original numerical primitives; no independent parser/RNG/CUDA arithmetic or current-cohort native decoder macro-by-macro claim.",
            "First-stream appearance and zero forecast error do not establish a semantic object dictionary."],
        "new_neural_calls": 0, "raw_arrays_downloaded": False}
    target = ROOT / "actual-saved-report-verification-v1.json"
    with target.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt_sha256": sha(target), "qwen_and_denoiser": receipt["qwen_and_denoiser"],
                      "decoder_terms": receipt["decoder_three_term_comparisons"], "signed_response_errors": receipt["signed_response_errors"],
                      "pixels": receipt["decoder_responses"]["pixels"], "first_changes": earliest,
                      "latent_commits": commits, "actual_cli_queries": len(queries)}, indent=2))


if __name__ == "__main__":
    main()

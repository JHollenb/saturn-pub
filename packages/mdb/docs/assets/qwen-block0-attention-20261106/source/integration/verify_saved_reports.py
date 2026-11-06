"""Authenticate three actual leases, recount saved results and exercise MDB CLI."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from mdb.job_support import scheduler_api

ROOT = Path(__file__).resolve().parent
PACK = ROOT / "compact-collection-01/extracted/binding-minimal-result"
BRIDGE = ROOT / "bridge-collection-01/extracted/binding-minimal-result"
OUT = ROOT / "saved-report-verification-v1"


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
    if (status != 200 or job["job_id"] != job_id or job["state"] != "succeeded"
            or result.get("returncode") != 0 or result.get("status") != "ok"):
        raise ValueError("Exact successful lease differs")
    for envelope in (job["config"], job["submitted_request"]["config"]):
        for key, value in request.items():
            if envelope.get(key) != value:
                raise ValueError("Actual submitted request differs: " + key)
    if job["config"]["retry_on_kill"] is not False:
        raise ValueError("Actual retry policy differs")
    if result["diagnostics"]["command"] != read(run / "admission-command.json")["command"]:
        raise ValueError("Executed command differs")
    reservation = job["submitted_request"]["reservation"]
    if any(reservation[key] != request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads")):
        raise ValueError("Actual reservation differs")
    if bool(job["submitted_request"]["needs"].get("cuda", False)) != gpu:
        raise ValueError("Declared device differs")
    digests = [job["payload_custody"][lane]["sha256"] for lane in ("declared", "sealed", "executed")]
    digests.append(result["executed_payload"]["sha256"])
    if len(set(digests)) != 1:
        raise ValueError("Actual payload custody differs")
    (OUT / (job_id + ".job.json")).write_bytes(raw)
    status, log, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = result["artifacts"]["job_log"]
    if status != 200 or len(log) != expected["bytes"] or hashlib.sha256(log).hexdigest() != expected["sha256"]:
        raise ValueError("Complete worker log differs")
    (OUT / (job_id + ".worker.log")).write_bytes(log)
    return {"job_id": job_id, "state": job["state"], "request_sha256": request["request_sha256"],
            "job_sha256": hashlib.sha256(raw).hexdigest(), "worker_log_sha256": hashlib.sha256(log).hexdigest(),
            "executed_payload_sha256": digests[0], "reservation": reservation,
            **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")}}


def summarize(rows):
    return {"field_comparisons": len(rows), "scalar_positions": sum(row["elements"] for row in rows),
            "bitwise_exact_fields": sum(row["bitwise_exact"] for row in rows),
            "changed_values": sum(row["changed_values"] for row in rows),
            "maxabs": max(row["maxabs"] for row in rows), "max_rms": max(row["rms"] for row in rows)}


def scalar_reduction_differences(value):
    if isinstance(value, dict):
        if "saved" in value and "recomputed" in value and "signed_difference" in value:
            return [value]
        return [row for child in value.values() for row in scalar_reduction_differences(child)]
    if isinstance(value, list):
        return [row for child in value for row in scalar_reduction_differences(child)]
    return []


def main():
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    OUT.mkdir(exist_ok=False)
    for directory, expected in ((PACK, "1b306fe32051e63b40961151fab246475f2c8fc69ebbcfad212fabd5dba9ac48"),
                                (BRIDGE, "4874c486decd965c1f756a7c62d4eafb380262141959dff9d7f4f46debdcd4a2")):
        if sha(directory / "files.json") != expected:
            raise ValueError("Collected compact manifest differs")
        for name, digest in read(directory / "files.json").items():
            if sha(directory / name) != digest:
                raise ValueError("Collected report differs: " + name)
    jobs = {name: authenticate_job(ROOT / run, job_id, gpu) for name, run, job_id, gpu in (
        ("native", "run-01", "job-0a3c41fce1d0", True),
        ("full_saved_audit", "audit-01", "job-2d91d9d5163a", False),
        ("signed_vector_bridge", "bridge-01", "job-c12bda14d64e", False))}
    audit, bridge = read(PACK / "audit.json"), read(BRIDGE / "bridge.json")
    matched = [row for case in audit["matched_functional_vs_native"].values() for row in case.values()]
    replay = [row for case in audit["historical_replay"].values() for lane in ("forecast", "native") for row in case[lane].values()]
    exact_bridges = {name: {key: row[key] for key in (
        "A_vs_native_short_shared_prefix", "B_vs_native_long_shared_prefix", "C_vs_native_long_shared_prefix",
        "signed_vector_bridge_shared_prefix", "C_minus_B_complete", "first_vector_sha256", "historical_vector_sha256")}
        for name, row in bridge["ports"].items()}
    reductions = scalar_reduction_differences(audit["saved_vs_recomputed_reduction_differences"])
    profiles = [{"label": row["label"], "status": row["profiler"]["status"],
                 "dispatch_events": [event for event in row["profiler"].get("events", []) if "scaled_dot_product" in event or "flash" in event.lower() or "fmha" in event.lower()],
                 "is_causal": row["is_causal"], "enable_gqa": row["enable_gqa"],
                 "explicit_mask": row["mask"] is not None}
                for row in read(PACK / "source-sdpa.json")]
    queries = []
    catalog = PACK / "catalog/equations"
    for path in sorted(catalog.glob("*.json")):
        record = read(path)
        context = record["tested_context"]
        variants = (("exact", context, 1),
                    ("changed-clock", {**context, "clock": {"component": "qwen", "block": 1, "port": "H1"}}, 0),
                    ("changed-bank", {**context, "fixed_triangle_bank": "short"}, 0))
        for kind, query, expected in variants:
            name = context["occurrence"] + "." + kind
            context_path = OUT / (name + ".context.json")
            context_path.write_text(json.dumps(query, indent=2, sort_keys=True) + "\n")
            command = [sys.executable, "-I", "-B", "-m", "mdb", "equations", "find", "--catalog", str(catalog),
                       "--alias", record["alias"], "--applicable-to", str(context_path)]
            process = subprocess.run(command, cwd=OUT, capture_output=True, text=True, check=True)
            matches = json.loads(process.stdout)
            if len(matches) != expected or matches and matches[0]["fingerprint"] != record["fingerprint"]:
                raise ValueError("Installed MDB applicability search differs")
            output = OUT / (name + ".cli.json")
            output.write_text(process.stdout)
            queries.append({"occurrence": context["occurrence"], "kind": kind, "results": len(matches),
                            "command": command, "context_sha256": sha(context_path), "output_sha256": sha(output)})
    result = {
        "schema": "mdb-qwen-block0-actual-saved-report-verification-v1", "jobs": jobs,
        "audit_sha256": sha(PACK / "audit.json"), "signed_bridge_sha256": sha(BRIDGE / "bridge.json"),
        "matched_full_fields": summarize(matched), "historical_replays": summarize(replay),
        "shared_prefix_cross_case": audit["pair_comparisons"]["native"], "exact_vector_bridges": exact_bridges,
        "signed_bridge_totals": summarize([row["signed_vector_bridge_shared_prefix"] for row in bridge["ports"].values()]),
        "complete_C_minus_B_totals": summarize([row["C_minus_B_complete"] for row in bridge["ports"].values()]),
        "head_coordinates": read(BRIDGE / "head-coordinates.json"), "token_changes": read(BRIDGE / "token-changes.json"),
        "actual_sdpa_profiles": profiles, "actual_census": audit["actual"],
        "source_summary_reduction_diagnostics": {"different_summaries": len(reductions),
            "max_absolute_signed_reduction_difference": max((abs(row["signed_difference"]) for row in reductions), default=0),
            "scope": "Independent FP64 scalar reductions; raw field equality and scientific effects are separately retained."},
        "catalog_cli": queries, "collections": [read(ROOT / name / "collection-receipt.json") for name in ("compact-collection-01", "bridge-collection-01")],
        "port_provenance": audit["port_provenance"], "dependent_repeated_coordinates": True, "raw_arrays_downloaded": False,
        "interpretation": "Known-source block0 replay plus fixed-bank primitive interventions. Explicit-mask versus causal-flag execution exactly bridges the original prefix drift through H1; grouped-versus-repeated causal representation has zero complete-bank effect in this specimen. Shared original primitives, not independent kernel arithmetic, a new image prediction, a factorial attribution, or a universal GQA law.",
    }
    target = ROOT / "actual-saved-report-verification-v1.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(target), "sha256": sha(target), "matched": result["matched_full_fields"],
                      "signed_bridge": result["signed_bridge_totals"], "complete_C_minus_B": result["complete_C_minus_B_totals"],
                      "catalog_commands": len(queries), "reduction_diagnostics": result["source_summary_reduction_diagnostics"]}, indent=2))


if __name__ == "__main__":
    main()

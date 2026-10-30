"""Offline verification and bottleneck analysis of saved discovery upgrade runs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def refs(value):
    if isinstance(value, dict):
        if set(value) == {"$value"}:
            return {value["$value"]}
        return set().union(*(refs(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(refs(item) for item in value))
    return set()


def profile(root):
    data = read(root / "profile.json")
    sync = [row for row in data["functions"] if "posix.fsync" in row["function"]]
    return {"profile_seconds": data["total_seconds"],
            "fsync_calls": sum(row["calls"] for row in sync),
            "fsync_self_seconds": sum(row["self_seconds"] for row in sync),
            "top_self_functions": data["functions"][:10]}


def package_proof(root):
    wheel = next((root / "payload").glob("mdb_runtime-*.whl"))
    distribution = importlib.metadata.distribution("mdb-runtime")
    source = Path(__file__).resolve().parents[1] / "src"
    rows = {}
    with zipfile.ZipFile(wheel) as archive:
        for path in sorted((source / "mdb").rglob("*.py")):
            name = path.relative_to(source).as_posix()
            raw = path.read_bytes()
            if raw != archive.read(name) or raw != distribution.locate_file(name).read_bytes():
                raise ValueError("source/installed/executed payload module differs: " + name)
            rows[name] = sha(path)
    return {"payload_wheel_sha256": sha(wheel), "source_installed_payload_equal": True,
            "module_sha256": rows}


def analyze(root, baseline):
    from mdb.discovery_catalog import StructuralCatalog
    from mdb.forward_symbols import verify_trace

    report = read(root / "report.json")
    job = read(root / "job.json")
    request = read(root / "request.json")
    if (report["job_id"] != job["job_id"] or report["request_sha256"] != request["request_sha256"]
            or report["mechanics_status"] != "valid" or job["state"] != "succeeded"):
        raise ValueError("final run identity/status differs")
    data = root / "extracted/upgrade-result"
    client = root.parent / (root.name + "-client")
    from mdb.discovery_script import DiscoveryJournal

    journal_rows = {name: len(DiscoveryJournal(path).verify()) for name, path in {
        "remote": client / "cli/journal", "resumed": data / "resumed/journal"}.items()}
    retry = read(client / "retry-proof.json")
    cli = read(client / "cli/result.json")
    if retry["first"] != retry["repeat"] or retry["changed_content"]["ok"] is not False or not cli["ok"]:
        raise ValueError("remote retry or CLI result differs")
    batch = read(data / "batch.json")
    rows = read(data / "variants.json")["rows"]
    trace = read(data / "native-trace.json")
    verify_trace(trace)
    catalog = StructuralCatalog(trace)
    law = catalog.resolve("module:post.1")
    if law != report["durable"]["law"]:
        raise ValueError("catalog is not the qualified law")
    tainted = set(law["outputs"])
    descendants = []
    for event in trace["events"]:
        if tainted.intersection(event["inputs"]):
            tainted.update(event["outputs"])
            descendants.append(event["sequence"])
    structural_heads = sorted(key for key, value in trace["outputs"].items() if tainted.intersection(refs(value)))
    groups = {}
    for row in rows:
        groups.setdefault(row["binding"]["lane_id"], []).append(row)
    changes, consumer_union = [], set()
    all_heads = set(trace["outputs"])
    for lane, variants in groups.items():
        native = next(row for row in variants if row["dose"] == 1)
        for row in variants:
            changed = sorted(key for key in row["heads"]
                             if row["heads"][key]["sha256"] != native["heads"][key]["sha256"])
            consumer_union.update(changed)
            scalar_flags = row["raw_heads"]["flags"][0]
            native_flags = native["raw_heads"]["flags"][0]
            changes.append({"lane_id": lane, "dose": row["dose"], "changed_heads": changed,
                "registers_equal_native": row["predicted_register_words"] == native["predicted_register_words"],
                "flags_equal_native": row["flags"] == native["flags"],
                "length_equal_native": row["predicted_length"] == native["predicted_length"],
                "effects_equal_native": (row["store"], row["syscall"]) == (native["store"], native["syscall"]),
                "flag_max_abs_effect": max(abs(a - b) for a, b in zip(scalar_flags, native_flags)),
                "flag_min_sign_margin": row["minimum_sign_margins"]["flags"],
                "predicted_RDI": int(row["predicted_register_words"][7], 16)})
    if not consumer_union.issubset(structural_heads):
        raise ValueError("observed effects violate source-bound descendant graph")
    head_panels = [*batch["all_head_controls"], *batch["candidate_all_head_controls"],
                   *batch["reordered_all_head_controls"], *batch["parked_all_head_controls"],
                   *batch["refill_all_head_controls"]]
    old_profile = profile(baseline / "extracted/upgrade-result")
    new_profile = profile(data)
    phases = {row["phase"]: row for row in report["phases"]}
    old_stats = read(baseline / "extracted/upgrade-result/profile.json")
    old_save = next(row["cumulative_seconds"] for row in old_stats["functions"]
                    if row["function"] == "save_session_frontier")
    return {"schema": "mdb-discovery-upgrades-analysis-v1", "job_id": report["job_id"],
        "claim_scope": "bounded CPU R3 mechanics; exploratory computation variants",
        "source": {"analyzer_sha256": sha(Path(__file__)), "report_sha256": sha(root / "report.json"),
                   "request_sha256": report["request_sha256"], "worker_sha256": request["source_hashes"]["discovery_upgrades.py"],
                   "trace_fingerprint": trace["fingerprint"], "program_id": catalog.program_id,
                   "baseline_job": read(baseline / "report.json")["job_id"],
                   "baseline_profile_sha256": sha(baseline / "extracted/upgrade-result/profile.json")},
        "package_proof": package_proof(root),
        "client": {"commands_executed": cli["commands_executed"],
            "same_id_same_content_exact_response": True, "changed_content_refused": True,
            "verified_journal_records": journal_rows, "retry_proof_sha256": sha(client / "retry-proof.json"),
            "cli_result_sha256": sha(client / "cli/result.json")},
        "physical_model_forwards": report["physical_model_forwards"],
        "batch": {"rounds": batch["rounds"], "work": batch["adapter_work"],
                  "maximum_all_head_abs_difference": max(head["max_abs"] for row in head_panels for head in row["all_heads"].values()),
                  "numerical_profile": batch["profile"]["numerical_contract"],
                  "decoded_B1_B4_parity": "not established by retained numeric summaries"},
        "durable": report["durable"], "remote": report["remote"],
        "bottlenecks": {"old_profile": old_profile, "new_profile": new_profile,
            "old_save_session_seconds": old_save, "new_save_session_seconds": phases["durable_save"]["seconds"],
            "observed_save_ratio": old_save / phases["durable_save"]["seconds"],
            "comparison_scope": "one cold before/after observation; earlier run later failed in CLI dispatch",
            "phases": report["phases"], "job_elapsed_seconds": job["result"]["elapsed_s"],
            "peak_process_tree_rss_mb": job["result"]["peak_rss_mb"]},
        "artifact_publication": {"status": job["result"].get("artifacts", {}).get("status"),
            "collection": "worker log archive; SHA-256 verified before extraction",
            "archive_sha256": report["archive_sha256"]},
        "variants": {"rows": len(rows), "distinct_contexts": len(groups),
            "dose_levels": sorted({row["dose"] for row in rows}), "effects": changes,
            "structural_consumer_heads": structural_heads, "observed_changed_heads": sorted(consumer_union),
            "structural_and_observed_head_sets_agree": set(structural_heads) == consumer_union,
            "unchanged_heads": sorted(all_heads - consumer_union), "descendant_events": len(descendants),
            "interpretation": "consumer-incidence and continuous response; no semantic operation name established"},
        "limits": ["batching is full native screening; component replay remains scalar",
                   "durable restoration selects one branch and requires explicit factory/lease rebinding as applicable",
                   "native first-eager pause is not provided", "four contexts of one byte program; not broad model laws"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Retain the primary analyzer source before performing the analysis.
    snapshot = args.output.parent / (args.output.stem + ".source.py")
    with snapshot.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    result = analyze(args.root, args.baseline)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"job_id": result["job_id"], "output": str(args.output),
                      "save_ratio": result["bottlenecks"]["observed_save_ratio"]}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Verify and summarize the immutable discovery qualification artifacts offline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    from mdb.discovery_script import DiscoveryJournal

    report = json.loads((root / "report.json").read_text())
    job = json.loads((root / "job.json").read_text())
    request = json.loads((root / "request.json").read_text())
    output = root / "extracted/discovery-result"
    if (report["job_id"] != job["job_id"] or report["request_sha256"] != request["request_sha256"]
            or job["state"] != "succeeded" or report["mechanics_status"] != "valid"):
        raise ValueError("qualification job/request/status mismatch")
    records = DiscoveryJournal(output / "journal").verify()
    cli_records = DiscoveryJournal(output / "cli-result/journal").verify()
    stops = json.loads((output / "four-lane-stops.json").read_text())
    if len(stops["results"]) != 4:
        raise ValueError("missing resident lane results")
    heads = {"regs", "flags", "dest", "src", "width", "length", "syscall", "imm", "store", "store_addr", "store_value"}
    audit = report["edited_fresh_native_all_heads"][0]
    if set(audit) != heads or any(not value["exact"] or value["max_abs"] != 0 for value in audit.values()):
        raise ValueError("edited numerical audit is incomplete or differs")
    if set(report["restored_native_all_heads"]) != heads or not all(report["restored_native_all_heads"].values()):
        raise ValueError("restored numerical control is incomplete or differs")
    recipe = output / "root-recipe"
    recipe_manifest = json.loads((recipe / "capture.json").read_text())
    for name in ("trace.json", "roots.pt"):
        if sha(recipe / name) != recipe_manifest["files"][name]:
            raise ValueError("root recipe bytes changed")
    cli = json.loads((output / "cli-invocation.json").read_text())
    if cli["returncode"] != 0:
        raise ValueError("installed CLI execution failed")
    expected_script = Path(__file__).resolve().parents[1] / "examples/discovery/post1.mdb"
    # The executable lines, excluding explanatory comments, match the published example.
    executable = lambda text: [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if executable(expected_script.read_text()) != executable((output / "example.mdb").read_text()):
        raise ValueError("published example differs from the executed CLI script")
    return {"schema": "mdb-discovery-qualification-summary-v1", "job_id": report["job_id"],
            "mechanics_status": "valid", "scientific_claim": "debugger mechanics and bounded numerical parity",
            "checkpoint_sha256": report["checkpoint_sha256"], "backend": report["backend"],
            "worker_seconds": report["worker_seconds"], "peak_process_tree_rss_mb": job["result"]["peak_rss_mb"],
            "physical_forwards": report["total_physical_forwards"],
            "trace_captures": len(report["captures"]), "native_controls": len(report["native_controls_all_heads"]),
            "independent_native_audits": report["additional_full_forward_audits"],
            "four_lane_stops": [{"lane_id": row["lane"]["lane_id"], "pc": row["lane"]["pc"],
                                  "stop_causes": [cause["kind"] for cause in row["stop_causes"]]} for row in stops["results"]],
            "law": {key: report["law"][key] for key in ("site_id", "name", "module", "entry", "exit", "inputs", "outputs")},
            "update_pc": report["update_pc"], "native_all_head_audit_exact": True,
            "restored_native_all_head_exact": True, "branch_vs_native": report["branch_vs_native"],
            "predicted_RDI": [252, int(report["committed_first"]["register_words"][7], 16),
                               int(report["committed_second"]["register_words"][7], 16)],
            "predicted_offsets": [0, report["committed_first"]["byte_offset"], report["committed_second"]["byte_offset"]],
            "serving_inputs": "raw bytes and before-state only; no oracle state", "limits": report["limits"],
            "cli": report["installed_cli"], "main_controller_work": report["work"],
            "main_journal_records": len(records), "cli_journal_records": len(cli_records),
            "artifact_publication_status": job["result"].get("artifacts", {}).get("status"),
            "local_custody": "hash-verified archive recovered from worker log",
            "archive_sha256": report["archive_sha256"],
            "source": {"analyzer_sha256": sha(Path(__file__)),
                       "worker_sha256": request["source_hashes"]["discovery_isa.py"],
                       "report_sha256": sha(root / "report.json"),
                       "request_sha256": report["request_sha256"],
                       "mdb_payload_wheel_sha256": sha(next((root / "payload").glob("mdb_runtime-*.whl")))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"job_id": result["job_id"], "mechanics_status": result["mechanics_status"],
                      "physical_forwards": result["physical_forwards"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

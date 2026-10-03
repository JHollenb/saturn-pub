"""Compact, validated Git record; large evidence stays in the custody store."""
from hashlib import sha256
import json
from pathlib import Path
import sys


def main():
    job = Path(sys.argv[1]).resolve()
    report = json.loads((job / "report.json").read_text())
    scheduler = json.loads((job / "job.json").read_text())
    archive = json.loads((job / "archive.json").read_text())
    audit = json.loads((job / "posttermination-verification.json").read_text())
    request = json.loads((job / "request.json").read_text())
    if scheduler["state"] != "succeeded" or report["mechanics_status"] != "valid":
        raise RuntimeError("family proof is not completed")
    if report["source_hashes"] != request["source_hashes"] or report["request_sha256"] != request["request_sha256"]:
        raise RuntimeError("report source/request seal differs")
    if not audit["rows"] or any(r["verification"]["mechanics_status"] != "valid" for r in audit["rows"]):
        raise RuntimeError("independent archive closure is incomplete")
    if audit.get("incomplete_stores") or not report["fresh"]:
        raise RuntimeError("unselected stores or missing fresh-process evidence")
    memory = report["cases"][0]["memory"]
    record = {"schema": "mdb-family-proof-summary-v2", "family": report["family"],
        "job_id": scheduler["job_id"], "scheduler_state": scheduler["state"],
        "report_sha256": sha256((job / "report.json").read_bytes()).hexdigest(),
        "request_sha256": report["request_sha256"],
        "source_map_sha256": sha256(json.dumps(report["source_hashes"], sort_keys=True).encode()).hexdigest(),
        "archive": archive, "mechanics": report["mechanics"],
        "elapsed_seconds": report["elapsed_seconds"], "identity": report["cases"][0]["identity"],
        "source_store_removed": report["original_first_store_removed"],
        "fresh_exact": all(r["exact"] for r in report["fresh"]),
        "memory": {k: memory[k] for k in ("shared", "panel_evictions", "panel_hydrations",
            "forced_budgets_bytes", "resident_unions_bytes", "private_replays", "earlier_restore_exact")},
        "cases": [{k: c[k] for k in ("index", "prompt", "seed", "native_exact", "stepped_exact", "noop_exact",
                    "prepare_seconds", "native_seconds", "bench_seconds")} for c in report["cases"]],
        "independent_archive": audit["rows"], "debugger_status": report["debugger_status"],
        "boundary": "Qualified checkpoint/numerical lane and bounded native-consumer mechanics. Discovery signed observations retained; discovery quality and deployed broker qualification are separate."}
    output = Path(sys.argv[2]).resolve()
    if output.exists(): raise RuntimeError("compact record already exists")
    output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"family": record["family"], "cases": len(record["cases"]), "output": str(output)}))


if __name__ == "__main__": main()

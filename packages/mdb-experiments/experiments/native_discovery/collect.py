"""Collect the exact admitted job's logs, receipts and result marker (no lineage substitution)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdb_lanes.job_support import guarded_job_receipt, terminal_job
from mdb_lanes.native_discovery import save_json


def collect(stage):
    job = json.loads((stage / "submission.json").read_text())["job_id"]
    _, metadata, logs = terminal_job(job)
    print(json.dumps({"job_id": job, "state": metadata["state"]}))
    if metadata["state"] not in {"succeeded", "failed", "timeout", "cancelled", "lost", "killed_vram", "killed_ram"}:
        return
    save_json(stage / "job.json", metadata)
    (stage / "worker.log").write_text(logs)
    save_json(stage / "admission-receipt.json", guarded_job_receipt(job))
    marker = "MDB_NATIVE_DISCOVERY_RESULT="
    records = [json.loads(line[len(marker):]) for line in logs.splitlines() if line.startswith(marker)]
    if records:
        if len(records) != 1:
            raise ValueError("more than one result for a finite job")
        save_json(stage / "result.json", records[0])
        print(json.dumps({"artifact": records[0]["artifact"],
                          "elapsed_seconds": records[0]["summary"]["elapsed_seconds"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", type=Path)
    collect(p.parse_args().stage)

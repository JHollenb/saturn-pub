"""Collect one completed finite MDB CLI job without following retry children."""
import argparse
import json
from pathlib import Path

from mdb.job_support import terminal_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id")
    parser.add_argument("--out", required=True, type=Path, help="new local result directory")
    args = parser.parse_args()
    job_id, job, logs = terminal_job(args.job_id, wait=False)
    if job["state"] not in {"succeeded", "failed", "cancelled", "killed_ram", "killed_vram", "timeout", "lost"}:
        print(json.dumps({"job_id": job_id, "state": job["state"], "collected": False}))
        return 2
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    (args.out / "worker.log").write_text(logs)
    marker = "SATURN_BENCH_REPORT="
    report = next((json.loads(line[len(marker):]) for line in logs.splitlines()
                   if line.startswith(marker)), None)
    environment_marker = "MDB_EXECUTION_ENVIRONMENT="
    environment = next((json.loads(line[len(environment_marker):]) for line in logs.splitlines()
                        if line.startswith(environment_marker)), None)
    if environment is not None:
        (args.out / "execution-environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if report is not None:
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"job_id": job_id, "state": job["state"], "report": bool(report),
                      "out": str(args.out)}))
    return 0 if job["state"] == "succeeded" and report is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())

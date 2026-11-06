"""Capture exact scheduler state/logs while printing only finite job metadata."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["MRUN_URL"] = "http://192.168.68.68:9025"


def main():
    from mdb.job_support import scheduler_api

    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    job_id = json.loads((args.run / "submission.json").read_text())["job_id"]
    api = scheduler_api()
    _, raw, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    if isinstance(raw, str):
        raw = raw.encode()
    job = json.loads(raw)
    index = len(list(args.run.glob("job-observed-*.json")))
    (args.run / (f"job-observed-{index:02d}.json")).write_bytes(raw)
    result = job.get("result") or {}
    report = dict(job_id=job_id, state=job["state"],
        reservation=job.get("reservation"), admission_reservation=job.get("admission_reservation"),
        result={key: result.get(key) for key in ("status", "returncode", "elapsed_s", "peak_rss_mb", "peak_vram_mb")})
    if job["state"] in ("succeeded", "failed", "cancelled"):
        final = args.run / "job-completed.json"
        if not final.exists():
            final.write_bytes(raw)
        _, log, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
        if isinstance(log, str):
            log = log.encode()
        (args.run / "job.log").write_bytes(log)
        report["job_record_sha256"] = hashlib.sha256(raw).hexdigest()
        report["log_bytes"] = len(log)
        report["log_tail"] = log.decode(errors="replace")[-1800:]
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

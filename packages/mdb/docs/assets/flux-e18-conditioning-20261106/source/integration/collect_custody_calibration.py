"""Collect only authenticated CUDA synthetic-custody timing metadata."""
import hashlib
import json
import os
import runpy
import subprocess
from pathlib import Path

from mdb.job_support import scheduler_api

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "custody-calibration-02"
OUT = ROOT / "custody-calibration-collection-v1"
CODE = ROOT.parents[1] / "saturn-pub/packages/mdb/qualification"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"
    request = json.loads((RUN / "request.json").read_text())
    job_id = json.loads((RUN / "submission.json").read_text())["job_id"]
    api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    if status != 200:
        raise ValueError("Actual job missing")
    job = json.loads(raw)
    validator = CODE / "collect_flux_initial_noise.py"
    if digest(validator.read_bytes()) != "ce3930b71b58dab832c09b7d4b0c4e3eefb8bc6435a72e4605992d415be38765":
        raise ValueError("Qualified job validator changed")
    runpy.run_path(str(validator))["validate_job"](job, job_id, request)
    result = job["result"]
    if result["diagnostics"]["command"] != json.loads((RUN / "admission-command.json").read_text())["command"]:
        raise ValueError("Actual command differs")
    reservation = job["submitted_request"]["reservation"]
    if any(reservation[key] != request["resource_exception"][key] for key in ("ram_mb", "vram_mb", "cpu_threads")):
        raise ValueError("Actual reservation differs")
    if not job["submitted_request"]["needs"]["cuda"] or job["config"]["retry_on_kill"]:
        raise ValueError("Actual CUDA/retry admission differs")
    status, log, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    expected = result["artifacts"]["job_log"]
    if status != 200 or digest(log) != expected["sha256"] or len(log) != expected["bytes"]:
        raise ValueError("Actual full log differs")
    rows = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    if len(rows) != 1:
        raise ValueError("One published directory required")
    marker = rows[0]
    root = Path(request["config"]["output_scratch"]) / job_id / request["config"]["output_name"]
    if marker["job_id"] != job_id or marker["request_sha256"] != request["request_sha256"] or marker["output_root"] != str(root):
        raise ValueError("Directory source identity differs")
    script = "import json\nfrom pathlib import Path\np=Path(" + repr(str(root)) + ")\nout={}\nfor n in ('files.json','summary.json','report.json'):\n q=p/n\n if q.is_symlink() or q.stat().st_size>2097152: raise ValueError('Invalid compact member')\n out[n]=q.read_text()\nprint(json.dumps(out))\n"
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script,
                             text=True, capture_output=True, check=True, timeout=60)
    if len(process.stdout.encode()) > 8 << 20:
        raise ValueError("Compact transport bound exceeded")
    files = {name: body.encode() for name, body in json.loads(process.stdout).items()}
    if digest(files["files.json"]) != marker["files_manifest_sha256"] or digest(files["summary.json"]) != marker["summary_sha256"]:
        raise ValueError("Actual published compact digests differ")
    inventory = json.loads(files["files.json"])
    summary = json.loads(files["summary.json"])
    report = json.loads(files["report.json"])
    if inventory["job_id"] != job_id or inventory["request_sha256"] != request["request_sha256"] or summary["job_id"] != job_id:
        raise ValueError("Collected source IDs differ")
    for name in ("summary.json", "report.json"):
        if inventory["files"][name] != {"sha256": digest(files[name]), "bytes": len(files[name])}:
            raise ValueError("Collected member descriptor differs")
    if summary["report_sha256"] != digest(files["report.json"]) or report["device"] != "cuda" or not report["synthetic"]:
        raise ValueError("Synthetic CUDA report binding differs")
    OUT.mkdir(exist_ok=False)
    for name, body in files.items():
        (OUT / name).write_bytes(body)
    (OUT / "job.json").write_bytes(raw)
    (OUT / "worker.log").write_bytes(log)
    receipt = {"schema": "mdb-e18-custody-actual-collection-v1", "job_id": job_id,
               "request_sha256": request["request_sha256"], "executed_payload": result["executed_payload"],
               "native_arrays_downloaded": False, "synthetic_arrays_downloaded": False, "checkpoint_reads": 0,
               "collected_members": {name: {"bytes": len(body), "sha256": digest(body)} for name, body in files.items()},
               **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")},
               "report_summary": summary, "scope": "Synthetic public StateCut/LocalStore operations; excludes Session guards, Acts and neural arithmetic."}
    target = OUT / "collection-receipt.json"
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt_sha256": digest(target.read_bytes()), "job_id": job_id,
                      **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")},
                      "timings": summary["timings"], "build_seconds": summary["build_seconds"],
                      "store_equivalence": summary["exact_metadata_and_store_equivalence"]}, indent=2))


if __name__ == "__main__":
    main()

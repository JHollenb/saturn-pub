"""Read the successful job-bound raw inventory/summary without tensor transport."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit-01"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    config = json.loads((AUDIT / "plan.json").read_text())
    job = json.loads((AUDIT / "source-job.json").read_text())
    request = json.loads((ROOT / "run-01/request.json").read_text())
    log = (AUDIT / "source-worker.log").read_bytes()
    expected = job["result"]["artifacts"]["job_log"]
    if sha(log) != expected["sha256"] or len(log) != expected["bytes"]:
        raise ValueError("Authenticated native log differs")
    markers = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines() if line.startswith("MDB_DIRECTORY_READY ")]
    if len(markers) != 1:
        raise ValueError("One raw publication required")
    marker = markers[0]
    if (job["state"] != "succeeded" or marker["job_id"] != job["job_id"]
            or marker["request_sha256"] != request["request_sha256"]
            or marker["output_root"] != config["native_root"]
            or marker["files_manifest_sha256"] != config["native_manifest_sha256"]):
        raise ValueError("Native directory source differs")
    script = "import json\nfrom pathlib import Path\nr=Path(" + repr(config["native_root"]) + ")\nfiles={}\nfor name in ('files.json','summary.json','timeline.jsonl','native-contract.json'):\n p=r/name\n if p.is_symlink() or p.stat().st_size>4194304: raise ValueError('Bounded metadata required')\n files[name]=p.read_text()\nprint(json.dumps(files))\n"
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "beast", "python3", "-I", "-"], input=script, text=True,
                             capture_output=True, check=True, timeout=60)
    if len(process.stdout.encode()) > 16 << 20:
        raise ValueError("Metadata transport oversized")
    files = {name: value.encode() for name, value in json.loads(process.stdout).items()}
    if sha(files["files.json"]) != marker["files_manifest_sha256"] or sha(files["summary.json"]) != marker["summary_sha256"]:
        raise ValueError("Raw wrapper/summary differs")
    inventory = json.loads(files["files.json"])
    for name, body in files.items():
        if name != "files.json" and inventory["files"][name] != {"bytes": len(body), "sha256": sha(body)}:
            raise ValueError("Raw metadata member differs: " + name)
    out = ROOT / "native-metadata-collection-v1"
    out.mkdir(exist_ok=False)
    for name, body in files.items():
        (out / name).write_bytes(body)
    events = [json.loads(line) for line in files["timeline.jsonl"].decode().splitlines()]
    summary = json.loads(files["summary.json"])
    result = {"schema": "mdb-e18-native-metadata-collection-v1", "job_id": job["job_id"],
              "request_sha256": request["request_sha256"], "raw_manifest_sha256": sha(files["files.json"]),
              "raw_members": len(inventory["files"]), "raw_bytes": inventory["expanded_bytes"],
              "wrapper_bytes_additional": len(files["files.json"]), "counts": summary["counts"],
              "forecast_seal": summary["forecast_seal"],
              "timeline": events,
              "executed_payload": job["result"]["executed_payload"],
              **{key: job["result"][key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")},
              "raw_tensors_downloaded": False, "numeric_prediction_outcomes": "Pending independent saved-only full-array audit"}
    target = ROOT / "native-metadata-verification-v1.json"
    with target.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt_sha256": sha(target.read_bytes()), "job_id": result["job_id"],
                      "raw_members": result["raw_members"], "raw_bytes": result["raw_bytes"],
                      "seal": summary["forecast_seal"], "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()

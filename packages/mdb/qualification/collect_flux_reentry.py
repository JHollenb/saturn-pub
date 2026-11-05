"""Planned authenticated SSH collection; no archive or model execution.

Only a succeeded, request-bound Beast job is accepted. The job's authenticated
scheduler record supplies its working directory. Independent remote inventories
bracket an allowlisted rsync, then every local regular file is hashed again.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

MAX_TOTAL = 8 << 30
MAX_MEMBER = 1 << 30
MAX_FILES = 20000
RESULT = "flux-reentry-result"
METHOD = "planned-authenticated-ssh-manifest"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(name):
    if not isinstance(name, str):
        raise ValueError("unsafe result member path")
    path = PurePosixPath(name)
    if (not name or path.is_absolute()
            or path.as_posix() != name or ".." in path.parts or "\\" in name
            or any(ord(c) < 32 for c in name) or name == "."):
        raise ValueError("unsafe result member path")
    return name


def inventory(root, *, max_total=MAX_TOTAL, max_member=MAX_MEMBER, max_files=MAX_FILES):
    root = Path(root)
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError("symlink result root/ancestor")
    if not root.is_dir():
        raise ValueError("missing result directory")
    files, total = {}, 0
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            path = Path(directory) / name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("non-directory result child")
        for name in names:
            path = Path(directory) / name
            key = relative(path.relative_to(root).as_posix())
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("nonregular result member")
            total += before.st_size
            if before.st_size > max_member or total > max_total or len(files) >= max_files:
                raise ValueError("result exceeds declared collection bounds")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError("result file changed before read")
                digest = hashlib.sha256()
                read_bytes = 0
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    read_bytes += len(chunk)
                    if read_bytes > before.st_size or read_bytes > max_member:
                        raise ValueError("result file grew beyond declared bounds while hashing")
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
            if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("result file changed while hashing")
            files[key] = {"sha256": digest.hexdigest(), "bytes": before.st_size, "filetype": "regular"}
    return {"files": dict(sorted(files.items())), "file_count": len(files), "expanded_bytes": total}


def verify_manifest(root, actual, job_id, request):
    root = Path(root)
    manifest = json.loads((root / "remote-manifest.json").read_text())
    expected = {k: v for k, v in actual["files"].items() if k != "remote-manifest.json"}
    if (set(manifest) != {"schema", "job_id", "request_sha256", "files"}
            or manifest["schema"] != "mdb-step-boundary-remote-manifest-v1"
            or manifest["job_id"] != job_id or manifest["request_sha256"] != request["request_sha256"]
            or manifest["files"] != expected):
        raise ValueError("producer manifest does not cover exact result")
    for name in ("source.json", "summary.json"):
        value = json.loads((root / name).read_text())
        if value["job_id"] != job_id or value["request_sha256"] != request["request_sha256"]:
            raise ValueError("producer source/summary job/request mismatch")
    source = json.loads((root / "source.json").read_text())
    if source["source_hashes"] != request["source_hashes"]:
        raise ValueError("producer source cohort differs")
    if json.loads((root / "plan.json").read_text()) != request["config"]:
        raise ValueError("producer plan differs")
    if "transaction/seal.json" not in actual["files"]:
        raise ValueError("producer transaction seal absent")
    return manifest


def validate_job(job, job_id, request):
    if not re.fullmatch(r"job-[a-zA-Z0-9_-]+", job_id):
        raise ValueError("invalid job id")
    result = job.get("result") or {}
    if (job.get("job_id") != job_id or job.get("state") != "succeeded"
            or job.get("assigned_host") != "beast" or result.get("returncode") != 0
            or result.get("status") != "ok"):
        raise ValueError("collection requires succeeded authenticated Beast job")
    for envelope in (job.get("config", {}), job.get("submitted_request", {}).get("config", {})):
        if any(envelope.get(k) != v for k, v in request.items()):
            raise ValueError("scheduler request differs from staged request")
    custody = job.get("payload_custody", {})
    executed = result.get("executed_payload", {})
    for key in ("declared", "sealed", "executed"):
        if any(custody.get(key, {}).get(k) != executed.get(k) for k in ("sha256", "size_bytes")):
            raise ValueError("executed payload custody differs")
    if (not re.fullmatch(r"[0-9a-f]{64}", executed.get("sha256", ""))
            or type(executed.get("size_bytes")) is not int or executed["size_bytes"] <= 0
            or executed.get("host") != "beast"):
        raise ValueError("executed payload identity absent")
    cwd = result.get("diagnostics", {}).get("cwd", "")
    if cwd != "/mnt/hdd8tb/mrun-work/" + job_id:
        raise ValueError("unexpected authenticated job working directory")
    return cwd + "/" + RESULT


def preserve(path, raw):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("refuse to replace existing collection evidence: " + path.name)
        return
    with path.open("xb") as stream:
        stream.write(raw)


def save(path, value):
    preserve(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def remote_script(remote_path):
    # No remote path is interpolated into a shell command; it is a Python literal.
    header = "import os,stat,json,hashlib\nfrom pathlib import Path,PurePosixPath\n"
    header += f"MAX_TOTAL={MAX_TOTAL}\nMAX_MEMBER={MAX_MEMBER}\nMAX_FILES={MAX_FILES}\n"
    return header + inspect.getsource(relative) + "\n" + inspect.getsource(inventory) + (
        "\nprint(json.dumps(inventory(" + repr(remote_path) + "),sort_keys=True))\n")


def collect(run_dir, job_id, *, api=None, runner=None):
    """Collect exactly one completed result; optional dependencies enable tiny tests."""
    if not isinstance(job_id, str) or not re.fullmatch(r"job-[a-zA-Z0-9_-]+", job_id):
        raise ValueError("invalid job id")
    runner = runner or subprocess.run
    run = Path(run_dir).resolve()
    request = json.loads((run / "request.json").read_text())
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("staged request digest differs")
    endpoint = os.environ.setdefault("MRUN_URL", request["scheduler_url"])
    if endpoint != request["scheduler_url"]:
        raise ValueError("scheduler endpoint differs from sealed request")
    if api is None:
        from mdb.job_support import scheduler_api
        api = scheduler_api()
    config = request["config"]
    limits = config["transport_limits"]
    if (config.get("transport_method") != METHOD or limits.get("max_expanded_bytes") != MAX_TOTAL
            or limits.get("max_member_bytes") != MAX_MEMBER):
        raise ValueError("planned collection contract differs")
    for name, digest in request["source_hashes"].items():
        relative(name)
        if sha(run / "payload" / name) != digest:
            raise ValueError("staged source bytes differ")
    status, raw, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    if status != 200:
        raise ValueError("scheduler job unavailable")
    job = json.loads(raw)
    remote = validate_job(job, job_id, request)
    preserve(run / "collection-job.json", raw)
    status, events, _ = api.request("GET", "/api/jobs/" + job_id + "/events", timeout_s=30)
    if status != 200 or len(events) > 8 << 20:
        raise ValueError("scheduler events unavailable or oversized")
    preserve(run / "collection-events.json", events)
    status, logs, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    if status != 200 or len(logs) > 16 << 20:
        raise ValueError("compact scheduler log unavailable or oversized")
    expected_log = job.get("result", {}).get("artifacts", {}).get("job_log", {})
    if expected_log and (expected_log.get("bytes") != len(logs)
                         or expected_log.get("sha256") != hashlib.sha256(logs).hexdigest()):
        raise ValueError("scheduler log byte/hash binding differs")
    preserve(run / "collection-worker.log", logs)
    command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "beast", "python3", "-"]
    script = remote_script(remote)
    def scan():
        response = runner(command, input=script, text=True, capture_output=True, check=True, timeout=600)
        if len(response.stdout) > 16 << 20:
            raise ValueError("remote inventory response oversized")
        value = json.loads(response.stdout)
        if (value["file_count"] != len(value["files"]) or value["file_count"] > MAX_FILES
                or value["expanded_bytes"] != sum(v["bytes"] for v in value["files"].values())
                or value["expanded_bytes"] > MAX_TOTAL):
            raise ValueError("remote inventory bounds differ")
        for name, item in value["files"].items():
            relative(name)
            if (set(item) != {"sha256", "bytes", "filetype"} or item["filetype"] != "regular"
                    or type(item["bytes"]) is not int or not 0 <= item["bytes"] <= MAX_MEMBER
                    or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
                raise ValueError("invalid remote member descriptor")
        return value
    before = scan()
    save(run / "collection-remote-before.json", before)
    if shutil.disk_usage(run).free < before["expanded_bytes"] + (128 << 20):
        raise ValueError("insufficient local collection space")
    destination = run / "extracted" / RESULT
    destination.mkdir(parents=True, exist_ok=False)
    allowlist = run / "collection-file-list.bin"
    preserve(allowlist, b"".join(name.encode() + b"\0" for name in before["files"]))
    copied = runner(["rsync", "--recursive", "--times", "--protect-args", "--no-links", "--no-devices", "--no-specials",
        "--from0", "--files-from=" + str(allowlist), "--max-size=" + str(MAX_MEMBER), "--timeout=120",
        "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes", "beast:" + remote + "/", str(destination) + "/"],
        text=True, capture_output=True, check=True, timeout=1800)
    save(run / "collection-copy.json", {"returncode": copied.returncode, "stdout": copied.stdout, "stderr": copied.stderr})
    after = scan()
    save(run / "collection-remote-after.json", after)
    if before != after:
        raise ValueError("remote evidence changed during transfer")
    local = inventory(destination)
    if local != before:
        raise ValueError("local evidence differs from authenticated inventory")
    verify_manifest(destination, local, job_id, request)
    save(run / "collection-local-inventory.json", local)
    receipt = {"schema": "mdb-reentry-ssh-collection-v1", "collection_mode": METHOD,
        "job_id": job_id, "scheduler_status": job["state"], "request_sha256": request["request_sha256"],
        "remote_path": remote, "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"],
        "archive_sha256": None, "model_calls_during_collection": 0,
        "executed_payload_sha256": job["result"]["executed_payload"]["sha256"],
        "manifest_sha256": sha(destination / "remote-manifest.json"),
        "summary_sha256": sha(destination / "summary.json"), "seal_sha256": sha(destination / "transaction/seal.json"),
        "limits": {"max_expanded_bytes": MAX_TOTAL, "max_member_bytes": MAX_MEMBER, "max_files": MAX_FILES},
        "collector_source_sha256": sha(Path(__file__)), "remote_inventory_source_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "bindings": {name: sha(run / name) for name in ("request.json", "collection-job.json", "collection-events.json",
            "collection-worker.log", "collection-remote-before.json", "collection-remote-after.json",
            "collection-local-inventory.json", "collection-copy.json", "collection-file-list.bin")},
        "scope": "Predeclared authenticated SSH normal collection, not archive recovery. Actual scheduler bytes retained; complete producer manifest and full before/after/local file identities agree. Scientific audit remains separate."}
    save(run / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("job_id")
    args = parser.parse_args()
    print(json.dumps(collect(args.run_dir, args.job_id)))


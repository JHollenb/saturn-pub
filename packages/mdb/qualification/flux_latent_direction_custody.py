"""Authenticated custody for the failed latent-direction native campaign.

The native worker completed its scientific work and failed while asking the
producer helper for a remote manifest.  This module is deliberately a
collector, not a producer: it never writes to the Beast result directory,
never launches a model, and never changes the scheduler's failed status.  A
successful job is delegated to the existing native producer unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import runpy
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
LEGACY_COLLECTOR_SHA256 = "ce3930b71b58dab832c09b7d4b0c4e3eefb8bc6435a72e4605992d415be38765"
RESULT = "latent-direction-native-result"
SCHEMA = "mdb-latent-direction-native-recovery-collection-v1"
SOURCE_MANIFEST_SCHEMA = SCHEMA
KNOWN_FAILURE = "post-exec-finalization-keyerror-finalize_manifest"
MAX_TOTAL, MAX_MEMBER, MAX_FILES = 4 << 30, 1 << 30, 20000
CELLS = ("c0-o1", "c1-o0")
BRANCHES = (
    "c0-o1.d-c0-o1", "c0-o1.d-c1-o0", "c0-o1.sham",
    "c1-o0.d-c0-o1", "c1-o0.d-c1-o0", "c1-o0.sham",
)

_legacy = HERE / "collect_flux_initial_noise.py"
if hashlib.sha256(_legacy.read_bytes()).hexdigest() != LEGACY_COLLECTOR_SHA256:
    raise ValueError("qualified generic collection primitive source differs")
_C = runpy.run_path(str(_legacy))


def sha(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: os.PathLike[str] | str):
    return json.loads(Path(path).read_text())


def _json(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def save(path: os.PathLike[str] | str, value) -> None:
    path = Path(path)
    if path.exists():
        if path.read_bytes() != _json(value):
            raise ValueError("refuse to replace immutable custody evidence: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_json(value))


def preserve(path: os.PathLike[str] | str, raw: bytes) -> None:
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("refuse to replace immutable custody evidence: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def relative(name: str) -> str:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ValueError("unsafe result member path")
    value = PurePosixPath(name)
    if value.is_absolute() or value.as_posix() != name or any(part in {".", ".."} for part in value.parts):
        raise ValueError("unsafe result member path")
    return name


def inventory(root: os.PathLike[str] | str, *, max_total: int = MAX_TOTAL,
              max_member: int = MAX_MEMBER, max_files: int = MAX_FILES) -> dict:
    """Hash a regular-file tree without following links or replacing bytes."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("result root must be a real directory")
    for parent in (root, *root.parents):
        if parent.is_symlink():
            raise ValueError("result root/ancestor is a symlink")
    files, total = {}, 0
    for directory, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            if (Path(directory) / name).is_symlink():
                raise ValueError("result contains a symlink directory")
        for name in sorted(names):
            path = Path(directory) / name
            rel = relative(path.relative_to(root).as_posix())
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("result contains a nonregular member")
            if before.st_size > max_member or len(files) >= max_files:
                raise ValueError("result exceeds declared collection bounds")
            total += before.st_size
            if total > max_total:
                raise ValueError("result exceeds declared expanded-byte bound")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError("result file changed before hashing")
                digest = hashlib.sha256()
                count = 0
                for block in iter(lambda: stream.read(1 << 20), b""):
                    count += len(block)
                    if count > before.st_size or count > max_member:
                        raise ValueError("result file grew while hashing")
                    digest.update(block)
                after = os.fstat(stream.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("result file changed while hashing")
            files[rel] = {"sha256": digest.hexdigest(), "bytes": before.st_size, "filetype": "regular"}
    return {"files": dict(sorted(files.items())), "file_count": len(files), "expanded_bytes": total}


def _request_digest(request: dict) -> None:
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request.get("request_sha256"):
        raise ValueError("request self digest differs")


def _job_common(job: dict, job_id: str, request: dict, *, failed: bool) -> str:
    if not isinstance(job_id, str) or not re.fullmatch(r"job-[A-Za-z0-9_-]+", job_id):
        raise ValueError("invalid job id")
    result = job.get("result") or {}
    expected_state = "failed" if failed else "succeeded"
    if (job.get("job_id") != job_id or job.get("state") != expected_state
            or job.get("assigned_host") != "beast" or result.get("status") != ("failed" if failed else "ok")
            or result.get("returncode") != (1 if failed else 0)):
        raise ValueError("authenticated Beast job status differs")
    for envelope in (job.get("config", {}), job.get("submitted_request", {}).get("config", {})):
        if any(envelope.get(key) != value for key, value in request.items()):
            raise ValueError("scheduler request differs from staged request")
    custody = job.get("payload_custody", {})
    executed = result.get("executed_payload", {})
    for key in ("declared", "sealed", "executed"):
        if any(custody.get(key, {}).get(field) != executed.get(field) for field in ("sha256", "size_bytes")):
            raise ValueError("executed payload custody differs")
    if (not re.fullmatch(r"[0-9a-f]{64}", executed.get("sha256", ""))
            or type(executed.get("size_bytes")) is not int or executed["size_bytes"] <= 0
            or executed.get("host") != "beast"):
        raise ValueError("executed payload identity absent")
    cwd = result.get("diagnostics", {}).get("cwd", "")
    if cwd != "/mnt/hdd8tb/mrun-work/" + job_id:
        raise ValueError("unexpected authenticated job working directory")
    return cwd + "/" + RESULT


def validate_job(job: dict, job_id: str, request: dict, collection: dict | None = None) -> str:
    """Validate scheduler identity, preserving failed versus succeeded status."""
    _request_digest(request)
    if job.get("state") == "succeeded":
        return _job_common(job, job_id, request, failed=False)
    if not collection or collection.get("schema") != SCHEMA:
        raise ValueError("failed job requires an explicit recovery collection descriptor")
    if (collection.get("job_id") != job_id or collection.get("request_sha256") != request.get("request_sha256")
            or collection.get("failure_kind") != KNOWN_FAILURE):
        raise ValueError("recovery descriptor is not bound to the known finalizer failure")
    return _job_common(job, job_id, request, failed=True)


def _line_count(path: Path) -> int:
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def _verify_recovery(data: Path, actual: dict, job_id: str, request: dict,
                     collection: dict, run: Path | None) -> dict:
    if collection.get("schema") != SCHEMA or collection.get("job_id") != job_id:
        raise ValueError("collector-derived recovery collection identity differs")
    if collection.get("request_sha256") != request["request_sha256"]:
        raise ValueError("recovery request identity differs")
    if "remote-manifest.json" in actual["files"]:
        raise ValueError("failed execution must not acquire a producer remote manifest")
    failure_path = data / "failure.json"
    failure = read(failure_path)
    if (failure.get("error") != "'finalize_manifest'" or failure.get("error_type") != "KeyError"
            or not str(failure.get("scope", "")).startswith("Failed finite attempt")):
        raise ValueError("recovery accepts only the known post-exec finalize_manifest failure")
    config = request.get("config", {})
    expected_counts = config.get("counts")
    summary = read(data / "summary.json")
    source = read(data / "source.json")
    plan = read(data / "plan.json")
    if (plan != config or summary.get("job_id") != job_id or summary.get("request_sha256") != request["request_sha256"]
            or source.get("job_id") != job_id or source.get("request_sha256") != request["request_sha256"]
            or source.get("source_hashes") != request.get("source_hashes") or summary.get("counts") != expected_counts):
        raise ValueError("recovered source/plan/summary identity or census differs")
    if set(config.get("conditions", CELLS)) != set(CELLS):
        raise ValueError("native condition census differs")
    pngs = sorted(name for name in actual["files"] if name.lower().endswith(".png"))
    if len(pngs) != 12 or sum(name.endswith(".predicted.png") for name in pngs) != 6 or sum(name.endswith(".native.png") for name in pngs) != 6:
        raise ValueError("native PNG census differs")
    native = sorted(name for name in actual["files"] if name.startswith("native/") and name.endswith(".jsonl"))
    if native != ["native/" + name + ".jsonl" for name in BRANCHES]:
        raise ValueError("native branch ledger census differs")
    if any(_line_count(data / name) != 84 for name in native):
        raise ValueError("each native branch must retain exactly84 rows")
    rows = sorted(name for name in actual["files"] if name.startswith("rows/") and name.endswith(".json"))
    if rows != ["rows/" + name + ".json" for name in BRANCHES]:
        raise ValueError("native row summary census differs")
    for name in rows:
        row = read(data / name)
        expected_diagnostics = 84 if name.endswith(".sham.json") else 0
        if len(row.get("native_replay_diagnostics", [])) != expected_diagnostics:
            raise ValueError("native row summary diagnostic census differs")
    receipts = data / "receipt-emissions.jsonl"
    if _line_count(receipts) != 516:
        raise ValueError("receipt census differs")
    if (summary["counts"].get("acts") != 12 or summary["counts"].get("receipt_emissions") != 516
            or summary["counts"].get("adapter_steps") != 504 or summary["counts"].get("images") != 12):
        raise ValueError("summary physical census differs")
    transactions = []
    for cell in CELLS:
        verification = read(data / "transactions" / cell / "verification.json")
        if verification.get("phase") != "closed" or verification.get("verified") is not True:
            raise ValueError("transaction is not closed and verified: " + cell)
        transactions.append(cell)
    source_integrity = source.get("source_integrity", {})
    if source_integrity.get("unchanged") is not True:
        raise ValueError("parent source integrity is not closed")
    recipe = read(data / "parameter-artifacts.json")
    artifacts = recipe.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 169:
        raise ValueError("exact169 checkpoint recipes required")
    if any(entry.get("path") in actual["files"] for entry in artifacts):
        raise ValueError("per-run parameter archives must remain omitted")
    expected_retriever = config.get("expected_parameter_artifacts_module_sha256")
    if expected_retriever and sha(data / "retrieve_weights.py") != expected_retriever:
        raise ValueError("standalone retrieval helper differs")
    if run is not None:
        job_path = Path(run) / "job.json"
        if job_path.exists() and read(job_path).get("state") != "failed":
            raise ValueError("recovery overwrote the original failed scheduler state")
    return {"failure": failure, "summary": summary, "source": source, "plan": plan,
            "transactions": transactions, "manifest_origin": "collector-derived-source-manifest"}


def verify_manifest(data: os.PathLike[str] | str, actual: dict, job_id: str, request: dict,
                    collection: dict | None = None, run: os.PathLike[str] | str | None = None) -> dict:
    """Verify normal producer output or strict collector-derived failed output."""
    data = Path(data)
    if collection and collection.get("schema") == SCHEMA:
        return _verify_recovery(data, actual, job_id, request, collection, Path(run) if run else None)
    # The succeeded lane is delegated to the pre-existing native verifier.
    module = runpy.run_path(str(HERE / "submit_flux_native_corner.py"))
    return module["verify_manifest"](data, actual, job_id, request)


def _normal_producer(h, run, basename, collection_schema, job_file, *, expected_runtime_wheels=None):
    callback = h.get("normal_producer") if isinstance(h, dict) else None
    if callback is not None:
        return callback(h, run, basename, collection_schema, job_file,
                        expected_runtime_wheels=expected_runtime_wheels)
    module = runpy.run_path(str(HERE / "publish_flux_native_corner.py"))
    return module["producer"](h, run, basename, collection_schema, job_file,
                               expected_runtime_wheels=expected_runtime_wheels)


def _recovery_producer(h, run, basename, collection_schema, job_file, *, expected_runtime_wheels=None):
    run = Path(run).resolve()
    request, job, collection = [read(run / name) for name in ("request.json", job_file, "collection.json")]
    if basename != RESULT or collection_schema != "mdb-latent-direction-native-collection-v1":
        raise ValueError("recovery producer requires the latent-direction native result lane")
    _request_digest(request)
    validate_job(job, job["job_id"], request, collection)
    bindings = collection.get("bindings", {})
    for name, digest in bindings.items():
        if sha(run / name) != digest:
            raise ValueError("recovery custody binding differs: " + name)
    command = read(run / "admission-command.json")
    if job.get("result", {}).get("diagnostics", {}).get("command") != command.get("command"):
        raise ValueError("recovery admitted command differs")
    expected_log = job.get("result", {}).get("artifacts", {}).get("job_log")
    log = run / "worker.log"
    if (not isinstance(expected_log, dict) or not re.fullmatch(r"[0-9a-f]{64}", expected_log.get("sha256", ""))
            or type(expected_log.get("bytes")) is not int
            or sha(log) != expected_log["sha256"] or log.stat().st_size != expected_log["bytes"]):
        raise ValueError("recovery scheduler log differs")
    wheels = request.get("runtime_wheels")
    if not isinstance(wheels, dict) or not wheels:
        raise ValueError("request lacks explicit runtime wheel custody")
    for name, digest in {**request.get("source_hashes", {}), **wheels}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("recovery source/runtime bytes differ: " + name)
    data = run / "extracted" / basename
    actual = inventory(data)
    snapshots = [read(run / name) for name in
                 ("collection-remote-before.json", "collection-remote-after.json", "collection-local-inventory.json")]
    if any(snapshot != actual for snapshot in snapshots):
        raise ValueError("recovery before/after/local inventories differ")
    limits = request.get("config", {}).get("transport_limits", {})
    if (actual["expanded_bytes"] > limits.get("max_expanded_bytes", -1)
            or any(item["bytes"] > limits.get("max_member_bytes", -1) for item in actual["files"].values())):
        raise ValueError("recovery result exceeds declared transport limits")
    source_manifest = read(run / "collection-source-manifest.json")
    if (source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
            or source_manifest.get("provenance") != "collector-derived"
            or source_manifest.get("producer_remote_manifest_present") is not False
            or source_manifest.get("job_id") != job["job_id"]
            or source_manifest.get("request_sha256") != request["request_sha256"]
            or source_manifest.get("failure_kind") != KNOWN_FAILURE
            or source_manifest.get("remote_before_sha256") != sha(run / "collection-remote-before.json")
            or source_manifest.get("remote_after_sha256") != sha(run / "collection-remote-after.json")
            or source_manifest.get("local_inventory_sha256") != sha(run / "collection-local-inventory.json")
            or source_manifest.get("worker_log_sha256") != sha(log)
            or source_manifest.get("executed_payload_sha256") != job["result"]["executed_payload"]["sha256"]):
        raise ValueError("collector-derived source manifest differs")
    if (collection.get("source_manifest_sha256") != sha(run / "collection-source-manifest.json")
            or collection.get("manifest_origin") != "collector-derived-source-manifest"
            or collection.get("remote_manifest_present") is not False
            or collection.get("file_count") != actual["file_count"]
            or collection.get("expanded_bytes") != actual["expanded_bytes"]):
        raise ValueError("recovery collection receipt does not bind source manifest")
    verified = verify_manifest(data, actual, job["job_id"], request, collection=collection, run=run)
    result = {"run": run, "data": data, "request": request, "job": job, "collection": collection,
        "inventory": actual, "source": verified["source"], "summary": verified["summary"],
        "native_execution_status": "failed-post-exec-finalization",
        "manifest_origin": "collector-derived-source-manifest",
        "producer_remote_manifest_present": False,
        "scientific_work_completed": True,
        "numeric_interpretation_during_collection": False}
    return result


def producer(h, run, basename, normalcollectionschema, job_file, *, expected_runtime_wheels=None):
    """Dispatch unchanged succeeded production or strict failed recovery."""
    collection = read(Path(run) / "collection.json")
    if collection.get("schema") == SCHEMA:
        return _recovery_producer(h, run, basename, normalcollectionschema, job_file,
                                  expected_runtime_wheels=expected_runtime_wheels)
    return _normal_producer(h, run, basename, normalcollectionschema, job_file,
                            expected_runtime_wheels=expected_runtime_wheels)


def _remote_script(remote_path: str) -> str:
    header = ("import os,stat,json,hashlib\nfrom pathlib import Path,PurePosixPath\n"
              f"MAX_TOTAL={MAX_TOTAL}\nMAX_MEMBER={MAX_MEMBER}\nMAX_FILES={MAX_FILES}\n")
    return header + inspect.getsource(relative) + "\n" + inspect.getsource(inventory) + (
        "\nprint(json.dumps(inventory(" + repr(remote_path) + "),sort_keys=True))\n")


def _scan(remote_path: str, runner) -> tuple[dict, str]:
    script = _remote_script(remote_path)
    result = runner(["rtk", "proxy", "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                     "beast", "python3", "-"], input=script, text=True, capture_output=True,
                   check=True, timeout=600)
    if len(result.stdout) > 16 << 20:
        raise ValueError("remote inventory response oversized")
    value = json.loads(result.stdout)
    if value.get("file_count") != len(value.get("files", {})) or value.get("expanded_bytes") != sum(
            item["bytes"] for item in value.get("files", {}).values()):
        raise ValueError("remote inventory arithmetic differs")
    for name, item in value["files"].items():
        relative(name)
        if (set(item) != {"sha256", "bytes", "filetype"} or item["filetype"] != "regular"
                or type(item["bytes"]) is not int or item["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])):
            raise ValueError("invalid remote inventory member")
    return value, script


def recover(run: os.PathLike[str] | str, *, api=None, runner=None, job_id: str | None = None) -> dict:
    """Read-only authenticate and collect the original failed result tree."""
    runner = runner or subprocess.run
    run = Path(run).resolve()
    request = read(run / "request.json")
    _request_digest(request)
    for name, digest in {**request.get("source_hashes", {}), **request.get("runtime_wheels", {})}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("staged source/runtime bytes differ: " + name)
    if job_id is None:
        submission = read(run / "submission.json")
        job_id = submission.get("job_id")
    if api is None:
        from mdb.job_support import scheduler_api
        api = scheduler_api()
    status, raw_job, _ = api.request("GET", "/api/jobs/" + job_id, timeout_s=30)
    if status != 200:
        raise ValueError("scheduler job unavailable")
    job = json.loads(raw_job)
    remote = _job_common(job, job_id, request, failed=True)
    status, raw_events, _ = api.request("GET", "/api/jobs/" + job_id + "/events", timeout_s=30)
    if status != 200 or len(raw_events) > 16 << 20:
        raise ValueError("scheduler events unavailable or oversized")
    status, raw_log, _ = api.request("GET", "/api/jobs/" + job_id + "/logs?offset=0&wait_s=1", timeout_s=30)
    if status != 200 or len(raw_log) > 24 << 20:
        raise ValueError("scheduler log unavailable or oversized")
    expected_log = job.get("result", {}).get("artifacts", {}).get("job_log")
    if (not isinstance(expected_log, dict) or not re.fullmatch(r"[0-9a-f]{64}", expected_log.get("sha256", ""))
            or type(expected_log.get("bytes")) is not int
            or expected_log["bytes"] != len(raw_log)
            or expected_log["sha256"] != hashlib.sha256(raw_log).hexdigest()):
        raise ValueError("scheduler log identity differs")
    command_path = run / "admission-command.json"
    command = read(command_path)
    if job.get("result", {}).get("diagnostics", {}).get("command") != command.get("command"):
        raise ValueError("actual scheduler command differs")
    preserve(run / "job.json", raw_job)
    preserve(run / "events.json", raw_events)
    preserve(run / "worker.log", raw_log)
    descriptor = {"schema": SCHEMA, "job_id": job_id, "request_sha256": request["request_sha256"],
                  "failure_kind": KNOWN_FAILURE}
    destination = run / "extracted" / RESULT
    before, script = _scan(remote, runner)
    save(run / "collection-remote-before.json", before)
    if shutil.disk_usage(run).free < before["expanded_bytes"] + (128 << 20):
        raise ValueError("insufficient local collection space")
    allowlist = run / "collection-files.bin"
    preserve(allowlist, b"".join(name.encode() + b"\0" for name in before["files"]))
    if destination.exists():
        if inventory(destination) != before:
            raise ValueError("existing local recovery tree differs from remote-before inventory")
        copied = None
    else:
        destination.mkdir(parents=True, exist_ok=False)
        copied = runner(["rtk", "proxy", "rsync", "--recursive", "--times", "--protect-args", "--no-links",
                         "--no-devices", "--no-specials", "--from0", "--files-from=" + str(allowlist),
                         "--max-size=" + str(MAX_MEMBER), "--timeout=120",
                         "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes",
                         "beast:" + remote + "/", str(destination) + "/"], text=True,
                        capture_output=True, check=True, timeout=1800)
    if copied is not None:
        save(run / "collection-copy.json", {"returncode": copied.returncode,
            "stdout": copied.stdout, "stderr": copied.stderr})
    after, _ = _scan(remote, runner)
    local = inventory(destination)
    if before != after or before != local:
        raise ValueError("remote-before/after/local inventories differ")
    save(run / "collection-remote-after.json", after)
    save(run / "collection-local-inventory.json", local)
    verify_manifest(destination, local, job_id, request, collection=descriptor, run=run)
    source_manifest = {"schema": SOURCE_MANIFEST_SCHEMA, "provenance": "collector-derived",
        "producer_remote_manifest_present": False, "job_id": job_id,
        "request_sha256": request["request_sha256"], "scheduler_status": job["state"],
        "failure_kind": KNOWN_FAILURE, "remote_path": remote,
        "remote_before_sha256": sha(run / "collection-remote-before.json"),
        "remote_after_sha256": sha(run / "collection-remote-after.json"),
        "local_inventory_sha256": sha(run / "collection-local-inventory.json"),
        "source_inventory_sha256": sha(destination / "source.json"),
        "failure_sha256": sha(destination / "failure.json"),
        "worker_log_sha256": sha(run / "worker.log"),
        "executed_payload_sha256": job["result"]["executed_payload"]["sha256"],
        "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"],
        "scope": "Read-only recovery of a failed post-exec finalizer; no model calls or remote writes."}
    save(run / "collection-source-manifest.json", source_manifest)
    descriptor.update({"scheduler_status": job["state"], "remote_path": remote,
        "remote_manifest_present": False, "manifest_origin": "collector-derived-source-manifest",
        "file_count": local["file_count"], "expanded_bytes": local["expanded_bytes"],
        "collector_source_sha256": sha(Path(__file__)),
        "source_manifest_sha256": sha(run / "collection-source-manifest.json"),
        "executed_payload_sha256": job["result"]["executed_payload"]["sha256"]})
    names = ("request.json", "job.json", "events.json", "worker.log", "admission-command.json",
             "collection-remote-before.json", "collection-remote-after.json", "collection-local-inventory.json",
             "collection-copy.json", "collection-files.bin", "collection-source-manifest.json")
    descriptor["bindings"] = {name: sha(run / name) for name in names}
    receipt = {"schema": SCHEMA, "collection_mode": "authenticated-ssh-manifest-recovery",
        **descriptor, "remote_inventory_source_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "native_execution_status": "failed-post-exec-finalization", "scientific_work_completed": True,
        "model_calls_during_collection": 0, "scope": source_manifest["scope"]}
    save(run / "collection.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--recover", action="store_true", help="collect the original failed job read-only")
    args = parser.parse_args()
    if not args.recover:
        parser.error("the only operation is --recover RUN")
    print(json.dumps(recover(args.run), sort_keys=True))

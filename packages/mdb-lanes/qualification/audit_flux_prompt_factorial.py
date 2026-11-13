"""Run the frozen saved-evidence checker near its checkpoint, on a CPU lease.

No pipeline is loaded and no native forward is called. The checker still
performs 448 learned-weight mathematical transitions. Hardlinks refer only to
authenticated immutable producer files; the temporary weight cache is separate.
"""
from __future__ import annotations

import base64
import faulthandler
import gzip
import hashlib
import json
import os
import runpy
import stat
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
RESULT = "flux-prompt-factorial-result"
MARKER = "MDB_FACTORIAL_AUDIT_CAPSULE="
MAX_REPORT = 32 << 20


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def relative(name):
    path = PurePosixPath(name)
    if not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("unsafe evidence member")
    return path


def hardlink_inventory(source, target, inventory):
    """Do not follow links, overwrite contents, or set source mode/mtime.

    Creating a hardlink shares the inode and changes its link count and ctime.
    The authenticated inventory covers regular-file type, bytes and hashes.
    """
    source, target = Path(source), Path(target)
    if not source.is_dir() or source.is_symlink() or target.exists():
        raise ValueError("source directory or fresh hardlink destination differs")
    target.mkdir(parents=True)
    for name, expected in inventory["files"].items():
        parts = relative(name).parts
        incoming = source
        for part in parts[:-1]:
            incoming = incoming / part
            if not stat.S_ISDIR(incoming.lstat().st_mode):
                raise ValueError("non-directory evidence ancestor")
        incoming = incoming / parts[-1]
        before = incoming.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected["bytes"] or sha(incoming) != expected["sha256"]:
            raise ValueError("source member differs")
        output = target.joinpath(*parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.link(incoming, output, follow_symlinks=False)
        after = output.lstat()
        if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise ValueError("hardlink inode changed")


def release_cache(cache, manifest):
    """Delete only recipe-addressed files created inside this worker cache."""
    cache = Path(cache).resolve()
    if cache.name != "audit-parameter-cache" or not cache.is_relative_to(Path.cwd().resolve()):
        raise ValueError("unowned cache cleanup refused")
    names = {str(relative(item["path"])) for item in manifest["artifacts"]}
    found = {p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file() or p.is_symlink()}
    if found != names:
        raise ValueError("cache contains unexpected or missing members")
    released = 0
    for name in sorted(names):
        path = cache / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("cache member is not regular")
        released += path.stat().st_size
        path.unlink()
    for directory in sorted((p for p in cache.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()
    cache.rmdir()
    return {"files": len(names), "logical_bytes": released, "cache_removed": not cache.exists()}


def main():
    started = time.monotonic()
    faulthandler.dump_traceback_later(90, repeat=True)
    config = read(os.environ["SATURN_RUN_CONFIG"])
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    thread_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "VECLIB_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for variable in thread_variables:
        os.environ[variable] = str(config["cpu_threads"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("independent audit must have no visible CUDA device")
    if config["native_framework_forwards"] != 0 or config["mathematical_transitions"] != 448:
        raise ValueError("audit work declaration differs")
    parent = HERE / "parent-run"
    metadata = read(parent / "collection.json")
    parent_request = read(parent / "request.json")
    if (metadata["job_id"] != config["parent_job_id"] or metadata["request_sha256"] != config["parent_request_sha256"]
            or parent_request["request_sha256"] != config["parent_request_sha256"]):
        raise ValueError("parent admission identity differs")
    collector = runpy.run_path(str(HERE / "collect_flux_prompt_factorial.py"))
    expected = read(parent / "collection-local-inventory.json")
    source = Path(config["parent_remote_result"])
    actual_source = collector["validate_job"](read(parent / "collection-job.json"), config["parent_job_id"], parent_request)
    if str(source) != actual_source or collector["inventory"](source) != expected:
        raise ValueError("remote parent bytes differ from authenticated collection")
    target = parent / "extracted" / RESULT
    hardlink_inventory(source, target, expected)
    if collector["inventory"](target) != expected:
        raise ValueError("local hardlink inventory differs")
    print("MDB_FACTORIAL_AUDIT_PHASE " + json.dumps({"phase": "parent-linked", "files": expected["file_count"],
        "bytes": expected["expanded_bytes"], "native_framework_forwards": 0}), flush=True)
    cache = HERE / "audit-parameter-cache"
    if cache.exists():
        raise ValueError("audit cache must start absent")
    analyzer = runpy.run_path(str(HERE / "analyze_flux_prompt_factorial.py"))
    memoization = config.get("parameter_memoization", {"enabled": False})
    memo = None
    if memoization["enabled"]:
        adapter_path = HERE / "memoize_flux_audit_parameters.py"
        if sha(adapter_path) != request["source_hashes"][adapter_path.name]:
            raise ValueError("declared parameter I/O adapter source differs")
        adapter = runpy.run_path(str(adapter_path))
        memo = adapter["install"](analyzer, max_decoded_bytes=memoization["max_decoded_bytes"],
            storage_codec=memoization.get("storage_codec", "float32"),
            max_retained_bytes=memoization.get("max_retained_bytes"))

    def progress(value):
        if time.monotonic() - started > config["max_wall_seconds"]:
            raise RuntimeError("finite audit wall budget exhausted")
        print("MDB_FACTORIAL_AUDIT_PHASE " + json.dumps(value, sort_keys=True), flush=True)

    report = analyzer["analyze"](parent, parameter_root=cache, checkpoint=Path(config["checkpoint_root"]), progress=progress)
    parameter_reads = memo.verify() if memo is not None else {"mode": "uncached-frozen-evidence"}
    if memo is not None:
        parameter_reads["adapter_source_sha256"] = sha(adapter_path)
    if collector["inventory"](source) != expected or collector["inventory"](target) != expected:
        raise ValueError("parent evidence changed during independent audit")
    released = release_cache(cache, read(target / "parameter-artifacts.json"))
    report["near_data_execution"] = {"schema": "mdb-near-data-factorial-audit-v1",
        "audit_job_id": os.environ.get("MRUN_JOB_ID"), "audit_request_sha256": request["request_sha256"],
        "audit_worker_sha256": sha(Path(__file__)), "parent_job_id": config["parent_job_id"],
        "parent_request_sha256": config["parent_request_sha256"], "parent_inventory_unchanged": True,
        "data_access": "authenticated byte-preserving same-host regular-file hardlinks; shared inode, changed link count/ctime, preserved contents/mode/mtime",
        "native_framework_forwards": 0, "mathematical_transitions": 448, "parameter_cache_cleanup": released,
        "parameter_read_cache": parameter_reads,
        "thread_environment": {variable: os.environ[variable] for variable in thread_variables},
        "seconds": time.monotonic() - started}
    raw = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_REPORT:
        raise ValueError("audit report exceeds declared bound")
    with (HERE / "audit.json").open("xb") as stream:
        stream.write(raw)
    compressed = gzip.compress(raw, mtime=0)
    if len(compressed) > 8 << 20:
        raise ValueError("compressed audit report exceeds bound")
    print(MARKER + json.dumps({"schema": "mdb-actual-audit-capsule-v1", "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "gzip_bytes": len(compressed),
        "gzip_sha256": hashlib.sha256(compressed).hexdigest(), "base64_gzip": base64.b64encode(compressed).decode()}), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()

"""Zero-model scalar LocalStore durability timing; stage first, submit explicitly."""
from __future__ import annotations

import argparse
import base64
import errno
import faulthandler
import gzip
import hashlib
import io
import json
import os
import re
import runpy
import signal
import sys
import tarfile
import time
from pathlib import Path

ROOTS = ("/mnt/ssd1tb/mdb-diagnostics", "/mnt/hdd8tb/mrun-work")
OUTPUT = "flux-storage-profile-result"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def profile_case(root, *, label, event):
    """Exercise unchanged public LocalStore, instrumenting stdlib durability calls."""
    from saturn_pub import StateCut
    from saturn_pub.store import LocalStore

    root = Path(root)
    event("mkdir_before", label=label, path=str(root))
    root.mkdir(parents=True, exist_ok=False)
    event("mkdir_after", label=label, path=str(root))
    original_sync, original_link = os.fsync, os.link
    timings = []

    def measured(operation, function, *args, **kwargs):
        target = str(args[0])
        if operation == "fsync":
            try:
                target = os.readlink("/proc/self/fd/" + str(args[0]))
            except OSError:
                target = "fd:" + str(args[0])
        event("io_before", label=label, operation=operation, target=target)
        start, cpu = time.perf_counter(), time.process_time()
        try:
            value = function(*args, **kwargs)
        except BaseException as error:
            event("io_error", label=label, operation=operation, target=target, error=repr(error))
            raise
        record = {"operation": operation, "target": target,
            "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}
        timings.append(record)
        event("io_after", label=label, **record)
        return value

    os.fsync = lambda *args, **kwargs: measured("fsync", original_sync, *args, **kwargs)
    os.link = lambda *args, **kwargs: measured("link", original_link, *args, **kwargs)
    try:
        cut = StateCut("zero-model-storage-profile", {"device": "cpu", "scope": "scalar-only"},
            "diagnostic", None, {"counter": 0})
        store = LocalStore(root)
        event("save_before", label=label, cut=cut.fingerprint)
        started = time.perf_counter()
        identifier = store.save(cut)
        elapsed = time.perf_counter() - started
        event("save_after", label=label, cut=identifier, wall_seconds=elapsed)
        # Full byte/hash readback uses the public loader, without further writes.
        event("load_before", label=label, cut=identifier)
        loaded = store.load(identifier)
        if loaded.fingerprint != identifier or loaded.payload != {"counter": 0}:
            raise ValueError("scalar durable readback differs")
        event("load_after", label=label, cut=identifier)
        return {"label": label, "path": str(root), "cut": identifier, "save_seconds": elapsed,
            "io": timings, "admission": store.last_admission, "readback_exact": True,
            "files": {path.relative_to(root).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(root.rglob("*")) if path.is_file()}}
    finally:
        os.fsync, os.link = original_sync, original_link


def profile_cases(roots, job, *, event, rows=None):
    """Keep independent filesystem failures local; never retry or overwrite."""
    rows = [] if rows is None else rows
    for label, parent in zip(("ssd", "hdd"), roots, strict=True):
        path = Path(parent) / job / "scalar-localstore"
        case_events = []
        def case_event(kind, **fields):
            case_events.append({"kind": kind, **fields})
            event(kind, **fields)
        try:
            row = profile_case(path, label=label, event=case_event)
        except TimeoutError:
            # The lease-wide deadline is not an optional filesystem result.
            raise
        except OSError as error:
            if error.errno in {errno.ENOMEM, errno.EMFILE, errno.ENFILE}:
                raise
            row = {"label": label, "path": str(path), "status": "errored", "errno": error.errno,
                "error": f"{type(error).__name__}: {error}", "readback_exact": False}
            case_event("case_error", **row)
        else:
            row["status"] = "complete"
        row["events"] = case_events
        rows.append(row)
    return rows


def emit(output, report):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz", compresslevel=1) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_symlink():
                raise ValueError("diagnostic archive cannot contain symlinks")
            if path.is_file():
                bundle.add(path, arcname=output.name + "/" + path.relative_to(output).as_posix(), recursive=False)
    if archive.tell() > 1 << 20:
        raise ValueError("metadata-only diagnostic archive exceeds 1MiB")
    raw = archive.getvalue()
    bound_report = {**report, "archive_sha256": hashlib.sha256(raw).hexdigest(), "archive_bytes": len(raw)}
    print("MDB_DISCOVERY_UPGRADES_REPORT_V1 " + base64.b64encode(gzip.compress(json.dumps(bound_report).encode())).decode(), flush=True)
    print("MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 " + base64.b64encode(raw).decode(), flush=True)


def metadata_requirements(environment):
    """The public mrun import closure needs NumPy, without model frameworks."""
    selected = [value for value in environment["requirements"] if re.match(r"^numpy(?:[<>=!~\[]|$)", value)]
    if len(selected) != 1:
        raise ValueError("expected one existing public NumPy requirement")
    return selected


def metadata_worker_command():
    """Retain public bootstrap/wheels and its explicit NumPy-only dependency."""
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.payload import runtime_wheel_names, worker_command

    command = worker_command(Path(__file__).name)
    environment = intent()
    extras = list(environment["requirements"])
    selected = metadata_requirements(environment)
    wheels = ["./" + name for name in runtime_wheel_names()]
    result, kept, dropped = [], [], []
    index = 0
    while index < len(command):
        if command[index] == "--with":
            requirement = command[index + 1]
            if requirement in selected:
                result.extend(command[index:index + 2])
                kept.append(requirement)
            elif requirement in extras:
                dropped.append(requirement)
            elif requirement in wheels:
                result.extend(command[index:index + 2])
                kept.append(requirement)
            else:
                raise ValueError("unexpected public bootstrap requirement")
            index += 2
        else:
            result.append(command[index])
            index += 1
    if kept != wheels + selected or dropped != [value for value in extras if value not in selected] or len(wheels) != 4:
        raise ValueError("metadata bootstrap dependency contract differs")
    return result + ["--worker"]


def worker():
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    job = os.environ["MRUN_JOB_ID"]
    if (config != request["config"] or config["roots"] != list(ROOTS)
            or not re.fullmatch(r"job-[a-zA-Z0-9-]+", job)
            or any(sha(path) != digest for path, digest in request["source_hashes"].items())):
        raise ValueError("diagnostic request/source/path contract differs")
    framework_names = ("torch", "numpy", "diffusers", "transformers")
    frameworks_at_entry = [name for name in framework_names if name in sys.modules]
    output = Path(OUTPUT)
    output.mkdir(exist_ok=False)
    save(output / "plan.json", config)
    save(output / "request.json", request)
    started, events = time.monotonic(), []
    def event(kind, **fields):
        value = {"kind": kind, "seconds": time.monotonic() - started, **fields}
        events.append(value)
        print("MDB_FLUX_STORAGE_IO " + json.dumps(value, sort_keys=True), flush=True)
    report = {"schema": "mdb-flux-storage-profile-v1", "job_id": job,
        "request_sha256": request["request_sha256"], "source_hashes": request["source_hashes"],
        "model_loads": 0, "native_calls": 0, "tensor_bytes": 0, "cases": [],
        "frameworks_at_entry": frameworks_at_entry,
        "scope": "Two scalar LocalStore saves; sequential SSD-first then one HDD case. No model or tensor execution."}
    def expired(signum, frame):
        raise TimeoutError("finite180s diagnostic wall exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(180)
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        profile_cases(ROOTS, job, event=event, rows=report["cases"])
        complete = sum(row["status"] == "complete" for row in report["cases"])
        report["mechanics_status"] = "valid" if complete == 2 else "incomplete"
        report["instrument_status"] = "two-case-readback-qualified" if complete == 2 else "comparison-incomplete"
    except Exception as error:
        report.update(mechanics_status="errored", error=f"{type(error).__name__}: {error}")
        event("error", error=report["error"])
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        faulthandler.cancel_dump_traceback_later()
    report["worker_seconds"] = time.monotonic() - started
    report["frameworks_added_after_entry"] = [name for name in framework_names
        if name in sys.modules and name not in frameworks_at_entry]
    with (output / "events.jsonl").open("x") as stream:
        for value in events:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
    save(output / "summary.json", report)
    emit(output, report)
    if report["mechanics_status"] != "valid":
        raise SystemExit(1)


def stage(run_dir):
    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    config = {"schema": "mdb-flux-storage-profile-plan-v1", "roots": list(ROOTS),
        "case_order": ["ssd", "hdd"], "cases_per_root": 1, "payload": {"counter": 0},
        "max_wall_seconds": 180, "model_loads": 0, "native_calls": 0}
    save(run_dir / "plan.json", config)
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "path_token_allowlist": [OUTPUT, *ROOTS], "hashed_roots": [source.name, "plan.json", preflight.name]})
    environment = intent()
    environment["requirements"] = metadata_requirements(environment)
    extras = [(run_dir / "plan.json", "plan.json"), (preflight, preflight.name)]
    request = {"schema": "mdb-flux-storage-profile-request-v1", "config": config,
        "task_family": "analysis", "analysis_kind": "scalar-filesystem-durability", "needs": {"cpu": True},
        "execution_environment": environment,
        "worker_command": metadata_worker_command(),
        "source_hashes": {name: sha(path) for path, name in [(source, source.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_free=True, model_loads=0,
            logical_operations=2, logical_checkpoint_replays=0, max_local_work_items=2,
            extra={"task_family": "analysis"},
            checkpoint_policy="Two scalar storage diagnostics in one CPU lease; zero models, tensors or retries"),
        "resources": {"ram_mb": 512, "cpu_threads": 1, "disk_gb": 1, "est_wall_s": 30,
            "timeout_s": 240, "host_pin": "beast"}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=source, config=config, request=request, additional_files=extras)
    return {"submitted": False, "run_dir": str(run_dir), "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb_lanes.job_support import guarded_mrun_launch

    run_dir = Path(run_dir)
    request = json.loads((run_dir / "request.json").read_text())
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or any(sha(run_dir / "payload" / path) != digest for path, digest in request["source_hashes"].items())):
        raise ValueError("staged diagnostic request/source changed")
    if (run_dir / "submission.json").exists():
        raise ValueError("diagnostic already admitted")
    if request["worker_command"] != metadata_worker_command():
        raise ValueError("staged metadata-only bootstrap changed")
    job = guarded_mrun_launch(request["worker_command"], request=request,
        experiment="mdb-flux-storage-profile", task="scalar-localstore-ssd-then-hdd", model=None,
        payload=run_dir / "payload", prefer_host="beast", pin="beast", gpu=False, needs=request["needs"],
        ram_mb=512, cpu_threads=1, disk_gb=1, est_wall_s=30, timeout_s=240, retry_on_kill=False, preflight=True,
        note="Zero models/tensors/CUDA. One scalar public LocalStore save on SSD, then one HDD save. Log every fsync/link before+after;30s traceback,180s inner/240s outer. No retries; storage diagnosis only.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def collect(run_dir, job_id):
    helper = Path(__file__).with_name("discovery_upgrades.py")
    return runpy.run_path(str(helper))["collect"](Path(run_dir), job_id,
        max_expanded_bytes=1 << 20, max_compressed_bytes=1 << 20, log_storage="gzip")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect", metavar="JOB_ID")
    args = parser.parse_args()
    if args.worker:
        worker()
    elif args.collect and args.run_dir:
        print(json.dumps(collect(args.run_dir, args.collect), sort_keys=True))
    elif args.run_dir:
        print(json.dumps(submit_staged(args.run_dir) if args.submit_staged else stage(args.run_dir), sort_keys=True))
    else:
        parser.error("--worker or --run-dir required; staging never submits")


if __name__ == "__main__":
    main()

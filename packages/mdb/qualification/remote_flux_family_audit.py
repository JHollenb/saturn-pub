"""Separate zero-forward CPU audit and compact retrieval for the resident assay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path, PurePosixPath


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def relative(value):
    require(isinstance(value, str) and value and "\\" not in value and "\0" not in value, "unsafe artifact path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value, "unsafe artifact path")
    return Path(value)


def copy_relative(source, output, name):
    name = relative(name)
    source, target = Path(source) / name, Path(output) / name
    require(source.is_file() and not any(path.is_symlink() for path in (source, *source.parents)), "nonregular source artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), "compact artifact already exists")
    with source.open("rb") as incoming, target.open("xb") as destination:
        shutil.copyfileobj(incoming, destination)


def request_seal(request):
    unsealed = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(unsealed, sort_keys=True, allow_nan=False).encode()).hexdigest()


COMPACT = {"max_archive_bytes": 128 << 20, "max_expanded_bytes": 256 << 20,
           "max_files": 512, "max_member_bytes": 32 << 20, "chunk_bytes": 4 << 20,
           "max_log_bytes": 192 << 20}


def worker():
    config = read(os.environ.get("SATURN_RUN_CONFIG", "saturn_run_config.json"))
    request = read(os.environ.get("SATURN_DEBUG_REQUEST", "saturn_debug_request.json"))
    if request_seal(request) != request["request_sha256"] or config != request["config"] or any(sha(relative(name)) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("audit source/request differs")
    os.environ["MRUN_URL"] = config["endpoint"]
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "2"
    custody = runpy.run_path("flux_family_custody.py")
    raw = Path(config["raw_store"])
    if raw.exists():
        if not config["reuse_collected"]:
            raise ValueError("refusing to overwrite collected raw evidence")
    else:
        if config["reuse_collected"]:
            raise ValueError("declared retained evidence does not exist")
        shutil.copytree("input-run", raw)
        custody["collect"](raw, config["producer_job_id"], config["producer_limits"])
    if read(raw / "request.json")["request_sha256"] != config["producer_request_sha256"]:
        raise ValueError("wrong producer request")
    audit = runpy.run_path("analyze_flux_family.py")["audit"](raw)
    audit["auditor_source_sha256"] = sha("analyze_flux_family.py")
    analysis = raw / ("analysis-" + os.environ["MRUN_JOB_ID"] + ".json")
    save(analysis, audit)
    shutil.copyfile("analyze_flux_family.py", raw / (analysis.stem + "-source.py"))
    data = raw / "extracted/flux-family-result"
    output = Path("flux-family-audit-result")
    output.mkdir(exist_ok=False)
    shutil.copyfile(analysis, output / "producer-analysis.json")
    shutil.copyfile("analyze_flux_family.py", output / "producer-analysis-source.py")
    for name in ("request.json", "job.json", "report.json", "log-custody.json", "archive-custody.json"):
        shutil.copyfile(raw / name, output / ("producer-" + name))
    for name in ("plan.json", "sources.json", "summary.json", "resolver.json", "support_selection.json", "timeline.jsonl", "files.json"):
        shutil.copyfile(data / name, output / ("producer-" + name))
    for path in sorted((data / "rows").glob("*.json")):
        target = output / "rows" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        image = read(path)["image"]["path"]
        copy_relative(data, output, image)
    report = {"schema": "mdb-flux-family-remote-audit-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "producer_job_id": config["producer_job_id"],
        "producer_request_sha256": config["producer_request_sha256"],
        "producer_archive_sha256": read(raw / "report.json")["archive_sha256"],
        "producer_analysis_sha256": sha(analysis), "mechanics_status": audit["mechanics_status"],
        "raw_store": {"host": "beast", "path": str(raw)}, "new_model_calls": 0,
        "images": audit["counts"]["images"], "terminal_status": "unassessed",
        "scope": "Independent CPU audit verifies complete arrays/CAS/projections/receipts/PNG/selector evidence on storage host; compact view returns original PNGs and audited records; full arrays remain in raw store"}
    save(output / "audit-receipt.json", report)
    save(output / "files.json", {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    custody["emit"](output, report, COMPACT)


def verify_compact(run_dir):
    """Verify compact transport and its source-bound full-audit attestation.

    The complete producer archive remains remote. Its copied custody records and
    independent audit are bound here; they do not substitute for retrieving that
    archive when an independent local full-array re-audit is required.
    """
    root = Path(run_dir)
    custody = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))
    transport = custody["verify"](root, COMPACT)
    request, envelope = read(root / "request.json"), read(root / "report.json")
    require(envelope["archive_root"] == "flux-family-audit-result", "wrong compact archive root")
    data, config = root / "extracted/flux-family-audit-result", request["config"]
    report = read(data / "audit-receipt.json")
    producer, producer_request = read(data / "producer-report.json"), read(data / "producer-request.json")
    analysis = read(data / "producer-analysis.json")
    job = config["producer_job_id"]
    require(report == envelope["report"] and report["job_id"] == transport["job_id"]
            and report["request_sha256"] == request["request_sha256"]
            and report["new_model_calls"] == request["new_model_calls"] == 0, "compact audit receipt differs")
    require(request_seal(producer_request) == producer_request["request_sha256"] == config["producer_request_sha256"]
            == report["producer_request_sha256"] == producer["request_sha256"]
            and report["producer_job_id"] == producer["job_id"] == job
            and producer["report"]["job_id"] == job and producer["report"]["request_sha256"] == producer_request["request_sha256"], "producer request/job binding differs")
    require(sha(data / "producer-request.json") == request["source_hashes"]["input-run/request.json"]
            and report["producer_archive_sha256"] == producer["archive_sha256"]
            and report["producer_analysis_sha256"] == sha(data / "producer-analysis.json")
            and analysis["auditor_source_sha256"] == sha(data / "producer-analysis-source.py") == request["source_hashes"]["analyze_flux_family.py"], "producer archive/analysis/source binding differs")
    require(analysis["custody"]["job_id"] == job
            and analysis["custody"]["request_sha256"] == producer_request["request_sha256"]
            and analysis["custody"]["archive_sha256"] == producer["archive_sha256"]
            and analysis["mechanics_status"] == report["mechanics_status"] == "valid"
            and analysis["new_model_calls"] == 0, "full audit authority differs")
    archive, log = read(data / "producer-archive-custody.json"), read(data / "producer-log-custody.json")
    require(archive["complete"] is True and log["complete"] is True
            and archive["job_id"] == log["job_id"] == job and archive["archive_sha256"] == producer["archive_sha256"]
            and log["job_json_sha256"] == sha(data / "producer-job.json"), "producer custody attestation differs")
    manifest = read(data / "producer-files.json")
    require(sha(data / "producer-files.json") == producer["files_manifest_sha256"], "producer file manifest differs")
    for name in ("plan.json", "sources.json", "summary.json", "resolver.json", "support_selection.json", "timeline.jsonl"):
        require(sha(data / ("producer-" + name)) == manifest[name], "copied producer metadata differs: " + name)
    require(read(data / "producer-plan.json") == producer_request["config"]
            and read(data / "producer-summary.json") == producer["report"], "producer plan/summary differs")
    planned = {row["id"] for row in producer_request["config"]["rows"]}
    require({path.stem for path in (data / "rows").glob("*.json")} == planned
            and len(planned) == analysis["counts"]["images"] == report["images"], "compact row inventory differs")
    for name in planned:
        row_path = data / "rows" / (name + ".json")
        row = read(row_path)
        image = relative(row["image"]["path"])
        require(row["id"] == name and sha(row_path) == manifest["rows/" + name + ".json"]
                and sha(data / image) == row["image"]["sha256"] == manifest[image.as_posix()], "compact row/image differs")
    return {**transport, "compact_binding_status": "verified", "producer_job_id": job,
            "producer_archive_sha256": producer["archive_sha256"], "producer_analysis_sha256": sha(data / "producer-analysis.json"),
            "images": len(planned), "new_model_calls": 0,
            "scope": "compact custody plus source-bound remote full-array audit attestation; raw producer archive remains remote"}


def verify_and_record(run_dir):
    """Reverify every time; create the receipt once or retain identical JSON."""
    result = verify_compact(run_dir)
    receipt = Path(run_dir) / "compact-verification.json"
    try:
        save(receipt, result)
    except FileExistsError:
        require(not receipt.is_symlink() and receipt.is_file(), "existing compact verification receipt is not a regular file")
        require(read(receipt) == result, "existing compact verification receipt differs from newly verified result")
    return result


def stage(run_dir, producer, *, reuse_collected=False):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here, run_dir, producer = Path(__file__).resolve().parent, Path(run_dir), Path(producer)
    source = read(producer / "request.json")
    require(request_seal(source) == source["request_sha256"], "producer request self seal differs")
    job = read(producer / "submission.json")["job_id"]
    config = {"schema": "mdb-flux-family-remote-audit-plan-v1", "producer_job_id": job,
        "producer_request_sha256": source["request_sha256"], "producer_limits": source["config"]["limits"],
        "endpoint": "http://192.168.68.68:9025", "raw_store": "/mnt/hdd8tb/mdb/evidence/flux-family/" + job,
        "reuse_collected": reuse_collected}
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    extras = [(producer / "request.json", "input-run/request.json")]
    for name, expected in source["source_hashes"].items():
        path = producer / "payload" / relative(name)
        if sha(path) != expected:
            raise ValueError("producer staged source differs")
        extras.append((path, "input-run/payload/" + name))
    names = ("analyze_flux_family.py", "flux_family_custody.py", "flux_family_plan.py", "analyze_flux_mlp_forecast.py",
             "analyze_flux_concept_routes.py", "summarize_context_window.py")
    extras += [(here / name, name) for name in names]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [Path(__file__).name, *names, "input-run", preflight.name],
                     "path_token_allowlist": ["flux-family-audit-result", "input-run", config["raw_store"], "/mnt/hdd8tb/mdb/evidence/flux-family"]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-flux-family-remote-audit-request-v1", "config": config, "execution_environment": intent(),
        "task_family": "analysis", "needs": {"cpu": True}, "new_model_calls": 0,
        "source_hashes": {name: sha(path) for path, name in [(Path(__file__), Path(__file__).name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True,
            logical_operations=1, logical_branches=1, logical_checkpoint_replays=0, max_local_work_items=1,
            extra={"task_family": "analysis"},
            checkpoint_policy="read-only exact producer collection/full audit; no model instantiation or scheduler replay")}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=Path(__file__), config=config, request=request, additional_files=extras)
    job = guarded_mrun_launch(worker_command(Path(__file__).name), request=request, model=None, task="offline-audit",
        experiment="mdb-flux-family-offline-audit", payload=run_dir / "payload", pin="beast", needs=request["needs"], gpu=False,
        ram_mb=8192, cpu_threads=2, timeout_s=600, retry_on_kill=False, preflight=True,
        note="CPU2 model-free full six-panel audit;0model loads/forwards. Stream multipart producer archive,CAS/nativeports/Acts/latents/PNGs/seals/selectors;host-local rawretention,return47originalPNGs+compactrecords. Bounded8GiBRAM/600s,no retries;storage-host pin required.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    if os.environ.get("MRUN_JOB_ID"):
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run_dir", type=Path)
        parser.add_argument("--producer", type=Path)
        parser.add_argument("--reuse-collected", action="store_true")
        parser.add_argument("--collect")
        parser.add_argument("--verify", action="store_true")
        args = parser.parse_args()
        if args.collect or args.verify:
            custody = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))
            result = verify_and_record(args.run_dir) if args.verify else custody["collect"](args.run_dir, args.collect, COMPACT)
        else:
            if args.producer is None:
                parser.error("--producer required")
            result = stage(args.run_dir, args.producer, reuse_collected=args.reuse_collected)
        print(json.dumps(result))

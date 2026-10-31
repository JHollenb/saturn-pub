"""Audit large retained merge evidence on its storage host; return compact views."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import runpy
import shutil
import tarfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def worker():
    config = json.loads(Path(os.environ["SATURN_RUN_CONFIG"]).read_text())
    request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
    if config != request["config"] or any(sha(Path(name)) != value for name, value in request["source_hashes"].items()):
        raise ValueError("audit request/source differs")
    os.environ["MRUN_URL"] = config["endpoint"]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    raw = Path(config["store_path"])
    if raw.exists():
        raise ValueError("refusing to overwrite existing retained evidence")
    shutil.copytree("input-run", raw)
    submitted = json.loads((raw / "request.json").read_text())
    if submitted["request_sha256"] != config["model_request_sha256"]:
        raise ValueError("producer request differs")
    support = runpy.run_path("discovery_upgrades.py")
    collected = support["collect"](raw, config["model_job"], max_compressed_bytes=256 << 20,
        max_expanded_bytes=512 << 20, log_storage="gzip")
    if collected["state"] != "succeeded":
        raise ValueError("producer not successful; raw evidence retained")
    analysis = runpy.run_path("analyze_flux_merge.py")["analyze"](raw)
    save(raw / "analysis-v1.json", analysis)
    output = Path("offline-audit-result")
    output.mkdir(exist_ok=False)
    for name in ("request.json", "report.json", "job.json", "log-custody.json", "analysis-v1.json"):
        shutil.copyfile(raw / name, output / ("producer-" + name))
    data = raw / "extracted/flux-merge-result"
    for name in ("source.json", "plan.json", "summary.json", "files.json", "transaction-report.json"):
        shutil.copyfile(data / name, output / name)
    for path in sorted((data / "rows").glob("*.json")):
        relative = Path("rows") / path.name
        (output / relative).parent.mkdir(exist_ok=True)
        shutil.copyfile(path, output / relative)
        row = json.loads(path.read_text())
        shutil.copyfile(data / row["image"]["path"], output / row["image"]["path"])
    summary = {"schema": "mdb-flux-remote-offline-audit-v1", "job_id": os.environ["MRUN_JOB_ID"],
        "request_sha256": request["request_sha256"], "producer_job_id": config["model_job"],
        "producer_archive_sha256": analysis["source"]["archive_sha256"], "producer_audit_sha256": sha(raw / "analysis-v1.json"),
        "raw_store": {"host": "beast", "path": str(raw), "log_custody_sha256": sha(raw / "log-custody.json")},
        "mechanics_status": "valid", "terminal_status": "unassessed", "model_forwards": 0,
        "scope": "separate CPU process verifies full source archive, typed arrays, equations and futures on storage host; returned view retains original PNG bytes and compact records; full port/latent/CAS data remain in the declared raw store and original producer archive"}
    save(output / "audit-receipt.json", summary)
    support["emit"](output, summary)


def stage(run_dir, original):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    source = json.loads((original / "request.json").read_text())
    producer = json.loads((original / "submission.json").read_text())["job_id"]
    config = {"schema": "mdb-flux-remote-audit-plan-v1", "model_job": producer,
        "model_request_sha256": source["request_sha256"], "endpoint": "http://192.168.68.68:9025",
        "store_path": "/mnt/hdd8tb/mdb/evidence/flux-merge/" + producer}
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    extras = [(original / "request.json", "input-run/request.json")]
    for name, expected in source["source_hashes"].items():
        path = original / "payload" / name
        if sha(path) != expected:
            raise ValueError("producer staged bytes changed")
        extras.append((path, "input-run/payload/" + name))
    names = ("analyze_flux_merge.py", "summarize_context_window.py", "analyze_flux_mlp_forecast.py", "analyze_flux_concept_routes.py", "discovery_upgrades.py")
    extras += [(here / name, name) for name in names]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": ["remote_flux_audit.py", *names, "input-run", "preflight.v1.json"],
        "path_token_allowlist": ["offline-audit-result", "input-run", config["store_path"], "/mnt/hdd8tb/mdb/evidence/flux-merge"]})
    extras.append((preflight, preflight.name))
    worker_path = here / "remote_flux_audit.py"
    request = {"schema": "mdb-flux-remote-audit-request-v1", "config": config, "task_family": "analysis",
        "needs": {"cpu": True}, "execution_environment": intent(), "model_forwards": 0,
        "source_hashes": {name: sha(path) for path, name in [(worker_path, worker_path.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=1, logical_branches=1,
            logical_checkpoint_replays=0, max_local_work_items=1, extra={"task_family": "analysis"},
            checkpoint_policy="read-only producer archive audit; no model instantiation; retained host-local source tree and compact view")}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker_path, config=config, request=request, additional_files=extras)
    command = worker_command(worker_path.name)
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=None, task="offline-audit", experiment="mdb-flux-remote-audit",
        payload=run_dir / "payload", pin="beast", gpu=False, needs=request["needs"], ram_mb=4096, cpu_threads=2,
        timeout_s=180, retry_on_kill=False, preflight=True,
        note="CPU2 offline merge evidence audit;0model loads/forwards. Fetch completed producer archive, verify full CAS/ports/merge equations/latents/PNGs, retain raw tree on Beast HDD and return only compact audit+original11PNGs. 4GB bounds compressedlog+archive/NumPy working set. Host pin required for durable host-local storage; no retries.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def verify_compact(run_dir):
    helper = runpy.run_path(str(Path(__file__).with_name("summarize_context_window.py")))
    request, report, job = (json.loads((run_dir / name).read_text()) for name in ("request.json", "report.json", "job.json"))
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"] or report["request_sha256"] != request["request_sha256"]:
        raise ValueError("audit request/report differs")
    if job["state"] != "succeeded" or job["job_id"] != report["job_id"]:
        raise ValueError("audit job did not succeed")
    for name, expected in request["source_hashes"].items():
        if sha(run_dir / "payload" / name) != expected:
            raise ValueError("staged auditor source differs")
    marker = "MDB_DISCOVERY_UPGRADES_ARCHIVE_V1 "
    lines = [line[len(marker):] for line in helper["collector_log"](run_dir, max_bytes=128 << 20).splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise ValueError("expected one compact archive")
    raw = base64.b64decode(lines[0], validate=True)
    if hashlib.sha256(raw).hexdigest() != report["archive_sha256"]:
        raise ValueError("compact archive differs")
    output = run_dir / "extracted/offline-audit-result"
    inventory = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 128 or sum(member.size for member in members) > 64 << 20:
            raise ValueError("compact archive geometry exceeds finite bound")
        for member in members:
            parts = Path(member.name).parts
            if not member.isfile() or not parts or parts[0] != "offline-audit-result" or ".." in parts:
                raise ValueError("unsafe compact archive member")
            relative = Path(*parts[1:])
            if relative.as_posix() in inventory or sha(output / relative) != hashlib.sha256(archive.extractfile(member).read()).hexdigest():
                raise ValueError("compact member differs")
            inventory.add(relative.as_posix())
    if inventory != {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}:
        raise ValueError("compact inventory differs")
    receipt, producer, audit = (json.loads((output / name).read_text()) for name in ("audit-receipt.json", "producer-report.json", "producer-analysis-v1.json"))
    if receipt["producer_job_id"] != request["config"]["model_job"] or producer["job_id"] != receipt["producer_job_id"]:
        raise ValueError("wrong producer")
    if producer["archive_sha256"] != receipt["producer_archive_sha256"] or producer["archive_sha256"] != audit["source"]["archive_sha256"] or sha(output / "producer-analysis-v1.json") != receipt["producer_audit_sha256"]:
        raise ValueError("producer audit/archive binding differs")
    manifest = json.loads((output / "files.json").read_text())
    for path in output.rglob("*"):
        if path.is_file() and path.relative_to(output).as_posix() in manifest:
            if sha(path) != manifest[path.relative_to(output).as_posix()]:
                raise ValueError("returned producer bytes differ")
    return {"schema": "mdb-flux-compact-custody-verification-v1", "job_id": job["job_id"],
        "producer_job_id": producer["job_id"], "verified_compact_files": len(inventory), "compact_archive_sha256": report["archive_sha256"],
        "producer_archive_sha256": producer["archive_sha256"], "producer_audit_sha256": receipt["producer_audit_sha256"],
        "raw_store": receipt["raw_store"], "verified": True,
        "scope": "local compact view and original image bytes verified; full arrays verified by the pinned separate CPU audit on the storage host"}


if __name__ == "__main__":
    if os.environ.get("MRUN_JOB_ID"):
        worker()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("run_dir", type=Path)
        parser.add_argument("--producer", type=Path)
        parser.add_argument("--collect")
        parser.add_argument("--verify", action="store_true")
        args = parser.parse_args()
        if args.verify:
            result = verify_compact(args.run_dir.resolve())
            save(args.run_dir / "compact-verification.json", result)
        elif args.collect:
            result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
                args.run_dir, args.collect, log_storage="gzip")
        else:
            if args.producer is None:
                parser.error("--producer required")
            result = stage(args.run_dir.resolve(), args.producer.resolve())
        print(json.dumps(result))

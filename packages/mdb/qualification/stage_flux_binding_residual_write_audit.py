"""Admit a saved-only CPU audit of the residual/net-write binding panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shlex
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "audit_flux_binding_residual_write.py"
PRODUCER = "flux_binding_residual_write.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def prior():
    return runpy.run_path(str(HERE / "stage_common_differential_saved_audit.py"))


def command(request):
    from mdb.payload import worker_command

    prior()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(1, "CUDA_VISIBLE_DEVICES=")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        args.insert(1, name + "=4")
    args.insert(args.index("python") + 1, "-B")
    args.extend(["--config", "saturn_run_config.json"])
    return args


def stage(run, source):
    from mdb.job_support import one_lease_scheduling_contract, scheduler_api
    from mdb.payload import stage_worker

    run, source = Path(run).resolve(), Path(source).resolve()
    worker_path = HERE / WORKER
    worker = runpy.run_path(str(worker_path))
    parent = read(source / "request.json")
    if (identity(parent) != parent["request_sha256"] or parent["config"]["schema"] != "mdb-flux-binding-residual-write-v1"
            or parent["source_hashes"].get(PRODUCER) != worker["PRODUCER_SHA256"]):
        raise ValueError("exact residual/write producer required")
    os.environ["MRUN_URL"] = parent["scheduler_url"]
    prior()["bind"](parent["execution_environment"])
    for name, expected in {**parent["source_hashes"], **parent["runtime_wheels"]}.items():
        if sha(source / "payload" / name) != expected:
            raise ValueError("original staged bytes changed: " + name)
    jid = read(source / "submission.json")["job_id"]
    api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + jid, timeout_s=30)
    job = json.loads(raw)
    if status != 200 or job["state"] != "succeeded" or job["assigned_host"] != "beast":
        raise ValueError("one succeeded native job required")
    prior()["validate_native_custody"](parent, job, read(source / "admission-command.json")["command"])
    status, log, _ = api.request("GET", "/api/jobs/" + jid + "/logs?offset=0&wait_s=1", timeout_s=30)
    if status != 200 or len(log) > 16 << 20:
        raise ValueError("bounded complete native log required")
    declared = job["result"].get("artifacts", {}).get("job_log")
    if declared and (len(log) != declared["bytes"] or hashlib.sha256(log).hexdigest() != declared["sha256"]):
        raise ValueError("native log bytes differ")
    root = Path(parent["config"]["output_scratch"]) / jid / parent["config"]["output_name"]
    remote = shlex.join(["python3", "-I", "-c", prior()["REMOTE"], str(root)])
    info = json.loads(subprocess.run(["ssh", "beast", remote], check=True, capture_output=True, text=True).stdout)
    publications = [json.loads(line.removeprefix("MDB_DIRECTORY_READY ")) for line in log.decode().splitlines()
                    if line.startswith("MDB_DIRECTORY_READY ")]
    if len(publications) != 1:
        raise ValueError("one native directory publication required")
    publication = publications[0]
    if (info["job_id"] != jid or info["request_sha256"] != parent["request_sha256"]
            or publication["job_id"] != jid or publication["request_sha256"] != parent["request_sha256"]
            or publication["output_root"] != str(root)
            or publication["files_manifest_sha256"] != info["files"]["files.json"]["sha256"]
            or publication["summary_sha256"] != info["files"]["summary.json"]["sha256"]):
        raise ValueError("native publication identity differs")
    config = worker["make_config"](dict(native_root=str(root), native_job_id=jid,
        native_request_sha256=parent["request_sha256"], native_manifest_sha256=publication["files_manifest_sha256"],
        native_summary_sha256=publication["summary_sha256"], native_publication=publication,
        publication_id=hashlib.sha256(json.dumps(publication, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        output_scratch=str(root.parents[1])))
    worker["validate_config"](config)
    run.mkdir(parents=True, exist_ok=False)
    sources = {WORKER: worker_path}
    for name, expected in worker["HELPERS"].items():
        path = HERE / name
        if Path(name).name != name or sha(path) != expected:
            raise ValueError("saved helper bytes changed: " + name)
        sources[name] = path
    for name, value in (("source-job.json", job), ("source-metadata.json", info), ("source-request.json", parent),
                        ("source-publication.json", publication), ("plan.json", config)):
        save(run / name, value)
        sources[name] = run / name
    (run / "source-worker.log").write_bytes(log)
    sources["source-worker.log"] = run / "source-worker.log"
    save(run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=[WORKER],
        path_token_allowlist=[str(root), config["output_scratch"], "/mnt/big/llm-models/FLUX.2-klein-4B"]))
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-binding-residual-write-audit-request-v1", panel="residual-write-saved-audit", config=config,
        scheduler_url=parent["scheduler_url"], execution_environment=parent["execution_environment"], runtime_wheels=parent["runtime_wheels"],
        source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
        source_job_id=jid, source_request_sha256=parent["request_sha256"], source_terminal_state="succeeded",
        needs={"cpu": True, "host": "beast"}, backend="numpy-and-qualified-cpu-cast", device="cpu", dtype="float64",
        task_family="analysis", resource_exception=dict(ram_mb=4096, vram_mb=0, cpu_threads=4, disk_gb=2,
                                                        est_wall_s=120, timeout_s=600),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=64,
            logical_branches=1, logical_checkpoint_replays=0, max_local_work_items=128,
            extra={"task_family": "analysis"}, checkpoint_policy="Saved-only source arithmetic,actual CPU cast and full future audit;no models/CUDA/forwards;one finite CPU lease,no retries;retain16original PNGs and311parameter recipes."))
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker_path, config=config, request=request,
                 additional_files=[(path, name) for name, path in sources.items() if name != WORKER])
    validate(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"], submitted=False))
    return dict(request_sha256=request["request_sha256"], native_job_id=jid, submitted=False)


def validate(run, request):
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("saved-audit request or launcher changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("staged audit bytes changed: " + name)


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate(run, request)
    if (run / "submission.json").exists():
        raise ValueError("audit already submitted")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"]))
    job = guarded_mrun_launch(args, request=request, experiment="mdb-binding-residual-write-audit", task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=False, prefer_host="beast", retry_on_kill=False, preflight=True,
        note="Saved-only residual/write audit:independentFP64sourcefactorial,actualCPU BF16cast,endpoint/RWknowncontroljoins,8fullimagepairs48ports432transitions/lane16Acts/lane,fullsigned interactions.16originalPNGs compact+311recipes,0models/CUDA,4CPU4GiB120sestimate600sfinitecap,no retries.", **request["resource_exception"])
    save(run / "submission.json", dict(job_id=str(job), request_sha256=request["request_sha256"]))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and not args.source:
        parser.error("stage requires --source")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.source), sort_keys=True))

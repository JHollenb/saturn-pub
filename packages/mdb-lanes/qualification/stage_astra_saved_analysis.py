"""Admit one finite, CPU-only analysis of a named original campaign directory."""
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
WORKERS = {"qwen-recovery": "recover_qwen_spectral_readout.py", "flux-audit": "audit_astra_flux_campaign.py"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def runtime():
    return runpy.run_path(str(HERE / "stage_astra_generalization.py"))


REMOTE = r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); names=('summary.json','files.json','runtime.json','source.json','transaction/seal.json')
result={}
for name in names:
    path=root/name
    if not path.exists():continue
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or path.stat().st_size>16<<20:raise ValueError('unsafe or oversized source metadata')
    raw=path.read_bytes(); result[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
summary=json.loads((root/'summary.json').read_text())
inventory=None
if sys.argv[2]=='qwen-recovery':
    files={}; total=0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():raise ValueError('source symlink refused')
        if not path.is_file():continue
        before=path.stat()
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source changed during inventory')
        files[path.relative_to(root).as_posix()]=dict(bytes=after.st_size,sha256=digest);total+=after.st_size
    inventory=dict(schema='mdb-qwen-spectral-source-inventory-v1',job_id=summary['job_id'],request_sha256=summary['request_sha256'],files=files,expanded_bytes=total)
print(json.dumps(dict(files=result,job_id=summary['job_id'],request_sha256=summary['request_sha256'],inventory=inventory),sort_keys=True))
'''


def stage(panel, run, source):
    from mdb_lanes.job_support import one_lease_scheduling_contract, request_sha256, scheduler_api
    from mdb_lanes.payload import stage_worker

    run, source = Path(run).resolve(), Path(source).resolve(); parent = read(source / "request.json")
    if identity(parent) != parent["request_sha256"]:
        raise ValueError("original request changed")
    os.environ["MRUN_URL"] = parent["scheduler_url"]
    controller = runtime(); controller["bind"](parent["execution_environment"])
    for name, expected in {**parent["source_hashes"], **parent["runtime_wheels"]}.items():
        if sha(source / "payload" / name) != expected:
            raise ValueError("original staged source changed: " + name)
    jid = read(source / "submission.json")["job_id"]; api = scheduler_api()
    status, raw, _ = api.request("GET", "/api/jobs/" + jid, timeout_s=30); job = json.loads(raw)
    expected_state = "failed" if panel == "qwen-recovery" else "succeeded"
    if status != 200 or job["state"] != expected_state or job["assigned_host"] != "beast":
        raise ValueError("named terminal source job required")
    if (job["config"].get("saturn_request_sha256") != request_sha256(parent)
            or job["result"]["diagnostics"]["command"] != read(source / "admission-command.json")["command"]):
        raise ValueError("original command/request custody differs")
    status, log, _ = api.request("GET", "/api/jobs/" + jid + "/logs?offset=0&wait_s=1", timeout_s=30)
    if status != 200 or len(log) > 16 << 20:
        raise ValueError("bounded original log required")
    declared_log = job["result"].get("artifacts", {}).get("job_log")
    if declared_log and (len(log) != declared_log["bytes"] or hashlib.sha256(log).hexdigest() != declared_log["sha256"]):
        raise ValueError("original log checksum differs")
    root = Path(parent["config"]["output_scratch"]) / jid / parent["config"]["output_name"]
    info = json.loads(subprocess.run(["ssh", "beast", shlex.join(["python3", "-I", "-c", REMOTE, str(root), panel])], check=True, capture_output=True, text=True).stdout)
    if info["job_id"] != jid or info["request_sha256"] != parent["request_sha256"]:
        raise ValueError("original completed science summary identity differs")
    publications = [json.loads(row.removeprefix("MDB_DIRECTORY_READY ")) for row in log.decode().splitlines() if row.startswith("MDB_DIRECTORY_READY ")]
    if panel == "flux-audit":
        if len(publications) != 1 or publications[0]["files_manifest_sha256"] != info["files"]["files.json"]["sha256"]:
            raise ValueError("one source-native publication required")
        publication = publications[0]
        if publication["request_sha256"] != parent["request_sha256"] or publication["job_id"] != jid or publication["output_root"] != str(root):
            raise ValueError("native publication source differs")
    else:
        if publications or "torch.torch_version.TorchVersion" not in log.decode() or "publish_accounts" not in log.decode():
            raise ValueError("only the diagnosed post-science Qwen publisher failure is recoverable here")
        publication = None
    run.mkdir(parents=True, exist_ok=False)
    save(run / "source-job.json", job); save(run / "source-metadata.json", {key: value for key, value in info.items() if key != "inventory"})
    save(run / "source-request.json", parent); save(run / "source-publication.json", publication)
    (run / "source-worker.log").write_bytes(log)
    if info["inventory"] is not None:
        save(run / "source-inventory.json", info["inventory"])
    worker_path = HERE / WORKERS[panel]; worker = runpy.run_path(str(worker_path))
    base = dict(native_root=str(root), native_job_id=jid, source_job=jid, native_request_sha256=parent["request_sha256"],
        native_manifest_sha256=info["files"]["files.json"]["sha256"] if "files.json" in info["files"] else None,
        native_summary_sha256=info["files"]["summary.json"]["sha256"], output_scratch=str(root.parents[1]),
        native_publication_id=None if publication is None else hashlib.sha256(json.dumps(publication, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        native_publication=publication)
    base["publication_id"] = base["native_publication_id"]
    if panel == "qwen-recovery":
        base.update(source_job="source-job.json", source_job_sha256=sha(run / "source-job.json"),
            source_worker_log="source-worker.log", source_worker_log_sha256=sha(run / "source-worker.log"),
            source_inventory="source-inventory.json", source_inventory_sha256=sha(run / "source-inventory.json"))
    config = worker["make_config"](base); counts = worker["validate_config"](config)
    sources = {WORKERS[panel]: worker_path}
    for name, expected in dict(worker.get("HELPERS", {})).items():
        candidate = source / "payload" / name
        if not candidate.exists():
            candidate = HERE / name
        if sha(candidate) != expected:
            raise ValueError("saved analysis helper changed: " + name)
        sources[name] = candidate
    for name in ("source-job.json", "source-metadata.json", "source-request.json", "source-publication.json", "source-worker.log", "source-inventory.json"):
        if not (run / name).exists():
            continue
        sources[name] = run / name
    save(run / "plan.json", config); sources["plan.json"] = run / "plan.json"
    save(run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=[name for name in sources if name.endswith(".py")],
        path_token_allowlist=[str(root), config["output_scratch"], "/mnt/big/llm-models/FLUX.2-klein-4B"]))
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-astra-saved-analysis-request-v1", panel=panel, config=config,
        scheduler_url=parent["scheduler_url"], execution_environment=parent["execution_environment"], runtime_wheels=parent["runtime_wheels"],
        source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
        source_job_id=jid, source_request_sha256=parent["request_sha256"], source_terminal_state=expected_state,
        needs={"cpu": True, "host": "beast"}, backend="numpy", device="cpu", dtype="float64", task_family="analysis",
        timeout_s=900, resource_exception=dict(ram_mb=4096, vram_mb=0, cpu_threads=4, disk_gb=2, est_wall_s=300),
        scheduling_contract=one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=128,
            logical_branches=1, logical_checkpoint_replays=0, max_local_work_items=256,
            extra={"task_family": "analysis"}, checkpoint_policy="Saved-only original bytes, zero learned/model/native forwards, no retries; compact reports and metadata only."))
    request["request_sha256"] = identity(request); save(run / "request.json", request)
    stage_worker(run / "payload", worker=worker_path, config=config, request=request,
                 additional_files=[(path, name) for name, path in sources.items() if name != WORKERS[panel]])
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("staged saved-analysis source differs")
    return dict(request_sha256=request["request_sha256"], source_job=jid, planned=counts)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    run = Path(run).resolve(); request = read(run / "request.json")
    if identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"] or (run / "submission.json").exists():
        raise ValueError("saved analysis changed or already submitted")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("saved analysis payload changed: " + name)
    os.environ["MRUN_URL"] = request["scheduler_url"]; runtime()["bind"](request["execution_environment"])
    args = worker_command(WORKERS[request["panel"]]); args.insert(1, "CUDA_VISIBLE_DEVICES=")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        args.insert(1, name + "=4")
    args.insert(args.index("python") + 1, "-B"); args.extend(["--config", "saturn_run_config.json"])
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"]))
    jid = guarded_mrun_launch(args, request=request, experiment="mdb-astra-" + request["panel"], task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=False, prefer_host="beast", timeout_s=request["timeout_s"],
        retry_on_kill=False, preflight=True, **request["resource_exception"],
        note="Saved-only Astra computation account: verify original sealed predictions, native arrays, consumed ports and counterexamples. CPU4/RAM4GiB,300s estimate/900s outer;zero model/native forwards, no parameter bank, no retries or nested jobs. Compact evidence only.")
    save(run / "submission.json", dict(job_id=str(jid), request_sha256=request["request_sha256"]))
    return dict(job_id=str(jid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("panel", choices=WORKERS)
    parser.add_argument("run", type=Path); parser.add_argument("--source", type=Path); parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.source is None:
        parser.error("staging requires --source")
    print(json.dumps(submit(args.run) if args.submit else stage(args.panel, args.run, args.source), sort_keys=True))

"""Stage/submit one CUDA synthetic custody calibration; never load a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "run_flux_e18_custody_calibration.py"
PINS = {"benchmark_flux_e18_custody.py": "3e8b81894cde296cf5ff15509ba68c6c35ef60020b640b1a733d201d1005fd66",
        "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926"}
QUEUE_NOTE = "Synthetic E18 custody calibration,CUDA/B1 production-shaped fields only;9 genuine public StateCuts,8 timed storage/metadata steps per v1/v2 mode. Zero model,adapter,Act,receipt or checkpoint calls. Compare exact persisted bytes and separate clone/store costs; filesystem-order confound retained. CPU4/RAM2GiB/VRAM1.5GiB,disk1GB,60s estimate,300s outer,no retries. Raw synthetic arrays stay remote; collect small report only."


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def commands(request):
    from mdb.payload import worker_command

    runpy.run_path(str(HERE / "stage_qwen_block0_attention.py"))["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH", "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def stage(run, parent_run):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, parent_run = Path(run).resolve(), Path(parent_run).resolve()
    parent = read(parent_run / "request.json")
    if identity(parent) != "5579dc04e6fbcb20d675386833b4e3cc545f0f071383c2443cb72b108c260c36":
        raise ValueError("Qualified installed source cohort required")
    for name, digest in PINS.items():
        if sha(HERE / name) != digest:
            raise ValueError("Frozen source changed: " + name)
    run.mkdir(parents=True, exist_ok=False)
    config = {"device": "cuda", "profile": "production", "repeats": 2, "cpu_threads": 4,
              "output_scratch": "/mnt/big/mdb/research", "output_name": "e18-custody-calibration-result",
              "transport_limits": {"max_expanded_bytes": 1 << 30, "max_member_bytes": 64 << 20}}
    save(run / "plan.json", config)
    sources = {name: HERE / name for name in (WORKER, *PINS, Path(__file__).name, "stage_qwen_block0_attention.py")}
    sources["plan.json"] = run / "plan.json"
    save(run / "preflight.v1.json", {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [*sources, "preflight.v1.json"],
        "path_token_allowlist": [config["output_scratch"], config["output_name"]],
        "synthetic_only_no_models_or_fabricated_receipts": True})
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    request = {"schema": "mdb-e18-custody-calibration-request-v1", "config": config,
        "execution_environment": parent["execution_environment"], "scheduler_url": parent["scheduler_url"],
        "task_family": "analysis", "backend": "torch-cuda-public-statecut-store", "device": "cuda", "dtype": "bfloat16",
        "needs": {"cuda": True, "host": "beast"}, "runtime_wheels": parent["runtime_wheels"],
        "source_hashes": {name: sha(path) for name, path in sources.items()}, "controller_sha256": sha(__file__),
        "resource_exception": {"ram_mb": 2048, "vram_mb": 1536, "cpu_threads": 4, "disk_gb": 1, "est_wall_s": 60, "timeout_s": 300,
            "calibration": "Synthetic production geometry9cuts,8steps/mode, no real model; unmeasured CUDA storage overhead. Tiny CPU equivalence qualified; measure clone/store costs and peaks without neural forwards."},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True,
            logical_operations=16, logical_branches=2, max_local_work_items=32,
            extra={"task_family": "analysis", "synthetic_public_StateCuts": 9, "neural_forwards": 0, "Session_receipts": 0},
            checkpoint_policy="One bounded synthetic custody timing experiment; no checkpoint, adapter, Act or real transition claims; no retries or nested jobs.")}
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    commands(request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
        additional_files=[(path, name) for name, path in sources.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged closure changed: " + name)
    save(run / "staged-command.json", {"command": commands(request), "request_sha256": request["request_sha256"], "submitted": False})
    return {"request_sha256": request["request_sha256"], "source_files": len(sources), "submitted": False}


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    if (run / "submission.json").exists() or identity(request) != request["request_sha256"] or sha(__file__) != request["controller_sha256"]:
        raise ValueError("Existing submission or changed request/controller")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged closure changed: " + name)
    if len(QUEUE_NOTE) > 500:
        raise ValueError("Queue note exceeds scheduler ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = commands(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"]})
    resources = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-e18-custody-calibration", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.parent_run is None:
        parser.error("--parent-run required to stage")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.parent_run), sort_keys=True))

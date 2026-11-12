"""Stage one finite lease with sealed binding and fixed-payload receiver futures."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_lifecycle_campaign.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({k: v for k, v in request.items() if k != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prior_launcher():
    return runpy.run_path(str(HERE / "stage_common_differential.py"))


def command(request):
    from mdb.payload import worker_command
    prior_launcher()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    args.extend(["--config", "saturn_run_config.json"])
    for key, name in dict(tmpdir="TMPDIR", cuda_cache_path="CUDA_CACHE_PATH", torchinductor_cache_dir="TORCHINDUCTOR_CACHE_DIR").items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        args.insert(1, name + "=4")
    return args


def validate(run, request):
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("request or launcher changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if Path(name).name != name or sha(run / "payload" / name) != expected:
            raise ValueError("staged source/wheel changed: " + name)


def stage(run, parent_run, binding_plan_path):
    from mdb.job_support import one_lease_scheduling_contract, request_sha256
    from mdb.payload import stage_worker

    run, parent_run, binding_plan_path = (Path(p).resolve() for p in (run, parent_run, binding_plan_path))
    parent = read(parent_run / "request.json")
    worker = runpy.run_path(str(HERE / WORKER))
    if identity(parent) != worker["SOURCE_CD"]["request_sha256"] or parent["source_hashes"].get(worker["COMMON"]) != worker["COMMON_SHA256"]:
        raise ValueError("qualified C/D parent required")
    job = read(parent_run / "job.json")
    pub = read(parent_run / "publication.json")
    if (job["job_id"] != worker["SOURCE_CD"]["job_id"] or job["state"] != "succeeded"
            or job["config"]["saturn_request_sha256"] != request_sha256(parent)
            or job["result"]["diagnostics"]["command"] != read(parent_run / "admission-command.json")["command"]
            or pub["output_root"] != worker["SOURCE_CD"]["root"]
            or pub["files_manifest_sha256"] != worker["SOURCE_CD"]["manifest_sha256"]):
        raise ValueError("qualified source execution/custody differs")
    for name, expected in {**parent["source_hashes"], **parent["runtime_wheels"]}.items():
        if sha(parent_run / "payload" / name) != expected:
            raise ValueError("qualified staged source changed: " + name)
    config = worker["make_config"](parent["config"], read(binding_plan_path))
    common, _, old = worker["common_helpers"]()
    sources = {WORKER: HERE / WORKER, worker["COMMON"]: HERE / worker["COMMON"],
        "flux_binding_support_sources.py": HERE / "flux_binding_support_sources.py"}
    for name, digest in common["HELPERS"].items():
        if sha(HERE / name) != digest:
            raise ValueError("qualified helper changed: " + name)
        sources[name] = HERE / name
    for key in old["_source_keys"]():
        name = config[key]
        if Path(name).name != name:
            raise ValueError("flat source metadata required")
        path = parent_run / "payload" / name
        if sha(path) != parent["source_hashes"][name]:
            raise ValueError("original metadata changed: " + name)
        sources[name] = path
        config[key + "_sha256"] = sha(path)
    local = dict(config)
    for key in old["_source_keys"]():
        local[key] = str(sources[config[key]])
    counts = worker["validate_config"](local)
    run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(binding_plan_path, run / "binding-prior-plan.json")
    save(run / "plan.json", config)
    save(run / "count-contract.json", counts)
    for name in ("binding-prior-plan.json", "plan.json", "count-contract.json"):
        sources[name] = run / name
    roots = {v for k, v in config.items() if isinstance(v, str) and (k.endswith(("_root", "_path")) or k == "output_scratch")}
    roots.add(config["cd_source"]["root"])
    save(run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=list(sources),
        path_token_allowlist=sorted(roots), no_new_native_targets_before_global_seal=True,
        no_persistent_parameter_payload_exports=True))
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-flux-lifecycle-request-v1", panel="binding-support-and-fixed-payload-receivers", config=config,
        execution_environment=parent["execution_environment"], scheduler_url=parent["scheduler_url"], runtime_wheels=parent["runtime_wheels"],
        source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
        prior_binding_plan_sha256=sha(binding_plan_path), source_request_sha256=parent["request_sha256"],
        model="flux2-klein-4b-distilled", backend="torch-native-lifecycle", device="cuda", dtype="bfloat16",
        needs={"cuda": True, "host": "beast"}, task_family="analysis", height=512, width=512, steps=4, flux_mode="capture", capture_sites=True,
        workload_geometry={**counts, "peak_physical_batch": 1, "image_tokens": 1024, "text_tokens": 512,
            "output_bytes_bound": config["transport_limits"]["max_expanded_bytes"]},
        resource_exception=dict(ram_mb=16384, vram_mb=13000, cpu_threads=4, disk_gb=48, est_wall_s=660, timeout_s=1800,
            calibration="Prior2component job-58dbdc76f751 measuredRSS10977MB/VRAM8242MB/474.38s for1760transitions perlane and34decodes. Priorjob-14809ca3fbdc measured561.06s/RSS20531MB/VRAM8370MB including Qwen3rdcomponent+newprefix. Thisrun omitsQwen/prefix,has1964transitions/lane,52decodes,and fullnativeVAEidentity;16GiBRAM/13000MBVRAM measuredexception. Expected8-11min,finite1500sworker/1800souter;hostpinforimmutableoriginalStateCuts/checkpoints."),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=26, logical_operations=120,
            logical_checkpoint_replays=26, max_local_work_items=256,
            extra={"task_family": "analysis", "component_hydrations": 2, "count_contract": counts},
            checkpoint_policy="One finite CUDA B1 lease,immutable observed parents,in-process forks/rewinds;all26 ownstateforecasts sealed before native targets;no nested submissions/retries."))
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    prior_launcher()["bind"](request["execution_environment"])
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
        additional_files=[(path, name) for name, path in sources.items() if name != WORKER])
    validate(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"], submitted=False))
    return dict(request_sha256=request["request_sha256"], counts=counts, source_files=len(sources), submitted=False)


def submit(run):
    from mdb.job_support import guarded_mrun_launch
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate(run, request)
    if (run / "submission.json").exists():
        raise ValueError("finite campaign already submitted")
    review = read(run / "finite-run-review-v1.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("request-bound readiness review required")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resources = {k: v for k, v in request["resource_exception"].items() if k != "calibration"}
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-lifecycle", task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=True, pin="beast", retry_on_kill=False, preflight=True,
        note="Binding4contexts×4exactrowSETarms+fixed613C/D/B×cone/fox5arms=26futures(15known11new).1964F+1964Nadvances,1774nativeblocks,52VAEs. Globalfullstate/imageforecastsealfirst. Noencoder/noise/priming/retries/nesting.2originalcomponents,1GPUatonce,8-11minestimate;1500sworker/1800slease.", **resources)
    save(run / "submission.json", dict(job_id=str(job), request_sha256=request["request_sha256"]))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--binding-plan", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and (not args.parent_run or not args.binding_plan):
        parser.error("stage requires --parent-run and --binding-plan")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.parent_run, args.binding_plan), sort_keys=True))

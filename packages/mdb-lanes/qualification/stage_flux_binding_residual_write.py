"""Admit one sealed residual-versus-net-write binding campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_binding_residual_write.py"


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


def environment_helpers():
    return runpy.run_path(str(HERE / "stage_common_differential.py"))


def command(request):
    from mdb_lanes.payload import worker_command

    environment_helpers()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    args.extend(["--config", "saturn_run_config.json"])
    for key, name in dict(tmpdir="TMPDIR", cuda_cache_path="CUDA_CACHE_PATH",
                          torchinductor_cache_dir="TORCHINDUCTOR_CACHE_DIR").items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        args.insert(1, name + "=4")
    return args


def validate(run, request):
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("request or launcher changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if Path(name).name != name or sha(run / "payload" / name) != expected:
            raise ValueError("staged bytes changed: " + name)


def stage(run, parent_run):
    from mdb_lanes.job_support import one_lease_scheduling_contract, request_sha256
    from mdb_lanes.payload import stage_worker

    run, parent_run = Path(run).resolve(), Path(parent_run).resolve()
    parent = read(parent_run / "request.json")
    worker = runpy.run_path(str(HERE / WORKER))
    source = worker["SOURCE"]
    if identity(parent) != source["request_sha256"]:
        raise ValueError("exact completed lifecycle parent required")
    job, publication = read(parent_run / "job.json"), read(parent_run / "publication.json")
    if (job["job_id"] != source["job_id"] or job["state"] != "succeeded" or job["assigned_host"] != "beast"
            or job["config"]["saturn_request_sha256"] != request_sha256(parent)
            or job["result"]["diagnostics"]["command"] != read(parent_run / "admission-command.json")["command"]
            or publication["job_id"] != source["job_id"] or publication["request_sha256"] != source["request_sha256"]
            or publication["output_root"] != source["root"]
            or publication["files_manifest_sha256"] != source["manifest_sha256"]):
        raise ValueError("source execution or publication custody differs")
    for name, expected in {**parent["source_hashes"], **parent["runtime_wheels"]}.items():
        if sha(parent_run / "payload" / name) != expected:
            raise ValueError("parent staged bytes changed: " + name)
    config = worker["make_config"](parent["config"])
    old = worker["common_helpers"]()[-1]
    sources = {WORKER: HERE / WORKER}
    for name, expected in worker["HELPERS"].items():
        path = HERE / name
        if Path(name).name != name or sha(path) != expected:
            raise ValueError("frozen helper changed: " + name)
        sources[name] = path
    for key in old["_source_keys"]():
        name = config[key]
        if Path(name).name != name:
            raise ValueError("flat metadata path required")
        path = parent_run / "payload" / name
        if sha(path) != parent["source_hashes"][name]:
            raise ValueError("original source metadata changed: " + name)
        sources[name] = path
        config[key + "_sha256"] = sha(path)
    local = dict(config)
    for key in old["_source_keys"]():
        local[key] = str(sources[config[key]])
    counts = worker["validate_config"](local)
    run.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan.json", config), ("count-contract.json", counts),
                        ("source-job.json", job), ("source-publication.json", publication)):
        save(run / name, value)
        sources[name] = run / name
    roots = {value for key, value in config.items() if isinstance(value, str)
             and (key.endswith(("_root", "_path")) or key == "output_scratch")}
    roots.add(source["root"])
    save(run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=list(sources),
        path_token_allowlist=sorted(roots), all_new_native_targets_after_global_seal=True,
        no_persistent_parameter_payload_exports=True))
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-binding-residual-write-request-v1", panel="inherited-residual-versus-finite-net-write",
        config=config, execution_environment=parent["execution_environment"], scheduler_url=parent["scheduler_url"],
        runtime_wheels=parent["runtime_wheels"], source_hashes={n: sha(p) for n, p in sources.items()},
        launcher_sha256=sha(__file__), source_request_sha256=parent["request_sha256"],
        model="flux2-klein-4b-distilled", backend="torch-native-binding-residual-write", device="cuda", dtype="bfloat16",
        needs={"cuda": True, "host": "beast"}, task_family="analysis", height=512, width=512, steps=4,
        flux_mode="capture", capture_sites=True,
        workload_geometry={**counts, "peak_physical_batch": 1, "image_tokens": 1024, "text_tokens": 512,
                           "output_bytes_bound": config["transport_limits"]["max_expanded_bytes"]},
        resource_exception=dict(ram_mb=16384, vram_mb=13000, cpu_threads=4, disk_gb=24,
            est_wall_s=360, timeout_s=1200,
            calibration="Lifecycle job-b0941d817597 measured493.43scheduler seconds,11175MB RSS,8274MB VRAM with the same2components/fullVAE/loader profile and1964advances+26decodes perlane. This assay has432advances+8decodes perlane,source-onlyFP64 arithmetic and no encoder/prefix. Same16GiBRAM/13GBVRAM calibrated envelope;expected4-6min,900sworker/1200souter. Hostpin is required by immutable sourceStateCuts/checkpoint paths."),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=8, logical_operations=40,
            logical_checkpoint_replays=8, max_local_work_items=128,
            extra={"task_family": "analysis", "component_hydrations": 2, "count_contract": counts},
            checkpoint_policy="One finite CUDA B1 lease;8own-state futures globallysealed before native targets;in-process publicStateCut forks/Acts/rewinds;no encoder/noise/sourceprefix,nestedjobs or retries."))
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    environment_helpers()["bind"](request["execution_environment"])
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in sources.items() if name != WORKER])
    validate(run, request)
    save(run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"], submitted=False))
    return dict(request_sha256=request["request_sha256"], source_files=len(sources), counts=counts, submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate(run, request)
    if (run / "submission.json").exists():
        raise ValueError("finite assay already submitted")
    review = read(run / "finite-run-review-v1.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("request-bound readiness review required")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"],
                                              review_sha256=sha(run / "finite-run-review-v1.json")))
    resources = {k: v for k, v in request["resource_exception"].items() if k != "calibration"}
    job = guarded_mrun_launch(args, request=request, experiment="mdb-binding-residual-write", task="analysis", model=None,
        payload=run / "payload", needs=request["needs"], gpu=True, pin="beast", retry_on_kill=False, preflight=True,
        note="Binding anchor+wording x Q/residual-only/net-write-only/both=8futures(4known4new).432F+432Nadvances,392nativeblocks,16VAEs,16Acts/lane. DirectactualCPU FP64toBF16 endpointcast;fixedQimage/nontext;allfullimageforecastssealedfirst.2originalcomponents,max1GPU,4-6minestimate,900sworker/1200slease,no newencoder/noise/sourceprefix/retries/nesting.", **resources)
    save(run / "submission.json", dict(job_id=str(job), request_sha256=request["request_sha256"]))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and not args.parent_run:
        parser.error("stage requires --parent-run")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.parent_run), sort_keys=True))

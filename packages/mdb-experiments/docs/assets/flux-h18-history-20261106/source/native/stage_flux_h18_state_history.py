"""Stage metadata for one reviewed H18 history assay; submit is explicit.

No model, tokenizer, checkpoint/archive, remote host or scheduler is contacted
by staging. Original source metadata/709 recipes and qualified wheels are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_h18_state_history.py"
WORKER_SHA256 = "d1e57904f88cc4e6ccc062a966406d7fe7e2a0dfa669ad18985f640969ee2c75"
BASE = "stage_flux_e18_conditioning_act.py"
BASE_SHA256 = "c542b0576f955e10f286fb5db6fc24a4aee7f9b79ad73cc91bed108bc0b81ccf"
PARENT_REQUEST = "491d0611ac1ebbf1a6451552d93b83288713175bc0fac019a7d9251f956f3d34"
RESOURCE_JOB_SHA256 = "ba768527281dee17706f9344ce6feaf9b3f88ccd1c8d74ea311bd5e3e5c6fdf5"
RESOURCE_JOB_ID = "job-9339e3ed1472"
QUEUE_NOTE = (
    "H18 sign-history,short raw/seed277,B1/512sq/4steps:Qwen36 functional blocks(27+9),45 native(1full36+LM,9suffix);2 own-E branches/lane. "
    "224 functional+224 native FLUX,4 genuine FLUX Acts/228receipts,8 context operands,2 functional+4 native VAE. "
    "Both futures+signed vectors sealed before native;3CPU hydrations/6placements,max1GPU,2noise draws. CPU4/RAM24GiB/VRAM13GiB,disk12GB;900est/1200worker/1500outer,no retries/nesting. Qwen suffix is component execution,not Act."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def base():
    if sha(HERE / BASE) != BASE_SHA256:
        raise ValueError("Qualified E18 metadata controller differs")
    return runpy.run_path(str(HERE / BASE))


def command(request):
    from mdb.payload import worker_command

    base()["base"]()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
                      "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def resource_binding(path, parent):
    """Bind one immutable actual scheduler measurement, without fetching it."""
    if sha(path) != RESOURCE_JOB_SHA256:
        raise ValueError("Exact E18 measured source job required")
    job = read(path)
    result = job["result"]
    if job["job_id"] != RESOURCE_JOB_ID or job["state"] != "succeeded" or result["returncode"] != 0 or result["status"] != "ok":
        raise ValueError("Measured source execution did not succeed")
    if job["config"]["request_sha256"] != parent["request_sha256"] or any(
            job["config"][key] != parent[key] for key in ("config", "source_hashes", "runtime_wheels")):
        raise ValueError("Measured source job/request differs")
    records = [job["payload_custody"][key] for key in ("declared", "sealed", "executed")]
    records.append(result["executed_payload"])
    if len({(row["sha256"], row["size_bytes"]) for row in records}) != 1 or result["diagnostics"]["command"] != base()["command"](parent):
        raise ValueError("Measured source payload/command custody differs")
    return dict(job_id=job["job_id"], request_sha256=parent["request_sha256"], source_job_sha256=sha(path),
                elapsed_s=result["elapsed_s"], peak_rss_mb=result["peak_rss_mb"], peak_vram_mb=result["peak_vram_mb"],
                executed_payload={key: records[0][key] for key in ("sha256", "size_bytes")},
                scope="Measured E18 baseline only; H18 successor adds18Qwen block evaluations and named history/ports, not another hydration/placement")


def stage(run, parent_run, resource_job):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, parent_run, resource_job = (Path(path).resolve() for path in (run, parent_run, resource_job))
    parent = read(parent_run / "request.json")
    if identity(parent) != parent.get("request_sha256") or parent["request_sha256"] != PARENT_REQUEST:
        raise ValueError("Exact qualified E18 parent request required")
    e18 = base()
    if parent["launcher_sha256"] != BASE_SHA256 or parent["source_hashes"].get(e18["WORKER"]) != e18["WORKER_SHA256"]:
        raise ValueError("Original E18 source closure differs")
    measured = resource_binding(resource_job, parent)
    if sha(HERE / WORKER) != WORKER_SHA256:
        raise ValueError("Qualified H18 producer differs")
    utilities = e18["base"]()
    worker = runpy.run_path(str(HERE / WORKER))
    inherited = {**utilities["HELPERS"], "flux_raw_to_image.py": "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522",
                 e18["WORKER"]: e18["WORKER_SHA256"]}
    helpers = {**inherited, "qwen_h18_state_history.py": worker["HISTORY_SHA256"]}
    for name, digest in helpers.items():
        if sha(HERE / name) != digest or (name in inherited and parent["source_hashes"].get(name) != digest):
            raise ValueError("Original qualified graph/helper differs: " + name)
    metadata_names = (set(utilities["PATHS"].values()) - set(helpers)) | {"model_index.json", "source-metadata-receipt-v1.json"}
    for name in sorted(metadata_names | set(inherited) | {BASE, utilities["__file__"].split("/")[-1]}):
        if sha(parent_run / "payload" / name) != parent["source_hashes"][name]:
            raise ValueError("Original metadata/source payload differs: " + name)
    wheels = parent.get("runtime_wheels")
    if not isinstance(wheels, dict) or not wheels:
        raise ValueError("Qualified original wheels are required")
    for name, digest in wheels.items():
        if sha(parent_run / "payload" / name) != digest:
            raise ValueError("Original qualified wheel differs: " + name)
    run.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in sorted(metadata_names):
        shutil.copyfile(parent_run / "payload" / name, run / name)
        copied[name] = run / name
    shutil.copyfile(resource_job, run / "resource-source-job.json")
    copied["resource-source-job.json"] = run / "resource-source-job.json"
    config = worker["make_config"](parent["config"])
    config.update(utilities["PATHS"])
    config.update(helper_source_sha256=helpers, worker_sha256=WORKER_SHA256)
    config["transport_geometry"] = utilities["decoder_output_budget"](read(run / "vae-config.json"), 512, 512)
    if config["transport_geometry"]["max_member_bytes"] != config["transport_limits"]["max_member_bytes"]:
        raise ValueError("Geometry-derived NPY allowance differs")
    for key, name in utilities["PATHS"].items():
        config[key + "_sha256"] = sha(HERE / name if name in helpers else run / name)
    local = {**config, **{key: str(HERE / name if name in helpers else run / name) for key, name in utilities["PATHS"].items()}}
    counts = worker["validate_config"](local)
    recipes = e18["recipe_geometry"]({family: run / config[family + "_manifest"] for family in ("qwen", "flux", "vae")})
    provenance = dict(schema="mdb-h18-history-source-cohort-v1", parent_request_sha256=parent["request_sha256"],
        parent_request_file_sha256=sha(parent_run / "request.json"), parent_launcher_sha256=BASE_SHA256,
        source_metadata_receipt_sha256=sha(run / "source-metadata-receipt-v1.json"),
        usage="Original metadata,709recipes and qualified wheels only; no historical activations or learned weight payloads",
        original_payload_files={name: parent["source_hashes"][name] for name in sorted(metadata_names)},
        runtime_wheels=wheels, recipes=recipes, checkpoint_bytes_read=0, resource_measurement=measured)
    for name, value in (("plan.json", config), ("count-contract.json", counts), ("source-cohort.json", provenance)):
        save(run / name, value)
        copied[name] = run / name
    source_pairs = {WORKER: HERE / WORKER, **{name: HERE / name for name in helpers},
                    BASE: HERE / BASE, e18["BASE"]: HERE / e18["BASE"], Path(__file__).name: Path(__file__), **copied}
    save(run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=[*source_pairs, "preflight.v1.json"],
        path_token_allowlist=[config[key] for key in ("qwen_parameter_root", "flux_parameter_root", "vae_parameter_root", "output_scratch", "output_name")],
        no_native_targets_before_global_forecast_seal=True, no_persistent_parameter_payload_exports=True,
        genuine_public_FLUX_Acts=4, genuine_Qwen_Acts=0, expected_native_receipt_occurrences=228))
    source_pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-flux-h18-state-history-request-v1", config=config, parent_request_sha256=parent["request_sha256"],
        execution_environment=parent["execution_environment"], scheduler_url=parent["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-native-H18-state-history",
        device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True, height=512, width=512, steps=4,
        needs=dict(cuda=True, host="beast"), runtime_wheels=wheels,
        workload_geometry=dict(model_loads=1, pipeline_assemblies=1, physical_component_hydrations=3, peak_physical_batch=1, **counts),
        source_hashes={name: sha(path) for name, path in source_pairs.items()}, launcher_sha256=sha(__file__), base_launcher_sha256=BASE_SHA256,
        resource_exception=dict(ram_mb=24576, vram_mb=13312, cpu_threads=4, disk_gb=12, est_wall_s=900, timeout_s=1500,
            measurement=measured, named_history_and_port_payload_estimate_bytes=580911104,
            justification="Actual E18 sourcejob545.01s/RSS20462.1MB/VRAM8374MB, same3once-hydrated CPU-retained components/6placements/B1. Successor adds9functional+9nativeQwen blocks and approximately581MB named raw history/ports, not extra placements. Static payload is not peakRAM;900s expected/1200worker/1500outer remains a conservative reviewed estimate, not a measured successor runtime."),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=4, logical_operations=16, max_local_work_items=64,
            extra=dict(pipeline_assemblies=1, physical_component_hydrations=3, max_gpu_components=1, cuda_placements=6, cpu_offloads=6,
                forecast_transitions=224, native_transitions=224, genuine_FLUX_Acts=4, genuine_Qwen_Acts=0,
                receipt_occurrences=228, context_operand_occurrences=8, Qwen_functional_blocks=36, Qwen_native_blocks=45,
                Qwen_native_full_calls=1, Qwen_native_suffix_calls=9,
                logical_operations_scope="Bounded component/branch phases and4Acts; Qwen blocks, FLUX transitions and native receipts have separate censuses; no flattened460call claim.",
                model_loads_scope="One composite assembly,3once-hydrated CPU-retained components,max1GPU; no per-branch lease or model acquisition."),
            checkpoint_policy="Both own H18 histories, complete112-transition FLUX/decoder futures and FP64coherent-minus-direct vectors globally sealed before native learned targets. Native Qwen suffix is source-qualified component execution, not an Act. Four genuine FLUX conditioning Acts use own-lane E only; no refresh/retry/nested submission."))
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in source_pairs.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **wheels}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged source/wheel closure differs: " + name)
    save(run / "staged-command.json", dict(request_sha256=request["request_sha256"], command=command(request), submitted=False))
    return dict(request_sha256=request["request_sha256"], source_files=len(source_pairs), recipe_tensors=709, checkpoint_bytes_read=0, submitted=False)


def inspect_run(run):
    run = Path(run).resolve()
    request = read(run / "request.json")
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Request/launcher identity differs")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Payload source/wheel differs: " + name)
    return dict(request_sha256=request["request_sha256"], source_files=len(request["source_hashes"]), runtime_wheels=request["runtime_wheels"],
        counts=read(run / "payload/count-contract.json"), resources=request["resource_exception"],
        submitted=(run / "submission.json").exists(), finite_review_present=(run / "finite-run-review-v1.json").is_file())


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    inspected, request = inspect_run(run), read(run / "request.json")
    if inspected["submitted"]:
        raise ValueError("Original submission already exists")
    review = read(run / "finite-run-review-v1.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("Exact finite resource/source review required")
    if len(QUEUE_NOTE) > 500:
        raise ValueError("Queue note exceeds scheduler ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args, resources = command(request), request["resource_exception"]
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-h18-state-history", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--resource-job", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        result = inspect_run(args.run)
    elif args.submit:
        result = submit(args.run)
    else:
        if args.parent_run is None or args.resource_job is None:
            parser.error("--parent-run and --resource-job are required to stage")
        result = stage(args.run, args.parent_run, args.resource_job)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

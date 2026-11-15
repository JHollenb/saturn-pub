"""Prepare, inspect or explicitly submit one reviewed direct-E18 Act assay.

Staging reads source/recipe metadata and qualified wheels only. It does not
hydrate parameters, tokenize, contact the host or execute any model.
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
WORKER = "flux_e18_conditioning_act.py"
WORKER_SHA256 = "45c512de2dff3252c38a182cf4291b392cf35a159c2bd83b35900c28005cb343"
BASE = "stage_flux_raw_to_image.py"
BASE_SHA256 = "855ee39d6d4c9c4a9f8fcefb425c33d8ff327b9f24a0f9dbf3066c8389040e99"
CALIBRATION_RECEIPT_SHA256 = "6fab331138fe042fac47de734b5495ca67782304d2edcfa44c6a1569332c414f"
PARENTS = {
    "982cbb95829fe1eb77c444e1beaa424e5535c18e4ec8bcb7ca6690e61d8d330e":
        ("flux_raw_to_image.py", "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522"),
    "783e8dc0c0a731c458eed739c5d8b68c932bf622ef5f426d7f484596a8c011dd":
        ("flux_raw_mask_pair.py", "3ee62e9174186935b32ff6e9e33e18e46aa978c4442b19f02982919e63bf3912"),
}
QUEUE_NOTE = (
    "Direct E18 conditioning Act,short raw/seed277,B1/512sq/4steps:1 functionalQwen27+1 nativeQwen36/LM;2 own-E branches/lane;224 functional+224 native FLUX,4 real Acts/228receipts,8 context operands,2 functional+4 native VAE. "
    "3CPU hydrations/6placements,max1GPU,2noise draws. Both forecasts+signed responses sealed before native learned targets. CPU4/RAM24GiB/VRAM13GiB,disk12GiB;900s estimate/1200worker/1500outer,no retries/nesting."
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
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def base():
    if sha(HERE / BASE) != BASE_SHA256:
        raise ValueError("Qualified metadata/launch utility differs")
    return runpy.run_path(str(HERE / BASE))


def command(request):
    from mdb.payload import worker_command

    base()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
                      "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def recipe_geometry(paths):
    """Validate original retrieval metadata, never open a checkpoint/archive."""
    from mdb.parameter_artifacts import validate_manifest

    result = {}
    for family, count in (("qwen", 398), ("flux", 169), ("vae", 142)):
        manifest = read(paths[family])
        validate_manifest(manifest)
        rows = manifest["artifacts"]
        names = {row["source"]["tensor"] for row in rows}
        if len(rows) != count or len(names) != count:
            raise ValueError("Original recipe tensor census differs: " + family)
        files = {}
        for row in rows:
            source = row["source"]
            previous = files.setdefault(source["file"], source["file_sha256"])
            if previous != source["file_sha256"]:
                raise ValueError("Recipe source file has conflicting hashes")
        result[family] = dict(tensors=count, manifest_sha256=sha(paths[family]),
                              payload_bytes=sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in rows),
                              source_checkpoints=files, checkpoint_bytes_read=0)
    return result


def stage(run, parent_run, custody_calibration):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, parent_run = Path(run).resolve(), Path(parent_run).resolve()
    custody_calibration = Path(custody_calibration).resolve()
    if sha(custody_calibration) != CALIBRATION_RECEIPT_SHA256:
        raise ValueError("Exact authenticated synthetic custody measurement required")
    calibration = read(custody_calibration)
    parent = read(parent_run / "request.json")
    parent_id = parent.get("request_sha256")
    if identity(parent) != parent_id or parent_id not in PARENTS:
        raise ValueError("Exact qualified original raw or paired metadata request required")
    parent_worker, parent_worker_sha = PARENTS[parent_id]
    if parent["source_hashes"].get(parent_worker) != parent_worker_sha:
        raise ValueError("Original metadata producer source identity differs")
    if sha(HERE / WORKER) != WORKER_SHA256:
        raise ValueError("Qualified E18 worker differs")
    utilities = base()
    worker = runpy.run_path(str(HERE / WORKER))
    helpers = {**utilities["HELPERS"], worker["PRIOR"]: worker["PRIOR_SHA256"]}
    for name, digest in helpers.items():
        if sha(HERE / name) != digest or parent["source_hashes"].get(name) != digest:
            raise ValueError("Original qualified helper closure differs: " + name)
    # The parent is used only as a source/config/recipe cohort, never as a claim
    # that its original failed exporter job succeeded or as activation evidence.
    metadata_names = set(utilities["PATHS"].values()) - set(helpers)
    metadata_names.update({"model_index.json", "source-metadata-receipt-v1.json"})
    for name in sorted(metadata_names | {parent_worker} | set(helpers)):
        if sha(parent_run / "payload" / name) != parent["source_hashes"][name]:
            raise ValueError("Original metadata/source payload differs: " + name)
    wheels = parent.get("runtime_wheels")
    if not isinstance(wheels, dict) or not wheels:
        raise ValueError("Qualified original wheel cohort is required")
    for name, digest in wheels.items():
        if sha(parent_run / "payload" / name) != digest:
            raise ValueError("Original qualified wheel differs: " + name)
    run.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in sorted(metadata_names):
        shutil.copyfile(parent_run / "payload" / name, run / name)
        copied[name] = run / name
    shutil.copyfile(custody_calibration, run / "custody-calibration-receipt.json")
    copied["custody-calibration-receipt.json"] = run / "custody-calibration-receipt.json"
    config = worker["make_config"](parent["config"])
    config.update(utilities["PATHS"])
    config["helper_source_sha256"] = helpers
    config["worker_sha256"] = WORKER_SHA256
    config["transport_geometry"] = utilities["decoder_output_budget"](read(run / "vae-config.json"), 512, 512)
    if config["transport_geometry"]["max_member_bytes"] != config["transport_limits"]["max_member_bytes"]:
        raise ValueError("Geometry-derived NPY header allowance differs")
    for key, name in utilities["PATHS"].items():
        config[key + "_sha256"] = sha(HERE / name if name in helpers else run / name)
    local = {**config, **{key: str(HERE / name if name in helpers else run / name) for key, name in utilities["PATHS"].items()}}
    counts = worker["validate_config"](local)
    geometry = recipe_geometry({family: run / config[family + "_manifest"] for family in ("qwen", "flux", "vae")})
    provenance = dict(schema="mdb-e18-source-cohort-v1", parent_request_sha256=parent_id,
                      parent_request_file_sha256=sha(parent_run / "request.json"),
                      source_metadata_receipt_sha256=sha(run / "source-metadata-receipt-v1.json"),
                      usage="Source/config/709recipe metadata and exact wheel cohort only; no historical activations or claimed parent-job success",
                      original_payload_files={name: parent["source_hashes"][name] for name in sorted(metadata_names)},
                      runtime_wheels=wheels, recipes=geometry, checkpoint_bytes_read=0)
    provenance["custody_calibration_receipt_sha256"] = CALIBRATION_RECEIPT_SHA256
    for name, value in (("plan.json", config), ("count-contract.json", counts), ("source-cohort.json", provenance)):
        save(run / name, value)
        copied[name] = run / name
    source_pairs = {WORKER: HERE / WORKER, **{name: HERE / name for name in helpers},
                    BASE: HERE / BASE, Path(__file__).name: Path(__file__), **copied}
    preflight = dict(schema="mrun-preflight-config-v1", hashed_roots=[*source_pairs, "preflight.v1.json"],
                     path_token_allowlist=[config[key] for key in ("qwen_parameter_root", "flux_parameter_root", "vae_parameter_root", "output_scratch", "output_name")],
                     no_native_targets_before_global_forecast_seal=True, no_persistent_parameter_payload_exports=True,
                     genuine_public_conditioning_Acts=4, expected_native_receipt_occurrences=228)
    save(run / "preflight.v1.json", preflight)
    source_pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-flux-e18-conditioning-request-v1", config=config, parent_request_sha256=parent_id,
        execution_environment=parent["execution_environment"], scheduler_url=parent["scheduler_url"],
        model="flux2-klein-4b-distilled", task_family="analysis", backend="torch-native-E18-conditioning-act",
        device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True, height=512, width=512, steps=4,
        needs={"cuda": True, "host": "beast"}, runtime_wheels=wheels,
        workload_geometry=dict(model_loads=1, pipeline_assemblies=1, physical_component_hydrations=3,
                               peak_physical_batch=1, **counts),
        source_hashes={name: sha(path) for name, path in source_pairs.items()}, launcher_sha256=sha(__file__),
        base_launcher_sha256=BASE_SHA256,
        resource_exception=dict(ram_mb=24576, vram_mb=13312, cpu_threads=4, disk_gb=12, est_wall_s=900, timeout_s=1500,
            calibration_receipt_sha256=CALIBRATION_RECEIPT_SHA256, calibration_job_id=calibration["job_id"],
            calibration="Original raw single329.23s/RSS20815.5MB/VRAM8374MB; same3CPU-retained components/6placements/B1, now2denoiser paths per lane with real Session hashing/capture. Actual synthetic CUDA custody job-d1af1e9981fd29.02s/RSS869.8/VRAM450MB:9cuts/8timed steps, exact67,073,072B stores, build0.3648s; v1metadata0.00459/store1.1432s17calls,v2metadata0.0000709/store0.75376s9calls. Modest copy/store overhead only: excludes Session guards/neural arithmetic; v1-to-v2 cache-order confounded. No full E18 runtime or speedup claim;900s conservative estimate/1200s cooperative/1500s outer subject to exact independent review."),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=4,
            logical_operations=16, max_local_work_items=64,
            extra=dict(pipeline_assemblies=1, physical_component_hydrations=3,
                max_gpu_components=1, cuda_placements=6, cpu_offloads=6, forecast_transitions=224, native_transitions=224,
                genuine_Acts=4, receipt_occurrences=228, context_operand_occurrences=8,
                component_or_transition_or_Act_calls=460,
                logical_operations_scope="12component passes+4Acts;448denoiser transitions,2Qwen calls,6decoder calls+4Acts counted separately as460physical component/transition/action occurrences, not independent tasks.",
                model_loads_scope="One composite assembly; exactly3once-hydrated CPU-retained components, max1GPU; four B1 branch paths."),
            checkpoint_policy="Both own-E functional branches, complete pixels and signed edit-minus-sham hashes globally sealed before native Qwen. Each lane forks its own initial typed parent, applies2genuine conditioning Acts; no crossinstalled E, refresh, retry or nested submission."))
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in source_pairs.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **wheels}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged source/wheel closure differs: " + name)
    save(run / "staged-command.json", dict(request_sha256=request["request_sha256"], command=command(request), submitted=False))
    return dict(request_sha256=request["request_sha256"], source_files=len(source_pairs),
                recipe_tensors=sum(row["tensors"] for row in geometry.values()), checkpoint_bytes_read=0, submitted=False)


def inspect_run(run):
    run = Path(run).resolve()
    request = read(run / "request.json")
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Request/launcher identity differs")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Payload source/wheel differs: " + name)
    return dict(request_sha256=request["request_sha256"], source_files=len(request["source_hashes"]),
                runtime_wheels=request["runtime_wheels"], counts=read(run / "payload/count-contract.json"),
                resources=request["resource_exception"], submitted=(run / "submission.json").exists(),
                finite_review_present=(run / "finite-run-review-v1.json").is_file())


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    inspected = inspect_run(run)
    request = read(run / "request.json")
    review = read(run / "finite-run-review-v1.json")
    if inspected["submitted"]:
        raise ValueError("Original submission already exists")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("Exact finite resource/source review required")
    if len(QUEUE_NOTE) > 500:
        raise ValueError("Queue note exceeds scheduler ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    save(run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=sha(run / "finite-run-review-v1.json")))
    resources = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-e18-conditioning-act", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--custody-calibration", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        result = inspect_run(args.run)
    elif args.submit:
        result = submit(args.run)
    else:
        if args.parent_run is None or args.custody_calibration is None:
            parser.error("--parent-run and --custody-calibration are required to stage")
        result = stage(args.run, args.parent_run, args.custody_calibration)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

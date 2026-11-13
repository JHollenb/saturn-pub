"""Stage the fixed two-context raw-image cohort; submit only its reviewed bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_raw_mask_pair.py"
WORKER_SHA256 = "d5dea6923d8865d898877890a968428efaa7993055071d4dc43028a826846e34"
BASE = "stage_flux_raw_to_image.py"
BASE_SHA256 = "9afc025652193e86584b524568247d8e386c37f6aeb653236eb78f60587089e5"
PARENT_REQUEST = "982cbb95829fe1eb77c444e1beaa424e5535c18e4ec8bcb7ca6690e61d8d330e"
QUEUE_NOTE = (
    "Fixed short/long raw-context mask/GQA pair,seed277,B1/512sq/4steps: both Qwen27+112FLUX+functional VAE forecasts and signed hashes sealed before either native target. "
    "3 CPU-retained hydrations,6 placements/offloads,max1GPU;224+224FLUX,2+2Qwen,2+4VAE,4noise draws. "
    "Measured single329s/RSS20816/VRAM8374MB;pair750s estimate,1200s worker/1500s outer,RAM24GiB/VRAM13GiB/CPU4;no retries/nesting."
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
    return hashlib.sha256(json.dumps(
        {key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()


def base():
    if sha(HERE / BASE) != BASE_SHA256:
        raise ValueError("Qualified base launcher differs")
    return runpy.run_path(str(HERE / BASE))


def command(request):
    from mdb_lanes.payload import worker_command

    base()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
                      "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def stage(run, parent_run):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, parent_run = Path(run).resolve(), Path(parent_run).resolve()
    original = read(parent_run / "request.json")
    if identity(original) != original.get("request_sha256") or original["request_sha256"] != PARENT_REQUEST:
        raise ValueError("Exact original raw-input source request is required")
    if sha(HERE / WORKER) != WORKER_SHA256:
        raise ValueError("Qualified paired worker differs")
    utilities = base()
    worker = runpy.run_path(str(HERE / WORKER))
    helpers = {**utilities["HELPERS"], worker["PRIOR"]: worker["PRIOR_SHA256"]}
    for name, digest in helpers.items():
        if sha(HERE / name) != digest:
            raise ValueError("Qualified arithmetic source changed: " + name)
    run.mkdir(parents=True, exist_ok=False)
    metadata_names = set(utilities["PATHS"].values()) - set(helpers)
    metadata_names.update({"model_index.json", "source-metadata-receipt-v1.json"})
    copied = {}
    for name in sorted(metadata_names):
        source = parent_run / "payload" / name
        if sha(source) != original["source_hashes"][name]:
            raise ValueError("Original metadata closure changed: " + name)
        shutil.copyfile(source, run / name)
        copied[name] = run / name
    config = worker["make_config"](original["config"])
    config["helper_source_sha256"] = helpers
    config["worker_sha256"] = WORKER_SHA256
    config["transport_geometry"] = utilities["decoder_output_budget"](read(run / "vae-config.json"), 512, 512)
    if config["transport_geometry"]["max_member_bytes"] != config["transport_limits"]["max_member_bytes"]:
        raise ValueError("Geometry-derived NPY framing allowance differs")
    local = dict(config)
    for key, name in utilities["PATHS"].items():
        local[key] = str(HERE / name if name in helpers else run / name)
    counts = worker["validate_config"](local)
    save(run / "plan.json", config)
    save(run / "count-contract.json", counts)
    copied.update({name: run / name for name in ("plan.json", "count-contract.json")})
    preflight = {
        "schema": "mrun-preflight-config-v1",
        "hashed_roots": [WORKER, *helpers, *copied],
        "path_token_allowlist": [config[key] for key in ("qwen_parameter_root", "flux_parameter_root", "vae_parameter_root", "output_scratch", "output_name")],
        "no_native_targets_before_global_forecast_seal": True,
        "no_persistent_parameter_payload_exports": True,
    }
    save(run / "preflight.v1.json", preflight)
    copied["preflight.v1.json"] = run / "preflight.v1.json"
    source_pairs = {WORKER: HERE / WORKER, **{name: HERE / name for name in helpers}, **copied}
    request = {
        "schema": "mdb-flux-raw-mask-pair-request-v1", "config": config,
        "parent_request_sha256": PARENT_REQUEST,
        "execution_environment": original["execution_environment"],
        "scheduler_url": original["scheduler_url"],
        "model": "flux2-klein-4b-distilled", "task_family": "analysis",
        "backend": "torch-native-raw-mask-pair", "device": "cuda", "dtype": "bfloat16",
        "flux_mode": "capture", "capture_sites": True, "height": 512, "width": 512, "steps": 4,
        "needs": {"cuda": True, "host": "beast"},
        "workload_geometry": {
            "model_loads": 1, "pipeline_assemblies": 1, "physical_component_hydrations": 3,
            "peak_physical_batch": 1, "max_gpu_components": 1, "cuda_placements": 6, "cpu_offloads": 6,
            "functional_qwen_prefixes": 2, "native_full_qwen_calls": 2,
            "functional_denoiser_transitions": 224, "native_denoiser_transitions": 224,
            "functional_decoder_calls": 2, "native_vae_calls": 4, "noise_draws": 4,
            "sdpa_argument_observations": {"forecast": 54, "native": 72},
            "output_bytes_bound": 8 << 30,
        },
        "runtime_wheels": original["runtime_wheels"],
        "source_hashes": {name: sha(path) for name, path in source_pairs.items()},
        "launcher_sha256": sha(__file__), "base_launcher_sha256": BASE_SHA256,
        "resource_exception": {
            "ram_mb": 24576, "vram_mb": 13000, "cpu_threads": 4, "disk_gb": 12,
            "est_wall_s": 750, "timeout_s": 1500,
            "calibration": "One-case original job-7efe0a8100bb measured329.23s/RSS20815.5MB/VRAM8374MB. Same three retained CPU components, six placements and B1 sequential phases. Two paths and streamed FP64 response hashing, no parallel GPU forwards or persistent delta/weight bank. Paired GQA lowering/peak/runtime unmeasured;750s estimate below15min,1200s cooperative events/1500s hard outer. Observe misses and peaks.",
        },
        "scheduling_contract": one_lease_scheduling_contract(
            model_loads=1, logical_branches=4, logical_operations=14, max_local_work_items=64,
            extra={"pipeline_assemblies": 1, "physical_component_hydrations": 3,
                   "component_names": ["qwen-text-encoder", "flux-transformer", "vae-decoder"],
                   "max_gpu_components": 1, "cuda_placements": 6, "cpu_offloads": 6,
                   "forecast_transitions": 224, "native_transitions": 224,
                   "case_order": ["short", "long"],
                   "logical_operations_scope": "14 component passes:2 functional Qwen prefixes,2 full native Qwen,2 functional and2 native112-transition denoiser paths,2 functional and4 native decoder calls. The448 denoiser transitions and458 combined component/transition calls remain separately counted, not256 independent local branches.",
                   "component_or_transition_calls": 458,
                   "model_loads_scope": "One composite pipeline assembly; three once-hydrated CPU-retained components, two B1 paths per phase."},
            checkpoint_policy="All two-case own Qwen/112-transition/functional-pixel forecasts and complete signed-response commitments globally sealed before either native target; no retries or nested submissions.",
        ),
    }
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in source_pairs.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged source/wheel changed: " + name)
    save(run / "staged-command.json", {"request_sha256": request["request_sha256"],
                                       "command": command(request), "submitted": False})
    return {"request_sha256": request["request_sha256"], "source_files": len(source_pairs), "submitted": False}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    review = read(run / "finite-run-review-v1.json")
    if (run / "submission.json").exists() or identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Submitted specimen, request or launcher changed")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("Exact finite-run review required")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Reviewed payload differs: " + name)
    if len(QUEUE_NOTE) > 500:
        raise ValueError("Queue note exceeds ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    resources = request["resource_exception"]
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"],
                                          "review_sha256": sha(run / "finite-run-review-v1.json")})
    job = guarded_mrun_launch(
        args, request=request, experiment="mdb-flux-raw-mask-pair", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE,
    )
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.parent_run is None:
        parser.error("--parent-run is required to stage")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.parent_run), sort_keys=True))

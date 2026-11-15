"""Prepare one finite raw-input composition; submission stays a separate action."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_raw_to_image.py"
QWEN_PARENT_REQUEST = "aacdb95dd43145d44f4a68c292a16764c6292d0b3b32236e223b58751c4b0d51"
HELPERS = {
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "2a953ca5ae3bc507e6f79797595a0c69c68325df307a63e4d90bfe88142f62b0",
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_source_receiver.py": "77c0c717dd82afe7c039a3179bf0f34e2f44d77ed3fd14fb0c87893946c11dc2",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926",
}
PATHS = {
    "qwen_config": "qwen-config.json",
    "qwen_manifest": "qwen-parameter-artifacts.json",
    "qwen_source_binding": "qwen-source-binding.json",
    "qwen_retrieval_script": "retrieve_qwen_weights.py",
    "qwen_recipe_helper": "flux_source_receiver.py",
    "flux_config": "flux-transformer-config.json",
    "flux_scheduler_config": "scheduler-config.json",
    "flux_manifest": "flux-parameter-artifacts.json",
    "flux_retrieval_script": "retrieve_flux_weights.py",
    "flux_recipe_helper": "flux_source_receiver.py",
    "vae_config": "vae-config.json",
    "vae_manifest": "vae-parameter-artifacts.json",
    "vae_retrieval_script": "retrieve_vae_weights.py",
    "vae_recipe_helper": "flux_source_receiver.py",
}
QUEUE_NOTE = (
    "Fresh raw prompt/seed277: own Qwen27,112 FLUX transitions,functional VAE/pixels,global seal then native Qwen36+LM,112 transitions,2 native VAE calls. "
    "One pipeline assembly,3 CPU-retained component hydrations,6 CUDA placements/offloads,max1 GPU component;2 seeded noise draws. "
    "CUDA B1 512sq/4steps;RAM24GiB/VRAM13GiB/CPU4,750s estimate/1200s worker/1500s outer;no retries/nesting."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def decoder_output_budget(vae_config, height, width):
    """Bound saved BF16 decoder macros, including their NPY framing.

    The original producer retained an array with exactly 128 MiB of payload;
    its 128-byte header exceeded the original 128 MiB member allowance.
    This is the fixed B1, untiled decoder profile, not arbitrary model sizing.
    """
    channels = list(reversed(vae_config["block_out_channels"]))
    scale = 2 ** (len(channels) - 1)
    h, w = height // scale, width // scale
    payloads = [channels[0] * h * w * 2]
    for index, channel in enumerate(channels):
        if index < len(channels) - 1:
            h, w = h * 2, w * 2
        payloads.append(channel * h * w * 2)
    largest = max(payloads)
    return {"largest_tensor_payload_bytes": largest, "npy_header_allowance_bytes": 4096,
            "max_member_bytes": largest + 4096,
            "scope": "B1 BF16 retained decoder macros at fixed image geometry; archive framing is additional to tensor payload."}


def identity(request):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()


def bind(environment):
    from mdb.execution_environment import intent

    for key, name in {
        "uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
        "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
        "uv_link_mode": "MDB_WORKER_UV_LINK_MODE",
    }.items():
        os.environ[name] = environment[key]
    installed = intent()
    if installed != {key: environment[key] for key in installed}:
        raise ValueError("Installed dependency intent changed")


def command(request):
    from mdb.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {
        "tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
        "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR",
    }.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def stage(run, planning, qwen_run):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, planning, qwen_run = (Path(path).resolve() for path in (run, planning, qwen_run))
    parent = read(qwen_run / "request.json")
    if identity(parent) != parent.get("request_sha256") or parent["request_sha256"] != QWEN_PARENT_REQUEST:
        raise ValueError("The qualified run-08 Qwen parent request is required")
    metadata = read(planning / "source-metadata-receipt-v1.json")
    inputs = planning / "source-metadata"
    for name, row in metadata["source_files"].items():
        path = inputs / name
        if sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Bound source metadata changed: " + name)
    for name, digest in HELPERS.items():
        if sha(HERE / name) != digest:
            raise ValueError("Qualified arithmetic helper changed: " + name)
    run.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name in set(PATHS.values()) | {"model_index.json"}:
        if name in HELPERS:
            continue
        shutil.copyfile(inputs / name, run / name)
        copied[name] = run / name
    shutil.copyfile(planning / "source-metadata-receipt-v1.json", run / "source-metadata-receipt-v1.json")
    copied["source-metadata-receipt-v1.json"] = run / "source-metadata-receipt-v1.json"
    output_budget = decoder_output_budget(read(run / "vae-config.json"), 512, 512)
    config = {
        "schema": "mdb-flux-raw-to-image-plan-v1",
        "prompt": "A cobalt blue cube to the right of a matte orange ring on a gray tabletop. Two separate objects, simple studio product photograph.",
        "seed": 277, "batch_size": 1, "height": 512, "width": 512, "steps": 4,
        "guidance_scale": 1.0, "reference_image": False, "device": "cuda", "dtype": "bfloat16",
        "flux_mode": "capture", "capture_sites": True, "retry": False, "nested_mrun": False,
        "max_gpu_components": 1, "component_hydrations": 3, "cuda_placements": 6, "offloads": 6,
        "qwen_manifest_entries": 398, "flux_manifest_entries": 169, "vae_manifest_entries": 142,
        "qwen_parameter_root": parent["config"]["qwen_parameter_root"],
        "flux_parameter_root": parent["config"]["flux_parameter_root"],
        "vae_parameter_root": parent["config"]["flux_parameter_root"] + "/vae",
        "qwen_rope_buffer_policy": parent["config"]["rope_buffer_policy"],
        "rope_buffer_policy": parent["config"]["rope_buffer_policy"],
        "attention_dispatch": parent["config"]["attention_dispatch"],
        "apply_chat_template": parent["config"]["apply_chat_template"],
        "cohort": "prospective-new", "max_length": 512, "padding": "max_length",
        "padding_side": "right", "truncation": True, "all_512_rows_retained": True,
        "prefix_layers": 27, "full_layers": 36, "selected_hidden_states": [9, 18, 27],
        "flux_in_channels": 128, "source_reduction_policy": True, "cpu_threads": 4,
        "decoder_frontend_profile": "cpu-unpack-inverse-bn-unpatch-v1",
        "native_sources": metadata["native_sources"], "helper_source_sha256": HELPERS,
        "worker_sha256": sha(HERE / WORKER), "max_wall_seconds": 1200,
        "forecast_transaction_bytes": 4 << 30,
        "transport_limits": {"max_expanded_bytes": 4 << 30, "max_member_bytes": output_budget["max_member_bytes"]},
        "transport_geometry": output_budget,
        "output_scratch": "/mnt/big/mdb/research", "output_name": "raw-to-image-result",
        "source_metadata_receipt_sha256": sha(planning / "source-metadata-receipt-v1.json"),
        "parameter_payload_bytes": {name: row["payload_bytes"] for name, row in metadata["families"].items()},
        **PATHS,
    }
    # Retain the source-bound Qwen/tokenizer runtime contract, including the
    # explicitly runtime-observed Linux extension rather than a Darwin pin.
    config.update({key: value for key, value in parent["config"].items()
                   if key.startswith(("qwen_native_", "flux_native_", "tokenizers_", "jinja_"))
                   or key in {"transformers_version", "diffusers_version"}})
    for key, name in PATHS.items():
        config[key + "_sha256"] = sha(HERE / name if name in HELPERS else run / name)
    local = dict(config)
    for key, name in PATHS.items():
        local[key] = str(HERE / name if name in HELPERS else run / name)
    worker = runpy.run_path(str(HERE / WORKER))
    counts = worker["validate_config"](local)
    save(run / "plan.json", config)
    save(run / "count-contract.json", counts)
    copied.update({name: run / name for name in ("plan.json", "count-contract.json")})
    preflight = {
        "schema": "mrun-preflight-config-v1",
        "hashed_roots": [WORKER, *HELPERS, *copied],
        "path_token_allowlist": [config[key] for key in ("qwen_parameter_root", "flux_parameter_root", "vae_parameter_root", "output_scratch", "output_name")],
        "no_native_targets_before_global_forecast_seal": True,
        "no_persistent_parameter_payload_exports": True,
    }
    save(run / "preflight.v1.json", preflight)
    copied["preflight.v1.json"] = run / "preflight.v1.json"
    source_pairs = {WORKER: HERE / WORKER, **{name: HERE / name for name in HELPERS}, **copied}
    request = {
        "schema": "mdb-flux-raw-to-image-request-v1", "config": config,
        "execution_environment": parent["execution_environment"], "scheduler_url": parent["scheduler_url"],
        "model": "flux2-klein-4b-distilled", "task_family": "analysis", "backend": "torch-native-raw-composition",
        "device": "cuda", "dtype": "bfloat16", "flux_mode": "capture", "capture_sites": True,
        "height": 512, "width": 512, "steps": 4, "needs": {"cuda": True, "host": "beast"},
        "workload_geometry": {
            "model_loads": 1, "pipeline_assemblies": 1, "physical_component_hydrations": 3,
            "peak_physical_batch": 1, "max_gpu_components": 1, "cuda_placements": 6, "cpu_offloads": 6,
            "functional_qwen_prefixes": 1, "native_full_qwen_calls": 1,
            "functional_denoiser_transitions": 112, "native_denoiser_transitions": 112,
            "functional_decoder_calls": 1, "native_vae_calls": 2, "noise_draws": 2,
            "output_bytes_bound": 4 << 30,
        },
        "runtime_wheels": parent["runtime_wheels"],
        "source_hashes": {name: sha(path) for name, path in source_pairs.items()},
        "launcher_sha256": sha(__file__),
        "resource_exception": {
            "ram_mb": 24576, "vram_mb": 13000, "cpu_threads": 4, "disk_gb": 8,
            "est_wall_s": 750, "timeout_s": 1500,
            "calibration": "Prospective composite profile, not measured peak: 15,895,266,374B original CPU parameter payload plus bounded recipe widening/copy, runtime and captures. Qwen measured4461.6MB RSS/8790MB VRAM; prior448 denoiser transitions482.2s; this specimen224 total plus one prefix/full reference and three decoder executions. One GPU component at a time, no no-clamp/resident override. Observe peak and preserve failures.",
        },
        "scheduling_contract": one_lease_scheduling_contract(
            model_loads=1, logical_branches=2, logical_operations=229, max_local_work_items=256,
            extra={"pipeline_assemblies": 1, "physical_component_hydrations": 3,
                   "component_names": ["qwen-text-encoder", "flux-transformer", "vae-decoder"],
                   "max_gpu_components": 1, "cuda_placements": 6, "cpu_offloads": 6,
                   "forecast_transitions": 112, "native_transitions": 112,
                   "model_loads_scope": "One composite FLUX pipeline assembly; three separately counted once-hydrated components, retained on CPU."},
            checkpoint_policy="All own-state Qwen/112-transition/functional-pixel forecasts globally sealed before independent native inputs and learned targets; CPU-retained phase residency, finite no retry or nested submissions.",
        ),
    }
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in source_pairs.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != digest:
            raise ValueError("Staged source/wheel differs: " + name)
    save(run / "staged-command.json", {"request_sha256": request["request_sha256"], "command": command(request), "submitted": False})
    return {"request_sha256": request["request_sha256"], "source_files": len(source_pairs), "submitted": False}


def submit(run):
    from mdb.job_support import guarded_mrun_launch

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
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"], "review_sha256": sha(run / "finite-run-review-v1.json")})
    resources = request["resource_exception"]
    job = guarded_mrun_launch(
        args, request=request, experiment="mdb-flux-raw-to-image", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE,
    )
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--planning", type=Path)
    parser.add_argument("--qwen-run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and (args.planning is None or args.qwen_run is None):
        parser.error("--planning and --qwen-run required for stage-only action")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.planning, args.qwen_run), sort_keys=True))

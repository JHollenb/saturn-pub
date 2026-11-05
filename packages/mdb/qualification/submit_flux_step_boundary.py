"""Stage and admit one sealed remaining-step forecast and three native futures."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import runpy
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execution_intent():
    from mdb.execution_environment import intent

    result = intent()
    for key, variable, fallback in (
        ("uv_cache_dir", "MDB_WORKER_UV_CACHE_DIR", "/mnt/hdd8tb/mdb/uv-cache"),
        ("torch_home", "MDB_WORKER_TORCH_HOME", "/mnt/hdd8tb/mdb/torch"),
        ("extra_model_roots", "MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models"),
    ):
        result[key] = os.environ.get(variable, fallback)
    if any(not result[k].startswith("/") or result[k].startswith("/Users/")
           for k in ("uv_cache_dir", "torch_home")):
        raise ValueError("Linux CUDA cache paths must be worker-native and absolute")
    return result


def sealed_worker_command(request):
    """Bind actual launch arguments to staged intent despite ambient defaults."""
    from mdb.payload import worker_command

    environment = request["execution_environment"]
    if any(not environment[k].startswith("/") or environment[k].startswith("/Users/")
           for k in ("uv_cache_dir", "torch_home")):
        raise ValueError("Linux CUDA cache paths must be worker-native and absolute")
    command = worker_command("flux_step_boundary_campaign.py")
    for variable, key in (("UV_LINK_MODE", "uv_link_mode"), ("UV_CACHE_DIR", "uv_cache_dir"),
                          ("TORCH_HOME", "torch_home"), ("LLM_MODELS_EXTRA_ROOT", "extra_model_roots")):
        positions = [i for i, value in enumerate(command) if value.startswith(variable + "=")]
        if len(positions) != 1:
            raise ValueError("public worker environment ABI differs: " + variable)
        command[positions[0]] = variable + "=" + environment[key]
    command[command.index("--python") + 1] = environment["python"]
    requirements = [command[i + 1] for i, value in enumerate(command) if value == "--with"]
    if [value for value in requirements if not value.startswith("./")] != environment["requirements"]:
        raise ValueError("worker dependency requirements differ from staged intent")
    return command


def storage_plan(config):
    """Exact major member geometry; enforce total again on complete output."""
    import numpy as np

    shapes = {"fused_weight": ((27648, 3072), "<u2"), "output_weight": ((3072, 12288), "<u2"),
        "native_register": ((1, 1536, 3072), "<u2"), "predicted_register": ((1, 1536, 3072), "<f4")}
    members = {}
    for name, (shape, dtype) in shapes.items():
        header = io.BytesIO()
        value_type = np.dtype(dtype)
        np.lib.format.write_array_header_1_0(header, {"descr": np.lib.format.dtype_to_descr(value_type),
            "fortran_order": False, "shape": shape})
        members[name] = {"shape": list(shape), "dtype": str(value_type),
            "bytes": header.tell() + math.prod(shape) * value_type.itemsize, "header_bytes": header.tell()}
    lower = 14 * (members["fused_weight"]["bytes"] + members["output_weight"]["bytes"])
    forecast_outputs = 14 * members["predicted_register"]["bytes"]
    limits = config["transport_limits"]
    if (any(v["bytes"] > limits["max_member_bytes"] for v in members.values())
            or lower + forecast_outputs > limits["max_expanded_bytes"]
            or forecast_outputs > config["transaction_max_bytes"]
            or config["transaction_max_bytes"] > limits["max_member_bytes"]):
        raise ValueError("complete forecast/parameter geometry exceeds declared custody bounds")
    return {"schema": "mdb-flux-step-boundary-storage-plan-v1", "members": members,
        "parameter_bytes_lower_bound": lower, "complete_output_bytes_lower_bound": forecast_outputs,
        "transaction_max_bytes": config["transaction_max_bytes"], **limits,
        "scope": "Major NPY framing and lower bounds; stream one parameter bank at a time. The full result inventory independently enforces total and member limits. No compressed archive or stdout tensor transport."}


def stage(run_dir):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    worker = here / "flux_step_boundary_campaign.py"
    engine = runpy.run_path(str(worker))
    c = engine["contract"]({})
    config = {"schema": "mdb-flux-step-boundary-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "context": c["context"], "height": 512, "width": 512, "steps": 4,
        "block_indices": c["block_indices"], "selector": {"logical_step": 2, "phase": "single", "index": 6},
        "cpu_threads": 4, "query_chunk": 64, "projection_chunk": 256, "neuron_chunk": 256,
        "retain_fields": False, "sample_indices": engine["SAMPLE_INDICES"],
        "image_sample_indices": engine["IMAGE_SAMPLE_INDICES"], "branches": c["branches"],
        "branch_execution_order": c["order"], "observation_plan": c["observation_plan"],
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        **{"max_" + key: c["counts"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")},
        "max_wall_seconds": 900, "transaction_max_bytes": 768 << 20,
        "transport_method": "planned-authenticated-ssh-manifest",
        "transport_limits": {"max_expanded_bytes": 8 << 30, "max_member_bytes": 1 << 30},
        "hypotheses": {
            "H-chain": "Starting from one clean observed step2/pre6 state, all14 complete single blocks6..19 consume their own predicted complete I/T predecessors, with actual source-bound weights. No target activation or native intermediate refresh.",
            "H-readout": "Actual norm_out scale/shift conditioning, population LayerNorm and proj_out produce predicted noise from predicted post19 image. Actual incoming temb is supplied and bound. Readout native entries occur after seal.",
            "H-commit": "Use archived sigma2/sigma3 and source-bound deterministic Euler arithmetic with qualified weak scalar BF16 promotion to predict committed Z3. Clear I/T/context/noise; retain C, IDs and full schedule and increment native clock to step3/project.",
            "H-consumer": "After one seal, obtain the16-transition native reference to step3/project. Compare native, exact latent sham and sealed predicted latent using typed Acts at that same native boundary. Each executes28 native final-step transitions and actual VAE; no predicted next projection or next-step transformer claim.",
            "H-error": "Retain all14 native block boundaries, native readout noise/Z3, full native continuation arrays, installed/final candidate cuts and64 fixed candidate witnesses. Report signed errors and image effects without numerical pass/fail gates.",
            "H-scope": "One fresh seed89/prompt, one observed prefix, one prediction horizon, no donor. This is not raw-prompt-to-image prediction, object semantics, automatic family induction, universality or measured execution speedup."},
        "terminal_status": "unassessed",
        "scope": "Seed89 clean observed step2/pre6:14 self-fed complete single blocks plus actual mathematical readout/Euler law sealed before targets, then predicted Z3 consumed by native step3.168native/2Acts/170receipts/3VAE. One load/encode/noise; no retries."}
    engine["validate_config"](config)
    config["storage_plan"] = storage_plan(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    names = ["flux_step_boundary_reference.py", "flux_transition_reference.py", "flux_mlp_replacement.py",
        "flux_attention_replacement_campaign.py", "flux_single_mlp_campaign.py", "flux_single_contraction.py",
        "flux_single_contexts.py", "flux_single_kv.py", "flux_family_custody.py"]
    extras = [(here / name, name) for name in names] + [(run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *names, "plan.json", preflight.name],
        "path_token_allowlist": ["flux-step-boundary-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    independent_names = ["flux_step_boundary_audit_math.py", "analyze_flux_transition_chain.py",
        "analyze_flux_attention_replacement.py", "analyze_flux_mlp_replacement.py"]
    request = {"schema": "mdb-flux-step-boundary-request-v1", "config": config,
        "launcher_source_sha256": sha(Path(__file__)),
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda",
        "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True},
        "execution_environment": execution_intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "independent_math_sources": {name: sha(here / name) for name in independent_names},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=3,
            logical_branches=3, logical_checkpoint_replays=3, max_local_work_items=20,
            checkpoint_policy="Clean prefix68; seal14 blocks+readout+Euler before current targets. Native reference16; three common project-boundary28-transition futures;2 typed latent Acts. No retries."),
        "resource_exception": {"ram_mb": 8192, "vram_mb": 13000, "disk_gb": 10,
            "calibration_job": "job-af3d6979dd41", "measured_peak_rss_mb": 4567.6, "measured_peak_vram_mb": 9452.0,
            "reason": "Same released-encoder B1 CUDA BF16 512px lane. Predecessor237native/9VAE used4.57GB RSS/9.45GB VRAM. New168native/3VAE, streamed one widened parameter set instead of three retained,14 complete output arrays264MB and full native current boundaries.8GiB RAM conservative measured allowance;13GB VRAM unchanged. Payload lives on measured /mnt/hdd8tb with ample space; bound output8GiB. Expected480s, worker900s, outer1200s, no speedup claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request,
        config_bytes=raw, additional_files=extras)
    freeze = run_dir / "independent-math-freeze"
    freeze.mkdir()
    import shutil
    for name in independent_names:
        shutil.copyfile(here / name, freeze / name)
    return {"submitted": False, "run_dir": str(run_dir), "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch

    if (run_dir / "submission.json").exists():
        raise ValueError("finite assay has already been admitted")
    request = read(run_dir / "request.json")
    unsigned = {k: v for k, v in request.items() if k != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or sha(Path(__file__)) != request["launcher_source_sha256"]
            or any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items())
            or any(sha(run_dir / "independent-math-freeze" / name) != digest for name, digest in request["independent_math_sources"].items())
            or read(run_dir / "plan.json") != request["config"]):
        raise ValueError("staged source/plan/independent math/launcher changed")
    config, resources = request["config"], request["resource_exception"]
    note = ("Klein4B CUDA BF16 B1 512px seed89:clean prefix68; seal14full single blocks6..19+readout+Euler. "
        "Native reference16; native/sham/predicted Z3 replay3x28+3VAE.168native/2Acts/170receipts. "
        "1load/encode/noise,4CPU,8GiB RAM/13GB VRAM,streamed weights; worker900s/outer1200s,no retries. "
        "Planned SSH manifest collection8GiB, no stdout tensor export. Stop on mechanics/census breach; retain numerical misses.")
    if len(note) > 500:
        raise ValueError("queue note too long")
    command = sealed_worker_command(request)
    save(run_dir / "admission-command.json", {"command": command, "request_sha256": request["request_sha256"],
        "launcher_source_sha256": sha(Path(__file__))})
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward",
        experiment="mdb-flux-step-boundary", payload=run_dir / "payload", prefer_host="beast",
        gpu=True, needs=request["needs"], ram_mb=resources["ram_mb"], vram_mb=resources["vram_mb"],
        disk_gb=resources["disk_gb"], cpu_threads=4, est_wall_s=480, timeout_s=1200,
        retry_on_kill=False, preflight=True, note=note)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    args = parser.parse_args()
    print(json.dumps(submit_staged(args.run_dir.resolve()) if args.submit_staged else stage(args.run_dir.resolve())))

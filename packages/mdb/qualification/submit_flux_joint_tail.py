"""Stage one own-state twenty-two-transition forecast and six native futures."""
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
    command = worker_command("flux_joint_tail_campaign.py")
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



PRIOR_WORKER_SHA256 = "90830ab12ad8b22689cc83c1294767c335dcb9f0b5f0d99766b4bed4da9fe9d0"
TRACKED_ANALYSIS_REF = "94a8e75"
PRIOR_HELPERS = ("flux_joint_chain_campaign.py", "flux_reentry_campaign.py", "flux_reentry_reference.py",
    "flux_step_boundary_campaign.py", "flux_step_boundary_reference.py",
    "flux_transition_reference.py", "flux_mlp_replacement.py", "flux_attention_replacement_campaign.py",
    "flux_single_contexts.py")
CORRECTED_ANALYZERS = ("analyze_flux_reentry.py",)
NEW_ANALYZERS = ("analyze_flux_joint_tail.py", "collect_flux_joint_tail.py")


def scheduler_endpoint(request=None):
    expected = "http://192.168.68.68:9025" if request is None else request["scheduler_url"]
    endpoint = os.environ.setdefault("MRUN_URL", expected)
    if not endpoint.startswith(("http://", "https://")) or (request is not None and endpoint != expected):
        raise ValueError("scheduler endpoint differs from sealed request")
    return endpoint


def closure_module_sha256():
    import inspect

    import mdb.flux_forecast_closure as closure

    return sha(Path(inspect.getfile(closure)))


def joint_closure_module_sha256():
    import inspect

    import mdb.flux_joint_forecast_closure as closure

    return sha(Path(inspect.getfile(closure)))


def tail_closure_module_sha256():
    import inspect

    import mdb.flux_joint_tail_closure as closure

    return sha(Path(inspect.getfile(closure)))


def prior_sources(prior_run):
    """Read immutable qualified payload, never live legacy scientific helpers."""
    request = read(prior_run / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request.get("request_sha256"):
        raise ValueError("qualified prior request seal differs")
    expected = request["source_hashes"]
    result = {}
    for name in PRIOR_HELPERS:
        path = prior_run / "payload" / name
        if name not in expected or sha(path) != expected[name]:
            raise ValueError("qualified prior scientific helper differs: " + name)
        result[name] = path
    if expected["flux_joint_chain_campaign.py"] != PRIOR_WORKER_SHA256:
        raise ValueError("prior native project/readout profile differs")
    return result, request


def independent_sources(here, prior_run):
    """Preserve qualified old math; explicitly override the corrected old reader."""
    import subprocess

    request = read(prior_run / "request.json")
    result, authorities = {}, {}
    for name, expected in request["independent_math_sources"].items():
        path = prior_run / "independent-math-freeze" / name
        if sha(path) != expected:
            raise ValueError("prior independent mathematical source differs: " + name)
        result[name] = path.read_bytes()
        authorities[name] = {"authority": "prior request-pinned independent-math-freeze", "sha256": expected}
    for name in CORRECTED_ANALYZERS:
        raw = subprocess.run(["git", "show", TRACKED_ANALYSIS_REF + ":packages/mdb/qualification/" + name],
            cwd=here.parents[2], check=True, capture_output=True).stdout
        result[name] = raw
        authorities[name] = {"authority": "corrected reader, tracked Git " + TRACKED_ANALYSIS_REF,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "original_preoutcome_sha256": request["independent_math_sources"][name]}
    for name in NEW_ANALYZERS:
        result[name] = (here / name).read_bytes()
        authorities[name] = {"authority": "new pre-outcome saved-only analysis/custody source",
            "sha256": hashlib.sha256(result[name]).hexdigest()}
    return result, authorities


def storage_plan(config):
    """Major native parameter/complete prediction framing; all-file bound is separate."""
    import numpy as np

    shapes = {"fused_weight": ((27648, 3072), "<u2"), "output_weight": ((3072, 12288), "<u2"),
        "native_register": ((1, 1536, 3072), "<u2"), "predicted_register": ((1, 1536, 3072), "<f4"),
        "image_embed_weight": ((3072, 128), "<u2"), "text_embed_weight": ((3072, 7680), "<u2"),
        "time1_weight": ((3072, 256), "<u2"), "time2_weight": ((3072, 3072), "<u2"),
        "joint_mod_weight": ((18432, 3072), "<u2"), "single_mod_weight": ((9216, 3072), "<u2"),
        "joint_square_weight": ((3072, 3072), "<u2"), "joint_ff_in_weight": ((18432, 3072), "<u2"),
        "joint_ff_out_weight": ((3072, 9216), "<u2")}
    members = {}
    for name, (shape, dtype) in shapes.items():
        header = io.BytesIO()
        value_type = np.dtype(dtype)
        np.lib.format.write_array_header_1_0(header, {"descr": np.lib.format.dtype_to_descr(value_type),
            "fortran_order": False, "shape": shape})
        members[name] = {"shape": list(shape), "dtype": str(value_type),
            "bytes": header.tell() + math.prod(shape) * value_type.itemsize, "header_bytes": header.tell()}
    old = 14 * (members["fused_weight"]["bytes"] + members["output_weight"]["bytes"])
    project = sum(members[name]["bytes"] for name in ("image_embed_weight", "text_embed_weight", "time1_weight",
        "time2_weight", "single_mod_weight")) + 2 * members["joint_mod_weight"]["bytes"]
    joint_bank = 8 * members["joint_square_weight"]["bytes"] + 2 * (
        members["joint_ff_in_weight"]["bytes"] + members["joint_ff_out_weight"]["bytes"])
    joint = 5 * joint_bank
    outputs = 20 * members["predicted_register"]["bytes"]
    limits = config["transport_limits"]
    if (any(value["bytes"] > limits["max_member_bytes"] for value in members.values())
            or old + project + joint + outputs > limits["max_expanded_bytes"]
            or outputs > config["transaction_max_bytes"]
            or config["transaction_max_bytes"] > limits["max_member_bytes"]):
        raise ValueError("complete joint-tail forecast/parameter geometry exceeds declared custody bounds")
    return {"schema": "mdb-flux-joint-tail-storage-plan-v1", "members": members,
        "prior_parameter_bytes_lower_bound": old, "project_parameter_bytes_lower_bound": project,
        "joint_parameter_bytes_lower_bound": joint, "joint_parameter_banks": 5,
        "one_joint_parameter_bank_bytes_lower_bound": joint_bank, "parameter_bytes_lower_bound": old + project + joint,
        "complete_output_bytes_lower_bound": outputs, "transaction_max_bytes": config["transaction_max_bytes"],
        **limits, "scope": "Major NPY framing/lower bounds, excluding small norm weights/context and witnesses. One parameter bank widened at a time; actual all-file/member inventory remains authoritative. No stdout tensor/archive transport."}


def stage(run_dir, *, prior_run=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    prior_run = Path(prior_run) if prior_run is not None else here.parents[3] / "artifacts/mdb-flux-joint-chain-20261104/run-01"
    old, prior_request = prior_sources(prior_run)
    independent, authorities = independent_sources(here, prior_run)
    worker = here / "flux_joint_tail_campaign.py"
    engine = runpy.run_path(str(worker))
    c = engine["contract"]({})
    config = {"schema": "mdb-flux-joint-tail-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "expected_closure_module_sha256": closure_module_sha256(),
        "expected_joint_closure_module_sha256": joint_closure_module_sha256(),
        "expected_tail_closure_module_sha256": tail_closure_module_sha256(),
        "context": c["context"], "height": 512, "width": 512, "steps": 4, "block_indices": c["block_indices"],
        "selector": {"logical_step": 2, "phase": "single", "index": 6},
        "cpu_threads": 4, "query_chunk": 64, "projection_chunk": 256, "neuron_chunk": 256, "retain_fields": False,
        "sample_indices": engine["SAMPLE_INDICES"], "image_sample_indices": engine["IMAGE_SAMPLE_INDICES"],
        "branches": c["branches"], "branch_execution_order": c["order"], "observation_plan": c["observation_plan"],
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        **{"max_" + key: c["counts"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")},
        "max_wall_seconds": 900, "transaction_max_bytes": 768 << 20,
        "transport_method": "planned-authenticated-ssh-manifest",
        "transport_limits": {"max_expanded_bytes": 12 << 30, "max_member_bytes": 1 << 30},
        "hypotheses": {
            "H-chain": "Clean observed step2/pre6; all14 complete single blocks, readout/Euler, next project and joint0..4 consume their own complete predictions. No native intermediate refresh.",
            "H-tail": "Predicted joint1 streams feed three distinct source-bound joint2/3/4 attention-then-MLP recurrences, before any selected target entry.",
            "H-consumer": "Native reference22 reaches step3/single0. Six native futures share22-transition suffixes. Five atomic closure Acts; bridges of3/2/1 joint operations separate local and carried error. No raw prompt/RGB prediction.",
            "H-error": "Preserve full/native field residuals, inherited-versus-local finite differences, signed reference/witness trajectories and all six actual final images. No numerical accuracy or semantic gates."},
        "terminal_status": "unassessed",
        "scope": "One fresh seed103 prompt/history;228native/5Acts/233receipts/6VAE,1load/encode/noise. Observed pre6, whole22-transition own-state prediction to single0, then actual native remaining suffix. No retries, raw-prompt/RGB forecast, semantic family induction or speedup claim."}
    engine["validate_config"](config)
    config["storage_plan"] = storage_plan(config)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    extras = [(path, name) for name, path in old.items()]
    extras.append((run_dir / "plan.json", "plan.json"))
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *old, "plan.json", preflight.name],
        "path_token_allowlist": ["flux-joint-tail-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-joint-tail-request-v1", "config": config, "scheduler_url": scheduler_endpoint(),
        "launcher_source_sha256": sha(Path(__file__)), "task_family": "forward", "model": config["model"],
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512,
        "width": 512, "needs": {"cuda": True}, "execution_environment": execution_intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "source_authorities": {"prior_request_sha256": prior_request["request_sha256"],
            "prior_worker_sha256": PRIOR_WORKER_SHA256, "legacy_sources": {name: sha(path) for name, path in old.items()}},
        "independent_math_sources": {name: hashlib.sha256(value).hexdigest() for name, value in independent.items()},
        "independent_source_authorities": authorities,
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=6, logical_branches=6,
            logical_checkpoint_replays=6, max_local_work_items=40,
            checkpoint_policy="Clean prefix68. Seal full remaining step/project/joint0..4 before targets. Reference22, bridges3/2/1, six22-transition suffixes,5 atomic closure Acts; no per-row jobs/retries."),
        "resource_exception": {"ram_mb": 12288, "vram_mb": 13000, "disk_gb": 14,
            "calibration_job": "job-0cda5797a845", "measured_peak_rss_mb": 8482.1,
            "measured_peak_vram_mb": 9590.0,
            "reason": "Completed prior joint-chain lane measured8482.1MB RSS/9590MB VRAM. Three additional joint parameter banks add1404MiB BF16 storage; widened matrices streamed with six native futures. Retain12288MB RAM/13000MB VRAM measured exception, CPU4 and disk14GB; actual output filesystem must pass12GiB bound. Expected480s/worker900s/outer1200s, no retries."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    freeze = run_dir / "independent-math-freeze"
    freeze.mkdir()
    for name, content in independent.items():
        with (freeze / name).open("xb") as stream:
            stream.write(content)
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
    scheduler_endpoint(request)
    config, resources = request["config"], request["resource_exception"]
    note = ("Klein4B CUDA BF16 B1 512 seed103:clean pre6; seal22ownstate transitions throughproject/joint0..4. "
        "Reference22,bridges3/2/1,6x22native suffix.228native/5Acts/233receipts/6VAE,1load/encode/noise. "
        "4CPU12288RAM13000VRAM14GBdisk; worker900/outer1200s,no retries. "
        "Stream matrices; SSH12GiB/1GiBmember. Keep allnumeric misses; stop mechanics/census breach.")
    if len(note) > 500:
        raise ValueError("queue note too long")
    command = sealed_worker_command(request)
    save(run_dir / "admission-command.json", {"command": command, "request_sha256": request["request_sha256"],
        "launcher_source_sha256": sha(Path(__file__))})
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward",
        experiment="mdb-flux-joint-tail", payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"],
        ram_mb=resources["ram_mb"], vram_mb=resources["vram_mb"], disk_gb=resources["disk_gb"], cpu_threads=4,
        est_wall_s=480, timeout_s=1200, retry_on_kill=False, preflight=True, note=note)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    args = parser.parse_args()
    print(json.dumps(submit_staged(args.run_dir.resolve()) if args.submit_staged
        else stage(args.run_dir.resolve(), prior_run=args.prior_run)))

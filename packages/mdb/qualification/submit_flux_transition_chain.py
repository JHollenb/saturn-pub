"""Stage, admit and collect a bounded two- or three-block prediction chain."""
from __future__ import annotations

import argparse
import hashlib
import json
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
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def execution_intent():
    """Use the measured Linux cache lane for this host-local checkpoint assay."""
    from mdb.execution_environment import intent

    value = intent()
    for key, variable, fallback in (
        ("uv_cache_dir", "MDB_WORKER_UV_CACHE_DIR", "/mnt/hdd8tb/mdb/uv-cache"),
        ("torch_home", "MDB_WORKER_TORCH_HOME", "/mnt/hdd8tb/mdb/torch"),
        ("extra_model_roots", "MDB_WORKER_MODEL_ROOTS", "/mnt/big/llm-models"),
    ):
        value[key] = os.environ.get(variable, fallback)
    if any(value[key].startswith("/Users/") or not value[key].startswith("/") for key in ("uv_cache_dir", "torch_home")):
        raise ValueError("Linux CUDA worker requires absolute worker-native cache paths")
    return value


def sealed_worker_command(request):
    """Bind the public command builder to the staged request, not ambient paths."""
    from mdb.payload import worker_command

    environment = request["execution_environment"]
    if any(not environment[k].startswith("/") or environment[k].startswith("/Users/") for k in ("uv_cache_dir", "torch_home")):
        raise ValueError("Linux CUDA worker requires absolute worker-native cache paths")
    command = worker_command("flux_transition_chain_campaign.py")
    for variable, key in (("UV_LINK_MODE", "uv_link_mode"), ("UV_CACHE_DIR", "uv_cache_dir"),
                          ("TORCH_HOME", "torch_home"), ("LLM_MODELS_EXTRA_ROOT", "extra_model_roots")):
        positions = [i for i, value in enumerate(command) if value.startswith(variable + "=")]
        if len(positions) != 1:
            raise ValueError("public worker command environment ABI differs: " + variable)
        command[positions[0]] = variable + "=" + environment[key]
    python_index = command.index("--python") + 1
    command[python_index] = environment["python"]
    requirements = [command[i + 1] for i, value in enumerate(command) if value == "--with"]
    if [value for value in requirements if not value.startswith("./")] != environment["requirements"]:
        raise ValueError("sealed worker dependency requirements differ from public command")
    return command


def storage_plan(config):
    """Check actual NPY framing for the largest declared parameter/state members.

    This is a necessary pre-admission storage check, not a prediction of every
    future artifact's size. The complete inventory remains bounded at export.
    """
    import io
    import math

    import numpy as np

    h, m, length = 3072, 9216, 1536
    shapes = {"fused_weight": ((3*h + 2*m, h), "<u2"), "output_weight": ((h, h+m), "<u2"),
        "full_native_register": ((1, length, h), "<u2"), "predicted_register": ((1, length, h), "<f4")}
    members = {}
    for name, (shape, dtype) in shapes.items():
        header = io.BytesIO()
        value_type = np.dtype(dtype)
        np.lib.format.write_array_header_1_0(header, {"descr": np.lib.format.dtype_to_descr(value_type),
            "fortran_order": False, "shape": shape})
        members[name] = {"shape": list(shape), "stored_dtype": str(value_type), "header_bytes": header.tell(),
            "bytes": header.tell() + math.prod(shape) * value_type.itemsize}
    limits = config["transport_limits"]
    member_limit = limits.get("max_member_bytes", 128 << 20)
    if any(value["bytes"] > member_limit for value in members.values()):
        raise ValueError("declared complete parameter/state NPY exceeds transport member bound")
    indices = config.get("block_indices", [6, 7])
    if indices not in ([6, 7], [6, 7, 8]):
        raise ValueError("unqualified chain storage geometry")
    parameter_lower_bound = len(indices) * (members["fused_weight"]["bytes"] + members["output_weight"]["bytes"])
    if parameter_lower_bound > limits["max_expanded_bytes"]:
        raise ValueError("declared chain parameter storage exceeds expanded transport bound")
    return {"schema": "mdb-flux-chain-storage-plan-v1", "members": members,
        "max_member_bytes": member_limit, "block_indices": indices,
        "parameter_bytes_lower_bound": parameter_lower_bound,
        **({"two_block_parameter_bytes_lower_bound": parameter_lower_bound} if len(indices) == 2 else {}),
        "max_expanded_bytes": limits["max_expanded_bytes"],
        "max_archive_bytes": limits.get("max_archive_bytes"),
        "scope": "Exact NPY framing and byte geometry for declared member types; necessary expanded parameter lower bound only, not a compressed-size prediction. Complete runtime inventory/archive still enforces total size and all members."}


def stage(run_dir, *, chain_length=2):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    worker = here / "flux_transition_chain_campaign.py"
    engine = runpy.run_path(str(worker))
    if type(chain_length) is not int or chain_length not in (2, 3):
        raise ValueError("only qualified two- or three-block chains may be staged")
    indices = list(range(6, 6 + chain_length))
    c = engine["contract"]({"block_indices": indices})
    last = indices[-1]
    config = {"schema": "mdb-flux-transition-chain-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "context": c["context"], "height": 512, "width": 512, "steps": 4, "block_indices": indices,
        "selector": {"logical_step": 3, "phase": "single", "index": 6}, "cpu_threads": 4,
        "query_chunk": 64, "projection_chunk": 256, "neuron_chunk": 256, "retain_fields": False,
        "sample_indices": engine["SAMPLE_INDICES"], "branches": c["branches"],
        "branch_execution_order": c["order"], "observation_plan": c["observation_plan"],
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        **{"max_" + key: c["counts"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")},
        "max_wall_seconds": 900, "archive_compression_level": 1,
        "transport_limits": {"max_archive_bytes": (1536 if chain_length == 2 else 2048) << 20,
            "max_expanded_bytes": 3072 << 20, "max_member_bytes": 256 << 20},
        "hypotheses": {
            "H-chain": "Independently calculate complete block6 text+image outputs from observed pre6 state and actual weights, then calculate block7 from precisely that predicted complete output. Freeze both roles and stages before current-step native6/7 entry; no native intermediate refresh.",
            "H-complete": "Both blocks include population LayerNorm, time modulation, all-token QKV, learned Q/K RMSNorm, upstream RoPE, full unmasked attention, SwiGLU, actual fused output projection, gated write and residual addition under explicit staged BF16 reference arithmetic. No current-target activation is an input.",
            "H-native": "After the seal obtain native6/7 references. Seven common-post7 futures include native low/high, exact low sham, one predicted block then native7 for both histories, and two recursively predicted blocks for both histories. Both I/T are installed through typed Acts, all other payload data preserved, cursor advanced by declared native parents.",
            "H-decomposition": "The native7(pred6) pred1 arms split total post7 error into local second-block discrepancy plus transported first-block discrepancy. Retain complete vectors and signed inner products; neither monotonic growth nor a quality threshold is assumed.",
            "H-continuation": "At each common post7 boundary and14 actual suffix exits retain complete reference arrays, candidate full signed worker metrics and64 fixed witnesses. Candidate intermediate norms remain observations without full independent-array authority. Final RGB remains observed native continuation.",
            "H-scope": "One fresh seed79 prompt and two dependent own-input engineered histories. Observed prefix and upstream modulation/RoPE remain supplied. This is an independently predicted two-block segment, not raw-prompt-to-image prediction, semantic naming, skipped-execution speedup or automatic family induction."},
        "terminal_status": "unassessed",
        "scope": "Fresh seed79: complete both-stream block6→7 prediction chain from observed pre6 state; one public seal before either current target. Seven native suffix futures,212native transitions/14Acts/226receipts/7VAE, no direct block calls or retries. Numerical misses retained."}
    if chain_length == 3:
        config["hypotheses"] = {
            "H-chain": "Calculate complete both-stream block6 output from observed pre6 state and actual parameters, block7 from precisely predicted6, and block8 from precisely predicted7. Seal both histories and all stages before current native6/7/8. No native intermediate refresh or correction.",
            "H-complete": config["hypotheses"]["H-complete"].replace("Both blocks", "All three blocks"),
            "H-native": "Nine futures share actual common post8. Native low/high and exact low sham; pred1 installs predicted6 then native7/8; pred2 installs predicted7 then native8; pred3 installs predicted8. Both I/T typed Acts preserve native layout and other state; no native step between slot installations.",
            "H-decomposition": "Retained native/bridge/common I/T support two-term local+carried decomposition at blocks7/8, and exact post8 three-term telescoping across pred3/pred2/pred1/native. Preserve signed statistics and all pairwise inner products. Nonlinear finite differences are not a transition applied to an error vector. No monotonicity or cancellation expectation.",
            "H-continuation": "Common post8 plus13 native suffix exits, complete reference arrays, candidate full worker metrics and64 fixed witnesses. Complete installed, bridge-exit, common and final cuts retained. Final images are native VAE observations, no independently predicted suffix claim.",
            "H-scope": "Fresh seed83, one prompt and two dependent engineered histories. Within-run pred1/2/3 share parent states; comparison to seed79 also changes seed. Observed prefix and upstream modulation/RoPE are supplied. No raw prompt-to-image, semantic naming, speedup or automatic family induction claim."}
        config["scope"] = "Fresh seed83: complete self-fed I/T block6→7→8 chain sealed before current targets. Nine matched native futures;237 native/18 Acts/255 receipts/9 VAE; block7/8 and telescoping post8 error decomposition. Numerical misses retained; no retries."
    engine["validate_config"](config)
    config["storage_plan"] = storage_plan(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    names = ["flux_transition_reference.py", "flux_mlp_replacement.py", "flux_attention_replacement_campaign.py",
        "flux_single_mlp_campaign.py", "flux_single_contraction.py", "flux_single_contexts.py",
        "flux_single_kv.py", "flux_family_custody.py"]
    extras = [(here / name, name) for name in names] + [(run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *names, "plan.json", preflight.name],
        "path_token_allowlist": ["flux-transition-chain-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    n = c["counts"]["images"]
    request = {"schema": "mdb-flux-transition-chain-request-v1", "config": config,
        "launcher_source_sha256": sha(Path(__file__)),
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda", "dtype": "bfloat16",
        "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True}, "execution_environment": execution_intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=n, logical_branches=n,
            logical_checkpoint_replays=n, max_local_work_items=20,
            checkpoint_policy=f"108 builder/prep transitions; seal complete both-stream blocks6..{last} before current targets. Reuse native prefix cuts and actual bridges; {n} common {c['suffix_steps']}-step suffixes. No retries."),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 10,
            "calibration_job": "job-947395d68069", "measured_peak_rss_mb": 8539.1, "measured_peak_vram_mb": 9646.0,
            "reason": "Same qualified encoder-release CUDA B1 512px lane. Previous fullA/M nine-future assay took264.87runner seconds with8.54GB RSS/9.65GB VRAM. New two complete fused weight sets+output projections retain approximately491MB BF16 sources and about1GB immutable CPU FP32 parameter operands. Only four full output fields and sampled intermediates retained;64query/256row tiles and seven instead of nine suffixes. Expected420s; no matched speedup claim."}}
    if chain_length == 3:
        request["resource_exception"].update(calibration_job="job-48dc7e610cd2", measured_peak_rss_mb=4252.5,
            measured_peak_vram_mb=9446.0,
            reason="Same encoder-release CUDA BF16 B1 512px execution lane; predecessor completed212 native/14 Acts/7 VAE in166.522 worker seconds before failed export. Runner observed4.25GB RSS/9.45GB VRAM. Three blocks add one244MB BF16 parameter set/about489MB immutable FP32 operands plus two output fields and two futures. Keep measured conservative24GB RAM/13GB VRAM,4 CPU threads; expected480s, no speedup claim.")
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return {"submitted": False, "run_dir": str(run_dir), "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch

    if (run_dir / "submission.json").exists():
        raise ValueError("transition-chain assay already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items())
            or read(run_dir / "plan.json") != request["config"]):
        raise ValueError("staged transition-chain request/source/plan changed")
    config, resources = request["config"], request["resource_exception"]
    n = len(config["branches"])
    steps = config["max_adapter_steps"]
    note = (f"Klein4B CUDA BF16 B1 512px seed{config['context']['seed']}:108builder/prep; "
        f"self-fed I/T blocks{config['block_indices']} sealed before targets. "
        f"{steps}native/{config['max_acts']}Acts/{n}VAE; bridges split local/carried and telescoping error. "
        "One load/encode/noise,4CPU,24GB RAM/13GB VRAM; worker900s/outer1200s,no retries. "
        "B1 sequential shared-prefix replay, bounded artifact archive. Numerical misses retained.")
    if len(note) > 500:
        raise ValueError("transition-chain queue note too long")
    job = guarded_mrun_launch(sealed_worker_command(request), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-transition-chain", payload=run_dir / "payload",
        prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=resources["ram_mb"], vram_mb=resources["vram_mb"],
        disk_gb=resources["disk_gb"], cpu_threads=4, est_wall_s=420 if n == 7 else 480,
        timeout_s=1200, retry_on_kill=False, preflight=True, note=note)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect")
    parser.add_argument("--chain-length", type=int, choices=(2, 3), default=2, help="Used only when staging a new run")
    args = parser.parse_args()
    if args.collect:
        limits = read(args.run_dir / "request.json")["config"]["transport_limits"]
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](args.run_dir.resolve(), args.collect, limits=limits)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        result = stage(args.run_dir.resolve(), chain_length=args.chain_length)
    print(json.dumps(result))

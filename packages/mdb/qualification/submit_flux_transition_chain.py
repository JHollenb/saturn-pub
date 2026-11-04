"""Stage, admit and collect one finite two-block prediction chain assay."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def stage(run_dir):
    from mdb.execution_environment import intent
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    here = Path(__file__).resolve().parent
    worker = here / "flux_transition_chain_campaign.py"
    engine = runpy.run_path(str(worker))
    config = {"schema": "mdb-flux-transition-chain-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "context": engine["CONTEXT"], "height": 512, "width": 512, "steps": 4, "block_indices": [6, 7],
        "selector": {"logical_step": 3, "phase": "single", "index": 6}, "cpu_threads": 4,
        "query_chunk": 64, "projection_chunk": 256, "neuron_chunk": 256, "retain_fields": False,
        "sample_indices": engine["SAMPLE_INDICES"], "branches": engine["BRANCHES"],
        "branch_execution_order": engine["ORDER"], "observation_plan": engine["OBSERVATION_PLAN"],
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        **{"max_" + key: engine["COUNTS"][key] for key in ("adapter_steps", "acts", "candidate_calls", "images", "vae_calls")},
        "max_wall_seconds": 900, "archive_compression_level": 1,
        "transport_limits": {"max_archive_bytes": 1536 << 20, "max_expanded_bytes": 3072 << 20},
        "hypotheses": {
            "H-chain": "Independently calculate complete block6 text+image outputs from observed pre6 state and actual weights, then calculate block7 from precisely that predicted complete output. Freeze both roles and stages before current-step native6/7 entry; no native intermediate refresh.",
            "H-complete": "Both blocks include population LayerNorm, time modulation, all-token QKV, learned Q/K RMSNorm, upstream RoPE, full unmasked attention, SwiGLU, actual fused output projection, gated write and residual addition under explicit staged BF16 reference arithmetic. No current-target activation is an input.",
            "H-native": "After the seal obtain native6/7 references. Seven common-post7 futures include native low/high, exact low sham, one predicted block then native7 for both histories, and two recursively predicted blocks for both histories. Both I/T are installed through typed Acts, all other payload data preserved, cursor advanced by declared native parents.",
            "H-decomposition": "The native7(pred6) pred1 arms split total post7 error into local second-block discrepancy plus transported first-block discrepancy. Retain complete vectors and signed inner products; neither monotonic growth nor a quality threshold is assumed.",
            "H-continuation": "At each common post7 boundary and14 actual suffix exits retain complete reference arrays, candidate full signed worker metrics and64 fixed witnesses. Candidate intermediate norms remain observations without full independent-array authority. Final RGB remains observed native continuation.",
            "H-scope": "One fresh seed79 prompt and two dependent own-input engineered histories. Observed prefix and upstream modulation/RoPE remain supplied. This is an independently predicted two-block segment, not raw-prompt-to-image prediction, semantic naming, skipped-execution speedup or automatic family induction."},
        "terminal_status": "unassessed",
        "scope": "Fresh seed79: complete both-stream block6→7 prediction chain from observed pre6 state; one public seal before either current target. Seven native suffix futures,212native transitions/14Acts/226receipts/7VAE, no direct block calls or retries. Numerical misses retained."}
    engine["validate_config"](config)
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
    request = {"schema": "mdb-flux-transition-chain-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "backend": "diffusers", "device": "cuda", "dtype": "bfloat16",
        "batch_size": 1, "height": 512, "width": 512, "needs": {"cuda": True}, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=7, logical_branches=7,
            logical_checkpoint_replays=7, max_local_work_items=16,
            checkpoint_policy="108 builder/prep transitions; seal both complete block6/7 chains before current native entries. Four native prefix transitions, two pred1 native7 bridges, seven14-step suffixes; both I/T typed Acts. No retries."),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 10,
            "calibration_job": "job-947395d68069", "measured_peak_rss_mb": 8539.1, "measured_peak_vram_mb": 9646.0,
            "reason": "Same qualified encoder-release CUDA B1 512px lane. Previous fullA/M nine-future assay took264.87runner seconds with8.54GB RSS/9.65GB VRAM. New two complete fused weight sets+output projections retain approximately491MB BF16 sources and about1GB immutable CPU FP32 parameter operands. Only four full output fields and sampled intermediates retained;64query/256row tiles and seven instead of nine suffixes. Expected420s; no matched speedup claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return {"submitted": False, "run_dir": str(run_dir), "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb.job_support import guarded_mrun_launch
    from mdb.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("transition-chain assay already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]
            or any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items())
            or read(run_dir / "plan.json") != request["config"]):
        raise ValueError("staged transition-chain request/source/plan changed")
    config, resources = request["config"], request["resource_exception"]
    note = ("Klein4B CUDA BF16 B1 512px seed79:108builder/prep; full both-stream blocks6→7 NumPy chain sealed before native6/7. "
        "212native/14Acts/226receipts/7VAE; pred1 native7 bridges separate local vs propagated error. "
        "One load/encode/noise,4CPU,24GB RAM/13GB VRAM; expected420s,worker900s/outer1200s,no retries. "
        "Seven15-boundary traces; archive1536MiB/expanded3072MiB. Numerical misses retained.")
    if len(note) > 500:
        raise ValueError("transition-chain queue note too long")
    job = guarded_mrun_launch(worker_command("flux_transition_chain_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-transition-chain", payload=run_dir / "payload",
        prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=resources["ram_mb"], vram_mb=resources["vram_mb"],
        disk_gb=resources["disk_gb"], cpu_threads=4, est_wall_s=420, timeout_s=1200, retry_on_kill=False, preflight=True, note=note)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        limits = read(args.run_dir / "request.json")["config"]["transport_limits"]
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](args.run_dir.resolve(), args.collect, limits=limits)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        result = stage(args.run_dir.resolve())
    print(json.dumps(result))

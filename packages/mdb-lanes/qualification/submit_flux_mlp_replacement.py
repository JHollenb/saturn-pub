"""Stage one full-field MLP prediction/replacement assay in one CUDA lease."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir, *, submit=True):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    here = Path(__file__).resolve().parent
    worker = here / "flux_mlp_replacement_campaign.py"
    engine = runpy.run_path(str(worker))
    helper = runpy.run_path(str(here / "flux_mlp_replacement.py"))
    config = {"schema": "mdb-flux-mlp-replacement-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "context": engine["CONTEXT"], "height": 512, "width": 512, "steps": 4, "cpu_threads": 4,
        "single_index": 6, "selector": {"logical_step": 3, "phase": "single", "index": 6},
        "query_indices": engine["QUERIES"], "neuron_indices": engine["NEURONS"], "neuron_chunk": 256,
        "array_storage": "bf16-bits", "archive_integer_ids": "native-int64", "branches": helper["BRANCHES"],
        "max_adapter_steps": 245, "max_acts": 11, "max_candidate_calls": 9, "max_images": 9, "max_vae_calls": 9,
        "max_wall_seconds": 1200, "archive_compression_level": 1,
        "instrument_revision": 2, "incomplete_predecessor_job": "job-9999f2088d3b",
        "timing_repair": "Preserve the first900s timeout as incomplete execution. Add worker/import/model/encode/residency/noise/schedule/session/builder/preparation phases,15s cooperative progress and90s faulthandler stacks. Scientific geometry, predictor and nine branches unchanged; worker1200s includes its imports, outer1800s includes bootstrap and export. No automatic retry.",
        "hypotheses": {
            "H-field": "From observed pre6 image state, time modulation and every actual paired MLP weight row, calculate complete M[1,1024,9216] for both roles using population LayerNorm, staged BF16 affine, FP64 linear sums and staged BF16 SwiGLU. Seal before all current-step single6 calls. Earlier-step module6 calls are permitted in the 108-step builder. Retain every mismatch and error without a numerical quality gate.",
            "H-substitution": "Sealed predicted M fields enter the actual native to_out image-MLP slice. Attention, recipient residual/gate and text remain native; native fused MLP still executes before replacement. The complete predicted field is consumed, not merely sampled forecasts or measured donor activations.",
            "H-matched": "Compare low.predicted with zero and high.predicted with text to isolate native-versus-reference numerical discrepancy. Compare M.predicted with M.measured on the same zero parent to measure forecast error relative to the measured high-M transfer effect. No predeclared effect direction or error threshold.",
            "H-placement": "Repeat the measured/predicted high-M pair after a fixed roll512 along the image-token axis. This preserves values and energy but changes placement; it is a spatial contrast, not a guaranteed destructive null. Report all native latent/RGB consequences, including near-zero or unexpectedly stronger effects.",
            "H-controls": "Sham substitutes native low M into low. Archive exact payload cast, full image-M equality to sealed or measured fields (including permutation), unchanged attention/text footprint, residual/gate, native-prefix candidate comparisons, seven image-only Acts and all suffix receipts. Forecast fidelity and sham replay are separate observations.",
            "H-scope": "One fresh seed71, same prompt/checkpoint/site. Both I and T may differ at pre6 after the direct matched-image/different-T post0 construction. No isolated text cause, new KV-equivalence assay, independently predicted attention/text, whole-model image forecast, calibrated object semantics or automatic family discovery."},
        "scope": "Fresh seed71 single6: predict two complete image-side MLP activation fields before target observation, lower sealed fields at the native consumer, and replay nine native RGB futures. One load/encode/noise,108 builder plus137 panel transitions,11 Acts,256 receipts,nine direct blocks/nine VAE calls. Conditional observed-state-and-weight forecast, trusted callbacks with external CUDA guards/census; no hermetic isolation, runtime independence or speedup claim.",
        "terminal_status": "unassessed"}
    engine["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    names = ["flux_mlp_replacement.py", "flux_single_mlp.py", "flux_single_mlp_campaign.py",
        "flux_single_contraction.py", "flux_single_contexts.py", "flux_single_kv.py", "flux_family_custody.py"]
    extras = [(here / name, name) for name in names] + [(run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *names, "plan.json", preflight.name],
        "path_token_allowlist": ["flux-mlp-replacement-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-mlp-replacement-request-v1", "config": config, "task_family": "forward",
        "model": config["model"], "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "needs": {"cuda": True}, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9,
            logical_checkpoint_replays=9, max_local_work_items=16,
            checkpoint_policy="Fresh encode/noise,96-step/3-Act source builder plus12-step/1-Act preparation. Full image-M forecast sealed before current target. Two native/seven edited direct blocks, two native prefixes, nine15-step suffixes.245 transitions11Acts256receipts9VAE. One lease,no retries."),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 8,
            "ram_kill_factor": RAM_KILL_FACTOR, "ram_kill_ceiling_mb": kill_ceiling_mb(24000),
            "calibration_job": "job-1f1587e6983b", "measured_peak_rss_mb": 16059, "measured_peak_vram_mb": 9282,
            "reason": "Qualified full-encoder 24GB RAM/13GB VRAM lane; release encoder after one encode. Prior seed67 entry assay RSS8248MB/VRAM9754MB. New full-field NumPy predictor tiles256 paired neurons, roughly232GFLOP FP64 total, weights113MB BF16/226MB FP32 and two38MB predicted fields. Retain only two full native candidate captures; archive edited ports immediately. No throughput claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return submit_staged(run_dir) if submit else {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("full MLP replacement assay already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("replacement request self seal differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged replacement payload changed")
    config, resource = read(run_dir / "plan.json"), request["resource_exception"]
    if config != request["config"]:
        raise ValueError("staged replacement plan changed")
    job = guarded_mrun_launch(worker_command("flux_mlp_replacement_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-mlp-replacement", payload=run_dir / "payload", prefer_host="beast",
        gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"], disk_gb=resource["disk_gb"],
        cpu_threads=4, est_wall_s=1200, timeout_s=1800, retry_on_kill=False, preflight=True,
        note="Repeat after job-9999f2088d3b timed out; same seed71 assay, Luna reviewed. Phase markers/15s progress/90s stacks. Worker1200s, outer1800s, no retries. Klein4B CUDA BF16 B1 512px:108 builder steps/4 Acts; seal two full M[1024,9216] forecasts before single6.245 native,11 Acts,9 direct,9 VAE,256 receipts. One load/encode/noise,256-neuron tiles,4 CPU threads;24GB RAM/13GB VRAM. Stop at budget; if stalled before sources_prepared, profile1-3 transitions next.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit-staged", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](args.run_dir.resolve(), args.collect)
    elif args.submit_staged:
        result = submit_staged(args.run_dir.resolve())
    else:
        result = stage(args.run_dir.resolve(), submit=not args.stage_only)
    print(json.dumps(result))

"""Stage and admit one fresh-seed early-MLP forecast in one finite CUDA lease."""
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
    worker = here / "flux_single_mlp_campaign.py"
    engine = runpy.run_path(str(worker))
    helper = runpy.run_path(str(here / "flux_single_contraction.py"))
    config = {"schema": "mdb-flux-single-mlp-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "context": engine["CONTEXT"], "height": 512, "width": 512, "steps": 4, "cpu_threads": 4,
        "single_index": 6, "selector": {"logical_step": 3, "phase": "single", "index": 6},
        "query_indices": engine["QUERIES"], "neuron_indices": engine["NEURONS"],
        "output_dims": list(range(0, 3072, 96)), "array_storage": "bf16-bits", "archive_integer_ids": "native-int64",
        "instrument_revision": 2, "supersedes_instrument_failure": "job-ec1f4d871de8", "branches": helper["BRANCHES"],
        "max_adapter_steps": 245, "max_acts": 14, "max_candidate_calls": 5, "max_images": 9, "max_vae_calls": 9,
        "max_wall_seconds": 780, "archive_compression_level": 1,
        "hypotheses": {
            "H-entry": "From observed pre6 image state, time modulation and actual selected fused weight rows, independently calculate full-width LayerNorm, time affine, paired MLP U/V and SwiGLU at eight image queries and 32 neurons for both sources. Seal before any logical-step3 single6 call. Earlier-timestep module6 calls are permitted in the 108-step builder. Retain all FP64 and explicit BF16 reference errors; no fitted coefficients, error gate or whole-MLP prediction claim.",
            "H-partitions": "Select A/M/AM image features entering native to_out; recipient residual, gate and all text features remain low. Mixed T equals native low T. AM image projection equals native high image projection; AM image update need not equal high because the residual differs.",
            "H-numeric": "A separate later projection forecast uses two observed source A/M operands and actual selected output rows. Seal before three edited calls. AM projection consistency and A/M new predictions remain separate from the earlier MLP forecast.",
            "H-controls": "Sham repeats low and coherent repeats native high endpoints. Candidate full I/T equals the native prefix and native conditioning is preserved. These are instrument controls; retain explicit misses.",
            "H-geometry": "Explore fresh-seed opposing-write geometry, the effective U=post-pre FP64 distance identity and the full A/M factorial. No predeclared positive contraction or M-over-A ordering. The prior M-over-A prediction miss remains immutable.",
            "H-scope": "One fresh seed67 context, same prompt and checkpoint. Direct matched-image/different-T post0 construction followed by native1..5; no fresh K/V equivalence assay. Nine dependent native futures; no object dictionary, semantic repair, raw-prompt image forecast or automatic family induction."},
        "scope": "Fresh seed67 single6: predict sampled MLP features before their first current-step observation, then execute an A/M feature factorial and nine native RGB futures. One load/encode/noise, 108 builder plus 137 panel transitions, 14 Acts, 259 receipts, five direct blocks and nine VAE calls. Conditional state-and-weight forecast using trusted callback dataflow; no hermetic isolation or whole-model/raw-prompt forecast.",
        "terminal_status": "unassessed"}
    engine["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    names = ["flux_single_mlp.py", "flux_single_contraction.py", "flux_single_contraction_campaign.py",
        "flux_single_contexts.py", "flux_single_kv.py", "flux_family_custody.py"]
    extras = [(here / name, name) for name in names] + [(run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [worker.name, *names, "plan.json", preflight.name],
        "path_token_allowlist": ["flux-single-mlp-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-single-mlp-request-v1", "config": config, "task_family": "forward",
        "model": config["model"], "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "needs": {"cuda": True}, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=9, logical_branches=9,
            logical_checkpoint_replays=9, max_local_work_items=16,
            checkpoint_policy="Fresh encode/noise, 96-step/3-Act source builder plus 12-step/1-Act preparation; pre6 early MLP seal. Prepared panel: two native and three mixed direct blocks, two prefixes, nine 15-step suffixes. 245 native transitions, 14 Acts, 259 receipts, nine VAE calls, lossless BF16 archive, one lease and no retries."),
        "resource_exception": {"ram_mb": 24000, "vram_mb": 13000, "disk_gb": 8,
            "ram_kill_factor": RAM_KILL_FACTOR, "ram_kill_ceiling_mb": kill_ceiling_mb(24000),
            "calibration_job": "job-1f1587e6983b", "measured_peak_rss_mb": 16059, "measured_peak_vram_mb": 9282,
            "reason": "Qualified full-encoder startup lane: 24GB RAM/13GB VRAM, CUDA BF16 batch one; release the encoder after one encode. The no-encoder 16GB calibration is inapplicable. Lossless BF16 archives reduce expanded capture bytes; no speedup claim. Measured admission exception for this native checkpoint."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    return submit_staged(run_dir) if submit else {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}


def submit_staged(run_dir):
    from mdb_lanes.job_support import guarded_mrun_launch
    from mdb_lanes.payload import worker_command

    if (run_dir / "submission.json").exists():
        raise ValueError("fresh MLP-entry assay already admitted")
    request = read(run_dir / "request.json")
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest() != request["request_sha256"]:
        raise ValueError("fresh MLP request self seal differs")
    if any(sha(run_dir / "payload" / name) != digest for name, digest in request["source_hashes"].items()):
        raise ValueError("staged fresh MLP payload changed")
    config, resource = read(run_dir / "plan.json"), request["resource_exception"]
    if config != request["config"]:
        raise ValueError("staged fresh MLP plan changed")
    job = guarded_mrun_launch(worker_command("flux_single_mlp_campaign.py"), request=request,
        model=config["model"], task="forward", experiment="mdb-flux-single-mlp", payload=run_dir / "payload", prefer_host="beast",
        gpu=True, needs=request["needs"], ram_mb=resource["ram_mb"], vram_mb=resource["vram_mb"], disk_gb=resource["disk_gb"],
        cpu_threads=4, est_wall_s=300, timeout_s=900, retry_on_kill=False, preflight=True,
        note="Klein 4B CUDA BF16, batch 1, 512px, fresh seed67. One encode/noise;108 builder transitions/4 Acts prepare pre-single6. Seal sampled LayerNorm/time-affine/MLP U,V/SwiGLU forecasts before any current-step single6 target. Nine native futures with A/M factorial:245 transitions14Acts5directcalls9VAE. Retain misses and geometry trends; lossless BF16 ports. One resident load, no retries,900s lease stop;24GB fullencoder RAM calibration.")
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

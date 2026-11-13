"""Submit one fresh-prompt, same-parent joint attention/MLP coupling assay."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def stage(run_dir, *, submit=True):
    from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    campaign = runpy.run_path(str(here / "flux_joint_self_campaign.py"))
    config = {
        "schema": "mdb-flux-joint-self-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "prompt": "A red sphere to the left of a blue cube on a plain white background. Two separate objects, simple studio product photograph.",
        "seed": 47, "height": 512, "width": 512, "steps": 4, "cpu_threads": 4,
        "selector": {"logical_step": 1, "phase": "joint", "index": 4},
        "branches": campaign["BRANCHES"],
        "preview_rows": ["native", "Azero.recompute", "Azero.hold", "Mzero"], "preview_steps": [2, 3],
        "max_adapter_steps": 1200, "max_candidate_calls": 13,
        "max_final_images": 12, "max_preview_images": 8, "max_images": 20, "max_vae_calls": 21,
        "max_wall_seconds": 840, "max_archive_bytes": 512 << 20,
        "archive_compression_level": 1,
        "hypotheses": {
            "H-A-vs-M": "Tentative extrapolation: final matched-native latent L2(Azero.recompute) > L2(Mzero). Retain misses; geometric influence is not a semantic score.",
            "H-compensation": "Tentative compensation prediction: Ahalf.recompute final latent is closer to native than Ahalf.hold. A sign reversal or miss remains evidence.",
            "H-Mzero-shape": "Tentative visual prediction: Mzero retains two separate objects and a sphere-like red object. Retain uncertain/mixed geometry, blue shape, ordering, palette and collateral separately.",
            "H-controls": "Native port recompute equals the unmodified adapter step; sham reproduces native scheduler latents and final PNG exactly. Direct live-latent VAE preview agrees with completed public decode on native final cut.",
            "H-scope": "One fresh literal prompt encoded once, own seed47 noise and conditioning throughout, no donor or imported prefix. Native prefix and same-parent ports observed before seal; no RGB rendered before seal. Original MLP can be held only from this native parent. Suffixes are executed natively; these are empirical rankings, not independent RGB forecasts. Text rollback removes only this block's text update; prior text/residual and later conditioning remain."
        },
        "scope": "Fresh single-prompt native seed47 step1 joint4: scale actual own attention/MLP output operands, compare MLP recomputation versus same-parent native M return, selected own pre-block T rollback, own C throughout. 12 dependent final futures plus8 noisy live-latent previews;21 VAE calls including final preview parity. No raw-prompt numerical image forecast.",
        "terminal_status": "unassessed",
    }
    campaign["validate_config"](config)
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    worker = here / "flux_joint_self_campaign.py"
    extras = [(here / "flux_joint_self.py", "flux_joint_self.py")]
    extras += [(here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [worker.name, *(name for _, name in extras), preflight.name],
        "path_token_allowlist": ["flux-joint-self-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    ram_mb, vram_mb = 16000, 13000
    if kill_ceiling_mb(ram_mb) >= 66512 * .95:
        raise ValueError("RAM kill envelope cannot fit measured reference host")
    request = {"schema": "mdb-flux-joint-self-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True},
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": intent(),
        "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=12,
            logical_branches=12, logical_checkpoint_replays=12, max_local_work_items=16,
            checkpoint_policy="Encode one literal prompt once; native prefix to step1joint4; capture same-parent native ports before seal;12 in-process futures,8 intermediate previews,21 VAE calls incl native-final parity; own C/no donor/no retries"),
        "resource_exception": {"ram_mb": ram_mb, "vram_mb": vram_mb,
            "ram_kill_factor": RAM_KILL_FACTOR, "ram_kill_ceiling_mb": kill_ceiling_mb(ram_mb),
            "reference_host_mb": 66512, "calibration_job": "job-947ba45b47f9",
            "measured_peak_rss_mb": 3210, "measured_peak_vram_mb": 9390,
            "native_encoding_calibration_job": "job-c2a467736b50", "native_measured_peak_rss_mb": 10697.6,
            "reason": "Same CUDA BF16 B1 512px4step resident model/VAE; encode one fresh prompt on CUDA with transformer/VAE on CPU then offload encoder. CAS deduplicates retained BF16 ports; calibrated16GB RAM/13GB VRAM envelope."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-joint-self", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=ram_mb, vram_mb=vram_mb, cpu_threads=4, est_wall_s=540,
        timeout_s=config["max_wall_seconds"], retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47 fresh sphere prompt once. Step1joint4 own A/M0/.5/1, recompute vs same-parent M hold, selected own-preT rollback; ownC always/no donor.12 final futures+8 live noisy previews; <=1200steps/13candidateblocks/21VAE/840s. One load/no retries; keep misses.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        result = stage(args.run_dir.resolve(), submit=not args.stage_only)
    print(json.dumps(result))

"""Admit one donor-free, sealed Klein continuation-to-RGB experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def plan(inventory):
    return {"schema": "mdb-flux-trajectory-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": inventory["model_path"], "expected_checkpoint_sha256": inventory["checkpoint_sha256"],
        "prompt": "A red cube to the left of a blue sphere on a plain white background. Two separate objects, simple studio product photograph.",
        "seed": 23, "height": 512, "width": 512, "steps": 4, "guidance_scale": 1., "cpu_threads": 4,
        "max_transitions": 111, "max_native_transitions": 111, "max_images": 2,
        "witness_values": 4096, "max_wall_seconds": 840, "max_archive_bytes": 128 << 20,
        "scope": "observed typed parent before joint0; all remaining transformer transitions, later projections, Euler steps and functional VAE consume own predicted states",
        "arithmetic": "separately wired Torch functional equations; shared low-level CUDA BF16 kernels with native reference",
        "observed": "native prompt conditioning, step0 projection/context, initial latent, position IDs and fixed schedule",
        "forbidden": "native suffix module calls or intermediate state refresh during prediction; no donor or intervention",
        "retention": "immutable full CPU boundary bytes through comparison; sealed descriptors and deterministic witnesses; full scheduler latents, decoder output and RGB retained",
        "limits": "full intermediate arrays are RAM-only; offline audit recomputes witnesses and retained endpoints, not every full-array worker aggregate",
        "terminal_status": "unassessed"}


def stage(run_dir, *, inventory):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=False)
    config = plan(inventory)
    save(run_dir / "plan.json", config)
    names = ["flux_trajectory_forecast_campaign.py", "flux_functional_denoiser.py", "flux_functional_vae.py",
             "campaign_support.py", "plan.json", "preflight.v1.json"]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": names,
                     "path_token_allowlist": ["flux-trajectory-result", config["model_path"]]})
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    worker = here / names[0]
    extras = [(here / name, name) for name in names[1:3]] + [
        (here / "discovery_upgrades.py", "campaign_support.py"),
        (run_dir / "plan.json", "plan.json"), (preflight, preflight.name)]
    request = {"schema": "mdb-flux-trajectory-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True},
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1,
        "height": 512, "width": 512, "execution_environment": intent(), "config_bytes_sha256": sha(raw),
        "source_hashes": {name: sha(path.read_bytes()) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=2,
            logical_branches=2, logical_checkpoint_replays=1, max_local_work_items=4,
            checkpoint_policy="one observed typed parent; one functional own-state prediction, global seal, one unedited native suffix"),
        "resource_exception": {"ram_mb": 32000, "vram_mb": 13000, "cpu_threads": 4,
            "calibration_job": "job-272e83f859d4", "measured_peak_vram_mb": 9418.0, "measured_peak_rss_mb": 11864.1,
            "reason": "same resident checkpoint/512px BF16/VAE profile; new full boundary cache is on CPU. Generalized-history ceiling15770MB exceeds host admission15557MB despite prior measured9418MB."},
        "selection_policy": "one fixed previously rendered prompt with fresh seed23 and pinned checkpoint; numerical continuation test; preserve all errors"}
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward",
        experiment="mdb-flux-trajectory-forecast", payload=run_dir / "payload", prefer_host="beast",
        gpu=True, needs=request["needs"], ram_mb=32000, vram_mb=13000, cpu_threads=4,
        est_wall_s=300, timeout_s=840, retry_on_kill=False, preflight=True,
        note="Donor-free Klein4B CUDA BF16, one fixed512px/4step/fresh-seed23 scene. Observed project0 parent; functional own-state111 transitions+VAE forecast, seal before111 native transitions+VAE comparison. Shared kernels; all module guards. CPUcache~2GB. VRAM13GB calibrated prior sameprofile9418MB peak, avoiding unschedulable history ceiling. No retries; stop after2images/archive.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        helper = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))
        result = helper["collect"](args.run_dir, args.collect, max_compressed_bytes=128 << 20, max_expanded_bytes=512 << 20)
    else:
        if args.inventory is None:
            parser.error("--inventory from a completed checkpoint inventory is required")
        result = stage(args.run_dir.resolve(), inventory=json.loads(args.inventory.read_text()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

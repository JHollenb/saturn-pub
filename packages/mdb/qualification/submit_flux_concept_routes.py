"""Admit a finite native-consumer map of conditioning and working-text routes."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def plan(inventory):
    suffix = " on a plain white background. Two separate objects, simple studio product photograph."
    return {"schema": "mdb-flux-concept-routes-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": inventory["model_path"], "expected_checkpoint_sha256": inventory["checkpoint_sha256"],
        "prompts": [
            {"id": "base", "text": "A red cube to the left of a blue sphere" + suffix},
            {"id": "relation", "text": "A red cube to the right of a blue sphere" + suffix},
            {"id": "color", "text": "A blue cube to the left of a red sphere" + suffix},
            {"id": "shape", "text": "A red sphere to the left of a blue cube" + suffix}],
        "seed": 31, "height": 512, "width": 512, "steps": 4, "guidance_scale": 1., "cpu_threads": 4,
        "cut_steps": [0, 1], "arms": ["text", "conditioning", "both"],
        "max_images": 24, "max_adapter_steps": 3000, "max_wall_seconds": 840,
        "max_archive_bytes": 512 << 20,
        "scope": "whole-field source/carrier factorization; observed native prefixes through step1 projection; no numerical forecast of altered images",
        "hypotheses": {
            "R-C": "Replacing stored conditioning while restoring recipient working text will move final images toward donor attributes/relations after later prompt rereads.",
            "R-T": "Replacing working text alone will be less persistent because later timesteps rebuild text from recipient conditioning.",
            "R-TC0": "Both carriers at step0 reproduce the donor numerical future exactly when image, latent, context and IDs are common.",
            "MLP": "First image MLP input/write responds to more than one prompt contrast; its whole-module role may be shared rather than one concept."},
        "controls": "same-parent two-Act shams at both cuts; coherent step0 donor both; native outputs for all four prompts",
        "semantics": "relation/color/shape identify supplied prompt contrasts; all semantic naming remains candidate; color centroids cannot certify shape",
        "terminal_status": "unassessed"}


def stage(run_dir, inventory):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=False)
    config = plan(inventory)
    save(run_dir / "plan.json", config)
    names = ["flux_concept_routes_campaign.py", "flux_concept_routes.py", "flux_trajectory_forecast_campaign.py",
             "campaign_support.py", "plan.json", "preflight.v1.json"]
    preflight = run_dir / names[-1]
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": names,
                     "path_token_allowlist": ["flux-concept-result", config["model_path"]]})
    extras = [(here / name, name) for name in names[1:3]] + [(here / "discovery_upgrades.py", "campaign_support.py"),
        (run_dir / "plan.json", "plan.json"), (preflight, preflight.name)]
    worker = here / names[0]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-concept-routes-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True},
        "backend": "diffusers", "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=24,
            logical_branches=24, logical_checkpoint_replays=24, max_local_work_items=32,
            checkpoint_policy="four prompt preparations; same native cut per recipient/timestep; 24 total RGB outcomes; no retries"),
        "resource_exception": {"ram_mb": 32000, "vram_mb": 13000, "calibration_job": "job-c2a467736b50",
            "measured_peak_vram_mb": 9470., "measured_peak_rss_mb": 10697.6,
            "measured_elapsed_seconds": 736.55,
            "reason": "completed same 24-image resident checkpoint/512px CUDA BF16 scalar profile; bounded cuts and port arrays; history ceiling previously exceeded host VRAM capacity"},
        "selection_policy": "four predeclared prompt contrasts, fresh seed31, two cut times; all outputs and counterexamples retained"}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward",
        experiment="mdb-flux-concept-routes", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=32000, vram_mb=13000, cpu_threads=4, est_wall_s=780,
        timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 scalar512px/4steps/seed31. One load:4native prompts (relation/color/shape),18carrier swaps+2shams=24RGB. Same-parent cuts at steps0/1; stored conditioning vs working text; native suffixes only. Prefixes observed and hypotheses sealed before RGB. Measured9470MB VRAM/736.55s. <=3000 adapter steps; no retries.")
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
        result = helper["collect"](args.run_dir, args.collect, max_compressed_bytes=512 << 20, max_expanded_bytes=1024 << 20)
    else:
        if args.inventory is None:
            parser.error("--inventory required")
        result = stage(args.run_dir.resolve(), json.loads(args.inventory.read_text()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

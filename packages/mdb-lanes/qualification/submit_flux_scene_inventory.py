"""Stage/submit/collect one matched natural-scene factorial through public MDB."""
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
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def stage(run_dir, *, submit=True):
    from mrun.protocol import kill_ceiling_mb

    from mdb_lanes.execution_environment import intent
    from mdb_lanes.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    config = {"schema": "mdb-flux-scene-inventory-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": "/mnt/big/llm-models/FLUX.2-klein-4B",
        "expected_checkpoint_sha256": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
        "seed": 47, "height": 512, "width": 512, "steps": 4, "cpu_threads": 4,
        "selector": {"logical_step": 1, "phase": "joint", "index": 4},
        **runpy.run_path(str(here/"flux_scene_inventory.py"))["make_plan"](),
        "max_adapter_steps": 3000, "max_wall_seconds": 840,
        "hypotheses": {
            "H-sham": "Same-parent image clone gives exact native scheduler latents and PNG.",
            "H-spatial-selectivity": "Tentative: shapeA.top causes greater red-component-region pixel change than shapeA.shifted, and colorB.top causes greater blue-component-region pixel change than colorB.shifted. Regions are measured only after all futures. Effects on the other object and background stay separate.",
            "H-persistence": "Tentative: full image-state replacement has more source-directed final RGB projection than top-quarter replacement for each factor. Recipient text and conditioning stay native. Repair or reversed ranks are counterexamples.",
            "H-target-collateral": "Tentative: shapeA top/full preserve B blue; colorB top/full preserve A red and left ordering. Relation effects may be distributed. Shape category observations are separate and exploratory.",
            "H-inventory": "Baseline compliance expectation: two major colored components, A red, B blue/green, centroid ordering follows requested left/right. This is a prompt-compliance expectation, not a discovered internal semantic law.",
        },
        "scope": "Eight fresh same-seed natural histories, full shapeA/colorB/relation factorial; early single4 and later joint4 observations; actual native attention/MLP returns;18 dependent final futures; prefix-only top-quarter supports and shifted controls; recipient text/C retained. No independently calculated RGB or automatic shape semantics.",
        "terminal_status": "unassessed"}
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir/"plan.json", config)
    worker = here/"flux_scene_inventory_campaign.py"
    extras = [(here/"flux_scene_inventory.py", "flux_scene_inventory.py"),
              (here/"discovery_upgrades.py", "campaign_support.py"), (run_dir/"plan.json", "plan.json")]
    preflight = run_dir/"preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
                    "hashed_roots": [worker.name, *(n for _, n in extras), preflight.name],
                    "path_token_allowlist": ["flux-scene-inventory-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    if kill_ceiling_mb(16000) >= 66512*.95:
        raise ValueError("RAM kill ceiling cannot fit reference host")
    request = {"schema": "mdb-flux-scene-inventory-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True}, "backend": "diffusers",
        "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {n: sha(p) for p, n in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=18,
            logical_branches=18, logical_checkpoint_replays=18, max_local_work_items=18,
            checkpoint_policy="8 natural prefixes, single4/joint4 StateCuts,18 native futures;8 encodes/1 resident load;B1;no nested jobs/retries/training;prefix-only supports sealed before RGB"),
        "resource_exception": {"ram_mb": 16000, "vram_mb": 13000, "calibration_job": "job-1f1587e6983b",
            "measured_peak_rss_mb": 16059, "measured_peak_vram_mb": 9282,
            "reason": "Same installed resident Klein CUDA BF16 B1 512px4step lane. Eight encodes serialized before encoder offload; bounded retained cuts and native ports.16GB reservation plus normal kill envelope fits66512MB host;13GB VRAM retains calibrated headroom."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir/"request.json", request)
    stage_worker(run_dir/"payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    if not submit:
        return {"run_dir": str(run_dir), "submitted": False, "request_sha256": request["request_sha256"]}
    job = guarded_mrun_launch(worker_command(worker.name), request=request, model=config["model"], task="forward",
        experiment="mdb-flux-scene-inventory", payload=run_dir/"payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=16000, vram_mb=13000, cpu_threads=4, est_wall_s=660,
        timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px seed47.8 natural shapeA/colorB/ordering cells;18 finalRGB=8refs+sham+3factors*(full/top/shifted). Actual A/M native ports,3prefixcuts/history;supports frozen before RGB;own recipient T/C always.1load/8serialencodes/no retries. Stop18images/3000adaptersteps/840s; exportgzip1.")
    save(run_dir/"submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--stage-only", action="store_true")
    p.add_argument("--collect")
    p.add_argument("--status")
    args = p.parse_args()
    if args.status:
        from mdb_lanes.job_support import terminal_job

        _, job, logs = terminal_job(args.status)
        phases = [line for line in logs.splitlines() if line.startswith("MDB_FLUX_SCENE_INVENTORY_PHASE ")]
        body = job.get("result") or {}
        result = {"job_id": args.status, "state": job["state"], "status_detail": job.get("status_detail"),
                  "resources": {k:body.get(k) for k in ("elapsed_s","peak_rss_mb","peak_vram_mb")},
                  "last_phases": phases[-3:], "last_error_lines": logs[-1600:] if job["state"]=="failed" else None}
    elif args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, log_storage="gzip", max_compressed_bytes=512 << 20, max_expanded_bytes=1 << 30)
    else:
        result = stage(args.run_dir.resolve(), submit=not args.stage_only)
    print(json.dumps(result, sort_keys=True))

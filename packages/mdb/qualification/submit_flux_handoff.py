"""Reuse two verified prior cuts in one finite within-step carrier experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def stage(run_dir, predecessor):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    run_dir.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    data = predecessor / "extracted/flux-concept-result"
    manifest = json.loads((data / "files.json").read_text())
    source = json.loads((data / "source.json").read_text())
    old_plan = json.loads((data / "plan.json").read_text())
    report = json.loads((predecessor / "report.json").read_text())
    if report["job_id"] != "job-c2a467736b50" or report["archive_sha256"] != "24454082c80c1c646b8a7e4948b948fa4765f49ff5ca4b1f4edff3e964cc409d":
        raise ValueError("unexpected predecessor")
    if sha(data / "source.json") != manifest["source.json"]:
        raise ValueError("predecessor source differs")
    cut_ids = {key: source["cuts"][key]["0"]["projected"] for key in ("base", "relation")}
    relatives = set()
    for cut_id in cut_ids.values():
        relative = f"typed-store/cuts/{cut_id}.json"
        relatives.add(relative)
        cut = json.loads((data / relative).read_text())
        relatives.update(f"typed-store/pages/{value}.json" for value in cut["slots"].values())
        relatives.update(f"typed-store/blobs/{value}.safetensors" for value in cut["physical_blobs"])
    extras = []
    for relative in sorted(relatives):
        path = data / relative
        if sha(path) != manifest[relative]:
            raise ValueError("predecessor CAS bytes differ")
        extras.append((path, relative.replace("typed-store/", "input-store/", 1)))
    config = {key: old_plan[key] for key in ("model", "model_path", "expected_checkpoint_sha256", "seed", "height", "width", "steps", "guidance_scale", "cpu_threads")}
    config.update(schema="mdb-flux-handoff-plan-v1", input_cuts=cut_ids,
        predecessor={"job_id": report["job_id"], "archive_sha256": report["archive_sha256"], "source_sha256": sha(data / "source.json")},
        prompts=old_plan["prompts"][:2], max_images=12, max_adapter_steps=1500, max_wall_seconds=840,
        boundaries=[{"id": "after_joint0", "selector": {"logical_step": 0, "phase": "joint", "index": 1}, "blocks_completed": 1},
            {"id": "after_joint4", "selector": {"logical_step": 0, "phase": "single", "index": 0}, "blocks_completed": 5},
            {"id": "after_single9", "selector": {"logical_step": 0, "phase": "single", "index": 10}, "blocks_completed": 15},
            {"id": "after_single18", "selector": {"logical_step": 0, "phase": "single", "index": 19}, "blocks_completed": 24}],
        hypotheses={"H-image": "After the late boundary, restoring base text will retain the changed spatial relation: image state carries the future.",
            "H-text": "At the early boundary, restoring base image will retain the changed spatial relation: remaining working text can still write it.",
            "H-handoff": "The relative influence of image versus text restoration shifts across the four boundaries; monotonicity is a hypothesis, not a gate.",
            "H-controls": "Late sham reproduces changed exactly; restoring both carriers reproduces base exactly."},
        scope="single known seed31 relation contrast; whole-carrier causal restores at four block boundaries, not numerical RGB forecasting or minimal semantic localization",
        terminal_status="unassessed")
    save(run_dir / "plan.json", config)
    extras += [(here / "flux_trajectory_forecast_campaign.py", "flux_trajectory_forecast_campaign.py"),
        (here / "discovery_upgrades.py", "campaign_support.py"), (run_dir / "plan.json", "plan.json")]
    worker = here / "flux_handoff_campaign.py"
    roots = [worker.name, "flux_trajectory_forecast_campaign.py", "campaign_support.py", "plan.json", "preflight.v1.json", "input-store"]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": roots,
        "path_token_allowlist": ["flux-handoff-result", "input-store", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    request = {"schema": "mdb-flux-handoff-request-v1", "config": config,
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True}, "backend": "diffusers",
        "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=12,
            logical_branches=12, logical_checkpoint_replays=12, max_local_work_items=16,
            checkpoint_policy="reuse two verified projected cuts; two references; eight single-carrier restores; two controls; no retries"),
        "resource_exception": {"ram_mb": 32000, "vram_mb": 13000, "calibration_job": "job-c2a467736b50",
            "measured_peak_vram_mb": 9470., "measured_peak_rss_mb": 10697.6, "measured_elapsed_seconds": 736.55,
            "reason": "same resident512px scalar CUDA BF16 checkpoint; half as many RGB outcomes and zero prompt encodings; automatic VRAM ceiling previously exceeded host"}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=config["model"], task="forward",
        experiment="mdb-flux-block-handoff", payload=run_dir / "payload", prefer_host="beast", gpu=True,
        needs=request["needs"], ram_mb=32000, vram_mb=13000, cpu_threads=4, est_wall_s=540,
        timeout_s=840, retry_on_kill=False, preflight=True,
        note="Klein4B CUDA BF16 B1 512px4steps seed31. Hydrate prior projected base/relation cuts, no encoder/prefix repeat. Working-text-only changed branch vs base; restore text or image after1/5/15/24blocks.2refs+8restores+2controls=12RGB,<=1500native steps. Sealed hypotheses; all signed effects retained. One load; measured prior9470MBVRAM; no retries.")
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--collect")
    args = parser.parse_args()
    if args.collect:
        result = runpy.run_path(str(Path(__file__).with_name("discovery_upgrades.py")))["collect"](
            args.run_dir, args.collect, max_compressed_bytes=256 << 20, max_expanded_bytes=512 << 20)
    else:
        if args.predecessor is None:
            parser.error("--predecessor required")
        result = stage(args.run_dir.resolve(), args.predecessor.resolve())
    print(json.dumps(result))

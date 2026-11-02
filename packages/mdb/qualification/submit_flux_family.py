"""Freeze, admit or directly collect one resident six-panel Klein campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path

WORKERS = ("flux_family_campaign.py", "flux_family_plan.py", "flux_family_ops.py", "flux_family_custody.py",
           "flux_parallel_merge.py", "flux_functional_denoiser.py", "flux_trajectory_forecast_campaign.py")
NOTE = ("One resident Klein4B CUDA BF16 B1,47 dependent512px/4step RGB futures across six panels: "
        "K/V/Q sources,R/A merge,contrast specificity,heldout resolver,support omission/rescue,records. "
        "Prefix-only supports;development fit then heldout seal before heldout suffixes. "
        "One lease/load,no per-row jobs,no retries;max10000 adapter steps,3000s worker. "
        "Calibrated18GiBRAM/13GBVRAM,cpu4;stop after all47 rows and bounded multipart archive.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def make_plan(prior_path):
    functions = runpy.run_path(str(Path(__file__).with_name("flux_family_plan.py")))
    result = functions["make_plan"](read(prior_path))
    result["historical_plan_sha256"] = sha(prior_path)
    return result


def resource_envelope(ram_mb=18432, *, kill_factor=None, reference_host_mb=66512, margin=.05):
    if kill_factor is None:
        from mrun.protocol import RAM_KILL_FACTOR, kill_ceiling_mb

        kill_factor, ceiling_source = RAM_KILL_FACTOR, "installed mrun.protocol.kill_ceiling_mb"
        ceiling = kill_ceiling_mb(ram_mb)
    else:
        ceiling_source, ceiling = "explicit multiplicative override", ram_mb * kill_factor
    require(all(type(value) in {int, float} and math.isfinite(value) for value in (ram_mb, kill_factor, reference_host_mb, margin))
            and ram_mb > 0 and kill_factor >= 1 and reference_host_mb > 0 and 0 <= margin < 1, "invalid memory admission geometry")
    available = reference_host_mb * (1 - margin)
    require(ceiling <= available, "RAM reservation kill ceiling exceeds reference host capacity with margin")
    return {"ram_mb": ram_mb, "ram_kill_factor": kill_factor, "ram_kill_ceiling_mb": ceiling,
            "ram_kill_factor_source": "installed mrun.protocol.RAM_KILL_FACTOR" if ceiling_source.startswith("installed") else "explicit override",
            "ram_kill_ceiling_source": ceiling_source,
            "reference_host_ram_mb": reference_host_mb, "reference_host_margin": margin,
            "reference_host_available_mb": available, "scope": "measured reference-host check; scheduler remains placement/admission authority"}


def stage(run_dir, plan_path):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here, run_dir, plan_path = Path(__file__).resolve().parent, Path(run_dir), Path(plan_path)
    config = read(plan_path)
    required = runpy.run_path(str(here / "flux_family_plan.py"))["make_plan"](config)
    require({key: config.get(key) for key in required} == required
            and not set(config) - set(required) - {"historical_plan_sha256"}, "campaign differs from declared six-panel geometry")
    require(len(config["rows"]) == config["max_images"] == 47 and len({row["id"] for row in config["rows"]}) == 47,
            "requires all47 unique declared dependent futures")
    require(len(NOTE) <= 500, "queue note exceeds finite scheduler bound")
    envelope = resource_envelope()
    run_dir.mkdir(parents=True, exist_ok=False)
    save(run_dir / "plan.json", config)
    extras = [(here / name, name) for name in WORKERS[1:]] + [(run_dir / "plan.json", "plan.json"), (plan_path, "campaign-plan.json")]
    preflight = run_dir / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKERS[0], *[name for _, name in extras], preflight.name],
                     "path_token_allowlist": ["flux-family-result", config["model_path"]]})
    extras.append((preflight, preflight.name))
    raw = (json.dumps(config, sort_keys=True, allow_nan=False) + "\n").encode()
    request = {"schema": "mdb-flux-family-request-v1", "config": config, "campaign_plan_sha256": sha(plan_path),
        "task_family": "forward", "model": config["model"], "needs": {"cuda": True}, "backend": "diffusers",
        "device": "cuda", "dtype": "bfloat16", "batch_size": 1, "height": 512, "width": 512,
        "execution_environment": intent(), "config_bytes_sha256": hashlib.sha256(raw).hexdigest(),
        "source_hashes": {name: sha(path) for path, name in [(here / WORKERS[0], WORKERS[0]), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_loads=1, logical_operations=47, logical_branches=47,
            logical_checkpoint_replays=47, max_local_work_items=64,
            checkpoint_policy="one resident model;development and heldout local phases;47dependent scalar futures;no scheduler per transition/row"),
        "resource_exception": {**envelope, "vram_mb": 13000, "cpu_threads": 4, "calibration_job": "job-230b7bf6de6d",
            "measured_peak_rss_mb": 11954, "scheduler_peak_rss_mb": 10055.8, "measured_peak_vram_mb_approx": 9500,
            "source_cache_bytes_from_declared_shapes": 566286336,
            "sizing_status": "conservative extrapolation from prior measured worker; exact47-row peak not yet measured",
            "reason": "same resident512px B1CUDA BF16checkpoint;540MiB sourcecache,one-at-a-time transients,metadata-only transactions,disk-streamed custody;independently reviewed18GiB envelope"},
        "selection_policy": "all47 frozen dependent rows;heldout responses never select supports or train resolver;retain all counterexamples"}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=here / WORKERS[0], config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(WORKERS[0])
    # Public wheel installation and offline execution remain the payload helper's contract.
    job = guarded_mrun_launch(command, request=request, experiment="mdb-flux-six-panel-family", task="forward", model=config["model"],
        payload=run_dir / "payload", prefer_host="beast", gpu=True, needs=request["needs"], ram_mb=envelope["ram_mb"], vram_mb=13000,
        cpu_threads=4, est_wall_s=1800, timeout_s=3300, retry_on_kill=False, preflight=True, note=NOTE)
    save(run_dir / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job), "run_dir": str(run_dir), "rows": 47, "model_loads": 1}


def collect(run_dir, job_id):
    run_dir = Path(run_dir)
    submission = run_dir / "submission.json"
    require(not submission.exists() or read(submission)["job_id"] == job_id, "collection must address the exact submitted job")
    limits = read(run_dir / "request.json")["config"]["limits"]
    return runpy.run_path(str(Path(__file__).with_name("flux_family_custody.py")))["collect"](run_dir, job_id, limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--make-plan", type=Path, help="prior pinned plan supplying checkpoint identity")
    actions.add_argument("--plan", type=Path, help="frozen47-row campaign")
    actions.add_argument("--collect", help="exact completed job; never launches/retries models")
    args = parser.parse_args()
    if args.make_plan:
        save(args.path, make_plan(args.make_plan))
        result = {"plan": str(args.path), "sha256": sha(args.path), "rows": 47}
    elif args.collect:
        result = collect(args.path, args.collect)
    else:
        result = stage(args.path.resolve(), args.plan.resolve())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

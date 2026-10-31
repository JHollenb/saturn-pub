"""Admit one finite diffusion MLP forecast or a model-free checkpoint inventory."""
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
    return {"schema": "mdb-flux-mlp-forecast-plan-v1", "model": "flux2-klein-4b-distilled",
        "model_path": inventory["model_path"], "expected_checkpoint_sha256": inventory["checkpoint_sha256"],
        "prompts": [
            {"id": "left", "text": "A red cube to the left of a blue sphere on a plain white background. Two separate objects, simple studio product photograph."},
            {"id": "right", "text": "A red cube to the right of a blue sphere on a plain white background. Two separate objects, simple studio product photograph."}],
        "seed": 17, "height": 512, "width": 512, "steps": 4, "guidance_scale": 1., "cpu_threads": 4,
        "target_block": 0, "target_step": 0,
        "arms": ["native", "sham", "forecastself", "donorfull", "donorleft", "donorright"],
        "max_images": 12, "max_adapter_steps": 2048, "max_archive_bytes": 512 << 20,
        "forecast_scope": "observed image FF input/residual/gate, external checkpoint weights; independent FP64 FF and gated write; RGB is native observation",
        "semantic_hypothesis": "donor gated-write transfer shifts red-cube versus blue-sphere horizontal ordering toward donor; retain object loss, unchanged relations and collateral",
        "mask_authority": "fixed coordinate halves from actual image IDs; supplied spatial addresses, not discovered object masks",
        "controls": "actual-native copy sham; analytic self-write reinstall measures downstream effect of approximation; left/right preserve unmasked native rows and recipient text",
        "terminal_status": "unassessed"}


def stage(run_dir, *, inventory=None):
    from mdb.execution_environment import intent
    from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
    from mdb.payload import stage_worker, worker_command

    here = Path(__file__).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=False)
    model_free = inventory is None
    config = {"model": "flux2-klein-4b-distilled"} if model_free else plan(inventory)
    worker = here / ("flux_model_inventory.py" if model_free else "flux_mlp_forecast_campaign.py")
    save(run_dir / "plan.json", config)
    preflight = run_dir / "preflight.v1.json"
    names = [worker.name, "campaign_support.py", "plan.json", preflight.name]
    if not model_free:
        names.append("flux_mlp_equations.py")
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": names,
        "path_token_allowlist": ["flux-inventory-result", "flux-mlp-result"]
            + ([] if model_free else [config["model_path"]])})
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    extras = [(here / "discovery_upgrades.py", "campaign_support.py"),
              (run_dir / "plan.json", "plan.json"), (preflight, preflight.name)]
    if not model_free:
        extras.append((here / "flux_mlp_equations.py", "flux_mlp_equations.py"))
    request = {"schema": "mdb-flux-forecast-request-v1", "config": config,
        "task_family": "analysis" if model_free else "forward", "model": None if model_free else config["model"],
        "needs": {"cpu": True} if model_free else {"cuda": True},
        "backend": "metadata" if model_free else "diffusers", "device": "cpu" if model_free else "cuda",
        "dtype": "float32" if model_free else "bfloat16", "batch_size": 1,
        "height": config.get("height", 512), "width": config.get("width", 512),
        "execution_environment": intent(), "config_bytes_sha256": sha(raw),
        "source_hashes": {name: sha(path.read_bytes()) for path, name in [(worker, worker.name), *extras]},
        "scheduling_contract": one_lease_scheduling_contract(model_free=model_free, model_loads=0 if model_free else 1,
            logical_operations=1 if model_free else 12, logical_branches=None if model_free else 12,
            logical_checkpoint_replays=0 if model_free else 12,
            extra={"task_family": "analysis"} if model_free else None,
            max_local_work_items=64, checkpoint_policy="model-free inventory" if model_free else
            "one resident Klein; genuine partial prefixes, one forecast seal, typed common-parent branches and native RGB"),
        "selection_policy": "two fixed mirrored scenes and one seed; preserve all numerical and semantic outcomes"}
    request["request_sha256"] = sha(json.dumps(request, sort_keys=True, allow_nan=False).encode())
    save(run_dir / "request.json", request)
    stage_worker(run_dir / "payload", worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
    command = worker_command(worker.name)
    if "--offline" in command:
        command.remove("--offline")
    job = guarded_mrun_launch(command, request=request, model=request["model"], task=request["task_family"],
        experiment="mdb-flux-checkpoint-inventory" if model_free else "mdb-flux-mlp-forecast",
        payload=run_dir / "payload", prefer_host="beast", gpu=not model_free, needs=request["needs"],
        timeout_s=180 if model_free else 840, retry_on_kill=False, preflight=True,
        note="No model load: resolve/hash local Klein checkpoint and config." if model_free else
        "One finite CUDA BF16 Klein4B:2mirrored512px scenes,4steps,seed17,12nativeRGB branches; joint0 FF/write forecasts sealed first. External FF weights; CPU FP64 math. Actual sham+self-forecast+donor full/halves, <=2048 adapter steps, no retries. Stop after images/archive.")
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
        result = stage(args.run_dir.resolve(), inventory=None if args.inventory is None else json.loads(args.inventory.read_text()))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

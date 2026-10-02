"""Finite, source-sealed MDB qualification through one mrun lease."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from mdb.bench_resources import diffusion_resources
from mdb.job_support import (guarded_mrun_launch, one_lease_restart_proof_scheduling_contract,
    source_hash_manifest, terminal_job)
from mdb.payload import package_files, stage_worker, worker_command
from mdb.tools import scheduler_api
from mdb.execution_environment import intent

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = {"qwen": "qwen2.5-0.5b-instruct", "mamba": "mamba-130m",
          "flux": "flux2-klein-4b-distilled", "sdxl": "illustrious-xl"}
EXPECTED_WEIGHTS = {
    "qwen": "2b226fe4944dc744596b1142a15d6676160e9d06f245d31c1a36633f6c86baa3",
    "mamba": "afde3cdedd34132cfe0653204f1066abf4489aaa1c737bddeba03a4b599855b8",
    "flux": "dd976d67036c4a0a55642f702dcb3740ba7a4f367557596572baac168330904f",
    "sdxl": "7927c57ac742f6cb244638e01553d0ce7b4df2e699ea8a55c2e8b93170c0ae74"}


def specification(family):
    kind = "qwen" if family == "qwen" else "families"
    config = json.loads((HERE / f"{kind}_config.json").read_text())
    config.update(model=MODELS[family], family=family, cpu_threads=4,
                  expected_weights_sha256=EXPECTED_WEIGHTS[family],
                  max_wall_seconds=840 if family == "qwen" else 6000)
    profile = None
    basis = "intent-first model planning"
    if family in {"flux", "sdxl"}:
        profile, basis = diffusion_resources(config["model"], height=512, width=512, steps=4,
                                             state_budget_bytes=config["state_budget_bytes"])
        if basis.get("phase_vram_budget_decimal_mb") is not None:
            config["phase_vram_budget_mb"] = basis["phase_vram_budget_decimal_mb"]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    worker = HERE / f"{kind}_worker.py"
    extras = [(HERE / "debugger.script", "debugger.script")] if family == "qwen" else []
    sources = {package + "/" + relative: path for package in ("mdb", "mrun")
               for relative, path in package_files(package).items()}
    sources["worker.py"] = worker
    sources.update({name: path for path, name in extras})
    request = {"schema": "mdb-qualification-request-v1", "model": config["model"], "family": family,
        "task_family": "forward",
        "device": "cuda", "needs": {"cuda": True}, "config_bytes_sha256": sha256(raw).hexdigest(),
        "source_hashes": source_hash_manifest(sources), "resource_basis": basis,
        "resource_profile": profile.as_dict() if profile else None,
        "scheduler_endpoint": scheduler_api()._resolve(),
        "geometry": {"simultaneous_models": 1, "tokens": 32 if family in {"qwen", "mamba"} else None,
                     "prompts": 3, "seeds": 1 if family in {"qwen", "mamba"} else 2,
                     "height": 512, "width": 512, "steps": 4},
        "timeout_s": config["max_wall_seconds"], "execution_environment": intent(),
        "scheduling_contract": one_lease_restart_proof_scheduling_contract(
            student_process_reconstructions=2, logical_operations=128, logical_checkpoint_replays=64,
            max_local_work_items=256, checkpoint_policy="sequential resident parent and fresh child; no retry")}
    request["settings_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    return config, raw, request, worker, extras, profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "submit", "collect"))
    parser.add_argument("family", choices=MODELS)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.action == "collect":
        job_id, job, logs = terminal_job(args.job, wait=False)
        output = ROOT / "results/qualification" / job_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "job.json").write_text(json.dumps(job, indent=2))
        (output / "worker.log").write_text(logs)
        marker = "SATURN_BENCH_PROOF_REPORT=" if args.family == "qwen" else "BENCH_FAMILY_REPORT="
        report = next((json.loads(line[len(marker):]) for line in logs.splitlines() if line.startswith(marker)), None)
        if report:
            request = json.loads((output / "request.json").read_text())
            if report["request_sha256"] != request["request_sha256"] or report["source_hashes"] != request["source_hashes"]:
                raise RuntimeError("qualification report/source seal mismatch")
            (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"job_id": job_id, "state": job.get("state"), "report": bool(report)}))
        return
    config, raw, request, worker, extras, profile = specification(args.family)
    if args.action == "preflight":
        print(json.dumps({"config": config, "request": request}, sort_keys=True))
        return
    review = json.loads(args.review.read_text()) if args.review else None
    if args.family != "qwen" and (not review or review.get("status") != "reviewed"
            or review.get("source_hashes") != request["source_hashes"]
            or request["settings_sha256"] not in review.get("settings_sha256", [])):
        raise RuntimeError("long proof requires independent review bound to these exact sources and settings")
    request["runtime_review"] = review
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-proof-") as directory:
        stage_worker(directory, worker=worker, config=config, request=request, config_bytes=raw,
                     additional_files=extras)
        command = worker_command(worker.name)
        command = [command[0], f"MRUN_SERVER_URL={request['scheduler_endpoint']}",
                   "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command, request=request, experiment="mdb-standalone-qualification",
            task=f"proof-{args.family}", model=config["model"], payload=Path(directory),
            prefer_host="beast", gpu=True, needs={"cuda": True}, detach=True,
            ram_mb=profile.ram_mb if profile else None, vram_mb=profile.vram_mb if profile else None,
            disk_gb=profile.disk_gb if profile else None, cpu_threads=profile.cpu_threads if profile else None,
            timeout_s=config["max_wall_seconds"],
            retry_on_kill=False, preflight=True,
            note=f"MDB {args.family}: 3 prompts;32 tokens or 2 seeds,512px,4 steps; cuts, native parity, siblings, eviction, fresh child, debugger, discovery. One resident model and lease; sequential CUDA; no retries.")
    output = ROOT / "results/qualification" / job
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2))
    (output / "config.json").write_bytes(raw)
    print(json.dumps({"job_id": job, "settings_sha256": request["settings_sha256"]}))


if __name__ == "__main__":
    main()

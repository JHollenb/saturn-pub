"""Finite legacy continuation assay, one admitted model and no retries."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from mdb.bench_resources import diffusion_resources
from mdb.execution_environment import intent
from mdb.job_support import guarded_mrun_launch, source_hash_manifest, one_lease_scheduling_contract
from mdb.payload import package_files, stage_worker, worker_command
from mdb.tools import scheduler_api

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("mamba", "flux", "sdxl"))
    args = parser.parse_args()
    config = json.loads((HERE / "legacy-configs.json").read_text())[args.family]
    if config.get("prepared_store"):
        prior_job = Path(config["prepared_store"]).parent.name
        prior = scheduler_api().json("GET", f"/api/jobs/{prior_job}")
        if prior["state"] not in {"failed", "succeeded", "cancelled", "timeout"}:
            raise RuntimeError("prepared handoff source job must be terminal")
        config["prepared_source_job"] = {"job_id": prior_job, "state": prior["state"],
            "job_record_sha256": sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()}
    worker = HERE / "legacy_worker.py"
    extras = [(HERE / "families_worker.py", "family_helpers.py")]
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    profile = None
    if args.family != "mamba":
        profile, basis = diffusion_resources(config["model"], height=512, width=512, steps=4,
                                             state_budget_bytes=512 << 20)
        if basis.get("phase_vram_budget_decimal_mb"):
            config["phase_vram_budget_mb"] = basis["phase_vram_budget_decimal_mb"]
            raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    sources = {package + "/" + relative: path for package in ("mdb", "mrun")
               for relative, path in package_files(package).items()}
    sources.update({"worker.py": worker, "family_helpers.py": extras[0][0]})
    request = {"schema": "mdb-legacy-request-v1", "model": config["model"], "task_family": "forward",
               "device": "cuda", "needs": {"cuda": True}, "execution_environment": intent(),
               "source_hashes": source_hash_manifest(sources),
               "config_bytes_sha256": sha256(raw).hexdigest(),
               "scheduler_endpoint": scheduler_api()._resolve(),
               "scheduling_contract": one_lease_scheduling_contract(logical_operations=1,
                   logical_checkpoint_replays=1, checkpoint_policy="copied legacy cut; one finite suffix")}
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-legacy-") as directory:
        stage_worker(directory, worker=worker, config=config, request=request,
                     config_bytes=raw, additional_files=extras)
        command = worker_command(worker.name)
        command = [command[0], f"MRUN_SERVER_URL={request['scheduler_endpoint']}",
                   "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command, request=request, model=config["model"],
            experiment="mdb-legacy-continuation", task=args.family, payload=Path(directory),
            prefer_host="beast", gpu=True, needs={"cuda": True}, detach=True,
            ram_mb=profile.ram_mb if profile else None, vram_mb=profile.vram_mb if profile else None,
            cpu_threads=profile.cpu_threads if profile else None,
            disk_gb=profile.disk_gb if profile else None, timeout_s=840,
            retry_on_kill=False, preflight=True,
            note="MDB legacy copy: verify old export, resume one Mamba token or four-step image suffix. One model, one lease, no retries; original store read-only.")
    output = HERE.parent / "results/qualification" / job
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2))
    (output / "config.json").write_bytes(raw)
    print(json.dumps({"job_id": job}))


if __name__ == "__main__": main()

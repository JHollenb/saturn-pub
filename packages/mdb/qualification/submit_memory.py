"""One finite lease for the outstanding Mamba memory mechanics panel."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from mdb.execution_environment import intent
from mdb.job_support import guarded_mrun_launch, source_hash_manifest, one_lease_restart_proof_scheduling_contract
from mdb.payload import package_files, stage_worker, worker_command
from mdb.tools import scheduler_api

HERE = Path(__file__).resolve().parent


def main():
    parent = HERE.parent/"results/qualification/job-eb8c686d27ed/report.json"
    raw_parent = parent.read_bytes(); proof = json.loads(raw_parent)
    if proof["family"] != "mamba" or any(not value for key, value in proof["mechanics"].items() if key != "memory"):
        raise RuntimeError("parent execution panels are not qualified")
    case = proof["cases"][0]
    config = {"model": "mamba-130m", "family": "mamba", "cpu_threads": 4,
        "expected_weights_sha256": case["identity"]["weights_sha256"],
        "expected_identity": case["identity"], "expected": case["baseline"], "prompt": case["prompt"],
        "state_budget_bytes": 536870912,
        "parent_proof": {"job_id": "job-eb8c686d27ed", "report_sha256": sha256(raw_parent).hexdigest(),
                         "boundary": "three native/stepped/replay/fresh cases qualified; memory instrument underfilled"}}
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    worker = HERE/"memory_worker.py"
    extras = [(HERE/"families_worker.py", "family_helpers.py")]
    sources = {package+"/"+relative: path for package in ("mdb", "mrun")
               for relative, path in package_files(package).items()}
    sources.update({"worker.py": worker, "family_helpers.py": extras[0][0]})
    request = {"schema": "mdb-memory-completion-request-v1", "model": config["model"], "task_family": "forward",
        "device": "cuda", "needs": {"cuda": True}, "config_bytes_sha256": sha256(raw).hexdigest(),
        "source_hashes": source_hash_manifest(sources), "execution_environment": intent(),
        "scheduler_endpoint": scheduler_api()._resolve(), "timeout_s": 840,
        "storage": {"root": "/mnt/hdd8tb/mdb/qualification", "filesystem": "ext4", "reason": "FUSE/NTFS export ENOSPC despite available bytes"},
        "geometry": {"simultaneous_models": 1, "prompt_cases": 1, "tokens": 32, "private_branches": 3},
        "scheduling_contract": one_lease_restart_proof_scheduling_contract(student_process_reconstructions=2,
            logical_operations=128, logical_checkpoint_replays=4, max_local_work_items=256,
            checkpoint_policy="three private cuts; source removed before fresh child; no retry")}
    inventory = HERE.parent/"results/qualification/cache-materialization-v2.json"
    request["cache_materialization"] = {"path": "/mnt/hdd8tb/mdb/cache-materialization-v2.json",
                                       "sha256": sha256(inventory.read_bytes()).hexdigest()}
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-memory-") as directory:
        stage_worker(directory, worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
        command = worker_command(worker.name)
        command = [command[0], f"MRUN_SERVER_URL={request['scheduler_endpoint']}", "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command, request=request, experiment="mdb-memory-completion", task="mamba",
            model=config["model"], payload=Path(directory), prefer_host="beast", gpu=True, needs={"cuda": True},
            detach=True, timeout_s=840, retry_on_kill=False, preflight=True,
            note="Focused Mamba memory completion: one prompt,32tokens; three siblings with measured eviction/hydration, private-cut replay, ext4 export, remove source, fresh child. Prior3-case execution proof retained. One model/lease; no retries.")
    output = HERE.parent/"results/qualification"/job
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_bytes(raw); (output/"request.json").write_text(json.dumps(request, indent=2))
    print(json.dumps({"job_id": job}))


if __name__ == "__main__": main()

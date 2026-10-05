"""A finite continuation, without repeating the completed memory execution."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from mdb.execution_environment import intent
from mdb.job_support import guarded_mrun_launch, source_hash_manifest, one_lease_scheduling_contract
from mdb.payload import package_files, stage_worker, worker_command
from mdb.tools import scheduler_api

HERE = Path(__file__).resolve().parent


def main():
    parent = HERE.parent / "results/qualification/job-d6aa0de67add"
    config = json.loads((parent / "config.json").read_text())
    config.update(parent_job=parent.name, panel_sha256=sha256((parent / "panel-before-export.json").read_bytes()).hexdigest())
    raw = (json.dumps(config, sort_keys=True)+"\n").encode()
    worker = HERE / "memory_finish_worker.py"
    extras = [(HERE / "families_worker.py", "family_helpers.py")]
    sources = {package+"/"+relative: path for package in ("mdb", "mrun")
               for relative, path in package_files(package).items()}
    sources.update({"worker.py": worker, "family_helpers.py": extras[0][0]})
    inventory = HERE.parent / "results/qualification/cache-materialization-v2.json"
    request = {"schema": "mdb-memory-finish-request-v1", "model": config["model"], "task_family": "forward",
        "device": "cuda", "needs": {"cuda": True}, "config_bytes_sha256": sha256(raw).hexdigest(),
        "source_hashes": source_hash_manifest(sources), "execution_environment": intent(),
        "scheduler_endpoint": scheduler_api()._resolve(), "timeout_s": 840,
        "cache_materialization": {"path": "/mnt/hdd8tb/mdb/cache-materialization-v2.json",
                                  "sha256": sha256(inventory.read_bytes()).hexdigest()},
        "scheduling_contract": one_lease_scheduling_contract(logical_operations=32,
            logical_checkpoint_replays=1, checkpoint_policy="retained export, delete terminal parent private source, fresh32tokens; no repeat")}
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-memory-finish-") as directory:
        stage_worker(directory, worker=worker, config=config, request=request, config_bytes=raw, additional_files=extras)
        command = worker_command(worker.name)
        command = [command[0], f"MRUN_SERVER_URL={request['scheduler_endpoint']}", "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command, request=request, experiment="mdb-memory-completion", task="mamba-finish",
            model=config["model"], payload=Path(directory), prefer_host="beast", gpu=True, needs={"cuda": True},
            detach=True, timeout_s=840, retry_on_kill=False, preflight=True,
            note="Continue retained Mamba memory export only: audit/copy, remove terminal parent private source, one fresh32token continuation. Earlier native/3private replay/eviction panels preserved. One model/lease, no retries.")
    output = HERE.parent / "results/qualification" / job
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(raw); (output / "request.json").write_text(json.dumps(request, indent=2))
    print(json.dumps({"job_id": job}))


if __name__ == "__main__": main()

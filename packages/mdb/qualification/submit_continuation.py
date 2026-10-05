"""Submit only the unfinished portable Qwen continuation panel."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from mdb.execution_environment import intent
from mdb.job_support import guarded_mrun_launch, source_hash_manifest, one_lease_scheduling_contract
from mdb.payload import package_files, stage_worker, worker_command
from mdb.tools import scheduler_api

HERE = Path(__file__).resolve().parent
PARENT = "job-697f7259acdb"


def main():
    resume = HERE.parent / "results/qualification" / PARENT / "resume.json"
    config = {"model": "qwen2.5-0.5b-instruct", "parent_job": PARENT,
              "parent_resume_sha256": sha256(resume.read_bytes()).hexdigest()}
    raw = (json.dumps(config, sort_keys=True) + "\n").encode()
    worker = HERE / "qwen_finish_worker.py"
    extras = [(HERE / "qwen_worker.py", "qwen_helpers.py"), (resume, "parent-resume.json")]
    sources = {package + "/" + relative: path for package in ("mdb", "mrun")
               for relative, path in package_files(package).items()}
    sources.update({"worker.py": worker, **{name: path for path, name in extras}})
    request = {"schema": "mdb-continuation-request-v1", "model": config["model"], "task_family": "forward",
               "device": "cuda", "needs": {"cuda": True}, "execution_environment": intent(),
               "source_hashes": source_hash_manifest(sources),
               "config_bytes_sha256": sha256(raw).hexdigest(),
               "scheduler_endpoint": scheduler_api()._resolve(),
               "scheduling_contract": one_lease_scheduling_contract(logical_operations=7,
                   logical_checkpoint_replays=7, checkpoint_policy="seven source-deleted parent exports; remaining continuation only")}
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-continuation-") as directory:
        stage_worker(directory, worker=worker, config=config, request=request,
                     config_bytes=raw, additional_files=extras)
        command = worker_command(worker.name)
        command = [command[0], f"MRUN_SERVER_URL={request['scheduler_endpoint']}",
                   "CUBLAS_WORKSPACE_CONFIG=:4096:8", *command[1:]]
        job = guarded_mrun_launch(command, request=request, model=config["model"],
            experiment="mdb-portable-continuation", task="qwen", payload=Path(directory),
            prefer_host="beast", gpu=True, needs={"cuda": True}, detach=True,
            timeout_s=840, retry_on_kill=False, preflight=True,
            note="Complete interrupted Qwen continuation panel: seven copied exports,224tokens total, three stock-native comparisons. One model, one lease; no parent suite repeat or automatic retry.")
    output = HERE.parent / "results/qualification" / job
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2))
    (output / "config.json").write_bytes(raw)
    print(json.dumps({"job_id": job}))


if __name__ == "__main__": main()

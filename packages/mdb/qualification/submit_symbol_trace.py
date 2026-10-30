"""Stage or directly collect one finite CUDA symbol-trace workflow."""

import argparse
import base64
import hashlib
import json
import tempfile
from pathlib import Path

from mdb.execution_environment import intent
from mdb.job_support import (
    decode_marker,
    guarded_job_receipt,
    guarded_mrun_launch,
    one_lease_scheduling_contract,
    terminal_job,
)
from mdb.payload import stage_worker, worker_command

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results/symbol-trace"


def write_once(path, raw):
    if path.exists():
        if path.read_bytes() != raw:
            raise RuntimeError("retained terminal evidence changed: " + str(path))
    else:
        with path.open("xb") as handle:
            handle.write(raw)


def json_once(path, value):
    write_once(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def collect(job_id):
    identifier, job, logs = terminal_job(job_id)
    if job.get("state") not in {
        "succeeded", "failed", "killed_ram", "killed_vram", "timeout", "cancelled", "lost"
    }:
        raise RuntimeError("job is still active; collect the same job after it finishes")
    root = RESULTS / identifier
    root.mkdir(parents=True, exist_ok=True)
    json_once(root / "job.json", job)
    write_once(root / "worker.log", logs.encode())
    json_once(root / "custody.json", guarded_job_receipt(identifier))
    report = decode_marker(logs, "SYMBOL_TRACE_REPORT_V1 ")
    if report is not None:
        if report.get("schema") != "mdb-symbol-trace-qualification-v1":
            raise RuntimeError("worker report is malformed: " + str(report.get("error")))
        json_once(root / "report.json", report)
        marker = "SYMBOL_TRACE_ARCHIVE_V1 "
        encoded = next((line[len(marker):] for line in logs.splitlines()
                        if line.startswith(marker)), None)
        if encoded is None:
            raise RuntimeError("completed report has no retained project archive")
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != report["archive"]["sha256"]:
            raise RuntimeError("native project archive differs from its worker seal")
        write_once(root / "native-project.tar.gz", raw)
    return {"job_id": identifier, "state": job.get("state"),
            "report": report is not None, "output": str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", metavar="JOB_ID")
    args = parser.parse_args()
    if args.collect:
        print(json.dumps(collect(args.collect)))
        return
    config = {
        "model": "qwen2.5-0.5b-instruct", "task_family": "forward", "backend": "hf",
        "device": "cuda", "dtype": "float32", "seq_len": 8,
        "prompt": "The capital of France is", "max_prompt_tokens": 8,
        "backend_options": {"physical_batch_size": 1, "full_prefill_logits": True},
    }
    worker = HERE / "symbol_trace_worker.py"
    driver = HERE.parent / "examples/symbol-trace/run_demo.py"
    request = {
        **config, "schema": "mdb-symbol-trace-qualification-request-v1",
        "execution_environment": intent(),
        "needs": {"cuda": True},
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (worker, driver)},
        "geometry": {"resident_models": 1, "prompt_tokens_max": 8,
                     "logical_branch_limit": 8, "interventions": 1,
                     "native_transition_limit": 256},
        "scheduling_contract": one_lease_scheduling_contract(
            logical_branches=8, logical_operations=256, logical_checkpoint_replays=1,
            max_local_work_items=256,
        ),
    }
    request["request_sha256"] = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-symbol-trace-") as directory:
        stage_worker(directory, worker=worker, additional_workers=(driver,),
                     config=config, request=request)
        job_id = guarded_mrun_launch(
            worker_command(worker.name), request=request, experiment="mdb-symbol-trace",
            task="forward", model=config["model"], payload=Path(directory),
            prefer_host="beast", gpu=True, needs={"cuda": True}, timeout_s=300,
            retry_on_kill=False, preflight=True,
            note=("Symbol history qualification: one Qwen0.5B FP32/eager resident, <=8 prompt "
                  "tokens, two time-qualified bindings and one zero-carrier fork, <=256 native "
                  "transitions including full first-token suffixes, exact restore/replay and "
                  "offline trace reopen; stop after one "
                  "report and archive or 300s. One finite CUDA lease; no retries."),
        )
    root = RESULTS / str(job_id)
    root.mkdir(parents=True, exist_ok=False)
    json_once(root / "request.json", request)
    print(json.dumps({"job_id": str(job_id), "output": str(root)}))


if __name__ == "__main__":
    main()

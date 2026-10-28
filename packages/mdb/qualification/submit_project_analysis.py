"""Stage installed public runtimes and submit one finite project workflow qualification."""

import argparse
import json
import tempfile
from pathlib import Path

from mdb.job_support import guarded_mrun_launch, one_lease_scheduling_contract
from mdb.payload import stage_worker, worker_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", metavar="JOB_ID")
    args = parser.parse_args()
    if args.collect:
        from mdb.job_support import guarded_job_receipt, terminal_job

        job_id, job, logs = terminal_job(args.collect)
        output = Path(__file__).parents[1] / "results/project-analysis" / job_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        (output / "worker.log").write_text(logs)
        (output / "custody.json").write_text(
            json.dumps(guarded_job_receipt(job_id), indent=2) + "\n"
        )
        marker = "MDB_PROJECT_ANALYSIS_REPORT="
        report = next(
            (
                json.loads(line[len(marker) :])
                for line in logs.splitlines()
                if line.startswith(marker)
            ),
            None,
        )
        if report:
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "state": job.get("state"),
                    "report": bool(report),
                    "output": str(output),
                }
            )
        )
        return
    config = {
        "model": "qwen2.5-0.5b-instruct",
        "task_family": "forward",
        "backend": "hf",
        "device": "cuda",
        "dtype": "float32",
        "seq_len": 8,
        "backend_options": {"physical_batch_size": 1, "full_prefill_logits": False},
    }
    request = {
        **config,
        "schema": "mdb-project-analysis-qualification-request-v1",
        "needs": {"cuda": True},
        "geometry": {
            "models": 1,
            "prompt_tokens": 8,
            "physical_transitions": 8,
            "interventions": 1,
        },
        "scheduling_contract": one_lease_scheduling_contract(
            logical_branches=2,
            logical_operations=24,
            logical_checkpoint_replays=1,
            max_local_work_items=32,
        ),
    }
    with tempfile.TemporaryDirectory(prefix="mdb-project-analysis-") as directory:
        stage_worker(
            directory,
            worker=Path(__file__).with_name("project_analysis_worker.py"),
            config=config,
            request=request,
        )
        job = guarded_mrun_launch(
            worker_command("project_analysis_worker.py"),
            request=request,
            experiment="mdb-project-analysis-qualification",
            task="forward",
            model=config["model"],
            payload=Path(directory),
            prefer_host="beast",
            gpu=True,
            needs={"cuda": True},
            timeout_s=300,
            retry_on_kill=False,
            preflight=True,
            note="Project CLI qualification: one Qwen0.5B FP32 resident, <=8 prompt tokens, 8 native transitions, 1 intervention, offline framework-free reopen; stop after report or 300s. One lease, no retries.",
        )
    print(json.dumps({"job_id": job, "request": request}))


if __name__ == "__main__":
    main()

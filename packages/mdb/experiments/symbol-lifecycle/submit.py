"""Submit one guarded CUDA symbol-lifecycle assay through mrun.

The script only prepares and submits the job.  ``--wait`` explicitly collects
the one finite lease and decodes the worker's content-addressed log marker.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results/symbol-lifecycle"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--wait", action="store_true", help="wait for the submitted job and print the report")
    parser.add_argument("--collect", metavar="JOB_ID", help="directly collect a submitted job into the immutable results directory")
    parser.add_argument("--host", default="beast", help="soft host preference; CUDA need remains authoritative")
    return parser.parse_args(argv)


def submit(args: argparse.Namespace) -> dict:
    from mdb.bench_loading import resolve_bench_model
    from mdb.execution_environment import intent
    from mdb.job_support import (
        decode_marker,
        guarded_mrun_launch,
        one_lease_scheduling_contract,
        source_hash_manifest,
        stage_worker,
        terminal_job,
        worker_command,
    )

    if args.collect:
        from mdb.job_support import guarded_job_receipt, terminal_job

        job_id, body, logs = terminal_job(args.collect, wait=args.wait, max_wait_s=900)
        output = RESULTS / job_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "job.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "worker.log").write_text(logs, encoding="utf-8")
        custody = guarded_job_receipt(job_id)
        (output / "custody.json").write_text(json.dumps(custody, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = decode_marker(logs, "SYMBOL_LIFECYCLE_REPORT_V1 ")
        if report is not None:
            (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        archive = next(
            (line[len("SYMBOL_LIFECYCLE_PROJECT_ARCHIVE_V1 ") :]
             for line in logs.splitlines()
             if line.startswith("SYMBOL_LIFECYCLE_PROJECT_ARCHIVE_V1 ")),
            None,
        )
        if archive:
            import base64

            archive_bytes = base64.b64decode(archive)
            archive_path = output / "native-project-archive.tar.gz"
            archive_path.write_bytes(archive_bytes)
            if report and report.get("native_project_archive", {}).get("sha256") != sha256(archive_bytes).hexdigest():
                raise RuntimeError("collected native project archive does not match the worker report seal")
        result = {
            "job_id": job_id,
            "state": body.get("state"),
            "report": report is not None,
            "archive": bool(archive),
            "output": str(output),
        }
        return result

    worker = Path(__file__).with_name("worker.py").resolve()
    spec = resolve_bench_model(args.model)
    contract = one_lease_scheduling_contract(
        # One shared baseline plus seven one-token candidate streams keeps the
        # counted native transitions and registered Acts below the 256-step
        # execution budget for Qwen2.5-0.5B (24 layers).
        logical_branches=8,
        # The raw native count is 232 suffix transitions + 11 registered Acts
        # (243).  Reserve the remaining bounded helper calls explicitly: two
        # independent prefills and nine restores, for 254 planned operations;
        # advertise the 256 hard ceiling to the scheduler.
        logical_operations=256,
        logical_checkpoint_replays=1,
        checkpoint_policy="one resident worker; one shared baseline, seven public Session candidate forks, native one-token suffix probes, exact restore, and Project export/reopen stay in process",
        extra={
            "native_probe_arms": 7,
            "native_probe_streams": 8,
            "native_margin_streams": 0,
            "shared_baseline_streams": 1,
            "continuation_tokens_per_stream": 1,
            "planned_native_transitions": 232,
            "planned_act_operations": 11,
            "planned_prefill_forward_calls": 2,
            "planned_restore_operations": 9,
            "planned_bounded_operations": 254,
            "raw_counted_operations": 243,
            "operation_budget": 256,
            "logical_count_basis": "Qwen2.5-0.5B-24-layer-one-token-native-suffix-plus-Acts",
        },
    )
    config = {
        "model": args.model,
        "prompt": "France Paris The capital of France is",
        "query": " Paris or Rome?",
        "backend": "hf",
        "device": "cuda",
        "dtype": "bf16",
        "seq_len": 32,
        "physical_batch_size": 1,
        "full_prefill_logits": False,
        "backend_options": {"physical_batch_size": 1, "full_prefill_logits": False},
        "task": "forward",
        "native_consumer": "adapter-declared-at-runtime",
        "controls": ["parent-copy", "matched-source-donor", "wrong-target-row-same-donor",
                     "single-row-zero", "coalition-zero", "coalition-half-scale"],
        "operation_budget": 256,
    }
    config_bytes = (json.dumps(config, sort_keys=True) + "\n").encode("utf-8")
    request = {
        "schema": "mdb-symbol-lifecycle-request-v1",
        "model": spec.to_dict(),
        "task_family": "native-symbol-lifecycle",
        "task": "forward",
        "needs": {"cuda": True},
        "source_hashes": source_hash_manifest({"worker.py": worker}),
        "scheduling_contract": contract,
        "config_bytes_sha256": sha256(config_bytes).hexdigest(),
        "execution_environment": intent(),
        "workload": {
            "prompt_tokens": "runtime-tokenized",
            "seq_len": 32,
            "physical_batch_size": 1,
            "full_prefill_logits": False,
            "native_consumer": "adapter-declared-at-runtime",
            "physical_model_loads": 1,
            "controls": config["controls"],
            "operation_budget": 256,
            "stop_condition": "one report marker after export/reopen and parent restore",
        },
    }
    request["request_sha256"] = sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mdb-symbol-lifecycle-") as directory:
        stage = Path(directory)
        stage_worker(stage, worker=worker, config=config, request=request, config_bytes=config_bytes)
        # This path is produced by the worker, not an unstaged input. Keep
        # strict closure checks on every input and declare just this output.
        (stage / "preflight.v1.json").write_text(json.dumps({
            "schema_version": "mrun-preflight-config-v1",
            "path_token_allowlist": ["symbol-lifecycle-report.json"],
        }) + "\n", encoding="utf-8")
        command = worker_command("worker.py", include_observability_extras=False) + [
            "--mode",
            "native",
            "--model",
            args.model,
            "--output",
            "symbol-lifecycle-report.json",
            "--root",
            "symbol-lifecycle-session",
            "--export-root",
            "symbol-lifecycle-export",
            "--project-root",
            "symbol-lifecycle-project",
        ]
        job_id = guarded_mrun_launch(
            command,
            request=request,
            experiment="mdb-symbol-lifecycle",
            task="forward",
            model=spec.registry_name,
            payload=stage,
            mrun_config={**request, **config, "model": spec.registry_name, "model_identity": spec.to_dict()},
            prefer_host=args.host,
            gpu=True,
            needs={"cuda": True},
            timeout_s=840,
            detach=True,
            retry_on_kill=False,
            preflight=True,
            note=(
                "Native Qwen symbol lifecycle: one BF16 resident model, one short prompt, "
                "one committed-cache row coalition, registered public Session memory Act, "
                "one shared baseline plus seven bounded one-token native suffix arms, "
                "Paris/Rome and Berlin readout margins, exact restore, Project export/reopen; "
                "stop after one report marker; no retries."
            ),
        )
    output = RESULTS / str(job_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"job_id": str(job_id), "request": request, "waited": False, "submission_output": str(output)}
    if args.wait:
        terminal_id, body, logs = terminal_job(str(job_id), wait=True, max_wait_s=900)
        result.update({"job_id": terminal_id, "state": body.get("state"), "result": body.get("result"), "waited": True})
        result["worker_report"] = decode_marker(logs, "SYMBOL_LIFECYCLE_REPORT_V1 ")
        result["log_tail"] = logs[-4000:]
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(submit(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

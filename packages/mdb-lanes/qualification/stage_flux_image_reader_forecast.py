"""Stage a sealed conditional image-reader forecast from authenticated saved cuts."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_image_reader_forecast.py"
HELPERS = ("flux_image_tail_microscope.py", "flux_reader_arithmetic.py",
           "flux_source_receiver.py", "flux_reentry_reference.py", "flux_family_custody.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def request_identity(request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()


def basis():
    return runpy.run_path(str(HERE / "stage_flux_image_tail_microscope.py"))


def command(request):
    from mdb_lanes.payload import worker_command
    basis()["bind"](request["execution_environment"])
    result = worker_command(WORKER)
    index = next(i for i in range(len(result)) if result[i:i + 4] ==
                 ["python", "-u", "-m", "mdb_lanes.worker_bootstrap"])
    result.insert(index + 1, "-B")
    environment = request["execution_environment"]
    for key, name in (("tmpdir", "TMPDIR"), ("cuda_cache_path", "CUDA_CACHE_PATH"),
                      ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR")):
        result.insert(1, name + "=" + environment[key])
    return result


def validate_payload(run, request):
    if request_identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("forecast request or launcher changed")
    if sha(HERE / "stage_flux_image_tail_microscope.py") != request["basis_launcher_sha256"]:
        raise ValueError("forecast basis launcher changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("forecast payload changed: " + name)


def stage(run, source_run):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run, source_run = Path(run).resolve(), Path(source_run).resolve()
    original, job = read(source_run / "request.json"), read(source_run / "job.json")
    basis()["validate_payload"](source_run, original)
    if (job["job_id"] != "job-3fb176d80f27" or job["state"] != "succeeded"
            or read(source_run / "submission.json")["job_id"] != job["job_id"]):
        raise ValueError("completed qualified image-tail source required")
    collector = HERE / "collect_flux_initial_noise.py"
    if sha(collector) != "e59cb6e3c1f4f0b03f0163f4db8544952def504090a3fdded8e61f530880b9ad":
        raise ValueError("qualified actual-job validator changed")
    runpy.run_path(str(collector))["validate_job"](job, job["job_id"], original)
    custody = job["payload_custody"]
    identities = [custody[stage_name]["sha256"] for stage_name in ("declared", "sealed", "executed")]
    if len(set(identities)) != 1:
        raise ValueError("source executed payload is not its sealed payload")
    basis()["bind"](original["execution_environment"])
    config = dict(original["config"])
    config.update(schema="mdb-image-reader-forecast-plan-v1", reference_head=False,
        helper_source_sha256={name: sha(HERE / name) for name in HELPERS},
        forecast_cases=["sham", "midpoint"],
        midpoint_recipe={"kind": "fp64-average-then-fp32-bf16-once", "dose": 0.5,
                         "source_register": "image", "carried_state": "all non-image fields from original sham"},
        fixed_head_control=True, transaction_max_bytes=512 << 20, transaction_array_storage="files-v1",
        observation_frontier="Own incoming image/residual/modulation and native positioned QKV; prefix aborts before SDPA.",
        forecast_expectation="Complete image-return value equality using original-policy native primitives; preserve all discrepancies if it misses.",
        source_microscope_request_sha256=original["request_sha256"],
        source_microscope_job=job["job_id"],
        source_microscope_job_sha256=sha(source_run / "job.json"),
        max_wall_seconds=300,
        selection="Predeclared numerical midpoint between two known pre-joint0 images plus known sham, with sham text/modulation/RoPE/clock.",
        scope="Prospective conditional composition of original-policy native primitives. Two aborted QKV prefixes before attention, separate forecast SDPA/tail calls, full-image cohort sealed before two complete native joint0 targets. No independent Flash arithmetic, new prompt/seed, full model, encoder, noise, suffix, VAE or object semantics.")
    run.mkdir(parents=True, exist_ok=False)
    copies = ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json")
    for name in copies:
        shutil.copyfile(source_run / "payload" / name, run / name)
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in HELPERS]
    extras.extend((run / name, name) for name in copies)
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1",
        "hashed_roots": [WORKER, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": [config["input_root"], config["parameter_root"],
                                 config["output_scratch"], config["output_name"]]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-image-reader-forecast-request-v1", "config": config,
        "execution_environment": original["execution_environment"], "scheduler_url": original["scheduler_url"],
        "source_microscope_run": str(source_run),
        "model": "flux2-klein-4b-distilled", "task_family": "analysis", "backend": "torch-component-native-primitives",
        "device": "cuda", "dtype": "bfloat16", "needs": {"cuda": True, "host": "beast"},
        "source_hashes": {name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
        "runtime_wheels": original["runtime_wheels"], "launcher_sha256": sha(__file__),
        "basis_launcher_sha256": sha(HERE / "stage_flux_image_tail_microscope.py"),
        "source_job_validator_sha256": sha(collector),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True,
            logical_operations=9, logical_branches=2, logical_checkpoint_replays=2, max_local_work_items=16,
            extra={"task_family": "analysis", "native_complete_targets": 2, "aborted_prefixes": 2},
            checkpoint_policy="One component-only CUDA lease; explicit input and forecast seals; two aborted pre-SDPA prefixes and two complete targets, no retries."),
        "resource_exception": {"ram_mb": 8192, "vram_mb": 3072, "cpu_threads": 4,
            "disk_gb": 2, "est_wall_s": 180, "timeout_s": 900,
            "calibration": "Same original16 joint0 BF16 tensors and full B1x1024 image/512 text geometry as job-3fb176d80f27: measured peakRSS4148.6MB/VRAM1574MB,101.87s scheduler/63.469s scientific. This cut removes independent FP64 heads/tails and local arithmetic panels, adds two aborted prefixes plus three native-primitive reader forecasts before two targets. Retain the measured8192MBRAM/3072MBVRAM headroom; no full model or persistent parameter bank."}}
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", {"command": command(request), "request_sha256": request["request_sha256"]})
    return {"request_sha256": request["request_sha256"], "submitted": False,
            "source_job": job["job_id"], "worker_sha256": request["source_hashes"][WORKER]}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("forecast already submitted")
    review = read(run / "finite-run-review-v1.json")
    if (review.get("request_sha256") != request["request_sha256"]
            or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("request-bound finite-run review required")
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"],
        "review_sha256": sha(run / "finite-run-review-v1.json")})
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-image-reader-forecast",
        task="analysis", model=None, payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True,
        note="Prospective conditional image reader: one predeclared BF16 midpoint plus sham at saved step1/joint0. Two aborted QKV prefixes STOP BEFORE SDPA; separately counted original-policy native primitive forecasts sealed before two complete native targets. One finite component CUDA lease; no fullmodel/encoder/noise/suffix/VAE, no retries. Raw operands remain near data; compact reports only collected.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and args.source_run is None:
        parser.error("--source-run is required for staging")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.source_run)))

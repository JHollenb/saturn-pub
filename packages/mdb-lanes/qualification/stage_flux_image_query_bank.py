"""Stage a fixed-recipient image-query by complete-KV-bank causal assay."""
from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_image_query_bank.py"
HELPERS = ("flux_image_reader_forecast.py", "flux_image_tail_microscope.py", "flux_reader_arithmetic.py",
           "flux_source_receiver.py", "flux_reentry_reference.py", "audit_flux_image_tail_microscope.py",
           "flux_family_custody.py")


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
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in (("tmpdir", "TMPDIR"), ("cuda_cache_path", "CUDA_CACHE_PATH"),
                      ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR")):
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def validate_payload(run, request):
    if request_identity(request) != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("query-bank request/launcher changed")
    if sha(HERE / "stage_flux_image_tail_microscope.py") != request["basis_launcher_sha256"]:
        raise ValueError("query-bank launcher basis changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("query-bank staged payload changed: " + name)


def stage(run, source_run, source_compact):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run, source_run, source_compact = (Path(path).resolve() for path in (run, source_run, source_compact))
    original, job = read(source_run / "request.json"), read(source_run / "job.json")
    reader_launcher = runpy.run_path(str(HERE / "stage_flux_image_reader_forecast.py"))
    reader_launcher["validate_payload"](source_run, original)
    collector = HERE / "collect_flux_initial_noise.py"
    if sha(collector) != "e59cb6e3c1f4f0b03f0163f4db8544952def504090a3fdded8e61f530880b9ad":
        raise ValueError("qualified actual-job validator changed")
    runpy.run_path(str(collector))["validate_job"](job, job["job_id"], original)
    if job["state"] != "succeeded" or job["job_id"] != "job-720536f238d0":
        raise ValueError("completed image-reader source required")
    audit = read(source_compact / "audit.json")
    manifest = read(source_compact / "files.json")
    for name, digest in manifest.items():
        if sha(source_compact / name) != digest:
            raise ValueError("source compact evidence changed: " + name)
    if (audit["job"] != job["job_id"] or audit["native_request_sha256"] != original["request_sha256"]
            or sha(source_compact / "source-files.json") != audit["native_manifest_sha256"]):
        raise ValueError("source compact/native lineage differs")
    basis()["bind"](original["execution_environment"])
    config = dict(original["config"])
    config.update(schema="mdb-image-query-bank-plan-v1", source_root=str(Path(config["output_scratch"]) / job["job_id"] / "binding-minimal-result"),
        source_manifest_sha256=audit["native_manifest_sha256"], source_request_sha256=original["request_sha256"],
        source_job_id=job["job_id"], source_forecast_seal_sha256=audit["forecast_seal_sha256"],
        source_input_seal_sha256=audit["input_seal_sha256"],
        helper_source_sha256={name: sha(HERE / name) for name in HELPERS},
        forecast_cases=["q0-b0", "q0-b1", "q1-b0", "q1-b1"],
        fixed_recipient="midpoint", transaction_max_bytes=512 << 20, transaction_array_storage="files-v1",
        predeclared_expectations={
            "numerical": "All four complete image-write forecasts and their signed contrasts should have exact value equality under original-policy native primitives; retain misses.",
            "text_separation": "The two source text-Q portions are unchanged; at each fixed bank, changing image Q alone should leave the current block text exit exactly unchanged.",
            "controls": "00 at midpoint residual should replay the earlier fixed-sham-head prediction;11 should replay the earlier midpoint native image. 00 is not a historical sham block.",
            "exploratory": "Relative query/bank effects, their sign, interaction and downstream roles are not assumed; preserve all signed values without a dominance gate."},
        max_wall_seconds=300,
        observation_frontier="Saved exact positioned Q0/Q1 and complete K0V0/K1V1; same original midpoint image residual/modulation for all four.",
        forecast_expectation="Four complete original-policy native-primitive image-return predictions sealed before four full native joint0 consumers with declared QKV substitutions; retain all numerical misses.",
        selection="Frozen full query and full key/value banks from prior sham/midpoint; source0=sham,source1=midpoint. Recipient non-attention state is midpoint throughout.",
        scope="Fixed-recipient image-query by whole-KV-bank causal square at observed positioned-QKV frontier. Four native primitive forecasts, then four full component consumers with dispatch-entry QKV substitutions. No independent kernel arithmetic, new prompt/seed, encoder, full model, suffix or VAE. Signed effects and interactions, no additive causal shares or semantic names.")
    run.mkdir(parents=True, exist_ok=False)
    copies = ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json")
    for name in copies:
        shutil.copyfile(source_run / "payload" / name, run / name)
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in HELPERS]
    extras.extend((run / name, name) for name in copies)
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKER, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": [config["source_root"], config["input_root"], config["parameter_root"], config["output_scratch"], config["output_name"]]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-image-query-bank-request-v1", "config": config,
        "execution_environment": original["execution_environment"], "scheduler_url": original["scheduler_url"],
        "model": "flux2-klein-4b-distilled", "task_family": "analysis", "backend": "torch-component-native-primitives",
        "device": "cuda", "dtype": "bfloat16", "needs": {"cuda": True, "host": "beast"},
        "source_hashes": {name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
        "runtime_wheels": original["runtime_wheels"], "launcher_sha256": sha(__file__),
        "basis_launcher_sha256": sha(HERE / "stage_flux_image_tail_microscope.py"),
        "source_job_validator_sha256": sha(collector), "source_reader_launcher_sha256": sha(HERE / "stage_flux_image_reader_forecast.py"),
        "source_job_record_sha256": sha(source_run / "job.json"), "source_compact_audit_sha256": sha(source_compact / "audit.json"),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=8,
            logical_branches=4, logical_checkpoint_replays=4, max_local_work_items=16,
            extra={"task_family": "analysis", "native_complete_targets": 4, "forecast_sdpa_calls": 4},
            checkpoint_policy="One component-only CUDA lease; four sealed query/bank forecasts then four full joint0 recipients with explicit QKV boundary substitutions, no suffix/retries."),
        "resource_exception": {"ram_mb": 8192, "vram_mb": 3072, "cpu_threads": 4, "disk_gb": 3,
            "est_wall_s": 180, "timeout_s": 900,
            "calibration": "Same original16 joint0 BF16 tensors/B1x1024image512text as job720536f238d0:46.2s elapsed,RSS3421.2MB,VRAM1558MB. Four forecasts/four full targets, no newprefix or FP64 equations; authenticated raw source567MB and files-v1 transaction, retain8192/3072MB safety envelope previously measured at RSS4148.6/VRAM1574MB. Geometry remains B1 sequential to preserve kernel contract, no speedup claim."}}
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", {"command": command(request), "request_sha256": request["request_sha256"]})
    return {"request_sha256": request["request_sha256"], "submitted": False, "source_job": job["job_id"]}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("query-bank run already submitted")
    review = read(run / "finite-run-review-v1.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("request-bound finite review required")
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"], "review_sha256": sha(run / "finite-run-review-v1.json")})
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-image-query-bank", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True,
        note="Fixed-recipient image Q0/Q1 by whole K0V0/K1V1 square:4native primitive forecasts sealed BEFORE4full joint0 targets. Each target uses same midpointimage/residual/modulation, actual fullblock recomputes QKV then declared dispatch replacement selects prior savedQ/K/V beforeoriginalSDPA. B1x1024image/512text sequential, original16weights, zero fullmodel/encode/noise/suffix/VAE,no retries. RawarraysBeast;compactauditonlycollected. Measures signedQ/bank effects/interaction, no semantic shares.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-compact", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and (args.source_run is None or args.source_compact is None):
        parser.error("source run and compact evidence are required")
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.source_run, args.source_compact)))

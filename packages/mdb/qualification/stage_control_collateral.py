"""Source-bound, finite Qwen/FLUX control-versus-collateral assays.

Staging executes definitions only. Scientific execution is admitted by mrun;
original numerical evidence and the previous campaign's workers stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKERS = {
    "qwen": "qwen_contrast_collateral_campaign.py",
    "flux": "flux_centered_write_campaign.py",
}
NOTES = {
    "qwen": "Separate contrast control from collateral: 4 fresh Qwen CPU FP32 prefixes, 36 signed full-vocabulary readouts, native private-cache futures. Source-only two-write decomposition; all forecasts sealed before targets. One hydration, B1, original parameter recipes, no weight bank. Intent-first recorder sizing; 900s worker/1200s lease, no retries or nesting.",
    "flux": "Matched centered writes: 2 fresh FLUX roots, 12 image futures, fixed old spectral vectors, signed output Acts on each lane's own state. 1322 transitions/lane, 26 decodes; global pretarget seal, B1/512sq/4steps. One pipeline, 3 hydrations, max1 GPU component. Prior20 futures took550s; expect6-8min. RAM24GiB/VRAM13000MB/CPU4; 1800s worker/2100s lease, no retries/nesting.",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False,
    ).encode()).hexdigest()


def bind(environment):
    from mdb.execution_environment import intent

    names = {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
             "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
             "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}
    for key, name in names.items():
        os.environ[name] = environment[key]
    actual = intent()
    if actual != {key: environment[key] for key in actual}:
        raise ValueError("Installed worker environment differs from qualified runtime")


def command(request):
    from mdb.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKERS[request["panel"]])
    args.insert(args.index("python") + 1, "-B")
    args.extend(["--config", "saturn_run_config.json"])
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
                      "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def validate(run, request):
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Staged request or launcher changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if Path(name).name != name or sha(run / "payload" / name) != expected:
            raise ValueError("Staged source/wheel differs: " + name)


def stage(panel, run, runtime_parent, qwen_parent, design, extra_sources=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    run, runtime_parent, qwen_parent, design = (
        Path(path).resolve() for path in (run, runtime_parent, qwen_parent, design)
    )
    parent = read(runtime_parent / "request.json")
    if identity(parent) != parent.get("request_sha256"):
        raise ValueError("Qualified runtime parent request changed")
    design_data = read(design)
    if design_data["schema"] != "mdb-control-collateral-design-v1":
        raise ValueError("Unsupported control/collateral design")
    worker_path = HERE / WORKERS[panel]
    worker = runpy.run_path(str(worker_path))
    base = parent["config"] if panel == "flux" else read(qwen_parent)["config"]
    if panel == "qwen" and sha(qwen_parent) != "57594cbb5739ede8e1a1904f7ec0ae30d19cab986be674f4fd7a3ea6315c809b":
        raise ValueError("Original Qwen parent metadata changed")
    config = worker["make_config"](base)
    run.mkdir(parents=True, exist_ok=False)
    sources = {WORKERS[panel]: worker_path}
    for name, expected in worker["HELPERS"].items():
        if Path(name).name != name or sha(HERE / name) != expected:
            raise ValueError("Worker helper is not frozen: " + name)
        sources[name] = HERE / name
    for name, path in (extra_sources or {}).items():
        if Path(name).name != name or name in sources:
            raise ValueError("Extra source name unsafe or duplicated: " + name)
        sources[name] = Path(path).resolve()
    selection = dict(schema="mdb-control-collateral-panel-v1", panel=panel,
                     campaign_design_sha256=sha(design), design=design_data[panel],
                     contexts=worker.get("CONTEXTS", worker.get("PROMPTS")))
    save(run / "selection-protocol.json", selection)
    sources["selection-protocol.json"] = run / "selection-protocol.json"
    if panel == "flux":
        path_keys = worker["path_keys"]()
        config["selection_protocol"] = "selection-protocol.json"
        for key in path_keys:
            name = config[key]
            if Path(name).name != name:
                raise ValueError("Payload sources must have flat names: " + name)
            if name not in sources:
                source = runtime_parent / "payload" / name
                if sha(source) != parent["source_hashes"][name]:
                    raise ValueError("Original source metadata changed: " + name)
                sources[name] = source
            config[key + "_sha256"] = sha(sources[name])
    else:
        path_keys = ()
        retriever = runtime_parent / "payload" / "retrieve_flux_weights.py"
        if sha(retriever) != worker["RETRIEVER_SHA256"]:
            raise ValueError("Exact standalone parameter retriever differs")
        sources["retrieve_weights.py"] = retriever
    local = dict(config)
    for key in path_keys:
        local[key] = str(sources[config[key]])
    counts = worker["validate_config"](local)
    save(run / "plan.json", config)
    save(run / "count-contract.json", counts)
    shutil.copyfile(design, run / "campaign-design-v1.json")
    for name in ("plan.json", "count-contract.json", "campaign-design-v1.json"):
        sources[name] = run / name
    paths = {value for key, value in config.items() if isinstance(value, str)
             and (key.endswith(("_root", "_path")) or key in ("output_scratch", "output_name"))}
    preflight = dict(schema="mrun-preflight-config-v1", hashed_roots=list(sources),
                     path_token_allowlist=sorted(paths),
                     no_native_targets_before_global_forecast_seal=True,
                     no_persistent_parameter_payload_exports=True)
    save(run / "preflight.v1.json", preflight)
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    shared = dict(schema="mdb-control-collateral-request-v1", panel=panel, config=config,
                  execution_environment=parent["execution_environment"], scheduler_url=parent["scheduler_url"],
                  task_family="recorder" if panel == "qwen" else "analysis", runtime_wheels=parent["runtime_wheels"],
                  source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
                  campaign_design_sha256=sha(design), runtime_parent_request_sha256=parent["request_sha256"])
    if panel == "flux":
        shared.update(model="flux2-klein-4b-distilled", backend="torch-native-centered-writes", device="cuda", dtype="bfloat16",
            needs={"cuda": True, "host": "beast"}, flux_mode="capture", capture_sites=True, height=512, width=512, steps=4,
            workload_geometry={**counts, "peak_physical_batch": 1, "image_tokens": 1024, "text_tokens": 512,
                               "output_bytes_bound": config["transport_limits"]["max_expanded_bytes"]},
            resource_exception=dict(ram_mb=24576, vram_mb=13000, cpu_threads=4, disk_gb=48,
                est_wall_s=480, timeout_s=2100,
                calibration="Measured previous source-bound job-e880bd908d91:20 futures,2208F+2208N transitions,44 decodes,549.73s,peakRSS20232MB/VRAM8672MB. New panel12 futures,1322 transitions/lane,26 decodes and fixed saved spectral vectors. Expect6-8min with original once-hydrated CPU components, sequential B1 CUDA,max1 GPU component;1800s worker/2100s outer are finite ceilings. Same24GiB RAM/13000MB VRAM reservation protects measured geometry."))
        branches = counts["image_futures"]
        replays = branches
        operations = 82
    else:
        shared.update(model="qwen2.5-0.5b-instruct", backend="hf", device="cpu", dtype="float32", needs={"cpu": True},
            capture_sites=True, seq_len=config["max_prompt_tokens"], batch_size=1,
            backend_options=dict(physical_batch_size=1, full_prefill_logits=True),
            workload_geometry={**counts, "peak_physical_batch": 1, "full_vocabulary_rows": 151936,
                               "output_bytes_bound": config["transport_limits"]["max_expanded_bytes"]},
            resource_basis="Intent-first CPU recorder plan. Prior same checkpoint job-956049ba0eeb peakRSS6917MB/science59s/elapsed98.44s,100readouts,128cachedsteps. New36readouts,160cachedsteps,4prefixes; same original parameter recipes and bounded FP64 head chunks. No persistent bank; actual4CPUthreads.",
            timeout_s=1200)
        branches = counts.get("component_readout_targets", 36)
        replays = counts.get("ordinary_prefill_adjudications", 20)
        operations = counts.get("native_physical_entries_max", 244)
    shared["scheduling_contract"] = one_lease_scheduling_contract(
        model_loads=1, logical_branches=branches, logical_operations=operations,
        logical_checkpoint_replays=replays, max_local_work_items=256,
        extra={"task_family": shared["task_family"], "count_contract": counts,
               "component_operation_unit": "component passes and complete future branches; primitive counts separate"},
        checkpoint_policy="One finite lease; shared immutable parents; global pretarget seal; original native consumers; no nested submissions or retries.")
    shared["request_sha256"] = identity(shared)
    save(run / "request.json", shared)
    bind(shared["execution_environment"])
    stage_worker(run / "payload", worker=worker_path, config=config, request=shared,
                 additional_files=[(path, name) for name, path in sources.items() if name != WORKERS[panel]])
    validate(run, shared)
    save(run / "staged-command.json", dict(command=command(shared), request_sha256=shared["request_sha256"], submitted=False))
    return dict(request_sha256=shared["request_sha256"], counts=counts, source_files=len(sources), submitted=False)


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    request = read(run / "request.json")
    validate(run, request)
    if (run / "submission.json").exists():
        raise ValueError("Panel has already been submitted")
    review = read(run / "finite-run-review-v1.json")
    if (review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != []
            or review.get("review_status") != "ready-for-finite-run"):
        raise ValueError("Request-bound finite-run review required")
    panel = request["panel"]
    note = NOTES[panel]
    if len(note) > 500:
        raise ValueError("Queue note exceeds500-character ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    save(run / "admission-command.json", dict(command=args, review_sha256=sha(run / "finite-run-review-v1.json"), request_sha256=request["request_sha256"]))
    options = dict(prefer_host="beast", gpu=False, timeout_s=request["timeout_s"]) if panel == "qwen" else dict(
        pin="beast", gpu=True, **{key: request["resource_exception"][key] for key in
            ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")})
    job = guarded_mrun_launch(args, request=request, experiment="mdb-control-collateral-" + panel,
        task=request["task_family"], model=request["model"] if panel == "qwen" else None,
        payload=run / "payload", needs=request["needs"], retry_on_kill=False,
        preflight=True, note=note, **options)
    save(run / "submission.json", dict(job_id=str(job), request_sha256=request["request_sha256"]))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=WORKERS)
    parser.add_argument("run", type=Path)
    parser.add_argument("--runtime-parent", type=Path)
    parser.add_argument("--qwen-parent", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and not all((args.runtime_parent, args.qwen_parent, args.design)):
        parser.error("stage requires --runtime-parent, --qwen-parent and --design")
    sources = {}
    for item in args.source:
        name, path = item.split("=", 1)
        if name in sources:
            parser.error("duplicate source name")
        sources[name] = path
    print(json.dumps(submit(args.run) if args.submit else stage(
        args.panel, args.run, args.runtime_parent, args.qwen_parent, args.design, sources,
    ), sort_keys=True))

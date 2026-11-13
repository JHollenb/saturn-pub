"""Stage and admit one source-bound panel of the Astra computation campaign.

Staging imports definitions only. Execution is one guarded finite mrun lease;
the CPU and CUDA panels retain their own numerical and consumer contracts.
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
WORKERS = {"qwen": "qwen_spectral_readout_campaign.py", "flux": "flux_gauge_spectral_campaign.py"}
NOTES = {
    "qwen": "Astra law transfer:4fresh QwenCPUFP32 contexts,100two-write full-vocab cells,24native MLP probes,20ordinary adjudication/shams,<=128own-token cached successors.4prefixes shared;all forecasts sealed beforetargets.1hydrate,B1,originalparameter recipes,no weightbank. Intent-first CPU recorder sizing;900s worker/1200s outer,no retries/nesting.",
    "flux": "Astra laws:2wordingsx2seeds,20FLUX image futures;LNoffset/null and source-spectral additive writers.2208F+2208N transitions,24FFprobes/lane,20F+24N VAE;all forecasts sealed beforetargets.1pipeline/3hydrates/5phaseplacements,max1GPU,B1/512sq/4steps. RAM24GiB/VRAM13000MB/CPU4,10800s science/12600s outer;no retries/nesting.Compact collection only.",
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def bind(environment):
    from mdb_lanes.execution_environment import intent

    names = {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
             "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
             "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}
    for key, name in names.items():
        os.environ[name] = environment[key]
    actual = intent()
    if actual != {key: environment[key] for key in actual}:
        raise ValueError("Installed worker environment differs from qualified runtime")


def command(request):
    from mdb_lanes.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKERS[request["panel"]])
    args.insert(args.index("python") + 1, "-B")
    if request["panel"] == "qwen":
        args.extend(["--config", "saturn_run_config.json"])
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH",
                      "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return args


def validate(run, request):
    if identity(request) != request.get("request_sha256") or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("Staged request or launcher changed")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("Staged source/wheel differs: " + name)


def stage(panel, run, runtime_parent, qwen_parent, design):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, runtime_parent, qwen_parent, design = (Path(path).resolve() for path in (run, runtime_parent, qwen_parent, design))
    parent = read(runtime_parent / "request.json")
    if identity(parent) != parent.get("request_sha256"):
        raise ValueError("Qualified runtime parent request changed")
    design_data = read(design)
    if design_data["schema"] != "mdb-astra-generalization-design-v1":
        raise ValueError("Unsupported campaign design")
    worker_path = HERE / WORKERS[panel]
    worker = runpy.run_path(str(worker_path))
    base = parent["config"] if panel == "flux" else read(qwen_parent)["config"]
    if panel == "qwen" and sha(qwen_parent) != "57594cbb5739ede8e1a1904f7ec0ae30d19cab986be674f4fd7a3ea6315c809b":
        raise ValueError("Original Qwen parent metadata changed")
    config = worker["make_config"](base)
    run.mkdir(parents=True, exist_ok=False)
    sources = {WORKERS[panel]: worker_path}
    for name, expected in worker["HELPERS"].items():
        if sha(HERE / name) != expected:
            raise ValueError("Worker helper is not frozen: " + name)
        sources[name] = HERE / name
    selection = dict(schema="mdb-astra-panel-selection-v1", campaign_design_sha256=sha(design),
                     contexts=worker.get("CONTEXTS", worker.get("PROMPTS")), panel=panel,
                     design=design_data[panel])
    save(run / "selection-protocol.json", selection)
    sources["selection-protocol.json"] = run / "selection-protocol.json"
    if panel == "flux":
        path_keys = worker["path_keys"]()
        config["selection_protocol"] = "selection-protocol.json"
        for key in path_keys:
            name = config[key]
            if name in sources:
                continue
            source = runtime_parent / "payload" / name
            if sha(source) != parent["source_hashes"][name]:
                raise ValueError("Original source metadata changed: " + name)
            shutil.copyfile(source, run / name)
            sources[name] = run / name
        for key in path_keys:
            config[key + "_sha256"] = sha(sources[config[key]])
    else:
        path_keys = ()
        retriever = runtime_parent / "payload" / "retrieve_flux_weights.py"
        if sha(retriever) != worker["RETRIEVER_SHA256"]:
            raise ValueError("Exact standalone parameter retriever differs")
        shutil.copyfile(retriever, run / "retrieve_weights.py")
        sources["retrieve_weights.py"] = run / "retrieve_weights.py"
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
                     path_token_allowlist=sorted(paths), no_native_targets_before_global_forecast_seal=True,
                     no_persistent_parameter_payload_exports=True)
    save(run / "preflight.v1.json", preflight)
    sources["preflight.v1.json"] = run / "preflight.v1.json"
    shared = dict(schema="mdb-astra-generalization-request-v1", panel=panel, config=config,
                  execution_environment=parent["execution_environment"], scheduler_url=parent["scheduler_url"],
                  task_family="recorder" if panel == "qwen" else "analysis", runtime_wheels=parent["runtime_wheels"],
                  source_hashes={name: sha(path) for name, path in sources.items()}, launcher_sha256=sha(__file__),
                  campaign_design_sha256=sha(design), runtime_parent_request_sha256=parent["request_sha256"])
    if panel == "flux":
        shared.update(model="flux2-klein-4b-distilled", backend="torch-native-gauge-spectral", device="cuda", dtype="bfloat16",
            needs={"cuda": True, "host": "beast"}, flux_mode="capture", capture_sites=True, height=512, width=512, steps=4,
            workload_geometry={**counts, "peak_physical_batch": 1, "image_tokens": 1024, "text_tokens": 512,
                               "output_bytes_bound": config["transport_limits"]["max_expanded_bytes"]},
            resource_exception=dict(ram_mb=24576, vram_mb=13000, cpu_threads=4, disk_gb=48, est_wall_s=9000, timeout_s=12600,
                calibration="Prior original-profile raw-to-image job-7efe0a8100bb RSS20815.5MB/VRAM8374MB/329.23s and paired job-0b6c4340d5e6 RSS20718.1MB/VRAM8374MB/411.95s. Same once-hydrated CPU components, B1 sequential CUDA and maximum1 GPU component. Extra matrix-free FP64 spectral factors transient, selected-state/FF retention streamed.4416 transitions scale the parent transition count by19.7x;9000s estimate and10800s worker cap are prospective finite envelopes, not measured throughput.24GiB RAM guard ceiling fits66.5GB physical after margins; preserve resource misses."))
        operations = 174
    else:
        shared.update(model="qwen2.5-0.5b-instruct", backend="hf", device="cpu", dtype="float32", needs={"cpu": True},
            capture_sites=True, seq_len=config["max_prompt_tokens"], batch_size=1,
            backend_options=dict(physical_batch_size=1, full_prefill_logits=True),
            workload_geometry={**counts, "peak_physical_batch": 1, "full_vocabulary_rows": 151936,
                               "output_bytes_bound": config["transport_limits"]["max_expanded_bytes"]},
            resource_basis="Intent-first source FP32 CPU recorder sizing. Prior same checkpoint measured6176.2MB RSS, recorder plan7825.7MB. No persistent parameter bank; one readonly full-head view, bounded FP64 chunks, streaming100cell logits. Native actualthreads4; record resolved reservation and peaks.",
            timeout_s=1200)
        operations = 168
    shared["scheduling_contract"] = one_lease_scheduling_contract(model_loads=1, logical_branches=20 if panel == "flux" else 100,
        logical_operations=operations, logical_checkpoint_replays=20 if panel == "flux" else 16, max_local_work_items=256,
        extra={"task_family": shared["task_family"], "count_contract": counts, "component_operation_unit": "whole component passes and whole future branches; physical primitive/transition counts separate"},
        checkpoint_policy="One finite panel lease, shared immutable parents, pretarget global seal, original native consumers, no nested submissions or retries; CPU/CUDA panels remain separately qualified.")
    shared["request_sha256"] = identity(shared)
    save(run / "request.json", shared)
    bind(shared["execution_environment"])
    stage_worker(run / "payload", worker=worker_path, config=config, request=shared,
                 additional_files=[(path, name) for name, path in sources.items() if name != WORKERS[panel]])
    validate(run, shared)
    save(run / "staged-command.json", dict(command=command(shared), request_sha256=shared["request_sha256"], submitted=False))
    return dict(request_sha256=shared["request_sha256"], counts=counts, source_files=len(sources), submitted=False)


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

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
    job = guarded_mrun_launch(args, request=request, experiment="mdb-astra-generalization-" + panel,
        task=request["task_family"], model=request["model"] if panel == "qwen" else None, payload=run / "payload", needs=request["needs"],
        retry_on_kill=False, preflight=True, note=note, **options)
    save(run / "submission.json", dict(job_id=str(job), request_sha256=request["request_sha256"]))
    return dict(job_id=str(job))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=WORKERS)
    parser.add_argument("run", type=Path)
    parser.add_argument("--runtime-parent", type=Path)
    parser.add_argument("--qwen-parent", type=Path)
    parser.add_argument("--design", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if not args.submit and not all((args.runtime_parent, args.qwen_parent, args.design)):
        parser.error("stage requires --runtime-parent, --qwen-parent and --design")
    print(json.dumps(submit(args.run) if args.submit else stage(args.panel, args.run, args.runtime_parent, args.qwen_parent, args.design), sort_keys=True))

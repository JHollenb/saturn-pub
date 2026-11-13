"""Metadata-only cached-QKV staging; root alone executes stage or submission.

Inherited source arrays and checkpoint payloads are never read by this frontend.
The measured component reservation is a planning exception, not a new peak claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRANSPORT = "stage_qwen_row_sign.py"
TRANSPORT_SHA256 = "8236875035d80245bcd8f3a6ddcbc6ce6b7e79cefd94800acf6c5dc8a3eedb40"
WORKER = "qwen_cached_qkv_campaign.py"
WORKER_SHA256 = "5905d5ce9d91870aebdb2bac4891a0a60e57979dbfeb44f189cf48b6edad5d3e"
PROTOCOL_SHA256 = "8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff"
SUBSET_SHA256 = "425502121b4091b10124c548c98127785420233b12e95c8f1d972a186026dccf"
PARENT_FORECAST_SHA256 = "0e716c88c8bca36ebd38e92963cce78fd3fc5dc798ab6da46623c8dd80499cda"
RESOURCE_JOB_SHA256 = "3a67fc9408ca1cf338e3b96593a8ac4f64aed32df2b5afdcbbb5fabcef959888"
RESOURCE = dict(cpu_threads=4, ram_mb=4096, vram_mb=2048, disk_gb=3, est_wall_s=60, timeout_s=300)
QUEUE_NOTE = ("Conditional cached-QKV block18,B1/512:2 cached SDPA/tails+2 full functional edited blocks sealed before3 native "
    "component targets. One11-weight/201.9MB hydration;no full encoder/LM/tokenizer/Flux/VAE/noise. Native74 measured47.22s/"
    "RSS1795MB/VRAM826MB;extra2SDPA+6linears unmeasured. CPU4/RAM4GiB/VRAM2GiB,disk3GB;60s estimate/240s assay/300s outer;"
    "one CUDA Beast lease,strict preflight,no retries/nesting;misses retained.")


@lru_cache(maxsize=1)
def transport():
    path = HERE / TRANSPORT
    if hashlib.sha256(path.read_bytes()).hexdigest() != TRANSPORT_SHA256:
        raise ValueError("Frozen staging transport changed")
    return runpy.run_path(str(path))


def utilities():
    return transport()["utilities"]()


def producer():
    h = utilities()
    h["require"](h["sha"](HERE / WORKER) == WORKER_SHA256, "Qualified cached-QKV adapter changed")
    worker = runpy.run_path(str(HERE / WORKER))
    for name, digest in worker["HELPERS"].items():
        h["require"](Path(name).name == name and h["sha"](HERE / name) == digest, "Qualified component/predictor/helper changed: " + name)
    return worker


def source_descriptors(protocol, wrapper, seal, positive):
    """Join inherited metadata without loading arrays or changing capture strides."""
    require = utilities()["require"]
    declarations = seal["declarations"]["frontier"]["external_files"]
    for name, row in protocol["source_arrays"].items():
        expected = dict(bytes=row["bytes"], sha256=row["sha256"])
        require(wrapper["files"].get(row["path"]) == declarations.get(row["path"]) == expected, "Inherited source not joined to raw/global seal")
        if name in positive:
            require(row == positive[name], "Original positive descriptor/layout differs: " + name)


def metadata(parent_run, compact, selection, frontier, subset):
    from mdb_lanes.parameter_artifacts import validate_manifest

    h, t = utilities(), transport()
    read, sha, require = h["read"], h["sha"], h["require"]
    parent_run, compact, selection, frontier, subset = map(Path, (parent_run, compact, selection, frontier, subset))
    parent = read(parent_run / "request.json")
    require(h["identity"](parent) == parent.get("request_sha256") == t["PARENT_REQUEST"], "Exact turn13 parent request required")
    require(sha(compact / "files.json") == t["COMPACT_SHA256"], "Accepted parent compact manifest differs")
    for name, digest in read(compact / "files.json").items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(compact / name) == digest, "Parent compact metadata changed: " + name)
    require(read(compact / "source-request.json") == parent, "Parent request copies differ")
    require(sha(selection / "protocol.json") == PROTOCOL_SHA256, "New cached-QKV protocol differs")
    protocol = read(selection / "protocol.json")
    wrapper = read(compact / "source-files.json")
    require(sha(compact / "source-files.json") == protocol["parent_raw_manifest_sha256"] and wrapper["job_id"] == protocol["parent_job"]
        and wrapper["request_sha256"] == parent["request_sha256"] == protocol["parent_request_sha256"], "Parent raw source identity differs")
    case = protocol["source_case"]
    require(Path(case).name == case, "Unsafe source case")
    forecast = frontier / "cases" / case / "forecast.json"
    require(sha(forecast) == PARENT_FORECAST_SHA256, "Authenticated parent forecast metadata differs")
    positive = read(forecast)["ports"]["plus"]
    seal = frontier / "transaction/seal.json"
    require(sha(seal) == protocol["parent_global_seal_sha256"] == read(compact / "source-summary.json")["forecast_seal"], "Original pretarget seal differs")
    source_descriptors(protocol, wrapper, read(seal), positive)
    full_path = parent_run / "payload/qwen-parameter-artifacts.json"
    require(sha(subset) == SUBSET_SHA256 and sha(full_path) == t["PARENT_MANIFEST_SHA256"], "Fixed original recipe identities differ")
    full, selected = validate_manifest(read(full_path)), validate_manifest(read(subset))
    original = [row for row in full["artifacts"] if row["source"]["tensor"].startswith("model.layers.18.")]
    require(len(full["artifacts"]) == 398 and len(original) == 11 and selected["artifacts"] == original, "Original11 recipe entries differ")
    paths = dict(qwen_config=parent_run / "payload/qwen-config.json", qwen_parameter_manifest=subset,
        parent_parameter_manifest=full_path, qwen_source_binding=parent_run / "payload/qwen-source-binding.json",
        retrieval_script=parent_run / "payload/retrieve_qwen_weights.py", selection_protocol=selection / "protocol.json",
        parent_files=compact / "source-files.json", parent_request=parent_run / "request.json", parent_source=compact / "source-source.json",
        parent_runtime=compact / "source-runtime.json", parent_seal=seal, parent_forecast=forecast)
    for name in ("qwen-config.json", "qwen-source-binding.json", "retrieve_qwen_weights.py", "qwen-parameter-artifacts.json"):
        require(sha(parent_run / "payload" / name) == parent["source_hashes"][name], "Original metadata differs: " + name)
    return parent, protocol, paths


def measured_component(path, parent):
    h = utilities()
    h["require"](h["sha"](path) == RESOURCE_JOB_SHA256, "Actual measured component job record changed")
    job = h["read"](path)
    h["require"](job["job_id"] == "job-74efe5506ecf" and job["state"] == "succeeded" and job["result"]["returncode"] == 0,
        "Successful measured component required")
    h["require"](job["config"]["runtime_wheels"] == parent["runtime_wheels"] and job["config"]["execution_environment"] == parent["execution_environment"], "Measured component source environment differs")
    return dict(job_id=job["job_id"], job_record_sha256=RESOURCE_JOB_SHA256,
        **{key: job["result"][key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")},
        scope="Same original11-weight hydration,2full functional and3native blocks. New candidate adds2SDPA+6linears; new peak/time unmeasured. Whole-shard hash startup is distinct from201861632B selected payload.")


def command(request):
    from mdb_lanes.payload import worker_command

    utilities()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH", "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return [*args, "--config", "saturn_run_config.json"]


def stage(run, parent_run, compact, selection, frontier, subset, resource_job):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    h, worker = utilities(), producer()
    parent, protocol, paths = metadata(parent_run, compact, selection, frontier, subset)
    measurement = measured_component(resource_job, parent)
    run = Path(run).resolve(); run.mkdir(parents=True, exist_ok=False)
    names = {key: key.replace("_", "-") + Path(path).suffix for key, path in paths.items()}
    for key, path in paths.items():
        shutil.copyfile(path, run / names[key])
    base = {key: names[key] for key in paths}
    base.update({key + "_sha256": h["sha"](path) for key, path in paths.items()})
    base.update(parameter_root=parent["config"]["parameter_root"], input_root=protocol["parent_raw_root"])
    config = worker["make_config"](base)
    counts = worker["validate_config"]({**config, **{key: str(run / names[key]) for key in paths}})
    h["require"](worker["PHYSICAL_PLAN"] == protocol["physical_plan"] and set(protocol["source_arrays"]) == worker["SOURCE_KEYS"], "New protocol/source contract differs")
    for name, value in (("count-contract.json", counts), ("resource-basis.json", measurement)):
        h["save"](run / name, value)
    row = transport()
    pairs = {WORKER: HERE / WORKER, Path(__file__).name: Path(__file__), TRANSPORT: HERE / TRANSPORT,
        row["UTILITIES"]: HERE / row["UTILITIES"], **{name: HERE / name for name in worker["HELPERS"]},
        **{names[key]: run / names[key] for key in paths}, **{name: run / name for name in ("count-contract.json", "resource-basis.json")}}
    h["save"](run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=[*pairs, "preflight.v1.json"],
        path_token_allowlist=[config[key] for key in ("parameter_root", "input_root", "output_scratch", "output_name")]))
    pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-qwen-cached-qkv-request-v1", config=config, execution_environment=parent["execution_environment"],
        scheduler_url=parent["scheduler_url"], runtime_wheels=parent["runtime_wheels"], model="qwen3-block18-in-flux2-klein-4b", task_family="analysis",
        backend="torch-native-qwen-cached-qkv", device="cuda", dtype="bfloat16", needs={"cuda": True, "host": "beast"},
        source_hashes={name: h["sha"](path) for name, path in pairs.items()}, launcher_sha256=h["sha"](__file__),
        resource_exception={**RESOURCE, "measurement": measurement}, workload_geometry=protocol["physical_plan"],
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=3, logical_operations=7,
            logical_checkpoint_replays=0, max_local_work_items=7, extra=protocol["physical_plan"],
            checkpoint_policy="One finite B1 component lease;2cached+2full functional predictions globally seal before3native targets; no retry/nesting/fullmodel/LM/tokenizer/Flux/VAE/noise."))
    request["request_sha256"] = h["identity"](request)
    h["save"](run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
        additional_files=[(path, name) for name, path in pairs.items() if name != WORKER])
    inspect_run(run)
    h["save"](run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"], submitted=False))
    return dict(request_sha256=request["request_sha256"], source_files=len(pairs), recipes=11, activation_bytes_copied=0, submitted=False)


def inspect_run(run):
    h, run = utilities(), Path(run)
    request = h["read"](run / "request.json")
    h["require"](h["identity"](request) == request.get("request_sha256") and request["launcher_sha256"] == h["sha"](__file__), "Request/frontend identity differs")
    h["require"](request["source_hashes"].get(WORKER) == WORKER_SHA256 and all(request["resource_exception"].get(k) == v for k,v in RESOURCE.items()), "Producer/resource contract differs")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        h["require"](h["sha"](run / "payload" / name) == digest, "Payload source/wheel changed: " + name)
    seal = h["read"](run / "payload/payload-seal.json")
    h["require"](h["read"](run / "payload/saturn_debug_request.json") == request and h["read"](run / "payload/saturn_run_config.json") == request["config"], "Payload request/config differs")
    h["require"](seal["files"] == request["source_hashes"] and seal["wheels"] == request["runtime_wheels"] and seal["worker"] == WORKER
        and seal["worker_sha256"] == WORKER_SHA256 and seal["config_bytes_sha256"] == h["sha"](run / "payload/saturn_run_config.json"), "Payload seal differs")
    return request


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    h, run = utilities(), Path(run)
    request = inspect_run(run)
    h["require"](not (run / "submission.json").exists(), "Submission already exists")
    review = h["read"](run / "finite-run-review-v1.json")
    h["require"](review.get("request_sha256") == request["request_sha256"] and review.get("blocking_issues") == [] and review.get("review_status") == "ready-for-finite-run", "Exact finite review required")
    h["require"](len(QUEUE_NOTE) <= 500, "Queue note exceeds scheduler limit")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    h["save"](run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=h["sha"](run / "finite-run-review-v1.json")))
    job = guarded_mrun_launch(args, request=request, experiment="mdb-qwen-cached-qkv", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True, **RESOURCE, retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    h["save"](run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run", type=Path)
    for name in ("parent-run", "compact", "selection", "frontier", "parameter-subset", "resource-job"):
        parser.add_argument("--" + name, type=Path)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument("--inspect", action="store_true"); mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.inspect or args.submit:
        result = submit(args.run) if args.submit else inspect_run(args.run)
    else:
        fields = ("parent_run", "compact", "selection", "frontier", "parameter_subset", "resource_job")
        if any(getattr(args, name) is None for name in fields):
            parser.error("all source metadata and --resource-job arguments are required to stage")
        result = stage(args.run, *(getattr(args, name) for name in fields))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Metadata-only block19 two-world staging; root alone stages/submits originals.

The default launch uses intent-first sizing. An explicit measured-component
reservation exception is available when planning lacks this component geometry.
Inherited arrays and checkpoint payloads stay at their two authenticated roots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
UTILITIES = "stage_qwen_sign_shortcut.py"
UTILITIES_SHA256 = "6f083cc6a039d50f8f206731d4a0fc87e76f9bf464c0d8cedba4caaecf04887b"
WORKER = "qwen_error_consumer_campaign.py"
WORKER_SHA256 = "ad82b86dd837a1f2bc9134f2c5bda3dcb097e8f8583e7849fcfbe8d5c49b229b"
PROTOCOL_SHA256 = "4e62bc3fa8adcd059ac12f775effa0010752e3c5e0cca2a2ebc4702ecef7f847"
SUBSET_SHA256 = "69a9189b00413ccb01943718a68eba2865181110aa30b04f17668a8e2e1ffb8c"
RESOURCE_JOB_SHA256 = "3a67fc9408ca1cf338e3b96593a8ac4f64aed32df2b5afdcbbb5fabcef959888"
RESOURCE = dict(cpu_threads=4, ram_mb=4096, vram_mb=2048, disk_gb=3, est_wall_s=60)
OUTER_TIMEOUT = 300
QUEUE_NOTE = ("Block19 explicit H19 worlds,B1/512:2 full functional forecasts globally sealed before2 original native "
    "component targets;one11-recipe/201.9MB hydration. Seven inherited arrays/two Beast roots;no extra baseline/fullmodel/"
    "LM/tokenizer/Flux/VAE/noise. Intent-first sizing;optional documented component exception. 240s assay/300s outer;"
    "one CUDA lease,strict preflight,no retries/nesting;all numerical misses retained.")


@lru_cache(maxsize=1)
def utilities():
    path = HERE / UTILITIES
    if hashlib.sha256(path.read_bytes()).hexdigest() != UTILITIES_SHA256:
        raise ValueError("Frozen staging utilities changed")
    return runpy.run_path(str(path))


def producer():
    h = utilities()
    h["require"](h["sha"](HERE / WORKER) == WORKER_SHA256, "Qualified explicit-world producer changed")
    worker = runpy.run_path(str(HERE / WORKER))
    for name, digest in worker["HELPERS"].items():
        h["require"](Path(name).name == name and h["sha"](HERE / name) == digest, "Qualified producer helper changed: " + name)
    return worker


def source_descriptors(protocol, world_files, world_seal, context_files, context_seal, forecast, shortcut):
    """Authenticate seven array references; never open their NPY payloads."""
    require = utilities()["require"]
    expected_worlds = {"full-input": forecast["arms"]["prefix-negative"]["ports"]["Hnext"],
        "cheap-input": shortcut["arms"]["prefix-negative"]["ports"]["Hnext"]}
    require(protocol["world_order"] == ["full-input", "cheap-input"] and protocol["world_sources"] == expected_worlds,
        "Explicit saved forecast world descriptors differ")
    for label, wrapper, seal, records in (("world", world_files, world_seal, protocol["world_sources"]),
            ("context", context_files, context_seal, protocol["shared_context_sources"])):
        root = protocol["roots"][label]
        require(wrapper["job_id"] == root["job_id"] and wrapper["request_sha256"] == root["request_sha256"], "Inherited raw root identity differs: " + label)
        external = seal["declarations"]["frontier"]["external_files"]
        for name, row in records.items():
            path = Path(row["path"])
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe inherited member: " + name)
            require(wrapper["files"].get(row["path"]) == external.get(row["path"]) == dict(bytes=row["bytes"], sha256=row["sha256"]),
                "Inherited member not joined to raw/global pretarget seal: " + name)
    require(len(protocol["world_sources"]) == 2 and len(protocol["shared_context_sources"]) == 5,
        "Exactly two world and five context arrays required")


def metadata(world_run, world_compact, world_seal, context_run, context_compact, context_seal, selection, parameter_subset):
    from mdb.parameter_artifacts import validate_manifest

    h = utilities()
    read, sha, require = (h[name] for name in ("read", "sha", "require"))
    world_run, world_compact, world_seal, context_run, context_compact, context_seal, selection, parameter_subset = map(Path,
        (world_run, world_compact, world_seal, context_run, context_compact, context_seal, selection, parameter_subset))
    require(sha(selection / "protocol.json") == PROTOCOL_SHA256, "Frozen block19 selection differs")
    protocol = read(selection / "protocol.json")
    requests = {"world": read(world_run / "request.json"), "context": read(context_run / "request.json")}
    wrappers = {}
    for label, compact, seal in (("world", world_compact, world_seal), ("context", context_compact, context_seal)):
        root, request = protocol["roots"][label], requests[label]
        require(h["identity"](request) == request.get("request_sha256") == root["request_sha256"], "Original source request differs: " + label)
        require(sha(compact / "source-files.json") == root["raw_manifest_sha256"] and sha(seal) == root["global_seal_sha256"],
            "Original raw manifest/global seal differs: " + label)
        wrappers[label] = read(compact / "source-files.json")
    for copied, original in (("source-forecast.json", "forecast.json"), ("source-shortcut.json", "shortcut.json")):
        path = world_compact / copied
        require(sha(path) == wrappers["world"]["files"][original]["sha256"] and path.stat().st_size == wrappers["world"]["files"][original]["bytes"],
            "Original world descriptor metadata differs: " + original)
    source_descriptors(protocol, wrappers["world"], read(world_seal), wrappers["context"], read(context_seal),
        read(world_compact / "source-forecast.json"), read(world_compact / "source-shortcut.json"))
    full_path = context_run / "payload/qwen-parameter-artifacts.json"
    require(sha(parameter_subset) == SUBSET_SHA256 == protocol["qwen_parameter_manifest_sha256"]
        and sha(full_path) == protocol["parent_parameter_manifest_sha256"], "Original block19 recipe identities differ")
    full, subset = (validate_manifest(read(path)) for path in (full_path, parameter_subset))
    expected = [row for row in full["artifacts"] if row["source"]["tensor"].startswith("model.layers.19.")]
    require(len(full["artifacts"]) == 398 and len(expected) == 11 and subset["artifacts"] == expected, "Exact original eleven block19 recipes required")
    paths = dict(selection_protocol=selection / "protocol.json", qwen_parameter_manifest=parameter_subset, parent_parameter_manifest=full_path,
        qwen_config=context_run / "payload/qwen-config.json", qwen_source_binding=context_run / "payload/qwen-source-binding.json",
        retrieval_script=context_run / "payload/retrieve_qwen_weights.py", world_files=world_compact / "source-files.json", world_seal=world_seal,
        world_request=world_run / "request.json", world_forecast=world_compact / "source-forecast.json", world_shortcut=world_compact / "source-shortcut.json",
        context_files=context_compact / "source-files.json", context_seal=context_seal, context_request=context_run / "request.json")
    for key in ("qwen_config", "qwen_source_binding", "parent_parameter_manifest"):
        require(sha(paths[key]) == protocol[key + "_sha256"], "Original model metadata differs: " + key)
    for name in ("qwen-config.json", "qwen-source-binding.json", "retrieve_qwen_weights.py", "qwen-parameter-artifacts.json"):
        require(sha(context_run / "payload" / name) == requests["context"]["source_hashes"][name], "Authenticated source metadata differs: " + name)
    require(requests["world"]["execution_environment"] == requests["context"]["execution_environment"], "Source execution environments differ")
    return requests["world"], requests["context"], protocol, paths


def sizing(resource_job=None):
    if resource_job is None:
        return dict(mode="intent-first", timeout_s=OUTER_TIMEOUT, reservation=None)
    h = utilities()
    h["require"](h["sha"](resource_job) == RESOURCE_JOB_SHA256, "Actual measured component job changed")
    job = h["read"](resource_job)
    h["require"](job["job_id"] == "job-74efe5506ecf" and job["state"] == "succeeded" and job["result"]["returncode"] == 0,
        "Successful measured component required")
    result = job["result"]
    h["require"]((result["elapsed_s"], result["peak_rss_mb"], result["peak_vram_mb"]) == (47.22, 1795.4, 826.0), "Measured component observations differ")
    return dict(mode="measured-component-exception", timeout_s=OUTER_TIMEOUT, reservation=dict(RESOURCE),
        measurement=dict(job_id=job["job_id"], job_record_sha256=RESOURCE_JOB_SHA256,
            **{key: result[key] for key in ("elapsed_s", "peak_rss_mb", "peak_vram_mb")}),
        rationale="Block18 measured2full functional+3native B1/512, same201861632B selected payload; block19 uses2+2. Whole-shard hashing distinct. New runtime/peak remain unmeasured; explicit exception only when planner lacks component geometry.")


def runtime_wheels():
    from mdb.payload import build_runtime_wheel, runtime_distributions

    with tempfile.TemporaryDirectory(prefix="mdb-component-cohort-") as temporary:
        return {path.name: utilities()["sha"](path) for package, distribution, version in runtime_distributions(require_installed=True)
            for path in [build_runtime_wheel(temporary, package=package, distribution=distribution, version=version)]}


def command(request):
    from mdb.payload import worker_command

    utilities()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH", "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return [*args, "--config", "saturn_run_config.json"]


def stage(run, world_run, world_compact, world_seal, context_run, context_compact, context_seal, selection, parameter_subset, *, resource_job=None):
    from mdb.job_support import one_lease_scheduling_contract
    from mdb.payload import stage_worker

    h, worker = utilities(), producer()
    world, context, protocol, paths = metadata(world_run, world_compact, world_seal, context_run, context_compact, context_seal, selection, parameter_subset)
    h["require"](set(paths) == set(worker["PATH_KEYS"]) and protocol["physical_plan"] == worker["C"]["contract"](), "Producer path/count ABI differs")
    basis, wheels = sizing(resource_job), runtime_wheels()
    run = Path(run).resolve(); run.mkdir(parents=True, exist_ok=False)
    names = {key: key.replace("_", "-") + path.suffix for key, path in paths.items()}
    for key, path in paths.items():
        shutil.copyfile(path, run / names[key])
    base = {key: names[key] for key in paths}
    base.update({key + "_sha256": h["sha"](path) for key, path in paths.items()})
    base.update(parameter_root=context["config"]["parameter_root"], world_root=protocol["roots"]["world"]["root"], context_root=protocol["roots"]["context"]["root"])
    config = worker["make_config"](base)
    counts = worker["validate_config"]({**config, **{key: str(run / names[key]) for key in paths}})
    for name, value in (("count-contract.json", counts), ("resource-basis.json", basis)):
        h["save"](run / name, value)
    pairs = {WORKER: HERE / WORKER, Path(__file__).name: Path(__file__), UTILITIES: HERE / UTILITIES,
        **{name: HERE / name for name in worker["HELPERS"]}, **{names[key]: run / names[key] for key in paths},
        **{name: run / name for name in ("count-contract.json", "resource-basis.json")}}
    h["save"](run / "preflight.v1.json", dict(schema="mrun-preflight-config-v1", hashed_roots=[*pairs, "preflight.v1.json"],
        path_token_allowlist=[config[key] for key in (*worker["ROOT_KEYS"], "output_scratch", "output_name")]))
    pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-qwen-error-consumer-request-v1", config=config, execution_environment=world["execution_environment"],
        scheduler_url=world["scheduler_url"], runtime_wheels=wheels, model="qwen3-block19-in-flux2-klein-4b", task_family="analysis",
        backend="torch-native-qwen-error-consumer", device="cuda", dtype="bfloat16", needs={"cuda": True, "host": "beast"},
        source_hashes={name: h["sha"](path) for name, path in pairs.items()}, launcher_sha256=h["sha"](__file__), sizing=basis,
        resource_exception=basis["reservation"], workload_geometry=protocol["physical_plan"],
        source_roots=protocol["roots"], source_runtime_wheels=dict(world=world["runtime_wheels"], context=context["runtime_wheels"]),
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=2, logical_operations=4, logical_checkpoint_replays=0,
            max_local_work_items=4, extra=protocol["physical_plan"], checkpoint_policy="One finite B1 block19 lease;two explicit own-input functional forecasts globally seal before exactly two native targets;no extra baseline/retries/nesting."))
    request["request_sha256"] = h["identity"](request)
    h["save"](run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
        additional_files=[(path, name) for name, path in pairs.items() if name != WORKER])
    inspect_run(run)
    h["save"](run / "staged-command.json", dict(command=command(request), request_sha256=request["request_sha256"], submitted=False))
    return dict(request_sha256=request["request_sha256"], source_files=len(pairs), recipes=11, inherited_array_references=7, activation_bytes_copied=0, sizing=basis["mode"], submitted=False)


def inspect_run(run):
    h, run = utilities(), Path(run)
    request = h["read"](run / "request.json")
    h["require"](h["identity"](request) == request.get("request_sha256") and request["launcher_sha256"] == h["sha"](__file__), "Request/frontend identity differs")
    h["require"](request["source_hashes"].get(WORKER) == WORKER_SHA256 and request["source_hashes"].get(UTILITIES) == UTILITIES_SHA256, "Qualified source closure differs")
    basis = request["sizing"]
    h["require"](basis["mode"] in ("intent-first", "measured-component-exception") and basis["timeout_s"] == OUTER_TIMEOUT
        and request["resource_exception"] == basis["reservation"] == (RESOURCE if basis["mode"] == "measured-component-exception" else None), "Declared sizing mode/resources differ")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        h["require"](h["sha"](run / "payload" / name) == digest, "Payload source/wheel changed: " + name)
    h["require"](h["read"](run / "payload/saturn_debug_request.json") == request and h["read"](run / "payload/saturn_run_config.json") == request["config"], "Payload request/config differs")
    seal = h["read"](run / "payload/payload-seal.json")
    h["require"](seal["files"] == request["source_hashes"] and seal["wheels"] == request["runtime_wheels"] and seal["worker"] == WORKER
        and seal["worker_sha256"] == WORKER_SHA256 and seal["config_bytes_sha256"] == h["sha"](run / "payload/saturn_run_config.json"), "Payload seal differs")
    return request


def submit(run):
    from mdb.job_support import guarded_mrun_launch

    h, run = utilities(), Path(run)
    request = inspect_run(run)
    h["require"](not (run / "submission.json").exists(), "Submission already exists")
    review = h["read"](run / "finite-run-review-v1.json")
    h["require"](review.get("request_sha256") == request["request_sha256"] and review.get("blocking_issues") == [] and review.get("review_status") == "ready-for-finite-run", "Exact finite review required")
    h["require"](len(QUEUE_NOTE) <= 500, "Queue note exceeds scheduler limit")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    h["save"](run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=h["sha"](run / "finite-run-review-v1.json")))
    reservation = request["resource_exception"] or {}
    job = guarded_mrun_launch(args, request=request, experiment="mdb-qwen-error-consumer", task="analysis",
        model=request["model"] if not reservation else None, payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        timeout_s=OUTER_TIMEOUT, retry_on_kill=False, preflight=True, note=QUEUE_NOTE, **reservation)
    h["save"](run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    fields = ("world-run", "world-compact", "world-seal", "context-run", "context-compact", "context-seal", "selection", "parameter-subset")
    for name in (*fields, "resource-job"):
        parser.add_argument("--" + name, type=Path)
    mode = parser.add_mutually_exclusive_group(); mode.add_argument("--inspect", action="store_true"); mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.inspect or args.submit:
        result = submit(args.run) if args.submit else inspect_run(args.run)
    else:
        if any(getattr(args, name.replace("-", "_")) is None for name in fields):
            parser.error("all two-root source metadata, selection and subset arguments are required to stage")
        result = stage(args.run, *(getattr(args, name.replace("-", "_")) for name in fields), resource_job=args.resource_job)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Metadata-only row-sign staging. Only the owning root executes stage/submit.

Thirteen inherited functional arrays stay at their authenticated parent path.
No model, tokenizer, parameter payload, activation array or service is read here.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
UTILITIES = "stage_qwen_sign_shortcut.py"
UTILITIES_SHA256 = "d73f85f73030dff5f081d3742ffc750eb966a27fee694c4782488f270c806a72"
WORKER = "qwen_row_sign_campaign.py"
WORKER_SHA256 = "b3d4ae8b0cbddf3c0477ace8644b19e25f42a5f820f2def9b06981da7cfd99d4"
PROTOCOL_SHA256 = "d942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9"
SUBSET_SHA256 = "425502121b4091b10124c548c98127785420233b12e95c8f1d972a186026dccf"
PARENT_MANIFEST_SHA256 = "d5060dd8137bf34568d3d9b0fda805c5979133b0c4a4538e23579196cb5d3374"
PARENT_REQUEST = "1a9b11e4bc739410dcd165b31cd0c0452aa5f0583e7731598bc1e2b8a87bdd5a"
COMPACT_SHA256 = "73a1eb8cd5e0557ac893a1d6d14e6c24a0ea2dbb4107684cf47d748dbb8d0d00"
SDPA_SHA256 = "b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46"
DERIVED_SHA256 = "34d59492473e3496681da2cedb1d2a5a8c42d51ebb96900f3937782f81db3f4d"
QUEUE_NOTE = ("Qwen block18 conditional row-sign:2 functional edited blocks+2 shortcut down projections globally sealed "
              "before3 native component calls;B1/512,no full encoder/LM/embedding/tokenizer/Flux/VAE/noise. One 201.9MB "
              "11-tensor hydration;~4.97GB shard hash startup separate. CPU4/RAM4GiB/VRAM2GiB,disk3GB,60s estimate/240s "
              "assay/300s outer;single CUDA Beast lease,no retries/nesting. Numerical misses retained.")
RESOURCE = dict(cpu_threads=4, ram_mb=4096, vram_mb=2048, disk_gb=3, est_wall_s=60, timeout_s=300)


def utilities():
    import hashlib

    path = HERE / UTILITIES
    if hashlib.sha256(path.read_bytes()).hexdigest() != UTILITIES_SHA256:
        raise ValueError("Frozen metadata utilities changed")
    return runpy.run_path(str(path))


def producer():
    h = utilities()
    h["require"](h["sha"](HERE / WORKER) == WORKER_SHA256, "Qualified row-sign producer changed")
    worker = runpy.run_path(str(HERE / WORKER))
    for name, digest in worker["HELPERS"].items():
        h["require"](Path(name).name == name and h["sha"](HERE / name) == digest, "Producer helper changed: " + name)
    return worker


def metadata(parent_run, compact, selection, frontier):
    """Validate JSON/source metadata only, including exact original recipe entries."""
    from mdb_lanes.parameter_artifacts import validate_manifest

    h = utilities()
    read, sha, require = h["read"], h["sha"], h["require"]
    parent_run, compact, selection, frontier = map(Path, (parent_run, compact, selection, frontier))
    parent = read(parent_run / "request.json")
    require(h["identity"](parent) == parent.get("request_sha256") == PARENT_REQUEST, "Exact corrected parent request required")
    require(sha(compact / "files.json") == COMPACT_SHA256, "Accepted parent compact manifest changed")
    for name, digest in read(compact / "files.json").items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(compact / name) == digest, "Compact metadata custody differs: " + name)
    require(read(compact / "source-request.json") == parent, "Compact/original parent requests differ")
    protocol = read(selection / "protocol.json")
    require(sha(selection / "protocol.json") == PROTOCOL_SHA256, "Frozen row-sign protocol changed")
    wrapper = read(compact / "source-files.json")
    require(sha(compact / "source-files.json") == protocol["parent_raw_manifest_sha256"], "Original raw manifest changed")
    require(wrapper["job_id"] == protocol["parent_job"] and wrapper["request_sha256"] == PARENT_REQUEST, "Original raw identity differs")
    require(len(protocol["source_arrays"]) == 13, "Thirteen inherited source arrays required")
    for row in protocol["source_arrays"].values():
        indexed = wrapper["files"].get(row["path"])
        require(indexed == {"bytes": row["bytes"], "sha256": row["sha256"]}, "Inherited array manifest join differs")
    subset_path, full_path = selection / "qwen-block18-parameter-artifacts.json", parent_run / "payload/qwen-parameter-artifacts.json"
    require(sha(subset_path) == SUBSET_SHA256 and sha(full_path) == PARENT_MANIFEST_SHA256, "Fixed subset/full recipe identity differs")
    full, subset = validate_manifest(read(full_path)), validate_manifest(read(subset_path))
    expected = [row for row in full["artifacts"] if row["source"]["tensor"].startswith("model.layers.18.")]
    require(len(full["artifacts"]) == 398 and len(subset["artifacts"]) == len(expected) == 11, "Original recipe counts differ")
    require(sorted(subset["artifacts"], key=lambda r: r["path"]) == sorted(expected, key=lambda r: r["path"]), "Subset entries differ from original recipes")
    receipt = read(selection / "parameter-selection.json")
    require(receipt["subset_manifest_sha256"] == SUBSET_SHA256 and receipt["parent_manifest_sha256"] == PARENT_MANIFEST_SHA256 and receipt["protocol_sha256"] == PROTOCOL_SHA256, "Subset selection receipt differs")
    require(sha(selection / "derived-predictions.json") == DERIVED_SHA256, "Frozen derived predictions changed")
    seal = frontier / "transaction/seal.json"
    require(sha(seal) == protocol["parent_global_seal_sha256"] == read(compact / "source-summary.json")["forecast_seal"], "Original global seal differs")
    paths = {"qwen_config": parent_run / "payload/qwen-config.json", "qwen_parameter_manifest": subset_path,
        "parent_parameter_manifest": full_path, "qwen_source_binding": parent_run / "payload/qwen-source-binding.json",
        "retrieval_script": parent_run / "payload/retrieve_qwen_weights.py", "selection_protocol": selection / "protocol.json",
        "parent_files": compact / "source-files.json", "parent_request": parent_run / "request.json",
        "parent_source": compact / "source-source.json", "parent_runtime": compact / "source-runtime.json", "parent_seal": seal}
    for name in ("qwen-config.json", "qwen-source-binding.json", "retrieve_qwen_weights.py", "qwen-parameter-artifacts.json"):
        require(sha(parent_run / "payload" / name) == parent["source_hashes"][name], "Original metadata source differs: " + name)
    return parent, protocol, paths


def command(request):
    from mdb_lanes.payload import worker_command

    utilities()["bind"](request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    for key, name in {"tmpdir": "TMPDIR", "cuda_cache_path": "CUDA_CACHE_PATH", "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR"}.items():
        args.insert(1, name + "=" + request["execution_environment"][key])
    return [*args, "--config", "saturn_run_config.json"]


def stage(run, parent_run, compact, selection, frontier):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    h, worker = utilities(), producer()
    parent, protocol, paths = metadata(parent_run, compact, selection, frontier)
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    # Rename the full original manifest to avoid colliding with the exact subset.
    names = {key: ("parent-parameter-artifacts.json" if key == "parent_parameter_manifest" else
                   "parent-request.json" if key == "parent_request" else "parent-seal.json" if key == "parent_seal" else path.name)
             for key, path in paths.items()}
    for key, path in paths.items():
        shutil.copyfile(path, run / names[key])
    for name in ("parameter-selection.json", "derived-predictions.json"):
        shutil.copyfile(Path(selection) / name, run / name)
    base = {key: names[key] for key in paths}
    base.update({key + "_sha256": h["sha"](path) for key, path in paths.items()})
    base.update(parameter_root=parent["config"]["parameter_root"], input_root=protocol["parent_raw_root"],
        qwen_native_source_sha256=parent["config"]["qwen_native_source_sha256"], sdpa_source_sha256=SDPA_SHA256)
    config = worker["make_config"](base)
    local = {**config, **{key: str(run / names[key]) for key in paths}}
    counts = worker["validate_config"](local)
    h["save"](run / "count-contract.json", counts)
    pairs = {WORKER: HERE / WORKER, Path(__file__).name: Path(__file__), UTILITIES: HERE / UTILITIES,
        **{name: HERE / name for name in worker["HELPERS"]}, **{names[key]: run / names[key] for key in paths},
        **{name: run / name for name in ("parameter-selection.json", "derived-predictions.json")},
        "count-contract.json": run / "count-contract.json"}
    preflight = dict(schema="mrun-preflight-config-v1", hashed_roots=[*pairs, "preflight.v1.json"],
        path_token_allowlist=[config["parameter_root"], config["input_root"], config["output_scratch"], config["output_name"]])
    h["save"](run / "preflight.v1.json", preflight)
    pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = dict(schema="mdb-qwen-row-sign-request-v1", config=config, execution_environment=parent["execution_environment"],
        scheduler_url=parent["scheduler_url"], parent_request_sha256=PARENT_REQUEST, runtime_wheels=parent["runtime_wheels"],
        model="qwen3-block18-in-flux2-klein-4b", task_family="analysis", backend="torch-native-qwen-row-sign", device="cuda",
        dtype="bfloat16", needs={"cuda": True, "host": "beast"}, source_hashes={name: h["sha"](path) for name, path in pairs.items()},
        launcher_sha256=h["sha"](__file__), resource_exception={**RESOURCE,
            "calibration": "Unmeasured component successor: 201861632B BF16 parameters; original source-shard hash streams ~4.97GB once. This startup I/O is distinct from selected component payload and from arithmetic. No successor runtime/peak claim."},
        workload_geometry={"component_hydrations": 1, "cuda_placements": 1, "cpu_offloads": 1,
            **protocol["physical_plan"], "inherited_source_arrays": 13, "parameter_payload_bytes": 201861632},
        scheduling_contract=one_lease_scheduling_contract(model_loads=1, logical_branches=3, logical_operations=7,
            logical_checkpoint_replays=0, max_local_work_items=7, extra=protocol["physical_plan"],
            checkpoint_policy="One finite B1 component lease; two functional blocks+two shortcuts globally seal before three native targets. No retries/nested leases/full encoder/LM/tokenizer/Flux/VAE/noise."))
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
    h["require"](request["source_hashes"].get(WORKER) == WORKER_SHA256 and request["source_hashes"].get(UTILITIES) == UTILITIES_SHA256, "Frozen producer/utilities differ")
    h["require"](all(request["resource_exception"].get(k) == v for k, v in RESOURCE.items()), "Fixed component resources differ")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        h["require"](h["sha"](run / "payload" / name) == digest, "Payload source/wheel changed: " + name)
    h["require"](h["read"](run / "payload/saturn_debug_request.json") == request
        and h["read"](run / "payload/saturn_run_config.json") == request["config"], "Payload request/config differs")
    seal = h["read"](run / "payload/payload-seal.json")
    h["require"](seal["files"] == request["source_hashes"] and seal["wheels"] == request["runtime_wheels"]
        and seal["worker"] == WORKER and seal["worker_sha256"] == WORKER_SHA256
        and seal["config_bytes_sha256"] == h["sha"](run / "payload/saturn_run_config.json"), "Payload seal differs")
    return request


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    h, run = utilities(), Path(run)
    request = inspect_run(run)
    h["require"](not (run / "submission.json").exists(), "Original submission already exists")
    review = h["read"](run / "finite-run-review-v1.json")
    h["require"](review.get("request_sha256") == request["request_sha256"] and review.get("blocking_issues") == [] and review.get("review_status") == "ready-for-finite-run", "Exact finite review required")
    h["require"](len(QUEUE_NOTE) <= 500, "Queue note exceeds scheduler limit")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    args = command(request)
    h["save"](run / "admission-command.json", dict(command=args, request_sha256=request["request_sha256"], review_sha256=h["sha"](run / "finite-run-review-v1.json")))
    job = guarded_mrun_launch(args, request=request, experiment="mdb-qwen-row-sign", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True, **RESOURCE,
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    h["save"](run / "submission.json", dict(job_id=str(job)))
    return dict(job_id=str(job))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    for name in ("parent-run", "compact", "selection", "frontier"):
        parser.add_argument("--" + name, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.inspect or args.submit:
        result = submit(args.run) if args.submit else inspect_run(args.run)
    else:
        if any(getattr(args, name) is None for name in ("parent_run", "compact", "selection", "frontier")):
            parser.error("--parent-run, --compact, --selection and --frontier are required to stage")
        result = stage(args.run, args.parent_run, args.compact, args.selection, args.frontier)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Stage two component captures from authenticated MDB cuts; keep raw traces near data."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_image_tail_microscope.py"
HELPERS = ("flux_reader_arithmetic.py", "flux_source_receiver.py", "flux_reentry_reference.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def bind(environment):
    from mdb_lanes.execution_environment import intent
    for key, name in {"uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR", "torch_home": "MDB_WORKER_TORCH_HOME",
                      "extra_model_roots": "MDB_WORKER_MODEL_ROOTS", "python": "MDB_WORKER_PYTHON",
                      "uv_link_mode": "MDB_WORKER_UV_LINK_MODE"}.items():
        os.environ[name] = environment[key]
    if intent() != {key: environment[key] for key in intent()}:
        raise ValueError("installed execution requirements differ from the source cohort")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"


def select_inputs(source, inventory):
    """Only exact source cuts and their complete pages/blobs enter the new worker."""
    files, parents = {}, {}

    def admit(name):
        record = inventory[name]
        path = source / name
        if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("authenticated source changed: " + name)
        files[name] = {"sha256": record["sha256"], "bytes": record["bytes"]}

    for role, branch in (("sham", "c1-o0.sham"), ("edit", "c1-o0.d-c0-o1")):
        name = "native/" + branch + ".jsonl"
        admit(name)
        rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
        if len(rows) != 84 or (rows[0]["logical_step"], rows[0]["local_index"], rows[1]["local_index"]) != (1, 0, 1):
            raise ValueError("source ledger geometry differs")
        capture = rows[0]["capture"]
        if not capture["retained_full_cut"] or capture["point"] != {
                "edge": "before", "family": "flux2-klein", "index": 0, "local_clock": 0,
                "logical_step": 1, "next_operation": "joint_block.0", "operator": "joint", "phase": "joint"}:
            raise ValueError("source is not the complete step1/joint0 entry")
        name = "typed-store/cuts/" + capture["cut"] + ".json"
        admit(name)
        cut = read(source / name)
        if cut["fingerprint"] != capture["cut"] or cut["execution_point"] != capture["point"]:
            raise ValueError("source cut identity differs")
        for page in cut["slots"].values():
            admit("typed-store/pages/" + page + ".json")
        for blob in cut["physical_blobs"]:
            admit("typed-store/blobs/" + blob + ".safetensors")
        parents[role] = {"store": "typed-store", "cut": capture["cut"], "branch": branch,
            "historical_image_descriptor": rows[1]["outputs"]["image"]["descriptor"],
            "historical_text_descriptor": rows[1]["outputs"]["text"]["descriptor"]}
    for name in ("source.json", "parameter-artifacts.json", "retrieve_weights.py"):
        admit(name)
    return parents, files


def command(request):
    from mdb_lanes.payload import worker_command
    bind(request["execution_environment"])
    result = worker_command(WORKER)
    index = next(i for i in range(len(result)) if result[i:i + 4] == ["python", "-u", "-m", "mdb_lanes.worker_bootstrap"])
    result.insert(index + 1, "-B")
    # Retain the source's cache locations while avoiding bytecode writes.
    env = request["execution_environment"]
    for key, name in (("tmpdir", "TMPDIR"), ("cuda_cache_path", "CUDA_CACHE_PATH"),
                      ("torchinductor_cache_dir", "TORCHINDUCTOR_CACHE_DIR")):
        result.insert(1, name + "=" + env[key])
    return result


def validate_payload(run, request):
    unsigned = {key: value for key, value in request.items() if key != "request_sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != request["request_sha256"] or sha(__file__) != request["launcher_sha256"]:
        raise ValueError("request or launcher differs")
    for name, expected in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        if sha(run / "payload" / name) != expected:
            raise ValueError("payload differs: " + name)


def stage(run, campaign):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker
    run, campaign = Path(run).resolve(), Path(campaign).resolve()
    native = campaign / "latent-direction-native-01"
    source = native / "extracted/latent-direction-native-result"
    collection, old = read(native / "collection.json"), read(native / "request.json")
    analysis = read(campaign / "latent-direction-analysis-v1/analysis.json")
    if (collection["job_id"] != "job-46a52ffae499" or collection["scheduler_status"] != "failed"
            or not collection["scientific_work_completed"]
            or collection["manifest_origin"] != "collector-derived-source-manifest"
            or analysis["mechanics_status"] != "authenticated-saved-array-reductions"):
        raise ValueError("accepted recovered cross evidence required")
    # Bind the actual recovered inventory through its immutable collection receipt.
    inventory_path = native / "collection-local-inventory.json"
    if sha(inventory_path) != collection["bindings"][inventory_path.name]:
        raise ValueError("source inventory binding changed")
    inventory = read(inventory_path)["files"]
    parents, files = select_inputs(source, inventory)
    binding_path = campaign / "run-01/extracted/flux-prompt-factorial-result/binding.json"
    original_inventory = read(binding_path.parent / "files.json")
    if sha(binding_path) != original_inventory["binding.json"]:
        raise ValueError("original parameter binding changed")
    joint = read(binding_path)["joint0"]
    if set(joint["parameters"]) != set(joint["native_parameters"]):
        raise ValueError("complete original block parameter binding required")
    bind(old["execution_environment"])
    module_path = Path(importlib.metadata.distribution("diffusers").locate_file("diffusers/models/transformers/transformer_flux2.py"))
    config = {"schema": "mdb-flux-image-tail-plan-v1", "input_root": collection["remote_path"],
        "inputs": parents, "input_files": files, "parameter_manifest": "parameter-artifacts.json",
        "retrieval_script": "retrieve_weights.py",
        "parameter_root": old["config"]["model_path"], "joint_parameters": joint["parameters"],
        "module_config": {"dim": 3072, "num_attention_heads": 24, "attention_head_dim": 128,
            "mlp_ratio": 3.0, "eps": joint["layernorm_eps"], "bias": False},
        "native_source_sha256": sha(module_path), "helper_source_sha256": {name: sha(HERE / name) for name in HELPERS},
        "expected_model_identity": read(source / "source.json")["model_identity"], "device": "cuda",
        "source_reduction_policy": True, "reference_head": True, "query_chunk": 64,
        "cpu_threads": 4, "output_scratch": "/mnt/big/mdb/research", "output_name": "binding-minimal-result",
        "directory_custody": True, "max_wall_seconds": 600,
        "source_job": collection["job_id"], "source_collection_sha256": sha(native / "collection.json"),
        "source_inventory_sha256": sha(inventory_path), "source_binding_sha256": sha(binding_path),
        "selection": "Retrospective h,l edit and its own h sham, both exact observed step1/joint0 inputs.",
        "scope": "Two complete native joint0 component captures; image-facing observed-operand native and independent arithmetic tails. No full model/pipeline, encoder, noise draw, suffix, VAE or new prospective forecast. Finite numerical substitutions, not semantic rules or causal shares."}
    run.mkdir(parents=True, exist_ok=False)
    for target, path in (("parameter-artifacts.json", source / "parameter-artifacts.json"),
                         ("retrieve_weights.py", source / "retrieve_weights.py"), ("source-binding.json", binding_path)):
        shutil.copyfile(path, run / target)
    save(run / "plan.json", config)
    extras = [(HERE / name, name) for name in HELPERS]
    extras += [(run / name, name) for name in ("parameter-artifacts.json", "retrieve_weights.py", "source-binding.json")]
    preflight = run / "preflight.v1.json"
    save(preflight, {"schema_version": "mrun-preflight-config-v1", "hashed_roots": [WORKER, *[name for _, name in extras], preflight.name],
        "path_token_allowlist": [config["input_root"], config["parameter_root"], config["output_scratch"], "binding-minimal-result"]})
    extras.append((preflight, preflight.name))
    request = {"schema": "mdb-flux-image-tail-request-v1", "config": config,
        "execution_environment": old["execution_environment"], "scheduler_url": old["scheduler_url"],
        "model": "flux2-klein-4b-distilled", "task_family": "analysis", "backend": "torch-component-and-numpy",
        "device": "cuda", "dtype": "bfloat16-native/float64-reference", "needs": {"cuda": True, "host": "beast"},
        "source_hashes": {name: sha(path) for path, name in [(HERE / WORKER, WORKER), *extras]},
        "runtime_wheels": old["runtime_wheels"], "launcher_sha256": sha(__file__),
        "scheduling_contract": one_lease_scheduling_contract(model_loads=0, model_free=True, logical_operations=16,
            logical_branches=2, logical_checkpoint_replays=2, max_local_work_items=32,
            extra={"task_family": "analysis", "native_component_calls": 2},
            checkpoint_policy="One component-only CUDA lease; two saved-state joint0 calls plus finite operand/tail lanes; no full pipeline or native suffix, no retries."),
        "resource_exception": {"ram_mb": 8192, "vram_mb": 3072, "cpu_threads": 4, "disk_gb": 4,
            "est_wall_s": 300, "timeout_s": 1200,
            "calibration": "Prior reader-attention RSS2252.7MB/VRAM824MB includes three text matrices188.7MB BF16. This uses16 joint0 tensors about490.7MB BF16 (about981.5MB FP32 views), B1x1024 image/B1x512 text and bounded one-case FP64 equations. Earlier original single-block CPU assay job-8367b9ecabe1 peaked3922.7MB. 8192MBRAM/3072MBVRAM allow complete native block plus tail/reference working sets; no full model or GPU FP64 bank. Trace remains on Beast, compact audit only collected. This is a measured-geometry safety envelope, not a speed claim."}}
    request["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request, additional_files=extras)
    validate_payload(run, request)
    save(run / "staged-command.json", {"command": command(request), "request_sha256": request["request_sha256"]})
    return {"request_sha256": request["request_sha256"], "input_files": len(files), "submitted": False}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch
    run = Path(run).resolve()
    request = read(run / "request.json")
    validate_payload(run, request)
    if (run / "submission.json").exists():
        raise ValueError("run already submitted")
    review = read(run / "long-run-review-final.json")
    if review.get("request_sha256") != request["request_sha256"] or review.get("blocking_issues") != [] or review.get("review_status") != "ready-for-finite-run":
        raise ValueError("source-bound finite-run review required")
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"], "review_sha256": sha(run / "long-run-review-final.json")})
    resource = request["resource_exception"]
    job = guarded_mrun_launch(args, request=request, experiment="mdb-flux-image-tail-microscope", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resource[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True,
        note="Image-tail microscope:2 unchanged step1/joint0 component calls from saved h.sham/h,l cuts; original16joint weights, fullB1x1024image/512text. Native actualH/toggled/referenceH and independent local/carried tails; no fullmodel/suffix/VAE. One finite CUDA lease;300s estimate/600s scientific/1200s outer,no retries. Rawtrace remains near data, compact audit only returned.")
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(submit(args.run) if args.submit else stage(args.run, args.campaign)))

"""Stage the prospective Qwen sign-shortcut campaign without executing it.

Staging binds only the source metadata, six producer-declared helper files, and
the three literal raw-prompt files.  It does not read historical activation
arrays, construct a model, contact mrun, or submit a job.  ``--submit`` remains
an explicit follow-up for the owning launcher after a finite review.
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
WORKER = "qwen_sign_shortcut_campaign.py"
WORKER_SHA256 = "c5507557ad740a58341e4f99f4b384a9d27b82b6c268e2838469a19c2c8bfe76"
WORKER_TEST_SHA256 = "7228c7b91a64f80a10c186cb01a8fc5b87e5755954dcaf61fd8285eafb799ffc"
PARENT_REQUEST = "a69c20845dba8c20be4b76eff33ddfc89e27d8b016730b19029a2be8e9f55dea"
RESOURCE_JOB_ID = "job-ebae1e4bd523"
RESOURCE_JOB_SHA256 = "d031b1bcb96c43b62235929b03770e368e325a3a24bc78b0eab629ce3fc512ca"
RESOURCE_RECEIPT_SHA256 = "edca8760e2698069a32a58673647aa9cb29aaa3ff8e827607f9ace200fbf1dd9"
PRODUCER_QUALIFICATION_SHA256 = "a0d64ab540979ff01dcc09de1ae5c4c8a0b52a56d42b5afb6b3dd23d1efa292b"
PROTOCOL_SHA256 = "5ce41dda16fdfd687aba584f9aaedc389d0ec8bf088b2ea63de8b4456ddf9cf2"
METADATA = (
    "qwen-config.json",
    "qwen-parameter-artifacts.json",
    "qwen-source-binding.json",
    "retrieve_qwen_weights.py",
)
RAW = ("protocol.json", "short-objects.txt", "short-procedure.txt", "long-description.txt")
BASELINE_ROOT = Path("artifacts/mdb-qwen-sign-shortcut-20261106/source-baseline-collection-v1")
PRODUCER_QUALIFICATION = Path("artifacts/mdb-qwen-sign-shortcut-20261106/producer-qualification-v1/qualification.json")
SELECTION_ROOT = Path("artifacts/mdb-qwen-sign-shortcut-20261106/selection-v1")
QUEUE_NOTE = (
    "Qwen sign-shortcut,B1/512,3 raw contexts:60 functional blocks+3 own down shortcuts sealed before3 native full36+suffix calls (111 native blocks/6 native roots); Qwen-only,no Flux/VAE/noise/extra Acts. One meta-to-CUDA BF16 hydration/offload,no parameter exports. CUDA Beast,CPU4/RAM16GiB/VRAM10GiB,disk4GB;300s estimate/600s assay/900s outer,one lease,no retries/nesting."
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(request):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True,
        allow_nan=False,
    ).encode()).hexdigest()


def producer():
    path = HERE / WORKER
    require(sha(path) == WORKER_SHA256, "qualified sign-shortcut producer changed")
    namespace = runpy.run_path(str(path))
    helpers = namespace.get("HELPERS")
    require(isinstance(helpers, dict) and len(helpers) == 6, "producer helper closure is not the qualified six-file ABI")
    for name, digest in helpers.items():
        require(Path(name).name == name and sha(HERE / name) == digest, "qualified helper changed: " + name)
    return namespace, dict(helpers)


def bind(environment):
    from mdb_lanes.execution_environment import intent

    for key, name in {
        "uv_cache_dir": "MDB_WORKER_UV_CACHE_DIR",
        "torch_home": "MDB_WORKER_TORCH_HOME",
        "extra_model_roots": "MDB_WORKER_MODEL_ROOTS",
        "python": "MDB_WORKER_PYTHON",
        "uv_link_mode": "MDB_WORKER_UV_LINK_MODE",
    }.items():
        os.environ[name] = environment[key]
    require(intent() == {key: environment[key] for key in intent()}, "installed execution requirements differ from source cohort")
    os.environ["MRUN_URL"] = "http://192.168.68.68:9025"


def command(request):
    from mdb_lanes.payload import worker_command

    bind(request["execution_environment"])
    args = worker_command(WORKER)
    args.insert(args.index("python") + 1, "-B")
    environment = request["execution_environment"]
    for key, name in {
        "tmpdir": "TMPDIR",
        "cuda_cache_path": "CUDA_CACHE_PATH",
        "torchinductor_cache_dir": "TORCHINDUCTOR_CACHE_DIR",
    }.items():
        args.insert(1, name + "=" + environment[key])
    args.extend(["--config", "saturn_run_config.json"])
    return args


def validate_resource_job(path, parent):
    """Bind the fresh source measurement without treating it as a successor run."""
    path = Path(path).resolve()
    require(sha(path) == RESOURCE_JOB_SHA256, "fresh Qwen baseline job changed")
    job = read(path)
    result = job.get("result", {})
    baseline_config = job.get("config", {})
    baseline_plan = baseline_config.get("config", {})
    require(job.get("job_id") == RESOURCE_JOB_ID and job.get("state") == "succeeded", "fresh baseline did not succeed")
    require(job.get("assigned_host") == "beast" and job.get("needs") == {"cuda": True, "host": "beast", "payload_custody_v2": True}, "fresh baseline host/device differs")
    require(result.get("returncode") == 0 and result.get("status") == "ok", "fresh baseline result is not successful")
    require(result.get("peak_rss_mb") == 4461.6 and result.get("peak_vram_mb") == 8790.0 and result.get("elapsed_s") == 95.23, "fresh baseline measurements changed")
    require(baseline_config.get("runtime_wheels") == parent["runtime_wheels"], "fresh baseline wheels differ")
    require(baseline_config.get("execution_environment") == parent["execution_environment"], "fresh baseline environment differs")
    parent_config = parent["config"]
    for key in ("qwen_config_sha256", "qwen_manifest_sha256", "qwen_source_binding_sha256", "qwen_retrieval_script_sha256"):
        require(baseline_plan.get(key) == parent_config[key], "fresh baseline source differs: " + key)
    custody = job.get("payload_custody", {})
    records = [custody.get(name) for name in ("declared", "sealed", "executed")]
    records.append(result.get("executed_payload"))
    require(all(isinstance(row, dict) for row in records), "fresh baseline custody is incomplete")
    require(len({(row.get("sha256"), row.get("size_bytes")) for row in records}) == 1, "fresh baseline payload custody differs")
    receipt = path.parent / "receipt.json"
    require(sha(receipt) == RESOURCE_RECEIPT_SHA256, "fresh baseline receipt changed")
    receipt_data = read(receipt)
    require(receipt_data.get("job_id") == RESOURCE_JOB_ID and receipt_data.get("fresh_actual_job_sha256") == RESOURCE_JOB_SHA256, "fresh baseline receipt binding differs")
    return dict(
        schema="mdb-qwen-sign-shortcut-resource-baseline-v1",
        job_id=RESOURCE_JOB_ID,
        job_sha256=RESOURCE_JOB_SHA256,
        receipt_sha256=RESOURCE_RECEIPT_SHA256,
        request_sha256=receipt_data["request_sha256"],
        elapsed_s=result["elapsed_s"],
        peak_rss_mb=result["peak_rss_mb"],
        peak_vram_mb=result["peak_vram_mb"],
        payload={"sha256": records[0]["sha256"], "size_bytes": records[0]["size_bytes"]},
        scope="Fresh Qwen prefix source measurement only; successor adds three contexts, shortcut down projections, and native suffixes. Memory/work ratios are planning observations, not a measured successor runtime.",
    )


def source_metadata(parent_run, selection_root, run):
    parent_run, selection_root, run = map(Path, (parent_run, selection_root, run))
    parent = read(parent_run / "request.json")
    payload = parent_run / "payload"
    copied = {}
    for name in METADATA:
        source = payload / name
        require(source.is_file() and parent["source_hashes"].get(name) == sha(source), "parent Qwen metadata differs: " + name)
        shutil.copyfile(source, run / name)
        copied[name] = run / name
    protocol = selection_root / "protocol.json"
    require(sha(protocol) == PROTOCOL_SHA256, "selection protocol changed")
    protocol_data = read(protocol)
    cases = protocol_data.get("cases")
    require(isinstance(cases, list) and len(cases) == 3, "three fixed raw contexts required")
    for row in cases:
        name = row.get("raw_file")
        require(isinstance(name, str) and Path(name).name == name and name in RAW, "unsafe raw prompt member")
        source = selection_root / name
        require(source.read_bytes() == row["raw_text"].encode() and source.stat().st_size == row["raw_bytes"] and sha(source) == row["raw_sha256"], "literal raw prompt changed: " + name)
    for name in RAW:
        source = protocol if name == "protocol.json" else selection_root / name
        shutil.copyfile(source, run / name)
        copied[name] = run / name
    return parent, protocol_data, copied


def build_config(parent, helpers, binding=None):
    """Create the producer's old-alias config from the parent Qwen metadata."""
    parent_config = parent["config"]
    if binding is None:
        parent_binding = Path(parent["config"]["qwen_source_binding"])
        binding = read(parent_binding if parent_binding.is_file() else "qwen-source-binding.json")
    if "checkpoint" not in binding:
        raise ValueError("Qwen source binding lacks checkpoint metadata")
    base = {
        "qwen_config": "qwen-config.json",
        "qwen_parameter_manifest": "qwen-parameter-artifacts.json",
        "qwen_source_binding": "qwen-source-binding.json",
        "retrieval_script": "retrieve_qwen_weights.py",
        "qwen_config_sha256": parent_config["qwen_config_sha256"],
        "qwen_parameter_manifest_sha256": parent_config["qwen_manifest_sha256"],
        "qwen_source_binding_sha256": parent_config["qwen_source_binding_sha256"],
        "retrieval_script_sha256": parent_config["qwen_retrieval_script_sha256"],
        "parameter_root": binding["checkpoint"]["text_encoder_root"],
        "tokenizer_path": binding["checkpoint"]["model_root"] + "/tokenizer",
    }
    for key in ("qwen_native_source_sha256", "tokenizers_wrapper_source_sha256", "tokenizers_native_source_sha256",
                "jinja_environment_source_sha256", "jinja_sandbox_source_sha256"):
        base[key] = parent_config[key]
    config = producer()[0]["make_config"](base)
    config.update(
        device="cuda",
        dtype="bfloat16",
        batch_size=1,
        max_length=512,
        padding="max_length",
        padding_side="right",
        truncation=True,
        apply_chat_template=dict(add_generation_prompt=True, enable_thinking=False, messages=[{"role": "user", "content": "<raw prompt>"}]),
        cohort="prospective-new-sign-shortcut",
        qwen_only=True,
        qwen_manifest_entries=398,
        no_flux_vae_noise=True,
        parameter_exports=0,
        component_hydrations=1,
        cuda_placements=1,
        cpu_offloads=1,
        global_forecast_seal_before_native=True,
        native_reference_policy="three full Qwen3ForCausalLM calls plus one H19 suffix block per context; native targets begin only after all forecast/shortcut responses seal",
        model_loads=1,
        retry=False,
        nested_mrun=False,
        max_wall_seconds=600,
        output_scratch="/mnt/big/mdb/research",
        output_name="qwen-sign-shortcut-result",
    )
    return config


def stage(run, parent_run, resource_job, selection=SELECTION_ROOT):
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, parent_run, resource_job, selection = (Path(path).resolve() for path in (run, parent_run, resource_job, selection))
    parent = read(parent_run / "request.json")
    require(identity(parent) == PARENT_REQUEST and parent.get("request_sha256") == PARENT_REQUEST, "exact H18 parent request required")
    require(parent["config"].get("qwen_manifest_sha256") == "d5060dd8137bf34568d3d9b0fda805c5979133b0c4a4538e23579196cb5d3374", "parent Qwen manifest differs")
    producer_namespace, helpers = producer()
    require(sha(PRODUCER_QUALIFICATION) == PRODUCER_QUALIFICATION_SHA256, "producer qualification receipt changed")
    producer_receipt = read(PRODUCER_QUALIFICATION)
    require(producer_receipt.get("status") == "qualified-tiny-mechanics-only" and producer_receipt.get("source_hashes", {}).get("qualification/" + WORKER) == WORKER_SHA256, "final producer receipt does not bind this worker")
    baseline = validate_resource_job(resource_job, parent)
    run.mkdir(parents=True, exist_ok=False)
    parent, protocol, copied = source_metadata(parent_run, selection, run)
    binding = read(run / "qwen-source-binding.json")
    manifest = read(run / "qwen-parameter-artifacts.json")
    qconfig = read(run / "qwen-config.json")
    require(len(manifest.get("artifacts", ())) == 398 and binding["checkpoint"]["selected_tensor_count"] == 398, "Qwen 398-recipe geometry differs")
    require(binding["checkpoint"]["selected_payload_bytes"] == 8044936192 and qconfig["hidden_size"] == 2560 and qconfig["num_hidden_layers"] == 36, "Qwen checkpoint geometry differs")
    config = build_config(parent, helpers, binding)
    config["qwen_source_binding"] = "qwen-source-binding.json"
    config["qwen_parameter_manifest"] = "qwen-parameter-artifacts.json"
    config["qwen_config"] = "qwen-config.json"
    config["retrieval_script"] = "retrieve_qwen_weights.py"
    local = dict(config, **{key: str(run / config[key]) for key in ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "retrieval_script")})
    counts = producer_namespace["validate_config"](local)
    require(counts["functional_reference_blocks"] == 60 and counts["shortcut_down_projections"] == 3 and counts["native_blocks"] == 111, "producer count contract differs")
    provenance = {
        "schema": "mdb-qwen-sign-shortcut-source-cohort-v1",
        "scope": "Prospective Qwen-only raw-prompt cohort; no historical activation inputs or FLUX/VAE/noise inputs are staged.",
        "parent_request_sha256": PARENT_REQUEST,
        "parent_job": "H18 source cohort request only; no parent activation member selected",
        "producer_worker_sha256": WORKER_SHA256,
        "producer_test_sha256": WORKER_TEST_SHA256,
        "producer_qualification_sha256": PRODUCER_QUALIFICATION_SHA256,
        "helper_source_sha256": helpers,
        "metadata_sha256": {name: sha(run / name) for name in METADATA},
        "raw_prompt_sha256": {name: sha(run / name) for name in RAW},
        "selection_protocol_sha256": PROTOCOL_SHA256,
        "qwen_geometry": {"artifacts": len(manifest["artifacts"]), "selected_payload_bytes": binding["checkpoint"]["selected_payload_bytes"], "layers": qconfig["num_hidden_layers"], "hidden_size": qconfig["hidden_size"], "intermediate_size": qconfig["intermediate_size"]},
        "tokenizer_path_policy": "sourcebinding.checkpoint.model_root + /tokenizer",
        "resource_measurement": baseline,
        "checkpoint_bytes_read_during_staging": 0,
        "model_construction_during_staging": False,
    }
    for name, value in (("plan.json", config), ("count-contract.json", counts), ("source-cohort.json", provenance)):
        save(run / name, value)
        copied[name] = run / name
    source_pairs = {WORKER: HERE / WORKER, Path(__file__).name: Path(__file__), **{name: HERE / name for name in helpers}, **copied}
    preflight = {
        "schema": "mrun-preflight-config-v1",
        "hashed_roots": [*source_pairs, "preflight.v1.json"],
        "path_token_allowlist": [config["parameter_root"], config["tokenizer_path"], config["output_scratch"], config["output_name"]],
        "no_native_targets_before_global_forecast_seal": True,
        "no_flux_vae_noise_calls": True,
        "no_persistent_parameter_payload_exports": True,
        "no_historical_activation_inputs": True,
    }
    save(run / "preflight.v1.json", preflight)
    copied["preflight.v1.json"] = run / "preflight.v1.json"
    source_pairs["preflight.v1.json"] = run / "preflight.v1.json"
    request = {
        "schema": "mdb-qwen-sign-shortcut-request-v1",
        "config": config,
        "parent_request_sha256": PARENT_REQUEST,
        "execution_environment": parent["execution_environment"],
        "scheduler_url": parent["scheduler_url"],
        "model": "qwen3-in-flux2-klein-4b",
        "task_family": "analysis",
        "backend": "torch-native-qwen-sign-shortcut",
        "device": "cuda",
        "dtype": "bfloat16",
        "needs": {"cuda": True, "host": "beast"},
        "runtime_wheels": parent["runtime_wheels"],
        "workload_geometry": {
            "model_loads": 1,
            "peak_physical_batch": 1,
            "contexts": 3,
            "sequence_rows": 512,
            "functional_reference_blocks": 60,
            "shortcut_down_projections": 3,
            "native_blocks": 111,
            "native_full_model_calls": 3,
            "native_suffix_calls": 3,
            "native_root_calls": 6,
            "native_LM_calls": 3,
            "functional_forecast_module_entries": 0,
            "component_hydrations": 1,
            "cuda_placements": 1,
            "cpu_offloads": 1,
            "parameter_exports": 0,
            "flux_calls": 0,
            "vae_calls": 0,
            "noise_draws": 0,
            "output_bytes_bound": 2 << 30,
        },
        "source_hashes": {name: sha(path) for name, path in source_pairs.items()},
        "launcher_sha256": sha(__file__),
        "resource_exception": {
            "ram_mb": 16384,
            "vram_mb": 10240,
            "cpu_threads": 4,
            "disk_gb": 4,
            "est_wall_s": 300,
            "timeout_s": 900,
            "calibration": "Fresh source Qwen prefix baseline job-ebae1e4bd523 measured 95.23s/RSS4461.6MB/VRAM8790MB with the same 398-recipe source and one CUDA Qwen load. Successor adds one context, three shortcut down projections, and three native suffixes while retaining B1/512 and no FLUX/VAE/noise; 16GiB RAM is a conservative working-set declaration for the 8.04GB source-bound Qwen resident and tokenizer/transient overhead, 10GiB VRAM preserves measured margin. Successor peak remains to be observed.",
            "measurement": baseline,
        },
        "scheduling_contract": one_lease_scheduling_contract(
            model_loads=1,
            logical_branches=3,
            logical_operations=6,
            logical_checkpoint_replays=0,
            max_local_work_items=6,
            extra={
                "task_family": "analysis",
                "contexts": 3,
                "functional_reference_blocks": 60,
                "shortcut_down_projections": 3,
                "native_blocks": 111,
                "native_root_calls": 6,
                "native_full_model_calls": 3,
                "native_suffix_calls": 3,
                "component_hydrations": 1,
                "cuda_placements": 1,
                "cpu_offloads": 1,
                "parameter_exports": 0,
                "flux_calls": 0,
                "vae_calls": 0,
                "noise_draws": 0,
                "logical_operation_scope": "Three in-process raw contexts; complete functional histories and sign-shortcut responses seal before each native full encoder plus suffix reference. Block/primitive counts remain explicit physical censuses, not independent trials.",
            },
            checkpoint_policy="One finite resident CUDA lease; all three tokenized inputs, 60 functional reference blocks, three shortcut down projections, and signed responses seal before any native learned target. No retry, nested submission, FLUX/VAE/noise, or parameter export.",
        ),
    }
    request["request_sha256"] = identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(path, name) for name, path in source_pairs.items() if name != WORKER])
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        require(sha(run / "payload" / name) == digest, "staged source/wheel closure differs: " + name)
    save(run / "staged-command.json", {"command": command(request), "request_sha256": request["request_sha256"], "submitted": False})
    return {"request_sha256": request["request_sha256"], "source_files": len(source_pairs), "recipe_tensors": 398, "checkpoint_bytes_read": 0, "submitted": False}


def inspect_run(run):
    run = Path(run).resolve()
    request = read(run / "request.json")
    require(identity(request) == request.get("request_sha256") and sha(__file__) == request["launcher_sha256"], "request/launcher identity differs")
    for name, digest in {**request["source_hashes"], **request["runtime_wheels"]}.items():
        require(sha(run / "payload" / name) == digest, "payload source/wheel differs: " + name)
    return {"request_sha256": request["request_sha256"], "source_files": len(request["source_hashes"]), "runtime_wheels": request["runtime_wheels"], "counts": read(run / "payload/count-contract.json"), "resources": request["resource_exception"], "submitted": (run / "submission.json").exists(), "finite_review_present": (run / "finite-run-review-v1.json").is_file()}


def submit(run):
    from mdb_lanes.job_support import guarded_mrun_launch

    run = Path(run).resolve()
    inspected, request = inspect_run(run), read(run / "request.json")
    require(not inspected["submitted"], "original submission already exists")
    review = read(run / "finite-run-review-v1.json")
    require(review.get("request_sha256") == request["request_sha256"] and review.get("blocking_issues") == [] and review.get("review_status") == "ready-for-finite-run", "exact finite resource/source review required")
    require(len(QUEUE_NOTE) <= 500, "queue note exceeds scheduler ABI")
    os.environ["MRUN_URL"] = request["scheduler_url"]
    resources = request["resource_exception"]
    args = command(request)
    save(run / "admission-command.json", {"command": args, "request_sha256": request["request_sha256"], "review_sha256": sha(run / "finite-run-review-v1.json")})
    job = guarded_mrun_launch(args, request=request, experiment="mdb-qwen-sign-shortcut", task="analysis", model=None,
        payload=run / "payload", pin="beast", needs=request["needs"], gpu=True,
        **{key: resources[key] for key in ("ram_mb", "vram_mb", "cpu_threads", "disk_gb", "est_wall_s", "timeout_s")},
        retry_on_kill=False, preflight=True, note=QUEUE_NOTE)
    save(run / "submission.json", {"job_id": str(job)})
    return {"job_id": str(job)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--resource-job", type=Path)
    parser.add_argument("--selection", type=Path, default=SELECTION_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.inspect:
        result = inspect_run(args.run)
    elif args.submit:
        result = submit(args.run)
    else:
        if args.parent_run is None or args.resource_job is None:
            parser.error("--parent-run and --resource-job are required to stage")
        result = stage(args.run, args.parent_run, args.resource_job, args.selection)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

"""Stage one finite, source-bound prospective raw-prompt Qwen prefix lease.

This module stages a payload and writes the exact request.  It deliberately
does not submit to mrun; the parent controller performs request-bound review
and the single guarded launch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import runpy
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = "flux_functional_qwen_prefix_worker.py"
HELPERS = (
    "flux_functional_qwen_prefix.py",
    "flux_source_receiver.py",
    "flux_image_tail_microscope.py",
)
PROMPTS = (
    "A yellow cone to the left of a green cylinder on a plain white background. Two separate objects, simple studio product photograph.",
    "A small violet torus above a large copper pyramid on a plain ivory background. Two separate objects, simple studio product photograph.",
)
QUEUE_NOTE = (
    "Prospective raw-prompt Qwen3 prefix:2 tokenizations,2 functional forecasts sealed before"
    "2 full Qwen3ForCausalLM targets,28 states/case;4 native FLUX context_embedder ports."
    "One CUDA B1 lease, all398 recipes streamed into meta model, no RGB/VAE/full FLUX calls,"
    "no retry/nested jobs. RAM16GiB/VRAM13GiB,CPU4;600s estimate/1200s science/1500s outer."
)


def read(path: str | Path):
    return json.loads(Path(path).read_text())


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _runtime_request():
    # The prior validated CUDA cohort supplies only public execution-environment
    # and runtime-wheel metadata.  No experiment result or source state is used.
    return read(Path(__file__).parents[4] / "artifacts/mdb-flux-functional-pass-20261106/run-01/request.json")


def _installed_source_hashes():
    import inspect

    import jinja2.environment
    import jinja2.sandbox
    import tokenizers
    import tokenizers.tokenizers
    from diffusers.models.transformers import transformer_flux2
    from transformers.models.qwen3 import modeling_qwen3

    qwen = Path(inspect.getfile(modeling_qwen3))
    flux = Path(inspect.getfile(transformer_flux2))
    tokenizer_wrapper = Path(inspect.getfile(tokenizers))
    tokenizer_native = Path(inspect.getfile(tokenizers.tokenizers))
    jinja_environment = Path(inspect.getfile(jinja2.environment))
    jinja_sandbox = Path(inspect.getfile(jinja2.sandbox))
    return {
        "qwen_native_source_sha256": sha(qwen),
        "qwen_native_source_path": str(qwen),
        "flux_native_source_sha256": sha(flux),
        "flux_native_source_path": str(flux),
        "tokenizers_wrapper_source_sha256": sha(tokenizer_wrapper),
        "tokenizers_wrapper_source_path": str(tokenizer_wrapper),
        # The preparation host is Darwin while the single CUDA lease runs on
        # Linux.  Do not pin a platform-specific Rust extension from the
        # preparation host.  The worker binds and seals the actual extension
        # loaded on the CUDA host before tokenization/forecast.
        "tokenizers_native_source_sha256": "runtime-bound",
        "tokenizers_native_source_path": "runtime-bound-on-CUDA-host",
        "tokenizers_native_preparation_source_sha256": sha(tokenizer_native),
        "tokenizers_native_preparation_source_path": str(tokenizer_native),
        "jinja_environment_source_sha256": sha(jinja_environment),
        "jinja_environment_source_path": str(jinja_environment),
        "jinja_sandbox_source_sha256": sha(jinja_sandbox),
        "jinja_sandbox_source_path": str(jinja_sandbox),
        "transformers_version": importlib.metadata.version("transformers"),
        "diffusers_version": importlib.metadata.version("diffusers"),
    }


def stage(run: str | Path, planning: str | Path, flux_planning: str | Path) -> dict[str, str | int | bool]:
    from mdb_lanes.job_support import one_lease_scheduling_contract
    from mdb_lanes.payload import stage_worker

    run, planning, flux_planning = (Path(value).resolve() for value in (run, planning, flux_planning))
    run.mkdir(parents=True, exist_ok=False)
    source_binding = read(planning / "prospective-source-binding.json")
    qwen_config = source_binding["source_files"]["text_encoder/config.json"]["json"]
    flux_plan = read(flux_planning / "plan.json")
    prior_request = _runtime_request()
    source_versions = _installed_source_hashes()

    qwen_manifest = planning / "parameter-artifacts.json"
    flux_manifest = flux_planning / "parameter-artifacts.json"
    qwen_retriever = planning / "retrieve_weights.py"
    if not qwen_retriever.exists():
        qwen_retriever = planning.parent / "planning-v1" / "retrieve_weights.py"
    flux_retriever = flux_planning / "retrieve_weights.py"
    flux_config = flux_planning / "transformer-config.json"
    binding_path = planning / "prospective-source-binding.json"
    copies = {
        "qwen-config.json": None,
        "qwen-parameter-artifacts.json": qwen_manifest,
        "qwen-source-binding.json": binding_path,
        "retrieve_qwen_weights.py": qwen_retriever,
        "qwen-recipe-helper.py": HERE / "flux_source_receiver.py",
        "flux-parameter-artifacts.json": flux_manifest,
        "retrieve_flux_weights.py": flux_retriever,
        "flux-transformer-config.json": flux_config,
        "flux-recipe-helper.py": HERE / "flux_source_receiver.py",
    }
    (run / "qwen-config.json").write_text(json.dumps(qwen_config, sort_keys=True, indent=2) + "\n")
    copies["qwen-config.json"] = run / "qwen-config.json"
    for target, source in copies.items():
        assert source is not None
        if target != "qwen-config.json":
            shutil.copyfile(source, run / target)
    shutil.copyfile(planning / "metadata-build-receipt.json", run / "qwen-metadata-build-receipt.json")

    config = {
        "schema": "mdb-qwen-prefix-plan-v1",
        "cohort": "prospective-new",
        "prompts": list(PROMPTS),
        "apply_chat_template": {
            "messages": [{"role": "user", "content": "<raw prompt>"}],
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "padding": "max_length",
        "padding_side": "right",
        "truncation": True,
        "max_length": 512,
        "batch_size": 1,
        "all_512_rows_retained": True,
        "prefix_layers": 27,
        "full_layers": 36,
        "selected_hidden_states": [9, 18, 27],
        "attention_dispatch": {
            "backend": "sdpa",
            "gqa_policy": "repeat_kv",
            "source": "transformers-qwen3-modeling-" + source_versions["qwen_native_source_sha256"],
        },
        "rope_buffer_policy": {
            "init_device": "cpu",
            "init_dtype": "float32",
            "placement_dtype": "float32",
            "source": "transformers-qwen3-rotary-embedding-v1-from-pretrained-device-only",
        },
        "qwen_config": "qwen-config.json",
        "qwen_manifest": "qwen-parameter-artifacts.json",
        "qwen_source_binding": "qwen-source-binding.json",
        "qwen_retrieval_script": "retrieve_qwen_weights.py",
        "qwen_recipe_helper": "qwen-recipe-helper.py",
        "qwen_parameter_root": source_binding["checkpoint"]["text_encoder_root"],
        "flux_config": "flux-transformer-config.json",
        "flux_manifest": "flux-parameter-artifacts.json",
        "flux_retrieval_script": "retrieve_flux_weights.py",
        "flux_recipe_helper": "flux-recipe-helper.py",
        "flux_parameter_root": flux_plan["parameter_root"],
        "device": "cuda",
        "dtype": "bfloat16",
        "retry": False,
        "max_wall_seconds": 1200,
        "transport_limits": {"max_expanded_bytes": 2 << 30, "max_member_bytes": 128 << 20},
        "output_scratch": "/mnt/big/mdb/research",
        "output_name": "qwen-prefix-result",
        "source_metadata": {
            "qwen_metadata_build_receipt_sha256": sha(planning / "metadata-build-receipt.json"),
            "qwen_parameter_manifest_sha256": sha(qwen_manifest),
            "qwen_source_binding_sha256": sha(binding_path),
            "qwen_full_parameter_count": 398,
            "qwen_functional_parameter_count": 298,
            "qwen_reference_only_parameter_count": 100,
            "flux_parameter_manifest_sha256": sha(flux_manifest),
            "flux_context_embedder_tensor": "context_embedder.weight",
            "flux_context_embedder_shape": [3072, 7680],
            "flux_context_embedder_bias": False,
            "historical_chat_template_bound": False,
        },
        **source_versions,
    }
    for key in ("qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script", "qwen_recipe_helper",
                "flux_config", "flux_manifest", "flux_retrieval_script", "flux_recipe_helper"):
        config[key + "_sha256"] = sha(run / config[key])
    # Keep local files relative to the staged payload while retaining remote
    # checkpoint roots only as source-bound immutable locations.
    local_config = dict(config)
    for key in ("qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script", "qwen_recipe_helper",
                "flux_config", "flux_manifest", "flux_retrieval_script", "flux_recipe_helper"):
        local_config[key] = str(run / config[key])
    worker = runpy.run_path(str(HERE / WORKER))
    declared = worker["validate_config"](local_config)
    save(run / "plan.json", config)
    save(run / "count-contract.json", declared)
    preflight = {
        "schema": "mrun-preflight-config-v1",
        "hashed_roots": [WORKER, *HELPERS, *copies.keys(), "qwen-metadata-build-receipt.json", "plan.json", "count-contract.json"],
        "path_token_allowlist": [config["qwen_parameter_root"], config["flux_parameter_root"], config["output_scratch"], config["output_name"]],
        "no_source_payload_exports": True,
        "no_native_calls_during_forecast": True,
    }
    save(run / "preflight.v1.json", preflight)
    environment = prior_request["execution_environment"]
    request = {
        "schema": "mdb-qwen-prefix-request-v1",
        "config": config,
        "execution_environment": environment,
        "scheduler_url": prior_request["scheduler_url"],
        "model": "flux2-klein-4b-distilled",
        "task_family": "analysis",
        "backend": "torch-native-qwen-prefix",
        "device": "cuda",
        "dtype": "bfloat16",
        "flux_mode": "capture",
        "capture_sites": True,
        "height": 512,
        "width": 512,
        "steps": 1,
        "workload_geometry": {
            "model_loads": 1,
            "peak_physical_batch": 1,
            "qwen_native_encoder_calls": 2,
            "qwen_full_layers_per_call": 36,
            "functional_forecast_calls": 2,
            "context_embedder_native_calls": 4,
            "physical_module_calls": 6,
            "retained_hidden_states": 56,
            "output_bytes_bound": 2 << 30,
        },
        "needs": {"cuda": True, "host": "beast"},
        "runtime_wheels": prior_request["runtime_wheels"],
        "resource_exception": {
            "ram_mb": 16384,
            "vram_mb": 13000,
            "cpu_threads": 4,
            "disk_gb": 8,
            "est_wall_s": 600,
            "timeout_s": 1500,
            "calibration": "No original execution in preparation. RAM16GiB is a conservative working-set bound for a meta-constructed 8.04GB BF16 Qwen resident plus CUDA/runtime/transient logits and tokenizer; VRAM13GiB is a declared measured-envelope candidate to be admitted/observed by mrun, with no no-clamp or resident-worker override.",
        },
        "source_hashes": {},
        "launcher_sha256": sha(__file__),
        "scheduling_contract": one_lease_scheduling_contract(
            model_loads=1,
            logical_branches=2,
            logical_operations=8,
            max_local_work_items=16,
            extra={"physical_calls": 6, "full_qwen_layers": 72, "functional_hidden_states": 56, "context_component_calls": 4},
            checkpoint_policy="One finite resident CUDA lease; both tokenized forecasts and all 28 states/case sealed before two full native Qwen3ForCausalLM calls and four native context_embedder consumers. No retry or nested submission.",
        ),
    }
    source_pairs = [
        (WORKER, HERE / WORKER),
        *[(name, HERE / name) for name in HELPERS],
        *[(target, run / target) for target in copies],
        ("qwen-metadata-build-receipt.json", run / "qwen-metadata-build-receipt.json"),
        ("plan.json", run / "plan.json"),
        ("count-contract.json", run / "count-contract.json"),
        ("preflight.v1.json", run / "preflight.v1.json"),
    ]
    request["source_hashes"] = {target: sha(source) for target, source in source_pairs}
    request["request_sha256"] = request_identity(request)
    save(run / "request.json", request)
    stage_worker(run / "payload", worker=HERE / WORKER, config=config, request=request,
                 additional_files=[(HERE / name, name) for name in HELPERS]
                 + [(run / target, target) for target in copies]
                 + [(run / "qwen-metadata-build-receipt.json", "qwen-metadata-build-receipt.json")]
                 + [(run / "plan.json", "plan.json"), (run / "count-contract.json", "count-contract.json"), (run / "preflight.v1.json", "preflight.v1.json")])
    for target, digest in request["source_hashes"].items():
        staged = run / "payload" / target
        require(staged.is_file() and sha(staged) == digest, "staged source differs: " + target)
    save(run / "staged-command.json", {"request_sha256": request["request_sha256"], "worker": WORKER, "one_lease": True, "submitted": False})
    return {"request_sha256": request["request_sha256"], "source_files": len(request["source_hashes"]), "submitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--planning", type=Path, required=True)
    parser.add_argument("--flux-planning", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.run, args.planning, args.flux_planning), sort_keys=True))

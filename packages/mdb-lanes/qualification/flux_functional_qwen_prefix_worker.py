"""Prospective raw-prompt Qwen3 prefix forecast and native comparison.

This worker intentionally has one narrow scientific boundary: it forms two
tokenized raw-prompt conditioning prefixes, seals the complete functional
forecast, then calls the original Qwen3ForCausalLM twice and retains the
corresponding first 28 hidden states.  A native FLUX ``context_embedder``
component consumes both predicted and reference embeddings only after the
encoder targets exist.  The component result is a conditioning diagnostic,
not a forecast of RGB or a full FLUX/session execution.

The Qwen weights are loaded one recipe at a time from the public immutable
RecipeBank into a meta-constructed model already placed on CUDA.  This avoids
an unnecessary CPU copy of the 8.04 GB checkpoint while preserving all 398
named recipes, including the nine reference-only tail blocks and final norm.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import runpy
import shutil
import time
from pathlib import Path
from typing import Any

PROMPTS = (
    "A yellow cone to the left of a green cylinder on a plain white background. Two separate objects, simple studio product photograph.",
    "A small violet torus above a large copper pyramid on a plain ivory background. Two separate objects, simple studio product photograph.",
)
SELECTED_HIDDEN_STATES = (9, 18, 27)
PREFIX_LAYERS = 27
FULL_LAYERS = 36
QWEN_PARAMETER_COUNT = 398
FUNCTIONAL_PARAMETER_COUNT = 298
REFERENCE_ONLY_PARAMETER_COUNT = 100
CONTEXT_EMBEDDER_SHAPE = (3072, 7680)
EXECUTION_ID = "mdb-flux-functional-qwen-prefix-v1"
ROPE_POLICY = {
    "init_device": "cpu",
    "init_dtype": "float32",
    "placement_dtype": "float32",
    "source": "transformers-qwen3-rotary-embedding-v1-from-pretrained-device-only",
}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def helper() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).with_name("flux_functional_qwen_prefix.py")))


def _profile(config: dict[str, Any]) -> dict[str, Any]:
    qwen = helper()
    return qwen["_native_profile"](read(config["qwen_config"]), PREFIX_LAYERS)


def contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "mdb-qwen-prefix-count-contract-v1",
        "cases": 2,
        "forecast": {
            "tokenizer_calls": 2,
            "functional_prefix_calls": 2,
            "functional_module_forwards": 0,
            "embedding_states": 2,
            "prefix_hidden_states": 54,
            "selected_hidden_states": 6,
            "conditioning_ports": 2,
            "context_embedder_forecast_linear_calls": 2,
            "native_encoder_calls": 0,
        },
        "native": {
            "native_encoder_calls": 2,
            "full_native_layers_per_call": FULL_LAYERS,
            "retained_hidden_states": 56,
            "retained_embedding_ports": 2,
            "tail_layers_executed_but_not_retained": 18,
            "native_module_forwards": 2,
        },
        "consumers": {
            "context_embedder_native_calls": 4,
            "predicted_embedding_calls": 2,
            "reference_embedding_calls": 2,
            "full_flux_transformer_calls": 0,
            "vae_calls": 0,
            "rgb_forecasts": 0,
        },
        "model_loads": 1,
        "encodes": 2,
        "noise_draws": 0,
        "vae_calls": 0,
        "execution_id": EXECUTION_ID,
        "predictions": [
            "The functional prefix arithmetic and selected conditioning are sealed before any native encoder comparison.",
            "Native Qwen3ForCausalLM hidden states at all retained prefix indices are measured against the forecast; every miss remains evidence.",
            "The original FLUX context_embedder receives predicted and reference E after target entry; this is a component-port diagnostic, not an RGB prediction.",
        ],
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    require(config.get("schema") == "mdb-qwen-prefix-plan-v1", "unexpected Qwen prefix plan schema")
    require(tuple(config.get("prompts", ())) == PROMPTS, "raw-prompt cohort differs")
    require(config.get("cohort") == "prospective-new", "historical cohort cannot be repaired")
    require(config.get("device") == "cuda" and config.get("dtype") == "bfloat16", "CUDA BF16 is required")
    require(config.get("batch_size") == 1 and config.get("max_length") == 512, "B1 512-token contract differs")
    require(config.get("padding") == "max_length" and config.get("padding_side") == "right", "right max padding required")
    require(config.get("truncation") is True and config.get("all_512_rows_retained") is True, "token row contract differs")
    require(config.get("apply_chat_template") == {
        "add_generation_prompt": True,
        "enable_thinking": False,
        "messages": [{"role": "user", "content": "<raw prompt>"}],
    }, "chat-template contract differs")
    require(config.get("prefix_layers") == PREFIX_LAYERS and config.get("full_layers") == FULL_LAYERS, "Qwen depth differs")
    require(tuple(config.get("selected_hidden_states", ())) == SELECTED_HIDDEN_STATES, "selected state indices differ")
    for key in ("qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script", "flux_config", "flux_manifest", "flux_retrieval_script"):
        require(key in config, "missing source binding: " + key)
        require(sha(config[key]) == config[key + "_sha256"], "source metadata changed: " + key)
    qwen_config = read(config["qwen_config"])
    require(qwen_config["architectures"] == ["Qwen3ForCausalLM"], "original Qwen architecture required")
    require(qwen_config["hidden_size"] == 2560 and qwen_config["num_attention_heads"] == 32
            and qwen_config["num_key_value_heads"] == 8 and qwen_config["head_dim"] == 128
            and qwen_config["intermediate_size"] == 9728 and qwen_config["num_hidden_layers"] == 36,
            "original Qwen geometry differs")
    require(qwen_config["attention_bias"] is False and qwen_config["attention_dropout"] == 0.0,
            "Qwen bias/dropout policy differs")
    require(qwen_config["tie_word_embeddings"] is True, "tied Qwen embedding/lm_head required")
    profile = _profile(config)
    require(profile["attention_width"] == 4096 and profile["hidden_size"] == 2560, "unequal Qwen widths not bound")
    require(config.get("attention_dispatch", {}).get("backend") in {"eager", "sdpa"}, "attention backend not bound")
    require(config["attention_dispatch"].get("gqa_policy") in {"repeat_kv", "enable_gqa"}, "GQA policy not bound")
    require(config["attention_dispatch"].get("source"), "attention dispatch source not bound")
    require(config.get("rope_buffer_policy") == ROPE_POLICY, "RoPE construction/placement policy differs")
    qwen_manifest = read(config["qwen_manifest"])
    names = {record["source"]["tensor"] for record in qwen_manifest["artifacts"]}
    require(len(names) == QWEN_PARAMETER_COUNT and len(qwen_manifest["artifacts"]) == QWEN_PARAMETER_COUNT,
            "full 398 Qwen recipe manifest required")
    functional = {"model.embed_tokens.weight"} | {
        f"model.layers.{layer}.{suffix}"
        for layer in range(PREFIX_LAYERS)
        for suffix in helper()["QWEN_LAYER_PARAMETER_SUFFIXES"]
    }
    require(len(names & functional) == FUNCTIONAL_PARAMETER_COUNT, "functional Qwen recipe subset differs")
    require("model.norm.weight" in names and all(f"model.layers.{i}.input_layernorm.weight" in names for i in range(27, 36)),
            "reference-only tail/final norm recipes missing")
    flux_manifest = read(config["flux_manifest"])
    context = [record for record in flux_manifest["artifacts"] if record["source"]["tensor"] == "context_embedder.weight"]
    require(len(context) == 1 and tuple(context[0]["source"]["shape"]) == CONTEXT_EMBEDDER_SHAPE
            and context[0]["source"]["dtype"] == "BF16", "original context_embedder recipe differs")
    require(not any(record["source"]["tensor"] == "context_embedder.bias" for record in flux_manifest["artifacts"]),
            "context_embedder bias must remain absent as in native FLUX")
    require(0 < config.get("max_wall_seconds", 1200) <= 1800 and config.get("retry", False) is False,
            "finite no-retry budget differs")
    return contract(config)


def _npy_bytes(array) -> bytes:
    import numpy as np

    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _save_bits(output: Path, relative: str, value, *, kind: str, case: str, index: int | None = None) -> dict[str, Any]:
    import numpy as np
    import torch

    require(isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16, f"{kind} must be BF16")
    contiguous = value.detach().contiguous()
    bits = contiguous.view(torch.uint16).cpu().numpy()
    raw = _npy_bytes(bits)
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "content_sha256": hashlib.sha256(bits.tobytes(order="C")).hexdigest(),
        "shape": list(value.shape),
        "stored_dtype": str(np.asarray(bits).dtype),
        "logical_dtype": "torch.bfloat16",
        "kind": kind,
        "case": case,
        "index": index,
        "all_rows_retained": int(value.shape[-2] if value.ndim >= 2 else value.shape[0]) == 512,
    }


def _tokenize(tokenizer, prompt: str, config: dict[str, Any]):
    import torch

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if input_ids.ndim == 1:
        input_ids, attention_mask = input_ids.unsqueeze(0), attention_mask.unsqueeze(0)
    require(tuple(input_ids.shape) == (1, 512) and tuple(attention_mask.shape) == (1, 512), "tokenizer did not retain 512 rows")
    require(tokenizer.padding_side == "right", "tokenizer padding side changed")
    require(input_ids.dtype == torch.int64 and attention_mask.dtype in (torch.int64, torch.int32), "tokenizer dtypes differ")
    return formatted, input_ids, attention_mask


def _source_files(binding: dict[str, Any]) -> list[dict[str, Any]]:
    return list(binding["source_files"].values())


def _copy_tokenizer_sources(output: Path, binding: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for relative, row in binding["source_files"].items():
        if not relative.startswith("tokenizer/") and relative not in {"text_encoder/config.json", "text_encoder/generation_config.json"}:
            continue
        source = Path(row["path"])
        require(source.is_file() and source.stat().st_size == row["bytes"] and sha(source) == row["sha256"], "bound tokenizer source changed: " + relative)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha(target) == row["sha256"] and target.stat().st_size == row["bytes"], "tokenizer custody copy differs")
        records[relative] = {"path": str(source), "retained": str(target.relative_to(output)), "sha256": row["sha256"], "bytes": row["bytes"]}
    return records


def _load_recipe_bank(module_path: Path, manifest_path: Path, parameter_root: str):
    module = runpy.run_path(str(module_path))
    manifest = read(manifest_path)
    return module["RecipeBank"](manifest, parameter_root, max_cache_bytes=256 << 20), manifest


def _recipe(record: dict[str, Any], name: str) -> dict[str, Any]:
    source = record["source"]
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "parameter_tensor": name,
        "stored_shape": source["shape"],
        "descriptor": {"sha256": source["payload_sha256"]},
    }


def _load_qwen(config: dict[str, Any], device, bank, manifest):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    qconfig = Qwen3Config.from_dict(read(config["qwen_config"]))
    qconfig._attn_implementation = config["attention_dispatch"]["backend"]
    with torch.device("meta"):
        model = Qwen3ForCausalLM(qconfig)
    # ``Qwen3Config.dtype`` does not force parameters created on the meta
    # device to BF16 in the installed Transformers build.  Set the dtype on
    # the meta graph before allocating CUDA storage so every original recipe
    # is assigned into an already-BF16 parameter, with no FP32 CPU duplicate.
    model = model.to(dtype=torch.bfloat16)
    model = model.to_empty(device=device)
    model.tie_weights()
    records = {row["source"]["tensor"]: row for row in manifest["artifacts"]}
    with torch.inference_mode():
        for name, record in records.items():
            raw = bank.load(_recipe(record, name))
            tensor = torch.from_numpy(raw).to(device=device, dtype=torch.bfloat16)
            target = model.get_parameter(name)
            require(tuple(target.shape) == tuple(tensor.shape), "Qwen parameter shape differs: " + name)
            target.copy_(tensor)
            del tensor, raw
        # tie_word_embeddings makes lm_head an alias of embed_tokens; bind and
        # verify that alias explicitly because lm_head is intentionally absent
        # from the 398 encoder recipe manifest.
        model.lm_head.weight = model.model.embed_tokens.weight
    model.eval().requires_grad_(False)
    _bind_native_rope_buffers(model, qconfig, device, placement_dtype=ROPE_POLICY["placement_dtype"])
    return model


def _bind_native_rope_buffers(model, config, device, *, placement_dtype: str = "bfloat16") -> None:
    import torch

    # Instantiate the installed native class on CPU exactly as the source
    # does, then apply the declared placement.  The production profile uses
    # device-only movement, matching from_pretrained(dtype=BF16).to(cuda):
    # nonpersistent inv_freq/original_inv_freq remain CPU-constructor FP32.
    # The explicit BF16 option remains available for the v1 tiny profile.
    require(placement_dtype in {"float32", "bfloat16"}, "unsupported native RoPE placement dtype")
    rotary_type = type(model.model.rotary_emb)
    native_rotary = rotary_type(config=config, device=torch.device("cpu"))
    if placement_dtype == "bfloat16":
        native_rotary = native_rotary.to(device=device, dtype=torch.bfloat16)
    else:
        native_rotary = native_rotary.to(device=device)
    model.model.rotary_emb = native_rotary


def _load_context_component(config: dict[str, Any], device, bank, manifest):
    import torch
    from diffusers import Flux2Transformer2DModel

    record = next(row for row in manifest["artifacts"] if row["source"]["tensor"] == "context_embedder.weight")
    weight = torch.from_numpy(bank.load(_recipe(record, "context_embedder.weight"))).to(device=device, dtype=torch.bfloat16)
    with torch.device("meta"):
        flux = Flux2Transformer2DModel.from_config(read(config["flux_config"]))
    component = flux.context_embedder.to_empty(device=device)
    component.load_state_dict({"weight": weight}, strict=True, assign=True)
    component.eval()
    require(tuple(component.weight.shape) == CONTEXT_EMBEDDER_SHAPE and component.bias is None, "native context component differs")
    return flux, component, weight


def _model_identity(model) -> dict[str, str]:
    import torch

    result = {}
    for name, value in model.named_parameters():
        raw = value.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes(order="C")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def _tensor_identity(value) -> str:
    import torch

    raw = value.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _runtime_contract(config: dict[str, Any], torch) -> dict[str, Any]:
    import importlib.metadata
    import inspect

    import jinja2.environment
    import jinja2.sandbox
    import tokenizers
    import tokenizers.tokenizers
    from diffusers.models.transformers import transformer_flux2
    from transformers.models.qwen3 import modeling_qwen3

    modules = {
        "qwen_native_source_sha256": modeling_qwen3,
        "flux_native_source_sha256": transformer_flux2,
        "tokenizers_wrapper_source_sha256": tokenizers,
        "tokenizers_native_source_sha256": tokenizers.tokenizers,
        "jinja_environment_source_sha256": jinja2.environment,
        "jinja_sandbox_source_sha256": jinja2.sandbox,
    }
    sources = {}
    for key, module in modules.items():
        path = Path(inspect.getfile(module))
        digest = sha(path)
        expected = config[key]
        runtime_bound = expected == "runtime-bound"
        if not runtime_bound:
            require(digest == expected, "installed implementation source differs: " + key)
        sources[key] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "binding": "runtime-observed-on-CUDA-host" if runtime_bound else "staged-source-hash",
            "expected_sha256": expected,
        }
    flags = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "num_threads": int(torch.get_num_threads()),
    }
    return {
        "schema": "mdb-qwen-prefix-runtime-contract-v1",
        "implementation_sources": sources,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "tokenizers", "jinja2", "diffusers", "safetensors")
        },
        "numeric_flags_before_forecast": flags,
        "attention_dispatch_source": config["attention_dispatch"],
        "rope_buffer_policy": config["rope_buffer_policy"],
        "native_kernel_selection": "not independently profiled; runtime uses the bound Transformers SDPA dispatch",
    }


def run(config: dict[str, Any], *, progress=None) -> dict[str, Any]:
    declared = validate_config(config)
    import numpy as np
    import torch
    from torch.nn import functional as F

    from mdb_lanes.forecast_transaction import ForecastPayload, ForecastTransaction, verify_transaction

    output = Path(config["output"]).resolve()
    require(not output.exists(), "output must be a new directory")
    output.mkdir(parents=True)
    request_record = config.get("_request_record")
    effective_plan = {key: value for key, value in config.items() if key != "_request_record"}
    effective_plan["request_sha256"] = config.get("request_sha256")
    effective_plan["job_id"] = os.environ.get("MRUN_JOB_ID")
    effective_plan["effective_binding"] = {
        "request_sha256": config.get("request_sha256"),
        "job_id": os.environ.get("MRUN_JOB_ID"),
    }
    save(output / "plan.json", effective_plan)
    if request_record is not None:
        save(output / "request.json", request_record)
    start = time.monotonic()
    event_rows: list[dict[str, Any]] = []

    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime_contract = _runtime_contract(config, torch)
    save(output / "runtime-contract.json", runtime_contract)
    runtime_contract_sha256 = sha(output / "runtime-contract.json")

    def event(kind: str, **detail: Any) -> None:
        row = {"kind": kind, "seconds": time.monotonic() - start, **detail}
        require(row["seconds"] <= config["max_wall_seconds"], "finite worker wall budget exceeded")
        event_rows.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress:
            progress(row)

    require(torch.cuda.is_available(), "CUDA is required for the original Qwen cohort")
    device = torch.device("cuda")
    qwen_binding = read(config["qwen_source_binding"])
    source_copy_keys = (
        "qwen_config", "qwen_manifest", "qwen_source_binding", "qwen_retrieval_script",
        "flux_config", "flux_manifest", "flux_retrieval_script",
    )
    retained_metadata = {}
    for key in source_copy_keys:
        source = Path(config[key])
        target = output / "source" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        retained_metadata[key] = {"path": str(target.relative_to(output)), "sha256": sha(target), "bytes": target.stat().st_size}
    tokenizer_records = _copy_tokenizer_sources(output, qwen_binding)
    tokenizer_root = qwen_binding["checkpoint"]["model_root"] + "/tokenizer"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, padding_side="right")
    tokenizer_metadata = read(output / "source" / "tokenizer" / "tokenizer_config.json")
    metadata_tokenizer_class = tokenizer_metadata.get("tokenizer_class")
    # Transformers 5.14.1 exposes this fast tokenizer through the public
    # ``Qwen2Tokenizer`` class name while ``is_fast`` identifies the actual
    # Rust-backed implementation.  Bind the runtime class to the copied
    # tokenizer metadata, while retaining the source binding's canonical
    # fast-class spelling as a separate prospective recipe field.
    require(bool(getattr(tokenizer, "is_fast", False)), "pinned tokenizer is not fast")
    require(type(tokenizer).__name__ in {"Qwen2Tokenizer", "Qwen2TokenizerFast"}, "pinned tokenizer implementation differs")
    require(type(tokenizer).__name__ == metadata_tokenizer_class, "runtime tokenizer class differs from copied tokenizer metadata")
    declared_tokenizer_class = qwen_binding["tokenizer_runtime"]["tokenizer_class"]
    require(declared_tokenizer_class == "Qwen2TokenizerFast", "prospective tokenizer binding differs")
    event("tokenizer_loaded", tokenizer_class=type(tokenizer).__name__, metadata_tokenizer_class=metadata_tokenizer_class,
          declared_tokenizer_class=declared_tokenizer_class, is_fast=bool(tokenizer.is_fast), retained_sources=len(tokenizer_records))
    bank, qwen_manifest = _load_recipe_bank(Path(config["qwen_recipe_helper"]), Path(config["qwen_manifest"]), config["qwen_parameter_root"])
    model = _load_qwen(config, device, bank, qwen_manifest)
    model_identity_before = _model_identity(model)
    event("qwen_model_hydrated", model_loads=1, recipes=len(qwen_manifest["artifacts"]), payload_reads=bank.payload_reads)
    flux_bank, flux_manifest = _load_recipe_bank(Path(config["flux_recipe_helper"]), Path(config["flux_manifest"]), config["flux_parameter_root"])
    flux_meta, context_component, context_weight = _load_context_component(config, device, flux_bank, flux_manifest)
    context_weight_identity_before = _tensor_identity(context_weight)
    event("context_component_hydrated", native_class=type(context_component).__name__, shape=list(context_weight.shape))

    qwen = helper()
    cases: dict[str, dict[str, Any]] = {}
    forecast_payloads: dict[str, np.ndarray] = {}
    for case_index, prompt in enumerate(PROMPTS):
        case = f"prompt-{case_index:02d}"
        formatted, ids_cpu, mask_cpu = _tokenize(tokenizer, prompt, config)
        ids, mask = ids_cpu.to(device), mask_cpu.to(device)
        token_binding = {
            "cohort": "prospective-new",
            "parser_stage": "tokenizer",
            "implementation_pinned": True,
            "declared_tokenizer_class": declared_tokenizer_class,
            "tokenizer_class": type(tokenizer).__name__,
            "metadata_tokenizer_class": metadata_tokenizer_class,
            "tokenizer_is_fast": bool(tokenizer.is_fast),
            "template_files": {key: value["sha256"] for key, value in tokenizer_records.items() if key.startswith("tokenizer/")},
            "chat_template_call": config["apply_chat_template"],
        }
        bound = qwen["bind_prospective_tokenized_input"](ids, mask, tokenizer_binding=token_binding)
        parameters = {"model.embed_tokens.weight": model.model.embed_tokens.weight}
        for layer in range(PREFIX_LAYERS):
            parameters.update({
                f"model.layers.{layer}.{suffix}": model.state_dict()[f"model.layers.{layer}.{suffix}"]
                for suffix in qwen["QWEN_LAYER_PARAMETER_SUFFIXES"]
            })
        with qwen["module_entry_guard"](model) as entries:
            forecast = qwen["functional_qwen_prefix"](
                bound["input_ids"], bound["attention_mask"], parameters, model.config,
                attention_dispatch=config["attention_dispatch"],
                rope_buffer_policy=config["rope_buffer_policy"],
                prefix_layers=PREFIX_LAYERS, selected_hidden_states=SELECTED_HIDDEN_STATES,
            )
        require(entries == [], "functional forecast entered a native module")
        E = forecast["conditioning"]
        predicted_context = F.linear(E, context_weight)
        arrays = []
        for index in range(PREFIX_LAYERS + 1):
            arrays.append(_save_bits(output, f"arrays/{case}/forecast-hidden-{index:02d}.npy", forecast["hidden_states"][index], kind="forecast-hidden-state", case=case, index=index))
        e_record = _save_bits(output, f"arrays/{case}/forecast-conditioning.npy", E, kind="forecast-conditioning", case=case)
        context_record = _save_bits(output, f"arrays/{case}/forecast-context-embedder.npy", predicted_context, kind="forecast-context-embedder", case=case)
        ids_raw, mask_raw = ids_cpu.numpy().astype(np.int64), mask_cpu.numpy().astype(np.int64)
        cases[case] = {
            "case": case,
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt": prompt,
            "formatted_chat": formatted,
            "formatted_chat_sha256": hashlib.sha256(formatted.encode("utf-8")).hexdigest(),
            "input_ids": ids_raw.tolist(),
            "attention_mask": mask_raw.tolist(),
            "input_ids_sha256": hashlib.sha256(ids_raw.tobytes()).hexdigest(),
            "attention_mask_sha256": hashlib.sha256(mask_raw.tobytes()).hexdigest(),
            "active_rows": int(mask_raw.sum()),
            "padded_rows": int(mask_raw.size - mask_raw.sum()),
            "forecast_hidden": arrays,
            "forecast_conditioning": e_record,
            "forecast_context_embedder": context_record,
            "profile": forecast["profile"],
            "position_ids": forecast["position_ids"].detach().cpu().tolist(),
            "native_module_forwards_during_forecast": 0,
        }
        forecast_payloads[case] = np.frombuffer(E.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes(), dtype=np.uint16).reshape(tuple(E.shape))
        event("forecast_case_complete", case=case, active_rows=cases[case]["active_rows"], padded_rows=cases[case]["padded_rows"], native_forwards=0)
        del forecast, E, predicted_context, ids, mask

    save(output / "forecast-cases.json", cases)
    source_inventory = {
        "runtime-contract.json": {
            "sha256": runtime_contract_sha256,
            "bytes": (output / "runtime-contract.json").stat().st_size,
        }
    }
    for path in sorted((output / "source").rglob("*")):
        if path.is_file():
            source_inventory[str(path.relative_to(output))] = {"sha256": sha(path), "bytes": path.stat().st_size}
    tx = ForecastTransaction(
        output / "forecast-transaction", model_identity="qwen3-flux2-klein-4b-original",
        input_identity=hashlib.sha256(canonical({case: {"input_ids_sha256": row["input_ids_sha256"], "attention_mask_sha256": row["attention_mask_sha256"]} for case, row in cases.items()})).hexdigest(),
        source_identity=hashlib.sha256(canonical({"qwen_manifest": sha(config["qwen_manifest"]), "flux_manifest": sha(config["flux_manifest"])})).hexdigest(),
        frontier={"kind": "raw-input", "parser": "pinned Qwen chat-template tokenizer", "source_inventory": source_inventory},
        execution_scope="External CUDA Qwen3ForCausalLM and native FLUX context_embedder calls after global seal; transaction is metadata/array custody.",
        budgets={"max_predictions": 1, "max_executions": 1, "max_native_calls": 8, "max_bytes": 512 << 20, "max_files": 4096, "max_wall_ms": config["max_wall_seconds"] * 1000},
    )
    tx.frontier("tokenized-inputs", {case: {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"], "prompt_utf8_sha256": row["prompt_utf8_sha256"]} for case, row in cases.items()})
    tx.predict(
        "qwen-prefix-forecast",
        lambda _context: {
            "predictions": {
                "cases": cases,
                "global_forecast": "all 2 prompts, 28 hidden states and context ports sealed before native encoder targets",
                "runtime_contract_sha256": runtime_contract_sha256,
            },
                # The transaction stores native uint16 BF16 bit arrays without
                # a numeric cast.  Its public cast vocabulary intentionally
                # excludes uint16; the logical BF16 interpretation is carried
                # in provenance and the persisted array descriptor.
                "payloads": {f"{case}.conditioning": ForecastPayload(value=forecast_payloads[case], dtype=None, provenance={"logical_dtype": "bfloat16", "shape": list(forecast_payloads[case].shape)}) for case in cases},
        },
    )
    seal = tx.seal()
    event("forecast_globally_sealed", seal_sha256=seal["seal_sha256"], native_encoder_calls=0)

    native_records: dict[str, dict[str, Any]] = {}
    consumer_records: dict[str, dict[str, Any]] = {}
    consumer_operands: list[dict[str, Any]] = []
    native_release = False

    def qwen_native_entry(_module, _args):
        require(native_release, "native Qwen encoder entered before global forecast release")

    qwen_entry_handle = model.register_forward_pre_hook(qwen_native_entry, prepend=True)

    def context_operand(_module, args):
        import torch

        value = args[0]
        require(native_release, "native context component entered before global forecast release")
        require(isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16, "context operand must be BF16")
        raw = value.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes(order="C")
        consumer_operands.append({
            "ordinal": len(consumer_operands),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "consumer": "native context_embedder prehook operand",
        })

    operand_handle = context_component.register_forward_pre_hook(context_operand, prepend=True)

    def native_targets_and_consumers(_context):
        import torch

        nonlocal native_release
        native_release = True
        for case_index, prompt in enumerate(PROMPTS):
            case = f"prompt-{case_index:02d}"
            row = cases[case]
            ids = torch.tensor(row["input_ids"], dtype=torch.int64, device=device)
            mask = torch.tensor(row["attention_mask"], dtype=torch.int64, device=device)
            with torch.inference_mode():
                native_output = model(
                    ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True
                )
            require(len(native_output.hidden_states) >= FULL_LAYERS + 1, "native full Qwen hidden-state suffix did not execute")
            retained = []
            for index in range(PREFIX_LAYERS + 1):
                retained.append(_save_bits(output, f"arrays/{case}/native-hidden-{index:02d}.npy", native_output.hidden_states[index], kind="native-hidden-state", case=case, index=index))
            native_e = torch.cat([native_output.hidden_states[index] for index in SELECTED_HIDDEN_STATES], dim=-1)
            native_e_record = _save_bits(output, f"arrays/{case}/native-conditioning.npy", native_e, kind="native-conditioning", case=case)
            with torch.inference_mode():
                predicted_port = context_component(torch.from_numpy(forecast_payloads[case].copy()).view(torch.uint16).view(torch.bfloat16).to(device=device))
                reference_port = context_component(native_e)
            predicted_record = _save_bits(output, f"arrays/{case}/native-context-from-predicted.npy", predicted_port, kind="native-context-from-predicted", case=case)
            reference_record = _save_bits(output, f"arrays/{case}/native-context-from-reference.npy", reference_port, kind="native-context-from-reference", case=case)
            native_records[case] = {
                "case": case,
                "retained_hidden": retained,
                "native_conditioning": native_e_record,
                "full_native_hidden_state_count": len(native_output.hidden_states),
                "tail_layers_executed": FULL_LAYERS - PREFIX_LAYERS,
                "output_hidden_states": True,
                "use_cache": False,
                "attention_backend": config["attention_dispatch"],
            }
            consumer_records[case] = {
                "predicted_input": cases[case]["forecast_conditioning"],
                "reference_input": native_e_record,
                "predicted_output": predicted_record,
                "reference_output": reference_record,
                "native_component": "Flux2Transformer2DModel.context_embedder (nn.Linear, bias=False)",
                "full_flux_transformer_calls": 0,
            }
            event("native_case_complete", case=case, retained_hidden_states=len(retained), full_native_hidden_state_count=len(native_output.hidden_states), context_calls=2)
            del native_output, native_e, predicted_port, reference_port, ids, mask
        native_release = False
        return {"native": native_records, "consumers": consumer_records}

    tx.execute("native-targets-and-context-consumers", native_targets_and_consumers)
    native_release = False
    qwen_entry_handle.remove()
    tx.verify()
    operand_handle.remove()
    require(len(consumer_operands) == 4, "native context operand census differs")
    tx.close()
    save(output / "native-cases.json", native_records)
    save(output / "context-consumers.json", consumer_records)
    save(output / "context-consumer-operands.json", consumer_operands)
    save(output / "source-inventory.json", source_inventory)
    model_identity_after = _model_identity(model)
    require(model_identity_after == model_identity_before, "original Qwen parameters changed during native targets")
    context_weight_identity_after = _tensor_identity(context_weight)
    require(context_weight_identity_after == context_weight_identity_before, "native context_embedder parameters changed")
    save(output / "source.json", {
        "schema": "mdb-qwen-prefix-source-v1",
        "job_id": os.environ.get("MRUN_JOB_ID"),
        "request_sha256": config.get("request_sha256"),
        "worker_sha256": sha(__file__),
        "helper_source_sha256": {
            "flux_functional_qwen_prefix.py": sha(Path(__file__).with_name("flux_functional_qwen_prefix.py")),
            "flux_source_receiver.py": sha(Path(__file__).with_name("flux_source_receiver.py")),
            "flux_image_tail_microscope.py": sha(Path(__file__).with_name("flux_image_tail_microscope.py")),
        },
        "retained_metadata": retained_metadata,
        "tokenizer_sources": tokenizer_records,
        "runtime_contract_sha256": runtime_contract_sha256,
        "qwen_parameter_manifest_sha256": sha(config["qwen_manifest"]),
        "flux_parameter_manifest_sha256": sha(config["flux_manifest"]),
        "qwen_parameter_identity_before": model_identity_before,
        "qwen_parameter_identity_after": model_identity_after,
        "context_embedder_weight_identity_before": context_weight_identity_before,
        "context_embedder_weight_identity_after": context_weight_identity_after,
        "effective_plan_sha256": sha(output / "plan.json"),
        "request_record_sha256": sha(output / "request.json") if request_record is not None else None,
        "no_payload_exports": True,
        "no_rgb_forecast": True,
    })
    save(output / "native-contract.json", {
        "schema": "mdb-qwen-prefix-native-contract-v1",
        "model_class": type(model).__name__,
        "qwen_architecture": "Qwen3ForCausalLM",
        "native_encoder_calls": 2,
        "context_embedder_native_calls": 4,
        "full_native_layers": FULL_LAYERS,
        "retained_states_per_case": PREFIX_LAYERS + 1,
        "context_embedder_shape": list(CONTEXT_EMBEDDER_SHAPE),
        "context_embedder_bias": False,
        "context_embedder_weight_identity_before": context_weight_identity_before,
        "context_embedder_weight_identity_after": context_weight_identity_after,
        "rope_buffer_policy": ROPE_POLICY,
        "attention_dispatch": config["attention_dispatch"],
        "forecast_native_module_forwards": 0,
        "no_rgb_forecast": True,
    })
    summary = {
        "schema": "mdb-qwen-prefix-summary-v1",
        "execution_id": EXECUTION_ID,
        "request_sha256": config.get("request_sha256"),
        "job_id": os.environ.get("MRUN_JOB_ID"),
        "counts": declared,
        "cases": list(cases),
        "forecast_seal": seal,
        "transaction": tx.report(),
        "transaction_verification": verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        "mechanics_status": "complete",
        "instrument_status": "full prefix arrays, native targets, and native component ports retained",
        "trend_status": "observed-uninterpreted",
        "terminal_status": "unassessed",
        "claim_limit": "conditioning formation and first native context projection only; no edited-RGB or full-session forecast",
        "source_payload_exports": 0,
        "source_mutations": 0,
        "seconds": time.monotonic() - start,
        "timeline_events": len(event_rows),
    }
    save(output / "summary.json", summary)
    return summary


def worker(config_path: str | Path) -> None:
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path)
    request_record = None
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request_record = read(os.environ["SATURN_DEBUG_REQUEST"])
        request_identity = hashlib.sha256(
            json.dumps(
                {key: value for key, value in request_record.items() if key != "request_sha256"},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        require(request_identity == request_record["request_sha256"], "request identity differs")
        require(request_record["config"] == config, "request/config differs")
        for name, digest in request_record["source_hashes"].items():
            require(sha(Path.cwd() / name) == digest, "staged source differs: " + name)
        config = dict(config, request_sha256=request_record["request_sha256"], _request_record=request_record)
    if "output" not in config:
        config = dict(
            config,
            output=str(
                Path(config["output_scratch"])
                / os.environ.get("MRUN_JOB_ID", "local")
                / config.get("output_name", "qwen-prefix-result")
            ),
        )
    summary = run(config, progress=lambda row: print("MDB_QWEN_PREFIX_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    helper_module = runpy.run_path(str(Path(__file__).with_name("flux_image_tail_microscope.py")))
    ready = helper_module["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)

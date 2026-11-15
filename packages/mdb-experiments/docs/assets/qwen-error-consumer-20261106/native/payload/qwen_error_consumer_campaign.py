"""Two inherited forecast H19 worlds through original Qwen block19 only."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import os
import runpy
import shutil
from pathlib import Path

SCHEMA = "mdb-qwen-error-consumer-campaign-v1"
RESULT = "qwen-error-consumer-result"
PROTOCOL_SHA256 = "4e62bc3fa8adcd059ac12f775effa0010752e3c5e0cca2a2ebc4702ecef7f847"
SUBSET_SHA256 = "69a9189b00413ccb01943718a68eba2865181110aa30b04f17668a8e2e1ffb8c"
COMPONENT_SHA256 = "c93324ee76156f2fba16f70d7cef78b7accbf6020f0c899cb04d856c348b5762"
_path = Path(__file__).with_name("qwen_component_experiment_v2.py")
if hashlib.sha256(_path.read_bytes()).hexdigest() != COMPONENT_SHA256:
    raise ValueError("qualified explicit-world component evaluator differs")
C = runpy.run_path(str(_path))
ROW = C["ROW"]
HELPERS = {"qwen_component_experiment_v2.py": COMPONENT_SHA256, **C["HELPERS"]}
PATH_KEYS = ("selection_protocol", "qwen_parameter_manifest", "parent_parameter_manifest", "qwen_config", "qwen_source_binding", "retrieval_script",
    "world_files", "world_seal", "world_request", "world_forecast", "world_shortcut", "context_files", "context_seal", "context_request")
ROOT_KEYS = ("world_root", "context_root", "parameter_root")
CONTEXT_KEYS = {"input_ids", "attention_mask", "position_ids", "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}
require, canonical, sha, read, save = (C[key] for key in ("require", "canonical", "sha", "read", "save"))


def make_config(base):
    config = {key: base[key] for key in (*PATH_KEYS, *(key + "_sha256" for key in PATH_KEYS), *ROOT_KEYS) if key in base}
    config.setdefault("selection_protocol", "protocol.json")
    config.setdefault("selection_protocol_sha256", PROTOCOL_SHA256)
    config.setdefault("qwen_parameter_manifest_sha256", SUBSET_SHA256)
    config.update(schema=SCHEMA, helper_source_sha256=dict(HELPERS), qwen_native_source_sha256=ROW["NATIVE_SHA256"], sdpa_source_sha256=ROW["SDPA_SHA256"],
        attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa"), qwen_rope_buffer_policy=dict(ROW["ROPE_POLICY"]),
        component_index=19, horizon=20, sequence_rows=512, max_wall_seconds=240, forecast_transaction_bytes=64 << 20, num_threads=4,
        output_scratch="/mnt/big/mdb/research", output_name=RESULT, transport_limits=dict(max_expanded_bytes=2 << 30, max_member_bytes=64 << 20))
    return config


def validate_config(config):
    fixed = make_config(config)
    for key in fixed.keys() - set(PATH_KEYS) - {key + "_sha256" for key in PATH_KEYS} - set(ROOT_KEYS):
        require(config.get(key) == fixed[key], "fixed configuration differs: " + key)
    for key in (*PATH_KEYS, *ROOT_KEYS):
        require(isinstance(config.get(key), str) and bool(config[key]), "missing source path: " + key)
    for key in PATH_KEYS:
        require(isinstance(config.get(key + "_sha256"), str) and len(config[key + "_sha256"]) == 64, "missing metadata identity: " + key)
    require(config["selection_protocol_sha256"] == PROTOCOL_SHA256 and config["qwen_parameter_manifest_sha256"] == SUBSET_SHA256
        and config["retrieval_script_sha256"] == ROW["BASE"]["RETRIEVER_SHA256"], "frozen source identities differ")
    require(not any(key.startswith(("flux_", "vae_", "tokenizer_")) for key in config), "unrelated components forbidden")
    return C["contract"]()


def helpers():
    for name, expected in HELPERS.items():
        require(sha(Path(__file__).with_name(name)) == expected, "frozen helper differs: " + name)
    return C


def run(config):
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    validate_config(config); require(torch.cuda.is_available(), "original component requires CUDA")
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=False)
    for key in PATH_KEYS:
        require(sha(config[key]) == config[key + "_sha256"], "metadata identity differs: " + key)
        destination = output / "metadata" / (key + Path(config[key]).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(config[key], destination)
    protocol = read(config["selection_protocol"])
    require(protocol["physical_plan"] == C["contract"]() and protocol["component_index"] == 19 and protocol["horizon"] == 20
        and protocol["sequence_rows"] == 512 and protocol["world_order"] == ["full-input", "cheap-input"]
        and set(protocol["shared_context_sources"]) == CONTEXT_KEYS and protocol["qwen_parameter_manifest_sha256"] == SUBSET_SHA256,
        "frozen world/source protocol differs")
    require(all(config[key + "_sha256"] == protocol[key + "_sha256"] for key in ("qwen_config", "qwen_source_binding", "parent_parameter_manifest")), "original metadata lineage differs")
    torch.set_num_threads(config["num_threads"]); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    runtime = ROW["runtime_contract"](config); save(output / "runtime-contract.json", runtime); save(output / "plan.json", config)
    tensors, references, layouts, verify_sources = C["load_sources"](config, protocol, torch.device("cuda"))
    save(output / "layout-restoration.json", layouts)
    qconfig = Qwen3Config.from_dict(read(config["qwen_config"])); qconfig._attn_implementation = "sdpa"
    require((qconfig.hidden_size, qconfig.intermediate_size, qconfig.num_hidden_layers, qconfig.num_attention_heads, qconfig.num_key_value_heads, qconfig.head_dim)
        == (2560, 9728, 36, 32, 8, 128), "original geometry differs")
    selected, parent = (read(config[key]) for key in ("qwen_parameter_manifest", "parent_parameter_manifest"))
    original = [row for row in parent["artifacts"] if row["source"]["tensor"].startswith("model.layers.19.")]
    require(len(parent["artifacts"]) == 398 and len(original) == 11 and selected["artifacts"] == original
        and sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in original) == 201861632, "original eleven recipes differ")
    helpers(); h = ROW["helpers"](); bank = h["flux_source_receiver.py"]["RecipeBank"](selected, config["parameter_root"])
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(qconfig, layer_idx=19).to(dtype=torch.bfloat16)
    layer = layer.to_empty(device=torch.device("cuda"))
    with torch.inference_mode():
        for row in original:
            name = row["source"]["tensor"]
            value = bank.load(h["flux_functional_qwen_prefix_worker.py"]["_recipe"](row, name))
            layer.get_parameter(name.removeprefix("model.layers.19.")).copy_(torch.from_numpy(value).to(device="cuda", dtype=torch.bfloat16))
    layer.eval().requires_grad_(False)
    before = C["identities"](layer, 19)["payloads"]
    require(all(before[row["source"]["tensor"]] == dict(shape=row["source"]["shape"], dtype="torch.bfloat16", sha256=row["source"]["payload_sha256"]) for row in original), "restored payload identity differs")
    io = dict(payload_reads=bank.payload_reads, load_requests=bank.reads, cache_hits=bank.cache_hits, parameter_payload_bytes=bank.bytes,
        peak_cached_payload_bytes=bank.peak_retained, checkpoint_files_verified_once=len(bank.signatures), checkpoint_bytes_verified_once=sum(row[2] for row in bank.signatures.values()))
    bank.bits.clear(); bank.retained = 0
    save(output / "source.json", dict(job_id=config["job_id"], request_sha256=config["request_sha256"], helper_source_sha256=HELPERS,
        source_hashes=config.get("request_source_hashes", {}), frontier=C["scope"](19), roots=protocol["roots"],
        metadata={key: dict(path="metadata/" + key + Path(config[key]).suffix, sha256=config[key + "_sha256"]) for key in PATH_KEYS},
        selected_original_names=sorted(before), parameter_recipe_io=io, hydration=dict(component_hydrations=1, cuda_placements=1,
            strategy="original eleven recipes into one meta BF16 direct-device DecoderLayer")))
    try:
        return C["execute_worlds"](output, layer, {name: tensors[name] for name in protocol["world_order"]}, {name: tensors[name] for name in CONTEXT_KEYS}, protocol,
            references=references, source_identity={**HELPERS, "qwen_native": ROW["NATIVE_SHA256"], "sdpa_native": ROW["SDPA_SHA256"]},
            source_metadata=dict(roots=protocol["roots"]), verify_sources=verify_sources, max_wall_seconds=config["max_wall_seconds"],
            transaction_bytes=config["forecast_transaction_bytes"], job_id=config["job_id"], request_sha256=config["request_sha256"])
    finally:
        layer.to("cpu")


def worker(config_path):
    print("two inherited H19 worlds; complete forecasts before two native block19 consumers", flush=True)
    faulthandler.enable(); faulthandler.dump_traceback_later(90, repeat=True); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path); validate_config(config)
    request_path = os.environ.get("SATURN_DEBUG_REQUEST"); require(request_path is not None, "source-bound request required")
    request = read(request_path)
    require(ROW["BASE"]["request_identity"](request) == request.get("request_sha256") and request["config"] == config, "request/config differs")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "executed source differs: " + name)
    job = os.environ["MRUN_JOB_ID"]; require(job.startswith("job-") and "/" not in job, "invalid actual job ID")
    derived = dict(config, request_source_hashes=request["source_hashes"], request_sha256=request["request_sha256"], job_id=job,
        output=str(Path(config["output_scratch"]) / job / config["output_name"]))
    run(derived); output = Path(derived["output"])
    shutil.copyfile(request_path, output / "request.json"); shutil.copyfile(config_path, output / "declared-config.json")
    receipt = ROW["helpers"]()["flux_image_tail_microscope.py"]["finalize"](output, job_id=job, request_sha256=request["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(receipt).decode(), flush=True); faulthandler.cancel_dump_traceback_later()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)

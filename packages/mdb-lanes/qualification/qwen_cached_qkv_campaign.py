"""Source-bound cached positioned-QKV component panel; no raw-input claim."""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import os
import runpy
import shutil
from functools import lru_cache
from pathlib import Path

SCHEMA = "mdb-qwen-cached-qkv-campaign-v1"
RESULT = "qwen-cached-qkv-result"
PROTOCOL_SHA256 = "8c53ae9417ba40420e806cb634d59931fb0131f95d0329e5b34e078429503bff"
COMPONENT_SHA256 = "6efd05f0a4128aee469c0eb2e65d9f2da508ffb029091ab8e2b604fd5cb61f26"
PREDICTOR_SHA256 = "95d356185b8bdd64479bc1e7937fa034aa24dc5f5b011b7075f0430e31c1bb21"
_component = Path(__file__).with_name("qwen_component_experiment.py")
if hashlib.sha256(_component.read_bytes()).hexdigest() != COMPONENT_SHA256:
    raise ValueError("qualified shared component evaluator differs")
C = runpy.run_path(str(_component))
ROW = C["ROW"]
HELPERS = {**C["HELPERS"], "qwen_component_experiment.py": COMPONENT_SHA256, "qwen_cached_qkv.py": PREDICTOR_SHA256}
PATH_KEYS = (*ROW["PATH_KEYS"], "parent_forecast")
SOURCE_KEYS = (ROW["SOURCE_KEYS"] - {"gate", "up", "product"}) | {"q", "k", "v"}
PHYSICAL_PLAN = dict(cached_attention_downstream_evaluations=2, cached_SDPA=2, cached_linear=8,
    full_functional_edited_blocks=2, functional_SDPA=2, functional_linear=14, native_block_calls=3, native_SDPA=3,
    native_linear=21, component_hydrations=1, full_model=0, LM=0, embedding=0, tokenizer=0, FLUX=0, VAE=0, noise=0)
require, canonical, sha, read, save = (C[key] for key in ("require", "canonical", "sha", "read", "save"))


@lru_cache(maxsize=1)
def helpers():
    for name, expected in HELPERS.items():
        require(sha(Path(__file__).with_name(name)) == expected, "frozen helper differs: " + name)
    return C, runpy.run_path(str(Path(__file__).with_name("qwen_cached_qkv.py")))


def contract():
    return dict(PHYSICAL_PLAN, native_authorizations=3, forecast_module_attempts=0)


def make_config(base):
    config = {key: base[key] for key in (*PATH_KEYS, *(key + "_sha256" for key in PATH_KEYS), "parameter_root", "input_root") if key in base}
    config.setdefault("selection_protocol", "protocol.json")
    config.setdefault("selection_protocol_sha256", PROTOCOL_SHA256)
    config.setdefault("qwen_parameter_manifest_sha256", ROW["SUBSET_SHA256"])
    config.update(schema=SCHEMA, helper_source_sha256=dict(HELPERS), qwen_native_source_sha256=ROW["NATIVE_SHA256"], sdpa_source_sha256=ROW["SDPA_SHA256"],
        attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa"), qwen_rope_buffer_policy=dict(ROW["ROPE_POLICY"]),
        edit_index=18, horizon=19, sequence_rows=512, max_wall_seconds=240, forecast_transaction_bytes=64 << 20, num_threads=4,
        output_scratch="/mnt/big/mdb/research", output_name=RESULT, transport_limits=dict(max_expanded_bytes=2 << 30, max_member_bytes=64 << 20))
    return config


def validate_config(config):
    fixed = make_config(config)
    for key in fixed.keys() - set(PATH_KEYS) - {key + "_sha256" for key in PATH_KEYS} - {"parameter_root", "input_root"}:
        require(config.get(key) == fixed[key], "fixed configuration differs: " + key)
    for key in (*PATH_KEYS, "parameter_root", "input_root"):
        require(isinstance(config.get(key), str) and bool(config[key]), "missing source path: " + key)
    for key in PATH_KEYS:
        require(isinstance(config.get(key + "_sha256"), str) and len(config[key + "_sha256"]) == 64, "missing metadata identity: " + key)
    require(config["selection_protocol_sha256"] == PROTOCOL_SHA256 and config["qwen_parameter_manifest_sha256"] == ROW["SUBSET_SHA256"]
        and config["retrieval_script_sha256"] == ROW["BASE"]["RETRIEVER_SHA256"], "fixed source identities differ")
    require(not any(key.startswith(("flux_", "vae_", "tokenizer_")) for key in config), "unrelated components forbidden")
    return contract()


def callbacks():
    component, predictor = helpers()
    h = ROW["helpers"](); graph, history = h["flux_functional_qwen_prefix.py"], h["qwen_h18_state_history.py"]
    def cached(hidden, sources, parameters, profile, context, dispatch, signs, observe):
        return predictor["predict"](graph, hidden, sources, parameters, profile, 18, context, dispatch, signs, observe)
    def functional(hidden, _sources, parameters, profile, context, dispatch, _signs, observe):
        history["_block"](graph, hidden, parameters, profile, 18, context, dispatch, lambda _lane, name, value: observe(name, value), "functional")
        return dict(counts=dict(SDPA=1, linear=7, RMSNorm=4, native_module_forwards=0), scope="complete edited functional reference")
    return dict(cached=cached, functional=functional)


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
    torch.set_num_threads(config["num_threads"]); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    runtime = ROW["runtime_contract"](config); save(output / "runtime-contract.json", runtime); save(output / "plan.json", config)
    protocol = read(config["selection_protocol"])
    require(protocol["physical_plan"] == PHYSICAL_PLAN and set(protocol["source_arrays"]) == SOURCE_KEYS and protocol["sequence_rows"] == 512
        and protocol["edit_index"] == 18 and protocol["horizon"] == 19, "frozen panel/source contract differs")
    sources, references, layouts, verify_sources = C["inherited_sources"](config, protocol, torch.device("cuda"))
    save(output / "layout-restoration.json", layouts)
    qconfig = Qwen3Config.from_dict(read(config["qwen_config"])); qconfig._attn_implementation = "sdpa"
    require((qconfig.hidden_size, qconfig.intermediate_size, qconfig.num_hidden_layers, qconfig.num_attention_heads, qconfig.num_key_value_heads, qconfig.head_dim) == (2560, 9728, 36, 32, 8, 128), "original geometry differs")
    selected, parent = (read(config[key]) for key in ("qwen_parameter_manifest", "parent_parameter_manifest"))
    original = [row for row in parent["artifacts"] if row["source"]["tensor"].startswith("model.layers.18.")]
    require(len(parent["artifacts"]) == 398 and len(original) == 11 and selected["artifacts"] == original
        and sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in original) == 201861632, "original eleven recipes differ")
    helpers(); h = ROW["helpers"](); bank = h["flux_source_receiver.py"]["RecipeBank"](selected, config["parameter_root"])
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(qconfig, layer_idx=18).to(dtype=torch.bfloat16)
    layer = layer.to_empty(device=torch.device("cuda"))
    with torch.inference_mode():
        for row in original:
            name = row["source"]["tensor"]
            value = bank.load(h["flux_functional_qwen_prefix_worker.py"]["_recipe"](row, name))
            layer.get_parameter(name.removeprefix("model.layers.18.")).copy_(torch.from_numpy(value).to(device="cuda", dtype=torch.bfloat16))
    layer.eval().requires_grad_(False)
    before = C["identities"](layer, 18)["payloads"]
    require(all(before[row["source"]["tensor"]] == dict(shape=row["source"]["shape"], dtype="torch.bfloat16", sha256=row["source"]["payload_sha256"]) for row in original), "restored payload identity differs")
    io = dict(payload_reads=bank.payload_reads, load_requests=bank.reads, cache_hits=bank.cache_hits, parameter_payload_bytes=bank.bytes,
        peak_cached_payload_bytes=bank.peak_retained, checkpoint_files_verified_once=len(bank.signatures), checkpoint_bytes_verified_once=sum(row[2] for row in bank.signatures.values()))
    bank.bits.clear(); bank.retained = 0
    save(output / "source.json", dict(job_id=config["job_id"], request_sha256=config["request_sha256"], helper_source_sha256=HELPERS,
        source_hashes=config.get("request_source_hashes", {}), frontier=C["FRONTIER_SCOPE"],
        metadata={key: dict(path="metadata/" + key + Path(config[key]).suffix, sha256=config[key + "_sha256"]) for key in PATH_KEYS},
        selected_original_names=sorted(before), parameter_recipe_io=io, hydration=dict(component_hydrations=1, cuda_placements=1,
            strategy="original eleven recipes into one meta BF16 direct-device DecoderLayer")))
    result = C["execute_panel"](output, layer, sources, references, protocol, callbacks=callbacks(), expected_counts=contract(),
        evaluation_counts=dict(cached="cached_attention_downstream_evaluations", functional="full_functional_edited_blocks"),
        source_identity={**HELPERS, "qwen_native": ROW["NATIVE_SHA256"], "sdpa_native": ROW["SDPA_SHA256"]},
        source_metadata=dict(input_root=config["input_root"], **{key: protocol[key] for key in ("parent_job", "parent_request_sha256", "parent_raw_manifest_sha256", "parent_global_seal_sha256", "source_case")}),
        verify_sources=verify_sources, max_wall_seconds=config["max_wall_seconds"], transaction_bytes=config["forecast_transaction_bytes"],
        job_id=config["job_id"], request_sha256=config["request_sha256"], result_schema=SCHEMA)
    layer.to("cpu")
    return result


def worker(config_path):
    print("cached-QKV conditional component; global seal before native targets", flush=True)
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

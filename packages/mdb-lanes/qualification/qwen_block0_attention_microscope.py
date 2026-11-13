"""Bounded source-QKV and equivalent-causal-SDPA diagnostic, ending at H1.

The fixed-bank triangle is an observed primitive intervention, not three native
source branches or an independent kernel implementation. Original H0/H1 are
historical replay evidence. All scientific misses remain observations.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

PREFIX_HELPER = "flux_functional_qwen_prefix.py"
PREFIX_SHA256 = "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a"
RECIPE_HELPER = "flux_source_receiver.py"
RECIPE_SHA256 = "d483e41bfac018c8225662e8417bcef46f6916fa2a14d910f16e9a12beb5250c"
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
ARMS = ("A.explicit-repeated", "B.causal-repeated", "C.causal-grouped")
CASES = ("short", "long")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def helpers():
    here = Path(__file__).parent
    require(sha(here / PREFIX_HELPER) == PREFIX_SHA256, "frozen Qwen arithmetic differs")
    require(sha(here / RECIPE_HELPER) == RECIPE_SHA256, "frozen recipe helper differs")
    return runpy.run_path(str(here / PREFIX_HELPER)), runpy.run_path(str(here / RECIPE_HELPER))


def contract(config=None):
    return {"schema": "mdb-qwen-block0-attention-counts-v1", "native_block_hydrations": 1,
            "native_block_calls": 2, "native_rotary_calls": 2,
            "functional_source_blocks": 2, "functional_embeddings": 2,
            "source_sdpa_calls": 2, "triangle_sdpa_calls": 3, "native_sdpa_calls": 2,
            "total_sdpa_calls": 7, "triangle_arms": list(ARMS), "fixed_bank_case": "long",
            "selected_parameter_tensors": 12, "native_block_parameters": 11,
            "full_encoder_calls": 0, "tokenizer_calls": 0, "FLUX_calls": 0, "VAE_calls": 0,
            "profiler_extra_forwards": 0, "persistent_weight_banks": 0,
            "primitive_scope": "source formation + three order-conditioned equivalent full-causal representations, not a factorial or future-image forecast"}


def validate_config(config):
    require(config["schema"] == "mdb-qwen-block0-attention-plan-v1", "unexpected plan schema")
    require(set(config["cases"]) == set(CASES) and config["fixed_bank_case"] == "long", "fixed source bank differs")
    tiny = config.get("profile") == "tiny-cpu"
    require(config["device"] == ("cpu" if tiny else "cuda"), "declared execution device differs")
    require(type(config["source_reduction_policy"]) is bool and (tiny or config["source_reduction_policy"] is True), "original reduction policy required")
    require(0 < config.get("max_wall_seconds", 600) <= 900, "finite block0 wall bound required")
    require(config.get("cpu_threads", 4) in range(1, 9), "CPU thread bound differs")
    require(config.get("shared_prefix_rows", 30) > 0, "shared prefix bound required")
    for key in ("qwen_config", "qwen_manifest", "retrieval_script"):
        require(sha(config[key]) == config[key + "_sha256"], "source metadata differs: " + key)
    require(config["retrieval_script_sha256"] == RETRIEVER_SHA256, "qualified standalone retriever differs")
    helpers()
    return contract(config)


def tensor_record(output, name, value):
    """Lossless BF16 bits or native integer/FP32, exact payload and layout metadata."""
    import numpy as np
    import torch

    from mdb_lanes.forward_recording import tensor_descriptor

    value = value.detach()
    array = value.contiguous().cpu()
    bits = array.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else array.numpy()
    path = Path(output) / (name + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, bits, allow_pickle=False)
    return {"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": sha(path),
            "raw_sha256": hashlib.sha256(bits.tobytes()).hexdigest(), "shape": list(value.shape),
            "stride": list(value.stride()), "dtype": str(value.dtype), "stored_dtype": str(bits.dtype),
            "encoding": "bf16-bits-v1" if value.dtype == torch.bfloat16 else "native-numpy-v1",
            "device_at_capture": str(value.device), "descriptor": tensor_descriptor(value)}


def load_input(root, record, device):
    import numpy as np
    import torch

    path = Path(root) / record["path"]
    require(not Path(record["path"]).is_absolute() and ".." not in Path(record["path"]).parts, "unsafe source array")
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "original source array differs")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == record["shape"], "source array shape differs")
    expected = record.get("stored_dtype", record.get("dtype"))
    require(str(value.dtype) == expected, "source stored dtype differs")
    result = torch.from_numpy(value)
    if record.get("encoding") == "bf16-bits-v1":
        require(result.dtype == torch.uint16, "BF16 source bits required")
        result = result.view(torch.bfloat16)
    return result.to(device)


def compare(left, right, *, token_axis=None, prefix=None):
    import torch

    if prefix is not None:
        left, right = left.narrow(token_axis, 0, prefix), right.narrow(token_axis, 0, prefix)
    require(left.shape == right.shape and left.numel() > 0, "comparison geometry differs")
    a, b = left.detach().double().cpu(), right.detach().double().cpu()
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "nonfinite diagnostic field")
    delta = a - b
    l2, r2, d2 = (float((value * value).sum()) for value in (a, b, delta))
    maximum = int(delta.abs().reshape(-1).argmax())
    return {"orientation": "left-minus-right", "shape": list(a.shape), "elements": a.numel(),
            "rms": (d2 / a.numel()) ** .5, "reference_rms": (r2 / a.numel()) ** .5,
            "relative_rms": None if r2 == 0 else (d2 / r2) ** .5,
            "signed_mean": float(delta.mean()), "maxabs": float(delta.abs().max()),
            "changed_values": int(torch.count_nonzero(delta)),
            "bitwise_exact": left.dtype == right.dtype and torch.equal(left.contiguous().view(torch.uint8).cpu(), right.contiguous().view(torch.uint8).cpu()),
            "cosine": None if l2 * r2 == 0 else float((a * b).sum()) / (l2 * r2) ** .5,
            "first_nonzero_flat": None if not bool((delta != 0).any()) else int((delta.reshape(-1) != 0).nonzero()[0]),
            "worst_flat": maximum}


def rope(config, hidden, graph):
    import torch

    profile = graph["_native_profile"](config, 1)
    positions = torch.arange(hidden.shape[1], device=hidden.device, dtype=torch.int64)[None]
    inv = 1.0 / (profile["rope_theta"] ** (torch.arange(0, profile["head_dim"], 2, dtype=torch.float32, device="cpu") / profile["head_dim"]))
    inv = inv.to(hidden.device)
    frequencies = (inv[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
    rotary = torch.cat((frequencies, frequencies), -1)
    return rotary.cos().to(hidden.dtype), rotary.sin().to(hidden.dtype)


def tail(hidden, head, weights, config, graph):
    import torch.nn.functional as functional

    linear, norm = graph["_linear"], graph["_rms_norm"]
    root = "model.layers.0."
    flat = head.transpose(1, 2).contiguous().reshape(hidden.shape[0], hidden.shape[1], -1)
    projection = linear(flat, weights[root + "self_attn.o_proj.weight"])
    residual = hidden + projection
    mlp_input = norm(residual, weights[root + "post_attention_layernorm.weight"], config.rms_norm_eps)
    gate, up = (linear(mlp_input, weights[root + "mlp." + part + ".weight"]) for part in ("gate_proj", "up_proj"))
    activated = functional.silu(gate) * up
    mlp = linear(activated, weights[root + "mlp.down_proj.weight"])
    return {"head": head, "projection": projection, "after_attention": residual, "mlp_input": mlp_input,
            "gate": gate, "up": up, "activation": activated, "mlp": mlp, "H1": residual + mlp}


def source_graph(ids, mask, weights, config, graph):
    import torch.nn.functional as functional

    p = graph["_native_profile"](config, 1)
    hidden = functional.embedding(ids, weights["model.embed_tokens.weight"])
    cos, sin = rope(config, hidden, graph)
    root = "model.layers.0."
    normalized = graph["_rms_norm"](hidden, weights[root + "input_layernorm.weight"], config.rms_norm_eps)
    raw = {part: graph["_linear"](normalized, weights[root + "self_attn." + part + "_proj.weight"]) for part in ("q", "k", "v")}
    shaped = {part: value.view(1, ids.shape[1], p["num_attention_heads"] if part == "q" else p["num_key_value_heads"], p["head_dim"]) for part, value in raw.items()}
    qn = graph["_rms_norm"](shaped["q"], weights[root + "self_attn.q_norm.weight"], config.rms_norm_eps).transpose(1, 2)
    kn = graph["_rms_norm"](shaped["k"], weights[root + "self_attn.k_norm.weight"], config.rms_norm_eps).transpose(1, 2)
    query = qn * cos[:, None] + graph["_rotate_half"](qn) * sin[:, None]
    key = kn * cos[:, None] + graph["_rotate_half"](kn) * sin[:, None]
    value = shaped["v"].transpose(1, 2)
    actual_mask, causal, mask_profile = graph["_mask_for_attention"](mask, batch_size=1, sequence_length=ids.shape[1], backend="sdpa", dtype=hidden.dtype, device=hidden.device)
    dispatch = graph["AttentionDispatch"].from_mapping({"backend": "sdpa", "gqa_policy": "enable_gqa" if actual_mask is None else "repeat_kv", "source": "source-bound installed Transformers SDPA"})
    attended = graph["_attention"](query, key, value, actual_mask, is_causal=causal, scale=p["head_dim"] ** -.5, groups=p["num_key_value_groups"], dispatch=dispatch)
    fields = {"H0": hidden, "normalized": normalized, "q_raw": raw["q"], "k_raw": raw["k"], "v_raw": raw["v"],
              "q_norm": qn, "k_norm": kn, "q": query, "k": key, "v": value, "cos": cos, "sin": sin}
    fields.update(tail(hidden, attended.transpose(1, 2), weights, config, graph))
    return fields, {"mask": actual_mask, "is_causal": causal, "dispatch": dispatch.to_dict(), "mask_profile": mask_profile}


def triangle(bank, weights, config, graph, call):
    import torch

    groups = config.num_attention_heads // config.num_key_value_heads
    repeated_k, repeated_v = (graph["_repeat_kv"](bank[part], groups) for part in ("k", "v"))
    sequence = bank["q"].shape[2]
    allowed = torch.ones((1, 1, sequence, sequence), device=bank["q"].device, dtype=torch.bool).tril()
    specs = [(ARMS[0], repeated_k, repeated_v, allowed, False, False),
             (ARMS[1], repeated_k, repeated_v, None, True, False),
             (ARMS[2], bank["k"], bank["v"], None, True, True)]
    result = {}
    for name, key, value, mask, causal, gqa in specs:
        head = call(name, bank["q"], key, value, mask, causal, gqa, config.head_dim ** -.5)
        result[name] = tail(bank["H0"], head, weights, config, graph)
    return result, {"explicit_causal_mask": allowed, "repeated_k": repeated_k, "repeated_v": repeated_v}


def canonical_kv(value, kv_heads, groups):
    """Original native repeat layout: contiguous groups share each KV head."""
    import torch

    if value.shape[1] == kv_heads:
        return value
    require(value.shape[1] == kv_heads * groups, "unrecognized consumed KV head geometry")
    grouped = value.reshape(value.shape[0], kv_heads, groups, value.shape[2], value.shape[3])
    canonical = grouped[:, :, 0]
    require(all(torch.equal(grouped[:, :, index].contiguous().view(torch.uint8), canonical.contiguous().view(torch.uint8)) for index in range(groups)), "consumed repeated KV heads are not canonical duplicates")
    return canonical


@contextmanager
def sdpa_recorder(rows, label, *, output, profile=False):
    """Capture unchanged actual calls; profiler failure never repeats a forward."""
    import torch
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention
    captured = []

    def observe(query, key, value, *args, **kwargs):
        index = len(rows)
        mask = args[0] if args else kwargs.get("attn_mask")
        row = {"label": label, "ordinal": index, "is_causal": bool(args[2] if len(args) > 2 else kwargs.get("is_causal", False)),
               "enable_gqa": bool(kwargs.get("enable_gqa", False)), "scale": kwargs.get("scale"),
               "dropout_p": args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0),
               "query": tensor_record(output, f"sdpa/{index}/query", query),
               "key": tensor_record(output, f"sdpa/{index}/key", key),
               "value": tensor_record(output, f"sdpa/{index}/value", value),
               "mask": None if mask is None else tensor_record(output, f"sdpa/{index}/mask", mask),
               "profiler": {"status": "not-requested", "selected_kernel": "unprofiled"}}
        profiler = None
        if profile:
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if query.device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profiler = torch.profiler.profile(activities=activities, record_shapes=True)
                profiler.__enter__()
            except Exception as error:
                row["profiler"] = {"status": "unavailable", "error": str(error), "selected_kernel": "unprofiled"}
                profiler = None
        try:
            result = original(query, key, value, *args, **kwargs)
        finally:
            if profiler is not None:
                try:
                    profiler.__exit__(None, None, None)
                    row["profiler"] = {"status": "observed", "events": sorted({event.name for event in profiler.events()}),
                                       "scope": "passive events for this existing call; no forced backend or extra call"}
                except Exception as error:
                    row["profiler"] = {"status": "instrument-broken", "error": str(error), "selected_kernel": "unprofiled"}
        row["output"] = tensor_record(output, f"sdpa/{index}/output", result)
        rows.append(row)
        captured.append({"q": query.detach().clone(), "k": key.detach().clone(), "v": value.detach().clone(), "head": result.detach().clone()})
        return result

    functional.scaled_dot_product_attention = observe
    try:
        yield captured
    finally:
        functional.scaled_dot_product_attention = original


def native_capture(block, hidden, mask, positions, rotary, rows, output, case, profile):
    """Exactly one unchanged native block call, with observation-only hooks."""
    ports = {"H0": hidden}
    handles = []
    names = {"input_layernorm": "normalized", "self_attn.q_proj": "q_raw", "self_attn.k_proj": "k_raw", "self_attn.v_proj": "v_raw",
             "self_attn.q_norm": "q_norm", "self_attn.k_norm": "k_norm", "self_attn.o_proj": "projection",
             "post_attention_layernorm": "mlp_input", "mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.down_proj": "mlp"}
    entries = []
    for name, module in block.named_modules():
        handles.append(module.register_forward_pre_hook(lambda _m, _a, name=name: entries.append(name), prepend=True))
        if name in names:
            handles.append(module.register_forward_hook(lambda _m, _a, out, name=name: ports.__setitem__(names[name], out.detach().clone())))
        if name == "mlp.down_proj":
            handles.append(module.register_forward_pre_hook(lambda _m, args: ports.__setitem__("activation", args[0].detach().clone()), prepend=True))
    try:
        cos, sin = rotary(hidden, positions)
        ports.update(cos=cos, sin=sin)
        with sdpa_recorder(rows, "native." + case, output=output, profile=profile) as actual:
            result = block(hidden_states=hidden, attention_mask=mask, position_ids=positions, position_embeddings=(cos, sin), use_cache=False)
        require(len(actual) == 1, "native block SDPA census differs")
        ports["H1"] = result
        ports["after_attention"] = hidden + ports["projection"]
        ports["q_norm"] = ports["q_norm"].transpose(1, 2)
        ports["k_norm"] = ports["k_norm"].transpose(1, 2)
        groups = block.self_attn.num_key_value_groups
        ports.update(q=actual[0]["q"], k=canonical_kv(actual[0]["k"], block.config.num_key_value_heads if hasattr(block, "config") else block.self_attn.config.num_key_value_heads, groups),
                     v=canonical_kv(actual[0]["v"], block.self_attn.config.num_key_value_heads, groups), head=actual[0]["head"])
        return ports, entries
    finally:
        for handle in handles:
            handle.remove()


def per_token_head(left, right, *, head_axis=False):
    """Small signed reductions, full fields retained separately."""
    # One transfer per full operand, not one CUDA synchronization per token.
    left, right = left.detach().cpu(), right.detach().cpu()
    if head_axis:
        return [[compare(left[:, head, token], right[:, head, token]) for token in range(left.shape[2])] for head in range(left.shape[1])]
    return [compare(left[:, token], right[:, token]) for token in range(left.shape[1])]


def vector_account(a, b, observed):
    """B-A and C-B telescope, cross terms signed, common source units."""

    x, y, z = (value.detach().double().cpu().reshape(-1) for value in (a, b, observed))
    energy = lambda value: float((value * value).sum())
    dot = float((x * y).sum())
    total = x + y
    return {"first_energy": energy(x), "second_energy": energy(y), "sum_energy": energy(total),
            "cross_dot": dot, "signed_cross_term": 2 * dot,
            "energy_identity_residual": energy(total) - energy(x) - energy(y) - 2 * dot,
            "first_vs_historical_cosine": None if energy(x) * energy(z) == 0 else float((x * z).sum()) / (energy(x) * energy(z)) ** .5,
            "second_vs_historical_cosine": None if energy(y) * energy(z) == 0 else float((y * z).sum()) / (energy(y) * energy(z)) ** .5,
            "historical_energy": energy(z), "coordinate_sum_elements": total.numel(),
            "scope": "order-conditioned vector telescope and alignment; not factorial interaction, intrinsic gain or causal shares"}


def run(config, *, progress=None):
    declared = validate_config(config)
    import torch
    import torch.nn.functional as functional
    from transformers import Qwen3Config
    from transformers.models.qwen3 import modeling_qwen3

    from mdb_lanes import parameter_artifacts

    graph, recipe = helpers()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    require(sha(inspect.getfile(parameter_artifacts)) == RETRIEVER_SHA256, "installed recipe validator differs")
    start = time.monotonic()
    events = []

    def event(kind, **detail):
        row = {"kind": kind, "seconds": time.monotonic() - start, **detail}
        require(row["seconds"] <= config.get("max_wall_seconds", 600), "finite block0 wall exceeded")
        events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress:
            progress(row)

    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
    if config["device"] == "cuda":
        require(torch.cuda.is_available(), "declared CUDA unavailable")
    import platform

    import transformers
    runtime = {"versions":{"torch":torch.__version__,"transformers":transformers.__version__,"python":platform.python_version()},
               "device":config["device"],"flags":{"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
               "float32_matmul_precision":torch.get_float32_matmul_precision(),"cuda_tf32":torch.backends.cuda.matmul.allow_tf32,
               "cuda_bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
               "cudnn_tf32":torch.backends.cudnn.allow_tf32,"cudnn_deterministic":torch.backends.cudnn.deterministic,
               "cudnn_benchmark":torch.backends.cudnn.benchmark,"cpu_threads":torch.get_num_threads()},
               "profiling":"passive only, on existing seven calls; no forced backend"}
    for name,expected in config.get("expected_versions",{}).items():
        require(runtime["versions"][name]==expected,"source runtime version differs: "+name)
    save(output / "runtime-contract.json", runtime)
    sources = {}
    import transformers.integrations.sdpa_attention as sdpa_module
    import transformers.masking_utils as masking_module
    for name, module in (("qwen3", modeling_qwen3), ("sdpa", sdpa_module), ("masking", masking_module)):
        digest = sha(inspect.getfile(module))
        require(digest == config["native_sources"][name], "native implementation source differs: " + name)
        sources[name] = {"sha256": digest, "path": inspect.getfile(module)}
    input_root = Path(config["input_root"])
    provenance = config["source_provenance"]
    source_manifest_path = input_root / "files.json"
    require(sha(source_manifest_path) == provenance["manifest_sha256"], "original source manifest differs")
    source_manifest = read(source_manifest_path)
    require(source_manifest["schema"] == "mdb-image-tail-files-v1" and source_manifest["job_id"] == provenance["job_id"]
            and source_manifest["request_sha256"] == provenance["request_sha256"], "original source wrapper identities differ")
    for name, row in config["input_files"].items():
        path = input_root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and not path.is_symlink(), "unsafe source member")
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "selected original member differs: " + name)
        require(source_manifest["files"][name] == row, "source selection not in original wrapper: " + name)
        target = output / "original-inputs" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(source_manifest_path, output / "original-inputs/files.json")
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    if config.get("_request_record"):
        save(output / "request.json", config["_request_record"])
    for key in ("qwen_config", "qwen_manifest", "retrieval_script"):
        shutil.copyfile(config[key], output / Path(config[key]).name)
    model_config = Qwen3Config.from_dict(read(config["qwen_config"]))
    model_config._attn_implementation = "sdpa"
    if config.get("profile") != "tiny-cpu":
        require((model_config.hidden_size, model_config.num_attention_heads, model_config.num_key_value_heads, model_config.head_dim, model_config.intermediate_size) == (2560,32,8,128,9728), "original unequal-width geometry differs")
    manifest = read(config["qwen_manifest"])
    require(config.get("profile") == "tiny-cpu" or len(manifest["artifacts"]) == 398, "original full recipe inventory differs")
    wanted = {"model.embed_tokens.weight", *("model.layers.0." + suffix for suffix in graph["QWEN_LAYER_PARAMETER_SUFFIXES"])}
    entries = {row["source"]["tensor"]: row for row in manifest["artifacts"]}
    require(wanted <= entries.keys() and len(wanted) == 12, "embedding and exact eleven block0 recipes required")
    bank = recipe["RecipeBank"](manifest, config["parameter_root"], max_cache_bytes=1)
    weights = {}
    for name in sorted(wanted):
        row = entries[name]
        value = bank.load({"path": row["path"], "sha256": row["sha256"], "parameter_tensor": name,
                           "stored_shape": row["source"]["shape"], "descriptor": {"sha256": row["source"]["payload_sha256"]}})
        weights[name] = torch.from_numpy(value).to(device=config["device"], dtype=torch.bfloat16)
        del value
    with torch.device("meta"):
        block = modeling_qwen3.Qwen3DecoderLayer(model_config, layer_idx=0).to(torch.bfloat16)
    block.load_state_dict({name.removeprefix("model.layers.0."): value for name,value in weights.items() if name.startswith("model.layers.0.")}, strict=True, assign=True)
    block.eval().requires_grad_(False)
    rotary = modeling_qwen3.Qwen3RotaryEmbedding(model_config, device=torch.device("cpu")).to(config["device"])
    require(rotary.inv_freq.dtype == torch.float32, "original CPU-FP32 rotary policy required")
    rotary_before = {name: tensor_record(output, "rotary-source/" + name, value) for name,value in rotary.named_buffers()}
    before = {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()} for name,value in weights.items()}
    native_calls, sdpa_rows, cases, source_values, source_inputs = {}, [], {}, {}, {}
    profile = bool(config.get("profile_sdpa", True))
    with torch.inference_mode(), graph["module_entry_guard"](block) as forbidden:
        for case in CASES:
            spec = config["cases"][case]
            sidecars = {lane: read(input_root / spec[lane + "_qwen"]) for lane in ("forecast", "native")}
            for lane, record in sidecars.items():
                require(spec[lane + "_qwen"] in config["input_files"], "source sidecar not declared")
                for key in ("input_ids", "attention_mask"):
                    require("cases/" + case + "/" + record[key]["path"] in config["input_files"], "source operand not declared")
                for index in (0,1):
                    require("cases/" + case + "/" + record["hidden_states"][index]["path"] in config["input_files"], "historical replay evidence not declared")
            root = input_root / "cases" / case
            ids = load_input(root, sidecars["native"]["input_ids"], config["device"])
            mask = load_input(root, sidecars["native"]["attention_mask"], config["device"])
            require(ids.shape == mask.shape and ids.shape[0] == 1, "source ID/mask geometry differs")
            require(config.get("profile") == "tiny-cpu" or ids.shape[1] == 512, "full512source required")
            with sdpa_recorder(sdpa_rows, "functional." + case, output=output, profile=profile):
                values, dispatch = source_graph(ids, mask, weights, model_config, graph)
            source_values[case] = values
            source_inputs[case] = (ids, mask, sidecars)
            cases[case] = {"functional": {name: tensor_record(output, "functional/" + case + "/" + name, value) for name,value in values.items()},
                           "dispatch": {key:value for key,value in dispatch.items() if key != "mask"},
                           "source_sidecars": spec,
                           "attended_rows":int(mask.sum()),"retained_rows":mask.numel(),
                           "original_mask": None if dispatch["mask"] is None else tensor_record(output,"functional/"+case+"/mask",dispatch["mask"])}
        prefix = config["shared_prefix_rows"]
        require(bool(torch.equal(source_inputs["short"][0][:,:prefix], source_inputs["long"][0][:,:prefix])), "declared common-token source binding differs")

        def primitive(name, query, key, value, mask, causal, gqa, scale):
            with sdpa_recorder(sdpa_rows, name, output=output, profile=profile):
                return functional.scaled_dot_product_attention(query,key,value,attn_mask=mask,is_causal=causal,enable_gqa=gqa,dropout_p=0.0,scale=scale)

        panel, grid_inputs = triangle(source_values["long"], weights, model_config, graph, primitive)
        panel_records = {arm:{name:tensor_record(output,"triangle/"+arm+"/"+name,value) for name,value in ports.items()} for arm,ports in panel.items()}
        grid_records = {name:tensor_record(output,"triangle-inputs/"+name,value) for name,value in grid_inputs.items()}
    require(forbidden == [], "functional source or triangle entered native block")
    event("functional_sources_and_triangle_complete", sdpa_calls=len(sdpa_rows), native_block_calls=0)
    functional_files = {str(path.relative_to(output)): {"bytes":path.stat().st_size,"sha256":sha(path)} for path in sorted(output.rglob("*")) if path.is_file() and path.name != "timeline.jsonl"}
    save(output / "functional-seal.json", {"schema":"mdb-qwen-block0-functional-fields-v1","cases":cases,"triangle":panel_records,"grid_inputs":grid_records,
                                          "sdpa_receipts":sdpa_rows.copy(),"external_files":functional_files,
                                          "parameter_identity":before,"native_block_calls":0,"scope":"source-generated conditional primitive diagnostics, no novel raw-image forecast"})
    functional_seal_sha256 = sha(output / "functional-seal.json")
    event("functional_fields_sealed", seal_sha256=functional_seal_sha256, sdpa_calls=5, native_block_calls=0)
    for case in CASES:
        ids, mask, sidecars = source_inputs[case]
        values = source_values[case]
        actual_mask, _, _ = graph["_mask_for_attention"](mask,batch_size=1,sequence_length=ids.shape[1],backend="sdpa",dtype=torch.bfloat16,device=ids.device)
        positions = torch.arange(ids.shape[1],device=ids.device,dtype=torch.int64)[None]
        event("native_block_entry",case=case, functional_seal_sha256=functional_seal_sha256)
        with torch.inference_mode():
            native, module_entries = native_capture(block,values["H0"],actual_mask,positions,rotary,sdpa_rows,output,case,profile)
        native_calls[case]=module_entries
        cases[case]["native"]={name:tensor_record(output,"native/"+case+"/"+name,value) for name,value in native.items()}
        cases[case]["matched_functional_vs_native"]={name:compare(values[name],native[name]) for name in native if name in values}
        historical={}
        for lane,record in sidecars.items():
            historical[lane]={"H0":compare(values["H0"],load_input(input_root/"cases"/case,record["hidden_states"][0],config["device"])),
                              "H1":compare(native["H1"],load_input(input_root/"cases"/case,record["hidden_states"][1],config["device"]))}
        cases[case]["historical_replay"]=historical
        cases[case]["module_entries"]=module_entries
        source_values[case+".native"]=native
        event("native_block_completed",case=case,sdpa_calls=len(sdpa_rows))
    compare_ports=("H0","normalized","q_raw","k_raw","v_raw","q_norm","k_norm","q","k","v","head","projection","after_attention","mlp_input","mlp","H1")
    pair={lane:{name:compare(source_values["long"+suffix][name],source_values["short"+suffix][name],token_axis=2 if name in ("q_norm","k_norm","q","k","v","head") else 1,prefix=prefix) for name in compare_ports} for lane,suffix in (("functional",""),("native",".native"))}
    accounts={}
    for name in ("head","projection","after_attention","mlp_input","mlp","H1"):
        axis=2 if name=="head" else 1
        a,b,c=(panel[arm][name] for arm in ARMS)
        historical=source_values["long.native"][name].double()-source_values["short.native"][name].double()
        delta1=b.double()-a.double()
        delta2=c.double()-b.double()
        sl=lambda value:value.narrow(axis,0,prefix)
        accounts[name]={"B-minus-A":{"full":compare(b,a),"shared_prefix":compare(b,a,token_axis=axis,prefix=prefix)},
                        "C-minus-B":{"full":compare(c,b),"shared_prefix":compare(c,b,token_axis=axis,prefix=prefix)},
                        "C-minus-A":{"full":compare(c,a),"shared_prefix":compare(c,a,token_axis=axis,prefix=prefix)},
                        "prefix_telescope":vector_account(sl(delta1),sl(delta2),sl(historical)),
                        "coordinate_identity":bool(torch.equal(sl(c.double()-a.double()),sl(delta1+delta2)))}
    detailed={"historical_head":per_token_head(source_values["long.native"]["head"],source_values["short.native"]["head"],head_axis=True),
              "triangle_head":{left+"-minus-"+right:per_token_head(panel[left]["head"],panel[right]["head"],head_axis=True) for left,right in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[1]))},
              "historical_H1":per_token_head(source_values["long.native"]["H1"],source_values["short.native"]["H1"])}
    after={name:{"shape":list(value.shape),"dtype":str(value.dtype),"sha256":hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()} for name,value in weights.items()}
    require(before==after,"original parameters changed")
    rotary_after = {name: {"shape":list(value.shape),"dtype":str(value.dtype),"raw_sha256":hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()} for name,value in rotary.named_buffers()}
    require(all(rotary_after[name]["shape"]==row["shape"] and rotary_after[name]["dtype"]==row["dtype"] and rotary_after[name]["raw_sha256"]==row["raw_sha256"] for name,row in rotary_before.items()), "original rotary buffers changed")
    require(len(sdpa_rows)==7 and len(native_calls)==2 and all(len([name for name in entries if name==""])==1 for entries in native_calls.values()),"physical source/SDPA/block census differs")
    payload_bytes=sum(row["source"]["data_offsets"][1]-row["source"]["data_offsets"][0] for name,row in entries.items() if name in wanted)
    require(sha(output / "functional-seal.json") == functional_seal_sha256, "pretarget functional seal changed")
    for name,row in functional_files.items():
        require((output/name).stat().st_size == row["bytes"] and sha(output/name) == row["sha256"], "pretarget functional/source file changed: " + name)
    save(output/"cases.json",cases)
    save(output/"triangle.json",{"arms":panel_records,"inputs":grid_records,"accounts":accounts,"fixed_bank_case":"long","full_causal_equivalence_rows":ids.shape[1],"historical_bridge_valid_prefix_rows":prefix})
    save(output/"sdpa.json",sdpa_rows)
    save(output/"per-token-head.json",detailed)
    save(output/"pair-comparisons.json",pair)
    save(output/"timeline.json",events)
    save(output/"source.json",{"schema":"mdb-qwen-block0-attention-source-v1","job_id":os.environ.get("MRUN_JOB_ID"),"request_sha256":config.get("request_sha256"),"worker_sha256":sha(__file__),"native_sources":sources,"parameter_identity_before":before,"parameter_identity_after":after,"rotary_buffers_before":rotary_before,"rotary_buffers_after":rotary_after,"input_files":config["input_files"],"original_inputs_copy":"original-inputs","functional_seal_sha256":functional_seal_sha256,"helper_sources":{PREFIX_HELPER:PREFIX_SHA256,RECIPE_HELPER:RECIPE_SHA256},"source_provenance":provenance,"selected_recipe_names":sorted(wanted),"selected_payload_bytes":payload_bytes,"recipe_reads":bank.reads,"recipe_payload_reads":bank.payload_reads,"source_payload_inventory_bytes":bank.bytes})
    first_prefix_difference=next((name for name in compare_ports if not pair["native"][name]["bitwise_exact"]),None)
    head_account=accounts["head"]["prefix_telescope"]
    hypotheses={"source":"Astra holistic07: predictions are diagnostic, never quality gates",
                "positioned_prefix_QKV_exact":{name:pair["native"][name]["bitwise_exact"] for name in ("q","k","v")},
                "first_retained_prefix_difference_in_declared_port_order":first_prefix_difference,
                "nonzero_historical_head":head_account["historical_energy"]>0,
                "equivalent_contrast_nonzero_positive_alignment":{
                    "B-minus-A":head_account["first_energy"]>0 and head_account["first_vs_historical_cosine"] is not None and head_account["first_vs_historical_cosine"]>0,
                    "C-minus-B":head_account["second_energy"]>0 and head_account["second_vs_historical_cosine"] is not None and head_account["second_vs_historical_cosine"]>0}}
    summary={"schema":"mdb-qwen-block0-attention-summary-v1","job_id":os.environ.get("MRUN_JOB_ID"),"request_sha256":config.get("request_sha256"),"contract":declared,"actual":{"native_block_calls":2,"native_rotary_calls":2,"sdpa_calls":len(sdpa_rows),"source_parameter_reads":bank.reads,"selected_payload_bytes":payload_bytes},"shared_prefix_rows":prefix,"pair_comparisons":pair,"triangle_accounts":accounts,"hypotheses":hypotheses,"mechanics_status":"complete","trend_status":"observed-uninterpreted","terminal_status":"unassessed","seconds":time.monotonic()-start,"limits":["Known original first-layer replay; new triangle consists of primitive interventions, not observed native source branches.","Only shared prefix supports cross-case causal-equivalence bridge; full triangle uses one fixed complete bank.","No semantic future-token reading, intrinsic gain, causal percentage or full-factorial interaction claim."]}
    save(output/"summary.json",summary)
    return summary


def worker(path):
    import faulthandler

    faulthandler.dump_traceback_later(120,repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    print("MDB_QWEN_BLOCK0_BOOTSTRAP",flush=True)
    config=read(path)
    if os.environ.get("SATURN_DEBUG_REQUEST"):
        request=read(os.environ["SATURN_DEBUG_REQUEST"])
        body={key:value for key,value in request.items() if key!="request_sha256"}
        require(hashlib.sha256(json.dumps(body,sort_keys=True,allow_nan=False).encode()).hexdigest()==request["request_sha256"] and request["config"]==config,"request/config differs")
        for name,digest in request["source_hashes"].items():
            require(sha(name)==digest,"executed source differs: "+name)
        config=dict(config,request_sha256=request["request_sha256"],_request_record=request)
    if "output" not in config:
        config=dict(config,output=str(Path(config["output_scratch"])/os.environ.get("MRUN_JOB_ID","local")/config.get("output_name","binding-minimal-result")))
    summary=run(config,progress=lambda row:print("MDB_QWEN_BLOCK0_PHASE "+json.dumps(row,sort_keys=True),flush=True))
    finalizer=runpy.run_path(str(Path(__file__).with_name("flux_image_tail_microscope.py")))
    ready=finalizer["finalize"](config["output"],job_id=summary["job_id"],request_sha256=summary["request_sha256"],**config.get("transport_limits",{"max_expanded_bytes":1<<30,"max_member_bytes":64<<20}))
    print("MDB_DIRECTORY_READY "+json.dumps(ready,sort_keys=True),flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=os.environ.get("SATURN_RUN_CONFIG"),required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)

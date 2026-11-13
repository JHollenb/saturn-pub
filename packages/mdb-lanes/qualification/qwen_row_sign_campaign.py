"""Inherited own-functional H18 row signs, sealed before three native blocks.

Only the original eleven block weights are restored. This conditional component
experiment is neither a new raw-prompt forecast nor a genuine public Qwen Act.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import inspect
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

SCHEMA = "mdb-qwen-row-sign-campaign-v1"
RESULT = "qwen-row-sign-result"
PROTOCOL_SHA256 = "d942001a79784ebb1bc5d5ed3f3946eaa03b4629f2fe13c7493afbaaf85b51e9"
SUBSET_SHA256 = "425502121b4091b10124c548c98127785420233b12e95c8f1d972a186026dccf"
BASE_SHA256 = "c5507557ad740a58341e4f99f4b384a9d27b82b6c268e2838469a19c2c8bfe76"
_base_path = Path(__file__).with_name("qwen_sign_shortcut_campaign.py")
if hashlib.sha256(_base_path.read_bytes()).hexdigest() != BASE_SHA256:
    raise ValueError("frozen campaign utility source differs")
BASE = runpy.run_path(str(_base_path))
HELPERS = {**BASE["HELPERS"], "qwen_sign_shortcut_campaign.py": BASE_SHA256,
    "qwen_row_sign_shortcut.py": "940c3cfaafeaeb3f0df5746b92b6a8b0878255769c4f230b2a53bedfc0110c1c"}
PATH_KEYS = ("qwen_config", "qwen_parameter_manifest", "parent_parameter_manifest", "qwen_source_binding",
    "retrieval_script", "selection_protocol", "parent_files", "parent_request", "parent_source", "parent_runtime", "parent_seal")
NATIVE_SHA256 = "fbdcfeeb1b54135ca67ba7df924da92f4b264e1252b517eea2de989289ebaeab"
SDPA_SHA256 = "b72c5c96ea070d594f36e774cfc799160a0ecb157353502a61eb3c5ea1e35b46"
ROPE_POLICY = BASE["ROPE_POLICY"]
ARMS = ("all-negative", "prefix-negative")
PORTS = ("head", "R", "M", "Hnext")
SOURCE_KEYS = {"h", "head", "R", "gate", "up", "product", "M", "Hnext", "input_ids", "attention_mask", "position_ids",
    "rope_buffer:model.rotary_emb.inv_freq", "rope_buffer:model.rotary_emb.original_inv_freq"}
FRONTIER_SCOPE = dict(kind="architectural", origin="inherited-functional-state", scope="conditional block18 hidden-state interface",
    raw_input_prediction=False, genuine_record_prefix_capture=False)
require, canonical, sha, read, save, inventory, verify_inventory, tensor_record = (
    BASE[name] for name in ("require", "canonical", "sha", "read", "save", "inventory", "verify_inventory", "tensor_record"))


@lru_cache(maxsize=1)
def helpers():
    result = {}
    for name, expected in HELPERS.items():
        path = Path(__file__).with_name(name)
        require(sha(path) == expected, "frozen helper differs: " + name)
        result[name] = runpy.run_path(str(path))
    return result


def contract():
    return dict(functional_blocks=2, shortcut_down_projections=2, native_blocks=3,
        functional_SDPA=2, shortcut_SDPA=0, native_SDPA=3,
        functional_linear=14, shortcut_linear=2, native_linear=21, native_authorizations=3,
        forecast_module_attempts=0, full_model=0, LM=0, embedding=0, tokenizer=0, FLUX=0, VAE=0, noise=0,
        component_hydrations=1, cuda_placements=1, cpu_offloads=1, genuine_qwen_acts=0)


def make_config(base):
    keys = (*PATH_KEYS, *(key + "_sha256" for key in PATH_KEYS), "parameter_root", "input_root")
    config = {key: base[key] for key in keys if key in base}
    config.setdefault("selection_protocol", "protocol.json")
    config.setdefault("selection_protocol_sha256", PROTOCOL_SHA256)
    config.setdefault("qwen_parameter_manifest_sha256", SUBSET_SHA256)
    config.update(schema=SCHEMA, helper_source_sha256=dict(HELPERS), qwen_native_source_sha256=NATIVE_SHA256,
        sdpa_source_sha256=SDPA_SHA256, attention_dispatch=dict(backend="sdpa", gqa_policy="enable_gqa"),
        qwen_rope_buffer_policy=dict(ROPE_POLICY), edit_index=18, horizon=19, fixed_split=256, sequence_rows=512,
        max_wall_seconds=240, forecast_transaction_bytes=64 << 20, num_threads=4,
        output_scratch="/mnt/big/mdb/research", output_name=RESULT,
        transport_limits=dict(max_expanded_bytes=2 << 30, max_member_bytes=64 << 20))
    return config


def validate_config(config):
    fixed = make_config(config)
    for key in fixed.keys() - set(PATH_KEYS) - {key + "_sha256" for key in PATH_KEYS} - {"parameter_root", "input_root"}:
        require(config.get(key) == fixed[key], "fixed configuration differs: " + key)
    for key in (*PATH_KEYS, "parameter_root", "input_root"):
        require(isinstance(config.get(key), str) and bool(config[key]), "missing source path: " + key)
    for key in PATH_KEYS:
        require(isinstance(config.get(key + "_sha256"), str) and len(config[key + "_sha256"]) == 64, "missing metadata identity: " + key)
    require(config["selection_protocol_sha256"] == PROTOCOL_SHA256 and config["qwen_parameter_manifest_sha256"] == SUBSET_SHA256,
        "frozen selection/subset differs")
    require(config["retrieval_script_sha256"] == BASE["RETRIEVER_SHA256"], "standalone retriever differs")
    require(not any(key.startswith(("flux_", "vae_", "tokenizer_")) for key in config), "unrelated component configuration forbidden")
    return contract()


def runtime_contract(config):
    import torch
    import transformers
    import transformers.integrations.sdpa_attention as sdpa
    import transformers.models.qwen3.modeling_qwen3 as native

    sources = {}
    for key, module in (("qwen_native_source_sha256", native), ("sdpa_source_sha256", sdpa)):
        path = Path(inspect.getfile(module))
        require(sha(path) == config[key], "installed native source differs: " + key)
        sources[key] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    require(torch.__version__.split("+")[0] == "2.13.0" and transformers.__version__ == "5.14.1", "native package versions differ")
    return dict(torch=str(torch.__version__), transformers=transformers.__version__, sources=sources,
        flags={**helpers()["qwen_sign_shortcut.py"]["execution_profile"](), "num_threads": torch.get_num_threads(),
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic}, selected_kernels="not independently profiled")


def inherited_sources(config, protocol, device):
    """Authenticate metadata first, then read exactly the selected thirteen arrays."""
    import numpy as np
    import torch

    files, request, seal = (read(config[key]) for key in ("parent_files", "parent_request", "parent_seal"))
    require(config["input_root"] == protocol["parent_raw_root"] and sha(config["parent_files"]) == protocol["parent_raw_manifest_sha256"]
        and sha(config["parent_seal"]) == protocol["parent_global_seal_sha256"], "parent raw/seal binding differs")
    require(BASE["request_identity"](request) == request["request_sha256"] == protocol["parent_request_sha256"]
        and files["job_id"] == protocol["parent_job"] and files["request_sha256"] == request["request_sha256"], "parent request identity differs")
    require(seal["schema"] == "mdb-global-forecast-seal-v1" and not seal["violations"] and seal["pre_release_work"]["native_calls"] == 0,
        "parent pretarget seal differs")
    for key, name in (("parent_source", "source.json"), ("parent_runtime", "runtime.json"), ("parent_parameter_manifest", "qwen-parameter-artifacts.json"),
            ("qwen_config", "qwen-config.json"), ("qwen_source_binding", "qwen-source-binding.json"), ("retrieval_script", "retrieve_qwen_weights.py")):
        require(sha(config[key]) == files["files"][name]["sha256"], "parent metadata lineage differs: " + key)
    require(set(protocol["source_arrays"]) == SOURCE_KEYS, "exact inherited source selection differs")
    loaded, references, signatures = {}, {}, {}
    for name, row in protocol["source_arrays"].items():
        relative = row["path"]
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe inherited path")
        expected = dict(bytes=row["bytes"], sha256=row["sha256"])
        require(files["files"][relative] == expected and seal["declarations"]["frontier"]["external_files"][relative] == expected,
            "selected source not in raw inventory/global pretarget seal")
        path = Path(config["input_root"]) / relative
        require(not path.is_symlink() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "inherited file identity differs")
        value = np.load(path, allow_pickle=False)
        require(list(value.shape) == row["shape"] and str(value.dtype) == row["stored_dtype"]
            and hashlib.sha256(value.tobytes(order="C")).hexdigest() == row["content_sha256"], "inherited array schema/payload differs")
        tensor = torch.from_numpy(value.copy())
        tensor = tensor.view(torch.bfloat16) if row["logical_dtype"] == "torch.bfloat16" else tensor
        require(str(tensor.dtype) == row["logical_dtype"] and list(tensor.stride()) == row["stride"] and bool(torch.isfinite(tensor).all()), "inherited tensor layout/finite contract differs")
        loaded[name], references[name] = tensor.to(device), dict(row, inherited=True, root="input_root")
        stat = path.stat()
        signatures[relative] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    def verify():
        for relative, signature in signatures.items():
            stat = (Path(config["input_root"]) / relative).stat()
            require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature, "inherited source changed")
    return loaded, references, verify


def identities(layer, edit_index):
    payloads = BASE["parameter_signature"](layer)
    prefix = f"model.layers.{edit_index}."
    return dict(payloads={prefix + key: row for key, row in payloads.items()},
        versions={prefix + name: dict(object_id=id(value), data_ptr=value.data_ptr(), version=value._version)
            for name, value in layer.named_parameters()})


def live_descriptor(value):
    """Bind heterogeneous integer, FP32 RoPE and BF16 source/context tensors."""
    import torch

    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype), device=str(value.device),
        content_sha256=hashlib.sha256(raw.tobytes()).hexdigest(), version=None if value.is_inference() else value._version,
        object_id=id(value), data_ptr=value.data_ptr())


@contextmanager
def census(phase, counts, calls, capture):
    from torch.nn import functional as F

    old_sdpa, old_linear = F.scaled_dot_product_attention, F.linear
    def sdpa(*args, **kwargs):
        counts[phase["name"] + "_SDPA"] += 1
        result = old_sdpa(*args, **kwargs)
        if phase["name"] == "native":
            mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
            require(mask is None and kwargs.get("is_causal") is True and kwargs.get("enable_gqa") is True
                and kwargs.get("dropout_p", 0.) == 0., "native delegated dispatch differs")
            calls.append(dict(arm=phase["arm"], QKV=[helpers()["qwen_h18_state_history.py"]["_tensor_metadata"](x) for x in args[:3]],
                attention_mask=None, is_causal=kwargs["is_causal"], enable_gqa=kwargs["enable_gqa"],
                dropout_p=kwargs.get("dropout_p", 0.), scale=kwargs.get("scale")))
            capture("head", result)
        return result
    def linear(*args, **kwargs):
        counts[phase["name"] + "_linear"] += 1
        return old_linear(*args, **kwargs)
    F.scaled_dot_product_attention, F.linear = sdpa, linear
    try:
        yield
    finally:
        F.scaled_dot_product_attention, F.linear = old_sdpa, old_linear


def execute_component(output, layer, sources, references, *, edit_index=18, split=256, source_identity=None,
                      source_metadata=None, verify_sources=lambda: None, before_execute=None,
                      max_wall_seconds=240, transaction_bytes=64 << 20, job_id=None, request_sha256=None):
    """Actual writer/guards/Tx shared by original CUDA execution and tiny CPU QA."""
    import numpy as np
    import torch
    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output, h = Path(output), helpers()
    graph, history, shortcut = (h[name] for name in ("flux_functional_qwen_prefix.py", "qwen_h18_state_history.py", "qwen_row_sign_shortcut.py"))
    length = sources["h"].shape[1]
    native_config = layer.self_attn.config
    require(set(sources) == SOURCE_KEYS and 0 < split < length and not layer.training and not native_config.attention_bias
        and native_config.attention_dropout == 0., "source/component geometry or evaluation policy differs")
    require(bool((sources["attention_mask"] == 1).all()), "inherited source must be all attended")
    parameters = {f"model.layers.{edit_index}." + name: value for name, value in layer.named_parameters()}
    profile = graph["_native_profile"](native_config, edit_index + 1)
    dispatch = graph["AttentionDispatch"].from_mapping(dict(backend="sdpa", gqa_policy="enable_gqa", source=NATIVE_SHA256))
    context = history["_context"](graph, sources["input_ids"], sources["attention_mask"], profile, dispatch, ROPE_POLICY, torch.bfloat16)
    require(context[0] is None and context[1] is True and torch.equal(context[2], sources["position_ids"]), "regenerated context differs")
    expected_inv = 1. / (profile["rope_theta"] ** (torch.arange(0, profile["head_dim"], 2, dtype=torch.float32) / profile["head_dim"]))
    require(all(torch.equal(sources[name].cpu(), expected_inv) for name in SOURCE_KEYS if name.startswith("rope_buffer:")), "inherited CPU FP32 RoPE policy differs")
    context_record = dict(position_ids=tensor_record(output, "context/position_ids.npy", context[2]),
        cos=tensor_record(output, "context/cos.npy", context[3]), sin=tensor_record(output, "context/sin.npy", context[4]),
        attention_mask=None, is_causal=True, enable_gqa=True, use_cache=False, past_key_values=None,
        source="regenerated from authenticated IDs/mask/config and original CPU FP32 RoPE policy; no native kwargs borrowed")
    identities_before = identities(layer, edit_index)
    save(output / "parameter-identity-before.json", identities_before)
    frontier = dict(source_metadata or {}, kind="inherited-own-functional-positive-ports", source_arrays=references,
        context=context_record, edit_index=edit_index, horizon=edit_index + 1, fixed_split=split, sequence_rows=length, genuine_qwen_act=False,
        transaction_frontier=dict(FRONTIER_SCOPE))
    save(output / "source-frontier.json", frontier)
    frozen_live = {key: live_descriptor(value) for key, value in sources.items()}
    counts = {key: 0 for key in ("functional_blocks", "shortcut_down_projections", "native_blocks", "functional_SDPA", "shortcut_SDPA", "native_SDPA",
        "functional_linear", "shortcut_linear", "native_linear", "native_authorizations", "forecast_module_attempts", "native_module_entries")}
    released, phase, events, sdpa_calls = dict(native=False), dict(name="functional", arm=None), [], []
    forecasts, shortcuts, responses, natives, inputs, entries = {}, {}, {}, {}, {}, []
    started = time.monotonic()
    def event(kind, **fields):
        require(time.monotonic() - started <= max_wall_seconds, "finite assay wall bound exceeded")
        row = dict(number=len(events), kind=kind, native_module_entries=counts["native_module_entries"], **fields)
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
    def verify():
        verify_sources()
        require(identities(layer, edit_index) == identities_before and all(live_descriptor(value) == frozen_live[key]
            for key, value in sources.items()), "source/component live identity changed")
    active_ports = {}
    def capture(name, value):
        require(bool(torch.isfinite(value).all()), "new port contains nonfinite values")
        active_ports[name] = tensor_record(output, f"native/{phase['arm']}/{name}.npy", value)
    def hook_entry(module, args, kwargs):
        require(released["native"], "native component entered before global seal")
        require(kwargs.get("attention_mask") is None and kwargs.get("past_key_values") is None and kwargs.get("use_cache") is False,
            "native context/cache differs")
        require(torch.equal(kwargs["position_ids"], context[2]) and all(torch.equal(x, y) for x, y in zip(kwargs["position_embeddings"], context[3:5])), "native consumed context differs")
        capture("h", args[0]); inputs[phase["arm"]] = active_ports["h"]
        consumed = dict(position_ids=live_descriptor(kwargs["position_ids"]),
            cos=live_descriptor(kwargs["position_embeddings"][0]), sin=live_descriptor(kwargs["position_embeddings"][1]),
            attention_mask=kwargs["attention_mask"], use_cache=kwargs["use_cache"], past_key_values=kwargs["past_key_values"])
        entries.append(dict(arm=phase["arm"], context=context_record, consumed_context=consumed, input=active_ports["h"]))
    handles = [layer.register_forward_pre_hook(hook_entry, with_kwargs=True),
        layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, args: capture("R", args[0])),
        layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args: capture("product", args[0]))]
    for name, module in (("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj), ("M", layer.mlp.down_proj)):
        handles.append(module.register_forward_hook(lambda _m, _args, value, name=name: capture(name, value)))
    try:
        with BASE["learned_guard"](layer, released, counts), census(phase, counts, sdpa_calls, capture), torch.inference_mode():
            for arm in ARMS:
                phase.update(name="shortcut", arm=arm)
                signs = torch.ones(length, dtype=torch.bfloat16, device=sources["h"].device)
                signs[:length if arm == "all-negative" else split] = -1
                permitted = torch.ones(length, length, dtype=torch.bool, device=signs.device).tril()
                attention = dict(backend="sdpa", gqa_policy="enable_gqa", query_rows="all-attended", mask_identity=references["attention_mask"]["content_sha256"],
                    positions_identity=references["position_ids"]["content_sha256"], rope_identity=hashlib.sha256(canonical(ROPE_POLICY)).hexdigest())
                positive = {key: sources[key] for key in ("R", "gate", "up")}
                weight = parameters[f"model.layers.{edit_index}.mlp.down_proj.weight"]
                p = shortcut["make_profile"](positive, weight, signs, permitted, source_identity=source_identity or {"synthetic": "1" * 64},
                    frontier_kind="own-functional-positive-ports", attention_profile=attention, positive_head=sources["head"], head_query_axis=1)
                prediction = shortcut["predict_row_sign"](positive, weight, signs, permitted, profile=p, positive_head=sources["head"], head_query_axis=1)
                aliases = dict(Rhat="R", Ghat="gate", Uhat="up", headhat="head")
                shortcuts[arm] = {key: value for key, value in prediction.items() if key not in ("ports", "support_closed")}
                shortcuts[arm].update(ports={aliases.get(key, key): tensor_record(output, f"shortcut/{arm}/{aliases.get(key, key)}.npy", value) for key, value in prediction["ports"].items()},
                    row_signs=tensor_record(output, f"shortcut/{arm}/row-signs.npy", signs), permitted=tensor_record(output, f"shortcut/{arm}/permitted.npy", permitted),
                    support_closed=tensor_record(output, f"shortcut/{arm}/support-closed.npy", prediction["support_closed"]))
                counts["shortcut_down_projections"] += 1
                phase["name"] = "functional"
                ports = {}
                def observe(_lane, name, value):
                    require(bool(torch.isfinite(value).all()), "functional port contains nonfinite values")
                    ports[name] = tensor_record(output, f"forecast/{arm}/{name}.npy", value)
                with history["_no_modules"](), history["_frozen_parameters"](parameters):
                    history["_block"](graph, signs.view(1, -1, 1) * sources["h"], parameters, profile, edit_index, context, dispatch, observe, arm)
                forecasts[arm] = dict(ports=ports, counts=dict(blocks=1, SDPA=1, linear=7))
                counts["functional_blocks"] += 1
                responses[arm] = {}
                for lane, bank in (("shortcut", shortcuts[arm]["ports"]), ("functional", ports)):
                    responses[arm][lane] = {}
                    for name in PORTS:
                        delta = BASE["load_array"](output, bank[name]).astype(np.float64) - sources[name].float().cpu().numpy().astype(np.float64)
                        responses[arm][lane][name] = dict(shape=list(delta.shape), elements=delta.size, dtype="float64",
                            orientation="edited-minus-own-positive", sha256=hashlib.sha256(delta.astype("<f8").tobytes(order="C")).hexdigest())
                event("forecast_arm_completed", arm=arm)
            save(output / "forecast.json", dict(positive={key: references[key] for key in ("h", "head", "R", "gate", "up", "product", "M", "Hnext")}, arms=forecasts))
            save(output / "shortcut.json", dict(arms=shortcuts))
            save(output / "forecast-responses.json", dict(orientation="edited-minus-own-positive", arms=responses))
            verify()
            sealed_files = inventory(output, exclude=("timeline.jsonl",))
            tx = ForecastTransaction(output / "transaction", model_identity="original-qwen3-decoder-layer", input_identity=references["h"]["content_sha256"],
                source_identity=hashlib.sha256(canonical(source_identity or {"synthetic": "1" * 64})).hexdigest(),
                frontier=dict(FRONTIER_SCOPE, conditional_frontier="own-functional-positive-H18-and-ports", external_files=sealed_files, inherited_sources=references),
                execution_scope="Two complete conditional functional blocks and two row shortcuts before three original standalone DecoderLayer native targets; no Qwen Act.",
                array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=3, max_bytes=transaction_bytes, max_files=30000, max_wall_ms=max_wall_seconds * 1000))
            tx.frontier("inherited-positive", dict(source_arrays=references, context=context_record, scope=FRONTIER_SCOPE))
            tx.predict("cohort", lambda _ctx: dict(predictions=dict(external_files=sealed_files, inherited_sources=references, response_hashes=responses, counts=dict(counts), native_targets_at_seal=0)))
            seal = tx.seal()
            event("cohort_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets_at_seal=0)
            def execute(_ctx):
                if before_execute is not None:
                    before_execute(output, sealed_files)
                verify_inventory(output, sealed_files); verify()
                released["native"] = True
                event("native_targets_released", seal_sha256=seal["seal_sha256"])
                for arm in ("baseline", *ARMS):
                    verify()
                    phase.update(name="native", arm=arm); active_ports.clear()
                    counts["native_authorizations"] += 1
                    event("native_component_authorized", arm=arm)
                    hidden = sources["h"] if arm == "baseline" else sources["h"] * torch.from_numpy(BASE["load_array"](output, shortcuts[arm]["row_signs"])).to(sources["h"].device, torch.bfloat16).view(1, -1, 1)
                    capture("Hnext", layer(hidden, attention_mask=None, position_ids=context[2], position_embeddings=context[3:5], past_key_values=None, use_cache=False))
                    natives[arm] = dict(ports=dict(active_ports)); counts["native_blocks"] += 1
                    event("native_component_completed", arm=arm)
                return dict(counts=counts, arms=list(natives))
            tx.execute("native-verification", execute); tx.close()
    finally:
        for handle in reversed(handles):
            handle.remove()
    verify(); verify_inventory(output, sealed_files)
    for key, expected in contract().items():
        if key in counts:
            require(counts[key] == expected, "observed primitive/component census differs: " + key)
    save(output / "native.json", dict(baseline=natives.pop("baseline"), arms=natives, head_to_canonical=[0, 2, 1, 3], context=context_record, inputs=inputs))
    save(output / "native-sdpa.json", dict(calls=sdpa_calls))
    save(output / "native-contract.json", dict(expected=contract(), observed=counts, block_entries=entries))
    save(output / "parameter-identity-after.json", identities(layer, edit_index))
    result = dict(schema=SCHEMA, job_id=job_id, request_sha256=request_sha256, counts=counts, contract=contract(), forecast_seal=seal["seal_sha256"],
        transaction=verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"], max_bytes=transaction_bytes), seconds=time.monotonic() - started,
        mechanics_status="completed", numerical_status="all rows retained; errors, premises and support are observations", genuine_qwen_act=False,
        frontier=dict(FRONTIER_SCOPE))
    save(output / "summary.json", result)
    return result


def run(config):
    import torch
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    validate_config(config)
    require(torch.cuda.is_available(), "original component requires CUDA")
    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=False)
    for key in PATH_KEYS:
        require(sha(config[key]) == config[key + "_sha256"], "metadata identity differs: " + key)
        destination = output / "metadata" / (key + Path(config[key]).suffix)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(config[key], destination)
    torch.set_num_threads(config["num_threads"]); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    runtime = runtime_contract(config); save(output / "runtime-contract.json", runtime); save(output / "plan.json", config)
    qconfig = Qwen3Config.from_dict(read(config["qwen_config"])); qconfig._attn_implementation = "sdpa"
    require((qconfig.hidden_size, qconfig.intermediate_size, qconfig.num_hidden_layers, qconfig.num_attention_heads, qconfig.num_key_value_heads, qconfig.head_dim) == (2560, 9728, 36, 32, 8, 128), "original geometry differs")
    selected, parent = (read(config[key]) for key in ("qwen_parameter_manifest", "parent_parameter_manifest"))
    original = [row for row in parent["artifacts"] if row["source"]["tensor"].startswith("model.layers.18.")]
    require(len(parent["artifacts"]) == 398 and len(original) == 11 and selected["artifacts"] == original
        and sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in original) == 201861632, "exact original eleven recipes differ")
    protocol = read(config["selection_protocol"])
    sources, references, verify_sources = inherited_sources(config, protocol, torch.device("cuda"))
    h = helpers(); bank = h["flux_source_receiver.py"]["RecipeBank"](selected, config["parameter_root"])
    with torch.device("meta"):
        layer = Qwen3DecoderLayer(qconfig, layer_idx=18).to(dtype=torch.bfloat16)
    layer = layer.to_empty(device=torch.device("cuda"))
    with torch.inference_mode():
        for row in original:
            name = row["source"]["tensor"]
            value = bank.load(h["flux_functional_qwen_prefix_worker.py"]["_recipe"](row, name))
            layer.get_parameter(name.removeprefix("model.layers.18.")).copy_(torch.from_numpy(value).to(device="cuda", dtype=torch.bfloat16))
    layer.eval().requires_grad_(False)
    before = identities(layer, 18)["payloads"]
    require(all(before[row["source"]["tensor"]] == dict(shape=row["source"]["shape"], dtype="torch.bfloat16", sha256=row["source"]["payload_sha256"]) for row in original), "restored original payload differs")
    recipe_io = dict(payload_reads=bank.payload_reads, load_requests=bank.reads, cache_hits=bank.cache_hits, parameter_payload_bytes=bank.bytes,
        peak_cached_payload_bytes=bank.peak_retained, checkpoint_files_verified_once=len(bank.signatures),
        checkpoint_bytes_verified_once=sum(signature[2] for signature in bank.signatures.values()))
    bank.bits.clear(); bank.retained = 0
    save(output / "source.json", dict(job_id=config["job_id"], request_sha256=config["request_sha256"], helper_source_sha256=HELPERS,
        frontier=dict(FRONTIER_SCOPE),
        source_hashes=config.get("request_source_hashes", {}), metadata={key: dict(path="metadata/" + key + Path(config[key]).suffix, sha256=config[key + "_sha256"]) for key in PATH_KEYS},
        selected_original_names=sorted(before), parameter_recipe_io=recipe_io, hydration=dict(component_hydrations=1, cuda_placements=1, strategy="meta BF16 direct device allocation; eleven unchanged original recipes")))
    result = execute_component(output, layer, sources, references, source_identity={**HELPERS, "qwen_native": NATIVE_SHA256, "sdpa_native": SDPA_SHA256},
        source_metadata=dict(input_root=config["input_root"], **{key: protocol[key] for key in ("parent_job", "parent_request_sha256", "parent_raw_manifest_sha256", "parent_global_seal_sha256", "source_case")}),
        verify_sources=verify_sources, job_id=config["job_id"], request_sha256=config["request_sha256"], max_wall_seconds=config["max_wall_seconds"], transaction_bytes=config["forecast_transaction_bytes"])
    layer.to("cpu")
    return result


def worker(config_path):
    print("qwen-row-sign bootstrap; inherited functional frontier; global seal before native component", flush=True)
    faulthandler.enable(); faulthandler.dump_traceback_later(90, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path); validate_config(config)
    request_path = os.environ.get("SATURN_DEBUG_REQUEST"); require(request_path is not None, "actual source-bound request required")
    request = read(request_path)
    require(BASE["request_identity"](request) == request.get("request_sha256") and request["config"] == config, "source request/config differs")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "executed source differs: " + name)
    job_id = os.environ["MRUN_JOB_ID"]; require(job_id.startswith("job-") and "/" not in job_id, "invalid actual job ID")
    derived = dict(config, request_source_hashes=request["source_hashes"], request_sha256=request["request_sha256"], job_id=job_id,
        output=str(Path(config["output_scratch"]) / job_id / config["output_name"]))
    run(derived)
    output = Path(derived["output"]); shutil.copyfile(request_path, output / "request.json"); shutil.copyfile(config_path, output / "declared-config.json")
    receipt = helpers()["flux_image_tail_microscope.py"]["finalize"](output, job_id=job_id, request_sha256=request["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(receipt).decode(), flush=True)
    faulthandler.cancel_dump_traceback_later()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)

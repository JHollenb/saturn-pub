"""Reusable inherited-source Qwen component panels with prospective custody.

Named arms, forecast callbacks and windows are supplied by a frozen protocol.
The panel never creates an own-positive block or a whole-model native target.
"""
from __future__ import annotations

import hashlib
import runpy
import time
from contextlib import contextmanager
from pathlib import Path

ROW_SHA256 = "fbd8eceaf1a9ef8917ecdc39a98ff960940aab9da8ecb4697c25c1a33d545f47"
_path = Path(__file__).with_name("qwen_row_sign_campaign.py")
if hashlib.sha256(_path.read_bytes()).hexdigest() != ROW_SHA256:
    raise ValueError("frozen qualified component mechanics differ")
ROW = runpy.run_path(str(_path))
HELPERS = {"qwen_row_sign_campaign.py": ROW_SHA256, **ROW["HELPERS"]}
require, canonical, sha, read, save, inventory, verify_inventory, tensor_record = (
    ROW[key] for key in ("require", "canonical", "sha", "read", "save", "inventory", "verify_inventory", "tensor_record"))
live_descriptor, identities = ROW["live_descriptor"], ROW["identities"]
FRONTIER_SCOPE = ROW["FRONTIER_SCOPE"]
PORTS = ("head", "R", "M", "Hnext")


def dense_layout(shape, stride):
    require(len(shape) == len(stride) > 0 and all(type(n) is int and n > 0 for n in shape)
        and all(type(n) is int and n >= 0 for n in stride), "invalid dense layout schema")
    extent = 1
    for step, size in sorted((step, size) for size, step in zip(shape, stride) if size > 1):
        require(step == extent, "layout must be nonoverlapping and dense")
        extent *= size
    return extent


def restore_array(path, row, device):
    """Restore declared strides from logical NPY coordinates without bit changes."""
    import numpy as np
    import torch

    path = Path(path)
    require(not path.is_symlink() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "inherited archive differs")
    array = np.load(path, allow_pickle=False)
    require(list(array.shape) == row["shape"] and str(array.dtype) == row["stored_dtype"]
        and hashlib.sha256(array.tobytes(order="C")).hexdigest() == row["content_sha256"], "inherited logical payload differs")
    value = torch.from_numpy(array.copy())
    value = value.view(torch.bfloat16) if row["logical_dtype"] == "torch.bfloat16" else value
    require(str(value.dtype) == row["logical_dtype"] and bool(torch.isfinite(value).all()), "inherited dtype/finite contract differs")
    dense_layout(row["shape"], row["stride"])
    restored = torch.empty_strided(row["shape"], row["stride"], dtype=value.dtype, device=device)
    restored.copy_(value)
    descriptor = live_descriptor(restored)
    wanted = torch.device(device)
    if wanted.type == "cuda" and wanted.index is None:
        wanted = torch.device("cuda", torch.cuda.current_device())
    require(descriptor["content_sha256"] == row["content_sha256"] and descriptor["stride"] == row["stride"]
        and restored.device == wanted, "dense restoration changed bits/layout/device")
    return restored, dict(loaded_logical=live_descriptor(value), restored=descriptor, value_bit_exact=True,
        policy="canonical NPY logical coordinates copied into explicitly allocated original dense strides")


def inherited_sources(config, protocol, device):
    files, request, seal, forecast = (read(config[key]) for key in ("parent_files", "parent_request", "parent_seal", "parent_forecast"))
    require(config["input_root"] == protocol["parent_raw_root"] and sha(config["parent_files"]) == protocol["parent_raw_manifest_sha256"]
        and sha(config["parent_seal"]) == protocol["parent_global_seal_sha256"], "parent inventory/seal identity differs")
    require(ROW["BASE"]["request_identity"](request) == request["request_sha256"] == protocol["parent_request_sha256"]
        and files["job_id"] == protocol["parent_job"] and files["request_sha256"] == request["request_sha256"], "parent request identity differs")
    require(seal["schema"] == "mdb-global-forecast-seal-v1" and not seal["violations"] and seal["pre_release_work"]["native_calls"] == 0,
        "parent pretarget seal differs")
    external = seal["declarations"]["frontier"]["external_files"]
    metadata = dict(parent_source="source.json", parent_runtime="runtime.json", parent_parameter_manifest="qwen-parameter-artifacts.json",
        qwen_config="qwen-config.json", qwen_source_binding="qwen-source-binding.json", retrieval_script="retrieve_qwen_weights.py",
        parent_forecast=f"cases/{protocol['source_case']}/forecast.json")
    for key, relative in metadata.items():
        require(sha(config[key]) == files["files"][relative]["sha256"] == external[relative]["sha256"], "parent metadata lineage differs: " + key)
    sources, references, restorations, signatures = {}, {}, {}, {}
    for name, row in protocol["source_arrays"].items():
        relative = row["path"]
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe inherited path")
        binding = dict(bytes=row["bytes"], sha256=row["sha256"])
        require(files["files"][relative] == binding and external[relative] == binding, "source absent from raw inventory/global seal")
        if name in forecast["ports"]["plus"]:
            require(row == forecast["ports"]["plus"][name], "source/layout descriptor differs from parent forecast")
        elif name == "position_ids":
            require(row == forecast["position_ids"], "position descriptor differs from parent forecast")
        path = Path(config["input_root"]) / relative
        sources[name], restorations[name] = restore_array(path, row, device)
        references[name] = dict(row, inherited=True, root="input_root")
        stat = path.stat()
        signatures[relative] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    def verify():
        for relative, signature in signatures.items():
            stat = (Path(config["input_root"]) / relative).stat()
            require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == signature, "inherited source file changed")
    return sources, references, dict(arrays=restorations), verify


def context_for(layer, sources, output, edit_index):
    import torch

    h = ROW["helpers"]()
    graph, history = h["flux_functional_qwen_prefix.py"], h["qwen_h18_state_history.py"]
    profile = graph["_native_profile"](layer.self_attn.config, edit_index + 1)
    dispatch = graph["AttentionDispatch"].from_mapping(dict(backend="sdpa", gqa_policy="enable_gqa", source=ROW["NATIVE_SHA256"]))
    require(bool((sources["attention_mask"] == 1).all()), "all-attended source required")
    context = history["_context"](graph, sources["input_ids"], sources["attention_mask"], profile, dispatch, ROW["ROPE_POLICY"], torch.bfloat16)
    require(context[0] is None and context[1] is True and torch.equal(context[2], sources["position_ids"]), "regenerated context differs")
    inv = 1. / (profile["rope_theta"] ** (torch.arange(0, profile["head_dim"], 2, dtype=torch.float32) / profile["head_dim"]))
    require(all(torch.equal(sources[key].cpu(), inv) for key in sources if key.startswith("rope_buffer:")), "CPU FP32 RoPE source differs")
    record = dict(position_ids=tensor_record(output, "context/position_ids.npy", context[2]),
        cos=tensor_record(output, "context/cos.npy", context[3]), sin=tensor_record(output, "context/sin.npy", context[4]),
        attention_mask=None, is_causal=True, enable_gqa=True, use_cache=False, past_key_values=None,
        source="regenerated from authenticated IDs/mask/config and CPU FP32 RoPE; no native kwargs borrowed")
    return graph, history, profile, dispatch, context, record


def response(root, edited, positive):
    import numpy as np

    delta = ROW["BASE"]["load_array"](root, edited).astype(np.float64) - positive.float().cpu().numpy().astype(np.float64)
    return dict(shape=list(delta.shape), elements=delta.size, dtype="float64", orientation="edited-minus-own-positive",
        sha256=hashlib.sha256(delta.astype("<f8").tobytes(order="C")).hexdigest())


@contextmanager
def primitives(phase, counts, calls, active_ports, capture):
    from torch.nn import functional as F

    old_sdpa, old_linear = F.scaled_dot_product_attention, F.linear
    def sdpa(*args, **kwargs):
        lane, arm = phase["lane"], phase["arm"]
        counts[lane + "_SDPA"] += 1
        mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
        require(mask is None and kwargs.get("is_causal") is True and kwargs.get("enable_gqa") is True
            and kwargs.get("dropout_p", 0.) == 0. and kwargs.get("scale") == args[0].shape[-1]**-0.5, "actual SDPA profile differs")
        operands = [live_descriptor(value) for value in args[:3]]
        for name, value, descriptor in zip(("q", "k", "v"), args[:3], operands):
            if lane == "native":
                capture(name, value)
            record = active_ports[name]
            require(record["content_sha256"] == descriptor["content_sha256"] and record["stride"] == descriptor["stride"]
                and record["shape"] == descriptor["shape"] and record["logical_dtype"] == descriptor["dtype"] and record["device_at_capture"] == descriptor["device"],
                "actual SDPA operand differs from saved port")
        result = old_sdpa(*args, **kwargs)
        if lane == "native":
            capture("head", result)
        calls.append(dict(lane=lane, arm=arm, QKV=[active_ports[name] for name in ("q", "k", "v")], operands=operands,
            attention_mask=None, is_causal=kwargs["is_causal"], enable_gqa=kwargs["enable_gqa"],
            dropout_p=kwargs.get("dropout_p", 0.), scale=kwargs.get("scale"), head_return=live_descriptor(result)))
        return result
    def linear(*args, **kwargs):
        counts[phase["lane"] + "_linear"] += 1
        return old_linear(*args, **kwargs)
    F.scaled_dot_product_attention, F.linear = sdpa, linear
    try:
        yield
    finally:
        F.scaled_dot_product_attention, F.linear = old_sdpa, old_linear


def execute_panel(output, layer, sources, references, protocol, *, callbacks, expected_counts, source_identity,
                  evaluation_counts=None, source_metadata=None, verify_sources=lambda: None, before_execute=None,
                  max_wall_seconds=240, transaction_bytes=64 << 20, job_id=None, request_sha256=None,
                  result_schema="mdb-qwen-component-panel-v1"):
    import torch
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    output, index = Path(output), protocol["edit_index"]
    require(bool(callbacks) and "native" not in callbacks and not layer.training and not layer.self_attn.config.attention_bias
        and layer.self_attn.config.attention_dropout == 0., "invalid forecast lanes/component policy")
    require(all(isinstance(name, str) and name.replace("-", "").isalnum() and callable(callback) for name, callback in callbacks.items()), "safe named callbacks required")
    arms = protocol["arms"]
    require(len({arm["id"] for arm in arms}) == len(arms) > 0, "unique named arms required")
    require(all(arm["id"].replace("-", "").isalnum() and arm["id"] != "baseline" for arm in arms), "unsafe or reserved arm identifier")
    length = sources["h"].shape[1]
    require(protocol["sequence_rows"] == length, "source row geometry differs")
    graph, history, profile, dispatch, context, context_record = context_for(layer, sources, output, index)
    parameters = {f"model.layers.{index}." + name: value for name, value in layer.named_parameters()}
    before = identities(layer, index)
    live_sources = {key: live_descriptor(value) for key, value in sources.items()}
    save(output / "parameter-identity-before.json", before)
    save(output / "source-frontier.json", dict(source_metadata or {}, source_arrays=references, context=context_record,
        frontier=FRONTIER_SCOPE, edit_index=index, horizon=index + 1, sequence_rows=length, genuine_qwen_act=False))
    counts = {key: 0 for key in expected_counts}
    counts.update(component_hydrations=1, native_authorizations=0, forecast_module_attempts=0, native_module_entries=0)
    for lane in (*callbacks, "native"):
        counts[lane + "_SDPA"] = counts[lane + "_linear"] = 0
    counts.update({lane + "_evaluations": 0 for lane in callbacks})
    released, phase, active_ports = dict(native=False), dict(lane=None, arm=None), {}
    events, calls, entries, forecasts, responses, natives, inputs = [], [], [], {}, {}, {}, {}
    started = time.monotonic()
    def event(kind, **fields):
        require(time.monotonic() - started <= max_wall_seconds, "finite panel wall bound exceeded")
        row = dict(number=len(events), kind=kind, native_module_entries=counts["native_module_entries"], **fields)
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
    def verify():
        verify_sources()
        require(identities(layer, index) == before and all(live_descriptor(value) == live_sources[key] for key, value in sources.items()),
            "live source/component identity changed")
    def capture(name, value):
        require(bool(torch.isfinite(value).all()), "new port contains nonfinite values")
        active_ports[name] = tensor_record(output, f"{phase['lane']}/{phase['arm']}/{name}.npy", value)
    def native_entry(_module, args, kwargs):
        require(released["native"] and kwargs.get("attention_mask") is None and kwargs.get("past_key_values") is None
            and kwargs.get("use_cache") is False, "native entry lacks seal/context/cache contract")
        actual = dict(position_ids=kwargs["position_ids"], cos=kwargs["position_embeddings"][0], sin=kwargs["position_embeddings"][1])
        consumed = {key: live_descriptor(value) for key, value in actual.items()}
        require(all(consumed[key]["content_sha256"] == context_record[key]["content_sha256"]
            and consumed[key]["shape"] == context_record[key]["shape"] and consumed[key]["stride"] == context_record[key]["stride"]
            and consumed[key]["dtype"] == context_record[key]["logical_dtype"] and consumed[key]["device"] == context_record[key]["device_at_capture"] for key in actual), "native consumed context differs")
        consumed.update(attention_mask=None, use_cache=False, past_key_values=None)
        capture("h", args[0]); inputs[phase["arm"]] = active_ports["h"]
        entries.append(dict(arm=phase["arm"], input=active_ports["h"], context=context_record, consumed_context=consumed))
    handles = [layer.register_forward_pre_hook(native_entry, with_kwargs=True),
        layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, args: capture("R", args[0])),
        layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args: capture("product", args[0]))]
    for name, module in (("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj), ("M", layer.mlp.down_proj)):
        handles.append(module.register_forward_hook(lambda _m, _a, value, name=name: capture(name, value)))
    try:
        with ROW["BASE"]["learned_guard"](layer, released, counts), primitives(phase, counts, calls, active_ports, capture), torch.inference_mode():
            for arm in arms:
                identifier = arm["id"]
                signs = torch.tensor(arm["row_signs"], dtype=torch.bfloat16, device=sources["h"].device)
                require(signs.shape == (length,) and bool(((signs == 1) | (signs == -1)).all()), "invalid predeclared signs")
                windows = arm["windows"]
                require(windows and windows[0][0] == 0 and windows[-1][1] == length and all(0 <= a < b <= length for a, b in windows)
                    and all(left[1] == right[0] for left, right in zip(windows, windows[1:])), "declared windows must partition rows")
                permitted = torch.ones(length, length, dtype=torch.bool, device=signs.device).tril()
                cross = permitted & (signs[:, None] != signs[None, :])
                closed = permitted.any(-1) & ~cross.any(-1)
                support = dict(total_rows=length, closed_rows=int(closed.sum()), permitted_cross_sign_edges=int(cross.sum()), use="diagnostic only; every row retained")
                require(support["closed_rows"] == arm["structural_support_closed_rows"] and support["permitted_cross_sign_edges"] == arm["permitted_cross_sign_edges"], "support instrument differs from protocol")
                row = dict(windows=windows, row_signs=tensor_record(output, f"arms/{identifier}/signs.npy", signs),
                    permitted=tensor_record(output, f"arms/{identifier}/permitted.npy", permitted), support_closed=tensor_record(output, f"arms/{identifier}/support.npy", closed), support=support, lanes={})
                for lane, callback in callbacks.items():
                    phase.update(lane=lane, arm=identifier); active_ports.clear()
                    with history["_no_modules"](), history["_frozen_parameters"](parameters):
                        metadata = callback(signs.view(1, -1, 1) * sources["h"], sources, parameters, profile, context, dispatch, signs, capture)
                    require(set(("h", "q", "k", "v", *PORTS)).issubset(active_ports), "forecast callback omitted required ports")
                    row["lanes"][lane] = dict(ports=dict(active_ports), metadata=metadata)
                    counts[lane + "_evaluations"] += 1
                    responses.setdefault(identifier, {})[lane] = {name: response(output, active_ports[name], sources[name]) for name in PORTS}
                    event("forecast_lane_completed", arm=identifier, lane=lane)
                forecasts[identifier] = row
            save(output / "forecast.json", dict(positive={key: references[key] for key in ("h", "q", "k", "v", *PORTS)}, arms=forecasts))
            save(output / "forecast-responses.json", dict(orientation="edited-minus-own-positive", arms=responses))
            # Physical lane counts are separate from protocol vocabulary.
            for lane, physical in (evaluation_counts or {}).items():
                require(lane in callbacks and physical in expected_counts, "evaluation count alias differs")
                counts[physical] = counts[lane + "_evaluations"]
            verify()
            sealed_files = inventory(output, exclude=("timeline.jsonl",))
            tx = ForecastTransaction(output / "transaction", model_identity="original-qwen3-decoder-layer", input_identity=references["h"]["content_sha256"],
                source_identity=hashlib.sha256(canonical(source_identity)).hexdigest(), frontier=dict(FRONTIER_SCOPE, external_files=sealed_files, inherited_sources=references),
                execution_scope="Named conditional component forecasts and complete responses before standalone native targets; no raw forecast or Qwen Act.",
                array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1 + len(arms), max_bytes=transaction_bytes, max_files=30000, max_wall_ms=max_wall_seconds * 1000))
            tx.frontier("inherited-positive", dict(source_arrays=references, context=context_record, scope=FRONTIER_SCOPE))
            tx.predict("cohort", lambda _ctx: dict(predictions=dict(external_files=sealed_files, inherited_sources=references, response_hashes=responses, counts=dict(counts), native_targets_at_seal=0)))
            seal = tx.seal(); event("cohort_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets_at_seal=0)
            def execute(_ctx):
                if before_execute is not None:
                    before_execute(output, sealed_files)
                verify_inventory(output, sealed_files); verify(); released["native"] = True
                event("native_targets_released", seal_sha256=seal["seal_sha256"])
                for arm in (None, *arms):
                    identifier = "baseline" if arm is None else arm["id"]
                    verify(); phase.update(lane="native", arm=identifier); active_ports.clear()
                    counts["native_authorizations"] += 1; event("native_component_authorized", arm=identifier)
                    hidden = sources["h"] if arm is None else sources["h"] * torch.tensor(arm["row_signs"], dtype=torch.bfloat16, device=sources["h"].device).view(1, -1, 1)
                    capture("Hnext", layer(hidden, attention_mask=None, position_ids=context[2], position_embeddings=context[3:5], past_key_values=None, use_cache=False))
                    natives[identifier] = dict(ports=dict(active_ports)); counts["native_block_calls"] = counts.get("native_block_calls", 0) + 1
                    event("native_component_completed", arm=identifier)
                return dict(counts=counts, arms=list(natives))
            tx.execute("native-verification", execute); tx.close()
    finally:
        for handle in reversed(handles):
            handle.remove()
    verify(); verify_inventory(output, sealed_files)
    for key, value in expected_counts.items():
        require(counts[key] == value, "actual physical census differs: " + key)
    save(output / "native.json", dict(baseline=natives.pop("baseline"), arms=natives, inputs=inputs, context=context_record, head_to_canonical=[0, 2, 1, 3]))
    save(output / "sdpa.json", dict(calls=calls)); save(output / "native-sdpa.json", dict(calls=[row for row in calls if row["lane"] == "native"]))
    save(output / "native-contract.json", dict(expected=expected_counts, observed=counts, block_entries=entries))
    save(output / "parameter-identity-after.json", identities(layer, index))
    result = dict(schema=result_schema, job_id=job_id, request_sha256=request_sha256, counts=counts, contract=expected_counts,
        frontier=FRONTIER_SCOPE, forecast_seal=seal["seal_sha256"], seconds=time.monotonic() - started,
        transaction=verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"], max_bytes=transaction_bytes),
        mechanics_status="completed", numerical_status="all rows retained; numerical/bit/signed-zero relations are observations", genuine_qwen_act=False)
    save(output / "summary.json", result)
    return result

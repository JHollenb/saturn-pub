"""Observed-prefix Qwen spectral/parity writes and conditional native readout.

One fixed source-only forecast cohort precedes all native suffix targets.
Ordinary first-token-fed futures are measured autonomous branches, not forecasts.
"""
from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import runpy
import time
from collections.abc import Mapping
from pathlib import Path

SCHEMA = "mdb-qwen-spectral-readout-campaign-v1"
RESULT = "qwen-spectral-readout-result"
PROMPTS = [
    dict(id="placard", text="A placard has the word red written in white paint. The word on the placard is"),
    dict(id="sequence", text="Complete the sequence: 3, 6, 12, 24,"),
    dict(id="code", text="Python:\ndef negate(value):\n    return"),
    dict(id="translation", text="Translate to German: river\nGerman:"),
]
DOSES = [0., .125, .25, .5, 1.]
HELPERS = {"qwen_spectral_readout_math.py": "11158458215504818147170b9d2b03522cc9a3d18f3440452c57ddd525a73eee",
    "qwen_spectral_parameter_recipes.py": "6130b4d70d59c37a26c6d8251f818148583d0070b95b5cdd97b05386815f5468",
    "flux_image_tail_microscope.py": "385c5701832b3b56e5cdb16cc370b0f25660bfca7cb3a9396a17cfe63e32d926"}
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
CHECKPOINT = {
    "config.json": dict(bytes=659, sha256="18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"),
    "generation_config.json": dict(bytes=242, sha256="e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6"),
    "merges.txt": dict(bytes=1671839, sha256="599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "model.safetensors": dict(bytes=988097824, sha256="fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
    "tokenizer.json": dict(bytes=7031645, sha256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    "tokenizer_config.json": dict(bytes=7305, sha256="5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
    "vocab.json": dict(bytes=2776833, sha256="ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def request_identity(request):
    """Match the staged request ABI, whose canonical JSON keeps default spaces."""
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


def authenticate_request(request, config, source_root):
    require(request.get("config") == config and request.get("request_sha256") == request_identity(request), "source-bound request/config identity differs")
    require(isinstance(request.get("source_hashes"), dict) and Path(__file__).name in request["source_hashes"], "worker must belong to request source closure")
    for name, expected in request["source_hashes"].items():
        require(Path(name).name == name and sha(Path(source_root) / name) == expected, "request source changed: " + name)
    require(all(request["source_hashes"].get(name) == expected for name, expected in HELPERS.items()), "qualified helper closure missing from request")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            result.update(block)
    return result.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=json_metadata); stream.write("\n")


def json_metadata(value):
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("unsupported JSON metadata: " + type(value).__name__)


def make_config(base):
    config = {key: base[key] for key in ("model_path", "retrieval_script", "output_scratch") if key in base}
    config.setdefault("model_path", "/mnt/big/llm-models/Qwen2.5-0.5B-Instruct")
    config.setdefault("retrieval_script", "retrieve_weights.py")
    config.setdefault("output_scratch", "/mnt/big/mdb/research")
    config.update(schema=SCHEMA, output_name=RESULT, prompts=copy.deepcopy(PROMPTS), doses=list(DOSES), consumer_strings=[" red", " blue"],
        expected_model_files=copy.deepcopy(CHECKPOINT), expected_weights_sha256="2b226fe4944dc744596b1142a15d6676160e9d06f245d31c1a36633f6c86baa3",
        helper_source_sha256=dict(HELPERS), retrieval_script_sha256=RETRIEVER_SHA256,
        device="cpu", dtype="float32", attention="eager", cpu_threads=4, seed=2718, lanczos_steps=128,
        max_prompt_tokens=64, max_prefix_ops=16384, continuation_tokens=8, max_future_cells=4,
        max_native_entries=320, max_wall_seconds=900, transaction_bytes=512 << 20,
        transport_limits=dict(max_expanded_bytes=1 << 30, max_member_bytes=512 << 20),
        scope="Conditional observed native prefix; one-token final MLP/readout equations; autonomous ordinary future tokens measured only")
    return config


def validate_config(config):
    fixed = make_config(config)
    for key in fixed.keys() - {"model_path", "retrieval_script", "output_scratch"}:
        require(config.get(key) == fixed[key], "fixed spectral contract differs: " + key)
    for key in ("model_path", "retrieval_script", "output_scratch"):
        require(isinstance(config.get(key), str) and bool(config[key]), "explicit source/output path required: " + key)
    require(not any(key.startswith(("donor", "flux", "vae")) for key in config), "unrelated/donor components forbidden")
    return dict(prefixes=4, native_MLP_bases=4, component_readout_targets=100, ordinary_prefill_adjudications=16,
        ordinary_clone_shams=4, spectral_antipode_targets=16, random_antipode_targets=8, autonomous_cached_calls_max=128,
        native_physical_entries_max=280, model_hydrations=1, full_vocabulary_forecast_rows=100)


def helpers():
    result = {}
    for name, expected in HELPERS.items():
        path = Path(__file__).with_name(name)
        require(sha(path) == expected, "qualified helper changed: " + name)
        result[name] = runpy.run_path(str(path))
    return result


def cache_arrays(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache, strict=True))


def cache_snapshot(cache):
    from mdb.forward_recording import tensor_descriptor

    return [dict(layer=index, key=tensor_descriptor(key), value=tensor_descriptor(value)) for index, (key, value) in enumerate(cache_arrays(cache))]


def parameter_snapshot(model):
    from mdb.forward_recording import tensor_descriptor

    seen, result = {}, {}
    for name, value in model.named_parameters(remove_duplicate=False):
        if id(value) not in seen:
            seen[id(value)] = tensor_descriptor(value)
        result[name] = seen[id(value)]
    return result


def metric(predicted, actual):
    import numpy as np

    left, right = np.asarray(predicted, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(), "finite comparable endpoint geometry required")
    delta = left-right
    rounded = left.astype(np.float32); native = right.astype(np.float32)
    shared_zero = (rounded == 0) & (native == 0)
    return dict(elements=left.size, max_abs=float(np.max(np.abs(delta))), rms=float(np.sqrt(np.mean(delta*delta))),
        orientation="predicted-minus-observed; FP32 raw-bit comparison rounds the independent reference once",
        signed_mean=float(np.mean(delta)), reference_rms=float(np.sqrt(np.mean(right*right))),
        changed_values=int(np.count_nonzero(delta)), rounded_FP32_bit_changed=int(np.count_nonzero(rounded.view(np.uint32) != native.view(np.uint32))),
        shared_zeros=int(np.count_nonzero(shared_zero)), signed_zero_defects=int(np.count_nonzero(shared_zero & (np.signbit(rounded) != np.signbit(native)))))


def prefix_with_cache(model, ids, mask, positions, cache, *, identity, execution, maximum):
    """Record tensor-only wrapper roots; DynamicCache is caller-owned closure state."""
    import torch

    from mdb.forward_prefix import record_prefix

    class Body(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.native = model.model; self.training = False

        def forward(self, token_ids, causal_mask, position_ids):
            return self.native(token_ids, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=cache, use_cache=True, return_dict=True)

    return record_prefix(Body(), (ids, mask, positions), stop_before=f"native.layers.{len(model.model.layers)-1}.post_attention_layernorm",
        model_identity=identity, input_identity=digest(dict(ids=ids.tolist(), positions=positions.tolist())), execution=execution, max_operations=maximum)


def campaign(model, prompt_inputs, consumer_ids, output, *, model_identity, source_identity, settings, decode=lambda ids: str(ids), prepare_output=None):
    """Shared explicit-input mechanics; worker alone supplies the frozen real plan."""
    import numpy as np
    import torch
    from transformers.cache_utils import DynamicCache

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    math = helpers()["qwen_spectral_readout_math.py"]
    layer = model.model.layers[-1]; hidden = model.config.hidden_size; eps = float(model.config.rms_norm_eps)
    require(all(parameter.dtype == torch.float32 and parameter.device.type == "cpu" for parameter in model.parameters()), "CPU FP32 original execution required")
    require(all(getattr(layer.mlp, name).bias is None for name in ("gate_proj", "up_proj", "down_proj")) and model.lm_head.bias is None, "bias-free original MLP/head required")
    weights = math["prepare"]({name: value.detach().numpy() for name, value in dict(postnorm=layer.post_attention_layernorm.weight,
        gate=layer.mlp.gate_proj.weight, up=layer.mlp.up_proj.weight, down=layer.mlp.down_proj.weight, finalnorm=model.model.norm.weight).items()})
    head = math["readonly"](model.lm_head.weight.detach().numpy(), np.float32)
    red, blue = consumer_ids
    require(type(red) is type(blue) is int and red != blue and 0 <= min(red, blue) and max(red, blue) < head.shape[0], "two distinct prebound vocabulary rows required")
    consumer = math["readonly"](weights["finalnorm"] * (head[red].astype(np.float64)-head[blue].astype(np.float64)))
    eos = model.generation_config.eos_token_id
    eos_tokens = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    if prepare_output is not None:
        prepare_output(output)
    started = time.monotonic()
    counts = dict(prefixes=0, native_MLP_bases=0, component_readouts=0, parity_targets=0,
        ordinary_prefills=0, ordinary_shams=0, cached_futures=0, physical_entries=0,
        final_postnorm_entries=0, final_MLP_entries=0, final_norm_entries=0, head_entries=0, denied_preseal_entries=0)
    phase = dict(released=False, prefix_active=False, lane="source")
    events = []

    def event(kind, **detail):
        row = dict(number=len(events), kind=kind, **detail); events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")

    def charge(name):
        counts[name] += 1; counts["physical_entries"] += 1
        require(counts["physical_entries"] <= settings["max_native_entries"] and time.monotonic()-started < settings["max_wall_seconds"], "finite native entry/wall allowance exhausted")

    def root_guard(_module, _args):
        if not phase["released"] or phase["lane"] not in ("ordinary", "sham", "future"):
            counts["denied_preseal_entries"] += 1
            raise RuntimeError("ordinary LM entry outside released declared consumer")
        charge({"ordinary": "ordinary_prefills", "sham": "ordinary_shams", "future": "cached_futures"}[phase["lane"]])

    def tail_guard(name):
        def guard(_module, _args):
            if not phase["released"]:
                counts["denied_preseal_entries"] += 1
                raise RuntimeError("final native suffix entry before global seal")
            counts[name] += 1
        return guard

    def body_guard(_module, _args):
        require(phase["released"] or phase["prefix_active"], "native body outside declared prefix/released target")

    hooks = [model.register_forward_pre_hook(root_guard), model.model.register_forward_pre_hook(body_guard)]
    hooks += [module.register_forward_pre_hook(tail_guard(name)) for name, module in
        (("final_postnorm_entries", layer.post_attention_layernorm), ("final_MLP_entries", layer.mlp), ("final_norm_entries", model.model.norm), ("head_entries", model.lm_head))]
    originals = parameter_snapshot(model)
    prefixes, parents, parent_signatures = {}, {}, {}
    try:
        search = math["lanczos"](math["even_action"](weights, consumer), hidden, steps=settings["lanczos_steps"], seed=settings["seed"])
        for prompt in prompt_inputs:
            key, ids = prompt["id"], prompt["ids"]
            require(key not in prefixes and ids.ndim == 2 and ids.shape[0] == 1 and 1 <= ids.shape[1] <= settings["max_prompt_tokens"], "unique bounded raw context required")
            positions = torch.arange(ids.shape[1]).unsqueeze(0)
            mask = torch.triu(torch.full((1, 1, ids.shape[1], ids.shape[1]), torch.finfo(torch.float32).min), diagonal=1)
            cache = DynamicCache(config=model.config); phase["prefix_active"] = True; charge("prefixes")
            try:
                capture = prefix_with_cache(model, ids, mask, positions, cache, identity=model_identity,
                    execution=dict(backend="cpu", dtype="float32", attention="eager", threads=torch.get_num_threads()), maximum=settings["max_prefix_ops"])
            finally:
                phase["prefix_active"] = False
            require(cache.get_seq_length() == ids.shape[1] and len(cache_arrays(cache)) == len(model.model.layers), "complete caller-owned prompt KV required")
            save(output / "prefixes" / (key+".json"), capture.trace)
            cut = capture.output["boundary"]["args"][0][:, -1:, :].detach().contiguous()
            parent_signatures[key] = cache_snapshot(cache); parents[key] = cache
            prefixes[key] = dict(prompt=dict(id=key, text=prompt["text"]), cut=cut.numpy().copy(), token_ids=ids.numpy().copy(),
                positions=positions.numpy().copy(), mask=mask.numpy().copy(), cut_descriptor=tensor_descriptor(cut),
                trace_fingerprint=capture.trace["fingerprint"], prefix_contract=capture.trace["metadata"]["prefix_contract"],
                cache=[dict(key=a.numpy().copy(), value=b.numpy().copy()) for a, b in cache_arrays(cache)], cache_descriptors=parent_signatures[key])
            event("genuine_prefix_complete", prompt=key, cache_digest=digest(parent_signatures[key]), final_suffix_entries=counts["final_MLP_entries"])
            del capture
        require(all(counts[name] == 0 for name in ("ordinary_prefills", "ordinary_shams", "cached_futures", "final_postnorm_entries", "final_MLP_entries", "final_norm_entries", "head_entries", "denied_preseal_entries")), "pretarget suffix exclusion differs")
        transaction = ForecastTransaction(output / "transaction", model_identity=model_identity,
            input_identity=digest({key: value["cut_descriptor"] for key, value in prefixes.items()}), source_identity=source_identity,
            frontier=dict(kind="architectural", origin="genuine-native-prefix", scope="last-token before final MLP; full prompt KV observed; conditional suffix only"),
            execution_scope="Explicit guarded CPU FP32 eager model; independent NumPy FP64 equations; no parameter bank",
            budgets=dict(max_bytes=settings["transaction_bytes"], max_wall_ms=settings["max_wall_seconds"]*1000, max_native_calls=settings["max_native_entries"]))
        with transaction:
            transaction.frontier("prefixes", prefixes)
            transaction.frontier("spectral-source", dict(search=search, consumer=consumer, consumer_ids=list(consumer_ids),
                factor_identity={name: dict(shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest()) for name, value in weights.items()},
                full_head=dict(shape=list(head.shape), dtype=str(head.dtype), sha256=hashlib.sha256(memoryview(head).cast("B")).hexdigest()), pretarget_counts=dict(counts)))

            def predict(context):
                rows, payloads = {}, {}
                for index, (key, prefix) in enumerate(context.frontiers["prefixes"].items()):
                    raw = prefix["cut"].reshape(-1); rho = float(np.linalg.norm(raw.astype(np.float64)))
                    random = np.random.default_rng(settings["seed"]+index+1).standard_normal(hidden); random /= np.linalg.norm(random)
                    directions = {name: value["direction"] for name, value in search["extremes"].items()}; directions["random"] = random
                    parity = {name: math["antipodal"](direction, rho, weights, eps, consumer) for name, direction in directions.items()}
                    writes = {name: parity[name]["write"] for name in ("plus", "minus")}
                    original = math["mlp"](raw, weights, eps)
                    grid = math["grid_forecast"](raw.astype(np.float64)+original["delta"], writes, weights["finalnorm"], head, eps, doses=settings["doses"])
                    selection = math["select_futures"](grid)
                    # Complete ties can contain the entire vocabulary. Compact ID
                    # arrays bound storage even in that case; selection is unchanged.
                    for row in grid["rows"].values():
                        for name in ("co_maximizers", "logit_co_maximizers"):
                            row[name] = np.asarray(row[name], dtype=np.uint32)
                    rows[key] = dict(base=original, parity=parity, grid=grid, selection=selection, payload_geometry=math["payload_geometry"](writes))
                    for name, write in writes.items():
                        payloads[key+"."+name] = write
                    for name, value in parity.items():
                        for sign in ("positive", "negative"):
                            payloads[key+"."+name+"."+sign] = value[sign+"_input"]
                return dict(predictions=dict(prompts=rows, directions_once=True, continuation_forecast=False,
                    scope="Own observed prefix, full-vocabulary one-token readout only; all future-cell selection uses these predictions"), payloads=payloads)

            transaction.predict("cohort", predict)
            require(parameter_snapshot(model) == originals, "native parameters changed before global seal")
            seal = transaction.seal(); phase["released"] = True
            event("global_forecasts_sealed", seal_sha256=seal["seal_sha256"], pretarget_counts=dict(counts))

            def execute(context):
                predicted = context.predictions["cohort"]["prompts"]; payloads = context.payloads["cohort"]
                observed = {}
                for key, prefix in context.frontiers["prefixes"].items():
                    cut = torch.from_numpy(prefix["cut"].copy()); phase["lane"] = "component"; charge("native_MLP_bases")
                    with torch.inference_mode():
                        normalized = layer.post_attention_layernorm(cut); delta = layer.mlp(normalized); base = cut+delta
                    values = predicted[key]; writes = {name: torch.from_numpy(payloads[key+"."+name].copy()) for name in ("plus", "minus")}
                    component, parity, ordinary = {}, {}, {}
                    for name in ("plus", "minus", "random"):
                        for sign in ("positive", "negative"):
                            charge("parity_targets")
                            operand = torch.from_numpy(payloads[key+"."+name+"."+sign].copy()).reshape(1, 1, hidden)
                            with torch.inference_mode():
                                postnorm = layer.post_attention_layernorm(operand); update = layer.mlp(postnorm)
                            reference = values["parity"][name][sign]
                            parity[name+"."+sign] = dict(input=tensor_descriptor(operand), postnorm=postnorm.numpy().copy(), delta=update.numpy().copy(),
                                errors=dict(postnorm=metric(reference["postnorm"], postnorm.numpy().reshape(-1)), delta=metric(reference["delta"], update.numpy().reshape(-1))))
                    native_even = {}
                    for name in ("plus", "minus", "random"):
                        even = (parity[name+".positive"]["delta"].astype(np.float64)+parity[name+".negative"]["delta"].astype(np.float64))/2
                        native_even[name] = dict(value=even, consumer=float(consumer @ even.reshape(-1)),
                            full_order_error=metric(values["parity"][name]["even"], even.reshape(-1)),
                            bilinear_error=metric(values["parity"][name]["even_bilinear"], even.reshape(-1)))
                    for row_id, row in values["grid"]["rows"].items():
                        charge("component_readouts")
                        with torch.inference_mode():
                            carrier = base if row["alpha"] == row["beta"] == 0 else (base+row["alpha"]*writes["plus"])+row["beta"]*writes["minus"]
                            final = model.model.norm(carrier); logits = model.lm_head(final)[0, 0]
                        actual = logits.numpy().copy()
                        baseline = actual if row_id == "0,0" else component["0,0"]["logits"]
                        predicted_baseline = values["grid"]["rows"]["0,0"]["logits"]
                        component[row_id] = dict(alpha=row["alpha"], beta=row["beta"], logits=actual,
                            carrier=carrier.numpy().copy(), final_norm=final.numpy().copy(), greedy_token=int(logits.argmax()),
                            co_maximizers=np.flatnonzero(actual == actual.max()).astype(np.uint32), winner_text=decode([int(logits.argmax())]),
                            consumer_contrast=float(actual[red])-float(actual[blue]),
                            error=metric(row["logits"], actual), response_error=metric(row["logits"]-predicted_baseline, actual.astype(np.float64)-baseline.astype(np.float64)))
                    selected = values["selection"]["rows"]
                    for row_id in [*selected, "sham"]:
                        row = values["grid"]["rows"]["0,0" if row_id == "sham" else row_id]
                        received = {}

                        def observe_cut(_module, args):
                            received["cut"] = args[0][:, -1:, :].detach().clone()

                        def write(_module, args):
                            value = args[0]; received["before"] = value[:, -1:, :].detach().clone()
                            if row_id == "sham":
                                changed = value.clone()
                            elif row["alpha"] == row["beta"] == 0:
                                changed = value
                            else:
                                changed = value.clone(); changed[:, -1:, :] = (value[:, -1:, :]+row["alpha"]*writes["plus"])+row["beta"]*writes["minus"]
                            received["after"] = changed[:, -1:, :].detach().clone()
                            received["earlier_rows_exact"] = bool(torch.equal(changed[:, :-1], value[:, :-1]))
                            return (changed,)

                        handles = [layer.post_attention_layernorm.register_forward_pre_hook(observe_cut), model.model.norm.register_forward_pre_hook(write)]
                        ids = torch.from_numpy(prefix["token_ids"].copy()); phase["lane"] = "sham" if row_id == "sham" else "ordinary"
                        try:
                            with torch.inference_mode():
                                first = model(ids, attention_mask=torch.from_numpy(prefix["mask"].copy()), position_ids=torch.from_numpy(prefix["positions"].copy()),
                                    past_key_values=DynamicCache(config=model.config), use_cache=True, return_dict=True)
                        finally:
                            for handle in handles:
                                handle.remove()
                        actual = first.logits[0, -1].detach().numpy().copy(); token = int(first.logits[0, -1].argmax())
                        own_cache = cache_snapshot(first.past_key_values)
                        future_rows = []; private = copy.deepcopy(first.past_key_values)
                        require(all(a.data_ptr() != b.data_ptr() for left, right in zip(cache_arrays(private), cache_arrays(first.past_key_values), strict=True) for a, b in zip(left, right, strict=True)), "private future cache aliases its ordinary parent")
                        previous = token
                        if row_id != "sham":
                            phase["lane"] = "future"
                            for step in range(settings["continuation_tokens"]):
                                before_length = private.get_seq_length()
                                with torch.inference_mode():
                                    future = model(torch.tensor([[previous]]), past_key_values=private, use_cache=True, return_dict=True)
                                private = future.past_key_values; following = int(future.logits[0, -1].argmax())
                                future_rows.append(dict(step=step+1, consumed_token=previous, greedy_token=following,
                                    emitted_eos=following in eos_tokens,
                                    cache_length_before=before_length, cache_length_after=private.get_seq_length(),
                                    logits=tensor_descriptor(future.logits[:, -1]), text=decode([following]), authority="Autonomous unchanged native model, ordinary own-KV history; no numerical forecast"))
                                previous = following
                        require(cache_snapshot(first.past_key_values) == own_cache, "private future mutated its ordinary parent")
                        reference_id = "0,0" if row_id == "sham" else row_id
                        ordinary[row_id] = dict(logits=actual, greedy_token=token, generated_tokens=[token]+[value["greedy_token"] for value in future_rows],
                            first_token_is_eos=token in eos_tokens, eos_policy="Observed; declared fixed future budget continues after EOS",
                            generated_text=decode([token]+[value["greedy_token"] for value in future_rows]), continuation=future_rows,
                            prefix_cut_error=metric(prefix["cut"], received["cut"].numpy()),
                            component_vs_ordinary=metric(component[reference_id]["logits"], actual), predicted_vs_ordinary=metric(row["logits"], actual),
                            original_receiving_state=received["before"].numpy().copy(), written_state=received["after"].numpy().copy(),
                            earlier_rows_exact=received["earlier_rows_exact"], ordinary_parent_cache=own_cache,
                            cache_exact_vs_observed_prefix=own_cache == parent_signatures[key], future_cache_authority="Always private copy of this ordinary full-prefill result; no conditional fallback")
                    observed[key] = dict(component=component, parity=parity, ordinary=ordinary, selection=values["selection"],
                        native_even=native_even,
                        native_base=dict(postnorm=normalized.numpy().copy(), delta=delta.numpy().copy(), residual=base.numpy().copy()),
                        sham_vs_baseline=metric(ordinary["sham"]["logits"], ordinary["0,0"]["logits"]))
                    require(cache_snapshot(parents[key]) == parent_signatures[key], "immutable observed prefix KV changed")
                    event("prompt_targets_complete", prompt=key, selected_future_rows=list(selected), counts=dict(counts))
                return dict(prompts=observed, counts=dict(counts), scope="Native component suffixes and ordinary adjudication are separate; autonomous continuation observed only")

            transaction.execute("native", execute)
        require(parameter_snapshot(model) == originals, "native parameters changed during assay")
        verified = verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"])
        predictions = transaction.prediction("cohort"); observations = transaction.observation("native")
        summary = dict(schema=SCHEMA, model_identity=model_identity, counts=dict(counts), forecast_seal=seal,
            job_id=settings.get("job_id"), request_sha256=settings.get("request_sha256"),
            input_identity=digest({key: value["cut_descriptor"] for key, value in prefixes.items()}),
            transaction=verified, prompts=len(prefixes), consumer_ids=list(consumer_ids), seconds=time.monotonic()-started,
            limitations=["Genuine observed prefix, conditional own-state suffix; no raw-prompt or full-continuation forecast.",
                "Component1-token MLP/readout and ordinary prefill retain separate finite profiles/errors.",
                "Strongest found Ritz directions; payload Gram diagnoses effective dimension.", "All outcomes/counterexamples retained; no numerical success or token-divergence gate."])
        save(output / "summary.json", summary)
        return dict(summary=summary, predictions=predictions, observations=observations)
    finally:
        for hook in hooks:
            hook.remove()


def checkpoint_inventory(root, expected):
    """Only the declared loader files, authenticated before any model hydration."""
    root = Path(root); inventory, signatures = {}, {}
    require(root.is_dir(), "local original checkpoint root required")
    require(sorted(path.name for path in root.glob("*.safetensors")) == sorted(name for name in expected if name.endswith(".safetensors")), "original safetensor file set differs")
    for name, row in expected.items():
        require(Path(name).name == name, "flat checkpoint file name required")
        path = root / name
        require(path.is_file() and not path.is_symlink(), "original regular checkpoint file required: " + name)
        before = path.stat(); actual = dict(bytes=before.st_size, sha256=sha(path)); after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "checkpoint changed while hashing")
        require(actual == row, "original checkpoint identity differs: " + name)
        inventory[name] = actual; signatures[name] = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    return inventory, signatures


def tokenize(tokenizer, config):
    import torch

    encodings = {text: tokenizer.encode(text, add_special_tokens=False) for text in config["consumer_strings"]}
    require(all(len(value) == 1 and type(value[0]) is int for value in encodings.values()), "both fixed consumer strings must be single tokens")
    consumers = [encodings[text][0] for text in config["consumer_strings"]]
    require(len(set(consumers)) == 2, "fixed consumer strings must name distinct tokens")
    prompts = []
    for prompt in config["prompts"]:
        values = tokenizer.encode(prompt["text"], add_special_tokens=False)
        require(1 <= len(values) <= config["max_prompt_tokens"] and all(type(value) is int for value in values), "bounded nonempty raw prompt tokenization required")
        prompts.append(dict(prompt, ids=torch.tensor([values], dtype=torch.int64)))
    return prompts, consumers, encodings


def publish_accounts(result, output, *, execution, source_identity, run_id):
    """Existing public records from retained arrays; no numerical model execution."""
    import numpy as np

    from mdb.computation_atlas import ComputationAtlas
    from mdb.equation_comparison import EquationCatalog, compare_equations

    output = Path(output); catalog = EquationCatalog(output / "catalog"); rows, effects, future, exceptions = [], {}, {}, {}
    references = [dict(path="summary.json", sha256=sha(output / "summary.json")), dict(path="transaction/seal.json", sha256=sha(output / "transaction/seal.json"))]
    summary = result["summary"]
    for key, predicted in result["predictions"]["prompts"].items():
        observed = result["observations"]["prompts"][key]
        for cell, value in predicted["grid"]["rows"].items():
            actual = observed["component"][cell]
            context = dict(model_identity=summary["model_identity"], source_identity=source_identity, execution=execution,
                horizon="conditional-last-token-final-readout", prompt=key, input_identity=summary["input_identity"],
                alpha=value["alpha"], beta=value["beta"], receiving_port="model.norm:input:last-token", cache="genuine-observed-prompt; private ordinary future", clock="after-one-global-forecast-seal")
            report = compare_equations(dict(logits=value["logits"]), dict(logits=actual["logits"]), decision_rules=dict(logits="argmax"),
                baseline=dict(logits=observed["component"]["0,0"]["logits"]), predicted_baseline=dict(logits=predicted["grid"]["rows"]["0,0"]["logits"]), max_elements=len(value["logits"]))
            record = catalog.record(law_id="bias-free-even-MLP-two-write-RMS-readout", route_id="observed-final-MLP-to-full-head",
                binding={name: context[name] for name in ("model_identity", "source_identity", "execution", "input_identity", "horizon")} | dict(frontier="genuine-native-final-pre-MLP; own-state additive final-norm input"),
                report=report, conditions=[dict(field=name, op="eq", value=item) for name, item in context.items()], tested_context=context,
                readers=["full-vocabulary-greedy", "fixed-red-minus-blue"], evidence=dict(provenance=references, prompt=key, cell=cell, conditional=True, raw_prompt_forecast=False, continuation_forecast=False))
            identifier = record["fingerprint"]
            effects[identifier] = [dict(consumer_id="fixed-red-minus-blue", contrast_id="dose-minus-own-zero-zero", units="native-FP32-logit",
                value=actual["consumer_contrast"]-observed["component"]["0,0"]["consumer_contrast"], provenance=references)]
            selected = observed["ordinary"].get(cell)
            future[identifier] = dict(status="measured-autonomous-native" if selected else "not-measured", provenance=references if selected else [])
            if selected:
                future[identifier].update(generated_tokens=selected["generated_tokens"], generated_text=selected["generated_text"], numerical_forecast=False,
                    authority=selected["future_cache_authority"], component_vs_ordinary=dict(selected["component_vs_ordinary"]))
            exceptions[identifier] = [dict(claim_id="unconditional-FP32-byte-identity", status="finite-error-retained",
                error=dict(actual["error"]), decision_changed=int(actual["greedy_token"] != value["greedy_token"]), provenance=references)]
            rows.append(dict(prompt=key, cell=cell, record_id=identifier, predicted_token=value["greedy_token"], native_token=actual["greedy_token"],
                predicted_co_maximizers=value["co_maximizers"].tolist(), native_co_maximizers=actual["co_maximizers"].tolist(), winner_text=actual["winner_text"],
                endpoint_error=actual["error"], response_error=actual["response_error"], native_contrast=actual["consumer_contrast"], selected_future=bool(selected)))
    receipts = ComputationAtlas(output / "atlas").ingest(output / "catalog", [value["record_id"] for value in rows],
        provenance=dict(run_id=run_id, artifacts=references), signed_effects=effects, continuation=future, counterexamples=exceptions)
    contexts = {}
    for key, value in result["predictions"]["prompts"].items():
        geometry = value["payload_geometry"]
        contexts[key] = dict(selection=value["selection"], payload_geometry={name: item.tolist() if isinstance(item, np.ndarray) else item for name, item in geometry.items()},
            parity={name: dict(consumer_even=pair["consumer_even"], consumer_bilinear=pair["consumer_bilinear"], parity_error=pair["parity_error"], raw_norm=pair["requested_raw_norm"], realized_raw_norm=pair["realized_raw_norm"]) for name, pair in value["parity"].items()},
            ordinary={name: {field: item for field, item in row.items() if field not in ("logits", "original_receiving_state", "written_state")} for name, row in result["observations"]["prompts"][key]["ordinary"].items()})
    report = dict(schema="mdb-qwen-spectral-readout-accounts-v1", rows=rows, prompts=contexts, record_ids=[value["record_id"] for value in rows],
        atlas_ids=[value["fingerprint"] for value in receipts], mechanics_status="valid", terminal_status="unassessed",
        coverage=dict(prompt_grid_rows=len(rows), full_vocabulary=True, autonomous_selection="own presealed forecasts", independent_contexts=len(contexts)),
        limitations=summary["limitations"])
    save(output / "accounts.json", report)
    return report


def worker(config_path):
    """Finite source-bound original worker; no checkpoint acquisition or retry."""
    print("qwen-spectral bootstrap; observed prefix, one global seal, finite CPU suffixes", flush=True)
    config_path = Path(config_path).resolve(); config = json.loads(config_path.read_text()); planned = validate_config(config)
    request_path = Path(os.environ["SATURN_DEBUG_REQUEST"]).resolve(); request = json.loads(request_path.read_text())
    authenticate_request(request, config, config_path.parent)
    require(sha(config["retrieval_script"]) == RETRIEVER_SHA256, "standalone retriever changed")
    job_id = os.environ["MRUN_JOB_ID"]; require(job_id.startswith("job-") and Path(job_id).name == job_id, "actual job identity required")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(config["cpu_threads"])
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(config["cpu_threads"]); torch.manual_seed(config["seed"]); torch.use_deterministic_algorithms(True)
    faulthandler.enable(); faulthandler.dump_traceback_later(60, repeat=True)
    try:
        checkpoint = Path(config["model_path"]); inventory, signatures = checkpoint_inventory(checkpoint, config["expected_model_files"])
        weights_sha256 = digest({name: value["sha256"] for name, value in inventory.items() if name.endswith(".safetensors")})
        require(weights_sha256 == config["expected_weights_sha256"], "original weight-file identity differs")
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
        layer = model.model.layers[-1]; eps = float(model.config.rms_norm_eps)
        require(model.config.model_type == "qwen2" and model.config.hidden_act == "silu" and model.config._attn_implementation == "eager" and model.config.hidden_size <= 2048 and len(model.model.layers) <= 32, "bound Qwen2 SiLU/eager geometry required")
        require(type(layer.post_attention_layernorm).__name__ == type(model.model.norm).__name__ == "Qwen2RMSNorm" and np.isfinite(eps) and eps > 0
            and layer.post_attention_layernorm.variance_epsilon == model.model.norm.variance_epsilon == eps, "native RMSNorm type/epsilon differs")
        prompts, consumers, encodings = tokenize(tokenizer, config)
        identity = dict(model_type="qwen2", weights_sha256=weights_sha256, checkpoint_files_sha256=digest(inventory), config_sha256=digest(model.config.to_dict()), source="Qwen2.5-0.5B-Instruct original checkpoint")
        execution = dict(backend="cpu", dtype="float32", attention="eager", threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
            mkldnn=torch.backends.mkldnn.enabled, matmul_precision=torch.get_float32_matmul_precision(), numpy=np.__version__, torch=str(torch.__version__), transformers=importlib.metadata.version("transformers"),
            component_shape="1x1xH", ordinary_prefill_shape="1xT(prompt)xH", kernel_profile="Source-bound CPU eager; exact selected hardware kernels unprofiled")
        kinds = (type(model), type(model.model), type(layer), type(layer.mlp), type(layer.self_attn), type(model.model.norm), type(model.config), type(tokenizer))
        native_sources = {str(Path(inspect.getfile(kind)).resolve()): sha(inspect.getfile(kind)) for kind in kinds}
        source_identity = dict(request["source_hashes"]) | {"native:"+name: value for name, value in native_sources.items()} | dict(parameter_retriever=RETRIEVER_SHA256)
        qualified = helpers(); output = Path(config["output_scratch"]) / job_id / RESULT

        def prepare_output(path):
            save(path / "request.json", request); save(path / "declared-config.json", config)
            save(path / "source.json", dict(schema=SCHEMA, job_id=job_id, request_sha256=request["request_sha256"], model_identity=identity, execution=execution,
                source_identity=source_identity, checkpoint_files=inventory, model_config=model.config.to_dict(), consumer_encodings=encodings, consumer_ids=consumers,
                prompt_encodings={value["id"]: value["ids"].tolist() for value in prompts}, genuine_record_prefix_capture=True, raw_input_prediction=False,
                scope=config["scope"], planned_counts=planned, pretrained_parameter_bank_written=False))
            save(path / "runtime.json", dict(versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "mdb-runtime", "saturn-pub")},
                native_sources=native_sources, execution=execution, platform=platform.platform(), machine=platform.machine(), torch_build=torch.__config__.show()))
            qualified["qwen_spectral_parameter_recipes.py"]["prepare"](model, checkpoint, inventory, path, config["retrieval_script"])

        result = campaign(model, prompts, consumers, output, model_identity=identity, source_identity=source_identity, settings=dict(config, job_id=job_id, request_sha256=request["request_sha256"]),
            decode=lambda values: tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False), prepare_output=prepare_output)
        for name, expected in signatures.items():
            info = (checkpoint / name).stat()
            require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns) == expected, "original checkpoint mutated during run")
        accounts = publish_accounts(result, output, execution=execution, source_identity=source_identity, run_id=job_id)
        ready = qualified["flux_image_tail_microscope.py"]["finalize"](output, job_id=job_id, request_sha256=request["request_sha256"], **config["transport_limits"])
        ready["equation_records"] = len(accounts["rows"])
        print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
        return ready
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)

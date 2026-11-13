"""Finite generated Qwen primitive-family forecasts and independent saved audit.

Forecasts use explicit tensor equations and original checkpoint parameters.
Native reference module entry remains forbidden until the complete numeric
forecast, including every own-token/own-KV continuation, has been sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import runpy
import shutil
import struct
import time
from pathlib import Path

SCHEMA = "mdb-qwen-primitive-family-campaign-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def read(path):
    return json.loads(Path(path).read_text())


def validate(config):
    from mdb.computation_protocol import digest, verify_plan
    require(config["schema"] == SCHEMA and config["family"] in {"attention", "mlp"}, "unknown campaign")
    plan = verify_plan(config["compiled_protocol"])
    require(len(plan["actions"]) == 6 and len(plan["candidate_menu"]) == 2, "six-role factorial required")
    require(plan["specification"]["model_identity"] == config["weights_sha256"]
        and plan["specification"]["clock"]["layer"] == config["layer"]
        and plan["specification"]["continuation"]["transitions"] == config["tokens"], "model/clock/token horizon differs from compiled program")
    require(1 <= len(config["contexts"]) <= 4 and 2 <= config["tokens"] <= 8, "finite context/token budget")
    require(config["dtype"] == "float32" and config["device"] == "cpu" and config["layer"] == 12,
            "qualified CPU FP32 layer12 profile required")
    require(config["scientific_gates"] == [] and not config["retry"] and not config["nested_mrun"], "exploratory finite lease required")
    require(config["phase"] in {"development", "fresh"}, "unknown phase")
    if config["phase"] == "fresh":
        rule = config["effect_rule"]
        expected = {row["id"]: dict(context_sha256=digest(dict(seed=row["seed"], prompts=row["prompts"])),
            state_pair_sha256=digest(dict(seed=row["seed"], prompts=sorted(row["prompts"].values())))) for row in config["contexts"]}
        require(rule["plan_sha256"] == plan["plan_sha256"] and rule["fresh_context_metadata"] == expected, "prospective rule differs")
    return plan


def prepare_parameters(root, model, output):
    """Bind original archive bytes, actual runtime parameters and tie aliases."""
    from mdb.forward_recording import tensor_descriptor
    files, headers, tensors = {}, {}, {}
    for path in sorted(Path(root).iterdir()):
        if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}:
            files[path.name] = dict(sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix != ".safetensors":
            continue
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            raw = stream.read(size)
            header = json.loads(raw)
            headers[path.name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=size)
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                start, end = entry["data_offsets"]
                stream.seek(8 + size + start)
                arraysha = hashlib.sha256(stream.read(end - start)).hexdigest()
                tensors[name] = dict(file=path.name, archive_entry=entry, archive_bytes_sha256=arraysha)
    manifest = dict(schema="mdb-qwen-original-parameter-artifacts-v1", checkpoint_files=files,
        archive_headers=headers, tensors=tensors,
        runtime_parameters={name: tensor_descriptor(value) for name, value in model.state_dict().items()},
        aliases={"lm_head.weight": "model.embed_tokens.weight"} if model.config.tie_word_embeddings else {},
        conversion="Original safetensors -> Qwen2ForCausalLM.from_pretrained(dtype=torch.float32); no learned coefficients or fitted parameters",
        persistent_parameter_exports=False)
    save(output / "parameter-artifacts.json", manifest)
    script = Path(__file__).with_name("retrieve_qwen_family_weights.py")
    shutil.copyfile(script, output / "retrieve_weights.py")
    verification = runpy.run_path(str(script))["retrieve"](manifest, root)
    save(output / "parameter-retrieval-verification.json", verification)
    return manifest


def run(config, request):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2 as native

    from mdb.computation_protocol import execute_program, run_graph
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    math = runpy.run_path(str(Path(__file__).with_name("qwen_family_math.py")))
    plan = validate(config)
    started = time.monotonic()
    root = Path(config["output_scratch"]) / ("qwen-family-" + os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "plan.json", config)
    save(root / "request.json", request)
    torch.set_num_threads(config["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    weight_path = Path(config["model_path"]) / "model.safetensors"
    file_hash = sha(weight_path)
    require(file_hash == config["checkpoint_file_sha256"] and hashlib.sha256(canonical({"model.safetensors": file_hash})).hexdigest() == config["weights_sha256"],
            "original model file or named weight inventory differs")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    model = Qwen2ForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
    prepare_parameters(config["model_path"], model, root)
    engine = math["FunctionalQwen"](dict(model.named_parameters(remove_duplicate=False)), model.config)
    save(root / "runtime.json", dict(torch=str(torch.__version__), transformers=str(transformers.__version__),
        native_source_sha256=sha(inspect.getfile(native)), math_source_sha256=sha(Path(__file__).with_name("qwen_family_math.py")),
        dtype="torch.float32", device="cpu", threads=torch.get_num_threads(), attention="eager",
        full_native_modules_forbidden_before_seal=True, checkpoint_sha256=file_hash, weights_identity_sha256=config["weights_sha256"]))
    frozen = [(name, id(value), value._version, value.data_ptr()) for name, value in model.named_parameters()]
    counts = dict(native_calls=0, denied_entries=0, functional_calls=0, graph_calls=0, native_hybrid_kernels=0)
    released = False
    sources, predictions, observations = {}, {}, {}
    ids = {}
    for context in config["contexts"]:
        cid = context["id"]
        ids[cid] = {side: tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
                    for side, text in context["prompts"].items()}
        require(ids[cid]["P"].shape == ids[cid]["Q"].shape and 1 <= ids[cid]["P"].shape[1] <= config["max_prompt_tokens"],
                "matched token-position geometry required: " + cid)
    save(root / "inputs.json", {cid: {side: value.tolist() for side, value in pair.items()} for cid, pair in ids.items()})
    labels = {row["id"]: {side: tokenizer.encode(text, add_special_tokens=False)
        for side, text in row["answer_labels"].items()} for row in config["contexts"]}
    require(all(len(value) == 1 for pair in labels.values() for value in pair.values()), "predeclared answer labels must each be one token")
    save(root / "answer-labels.json", labels)

    def event(kind, **fields):
        elapsed = time.monotonic() - started
        require(elapsed < config["max_wall_seconds"], "finite wall budget exhausted")
        row = dict(kind=kind, seconds=elapsed, counts=dict(counts), **fields)
        with (root / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        print("MDB_QWEN_FAMILY " + canonical(row).decode(), flush=True)

    def guard(module, args):
        if not released:
            counts["denied_entries"] += 1
            raise RuntimeError("native model module entered before global forecast seal")
        if module is model:
            counts["native_calls"] += 1
            require(counts["native_calls"] <= 6 * len(ids) * config["tokens"], "native physical-forward budget exhausted")

    guards = [module.register_forward_pre_hook(guard, prepend=True) for module in model.modules()]

    def arrays_path(lane, cid, action, step):
        path = root / lane / cid / action / (f"{step:02d}.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def persist(path, result, input_ids, token):
        data = dict(logits=result["logits"][:, -1:, :].detach().numpy(), input_ids=input_ids.numpy(), token=np.asarray([token], dtype=np.int64))
        cache = result["cache"]
        for index, (key, value) in enumerate(cache):
            data[f"k{index:02d}"] = key.detach().numpy()
            data[f"v{index:02d}"] = value.detach().numpy()
        if "endpoint" in result:
            data["endpoint"] = result["endpoint"].detach().numpy()
        for name, value in data.items():
            require(np.isfinite(value).all(), "nonfinite retained array: " + name)
        with path.open("xb") as stream:
            np.savez(stream, **data)
        return dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size,
                    token_id=token, token_text=tokenizer.decode([token]), fields=sorted(data))

    def persist_ports(lane, cid, action, ports):
        path = root / (lane + "-ports") / cid / (action + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: value.detach().numpy() for name, value in ports.items() if isinstance(value, torch.Tensor)}
        require(arrays and all(np.isfinite(value).all() for value in arrays.values()), "finite actual selected-layer ports required")
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        return dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size,
            descriptors={name: dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()))
                         for name, value in ports.items() if isinstance(value, torch.Tensor)},
            scalar_contract={name: value for name, value in ports.items() if isinstance(value, (int, float, bool))},
            layout_scope="Owned captured tensor layout; NPZ stores logical C-order values")

    def forecast_source(context, side):
        cid = context["id"]
        captured = {}
        def capture(layer, stage, ports):
            if layer == config["layer"] and stage == ("attention" if config["family"] == "attention" else "mlp"):
                captured.update({key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                                 for key, value in ports.items()})
        result = engine.forward(ids[cid][side], capture=capture, logits_to_keep=1)
        counts["functional_calls"] += 1
        require(captured, "functional source capture did not run")
        result["endpoint"] = (captured["primitive_return"][:, :, -1:, :] if config["family"] == "attention"
                              else captured["primitive_return"][:, -1:, :]).detach().clone()
        sources.setdefault(cid, {})[side] = captured
        action = "source" if side == "P" else "recipient"
        predictions[cid + "/" + action] = dict(initial=result, ports=persist_ports("forecast", cid, action, captured))

    def source_ports(pair):
        if config["family"] == "attention":
            return {side: dict(q=port["q"][:, :, -1:, :], k=port["k"], v=port["v"]) for side, port in pair.items()}
        return {side: dict(G=port["gate"][:, -1:, :], U=port["up"][:, -1:, :]) for side, port in pair.items()}

    def endpoint(pair, bindings, operands):
        counts["graph_calls"] += 1
        if config["family"] == "attention":
            kernels = dict(sdpa=lambda q, k, v: math["eager_attention"](q, k, v, operands["mask"], operands["scaling"], operands["kv_groups"]))
        else:
            kernels = dict(silu=F.silu, multiply=torch.mul)
        def check(name, value, port):
            require(len(value.shape) == len(port["shape"]) and all(
                not isinstance(size, int) or size == value.shape[index] for index, size in enumerate(port["shape"]))
                and str(value.dtype) == port["dtype"], "declared graph source geometry differs: " + name)
        return run_graph(plan, source_ports(pair), bindings, kernels, check=check)

    def forecast_hybrid(context, action):
        cid, pair = context["id"], sources[context["id"]]
        bindings = action["bindings"]
        def intervene(layer, stage, ports):
            if layer != config["layer"]:
                return None
            if stage == "attention" and config["family"] == "attention":
                route, value = pair[bindings["route"]], pair[bindings["value"]]
                return dict(q=route["q"], k=route["k"], v=value["v"], tail_only=True)
            if stage == "mlp" and config["family"] == "mlp":
                gate, up = ports["gate"].clone(), ports["up"].clone()
                gate[:, -1:, :] = pair[bindings["gate"]]["gate"][:, -1:, :]
                up[:, -1:, :] = pair[bindings["value"]]["up"][:, -1:, :]
                return dict(gate=gate, up=up)
            return None
        landed = {}
        def graph_kernel(layer, stage, operands):
            value = endpoint(pair, bindings, operands)
            landed["endpoint"] = value.clone()
            if stage == "mlp":
                result = F.silu(operands["gate"]) * operands["up"]
                result[:, -1:, :] = value
                return result
            return value
        captured = {}
        def capture(layer, stage, ports):
            if layer == config["layer"] and stage == ("attention" if config["family"] == "attention" else "mlp"):
                captured.update({key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                    for key, value in ports.items()})
        result = engine.forward(ids[cid]["Q"], intervene=intervene, graph_kernel=graph_kernel, capture=capture, logits_to_keep=1)
        counts["functional_calls"] += 1
        result.update(landed)
        predictions[cid + "/" + action["id"]] = dict(initial=result, ports=persist_ports("forecast", cid, action["id"], captured))

    def forecast_futures():
        for key, row in predictions.items():
            cid, action = key.split("/")
            incoming = ids[cid]["P" if action == "source" else "Q"]
            result, steps = row.pop("initial"), []
            for step in range(config["tokens"]):
                token = int(result["logits"][0, -1].argmax())
                steps.append(persist(arrays_path("forecast", cid, action, step), result, incoming, token))
                if step + 1 < config["tokens"]:
                    incoming = torch.tensor([[token]], dtype=torch.long)
                    result = engine.forward(incoming, cache=result["cache"], logits_to_keep=1)
                    counts["functional_calls"] += 1
            row["steps"] = steps
            row["text"] = tokenizer.decode([step["token_id"] for step in steps])
            event("forecast_future_complete", case=key, tokens=[step["token_id"] for step in steps])

    native_sources = {}
    original_eager = native.eager_attention_forward
    active = dict(context=None, side=None, action=None, prefill=False, capture=None)
    layer = model.model.layers[config["layer"]]
    native_hooks = []

    def eager(module, q, k, v, attention_mask, *args, **kwargs):
        value = original_eager(module, q, k, v, attention_mask, *args, **kwargs)
        if module is not layer.self_attn or not active["prefill"] or config["family"] != "attention":
            return value
        capture = active["capture"]
        capture.update(q=q.detach().clone(), k=k.detach().clone(), v=v.detach().clone(),
            original_q=q.detach().clone(), original_k=k.detach().clone(), original_v=v.detach().clone())
        if attention_mask is not None:
            capture["mask"] = attention_mask.detach().clone()
        capture["endpoint"] = value[0][:, -1:, :, :].transpose(1, 2).detach().clone()
        action = active["action"]
        if action["role"] == "natural-reference":
            return value
        pair, bindings = native_sources[active["context"]], action["bindings"]
        route, values = pair[bindings["route"]], pair[bindings["value"]]
        capture.update(q=route["q"][:, :, -1:, :].detach().clone(), k=route["k"].detach().clone(), v=values["v"].detach().clone())
        mask = attention_mask[:, :, -1:, :] if attention_mask is not None else None
        if mask is not None:
            capture["mask"] = mask.detach().clone()
        mixed, weights = original_eager(module, route["q"][:, :, -1:, :], route["k"], values["v"], mask, *args, **kwargs)
        counts["native_hybrid_kernels"] += 1
        changed = value[0].clone()
        changed[:, -1:, :, :] = mixed
        capture["endpoint"] = mixed.transpose(1, 2).detach().clone()
        return changed, value[1]

    def mlp_hook(name):
        def hook(module, args, value):
            if not active["prefill"] or config["family"] != "mlp":
                return value
            active["capture"][name] = value.detach().clone()
            active["capture"]["original_" + name] = value.detach().clone()
            action = active["action"]
            if action["role"] == "natural-reference":
                return value
            factor = "gate" if name == "gate" else "value"
            source = native_sources[active["context"]][action["bindings"][factor]][name]
            changed = value.clone()
            changed[:, -1:, :] = source[:, -1:, :]
            active["capture"][name] = changed.detach().clone()
            return changed
        return hook

    native_hooks += [layer.mlp.gate_proj.register_forward_hook(mlp_hook("gate")),
                     layer.mlp.up_proj.register_forward_hook(mlp_hook("up"))]
    def product_capture(module, args):
        if active["prefill"] and config["family"] == "mlp":
            active["capture"]["endpoint"] = args[0][:, -1:, :].detach().clone()
    native_hooks.append(layer.mlp.down_proj.register_forward_pre_hook(product_capture))
    def retain_native(name, value):
        if active["prefill"]:
            active["capture"][name] = value.detach().clone()
    if config["family"] == "attention":
        native_hooks.append(layer.input_layernorm.register_forward_pre_hook(lambda module, args: retain_native("residual", args[0])))
        native_hooks.append(layer.input_layernorm.register_forward_hook(lambda module, args, value: retain_native("normalized", value)))
        native_hooks.append(layer.self_attn.o_proj.register_forward_hook(lambda module, args, value: retain_native("attention_return", value)))
        native_hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda module, args: retain_native("post_attention", args[0])))
    else:
        native_hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda module, args: retain_native("residual", args[0])))
        native_hooks.append(layer.post_attention_layernorm.register_forward_hook(lambda module, args, value: retain_native("normalized", value)))
        native_hooks.append(layer.mlp.down_proj.register_forward_hook(lambda module, args, value: retain_native("mlp_return", value)))

    def cache_values(cache):
        return tuple((item.keys.detach().clone(), item.values.detach().clone()) for item in cache.layers)

    def native_future(context, action):
        cid, aid = context["id"], action["id"]
        capture = {}
        active.update(context=cid, action=action, prefill=True, capture=capture)
        incoming = ids[cid]["P" if action.get("source") == "P" else "Q"]
        steps, cache = [], None
        for step in range(config["tokens"]):
            result = model(incoming, past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
            token = int(result.logits[0, -1].argmax())
            cache = result.past_key_values
            retained = dict(logits=result.logits, cache=cache_values(cache))
            if step == 0 and "endpoint" in capture:
                retained["endpoint"] = capture["endpoint"]
            steps.append(persist(arrays_path("native", cid, aid, step), retained, incoming, token))
            if step == 0 and action["role"] == "natural-reference":
                native_sources.setdefault(cid, {})[action["source"]] = capture
            active["prefill"] = False
            incoming = torch.tensor([[token]], dtype=torch.long)
        observations[cid + "/" + aid] = dict(steps=steps, text=tokenizer.decode([step["token_id"] for step in steps]),
            ports=persist_ports("native", cid, aid, capture))
        event("native_future_complete", case=cid + "/" + aid, tokens=[step["token_id"] for step in steps])

    transaction = ForecastTransaction(root / "transaction", model_identity=config["weights_sha256"],
        input_identity=hashlib.sha256(canonical(config["contexts"])).hexdigest(), source_identity=request["source_hashes"],
        frontier=dict(kind="raw-input", tokenizer="original tokenizer; literal prompts; add_special_tokens=False"),
        execution_scope="Externally guarded original native CPU FP32 eager Qwen; independent functional raw-prompt and own-KV forecasts",
        budgets=dict(max_wall_ms=config["max_wall_seconds"] * 1000, max_bytes=32 << 20))
    try:
        with torch.inference_mode(), transaction:
            transaction.frontier("raw_inputs", dict(contexts=config["contexts"], ids={cid: {side: value.numpy() for side, value in pair.items()} for cid, pair in ids.items()}))
            transaction.frontier("checkpoint", dict(manifest_sha256=sha(root / "parameter-artifacts.json"), weights_sha256=config["weights_sha256"]))
            def forecast_sources(_):
                for context in config["contexts"]:
                    for side in ("Q", "P"):
                        forecast_source(context, side)
                return dict(captured_sources=2 * len(ids), native_calls=counts["native_calls"])
            def forecast_endpoints(_):
                for context in config["contexts"]:
                    for action in plan["actions"]:
                        if action["role"] != "natural-reference":
                            forecast_hybrid(context, action)
                return dict(generated_endpoints=4 * len(ids))
            def forecast_continuations(_):
                forecast_futures()
                transaction.predict("all_futures", lambda _: dict(predictions=predictions))
                return dict(futures=len(predictions), own_tokens=True, own_kv=True)
            def seal(_):
                nonlocal released
                require(counts["native_calls"] == counts["denied_entries"] == 0, "forecast native-exclusion violation")
                receipt = transaction.seal()
                transaction.release()
                released = True
                native.eager_attention_forward = eager
                event("global_forecast_sealed", seal_sha256=receipt["seal_sha256"])
                return dict(seal_sha256=receipt["seal_sha256"], native_calls_before_seal=0)
            def native_source_phase(_):
                def execute(_context):
                    for context in config["contexts"]:
                        for action in plan["actions"][:2]:
                            native_future(context, action)
                    return dict(observations=dict(observations))
                transaction.execute("natural_targets", execute)
                return dict(natural_futures=2 * len(ids))
            def native_endpoint_phase(_):
                def execute(_context):
                    for context in config["contexts"]:
                        for action in plan["actions"][2:]:
                            native_future(context, action)
                    return dict(observations={key: value for key, value in observations.items()
                        if key.rsplit("/", 1)[-1] not in {"source", "recipient"}})
                transaction.execute("hybrid_targets", execute)
                return dict(hybrid_futures=4 * len(ids))
            handlers = dict(forecast_sources=forecast_sources, forecast_endpoints=forecast_endpoints,
                forecast_continuations=forecast_continuations, forecast_decodes=lambda _: dict(status="token argmax committed in each own-state step; no image decoder"),
                seal=seal, native_sources=native_source_phase, native_endpoints=native_endpoint_phase,
                native_continuations=lambda _: dict(status="executed with each native branch above", cached_steps=(config["tokens"] - 1) * len(predictions)),
                native_decodes=lambda _: dict(status="literal tokenizer decode only"), audit=lambda _: dict(status="saved-reader required"))
            receipt = execute_program(plan, handlers)
            save(root / "program-receipt.json", receipt)
        verified = verify_transaction(root / "transaction")
        save(root / "transaction-verification.json", verified)
        require(frozen == [(name, id(value), value._version, value.data_ptr()) for name, value in model.named_parameters()], "checkpoint parameters mutated")
        save(root / "forecast-index.json", predictions)
        save(root / "native-index.json", observations)
        save(root / "summary.json", dict(schema=SCHEMA, job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"],
            family=config["family"], phase=config["phase"], counts=counts, seconds=time.monotonic()-started,
            futures=len(predictions), tokens_per_future=config["tokens"], frontier="raw literal prompt IDs, original parameters; no observed activation prefix",
            receiver="Q residual and prompt cache; only selected block's final-token formed output edited; upper-layer writes recomputed",
            terminal_status="unassessed", independent_audit="required"))
    finally:
        native.eager_attention_forward = original_eager
        for handle in guards + native_hooks:
            handle.remove()
    inventory = {str(path.relative_to(root)): dict(sha256=sha(path), bytes=path.stat().st_size)
                 for path in sorted(root.rglob("*")) if path.is_file()}
    save(root / "files.json", dict(files=inventory))
    publication = dict(output_root=str(root), files_manifest_sha256=sha(root / "files.json"),
        request_sha256=request["request_sha256"], job_id=os.environ["MRUN_JOB_ID"])
    print("MDB_QWEN_FAMILY_READY " + canonical(publication).decode(), flush=True)
    return publication


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config = read(os.environ["SATURN_RUN_CONFIG"])
    request = read(os.environ["SATURN_DEBUG_REQUEST"])
    run(config, request)

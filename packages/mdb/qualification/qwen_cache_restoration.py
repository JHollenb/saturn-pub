"""Four typed cache restorations, sealed predictions, and original Qwen futures.

The source frontier is observed post-prefill K/V plus a common pending token.
No old continuation is opened here. This worker performs full tensor equations;
it does not learn a surrogate or claim a raw-prompt forecast at this frontier.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def require(value, message):
    if not value:
        raise ValueError(message)


def bit_equal(left, right):
    import torch
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def support(q, e, first_layer):
    """Prove exactly where source states differ, including signed-zero bits."""
    import torch
    require(len(q) == len(e) and 0 <= first_layer < len(q), "cache layer geometry differs")
    length = q[0][0].shape[-2]
    rows = []
    for index, (qp, ep) in enumerate(zip(q, e, strict=True)):
        for name, left, right in zip(("k", "v"), qp, ep, strict=True):
            require(left.ndim == 4 and left.shape == right.shape and left.shape[-2] == length
                    and left.dtype == right.dtype == torch.float32
                    and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()),
                    "finite FP32 matched cache geometry required")
            different = left.contiguous().view(torch.int32) != right.contiguous().view(torch.int32)
            changed_rows = different.any(dim=-1).any(dim=1).any(dim=0).nonzero().flatten().tolist()
            require(not changed_rows or (index >= first_layer and changed_rows == [length - 1]),
                    f"source difference outside declared row/layer support: {name}{index:02d}")
            rows.append(dict(field=f"{name}{index:02d}", layer=index,
                changed_elements=int(different.sum()), changed_rows=changed_rows,
                max_abs=float((left - right).abs().max())))
    return dict(cache_length=length, first_layer=first_layer, rows=rows,
                declared_support_verified=True)


def corners(q, e, first_layer, pending):
    """Apply explicit read/write Acts to one content-addressed observed parent."""
    import torch

    from mdb.titan import ActRegistry, ActSpec, Frame

    support(q, e, first_layer)
    values = {f"{name}{index:02d}": value.clone()
              for index, pair in enumerate(q) for name, value in zip(("k", "v"), pair, strict=True)}
    values.update({f"e.{name}{index:02d}": value.clone()
                   for index, pair in enumerate(e) for name, value in zip(("k", "v"), pair, strict=True)})
    values.update(pending=torch.tensor([[pending]], dtype=torch.long),
                  position=torch.tensor([[q[0][0].shape[-2]]], dtype=torch.long))
    kinds = {name: "pending-token" if name == "pending" else "cache-position" if name == "position"
             else "post-rope-key" if name.replace("e.", "").startswith("k") else "value-cache"
             for name in values}
    abis = {name: dict(shape=list(value.shape), dtype=str(value.dtype)) for name, value in values.items()}
    parent = Frame.from_values(values, kinds=kinds, abis=abis,
        program_fingerprint="qwen-cache-KV-restoration-v1",
        metadata=dict(frontier="observed-post-prefill-cache", first_layer=first_layer,
                      row=q[0][0].shape[-2]-1, original_intervention_expired=True))
    record, result = dict(parent=parent.to_dict(), corners={}), {}
    registry = ActRegistry()
    q_names = tuple(f"{name}{index:02d}" for index in range(len(q)) for name in ("k", "v"))
    for role in ("QQ", "EQ", "QE", "EE"):
        writes = tuple(f"{name}{index:02d}" for index in range(first_layer, len(q))
                       for name, side in zip(("k", "v"), role, strict=True) if side == "E")
        spec = ActSpec(name="qwen-cache-restore-" + role, abi_version="v1",
            reads=(*q_names, *("e." + name for name in writes), "pending", "position"),
            writes=writes, numerical_contract="exact-source-row-replacement",
            write_kinds={name: kinds[name] for name in writes},
            write_abis={name: abis[name] for name in writes})

        def apply(inputs, destinations=writes):
            output = {}
            for name in destinations:
                value = inputs[name].clone()
                value[:, :, -1:, :] = inputs["e." + name][:, :, -1:, :]
                output[name] = value
            return output, dict(selected_fields=len(destinations), last_row_only=True)

        registry.register(spec, "copy-final-row", apply)
        child, evidence = registry.run(spec.identity, "copy-final-row", parent)
        cache = tuple(tuple(child.values[f"{name}{index:02d}"].clone() for name in ("k", "v"))
                      for index in range(len(q)))
        for index, pair in enumerate(cache):
            for name, side, value, original, edited in zip(("k", "v"), role, pair, q[index], e[index], strict=True):
                expected = original.clone()
                if index >= first_layer and side == "E":
                    expected[:, :, -1:, :] = edited[:, :, -1:, :]
                require(bit_equal(value, expected), "typed Act landed outside declared byte replacement")
        result[role] = cache
        record["corners"][role] = dict(before=parent.to_dict(), after=child.to_dict(),
            act=spec.to_dict(), receipt=evidence.to_dict())
    return result, record


def native_cache(model, values):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    for index, (key, value) in enumerate(values):
        cache.update(key.clone(), value.clone(), index)
    return cache


def future(step_function, cache, pending, choices, persist):
    """Each lane feeds its own argmax back; initial pending token is shared."""
    import torch
    incoming = torch.tensor([[pending]], dtype=torch.long)
    steps = []
    for step in range(choices):
        result = step_function(incoming, cache)
        token = int(result["logits"][0, -1].argmax())
        steps.append(persist(step, result, incoming, token))
        incoming = torch.tensor([[token]], dtype=torch.long)
        cache = result["cache"]
    return steps


def run(config, request):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen2ForCausalLM
    from transformers.models.qwen2 import modeling_qwen2 as native

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    payload = Path(__file__).resolve().parent
    design = read(payload / "design.json")
    require(config["design_sha256"] == sha(payload / "design.json"), "frozen design differs")
    require(design["new_choices_per_corner"] == 5 and design["new_forward_census"] ==
            dict(functional=20, native=20, prefills=0, model_loads=1), "finite work geometry differs")
    root = Path(config["output_scratch"]) / ("qwen-cache-" + os.environ["MRUN_JOB_ID"])
    root.mkdir(parents=True, exist_ok=False)
    save(root / "design.json", design)
    save(root / "request.json", request)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    counts = dict(functional_calls=0, native_calls=0, denied_entries=0)
    released = False

    def event(kind, **fields):
        require(time.monotonic() - started < config["max_wall_seconds"], "finite wall budget exhausted")
        row = dict(kind=kind, counts=dict(counts), seconds=time.monotonic()-started, **fields)
        with (root / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        print("MDB_QWEN_CACHE " + canonical(row).decode(), flush=True)

    parents = []
    for side in ("q", "e"):
        name = "parent-" + side + ".npz"
        require(sha(payload / name) == design["source_files"][name]["sha256"], "captured source differs")
        with np.load(payload / name, allow_pickle=False) as arrays:
            require(arrays["token"].tolist() == [design["pending_token"]], "source pending tokens differ")
            cache = tuple(tuple(torch.from_numpy(arrays[f"{name}{index:02d}"].copy()) for name in ("k", "v"))
                          for index in range(design["layers"]))
        parents.append(cache)
        target = root / "parents" / (side + ".npz")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(payload / name, target)
    q, e = parents
    require(q[0][0].shape[-2] == design["cache_length"], "source clock differs")
    save(root / "support.json", support(q, e, design["changed_layer_start"]))
    states, frames = corners(q, e, design["changed_layer_start"], design["pending_token"])
    save(root / "frames.json", frames)
    for name in ("parent-files.json", "parent-parameter-artifacts.json", "parent-runtime.json"):
        shutil.copyfile(payload / name, root / name)

    require(sha(Path(config["model_path"]) / "model.safetensors") == design["checkpoint_file_sha256"],
            "original checkpoint differs")
    model = Qwen2ForCausalLM.from_pretrained(config["model_path"], local_files_only=True,
        dtype=torch.float32, attn_implementation="eager").cpu().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
    prepare = runpy.run_path(str(payload / "qwen_family_campaign.py"))["prepare_parameters"]
    manifest = prepare(config["model_path"], model, root)
    require(manifest == read(payload / "parent-parameter-artifacts.json"), "parameter/tokenizer/config bindings drifted")
    prior_runtime = read(payload / "parent-runtime.json")
    require(sha(inspect.getfile(native)) == prior_runtime["native_source_sha256"] and
            str(torch.__version__) == prior_runtime["torch"] and
            str(transformers.__version__) == prior_runtime["transformers"], "qualified native runtime drifted")
    engine = runpy.run_path(str(payload / "qwen_family_math.py"))["FunctionalQwen"](
        dict(model.named_parameters(remove_duplicate=False)), model.config)
    require(engine.layers == design["layers"], "model/cache layers differ")
    save(root / "runtime.json", dict(**prior_runtime, source_frontier="observed post-prefill cache",
         physical_batch=1, compiled_choices=5, model_loads=1, forecast_module_exclusion=True))
    parameter_versions = [(name, id(value), value._version) for name, value in model.named_parameters()]

    def guard(module, args):
        if not released:
            counts["denied_entries"] += 1
            raise RuntimeError("native module entered before complete forecast seal")
        if module is model:
            counts["native_calls"] += 1
            require(counts["native_calls"] <= 20, "native forward budget exceeded")

    guards = [module.register_forward_pre_hook(guard, prepend=True) for module in model.modules()]
    captures = {}

    def native_port(name, value):
        captures[name] = value.detach().clone()

    hooks = []
    for index, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            lambda module, args, value, name=f"a{index:02d}": native_port(name, value)))
        hooks.append(layer.register_forward_hook(
            lambda module, args, value, name=f"r{index:02d}": native_port(
                name, value[0] if isinstance(value, tuple) else value)))

    def functional_step(incoming, cache):
        ports = {}

        def capture(index, stage, values):
            if stage in ("attention", "block"):
                key, field = ("a", "attention_return") if stage == "attention" else ("r", "output")
                ports[f"{key}{index:02d}"] = values[field].detach().clone()

        result = engine.forward(incoming, cache=cache, capture=capture, logits_to_keep=1)
        counts["functional_calls"] += 1
        require(counts["functional_calls"] <= 20, "functional forward budget exceeded")
        result["ports"] = ports
        return result

    def native_step(incoming, cache):
        captures.clear()
        result = model(incoming, past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
        return dict(logits=result.logits, cache=result.past_key_values, ports=dict(captures))

    def persist(lane, role, step, result, incoming, token):
        cache = result["cache"]
        if lane == "native":
            cache = tuple((item.keys, item.values) for item in cache.layers)
        data = dict(logits=result["logits"][:, -1:, :].detach().numpy(),
                    input_ids=incoming.numpy(), token=np.asarray([token], dtype=np.int64))
        for index, pair in enumerate(cache):
            for name, value in zip(("k", "v"), pair, strict=True):
                data[f"{name}{index:02d}"] = value.detach().numpy()
        data.update({name: value.detach().numpy() for name, value in result["ports"].items()})
        require(len(result["ports"]) == 2 * engine.layers, "declared native/functional ports did not execute")
        for name, value in data.items():
            require(bool(np.isfinite(value).all()), "nonfinite retained value: " + name)
        path = root / lane / role / f"{step:02d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez(stream, **data)
        return dict(path=str(path.relative_to(root)), sha256=sha(path), bytes=path.stat().st_size, token_id=token)

    forecasts, observations = {}, {}
    transaction = ForecastTransaction(root / "transaction", model_identity=design["weights_sha256"],
        input_identity=hashlib.sha256(canonical(design["source_files"])).hexdigest(),
        source_identity=request["source_hashes"], frontier=dict(kind="architectural",
            description="observed post-prefill cache", source_job=design["source_job"]),
        execution_scope="Full CPU FP32 Qwen equations from declared observed cache; original guarded native continuation",
        budgets=dict(max_wall_ms=config["max_wall_seconds"]*1000, max_bytes=32 << 20))
    try:
        with torch.inference_mode(), transaction:
            transaction.frontier("observed_cache", dict(
                Q={f"{name}{i:02d}": value.numpy() for i, pair in enumerate(q) for name, value in zip(("k", "v"), pair, strict=True)},
                E={f"{name}{i:02d}": value.numpy() for i, pair in enumerate(e) for name, value in zip(("k", "v"), pair, strict=True)},
                pending=design["pending_token"], cache_position=design["cache_length"],
                parameter_manifest_sha256=sha(root / "parameter-artifacts.json")))
            for role, cache in states.items():
                steps = future(functional_step, cache, design["pending_token"], 5,
                    lambda step, result, incoming, token, role=role: persist("forecast", role, step, result, incoming, token))
                forecasts[role] = dict(steps=steps, text=tokenizer.decode([step["token_id"] for step in steps]))
                event("forecast_future_complete", role=role, tokens=[step["token_id"] for step in steps])
            save(root / "forecast-index.json", forecasts)
            transaction.predict("all_futures", lambda _: dict(predictions=forecasts))
            require(counts == dict(functional_calls=20, native_calls=0, denied_entries=0), "forecast exclusion/count contract violated")
            seal = transaction.seal()
            transaction.release()
            released = True
            event("global_forecast_sealed", seal_sha256=seal["seal_sha256"])

            def execute(_):
                for role, cache in states.items():
                    steps = future(native_step, native_cache(model, cache), design["pending_token"], 5,
                        lambda step, result, incoming, token, role=role: persist("native", role, step, result, incoming, token))
                    observations[role] = dict(steps=steps, text=tokenizer.decode([step["token_id"] for step in steps]))
                    event("native_future_complete", role=role, tokens=[step["token_id"] for step in steps])
                return observations

            transaction.execute("restored_native_futures", execute)
        require(counts == dict(functional_calls=20, native_calls=20, denied_entries=0), "completed forward census differs")
        require(parameter_versions == [(name, id(value), value._version) for name, value in model.named_parameters()], "parameters mutated")
        for side in ("q", "e"):
            require(sha(root / "parents" / (side + ".npz")) == design["source_files"]["parent-" + side + ".npz"]["sha256"], "observed parent mutated")
        # Test the in-memory parents too; immutable source files alone are insufficient.
        for source, original in zip(("q", "e"), (q, e), strict=True):
            with np.load(root / "parents" / (source + ".npz"), allow_pickle=False) as arrays:
                for index, pair in enumerate(original):
                    for name, value in zip(("k", "v"), pair, strict=True):
                        require(value.numpy().tobytes() == arrays[f"{name}{index:02d}"].tobytes(), "in-memory parent mutated")
        save(root / "native-index.json", observations)
        save(root / "counts.json", counts)
        save(root / "transaction-verification.json", verify_transaction(root / "transaction"))
        event("complete", final_cache_length=design["cache_length"] + 5)
    finally:
        for handle in guards + hooks:
            handle.remove()
    files = {str(path.relative_to(root)): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(root.rglob("*")) if path.is_file()}
    save(root / "files.json", dict(files=files))
    publication = dict(output_root=str(root), files_manifest_sha256=sha(root / "files.json"),
                       job_id=os.environ["MRUN_JOB_ID"], request_sha256=request["request_sha256"])
    print("MDB_QWEN_CACHE_READY " + canonical(publication).decode(), flush=True)
    return publication


if __name__ == "__main__":
    run(read(os.environ["SATURN_RUN_CONFIG"]), read(os.environ["SATURN_DEBUG_REQUEST"]))

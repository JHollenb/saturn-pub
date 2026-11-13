"""Explicit saved-input worlds, complete primitive forecasts, then native consumers.

There is no implicit positive world, row-sign intervention, or whole-model call.
"""
from __future__ import annotations

import hashlib
import runpy
import time
from pathlib import Path

COMPONENT_SHA256 = "6efd05f0a4128aee469c0eb2e65d9f2da508ffb029091ab8e2b604fd5cb61f26"
_path = Path(__file__).with_name("qwen_component_experiment.py")
if hashlib.sha256(_path.read_bytes()).hexdigest() != COMPONENT_SHA256:
    raise ValueError("frozen component utilities differ")
C = runpy.run_path(str(_path))
ROW = C["ROW"]
HELPERS = {"qwen_component_experiment.py": COMPONENT_SHA256, **C["HELPERS"]}
require, canonical, sha, read, save, inventory, verify_inventory, tensor_record, live_descriptor, identities = (
    C[key] for key in ("require", "canonical", "sha", "read", "save", "inventory", "verify_inventory", "tensor_record", "live_descriptor", "identities"))
PORTS = ("head", "projection", "R", "M", "Hnext")
SUPPORT = dict(status="not-applicable", reason="explicit saved input worlds; no row-sign support claim")


def scope(index):
    return dict(kind="architectural", origin="inherited-functional-state", scope=f"conditional block{index} hidden-state interface",
        raw_input_prediction=False, genuine_record_prefix_capture=False, genuine_qwen_act=False)


def contract(worlds=2):
    return dict(functional_blocks=worlds, functional_SDPA=worlds, functional_linear=7 * worlds,
        native_block_calls=worlds, native_SDPA=worlds, native_linear=7 * worlds, component_hydrations=1,
        native_authorizations=worlds, forecast_module_attempts=0, native_module_entries=15 * worlds,
        full_model=0, LM=0, embedding=0, tokenizer=0, FLUX=0, VAE=0, noise=0)


def load_sources(config, protocol, device):
    """Authenticate two pretarget parent inventories before reading seven arrays."""
    roots, inventories, externals = {}, {}, {}
    for alias in ("world", "context"):
        declared = protocol["roots"][alias]
        files, seal, request = (read(config[alias + "_" + name]) for name in ("files", "seal", "request"))
        require(config[alias + "_root"] == declared["root"]
            and sha(config[alias + "_files"]) == declared["raw_manifest_sha256"]
            and sha(config[alias + "_seal"]) == declared["global_seal_sha256"], "parent custody differs: " + alias)
        require(ROW["BASE"]["request_identity"](request) == request["request_sha256"] == declared["request_sha256"]
            and files["request_sha256"] == request["request_sha256"] and files["job_id"] == declared["job_id"], "parent request differs: " + alias)
        require(seal["schema"] == "mdb-global-forecast-seal-v1" and not seal["violations"]
            and seal["pre_release_work"]["native_calls"] == 0, "parent pretarget seal differs: " + alias)
        roots[alias] = declared
        inventories[alias] = files["files"]
        externals[alias] = seal["declarations"]["frontier"]["external_files"]
    signatures, layouts, references, tensors = {}, {}, {}, {}
    def load(name, alias, row):
        relative = row["path"]
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "unsafe inherited member")
        binding = dict(bytes=row["bytes"], sha256=row["sha256"])
        require(inventories[alias].get(relative) == externals[alias].get(relative) == binding, "source missing from pretarget parent inventory")
        path = Path(config[alias + "_root"]) / relative
        value, layouts[name] = C["restore_array"](path, row, device)
        references[name] = dict(row, inherited=True, root=alias)
        stat = path.stat(); signatures[path] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        tensors[name] = value
    for world in protocol["world_order"]:
        row = protocol["world_sources"][world]
        sidecar_key = "world_forecast" if world == protocol["baseline_world"] else "world_shortcut"
        sidecar = read(config[sidecar_key])
        expected = sidecar["arms"]["prefix-negative"]["ports"]["Hnext"]
        require({key: value for key, value in row.items() if key not in ("root", "inherited")} == expected,
            "world source/layout differs from original forecast")
        # Bind the exact sidecar as well as its selected member to the parent's seal.
        relative = "forecast.json" if sidecar_key == "world_forecast" else "shortcut.json"
        binding = dict(bytes=Path(config[sidecar_key]).stat().st_size, sha256=sha(config[sidecar_key]))
        require(inventories["world"].get(relative) == externals["world"].get(relative) == binding, "world sidecar lineage differs")
        load(world, "world", row)
    for name, row in protocol["shared_context_sources"].items():
        load(name, "context", row)
    def verify():
        for path, before in signatures.items():
            stat = path.stat()
            require((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == before, "inherited source file changed")
    return tensors, references, dict(arrays=layouts), verify


def difference(root, left, right, orientation):
    import numpy as np

    delta = ROW["BASE"]["load_array"](root, left).astype(np.float64) - ROW["BASE"]["load_array"](root, right).astype(np.float64)
    return dict(shape=list(delta.shape), elements=delta.size, dtype="float64", orientation=orientation,
        sha256=hashlib.sha256(delta.astype("<f8").tobytes(order="C")).hexdigest())


def addition_diagnostics(output, forecasts, natives, inputs, left, right):
    """Saved-array finite-addition and cancellation observations, no projection."""
    import numpy as np

    def scalar(value):
        return dict(elements=value.size, max_abs=float(np.max(np.abs(value))), rms=float(np.sqrt(np.mean(value * value))),
            signed_mean=float(np.mean(value)), changed=int(np.count_nonzero(value)))
    reports = {}
    delta_h = inputs[left].float().cpu().numpy().astype(np.float64) - inputs[right].float().cpu().numpy().astype(np.float64)
    for lane, worlds in (("functional", forecasts), ("native", natives)):
        delta = {name: ROW["BASE"]["load_array"](output, worlds[left]["ports"][name]).astype(np.float64)
            - ROW["BASE"]["load_array"](output, worlds[right]["ports"][name]).astype(np.float64) for name in ("projection", "R", "M", "Hnext")}
        r, m = delta["R"], delta["M"]
        dot = float(np.sum(r * m)); norm = float(np.linalg.norm(r.ravel()) * np.linalg.norm(m.ravel()))
        reports[lane] = dict(input_delta=scalar(delta_h), output_delta=scalar(delta["Hnext"]),
            residual_addition_defect=scalar(r - delta_h - delta["projection"]),
            final_addition_defect=scalar(delta["Hnext"] - r - m), R_M_dot=dot, R_M_cosine=None if norm == 0 else dot / norm,
            R_M_cross_term=2 * dot, scope="FP64 reductions of saved finite outputs; no learned RHS; observations only")
    return reports


def execute_worlds(output, layer, worlds, context_sources, protocol, *, source_identity, references=None,
                   source_metadata=None, verify_sources=lambda: None, before_execute=None, functional=None,
                   max_wall_seconds=240, transaction_bytes=64 << 20, job_id=None, request_sha256=None):
    import torch

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output, index = Path(output), protocol["component_index"]
    identifiers = protocol["world_order"]
    require(list(worlds) == identifiers and len(set(identifiers)) == len(identifiers) >= 2
        and all(name.replace("-", "").isalnum() for name in identifiers), "explicit unique world order required")
    require(not layer.training and not layer.self_attn.config.attention_bias and layer.self_attn.config.attention_dropout == 0., "component policy differs")
    require(all(value.dtype == torch.bfloat16 and list(value.shape) == list(worlds[identifiers[0]].shape) for value in worlds.values())
        and worlds[identifiers[0]].shape[1] == protocol["sequence_rows"], "world input geometry differs")
    response = protocol["response"]; left, right = response["left_world"], response["right_world"]
    require(left in worlds and right == protocol["baseline_world"] and left != right and response["ports"] == list(PORTS), "response world/port contract differs")
    graph, history, profile, dispatch, context, context_record = C["context_for"](layer, context_sources, output, index)
    parameters = {f"model.layers.{index}." + name: value for name, value in layer.named_parameters()}
    before = identities(layer, index); live_sources = {name: live_descriptor(value) for name, value in {**worlds, **context_sources}.items()}
    references = references or {}
    frontier = scope(index)
    save(output / "parameter-identity-before.json", before)
    save(output / "source-frontier.json", dict(source_metadata or {}, worlds={name: dict(input_reference=references.get(name)) for name in identifiers},
        shared_context={name: references.get(name) for name in context_sources}, context=context_record, frontier=frontier, support=SUPPORT))
    counts = {key: 0 for key in contract(len(worlds))}; counts["component_hydrations"] = 1
    released, phase, ports = dict(native=False), dict(lane=None, arm=None), {}
    events, calls, entries, forecasts, natives = [], [], [], {}, {}
    started = time.monotonic()
    def event(kind, **fields):
        require(time.monotonic() - started <= max_wall_seconds, "finite panel wall bound exceeded")
        row = dict(number=len(events), kind=kind, native_module_entries=counts["native_module_entries"], **fields)
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
    def verify():
        verify_sources()
        require(identities(layer, index) == before and all(live_descriptor(value) == live_sources[name]
            for name, value in {**worlds, **context_sources}.items()), "live source/component identity changed")
    def capture(name, value):
        require(bool(torch.isfinite(value).all()), "new port contains nonfinite values")
        ports[name] = tensor_record(output, f"{phase['lane']}/{phase['arm']}/{name}.npy", value)
    def entry(_module, args, kwargs):
        require(released["native"] and kwargs.get("attention_mask") is None and kwargs.get("past_key_values") is None
            and kwargs.get("use_cache") is False, "native entry lacks seal/context/cache contract")
        consumed = {name: live_descriptor(value) for name, value in dict(position_ids=kwargs["position_ids"], cos=kwargs["position_embeddings"][0], sin=kwargs["position_embeddings"][1]).items()}
        require(all(all(consumed[name][live] == context_record[name][saved] for live, saved in
            (("content_sha256", "content_sha256"), ("shape", "shape"), ("stride", "stride"), ("dtype", "logical_dtype"), ("device", "device_at_capture"))) for name in consumed), "actual native context differs")
        actual = live_descriptor(args[0]); expected = live_sources[phase["arm"]]
        require(actual == expected, "actual native input world differs")
        entries.append(dict(world=phase["arm"], input=actual, input_reference=references.get(phase["arm"]), consumed_context=consumed,
            attention_mask=None, use_cache=False, past_key_values=None))
    handles = [layer.register_forward_pre_hook(entry, with_kwargs=True),
        layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, args: capture("R", args[0])),
        layer.mlp.down_proj.register_forward_pre_hook(lambda _m, args: capture("product", args[0]))]
    for name, module in (("projection", layer.self_attn.o_proj), ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj), ("M", layer.mlp.down_proj)):
        handles.append(module.register_forward_hook(lambda _m, _a, value, name=name: capture(name, value)))
    try:
        with ROW["BASE"]["learned_guard"](layer, released, counts), C["primitives"](phase, counts, calls, ports, capture), torch.inference_mode():
            for name in identifiers:
                phase.update(lane="functional", arm=name); ports.clear()
                with history["_no_modules"](), history["_frozen_parameters"](parameters):
                    if functional is None:
                        history["_block"](graph, worlds[name], parameters, profile, index, context, dispatch,
                            lambda _lane, port, value: capture(port, value), "functional")
                    else:
                        functional(worlds[name], capture)
                require(set((*PORTS, "q", "k", "v")).issubset(ports), "functional forecast omitted complete ports")
                forecasts[name] = dict(input_reference=references.get(name), consumed_input_descriptor=live_descriptor(worlds[name]), ports=dict(ports))
                counts["functional_blocks"] += 1; event("functional_world_completed", world=name)
            responses = {name: difference(output, forecasts[left]["ports"][name], forecasts[right]["ports"][name], response["orientation"]) for name in PORTS}
            save(output / "forecast.json", dict(worlds=forecasts, support=SUPPORT))
            save(output / "forecast-responses.json", dict(response, baseline_world=right, fields=responses))
            verify(); sealed_files = inventory(output, exclude=("timeline.jsonl",))
            tx = ForecastTransaction(output / "transaction", model_identity=f"original-qwen3-decoder-layer-{index}",
                input_identity=hashlib.sha256(canonical({name: live_sources[name]["content_sha256"] for name in identifiers})).hexdigest(),
                source_identity=hashlib.sha256(canonical(source_identity)).hexdigest(),
                frontier=dict(frontier, external_files=sealed_files, inherited_sources=references), execution_scope="Explicit own saved-input worlds; complete futures sealed before standalone native consumers",
                array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=len(worlds), max_bytes=transaction_bytes, max_files=30000, max_wall_ms=max_wall_seconds * 1000))
            tx.frontier("inherited-worlds", dict(worlds={name: references.get(name) for name in identifiers}, context=context_record, scope=frontier))
            tx.predict("cohort", lambda _ctx: dict(predictions=dict(external_files=sealed_files, inherited_sources=references, response_hashes=responses,
                counts=dict(counts), native_targets_at_seal=0)))
            seal = tx.seal(); event("cohort_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets_at_seal=0)
            def execute(_ctx):
                if before_execute is not None:
                    before_execute(output, sealed_files)
                verify_inventory(output, sealed_files); verify(); released["native"] = True
                event("native_targets_released", seal_sha256=seal["seal_sha256"])
                for name in identifiers:
                    verify(); phase.update(lane="native", arm=name); ports.clear()
                    counts["native_authorizations"] += 1; event("native_component_authorized", world=name)
                    capture("Hnext", layer(worlds[name], attention_mask=None, position_ids=context[2], position_embeddings=context[3:5], past_key_values=None, use_cache=False))
                    natives[name] = dict(actual_input_descriptor=live_descriptor(worlds[name]), input_reference=references.get(name), ports=dict(ports))
                    counts["native_block_calls"] += 1; event("native_component_completed", world=name)
                return dict(counts=counts, worlds=list(natives))
            tx.execute("native-verification", execute); tx.close()
    finally:
        for handle in reversed(handles):
            handle.remove()
    verify(); verify_inventory(output, sealed_files)
    require(counts == contract(len(worlds)), "actual physical census differs")
    save(output / "native.json", dict(worlds=natives, context=context_record, head_to_canonical=[0, 2, 1, 3]))
    save(output / "sdpa.json", dict(calls=calls))
    save(output / "native-contract.json", dict(expected=contract(len(worlds)), observed=counts, block_entries=entries,
        projection_provenance="actual passive o_proj output hook", R_provenance="actual post_attention_layernorm input hook"))
    save(output / "addition-diagnostics.json", addition_diagnostics(output, forecasts, natives, worlds, left, right))
    save(output / "parameter-identity-after.json", identities(layer, index))
    result = dict(schema="mdb-qwen-explicit-worlds-v2", job_id=job_id, request_sha256=request_sha256, counts=counts, contract=contract(len(worlds)),
        frontier=frontier, support=SUPPORT, forecast_seal=seal["seal_sha256"], seconds=time.monotonic() - started,
        transaction=verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"], max_bytes=transaction_bytes),
        mechanics_status="completed", numerical_status="all worlds/rows retained; numeric, bit and signed-zero relations are observations")
    save(output / "summary.json", result)
    return result

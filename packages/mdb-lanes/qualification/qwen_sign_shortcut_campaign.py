"""Prospective three-context Qwen-only conditional sign-shortcut experiment.

Positive ports come from each raw input's own functional prefix. The shortcut
receives only R/G/U and the original down weight, before the negative functional
reference starts. All complete forecasts and signed responses precede any native
model entry. This is neither a Qwen Act nor an independent-kernel claim.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import inspect
import json
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

SCHEMA = "mdb-qwen-sign-shortcut-campaign-v1"
RESULT = "qwen-sign-shortcut-result"
PROTOCOL_SHA256 = "5ce41dda16fdfd687aba584f9aaedc389d0ec8bf088b2ea63de8b4456ddf9cf2"
HELPERS = {
    "qwen_sign_shortcut.py": "8e59051a34bc8e89c9376fd29aec966df9cbe84560c81f86910504172caaa457",
    "qwen_h18_state_history.py": "17084d404f5ae00557e4ad8e94a8b84704787c47ebf9c875228cb452c4857036",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "7f13836ae94c84b7ed0f7df67b1f7b1d28a7e534a83d509fa7320a09a6d845df",
    "flux_source_receiver.py": "d483e41bfac018c8225662e8417bcef46f6916fa2a14d910f16e9a12beb5250c",
    "flux_image_tail_microscope.py": "b9efb61874005c8b11307c197a2ea5299078b42d5b4ab4b5cc719f7a4a4a32bb",
}
RETRIEVER_SHA256 = "7cef6f04829dc8638718c25cffa9e3e5ca86703e19cb770d57aa516b719677ae"
PATH_KEYS = ("qwen_config", "qwen_parameter_manifest", "qwen_source_binding", "retrieval_script")
RUNTIME_KEYS = ("qwen_native_source_sha256", "tokenizers_wrapper_source_sha256", "tokenizers_native_source_sha256",
                "jinja_environment_source_sha256", "jinja_sandbox_source_sha256")
ROPE_POLICY = dict(init_device="cpu", init_dtype="float32", placement_dtype="float32", source="original native CPU initializer; device-only placement")
ENDPOINTS = ("product", "M", "Hnext")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def request_identity(request):
    return hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def selected_raw_inputs(protocol_path, output, *, expected_sha256=PROTOCOL_SHA256):
    """Read only predeclared UTF8 sources; never tokenize or adapt selection."""
    protocol_path, output = Path(protocol_path), Path(output)
    require(sha(protocol_path) == expected_sha256, "predeclared selection changed")
    protocol = read(protocol_path)
    require(len(protocol["cases"]) == 3, "fixed raw selection differs")
    shutil.copyfile(protocol_path, output / "protocol.json")
    for row in protocol["cases"]:
        require(Path(row["raw_file"]).name == row["raw_file"], "unsafe raw selection path")
        path = protocol_path.parent / row["raw_file"]
        require(path.stat().st_size == row["raw_bytes"] and sha(path) == row["raw_sha256"] and
                path.read_bytes() == row["raw_text"].encode(), "literal predeclared raw text differs")
        shutil.copyfile(path, output / path.name)
    return protocol["cases"]


@lru_cache(maxsize=1)
def helpers():
    root = Path(__file__).parent
    result = {}
    for name, expected in HELPERS.items():
        require(sha(root / name) == expected, "frozen helper source differs: " + name)
        result[name] = runpy.run_path(str(root / name))
    return result


def contract(*, edit_index=18, horizon=19, native_layers=36, cases=3):
    require(horizon == edit_index + 1 and 0 < edit_index < native_layers and cases == 3, "one-block shortcut geometry differs")
    return dict(schema=SCHEMA, contexts=cases, edit_index=edit_index, horizon=horizon,
        functional_reference_blocks=cases * (horizon + 1), functional_reference_SDPA=cases * (horizon + 1),
        shortcut_down_projections=cases, shortcut_attention=0, shortcut_RMSNorm=0,
        native_blocks=cases * (native_layers + 1), native_SDPA=cases * (native_layers + 1),
        native_full_model_calls=cases, native_LM_calls=cases, native_suffix_calls=cases,
        native_authorizations=2 * cases, genuine_qwen_acts=0, component_hydrations=1,
        cuda_placements=1, cpu_offloads=1, forecast_module_entries=0,
        no_flux_vae_noise=True, native_unused_states="after horizon through full native model final norm/LM are reference-only",
        limits=["Counts are component/primitive occurrences, not independent trials.",
                "Named ports do not establish an independent Flash/GEMM implementation or selected kernels."])


def make_config(base):
    config = {key: base[key] for key in (*PATH_KEYS, *RUNTIME_KEYS, "parameter_root", "tokenizer_path") if key in base}
    for key in PATH_KEYS:
        if key + "_sha256" in base:
            config[key + "_sha256"] = base[key + "_sha256"]
    config.update(schema=SCHEMA, selection_protocol="protocol.json", selection_protocol_sha256=PROTOCOL_SHA256,
        helper_source_sha256=dict(HELPERS), attention_dispatch=dict(backend="sdpa", gqa_policy="repeat_kv"),
        qwen_rope_buffer_policy=dict(ROPE_POLICY), edit_index=18, horizon=19, selected=[9, 18, 19],
        max_wall_seconds=600, forecast_transaction_bytes=64 << 20, num_threads=4,
        output_scratch="/mnt/big/mdb/research", output_name=RESULT,
        transport_limits=dict(max_expanded_bytes=2 << 30, max_member_bytes=64 << 20))
    return config


def validate_config(config):
    require(config.get("schema") == SCHEMA and config.get("helper_source_sha256") == HELPERS, "campaign/source schema differs")
    require(config.get("selection_protocol_sha256") == PROTOCOL_SHA256, "predeclared raw selection differs")
    require((config.get("edit_index"), config.get("horizon"), config.get("selected")) == (18, 19, [9, 18, 19]), "declared history differs")
    require(config.get("qwen_rope_buffer_policy") == ROPE_POLICY and config.get("attention_dispatch") == dict(backend="sdpa", gqa_policy="repeat_kv"), "original Qwen policy differs")
    require(config.get("transport_limits") == dict(max_expanded_bytes=2 << 30, max_member_bytes=64 << 20), "finite output bounds differ")
    require(config.get("max_wall_seconds") == 600 and config.get("forecast_transaction_bytes") == 64 << 20 and config.get("num_threads") == 4, "finite work bounds differ")
    for key in (*PATH_KEYS, "selection_protocol", "parameter_root", "tokenizer_path", *RUNTIME_KEYS):
        require(isinstance(config.get(key), str) and bool(config[key]), "missing Qwen-only config: " + key)
    for key in PATH_KEYS:
        require(isinstance(config.get(key + "_sha256"), str) and len(config[key + "_sha256"]) == 64, "missing file identity: " + key)
    require(config["retrieval_script_sha256"] == RETRIEVER_SHA256, "qualified standalone retriever differs")
    require(not any(key.startswith(("flux_", "vae_")) for key in config), "non-Qwen component config forbidden")
    return contract()


def inventory(root, *, exclude=()):
    records = {}
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "output symlink forbidden")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in exclude:
            records[relative] = dict(bytes=path.stat().st_size, sha256=sha(path))
    return records


def verify_inventory(root, records):
    for relative, row in records.items():
        path = Path(root) / relative
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "sealed external file changed: " + relative)


def tensor_record(root, relative, value):
    import numpy as np
    import torch

    require(value.layout == torch.strided, "only dense strided saved tensors supported")
    path = Path(root) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = value.detach().cpu().contiguous()
    array = cpu.view(torch.uint16).numpy() if value.dtype == torch.bfloat16 else cpu.numpy()
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return dict(path=relative, sha256=sha(path), bytes=path.stat().st_size, shape=list(value.shape),
        stride=list(value.stride()), logical_dtype=str(value.dtype), stored_dtype=str(array.dtype),
        content_sha256=hashlib.sha256(memoryview(array).cast("B")).hexdigest(), device_at_capture=str(value.device))


def live_input_matches(value, record):
    cpu = value.detach().cpu().contiguous().numpy()
    return list(value.shape) == record["shape"] and str(value.dtype) == record["logical_dtype"] and \
        hashlib.sha256(memoryview(cpu).cast("B")).hexdigest() == record["content_sha256"]


def load_array(root, row):
    import numpy as np

    path = Path(root) / row["path"]
    require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "saved array differs")
    array = np.load(path, allow_pickle=False)
    require(list(array.shape) == row["shape"] and str(array.dtype) == row["stored_dtype"] and
            hashlib.sha256(memoryview(array).cast("B")).hexdigest() == row["content_sha256"], "saved payload differs")
    return (array.astype(np.uint32) << 16).view(np.float32) if row["logical_dtype"] == "torch.bfloat16" else array


def tree_record(root, stem, value):
    import torch

    if isinstance(value, torch.Tensor):
        return dict(tensor=tensor_record(root, stem + ".npy", value))
    if isinstance(value, (list, tuple)):
        return [tree_record(root, stem + "/" + str(index), child) for index, child in enumerate(value)]
    if isinstance(value, dict):
        return {str(key): tree_record(root, stem + "/" + str(key), child) for key, child in value.items()}
    require(value is None or type(value) in (bool, str, int, float), "unsupported native context value")
    return value


def signed_hash(root, negative, positive):
    import numpy as np

    n, p = load_array(root, negative), load_array(root, positive)
    require(n.shape == p.shape, "signed response shape differs")
    delta = n.astype(np.float64) - p.astype(np.float64)
    return dict(shape=list(delta.shape), elements=delta.size, dtype="float64",
        sha256=hashlib.sha256(memoryview(delta).cast("B")).hexdigest(), orientation="negative-minus-own-positive")


def compare(root, prediction, reference):
    import numpy as np

    p, n = load_array(root, prediction), load_array(root, reference)
    require(p.shape == n.shape, "comparison shape differs")
    delta = p.astype(np.float64) - n.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta * delta)))
    scale = float(np.sqrt(np.mean(n.astype(np.float64) ** 2)))
    zero = (p == 0) & (n == 0)
    return dict(elements=p.size, orientation="prediction-minus-reference", absolute_rms=rms,
        reference_rms=scale, reference_relative_rms=None if scale == 0 else rms / scale,
        signed_mean=float(delta.mean()), max_abs=float(np.abs(delta).max()),
        p95_abs=float(np.quantile(np.abs(delta), .95)), p99_abs=float(np.quantile(np.abs(delta), .99)),
        changed_numeric=int(np.count_nonzero(delta)), bit_exact=prediction["content_sha256"] == reference["content_sha256"],
        signed_zero_mismatches=int(np.count_nonzero(zero & (np.signbit(p) != np.signbit(n)))))


def compare_response(root, pminus, pplus, nminus, nplus):
    import numpy as np

    p = load_array(root, pminus).astype(np.float64) - load_array(root, pplus).astype(np.float64)
    n = load_array(root, nminus).astype(np.float64) - load_array(root, nplus).astype(np.float64)
    delta = p - n
    pp, nn, dot = float(np.sum(p * p)), float(np.sum(n * n)), float(np.sum(p * n))
    return dict(elements=p.size, orientation="predicted-negative-minus-own-positive minus reference-negative-minus-own-positive",
        predicted_response_rms=float(np.sqrt(pp / p.size)), reference_response_rms=float(np.sqrt(nn / n.size)),
        error_rms=float(np.sqrt(np.mean(delta * delta))), reference_relative_rms=None if nn == 0 else float(np.sqrt(np.sum(delta * delta) / nn)),
        signed_mean=float(delta.mean()), max_abs=float(np.abs(delta).max()),
        dot=dot, cosine=None if pp == 0 or nn == 0 else dot / (pp * nn) ** .5)


@contextmanager
def learned_guard(model, released, counts):
    def entry(_module, _args):
        if not released["native"]:
            counts["forecast_module_attempts"] += 1
            raise RuntimeError("native learned entry forbidden before all forecasts seal")
        counts["native_module_entries"] += 1
    handles = [module.register_forward_pre_hook(entry) for module in model.modules()]
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def primitive_census(phase, counts):
    from torch.nn import functional as F

    original_sdpa, original_linear = F.scaled_dot_product_attention, F.linear

    def sdpa(*args, **kwargs):
        counts[phase["name"] + "_observed_SDPA"] += 1
        return original_sdpa(*args, **kwargs)

    def linear(*args, **kwargs):
        if phase["name"] == "shortcut":
            counts["shortcut_observed_linear"] += 1
        return original_linear(*args, **kwargs)

    F.scaled_dot_product_attention, F.linear = sdpa, linear
    try:
        yield
    finally:
        F.scaled_dot_product_attention, F.linear = original_sdpa, original_linear


def execute_cases(output, model, cases, *, graph, history, shortcut, edit_index=18, horizon=19,
                  selected=(9, 18, 19), rope_policy=None, source_identity=None, max_wall_seconds=600,
                  transaction_bytes=64 << 20, before_execute=None, job_id=None, request_sha256=None):
    """Shared actual-model orchestration for production and tiny CPU mechanics.

    cases is a fixed three-item list with id, IDs, attention mask, and metadata.
    No native output is accepted by the forecast callback. Arrays are serialized
    per case, then live histories released; no persistent parameter bank is made.
    """
    import torch

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output = Path(output)
    require(output.is_dir() and len(cases) == 3 and len({row["id"] for row in cases}) == 3, "fixed three-context cohort required")
    rope_policy = dict(ROPE_POLICY if rope_policy is None else rope_policy)
    sources = dict(source_identity or {"synthetic_component": "1" * 64})
    declared = contract(edit_index=edit_index, horizon=horizon, native_layers=len(model.model.layers))
    counts = dict(forecast_module_attempts=0, native_module_entries=0, native_authorizations=0,
                  functional_reference_blocks=0, functional_reference_SDPA=0, shortcut_down_projections=0,
                  native_blocks=0, native_SDPA=0, native_full_model_calls=0, native_suffix_calls=0,
                  functional_observed_SDPA=0, shortcut_observed_SDPA=0, native_observed_SDPA=0, shortcut_observed_linear=0)
    released = dict(native=False)
    started = time.monotonic()
    events = []

    def event(kind, **fields):
        require(time.monotonic() - started <= max_wall_seconds, "finite campaign wall bound exceeded")
        row = dict(number=len(events), kind=kind, native_module_entries=counts["native_module_entries"], **fields)
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as handle:
            handle.write(canonical(row) + b"\n")

    def save_history(case_root, result, ports, *, native=False):
        retained = dict(common={str(index): tensor_record(output, f"{case_root}/common/{index}.npy", value)
                               for index, value in result["common"].items() if index <= horizon},
            minus={str(index): tensor_record(output, f"{case_root}/minus/{index}.npy", value) for index, value in result["minus"].items()},
            ports=ports, counts=result["counts"])
        if native:
            retained.update(source_context=tree_record(output, case_root + "/source-context", result["source_context"]),
                consumed=tree_record(output, case_root + "/consumed", result["consumed"]),
                module_entries=result["module_entries"], block_entries=result["block_entries"], sdpa_calls=result["sdpa_calls"],
                baseline_logits_shape=result["baseline_logits_shape"], genuine_qwen_act=False,
                reference_only={str(index): dict(shape=list(value.shape), dtype=str(value.dtype), retained=False)
                                for index, value in result["common"].items() if index > horizon})
        else:
            retained.update(position_ids=tensor_record(output, case_root + "/position_ids.npy", result["position_ids"]),
                profile=result["profile"], mask_profile=result["mask_profile"])
        return retained

    forecasts, natives, shortcuts, responses, case_inputs = {}, {}, {}, {}, {}
    parameters = dict(model.named_parameters())
    phase = dict(name="functional")
    with learned_guard(model, released, counts), primitive_census(phase, counts):
        for row in cases:
            case = row["id"]
            require(case.replace("-", "").isalnum(), "unsafe case identifier")
            ids, mask = row["input_ids"], row["attention_mask"]
            require(ids.shape == mask.shape and ids.ndim == 2 and ids.shape[0] == 1, "own B1 source geometry differs")
            root = "cases/" + case
            inputs = dict(metadata=row.get("metadata", {}), input_ids=tensor_record(output, root + "/inputs/ids.npy", ids),
                          attention_mask=tensor_record(output, root + "/inputs/mask.npy", mask))
            save(output / root / "inputs.json", inputs)
            case_inputs[case] = inputs
            padded = bool((mask == 0).any())
            dispatch = dict(backend="sdpa", gqa_policy="repeat_kv" if padded else "enable_gqa",
                            source=sources.get("qwen_native", next(iter(sources.values()))))
            positive, ports, state = {}, {"plus": {}, "minus": {}}, dict(shortcut=None, minus_started=False)

            def observe(lane, name, value):
                if lane == "minus" and not state["minus_started"]:
                    require(state["shortcut"] is not None, "negative reference began before positive-only shortcut")
                    state["minus_started"] = True
                    event("functional_negative_reference_started", case=case)
                ports[lane][name] = tensor_record(output, f"{root}/forecast/ports/{lane}/{name}.npy", value)
                if lane == "plus" and name in ("R", "gate", "up"):
                    positive[name] = value.detach().clone()
                if lane == "plus" and name == "Hnext":
                    require(not state["minus_started"] and set(positive) == {"R", "gate", "up"}, "positive frontier incomplete")
                    positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.int64).unsqueeze(0)
                    attention = dict(backend="sdpa", gqa_policy=dispatch["gqa_policy"], query_rows="partially-padded" if padded else "all-attended",
                        mask_identity=inputs["attention_mask"]["content_sha256"],
                        positions_identity=hashlib.sha256(memoryview(positions.cpu().numpy()).cast("B")).hexdigest(),
                        rope_identity=hashlib.sha256(canonical(dict(policy=rope_policy, config=model.config.to_dict()))).hexdigest())
                    weight = parameters[f"model.layers.{edit_index}.mlp.down_proj.weight"]
                    profile = shortcut["make_profile"](positive, weight, source_identity=sources,
                        frontier_kind="own-functional-positive-ports", attention_profile=attention)
                    phase["name"] = "shortcut"
                    try:
                        prediction = shortcut["predict_negative"](positive, weight, profile=profile)
                    finally:
                        phase["name"] = "functional"
                    metadata = {key: child for key, child in prediction.items() if key != "ports"}
                    metadata["ports"] = {key: tensor_record(output, f"{root}/shortcut/{key}.npy", child) for key, child in prediction["ports"].items()}
                    metadata["frontier"] = dict(only_positive_basis=["R", "gate", "up"], basis={key: ports["plus"][key] for key in ("R", "gate", "up")},
                        before_first_minus_callback=True, down_parameter=f"model.layers.{edit_index}.mlp.down_proj.weight")
                    save(output / root / "shortcut.json", metadata)
                    state["shortcut"] = metadata
                    counts["shortcut_down_projections"] += 1
                    event("positive_only_shortcut_completed", case=case, shortcut_sha256=sha(output / root / "shortcut.json"), no_negative_ports_accessed=True)

            event("functional_case_started", case=case)
            result = history["functional_state_history"](graph, ids, mask, parameters, model.config,
                attention_dispatch=dispatch, rope_buffer_policy=rope_policy, edit_index=edit_index, horizon=horizon,
                selected=selected, observe=observe)
            require(state["shortcut"] is not None and state["minus_started"], "shortcut/reference callbacks absent")
            record = save_history(root + "/forecast", result, ports)
            save(output / root / "forecast.json", record)
            forecasts[case], shortcuts[case] = record, state["shortcut"]
            counts["functional_reference_blocks"] += result["counts"]["blocks"]
            counts["functional_reference_SDPA"] += result["counts"]["SDPA"]
            responses[case] = {lane: {key: signed_hash(output, negative[key], ports["plus"][key]) for key in ENDPOINTS}
                for lane, negative in (("shortcut", state["shortcut"]["ports"]), ("full_functional", ports["minus"]))}
            event("functional_case_completed", case=case, dispatch=dispatch)
            del result
            positive.clear()
            state.clear()
        save(output / "forecast-responses.json", dict(orientation="negative-minus-own-positive", cases=responses))
        sealed_files = inventory(output, exclude=("timeline.jsonl",))
        tx = ForecastTransaction(output / "transaction", model_identity="original-qwen3-bias-free",
            input_identity=hashlib.sha256(canonical({row["id"]: row.get("metadata", {}) for row in cases})).hexdigest(),
            source_identity=hashlib.sha256(canonical(sources)).hexdigest(),
            frontier=dict(kind="raw-input", conditional_frontier="own-functional-positive-R-G-U", cases=[row["id"] for row in cases], edit_index=edit_index,
                          horizon=horizon, external_files=sealed_files),
            execution_scope="Three own raw-input prefixes, positive-only conditional shortcuts and complete negative functional references before native verification.",
            array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=1, max_bytes=transaction_bytes,
                         max_files=30000, max_wall_ms=max_wall_seconds * 1000))
        tx.predict("cohort", lambda _ctx: dict(predictions=dict(external_files=sealed_files, cases=list(forecasts),
            response_hashes=responses, counts=dict(counts), native_targets_at_seal=0)))
        seal = tx.seal()
        event("cohort_globally_sealed", seal_sha256=seal["seal_sha256"], forecast_cases=3, native_targets_at_seal=0)

        def execute(_ctx):
            if before_execute is not None:
                before_execute(output, sealed_files)
            verify_inventory(output, sealed_files)
            released["native"] = True
            event("native_targets_released", seal_sha256=seal["seal_sha256"])

            def authorize():
                require(released["native"] and len(forecasts) == 3 and counts["forecast_module_attempts"] == 0, "native target lacks complete global seal")
                counts["native_authorizations"] += 1
                event("native_component_authorized")

            for row in cases:
                case, root = row["id"], "cases/" + row["id"]
                require(live_input_matches(row["input_ids"], case_inputs[case]["input_ids"]) and
                        live_input_matches(row["attention_mask"], case_inputs[case]["attention_mask"]), "own native IDs/mask changed after forecast")
                ports = {"plus": {}, "minus": {}}

                def capture(lane, name, value):
                    ports[lane][name] = tensor_record(output, f"{root}/native/ports/{lane}/{name}.npy", value)

                phase["name"] = "native"
                result = history["native_state_history"](model, row["input_ids"], row["attention_mask"], authorize_native=authorize,
                    edit_index=edit_index, horizon=horizon, selected=selected, observe=capture)
                record = save_history(root + "/native", result, ports, native=True)
                save(output / root / "native.json", record)
                natives[case] = record
                counts["native_blocks"] += result["counts"]["baseline_blocks"] + result["counts"]["suffix_blocks"]
                counts["native_SDPA"] += result["counts"]["SDPA"]
                counts["native_full_model_calls"] += result["counts"]["full_model_calls"]
                counts["native_suffix_calls"] += result["counts"]["suffix_blocks"]
                event("native_case_completed", case=case)
                del result
            return dict(counts=counts, cases=list(natives))

        tx.execute("native-verification", execute)
        tx.close()
    verify_inventory(output, sealed_files)
    require(counts["forecast_module_attempts"] == 0, "forecast module attempts occurred")
    for key in ("functional_reference_blocks", "functional_reference_SDPA", "shortcut_down_projections", "native_blocks",
                "native_SDPA", "native_full_model_calls", "native_suffix_calls", "native_authorizations"):
        require(counts[key] == declared[key], "actual physical count differs: " + key)
    require(counts["functional_observed_SDPA"] == declared["functional_reference_SDPA"] and
            counts["native_observed_SDPA"] == declared["native_SDPA"] and counts["shortcut_observed_SDPA"] == 0 and
            counts["shortcut_observed_linear"] == declared["shortcut_down_projections"], "observed primitive census differs")
    comparisons = {}
    for case in forecasts:
        f, n, s = forecasts[case]["ports"], natives[case]["ports"], shortcuts[case]["ports"]
        lanes = dict(shortcut=s, full_functional=f["minus"], native=n["minus"])
        bases = dict(shortcut=f["plus"], full_functional=f["plus"], native=n["plus"])
        pairs = (("shortcut", "full_functional"), ("full_functional", "native"), ("shortcut", "native"))
        premise = {}
        for lane, ports in (("functional", f), ("native", n)):
            premise[lane] = {}
            for name in ("R", "gate", "up"):
                plus = torch.from_numpy(load_array(output, ports["plus"][name])).to(torch.bfloat16)
                minus = torch.from_numpy(load_array(output, ports["minus"][name])).to(torch.bfloat16)
                premise[lane][name] = history["oddness_diagnostic"](plus, minus)
        comparisons[case] = dict(endpoint={a + "-minus-" + b: {key: compare(output, lanes[a][key], lanes[b][key]) for key in ENDPOINTS} for a, b in pairs},
            signed_response={a + "-minus-" + b: {key: compare_response(output, lanes[a][key], bases[a][key], lanes[b][key], bases[b][key]) for key in ENDPOINTS} for a, b in pairs},
            premise_diagnostics=premise,
            native_response_hashes={key: signed_hash(output, n["minus"][key], n["plus"][key]) for key in ENDPOINTS})
    save(output / "comparisons.json", comparisons)
    result = dict(schema=SCHEMA, job_id=job_id, request_sha256=request_sha256, counts=counts, contract=declared, forecast_seal=seal["seal_sha256"],
        transaction=verify_transaction(output / "transaction", expected_seal_sha256=seal["seal_sha256"], max_bytes=transaction_bytes),
        cases=list(forecasts), comparisons="comparisons.json", native_execution="three own native full baselines plus one suffix block each; no Qwen Acts",
        seconds=time.monotonic() - started, mechanics_status="completed", numerical_status="observations; no accuracy or oddness gate")
    save(output / "summary.json", result)
    return result


def parameter_signature(model):
    """Stream live BF16 payloads, retaining no FP32 parameter bank."""
    import torch

    records = {}
    for name, value in model.named_parameters():
        require(value.dtype == torch.bfloat16, "learned parameter dtype differs")
        flat, digest = value.detach().reshape(-1), hashlib.sha256()
        for start in range(0, flat.numel(), 4 << 20):
            raw = flat[start:start + (4 << 20)].cpu().contiguous().view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast("B"))
        records[name] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())
    return records


def runtime_contract(config):
    import jinja2.environment
    import jinja2.sandbox
    import tokenizers
    import tokenizers.tokenizers
    import torch
    import transformers
    import transformers.models.qwen3.modeling_qwen3

    paths = dict(qwen_native_source_sha256=Path(inspect.getfile(transformers.models.qwen3.modeling_qwen3)),
        tokenizers_wrapper_source_sha256=Path(inspect.getfile(tokenizers)),
        tokenizers_native_source_sha256=Path(inspect.getfile(tokenizers.tokenizers)),
        jinja_environment_source_sha256=Path(inspect.getfile(jinja2.environment)), jinja_sandbox_source_sha256=Path(inspect.getfile(jinja2.sandbox)))
    records = {}
    for key, path in paths.items():
        expected, actual = config[key], sha(path)
        require(expected == actual or (key == "tokenizers_native_source_sha256" and expected == "runtime-bound"), "installed original source differs: " + key)
        records[key] = dict(path=str(path), sha256=actual, bytes=path.stat().st_size, expected_sha256=expected,
            binding="runtime-observed-on-CUDA-host" if expected == "runtime-bound" else "staged-source-hash")
    return dict(torch=torch.__version__, transformers=transformers.__version__, tokenizers=tokenizers.__version__, sources=records,
        flags={**helpers()["qwen_sign_shortcut.py"]["execution_profile"](), "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
               "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "num_threads": torch.get_num_threads()}, selected_kernels="not independently profiled")


def run(config):
    """Original-weight entrypoint. Never used by local synthetic qualification."""
    import torch
    from transformers import AutoTokenizer

    validate_config(config)
    require(torch.cuda.is_available(), "production source campaign requires CUDA")
    h = helpers()
    loader = h["flux_functional_qwen_prefix_worker.py"]
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    for key in PATH_KEYS:
        require(sha(config[key]) == config[key + "_sha256"], "source metadata differs: " + key)
        shutil.copyfile(config[key], output / Path(config[key]).name)
    raw_rows = selected_raw_inputs(config["selection_protocol"], output)
    qconfig = read(config["qwen_config"])
    require((qconfig["hidden_size"], qconfig["intermediate_size"], qconfig["num_hidden_layers"], qconfig["num_attention_heads"], qconfig["num_key_value_heads"], qconfig["head_dim"]) ==
        (2560, 9728, 36, 32, 8, 128), "original unequal-width Qwen geometry differs")
    require(not qconfig.get("attention_bias", False) and qconfig.get("attention_dropout", 0) == 0 and qconfig.get("tie_word_embeddings", True), "original sign-source assumptions differ")
    torch.set_num_threads(config["num_threads"])
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = runtime_contract(config)
    save(output / "runtime.json", runtime)
    save(output / "plan.json", config)
    binding = read(config["qwen_source_binding"])
    tokenizer_files = loader["_copy_tokenizer_sources"](output, binding)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True, trust_remote_code=False)
    cases = []
    for row in raw_rows:
        formatted, ids, mask = loader["_tokenize"](tokenizer, row["raw_text"], config)
        untruncated = tokenizer.apply_chat_template([dict(role="user", content=row["raw_text"])], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, padding=False, truncation=False, return_dict=True)
        cases.append(dict(id=row["case_id"], input_ids=ids.to("cuda"), attention_mask=mask.to("cuda"),
            metadata=dict(raw_text=row["raw_text"], raw_sha256=row["raw_sha256"], formatted_prompt=formatted,
                formatted_sha256=hashlib.sha256(formatted.encode()).hexdigest(), intended_regime=row["intended_mask_regime"],
                actual_attended_rows=int(mask.sum()), actual_padded_rows=int((mask == 0).sum()),
                untruncated_token_count=len(untruncated["input_ids"]))))
    bank, manifest = loader["_load_recipe_bank"](Path(__file__).with_name("flux_source_receiver.py"), Path(config["qwen_parameter_manifest"]), config["parameter_root"])
    require(len(manifest["artifacts"]) == 398, "original Qwen recipe count differs")
    model = loader["_load_qwen"](config, torch.device("cuda"), bank, manifest)
    recipe_io = dict(payload_reads=bank.payload_reads, load_requests=bank.reads, cache_hits=bank.cache_hits,
                     parameter_payload_bytes=bank.bytes, peak_cached_payload_bytes=bank.peak_retained,
                     checkpoint_files_verified_once=len(bank.signatures))
    bank.bits.clear()
    bank.retained = 0
    before = parameter_signature(model)
    expected = {row["source"]["tensor"]: row["source"] for row in manifest["artifacts"]}
    require(set(before) == set(expected) and all(row["sha256"] == expected[name]["payload_sha256"] and row["shape"] == expected[name]["shape"] for name, row in before.items()), "restored live Qwen parameter identity differs")
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    require(tied, "original tied LM alias absent")
    save(output / "source.json", dict(job_id=config["job_id"], request_sha256=config["request_sha256"],
        source_hashes=config.get("request_source_hashes", {}), helper_source_sha256=HELPERS, tokenizer_files=tokenizer_files,
        parameter_manifest_sha256=config["qwen_parameter_manifest_sha256"], original_parameter_payloads=before,
        parameter_recipe_io=recipe_io,
        qwen_tied_alias_before=tied,
        rope_buffers={name: tensor_record(output, "rope-before/" + name + ".npy", value) for name, value in model.named_buffers()},
        hydration=dict(component_hydrations=1, cuda_placements=1, strategy="original loader meta BF16 direct device allocation; no full CPU FP32 duplicate")))
    result = execute_cases(output, model, cases, graph=h["flux_functional_qwen_prefix.py"], history=h["qwen_h18_state_history.py"], shortcut=h["qwen_sign_shortcut.py"],
        source_identity={**HELPERS, "qwen_native": runtime["sources"]["qwen_native_source_sha256"]["sha256"]},
        max_wall_seconds=config["max_wall_seconds"], transaction_bytes=config["forecast_transaction_bytes"],
        job_id=config["job_id"], request_sha256=config["request_sha256"])
    after = parameter_signature(model)
    require(after == before and (model.lm_head.weight is model.model.embed_tokens.weight) == tied, "original learned payload/alias changed")
    save(output / "parameter-identity-after.json", dict(payloads=after, qwen_tied_alias_after=tied,
        rope_buffers={name: tensor_record(output, "rope-after/" + name + ".npy", value) for name, value in model.named_buffers()}))
    model.to("cpu")
    return result


def worker(config_path):
    print("qwen-sign-shortcut bootstrap; no native target before global cohort seal", flush=True)
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path)
    validate_config(config)
    request_path = os.environ.get("SATURN_DEBUG_REQUEST")
    require(request_path is not None, "actual source-bound request required")
    request = read(request_path)
    require(request_identity(request) == request.get("request_sha256"), "canonical experimental request identity differs")
    require(request["config"] == config, "debug request/config differs before output derivation")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "executed source differs: " + name)
    derived = dict(config, request_source_hashes=request["source_hashes"], request_sha256=request["request_sha256"])
    job_id = os.environ["MRUN_JOB_ID"]
    require(job_id.startswith("job-") and "/" not in job_id, "invalid actual job ID")
    derived["job_id"] = job_id
    derived["output"] = str(Path(config["output_scratch"]) / job_id / config["output_name"])
    run(derived)
    output = Path(derived["output"])
    shutil.copyfile(request_path, output / "request.json")
    shutil.copyfile(config_path, output / "declared-config.json")
    receipt = helpers()["flux_image_tail_microscope.py"]["finalize"](output, job_id=job_id,
        request_sha256=request["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(receipt).decode(), flush=True)
    faulthandler.cancel_dump_traceback_later()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    worker(parser.parse_args().config)

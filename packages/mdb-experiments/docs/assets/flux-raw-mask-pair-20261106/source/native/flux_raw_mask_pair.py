"""Paired raw-input mask/GQA assay using unchanged qualified arithmetic graphs.

Two B1 cases are phase-batched. All complete image forecasts and signed-response
recipes are sealed before either native learned target. The raw text contrast
also changes content/truncation; it does not isolate a causal padding effect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

PRIOR = "flux_raw_to_image.py"
PRIOR_SHA256 = "1c16c1a90442194e97398ea8d0cda5cb4d7686434d45a16eec6ba6fa9b103522"
SHORT = ("A cobalt blue cube to the right of a matte orange ring on a gray tabletop. "
         "Two separate objects, simple studio product photograph.")
LONG = SHORT + 128 * " The tabletop is gray and the lighting is even."
CASES = {"short": SHORT, "long": LONG}
EXECUTION_ID = "mdb-flux-raw-mask-pair-v1"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def prior():
    path = Path(__file__).with_name(PRIOR)
    require(sha(path) == PRIOR_SHA256, "frozen raw composition source differs")
    return runpy.run_path(str(path))


def contract(config=None):
    return {
        "schema": "mdb-flux-raw-mask-pair-count-contract-v1",
        "execution_id": EXECUTION_ID,
        "case_order": list(CASES), "batch_size": 1, "seed": 277,
        "qwen_functional_calls": 2, "qwen_native_calls": 2,
        "qwen_forecast_layers_per_case": 27, "qwen_native_layers_per_case": 36,
        "forecast_hidden_states_per_case": 28,
        "functional_transitions": 224, "native_transitions": 224,
        "functional_decode_calls": 2, "native_vae_calls": 4,
        "noise_draws": 4, "component_hydrations": 3,
        "cuda_placements": 6, "cpu_offloads": 6, "max_gpu_components": 1,
        "forecast_module_entries": 0, "pngs": 6,
        "global_forecast_seals": 1, "persistent_weight_banks": 0,
        "sdpa_argument_observations": {"forecast": 54, "native": 72},
        "scope": "conditional shared-primitive composition; content/truncation and mask/dispatch contrast, not isolated padding causality or independent kernels",
    }


def make_config(base):
    config = dict(base)
    for key in ("worker_sha256", "request_sha256", "job_id", "_request_record", "output"):
        config.pop(key, None)
    config.update({
        "schema": "mdb-flux-raw-mask-pair-plan-v1", "prompt": SHORT,
        "cases": dict(CASES), "case_order": list(CASES), "seed": 277,
        "output_name": "raw-mask-pair-result", "max_wall_seconds": 1200,
        "forecast_transaction_bytes": 1 << 30,
        "transport_limits": {"max_expanded_bytes": 8 << 30, "max_member_bytes": 134221824},
        "expected_mask_regimes": {"short": "partially-padded", "long": "all-attended"},
        "response_recipe": {"orientation": "long-minus-short", "arithmetic": "float64(long)-float64(short)",
                            "hash_encoding": "C-order little-endian float64", "duplicate_delta_bank": False},
    })
    config["helper_source_sha256"] = {**config.get("helper_source_sha256", {}), PRIOR: PRIOR_SHA256}
    return config


def validate_config(config):
    require(config.get("schema") == "mdb-flux-raw-mask-pair-plan-v1", "paired plan schema differs")
    require(config.get("cases") == CASES and config.get("case_order") == list(CASES), "fixed raw prompt bytes/order differ")
    require(config.get("expected_mask_regimes") == {"short": "partially-padded", "long": "all-attended"}, "mask hypotheses differ")
    require(config.get("response_recipe") == make_config({})["response_recipe"], "response recipe differs")
    require(config.get("transport_limits") == {"max_expanded_bytes": 8 << 30, "max_member_bytes": 134221824}, "paired storage bounds differ")
    old = prior()
    # Only the predecessor's configuration validator is reused; its run() and
    # function globals are never changed. The arithmetic profile stays identical.
    source_config = dict(config, schema="mdb-flux-raw-to-image-plan-v1", prompt=SHORT)
    source_config.pop("worker_sha256", None)
    old["validate_config"](source_config)
    if config.get("worker_sha256"):
        require(sha(__file__) == config["worker_sha256"], "paired worker source differs")
    require(config["attention_dispatch"]["backend"] == "sdpa", "paired dispatch requires SDPA")
    return contract(config)


def dispatch_for_mask(mask, source):
    attended = int(mask.sum().item())
    total = mask.numel()
    require(attended > 0 and bool(((mask == 0) | (mask == 1)).all()), "invalid retained attention mask")
    all_attended = attended == total
    return {
        "backend": "sdpa", "gqa_policy": "enable_gqa" if all_attended else "repeat_kv",
        "source": source,
    }, {"attended_rows": attended, "padded_rows": total - attended,
        "regime": "all-attended" if all_attended else "partially-padded",
        "expected_mask": "None" if all_attended else "explicit-bool-causal-key-padding",
        "expected_is_causal": all_attended, "expected_enable_gqa": all_attended}


def verify_sdpa_observations(rows, case, lane, expected_count, regime):
    selected = [row for row in rows if row["case"] == case and row["lane"] == lane]
    require(len(selected) == expected_count, "actual Qwen SDPA occurrence count differs")
    # Argument/profile differences are retained diagnostics, not precision gates.
    return {"calls": len(selected), "expected_regime": regime,
            "all_argument_profiles_match_declared_regime": all(
                (row["mask"] is None) == (regime == "all-attended")
                and row["is_causal"] == (regime == "all-attended")
                and row["enable_gqa"] == (regime == "all-attended")
                for row in selected),
            "selected_kernel": "unprofiled"}


@contextmanager
def record_sdpa(rows, *, case, lane):
    """Observe actual primitive arguments, call the original exactly once."""
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention

    def descriptor(value):
        return {"shape": list(value.shape), "stride": list(value.stride()),
                "dtype": str(value.dtype), "device": str(value.device)}

    def observe(query, key, value, *args, **kwargs):
        # Preserve every positional/keyword argument, including explicit defaults.
        mask = args[0] if args else kwargs.get("attn_mask")
        causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
        rows.append({"case": case, "lane": lane, "ordinal": len(rows),
                     "query": descriptor(query), "key": descriptor(key), "value": descriptor(value),
                     "mask": None if mask is None else descriptor(mask),
                     "is_causal": bool(causal), "enable_gqa": bool(kwargs.get("enable_gqa", False)),
                     "configured_backend": "sdpa", "selected_kernel": "unprofiled"})
        return original(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = observe
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


def orchestrate(backend, seal, execute):
    """Small testable phase scheduler; no case reaches native before the cohort."""
    forecasts = {case: {} for case in CASES}
    native = {case: {} for case in CASES}
    for phase in ("qwen", "flux", "vae"):
        backend.place(phase, True)
        try:
            for case in CASES:
                forecasts[case][phase] = backend.forecast(phase, case, forecasts[case])
        finally:
            backend.place(phase, False)
    seal(forecasts)

    def native_targets():
        backend.release()
        for phase in ("qwen", "flux", "vae"):
            backend.place(phase, True)
            try:
                for case in CASES:
                    native[case][phase] = backend.native(phase, case, forecasts[case], native[case])
            finally:
                backend.place(phase, False)
        return native

    execute(native_targets)
    return forecasts, native


def response_hash(short, long):
    """Hash exact FP64 signed response without retaining another full tensor."""
    import numpy as np

    require(tuple(short.shape) == tuple(long.shape), "response field geometry differs")
    digest = hashlib.sha256()
    left = short.detach().cpu().reshape(-1)
    right = long.detach().cpu().reshape(-1)
    for offset in range(0, left.numel(), 1 << 18):
        delta = right[offset:offset + (1 << 18)].double() - left[offset:offset + (1 << 18)].double()
        digest.update(np.asarray(delta.numpy(), dtype="<f8").tobytes())
    return {"shape": list(short.shape), "elements": short.numel(), "dtype": "float64",
            "orientation": "long-minus-short", "sha256": digest.hexdigest()}


def _tensor_leaves(value, path=""):
    if hasattr(value, "detach"):
        yield path, value
    elif isinstance(value, dict) or hasattr(value, "items"):
        for key, item in sorted(value.items()):
            yield from _tensor_leaves(item, path + "/" + str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, path + "/" + str(index))


def seal_responses(output, old):
    """Stream forecast pairs only. No native artifact exists at this boundary."""
    import numpy as np
    import torch
    from saturn_pub.store import LocalStore

    roots = {case: output / "cases" / case for case in CASES}
    result = {"schema": "mdb-raw-mask-pair-responses-v1", "recipe": make_config({})["response_recipe"], "qwen": {}, "denoiser": []}

    def load(root, record):
        array = np.load(root / record["path"], allow_pickle=False)
        if array.dtype == np.uint16:
            return torch.from_numpy(array).view(torch.bfloat16)
        return torch.from_numpy(array)

    q = {case: read(root / "forecast-qwen.json") for case, root in roots.items()}
    for index in range(28):
        result["qwen"][str(index)] = response_hash(load(roots["short"], q["short"]["hidden_states"][index]), load(roots["long"], q["long"]["hidden_states"][index]))
    result["qwen"]["conditioning"] = response_hash(load(roots["short"], q["short"]["conditioning"]), load(roots["long"], q["long"]["conditioning"]))
    stores = {case: LocalStore(root / "forecast-state-store") for case, root in roots.items()}
    rows = {case: read(root / "forecast-state-rows.json") for case, root in roots.items()}
    for index, (left, right) in enumerate(zip(rows["short"], rows["long"], strict=True)):
        require(left["clock"] == right["clock"], "paired forecast clocks differ")
        a = stores["short"].load(left["cut"]).payload
        b = stores["long"].load(right["cut"]).payload
        leaves_a, leaves_b = dict(_tensor_leaves(a)), dict(_tensor_leaves(b))
        require(leaves_a.keys() == leaves_b.keys(), "paired field liveness differs")
        fields = {name: response_hash(leaves_a[name], leaves_b[name]) for name in leaves_a}
        result["denoiser"].append({"number": index - 1, "clock": left["clock"], "fields": fields})
        del a, b, leaves_a, leaves_b
    dec = {case: read(root / "decoder-forecast.json") for case, root in roots.items()}
    result["decoder"] = {name: response_hash(load(roots["short"], dec["short"][name]), load(roots["long"], dec["long"][name])) for name in ("decoded", "pixels")}
    save(output / "forecast-responses.json", result)
    return result


def run(config, *, progress=None):
    declared = validate_config(config)
    old = prior()
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from transformers import AutoTokenizer

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    output = Path(config["output"]).resolve()
    require(all(not output.is_relative_to(Path(config[key]).resolve().parent) for key in old["_source_keys"]()), "output overlaps immutable source")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    if config.get("_request_record"):
        save(output / "request.json", config["_request_record"])
    started = time.monotonic()
    counters = {key: 0 for key in ("component_hydrations", "cuda_placements", "cpu_offloads", "noise_draws", "functional_transitions", "native_transitions", "functional_decode_calls", "native_vae_calls", "native_qwen_calls", "native_module_entries", "guard_refusals")}
    released = {"native": False}
    events, placements, sdpa_rows = [], [], []
    case_counts = {case: {} for case in CASES}
    data = {case: {} for case in CASES}

    def event(kind, **detail):
        row = {"kind": kind, "seconds": time.monotonic() - started, **detail}
        require(row["seconds"] <= config["max_wall_seconds"], "paired finite wall exceeded")
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
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(config.get("source_reduction_policy", True))
    require(torch.cuda.is_available(), "production pair requires CUDA")
    modules = old["_helper_modules"]({key: value for key, value in config.items() if key != "worker_sha256"})
    qw, qg = modules[old["QWEN_WORKER"]], modules[old["QWEN_HELPER"]]
    dg, vg = modules[old["DENOISER_HELPER"]], modules[old["DECODER_HELPER"]]
    rope = config.get("rope_buffer_policy", config.get("qwen_rope_buffer_policy"))
    config = dict(config, rope_buffer_policy=rope)
    runtime = qw["_runtime_contract"](config, torch)
    runtime["native_diffusers_sources"] = old["_native_source_contract"](config)["sources"]
    runtime["case_dispatch"] = "retained mask-bound; actual SDPA args observed separately; selected kernels unprofiled"
    save(output / "runtime-contract.json", runtime)
    sources = {}
    for key in old["_source_keys"]():
        source = Path(config[key])
        target = output / "source" / key / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[key] = {"path": str(target.relative_to(output)), "sha256": sha(target), "bytes": target.stat().st_size}
    binding = read(config["qwen_source_binding"])
    tokenizer_files = qw["_copy_tokenizer_sources"](output, binding)
    tokenizer = AutoTokenizer.from_pretrained(binding["checkpoint"]["model_root"] + "/tokenizer", local_files_only=True, padding_side="right")
    metadata = read(output / "source/tokenizer/tokenizer_config.json")
    require(tokenizer.is_fast and type(tokenizer).__name__ == metadata["tokenizer_class"], "pinned parser differs")
    for case, prompt in CASES.items():
        root = output / "cases" / case
        root.mkdir(parents=True)
        formatted, ids, mask = qw["_tokenize"](tokenizer, prompt, config)
        uncropped = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
            padding=False, truncation=False,
        )
        dispatch, mask_record = dispatch_for_mask(mask, config["attention_dispatch"]["source"])
        mask_record["expected_regime"] = config["expected_mask_regimes"][case]
        mask_record["expected_regime_observed"] = mask_record["regime"] == mask_record["expected_regime"]
        data[case].update(root=root, formatted=formatted, ids=ids, mask=mask, dispatch=dispatch, mask_record=mask_record)
        save(root / "input.json", {"prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "formatted_chat": formatted, "dispatch": dispatch, "mask": mask_record, "uncropped_token_count": len(uncropped), "truncation": "pinned tokenizer right truncation at512; no text tuning", "raw_content_contrast_not_isolated_mask": True})
    qb, qm = qw["_load_recipe_bank"](Path(config["qwen_recipe_helper"]), Path(config["qwen_manifest"]), config["qwen_parameter_root"])
    qwen = qw["_load_qwen"](config, torch.device("cpu"), qb, qm)
    flux, _fb, _fm = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae, _vb, _vm = old["_load_vae"](config, runpy.run_path(config["vae_recipe_helper"]))
    models = {"qwen": qwen, "flux": flux, "vae": vae}
    prefixes = {"qwen": None, "flux": None, "vae": ("post_quant_conv.", "decoder.", "bn.")}
    identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    tied = qwen.lm_head.weight is qwen.model.embed_tokens.weight
    require(tied, "Qwen tied parameter alias missing")
    counters["component_hydrations"] = 3
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1) * 2)
    pipeline = {"Flux2KleinPipeline": Flux2KleinPipeline}

    class Backend:
        def place(self, phase, to_cuda):
            model = models[phase]
            before = old["_component_signature"](model, prefixes=prefixes[phase])
            device = "cuda" if to_cuda else "cpu"
            if phase == "vae":
                model.post_quant_conv.to(device)
                model.decoder.to(device)
                require(model.bn.running_mean.device.type == model.bn.running_var.device.type == "cpu", "BN frontend left CPU")
            else:
                model.to(device)
            after = old["_component_signature"](model, prefixes=prefixes[phase])
            old["_assert_signature_equal"](before, after, phase)
            placements.append({"component": phase, "direction": device, "before_signature": before, "after_signature": after})
            counters["cuda_placements" if to_cuda else "cpu_offloads"] += 1
            event("component_placement", component=phase, direction=device)
            if not to_cuda:
                torch.cuda.empty_cache()

        def release(self):
            require(tx is not None and not released["native"], "native release outside sealed cohort")
            released["native"] = True
            event("native_targets_released", forecast_seal=seal_record["seal_sha256"])

        def forecast(self, phase, case, _prior):
            return self.stage(phase, case, "forecast")

        def native(self, phase, case, _forecast, _prior):
            return self.stage(phase, case, "native")

        def stage(self, phase, case, lane):
            item = data[case]
            root, counts = item["root"], case_counts[case]
            event("phase_entry", case=case, component=phase, lane=lane, native_module_entries=counters["native_module_entries"])
            if phase == "qwen":
                ids, mask = item["ids"].to("cuda"), item["mask"].to("cuda")
                with record_sdpa(sdpa_rows, case=case, lane=lane), torch.inference_mode():
                    if lane == "forecast":
                        parameters = {"model.embed_tokens.weight": qwen.model.embed_tokens.weight}
                        state = qwen.state_dict()
                        for layer in range(27):
                            parameters.update({f"model.layers.{layer}.{suffix}": state[f"model.layers.{layer}.{suffix}"] for suffix in qg["QWEN_LAYER_PARAMETER_SUFFIXES"]})
                        bound = qg["bind_prospective_tokenized_input"](ids, mask, tokenizer_binding={"cohort": "prospective-new", "parser_stage": "tokenizer", "implementation_pinned": True, "template_files": tokenizer_files})
                        result = qg["functional_qwen_prefix"](bound["input_ids"], bound["attention_mask"], parameters, qwen.config, attention_dispatch=item["dispatch"], rope_buffer_policy=rope, prefix_layers=27, selected_hidden_states=(9, 18, 27))
                        states, conditioning = result["hidden_states"], result["conditioning"]
                    else:
                        result = qwen(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
                        states = result.hidden_states
                        require(len(states) == 37, "native full Qwen state count differs")
                        conditioning = torch.cat([states[index] for index in (9, 18, 27)], dim=-1)
                        counters["native_qwen_calls"] += 1
                record = {"prompt": CASES[case], "formatted_chat": item["formatted"], "dispatch": item["dispatch"], "mask_observation": item["mask_record"],
                          "input_ids": old["_save_tensor"](root, f"qwen/{lane}/input-ids.npy", ids, logical_dtype="torch.int64"),
                          "attention_mask": old["_save_tensor"](root, f"qwen/{lane}/attention-mask.npy", mask, logical_dtype="torch.int64"),
                          "hidden_states": [old["_save_tensor"](root, f"qwen/{lane}/hidden-{index:02d}.npy", states[index]) for index in range(28)],
                          "conditioning": old["_save_tensor"](root, f"qwen/{lane}/conditioning.npy", conditioning),
                          "executed_full_layers": 36 if lane == "native" else None, "forecast_prefix_layers": 27 if lane == "forecast" else None}
                record["actual_sdpa_argument_observations"] = verify_sdpa_observations(sdpa_rows, case, lane, 27 if lane == "forecast" else 36, item["mask_record"]["regime"])
                if lane == "native":
                    record["last_hidden_state"] = old["_save_tensor"](root, "qwen/native/last-hidden-state.npy", states[-1])
                else:
                    record["position_ids"] = old["_save_tensor"](root, "qwen/forecast/position-ids.npy", result["position_ids"], logical_dtype="torch.int64")
                    record["profile"] = result["profile"]
                item[lane + "_E"] = conditioning.detach().cpu()
                save(root / (lane + "-qwen.json"), record)
                return record
            if phase == "flux":
                dg["validate_model"](flux)
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
                adapter = Flux2KleinAdapter(flux, scheduler, granularity="operation", device="cuda")
                conditioning = item[lane + "_E"].to("cuda")
                frontend = old["_frontend"](config, conditioning, "cuda", pipeline)
                counters["noise_draws"] += 1
                record = old["_frontend_record"](root, lane, frontend)
                if lane == "native":
                    for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas"):
                        require(torch.equal(frontend[key].cpu(), item["forecast_frontend"][key]), "independent native frontend differs: " + key)
                else:
                    item["forecast_frontend"] = {key: value.detach().cpu() for key, value in frontend.items() if hasattr(value, "detach")}
                state, rows = old["_trajectory"](root, flux, adapter, frontend, conditioning, lane=lane, graph=dg, store_name=lane + "-state-store", counters=counts)
                item[lane + "_state"] = {key: value.detach().cpu() if hasattr(value, "detach") else value for key, value in state.items()}
                counters["functional_transitions" if lane == "forecast" else "native_transitions"] += len(rows) - 1
                save(root / ("frontend-" + lane + ".json"), record)
                return {"rows": lane + "-state-rows.json", "transitions": len(rows) - 1}
            if lane == "forecast":
                record = old["_functional_decoder"](root, vae, item["forecast_state"], config, vg, pipeline, processor, counts)
                counters["functional_decode_calls"] += 1
                save(root / "decoder-forecast.json", record)
            else:
                record = {}
                for origin, state_name, basename in (("own_forecast", "forecast_state", "own"), ("native_reference", "native_state", "reference")):
                    record[basename] = old["_native_decoder"](root, vae, item[state_name], config, origin, pipeline, processor, counts, {})
                    counters["native_vae_calls"] += 1
                    save(root / ("decoder-native-" + basename + ".json"), record[basename])
            event("phase_completed", case=case, component=phase, lane=lane)
            return record

    tx, seal_record = None, None
    forecast_inventory = {}

    def seal(forecasts):
        nonlocal tx, seal_record, forecast_inventory
        require(counters["native_module_entries"] == 0 and not released["native"], "native entry before paired seal")
        common_frontend = {}
        for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas", "generator_before", "generator_after"):
            left = data["short"]["forecast_frontend"][key]
            right = data["long"]["forecast_frontend"][key]
            require(torch.equal(left, right), "same-seed paired source differs: " + key)
            common_frontend[key] = {"exact_equal": True, "descriptor": old["_tensor_signature"](left)}
        save(output / "common-frontend.json", common_frontend)
        responses = seal_responses(output, old)
        save(output / "sdpa-forecast.json", [row for row in sdpa_rows if row["lane"] == "forecast"])
        forecast_inventory = old["_inventory"](output, exclude=("timeline.jsonl",))
        tx = ForecastTransaction(output / "forecast-transaction", model_identity="flux2-klein-4b-raw-mask-pair-original",
                                 input_identity=hashlib.sha256(canonical({"cases": CASES, "seed": 277})).hexdigest(),
                                 source_identity=hashlib.sha256(canonical({key: config[key + "_sha256"] for key in old["_source_keys"]()})).hexdigest(),
                                 frontier={"kind": "raw-input", "cases": CASES, "seed": 277, "source_inventory": forecast_inventory},
                                 execution_scope="Both complete Qwen/FLUX/decoder forecasts precede either native learned target; all512rows consumed; selected kernels unprofiled.",
                                 budgets={"max_predictions": 1, "max_executions": 1, "max_native_calls": 8, "max_bytes": config["forecast_transaction_bytes"], "max_files": 30000, "max_wall_ms": config["max_wall_seconds"] * 1000})
        tx.predict("paired-raw-composition", lambda _context: {"predictions": {"cases": forecasts, "responses": {"path": "forecast-responses.json", "sha256": sha(output / "forecast-responses.json"), "recipe": responses["recipe"]}, "external_files": forecast_inventory, "native_targets_at_seal": 0}})
        seal_record = tx.seal()
        event("forecast_globally_sealed", seal_sha256=seal_record["seal_sha256"], native_module_entries=0, case_count=2)

    def execute(callback):
        tx.execute("both_native_qwen_flux_vae_chains", lambda _context: callback())
        tx.close()

    try:
        with old["_learned_forward_guard"](list(models.values()), released, counters):
            forecasts, natives = orchestrate(Backend(), seal, execute)
        after = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
        for name in models:
            old["_assert_signature_equal"](identities[name], after[name], name)
        tied_after = qwen.lm_head.weight is qwen.model.embed_tokens.weight
        require(tied_after == tied, "tied parameter alias changed")
        old["_verify_inventory"](output, forecast_inventory)
        expected_counts = {key: declared[key] for key in ("component_hydrations", "cuda_placements", "cpu_offloads", "noise_draws", "functional_transitions", "native_transitions", "functional_decode_calls", "native_vae_calls")}
        expected_counts["native_qwen_calls"] = declared["qwen_native_calls"]
        require({key: counters[key] for key in expected_counts} == expected_counts, "actual physical census differs")
        for case in CASES:
            for lane, expected in (("forecast", 27), ("native", 36)):
                verify_sdpa_observations(sdpa_rows, case, lane, expected, data[case]["mask_record"]["regime"])
        save(output / "sdpa-native.json", [row for row in sdpa_rows if row["lane"] == "native"])
        save(output / "native-contract.json", {"counts": counters, "case_counts": case_counts, "placement_records": placements, "component_identity_before": identities, "component_identity_after": after, "qwen_tied_alias_before": tied, "qwen_tied_alias_after": tied_after, "forecast_module_entries": 0, "selected_kernel": "unprofiled", "case_dispatch": {case: data[case]["mask_record"] for case in CASES}})
        save(output / "source.json", {"schema": "mdb-flux-raw-mask-pair-source-v1", "job_id": os.environ.get("MRUN_JOB_ID"), "request_sha256": config.get("request_sha256"), "worker_sha256": sha(__file__), "helper_source_sha256": config["helper_source_sha256"], "source_files": sources, "tokenizer_files": tokenizer_files, "runtime_contract_sha256": sha(output / "runtime-contract.json"), "forecast_seal": seal_record})
        summary = {"schema": "mdb-flux-raw-mask-pair-summary-v1", "job_id": os.environ.get("MRUN_JOB_ID"), "request_sha256": config.get("request_sha256"), "counts": counters, "contract": declared, "cases": {case: {"forecast": forecasts[case], "native": natives[case], "mask": data[case]["mask_record"]} for case in CASES}, "forecast_seal": seal_record, "transaction": verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal_record["seal_sha256"], max_bytes=config["forecast_transaction_bytes"]), "mechanics_status": "complete", "trend_status": "observed-uninterpreted", "terminal_status": "unassessed", "seconds": time.monotonic() - started}
        save(output / "summary.json", summary)
        return summary
    finally:
        if tx is not None:
            tx.close()
        torch.cuda.empty_cache()


def worker(config_path):
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print("MDB_RAW_MASK_PAIR_BOOTSTRAP", flush=True)
    config = read(config_path)
    if os.environ.get("SATURN_DEBUG_REQUEST"):
        request = read(os.environ["SATURN_DEBUG_REQUEST"])
        body = {key: value for key, value in request.items() if key != "request_sha256"}
        require(hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest() == request["request_sha256"], "request identity differs")
        require(request["config"] == config, "request/config differs")
        for name, expected in request["source_hashes"].items():
            path = Path.cwd() / name
            if not path.is_file():
                path = Path(config.get(name, name))
            require(path.is_file() and sha(path) == expected, "staged source differs: " + name)
        config = dict(config, request_sha256=request["request_sha256"], _request_record=request)
    if "output" not in config:
        config = dict(config, output=str(Path(config["output_scratch"]) / os.environ.get("MRUN_JOB_ID", "local") / config["output_name"]))
    summary = run(config, progress=lambda row: print("MDB_RAW_MASK_PAIR_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    finalizer = runpy.run_path(str(Path(__file__).with_name("flux_image_tail_microscope.py")))
    ready = finalizer["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)


if __name__ == "__main__":
    main()

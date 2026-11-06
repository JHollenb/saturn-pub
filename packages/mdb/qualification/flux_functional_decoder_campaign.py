"""Sealed four-endpoint decoder replication with complete macro checkpoints.

Endpoints were already observed. Original Torch primitives form predictions;
native comparisons follow the global seal. Full macro tensors use bounded CAS
chunks. Micro outputs retain full hashes and fixed witnesses, with at most one
post-observation hash-matched prefix reconstruction per case after a mismatch.
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
from pathlib import Path

CASES = ("sham", "HH", "HL", "new-3over8")
HELPERS = ("flux_functional_decoder.py", "flux_functional_vae.py", "flux_source_receiver.py", "flux_image_tail_microscope.py")
DECODER_SHA256 = "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f"
VAE_SHA256 = "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61"
EXECUTION_ID = "mdb-functional-decoder-campaign-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def endpoint_comparison(predicted, observed):
    from mdb.equation_comparison import compare_equations

    return compare_equations(predicted, observed, max_elements=sum(int(value.size) for value in predicted.values()))


def helpers(config):
    here = Path(__file__).parent
    for name in HELPERS:
        if sha(here / name) != config["helper_source_sha256"][name]:
            raise ValueError("qualified helper differs: " + name)
    if config["helper_source_sha256"][HELPERS[0]] != DECODER_SHA256 or config["helper_source_sha256"][HELPERS[1]] != VAE_SHA256:
        raise ValueError("frozen decoder/VAE helpers required")
    return [runpy.run_path(str(here / name)) for name in HELPERS]


def source_closure(root, inputs):
    result = set()
    for row in inputs.values():
        store = Path(row["store"])
        if store.is_absolute() or ".." in store.parts:
            raise ValueError("unsafe original source store")
        name = store / "cuts" / (row["cut"] + ".json")
        cut = json.loads((root / name).read_text())
        result.add(name.as_posix())
        result.update((store / "pages" / (v + ".json")).as_posix() for v in cut["slots"].values())
        result.update((store / "blobs" / (v + ".safetensors")).as_posix() for v in cut["physical_blobs"])
    return result


def geometry(config):
    """Static shape/count bounds from the frozen untiled decoder graph."""
    spec = json.loads(Path(config["vae_config"]).read_text())
    channels = list(reversed(spec.get("decoder_block_out_channels") or spec["block_out_channels"]))
    scale = 2 ** (len(channels) - 1)
    h, w, z = config["height"] // scale, config["width"] // scale, spec["latent_channels"]
    shapes = {"latent_unpacked": [1, 4 * z, h // 2, w // 2], "latent_denormalized": [1, 4 * z, h // 2, w // 2],
        "latent_unpatched": [1, z, h, w], "post_quant_conv": [1, z, h, w], "decoder_input": [1, z, h, w], "decoder_mid": [1, channels[0], h, w]}
    for i, c in enumerate(channels):
        if i < len(channels) - 1:
            h, w = h * 2, w * 2
        shapes["decoder_up_" + str(i)] = [1, c, h, w]
    shapes.update(decoded=[1, 3, h, w], pixels=[1, h, w, 3])
    def elements(shape):
        count = 1
        for item in shape:
            count *= item
        return count
    sizes = {name: elements(shape) * (1 if name == "pixels" else 2) for name, shape in shapes.items()}
    # Largest nearest interpolation has the preceding block's channels at the
    # next spatial size. This bounds a first primitive discrepancy tensor.
    largest = max(max(sizes.values()), max((2 * channels[i] * shapes["decoder_up_" + str(i + 1)][2] * shapes["decoder_up_" + str(i + 1)][3] for i in range(len(channels) - 1)), default=0))
    return dict(macro_shapes=shapes, macro_bytes=sizes, macro_bytes_per_case_lane=sum(sizes.values()), eight_lane_macro_bytes=8 * sum(sizes.values()),
        first_micro_tensor_bytes_bound=largest, conditional_full_diagnostic_bytes_bound=8 * largest,
        raw_tensor_bytes_bound=8 * (sum(sizes.values()) + largest), chunk_bytes=config.get("chunk_bytes", 32 << 20),
        work_bound=dict(functional_complete_decodes=4, native_complete_decodes=4, conditional_functional_prefixes=4, native_extra_decodes=0))


def validate_config(config):
    _, _, _, m = helpers(config)
    require = m["require"]
    require(set(config["inputs"]) == set(CASES), "four unchanged observed endpoints required")
    for key in ("vae_config", "parameter_manifest", "retrieval_script"):
        require(sha(config[key]) == config[key + "_sha256"], "original metadata differs: " + key)
    require(sha(config["retrieval_script"]) == m["PARAMETER_RETRIEVER_SHA256"], "qualified standalone retriever required")
    require(type(config["source_reduction_policy"]) is bool and 1 <= config.get("cpu_threads", 4) <= 4, "explicit numeric policy required")
    require(256 <= config.get("chunk_bytes", 32 << 20) <= 32 << 20 and 1 <= config.get("witness_values", 16) <= 64, "bounded chunk/witness geometry required")
    require(0 < config.get("max_wall_seconds", 600) <= 1200, "finite decoder wall required")
    spec = json.loads(Path(config["vae_config"]).read_text())
    tiny = config.get("profile") == "tiny-cpu"
    require(config.get("device") == ("cpu" if tiny else "cuda"), "declared execution device differs")
    if tiny:
        require(len(spec["block_out_channels"]) <= 3 and max(spec["block_out_channels"]) <= 32, "tiny-only profile required")
    else:
        require(config["vae_config_sha256"] == "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948" and config["height"] == config["width"] == 512, "original decoder geometry required")
        require(config["source_provenance"]["source_job"] == "job-f7d249b6d3bb" and config["source_provenance"]["accepted_auditor_sha256"] == "5027875c03f9ceb3157b4b9d9632366ff5c9b1c4beb429b8b73d79aad0db6b08", "accepted current saved-only source audit required")
    root = Path(config["input_root"]).resolve()
    require(set(config["input_files"]) == source_closure(root, config["inputs"]), "source whitelist must contain only four complete declared cuts")
    for name, record in config["input_files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "original source custody differs")
    return dict(cases=list(CASES), functional_decodes=4, native_decodes=4, conditional_diagnostic_prefixes_bound=4, full_primitive_retention=False, chunk_bytes=config.get("chunk_bytes", 32 << 20), geometry=geometry(config))


def run(config, *, progress=None):
    declared = validate_config(config)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKLFlux2
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from PIL import Image
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.equation_comparison import EquationCatalog
    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    graph, old, recipes, m = helpers(config)
    require, save, identity = m["require"], m["save"], m["identity"]
    output = Path(config["output"]).resolve()
    require(not any(output.is_relative_to(Path(config[key]).resolve()) for key in ("input_root", "parameter_root")), "output overlaps immutable original source")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    started, handles, transactions = time.monotonic(), [], []
    active = dict(lane="setup", case=None, attention=None, upsample=None)
    totals = dict(model_loads=0, decoder_loads=0, transformer_loads=0, functional_decode_calls=0, native_decode_calls=0, diagnostic_prefix_calls=0,
        forecast_module_entries=0, diagnostic_module_entries=0, native_module_entries=0, guard_refusals=0, encodes=0, noise_draws=0, denoiser_transitions=0)
    counts = {lane: {case: {} for case in CASES} for lane in ("forecast", "native", "diagnostic")}
    records = {lane: {case: {} for case in CASES} for lane in ("forecast", "native")}
    micros = {lane: {case: [] for case in CASES} for lane in ("forecast", "native")}
    first_micro, first_macro, diagnostics = dict.fromkeys(CASES), dict.fromkeys(CASES), {}
    parents, source_execution, input_records = {}, {}, {}
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.get_deterministic_debug_mode(), torch.get_num_threads(), torch.get_float32_matmul_precision())
    original_sdpa, original_interpolate = F.scaled_dot_product_attention, F.interpolate
    forecast_tx, sealed_files, sealed_signatures = None, {}, {}
    stores = {lane: LocalStore(output / (lane + "-store")) for lane in ("forecast", "native", "diagnostic")}

    def event(kind, **detail):
        row = dict(kind=kind, seconds=time.monotonic() - started, **detail)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress:
            progress(row)
        require(row["seconds"] <= config.get("max_wall_seconds", 600), "finite decoder wall exceeded")

    def inventory(prefixes):
        paths = [p for prefix in prefixes for p in ([output / prefix] if (output / prefix).is_file() else sorted((output / prefix).rglob("*"))) if p.is_file()]
        return {p.relative_to(output).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}

    def verify_files(files, signatures=None):
        result = {}
        for name, row in files.items():
            p = output / name
            stat = p.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            require(p.resolve().is_relative_to(output) and not p.is_symlink(), "external evidence path differs")
            if signatures is None:
                require(stat.st_size == row["bytes"] and sha(p) == row["sha256"], "sealed external evidence differs")
            else:
                require(signature == signatures[name], "sealed external evidence signature changed")
            result[name] = signature
        return result

    def guard_target():
        if forecast_tx is None or forecast_tx.phase != "execute":
            totals["guard_refusals"] += 1
            raise ValueError("native decoder outside globally sealed execution")
        forecast_tx.verify()
        verify_files(sealed_files, sealed_signatures)

    def descriptor(value):
        require(bool(torch.isfinite(value).all()), "nonfinite decoder tensor")
        return dict(tensor_descriptor(value), stride=list(value.stride()))

    def witness(value):
        flat = value.detach().reshape(-1)
        indices = np.unique(np.linspace(0, flat.numel() - 1, min(config.get("witness_values", 16), flat.numel())).astype(np.int64))
        return dict(descriptor=descriptor(value), indices=indices.tolist(), values=flat[torch.as_tensor(indices, device=flat.device)].float().cpu().tolist(), full_values_retained=False)

    def checkpoint(name, value, *, lane=None, parent=None, ordinal=None):
        lane, case = lane or active["lane"], active["case"]
        d = descriptor(value)
        flat = value.detach().reshape(-1)
        elements = max(1, config.get("chunk_bytes", 32 << 20) // value.element_size())
        chunks = tuple(flat[i:i + elements].to("cpu").contiguous() for i in range(0, flat.numel(), elements))
        parent = parents.get((lane, case), input_records[case]["cut"]) if parent is None else parent
        number = len(records[lane][case]) if ordinal is None and lane in records else (ordinal or 0)
        chunk_keys = ["chunk_" + str(i).zfill(5) for i in range(len(chunks))]
        # Each tensor is a separate top-level slot: LocalStore otherwise groups
        # tuple members into one safetensors blob, exceeding transport bounds.
        payload = dict(schema="mdb-decoder-macro-chunks-v1", chunk_keys=chunk_keys, descriptor=d, name=name, original_source_cut=input_records[case]["cut"], **dict(zip(chunk_keys, chunks, strict=True)))
        execution = {**source_execution[case], "component_decoder": dict(id=EXECUTION_ID, lane=lane, case=case, source_cut=input_records[case]["cut"], not_session_act=True, complete_flux_advance=False)}
        point = dict(family="flux2-klein-decoder", logical_step=4, phase="decoder", operator=name, index=number, local_clock=number, edge="after")
        cut = StateCut(config["expected_source_model_identity"], execution, "decoder-replication:" + name, parent, payload, point)
        stores[lane].save(cut)
        row = dict(cut=cut.fingerprint, parent=parent, name=name, descriptor=d, chunk_keys=chunk_keys, chunk_elements=elements, chunk_count=len(chunks), elements=flat.numel(), original_source_cut=input_records[case]["cut"])
        if lane in records:
            records[lane][case][name] = row
            parents[lane, case] = cut.fingerprint
        return row

    def macro(name, value):
        checkpoint(name, value)

    def micro(name, kind, inputs, value):
        lane, case = active["lane"], active["case"]
        ordinal = active["ordinals"].get(name, 0)
        active["ordinals"][name] = ordinal + 1
        row = dict(ordinal=len(micros[lane][case]), id=name + "." + str(ordinal), name=name, kind=kind,
            inputs=[witness(v) for v in inputs if isinstance(v, torch.Tensor)], output=witness(value))
        micros[lane][case].append(row)
        if lane == "native":
            expected = micros["forecast"][case][row["ordinal"]]
            require((row["id"], row["kind"]) == (expected["id"], expected["kind"]), "native primitive ordering differs")
            if first_micro[case] is None and row["output"]["descriptor"] != expected["output"]["descriptor"]:
                native_record = checkpoint("first_micro_native", value, lane="diagnostic", parent=input_records[case]["cut"], ordinal=row["ordinal"])
                first_micro[case] = dict(id=row["id"], ordinal=row["ordinal"], name=name, kind=kind, sealed_forecast=expected["output"], actual_native=row["output"], native_full=native_record,
                    coverage="Native full output retained immediately; forecast original retention is full hash plus fixed witnesses only.")
        if name == "post_quant_conv":
            macro("post_quant_conv", value)
            if lane == "forecast":
                macro("decoder_input", value)
        elif name == "decoder.mid_block":
            macro("decoder_mid", value)
        elif name.startswith("decoder.up_blocks.") and kind == "up_block":
            macro("decoder_up_" + name.rsplit(".", 1)[1], value)
        elif name == "decoder":
            macro("decoded", value)

    def upstream(state, *, native=False):
        if native:
            guard_target()
            value = Flux2KleinPipeline._unpack_latents_with_ids(state["latent"], state["img_ids"], config["height"] // (scale * 2), config["width"] // (scale * 2))
        else:
            value = old["unpack_latents"](state["latent"], state["img_ids"], config["height"] // (scale * 2), config["width"] // (scale * 2))
        macro("latent_unpacked", value)
        mean = vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(value)
        std = (vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(value)
        value = value * std + mean
        macro("latent_denormalized", value)
        value = (Flux2KleinPipeline._unpatchify_latents(value) if native else old["unpatchify_latents"](value)).to(device)
        macro("latent_unpatched", value)
        return value

    def unchunk(row, lane):
        cut = stores[lane].load(row["cut"], device="cpu")
        cut.verify()
        return tuple(cut.payload[key] for key in row["chunk_keys"])

    def metric_chunks(arrays):
        # Two arrays compare directly; four compute case-minus-sham responses.
        sums = dict(n=0, error2=0., native2=0., predicted2=0., signed=0., native_signed=0., predicted_signed=0., maxabs=0., changed=0)
        for chunks in zip(*arrays, strict=True):
            values = [v.float().numpy().astype(np.float64) for v in chunks]
            predicted, actual = (values if len(values) == 2 else (values[0] - values[1], values[2] - values[3]))
            delta = predicted - actual
            sums["n"] += delta.size
            sums["error2"] += float(np.sum(delta * delta))
            sums["native2"] += float(np.sum(actual * actual))
            sums["predicted2"] += float(np.sum(predicted * predicted))
            sums["signed"] += float(delta.sum())
            sums["native_signed"] += float(actual.sum())
            sums["predicted_signed"] += float(predicted.sum())
            sums["maxabs"] = max(sums["maxabs"], float(abs(delta).max()))
            sums["changed"] += int(np.count_nonzero(delta))
        n = sums["n"]
        rms, native_rms = (sums["error2"] / n) ** .5, (sums["native2"] / n) ** .5
        return dict(rms=rms, relative_rms=None if native_rms == 0 else rms / native_rms, native_rms=native_rms,
            predicted_rms=(sums["predicted2"] / n) ** .5, signed_mean=sums["signed"] / n, native_signed_mean=sums["native_signed"] / n,
            predicted_signed_mean=sums["predicted_signed"] / n, maxabs=sums["maxabs"], changed_values=sums["changed"], sample_count=n,
            orientation="predicted-minus-native; responses case-minus-same-lane-sham", quantiles="Full values retained; this worker streams moments/max, not full-array quantiles.")

    try:
        torch.set_num_threads(config.get("cpu_threads", 4))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
        device = config["device"]
        spec = json.loads(Path(config["vae_config"]).read_text())
        manifest = json.loads(Path(config["parameter_manifest"]).read_text())
        bank = recipes["RecipeBank"](manifest, config["parameter_root"], max_cache_bytes=1)
        with torch.device("meta"):
            vae = AutoencoderKLFlux2.from_config(spec)
        state_dict = vae.state_dict()
        required = {name for name in state_dict if name.startswith(("decoder.", "post_quant_conv.")) or name in ("bn.running_mean", "bn.running_var")}
        require({row["source"]["tensor"] for row in manifest["artifacts"]} == required and len(manifest["artifacts"]) == len(required), "exact decoder/BN recipe set required")
        require(config.get("profile") == "tiny-cpu" or len(required) == 142, "original142 decoder entries required")
        weights, expected_weights = {}, {}
        for row in manifest["artifacts"]:
            source = row["source"]
            name = source["tensor"]
            require(source["dtype"] == "BF16" and list(state_dict[name].shape) == source["shape"], "original tensor geometry/dtype differs")
            value = torch.from_numpy(bank.load(dict(path=row["path"], sha256=row["sha256"], parameter_tensor=name, stored_shape=source["shape"], descriptor=dict(sha256=source["payload_sha256"])))).to(torch.bfloat16)
            require(descriptor(value)["sha256"] == source["payload_sha256"], "original tensor restoration differs")
            weights[name] = value.to("cpu" if name.startswith("bn.") else device)
            expected_weights[name] = source["payload_sha256"]
        loaded = vae.load_state_dict(weights, strict=False, assign=True)
        require(not loaded.unexpected_keys and set(loaded.missing_keys) == set(state_dict) - required - {"bn.num_batches_tracked"}, "unused VAE boundary differs")
        vae.bn.num_batches_tracked = torch.tensor(0, dtype=torch.int64)
        vae.eval().requires_grad_(False)
        totals["model_loads"] = totals["decoder_loads"] = 1
        profile = "tiny" if config.get("profile") == "tiny-cpu" else "original"
        decoder_contract = graph["validate_model"](vae, profile=profile)
        scale = 2 ** (len(spec["block_out_channels"]) - 1)
        image_processor = Flux2ImageProcessor(vae_scale_factor=scale * 2)
        upstream_sources = {"pipeline_flux2_klein.py": sha(inspect.getfile(Flux2KleinPipeline)), "flux2_image_processor.py": sha(inspect.getfile(Flux2ImageProcessor))}
        require(upstream_sources == config["upstream_source_sha256"], "native upstream source differs")
        signature = {name: (id(v), v.data_ptr(), v._version) for name, v in vae.state_dict(keep_vars=True).items() if name in required}
        input_store = LocalStore(output / "input-store")
        for case in CASES:
            row = config["inputs"][case]
            cut = LocalStore(Path(config["input_root"]) / row["store"]).load(row["cut"], device="cpu")
            cut.verify()
            p, component = cut.payload, cut.execution["component_pass"]
            require(cut.model_identity == config["expected_source_model_identity"] and (p["step"], p["phase"], p["cursor"]) == (4, "project", 0)
                and all(p[k] is None for k in ("image", "text", "context", "noise_prediction")) and len(p["timesteps"]) == 4,
                "complete current predicted terminal S4 cut required")
            require((component["id"], component["lane"], component["case"]) == ("mdb-flux-functional-remaining-v1", "forecast", case), "current predicted source lineage required")
            require(p["latent"].dtype == p["conditioning"].dtype == torch.bfloat16 and p["img_ids"].dtype == p["txt_ids"].dtype == torch.int64, "source original tensor dtype differs")
            input_store.save(cut)
            input_records[case] = dict(cut=cut.fingerprint, parent=cut.parent, point=dict(cut.execution_point), payload=describe(p))
            source_execution[case] = dict(cut.execution)
        del weights, state_dict, loaded, value, p, cut
        input_files = inventory(["input-store"])
        input_tx = ForecastTransaction(output / "input-transaction", model_identity=config["expected_source_model_identity"], input_identity=identity(input_records), source_identity=config["parameter_manifest_sha256"], array_storage="files-v1", frontier=dict(kind="architectural", observed="Four already-observed final predicted latents; decoder replication only"), budgets=dict(max_bytes=64 << 20))
        transactions.append(input_tx)
        input_tx.predict("inputs", lambda context: dict(predictions=dict(cases=input_records, external_files=input_files)))
        input_seal = input_tx.seal()
        input_tx.close()
        event("inputs_sealed", seal=input_seal)
        names = {id(module): name for name, module in vae.named_modules()}
        allowed = {id(module) for root in (vae.post_quant_conv, vae.decoder) for module in root.modules()}

        def global_entry(module, args):
            lane = active["lane"]
            if lane in ("forecast", "diagnostic"):
                totals[lane + "_module_entries"] += 1
                totals["guard_refusals"] += 1
                raise ValueError("functional decoder entered native module.forward")
            require(lane == "native" and id(module) in allowed, "undeclared native decoder module")
            totals["native_module_entries"] += 1
            if module is vae.post_quant_conv or module is vae.decoder:
                guard_target()
            if module is vae.decoder:
                macro("decoder_input", args[0])
                event("native_decoder_entry", case=active["case"], actual_after_post_quant_descriptor=descriptor(args[0]))
            if type(module).__name__ == "Attention":
                active["attention"] = names[id(module)]
            if type(module).__name__ == "Upsample2D":
                active["upsample"] = names[id(module)]
        handles.append(torch.nn.modules.module.register_module_forward_pre_hook(global_entry))

        def native_output(module, args, value):
            if active["lane"] != "native":
                return
            name, typename = names[id(module)], type(module).__name__
            kind = {"Conv2d": "conv2d", "Linear": "linear", "GroupNorm": "group_norm", "SiLU": "silu", "ResnetBlock2D": "resnet", "Attention": "attention", "Upsample2D": "upsample", "UNetMidBlock2D": "mid_block", "UpDecoderBlock2D": "up_block", "Decoder": "decoder", "Dropout": "dropout_identity"}.get(typename)
            if kind:
                census = counts["native"][active["case"]]
                census[kind] = census.get(kind, 0) + 1
                if kind != "dropout_identity":
                    micro(name, kind, args, value)
        for module in vae.modules():
            if id(module) in allowed:
                handles.append(module.register_forward_hook(native_output))
        for case in CASES:
            active.update(lane="forecast", case=case, ordinals={})
            input_tx.verify()
            verify_files(input_files)
            p = input_store.load(input_records[case]["cut"], device="cpu").payload
            with torch.inference_mode():
                argument = upstream(p)
                totals["functional_decode_calls"] += 1
                decoded = graph["functional_decode"](vae, argument, observe=micro, counts=counts["forecast"][case], profile=profile)
                pixels = old["postprocess_pixels"](decoded)
                macro("pixels", torch.from_numpy(pixels))
            directory = output / "forecast-images"
            directory.mkdir(exist_ok=True)
            Image.fromarray(pixels[0]).save(directory / (case + ".png"))
            save(output / "forecast-micro" / (case + ".json"), micros["forecast"][case])
            event("forecast_case_completed", case=case, primitive_outputs=len(micros["forecast"][case]), module_entries=totals["forecast_module_entries"])
            del p, argument, decoded, pixels
        save(output / "forecast-macros.json", records["forecast"])
        sealed_files = inventory(["forecast-store", "forecast-images", "forecast-micro", "forecast-macros.json"])
        forecast_tx = ForecastTransaction(output / "forecast-transaction", model_identity=config["expected_source_model_identity"], input_identity=input_seal["seal_sha256"], source_identity=config["parameter_manifest_sha256"], array_storage="files-v1", frontier=dict(kind="architectural", observed="Already-observed endpoints, complete internal macro retention; no native decoded arrays read"), budgets=dict(max_bytes=256 << 20, max_files=16384))
        transactions.append(forecast_tx)
        forecast_tx.predict("cohort", lambda context: dict(predictions=dict(macros=records["forecast"], micro_files={case: "forecast-micro/" + case + ".json" for case in CASES}, external_files=sealed_files, input_seal=input_seal, native_decode_calls_at_seal=0)))
        forecast_seal = forecast_tx.seal()
        sealed_signatures = verify_files(sealed_files)
        event("cohort_globally_sealed", seal=forecast_seal, native_decode_calls=totals["native_decode_calls"])

        def sdpa(*args, **kwargs):
            value = original_sdpa(*args, **kwargs)
            if active["lane"] == "native":
                census = counts["native"][active["case"]]
                census["sdpa"] = census.get("sdpa", 0) + 1
                micro(active["attention"] + ".sdpa", "sdpa", args, value)
            return value
        def interpolate(*args, **kwargs):
            value = original_interpolate(*args, **kwargs)
            if active["lane"] == "native":
                census = counts["native"][active["case"]]
                census["interpolate_nearest"] = census.get("interpolate_nearest", 0) + 1
                micro(active["upsample"] + ".interpolate", "interpolate_nearest", args, value)
            return value
        F.scaled_dot_product_attention, F.interpolate = sdpa, interpolate

        def execute(context):
            for case in CASES:
                guard_target()
                active.update(lane="native", case=case, ordinals={})
                p = input_store.load(input_records[case]["cut"], device="cpu").payload
                with torch.inference_mode():
                    argument = upstream(p, native=True)
                    event("native_decode_entry", case=case, actual_before_post_quant_descriptor=descriptor(argument))
                    totals["native_decode_calls"] += 1
                    decoded = vae.decode(argument, return_dict=False)[0]
                    images = image_processor.postprocess(decoded, output_type="pil")
                    pixels = np.stack([np.asarray(image) for image in images])
                    macro("pixels", torch.from_numpy(pixels))
                directory = output / "native-images"
                directory.mkdir(exist_ok=True)
                images[0].save(directory / (case + ".png"))
                save(output / "native-micro" / (case + ".json"), micros["native"][case])
                require(counts["native"][case] == counts["forecast"][case], "actual primitive census differs")
                require(len(micros["native"][case]) == len(micros["forecast"][case]), "full primitive checkpoint count differs")
                event("native_case_completed", case=case, native_decode_calls=totals["native_decode_calls"])
                del p, argument, decoded, pixels, images
            # Diagnostic reconstruction occurs only after every native target.
            class PrefixCaptured(Exception):
                pass
            for case in CASES:
                first = first_micro[case]
                if first is None:
                    diagnostics[case] = dict(status="not-needed", prefix_calls=0)
                    continue
                active.update(lane="diagnostic", case=case)
                totals["diagnostic_prefix_calls"] += 1
                result = dict(status="unavailable", prefix_calls=1, original_sealed=first["sealed_forecast"], native_full=first["native_full"],
                    retention_stage="post-observation prefix reconstruction, not pretarget full micro retention", unchanged_input=input_records[case]["cut"], helper_sha256=DECODER_SHA256)
                index = 0
                def capture(name, kind, inputs, value):
                    nonlocal index
                    if index == first["ordinal"]:
                        actual = descriptor(value)
                        result["replay_descriptor"] = actual
                        sealed = first["sealed_forecast"]["descriptor"]
                        if all(actual[k] == sealed[k] for k in ("sha256", "shape", "dtype")):
                            result["forecast_full"] = checkpoint("first_micro_reconstructed", value, lane="diagnostic", parent=input_records[case]["cut"], ordinal=index)
                            result["status"] = "hash-matched-reconstruction"
                            result["original_layout_exact"] = actual["stride"] == sealed["stride"]
                        else:
                            result["status"] = "replay-differs-from-original-sealed-hash"
                        raise PrefixCaptured()
                    index += 1
                event("diagnostic_prefix_started", case=case, id=first["id"], native_decode_calls=totals["native_decode_calls"])
                p = value = argument = None
                try:
                    p = input_store.load(input_records[case]["cut"], device="cpu").payload
                    with torch.inference_mode():
                        value = old["unpack_latents"](p["latent"], p["img_ids"], config["height"] // (scale * 2), config["width"] // (scale * 2))
                        mean = vae.bn.running_mean.cpu().view(1, -1, 1, 1).to(value)
                        std = (vae.bn.running_var.cpu().view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(value)
                        argument = old["unpatchify_latents"](value * std + mean).to(device)
                        graph["functional_decode"](vae, argument, observe=capture, counts=counts["diagnostic"][case], profile=profile)
                except PrefixCaptured:
                    pass
                except Exception as exc:
                    result.update(status="reconstruction-unavailable", error=type(exc).__name__ + ": " + str(exc))
                if "forecast_full" in result:
                    result["full_pair_metric"] = metric_chunks([unchunk(result["forecast_full"], "diagnostic"), unchunk(first["native_full"], "diagnostic")])
                del p, value, argument
                diagnostics[case] = result
                event("diagnostic_prefix_completed", case=case, status=result["status"], counts=counts["diagnostic"][case], native_decode_calls=totals["native_decode_calls"])
            return dict(native_macros=records["native"], first_micro_difference=first_micro, diagnostics=diagnostics, counts=counts, totals=totals)
        forecast_tx.execute("native_decoders_and_optional_diagnostics", execute)
        forecast_tx.close()
        verify_files(sealed_files)
        verify_files(input_files)
        save(output / "native-macros.json", records["native"])
        comparisons, micro_comparisons, responses = {}, {}, {}
        for case in CASES:
            require(set(records["forecast"][case]) == set(records["native"][case]), "full macro coverage differs")
            comparisons[case] = {}
            for name, predicted in records["forecast"][case].items():
                actual = records["native"][case][name]
                metric = metric_chunks([unchunk(predicted, "forecast"), unchunk(actual, "native")])
                metric.update(bitwise_exact=predicted["descriptor"]["sha256"] == actual["descriptor"]["sha256"], layout_exact=predicted["descriptor"]["stride"] == actual["descriptor"]["stride"], dtype_exact=predicted["descriptor"]["dtype"] == actual["descriptor"]["dtype"])
                comparisons[case][name] = metric
                if first_macro[case] is None and not metric["bitwise_exact"]:
                    first_macro[case] = name
            micro_comparisons[case] = [dict(id=p["id"], ordinal=p["ordinal"], full_descriptor_exact=p["output"]["descriptor"] == n["output"]["descriptor"], witness_exact=p["output"]["indices"] == n["output"]["indices"] and p["output"]["values"] == n["output"]["values"], input_descriptors_exact=[v["descriptor"] for v in p["inputs"]] == [v["descriptor"] for v in n["inputs"]]) for p, n in zip(micros["forecast"][case], micros["native"][case], strict=True)]
        for case in CASES[1:]:
            responses[case] = {name: metric_chunks([unchunk(records["forecast"][case][name], "forecast"), unchunk(records["forecast"]["sham"][name], "forecast"), unchunk(records["native"][case][name], "native"), unchunk(records["native"]["sham"][name], "native")]) for name in records["forecast"][case]}
        save(output / "comparisons.json", dict(macros=comparisons, micro=micro_comparisons, first_micro_difference=first_micro, first_macro_difference=first_macro, diagnostics=diagnostics))
        save(output / "signed-responses.json", responses)
        def publish_catalog():
            catalog = EquationCatalog(output / "catalog/equations")
            for case in CASES:
                values = {lane: {name: torch.cat(unchunk(records[lane][case][name], lane)).reshape(records[lane][case][name]["descriptor"]["shape"]).float().numpy() for name in ("decoded", "pixels")} for lane in ("forecast", "native")}
                context = dict(case=case, source_cut=input_records[case]["cut"], model_identity=config["expected_source_model_identity"], helper_sha256=DECODER_SHA256, parameter_manifest_sha256=config["parameter_manifest_sha256"], clock="final-step4-predicted-latent", forecast_seal=forecast_seal["seal_sha256"])
                catalog.record(law_id="law.flux.functional-decoder-replication.v1", route_id="route.flux.original-decoder.v1", alias="cmp.flux.decoder." + case,
                    binding=dict(model_identity=context["model_identity"], execution=dict(id=EXECUTION_ID, dtype="torch.bfloat16", device=device, replication=True), input_identity=context["source_cut"], source_identity=context["helper_sha256"], frontier=context["clock"], horizon=dict(decoder_passes=1)),
                    report=endpoint_comparison(values["forecast"], values["native"]), conditions=[dict(field=k, op="eq", value=v) for k, v in context.items()], tested_context=context,
                    readers=["same-latent-native-VAE"], evidence=dict(replication=True, independent_kernel_arithmetic=False, full_macro_retention=True, full_original_micro_retention=False))
        for name, value in vae.state_dict(keep_vars=True).items():
            if name in required:
                require((id(value), value.data_ptr(), value._version) == signature[name] and descriptor(value)["sha256"] == expected_weights[name], "original decoder tensor changed")
        for key in ("vae_config", "parameter_manifest", "retrieval_script"):
            require(sha(config[key]) == config[key + "_sha256"], "source metadata changed")
            shutil.copyfile(config[key], output / Path(config[key]).name)
        require(totals["functional_decode_calls"] == totals["native_decode_calls"] == 4 and totals["diagnostic_prefix_calls"] <= 4, "bounded physical decoder census differs")
        save(output / "native-contract.json", dict(decoder=decoder_contract, upstream_source_sha256=upstream_sources, parameter_names=sorted(required), original_parameter_bytes=bank.bytes,
            unused_counter=dict(name="bn.num_batches_tracked", value=0, used_by_decode=False, pretrained_claim=False), device=device, dtype="torch.bfloat16", inference_mode=True, autocast=False, tf32=False, deterministic=True,
            frontend=dict(input_state_device="cpu", unpack_device="cpu", bn_statistics_device="cpu", bn_sqrt_device="cpu", bn_multiply_device="cpu", bn_add_device="cpu", unpatch_device="cpu", decoder_argument_device=device, transfer_boundary="After CPU unpatch, before post_quant_conv; applies to forecast, native and diagnostic lanes."),
            source_reduction_policy=config["source_reduction_policy"], macro_encoding="Original full descriptor/stride plus ordered flat CAS tensor chunks, each<=32MiB; full numeric reconstruction; component-derived snapshots, no complete FLUX.advance or Session/Act claim.",
            input_semantics="latent_unpatched is BEFOREpost_quant_conv VAE.decode argument; post_quant_conv/decoder_input are AFTERpost_quant_conv actual decoder entry.",
            replication=True, micro_retention="Full hashes+fixed witnesses. First native differing tensor retained immediately; optional post-observation unchanged prefix only if original presealed output hash restores; never pretarget full micro retention.", count_contract=declared))
        save(output / "source.json", dict(schema="mdb-functional-decoder-source-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"), worker_sha256=sha(__file__), helper_source_sha256=config["helper_source_sha256"], source_provenance=config.get("source_provenance"), input_records=input_records, input_seal=input_seal, forecast_seal=forecast_seal, original_decoder_tensor_sha256=expected_weights))
        summary = dict(schema="mdb-functional-decoder-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"), counts=counts, totals=totals, input_seal=input_seal, forecast_seal=forecast_seal,
            transaction=forecast_tx.report(), verification=verify_transaction(output / "forecast-transaction", expected_seal_sha256=forecast_seal["seal_sha256"]), cases=list(CASES),
            complete_macro_cuts={lane: sum(len(v) for v in panel.values()) for lane, panel in records.items()}, first_micro_difference=first_micro, first_macro_difference=first_macro, diagnostics=diagnostics,
            parameter_payload_reads=bank.payload_reads, parameter_source_bytes=bank.bytes, parameter_cache_peak_bytes=bank.peak_retained, seconds=time.monotonic() - started, final_resident_parameter_check="verified-before-optional-catalog", optional_catalog_report="catalog-report.json",
            geometry=declared["geometry"], actual_store_bytes={lane: sum(p.stat().st_size for p in stores[lane].root.rglob("*") if p.is_file()) for lane in stores},
            mechanics_status="complete", instrument_status="full macro states; full primitive hashes/fixed witnesses; conditional reconstruction explicit", trend_status="observed-uninterpreted", terminal_status="unassessed", act_receipts=0, session_continue_receipts=0)
        save(output / "summary.json", summary)
        try:
            publish_catalog()
            optional = dict(status="complete", records=4)
        except Exception as exc:
            optional = dict(status="reporting-error", error=type(exc).__name__ + ": " + str(exc), primary_evidence_preserved=True)
        save(output / "catalog-report.json", optional)
        summary["optional_catalog"] = optional
        return summary
    finally:
        try:
            for transaction in transactions:
                transaction.close()
        finally:
            F.scaled_dot_product_attention, F.interpolate = original_sdpa, original_interpolate
            for handle in handles:
                handle.remove()
            torch.set_float32_matmul_precision(original_flags[7])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_flags[:5]
            torch.set_deterministic_debug_mode(original_flags[5])
            torch.set_num_threads(original_flags[6])


def worker(config_path):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    *_, m = helpers(config)
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        m["require"](request["config"] == config, "request/config differs")
        for name, digest in request["source_hashes"].items():
            m["require"](sha(Path.cwd() / name) == digest, "staged source differs")
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "functional-decoder-result"))
    summary = run(config, progress=lambda row: print("MDB_DECODER_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    ready = m["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)

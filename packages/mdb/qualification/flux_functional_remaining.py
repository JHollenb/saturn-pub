"""Remaining two own-state FLUX passes with source-bound real VAE consumers.

The unchanged functional helper supplies the forecast graph. Public native
adapter transitions supply verification; real VAE decodes consume both final lanes.
All complete future cuts seal, then four real predicted-latent decodes seal before any future target.
"""
from __future__ import annotations

import argparse
import faulthandler
import inspect
import json
import os
import runpy
import shutil
import time
from pathlib import Path

HELPERS = ("flux_functional_denoiser.py", "flux_image_tail_microscope.py", "flux_source_receiver.py")
CASES = ("sham", "HH", "HL", "new-3over8")
EXECUTION_ID = "mdb-flux-functional-remaining-v1"
FUNCTIONAL_SOURCE = "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923"
PREDICTIONS = ["Two complete remaining original-policy passes match the separate native-reference lane at covered boundaries.",
               "Signed finite responses from sham match; preserve all misses and first covered discrepancy.",
               "Four real VAE consumers of predicted terminal latents conditionally forecast reference-lane RGB; decoder computation is observed, not independently predicted."]


def helpers(config):
    here = Path(__file__).parent
    modules = [runpy.run_path(str(here / name)) for name in HELPERS]
    for name in HELPERS:
        modules[1]["require"](modules[1]["sha"](here / name) == config["helper_source_sha256"][name], "qualified helper differs: " + name)
    modules[1]["require"](config["helper_source_sha256"][HELPERS[0]] == FUNCTIONAL_SOURCE, "original functional helper required")
    return modules


def geometry(config):
    spec = json.loads(Path(config["transformer_config"]).read_text())
    return spec["num_layers"], spec["num_single_layers"]


def contract(config):
    joints, singles = geometry(config)
    one = dict(transitions=joints + singles + 3, linear=9 + 12 * joints + 2 * singles,
               layernorm=4 * joints + singles + 1, rmsnorm=4 * joints + 2 * singles,
               swiglu=2 * joints + singles, sdpa=joints + singles, rotary=2 * (joints + singles), rope=2, timestep=1)
    total = {key: 8 * value for key, value in one.items()}
    return dict(cases=list(CASES), per_pass=one, per_case_transitions=2 * one["transitions"],
                forecast={**total, "module_entries": 0, "full_blocks": 0, "scheduler_calls": 0},
                native={**total, "full_blocks": 8 * (joints + singles), "scheduler_calls": 8, "module_entries": 8 * (24 + 29 * joints + 9 * singles)},
                model_loads=1, decoder_loads=1, reference_cut_reads=4, predicted_decoder_calls=4, native_reference_decoder_calls=4,
                encodes=0, noise_draws=0, act_receipts=0, session_continue_receipts=0,
                predictions=PREDICTIONS, execution_id=EXECUTION_ID)



def source_closure(root, sources):
    """Exact supplied cut/page/blob whitelist, excluding later native evidence."""
    result = set()
    for source in sources.values():
        store = Path(source["store"])
        path = store / "cuts" / (source["cut"] + ".json")
        data = json.loads((root / path).read_text())
        result.add(path.as_posix())
        result.update((store / "pages" / (value + ".json")).as_posix() for value in data["slots"].values())
        result.update((store / "blobs" / (value + ".safetensors")).as_posix() for value in data["physical_blobs"])
    return result


def validate_config(config):
    _, m, _ = helpers(config)
    require, sha = m["require"], m["sha"]
    for key in ("transformer_config", "scheduler_config", "binding", "parameter_manifest", "retrieval_script", "vae_config", "vae_parameter_manifest"):
        require(sha(config[key]) == config[key + "_sha256"], "source metadata differs: " + key)
    spec = json.loads(Path(config["transformer_config"]).read_text())
    require(set(config["inputs"]) == set(config["references"]) == set(CASES), "four unchanged prior output cases required")
    require(type(config["source_reduction_policy"]) is bool and 1 <= config.get("cpu_threads", 4) <= 4,
            "explicit numerical policy required")
    require(0 < config.get("max_wall_seconds", 900) <= 1800 and 1 <= config.get("witness_values", 16) <= 64, "finite work/witness budget required")
    if config.get("profile") == "tiny-cpu":
        require(config.get("device") == "cpu" and spec["num_layers"] == spec["num_single_layers"] == 2
                and spec["num_attention_heads"] * spec["attention_head_dim"] <= 64, "tiny-only profile required")
    else:
        require(config.get("device", "cuda") == "cuda" and config["source_reduction_policy"] is True
                and (spec["num_layers"], spec["num_single_layers"], spec["num_attention_heads"], spec["attention_head_dim"], spec["in_channels"], spec["joint_attention_dim"])
                == (5, 20, 24, 128, 128, 7680) and spec["guidance_embeds"] is False, "production geometry/policy differs")
        require(config["transformer_config_sha256"] == "09733c74a3da6d17dd0a0472a091a8950c7c6935889c32c16cc800ede05029de"
                and config["scheduler_config_sha256"] == "067afb012cef64553a763447d1efd93daeffcc0123ca7e25b09f8de20b90762e", "original configuration required")
    require(sha(config["retrieval_script"]) == m["PARAMETER_RETRIEVER_SHA256"], "qualified retriever differs")
    for root_key, sources_key, files_key in (("input_root", "inputs", "input_files"), ("reference_root", "references", "reference_files")):
        root = Path(config[root_key]).resolve()
        for source in config[sources_key].values():
            require(not Path(source["store"]).is_absolute() and ".." not in Path(source["store"]).parts, "unsafe source store")
        if root_key == "input_root":
            require(set(config[files_key]) == source_closure(root, config[sources_key]), "source whitelist must contain exactly complete declared cuts")
            for relative, record in config[files_key].items():
                path = root / relative
                require(path.resolve().is_relative_to(root) and not path.is_symlink() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "source custody differs: " + relative)
    return contract(config)


def measure(predicted, native):
    import numpy as np

    a, b = np.asarray(predicted, np.float64), np.asarray(native, np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("measurement geometry/nonfinite")
    delta = a - b
    rms = lambda value: float(np.sqrt(np.mean(value * value)))
    scale = rms(b)
    denominator = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    return dict(orientation="predicted-minus-native", rms=rms(delta), native_rms=scale, predicted_rms=rms(a),
                relative_rms=None if scale == 0 else rms(delta) / scale, signed_mean=float(delta.mean()), maxabs=float(abs(delta).max()),
                changed_values=int(np.count_nonzero(delta)), sample_count=int(delta.size), exact=bool(np.array_equal(a, b)),
                cosine=None if denominator == 0 else float(np.dot(a.ravel(), b.ravel()) / denominator),
                abs_error_quantiles={str(q): float(np.quantile(abs(delta), q)) for q in (.5, .95, .99, 1.)})


def run(config, *, progress=None):
    declared = validate_config(config)
    import numpy as np
    import torch
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2Transformer2DModel,
    )
    from diffusers.models import attention_dispatch
    from diffusers.models.transformers import transformer_flux2 as native
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.core import StateCut
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction
    from mdb.forward_recording import tensor_descriptor

    graph, m, recipes = helpers(config)
    require, save, sha = m["require"], m["save"], m["sha"]
    require(sha(inspect.getfile(native)) == config["native_source_sha256"], "qualified native source differs")
    output = Path(config["output"]).resolve()
    require(all(not output.is_relative_to(Path(config[key]).resolve()) for key in ("input_root", "reference_root", "parameter_root", "vae_parameter_root")), "output overlaps immutable source")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", config)
    start, handles, transactions = time.monotonic(), [], []
    counts = {lane: dict.fromkeys((*declared["per_pass"], "full_blocks", "scheduler_calls", "module_entries"), 0) for lane in ("forecast", "native")}
    totals = dict(model_loads=0, decoder_loads=0, reference_cut_reads=0, encodes=0, noise_draws=0, vae_calls=0, predicted_decoder_calls=0, native_reference_decoder_calls=0, guard_refusals=0)
    decoder_counts = {lane: dict(module_entries=0, conv2d=0, groupnorm=0, linear=0, attention=0, resnet=0, upsample=0, silu=0, sdpa=0, decoder_entries=0) for lane in ("predicted", "native_reference")}
    active = dict(lane="setup", case=None, phase=None, index=0, ordinal={}, ports=[], positioned=[])
    original_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark,
                      torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                      torch.get_deterministic_debug_mode(), torch.get_num_threads(), torch.get_float32_matmul_precision())
    original_dispatch, original_rotate, original_scheduler = native.dispatch_attention_fn, native.apply_rotary_emb, FlowMatchEulerDiscreteScheduler.step
    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    g = graph["functional_project"].__wrapped__.__globals__
    changed_globals = {}
    forecast_tx, forecast_inventory, forecast_signatures = None, {}, {}
    latent_tx = None
    latent_inventory, latent_signatures = {}, {}
    selected_profiles = {}

    def event(kind, **detail):
        row = dict(kind=kind, seconds=time.monotonic() - start, **detail)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)
        require(row["seconds"] <= config.get("max_wall_seconds", 900), "finite pass wall exceeded")

    def inventory(prefixes):
        return {path.relative_to(output).as_posix(): dict(sha256=sha(path), bytes=path.stat().st_size)
                for prefix in prefixes for path in sorted((output / prefix).rglob("*")) if path.is_file()}

    def verify_files(records, *, signatures=None):
        result = {}
        for relative, record in records.items():
            path = output / relative
            stat = path.stat()
            signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            require(not path.is_symlink() and path.resolve().is_relative_to(output), "external evidence path changed")
            if signatures is None:
                require(stat.st_size == record["bytes"] and sha(path) == record["sha256"], "sealed external evidence changed: " + relative)
            else:
                require(signature == signatures[relative], "sealed external evidence signature changed: " + relative)
            result[relative] = signature
        return result

    def guard_target():
        if forecast_tx is None or forecast_tx.phase != "execute":
            totals["guard_refusals"] += 1
            raise ValueError("native target outside globally sealed execution")
        forecast_tx.verify()
        verify_files(forecast_inventory, signatures=forecast_signatures)

    def array(value):
        return value.detach().float().cpu().numpy()

    def witness(value):
        require(bool(torch.isfinite(value).all()), "nonfinite primitive output")
        descriptor = dict(tensor_descriptor(value), stride=list(value.stride()))
        flat = value.detach().reshape(-1)
        indices = np.unique(np.linspace(0, flat.numel() - 1, min(config.get("witness_values", 16), flat.numel())).astype(np.int64))
        return dict(descriptor=descriptor, indices=indices.tolist(), values=flat[torch.as_tensor(indices, device=flat.device)].float().cpu().tolist(), full_values_retained=False)

    def tensors(value, prefix=""):
        if isinstance(value, torch.Tensor):
            return {prefix: witness(value)}
        if isinstance(value, (tuple, list)):
            return {key: item for index, child in enumerate(value) for key, item in tensors(child, prefix + "." + str(index)).items()}
        return {}

    def record_port(kind, name, args, result):
        ordinal = active["ordinal"].get(name, 0)
        active["ordinal"][name] = ordinal + 1
        active["ports"].append(dict(id=name + "." + str(ordinal), kind=kind,
            fields={**tensors(args, "input"), **tensors(result, "output")}))

    try:
        torch.set_num_threads(config.get("cpu_threads", 4))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = False, True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = config["source_reduction_policy"]
        backend_name = attention_dispatch._AttentionBackendRegistry.get_active_backend()[0]
        require(str(getattr(backend_name, "value", backend_name)) == "native", "actual native attention backend required")
        device = config.get("device", "cuda")
        model_spec = json.loads(Path(config["transformer_config"]).read_text())
        scheduler_spec = json.loads(Path(config["scheduler_config"]).read_text())
        manifest = json.loads(Path(config["parameter_manifest"]).read_text())
        require(config.get("profile") == "tiny-cpu" or len(manifest["artifacts"]) == 169, "original169 tensor recipes required")
        event("checkpoint_validation_started", recipes=len(manifest["artifacts"]))
        bank = recipes["RecipeBank"](manifest, config["parameter_root"], max_cache_bytes=1)
        with torch.device("meta"):
            model = Flux2Transformer2DModel.from_config(model_spec)
        actual_names = set(model.state_dict())
        recipe_names = {record["source"]["tensor"] for record in manifest["artifacts"]}
        require(actual_names == recipe_names and len(recipe_names) == len(manifest["artifacts"]), "state_dict must exactly match original named recipes")
        weights, expected_weights = {}, {}
        for record in manifest["artifacts"]:
            source = record["source"]
            name = source["tensor"]
            require(list(model.state_dict()[name].shape) == source["shape"] and source["dtype"] == "BF16", "original parameter shape/dtype differs")
            recipe = dict(path=record["path"], sha256=record["sha256"], parameter_tensor=name, stored_shape=source["shape"], descriptor=dict(sha256=source["payload_sha256"]))
            value = torch.from_numpy(bank.load(recipe)).to(torch.bfloat16)
            require(tensor_descriptor(value)["sha256"] == source["payload_sha256"], "original parameter byte restoration differs")
            weights[name], expected_weights[name] = value.to(device), source["payload_sha256"]
        model.load_state_dict(weights, strict=True, assign=True)
        model.eval().requires_grad_(False)
        del weights, value
        totals["model_loads"] += 1
        event("original_transformer_hydrated", model_loads=totals["model_loads"], parameters=len(expected_weights))
        graph["validate_model"](model)
        require(all(not getattr(block.attn, "fused_projections", False) for block in model.transformer_blocks), "unfused original joint projections required")
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_spec)
        adapter = Flux2KleinAdapter(model, scheduler, granularity="operation", device=device)
        component_sources = {Path(inspect.getfile(kind)).name: sha(inspect.getfile(kind)) for kind in
                             (Flux2KleinAdapter, FlowMatchEulerDiscreteScheduler, type(model.norm_out), type(model.time_guidance_embed.time_proj))}
        component_sources["attention_dispatch.py"] = sha(inspect.getfile(attention_dispatch))
        require(component_sources == {"flux2.py": "2004b81971a258f1ca6295fe69d24767dc6bb6b92e103fa30cf6ccb0249b5a86",
                "scheduling_flow_match_euler_discrete.py": "8b96f25a6170480fb02decb58904e026250d6d68be5de8a7b8785b4c721e9d80",
                "normalization.py": "e92ebbb130082578f3cff3361891a1adaa32331347e7bb09e4c70b208a0a6059",
                "embeddings.py": "4eb810f715786eb1f24f2a4641e529817f7c951b9caf2f7925811329a03ec796",
                "attention_dispatch.py": "2acfeb88d200a6466c0b4c718e5699efd8dc2e61d8b7c96d9d7c938ffbb25bc8"}, "original native component source differs")
        # Hydrate only decoder/normalization tensors. Unused VAE encoder and
        # quant_conv stay meta; they are never called or copied to evidence.
        vae_spec = json.loads(Path(config["vae_config"]).read_text())
        vae_manifest = json.loads(Path(config["vae_parameter_manifest"]).read_text())
        vae_bank = recipes["RecipeBank"](vae_manifest, config["vae_parameter_root"], max_cache_bytes=1)
        with torch.device("meta"):
            vae = AutoencoderKLFlux2.from_config(vae_spec)
        vae_state = vae.state_dict()
        required_vae = {name for name in vae_state if name.startswith(("decoder.", "post_quant_conv.")) or name in ("bn.running_mean", "bn.running_var")}
        require({row["source"]["tensor"] for row in vae_manifest["artifacts"]} == required_vae and len(vae_manifest["artifacts"]) == len(required_vae), "exact decoder/BN recipe set required")
        if config.get("profile") != "tiny-cpu":
            require(len(required_vae) == 142 and config["vae_config_sha256"] == "0d6dfb69ae95a5e2ac9836284bbb63d8b38ce67b25ba2dff380752b2a10ab948", "original VAE configuration/142entries required")
            require({row["source"]["file_sha256"] for row in vae_manifest["artifacts"]} == {"ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"}, "original VAE checkpoint required")
        vae_weights, expected_vae = {}, {}
        for row in vae_manifest["artifacts"]:
            source = row["source"]
            name = source["tensor"]
            require(source["dtype"] == "BF16" and list(vae_state[name].shape) == source["shape"], "decoder recipe geometry/dtype differs")
            value = torch.from_numpy(vae_bank.load(dict(path=row["path"], sha256=row["sha256"], parameter_tensor=name, stored_shape=source["shape"], descriptor=dict(sha256=source["payload_sha256"])))).to(torch.bfloat16)
            require(tensor_descriptor(value)["sha256"] == source["payload_sha256"], "decoder byte restoration differs")
            vae_weights[name] = value.to("cpu" if name.startswith("bn.") else device)
            expected_vae[name] = source["payload_sha256"]
        result = vae.load_state_dict(vae_weights, strict=False, assign=True)
        require(not result.unexpected_keys and set(result.missing_keys) == set(vae_state) - required_vae - {"bn.num_batches_tracked"}, "unused VAE component boundary differs")
        vae.bn.num_batches_tracked = torch.tensor(0, dtype=torch.int64)
        vae.eval().requires_grad_(False)
        require(not vae.use_tiling and not vae.use_slicing and tuple(vae.config.patch_size) == (2, 2), "untiled original VAE required")
        totals["decoder_loads"] += 1
        vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        require(config["height"] % (2 * vae_scale) == config["width"] % (2 * vae_scale) == 0, "decoder image geometry differs")
        image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale * 2)
        import diffusers.image_processor as generic_image_processor
        import diffusers.models.attention_processor as vae_attention
        import diffusers.models.resnet as vae_resnet
        decoder_sources = {"autoencoder_kl_flux2.py": sha(inspect.getfile(AutoencoderKLFlux2)), "vae.py": sha(inspect.getfile(type(vae.decoder))),
                           "resnet.py": sha(inspect.getfile(vae_resnet)), "attention_processor.py": sha(inspect.getfile(vae_attention)),
                           "image_processor.py": sha(inspect.getfile(generic_image_processor)), "flux2_image_processor.py": sha(inspect.getfile(Flux2ImageProcessor)),
                           "pipeline_flux2_klein.py": sha(inspect.getfile(Flux2KleinPipeline))}
        require(decoder_sources == config["decoder_source_sha256"], "original native decoder source differs")
        vae_signature = {name: (id(value), value.data_ptr(), value._version) for name, value in vae.state_dict(keep_vars=True).items() if name in required_vae}
        decoder_contract = dict(source_sha256=decoder_sources, config_sha256=config["vae_config_sha256"], parameter_manifest_sha256=config["vae_parameter_manifest_sha256"], tensors=len(required_vae),
            parameter_bytes=sum(row["source"]["data_offsets"][1] - row["source"]["data_offsets"][0] for row in vae_manifest["artifacts"]), loaded_names=sorted(required_vae),
            unused_meta_names=sorted(set(vae_state) - required_vae - {"bn.num_batches_tracked"}), unused_initialized_counter=dict(name="bn.num_batches_tracked", value=0, used_by_decode=False, pretrained_tensor_claim=False), normalization="CPU BN statistics/sqrt before native BF16 multiply/add; native pipeline ID unpack/unpatch helpers", independent_vae_prediction=False,
            dtype="torch.bfloat16", native_vae_device=device, normalization_device="cpu", height=config["height"], width=config["width"], module_graph={name: type(module).__name__ for name, module in vae.named_modules()})
        del vae_weights, vae_state, value, result
        event("original_decoder_hydrated", tensors=len(required_vae), decoder_loads=1, unused_components="meta encoder/quant_conv; no forward")
        signature = {name: (id(value), value.data_ptr(), value._version) for name, value in model.named_parameters()}
        names = {id(module): name for name, module in model.named_modules()}
        blocks = {id(module) for module in (*model.transformer_blocks, *model.single_transformer_blocks)}
        input_store = LocalStore(output / "input-store")
        input_records, source_execution = {}, {}
        for role in CASES:
            source = config["inputs"][role]
            cut = LocalStore(Path(config["input_root"]) / source["store"]).load(source["cut"], device="cpu")
            cut.verify()
            p = cut.payload
            require(cut.model_identity == config["expected_source_model_identity"] and (p["step"], p["phase"], p["cursor"]) == (2, "project", 0)
                    and all(p[key] is None for key in ("image", "text", "context", "noise_prediction")), "complete prior sealed predicted step2 project source required")
            require(dict(cut.execution).get("component_pass", {}).get("lane") == "forecast" and dict(cut.execution)["component_pass"]["id"] == "mdb-flux-functional-pass-v1" and dict(cut.execution)["component_pass"]["case"] == role, "archived prior forecast lineage required")
            require(p["latent"].dtype == p["conditioning"].dtype == torch.bfloat16 and p["img_ids"].dtype == p["txt_ids"].dtype == torch.int64, "original input dtype differs")
            input_store.save(cut)
            input_records[role] = dict(cut=cut.fingerprint, parent=cut.parent, point=dict(cut.execution_point), payload=describe(p))
            source_execution[role] = dict(cut.execution)
        # Inputs are the exact archived predictions. Never derive new-3over8 again.
        for case in CASES:
            p = input_store.load(input_records[case]["cut"], device="cpu").payload
            require(len(p["timesteps"]) == 4 and len(p["sigmas"]) == 5, "original complete four-step schedule required")
        del p
        input_inventory = inventory(["input-store"])
        input_tx = ForecastTransaction(output / "input-transaction", model_identity=config["expected_source_model_identity"], input_identity=m["identity"](input_records), source_identity=config["parameter_manifest_sha256"], array_storage="files-v1", frontier=dict(kind="architectural", observed="Four archived sealed PREDICTED step2 commits; no rederivation or observed next-project output"), budgets=dict(max_wall_ms=config.get("max_wall_seconds", 900) * 1000, max_bytes=64 << 20))
        transactions.append(input_tx)
        input_tx.predict("inputs", lambda context: dict(predictions=dict(cases=input_records, external_files=input_inventory, target_entries=0)))
        input_seal = input_tx.seal()
        input_tx.close()
        verify_files(input_inventory)
        event("inputs_globally_sealed", seal=input_seal, full_target_entries=0)

        def module_entry(module, args):
            lane = active["lane"]
            if lane == "forecast":
                totals["guard_refusals"] += 1
                raise ValueError("functional prediction entered original native module.forward")
            require(lane == "native", "undeclared native module entry")
            require(forecast_tx is not None and forecast_tx.phase == "execute", "native target outside globally sealed execution")
            if id(module) in blocks or active["phase"] == "project" and module is model.time_guidance_embed:
                guard_target()
                event("native_entry", lane=lane, case=active["case"], phase=active["phase"], index=active["index"], module=names[id(module)])
            counts[lane]["module_entries"] += 1
            if id(module) in blocks:
                counts[lane]["full_blocks"] += 1
        for module in model.modules():
            handles.append(module.register_forward_pre_hook(module_entry, prepend=True))
            kind = ("linear" if isinstance(module, torch.nn.Linear) else "layernorm" if isinstance(module, torch.nn.LayerNorm) else "rmsnorm" if isinstance(module, torch.nn.RMSNorm)
                    else "swiglu" if isinstance(module, native.Flux2SwiGLU) else "rope" if module is model.pos_embed else "timestep" if module is model.time_guidance_embed.time_proj else None)
            if kind:
                def observed(module, args, result, kind=kind):
                    counts[active["lane"]][kind] += 1
                    record_port(kind, names[id(module)], args, result)
                handles.append(module.register_forward_hook(observed))

        def global_entry(module, args):
            if active["lane"] == "forecast":
                totals["guard_refusals"] += 1
                raise ValueError("functional prediction entered original native module.forward")
        handles.append(torch.nn.modules.module.register_module_forward_pre_hook(global_entry))
        def decoder_entry(module, args):
            origin = active.get("decoder_origin")
            require(active["lane"] == "decoder" and origin in decoder_counts, "decoder module outside declared real consumer")
            if module is vae.decoder:
                if origin == "predicted":
                    require(latent_tx is not None and latent_tx.phase == "execute", "predicted decoder before latent cohort seal")
                    latent_tx.verify()
                    verify_files(latent_inventory, signatures=latent_signatures)
                else:
                    guard_target()
            census = decoder_counts[origin]
            census["module_entries"] += 1
            typename = type(module).__name__
            kind = ("conv2d" if isinstance(module, torch.nn.Conv2d) else "groupnorm" if isinstance(module, torch.nn.GroupNorm) else "linear" if isinstance(module, torch.nn.Linear)
                    else "attention" if typename == "Attention" else "resnet" if typename == "ResnetBlock2D" else "upsample" if typename == "Upsample2D" else "silu" if isinstance(module, torch.nn.SiLU) else None)
            if kind:
                census[kind] += 1
            if module is vae.decoder:
                census["decoder_entries"] += 1
                value = args[0]
                active["actual_decoder_input"] = dict(active["save_decoder_operand"]("actual-decoder-input-bf16", value.detach().contiguous().view(torch.uint16).cpu().numpy()), native_descriptor=dict(tensor_descriptor(value), stride=list(value.stride())), encoding="bf16-bits-v1", meaning="Actual args[0] consumed by native decoder after native post_quant_conv; observation hook returns None")
                event("native_decoder_entry", origin=origin, case=active["case"], native_future_transitions=counts["native"]["transitions"])
        for module in vae.modules():
            handles.append(module.register_forward_pre_hook(decoder_entry, prepend=True))

        def native_sdpa(*args, **kwargs):
            if active["lane"] == "decoder":
                decoder_counts[active["decoder_origin"]]["sdpa"] += 1
            return original_sdpa(*args, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = native_sdpa

        def functional_wrapper(key, kind):
            original = g[key]
            changed_globals[key] = original
            def call(*args, **kwargs):
                counts["forecast"][kind] += 1
                if key == "_attention" and "forecast" not in selected_profiles:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                        result = original(*args, **kwargs)
                    selected_profiles["forecast"] = [dict(name=item.key, calls=item.count) for item in profile.key_averages() if "scaled_dot" in item.key]
                else:
                    result = original(*args, **kwargs)
                if key == "_swiglu":
                    label, operands = active.get("swiglu_label", f"single_transformer_blocks.{active['index']}.attn.mlp_act_fn"), args
                elif key == "_rotate":
                    label, operands = "rotary", args
                    active["positioned"].append(result)
                elif key == "_time_projection":
                    label, operands = "time_guidance_embed.time_proj", args[:1]
                elif key == "_attention":
                    label, operands = "attention", (*active["positioned"][-2:], args[2])
                elif key == "_rope":
                    label, operands = names[id(args[0])], (args[1][0] if args[1].ndim == 3 else args[1],)
                else:
                    label, operands = names[id(args[0])], args[1:]
                record_port(kind, label, operands, result)
                return result
            g[key] = call
        for key, kind in (("_linear", "linear"), ("_norm", "layernorm"), ("_rms", "rmsnorm"), ("_swiglu", "swiglu"), ("_rope", "rope"), ("_time_projection", "timestep"), ("_rotate", "rotary"), ("_attention", "sdpa")):
            functional_wrapper(key, kind)
        changed_globals["_ff"] = g["_ff"]
        def ff(module, value):
            active["swiglu_label"] = names[id(module.act_fn)]
            try:
                return changed_globals["_ff"](module, value)
            finally:
                active.pop("swiglu_label", None)
        g["_ff"] = ff

        def rotate(*args, **kwargs):
            result = original_rotate(*args, **kwargs)
            counts[active["lane"]]["rotary"] += 1
            record_port("rotary", "rotary", args[:2], result)
            return result
        def dispatch(*args, **kwargs):
            if active["lane"] not in selected_profiles:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                    result = original_dispatch(*args, **kwargs)
                selected_profiles[active["lane"]] = [dict(name=item.key, calls=item.count) for item in profile.key_averages() if "scaled_dot" in item.key]
            else:
                result = original_dispatch(*args, **kwargs)
            counts[active["lane"]]["sdpa"] += 1
            record_port("sdpa", "attention", args[:3], result.flatten(2, 3).to(args[0].dtype))
            return result
        def scheduler_step(module, *args, **kwargs):
            require(active["lane"] == "native", "native scheduler outside verification")
            guard_target()
            counts["native"]["scheduler_calls"] += 1
            event("native_scheduler_entry", case=active["case"])
            return original_scheduler(module, *args, **kwargs)
        native.apply_rotary_emb, native.dispatch_attention_fn, FlowMatchEulerDiscreteScheduler.step = rotate, dispatch, scheduler_step
        forecast_store, native_store = (LocalStore(output / name) for name in ("prediction-store", "native-store"))
        rows = {lane: {case: [] for case in CASES} for lane in ("forecast", "native")}
        comparisons, consumers, reference_records, reference_execution = {}, {}, {}, {}

        def publish(state, *, lane, case, parent, store):
            execution = {**(source_execution if lane == "forecast" else reference_execution)[case], "component_pass": dict(id=EXECUTION_ID, lane=lane, case=case,
                source_cut=(input_records if lane == "forecast" else reference_records)[case]["cut"], verification_adapter_identity=adapter.model_identity, receipt_kind="component-derived StateCut, not Session/Act receipt")}
            cut = StateCut(config["expected_source_model_identity"], execution, adapter.boundary(state), parent, state, adapter.point(state).to_dict(),
                dict(schema="mdb-functional-remaining-state-v1", phase=state["phase"], cursor=state["cursor"], complete_state=True))
            store.save(cut)
            return dict(cut=cut.fingerprint, parent=cut.parent, point=dict(cut.execution_point), boundary=cut.boundary,
                        fields={key: describe(value) for key, value in state.items() if key in ("image", "text", "latent", "context", "noise_prediction")})

        def functional_step(state):
            after = dict(state)
            phase, index = state["phase"], state["cursor"]
            if phase == "project":
                after["image"], after["text"], after["context"] = graph["functional_project"](model, state["latent"], state["conditioning"], state["img_ids"], state["txt_ids"], state["timesteps"][state["step"]])
                after["phase"] = "joint"
            elif phase == "joint":
                after["image"], after["text"] = graph["functional_joint"](model, index, state["image"], state["text"], state["context"])
                after["cursor"] = index + 1
                if after["cursor"] == len(model.transformer_blocks):
                    after["phase"], after["cursor"] = "single", 0
            elif phase == "single":
                after["image"], after["text"] = graph["functional_single"](model, index, state["image"], state["text"], state["context"])
                after["cursor"] = index + 1
                if after["cursor"] == len(model.single_transformer_blocks):
                    after["phase"] = "readout"
            elif phase == "readout":
                after["noise_prediction"] = graph["functional_readout"](model, state["image"], state["context"])
                after["phase"] = "scheduler"
            else:
                require(phase == "scheduler", "unknown functional phase")
                after["latent"] = graph["functional_scheduler"](state["latent"], state["noise_prediction"], state["sigmas"], state["step"])
                after.update(step=state["step"] + 1, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
            return after

        def compare_states(predicted, actual):
            result = {}
            for key in ("image", "text", "latent", "noise_prediction"):
                require((predicted[key] is None) == (actual[key] is None), "complete-state reset mismatch")
                if actual[key] is not None:
                    result[key] = measure(array(predicted[key]), array(actual[key]))
            if actual["context"] is not None:
                for key in ("temb", "img_mod", "txt_mod", "single_mod"):
                    result["context." + key] = measure(array(predicted["context"][key]), array(actual["context"][key]))
                for index in (0, 1):
                    result["context.rope." + str(index)] = measure(array(predicted["context"]["rope"][index]), array(actual["context"]["rope"][index]))
            return result

        def pass_case(lane, case):
            if lane == "forecast":
                input_tx.verify()
                verify_files(input_inventory)
            if lane == "forecast":
                parent = input_records[case]["cut"]
                state = input_store.load(parent, device=device).payload
            else:
                guard_target()
                root = Path(config["reference_root"])
                require(set(config["reference_files"]) == source_closure(root, config["references"]), "reference whitelist must contain exactly complete declared cuts")
                for relative, record in config["reference_files"].items():
                    path = root / relative
                    require(path.resolve().is_relative_to(root) and not path.is_symlink() and sha(path) == record["sha256"] and path.stat().st_size == record["bytes"], "reference custody differs")
                source = config["references"][case]
                cut = LocalStore(root / source["store"]).load(source["cut"], device=device)
                cut.verify()
                state = cut.payload
                require(dict(cut.execution).get("component_pass", {}).get("lane") == "native" and dict(cut.execution)["component_pass"]["id"] == "mdb-flux-functional-pass-v1" and dict(cut.execution)["component_pass"]["case"] == case, "separate prior native-reference lineage required")
                require(cut.model_identity == config["expected_source_model_identity"] and (state["step"], state["phase"], state["cursor"]) == (2, "project", 0), "prior native-reference source differs")
                require(all(state[key] is None for key in ("image", "text", "context", "noise_prediction")), "reference source slots not reset")
                LocalStore(output / "reference-input-store").save(cut)
                reference_execution[case] = dict(cut.execution)
                reference_records[case] = dict(cut=cut.fingerprint, parent=cut.parent, payload=describe(state), point=dict(cut.execution_point))
                parent = cut.fingerprint
                totals["reference_cut_reads"] += 1
                event("reference_cut_read", case=case, cut=parent, forecast_seal=cohort_seal)

            original_carried = {key: describe(value) for key, value in state.items() if key in ("conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")}
            for number in range(declared["per_case_transitions"]):
                active.update(lane=lane, case=case, phase=state["phase"], index=state["cursor"], ordinal={}, ports=[], positioned=[])
                before = dict(step=state["step"], phase=state["phase"], cursor=state["cursor"])
                with torch.inference_mode():
                    state = functional_step(state) if lane == "forecast" else adapter.advance(state)
                counts[lane]["transitions"] += 1
                require({key: describe(state[key]) for key in original_carried} == original_carried, "undeclared carried-state mutation")
                record = publish(state, lane=lane, case=case, parent=parent, store=forecast_store if lane == "forecast" else native_store)
                parent = record["cut"]
                ports_path = f"{lane}-ports/{case}/{number:02d}.json"
                save(output / ports_path, dict(before=before, ports=active["ports"], scope="Full descriptors and fixed witness values; complete block state retained separately."))
                row = dict(number=number, before=before, state=record, ports=ports_path)
                rows[lane][case].append(row)
                if lane == "native":
                    predicted = forecast_store.load(rows["forecast"][case][number]["state"]["cut"], device="cpu").payload
                    values = compare_states(predicted, state)
                    expected_ports = json.loads((output / rows["forecast"][case][number]["ports"]).read_text())["ports"]
                    expected_by_id = {port["id"]: port for port in expected_ports}
                    actual_by_id = {port["id"]: port for port in active["ports"]}
                    ports = {}
                    for key, expected_port in expected_by_id.items():
                        actual_port = actual_by_id.get(key)
                        fields = expected_port["fields"]
                        available = actual_port is not None and set(fields) == set(actual_port["fields"])
                        exact = available and all({k: v for k, v in fields[name]["descriptor"].items() if k != "stride"} == {k: v for k, v in actual_port["fields"][name]["descriptor"].items() if k != "stride"} for name in fields)
                        ports[key] = dict(available=available, exact=exact,
                            layout_exact=available and all(fields[name]["descriptor"]["stride"] == actual_port["fields"][name]["descriptor"]["stride"] for name in fields))
                    comparisons.setdefault(case, []).append(dict(number=number, before=before, fields=values, primitive_ports=ports,
                        unmatched_native_ports=sorted(set(actual_by_id) - set(expected_by_id))))
                event("boundary_completed", lane=lane, case=case, number=number, before=before, cut=record["cut"])
            require((state["step"], state["phase"], state["cursor"]) == (4, "project", 0) and all(state[key] is None for key in ("image", "text", "context", "noise_prediction")), "pass commit/reset closure differs")
            save(output / (lane + "-rows") / (case + ".json"), rows[lane][case])

        def decode_case(case, origin, store, source_row):
            if origin == "predicted":
                require(latent_tx is not None and latent_tx.phase == "execute" and totals["reference_cut_reads"] == counts["native"]["transitions"] == 0, "predicted decode before global latent seal or after native future")
                latent_tx.verify()
                verify_files(latent_inventory, signatures=latent_signatures)
            else:
                guard_target()
                require(counts["native"]["transitions"] == declared["native"]["transitions"], "all native futures must complete before reference decodes")
            cut = store.load(source_row["cut"], device=device)
            state = cut.payload
            require(state["step"] == len(state["timesteps"]) == 4 and state["phase"] == "project" and all(state[key] is None for key in ("image", "text", "context", "noise_prediction")), "complete final state required by real decoder")
            active.update(lane="decoder", decoder_origin=origin, case=case, phase="decode")
            prefix = "predicted-decoder" if origin == "predicted" else "native-reference-decoder"
            def full_array(name, value):
                path = Path(prefix) / case / (name + ".npy")
                target = output / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                return dict(path=path.as_posix(), sha256=sha(target), bytes=target.stat().st_size, shape=list(value.shape), dtype=str(value.dtype))
            active["save_decoder_operand"] = full_array
            with torch.inference_mode():
                unpacked = Flux2KleinPipeline._unpack_latents_with_ids(state["latent"], state["img_ids"], config["height"] // (vae_scale * 2), config["width"] // (vae_scale * 2))
                mean = vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(unpacked)
                std = (vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(unpacked)
                decoder_input = Flux2KleinPipeline._unpatchify_latents(unpacked * std + mean)
                input_record = dict(full_array("decoder-input-bf16", decoder_input.contiguous().view(torch.uint16).cpu().numpy()), native_descriptor=dict(tensor_descriptor(decoder_input), stride=list(decoder_input.stride())), encoding="bf16-bits-v1", meaning="Actual native VAE.decode argument before native post_quant_conv")
                before_entries = decoder_counts[origin]["decoder_entries"]
                decoded = vae.decode(decoder_input.to(device), return_dict=False)[0]
                require(decoder_counts[origin]["decoder_entries"] == before_entries + 1, "actual native decoder did not run")
                totals["vae_calls"] += 1
                totals[origin + "_decoder_calls"] += 1
                decoded_record = dict(full_array("decoded-bf16", decoded.contiguous().view(torch.uint16).cpu().numpy()), native_descriptor=dict(tensor_descriptor(decoded), stride=list(decoded.stride())), encoding="bf16-bits-v1")
                image = image_processor.postprocess(decoded, output_type="pil")[0]
                pixels = np.asarray(image)
                pixel_record = full_array("pixels", pixels)
                png_path = output / prefix / case / "image.png"
                image.save(png_path)
            record = dict(origin=origin, input_cut=cut.fingerprint, consumed_state=describe(state), decoder_input=input_record, actual_decoder_input=active["actual_decoder_input"], decoded=decoded_record, pixels=pixel_record,
                png=dict(path=png_path.relative_to(output).as_posix(), sha256=sha(png_path), bytes=png_path.stat().st_size), independent_decoder_prediction=False,
                label="real decoder on predicted final latent" if origin == "predicted" else "real decoder on separate native-reference final latent",
                native_future_transitions=counts["native"]["transitions"], decoder_contract_sha256=m["identity"](decoder_contract))
            save(output / prefix / case / "consumer.json", record)
            event("real_decoder_completed", case=case, origin=origin, input_cut=cut.fingerprint, pixels=pixel_record, native_future_transitions=counts["native"]["transitions"])
            active.pop("save_decoder_operand")
            active.pop("actual_decoder_input")
            return record

        for case in CASES:
            pass_case("forecast", case)
        require(counts["native"]["full_blocks"] == counts["native"]["scheduler_calls"] == 0, "native target entered before cohort seal")
        latent_inventory = inventory(["prediction-store", "forecast-ports", "forecast-rows"])
        latent_tx = ForecastTransaction(output / "latent-forecast-transaction", model_identity=config["expected_source_model_identity"], input_identity=input_seal["seal_sha256"], source_identity=config["parameter_manifest_sha256"], array_storage="files-v1", frontier=dict(kind="architectural", observed="Four unchanged prior PREDICTED step2 commits"), budgets=dict(max_wall_ms=config.get("max_wall_seconds", 900)*1000, max_bytes=256 << 20, max_files=16384))
        transactions.append(latent_tx)
        latent_tx.predict("cohort", lambda context: dict(predictions=dict(cases=rows["forecast"], external_files=latent_inventory, forecast_counts=counts["forecast"], native_full_blocks_at_seal=0, reference_cut_reads=0)))
        latent_seal = latent_tx.seal()
        latent_signatures = verify_files(latent_inventory)
        event("latents_globally_sealed", seal=latent_seal, native_full_blocks=0, reference_cut_reads=0)
        def predicted_decoders(context):
            for case in CASES:
                consumers.setdefault(case, {})["predicted"] = decode_case(case, "predicted", forecast_store, rows["forecast"][case][-1]["state"])
            return consumers
        latent_tx.execute("real_predicted_latent_decoders", predicted_decoders)
        latent_tx.close()
        forecast_inventory = inventory(["prediction-store", "forecast-ports", "forecast-rows", "predicted-decoder"])
        forecast_tx = ForecastTransaction(output / "forecast-transaction", model_identity=config["expected_source_model_identity"], input_identity=latent_seal["seal_sha256"], source_identity=config["vae_parameter_manifest_sha256"], array_storage="files-v1", frontier=dict(kind="architectural", observed="Four actual VAE consumers of predicted final latents; no native future denoiser or native-reference input read"), budgets=dict(max_wall_ms=config.get("max_wall_seconds", 900)*1000, max_bytes=256 << 20, max_files=16384))
        transactions.append(forecast_tx)
        forecast_tx.predict("cohort", lambda context: dict(predictions=dict(cases=rows["forecast"], decoded_consumers=consumers, external_files=forecast_inventory, input_seal=input_seal, latent_seal=latent_seal, predictions=PREDICTIONS, forecast_counts=counts["forecast"], native_full_blocks_at_seal=0, reference_cut_reads=0)))
        cohort_seal = forecast_tx.seal()
        forecast_signatures = verify_files(forecast_inventory)
        event("cohort_globally_sealed", seal=cohort_seal, native_full_blocks=0, native_scheduler_calls=0, reference_cut_reads=0)
        def native_targets(context):
            for case in CASES:
                pass_case("native", case)
            for case in CASES:
                consumers[case]["native_reference"] = decode_case(case, "native_reference", native_store, rows["native"][case][-1]["state"])
                predicted = np.load(output / consumers[case]["predicted"]["pixels"]["path"], allow_pickle=False)
                actual = np.load(output / consumers[case]["native_reference"]["pixels"]["path"], allow_pickle=False)
                consumers[case]["comparison"] = measure(predicted, actual)
                for key in ("decoded", "decoder_input", "actual_decoder_input"):
                    values = {origin: (np.load(output / consumers[case][origin][key]["path"], allow_pickle=False).astype(np.uint32) << 16).view(np.float32) for origin in ("predicted", "native_reference")}
                    consumers[case][key + "_comparison"] = measure(values["predicted"], values["native_reference"])
            return dict(final_cuts={case: rows["native"][case][-1]["state"]["cut"] for case in CASES}, consumers=consumers)
        forecast_tx.execute("native_targets_and_reference_decoders", native_targets)
        forecast_tx.close()
        verify_files(forecast_inventory)
        verify_files(input_inventory)
        for lane in counts:
            expected = declared["forecast" if lane == "forecast" else lane]
            require(all(counts[lane][key] == value for key, value in expected.items()), "physical primitive census differs: " + lane)
        require(totals["model_loads"] == totals["decoder_loads"] == 1 and totals["reference_cut_reads"] == totals["predicted_decoder_calls"] == totals["native_reference_decoder_calls"] == 4, "model/reference/decoder census differs")
        for name, value in model.named_parameters():
            require((id(value), value.data_ptr(), value._version) == signature[name] and tensor_descriptor(value)["sha256"] == expected_weights[name], "original parameter changed")
        require(decoder_counts["predicted"] == decoder_counts["native_reference"] and decoder_counts["predicted"]["decoder_entries"] == 4, "real decoder census differs")
        for name, value in vae.state_dict(keep_vars=True).items():
            if name in required_vae:
                require((id(value), value.data_ptr(), value._version) == vae_signature[name] and tensor_descriptor(value)["sha256"] == expected_vae[name], "original decoder tensor changed")
        accounts = []
        for number in range(declared["per_case_transitions"]):
            sham_states = {lane: (forecast_store if lane == "forecast" else native_store).load(rows[lane]["sham"][number]["state"]["cut"], device="cpu").payload for lane in ("forecast", "native")}
            panel = {}
            for case in CASES[1:]:
                fields = {}
                states = {lane: (forecast_store if lane == "forecast" else native_store).load(rows[lane][case][number]["state"]["cut"], device="cpu").payload for lane in ("forecast", "native")}
                for key in ("image", "text", "latent", "noise_prediction"):
                    if states["native"][key] is not None:
                        responses = {lane: array(states[lane][key]).astype(np.float64) - array(sham_states[lane][key]).astype(np.float64) for lane in ("forecast", "native")}
                        fields[key] = measure(responses["forecast"], responses["native"])
                        fields[key]["native_signed_mean"] = float(responses["native"].mean())
                        fields[key]["predicted_signed_mean"] = float(responses["forecast"].mean())
                panel[case] = fields
            accounts.append(dict(number=number, before=rows["forecast"]["sham"][number]["before"], responses=panel))
        image_responses = {}
        for case in CASES[1:]:
            values = {}
            for origin in ("predicted", "native_reference"):
                image = np.load(output / consumers[case][origin]["pixels"]["path"], allow_pickle=False).astype(np.float64)
                sham = np.load(output / consumers["sham"][origin]["pixels"]["path"], allow_pickle=False).astype(np.float64)
                values[origin] = image - sham
            image_responses[case] = dict(measure(values["predicted"], values["native_reference"]), predicted_signed_mean=float(values["predicted"].mean()), native_signed_mean=float(values["native_reference"].mean()))
        save(output / "response-accounts.json", dict(orientation="case-minus-sham, forecast-minus-native response error", boundaries=accounts, images=image_responses, ordering_prediction=None))
        first_misses = {}
        for case, panel in comparisons.items():
            first_misses[case] = next((dict(number=row["number"], before=row["before"], fields=[name for name, value in row["fields"].items() if not value["exact"]], primitive_ports=[name for name, value in row["primitive_ports"].items() if not value["exact"]]) for row in panel if any(not value["exact"] for value in row["fields"].values()) or any(not value["exact"] for value in row["primitive_ports"].values())), None)
        save(output / "comparisons.json", dict(boundaries=comparisons, first_covered_mismatch=first_misses, consumers=consumers))
        save(output / "native-contract.json", dict(verification_adapter_identity=adapter.model_identity, verification_adapter_execution=adapter.execution, source_model_identity=config["expected_source_model_identity"], source_execution=source_execution, known_control_occurrence="step2/project already observed by prior eight consumers; recomputed here from archived predicted commits", first_as_yet_unobserved_occurrence="step2/joint0",
            component_source_sha256=component_sources, selected_aten_profiles=selected_profiles,
            original_parameter_count=len(expected_weights), original_parameter_bytes=sum(math_count["source"]["data_offsets"][1] - math_count["source"]["data_offsets"][0] for math_count in manifest["artifacts"]),
            dtype="torch.bfloat16", device=device, source_reduction_policy=config["source_reduction_policy"], tf32=False, inference_mode=True, autocast=False, deterministic=True, actual_attention_backend=str(getattr(backend_name, "value", backend_name)),
            count_contract=declared, forecast_helper_sha256=config["helper_source_sha256"][HELPERS[0]], primitive_scope=graph["PROFILE"], consumers="Eight real source-bound VAE consumers; conditional image forecast, not independent VAE prediction", decoder_counts=decoder_counts, decoder_contract=decoder_contract, parameter_storage="One resident original transformer in memory; recipes only in persistent evidence."))
        for key in ("transformer_config", "scheduler_config", "parameter_manifest", "retrieval_script", "binding", "vae_config", "vae_parameter_manifest"):
            require(sha(config[key]) == config[key + "_sha256"], "source metadata changed")
            shutil.copyfile(config[key], output / Path(config[key]).name)
        for relative, record in config["input_files"].items():
            require(sha(Path(config["input_root"]) / relative) == record["sha256"], "original source changed")
        save(output / "source.json", dict(schema="mdb-functional-remaining-source-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"), worker_sha256=sha(__file__), helper_source_sha256=config["helper_source_sha256"], source_provenance=config.get("source_provenance"), input_records=input_records, reference_records=reference_records, input_seal=input_seal, latent_seal=latent_seal, forecast_seal=cohort_seal, original_parameter_sha256=expected_weights, original_decoder_tensor_sha256=expected_vae, decoder_contract=decoder_contract))
        summary = dict(schema="mdb-functional-remaining-summary-v1", job_id=os.environ.get("MRUN_JOB_ID"), request_sha256=config.get("request_sha256"), counts=counts, decoder_counts=decoder_counts, totals=totals, input_seal=input_seal, latent_seal=latent_seal, forecast_seal=cohort_seal,
            transaction=forecast_tx.report(), verification=verify_transaction(output / "forecast-transaction", expected_seal_sha256=cohort_seal["seal_sha256"]), cases=list(CASES), complete_forecast_cuts=4 * declared["per_case_transitions"], complete_native_cuts=4 * declared["per_case_transitions"], real_decoder_consumers=8,
            first_covered_mismatch=first_misses, seconds=time.monotonic() - start, parameter_payload_reads=bank.payload_reads, parameter_source_bytes=bank.bytes, parameter_cache_peak_bytes=bank.peak_retained, decoder_parameter_payload_reads=vae_bank.payload_reads, decoder_parameter_source_bytes=vae_bank.bytes, decoder_parameter_cache_peak_bytes=vae_bank.peak_retained,
            mechanics_status="complete", instrument_status="full-state and fixed-witness comparisons retained", trend_status="observed-uninterpreted", terminal_status="unassessed", act_receipts=0, session_continue_receipts=0)
        save(output / "summary.json", summary)
        return summary
    finally:
        try:
            for transaction in transactions:
                transaction.close()
        finally:
            for key, value in changed_globals.items():
                g[key] = value
            native.dispatch_attention_fn, native.apply_rotary_emb, FlowMatchEulerDiscreteScheduler.step = original_dispatch, original_rotate, original_scheduler
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
            for handle in handles:
                handle.remove()
            torch.set_float32_matmul_precision(original_flags[7])
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_flags[:5]
            torch.set_deterministic_debug_mode(original_flags[5])
            torch.set_num_threads(original_flags[6])


def worker(config_path):
    print("MDB_FUNCTIONAL_REMAINING_SETUP bootstrap", flush=True)
    faulthandler.dump_traceback_later(90, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "4"
    config = json.loads(Path(config_path).read_text())
    _, m, _ = helpers(config)
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = json.loads(Path(os.environ["SATURN_DEBUG_REQUEST"]).read_text())
        m["require"](request["config"] == config, "request/config differs")
        for name, digest in request["source_hashes"].items():
            m["require"](m["sha"](Path.cwd() / name) == digest, "staged source differs")
        config = dict(config, request_sha256=request["request_sha256"])
    if "output" not in config:
        config["output"] = str(Path(config["output_scratch"]) / os.environ["MRUN_JOB_ID"] / config.get("output_name", "functional-remaining-result"))
    summary = run(config, progress=lambda row: print("MDB_FUNCTIONAL_REMAINING_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    ready = m["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)

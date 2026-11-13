"""Prospective raw-prompt -> FLUX trajectory -> VAE composition.

This worker owns one bounded, source-bound composition assay.  It forms one
conditioning tensor from a raw prompt with the qualified Qwen graph, forecasts
the complete four-step FLUX trajectory with the qualified functional graph,
and forecasts the decoder with the qualified explicit VAE graph.  Only after
the complete forecast transaction is sealed does it run the original Qwen,
FLUX and VAE consumers on independently formed native inputs.

The three original components are hydrated once on CPU and placed on CUDA one
at a time.  This keeps the component identities and nonpersistent buffers
visible while avoiding a co-resident Qwen/transformer/VAE allocation.  The
worker is intentionally a plain script: production launchers provide the
immutable recipe/config paths and request binding; no private Saturn imports
or implicit checkpoint discovery are allowed.
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
from typing import Any

PROMPT = (
    "A cobalt blue cube to the right of a matte orange ring on a gray tabletop. "
    "Two separate objects, simple studio product photograph."
)
SEED = 277
STEPS = 4
HEIGHT = WIDTH = 512
PREFIX_LAYERS = 27
SELECTED_HIDDEN_STATES = (9, 18, 27)
EXECUTION_ID = "mdb-flux-raw-to-image-v1"
QWEN_HELPER = "flux_functional_qwen_prefix.py"
QWEN_WORKER = "flux_functional_qwen_prefix_worker.py"
DENOISER_HELPER = "flux_functional_denoiser.py"
DECODER_HELPER = "flux_functional_decoder.py"
NATIVE_SOURCE_MODULES = {
    "diffusers/pipelines/flux2/pipeline_flux2_klein.py": "diffusers.pipelines.flux2.pipeline_flux2_klein",
    "diffusers/pipelines/flux2/image_processor.py": "diffusers.pipelines.flux2.image_processor",
    "diffusers/models/transformers/transformer_flux2.py": "diffusers.models.transformers.transformer_flux2",
    "diffusers/models/autoencoders/autoencoder_kl_flux2.py": "diffusers.models.autoencoders.autoencoder_kl_flux2",
    "diffusers/schedulers/scheduling_flow_match_euler_discrete.py": "diffusers.schedulers.scheduling_flow_match_euler_discrete",
    "diffusers/utils/torch_utils.py": "diffusers.utils.torch_utils",
}


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path(config: dict[str, Any], *names: str) -> str:
    for name in names:
        value = config.get(name)
        if value:
            return str(value)
    raise ValueError("missing source binding: " + "/".join(names))


def _source_keys() -> tuple[str, ...]:
    return (
        "qwen_config",
        "qwen_manifest",
        "qwen_source_binding",
        "qwen_retrieval_script",
        "qwen_recipe_helper",
        "flux_config",
        "flux_scheduler_config",
        "flux_manifest",
        "flux_retrieval_script",
        "flux_recipe_helper",
        "vae_config",
        "vae_manifest",
        "vae_retrieval_script",
        "vae_recipe_helper",
    )


def _helper_modules(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    here = Path(__file__).parent
    names = (QWEN_WORKER, QWEN_HELPER, DENOISER_HELPER, DECODER_HELPER)
    modules = {name: runpy.run_path(str(here / name)) for name in names}
    expected = config.get("helper_source_sha256", {})
    for name, digest in expected.items():
        path = here / name
        require(path.is_file() and sha(path) == digest, "qualified helper differs: " + name)
    if config.get("worker_sha256"):
        require(sha(__file__) == config["worker_sha256"], "worker source differs")
    return modules


def contract(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the physical and logical work contract before any model load."""

    return {
        "schema": "mdb-flux-raw-to-image-count-contract-v1",
        "prompt_cases": 1,
        "qwen": {
            "functional_module_forwards": 0,
            "forecast_hidden_states": 28,
            "native_calls": 1,
            "native_full_layers_per_call": 36,
            "native_retained_hidden_states": 28,
            "selected_conditioning_states": list(SELECTED_HIDDEN_STATES),
        },
        "denoiser": {
            "forecast_transitions": STEPS * 28,
            "native_transitions": STEPS * 28,
            "forecast_module_forwards": 0,
            "native_scheduler_commits": STEPS,
        },
        "decoder": {
            "functional_calls": 1,
            "native_calls": 2,
            "forecast_module_forwards": 0,
            "native_decoder_operands": 2,
            "whole_image_shared_primitive_forecast": True,
            "independent_decoder_kernel_arithmetic": False,
        },
        "resources": {
            "pipeline_assemblies": 1,
            "component_hydrations": 3,
            "cuda_placements": 6,
            "cpu_offloads": 6,
            "max_gpu_components": 1,
            "persistent_weight_banks": 0,
        },
        "noise_draws": 2,
        "model_loads": 1,
        "execution_id": EXECUTION_ID,
        "predictions": [
            "One complete raw prompt and seed form the entire functional Qwen, FLUX and explicit VAE forecast.",
            "The native lane independently regenerates the same initial noise, IDs and schedule after the global forecast seal.",
            "The functional decoder produces a complete pretarget image through shared Torch primitives; native VAE calls validate the own forecast latent and the separate native-reference final latent. This is not an independent-kernel or standalone RGB forecast claim.",
        ],
    }


def _validate_geometry(config: dict[str, Any]) -> None:
    require(config.get("schema") == "mdb-flux-raw-to-image-plan-v1", "unexpected raw-to-image plan schema")
    require(config.get("prompt") == PROMPT and config.get("seed") == SEED, "raw specimen differs")
    require(
        config.get("batch_size") == 1
        and config.get("height") == HEIGHT
        and config.get("width") == WIDTH
        and config.get("steps") == STEPS,
        "raw composition geometry differs",
    )
    require(config.get("device") == "cuda" and config.get("dtype") == "bfloat16", "production CUDA BF16 profile required")
    require(config.get("guidance_scale", 1.0) == 1.0 and config.get("reference_image", False) is False, "guidance/reference profile differs")
    require(config.get("retry", False) is False and config.get("nested_mrun", False) is False, "finite no-retry lease required")
    require(config.get("flux_mode") == "capture" and config.get("capture_sites") is True, "supported capture workload is required")
    require(config.get("max_gpu_components") == 1, "co-resident learned components are outside the profile")
    require(config.get("component_hydrations") == 3 and config.get("cuda_placements") == 6, "placement census differs")
    require(config.get("offloads") == 6, "offload census differs")
    rope_policy = config.get("rope_buffer_policy", config.get("qwen_rope_buffer_policy", {}))
    require(rope_policy.get("placement_dtype") == "float32", "production FP32 RoPE placement is required")
    require(rope_policy.get("init_device") == "cpu", "Qwen RoPE CPU initialization is required")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    _validate_geometry(config)
    _helper_modules(config)
    for key in _source_keys():
        source = Path(_path(config, key))
        require(source.is_file(), "bound source is not a file: " + key)
        expected = config.get(key + "_sha256")
        require(expected and sha(source) == expected, "source metadata differs: " + key)
    require(config.get("qwen_manifest_entries") == 398, "full Qwen recipe count differs")
    require(config.get("flux_manifest_entries") == 169, "full FLUX recipe count differs")
    require(config.get("vae_manifest_entries") == 142, "decoder recipe count differs")
    native_sources = config.get("native_sources", {})
    require(set(native_sources) == set(NATIVE_SOURCE_MODULES), "native diffusers source binding is incomplete")
    require(0 < int(config.get("max_wall_seconds", 1200)) <= 2400, "finite wall budget differs")
    require(1 <= int(config.get("cpu_threads", 4)) <= 8, "CPU thread bound differs")
    return contract(config)


def _native_source_contract(config: dict[str, Any]) -> dict[str, Any]:
    import importlib
    import inspect

    records = {}
    for relative, module_name in NATIVE_SOURCE_MODULES.items():
        module = importlib.import_module(module_name)
        path = Path(inspect.getfile(module))
        actual = sha(path)
        binding = config["native_sources"][relative]
        expected = binding.get("sha256") if isinstance(binding, dict) else binding
        require(expected == "runtime-bound" or actual == expected, "native diffusers source differs: " + relative)
        records[relative] = {
            "module": module_name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "binding": "runtime-observed" if expected == "runtime-bound" else "staged-source-hash",
        }
    return {"schema": "mdb-flux-native-source-contract-v1", "sources": records}


def _recipe(record: dict[str, Any], name: str) -> dict[str, Any]:
    source = record["source"]
    return {
        "path": record["path"],
        "sha256": record["sha256"],
        "parameter_tensor": name,
        "stored_shape": source["shape"],
        "descriptor": {"sha256": source["payload_sha256"]},
    }


def _load_bank(helper_module: dict[str, Any], manifest_path: str, helper_path: str, root: str):
    manifest = read(manifest_path)
    bank = helper_module["RecipeBank"](manifest, root, max_cache_bytes=1)
    return bank, manifest


def _load_transformer(config: dict[str, Any], helper_module: dict[str, Any]):
    import torch
    from diffusers import Flux2Transformer2DModel

    spec = read(config["flux_config"])
    bank, manifest = _load_bank(helper_module, config["flux_manifest"], config["flux_recipe_helper"], config["flux_parameter_root"])
    with torch.device("meta"):
        model = Flux2Transformer2DModel.from_config(spec)
    expected_names = {row["source"]["tensor"] for row in manifest["artifacts"]}
    require(set(model.state_dict()) == expected_names and len(expected_names) == 169, "FLUX recipe names differ")
    weights = {}
    for row in manifest["artifacts"]:
        name = row["source"]["tensor"]
        require(list(model.state_dict()[name].shape) == row["source"]["shape"] and row["source"]["dtype"] == "BF16", "FLUX tensor geometry differs")
        value = torch.from_numpy(bank.load(_recipe(row, name))).to(torch.bfloat16)
        weights[name] = value
    model.load_state_dict(weights, strict=True, assign=True)
    model.eval().requires_grad_(False)
    del weights, value
    return model, bank, manifest


def _load_vae(config: dict[str, Any], helper_module: dict[str, Any]):
    import torch
    from diffusers import AutoencoderKLFlux2

    spec = read(config["vae_config"])
    bank, manifest = _load_bank(helper_module, config["vae_manifest"], config["vae_recipe_helper"], config["vae_parameter_root"])
    with torch.device("meta"):
        vae = AutoencoderKLFlux2.from_config(spec)
    state = vae.state_dict()
    required = {
        name
        for name in state
        if name.startswith(("decoder.", "post_quant_conv.")) or name in {"bn.running_mean", "bn.running_var"}
    }
    names = {row["source"]["tensor"] for row in manifest["artifacts"]}
    require(names == required and len(names) == 142, "VAE decoder/BN recipe set differs")
    weights = {}
    for row in manifest["artifacts"]:
        name = row["source"]["tensor"]
        require(list(state[name].shape) == row["source"]["shape"] and row["source"]["dtype"] == "BF16", "VAE tensor geometry differs")
        weights[name] = torch.from_numpy(bank.load(_recipe(row, name))).to(torch.bfloat16)
    result = vae.load_state_dict(weights, strict=False, assign=True)
    require(not result.unexpected_keys and set(result.missing_keys) == set(state) - required - {"bn.num_batches_tracked"}, "VAE component boundary differs")
    vae.bn.num_batches_tracked = torch.tensor(0, dtype=torch.int64)
    vae.eval().requires_grad_(False)
    require(not vae.use_tiling and not vae.use_slicing, "tiled/sliced VAE is outside the profile")
    del weights, result, state
    return vae, bank, manifest


def _tensor_signature(value) -> dict[str, Any]:
    import torch

    from mdb_lanes.forward_recording import tensor_descriptor

    require(isinstance(value, torch.Tensor) and value.device.type != "meta", "signature requires materialized tensor")
    descriptor = tensor_descriptor(value)
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "sha256": descriptor["sha256"],
    }


def _component_signature(component, *, prefixes: tuple[str, ...] | None = None) -> dict[str, Any]:
    values = {}
    for name, value in (*component.named_parameters(), *component.named_buffers()):
        if prefixes and not name.startswith(prefixes):
            continue
        if value.device.type == "meta":
            continue
        values[name] = _tensor_signature(value)
    return values


def _assert_signature_equal(before: dict[str, Any], after: dict[str, Any], label: str) -> None:
    require(set(before) == set(after), label + " tensor census changed")
    for name in before:
        require(before[name]["shape"] == after[name]["shape"] and before[name]["dtype"] == after[name]["dtype"] and before[name]["sha256"] == after[name]["sha256"], label + " tensor changed: " + name)


def _save_tensor(output: Path, relative: str, value, *, logical_dtype: str | None = None) -> dict[str, Any]:
    import numpy as np
    import torch

    require(isinstance(value, torch.Tensor) and value.device.type != "meta", "materialized tensor required")
    value_cpu = value.detach().contiguous().cpu()
    if value_cpu.dtype == torch.bfloat16:
        stored = value_cpu.view(torch.uint16).numpy()
        encoding = "bf16-bits-v1"
    else:
        stored = value_cpu.numpy()
        encoding = "native-numpy-v1"
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, stored, allow_pickle=False)
    raw = stored.tobytes(order="C")
    return {
        "path": relative,
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "logical_dtype": logical_dtype or str(value.dtype),
        "stored_dtype": str(stored.dtype),
        "encoding": encoding,
        "device_at_capture": str(value.device),
    }


def _save_bytes(output: Path, relative: str, value) -> dict[str, Any]:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    path = output / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size, "shape": list(array.shape), "dtype": str(array.dtype)}


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


@contextmanager
def _learned_forward_guard(components: list[Any], released: dict[str, bool], counters: dict[str, int]):
    """Reject every learned module forward until the forecast seal releases it."""

    handles = []

    def entered(module, _args):
        if released["native"]:
            counters["native_module_entries"] = counters.get("native_module_entries", 0) + 1
            return
        counters["guard_refusals"] = counters.get("guard_refusals", 0) + 1
        raise ValueError("native learned module.forward entered before global forecast seal: " + type(module).__name__)

    for component in components:
        if component is None:
            continue
        for module in component.modules():
            handles.append(module.register_forward_pre_hook(entered, prepend=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _frontend(config: dict[str, Any], conditioning, device, pipeline_module):
    """Independently form noise, IDs and the shifted native schedule."""

    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED)
    before = generator.get_state().clone()
    latent_height = 2 * (HEIGHT // (8 * 2))
    latent_width = 2 * (WIDTH // (8 * 2))
    channels = int(config.get("flux_in_channels", 128)) // 4
    raw = torch.randn((1, channels * 4, latent_height // 2, latent_width // 2), generator=generator, device=device, dtype=torch.bfloat16)
    after = generator.get_state().clone()
    latent_ids = pipeline_module["Flux2KleinPipeline"]._prepare_latent_ids(raw).to(device)
    packed = pipeline_module["Flux2KleinPipeline"]._pack_latents(raw)
    txt_ids = pipeline_module["Flux2KleinPipeline"]._prepare_text_ids(conditioning).to(device)
    base_sigmas = np.linspace(1.0, 1 / STEPS, STEPS)
    mu = compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=STEPS)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
    scheduler.set_timesteps(num_inference_steps=STEPS, device=device, sigmas=base_sigmas, mu=mu)
    timesteps = scheduler.timesteps.detach().clone()
    sigmas = scheduler.sigmas.detach().clone()
    require(timesteps.dtype == torch.float32 and sigmas.dtype == torch.float32, "schedule dtype differs")
    return {
        "raw_noise": raw,
        "packed_noise": packed,
        "img_ids": latent_ids,
        "txt_ids": txt_ids,
        "timesteps": timesteps,
        "sigmas": sigmas,
        "mu": float(mu),
        "generator_before": before,
        "generator_after": after,
        "base_sigmas": base_sigmas.tolist(),
        "scheduler_config": read(config["flux_scheduler_config"]),
    }


def _frontend_record(output: Path, lane: str, frontend: dict[str, Any]) -> dict[str, Any]:
    base = "frontend/" + lane
    records = {
        "raw_noise": _save_tensor(output, base + "/raw-noise.npy", frontend["raw_noise"]),
        "packed_noise": _save_tensor(output, base + "/packed-noise.npy", frontend["packed_noise"]),
        "img_ids": _save_tensor(output, base + "/img-ids.npy", frontend["img_ids"], logical_dtype="torch.int64"),
        "txt_ids": _save_tensor(output, base + "/txt-ids.npy", frontend["txt_ids"], logical_dtype="torch.int64"),
        "timesteps": _save_tensor(output, base + "/timesteps.npy", frontend["timesteps"], logical_dtype="torch.float32"),
        "sigmas": _save_tensor(output, base + "/sigmas.npy", frontend["sigmas"], logical_dtype="torch.float32"),
        "generator_before": _save_bytes(output, base + "/generator-before.npy", frontend["generator_before"]),
        "generator_after": _save_bytes(output, base + "/generator-after.npy", frontend["generator_after"]),
    }
    records.update({"lane": lane, "seed": SEED, "mu": frontend["mu"], "base_sigmas": frontend["base_sigmas"], "scheduler_config": frontend["scheduler_config"]})
    save(output / (base + ".json"), records)
    return records


def _initial_state(adapter, frontend: dict[str, Any], conditioning):
    session = adapter.session(
        latent=frontend["packed_noise"],
        conditioning=conditioning,
        img_ids=frontend["img_ids"],
        txt_ids=frontend["txt_ids"],
        timesteps=frontend["timesteps"],
        sigmas=frontend["sigmas"],
    )
    return session.capture(retain=False).payload


def _save_cut(output: Path, store, adapter, state: dict[str, Any], *, lane: str, number: int, parent: str | None) -> dict[str, Any]:
    from saturn_pub.core import StateCut

    execution = {
        **dict(adapter.execution),
        "composition": EXECUTION_ID,
        "lane": lane,
        "transition": number,
        "receipt_kind": "component-derived StateCut, not Session/Act receipt",
    }
    cut = StateCut(
        adapter.model_identity,
        execution,
        adapter.boundary(state),
        parent,
        state,
        adapter.point(state).to_dict(),
        adapter.surface(state).to_dict(),
    )
    store.save(cut)
    return {
        "number": number,
        "cut": cut.fingerprint,
        "parent": parent,
        "boundary": cut.boundary,
        "execution_point": dict(cut.execution_point or {}),
        "state": {key: _tensor_signature(value) if hasattr(value, "device") and value.device.type != "meta" else value for key, value in state.items() if key in {"latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"}},
        "clock": {key: state[key] for key in ("step", "phase", "cursor")},
        "reset": all(state[key] is None for key in ("image", "text", "context", "noise_prediction")) if state["phase"] == "project" else None,
    }


def _trajectory(output: Path, model, adapter, frontend: dict[str, Any], conditioning, *, lane: str, graph: dict[str, Any], store_name: str, counters: dict[str, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from saturn_pub.store import LocalStore

    store = LocalStore(output / store_name)
    state = _initial_state(adapter, frontend, conditioning)
    initial = _save_cut(output, store, adapter, state, lane=lane, number=-1, parent=None)
    rows = [initial]
    parent = initial["cut"]

    def boundary(event, after):
        nonlocal parent
        number = len(rows) - 1
        row = _save_cut(output, store, adapter, after, lane=lane, number=number, parent=parent)
        row["event"] = dict(event)
        rows.append(row)
        parent = row["cut"]

    if lane == "forecast":
        final = graph["functional_trajectory"](model, state, on_boundary=boundary, max_transitions=STEPS * 28)
        counters["forecast_transitions"] = len(rows) - 1
        counters["functional_transitions"] = len(rows) - 1
    else:
        final = state
        for number in range(STEPS * 28):
            before = {key: final[key] for key in ("step", "phase", "cursor")}
            with torch.inference_mode():
                final = adapter.advance(final)
            counters["native_transitions"] = counters.get("native_transitions", 0) + 1
            row = _save_cut(output, store, adapter, final, lane=lane, number=number, parent=parent)
            row["event"] = {"step": before["step"], "phase": before["phase"], "index": before["cursor"], "next_step": final["step"], "next_phase": final["phase"], "next_index": final["cursor"]}
            rows.append(row)
            parent = row["cut"]
    require((final["step"], final["phase"], final["cursor"]) == (STEPS, "project", 0), lane + " trajectory clock did not close")
    require(all(final[key] is None for key in ("image", "text", "context", "noise_prediction")), lane + " final working fields did not reset")
    save(output / (store_name.replace("-store", "-rows.json")), rows)
    return final, rows


def _decoder_input(vae, state: dict[str, Any], config: dict[str, Any], pipeline_module):
    """Apply the qualified CPU ID scatter/BN inverse/unpatch boundary."""

    import torch

    pipeline = pipeline_module["Flux2KleinPipeline"]
    scale = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_cpu = state["latent"].detach().cpu()
    img_ids_cpu = state["img_ids"].detach().cpu()
    unpacked = pipeline._unpack_latents_with_ids(
        latent_cpu, img_ids_cpu, HEIGHT // (scale * 2), WIDTH // (scale * 2)
    )
    mean = vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(unpacked)
    std = (vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(unpacked)
    decoder_input = pipeline._unpatchify_latents(unpacked * std + mean)
    require(decoder_input.dtype == torch.bfloat16 and decoder_input.device.type == "cpu", "decoder CPU upstream profile differs")
    require(tuple(decoder_input.shape) == (1, 32, 64, 64), "decoder input geometry differs")
    return decoder_input


def _postprocess_pixels(image_processor, decoded):
    import numpy as np

    images = image_processor.postprocess(decoded, output_type="pil")
    pixels = np.asarray(images[0])
    require(pixels.dtype == np.uint8 and pixels.shape == (HEIGHT, WIDTH, 3), "uint8 image geometry differs")
    return pixels, images[0]


def _functional_decoder(output: Path, vae, state: dict[str, Any], config: dict[str, Any], graph: dict[str, Any], pipeline_module, image_processor, counters: dict[str, int]) -> dict[str, Any]:
    import torch

    decoder_input_cpu = _decoder_input(vae, state, config, pipeline_module)
    decoder_input = decoder_input_cpu.to(next(vae.decoder.parameters()).device)
    macros: dict[str, dict[str, Any]] = {}
    selected = {
        "post_quant_conv",
        "decoder.conv_in",
        "decoder.mid_block",
        "decoder.conv_norm_out",
        "decoder.conv_act",
        "decoder.conv_out",
        "decoder",
    }
    selected.update(f"decoder.up_blocks.{index}" for index in range(len(vae.decoder.up_blocks)))

    def observe(name, _kind, _inputs, value):
        if name in selected and name not in macros:
            ordinal = len(macros)
            macros[name] = _save_tensor(output, f"decoder/forecast/macros/{ordinal:02d}-{_safe_name(name)}.npy", value)

    with torch.inference_mode():
        decoded = graph["functional_decode"](vae, decoder_input, observe=observe, counts={}, profile="original")
        decoded_record = _save_tensor(output, "decoder/forecast/decoded.npy", decoded)
        pixels, image = _postprocess_pixels(image_processor, decoded)
    pixel_record = _save_bytes(output, "decoder/forecast/pixels.npy", pixels)
    image_path = output / "decoder/forecast/image.png"
    image.save(image_path)
    require(len(macros) == len(selected), "functional decoder macro census incomplete")
    counters["functional_decode_calls"] = counters.get("functional_decode_calls", 0) + 1
    return {
        "origin": "forecast",
        "decoder_input": _save_tensor(output, "decoder/forecast/decoder-input.npy", decoder_input_cpu),
        "macros": macros,
        "decoded": decoded_record,
        "pixels": pixel_record,
        "png": {"path": str(image_path.relative_to(output)), "sha256": sha(image_path), "bytes": image_path.stat().st_size},
        "macro_contract": {
            "module_macros": len(macros),
            "decoder_input_boundary": 1,
            "total_saved_boundaries": len(macros) + 1,
        },
        "independent_decoder_prediction": True,
        "native_module_forwards": 0,
    }


def _native_decoder(output: Path, vae, state: dict[str, Any], config: dict[str, Any], origin: str, pipeline_module, image_processor, counters: dict[str, int], prehook_records: dict[str, Any]) -> dict[str, Any]:
    import torch

    decoder_input_cpu = _decoder_input(vae, state, config, pipeline_module)
    decoder_input = decoder_input_cpu.to(next(vae.decoder.parameters()).device)
    key = "native-own" if origin == "own_forecast" else "native-reference"
    prehook_records[key] = {"decoder_input": _save_tensor(output, f"decoder/{key}/decoder-input.npy", decoder_input_cpu)}

    def decoder_entry(_module, args):
        operand = args[0]
        prehook_records[key]["actual_after_post_quant"] = _save_tensor(output, f"decoder/{key}/actual-after-post-quant.npy", operand)

    def post_quant_entry(_module, args):
        operand = args[0]
        prehook_records[key]["actual_before_post_quant"] = _save_tensor(output, f"decoder/{key}/actual-before-post-quant.npy", operand)

    input_handle = vae.post_quant_conv.register_forward_pre_hook(post_quant_entry, prepend=True)
    handle = vae.decoder.register_forward_pre_hook(decoder_entry, prepend=True)
    try:
        with torch.inference_mode():
            decoded = vae.decode(decoder_input, return_dict=False)[0]
            decoded_record = _save_tensor(output, f"decoder/{key}/decoded.npy", decoded)
            pixels, image = _postprocess_pixels(image_processor, decoded)
    finally:
        input_handle.remove()
        handle.remove()
    pixel_record = _save_bytes(output, f"decoder/{key}/pixels.npy", pixels)
    image_path = output / f"decoder/{key}/image.png"
    image.save(image_path)
    counters["native_vae_calls"] = counters.get("native_vae_calls", 0) + 1
    require("actual_before_post_quant" in prehook_records[key] and "actual_after_post_quant" in prehook_records[key], "native decoder operands were not captured")
    require(prehook_records[key]["actual_before_post_quant"]["content_sha256"] == prehook_records[key]["decoder_input"]["content_sha256"], "native decoder input custody differs")
    return {
        "origin": origin,
        "decoder_input": prehook_records[key]["decoder_input"],
        "actual_before_post_quant": prehook_records[key]["actual_before_post_quant"],
        "actual_after_post_quant": prehook_records[key]["actual_after_post_quant"],
        "operand_boundary": "after_post_quant_conv_before_decoder",
        "decoded": decoded_record,
        "pixels": pixel_record,
        "png": {"path": str(image_path.relative_to(output)), "sha256": sha(image_path), "bytes": image_path.stat().st_size},
        "independent_decoder_prediction": False,
        "native_module_forwards": True,
    }


def _inventory(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "files.json":
            continue
        relative = path.relative_to(root).as_posix()
        if any(relative == prefix or relative.startswith(prefix.rstrip("/") + "/") for prefix in exclude):
            continue
        records[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return records


def _verify_inventory(root: Path, records: dict[str, dict[str, Any]]) -> None:
    for relative, record in records.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "sealed evidence path changed: " + relative)
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "sealed evidence changed: " + relative)


def run(config: dict[str, Any], *, progress=None) -> dict[str, Any]:
    declared = validate_config(config)
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    modules = _helper_modules(config)
    qworker = modules[QWEN_WORKER]
    qwen_graph = modules[QWEN_HELPER]
    denoiser = modules[DENOISER_HELPER]
    decoder_graph = modules[DECODER_HELPER]
    output = Path(config["output"]).resolve()
    forbidden_roots = [Path(_path(config, key)).resolve().parent for key in _source_keys()]
    require(all(not output.is_relative_to(root) for root in forbidden_roots), "output overlaps immutable source")
    output.mkdir(parents=True, exist_ok=False)
    effective_plan = {key: value for key, value in config.items() if key != "_request_record"}
    effective_plan["request_sha256"] = config.get("request_sha256")
    effective_plan["job_id"] = os.environ.get("MRUN_JOB_ID")
    save(output / "plan.json", effective_plan)
    if config.get("_request_record") is not None:
        save(output / "request.json", config["_request_record"])
    start = time.monotonic()
    events: list[dict[str, Any]] = []
    counters = {
        "pipeline_assemblies": 1,
        "component_hydrations": 0,
        "cuda_placements": 0,
        "cpu_offloads": 0,
        "noise_draws": 0,
        "functional_transitions": 0,
        "native_transitions": 0,
        "functional_decode_calls": 0,
        "native_vae_calls": 0,
        "native_qwen_calls": 0,
        "native_module_entries": 0,
        "guard_refusals": 0,
    }
    released = {"native": False}
    transactions = []
    forecast_inventory: dict[str, dict[str, Any]] = {}
    placement_records: list[dict[str, Any]] = []

    def event(kind: str, **detail: Any) -> None:
        row = {"kind": kind, "seconds": time.monotonic() - start, **detail}
        require(row["seconds"] <= int(config.get("max_wall_seconds", 1200)), "finite raw-to-image wall budget exceeded")
        events.append(row)
        with (output / "timeline.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        if progress is not None:
            progress(row)

    def placement(component, name: str, *, to_cuda: bool, vae=False) -> None:
        before = _component_signature(component, prefixes=("post_quant_conv.", "decoder.", "bn.") if vae else None)
        if vae:
            target = "cuda" if to_cuda else "cpu"
            component.post_quant_conv.to(target)
            component.decoder.to(target)
            require(component.bn.running_mean.device.type == "cpu" and component.bn.running_var.device.type == "cpu", "VAE BN left CPU profile")
        else:
            component.to("cuda" if to_cuda else "cpu")
        after = _component_signature(component, prefixes=("post_quant_conv.", "decoder.", "bn.") if vae else None)
        _assert_signature_equal(before, after, name)
        placement_records.append({
            "component": name,
            "direction": "cuda" if to_cuda else "cpu",
            "tensor_count": len(after),
            "before_sha256": hashlib.sha256(canonical(before)).hexdigest(),
            "after_sha256": hashlib.sha256(canonical(after)).hexdigest(),
            "before_signature": before,
            "after_signature": after,
            "identity_preserved": True,
        })
        counters["cuda_placements" if to_cuda else "cpu_offloads"] += 1
        event("component_placement", component=name, direction="cuda" if to_cuda else "cpu", tensors=len(after), identity_preserved=True,
              before_sha256=placement_records[-1]["before_sha256"], after_sha256=placement_records[-1]["after_sha256"])

    try:
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(config.get("source_reduction_policy", True))
        require(torch.cuda.is_available(), "CUDA is required for the production raw-to-image assay")
        device = torch.device("cuda")
        rope_policy = config.get("rope_buffer_policy", config.get("qwen_rope_buffer_policy"))
        config = dict(config, rope_buffer_policy=rope_policy)
        runtime_contract = qworker["_runtime_contract"](config, torch)
        native_source_contract = _native_source_contract(config)
        runtime_contract["native_diffusers_sources"] = native_source_contract["sources"]
        save(output / "runtime-contract.json", runtime_contract)
        runtime_contract_sha256 = sha(output / "runtime-contract.json")
        event("runtime_contract_bound", runtime_contract_sha256=runtime_contract_sha256)

        qwen_binding = read(config["qwen_source_binding"])
        source_copies = {}
        for key in _source_keys():
            source = Path(_path(config, key))
            target = output / "source" / key / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            source_copies[key] = {"path": str(target.relative_to(output)), "sha256": sha(target), "bytes": target.stat().st_size}
        tokenizer_records = qworker["_copy_tokenizer_sources"](output, qwen_binding)
        tokenizer_root = qwen_binding["checkpoint"]["model_root"] + "/tokenizer"
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True, padding_side="right")
        metadata = read(output / "source" / "tokenizer" / "tokenizer_config.json")
        require(bool(getattr(tokenizer, "is_fast", False)) and type(tokenizer).__name__ == metadata.get("tokenizer_class"), "pinned tokenizer runtime differs")
        event("tokenizer_bound", tokenizer_class=type(tokenizer).__name__, source_files=len(tokenizer_records))

        qwen_bank, qwen_manifest = qworker["_load_recipe_bank"](
            Path(config["qwen_recipe_helper"]), Path(config["qwen_manifest"]), config["qwen_parameter_root"]
        )
        qwen_model = qworker["_load_qwen"](config, torch.device("cpu"), qwen_bank, qwen_manifest)
        counters["component_hydrations"] += 1
        qwen_identity_before = _component_signature(qwen_model)
        qwen_alias_before = qwen_model.lm_head.weight is qwen_model.model.embed_tokens.weight
        require(qwen_alias_before, "Qwen tied embedding/lm_head alias missing")
        event("qwen_hydrated_cpu", recipes=len(qwen_manifest["artifacts"]), tied_alias=qwen_alias_before)

        flux_helper = runpy.run_path(config["flux_recipe_helper"])
        flux_model, flux_bank, flux_manifest = _load_transformer(config, flux_helper)
        counters["component_hydrations"] += 1
        flux_identity_before = _component_signature(flux_model)
        event("flux_hydrated_cpu", recipes=len(flux_manifest["artifacts"]))
        vae_helper = runpy.run_path(config["vae_recipe_helper"])
        vae_model, vae_bank, vae_manifest = _load_vae(config, vae_helper)
        counters["component_hydrations"] += 1
        vae_identity_before = _component_signature(vae_model, prefixes=("post_quant_conv.", "decoder.", "bn."))
        event("vae_hydrated_cpu", recipes=len(vae_manifest["artifacts"]), encoder="meta")

        # Guard every learned component before any forecast.  The guards remain
        # installed while components move between CPU and CUDA, so a forbidden
        # module.forward cannot be hidden by a phase boundary.
        with _learned_forward_guard([qwen_model, flux_model, vae_model], released, counters):
            formatted, ids_cpu, mask_cpu = qworker["_tokenize"](tokenizer, PROMPT, config)
            placement(qwen_model, "qwen", to_cuda=True)
            qwen_parameters = {"model.embed_tokens.weight": qwen_model.model.embed_tokens.weight}
            for layer in range(PREFIX_LAYERS):
                qwen_parameters.update({
                    f"model.layers.{layer}.{suffix}": qwen_model.state_dict()[f"model.layers.{layer}.{suffix}"]
                    for suffix in qwen_graph["QWEN_LAYER_PARAMETER_SUFFIXES"]
                })
            ids_cuda, mask_cuda = ids_cpu.to(device), mask_cpu.to(device)
            bound = qwen_graph["bind_prospective_tokenized_input"](
                ids_cuda,
                mask_cuda,
                tokenizer_binding={
                    "cohort": "prospective-new",
                    "parser_stage": "tokenizer",
                    "implementation_pinned": True,
                    "tokenizer_class": type(tokenizer).__name__,
                    "metadata_tokenizer_class": metadata.get("tokenizer_class"),
                    "tokenizer_is_fast": bool(tokenizer.is_fast),
                    "template_files": {key: value["sha256"] for key, value in tokenizer_records.items()},
                    "chat_template_call": config.get("apply_chat_template", {}),
                },
            )
            with qwen_graph["module_entry_guard"](qwen_model) as entries:
                qwen_forecast = qwen_graph["functional_qwen_prefix"](
                    bound["input_ids"],
                    bound["attention_mask"],
                    qwen_parameters,
                    qwen_model.config,
                    attention_dispatch=config["attention_dispatch"],
                    rope_buffer_policy=rope_policy,
                    prefix_layers=PREFIX_LAYERS,
                    selected_hidden_states=SELECTED_HIDDEN_STATES,
                )
            require(entries == [], "Qwen forecast entered native module")
            forecast_qwen = {
                "prompt": PROMPT,
                "formatted_chat": formatted,
                "input_ids": _save_tensor(output, "qwen/forecast/input-ids.npy", ids_cuda, logical_dtype="torch.int64"),
                "attention_mask": _save_tensor(output, "qwen/forecast/attention-mask.npy", mask_cuda, logical_dtype="torch.int64"),
                "position_ids": _save_tensor(output, "qwen/forecast/position-ids.npy", qwen_forecast["position_ids"], logical_dtype="torch.int64"),
                "hidden_states": [],
                "conditioning": None,
                "profile": qwen_forecast["profile"],
                "native_module_forwards": 0,
            }
            for index in range(PREFIX_LAYERS + 1):
                forecast_qwen["hidden_states"].append(_save_tensor(output, f"qwen/forecast/hidden-{index:02d}.npy", qwen_forecast["hidden_states"][index]))
            forecast_qwen["conditioning"] = _save_tensor(output, "qwen/forecast/conditioning.npy", qwen_forecast["conditioning"])
            forecast_E_cpu = qwen_forecast["conditioning"].detach().cpu()
            event("qwen_forecast_complete", hidden_states=28, conditioning_shape=list(forecast_E_cpu.shape), native_forwards=0)
            placement(qwen_model, "qwen", to_cuda=False)
            del qwen_parameters, qwen_forecast, ids_cuda, mask_cuda
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            placement(flux_model, "flux", to_cuda=True)
            denoiser["validate_model"](flux_model)
            from diffusers import FlowMatchEulerDiscreteScheduler

            scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
            adapter = Flux2KleinAdapter(flux_model, scheduler, granularity="operation", device="cuda")
            frontend_forecast = _frontend(config, forecast_E_cpu.to(device), device, {"Flux2KleinPipeline": Flux2KleinPipeline})
            counters["noise_draws"] += 1
            frontend_forecast_record = _frontend_record(output, "forecast", frontend_forecast)
            final_forecast_state, forecast_rows = _trajectory(output, flux_model, adapter, frontend_forecast, forecast_E_cpu.to(device), lane="forecast", graph=denoiser, store_name="forecast-state-store", counters=counters)
            event("flux_forecast_complete", transitions=len(forecast_rows) - 1)
            placement(flux_model, "flux", to_cuda=False)
            del adapter, scheduler
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            placement(vae_model, "vae", to_cuda=True, vae=True)
            vae_scale = 2 ** (len(vae_model.config.block_out_channels) - 1)
            image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale * 2)
            forecast_decoder = _functional_decoder(output, vae_model, final_forecast_state, config, decoder_graph, {"Flux2KleinPipeline": Flux2KleinPipeline}, image_processor, counters)
            event("decoder_forecast_complete", functional_decode_calls=counters["functional_decode_calls"], macros=len(forecast_decoder["macros"]))
            placement(vae_model, "vae", to_cuda=False, vae=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            save(output / "forecast-qwen.json", forecast_qwen)
            save(output / "frontend-forecast.json", frontend_forecast_record)
            save(output / "decoder-forecast.json", forecast_decoder)
            source_inventory = _inventory(output, exclude=("forecast-transaction", "timeline.jsonl", "native-state-store", "native-state-rows.json", "native-qwen", "decoder/native-own", "decoder/native-reference"))
            forecast_tx = ForecastTransaction(
                output / "forecast-transaction",
                model_identity="flux2-klein-4b-raw-to-image-original",
                input_identity=hashlib.sha256(canonical({"prompt": PROMPT, "seed": SEED, "conditioning": forecast_qwen["conditioning"], "frontend": frontend_forecast_record})).hexdigest(),
                source_identity=hashlib.sha256(canonical({key: config[key + "_sha256"] for key in _source_keys()})).hexdigest(),
                frontier={"kind": "raw-input", "composition": "raw-prompt-complete-composition", "source_inventory": source_inventory},
                execution_scope="All functional Qwen, FLUX denoiser and explicit decoder work precedes native learned-model targets; native lane consumes independently formed state.",
                budgets={"max_predictions": 1, "max_executions": 1, "max_native_calls": 4, "max_bytes": int(config.get("forecast_transaction_bytes", 4 << 30)), "max_files": 20000, "max_wall_ms": int(config.get("max_wall_seconds", 1200)) * 1000},
            )
            transactions.append(forecast_tx)
            forecast_tx.predict(
                "raw-composition-forecast",
                lambda _context: {
                    "predictions": {
                        "prompt": PROMPT,
                        "seed": SEED,
                        "qwen": forecast_qwen,
                        "frontend": frontend_forecast_record,
                        "denoiser": {"rows": "forecast-state-rows.json", "transitions": len(forecast_rows) - 1},
                        "decoder": forecast_decoder,
                        "native_targets_at_seal": 0,
                        "external_files": source_inventory,
                    },
                },
            )
            seal = forecast_tx.seal()
            forecast_inventory = _inventory(output, exclude=("forecast-transaction", "timeline.jsonl", "native-state-store", "native-state-rows.json", "native-qwen", "decoder/native-own", "decoder/native-reference", "native-contract.json", "comparisons.json", "summary.json", "source.json", "files.json"))
            event("forecast_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets=0)
            native_result: dict[str, Any] = {}

            def execute_native(_context):
                """Run all learned native targets inside the sealed transaction window."""

                released["native"] = True
                # Native Qwen target: independently placed original model and
                # the same frozen parser bytes; no forecast hidden state is fed
                # into this call.
                placement(qwen_model, "qwen", to_cuda=True)
                ids_native, mask_native = ids_cpu.to(device), mask_cpu.to(device)
                with torch.inference_mode():
                    native_output = qwen_model(input_ids=ids_native, attention_mask=mask_native, output_hidden_states=True, use_cache=False)
                counters["native_qwen_calls"] += 1
                require(len(native_output.hidden_states) - 1 == int(config.get("qwen_full_layers", 36)), "native Qwen hidden-state count differs")
                native_qwen = {
                    "prompt": PROMPT,
                    "input_ids": _save_tensor(output, "qwen/native/input-ids.npy", ids_native, logical_dtype="torch.int64"),
                    "attention_mask": _save_tensor(output, "qwen/native/attention-mask.npy", mask_native, logical_dtype="torch.int64"),
                    "hidden_states": [],
                    "executed_full_layers": len(native_output.hidden_states) - 1,
                    "native_module_forwards": 1,
                }
                for index in range(PREFIX_LAYERS + 1):
                    native_qwen["hidden_states"].append(_save_tensor(output, f"qwen/native/hidden-{index:02d}.npy", native_output.hidden_states[index]))
                native_E_cpu = torch.cat([native_output.hidden_states[index] for index in SELECTED_HIDDEN_STATES], dim=-1).detach().cpu()
                native_qwen["conditioning"] = _save_tensor(output, "qwen/native/conditioning.npy", native_E_cpu)
                native_qwen["last_hidden_state"] = _save_tensor(output, "qwen/native/last-hidden-state.npy", native_output.hidden_states[-1])
                native_qwen["conditioning_matches_forecast"] = torch.equal(forecast_E_cpu, native_E_cpu)
                event("qwen_native_complete", executed_full_layers=native_qwen["executed_full_layers"], retained_hidden_states=28)
                placement(qwen_model, "qwen", to_cuda=False)
                del native_output, ids_native, mask_native
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                placement(flux_model, "flux", to_cuda=True)
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
                adapter = Flux2KleinAdapter(flux_model, scheduler, granularity="operation", device="cuda")
                frontend_native = _frontend(config, native_E_cpu.to(device), device, {"Flux2KleinPipeline": Flux2KleinPipeline})
                counters["noise_draws"] += 1
                frontend_native_record = _frontend_record(output, "native", frontend_native)
                for key in ("raw_noise", "packed_noise", "img_ids", "txt_ids", "timesteps", "sigmas"):
                    require(torch.equal(frontend_forecast[key].detach().cpu(), frontend_native[key].detach().cpu()), "independent frontend differs: " + key)
                final_native_state, native_rows = _trajectory(output, flux_model, adapter, frontend_native, native_E_cpu.to(device), lane="native", graph=denoiser, store_name="native-state-store", counters=counters)
                event("flux_native_complete", transitions=len(native_rows) - 1)
                placement(flux_model, "flux", to_cuda=False)
                del adapter, scheduler
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                placement(vae_model, "vae", to_cuda=True, vae=True)
                prehook_records: dict[str, Any] = {}
                native_decoder_own = _native_decoder(output, vae_model, final_forecast_state, config, "own_forecast", {"Flux2KleinPipeline": Flux2KleinPipeline}, image_processor, counters, prehook_records)
                native_decoder_reference = _native_decoder(output, vae_model, final_native_state, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, image_processor, counters, prehook_records)
                placement(vae_model, "vae", to_cuda=False, vae=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                post_quant_forecast = forecast_decoder["macros"]["post_quant_conv"]
                post_quant_native = native_decoder_own["actual_after_post_quant"]
                post_quant_exact_join = post_quant_native["content_sha256"] == post_quant_forecast["content_sha256"]
                event("native_consumers_complete", native_vae_calls=counters["native_vae_calls"], post_quant_exact_join=post_quant_exact_join)

                save(output / "native-qwen.json", native_qwen)
                save(output / "frontend-native.json", frontend_native_record)
                save(output / "decoder-native-own.json", native_decoder_own)
                save(output / "decoder-native-reference.json", native_decoder_reference)
                native_result.update({
                    "native_qwen": native_qwen,
                    "frontend_native_record": frontend_native_record,
                    "native_state_rows": len(native_rows) - 1,
                    "native_decoder_own": native_decoder_own,
                    "native_decoder_reference": native_decoder_reference,
                    "post_quant_exact_join": post_quant_exact_join,
                })
                return native_result

            forecast_tx.execute("native_qwen_flux_and_vae_targets", execute_native)
            forecast_tx.close()
            native_qwen = native_result["native_qwen"]

        qwen_identity_after = _component_signature(qwen_model)
        flux_identity_after = _component_signature(flux_model)
        vae_identity_after = _component_signature(vae_model, prefixes=("post_quant_conv.", "decoder.", "bn."))
        _assert_signature_equal(qwen_identity_before, qwen_identity_after, "Qwen")
        _assert_signature_equal(flux_identity_before, flux_identity_after, "FLUX")
        _assert_signature_equal(vae_identity_before, vae_identity_after, "VAE")
        require((qwen_model.lm_head.weight is qwen_model.model.embed_tokens.weight) == qwen_alias_before, "Qwen tied alias changed")

        _verify_inventory(output, forecast_inventory)
        save(output / "native-contract.json", {"forecast_seal": seal, "counts": counters, "component_hydrations": 3, "cuda_placements": 6, "cpu_offloads": 6, "placement_records": placement_records, "qwen_identity_before": qwen_identity_before, "qwen_identity_after": qwen_identity_after, "flux_identity_before": flux_identity_before, "flux_identity_after": flux_identity_after, "vae_identity_before": vae_identity_before, "vae_identity_after": vae_identity_after, "qwen_tied_alias": qwen_alias_before, "native_module_forwards_before_seal": 0, "whole_image_shared_primitive_forecast": True, "independent_decoder_kernel_arithmetic": False})
        save(output / "comparisons.json", {"conditioning": {"forecast": forecast_qwen["conditioning"], "native": native_qwen["conditioning"], "exact_equal": native_qwen["conditioning_matches_forecast"], "status": "diagnostic-only"}, "frontend": {"raw_noise_equal": True, "packed_noise_equal": True, "img_ids_equal": True, "txt_ids_equal": True, "timesteps_equal": True, "sigmas_equal": True}, "decoder": {"native_input_custody": {"own": "exact-module-entry-join", "reference": "exact-module-entry-join"}, "forecast_vs_native_own": {"pixels": "diagnostic-only", "decoded": "diagnostic-only", "post_quant_operand": "diagnostic-only", "post_quant_exact_equal": native_result["post_quant_exact_join"]}, "native_own_vs_native_reference": {"pixels": "diagnostic-only", "decoded": "diagnostic-only"}}, "terminal_status": "unassessed", "whole_image_shared_primitive_forecast": True, "independent_rgb_forecast_claim": False})
        source = {"schema": "mdb-flux-raw-to-image-source-v1", "job_id": os.environ.get("MRUN_JOB_ID"), "request_sha256": config.get("request_sha256"), "worker_sha256": sha(__file__), "helper_source_sha256": config.get("helper_source_sha256", {}), "source_files": source_copies, "tokenizer_files": tokenizer_records, "runtime_contract_sha256": runtime_contract_sha256, "forecast_seal": seal, "qwen_parameter_identity_before": qwen_identity_before, "qwen_parameter_identity_after": qwen_identity_after, "flux_parameter_identity_before": flux_identity_before, "flux_parameter_identity_after": flux_identity_after, "vae_parameter_identity_before": vae_identity_before, "vae_parameter_identity_after": vae_identity_after, "qwen_tied_alias": qwen_alias_before, "placement_records": placement_records, "physical_counts": counters}
        save(output / "source.json", source)
        summary = {"schema": "mdb-flux-raw-to-image-summary-v1", "job_id": os.environ.get("MRUN_JOB_ID"), "request_sha256": config.get("request_sha256"), "execution_id": EXECUTION_ID, "prompt": PROMPT, "seed": SEED, "counts": counters, "contract": declared, "forecast_seal": seal, "transaction": verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]), "mechanics_status": "complete", "instrument_status": "full Qwen/denoiser/decoder arrays and native consumer joins retained", "trend_status": "observed-uninterpreted", "terminal_status": "unassessed", "whole_image_shared_primitive_forecast": True, "independent_rgb_forecast_claim": False, "seconds": time.monotonic() - start}
        save(output / "summary.json", summary)
        return summary
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def worker(config_path: str | Path) -> None:
    import faulthandler

    bootstrap = time.monotonic()
    faulthandler.dump_traceback_later(120, repeat=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path)
    request = None
    if "SATURN_DEBUG_REQUEST" in os.environ:
        request = read(os.environ["SATURN_DEBUG_REQUEST"])
        request_identity = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(request_identity == request.get("request_sha256"), "request identity differs")
        require(request.get("config") == config, "request/config differs")
        for name, expected in request.get("source_hashes", {}).items():
            staged = Path.cwd() / name
            if not staged.is_file():
                staged = Path(config.get(name, name))
            require(staged.is_file() and sha(staged) == expected, "staged source differs: " + name)
        config = dict(config, request_sha256=request["request_sha256"], _request_record=request)
    if "output" not in config:
        config = dict(config, output=str(Path(config["output_scratch"]) / os.environ.get("MRUN_JOB_ID", "local") / config.get("output_name", "raw-to-image-result")))
    summary = run(config, progress=lambda row: print("MDB_RAW_TO_IMAGE_PHASE " + json.dumps(row, sort_keys=True), flush=True))
    finalizer = runpy.run_path(str(Path(__file__).with_name("flux_image_tail_microscope.py")))
    ready = finalizer["finalize"](config["output"], job_id=summary["job_id"], request_sha256=summary["request_sha256"], **config.get("transport_limits", {}))
    ready["worker_seconds_including_bootstrap"] = time.monotonic() - bootstrap
    print("MDB_DIRECTORY_READY " + json.dumps(ready, sort_keys=True), flush=True)
    faulthandler.cancel_dump_traceback_later()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    args = parser.parse_args()
    worker(args.config)


if __name__ == "__main__":
    main()

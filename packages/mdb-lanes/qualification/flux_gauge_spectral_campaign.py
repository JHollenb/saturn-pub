"""Conditional native-prefix FLUX gauge and source-factor even-writer futures.

Gauge edits are public post-project state Acts. Spectral edits are public
joint0-exit endpoint Acts, qualified separately by original FF operand assays.
Neither is a claim that an edited native block internally produced its endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import runpy
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = "mdb-flux-gauge-spectral-v1"
RESULT = "flux-gauge-spectral-result"
RAW_SHA256 = "b04db3835522a007a15293ef177e4463cf05e9d01122d47d42531dc72bd9700a"
HELPERS = {
    "flux_raw_to_image.py": RAW_SHA256,
    "flux_functional_denoiser.py": "cb794daa5818ec3f9107f23839a41010f9569148da3708e39cedd364dff21923",
    "flux_functional_decoder.py": "e2712298e42b37a263dea1b24f0f491082b8a01086b16fa66577f474c68f491f",
    "flux_functional_vae.py": "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61",
    "flux_functional_qwen_prefix.py": "763ae441d33396fe5cede5d96214c76e01aae779916ce5cf28e79cf5f5eb618a",
    "flux_functional_qwen_prefix_worker.py": "7f13836ae94c84b7ed0f7df67b1f7b1d28a7e534a83d509fa7320a09a6d845df",
    "flux_image_tail_microscope.py": "b9efb61874005c8b11307c197a2ea5299078b42d5b4ab4b5cc719f7a4a4a32bb",
    "flux_source_receiver.py": "d483e41bfac018c8225662e8417bcef46f6916fa2a14d910f16e9a12beb5250c",
}
PROMPTS = (
    "A yellow ceramic cone to the left of a teal metal cylinder on a white tabletop. Two separate objects, simple studio photograph.",
    "A teal metal cylinder to the left of a yellow ceramic cone on a white tabletop. Two separate objects, simple studio photograph.",
)
CONTEXTS = tuple(dict(id=f"c{p}s{seed}", prompt=prompt, seed=seed, primary=(p == 0 and seed == 391) or (p == 1 and seed == 457))
    for p, prompt in enumerate(PROMPTS) for seed in (391, 457))
PRIMARY_ARMS = ("baseline", "sham", "gauge", "mean-zero-control", "top-even-add", "bottom-even-add", "random-even-add")
CROSSED_ARMS = ("baseline", "top-even-add", "bottom-even-add")
DIRECTIONS = ("top", "bottom", "random")
CONDITIONAL_SCOPE = dict(kind="architectural", origin="actual-native-postproject-StateCut",
    scope="fresh prompt/seed native Qwen and project prefixes; complete own-state suffix forecasts",
    raw_prompt_forecast=False, independent_decoder_prediction=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def read(path):
    return json.loads(Path(path).read_text())


def helpers():
    for name, expected in HELPERS.items():
        require(sha(Path(__file__).with_name(name)) == expected, "frozen helper differs: " + name)
    return runpy.run_path(str(Path(__file__).with_name("flux_raw_to_image.py")))


def path_keys():
    return (*helpers()["_source_keys"](), "selection_protocol")


def contract():
    return dict(contexts=4, image_futures=20, native_qwen_source_calls=2, native_project_source_transitions=4, noise_draws=4,
        logical_suffix_transitions_per_lane=2220, shared_joint0_reuses_per_lane=12,
        functional_transitions=2208, native_target_transitions=2208, common_joint0_per_lane=4, edited_joint0_per_lane=4,
        functional_ff_assays=24, native_ff_assays=24, ff_assay_linears_per_lane=48,
        functional_vae_calls=20, native_vae_repeat_calls=4, native_vae_reference_calls=20,
        gauge_or_sham_acts_per_lane=12, spectral_endpoint_acts_per_lane=10,
        native_source_restores=4, component_hydrations=3, cuda_placements=5, cpu_offloads=5,
        functional_module_forwards=0, persistent_weight_banks=0, max_gpu_components=1)


def make_config(base):
    config = {key: value for key, value in base.items() if key not in ("prompt", "seed", "worker_sha256", "request_sha256", "job_id", "output", "_request_record")}
    config.update(schema=SCHEMA, contexts=list(CONTEXTS), primary_arms=list(PRIMARY_ARMS), crossed_arms=list(CROSSED_ARMS),
        dose=.25, gauge_amplitude=.5, spectral_iterations=64, direction_seed=602, spectral_site=dict(step=0, phase="joint", index=0, stream="image.ff"),
        spectral_coordinate="gate_ff * (e0-e1)/sqrt2; actual gated coordinate contrast, no semantic label",
        endpoint_order="fl(R_attention + fl(gate_ff * fl(M_base + fl(.25 * even_write))))",
        spectral_intervention="public joint0-exit image endpoint Act; separate original FF +/- operand assays",
        height=512, width=512, steps=4, batch_size=1, device="cuda", dtype="bfloat16", flux_mode="capture", capture_sites=True,
        component_hydrations=3, cuda_placements=5, offloads=5, max_gpu_components=1,
        output_name=RESULT, helper_source_sha256=dict(HELPERS), max_wall_seconds=10800,
        forecast_transaction_bytes=1 << 30, transport_limits=dict(max_expanded_bytes=16 << 30, max_member_bytes=134221824),
        retention="full edit/first-downstream/noise/commit states and FF assays; full hashes/statistics elsewhere",
        decoder_contract="20 functional shared-primitive VAE forecasts;20 actual native-final decodes and4 exact same-native-final-latent repeats",
        scientific_gates=[])
    return config


def validate_config(config):
    fixed = make_config({})
    for key, expected in fixed.items():
        require(config.get(key) == expected, "frozen campaign field differs: " + key)
    require(config.get("guidance_scale", 1.) == 1. and config.get("reference_image", False) is False
        and config.get("retry", False) is False and config.get("nested_mrun", False) is False, "unsupported execution profile")
    for key in path_keys():
        require(config.get(key) and config.get(key + "_sha256") and sha(config[key]) == config[key + "_sha256"], "source metadata differs: " + key)
    require(read(config["selection_protocol"])["contexts"] == list(CONTEXTS), "frozen raw context selection differs")
    require(config.get("qwen_manifest_entries") == 398 and config.get("flux_manifest_entries") == 169 and config.get("vae_manifest_entries") == 142, "original recipe counts differ")
    return contract()


def gauge_replacement(value, kind, amplitude=.5):
    """Nominal norm matching; preserve actual BF16 rounding/leakage in evidence."""
    import torch

    require(value.ndim == 3 and value.dtype == torch.bfloat16, "gauge requires a BF16 B,T,C state slot")
    require(kind in ("sham", "gauge", "mean-zero-control"), "unknown state edit")
    if kind == "sham":
        return value.clone()
    rows = torch.where(torch.arange(value.shape[1], device=value.device) % 2 == 0, amplitude, -amplitude).to(torch.float32).view(1, -1, 1)
    if kind == "mean-zero-control":
        require(value.shape[-1] % 2 == 0, "centered control requires an even channel width")
        rows = rows * torch.where(torch.arange(value.shape[-1], device=value.device) % 2 == 0, 1., -1.).view(1, 1, -1)
    return (value.float() + rows).to(value.dtype)


def leakage(original, changed):
    import torch

    delta = changed.double() - original.double()
    centered = delta - delta.mean(-1, keepdim=True)
    return dict(elements=delta.numel(), delta_rms=float(delta.square().mean().sqrt()), centered_leakage_rms=float(centered.square().mean().sqrt()),
        max_abs=float(delta.abs().max()), changed_bits=int((original.view(torch.int16) != changed.view(torch.int16)).sum()),
        row_mean_min=float(delta.mean(-1).min()), row_mean_max=float(delta.mean(-1).max()),
        scope="actual postrounding BF16 displacement; nominal norms need not match after cast")


def factor_directions(weight_in, weight_out, gate, *, iterations=64, random_seed=602):
    """Bounded deterministic Ritz candidates for the original-factor symmetric B.

No dense B, Jacobian, learned forward, or exported parameter bank is formed.
Top/bottom mean extremal FOUND Ritz values; residuals are observations, not gates.
"""
    import torch

    require(weight_in.ndim == weight_out.ndim == 2 and weight_in.shape[0] == 2 * weight_out.shape[1]
        and weight_in.shape[1] == weight_out.shape[0] and gate.numel() == weight_out.shape[0], "FF factor geometry differs")
    width = weight_in.shape[1]; steps = min(iterations, width)
    require(width >= 2 and steps >= 2, "spectral search needs two dimensions")
    wg, wu = (part.detach().to(torch.float64) for part in weight_in.chunk(2, dim=0))
    contrast = torch.zeros(width, device=wg.device, dtype=torch.float64); contrast[0] = 1 / math.sqrt(2); contrast[1] = -1 / math.sqrt(2)
    coordinate = gate.detach().reshape(-1).double() * contrast
    d = weight_out.detach().double().T @ coordinate
    work = dict(operator_calls=0, factor_matvecs=0, projection_matvecs=1, ritz_eigh=0, reorthogonalization_passes=0)
    def operator(v):
        work["operator_calls"] += 1; work["factor_matvecs"] += 4
        return (wg.T @ (d * (wu @ v)) + wu.T @ (d * (wg @ v))) / 4
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    first = torch.randn(width, generator=generator, dtype=torch.float64).to(wg.device)
    first /= first.norm()
    basis, products = [], []
    vector = first
    for _ in range(steps):
        basis.append(vector); product = operator(vector); products.append(product)
        candidate = product.clone()
        for _pass in range(2):
            for old in basis:
                candidate -= torch.dot(old, candidate) * old
            work["reorthogonalization_passes"] += 1
        norm = candidate.norm()
        if float(norm) < 1e-14:
            break
        vector = candidate / norm
    q, bq = torch.stack(basis, dim=1), torch.stack(products, dim=1)
    small = q.T @ bq; small = (small + small.T) / 2
    values, vectors = torch.linalg.eigh(small); work["ritz_eigh"] += 1
    result, report = {}, {}
    for name, index in (("top", -1), ("bottom", 0)):
        v = q @ vectors[:, index]; v /= v.norm()
        v *= torch.where(v[v.abs().argmax()] < 0, -1., 1.)
        bv = operator(v); eigenvalue = torch.dot(v, bv); residual = (bv - eigenvalue * v).norm()
        result[name] = v
        report[name] = dict(eigenvalue=float(eigenvalue), residual_norm=float(residual), operator_norm=float(bv.norm()),
            relative_residual=None if float(bv.norm()) == 0 else float(residual / bv.norm()), scope="strongest-found bounded Ritz candidate; no convergence gate")
    random = torch.randn(width, generator=generator, dtype=torch.float64).to(wg.device); random /= random.norm()
    result["random"] = random
    bv = operator(random); report["random"] = dict(rayleigh=float(torch.dot(random, bv)), norm=float(random.norm()), source="fixed CPU Gaussian seed then unit norm; no target selection")
    return result, dict(directions=report, basis_rank=len(basis), requested_iterations=iterations, work=work,
        coordinate=coordinate, scope="original BF16 factors promoted to FP64; real quadratic coordinate account, not BF16 regrouping identity")


def directional_inputs(x, direction):
    """Each row keeps its own source normalized/modulated input norm."""
    norm = x.double().square().sum(-1, keepdim=True).sqrt()
    positive = (norm * direction.reshape(1, 1, -1)).to(x.dtype)
    return positive, -positive


def even_write(positive, negative):
    return (positive + negative) * .5


def augmented_endpoint(residual, gate, baseline_write, even, dose=.25):
    return residual + gate * (baseline_write + dose * even)


def scalar_rows(value):
    """Complete FP64 row values, with a canonical hash and no ranking gate."""
    values = value.detach().double().contiguous().cpu()
    return dict(shape=list(values.shape), dtype="float64", elements=values.numel(), values=values.reshape(-1).tolist(),
        sha256=hashlib.sha256(values.numpy().astype("<f8").tobytes(order="C")).hexdigest(),
        rms=float(values.square().mean().sqrt()), signed_mean=float(values.mean()), min=float(values.min()), max=float(values.max()))


def reduced_law_prediction(x, direction, rayleigh, even, coordinate, positive, negative):
    """Source-only quadratic law versus already computed original-order FF work.

    The ideal direction/radius is unrounded. The actual operands and finite even
    write retain BF16 casts and operation order; no regrouping equality is assumed.
    """
    norm_squared = x.double().square().sum(-1)
    ideal = norm_squared * rayleigh
    finite = (even.double() * coordinate.reshape(1, 1, -1)).sum(-1)
    return dict(rayleigh=rayleigh, direction_norm_squared=float(direction.double().square().sum()),
        source_input_norm_squared=scalar_rows(norm_squared), ideal_unrounded=scalar_rows(ideal), full_finite_forecast=scalar_rows(finite),
        ideal_minus_full_finite=scalar_rows(ideal - finite), actual_positive_operand_norm_squared=scalar_rows(positive.double().square().sum(-1)),
        actual_negative_operand_norm_squared=scalar_rows(negative.double().square().sum(-1)),
        scope="rho(original modulated x)^2 * source-factor Rayleigh versus dot(cFF, original-order BF16 even write); observations, no BF16 regrouping claim")


def reduced_ordering(directions):
    names = list(DIRECTIONS)
    return {field: dict(direction_indices=names, descending_per_row=[sorted(range(len(names)),
        key=lambda index: -directions[names[index]][field]["values"][row])
        for row in range(directions[names[0]][field]["elements"])], scope="observed source-only ordering; ties stable; no gate")
        for field in ("ideal_unrounded", "full_finite_forecast")}


def apply_slot_edit(session, slot, replacement):
    """One real public exact-shape patch; callers archive both complete cuts."""
    import torch
    from saturn_pub.core import Act
    from saturn_pub.values import describe, digest

    before = session.capture(retain=False); original = before.payload[slot]
    act = Act.patch(slot, replacement, torch.ones_like(original, dtype=torch.bool), mode="replace", expected_digest=digest(describe(original)))
    receipt = session.apply(act); after = session.capture(retain=False)
    body = receipt.to_dict()
    require(body["parent"] == before.fingerprint and body["result"] == after.fingerprint, "actual Act custody differs")
    return before, after, receipt


def step_equations(graph, model, state):
    """One own-state transition using the unchanged qualified primitive functions."""
    out = dict(state); step, phase, index = state["step"], state["phase"], state["cursor"]
    if phase == "project":
        out["image"], out["text"], out["context"] = graph["functional_project"](model, state["latent"], state["conditioning"], state["img_ids"], state["txt_ids"], state["timesteps"][step])
        out.update(phase="joint", cursor=0)
    elif phase == "joint":
        out["image"], out["text"] = graph["functional_joint"](model, index, state["image"], state["text"], state["context"])
        out["cursor"] = index + 1
        if out["cursor"] == len(model.transformer_blocks):
            out.update(phase="single", cursor=0)
    elif phase == "single":
        out["image"], out["text"] = graph["functional_single"](model, index, state["image"], state["text"], state["context"])
        out["cursor"] = index + 1
        if out["cursor"] == len(model.single_transformer_blocks):
            out["phase"] = "readout"
    elif phase == "readout":
        out["noise_prediction"] = graph["functional_readout"](model, state["image"], state["context"]); out["phase"] = "scheduler"
    elif phase == "scheduler":
        out["latent"] = graph["functional_scheduler"](state["latent"], state["noise_prediction"], state["sigmas"], step)
        out.update(step=step + 1, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
    else:
        raise ValueError("invalid functional state clock")
    return out


def selected_joint0(graph, model, state):
    """Original-order joint0 equations plus the actual image FF writer operands."""
    import torch

    block = model.transformer_blocks[0]; image, text, context = (state[name] for name in ("image", "text", "context"))
    (shift, scale, gate), (shift_ff, scale_ff, gate_ff) = graph["_mods"](context["img_mod"], 2)
    (cshift, cscale, cgate), (cshift_ff, cscale_ff, cgate_ff) = graph["_mods"](context["txt_mod"], 2)
    image_ln, text_ln = graph["_norm"](block.norm1, image), graph["_norm"](block.norm1_context, text)
    x = (1 + scale) * image_ln + shift
    c = (1 + cscale) * text_ln + cshift
    attn = block.attn
    q, k, v = (graph["_linear"](getattr(attn, name), x).unflatten(-1, (-1, attn.head_dim)) for name in ("to_q", "to_k", "to_v"))
    cq, ck, cv = (graph["_linear"](getattr(attn, name), c).unflatten(-1, (-1, attn.head_dim)) for name in ("add_q_proj", "add_k_proj", "add_v_proj"))
    q = torch.cat([graph["_rms"](attn.norm_added_q, cq), graph["_rms"](attn.norm_q, q)], dim=1)
    k = torch.cat([graph["_rms"](attn.norm_added_k, ck), graph["_rms"](attn.norm_k, k)], dim=1)
    out = graph["_attention"](q, k, torch.cat([cv, v], dim=1), context["rope"])
    cout, out = out[:, :text.shape[1]], out[:, text.shape[1]:]
    residual = image + gate * graph["_linear"](attn.to_out[0], out)
    operand = graph["_norm"](block.norm2, residual) * (1 + scale_ff) + shift_ff
    write = graph["_ff"](block.ff, operand)
    image = residual + gate_ff * write
    text = text + cgate * graph["_linear"](attn.to_add_out, cout)
    text = text + cgate_ff * graph["_ff"](block.ff_context, graph["_norm"](block.norm2_context, text) * (1 + cscale_ff) + cshift_ff)
    return dict(state, image=image, text=text, phase="joint", cursor=1), dict(R_attention=residual, x=operand, M_base=write,
        gate_ff=gate_ff, image_ln=image_ln, text_ln=text_ln)


def retain_transition(event):
    return event["phase"] in ("readout", "scheduler") or (event["step"] == 0 and event["phase"] == "joint" and event["index"] in (0, 1))


def signed_fields(left, right, prefix=""):
    import torch

    if isinstance(left, torch.Tensor):
        require(isinstance(right, torch.Tensor) and left.shape == right.shape, "response tensor shape differs")
        value = (left.double() - right.double()).contiguous().cpu()
        raw = value.numpy().astype("<f8").tobytes()
        return {prefix or "/": dict(shape=list(value.shape), elements=value.numel(), dtype="float64", sha256=hashlib.sha256(raw).hexdigest(),
            rms=float(value.square().mean().sqrt()), signed_mean=float(value.mean()), max_abs=float(value.abs().max()), changed=int((value != 0).sum()))}
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), "response field tree differs")
        return {key: row for name in left for key, row in signed_fields(left[name], right[name], prefix + "/" + name).items()}
    if isinstance(left, (tuple, list)):
        require(isinstance(right, (tuple, list)) and len(left) == len(right), "response sequence tree differs")
        return {key: row for number, (a, b) in enumerate(zip(left, right)) for key, row in signed_fields(a, b, prefix + "/" + str(number)).items()}
    return {}


def tensor_tree(value, old):
    import torch

    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "nonfinite boundary tensor")
        flat = value.detach().reshape(-1)
        indices = torch.linspace(0, max(0, flat.numel() - 1), min(16, flat.numel()), device=flat.device).long()
        return dict(tensor=old["_tensor_signature"](value), elements=value.numel(),
            rms=float(value.double().square().mean().sqrt()), signed_mean=float(value.double().mean()),
            witness_indices=indices.cpu().tolist(), witness_values=flat[indices].double().cpu().tolist())
    if isinstance(value, dict):
        return {key: tensor_tree(item, old) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(item, old) for item in value]
    return value


def component_cut(adapter, state, parent):
    from saturn_pub.core import StateCut

    return StateCut(adapter.model_identity, {**dict(adapter.execution), "composition": SCHEMA,
        "provenance": "component-derived equations; not a native advance receipt"}, adapter.boundary(state), parent, state,
        adapter.point(state).to_dict(), adapter.surface(state).to_dict())


def cut_record(cut, store="state-store"):
    return dict(cut=cut.fingerprint, parent=cut.parent, model_identity=cut.model_identity, execution_point=dict(cut.execution_point), store=store)


@contextmanager
def instrument(models, phase, released, observed):
    """Actual shared-kernel and module-entry census with a global target guard."""
    from torch.nn import functional as F

    linear, sdpa = F.linear, F.scaled_dot_product_attention
    handles = []
    def module_entry(_module, _args):
        lane = phase["kind"]
        require(lane and not lane.startswith("forecast"), "module forward forbidden during functional forecasting")
        require(not lane.startswith("target") or released["native"], "native target before global seal")
        observed.setdefault(lane, {}).setdefault("module_entries", 0)
        observed[lane]["module_entries"] += 1
    for model in models:
        for module in model.modules():
            handles.append(module.register_forward_pre_hook(module_entry))
    def invoke(name, function, *args, **kwargs):
        lane = phase["kind"]; require(bool(lane), "unclassified primitive work")
        observed.setdefault(lane, {}).setdefault(name, 0); observed[lane][name] += 1
        return function(*args, **kwargs)
    F.linear = lambda *args, **kwargs: invoke("linear", linear, *args, **kwargs)
    F.scaled_dot_product_attention = lambda *args, **kwargs: invoke("SDPA", sdpa, *args, **kwargs)
    try:
        yield
    finally:
        F.linear, F.scaled_dot_product_attention = linear, sdpa
        for handle in reversed(handles):
            handle.remove()


def frontend(config, conditioning, seed, pipeline):
    """Same original noise/packing/dynamic-schedule operations, explicit new seed."""
    import numpy as np
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

    generator = torch.Generator(device="cuda").manual_seed(seed); before = generator.get_state().clone()
    raw = torch.randn((1, 128, 32, 32), generator=generator, device="cuda", dtype=torch.bfloat16)
    after = generator.get_state().clone()
    packed = pipeline._pack_latents(raw)
    ids = pipeline._prepare_latent_ids(raw).to("cuda"); txt_ids = pipeline._prepare_text_ids(conditioning).to("cuda")
    sigmas = np.linspace(1., .25, 4); mu = compute_empirical_mu(image_seq_len=packed.shape[1], num_steps=4)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"]))
    scheduler.set_timesteps(num_inference_steps=4, device="cuda", sigmas=sigmas, mu=mu)
    return dict(raw_noise=raw, packed_noise=packed, img_ids=ids, txt_ids=txt_ids,
        timesteps=scheduler.timesteps.detach().clone(), sigmas=scheduler.sigmas.detach().clone(),
        mu=float(mu), generator_before=before, generator_after=after, base_sigmas=sigmas.tolist(), scheduler_config=dict(scheduler.config))


def run(config, *, progress=None):
    declared, old = validate_config(config), helpers()
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from saturn_pub.adapters.flux2 import Flux2KleinAdapter
    from saturn_pub.core import Session
    from saturn_pub.store import LocalStore
    from saturn_pub.values import describe
    from transformers import AutoTokenizer

    from mdb_lanes.forecast_transaction import ForecastTransaction, verify_transaction

    output = Path(config["output"]); output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", {key: value for key, value in config.items() if key != "_request_record"})
    if config.get("_request_record"):
        save(output / "request.json", config["_request_record"])
    started = time.monotonic(); events, receipts, placements, observed = [], [], [], {}
    phase, released = dict(kind="source-loading"), dict(native=False)
    counts = {key: 0 for key in declared if "per_lane" not in key and key not in ("max_gpu_components",)}
    counts.update(contexts=4, image_futures=20)
    def event(kind, **details):
        row = dict(number=len(events), kind=kind, seconds=time.monotonic() - started, **details)
        require(row["seconds"] <= config["max_wall_seconds"], "finite campaign wall bound exceeded")
        events.append(row)
        with (output / "timeline.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
        if progress:
            progress(row)
    def receipt(kind, root, arm, lane, value):
        row = dict(number=len(receipts), kind=kind, context=root, arm=arm, lane=lane, receipt=value.to_dict())
        receipts.append(row)
        with (output / "receipt-emissions.jsonl").open("ab") as stream:
            stream.write(canonical(row) + b"\n")
    torch.set_num_threads(config.get("cpu_threads", 4)); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    require(torch.cuda.is_available(), "original campaign requires CUDA")
    modules = old["_helper_modules"]({key: value for key, value in config.items() if key != "worker_sha256"})
    qw, graph, decoder_graph = (modules[old[key]] for key in ("QWEN_WORKER", "DENOISER_HELPER", "DECODER_HELPER"))
    runtime = qw["_runtime_contract"](config, torch); runtime["native_diffusers_sources"] = old["_native_source_contract"](config)["sources"]
    save(output / "runtime-contract.json", runtime)
    sources = {}
    for key in path_keys():
        source = Path(config[key]); target = output / "source" / key / source.name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        sources[key] = dict(path=str(target.relative_to(output)), sha256=sha(target), bytes=target.stat().st_size)
    binding = read(config["qwen_source_binding"]); tokenizer_files = qw["_copy_tokenizer_sources"](output, binding)
    tokenizer = AutoTokenizer.from_pretrained(binding["checkpoint"]["model_root"] + "/tokenizer", local_files_only=True, padding_side="right")
    require(tokenizer.is_fast and tokenizer.padding_side == "right", "qualified fast right-padding tokenizer required")
    qb, qm = qw["_load_recipe_bank"](Path(config["qwen_recipe_helper"]), Path(config["qwen_manifest"]), config["qwen_parameter_root"])
    qwen = qw["_load_qwen"](config, torch.device("cpu"), qb, qm)
    flux, fb, fm = old["_load_transformer"](config, runpy.run_path(config["flux_recipe_helper"]))
    vae, vb, vm = old["_load_vae"](config, runpy.run_path(config["vae_recipe_helper"]))
    require(len(qm["artifacts"]) == 398 and len(fm["artifacts"]) == 169 and len(vm["artifacts"]) == 142, "hydrated recipe census differs")
    graph["validate_model"](flux)
    require(len(flux.transformer_blocks) == 5 and len(flux.single_transformer_blocks) == 20
        and flux.transformer_blocks[0].ff.linear_in.bias is None and flux.transformer_blocks[0].ff.linear_out.bias is None,
        "original gauge/parity model applicability differs")
    ff_applicability = dict(site="transformer_blocks.0.ff", linear_in_bias_none=flux.transformer_blocks[0].ff.linear_in.bias is None,
        linear_out_bias_none=flux.transformer_blocks[0].ff.linear_out.bias is None, activation="SiLU(gate)*up",
        source_factors=["transformer_blocks.0.ff.linear_in.weight", "transformer_blocks.0.ff.linear_out.weight"],
        scope="actual bias-free FF applicability checked before parity formation; no error/ordering gate")
    save(output / "ff-applicability.json", ff_applicability)
    counts["component_hydrations"] = 3
    models = dict(qwen=qwen, flux=flux, vae=vae)
    prefixes = dict(qwen=None, flux=None, vae=("post_quant_conv.", "decoder.", "bn."))
    identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    tied = qwen.lm_head.weight is qwen.model.embed_tokens.weight; require(tied, "Qwen tied alias missing")
    def place(name, device):
        model = models[name]
        if name == "vae":
            model.post_quant_conv.to(device); model.decoder.to(device)
            require(vae.bn.running_mean.device.type == vae.bn.running_var.device.type == "cpu", "VAE BN left CPU")
        else:
            model.to(device)
        signature = old["_component_signature"](model, prefixes=prefixes[name])
        old["_assert_signature_equal"](identities[name], signature, name)
        placements.append(dict(component=name, device=device, signature=signature))
        counts["cuda_placements" if device == "cuda" else "cpu_offloads"] += 1
        event("placement", component=name, device=device)
        if device == "cpu":
            torch.cuda.empty_cache()
    store = LocalStore(output / "state-store")
    def retained(cut):
        store.save(cut); return cut_record(cut)
    def component_record(adapter, state, parent):
        return retained(component_cut(adapter, state, parent))
    def patch(session, slot, replacement, root, arm, lane):
        before, after, value = apply_slot_edit(session, slot, replacement)
        record = dict(slot=slot, before=retained(before), after=retained(after), receipt=value.to_dict(),
            all_other_slots_unchanged=describe({key: item for key, item in before.payload.items() if key != slot}) == describe({key: item for key, item in after.payload.items() if key != slot}))
        require(record["all_other_slots_unchanged"], "Act changed an undeclared slot")
        receipt("act", root, arm, lane, value)
        counter = "spectral_endpoint_acts_per_lane" if arm.endswith("-even-add") else "gauge_or_sham_acts_per_lane"
        counts[counter + "_" + lane] = counts.get(counter + "_" + lane, 0) + 1
        return record
    roots, qwen_memo, inputs, ff_records = {}, {}, {}, {"forecast": {}, "native": {}}
    reduced_predictions = {}
    frontier_cuts, paths, row_records = {}, {}, {}
    processor = Flux2ImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1) * 2)
    with instrument(list(models.values()), phase, released, observed), torch.inference_mode():
        place("qwen", "cuda"); phase["kind"] = "source-qwen"
        for number, prompt in enumerate(PROMPTS):
            formatted, ids, mask = qw["_tokenize"](tokenizer, prompt, config)
            result = qwen(input_ids=ids.to("cuda"), attention_mask=mask.to("cuda"), output_hidden_states=True, use_cache=False)
            conditioning = torch.cat([result.hidden_states[index] for index in (9, 18, 27)], dim=-1)
            require(tuple(conditioning.shape) == (1, 512, 7680) and conditioning.dtype == torch.bfloat16
                and tuple(ids.shape) == tuple(mask.shape) == (1, 512), "native source conditioning/token geometry differs")
            qwen_memo[number] = conditioning.detach().cpu()
            input_record = dict(prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), formatted_chat=formatted,
                input_ids=old["_save_tensor"](output, f"inputs/c{number}/ids.npy", ids), attention_mask=old["_save_tensor"](output, f"inputs/c{number}/mask.npy", mask),
                selected_hidden_states=[9, 18, 27], conditioning=old["_save_tensor"](output, f"inputs/c{number}/conditioning.npy", conditioning),
                actual_full_layers=36, actual_LM_readout=True, memo_scope="same literal formatted prompt; independent of diffusion seed")
            inputs[number] = input_record; counts["native_qwen_source_calls"] += 1
            del result, conditioning
        save(output / "conditioning-memo.json", inputs); place("qwen", "cpu")
        place("flux", "cuda"); phase["kind"] = "source-project"
        adapter = Flux2KleinAdapter(flux, FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"])), granularity="operation", device="cuda")
        for selected in CONTEXTS:
            identifier = selected["id"]; prompt_index = 0 if selected["prompt"] == PROMPTS[0] else 1
            conditioning = qwen_memo[prompt_index].to("cuda")
            front = frontend(config, conditioning, selected["seed"], Flux2KleinPipeline); counts["noise_draws"] += 1
            # The old record helper's seed label is fixed277; write this new schema explicitly.
            front_record = {name: old["_save_tensor"](output, f"inputs/{identifier}/{name}.npy", value) if name not in ("generator_before", "generator_after")
                else old["_save_bytes"](output, f"inputs/{identifier}/{name}.npy", value) for name, value in front.items() if isinstance(value, torch.Tensor)}
            save(output / f"inputs/{identifier}/frontend.json", dict(front_record, seed=selected["seed"], mu=front["mu"], base_sigmas=front["base_sigmas"], conditioning_memo=f"c{prompt_index}"))
            session = adapter.session(latent=front["packed_noise"], conditioning=conditioning, img_ids=front["img_ids"], txt_ids=front["txt_ids"], timesteps=front["timesteps"], sigmas=front["sigmas"])
            before = session.capture(retain=False); retained(before)
            value = session.continue_(); receipt("source-project", identifier, "common", "source", value)
            parent = session.capture(retain=False); frontier_cuts[identifier] = retained(parent)
            require(parent.payload["phase"] == "joint" and parent.payload["cursor"] == parent.payload["step"] == 0, "source cut is not post-project")
            counts["native_project_source_transitions"] += 1
            roots[identifier] = dict(selected, arms=list(PRIMARY_ARMS if selected["primary"] else CROSSED_ARMS), source=frontier_cuts[identifier])
            del front, session, before, parent, conditioning
        save(output / "source-frontiers.json", dict(scope=CONDITIONAL_SCOPE, roots=roots))
        phase["kind"] = "forecast-denoiser"
        direction_values, direction_report = None, None
        for identifier, root in roots.items():
            parent = store.load(root["source"]["cut"], device="cuda")
            common, account = selected_joint0(graph, flux, parent.payload); counts["functional_transitions"] += 1; counts["common_joint0_per_lane_forecast"] = counts.get("common_joint0_per_lane_forecast", 0) + 1
            common_record = component_record(adapter, common, parent.fingerprint)
            account_records = {name: old["_save_tensor"](output, f"writer/{identifier}/forecast/{name}.npy", value) for name, value in account.items()}
            if direction_values is None:
                direction_values, direction_report = factor_directions(flux.transformer_blocks[0].ff.linear_in.weight,
                    flux.transformer_blocks[0].ff.linear_out.weight, account["gate_ff"], iterations=config["spectral_iterations"], random_seed=config["direction_seed"])
                coordinate = direction_report.pop("coordinate")
                direction_report["coordinate"] = old["_save_tensor"](output, "directions/gated-coordinate.npy", coordinate)
                direction_report["vectors"] = {name: old["_save_tensor"](output, f"directions/{name}.npy", value) for name, value in direction_values.items()}
                direction_report["source_context"] = identifier
                save(output / "spectral-source.json", direction_report)
            ff_records["forecast"][identifier] = {}
            reduced_predictions[identifier] = dict(directions={})
            endpoints = {}
            for name, direction in direction_values.items():
                positive, negative = directional_inputs(account["x"], direction)
                predicted = []
                for sign, operand in (("plus", positive), ("minus", negative)):
                    result = graph["_ff"](flux.transformer_blocks[0].ff, operand); counts["functional_ff_assays"] += 1
                    record = dict(input=old["_save_tensor"](output, f"ff-assays/forecast/{identifier}/{name}/{sign}-input.npy", operand),
                        output=old["_save_tensor"](output, f"ff-assays/forecast/{identifier}/{name}/{sign}-output.npy", result))
                    ff_records["forecast"][identifier].setdefault(name, {})[sign] = record; predicted.append(result)
                even = even_write(*predicted)
                ff_records["forecast"][identifier][name]["even"] = old["_save_tensor"](output, f"ff-assays/forecast/{identifier}/{name}/even.npy", even)
                rayleigh = direction_report["directions"][name]["eigenvalue" if name != "random" else "rayleigh"]
                reduced_predictions[identifier]["directions"][name] = reduced_law_prediction(account["x"], direction, rayleigh,
                    even, coordinate, positive, negative)
                if name + "-even-add" in root["arms"]:
                    endpoint = augmented_endpoint(account["R_attention"], account["gate_ff"], account["M_base"], even, config["dose"])
                    endpoints[name + "-even-add"] = old["_save_tensor"](output, f"writer/{identifier}/endpoints/{name}.npy", endpoint)
            reduced_predictions[identifier]["ordering"] = reduced_ordering(reduced_predictions[identifier]["directions"])
            save(output / f"writer/{identifier}/account.json", dict(original=account_records, endpoints=endpoints,
                coordinate=direction_report["coordinate"], source_context=direction_report["source_context"], endpoint_contract="sealed finite functional receiving residual/gate/base M + dose*even; endpoint Act, not altered native block"))
            for arm in root["arms"]:
                branch_root = output / "cases" / identifier / arm; branch_root.mkdir(parents=True)
                acts = []
                if arm in ("gauge", "mean-zero-control"):
                    branch = Session(adapter, parent.payload, parent=parent.parent).fork(parent)
                    leak = {}
                    for slot in ("image", "text"):
                        own = branch.read(slot); replacement = gauge_replacement(own, arm, config["gauge_amplitude"])
                        leak[slot] = leakage(own, replacement); acts.append(patch(branch, slot, replacement, identifier, arm, "forecast"))
                    state, selected_account = selected_joint0(graph, flux, branch.capture(retain=False).payload)
                    counts["functional_transitions"] += 1; counts["edited_joint0_per_lane_forecast"] = counts.get("edited_joint0_per_lane_forecast", 0) + 1
                    start_record = component_record(adapter, state, acts[-1]["after"]["cut"])
                    save(branch_root / "gauge-leakage.json", leak)
                    save(branch_root / "joint0-account-forecast.json", {name: old["_save_tensor"](branch_root, "joint0/forecast/" + name + ".npy", value) for name, value in selected_account.items()})
                else:
                    state = common; start_record = common_record
                    if arm == "sham" or arm.endswith("-even-add"):
                        branch = Session(adapter, state, parent=common_record["cut"])
                        for slot in (("image", "text") if arm == "sham" else ("image",)):
                            replacement = branch.read(slot) if arm == "sham" else load_tensor(output, endpoints[arm], "cuda")
                            acts.append(patch(branch, slot, replacement, identifier, arm, "forecast"))
                        state = branch.capture(retain=False).payload; start_record = acts[-1]["after"]
                save(branch_root / "forecast-acts.json", acts)
                rows = [dict(number=0, shared_joint0=arm not in ("gauge", "mean-zero-control"), retained=start_record, fields=tensor_tree(state, old))]
                previous = start_record["cut"]
                for number in range(1, 111):
                    clock = {key: state[key] for key in ("step", "phase", "cursor")}
                    state = step_equations(graph, flux, state); counts["functional_transitions"] += 1
                    event_row = dict(step=clock["step"], phase=clock["phase"], index=clock["cursor"])
                    row = dict(number=number, event=event_row, fields=tensor_tree(state, old))
                    if retain_transition(event_row):
                        row["retained"] = component_record(adapter, state, previous); previous = row["retained"]["cut"]
                    rows.append(row)
                require(state["step"] == 4 and state["phase"] == "project" and state["cursor"] == 0, "forecast suffix did not close")
                save(branch_root / "forecast-rows.json", rows)
                paths[(identifier, arm)] = dict(forecast_final=previous, root=str(branch_root.relative_to(output)))
                row_records[(identifier, arm)] = rows
                event("forecast_suffix_completed", context=identifier, arm=arm, final_cut=previous)
                del state, rows
            del parent, common, account, positive, negative, predicted, even
        save(output / "ff-forecasts.json", ff_records["forecast"])
        save(output / "reduced-law-predictions.json", dict(schema="mdb-flux-reduced-even-law-predictions-v1", roots=reduced_predictions,
            applicability=ff_applicability, coordinate=direction_report["coordinate"], directions=direction_report["vectors"],
            native_targets_at_formation=0, scope="Complete pretarget scalar forecasts; ideal real algebra and full finite FF forecast kept distinct; no quality gate"))
        place("flux", "cpu"); del adapter, direction_values, branch
        place("vae", "cuda"); phase["kind"] = "forecast-vae"
        for (identifier, arm), path in paths.items():
            branch_root = output / path["root"]; final = store.load(path["forecast_final"], device="cpu").payload
            argument = old["_decoder_input"](vae, final, config, {"Flux2KleinPipeline": Flux2KleinPipeline}).to("cuda")
            decoded = decoder_graph["functional_decode"](vae, argument, observe=None, counts={}, profile="original")
            record = decoder_result(old, branch_root, "forecast", decoded, processor)
            record["decoder_input"] = old["_save_tensor"](branch_root, "decoder/forecast/decoder-input.npy", argument)
            record.update(scope="full shared-primitive functional VAE forecast; no independent kernel claim", macros_retained=False)
            save(branch_root / "decoder-forecast.json", record); counts["functional_vae_calls"] += 1
        response_records = {}
        for (identifier, arm), path in paths.items():
            if arm == "baseline":
                continue
            reference = paths[(identifier, "baseline")]
            rows = row_records[(identifier, arm)]; baseline_rows = row_records[(identifier, "baseline")]
            selected_responses = []
            for row, base_row in zip(rows, baseline_rows):
                if "retained" in row and "retained" in base_row:
                    left = store.load(row["retained"]["cut"], device="cpu").payload
                    right = store.load(base_row["retained"]["cut"], device="cpu").payload
                    selected_responses.append(dict(number=row["number"], left_cut=row["retained"]["cut"], right_cut=base_row["retained"]["cut"], fields=signed_fields(left, right)))
            left_decoder = read(output / path["root"] / "decoder-forecast.json")
            right_decoder = read(output / reference["root"] / "decoder-forecast.json")
            decoder_responses = {name: signed_fields(load_tensor(output / path["root"], left_decoder[name], "cpu"),
                load_tensor(output / reference["root"], right_decoder[name], "cpu"))["/"] for name in ("decoded", "pixels")}
            response_records[identifier + "/" + arm] = dict(orientation="arm-minus-own-root-baseline", full_retained_states=selected_responses,
                decoder=decoder_responses, omitted_intermediate_scope="full boundary hashes/witnesses/statistics retained; no full delta bank or full norm reconstruction if a nonretained hash differs")
        save(output / "forecast-responses.json", response_records)
        place("vae", "cpu")
        # Metadata and every complete selected forecast/image are fixed before any learned target.
        forecast_inventory = old["_inventory"](output, exclude=("timeline.jsonl", "receipt-emissions.jsonl"))
        tx = ForecastTransaction(output / "forecast-transaction", model_identity=SCHEMA, input_identity=hashlib.sha256(canonical(frontier_cuts)).hexdigest(),
            source_identity=hashlib.sha256(canonical(HELPERS)).hexdigest(), frontier=dict(CONDITIONAL_SCOPE, external_files=forecast_inventory),
            execution_scope="Two actual native prompt prefixes, four source project cuts; all20 own-state image forecasts before FF/denoiser/VAE targets",
            array_storage="files-v1", budgets=dict(max_predictions=1, max_executions=1, max_native_calls=3000, max_files=30000, max_bytes=config["forecast_transaction_bytes"], max_wall_ms=config["max_wall_seconds"] * 1000))
        tx.frontier("native-prefixes", dict(cuts=frontier_cuts, scope=CONDITIONAL_SCOPE))
        tx.predict("all-futures", lambda _ctx: dict(predictions=dict(external_files=forecast_inventory,
            paths={root + "/" + arm: path for (root, arm), path in paths.items()}, ff=ff_records["forecast"], targets_at_seal=0)))
        seal = tx.seal(); event("all_forecasts_globally_sealed", seal_sha256=seal["seal_sha256"], native_targets_at_seal=0)
        def targets(_ctx):
            old["_verify_inventory"](output, forecast_inventory); released["native"] = True
            event("native_targets_released", seal_sha256=seal["seal_sha256"])
            place("flux", "cuda"); phase["kind"] = "target-ff"
            for identifier, directions in ff_records["forecast"].items():
                ff_records["native"][identifier] = {}
                for name, records in directions.items():
                    results = []
                    for sign in ("plus", "minus"):
                        operand = load_tensor(output, records[sign]["input"], "cuda")
                        captured = {}
                        handle = flux.transformer_blocks[0].ff.register_forward_pre_hook(lambda _m, args: captured.update(actual_input=old["_tensor_signature"](args[0])))
                        try:
                            result = flux.transformer_blocks[0].ff(operand)
                        finally:
                            handle.remove()
                        require(all(captured["actual_input"][key] == records[sign]["input"][saved] for key, saved in
                            (("sha256", "content_sha256"), ("shape", "shape"), ("stride", "stride"), ("dtype", "logical_dtype"), ("device", "device_at_capture"))),
                            "actual FF operand differs from sealed input")
                        ff_records["native"][identifier].setdefault(name, {})[sign] = dict(actual_input=captured["actual_input"], source_input=records[sign]["input"],
                            output=old["_save_tensor"](output, f"ff-assays/native/{identifier}/{name}/{sign}-output.npy", result))
                        results.append(result); counts["native_ff_assays"] += 1
                    ff_records["native"][identifier][name]["even"] = old["_save_tensor"](output, f"ff-assays/native/{identifier}/{name}/even.npy", even_write(*results))
            save(output / "ff-native.json", ff_records["native"])
            adapter = Flux2KleinAdapter(flux, FlowMatchEulerDiscreteScheduler.from_config(read(config["flux_scheduler_config"])), granularity="operation", device="cuda")
            phase["kind"] = "target-denoiser"
            for identifier, root in roots.items():
                parent = store.load(root["source"]["cut"], device="cuda")
                master = Session(adapter, parent.payload, parent=parent.parent)
                common_record, common_account = native_joint0(old, output, store, master, identifier, "common", receipt)
                counts["native_target_transitions"] += 1; counts["common_joint0_per_lane_native"] = counts.get("common_joint0_per_lane_native", 0) + 1
                common = master.capture(retain=False)
                for arm in root["arms"]:
                    branch_root = output / paths[(identifier, arm)]["root"]; acts = []
                    if arm in ("gauge", "mean-zero-control"):
                        branch = master.fork(parent)
                        for slot in ("image", "text"):
                            acts.append(patch(branch, slot, gauge_replacement(branch.read(slot), arm, config["gauge_amplitude"]), identifier, arm, "native"))
                        start_record, _account = native_joint0(old, output, store, branch, identifier, arm, receipt)
                        counts["native_target_transitions"] += 1; counts["edited_joint0_per_lane_native"] = counts.get("edited_joint0_per_lane_native", 0) + 1
                    else:
                        branch = master.fork(common); start_record = common_record
                        if arm == "sham" or arm.endswith("-even-add"):
                            for slot in (("image", "text") if arm == "sham" else ("image",)):
                                replacement = branch.read(slot) if arm == "sham" else load_tensor(output, read(output / f"writer/{identifier}/account.json")["endpoints"][arm], "cuda")
                                acts.append(patch(branch, slot, replacement, identifier, arm, "native"))
                            start_record = acts[-1]["after"]
                    save(branch_root / "native-acts.json", acts)
                    state = branch.capture(retain=False).payload
                    rows = [dict(number=0, shared_joint0=arm not in ("gauge", "mean-zero-control"), retained=start_record, fields=tensor_tree(state, old))]
                    for number in range(1, 111):
                        clock = {key: state[key] for key in ("step", "phase", "cursor")}
                        value = branch.continue_(); receipt("native", identifier, arm, "native", value)
                        counts["native_target_transitions"] += 1
                        after = branch.capture(retain=False); state = after.payload
                        event_row = dict(step=clock["step"], phase=clock["phase"], index=clock["cursor"])
                        row = dict(number=number, event=event_row, fields=tensor_tree(state, old), receipt=value.fingerprint)
                        if retain_transition(event_row):
                            row["retained"] = retained(after)
                        rows.append(row)
                    require(state["step"] == 4 and state["phase"] == "project" and state["cursor"] == 0, "native suffix did not close")
                    save(branch_root / "native-rows.json", rows); paths[(identifier, arm)]["native_final"] = after.fingerprint
                    require("retained" in rows[-1], "final native cut not retained")
                    event("native_suffix_completed", context=identifier, arm=arm, final_cut=after.fingerprint)
                    if arm == "baseline":
                        value = branch.restore(parent); receipt("restore", identifier, arm, "native", value)
                        restored = branch.capture(retain=False)
                        require(restored.fingerprint == parent.fingerprint, "complete native rewind did not restore source cut")
                        save(branch_root / "native-rewind.json", dict(before=rows[-1]["retained"], after=retained(restored), receipt=value.to_dict(),
                            scope="actual completedS4 trajectory restored to complete post-projectS0 source; sham futures separately replay shared own parent"))
                        counts["native_source_restores"] += 1
                save(output / f"writer/{identifier}/native-account.json", common_account)
                del parent, master, common, branch, state, rows, after
            place("flux", "cpu"); phase["kind"] = "target-vae"; place("vae", "cuda")
            for (identifier, arm), path in paths.items():
                branch_root = output / path["root"]; native_final = store.load(path["native_final"], device="cpu").payload
                record = old["_native_decoder"](branch_root, vae, native_final, config, "native_reference", {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, {}, {})
                save(branch_root / "decoder-native-reference.json", record); counts["native_vae_reference_calls"] += 1
                if roots[identifier]["primary"] and arm in ("baseline", "gauge"):
                    record = native_repeat_decode(old, branch_root, vae, native_final, config,
                        {"Flux2KleinPipeline": Flux2KleinPipeline}, processor, path["native_final"])
                    save(branch_root / "decoder-native-repeat.json", record); counts["native_vae_repeat_calls"] += 1
            place("vae", "cpu")
            return dict(counts=counts, observed_primitive_work=observed, target_paths=len(paths))
        tx.execute("original-native-targets", targets); tx.close()
    final_identities = {name: old["_component_signature"](model, prefixes=prefixes[name]) for name, model in models.items()}
    for name in models:
        old["_assert_signature_equal"](identities[name], final_identities[name], name)
    require(tied == (qwen.lm_head.weight is qwen.model.embed_tokens.weight), "Qwen alias changed")
    old["_verify_inventory"](output, forecast_inventory)
    for key in ("functional_transitions", "native_target_transitions", "functional_ff_assays", "native_ff_assays", "functional_vae_calls", "native_vae_reference_calls", "native_vae_repeat_calls", "native_qwen_source_calls", "native_project_source_transitions", "noise_draws", "native_source_restores", "component_hydrations", "cuda_placements", "cpu_offloads"):
        require(counts[key] == declared[key], "actual census differs: " + key)
    for lane in ("forecast", "native"):
        require(counts["gauge_or_sham_acts_per_lane_" + lane] == 12 and counts["spectral_endpoint_acts_per_lane_" + lane] == 10, "actual Act census differs")
    save(output / "state-index.json", {root + "/" + arm: path for (root, arm), path in paths.items()})
    save(output / "native-contract.json", dict(counts=counts, observed=observed, placements=placements, identities_before=identities, identities_after=final_identities,
        conditional_scope=CONDITIONAL_SCOPE, no_modified_native_block=True, source_prefix_calls_before_seal="2Qwen full/LM+4project; not forecast targets", full_microarrays_retained=False))
    save(output / "source.json", dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], sources=sources,
        helper_source_sha256=HELPERS, tokenizer_files=tokenizer_files, tied_alias=tied, parameter_identities=final_identities,
        parameter_recipe_counts=dict(qwen=len(qm["artifacts"]), flux=len(fm["artifacts"]), vae=len(vm["artifacts"])),
        final_resident_parameter_checks="completed before publication/finalizer", forecast_seal=seal["seal_sha256"]))
    result = dict(schema=SCHEMA, job_id=config["job_id"], request_sha256=config["request_sha256"], counts=counts, contract=declared,
        observed=observed, forecast_seal=seal["seal_sha256"], transaction=verify_transaction(output / "forecast-transaction", expected_seal_sha256=seal["seal_sha256"]),
        mechanics_status="completed", instrument_status="selected full states/images plus full boundary hashes/witnesses; complete FF assays;4same-native-final-latent decoder repeats",
        trend_status="uninterpreted observations retained", terminal_status="unassessed", seconds=time.monotonic() - started)
    save(output / "summary.json", result)
    return result


def load_tensor(root, record, device):
    import numpy as np
    import torch

    path = Path(root) / record["path"]
    require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "sealed array changed")
    value = np.load(path, allow_pickle=False)
    require(list(value.shape) == record["shape"] and str(value.dtype) == record.get("stored_dtype", record.get("dtype")), "sealed array geometry/dtype changed")
    if "content_sha256" in record:
        require(hashlib.sha256(value.tobytes()).hexdigest() == record["content_sha256"], "sealed array payload changed")
    tensor = torch.from_numpy(value.copy())
    tensor = tensor.view(torch.bfloat16) if record.get("logical_dtype") == "torch.bfloat16" else tensor
    return tensor.to(device)


def decoder_result(old, root, lane, decoded, processor):
    pixels, image = old["_postprocess_pixels"](processor, decoded)
    base = "decoder/" + lane
    decoded_record = old["_save_tensor"](root, base + "/decoded.npy", decoded)
    pixel_record = old["_save_bytes"](root, base + "/pixels.npy", pixels)
    target = root / base / "image.png"; image.save(target)
    return dict(decoded=decoded_record, pixels=pixel_record, png=dict(path=str(target.relative_to(root)), sha256=sha(target), bytes=target.stat().st_size))


def native_repeat_decode(old, branch_root, vae, native_final, config, pipeline, processor, native_final_cut):
    """Repeat the same native-final latent without colliding with primary files."""
    repeat_root = branch_root / "native-repeat"; repeat_root.mkdir(parents=True, exist_ok=False)
    record = old["_native_decoder"](repeat_root, vae, native_final, config, "same_native_final_latent_repeat", pipeline, processor, {}, {})
    def prefix(value):
        if isinstance(value, dict):
            return {key: "native-repeat/" + item if key == "path" else prefix(item) for key, item in value.items()}
        if isinstance(value, list):
            return [prefix(item) for item in value]
        return value
    return dict(prefix(record), source_native_final_cut=native_final_cut, primary_reference="decoder-native-reference.json",
        control_scope="duplicate original native VAE decode of the exact same native-final state; no forecast latent consumed")


def native_joint0(old, output, store, session, root, arm, receipt):
    """Passive actual FF/R/LN capture only; every hook returns None."""
    model = session.adapter.model; block = model.transformer_blocks[0]; records = {}
    base = f"writer/{root}/native-{arm}"
    def record(name, value):
        records[name] = old["_save_tensor"](output, base + "/" + name + ".npy", value)
    handles = [block.norm2.register_forward_pre_hook(lambda _m, args: record("R_attention", args[0])),
        block.ff.register_forward_pre_hook(lambda _m, args: record("x", args[0])),
        block.ff.register_forward_hook(lambda _m, _args, value: record("M_base", value)),
        block.norm1.register_forward_hook(lambda _m, _args, value: record("image_ln", value)),
        block.norm1_context.register_forward_hook(lambda _m, _args, value: record("text_ln", value))]
    try:
        value = session.continue_()
    finally:
        for handle in reversed(handles):
            handle.remove()
    receipt("native-joint0", root, arm, "native", value)
    cut = session.capture(retain=False); store.save(cut)
    return cut_record(cut), records


def worker(config_path):
    import faulthandler

    faulthandler.enable(); faulthandler.dump_traceback_later(120, repeat=True); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = read(config_path); validate_config(config)
    request_path = os.environ.get("SATURN_DEBUG_REQUEST"); require(request_path, "source-bound MDB request required")
    request = read(request_path)
    identity = hashlib.sha256(json.dumps({key: value for key, value in request.items() if key != "request_sha256"}, sort_keys=True, allow_nan=False).encode()).hexdigest()
    require(identity == request.get("request_sha256") and request.get("config") == config, "request/config identity differs")
    for name, expected in request["source_hashes"].items():
        require(sha(Path(config_path).parent / name) == expected, "executed source changed: " + name)
    job = os.environ["MRUN_JOB_ID"]; require(job.startswith("job-") and "/" not in job, "invalid job identity")
    derived = dict(config, job_id=job, request_sha256=identity, _request_record=request,
        output=str(Path(config["output_scratch"]) / job / config["output_name"]))
    result = run(derived, progress=lambda row: print("MDB_GAUGE_SPECTRAL_PHASE " + canonical(row).decode(), flush=True))
    finalizer = runpy.run_path(str(Path(__file__).with_name("flux_image_tail_microscope.py")))
    ready = finalizer["finalize"](derived["output"], job_id=result["job_id"], request_sha256=identity, **config["transport_limits"])
    print("MDB_DIRECTORY_READY " + canonical(ready).decode(), flush=True); faulthandler.cancel_dump_traceback_later()
    return ready


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default=os.environ.get("SATURN_RUN_CONFIG"), required="SATURN_RUN_CONFIG" not in os.environ)
    worker(parser.parse_args().config)

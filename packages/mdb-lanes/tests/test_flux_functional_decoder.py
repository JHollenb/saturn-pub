"""Tiny installed BF16 decoder preparation, forecast before native comparisons."""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLFlux2

HERE = Path(__file__).parents[1]
G = runpy.run_path(str(HERE / "qualification/flux_functional_decoder.py"))


def fixture(*, attention=True, unequal_channels=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng():
        torch.manual_seed(149)
        channels = (4, 8) if unequal_channels else (4, 4)
        vae = AutoencoderKLFlux2(block_out_channels=channels, decoder_block_out_channels=channels,
            down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
            layers_per_block=1, latent_channels=2, norm_num_groups=1, sample_size=8,
            use_post_quant_conv=True, mid_block_add_attention=attention).eval().to(torch.bfloat16).requires_grad_(False)
        z = torch.randn(1, 2, 4, 4, dtype=torch.bfloat16).transpose(2, 3)
    return vae, z


def descriptor(value):
    raw = value.detach().contiguous().view(torch.uint16).numpy().tobytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))


def metric(p, n):
    a, b = p.float().numpy().astype(np.float64), n.float().numpy().astype(np.float64)
    d = a - b
    return dict(rms=float(np.sqrt(np.mean(d * d))), signed_mean=float(d.mean()), maxabs=float(abs(d).max()),
                exact=descriptor(p) == descriptor(n), bitwise=descriptor(p)["sha256"] == descriptor(n)["sha256"])


def qualify(vae, z, tmp_path):
    before = {name: descriptor(value) for name, value in vae.state_dict().items() if value.dtype == torch.bfloat16}
    input_before = descriptor(z)
    predicted, native, ordinals, counts = {}, {}, {}, {}
    def record(target, name, value):
        key = name + "." + str(ordinals.get(name, 0))
        ordinals[name] = ordinals.get(name, 0) + 1
        target[key] = value.detach().clone(memory_format=torch.preserve_format)
    def observe(name, kind, inputs, output):
        record(predicted, name, output)
    original_call = torch.nn.Module._call_impl
    def forbidden(*args, **kwargs):
        raise AssertionError("functional decoder invoked module.forward")
    torch.nn.Module._call_impl = forbidden
    try:
        output = G["functional_decode"](vae, z, observe=observe, counts=counts, profile="tiny")
    finally:
        torch.nn.Module._call_impl = original_call
    # Native work begins only after the complete functional output exists.
    ordinals.clear()
    handles = []
    active = dict(attention=None, upsample=None)
    for name, module in vae.named_modules():
        if not (name == "post_quant_conv" or name.startswith("decoder")) or type(module) in {torch.nn.ModuleList, torch.nn.Dropout}:
            continue
        if type(module).__name__ == "Attention":
            handles.append(module.register_forward_pre_hook(lambda module, args, name=name: active.update(attention=name)))
        if type(module).__name__ == "Upsample2D":
            handles.append(module.register_forward_pre_hook(lambda module, args, name=name: active.update(upsample=name)))
        handles.append(module.register_forward_hook(lambda module, args, result, name=name: record(native, name, result)))
    original_sdpa, original_interpolate = F.scaled_dot_product_attention, F.interpolate
    def sdpa(*args, **kwargs):
        value = original_sdpa(*args, **kwargs)
        record(native, active["attention"] + ".sdpa", value)
        return value
    def interpolate(*args, **kwargs):
        value = original_interpolate(*args, **kwargs)
        record(native, active["upsample"] + ".interpolate", value)
        return value
    F.scaled_dot_product_attention, F.interpolate = sdpa, interpolate
    try:
        with torch.inference_mode():
            actual = vae.decode(z, return_dict=False)[0]
    finally:
        F.scaled_dot_product_attention, F.interpolate = original_sdpa, original_interpolate
        for handle in handles:
            handle.remove()
    assert set(predicted) == set(native)
    comparisons = {name: metric(predicted[name], native[name]) for name in predicted}
    report = dict(profile=G["PROFILE"], counts=counts, checkpoint_count=len(comparisons), checkpoints=comparisons,
                  output=metric(output, actual), first_discrepancy=next((name for name, row in comparisons.items() if not row["bitwise"]), None),
                  source=G["source_contract"](), native_forward_order="All native comparison calls after functional output completed.",
                  independent_kernel_arithmetic=False, quality_gate=False)
    (tmp_path / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    assert descriptor(z) == input_before
    assert {name: descriptor(value) for name, value in vae.state_dict().items() if value.dtype == torch.bfloat16} == before
    assert output.shape == actual.shape and output.dtype == actual.dtype == torch.bfloat16
    assert counts["decoder"] == 1 and counts["resnet"] == 6 and counts["upsample"] == 1
    # Numeric parity is reported, never used to reject a precision observation.
    assert len(comparisons) > 40
    return report


@pytest.mark.parametrize("attention,unequal", [(True, False), (True, True), (False, True)])
def test_complete_functional_graph_before_native_full_checkpoints(tmp_path, attention, unequal):
    vae, z = fixture(attention=attention, unequal_channels=unequal)
    report = qualify(vae, z, tmp_path)
    assert report["counts"].get("sdpa", 0) == int(attention)
    assert (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize("fault", ["tiling", "slicing", "training", "attention-processor", "dropout", "conditional", "geometry", "input-batch"])
def test_explicit_applicability_refusals(fault):
    vae, z = fixture()
    if fault == "tiling":
        vae.use_tiling = True
    elif fault == "slicing":
        vae.use_slicing = True
    elif fault == "training":
        vae.train()
    elif fault == "attention-processor":
        vae.decoder.mid_block.attentions[0].processor = object()
    elif fault == "dropout":
        vae.decoder.mid_block.resnets[0].dropout.p = .1
    elif fault == "conditional":
        vae.decoder.mid_block.resnets[0].time_embedding_norm = "scale_shift"
    elif fault == "geometry":
        with pytest.raises(ValueError, match="original FLUX2"):
            G["functional_decode"](vae, z)
        return
    else:
        z = z.expand(2, -1, -1, -1)
    with pytest.raises(ValueError):
        G["functional_decode"](vae, z, profile="tiny")


def test_native_source_contract_and_no_ambient_precision_policy_change(tmp_path):
    vae, z = fixture()
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.get_deterministic_debug_mode())
    result = qualify(vae, z, tmp_path)
    assert original == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.get_deterministic_debug_mode())
    assert result["source"]["files"] == G["SOURCE_FILES"]
    assert inspect.getfile(AutoencoderKLFlux2).endswith("autoencoder_kl_flux2.py")


def test_observation_callback_error_restores_forecast_caller_guard():
    vae, z = fixture()
    original = torch.nn.Module._call_impl
    def forbidden(*args, **kwargs):
        raise AssertionError("module forward occurred")
    torch.nn.Module._call_impl = forbidden
    def observe(*args):
        raise RuntimeError("observer interrupted")
    try:
        with pytest.raises(RuntimeError, match="observer interrupted"):
            G["functional_decode"](vae, z, observe=observe, profile="tiny")
        assert torch.nn.Module._call_impl is forbidden
    finally:
        torch.nn.Module._call_impl = original


def test_reuses_prior_equations_without_changing_prior_or_torch_namespaces():
    vae, z = fixture(unequal_channels=True)
    old_path = HERE / "qualification/flux_functional_vae.py"
    before_sha = hashlib.sha256(old_path.read_bytes()).hexdigest()
    old = runpy.run_path(str(old_path))
    old_globals = dict(old["_conv"].__globals__)
    primitives = {name: getattr(F, name) for name in ("conv2d", "group_norm", "silu", "linear", "interpolate", "scaled_dot_product_attention")}
    # Construct the exact upstream packing of the supplied input; default BN
    # statistics widen to identity after the original BF16 scalar operations.
    patched = z.reshape(1, 2, 2, 2, 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 8, 2, 2)
    packed = patched.permute(0, 2, 3, 1).reshape(1, 4, 8)
    ids = torch.tensor([[[0, y, x, 0] for y in range(2) for x in range(2)]], dtype=torch.int64)
    pipe = types.SimpleNamespace(vae=vae, vae_scale_factor=2,
                                 image_processor=types.SimpleNamespace(config=types.SimpleNamespace(do_normalize=True)))
    with torch.inference_mode():
        original = old["decode_functional"](pipe, packed, ids, 8, 8)
        current = G["functional_decode"](vae, z, profile="tiny")
    assert torch.equal(original["latents"], z)
    assert descriptor(original["decoded"]) == descriptor(current)
    assert all(old["_conv"].__globals__[key] is value for key, value in old_globals.items())
    assert all(getattr(F, name) is value for name, value in primitives.items())
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == before_sha == G["PRIOR_HELPER_SHA256"]

"""Framework-free conditional FLUX2 readout and deterministic Euler reference.

Explicit BF16 boundaries describe an arithmetic stream, not exact CUDA kernel
emulation. Only supplied state, parameters and scalar schedule values are read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SOURCE_CONTRACT = {
    "diffusers": "0.39.0", "torch_scalar_qualification": "2.13.0",
    "AdaLayerNormContinuous.forward_source_sha256": "da3e50ebcd024fc2040e7e4afc211c8717142198b743486e507c0770fa4b2909",
    "AdaLayerNormContinuous.module_sha256": "e92ebbb130082578f3cff3361891a1adaa32331347e7bb09e4c70b208a0a6059",
    "FlowMatchEulerDiscreteScheduler.step_source_sha256": "65d8b135d3627d2d4af2e3e84f4bff3ac7d6e6b07f3b9cae96ee8d456eb1a873",
    "FlowMatchEulerDiscreteScheduler.module_sha256": "8b96f25a6170480fb02decb58904e026250d6d68be5de8a7b8785b4c721e9d80",
    "Flux2KleinAdapter.module_sha256": "2004b81971a258f1ca6295fe69d24767dc6bb6b92e103fa30cf6ccb0249b5a86",
    "norm_type": "population LayerNorm, elementwise_affine=False",
    "modulation_order": ["scale", "shift"],
    "flow_profile": "deterministic, scalar 0d FP32 sigmas, no per-token timesteps, noise dtype BF16",
    "scalar_scope": "CPU Torch2.13 hostile scalar fixtures qualified; actual CUDA operator profile must be checked independently by the consumer instrument.",
}


def _source_contract():
    return {key: list(value) if isinstance(value, list) else value for key, value in SOURCE_CONTRACT.items()}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _bf16(value):
    value = np.asarray(value, dtype=np.float32)
    _require(np.isfinite(value).all(), "nonfinite FP32 reference intermediate")
    bits = value.view(np.uint32)
    result = ((bits + np.uint32(32767) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)
    _require(np.isfinite(result).all(), "BF16 reference intermediate overflow")
    return result


def _native(value, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float32 and value.size
             and np.isfinite(value).all()
             and not np.any(np.ascontiguousarray(value).view(np.uint32) & np.uint32(65535)),
             name + " must preserve finite exact native BF16 values widened to FP32")
    return value


def _descriptor(value):
    value = np.ascontiguousarray(value)
    return {"shape": list(value.shape), "dtype": "float32-exact-bf16-widening",
            "fp32_le_sha256": hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest(),
            "bf16_le_sha256": hashlib.sha256((value.view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest()}


def _linear(value, weight, bias, chunk):
    wide = value.astype(np.float64)
    result = np.empty((*value.shape[:-1], weight.shape[0]), np.float32)
    for first in range(0, weight.shape[0], chunk):
        last = min(first + chunk, weight.shape[0])
        part = wide @ weight[first:last].astype(np.float64).T
        if bias is not None:
            part += bias[first:last].astype(np.float64)
        result[..., first:last] = _bf16(part)
    return result


def _silu(value):
    wide = value.astype(np.float64)
    sigmoid = np.empty(wide.shape, np.float64)
    positive = wide >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-wide[positive]))
    negative_exp = np.exp(wide[~positive])
    sigmoid[~positive] = negative_exp / (1 + negative_exp)
    return _bf16(wide * sigmoid)


def predict_readout(image, temb, *, modulation_weight, modulation_bias=None,
                    projection_weight, projection_bias=None, eps, neuron_chunk=256,
                    sample_indices=None, retain_fields=False):
    """Complete AdaLNContinuous(scale,shift) and learned output projection.

    `image` is BF16 native/predicted state widened to FP32, [1,N,H]; `temb`
    is observed upstream time context [1,T]. Parameters are complete actual
    BF16 values widened to FP32. No native module or framework is called.
    """
    image, temb = _native(image, "image"), _native(temb, "temb")
    modulation_weight = _native(modulation_weight, "modulation_weight")
    projection_weight = _native(projection_weight, "projection_weight")
    _require(image.ndim == 3 and image.shape[0] == 1 and min(image.shape[1:]) > 0
             and temb.ndim == 2 and temb.shape[0] == 1 and temb.shape[1] > 0,
             "readout input geometry differs")
    length, width = image.shape[1:]
    _require(modulation_weight.shape == (2 * width, temb.shape[1])
             and projection_weight.ndim == 2 and projection_weight.shape[1] == width
             and projection_weight.shape[0] > 0 and type(eps) in {int, float} and np.isfinite(eps) and eps > 0
             and type(neuron_chunk) is int and 1 <= neuron_chunk <= 4096 and type(retain_fields) is bool,
             "readout parameter/epsilon/chunk geometry differs")
    for name, value, size in (("modulation_bias", modulation_bias, 2 * width),
                              ("projection_bias", projection_bias, projection_weight.shape[0])):
        if value is not None:
            _native(value, name)
            _require(value.shape == (size,), name + " geometry differs")
    indices = (np.unique(np.linspace(0, length - 1, min(8, length)).astype(np.int64)).tolist()
               if sample_indices is None else list(sample_indices))
    _require(indices and len(set(indices)) == len(indices)
             and all(type(i) is int and 0 <= i < length for i in indices), "readout sample indices differ")
    silu_temb = _silu(temb)
    modulation = _linear(silu_temb, modulation_weight, modulation_bias, neuron_chunk)
    scale, shift = np.split(modulation, 2, axis=-1)
    wide = image.astype(np.float64)
    centered = wide - wide.mean(axis=-1, keepdims=True)
    normalized = _bf16(centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps))
    factor = _bf16(np.add(np.float32(1), scale, dtype=np.float32))
    product = _bf16(np.multiply(normalized, factor[:, None], dtype=np.float32))
    modulated = _bf16(np.add(product, shift[:, None], dtype=np.float32))
    noise = _readonly(_linear(modulated, projection_weight, projection_bias, neuron_chunk))
    result = {"noise": noise, "noise_descriptor": _descriptor(noise), "input_descriptor": _descriptor(image),
        "sampled": {"indices": indices, "layernorm": _readonly(normalized[:, indices]),
            "modulated": _readonly(modulated[:, indices]), "noise": _readonly(noise[:, indices]),
            "scale": _readonly(scale), "shift": _readonly(shift), "silu_temb": _readonly(silu_temb)},
        "hidden_width": width, "output_width": projection_weight.shape[0], "sequence_length": length,
        "conditioning_width": temb.shape[1], "eps": float(eps), "neuron_chunk": neuron_chunk,
        "retain_fields": retain_fields, "source_contract": _source_contract(),
        "arithmetic": "B is FP64/FP32→FP32→RNE BF16 widened FP32. B(SiLU(temb)); complete modulation linear FP64 sums+bias→B; split SCALE then SHIFT. Population LayerNorm FP64→B; B(1+scale); B(norm*factor); B(product+shift); complete learned projection FP64 sums+bias→B(noise).",
        "scope": "Conditional full-image readout from supplied image/time context and actual parameters. No native current readout outputs, framework/module calls or RGB prediction; FP64 reference reductions do not emulate exact CUDA kernels."}
    if retain_fields:
        result.update(layernorm=_readonly(normalized), modulated=_readonly(modulated))
    return result


def predict_flow_commit(latent, noise, sigma, sigma_next):
    """Deterministic scalar FlowMatchEuler update with explicit weak dtype.

    A zero-dimensional FP32 dt and a dimensional BF16 noise tensor have BF16
    multiplication dtype under the qualified Torch profile. dt rounds to BF16
    before multiplication. The sample is separately upcast to FP32 for addition.
    """
    latent, noise = _native(latent, "latent"), _native(noise, "noise")
    _require(latent.ndim == 3 and latent.shape[0] == 1 and latent.shape == noise.shape, "flow sample/noise geometry differs")
    scalars = []
    for name, value in (("sigma", sigma), ("sigma_next", sigma_next)):
        raw = np.asarray(value)
        _require(raw.shape == () and raw.dtype.kind == "f" and np.isfinite(raw).all(), name + " must be a finite float scalar")
        number = np.asarray(value, dtype=np.float32)
        _require(np.isfinite(number).all(), name + " overflows FP32")
        scalars.append(number)
    current, following = scalars
    dt = np.subtract(following, current, dtype=np.float32)
    dt_bf16 = _bf16(dt)
    update = _bf16(np.multiply(dt_bf16, noise, dtype=np.float32))
    result = _readonly(_bf16(np.add(latent, update, dtype=np.float32)))
    return {"latent": result, "latent_descriptor": _descriptor(result), "input_descriptor": _descriptor(latent),
        "noise_descriptor": _descriptor(noise), "update_descriptor": _descriptor(update),
        "sigma_fp32": float(current), "sigma_next_fp32": float(following),
        "dt_fp32": float(dt), "dt_bf16": float(dt_bf16),
        "dt_fp32_le_sha256": hashlib.sha256(dt.astype("<f4", copy=False).tobytes()).hexdigest(),
        "source_contract": _source_contract(),
        "adapter_clock_rule": {"step_increment": 1, "phase": "project", "cursor": 0,
            "cleared_working_slots": ["text", "image", "context", "noise_prediction"]},
        "arithmetic": "d=F32(sigma_next-sigma), dB=B(d), update=B(F32(dB*noiseB)), latent_next=B(F32(F32(latentB)+updateB)). The BF16 scalar conversion precedes dense multiplication; no F64 scheduler arithmetic.",
        "scope": "Deterministic scalar FlowMatchEuler commit only; no stochastic noise/per-token timesteps/learned next-step projection. Clock reset and cleared working streams are separate typed-state operations."}


def qualify_torch_scalar_promotion(torch, device="cuda"):
    """Tiny native operator qualification, explicitly separate from forecasts.

    The caller supplies Torch. No model, scheduler, random draw or parameter is
    loaded. A disagreement rejects this numerical ABI, not scientific evidence.
    """
    values = (torch.arange(0x3b00, 0x4300, dtype=torch.int32, device=device) << 16).view(torch.float32).to(torch.bfloat16)
    cases = []
    for number in (-.33333334, -.17320508, .14012345):
        dt = torch.tensor(number, dtype=torch.float32, device=device)
        native = dt * values
        before = dt.to(torch.bfloat16) * values
        wide = (dt * values.float()).to(torch.bfloat16)
        mismatch_before = int((native.view(torch.int16) != before.view(torch.int16)).sum().item())
        mismatch_wide = int((native.view(torch.int16) != wide.view(torch.int16)).sum().item())
        _require(native.dtype == torch.bfloat16 and mismatch_before == 0 and mismatch_wide > 0,
                 "native scalar multiplication does not qualify declared BF16-before-product ABI")
        first = int((before.view(torch.int16) != wide.view(torch.int16)).nonzero()[0].item())
        cases.append({"dt_fp32": float(dt.item()), "dt_bf16": float(dt.to(torch.bfloat16).float().item()),
            "values": int(values.numel()), "native_dtype": str(native.dtype), "dt_device": str(dt.device),
            "noise_device": str(values.device), "native_vs_bf16_dt_mismatches": mismatch_before,
            "native_vs_fp32_dt_mismatches": mismatch_wide, "edge": {"noise": float(values[first].float().item()),
                "native": float(native[first].float().item()), "bf16_dt": float(before[first].float().item()),
                "fp32_dt": float(wide[first].float().item())}})
    return {"schema": "mdb-flux-scalar-promotion-qualification-v1", "qualified": True,
        "torch_version": str(torch.__version__), "device": str(device), "cases": cases,
        "torch_python_module_sha256": hashlib.sha256(Path(torch.__file__).read_bytes()).hexdigest(),
        "source_contract": _source_contract(), "model_calls": 0,
        "scope": "Native device operator fixture only. Establishes scalar/dense multiplication dtype and rounding for these hostile cases, not full scheduler/CUDA kernel equivalence."}


__all__ = ["SOURCE_CONTRACT", "predict_readout", "predict_flow_commit", "qualify_torch_scalar_promotion"]

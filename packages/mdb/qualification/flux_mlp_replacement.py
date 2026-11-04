"""Full image-side entry-state MLP reference and sealed native-consumer lowering.

Prediction uses only NumPy and incoming state/modulation/weights. Native
consumer functions are separate and used only after forecast sealing. They
still execute the original MLP before replacing its image-side output slice.
"""
from __future__ import annotations

import runpy

QUERY_INDICES = (0, 31, 32, 255, 511, 512, 767, 1023)
NEURON_INDICES = tuple(range(0, 9216, 288))
BRANCHES = [
    {"id": "zero", "base": "zero", "feature": "native", "source": "zero", "roll": 0},
    {"id": "text", "base": "text", "feature": "native", "source": "text", "roll": 0},
    {"id": "sham", "base": "zero", "feature": "measured", "source": "zero", "roll": 0},
    {"id": "low.predicted", "base": "zero", "feature": "predicted", "source": "zero", "roll": 0},
    {"id": "M.measured", "base": "zero", "feature": "measured", "source": "text", "roll": 0},
    {"id": "M.predicted", "base": "zero", "feature": "predicted", "source": "text", "roll": 0},
    {"id": "M.measured.roll", "base": "zero", "feature": "measured", "source": "text", "roll": 512},
    {"id": "M.predicted.roll", "base": "zero", "feature": "predicted", "source": "text", "roll": 512},
    {"id": "high.predicted", "base": "text", "feature": "predicted", "source": "text", "roll": 0}]


def _bf16(value):
    import numpy as np

    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)


def _silu(value):
    import numpy as np

    value = np.asarray(value, dtype=np.float64)
    decay = np.exp(-np.abs(value))
    return np.where(value >= 0, value / (1 + decay), value * decay / (1 + decay))


def _readonly(value):
    import numpy as np

    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def predict_mlp_field(residual_image, single_mod, weight, bias=None, *, eps, hidden_width=None, mlp_width=None,
        neuron_chunk=256, query_indices=QUERY_INDICES, neuron_indices=NEURON_INDICES):
    """Return the complete explicit BF16-reference image MLP activation field.

    ``weight`` is [2I,H], first the complete u-half then the complete v-half,
    extracted from actual fused parameter rows 3H:3H+2I. Native BF16 values
    enter as FP32 widenings. No native norm, MLP or attention output is an input.
    Matrix widening is bounded to two ``neuron_chunk`` row tiles. The full
    analytic FP64 MLP field is never allocated or returned.
    """
    import numpy as np

    if (not isinstance(residual_image, np.ndarray) or residual_image.ndim != 3 or residual_image.shape[0] != 1
            or not isinstance(weight, np.ndarray) or weight.ndim != 2 or weight.shape[0] % 2):
        raise ValueError("invalid full MLP entry/weight geometry")
    h = residual_image.shape[-1]
    i = weight.shape[0] // 2
    if ((hidden_width is not None and hidden_width != h) or (mlp_width is not None and mlp_width != i)
            or h < 1 or i < 1 or residual_image.shape[1] < 1 or weight.shape[1] != h
            or not isinstance(single_mod, np.ndarray) or single_mod.shape not in {(1, 3 * h), (1, 1, 3 * h)}
            or (bias is not None and (not isinstance(bias, np.ndarray) or bias.shape != (2 * i,)))
            or type(neuron_chunk) is not int or not 1 <= neuron_chunk <= 4096
            or type(eps) not in {int, float} or not np.isfinite(eps) or eps <= 0
            or not query_indices or not neuron_indices or len(set(query_indices)) != len(query_indices)
            or len(set(neuron_indices)) != len(neuron_indices)
            or any(type(index) is not int or not 0 <= index < residual_image.shape[1] for index in query_indices)
            or any(type(index) is not int or not 0 <= index < i for index in neuron_indices)
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (residual_image, single_mod, weight))
            or (bias is not None and (bias.dtype != np.float32 or not np.isfinite(bias).all()))):
        raise ValueError("invalid full MLP source/dtype/epsilon/chunk/sample contract")
    x = np.asarray(residual_image, dtype=np.float64)
    shift, scale, _gate = np.split(np.asarray(single_mod, dtype=np.float64).reshape(1, 1, 3 * h), 3, axis=-1)
    centered = x - x.mean(axis=-1, keepdims=True)
    norm = _bf16(centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps))
    factor = _bf16(1 + scale)
    modulated = _bf16(_bf16(factor * norm).astype(np.float64) + shift)
    del x, centered
    if not np.isfinite(modulated).all():
        raise ValueError("nonfinite full MLP normalized/modulated reference")
    mod64 = modulated.astype(np.float64)
    output = np.empty((1, residual_image.shape[1], i), dtype=np.float32)
    sampled = {"query_indices": list(query_indices), "neuron_indices": list(neuron_indices),
        "layernorm": _readonly(norm[:, query_indices]), "modulated": _readonly(modulated[:, query_indices])}
    sample_arrays = {name: np.empty((1, len(query_indices), len(neuron_indices)), dtype=np.float32) for name in ("u", "v", "silu_u", "activation")}
    for start in range(0, i, neuron_chunk):
        end = min(i, start + neuron_chunk)
        u = mod64 @ np.asarray(weight[start:end], dtype=np.float64).T
        v = mod64 @ np.asarray(weight[i + start:i + end], dtype=np.float64).T
        if bias is not None:
            u += np.asarray(bias[start:end], dtype=np.float64)
            v += np.asarray(bias[i + start:i + end], dtype=np.float64)
        ub, vb = _bf16(u), _bf16(v)
        silu = _bf16(_silu(ub))
        activated = _bf16(silu * vb)
        if any(not np.isfinite(value).all() for value in (ub, vb, silu, activated)):
            raise ValueError("nonfinite full MLP activation reference")
        output[:, :, start:end] = activated
        selected = [(target, neuron - start) for target, neuron in enumerate(neuron_indices) if start <= neuron < end]
        for target, column in selected:
            for name, value in (("u", ub), ("v", vb), ("silu_u", silu), ("activation", activated)):
                sample_arrays[name][:, :, target] = value[:, query_indices, column]
    sampled.update({name: _readonly(value) for name, value in sample_arrays.items()})
    return {"activation": _readonly(output), "sampled": sampled, "hidden_width": h, "mlp_width": i,
        "neuron_chunk": neuron_chunk,
        "arithmetic": "Full image population-variance LayerNorm FP64(no affine, bound eps)→B; B(1+scale); B(product); B(+shift). Full paired actual BF16-weight rows widened FP32 then tiled FP64 sums+bias→B(u/v); B(SiLU(Bu)); B(product). B is FP64→FP32→RNE-BF16 widened FP32. Explicit rounding reference, not a bit-exact CUDA kernel emulator.",
        "scope": "Complete image-side MLP field conditional on incoming state/modulation/full paired weights. No native current-site activations enter prediction; no full analytic FP64 field retained; final RGB and continuations are native observations, not predicted."}


def prepare_mlp_payload(prediction, reference, *, roll=0):
    """Lower sealed FP32 BF16 values to the native operand layout, losslessly.

    ``reference`` supplies only shape/dtype/device/strides. Its values do not
    contribute to the payload. Rolling uses the image-token axis (axis 1).
    """
    import numpy as np
    import torch

    if (not isinstance(prediction, np.ndarray) or prediction.dtype != np.float32 or prediction.ndim != 3
            or prediction.shape[0] != 1 or not isinstance(reference, torch.Tensor) or reference.dtype != torch.bfloat16
            or tuple(reference.shape) != prediction.shape or type(roll) is not int or not 0 <= roll < prediction.shape[1]
            or not np.isfinite(prediction).all() or np.any(prediction.view(np.uint32) & np.uint32(0xffff))):
        raise ValueError("sealed MLP payload requires finite FP32 BF16 bits matching native image operand")
    values = np.roll(prediction, roll, axis=1) if roll else prediction
    with torch.inference_mode():
        tensor = torch.from_numpy(np.array(values, copy=True)).to(device=reference.device, dtype=torch.bfloat16)
        payload = torch.empty_strided(reference.shape, reference.stride(), dtype=reference.dtype, device=reference.device)
        payload.copy_(tensor)
        if not np.array_equal(payload.float().cpu().numpy().view(np.uint32), values.view(np.uint32)):
            raise ValueError("native MLP payload cast changed sealed BF16 values")
    return payload


def capture_replacement(block, cut, recipient_native, mlp_payload, *, source_parent=None, single_index=6,
        capture_path="flux_single_contraction.py"):
    """Execute native block with only its image MLP slice replaced after seal.

    Native attention and MLP still run. The recipient's native concat supplies
    unchanged image attention and text operands. No skipping or speedup claim.
    """
    import torch

    if type(single_index) is not int or single_index != 6:
        raise ValueError("MLP replacement is qualified only for declared single6")
    ports = recipient_native["ports"]
    original = ports["original_concat"]
    t, h = recipient_native["text_rows"], recipient_native["attention_width"]
    reference = original[:, t:, h:]
    if (recipient_native["input_parent"] != cut.fingerprint or not isinstance(mlp_payload, torch.Tensor)
            or mlp_payload.shape != reference.shape or mlp_payload.dtype != reference.dtype or mlp_payload.device != reference.device
            or not bool(torch.isfinite(mlp_payload).all())):
        raise ValueError("MLP replacement native parent/shape/dtype/device/finite differs")
    with torch.inference_mode():
        donor = original.detach().clone()
        donor[:, t:, h:] = mlp_payload
    value = runpy.run_path(capture_path)["capture_candidate"](block, cut,
        source={"parent": source_parent or cut.fingerprint, "concat": donor, "text_rows": t, "attention_width": h}, selection="M")
    used, native = value["ports"]["used_concat"], value["ports"]["original_concat"]
    checks = {"text_concat_exact_recipient": torch.equal(used[:, :t], native[:, :t]),
        "image_attention_exact_recipient": torch.equal(used[:, t:, :h], native[:, t:, :h]),
        "image_mlp_exact_payload": torch.equal(used[:, t:, h:], mlp_payload),
        "native_original_concat_exact_recipient": torch.equal(value["ports"]["original_concat"], original),
        "post_text_exact_recipient_native": torch.equal(value["text"], recipient_native["text"])}
    if not all(checks[key] for key in ("text_concat_exact_recipient", "image_attention_exact_recipient", "image_mlp_exact_payload")):
        raise ValueError("MLP replacement violated the declared native operand footprint")
    value["replacement_controls"] = checks
    return value


__all__ = ["BRANCHES", "QUERY_INDICES", "NEURON_INDICES", "predict_mlp_field", "prepare_mlp_payload", "capture_replacement"]

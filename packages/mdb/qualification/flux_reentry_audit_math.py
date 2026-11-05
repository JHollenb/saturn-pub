"""Independent saved-only project/joint recurrence; no campaign predictor import.

The complete operands are parameters and predecessor predictions. Native target
activations are never inputs. Double precision sums are a mathematical reference,
with each declared BF16 operation rounded separately; this is not CUDA emulation.
"""
from __future__ import annotations

import numpy as np


def B(value):
    value = np.asarray(value, np.float32)
    if not value.size or not np.isfinite(value).all():
        raise ValueError("nonfinite independent operand")
    words = np.ascontiguousarray(value).view(np.uint32)
    result = ((words + 0x7fff + ((words >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32).reshape(value.shape)
    if not np.isfinite(result).all():
        raise ValueError("independent BF16 overflow")
    return result


def _linear(value, parameters, prefix):
    weight = np.asarray(parameters[prefix + ".weight"])
    bias = parameters.get(prefix + ".bias")
    if value.shape[-1] != weight.shape[1]:
        raise ValueError("independent projection width differs")
    result = np.empty((*value.shape[:-1], weight.shape[0]), np.float32)
    x = value.astype(np.float64)
    for start in range(0, weight.shape[0], 128):
        stop = min(start + 128, weight.shape[0])
        y = x @ weight[start:stop].astype(np.float64).T
        if bias is not None:
            y += bias[start:stop].astype(np.float64)
        result[..., start:stop] = B(y)
    return result


def _silu(value):
    x = value.astype(np.float64)
    exp = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1 / (1 + exp), exp / (1 + exp))
    return B(x * sigmoid)


def _normalize(value, shift, scale, eps):
    wide = value.astype(np.float64)
    centered = wide - np.mean(wide, axis=-1, keepdims=True)
    normalized = B(centered / np.sqrt(np.mean(np.square(centered), axis=-1, keepdims=True) + eps))
    return B(B(normalized * B(1 + scale)) + shift)


def independent_project(latent, conditioning, img_ids, txt_ids, timestep, parameters,
                        *, axes_dims_rope, rope_theta, timestep_channels):
    """Rebuild time, context and both streams from the incoming predicted latent."""
    scalar = B(B(B(np.asarray(timestep, np.float32).reshape(1)) / np.float32(1000)) * np.float32(1000))
    half = timestep_channels // 2
    log_frequency = np.asarray(-np.log(10000.0), np.float32) * np.arange(half, dtype=np.float32)
    log_frequency = np.asarray(log_frequency / np.float32(half), np.float32)
    frequency = np.exp(log_frequency.astype(np.float64)).astype(np.float32)
    angles = np.asarray(scalar[:, None] * frequency[None], np.float32)
    encoding = np.concatenate([np.cos(angles.astype(np.float64)).astype(np.float32),
        np.sin(angles.astype(np.float64)).astype(np.float32)], axis=-1)
    if timestep_channels % 2:
        encoding = np.pad(encoding, ((0, 0), (0, 1)))
    time_prefix = "time_guidance_embed.timestep_embedder"
    temb = _linear(_silu(_linear(B(encoding), parameters, time_prefix + ".linear_1")),
        parameters, time_prefix + ".linear_2")
    activated = _silu(temb)
    context = {"temb": temb}
    for port, prefix in (("img_mod", "double_stream_modulation_img.linear"),
            ("txt_mod", "double_stream_modulation_txt.linear"), ("single_mod", "single_stream_modulation.linear")):
        context[port] = _linear(activated, parameters, prefix)
    # Concatenate text IDs before image IDs. Each axis uses repeated adjacent pairs.
    positions = []
    for ids in (txt_ids, img_ids):
        array = np.asarray(ids)
        if array.ndim == 3:
            array = array[0]
        positions.append(array.astype(np.float32).astype(np.float64))
    positions = np.concatenate(positions, axis=0)
    cosine, sine = [], []
    for axis, dimension in enumerate(axes_dims_rope):
        exponents = np.arange(0, dimension, 2, dtype=np.float64) / dimension
        angle = positions[:, axis, None] / np.power(float(rope_theta), exponents)[None]
        cosine.append(np.repeat(np.cos(angle), 2, axis=1).astype(np.float32))
        sine.append(np.repeat(np.sin(angle), 2, axis=1).astype(np.float32))
    context["rope"] = (np.concatenate(cosine, axis=1), np.concatenate(sine, axis=1))
    return {"image": _linear(latent, parameters, "x_embedder"),
        "text": _linear(conditioning, parameters, "context_embedder"), "context": context}


def independent_joint(image, text, context, parameters, *, heads, layernorm_eps, qk_eps):
    """Serial attention then receiving-state MLP for each residual stream."""
    width = image.shape[-1]
    dimension = width // heads
    mods = [np.split(context[name].reshape(1, 1, 6 * width), 6, axis=-1) for name in ("img_mod", "txt_mod")]
    residuals = [image, text]
    queries, keys, values = [], [], []
    for stream, modulation, projections, norms in (
            (text, mods[1], ("add_q_proj", "add_k_proj", "add_v_proj"), ("norm_added_q", "norm_added_k")),
            (image, mods[0], ("to_q", "to_k", "to_v"), ("norm_q", "norm_k"))):
        normalized = _normalize(stream, modulation[0], modulation[1], layernorm_eps)
        vectors = [_linear(normalized, parameters, "attn." + name).reshape(1, stream.shape[1], heads, dimension)
            for name in projections]
        for index, name in enumerate(norms):
            x = vectors[index].astype(np.float64)
            inverse = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + qk_eps)
            vectors[index] = B(x * inverse * parameters["attn." + name + ".weight"].astype(np.float64))
        queries.append(vectors[0])
        keys.append(vectors[1])
        values.append(vectors[2])
    q, k, v = [np.concatenate(parts, axis=1) for parts in (queries, keys, values)]
    cos, sin = [x[None, :, None] for x in context["rope"]]
    def rotary(x):
        swapped = np.stack((-x[..., 1::2], x[..., 0::2]), axis=-1).reshape(x.shape)
        return B(np.asarray(x * cos, np.float32) + np.asarray(swapped * sin, np.float32))
    q, k = rotary(q), rotary(k)
    k = k.astype(np.float64).transpose(0, 2, 3, 1)
    v = v.astype(np.float64).transpose(0, 2, 1, 3)
    attention = np.empty((1, q.shape[1], width), np.float32)
    for start in range(0, q.shape[1], 32):
        stop = min(start + 32, q.shape[1])
        query = q[:, start:stop].astype(np.float64).transpose(0, 2, 1, 3)
        logits = (query @ k) / np.sqrt(float(dimension))
        exponential = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probability = exponential / exponential.sum(axis=-1, keepdims=True)
        attention[:, start:stop] = B((probability @ v).transpose(0, 2, 1, 3).reshape(1, stop - start, width))
    text_attention, image_attention = np.split(attention, [text.shape[1]], axis=1)
    output = {}
    for name, residual, modulation, attended, projection, mlp in (
            ("image", residuals[0], mods[0], image_attention, "attn.to_out.0", "ff"),
            ("text", residuals[1], mods[1], text_attention, "attn.to_add_out", "ff_context")):
        after_attention = B(residual + B(modulation[2] * _linear(attended, parameters, projection)))
        receiving = _normalize(after_attention, modulation[3], modulation[4], layernorm_eps)
        gate, value = np.split(_linear(receiving, parameters, mlp + ".linear_in"), 2, axis=-1)
        written = _linear(B(_silu(gate) * value), parameters, mlp + ".linear_out")
        output[name] = B(after_attention + B(modulation[5] * written))
    return output

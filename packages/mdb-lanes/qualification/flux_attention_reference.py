"""Entry-state-only NumPy reference for the full image attention field.

The incoming text/image register, time modulation, weights and upstream RoPE
are inputs. No native current-block Q/K/V, normalization or attention result
is accepted. BF16 boundaries are explicit; CUDA reductions are not emulated.
"""
from __future__ import annotations

QUERY_INDICES = (0, 31, 32, 255, 511, 512, 767, 1023)


def _bf16(value):
    import numpy as np

    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)).view(np.float32)


def _readonly(value):
    import numpy as np

    value = np.asarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def predict_attention_field(residual, single_mod, qkv_weight, qkv_bias=None, *, text_rows, heads,
        norm_q_weight, norm_k_weight, layernorm_eps, qk_eps, rope_cos, rope_sin,
        query_chunk=64, projection_chunk=256, query_indices=QUERY_INDICES, key_indices=None):
    """Predict image A before the fused attention/MLP output projection.

    Inputs are finite FP32 widenings of native BF16 values, except FP32 RoPE.
    ``residual`` is [1,T+N,H]; QKV rows [3H,H] are the first rows of the actual
    fused parameter, ordered Q,K,V. Learned Q/K RMSNorm weights are [H/heads].
    RoPE cos/sin are the observed upstream [T+N,H/heads] tensors. Attention is
    unmasked, noncausal, dropout-free, with scale 1/sqrt(head_dim).

    Only complete BF16-reference A is retained, widened to FP32. FP64 score
    matrices are bounded by ``query_chunk`` image queries; projection weight
    widening is bounded by ``projection_chunk`` output rows. All keys/values,
    including the text bank, participate in every image query.
    """
    import numpy as np

    if (not isinstance(residual, np.ndarray) or residual.ndim != 3 or residual.shape[0] != 1
            or type(heads) is not int or heads < 1 or type(text_rows) is not int
            or not 0 < text_rows < residual.shape[1]):
        raise ValueError("invalid concatenated text/image attention source")
    _, length, width = residual.shape
    if width < 1 or width % heads or (width // heads) % 2:
        raise ValueError("attention heads require a positive even RoPE width")
    dim, image_rows = width // heads, length - text_rows
    if key_indices is None:
        key_indices = list(dict.fromkeys((0, text_rows - 1, text_rows, length - 1)))
    if (not isinstance(single_mod, np.ndarray) or single_mod.shape not in {(1, 3 * width), (1, 1, 3 * width)}
            or not isinstance(qkv_weight, np.ndarray) or qkv_weight.shape != (3 * width, width)
            or (qkv_bias is not None and (not isinstance(qkv_bias, np.ndarray) or qkv_bias.shape != (3 * width,)))
            or any(not isinstance(value, np.ndarray) or value.shape != (dim,) for value in (norm_q_weight, norm_k_weight))
            or any(not isinstance(value, np.ndarray) or value.shape != (length, dim) for value in (rope_cos, rope_sin))
            or type(query_chunk) is not int or not 1 <= query_chunk <= 128
            or type(projection_chunk) is not int or not 1 <= projection_chunk <= 4096
            or any(type(eps) not in {float, int} or not np.isfinite(eps) or eps <= 0 for eps in (layernorm_eps, qk_eps))
            or not query_indices or len(set(query_indices)) != len(query_indices)
            or any(type(index) is not int or not 0 <= index < image_rows for index in query_indices)
            or not key_indices or len(set(key_indices)) != len(key_indices)
            or any(type(index) is not int or not 0 <= index < length for index in key_indices)):
        raise ValueError("invalid attention geometry/weights/RoPE/epsilon/chunk/sample contract")
    sources = (residual, single_mod, qkv_weight, norm_q_weight, norm_k_weight, rope_cos, rope_sin)
    if qkv_bias is not None:
        sources += (qkv_bias,)
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in sources):
        raise ValueError("attention sources must be finite FP32 native widenings")
    # Source BF16 tensors and FP32 trigonometric tensors have different ABIs.
    bf16_sources = (residual, single_mod, qkv_weight, norm_q_weight, norm_k_weight)
    if qkv_bias is not None:
        bf16_sources += (qkv_bias,)
    if any(np.any(np.ascontiguousarray(value).view(np.uint32) & np.uint32(0xffff)) for value in bf16_sources):
        raise ValueError("native BF16 sources must have exact widened BF16 bits")

    x = residual.astype(np.float64)
    shift, scale, _gate = np.split(single_mod.astype(np.float64).reshape(1, 1, 3 * width), 3, axis=-1)
    centered = x - x.mean(axis=-1, keepdims=True)
    norm = _bf16(centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + layernorm_eps))
    del x, centered
    modulated = _bf16(_bf16(_bf16(1 + scale) * norm).astype(np.float64) + shift)
    mod64 = modulated.astype(np.float64)
    projected = np.empty((1, length, 3 * width), dtype=np.float32)
    for start in range(0, 3 * width, projection_chunk):
        stop = min(start + projection_chunk, 3 * width)
        tile = mod64 @ qkv_weight[start:stop].astype(np.float64).T
        if qkv_bias is not None:
            tile += qkv_bias[start:stop].astype(np.float64)
        projected[:, :, start:stop] = _bf16(tile)
    del mod64
    q, k, v = (value.reshape(1, length, heads, dim) for value in np.split(projected, 3, axis=-1))

    def rms(value, weight):
        wide = value.astype(np.float64)
        return _bf16((wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + qk_eps)) * weight.astype(np.float64))

    def rotate(value):
        # Installed apply_rotary_emb: x.float()*cos + rotate(x).float()*sin.
        rotated = np.empty_like(value)
        rotated[..., 0::2], rotated[..., 1::2] = -value[..., 1::2], value[..., 0::2]
        return _bf16(value * rope_cos[None, :, None, :] + rotated * rope_sin[None, :, None, :])

    q_norm, k_norm = rms(q, norm_q_weight), rms(k, norm_k_weight)
    q_rope, k_rope = rotate(q_norm), rotate(k_norm)
    if any(not np.isfinite(value).all() for value in (norm, modulated, projected, q_norm, k_norm, q_rope, k_rope)):
        raise ValueError("nonfinite attention intermediate reference")
    positions = [text_rows + index for index in query_indices]
    sampled = {"query_indices": list(query_indices), "input": _readonly(residual[:, positions]),
        "layernorm": _readonly(norm[:, positions]),
        "modulated": _readonly(modulated[:, positions]), "q_raw": _readonly(q[:, positions]),
        "k_raw": _readonly(k[:, positions]), "v": _readonly(v[:, positions]),
        "q_norm": _readonly(q_norm[:, positions]), "k_norm": _readonly(k_norm[:, positions]),
        "q_rope": _readonly(q_rope[:, positions]), "k_rope": _readonly(k_rope[:, positions])}
    sampled["key_indices"] = list(key_indices)
    sampled["key_samples"] = {name: _readonly(value[:, key_indices]) for name, value in (
        ("input", residual), ("layernorm", norm), ("modulated", modulated), ("q_raw", q), ("k_raw", k),
        ("v", v), ("q_norm", q_norm), ("k_norm", k_norm), ("q_rope", q_rope), ("k_rope", k_rope))}
    del norm, modulated, q_norm, k_norm
    keys = k_rope.astype(np.float64).transpose(0, 2, 3, 1)
    values = v.astype(np.float64).transpose(0, 2, 1, 3)
    output = np.empty((1, image_rows, width), dtype=np.float32)
    mass = np.empty((1, len(query_indices), heads), dtype=np.float64)
    image_mass = np.empty_like(mass)
    analytic = np.empty((1, len(query_indices), width), dtype=np.float64)
    for start in range(0, image_rows, query_chunk):
        stop = min(start + query_chunk, image_rows)
        queries = q_rope[:, text_rows + start:text_rows + stop].astype(np.float64).transpose(0, 2, 1, 3)
        scores = np.matmul(queries, keys) / np.sqrt(float(dim))
        scores -= scores.max(axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attended = np.matmul(probabilities, values).transpose(0, 2, 1, 3).reshape(1, stop - start, width)
        output[:, start:stop] = _bf16(attended)
        for target, index in enumerate(query_indices):
            if start <= index < stop:
                analytic[:, target] = attended[:, index - start]
                mass[:, target] = probabilities[:, :, index - start, :text_rows].sum(axis=-1)
                image_mass[:, target] = probabilities[:, :, index - start, text_rows:].sum(axis=-1)
    if not np.isfinite(output).all():
        raise ValueError("nonfinite full image attention reference")
    sampled.update(attention=_readonly(output[:, query_indices]), attention_fp64=_readonly(analytic),
        text_mass=_readonly(mass), image_mass=_readonly(image_mass))
    return {"attention": _readonly(output), "sampled": sampled, "hidden_width": width, "heads": heads,
        "head_dim": dim, "text_rows": text_rows, "image_rows": image_rows, "query_chunk": query_chunk,
        "projection_chunk": projection_chunk, "layernorm_eps": float(layernorm_eps), "qk_eps": float(qk_eps),
        "arithmetic": "B=FP64/FP32→FP32→RNE BF16 widened FP32. Population LayerNorm FP64(no affine,bound epsilon)→B; B(1+scale); B(product); B(+shift). Actual fused first3H rows tiled FP64 linear+bias→B(Q,K,V). Learned Q/K RMSNorm: FP64 mean-square+bound epsilon, multiply actual weight, then B. RoPE interleaved(-odd,even) with separate FP32 multiplies and FP32 add then B, as installed apply_rotary_emb. Head-transposed batched FP64 matmul QK/sqrt(head_dim), softmax on the complete text+image key axis, then batched FP64 matmul with V, noncausal/unmasked/dropout-free; B of complete image attention. This declares a reference arithmetic stream, not exact CUDA LayerNorm/RMSNorm/GEMM/SDPA emulation.",
        "scope": "Full image-side attention field conditional on incoming concatenated native state, time modulation, actual weights and upstream positional cos/sin. No current-target native Q/K/V/norm/attention output inputs; no native/module/framework calls; downstream RGB remains an observed native future."}


__all__ = ["QUERY_INDICES", "predict_attention_field"]

"""Framework-free FLUX2 project and first joint-block reference.

Only incoming latent/conditioning/IDs/schedule and explicit native parameters
are accepted. Native time contexts, RoPE and joint activations are not inputs.
BF16 boundaries and FP32 time/RoPE arithmetic are explicit; FP64 reductions
are mathematical references, not emulation of CUDA linear/SDPA kernels.
"""
from __future__ import annotations

import hashlib

import numpy as np

SOURCE_CONTRACT = {
    "diffusers_version": "0.39.0",
    "torch_qualification_version": "2.13.0",
    "transformer_module_sha256": "0a3c56ed3ca28a416492b50953a433f20b8a89fd26c7b7a49b1b6b45d5236006",
    "embeddings_module_sha256": "4eb810f715786eb1f24f2a4641e529817f7c951b9caf2f7925811329a03ec796",
    "adapter_module_sha256": "2004b81971a258f1ca6295fe69d24767dc6bb6b92e103fa30cf6ccb0249b5a86",
    "Flux2TransformerBlock.source_sha256": "84dcf1d849e087a2993ebd76e8b8388dee3095be7457da99ca0b12e3022c4114",
    "Flux2AttnProcessor.source_sha256": "d578e72ae2230016bb230258cd86efa9e12815d96aa50c8d5bf0523b7a110c6e",
    "Flux2PosEmbed.source_sha256": "1364fd0183fa80d5f323d77ce437154131efda6eaf7d8871eb2c446f53e78213",
    "get_timestep_embedding.source_sha256": "6028d48c06ba9784cb4382279653485b220c9219f3ff4cdfd503a938004a1ec2",
    "get_1d_rotary_pos_embed.source_sha256": "0734204c538630d0ea99be01231bdcd3cf806a5970b9c6fca2da11a50ddbdfdf",
    "apply_rotary_emb.source_sha256": "a25267d425f8c5eaa84151b6914e988cdd0ad0b857bb74495477a3644c819164",
}
PROJECT_LINEAR_NAMES = ("x_embedder", "context_embedder", "time_guidance_embed.timestep_embedder.linear_1",
    "time_guidance_embed.timestep_embedder.linear_2", "double_stream_modulation_img.linear",
    "double_stream_modulation_txt.linear", "single_stream_modulation.linear")
JOINT_LINEAR_NAMES = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj",
    "attn.add_v_proj", "attn.to_out.0", "attn.to_add_out", "ff.linear_in", "ff.linear_out",
    "ff_context.linear_in", "ff_context.linear_out")
JOINT_NORM_NAMES = ("attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_q.weight", "attn.norm_added_k.weight")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _readonly(value):
    x = np.asarray(value)
    return np.frombuffer(x.tobytes(), dtype=x.dtype).reshape(x.shape)


def bf16(value):
    x = np.asarray(value, dtype=np.float32)
    _require(x.size and np.isfinite(x).all(), "nonfinite/empty BF16 intermediate")
    bits = np.ascontiguousarray(x).view(np.uint32)
    out = ((bits + np.uint32(32767) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32).reshape(x.shape)
    _require(np.isfinite(out).all(), "BF16 rounding overflow")
    return out


def _native(value):
    x = np.asarray(value)
    _require(x.dtype == np.float32 and x.size and np.isfinite(x).all()
        and not np.any(np.ascontiguousarray(x).view(np.uint32) & 65535), "expected exact finite BF16 widening")
    return x


def descriptor(value):
    x = _native(value)
    return {"shape": list(x.shape), "dtype": "float32-exact-bf16-widening",
        "fp32_le_sha256": hashlib.sha256(np.asarray(x, dtype="<f4").tobytes()).hexdigest(),
        "bf16_le_sha256": hashlib.sha256((np.ascontiguousarray(x).view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest()}


def _chunks(*values):
    _require(all(type(v) is int and 1 <= v <= 4096 for v in values), "invalid bounded chunk size")


def _weights(weights, linears, norms=()):
    required = {name + ".weight" for name in linears} | set(norms)
    optional = {name + ".bias" for name in linears}
    _require(isinstance(weights, dict) and required <= weights.keys() and not weights.keys() - required - optional,
        "native parameter inventory differs")
    for key, value in weights.items():
        if value is None:
            _require(key in optional, "only missing biases may be None")
        else:
            _native(value)


def _linear(value, weights, name, chunk):
    x, w = _native(value), _native(weights[name + ".weight"])
    b = weights.get(name + ".bias")
    _require(w.ndim == 2 and x.shape[-1] == w.shape[1], "linear geometry differs: " + name)
    if b is not None:
        _require(b.shape == (w.shape[0],), "linear bias geometry differs")
    wide = x.astype(np.float64)
    out = np.empty((*x.shape[:-1], w.shape[0]), np.float32)
    for first in range(0, w.shape[0], chunk):
        last = min(first + chunk, w.shape[0])
        projected = wide @ w[first:last].astype(np.float64).T
        if b is not None:
            projected += b[first:last].astype(np.float64)
        out[..., first:last] = bf16(projected)
    return out


def _silu(value):
    x = np.asarray(value, dtype=np.float64)
    probability = np.empty_like(x)
    positive = x >= 0
    probability[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    probability[~positive] = exp / (1 + exp)
    return bf16(x * probability)


def _ln(value, eps):
    x = _native(value).astype(np.float64)
    centered = x - x.mean(axis=-1, keepdims=True)
    return bf16(centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps))


def _modulate(value, shift, scale, eps):
    return bf16(bf16(_ln(value, eps) * bf16(1 + scale)) + shift)


def predict_timestep(timestep, channels=256):
    """Native adapter B(t), B(t/1000), B(*1000), then F32 Timesteps."""
    raw = np.asarray(timestep)
    _require(raw.shape in {(), (1,)} and raw.dtype == np.float32 and np.isfinite(raw).all()
        and type(channels) is int and channels >= 2 and channels <= 4096,
        "timestep must be one actual finite FP32 schedule scalar")
    t = bf16(raw.reshape(1))
    divided = bf16(np.divide(t, np.float32(1000), dtype=np.float32))
    returned = bf16(np.multiply(divided, np.float32(1000), dtype=np.float32))
    half = channels // 2
    exponent = np.multiply(np.float32(-np.log(10000.0)), np.arange(half, dtype=np.float32), dtype=np.float32)
    exponent = np.divide(exponent, np.float32(half), dtype=np.float32)
    frequencies = np.exp(exponent.astype(np.float64)).astype(np.float32)
    angle = np.multiply(returned[:, None], frequencies[None, :], dtype=np.float32)
    # Timesteps has flip_sin_to_cos=True and downscale_freq_shift=0.
    embedded = np.concatenate((np.cos(angle.astype(np.float64)).astype(np.float32),
        np.sin(angle.astype(np.float64)).astype(np.float32)), axis=-1)
    if channels % 2:
        embedded = np.pad(embedded, ((0, 0), (0, 1)))
    return {"native_timestep": _readonly(t), "divided_timestep": _readonly(divided),
        "roundtrip_timestep": _readonly(returned), "frequencies": _readonly(frequencies),
        "sinusoidal_fp32": _readonly(embedded), "sinusoidal_bf16": _readonly(bf16(embedded))}


def predict_rope(img_ids, txt_ids, *, axes_dims_rope, rope_theta=2000):
    """Actual Flux2 IDs.float(), F64 frequency/angle/cos/sin, return F32."""
    _require(isinstance(axes_dims_rope, (tuple, list)) and axes_dims_rope
        and all(type(v) is int and v > 0 and v % 2 == 0 for v in axes_dims_rope)
        and isinstance(rope_theta, (int, float)) and np.isfinite(rope_theta) and rope_theta > 0,
        "invalid RoPE axes/theta")
    outputs = []
    for ids in (txt_ids, img_ids):
        ids = np.asarray(ids)
        if ids.ndim == 3:
            _require(ids.shape[0] == 1, "RoPE requires physical batch one")
            ids = ids[0]
        _require(ids.ndim == 2 and ids.shape[0] and ids.shape[1] == len(axes_dims_rope)
            and ids.dtype in (np.dtype("int64"), np.dtype("float32")) and np.isfinite(ids).all(), "invalid native token IDs")
        positions = ids.astype(np.float32).astype(np.float64)
        cos, sin = [], []
        for axis, dim in enumerate(axes_dims_rope):
            frequency = 1 / np.power(float(rope_theta), np.arange(0, dim, 2, dtype=np.float64) / dim)
            angle = positions[:, axis, None] * frequency[None, :]
            cos.append(np.repeat(np.cos(angle), 2, axis=1).astype(np.float32))
            sin.append(np.repeat(np.sin(angle), 2, axis=1).astype(np.float32))
        outputs.append((np.concatenate(cos, axis=-1), np.concatenate(sin, axis=-1)))
    return tuple(_readonly(np.concatenate((outputs[0][i], outputs[1][i]), axis=0)) for i in range(2))


def predict_project(latent, conditioning, img_ids, txt_ids, timestep, weights, *, axes_dims_rope,
                    rope_theta=2000, timestep_channels=256, projection_chunk=256):
    """Predict complete project I/T/context from own committed Z and carried C."""
    _chunks(projection_chunk)
    _weights(weights, PROJECT_LINEAR_NAMES)
    z, c = _native(latent), _native(conditioning)
    _require(z.ndim == c.ndim == 3 and z.shape[0] == c.shape[0] == 1, "project requires batch-one packed latent and conditioning")
    time = predict_timestep(timestep, timestep_channels)
    temb = _linear(_silu(_linear(time["sinusoidal_bf16"], weights,
        "time_guidance_embed.timestep_embedder.linear_1", projection_chunk)), weights,
        "time_guidance_embed.timestep_embedder.linear_2", projection_chunk)
    activated = _silu(temb)
    context = {"temb": _readonly(temb),
        "img_mod": _readonly(_linear(activated, weights, "double_stream_modulation_img.linear", projection_chunk)),
        "txt_mod": _readonly(_linear(activated, weights, "double_stream_modulation_txt.linear", projection_chunk)),
        "single_mod": _readonly(_linear(activated, weights, "single_stream_modulation.linear", projection_chunk)),
        "rope": predict_rope(img_ids, txt_ids, axes_dims_rope=axes_dims_rope, rope_theta=rope_theta)}
    image = _readonly(_linear(z, weights, "x_embedder", projection_chunk))
    text = _readonly(_linear(c, weights, "context_embedder", projection_chunk))
    width = image.shape[-1]
    _require(text.shape[-1] == temb.shape[-1] == width
        and np.asarray(img_ids).shape[-2] == image.shape[1]
        and np.asarray(txt_ids).shape[-2] == text.shape[1]
        and context["img_mod"].shape == context["txt_mod"].shape == (1, 6 * width)
        and context["single_mod"].shape == (1, 3 * width)
        and context["rope"][0].shape[0] == image.shape[1] + text.shape[1], "project complete-state geometry differs")
    return {"image": image, "text": text, "context": context, "timestep": time,
        "image_descriptor": descriptor(image), "text_descriptor": descriptor(text),
        "input_latent_descriptor": descriptor(z), "conditioning_descriptor": descriptor(c),
        "source_contract": dict(SOURCE_CONTRACT),
        "arithmetic": "B=FP64/FP32 to F32 to RNE BF16. Adapter t=B(schedule); B(t/1000), B(*1000). Timesteps frequency/angle operations F32, transcendentals FP64 rounded F32, cos then sin, cast B. Time linear1->B, SiLU->B, linear2->B. Modulation SiLU->B then linear->B. Image/text projection linear->B. RoPE IDs->F32, frequency/angle/trig F64->F32, T before I.",
        "scope": "Complete new project from predicted incoming latent and carried native conditioning/IDs/schedule/parameters. Guidance=None, no native project or future contexts supplied. Transcendental/native-kernel rounding differences remain observations."}


def predict_joint(image, text, context, weights, *, heads, layernorm_eps=1e-6, qk_eps=1e-6,
                  query_chunk=64, projection_chunk=256, retain_fields=False):
    """Complete sequential two-stream block: attention write then receiving MLP."""
    _chunks(query_chunk, projection_chunk)
    _weights(weights, JOINT_LINEAR_NAMES, JOINT_NORM_NAMES)
    i, t = _native(image), _native(text)
    _require(i.ndim == t.ndim == 3 and i.shape[0] == t.shape[0] == 1 and i.shape[-1] == t.shape[-1]
        and type(heads) is int and heads > 0 and i.shape[-1] % heads == 0 and (i.shape[-1] // heads) % 2 == 0
        and all(type(e) in {int, float} and np.isfinite(e) and e > 0 for e in (layernorm_eps, qk_eps))
        and type(retain_fields) is bool, "joint input geometry/normalization differs")
    width, text_rows = i.shape[-1], t.shape[1]
    mods = []
    for key in ("img_mod", "txt_mod"):
        mod = _native(context[key])
        _require(mod.shape in {(1, 6 * width), (1, 1, 6 * width)}, "joint modulation shape differs")
        mods.append(np.split(mod.reshape(1, 1, 6 * width), 6, axis=-1))
    cos, sin = map(np.asarray, context["rope"])
    dim = width // heads
    _require(all(x.dtype == np.float32 and x.shape == (i.shape[1] + t.shape[1], dim)
        and np.isfinite(x).all() for x in (cos, sin)), "joint RoPE geometry differs")
    normalized_i = _modulate(i, mods[0][0], mods[0][1], layernorm_eps)
    normalized_t = _modulate(t, mods[1][0], mods[1][1], layernorm_eps)

    def rms(x, name):
        gamma = weights[name]
        _require(gamma.shape == (dim,), "joint QK norm width differs")
        wide = x.astype(np.float64)
        return bf16(wide / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + qk_eps) * gamma.astype(np.float64))

    streams = []
    for value, names, norms in ((normalized_t, ("add_q_proj", "add_k_proj", "add_v_proj"), ("norm_added_q", "norm_added_k")),
                               (normalized_i, ("to_q", "to_k", "to_v"), ("norm_q", "norm_k"))):
        q, k, v = [_linear(value, weights, "attn." + name, projection_chunk).reshape(1, value.shape[1], heads, dim) for name in names]
        streams.append((rms(q, "attn." + norms[0] + ".weight"), rms(k, "attn." + norms[1] + ".weight"), v))
    q, k, v = [np.concatenate((streams[0][j], streams[1][j]), axis=1) for j in range(3)]

    def rotate(x):
        rotated = np.empty_like(x)
        rotated[..., 0::2], rotated[..., 1::2] = -x[..., 1::2], x[..., 0::2]
        return bf16(np.add(np.multiply(x, cos[None, :, None, :], dtype=np.float32),
            np.multiply(rotated, sin[None, :, None, :], dtype=np.float32), dtype=np.float32))

    q, k = rotate(q), rotate(k)
    keys, values = k.astype(np.float64).transpose(0, 2, 3, 1), v.astype(np.float64).transpose(0, 2, 1, 3)
    attention = np.empty((1, q.shape[1], width), np.float32)
    for first in range(0, q.shape[1], query_chunk):
        last = min(first + query_chunk, q.shape[1])
        score = q[:, first:last].astype(np.float64).transpose(0, 2, 1, 3) @ keys / np.sqrt(float(dim))
        score -= score.max(axis=-1, keepdims=True)
        p = np.exp(score)
        p /= p.sum(axis=-1, keepdims=True)
        attention[:, first:last] = bf16((p @ values).transpose(0, 2, 1, 3).reshape(1, last - first, width))
    attention_t = _linear(attention[:, :text_rows], weights, "attn.to_add_out", projection_chunk)
    attention_i = _linear(attention[:, text_rows:], weights, "attn.to_out.0", projection_chunk)
    outputs, stages = [], {}
    for role, residual, attended, mod, ff in (("image", i, attention_i, mods[0], "ff"),
                                            ("text", t, attention_t, mods[1], "ff_context")):
        after_attention = bf16(residual + bf16(mod[2] * attended))
        ff_input = _modulate(after_attention, mod[3], mod[4], layernorm_eps)
        paired = _linear(ff_input, weights, ff + ".linear_in", projection_chunk)
        _require(paired.shape[-1] % 2 == 0, "joint SwiGLU paired width differs")
        u, v_mlp = np.split(paired, 2, axis=-1)
        activation = bf16(_silu(u) * v_mlp)
        mlp = _linear(activation, weights, ff + ".linear_out", projection_chunk)
        output = _readonly(bf16(after_attention + bf16(mod[5] * mlp)))
        outputs.append(output)
        indices = np.unique(np.linspace(0, output.shape[1] - 1, min(8, output.shape[1])).astype(np.int64)).tolist()
        stages[role] = {"indices": indices, **{key: _readonly(value[:, indices]) for key, value in
            (("after_attention", after_attention), ("mlp_input", ff_input), ("mlp_output", mlp), ("output", output))}}
        if retain_fields:
            stages[role].update(full_after_attention=_readonly(after_attention), full_mlp_output=_readonly(mlp))
    return {"image": outputs[0], "text": outputs[1], "image_descriptor": descriptor(outputs[0]),
        "text_descriptor": descriptor(outputs[1]), "input_image_descriptor": descriptor(i),
        "input_text_descriptor": descriptor(t), "sampled": stages, "heads": heads, "head_dim": dim,
        "source_contract": dict(SOURCE_CONTRACT),
        "arithmetic": "Separate I/T population LN F64->B; B(1+scale), B(product), B(+shift). Separate Q/K/V linears F64->B; four learned RMS norms F64->B; concatenate T/I; separate F32 RoPE multiplies/add->B. All-query noncausal unmasked F64 softmax/AV->B; separate output linears->B. B(R+B(gate*A)) then recomputed LN/modulation, paired linear->B, B(SiLU(u))*v->B, output linear->B, B(U+B(gate*M)).",
        "scope": "Complete first joint block conditional on own predicted project I/T/context and exact native parameters. No dropout, KV cache, guidance or FP16 clipping. Reference reductions do not emulate CUDA kernels; all discrepancies are observations."}

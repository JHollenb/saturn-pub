"""Conditional positioned-QKV sign transformation and original attention tail.

No input normalization, QKV formation, rotary, positive R/G/U reuse or native
module entry is performed here. The component panel supplies custody and guards.
"""
from __future__ import annotations

SCHEMA = "mdb-qwen-cached-qkv-v1"


def signed_dense(value, row_signs):
    import torch

    if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != row_signs.numel() or value.dtype != torch.bfloat16:
        raise ValueError("positioned QKV must have BF16 B,H,T,D geometry")
    if row_signs.dtype != value.dtype or row_signs.device != value.device or not bool(((row_signs == 1) | (row_signs == -1)).all()):
        raise ValueError("exact BF16 same-device row signs required")
    output = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype, device=value.device)
    torch.mul(value, row_signs.view(1, 1, -1, 1), out=output)
    if output.stride() != value.stride():
        raise ValueError("signed positioned QKV layout changed")
    return output


def predict(graph, hidden, sources, parameters, profile, index, context, dispatch, row_signs, observe):
    import torch
    from torch.nn import functional as F

    batch, length, width = hidden.shape
    heads, kv, dim, inter = (profile[key] for key in ("num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size"))
    if batch != 1 or row_signs.shape != (length,) or hidden.dtype != torch.bfloat16 or context[0] is not None or context[1] is not True:
        raise ValueError("cached predictor requires declared all-attended BF16 context")
    def emit(name, value):
        observe(name, value)
        return value
    def parameter(name, shape):
        return graph["_check_parameter"](parameters, f"model.layers.{index}." + name + ".weight", shape, hidden.dtype)
    emit("h", hidden)
    q, k, v = (emit(name, signed_dense(sources[name], row_signs)) for name in ("q", "k", "v"))
    if q.shape != (batch, heads, length, dim) or k.shape != (batch, kv, length, dim) or v.shape != k.shape:
        raise ValueError("bound positioned QKV profile geometry differs")
    head = emit("head", graph["_attention"](q, k, v, context[0], is_causal=context[1], scale=dim**-0.5,
        groups=profile["num_key_value_groups"], dispatch=dispatch))
    projection = emit("projection", graph["_linear"](head.reshape(batch, length, -1), parameter("self_attn.o_proj", (width, heads * dim))))
    residual = emit("R", hidden + projection)
    post = emit("postnorm", graph["_rms_norm"](residual, parameter("post_attention_layernorm", (width,)), profile["rms_norm_eps"]))
    gate = emit("gate", graph["_linear"](post, parameter("mlp.gate_proj", (inter, width))))
    up = emit("up", graph["_linear"](post, parameter("mlp.up_proj", (inter, width))))
    activation = emit("activation", F.silu(gate))
    product = emit("product", activation * up)
    mlp = emit("M", graph["_linear"](product, parameter("mlp.down_proj", (width, inter))))
    emit("Hnext", residual + mlp)
    return dict(schema=SCHEMA, formula="positioned Q'=sQ,K'=sK,V'=sV; original SDPA/O/residual/postnorm/SwiGLU/down/Hnext",
        counts=dict(SDPA=1, linear=4, RMSNorm=1, native_module_forwards=0),
        omitted=dict(input_RMSNorm=1, QKV_linear=3, QK_RMSNorm=2, rotary_applications=2),
        scope="conditional inherited positioned-QKV frontier; finite relations are observations")

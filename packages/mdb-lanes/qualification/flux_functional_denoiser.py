"""Explicit eval FLUX.2 Klein graph, sharing only Torch primitive kernels.

Reads supplied native parameter storage without copying or calling any module.
The arithmetic profile matches Diffusers 0.39's unsharded native attention path:
F.linear, F.layer_norm, F.rms_norm, F.silu and F.scaled_dot_product_attention
are shared kernels. This is an independently orchestrated graph, not an
independent numerical kernel implementation or a compressed model. No reference
KV, LoRA, dropout, tensor parallelism, stochastic scheduler or training support.
All inputs/contexts must belong to the same pinned model and execution profile.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

SHARED_KERNELS = ("linear", "layer_norm", "rms_norm", "silu", "scaled_dot_product_attention")
PROFILE = "flux2-klein-functional-diffusers-0.39-native-attention-v1"


def validate_model(model):
    """Check the bounded module layout without invoking a model or copying weights."""
    if model.training:
        raise ValueError("functional prediction requires eval mode")
    if model.config.guidance_embeds:
        raise ValueError("only guidance-free distilled Klein is implemented")
    if next(model.parameters()).dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("only FP32/BF16 arithmetic profiles are implemented")
    if getattr(model, "peft_config", None):
        raise ValueError("LoRA/adapters are outside this functional profile")
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and type(module) is not torch.nn.Linear:
            raise ValueError("wrapped/sharded linear outside functional profile")
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise ValueError("nonzero dropout outside functional profile")
    for block in (*model.transformer_blocks, *model.single_transformer_blocks):
        processor = block.attn.processor
        if type(processor).__name__ not in {"Flux2AttnProcessor", "Flux2ParallelSelfAttnProcessor"}:
            raise ValueError("custom/KV attention processor outside functional profile")
        if getattr(processor, "_attention_backend", None) not in (None, "native"):
            raise ValueError("functional graph requires native SDPA attention")
        if getattr(processor, "_parallel_config", None) is not None:
            raise ValueError("parallel attention outside functional profile")
    if next(model.parameters()).device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU/CUDA arithmetic profiles are implemented")


def _linear(module, x):
    return F.linear(x, module.weight, module.bias)


def _norm(module, x):
    return F.layer_norm(x, module.normalized_shape, module.weight, module.bias, module.eps)


def _rms(module, x):
    return F.rms_norm(x, module.normalized_shape, module.weight, module.eps)


def _swiglu(x):
    a, b = x.chunk(2, dim=-1)
    return F.silu(a) * b


def _ff(module, x):
    return _linear(module.linear_out, _swiglu(_linear(module.linear_in, x)))


def _mods(value, count):
    parts = (value.unsqueeze(1) if value.ndim == 2 else value).chunk(3 * count, dim=-1)
    return tuple(parts[3 * index:3 * index + 3] for index in range(count))


def _time_projection(timesteps, channels):
    half = channels // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device)
    emb = timesteps[:, None].float() * torch.exp(exponent / half)[None, :]
    emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
    return F.pad(emb, (0, 1)) if channels % 2 else emb


def _time_embedding(module, timestep):
    sample = _time_projection(timestep, module.time_proj.num_channels).to(timestep.dtype)
    embedder = module.timestep_embedder
    return _linear(embedder.linear_2, F.silu(_linear(embedder.linear_1, sample)))


def _rope(module, ids):
    if ids.ndim == 3:
        ids = ids[0]
    cos, sin = [], []
    for axis, width in enumerate(module.axes_dim):
        if width % 2:
            raise ValueError("RoPE axes must be even")
        frequencies = 1.0 / (module.theta ** (torch.arange(0, width, 2, dtype=torch.float64, device=ids.device) / width))
        angles = torch.outer(ids.float()[:, axis], frequencies)
        cos.append(angles.cos().repeat_interleave(2, dim=1).float())
        sin.append(angles.sin().repeat_interleave(2, dim=1).float())
    return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def _rotate(x, rope):
    cos, sin = (value[None, :, None, :] for value in rope)
    real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack([-imag, real], dim=-1).flatten(3)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def _attention(q, k, v, rope):
    q, k = _rotate(q, rope), _rotate(k, rope)
    out = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                                        v.permute(0, 2, 1, 3), dropout_p=0., is_causal=False)
    return out.permute(0, 2, 1, 3).flatten(2, 3).to(q.dtype)


@torch.inference_mode()
def functional_project(model, latent, conditioning, img_ids, txt_ids, timestep):
    """Project a raw packed latent. timestep uses adapter scheduler units (0..1000).

    Returns (image, text, context). Every context after the initial observed cut
    is computed here from its own timestep; no later native activations are read.
    Klein's distilled guidance-free time embedder is the declared profile.
    """
    timestep = torch.as_tensor(timestep, device=latent.device).reshape(-1).to(latent.dtype)
    temb = _time_embedding(model.time_guidance_embed, (timestep / 1000).to(latent.dtype) * 1000)
    img_rope, txt_rope = _rope(model.pos_embed, img_ids), _rope(model.pos_embed, txt_ids)
    context = {"temb": temb,
               "img_mod": _linear(model.double_stream_modulation_img.linear, F.silu(temb)),
               "txt_mod": _linear(model.double_stream_modulation_txt.linear, F.silu(temb)),
               "single_mod": _linear(model.single_stream_modulation.linear, F.silu(temb)),
               "rope": tuple(torch.cat([text, image], dim=0) for text, image in zip(txt_rope, img_rope))}
    return _linear(model.x_embedder, latent), _linear(model.context_embedder, conditioning), context


@torch.inference_mode()
def functional_joint(model, index, image, text, context):
    """One complete joint block; returns (image, text)."""
    block = model.transformer_blocks[index]
    (shift, scale, gate), (shift_ff, scale_ff, gate_ff) = _mods(context["img_mod"], 2)
    (cshift, cscale, cgate), (cshift_ff, cscale_ff, cgate_ff) = _mods(context["txt_mod"], 2)
    x = (1 + scale) * _norm(block.norm1, image) + shift
    c = (1 + cscale) * _norm(block.norm1_context, text) + cshift
    attn = block.attn
    q, k, v = (_linear(getattr(attn, name), x).unflatten(-1, (-1, attn.head_dim)) for name in ("to_q", "to_k", "to_v"))
    cq, ck, cv = (_linear(getattr(attn, name), c).unflatten(-1, (-1, attn.head_dim)) for name in ("add_q_proj", "add_k_proj", "add_v_proj"))
    q = torch.cat([_rms(attn.norm_added_q, cq), _rms(attn.norm_q, q)], dim=1)
    k = torch.cat([_rms(attn.norm_added_k, ck), _rms(attn.norm_k, k)], dim=1)
    out = _attention(q, k, torch.cat([cv, v], dim=1), context["rope"])
    cout, out = out[:, :text.shape[1]], out[:, text.shape[1]:]
    image = image + gate * _linear(attn.to_out[0], out)
    image = image + gate_ff * _ff(block.ff, _norm(block.norm2, image) * (1 + scale_ff) + shift_ff)
    text = text + cgate * _linear(attn.to_add_out, cout)
    text = text + cgate_ff * _ff(block.ff_context, _norm(block.norm2_context, text) * (1 + cscale_ff) + cshift_ff)
    if text.dtype == torch.float16:
        text = text.clip(-65504, 65504)
    return image, text


@torch.inference_mode()
def functional_single(model, index, image, text, context):
    """One fused attention/MLP single block; returns split (image, text)."""
    block = model.single_transformer_blocks[index]
    shift, scale, gate = _mods(context["single_mod"], 1)[0]
    combined = torch.cat([text, image], dim=1)
    x = (1 + scale) * _norm(block.norm, combined) + shift
    attn = block.attn
    qkv, mlp = _linear(attn.to_qkv_mlp_proj, x).split([3 * attn.inner_dim, attn.mlp_hidden_dim * 2], dim=-1)
    q, k, v = (value.unflatten(-1, (-1, attn.head_dim)) for value in qkv.chunk(3, dim=-1))
    attention = _attention(_rms(attn.norm_q, q), _rms(attn.norm_k, k), v, context["rope"])
    combined = combined + gate * _linear(attn.to_out, torch.cat([attention, _swiglu(mlp)], dim=-1))
    if combined.dtype == torch.float16:
        combined = combined.clip(-65504, 65504)
    return combined[:, text.shape[1]:], combined[:, :text.shape[1]]


@torch.inference_mode()
def functional_readout(model, image, context):
    scale, shift = _linear(model.norm_out.linear, F.silu(context["temb"]).to(image.dtype)).chunk(2, dim=1)
    return _linear(model.proj_out, _norm(model.norm_out.norm, image) * (1 + scale)[:, None, :] + shift[:, None, :])


@torch.inference_mode()
def functional_scheduler(latent, noise, sigmas, step):
    """Deterministic FlowMatch Euler, preserving native BF16 scalar promotion."""
    return (latent.to(torch.float32) + (sigmas[step + 1] - sigmas[step]) * noise).to(noise.dtype)


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


@torch.inference_mode()
def functional_trajectory(model, state, on_boundary=None, *, max_transitions=2048):
    """Consume a captured operation-granularity state through final latent.

    Calls on_boundary(event, state) synchronously after every transition. Events
    name before step/phase/index and next_step/next_phase/next_index. State tensors
    are borrowed snapshots: callbacks must copy retained evidence, and must not
    mutate them. The caller's complete initial tensor tree is cloned once.
    Only fixed conditioning/IDs/sigmas and the initial live context are observed;
    all later states and contexts consume this graph's own predictions.
    """
    validate_model(model)
    if type(max_transitions) is not int or max_transitions < 1:
        raise ValueError("finite positive transition budget required")
    out = _clone_tree(state)
    transitions = 0
    while out["step"] < len(out["timesteps"]):
        if transitions >= max_transitions:
            raise ValueError("functional transition budget exhausted")
        step, phase, index = out["step"], out["phase"], out["cursor"]
        after = dict(out)
        if phase == "project":
            after["image"], after["text"], after["context"] = functional_project(
                model, out["latent"], out["conditioning"], out["img_ids"], out["txt_ids"], out["timesteps"][step])
            after["phase"], after["cursor"] = "joint", 0
        elif phase == "joint" and 0 <= index < len(model.transformer_blocks):
            after["image"], after["text"] = functional_joint(model, index, out["image"], out["text"], out["context"])
            after["cursor"] = index + 1
            if after["cursor"] == len(model.transformer_blocks):
                after["phase"], after["cursor"] = "single", 0
        elif phase == "single" and 0 <= index < len(model.single_transformer_blocks):
            after["image"], after["text"] = functional_single(model, index, out["image"], out["text"], out["context"])
            after["cursor"] = index + 1
            if after["cursor"] == len(model.single_transformer_blocks):
                after["phase"] = "readout"
        elif phase == "readout":
            after["noise_prediction"] = functional_readout(model, out["image"], out["context"])
            after["phase"] = "scheduler"
        elif phase == "scheduler":
            after["latent"] = functional_scheduler(out["latent"], out["noise_prediction"], out["sigmas"], step)
            after.update(step=step + 1, phase="project", cursor=0, image=None, text=None, context=None, noise_prediction=None)
        else:
            raise ValueError("unsupported or malformed operation frontier")
        transitions += 1
        if on_boundary is not None:
            on_boundary({"step": step, "phase": phase, "index": index, "next_step": after["step"],
                         "next_phase": after["phase"], "next_index": after["cursor"]}, after)
        out = after
    return out

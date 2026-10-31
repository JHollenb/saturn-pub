"""Explicit functional decoder for the untiled Diffusers 0.39 Klein VAE.

The supplied pipeline provides weights and configuration only. This code never
calls a VAE/module forward, decode, pipeline latent helper or image processor.
It shares Torch's convolution, GroupNorm, SiLU, nearest interpolation, linear
and SDPA primitives with the native implementation; it is not independent of
those kernels or a cross-backend precision emulator. No temporal VAE is used.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _named(value, expected):
    _require(type(value).__name__ == expected, "unsupported decoder module: " + type(value).__name__)


def _conv(module, value):
    _require(type(module) is torch.nn.Conv2d and module.padding_mode == "zeros", "requires plain zero-padded Conv2d")
    return F.conv2d(value, module.weight, module.bias, module.stride, module.padding, module.dilation, module.groups)


def _norm(module, value):
    _require(type(module) is torch.nn.GroupNorm and math.isfinite(module.eps) and module.eps > 0,
             "requires finite plain GroupNorm")
    return F.group_norm(value, module.num_groups, module.weight, module.bias, module.eps)


def _silu(module, value):
    _require(type(module) is torch.nn.SiLU and not module.inplace, "requires out-of-place SiLU")
    return F.silu(value)


def _linear(module, value):
    _require(type(module) is torch.nn.Linear, "requires plain linear projection")
    return F.linear(value, module.weight, module.bias)


def _dropout_identity(module):
    _require(type(module) is torch.nn.Dropout and not module.training, "dropout must be in eval mode")


def _resnet(module, value):
    _named(module, "ResnetBlock2D")
    _require(module.time_emb_proj is None and module.upsample is None and module.downsample is None
             and module.pre_norm and module.time_embedding_norm in {"default", "group"},
             "conditioned/resampling Resnet is outside the Klein decoder profile")
    _require(math.isfinite(module.output_scale_factor) and module.output_scale_factor > 0, "invalid residual scale")
    hidden = _conv(module.conv1, _silu(module.nonlinearity, _norm(module.norm1, value)))
    hidden = _silu(module.nonlinearity, _norm(module.norm2, hidden))
    _dropout_identity(module.dropout)
    hidden = _conv(module.conv2, hidden)
    skip = value if module.conv_shortcut is None else _conv(module.conv_shortcut, value)
    return (skip + hidden) / module.output_scale_factor


def _attention(module, value):
    """Plain AttnProcessor2_0 self-attention, explicitly reconstructed."""
    _named(module, "Attention")
    _named(module.processor, "AttnProcessor2_0")
    _require(module.spatial_norm is None and module.norm_q is None and module.norm_k is None
             and module.added_kv_proj_dim is None and not module.norm_cross and module.scale_qk
             and not module.fused_projections and len(module.to_out) == 2,
             "custom/fused/conditioned attention is outside the Klein VAE profile")
    _require(type(module.heads) is int and module.heads > 0
             and math.isfinite(module.rescale_output_factor) and module.rescale_output_factor > 0,
             "invalid attention geometry/scale")
    residual = value
    batch, channels, height, width = value.shape
    hidden = value.view(batch, channels, height * width).transpose(1, 2)
    if module.group_norm is not None:
        hidden = _norm(module.group_norm, hidden.transpose(1, 2)).transpose(1, 2)
    query, key, projected = (_linear(layer, hidden) for layer in (module.to_q, module.to_k, module.to_v))
    _require(key.shape[-1] % module.heads == 0 and query.shape[-1] == projected.shape[-1] == key.shape[-1],
             "unsupported attention projection dimensions")
    head_dim = key.shape[-1] // module.heads
    query, key, projected = (item.view(batch, -1, module.heads, head_dim).transpose(1, 2)
                             for item in (query, key, projected))
    hidden = F.scaled_dot_product_attention(query, key, projected, attn_mask=None, dropout_p=0., is_causal=False)
    hidden = hidden.transpose(1, 2).reshape(batch, -1, module.heads * head_dim).to(query.dtype)
    hidden = _linear(module.to_out[0], hidden)
    _dropout_identity(module.to_out[1])
    hidden = hidden.transpose(-1, -2).reshape(batch, channels, height, width)
    if module.residual_connection:
        hidden = hidden + residual
    return hidden / module.rescale_output_factor


def _upsample(module, value):
    _named(module, "Upsample2D")
    _require(module.norm is None and not module.use_conv_transpose and module.interpolate and module.name == "conv"
             and value.shape[1] == module.channels, "unsupported decoder upsampling profile")
    # The pinned Torch >=2.1 supports native BF16 nearest interpolation.
    value = F.interpolate(value, scale_factor=2., mode="nearest")
    return _conv(module.conv, value) if module.use_conv else value


def unpack_latents(packed, ids, height, width):
    """Scatter explicit (T,H,W,L) IDs; require one complete unique image grid."""
    _require(isinstance(packed, torch.Tensor) and isinstance(ids, torch.Tensor)
             and packed.ndim == ids.ndim == 3 and packed.shape[0] == ids.shape[0] == 1
             and ids.shape[1] == packed.shape[1] and ids.shape[2] == 4
             and type(height) is int and type(width) is int and height > 0 and width > 0
             and packed.shape[1] == height * width, "requires batch-one packed complete image grid")
    _require(packed.dtype in {torch.float32, torch.bfloat16} and bool(torch.isfinite(packed).all())
             and bool(torch.isfinite(ids).all()), "nonfinite or unsupported packed latent")
    _require(bool((ids == ids.to(torch.int64)).all()), "image IDs must be integral")
    positions = ids.to(device=packed.device, dtype=torch.int64)
    _require(bool((positions[..., 0] == 0).all()) and bool((positions[..., 3] == 0).all())
             and bool(((positions[..., 1] >= 0) & (positions[..., 1] < height)).all())
             and bool(((positions[..., 2] >= 0) & (positions[..., 2] < width)).all()), "image IDs escape the declared image plane")
    flat = positions[0, :, 1] * width + positions[0, :, 2]
    _require(torch.unique(flat).numel() == height * width, "duplicate/missing image positions")
    output = torch.zeros((height * width, packed.shape[2]), device=packed.device, dtype=packed.dtype)
    output.scatter_(0, flat.unsqueeze(1).expand(-1, packed.shape[2]), packed[0])
    return output.view(height, width, packed.shape[2]).permute(2, 0, 1).unsqueeze(0)


def unpatchify_latents(value):
    _require(value.ndim == 4 and value.shape[1] % 4 == 0, "requires 2x2 patched latent channels")
    batch, channels, height, width = value.shape
    return value.reshape(batch, channels // 4, 2, 2, height, width).permute(0, 1, 4, 2, 5, 3).reshape(
        batch, channels // 4, height * 2, width * 2)


def postprocess_pixels(decoded, *, do_normalize=True):
    """Native BF16/FP32 denormalize, FP32 NumPy conversion and ties-even uint8."""
    _require(decoded.ndim == 4 and decoded.shape[0] == 1 and decoded.shape[1] == 3
             and bool(torch.isfinite(decoded).all()) and type(do_normalize) is bool,
             "requires one finite RGB decoder output")
    image = (decoded * .5 + .5).clamp(0, 1) if do_normalize else decoded
    values = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    return np.ascontiguousarray((values * 255).round().astype(np.uint8))


@torch.inference_mode()
def decode_functional(pipe, packed_latent, img_ids, height, width, callback=None):
    """Decode supplied packed latent without calling any native module method.

    Returns {pixels: NumPy uint8 [1,H,W,3], decoded: raw Tensor [1,3,H,W],
    latents: unpatched VAE input, decoder_input: post-quant decoder argument}.
    Precision/device follow actual supplied VAE
    weights. BN statistics are square-rooted on CPU before casting, matching
    the public phase-offloaded adapter. callback(name, owned_tensor_clone) is
    optional and observes latent stages, mid/up blocks and raw decoded pixels.
    Untiled eval AutoencoderKLFlux2, 2x2 patches, GroupNorm/SiLU decoder and
    AttnProcessor2_0 are supported. Unsupported profiles fail before release.
    """
    vae = pipe.vae
    _named(vae, "AutoencoderKLFlux2")
    _require(not vae.training and not vae.use_tiling and not vae.use_slicing
             and tuple(vae.config.patch_size) == (2, 2) and vae.config.act_fn == "silu"
             and set(vae.config.up_block_types) == {"UpDecoderBlock2D"}
             and vae.config.out_channels == 3, "requires untiled eval Klein 2D VAE profile")
    _require(not any(hasattr(module, "_hf_hook") for module in vae.modules()), "VAE offload hooks are outside functional decoding")
    scale = 2 ** (len(vae.config.block_out_channels) - 1)
    _require(type(height) is int and type(width) is int and height > 0 and width > 0
             and height % (scale * 2) == width % (scale * 2) == 0
             and pipe.vae_scale_factor == scale, "output geometry differs from VAE scale")
    decoder = vae.decoder
    _named(decoder, "Decoder")
    _require(not decoder.training and not decoder.gradient_checkpointing, "decoder must use plain eval execution")
    _require(len(decoder.up_blocks) == len(vae.config.up_block_types), "decoder up-block inventory differs")
    _require(math.isfinite(vae.config.batch_norm_eps) and vae.config.batch_norm_eps > 0
             and vae.bn.running_mean.numel() == vae.bn.running_var.numel() == 4 * vae.config.latent_channels,
             "latent batch-normalization profile differs")
    _require(bool(getattr(pipe.image_processor.config, "do_normalize", True)), "native RGB normalization must be enabled")

    def emit(name, value):
        if callback is not None:
            callback(name, value.detach().clone())

    latent = unpack_latents(packed_latent, img_ids, height // (scale * 2), width // (scale * 2))
    _require(latent.shape[1] == 4 * vae.config.latent_channels, "packed latent channel count differs")
    emit("latent_unpacked", latent)
    mean = vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(latent)
    std = (vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(latent)
    latent = latent * std + mean
    emit("latent_denormalized", latent)
    latent = unpatchify_latents(latent).to(device=decoder.conv_in.weight.device)
    emit("latent_unpatched", latent)
    _require(latent.dtype == decoder.conv_in.weight.dtype, "native packed/VAE dtype differs; declare an explicit cast upstream")
    hidden = latent
    if vae.post_quant_conv is not None:
        hidden = _conv(vae.post_quant_conv, hidden)
        emit("post_quant_conv", hidden)
    decoder_input = hidden
    emit("decoder_input", decoder_input)
    hidden = _conv(decoder.conv_in, hidden)
    mid = decoder.mid_block
    _named(mid, "UNetMidBlock2D")
    _require(len(mid.resnets) == len(mid.attentions) + 1, "mid-block inventory differs")
    hidden = _resnet(mid.resnets[0], hidden)
    for attention, resnet in zip(mid.attentions, mid.resnets[1:], strict=True):
        if attention is not None:
            hidden = _attention(attention, hidden)
        hidden = _resnet(resnet, hidden)
    emit("decoder_mid", hidden)
    for index, block in enumerate(decoder.up_blocks):
        _named(block, "UpDecoderBlock2D")
        for resnet in block.resnets:
            hidden = _resnet(resnet, hidden)
        if block.upsamplers is not None:
            for upsampler in block.upsamplers:
                hidden = _upsample(upsampler, hidden)
        emit("decoder_up_" + str(index), hidden)
    decoded = _conv(decoder.conv_out, _silu(decoder.conv_act, _norm(decoder.conv_norm_out, hidden)))
    _require(tuple(decoded.shape) == (1, 3, height, width), "decoded image geometry differs")
    emit("decoded", decoded)
    return {"pixels": postprocess_pixels(decoded), "decoded": decoded, "latents": latent, "decoder_input": decoder_input}

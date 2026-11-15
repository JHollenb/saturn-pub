"""Explicit original-policy FLUX2 VAE decoder graph using shared Torch kernels.

Starts at the actual VAE.decode argument: post_quant_conv, decoder conv/mid/up,
GroupNorm/SiLU/output. ID unpacking and BN inverse normalization are upstream.
Reads supplied original parameter storage; calls no module forward, pipeline,
native decoder, scheduler, or state refresh. This is not independent numerical
kernel arithmetic and is only preparation until qualified on real geometry.
"""
from __future__ import annotations

import hashlib
import inspect
import runpy
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PROFILE = "flux2-functional-decoder-diffusers0.39-shared-native-primitives-v1"
SHARED_KERNELS = ("conv2d", "group_norm", "silu", "interpolate-nearest", "linear", "scaled_dot_product_attention", "dropout-eval")
PRIOR_HELPER_SHA256 = "576b94f3cd987e8849f6d4772c90a77f14fdfc9210dea608a9a3d16cf0f99a61"
SOURCE_FILES = {
    "autoencoder_kl_flux2.py": "7d9a976c1e4f42615e8c422f1643d86b49c4339221bd04b67f518b718ebd6c2d",
    "vae.py": "8e6abad3bd7b7806dd9c6c451b2438641ef728d7884e6bfed37b867f719b98fc",
    "resnet.py": "329975c03ddf3baa5528ace0e2be62f4224746700be4fda57dcbabb209569e9c",
    "upsampling.py": "d6410c71fa01c5363c6130784a709eb13b5be7e6b401dba7fa930f59b5e5559d",
    "unet_2d_blocks.py": "545d158c9d98fb8971c501303117fd3e32d27ae28506b89005c65548404f7437",
    "attention_processor.py": "7158a23bff0ce5bb1c03d66e2441dfd8d169b3598b2d494d32c6f88a3301ea01",
}
SOURCE_METHODS = {
    "Decoder": "be96bf1aec1d895a8af9bc0167c2d528aa57c382f558964cc4405a1e4c1db14c",
    "ResnetBlock2D": "e3f38e1722955caf8219056735b2d67adfdc670553e701714bc47f4dac1a31e7",
    "Upsample2D": "3c0e13df5e4ac796aacf0bd6b58c9b56496acf59554d9057b94f0ee0fe00d1fb",
    "UNetMidBlock2D": "4e74348d59aa87c19b95fd45139daee398260f03ec9849323fa196c218f47d1f",
    "UpDecoderBlock2D": "a4a83c23b787086acdd6f1eef54f4e587e12f31f72c7f493ca44291cdc2ecef8",
    "Attention": "e6822ac3500626c7318b56f135752693a1222976154cbcc8c2e6b5f840d82b6b",
    "AttnProcessor2_0": "bf23ea5ae40404272695b752532af4cef782d7626168f17e9f0d6652a941dd67",
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def source_contract():
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from diffusers.models.autoencoders.vae import Decoder
    from diffusers.models.resnet import ResnetBlock2D, Upsample2D
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D

    kinds = (AutoencoderKLFlux2, Decoder, ResnetBlock2D, Upsample2D, UNetMidBlock2D, UpDecoderBlock2D, Attention, AttnProcessor2_0)
    files = {Path(inspect.getfile(kind)).name: hashlib.sha256(Path(inspect.getfile(kind)).read_bytes()).hexdigest() for kind in kinds}
    methods = {kind.__name__: hashlib.sha256(inspect.getsource(kind.__call__ if kind is AttnProcessor2_0 else kind.forward).encode()).hexdigest() for kind in kinds[1:]}
    _require(files == SOURCE_FILES and methods == SOURCE_METHODS, "qualified native decoder source differs")
    helper = Path(__file__).with_name("flux_functional_vae.py")
    _require(hashlib.sha256(helper.read_bytes()).hexdigest() == PRIOR_HELPER_SHA256, "frozen prior functional VAE differs")
    return dict(files=files, methods=methods, profile=PROFILE, shared_kernels=list(SHARED_KERNELS),
                prior_helper_sha256=PRIOR_HELPER_SHA256, prior_scope="Previously used shared-kernel functional VAE on seed23/red-cube blue-sphere; this wrapper adds decoder-entry applicability and full checkpoint observation, not first functional decoding.",
                independent_kernel_arithmetic=False, numerical_scope="Explicit orchestration of original Torch primitives; native kernel policies are inherited unchanged.")


def validate_model(vae, *, profile="original"):
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from diffusers.models.autoencoders.vae import Decoder
    from diffusers.models.resnet import ResnetBlock2D, Upsample2D
    from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D

    contract = source_contract()
    _require(type(vae) is AutoencoderKLFlux2 and type(vae.decoder) is Decoder and not vae.training,
             "original eval AutoencoderKLFlux2/Decoder required")
    _require(profile in {"original", "tiny"}, "explicit applicability profile required")
    _require(not vae.use_tiling and not vae.use_slicing and not vae.decoder.gradient_checkpointing,
             "untiled unsliced noncheckpointed decoder required")
    _require(not getattr(vae, "peft_config", None) and not getattr(vae, "hf_device_map", None), "wrapped/offloaded/adapter decoder outside profile")
    _require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled("cpu"), "autocast outside decoder profile")
    _require(vae.config.act_fn == "silu" and tuple(vae.config.patch_size) == (2, 2)
             and vae.post_quant_conv is not None, "original SiLU/post-quant/patch geometry required")
    if profile == "original":
        _require(tuple(vae.config.block_out_channels) == (128, 256, 512, 512) and vae.config.latent_channels == 32
                 and vae.config.layers_per_block == 2 and vae.config.norm_num_groups == 32
                 and vae.config.in_channels == vae.config.out_channels == 3 and vae.config.mid_block_add_attention,
                 "original FLUX2 VAE geometry required")
    else:
        _require(len(vae.decoder.up_blocks) <= 3 and max(vae.config.block_out_channels) <= 32,
                 "tiny qualification profile only")
    selected = {"post_quant_conv": vae.post_quant_conv, "decoder": vae.decoder}
    allowed = {torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.SiLU, torch.nn.Dropout, torch.nn.ModuleList,
               Decoder, ResnetBlock2D, Upsample2D, UNetMidBlock2D, UpDecoderBlock2D, Attention}
    for prefix, root in selected.items():
        for name, module in root.named_modules():
            _require(type(module) in allowed, "custom/wrapped decoder module outside profile: " + prefix + "." + name)
            _require(not module.training, "all decoder modules must be eval")
            if isinstance(module, torch.nn.Conv2d):
                _require(module.padding_mode == "zeros", "nonzero-padding-mode convolution unsupported")
            if isinstance(module, torch.nn.SiLU):
                _require(not module.inplace, "inplace activation outside immutable operand graph")
            if isinstance(module, torch.nn.Dropout):
                _require(module.p == 0, "nonzero dropout outside original profile")
            if isinstance(module, ResnetBlock2D):
                _require(module.upsample is None and module.downsample is None and module.time_emb_proj is None
                         and module.time_embedding_norm in {"default", "group"} and type(module.norm1) is type(module.norm2) is torch.nn.GroupNorm,
                         "conditional/resampling Resnet outside profile")
            if isinstance(module, Upsample2D):
                _require(not module.use_conv_transpose and module.norm is None and module.use_conv
                         and module.name == "conv" and module.interpolate, "only native nearest-plus-conv upsample supported")
            if isinstance(module, Attention):
                _require(type(module.processor) is AttnProcessor2_0 and module.spatial_norm is None
                         and module.group_norm is not None and module.norm_q is None and module.norm_k is None
                         and not module.norm_cross and not module.is_cross_attention
                         and not getattr(module, "fused_projections", False), "original unfused self-attention processor required")
        for parameter in root.parameters():
            _require(parameter.dtype == torch.bfloat16 and parameter.device.type in {"cpu", "cuda"}
                     and not parameter.requires_grad, "frozen BF16 CPU/CUDA decoder parameters required")
    parameter_names = [name for name, _ in vae.named_parameters() if name.startswith(("post_quant_conv.", "decoder."))]
    contract.update(applicability=profile, parameter_names=parameter_names, decoder_parameter_tensors=len(parameter_names),
                    upstream_bn_statistics_used=False, native_module_forwards=0, dtype="torch.bfloat16", untiled=True, unsliced=True,
                    reuse="Frozen prior arithmetic function code in an isolated instrumentation namespace; original helper globals and Torch functional namespace unchanged.",
                    checkpoint_limits="Eval zero-p dropout is checked/counted as identity, without a fabricated separate tensor checkpoint.",
                    force_upcast_scope="Direct supplied BF16 decode; pipeline force_upcast policy is not invoked or silently reproduced.")
    return contract


def _emit(observe, counts, name, kind, inputs, output):
    counts[kind] = counts.get(kind, 0) + 1
    if observe is not None:
        observe(name, kind, inputs, output)
    return output


def _instrumented(vae, observe, counts):
    """Use identical frozen function code in a new observation-only namespace.

    Neither prior functions' globals, their module, nor Torch's F namespace are
    mutated. The proxy delegates the same primitives with the same arguments.
    Zero-p eval dropout is checked/counts as identity, without inventing an
    intermediate tensor checkpoint not supplied to the old identity validator.
    """
    helper = Path(__file__).with_name("flux_functional_vae.py")
    _require(hashlib.sha256(helper.read_bytes()).hexdigest() == PRIOR_HELPER_SHA256, "prior helper changed")
    old = runpy.run_path(str(helper))
    base = old["_conv"].__globals__
    namespace = dict(base)
    for name, value in base.items():
        if isinstance(value, types.FunctionType) and value.__globals__ is base:
            namespace[name] = types.FunctionType(value.__code__, namespace, name, value.__defaults__, value.__closure__)
    unchanged = dict(namespace)
    names = {id(module): name for name, module in vae.named_modules()}
    active = []

    class Primitives:
        def __getattr__(self, name):
            return getattr(F, name)

        def scaled_dot_product_attention(self, *args, **kwargs):
            return _emit(observe, counts, active[-1] + ".sdpa", "sdpa", args,
                         F.scaled_dot_product_attention(*args, **kwargs))

        def interpolate(self, *args, **kwargs):
            return _emit(observe, counts, active[-1] + ".interpolate", "interpolate_nearest", args,
                         F.interpolate(*args, **kwargs))

    namespace["F"] = Primitives()
    for key, kind in (("_conv", "conv2d"), ("_norm", "group_norm"), ("_silu", "silu"), ("_linear", "linear"),
                      ("_resnet", "resnet"), ("_attention", "attention"), ("_upsample", "upsample")):
        def instrument(module, value, key=key, kind=kind):
            name = names[id(module)]
            active.append(name)
            try:
                output = unchanged[key](module, value)
            finally:
                active.pop()
            return _emit(observe, counts, name, kind, (value,), output)
        namespace[key] = instrument

    def dropout(module):
        unchanged["_dropout_identity"](module)
        counts["dropout_identity"] = counts.get("dropout_identity", 0) + 1

    namespace["_dropout_identity"] = dropout
    return namespace


@torch.inference_mode()
def functional_decode(vae, decoder_input, *, observe=None, counts=None, profile="original"):
    """Return complete decoded BF16 output from one supplied decode argument.

    ``observe(name, kind, inputs_tuple, output)`` sees named primitive/module
    checkpoints. It must not mutate operands. It receives no target output.
    ``counts`` optionally accumulates actual functional primitive invocations.
    No kernel/backend/TF32/reduction flags or model parameters are changed.
    """
    validate_model(vae, profile=profile)
    parameter = vae.post_quant_conv.weight
    _require(decoder_input.ndim == 4 and decoder_input.shape[0] == 1 and decoder_input.dtype == parameter.dtype
             and decoder_input.device == parameter.device and decoder_input.shape[1] == parameter.shape[1]
             and bool(torch.isfinite(decoder_input).all()), "B1 finite original decoder input required")
    _require(max(vae.config.block_out_channels) * decoder_input.shape[-2] * decoder_input.shape[-1]
             * 4 ** (len(vae.decoder.up_blocks) - 1) * 2 < 2 ** 31,
             "large-interpolation contiguous fallback is outside frozen helper applicability")
    counts = {} if counts is None else counts
    ops = _instrumented(vae, observe, counts)
    x = ops["_conv"](vae.post_quant_conv, decoder_input)
    decoder = vae.decoder
    decoder_entry = x
    x = ops["_conv"](decoder.conv_in, x)
    mid_entry = x
    x = ops["_resnet"](decoder.mid_block.resnets[0], x)
    for i, (attention, resnet) in enumerate(zip(decoder.mid_block.attentions, decoder.mid_block.resnets[1:], strict=True)):
        if attention is not None:
            x = ops["_attention"](attention, x)
        x = ops["_resnet"](resnet, x)
    x = _emit(observe, counts, "decoder.mid_block", "mid_block", (mid_entry,), x)
    for i, up in enumerate(decoder.up_blocks):
        entry = x
        for j, resnet in enumerate(up.resnets):
            x = ops["_resnet"](resnet, x)
        if up.upsamplers is not None:
            for j, module in enumerate(up.upsamplers):
                x = ops["_upsample"](module, x)
        x = _emit(observe, counts, f"decoder.up_blocks.{i}", "up_block", (entry,), x)
    x = ops["_norm"](decoder.conv_norm_out, x)
    x = ops["_silu"](decoder.conv_act, x)
    x = ops["_conv"](decoder.conv_out, x)
    return _emit(observe, counts, "decoder", "decoder", (decoder_entry,), x)


__all__ = ["PROFILE", "SHARED_KERNELS", "SOURCE_FILES", "SOURCE_METHODS", "functional_decode", "source_contract", "validate_model"]

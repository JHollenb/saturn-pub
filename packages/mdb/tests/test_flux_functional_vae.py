"""Tiny random VAE parity and exclusion tests; no checkpoint/model downloads."""
from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline  # noqa: E402
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor  # noqa: E402

FUNCTIONAL = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_functional_vae.py"))


def tiny_pipe(*, dtype=torch.float32, post_quant=True, attention=True):
    torch.manual_seed(19)
    vae = AutoencoderKLFlux2(block_out_channels=(8, 16), decoder_block_out_channels=(8, 16),
        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
        layers_per_block=1, latent_channels=2, norm_num_groups=4, sample_size=16,
        use_post_quant_conv=post_quant, mid_block_add_attention=attention).eval().requires_grad_(False).to(dtype=dtype)
    with torch.no_grad():
        vae.bn.running_mean.copy_(torch.linspace(-.3, .4, 8))
        vae.bn.running_var.copy_(torch.linspace(.4, 1.3, 8))
    pipe = SimpleNamespace(vae=vae, vae_scale_factor=2, image_processor=Flux2ImageProcessor(vae_scale_factor=4))
    ids = torch.tensor([[[0, y, x, 0] for y in range(4) for x in range(4)]], dtype=torch.int64)
    latent = torch.randn(1, 16, 8, dtype=dtype)
    order = torch.tensor([6, 3, 15, 0, 4, 9, 1, 14, 5, 11, 8, 7, 2, 12, 10, 13])
    return pipe, latent[:, order], ids[:, order]


@torch.inference_mode()
def native_decode(pipe, packed, ids):
    value = Flux2KleinPipeline._unpack_latents_with_ids(packed, ids, 4, 4)
    mean = pipe.vae.bn.running_mean.detach().cpu().view(1, -1, 1, 1).to(value)
    std = (pipe.vae.bn.running_var.detach().cpu().view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).sqrt().to(value)
    value = Flux2KleinPipeline._unpatchify_latents(value * std + mean)
    observed = {}

    def observe(_module, args):
        observed["input"] = args[0].clone()

    hook = pipe.vae.decoder.register_forward_pre_hook(observe)
    try:
        decoded = pipe.vae.decode(value, return_dict=False)[0]
    finally:
        hook.remove()
    pixels = np.stack([np.asarray(image) for image in pipe.image_processor.postprocess(decoded, output_type="pil")])
    return decoded, pixels, value, observed["input"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("post_quant,attention", [(True, True), (False, False)])
def test_complete_functional_decode_matches_native_without_any_module_entry(monkeypatch, dtype, post_quant, attention):
    pipe, packed, ids = tiny_pipe(dtype=dtype, post_quant=post_quant, attention=attention)
    expected, pixels, latent, decoder_input = native_decode(pipe, packed, ids)
    before = packed.clone()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native module/helper executed during forecast")

    for module in pipe.vae.modules():
        monkeypatch.setattr(module, "forward", forbidden)
    monkeypatch.setattr(pipe.vae, "decode", forbidden)
    monkeypatch.setattr(pipe.image_processor, "postprocess", forbidden)
    monkeypatch.setattr(Flux2KleinPipeline, "_unpack_latents_with_ids", forbidden)
    monkeypatch.setattr(Flux2KleinPipeline, "_unpatchify_latents", forbidden)
    hook = torch.nn.modules.module.register_module_forward_pre_hook(forbidden)
    seen = {}
    try:
        result = FUNCTIONAL["decode_functional"](pipe, packed, ids, 16, 16, lambda name, value: seen.setdefault(name, value))
    finally:
        hook.remove()
    torch.testing.assert_close(result["decoded"], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["latents"], latent, rtol=0, atol=0)
    torch.testing.assert_close(result["decoder_input"], decoder_input, rtol=0, atol=0)
    torch.testing.assert_close(packed, before, rtol=0, atol=0)
    np.testing.assert_array_equal(result["pixels"], pixels)
    assert {"latent_unpacked", "latent_denormalized", "latent_unpatched", "decoder_mid", "decoder_up_0", "decoder_up_1", "decoded"} <= set(seen)


def test_callback_receives_owned_witness_not_mutable_computation():
    pipe, packed, ids = tiny_pipe()
    expected = FUNCTIONAL["decode_functional"](pipe, packed, ids, 16, 16)

    def observe(_name, value):
        value.zero_()

    result = FUNCTIONAL["decode_functional"](pipe, packed, ids, 16, 16, observe)
    torch.testing.assert_close(result["decoded"], expected["decoded"], rtol=0, atol=0)
    np.testing.assert_array_equal(result["pixels"], expected["pixels"])


def test_ids_unscatter_uses_coordinates_not_token_order_and_patch_channel_order():
    packed = torch.tensor([[[20., 21., 22., 23.], [10., 11., 12., 13.]]])
    ids = torch.tensor([[[0, 0, 1, 0], [0, 0, 0, 0]]])
    result = FUNCTIONAL["unpatchify_latents"](FUNCTIONAL["unpack_latents"](packed, ids, 1, 2))
    torch.testing.assert_close(result, torch.tensor([[[[10., 11., 20., 21.], [12., 13., 22., 23.]]]]))


@pytest.mark.parametrize("kind", ["duplicate", "fractional", "offplane", "missing"])
def test_ambiguous_grid_is_rejected_before_decoder_execution(kind):
    pipe, packed, ids = tiny_pipe()
    if kind == "duplicate":
        ids[:, 0] = ids[:, 1]
    elif kind == "fractional":
        ids = ids.float()
        ids[0, 0, 1] += .5
    elif kind == "offplane":
        ids[0, 0, 0] = 1
    else:
        packed, ids = packed[:, :-1], ids[:, :-1]
    with pytest.raises(ValueError, match="grid|integral|plane|positions"):
        FUNCTIONAL["decode_functional"](pipe, packed, ids, 16, 16)


def test_pixel_rounding_uses_numpy_ties_even_and_bf16_denormalization_first():
    # Half integer pixel values land on the nearest even value after conversion.
    decoded = torch.tensor([[[[-1., 0., 1.]], [[-1., 0., 1.]], [[-1., 0., 1.]]]])
    pixels = FUNCTIONAL["postprocess_pixels"](decoded)
    np.testing.assert_array_equal(pixels[0, 0, :, 0], [0, 128, 255])
    bf16 = torch.tensor([[[[-.25, .25]], [[-.25, .25]], [[-.25, .25]]]], dtype=torch.bfloat16)
    pixels = FUNCTIONAL["postprocess_pixels"](bf16)
    np.testing.assert_array_equal(pixels[0, 0, :, 0], [96, 159])


@pytest.mark.parametrize("feature", ["tiling", "train", "processor", "patch", "normalization"])
def test_unsupported_native_profile_is_not_silently_approximated(feature):
    pipe, packed, ids = tiny_pipe()
    if feature == "tiling":
        pipe.vae.use_tiling = True
    elif feature == "train":
        pipe.vae.train()
    elif feature == "processor":
        pipe.vae.decoder.mid_block.attentions[0].processor = object()
    elif feature == "patch":
        pipe.vae.register_to_config(patch_size=(1, 1))
    else:
        pipe.image_processor.register_to_config(do_normalize=False)
    with pytest.raises(ValueError, match="profile|module|normalization"):
        FUNCTIONAL["decode_functional"](pipe, packed, ids, 16, 16)

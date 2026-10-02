import torch
import pytest
from diffusers import Flux2Transformer2DModel, UNet2DConditionModel
from mdb.bench_native_forward import NativeForwardMachine
from mdb.bench_pages import BenchPages


@pytest.mark.parametrize("family", ["flux", "sdxl"])
def test_native_forward_partial_cut_without_prefix_reexecution(family, tmp_path):
    torch.manual_seed(33)
    if family == "flux":
        model = Flux2Transformer2DModel(in_channels=8, out_channels=8, num_layers=2,
            num_single_layers=2, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=16, timestep_guidance_channels=16, axes_dims_rope=(2, 2, 2, 2)).eval()
        kwargs = dict(hidden_states=torch.randn(1, 4, 8), encoder_hidden_states=torch.randn(1, 3, 16),
            timestep=torch.tensor([0.5]), img_ids=torch.zeros(4, 4), txt_ids=torch.zeros(3, 4), return_dict=False)
    else:
        model = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(16, 32),
            layers_per_block=1, cross_attention_dim=16, attention_head_dim=4, norm_num_groups=4).eval()
        kwargs = dict(sample=torch.randn(1, 4, 8, 8), timestep=torch.tensor(500),
            encoder_hidden_states=torch.randn(1, 3, 16), return_dict=False)
    with torch.inference_mode():
        expected = model(**kwargs)[0]
        machine = NativeForwardMachine(model)
        frame = machine.start(kwargs); machine.advance(frame)
        assert frame["blocks"] == 1 and not frame["done"]
        pages = BenchPages(tmp_path, device="cpu", budget=16 << 20)
        cut = pages.encode(frame)
        fresh = NativeForwardMachine(model)
        restored = pages.decode(cut, private=True)
        # If restore reran the prefix, this executed first native block would fail.
        first = model.transformer_blocks[0] if family == "flux" else model.down_blocks[0]
        original = first.forward
        def forbidden(*args, **kwargs): raise AssertionError("completed prefix reexecuted")
        first.forward = forbidden
        try: fresh.advance(restored, finish=True)
        finally: first.forward = original
        assert restored["done"]
        assert torch.equal(restored["output"][0], expected)

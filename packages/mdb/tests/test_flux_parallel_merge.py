import runpy
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock


def helper():
    return runpy.run_path(str(Path(__file__).parent.parent / "qualification/flux_parallel_merge.py"))["merge_candidate"]


def test_real_native_parallel_block_merge_and_column_interventions():
    torch.manual_seed(19)
    block = Flux2SingleTransformerBlock(8, 2, 4, mlp_ratio=2).eval()
    residuals = [torch.randn(1, 5, 8) for _ in range(2)]
    modulation = torch.randn(1, 24)
    gate = modulation[:, None, 16:]
    captured, outputs = [], []
    hook = block.attn.to_out.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            for residual in residuals:
                outputs.append(block(residual, None, modulation))
    finally:
        hook.remove()
    kwargs = dict(recipient_residual=residuals[0], donor_residual=residuals[1],
        recipient_operands=captured[0], donor_operands=captured[1], gate=gate,
        projection=block.attn.to_out, attention_width=8, text_rows=2)
    with torch.inference_mode():
        self_result = helper()(**kwargs, mask="000")
        all_result = helper()(**kwargs, mask="111")
        attention = helper()(**kwargs, mask="010")
        mlp = helper()(**kwargs, mask="001")
    assert torch.equal(self_result["updated"], outputs[0])
    assert torch.equal(all_result["updated"][:, 2:], outputs[1][:, 2:])
    assert torch.equal(all_result["updated"][:, :2], outputs[0][:, :2])
    assert torch.equal(attention["operands"][:, 2:, :8], captured[1][:, 2:, :8])
    assert torch.equal(attention["operands"][:, 2:, 8:], captured[0][:, 2:, 8:])
    assert torch.equal(mlp["operands"][:, 2:, :8], captured[0][:, 2:, :8])
    assert not torch.equal(attention["updated"], mlp["updated"])
    with pytest.raises(ValueError, match="mask"):
        helper()(**kwargs, mask="attn")

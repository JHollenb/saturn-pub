"""Conditioning composition and observed-entry linear projection mechanics."""
import runpy
from pathlib import Path
import numpy as np
import torch

helper=runpy.run_path(str(Path(__file__).parents[1]/"qualification/flux_conditioning_program.py"))


def test_conditioning_interaction_adds_back_to_joint_without_mutating_sources():
    gen=torch.Generator().manual_seed(65)
    values={k:torch.randn((1,7,16),generator=gen).to(torch.bfloat16).float().numpy() for k in ("base","color","shape","joint")}
    before={k:v.copy() for k,v in values.items()};cases=helper["conditioning_cases"](values)
    correction=values["joint"].astype(np.float64)-cases["additive"]
    np.testing.assert_array_equal(helper["bf16"](cases["additive"]+correction),values["joint"])
    for k,v in before.items():np.testing.assert_array_equal(v,values[k])


def test_projection_prediction_needs_no_current_module_call_and_matches_tiny_native():
    torch.manual_seed(321);layer=torch.nn.Linear(16,12,bias=True).to(torch.bfloat16).eval();x=torch.randn(1,9,16).to(torch.bfloat16)
    def guard(module,args):raise AssertionError("native projector entered during forecast")
    handle=layer.register_forward_pre_hook(guard)
    try:predicted=helper["predict_projection"](x.float().numpy(),layer.weight.detach().float().numpy(),layer.bias.detach().float().numpy(),chunk=5)
    finally:handle.remove()
    with torch.inference_mode():actual=layer(x).float().numpy()
    assert np.max(np.abs(predicted-actual))<.016
    other=helper["predict_projection"](x.float().numpy(),layer.weight.detach().float().numpy(),layer.bias.detach().float().numpy(),chunk=12)
    np.testing.assert_array_equal(predicted,other)

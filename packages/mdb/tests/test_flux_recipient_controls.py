"""Recipient-only projection recipes and own-C native continuation mechanics."""
from __future__ import annotations

import inspect
import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock
from saturn_pub import Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.native_navigation import NativeCutNavigator

Q = Path(__file__).parents[1] / "qualification"
candidate = runpy.run_path(str(Q / "flux_parallel_merge.py"))["recipient_candidate"]
worker = runpy.run_path(str(Q / "flux_merge_campaign.py"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_own_projection_identity_gains_text_preservation_and_bias_use_native_full_shape(dtype, monkeypatch):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(263)
        block = Flux2SingleTransformerBlock(8, 2, 4, mlp_ratio=2.).eval().to(dtype)
        original = block.attn.to_out
        block.attn.to_out = torch.nn.Linear(original.in_features, original.out_features, bias=True).to(dtype).eval()
        block.attn.to_out.weight.data.copy_(original.weight)
        block.attn.to_out.bias.data.fill_(.2)
        residual = torch.randn(1, 5, 8).to(dtype)
        modulation = torch.randn(1, 24).to(dtype)
    observed = []
    hook = block.attn.to_out.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    try:
        with torch.inference_mode():
            native = block(residual, None, modulation)
    finally:
        hook.remove()
    operands, gate = observed[0], modulation[:, None, 16:]
    original_residual, original_operands = residual.clone(), operands.clone()
    linear = torch.nn.functional.linear
    calls = []

    def watched(value, weight, bias):
        calls.append((tuple(value.shape), weight.data_ptr(), bias.data_ptr()))
        return linear(value, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", watched)
    for label, gains in worker["RECIPIENT_GAINS"].items():
        calls.clear()
        with torch.inference_mode():
            result = candidate(residual=residual, operands=operands, gate=gate, projection=block.attn.to_out,
                               attention_width=8, text_rows=2, gains=gains)
        assert calls == [(tuple(operands.shape), block.attn.to_out.weight.data_ptr(), block.attn.to_out.bias.data_ptr())]
        torch.testing.assert_close(result["updated"][:, :2], native[:, :2], rtol=0., atol=0.)
        torch.testing.assert_close(result["operands"][:, 2:, :8], operands[:, 2:, :8] * gains[0], rtol=0., atol=0.)
        torch.testing.assert_close(result["operands"][:, 2:, 8:], operands[:, 2:, 8:] * gains[1], rtol=0., atol=0.)
        if label == "11":
            torch.testing.assert_close(result["updated"], native, rtol=0., atol=0.)
        if label == "00":
            expected_bias = block.attn.to_out.bias.expand_as(result["projection"][:, 2:])
            torch.testing.assert_close(result["projection"][:, 2:], expected_bias, rtol=0., atol=0.)
            torch.testing.assert_close(result["updated"][:, 2:], residual[:, 2:] + gate * expected_bias, rtol=0., atol=0.)
    torch.testing.assert_close(residual, original_residual, rtol=0., atol=0.)
    torch.testing.assert_close(operands, original_operands, rtol=0., atol=0.)


def test_recipe_has_no_donor_argument_or_lookup_and_refuses_invalid_gains():
    assert set(inspect.signature(candidate).parameters) == {"residual", "operands", "gate", "projection", "attention_width", "text_rows", "gains"}
    args = dict(residual=torch.zeros(1, 3, 2), operands=torch.ones(1, 3, 4), gate=torch.ones(1, 1, 2),
                projection=torch.nn.Linear(4, 2), attention_width=2, text_rows=1, gains=(1., .5))
    with torch.inference_mode():
        first = candidate(**args)
        candidate.__globals__["donor_residual"] = torch.full((1, 3, 2), 1000.)
        candidate.__globals__["donor_operands"] = object()
        second = candidate(**args)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0., atol=0.)
    with pytest.raises(TypeError, match="donor"):
        candidate(**args, donor_residual=torch.zeros(1, 3, 2))
    for bad in ((1., float("nan")), (1., 2.), (True, 1.), (1.,)):
        with pytest.raises(ValueError, match="gains"):
            candidate(**{**args, "gains": bad})


def test_adjacent_mode_is_narrow_and_keeps_original_block4_mode():
    config = {"panel": "shape-step1-recipient-controls", "block": 5, "reset_step": 2,
        "prefix_selector": {"logical_step": 1, "phase": "single", "index": 0},
        "selector": {"logical_step": 1, "phase": "single", "index": 5},
        "recipient_gain_rows": {"self." + name: list(gains) for name, gains in worker["RECIPIENT_GAINS"].items()}}
    assert worker["validate_shape_writer_config"](config)
    assert worker["validate_shape_writer_config"]({**config, "panel": "shape-step1-writer", "block": 4,
                                                 "selector": {"logical_step": 1, "phase": "single", "index": 4}})
    for changed in ({"block": 4}, {"reset_step": 3}, {"selector": {"logical_step": 0, "phase": "single", "index": 5}},
                    {"recipient_gain_rows": {"self.11": [1., 0.]}}):
        with pytest.raises(ValueError):
            worker["validate_shape_writer_config"]({**config, **changed})
    assert not worker["validate_shape_writer_config"]({})


def test_recipient_own_C_skips_reset_and_common_future_resets_once_before_native_read(tmp_path):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(269)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        base_C = torch.randn(1, 3, 6)
        session = adapter.session(latent=torch.randn(1, 4, 4), conditioning=base_C + .4,
            img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 666., 333.]), sigmas=torch.tensor([1., .666, .333, 0.]))
    NativeCutNavigator(session, max_steps=100).until(logical_step=1, phase="readout")
    parent = session.capture()
    store = LocalStore(tmp_path)
    results = {}
    for policy in ("recipient-own-C", "common-base-reset"):
        nav = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=100, on_receipt=store.receipt)
        resets, actual_next_text = [], None
        while nav.point["logical_step"] < 3:
            point = nav.point
            if worker["requires_conditioning_reset"]({"panel": "shape-step1-recipient-controls"}, {"continuation_policy": policy}) and point["logical_step"] == 2 and point["phase"] == "project":
                before = nav.capture()
                resets.append(worker["reset_shape_conditioning"](nav, base_C, step=2, name="common.C",
                    source_cut="observed-base-C", store=store))
                assert describe(nav.capture().payload) == describe({**before.payload, "conditioning": base_C})
            nav.step()
            if point["logical_step"] == 2 and point["phase"] == "project":
                actual_next_text = nav.session.read("text")
        results[policy] = (resets, actual_next_text, nav.capture())
    own, common = results["recipient-own-C"], results["common-base-reset"]
    assert len(own[0]) == 0 and len(common[0]) == 1
    assert torch.equal(own[2].payload["conditioning"], parent.payload["conditioning"])
    assert torch.equal(common[2].payload["conditioning"], base_C)
    assert not torch.equal(own[1], common[1])
    assert not torch.equal(own[2].payload["latent"], common[2].payload["latent"])

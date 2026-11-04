"""Tiny native sequential joint writes, own-state Acts, and live VAE previews."""
from __future__ import annotations

import io
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.configuration_utils import FrozenDict
from PIL import Image
from saturn_pub import Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.native_navigation import NativeCutNavigator

QUALIFICATION = Path(__file__).parents[1] / "qualification"
q = runpy.run_path(str(QUALIFICATION / "flux_joint_self.py"))
campaign = runpy.run_path(str(QUALIFICATION / "flux_joint_self_campaign.py"))


def specimen(dtype=torch.float32, pipeline=None):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(229)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=2, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(dtype)
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation", pipeline=pipeline)
        session = adapter.session(latent=torch.randn(1, 4, 4, dtype=dtype), conditioning=torch.randn(1, 3, 6, dtype=dtype),
            img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 500.]), sigmas=torch.tensor([1., .5, 0.]))
    nav = NativeCutNavigator(session, max_steps=30)
    nav.until(logical_step=0, phase="joint", index=1)
    return model, adapter, nav


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_output_scaling_own_hold_and_current_text(dtype):
    model, _, nav = specimen(dtype)
    cut = nav.capture()
    context = describe(cut.payload["context"])
    block = model.transformer_blocks[1]
    native = q["joint_candidate_self"](block, cut)
    recompute = q["joint_candidate_self"](block, cut, attention_gain=0.)
    hold = q["joint_candidate_self"](block, cut, attention_gain=0., mlp_policy="hold-native", held_mlp=native["held_mlp"])
    half = q["joint_candidate_self"](block, cut, attention_gain=.5, mlp_gain=.5)
    mzero = q["joint_candidate_self"](block, cut, mlp_gain=0.)
    both = q["joint_candidate_self"](block, cut, attention_gain=0., mlp_gain=0.)
    nav.step()
    assert torch.equal(native["image"], nav.session.read("image"))
    assert torch.equal(native["text"], nav.session.read("text"))
    assert torch.equal(recompute["ports"]["intermediate"], cut.payload["image"])
    assert torch.equal(recompute["ports"]["mlp_input"], hold["ports"]["mlp_input"])
    assert not torch.equal(recompute["ports"]["mlp_input"], native["ports"]["mlp_input"])
    assert torch.equal(hold["ports"]["mlp_selected"], native["ports"]["mlp_raw"])
    assert torch.equal(hold["ports"]["mlp_raw"], recompute["ports"]["mlp_raw"])
    assert not torch.equal(hold["image"], recompute["image"])
    assert torch.equal(half["ports"]["attention_used"], .5 * native["ports"]["attention_raw"])
    assert torch.equal(half["ports"]["mlp_used"], .5 * half["ports"]["mlp_raw"])
    assert torch.equal(mzero["image"], native["ports"]["intermediate"])
    assert torch.equal(both["image"], cut.payload["image"])
    for value in (recompute, hold, half, mzero, both):
        assert value["equation_exact"] and torch.equal(value["text"], native["text"])
        assert torch.equal(value["ports"]["gate_attention"], native["ports"]["gate_attention"])
        assert torch.equal(value["ports"]["gate_mlp"], native["ports"]["gate_mlp"])
    assert describe(cut.payload["context"]) == context
    assert not block.attn._forward_hooks and not block.ff._forward_hooks and not block.norm2._forward_pre_hooks


def test_held_parent_validation_and_exception_hook_cleanup(monkeypatch):
    model, _, nav = specimen()
    block, cut = model.transformer_blocks[1], nav.capture()
    native = q["joint_candidate_self"](block, cut)
    with pytest.raises(ValueError, match="same parent"):
        q["joint_candidate_self"](block, cut, mlp_policy="hold-native", held_mlp={**native["held_mlp"], "parent": "foreign"})
    with pytest.raises(ValueError, match="recompute"):
        q["joint_candidate_self"](block, cut, held_mlp=native["held_mlp"])
    with pytest.raises(ValueError, match="finite"):
        q["joint_candidate_self"](block, cut, attention_gain=float("nan"))

    def broken(*args, **kwargs):
        raise RuntimeError("native FF failure")

    monkeypatch.setattr(block.ff, "forward", broken)
    with pytest.raises(RuntimeError, match="native FF"):
        q["joint_candidate_self"](block, cut)
    assert not block.attn._forward_hooks and not block.ff._forward_hooks and not block.ff._forward_pre_hooks
    assert not block.norm2._forward_pre_hooks


def test_typed_own_text_rollback_exact_closure_and_wrong_clock(tmp_path):
    _, adapter, nav = specimen()
    parent = nav.capture()
    nav.step()
    post = nav.capture()
    store = LocalStore(tmp_path)
    value = post.payload["image"] * .5
    image = q["install_joint_port"](nav, "image", value, name="own-image", store=store, expected_point=post.execution_point)
    text = q["install_joint_port"](nav, "text", parent.payload["text"], name="own-pre-text", store=store, expected_point=post.execution_point)
    expected = {**post.payload, "image": value, "text": parent.payload["text"]}
    assert describe(nav.capture().payload) == describe(expected)
    assert image["post_act"] == text["parent"]
    assert image["receipt"]["operation"] == text["receipt"]["operation"] == "apply"
    assert torch.equal(nav.capture().payload["conditioning"], parent.payload["conditioning"])
    restored = Session.from_cut(adapter, nav.capture())
    assert describe(restored.capture().payload) == describe(expected)
    before = nav.capture().fingerprint
    with pytest.raises(ValueError, match="clock"):
        q["install_joint_port"](nav, "text", parent.payload["text"], name="wrong", store=store,
            expected_point={**dict(post.execution_point), "logical_step": 1})
    assert nav.capture().fingerprint == before


def test_live_preview_native_final_parity_and_no_clock_mutation():
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.bn = torch.nn.BatchNorm2d(4)
            self.config = FrozenDict(batch_norm_eps=1e-5)
            self.calls = 0

        def decode(self, value, return_dict):
            self.calls += 1
            return (value[:, :3].tanh(),)

    def postprocess(value, output_type):
        assert output_type == "pil"
        array = ((value[0].float().permute(1, 2, 0).numpy() / 2 + .5).clip(0, 1) * 255).round().astype("uint8")
        return [Image.fromarray(array)]

    pipe = SimpleNamespace(vae=VAE(), vae_scale_factor=1,
        _unpack_latents_with_ids=lambda latent, ids, height, width: latent.transpose(1, 2).reshape(1, 4, height, width),
        _unpatchify_latents=lambda latent: latent,
        image_processor=SimpleNamespace(postprocess=postprocess))
    _, adapter, nav = specimen(pipeline=pipe)
    nav.until(logical_step=1, phase="project")
    live = nav.capture()
    q["decode_live_preview"](pipe, live, height=4, width=4)
    assert nav.capture().fingerprint == live.fingerprint
    adapter.finish(nav.session)
    final = nav.capture()
    actual = adapter.decode(nav.session, height=4, width=4)
    preview = q["decode_live_preview"](pipe, final, height=4, width=4)
    assert np.array_equal(np.asarray(actual), np.asarray(preview))
    left, right = io.BytesIO(), io.BytesIO()
    actual.save(left, format="PNG")
    preview.save(right, format="PNG")
    assert left.getvalue() == right.getvalue() and pipe.vae.calls == 3
    assert nav.capture().fingerprint == final.fingerprint


def test_fixed_fresh_campaign_config_contract():
    config = {"selector": {"logical_step": 1, "phase": "joint", "index": 4}, "branches": campaign["BRANCHES"],
        "preview_rows": campaign["PREVIEW_ROWS"], "preview_steps": [2, 3], "seed": 47, "steps": 4,
        "height": 512, "width": 512, "prompt": "one fresh own prompt", "max_adapter_steps": 1200,
        "max_candidate_calls": 13, "max_final_images": 12, "max_preview_images": 8, "max_images": 20,
        "max_vae_calls": 21, "max_wall_seconds": 840}
    campaign["validate_config"](config)
    assert len(config["branches"]) == 12
    for change in ({"selector": {"logical_step": 1, "phase": "single", "index": 4}},
                   {"max_vae_calls": 20}, {"preview_steps": [1, 3]}, {"branches": campaign["BRANCHES"][:-1]}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config, **change})

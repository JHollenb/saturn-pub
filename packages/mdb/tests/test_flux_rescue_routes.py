"""Actual tiny native joint0 ports and same-history image/text route closure."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.models.transformers import transformer_flux2
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1] / "qualification"
q = runpy.run_path(str(HERE / "flux_rescue_routes.py"))
joint = runpy.run_path(str(HERE / "flux_joint_self.py"))
campaign = runpy.run_path(str(HERE / "flux_rescue_routes_campaign.py"))


def specimen(dtype=torch.float32):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(821)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=5, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(dtype)
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        session = adapter.session(latent=torch.randn(1, 4, 4, dtype=dtype), conditioning=torch.randn(1, 3, 6, dtype=dtype),
            img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 800., 600., 300.]), sigmas=torch.tensor([1., .8, .6, .3, 0.]))
    own = NativeCutNavigator(session, max_steps=48)
    own.until(logical_step=3, phase="project")
    pre = own.capture()
    zero = NativeCutNavigator(Session.from_cut(adapter, pre), max_steps=40)
    zero.apply(Act.replace("conditioning", torch.zeros_like(pre.payload["conditioning"]), name="own-C-zero"))
    zero.step()
    own.step()
    return model, adapter, zero.capture(), own.capture()


def capture(model, cut, source=None):
    return q["capture_candidate"](model.transformer_blocks[0], cut,
        joint_candidate=joint["joint_candidate_self"], text_kv_source=source)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_positioned_kv_partition_recipe_and_image_parity(dtype):
    model, adapter, zero, own = specimen(dtype)
    q["validate_paired_cuts"](zero, own)
    original = transformer_flux2.dispatch_attention_fn
    baseline, source = capture(model, zero), capture(model, own)
    kv = capture(model, zero, {"parent": own.fingerprint, "text_rows": source["text_rows"], **source["observed_qkv"]})
    text = kv["text_rows"]
    assert transformer_flux2.dispatch_attention_fn is original
    assert torch.equal(kv["used_qkv"]["q"], baseline["observed_qkv"]["q"])
    for key in ("k", "v"):
        assert torch.equal(kv["used_qkv"][key][:, :text], source["observed_qkv"][key][:, :text])
        assert torch.equal(kv["used_qkv"][key][:, text:], baseline["observed_qkv"][key][:, text:])
        assert torch.equal(source["observed_qkv"][key][:, text:], baseline["observed_qkv"][key][:, text:])
    assert torch.equal(kv["image"], source["image"])
    assert not torch.equal(kv["text"], source["text"])
    assert all(kv["partition_controls"].values()) and kv["dispatch_calls"] == 1
    for cut, candidate in ((zero, baseline), (own, source)):
        native = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=2)
        native.step()
        assert torch.equal(native.session.read("image"), candidate["image"])
        assert torch.equal(native.session.read("text"), candidate["text"])


def test_dispatch_and_scoped_hooks_restore_after_invalid_source():
    model, _, zero, own = specimen()
    original = transformer_flux2.dispatch_attention_fn
    with pytest.raises(ValueError, match="geometry"):
        capture(model, zero, {"parent": own.fingerprint, "text_rows": 3, "k": torch.zeros(1), "v": torch.zeros(1)})
    assert transformer_flux2.dispatch_attention_fn is original
    block = model.transformer_blocks[0]
    assert not block.attn._forward_hooks and not block.ff._forward_hooks
    assert not block.ff._forward_pre_hooks and not block.norm2._forward_pre_hooks


def test_late_common_parent_factorial_closure_and_coherent_future(tmp_path):
    _, adapter, zero, own = specimen(torch.bfloat16)
    fronts = {}
    for name, cut in (("zero", zero), ("own", own)):
        n = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=12)
        n.until(logical_step=3, phase="single", index=0)
        assert n.steps == 5
        fronts[name] = n.capture()
    store = LocalStore(tmp_path)
    mixed = NativeCutNavigator(Session.from_cut(adapter, fronts["zero"]), max_steps=12)
    action = q["install_route_port"](mixed, "image", fronts["own"].payload["image"], name="I1T0",
        store=store, joint_install=joint["install_joint_port"])
    assert action["parent"] == fronts["zero"].fingerprint
    expected = {**fronts["zero"].payload, "image": fronts["own"].payload["image"]}
    assert describe(mixed.capture().payload) == describe(expected)
    second = q["install_route_port"](mixed, "text", fronts["own"].payload["text"], name="I1T1",
        store=store, joint_install=joint["install_joint_port"])
    assert second["parent"] == action["post_act"]
    assert torch.equal(mixed.session.read("conditioning"), zero.payload["conditioning"])
    reference = NativeCutNavigator(Session.from_cut(adapter, fronts["own"]), max_steps=12)
    mixed.until(logical_step=4, phase="project")
    reference.until(logical_step=4, phase="project")
    assert mixed.steps == reference.steps == 4
    assert torch.equal(mixed.session.read("latent"), reference.session.read("latent"))


def test_early_image_only_write_preserves_native_text_and_other_slots(tmp_path):
    model, adapter, zero, own = specimen()
    source = capture(model, own)
    candidate = capture(model, zero, {"parent": own.fingerprint, "text_rows": source["text_rows"], **source["observed_qkv"]})
    n = NativeCutNavigator(Session.from_cut(adapter, zero), max_steps=2)
    n.step()
    post = n.capture()
    action = q["install_route_port"](n, "image", candidate["image"], name="KV-image-only",
        store=LocalStore(tmp_path), joint_install=joint["install_joint_port"])
    assert action["parent"] == post.fingerprint
    assert describe(n.capture().payload) == describe({**post.payload, "image": candidate["image"]})


def test_wrong_source_clock_or_write_clock_is_refused_without_mutation(tmp_path):
    _, adapter, zero, own = specimen()
    n = NativeCutNavigator(Session.from_cut(adapter, zero), max_steps=3)
    before = n.capture().fingerprint
    with pytest.raises(ValueError, match="route write"):
        q["install_route_port"](n, "image", own.payload["image"], name="too-early",
            store=LocalStore(tmp_path), joint_install=joint["install_joint_port"])
    assert n.capture().fingerprint == before
    n.step()
    with pytest.raises(ValueError, match="joint0 clock"):
        q["validate_paired_cuts"](n.capture(), own)


def test_fixed_panel_physical_census_and_finite_budget_contract():
    branches = campaign["BRANCHES"]
    config = {"branches": branches, "input_cuts": {"zero": "a" * 64, "own": "b" * 64},
        "input_store": "input-store", "source_selection": "source-selection.json", "steps": 4,
        "height": 512, "width": 512, "seed": 47, "max_adapter_steps": 260, "max_acts": 11,
        "max_candidate_calls": 3, "max_images": 9, "max_vae_calls": 9, "max_wall_seconds": 600, "archive_compression_level": 1}
    campaign["validate_config"](config)
    assert len(branches) == 9 and sum(row["kind"] == "early" for row in branches) == 3
    assert 2 * 5 + 6 * 22 + 3 * 26 == 220
    assert sum(int(row["image_source"] is not None) + int(row["text_source"] is not None)
               for row in branches if row["kind"] != "reference") == 11
    for change in ({"max_adapter_steps": 219}, {"max_candidate_calls": 4}, {"branches": branches[:-1]}, {"max_images": True}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config, **change})

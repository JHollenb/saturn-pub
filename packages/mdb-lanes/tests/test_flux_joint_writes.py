"""Actual tiny joint consumer: sequential ports and typed image installation."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb_lanes.native_navigation import NativeCutNavigator

q = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_joint_writes.py"))


def specimen():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(229)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=2, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        session = adapter.session(latent=torch.randn(1, 4, 4), conditioning=torch.randn(1, 3, 6),
            img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 500.]), sigmas=torch.tensor([1., .5, 0.]))
    nav = NativeCutNavigator(session, max_steps=20)
    nav.until(logical_step=0, phase="joint", index=1)
    return model, adapter, nav


def test_actual_joint_equation_self_substitution_and_native_parity(tmp_path):
    model, adapter, nav = specimen()
    before = nav.capture()
    block = model.transformer_blocks[1]
    native = q["joint_candidate"](block, before)
    edited = q["joint_candidate"](block, before, attention=native["ports"]["attention"], mlp=native["ports"]["mlp"])
    nav.step()
    assert torch.equal(native["image"], nav.session.read("image"))
    assert torch.equal(native["text"], nav.session.read("text"))
    assert torch.equal(edited["image"], native["image"])
    assert native["equation_exact"] and edited["equation_exact"]
    assert not block.attn._forward_hooks and not block.ff._forward_hooks
    store = LocalStore(tmp_path)
    record = q["install_image"](nav, edited["image"], name="self", store=store)
    assert record["undeclared_slots_exact"] and record["receipt"]["operation"] == "apply"
    assert describe(nav.capture().payload) == describe(Session.from_cut(adapter, nav.capture()).capture().payload)


def test_attention_changes_receiving_mlp_but_not_current_text():
    model, _, nav = specimen()
    before = nav.capture()
    block = model.transformer_blocks[1]
    native = q["joint_candidate"](block, before)
    edited = q["joint_candidate"](block, before, attention=native["ports"]["attention"] + .2)
    assert not torch.equal(edited["ports"]["mlp"], native["ports"]["mlp"])
    assert not torch.equal(edited["image"], native["image"])
    assert torch.equal(edited["text"], native["text"])
    assert nav.capture().fingerprint == before.fingerprint


def test_mlp_only_replacement_keeps_attention_and_removes_hooks():
    model, _, nav = specimen()
    block, before = model.transformer_blocks[1], nav.capture()
    native = q["joint_candidate"](block, before)
    edited = q["joint_candidate"](block, before, mlp=torch.zeros_like(native["ports"]["mlp"]))
    assert torch.equal(edited["ports"]["attention"], native["ports"]["attention"])
    assert not torch.equal(edited["image"], native["image"])
    with pytest.raises(ValueError, match="contract"):
        q["joint_candidate"](block, before, mlp=torch.zeros(1))
    assert not block.attn._forward_hooks and not block.ff._forward_hooks


def test_reconstruction_control_restores_native_future_despite_different_text(tmp_path):
    model, adapter, base = specimen()
    cut = base.capture()
    native = q["joint_candidate"](model.transformer_blocks[1], cut)
    changed = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=20)
    changed.apply(Act.replace("text", cut.payload["text"] + .3, name="changed-text"))
    edited = q["joint_candidate"](model.transformer_blocks[1], changed.capture(),
        attention=native["ports"]["attention"], mlp=native["ports"]["mlp"])
    assert torch.equal(edited["image"], native["image"])
    assert not torch.equal(edited["text"], native["text"])
    base.step()
    changed.step()
    q["install_image"](changed, edited["image"], name="reconstruct-I", store=LocalStore(tmp_path))
    changed.apply(Act.replace("text", base.session.read("text"), name="common-T"))
    assert describe(changed.capture().payload) == describe(base.capture().payload)
    adapter.finish(base.session)
    adapter.finish(changed.session)
    assert torch.equal(base.session.read("latent"), changed.session.read("latent"))

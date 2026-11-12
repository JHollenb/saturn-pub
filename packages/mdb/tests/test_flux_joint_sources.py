"""Tiny real joint attention: exact source joins and text disposal control."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from diffusers.models.transformers import transformer_flux2 as native
from saturn_pub import Act, Session
from saturn_pub.values import describe

from mdb.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1]
specimen = runpy.run_path(str(HERE / "tests/test_flux_joint_writes.py"))["specimen"]
candidate = runpy.run_path(str(HERE / "qualification/flux_joint_sources.py"))["source_candidate"]


def sources():
    model, adapter, base = specimen()
    b = base.capture()
    changed = NativeCutNavigator(Session.from_cut(adapter, b), max_steps=20)
    changed.apply(Act.replace("text", b.payload["text"] + torch.linspace(-.3, .3, b.payload["text"].shape[-1]), name="text-source"))
    changed.apply(Act.replace("image", b.payload["image"] + torch.linspace(.2, -.2, b.payload["image"].shape[-1]), name="image-source"))
    return model.transformer_blocks[1], adapter, b, changed.capture()


def test_native_capture_and_qkv_all_join_attention_output():
    block, _, b, d = sources()
    original = native.dispatch_attention_fn
    left, right = candidate(block, b), candidate(block, d)
    same = candidate(block, d, donor=right["qkv"]["observed"], mask="111")
    assert torch.equal(same["image"], right["image"])
    edited = candidate(block, d, donor=left["qkv"]["observed"], mask="111")
    assert torch.equal(edited["ports"]["attention"], left["ports"]["attention"])
    direct = candidate(block, d, attention=left["ports"]["attention"])
    assert torch.equal(edited["image"], direct["image"])
    assert native.dispatch_attention_fn is original


def test_text_query_has_no_image_effect_and_common_text_disposal_closes_future():
    block, adapter, b, d = sources()
    base, changed = candidate(block, b), candidate(block, d)
    edited = candidate(block, d, donor=base["qkv"]["observed"], text_query=True)
    assert torch.equal(edited["image"], changed["image"])
    assert not torch.equal(edited["text"], changed["text"])
    native_branch = NativeCutNavigator(Session.from_cut(adapter, d), max_steps=20)
    native_branch.step()
    another = Session.from_cut(adapter, native_branch.capture())
    native_branch.apply(Act.replace("text", base["text"], name="same-text"))
    another.apply(Act.replace("image", edited["image"], name="query-image"))
    another.apply(Act.replace("text", base["text"], name="same-text"))
    assert describe(native_branch.capture().payload) == describe(another.capture().payload)
    adapter.finish(native_branch.session)
    adapter.finish(another)
    assert torch.equal(native_branch.session.read("latent"), another.read("latent"))


def test_locked_mlp_is_distinct_from_attention_induced_mlp_and_scope_restores():
    block, _, b, d = sources()
    base, changed = candidate(block, b), candidate(block, d)
    recomputed = candidate(block, d, attention=base["ports"]["attention"])
    locked = candidate(block, d, attention=base["ports"]["attention"], mlp=changed["ports"]["mlp"])
    assert not torch.equal(recomputed["ports"]["mlp"], changed["ports"]["mlp"])
    assert not torch.equal(recomputed["image"], locked["image"])
    original = native.dispatch_attention_fn
    bad = {**base["qkv"]["observed"], "q": torch.zeros(1)}
    with pytest.raises(ValueError, match="contract"):
        candidate(block, d, donor=bad, mask="100")
    assert native.dispatch_attention_fn is original
    assert not block.attn._forward_hooks and not block.ff._forward_hooks

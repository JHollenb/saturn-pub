"""Native tiny-module proof of fixed-parent consumer reuse and its limits."""
import runpy
from pathlib import Path

import pytest
import torch
from saturn_pub import Act, Session
from saturn_pub.store import LocalStore

from mdb_lanes.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1]
q = runpy.run_path(str(HERE / "qualification/flux_predicted_future.py"))
legacy = runpy.run_path(str(HERE / "qualification/flux_single_contraction.py"))
replace = runpy.run_path(str(HERE / "qualification/flux_mlp_replacement.py"))
tiny = runpy.run_path(str(HERE / "tests/test_flux_single_contexts.py"))


def setup():
    model, adapter, session = tiny["tiny_session"]()
    nav = NativeCutNavigator(session, max_steps=112)
    nav.until(logical_step=3, phase="single", index=6)
    cut = nav.capture()
    block = model.single_transformer_blocks[6]
    native = legacy["capture_candidate"](block, cut)
    return adapter, nav, cut, block, native


def test_cached_consumption_matches_native_for_all_doses_without_entering_mlp_or_attention():
    _adapter, _nav, cut, block, native = setup()
    t, h = native["text_rows"], native["attention_width"]
    field = native["ports"]["original_concat"][:, t:, h:]
    for dose in q["dose_values"]():
        values = (field.float() + dose * .125).to(torch.bfloat16)
        full = replace["capture_replacement"](block, cut, native, values,
            capture_path=str(HERE / "qualification/flux_single_contraction.py"))
        def forbidden(*args):
            raise AssertionError("cached consumer entered a bypassed module")
        handles = [module.register_forward_pre_hook(forbidden) for module in
            (block.norm, block.attn.to_qkv_mlp_proj, block.attn.mlp_act_fn, block.attn.norm_q)]
        try:
            cached = q["consume_cached"](block, cut, native, values)
        finally:
            for handle in handles:
                handle.remove()
        assert torch.equal(cached["image"], full["image"])
        assert torch.equal(cached["text"], full["text"])


def test_cached_measured_identity_closes_typed_native_future(tmp_path):
    adapter, nav, cut, block, native = setup()
    t, h = native["text_rows"], native["attention_width"]
    cached = q["consume_cached"](block, cut, native, native["ports"]["original_concat"][:, t:, h:])
    nav.step()
    branch = NativeCutNavigator(Session.from_cut(adapter, nav.capture()), max_steps=15)
    legacy["install_port"](branch, "image", cached["image"], name="cached-identity", store=LocalStore(tmp_path))
    nav.until(logical_step=4, phase="project")
    branch.until(logical_step=4, phase="project")
    assert torch.equal(nav.session.read("latent"), branch.session.read("latent"))


def test_reuse_rejects_another_parent_or_nonfinite_operand():
    adapter, _nav, cut, block, native = setup()
    t, h = native["text_rows"], native["attention_width"]
    field = native["ports"]["original_concat"][:, t:, h:]
    altered = NativeCutNavigator(Session.from_cut(adapter, cut), max_steps=1)
    altered.apply(Act.replace("image", cut.payload["image"] + 1, name="different-recipient"))
    with pytest.raises(ValueError, match="parent"):
        q["consume_cached"](block, altered.capture(), native, field)
    bad = field.clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="profile"):
        q["consume_cached"](block, cut, native, bad)

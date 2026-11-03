"""Real tiny consumer qualification of the step1/common-future instrument."""
import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.native_navigation import NativeCutNavigator

Q = Path(__file__).parents[1] / "qualification"
worker = runpy.run_path(str(Q / "flux_handoff_campaign.py"))
project = worker["project_matched_sources"]
reset = worker["reset_conditioning"]


def parents():
    torch.set_num_threads(1)
    torch.manual_seed(171)
    model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    source = torch.randn(1, 3, 6)
    s = adapter.session(latent=torch.randn(1, 4, 4), conditioning=source + .4,
        img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
        timesteps=torch.tensor([1000., 750., 500., 250.]), sigmas=torch.tensor([1., .75, .5, .25, 0.]))
    NativeCutNavigator(s, max_steps=6).until(logical_step=1, phase="project")
    changed = s.capture()
    s.apply(Act.replace("conditioning", source, name="initial.C.base"))
    return adapter, s.capture(), changed


def test_projected_pair_and_common_reset_close_both_carrier_control(tmp_path):
    adapter, base, changed = parents()
    store = LocalStore(tmp_path)
    navs, projected = project(adapter, base, changed, on_receipt=store.receipt, max_steps=100)
    assert torch.equal(projected[0].payload["image"], projected[1].payload["image"])
    assert torch.equal(projected[0].payload["latent"], projected[1].payload["latent"])
    assert not torch.equal(projected[0].payload["text"], projected[1].payload["text"])
    for n in navs:
        n.step()
    frontiers = [n.capture() for n in navs]

    def finish(n):
        while n.point["logical_step"] < 4:
            if n.point["logical_step"] == 2 and n.point["phase"] == "project":
                before = n.capture().payload
                row = reset(n, base.payload["conditioning"], step=2, name="future.C", store=store)
                assert row["receipt"]["operation"] == "apply"
                assert describe(n.capture().payload) == describe({**before, "conditioning": base.payload["conditioning"]})
            n.step()
        return n.session.read("latent")

    expected_base, expected_changed = [finish(n) for n in navs]
    for sham, expected in ((False, expected_base), (True, expected_changed)):
        n = NativeCutNavigator(Session.from_cut(adapter, frontiers[1]), max_steps=100, on_receipt=store.receipt)
        for slot in ("text", "image"):
            n.apply(Act.replace(slot, frontiers[1 if sham else 0].payload[slot], name="restore." + slot))
        assert torch.equal(finish(n), expected)


def test_source_reset_refuses_wrong_clock_without_mutation(tmp_path):
    adapter, base, _ = parents()
    nav = NativeCutNavigator(Session.from_cut(adapter, base), max_steps=100)
    before = nav.capture()
    with pytest.raises(ValueError, match="declared project"):
        reset(nav, base.payload["conditioning"], step=2, name="bad", store=LocalStore(tmp_path))
    assert nav.capture().fingerprint == before.fingerprint


def test_source_pair_refuses_changed_latent_before_any_projection():
    adapter, base, changed = parents()
    s = Session.from_cut(adapter, changed)
    s.apply(Act.replace("latent", changed.payload["latent"] + .1, name="wrong.latent"))
    with pytest.raises(ValueError, match="nuisance"):
        project(adapter, base, s.capture(), on_receipt=None, max_steps=100)

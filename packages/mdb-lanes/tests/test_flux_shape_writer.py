"""Tiny native step1 writer: original-cut lineage and exact future C reset."""
from __future__ import annotations

import runpy
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Act, Session, StateCut
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb_lanes.native_navigation import NativeCutNavigator

worker = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_merge_campaign.py"))


def origins():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(219)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=6,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        session = adapter.session(latent=torch.randn(1, 4, 4), conditioning=torch.randn(1, 3, 6),
            img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
            timesteps=torch.tensor([1000., 750., 500., 250.]), sigmas=torch.tensor([1., .75, .5, .25, 0.]))
    NativeCutNavigator(session, max_steps=100).until(logical_step=1, phase="project")
    common = session.capture()
    cuts = {}
    for name in ("base", "changed"):
        branch = Session.from_cut(adapter, common)
        if name == "changed":
            branch.apply(Act.replace("conditioning", common.payload["conditioning"] + .4, name="native-prefix-C"))
        nav = NativeCutNavigator(branch, max_steps=100)
        nav.until(logical_step=1, phase="single", index=0)
        cuts[name] = nav.capture()
    return adapter, cuts


def selected(adapter, cuts, store, on_receipt=None):
    return worker["advance_shape_sources"](adapter, cuts,
        prefix_selector={"logical_step": 1, "phase": "single", "index": 0},
        selector={"logical_step": 1, "phase": "single", "index": 4},
        store=store, max_steps=100, on_receipt=on_receipt)


def test_native_source_prefix_records_original_to_selected_chain_and_preserves_origins(tmp_path):
    adapter, cuts = origins()
    store, observed = LocalStore(tmp_path), []
    before = {name: cut.fingerprint for name, cut in cuts.items()}
    output, receipts = selected(adapter, cuts, store, observed.append)
    assert len(observed) == 8 and all(len(values) == 4 for values in receipts.values())
    for name in ("base", "changed"):
        previous = cuts[name].fingerprint
        for index, receipt in enumerate(receipts[name]):
            assert receipt["parent"] == previous and receipt["steps"] == 1
            assert receipt["transitions"][0]["before"]["index"] == index
            previous = receipt["result"]
        assert previous == output[name].fingerprint
        assert output[name].execution_point["index"] == 4
        assert cuts[name].fingerprint == before[name]
        assert store.load(output[name].fingerprint).fingerprint == output[name].fingerprint
    assert not torch.equal(output["base"].payload["conditioning"], output["changed"].payload["conditioning"])
    assert torch.equal(output["base"].payload["latent"], output["changed"].payload["latent"])


def test_typed_reset_at_step2_preserves_every_other_field_and_native_future_source(tmp_path):
    adapter, cuts = origins()
    store = LocalStore(tmp_path)
    sources, _ = selected(adapter, cuts, store)
    nav = NativeCutNavigator(Session.from_cut(adapter, sources["changed"]), max_steps=100, on_receipt=store.receipt)
    nav.step()  # selected single4; its actual changed post-block text is retained
    post = nav.capture()
    nav.until(logical_step=2, phase="project")
    before = nav.capture()
    assert torch.equal(before.payload["conditioning"], cuts["changed"].payload["conditioning"])
    unreset = Session.from_cut(adapter, before)
    row = worker["reset_shape_conditioning"](nav, cuts["base"].payload["conditioning"], step=2,
        name="branch.future-source-base", source_cut=cuts["base"].fingerprint, store=store)
    after = nav.capture()
    assert row["parent"] == before.fingerprint and row["post_act"] == after.fingerprint
    assert row["point"]["logical_step"] == 2 and row["point"]["phase"] == "project"
    assert row["source_cut"] == cuts["base"].fingerprint and row["undeclared_slots_exact"]
    expected = {**before.payload, "conditioning": cuts["base"].payload["conditioning"]}
    assert describe(after.payload) == describe(expected)
    assert store.load(row["post_act"]).fingerprint == after.fingerprint
    assert (tmp_path / "receipts" / (row["receipt_id"] + ".json")).is_file()
    coherent = Session.from_cut(adapter, before)
    coherent.apply(Act.replace("conditioning", cuts["base"].payload["conditioning"], name="independent-native-C-reference"))
    nav.step()
    coherent.continue_(1)
    unreset.continue_(1)
    assert describe(nav.capture().payload) == describe(coherent.capture().payload)
    assert not torch.equal(nav.session.read("text"), unreset.read("text"))
    assert torch.equal(nav.session.read("image"), unreset.read("image"))
    adapter.finish(nav.session)
    adapter.finish(coherent)
    assert torch.equal(nav.session.read("latent"), coherent.read("latent"))
    assert post.execution_point["index"] == 5


def test_wrong_reset_clock_refuses_without_mutation_or_receipt(tmp_path):
    adapter, cuts = origins()
    nav = NativeCutNavigator(Session.from_cut(adapter, cuts["changed"]), max_steps=100)
    before = nav.capture()
    store = LocalStore(tmp_path)
    with pytest.raises(ValueError, match="declared project"):
        worker["reset_shape_conditioning"](nav, cuts["base"].payload["conditioning"], step=2,
            name="wrong", source_cut=cuts["base"].fingerprint, store=store)
    assert nav.capture().fingerprint == before.fingerprint
    assert not list((tmp_path / "receipts").glob("*.json"))


@pytest.mark.parametrize("mismatch", ["latent", "clock"])
def test_prefix_rejects_nuisance_mismatch_before_native_work(tmp_path, mismatch):
    adapter, cuts = origins()
    cut = cuts["changed"]
    if mismatch == "latent":
        payload = {**cut.payload, "latent": cut.payload["latent"] + .1}
        changed = StateCut(cut.model_identity, cut.execution, cut.boundary, cut.parent, payload,
            execution_point=cut.execution_point, surface=cut.surface)
    else:
        branch = Session.from_cut(adapter, cut)
        branch.continue_(1)
        changed = branch.capture()
    bad = {"base": cuts["base"], "changed": changed}
    calls = []
    with pytest.raises(ValueError, match="clock|nuisance"):
        selected(adapter, bad, LocalStore(tmp_path), calls.append)
    assert not calls


def test_mode_is_explicit_and_wrong_clock_configuration_refuses():
    assert not worker["validate_shape_writer_config"]({})
    config = {"panel": "shape-step1-writer", "prefix_selector": {"logical_step": 1, "phase": "single", "index": 0},
              "selector": {"logical_step": 1, "phase": "single", "index": 4}, "block": 4, "reset_step": 2}
    assert worker["validate_shape_writer_config"](config)
    with pytest.raises(ValueError, match="declared"):
        worker["validate_shape_writer_config"]({**config, "reset_step": 3})

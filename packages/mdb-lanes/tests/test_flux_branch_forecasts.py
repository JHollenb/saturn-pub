"""Small real-consumer checks for the forecast panel and immutable evidence."""
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Session, StateCut
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore

Q = Path(__file__).parents[1] / "qualification"
panel = runpy.run_path(str(Q / "flux_branch_forecasts.py"))
functions = runpy.run_path(str(Q / "flux_functional_denoiser.py"))
compare = runpy.run_path(str(Q / "flux_trajectory_forecast_campaign.py"))["compare"]


def test_selected_port_hydration_verifies_source_manifest_page_and_blob(tmp_path):
    cut = StateCut(model_identity="tiny", execution={}, boundary="observed", parent=None,
                   _payload={"text": torch.tensor([[1., 2.]], dtype=torch.bfloat16)})
    store = LocalStore(tmp_path / "store")
    store.save(cut)
    record = json.loads((store.root / "cuts" / (cut.fingerprint + ".json")).read_text())
    page = json.loads((store.root / "pages" / (record["slots"]["text"] + ".json")).read_text())
    metadata = tmp_path / "carrier.json"
    metadata.write_text(json.dumps({"cut": record, "page": page}))
    blob = store.root / "blobs" / (page["tensor_blob"] + ".safetensors")
    port = panel["load_carrier"](metadata, blob, device="cpu")
    assert torch.equal(port["text"], cut.payload["text"])
    page["value"]["shape"] = [10]
    metadata.write_text(json.dumps({"cut": record, "page": page}))
    with pytest.raises(ValueError, match="identity"):
        panel["load_carrier"](metadata, blob, device="cpu")


def test_full_cache_is_immutable_detects_tampering_and_keeps_signed_error():
    cache = panel["BoundaryCache"](witness_values=4, max_bytes=100)
    source = torch.tensor([1., -2.])
    sealed = cache.snapshot("row", source, retain_full=True)
    source[0] = 9.
    assert np.array_equal(sealed["full_values"], [1., -2.])
    with pytest.raises(ValueError):
        cache.values["row"].setflags(write=True)
    result = cache.inspect("row", torch.tensor([2., -3.]), sealed, compare=compare)
    assert result["full_error"]["changed_elements"] == 2
    assert result["full_error"]["signed_mean"] == 0.
    cache.values["row"] = np.array([8., -2.], dtype=np.float32)
    with pytest.raises(ValueError, match="sealed"):
        cache.inspect("row", source, sealed, compare=compare)
    with pytest.raises(RuntimeError, match="budget"):
        cache.snapshot("too-big", torch.zeros(100))


def test_four_fresh_parent_futures_use_same_text_acts_and_close_functional_trajectory(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(124)
    model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    args = dict(latent=torch.randn(1, 4, 4), conditioning=torch.randn(1, 3, 6),
        img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4),
        timesteps=torch.tensor([1000., 500.]), sigmas=torch.tensor([1., .5, 0.]))
    names = ["base", "relation", "color", "shape"]
    supplied = {}
    for index, name in enumerate(names):
        session = adapter.session(**dict(args, conditioning=args["conditioning"] + .3 * index))
        session.continue_(1)
        supplied[name] = session.capture()
    fresh = adapter.session(**dict(args, latent=torch.randn(1, 4, 4)))
    fresh.continue_(1)
    parent = fresh.capture()
    store = LocalStore(tmp_path / "store")
    store.save(parent)
    ports = {name: {"text": cut.payload["text"], "manifest": cut.manifest()} for name, cut in supplied.items()}
    parents, acts = panel["text_branch_parents"](adapter, parent, ports, names, store)
    assert set(acts) == set(names[1:])
    for name, cut in parents.items():
        assert torch.equal(cut.payload["conditioning"], parent.payload["conditioning"])
        assert torch.equal(cut.payload["latent"], parent.payload["latent"])
        assert torch.equal(cut.payload["image"], parent.payload["image"])
        assert torch.equal(cut.payload["text"], supplied[name].payload["text"])
        assert not torch.equal(cut.payload["latent"], supplied[name].payload["latent"])
        state = panel["unpack"](panel["pack"](cut.payload), device="cpu")
        with torch.inference_mode():
            predicted = functions["functional_trajectory"](model, state)
            native = Session.from_cut(adapter, cut)
            adapter.finish(native)
        torch.testing.assert_close(predicted["latent"], native.read("latent"), rtol=1e-5, atol=1e-6)


def test_cross_time_carrier_is_refused_by_actual_parent_helper(tmp_path):
    torch.set_num_threads(1)
    model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval()
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    session = adapter.session(latent=torch.randn(1, 4, 4), conditioning=torch.randn(1, 3, 6),
        img_ids=torch.zeros(1, 4, 4), txt_ids=torch.zeros(1, 3, 4), timesteps=torch.tensor([1000., 500.]), sigmas=torch.tensor([1., .5, 0.]))
    session.continue_(1)
    parent = session.capture()
    session.continue_(5)
    later = session.capture()
    with pytest.raises(ValueError, match="contract|context"):
        panel["text_branch_parents"](adapter, parent, {"relation": {"text": later.payload["text"], "manifest": later.manifest()}}, ["base", "relation"], LocalStore(tmp_path))

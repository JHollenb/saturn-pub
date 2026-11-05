"""Public typed admission preserves full closure and native execution."""
from __future__ import annotations

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.values import clone, describe

from mdb.flux_forecast_closure import PORTS, FluxForecastClosureAdapter, closure_act


def fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(97)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=2, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False).eval().to(torch.bfloat16)
        values = {"latent": torch.randn(1, 4, 4, dtype=torch.bfloat16),
            "conditioning": torch.randn(1, 3, 6, dtype=torch.bfloat16),
            "img_ids": torch.zeros(1, 4, 4), "txt_ids": torch.zeros(1, 3, 4),
            "timesteps": torch.tensor([1000., 300.]), "sigmas": torch.tensor([1., .3, 0.])}
    base = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    candidate = FluxForecastClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    return base, candidate, base.session(**values), candidate.session(**values)


@pytest.mark.parametrize("cursor", [0, 1])
def test_unedited_native_program_and_atomic_sham_are_exact(cursor):
    base, adapter, reference, predicted = fixture()
    assert base.model_identity == adapter.model_identity
    assert base.execution != adapter.execution
    assert base.execution["adapter"] == "native-boundary-v2"
    assert predicted.capture().surface["state_schema"] == "mdb-flux2-forecast-closure-state-v1"
    reference.continue_(1 + cursor)
    predicted.continue_(1 + cursor)
    before = predicted.capture()
    assert describe(before.payload) == describe(reference.capture().payload)
    action = closure_act(before, **{key: before.payload[key] for key in PORTS}, name="exact-sham")
    receipt = predicted.apply(action)
    assert set(receipt.to_dict()["act"]["writes"]) == set(PORTS)
    assert describe(predicted.capture().payload) == describe(before.payload)
    while predicted.capture().payload["step"] < 2:
        predicted.continue_()
        reference.continue_()
        assert describe(predicted.capture().payload) == describe(reference.capture().payload)


def test_changed_complete_context_is_actually_consumed_and_other_state_preserved():
    _, adapter, _, session = fixture()
    session.continue_()
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    values["context"]["img_mod"] += torch.tensor(.25, dtype=torch.bfloat16)
    values["latent"] += torch.tensor(.25, dtype=torch.bfloat16)
    action = closure_act(before, **values)
    session.apply(action)
    installed = session.capture()
    assert describe(installed.payload) == describe({**before.payload, **values})
    assert installed.execution_point == before.execution_point
    plain = Session.from_cut(adapter, before)
    plain.continue_()
    session.continue_()
    assert not torch.equal(session.read("image"), plain.read("image"))
    assert describe(session.capture().payload["context"]) == describe(values["context"])


@pytest.mark.parametrize("slot", ["latent", "context", "image", "text", "step", "conditioning"])
def test_partial_or_clock_writes_refuse_without_state_change(slot):
    _, _, _, session = fixture()
    session.continue_()
    before = session.capture()
    with pytest.raises(ValueError):
        session.apply(Act.replace(slot, before.payload[slot]))
    assert session.capture().fingerprint == before.fingerprint


@pytest.mark.parametrize("bad", ["missing-rope", "rope-type", "mod-shape", "mod-dtype", "nonfinite"])
def test_bad_context_refuses_before_admission(bad):
    _, _, _, session = fixture()
    session.continue_()
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    ctx = values["context"]
    if bad == "missing-rope":
        del ctx["rope"]
    elif bad == "rope-type":
        ctx["rope"] = list(ctx["rope"])
    elif bad == "mod-shape":
        ctx["img_mod"] = ctx["img_mod"][:, :-1]
    elif bad == "mod-dtype":
        ctx["img_mod"] = ctx["img_mod"].float()
    else:
        ctx["temb"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="payload"):
        closure_act(before, **values)
    assert session.capture().fingerprint == before.fingerprint


def test_clock_and_source_preconditions_cannot_be_rebound():
    _, _, _, session = fixture()
    initial = session.capture()
    with pytest.raises(ValueError, match="joint0/joint1"):
        closure_act(initial, **{key: initial.payload[key] for key in PORTS})
    session.continue_()
    before = session.capture()
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    values = {key: clone(before.payload[key]) for key in PORTS}
    values["image"] += torch.tensor(.5, dtype=torch.bfloat16)
    session.apply(closure_act(before, **values))
    changed = session.capture()
    with pytest.raises(ValueError, match="precondition changed"):
        session.apply(action)
    assert session.capture().fingerprint == changed.fingerprint
    session.continue_(2)
    later = session.capture()
    with pytest.raises(ValueError, match="joint0/joint1"):
        closure_act(later, **{key: later.payload[key] for key in PORTS})


@pytest.mark.parametrize("field", ["cursor", "conditioning", "sigmas", "img_ids"])
def test_carried_sources_are_bound_even_when_all_dynamic_tensors_match(field):
    _, adapter, _, session = fixture()
    session.continue_()
    before = session.capture()
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    altered = before.payload
    if field == "cursor":
        altered[field] = 1
    else:
        altered[field] = altered[field] + 1
    other = Session(adapter, altered)
    unchanged = other.capture()
    with pytest.raises(ValueError, match="precondition changed"):
        other.apply(action)
    assert other.capture().fingerprint == unchanged.fingerprint

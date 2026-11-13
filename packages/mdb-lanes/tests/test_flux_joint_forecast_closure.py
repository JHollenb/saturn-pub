"""Tiny native joint1/2 admission, unchanged native advancement and v1 policy."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.values import clone, describe, digest

import mdb_lanes.flux_forecast_closure as previous
import mdb_lanes.flux_joint_forecast_closure as extension
from mdb_lanes.flux_joint_forecast_closure import (
    ADAPTER_ID,
    CURSORS,
    PORTS,
    SOURCE_PORTS,
    FluxJointForecastClosureAdapter,
    closure_act,
)


def fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(101)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=3, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False).eval().to(torch.bfloat16)
        values = {"latent": torch.randn(1, 4, 4, dtype=torch.bfloat16),
            "conditioning": torch.randn(1, 3, 6, dtype=torch.bfloat16),
            "img_ids": torch.zeros(1, 4, 4), "txt_ids": torch.zeros(1, 3, 4),
            "timesteps": torch.tensor([1000., 300.]), "sigmas": torch.tensor([1., .3, 0.])}
    base = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    adapter = FluxJointForecastClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    return base, adapter, base.session(**values), adapter.session(**values)


@pytest.mark.parametrize("cursor", CURSORS)
def test_joint_sham_and_full_native_continuation_are_exact(cursor):
    base, adapter, reference, session = fixture()
    assert FluxJointForecastClosureAdapter.advance is Flux2KleinAdapter.advance
    assert FluxJointForecastClosureAdapter.validate is previous.FluxForecastClosureAdapter.validate
    assert base.model_identity == adapter.model_identity and base.execution != adapter.execution
    assert adapter.execution["adapter"] == ADAPTER_ID
    assert adapter.execution["forecast_closure"]["cursors"] == [1, 2]
    reference.continue_(1 + cursor)
    session.continue_(1 + cursor)
    before = session.capture()
    assert describe(before.payload) == describe(reference.capture().payload)
    assert before.surface["state_schema"] == extension.STATE_SCHEMA
    assert set(adapter.addresses(before.payload)) == set((*PORTS, *SOURCE_PORTS))
    assert {slot["name"] for slot in before.surface["slots"] if slot["writable"]} == set(PORTS)
    action = closure_act(before, **{key: before.payload[key] for key in PORTS}, name="joint-exact-sham")
    assert set(action.preconditions) == set((*PORTS, *SOURCE_PORTS))
    assert action.preconditions["noise_prediction"] == digest(describe(None))
    receipt = session.apply(action)
    assert receipt.to_dict()["act"]["operation"]["id"] == extension.OPERATION_ID
    assert describe(session.capture().payload) == describe(before.payload)
    while session.capture().payload["step"] < 2:
        session.continue_()
        reference.continue_()
        assert describe(session.capture().payload) == describe(reference.capture().payload)


@pytest.mark.parametrize("cursor", CURSORS)
def test_non_native_context_is_consumed_by_unmodified_native_advance(cursor):
    base, adapter, _, session = fixture()
    session.continue_(1 + cursor)
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    values["context"]["img_mod"] += torch.tensor(.5, dtype=torch.bfloat16)
    session.apply(closure_act(before, **values))
    installed = session.capture()
    assert installed.execution_point == before.execution_point
    assert describe(installed.payload) == describe({**before.payload, **values})
    direct_native = base.advance(installed.payload)
    plain = Session.from_cut(adapter, before)
    session.continue_()
    plain.continue_()
    assert describe(session.capture().payload) == describe(direct_native)
    assert not torch.equal(session.read("image"), plain.read("image"))
    assert describe(session.capture().payload["context"]) == describe(values["context"])


@pytest.mark.parametrize("slot", (*PORTS, "step", "phase", "cursor", "conditioning", "sigmas"))
@pytest.mark.parametrize("cursor", CURSORS)
def test_partial_or_source_clock_write_refused_without_mutation(cursor, slot):
    _, _, _, session = fixture()
    session.continue_(1 + cursor)
    before = session.capture()
    with pytest.raises(ValueError):
        session.apply(Act.replace(slot, before.payload[slot]))
    assert session.capture().fingerprint == before.fingerprint


@pytest.mark.parametrize("field", ("step", "phase", "cursor", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"))
def test_same_dynamic_payload_cannot_rebind_any_carried_source_or_clock(field):
    _, adapter, _, session = fixture()
    session.continue_(2)
    before = session.capture()
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    altered = before.payload
    if field == "step":
        altered[field] = 1
    elif field == "phase":
        altered[field] = "single"
    elif field == "cursor":
        altered[field] = 2
    else:
        altered[field] = altered[field] + 1
    other = Session(adapter, altered)
    unchanged = other.capture()
    # A different phase also removes the opt-in readable closure surface.
    with pytest.raises(ValueError, match="precondition changed|unsupported address"):
        other.apply(action)
    assert other.capture().fingerprint == unchanged.fingerprint


def test_v1_source_policy_and_cross_adapter_cut_isolation_are_unchanged():
    _, adapter, _, session = fixture()
    assert hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() == extension.DEPENDENCY_SHA256
    old = previous.FluxForecastClosureAdapter(adapter.model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    initial = session.capture().payload
    old_session = old.session(**{key: initial[key] for key in
        ("latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")})
    old_session.continue_()
    previous.closure_act(old_session.capture(), **{key: old_session.capture().payload[key] for key in PORTS})
    session.continue_()
    joint0 = session.capture()
    with pytest.raises(ValueError, match="joint1/joint2"):
        closure_act(joint0, **{key: joint0.payload[key] for key in PORTS})
    with pytest.raises(ValueError, match="invalidate"):
        adapter.write(joint0.payload, {key: joint0.payload[key] for key in PORTS}, ())
    old_session.continue_(2)
    old_joint2 = old_session.capture()
    with pytest.raises(ValueError, match="joint0/joint1"):
        previous.closure_act(old_joint2, **{key: old_joint2.payload[key] for key in PORTS})
    with pytest.raises(ValueError, match="joint1/joint2"):
        closure_act(old_joint2, **{key: old_joint2.payload[key] for key in PORTS})


@pytest.mark.parametrize("bad", ("missing-rope", "list-rope", "dtype", "shape", "nan"))
def test_complete_context_layout_and_finite_contract_is_reused(bad):
    _, _, _, session = fixture()
    session.continue_(3)
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    context = values["context"]
    if bad == "missing-rope":
        del context["rope"]
    elif bad == "list-rope":
        context["rope"] = list(context["rope"])
    elif bad == "dtype":
        context["rope"] = tuple(value.to(torch.bfloat16) for value in context["rope"])
    elif bad == "shape":
        context["img_mod"] = context["img_mod"][:, :-1]
    else:
        context["temb"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="payload"):
        closure_act(before, **values)
    assert session.capture().fingerprint == before.fingerprint


def test_layout_adaptation_preserves_values_without_silent_casts():
    _, _, _, session = fixture()
    session.continue_(3)
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    values["image"] = values["image"].transpose(1, 2).contiguous().transpose(1, 2)
    assert values["image"].stride() != before.payload["image"].stride()
    action = closure_act(before, **values)
    session.apply(action)
    installed = session.capture()
    assert describe(installed.payload) == describe(before.payload)
    assert installed.payload["image"].stride() == before.payload["image"].stride()


def test_changed_dependency_source_is_refused_before_adapter_admission(monkeypatch):
    _, _, _, session = fixture()
    session.continue_(2)
    before = session.capture()
    monkeypatch.setattr(extension, "DEPENDENCY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="dependency source differs"):
        closure_act(before, **{key: before.payload[key] for key in PORTS})
    assert session.capture().fingerprint == before.fingerprint

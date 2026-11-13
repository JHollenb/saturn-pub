"""Actual tiny five-joint native execution and atomic tail admission."""
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
import mdb_lanes.flux_joint_forecast_closure as joint
import mdb_lanes.flux_joint_tail_closure as extension
from mdb_lanes.flux_joint_tail_closure import (
    PORTS,
    SOURCE_PORTS,
    FluxJointTailClosureAdapter,
    closure_act,
)

BOUNDARIES = (("joint", 2), ("joint", 3), ("joint", 4), ("single", 0))


def fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(103)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=5, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False).eval().to(torch.bfloat16)
        values = {"latent": torch.randn(1, 4, 4, dtype=torch.bfloat16),
            "conditioning": torch.randn(1, 3, 6, dtype=torch.bfloat16),
            "img_ids": torch.zeros(1, 4, 4), "txt_ids": torch.zeros(1, 3, 4),
            "timesteps": torch.tensor([1000., 300.]), "sigmas": torch.tensor([1., .3, 0.])}
    base = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    adapter = FluxJointTailClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    return base, adapter, base.session(**values), adapter.session(**values)


def seek(session, phase, cursor):
    while (session.capture().payload["phase"], session.capture().payload["cursor"]) != (phase, cursor):
        session.continue_()


@pytest.mark.parametrize("phase,cursor", BOUNDARIES)
def test_atomic_sham_has_thirteen_sources_and_native_complete_continuation(phase, cursor):
    base, adapter, reference, session = fixture()
    assert FluxJointTailClosureAdapter.advance is Flux2KleinAdapter.advance
    assert FluxJointTailClosureAdapter.validate is previous.FluxForecastClosureAdapter.validate
    seek(reference, phase, cursor)
    seek(session, phase, cursor)
    before = session.capture()
    assert describe(before.payload) == describe(reference.capture().payload)
    assert adapter.execution["adapter"] == extension.ADAPTER_ID
    assert before.surface["state_schema"] == extension.STATE_SCHEMA
    assert set(adapter.addresses(before.payload)) == set((*PORTS, *SOURCE_PORTS))
    assert {slot["name"] for slot in before.surface["slots"] if slot["writable"]} == set(PORTS)
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    assert len(action.preconditions) == 13
    assert action.preconditions == {key: digest(describe(before.payload[key])) for key in (*PORTS, *SOURCE_PORTS)}
    receipt = session.apply(action)
    assert receipt.to_dict()["act"]["operation"]["id"] == extension.OPERATION_ID
    assert describe(session.capture().payload) == describe(before.payload)
    assert session.capture().execution_point == before.execution_point
    while session.capture().payload["step"] < 2:
        session.continue_()
        reference.continue_()
        assert describe(session.capture().payload) == describe(reference.capture().payload)
    assert base.model_identity == adapter.model_identity


@pytest.mark.parametrize("phase,cursor", BOUNDARIES)
def test_changed_context_is_consumed_by_unmodified_native_advance(phase, cursor):
    base, adapter, _, session = fixture()
    seek(session, phase, cursor)
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    key = "img_mod" if phase == "joint" else "single_mod"
    values["context"][key] += torch.tensor(.5, dtype=torch.bfloat16)
    session.apply(closure_act(before, **values))
    installed = session.capture()
    assert describe(installed.payload) == describe({**before.payload, **values})
    plain = Session.from_cut(adapter, before)
    expected = base.advance(installed.payload)
    session.continue_()
    plain.continue_()
    assert describe(session.capture().payload) == describe(expected)
    assert not torch.equal(session.read("image"), plain.read("image"))
    assert describe(session.capture().payload["context"]) == describe(values["context"])


@pytest.mark.parametrize("phase,cursor", BOUNDARIES)
@pytest.mark.parametrize("slot", (*PORTS, "step", "phase", "cursor", "conditioning", "sigmas"))
def test_partial_or_source_clock_writes_are_refused_without_mutation(phase, cursor, slot):
    _, _, _, session = fixture()
    seek(session, phase, cursor)
    before = session.capture()
    with pytest.raises(ValueError):
        session.apply(Act.replace(slot, before.payload[slot]))
    assert session.capture().fingerprint == before.fingerprint


@pytest.mark.parametrize("field", ("step", "phase", "cursor", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"))
def test_atomic_action_cannot_be_rebound_to_another_source_or_clock(field):
    _, adapter, _, session = fixture()
    seek(session, "joint", 2)
    before = session.capture()
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    altered = before.payload
    if field == "step":
        altered[field] = 1
    elif field == "phase":
        altered[field] = "single"
    elif field == "cursor":
        altered[field] = 4
    else:
        altered[field] = altered[field] + 1
    other = Session(adapter, altered)
    unchanged = other.capture()
    with pytest.raises(ValueError, match="precondition changed|unsupported address"):
        other.apply(action)
    assert other.capture().fingerprint == unchanged.fingerprint


@pytest.mark.parametrize("bad", ("missing-rope", "list-rope", "dtype", "shape", "nan"))
def test_context_geometry_finite_and_tuple_contract_is_reused(bad):
    _, _, _, session = fixture()
    seek(session, "single", 0)
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


def test_layout_copy_preserves_values_and_direct_invalidations_are_refused():
    _, adapter, _, session = fixture()
    seek(session, "single", 0)
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    values["image"] = values["image"].transpose(1, 2).contiguous().transpose(1, 2)
    assert values["image"].stride() != before.payload["image"].stride()
    session.apply(closure_act(before, **values))
    assert describe(session.capture().payload) == describe(before.payload)
    with pytest.raises(ValueError, match="four atomic"):
        adapter.write(before.payload, {key: before.payload[key] for key in PORTS}, ("context",))


def test_previous_sources_admission_policies_and_cross_adapter_identity_unchanged():
    _, adapter, _, session = fixture()
    assert hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest() == extension.DEPENDENCY_SHA256
    assert hashlib.sha256(Path(joint.__file__).read_bytes()).hexdigest() == extension.JOINT_DEPENDENCY_SHA256
    for old_class, old_builder, accepted, rejected in (
        (previous.FluxForecastClosureAdapter, previous.closure_act, ("joint", 0), ("joint", 2)),
        (joint.FluxJointForecastClosureAdapter, joint.closure_act, ("joint", 2), ("joint", 3)),
    ):
        old = old_class(adapter.model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        initial = session.capture().payload
        old_session = old.session(**{key: initial[key] for key in
            ("latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")})
        seek(old_session, *accepted)
        cut = old_session.capture()
        old_builder(cut, **{key: cut.payload[key] for key in PORTS})
        with pytest.raises(ValueError, match="joint-tail"):
            closure_act(cut, **{key: cut.payload[key] for key in PORTS})
        seek(old_session, *rejected)
        cut = old_session.capture()
        with pytest.raises(ValueError):
            old_builder(cut, **{key: cut.payload[key] for key in PORTS})
    for phase, cursor in (("joint", 0), ("joint", 1), ("single", 1)):
        _, _, _, tail = fixture()
        seek(tail, phase, cursor)
        cut = tail.capture()
        with pytest.raises(ValueError, match="joint-tail"):
            closure_act(cut, **{key: cut.payload[key] for key in PORTS})


@pytest.mark.parametrize("dependency", ("DEPENDENCY_SHA256", "JOINT_DEPENDENCY_SHA256"))
def test_dependency_source_drift_is_refused(dependency, monkeypatch):
    _, _, _, session = fixture()
    seek(session, "joint", 2)
    before = session.capture()
    monkeypatch.setattr(extension, dependency, "0" * 64)
    with pytest.raises(ValueError, match="dependency source differs"):
        closure_act(before, **{key: before.payload[key] for key in PORTS})
    assert session.capture().fingerprint == before.fingerprint

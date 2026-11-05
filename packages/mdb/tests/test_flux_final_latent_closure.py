"""Tiny native four-step final-latent admission, reset, and rollback mechanics."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Act, Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.values import clone, describe, digest

import mdb.flux_final_latent_closure as extension
import mdb.flux_forecast_closure as previous
import mdb.flux_joint_forecast_closure as joint
import mdb.flux_joint_tail_closure as tail
from mdb.flux_final_latent_closure import (
    PORTS,
    SOURCE_PORTS,
    FluxFinalLatentClosureAdapter,
    closure_act,
)


def fixture(steps=4):
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(107)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=5, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False).eval().to(torch.bfloat16)
        values = {"latent": torch.randn(1, 4, 4, dtype=torch.bfloat16),
            "conditioning": torch.randn(1, 3, 6, dtype=torch.bfloat16),
            "img_ids": torch.zeros(1, 4, 4), "txt_ids": torch.zeros(1, 3, 4),
            "timesteps": torch.tensor([1000., 700., 400., 100.]),
            "sigmas": torch.tensor([1., .7, .4, .1, 0.])}
        if steps == 2:
            values["timesteps"] = torch.tensor([1000., 100.])
            values["sigmas"] = torch.tensor([1., .1, 0.])
    base = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    adapter = FluxFinalLatentClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    return base, adapter, base.session(**values), adapter.session(**values)


def seek(session, terminal=False):
    steps = len(session.capture().payload["timesteps"])
    target = (steps, "project", 0) if terminal else (steps-1, "single", 0)
    while tuple(session.capture().payload[key] for key in ("step", "phase", "cursor")) != target:
        session.continue_()


@pytest.mark.parametrize("terminal", (False, True))
def test_thirteen_source_atomic_sham_is_exact_and_native_advance_unchanged(terminal):
    base, adapter, reference, session = fixture()
    assert FluxFinalLatentClosureAdapter.advance is Flux2KleinAdapter.advance
    assert FluxFinalLatentClosureAdapter.validate is previous.FluxForecastClosureAdapter.validate
    seek(reference, terminal)
    seek(session, terminal)
    before = session.capture()
    assert describe(before.payload) == describe(reference.capture().payload)
    assert before.surface["state_schema"] == extension.STATE_SCHEMA
    assert set(adapter.addresses(before.payload)) == set((*PORTS, *SOURCE_PORTS))
    assert {slot["name"] for slot in before.surface["slots"] if slot["writable"]} == set(PORTS)
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    assert len(action.preconditions) == 13
    assert action.preconditions == {key: digest(describe(before.payload[key])) for key in (*PORTS, *SOURCE_PORTS)}
    receipt = session.apply(action).to_dict()
    assert receipt["act"]["operation"]["id"] == extension.OPERATION_ID
    assert describe(session.capture().payload) == describe(before.payload)
    assert session.capture().execution_point == before.execution_point
    if terminal:
        assert all(before.payload[key] is None for key in ("image", "text", "context", "noise_prediction"))
        after = session.capture()
        with pytest.raises(ValueError, match="already complete"):
            session.continue_()
        assert session.capture().fingerprint == after.fingerprint
    else:
        while session.capture().payload["step"] < 4:
            session.continue_()
            reference.continue_()
            assert describe(session.capture().payload) == describe(reference.capture().payload)
    assert base.model_identity == adapter.model_identity


def test_non_native_complete_single_context_is_consumed_then_cleared_by_native_scheduler():
    base, adapter, _, session = fixture()
    seek(session)
    before = session.capture()
    values = {key: clone(before.payload[key]) for key in PORTS}
    values["context"]["single_mod"] += torch.tensor(.5, dtype=torch.bfloat16)
    session.apply(closure_act(before, **values))
    installed = session.capture()
    plain = Session.from_cut(adapter, before)
    session.continue_()
    plain.continue_()
    assert describe(session.capture().payload) == describe(base.advance(installed.payload))
    assert not torch.equal(session.read("image"), plain.read("image"))
    seek(session, terminal=True)
    assert all(session.capture().payload[key] is None for key in ("image", "text", "context", "noise_prediction"))


def test_predicted_terminal_latent_only_in_four_port_closure_and_exact_rollback():
    _, _, _, session = fixture()
    seek(session, terminal=True)
    before = session.capture()
    predicted = before.payload["latent"] + torch.tensor(.25, dtype=torch.bfloat16)
    session.apply(closure_act(before, latent=predicted, image=None, text=None, context=None))
    after = session.capture()
    assert torch.equal(after.payload["latent"], predicted)
    assert after.execution_point == before.execution_point
    assert describe(after.payload) == describe({**before.payload, "latent": predicted})
    session.restore(before)
    assert describe(session.capture().payload) == describe(before.payload)
    assert session.capture().execution_point == before.execution_point


@pytest.mark.parametrize("terminal", (False, True))
@pytest.mark.parametrize("slot", (*PORTS, "step", "phase", "cursor", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas", "noise_prediction"))
def test_every_partial_port_or_source_clock_write_is_refused_without_mutation(terminal, slot):
    _, _, _, session = fixture()
    seek(session, terminal)
    before = session.capture()
    with pytest.raises(ValueError):
        session.apply(Act.replace(slot, before.payload[slot]))
    assert session.capture().fingerprint == before.fingerprint


@pytest.mark.parametrize("slot", ("context", "image", "text"))
def test_terminal_reset_cannot_be_repopulated(slot):
    _, _, _, session = fixture()
    seek(session)
    live_value = clone(session.capture().payload[slot])
    seek(session, terminal=True)
    before = session.capture()
    values = {key: before.payload[key] for key in PORTS}
    values[slot] = live_value
    with pytest.raises(ValueError, match="remain None"):
        closure_act(before, **values)
    assert session.capture().fingerprint == before.fingerprint


@pytest.mark.parametrize("field", ("step", "phase", "cursor", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"))
def test_atomic_terminal_action_cannot_rebind_sources_or_clock(field):
    _, adapter, _, session = fixture()
    seek(session, terminal=True)
    before = session.capture()
    action = closure_act(before, **{key: before.payload[key] for key in PORTS})
    altered = before.payload
    if field in {"phase", "cursor"}:
        altered[field] = "single" if field == "phase" else 1
        with pytest.raises(ValueError):
            Session(adapter, altered)
        return
    if field == "step":
        altered[field] = 3
    else:
        altered[field] = altered[field] + 1
    other = Session(adapter, altered)
    unchanged = other.capture()
    with pytest.raises(ValueError, match="precondition changed|unsupported address"):
        other.apply(action)
    assert other.capture().fingerprint == unchanged.fingerprint


@pytest.mark.parametrize("bad", ("dtype", "shape", "nan"))
def test_terminal_latent_preserves_dtype_shape_and_finite_contract(bad):
    _, _, _, session = fixture()
    seek(session, terminal=True)
    before = session.capture()
    latent = clone(before.payload["latent"])
    if bad == "dtype":
        latent = latent.float()
    elif bad == "shape":
        latent = latent[:, :-1]
    else:
        latent[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="payload"):
        closure_act(before, latent=latent, image=None, text=None, context=None)
    assert session.capture().fingerprint == before.fingerprint


def test_valid_native_other_step_single0_is_not_an_admitted_final_boundary():
    _, adapter, _, session = fixture()
    session.continue_(6)
    cut = session.capture()
    assert cut.payload["step"] == 0 and cut.payload["phase"] == "single" and cut.payload["cursor"] == 0
    with pytest.raises(ValueError, match="final-step/single0"):
        closure_act(cut, **{key: cut.payload[key] for key in PORTS})
    with pytest.raises(ValueError):
        adapter.write(cut.payload, {key: cut.payload[key] for key in PORTS}, ())


def test_prior_adapters_and_sources_remain_distinct_and_terminal_read_only():
    _, adapter, _, session = fixture()
    for module, expected, cls in ((previous, extension.DEPENDENCY_SHA256, previous.FluxForecastClosureAdapter),
            (joint, extension.JOINT_DEPENDENCY_SHA256, joint.FluxJointForecastClosureAdapter),
            (tail, extension.TAIL_DEPENDENCY_SHA256, tail.FluxJointTailClosureAdapter)):
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
        old = cls(adapter.model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        initial = session.capture().payload
        old_session = old.session(**{key: initial[key] for key in
            ("latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")})
        seek(old_session, terminal=True)
        cut = old_session.capture()
        with pytest.raises(ValueError, match="final-step/single0"):
            closure_act(cut, **{key: cut.payload[key] for key in PORTS})
        with pytest.raises(ValueError, match="read-only|invalidate"):
            old.write(cut.payload, {key: cut.payload[key] for key in PORTS}, ())


@pytest.mark.parametrize("dependency", ("DEPENDENCY_SHA256", "JOINT_DEPENDENCY_SHA256", "TAIL_DEPENDENCY_SHA256"))
def test_dependency_drift_refused(dependency, monkeypatch):
    _, _, _, session = fixture()
    seek(session, terminal=True)
    before = session.capture()
    monkeypatch.setattr(extension, dependency, "0" * 64)
    with pytest.raises(ValueError, match="dependency source differs"):
        closure_act(before, **{key: before.payload[key] for key in PORTS})


@pytest.mark.parametrize("terminal", (False, True))
def test_admitted_clocks_are_schedule_relative_for_short_native_fixture(terminal):
    _, _, _, session = fixture(steps=2)
    seek(session, terminal)
    before = session.capture()
    assert before.payload["step"] == (2 if terminal else 1)
    session.apply(closure_act(before, **{key: before.payload[key] for key in PORTS}))
    assert describe(session.capture().payload) == describe(before.payload)

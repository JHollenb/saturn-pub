"""Tiny public-native prefix construction; no learned model or future oracle."""
from dataclasses import replace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub import Session, StateCut
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.values import describe, digest

import mdb.flux_derived_terminal as helper
from mdb.flux_final_latent_closure import FluxFinalLatentClosureAdapter


def fixture():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(109)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=1, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False).eval().to(torch.bfloat16)
        values = {"latent": torch.randn(1, 4, 4, dtype=torch.bfloat16),
            "conditioning": torch.randn(1, 3, 6, dtype=torch.bfloat16),
            "img_ids": torch.zeros(1, 4, 4), "txt_ids": torch.zeros(1, 3, 4),
            "timesteps": torch.tensor([1000., 700., 400., 100.]),
            "sigmas": torch.tensor([1., .7, .4, .1, 0.])}
    adapter = FluxFinalLatentClosureAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    session = adapter.session(**values)
    session.continue_(10)
    prefix = session.capture()
    assert tuple(prefix.payload[k] for k in ("step", "phase", "cursor")) == (2, "project", 0)
    prediction = prefix.payload["latent"] + torch.tensor(.25, dtype=torch.bfloat16)
    return adapter, session, prefix, prediction


def test_prefix_to_terminal_then_one_public_act_has_no_native_future_execution():
    adapter, session, prefix, prediction = fixture()
    observed = describe(prefix.payload)
    entries = []
    hooks = [module.register_forward_pre_hook(lambda *_: entries.append("entry")) for module in adapter.model.modules()]
    try:
        result = helper.derive_terminal(adapter, prefix, predicted_latent=prediction)
        assert not entries
        assert result.scaffold.parent == prefix.fingerprint
        state = result.scaffold.payload
        assert torch.equal(state["latent"], prefix.payload["latent"])
        assert state["step"] == len(state["timesteps"]) == 4
        assert state["phase"] == "project" and state["cursor"] == 0
        assert result.scaffold.execution_point["index"] is None
        assert all(state[k] is None for k in ("context", "image", "text", "noise_prediction"))
        assert all(describe(state[k]) == describe(prefix.payload[k]) for k in
            ("conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"))
        target = Session.from_cut(adapter, result.scaffold)
        applied = target.apply(result.action)
        assert applied.to_dict()["operation"] == "apply"
        assert torch.equal(target.read("latent"), prediction)
        assert describe(target.capture().payload) == describe({**state, "latent": prediction})
        assert not entries
        record = result.derivation
        assert record["native_steps"] == record["acts_applied"] == record["execution_receipt_emissions"] == 0
        assert record["fingerprint"] == digest({k: v for k, v in record.items() if k != "fingerprint"})
        assert record["source_preconditions"] == {k: digest(describe(v)) for k, v in prefix.payload.items()}
        assert len(record["source_preconditions"]) == 13
        assert len(result.action.preconditions) == 13
        assert record["predicted_terminal_payload"] == describe(target.capture().payload)
        assert session.capture().fingerprint == prefix.fingerprint
        assert describe(prefix.payload) == observed
    finally:
        for hook in hooks:
            hook.remove()


def test_later_native_terminal_has_identical_nonlatent_contract_not_oracle_latent():
    adapter, session, prefix, prediction = fixture()
    result = helper.derive_terminal(adapter, prefix, predicted_latent=prediction)
    while session.capture().payload["step"] < 4:
        session.continue_()
    native = session.capture()
    shell = result.scaffold
    assert native.model_identity == shell.model_identity
    assert dict(native.execution) == dict(shell.execution)
    assert dict(native.execution_point) == dict(shell.execution_point)
    assert all(describe(native.payload[k]) == describe(shell.payload[k]) for k in shell.payload if k != "latent")
    mismatched = native.payload
    mismatched["conditioning"] += 1
    wrong = Session(adapter, mismatched).capture()
    assert describe(wrong.payload["conditioning"]) != describe(shell.payload["conditioning"])
    # No native terminal cut or latent was an input to the derivation.
    assert result.derivation["placeholder_latent_source"]["cut"] == prefix.fingerprint


def test_mutating_supplied_tensor_returned_metadata_or_readback_cannot_change_derivation():
    adapter, _, prefix, prediction = fixture()
    expected = prediction.clone()
    result = helper.derive_terminal(adapter, prefix, predicted_latent=prediction)
    fingerprint = result.scaffold.fingerprint
    record_fingerprint = result.derivation["fingerprint"]
    prediction.zero_()
    result.scaffold.payload["latent"].zero_()
    result.derivation["source_preconditions"].clear()
    result.scaffold.verify()
    assert result.scaffold.fingerprint == fingerprint
    assert result.derivation["fingerprint"] == record_fingerprint
    target = Session.from_cut(adapter, result.scaffold)
    target.apply(result.action)
    assert torch.equal(target.read("latent"), expected)


@pytest.mark.parametrize("field", ("execution_point", "boundary", "surface", "model_identity", "execution"))
def test_source_metadata_cannot_lie_about_native_boundary_or_identity(field):
    adapter, _, prefix, prediction = fixture()
    if field == "execution_point":
        value = {**dict(prefix.execution_point), "logical_step": 3}
    elif field == "boundary":
        value = "wrong-source-boundary"
    elif field == "surface":
        value = {**dict(prefix.surface), "state_schema": "wrong-source-schema"}
    elif field == "model_identity":
        value = "wrong-model"
    else:
        value = {**dict(prefix.execution), "adapter": "wrong-adapter"}
    wrong = replace(prefix, **{field: value})
    with pytest.raises(ValueError, match="identity differs|metadata does not match"):
        helper.derive_terminal(adapter, wrong, predicted_latent=prediction)


@pytest.mark.parametrize("bad", ("live-transformer", "completed", "uncleared-image", "wrong-clock-type", "extra-slot"))
def test_source_must_be_complete_valid_live_project_with_reset_slots(bad):
    adapter, session, prefix, prediction = fixture()
    state = prefix.payload
    if bad == "live-transformer":
        session.continue_()
        wrong = session.capture()
    else:
        if bad == "completed":
            state["step"] = len(state["timesteps"])
        elif bad == "uncleared-image":
            state["image"] = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
        elif bad == "wrong-clock-type":
            state["step"] = 2.0
        else:
            state["extra"] = 1
        wrong = StateCut(prefix.model_identity, prefix.execution, prefix.boundary, prefix.parent,
            state, prefix.execution_point, prefix.surface)
    with pytest.raises(ValueError):
        helper.derive_terminal(adapter, wrong, predicted_latent=prediction)


@pytest.mark.parametrize("bad", ("dtype", "shape", "nan", "device"))
def test_prediction_must_match_qualified_prefix_latent_geometry_and_type(bad):
    adapter, _, prefix, prediction = fixture()
    if bad == "dtype":
        prediction = prediction.float()
    elif bad == "shape":
        prediction = prediction[:, :-1]
    elif bad == "nan":
        prediction[0, 0, 0] = float("nan")
    else:
        prediction = torch.empty(prediction.shape, dtype=prediction.dtype, device="meta")
    with pytest.raises(ValueError, match="payload"):
        helper.derive_terminal(adapter, prefix, predicted_latent=prediction)


def test_prediction_layout_is_adapted_without_value_or_dtype_casts():
    adapter, _, prefix, prediction = fixture()
    prediction = prediction.transpose(1, 2).contiguous().transpose(1, 2)
    assert prediction.stride() != prefix.payload["latent"].stride()
    result = helper.derive_terminal(adapter, prefix, predicted_latent=prediction)
    target = Session.from_cut(adapter, result.scaffold)
    target.apply(result.action)
    assert torch.equal(target.read("latent"), prediction)
    assert target.read("latent").stride() == prefix.payload["latent"].stride()


def test_action_cannot_move_to_another_prefix_source_conditioning():
    adapter, _, prefix, prediction = fixture()
    result = helper.derive_terminal(adapter, prefix, predicted_latent=prediction)
    other_state = prefix.payload
    other_state["conditioning"] += 1
    other_prefix = Session(adapter, other_state, parent=prefix.parent).capture()
    other = helper.derive_terminal(adapter, other_prefix, predicted_latent=prediction)
    target = Session.from_cut(adapter, other.scaffold)
    before = target.capture()
    with pytest.raises(ValueError, match="precondition changed"):
        target.apply(result.action)
    assert target.capture().fingerprint == before.fingerprint


def test_wrong_adapter_cannot_claim_the_opt_in_terminal_contract():
    adapter, _, prefix, prediction = fixture()
    plain = Flux2KleinAdapter(adapter.model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    with pytest.raises(ValueError, match="qualified final-latent"):
        helper.derive_terminal(plain, prefix, predicted_latent=prediction)


@pytest.mark.parametrize("pin", ("ADAPTER_SOURCE_SHA256", "CLOSURE_SOURCE_SHA256"))
def test_source_dependency_drift_is_refused(pin, monkeypatch):
    adapter, _, prefix, prediction = fixture()
    monkeypatch.setattr(helper, pin, "0" * 64)
    with pytest.raises(ValueError, match="source differs"):
        helper.derive_terminal(adapter, prefix, predicted_latent=prediction)

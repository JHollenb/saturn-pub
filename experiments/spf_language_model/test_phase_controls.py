"""Tiny reference/native/snapshot mechanics for the bounded frequency control."""

import copy

import numpy as np
import pytest
import torch
from models import build_spf_model
from phase_controls import apply_transition_control


def test_reference_rotation_removed_without_amplitude_or_other_plane_changes():
    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    ids = torch.tensor([[94, 65, 97, 63]])
    x = model.embedding(ids)
    before = model.layers[0].parameters_cartesian_reference(x)
    architecture = copy.deepcopy(model.architecture)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    receipt = apply_transition_control(model)
    after = model.layers[0].parameters_cartesian_reference(x)
    assert receipt["actual_transition_frequency"] == 0
    assert receipt["changed_buffers"][0]["changed"]
    assert torch.count_nonzero(after["a"][..., 1]) == 0
    magnitude = torch.exp2(after["decay_log2"])
    torch.testing.assert_close(after["a"][..., 0], magnitude.expand_as(after["a"][..., 0]))
    assert bool((after["a"][..., 0] < 1).all())
    for key in ("b", "c", "v", "addresses", "features", "decay_log2"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert model.architecture == architecture
    for name, value in model.state_dict().items():
        if name != "layers.0.transition_frequencies":
            assert torch.equal(state[name], value)
    assert torch.equal(model.layers[1].transition_frequencies, torch.ones(2))


@torch.no_grad()
def test_canonical_native_zero_frequency_preserves_gains_and_decay():
    pytest.importorskip("spf.tensor.sequence_mixer")
    from spf.tensor.core import SPF32_WORD_ZERO, unpack_L, unpack_theta

    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    layer = model.layers[0]
    x = model.embedding(torch.tensor([[94, 65, 97, 63]]))
    detail = {"addresses": layer.control_addresses(x), "v": layer.v_values(x)}
    before = layer.export_native_inputs(detail)
    apply_transition_control(model)
    after = layer.export_native_inputs(detail)
    assert after["program"].transition_frequencies == (0, 0)
    assert np.count_nonzero(unpack_theta(after["parameters"].forward.a)) == 0
    assert np.all(after["parameters"].forward.a != SPF32_WORD_ZERO)
    np.testing.assert_array_equal(
        unpack_L(before["parameters"].forward.a), unpack_L(after["parameters"].forward.a)
    )
    for key in ("b", "c"):
        np.testing.assert_array_equal(
            getattr(before["parameters"].forward, key), getattr(after["parameters"].forward, key)
        )
    np.testing.assert_array_equal(before["values"], after["values"])
    np.testing.assert_array_equal(
        before["clocked_parameters"].forward.log_q16, after["clocked_parameters"].forward.log_q16
    )
    assert np.count_nonzero(after["clocked_parameters"].forward.phase_q32) == 0
    for accumulator in ("packed", "clocked"):
        result = after["program"].execute(
            after["addresses"], after["values"], accumulator=accumulator
        )
        assert result.counters.fallback_conversions == 0


def test_public_snapshot_restores_frequency_buffer_and_complete_parent(tmp_path):
    pytest.importorskip("saturn_pub.training")
    pytest.importorskip("mdb.research_session")
    from runtime import initialize_runtime

    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    optimizer = torch.optim.AdamW(model.parameters())
    runtime = initialize_runtime(
        model,
        optimizer,
        {"updates": 0},
        tmp_path / "runtime",
        {"model_id": "tiny-control-mechanics"},
    )
    parent = runtime.capture()
    old = model.model_sha256()
    apply_transition_control(model)
    assert model.model_sha256() != old
    runtime.restore(parent)
    assert model.model_sha256() == old
    assert runtime.controller.current_state_fingerprint() == parent.fingerprint
    assert torch.equal(model.layers[0].transition_frequencies, torch.ones(2))


def test_noop_control_and_scope_validation():
    model = build_spf_model(d_model=4, n_layers=2, state_size=2, seed=17)
    receipt = apply_transition_control(model, frequency=1)
    assert not receipt["changed_buffers"][0]["changed"]
    assert receipt["model_sha256_before"] == receipt["model_sha256_after"]
    for kwargs in ({"frequency": 0.5}, {"frequency": True}, {"layers": (0, 1)}):
        with pytest.raises(ValueError):
            apply_transition_control(model, **kwargs)

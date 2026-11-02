"""Synthetic component-instrument mechanics, without training or quality gates."""

import torch
from memory_tasks import MemoryTask, as_memory_batch
from models import build_spf_model
from phase_diagnosis import (
    diagnose_phase_model,
    phase_component_diagnosis,
    query_parameter_gradients,
)


def test_endpoint_trace_and_gradients_preserve_model_and_existing_gradients(tmp_path):
    model = build_spf_model(d_model=4, n_layers=1, state_size=8, seed=17)
    task = MemoryTask("recall")
    batches = [
        as_memory_batch(*task.sample_batch(4, (delay,), split="feedback", seed=811))
        for delay in (0, 2)
    ]
    old = model.model_sha256()
    model.head.bias.grad = torch.ones_like(model.head.bias)
    stored = model.head.bias.grad.clone()
    result = diagnose_phase_model(
        model,
        batches,
        tmp_path / "panel",
        {"model_id": "tiny-untrained-mechanics"},
        include_native=False,
    )
    assert model.model_sha256() == old and result["model_parameters_unchanged"]
    assert torch.equal(model.head.bias.grad, stored)
    assert result["controller_structure"][0]["transition_read_characters_equal"]
    assert result["controller_structure"][0]["plane_ranks"]["transition"] == 6
    assert len(result["gradient_delay_contrasts"]) == 1
    for row in result["rows"]:
        layer = row["layers"][0]
        assert len(layer["carrier_delta_rms_by_token"]) == row["metadata"]["sequence_length"]
        assert layer["source_write_delta_rms"] > 0
        # Same layer0 suffix bytes must generate identical token-local planes.
        assert layer["suffix_parameter_delta_rms"] == {key: 0.0 for key in ("a", "b", "c", "v")}
        assert row["query_parameter_gradients"]["parameters"]["layers.0.decay_raw"]["elements"] == 8
    assert (tmp_path / "panel/component-endpoint-000.pt").exists()


def test_query_parameter_gradient_does_not_populate_grad_fields():
    model = build_spf_model(d_model=4, n_layers=1, state_size=2, seed=17)
    ids = torch.tensor([[94, 65, 97, 63], [94, 66, 98, 63]])
    detail, vector = query_parameter_gradients(model, ids, torch.tensor([65, 66]))
    assert vector.numel() == sum(parameter.numel() for parameter in model.parameters())
    assert detail["global_gradient_l2"] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_root_wrapper_runs_components_without_a_duplicate_native_panel(tmp_path):
    model = build_spf_model(d_model=4, n_layers=1, state_size=2, seed=17)
    result = phase_component_diagnosis(
        model,
        MemoryTask("recall"),
        tmp_path / "panel",
        {"model_id": "tiny-untrained-mechanics"},
        delays=(0, 2),
        seeds=(81103,),
        batch_size=4,
    )
    assert len(result["rows"]) == 2
    assert result["native_memory"] is None
    assert not (tmp_path / "panel/native-memory").exists()

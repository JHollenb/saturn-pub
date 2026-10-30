from __future__ import annotations

import pytest
import torch

from mdb.forward_recording import ATenStepper, ForwardRecordingError, record_forward


def capture(model, value):
    return record_forward(model.eval(), (value,), model_identity="tiny-frozen-test",
                          execution={"backend": "cpu-fp32"}, input_identity={"case": "anonymous-0"})


def test_native_operation_replay_uses_roots_and_reproduces_every_output():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.GELU(), torch.nn.Linear(4, 2))
    recorded = capture(model, torch.tensor([[1.0, -2.0, 3.0]]))
    machine = ATenStepper(recorded.trace, recorded.initial_values)
    assert machine.pc == 0
    assert set(machine.symbols()["values"]) == set(recorded.initial_values)
    assert all(recorded.trace["values"][key]["producer"] is None for key in machine.environment)
    first = machine.step()
    assert first["exact_outputs"] is True
    assert machine.pc == 1
    assert set(first["event"]["outputs"]) <= set(machine.symbols()["values"])
    output = machine.run()
    assert torch.equal(output, recorded.output)
    assert all(step["exact_outputs"] for step in machine.steps)
    assert len(machine.steps) == len(recorded.trace["events"])
    assert recorded.trace["semantic_labels_supplied"] is False
    assert any(event["module_path"] == "0" for event in recorded.trace["events"])
    assert not model[0]._forward_hooks
    assert not model[0]._forward_pre_hooks


def test_intermediate_oracle_roots_and_wrong_input_geometry_are_rejected():
    recorded = capture(torch.nn.Linear(3, 2), torch.ones(1, 3))
    with pytest.raises(ForwardRecordingError, match="intermediate oracles"):
        ATenStepper(recorded.trace, recorded.observed_values)
    initial = dict(recorded.initial_values)
    key = next(iter(initial))
    initial[key] = initial[key].reshape(-1)
    with pytest.raises(ForwardRecordingError, match="root bytes"):
        ATenStepper(recorded.trace, initial)


def test_in_place_operator_refuses_before_mutating_input_and_removes_hooks():
    class Mutating(torch.nn.Module):
        def forward(self, value):
            return value.add_(1)

    model, value = Mutating(), torch.ones(1, 2)
    with pytest.raises(ForwardRecordingError, match="alias-state contract"):
        capture(model, value)
    assert torch.equal(value, torch.ones(1, 2))
    assert not model._forward_hooks
    assert not model._forward_pre_hooks


def test_native_dictionary_outputs_and_scalar_results_remain_observed():
    class Structured(torch.nn.Module):
        def forward(self, value):
            return {"state": value.square() + 2, "positive": bool(value.sum() > 0)}

    recorded = capture(Structured(), torch.tensor([[-1.0, 3.0]]))
    machine = ATenStepper(recorded.trace, recorded.initial_values)
    replay = machine.run()
    assert torch.equal(replay["state"], recorded.output["state"])
    assert replay["positive"] is recorded.output["positive"]
    assert all(step["exact_outputs"] for step in machine.steps)


def test_semantic_label_maps_cannot_enter_the_recorder():
    model = torch.nn.Linear(3, 2).eval()
    with pytest.raises(ForwardRecordingError, match="no label maps"):
        record_forward(model, (torch.ones(1, 3),), model_identity="tiny", execution={"backend": "cpu"},
                       input_identity={"case": "0"}, metadata={"labels": ["invented-thought"]})


def test_run_requires_exactness_for_previously_executed_steps_too():
    from mdb.forward_symbols import seal_trace

    class Increment(torch.nn.Module):
        def forward(self, value):
            return value + 1

    recorded = capture(Increment(), torch.ones(1, 2))
    body = {key: value for key, value in recorded.trace.items() if key != "fingerprint"}
    output_id = body["events"][0]["outputs"][0]
    body["values"][output_id]["sha256"] = "0" * 64
    machine = ATenStepper(seal_trace(body), recorded.initial_values)
    assert machine.step()["exact_outputs"] is False
    with pytest.raises(ForwardRecordingError, match="differs at step 0"):
        machine.run()
    assert torch.equal(machine.run(require_exact=False), recorded.output)


def test_live_readable_symbols_exclude_future_activations_and_keep_actual_values():
    class Pipeline(torch.nn.Module):
        def forward(self, value):
            return (value + 1) * 3

    recorded = capture(Pipeline(), torch.ones(1, 2))
    machine = ATenStepper(recorded.trace, recorded.initial_values)
    machine.step()
    visible = machine.symbols(readable=True)
    produced = recorded.trace["events"][0]["outputs"][0]
    future = recorded.trace["events"][1]["outputs"][0]
    assert produced in visible["values"]
    assert future not in visible["values"]
    assert visible["values"][produced]["preview"] == [2., 2.]
    assert "+ 1" in visible["values"][produced]["readable"]["expression"]
    assert visible["values"][produced]["readable"]["uses"] == [1]

from __future__ import annotations

import pytest
import torch

from mdb.discovery_runtime import ReplayForward
from mdb.forward_prefix import ForwardPrefixError, record_prefix
from mdb.forward_symbols import verify_trace


class Counter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value, *, bias=0):
        self.calls += 1
        return value + bias + 100


class PrefixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(3, 3)
        self.post = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.LayerNorm(3), torch.nn.Linear(3, 5), torch.nn.GELU(), torch.nn.Linear(5, 3))])
        self.future = Counter()
        self.later = Counter()

    def forward(self, value, *, use_future=True):
        decoded = self.decoder(value)
        state = decoded + self.post[0](decoded)
        if use_future:
            state = self.future(state, bias=7)
        return self.later(state)


def capture(model, *, observe=("decoder",), **kwargs):
    return record_prefix(model, (torch.ones(1, 3),), stop_before="future", observe=observe,
                         model_identity="tiny-prefix", execution={"backend": "cpu-fp32"},
                         input_identity="tiny-input", **kwargs)


def hook_inventory(model):
    return {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
            for name, module in model.named_modules()}


def test_real_boundary_aborts_before_body_and_preserves_completed_observations():
    torch.manual_seed(7)
    model = PrefixModel().eval()
    events = []
    pre = model.future.register_forward_pre_hook(lambda module, args: events.append("future-pre"))
    existing = model.decoder.register_forward_hook(lambda module, args, out: events.append("decoder"))
    before = hook_inventory(model)
    try:
        result = capture(model)
        assert model.future.calls == model.later.calls == 0
        assert events == ["decoder"]
        assert hook_inventory(model) == before
        assert not model.training and all(not m.training for m in model.modules())
        verify_trace(result.trace)
        assert result.trace["capture_status"] == "complete"
        extent = result.trace["metadata"]["prefix_contract"]
        assert extent["execution_extent"] == "prefix-wrapper"
        assert extent["full_model_outputs_computed"] is False
        assert extent["observations"]["decoder"]["completed_invocations"] == 1
        paths = {row["module_path"] for row in result.trace["events"]}
        assert not any(path.startswith(("native.future", "native.later")) for path in paths)
        assert result.output["boundary"]["kwargs"] == {"bias": 7}
        with torch.no_grad():
            z = model.decoder(torch.ones(1, 3))
            expected = z + model.post[0](z)
        assert torch.equal(result.output["observed"]["decoder"][0], z)
        assert torch.equal(result.output["boundary"]["args"][0], expected)
    finally:
        pre.remove()
        existing.remove()


def test_prefix_replays_and_forks_with_typed_source_act_without_running_future():
    torch.manual_seed(11)
    model = PrefixModel().eval().requires_grad_(False)
    recorded = capture(model)
    replay = ReplayForward(recorded)
    law = replay.catalog.resolve("module:native.post.0")
    while replay.pc < law["exit"]:
        replay.advance_one()
    parent = replay.capture()
    changed = replay.fork(parent)
    port = changed.catalog.resolve("module:native.post.0:output")["value_id"]
    receipt = changed.apply(port, torch.zeros_like(changed.read(port)))
    assert receipt["operation"] == "mdb.discovery.ssa-act"
    while not changed.complete:
        changed.advance_one()
    output = changed.outputs()
    assert torch.equal(output["boundary"]["args"][0], output["observed"]["decoder"][0])
    assert torch.equal(replay.read(port), recorded.observed_values[port])
    while not replay.complete:
        replay.advance_one()
    assert torch.equal(replay.outputs()["boundary"]["args"][0], recorded.output["boundary"]["args"][0])
    assert model.future.calls == model.later.calls == 0


def test_unreached_boundary_and_missing_observation_clean_up_all_added_hooks():
    model = PrefixModel().eval()
    before = hook_inventory(model)
    with pytest.raises(ForwardPrefixError, match="was not reached"):
        capture(model, kwargs={"use_future": False})
    assert hook_inventory(model) == before
    assert model.future.calls == 0 and model.later.calls == 1
    with pytest.raises(ForwardPrefixError, match="observations did not complete"):
        capture(model, observe=("later",))
    assert hook_inventory(model) == before
    assert model.future.calls == 0 and model.later.calls == 1


def test_original_exception_propagates_and_hooks_are_cleaned():
    model = PrefixModel().eval()
    def broken(module, args):
        raise RuntimeError("original prefix failure")
    handle = model.decoder.register_forward_pre_hook(broken)
    before = hook_inventory(model)
    try:
        with pytest.raises(RuntimeError, match="original prefix failure"):
            capture(model)
        assert hook_inventory(model) == before
        assert model.future.calls == model.later.calls == 0
    finally:
        handle.remove()


def test_unsupported_occurrence_and_training_rejected_before_execution():
    model = PrefixModel().eval()
    before = hook_inventory(model)
    with pytest.raises(ForwardPrefixError, match="occurrence=0"):
        capture(model, occurrence=1)
    assert hook_inventory(model) == before
    model.train()
    with pytest.raises(ForwardPrefixError, match="already in evaluation"):
        capture(model)
    assert model.training and model.decoder.training
    assert model.future.calls == model.later.calls == 0


def test_completed_repeated_observations_are_ordered_without_copy_ops():
    class Repeated(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stage = torch.nn.Identity()
            self.future = Counter()

        def forward(self, value):
            first = self.stage(value + 1)
            second = self.stage(first + 2)
            return self.future(second)
    model = Repeated().eval()
    recorded = record_prefix(model, (torch.ones(1, 2),), stop_before="future", observe=("stage",),
        model_identity="tiny-repeated-prefix", execution={"backend": "cpu-fp32"}, input_identity="ones")
    observed = recorded.output["observed"]["stage"]
    assert len(observed) == 2 and torch.equal(observed[0], torch.full((1, 2), 2.0))
    assert torch.equal(observed[1], torch.full((1, 2), 4.0))
    assert len(recorded.trace["events"]) == 2
    assert model.future.calls == 0

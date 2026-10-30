from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from mdb.discovery_runtime import ReplayForward
from mdb.forward_ports import NativePortCatalog, NativePortError
from mdb.forward_prefix import record_prefix
from mdb.forward_recording import ATenStepper, ForwardRecordingError, record_forward
from mdb.forward_symbols import seal_trace, verify_trace


def capture(model, args, kwargs=None, **options):
    return record_forward(model.eval(), args, kwargs, model_identity="tiny-native-port-test",
                          execution={"backend": "cpu-fp32"}, input_identity="fixed-input",
                          capture_ports=True, **options)


def hooks(model):
    return {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
            for name, module in model.named_modules()}


class BiasedAffine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[2., -1.], [1., 3.]]))
        self.bias = torch.nn.Parameter(torch.tensor([7., -5.]))

    def forward(self, value):
        return value @ self.weight.T + self.bias


class AffineConsumer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = BiasedAffine()

    def forward(self, value):
        return self.affine(value) * 3


def test_actual_return_includes_bias_and_is_live_before_its_first_consumer():
    model = AffineConsumer().eval()
    recorded = capture(model, (torch.tensor([[1., 4.]]),))
    catalog = NativePortCatalog(recorded.trace)
    port = catalog.resolve("module:affine:return")
    mm = next(event for event in recorded.trace["events"] if event["operator"] == "aten.mm.default")
    assert port["value_id"] != mm["outputs"][0]
    assert torch.equal(recorded.observed_values[port["value_id"]], torch.tensor([[5., 8.]]))
    use = catalog.first_consumers("module:affine:output")[0]
    assert use["first_consumer"]["operator"] == "aten.mul.Tensor"
    assert use["first_consumer"]["pc"] == port["exit_pc"]
    assert not use["already_consumed"]
    weight = catalog.resolve("parameter:affine.weight")
    bias = catalog.resolve("parameter:affine.bias")
    assert recorded.trace["values"][weight["value_id"]]["producer"] is None
    assert torch.equal(recorded.initial_values[bias["value_id"]], model.affine.bias)
    assert catalog.first_consumers("parameter:affine.weight")[0]["first_consumer"]["operator"] == "aten.permute.default"
    replay = ReplayForward(recorded)
    while replay.pc < port["available_pc"]:
        replay.advance_one()
    assert torch.equal(replay.read(port["value_id"]), torch.tensor([[5., 8.]]))
    replay.apply(port["value_id"], torch.zeros(1, 2))
    while not replay.complete:
        replay.advance_one()
    assert torch.equal(replay.outputs(), torch.zeros(1, 2))
    assert torch.equal(ATenStepper(recorded.trace, recorded.initial_values).run(), recorded.output)


def test_kwargs_and_dictionary_keys_are_observed_without_semantic_annotation():
    class Keyword(torch.nn.Module):
        def forward(self, value, *, symbols, gain):
            return {"symbols": value + symbols * gain, "nested": (value, None)}

    value, other = torch.tensor([2., 3.]), torch.tensor([5., 7.])
    recorded = capture(Keyword(), (value,), {"symbols": other, "gain": 2.})
    catalog = NativePortCatalog(recorded.trace)
    kwargs = catalog.resolve("module:<root>:kwargs")
    assert kwargs["encoded"]["gain"] == 2.
    selected = catalog.resolve("module:<root>:input", path=["kwargs", "symbols"])
    returned = catalog.resolve("module:<root>:return", path=["symbols"])
    assert torch.equal(recorded.initial_values[selected["value_id"]], other)
    assert torch.equal(recorded.observed_values[returned["value_id"]], torch.tensor([12., 17.]))
    nested = catalog.resolve("module:<root>:return", path=["nested", 1])
    assert nested["encoded"] is None and nested["values"] == []
    verify_trace(recorded.trace)


def test_existing_input_and_return_hooks_are_included_in_actual_ports_and_preserved():
    model = torch.nn.Identity().eval()
    before = model.register_forward_pre_hook(lambda module, args: (args[0] + 4,))
    after = model.register_forward_hook(lambda module, args, output: output * 2)
    inventory = hooks(model)
    try:
        recorded = capture(model, (torch.tensor([3.]),))
        catalog = NativePortCatalog(recorded.trace)
        inp, out = (catalog.resolve("module:<root>:" + edge)["value_id"] for edge in ("input", "return"))
        assert torch.equal(recorded.observed_values[inp], torch.tensor([7.]))
        assert torch.equal(recorded.observed_values[out], torch.tensor([14.]))
        assert hooks(model) == inventory
    finally:
        before.remove()
        after.remove()


def test_shared_repeated_module_calls_keep_physical_occurrences_and_state_aliases():
    class Shared(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = BiasedAffine()
            self.right = self.left

        def forward(self, value):
            return self.right(self.left(value))

    recorded = capture(Shared(), (torch.ones(1, 2),))
    catalog = NativePortCatalog(recorded.trace)
    with pytest.raises(NativePortError, match="2 matching invocations"):
        catalog.resolve("module:left:return")
    first = catalog.resolve("module:left:return", occurrence=0)
    second = catalog.resolve("module:right:return", occurrence=1)
    assert first["module_id"] == second["module_id"]
    assert second["declared_module_aliases"] == ["left", "right"]
    assert first["value_id"] != second["value_id"]
    assert catalog.resolve("call:" + second["call_id"] + ":return")["value_id"] == second["value_id"]
    one, two = (catalog.resolve("parameter:" + name + ".weight") for name in ("left", "right"))
    assert one["value_id"] == two["value_id"]
    assert one["declared_aliases"] == ["left.weight", "right.weight"]


def test_zero_op_returns_unused_parameters_buffers_and_absent_bias_are_real_roots():
    class PassThrough(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = torch.nn.Parameter(torch.tensor([9.]))
            self.register_parameter("bias", None)
            self.register_buffer("cache", torch.tensor([11.]))

        def forward(self, value):
            return value

    recorded = capture(PassThrough(), (torch.tensor([4.]),))
    assert recorded.trace["events"] == []
    catalog = NativePortCatalog(recorded.trace)
    assert catalog.resolve("module:<root>:input")["value_id"] == catalog.resolve("module:<root>:return")["value_id"]
    assert catalog.resolve("module:<root>:return")["entry_pc"] == catalog.resolve("module:<root>:return")["exit_pc"] == 0
    assert catalog.resolve("parameter:unused")["value_id"] in recorded.initial_values
    assert catalog.resolve("buffer:cache")["value_id"] in recorded.initial_values
    assert catalog.resolve("parameter:bias")["absent"] is True
    assert catalog.first_consumers("parameter:unused")[0]["first_consumer"] is None
    assert torch.equal(ATenStepper(recorded.trace, recorded.initial_values).run(), torch.tensor([4.]))


def test_view_and_pass_through_returns_keep_direct_ssa_edges_separate_from_alias_uses():
    class Viewed(torch.nn.Module):
        def forward(self, value):
            return value.view(2, 2)

    model = torch.nn.Sequential(Viewed(), torch.nn.Identity()).eval()
    recorded = capture(model, (torch.arange(4.),))
    catalog = NativePortCatalog(recorded.trace)
    root_input = catalog.resolve("module:<root>:input")["value_id"]
    view = catalog.resolve("module:0:return")["value_id"]
    assert root_input != view
    assert catalog.resolve("module:1:return")["value_id"] == view
    assert catalog.first_consumers("module:<root>:input")[0]["first_consumer"]["operator"] == "aten.view.default"
    assert catalog.first_consumers("module:0:return")[0]["first_consumer"] is None


def test_first_use_after_return_does_not_hide_an_earlier_direct_ssa_consumer():
    class ReadAndReturn(torch.nn.Module):
        def forward(self, value):
            value + 3
            return value

    class Later(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.reader = ReadAndReturn()

        def forward(self, value):
            return self.reader(value) * 7

    recorded = capture(Later(), (torch.ones(1),))
    catalog = NativePortCatalog(recorded.trace)
    use = catalog.first_consumers("module:reader:return")[0]
    assert use["first_consumer"]["operator"] == "aten.add.Tensor"
    assert use["first_after_port"]["operator"] == "aten.mul.Tensor"
    assert use["first_consumer"]["pc"] < use["port_pc"] == use["first_after_port"]["pc"]
    assert use["already_consumed"] is True


def test_convolution_return_and_named_bias_are_not_inferred_affine_scopes():
    conv = torch.nn.Conv1d(1, 1, 2, bias=True).eval()
    with torch.no_grad():
        conv.weight.fill_(2.)
        conv.bias.fill_(3.)
    recorded = capture(conv, (torch.tensor([[[1., 2., 4.]]]),))
    catalog = NativePortCatalog(recorded.trace)
    returned = catalog.resolve("module:<root>:return")
    assert torch.equal(recorded.observed_values[returned["value_id"]], torch.tensor([[[9., 15.]]]))
    assert catalog.resolve("parameter:bias")["value_id"] in recorded.trace["events"][0]["inputs"]


def test_caught_child_exception_has_no_fabricated_return_and_hooks_close():
    class Explodes(torch.nn.Module):
        def forward(self, value):
            raise RuntimeError("declared child failure")

    class Catches(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Explodes()

        def forward(self, value):
            try:
                return self.child(value)
            except RuntimeError:
                return value + 1

    model = Catches().eval()
    inventory = hooks(model)
    recorded = capture(model, (torch.ones(1),))
    catalog = NativePortCatalog(recorded.trace)
    child = next(call for call in catalog.list(kind="call") if call["parent"] is not None)
    assert child["status"] == "aborted" and "return" not in child
    with pytest.raises(NativePortError, match="aborted"):
        catalog.resolve("module:child:return")
    assert catalog.resolve("module:<root>:return")["status"] == "returned"
    assert hooks(model) == inventory
    with pytest.raises(RuntimeError, match="declared child failure"):
        capture(Explodes(), (torch.ones(1),))
    assert not model.child._forward_hooks and not model.child._forward_pre_hooks


def test_none_return_inside_an_existing_exception_handler_is_not_aborted():
    class NoneReturn(torch.nn.Module):
        def forward(self, value):
            return None

    try:
        raise RuntimeError("outer exception remains active")
    except RuntimeError:
        recorded = capture(NoneReturn(), (torch.ones(1),))
    port = NativePortCatalog(recorded.trace).resolve("module:<root>:return")
    assert port["encoded"] is None and port["status"] == "returned"


def test_prefix_abort_does_not_pop_parent_or_claim_the_target_return():
    class Prefix(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.before = torch.nn.Linear(2, 2)
            self.future = torch.nn.Identity()

        def forward(self, value):
            return self.future(self.before(value))

    model = Prefix().eval()
    inventory = hooks(model)
    recorded = record_prefix(model, (torch.ones(1, 2),), stop_before="future", observe=("before",),
        model_identity="tiny-prefix-ports", execution={"backend": "cpu"}, input_identity="input", capture_ports=True)
    catalog = NativePortCatalog(recorded.trace)
    assert catalog.resolve("module:native.before:return")["status"] == "returned"
    with pytest.raises(NativePortError, match="0 matching invocations"):
        catalog.resolve("module:native.future:return")
    native = next(call for call in catalog.list(kind="call") if call["module_id"] == next(
        module["id"] for module in catalog.list(kind="module") if module["canonical_path"] == "native"))
    assert native["status"] == "aborted" and "return" not in native
    assert catalog.resolve("module:<root>:return")["status"] == "returned"
    assert hooks(model) == inventory


def test_default_false_trace_identity_and_root_inventory_are_unchanged():
    model = BiasedAffine().eval()
    options = {"model_identity": "legacy", "execution": {"backend": "cpu"}, "input_identity": "fixed"}
    value = torch.ones(1, 2)
    implicit = record_forward(model, (value,), **options)
    explicit = record_forward(model, (value,), capture_ports=False, **options)
    assert implicit.trace == explicit.trace
    assert implicit.initial_values.keys() == explicit.initial_values.keys()
    assert "native_ports" not in implicit.trace["metadata"]
    with pytest.raises(NativePortError, match="capture_ports=True"):
        NativePortCatalog(implicit.trace)


def test_catalog_rejects_resealed_future_return_and_unknown_named_root():
    recorded = capture(AffineConsumer(), (torch.ones(1, 2),))
    for tamper in ("future", "parameter"):
        body = json.loads(json.dumps({key: value for key, value in recorded.trace.items() if key != "fingerprint"}))
        if tamper == "future":
            body["metadata"]["native_ports"]["calls"][1]["exit_pc"] = 0
        else:
            body["metadata"]["native_ports"]["state"]["parameters"][0]["value_id"] = "v999999"
        with pytest.raises(NativePortError):
            NativePortCatalog(seal_trace(body))


def test_dynamic_hook_mutation_refuses_instead_of_claiming_an_early_hook_return():
    class Dynamic(torch.nn.Module):
        def forward(self, value):
            self.added = self.register_forward_hook(lambda module, args, result: result + 9)
            return value

    model = Dynamic().eval()
    try:
        with pytest.raises(ForwardRecordingError, match="hooks changed"):
            capture(model, (torch.ones(1),))
        assert len(model._forward_hooks) == 1  # the user-owned dynamic hook remains
        assert not model._forward_pre_hooks
    finally:
        model.added.remove()


def test_framework_free_catalog_and_cli_in_subprocess(tmp_path):
    recorded = capture(torch.nn.Identity(), (torch.ones(1),))
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(recorded.trace))
    script = """
import sys
sys.modules['torch'] = None
sys.modules['numpy'] = None
from mdb.forward_ports import main
main([sys.argv[1], '--select', 'module:<root>:return', '--first-consumers'])
"""
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=True, text=True,
                            capture_output=True, env=os.environ.copy())
    assert json.loads(result.stdout)[0]["first_consumer"] is None

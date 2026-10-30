import operator
from dataclasses import replace

import pytest
import torch
from saturn_pub.core import Act
from saturn_pub.symbols import SymbolTable
from torch import nn

from mdb.forward_debugger import (
    ForwardAdapter,
    ForwardDebuggerError,
    auto_symbol_table,
    create_forward_debugger,
    create_forward_session,
    execution_table,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 4)
        self.right = nn.Linear(4, 2)

    def forward(self, values):
        return self.right(torch.relu(self.left(values)))


def _session():
    torch.manual_seed(17)
    model = TinyMLP().eval()
    values = torch.tensor([[0.25, -0.5, 1.0]])
    return model, values, create_forward_session(
        model, (values,), model_identity="tiny-mlp-v1"
    )


def _finish(session):
    while not session.read("done"):
        session.continue_(1)


def test_fx_nodes_step_as_native_public_transitions_and_match_forward():
    model, values, session = _session()
    expected = model(values)
    initial = execution_table(session)
    assert initial["authority"] == "read-only-structural-forward-observation"
    assert initial["operation_level"] == "native-neural-graph"
    assert all(row["semantic_role"] == "unresolved" for row in initial["nodes"])
    assert session.read("cursor") == 0

    session.continue_(1)
    table = execution_table(session)
    assert table["cursor"] == 1
    assert any(row["status"] == "available" for row in table["nodes"])
    assert all("raw" in row for row in table["nodes"])

    _finish(session)
    torch.testing.assert_close(session.read("output"), expected)
    assert session.read("output_ready") is True
    assert session.receipts
    assert all(
        transition["native_operation"]["consumer"] == "torch-fx-original-forward"
        for receipt in session.receipts
        for transition in receipt.to_dict().get("transitions", [])
    )


def test_macro_continue_handles_non_dense_view_snapshots_like_single_steps():
    class Views(nn.Module):
        def forward(self, values):
            sliced = values[:, 1:5]
            spread = sliced.unsqueeze(1).expand(-1, 3, -1)
            return spread.sum(1) + sliced

    model = Views().eval()
    values = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    macro = create_forward_session(model, (values,), model_identity="view-layout-v1")
    single = create_forward_session(model, (values,), model_identity="view-layout-v1")
    count = sum(not spec.placeholder for spec in macro.adapter.node_specs)
    macro.continue_(count)
    _finish(single)
    assert torch.equal(macro.read("output"), model(values))
    assert torch.equal(macro.read("output"), single.read("output"))
    for spec in macro.adapter.node_specs:
        left, right = macro.read(spec.slot), single.read(spec.slot)
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
            assert left.stride() == right.stride()


def test_native_output_abi_retains_nested_tensor_structure_without_semantic_labels():
    class NestedOutput(nn.Module):
        def forward(self, values):
            transformed = values + 1
            return {"hidden": (transformed, values), "count": 1}

    values = torch.tensor([[2.0, 3.0]])
    model = NestedOutput().eval()
    session = create_forward_session(model, (values,), model_identity="nested-output-v1")
    _finish(session)
    expected = model(values)
    actual = session.read("output")
    torch.testing.assert_close(actual["hidden"][0], expected["hidden"][0])
    torch.testing.assert_close(actual["hidden"][1], expected["hidden"][1])
    table = execution_table(session)
    assert table["output"]["abi"] == "native-graph-output"
    assert table["output"]["descriptor"]["kind"] == "mapping"
    assert table["semantic_inference"] == "not-performed"
    assert table["output"]["sample"][0]["key"] == "hidden"


def test_auto_symbols_are_exact_structural_results_with_clock_bound_resolution():
    _model, _values, session = _session()
    session.continue_(2)
    table = auto_symbol_table(session)
    assert table.bindings
    assert all(binding.symbol.startswith("v") for binding in table.bindings)
    assert all("semantic role" in binding.claim for binding in table.bindings)
    assert all(binding.status == "observed" for binding in table.bindings)

    adapter = session.adapter
    for binding in table.bindings:
        assert binding.resolve(session, context=adapter.context) == binding.locations
    session.continue_(1)
    with pytest.raises(ValueError, match="clock mismatch"):
        table.bindings[0].resolve(session, context=adapter.context)


def test_public_statecut_restore_replays_the_same_graph_values():
    _model, _values, session = _session()
    root = session.capture(retain=True)
    session.continue_(2)
    middle = session.capture(retain=True)
    _finish(session)
    expected = session.read("output")
    middle.verify()

    session.restore(middle)
    assert session.read("done") is False
    _finish(session)
    torch.testing.assert_close(session.read("output"), expected)
    session.restore(root)
    assert session.read("cursor") == 0
    assert session.read("done") is False


def test_only_live_computed_tensors_are_writable_and_copy_zero_restore_exactly():
    model, values, session = _session()
    root = session.capture(retain=True)
    session.continue_(2)
    frame = session.inspect()
    slots = {row["name"]: row for row in frame.surface["slots"]}
    assert slots["v000"]["writable"] is False
    assert slots["v001"]["writable"] is False
    assert slots["v002"]["writable"] is True
    assert slots["v003"]["writable"] is False
    assert slots["output"]["writable"] is False
    assert slots["cursor"]["writable"] is False
    with pytest.raises(ValueError, match="read-only state port"):
        session.apply(Act(
            "stale-left",
            ("v001",),
            ("v001",),
            lambda reads: {"v001": reads["v001"].clone()},
        ))

    session.restore(root)
    session.continue_(1)
    middle = session.capture(retain=True)
    frame = session.inspect()
    slots = {row["name"]: row for row in frame.surface["slots"]}
    assert slots["v001"]["writable"] is True
    assert slots["v002"]["writable"] is False

    original_left = session.read("v001")
    copy_act = Act(
        "copy-left",
        ("v001",),
        ("v001",),
        lambda reads: {"v001": reads["v001"].clone()},
    )
    session.apply(copy_act)
    torch.testing.assert_close(session.read("v001"), original_left)

    zero_act = Act(
        "zero-left",
        ("v001",),
        ("v001",),
        lambda reads: {"v001": torch.zeros_like(reads["v001"])},
    )
    session.apply(zero_act)
    _finish(session)
    zero_output = session.read("output")
    expected_zero = model.right(torch.relu(torch.zeros_like(original_left)))
    torch.testing.assert_close(zero_output, expected_zero)

    session.restore(middle)
    torch.testing.assert_close(session.read("v001"), original_left)
    _finish(session)
    torch.testing.assert_close(session.read("output"), model(values))


def test_parameter_get_attr_is_observed_but_never_an_intervention_port():
    class ParameterRead(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, values):
            return values + self.offset

    model = ParameterRead().eval()
    session = create_forward_session(
        model, (torch.zeros(1, 2),), model_identity="parameter-get-attr-v1"
    )
    session.continue_(1)
    rows = {row["name"]: row for row in session.inspect().surface["slots"]}
    get_attr = next(row for row in rows.values() if row["role"] == "computed")
    assert get_attr["producer"].startswith("fx.get_attr:")
    assert get_attr["writable"] is False


def test_live_forward_debugger_refreshes_symbols_and_forward_table_after_step_restore():
    model, values, _session_for_reference = _session()
    debugger = create_forward_debugger(
        model, (values,), model_identity="live-forward-debugger-v1"
    )
    initial = debugger.execute("info symbols")
    assert initial["qualified"] == []

    debugger.execute("step")
    forward = debugger.execute("info forward")
    assert forward["cursor"] == 1
    assert forward["operation_level"] == "native-neural-graph"
    symbols = debugger.execute("info symbols")
    assert symbols["qualified"]
    assert all(row["symbol"].startswith("v") for row in symbols["qualified"])
    context = debugger.symbol_context
    capture = debugger.execute("capture middle")

    debugger.execute("step")
    assert debugger.execute("info forward")["cursor"] == 2
    debugger.execute("restore middle")
    assert debugger.execute("info forward")["cursor"] == 1
    restored = debugger.execute("info symbols")
    assert debugger.symbol_context == context
    assert restored["qualified"]
    assert debugger.execute("resolve v001")["locations"][0]["execution_point"] == (
        restored["qualified"][0]["locations"][0]["execution_point"]
    )
    assert capture["fingerprint"] == debugger.cuts["middle"].fingerprint


def test_observed_annotations_follow_exact_cut_and_never_change_model_execution():
    model, values, session = _session()
    debugger = create_forward_debugger(model, (values,), model_identity="tiny-mlp-v1")
    debugger.execute("step")
    cut = debugger.session.capture()
    original = auto_symbol_table(debugger.session).bindings[0]
    alias = replace(original, symbol="observed.test_value", claim="Known-wiring observation.")
    debugger.attach_annotations(SymbolTable((alias,)))
    debugger.execute("info symbols")
    assert any(row.symbol == alias.symbol for row in debugger.symbols.bindings)
    debugger.execute("step")
    debugger.execute("info symbols")
    assert all(row.symbol != alias.symbol for row in debugger.symbols.bindings)
    debugger.session.restore(cut)
    debugger.execute("info symbols")
    assert any(row.symbol == alias.symbol for row in debugger.symbols.bindings)
    with pytest.raises(ValueError, match="context"):
        debugger.attach_annotations(SymbolTable((replace(alias, context="wrong"),)))
    _finish(debugger.session)
    assert torch.equal(debugger.session.read("output"), model(values))
    assert session.read("cursor") == 0


def test_trace_uses_fx_proxies_and_rejects_stochastic_or_inplace_graphs():
    seen_numeric = []

    class Probe(nn.Module):
        def forward(self, values):
            seen_numeric.append(isinstance(values, torch.Tensor))
            return values + 1

    probe = Probe().eval()
    session = create_forward_session(
        probe, (torch.ones(1, 2),), model_identity="proxy-check-v1"
    )
    assert seen_numeric == [False]
    _finish(session)
    assert torch.equal(session.read("output"), torch.ones(1, 2) + 1)

    with pytest.raises(ForwardDebuggerError, match="stochastic"):
        create_forward_session(
            nn.Dropout(0.5).eval(), (torch.ones(1, 2),), model_identity="dropout-v1"
        )

    class InPlace(nn.Module):
        def forward(self, values):
            return values.add_(1)

    with pytest.raises(ForwardDebuggerError, match="in-place"):
        create_forward_session(
            InPlace().eval(), (torch.ones(1, 2),), model_identity="inplace-v1"
        )

    class FunctionalInPlace(nn.Module):
        def forward(self, values):
            return torch.relu_(values)

    with pytest.raises(ForwardDebuggerError, match="in-place"):
        create_forward_session(
            FunctionalInPlace().eval(), (torch.ones(1, 2),), model_identity="functional-inplace-v1"
        )

    with pytest.raises(ForwardDebuggerError, match="in-place"):
        create_forward_session(
            nn.Sequential(nn.ReLU(inplace=True)).eval(),
            (torch.ones(1, 2),),
            model_identity="module-inplace-v1",
        )


@pytest.mark.parametrize("target", ["__iadd__", operator.iadd])
def test_preflight_rejects_python_inplace_operators(target):
    graph = torch.fx.Graph()
    values = graph.placeholder("values")
    if isinstance(target, str):
        mutated = graph.call_method(target, (values, 1))
    else:
        mutated = graph.call_function(target, (values, 1))
    graph.output(mutated)
    model = torch.fx.GraphModule(nn.Module(), graph).eval()
    source = torch.ones(1, 2)
    with pytest.raises(ForwardDebuggerError, match="in-place"):
        ForwardAdapter(model, (source,), model_identity="python-inplace-v1")
    assert torch.equal(source, torch.ones(1, 2))


def test_trace_rejects_unproven_constant_randomness():
    class ConstantRandom(nn.Module):
        def forward(self, values):
            return values + torch.rand(1)

    with pytest.raises(ForwardDebuggerError, match="unproven get_attr"):
        create_forward_session(
            ConstantRandom().eval(), (torch.ones(1, 2),), model_identity="constant-rand-v1"
        )


def test_nonpersistent_buffer_drift_is_rejected_during_trace():
    class MutatingBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cache", torch.ones(1), persistent=False)

        def forward(self, values):
            self.cache.add_(1)
            return values + self.cache

    with pytest.raises(ForwardDebuggerError, match="mutated module state"):
        create_forward_session(
            MutatingBuffer().eval(), (torch.ones(1, 2),), model_identity="nonpersistent-v1"
        )


def test_scalar_set_module_attributes_are_sealed_without_entering_tensor_state():
    class SetMetadata(nn.Module):
        def __init__(self):
            super().__init__()
            self.snap = set()
            self.enabled = True

        def forward(self, values):
            return values + 1 if self.enabled else values

    model = SetMetadata().eval()
    session = create_forward_session(
        model, (torch.ones(1, 2),), model_identity="set-metadata-v1"
    )
    _finish(session)
    torch.testing.assert_close(session.read("output"), torch.full((1, 2), 2.0))


def test_nested_eval_modules_keep_exact_forward_after_graph_container_normalization():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, values):
            return self.linear(values)

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

        def forward(self, values):
            return self.inner(values)

    torch.manual_seed(29)
    model = Outer().eval()
    values = torch.tensor([[0.5, -1.0]])
    session = create_forward_session(model, (values,), model_identity="nested-eval-v1")
    _finish(session)
    torch.testing.assert_close(session.read("output"), model(values))

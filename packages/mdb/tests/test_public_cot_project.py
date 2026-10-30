from __future__ import annotations

import pytest
from saturn_pub import Act, Adapter, ExecutionPoint, Session, StateAddress, TransitionSpec
from saturn_pub.symbols import SymbolBinding

from mdb.project import Project
from mdb.public_cot_project import (
    PublicCotProjectError,
    export_public_cot_project,
    verify_public_cot_project,
)
from mdb.symbol_lifecycle import SymbolLifecycle, SymbolOccurrence, SymbolUse


class _Adapter(Adapter):
    model_identity = "public-cot-project-test-model"
    execution = {"family": "public-cot-test", "adapter": "test-native"}

    def boundary(self, state):
        return f"step:{state['value']}"

    def validate(self, state):
        if set(state) != {"value"} or type(state["value"]) is not int:
            raise ValueError("invalid test state")

    def advance(self, state):
        return {"value": state["value"] + 1}

    def addresses(self, state):
        return ("value",)

    def point(self, state):
        step = state["value"]
        return ExecutionPoint(
            "public-cot-test", step, "consume", operator="native-test", index=step
        )

    def transition(self, state):
        return TransitionSpec(
            "native-test",
            reads=("value",),
            writes=("value",),
            consumer="native-suffix",
            footprint="declared",
        )


def _session():
    return Session(_Adapter(), {"value": 0})


class _ForkAdapter(Adapter):
    model_identity = "public-cot-fork-test-model"
    execution = {"family": "public-cot-fork-test", "adapter": "test-native"}

    def boundary(self, state):
        return f"step:{state['clock']}"

    def validate(self, state):
        if set(state) != {"value", "clock"} or any(
            type(state[name]) is not int for name in ("value", "clock")
        ):
            raise ValueError("invalid fork state")

    def advance(self, state):
        return {"value": state["value"] + 1, "clock": state["clock"] + 1}

    def addresses(self, state):
        return ("value", "clock")

    def point(self, state):
        return ExecutionPoint(
            "public-cot-fork-test", state["clock"], "consume", operator="native-test"
        )

    def transition(self, state):
        return TransitionSpec(
            "native-test",
            reads=("value", "clock"),
            writes=("value", "clock"),
            consumer="native-suffix",
            footprint="declared",
        )


def _fork_host_operation(value):
    def operation(inputs):
        return {"value": value, "clock": inputs["clock"] + 1}

    return operation


_FORK_HOST_OPERATIONS = {}
for _value in (1, 2):
    _identifier = f"mdb.test.public-cot-fork.host-{_value}"
    _operation = _fork_host_operation(_value)
    Act.register(_identifier, "1", _operation, pure=True)
    _FORK_HOST_OPERATIONS[_value] = (_identifier, _operation)


def _fork_host_act(value):
    identifier, operation = _FORK_HOST_OPERATIONS[value]
    return Act.registered(
        f"host-row-{value}",
        ("value", "clock"),
        ("value", "clock"),
        operation_id=identifier,
        operation_version="1",
        implementation=operation,
        parameters={"value": value},
    )


def test_export_persists_actual_cuts_bindings_and_lifecycle_without_hydration(tmp_path, monkeypatch):
    native = _session()
    cut0 = native.capture()
    native.continue_()
    cut1 = native.capture()
    receipt1 = native.receipts[-1]
    native.continue_()
    cut2 = native.capture()
    receipt2 = native.receipts[-1]

    binding_session = Session.from_cut(native.adapter, cut1)
    binding = SymbolBinding.bind(
        binding_session,
        "cot.value",
        (StateAddress("value"),),
        context="prompt-context-sha256",
        context_slots=(),
        evidence=(f"cas://sha256/{cut1.fingerprint}",),
        claim="value carrier at the native consume boundary",
        status="observed",
    )
    definition = SymbolOccurrence.for_binding(
        binding,
        occurrence_id="definition-1",
        session_id="cot-session",
        source_cut=cut1.fingerprint,
        locations=binding.locations,
        receipt=receipt1.fingerprint,
        evidence_refs=(f"cas://sha256/{receipt1.fingerprint}",),
        status="observed",
    )
    use = SymbolUse(
        use_id="use-1",
        binding_fingerprint=binding.to_dict()["fingerprint"],
        session_id="cot-session",
        consumer="native-suffix",
        source_cut=cut1.fingerprint,
        parent_cut=cut1.fingerprint,
        result_cut=cut2.fingerprint,
        receipt=receipt2.fingerprint,
        locations=binding.locations,
        evidence_refs=(f"cas://sha256/{receipt2.fingerprint}",),
        status="observed",
    )
    lifecycle = SymbolLifecycle.define(binding, definitions=(definition,), uses=(use,))

    project = Project(tmp_path / "project")
    exported = export_public_cot_project(
        project,
        "cot-session",
        native,
        [cut0, cut1, cut2],
        bindings=[binding],
        lifecycles=[lifecycle],
    )
    assert exported["schema"] == "mdb-public-cot-project-v1"
    assert exported["payload_hydrated"] is False
    assert exported["lifecycles"][0]["definitions"] == 1
    assert exported["lifecycles"][0]["uses"] == 1

    def forbidden_load(*args, **kwargs):
        raise AssertionError("verification must not hydrate a StateCut payload")

    monkeypatch.setattr(project.store, "load", forbidden_load)
    proof = verify_public_cot_project(
        project,
        "cot-session",
        lifecycle_ids=(lifecycle.fingerprint,),
    )
    assert proof["native_payload_reads"] == 0
    assert len(proof["cut_storage"]) == 3
    assert proof["bindings"][0]["locations"][0]["cut"] == cut1.fingerprint
    assert proof["lifecycles"][0]["uses"] == 1


def test_export_scopes_same_point_fork_cuts_with_exact_binding_joins(tmp_path):
    native = Session(_ForkAdapter(), {"value": 0, "clock": 0})
    root_cut = native.capture()
    left = native.fork(root_cut)
    right = native.fork(root_cut)
    left_base = left.capture()
    right_base = right.capture()
    left.apply(_fork_host_act(1))
    right.apply(_fork_host_act(2))
    left_cut = left.capture()
    right_cut = right.capture()
    assert left_cut.execution_point == right_cut.execution_point
    assert left_cut.fingerprint != right_cut.fingerprint

    left_binding = SymbolBinding.bind(
        left,
        "cot.left",
        (StateAddress("value"),),
        context="left-context",
        context_slots=("value",),
        evidence=(f"cas://sha256/{left_cut.fingerprint}",),
        claim="left fork carrier",
        status="observed",
    )
    right_binding = SymbolBinding.bind(
        right,
        "cot.right",
        (StateAddress("value"),),
        context="right-context",
        context_slots=("value",),
        evidence=(f"cas://sha256/{right_cut.fingerprint}",),
        claim="right fork carrier",
        status="observed",
    )
    left_id = left_binding.to_dict()["fingerprint"]
    right_id = right_binding.to_dict()["fingerprint"]
    project = Project(tmp_path / "project")
    with pytest.raises(PublicCotProjectError, match="multiple native cuts"):
        export_public_cot_project(
            project,
            "cot-forks",
            left,
            [root_cut, left_base, right_base, left_cut, right_cut],
            bindings=[left_binding, right_binding],
        )

    proof = export_public_cot_project(
        project,
        "cot-forks",
        left,
        [root_cut, left_base, right_base, left_cut, right_cut],
        bindings=[left_binding, right_binding],
        binding_cut_ids={left_id: [left_cut.fingerprint], right_id: [right_cut.fingerprint]},
    )
    assert proof["binding_cut_joins"] == {
        left_id: [left_cut.fingerprint],
        right_id: [right_cut.fingerprint],
    }
    by_symbol = {row["symbol"]: row for row in proof["bindings"]}
    assert by_symbol["cot.left"]["locations"][0]["cut"] == left_cut.fingerprint
    assert by_symbol["cot.right"]["locations"][0]["cut"] == right_cut.fingerprint
    verified = verify_public_cot_project(project, "cot-forks")
    assert verified["binding_cut_joins"] == proof["binding_cut_joins"]

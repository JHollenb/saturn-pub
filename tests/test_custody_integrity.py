import json

import pytest

from saturn_pub.core import Act, Adapter, Session, StateCut
from saturn_pub.investigation import Arm, Investigation
from saturn_pub.program import GenericJSONAdapter, Instruction, Program, ReplayBundle
from saturn_pub.store import LocalStore
from saturn_pub.values import canonical, describe, digest, legacy_describe_v1


def _cut(payload, *, parent=None):
    return StateCut(
        "json-v1",
        {"family": "json"},
        "step",
        parent,
        payload,
        {"family": "json", "index": 0},
        {"slots": sorted(payload)},
    )


def test_typed_descriptors_distinguish_sequences_scalars_and_tensors():
    assert describe([1]) != describe((1,))
    assert describe(True) != describe(1)
    torch = pytest.importorskip("torch")
    tensor_descriptor = describe(torch.tensor(1))
    assert tensor_descriptor["kind"] == "tensor"
    assert tensor_descriptor != describe(1)


def test_changed_callable_cannot_reuse_registered_operation_metadata():
    def first(inputs):
        return {"value": inputs["value"] + 1}

    def changed(inputs):
        return {"value": inputs["value"] + 2}

    Act.register_operation("tests.increment", "1", first, pure=True)
    bound = Act.registered(
        "increment",
        ("value",),
        ("value",),
        operation_id="tests.increment",
        operation_version="1",
    )
    bound.require_reproducible()
    with pytest.raises(ValueError, match="different implementation"):
        Act.register_operation("tests.increment", "1", changed, pure=True)
    with pytest.raises(ValueError, match="digest mismatch"):
        Act.registered(
            "increment",
            ("value",),
            ("value",),
            operation_id="tests.increment",
            operation_version="1",
            implementation=changed,
        )


def test_operation_code_identity_ignores_checkout_path_and_source_lines():
    def compile_operation(filename, prefix, expression):
        namespace = {"__name__": "portable_operations"}
        source = prefix + "def operation(inputs):\n    return {'value': " + expression + "}\n"
        exec(compile(source, filename, "exec"), namespace)
        return namespace["operation"]

    first = compile_operation(
        "/checkout-a/package/ops.py", "", "(lambda item: item + 1)(inputs['value'])"
    )
    relocated = compile_operation(
        "/wheel/site-packages/package/ops.py",
        "\n\n\n",
        "(lambda item: item + 1)(inputs['value'])",
    )
    changed = compile_operation(
        "/wheel/site-packages/package/ops.py",
        "\n\n\n",
        "(lambda item: item + 2)(inputs['value'])",
    )
    first_digest = Act.register_operation("tests.portable-operation", "1", first, pure=True)
    assert (
        Act.register_operation("tests.portable-operation", "1", relocated, pure=True)
        == first_digest
    )
    with pytest.raises(ValueError, match="different implementation"):
        Act.register_operation("tests.portable-operation", "1", changed, pure=True)


def test_builtin_identity_and_masked_patch_are_stable():
    assert Act.add("value", 1).implementation_digest == Act.add("value", 9).implementation_digest
    torch = pytest.importorskip("torch")
    current = torch.tensor([1.0, 2.0, 3.0])
    expected = digest(describe(current))
    act = Act.patch("value", 10.0, [True, False, True], expected_digest=expected)
    result = act.implementation({"value": current})["value"]
    assert torch.equal(result, torch.tensor([10.0, 2.0, 10.0]))
    assert act.preconditions == {"value": expected}
    with pytest.raises(ValueError, match="bool"):
        Act.patch("value", 0, [1, 0, 1]).implementation({"value": current})


def test_mutated_builtin_and_registered_callable_state_are_refused():
    class ScalarAdapter(Adapter):
        model_identity = "scalar-v1"
        execution = {"family": "scalar"}

        def boundary(self, state):
            return "step"

        def validate(self, state):
            if type(state.get("value")) is not int:
                raise ValueError("integer required")

        def addresses(self, state):
            return ("value",)

        def advance(self, state):
            return {"value": state["value"] + 1}

    builtin = Act.add("value", 2)
    assert builtin.manifest()["operation"]["implementation_state_digest"]
    builtin.implementation.value = 3
    with pytest.raises(ValueError, match="implementation state changed"):
        builtin.verify_implementation()
    with pytest.raises(ValueError, match="implementation state changed"):
        builtin.require_reproducible()
    with pytest.raises(ValueError, match="implementation state changed"):
        Session(ScalarAdapter(), {"value": 0}).apply(builtin)

    class StatefulIncrement:
        def __init__(self):
            self.amount = 1

        def __call__(self, inputs):
            return {"value": inputs["value"] + self.amount}

    implementation = StatefulIncrement()
    Act.register_operation("tests.stateful-increment", "1", implementation, pure=True)
    registered = Act.registered(
        "stateful-increment",
        ("value",),
        ("value",),
        operation_id="tests.stateful-increment",
        operation_version="1",
    )
    implementation.amount = 2
    with pytest.raises(ValueError, match="implementation state changed"):
        registered.require_reproducible()
    with pytest.raises(ValueError, match="implementation state changed"):
        Session(ScalarAdapter(), {"value": 0}).apply(registered)


def test_safe_strided_snapshot_round_trip_restores_same_program(tmp_path):
    torch = pytest.importorskip("torch")

    class TensorAdapter(Adapter):
        model_identity = "tensor-layout-v1"
        execution = {"family": "tensor-layout"}

        def boundary(self, state):
            return "tensor-step"

        def validate(self, state):
            if set(state) != {"value"} or state["value"].shape != (4, 3):
                raise ValueError("invalid tensor state")

        def addresses(self, state):
            return ("value",)

        def advance(self, state):
            return {"value": state["value"] + 1}

    source = torch.arange(12).reshape(3, 4).transpose(0, 1)
    assert not source.is_contiguous()
    adapter = TensorAdapter()
    session = Session(adapter, {"value": source})
    cut = session.capture()
    assert cut.payload["value"].stride() == source.stride()
    descriptor = describe(cut.payload["value"])
    assert descriptor["layout"] == "strided-dense"
    assert descriptor["stride"] == list(source.stride())

    store = LocalStore(tmp_path)
    loaded = store.load(store.save(cut))
    assert loaded.fingerprint == cut.fingerprint
    assert loaded.payload["value"].stride() == source.stride()
    recipient = Session(adapter, {"value": torch.zeros((4, 3), dtype=source.dtype)})
    recipient.restore(loaded)
    assert torch.equal(recipient.read("value"), source)


def test_per_slot_cas_deduplicates_and_refuses_corrupt_hydration(tmp_path):
    store = LocalStore(tmp_path)
    parent = _cut({"unchanged": (1, 2), "changed": [3]})
    store.save(parent)
    child = _cut({"unchanged": (1, 2), "changed": [4]}, parent=parent.fingerprint)
    report = store.save_report(child)
    assert report["deduplicated_bytes"] > 0
    parent_record = json.loads((tmp_path / "cuts" / f"{parent.fingerprint}.json").read_text())
    child_record = json.loads((tmp_path / "cuts" / f"{child.fingerprint}.json").read_text())
    assert parent_record["slots"]["unchanged"] == child_record["slots"]["unchanged"]
    loaded = store.load(child.fingerprint)
    assert loaded.payload == child.payload
    page = tmp_path / "pages" / f"{child_record['slots']['changed']}.json"
    page.write_bytes(page.read_bytes() + b" ")
    with pytest.raises(ValueError, match="page digest"):
        store.load(child.fingerprint)


def test_legacy_reader_marks_historical_typing_non_authoritative(tmp_path):
    payload = {"items": (1, 2)}
    manifest = {
        "schema": "saturn-pub-statecut-v1",
        "model_identity": "legacy",
        "execution": {"family": "json"},
        "boundary": "step",
        "parent": None,
        "payload": legacy_describe_v1(payload),
    }
    identifier = digest(manifest)
    record = {
        "manifest": manifest,
        "fingerprint": identifier,
        "encoded": {
            "kind": "mapping",
            "items": {
                "items": {
                    "kind": "tuple",
                    "items": [
                        {"kind": "scalar", "value": 1},
                        {"kind": "scalar", "value": 2},
                    ],
                }
            },
        },
        "blob": None,
    }
    (tmp_path / "cuts").mkdir()
    (tmp_path / "cuts" / f"{identifier}.json").write_bytes(canonical(record))
    with pytest.warns(RuntimeWarning, match="non-authoritative"):
        migrated = LocalStore(tmp_path).load(identifier)
    assert migrated.fingerprint != identifier
    assert migrated.execution_point["authority"] == "non-authoritative-historical-v1"
    assert migrated.payload["items"] == (1, 2)


def test_replay_bundle_imports_data_only_after_generic_adapter_identity_check(tmp_path):
    specification = {
        "kind": "generic-json",
        "model_identity": "json-v1",
        "execution": {"family": "json"},
        "boundary": "step",
        "slots": ["value"],
        "advance": {"kind": "increment", "slot": "value", "amount": 1},
    }
    adapter = GenericJSONAdapter(specification)
    session = Session(adapter, {"value": 0})
    parent = session.capture()
    act = Act.add("value", 2)
    panel = Investigation("effect", (Arm("candidate", (act,)),), 1).run(
        session, lambda branch: {"value": branch.read("value")}, parent=parent
    )
    program = Program.compile(session, panel, "candidate", (Instruction(0, act),), context="json")
    source = LocalStore(tmp_path / "source")
    source.save(parent)
    bundle = ReplayBundle.create(
        program,
        source,
        branch_heads={"parent": parent.fingerprint},
        receipts=tuple(),
        adapter_factory=specification,
        include_data=True,
    )
    path = tmp_path / "bundle.json"
    bundle.save(path)
    reopened = ReplayBundle.load(path)
    destination = LocalStore(tmp_path / "destination")
    restored_adapter = reopened.restore(destination, allow_data=True)
    assert restored_adapter.model_identity == "json-v1"
    assert destination.load(parent.fingerprint).payload == {"value": 0}

    wrong = dict(specification)
    wrong["model_identity"] = "wrong"
    with pytest.raises(ValueError, match="identity differs"):
        ReplayBundle.create(
            program,
            source,
            branch_heads={"parent": parent.fingerprint},
            adapter_factory=wrong,
        )

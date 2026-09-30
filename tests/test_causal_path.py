import pytest

from saturn_pub import Adapter, Session
from saturn_pub.causal import CausalPath, PortObservation


class Counter(Adapter):
    model_identity = "causal-counter-v1"
    execution = {"family": "counter", "parity": "exact-integer"}

    def boundary(self, state):
        return f"counter:{state['value']}"

    def validate(self, state):
        if set(state) != {"value"} or type(state["value"]) is not int:
            raise ValueError("integer counter required")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        return {"value": state["value"] + 1}


def observation(kind, port, value, session, receipt, parent):
    return PortObservation.record(kind, port, value, session, receipt, parent=parent)


def test_causal_path_separates_effect_planes_and_emits_graph():
    session = Session(Counter(), {"value": 0})
    parent = session.capture().fingerprint
    receipt = session.step()
    rows = (
        observation("source", "input.value", 0, session, receipt, parent),
        observation("address", "counter.value", 0, session, receipt, parent),
        observation("writer", "increment", "+1", session, receipt, parent),
        observation("carrier", "state.value", 1, session, receipt, parent),
        observation("consumer", "counter.output", 1, session, receipt, parent),
        observation("first-divergence", "state.value", 1, session, receipt, parent),
        observation("repair", "counter.output", {"delta": 1}, session, receipt, parent),
        observation("collateral", "protected.value", {"delta": 0}, session, receipt, parent),
        observation("native-consumer", "counter.output", 1, session, receipt, parent),
    )
    path = CausalPath.from_observations(
        source="input.value",
        address="counter.value",
        carrier="state.value",
        writer="increment",
        consumer="counter.output",
        observations=rows,
    )

    assert path.first_divergence is rows[5]
    assert path.repair == (rows[6],)
    assert path.collateral == (rows[7],)
    assert path.native_consumer == (rows[8],)
    report = path.to_dict()
    assert report["effects"]["repair"] == [rows[6].fingerprint]
    assert report["observations"][0]["cut"]
    assert report["observations"][0]["receipt"] == receipt.fingerprint
    assert report["interpretation"].startswith("observations-only")
    assert [edge["relation"] for edge in report["edges"]] == [
        "routes-to",
        "selects-writer",
        "writes",
        "consumed-by",
    ]


def test_causal_path_rejects_unsupported_model_parent_and_clock():
    session = Session(Counter(), {"value": 0})
    parent = session.capture().fingerprint
    receipt = session.step()
    row = observation("source", "input.value", 0, session, receipt, parent)
    base = dict(
        source="input.value",
        address="counter.value",
        carrier="state.value",
        writer="increment",
        consumer="counter.output",
        model_identity=row.model_identity,
        parent=parent,
        clocks=(row.clock,),
    )
    CausalPath(**base, observations=(row,))

    bad_model = PortObservation(
        row.kind,
        row.port,
        row.value,
        "other-model",
        row.parent,
        row.clock,
        row.cut,
        row.receipt,
    )
    with pytest.raises(ValueError, match="model"):
        CausalPath(**base, observations=(bad_model,))
    with pytest.raises(ValueError, match="clock"):
        CausalPath(**{**base, "clocks": ({"logical_step": 999},)}, observations=(row,))

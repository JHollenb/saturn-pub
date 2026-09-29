import pytest

from saturn_pub import (
    Act,
    Adapter,
    ExecutionPoint,
    Session,
    SlotSpec,
    StateAddress,
    SurfaceManifest,
)


class ClockAdapter(Adapter):
    model_identity = "clock-v1"
    execution = {"family": "clock", "adapter": "json-native-v1"}

    def __init__(self):
        self.external = 0

    def boundary(self, state):
        return "tick"

    def point(self, state):
        return ExecutionPoint("clock", state["clock"], "tick", next_operation="increment")

    def surface(self, state):
        return SurfaceManifest(
            (
                SlotSpec("value", writable=True),
                SlotSpec("clock"),
                SlotSpec("context", role="source"),
            ),
            "clock-v1",
            "counter",
        )

    def addresses(self, state):
        return ("value", "clock", "context")

    def validate(self, state):
        if state["value"] < 0:
            raise ValueError("negative value")

    def capture_external(self):
        return self.external

    def restore_external(self, snapshot):
        self.external = snapshot

    def advance(self, state):
        self.external += 1
        if state["value"] == 2:
            raise RuntimeError("consumer fault")
        return {**state, "value": state["value"] + 1, "clock": state["clock"] + 1}


def make_session():
    return Session(ClockAdapter(), {"value": 0, "clock": 0, "context": [4, 5]})


def test_typed_port_clock_schema_role_and_copy():
    session = make_session()
    cut = session.capture()
    point = session.adapter.point(session._state)
    address = StateAddress("context", (1,), "source", point.fingerprint, "clock-v1")
    assert session.read_port(address) == 5
    session.step()
    with pytest.raises(ValueError, match="clock"):
        session.read_port(address)
    with pytest.raises(ValueError, match="role"):
        session.read_port(StateAddress("context", role="writer"))
    with pytest.raises(ValueError, match="schema"):
        session.read_port(StateAddress("context", state_schema="wrong"))
    assert cut.execution_point["logical_step"] == 0
    assert session.inspect().execution_point["logical_step"] == 1


def test_readonly_and_preservation_are_atomic():
    session = make_session()
    parent = session.capture()
    with pytest.raises(ValueError, match="read-only"):
        session.apply(Act.zero("clock"))
    session.adapter.write = lambda state, writes, invalidates: {**state, **writes, "clock": 8}
    with pytest.raises(ValueError, match="preserved"):
        session.apply(Act.add("value", 2))
    assert session.capture().fingerprint == parent.fingerprint
    assert len(session.attempts) == 2


def test_external_closure_failure_and_replay_cursor():
    session = make_session()
    parent = session.capture()
    with pytest.raises(RuntimeError):
        session.continue_(3)
    assert session.capture().fingerprint == parent.fingerprint
    assert session.adapter.external == 0
    candidate = session.replay(parent, steps=2)
    assert candidate.read("value") == 2
    assert session.capture().fingerprint == parent.fingerprint
    assert session.adapter.external == 0
    assert candidate.read("__saturn_external__") == 2
    session.restore(parent)
    assert session.adapter.external == 0
    assert len(session.attempts) == 1  # physical attempts survive rewind


def test_closure_refuses_missing_or_extra_state():
    session = make_session()
    with pytest.raises(ValueError, match="closure"):
        Session(session.adapter, {**session._state, "undeclared": 3})


def test_external_closure_durable_reopen(tmp_path):
    from saturn_pub.store import LocalStore

    session = make_session()
    session.continue_(2)
    store = LocalStore(tmp_path)
    cut = store.load(store.save(session.capture()))
    fresh = Session.from_cut(ClockAdapter(), cut)
    assert fresh.read("__saturn_external__") == 2
    assert fresh.capture().fingerprint == cut.fingerprint
    parent = fresh.capture()
    with pytest.raises(RuntimeError):
        fresh.continue_()
    assert fresh.capture().fingerprint == parent.fingerprint
    assert fresh.read("__saturn_external__") == 2


@pytest.mark.parametrize("operation", ["restore", "commit"])
def test_external_closure_rolls_back_if_publication_fails(operation):
    session = make_session()
    parent = session.capture()
    candidate = session.fork(parent)
    candidate.continue_()
    cut = candidate.capture()
    original = session._cut
    calls = 0

    def fail_publication(state, ancestry):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("publication fault")
        return original(state, ancestry)

    session._cut = fail_publication
    with pytest.raises(RuntimeError, match="publication fault"):
        if operation == "restore":
            session.restore(cut)
        else:
            session.commit(candidate)
    assert session.adapter.external == 0
    assert session.capture().fingerprint == parent.fingerprint
    assert session.attempts[-1].to_dict()["operation"] == operation

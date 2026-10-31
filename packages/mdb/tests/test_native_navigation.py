import pytest
from saturn_pub import Act, Adapter, Session
from saturn_pub.contracts import ExecutionPoint

from mdb.native_navigation import NativeCutNavigator


class Counter(Adapter):
    model_identity = "navigation-counter-v1"
    execution = {"family": "counter"}

    def boundary(self, state):
        return "step:" + str(state["value"])

    def point(self, state):
        return ExecutionPoint(family="counter", logical_step=state["value"], phase="counter")

    def validate(self, state):
        if type(state["value"]) is not int or state["value"] < 0:
            raise ValueError("nonnegative integer required")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        return {"value": state["value"] + 1}


def test_receipt_navigation_does_not_add_per_step_captures(monkeypatch):
    session = Session(Counter(), {"value": 0})
    receipts = []
    nav = NativeCutNavigator(session, max_steps=4, on_receipt=receipts.append)
    original = session.capture
    calls = []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(session, "capture", capture)
    session.continue_()
    direct_calls = len(calls)
    nav.capture()  # resynchronize after deliberately using the owned session
    calls.clear()
    nav.until(logical_step=3, phase="counter")
    assert len(calls) == 2 * direct_calls
    before = len(calls)
    for _ in range(100):
        assert nav.point["logical_step"] == 3
    assert len(calls) == before
    assert len(receipts) == nav.steps == 2


def test_fork_apply_and_finite_missing_breakpoint():
    nav = NativeCutNavigator(Session(Counter(), {"value": 0}), max_steps=3)
    cut = nav.capture()
    branch = nav.fork(cut)
    branch.apply(Act.replace("value", 5))
    assert branch.point["logical_step"] == 5
    assert nav.point["logical_step"] == 0
    with pytest.raises(ValueError, match="budget"):
        branch.until(logical_step=99)
    assert branch.steps == 3
    with pytest.raises(ValueError, match="selector"):
        nav.until(typo=3)
    with pytest.raises(ValueError):
        nav.step(True)


def test_failed_native_attempt_consumes_budget_without_advancing_cursor():
    class Broken(Counter):
        def advance(self, state):
            raise RuntimeError("injected native failure")

    nav = NativeCutNavigator(Session(Broken(), {"value": 0}), max_steps=1)
    before = nav.capture().fingerprint
    with pytest.raises(RuntimeError, match="injected"):
        nav.step()
    assert nav.steps == 1 and nav.point["logical_step"] == 0
    assert nav.capture().fingerprint == before
    with pytest.raises(ValueError, match="budget"):
        nav.step()

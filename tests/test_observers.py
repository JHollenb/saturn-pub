import math

import pytest
import torch

from saturn_pub import Act, Adapter, Session
from saturn_pub.cli import Debugger
from saturn_pub.observers import Observer


class Counter(Adapter):
    model_identity = "observer-counter-v1"
    execution = {"family": "counter", "parity": "exact-integer"}

    def boundary(self, state):
        return f"counter:{state['value']}"

    def validate(self, state):
        if set(state) != {"value"} or not isinstance(state["value"], (int, float)):
            raise ValueError("numeric counter required")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        return {"value": state["value"] + 1}


def test_observer_retains_safe_point_and_tracks_branches_independently():
    root = Session(Counter(), {"value": 0})
    left = root.fork()
    right = root.fork()
    observer = Observer()
    observer.add("value", "change")
    observer.sync(left, "left")
    observer.sync(right, "right")

    left_receipt = left.step()
    event = observer.evaluate(left, "left", receipt=left_receipt)[0]
    assert event.current == 1
    assert event.cut.payload["value"] == 1
    assert event.cut in left.history
    assert event.to_dict()["authority"] == "diagnostic-only"

    # The right baseline remains at its own cursor.
    right_receipt = right.step()
    assert observer.evaluate(right, "right", receipt=right_receipt)[0].current == 1


def test_threshold_and_nonfinite_predicates_are_diagnostics():
    class TensorCounter(Counter):
        def validate(self, state):
            if set(state) != {"value"} or not isinstance(state["value"], torch.Tensor):
                raise ValueError("tensor counter required")

    session = Session(TensorCounter(), {"value": torch.tensor([1.0])})
    observer = Observer()
    observer.add("value", "gt", 1.5)
    observer.add("value", "lt", -5)
    observer.add("value", "nonfinite")
    observer.sync(session, "root")
    event = observer.evaluate(session, "root")
    assert event == ()

    session.apply(Act.replace("value", torch.tensor([2.0])))
    assert [row.watchpoint.predicate for row in observer.evaluate(session, "root")] == ["gt"]
    session.apply(Act.replace("value", torch.tensor([math.inf])))
    assert [row.watchpoint.predicate for row in observer.evaluate(session, "root")] == [
        "gt",
        "nonfinite",
    ]


def test_debugger_watches_stop_microstep_and_rewind_resyncs():
    debugger = Debugger(Session(Counter(), {"value": 0}))
    debugger.execute("capture parent")
    watch = debugger.execute("watch value change")
    stopped = debugger.execute("continue")
    assert stopped["stop"] == "watchpoint"
    assert stopped["steps"] == 1
    assert stopped["events"][0]["watchpoint"]["identifier"] == watch["identifier"]
    assert stopped["events"][0]["cut"] in debugger.cuts

    assert debugger.execute("restore parent")["verified_exact"]
    # Restore seeds a fresh value=0 baseline, so the next 0→1 change is unambiguous.
    assert debugger.execute("step")["stop"] == "watchpoint"
    debugger.execute(f"unwatch {watch['identifier']}")
    assert debugger.execute("watchpoints") == {"watchpoints": []}


def test_watch_validation():
    observer = Observer()
    with pytest.raises(ValueError, match="threshold"):
        observer.add("value", "gt")
    with pytest.raises(ValueError, match="unsupported"):
        observer.add("value", "equal", 1)


def test_debugger_replay_diff_trace_and_generic_adapter_load(tmp_path):
    debugger = Debugger(Session(Counter(), {"value": 0}))
    captured = debugger.execute("capture parent")
    before = debugger.session.capture(retain=False).fingerprint
    replayed = debugger.execute("replay parent 2 future")
    assert replayed["frame"]["boundary"] == "counter:2"
    assert debugger.branches["future"].read("value") == 2
    assert debugger.active == "root"
    assert debugger.session.capture(retain=False).fingerprint == before
    assert not debugger.execute("diff future")["equal_payload"]
    assert len(debugger.execute("trace 2")["trace"]) == 2

    assert debugger.execute(f"save {tmp_path} parent")["saved"] == captured["fingerprint"]
    loaded = debugger.execute(f"load {tmp_path} {captured['fingerprint']} durable-parent")
    assert loaded["fingerprint"] == captured["fingerprint"]


def test_explicit_continue_budget_ignores_boundary_breakpoints():
    debugger = Debugger(Session(Counter(), {"value": 0}))
    debugger.execute("break counter:1")
    result = debugger.execute("continue 2")
    assert result["stop"] == "step-limit"
    assert result["steps"] == 2
    assert result["frame"]["boundary"] == "counter:2"

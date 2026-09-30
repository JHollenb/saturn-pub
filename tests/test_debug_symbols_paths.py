import json
from dataclasses import replace

import pytest

from saturn_pub import (
    Act,
    Adapter,
    ExecutionPoint,
    Session,
    SlotSpec,
    StateAddress,
    SurfaceManifest,
    TransitionSpec,
)
from saturn_pub.cli import Debugger
from saturn_pub.paths import (
    PathExecutionError,
    PathSchedule,
    TimedAct,
    Trajectory,
    compare_trajectories,
)
from saturn_pub.symbols import SymbolBinding, SymbolTable, backtrace


class Repair(Adapter):
    model_identity = "supplied-integer-repair-v1"
    execution = {"family": "controlled-repair", "program": "integer-copy"}

    def point(self, state):
        return ExecutionPoint(
            "controlled-repair",
            0,
            "run",
            index=state["cursor"],
            next_operation=self.boundary(state),
        )

    def boundary(self, state):
        return ("first", "repair", "emit", "halted")[state["cursor"]]

    def surface(self, state):
        return SurfaceManifest(
            (
                SlotSpec("source", role="source"),
                SlotSpec("value", writable=True, role="carrier"),
                SlotSpec("output", role="output"),
                SlotSpec("cursor", role="clock"),
            ),
            "repair-v1",
            "emit",
        )

    def transition(self, state):
        return TransitionSpec(
            self.boundary(state),
            ("source",) if state["cursor"] < 2 else ("value",),
            ("value", "cursor") if state["cursor"] < 2 else ("output", "cursor"),
            footprint="declared",
        )

    def addresses(self, state):
        return tuple(state)

    def validate(self, state):
        if (
            set(state) != {"source", "value", "output", "cursor"}
            or any(type(v) is not int for v in state.values())
            or not 0 <= state["cursor"] <= 3
        ):
            raise ValueError("invalid repair state")

    def advance(self, state):
        if state["cursor"] == 3:
            raise ValueError("halted")
        result = dict(state)
        result["value" if state["cursor"] < 2 else "output"] = (
            state["source"] if state["cursor"] < 2 else state["value"]
        )
        result["cursor"] += 1
        return result

    def session(self):
        return Session(self, {"source": 7, "value": 0, "output": 0, "cursor": 0})


def bind(session):
    return SymbolBinding.bind(
        session,
        "target",
        (StateAddress("value", role="carrier"),),
        context="supplied-v1",
        context_slots=("source",),
        evidence=("fixture",),
        claim="supplied role",
    )


def test_symbols_reject_stale_clock_input_execution_and_ambiguous_support(tmp_path):
    session = Repair().session()
    binding = bind(session)
    binding.resolve(session, context="supplied-v1")
    with pytest.raises(ValueError, match="context"):
        binding.resolve(session, context="other")
    other = Session(session.adapter, {"source": 8, "value": 0, "output": 0, "cursor": 0})
    with pytest.raises(ValueError, match="input context"):
        binding.resolve(other, context="supplied-v1")
    with pytest.raises(ValueError, match="found 2"):
        SymbolTable((binding, binding)).resolve(session, "target", context="supplied-v1")
    cut = session.capture()
    path = tmp_path / "info.json"
    SymbolTable((binding,)).save(path, cut)
    loaded = SymbolTable.load(path, cut)
    assert loaded.resolve(session, "target", context="supplied-v1") == binding
    session.step()
    with pytest.raises(ValueError, match="clock"):
        binding.resolve(session, context="supplied-v1")
    with pytest.raises(ValueError, match="cut reference"):
        SymbolTable.load(path, session.capture())
    data = json.loads(path.read_text())
    data["bindings"][0]["claim"] = "unearned"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seal"):
        SymbolTable.load(path, cut)
    with pytest.raises(ValueError, match="execution"):
        replace(binding, execution="changed").resolve(other, context="supplied-v1")


def test_trace_measures_recovery_without_equating_ancestry_or_terminal_claims():
    session = Repair().session()
    parent = session.capture()
    native = Trajectory.run(
        session, name="native", parent=parent, steps=3, ports=("value", "output")
    )
    candidate = Trajectory.run(
        session,
        name="early",
        parent=parent,
        steps=3,
        ports=("value", "output"),
        schedule=PathSchedule((TimedAct(1, Act.zero("value")),)),
    )
    diff = compare_trajectories(native, candidate)
    assert diff["first_recorded_divergence"]["offset"] == 1
    assert diff["observable_windows"]["value"] == [{"diverged_at": 1, "recovered_at": 2}]
    assert diff["terminal_status"] == "not-assessed"
    assert native.points[-1].cut.fingerprint != candidate.points[-1].cut.fingerprint
    assert session.capture(retain=False).fingerprint == parent.fingerprint
    later = Trajectory.run(
        session,
        name="late",
        parent=parent,
        steps=3,
        ports=("value", "output"),
        schedule=PathSchedule((TimedAct(2, Act.zero("value")),)),
    )
    assert compare_trajectories(native, later)["observable_windows"]["output"] == [
        {"diverged_at": 3, "recovered_at": None}
    ]


def test_trace_rejects_wrong_parent_clock_and_modified_observations():
    session = Repair().session()
    parent = session.capture()
    trace = Trajectory.run(session, name="a", parent=parent, steps=3, ports=("value",))
    session.step()
    other = replace(trace, parent=session.capture())
    with pytest.raises(ValueError, match="common parent"):
        compare_trajectories(trace, other)
    bad_point = replace(trace.points[1], offset=99)
    other = replace(trace, points=(trace.points[0], bad_point, *trace.points[2:]))
    with pytest.raises(ValueError, match="not aligned"):
        compare_trajectories(trace, other)
    bad_point = replace(trace.points[1], values={"value": 42})
    other = replace(trace, points=(trace.points[0], bad_point, *trace.points[2:]))
    with pytest.raises(ValueError, match="observations"):
        compare_trajectories(trace, other)


def test_failed_arm_preserves_completed_events_and_parent():
    session = Repair().session()
    parent = session.capture()
    with pytest.raises(PathExecutionError) as error:
        Trajectory.run(
            session,
            name="refused",
            parent=parent,
            steps=3,
            ports=("value",),
            schedule=PathSchedule((TimedAct(2, Act.zero("source")),)),
        )
    assert len(error.value.points) == 2
    assert len(error.value.events) == 2
    assert error.value.cut.payload["cursor"] == 2
    assert error.value.attempts[-1].to_dict()["status"] == "refused"
    assert session.capture(retain=False).fingerprint == parent.fingerprint


def test_debugger_reuses_symbol_watch_and_provenance_stops_at_restore():
    session = Repair().session()
    parent = session.capture()
    table = SymbolTable()
    for cursor in range(4):
        table.bindings.append(bind(session))
        if cursor < 3:
            session.step()
    session.restore(parent)
    debugger = Debugger(session)
    debugger.symbols, debugger.symbol_context = table, "supplied-v1"
    assert debugger.execute("info steps")["next_transition"]["operation"] == "first"
    assert debugger.execute("info symbols")["structural"][0]["status"] == "adapter-declared"
    assert debugger.execute("resolve target")["status"] == "supplied"
    debugger.execute("watch symbol:target change")
    result = debugger.execute("continue")
    assert result["stop"] == "watchpoint"
    assert result["events"][0]["cut"] in debugger.cuts
    provenance = debugger.execute("backtrace value")
    assert provenance["writers"][0]["operation"] == "first"
    assert provenance["writers"][-1]["status"] == "ancestry-boundary"
    assert backtrace(session, "value")["authority"].endswith("not-semantic-proof")


def test_native_footprint_violation_is_atomic_and_evidenced():
    class Broken(Repair):
        def transition(self, state):
            return TransitionSpec("bad-write", ("source",), ("cursor",), footprint="declared")

    session = Broken().session()
    parent = session.capture()
    with pytest.raises(ValueError, match="undeclared slot: value"):
        session.step()
    assert session.capture().fingerprint == parent.fingerprint
    assert session.attempts[-1].to_dict()["status"] == "refused"


def test_trace_rejects_forged_ancestry_and_schedule():
    session = Repair().session()
    parent = session.capture()
    native = Trajectory.run(session, name="native", parent=parent, steps=3, ports=("value",))
    forged = replace(
        native, events=((native.events[0][0], native.events[-1][1]), *native.events[1:])
    )
    with pytest.raises(ValueError, match="ancestry"):
        forged.to_dict()
    forged = replace(native, schedule=PathSchedule((TimedAct(1, Act.zero("value")),)))
    with pytest.raises(ValueError, match="schedule"):
        forged.to_dict()


def test_abstention_and_missing_symbol_support_do_not_invent_semantics():
    session = Repair().session()
    binding = bind(session)
    abstained = replace(binding, status="abstained", locations=(), abstain_reason="no support")
    with pytest.raises(ValueError, match="abstained"):
        abstained.resolve(session, context="supplied-v1")
    with pytest.raises(ValueError, match="confidence"):
        replace(binding, confidence=float("nan"))
    debugger = Debugger(session)
    debugger.symbols = SymbolTable((binding,))
    debugger.symbol_context = "supplied-v1"
    debugger.execute("watch symbol:target change")
    with pytest.raises(ValueError, match="found 0"):
        debugger.execute("continue")
    assert session.read("cursor") == 1
    assert session.receipts[-1].to_dict()["operation"] == "continue"

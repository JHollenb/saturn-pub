from dataclasses import replace

import pytest

from saturn_pub import Act, Adapter, Session, StateAddress
from saturn_pub.cli import Debugger
from saturn_pub.observers import ObservationError, Observer
from saturn_pub.paths import PathExecutionError, PathSchedule, TimedAct, Trajectory
from saturn_pub.symbols import SymbolBinding, SymbolTable, backtrace


class Value(Adapter):
    model_identity = "audit-value-v1"
    execution = {"family": "audit-value"}

    def boundary(self, state):
        return "value"

    def validate(self, state):
        if set(state) != {"value"}:
            raise ValueError("invalid state")

    def addresses(self, state):
        return ("value",)

    def advance(self, state):
        return dict(state)


@pytest.mark.parametrize("before,after", [([1], (1,)), (None, 1), (True, 1), (1, 1.0)])
def test_change_watch_preserves_types_and_none_baselines(before, after):
    session = Session(Value(), {"value": before})
    observer = Observer()
    observer.add("value", "change")
    observer.sync(session, "root")
    receipt = session.apply(Act.replace("value", after))
    assert len(observer.evaluate(session, "root", receipt=receipt)) == 1


def test_failed_watch_registration_is_atomic():
    debugger = Debugger(Session(Value(), {"value": 0}))
    watch = debugger.execute("watch value change")
    with pytest.raises(ValueError, match="unsupported address"):
        debugger.execute("watch missing change")
    assert [row.identifier for row in debugger.observer.watchpoints] == [watch["identifier"]]
    assert debugger.execute("replace value 1")["watch_events"]


def test_foreign_receipt_and_mutated_event_refused_without_losing_baseline():
    session = Session(Value(), {"value": [1]})
    observer = Observer()
    observer.add("value", "change")
    observer.sync(session, "root")
    receipt = session.apply(Act.replace("value", [2]))
    wrong = Session(Value(), {"value": [99]}).step()
    with pytest.raises(ValueError, match="does not produce"):
        observer.evaluate(session, "root", receipt=wrong)
    event = observer.evaluate(session, "root", receipt=receipt)[0]
    event.current.append(3)
    with pytest.raises(ValueError, match="content changed"):
        event.to_dict()


def test_broken_symbol_watch_retains_other_fires_and_exact_failure_cut():
    from test_debug_symbols_paths import Repair, bind

    session = Repair().session()
    debugger = Debugger(session)
    debugger.symbols, debugger.symbol_context = SymbolTable((bind(session),)), "supplied-v1"
    debugger.execute("watch value change")
    invalid_after_step = debugger.execute("watch symbol:target change")
    with pytest.raises(ObservationError) as failure:
        debugger.execute("continue")
    assert len(failure.value.events) == len(debugger.watch_events) == 1
    cut = failure.value.cut
    assert cut.payload["value"] == 7
    assert cut.fingerprint in debugger.cuts
    assert debugger.watch_errors[-1]["cut"] == cut.fingerprint
    debugger.execute(f"unwatch {invalid_after_step['identifier']}")
    assert debugger.observer.evaluate(session, "root") == ()


def test_backtrace_batch_does_not_attribute_earlier_points_to_final_cut():
    from test_debug_symbols_paths import Repair

    session = Repair().session()
    receipt = session.continue_(3)
    writers = backtrace(session, "output")["writers"]
    # The repair assignment reads source, not the overwritten first writer's value.
    assert [row["operation"] for row in writers] == ["emit", "repair"]
    assert writers[0]["cut"] == receipt.to_dict()["result"]
    assert all(row["cut"] is None and row["cut_scope"] == "point-only" for row in writers[1:])


def test_macro_token_backtrace_expands_micro_receipts():
    from saturn_pub.adapters.qwen import QwenAdapter

    session = QwenAdapter.tiny(granularity="operation").session([5, 7, 11])
    macro = session.step(granularity="token")
    writers = backtrace(session, "tokens")["writers"]
    assert writers[0]["operation"] == "token_commit"
    assert all(row.get("enclosing_receipt") == macro.fingerprint for row in writers)
    assert all(row["status"] == "declared-dependency" for row in writers)


def test_false_offsets_and_duplicate_native_point_refused():
    from test_debug_symbols_paths import Repair

    session = Repair().session()
    trace = Trajectory.run(
        session, name="native", parent=session.capture(), steps=3, ports=("value",)
    )
    forged = replace(
        trace, points=tuple(replace(point, offset=point.offset + 100) for point in trace.points)
    )
    with pytest.raises(ValueError, match="consecutive"):
        forged.verify()
    forged = replace(trace, points=(trace.points[0], trace.points[0], *trace.points[1:]))
    with pytest.raises(ValueError, match="consecutive"):
        forged.verify()


def test_qualified_address_is_not_silently_rebound():
    from test_debug_symbols_paths import Repair

    session = Repair().session()
    with pytest.raises(ValueError, match="clock mismatch"):
        SymbolBinding.bind(
            session,
            "target",
            (StateAddress("value", execution_point="stale"),),
            context="fixture",
            context_slots=("source",),
            evidence=("fixture",),
            claim="fixture",
        )


def test_failed_evaluator_keeps_completed_state_and_events():
    from test_debug_symbols_paths import Repair

    session = Repair().session()
    parent = session.capture()

    def evaluator(branch):
        if branch.read("cursor") == 2:
            raise RuntimeError("instrument broken")
        return {"value": branch.read("value")}

    with pytest.raises(PathExecutionError) as failure:
        Trajectory.run(
            session,
            name="broken-evaluator",
            parent=parent,
            steps=3,
            ports=("value",),
            schedule=PathSchedule((TimedAct(1, Act.zero("value")),)),
            evaluator=evaluator,
        )
    assert len(failure.value.events) == 3
    assert len(failure.value.points) == 2
    assert failure.value.cut.payload["cursor"] == 2
    assert session.capture(retain=False).fingerprint == parent.fingerprint

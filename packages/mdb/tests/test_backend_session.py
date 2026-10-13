"""Public backend construction, capability refusals, and native debugger reuse."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest

from mdb import BenchError, BenchSession, ForwardResult, ModelRunner, RunnerCapabilities
from mdb.backends import register_backend
from mdb.debugger_adapters import AdapterCapabilities, CallbackDebuggerAdapter


class ArithmeticRunner(ModelRunner):
    def __init__(self):
        self.calls = 0
        self.closed = False

    def capabilities(self):
        return RunnerCapabilities("arithmetic", ("hidden",), ("hidden",), True)

    def forward(self, inputs, *, capture=(), interventions=None, **kwargs):
        self.calls += 1
        value = inputs * 2
        if "hidden" in (interventions or {}):
            value = interventions["hidden"](value)
        return ForwardResult(value + 1, {"hidden": value} if capture else {})

    def close(self):
        self.closed = True


def test_custom_runner_executes_and_refuses_before_forward():
    runner = ArithmeticRunner()
    with BenchSession.from_backend(runner, identity={"model_id": "arithmetic"}) as session:
        result = session.forward(
            3, capture=["hidden"], interventions={"hidden": lambda value: value + 5}
        )
        assert result.output == 12 and session.inspect("hidden") == 11
        for options in (
            {"capture": ["missing"]},
            {"interventions": {"hidden": {"op": "suppress"}}},
            {"capture": "hidden"},
        ):
            with pytest.raises(BenchError):
                session.forward(3, **options)
        assert runner.calls == 1
        assert "checkpoint" not in session.capabilities()["operations"]
        with pytest.raises(BenchError, match="stateful"):
            session.checkpoint()
        with pytest.raises(BenchError, match="stateful"):
            session.debugger()
    assert not runner.closed  # Caller ownership is explicit.
    with pytest.raises(BenchError, match="closed"):
        session.forward(1)


def test_registration_identity_ownership_and_failed_factory_cleanup():
    name = "arithmetic_" + uuid4().hex
    made = []

    def factory():
        runner = ArithmeticRunner()
        made.append(runner)
        return runner

    register_backend(name, factory)
    with pytest.raises(ValueError, match="already registered"):
        register_backend(name, factory)
    identity = {"model_id": "counter", "details": {"version": 1}}
    with BenchSession.from_backend(name, identity=identity) as session:
        identity["details"]["version"] = 2
        assert session.capabilities()["model_identity"]["details"]["version"] == 1
        assert session.forward(2).output == 5
    assert made[0].closed
    with pytest.raises(BenchError, match="debugger_adapter"):
        BenchSession.from_backend(name, identity={"model_id": "counter"}, debugger_adapter=object())
    assert made[1].closed


def test_missing_capture_is_not_reported_as_success():
    class Broken(ArithmeticRunner):
        def forward(self, *args, **kwargs):
            return ForwardResult(1)

    session = BenchSession.from_backend(Broken(), identity={"model_id": "broken"})
    with pytest.raises(BenchError, match="did not fire"):
        session.forward(1, capture=["hidden"])
    assert session.inspect()["forward_calls"] == 0


def test_bare_import_and_execution_with_every_framework_blocked():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'transformers', 'transformer_lens',
                                     'diffusers', 'mrun', 'saturn', 'saturn_pub'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
from mdb import (BenchSession, ModelRunner, RunnerCapabilities, ForwardResult,
                 TransformerLensRunner, TransformersRunner, available_backends)
class Runner(ModelRunner):
    def capabilities(self): return RunnerCapabilities('json')
    def forward(self, inputs, **kwargs): return ForwardResult(inputs + 1)
with BenchSession.from_backend(Runner(), identity={'model_id':'json'}) as session:
    assert session.forward(2).output == 3
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def counter_adapter():
    state = {"value": 0, "cut": "cut-0", "pending": None}
    cuts = {"cut-0": 0}

    def checkpoint(name=None):
        ref = f"cut-{len(cuts)}"
        cuts[ref] = state["value"]
        state["cut"] = ref
        return {"cut_id": ref, "name": name}

    def restore(ref):
        state.update(value=cuts[ref], cut=ref)
        return {"cut_id": ref, "exact": True}

    def preview(act):
        state["pending"] = (act.act_id, act.value)
        return {"act_id": act.act_id, "preview": True}

    def commit(ref):
        assert ref == state["pending"][0]
        state["value"] = state["pending"][1]
        state["pending"] = None
        return {"act_id": ref, "committed": True, **checkpoint()}

    def abort(ref):
        assert ref == state["pending"][0]
        state["pending"] = None
        return {"act_id": ref, "aborted": True}

    def step(count):
        state["value"] += count
        return {"steps": count, **checkpoint()}

    caps = AdapterCapabilities(
        family="counter",
        boundaries=("step",),
        inspectable_slots=("value",),
        writable_slots=("value",),
        mutable_closure=("value",),
        exact_replay=True,
        state_schema_id="test-counter-v1",
        backend="json",
    )
    return CallbackDebuggerAdapter(
        caps,
        status=lambda: {"cut_id": state["cut"]},
        inspect=lambda path: {"value": state["value"]},
        checkpoint=checkpoint,
        restore=restore,
        preview_act=preview,
        commit_act=commit,
        abort_act=abort,
        replay=restore,
        step=step,
        continue_run=lambda: {"state": "running"},
        resolve_value=lambda path: state["value"],
    )


def test_existing_adapter_uses_local_debugger_and_real_state():
    session = BenchSession.from_backend(counter_adapter(), identity={"model_id": "counter"})
    debugger = session.debugger()
    parent = debugger.checkpoint("parent")["cut_id"]
    debugger.pause()
    session.safe_point("step", 0)
    debugger.set_value("value", 9)
    assert debugger.commit()["committed"]
    assert session.resolve_value("value") == 9
    assert session.restore(parent)["exact"] and session.resolve_value("value") == 0
    session.step(2)
    assert session.resolve_value("value") == 2
    session.close()
    with pytest.raises(BenchError, match="closed"):
        debugger.status()


def test_debugger_and_runner_share_execution_serialization():
    entered, release, requested, inspected = Event(), Event(), Event(), Event()

    class Blocking(ArithmeticRunner):
        def forward(self, *args, **kwargs):
            entered.set()
            assert release.wait(5)
            return ForwardResult(1)

    adapter = counter_adapter()
    old = adapter._callbacks["status"]
    adapter._callbacks["status"] = lambda: (inspected.set(), old())[1]
    session = BenchSession.from_backend(
        Blocking(), debugger_adapter=adapter, identity={"model_id": "counter"}
    )
    debugger = session.debugger()

    def status():
        requested.set()
        return debugger.status()

    with ThreadPoolExecutor(2) as pool:
        future = pool.submit(session.forward, 1)
        assert entered.wait(5)
        read = pool.submit(status)
        assert requested.wait(5)
        try:
            assert not inspected.wait(0.05)
        finally:
            release.set()
        assert future.result().output == 1
        assert read.result()["runtime"]["cut_id"]
    assert inspected.is_set()

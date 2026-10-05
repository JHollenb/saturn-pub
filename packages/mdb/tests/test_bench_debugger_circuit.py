from __future__ import annotations

from types import SimpleNamespace

import pytest

from mdb.bench_debugger import BenchDebuggerAdapter
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import (
    BreakpointSpec,
    DebugCapability,
    DebugRequest,
    InMemoryDebugTransport,
)
from mdb.debugger_session import DebugSession
from mdb.debugger_worker import DebugWorkerController


class FakeBench:
    def __init__(self, family: str = "qwen2") -> None:
        self.family = family
        self.session_id = f"{family}-session"
        self.identity = {"model_id": f"fixture/{family}"}
        self.source_frames = None
        self.active = "main"
        self.branch = SimpleNamespace(cut="cut-0")
        self.calls: list[tuple[str, object]] = []
        self.adapter = object()  # exercise the family-neutral bench path

    def capabilities(self):
        return {
            "family": self.family,
            "execution_stages": ["token", "layer"],
            "writable": ["residual"],
            "mutable_closure": ["residual", "cursor"],
            "backend": "fixture",
        }

    def inspect(self, path=None):
        return {"path": path, "value": 0, "cut_id": self.branch.cut}

    def resolve_value(self, path):
        return 0

    def cursor(self):
        return {"token": 0}

    def checkpoint(self, name=None):
        self.branch = SimpleNamespace(cut="cut-1")
        return self.branch.cut

    def restore(self, cut_id):
        self.branch = SimpleNamespace(cut=cut_id)
        return {"cut_id": cut_id, "exact": True}

    def preview_edit(self, path, value, mode="replace"):
        self.calls.append(("preview_edit", {"path": path, "value": value, "mode": mode}))
        return {"act_id": "bench-act", "preview": True}

    def commit_edit(self, act_id):
        self.calls.append(("commit_edit", act_id))
        return {"act_id": act_id, "committed": True, "cut_id": self.branch.cut}

    def abort_edit(self, act_id):
        self.calls.append(("abort_edit", act_id))
        return {"act_id": act_id, "aborted": True, "cut_id": self.branch.cut}

    def step(self, kind, count=1):
        return {"kind": kind, "count": count}

    def compare(self, left, right):
        return {"left": left, "right": right}

    def fork(self, name):
        self.active = name
        return self.branch.cut

    def checkout(self, name):
        self.active = name
        return {"branch": name, "cut_id": self.branch.cut}

    def discover(self, spec):
        return {"spec": spec}

    def measure_circuit(self, spec):
        self.calls.append(("measure_circuit", spec))
        return {"report_ref": "report://fixture", "report": {"spec": spec}}

    def replay_circuit(self, evidence_ref):
        self.calls.append(("replay_circuit", evidence_ref))
        return {"evidence_ref": evidence_ref, "exact": True, "comparisons": []}


def _session(bench: FakeBench) -> tuple[DebugSession, DebugWorkerController]:
    adapter = BenchDebuggerAdapter(bench)
    capability = DebugCapability(
        job_id="job-fixture",
        session_id=bench.session_id,
        family=adapter.capabilities.family,
        model_identity=bench.identity,
        boundaries=adapter.capabilities.boundaries,
        inspectable_slots=adapter.capabilities.inspectable_slots,
        writable_slots=adapter.capabilities.writable_slots,
        exact_replay=True,
        optional_operations=adapter.optional_operations(),
    )
    worker = DebugWorkerController(capability, adapter)
    return DebugSession(
        InMemoryDebugTransport(worker), job_id="job-fixture"
    ), worker


def test_bench_adapter_advertises_circuit_ops_only_for_ar() -> None:
    ar = BenchDebuggerAdapter(FakeBench("qwen2"))
    assert "measure_circuit" in ar.optional_operations()
    assert "replay_circuit" in ar.optional_operations()

    mamba = BenchDebuggerAdapter(FakeBench("mamba"))
    assert "measure_circuit" in mamba.optional_operations()
    assert "replay_circuit" in mamba.optional_operations()

    diffusion = BenchDebuggerAdapter(FakeBench("flux"))
    assert "measure_circuit" not in diffusion.optional_operations()
    assert "replay_circuit" not in diffusion.optional_operations()


def test_live_circuit_ops_use_typed_spec_and_evidence_ref() -> None:
    bench = FakeBench()
    session, worker = _session(bench)
    session.add_breakpoint(BreakpointSpec("token", 0))
    worker.safe_point("token", 0)

    measured = session.measure_circuit({"sites": [1], "horizon": 1})
    assert measured["report_ref"] == "report://fixture"
    assert bench.calls[-1] == ("measure_circuit", {"sites": [1], "horizon": 1})

    replayed = session.replay_circuit("evidence://fixture")
    assert replayed["exact"] is True
    assert bench.calls[-1] == ("replay_circuit", "evidence://fixture")

    bad_spec = worker.handle(
        DebugRequest("measure_circuit", {"spec": "import os"})
    )
    assert not bad_spec.ok
    bad_ref = worker.handle(DebugRequest("replay_circuit", {"evidence_ref": {}}))
    assert not bad_ref.ok


def test_circuit_ops_preserve_pending_act_refusal() -> None:
    bench = FakeBench()
    session, worker = _session(bench)
    session.add_breakpoint(BreakpointSpec("token", 0))
    worker.safe_point("token", 0)
    session.set_value("residual", 1)
    with pytest.raises(Exception, match="pending Act"):
        session.measure_circuit({"sites": [1]})
    with pytest.raises(Exception, match="pending Act"):
        session.replay_circuit("evidence://fixture")
    assert session.abort()["aborted"] is True


def test_debugger_cli_dispatches_structured_circuit_commands() -> None:
    class Session:
        aliases = {}

        def measure_circuit(self, spec):
            return {"report_ref": "report://fixture", "spec": spec}

        def replay_circuit(self, evidence_ref):
            return {"evidence_ref": evidence_ref, "exact": True}

        def preview(self):
            return {}

        def commit(self):
            return {}

        def abort(self):
            return {}

        def inspect(self, path=None):
            return {"path": path}

        def record_refusal(self, command, error):
            raise AssertionError(f"unexpected refusal for {command}: {error}")

    shell = DebugShell(Session(), json_output=True)
    assert shell.execute('measure-circuit {"sites":[0]}')
    assert shell.execute("replay_circuit evidence://fixture")

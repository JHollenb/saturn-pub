from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mdb_lanes.bench_debugger import BenchDebuggerAdapter
from mdb_lanes.debugger_cli import DebugShell
from mdb_lanes.debugger_protocol import (
    BreakpointSpec,
    DebugCapability,
    DebugRequest,
    InMemoryDebugTransport,
)
from mdb_lanes.debugger_session import DebugSession
from mdb_lanes.debugger_worker import DebugWorkerController


class FakeBench:
    def __init__(self, family: str = "qwen2", available: bool = True) -> None:
        self.family = family
        self.available = available
        self.session_id = f"{family}-session"
        self.identity = {"model_id": f"fixture/{family}"}
        self.source_frames = None
        self.active = "main"
        self.branch = SimpleNamespace(cut="cut-0")
        self.calls: list[tuple[str, object]] = []
        self.adapter = object()

    def capabilities(self):
        return {
            "family": self.family,
            "execution_stages": ["token", "layer"],
            "writable": ["residual"],
            "mutable_closure": ["residual", "cursor"],
            "backend": "fixture",
            "circuit_discovery": {
                "protocol": "saturn-bench-circuit-discovery-v1",
                "available": self.available,
            },
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

    def discover_circuit(self, spec):
        self.calls.append(("discover_circuit", spec))
        return {
            "op": "discover_circuit",
            "cut_id": "cut-discovered",
            "report_ref": "cas://sha256/discovery",
            "report": {"schema": "saturn-bench-circuit-discovery-v1", "spec": spec},
        }


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
    return DebugSession(InMemoryDebugTransport(worker), job_id="job-fixture"), worker


def test_discover_circuit_is_qwen2_and_capability_gated() -> None:
    assert "discover_circuit" in BenchDebuggerAdapter(FakeBench()).optional_operations()
    assert "discover_circuit" not in BenchDebuggerAdapter(FakeBench(available=False)).optional_operations()
    assert "discover_circuit" not in BenchDebuggerAdapter(
        FakeBench(family="mamba", available=True)
    ).optional_operations()


def test_discover_circuit_uses_paused_worker_and_receipt() -> None:
    bench = FakeBench()
    session, worker = _session(bench)
    assert "discover_circuit" in session.breakable()["optional_operations"]
    with pytest.raises(Exception, match="requires a paused worker"):
        session.discover_circuit({"sites": [1]})

    session.add_breakpoint(BreakpointSpec("token", 0))
    worker.safe_point("token", 0)
    result = session.discover_circuit({"sites": [1], "seed": 0})
    assert result["op"] == "discover_circuit"
    assert result["report_ref"].startswith("cas://")
    assert bench.calls[-1] == ("discover_circuit", {"sites": [1], "seed": 0})
    assert session.current_cut == "cut-discovered"
    receipt = session.write_receipt()
    assert any(
        item["request"]["operation"] == "discover_circuit"
        for item in receipt["commands"]
    )


def test_discover_circuit_rejects_nonfinite_spec_before_callback() -> None:
    bench = FakeBench()
    session, worker = _session(bench)
    session.add_breakpoint(BreakpointSpec("token", 0))
    worker.safe_point("token", 0)

    response = worker.handle(
        DebugRequest("discover_circuit", {"spec": {"score": float("nan")}})
    )
    assert not response.ok
    assert "JSON values only" in (response.error or "")
    assert not bench.calls


def test_discover_circuit_cli_accepts_inline_and_file_specs(tmp_path: Path) -> None:
    class Session:
        aliases = {}

        def __init__(self):
            self.specs = []

        def discover_circuit(self, spec):
            self.specs.append(spec)
            return {"op": "discover_circuit", "spec": spec}

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

    session = Session()
    shell = DebugShell(session, json_output=True)
    assert shell.execute('discover-circuit --spec {"sites":[0]}')

    spec_file = tmp_path / "discover-spec.json"
    spec_file.write_text(json.dumps({"sites": [1], "seed": 3}), encoding="utf-8")
    assert shell.execute(f"discover_circuit --spec @{spec_file}")
    assert session.specs == [{"sites": [0]}, {"sites": [1], "seed": 3}]

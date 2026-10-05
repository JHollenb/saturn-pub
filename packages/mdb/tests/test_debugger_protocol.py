from __future__ import annotations

from dataclasses import dataclass

import pytest

from mdb.debugger_adapters import AdapterCapabilities, DebuggerAdapter
from mdb.debugger_cli import build_parser
from mdb.debugger_protocol import (
    ActSpec,
    BreakpointSpec,
    DebugCapability,
    DebugRequest,
    InMemoryDebugTransport,
    WatchSpec,
)
from mdb.debugger_session import DebugSession
from mdb.debugger_worker import DebugWorkerController


@dataclass
class FakeAdapter(DebuggerAdapter):
    paused: bool = False
    cut: int = 0

    def __post_init__(self) -> None:
        self.capabilities = AdapterCapabilities(
            family="autoregressive",
            boundaries=("token", "layer", "sequence", "expert"),
            inspectable_slots=("token_ids", "kv_cache", "cursor"),
            writable_slots=("token_ids", "kv_cache", "cursor"),
            mutable_closure=("token_ids", "kv_cache", "cursor"),
            exact_replay=True,
            state_schema_id="fake-v1",
            backend="cpu",
        )

    def status(self):
        return {"cut_id": f"cut-{self.cut}", "cursor": self.cut}

    def inspect(self, path=None):
        return {"path": path, "value": [self.cut]}

    def checkpoint(self, name=None):
        self.cut += 1
        return {"cut_id": f"cut-{self.cut}", "name": name, "parent_cut": f"cut-{self.cut - 1}"}

    def restore(self, cut_id):
        return {"cut_id": cut_id, "exact": True}

    def preview_act(self, act: ActSpec):
        return {"act_id": act.act_id, "preview": True, "exact": True}

    def commit_act(self, act_id):
        self.cut += 1
        return {"act_id": act_id, "cut_id": f"cut-{self.cut}", "committed": True}

    def abort_act(self, act_id):
        return {"act_id": act_id, "aborted": True}

    def replay(self, cut_id):
        return {"cut_id": cut_id, "replayed": True, "exact": True}

    def step(self, count=1):
        self.cut += count
        return {"cut_id": f"cut-{self.cut}", "steps": count}

    def continue_run(self):
        return {"state": "running"}


def make_session() -> tuple[DebugSession, DebugWorkerController]:
    adapter = FakeAdapter()
    capability = DebugCapability(
        job_id="job-1",
        session_id="session-1",
        family="autoregressive",
        model_identity={"model": "fake"},
        boundaries=adapter.capabilities.boundaries,
        inspectable_slots=adapter.capabilities.inspectable_slots,
        writable_slots=adapter.capabilities.writable_slots,
        exact_replay=True,
    )
    worker = DebugWorkerController(capability, adapter)
    return DebugSession(InMemoryDebugTransport(worker), job_id="job-1"), worker


def test_attach_break_pause_inspect_and_transactional_write() -> None:
    session, worker = make_session()
    breakpoint = session.add_breakpoint(BreakpointSpec("token", 2))
    assert breakpoint["breakpoint"]["kind"] == "token"
    worker.safe_point("token", 2)
    assert session.status()["state"] == "paused"
    session.checkpoint("manual")
    preview = session.set_value("kv_cache", {"layer": 1})
    assert preview["preview"] is True
    assert session.commit()["committed"] is True


def test_idempotent_request_and_unsafe_write_refused() -> None:
    session, worker = make_session()
    request = DebugRequest("status", request_id="same")
    first = worker.handle(request)
    second = worker.handle(request)
    assert first == second
    session.checkpoint("root")
    with pytest.raises(Exception, match="read-only"):
        session.set_value("weights.layer_0", 1)


def test_mutations_require_paused_safe_point_and_one_pending_act() -> None:
    session, worker = make_session()
    session.checkpoint("root")
    with pytest.raises(Exception, match="paused"):
        session.set_value("kv_cache", {"layer": 1})

    session.pause()
    worker.safe_point("token", 0)
    session.set_value("kv_cache", {"layer": 1})
    with pytest.raises(RuntimeError, match="already pending"):
        session.set_value("kv_cache", {"layer": 2})
    with pytest.raises(Exception, match="pending Act"):
        session.continue_run()
    assert session.abort()["aborted"] is True
    assert session.continue_run()["state"] == "running"


def test_watch_and_exact_replay_receipt() -> None:
    session, worker = make_session()
    watch = session.add_watch(WatchSpec("cursor", "gt 0"))
    assert watch["watch"]["path"] == "cursor"
    cut = session.checkpoint("root")["cut_id"]
    assert session.fork("candidate")["branch"] is True
    session.pause()
    worker.safe_point("token", 0)
    assert session.replay(cut)["exact"] is True
    receipt = session.write_receipt()
    assert receipt["schema"] == "saturn-debugger-session-receipt-v1"
    assert receipt["commands"]
    assert len(receipt["receipt_sha256"]) == 64


def test_safe_point_bridge_blocks_and_exposes_capability() -> None:
    session, worker = make_session()
    assert worker.capability.family == "autoregressive"
    status = worker.safe_point("token", 0, {"phase": "prefill"})
    assert status["state"] == "running"


def test_cli_surface_contains_unified_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["attach", "job-1", "--command", "status"]).job_id == "job-1"
    assert parser.parse_args(["debug", "job-1"]).command == "debug"
    assert parser.parse_args(["script", "debug.sat", "--job", "job-1"]).file.name == "debug.sat"
    assert parser.parse_args(["run", "--debug", "--", "python", "worker.py"]).debug is True

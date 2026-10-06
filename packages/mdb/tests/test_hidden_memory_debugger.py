from types import SimpleNamespace

import pytest

from mdb.bench_cli import run_commands
from mdb.bench_debugger import BenchDebuggerAdapter
from mdb.debugger_cli import DebugShell, _memory_table
from mdb.debugger_protocol import BreakpointSpec, DebugCapability, InMemoryDebugTransport
from mdb.debugger_session import DebugSession
from mdb.debugger_worker import DebugWorkerController


class MemoryBench:
    family, session_id, identity, source_frames, active = (
        "qwen2",
        "memory-session",
        {"model_id": "fixture/qwen"},
        None,
        "main",
    )

    def __init__(self):
        self.branch, self.calls = SimpleNamespace(cut="cut-0"), []

    def capabilities(self):
        return {
            "family": self.family,
            "execution_stages": ["token"],
            "writable": ["memory"],
            "mutable_closure": ["memory", "cursor"],
            "backend": "fixture",
            "operations": ["memory_read", "memory_write", "memory_diff", "memory_probe"],
        }

    def inspect(self, path=None):
        return {"cut_id": self.branch.cut, "path": path}

    def resolve_value(self, path):
        return None

    def cursor(self):
        return {"token": 0}

    def checkpoint(self, name=None):
        return self.branch.cut

    def restore(self, cut_id):
        self.branch.cut = cut_id
        return {"cut_id": cut_id}

    def preview_edit(self, path, value, mode="replace"):
        return {"act_id": "act"}

    def commit_edit(self, act_id):
        return {"cut_id": self.branch.cut}

    def abort_edit(self, act_id):
        return {"aborted": True}

    def step(self, kind, count=1):
        return {"count": count}

    def compare(self, left, right):
        return {"left": left, "right": right}

    def fork(self, name):
        return self.branch.cut

    def checkout(self, name):
        return {"cut_id": self.branch.cut}

    def discover(self, spec):
        return spec

    def memory_read(self, spec):
        self.calls.append(("read", spec))
        return {"rows": []}

    def memory_write(self, spec):
        self.calls.append(("write", spec))
        self.branch.cut = "cut-1"
        return {"op": "memory_write", "cut_id": "cut-1", "parent_cut": "cut-0"}

    def memory_diff(self, spec):
        self.calls.append(("diff", spec))
        return {"rows": []}

    def memory_probe(self, spec):
        self.calls.append(("probe", spec))
        return {"cut_id": self.branch.cut, "closure_equal": True}


def make_session():
    bench = MemoryBench()
    adapter = BenchDebuggerAdapter(bench)
    cap = DebugCapability(
        job_id="job",
        session_id=bench.session_id,
        family=adapter.capabilities.family,
        model_identity=bench.identity,
        boundaries=adapter.capabilities.boundaries,
        inspectable_slots=adapter.capabilities.inspectable_slots,
        writable_slots=adapter.capabilities.writable_slots,
        exact_replay=True,
        optional_operations=adapter.optional_operations(),
    )
    worker = DebugWorkerController(cap, adapter)
    return bench, DebugSession(InMemoryDebugTransport(worker), job_id="job"), worker


def test_memory_commands_aliases_guards_dispatch_and_output():
    bench, session, worker = make_session()
    session.checkpoint("root")
    session.memory_read({"cut_id": "@current"})
    assert bench.calls[-1] == ("read", {"cut_id": "cut-0"})
    with pytest.raises(Exception, match="paused"):
        session.memory_write({"mode": "zero"})
    session.add_breakpoint(BreakpointSpec("token", 0))
    worker.safe_point("token", 0)
    shell = DebugShell(session, json_output=True)
    assert shell.execute('memory write {"mode":"scale","value":0.5}')
    assert session.current_cut == "cut-1" and "memory" in shell._complete("mem")["matches"]
    assert (
        run_commands(
            bench, [{"op": "memory_diff", "spec": {"left_cut": "cut-0", "right_cut": "cut-1"}}]
        )[0]["op"]
        == "memory_diff"
    )
    rendered = _memory_table(
        {
            "native": {"answer": "A", "token_margin": -0.25},
            "candidate": {"answer": "B", "token_margin": 0.5},
            "signed_raw_logit_contrast": 0.75,
        }
    )
    assert "native" in rendered and "+0.75" in rendered


def test_diffusion_adapter_does_not_advertise_memory_hooks():
    bench = MemoryBench()
    bench.family = "flux"
    assert not (
        {"memory_read", "memory_write", "memory_diff", "memory_probe"}
        & set(BenchDebuggerAdapter(bench).optional_operations())
    )

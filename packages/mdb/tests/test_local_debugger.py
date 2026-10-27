"""Local command motion, capability refusals and dependency-free onboarding."""

import json
import subprocess
import sys

import pytest

from mdb.cli import main
from mdb.counter_debugger import CounterDebuggerAdapter
from mdb.debugger_cli import DebugShell
from mdb.debugger_protocol import DebuggerProtocolError, DebugRequest
from mdb.local_cli import doctor_report
from mdb.local_debugger import connect_local


def counter(**kwargs):
    adapter = CounterDebuggerAdapter()
    owner, session, motion = connect_local(adapter, identity={"model_id": "counter"}, **kwargs)
    return adapter, owner, session, motion


def test_continue_stops_on_break_and_watch_and_remains_paused():
    adapter, owner, session, _ = counter()
    shell = DebugShell(session, json_output=True)
    for command in ("checkpoint parent", "break step 3", "continue"):
        assert shell.execute(command)
    assert adapter.state["value"] == 3 and session.status()["state"] == "paused"
    assert shell.execute("watch value >= 5 level")
    assert shell.execute("continue")
    assert adapter.state["value"] == 5
    assert session.status()["pause_reason"]["watches"]
    assert shell.execute("restore parent")
    assert adapter.state["value"] == 0
    owner.close()


def test_real_counter_forks_and_preview_abort_commit_reverse():
    adapter, owner, session, _ = counter()
    parent = session.checkpoint("parent")["cut_id"]
    session.fork("candidate")
    session.set_value("value", 9)
    assert adapter.state["value"] == 0
    session.abort()
    assert adapter.state["value"] == 0
    session.set_value("value", 9)
    session.commit()
    assert adapter.state["value"] == 9 and adapter.branches["main"]["value"] == 0
    session.restore(parent)
    session.step(3)
    result = session.reverse_continue("value < 2")
    assert result["exact"] and adapter.state["value"] == 1
    before = adapter.state["cut"]
    with pytest.raises(DebuggerProtocolError, match="no matching"):
        session.reverse_continue("value < -100")
    assert adapter.state["cut"] == before
    owner.close()


def test_counter_integrity_idempotence_and_budget():
    adapter, owner, session, _ = counter(max_steps=2, max_new_tokens=10)
    request = DebugRequest("step", {"count": 1}, request_id="same-local-request")
    first = session.transport.request(request)
    assert session.transport.request(request) == first
    assert adapter.state["value"] == 1
    result = session.continue_run()
    assert result["stopped"] == "transition-budget" and adapter.state["value"] == 3
    ref = adapter.state["cut"]
    adapter.cuts[ref]["value"] = 99
    with pytest.raises(DebuggerProtocolError, match="changed"):
        session.restore(ref)
    assert adapter.state["value"] == 3
    owner.close()


def test_unknown_step_is_not_silently_replaced_and_pending_edit_stays_pending():
    adapter, owner, session, _ = counter()
    with pytest.raises(DebuggerProtocolError, match="unsupported"):
        session.step(kind="layer")
    assert adapter.state["cursor"] == 0
    session.set_value("value", 9)
    with pytest.raises(DebuggerProtocolError, match="pending Act"):
        session.step()
    assert session.status()["state"] == "paused"
    session.abort()
    assert adapter.state["value"] == 0
    owner.close()


def test_compact_frames_full_json_and_pre_session_help(capsys):
    _, owner, session, _ = counter()
    shell = DebugShell(session)
    assert shell.execute("step")
    compact = capsys.readouterr().out
    assert "(main) phase=counter step=1 value=1 cut=" in compact
    assert len(compact) < 250
    assert shell.execute("inspect --json")
    full = json.loads(capsys.readouterr().out)
    assert full["cursor"]["step"] == 1 and "cut_id" in full
    assert shell.execute("inspect --full")
    assert json.loads(capsys.readouterr().out)["value"] == 1
    assert main(["help"]) == 0
    help_text = capsys.readouterr().out
    for group in ("Motion", "Breaks", "State", "Memory", "History"):
        assert group in help_text
    owner.close()


def test_default_entry_doctor_alias_and_repl_without_a_job(monkeypatch, capsys):
    calls = iter(["step", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(calls))
    assert main(["debug"]) == 0
    assert "Counter demo" in capsys.readouterr().out
    assert main(["open", "--tiny", "--command", "step", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cursor"]["step"] == 1
    assert main(["doctor", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["scheduler"] == {"checked": False}


def test_local_flags_do_not_shadow_job_attachment(capsys):
    assert main(["debug", "job-1", "--tiny", "counter", "--command", "step"]) == 1
    assert "choose local debugging or job attachment" in capsys.readouterr().err
    assert main(["debug", "job-1", "--root", "unused", "--command", "step"]) == 1
    assert "cannot be combined" in capsys.readouterr().err


def test_doctor_network_is_opt_in_and_tokens_are_not_reported(monkeypatch):
    def broken(request, **kwargs):
        assert request.full_url == "http://example.invalid/health"
        assert request.get_header("Authorization") == "Bearer dummy-token"
        raise OSError("dummy-token must not be printed")

    monkeypatch.setattr("mdb.local_cli.urlopen", broken)
    assert doctor_report()["scheduler"]["checked"] is False
    report = doctor_report("http://example.invalid", "dummy-token")
    assert report["ok"] is False and report["scheduler"]["reachable"] is False
    assert "dummy-token" not in json.dumps(report)


def test_base_install_counter_and_doctor_block_model_packages():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','transformers','transformer_lens','mrun',
                                     'saturn','diffusers'}:
            raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
from mdb.cli import main
assert main(['debug','--tiny','--command','step','--json']) == 0
assert main(['doctor','--json']) == 0
assert main(['help']) == 0
"""
    subprocess.run([sys.executable, "-c", code], check=True)

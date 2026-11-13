"""REPL transcripts on tiny Qwen2 (BF16, seed 113), compared with checked-in files.

Regenerate after an intended change: MDB_UPDATE_GOLDEN=1 pytest tests/test_repl_golden.py
"""

import io
import json
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.ops import BadArgs, UnknownOp  # noqa: E402
from mdb.repl.__main__ import main  # noqa: E402
from mdb.repl.shell import Shell, split  # noqa: E402

GOLDEN = Path(__file__).parent / "repl_golden"
SCRIPTS = sorted(path.stem for path in GOLDEN.glob("*.mdb"))
ARGS = ["--tiny", "qwen2", "--dtype", "bf16", "--tokens", "1 5 9 3"]


_HEX = re.compile(r"\b[0-9a-f]{12,64}\b")
_FLOAT = re.compile(r"(?<![\w.])-?\d+\.\d+(?:e[+-]?\d+)?\b|(?<![\w.])-?\d+e[+-]?\d+\b")


def normalize(text: str) -> str:
    """What a golden transcript pins: addresses, operators, proof IDs, hit kinds, labels and
    counts. Build-specific numerics are masked: hex digests and cut fingerprints become
    ``<sha>`` and floats are rounded to 3 significant digits."""
    text = _HEX.sub("<sha>", text)
    return _FLOAT.sub(lambda m: f"{float(m.group(0)):.3g}", text)


def test_normalize_masks_digests_and_rounds_floats_but_keeps_pinned_text():
    line = ("t3.L0.n  bfloat16 [1, 1, 16]  {P02 P03}\n  norm 3.9992 min -2.1094 mean 1.5e-05 "
            "[measured]\n  pointer occurrence ~stepped sha256 742f42b2b5091a76 cut a5caf85c8929")
    out = normalize(line)
    assert "norm 4 min -2.11 mean 1.5e-05 [measured]" in out
    assert "sha256 <sha> cut <sha>" in out
    assert "t3.L0.n" in out and "{P02 P03}" in out and "[1, 1, 16]" in out
    assert normalize("hits 12 id 7 L10.h 2 staged") == "hits 12 id 7 L10.h 2 staged"
    assert normalize("norm 3.9991") == normalize("norm 3.9992")  # a rebuild's rounding noise
    assert normalize("norm 3.9") != normalize("norm 4.2")  # a real change still shows


@pytest.mark.parametrize("name", SCRIPTS)
def test_golden_transcript(name, capsys):
    assert main([*ARGS, "--script", str(GOLDEN / f"{name}.mdb")]) == 0
    text = capsys.readouterr().out
    expected = GOLDEN / f"{name}.out"
    if os.environ.get("MDB_UPDATE_GOLDEN") == "1":
        expected.write_text(text)
    assert normalize(text) == normalize(expected.read_text())


def test_json_mode_prints_one_json_object_per_command(capsys):
    assert main([*ARGS, "--json", "--script", str(GOLDEN / "breakpoints.mdb")]) == 0
    lines = capsys.readouterr().out.splitlines()
    commands = [line for line in (GOLDEN / "breakpoints.mdb").read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    assert len(lines) == len(commands) + 1  # + the start from --tokens
    results = [json.loads(line) for line in lines]
    assert results[0]["op"] == "start"
    assert [r.get("op") for r in results[1:3]] == ["break", "tbreak"]


def test_cli_engine_entry_dispatches_to_the_repl(capsys, tmp_path):
    from mdb.cli import main as cli_main

    script = tmp_path / "s.mdb"
    script.write_text("step\nwhere\n")
    assert cli_main(["engine", *ARGS, "--script", str(script)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "(mdb) start 1 5 9 3"
    assert "t3.L0.a" in out


def test_prefixes_aliases_repeat_record_and_source(tmp_path):
    from mdb.ops.debugger import open_tiny

    out = io.StringIO()
    shell = Shell(open_tiny("qwen2").ops(), out=out)
    assert shell.resolve("fin") == "finish"
    assert shell.resolve("bt") == "backtrace" and shell.resolve("u") == "until"
    assert shell.resolve("d") == "delete" and shell.resolve("i") == "info"
    with pytest.raises(UnknownOp, match="ambiguous"):
        shell.resolve("re")
    with pytest.raises(UnknownOp, match="undefined"):
        shell.resolve("zzz")
    record = tmp_path / "rec.mdb"
    shell.execute(f"record {record}")
    shell.execute("start 1 5 9 3")
    shell.execute("s")
    shell.execute("", repeat=True)
    assert shell.ctx.run.where == "t3.L0.q"
    shell.execute("record stop")
    assert record.read_text().splitlines() == ["start 1 5 9 3", "s", "s"]
    shell.execute("print")  # not a motion: an empty line still repeats the step
    shell.execute("", repeat=True)
    assert shell.ctx.run.where == "t3.L0.k"
    shell.execute("", repeat=False)
    assert shell.ctx.run.where == "t3.L0.k"
    assert shell.source(str(record)) == 0
    assert shell.ctx.run.where == "t3.L0.q"
    shell.execute("help step")
    assert "step [N]" in out.getvalue()
    assert shell.execute("quit") is False


def test_split_keeps_brackets_and_quotes_together():
    words = [w for w, _, _ in split('print logits[" hot", " cold"] top=3 "a b"')]
    assert words == ["print", 'logits[" hot", " cold"]', "top=3", "a b"]


# =========================================================================== fix round, part 1


def _shell(**options):
    from mdb.ops.debugger import open_tiny

    out = io.StringIO()
    shell = Shell(open_tiny("qwen2").ops(), out=out, **options)
    shell.execute("start 1 5 9 3")
    return shell, out


def test_source_stops_at_the_first_error_unless_keep_going(tmp_path):
    script = tmp_path / "s.mdb"
    script.write_text("step\nbreak L9.p\nstep\nwhere\n")
    shell, out = _shell()
    assert shell.source(str(script)) == 1
    assert shell.ctx.run.where == "t3.L0.a"  # the step after the error never ran
    assert "stopped sourcing" in out.getvalue() and "source -k" in out.getvalue()
    shell, out = _shell()
    assert shell.source(str(script), keep_going=True) == 1
    assert shell.ctx.run.where == "t3.L0.q"  # -k: the script went on
    shell, out = _shell()
    outer = tmp_path / "outer.mdb"
    outer.write_text(f"source -k {script}\nstep\n")
    shell.source(str(outer))
    assert shell.ctx.run.where == "t3.L0.k"  # -k errors do not stop the calling script
    inner = tmp_path / "inner.mdb"
    inner.write_text("break L9.p\nstep\n")
    outer.write_text(f"source {inner}\nstep\n")
    shell, out = _shell()
    shell.source(str(outer))
    assert shell.ctx.run.where == "t3.L0.h"  # a failing nested source stops the outer one too


def test_source_depth_is_bounded_even_for_a_script_that_sources_itself(tmp_path):
    script = tmp_path / "self.mdb"
    script.write_text(f"source {script}\n")
    shell, out = _shell()
    shell.execute(f"source {script}")
    assert f"nests more than {16} deep" in out.getvalue()
    assert shell.depth == 0 and shell.errors >= 1
    shell.execute("where")  # the session survives
    assert out.getvalue().rstrip().endswith("staged edits: 0, lookahead none")
    shell, out = _shell()
    shell.execute(f"source -k {script}")  # even keep-going cannot recurse forever
    assert out.getvalue().count("(mdb) source") == 0 or "deep" in out.getvalue()


def test_record_is_refused_while_sourcing_never_overwrites_and_is_0600(tmp_path):
    import stat

    shell, out = _shell()
    target = tmp_path / "rec.mdb"
    shell.execute(f"record {target}")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    shell.execute("step")
    shell.execute("record stop")
    assert target.read_text() == "step\n"
    shell.execute(f"record {target}")  # exists: refused, content intact
    assert shell.record_to is None and "exists" in out.getvalue()
    assert "record append" in out.getvalue() and target.read_text() == "step\n"
    shell.execute(f"record append {target}")
    shell.execute("step")
    shell.execute("record stop")
    assert target.read_text() == "step\nstep\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    # from a sourced script (auditor's s7_record.mdb): refused, nothing is created
    victim = tmp_path / "victim.txt"
    script = tmp_path / "s7.mdb"
    script.write_text(f"record {victim}\ncheckpoint\nrecord stop\n")
    shell, out = _shell()
    shell.source(str(script))
    assert not victim.exists() and shell.record_to is None
    assert "not available from a sourced script" in out.getvalue()


def test_ctrl_c_during_a_command_prints_where_and_the_session_survives(monkeypatch):
    from mdb.engine import run as runmod

    shell, out = _shell()
    original = runmod.Run.until
    calls = {"n": 0}

    def interrupting(self, target):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt
        return original(self, target)

    monkeypatch.setattr(runmod.Run, "until", interrupting)
    code = shell.loop(io.StringIO("until L0.q\nuntil L0.q\nwhere\n"))
    text = out.getvalue()
    assert "interrupted at t3.L0.h" in text and "Traceback" not in text
    assert shell.ctx.run.where == "t3.L0.q"  # the loop carried on after the interrupt
    assert code == 1 and shell.errors == 1


def test_an_interrupt_stops_a_running_script(tmp_path, monkeypatch):
    from mdb.engine import run as runmod

    script = tmp_path / "s.mdb"
    script.write_text("until L0.q\nstep\nstep\n")
    shell, out = _shell()
    original = runmod.Run.until

    def interrupting(self, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(runmod.Run, "until", interrupting)
    shell.source(str(script), keep_going=True)
    monkeypatch.setattr(runmod.Run, "until", original)
    assert out.getvalue().count("interrupted at") == 1  # the remaining lines did not run
    assert shell.ctx.run.where == "t3.L0.h"


def test_an_unexpected_exception_is_an_internal_error_not_a_traceback(monkeypatch):
    from mdb.ops import builtin

    shell, out = _shell()
    monkeypatch.setattr(builtin, "_stop",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("synthetic")))
    assert shell.execute("where") is True
    text = out.getvalue()
    assert "error (internal): KeyError: 'synthetic'" in text and "Traceback" not in text
    monkeypatch.undo()
    shell, out = _shell()
    monkeypatch.setattr(builtin, "_stop",
                        lambda *a, **k: (_ for _ in ()).throw(KeyError("synthetic")))
    shell.json_mode = True
    shell.execute("where")
    data = json.loads(out.getvalue().splitlines()[-1])
    assert data["error"]["type"] == "internal" and data["state"] == "unresolved"
    assert data["reason"].startswith("KeyError")


def test_json_error_lines_carry_a_state():
    shell, out = _shell(json_mode=True)
    shell.execute("break L9.p")
    data = json.loads(out.getvalue().splitlines()[-1])
    assert data["state"] == "refused" and "valid layers are 0..1" in data["reason"]
    shell.execute("where")
    assert json.loads(out.getvalue().splitlines()[-1])["state"] == "resolved"


def test_parser_edge_cases_from_the_audit():
    from mdb.repl.shell import parse

    assert parse("set", "q[h1]=2") == {"target": "q[h1]", "value": 2.0}
    assert parse("set", "L0.q[h1]=2 add") == {"target": "L0.q[h1]", "value": 2.0, "op": "add"}
    assert parse("set", "q = -1") == {"target": "q", "value": -1.0}
    assert parse("set", "q=-1") == {"target": "q", "value": -1.0}
    assert parse("set", "q[d=0:2]=L0.a[d=0:2]") == {"target": "q[d=0:2]", "value": "L0.a[d=0:2]"}
    assert parse("set", "q[d=0] = 3") == {"target": "q[d=0]", "value": 3.0}
    assert parse("run", 'budget=3 "hello world"') == {"budget": 3, "prompt": "hello world"}
    assert parse("run", '"hello world" budget=3') == {"budget": 3, "prompt": "hello world"}
    assert parse("run", "1 2 budget=3") == {"budget": 3, "prompt": [1, 2]}
    assert parse("run", "budget=3") == {"budget": 3}
    with pytest.raises(BadArgs, match="budget must be an integer"):
        parse("run", "budget=x")
    assert parse("up", "") == {} and parse("up", "2") == {"n": 2}
    assert parse("up", "o") == {"port": "o"} and parse("up", "o 2") == {"port": "o", "n": 2}
    assert parse("condition", "3 norm > 1") == {"id": 3, "expr": "norm > 1"}
    assert parse("condition", "3") == {"id": 3}
    assert parse("ignore", "3 2") == {"id": 3, "count": 2}
    assert parse("enable", "1 2") == {"ids": [1, 2]} and parse("disable", "") == {}
    assert parse("until", "L1.p budget=3") == {"target": "L1.p", "budget": 3}
    assert parse("info", "breakpoints offset=32 limit=8") == {
        "subject": "breakpoints", "offset": 32, "limit": 8}
    assert parse("continue", "3 budget=2") == {"n": 3, "budget": 2}


def test_gdb_session_end_to_end_through_the_shell():
    shell, out = _shell()
    for line in ("break L*.p", "break L0.o if norm > 0", "disable 1", "continue",
                 "enable 1", "ignore 1 1", "condition 2", "continue 2", "info breakpoints",
                 "run budget=1 1 5 9 3"):
        shell.execute(line)
    text = out.getvalue()
    assert "Breakpoint 2, L0.o" in text and "Will ignore next 1 crossings of breakpoint 1." in text
    assert "Breakpoint 2 now unconditional." in text and "Traceback" not in text
    assert shell.errors == 0, text


# --- round 3, part 2 -------------------------------------------------------------------------


def _json_lines(out):
    return [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]


def test_record_help_and_source_emit_json_in_json_mode(tmp_path):
    """G6: every line in --json mode parses; record says what it did."""
    shell, out = _shell(json_mode=True)
    out.truncate(0)
    out.seek(0)
    path = tmp_path / "session.mdb"
    script = tmp_path / "inner.mdb"
    script.write_text("where\n")
    for line in (f"record {path}", "where", "help", f"source {script}", "record stop"):
        shell.execute(line)
    objects = _json_lines(out)  # raises if any line is not JSON
    assert objects and all("op" in obj or "state" in obj for obj in objects)
    assert any(obj.get("op") == "record" for obj in objects)
    assert any(obj.get("op") == "help" and obj.get("text") for obj in objects)


def test_the_tokens_line_carries_a_provenance_qualifier():
    """G5: a sampled token downstream of an edit says so; an unedited run keeps the bare label."""
    shell, out = _shell()
    shell.execute("until sample")
    out.truncate(0)
    out.seek(0)
    shell.execute("print sample")
    plain = [x for x in out.getvalue().splitlines() if x.strip().startswith("tokens")]
    assert plain and plain[0].rstrip().endswith("[measured]"), out.getvalue()
    shell, out = _shell()
    shell.execute("until L0.a")
    shell.execute("set L0.v=0.5 add")
    shell.execute("until sample")
    out.truncate(0)
    out.seek(0)
    shell.execute("print sample")
    edited = [x for x in out.getvalue().splitlines() if x.strip().startswith("tokens")]
    assert edited and "under edit" in edited[0], out.getvalue()


def test_an_interrupt_between_the_session_and_the_cursor_is_rolled_back(monkeypatch):
    """G1 at the shell: Ctrl-C right after the native transition; the next command is correct."""
    from saturn_pub.core import Session

    shell, out = _shell()
    shell.execute("until L0.a")
    shell.execute("set L0.v=0.5 add")
    original = Session.continue_
    fired = {"n": 0}

    def interrupting(self, *a, **k):
        result = original(self, *a, **k)
        if not fired["n"]:
            fired["n"] = 1
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Session, "continue_", interrupting)
    shell.loop(io.StringIO("finish\nfinish\nprint h_out\n"))
    text = out.getvalue()
    assert "interrupted" in text and "Traceback" not in text
    assert shell.ctx.run.where == "t3.L0.h_out"
    assert "under edit" in text.split("interrupted", 1)[1]

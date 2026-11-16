"""Config through the front ends: ``mdb config show|explain|init|check``, ``mdb engine`` flags
and config defaults (device on tiny models, session, sink, init file, repl.defaults, the
per-run lookahead/retain_rows arguments), ``mdb serve`` defaults, model aliases and roots, and
the older selfcheck flags."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdb.cli import main as cli_main
from mdb.core import custody, settings
from mdb.core.errors import BadArgs


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture()
def here(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()          # the walk-up for a project file stops here
    monkeypatch.chdir(tmp_path)
    custody.clear()
    custody._STATE["env_read"] = custody._STATE["sinks_read"] = False
    yield tmp_path
    custody.clear()
    for sink in custody.sinks():
        sink.close()


# ------------------------------------------------------------------ mdb config


def test_config_init_writes_a_template_that_checks(here, capsys):
    assert cli_main(["config", "init"]) == 0
    user = settings.user_file()
    assert user.is_file() and "schema = \"mdb-config-v1\"" in user.read_text()
    assert cli_main(["config", "init"]) == 2                       # never overwrites
    assert "exists" in capsys.readouterr().err
    assert cli_main(["config", "init", "--project", str(here / "p")]) == 0
    assert (here / "p" / "mdb.toml").is_file()
    capsys.readouterr()
    assert cli_main(["config", "check", "--project", str(here / "p")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ok: ") and "project" in out and "user" in out


def test_config_show_and_explain_name_the_winning_source(here, capsys, monkeypatch):
    _write(settings.user_file(), '[model]\ndtype = "fp32"\n')
    _write(here / "mdb.toml", '[serve]\nport = 9100\n')
    monkeypatch.setenv("MDB_RECORDS", str(here / "r.jsonl"))
    assert cli_main(["config", "show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["settings"]["model.dtype"] == {"value": "fp32",
                                                "source": f"user file {settings.user_file()}"}
    assert shown["settings"]["serve.port"]["source"].startswith("project file")
    assert shown["settings"]["records.sink"]["source"] == \
        "environment MDB_RECORDS (older name of MDB_RECORDS_SINK)"
    assert [f["layer"] for f in shown["files"]] == ["project", "user"]
    assert cli_main(["config", "explain", "model.dtype"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].strip().startswith('=> "fp32"  user file') and lines[3].strip() == \
        '"bf16"  default'
    assert cli_main(["config", "explain", "model.lookahead"]) == 2
    assert "not a setting" in capsys.readouterr().err
    assert cli_main(["config", "explain", "model.nothing"]) == 2


def test_config_check_reports_bad_defaults_and_the_deprecated_storage_file(here, capsys):
    _write(here / "mdb.toml", '[repl]\ndefaults = { reads = { top = 3 } }\n')
    assert cli_main(["config", "check"]) == 2
    err = capsys.readouterr().err
    assert "repl.defaults.reads.top" in err and "not a display or budget default" in err
    _write(here / "mdb.toml", '[repl]\ndefaults = { print = { top = 99 } }\n')
    assert cli_main(["config", "check"]) == 2
    assert "<= 16" in capsys.readouterr().err
    _write(here / "mdb.toml", '[repl]\ndefaults = { print = { top = 9 } }\n')
    (here / ".mdb-storage.json").write_text("{}")
    assert cli_main(["config", "check"]) == 0
    assert "deprecated" in capsys.readouterr().err
    _write(here / "mdb.toml", "[model]\nprefill = 3\n")
    assert cli_main(["config", "check"]) == 2
    assert "'prefill' cannot be set in a config file" in capsys.readouterr().err


# ------------------------------------------------------------------ mdb engine

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _engine(*argv) -> list[dict]:
    from mdb.repl.__main__ import main

    return main(list(argv))


def _json_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def test_engine_reads_config_defaults_and_flags_win(here, capsys):
    project = here / "proj"
    script = _write(here / "s.mdb", "where\nprint L0.h\nx L0.h\n")
    _write(here / ".mdbinit", "# init\nbreak L1.p\n")
    _write(here / "mdb.toml", f'''[model]
dtype = "fp32"
[project]
dir = "{project}"
session = "cfg-session"
[records]
sink = "sinks/"
[repl]
init = ".mdbinit"
defaults = {{ print = {{ top = 7 }}, x = {{ count = 3 }} }}
''')
    assert _engine("tiny:qwen2", "--tokens", "1 5 9 3", "--script", str(script), "--json") == 0
    out = _json_lines(capsys.readouterr().out)
    ops = [line.get("op") for line in out]
    assert ops == ["break", "start", "where", "print", "x"]          # init ran first
    printed, examined = out[3], out[4]
    assert printed["dtype"] == "float32" and printed["k"] == 7
    assert examined["count"] == 3
    from mdb.core.project import Project

    assert [s["session_id"] for s in Project(project).sessions()] == ["cfg-session"]
    opened = [sink.path for sink in custody.sinks()]
    assert len(opened) == 1 and opened[0].parent == (here / "sinks").resolve()
    # flags win; --no-init skips the init file
    capsys.readouterr()
    assert _engine("tiny:qwen2", "--dtype", "bf16", "--no-init", "--session", "other",
                   "--tokens", "1 5 9 3", "--script", str(script), "--json") == 0
    out = _json_lines(capsys.readouterr().out)
    assert [line.get("op") for line in out] == ["start", "where", "print", "x"]
    assert out[2]["dtype"] == "bfloat16" and out[2]["k"] == 7
    assert "other" in [s["session_id"] for s in Project(project).sessions()]


def test_engine_honours_device_for_tiny_models(here, capsys, monkeypatch):
    """--device used to be ignored for tiny models; it is now applied (a bad one fails)."""
    from mdb.ops import debugger

    seen = []
    real = debugger.open_tiny

    def spy(*args, **kwargs):
        seen.append(kwargs.get("device"))
        return real(*args, **kwargs)

    monkeypatch.setattr(debugger, "open_tiny", spy)
    script = _write(here / "s.mdb", "where\n")
    assert _engine("tiny:qwen2", "--device", "cpu", "--tokens", "1 2", "--script",
                   str(script), "--json") == 0
    assert seen == ["cpu"]
    with pytest.raises(RuntimeError):
        real("qwen2", device="no-such-device")


def test_engine_start_takes_lookahead_and_retain_rows(here, capsys):
    script = _write(here / "s.mdb", "start lookahead=layer retain_rows=1 1 5 9 3\nwhere\n")
    assert _engine("tiny:qwen2", "--script", str(script), "--json") == 0
    out = _json_lines(capsys.readouterr().out)
    assert out[1]["lookahead"] == "layer"
    script.write_text("start lookahead=sideways 1 5\n")
    assert _engine("tiny:qwen2", "--script", str(script), "--json") == 1


def test_engine_records_and_custody_flags(here, capsys):
    prereg = _write(here / "prereg.md", "H1\n")
    script = _write(here / "s.mdb", "forecast M[4,9]\n")
    assert _engine("tiny:qwen2", "--tokens", "1 5 9 3", "--records", str(here / "r.jsonl"),
                   "--prereg", str(prereg), "--kernel-path", "stepped", "--script", str(script),
                   "--json") in (0, 1)
    records = custody.Sink(here / "r.jsonl").records()
    assert records, capsys.readouterr()
    held = records[0]["body"]["custody"]
    assert held["prereg_sha256"] == custody.file_sha256(prereg)
    assert held["kernel_path"] == "stepped" and held["script"] == str(script)
    assert "config" not in held                                      # no config file loaded


def test_engine_connect_refuses_local_settings(here, capsys):
    with pytest.raises(SystemExit):
        _engine("http://127.0.0.1:9", "--dtype", "fp32")
    assert "--dtype belong to the server" in capsys.readouterr().err


def test_engine_bad_config_exits_2_naming_the_key(here, capsys):
    _write(here / "mdb.toml", "[model]\ndevice = 3\n")
    assert _engine("tiny:qwen2") == 2
    assert "mdb.toml:model.device" in capsys.readouterr().err


# ------------------------------------------------------------------ model forms


def test_aliases_and_roots_resolve_the_same_everywhere(here, monkeypatch):
    from mdb.core.locate import locate_model, parse_target

    root = here / "models"
    checkpoint = root / "org" / "small-model"
    _write(checkpoint / "config.json", "{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(here / "empty-hub"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(here / "home"))
    _write(here / "mdb.toml", '[model]\nroots = ["models"]\n'
                              'aliases = { sm = "small-model", t = "tiny:llama", '
                              'loop = "sm" }\n')
    settings.reset()
    assert locate_model("small-model") == checkpoint
    assert locate_model("org/small-model") == checkpoint
    assert locate_model("sm") == checkpoint
    assert parse_target("sm") == ("model", "small-model")
    assert parse_target("t") == ("tiny", "llama")
    assert parse_target("http://h:1") == ("connect", "http://h:1")
    with pytest.raises(BadArgs, match="tiny models are"):
        parse_target("tiny:gpt2")
    with pytest.raises(BadArgs, match="names another alias"):
        locate_model("loop")
    monkeypatch.setenv("MDB_MODEL_ROOTS", str(here / "nowhere"))     # env replaces the file's
    settings.reset()
    with pytest.raises(BadArgs, match="no directory and no cached model"):
        locate_model("small-model")


def test_serve_open_debugger_resolves_names_like_engine(here):
    from mdb.serve.live import open_debugger

    with pytest.raises(BadArgs, match="no directory and no cached model"):
        open_debugger(here / "p", model="not-a-model")
    debugger = open_debugger(here / "p", tiny="qwen2", session="named", device="cpu")
    assert debugger.session_id == "named"


def test_serve_takes_project_and_port_from_config(here, capsys, monkeypatch):
    from mdb.serve import __main__ as serve_main

    with pytest.raises(SystemExit):
        serve_main.main([])                                          # no project anywhere
    assert "name the project" in capsys.readouterr().err
    _write(here / "mdb.toml", f'[project]\ndir = "{here / "p"}"\n[serve]\nport = 9123\n')
    monkeypatch.setitem(__import__("sys").modules, "uvicorn", None)  # stop before serving
    assert serve_main.main([]) == 2
    assert "needs the serve extra" in capsys.readouterr().err
    assert serve_main.main(["--port", "0"]) == 2
    assert "--port must be 1..65535" in capsys.readouterr().err


# ------------------------------------------------------------------ selfcheck flags


def test_selfcheck_old_flags_still_work_with_a_notice(here, capsys):
    from mdb.engine.selfcheck import main

    assert main(["--tiny", "qwen2", "--tokens", "1", "--json", str(here / "old.json")]) == 0
    err = capsys.readouterr().err
    assert "--tokens is now --generate" in err and "--json is now --out" in err
    assert json.loads((here / "old.json").read_text())["generated"] == 1
    assert main(["--tiny", "qwen2", "--generate", "1", "--out", str(here / "new.json")]) == 0
    assert "is now" not in capsys.readouterr().err
    assert json.loads((here / "new.json").read_text())["passed"] is True


def test_one_copy_of_each_list():
    """The torch-free lists the front ends use agree with the engine's."""
    from mdb.engine import models, run
    from mdb.ops import builtin

    assert settings.DTYPES == tuple(models.DTYPES)
    assert settings.TINY_FAMILIES == models.TINY_FAMILIES
    assert builtin.LOOKAHEADS == run.LOOKAHEADS


# ------------------------------------------------------------------ Python entry points


def test_python_entry_points_honour_the_config_and_explicit_arguments_win(here):
    from mdb.engine.models import load_local, tiny_model
    from mdb.ops.debugger import attach, open_tiny

    checkpoint = here / "ckpt"
    tiny_model("qwen2", "bf16").save_pretrained(checkpoint)
    _write(here / "mdb.toml", f'''[model]
dtype = "fp32"
aliases = {{ mine = "{checkpoint}" }}
[project]
dir = "proj"
session = "py-session"
[custody]
kernel_path = "stepped"
[records]
sink = "sinks/py.jsonl"
payloads = false
''')
    settings.reset()
    dtype = lambda model: next(model.parameters()).dtype  # noqa: E731
    assert dtype(tiny_model("qwen2")) == torch.float32
    assert dtype(tiny_model("qwen2", "bf16")) == torch.bfloat16
    assert dtype(load_local("mine")) == torch.float32                   # alias + config dtype
    assert dtype(load_local(checkpoint, dtype="bf16")) == torch.bfloat16
    debugger = open_tiny("qwen2")
    assert dtype(debugger.model) == torch.float32
    assert debugger.project is not None and debugger.session_id == "py-session"
    assert (here / "proj").resolve() == Path(debugger.project.root).resolve()
    other = open_tiny("qwen2", "bf16", project=False)                   # explicit: no project
    assert dtype(other.model) == torch.bfloat16 and other.project is None
    named = attach(tiny_model("qwen3"), project=here / "elsewhere", session="mine")
    assert named.session_id == "mine"
    assert custody.declare()["kernel_path"] == "stepped"
    assert custody.declare(kernel_path="batched")["kernel_path"] == "batched"
    sink = debugger.records_to()
    assert sink.path == (here / "sinks" / "py.jsonl").resolve() and not sink.payloads_on
    explicit = debugger.records_to(here / "own.jsonl", payloads=True)
    assert explicit.path == (here / "own.jsonl").absolute() and explicit.payloads_on


def _hello(env_extra, tmp):
    import os
    import re
    import subprocess
    import sys

    example = Path(__file__).resolve().parents[1] / "examples" / "00-hello-tiny" / "hello_tiny.py"
    src = Path(__file__).resolve().parents[1] / "src"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MDB_")}
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(PYTHONPATH=os.pathsep.join([str(src), str(src.parents[2] / "src")]),
               XDG_CONFIG_HOME=str(tmp), PYTHONDONTWRITEBYTECODE="1", **env_extra)
    done = subprocess.run([sys.executable, str(example)], capture_output=True, text=True,
                          env=env, cwd=tmp, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return re.findall(r"record ([0-9a-f]{16})", done.stdout)


def test_hello_tiny_id_unchanged_by_a_locations_file_and_changed_by_a_recorded_key(tmp_path):
    """Measured in the development venv: d0c1d6c733c4bfa1 (torch 2.13, transformers 5.13.1)
    with no config and with a serve-only file; the id depends on library versions, so the test
    compares runs rather than the literal."""
    plain = _hello({}, tmp_path / "none")
    serve_only = _write(tmp_path / "serve.toml", "[serve]\nport = 9000\n[repl]\nhistory = 'h'\n")
    located = _hello({"MDB_CONFIG": str(serve_only)}, tmp_path / "serve")
    dtype_file = _write(tmp_path / "dtype.toml", '[model]\ndtype = "bf16"\n')
    recorded = _hello({"MDB_CONFIG": str(dtype_file)}, tmp_path / "dtype")
    assert plain and located == plain
    assert recorded and recorded != plain               # same model, custody block added

"""Regression tests for the config audit findings (C1, C2, H1, M1-M5, L1-L7): what a project
file (possibly from a cloned repository) may set, the provenance of REPL init files and
defaults, history safety, and the settings snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdb.core import custody, settings
from mdb.core.errors import BadArgs
from mdb.core.settings import ConfigError


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    """A repository directory as the working directory; custody as in a new process."""
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    monkeypatch.chdir(root)
    custody.clear()
    custody.reread()
    yield root
    custody.clear()
    for sink in custody.sinks():
        sink.close()


def _resolve(**kw):
    settings.reset()
    return settings.current()


# ------------------------------------------------------------------ M1 / M5 / H1: project files


@pytest.mark.parametrize("text,key", [
    ('[records]\nsink = "~/.victim2"\nappend = true\n', "records.sink"),
    ('[records]\nsink = "../outside/r.jsonl"\n', "records.sink"),
    ('[storage]\nshared_tensor_store = "/elsewhere/pool"\n', "storage.shared_tensor_store"),
    ('[project]\ndir = "../../somewhere"\n', "project.dir"),
    ('[repl]\ninit = "/etc/mdbinit"\n', "repl.init"),
])
def test_project_file_paths_must_stay_inside_the_project(repo, text, key):
    _write(repo / "mdb.toml", text)
    with pytest.raises(ConfigError, match=f"mdb.toml:{key}: .* is outside the project"):
        _resolve()


def test_project_file_paths_inside_the_tree_are_fine(repo):
    _write(repo / "mdb.toml", '[records]\nsink = "out/{run}.jsonl"\n[project]\ndir = "runs/a"\n'
                              '[storage]\nshared_tensor_store = "pool"\n')
    got = _resolve()
    assert got.get("records.sink") == str(repo.resolve() / "out/{run}.jsonl")


@pytest.mark.parametrize("text,key", [
    ('[repl]\nhistory = "~/.victimrc"\n', "repl.history"),
    ('[serve]\nallow_hosts = ["evil.example"]\n', "serve.allow_hosts"),
    ('[repl]\ntrusted_projects = ["."]\n', "repl.trusted_projects"),
])
def test_project_file_may_not_set_user_only_keys(repo, text, key):
    _write(repo / "mdb.toml", text)
    with pytest.raises(ConfigError, match=f"mdb.toml:{key}: a project file may not set this"):
        _resolve()


def test_user_file_and_environment_may_set_them(repo, monkeypatch):
    _write(settings.user_file(), '[repl]\nhistory = "~/.mdb_history"\n'
                                 '[serve]\nallow_hosts = ["mdb.lan"]\n')
    got = _resolve()
    assert got.get("serve.allow_hosts") == ("mdb.lan",)
    monkeypatch.setenv("MDB_RECORDS_SINK", "/anywhere/r.jsonl")       # env is not confined
    assert _resolve().get("records.sink") == "/anywhere/r.jsonl"


def test_history_never_overwrites_a_file_that_is_not_history(tmp_path, monkeypatch):
    """H1: libedit refuses to read a non-history file; the shell must then not write it."""
    readline = pytest.importorskip("readline")
    from mdb.repl.shell import Shell

    victim = tmp_path / "victimrc"
    victim.write_text("export PATH=/usr/bin\nalias ll='ls -l'\n")
    original = victim.read_text()

    class Tty:
        def isatty(self):
            return True

    def run(history):
        shell = Shell(object(), history=history, out=open(tmp_path / "out.txt", "w"))
        monkeypatch.setattr("sys.stdin", Tty())
        monkeypatch.setattr("builtins.input", lambda prompt: "quit")
        shell.loop(__import__("sys").stdin)
        return shell

    try:
        readline.read_history_file(str(victim))
        readable = True
    except OSError:
        readable = False
    run(victim)
    if not readable:
        assert victim.read_text() == original
    fresh = tmp_path / "state" / "history"
    run(fresh)
    assert fresh.is_file()                                             # a new history file


# ------------------------------------------------------------------ C1: init files

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _engine(*argv):
    from mdb.repl.__main__ import main

    return main(list(argv))


def _symbol_table(sink: Path) -> dict:
    return [r for r in custody.Sink(sink).records() if r["kind"] == "symbol-table"][0]


def test_project_init_runs_only_when_trusted_and_is_recorded(repo, capsys):
    script = _write(repo / "s2.mdb", "run 1 5 9 3 7 2\nsymbols\n")
    init = _write(repo / ".mdbinit", "break t5.logits\n")
    _write(repo / "mdb.toml", '[repl]\ninit = ".mdbinit"\n')
    assert _engine("tiny:qwen2", "--dtype", "fp32", "--script", str(script), "--records",
                   str(repo / "a.jsonl"), "--kernel-path", "stepped") == 0
    assert "not running the init file" in capsys.readouterr().err
    untrusted = _symbol_table(repo / "a.jsonl")
    assert "init" not in untrusted["body"]["custody"]
    # trusted by the user file: it runs, and custody names its path and sha256
    _write(settings.user_file(), f'[repl]\ntrusted_projects = ["{repo}"]\n')
    custody.clear()
    custody.reread()
    assert _engine("tiny:qwen2", "--dtype", "fp32", "--script", str(script), "--records",
                   str(repo / "b.jsonl"), "--kernel-path", "stepped") == 0
    trusted = _symbol_table(repo / "b.jsonl")
    assert trusted["body"]["custody"]["init"] == [
        {"path": str(init.resolve()), "sha256": custody.file_sha256(init)}]
    assert trusted["id"] != untrusted["id"]


def test_init_from_the_command_line_is_recorded(repo, capsys):
    script = _write(repo / "s.mdb", "run 1 5 9 3 7 2\nsymbols\n")
    init = _write(repo / "mine.mdbinit", "break t5.logits\n")
    assert _engine("tiny:qwen2", "--init", str(init), "--script", str(script), "--records",
                   str(repo / "r.jsonl")) == 0
    held = _symbol_table(repo / "r.jsonl")["body"]["custody"]
    assert held["init"][0]["sha256"] == custody.file_sha256(init)


def test_init_file_may_not_pass_verdict_arguments(repo, capsys):
    """C1 case B: an init file that starts with prefill/lookahead/retain_rows is refused."""
    init = _write(repo / "i.mdbinit", "start prefill=0 lookahead=layer retain_rows=1 1 5 9 3\n")
    nested = _write(repo / "n.mdbinit", f"source {init}\n")
    script = _write(repo / "s.mdb", "where\n")
    for path in (init, nested):
        capsys.readouterr()
        _engine("tiny:qwen2", "--init", str(path), "--script", str(script), "--json")
        out = capsys.readouterr().out
        assert "an init file may not pass lookahead=, prefill=, retain_rows=" in out
    # the same line in the script itself is fine (custody hashes the script)
    script.write_text("start prefill=0 lookahead=layer retain_rows=1 1 5 9 3\n")
    assert _engine("tiny:qwen2", "--script", str(script), "--json") == 0


# ------------------------------------------------------------------ C2: repl.defaults


def test_repl_defaults_that_change_results_are_not_configurable(repo, capsys):
    _write(repo / "mdb.toml", "[repl]\ndefaults = { symbols = { limit = 1 } }\n")
    assert _engine("tiny:qwen2", "--tokens", "1 2") == 2
    assert "symbols.limit is not a display default" in capsys.readouterr().err
    _write(repo / "mdb.toml", "[repl]\ndefaults = { continue = { budget = 2 } }\n")
    assert _engine("tiny:qwen2", "--tokens", "1 2") == 2
    assert "continue.budget is not a display default" in capsys.readouterr().err
    from mdb.ops import registry

    assert sorted(f"{o.name}.{a.name}" for o in registry().values() for a in o.args
                  if a.configurable) == ["checkpoints.limit", "info.limit", "print.slice",
                                         "print.top", "sweep.max_calls", "x.count"]


def test_repl_defaults_are_recorded(repo):
    _write(repo / "mdb.toml", "[repl]\ndefaults = { print = { top = 7 } }\n")
    block = _resolve().provenance()
    assert block["sources"][0]["settings"] == {"repl.defaults": '{"print.top":7}'}


# ------------------------------------------------------------------ M2: aliases


@pytest.mark.parametrize("name", ["tiny:qwen2", "http://x", "https://x"])
def test_alias_may_not_shadow_tiny_or_addresses(repo, name):
    _write(repo / "mdb.toml", f'[model]\naliases = {{ "{name}" = "http://203.0.113.5:8765" }}\n')
    with pytest.raises(ConfigError, match="may not shadow"):
        _resolve()


def test_alias_use_is_printed(repo, capsys):
    _write(repo / "mdb.toml", '[model]\naliases = { demo = "tiny:qwen3" }\n')
    script = _write(repo / "s.mdb", "where\n")
    assert _engine("demo", "--tokens", "1 2", "--script", str(script), "--json") == 0
    assert "alias demo -> tiny:qwen3 (project file" in capsys.readouterr().err


# ------------------------------------------------------------------ M3: discovery


def test_outside_repo_and_home_only_the_working_directory_is_read(tmp_path, monkeypatch):
    walk = tmp_path / "walk"
    _write(walk / "mdb.toml", '[model]\ndtype = "fp32"\n')
    deep = walk / "a" / "b"
    deep.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert settings.find_project_file(deep) is None
    assert settings.find_project_file(walk)[0] == walk / "mdb.toml"


def test_a_file_another_user_owns_is_skipped(tmp_path, monkeypatch):
    root = tmp_path / "r"
    (root / ".git").mkdir(parents=True)
    _write(root / "mdb.toml", '[model]\ndtype = "fp32"\n')
    monkeypatch.setattr(settings, "_owned", lambda path: False)
    settings.SKIPPED.clear()
    assert settings.find_project_file(root) is None
    assert "not owned by the current user" in settings.SKIPPED[0]


# ------------------------------------------------------------------ M4: snapshot


def test_a_file_edited_after_load_does_not_change_the_block(repo):
    from mdb.ops.debugger import open_tiny

    path = _write(repo / "mdb.toml", '[model]\ndtype = "bf16"\n')
    sha = custody.file_sha256(path)
    debugger = open_tiny("qwen2")
    path.write_text('[model]\ndtype = "fp32"\n')
    block = custody.current()["config"]
    assert block["resolved"]["model.dtype"] == '"bf16"'
    assert block["sources"][0]["sha256"] == sha
    assert next(debugger.model.parameters()).dtype == torch.bfloat16


# ------------------------------------------------------------------ M5: serve


def test_allow_remote_needs_a_typed_allow_host(repo, capsys, monkeypatch):
    from mdb.serve import __main__ as serve_main

    _write(settings.user_file(), '[serve]\nallow_hosts = ["mdb.lan"]\n')
    assert serve_main.main(["--project", str(repo / "p"), "--host", "0.0.0.0",
                            "--allow-remote"]) == 2
    assert "--allow-remote needs --allow-host" in capsys.readouterr().err


# ------------------------------------------------------------------ Lows


def test_unrelated_broken_pyproject_is_skipped_but_a_broken_tool_mdb_is_not(repo):
    _write(repo / "pyproject.toml", '[project\nname = "x"\n')
    assert _resolve().files == []
    custody.reread()
    custody.current()                                                  # seals still work
    _write(repo / "pyproject.toml", '[tool.mdb\nmodel = 1\n')
    with pytest.raises(ConfigError, match="not valid TOML"):
        _resolve()


def test_no_records_append_overrides_a_config_file(repo, capsys):
    foreign = _write(repo / "r.jsonl", '{"x": 1}\n')
    _write(repo / "mdb.toml", '[records]\nsink = "r.jsonl"\nappend = true\n')
    assert _engine("tiny:qwen2", "--no-records-append", "--tokens", "1 2") == 2
    assert "already holds records" in capsys.readouterr().err
    assert foreign.read_text() == '{"x": 1}\n'


def test_call_arguments_that_override_the_config_are_listed(repo):
    from mdb.engine.models import tiny_model

    _write(repo / "mdb.toml", '[model]\ndtype = "bf16"\n')
    settings.reset()
    tiny_model("qwen2", "fp32")
    block = custody.current()["config"]
    assert block["resolved"]["model.dtype"] == '"bf16"'
    assert block["call_arguments"] == {"model.dtype": ['"fp32"']}


def test_declare_after_a_refused_sink_still_raises(repo, monkeypatch):
    """L6: declare()/clear() used to clear the stored sink error, so records then sealed with no
    sink file and no error."""
    from mdb.core.records import seal_record

    def _seal():
        return seal_record(session_id="s", kind="supplied-forecast",
                           model_identity={"family": "qwen2", "parameters_sha256": "0" * 64},
                           execution={"device": "cpu"}, authority="supplied",
                           body={"schema": "x", "target": {"row": 1}, "edit": {},
                                 "supplied": {"value": 1.0, "authority": "supplied"},
                                 "library": {"name": "t"}, "address_map": {},
                                 "sealed_at": {"where": "t1.logits"}})

    _write(repo / "taken.jsonl", '{"x": 1}\n')
    monkeypatch.setenv("MDB_RECORDS_SINK", str(repo / "taken.jsonl"))
    custody.reread()
    with pytest.raises(FileExistsError):
        _seal()
    custody.declare(kernel_path="k")
    with pytest.raises(FileExistsError):
        _seal()
    custody.clear()
    with pytest.raises(FileExistsError):
        _seal()


def test_tests_do_not_see_the_real_home():
    assert Path.home() != Path("/Users/jakeholl") and "xdg-config" in str(settings.user_file())


def test_unknown_device_is_badargs_not_a_traceback():
    from mdb.engine.models import check_device

    with pytest.raises(BadArgs):
        check_device("bogus")
    assert json.dumps(str(check_device("cpu"))) == '"cpu"'

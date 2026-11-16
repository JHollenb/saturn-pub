"""``mdb.core.settings``: precedence, every section, strict validation, refused verdict keys,
older environment names, the custody provenance block and unchanged record ids without a
config file. The autouse fixture in conftest.py points the user file at an empty directory."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mdb.core import custody, settings
from mdb.core.records import seal_record
from mdb.core.settings import ConfigError, resolve

IDENTITY = {"family": "qwen2", "parameters_sha256": "0" * 64}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture()
def place(tmp_path, monkeypatch):
    """A repository (``.git``) with a working directory inside it and an empty user dir."""
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    work = repo / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    return {"tmp": tmp_path, "repo": repo, "work": work, "user": tmp_path / "xdg/mdb/config.toml"}


def _env(place, **extra):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("MDB_") and key != "LLM_MODELS_EXTRA_ROOT"}
    env["XDG_CONFIG_HOME"] = str(place["tmp"] / "xdg")
    env.update(extra)
    return env


# ------------------------------------------------------------------ precedence


def test_precedence_cli_env_named_project_user_default(place):
    tmp = place["tmp"]
    assert resolve(env=_env(place), cwd=place["work"]).origin("model.dtype") == "default"
    _write(place["user"], '[model]\ndtype = "fp32"\ndevice = "cpu"\n')
    _write(place["repo"] / "mdb.toml", '[model]\ndtype = "bf16"\n')
    named = _write(tmp / "named.toml", '[model]\ndtype = "fp32"\n')
    layers = [
        ({}, {}, "bf16", "project file"),
        ({"MDB_CONFIG": str(named)}, {}, "fp32", "MDB_CONFIG file"),
        ({"MDB_CONFIG": str(named), "MDB_MODEL_DTYPE": "bf16"}, {}, "bf16", "environment"),
        ({"MDB_MODEL_DTYPE": "bf16"}, {"model.dtype": "fp32"}, "fp32", "command line"),
    ]
    for extra, cli, value, source in layers:
        got = resolve(cli, flags={"model.dtype": "--dtype"}, env=_env(place, **extra),
                      cwd=place["work"])
        assert got.get("model.dtype") == value
        assert got.origin("model.dtype").startswith(source), got.origin("model.dtype")
    # explain lists every layer that sets it, highest first, the first winning
    got = resolve({"model.dtype": "fp32"}, flags={"model.dtype": "--dtype"},
                  env=_env(place, MDB_CONFIG=str(named), MDB_MODEL_DTYPE="bf16"),
                  cwd=place["work"])
    sources = [item["source"].split(" ")[0] for item in got.explain("model.dtype")]
    assert sources == ["command", "environment", "MDB_CONFIG", "project", "user", "default"]
    assert [item["wins"] for item in got.explain("model.dtype")] == [True] + [False] * 5
    assert got.origin("model.device").startswith("user file")


def test_project_file_discovery(place):
    repo, work = place["repo"], place["work"]
    _write(repo.parent / "mdb.toml", '[model]\ndtype = "fp32"\n')   # above the repository root
    assert resolve(env=_env(place), cwd=work).files == []
    _write(repo / "pyproject.toml", '[project]\nname = "x"\n')           # no [tool.mdb]: skipped
    assert resolve(env=_env(place), cwd=work).files == []
    _write(repo / "pyproject.toml", '[project]\nname = "x"\n[tool.mdb.model]\ndtype = "fp32"\n')
    found = resolve(env=_env(place), cwd=work)
    assert found.get("model.dtype") == "fp32" and found.files[0].label == "project"
    _write(work / "mdb.toml", '[model]\ndevice = "meta"\n')               # nearer wins, alone
    found = resolve(env=_env(place), cwd=work)
    assert found.get("model.device") == "meta" and found.get("model.dtype") == "bf16"
    # a named project: its mdb.toml, no walk-up
    project = _write(place["tmp"] / "proj" / "mdb.toml", '[model]\ndtype = "fp32"\n').parent
    found = resolve(project=project, env=_env(place), cwd=work)
    assert [f.path for f in found.files] == [(project / "mdb.toml").resolve()]
    found = resolve(env=_env(place, MDB_PROJECT_DIR=str(project)), cwd=work)
    assert found.get("model.dtype") == "fp32"


def test_xdg_config_home_and_home_fallback(place, monkeypatch):
    _write(place["user"], '[serve]\nport = 9000\n')
    assert resolve(env=_env(place), cwd=place["work"]).get("serve.port") == 9000
    home = place["tmp"] / "home"
    _write(home / ".config/mdb/config.toml", '[serve]\nport = 9001\n')
    monkeypatch.setenv("HOME", str(home))
    env = _env(place)
    env.pop("XDG_CONFIG_HOME")
    assert settings.user_file(env) == home / ".config/mdb/config.toml"
    assert resolve(env=env, cwd=place["work"]).get("serve.port") == 9001


# ------------------------------------------------------------------ every section


def test_every_section_parses_with_paths_relative_to_the_file(place):
    base = place["tmp"] / "cfg"
    (base / "models").mkdir(parents=True)
    path = _write(base / "c.toml", f'''schema = "{settings.SCHEMA}"
[model]
roots = ["models", "/abs/models"]
aliases = {{ local = "models", q15 = "Qwen/Qwen2.5-1.5B", t = "tiny:qwen3" }}
dtype = "fp32"
device = "cpu"
[project]
dir = "runs/main"
session = "s1"
[records]
sink = "sinks/"
append = true
payloads = false
[custody]
kernel_path = "stepped"
[serve]
host = "0.0.0.0"
port = 9100
allow_hosts = ["mdb.lan"]
[repl]
init = ".mdbinit"
history = "~/hist"
defaults = {{ print = {{ top = 8 }}, "x.count" = 32 }}
[storage]
shared_tensor_store = "store"
[sae]
sources = {{ l12 = {{ path = "sae/l12", k = 50, layers = "0:2", port = "h", verify = true }} }}
''')
    got = resolve(env=_env(place, MDB_CONFIG=str(path)), cwd=place["work"]).as_dict()
    b = str(base.resolve())
    assert got["model.roots"] == (f"{b}/models", "/abs/models")
    assert got["model.aliases"] == {"local": f"{b}/models", "q15": "Qwen/Qwen2.5-1.5B",
                                    "t": "tiny:qwen3"}
    assert (got["model.dtype"], got["model.device"]) == ("fp32", "cpu")
    assert (got["project.dir"], got["project.session"]) == (f"{b}/runs/main", "s1")
    assert got["records.sink"] == f"{b}/sinks/"            # the trailing separator: a directory
    assert (got["records.append"], got["records.payloads"]) == (True, False)
    assert got["custody.kernel_path"] == "stepped"
    assert (got["serve.host"], got["serve.port"], got["serve.allow_hosts"]) == \
        ("0.0.0.0", 9100, ("mdb.lan",))
    assert got["repl.init"] == f"{b}/.mdbinit"
    assert got["repl.history"] == str(Path("~/hist").expanduser())
    assert got["repl.defaults"] == {"print.top": 8, "x.count": 32}
    assert got["storage.shared_tensor_store"] == f"{b}/store"
    assert got["sae.sources"] == {"l12": {"path": f"{b}/sae/l12", "k": 50, "layers": "0:2",
                                          "port": "h", "verify": True}}


@pytest.mark.parametrize("text,where", [
    ("[modle]\ndtype='bf16'\n", "c.toml:modle: unknown key"),
    ("[model]\ndtyp='bf16'\n", "c.toml:model.dtyp: unknown key"),
    ("[model]\ndtype='fp16'\n", "c.toml:model.dtype: expected one of bf16, fp32"),
    ("[model]\ndtype=3\n", "c.toml:model.dtype: expected one of bf16, fp32"),
    ("[serve]\nport='80'\n", "c.toml:serve.port: expected an integer"),
    ("[serve]\nport=70000\n", "c.toml:serve.port: expected a port"),
    ("[records]\nappend='yes'\n", "c.toml:records.append: expected true or false"),
    ("[model]\nroots='/x'\n", "c.toml:model.roots: expected a list of paths"),
    ("model = 1\n", "c.toml:model: expected a [model] table"),
    ("schema = 'mdb-config-v9'\n", "c.toml:schema: expected 'mdb-config-v1'"),
    ("[repl]\ndefaults = { print = { top = 'x' } }\n", "repl.defaults.print.top: expected a "
                                                       "number"),
    ("[sae]\nsources = { a = { k = 3 } }\n", "sae.sources.a: missing path"),
    ("[sae]\nsources = { a = { path = 'p', size = 3 } }\n", "sae.sources.a.size: unknown key"),
    ("[sae]\nsources = { a = { path = 'p', port = 'q' } }\n", "sae.sources.a.port: expected "
                                                             "one of h_out, h"),
    ("[model\n", "c.toml: not valid TOML"),
])
def test_unknown_keys_and_wrong_types_name_file_and_key(place, text, where):
    path = _write(place["tmp"] / "c.toml", text)
    with pytest.raises(ConfigError) as caught:
        resolve(env=_env(place, MDB_CONFIG=str(path)), cwd=place["work"])
    assert where in str(caught.value), str(caught.value)


@pytest.mark.parametrize("text,name", [
    ("[model]\nlookahead = 'layer'\n", "lookahead"),
    ("[records]\nretain_rows = 4\n", "retain_rows"),
    ("[project]\nprefill = 2\n", "prefill"),
    ("[custody]\nprereg = 'p.md'\n", "prereg"),
    ("[custody]\nfloor_spacings = 2.0\n", "floor_spacings"),
    ("[custody]\nmargin_floor_spacings = 3.0\n", "margin_floor_spacings"),
    ("[branches]\nrecheck = true\n", "branches"),
    ("[repl]\nrecheck = 'all'\n", "recheck"),
    ("[repl]\nmode = 'batched'\n", "mode"),
    ("[repl]\nbatch = 8\n", "batch"),
    ("[model]\nseed = 1\n", "seed"),
    ("[serve]\nallow_remote = true\n", "allow_remote"),
    ("[repl]\ndefaults = { start = { lookahead = 'layer' } }\n", "lookahead"),
    ("[sae]\nsources = { a = { path = 'p', recheck = true } }\n", "recheck"),
])
def test_verdict_settings_are_refused_by_name_with_the_reason(place, text, name):
    path = _write(place["tmp"] / "c.toml", text)
    with pytest.raises(ConfigError) as caught:
        resolve(env=_env(place, MDB_CONFIG=str(path)), cwd=place["work"])
    message = str(caught.value)
    assert f"{name!r} cannot be set in a config file" in message, message
    assert settings.REFUSED[name] in message


def test_missing_named_config_file_is_an_error(place):
    with pytest.raises(ConfigError, match="no such config file"):
        resolve(env=_env(place, MDB_CONFIG=str(place["tmp"] / "absent.toml")), cwd=place["work"])


# ------------------------------------------------------------------ environment


def test_older_environment_names_are_aliases(place):
    old = {"MDB_RECORDS": "/r/x.jsonl", "MDB_KERNEL_PATH": "batched",
           "MDB_SHARED_TENSOR_STORE": "/pool", "LLM_MODELS_EXTRA_ROOT": "/m1" + os.pathsep + "/m2"}
    got = resolve(env=_env(place, **old), cwd=place["work"])
    assert got.get("records.sink") == "/r/x.jsonl"
    assert got.get("custody.kernel_path") == "batched"
    assert got.get("storage.shared_tensor_store") == "/pool"
    assert got.get("model.roots") == ("/m1", "/m2")
    assert "older name of MDB_RECORDS_SINK" in got.origin("records.sink")
    same = resolve(env=_env(place, MDB_RECORDS="/r/x.jsonl", MDB_RECORDS_SINK="/r/x.jsonl"),
                   cwd=place["work"])
    assert same.origin("records.sink") == "environment MDB_RECORDS_SINK"
    with pytest.raises(ConfigError, match="twice with different values"):
        resolve(env=_env(place, MDB_RECORDS="/a", MDB_RECORDS_SINK="/b"), cwd=place["work"])


def test_environment_values_are_typed(place):
    for text, want in (("1", True), ("yes", True), ("TRUE", True), ("0", False), ("no", False)):
        assert resolve(env=_env(place, MDB_RECORDS_APPEND=text),
                       cwd=place["work"]).get("records.append") is want
    with pytest.raises(ConfigError, match="MDB_RECORDS_APPEND='maybe'"):
        resolve(env=_env(place, MDB_RECORDS_APPEND="maybe"), cwd=place["work"])
    with pytest.raises(ConfigError, match="MDB_SERVE_PORT"):
        resolve(env=_env(place, MDB_SERVE_PORT="eighty"), cwd=place["work"])
    with pytest.raises(ConfigError, match="MDB_MODEL_DTYPE"):
        resolve(env=_env(place, MDB_MODEL_DTYPE="fp16"), cwd=place["work"])
    assert resolve(env=_env(place, MDB_SERVE_ALLOW_HOSTS="a, b:9"),
                   cwd=place["work"]).get("serve.allow_hosts") == ("a", "b:9")


# ------------------------------------------------------------------ storage


def test_storage_setting_is_passed_to_saturn_pub_without_touching_the_environment(place,
                                                                                    monkeypatch):
    from saturn_pub.shared_store import configure, configured_store, shared_store_root

    pool = place["tmp"] / "pool"
    pool.mkdir()
    scope = place["tmp"] / "scope"
    configure(scope, place["tmp"] / "legacy-pool")                 # the old .mdb-storage.json
    assert settings.legacy_storage_file(scope / "a" / "b") == scope / ".mdb-storage.json"
    assert configured_store(scope / "x.npy").root == (place["tmp"] / "legacy-pool").resolve()
    named = _write(place["tmp"] / "c.toml", f'[storage]\nshared_tensor_store = "{pool}"\n')
    monkeypatch.setenv("MDB_CONFIG", str(named))
    before = dict(os.environ)
    settings.reset()
    settings.current()
    assert dict(os.environ) == before                              # no environment mutation
    assert shared_store_root() == pool
    assert configured_store(scope / "x.npy").root == pool.resolve()   # config over the old file
    settings.reset()                                                   # cleared again
    assert shared_store_root() is None
    monkeypatch.setenv("MDB_SHARED_TENSOR_STORE", "/elsewhere")       # the environment wins
    settings.current()
    assert shared_store_root() == Path("/elsewhere")
    assert "older name" in settings.current().origin("storage.shared_tensor_store")


# ------------------------------------------------------------------ custody provenance


def _seal():
    return seal_record(session_id="s", kind="supplied-forecast", model_identity=IDENTITY,
                       execution={"device": "cpu"}, authority="supplied",
                       body={"schema": "x", "target": {"row": 1}, "edit": {},
                             "supplied": {"value": 1.0, "authority": "supplied"},
                             "library": {"name": "t"}, "address_map": {},
                             "sealed_at": {"where": "t1.logits"}})


@pytest.fixture()
def fresh_custody():
    """As in a new process: nothing declared, environment and sink not read yet."""
    custody.clear()
    custody._STATE["env_read"] = custody._STATE["sinks_read"] = False
    for sink in custody.sinks():
        sink.close()
    yield
    custody.clear()
    for sink in custody.sinks():
        sink.close()


def test_no_config_file_leaves_records_unchanged_and_a_file_is_recorded(place, monkeypatch,
                                                                        fresh_custody):
    plain = _seal()
    assert "custody" not in plain["body"]
    path = _write(place["tmp"] / "c.toml",
                  '[model]\ndtype = "fp32"\n[custody]\nkernel_path = "stepped"\n'
                  '[serve]\nport = 9000\n')
    monkeypatch.setenv("MDB_CONFIG", str(path))
    custody._STATE["env_read"] = False
    held = _seal()
    block = held["body"]["custody"]
    assert held["id"] != plain["id"]
    assert block["kernel_path"] == "stepped"
    config = block["config"]
    assert config["sources"] == [{
        "path": str(path.resolve()), "layer": "MDB_CONFIG",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "settings": {"model.dtype": "fp32", "custody.kernel_path": "stepped"}}]
    assert config["resolved"] == {"model.dtype": "fp32", "custody.kernel_path": "stepped"}
    assert config["origins"]["model.dtype"].startswith("MDB_CONFIG file")
    monkeypatch.delenv("MDB_CONFIG")
    custody._STATE["env_read"] = False
    custody.clear()
    custody._STATE["env_read"] = False
    again = _seal()
    assert again["id"] == plain["id"] and "custody" not in again["body"]


@pytest.mark.parametrize("text", [
    '[serve]\nport = 9000\nallow_hosts = ["mdb.lan"]\n',
    '[repl]\nhistory = "h"\ndefaults = { print = { top = 8 } }\n',
    '[records]\npayloads = false\nappend = true\n',
    '[project]\ndir = "runs/x"\n',
])
def test_a_file_of_locations_only_leaves_ids_unchanged(place, monkeypatch, fresh_custody, text):
    plain = _seal()
    path = _write(place["tmp"] / "c.toml", text)
    monkeypatch.setenv("MDB_CONFIG", str(path))
    custody._STATE["env_read"] = False
    held = _seal()
    assert settings.current().files and settings.current().provenance() is None
    assert "custody" not in held["body"] and held["id"] == plain["id"]


def test_every_recorded_key_is_listed_by_the_file_that_sets_it(place, monkeypatch):
    (place["tmp"] / "m").mkdir()
    _write(place["user"], '[model]\ndevice = "cpu"\n[serve]\nport = 9000\n')
    named = _write(place["tmp"] / "c.toml", '[model]\nroots = ["m"]\naliases = { a = "b" }\n'
                   '[project]\nsession = "s1"\n[sae]\nsources = { l = { path = "p" } }\n')
    _write(place["work"] / "mdb.toml", '[repl]\nhistory = "h"\n')     # no recorded key
    got = resolve({"model.device": "meta"}, env=_env(place, MDB_CONFIG=str(named)),
                  cwd=place["work"]).provenance()
    assert [(s["layer"], sorted(s["settings"])) for s in got["sources"]] == [
        ("MDB_CONFIG", ["model.aliases", "model.roots", "project.session", "sae.sources"]),
        ("user", ["model.device"])]
    assert got["resolved"]["model.device"] == "meta"                    # the flag won
    assert got["origins"]["model.device"] == "command line model.device"
    assert got["resolved"]["model.roots"] == [str((place["tmp"] / "m").resolve())]
    assert set(got["resolved"]) == {"model.roots", "model.aliases", "model.device",
                                    "project.session", "sae.sources"}


def test_configured_sink_opens_even_when_custody_is_declared_in_code_first(place, monkeypatch,
                                                                           fresh_custody):
    sink = place["tmp"] / "sinks" / "r.jsonl"
    monkeypatch.setenv("MDB_RECORDS_SINK", str(sink))
    custody.declare(kernel_path="k")             # before the first seal (it used to skip the sink)
    record = _seal()
    assert [r["id"] for r in custody.Sink(sink).records()] == [record["id"]]


def test_config_sink_with_payloads_off(place, monkeypatch, fresh_custody):
    from mdb.core.bounded import put_json

    path = _write(place["tmp"] / "c.toml", '[records]\nsink = "out/r.jsonl"\npayloads = false\n')
    monkeypatch.setenv("MDB_CONFIG", str(path))
    custody._STATE["env_read"] = False
    put_json([{"x": list(range(10))}])
    _seal()
    sink = custody.Sink(place["tmp"] / "out" / "r.jsonl")
    assert len(sink.records()) == 1 and sink.payloads() == {}

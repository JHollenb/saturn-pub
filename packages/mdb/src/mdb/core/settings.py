"""mdb settings: every user setting's default, config-file key and environment variable, in one
table, and how a value is resolved (standard library only; no model framework).

Precedence, highest first:

1. the command line (``mdb engine --dtype fp32``);
2. the environment (``MDB_MODEL_DTYPE=fp32``; older names such as ``MDB_RECORDS`` still work);
3. the file named by ``MDB_CONFIG``;
4. the project file: ``<project>/mdb.toml`` when a project is named (``--project DIR`` or
   ``MDB_PROJECT_DIR``), else the first ``mdb.toml`` or ``pyproject.toml`` with a ``[tool.mdb]``
   table found walking up from the working directory (stopping at the repository root);
5. the user file ``$XDG_CONFIG_HOME/mdb/config.toml`` (default ``~/.config/mdb/config.toml``);
6. the defaults in ``KEYS``.

A config file holds locations and machine defaults. Settings that change a verdict (floors,
``recheck``, ``lookahead``, ``prefill``, ``retain_rows``, branch ``mode``/``batch``, a
preregistration, seeds) are refused by name: they stay in the call, where the script custody
hashes shows them. Unknown keys and wrong types are errors that name ``file:key``.

When a loaded file sets a key that reaches a record (``Key.recorded``), ``provenance()`` is the
block custody adds to every record: those files' paths, sha256 and recorded keys, and the
resolved values. With no such file nothing is added and record ids are unchanged. Python entry
points read ``value(name, given)``: an explicit argument wins, None takes the setting.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "mdb-config-v1"
PROVENANCE_SCHEMA = "mdb-config-provenance-v1"
FILE_NAME = "mdb.toml"

# The one copy of these lists in mdb: ``engine.models`` (torch) and every front end import them.
TINY_FAMILIES = ("qwen2", "qwen3", "llama", "mistral")
DTYPES = ("bf16", "fp32")
DEFAULT_DTYPE = "bf16"
DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 113
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
SAE_PORTS = ("h_out", "h")


class ConfigError(ValueError):
    """A config file, environment variable or flag that cannot be used (names where)."""


@dataclass(frozen=True)
class Key:
    """One setting. ``kind``: str, int, bool, path, paths, strs, aliases, defaults, sae."""

    name: str
    kind: str
    default: Any = None
    env: tuple[str, ...] = ()          # current name first, then older names still read
    choices: tuple[str, ...] = ()
    recorded: bool = False             # it reaches a record: which model, dtype, device, session,
                                       # SAE basis or custody label a record is about, or what
                                       # REPL commands run before the script (init, defaults)
    project: str = "yes"               # in a project file: "yes"; "inside" (a path that must
                                       # resolve inside the project file's directory); "no"
                                       # (command line, environment, MDB_CONFIG or user file only)
    help: str = ""


KEYS: dict[str, Key] = {key.name: key for key in (
    Key("model.roots", "paths", (), ("MDB_MODEL_ROOTS", "LLM_MODELS_EXTRA_ROOT"), recorded=True,
        help="directories searched for checkpoints by name after the Hugging Face cache"),
    Key("model.aliases", "aliases", {}, recorded=True,
        help="short name -> checkpoint directory or cached model name"),
    Key("model.dtype", "str", DEFAULT_DTYPE, ("MDB_MODEL_DTYPE",), DTYPES, recorded=True,
        help="weights dtype of a model mdb loads (recorded in every record's execution)"),
    Key("model.device", "str", DEFAULT_DEVICE, ("MDB_MODEL_DEVICE",), recorded=True,
        help="torch device of a model mdb loads (recorded in every record's execution)"),
    Key("project.dir", "path", None, ("MDB_PROJECT_DIR",), project="inside",
        help="project attached when no --project is given"),
    Key("project.session", "str", "auto", ("MDB_PROJECT_SESSION",), recorded=True,
        help="session id in the project; auto = family + digest of model and execution"),
    Key("records.sink", "str", None, ("MDB_RECORDS_SINK", "MDB_RECORDS"), project="inside",
        help="record sink: a file, a directory, or a name with {run}"),
    Key("records.append", "bool", False, ("MDB_RECORDS_APPEND",),
        help="add to a sink file another run wrote"),
    Key("records.payloads", "bool", True, ("MDB_RECORDS_PAYLOADS",),
        help="also write large payloads to the sink"),
    Key("custody.kernel_path", "str", None, ("MDB_CUSTODY_KERNEL_PATH", "MDB_KERNEL_PATH"),
        recorded=True, help="kernel path named in every record's custody block"),
    Key("serve.host", "str", DEFAULT_HOST, ("MDB_SERVE_HOST",), help="mdb serve bind host"),
    Key("serve.port", "int", DEFAULT_PORT, ("MDB_SERVE_PORT",), help="mdb serve port"),
    Key("serve.allow_hosts", "strs", (), ("MDB_SERVE_ALLOW_HOSTS",), project="no",
        help="extra Host headers mdb serve accepts (HOST or HOST:PORT)"),
    Key("repl.init", "path", None, ("MDB_REPL_INIT",), recorded=True, project="inside",
        help="command file run before --script or the prompt (a project file's runs only "
             "from a trusted project directory)"),
    Key("repl.trusted_projects", "paths", (), ("MDB_REPL_TRUSTED_PROJECTS",), project="no",
        help="directories whose project files may run their repl.init"),
    Key("repl.history", "path", None, ("MDB_REPL_HISTORY",), project="no",
        help="file keeping interactive command history"),
    Key("repl.defaults", "defaults", {}, recorded=True,
        help="OP.ARG = value: display and budget defaults of REPL commands"),
    Key("storage.shared_tensor_store", "path", None,
        ("MDB_STORAGE_SHARED_TENSOR_STORE", "MDB_SHARED_TENSOR_STORE"), project="inside",
        help="shared tensor pool (same filesystem as the projects using it)"),
    Key("sae.sources", "sae", {}, recorded=True, help="NAME = {path, layers, k, port, cache, verify, layer}"),
)}
SECTIONS = tuple(dict.fromkeys(name.split(".")[0] for name in KEYS))
SAE_FIELDS = {"path": "path", "layers": "str", "k": "int", "port": "str", "cache": "int",
              "verify": "bool", "layer": "int"}

_PER_CALL = ("it is written into the record body by the call that uses it, and a file the "
             "record does not name must not move it")
REFUSED: dict[str, str] = {
    **{name: "a floor decides whether a reading counts as a verdict; " + _PER_CALL
       + ": pass it per call (margin_floor(spacings=...), the op argument)"
       for name in ("floor", "floors", "floor_spacings", "margin_floor_spacings",
                    "stability_floor_spacings", "stable_k", "verdict")},
    "recheck": "it decides which branch arms are re-measured; pass Branches.run(recheck=...)",
    "lookahead": "it decides what a move inside a layer computes; pass it per run "
                 "(start lookahead=..., Debugger.run(lookahead=...))",
    "prefill": "it decides which prompt rows are stepped ports; pass it per run "
               "(start prefill=K, Debugger.run(prefill=K))",
    "retain_rows": "it decides which cuts a run keeps for rewinds; pass it per run "
                   "(start retain_rows=N, Debugger.run(retain_rows=N))",
    "mode": "branch modes decide how arms are measured; pass Branches.run(mode=...)",
    "batch": "branch batching decides how arms are measured; pass Branches.run(batch=...)",
    "branches": "branch settings decide how arms are measured; pass them to Branches.run",
    "prereg": "a preregistration belongs to one run; pass --prereg, MDB_PREREG or "
              "custody.declare(prereg=...)",
    "script": "custody hashes the script that ran; pass --script or MDB_SCRIPT",
    "seed": "a seed chooses the model or the samples; pass --seed per run",
    "allow_remote": "binding mdb serve beyond loopback has no authentication; type "
                    "--allow-remote each time",
}

_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off", "")


@dataclass
class Layer:
    """One source of values. ``kind``: cli, env, file, default."""

    kind: str
    label: str
    values: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    sha256: str | None = None
    names: dict[str, str] = field(default_factory=dict)   # key -> env var / flag that set it


def user_file(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "mdb" / "config.toml"


def _toml(path: Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - the declared dependency on Python 3.10
        import tomli as tomllib
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as error:
        raise ConfigError(f"{path}: cannot read: {error.strerror or error}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from None


def _pyproject_table(path: Path) -> dict[str, Any] | None:
    """``[tool.mdb]`` of a pyproject.toml, or None. An unparseable file with no ``[tool.mdb``
    text is not mdb's and is skipped; one that has it is an error."""
    try:
        data = _toml(path)
    except ConfigError:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return None
        if "[tool.mdb" in text:
            raise
        return None
    tool = data.get("tool")
    table = tool.get("mdb") if isinstance(tool, dict) else None
    return table if isinstance(table, dict) else None


def _refuse(where: str, name: str) -> None:
    if name in REFUSED:
        raise ConfigError(f"{where}: {name!r} cannot be set in a config file: {REFUSED[name]}")


def _check_value(where: str, key: Key, value: Any, base: Path | None) -> Any:
    """``value`` (from a file) as the setting's type; paths relative to the file's directory."""
    def bad(what: str) -> ConfigError:
        return ConfigError(f"{where}: expected {what}, got {value!r}")

    kind = key.kind
    if kind == "bool":
        if not isinstance(value, bool):
            raise bad("true or false")
        return value
    if kind == "int":
        if type(value) is not int:
            raise bad("an integer")
        if key.name == "serve.port" and not 1 <= value <= 65535:
            raise bad("a port 1..65535")
        return value
    if kind == "str":
        if key.choices and value not in key.choices:
            raise bad("one of " + ", ".join(key.choices))
        if not isinstance(value, str) or not value.strip():
            raise bad("non-empty text")
        if key.name == "records.sink" and base is not None:
            return _relative(value, base)
        return value
    if kind == "path":
        if not isinstance(value, str) or not value.strip():
            raise bad("a path")
        return _relative(value, base)
    if kind in ("paths", "strs"):
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip()
                                              for v in value):
            raise bad("a list of " + ("paths" if kind == "paths" else "text"))
        return tuple(_relative(v, base) if kind == "paths" else v for v in value)
    if kind == "aliases":
        if not isinstance(value, dict):
            raise bad("a table of NAME = \"model\"")
        for name, target in value.items():
            if name.startswith(("tiny:", "http://", "https://")):
                raise ConfigError(f"{where}.{name}: an alias may not shadow a tiny:FAMILY or "
                                  "http(s):// target")
            if not isinstance(target, str) or not target.strip():
                raise ConfigError(f"{where}.{name}: expected a model name or path, got "
                                  f"{target!r}")
        return {name: _alias(target, base) for name, target in value.items()}
    if kind == "defaults":
        return _defaults(where, value)
    if kind == "sae":
        return _sae(where, value, base)
    raise AssertionError(kind)


def _relative(text: str, base: Path | None) -> str:
    """``~`` expanded; a relative path in a file is relative to the file's directory (a
    trailing separator, meaning a directory, is kept)."""
    expanded = os.path.expanduser(text)
    if base is None or os.path.isabs(expanded):
        return expanded
    joined = str(base / expanded)
    return joined + os.sep if text.endswith(("/", os.sep)) and not joined.endswith(os.sep) \
        else joined


def _alias(target: str, base: Path | None) -> str:
    """An alias to an existing relative directory is relative to the file; else a name."""
    if base is not None and not os.path.isabs(os.path.expanduser(target)) and \
            (base / target).is_dir():
        return str(base / target)
    return os.path.expanduser(target)


def _defaults(where: str, value: Any) -> dict[str, Any]:
    """``repl.defaults``: ``{op = {arg = scalar}}`` or ``{"op.arg" = scalar}`` -> flat."""
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a table of OP.ARG = value")
    out: dict[str, Any] = {}
    for op, inner in value.items():
        items = inner.items() if isinstance(inner, dict) else [(None, inner)]
        for arg, scalar in items:
            name = f"{op}.{arg}" if arg is not None else op
            if "." not in name or name.count(".") != 1:
                raise ConfigError(f"{where}.{name}: write OP.ARG (print.top = 5)")
            _refuse(f"{where}.{name}", name.split(".")[1])
            if isinstance(scalar, (dict, list)) or isinstance(scalar, str):
                raise ConfigError(f"{where}.{name}: expected a number or true/false, got "
                                  f"{scalar!r}")
            out[name] = scalar
    return out


def _sae(where: str, value: Any, base: Path | None) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a table of NAME = {{path = ...}}")
    out = {}
    for name, entry in value.items():
        here = f"{where}.{name}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{here}: expected a table with path = ...")
        unknown = sorted(set(entry) - set(SAE_FIELDS))
        for item in unknown:
            _refuse(f"{here}.{item}", item)
        if unknown:
            raise ConfigError(f"{here}.{unknown[0]}: unknown key (known: "
                              f"{', '.join(SAE_FIELDS)})")
        if "path" not in entry:
            raise ConfigError(f"{here}: missing path")
        fields: dict[str, Any] = {}
        for item, kind in SAE_FIELDS.items():
            if item not in entry:
                continue
            spec = Key(f"sae.{item}", kind, choices=SAE_PORTS if item == "port" else ())
            fields[item] = _check_value(f"{here}.{item}", spec, entry[item], base)
        out[name] = fields
    return out


def _inside(value: str, base: Path) -> bool:
    target = Path(str(value).replace("{run}", "run")).expanduser()
    target = (base / target) if not target.is_absolute() else target
    try:
        Path(os.path.realpath(target)).relative_to(os.path.realpath(base))
        return True
    except ValueError:
        return False


def _file_layer(path: Path, label: str, table: dict[str, Any] | None = None) -> Layer:
    """A config file's values, validated: unknown keys, refused keys and types are errors. A
    project file (``label == "project"``: found from the project or by walking up, so possibly
    from a cloned repository) may not set the keys marked ``project="no"``, and the paths marked
    ``"inside"`` must resolve inside its directory."""
    raw = path.read_bytes() if path.is_file() else None
    if raw is None:
        raise ConfigError(f"{path}: no such config file")
    data = _toml(path) if table is None else table
    where = f"{path}" if table is None else f"{path}:[tool.mdb]"
    base = path.parent.resolve()
    values: dict[str, Any] = {}
    for section, body in data.items():
        if section == "schema":
            if body != SCHEMA:
                raise ConfigError(f"{where}:schema: expected {SCHEMA!r}, got {body!r}")
            continue
        _refuse(f"{where}:{section}", section)
        if section not in SECTIONS:
            raise ConfigError(f"{where}:{section}: unknown key (sections: "
                              f"{', '.join(SECTIONS)}; see docs/CONFIG.md)")
        if not isinstance(body, dict):
            raise ConfigError(f"{where}:{section}: expected a [{section}] table")
        for name, value in body.items():
            _refuse(f"{where}:{section}.{name}", name)
            key = KEYS.get(f"{section}.{name}")
            if key is None:
                known = ", ".join(k.split(".", 1)[1] for k in KEYS if k.startswith(section + "."))
                raise ConfigError(f"{where}:{section}.{name}: unknown key (in [{section}]: "
                                  f"{known})")
            checked = _check_value(f"{where}:{key.name}", key, value, base)
            if label == "project" and key.project == "no":
                raise ConfigError(f"{where}:{key.name}: a project file may not set this (it names "
                                  "a file mdb writes or a host it trusts); set it on the command "
                                  "line, in the environment or in the user file")
            if label == "project" and key.project == "inside" and not _inside(checked, base):
                raise ConfigError(f"{where}:{key.name}: {value!r} is outside the project "
                                  f"directory {base}; a project file names paths inside its "
                                  "own tree (set others on the command line, in the environment "
                                  "or in the user file)")
            values[key.name] = checked
    return Layer("file", label, values, path=path.resolve(),
                 sha256=hashlib.sha256(raw).hexdigest())


def _env_value(key: Key, name: str, text: str) -> Any:
    def bad(what: str) -> ConfigError:
        return ConfigError(f"environment {name}={text!r}: expected {what} ({key.name})")

    if key.kind == "bool":
        lowered = text.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise bad("1/true/yes or 0/false/no")
    if key.kind == "int":
        try:
            value = int(text)
        except ValueError:
            raise bad("an integer") from None
        if key.name == "serve.port" and not 1 <= value <= 65535:
            raise bad("a port 1..65535")
        return value
    if key.kind == "paths":
        return tuple(os.path.expanduser(p) for p in text.split(os.pathsep) if p.strip())
    if key.kind == "strs":
        return tuple(p.strip() for p in text.split(",") if p.strip())
    if key.kind == "path":
        return os.path.expanduser(text)
    if key.choices and text not in key.choices:
        raise bad("one of " + ", ".join(key.choices))
    return text


def _env_layer(env: dict[str, str]) -> Layer:
    layer = Layer("env", "environment")
    for key in KEYS.values():
        found = [(name, env[name]) for name in key.env if env.get(name, "").strip()]
        if not found:
            continue
        values = {name: _env_value(key, name, text) for name, text in found}
        if len(set(map(repr, values.values()))) > 1:
            listed = ", ".join(f"{name}={text!r}" for name, text in found)
            raise ConfigError(f"environment sets {key.name} twice with different values: "
                              f"{listed}; unset the older name")
        name = found[0][0]
        layer.values[key.name] = values[name]
        layer.names[key.name] = name
    return layer


def _repository_root(start: Path) -> Path | None:
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


SKIPPED: list[str] = []   # files discovery passed over (not the user's); config check prints them


def _owned(path: Path) -> bool:
    """Whether the current user owns ``path`` (as git's safe.directory: a file another user
    wrote is not read)."""
    getuid = getattr(os, "getuid", None)
    return getuid is None or path.stat().st_uid == getuid()


def find_project_file(start: Path) -> tuple[Path, dict[str, Any] | None] | None:
    """The first ``mdb.toml`` (table None) or ``pyproject.toml`` with ``[tool.mdb]`` walking up
    from ``start``, stopping at the repository root or the home directory. With neither above
    ``start`` (a scratch directory under /tmp), only ``start`` itself is looked at. A file the
    current user does not own is skipped (listed in ``SKIPPED``)."""
    start = start.resolve()
    stop = _repository_root(start)
    home = Path.home().resolve()
    if stop is None and home not in (start, *start.parents):
        stop = start
    for directory in (start, *start.parents):
        candidate = directory / FILE_NAME
        if candidate.is_file():
            if _owned(candidate):
                return candidate, None
            SKIPPED.append(f"{candidate}: not owned by the current user; not read")
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            if not _owned(pyproject):
                SKIPPED.append(f"{pyproject}: not owned by the current user; not read")
            else:
                table = _pyproject_table(pyproject)
                if table is not None:
                    return pyproject, table
        if directory == stop or directory == home:
            return None
    return None


class Settings:
    """Resolved settings: ``get(key)``, ``origin(key)``, ``explain(key)``, ``provenance()``."""

    def __init__(self, layers: list[Layer], args: dict[str, Any]):
        self.layers = layers                 # highest precedence first; the last is defaults
        self._args = args

    def _winner(self, name: str) -> Layer:
        if name not in KEYS:
            raise ConfigError(f"unknown setting {name!r} (settings: {', '.join(KEYS)})")
        for layer in self.layers:
            if name in layer.values:
                return layer
        raise AssertionError(name)

    def get(self, name: str) -> Any:
        return self._winner(name).values[name]

    def layer_of(self, name: str) -> Layer:
        return self._winner(name)

    def init_file(self) -> tuple[str | None, str | None]:
        """``(path, None)`` of the init file to run, or ``(None, reason)`` when a project
        file's ``repl.init`` is not trusted (its directory is not under a
        ``repl.trusted_projects`` entry): gdb's auto-load safe-path rule."""
        path = self.get("repl.init")
        layer = self._winner("repl.init")
        if path is None or layer.kind != "file" or layer.label != "project":
            return path, None
        here = os.path.realpath(layer.path.parent)
        for trusted in self.get("repl.trusted_projects"):
            root = os.path.realpath(os.path.expanduser(trusted))
            if here == root or here.startswith(root.rstrip(os.sep) + os.sep):
                return path, None
        return None, (f"not running the init file {path} named by the project file "
                      f"{layer.path}: add {layer.path.parent} to [repl] trusted_projects in the "
                      "user file, or pass --init FILE")

    def origin(self, name: str) -> str:
        return describe(self._winner(name), name)

    def explain(self, name: str) -> list[dict[str, Any]]:
        """Every layer that sets ``name``, highest first; the first one wins."""
        winner = self._winner(name)
        return [{"source": describe(layer, name), "value": layer.values[name],
                 "wins": layer is winner}
                for layer in self.layers if name in layer.values]

    @property
    def files(self) -> list[Layer]:
        return [layer for layer in self.layers if layer.kind == "file"]

    def as_dict(self) -> dict[str, Any]:
        return {name: self.get(name) for name in KEYS}

    def provenance(self) -> dict[str, Any] | None:
        """The block custody records, or None. Only files that set a key reaching a record
        (``Key.recorded``: model roots, aliases, dtype, device, session, kernel path, SAE
        sources, REPL init and defaults) count: each such file's path, sha256 and the recorded
        keys it sets, then the winning value and source of each of those keys. Values are JSON
        text (``"\"fp32\""``), so numbers in a setting are never read as measured claims. A file of locations only (serve, repl,
        records, storage, project dir) adds nothing, so record ids stay as without it."""
        sources, keys = [], []
        for layer in self.files:
            mine = {name: _text(value) for name, value in layer.values.items()
                    if KEYS[name].recorded}
            if mine:
                sources.append({"path": str(layer.path), "sha256": layer.sha256,
                                "layer": layer.label, "settings": mine})
                keys += [name for name in mine if name not in keys]
        if not sources:
            return None
        keys = [name for name in KEYS if name in keys]
        out = {"schema": PROVENANCE_SCHEMA, "sources": sources,
               "resolved": {name: _text(self.get(name)) for name in keys},
               "origins": {name: self.origin(name) for name in keys}}
        given = {name: sorted(values, key=repr) for name, values in EXPLICIT.items()
                 if name in keys}
        if given:   # a call argument overrode the setting (the record's execution says what ran)
            out["call_arguments"] = {name: [_text(v) for v in values]
                                     for name, values in given.items()}
        return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return list(value) if isinstance(value, tuple) else value


def _text(value: Any) -> str:
    """A setting's value as canonical JSON text: a record holds no bare numbers outside
    labelled claims, and a config value is a supplied setting, not a measurement."""
    import json

    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def describe(layer: Layer, name: str) -> str:
    if layer.kind == "file":
        return f"{layer.label} file {layer.path}"
    if layer.kind in ("env", "cli"):
        flag = layer.names.get(name)
        older = layer.kind == "env" and flag is not None and flag != KEYS[name].env[0]
        suffix = f" (older name of {KEYS[name].env[0]})" if older else ""
        return f"{'environment' if layer.kind == 'env' else 'command line'} {flag}{suffix}" \
            if flag else layer.label
    return "default"


def resolve(cli: dict[str, Any] | None = None, *, flags: dict[str, str] | None = None,
            project: str | Path | None = None, cwd: str | Path | None = None,
            env: dict[str, str] | None = None) -> Settings:
    """Resolve every setting. ``cli`` maps setting names to command-line values (``None``
    means not given); ``flags`` maps them to the flag spelled (for ``explain``); ``project`` is
    the project directory the command named (its ``mdb.toml`` is the project file)."""
    env = dict(os.environ if env is None else env)
    cwd = Path.cwd() if cwd is None else Path(cwd)
    cli_layer = Layer("cli", "command line",
                      {k: v for k, v in (cli or {}).items() if v is not None},
                      names={k: (flags or {}).get(k, k) for k in (cli or {})})
    for name in cli_layer.values:
        if name not in KEYS:
            raise ConfigError(f"unknown setting {name!r}")
    env_layer = _env_layer(env)
    files: list[Layer] = []
    named = env.get("MDB_CONFIG", "").strip()
    if named:
        files.append(_file_layer(Path(named).expanduser(), "MDB_CONFIG"))
    project_dir = project or cli_layer.values.get("project.dir") or \
        env_layer.values.get("project.dir")
    if project_dir is not None:
        candidate = Path(project_dir).expanduser() / FILE_NAME
        if candidate.is_file():
            files.append(_file_layer(candidate, "project"))
    else:
        found = find_project_file(cwd)
        if found is not None:
            path, table = found
            files.append(_file_layer(path, "project", table))
    user = user_file(env)
    if user.is_file():
        files.append(_file_layer(user, "user"))
    defaults = Layer("default", "default", {name: key.default for name, key in KEYS.items()})
    return Settings([cli_layer, env_layer, *files, defaults],
                    {"cli": cli, "flags": flags, "project": project, "cwd": cwd, "env": None})


# --------------------------------------------------------------------------- the process's

_ACTIVE: list[Settings] = []
_STORE_SET: list[bool] = []   # this module named the saturn_pub shared store


def _apply_store(root: str | None) -> None:
    """Hand ``storage.shared_tensor_store`` to saturn_pub (``set_shared_store``); with no
    store, clear one this module set before (the environment and ``.mdb-storage.json`` then
    apply as saturn_pub reads them)."""
    if root is None and not _STORE_SET:
        return
    from saturn_pub import shared_store

    setter = getattr(shared_store, "set_shared_store", None)
    if setter is None:
        if root is None:
            return
        raise ConfigError("storage.shared_tensor_store needs a saturn-pub with "
                          "shared_store.set_shared_store (install saturn-pub from the same "
                          "checkout as mdb)")
    try:
        setter(root)
    except ValueError as error:
        raise ConfigError(f"storage.shared_tensor_store = {root!r}: {error}") from None
    _STORE_SET[:] = [True] if root is not None else []


def activate(settings: Settings) -> Settings:
    """Make ``settings`` this process's (custody, sinks, model lookup, the Python entry points'
    defaults, SAE sources and the shared tensor store read it)."""
    store = settings.get("storage.shared_tensor_store")
    _apply_store(None if store is None else str(Path(store).expanduser().absolute()))
    _ACTIVE[:] = [settings]
    return settings


def current() -> Settings:
    """This process's settings, resolved from the environment and files on first use."""
    if not _ACTIVE:
        activate(resolve())
    return _ACTIVE[0]


EXPLICIT: dict[str, set] = {}   # setting -> values Python calls passed explicitly instead


def value(name: str, given: Any = None) -> Any:
    """``given`` when not None (an explicit argument wins), else the resolved setting. An
    explicit value that differs from the setting is remembered, so the custody config block
    shows that a call overrode it (``call_arguments``)."""
    resolved = current().get(name)
    if given is None:
        return resolved
    if isinstance(given, (str, int, float, bool)) and given != resolved:
        EXPLICIT.setdefault(name, set()).add(given)
    return given


def refresh() -> Settings:
    """Re-resolve with the same command line (the environment or files changed)."""
    if not _ACTIVE:
        return current()
    args = dict(_ACTIVE[0]._args)
    args.pop("env", None)
    return activate(resolve(**args))


def reset() -> None:
    """Forget the process's settings and clear a store this module set (tests)."""
    _ACTIVE.clear()
    EXPLICIT.clear()
    SKIPPED.clear()
    _apply_store(None)


def legacy_storage_file(start: str | Path) -> Path | None:
    """The nearest ``.mdb-storage.json`` above ``start`` (still read by saturn_pub; the
    ``[storage]`` section replaces it)."""
    path = Path(start).expanduser().resolve()
    for directory in (path, *path.parents):
        candidate = directory / ".mdb-storage.json"
        if candidate.is_file():
            return candidate
    return None


__all__ = ["ConfigError", "DEFAULT_DEVICE", "DEFAULT_DTYPE", "DEFAULT_HOST", "DEFAULT_PORT",
           "DEFAULT_SEED", "DTYPES", "KEYS", "Key", "REFUSED", "SCHEMA", "Settings",
           "TINY_FAMILIES", "activate", "current", "find_project_file", "refresh", "reset",
           "resolve", "user_file", "value"]

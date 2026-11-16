"""``mdb config show | explain KEY | init | check``: the resolved settings and where each came
from (``mdb.core.settings``). Loads no model framework.

    mdb config show [--project DIR] [--json]     every setting, its value and its source
    mdb config explain KEY [--project DIR]       each layer that sets KEY, highest first
    mdb config init [--project DIR] [--force]    write a commented template (user file, or
                                                 DIR/mdb.toml with --project)
    mdb config check [--project DIR]             validate files, environment and repl.defaults
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mdb.core import settings

TEMPLATE = f'''# mdb configuration (docs/CONFIG.md). Precedence: command line > MDB_* environment >
# MDB_CONFIG file > project mdb.toml > this user file > built-in defaults.
# Relative paths are relative to this file's directory. A project mdb.toml may not set
# repl.history, repl.trusted_projects or serve.allow_hosts, and its paths stay inside its tree. Settings that change a verdict
# (floors, recheck, lookahead, prefill, retain_rows, branch mode/batch, prereg, seed) are
# refused here: pass them per call, where custody sees them.
schema = "{settings.SCHEMA}"

[model]
# roots = ["/data/models"]               # searched by name after the Hugging Face cache
# aliases = {{ q15 = "Qwen/Qwen2.5-1.5B" }} # short name -> directory, cached name or tiny:FAMILY
# dtype = "{settings.DEFAULT_DTYPE}"                       # {" | ".join(settings.DTYPES)}; recorded in every record
# device = "{settings.DEFAULT_DEVICE}"                      # recorded in every record

[project]
# dir = "runs/main"                      # used when no --project is given
# session = "auto"                       # or a session id

[records]
# sink = "records/"                      # a file, a directory, or a name with {{run}}
# append = false
# payloads = true

[custody]
# kernel_path = "stepped"

[serve]
# host = "{settings.DEFAULT_HOST}"
# port = {settings.DEFAULT_PORT}
# allow_hosts = []

[repl]
# init = ".mdbinit"                      # commands run before --script or the prompt
# history = "~/.local/state/mdb/history" # user file only
# trusted_projects = ["~/work"]          # user file only: projects whose init files may run
# defaults = {{ print = {{ top = 8 }}, x = {{ count = 32 }} }}   # display defaults only

[storage]
# shared_tensor_store = "/data/mdb-store"   # same filesystem as the projects using it

[sae]
# sources = {{ l12 = {{ path = "~/sae/qwen-l12", k = 50, layers = "0:28" }} }}
'''


def _value(value) -> str:
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, sort_keys=True)


def _resolve(project: str | None) -> settings.Settings:
    return settings.resolve(project=project)


def _check_defaults(resolved: settings.Settings) -> list[str]:
    """repl.defaults against the op registry (problems as text)."""
    defaults = resolved.get("repl.defaults")
    if not defaults:
        return []
    from mdb.ops import BadArgs
    from mdb.ops.registry import check_default

    problems = []
    for name, value in defaults.items():
        try:
            check_default(name, value)
        except BadArgs as error:
            problems.append(f"{resolved.origin('repl.defaults')}: repl.defaults.{name}: "
                            f"{error.message}")
    return problems


def _warnings(resolved: settings.Settings, project: str | None) -> list[str]:
    out = []
    for root in resolved.get("model.roots"):
        if not Path(root).expanduser().is_dir():
            out.append(f"model.roots: {root} is not a directory "
                       f"({resolved.origin('model.roots')})")
    for key in ("repl.init",):
        path = resolved.get(key)
        if path is not None and not Path(path).expanduser().is_file():
            out.append(f"{key}: {path} does not exist ({resolved.origin(key)})")
    for name, source in resolved.get("sae.sources").items():
        if not Path(source["path"]).expanduser().exists():
            out.append(f"sae.sources.{name}: {source['path']} does not exist "
                       f"({resolved.origin('sae.sources')})")
    out += list(settings.SKIPPED)
    _, untrusted = resolved.init_file()
    if untrusted:
        out.append(untrusted)
    legacy = settings.legacy_storage_file(project or Path.cwd())
    if legacy is not None:
        note = ("it is ignored: storage.shared_tensor_store is set "
                f"({resolved.origin('storage.shared_tensor_store')})"
                if resolved.get("storage.shared_tensor_store") else "it is still read")
        out.append(f"deprecated: {legacy} configures the shared tensor store; move it to "
                   f"[storage] shared_tensor_store ({note})")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mdb config",
                                     description="show, explain, write or check mdb settings")
    verbs = parser.add_subparsers(dest="verb", required=True)
    show = verbs.add_parser("show", help="every setting with its value and source")
    show.add_argument("--json", action="store_true")
    explain = verbs.add_parser("explain", help="each source of one setting; the first wins")
    explain.add_argument("key", metavar="KEY", help="SECTION.NAME, e.g. model.dtype")
    init = verbs.add_parser("init", help="write a commented template")
    init.add_argument("--force", action="store_true", help="replace an existing file")
    verbs.add_parser("check", help="validate the files, the environment and repl.defaults")
    for sub in (show, explain, init, verbs.choices["check"]):
        sub.add_argument("--project", metavar="DIR",
                         help="the project whose mdb.toml to read (default: walk up from here)")
    args = parser.parse_args(argv)
    if args.verb == "init":
        target = (Path(args.project).expanduser() / settings.FILE_NAME if args.project
                  else settings.user_file())
        if target.exists() and not args.force:
            print(f"mdb config: {target} exists (--force replaces it)", file=sys.stderr)
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(TEMPLATE)
        print(f"wrote {target}")
        return 0
    try:
        resolved = _resolve(args.project)
    except settings.ConfigError as error:
        print(f"mdb config: {error}", file=sys.stderr)
        return 2
    if args.verb == "show":
        if args.json:
            print(json.dumps({"settings": {name: {"value": json.loads(_value(resolved.get(name))),
                                                  "source": resolved.origin(name)}
                                           for name in settings.KEYS},
                              "files": [{"path": str(layer.path), "sha256": layer.sha256,
                                         "layer": layer.label} for layer in resolved.files]},
                             indent=2, sort_keys=True))
            return 0
        files = resolved.files
        print("files: " + (", ".join(f"{layer.label} {layer.path}" for layer in files)
                           if files else "none (built-in defaults and the environment)"))
        width = max(map(len, settings.KEYS))
        for name in settings.KEYS:
            print(f"{name:<{width}}  {_value(resolved.get(name))}  ({resolved.origin(name)})")
        return 0
    if args.verb == "explain":
        key = settings.KEYS.get(args.key)
        if key is None:
            refused = args.key.split(".")[-1]
            if refused in settings.REFUSED:
                print(f"mdb config: {args.key!r} is not a setting: {settings.REFUSED[refused]}",
                      file=sys.stderr)
            else:
                print(f"mdb config: no setting {args.key!r} (settings: "
                      f"{', '.join(settings.KEYS)})", file=sys.stderr)
            return 2
        print(f"{key.name}: {key.help}")
        if key.env:
            print("  environment: " + ", ".join(
                [key.env[0], *[f"{name} (older name)" for name in key.env[1:]]]))
        for item in resolved.explain(key.name):
            mark = "=> " if item["wins"] else "   "
            print(f"  {mark}{_value(item['value'])}  {item['source']}")
        if key.recorded:
            print("  recorded: a config-file value is listed in every record's custody")
        if key.name == "storage.shared_tensor_store" and resolved.origin(key.name) == "default":
            legacy = settings.legacy_storage_file(args.project or Path.cwd())
            if legacy is not None:
                print(f"  deprecated source in use: {legacy} (move it to [storage])")
        return 0
    problems = _check_defaults(resolved)
    for line in problems:
        print(f"error: {line}", file=sys.stderr)
    for line in _warnings(resolved, args.project):
        print(f"warning: {line}", file=sys.stderr)
    if problems:
        return 2
    files = resolved.files
    print("ok: " + (", ".join(f"{layer.label} {layer.path}" for layer in files)
                    if files else "no config file (defaults and the environment)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

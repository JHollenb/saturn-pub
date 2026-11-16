"""Entry point of the port-level REPL (also ``mdb engine ...``).

    mdb engine MODEL [PROMPT] [--project DIR [--session NAME]] [--dtype bf16|fp32]
        [--device cpu|cuda] [--tokens "1 5 9 3"] [--script FILE] [--json]
        [--records PATH [--[no-]records-append] [--[no-]payloads]] [--kernel-path NAME]
        [--prereg FILE] [--init FILE | --no-init]

``MODEL`` is what to debug, as ``gdb PROGRAM`` names a program: a checkpoint directory, the
name of a model in the local Hugging Face cache (``qwen2.5-1.5b``; never downloaded),
``tiny:qwen2`` / ``tiny:qwen3`` / ``tiny:llama`` / ``tiny:mistral`` (a random tiny model), or
the ``http://`` address of a running ``mdb serve`` whose live session to drive. The same choices as flags:

    python -m mdb.repl --tiny qwen2|qwen3|llama|mistral [--dtype bf16|fp32] --tokens "1 5 9 3"
    python -m mdb.repl --model PATH|NAME [--dtype bf16|fp32] [--device cpu] --prompt TEXT
    python -m mdb.repl --connect http://127.0.0.1:8765 [--tokens ...|--prompt TEXT] [--script FILE]

``--project DIR`` attaches a project, so checkpoints are durable and ``restart NAME`` works in
a later process. Every default (dtype, device, project, sink, init file, ...) may come from a
config file or an ``MDB_*`` variable (``mdb.core.settings``; ``mdb config show``); a flag wins.
``--device`` applies to tiny models too (they are built on the CPU and moved).

``--connect`` drives the live session of a running ``mdb serve`` (``mdb.repl.remote``): the
model stays in the server process and this process needs no model framework.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(root: Path, dtype: str, device: str, want_tokenizer: bool,
          project: str | None, session: str | None = None) -> Any:
    """A local checkpoint through ``mdb.engine.models.load_local``; never downloads."""
    from mdb.engine.models import load_local
    from mdb.ops.debugger import attach

    model = load_local(root, dtype=dtype, device=device)  # the model only
    tokenizer = None
    has_tokenizer = any((root / name).is_file()
                        for name in ("tokenizer.json", "tokenizer_config.json"))
    if want_tokenizer or has_tokenizer:  # with one, token rows print as text, not ids
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    return attach(model, tokenizer, project=project, session=session)


_FRAMEWORK = ("torch", "numpy", "safetensors", "transformers", "accelerate")


def _missing(error: ModuleNotFoundError) -> int:
    """Exit status 2 with the extra to install when a model framework is absent."""
    name = (error.name or "").partition(".")[0]
    if name not in _FRAMEWORK:
        raise error
    print(f"mdb engine: needs {name}, which is not installed "
          "(pip install 'mdb-runtime[autoregressive]')", file=sys.stderr)
    return 2


# setting name -> the flag that sets it (``mdb config explain`` names the flag)
FLAGS = {"model.dtype": "--dtype", "model.device": "--device", "project.dir": "--project",
         "project.session": "--session", "records.sink": "--records",
         "records.append": "--[no-]records-append", "records.payloads": "--[no-]payloads",
         "custody.kernel_path": "--kernel-path", "repl.init": "--init"}


def _say_alias(prog: str, config: Any, name: str) -> None:
    """``alias NAME -> TARGET (source)`` on stderr whenever a configured alias was used."""
    target = config.get("model.aliases").get(str(name).strip())
    if target is not None:
        print(f"{prog}: alias {name} -> {target} ({config.origin('model.aliases')})",
              file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    from mdb.core import settings
    from mdb.core.errors import BadArgs
    from mdb.core.locate import parse_target

    parser = argparse.ArgumentParser(
        prog="python -m mdb.repl", description="gdb-style port-level model debugger",
        epilog="Defaults come from mdb.toml / ~/.config/mdb/config.toml and MDB_* variables "
               "(mdb config show; docs/CONFIG.md).")
    parser.add_argument("target", nargs="?", metavar="MODEL",
                        help="checkpoint directory, cached model name, model alias, "
                             "tiny:FAMILY, or the http:// address of a running `mdb serve`")
    parser.add_argument("text", nargs="?", metavar="PROMPT", help="prompt text to start on")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tiny", choices=settings.TINY_FAMILIES, help="random tiny model")
    source.add_argument("--model", metavar="PATH|NAME",
                        help="the same forms as MODEL")
    source.add_argument("--connect", metavar="URL",
                        help="drive the live session of a running `mdb serve` at URL")
    parser.add_argument("--dtype", choices=settings.DTYPES,
                        help=f"weights dtype (default {settings.DEFAULT_DTYPE}; [model] dtype)")
    parser.add_argument("--device", help=f"torch device, tiny models too (default "
                                          f"{settings.DEFAULT_DEVICE}; [model] device)")
    parser.add_argument("--seed", type=int, default=settings.DEFAULT_SEED,
                        help="tiny model seed")
    parser.add_argument("--tokens", help='prompt token ids, e.g. "1 5 9 3"')
    parser.add_argument("--prompt", help="prompt text (needs the checkpoint tokenizer)")
    parser.add_argument("--script", help="run commands from FILE, then exit")
    parser.add_argument("--json", action="store_true", help="print raw op results as JSON lines")
    parser.add_argument("--project", metavar="DIR",
                        help="attach a project: durable checkpoints, stored records "
                             "([project] dir)")
    parser.add_argument("--session", metavar="NAME",
                        help="session id in the project (default: family + model digest)")
    parser.add_argument("--records", metavar="PATH",
                        help="record sink: every sealed record is appended to PATH (a file, a "
                             "directory, or a name with {run}; [records] sink)")
    parser.add_argument("--records-append", action=argparse.BooleanOptionalAction,
                        default=None, help="add to a sink file another run wrote "
                                           "(--no-records-append overrides a config file)")
    parser.add_argument("--payloads", action=argparse.BooleanOptionalAction, default=None,
                        help="write large payloads to the sink (default yes; --no-payloads)")
    parser.add_argument("--kernel-path", metavar="NAME",
                        help="kernel path recorded in custody ([custody] kernel_path)")
    parser.add_argument("--prereg", metavar="FILE",
                        help="declare custody with this preregistration (hashed now)")
    parser.add_argument("--init", metavar="FILE",
                        help="commands run before --script or the prompt ([repl] init)")
    parser.add_argument("--no-init", action="store_true", help="skip the init file")
    args = parser.parse_args(argv)
    cli = {"model.dtype": args.dtype, "model.device": args.device, "project.dir": args.project,
           "project.session": args.session, "records.sink": args.records,
           "records.append": args.records_append, "records.payloads": args.payloads,
           "custody.kernel_path": args.kernel_path, "repl.init": args.init}
    try:
        config = settings.activate(settings.resolve(cli, flags=FLAGS, project=args.project))
        flagged = args.tiny or args.model or args.connect
        if args.target is not None and flagged:
            parser.error("give MODEL or one of --tiny/--model/--connect, not both")
        if args.target is not None or args.model:
            named = args.target if args.target is not None else args.model
            kind, value = parse_target(named)
            _say_alias("mdb engine", config, named)
            if kind == "connect":
                args.model, args.connect = None, value
            elif kind == "tiny":
                args.model, args.tiny = None, value
            else:
                args.model = value
        elif not flagged:
            parser.error("name what to debug: MODEL (a checkpoint directory, a cached model "
                         "name, tiny:qwen2, or an mdb serve address)")
    except (settings.ConfigError, BadArgs) as error:
        print(f"mdb engine: {getattr(error, 'message', error)}", file=sys.stderr)
        return 2
    if args.text is not None:
        if args.prompt:
            parser.error("give PROMPT or --prompt, not both")
        args.prompt = args.text
    if args.tokens and args.prompt:
        parser.error("give --tokens or --prompt, not both")
    if args.connect:
        given = [FLAGS[name] for name, value in cli.items()
                 if value is not None and name != "repl.init"]
        given += ["--prereg"] if args.prereg else []
        if given:
            parser.error(f"{', '.join(given)} belong to the server with --connect "
                         "(mdb serve ...)")
        if args.json:
            parser.error("--json is not available with --connect; for op JSON use "
                         "mdb.repl.remote.Remote(URL).op(NAME, ...)")
        from .remote import main as remote_main

        start = None
        if args.tokens:
            start = "start " + " ".join(args.tokens.replace(",", " ").split())
        elif args.prompt:
            start = "start " + json.dumps(args.prompt)
        return remote_main(args.connect, start=start, script=args.script)
    project = config.get("project.dir")
    session = config.get("project.session")
    session = None if session == "auto" else session
    if session is not None and project is None:
        parser.error("--session names a session of a project; give --project DIR too")
    try:
        from mdb.core import custody
        from mdb.ops import OpError
        from mdb.ops.debugger import open_tiny

        from .shell import Shell
    except ModuleNotFoundError as error:
        return _missing(error)

    dtype, device = config.get("model.dtype"), config.get("model.device")
    try:
        if args.prereg or args.kernel_path:
            import os

            custody.declare(script=args.script or os.environ.get("MDB_SCRIPT"),
                            prereg=args.prereg, kernel_path=config.get("custody.kernel_path"))
        custody.config_block()   # opens the configured sink now: a refused file fails here
        if args.tiny:
            if args.prompt:
                parser.error("tiny models have no tokenizer; use --tokens")
            debugger = open_tiny(args.tiny, dtype, seed=args.seed, project=project,
                                 session=session, device=device)
        else:
            from mdb.core.locate import locate_model

            debugger = _load(locate_model(args.model), dtype, device, bool(args.prompt),
                             project, session)
        shell = Shell(debugger.ops(), json_mode=args.json, history=config.get("repl.history"))
        shell.set_defaults(config.get("repl.defaults"),
                           source=config.origin("repl.defaults") + ": repl.defaults")
    except OpError as error:
        print(f"error ({error.kind}): {error.message}", file=sys.stderr)
        return 2
    except (settings.ConfigError, OSError) as error:
        print(f"mdb engine: {error}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as error:
        return _missing(error)
    init, untrusted = (None, None) if args.no_init else config.init_file()
    if untrusted:
        print(f"mdb engine: {untrusted}", file=sys.stderr)
    if init is not None:
        if not Path(init).expanduser().is_file():
            print(f"mdb engine: init file {init} ({config.origin('repl.init')}) does not exist",
                  file=sys.stderr)
            return 2
        shell.source_init(str(init), echo=not args.json)
    if args.tokens or args.prompt:
        words = (args.tokens or "").replace(",", " ").split()
        line = "start " + (" ".join(words) if args.tokens else json.dumps(args.prompt))
        if not args.json:
            shell.write(f"{shell.prompt}{line}")
        shell.execute(line)
    if args.script:
        shell.source(args.script, echo=not args.json)
        return 0 if shell.errors == 0 else 1
    return shell.loop()


if __name__ == "__main__":
    sys.exit(main())

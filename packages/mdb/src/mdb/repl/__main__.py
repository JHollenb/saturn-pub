"""Entry point of the port-level REPL (also ``mdb engine ...``).

    mdb engine MODEL [PROMPT] [--project DIR] [--dtype bf16|fp32] [--device cpu]
        [--tokens "1 5 9 3"] [--script FILE] [--json]

``MODEL`` is what to debug, as ``gdb PROGRAM`` names a program: a checkpoint directory, the
name of a model in the local Hugging Face cache (``qwen2.5-1.5b``; never downloaded),
``tiny:qwen2`` / ``tiny:qwen3`` / ``tiny:llama`` / ``tiny:mistral`` (a random tiny model), or
the ``http://`` address of a running ``mdb serve`` whose live session to drive. The same choices as flags:

    python -m mdb.repl --tiny qwen2|qwen3|llama|mistral [--dtype bf16|fp32] --tokens "1 5 9 3"
    python -m mdb.repl --model PATH|NAME [--dtype bf16|fp32] [--device cpu] --prompt TEXT
    python -m mdb.repl --connect http://127.0.0.1:8765 [--tokens ...|--prompt TEXT] [--script FILE]

``--project DIR`` attaches a project, so checkpoints are durable and ``restart NAME`` works in
a later process.

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
          project: str | None) -> Any:
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
    return attach(model, tokenizer, project=project)


_FRAMEWORK = ("torch", "numpy", "safetensors", "transformers", "accelerate")


def _missing(error: ModuleNotFoundError) -> int:
    """Exit status 2 with the extra to install when a model framework is absent."""
    name = (error.name or "").partition(".")[0]
    if name not in _FRAMEWORK:
        raise error
    print(f"mdb engine: needs {name}, which is not installed "
          "(pip install 'mdb-runtime[autoregressive]')", file=sys.stderr)
    return 2


# engine.models.TINY_FAMILIES, spelled out: this module must import without a model framework
TINY = ("qwen2", "qwen3", "llama", "mistral")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mdb.repl",
                                     description="gdb-style port-level model debugger")
    parser.add_argument("target", nargs="?", metavar="MODEL",
                        help="checkpoint directory, cached model name, tiny:FAMILY, "
                             "or the http:// address of a running `mdb serve`")
    parser.add_argument("text", nargs="?", metavar="PROMPT", help="prompt text to start on")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--tiny", choices=TINY, help="random tiny model")
    source.add_argument("--model", metavar="PATH|NAME",
                        help="checkpoint directory, or the name of a locally cached model")
    source.add_argument("--connect", metavar="URL",
                        help="drive the live session of a running `mdb serve` at URL")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=113, help="tiny model seed")
    parser.add_argument("--tokens", help='prompt token ids, e.g. "1 5 9 3"')
    parser.add_argument("--prompt", help="prompt text (needs the checkpoint tokenizer)")
    parser.add_argument("--script", help="run commands from FILE, then exit")
    parser.add_argument("--json", action="store_true", help="print raw op results as JSON lines")
    parser.add_argument("--project", metavar="DIR",
                        help="attach a project: durable checkpoints, stored records")
    args = parser.parse_args(argv)
    flagged = args.tiny or args.model or args.connect
    if args.target is not None:
        if flagged:
            parser.error("give MODEL or one of --tiny/--model/--connect, not both")
        if args.target.startswith(("http://", "https://")):
            args.connect = args.target
        elif args.target.startswith("tiny:"):
            args.tiny = args.target[len("tiny:"):]
            if args.tiny not in TINY:
                parser.error("tiny models are " + ", ".join(f"tiny:{name}" for name in TINY))
        else:
            args.model = args.target
    elif not flagged:
        parser.error("name what to debug: MODEL (a checkpoint directory, a cached model name, "
                     "tiny:qwen2, or an mdb serve address)")
    if args.text is not None:
        if args.prompt:
            parser.error("give PROMPT or --prompt, not both")
        args.prompt = args.text
    if args.tokens and args.prompt:
        parser.error("give --tokens or --prompt, not both")
    if args.connect:
        if args.project:
            parser.error("--project belongs to the server with --connect "
                         "(mdb serve --project DIR)")
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
    try:
        from mdb.ops import OpError
        from mdb.ops.debugger import open_tiny

        from .shell import Shell
    except ModuleNotFoundError as error:
        return _missing(error)

    try:
        if args.tiny:
            if args.prompt:
                parser.error("tiny models have no tokenizer; use --tokens")
            debugger = open_tiny(args.tiny, args.dtype, seed=args.seed, project=args.project)
        else:
            from mdb.core.locate import locate_model

            debugger = _load(locate_model(args.model), args.dtype, args.device,
                             bool(args.prompt), args.project)
    except OpError as error:
        print(f"error ({error.kind}): {error.message}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as error:
        return _missing(error)
    shell = Shell(debugger.ops(), json_mode=args.json)
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

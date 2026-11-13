"""Entry point of the port-level REPL (also ``mdb engine ...``).

    python -m mdb.repl --tiny qwen2|qwen3 [--dtype bf16|fp32] --tokens "1 5 9 3"
        [--script FILE] [--json]
    python -m mdb.repl --model PATH [--dtype bf16|fp32] [--device cpu] --prompt TEXT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: str, dtype: str, device: str, want_tokenizer: bool) -> Any:
    """A local checkpoint through ``mdb.engine.models.load_local``; never downloads."""
    from mdb.ops.debugger import attach

    root = Path(path).expanduser().resolve()
    config = root / "config.json"
    if not config.is_file():
        raise SystemExit(f"--model needs a Hugging Face checkpoint directory ({config} missing)")
    from mdb.engine.models import load_local
    model = load_local(root, dtype=dtype, device=device)  # the model only
    tokenizer = None
    if want_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    return attach(model, tokenizer)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mdb.repl",
                                     description="gdb-style port-level model debugger")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--tiny", choices=("qwen2", "qwen3"), help="random tiny model")
    source.add_argument("--model", help="local Hugging Face checkpoint directory")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=113, help="tiny model seed")
    parser.add_argument("--tokens", help='prompt token ids, e.g. "1 5 9 3"')
    parser.add_argument("--prompt", help="prompt text (needs the checkpoint tokenizer)")
    parser.add_argument("--script", help="run commands from FILE, then exit")
    parser.add_argument("--json", action="store_true", help="print raw op results as JSON lines")
    args = parser.parse_args(argv)
    if args.tokens and args.prompt:
        parser.error("give --tokens or --prompt, not both")
    from mdb.ops import OpError
    from mdb.ops.debugger import open_tiny

    from .shell import Shell

    try:
        if args.tiny:
            if args.prompt:
                parser.error("tiny models have no tokenizer; use --tokens")
            debugger = open_tiny(args.tiny, args.dtype, seed=args.seed)
        else:
            debugger = _load(args.model, args.dtype, args.device, bool(args.prompt))
    except OpError as error:
        print(f"error ({error.kind}): {error.message}", file=sys.stderr)
        return 2
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

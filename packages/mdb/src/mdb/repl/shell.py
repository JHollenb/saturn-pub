"""A gdb-style line REPL: every command is one ``mdb.ops.dispatch`` call, then ``render``.

Conventions: an empty line repeats the last motion command (interactive only); unique
prefixes resolve (``fin`` -> ``finish``); aliases ``s n c b p bt u d i l r q h rs rn rc``
(``u`` is until, ``d`` delete, as in gdb); ``help [verb]`` reads the registry. Local
commands: ``help``, ``source [-k] FILE``, ``record [append] FILE|stop``, ``quit``.

No traceback ever reaches the user: an unexpected exception is an ``internal`` error (the
registry converts it), Ctrl-C during a command prints ``interrupted at <where>`` and the
session carries on. ``source`` stops at its first error (as gdb) unless ``-k``, and nests at
most ``MAX_SOURCE_DEPTH`` deep. ``record`` is refused while sourcing, refuses an existing
file unless ``append``, and creates files with mode 0600.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO

from mdb.ops import OpError, dispatch, registry
from mdb.ops.errors import BadArgs, internal
from mdb.ops.lines import (  # noqa: F401 - the parser moved to mdb.ops.lines; re-exported
    _INT,
    _KEYWORD,
    ALIASES,
    SPECIAL,
    _after,
    _break_args,
    _compare_args,
    _condition_args,
    _consumer_option,
    _convert,
    _each_args,
    _equals_outside_brackets,
    _generic,
    _prompt_args,
    _set_args,
    _symbol_args,
    _up_args,
    _watch_args,
    _whatif_args,
    _with_consumer,
    parse,
    resolve_verb,
    route,
    split,
)

from .render import render, render_error

LOCAL = ("help", "source", "record", "quit")
MAX_SOURCE_DEPTH = 16


# --------------------------------------------------------------------------- shell


class Shell:
    def __init__(
        self,
        ctx: Any,
        *,
        json_mode: bool = False,
        out: TextIO | None = None,
        prompt: str = "(mdb) ",
    ):
        self.ctx = ctx
        self.json_mode = json_mode
        self.out = out if out is not None else sys.stdout
        self.prompt = prompt
        self.last_motion: str | None = None
        self.record_to: Path | None = None
        self.errors = 0
        self.depth = 0  # nesting of `source`
        self._interrupted = False  # Ctrl-C seen: a running script stops
        self._failed = False       # the last command (or a nested non -k source) failed

    def write(self, text: str) -> None:
        self.out.write(text + "\n")

    def resolve(self, word: str) -> str:
        return resolve_verb(word, LOCAL)

    def _error(self, error: OpError) -> None:
        self.errors += 1
        self._failed = True
        if self.json_mode:
            self.write(json.dumps({"error": error.to_json(), "state": error.state,
                                   "reason": error.message}, sort_keys=True))
        else:
            self.write(render_error(error))

    def _where(self) -> str:
        run = getattr(self.ctx, "run", None)
        try:
            return run.where if run is not None else "no run"
        except Exception:  # noqa: BLE001 - reporting the cursor must never fail
            return "an unknown position"

    def execute(self, line: str, *, repeat: bool = False) -> bool:
        """Run one command line; False means quit. Never raises: errors are printed (an
        unexpected one as ``internal``), Ctrl-C prints ``interrupted at <where>``."""
        text = line.strip()
        if self.depth == 0:
            self._interrupted = False
        if not text:
            if not repeat or self.last_motion is None:
                return True
            text = self.last_motion
        if text.startswith("#"):
            return True
        verb, _, rest = text.partition(" ")
        try:
            name = self.resolve(verb)
            if name == "quit":
                return False
            if name in LOCAL:
                self._local(name, rest.strip())
                if name != "record":
                    self._record(text, name)
                return True
            name, rest = route(name, rest)
            args = parse(name, rest)
            result = dispatch(self.ctx, name, args)
        except OpError as error:
            self._error(error)
            return True
        except KeyboardInterrupt:
            self.errors += 1
            self._failed = True
            self._interrupted = True
            where = self._where()
            if self.json_mode:
                self.write(json.dumps({"interrupted": where, "state": "unresolved",
                                       "reason": f"interrupted at {where}"}, sort_keys=True))
            else:
                self.write(f"interrupted at {where}")
            return True
        except Exception as error:  # noqa: BLE001 - the shell never shows a traceback
            self._error(internal(error))
            return True
        self.write(json.dumps(result, sort_keys=True) if self.json_mode else render(result))
        if registry()[name].motion:
            self.last_motion = text
        self._record(text, name)
        return True

    def _record(self, text: str, name: str) -> None:
        if self.record_to is not None and name != "record":
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            fd = os.open(self.record_to, flags, 0o600)
            with os.fdopen(fd, "a") as handle:
                handle.write(text + "\n")

    def _start_recording(self, rest: str) -> None:
        append = False
        words = rest.split(None, 1)
        if words and words[0] == "append":
            append, rest = True, (words[1] if len(words) > 1 else "")
        if self.depth:
            raise BadArgs("record is not available from a sourced script")
        if not rest:
            raise BadArgs("record [append] FILE | record stop")
        path = Path(rest).expanduser()
        if path.exists() and not append:
            raise BadArgs(f"{path} exists; use `record append {rest}` to add to it")
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_EXCL),
                         0o600))
        self.record_to = path
        self._say_record(path, append)

    def _say_record(self, path: Path | None, append: bool) -> None:
        """``record`` is JSON in ``--json`` mode like every other op."""
        if self.json_mode:
            self.write(json.dumps({"op": "record", "state": "resolved",
                                   "recording": None if path is None else str(path),
                                   "append": append}, sort_keys=True))
        elif path is None:
            self.write("recording stopped")
        else:
            self.write(f"recording commands to {path}" + (" (append)" if append else ""))

    def _local(self, name: str, rest: str) -> None:
        if name == "help":
            text = self.help(rest or None)
            self.write(json.dumps({"op": "help", "state": "resolved", "text": text},
                                  sort_keys=True) if self.json_mode else text)
        elif name == "source":
            keep = False
            if rest.startswith("-k "):
                keep, rest = True, rest[3:].strip()
            if not rest:
                raise BadArgs("source [-k] FILE")
            try:
                self.source(rest, keep_going=keep)
            except OSError as error:
                raise BadArgs(f"cannot read {rest}: {error.strerror or error}") from None
        elif name == "record":
            if rest in ("", "stop"):
                self.record_to = None
                self._say_record(None, False)
            else:
                self._start_recording(rest)

    def help(self, verb: str | None = None) -> str:
        ops = registry()
        if verb is None:
            lines = []
            for effect in ("native", "navigate", "read", "mutate"):
                lines.append(f"{effect}:")
                for name in sorted(ops):
                    item = ops[name]
                    if item.effect == effect and item.milestone is None:
                        usage = item.usage or name
                        if len(usage) > 42:
                            lines.append(f"  {usage}")
                            usage = ""
                        lines.append(f"  {usage:<44}{item.summary}")
            slots = sorted(name for name, item in ops.items() if item.milestone is not None)
            if slots:
                lines.append("not built yet: " + "; ".join(
                    f"{name}: {ops[name].milestone}" for name in slots))
            lines.append("local: help [VERB] | source [-k] FILE | record [append] FILE|stop "
                         "| quit")
            lines.append("aliases: " + " ".join(f"{k}={v}" for k, v in sorted(ALIASES.items())
                                                if k != v))
            return "\n".join(lines)
        name = self.resolve(verb)
        if name in LOCAL:
            return {"help": "help [VERB]: list commands or describe one",
                    "source": "source [-k] FILE: run commands from a file; stops at the first "
                              "error unless -k (keep going); nests at most "
                              f"{MAX_SOURCE_DEPTH} deep",
                    "record": "record [append] FILE|stop: write executed commands to FILE "
                              "(refuses an existing file unless `append`; mode 0600; not "
                              "available from a sourced script)",
                    "quit": "quit: leave the debugger"}[name]
        item = ops[name].to_json()
        lines = [f"{item['usage']}", f"  {item['summary']}",
                 f"  effect {item['effect']}, needs a live run: "
                 f"{'yes' if item['needs_live'] else 'no'}"]
        if "refused_until" in item:
            lines.append(f"  not built yet: {item['refused_until']}")
        for arg in item["args"]:
            extra = []
            if arg.get("choices"):
                extra.append("|".join(arg["choices"]))
            if "default" in arg:
                extra.append(f"default {arg['default']}")
            if arg["required"]:
                extra.append("required")
            text = f"  {arg['name']} ({arg['type']}{', ' + ', '.join(extra) if extra else ''})"
            if arg.get("help"):
                text += f": {arg['help']}"
            lines.append(text)
        return "\n".join(lines)

    def source(self, path: str, *, echo: bool = True, keep_going: bool = False) -> int:
        """Run a command file. It stops at the first error (as gdb) unless ``keep_going``
        (``source -k``); nested ``source`` stops too, and at most ``MAX_SOURCE_DEPTH`` deep.
        Returns the number of errors."""
        if self.depth >= MAX_SOURCE_DEPTH:
            raise BadArgs(f"source nests more than {MAX_SOURCE_DEPTH} deep (a script that "
                          f"sources itself?): {path}")
        before = self.errors
        self.depth += 1
        try:
            for number, raw in enumerate(Path(path).expanduser().read_text().splitlines(), 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if echo and not self.json_mode:  # json mode: stdout is JSON lines only
                    self.write(f"{self.prompt}{line}")
                self._failed = False
                if not self.execute(line):
                    break
                if self._interrupted:
                    break
                if self._failed and not keep_going:
                    if self.depth == 1:
                        reason = (f"stopped sourcing {path} at line {number} "
                                  "(use `source -k FILE` to keep going)")
                        self.write(json.dumps({"op": "source", "state": "refused",
                                               "reason": reason}, sort_keys=True)
                                   if self.json_mode else f"error: {reason}")
                    break
        finally:
            self.depth -= 1
            if keep_going:
                self._failed = False  # source -k: its errors do not fail the calling line
        return self.errors - before

    def loop(self, stream: TextIO | None = None) -> int:
        stream = stream if stream is not None else sys.stdin
        interactive = stream is sys.stdin and stream.isatty()
        if interactive:
            try:
                import readline  # noqa: F401 - enables line editing and history for input()
            except ImportError:
                pass
        while True:
            try:
                if interactive:
                    line = input(self.prompt)
                else:
                    line = stream.readline()
                    if not line:
                        break
                    line = line.rstrip("\n")
                    if line.strip() and not line.strip().startswith("#"):
                        self.write(f"{self.prompt}{line.strip()}")
            except EOFError:
                break
            except KeyboardInterrupt:
                self.write("")
                continue
            try:
                if not self.execute(line, repeat=interactive):
                    break
            except KeyboardInterrupt:  # between commands (execute handles its own)
                self.write(f"interrupted at {self._where()}")
        return 0 if self.errors == 0 else 1

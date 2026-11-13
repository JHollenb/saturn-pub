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
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from mdb.ops import OpError, dispatch, registry
from mdb.ops.errors import BadArgs, UnknownOp, internal
from mdb.ops.predict import parse_assignment

from .render import render, render_error

ALIASES = {
    "s": "step", "n": "next", "c": "continue", "b": "break", "p": "print", "bt": "backtrace",
    "u": "until", "d": "delete", "i": "info", "l": "list", "r": "run", "q": "quit",
    "h": "help", "rs": "reverse-step", "rn": "reverse-next", "rc": "reverse-continue",
    "where": "where", "exit": "quit",
}
LOCAL = ("help", "source", "record", "quit")
MAX_SOURCE_DEPTH = 16
_INT = re.compile(r"-?\d+")
_KEYWORD = re.compile(r"(?P<key>[a-z_]+)=(?P<value>.*)", re.S)


# --------------------------------------------------------------------------- tokenizing


def split(text: str) -> list[tuple[str, int, int]]:
    """Whitespace-separated words, keeping ``[...]`` and quoted text together.

    Returns ``(word, start, end)``; a word that is wholly a quoted string loses its quotes.
    """
    words, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start, depth, quote = i, 0, None
        while i < n:
            char = text[i]
            if quote:
                if char == "\\" and i + 1 < n:
                    i += 2
                    continue
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth = max(0, depth - 1)
            elif char.isspace() and depth == 0:
                break
            i += 1
        if quote:
            raise BadArgs(f"unterminated quote in {text[start:]!r}")
        word = text[start:i]
        if len(word) >= 2 and word[0] == word[-1] and word[0] in "\"'":
            word = json.loads(word) if word[0] == '"' else word[1:-1]
        words.append((word, start, i))
    return words


def _after(text: str, words: list[tuple[str, int, int]], index: int) -> str:
    return text[words[index][1]:].strip() if index < len(words) else ""


def _convert(arg: Any, raw: str) -> Any:
    kind = arg.type
    if kind == "int":
        if not _INT.fullmatch(raw):
            raise BadArgs(f"{arg.name} must be an integer, got {raw!r}")
        return int(raw)
    if kind == "number":
        try:
            return float(raw)
        except ValueError:
            raise BadArgs(f"{arg.name} must be a number, got {raw!r}") from None
    if kind == "bool":
        if raw.lower() not in ("true", "false", "on", "off", "1", "0"):
            raise BadArgs(f"{arg.name} must be true or false")
        return raw.lower() in ("true", "on", "1")
    if kind == "choice":
        found = [c for c in arg.choices if str(c).startswith(raw)]
        exact = [c for c in arg.choices if str(c) == raw]
        if exact:
            return exact[0]
        if len(found) != 1:
            raise BadArgs(f"{arg.name} is one of {', '.join(map(str, arg.choices))}")
        return found[0]
    if kind == "ints":
        items = raw.split()
        if not all(_INT.fullmatch(item) for item in items):
            raise BadArgs(f"{arg.name} takes integers")
        return [int(item) for item in items]
    if kind == "value":
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def _generic(op: Any, text: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    known = {arg.name: arg for arg in op.args}
    positional = [arg for arg in op.args if arg.positional]
    for word, _, _ in split(text):
        match = _KEYWORD.fullmatch(word)
        if match and match["key"] in known:
            args[match["key"]] = _convert(known[match["key"]], match["value"])
            continue
        if positional and positional[0].type == "ints":
            args.setdefault(positional[0].name, []).extend(
                _convert(positional[0], word))
            continue
        # A choice word may land in a later positional slot: "step layer", "step 3 layer".
        slot = None
        for arg in positional:
            if arg.type == "choice" and any(str(c).startswith(word) for c in arg.choices) \
                    and not _INT.fullmatch(word):
                slot = arg
                break
        if slot is None:
            if not positional:
                raise BadArgs(f"{op.name} takes no argument {word!r}")
            slot = positional[0]
        positional.remove(slot)
        args[slot.name] = _convert(slot, word)
    return args


def _prompt_args(text: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    words = split(text)
    rest = []
    for word, _, _ in words:
        match = _KEYWORD.fullmatch(word)
        if match and match["key"] == "budget":
            if not _INT.fullmatch(match["value"]):
                raise BadArgs(f"budget must be an integer, got {match['value']!r}")
            args["budget"] = int(match["value"])
        elif match and match["key"] == "name" and not rest:
            args["name"] = match["value"]
        else:
            rest.append(word)
    if rest:
        if all(_INT.fullmatch(word) for word in rest):
            args["prompt"] = [int(word) for word in rest]
        else:
            args["prompt"] = " ".join(rest)
    return args


def _break_args(text: str) -> dict[str, Any]:
    words = split(text)
    if not words:
        raise BadArgs("break needs an address pattern, e.g. break L*.p")
    args: dict[str, Any] = {"pattern": words[0][0]}
    if len(words) > 1:
        if words[1][0] != "if":
            raise BadArgs("write break PATTERN [if CONDITION]")
        args["condition"] = _after(text, words, 2)
    return args


def _watch_args(text: str) -> dict[str, Any]:
    words = split(text)
    if not words:
        raise BadArgs("watch needs an address, e.g. watch L0.h[d=3]")
    args: dict[str, Any] = {"address": words[0][0]}
    index = 1
    if index < len(words) and words[index][0] in ("change", "level", "edge"):
        args["mode"] = words[index][0]
        index += 1
    if index < len(words):
        if words[index][0] != "if":
            raise BadArgs("write watch ADDR [change|level|edge] [if PREDICATE]")
        args["predicate"] = _after(text, words, index + 1)
    return args


def _equals_outside_brackets(word: str) -> int:
    """Index of the first ``=`` that is not inside ``[...]`` or quotes, else -1."""
    depth, quote = 0, None
    for i, char in enumerate(word):
        if quote:
            quote = None if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char == "=" and depth == 0:
            return i
    return -1


_SET_USAGE = "set ADDR = VALUE [replace|add|scale|mix DOSE]  (or ADDR += X, -=, *=, /=)"
_SUGAR = ("+=", "-=", "*=", "/=")


def _set_args(text: str) -> dict[str, Any]:
    """``ADDR = VALUE [OP [DOSE]]``, ``ADDR VALUE [OP]`` or ``ADDR += X`` (``-= *= /=``), via
    ``mdb.ops.predict.parse_assignment`` (parsed, never evaluated)."""
    words = [word for word, _, _ in split(text)]
    if not words:
        raise BadArgs(f"write {_SET_USAGE}")
    joined = " ".join(words)
    if _equals_outside_brackets(joined) < 0:
        joined = " ".join([words[0], "=", *words[1:]])
    target, value, op, dose = parse_assignment(joined, usage=_SET_USAGE)
    args: dict[str, Any] = {"target": target, "value": value}
    if op is not None:
        args["op"] = op
    if dose is not None:
        args["dose"] = dose
    return args


_WHATIF_USAGE = ("write whatif ADDR = VALUE [replace|add|scale|mix DOSE] --consumer "
                 "M[a,b]|logits[i]|ADDR")


def _whatif_args(text: str) -> dict[str, Any]:
    """``whatif ADDR = VALUE [OP] --consumer C``; ``consumer=C`` is accepted too."""
    words = split(text)
    consumer, kept, i = None, [], 0
    while i < len(words):
        word = words[i][0]
        if word == "--consumer" and i + 1 < len(words):
            consumer, i = words[i + 1][0], i + 2
        elif word.startswith(("consumer=", "--consumer=")):
            consumer, i = word.split("=", 1)[1], i + 1
        else:
            kept.append(words[i])
            i += 1
    if not consumer or not kept:
        raise BadArgs(_WHATIF_USAGE)
    try:
        args = _set_args(" ".join(text[start:end] for _, start, end in kept))
    except BadArgs:
        raise BadArgs(_WHATIF_USAGE) from None
    return {**args, "consumer": consumer}


def _up_args(text: str) -> dict[str, Any]:
    args: dict[str, Any] = {}
    for word, _, _ in split(text):
        if _INT.fullmatch(word):
            args["n"] = int(word)
        elif "port" not in args:
            args["port"] = word
        else:
            raise BadArgs("write up [N] [PORT]")
    return args


def _condition_args(text: str) -> dict[str, Any]:
    words = split(text)
    if not words or not _INT.fullmatch(words[0][0]):
        raise BadArgs("write condition N [EXPR]  (no EXPR removes the condition)")
    args: dict[str, Any] = {"id": int(words[0][0])}
    expr = _after(text, words, 1)
    if expr:
        args["expr"] = expr
    return args


def _consumer_option(text: str) -> tuple[str | None, str]:
    """Take ``--consumer C`` (or ``--consumer=C``) out of a command line."""
    words = split(text)
    consumer, kept, i = None, [], 0
    while i < len(words):
        word = words[i][0]
        if word == "--consumer" and i + 1 < len(words):
            consumer, i = words[i + 1][0], i + 2
        elif word.startswith("--consumer="):
            consumer, i = word.split("=", 1)[1], i + 1
        else:
            kept.append(words[i])
            i += 1
    return consumer, " ".join(text[start:end] for _, start, end in kept)


def _with_consumer(name: str):
    """The generic parser of ``name`` that also takes ``--consumer C``."""
    def parse_args(text: str) -> dict[str, Any]:
        consumer, rest = _consumer_option(text)
        args = _generic(registry()[name], rest)
        if consumer is not None:
            args["consumer"] = consumer
        return args

    return parse_args


_COMPARE_USAGE = ("write compare A B [--consumer C]; a side is ADDR=VALUE [op] (ADDR += X ...) "
                  "or a checkpoint")
_OP_WORDS = ("replace", "add", "scale", "mix")


def _compare_args(text: str) -> dict[str, Any]:
    """``compare A B [--consumer C]``: a side is ``ADDR=VALUE [replace|add|scale|mix DOSE]``
    (spaces around ``=`` allowed; ``ADDR += X`` and the other sugar too) or a checkpoint
    name."""
    consumer, rest = _consumer_option(text)
    args: dict[str, Any] = {}
    words = [word for word, _, _ in split(rest)]
    sides: list[list[str]] = []
    i = 0
    while i < len(words):
        word = words[i]
        if i + 2 < len(words) and words[i + 1] in ("=", *_SUGAR):
            word, i = f"{word}{words[i + 1]}{words[i + 2]}", i + 2
        elif word.endswith(("=", *_SUGAR)) and _equals_outside_brackets(word) == len(word) - 1 \
                and i + 1 < len(words):
            word, i = f"{word}{words[i + 1]}", i + 1
        last = sides[-1] if sides else None
        if last and _equals_outside_brackets(last[0]) > 0 and word in _OP_WORDS \
                and len(last) == 1:
            last.append(word)
            if word == "mix" and i + 1 < len(words):
                last.append(words[i + 1])
                i += 1
        else:
            sides.append([word])
        i += 1
    if len(sides) != 2:
        raise BadArgs(_COMPARE_USAGE)
    args.update(a=" ".join(sides[0]), b=" ".join(sides[1]))
    if consumer is not None:
        args["consumer"] = consumer
    return args


def _symbol_args(text: str) -> dict[str, Any]:
    """``symbol NAME = ADDR[,ADDR...]``."""
    name, equals, addresses = text.partition("=")
    if not equals or not name.strip() or not addresses.strip():
        raise BadArgs("write symbol NAME = ADDR[,ADDR...]")
    return {"name": name.strip(), "addresses": addresses.strip()}


SPECIAL = {
    "start": _prompt_args,
    "run": _prompt_args,
    "break": _break_args,
    "tbreak": _break_args,
    "watch": _watch_args,
    "set": _set_args,
    "whatif": _whatif_args,
    "up": _up_args,
    "condition": _condition_args,
    "reads": _with_consumer("reads"),
    "symbols": _with_consumer("symbols"),
    "circuit": _with_consumer("circuit"),
    "compare": _compare_args,
    "symbol": _symbol_args,
}


def parse(name: str, text: str) -> dict[str, Any]:
    """Command-line text -> the op's JSON args."""
    if name in SPECIAL:
        return SPECIAL[name](text)
    found = registry()[name]
    if found.extra.get("accepts_any"):
        return {}
    return _generic(found, text)


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
        ops = registry()
        if word in ops or word in LOCAL:
            return word
        if word in ALIASES:
            return ALIASES[word]
        names = sorted(set(ops) | set(LOCAL))
        found = [name for name in names if name.startswith(word)]
        if len(found) == 1:
            return found[0]
        if not found:
            raise UnknownOp(f'undefined command "{word}"; try "help"')
        raise UnknownOp(f'ambiguous command "{word}": {", ".join(found)}')

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
            lines.append("not built yet: " + " ".join(
                f"{name}({ops[name].milestone})" for name in slots))
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

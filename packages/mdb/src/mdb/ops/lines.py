"""Command lines -> op arguments: the one parser of the REPL's command language.

``parse(name, text)`` turns the text after a verb into the op's JSON args; ``resolve_verb``
resolves aliases and unique prefixes. It lives in ``mdb.ops`` (moved unchanged from
``mdb.repl.shell``, which re-exports it) so an op that runs another op line, ``each``, and the
embedding API ``Debugger.each`` parse with the same code as the REPL. Nothing here evaluates
an expression or loads a model library.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import BadArgs, UnknownOp
from .predict import parse_assignment
from .registry import registry

ALIASES = {
    "s": "step", "n": "next", "c": "continue", "b": "break", "p": "print", "bt": "backtrace",
    "u": "until", "d": "delete", "i": "info", "l": "list", "r": "run", "q": "quit",
    "h": "help", "rs": "reverse-step", "rn": "reverse-next", "rc": "reverse-continue",
    "where": "where", "exit": "quit",
}
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
        if match and match["key"] in ("budget", "prefill"):
            if not _INT.fullmatch(match["value"]):
                raise BadArgs(f"{match['key']} must be an integer, got {match['value']!r}")
            args[match["key"]] = int(match["value"])
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
    """``symbol NAME = ADDR[,ADDR...]``; ``symbol NAME = whatif|compare.a|compare.b|EACH:ITEM
    [role=R] [readers=K] [causal=false]`` (a named symbol from a transplant result, M23);
    ``symbol show|delete NAME``."""
    from . import named

    words = text.split()
    if len(words) == 2 and words[0] in ("show", "delete"):
        return {"action": words[0], "name": words[1]}
    name, equals, addresses = text.partition("=")
    if not equals or not name.strip() or not addresses.strip():
        raise BadArgs("write symbol NAME = ADDR[,ADDR...], symbol NAME = whatif|compare.a|"
                      "compare.b|EACH:ITEM, or symbol show|delete NAME")
    first, _, rest = addresses.strip().partition(" ")
    if named.is_reference(first):
        return {"name": name.strip(), "result": first, **named.parse_args(rest)}
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


def resolve_verb(word: str, local: tuple[str, ...] = ()) -> str:
    """A verb as typed -> its op name: exact, alias, or unique prefix (``local`` adds the
    REPL's own commands to the candidates)."""
    ops = registry()
    if word in ops or word in local:
        return word
    if word in ALIASES:
        return ALIASES[word]
    names = sorted(set(ops) | set(local))
    found = [name for name in names if name.startswith(word)]
    if len(found) == 1:
        return found[0]
    if not found:
        raise UnknownOp(f'undefined command "{word}"; try "help"')
    raise UnknownOp(f'ambiguous command "{word}": {", ".join(found)}')


def route(name: str, rest: str) -> tuple[str, str]:
    """``set load NAME FILE``, ``set define NAME JSON`` and ``set show NAME`` are the
    ``promptset`` op (no address is named ``load``, ``define`` or ``show``); every other line
    keeps its verb."""
    words = rest.split(None, 1)
    if name == "set" and words and words[0] in ("load", "define", "show"):
        return "promptset", rest
    return name, rest


_EACH_USAGE = "write each SET [at=ADDR] [prefill=K|@ROLE] [keep]: OP LINE [; OP LINE ...]"


def _each_args(text: str) -> dict[str, Any]:
    """``each SET [at=ADDR] [prefill=K|@ROLE] [keep]: LINE``: the set and options before the
    first ``:``, the op line(s) after it, kept as text (the op parses them once, as a
    template)."""
    head, colon, line = text.partition(":")
    words = head.split()
    if not colon or not words or not line.strip():
        raise BadArgs(_EACH_USAGE)
    args: dict[str, Any] = {"set": words[0], "line": line.strip()}
    for word in words[1:]:
        if word == "keep":
            args["keep"] = True
        elif word.startswith("at="):
            args["at"] = word[3:]
        elif word.startswith("prefill=") and len(word) > 8:
            args["prefill"] = word[8:]
        else:
            raise BadArgs(_EACH_USAGE)
    return args


SPECIAL["each"] = _each_args

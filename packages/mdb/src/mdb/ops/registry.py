"""The one operation registry: every front end (REPL, ``--json``, server, remote worker)
dispatches through ``dispatch(ctx, name, args)``.

Ops take and return plain JSON: addresses travel as strings, tensors never (a tensor result is
a reduction). Every result passes ``check_bounded`` (64 KiB, no arrays). Every claimed number
is a ``num(value, authority)`` mapping, or sits inside a mapping with an ``authority`` key, or
is an integer under a key in ``INTEGER_KEYS`` (ids, indices, shapes, counts).

Effects: ``read`` changes nothing; ``navigate`` moves the cursor or the focus (it may run the
native model, replayably); ``mutate`` changes run values (edits) or debugger tables
(breakpoints, watches, displays, checkpoints); ``native`` builds a new run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mdb.core.pointers import RecordError, check_bounded

from .errors import BadArgs, OpError, Refused, UnknownOp, internal

EFFECTS = ("read", "navigate", "mutate", "native")
ARG_TYPES = (
    "str", "int", "number", "bool", "address", "pattern", "condition", "choice", "ints",
    "value", "prompt",
)
SCHEMA = "mdb-ops-v1"

# Integer fields that are identities, positions, shapes or counts, not claims about values.
INTEGER_KEYS = frozenset({
    "id", "ids", "index", "indices", "shape", "selected_shape", "row", "layer", "rows",
    "count", "hits", "n", "k", "offset", "tokens_total", "bytes", "level", "transition",
    "layers", "hidden_size", "heads", "kv_heads", "head_dim", "vocab_size", "intermediate_size",
    "max_positions", "budget", "ignore", "deleted", "numel", "argmax", "argmax_index",
    "position", "prompt_tokens", "staged", "committed", "token", "step",
    "total", "limit", "omitted", "hits_total", "nan", "posinf", "neginf", "edits_total",
    "deleted_total", "mask_count", "mask_shape", "under_edits_total", "staged_total", "committed_total", "depth", "line", "lines",
    "kv", "d", "staged_at", "staged_cells", "committed_cells", "realised_in_layer",
    "prefill", "first_row", "prompt_rows",       # a record's procedure (prefill split, M54)
})


@dataclass(frozen=True)
class Arg:
    """One argument. ``positional`` args fill in order from a command line."""

    name: str
    type: str
    required: bool = False
    default: Any = None
    choices: tuple = ()
    help: str = ""
    positional: bool = True
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.type not in ARG_TYPES:
            raise ValueError(f"arg type {self.type!r} is not one of {ARG_TYPES}")
        if self.type == "choice" and not self.choices:
            raise ValueError(f"choice arg {self.name} needs choices")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type, "required": self.required}
        if self.default is not None:
            out["default"] = self.default
        if self.choices:
            out["choices"] = list(self.choices)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.help:
            out["help"] = self.help
        return out


@dataclass(frozen=True)
class Op:
    name: str
    summary: str
    args: tuple[Arg, ...]
    effect: str
    needs_live: bool
    fn: Callable[..., dict[str, Any]]
    motion: bool = False
    milestone: str | None = None
    usage: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effect not in EFFECTS:
            raise ValueError(f"effect must be one of {EFFECTS}")
        names = [arg.name for arg in self.args]
        if len(names) != len(set(names)):
            raise ValueError(f"op {self.name} repeats an argument name")

    def to_json(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "summary": self.summary,
            "effect": self.effect,
            "needs_live": self.needs_live,
            "motion": self.motion,
            "args": [arg.to_json() for arg in self.args],
            "usage": self.usage or self.name,
        }
        if self.milestone is not None:
            out["refused_until"] = self.milestone
        return out


_REGISTRY: dict[str, Op] = {}


def register(op: Op, *, replace: bool = False) -> Op:
    if op.name in _REGISTRY and not replace:
        raise ValueError(f"op {op.name!r} is already registered")
    _REGISTRY[op.name] = op
    return op


def op(
    name: str,
    summary: str,
    *args: Arg,
    effect: str,
    needs_live: bool = True,
    motion: bool = False,
    usage: str = "",
):
    """Decorator registering ``fn(ctx, **args) -> dict`` as an op."""

    def wrap(fn):
        register(Op(name, summary, tuple(args), effect, needs_live, fn, motion, usage=usage))
        return fn

    return wrap


def slot(name: str, summary: str, milestone: str, usage: str = "") -> Op:
    """A named op that exists in the catalogue but refuses until its milestone lands."""

    def refuse(ctx, **_):
        raise Refused(f"not built yet: {milestone}")

    return register(
        Op(name, summary, (), "read", False, refuse, milestone=milestone, usage=usage,
           extra={"accepts_any": True})
    )


def registry() -> dict[str, Op]:
    return dict(_REGISTRY)


# (before, after): ``before(ctx, name, values) -> (values, notes)`` rewrites validated arguments
# (``ops.named`` expands symbol names into addresses; its notes go in the result as
# ``symbol_uses``), ``after(ctx, name, result)`` observes each successful result.
_HOOKS: list[tuple[Callable[..., Any] | None, Callable[..., Any] | None]] = []


def hook(*, before: Callable[..., Any] | None = None,
         after: Callable[..., Any] | None = None) -> None:
    """Register an argument rewriter and/or a result observer run by ``dispatch``."""
    if (before, after) not in _HOOKS:
        _HOOKS.append((before, after))


def describe() -> dict[str, Any]:
    """The whole catalogue as JSON (name, summary, args, effect, needs_live)."""
    return {"schema": SCHEMA, "ops": [_REGISTRY[name].to_json() for name in sorted(_REGISTRY)]}


def _check(arg: Arg, value: Any) -> Any:
    def bad(what: str):
        return BadArgs(f"argument {arg.name!r} must be {what}, got {value!r}")

    kind = arg.type
    if kind == "bool":
        if not isinstance(value, bool):
            raise bad("true or false")
        return value
    if kind == "int":
        if type(value) is not int:
            raise bad("an integer")
        if arg.minimum is not None and value < arg.minimum:
            raise bad(f">= {arg.minimum}")
        if arg.maximum is not None and value > arg.maximum:
            raise bad(f"<= {arg.maximum}")
        return value
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad("a number")
        if not math.isfinite(value):
            raise bad("finite")
        return value
    if kind == "choice":
        if value not in arg.choices:
            raise bad("one of " + ", ".join(map(str, arg.choices)))
        return value
    if kind == "ints":
        if not isinstance(value, list) or not value or any(type(v) is not int for v in value):
            raise bad("a non-empty list of integers")
        return list(value)
    if kind == "value":
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad("a number or an address")
        if not math.isfinite(value):
            raise bad("finite")
        return value
    if kind == "prompt":
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value and all(type(v) is int and v >= 0 for v in value):
            return list(value)
        raise bad("prompt text or a non-empty list of token ids")
    if not isinstance(value, str) or not value.strip():
        raise bad("non-empty text")
    if len(value) > 4096:
        raise bad("at most 4096 characters")
    return value.strip()


def validate(op: Op, args: dict[str, Any] | None) -> dict[str, Any]:
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise BadArgs(f"{op.name}: arguments must be a JSON object")
    if op.extra.get("accepts_any"):
        return {}
    known = {arg.name: arg for arg in op.args}
    unknown = sorted(set(args) - set(known))
    if unknown:
        raise BadArgs(f"{op.name}: unknown argument {unknown[0]!r} "
                      f"(arguments: {', '.join(known) or 'none'})")
    out = {}
    for arg in op.args:
        if arg.name in args and args[arg.name] is not None:
            out[arg.name] = _check(arg, args[arg.name])
        elif arg.required:
            raise BadArgs(f"{op.name}: missing argument {arg.name!r}")
        else:
            out[arg.name] = arg.default
    return out


def has_unresolved(node: Any) -> bool:
    """Whether any field of a result is ``{"state": "unresolved", ...}``."""
    if isinstance(node, dict):
        if node.get("state") == "unresolved":
            return True
        return any(has_unresolved(item) for item in node.values())
    if isinstance(node, (list, tuple)):
        return any(has_unresolved(item) for item in node)
    return False


def dispatch(ctx: Any, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate, run one op, and return its bounded JSON result.

    Raises ``UnknownOp``, ``BadArgs`` or ``Refused`` (all ``OpError``); a ``ValueError`` from
    the engine (a request the run cannot honour, such as a target behind the cursor) becomes
    ``Refused``, and any other ``Exception`` becomes ``OpError(kind="internal")`` carrying the
    exception type and message, so no traceback ever reaches a front end.

    Every result carries a top-level ``state``: ``resolved``, or ``unresolved`` when some
    requested part could not be computed (that part is ``{"state": "unresolved", "reason"}``).
    """
    found = _REGISTRY.get(name)
    if found is None:
        raise UnknownOp(f"no operation {name!r}")
    values = validate(found, args)
    if found.needs_live and getattr(ctx, "run", None) is None:
        raise Refused(f"{name} needs a live run; start one with `start`")
    notes: list[Any] = []
    try:
        for before, _ in _HOOKS:
            if before is not None:
                values, extra = before(ctx, found.name, values)
                notes += extra
        result = found.fn(ctx, **values)
    except OpError:
        raise
    except ValueError as error:
        raise Refused(str(error)) from error
    except Exception as error:  # noqa: BLE001 - the contract: never a traceback to the user
        raise internal(error) from error
    body = {"op": found.name, **result}
    if notes:
        body.setdefault("symbol_uses", notes)
    body.setdefault("state", "unresolved" if has_unresolved(result) else "resolved")
    try:
        out = check_bounded(body, f"{name} result")
    except RecordError as error:
        raise Refused(f"result exceeds the record bound: {error}") from error
    for _, after in _HOOKS:
        if after is not None:
            after(ctx, found.name, out)
    return out

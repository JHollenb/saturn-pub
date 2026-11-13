"""Breakpoints, watches and run-to-hit over a port-level ``Run``, kept as plain data.

A breakpoint is an address pattern (``L*.p``, ``t5.L1.o``, ``t*.logits``) plus an optional
condition; a watch is an address pattern plus an optional predicate and a mode:

* ``change``: fires when the selected value differs (bitwise) from the value seen at the
  previous matching stop (and the predicate, if any, holds). The first evaluation only
  records a baseline.
* ``level``: fires at every matching stop where the predicate holds.
* ``edge``: fires where the predicate holds and did not hold at the previous matching stop
  (the state before the first evaluation counts as not holding).

A pattern's selector (``L*.p[h0]``) does not restrict where it stops; it selects the part of
the port value that conditions and watches see.

Conditions are a small safe expression language, parsed with ``ast.parse(mode="eval")`` and
checked against a node whitelist::

    norm > 3 and argmax == 2          # reductions of the (selected) port value
    abs_max(v["h0,src=3"]) < 0.5       # a reduction of an address selector on the full value
    v[0, 0, 3] > 0                     # one element of the selected value (kept dimensions)

Reductions: ``norm min max mean abs_max argmax numel``. Allowed syntax: numeric literals,
``+ - * /``, comparisons, ``and or not``, unary minus, subscripts of ``v`` (integers or one
quoted selector) and calls of the reductions with one argument. Nothing else.

``run_forward`` walks the stops after the cursor in order, and at every stop matched by an
armed breakpoint or watch moves the cursor there with ``Run.until`` (exact port), evaluates,
and either stops or keeps going, until a hit, a token budget, or the end of the context.
``run_reverse`` walks backwards reading past values without moving, then moves to the hit
with ``Run.reverse_step``. Neither edits ``run.py`` or ``ports.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from mdb.core import address as addresses

from .ports import tensor_digest

GLOBALS = (*addresses.DEFAULT_GLOBALS, "start")
_D = ("batch", "row", "d_model")
GLOBAL_AXES = {
    "start": ("batch", "row"),
    "emb": _D,
    "fn": _D,
    "logits": ("batch", "vocab"),
    "sample": ("batch", "row"),
}
CELL_AXES = ("batch", "kv_head", "row", "head_dim")
REDUCTIONS = ("norm", "min", "max", "mean", "abs_max", "argmax", "numel")
WATCH_MODES = ("change", "level", "edge")
_MAX_CONDITION = 256
_MAX_NODES = 64


class ConditionError(ValueError):
    """A condition is malformed or uses syntax outside the whitelist."""


class SelectionError(ValueError):
    """A selector does not fit the value's axes."""


class SelectionRange(SelectionError):
    """A selector (or a breakpoint layer) is outside the size of its axis. Checked against the
    axis size before anything is materialised, so ``d=0:10**15`` costs nothing."""


# --------------------------------------------------------------------------- addresses


def stop_occurrence(row: int, label: str) -> addresses.Occurrence:
    """The occurrence address of a ``Run`` stop label (``L0.p``, ``logits``, ``start``)."""
    if label.startswith("L") and "." in label:
        layer, port = label.split(".", 1)
        return addresses.Occurrence(row, port, int(layer[1:]))
    return addresses.Occurrence(row, label)


def stop_axes(program: Any, label: str) -> tuple[str, ...]:
    if label in GLOBAL_AXES:
        return GLOBAL_AXES[label]
    return program.spec(label.split(".", 1)[1]).axes


def parse_pattern(text: str, program: Any) -> addresses.Pattern:
    """A breakpoint pattern; a bare port means ``L*.<port>``, a bare global ``t*.<global>``."""
    text = text.strip()
    base = text.split("[", 1)[0]
    if base in program.names:
        text = "L*." + text
    elif base in GLOBALS:
        text = "t*." + text
    pattern = addresses.parse_pattern(text, ports=program.names, global_ports=GLOBALS)
    if pattern.origin is not None or pattern.program is not None:
        raise addresses.AddressError("breakpoint origins and programs are not supported yet")
    if pattern.kind == "cell" and not isinstance(pattern.layer, int):
        raise addresses.AddressError("a cache-cell breakpoint needs a concrete layer, e.g. "
                                     "cache.L0.k[kv0]")
    if pattern.step is not None:
        raise addresses.AddressError("the decoder engine has no diffusion-step clock")
    return pattern


def location(pattern: addresses.Pattern) -> addresses.Pattern:
    """The pattern without its selector: where it stops (the selector picks what it sees)."""
    return dataclasses.replace(pattern, selector=addresses.Selector())


def stops_at(pattern: addresses.Pattern, occurrence: addresses.Occurrence) -> bool:
    """Whether a debugger stop matches a pattern; selectors never constrain a stop. A cache
    cell (``cache.L0.k[...]``) is read at the committed boundary, so it matches boundary stops
    only (the globals and every layer's ``h_out``)."""
    if pattern.kind == "cell":
        return occurrence.port in GLOBALS or occurrence.port == "h_out"
    matcher = getattr(addresses, "matches_stop", None)
    if matcher is not None:
        return matcher(pattern, occurrence)
    return addresses.matches(location(pattern), occurrence)


# --------------------------------------------------------------------------- selection


@dataclass
class Selection:
    """A selected part of a value: the sub-tensor (dimensions kept) and original indices."""

    value: torch.Tensor
    axes: tuple[str, ...]
    coords: list[list[int]]

    def original(self, flat: int) -> list[int]:
        index = []
        for size, coord in zip(reversed(self.value.shape), reversed(self.coords)):
            index.append(coord[flat % size])
            flat //= size
        return index[::-1]


def _indices(value: Any, size: int | None, key: str, axis: str = "") -> Sequence[int]:
    """The picked positions of one selector key: an int list or a lazy ``range``, validated
    against the axis ``size`` (None: unknown here, checked at evaluation) before any expansion."""
    where = f" (axis {axis!r} has size {size})" if axis else f" (size {size})"
    if isinstance(value, int):
        if size is not None and value >= size:
            raise SelectionRange(f"selector {key}={value} is out of range{where}")
        return (value,)
    if isinstance(value, addresses.Span):
        if size is not None and value.stop > size:
            raise SelectionRange(
                f"selector {key}={value.start}:{value.stop} is out of range{where}")
        return range(value.start, value.stop)
    items = tuple(value.values)
    if size is not None:
        bad = [item for item in items if item >= size]
        if bad:
            raise SelectionRange(f"selector {key}={bad[0]} is out of range{where}")
    return items


def _token_ids(tokens: tuple, tokenizer: Any) -> list[int]:
    ids = []
    for token in tokens:
        if token.id is not None:
            ids.append(token.id)
            continue
        if tokenizer is None:
            raise SelectionError(
                f"token text {token.text!r} needs a tokenizer; write it as {token.text!r}=ID"
            )
        encoded = tokenizer.encode(token.text, add_special_tokens=False)
        if len(encoded) != 1:
            raise SelectionError(f"{token.text!r} is {len(encoded)} tokens, not one")
        ids.append(int(encoded[0]))
    return ids


def plan(
    axes: tuple[str, ...],
    sizes: Sequence[int | None],
    selector: addresses.Selector,
    *,
    head_dim: int | None = None,
    row: int | None = None,
    tokenizer: Any = None,
) -> list[Sequence[int] | None]:
    """Bound-check a selector against axis ``sizes`` (None = unknown) and return the picked
    positions per axis (None = the whole axis). Nothing is allocated beyond the axis size."""
    if len(axes) != len(sizes):
        raise SelectionError(f"value has {len(sizes)} dimensions, axes {axes}")
    picks: list[Sequence[int] | None] = [None] * len(axes)
    keys = dict(selector.keys)

    def axis(*names: str) -> int | None:
        for name in names:
            if name in axes:
                return axes.index(name)
        return None

    for key, flat, block in (("h", "head*head_dim", "head"), ("kv", "kv_head*head_dim", "kv_head")):
        if key not in keys:
            continue
        at = axis(block)
        if at is not None:
            picks[at] = _indices(keys.pop(key), sizes[at], key, axes[at])
            continue
        at = axis(flat)
        if at is None or not head_dim:
            raise SelectionError(f"selector {key} does not apply to axes {axes}")
        count = None if sizes[at] is None else sizes[at] // head_dim
        heads = _indices(keys.pop(key), count, key, block)
        offsets: Sequence[int] = range(head_dim)
        if "d" in keys:
            offsets = _indices(keys.pop("d"), head_dim, "d", "head_dim")
        picks[at] = [h * head_dim + d for h in heads for d in offsets]
    for key, names in (("src", ("src",)), ("i", ("d_ff", "vocab")),
                       ("d", ("d_model", "head_dim", "head*head_dim", "kv_head*head_dim"))):
        if key not in keys:
            continue
        at = axis(*names)
        if at is None:
            raise SelectionError(f"selector {key} does not apply to axes {axes}")
        picks[at] = _indices(keys.pop(key), sizes[at], key, axes[at])
    if "row" in keys:
        at = axis("row")
        if at is None:
            raise SelectionError(f"selector row does not apply to axes {axes}")
        wanted = keys.pop("row")
        if row is not None:
            if wanted != row:
                raise SelectionError(f"this value holds only row {row}")
            picks[at] = (0,)
        else:
            picks[at] = _indices(wanted, sizes[at], "row", axes[at])
    if selector.tokens:
        at = axis("vocab")
        if at is None:
            raise SelectionError(f"token selectors apply to a vocabulary axis, not {axes}")
        if picks[at] is not None:
            raise SelectionError("select vocabulary entries by i or by token text, not both")
        ids = _token_ids(selector.tokens, tokenizer)
        for token_id in ids:
            _indices(token_id, sizes[at], "token", axes[at])
        picks[at] = ids  # given order kept: M[" hot"," cold"] is an ordered contrast
    return picks


def select(
    value: torch.Tensor,
    axes: tuple[str, ...],
    selector: addresses.Selector,
    *,
    head_dim: int | None = None,
    row: int | None = None,
    tokenizer: Any = None,
) -> Selection:
    """Apply an address selector to a port value with the given axis names.

    Keys map to axes: ``h`` head (or a head block of a flat ``head*head_dim`` axis), ``kv`` kv
    head (or block), ``src`` source position, ``row`` token row, ``i`` the d_ff or vocabulary
    axis, ``d`` d_model, head_dim, or the offset inside a selected head block. Token text
    selects vocabulary entries. ``row`` is absolute: when the value holds only the clock row
    (``row`` given), the selector must name that row. Ranges are checked against the axis size
    before they are expanded (``SelectionRange``).
    """
    if len(axes) != value.dim():
        raise SelectionError(f"value has {value.dim()} dimensions, axes {axes}")
    picks = plan(axes, list(value.shape), selector, head_dim=head_dim, row=row,
                 tokenizer=tokenizer)
    out = value
    coords = []
    for at, pick in enumerate(picks):
        if pick is None:
            coords.append(list(range(value.shape[at])))
            continue
        pick = list(pick)
        out = out.index_select(at, torch.tensor(pick, dtype=torch.long, device=out.device))
        coords.append(pick)
    return Selection(out, tuple(axes), coords)


def axis_sizes(adapter: Any, axes: tuple[str, ...]) -> list[int | None]:
    """Axis sizes knowable without a value (None: depends on the row, e.g. ``src``)."""
    heads, kv = getattr(adapter, "num_heads", None), getattr(adapter, "num_kv_heads", None)
    head_dim = getattr(adapter, "head_dim", None)
    inter = getattr(adapter, "intermediate_size", None)
    if inter is None:
        inter = getattr(getattr(getattr(adapter, "model", None), "config", None),
                        "intermediate_size", None)
    known = {
        "batch": 1, "d_model": getattr(adapter, "hidden_size", None), "head": heads,
        "kv_head": kv, "head_dim": head_dim, "d_ff": inter,
        "vocab": getattr(adapter, "vocab_size", None),
        "head*head_dim": None if heads is None or head_dim is None else heads * head_dim,
        "kv_head*head_dim": None if kv is None or head_dim is None else kv * head_dim,
    }
    return [known.get(name) for name in axes]


def _float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(torch.float64)


def nonfinite(value: torch.Tensor) -> dict[str, int]:
    """Integer counts of NaN, +Inf and -Inf elements."""
    flat = value.detach().reshape(-1)
    if not flat.is_floating_point():
        return {"nan": 0, "posinf": 0, "neginf": 0}
    return {"nan": int(torch.isnan(flat).sum()), "posinf": int(torch.isposinf(flat).sum()),
            "neginf": int(torch.isneginf(flat).sum())}


def reduce(selection: Selection) -> dict[str, Any]:
    """Plain-float reductions of a selection (callers attach authority labels).

    With NaN/Inf elements the statistics run over the finite elements (``over: "finite"``)
    and ``nonfinite`` holds the integer counts; with no finite element there are no statistics
    and ``unresolved`` says why.
    """
    value = selection.value
    if value.numel() == 0:
        raise SelectionError("the selection is empty")
    flat = _float(value).reshape(-1)
    finite = torch.isfinite(flat)
    bad = int(flat.numel() - finite.sum())
    out: dict[str, Any] = {"numel": int(value.numel())}
    if bad:
        out["over"] = "finite"
        out["nonfinite"] = nonfinite(flat)
        if bad == flat.numel():
            out["unresolved"] = "no finite element"
            return out
        positions = finite.nonzero().reshape(-1)
        work = flat[positions]
    else:
        positions, work = None, flat
    at = int(torch.argmax(work))
    argmax = at if positions is None else int(positions[at])
    out.update({
        "norm": float(torch.linalg.vector_norm(work)),
        "min": float(work.min()),
        "max": float(work.max()),
        "mean": float(work.mean()),
        "abs_max": float(work.abs().max()),
        "argmax": argmax,
        "argmax_index": selection.original(argmax),
    })
    return out


def top(selection: Selection, k: int) -> list[tuple[list[int], float]]:
    """The ``k`` largest finite entries by absolute value (stable: ties keep flat order);
    non-finite elements are not ranked (``nonfinite`` counts them)."""
    flat = _float(selection.value).reshape(-1)
    finite = torch.isfinite(flat)
    if bool(finite.all()):
        order = torch.sort(-flat.abs(), stable=True).indices[:k].tolist()
    else:
        positions = finite.nonzero().reshape(-1)
        ranked = torch.sort(-flat[positions].abs(), stable=True).indices[:k]
        order = positions[ranked].tolist()
    return [(selection.original(i), float(flat[i])) for i in order]


# --------------------------------------------------------------------------- conditions

_COMPARE = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}
_BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def parse_selector(text: str) -> addresses.Selector:
    try:
        return addresses.parse(f"t0.L0.h[{text}]").selector
    except addresses.AddressError as error:
        raise ConditionError(f"bad selector {text!r}: {error.reason}") from None


class Condition:
    """A parsed, whitelisted condition over reductions of one port value."""

    def __init__(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ConditionError("a condition is non-empty text")
        if len(text) > _MAX_CONDITION:
            raise ConditionError(f"a condition is at most {_MAX_CONDITION} characters")
        self.text = text.strip()
        try:
            tree = ast.parse(self.text, mode="eval")
        except SyntaxError as error:
            raise ConditionError(f"invalid condition: {error.msg}") from None
        self._nodes = 0
        self._check(tree.body)
        self.tree = tree.body

    # --- whitelist -----------------------------------------------------------------------

    def _check(self, node: ast.AST, *, value_ok: bool = False) -> None:
        self._nodes += 1
        if self._nodes > _MAX_NODES:
            raise ConditionError("condition is too complex")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ConditionError("only numeric literals are allowed")
            if not math.isfinite(node.value):
                raise ConditionError("literals must be finite")
            return
        if isinstance(node, ast.Name):
            if node.id in REDUCTIONS:
                return
            if node.id == "v" and value_ok:
                return
            raise ConditionError(
                f"unknown name {node.id!r}; use {', '.join(REDUCTIONS)} or v[...]"
            )
        if isinstance(node, ast.BoolOp):
            for item in node.values:
                self._check(item)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            self._check(node.operand)
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self._check(node.left)
            self._check(node.right)
            return
        if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
            self._check(node.left)
            for item in node.comparators:
                self._check(item)
            return
        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "v"):
                raise ConditionError("only v[...] may be subscripted")
            self._check_index(node.slice)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in REDUCTIONS:
                raise ConditionError(f"only reductions may be called: {', '.join(REDUCTIONS)}")
            if node.keywords or len(node.args) != 1:
                raise ConditionError(f"{node.func.id}(...) takes one value")
            self._check(node.args[0], value_ok=True)
            return
        raise ConditionError(f"syntax {type(node).__name__} is not allowed in a condition")

    def _check_index(self, node: ast.AST) -> None:
        items = node.elts if isinstance(node, ast.Tuple) else [node]
        if len(items) == 1 and isinstance(items[0], ast.Constant) and isinstance(
            items[0].value, str
        ):
            return
        for item in items:
            if isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub):
                item = item.operand
            if not (isinstance(item, ast.Constant) and type(item.value) is int):
                raise ConditionError('v[...] takes integers or one quoted selector, v["h0"]')

    def selector_texts(self) -> list[str]:
        """The quoted selector strings of ``v["..."]`` subscripts (checked at creation)."""
        found = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Subscript):
                items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
                if len(items) == 1 and isinstance(items[0], ast.Constant) and isinstance(
                    items[0].value, str
                ):
                    found.append(items[0].value)
        return found

    # --- evaluation ----------------------------------------------------------------------

    def evaluate(self, env: ConditionEnv) -> bool:
        return bool(self._eval(self.tree, env))

    def _eval(self, node: ast.AST, env: ConditionEnv) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "v":
                return env.selection
            return env.reduction(node.id, env.selection)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval(item, env) for item in node.values)
            return any(self._eval(item, env) for item in node.values)
        if isinstance(node, ast.UnaryOp):
            operand = _scalar(self._eval(node.operand, env))
            if isinstance(node.op, ast.Not):
                return not operand
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            left = _scalar(self._eval(node.left, env))
            right = _scalar(self._eval(node.right, env))
            if isinstance(node.op, ast.Div) and right == 0:
                raise ConditionError("division by zero in a condition")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.Compare):
            left = _scalar(self._eval(node.left, env))
            for op, item in zip(node.ops, node.comparators):
                right = _scalar(self._eval(item, env))
                if not _COMPARE[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.Subscript):
            return env.subscript(node.slice)
        if isinstance(node, ast.Call):
            return env.reduction(node.func.id, self._eval(node.args[0], env))
        raise ConditionError(f"cannot evaluate {type(node).__name__}")


def _scalar(value: Any) -> float:
    if isinstance(value, Selection):
        if value.value.numel() != 1:
            raise ConditionError(
                f"a value of {value.value.numel()} elements is not a number; reduce it"
            )
        return float(value.value.reshape(-1)[0].to(torch.float64))
    return value


@dataclass
class ConditionEnv:
    """What a condition sees at one stop: the full port value and the pattern's selection."""

    full: torch.Tensor
    selection: Selection
    axes: tuple[str, ...]
    head_dim: int | None = None
    row: int | None = None
    tokenizer: Any = None

    def reduction(self, name: str, value: Any) -> Any:
        if not isinstance(value, Selection):
            raise ConditionError(f"{name}(...) needs a value, not a number")
        result = reduce(value)
        if "unresolved" in result and name != "numel":
            raise ConditionError(f"{name}(...): {result['unresolved']}")
        return result[name]

    def subscript(self, node: ast.AST) -> Selection:
        items = node.elts if isinstance(node, ast.Tuple) else [node]
        if isinstance(items[0], ast.Constant) and isinstance(items[0].value, str):
            selector = parse_selector(items[0].value)
            return select(self.full, self.axes, selector, head_dim=self.head_dim, row=self.row,
                          tokenizer=self.tokenizer)
        index = []
        for item in items:
            negative = isinstance(item, ast.UnaryOp)
            raw = (item.operand if negative else item).value
            index.append(-raw if negative else raw)
        value = self.selection.value
        if len(index) != value.dim():
            raise ConditionError(f"v[...] needs {value.dim()} indices (dimensions are kept)")
        try:
            element = value[tuple(index)]
        except IndexError as error:
            raise ConditionError(f"v{index} is out of range: {error}") from None
        flat = 0
        for size, i in zip(value.shape, index):
            flat = flat * size + (i % size)
        coords = [[c] for c in self.selection.original(flat)]
        return Selection(element.reshape([1] * value.dim()), self.selection.axes, coords)


# --------------------------------------------------------------------------- provenance

UNDER_SHOWN = 8


def _selected_mask(shape: tuple[int, ...] | None, selection: Selection | None) -> torch.Tensor | None:
    """The selection as a bool mask over the port (None when it is the whole port or does not
    fit the port's shape)."""
    if selection is None or shape is None or len(selection.coords) != len(shape) or any(
        len(c) and max(c) >= size for c, size in zip(selection.coords, shape)
    ):
        return None
    chosen = torch.zeros(shape, dtype=torch.bool)
    index = tuple(torch.tensor(c, dtype=torch.long).view(
        [-1 if a == axis else 1 for a in range(len(shape))])
        for axis, c in enumerate(selection.coords))
    chosen[index] = True
    return chosen


def _unedited(run: Any, port: str | None) -> bool:
    """No edit anywhere in the run (the label is then measured without asking the engine).
    ``s`` is a replica port: it is always labelled by the engine."""
    return port != "s" and not run.edits and not run.committed_edits()


def provenance(run: Any, address: str, selection: Selection | None = None) -> dict[str, Any]:
    """Where a (selected) value came from, from ``Run.read_info`` and ``Run.edit_mask``.

    * ``authority`` ``measured | derived-exact | derived-local | supplied`` (None when the
      engine refused to label it, with ``state``/``reason``) and ``kind``
      (``native | edit | realisation | replica``);
    * a selection of an edited port is labelled by the operations of the edits whose masks
      cover the selected elements, staged or committed alike: every selected element's last
      write a replace -> ``supplied``; some written -> ``derived-exact`` (``mixed`` counts when
      some elements were not); none written -> the native label with ``same_port_edit``;
    * ``under_edits``: edits that fired before this value in program order (it keeps its own
      label, ``measured``), the first few as ``{address, op, staged}`` and ``under_edits_total``.
    """
    port = address.rsplit(".", 1)[-1] if "." in address else address
    if _unedited(run, port):
        return {"authority": "measured", "edited": False, "kind": "native", "by": None,
                "ops": [], "under_edits": [], "under_edits_total": 0, "replica_equal": None,
                "max_abs_diff": None, "port": port, "state": "resolved", "reason": None}
    wide = _selected_mask(run.address_shape(address), selection)
    info = run.read_info(address, select=wide)
    ups = info.get("upstream_edits") or []
    out: dict[str, Any] = {
        "authority": info["authority"], "edited": bool(info["edited"]), "kind": info["kind"],
        "by": info["by"], "ops": list(info.get("ops") or []),
        "under_edits": [{"address": f"t{u['row']}.L{u['layer']}.{u['port']}", "op": u["op"],
                         "staged": bool(u["staged"])} for u in ups[:UNDER_SHOWN]],
        "under_edits_total": len(ups),
        "replica_equal": info.get("replica_equal"), "max_abs_diff": info.get("max_abs_diff"),
        "replica_checked": info.get("replica_checked"), "port": port, "state": info["state"], "reason": info["reason"],
    }
    for key in ("realisation_exact", "realisation_max_abs_error"):
        if info.get(key) is not None:
            out[key] = info[key]
    if info["edited"]:
        mask = run.edit_mask(address)
        if mask is not None:
            chosen = mask if wide is None or wide.shape != mask.shape else mask & wide
            total = int(mask.numel() if wide is None or wide.shape != mask.shape else wide.sum())
            written = int(chosen.sum())
            if 0 < written < total:
                out["mixed"] = {"written": written, "untouched": total - written}
    if info.get("same_port_edit"):
        out["same_port_edit"] = dict(info["same_port_edit"])
    return out


def _cell_rows(run: Any, selection: Selection | None) -> list[int]:
    """The cache rows a cell selection covers (all rows written so far for no selection)."""
    if selection is not None and len(selection.coords) == len(CELL_AXES):
        return list(selection.coords[CELL_AXES.index("row")])
    return list(range(run.read_cache(0, "k").shape[2]))


# The ports whose edit can change the key (value) row this transition writes into the cache,
# per tensor kind: the layer input, the normed input, and the projection itself.
_CELL_WRITERS = {"k": ("h", "a", "k", "kn"), "v": ("h", "a", "v")}


def cell_provenance(
    run: Any, layer: int, selection: Selection | None = None, tensor: str | None = None
) -> dict[str, Any]:
    """A cache cell is read at the committed boundary: ``measured``, under the committed edits
    that can have reached it. An edit at (row r, layer l) reaches the cell (cache row j, layer L)
    when j == r and either L > l or (L == l and the edit is at a port that feeds this tensor's
    cache write), or j > r and L > l (a later row reads the changed rows through attention, and
    its own cache write at layer l is computed before that layer's attention)."""
    rows = _cell_rows(run, selection)
    writers = _CELL_WRITERS.get(tensor) if tensor else None

    def reaches(e: dict[str, Any]) -> bool:
        same_layer = layer == e["layer"] and (
            writers is None or run.program.spec(e["port"]).name in writers
        )
        for j in rows:
            if j == e["row"] and (layer > e["layer"] or same_layer):
                return True
            if j > e["row"] and layer > e["layer"]:
                return True
        return False

    ups = [e for e in run.committed_edits() if reaches(e)]
    return {"authority": "measured", "edited": False, "kind": "native", "by": None, "ops": [],
            "under_edits": [{"address": f"t{e['row']}.L{e['layer']}.{e['port']}",
                             "op": e["op"], "staged": False} for e in ups[:UNDER_SHOWN]],
            "under_edits_total": len(ups), "replica_equal": None, "max_abs_diff": None,
            "state": "resolved", "reason": None}


# --------------------------------------------------------------------------- table

HITS_SHOWN = 32


@dataclass
class Breakpoint:
    id: int
    pattern: str
    parsed: Any
    condition: Condition | None = None
    temporary: bool = False
    enabled: bool = True
    hits: int = 0
    kind: str = "break"
    ignore: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "tbreak" if self.temporary else "break",
            "pattern": self.pattern,
            "condition": None if self.condition is None else self.condition.text,
            "enabled": self.enabled,
            "hits": self.hits,
            "ignore": self.ignore,
        }


@dataclass
class Watch:
    id: int
    pattern: str
    parsed: Any
    predicate: Condition | None = None
    mode: str = "change"
    enabled: bool = True
    hits: int = 0
    kind: str = "watch"
    state: dict[str, Any] = field(default_factory=dict)
    ignore: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": "watch",
            "pattern": self.pattern,
            "predicate": None if self.predicate is None else self.predicate.text,
            "mode": self.mode,
            "enabled": self.enabled,
            "hits": self.hits,
            "ignore": self.ignore,
        }


def validate_item(parsed: addresses.Pattern, conditions: Sequence[Condition | None],
                  program: Any, adapter: Any) -> None:
    """Creation-time checks where the shape is knowable: a concrete layer inside the model,
    the pattern's selector and every ``v["..."]`` selector of the conditions inside the axis
    sizes of the stop's port. Never allocates (``plan`` only reads sizes)."""
    layers = getattr(adapter, "layers", None)
    if layers is not None and isinstance(parsed.layer, int) and parsed.layer >= layers:
        raise SelectionRange(
            f"no layer {parsed.layer} in a {layers}-layer model; valid layers are 0..{layers - 1}")
    port = parsed.port
    if parsed.kind == "cell":
        axes = CELL_AXES
    elif port in GLOBAL_AXES:
        axes = GLOBAL_AXES[port]
    elif port in getattr(program, "names", ()):
        axes = program.spec(port).axes
    else:
        return
    sizes = axis_sizes(adapter, axes)
    head_dim = getattr(adapter, "head_dim", None)
    row = None
    plan(axes, sizes, parsed.selector, head_dim=head_dim, row=row, tokenizer=_NoTokens())
    for condition in conditions:
        if condition is None:
            continue
        for text in condition.selector_texts():
            plan(axes, sizes, parse_selector(text), head_dim=head_dim, row=row,
                 tokenizer=_NoTokens())


class _NoTokens:
    """Creation-time stand-in tokenizer: token text selectors are checked at evaluation."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [0]


class Table:
    """Breakpoints and watches sharing one id space (``delete`` removes either)."""

    def __init__(self) -> None:
        self.items: dict[int, Breakpoint | Watch] = {}
        self._next = 1
        self.last_stop: dict[str, Any] | None = None

    def _id(self) -> int:
        number, self._next = self._next, self._next + 1
        return number

    def add_break(
        self, pattern: str, program: Any, *, condition: str | None = None,
        temporary: bool = False, adapter: Any = None,
    ) -> Breakpoint:
        parsed = parse_pattern(pattern, program)
        compiled = None if condition is None else Condition(condition)
        validate_item(parsed, [compiled], program, adapter)
        item = Breakpoint(self._id(), addresses.format_pattern(parsed), parsed, compiled,
                          temporary)
        self.items[item.id] = item
        return item

    def add_watch(
        self, pattern: str, program: Any, *, predicate: str | None = None, mode: str = "change",
        adapter: Any = None,
    ) -> Watch:
        if mode not in WATCH_MODES:
            raise ValueError(f"watch mode is one of {', '.join(WATCH_MODES)}")
        if mode != "change" and predicate is None:
            raise ValueError(f"a {mode} watch needs a predicate")
        parsed = parse_pattern(pattern, program)
        compiled = None if predicate is None else Condition(predicate)
        validate_item(parsed, [compiled], program, adapter)
        item = Watch(self._id(), addresses.format_pattern(parsed), parsed, compiled, mode)
        self.items[item.id] = item
        return item

    def _existing(self, ids: list[int]) -> list[Breakpoint | Watch]:
        missing = [i for i in ids if i not in self.items]
        if missing:
            raise ValueError(f"no breakpoint or watch {missing[0]}")
        return [self.items[i] for i in ids]

    def delete(self, ids: list[int] | None = None) -> list[int]:
        if ids is None:
            gone = sorted(self.items)
            self.items.clear()
            return gone
        self._existing(ids)
        for i in ids:
            self.items.pop(i, None)
        return sorted(set(ids))

    def set_enabled(self, ids: list[int] | None, enabled: bool) -> list[int]:
        found = list(self.items.values()) if ids is None else self._existing(ids)
        for item in found:
            item.enabled = enabled
        return sorted({item.id for item in found})

    def set_condition(self, number: int, text: str | None, program: Any,
                      adapter: Any = None) -> Breakpoint | Watch:
        """``condition N EXPR``; empty ``EXPR`` removes it (a watch's predicate)."""
        (item,) = self._existing([number])
        compiled = None if not text or not text.strip() else Condition(text)
        if isinstance(item, Watch) and compiled is None and item.mode != "change":
            raise ValueError(f"a {item.mode} watch needs a predicate")
        validate_item(item.parsed, [compiled], program, adapter)
        if isinstance(item, Watch):
            item.predicate = compiled
        else:
            item.condition = compiled
        return item

    def set_ignore(self, number: int, count: int) -> Breakpoint | Watch:
        (item,) = self._existing([number])
        if type(count) is not int or count < 0:
            raise ValueError("an ignore count is a nonnegative integer")
        item.ignore = count
        return item

    def armed(self) -> list[Breakpoint | Watch]:
        return [item for _, item in sorted(self.items.items()) if item.enabled]

    def listed(self) -> list[Breakpoint | Watch]:
        """Every breakpoint and watch, enabled or not, by id."""
        return [item for _, item in sorted(self.items.items())]

    def reset_watches(self) -> None:
        for item in self.items.values():
            if isinstance(item, Watch):
                item.state = {}
        self.last_stop = None


# --------------------------------------------------------------------------- continue


@dataclass
class _Probe:
    """One evaluation at one stop."""

    run: Any
    row: int
    label: str
    tokenizer: Any
    _value: torch.Tensor | None = None
    _cells: dict | None = None

    @property
    def address(self) -> str:
        return f"t{self.row}.{self.label}"

    def value(self, pattern: Any = None) -> torch.Tensor:
        if pattern is not None and pattern.kind == "cell":
            # one read per (layer, tensor): watches on cache.L0.k and cache.L0.v at the same
            # stop must not share a value
            if self._cells is None:
                self._cells = {}
            key = (pattern.layer, pattern.port)
            if key not in self._cells:
                self._cells[key] = self.run.read_cache(pattern.layer, pattern.port)
            return self._cells[key]
        if self._value is None:
            self._value = self.run.read(self.address)
        return self._value

    def prov(self, pattern: Any, selection: Selection) -> dict[str, Any]:
        if pattern.kind == "cell":
            return cell_provenance(self.run, pattern.layer, selection, pattern.port)
        return provenance(self.run, self.address, selection)

    def env(self, pattern: Any) -> ConditionEnv:
        if pattern.kind == "cell":
            full = self.value(pattern)
            selection = select(full, CELL_AXES, pattern.selector,
                               head_dim=self.run.adapter.head_dim, row=None,
                               tokenizer=self.tokenizer)
            return ConditionEnv(full, selection, CELL_AXES, self.run.adapter.head_dim, None,
                                self.tokenizer)
        axes = stop_axes(self.run.program, self.label)
        full = self.value()
        row = None if self.label in ("start",) else self.row
        if "row" not in axes:
            row = None
        selection = select(full, axes, pattern.selector, head_dim=self.run.adapter.head_dim,
                           row=row, tokenizer=self.tokenizer)
        return ConditionEnv(full, selection, axes, self.run.adapter.head_dim, row, self.tokenizer)


def _evaluate(item: Breakpoint | Watch, probe: _Probe, state: dict[str, Any]) -> dict | None:
    """Whether ``item`` fires at this stop; watches update ``state`` (their memory)."""
    env = probe.env(item.parsed)
    if isinstance(item, Breakpoint):
        if item.condition is not None and not item.condition.evaluate(env):
            return None
        kind = "advance" if item.kind == "advance" else item.describe()["kind"]
        return {"id": item.id, "kind": kind, "pattern": item.pattern}
    holds = True if item.predicate is None else item.predicate.evaluate(env)
    digest = tensor_digest(env.selection.value.contiguous())
    stats = reduce(env.selection)
    previous = state.get("digest")
    was = state.get("active", False)
    old = state.get("stats")
    prov = probe.prov(item.parsed, env.selection)
    old_prov = state.get("prov")
    state.update(digest=digest, active=holds, stats=stats, where=probe.address, prov=prov)
    if item.mode == "change":
        fires = previous is not None and digest != previous and holds
    elif item.mode == "level":
        fires = holds
    else:
        fires = holds and not was
    if not fires:
        return None
    hit = {"id": item.id, "kind": "watch", "pattern": item.pattern, "mode": item.mode,
           "new": stats, "new_provenance": prov}
    if old is not None:
        hit["old"] = old
        hit["old_provenance"] = old_prov
    return hit


def prime(run: Any, item: Watch, *, tokenizer: Any = None) -> bool:
    """Record a new watch's baseline at the cursor when the cursor matches it (as gdb does)."""
    row, index = run.parse(run.where)
    label = run.stops()[index]
    if not stops_at(item.parsed, stop_occurrence(row, label)):
        return False
    _evaluate(item, _Probe(run, row, label, tokenizer), item.state)
    return True


def _matching(items: list, occurrence: addresses.Occurrence) -> list:
    return [item for item in items if stops_at(item.parsed, occurrence)]


def _at_stop(items: list, matched: list, probe: _Probe, states: dict | None,
             table: Table | None) -> tuple[list[dict], dict | None]:
    """Evaluate the matched items at one stop. Returns ``(fired hits, error)``: ``error`` is
    ``{id, pattern, message}`` for the first item whose condition could not be evaluated (the
    motion stops there); ignore counts consume hits (gdb: counted, not reported)."""
    hits: list[dict] = []
    error = None
    for item in matched:
        if isinstance(item, Watch):
            state = item.state if states is None else states.setdefault(item.id, {})
        else:
            state = {}
        try:
            hit = _evaluate(item, probe, state)
        except Exception as caught:  # noqa: BLE001 - any evaluation failure stops the motion
            if error is None:
                error = {"id": item.id, "pattern": item.pattern,
                         "message": f"{type(caught).__name__}: {caught}"
                         if not isinstance(caught, ValueError) else str(caught)}
            continue
        if hit is None:
            continue
        live = None if table is None else table.items.get(item.id)
        if live is not None:
            live.hits += 1
            if live.ignore > 0:
                live.ignore -= 1
                continue
            if isinstance(live, Breakpoint) and live.temporary:
                del table.items[live.id]
        hits.append(hit)
    return hits, error


def _stopped(table: Table | None, where: str, hits: list[dict]) -> None:
    if table is not None:
        table.last_stop = {"where": where, "ids": [h["id"] for h in hits if h["id"] > 0]}


def _reason(hits: list[dict]) -> str:
    kinds = {hit["kind"] for hit in hits}
    return "watch" if kinds == {"watch"} else ("advance" if kinds == {"advance"}
                                               else "breakpoint")


def _capacity(error: ValueError) -> bool:
    return "capacity" in str(error)


def _end_of_context(run: Any, stops: tuple[str, ...], last_row: int) -> dict[str, Any]:
    """A normal stop at the last reachable stop: ``t<last>.logits`` (one row earlier when the
    engine cannot embed the final position), or where the cursor already is."""
    here = run.parse(run.where)
    at = stops.index("logits")
    for row in (last_row, last_row - 1):
        if row < 0 or (row, at) <= here:
            break
        try:
            run.until(f"t{row}.logits")
            break
        except ValueError as error:
            if not _capacity(error):
                raise
    return {"reason": "end-of-context", "where": run.where, "hits": []}


def run_forward(
    run: Any,
    table: Table | None,
    *,
    budget: int = 8,
    extra: list | None = None,
    tokenizer: Any = None,
    max_positions: int | None = None,
) -> dict[str, Any]:
    """Run to the next hit of an armed breakpoint/watch (or ``extra`` items).

    ``budget`` is the number of token rows the motion may ENTER (minimum 1): it stops at the
    ``start`` of row ``cursor row + budget`` with reason ``budget``. Per-item ignore counts
    pass hits (gdb ``ignore N COUNT``). Returns ``{reason, where, hits[, error]}`` where
    ``reason`` is ``breakpoint``, ``watch``, ``advance``, ``budget``, ``end-of-context`` (the
    model's context limit: a normal stop) or ``error`` (a condition or watch could not be
    evaluated at ``where``: the motion stops there, ``error`` has the id and message).
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    budget = max(1, budget)
    items = (table.armed() if table is not None else []) + list(extra or ())
    stops = run.stops()
    start_row, index = run.parse(run.where)
    row = start_row
    last_row = None if max_positions is None else max_positions - 1
    while True:
        index += 1
        if index == len(stops):
            row, index = row + 1, 0
        if last_row is not None and row > last_row:
            return _end_of_context(run, stops, last_row)
        if index == 0 and row - start_row >= budget:
            try:
                run.until(f"t{row}.start")
            except ValueError as error:
                if not _capacity(error) or last_row is None:
                    raise
                return _end_of_context(run, stops, last_row)
            return {"reason": "budget", "where": run.where, "hits": []}
        occurrence = stop_occurrence(row, stops[index])
        matched = _matching(items, occurrence)
        if not matched:
            continue
        try:
            run.until(f"t{row}.{stops[index]}")
        except ValueError as error:
            if not _capacity(error) or last_row is None:
                raise
            return _end_of_context(run, stops, last_row)
        probe = _Probe(run, row, stops[index], tokenizer)
        hits, error = _at_stop(items, matched, probe, None, table)
        if error is not None:
            _stopped(table, run.where, hits)
            return {"reason": "error", "where": run.where, "hits": hits, "error": error}
        if not hits:
            continue
        _stopped(table, run.where, hits)
        return {"reason": _reason(hits), "where": run.where, "hits": hits}


_BOUNDARY_UNITS = ("start", "emb", "fn", "logits", "sample")


def _boundary(label: str) -> bool:
    """Stops that are committed-cut marks: the globals and every layer's ``h_out``."""
    return label in _BOUNDARY_UNITS or label.endswith(".h_out")


def jump_back(run: Any, target: tuple[int, int]) -> None:
    """Move the cursor back to ``target`` with boundary restores (no layer is executed) and
    at most one forward re-probe inside the layer that holds ``target``."""
    stops = run.stops()
    row, index = target
    while not _boundary(stops[index]):  # the nearest committed boundary at or before it
        index -= 1
    boundary = (row, index)
    while True:
        here = run.parse(run.where)
        if here <= boundary:
            break
        previous_start = (here[0], 0) if here[1] > 0 else (here[0] - 1, 0)
        run.reverse_step("token" if previous_start >= boundary else "layer", 1)
    if boundary < target:
        run.until(f"t{target[0]}.{stops[target[1]]}")


def run_reverse(run: Any, table: Table, *, tokenizer: Any = None) -> dict[str, Any]:
    """Walk back to the previous hit and jump there with one cursor move.

    The hit is located by scanning the program's stops backwards, reading values only where a
    condition or watch needs them. Watches have their own memory on this backward walk,
    seeded with the value at the cursor when the cursor itself matches the watch, so a change
    between the previous matching stop and the cursor is found; their forward memory is left
    untouched.
    """
    stops = run.stops()
    here = run.parse(run.where)
    row, index = here
    root = (len(run.tokens) - 1, 0)
    # Cache cells are read at the current boundary only: a past boundary's cells are gone, so
    # cell breakpoints/watches are not evaluated on the backward walk.
    items = [i for i in table.armed() if i.parsed.kind != "cell"]
    states: dict[int, dict[str, Any]] = {}
    seed = _matching([i for i in items if isinstance(i, Watch)],
                     stop_occurrence(row, stops[index]))
    if seed:
        _at_stop(items, seed, _Probe(run, row, stops[index], tokenizer), states, None)
    while (row, index) > root:
        index -= 1
        if index < 0:
            row, index = row - 1, len(stops) - 1
        matched = _matching(items, stop_occurrence(row, stops[index]))
        if not matched:
            continue
        probe = _Probe(run, row, stops[index], tokenizer)
        hits, error = _at_stop(items, matched, probe, states, table)
        if hits or error is not None:
            jump_back(run, (row, index))
            if error is not None:
                _stopped(table, run.where, hits)
                return {"reason": "error", "where": run.where, "hits": hits, "error": error}
            _stopped(table, run.where, hits)
            return {"reason": _reason(hits), "where": run.where, "hits": hits}
    if (row, index) < here:
        jump_back(run, root)
    return {"reason": "start-of-run", "where": run.where, "hits": []}

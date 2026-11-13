"""Fail-closed Python-shaped algebra for live debugger Acts.

Expressions cross the debugger control plane as text and are evaluated inside
the native worker against an adapter-owned value resolver.  This module never
uses unrestricted ``eval``: only a small AST vocabulary, explicit functions,
and finite numeric constants are admitted.  Tensor arithmetic is delegated to
the value type returned by the native adapter (normally a torch Tensor).
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Sequence
from typing import Any


class DebuggerAlgebraError(ValueError):
    """An expression was malformed, unsupported, or produced an unsafe value."""


Resolver = Callable[[str], Any]


def _finite_scalar(value: Any, *, label: str) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise DebuggerAlgebraError(f"{label} must be finite")
        return value
    return value


def _check_finite(value: Any, *, label: str = "algebra result") -> Any:
    _finite_scalar(value, label=label)
    try:
        import torch

        if torch.is_tensor(value) and not bool(torch.isfinite(value).all().item()):
            raise DebuggerAlgebraError(f"{label} contains non-finite values")
    except ImportError:
        pass
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        try:
            mask = isfinite()
            all_value = getattr(mask, "all", None)
            if callable(all_value) and not bool(all_value().item()):
                raise DebuggerAlgebraError(f"{label} contains non-finite values")
        except DebuggerAlgebraError:
            raise
        except Exception as exc:  # noqa: BLE001 - tensor-like finite checks are optional
            raise DebuggerAlgebraError(f"could not validate {label}") from exc
    return value


def _mean(value: Any) -> Any:
    method = getattr(value, "mean", None)
    if callable(method):
        return method()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value:
        return sum(value) / len(value)
    if isinstance(value, (int, float)):
        return value
    raise DebuggerAlgebraError("mean() requires a numeric value")


def _norm(value: Any) -> Any:
    vector_norm = getattr(value, "norm", None)
    if callable(vector_norm):
        return vector_norm()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return math.sqrt(sum(float(item) ** 2 for item in value))
    if isinstance(value, (int, float)):
        return abs(value)
    raise DebuggerAlgebraError("norm() requires a numeric value")


def _clip(value: Any, low: Any, high: Any) -> Any:
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise DebuggerAlgebraError("clip bounds must be numeric constants")
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or low > high:
        raise DebuggerAlgebraError("clip bounds must be finite and ordered")
    clamp = getattr(value, "clamp", None)
    if callable(clamp):
        return clamp(min=float(low), max=float(high))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [min(max(item, low), high) for item in value]
    return min(max(value, low), high)


class _Evaluator:
    _BINOPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }
    _UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
    _FUNCTIONS = {
        "abs",
        "add",
        "clip",
        "copy",
        "delta",
        "mean",
        "mix",
        "norm",
        "read",
        "ref",
        "scale",
        "sub",
        "value",
        "zeros_like",
    }

    def __init__(self, resolver: Resolver, *, max_nodes: int = 128) -> None:
        self.resolver = resolver
        self.max_nodes = int(max_nodes)
        self.nodes = 0

    def evaluate(self, expression: str) -> Any:
        if not isinstance(expression, str) or not expression.strip():
            raise DebuggerAlgebraError("algebra expression must be non-empty")
        if len(expression) > 4096:
            raise DebuggerAlgebraError("algebra expression is too long")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise DebuggerAlgebraError(f"invalid algebra expression: {exc.msg}") from exc
        result = self.visit(tree.body)
        return _check_finite(result)

    def visit(self, node: ast.AST) -> Any:
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise DebuggerAlgebraError("algebra expression is too complex")
        if isinstance(node, ast.Constant):
            # String literals are restricted to logical address arguments such
            # as ``read("conditioner[text][joint][step=0][row=2]")``.  They
            # are data, never Python names, attributes, or executable code.
            if isinstance(node.value, str):
                return node.value
            if not isinstance(node.value, (int, float, bool)):
                raise DebuggerAlgebraError(
                    "only finite numeric or path-string constants are allowed"
                )
            return _finite_scalar(node.value, label="constant")
        if isinstance(node, ast.List):
            return [self.visit(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self.visit(item) for item in node.elts)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARYOPS:
            return _check_finite(self._UNARYOPS[type(node.op)](self.visit(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINOPS:
            left = self.visit(node.left)
            right = self.visit(node.right)
            try:
                return _check_finite(self._BINOPS[type(node.op)](left, right))
            except (ArithmeticError, TypeError, ValueError) as exc:
                raise DebuggerAlgebraError(f"algebra operation failed: {exc}") from exc
        if isinstance(node, ast.Name) and node.id in {"pi", "e"}:
            return math.pi if node.id == "pi" else math.e
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._FUNCTIONS:
                raise DebuggerAlgebraError("only declared debugger algebra functions are allowed")
            if node.keywords:
                raise DebuggerAlgebraError("keyword arguments are not allowed in algebra")
            args = [self.visit(arg) for arg in node.args]
            return self._call(node.func.id, args)
        raise DebuggerAlgebraError(
            f"unsupported algebra syntax: {type(node).__name__}; "
            "attributes, subscripts, comprehensions, and imports are refused"
        )

    def _read(self, args: list[Any]) -> Any:
        if len(args) != 1 or not isinstance(args[0], str) or not args[0].strip():
            raise DebuggerAlgebraError("read/ref/value/copy require one string PATH")
        return self.resolver(args[0])

    def _call(self, name: str, args: list[Any]) -> Any:
        if name in {"read", "ref", "value", "copy"}:
            return self._read(args)
        if name == "zeros_like":
            value = self._read(args)
            zeros_like = getattr(value, "new_zeros", None)
            if callable(zeros_like):
                return zeros_like(value.shape)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [0.0 for _ in value]
            return 0.0
        if name == "mean":
            if len(args) != 1:
                raise DebuggerAlgebraError("mean requires one value")
            return _mean(args[0])
        if name == "norm":
            if len(args) != 1:
                raise DebuggerAlgebraError("norm requires one value")
            return _norm(args[0])
        if name == "abs":
            if len(args) != 1:
                raise DebuggerAlgebraError("abs requires one value")
            absolute = getattr(args[0], "abs", None)
            return absolute() if callable(absolute) else abs(args[0])
        if name == "clip":
            if len(args) != 3:
                raise DebuggerAlgebraError("clip requires VALUE, LOW, HIGH")
            return _clip(*args)
        if name in {"add", "sub"}:
            if len(args) != 2:
                raise DebuggerAlgebraError(f"{name} requires two values")
            return (args[0] + args[1]) if name == "add" else (args[0] - args[1])
        if name == "scale":
            if len(args) != 2 or not isinstance(args[1], (int, float)):
                raise DebuggerAlgebraError("scale requires VALUE, finite numeric ALPHA")
            if not math.isfinite(float(args[1])):
                raise DebuggerAlgebraError("scale alpha must be finite")
            return args[0] * args[1]
        if name == "delta":
            if len(args) != 2 or not all(isinstance(arg, str) for arg in args):
                raise DebuggerAlgebraError("delta requires SOURCE_PATH, BASE_PATH")
            return self.resolver(args[0]) - self.resolver(args[1])
        if name == "mix":
            if len(args) != 3 or not isinstance(args[2], (int, float)):
                raise DebuggerAlgebraError("mix requires LEFT, RIGHT, finite numeric ALPHA")
            if not math.isfinite(float(args[2])):
                raise DebuggerAlgebraError("mix alpha must be finite")
            return args[0] + args[2] * (args[1] - args[0])
        raise DebuggerAlgebraError(f"unsupported algebra function: {name}")


def evaluate_algebra(expression: str, resolver: Resolver) -> Any:
    """Evaluate one bounded expression against an adapter-owned resolver."""

    return _Evaluator(resolver).evaluate(expression)


__all__ = ["DebuggerAlgebraError", "evaluate_algebra"]

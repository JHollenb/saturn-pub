"""A tiny typed Python-shaped console for the Saturn debugger.

The syntax looks like Python, but it is compiled to the same restricted algebra
expressions used by ``set-expr``.  It never calls ``eval`` and never executes
user code in the worker.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class DebuggerConsoleError(ValueError):
    """A console line is outside the typed debugger subset."""


@dataclass
class TypedDebuggerConsole:
    """Compile a small expression language into worker-side algebra."""

    edit: Callable[[str, Any], Any]
    preview: Callable[[], Any]
    commit: Callable[[], Any]
    abort: Callable[[], Any]
    inspect: Callable[[str], Any]
    bindings: dict[str, str] = field(default_factory=dict)
    references: dict[str, str] = field(default_factory=dict)

    _BINOPS = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.Pow: "**",
    }
    _UNARYOPS = {ast.UAdd: "+", ast.USub: "-"}
    _FUNCTIONS = {
        "read",
        "ref",
        "value",
        "copy",
        "scale",
        "add",
        "sub",
        "mix",
        "mean",
        "norm",
        "clip",
        "abs",
        "zeros_like",
        "delta",
    }

    def execute(self, line: str) -> Any:
        source = line.strip()
        if not source:
            return None
        if source in {"help", "?"}:
            return self.help()
        tree = ast.parse(source, mode="exec")
        if len(tree.body) != 1:
            raise DebuggerConsoleError("console accepts one typed statement at a time")
        statement = tree.body[0]
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            return self._command(statement.value)
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise DebuggerConsoleError("assignments must target one simple name")
            name = statement.targets[0].id
            self._safe_name(name)
            expression = self._compile(statement.value)
            self.bindings[name] = expression
            if isinstance(statement.value, ast.Call) and self._call_name(statement.value) in {
                "read",
                "ref",
                "value",
                "copy",
            }:
                path = self._path_argument(statement.value)
                self.references[name] = path
            return {"bound": name, "expression": expression}
        raise DebuggerConsoleError(
            "console supports assignments and preview(), commit(), abort(), inspect(), edit()"
        )

    def _command(self, call: ast.Call) -> Any:
        name = self._call_name(call)
        if call.keywords:
            raise DebuggerConsoleError("console keyword arguments are not allowed")
        if name in {"preview", "commit", "abort"}:
            if call.args:
                raise DebuggerConsoleError(f"{name} takes no arguments")
            return {"operation": name, "result": getattr(self, name)()}
        if name == "inspect":
            if len(call.args) != 1:
                raise DebuggerConsoleError("inspect(PATH) requires one path")
            path = self._literal_path(call.args[0])
            return self.inspect(path)
        if name == "edit":
            if len(call.args) != 2:
                raise DebuggerConsoleError("edit(PATH_OR_REF, VALUE) requires two arguments")
            target = self._target_path(call.args[0])
            value_node = call.args[1]
            value = self._compile(value_node)
            return {"operation": "edit", "result": self.edit(target, value)}
        raise DebuggerConsoleError(f"unsupported console command: {name}")

    def _compile(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, bool)) and not isinstance(node.value, complex):
                return repr(node.value)
            if isinstance(node.value, str):
                return repr(node.value)
            raise DebuggerConsoleError("only finite numbers, booleans, and paths are allowed")
        if isinstance(node, ast.Name):
            if node.id in self.bindings:
                return self.bindings[node.id]
            if node.id in {"pi", "e"}:
                return node.id
            raise DebuggerConsoleError(f"unknown typed value: {node.id}")
        if isinstance(node, (ast.List, ast.Tuple)):
            opener, closer = ("[", "]") if isinstance(node, ast.List) else ("(", ")")
            return opener + ", ".join(self._compile(item) for item in node.elts) + closer
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._UNARYOPS:
            return f"({self._UNARYOPS[type(node.op)]}{self._compile(node.operand)})"
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINOPS:
            return (
                f"({self._compile(node.left)} {self._BINOPS[type(node.op)]} "
                f"{self._compile(node.right)})"
            )
        if isinstance(node, ast.Call):
            name = self._call_name(node)
            if name not in self._FUNCTIONS:
                raise DebuggerConsoleError(f"unsupported typed algebra function: {name}")
            if node.keywords:
                raise DebuggerConsoleError("console keyword arguments are not allowed")
            if name in {"read", "ref", "value", "copy"}:
                path = self._path_argument(node)
                return f"{name}({path!r})"
            return f"{name}({', '.join(self._compile(item) for item in node.args)})"
        raise DebuggerConsoleError(
            f"unsupported console syntax: {type(node).__name__}; attributes, "
            "subscripts, and comprehensions are refused"
        )

    def _call_name(self, node: ast.Call) -> str:
        if not isinstance(node.func, ast.Name):
            raise DebuggerConsoleError("only simple debugger function names are allowed")
        return node.func.id

    def _literal_path(self, node: ast.AST) -> str:
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or not node.value.strip()
        ):
            raise DebuggerConsoleError("PATH must be a non-empty string literal")
        return node.value

    def _path_argument(self, node: ast.Call) -> str:
        if len(node.args) != 1:
            raise DebuggerConsoleError("read/ref/value/copy require one path")
        return self._literal_path(node.args[0])

    def _target_path(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            return self._literal_path(node)
        if isinstance(node, ast.Name) and node.id in self.references:
            return self.references[node.id]
        raise DebuggerConsoleError("edit target must be a path string or a direct read/ref binding")

    @staticmethod
    def _safe_name(name: str) -> None:
        if not name.isidentifier() or name.startswith("__"):
            raise DebuggerConsoleError("binding names must be ordinary identifiers")

    @staticmethod
    def help() -> str:
        return (
            "typed console: row = read('PATH'); candidate = row * 0.5 + read('OTHER'); "
            "edit(row, candidate); preview(); commit();"
        )


__all__ = ["DebuggerConsoleError", "TypedDebuggerConsole"]

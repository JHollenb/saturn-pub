"""Exact dirty closure and local memoization for declared acyclic dependencies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .values import clone, describe, digest


@dataclass(frozen=True)
class Cell:
    name: str
    reads: tuple[str, ...]
    implementation: Callable[[Mapping[str, Any]], Any]
    version: str


class DependencyGraph:
    """Topological cells; feedback is represented by a new trajectory step.

    The cache is instance-local and version-bound. The native consumer always runs.
    This is a transparent reference partial evaluator, not a universal neural compiler.
    """

    def __init__(self, cells: tuple[Cell, ...]):
        names = [cell.name for cell in cells]
        if len(set(names)) != len(names):
            raise ValueError("cell names must be unique")
        for index, cell in enumerate(cells):
            if not cell.version or any(x in names[index:] for x in cell.reads):
                raise ValueError("cells must be versioned and topologically ordered")
        self.cells = cells
        self._cache: dict[str, Any] = {}

    def dirty(self, changed: set[str]) -> set[str]:
        dirty = set(changed)
        for cell in self.cells:
            if dirty.intersection(cell.reads):
                dirty.add(cell.name)
        return dirty

    def run(
        self, inputs: Mapping[str, Any], consumer: Callable[[Mapping[str, Any]], Any]
    ) -> tuple[Any, dict[str, Any]]:
        if set(inputs).intersection(cell.name for cell in self.cells):
            raise ValueError("inputs cannot shadow computed cells")
        values = clone(dict(inputs))
        reused, executed = [], []
        for cell in self.cells:
            reads = {name: clone(values[name]) for name in cell.reads}
            key = digest({"name": cell.name, "version": cell.version, "reads": describe(reads)})
            if key in self._cache:
                reused.append(cell.name)
            else:
                self._cache[key] = clone(cell.implementation(reads))
                executed.append(cell.name)
            values[cell.name] = clone(self._cache[key])
        output = consumer(clone(values))
        return output, {"executed": executed, "reused": reused, "consumer_executed": True}

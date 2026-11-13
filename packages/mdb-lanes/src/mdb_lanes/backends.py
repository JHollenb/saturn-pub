"""Public, framework-independent execution seam and lazy backend factories.

Forward capture is separate from resumable state. A runner may bind an existing
DebuggerAdapter when it owns the complete mutable execution state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .errors import BenchError


@dataclass(frozen=True)
class RunnerCapabilities:
    backend: str
    capture_points: tuple[str, ...] = ()
    intervention_points: tuple[str, ...] = ()
    intervention_callbacks: bool = False
    intervention_specs: bool = False

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("backend must have a name")
        for name in ("capture_points", "intervention_points"):
            points = tuple(getattr(self, name))
            if any(not isinstance(point, str) or not point for point in points):
                raise ValueError("hook points must be nonempty strings")
            if len(set(points)) != len(points):
                raise ValueError("hook points must be unique")
            object.__setattr__(self, name, points)
        for name in ("intervention_callbacks", "intervention_specs"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("intervention capabilities must be booleans")

    def validate(self, capture, interventions):
        if isinstance(capture, str):
            raise BenchError("capture must be a sequence of hook names")
        capture = tuple(capture)
        if any(not isinstance(point, str) for point in capture):
            raise BenchError("capture points must be strings")
        if len(set(capture)) != len(capture):
            raise BenchError("capture points must be unique")
        unknown = set(capture) - set(self.capture_points)
        if unknown:
            raise BenchError(f"unsupported capture points: {sorted(unknown)}")
        if interventions is not None and not isinstance(interventions, Mapping):
            raise BenchError("interventions must map hook names to callbacks or specs")
        interventions = dict(interventions or {})
        for point, edit in interventions.items():
            if point not in self.intervention_points:
                raise BenchError(f"unsupported intervention point: {point}")
            if callable(edit):
                if not self.intervention_callbacks:
                    raise BenchError("backend does not support intervention callbacks")
            elif isinstance(edit, Mapping):
                if not self.intervention_specs:
                    raise BenchError("backend does not support declarative interventions")
                if edit.get("op") not in {"set", "add", "scale", "suppress"}:
                    raise BenchError("unsupported intervention operation")
                if edit["op"] != "suppress" and "value" not in edit:
                    raise BenchError("intervention requires a value")
                allowed = {"op"} if edit["op"] == "suppress" else {"op", "value"}
                if set(edit) - allowed:
                    raise BenchError(
                        "unknown intervention fields; scope edits explicitly in a callback"
                    )
            else:
                raise BenchError("intervention must be a callback or a mapping")
        return capture, interventions


@dataclass(frozen=True)
class ForwardResult:
    """Native tensors stay native; this is not a durable execution checkpoint."""

    output: Any = None
    captured: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelRunner(ABC):
    """Minimal executor, independent of model acquisition and fleet scheduling.

    Intervention callbacks take one native value and return a replacement. A runner
    must remove temporary hooks even on failure and never silently ignore an edit.
    Runners with mutable state expose their native debugger through
    ``debugger_adapter``; forward-only runners leave it as None.
    """

    debugger_adapter = None

    @abstractmethod
    def capabilities(self) -> RunnerCapabilities: ...

    @abstractmethod
    def forward(self, inputs, *, capture=(), interventions=None, **kwargs) -> ForwardResult: ...

    def close(self):
        """Release runner-owned resources; caller-owned models remain caller owned."""


_factories: dict[str, Callable[..., ModelRunner]] = {}
_registry_lock = RLock()


def register_backend(name: str, factory: Callable[..., ModelRunner]) -> None:
    """Register an explicit process-local factory; never replace one silently."""
    if not isinstance(name, str) or not name or not name.replace("-", "_").isidentifier():
        raise ValueError("backend name must be an identifier with optional hyphens")
    if not callable(factory):
        raise TypeError("backend factory must be callable")
    with _registry_lock:
        if name in _factories or name in {"transformerlens", "transformers"}:
            raise ValueError(f"backend already registered: {name}")
        _factories[name] = factory


def available_backends() -> tuple[str, ...]:
    with _registry_lock:
        return tuple(sorted({"transformerlens", "transformers", *_factories}))


def create_backend(name: str, **kwargs):
    with _registry_lock:
        factory = _factories.get(name)
    if factory is None:
        if name == "transformerlens":
            from .backend_torch import TransformerLensRunner

            factory = TransformerLensRunner
        elif name == "transformers":
            from .backend_torch import TransformersRunner

            factory = TransformersRunner
        else:
            raise BenchError(f"unknown backend: {name}; available: {available_backends()}")
    return factory(**kwargs)

"""Explicit local-call contracts for Saturn boundary experiments.

Saturn uses mrun for coarse admission and custody.  Once a worker has a
captured boundary, the branch grid belongs to the resident process.  This
module gives that distinction an executable seam: strict boundary plans accept
only callables explicitly marked as local, and Saturn-owned scheduler and
checkpoint entry points reject calls made while that plan is active.

This is an instrumented contract, not a Python sandbox.  Third-party code can
still perform effects that Saturn does not own; callers should therefore mark
only implementations whose local behavior they control.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_LOCAL_CALLABLE_ATTR = "__saturn_local_data_plane_callable__"
_LOCAL_CALLABLE_LABEL_ATTR = "__saturn_local_data_plane_label__"


class LocalDataPlaneViolation(RuntimeError):
    """Raised when a strict local boundary attempts a Saturn-owned side effect."""


@dataclass(frozen=True, slots=True)
class LocalDataPlaneContext:
    """Metadata attached to one strict local invocation."""

    operation_id: str


_CURRENT_CONTEXT: ContextVar[LocalDataPlaneContext | None] = ContextVar(
    "saturn_local_data_plane_context",
    default=None,
)


class LocalCallable:
    """Proxy that explicitly declares a callable safe for strict local plans.

    Functions and most module instances can be marked in place by
    :func:`as_local_callable`.  This proxy is the fallback for callable objects
    that reject attributes, while delegating non-call behavior to the target.
    """

    __slots__ = ("target", "label", _LOCAL_CALLABLE_ATTR, "__wrapped__")

    def __init__(self, target: Any, *, label: str = "") -> None:
        if not callable(target):
            raise TypeError("local callable target must be callable")
        self.target = target
        self.label = str(label or getattr(target, "__name__", type(target).__name__))
        setattr(self, _LOCAL_CALLABLE_ATTR, True)
        self.__wrapped__ = target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target, name)


def as_local_callable(target: Any, *, label: str = "") -> Any:
    """Mark a controlled implementation or consumer as local-only.

    Marking is intentionally explicit at the call site.  The original object
    is returned when it permits metadata, preserving module identity and
    signatures; otherwise a transparent proxy is returned.
    """

    if not callable(target):
        raise TypeError("local callable target must be callable")
    if is_local_callable(target):
        return target
    resolved_label = str(label or getattr(target, "__name__", type(target).__name__))
    try:
        setattr(target, _LOCAL_CALLABLE_ATTR, True)
        setattr(target, _LOCAL_CALLABLE_LABEL_ATTR, resolved_label)
    except (AttributeError, TypeError):
        return LocalCallable(target, label=resolved_label)
    return target


def is_local_callable(target: Any) -> bool:
    """Return whether a callable carries the explicit local-call marker."""

    return callable(target) and bool(getattr(target, _LOCAL_CALLABLE_ATTR, False))


def require_local_callable(target: Any, *, role: str) -> None:
    """Fail closed when a strict plan receives an unmarked callable."""

    if not is_local_callable(target):
        raise LocalDataPlaneViolation(
            f"strict local boundary requires an explicitly marked {role}; "
            "wrap it with saturn.as_local_callable(...)"
        )


def current_local_data_plane() -> LocalDataPlaneContext | None:
    """Return the active strict local scope, if one exists."""

    return _CURRENT_CONTEXT.get()


@contextmanager
def local_data_plane_scope(operation_id: str) -> Iterator[LocalDataPlaneContext]:
    """Guard Saturn-owned scheduler/checkpoint calls for one local invocation."""

    context = LocalDataPlaneContext(operation_id=str(operation_id or "local-boundary"))
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


def reject_external_operation(operation: str) -> None:
    """Reject a Saturn-owned external operation inside a strict local scope."""

    context = current_local_data_plane()
    if context is None:
        return
    raise LocalDataPlaneViolation(
        f"{operation} is forbidden inside strict local boundary "
        f"{context.operation_id!r}; use the captured artifact and current lease"
    )


__all__ = [
    "LocalCallable",
    "LocalDataPlaneContext",
    "LocalDataPlaneViolation",
    "as_local_callable",
    "current_local_data_plane",
    "is_local_callable",
    "local_data_plane_scope",
    "reject_external_operation",
    "require_local_callable",
]

"""Live, branch-aware diagnostic watchpoints over public Session state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .core import Receipt, Session, StateCut
from .values import clone, describe, digest, tensor

_PREDICATES = frozenset({"change", "gt", "lt", "nonfinite"})


def _describe(value: Any) -> Any:
    """Describe anomalous scalars without asking JSON to encode NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, Mapping):
        return {key: _describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_describe(item) for item in value]
    return describe(value)


@dataclass(frozen=True)
class Watchpoint:
    """A diagnostic predicate. It has no scientific or promotion authority."""

    identifier: str
    address: str
    predicate: str
    threshold: float | None = None

    def __post_init__(self) -> None:
        if not self.identifier or not self.address:
            raise ValueError("watch identifier and address must be non-empty")
        if self.predicate not in _PREDICATES:
            raise ValueError(f"unsupported watch predicate: {self.predicate}")
        if self.predicate in ("gt", "lt"):
            if self.threshold is None or not math.isfinite(float(self.threshold)):
                raise ValueError(f"{self.predicate} watch requires a finite threshold")
            object.__setattr__(self, "threshold", float(self.threshold))
        elif self.threshold is not None:
            raise ValueError(f"{self.predicate} watch does not accept a threshold")

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "address": self.address,
            "predicate": self.predicate,
            "threshold": self.threshold,
            "authority": "diagnostic-only",
        }


@dataclass(frozen=True)
class WatchEvent:
    """A watch fire bound to the exact retained safe-point cut and step receipt."""

    watchpoint: Watchpoint
    branch: str
    cut: StateCut = field(repr=False)
    execution_point: Mapping[str, Any]
    previous: Any
    current: Any
    receipt: str | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.branch:
            raise ValueError("watch event branch must be non-empty")
        self.cut.verify()
        point = clone(dict(self.execution_point))
        object.__setattr__(self, "execution_point", MappingProxyType(point))
        object.__setattr__(
            self,
            "fingerprint",
            digest(
                {
                    "schema": "saturn-pub-watch-event-v1",
                    "watchpoint": self.watchpoint.to_dict(),
                    "branch": self.branch,
                    "cut": self.cut.fingerprint,
                    "execution_point": point,
                    "previous": _describe(self.previous),
                    "current": _describe(self.current),
                    "receipt": self.receipt,
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-watch-event-v1",
            "fingerprint": self.fingerprint,
            "watchpoint": self.watchpoint.to_dict(),
            "branch": self.branch,
            "cut": self.cut.fingerprint,
            "boundary": self.cut.boundary,
            "execution_point": dict(self.execution_point),
            "previous": _describe(self.previous),
            "current": _describe(self.current),
            "receipt": self.receipt,
            "authority": "diagnostic-only",
        }


def _samples(value: Any) -> list[float]:
    if tensor(value):
        raw = value.detach().cpu().reshape(-1)
        return [float(item) for item in raw.tolist()]
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        result: list[float] = []
        for item in value.values():
            result.extend(_samples(item))
        return result
    if isinstance(value, (tuple, list)):
        result = []
        for item in value:
            result.extend(_samples(item))
        return result
    return []


def _fires(watch: Watchpoint, previous: Any, current: Any) -> bool:
    if watch.predicate == "change":
        return previous is not None and _describe(previous) != _describe(current)
    values = _samples(current)
    if not values:
        return False
    if watch.predicate == "nonfinite":
        return any(not math.isfinite(value) for value in values)
    if watch.predicate == "gt":
        return any(value > float(watch.threshold) for value in values)
    return any(value < float(watch.threshold) for value in values)


class Observer:
    """Own watch specifications and per-branch baselines outside Session."""

    def __init__(self) -> None:
        self._watches: dict[str, Watchpoint] = {}
        self._baselines: dict[tuple[str, str], Any] = {}
        self._next_identifier = 1

    @property
    def watchpoints(self) -> tuple[Watchpoint, ...]:
        return tuple(self._watches.values())

    def has_baseline(self, branch: str) -> bool:
        return all(
            (branch, watch.identifier) in self._baselines for watch in self._watches.values()
        )

    def add(
        self,
        address: str,
        predicate: str,
        threshold: float | None = None,
        *,
        identifier: str | None = None,
    ) -> Watchpoint:
        if identifier is None:
            while f"watch-{self._next_identifier}" in self._watches:
                self._next_identifier += 1
            identifier = f"watch-{self._next_identifier}"
            self._next_identifier += 1
        if identifier in self._watches:
            raise ValueError(f"watch already exists: {identifier}")
        watch = Watchpoint(identifier, address, predicate, threshold)
        self._watches[identifier] = watch
        return watch

    def remove(self, identifier: str) -> None:
        if identifier not in self._watches:
            raise ValueError(f"unknown watch: {identifier}")
        del self._watches[identifier]
        self._baselines = {
            key: value for key, value in self._baselines.items() if key[1] != identifier
        }

    def clear(self) -> None:
        self._watches.clear()
        self._baselines.clear()

    def sync(self, session: Session, branch: str) -> None:
        """Reset a branch after restore/rewind and seed every current watch."""
        self._baselines = {key: value for key, value in self._baselines.items() if key[0] != branch}
        for watch in self._watches.values():
            self._baselines[(branch, watch.identifier)] = session.read(watch.address)

    def evaluate(
        self,
        session: Session,
        branch: str,
        *,
        receipt: Receipt | None = None,
    ) -> tuple[WatchEvent, ...]:
        fired: list[tuple[Watchpoint, Any, Any]] = []
        for watch in self._watches.values():
            key = (branch, watch.identifier)
            current = session.read(watch.address)
            previous = self._baselines.get(key)
            if _fires(watch, previous, current):
                fired.append((watch, previous, current))
            self._baselines[key] = clone(current)
        if not fired:
            return ()
        # One retained cut is the shared safe point for every predicate fired by this transition.
        cut = session.capture()
        point = getattr(cut, "execution_point", None)
        if point is None:
            frame = session.inspect()
            point = getattr(frame, "execution_point", {"boundary": frame.boundary})
        receipt_id = None if receipt is None else receipt.fingerprint
        return tuple(
            WatchEvent(watch, branch, cut, point, previous, current, receipt_id)
            for watch, previous, current in fired
        )


__all__ = ["Observer", "WatchEvent", "Watchpoint"]

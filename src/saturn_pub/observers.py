"""Live, branch-aware diagnostic watchpoints over public Session state."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .core import Receipt, Session, StateCut
from .values import clone, describe, digest, tensor

_PREDICATES = frozenset({"change", "gt", "lt", "nonfinite"})
_UNSET = object()


def _describe(value: Any) -> Any:
    """Describe anomalous scalars without asking JSON to encode NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, Mapping):
        items = [{"key": describe(key), "value": _describe(item)} for key, item in value.items()]
        items.sort(key=lambda item: digest(item["key"]))
        return {"kind": "mapping", "items": items}
    if isinstance(value, (tuple, list)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_describe(item) for item in value],
        }
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
        object.__setattr__(self, "previous", clone(self.previous))
        object.__setattr__(self, "current", clone(self.current))
        if self.cut.execution_point is not None and point != dict(self.cut.execution_point):
            raise ValueError("watch execution point does not match cut")
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
        self.verify()
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

    def verify(self) -> None:
        self.cut.verify()
        body = {
            "schema": "saturn-pub-watch-event-v1",
            "watchpoint": self.watchpoint.to_dict(),
            "branch": self.branch,
            "cut": self.cut.fingerprint,
            "execution_point": dict(self.execution_point),
            "previous": _describe(self.previous),
            "current": _describe(self.current),
            "receipt": self.receipt,
        }
        if digest(body) != self.fingerprint:
            raise ValueError("watch event content changed")


class ObservationError(ValueError):
    """One broken watch does not erase completed fires from other watches."""

    def __init__(self, errors, events, cut):
        super().__init__("; ".join(f"{row['address']}: {row['reason']}" for row in errors))
        self.errors = tuple(errors)
        self.events = tuple(events)
        self.cut = cut


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
        return previous is not _UNSET and _describe(previous) != _describe(current)
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

    def __init__(self, reader: Callable[[Session, str], Any] | None = None) -> None:
        self._watches: dict[str, Watchpoint] = {}
        self._baselines: dict[tuple[str, str], Any] = {}
        self._next_identifier = 1
        self._read = reader or (lambda session, address: session.read(address))

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
        values = {
            (branch, watch.identifier): clone(self._read(session, watch.address))
            for watch in self._watches.values()
        }
        self._baselines = {key: value for key, value in self._baselines.items() if key[0] != branch}
        self._baselines.update(values)

    def evaluate(
        self,
        session: Session,
        branch: str,
        *,
        receipt: Receipt | None = None,
    ) -> tuple[WatchEvent, ...]:
        fired: list[tuple[Watchpoint, Any, Any]] = []
        current_values = {}
        errors = []
        for watch in self._watches.values():
            key = (branch, watch.identifier)
            try:
                current = self._read(session, watch.address)
                previous = self._baselines.get(key, _UNSET)
                fires = _fires(watch, previous, current)
            except Exception as exc:
                errors.append(
                    {
                        "identifier": watch.identifier,
                        "address": watch.address,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if fires:
                fired.append((watch, None if previous is _UNSET else previous, current))
            current_values[key] = clone(current)
        if not fired and not errors:
            self._baselines.update(current_values)
            return ()
        # One retained cut is the shared safe point for every predicate fired by this transition.
        cut = session.capture()
        point = getattr(cut, "execution_point", None)
        if point is None:
            frame = session.inspect()
            point = getattr(frame, "execution_point", {"boundary": frame.boundary})
        receipt_id = None if receipt is None else receipt.fingerprint
        if receipt is not None and receipt.to_dict().get("result") != cut.fingerprint:
            raise ValueError("watch receipt does not produce the observed cut")
        events = tuple(
            WatchEvent(watch, branch, cut, point, previous, current, receipt_id)
            for watch, previous, current in fired
        )
        self._baselines.update(current_values)
        if errors:
            raise ObservationError(errors, events, cut)
        return events


__all__ = ["Observer", "WatchEvent", "Watchpoint", "ObservationError"]

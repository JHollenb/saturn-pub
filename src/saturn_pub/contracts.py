"""Framework-neutral execution positions, ports, and continuation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .values import canonical, clone, digest


@dataclass(frozen=True)
class ExecutionPoint:
    family: str
    logical_step: int
    phase: str
    operator: str = ""
    index: int | None = None
    edge: str = "before"
    next_operation: str = ""
    local_clock: int | None = None

    def __post_init__(self):
        if not self.family or not self.phase or self.edge not in {"before", "after"}:
            raise ValueError("execution point requires family, phase and before/after edge")
        for value in (self.logical_step, self.index, self.local_clock):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("execution clocks and indices must be nonnegative integers")

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

    @property
    def fingerprint(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class SlotSpec:
    name: str
    role: str = "carrier"
    writable: bool = False
    persistence: str = "authoritative"
    producer: str = ""
    consumer: str = ""
    invalidates: tuple[str, ...] = ()
    schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or self.persistence not in {"authoritative", "derived", "evidence"}:
            raise ValueError("slot requires a name and declared persistence policy")
        if len(set(self.invalidates)) != len(self.invalidates):
            raise ValueError("duplicate slot invalidation")
        canonical(dict(self.schema))
        object.__setattr__(self, "schema", MappingProxyType(clone(dict(self.schema))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "writable": self.writable,
            "persistence": self.persistence,
            "producer": self.producer,
            "consumer": self.consumer,
            "invalidates": list(self.invalidates),
            "schema": dict(self.schema),
        }


@dataclass(frozen=True)
class SurfaceManifest:
    slots: tuple[SlotSpec, ...]
    state_schema: str
    consumer: str
    horizon: str = "native-suffix"

    def __post_init__(self):
        names = [slot.name for slot in self.slots]
        if not self.state_schema or not self.consumer or len(set(names)) != len(names):
            raise ValueError("surface requires schema, consumer, and unique slots")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-surface-v1",
            "state_schema": self.state_schema,
            "consumer": self.consumer,
            "horizon": self.horizon,
            "slots": [slot.to_dict() for slot in self.slots],
        }

    def slot(self, name: str) -> SlotSpec:
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise ValueError(f"undeclared state port: {name}")

    def validate(self, state: Mapping[str, Any]) -> None:
        if set(state) != {slot.name for slot in self.slots}:
            raise ValueError("state does not match its declared closure")


@dataclass(frozen=True)
class TransitionSpec:
    """Adapter-declared native footprint, never an executable second graph."""

    operation: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    consumer: str = "native-suffix"
    footprint: str = "unknown"

    def __post_init__(self):
        if not self.operation or self.footprint not in {"unknown", "declared"}:
            raise ValueError("transition requires an operation and footprint status")
        for name in ("reads", "writes", "invalidates"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for group in (self.reads, self.writes, self.invalidates):
            if len(set(group)) != len(group) or any(not name for name in group):
                raise ValueError("transition slots must be unique nonempty names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "consumer": self.consumer,
            "footprint": self.footprint,
        }


@dataclass(frozen=True)
class StateAddress:
    """A context-bound port and optional integer tensor/list coordinates."""

    slot: str
    selector: tuple[int, ...] = ()
    role: str = ""
    execution_point: str | None = None
    state_schema: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "selector", tuple(self.selector))
        if not self.slot or any(type(index) is not int for index in self.selector):
            raise ValueError("address requires a slot and integer selector")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "selector": list(self.selector),
            "role": self.role,
            "execution_point": self.execution_point,
            "state_schema": self.state_schema,
        }


@dataclass(frozen=True)
class NumericalContract:
    mode: str = "same-program-exact"
    program: str = "adapter-declared"
    comparator: str = "typed-content-digest"
    rtol: float = 0.0
    atol: float = 0.0

    def __post_init__(self):
        if self.mode not in {"same-program-exact", "bounded-numeric", "behavior-observed"}:
            raise ValueError("unknown numerical authority")
        if self.rtol < 0 or self.atol < 0:
            raise ValueError("numerical tolerances must be nonnegative")
        canonical(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))

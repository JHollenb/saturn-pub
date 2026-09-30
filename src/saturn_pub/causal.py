"""Typed causal-path evidence linked to exact cuts and receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .core import Receipt, Session, StateCut
from .values import clone, describe, digest

_KINDS = frozenset(
    {
        "source",
        "address",
        "carrier",
        "writer",
        "consumer",
        "first-divergence",
        "repair",
        "collateral",
        "native-consumer",
    }
)
_EFFECT_KINDS = ("first-divergence", "repair", "collateral", "native-consumer")


@dataclass(frozen=True)
class PortObservation:
    """One raw observation at a declared causal port or effect plane."""

    kind: str
    port: str
    value: Any
    model_identity: str
    parent: str
    clock: Mapping[str, Any]
    cut: str
    receipt: str
    note: str | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported causal observation kind: {self.kind}")
        for label, value in (
            ("port", self.port),
            ("model_identity", self.model_identity),
            ("parent", self.parent),
            ("cut", self.cut),
            ("receipt", self.receipt),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label} must be a non-empty string")
        clock = clone(dict(self.clock))
        if not clock:
            raise ValueError("clock must be a non-empty mapping")
        object.__setattr__(self, "clock", MappingProxyType(clock))
        object.__setattr__(self, "value", clone(self.value))
        object.__setattr__(self, "fingerprint", digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-port-observation-v1",
            "kind": self.kind,
            "port": self.port,
            "value": describe(self.value),
            "model_identity": self.model_identity,
            "parent": self.parent,
            "clock": dict(self.clock),
            "cut": self.cut,
            "receipt": self.receipt,
            "note": self.note,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def record(
        cls,
        kind: str,
        port: str,
        value: Any,
        session: Session,
        receipt: Receipt,
        *,
        parent: str,
        cut: StateCut | None = None,
    ) -> PortObservation:
        """Bind a live observation to the current exact cut and producing receipt."""
        cut = session.capture(retain=False) if cut is None else cut
        cut.verify()
        if cut.model_identity != session.adapter.model_identity or dict(cut.execution) != dict(
            session.adapter.execution
        ):
            raise ValueError("observation cut belongs to another model or execution contract")
        receipt_body = receipt.to_dict()
        if receipt_body.get("result") != cut.fingerprint:
            raise ValueError("observation cut is not the producing receipt result")
        point = getattr(cut, "execution_point", None)
        if point is None:
            point = getattr(session.inspect(), "execution_point", {"boundary": cut.boundary})
        return cls(
            kind,
            port,
            value,
            cut.model_identity,
            parent,
            point,
            cut.fingerprint,
            receipt.fingerprint,
        )


@dataclass(frozen=True)
class CausalPath:
    """An evidence graph over one model/root parent and declared clock support."""

    source: str
    address: str
    carrier: str
    writer: str
    consumer: str
    model_identity: str
    parent: str
    clocks: tuple[Mapping[str, Any], ...]
    observations: tuple[PortObservation, ...] = ()
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "source",
            "address",
            "carrier",
            "writer",
            "consumer",
            "model_identity",
            "parent",
        ):
            if not getattr(self, label):
                raise ValueError(f"{label} must be non-empty")
        clocks = tuple(MappingProxyType(clone(dict(clock))) for clock in self.clocks)
        if not clocks or any(not clock for clock in clocks):
            raise ValueError("causal path requires at least one non-empty supported clock")
        clock_ids = [digest(dict(clock)) for clock in clocks]
        if len(set(clock_ids)) != len(clock_ids):
            raise ValueError("supported clocks must be distinct")
        object.__setattr__(self, "clocks", clocks)
        for row in self.observations:
            if row.model_identity != self.model_identity:
                raise ValueError("observation model is outside causal-path support")
            if row.parent != self.parent:
                raise ValueError("observation parent is outside causal-path support")
            if digest(dict(row.clock)) not in clock_ids:
                raise ValueError("observation clock is outside causal-path support")
            if row.kind in {"source", "address", "carrier", "writer", "consumer"}:
                expected = getattr(self, row.kind)
                if row.port != expected:
                    raise ValueError(f"{row.kind} observation is bound to an undeclared port")
        divergence = [row for row in self.observations if row.kind == "first-divergence"]
        if len(divergence) > 1:
            raise ValueError("causal path can declare only one first divergence")
        object.__setattr__(self, "fingerprint", digest(self._body()))

    @property
    def first_divergence(self) -> PortObservation | None:
        return next((row for row in self.observations if row.kind == "first-divergence"), None)

    @property
    def repair(self) -> tuple[PortObservation, ...]:
        return tuple(row for row in self.observations if row.kind == "repair")

    @property
    def collateral(self) -> tuple[PortObservation, ...]:
        return tuple(row for row in self.observations if row.kind == "collateral")

    @property
    def native_consumer(self) -> tuple[PortObservation, ...]:
        return tuple(row for row in self.observations if row.kind == "native-consumer")

    def with_observations(self, *rows: PortObservation) -> CausalPath:
        return CausalPath(
            self.source,
            self.address,
            self.carrier,
            self.writer,
            self.consumer,
            self.model_identity,
            self.parent,
            self.clocks,
            self.observations + tuple(rows),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-causal-path-v1",
            "model_identity": self.model_identity,
            "parent": self.parent,
            "supported_clocks": [dict(clock) for clock in self.clocks],
            "nodes": {
                "source": self.source,
                "address": self.address,
                "writer": self.writer,
                "carrier": self.carrier,
                "consumer": self.consumer,
            },
            "edges": [
                {"from": self.source, "to": self.address, "relation": "routes-to"},
                {"from": self.address, "to": self.writer, "relation": "selects-writer"},
                {"from": self.writer, "to": self.carrier, "relation": "writes"},
                {"from": self.carrier, "to": self.consumer, "relation": "consumed-by"},
            ],
            "observations": [row.to_dict() for row in self.observations],
            "effects": {
                kind: [row.fingerprint for row in self.observations if row.kind == kind]
                for kind in _EFFECT_KINDS
            },
            "interpretation": "observations-only; no automatic scientific verdict",
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_observations(
        cls,
        *,
        source: str,
        address: str,
        carrier: str,
        writer: str,
        consumer: str,
        observations: Sequence[PortObservation],
    ) -> CausalPath:
        rows = tuple(observations)
        if not rows:
            raise ValueError("cannot infer causal-path support without observations")
        clocks: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            identifier = digest(dict(row.clock))
            if identifier not in seen:
                clocks.append(row.clock)
                seen.add(identifier)
        return cls(
            source,
            address,
            carrier,
            writer,
            consumer,
            rows[0].model_identity,
            rows[0].parent,
            tuple(clocks),
            rows,
        )


__all__ = ["CausalPath", "PortObservation"]

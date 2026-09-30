"""Bounded temporal assays and point-aligned, same-parent trajectory comparisons.

This is an execution/observation helper, not measured Program certification.
Every mutation and transition still uses Session. Full cuts are deliberately
opt-in through this tracing workflow rather than retained by ordinary execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .core import Act, Receipt, Session, StateCut
from .values import clone, describe, digest


class PathExecutionError(RuntimeError):
    """A failed arm retains completed cuts/receipts and the last valid state."""

    def __init__(self, error: Exception, *, points, events, cut, attempts):
        super().__init__(f"{type(error).__name__}: {error}")
        self.points = tuple(points)
        self.events = tuple(events)
        self.cut = cut
        self.attempts = tuple(attempts)


@dataclass(frozen=True)
class TimedAct:
    after_steps: int
    act: Act

    def __post_init__(self):
        if type(self.after_steps) is not int or self.after_steps < 0:
            raise ValueError("schedule offset must be a nonnegative integer")


@dataclass(frozen=True)
class PathSchedule:
    instructions: tuple[TimedAct, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "instructions", tuple(self.instructions))
        offsets = [item.after_steps for item in self.instructions]
        if offsets != sorted(offsets):
            raise ValueError("schedule must be ordered; same-offset Acts keep declared order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": [
                {"after_steps": item.after_steps, "act": item.act.manifest()}
                for item in self.instructions
            ],
            "authority": "exploratory-execution-not-measured-program",
        }


@dataclass(frozen=True)
class TracePoint:
    offset: int
    cut: StateCut
    values: Mapping[str, Any]
    metrics: Mapping[str, Any]
    receipts: tuple[Receipt, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "values", MappingProxyType(clone(dict(self.values))))
        object.__setattr__(self, "metrics", MappingProxyType(clone(dict(self.metrics))))
        object.__setattr__(self, "fingerprint", digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "cut": self.cut.fingerprint,
            "execution_point": dict(self.cut.execution_point or {}),
            "values": describe(dict(self.values)),
            "metrics": clone(dict(self.metrics)),
            "receipts": [row.to_dict() for row in self.receipts],
        }

    def verify(self) -> None:
        self.cut.verify()
        if digest(self._body()) != self.fingerprint:
            raise ValueError("trace observations changed")
        if any(
            describe(value) != describe(self.cut.payload[port])
            for port, value in self.values.items()
        ):
            raise ValueError("trace observations are not cut-backed slots")

    def to_dict(self) -> dict[str, Any]:
        self.verify()
        return {**self._body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class Trajectory:
    name: str
    parent: StateCut
    schedule: PathSchedule
    points: tuple[TracePoint, ...]
    events: tuple[tuple[Receipt, StateCut], ...]
    entry: StateCut

    @classmethod
    def run(
        cls,
        session: Session,
        *,
        name: str,
        parent: StateCut,
        steps: int,
        ports: tuple[str, ...],
        schedule: PathSchedule = PathSchedule(),
        evaluator: Callable[[Session], Mapping[str, Any]] | None = None,
    ) -> Trajectory:
        if not name or type(steps) is not int or steps < 1 or not ports:
            raise ValueError("trace requires name, positive budget and observed ports")
        if len(set(ports)) != len(ports):
            raise ValueError("trace ports must be unique")
        if any(item.after_steps > steps for item in schedule.instructions):
            raise ValueError("schedule extends past the declared horizon")
        parent.verify()
        branch = session.fork(parent)
        entry = branch.capture(retain=False)
        points = []
        events = []
        for offset in range(steps + 1):
            receipts = []
            try:
                if offset:
                    receipt = branch.step()
                    receipts.append(receipt)
                    events.append((receipt, branch.capture(retain=False)))
                for item in schedule.instructions:
                    if item.after_steps == offset:
                        receipt = branch.apply(item.act)
                        receipts.append(receipt)
                        events.append((receipt, branch.capture(retain=False)))
                cut = branch.capture(retain=False)
                values = {port: branch.read(port) for port in ports}
                metrics = {} if evaluator is None else dict(evaluator(branch.fork(cut)))
                # Refuse non-JSON/unstable metric artifacts before returning a trace.
                digest(metrics)
            except Exception as exc:
                raise PathExecutionError(
                    exc,
                    points=points,
                    events=events,
                    cut=branch.capture(retain=False),
                    attempts=branch.attempts,
                ) from exc
            points.append(TracePoint(offset, cut, clone(values), clone(metrics), tuple(receipts)))
        return cls(name, parent, schedule, tuple(points), tuple(events), entry)

    def verify(self) -> None:
        self.parent.verify()
        self.entry.verify()
        if len(self.points) < 2 or [p.offset for p in self.points] != list(range(len(self.points))):
            raise ValueError(
                "trace offsets must be consecutive from zero; not aligned to transitions"
            )
        if (
            self.entry.parent != self.parent.fingerprint
            or describe(self.entry.payload) != describe(self.parent.payload)
            or self.entry.model_identity != self.parent.model_identity
            or dict(self.entry.execution) != dict(self.parent.execution)
        ):
            raise ValueError("trace entry is not a fork of its declared parent")
        current = self.entry
        for receipt, cut in self.events:
            cut.verify()
            body = receipt.to_dict()
            if (
                body.get("parent") != current.fingerprint
                or body.get("result") != cut.fingerprint
                or cut.parent != current.fingerprint
                or cut.model_identity != self.parent.model_identity
                or dict(cut.execution) != dict(self.parent.execution)
                or body.get("model_identity") != cut.model_identity
                or body.get("execution") != dict(cut.execution)
                or body.get("execution_point") != dict(cut.execution_point or {})
            ):
                raise ValueError("trace event ancestry or execution mismatch")
            current = cut
        if [r.fingerprint for p in self.points for r in p.receipts] != [
            r.fingerprint for r, _ in self.events
        ]:
            raise ValueError("trace point receipts do not match retained events")
        event_cuts = {receipt.fingerprint: cut.fingerprint for receipt, cut in self.events}
        for point in self.points:
            point.verify()
            native_rows = [
                row.to_dict() for row in point.receipts if row.to_dict()["operation"] != "apply"
            ]
            if (
                len(native_rows) != (1 if point.offset else 0)
                or any(
                    row["operation"] != "continue" or row.get("steps") != 1 for row in native_rows
                )
                or (native_rows and point.receipts[0].to_dict()["operation"] != "continue")
            ):
                raise ValueError(
                    "trace offset does not count one native transition before its Acts"
                )
            expected = (
                event_cuts[point.receipts[-1].fingerprint]
                if point.receipts
                else self.entry.fingerprint
            )
            if point.cut.fingerprint != expected:
                raise ValueError("trace point cut is not its producing event")
        measured = [
            (point.offset, row.to_dict()["act"])
            for point in self.points
            for row in point.receipts
            if row.to_dict()["operation"] == "apply"
        ]
        declared = [(item.after_steps, item.act.manifest()) for item in self.schedule.instructions]
        if measured != declared:
            raise ValueError("trace schedule does not match recorded Acts")

    def to_dict(self) -> dict[str, Any]:
        self.verify()
        return {
            "schema": "saturn-pub-trajectory-v1",
            "name": self.name,
            "parent": self.parent.fingerprint,
            "entry": self.entry.fingerprint,
            "schedule": self.schedule.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "events": [
                {"receipt": receipt.to_dict(), "cut": cut.fingerprint}
                for receipt, cut in self.events
            ],
            "authority": "observations-only",
        }


def compare_trajectories(native: Trajectory, candidate: Trajectory) -> dict[str, Any]:
    """Exact selected-port differences and recovery windows at recorded points.

    A recovery means equality of that selected observable, not semantic repair.
    Evaluator measurements remain separate. Clock misalignment is an error;
    restoring/changing clocks cannot masquerade as an aligned comparison.
    """
    if native.parent.fingerprint != candidate.parent.fingerprint:
        raise ValueError("trajectory comparison requires one exact common parent")
    native.verify()
    candidate.verify()
    if len(native.points) != len(candidate.points):
        raise ValueError("trajectory horizons differ")
    rows = []
    windows: dict[str, list[dict[str, Any]]] = {}
    active: dict[str, int] = {}
    first = None
    for left, right in zip(native.points, candidate.points):
        left.cut.verify()
        right.cut.verify()
        for point in (left, right):
            if point.cut.model_identity != native.parent.model_identity or dict(
                point.cut.execution
            ) != dict(native.parent.execution):
                raise ValueError("trace cut model or execution differs from parent")
        if left.offset != right.offset or dict(left.cut.execution_point or {}) != dict(
            right.cut.execution_point or {}
        ):
            raise ValueError("trajectory execution points are not aligned")
        if set(left.values) != set(right.values):
            raise ValueError("trajectory observed ports differ")
        # Retained observations must still describe the sealed cut's port values.
        for point in (left, right):
            payload = point.cut.payload
            if any(
                describe(value) != describe(payload[port]) for port, value in point.values.items()
            ):
                raise ValueError("trace observations changed or are not cut-backed slots")
        different = []
        recovered = []
        for port in left.values:
            equal = describe(left.values[port]) == describe(right.values[port])
            if not equal:
                different.append(port)
                active.setdefault(port, left.offset)
            elif port in active:
                recovered.append(port)
                windows.setdefault(port, []).append(
                    {"diverged_at": active.pop(port), "recovered_at": left.offset}
                )
        if different and first is None:
            first = {
                "offset": left.offset,
                "ports": different,
                "execution_point": dict(right.cut.execution_point or {}),
            }
        rows.append(
            {
                "offset": left.offset,
                "native_cut": left.cut.fingerprint,
                "candidate_cut": right.cut.fingerprint,
                "different_ports": different,
                "recovered_ports": recovered,
                "native_metrics": clone(dict(left.metrics)),
                "candidate_metrics": clone(dict(right.metrics)),
                "receipts": [receipt.fingerprint for receipt in right.receipts],
            }
        )
    for port, offset in active.items():
        windows.setdefault(port, []).append({"diverged_at": offset, "recovered_at": None})
    return {
        "schema": "saturn-pub-trajectory-diff-v1",
        "parent": native.parent.fingerprint,
        "native": native.name,
        "candidate": candidate.name,
        "first_recorded_divergence": first,
        "observable_windows": windows,
        "rows": rows,
        "comparator": "exact-typed-selected-port-content",
        "interpretation": "observable recovery is not automatic semantic repair",
        "terminal_status": "not-assessed",
    }

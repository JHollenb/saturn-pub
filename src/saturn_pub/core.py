"""One transaction lifecycle for native model execution and counterfactual research."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .values import clone, describe, digest


@dataclass(frozen=True)
class Frame:
    """Immutable metadata descriptor; tensor readback is always a private copy."""

    boundary: str
    slots: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", MappingProxyType(clone(dict(self.slots))))


@dataclass(frozen=True)
class StateCut:
    """A sealed snapshot of an adapter's declared mutable execution closure."""

    model_identity: str
    execution: Mapping[str, Any]
    boundary: str
    parent: str | None
    _payload: Mapping[str, Any] = field(repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution", MappingProxyType(clone(dict(self.execution))))
        object.__setattr__(self, "_payload", clone(dict(self._payload)))
        object.__setattr__(self, "fingerprint", digest(self.manifest()))

    @property
    def payload(self) -> dict[str, Any]:
        return clone(self._payload)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-statecut-v1",
            "model_identity": self.model_identity,
            "execution": dict(self.execution),
            "boundary": self.boundary,
            "parent": self.parent,
            "payload": describe(self._payload),
        }

    def verify(self) -> None:
        if digest(self.manifest()) != self.fingerprint:
            raise ValueError("StateCut payload or metadata changed")


@dataclass(frozen=True)
class Act:
    """A registered operation gets exactly its declared reads and writes."""

    name: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    implementation: Callable[[Mapping[str, Any]], Mapping[str, Any]] = field(repr=False)
    invalidates: tuple[str, ...] = ()
    numerical_contract: str = "behavioral-intervention"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.writes or not callable(self.implementation):
            raise ValueError("Act requires a name, writes, and callable implementation")
        for group in (self.reads, self.writes, self.invalidates):
            if len(set(group)) != len(group) or any(not isinstance(x, str) or not x for x in group):
                raise ValueError("Act addresses must be unique non-empty strings")
        object.__setattr__(self, "parameters", MappingProxyType(clone(dict(self.parameters))))

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "numerical_contract": self.numerical_contract,
            "parameters": describe(dict(self.parameters)),
        }

    @classmethod
    def replace(cls, address: str, value: Any, *, name: str = "replace") -> Act:
        sealed = clone(value)
        return cls(
            name,
            (address,),
            (address,),
            lambda _: {address: clone(sealed)},
            parameters={"value": describe(sealed)},
        )

    @classmethod
    def add(cls, address: str, value: Any, *, dose: float = 1, name: str = "add") -> Act:
        sealed = clone(value)
        return cls(
            name,
            (address,),
            (address,),
            lambda x: {address: x[address] + dose * sealed},
            parameters={"value": describe(sealed), "dose": dose},
        )

    @classmethod
    def zero(cls, address: str) -> Act:
        return cls("zero", (address,), (address,), lambda x: {address: x[address] * 0})


@dataclass(frozen=True)
class Receipt:
    """Content-sealed metadata; outputs are summaries, never tensor payloads."""

    _body: bytes

    @classmethod
    def make(cls, **body: Any) -> Receipt:
        from .values import canonical

        return cls(canonical({"schema": "saturn-pub-receipt-v1", **body}))

    @property
    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(self._body).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        import json

        return {**json.loads(self._body), "fingerprint": self.fingerprint}


class Adapter(ABC):
    """Native executor contract. Execution state belongs to a Session, not the model."""

    model_identity: str
    execution: Mapping[str, Any]

    @abstractmethod
    def boundary(self, state: Mapping[str, Any]) -> str: ...

    @abstractmethod
    def validate(self, state: Mapping[str, Any]) -> None: ...

    @abstractmethod
    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]: ...

    @abstractmethod
    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]: ...

    def read(self, state: Mapping[str, Any], address: str) -> Any:
        if address not in self.addresses(state):
            raise ValueError(f"unsupported address at {self.boundary(state)}: {address}")
        return clone(state[address])

    def write(
        self, state: Mapping[str, Any], writes: Mapping[str, Any], invalidates: tuple[str, ...]
    ) -> Mapping[str, Any]:
        result = clone(state)
        for address, value in writes.items():
            if address not in self.addresses(state):
                raise ValueError(f"unsupported write address: {address}")
            old = state[address]
            if hasattr(old, "shape") and (old.shape != value.shape or old.dtype != value.dtype):
                raise ValueError(f"write shape/dtype mismatch: {address}")
            result[address] = clone(value)
        for address in invalidates:
            if address in writes:
                raise ValueError("cannot invalidate a written address")
            result.pop(address, None)
        self.validate(result)
        return result


class Session:
    """Resident native executor with isolated branches and explicit commit."""

    def __init__(self, adapter: Adapter, state: Mapping[str, Any], *, parent: str | None = None):
        adapter.validate(state)
        self.adapter = adapter
        self._state = clone(state)
        self._parent = parent
        self._fork_parent = parent
        self._aborted = False
        self.receipts: list[Receipt] = []
        self.history: list[StateCut] = []

    def inspect(self) -> Frame:
        return Frame(
            self.adapter.boundary(self._state),
            {
                a: describe(self.adapter.read(self._state, a))
                for a in self.adapter.addresses(self._state)
            },
        )

    def read(self, address: str) -> Any:
        return self.adapter.read(self._state, address)

    def capture(self, *, retain: bool = True) -> StateCut:
        cut = StateCut(
            self.adapter.model_identity,
            self.adapter.execution,
            self.adapter.boundary(self._state),
            self._parent,
            self._state,
        )
        if retain:
            self.history.append(cut)
        return cut

    def _compatible(self, cut: StateCut) -> None:
        cut.verify()
        if cut.model_identity != self.adapter.model_identity or dict(cut.execution) != dict(
            self.adapter.execution
        ):
            raise ValueError("incompatible model identity or execution contract")
        self.adapter.validate(cut.payload)
        if cut.boundary != self.adapter.boundary(cut.payload):
            raise ValueError("boundary does not match payload")

    def fork(self, cut: StateCut | None = None) -> Session:
        cut = cut or self.capture(retain=False)
        self._compatible(cut)
        return Session(self.adapter, cut.payload, parent=cut.fingerprint)

    def _record(self, operation: str, before: StateCut, **details: Any) -> Receipt:
        self._parent = before.fingerprint
        after = self.capture(retain=False)
        receipt = Receipt.make(
            operation=operation,
            parent=before.fingerprint,
            model_identity=self.adapter.model_identity,
            execution=dict(self.adapter.execution),
            result=after.fingerprint,
            boundary=after.boundary,
            **details,
        )
        self.receipts.append(receipt)
        return receipt

    def apply(self, act: Act) -> Receipt:
        before = self.capture(retain=False)
        reads = MappingProxyType({a: self.read(a) for a in act.reads})
        writes = dict(act.implementation(reads))
        if set(writes) != set(act.writes):
            raise ValueError("implementation must return exactly the declared writes")
        state = self.adapter.write(self._state, writes, act.invalidates)
        self._state = clone(state)
        return self._record("apply", before, act=act.manifest(), writes=describe(writes))

    def continue_(self, steps: int = 1) -> Receipt:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        before = self.capture(retain=False)
        # Publish only after all requested transitions succeed.
        candidate = clone(self._state)
        for _ in range(steps):
            candidate = dict(self.adapter.advance(candidate))
            self.adapter.validate(candidate)
        self._state = clone(candidate)
        return self._record("continue", before, steps=steps)

    def restore(self, cut: StateCut) -> Receipt:
        self._compatible(cut)
        before = self.capture(retain=False)
        self._state = cut.payload
        self._parent = cut.parent
        restored = self.capture(retain=False)
        if restored.fingerprint != cut.fingerprint:
            raise ValueError("restore fingerprint mismatch")
        receipt = Receipt.make(
            operation="restore",
            parent=before.fingerprint,
            result=cut.fingerprint,
            verified_exact=True,
            model_identity=self.adapter.model_identity,
            execution=dict(self.adapter.execution),
        )
        self.receipts.append(receipt)
        return receipt

    def commit(self, branch: Session) -> Receipt:
        parent = self.capture(retain=False)
        if (
            branch.adapter is not self.adapter
            or branch._origin != parent.fingerprint
            or branch._aborted
        ):
            raise ValueError("branch was not forked from the current parent")
        candidate = branch.capture(retain=False)
        self._compatible(candidate)
        self._state = candidate.payload
        return self._record("commit", parent, candidate=candidate.fingerprint)

    def abort(self, branch: Session, *, reason: str = "researcher discarded candidate") -> Receipt:
        """Keep candidate evidence inspectable, but remove its permission to commit."""
        if branch.adapter is not self.adapter:
            raise ValueError("candidate belongs to another resident adapter")
        parent = self.capture(retain=False)
        candidate = branch.capture(retain=False)
        branch._aborted = True
        receipt = Receipt.make(
            operation="abort",
            parent=parent.fingerprint,
            candidate=candidate.fingerprint,
            result=parent.fingerprint,
            reason=reason,
            parent_unchanged=True,
            model_identity=self.adapter.model_identity,
            execution=dict(self.adapter.execution),
        )
        self.receipts.append(receipt)
        return receipt

    @property
    def _origin(self) -> str | None:
        return self._fork_parent

    def compare(
        self, other: Session, evaluator: Callable[[Session], Mapping[str, Any]] | None = None
    ) -> dict[str, Any]:
        if other.adapter is not self.adapter:
            raise ValueError("comparison requires the same resident adapter")
        left, right = self.capture(retain=False), other.capture(retain=False)
        result = {
            "left": left.fingerprint,
            "right": right.fingerprint,
            "equal_payload": describe(left.payload) == describe(right.payload),
            "left_state": describe(left.payload),
            "right_state": describe(right.payload),
        }
        if evaluator:
            result["metrics"] = {
                "left": dict(evaluator(self.fork(left))),
                "right": dict(evaluator(other.fork(right))),
            }
        return result

"""One transaction lifecycle for native model execution and counterfactual research."""

from __future__ import annotations

import hashlib
import inspect
import marshal
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import CodeType, MappingProxyType
from typing import Any

from .contracts import (
    ExecutionPoint,
    NumericalContract,
    SlotSpec,
    StateAddress,
    SurfaceManifest,
    TransitionSpec,
)
from .values import clone, describe, digest, tensor

_OPERATION_REGISTRY: dict[tuple[str, str], tuple[str, Callable[..., Any], bool]] = {}


def _portable_code(code: CodeType) -> CodeType:
    """Remove source-location metadata while preserving executable semantics."""
    constants = tuple(
        _portable_code(value) if isinstance(value, CodeType) else value for value in code.co_consts
    )
    return code.replace(
        co_consts=constants,
        co_filename="<saturn-operation>",
        co_firstlineno=1,
        co_linetable=b"",
    )


def _implementation_value(value: Any, *, strict: bool) -> Any:
    try:
        return describe(value)
    except (TypeError, ValueError):
        if strict:
            raise ValueError("registered operation closure/defaults must be content-sealable")
        rendered = repr(value).encode()
        return {
            "kind": "opaque-exploratory",
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr_sha256": hashlib.sha256(rendered).hexdigest(),
        }


def _implementation_digest(implementation: Callable[..., Any], *, strict: bool = False) -> str:
    target = implementation
    closure = None
    defaults = None
    kwdefaults = None
    if inspect.ismethod(target):
        target = target.__func__
    if inspect.isfunction(target):
        code = target.__code__
        closure = [cell.cell_contents for cell in (target.__closure__ or ())]
        defaults = target.__defaults__
        kwdefaults = target.__kwdefaults__
        owner = {"module": target.__module__, "qualname": target.__qualname__}
    else:
        call = type(target).__call__
        code = call.__code__
        owner = {"module": type(target).__module__, "qualname": type(target).__qualname__}
    body = {
        **owner,
        "code_sha256": hashlib.sha256(marshal.dumps(_portable_code(code))).hexdigest(),
        "defaults": _implementation_value(defaults, strict=strict),
        "kwdefaults": _implementation_value(kwdefaults, strict=strict),
        "closure": _implementation_value(closure, strict=strict),
    }
    return digest(body)


def _implementation_state_digest(
    implementation: Callable[..., Any], *, strict: bool = False
) -> str:
    if inspect.ismethod(implementation):
        implementation = implementation.__func__
    if inspect.isfunction(implementation):
        state = {
            "kind": "function-state",
            "defaults": _implementation_value(implementation.__defaults__, strict=strict),
            "kwdefaults": _implementation_value(implementation.__kwdefaults__, strict=strict),
            "closure": _implementation_value(
                [cell.cell_contents for cell in (implementation.__closure__ or ())],
                strict=strict,
            ),
        }
    else:
        slots: dict[str, Any] = {}
        for owner in type(implementation).__mro__:
            declared = owner.__dict__.get("__slots__", ())
            if isinstance(declared, str):
                declared = (declared,)
            for name in declared:
                if name not in {"__dict__", "__weakref__"} and hasattr(implementation, name):
                    slots[f"{owner.__module__}.{owner.__qualname__}.{name}"] = getattr(
                        implementation, name
                    )
        state = {
            "kind": "callable-instance-state",
            "type": f"{type(implementation).__module__}.{type(implementation).__qualname__}",
            "dict": _implementation_value(
                dict(getattr(implementation, "__dict__", {})), strict=strict
            ),
            "slots": _implementation_value(slots, strict=strict),
        }
    return digest(state)


def _register_operation(
    operation_id: str,
    version: str,
    implementation: Callable[..., Any],
    *,
    pure: bool,
) -> str:
    if not operation_id or not version or not callable(implementation):
        raise ValueError("operation registration requires id, version, and callable")
    if not pure:
        raise ValueError("replayable operation registration requires an explicit pure=True")
    implementation_digest = _implementation_digest(implementation, strict=True)
    key = (operation_id, version)
    existing = _OPERATION_REGISTRY.get(key)
    if existing is not None and existing[0] != implementation_digest:
        raise ValueError("operation id/version is already bound to a different implementation")
    _OPERATION_REGISTRY[key] = (implementation_digest, implementation, True)
    return implementation_digest


class _ReplaceOperation:
    def __init__(self, address: str, value: Any):
        self.address = address
        self.value = clone(value)

    def __call__(self, _: Mapping[str, Any]) -> Mapping[str, Any]:
        return {self.address: clone(self.value)}


class _AddOperation:
    def __init__(self, address: str, value: Any, dose: float):
        self.address = address
        self.value = clone(value)
        self.dose = dose

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return {self.address: inputs[self.address] + self.dose * self.value}


class _ZeroOperation:
    def __init__(self, address: str):
        self.address = address

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        return {self.address: inputs[self.address] * 0}


def _all_bools(value: Any) -> bool:
    return type(value) is bool or (
        isinstance(value, list) and bool(value) and all(_all_bools(item) for item in value)
    )


class _PatchOperation:
    def __init__(self, address: str, value: Any, mask: Any, mode: str, dose: float):
        self.address = address
        self.value = clone(value)
        self.mask = clone(mask)
        self.mode = mode
        self.dose = dose

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        current = inputs[self.address]
        if not tensor(current):
            raise ValueError("patch currently requires a tensor-valued address")
        torch = __import__("torch")
        if tensor(self.mask):
            if self.mask.dtype != torch.bool:
                raise ValueError("patch tensor mask must have bool dtype")
            mask = self.mask.to(device=current.device)
        else:
            if not _all_bools(self.mask):
                raise ValueError("patch JSON mask must contain only booleans")
            mask = torch.tensor(self.mask, dtype=torch.bool, device=current.device)
        if tuple(mask.shape) != tuple(current.shape):
            raise ValueError("patch mask shape must exactly match the current tensor")
        value = self.value
        if tensor(value):
            if tuple(value.shape) != tuple(current.shape) or value.dtype != current.dtype:
                raise ValueError("patch value tensor shape/dtype must match the current tensor")
            value = value.to(device=current.device)
            selected = value[mask]
        elif type(value) not in (int, float, bool):
            raise ValueError("patch value must be a scalar or matching tensor")
        else:
            selected = value
        result = current.detach().clone()
        if self.mode == "replace":
            result[mask] = selected
        else:
            result[mask] = result[mask] + self.dose * selected
        return {self.address: result}


@dataclass(frozen=True)
class Frame:
    """Immutable metadata descriptor; tensor readback is always a private copy."""

    boundary: str
    slots: Mapping[str, Any]
    execution_point: Mapping[str, Any] = field(default_factory=dict)
    surface: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", MappingProxyType(clone(dict(self.slots))))
        object.__setattr__(
            self, "execution_point", MappingProxyType(clone(dict(self.execution_point)))
        )
        object.__setattr__(self, "surface", MappingProxyType(clone(dict(self.surface))))


@dataclass(frozen=True)
class StateCut:
    """A sealed snapshot of an adapter's declared mutable execution closure."""

    model_identity: str
    execution: Mapping[str, Any]
    boundary: str
    parent: str | None
    _payload: Mapping[str, Any] = field(repr=False)
    execution_point: Mapping[str, Any] | None = None
    surface: Mapping[str, Any] | None = None
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "execution", MappingProxyType(clone(dict(self.execution))))
        object.__setattr__(self, "_payload", clone(dict(self._payload)))
        if self.execution_point is not None:
            object.__setattr__(
                self,
                "execution_point",
                MappingProxyType(clone(dict(self.execution_point))),
            )
        if self.surface is not None:
            object.__setattr__(self, "surface", MappingProxyType(clone(dict(self.surface))))
        object.__setattr__(self, "fingerprint", digest(self.manifest()))

    @property
    def payload(self) -> dict[str, Any]:
        return clone(self._payload)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "saturn-pub-statecut-v2",
            "model_identity": self.model_identity,
            "execution": dict(self.execution),
            "boundary": self.boundary,
            "parent": self.parent,
            "payload": describe(self._payload),
            "execution_point": (
                None if self.execution_point is None else describe(dict(self.execution_point))
            ),
            "surface": None if self.surface is None else describe(dict(self.surface)),
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
    operation_id: str | None = None
    operation_version: str | None = None
    preconditions: Mapping[str, str] = field(default_factory=dict)
    preserve: tuple[str, ...] = ()
    implementation_digest: str = field(init=False)
    implementation_state_digest: str = field(init=False)
    operation_pure: bool = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.writes or not callable(self.implementation):
            raise ValueError("Act requires a name, writes, and callable implementation")
        for group in (self.reads, self.writes, self.invalidates):
            if len(set(group)) != len(group) or any(not isinstance(x, str) or not x for x in group):
                raise ValueError("Act addresses must be unique non-empty strings")
        object.__setattr__(self, "parameters", MappingProxyType(clone(dict(self.parameters))))
        if (self.operation_id is None) != (self.operation_version is None):
            raise ValueError("operation id and version must be supplied together")
        implementation_digest = _implementation_digest(
            self.implementation, strict=self.operation_id is not None
        )
        implementation_state_digest = _implementation_state_digest(
            self.implementation, strict=self.operation_id is not None
        )
        pure = False
        if self.operation_id is not None and self.operation_version is not None:
            registered = _OPERATION_REGISTRY.get((self.operation_id, self.operation_version))
            if registered is None:
                raise ValueError("operation id/version is not registered in this process")
            if registered[0] != implementation_digest:
                raise ValueError("registered operation implementation digest mismatch")
            pure = registered[2]
        for address, expected in self.preconditions.items():
            if not isinstance(address, str) or not address or not isinstance(expected, str):
                raise ValueError("Act preconditions require address-to-digest strings")
        if len(set(self.preserve)) != len(self.preserve) or any(
            not isinstance(address, str) or not address for address in self.preserve
        ):
            raise ValueError("Act preserve addresses must be unique non-empty strings")
        object.__setattr__(self, "preconditions", MappingProxyType(clone(dict(self.preconditions))))
        object.__setattr__(self, "implementation_digest", implementation_digest)
        object.__setattr__(self, "implementation_state_digest", implementation_state_digest)
        object.__setattr__(self, "operation_pure", pure)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "numerical_contract": self.numerical_contract,
            "parameters": describe(dict(self.parameters)),
            "operation": {
                "id": self.operation_id,
                "version": self.operation_version,
                "implementation_digest": self.implementation_digest,
                "implementation_state_digest": self.implementation_state_digest,
                "pure": self.operation_pure,
                "authority": "registered" if self.operation_id is not None else "exploratory",
            },
            "preconditions": dict(self.preconditions),
            "preserve": list(self.preserve),
        }

    @classmethod
    def register_operation(
        cls,
        operation_id: str,
        version: str,
        implementation: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        *,
        pure: bool,
    ) -> str:
        return _register_operation(operation_id, version, implementation, pure=pure)

    register = register_operation

    @classmethod
    def registered(
        cls,
        name: str,
        reads: tuple[str, ...],
        writes: tuple[str, ...],
        *,
        operation_id: str,
        operation_version: str,
        implementation: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        invalidates: tuple[str, ...] = (),
        numerical_contract: str = "behavioral-intervention",
        parameters: Mapping[str, Any] | None = None,
        preconditions: Mapping[str, str] | None = None,
        preserve: tuple[str, ...] = (),
    ) -> Act:
        registered = _OPERATION_REGISTRY.get((operation_id, operation_version))
        if registered is None:
            raise ValueError("operation id/version is not registered in this process")
        bound = registered[1] if implementation is None else implementation
        return cls(
            name,
            reads,
            writes,
            bound,
            invalidates,
            numerical_contract,
            {} if parameters is None else parameters,
            operation_id,
            operation_version,
            {} if preconditions is None else preconditions,
            preserve,
        )

    def require_reproducible(self) -> None:
        self.verify_implementation()
        if self.operation_id is None or self.operation_version is None or not self.operation_pure:
            raise ValueError("measured programs require a registered pure operation")
        registered = _OPERATION_REGISTRY.get((self.operation_id, self.operation_version))
        if registered is None or registered[0] != self.implementation_digest:
            raise ValueError("measured operation binding is unavailable or changed")

    def verify_implementation(self) -> None:
        observed_code = _implementation_digest(
            self.implementation, strict=self.operation_id is not None
        )
        observed_state = _implementation_state_digest(
            self.implementation, strict=self.operation_id is not None
        )
        if observed_code != self.implementation_digest:
            raise ValueError("Act implementation code or closure changed")
        if observed_state != self.implementation_state_digest:
            raise ValueError("Act implementation state changed")

    @classmethod
    def replace(cls, address: str, value: Any, *, name: str = "replace") -> Act:
        operation = _ReplaceOperation(address, value)
        _register_operation("saturn.builtin.replace", "1", operation, pure=True)
        return cls(
            name,
            (address,),
            (address,),
            operation,
            parameters={"value": clone(value)},
            operation_id="saturn.builtin.replace",
            operation_version="1",
        )

    @classmethod
    def add(cls, address: str, value: Any, *, dose: float = 1, name: str = "add") -> Act:
        operation = _AddOperation(address, value, dose)
        _register_operation("saturn.builtin.add", "1", operation, pure=True)
        return cls(
            name,
            (address,),
            (address,),
            operation,
            parameters={"value": clone(value), "dose": dose},
            operation_id="saturn.builtin.add",
            operation_version="1",
        )

    @classmethod
    def zero(cls, address: str) -> Act:
        operation = _ZeroOperation(address)
        _register_operation("saturn.builtin.zero", "1", operation, pure=True)
        return cls(
            "zero",
            (address,),
            (address,),
            operation,
            operation_id="saturn.builtin.zero",
            operation_version="1",
        )

    @classmethod
    def patch(
        cls,
        address: str,
        value: Any,
        mask: Any,
        *,
        mode: str = "replace",
        dose: float = 1,
        expected_digest: str | None = None,
    ) -> Act:
        if mode not in ("replace", "add"):
            raise ValueError("patch mode must be 'replace' or 'add'")
        if isinstance(dose, bool) or not isinstance(dose, (int, float)):
            raise ValueError("patch dose must be numeric")
        operation = _PatchOperation(address, value, mask, mode, dose)
        _register_operation("saturn.builtin.patch", "1", operation, pure=True)
        return cls(
            "patch",
            (address,),
            (address,),
            operation,
            parameters={
                "address": address,
                "value": clone(value),
                "mask": clone(mask),
                "mode": mode,
                "dose": dose,
            },
            operation_id="saturn.builtin.patch",
            operation_version="1",
            preconditions={} if expected_digest is None else {address: expected_digest},
        )


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

    def point(self, state: Mapping[str, Any]) -> ExecutionPoint:
        return ExecutionPoint(
            str(self.execution.get("family", type(self).__name__)),
            0,
            self.boundary(state),
            next_operation="advance",
        )

    def surface(self, state: Mapping[str, Any]) -> SurfaceManifest:
        writable = set(self.addresses(state))
        family = str(self.execution.get("family", type(self).__name__))
        return SurfaceManifest(
            tuple(SlotSpec(name, writable=name in writable) for name in sorted(state)),
            f"{family}-declared-state-v1",
            "native-suffix",
        )

    def validate_execution(self) -> None:
        """Adapters may reject drift of their frozen native executor."""

    def capture_external(self) -> Any:
        """Optional registered side-effect closure; default adapters must be pure."""
        return None

    def restore_external(self, snapshot: Any) -> None:
        if snapshot is not None:
            raise ValueError("adapter did not implement external closure restoration")

    def numerical_contract(self) -> NumericalContract:
        return NumericalContract(program=str(self.execution.get("adapter", "custom-adapter")))

    def transition(self, state: Mapping[str, Any]) -> TransitionSpec:
        """Describe the next safe transition; unspecified footprints stay unknown."""
        return TransitionSpec(self.point(state).next_operation or "advance")

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
        adapter.validate_execution()
        adapter.validate(state)
        adapter.surface(state).validate(state)
        self.adapter = adapter
        self._state = clone(state)
        self._external_state = clone(adapter.capture_external())
        self._parent = parent
        self._fork_parent = parent
        self._aborted = False
        self.receipts: list[Receipt] = []
        self.history: list[StateCut] = []
        self.attempts: list[Receipt] = []

    def inspect(self) -> Frame:
        self._activate()
        self.adapter.validate_execution()
        return Frame(
            self.adapter.boundary(self._state),
            {
                a: describe(self.adapter.read(self._state, a))
                for a in self.adapter.addresses(self._state)
            },
            self.adapter.point(self._state).to_dict(),
            self.adapter.surface(self._state).to_dict(),
        )

    def read(self, address: str) -> Any:
        self._activate()
        self.adapter.validate_execution()
        if address == "__saturn_external__" and self._external_state is not None:
            return clone(self._external_state)
        return self.adapter.read(self._state, address)

    def _activate(self) -> None:
        if self._external_state is not None:
            self.adapter.restore_external(clone(self._external_state))

    @staticmethod
    def _cut_state(cut: StateCut) -> tuple[dict[str, Any], Any]:
        payload = cut.payload
        return payload, payload.pop("__saturn_external__", None)

    def read_port(self, address: StateAddress) -> Any:
        surface = self.adapter.surface(self._state)
        slot = surface.slot(address.slot)
        point = self.adapter.point(self._state)
        if address.role and slot.role != address.role:
            raise ValueError("state port role mismatch")
        if address.execution_point and address.execution_point != point.fingerprint:
            raise ValueError("state port clock mismatch")
        if address.state_schema and address.state_schema != surface.state_schema:
            raise ValueError("state port schema mismatch")
        value = self.read(address.slot)
        if address.selector:
            try:
                if hasattr(value, "shape"):
                    value = value[address.selector]
                else:
                    for index in address.selector:
                        value = value[index]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError("state port selector is outside support") from exc
        return clone(value)

    def _cut(self, state: Mapping[str, Any], parent: str | None) -> StateCut:
        self.adapter.validate_execution()
        self.adapter.validate(state)
        surface = self.adapter.surface(state)
        surface.validate(state)
        external = clone(self.adapter.capture_external())
        payload = clone(dict(state))
        if "__saturn_external__" in payload:
            raise ValueError("reserved external closure slot used by adapter")
        if external is not None:
            payload["__saturn_external__"] = external
            surface = SurfaceManifest(
                surface.slots + (SlotSpec("__saturn_external__", role="runtime"),),
                surface.state_schema,
                surface.consumer,
                surface.horizon,
            )
        return StateCut(
            self.adapter.model_identity,
            self.adapter.execution,
            self.adapter.boundary(state),
            parent,
            payload,
            execution_point=self.adapter.point(state).to_dict(),
            surface=surface.to_dict(),
        )

    def capture(self, *, retain: bool = True) -> StateCut:
        self._activate()
        cut = self._cut(self._state, self._parent)
        if retain:
            self.history.append(cut)
        return cut

    @classmethod
    def from_cut(cls, adapter: Adapter, cut: StateCut) -> Session:
        """Hydrate a declared closure, including registered external state."""
        cut.verify()
        if cut.model_identity != adapter.model_identity or dict(cut.execution) != dict(
            adapter.execution
        ):
            raise ValueError("incompatible model identity or execution contract")
        state, external = cls._cut_state(cut)
        current = clone(adapter.capture_external())
        try:
            if external is not None:
                adapter.restore_external(clone(external))
            session = cls(adapter, state, parent=cut.parent)
            session._compatible(cut)
            session.restore(cut)
            return session
        finally:
            adapter.restore_external(current)

    def _compatible(self, cut: StateCut) -> None:
        self._activate()
        self.adapter.validate_execution()
        cut.verify()
        if cut.model_identity != self.adapter.model_identity or dict(cut.execution) != dict(
            self.adapter.execution
        ):
            raise ValueError("incompatible model identity or execution contract")
        state, external = self._cut_state(cut)
        current = clone(self.adapter.capture_external())
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            expected = self._cut(state, cut.parent)
            if cut.boundary != expected.boundary:
                raise ValueError("boundary does not match payload")
            if cut.execution_point is not None and dict(cut.execution_point) != dict(
                expected.execution_point or {}
            ):
                raise ValueError("execution point does not match payload")
            if cut.surface is not None and dict(cut.surface) != dict(expected.surface or {}):
                raise ValueError("state closure does not match adapter")
        finally:
            self.adapter.restore_external(current)

    def fork(self, cut: StateCut | None = None) -> Session:
        cut = cut or self.capture(retain=False)
        self._compatible(cut)
        state, external = self._cut_state(cut)
        current = clone(self.adapter.capture_external())
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            return Session(self.adapter, state, parent=cut.fingerprint)
        finally:
            self.adapter.restore_external(current)

    def _record(
        self,
        operation: str,
        before: StateCut,
        *,
        state: Mapping[str, Any] | None = None,
        **details: Any,
    ) -> Receipt:
        candidate = self._state if state is None else state
        after = self._cut(candidate, before.fingerprint)
        receipt = Receipt.make(
            operation=operation,
            parent=before.fingerprint,
            model_identity=self.adapter.model_identity,
            execution=dict(self.adapter.execution),
            result=after.fingerprint,
            boundary=after.boundary,
            execution_point=dict(after.execution_point or {}),
            numerical_contract=self.adapter.numerical_contract().to_dict(),
            **details,
        )
        self._state = clone(candidate)
        self._external_state = clone(self.adapter.capture_external())
        self._parent = before.fingerprint
        self.receipts.append(receipt)
        return receipt

    def _failed(self, operation: str, before: StateCut, error: Exception) -> None:
        self.attempts.append(
            Receipt.make(
                operation=operation,
                parent=before.fingerprint,
                result=before.fingerprint,
                status="refused",
                error=type(error).__name__,
                reason=str(error),
                physical_attempt=len(self.attempts) + 1,
            )
        )

    def apply(self, act: Act) -> Receipt:
        before = self.capture(retain=False)
        external = clone(self.adapter.capture_external())
        try:
            act.verify_implementation()
            for address, expected in act.preconditions.items():
                if digest(describe(self.read(address))) != expected:
                    raise ValueError(f"Act precondition changed: {address}")
            surface = self.adapter.surface(self._state)
            for address in act.writes:
                slot = surface.slot(address)
                if not slot.writable:
                    raise ValueError(f"read-only state port: {address}")
                if set(slot.invalidates) - set(act.invalidates):
                    raise ValueError(f"missing declared invalidations for {address}")
            preserved = {
                slot.name: describe(self._state[slot.name])
                for slot in surface.slots
                if slot.persistence == "authoritative"
                and slot.name not in set(act.writes) | set(act.invalidates)
            }
            preserved.update({a: describe(self.read(a)) for a in act.preserve})
            reads = MappingProxyType({a: self.read(a) for a in act.reads})
            writes = dict(act.implementation(reads))
            if act.operation_pure:
                act.verify_implementation()
            if set(writes) != set(act.writes):
                raise ValueError("implementation must return exactly the declared writes")
            state = self.adapter.write(clone(self._state), writes, act.invalidates)
            for address, value in preserved.items():
                if address not in state or describe(state[address]) != value:
                    raise ValueError(f"Act changed preserved port: {address}")
            return self._record(
                "apply", before, state=state, act=act.manifest(), writes=describe(writes)
            )
        except Exception as exc:
            self.adapter.restore_external(external)
            self._failed("apply", before, exc)
            raise

    def continue_(self, steps: int = 1) -> Receipt:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        before = self.capture(retain=False)
        external = clone(self.adapter.capture_external())
        try:
            candidate = clone(self._state)
            transitions = []
            for _ in range(steps):
                entry = self.adapter.point(candidate).to_dict()
                spec = self.adapter.transition(candidate)
                slots = {slot.name for slot in self.adapter.surface(candidate).slots}
                if set(spec.reads + spec.writes + spec.invalidates) - slots:
                    raise ValueError("transition footprint contains undeclared slots")
                preserved = (
                    {
                        name: describe(value)
                        for name, value in candidate.items()
                        if name not in set(spec.writes + spec.invalidates)
                    }
                    if spec.footprint == "declared"
                    else {}
                )
                candidate = dict(self.adapter.advance(candidate))
                for name, expected in preserved.items():
                    if name not in candidate or describe(candidate[name]) != expected:
                        raise ValueError(f"native transition changed undeclared slot: {name}")
                self.adapter.validate(candidate)
                self.adapter.surface(candidate).validate(candidate)
                self.adapter.validate_execution()
                transitions.append(
                    {
                        "before": entry,
                        "after": self.adapter.point(candidate).to_dict(),
                        "native_operation": spec.to_dict(),
                        "structural_operation": spec.operation,
                        "occurrence": digest(
                            {
                                "entry": entry,
                                "operation": spec.operation,
                                "transaction_parent": before.fingerprint,
                                "transition_index": len(transitions),
                                "execution": dict(self.adapter.execution),
                            }
                        ),
                    }
                )
            return self._record(
                "continue", before, state=candidate, steps=steps, transitions=transitions
            )
        except Exception as exc:
            self.adapter.restore_external(external)
            self._failed("continue", before, exc)
            raise

    def step(self, *, granularity: str = "transition", max_steps: int = 10000) -> Receipt:
        if granularity == "transition":
            return self.continue_(1)
        if granularity not in {"token", "denoise"}:
            raise ValueError("step granularity must be transition, token or denoise")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._activate()
        point = self.adapter.point(self._state)
        family = self.adapter.execution.get("family", "")
        if (granularity == "token" and "qwen" not in family) or (
            granularity == "denoise"
            and family not in {"ddim", "diffusion", "diffusion-ddim", "flux2-klein"}
        ):
            raise ValueError("macro-step is not supported by this family")
        before = self.capture(retain=False)
        branch = self.fork(before)
        try:
            for count in range(1, max_steps + 1):
                branch.continue_()
                if self.adapter.point(branch._state).logical_step != point.logical_step:
                    return self._record(
                        "step",
                        before,
                        state=branch._state,
                        steps=count,
                        granularity=granularity,
                        micro_receipts=[r.to_dict() for r in branch.receipts],
                    )
            raise ValueError("macro-step limit reached before a complete boundary")
        except Exception as exc:
            self.history.append(branch.capture())
            self._activate()
            self._failed("step", before, exc)
            raise

    def replay(self, cut: StateCut, *, steps: int = 1) -> Session:
        branch = self.fork(cut)
        branch.continue_(steps)
        return branch

    def restore(self, cut: StateCut) -> Receipt:
        self._compatible(cut)
        before = self.capture(retain=False)
        state, external = self._cut_state(cut)
        current = clone(self.adapter.capture_external())
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            restored = self._cut(state, cut.parent)
            if restored.fingerprint != cut.fingerprint:
                raise ValueError("restore fingerprint mismatch")
            receipt = Receipt.make(
                operation="restore",
                parent=before.fingerprint,
                result=cut.fingerprint,
                verified_exact=True,
                model_identity=self.adapter.model_identity,
                execution=dict(self.adapter.execution),
                execution_point=dict(cut.execution_point or {}),
                numerical_contract=self.adapter.numerical_contract().to_dict(),
            )
        except Exception as exc:
            self.adapter.restore_external(current)
            self._failed("restore", before, exc)
            raise
        self._state = state
        self._external_state = external
        self._parent = cut.parent
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
        state, external = self._cut_state(candidate)
        current = clone(self._external_state)
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            return self._record("commit", parent, state=state, candidate=candidate.fingerprint)
        except Exception as exc:
            self.adapter.restore_external(current)
            self._failed("commit", parent, exc)
            raise

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

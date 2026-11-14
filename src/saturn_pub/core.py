"""One transaction lifecycle for native model execution and counterfactual research."""

from __future__ import annotations

import copy
import hashlib
import inspect
import marshal
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from types import CodeType, MappingProxyType
from typing import Any, NamedTuple

from .contracts import (
    ExecutionPoint,
    NumericalContract,
    SlotSpec,
    StateAddress,
    SurfaceManifest,
    TransitionSpec,
)
from .values import clone, describe, describe_mapping, digest, tensor

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


def _statecut_manifest(
    model_identity: str,
    execution: Mapping[str, Any],
    boundary: str,
    parent: str | None,
    payload: Any,
    execution_point: Mapping[str, Any] | None,
    surface: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The sealed manifest of a StateCut, given the ``describe`` of its payload."""
    return {
        "schema": "saturn-pub-statecut-v2",
        "model_identity": model_identity,
        "execution": dict(execution),
        "boundary": boundary,
        "parent": parent,
        "payload": payload,
        "execution_point": None if execution_point is None else describe(dict(execution_point)),
        "surface": None if surface is None else describe(dict(surface)),
    }


def _slot_descriptors(described: Any) -> dict[str, Any] | None:
    """``{slot: describe(value)}`` from the ``describe`` of a payload mapping (None when a key is
    not a plain scalar, so the caller hashes instead)."""
    out: dict[str, Any] = {}
    for item in described["items"]:
        key = item["key"]
        if key.get("kind") != "scalar":
            return None
        out[key["value"]] = item["value"]
    return out


class _Appended:
    """A sealed tensor held as blocks: ``torch.cat(chunks, dim)`` is it bit for bit, with the
    standard contiguous layout it was sealed with. A cut holds a grown cache tensor this way, so
    consecutive cuts share every block and each adds only the rows its transition appended."""

    __slots__ = ("chunks", "dim")

    def __init__(self, chunks: tuple[Any, ...], dim: int):
        self.chunks, self.dim = chunks, dim

    def materialize(self, *, register: bool) -> Any:
        """The tensor (a fresh one). ``register``: the caller (a Session hydrating its own state)
        will treat it as immutable, so a later cut sharing it is sealed as these same blocks."""
        import torch

        with torch.inference_mode(False):
            full = torch.cat(self.chunks, dim=self.dim)
        if register:
            _register_appended(full, self)
        return full


_APPENDED: dict[int, tuple[Any, int, _Appended]] = {}


def _register_appended(full: Any, compact: _Appended) -> None:
    import weakref

    key = id(full)

    def forget(_: Any, key: int = key) -> None:
        entry = _APPENDED.get(key)
        if entry is not None and entry[0]() is None:
            _APPENDED.pop(key, None)

    _APPENDED[key] = (weakref.ref(full, forget), full._version, compact)


def _appended_of(value: Any) -> _Appended | None:
    """The block form registered for ``value`` (None when none is, or when ``value`` was
    written in place since it was registered)."""
    entry = _APPENDED.get(id(value))
    if entry is None or entry[0]() is not value or value._version != entry[1]:
        return None
    return entry[2]


def _standard_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= max(int(size), 1)
    return tuple(reversed(strides))


def seal_append(full: Any, base: Any, dim: int) -> bool:
    """Declare that ``full`` is ``base`` with rows appended along ``dim`` (bitwise: the caller
    checked it), so a cut that holds ``full`` holds ``base``'s blocks plus the appended rows only.
    ``full`` must have the standard contiguous layout (``torch.cat`` gives it) and keep a version
    counter; otherwise nothing is declared. Returns whether it was."""
    dim = dim % full.dim()
    if (
        not _versioned(full)
        or full.requires_grad
        or tuple(full.stride()) != _standard_strides(tuple(full.shape))
        or base.dim() != full.dim()
        or base.shape[dim] >= full.shape[dim]
    ):
        return False
    count = base.shape[dim]
    rest = full.narrow(dim, count, full.shape[dim] - count).clone(
        memory_format=__import__("torch").contiguous_format
    )
    known = _appended_of(base) if _versioned(base) else None
    if known is not None and known.dim == dim:
        chunks = (*known.chunks, rest)
    elif count == 0:
        chunks = (rest,)
    else:
        chunks = (_share(base), rest)
    _register_appended(full, _Appended(chunks, dim))
    return True


def _compact(value: Any) -> Any:
    """``_share`` for sealing: a tensor declared with ``seal_append`` is held as its blocks."""
    if tensor(value):
        if _versioned(value):
            known = _appended_of(value)
            if known is not None:
                return known
        return _share(value)
    if isinstance(value, Mapping):
        return {key: _compact(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_compact(item) for item in value)
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return copy.deepcopy(value)


def _expand(value: Any, *, register: bool) -> Any:
    """``value`` with every block form materialized (tensors and other leaves as they are;
    containers copied)."""
    if isinstance(value, _Appended):
        return value.materialize(register=register)
    if tensor(value):
        return value
    if isinstance(value, Mapping):
        return {key: _expand(item, register=register) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_expand(item, register=register) for item in value)
    if isinstance(value, list):
        return [_expand(item, register=register) for item in value]
    return copy.deepcopy(value)


def _tensor_leaves(value: Any, path: str = "") -> list[tuple[str, Any]]:
    """``(path, tensor)`` for every tensor in a payload (mappings, lists and tuples walked, and
    the blocks of a block form)."""
    if tensor(value):
        return [(path, value)]
    out: list[tuple[str, Any]] = []
    if isinstance(value, _Appended):
        for index, item in enumerate(value.chunks):
            out += _tensor_leaves(item, f"{path}<{index}>")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            out += _tensor_leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            out += _tensor_leaves(item, f"{path}[{index}]")
    return out


def held_bytes(*objects: Any) -> int:
    """Bytes of the distinct tensor storages that ``objects`` reference (StateCuts, Sessions,
    states, mappings and sequences of them); a storage two of them share counts once."""
    storages: dict[Any, int] = {}

    def walk(value: Any) -> None:
        if isinstance(value, StateCut):
            walk(value._payload)
        elif isinstance(value, Session):
            walk(value._state)
        else:
            for _, item in _tensor_leaves(value):
                storage = item.untyped_storage()
                storages[(str(item.device), storage.data_ptr())] = storage.nbytes()

    for item in objects:
        walk(item)
    return sum(storages.values())


def _versioned(value: Any) -> bool:
    """Whether a tensor tracks in-place writes (an inference tensor has no version counter)."""
    is_inference = getattr(value, "is_inference", None)
    return not (callable(is_inference) and is_inference())


def _share(value: Any) -> Any:
    """``value`` with its containers copied and its tensors shared, for a sealed (immutable)
    state: a tensor that tracks in-place writes is kept by reference, an inference tensor (no
    version counter, so a write to it could not be detected) is copied into a normal one."""
    if tensor(value):
        if value.requires_grad:
            return clone(value)
        if _versioned(value):
            return value
        import torch

        with torch.inference_mode(False):
            return value.clone(memory_format=torch.preserve_format)
    if isinstance(value, Mapping):
        return {key: _share(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_share(item) for item in value)
    if isinstance(value, list):
        return [_share(item) for item in value]
    return copy.deepcopy(value)


class InPlaceWrite(ValueError):
    """A tensor that a sealed cut or a transition's input state shares was written in place."""


def _versions(value: Any) -> tuple[tuple[str, Any, int], ...]:
    """``(path, tensor, version)`` of every tensor of ``value`` that tracks in-place writes."""
    return tuple(
        (path, item, item._version) for path, item in _tensor_leaves(value) if _versioned(item)
    )


def _check_versions(recorded: tuple[tuple[str, Any, int], ...], what: str) -> None:
    written = [path for path, item, version in recorded if item._version != version]
    if written:
        raise InPlaceWrite(f"{what} was written in place: {', '.join(written)}")


@dataclass(frozen=True)
class StateCut:
    """A sealed snapshot of an adapter's declared mutable execution closure.

    The payload is copied once at construction, unless the caller passes ``_shared=True``
    (Session only, for an adapter whose ``shares_state`` contract holds): then the cut keeps the
    state's tensors by reference (its containers are copied), so consecutive cuts of one session
    share every tensor a transition did not replace. A sealed tensor is immutable: the cut
    records each tensor's version counter, and every read of the payload (``payload``,
    ``slot``, ``verify``, a restore) refuses (``InPlaceWrite``, naming the slot) when one was
    written in place since. ``_described`` (private, Session only) is the ``describe`` of that
    payload when the caller already holds it for the very tensors it hands in, so the
    fingerprint is sealed without hashing them again; with it, ``_sealed`` is the fingerprint
    Session already digested from exactly these fields (``Session._seal``), so the manifest is
    not described twice. ``verify`` always re-hashes.
    """

    model_identity: str
    execution: Mapping[str, Any]
    boundary: str
    parent: str | None
    _payload: Mapping[str, Any] = field(repr=False)
    execution_point: Mapping[str, Any] | None = None
    surface: Mapping[str, Any] | None = None
    fingerprint: str = field(init=False)
    _described: InitVar[Any] = None
    _sealed: InitVar[str | None] = None
    _shared: InitVar[bool] = False

    def __post_init__(
        self, _described: Any = None, _sealed: str | None = None, _shared: bool = False
    ) -> None:
        object.__setattr__(self, "execution", MappingProxyType(clone(dict(self.execution))))
        payload = dict(self._payload)
        object.__setattr__(self, "_payload", _compact(payload) if _shared else clone(payload))
        object.__setattr__(self, "_guard", _versions(self._payload))
        if self.execution_point is not None:
            object.__setattr__(
                self,
                "execution_point",
                MappingProxyType(clone(dict(self.execution_point))),
            )
        if self.surface is not None:
            object.__setattr__(self, "surface", MappingProxyType(clone(dict(self.surface))))
        if _described is not None and _sealed is not None:
            object.__setattr__(self, "fingerprint", _sealed)  # Session sealed these very fields
            return
        payload = describe(self._payload) if _described is None else _described
        object.__setattr__(self, "fingerprint", digest(self._manifest(payload)))

    @property
    def payload(self) -> dict[str, Any]:
        self._check_sealed()
        return self._private(self._payload)

    def slot(self, name: str) -> Any:
        """One payload slot as a private copy (``payload[name]`` without copying the others)."""
        self._check_sealed()
        return self._private(self._payload[name])

    @staticmethod
    def _private(value: Any) -> Any:
        """A private copy: block forms are concatenated (already a fresh tensor), the rest
        cloned."""
        if isinstance(value, _Appended):
            return value.materialize(register=False)
        if isinstance(value, Mapping):
            return {key: StateCut._private(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(StateCut._private(item) for item in value)
        if isinstance(value, list):
            return [StateCut._private(item) for item in value]
        return clone(value)

    def _check_sealed(self) -> None:
        """Refuse when a payload tensor was written in place since the cut was sealed."""
        _check_versions(self._guard, "a sealed StateCut tensor")

    def _shared_payload(self) -> dict[str, Any]:
        """The payload with its containers copied and its (immutable) tensors shared, block
        forms materialized (and registered, so cuts of the hydrated state share the blocks): the
        state a Session hydrates from this cut when its adapter's ``shares_state`` holds."""
        self._check_sealed()
        return _share(_expand(self._payload, register=True))

    def _hydrate(self, shared: bool) -> tuple[Any, dict[str, Any]]:
        """``(describe of the payload, state)`` for a Session taking this cut: the payload is
        verified against the fingerprint (as ``verify``) and handed over as ``_shared_payload``
        (``shared``) or as a private copy, materializing block forms once."""
        self._check_sealed()
        if shared:
            state = _share(_expand(self._payload, register=True))
        else:
            state = self._private(self._payload)
        described = describe(state)
        if digest(self._manifest(described)) != self.fingerprint:
            raise ValueError("StateCut payload or metadata changed")
        return described, state

    def nbytes(self) -> int:
        """Bytes of the distinct tensor storages this cut references (shared ones included)."""
        return held_bytes(self)

    def _manifest(self, payload: Any) -> dict[str, Any]:
        return _statecut_manifest(
            self.model_identity,
            self.execution,
            self.boundary,
            self.parent,
            payload,
            self.execution_point,
            self.surface,
        )

    def manifest(self) -> dict[str, Any]:
        self._check_sealed()
        return self._manifest(describe(_expand(self._payload, register=False)))

    def _verified(self) -> Any:
        """Hash the payload once, refuse a cut whose content or metadata no longer matches its
        fingerprint, and return the payload's ``describe``."""
        self._check_sealed()
        described = describe(_expand(self._payload, register=False))
        if digest(self._manifest(described)) != self.fingerprint:
            raise ValueError("StateCut payload or metadata changed")
        return described

    def verify(self) -> None:
        self._verified()


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


class PortStop(Exception):
    """Raised by a port controller to end a read-only probe at a declared port."""


class Adapter(ABC):
    """Native executor contract. Execution state belongs to a Session, not the model.

    Adapters that expose ports accept ``advance(state, *, ports=controller)`` and install the
    controller around their native call; ``ports(state)`` names the hook sites the next
    transition offers. A controller may define a per-transition lifecycle: ``begin()`` is
    called before each transition's ``advance`` (in ``probe`` and ``continue_``);
    ``probe_check()`` after a probe's ``advance``; ``manifest()`` once after the same
    ``advance`` in ``continue_``, never earlier. A non-empty ``manifest()`` is recorded as that
    transition's ``port_edits`` and so must list only edits applied during that call; it may
    raise to refuse committing a transition whose staged edits did not take effect.

    A controller may also define ``check_ahead(ahead)``, called by ``continue_`` before anything
    runs with ``adapter.ports_ahead(state, steps)``; it raises to refuse a call in which none of
    the transitions would reach what the controller addresses. ``begin_call()`` / ``end_call()``
    bracket one ``continue_`` call (``end_call`` raises if a controller that carries edits was
    never installed by the call).
    """

    model_identity: str
    execution: Mapping[str, Any]
    # The immutable-state contract. True: ``advance`` (hooks and port controllers included)
    # never writes a tensor of its input state in place and may return the input's unchanged
    # tensors in its result. A Session then hands ``advance`` its state without copying it and
    # seals cuts that share tensors by reference (consecutive cuts share what a transition did
    # not replace); every transition's input tensors are checked by version counter afterwards
    # and a write is refused (``InPlaceWrite``). False (the default): every state handed to
    # ``advance`` is a private copy and every cut owns a copy of its payload.
    shares_state: bool = False

    @abstractmethod
    def boundary(self, state: Mapping[str, Any]) -> str: ...

    @abstractmethod
    def validate(self, state: Mapping[str, Any]) -> None: ...

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """``validate`` for a caller that ran ``validate_execution`` earlier in the same call; an
        adapter whose ``validate`` repeats a check of ``validate_execution`` may skip it here."""
        self.validate(state)

    @abstractmethod
    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]: ...

    @abstractmethod
    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]: ...

    def ports(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        """Hook sites of the next transition; empty means it accepts no port controller."""
        return ()

    def ports_ahead(self, state: Mapping[str, Any], steps: int) -> tuple[Any, ...] | None:
        """For each of the next ``steps`` transitions, the site index a port controller can
        address there (``None`` for a transition that offers no ports); ``None`` overall when
        the adapter cannot say without running."""
        return None

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

    def read_part(self, state: Mapping[str, Any], address: str, index: Any) -> Any:
        """``read(state, address)[index]``; an adapter may override it to copy only that part."""
        return self.read(state, address)[index]

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


def _flatten_execution(execution: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in execution.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(_flatten_execution(value, prefix=f"{name}."))
        else:
            out[name] = value
    return out


def _refuse_incompatible(cut: StateCut, adapter: Adapter) -> None:
    """Refuse a cut of another model identity or execution contract, naming every differing
    execution field (``numeric_scope.bf16_reduced_precision_reduction``, say)."""
    if cut.model_identity != adapter.model_identity:
        raise ValueError("incompatible model identity or execution contract (model_identity)")
    recorded, current = dict(cut.execution), dict(adapter.execution)
    if recorded == current:
        return
    left, right = _flatten_execution(recorded), _flatten_execution(current)
    differing = [
        f"{name}: cut {left.get(name, 'missing')!r}, adapter {right.get(name, 'missing')!r}"
        for name in sorted(set(left) | set(right))
        if left.get(name, "missing") != right.get(name, "missing")
    ]
    raise ValueError(
        "incompatible model identity or execution contract (" + "; ".join(differing) + ")"
    )


def _validate_state(adapter: Any, state: Mapping[str, Any]) -> None:
    """``adapter.validate_state(state)``, or ``validate`` for an adapter without it."""
    getattr(adapter, "validate_state", adapter.validate)(state)


def _adopt(value: Any) -> Any:
    """``clone(value)`` for a value its owner hands over and no longer references: the same
    content and layout, except that a contiguous tensor (of which ``clone`` would make an
    identical copy, strides included) is kept instead of copied."""
    if tensor(value):
        return value if value.is_contiguous() and not value.requires_grad else clone(value)
    if isinstance(value, Mapping):
        return {key: _adopt(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_adopt(item) for item in value)
    if isinstance(value, list):
        return [_adopt(item) for item in value]
    return copy.deepcopy(value)


class _Seal(NamedTuple):
    """What a StateCut of a state records, computed without copying the state."""

    payload: dict[str, Any]  # the state's values plus the external closure, not copied
    slots: dict[str, Any]  # {slot: describe(value)} of the state (external closure excluded)
    described: Any  # describe of the payload
    boundary: str
    point: dict[str, Any]
    surface: dict[str, Any]
    fingerprint: str


class Session:
    """Resident native executor with isolated branches and explicit commit."""

    def __init__(self, adapter: Adapter, state: Mapping[str, Any], *, parent: str | None = None):
        adapter.validate_execution()
        _validate_state(adapter, state)
        adapter.surface(state).validate(state)
        self._open(adapter, clone(state), parent)

    def _open(
        self,
        adapter: Adapter,
        state: dict[str, Any],
        parent: str | None,
        slots: dict[str, Any] | None = None,
    ) -> None:
        """Start on ``state``, validated by the caller and handed over (never copied here).

        ``_slots`` is ``{slot: describe(value)}`` of ``_state`` or None until computed. It is set
        only together with a validated ``_state``, and ``_state`` is private (every read copies),
        so a cut of it is sealed from these descriptors without hashing or validating again.
        """
        self.adapter = adapter
        self._state = state
        self._slots = slots
        self._external_state = clone(adapter.capture_external())
        self._parent = parent
        self._fork_parent = parent
        self._aborted = False
        self._poisoned: str | None = None  # an in-place write reached the state: see _advance_guarded
        self.receipts: list[Receipt] = []
        self.history: list[StateCut] = []
        self.attempts: list[Receipt] = []

    def _check_poison(self) -> None:
        if self._poisoned is not None:
            raise InPlaceWrite(
                f"{self._poisoned}; this session's state is unusable until restore(cut) of a "
                "cut that does not share the written tensor"
            )

    def inspect(self) -> Frame:
        self._check_poison()
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

    def read(self, address: str, index: Any = None) -> Any:
        """A private copy of ``address``; with ``index``, of ``address[index]`` only (one layer
        of a per-layer cache slot, say), through the adapter's ``read_part``."""
        self._check_poison()
        self._activate()
        self.adapter.validate_execution()
        if address == "__saturn_external__" and self._external_state is not None:
            value = clone(self._external_state)
            return value if index is None else value[index]
        if index is None:
            return self.adapter.read(self._state, address)
        read_part = getattr(self.adapter, "read_part", None)
        if read_part is None:
            return self.adapter.read(self._state, address)[index]
        return read_part(self._state, address, index)

    def _activate(self) -> None:
        if self._external_state is not None:
            self.adapter.restore_external(clone(self._external_state))

    @staticmethod
    def _cut_state(cut: StateCut, shared: bool = False) -> tuple[Any, dict[str, Any], Any]:
        """``(describe of the payload, state, external)`` of a cut, verified against its
        fingerprint: a private copy of its payload, or (``shared``, the adapter's
        ``shares_state`` contract) its immutable tensors shared."""
        described, payload = cut._hydrate(shared)
        return described, payload, payload.pop("__saturn_external__", None)

    def _shares(self) -> bool:
        return bool(getattr(self.adapter, "shares_state", False))

    def _input(self) -> dict[str, Any]:
        """The state handed to ``advance``: shared (containers copied, tensors by reference)
        under the adapter's ``shares_state`` contract, else a private copy."""
        return _share(self._state) if self._shares() else clone(self._state)

    def _advance_guarded(self, candidate: dict[str, Any], **options: Any) -> Any:
        """``adapter.advance(candidate, **options)``, refusing (``InPlaceWrite``, naming the
        slot) when the call wrote one of the input's tensors in place: under ``shares_state``
        those tensors are shared with sealed cuts and the session state. Checked on every exit,
        a port controller's ``PortStop`` included."""
        if not self._shares():
            return self.adapter.advance(candidate, **options)
        recorded = _versions(candidate)
        try:
            result = self.adapter.advance(candidate, **options)
        except BaseException:
            self._guard_input(recorded)
            raise
        self._guard_input(recorded)
        return result

    def _guard_input(self, recorded: tuple[tuple[str, Any, int], ...]) -> None:
        """Refuse a transition that wrote its input in place, and poison the session: its state
        shares the written tensor, so it refuses every use until a restore replaces it."""
        try:
            _check_versions(recorded, "the transition's input state")
        except InPlaceWrite as error:
            self._poisoned = str(error)
            raise

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
            return clone(value)  # a compact copy of the selected part, not a view of the read
        return value  # ``read`` already returned a private copy

    def _seal(
        self,
        state: Mapping[str, Any],
        parent: str | None,
        slots: Mapping[str, Any] | None = None,
        *,
        execution_checked: bool = False,
        state_checked: bool = False,
    ) -> _Seal:
        """What ``_cut(state, parent)`` records, copying nothing: validates as a cut does unless
        the caller already did in this call (``execution_checked``: ``validate_execution``;
        ``state_checked``: the state's own checks) and hashes only the slots missing from
        ``slots`` (``{slot: describe(value)}`` known for these very values)."""
        if not execution_checked:
            self.adapter.validate_execution()
        if not state_checked:
            _validate_state(self.adapter, state)
        surface = self.adapter.surface(state)
        if not state_checked:
            surface.validate(state)
        external = clone(self.adapter.capture_external())
        if "__saturn_external__" in state:
            raise ValueError("reserved external closure slot used by adapter")
        own = {
            name: slots[name] if slots is not None and name in slots else describe(value)
            for name, value in state.items()
        }
        payload, described = dict(state), dict(own)
        if external is not None:
            payload["__saturn_external__"] = external
            described["__saturn_external__"] = describe(external)
            surface = SurfaceManifest(
                surface.slots + (SlotSpec("__saturn_external__", role="runtime"),),
                surface.state_schema,
                surface.consumer,
                surface.horizon,
            )
        sealed = describe_mapping(described)
        boundary = self.adapter.boundary(state)
        point = self.adapter.point(state).to_dict()
        manifest = surface.to_dict()
        fingerprint = digest(
            _statecut_manifest(
                self.adapter.model_identity,
                self.adapter.execution,
                boundary,
                parent,
                sealed,
                point,
                manifest,
            )
        )
        return _Seal(payload, own, sealed, boundary, point, manifest, fingerprint)

    def _cut(
        self,
        state: Mapping[str, Any],
        parent: str | None,
        slots: Mapping[str, Any] | None = None,
        *,
        state_checked: bool = False,
    ) -> StateCut:
        seal = self._seal(state, parent, slots, state_checked=state_checked)
        return StateCut(
            self.adapter.model_identity,
            self.adapter.execution,
            seal.boundary,
            parent,
            seal.payload,
            execution_point=seal.point,
            surface=seal.surface,
            _described=seal.described,
            _sealed=seal.fingerprint,
        )

    def _head(self, *, execution_checked: bool = False, poisoned: bool = False) -> str:
        """The fingerprint ``capture()`` would give now, with its checks, copying nothing (and
        hashing nothing once the state's slot descriptors are known). ``poisoned``: allowed on a
        session whose state an in-place write reached (the restore that replaces it)."""
        if not poisoned:
            self._check_poison()
        self._activate()
        seal = self._seal(
            self._state,
            self._parent,
            self._slots,
            execution_checked=execution_checked,
            state_checked=self._slots is not None,
        )
        self._slots = seal.slots
        return seal.fingerprint

    def capture(self, *, retain: bool = True) -> StateCut:
        self._check_poison()
        self._activate()
        seal = self._seal(
            self._state, self._parent, self._slots, state_checked=self._slots is not None
        )
        self._slots = seal.slots
        cut = StateCut(
            self.adapter.model_identity,
            self.adapter.execution,
            seal.boundary,
            self._parent,
            seal.payload,
            execution_point=seal.point,
            surface=seal.surface,
            _described=seal.described,
            _sealed=seal.fingerprint,
            _shared=self._shares(),
        )
        if retain:
            self.history.append(cut)
        return cut

    @classmethod
    def from_cut(cls, adapter: Adapter, cut: StateCut) -> Session:
        """Hydrate a declared closure, including registered external state.

        One hash (``verify``), one copy of the payload and one validation: the hydrated state is
        that copy, its slot descriptors are the verified ones, and the restore receipt the
        session starts with is the one ``restore(cut)`` writes there.
        """
        _refuse_incompatible(cut, adapter)
        described, state, external = cls._cut_state(
            cut, bool(getattr(adapter, "shares_state", False))
        )
        slots = cls._state_slots(described)
        current = clone(adapter.capture_external())
        try:
            if external is not None:
                adapter.restore_external(clone(external))
            adapter.validate_execution()
            _validate_state(adapter, state)
            adapter.surface(state).validate(state)
            session = cls.__new__(cls)
            session._open(adapter, state, cut.parent, slots)
            session._activate()
            session._matches(cut, state, external)
            session._settle(cut, state, external, slots, session._head(execution_checked=True))
            return session
        finally:
            adapter.restore_external(current)

    @staticmethod
    def _state_slots(described: Any) -> dict[str, Any] | None:
        """Slot descriptors of the state a cut's payload hydrates to (its external closure is
        not part of the state)."""
        slots = _slot_descriptors(described)
        if slots is not None:
            slots.pop("__saturn_external__", None)
        return slots

    def _matches(
        self, cut: StateCut, state: Mapping[str, Any], external: Any, *, validate: bool = False
    ) -> None:
        """Refuse a cut whose boundary, execution point or state closure differs from what the
        adapter derives from its payload ``state`` with the cut's external closure installed
        (``validate``: run the state checks first, as building that cut would)."""
        current = clone(self.adapter.capture_external())
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            if validate:
                _validate_state(self.adapter, state)
            surface = self.adapter.surface(state)
            if validate:
                surface.validate(state)
            if clone(self.adapter.capture_external()) is not None:
                surface = SurfaceManifest(
                    surface.slots + (SlotSpec("__saturn_external__", role="runtime"),),
                    surface.state_schema,
                    surface.consumer,
                    surface.horizon,
                )
            if cut.boundary != self.adapter.boundary(state):
                raise ValueError("boundary does not match payload")
            if cut.execution_point is not None and dict(cut.execution_point) != (
                self.adapter.point(state).to_dict()
            ):
                raise ValueError("execution point does not match payload")
            if cut.surface is not None and dict(cut.surface) != surface.to_dict():
                raise ValueError("state closure does not match adapter")
        finally:
            self.adapter.restore_external(current)

    def _compatible(self, cut: StateCut) -> tuple[dict[str, Any], Any, dict[str, Any] | None]:
        """Refuse a cut this session cannot take; returns ``(state, external, slots)``: a private
        copy of its payload, its external closure and the state's verified slot descriptors."""
        self._activate()
        self.adapter.validate_execution()
        _refuse_incompatible(cut, self.adapter)
        described, state, external = self._cut_state(cut, self._shares())
        self._matches(cut, state, external, validate=True)
        return state, external, self._state_slots(described)

    def fork(self, cut: StateCut | None = None) -> Session:
        if cut is None:
            parent = self._head()
            state, external, slots = self._input(), self._external_state, self._slots
        else:
            state, external, slots = self._compatible(cut)
            parent = cut.fingerprint
        current = clone(self.adapter.capture_external())
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            branch = Session.__new__(Session)
            branch._open(self.adapter, state, parent, slots)
            return branch
        finally:
            self.adapter.restore_external(current)

    def _record(
        self,
        operation: str,
        before: str,
        *,
        state: Mapping[str, Any] | None = None,
        owned: bool = False,
        slots: Mapping[str, Any] | None = None,
        checked: bool = False,
        **details: Any,
    ) -> Receipt:
        """Make ``state`` (default: the current one) this session's state, with ``before`` (a
        fingerprint) as its parent, and write the receipt. ``owned``: the caller hands ``state``
        over and keeps no reference, so it is adopted rather than copied. ``slots``: descriptors
        known for some of its values. ``checked``: the caller validated it (and the executor)
        just now."""
        candidate = self._state if state is None else state
        adopted = _adopt(candidate) if owned else clone(candidate)
        seal = self._seal(adopted, before, slots, execution_checked=checked, state_checked=checked)
        receipt = Receipt.make(
            operation=operation,
            parent=before,
            model_identity=self.adapter.model_identity,
            execution=dict(self.adapter.execution),
            result=seal.fingerprint,
            boundary=seal.boundary,
            execution_point=dict(seal.point),
            numerical_contract=self.adapter.numerical_contract().to_dict(),
            **details,
        )
        self._state = adopted
        self._slots = seal.slots
        self._external_state = clone(self.adapter.capture_external())
        self._parent = before
        self.receipts.append(receipt)
        return receipt

    def _failed(self, operation: str, before: str, error: Exception) -> None:
        self.attempts.append(
            Receipt.make(
                operation=operation,
                parent=before,
                result=before,
                status="refused",
                error=type(error).__name__,
                reason=str(error),
                physical_attempt=len(self.attempts) + 1,
            )
        )

    def apply(self, act: Act) -> Receipt:
        before = self._head()
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

    def _accepts_ports(self) -> bool:
        return "ports" in inspect.signature(self.adapter.advance).parameters

    def _advance_options(self) -> dict[str, Any]:
        """``{"validated": True}`` for an adapter whose ``advance`` takes it: the session hands
        ``advance`` only states it validated (with ``validate_execution``) in the same call."""
        if "validated" in inspect.signature(self.adapter.advance).parameters:
            return {"validated": True}
        return {}

    def probe(self, controller: Any) -> dict[str, Any]:
        """Run the next transition on a throw-away candidate with ``controller`` installed.

        Read-only: records no receipt, attempt or history entry and leaves the session state
        and external closure as they were. ``PortStop`` ends the probe normally; any other
        exception propagates unchanged.
        """
        self._check_poison()
        self._activate()
        self.adapter.validate_execution()
        if not self._accepts_ports() or not self.adapter.ports(self._state):
            raise ValueError(f"no ports at {self.adapter.boundary(self._state)}")
        external = clone(self.adapter.capture_external())
        candidate = self._input()
        stopped = False
        begin = getattr(controller, "begin", None)
        if begin is not None:
            begin()
        options = self._advance_options()
        if options:  # the state is validated; re-check the executor after the controller's begin
            self.adapter.validate_execution()
        try:
            self._advance_guarded(candidate, ports=controller, **options)
        except PortStop:
            stopped = True
        finally:
            self.adapter.restore_external(external)
        check = getattr(controller, "probe_check", None)
        if check is not None:
            check()
        return {"boundary": self.adapter.boundary(self._state), "stopped": stopped}

    def continue_(self, steps: int = 1, *, ports: Any = None) -> Receipt:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if ports is not None and not self._accepts_ports():
            raise ValueError("adapter does not accept a port controller")
        if ports is not None:
            check_ahead = getattr(ports, "check_ahead", None)
            ahead = self.adapter.ports_ahead(self._state, steps) if check_ahead else None
            if ahead is not None:
                check_ahead(ahead)  # refuses before anything runs: no receipt, no attempt
        before = self._head()
        external = clone(self.adapter.capture_external())
        try:
            candidate = self._input()
            known = dict(self._slots or {})  # descriptors of candidate's values (clone keeps them)
            options = self._advance_options()
            transitions = []
            begin_call = getattr(ports, "begin_call", None)
            if begin_call is not None:
                begin_call()
            for _ in range(steps):
                entry = self.adapter.point(candidate).to_dict()
                spec = self.adapter.transition(candidate)
                slots = {slot.name for slot in self.adapter.surface(candidate).slots}
                if set(spec.reads + spec.writes + spec.invalidates) - slots:
                    raise ValueError("transition footprint contains undeclared slots")
                changed = set(spec.writes + spec.invalidates)
                preserved = (
                    {
                        name: known[name] if name in known else describe(value)
                        for name, value in candidate.items()
                        if name not in changed
                    }
                    if spec.footprint == "declared"
                    else {}
                )
                begin = getattr(ports, "begin", None)
                if begin is not None:
                    begin()
                candidate = dict(
                    self._advance_guarded(candidate, **options)
                    if ports is None
                    else self._advance_guarded(candidate, ports=ports, **options)
                )
                known = {}
                for name, expected in preserved.items():
                    observed = describe(candidate[name]) if name in candidate else None
                    if observed != expected:
                        raise ValueError(f"native transition changed undeclared slot: {name}")
                    known[name] = observed
                self.adapter.validate_execution()
                _validate_state(self.adapter, candidate)
                self.adapter.surface(candidate).validate(candidate)
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
                                "transaction_parent": before,
                                "transition_index": len(transitions),
                                "execution": dict(self.adapter.execution),
                            }
                        ),
                    }
                )
                manifest = getattr(ports, "manifest", None)
                edits = manifest() if manifest is not None else None
                if edits:
                    transitions[-1]["port_edits"] = edits
            end_call = getattr(ports, "end_call", None)
            if end_call is not None:
                end_call()
            return self._record(
                "continue",
                before,
                state=candidate,
                owned=True,
                slots=known,
                checked=True,
                steps=steps,
                transitions=transitions,
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
                        before.fingerprint,
                        state=branch._state,
                        slots=branch._slots,
                        steps=count,
                        granularity=granularity,
                        micro_receipts=[r.to_dict() for r in branch.receipts],
                    )
            raise ValueError("macro-step limit reached before a complete boundary")
        except Exception as exc:
            self.history.append(branch.capture())
            self._activate()
            self._failed("step", before.fingerprint, exc)
            raise

    def replay(self, cut: StateCut, *, steps: int = 1) -> Session:
        branch = self.fork(cut)
        branch.continue_(steps)
        return branch

    def restore(self, cut: StateCut) -> Receipt:
        state, external, slots = self._compatible(cut)
        return self._settle(
            cut, state, external, slots, self._head(execution_checked=True, poisoned=True)
        )

    def _settle(
        self,
        cut: StateCut,
        state: dict[str, Any],
        external: Any,
        slots: dict[str, Any] | None,
        before: str,
    ) -> Receipt:
        """The rest of ``restore`` once ``cut`` is verified and compatible and ``state`` (a
        private copy of its payload, validated) is in hand: check that it seals to the cut's
        fingerprint, write the receipt (``before``: the fingerprint left) and take it over."""
        current = clone(self.adapter.capture_external())
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            restored = self._seal(
                state, cut.parent, slots, execution_checked=True, state_checked=True
            )
            if restored.fingerprint != cut.fingerprint:
                raise ValueError("restore fingerprint mismatch")
            receipt = Receipt.make(
                operation="restore",
                parent=before,
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
        self._poisoned = None
        self._slots = restored.slots
        self._external_state = external
        self._parent = cut.parent
        self.receipts.append(receipt)
        return receipt

    def commit(self, branch: Session) -> Receipt:
        parent = self._head()
        if branch.adapter is not self.adapter or branch._origin != parent or branch._aborted:
            raise ValueError("branch was not forked from the current parent")
        candidate = branch.capture(retain=False)
        state, external, slots = self._compatible(candidate)
        current = clone(self._external_state)
        try:
            if external is not None:
                self.adapter.restore_external(clone(external))
            return self._record(
                "commit",
                parent,
                state=state,
                owned=True,
                slots=slots,
                candidate=candidate.fingerprint,
            )
        except Exception as exc:
            self.adapter.restore_external(current)
            self._failed("commit", parent, exc)
            raise

    def abort(self, branch: Session, *, reason: str = "researcher discarded candidate") -> Receipt:
        """Keep candidate evidence inspectable, but remove its permission to commit."""
        if branch.adapter is not self.adapter:
            raise ValueError("candidate belongs to another resident adapter")
        parent = self._head()
        candidate = branch._head()
        branch._aborted = True
        receipt = Receipt.make(
            operation="abort",
            parent=parent,
            candidate=candidate,
            result=parent,
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

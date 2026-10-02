"""Selective TITAN Frame/Act contracts for SATURN-owned execution.

TITAN is intentionally a boundary layer here. SATURN remains the owner of
tensor state, batching, cache lifetime, and serving; this module supplies
content-addressed frame identity, named act contracts, implementation
selection, and evidence for safe replacement.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from mdb.tensor_hash import tensor_bytes_sha256

FRAME_SCHEMA = "saturn-frame-v1"
CONTENT_FINGERPRINT_POLICY = "content-sha256-v1"
RUNTIME_DESCRIPTOR_POLICY = "runtime-descriptor-v1"
PRODUCER_CONTENT_POLICY = "producer-content-sha256-v1"
PRODUCER_DESCRIPTOR_POLICY = "producer-descriptor-v1"
FINGERPRINT_ABI_KEY = "saturn:fingerprint"


class FingerprintPolicy(str, Enum):
    """How newly-written Frame payloads receive an identity.

    ``CONTENT`` is the historical, durable behavior: tensor bytes are copied
    to CPU and hashed. ``RUNTIME_DESCRIPTOR`` is an explicitly weaker hot-path
    identity over tensor storage/version/type metadata. It never claims to be
    a content hash and is marked as non-durable in the slot ABI.
    """

    CONTENT = CONTENT_FINGERPRINT_POLICY
    RUNTIME_DESCRIPTOR = RUNTIME_DESCRIPTOR_POLICY


def _normalise_fingerprint_policy(value: FingerprintPolicy | str) -> FingerprintPolicy:
    try:
        return FingerprintPolicy(value)
    except ValueError as exc:
        choices = ", ".join(policy.value for policy in FingerprintPolicy)
        raise ValueError(
            f"unsupported fingerprint policy {value!r}; expected one of {choices}"
        ) from exc


def _require_sha256(value: str, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class ProvidedFingerprint:
    """A producer-issued slot fingerprint with explicit evidence strength.

    Use :meth:`content_sha256` when the producer already computed a durable
    payload digest. Its producer-owned hash convention remains explicit in
    the slot ABI rather than being confused with TITAN's convention. Use
    :meth:`descriptor_sha256` for an execution-local producer identity. Raw
    strings are deliberately not accepted by ``Frame`` because they cannot say
    whether they identify content or merely a live allocation/version.
    """

    fingerprint: str
    policy: str
    durable: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint",
            _require_sha256(self.fingerprint, label="provided fingerprint"),
        )
        if self.policy not in {PRODUCER_CONTENT_POLICY, PRODUCER_DESCRIPTOR_POLICY}:
            raise ValueError(f"unsupported provided fingerprint policy {self.policy!r}")
        if self.policy == PRODUCER_CONTENT_POLICY and not self.durable:
            raise ValueError("content fingerprints must be declared durable")
        if self.policy == PRODUCER_DESCRIPTOR_POLICY and self.durable:
            raise ValueError("producer descriptors cannot be declared durable")

    @classmethod
    def content_sha256(cls, fingerprint: str) -> ProvidedFingerprint:
        """Declare a producer-computed durable content SHA-256."""

        return cls(
            fingerprint=fingerprint,
            policy=PRODUCER_CONTENT_POLICY,
            durable=True,
        )

    @classmethod
    def descriptor_sha256(cls, fingerprint: str) -> ProvidedFingerprint:
        """Declare a producer-computed execution-local descriptor SHA-256."""

        return cls(
            fingerprint=fingerprint,
            policy=PRODUCER_DESCRIPTOR_POLICY,
            durable=False,
        )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    raise TypeError(f"value is not JSON-like: {type(value).__name__}")


def _digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _legacy_payload_fingerprint(value: Any) -> str:
    """Reproduce the pre-v2 fallback used by older CUDA Frame receipts."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            tensor = detach().cpu().contiguous()
            raw = tensor.numpy().tobytes()
            return _digest(
                {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "bytes": hashlib.sha256(raw).hexdigest(),
                }
            )
        except Exception:  # noqa: BLE001 - descriptor fallback is intentional
            pass
    return _digest({"type": type(value).__name__, "repr": repr(value)})


def _payload_fingerprint(value: Any) -> str:
    """Fingerprint tensor bytes without making dtype/device part of the fallback.

    ``numpy()`` does not support every torch dtype (notably bfloat16 on some
    versions). The old fallback hashed ``repr(tensor)``, which made an otherwise
    identical CUDA/CPU payload produce different Frame identities. Viewing the
    contiguous tensor as bytes keeps the content identity stable across device
    rehydration; the legacy helper remains available to verify old receipts.
    """

    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            tensor = detach()
            return _digest(
                {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "bytes": tensor_bytes_sha256(tensor),
                }
            )
        except Exception:  # noqa: BLE001 - opaque tensor-like fallback is intentional
            if isinstance(value, torch.Tensor):
                raise
            pass
    return _digest({"type": type(value).__name__, "repr": repr(value)})


def _is_json_value(value: Any) -> bool:
    """Return whether ``value`` can be content-hashed without payload IO."""

    try:
        _jsonable(value)
    except TypeError:
        return False
    return True


def _runtime_descriptor_fingerprint(value: Any) -> str:
    """Identify a live payload without reading its bytes.

    Tensor descriptors bind the storage address, view geometry, and mutation
    version. This is sufficient for execution lineage while the payload is
    live, but allocator addresses may be reused and custom kernels can mutate
    storage outside PyTorch's version counter. The result is therefore always
    advertised as non-durable evidence.
    """

    descriptor: dict[str, Any] = {
        "schema": RUNTIME_DESCRIPTOR_POLICY,
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            descriptor["shape"] = [int(dimension) for dimension in shape]
        except Exception:  # noqa: BLE001 - opaque shape descriptors are optional
            descriptor["shape"] = str(shape)
    for attribute in ("dtype", "device", "layout"):
        item = getattr(value, attribute, None)
        if item is not None:
            descriptor[attribute] = str(item)
    for method_name in ("stride", "storage_offset", "data_ptr"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            item = method()
            descriptor[method_name] = (
                [int(part) for part in item]
                if isinstance(item, (tuple, list))
                else int(item)
            )
        except Exception:  # noqa: BLE001 - descriptor fields are best effort
            continue
    try:
        version = getattr(value, "_version", None)
    except RuntimeError:
        # Tensors constructed under torch.inference_mode deliberately have no
        # version counter. Their storage/view descriptor still identifies the
        # live payload; the ABI already marks this policy non-durable.
        version = None
    if version is not None:
        try:
            descriptor["version"] = int(version)
        except Exception:  # noqa: BLE001 - custom tensor-like payload
            pass
    if "data_ptr" not in descriptor:
        # Opaque model-owned values have no storage descriptor. Object identity
        # is deliberately process-local and the ABI marks that limitation.
        descriptor["object_id"] = id(value)
    return _digest(descriptor)


def _fingerprint_abi(*, policy: str, durable: bool) -> dict[str, Any]:
    if policy == CONTENT_FINGERPRINT_POLICY:
        return {}
    return {
        FINGERPRINT_ABI_KEY: {
            "policy": policy,
            "durable": durable,
        }
    }


def _slot_fingerprint(
    value: Any,
    *,
    policy: FingerprintPolicy,
    provided: ProvidedFingerprint | None,
) -> tuple[str, dict[str, Any]]:
    if provided is not None:
        if not isinstance(provided, ProvidedFingerprint):
            raise TypeError(
                "slot_fingerprints values must be ProvidedFingerprint instances"
            )
        return provided.fingerprint, _fingerprint_abi(
            policy=provided.policy,
            durable=provided.durable,
        )
    if policy is FingerprintPolicy.CONTENT:
        content_authority = getattr(value, "full_content_fingerprint", None)
        if callable(content_authority):
            fingerprint = _require_sha256(
                content_authority(),
                label="producer content fingerprint",
            )
            return fingerprint, _fingerprint_abi(
                policy=PRODUCER_CONTENT_POLICY,
                durable=True,
            )
    else:
        # KV caches and derived contexts already own incremental/recipe identities.
        # Reuse that producer identity in the hot path rather than degrading to an
        # object address; it remains explicitly non-durable at the Frame boundary.
        producer_fingerprint = getattr(value, "fingerprint", None)
        if isinstance(producer_fingerprint, str):
            try:
                producer_fingerprint = _require_sha256(
                    producer_fingerprint,
                    label="producer descriptor fingerprint",
                )
            except ValueError:
                pass
            else:
                return producer_fingerprint, _fingerprint_abi(
                    policy=PRODUCER_DESCRIPTOR_POLICY,
                    durable=False,
                )
    if policy is FingerprintPolicy.CONTENT or _is_json_value(value):
        return _payload_fingerprint(value), {}
    return _runtime_descriptor_fingerprint(value), _fingerprint_abi(
        policy=RUNTIME_DESCRIPTOR_POLICY,
        durable=False,
    )


def _slot_abi(
    abi: Mapping[str, Any],
    fingerprint_abi: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(abi)
    if FINGERPRINT_ABI_KEY in result and FINGERPRINT_ABI_KEY not in fingerprint_abi:
        # A new durable content fingerprint supersedes an inherited descriptor
        # marker supplied in a caller-owned ABI mapping.
        result.pop(FINGERPRINT_ABI_KEY)
    result.update(fingerprint_abi)
    return result


def _reuse_slot_abi(
    previous: FrameSlot,
    replacement: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(previous.abi if replacement is None else replacement)
    if FINGERPRINT_ABI_KEY in previous.abi:
        result[FINGERPRINT_ABI_KEY] = previous.abi[FINGERPRINT_ABI_KEY]
    return result


@dataclass(frozen=True, slots=True)
class FrameSlot:
    """A typed, content-addressed reference to a SATURN state value."""

    name: str
    kind: str
    handle: str
    fingerprint: str
    abi: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not all((self.name, self.kind, self.handle, self.fingerprint)):
            raise ValueError("frame slots require name, kind, handle, and fingerprint")
        object.__setattr__(self, "abi", dict(_jsonable(self.abi)))

    @property
    def fingerprint_policy(self) -> str:
        evidence = self.abi.get(FINGERPRINT_ABI_KEY)
        if isinstance(evidence, Mapping):
            return str(evidence.get("policy") or CONTENT_FINGERPRINT_POLICY)
        return CONTENT_FINGERPRINT_POLICY

    @property
    def fingerprint_is_durable(self) -> bool:
        evidence = self.abi.get(FINGERPRINT_ABI_KEY)
        if isinstance(evidence, Mapping):
            return bool(evidence.get("durable", False))
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "handle": self.handle,
            "fingerprint": self.fingerprint,
            "abi": dict(self.abi),
        }


@dataclass(frozen=True, slots=True)
class Frame:
    """Immutable partial state; SATURN retains ownership of payload storage."""

    slots: Mapping[str, FrameSlot]
    values: Mapping[str, Any]
    parent_frame_id: str | None = None
    act: str = "root:v1"
    implementation: str = "saturn"
    program_fingerprint: str = "saturn"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint_policy: FingerprintPolicy | str = FingerprintPolicy.CONTENT
    frame_id: str = field(init=False)

    def __post_init__(self) -> None:
        slots = dict(self.slots)
        values = dict(self.values)
        if set(slots) != set(values):
            raise ValueError("frame slots and values must have identical names")
        object.__setattr__(self, "slots", slots)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", dict(_jsonable(self.metadata)))
        policy = _normalise_fingerprint_policy(self.fingerprint_policy)
        if policy is FingerprintPolicy.CONTENT and any(
            slot.abi.get(FINGERPRINT_ABI_KEY, {}).get("durable") is False
            for slot in slots.values()
            if isinstance(slot.abi.get(FINGERPRINT_ABI_KEY), Mapping)
        ):
            # Descriptor-preserving hydration constructs Frame directly. Infer
            # its write policy from its slots so subsequent acts stay on the
            # explicit fast path rather than silently returning to full hashes.
            policy = FingerprintPolicy.RUNTIME_DESCRIPTOR
        object.__setattr__(self, "fingerprint_policy", policy)
        identity = {
            "schema": FRAME_SCHEMA,
            "slots": {name: slot.to_dict() for name, slot in sorted(slots.items())},
            "parent_frame_id": self.parent_frame_id,
            "act": self.act,
            "implementation": self.implementation,
            "program_fingerprint": self.program_fingerprint,
            "metadata": self.metadata,
        }
        object.__setattr__(self, "frame_id", _digest(identity))

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
        *,
        kinds: Mapping[str, str] | None = None,
        abis: Mapping[str, Mapping[str, Any]] | None = None,
        handles: Mapping[str, str] | None = None,
        slot_fingerprints: Mapping[str, ProvidedFingerprint] | None = None,
        fingerprint_policy: FingerprintPolicy | str = FingerprintPolicy.CONTENT,
        parent_frame_id: str | None = None,
        act: str = "root:v1",
        implementation: str = "saturn",
        program_fingerprint: str = "saturn",
        metadata: Mapping[str, Any] | None = None,
    ) -> Frame:
        kinds = kinds or {}
        abis = abis or {}
        handles = handles or {}
        slot_fingerprints = slot_fingerprints or {}
        unknown_fingerprints = set(slot_fingerprints) - set(values)
        if unknown_fingerprints:
            raise ValueError(
                "slot_fingerprints declare unknown values: "
                f"{sorted(unknown_fingerprints)}"
            )
        policy = _normalise_fingerprint_policy(fingerprint_policy)
        slots: dict[str, FrameSlot] = {}
        for name, value in values.items():
            fingerprint, fingerprint_abi = _slot_fingerprint(
                value,
                policy=policy,
                provided=slot_fingerprints.get(name),
            )
            slots[name] = FrameSlot(
                name=name,
                kind=kinds.get(name, "opaque"),
                handle=handles.get(name, f"saturn://{name}"),
                fingerprint=fingerprint,
                abi=_slot_abi(abis.get(name, {}), fingerprint_abi),
            )
        return cls(
            slots=slots,
            values=dict(values),
            parent_frame_id=parent_frame_id,
            act=act,
            implementation=implementation,
            program_fingerprint=program_fingerprint,
            metadata=metadata or {},
            fingerprint_policy=policy,
        )

    @property
    def fingerprint(self) -> str:
        return self.frame_id

    def to_dict(self) -> dict[str, Any]:
        """Return metadata suitable for a debugger receipt.

        Frame values can be CUDA tensors or model-owned objects, so they are
        intentionally excluded.  Slot fingerprints and ABI declarations are
        enough to identify the state transition without moving tensor payloads
        through the control plane.
        """

        return {
            "schema": FRAME_SCHEMA,
            "frame_id": self.frame_id,
            "parent_frame_id": self.parent_frame_id,
            "act": self.act,
            "implementation": self.implementation,
            "program_fingerprint": self.program_fingerprint,
            "slots": {
                name: slot.to_dict() for name, slot in sorted(self.slots.items())
            },
            "metadata": dict(self.metadata),
        }

    def require(self, names: tuple[str, ...] | list[str]) -> None:
        missing = [name for name in names if name not in self.slots]
        if missing:
            raise ValueError(f"frame is missing required slots: {missing}")

    def content_fingerprints(
        self,
        names: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, str]:
        """Materialize durable content fingerprints for selected live values.

        This explicit evidence operation pays the historical tensor-to-CPU
        hashing cost even when the execution Frame uses runtime descriptors.
        It does not mutate the Frame or change its lineage identity.
        """

        selected = tuple(self.values) if names is None else tuple(names)
        self.require(selected)
        return {name: _payload_fingerprint(self.values[name]) for name in selected}

    def with_updates(
        self,
        updates: Mapping[str, Any],
        *,
        act: str,
        implementation: str,
        invalidates: tuple[str, ...] = (),
        kinds: Mapping[str, str] | None = None,
        abis: Mapping[str, Mapping[str, Any]] | None = None,
        handles: Mapping[str, str] | None = None,
        slot_fingerprints: Mapping[str, ProvidedFingerprint] | None = None,
        fingerprint_policy: FingerprintPolicy | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Frame:
        invalidated = set(invalidates)
        values = {name: value for name, value in self.values.items() if name not in invalidated}
        values.update(updates)
        kinds = kinds or {}
        abis = abis or {}
        handles = handles or {}
        slot_fingerprints = slot_fingerprints or {}
        unknown_fingerprints = set(slot_fingerprints) - set(updates)
        if unknown_fingerprints:
            raise ValueError(
                "with_updates slot_fingerprints require matching writes: "
                f"{sorted(unknown_fingerprints)}"
            )
        policy = (
            self.fingerprint_policy
            if fingerprint_policy is None
            else _normalise_fingerprint_policy(fingerprint_policy)
        )
        slots: dict[str, FrameSlot] = {}
        for name, value in values.items():
            previous = self.slots.get(name)
            if name not in updates and previous is not None:
                if name not in kinds and name not in abis and name not in handles:
                    # The payload and its contract are unchanged. Carry the
                    # immutable descriptor itself instead of re-reading bytes.
                    slots[name] = previous
                else:
                    slots[name] = FrameSlot(
                        name=name,
                        kind=kinds.get(name, previous.kind),
                        handle=handles.get(name, previous.handle),
                        fingerprint=previous.fingerprint,
                        abi=_reuse_slot_abi(previous, abis.get(name)),
                    )
                continue
            fingerprint, fingerprint_abi = _slot_fingerprint(
                value,
                policy=policy,
                provided=slot_fingerprints.get(name),
            )
            slots[name] = FrameSlot(
                name=name,
                kind=kinds.get(name, previous.kind if previous is not None else "opaque"),
                handle=handles.get(
                    name,
                    previous.handle if previous is not None else f"saturn://{name}",
                ),
                fingerprint=fingerprint,
                abi=_slot_abi(
                    abis.get(name, previous.abi if previous is not None else {}),
                    fingerprint_abi,
                ),
            )
        return Frame(
            slots=slots,
            values=values,
            parent_frame_id=self.frame_id,
            act=act,
            implementation=implementation,
            program_fingerprint=self.program_fingerprint,
            metadata={**self.metadata, **(metadata or {})},
            fingerprint_policy=policy,
        )


@dataclass(frozen=True, slots=True)
class ActSpec:
    """Versioned read/write and numerical contract for one model operation."""

    name: str
    abi_version: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    numerical_contract: str = "scalar-authority"
    tolerance: Mapping[str, float] = field(default_factory=dict)
    write_kinds: Mapping[str, str] = field(default_factory=dict)
    write_abis: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.abi_version}"

    def __post_init__(self) -> None:
        for name in ("reads", "writes", "invalidates"):
            values = tuple(getattr(self, name))
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique slot names")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "tolerance", dict(_jsonable(self.tolerance)))
        writes = set(self.writes)
        for field_name in ("write_kinds", "write_abis"):
            values = dict(getattr(self, field_name))
            unknown = set(values) - writes
            if unknown:
                raise ValueError(
                    f"{field_name} declares slots that are not in writes: {sorted(unknown)}"
                )
            object.__setattr__(self, field_name, dict(_jsonable(values)))

    def to_dict(self) -> dict[str, Any]:
        """Return the normalized, serializable operation contract."""

        return {
            "schema": "saturn-act-spec-v1",
            "identity": self.identity,
            "name": self.name,
            "abi_version": self.abi_version,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "numerical_contract": self.numerical_contract,
            "tolerance": dict(self.tolerance),
            "write_kinds": dict(self.write_kinds),
            "write_abis": dict(self.write_abis),
        }

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActEvidence:
    act: str
    implementation: str
    input_frame_id: str
    output_frame_id: str
    elapsed_ms: float
    numerical_contract: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "act": self.act,
            "implementation": self.implementation,
            "input_frame_id": self.input_frame_id,
            "output_frame_id": self.output_frame_id,
            "elapsed_ms": self.elapsed_ms,
            "numerical_contract": self.numerical_contract,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class OpaqueActEvidence:
    """Observed IO for an operation that has not earned an ``ActSpec`` yet."""

    act: str
    implementation: str
    input_frame_id: str
    output_frame_id: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    elapsed_ms: float | None = None
    status: str = "unclassified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reads", tuple(self.reads))
        object.__setattr__(self, "writes", tuple(self.writes))
        object.__setattr__(self, "metadata", dict(_jsonable(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "act": self.act,
            "implementation": self.implementation,
            "input_frame_id": self.input_frame_id,
            "output_frame_id": self.output_frame_id,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "elapsed_ms": self.elapsed_ms,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


def observe_unknown_act(
    frame: Frame,
    *,
    act: str,
    implementation: str,
    outputs: Mapping[str, Any],
    reads: tuple[str, ...] | list[str] = (),
    invalidates: tuple[str, ...] | list[str] = (),
    elapsed_ms: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Frame, OpaqueActEvidence]:
    """Record an unknown operation without making it executable.

    This is the discovery seam between a dynamic tracer and a promoted
    ``ActSpec``.  Reads are checked when supplied, writes are inferred from the
    observed output mapping, and the resulting evidence is explicitly marked
    ``unclassified``.  Callers must later promote the observation to an
    ``ActSpec`` before it can be selected or replayed as an implementation.
    """

    if not act or not implementation:
        raise ValueError("opaque observations require act and implementation names")
    read_names = tuple(reads)
    frame.require(read_names)
    output = frame.with_updates(
        outputs,
        act=f"{act}:observed",
        implementation=implementation,
        invalidates=tuple(invalidates),
        metadata={"unclassified": True, **(metadata or {})},
    )
    evidence = OpaqueActEvidence(
        act=act,
        implementation=implementation,
        input_frame_id=frame.frame_id,
        output_frame_id=output.frame_id,
        reads=read_names,
        writes=tuple(outputs),
        elapsed_ms=elapsed_ms,
        metadata=metadata or {},
    )
    return output, evidence


ActFunction = Callable[[Mapping[str, Any]], tuple[Mapping[str, Any], Mapping[str, Any]]]


class ActRegistry:
    """Registry that fail-closes undeclared reads/writes and bad evidence."""

    def __init__(self) -> None:
        self._specs: dict[str, ActSpec] = {}
        self._implementations: dict[tuple[str, str], ActFunction] = {}

    def register(self, spec: ActSpec, implementation: str, function: ActFunction) -> None:
        key = (spec.identity, implementation)
        if key in self._implementations:
            raise ValueError(f"implementation already registered: {spec.identity}/{implementation}")
        existing = self._specs.get(spec.identity)
        if existing is not None and existing != spec:
            raise ValueError(f"conflicting ActSpec declaration for {spec.identity}")
        self._specs[spec.identity] = spec
        self._implementations[key] = function

    def spec(self, identity: str) -> ActSpec:
        try:
            return self._specs[identity]
        except KeyError as exc:
            raise ValueError(f"unknown act: {identity}") from exc

    def run(
        self,
        identity: str,
        implementation: str,
        frame: Frame,
    ) -> tuple[Frame, ActEvidence]:
        spec = self.spec(identity)
        frame.require(spec.reads)
        try:
            function = self._implementations[(identity, implementation)]
        except KeyError as exc:
            raise ValueError(f"unknown implementation: {identity}/{implementation}") from exc
        started = time.perf_counter()
        # An Act is a capability boundary, not only a documentation record.
        # Hand the implementation exactly the slots declared by ``reads`` so
        # an otherwise valid Frame cannot leak adjacent teacher/debug state to
        # a student implementation through the backing mapping.
        declared_inputs = {name: frame.values[name] for name in spec.reads}
        outputs, metrics = function(declared_inputs)
        output_names = set(outputs)
        expected_names = set(spec.writes)
        if output_names != expected_names:
            raise ValueError(
                f"{identity}/{implementation} wrote {sorted(output_names)}, "
                f"expected {sorted(expected_names)}"
            )
        if metrics.get("numerical_contract_ok") is False:
            raise ValueError(f"{identity}/{implementation} reported a failed numerical contract")
        if "cosine" in metrics and "min_cosine" in spec.tolerance:
            if float(metrics["cosine"]) < float(spec.tolerance["min_cosine"]):
                raise ValueError(f"{identity}/{implementation} missed the cosine tolerance")
        output = frame.with_updates(
            outputs,
            act=identity,
            implementation=implementation,
            invalidates=spec.invalidates,
            kinds=spec.write_kinds,
            abis=spec.write_abis,
        )
        evidence = ActEvidence(
            act=identity,
            implementation=implementation,
            input_frame_id=frame.frame_id,
            output_frame_id=output.frame_id,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            numerical_contract=spec.numerical_contract,
            metrics=dict(_jsonable(metrics)),
        )
        return output, evidence


def trajectory_step_spec() -> ActSpec:
    """The first SATURN/TITAN shared boundary for learned transitions."""

    return ActSpec(
        name="trajectory_step",
        abi_version="v1",
        reads=("latent", "conditioning", "trajectory"),
        writes=("latent_next",),
        numerical_contract="teacher-transition-cosine",
    )

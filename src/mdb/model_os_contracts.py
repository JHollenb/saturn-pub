"""Canonical, tensor-free contracts for the Model OS control plane.

The contracts in this module deliberately describe model identity, state
ownership, programs, and native consumers without carrying model tensors.  A
runtime may attach tensors to an implementation-private state cut, but the
portable boundary is JSON-safe and content-addressed.  This keeps package
identity, replay, and evidence stable across mrun leases and backend adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from typing import Any


MODEL_OS_ABI_SCHEMA = "saturn-model-os-abi-v1"
MODEL_OS_STATE_SCHEMA = "saturn-model-os-state-schema-v1"
MODEL_OS_SURFACE_SCHEMA = "saturn-model-os-state-surface-v1"
MODEL_OS_PROGRAM_SCHEMA = "saturn-model-os-program-v1"
MODEL_OS_CONTINUATION_SCHEMA = "saturn-model-os-continuation-v1"
MODEL_OS_PACKAGE_SCHEMA = "saturn-model-os-package-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LIFETIMES = {"token", "frame", "branch", "session", "durable"}
_REPLAY_CONTRACTS = {"exact", "bounded", "candidate"}
_PROGRAM_OPS = {
    "Read",
    "Select",
    "Bind",
    "Transform",
    "Write",
    "Route",
    "Verify",
    "Repair",
    "Continue",
    "Emit",
    "Checkpoint",
    "Fork",
    "Commit",
    "Abort",
}


class ModelOSContractError(ValueError):
    """Raised when a portable Model OS contract is malformed or unsafe."""


def _json_safe(value: Any, *, path: str = "value") -> Any:
    """Return a canonical JSON-compatible copy, rejecting tensor payloads."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelOSContractError(f"{path} contains a non-finite float")
        return value

    # A tensor, array, or lazy device value must never leak into the portable
    # ABI.  Checking these protocols is intentionally conservative: callers
    # can provide a digest or a named artifact instead.
    if hasattr(value, "detach") or (
        hasattr(value, "shape") and hasattr(value, "dtype")
    ):
        raise ModelOSContractError(
            f"{path} contains tensor-like data; provide an artifact identity"
        )

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ModelOSContractError(f"{path} has a non-string key {key!r}")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result

    if isinstance(value, (tuple, list)):
        return [_json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]

    if isinstance(value, (set, frozenset)):
        normalized = [_json_safe(item, path=f"{path}[]") for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True))

    raise ModelOSContractError(
        f"{path} has unsupported type {type(value).__name__}; use JSON-safe metadata"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a contract value deterministically."""

    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def content_digest(value: Any) -> str:
    """Return the SHA-256 digest used by portable contract identities."""

    return sha256(canonical_json_bytes(value)).hexdigest()


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_safe(body)
    assert isinstance(normalized, dict)
    normalized["content_sha256"] = content_digest(normalized)
    return normalized


def _unseal(payload: Mapping[str, Any], *, schema: str) -> dict[str, Any]:
    normalized = _json_safe(payload)
    if not isinstance(normalized, dict):
        raise ModelOSContractError("contract payload must be an object")
    if normalized.get("schema") != schema:
        raise ModelOSContractError(
            f"expected schema {schema!r}, got {normalized.get('schema')!r}"
        )
    supplied = normalized.pop("content_sha256", None)
    if not isinstance(supplied, str) or not _SHA256_RE.fullmatch(supplied):
        raise ModelOSContractError("contract is missing a valid content_sha256")
    expected = content_digest(normalized)
    if supplied != expected:
        raise ModelOSContractError(
            f"contract digest mismatch: expected {expected}, got {supplied}"
        )
    return normalized


def _text(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelOSContractError(f"{name} must be a non-empty string")
    return value.strip()


def _texts(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ModelOSContractError(f"{name} must be a sequence of strings")
    result = tuple(_text(value, name=name) for value in values)
    return tuple(dict.fromkeys(result))


def _shape(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ModelOSContractError(f"{name} must be a sequence of dimensions")
    result = tuple(values)
    if any(not isinstance(value, int) or value < 0 for value in result):
        raise ModelOSContractError(f"{name} must contain non-negative integers")
    return result


@dataclass(frozen=True, slots=True)
class ModelABI:
    """Identity and native-consumer closure for one resident model."""

    model_id: str
    revision: str
    family: str
    backend: str
    state_schema: str
    tokenizer_identity: str = "unspecified"
    kv_cache_schema: str = "unspecified"
    device: str = "unspecified"
    dtype: str = "unspecified"
    numerical_contract: str = "bounded"
    checkpoint_sha256: str | None = None
    graph_identity: str = "unspecified"
    native_consumers: tuple[str, ...] = ()
    runtime_fingerprint: str = "unspecified"
    source_fingerprint: str = "unspecified"
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "revision",
            "family",
            "backend",
            "state_schema",
            "tokenizer_identity",
            "kv_cache_schema",
            "device",
            "dtype",
            "numerical_contract",
            "graph_identity",
            "runtime_fingerprint",
            "source_fingerprint",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        if self.checkpoint_sha256 is not None:
            checkpoint = _text(self.checkpoint_sha256, name="checkpoint_sha256")
            if not _SHA256_RE.fullmatch(checkpoint):
                raise ModelOSContractError("checkpoint_sha256 must be a lowercase SHA-256")
            object.__setattr__(self, "checkpoint_sha256", checkpoint)
        object.__setattr__(
            self,
            "native_consumers",
            _texts(self.native_consumers, name="native_consumers"),
        )
        object.__setattr__(self, "capabilities", _texts(self.capabilities, name="capabilities"))

    @property
    def identity(self) -> str:
        return f"{self.model_id}@{self.revision}"

    @classmethod
    def from_runtime_abi(
        cls,
        runtime_abi: Mapping[str, Any],
        *,
        model_id: str,
        revision: str,
        family: str,
        backend: str,
        tokenizer_identity: str,
        device: str = "unspecified",
        native_consumers: Sequence[str] = (),
        capabilities: Sequence[str] = (),
        source_fingerprint: str = "unspecified",
        state_schema_fingerprint: str | None = None,
    ) -> "ModelABI":
        """Adapt an mstack/mrun-style runtime ABI without importing either stack.

        The adapter keeps the Model OS identity canonical while allowing the
        physical runtime to retain its own ABI schema and detailed fields.
        """

        normalized = _json_safe(runtime_abi, path="runtime_abi")
        if not isinstance(normalized, dict):
            raise ModelOSContractError("runtime_abi must be an object")
        base_hash = normalized.get("base_content_hash")
        checkpoint = (
            str(base_hash)
            if isinstance(base_hash, str) and _SHA256_RE.fullmatch(base_hash)
            else None
        )
        return cls(
            model_id=model_id,
            revision=revision,
            family=family,
            backend=backend,
            state_schema=str(
                state_schema_fingerprint
                or normalized.get("state_schema")
                or "unspecified"
            ),
            tokenizer_identity=tokenizer_identity,
            kv_cache_schema=str(normalized.get("kv_cache_schema") or "unspecified"),
            device=device,
            dtype=str(normalized.get("deployment_dtype") or "unspecified"),
            numerical_contract=str(
                normalized.get("numerical_contract")
                or normalized.get("readout_and_tie_rules")
                or "bounded"
            ),
            checkpoint_sha256=checkpoint,
            graph_identity=str(
                normalized.get("graph_hash")
                or normalized.get("config_hash")
                or "unspecified"
            ),
            native_consumers=tuple(native_consumers),
            runtime_fingerprint=content_digest(normalized),
            source_fingerprint=source_fingerprint,
            capabilities=tuple(capabilities),
        )

    def to_payload(self) -> dict[str, Any]:
        return _sealed(
            {
                "schema": MODEL_OS_ABI_SCHEMA,
                "model_id": self.model_id,
                "revision": self.revision,
                "family": self.family,
                "backend": self.backend,
                "state_schema": self.state_schema,
                "tokenizer_identity": self.tokenizer_identity,
                "kv_cache_schema": self.kv_cache_schema,
                "device": self.device,
                "dtype": self.dtype,
                "numerical_contract": self.numerical_contract,
                "checkpoint_sha256": self.checkpoint_sha256,
                "graph_identity": self.graph_identity,
                "native_consumers": list(self.native_consumers),
                "runtime_fingerprint": self.runtime_fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "capabilities": list(self.capabilities),
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelABI":
        body = _unseal(payload, schema=MODEL_OS_ABI_SCHEMA)
        return cls(
            model_id=body["model_id"],
            revision=body["revision"],
            family=body["family"],
            backend=body["backend"],
            state_schema=body["state_schema"],
            tokenizer_identity=body["tokenizer_identity"],
            kv_cache_schema=body["kv_cache_schema"],
            device=body["device"],
            dtype=body["dtype"],
            numerical_contract=body["numerical_contract"],
            checkpoint_sha256=body["checkpoint_sha256"],
            graph_identity=body["graph_identity"],
            native_consumers=tuple(body["native_consumers"]),
            runtime_fingerprint=body["runtime_fingerprint"],
            source_fingerprint=body["source_fingerprint"],
            capabilities=tuple(body["capabilities"]),
        )


@dataclass(frozen=True, slots=True)
class StateSlot:
    """One named state location with explicit ownership and lifetime."""

    address: str
    value_type: str
    owner: str
    shape: tuple[int, ...] = ()
    dtype: str = "json"
    read_scopes: tuple[str, ...] = ()
    write_scopes: tuple[str, ...] = ()
    lifetime: str = "frame"
    branchable: bool = True
    persistable: bool = False
    invalidates: tuple[str, ...] = ()
    version: int = 0
    freshness: str = "unknown"
    lineage: str = "parent"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "address", _text(self.address, name="address"))
        object.__setattr__(self, "value_type", _text(self.value_type, name="value_type"))
        object.__setattr__(self, "owner", _text(self.owner, name="owner"))
        object.__setattr__(self, "shape", _shape(self.shape, name="shape"))
        object.__setattr__(self, "dtype", _text(self.dtype, name="dtype"))
        object.__setattr__(self, "read_scopes", _texts(self.read_scopes, name="read_scopes"))
        object.__setattr__(self, "write_scopes", _texts(self.write_scopes, name="write_scopes"))
        if self.lifetime not in _LIFETIMES:
            raise ModelOSContractError(
                f"lifetime must be one of {sorted(_LIFETIMES)}, got {self.lifetime!r}"
            )
        object.__setattr__(self, "invalidates", _texts(self.invalidates, name="invalidates"))
        if not isinstance(self.version, int) or self.version < 0:
            raise ModelOSContractError("state slot version must be a non-negative integer")
        object.__setattr__(self, "freshness", _text(self.freshness, name="freshness"))
        object.__setattr__(self, "lineage", _text(self.lineage, name="lineage"))
        metadata = _json_safe(self.metadata, path=f"slot:{self.address}.metadata")
        if not isinstance(metadata, dict):
            raise ModelOSContractError("slot metadata must be an object")
        object.__setattr__(self, "metadata", metadata)

    def to_payload(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "value_type": self.value_type,
            "owner": self.owner,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "read_scopes": list(self.read_scopes),
            "write_scopes": list(self.write_scopes),
            "lifetime": self.lifetime,
            "branchable": self.branchable,
            "persistable": self.persistable,
            "invalidates": list(self.invalidates),
            "version": self.version,
            "freshness": self.freshness,
            "lineage": self.lineage,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StateSlot":
        return cls(
            address=payload["address"],
            value_type=payload["value_type"],
            owner=payload["owner"],
            shape=tuple(payload.get("shape", ())),
            dtype=payload.get("dtype", "json"),
            read_scopes=tuple(payload.get("read_scopes", ())),
            write_scopes=tuple(payload.get("write_scopes", ())),
            lifetime=payload.get("lifetime", "frame"),
            branchable=bool(payload.get("branchable", True)),
            persistable=bool(payload.get("persistable", False)),
            invalidates=tuple(payload.get("invalidates", ())),
            version=payload.get("version", 0),
            freshness=payload.get("freshness", "unknown"),
            lineage=payload.get("lineage", "parent"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class StateSchema:
    """The portable schema of state that a Model OS program may address."""

    name: str
    version: int
    slots: tuple[StateSlot, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, name="state schema name"))
        if not isinstance(self.version, int) or self.version < 1:
            raise ModelOSContractError("state schema version must be a positive integer")
        slots = tuple(self.slots)
        addresses = [slot.address for slot in slots]
        if len(addresses) != len(set(addresses)):
            raise ModelOSContractError("state schema addresses must be unique")
        object.__setattr__(self, "slots", slots)

    @property
    def addresses(self) -> frozenset[str]:
        return frozenset(slot.address for slot in self.slots)

    def slot(self, address: str) -> StateSlot:
        for slot in self.slots:
            if slot.address == address:
                return slot
        raise ModelOSContractError(f"unknown state address {address!r}")

    def to_payload(self) -> dict[str, Any]:
        return _sealed(
            {
                "schema": MODEL_OS_STATE_SCHEMA,
                "name": self.name,
                "version": self.version,
                "slots": [slot.to_payload() for slot in self.slots],
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StateSchema":
        body = _unseal(payload, schema=MODEL_OS_STATE_SCHEMA)
        return cls(
            name=body["name"],
            version=body["version"],
            slots=tuple(StateSlot.from_payload(item) for item in body["slots"]),
        )


@dataclass(frozen=True, slots=True)
class StateSurface:
    """The branch and ownership view of a state schema exposed to a process."""

    surface_id: str
    state_schema: str
    addresses: tuple[str, ...]
    owner: str = "model-os"
    branch_model: str = "copy-on-write"
    persistence_model: str = "content-addressed"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface_id", _text(self.surface_id, name="surface_id"))
        object.__setattr__(self, "state_schema", _text(self.state_schema, name="state_schema"))
        object.__setattr__(self, "owner", _text(self.owner, name="surface owner"))
        object.__setattr__(self, "branch_model", _text(self.branch_model, name="branch_model"))
        object.__setattr__(
            self,
            "persistence_model",
            _text(self.persistence_model, name="persistence_model"),
        )
        addresses = _texts(self.addresses, name="surface addresses")
        if not addresses:
            raise ModelOSContractError("state surface must expose at least one address")
        object.__setattr__(self, "addresses", addresses)
        metadata = _json_safe(self.metadata, path=f"surface:{self.surface_id}.metadata")
        if not isinstance(metadata, dict):
            raise ModelOSContractError("state surface metadata must be an object")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_schema(
        cls,
        schema: StateSchema,
        *,
        surface_id: str = "default-state-surface",
        owner: str = "model-os",
    ) -> "StateSurface":
        return cls(
            surface_id=surface_id,
            state_schema=schema.fingerprint,
            addresses=tuple(slot.address for slot in schema.slots),
            owner=owner,
        )

    def to_payload(self) -> dict[str, Any]:
        return _sealed(
            {
                "schema": MODEL_OS_SURFACE_SCHEMA,
                "surface_id": self.surface_id,
                "state_schema": self.state_schema,
                "addresses": list(self.addresses),
                "owner": self.owner,
                "branch_model": self.branch_model,
                "persistence_model": self.persistence_model,
                "metadata": dict(self.metadata),
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "StateSurface":
        body = _unseal(payload, schema=MODEL_OS_SURFACE_SCHEMA)
        return cls(
            surface_id=body["surface_id"],
            state_schema=body["state_schema"],
            addresses=tuple(body["addresses"]),
            owner=body.get("owner", "model-os"),
            branch_model=body.get("branch_model", "copy-on-write"),
            persistence_model=body.get("persistence_model", "content-addressed"),
            metadata=body.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ProgramInstruction:
    """One declarative operation in a portable model program."""

    instruction_id: str
    op: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    native_consumer: str | None = None
    continuation: str = "same-step"
    authority: str = "model-os"
    dose_limit: float = 1.0
    replay_contract: str = "exact"
    arguments: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instruction_id",
            _text(self.instruction_id, name="instruction_id"),
        )
        if self.op not in _PROGRAM_OPS:
            raise ModelOSContractError(f"unsupported program operation {self.op!r}")
        object.__setattr__(self, "reads", _texts(self.reads, name="instruction reads"))
        object.__setattr__(self, "writes", _texts(self.writes, name="instruction writes"))
        object.__setattr__(self, "inputs", _texts(self.inputs, name="instruction inputs"))
        object.__setattr__(self, "outputs", _texts(self.outputs, name="instruction outputs"))
        if self.native_consumer is not None:
            object.__setattr__(
                self,
                "native_consumer",
                _text(self.native_consumer, name="native_consumer"),
            )
        object.__setattr__(self, "continuation", _text(self.continuation, name="continuation"))
        object.__setattr__(self, "authority", _text(self.authority, name="authority"))
        if not isinstance(self.dose_limit, (int, float)) or not math.isfinite(self.dose_limit):
            raise ModelOSContractError("dose_limit must be finite")
        if self.dose_limit < 0:
            raise ModelOSContractError("dose_limit must be non-negative")
        if self.replay_contract not in _REPLAY_CONTRACTS:
            raise ModelOSContractError(
                f"replay_contract must be one of {sorted(_REPLAY_CONTRACTS)}"
            )
        for name, value in (("arguments", self.arguments), ("metadata", self.metadata)):
            normalized = _json_safe(value, path=f"instruction:{self.instruction_id}.{name}")
            if not isinstance(normalized, dict):
                raise ModelOSContractError(f"instruction {name} must be an object")
            object.__setattr__(self, name, normalized)

    def to_payload(self) -> dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "op": self.op,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "native_consumer": self.native_consumer,
            "continuation": self.continuation,
            "authority": self.authority,
            "dose_limit": self.dose_limit,
            "replay_contract": self.replay_contract,
            "arguments": dict(self.arguments),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProgramInstruction":
        return cls(
            instruction_id=payload["instruction_id"],
            op=payload["op"],
            reads=tuple(payload.get("reads", ())),
            writes=tuple(payload.get("writes", ())),
            inputs=tuple(payload.get("inputs", ())),
            outputs=tuple(payload.get("outputs", ())),
            native_consumer=payload.get("native_consumer"),
            continuation=payload.get("continuation", "same-step"),
            authority=payload.get("authority", "model-os"),
            dose_limit=payload.get("dose_limit", 1.0),
            replay_contract=payload.get("replay_contract", "exact"),
            arguments=payload.get("arguments", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ModelProgram:
    """A typed, portable sequence of state operations and consumer calls."""

    program_id: str
    version: int
    instructions: tuple[ProgramInstruction, ...]
    portable: bool = True
    source_symbols: tuple[str, ...] = ()
    claim_level: str = "exploratory"
    terminal_claim: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "program_id", _text(self.program_id, name="program_id"))
        if not isinstance(self.version, int) or self.version < 1:
            raise ModelOSContractError("program version must be a positive integer")
        instructions = tuple(self.instructions)
        ids = [instruction.instruction_id for instruction in instructions]
        if len(ids) != len(set(ids)):
            raise ModelOSContractError("program instruction ids must be unique")
        if not instructions:
            raise ModelOSContractError("program must contain at least one instruction")
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(
            self,
            "source_symbols",
            _texts(self.source_symbols, name="source_symbols"),
        )
        object.__setattr__(self, "claim_level", _text(self.claim_level, name="claim_level"))

    @property
    def reads(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                address for instruction in self.instructions for address in instruction.reads
            )
        )

    @property
    def writes(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                address for instruction in self.instructions for address in instruction.writes
            )
        )

    @property
    def native_consumers(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                instruction.native_consumer
                for instruction in self.instructions
                if instruction.native_consumer is not None
            )
        )

    def to_payload(self) -> dict[str, Any]:
        return _sealed(
            {
                "schema": MODEL_OS_PROGRAM_SCHEMA,
                "program_id": self.program_id,
                "version": self.version,
                "instructions": [instruction.to_payload() for instruction in self.instructions],
                "portable": self.portable,
                "source_symbols": list(self.source_symbols),
                "claim_level": self.claim_level,
                "terminal_claim": self.terminal_claim,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelProgram":
        body = _unseal(payload, schema=MODEL_OS_PROGRAM_SCHEMA)
        return cls(
            program_id=body["program_id"],
            version=body["version"],
            instructions=tuple(
                ProgramInstruction.from_payload(item) for item in body["instructions"]
            ),
            portable=bool(body.get("portable", True)),
            source_symbols=tuple(body.get("source_symbols", ())),
            claim_level=body.get("claim_level", "exploratory"),
            terminal_claim=bool(body.get("terminal_claim", False)),
        )


@dataclass(frozen=True, slots=True)
class ConsumerContract:
    """The declared downstream consumer that closes a program step."""

    consumer_id: str
    reads: tuple[str, ...] = ()
    output_kind: str = "observation"
    continuation_tokens: int = 0
    replay_contract: str = "exact"
    native: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, name="consumer_id"))
        object.__setattr__(self, "reads", _texts(self.reads, name="consumer reads"))
        object.__setattr__(self, "output_kind", _text(self.output_kind, name="output_kind"))
        if not isinstance(self.continuation_tokens, int) or self.continuation_tokens < 0:
            raise ModelOSContractError("continuation_tokens must be a non-negative integer")
        if self.replay_contract not in _REPLAY_CONTRACTS:
            raise ModelOSContractError(
                f"replay_contract must be one of {sorted(_REPLAY_CONTRACTS)}"
            )
        metadata = _json_safe(self.metadata, path=f"consumer:{self.consumer_id}.metadata")
        if not isinstance(metadata, dict):
            raise ModelOSContractError("consumer metadata must be an object")
        object.__setattr__(self, "metadata", metadata)

    def to_payload(self) -> dict[str, Any]:
        return {
            "consumer_id": self.consumer_id,
            "reads": list(self.reads),
            "output_kind": self.output_kind,
            "continuation_tokens": self.continuation_tokens,
            "replay_contract": self.replay_contract,
            "native": self.native,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ConsumerContract":
        return cls(
            consumer_id=payload["consumer_id"],
            reads=tuple(payload.get("reads", ())),
            output_kind=payload.get("output_kind", "observation"),
            continuation_tokens=payload.get("continuation_tokens", 0),
            replay_contract=payload.get("replay_contract", "exact"),
            native=bool(payload.get("native", True)),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ContinuationContract:
    """The bounded future that a native consumer promises to expose."""

    continuation_id: str
    consumer_id: str
    input_state: tuple[str, ...] = ()
    output_state: tuple[str, ...] = ()
    horizon: int = 1
    replay_contract: str = "exact"
    exact: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuation_id",
            _text(self.continuation_id, name="continuation_id"),
        )
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, name="consumer_id"))
        object.__setattr__(
            self,
            "input_state",
            _texts(self.input_state, name="continuation input_state"),
        )
        object.__setattr__(
            self,
            "output_state",
            _texts(self.output_state, name="continuation output_state"),
        )
        if not isinstance(self.horizon, int) or self.horizon < 0:
            raise ModelOSContractError("continuation horizon must be non-negative")
        if self.replay_contract not in _REPLAY_CONTRACTS:
            raise ModelOSContractError(
                f"replay_contract must be one of {sorted(_REPLAY_CONTRACTS)}"
            )
        metadata = _json_safe(
            self.metadata,
            path=f"continuation:{self.continuation_id}.metadata",
        )
        if not isinstance(metadata, dict):
            raise ModelOSContractError("continuation metadata must be an object")
        object.__setattr__(self, "metadata", metadata)

    def to_payload(self) -> dict[str, Any]:
        return _sealed(
            {
                "schema": MODEL_OS_CONTINUATION_SCHEMA,
                "continuation_id": self.continuation_id,
                "consumer_id": self.consumer_id,
                "input_state": list(self.input_state),
                "output_state": list(self.output_state),
                "horizon": self.horizon,
                "replay_contract": self.replay_contract,
                "exact": self.exact,
                "metadata": dict(self.metadata),
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ContinuationContract":
        body = _unseal(payload, schema=MODEL_OS_CONTINUATION_SCHEMA)
        return cls(
            continuation_id=body["continuation_id"],
            consumer_id=body["consumer_id"],
            input_state=tuple(body.get("input_state", ())),
            output_state=tuple(body.get("output_state", ())),
            horizon=body.get("horizon", 1),
            replay_contract=body.get("replay_contract", "exact"),
            exact=bool(body.get("exact", True)),
            metadata=body.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class ModelPackage:
    """An installable Model OS program package and its admission metadata."""

    package_id: str
    version: int
    model_abi: ModelABI
    state_schema: StateSchema
    program: ModelProgram
    state_surface: StateSurface | None = None
    consumers: tuple[ConsumerContract, ...] = ()
    continuations: tuple[ContinuationContract, ...] = ()
    permissions: Mapping[str, Any] = field(default_factory=dict)
    resource_limits: Mapping[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    family_lowerings: Mapping[str, Any] = field(default_factory=dict)
    terminal_claim: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "package_id", _text(self.package_id, name="package_id"))
        if not isinstance(self.version, int) or self.version < 1:
            raise ModelOSContractError("package version must be a positive integer")
        if self.model_abi.state_schema != self.state_schema.fingerprint:
            raise ModelOSContractError(
                "model ABI state_schema does not match the package state schema"
            )
        surface = self.state_surface or StateSurface.from_schema(self.state_schema)
        if surface.state_schema != self.state_schema.fingerprint:
            raise ModelOSContractError(
                "state surface state_schema does not match the package state schema"
            )
        if not set(surface.addresses).issubset(self.state_schema.addresses):
            raise ModelOSContractError("state surface exposes an unknown state address")
        object.__setattr__(self, "state_surface", surface)
        consumers = tuple(self.consumers)
        consumer_ids = [consumer.consumer_id for consumer in consumers]
        if len(consumer_ids) != len(set(consumer_ids)):
            raise ModelOSContractError("consumer ids must be unique")
        object.__setattr__(self, "consumers", consumers)
        continuations = tuple(self.continuations)
        continuation_ids = [continuation.continuation_id for continuation in continuations]
        if len(continuation_ids) != len(set(continuation_ids)):
            raise ModelOSContractError("continuation ids must be unique")
        consumer_ids_set = set(consumer_ids)
        for continuation in continuations:
            if continuation.consumer_id not in consumer_ids_set:
                raise ModelOSContractError(
                    f"continuation references undeclared consumer {continuation.consumer_id!r}"
                )
            for address in (*continuation.input_state, *continuation.output_state):
                if address not in self.state_schema.addresses:
                    raise ModelOSContractError(
                        f"continuation references unknown state address {address!r}"
                    )
        object.__setattr__(self, "continuations", continuations)
        object.__setattr__(self, "evidence_refs", _texts(self.evidence_refs, name="evidence_refs"))
        for name, value in (
            ("permissions", self.permissions),
            ("resource_limits", self.resource_limits),
            ("family_lowerings", self.family_lowerings),
        ):
            normalized = _json_safe(value, path=f"package.{name}")
            if not isinstance(normalized, dict):
                raise ModelOSContractError(f"package {name} must be an object")
            object.__setattr__(self, name, normalized)
        self.validate()

    def validate(self) -> None:
        addresses = self.state_schema.addresses
        for address in (*self.program.reads, *self.program.writes):
            if address not in addresses:
                raise ModelOSContractError(
                    f"program {self.program.program_id} references unknown state "
                    f"address {address!r}"
                )
        consumers = {consumer.consumer_id for consumer in self.consumers}
        for consumer_id in self.program.native_consumers:
            if consumer_id not in consumers:
                raise ModelOSContractError(
                    f"program references undeclared native consumer {consumer_id!r}"
                )

    def consumer(self, consumer_id: str | None) -> ConsumerContract:
        if consumer_id is None:
            if len(self.consumers) != 1:
                raise ModelOSContractError(
                    "consumer_id is required when a package does not have exactly one consumer"
                )
            return self.consumers[0]
        for consumer in self.consumers:
            if consumer.consumer_id == consumer_id:
                return consumer
        raise ModelOSContractError(f"unknown consumer {consumer_id!r}")

    def continuation(self, consumer_id: str) -> ContinuationContract | None:
        matches = tuple(
            continuation
            for continuation in self.continuations
            if continuation.consumer_id == consumer_id
        )
        if len(matches) > 1:
            raise ModelOSContractError(
                f"multiple continuation contracts exist for consumer {consumer_id!r}"
            )
        return matches[0] if matches else None

    def to_payload(self) -> dict[str, Any]:
        return _sealed(
            {
                "schema": MODEL_OS_PACKAGE_SCHEMA,
                "package_id": self.package_id,
                "version": self.version,
                "model_abi": self.model_abi.to_payload(),
                "state_schema": self.state_schema.to_payload(),
                "program": self.program.to_payload(),
                "state_surface": self.state_surface.to_payload(),
                "consumers": [consumer.to_payload() for consumer in self.consumers],
                "continuations": [
                    continuation.to_payload() for continuation in self.continuations
                ],
                "permissions": dict(self.permissions),
                "resource_limits": dict(self.resource_limits),
                "evidence_refs": list(self.evidence_refs),
                "family_lowerings": dict(self.family_lowerings),
                "terminal_claim": self.terminal_claim,
            }
        )

    @property
    def fingerprint(self) -> str:
        return self.to_payload()["content_sha256"]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ModelPackage":
        body = _unseal(payload, schema=MODEL_OS_PACKAGE_SCHEMA)
        return cls(
            package_id=body["package_id"],
            version=body["version"],
            model_abi=ModelABI.from_payload(body["model_abi"]),
            state_schema=StateSchema.from_payload(body["state_schema"]),
            program=ModelProgram.from_payload(body["program"]),
            state_surface=(
                StateSurface.from_payload(body["state_surface"])
                if body.get("state_surface") is not None
                else None
            ),
            consumers=tuple(
                ConsumerContract.from_payload(item)
                for item in body.get("consumers", ())
            ),
            continuations=tuple(
                ContinuationContract.from_payload(item)
                for item in body.get("continuations", ())
            ),
            permissions=body.get("permissions", {}),
            resource_limits=body.get("resource_limits", {}),
            evidence_refs=tuple(body.get("evidence_refs", ())),
            family_lowerings=body.get("family_lowerings", {}),
            terminal_claim=bool(body.get("terminal_claim", False)),
        )


__all__ = [
    "MODEL_OS_ABI_SCHEMA",
    "MODEL_OS_CONTINUATION_SCHEMA",
    "MODEL_OS_PACKAGE_SCHEMA",
    "MODEL_OS_PROGRAM_SCHEMA",
    "MODEL_OS_STATE_SCHEMA",
    "MODEL_OS_SURFACE_SCHEMA",
    "ConsumerContract",
    "ContinuationContract",
    "ModelABI",
    "ModelOSContractError",
    "ModelPackage",
    "ModelProgram",
    "ProgramInstruction",
    "StateSchema",
    "StateSurface",
    "StateSlot",
    "canonical_json_bytes",
    "content_digest",
]

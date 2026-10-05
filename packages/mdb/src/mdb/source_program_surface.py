"""Typed coalition plans over family-local compiled source memory.

This module is the junction between Saturn's source-memory experiments and its
existing Model-OS/Thott control plane.  It deliberately keeps two things
separate:

* a :class:`SourceProgramSurface` is tensor-free metadata describing immutable,
  content-addressed source records;
* a :class:`SourceProgramPlan` is a query-derived coalition of those records,
  with operand, depth, time, coefficient, and recipient coordinates.

The tensor payload stays in a Thott raw-workspace artifact and is interpreted by
one family-local adapter.  Transformer K/V records are never flattened into one
embedding, and recurrent convolution/state operands are never mislabeled as
attention rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from .ar.source_set import ArSourceSet
    from .ar.mamba_state_program import MambaStateProgram
    from .titan import ActSpec

from mdb.future_forest import PersistentFutureForest
from mdb.thought_route_graph import build_thought_route_graph
def tensor_bytes_sha256(*args, **kwargs):
    from .tensor_hash import tensor_bytes_sha256 as implementation
    return implementation(*args, **kwargs)


def build_raw_workspace_artifact(*args, **kwargs):
    from .workspace_raw_artifact import build_raw_workspace_artifact as implementation
    return implementation(*args, **kwargs)


def validate_raw_workspace_artifact(*args, **kwargs):
    from .workspace_raw_artifact import validate_raw_workspace_artifact as implementation
    return implementation(*args, **kwargs)

SOURCE_PROGRAM_SURFACE_SCHEMA = "saturn-source-program-surface-v1"
SOURCE_PROGRAM_PLAN_SCHEMA = "saturn-source-program-plan-v1"
SOURCE_PROGRAM_SNAPSHOT_SCHEMA = "saturn-thott-source-program-snapshot-v1"
SOURCE_PROGRAM_RECEIPT_SCHEMA = "saturn-source-program-execution-receipt-v1"
SOURCE_PROGRAM_DISPATCH_ACT = "model-os.source-program.dispatch"


class SourceProgramError(ValueError):
    """A source surface, coalition, artifact, or receipt failed closed."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceProgramError("source-program value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require_sha256(value: str, *, label: str) -> str:
    candidate = str(value)
    if len(candidate) != 64 or any(char not in "0123456789abcdef" for char in candidate):
        raise SourceProgramError(f"{label} must be a lowercase SHA-256 digest")
    return candidate


def _finite(value: float, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SourceProgramError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class SourceComponentDescriptor:
    """One family-native tensor operand at one execution depth."""

    operand: str
    layer: int
    tensor_key: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not self.operand or not self.tensor_key:
            raise SourceProgramError("source components require operand and tensor key")
        if self.layer < 0 or not self.shape or any(value < 0 for value in self.shape):
            raise SourceProgramError("source component geometry is invalid")
        if not self.dtype.startswith("torch.") or self.bytes < 0:
            raise SourceProgramError("source component dtype or byte count is invalid")
        _require_sha256(self.sha256, label="source component hash")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operand": self.operand,
            "layer": self.layer,
            "tensor_key": self.tensor_key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceComponentDescriptor:
        return cls(
            operand=str(value.get("operand") or ""),
            layer=int(value.get("layer", -1)),
            tensor_key=str(value.get("tensor_key") or ""),
            shape=tuple(int(item) for item in value.get("shape", ())),
            dtype=str(value.get("dtype") or ""),
            sha256=str(value.get("sha256") or ""),
            bytes=int(value.get("bytes", -1)),
        )


@dataclass(frozen=True, slots=True)
class SourceRecordDescriptor:
    """One logical source address and all operands bound to that address."""

    address: str
    components: tuple[SourceComponentDescriptor, ...]
    source_position: int | None = None
    token_id: int | None = None
    semantic_role: str | None = None

    def __post_init__(self) -> None:
        if not self.address or not self.components:
            raise SourceProgramError("source records require an address and components")
        coordinates = [(item.layer, item.operand) for item in self.components]
        if len(coordinates) != len(set(coordinates)):
            raise SourceProgramError("source record repeats a layer/operand coordinate")
        if self.source_position is not None and self.source_position < 0:
            raise SourceProgramError("source position must be non-negative")
        if self.token_id is not None and self.token_id < 0:
            raise SourceProgramError("token id must be non-negative")
        object.__setattr__(
            self,
            "components",
            tuple(sorted(self.components, key=lambda row: (row.layer, row.operand))),
        )

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body = {
            "address": self.address,
            "source_position": self.source_position,
            "token_id": self.token_id,
            "semantic_role": self.semantic_role,
            "components": [item.to_dict() for item in self.components],
        }
        return {**body, "fingerprint": self.fingerprint} if include_fingerprint else body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceRecordDescriptor:
        record = cls(
            address=str(value.get("address") or ""),
            source_position=(
                None
                if value.get("source_position") is None
                else int(value["source_position"])
            ),
            token_id=None if value.get("token_id") is None else int(value["token_id"]),
            semantic_role=(
                None if value.get("semantic_role") is None else str(value["semantic_role"])
            ),
            components=tuple(
                SourceComponentDescriptor.from_dict(item)
                for item in value.get("components", ())
            ),
        )
        if value.get("fingerprint") not in (None, record.fingerprint):
            raise SourceProgramError("source record fingerprint changed")
        return record


@dataclass(frozen=True, slots=True)
class SourceProgramSurface:
    """Tensor-free inventory of one immutable family-local source program."""

    store_id: str
    family: str
    model_id: str
    revision: str
    source_abi: str
    native_consumer: str
    source_fingerprint: str
    records: tuple[SourceRecordDescriptor, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in (
            (self.store_id, "store id"),
            (self.family, "family"),
            (self.model_id, "model id"),
            (self.revision, "revision"),
            (self.source_abi, "source ABI"),
            (self.native_consumer, "native consumer"),
        ):
            if not value:
                raise SourceProgramError(f"{label} cannot be empty")
        _require_sha256(self.source_fingerprint, label="source fingerprint")
        if not self.records:
            raise SourceProgramError("source surface requires at least one record")
        addresses = [row.address for row in self.records]
        if len(addresses) != len(set(addresses)):
            raise SourceProgramError("source surface addresses must be unique")
        try:
            normalized_metadata = json.loads(_canonical(self.metadata).decode("utf-8"))
        except SourceProgramError:
            raise
        object.__setattr__(self, "metadata", normalized_metadata)
        object.__setattr__(
            self, "records", tuple(sorted(self.records, key=lambda row: row.address))
        )

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_fingerprint=False))

    @property
    def addresses(self) -> tuple[str, ...]:
        return tuple(row.address for row in self.records)

    def record(self, address: str) -> SourceRecordDescriptor:
        for row in self.records:
            if row.address == address:
                return row
        raise SourceProgramError(f"source surface has no address {address!r}")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body = {
            "schema": SOURCE_PROGRAM_SURFACE_SCHEMA,
            "store_id": self.store_id,
            "family": self.family,
            "model_id": self.model_id,
            "revision": self.revision,
            "source_abi": self.source_abi,
            "native_consumer": self.native_consumer,
            "source_fingerprint": self.source_fingerprint,
            "records": [row.to_dict() for row in self.records],
            "metadata": dict(self.metadata),
        }
        return {**body, "content_sha256": self.fingerprint} if include_fingerprint else body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceProgramSurface:
        if value.get("schema") != SOURCE_PROGRAM_SURFACE_SCHEMA:
            raise SourceProgramError("unsupported source-program surface schema")
        surface = cls(
            store_id=str(value.get("store_id") or ""),
            family=str(value.get("family") or ""),
            model_id=str(value.get("model_id") or ""),
            revision=str(value.get("revision") or ""),
            source_abi=str(value.get("source_abi") or ""),
            native_consumer=str(value.get("native_consumer") or ""),
            source_fingerprint=str(value.get("source_fingerprint") or ""),
            records=tuple(
                SourceRecordDescriptor.from_dict(item) for item in value.get("records", ())
            ),
            metadata=value.get("metadata") or {},
        )
        if value.get("content_sha256") != surface.fingerprint:
            raise SourceProgramError("source-program surface fingerprint changed")
        return surface


@dataclass(frozen=True, slots=True)
class SourceProgramArtifact:
    """A surface plus its exact opt-in Thott tensor artifact."""

    surface: SourceProgramSurface
    encoded: bytes
    raw_reference: Mapping[str, Any]
    raw_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        validation = validate_raw_workspace_artifact(
            self.encoded, expected_reference=self.raw_reference
        )
        if validation["tensor_count"] != len(
            [item for row in self.surface.records for item in row.components]
        ):
            raise SourceProgramError("source artifact tensor inventory changed")
        if hashlib.sha256(_canonical(self.raw_manifest)).hexdigest() != self.raw_reference.get(
            "manifest_sha256"
        ):
            raise SourceProgramError("source artifact manifest fingerprint changed")
        scope = self.raw_manifest.get("scope")
        if not isinstance(scope, Mapping) or scope.get(
            "source_program_surface_fingerprint"
        ) != self.surface.fingerprint:
            raise SourceProgramError("source artifact is bound to another surface")
        rows = self.raw_manifest.get("tensors")
        if not isinstance(rows, list):
            raise SourceProgramError("source artifact manifest is incomplete")
        by_original = {
            str(row.get("original_key")): row for row in rows if isinstance(row, Mapping)
        }
        components = [item for row in self.surface.records for item in row.components]
        if set(by_original) != {item.tensor_key for item in components}:
            raise SourceProgramError("source artifact keys do not match the surface")
        for item in components:
            raw = by_original[item.tensor_key]
            if (
                raw.get("sha256") != item.sha256
                or tuple(raw.get("shape") or ()) != item.shape
                or str(raw.get("dtype")) != item.dtype
                or int(raw.get("bytes", -1)) != item.bytes
            ):
                raise SourceProgramError("source component custody changed")

    def tensors(self) -> dict[str, torch.Tensor]:
        from safetensors.torch import load as load_safetensors
        loaded = load_safetensors(self.encoded)
        names = {
            str(row["original_key"]): str(row["tensor_name"])
            for row in self.raw_manifest["tensors"]
        }
        return {original: loaded[tensor_name] for original, tensor_name in names.items()}


def _component(
    operand: str, layer: int, tensor_key: str, value: torch.Tensor
) -> SourceComponentDescriptor:
    prepared = value.detach().to(device="cpu").contiguous()
    return SourceComponentDescriptor(
        operand=operand,
        layer=layer,
        tensor_key=tensor_key,
        shape=tuple(int(item) for item in prepared.shape),
        dtype=str(prepared.dtype),
        sha256=tensor_bytes_sha256(prepared),
        bytes=int(prepared.numel() * prepared.element_size()),
    )


def artifact_from_ar_source_set(
    source: ArSourceSet,
    *,
    store_id: str,
    semantic_roles: Mapping[int, str] | None = None,
    native_consumer: str = "autoregressive.native-attention",
) -> SourceProgramArtifact:
    """Lower paired vertical K/V source records without flattening depth."""

    roles = dict(semantic_roles or {})
    tensors: dict[str, torch.Tensor] = {}
    records: list[SourceRecordDescriptor] = []
    for position, token_id in enumerate(source.source_token_ids):
        components: list[SourceComponentDescriptor] = []
        for layer, (key, value) in enumerate(source.layers):
            for operand, tensor in (
                ("key", key[:, :, position : position + 1, :]),
                ("value", value[:, :, position : position + 1, :]),
            ):
                tensor_key = f"records/{position}/layers/{layer}/{operand}"
                prepared = tensor.detach().to(device="cpu").contiguous()
                tensors[tensor_key] = prepared
                components.append(_component(operand, layer, tensor_key, prepared))
        records.append(
            SourceRecordDescriptor(
                address=f"token:{position}",
                source_position=position,
                token_id=token_id,
                semantic_role=roles.get(position),
                components=tuple(components),
            )
        )
    surface = SourceProgramSurface(
        store_id=store_id,
        family="transformer-kv",
        model_id=source.model_id,
        revision=source.revision,
        source_abi=source.cache_abi,
        native_consumer=native_consumer,
        source_fingerprint=source.fingerprint,
        records=tuple(records),
        metadata={
            "source_token_ids": list(source.source_token_ids),
            "admitted": list(source.admitted),
            "record_order": [f"token:{index}" for index in range(source.record_count)],
            "quotient_symmetry": "joint-key-value-source-row-permutation",
        },
    )
    encoded, reference, manifest = build_raw_workspace_artifact(
        tensors,
        scope={
            "kind": "source-program",
            "family": surface.family,
            "source_program_surface_fingerprint": surface.fingerprint,
        },
    )
    return SourceProgramArtifact(surface, encoded, reference, manifest)


def artifact_from_mamba_state_program(
    source: MambaStateProgram,
    *,
    store_id: str,
    semantic_roles: Mapping[int, str] | None = None,
    native_consumer: str = "autoregressive.native-recurrence",
) -> SourceProgramArtifact:
    """Lower a Mamba prefix into depth-addressed conv/recurrent records."""

    roles = dict(semantic_roles or {})
    tensors: dict[str, torch.Tensor] = {}
    records: list[SourceRecordDescriptor] = []
    for layer, state in enumerate(source.layers):
        components: list[SourceComponentDescriptor] = []
        for operand, tensor in (("conv", state.conv), ("recurrent", state.recurrent)):
            tensor_key = f"layers/{layer}/{operand}"
            prepared = tensor.detach().to(device="cpu").contiguous()
            tensors[tensor_key] = prepared
            components.append(_component(operand, layer, tensor_key, prepared))
        records.append(
            SourceRecordDescriptor(
                address=f"layer:{layer}",
                semantic_role=roles.get(layer),
                components=tuple(components),
            )
        )
    surface = SourceProgramSurface(
        store_id=store_id,
        family="mamba-recurrent-state",
        model_id=source.model_id,
        revision=source.revision,
        source_abi="transformers-dynamic-cache-linear-attention-v1",
        native_consumer=native_consumer,
        source_fingerprint=str(source.manifest()["fingerprint"]),
        records=tuple(records),
        metadata={
            "source_token_ids": list(source.source_token_ids),
            "pending_token_id": source.pending_token_id,
            "execution_abi_sha256": source.execution_abi_sha256,
            "record_order": [f"layer:{index}" for index in range(len(source.layers))],
            "depth_contract": "ordered-not-permutation-invariant",
        },
    )
    encoded, reference, manifest = build_raw_workspace_artifact(
        tensors,
        scope={
            "kind": "source-program",
            "family": surface.family,
            "source_program_surface_fingerprint": surface.fingerprint,
        },
    )
    return SourceProgramArtifact(surface, encoded, reference, manifest)


@dataclass(frozen=True, slots=True)
class SourceProgramBinding:
    store_id: str
    surface_fingerprint: str
    family: str
    model_id: str
    revision: str
    source_abi: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.store_id, "store id"),
            (self.family, "family"),
            (self.model_id, "model id"),
            (self.revision, "revision"),
            (self.source_abi, "source ABI"),
        ):
            if not value:
                raise SourceProgramError(f"source-program binding {label} cannot be empty")
        _require_sha256(self.surface_fingerprint, label="surface binding")

    @classmethod
    def from_surface(cls, surface: SourceProgramSurface) -> SourceProgramBinding:
        return cls(
            surface.store_id,
            surface.fingerprint,
            surface.family,
            surface.model_id,
            surface.revision,
            surface.source_abi,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "store_id": self.store_id,
            "surface_fingerprint": self.surface_fingerprint,
            "family": self.family,
            "model_id": self.model_id,
            "revision": self.revision,
            "source_abi": self.source_abi,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceProgramBinding:
        return cls(
            **{
                key: str(value.get(key) or "")
                for key in (
                    "store_id",
                    "surface_fingerprint",
                    "family",
                    "model_id",
                    "revision",
                    "source_abi",
                )
            }
        )


@dataclass(frozen=True, slots=True)
class SourceProgramSelection:
    """One member of a source coalition with typed execution coordinates."""

    store_id: str
    address: str
    coefficient: float = 1.0
    operands: tuple[str, ...] = ()
    depths: tuple[int, ...] = ()
    depth_map: tuple[tuple[int, int], ...] = ()
    times: tuple[int, ...] = ()
    recipients: tuple[str, ...] = ("native-interpreter",)

    def __post_init__(self) -> None:
        if not self.store_id or not self.address or not self.recipients:
            raise SourceProgramError("source selection requires store, address, and recipient")
        object.__setattr__(self, "coefficient", _finite(self.coefficient, label="coefficient"))
        for name in ("operands", "recipients"):
            values = tuple(sorted({str(item) for item in getattr(self, name)}))
            if any(not item for item in values):
                raise SourceProgramError(f"{name} cannot contain empty values")
            object.__setattr__(self, name, values)
        for name in ("depths", "times"):
            values = tuple(sorted({int(item) for item in getattr(self, name)}))
            if any(item < 0 for item in values):
                raise SourceProgramError(f"{name} must be non-negative")
            object.__setattr__(self, name, values)
        mapping = tuple(
            sorted((int(source), int(recipient)) for source, recipient in self.depth_map)
        )
        if any(source < 0 or recipient < 0 for source, recipient in mapping):
            raise SourceProgramError("depth-map coordinates must be non-negative")
        if len({source for source, _ in mapping}) != len(mapping) or len(
            {recipient for _, recipient in mapping}
        ) != len(mapping):
            raise SourceProgramError("depth map must be one-to-one")
        if self.depths and mapping:
            raise SourceProgramError("use depths or depth_map, not both")
        object.__setattr__(self, "depth_map", mapping)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "store_id": self.store_id,
            "address": self.address,
            "coefficient": self.coefficient,
            "operands": list(self.operands),
            "depths": list(self.depths),
            "times": list(self.times),
            "recipients": list(self.recipients),
        }
        # Keep the empty form byte-compatible with v1 plans already stored in
        # Future Forest. Depth transport is an optional additive extension.
        if self.depth_map:
            body["depth_map"] = [
                {"source": source, "recipient": recipient}
                for source, recipient in self.depth_map
            ]
        return body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceProgramSelection:
        return cls(
            store_id=str(value.get("store_id") or ""),
            address=str(value.get("address") or ""),
            coefficient=float(value.get("coefficient", 1.0)),
            operands=tuple(str(item) for item in value.get("operands", ())),
            depths=tuple(int(item) for item in value.get("depths", ())),
            depth_map=tuple(
                (int(item["source"]), int(item["recipient"]))
                for item in value.get("depth_map", ())
            ),
            times=tuple(int(item) for item in value.get("times", ())),
            recipients=tuple(str(item) for item in value.get("recipients", ())),
        )


@dataclass(frozen=True, slots=True)
class SourceProgramOperation:
    store_id: str
    address: str
    record_fingerprint: str
    operand: str
    source_layer: int
    layer: int
    tensor_key: str
    coefficient: float
    time: int | None
    recipient: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_id": self.store_id,
            "address": self.address,
            "record_fingerprint": self.record_fingerprint,
            "operand": self.operand,
            "source_layer": self.source_layer,
            "layer": self.layer,
            "tensor_key": self.tensor_key,
            "coefficient": self.coefficient,
            "time": self.time,
            "recipient": self.recipient,
        }


def _selection_key(value: SourceProgramSelection) -> bytes:
    return _canonical(value.to_dict())


@dataclass(frozen=True, slots=True)
class SourceProgramPlan:
    """A permutation-invariant coalition over one compatible source-store family."""

    bindings: tuple[SourceProgramBinding, ...]
    selections: tuple[SourceProgramSelection, ...]
    resolver: str
    query_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bindings or not self.resolver:
            raise SourceProgramError("source plan requires bindings and resolver identity")
        _require_sha256(self.query_fingerprint, label="query fingerprint")
        bindings = tuple(sorted(self.bindings, key=lambda row: row.store_id))
        if len({row.store_id for row in bindings}) != len(bindings):
            raise SourceProgramError("source plan store bindings must be unique")
        compatibility = {
            (row.family, row.model_id, row.revision, row.source_abi) for row in bindings
        }
        if len(compatibility) != 1:
            raise SourceProgramError("one plan cannot mix family-local source ABIs")
        selections = tuple(sorted(self.selections, key=_selection_key))
        keys = [_selection_key(row) for row in selections]
        if len(keys) != len(set(keys)):
            raise SourceProgramError("source plan repeats an identical selection")
        known = {row.store_id for row in bindings}
        if any(row.store_id not in known for row in selections):
            raise SourceProgramError("source selection names an unbound store")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "metadata", json.loads(_canonical(self.metadata)))

    @property
    def abstained(self) -> bool:
        return not self.selections

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        body = {
            "schema": SOURCE_PROGRAM_PLAN_SCHEMA,
            "bindings": [row.to_dict() for row in self.bindings],
            "selections": [row.to_dict() for row in self.selections],
            "resolver": self.resolver,
            "query_fingerprint": self.query_fingerprint,
            "abstained": self.abstained,
            "metadata": dict(self.metadata),
        }
        return {**body, "content_sha256": self.fingerprint} if include_fingerprint else body

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceProgramPlan:
        if value.get("schema") != SOURCE_PROGRAM_PLAN_SCHEMA:
            raise SourceProgramError("unsupported source-program plan schema")
        plan = cls(
            bindings=tuple(
                SourceProgramBinding.from_dict(item) for item in value.get("bindings", ())
            ),
            selections=tuple(
                SourceProgramSelection.from_dict(item) for item in value.get("selections", ())
            ),
            resolver=str(value.get("resolver") or ""),
            query_fingerprint=str(value.get("query_fingerprint") or ""),
            metadata=value.get("metadata") or {},
        )
        if value.get("content_sha256") != plan.fingerprint:
            raise SourceProgramError("source-program plan fingerprint changed")
        return plan

    def expand(
        self,
        surfaces: Mapping[str, SourceProgramSurface],
        *,
        execution_time: int | None = None,
    ) -> tuple[SourceProgramOperation, ...]:
        _validate_surface_bindings(self.bindings, surfaces)
        operations: list[SourceProgramOperation] = []
        for selection in self.selections:
            record = surfaces[selection.store_id].record(selection.address)
            depth_map = dict(selection.depth_map)
            components = tuple(
                component
                for component in record.components
                if (not selection.operands or component.operand in selection.operands)
                and (not selection.depths or component.layer in selection.depths)
                and (not depth_map or component.layer in depth_map)
            )
            if not components:
                raise SourceProgramError(
                    f"selection {selection.store_id}:{selection.address} matches no component"
                )
            times: tuple[int | None, ...] = selection.times or (None,)
            if execution_time is not None:
                times = tuple(time for time in times if time in (None, execution_time))
            for component in components:
                for time in times:
                    for recipient in selection.recipients:
                        operations.append(
                            SourceProgramOperation(
                                store_id=selection.store_id,
                                address=selection.address,
                                record_fingerprint=record.fingerprint,
                                operand=component.operand,
                                source_layer=component.layer,
                                layer=depth_map.get(component.layer, component.layer),
                                tensor_key=component.tensor_key,
                                coefficient=selection.coefficient,
                                time=time,
                                recipient=recipient,
                            )
                        )
        return tuple(
            sorted(
                operations,
                key=lambda row: (
                    row.time if row.time is not None else -1,
                    row.layer,
                    row.recipient,
                    row.store_id,
                    row.address,
                    row.operand,
                ),
            )
        )


def _validate_surface_bindings(
    bindings: Sequence[SourceProgramBinding],
    surfaces: Mapping[str, SourceProgramSurface],
) -> None:
    if set(surfaces) != {row.store_id for row in bindings}:
        raise SourceProgramError("live source-store inventory changed")
    for binding in bindings:
        surface = surfaces[binding.store_id]
        if SourceProgramBinding.from_surface(surface) != binding:
            raise SourceProgramError(f"source-store binding changed for {binding.store_id!r}")


def compile_source_program_plan(
    surfaces: Sequence[SourceProgramSurface],
    selections: Sequence[SourceProgramSelection],
    *,
    resolver: str,
    query: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> SourceProgramPlan:
    """Compile a query-derived coalition and fail closed on unknown coordinates."""

    by_store = {surface.store_id: surface for surface in surfaces}
    if len(by_store) != len(surfaces):
        raise SourceProgramError("source surfaces repeat a store id")
    plan = SourceProgramPlan(
        bindings=tuple(SourceProgramBinding.from_surface(surface) for surface in surfaces),
        selections=tuple(selections),
        resolver=resolver,
        query_fingerprint=_digest(query),
        metadata={"query_schema": query.get("schema"), **dict(metadata or {})},
    )
    plan.expand(by_store)
    return plan


def source_program_dispatch_spec(plan: SourceProgramPlan) -> ActSpec:
    """Describe the metadata-level dispatch boundary as a Model-OS Act."""
    from .titan import ActSpec
    recipients = sorted(
        {recipient for selection in plan.selections for recipient in selection.recipients}
    ) or ["native-interpreter"]
    writes = tuple(f"source.program.dispatch.{recipient}" for recipient in recipients)
    return ActSpec(
        name=SOURCE_PROGRAM_DISPATCH_ACT,
        abi_version="v1",
        reads=tuple(
            ["source.program.query"]
            + [f"source.program.store.{binding.store_id}" for binding in plan.bindings]
        ),
        writes=writes,
        invalidates=tuple(f"{recipient}.downstream-state" for recipient in recipients),
        numerical_contract="finite-coalition-coefficients-and-identity-bound-payload-handles",
        tolerance={"maximum_absolute_nonfinite": 0.0},
        write_kinds={name: "json" for name in writes},
        write_abis={
            name: {
                "schema": SOURCE_PROGRAM_PLAN_SCHEMA,
                "plan_fingerprint": plan.fingerprint,
                "payload": "content-addressed-family-local-handle",
            }
            for name in writes
        },
    )


def _artifact_map(
    artifacts: Mapping[str, SourceProgramArtifact], plan: SourceProgramPlan
) -> dict[str, SourceProgramArtifact]:
    surfaces = {store_id: artifact.surface for store_id, artifact in artifacts.items()}
    _validate_surface_bindings(plan.bindings, surfaces)
    return dict(artifacts)


def materialize_ar_source_set(
    artifacts: Mapping[str, SourceProgramArtifact],
    plan: SourceProgramPlan,
    *,
    base_store_id: str,
    execution_time: int | None = None,
    recipient: str = "native-interpreter",
) -> ArSourceSet:
    """Lower a compatible coalition into one native paired K/V source set."""
    import torch
    from .ar.source_set import ArSourceSet
    resolved = _artifact_map(artifacts, plan)
    base = resolved.get(base_store_id)
    if base is None or base.surface.family != "transformer-kv":
        raise SourceProgramError("AR materialization requires a transformer K/V base store")
    surfaces = {name: item.surface for name, item in resolved.items()}
    tensors = {name: item.tensors() for name, item in resolved.items()}
    if execution_time is None and any(selection.times for selection in plan.selections):
        raise SourceProgramError("time-qualified AR plans require an execution time")
    operations = tuple(
        operation
        for operation in plan.expand(surfaces, execution_time=execution_time)
        if operation.recipient == recipient
    )
    if not operations and not plan.abstained:
        raise SourceProgramError(f"AR plan has no operations for recipient {recipient!r}")
    order = tuple(str(item) for item in base.surface.metadata.get("record_order", ()))
    if not order or set(order) != set(base.surface.addresses):
        raise SourceProgramError("AR base store has no valid physical record order")
    base_components = {
        (record.address, item.layer, item.operand): item
        for record in base.surface.records
        for item in record.components
    }
    output = {
        coordinate: torch.zeros(
            tuple(item.shape), dtype=getattr(torch, item.dtype.removeprefix("torch."))
        )
        for coordinate, item in base_components.items()
    }
    admitted: set[str] = set()
    for operation in operations:
        coordinate = (operation.address, operation.layer, operation.operand)
        if coordinate not in output:
            raise SourceProgramError("coalition source cannot be aligned to the base chart")
        value = tensors[operation.store_id][operation.tensor_key]
        if value.shape != output[coordinate].shape or value.dtype != output[coordinate].dtype:
            raise SourceProgramError("coalition source component ABI changed")
        output[coordinate].add_(value, alpha=operation.coefficient)
        admitted.add(operation.address)
    layers = []
    layer_count = max(item.layer for item in base_components.values()) + 1
    for layer in range(layer_count):
        layers.append(
            (
                torch.cat([output[(address, layer, "key")] for address in order], dim=-2),
                torch.cat([output[(address, layer, "value")] for address in order], dim=-2),
            )
        )
    token_ids = tuple(int(item) for item in base.surface.metadata["source_token_ids"])
    return ArSourceSet(
        model_id=base.surface.model_id,
        revision=base.surface.revision,
        source_token_ids=token_ids,
        layers=tuple(layers),
        admitted=tuple(address in admitted for address in order),
        cache_abi=base.surface.source_abi,
    )


def materialize_mamba_state_program(
    artifacts: Mapping[str, SourceProgramArtifact],
    plan: SourceProgramPlan,
    *,
    base_store_id: str,
    execution_time: int | None = None,
    recipient: str = "native-interpreter",
) -> MambaStateProgram:
    """Lower a compatible coalition into Mamba's ordered recurrent cache ABI."""
    import torch
    from .ar.mamba_state_program import MambaLayerState, MambaStateProgram
    resolved = _artifact_map(artifacts, plan)
    base = resolved.get(base_store_id)
    if base is None or base.surface.family != "mamba-recurrent-state":
        raise SourceProgramError("Mamba materialization requires a recurrent-state base store")
    surfaces = {name: item.surface for name, item in resolved.items()}
    tensors = {name: item.tensors() for name, item in resolved.items()}
    if execution_time is None and any(selection.times for selection in plan.selections):
        raise SourceProgramError("time-qualified Mamba plans require an execution time")
    operations = tuple(
        operation
        for operation in plan.expand(surfaces, execution_time=execution_time)
        if operation.recipient == recipient
    )
    if not operations and not plan.abstained:
        raise SourceProgramError(f"Mamba plan has no operations for recipient {recipient!r}")
    base_components = {
        (record.address, item.layer, item.operand): item
        for record in base.surface.records
        for item in record.components
    }
    output = {
        coordinate: torch.zeros(
            tuple(item.shape), dtype=getattr(torch, item.dtype.removeprefix("torch."))
        )
        for coordinate, item in base_components.items()
    }
    for operation in operations:
        coordinate = (f"layer:{operation.layer}", operation.layer, operation.operand)
        if coordinate not in output:
            raise SourceProgramError("recurrent coalition cannot be aligned to the base chart")
        value = tensors[operation.store_id][operation.tensor_key]
        if value.shape != output[coordinate].shape or value.dtype != output[coordinate].dtype:
            raise SourceProgramError("recurrent coalition component ABI changed")
        output[coordinate].add_(value, alpha=operation.coefficient)
    order = tuple(str(item) for item in base.surface.metadata.get("record_order", ()))
    layers = tuple(
        MambaLayerState(
            output[(address, index, "conv")],
            output[(address, index, "recurrent")],
        )
        for index, address in enumerate(order)
    )
    return MambaStateProgram(
        model_id=base.surface.model_id,
        revision=base.surface.revision,
        source_token_ids=tuple(int(item) for item in base.surface.metadata["source_token_ids"]),
        pending_token_id=int(base.surface.metadata["pending_token_id"]),
        layers=layers,
        execution_abi_sha256=base.surface.metadata.get("execution_abi_sha256"),
    )


@dataclass(frozen=True, slots=True)
class SourceProgramSnapshot:
    node_id: str
    label: str
    surface_fingerprints: Mapping[str, str]
    artifact_sha256s: Mapping[str, str]
    plan_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_PROGRAM_SNAPSHOT_SCHEMA,
            "tool": {"name": "Thott", "id": "saturn-thott"},
            "node_id": self.node_id,
            "label": self.label,
            "surface_fingerprints": dict(self.surface_fingerprints),
            "artifact_sha256s": dict(self.artifact_sha256s),
            "plan_fingerprint": self.plan_fingerprint,
        }


class ThottSourceProgramRegister:
    """Persist source programs as immutable Future-Forest states and branches."""

    def __init__(self, root: str | Path, *, forest_id: str, reopen: bool = False) -> None:
        path = Path(root)
        self.forest = (
            PersistentFutureForest.reopen(path)
            if reopen
            else PersistentFutureForest(path, forest_id=forest_id)
        )

    @classmethod
    def reopen(cls, root: str | Path) -> ThottSourceProgramRegister:
        forest = PersistentFutureForest.reopen(Path(root))
        return cls(root, forest_id=forest.forest_id, reopen=True)

    def capture(
        self,
        artifacts: SourceProgramArtifact | Mapping[str, SourceProgramArtifact],
        *,
        label: str,
        parent_id: str | None = None,
        plan: SourceProgramPlan | None = None,
        transition: str = "source-program-capture",
    ) -> SourceProgramSnapshot:
        if not label:
            raise SourceProgramError("source-program capture requires a label")
        artifact_map = (
            {artifacts.surface.store_id: artifacts}
            if isinstance(artifacts, SourceProgramArtifact)
            else dict(artifacts)
        )
        if not artifact_map or any(
            store_id != artifact.surface.store_id
            for store_id, artifact in artifact_map.items()
        ):
            raise SourceProgramError("captured source-store inventory is malformed")
        if plan is not None:
            _validate_surface_bindings(
                plan.bindings,
                {store_id: artifact.surface for store_id, artifact in artifact_map.items()},
            )
        refs: dict[str, Mapping[str, Any]] = {}
        for store_id, artifact in sorted(artifact_map.items()):
            refs[f"source-program/{store_id}/tensors"] = self.forest.put_blob(
                artifact.encoded,
                media_type="application/x-safetensors",
                metadata={"kind": "thott-source-program-tensors", "store": store_id},
            )
            refs[f"source-program/{store_id}/descriptor"] = self.forest.put_json_object(
                {
                    "surface": artifact.surface.to_dict(),
                    "raw_reference": dict(artifact.raw_reference),
                    "raw_manifest": dict(artifact.raw_manifest),
                },
                media_type="application/vnd.saturn.source-program+json",
                metadata={"kind": "thott-source-program-descriptor", "store": store_id},
            )
        if plan is not None:
            refs["source-program/plan"] = self.forest.put_json_object(
                plan.to_dict(),
                media_type="application/vnd.saturn.source-program-plan+json",
                metadata={"kind": "thott-source-program-plan"},
            )
        nodes = self.forest.list_nodes()
        if not nodes:
            if parent_id is not None:
                raise SourceProgramError("a root source capture cannot name a parent")
            node_id = self.forest.create_root(
                state_refs=refs,
                metadata={"label": label, "kind": transition},
            )
        else:
            parent = self.forest.head() if parent_id is None else parent_id
            node_id = self.forest.fork(
                parent,
                transition={"kind": transition},
                diff_refs=refs,
                metadata={"label": label},
            )
        return SourceProgramSnapshot(
            node_id=node_id,
            label=label,
            surface_fingerprints={
                store_id: artifact.surface.fingerprint
                for store_id, artifact in sorted(artifact_map.items())
            },
            artifact_sha256s={
                store_id: str(artifact.raw_reference["sha256"])
                for store_id, artifact in sorted(artifact_map.items())
            },
            plan_fingerprint=None if plan is None else plan.fingerprint,
        )

    def restore_bundle(
        self, node_id: str
    ) -> tuple[
        dict[str, SourceProgramArtifact],
        SourceProgramPlan | None,
        SourceProgramSnapshot,
    ]:
        state = self.forest.effective_state(node_id)
        descriptor_prefix = "source-program/"
        descriptor_suffix = "/descriptor"
        stores = sorted(
            address[len(descriptor_prefix) : -len(descriptor_suffix)]
            for address in state
            if address.startswith(descriptor_prefix) and address.endswith(descriptor_suffix)
        )
        if not stores:
            raise SourceProgramError("future node has no source-program stores")
        artifacts: dict[str, SourceProgramArtifact] = {}
        for store_id in stores:
            descriptor_address = f"source-program/{store_id}/descriptor"
            tensor_address = f"source-program/{store_id}/tensors"
            if tensor_address not in state:
                raise SourceProgramError(f"future node is missing tensors for {store_id!r}")
            descriptor = json.loads(
                self.forest.get_blob(state[descriptor_address]).decode("utf-8")
            )
            surface = SourceProgramSurface.from_dict(descriptor["surface"])
            if surface.store_id != store_id:
                raise SourceProgramError("restored store address and surface identity disagree")
            artifacts[store_id] = SourceProgramArtifact(
                surface=surface,
                encoded=self.forest.get_blob(state[tensor_address]),
                raw_reference=descriptor["raw_reference"],
                raw_manifest=descriptor["raw_manifest"],
            )
        raw_plan = (
            None
            if "source-program/plan" not in state
            else json.loads(self.forest.get_blob(state["source-program/plan"]).decode("utf-8"))
        )
        plan = None if raw_plan is None else SourceProgramPlan.from_dict(raw_plan)
        if plan is not None:
            _validate_surface_bindings(
                plan.bindings,
                {store_id: artifact.surface for store_id, artifact in artifacts.items()},
            )
        node = self.forest.verify_node(node_id)
        label = str((node.get("metadata") or {}).get("label") or "restored")
        snapshot = SourceProgramSnapshot(
            node_id=node_id,
            label=label,
            surface_fingerprints={
                store_id: artifact.surface.fingerprint
                for store_id, artifact in sorted(artifacts.items())
            },
            artifact_sha256s={
                store_id: str(artifact.raw_reference["sha256"])
                for store_id, artifact in sorted(artifacts.items())
            },
            plan_fingerprint=None if plan is None else plan.fingerprint,
        )
        return artifacts, plan, snapshot

    def restore(
        self, node_id: str
    ) -> tuple[SourceProgramArtifact, SourceProgramPlan | None, SourceProgramSnapshot]:
        """Restore a one-store capture; use ``restore_bundle`` for coalitions."""

        artifacts, plan, snapshot = self.restore_bundle(node_id)
        if len(artifacts) != 1:
            raise SourceProgramError("source-program node contains a coalition; restore the bundle")
        return next(iter(artifacts.values())), plan, snapshot


def build_source_program_execution_receipt(
    plan: SourceProgramPlan,
    *,
    parent_frame_id: str,
    native_consumer: str,
    output_fingerprint: str,
    mechanics_status: str,
    instrument_status: str,
    trend_status: str,
    terminal_status: str,
    controls: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a consumer receipt without collapsing evidence statuses into one gate."""

    if not parent_frame_id or not native_consumer:
        raise SourceProgramError("execution receipt requires parent and native consumer")
    _require_sha256(output_fingerprint, label="consumer output fingerprint")
    body = {
        "schema": SOURCE_PROGRAM_RECEIPT_SCHEMA,
        "plan_fingerprint": plan.fingerprint,
        "parent_frame_id": parent_frame_id,
        "native_consumer": native_consumer,
        "output_fingerprint": output_fingerprint,
        "statuses": {
            "mechanics": mechanics_status,
            "instrument": instrument_status,
            "trend": trend_status,
            "terminal": terminal_status,
        },
        "controls": dict(controls or {}),
    }
    return {**body, "content_sha256": _digest(body)}


def validate_source_program_execution_receipt(
    receipt: Mapping[str, Any], *, plan: SourceProgramPlan
) -> None:
    if receipt.get("schema") != SOURCE_PROGRAM_RECEIPT_SCHEMA:
        raise SourceProgramError("unsupported source-program receipt schema")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != _digest(body):
        raise SourceProgramError("source-program receipt fingerprint changed")
    if receipt.get("plan_fingerprint") != plan.fingerprint:
        raise SourceProgramError("execution receipt is bound to another plan")


def build_source_program_thought_graph(
    plan: SourceProgramPlan,
    surfaces: Mapping[str, SourceProgramSurface],
    *,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Join a causally executed plan to Thott's producer/carrier/consumer graph."""
    from .circuit_thought_trace import CircuitNode, CircuitThoughtRecorder
    validate_source_program_execution_receipt(receipt, plan=plan)
    _validate_surface_bindings(plan.bindings, surfaces)
    consumer = str(receipt["native_consumer"])
    recorder = CircuitThoughtRecorder(
        f"source-program:{plan.fingerprint}",
        specimen={
            "plan_fingerprint": plan.fingerprint,
            "parent_frame_id": receipt["parent_frame_id"],
            "statuses": receipt["statuses"],
        },
        model_identity={
            "bindings": [binding.to_dict() for binding in plan.bindings],
        },
    )
    recorder.add_node(
        CircuitNode(
            node_id="source-program-compiler",
            role="producer",
            address=f"source-program://plan/{plan.fingerprint}",
            component="coalition-resolver",
        )
    )
    recorder.add_node(
        CircuitNode(
            node_id="native-interpreter",
            role="consumer",
            address="source-program://native-interpreter",
            consumer=consumer,
        )
    )
    recorder.add_node(
        CircuitNode(
            node_id="native-readout",
            role="readout",
            address=f"consumer://{consumer}",
            consumer=consumer,
            metadata={"output_fingerprint": receipt["output_fingerprint"]},
        )
    )
    recorder.add_edge(
        "native-interpreter", "native-readout", carrier="native-state", consumer=consumer
    )
    operations = plan.expand(surfaces)
    selected = sorted({(row.store_id, row.address) for row in plan.selections})
    for index, (store_id, address) in enumerate(selected):
        record = surfaces[store_id].record(address)
        node_id = f"source-record-{index}"
        recorder.add_node(
            CircuitNode(
                node_id=node_id,
                role="carrier",
                address=f"source-program://{store_id}/{address}",
                position=record.source_position,
                component="family-local-source-record",
                consumer=consumer,
                metadata={
                    "record_fingerprint": record.fingerprint,
                    "semantic_role": record.semantic_role,
                    "operations": [
                        operation.to_dict()
                        for operation in operations
                        if operation.store_id == store_id and operation.address == address
                    ],
                },
            )
        )
        recorder.add_edge(
            "source-program-compiler", node_id, carrier="resolved-source-coordinate"
        )
        recorder.add_edge(
            node_id, "native-interpreter", carrier="family-local-source-payload", consumer=consumer
        )
    recorder.begin(phase="source-program-dispatch", execution_step=0)
    recorder.record(
        "native-readout",
        "consumer-commit",
        None,
        metadata={"receipt_fingerprint": receipt["content_sha256"]},
    )
    return build_thought_route_graph(thott_trace=recorder.to_dict())


__all__ = [
    "SOURCE_PROGRAM_DISPATCH_ACT",
    "SOURCE_PROGRAM_PLAN_SCHEMA",
    "SOURCE_PROGRAM_RECEIPT_SCHEMA",
    "SOURCE_PROGRAM_SNAPSHOT_SCHEMA",
    "SOURCE_PROGRAM_SURFACE_SCHEMA",
    "SourceComponentDescriptor",
    "SourceProgramArtifact",
    "SourceProgramBinding",
    "SourceProgramError",
    "SourceProgramOperation",
    "SourceProgramPlan",
    "SourceProgramSelection",
    "SourceProgramSnapshot",
    "SourceProgramSurface",
    "SourceRecordDescriptor",
    "ThottSourceProgramRegister",
    "artifact_from_ar_source_set",
    "artifact_from_mamba_state_program",
    "build_source_program_execution_receipt",
    "build_source_program_thought_graph",
    "compile_source_program_plan",
    "materialize_ar_source_set",
    "materialize_mamba_state_program",
    "source_program_dispatch_spec",
    "validate_source_program_execution_receipt",
]

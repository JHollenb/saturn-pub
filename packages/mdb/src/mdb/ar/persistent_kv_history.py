"""Content-addressed, metadata-diff persistence for autoregressive KV history.

Runtime ``PagedKVCache`` objects share tensor pages inside one process.  This
module extends the same idea across process and lease boundaries with a small
Merkle DAG:

* immutable safetensors page objects own key/value bytes;
* one genesis root owns the typed cache metadata;
* every descendant root carries only its parent identity and new page refs;
* hydration verifies the complete reachable closure before returning state.

The append API deliberately consumes ledger-shaped per-layer deltas rather
than a second full cache.  A caller therefore cannot accidentally publish a
mutated prefix while claiming that it is an append.  Page chunks are storage
segments, not runtime ``KVPage`` objects: decode-sized deltas may be shorter
than the runtime block size.  Exact hydration concatenates those immutable
segments and repacks them into the normal page-native runtime ABI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from mdb.ar.observe import tensor_sha256
from mdb.ar.paged_kv import PAGED_KV_SCHEMA, PagedKVCache
from mdb.qwen2_machine import KVLayerState, NormalizedKVCache

PERSISTENT_KV_METADATA_SCHEMA = "saturn-ar-persistent-kv-metadata-v1"
PERSISTENT_KV_PAGE_SCHEMA = "saturn-ar-persistent-kv-page-v1"
PERSISTENT_KV_ROOT_SCHEMA = "saturn-ar-persistent-kv-root-v1"
PERSISTENT_KV_CLOSURE_SCHEMA = "saturn-ar-persistent-kv-closure-v1"
PERSISTENT_KV_LAYOUT = "batch,kv_head,token,head_dim"
PERSISTENT_KV_NUMERICAL_CONTRACT = "bit-exact"

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class PersistentKVHistoryError(ValueError):
    """Raised when a KV history object or its reachable closure is invalid."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PersistentKVHistoryError("persistent KV metadata is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(value))


def _require_digest(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise PersistentKVHistoryError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PersistentKVHistoryError(f"{field_name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PersistentKVHistoryError(f"{field_name} must be a non-negative integer")
    return value


def _require_torch_dtype(value: Any, *, field_name: str = "dtype") -> str:
    if not isinstance(value, str) or not value.startswith("torch."):
        raise PersistentKVHistoryError(f"{field_name} must name a torch dtype")
    resolved = getattr(torch, value.removeprefix("torch."), None)
    if not isinstance(resolved, torch.dtype):
        raise PersistentKVHistoryError(f"{field_name} names an unsupported torch dtype")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise PersistentKVHistoryError(f"{label} schema mismatch: {', '.join(details)}")


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PersistentKVHistoryError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=object_pairs)
    except PersistentKVHistoryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersistentKVHistoryError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PersistentKVHistoryError(f"{label} must be a JSON object")
    return value


def _complete_normalized_cache(
    cache: NormalizedKVCache | PagedKVCache,
) -> NormalizedKVCache:
    normalized = cache.to_normalized() if isinstance(cache, PagedKVCache) else cache
    if not isinstance(normalized, NormalizedKVCache):
        raise PersistentKVHistoryError("expected a NormalizedKVCache or PagedKVCache")
    if not normalized.layers or any(layer is None for layer in normalized.layers):
        raise PersistentKVHistoryError("persistent KV history requires a complete cache")
    return normalized


@dataclass(frozen=True, slots=True)
class PersistentKVMetadata:
    """Typed contract stored once at the genesis of a KV history."""

    model_fingerprint: str
    layer_count: int
    block_size: int
    batch_size: int
    kv_head_count: int
    head_dim: int
    dtype: str
    cache_schema: str = PAGED_KV_SCHEMA
    layout: str = PERSISTENT_KV_LAYOUT
    numerical_contract: str = PERSISTENT_KV_NUMERICAL_CONTRACT
    schema: str = PERSISTENT_KV_METADATA_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERSISTENT_KV_METADATA_SCHEMA:
            raise PersistentKVHistoryError(f"unsupported persistent KV metadata {self.schema!r}")
        _require_digest(self.model_fingerprint, field_name="model_fingerprint")
        for name in (
            "layer_count",
            "block_size",
            "batch_size",
            "kv_head_count",
            "head_dim",
        ):
            _require_positive_int(getattr(self, name), field_name=name)
        _require_torch_dtype(self.dtype)
        if self.cache_schema != PAGED_KV_SCHEMA:
            raise PersistentKVHistoryError(
                f"persistent KV history requires cache schema {PAGED_KV_SCHEMA!r}"
            )
        if self.layout != PERSISTENT_KV_LAYOUT:
            raise PersistentKVHistoryError(
                f"persistent KV history requires layout {PERSISTENT_KV_LAYOUT!r}"
            )
        if self.numerical_contract != PERSISTENT_KV_NUMERICAL_CONTRACT:
            raise PersistentKVHistoryError("persistent KV history must be bit-exact")

    @classmethod
    def from_cache(
        cls,
        cache: NormalizedKVCache | PagedKVCache,
        *,
        model_fingerprint: str,
        block_size: int | None = None,
    ) -> PersistentKVMetadata:
        normalized = _complete_normalized_cache(cache)
        resolved_block_size = cache.block_size if isinstance(cache, PagedKVCache) else block_size
        if resolved_block_size is None:
            raise PersistentKVHistoryError("block_size is required for a normalized cache")
        layers = tuple(layer for layer in normalized.layers if layer is not None)
        first = layers[0]
        shape = tuple(int(value) for value in first.key.shape)
        if len(shape) != 4:
            raise PersistentKVHistoryError("KV tensors must be rank four")
        for layer in layers:
            if layer.key.shape != first.key.shape or layer.value.shape != first.value.shape:
                raise PersistentKVHistoryError("all KV layers must have identical geometry")
            if layer.key.dtype != first.key.dtype or layer.value.dtype != first.value.dtype:
                raise PersistentKVHistoryError("all KV layers must have one dtype")
        return cls(
            model_fingerprint=model_fingerprint,
            layer_count=len(layers),
            block_size=resolved_block_size,
            batch_size=shape[0],
            kv_head_count=shape[1],
            head_dim=shape[3],
            dtype=str(first.key.dtype),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PersistentKVMetadata:
        _require_exact_keys(
            value,
            required=frozenset(
                {
                    "schema",
                    "model_fingerprint",
                    "layer_count",
                    "block_size",
                    "batch_size",
                    "kv_head_count",
                    "head_dim",
                    "dtype",
                    "cache_schema",
                    "layout",
                    "numerical_contract",
                    "fingerprint",
                }
            ),
            label="persistent KV metadata",
        )
        result = cls(
            schema=value["schema"],
            model_fingerprint=value["model_fingerprint"],
            layer_count=value["layer_count"],
            block_size=value["block_size"],
            batch_size=value["batch_size"],
            kv_head_count=value["kv_head_count"],
            head_dim=value["head_dim"],
            dtype=value["dtype"],
            cache_schema=value["cache_schema"],
            layout=value["layout"],
            numerical_contract=value["numerical_contract"],
        )
        supplied = _require_digest(value["fingerprint"], field_name="metadata fingerprint")
        if supplied != result.fingerprint:
            raise PersistentKVHistoryError("persistent KV metadata fingerprint mismatch")
        return result

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_fingerprint": self.model_fingerprint,
            "layer_count": self.layer_count,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "kv_head_count": self.kv_head_count,
            "head_dim": self.head_dim,
            "dtype": self.dtype,
            "cache_schema": self.cache_schema,
            "layout": self.layout,
            "numerical_contract": self.numerical_contract,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class PersistentKVPageRef:
    """Typed reference to one immutable key/value page artifact."""

    shape: tuple[int, int, int, int]
    dtype: str
    key_sha256: str
    value_sha256: str
    artifact_sha256: str
    artifact_bytes: int
    content_sha256: str = field(init=False)
    schema: str = PERSISTENT_KV_PAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PERSISTENT_KV_PAGE_SCHEMA:
            raise PersistentKVHistoryError(f"unsupported persistent KV page {self.schema!r}")
        shape = tuple(self.shape)
        if len(shape) != 4:
            raise PersistentKVHistoryError("persistent KV page shape must be rank four")
        for index, value in enumerate(shape):
            _require_positive_int(value, field_name=f"page shape[{index}]")
        object.__setattr__(self, "shape", shape)
        _require_torch_dtype(self.dtype, field_name="page dtype")
        _require_digest(self.key_sha256, field_name="page key_sha256")
        _require_digest(self.value_sha256, field_name="page value_sha256")
        _require_digest(self.artifact_sha256, field_name="page artifact_sha256")
        _require_positive_int(self.artifact_bytes, field_name="page artifact_bytes")
        object.__setattr__(self, "content_sha256", _canonical_digest(self._content_payload()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PersistentKVPageRef:
        _require_exact_keys(
            value,
            required=frozenset(
                {
                    "schema",
                    "content_sha256",
                    "shape",
                    "dtype",
                    "key_sha256",
                    "value_sha256",
                    "artifact_sha256",
                    "artifact_bytes",
                }
            ),
            label="persistent KV page reference",
        )
        raw_shape = value["shape"]
        if not isinstance(raw_shape, list):
            raise PersistentKVHistoryError("persistent KV page shape must be an array")
        result = cls(
            schema=value["schema"],
            shape=tuple(raw_shape),
            dtype=value["dtype"],
            key_sha256=value["key_sha256"],
            value_sha256=value["value_sha256"],
            artifact_sha256=value["artifact_sha256"],
            artifact_bytes=value["artifact_bytes"],
        )
        supplied = _require_digest(value["content_sha256"], field_name="page content_sha256")
        if supplied != result.content_sha256:
            raise PersistentKVHistoryError("persistent KV page content fingerprint mismatch")
        return result

    @property
    def length(self) -> int:
        return self.shape[2]

    @property
    def tensor_bytes(self) -> int:
        dtype = getattr(torch, self.dtype.removeprefix("torch."))
        element_size = torch.empty((), dtype=dtype).element_size()
        elements = self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]
        return 2 * elements * element_size

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "key_sha256": self.key_sha256,
            "value_sha256": self.value_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_payload(),
            "content_sha256": self.content_sha256,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
        }


@dataclass(frozen=True, slots=True)
class PersistentKVLayerAppend:
    """One layer's ordered page refs for a single root delta."""

    layer_index: int
    start: int
    append_length: int
    pages: tuple[PersistentKVPageRef, ...]

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.layer_index, field_name="layer_index")
        _require_nonnegative_int(self.start, field_name="layer append start")
        _require_positive_int(self.append_length, field_name="layer append_length")
        pages = tuple(self.pages)
        if not pages or not all(isinstance(page, PersistentKVPageRef) for page in pages):
            raise PersistentKVHistoryError("layer append requires typed page references")
        if sum(page.length for page in pages) != self.append_length:
            raise PersistentKVHistoryError("layer page lengths do not equal append_length")
        object.__setattr__(self, "pages", pages)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PersistentKVLayerAppend:
        _require_exact_keys(
            value,
            required=frozenset({"layer_index", "start", "append_length", "pages"}),
            label="persistent KV layer append",
        )
        raw_pages = value["pages"]
        if not isinstance(raw_pages, list) or not all(isinstance(page, dict) for page in raw_pages):
            raise PersistentKVHistoryError("layer append pages must be an array of objects")
        return cls(
            layer_index=value["layer_index"],
            start=value["start"],
            append_length=value["append_length"],
            pages=tuple(PersistentKVPageRef.from_dict(page) for page in raw_pages),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "start": self.start,
            "append_length": self.append_length,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass(frozen=True, slots=True)
class PersistentKVRootManifest:
    """One immutable Merkle node containing only a KV-history append diff."""

    metadata_sha256: str
    parent_root_sha256: str | None
    parent_sequence_length: int
    append_length: int
    sequence_length: int
    layer_appends: tuple[PersistentKVLayerAppend, ...]
    metadata: PersistentKVMetadata | None = None
    schema: str = PERSISTENT_KV_ROOT_SCHEMA
    root_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != PERSISTENT_KV_ROOT_SCHEMA:
            raise PersistentKVHistoryError(f"unsupported persistent KV root {self.schema!r}")
        _require_digest(self.metadata_sha256, field_name="metadata_sha256")
        if self.parent_root_sha256 is not None:
            _require_digest(self.parent_root_sha256, field_name="parent_root_sha256")
        _require_nonnegative_int(
            self.parent_sequence_length,
            field_name="parent_sequence_length",
        )
        _require_positive_int(self.append_length, field_name="root append_length")
        _require_positive_int(self.sequence_length, field_name="root sequence_length")
        if self.sequence_length != self.parent_sequence_length + self.append_length:
            raise PersistentKVHistoryError("root sequence length does not close over its append")
        appends = tuple(self.layer_appends)
        indices = tuple(layer.layer_index for layer in appends)
        if not appends or indices != tuple(sorted(set(indices))):
            raise PersistentKVHistoryError("root layer appends must be unique and sorted")
        for layer in appends:
            if layer.start != self.parent_sequence_length:
                raise PersistentKVHistoryError("layer append start disagrees with parent length")
            if layer.append_length != self.append_length:
                raise PersistentKVHistoryError("layer append length disagrees with root length")
        object.__setattr__(self, "layer_appends", appends)
        if self.parent_root_sha256 is None:
            if self.parent_sequence_length != 0 or self.metadata is None:
                raise PersistentKVHistoryError(
                    "genesis root requires typed metadata and zero parent"
                )
            if self.metadata.fingerprint != self.metadata_sha256:
                raise PersistentKVHistoryError("genesis root metadata fingerprint mismatch")
        elif self.metadata is not None:
            raise PersistentKVHistoryError("child roots must inherit metadata by fingerprint")
        object.__setattr__(self, "root_sha256", _canonical_digest(self._payload()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PersistentKVRootManifest:
        _require_exact_keys(
            value,
            required=frozenset(
                {
                    "schema",
                    "root_sha256",
                    "metadata_sha256",
                    "parent_root_sha256",
                    "parent_sequence_length",
                    "append_length",
                    "sequence_length",
                    "layer_appends",
                }
            ),
            optional=frozenset({"metadata"}),
            label="persistent KV root manifest",
        )
        raw_appends = value["layer_appends"]
        if not isinstance(raw_appends, list) or not all(
            isinstance(layer, dict) for layer in raw_appends
        ):
            raise PersistentKVHistoryError("root layer_appends must be an array of objects")
        raw_metadata = value.get("metadata")
        if raw_metadata is not None and not isinstance(raw_metadata, dict):
            raise PersistentKVHistoryError("root metadata must be an object when present")
        result = cls(
            schema=value["schema"],
            metadata_sha256=value["metadata_sha256"],
            parent_root_sha256=value["parent_root_sha256"],
            parent_sequence_length=value["parent_sequence_length"],
            append_length=value["append_length"],
            sequence_length=value["sequence_length"],
            layer_appends=tuple(
                PersistentKVLayerAppend.from_dict(layer) for layer in raw_appends
            ),
            metadata=(
                None if raw_metadata is None else PersistentKVMetadata.from_dict(raw_metadata)
            ),
        )
        supplied = _require_digest(value["root_sha256"], field_name="root_sha256")
        if supplied != result.root_sha256:
            raise PersistentKVHistoryError("persistent KV root fingerprint mismatch")
        return result

    def _payload(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "metadata_sha256": self.metadata_sha256,
            "parent_root_sha256": self.parent_root_sha256,
            "parent_sequence_length": self.parent_sequence_length,
            "append_length": self.append_length,
            "sequence_length": self.sequence_length,
            "layer_appends": [layer.to_dict() for layer in self.layer_appends],
        }
        if self.metadata is not None:
            result["metadata"] = self.metadata.to_dict()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "root_sha256": self.root_sha256}


@dataclass(frozen=True, slots=True)
class PersistentKVClosureReceipt:
    """Tensor-free evidence for one fully verified Merkle closure."""

    root_sha256: str
    metadata_sha256: str
    sequence_length: int
    root_count: int
    page_reference_count: int
    unique_page_count: int
    unique_page_artifact_bytes: int
    unique_page_tensor_bytes: int
    schema: str = PERSISTENT_KV_CLOSURE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "root_sha256": self.root_sha256,
            "metadata_sha256": self.metadata_sha256,
            "sequence_length": self.sequence_length,
            "root_count": self.root_count,
            "page_reference_count": self.page_reference_count,
            "unique_page_count": self.unique_page_count,
            "unique_page_artifact_bytes": self.unique_page_artifact_bytes,
            "unique_page_tensor_bytes": self.unique_page_tensor_bytes,
            "storage": "parent-root-plus-appended-page-refs",
            "numerical_contract": PERSISTENT_KV_NUMERICAL_CONTRACT,
        }


@dataclass(frozen=True, slots=True)
class HydratedPersistentKV:
    """An exactly hydrated page-native cache and its verified custody receipt."""

    cache: PagedKVCache
    metadata: PersistentKVMetadata
    receipt: PersistentKVClosureReceipt
    root_chain: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ClosureIndex:
    manifests: tuple[PersistentKVRootManifest, ...]
    metadata: PersistentKVMetadata


@dataclass(frozen=True, slots=True)
class _VerifiedClosure:
    index: _ClosureIndex
    page_tensors: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    receipt: PersistentKVClosureReceipt


class PersistentKVHistoryStore:
    """Filesystem CAS for persistent KV page objects and Merkle roots.

    The store never overwrites an object.  An existing content path must match
    the candidate exactly, otherwise publication fails.  A verified parent is
    cached only as tensor-free metadata so repeated appends in one resident
    process are O(new delta); public verification and hydration always reread
    and rehash the durable closure to detect out-of-band tampering.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.pages_path = self.path / "pages"
        self.roots_path = self.path / "roots"
        self._verified_indices: dict[str, _ClosureIndex] = {}

    def page_path(self, content_sha256: str) -> Path:
        digest = _require_digest(content_sha256, field_name="page content_sha256")
        return self.pages_path / f"{digest}.safetensors"

    def root_path(self, root_sha256: str) -> Path:
        digest = _require_digest(root_sha256, field_name="root_sha256")
        return self.roots_path / f"{digest}.json"

    def publish_genesis(
        self,
        cache: NormalizedKVCache | PagedKVCache,
        *,
        metadata: PersistentKVMetadata,
    ) -> PersistentKVRootManifest:
        normalized = _complete_normalized_cache(cache)
        appends = {
            index: (layer.key, layer.value)
            for index, layer in enumerate(normalized.layers)
            if layer is not None
        }
        return self._publish(appends, metadata=metadata, parent_index=None)

    def append(
        self,
        parent_root_sha256: str,
        appends: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> PersistentKVRootManifest:
        parent_digest = _require_digest(parent_root_sha256, field_name="parent_root_sha256")
        parent_index = self._verified_indices.get(parent_digest)
        if parent_index is None:
            # Reopened stores earn the fast path once by verifying bytes, not
            # merely by trusting that hash-shaped filenames are a closure.
            parent_index = self._verify_and_load(parent_digest).index
        return self._publish(appends, metadata=None, parent_index=parent_index)

    def load_root(self, root_sha256: str) -> PersistentKVRootManifest:
        digest = _require_digest(root_sha256, field_name="root_sha256")
        path = self.root_path(digest)
        payload = self._read_regular_file(path, label=f"KV root {digest}")
        raw = _strict_json_object(payload, label=f"KV root {digest}")
        manifest = PersistentKVRootManifest.from_dict(raw)
        if manifest.root_sha256 != digest:
            raise PersistentKVHistoryError("KV root path does not match manifest identity")
        if payload != _canonical_json_bytes(manifest.to_dict()):
            raise PersistentKVHistoryError("KV root artifact is not canonical immutable JSON")
        return manifest

    def verify(self, root_sha256: str) -> PersistentKVClosureReceipt:
        """Reread and verify every root and unique page reachable from ``root``."""

        return self._verify_and_load(root_sha256).receipt

    def hydrate(
        self,
        root_sha256: str,
        *,
        device: torch.device | str | None = None,
    ) -> HydratedPersistentKV:
        """Verify the complete closure, then reconstruct an exact paged cache."""

        verified = self._verify_and_load(root_sha256)
        index = verified.index
        by_layer: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(index.metadata.layer_count)
        ]
        for manifest in index.manifests:
            for layer_append in manifest.layer_appends:
                for page in layer_append.pages:
                    by_layer[layer_append.layer_index].append(
                        verified.page_tensors[page.content_sha256]
                    )

        layers: list[KVLayerState] = []
        for layer_index, parts in enumerate(by_layer):
            if not parts:
                raise PersistentKVHistoryError(
                    f"verified closure contains no pages for layer {layer_index}"
                )
            if len(parts) == 1:
                key, value = parts[0]
            else:
                key = torch.cat(tuple(part[0] for part in parts), dim=-2)
                value = torch.cat(tuple(part[1] for part in parts), dim=-2)
            if device is not None:
                key = key.to(device)
                value = value.to(device)
            layers.append(KVLayerState(key, value))
        normalized = NormalizedKVCache(tuple(layers))
        cache = PagedKVCache.from_normalized(
            normalized,
            block_size=index.metadata.block_size,
        )
        if cache.common_sequence_length() != verified.receipt.sequence_length:
            raise PersistentKVHistoryError("hydrated cache length disagrees with verified root")
        return HydratedPersistentKV(
            cache=cache,
            metadata=index.metadata,
            receipt=verified.receipt,
            root_chain=tuple(manifest.root_sha256 for manifest in index.manifests),
        )

    def _publish(
        self,
        appends: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        *,
        metadata: PersistentKVMetadata | None,
        parent_index: _ClosureIndex | None,
    ) -> PersistentKVRootManifest:
        if parent_index is None:
            if metadata is None:
                raise PersistentKVHistoryError("genesis publication requires metadata")
            parent_manifest = None
            resolved_metadata = metadata
            parent_length = 0
        else:
            if metadata is not None:
                raise PersistentKVHistoryError("child publication inherits parent metadata")
            parent_manifest = parent_index.manifests[-1]
            resolved_metadata = parent_index.metadata
            parent_length = parent_manifest.sequence_length

        normalized, append_length = self._normalize_appends(appends, resolved_metadata)
        layer_appends: list[PersistentKVLayerAppend] = []
        for layer_index in range(resolved_metadata.layer_count):
            key, value = normalized[layer_index]
            refs = tuple(
                self._publish_page(
                    key[..., start : start + resolved_metadata.block_size, :],
                    value[..., start : start + resolved_metadata.block_size, :],
                )
                for start in range(0, append_length, resolved_metadata.block_size)
            )
            layer_appends.append(
                PersistentKVLayerAppend(
                    layer_index=layer_index,
                    start=parent_length,
                    append_length=append_length,
                    pages=refs,
                )
            )

        manifest = PersistentKVRootManifest(
            metadata_sha256=resolved_metadata.fingerprint,
            metadata=resolved_metadata if parent_manifest is None else None,
            parent_root_sha256=(
                None if parent_manifest is None else parent_manifest.root_sha256
            ),
            parent_sequence_length=parent_length,
            append_length=append_length,
            sequence_length=parent_length + append_length,
            layer_appends=tuple(layer_appends),
        )
        self._publish_immutable(
            self.root_path(manifest.root_sha256),
            _canonical_json_bytes(manifest.to_dict()),
            label="persistent KV root",
        )
        manifests = (
            (manifest,)
            if parent_index is None
            else (*parent_index.manifests, manifest)
        )
        self._verified_indices[manifest.root_sha256] = _ClosureIndex(
            manifests=manifests,
            metadata=resolved_metadata,
        )
        return manifest

    @staticmethod
    def _normalize_appends(
        appends: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        metadata: PersistentKVMetadata,
    ) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], int]:
        if not isinstance(appends, Mapping):
            raise PersistentKVHistoryError("KV append must be a layer-indexed mapping")
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in appends):
            raise PersistentKVHistoryError("KV append layer indices must be integers")
        expected_layers = set(range(metadata.layer_count))
        if set(appends) != expected_layers:
            raise PersistentKVHistoryError(
                "KV append must provide exactly one delta for every declared layer"
            )
        normalized: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        append_length: int | None = None
        for layer_index in range(metadata.layer_count):
            pair = appends[layer_index]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise PersistentKVHistoryError("each KV layer append must be a key/value pair")
            key, value = pair
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise PersistentKVHistoryError("KV append key/value payloads must be tensors")
            if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
                raise PersistentKVHistoryError("KV append tensors must be matching rank four")
            shape = tuple(int(item) for item in key.shape)
            if shape[0] != metadata.batch_size:
                raise PersistentKVHistoryError("KV append batch size disagrees with metadata")
            if shape[1] != metadata.kv_head_count or shape[3] != metadata.head_dim:
                raise PersistentKVHistoryError("KV append head geometry disagrees with metadata")
            if str(key.dtype) != metadata.dtype or value.dtype != key.dtype:
                raise PersistentKVHistoryError("KV append dtype disagrees with metadata")
            if key.device != value.device:
                raise PersistentKVHistoryError("KV append key/value devices must match")
            length = shape[2]
            _require_positive_int(length, field_name="KV append token length")
            if append_length is None:
                append_length = length
            elif length != append_length:
                raise PersistentKVHistoryError("all KV layers must append the same token count")
            normalized[layer_index] = (
                key.detach().to(device="cpu").contiguous().clone(),
                value.detach().to(device="cpu").contiguous().clone(),
            )
        if append_length is None:  # pragma: no cover - layer_count is positive
            raise PersistentKVHistoryError("KV append is empty")
        return normalized, append_length

    def _publish_page(self, key: torch.Tensor, value: torch.Tensor) -> PersistentKVPageRef:
        if key.device.type != "cpu" or value.device.type != "cpu":
            raise PersistentKVHistoryError("persistent KV pages must be staged on CPU")
        # Token-axis slices are generally non-contiguous even when their source
        # delta is packed.  Each CAS artifact owns dense immutable bytes.
        key = key.contiguous()
        value = value.contiguous()
        artifact = save_safetensors({"key": key, "value": value})
        candidate = PersistentKVPageRef(
            shape=tuple(int(item) for item in key.shape),
            dtype=str(key.dtype),
            key_sha256=tensor_sha256(key),
            value_sha256=tensor_sha256(value),
            artifact_sha256=_sha256(artifact),
            artifact_bytes=len(artifact),
        )
        path = self.page_path(candidate.content_sha256)
        if path.exists() or path.is_symlink():
            existing = self._reference_for_existing_page(path, candidate)
            self._load_page(existing)
            return existing
        self._publish_immutable(path, artifact, label="persistent KV page")
        # Reopen before making the page reachable from a root.  Publication is
        # therefore transactional at the root boundary even on flaky storage.
        self._load_page(candidate)
        return candidate

    def _reference_for_existing_page(
        self,
        path: Path,
        candidate: PersistentKVPageRef,
    ) -> PersistentKVPageRef:
        artifact = self._read_regular_file(path, label=f"KV page {candidate.content_sha256}")
        return PersistentKVPageRef(
            shape=candidate.shape,
            dtype=candidate.dtype,
            key_sha256=candidate.key_sha256,
            value_sha256=candidate.value_sha256,
            artifact_sha256=_sha256(artifact),
            artifact_bytes=len(artifact),
        )

    def _load_manifest_chain(self, root_sha256: str) -> _ClosureIndex:
        target = _require_digest(root_sha256, field_name="root_sha256")
        reversed_chain: list[PersistentKVRootManifest] = []
        seen: set[str] = set()
        cursor: str | None = target
        while cursor is not None:
            if cursor in seen:
                raise PersistentKVHistoryError("persistent KV root closure contains a cycle")
            seen.add(cursor)
            manifest = self.load_root(cursor)
            reversed_chain.append(manifest)
            cursor = manifest.parent_root_sha256
        manifests = tuple(reversed(reversed_chain))
        genesis = manifests[0]
        if genesis.metadata is None:
            raise PersistentKVHistoryError("persistent KV closure has no typed genesis metadata")
        metadata = genesis.metadata
        expected_parent: str | None = None
        expected_length = 0
        canonical_refs: dict[str, PersistentKVPageRef] = {}
        for manifest in manifests:
            if manifest.parent_root_sha256 != expected_parent:
                raise PersistentKVHistoryError("persistent KV root chain is not contiguous")
            if manifest.parent_sequence_length != expected_length:
                raise PersistentKVHistoryError("persistent KV root sequence closure is invalid")
            if manifest.metadata_sha256 != metadata.fingerprint:
                raise PersistentKVHistoryError("persistent KV root changed inherited metadata")
            if tuple(layer.layer_index for layer in manifest.layer_appends) != tuple(
                range(metadata.layer_count)
            ):
                raise PersistentKVHistoryError(
                    "persistent KV root does not append every declared layer"
                )
            for layer in manifest.layer_appends:
                for page in layer.pages:
                    if page.length > metadata.block_size:
                        raise PersistentKVHistoryError("persistent KV page exceeds block size")
                    if page.shape[0] != metadata.batch_size:
                        raise PersistentKVHistoryError("persistent KV page batch mismatch")
                    if page.shape[1] != metadata.kv_head_count:
                        raise PersistentKVHistoryError("persistent KV page head-count mismatch")
                    if page.shape[3] != metadata.head_dim or page.dtype != metadata.dtype:
                        raise PersistentKVHistoryError("persistent KV page type mismatch")
                    previous = canonical_refs.setdefault(page.content_sha256, page)
                    if previous.to_dict() != page.to_dict():
                        raise PersistentKVHistoryError(
                            "one page identity has inconsistent artifact metadata"
                        )
                    page_path = self.page_path(page.content_sha256)
                    if page_path.is_symlink() or not page_path.is_file():
                        raise PersistentKVHistoryError(
                            f"persistent KV closure is missing page {page.content_sha256}"
                        )
            expected_parent = manifest.root_sha256
            expected_length = manifest.sequence_length
        if expected_length != manifests[-1].sequence_length:
            raise PersistentKVHistoryError("persistent KV target length is invalid")
        return _ClosureIndex(manifests=manifests, metadata=metadata)

    def _verify_and_load(self, root_sha256: str) -> _VerifiedClosure:
        index = self._load_manifest_chain(root_sha256)
        page_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        refs: dict[str, PersistentKVPageRef] = {}
        page_reference_count = 0
        for manifest in index.manifests:
            for layer in manifest.layer_appends:
                for ref in layer.pages:
                    page_reference_count += 1
                    refs.setdefault(ref.content_sha256, ref)
                    if ref.content_sha256 not in page_tensors:
                        page_tensors[ref.content_sha256] = self._load_page(ref)
        target = index.manifests[-1]
        receipt = PersistentKVClosureReceipt(
            root_sha256=target.root_sha256,
            metadata_sha256=index.metadata.fingerprint,
            sequence_length=target.sequence_length,
            root_count=len(index.manifests),
            page_reference_count=page_reference_count,
            unique_page_count=len(refs),
            unique_page_artifact_bytes=sum(ref.artifact_bytes for ref in refs.values()),
            unique_page_tensor_bytes=sum(ref.tensor_bytes for ref in refs.values()),
        )
        self._verified_indices[target.root_sha256] = index
        return _VerifiedClosure(index=index, page_tensors=page_tensors, receipt=receipt)

    def _load_page(self, ref: PersistentKVPageRef) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.page_path(ref.content_sha256)
        artifact = self._read_regular_file(path, label=f"KV page {ref.content_sha256}")
        if len(artifact) != ref.artifact_bytes:
            raise PersistentKVHistoryError("persistent KV page artifact size mismatch")
        if _sha256(artifact) != ref.artifact_sha256:
            raise PersistentKVHistoryError("persistent KV page artifact fingerprint mismatch")
        try:
            tensors = load_safetensors(artifact)
        except (SafetensorError, RuntimeError, TypeError, ValueError) as exc:
            raise PersistentKVHistoryError("persistent KV page is not valid safetensors") from exc
        if set(tensors) != {"key", "value"}:
            raise PersistentKVHistoryError("persistent KV page must contain only key and value")
        key = tensors["key"]
        value = tensors["value"]
        if key.shape != value.shape or tuple(int(item) for item in key.shape) != ref.shape:
            raise PersistentKVHistoryError("persistent KV page tensor shape mismatch")
        if str(key.dtype) != ref.dtype or value.dtype != key.dtype:
            raise PersistentKVHistoryError("persistent KV page tensor dtype mismatch")
        if tensor_sha256(key) != ref.key_sha256:
            raise PersistentKVHistoryError("persistent KV page key fingerprint mismatch")
        if tensor_sha256(value) != ref.value_sha256:
            raise PersistentKVHistoryError("persistent KV page value fingerprint mismatch")
        return key, value

    @staticmethod
    def _read_regular_file(path: Path, *, label: str) -> bytes:
        if path.is_symlink() or not path.is_file():
            raise PersistentKVHistoryError(f"{label} is missing or not a regular file")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise PersistentKVHistoryError(f"cannot read {label}") from exc

    @staticmethod
    def _publish_immutable(path: Path, payload: bytes, *, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            existing = PersistentKVHistoryStore._read_regular_file(path, label=label)
            if existing != payload:
                raise PersistentKVHistoryError(f"{label} content address is already occupied")
            return

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                existing = PersistentKVHistoryStore._read_regular_file(path, label=label)
                if existing != payload:
                    raise PersistentKVHistoryError(
                        f"{label} content address was concurrently occupied"
                    ) from None
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise PersistentKVHistoryError(f"cannot publish immutable {label}") from exc
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass


__all__ = [
    "HydratedPersistentKV",
    "PERSISTENT_KV_CLOSURE_SCHEMA",
    "PERSISTENT_KV_LAYOUT",
    "PERSISTENT_KV_METADATA_SCHEMA",
    "PERSISTENT_KV_NUMERICAL_CONTRACT",
    "PERSISTENT_KV_PAGE_SCHEMA",
    "PERSISTENT_KV_ROOT_SCHEMA",
    "PersistentKVClosureReceipt",
    "PersistentKVHistoryError",
    "PersistentKVHistoryStore",
    "PersistentKVLayerAppend",
    "PersistentKVMetadata",
    "PersistentKVPageRef",
    "PersistentKVRootManifest",
]

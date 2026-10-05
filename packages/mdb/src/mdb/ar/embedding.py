"""Pluggable input-embedding boundary for the dense AR lane.

The decoder stack consumes a residual carrier.  Token IDs are one way to
produce that carrier, not a requirement of the downstream layers.  This
module keeps the default ``embed_tokens`` path authoritative while making the
boundary explicit enough to accept cached or externally supplied carriers.

There are two deliberately different promises for supplied carriers:

* ``verify_supplied=True`` computes the native row and requires
  ``torch.equal`` before accepting the supplied tensor.  This is the proof
  path and is bit exact, but it does not save the embedding lookup.
* ``verify_supplied=False`` bypasses the lookup.  It is a performance path,
  but its exactness is an input property and is reported as such.

``CachedEmbeddingProvider`` only caches outputs that have already earned the
native exactness receipt.  The cache is therefore an optimization, not a new
numerical authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol

import torch

from mdb.tensor_hash import tensor_bytes_sha256

EMBEDDING_BOUNDARY_SCHEMA = "saturn-ar-embedding-boundary-v1"
EMBEDDING_RECEIPT_SCHEMA = "saturn-ar-embedding-receipt-v1"
EMBEDDING_CONTENT_FINGERPRINT = "content-sha256-v1"
EMBEDDING_RUNTIME_DESCRIPTOR = "runtime-descriptor-v1"


class EmbeddingBoundaryError(ValueError):
    """Raised when an embedding carrier violates the AR boundary."""


class EmbeddingFingerprintPolicy(str, Enum):
    """Evidence strength for token/residual boundary receipts."""

    CONTENT = EMBEDDING_CONTENT_FINGERPRINT
    RUNTIME_DESCRIPTOR = EMBEDDING_RUNTIME_DESCRIPTOR


def _normalise_fingerprint_policy(
    value: EmbeddingFingerprintPolicy | str,
) -> EmbeddingFingerprintPolicy:
    try:
        return EmbeddingFingerprintPolicy(value)
    except ValueError as exc:
        raise EmbeddingBoundaryError(f"unsupported embedding fingerprint policy {value!r}") from exc


def _tensor_fingerprint(value: torch.Tensor) -> str:
    """Return a deterministic fingerprint without exposing tensor payloads."""

    tensor = value.detach()
    return hashlib.sha256(
        (
            f"{tensor.dtype}|{tuple(int(dim) for dim in tensor.shape)}|"
            f"{tensor_bytes_sha256(tensor)}"
        ).encode()
    ).hexdigest()


def _tensor_runtime_descriptor(value: torch.Tensor) -> str:
    """Return a process-local tensor descriptor without reading payload bytes."""

    storage = value.untyped_storage()
    try:
        version = int(value._version)
    except RuntimeError:
        version = None
    payload = {
        "schema": EMBEDDING_RUNTIME_DESCRIPTOR,
        "device": str(value.device),
        "dtype": str(value.dtype),
        "shape": [int(dim) for dim in value.shape],
        "stride": [int(item) for item in value.stride()],
        "storage_offset": int(value.storage_offset()),
        "storage_data_ptr": int(storage.data_ptr()),
        "storage_nbytes": int(storage.nbytes()),
        "version": version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingReceipt:
    """Tensor-free evidence for one token-ID/residual boundary crossing."""

    mode: str
    exact: bool
    provider: str
    input_ids_fingerprint: str
    embedding_fingerprint: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    verified: bool = False
    cache_hit: bool = False
    fingerprint_policy: str = EMBEDDING_CONTENT_FINGERPRINT
    fingerprints_durable: bool = True

    def __post_init__(self) -> None:
        policy = _normalise_fingerprint_policy(self.fingerprint_policy)
        object.__setattr__(self, "fingerprint_policy", policy.value)
        expected_durable = policy is EmbeddingFingerprintPolicy.CONTENT
        if self.fingerprints_durable is not expected_durable:
            raise EmbeddingBoundaryError(
                "embedding fingerprint durability does not match its declared policy"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EMBEDDING_RECEIPT_SCHEMA,
            "mode": self.mode,
            "exact": self.exact,
            "provider": self.provider,
            "input_ids_fingerprint": self.input_ids_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "verified": self.verified,
            "cache_hit": self.cache_hit,
            "fingerprint_policy": self.fingerprint_policy,
            "fingerprints_durable": self.fingerprints_durable,
        }


class EmbeddingProvider(Protocol):
    """Structural interface for a token-ID to residual-carrier provider."""

    last_receipt: EmbeddingReceipt | None

    def embed(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the residual carrier consumed by the first decoder layer."""

    def describe(self) -> Mapping[str, Any]:
        """Return a stable, tensor-free provider descriptor."""


def _validate_input_ids(input_ids: torch.Tensor) -> None:
    if not isinstance(input_ids, torch.Tensor):
        raise EmbeddingBoundaryError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise EmbeddingBoundaryError("input_ids must have shape [batch, non_empty_sequence]")
    if input_ids.dtype != torch.long:
        raise EmbeddingBoundaryError("input_ids must use torch.long")


class NativeEmbeddingProvider:
    """Authoritative wrapper around a resident ``embed_tokens`` module."""

    provider_name = "native-embed_tokens"

    def __init__(
        self,
        module: Any,
        *,
        verify_supplied: bool = True,
        provider_name: str = provider_name,
        fingerprint_policy: EmbeddingFingerprintPolicy | str = (
            EmbeddingFingerprintPolicy.CONTENT
        ),
    ) -> None:
        if not callable(module):
            raise EmbeddingBoundaryError("native embedding module must be callable")
        self.module = module
        self.verify_supplied = bool(verify_supplied)
        self.provider_name = str(provider_name)
        self.fingerprint_policy = _normalise_fingerprint_policy(fingerprint_policy)
        self.last_receipt: EmbeddingReceipt | None = None

    def _fingerprint(self, value: torch.Tensor) -> str:
        if self.fingerprint_policy is EmbeddingFingerprintPolicy.CONTENT:
            return _tensor_fingerprint(value)
        return _tensor_runtime_descriptor(value)

    def _module_layout(self) -> tuple[torch.device, torch.dtype, int | None] | None:
        """Best-effort layout discovery for standard HF embedding modules."""

        parameter = (
            next(iter(self.module.parameters()), None)
            if hasattr(self.module, "parameters")
            else None
        )
        if parameter is None:
            return None
        hidden_size = int(parameter.shape[-1]) if parameter.ndim >= 2 else None
        return parameter.device, parameter.dtype, hidden_size

    def _normalise_supplied(self, input_ids: torch.Tensor, supplied: torch.Tensor) -> torch.Tensor:
        if not isinstance(supplied, torch.Tensor):
            raise EmbeddingBoundaryError("inputs_embeds must be a torch.Tensor")
        if supplied.ndim != 3 or tuple(supplied.shape[:2]) != tuple(input_ids.shape):
            raise EmbeddingBoundaryError(
                "inputs_embeds must have shape [batch, sequence, hidden] matching input_ids"
            )
        layout = self._module_layout()
        if layout is not None:
            device, dtype, hidden_size = layout
            if hidden_size is not None and int(supplied.shape[-1]) != hidden_size:
                raise EmbeddingBoundaryError(
                    f"inputs_embeds hidden size {int(supplied.shape[-1])} does not match "
                    f"native embedding size {hidden_size}"
                )
            # A carrier must enter the decoder in the native execution layout.
            # Conversion is explicit here so the exact gate compares the bytes
            # that the decoder will actually consume.
            supplied = supplied.to(device=device, dtype=dtype)
        return supplied

    def _record(
        self,
        *,
        input_ids: torch.Tensor,
        embeddings: torch.Tensor,
        mode: str,
        exact: bool,
        verified: bool,
        cache_hit: bool = False,
    ) -> None:
        self.last_receipt = EmbeddingReceipt(
            mode=mode,
            exact=exact,
            provider=self.provider_name,
            input_ids_fingerprint=self._fingerprint(input_ids),
            embedding_fingerprint=self._fingerprint(embeddings),
            shape=tuple(int(dim) for dim in embeddings.shape),
            dtype=str(embeddings.dtype),
            device=str(embeddings.device),
            verified=verified,
            cache_hit=cache_hit,
            fingerprint_policy=self.fingerprint_policy.value,
            fingerprints_durable=(
                self.fingerprint_policy is EmbeddingFingerprintPolicy.CONTENT
            ),
        )

    def embed(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_input_ids(input_ids)
        if inputs_embeds is None:
            embeddings = self.module(input_ids)
            if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3:
                raise EmbeddingBoundaryError(
                    "native embedding module did not return a rank-3 tensor"
                )
            self._record(
                input_ids=input_ids,
                embeddings=embeddings,
                mode="native",
                exact=True,
                verified=False,
            )
            return embeddings

        supplied = self._normalise_supplied(input_ids, inputs_embeds)
        if self.verify_supplied:
            native = self.module(input_ids)
            if not isinstance(native, torch.Tensor) or native.shape != supplied.shape:
                raise EmbeddingBoundaryError("native and supplied embedding shapes differ")
            if native.dtype != supplied.dtype or native.device != supplied.device:
                raise EmbeddingBoundaryError("native and supplied embedding layouts differ")
            if not torch.equal(native, supplied):
                raise EmbeddingBoundaryError(
                    "supplied inputs_embeds failed the bit-exact native embedding gate"
                )
            self._record(
                input_ids=input_ids,
                embeddings=supplied,
                mode="supplied-verified",
                exact=True,
                verified=True,
            )
        else:
            self._record(
                input_ids=input_ids,
                embeddings=supplied,
                mode="supplied-unverified",
                exact=False,
                verified=False,
            )
        return supplied

    def describe(self) -> Mapping[str, Any]:
        return {
            "schema": EMBEDDING_BOUNDARY_SCHEMA,
            "provider": self.provider_name,
            "mode": "native",
            "native_exact": True,
            "accepts_supplied_residual": True,
            "verify_supplied_default": self.verify_supplied,
            "unverified_supplied_is_exact": False,
            "fingerprint_policy": self.fingerprint_policy.value,
            "fingerprints_durable": (
                self.fingerprint_policy is EmbeddingFingerprintPolicy.CONTENT
            ),
        }


class CachedEmbeddingProvider:
    """Exact cache layered over another embedding provider.

    Only native-exact, non-overridden results are retained.  Entries are
    keyed by token sequence and execution device/layout; the returned tensor is
    treated as borrowed read-only model input, matching the AR residual ABI.
    """

    provider_name = "cached-native-embed_tokens"

    def __init__(self, base: EmbeddingProvider, *, max_entries: int = 128) -> None:
        if max_entries < 1:
            raise EmbeddingBoundaryError("max_entries must be positive")
        if not callable(getattr(base, "embed", None)):
            raise EmbeddingBoundaryError("base embedding provider must expose embed()")
        self.base = base
        self.max_entries = int(max_entries)
        self._cache: OrderedDict[
            tuple[Any, ...], tuple[torch.Tensor, EmbeddingReceipt]
        ] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.last_receipt: EmbeddingReceipt | None = None

    @staticmethod
    def _key(input_ids: torch.Tensor) -> tuple[Any, ...]:
        return (
            str(input_ids.device),
            tuple(int(dim) for dim in input_ids.shape),
            tuple(int(token) for token in input_ids.detach().cpu().reshape(-1).tolist()),
        )

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def embed(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_input_ids(input_ids)
        # An override is caller-owned and must never silently become a native
        # cache entry.  The wrapped provider supplies the exactness receipt.
        if inputs_embeds is not None:
            embeddings = self.base.embed(input_ids, inputs_embeds=inputs_embeds)
            base_receipt = getattr(self.base, "last_receipt", None)
            if isinstance(base_receipt, EmbeddingReceipt):
                self.last_receipt = base_receipt
            else:
                self.last_receipt = None
            return embeddings

        key = self._key(input_ids)
        cached_entry = self._cache.get(key)
        if cached_entry is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            cached, receipt = cached_entry
            # The key proves the token sequence and the tensor is the exact object admitted on the
            # miss. Reuse that already-earned receipt instead of rereading both payloads on a hit.
            self.last_receipt = replace(
                receipt,
                mode="cached-native",
                provider=self.provider_name,
                cache_hit=True,
            )
            return cached

        self.misses += 1
        embeddings = self.base.embed(input_ids)
        base_receipt = getattr(self.base, "last_receipt", None)
        if isinstance(base_receipt, EmbeddingReceipt) and base_receipt.exact:
            cached_receipt = replace(
                base_receipt,
                provider=self.provider_name,
                cache_hit=False,
            )
            self._cache[key] = (embeddings.detach(), cached_receipt)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
            self.last_receipt = cached_receipt
        else:
            # A custom approximate provider remains approximate through the
            # wrapper; never label it native-exact just because it is cached.
            self.last_receipt = base_receipt
        return embeddings

    def describe(self) -> Mapping[str, Any]:
        base_describe = getattr(self.base, "describe", None)
        base = dict(base_describe()) if callable(base_describe) else {}
        return {
            "schema": EMBEDDING_BOUNDARY_SCHEMA,
            "provider": self.provider_name,
            "mode": "cached",
            "cache_exact_only": True,
            "max_entries": self.max_entries,
            "base": base,
        }

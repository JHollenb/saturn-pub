"""A typed autoregressive Saturn machine for the pinned Qwen2.5-0.5B base model.

The machine keeps model weights resident and represents row-local generation
state as an immutable, normalized prefix.  A committed prefix contains all
token IDs, while its KV cache covers every token except the last (pending)
token.  The next decode macro-step consumes that pending token, observes each
native decoder layer through a TITAN Act, computes logits, and commits the next
pending token.  Logits are therefore execution evidence, never durable state.

The implementation deliberately exposes two surfaces:

* :class:`Qwen2AutoregressiveMachine` is a direct, stateless execution API; and
* the ``compile_program_context`` / ``step_program_context`` methods implement
  mrun's family-owned lifecycle using only dict/list/tuple/tensor payloads.

Tests can supply a tiny Qwen-shaped model.  Production callers should use
``from_pretrained`` so the exact pinned model and tokenizer revision is loaded.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from mdb.tensor_hash import tensor_bytes_sha256, update_hash_with_tensor_bytes

if TYPE_CHECKING:
    from mdb.ar.embedding import EmbeddingProvider

from .titan import ActEvidence, ActRegistry, ActSpec, Frame

MACHINE_SCHEMA = "saturn-machine-v1"
PREFIX_SCHEMA = "saturn-llm-prefix-v1"
PROGRAM_CONTEXT_SCHEMA = "saturn-llm-program-context-v1"
KV_CACHE_SCHEMA = "saturn-normalized-kv-v1"
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SAMPLER = "greedy-argmax-v1"
IMPLEMENTATION = "transformers.qwen2.native"
SUPPORTED_NATIVE_MODEL_TYPES = frozenset(("qwen2", "qwen3"))
KV_FINGERPRINT_CONTENT = "content-sha256"
KV_FINGERPRINT_APPEND = "append-chain-sha256"
DERIVED_FINGERPRINT_CONTENT = "content-sha256"
DERIVED_FINGERPRINT_RECIPE = "derived-recipe-sha256"


class Qwen2MachineError(ValueError):
    """Raised when a model or prefix misses the autoregressive ABI."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach()
    return _sha256_json(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "bytes": tensor_bytes_sha256(tensor),
        }
    )


def _kv_delta_sha256(key: torch.Tensor, value: torch.Tensor) -> str:
    """Hash one KV append, never the prefix it extends.

    Keep this byte convention stable: persisted ledger chains use it.  Raw
    byte views support bfloat16 and make the digest independent of the source
    device.
    """

    hasher = hashlib.sha256()
    for tensor in (key, value):
        payload = tensor.detach()
        hasher.update(f"{payload.dtype}|{tuple(payload.shape)}|".encode())
        update_hash_with_tensor_bytes(hasher, payload)
    return hasher.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tensor_recipe(value: torch.Tensor | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
    }


def _tensor_geometry(value: torch.Tensor) -> Mapping[str, Any]:
    """Portable tensor geometry for producer identities (never residency)."""

    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
    }


def _attention_mask_recipe(value: torch.Tensor) -> Mapping[str, Any]:
    """Return an exact compact identity for the common all-ones mask.

    A Saturn row is batch-one and commit only appends ones.  The ordinary
    prompt therefore needs a one-scalar device reduction rather than copying
    and hashing the full mask every layer/cycle.  Unusual masks retain the
    old full-content identity and fail closed.
    """

    if value.numel() and bool(torch.all(value == 1).item()):
        return {
            "kind": "all-ones",
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    return {"kind": "content-sha256", "fingerprint": _tensor_sha256(value)}


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _as_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Qwen2MachineError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _config_payload(config: Any) -> Mapping[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            # Loading the same bytes by Hub ID or flat fleet path changes only
            # this provenance field. It must not make a durable prefix
            # non-portable across otherwise identical resident workers.
            return {key: value for key, value in payload.items() if key != "_name_or_path"}
    return {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("_") and isinstance(value, (str, bool, int, float, list, tuple))
    }


def _tokenizer_payload(tokenizer: Any | None) -> Mapping[str, Any]:
    if tokenizer is None:
        return {"tokenizer": None}
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else None
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        init_kwargs = {
            key: value
            for key, value in init_kwargs.items()
            if key not in {"name_or_path", "cache_dir"}
        }
    return {
        "class": type(tokenizer).__name__,
        "vocab": vocab if isinstance(vocab, Mapping) else None,
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "init_kwargs": init_kwargs,
    }


def _normalise_eos(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (tuple, list, set, frozenset)) else (value,)
    result: set[int] = set()
    for item in values:
        token = _as_int(item, "eos token ID")
        result.add(token)
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Fail-closed model/tokenizer identity carried by every durable prefix."""

    model_id: str
    revision: str
    config_sha256: str
    tokenizer_sha256: str

    def __post_init__(self) -> None:
        if not all((self.model_id, self.revision, self.config_sha256, self.tokenizer_sha256)):
            raise Qwen2MachineError("model identity fields must be non-empty")

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "config_sha256": self.config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelIdentity:
        try:
            return cls(
                model_id=str(value["model_id"]),
                revision=str(value["revision"]),
                config_sha256=str(value["config_sha256"]),
                tokenizer_sha256=str(value["tokenizer_sha256"]),
            )
        except (KeyError, TypeError) as exc:
            raise Qwen2MachineError("prefix has an invalid model identity") from exc


@dataclass(frozen=True, slots=True, repr=False)
class KVLayerState:
    """One immutable, normalized ``[batch, heads, seq, head_dim]`` KV pair.

    Imported/durable tensors default to a full content digest.  A trusted
    decoder producer can instead use :meth:`from_append`: its identity binds
    the prior immutable state, the newly-written delta digest, producer, and
    resulting geometry.  That makes normal decode identity O(delta), while
    :meth:`full_content_fingerprint` remains the explicit authority path.

    The scopes are deliberately not interchangeable.  An append identity is
    lineage evidence, not a claim that the complete tensor was re-read.
    """

    key: torch.Tensor
    value: torch.Tensor
    fingerprint: str | None = None
    fingerprint_scope: str = KV_FINGERPRINT_CONTENT
    parent_fingerprint: str | None = None
    delta_fingerprint: str | None = None
    producer: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, torch.Tensor) or not isinstance(self.value, torch.Tensor):
            raise Qwen2MachineError("normalized KV entries must be tensors")
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise Qwen2MachineError("normalized KV tensors must be rank four")
        if self.key.shape != self.value.shape:
            raise Qwen2MachineError("normalized key/value shapes must match")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise Qwen2MachineError("normalized key/value device and dtype must match")
        if self.fingerprint is None:
            if self.fingerprint_scope != KV_FINGERPRINT_CONTENT:
                raise Qwen2MachineError(
                    "producer KV identity requires an explicit fingerprint"
                )
            if any(
                item is not None
                for item in (
                    self.parent_fingerprint,
                    self.delta_fingerprint,
                    self.producer,
                )
            ):
                raise Qwen2MachineError(
                    "content KV identity cannot carry append provenance"
                )
            object.__setattr__(self, "fingerprint", self.full_content_fingerprint())
            return
        if not _is_sha256(self.fingerprint):
            raise Qwen2MachineError("KV fingerprint must be a lowercase SHA-256")
        if self.fingerprint_scope == KV_FINGERPRINT_CONTENT:
            if any(
                item is not None
                for item in (
                    self.parent_fingerprint,
                    self.delta_fingerprint,
                    self.producer,
                )
            ):
                raise Qwen2MachineError(
                    "content KV identity cannot carry append provenance"
                )
            if self.fingerprint != self.full_content_fingerprint():
                raise Qwen2MachineError(
                    "content KV fingerprint does not match tensor bytes"
                )
            return
        if self.fingerprint_scope != KV_FINGERPRINT_APPEND:
            raise Qwen2MachineError(
                f"unsupported KV fingerprint scope {self.fingerprint_scope!r}"
            )
        if self.parent_fingerprint is not None and not _is_sha256(
            self.parent_fingerprint
        ):
            raise Qwen2MachineError("append KV parent fingerprint is invalid")
        if not _is_sha256(self.delta_fingerprint):
            raise Qwen2MachineError("append KV delta fingerprint is invalid")
        if not isinstance(self.producer, str) or not self.producer:
            raise Qwen2MachineError("append KV identity requires its producer")
        expected = self._append_identity(
            parent_fingerprint=self.parent_fingerprint,
            delta_fingerprint=self.delta_fingerprint,
            producer=self.producer,
        )
        if self.fingerprint != expected:
            raise Qwen2MachineError("append KV fingerprint does not match provenance")

    def _append_identity(
        self,
        *,
        parent_fingerprint: str | None,
        delta_fingerprint: str,
        producer: str,
    ) -> str:
        return _sha256_json(
            {
                "schema": "saturn-kv-append-identity-v1",
                "parent": parent_fingerprint,
                "delta": delta_fingerprint,
                "producer": producer,
                "key": _tensor_geometry(self.key),
                "value": _tensor_geometry(self.value),
            }
        )

    @classmethod
    def from_append(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        prior: KVLayerState | None,
        delta_key: torch.Tensor,
        delta_value: torch.Tensor,
        producer: str,
        delta_fingerprint: str | None = None,
    ) -> KVLayerState:
        """Construct a merged producer state by hashing only its append."""

        if delta_key.shape != delta_value.shape or delta_key.ndim != 4:
            raise Qwen2MachineError("KV append delta must be matching rank-four tensors")
        if key.shape != value.shape or key.ndim != 4:
            raise Qwen2MachineError("merged KV append must be matching rank-four tensors")
        if key.shape[:2] != delta_key.shape[:2] or key.shape[-1] != delta_key.shape[-1]:
            raise Qwen2MachineError("merged KV geometry does not match its delta")
        prior_length = 0 if prior is None else prior.sequence_length
        if int(key.shape[-2]) != prior_length + int(delta_key.shape[-2]):
            raise Qwen2MachineError("merged KV length does not equal prior plus delta")
        if prior is not None:
            if (
                prior.key.shape[:2] != key.shape[:2]
                or prior.key.shape[-1] != key.shape[-1]
                or prior.key.device != key.device
                or prior.key.dtype != key.dtype
            ):
                raise Qwen2MachineError("merged KV geometry does not match its prior")
        if (
            key.device != delta_key.device
            or value.device != delta_value.device
            or key.dtype != delta_key.dtype
            or value.dtype != delta_value.dtype
        ):
            raise Qwen2MachineError("merged KV device/dtype does not match its delta")
        delta = delta_fingerprint or _kv_delta_sha256(delta_key, delta_value)
        if not _is_sha256(delta):
            raise Qwen2MachineError("KV append delta fingerprint is invalid")
        parent = None if prior is None else prior.fingerprint
        provisional = cls.__new__(cls)
        object.__setattr__(provisional, "key", key)
        object.__setattr__(provisional, "value", value)
        fingerprint = provisional._append_identity(
            parent_fingerprint=parent,
            delta_fingerprint=delta,
            producer=producer,
        )
        return cls(
            key,
            value,
            fingerprint=fingerprint,
            fingerprint_scope=KV_FINGERPRINT_APPEND,
            parent_fingerprint=parent,
            delta_fingerprint=delta,
            producer=producer,
        )

    @property
    def sequence_length(self) -> int:
        return int(self.key.shape[-2])

    def clone(self) -> KVLayerState:
        """Deep-copy payloads while preserving their already-proven identity."""

        return KVLayerState(
            self.key.detach().clone(),
            self.value.detach().clone(),
            fingerprint=self.fingerprint,
            fingerprint_scope=self.fingerprint_scope,
            parent_fingerprint=self.parent_fingerprint,
            delta_fingerprint=self.delta_fingerprint,
            producer=self.producer,
        )

    def full_content_fingerprint(self) -> str:
        """Explicitly hash the complete KV bytes for durable verification."""

        return _sha256_json(
            {"key": _tensor_sha256(self.key), "value": _tensor_sha256(self.value)}
        )

    def verify_full_content(self, expected: str) -> str:
        """Fail closed unless the complete KV bytes match ``expected``."""

        if not _is_sha256(expected):
            raise Qwen2MachineError("expected KV content fingerprint is invalid")
        observed = self.full_content_fingerprint()
        if observed != expected:
            raise Qwen2MachineError("KV full-content verification failed")
        return observed

    def __repr__(self) -> str:
        return (
            f"KVLayerState(shape={tuple(self.key.shape)}, "
            f"fingerprint={self.fingerprint}, scope={self.fingerprint_scope})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class NormalizedKVCache:
    """Transformers-independent KV state suitable for clone, fork, and StateCut."""

    layers: tuple[KVLayerState | None, ...]
    schema: str = KV_CACHE_SCHEMA
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema != KV_CACHE_SCHEMA:
            raise Qwen2MachineError(f"unsupported KV cache schema {self.schema!r}")
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(
            self,
            "fingerprint",
            _sha256_json(
                {
                    "schema": self.schema,
                    "layers": [None if item is None else item.fingerprint for item in self.layers],
                }
            ),
        )

    @classmethod
    def empty(cls, layer_count: int) -> NormalizedKVCache:
        return cls(tuple(None for _ in range(_as_int(layer_count, "layer_count", minimum=1))))

    def __repr__(self) -> str:
        lengths = [None if item is None else item.sequence_length for item in self.layers]
        return (
            f"NormalizedKVCache(lengths={lengths}, fingerprint={self.fingerprint}, "
            f"scope={self.fingerprint_scope})"
        )

    @property
    def fingerprint_scope(self) -> str:
        return (
            KV_FINGERPRINT_APPEND
            if any(
                item is not None
                and item.fingerprint_scope == KV_FINGERPRINT_APPEND
                for item in self.layers
            )
            else KV_FINGERPRINT_CONTENT
        )

    def clone(self) -> NormalizedKVCache:
        return NormalizedKVCache(
            tuple(None if item is None else item.clone() for item in self.layers)
        )

    def fork(self) -> NormalizedKVCache:
        """Share immutable layer payloads; an execution append is copy-on-write."""

        return NormalizedKVCache(self.layers)

    def full_content_fingerprint(self) -> str:
        """Explicit content authority for checkpoint/publication gates."""

        return _sha256_json(
            {
                "schema": self.schema,
                "layers": [
                    None if item is None else item.full_content_fingerprint()
                    for item in self.layers
                ],
            }
        )

    def verify_full_content(self, expected: str) -> str:
        if not _is_sha256(expected):
            raise Qwen2MachineError("expected KV-cache content fingerprint is invalid")
        observed = self.full_content_fingerprint()
        if observed != expected:
            raise Qwen2MachineError("KV-cache full-content verification failed")
        return observed

    def common_sequence_length(self, *, complete: bool = True) -> int:
        if complete and any(item is None for item in self.layers):
            raise Qwen2MachineError("committed KV cache has uninitialized layers")
        lengths = {item.sequence_length for item in self.layers if item is not None}
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise Qwen2MachineError(f"KV layers do not share one prefix length: {sorted(lengths)}")
        return next(iter(lengths))

    def export_tensor_slots(self, *, clone: bool = False) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for index, layer in enumerate(self.layers):
            if layer is None:
                raise Qwen2MachineError("cannot export an incomplete KV cache")
            key = layer.key.detach().clone() if clone else layer.key
            value = layer.value.detach().clone() if clone else layer.value
            result[f"kv_cache.layer_{index:03d}.key"] = key
            result[f"kv_cache.layer_{index:03d}.value"] = value
        return result

    @classmethod
    def from_tensor_slots(
        cls,
        slots: Mapping[str, Any],
        *,
        layer_count: int,
        clone: bool = False,
        device: torch.device | str | None = None,
        expected_dtype: torch.dtype | None = None,
    ) -> NormalizedKVCache:
        layers: list[KVLayerState] = []
        for index in range(_as_int(layer_count, "layer_count", minimum=1)):
            key_name = f"kv_cache.layer_{index:03d}.key"
            value_name = f"kv_cache.layer_{index:03d}.value"
            try:
                key, value = slots[key_name], slots[value_name]
            except KeyError as exc:
                raise Qwen2MachineError(f"prefix is missing {exc.args[0]}") from exc
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise Qwen2MachineError(f"{key_name}/{value_name} must be tensors")
            if expected_dtype is not None and (
                key.dtype != expected_dtype or value.dtype != expected_dtype
            ):
                raise Qwen2MachineError(
                    f"{key_name}/{value_name} dtype must be {expected_dtype}, "
                    f"got {key.dtype}/{value.dtype}"
                )
            key = key.to(device=device or key.device)
            value = value.to(device=device or value.device)
            layers.append(
                KVLayerState(
                    key.detach().clone() if clone else key,
                    value.detach().clone() if clone else value,
                )
            )
        return cls(tuple(layers))


@dataclass(slots=True)
class ResidentKVCacheHandle:
    """One mutable HF cache bound to one Saturn cycle.

    ``NormalizedKVCache`` remains the immutable, durable representation.  A
    resident handle is the execution representation: it is constructed once
    at the cycle boundary and then passed through context derivation and every
    native decoder layer.  This avoids rebuilding a ``DynamicCache`` (and
    rebinding every prior layer) for each layer while keeping ownership of
    committed KV in the StateLedger.

    The handle is deliberately not serializable or exposed as durable state.
    It must never outlive the cycle that owns it.
    """

    cache: Any
    normalized: NormalizedKVCache
    layer_count: int
    touched_layers: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if len(self.normalized.layers) != self.layer_count:
            raise Qwen2MachineError("resident cache layer count does not match model")

    def mark_touched(self, index: int) -> None:
        if not 0 <= index < self.layer_count:
            raise Qwen2MachineError(f"resident cache layer {index} is out of range")
        self.touched_layers.add(index)

    def layer(self, index: int) -> Any:
        layers = getattr(self.cache, "layers", None)
        if layers is None or index >= len(layers):
            raise Qwen2MachineError("resident cache does not expose native layers")
        return layers[index]


@dataclass(frozen=True, slots=True, repr=False)
class DerivedLayerContext:
    """Transient Qwen position/mask/RoPE inputs; always recomputed on restore.

    Direct construction retains the historical full-byte identity.  Native
    execution uses :meth:`from_recipe`, where positions/RoPE/masks are bound
    to the deterministic derivation recipe instead of copied to CPU and
    hashed every token.  Full bytes remain available through the explicit
    verifier and are never promoted to durable state.
    """

    position_ids: torch.Tensor
    cache_position: torch.Tensor
    causal_masks: Mapping[str, torch.Tensor | None]
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    fingerprint: str | None = None
    fingerprint_scope: str = DERIVED_FINGERPRINT_CONTENT
    recipe: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "causal_masks", dict(self.causal_masks))
        if len(self.position_embeddings) != 2:
            raise Qwen2MachineError("derived context requires cosine/sine tensors")
        if self.fingerprint is None:
            if self.fingerprint_scope != DERIVED_FINGERPRINT_CONTENT:
                raise Qwen2MachineError(
                    "derived recipe identity requires an explicit fingerprint"
                )
            if self.recipe is not None:
                raise Qwen2MachineError(
                    "full-content derived identity cannot carry a recipe"
                )
            object.__setattr__(self, "fingerprint", self.full_content_fingerprint())
            return
        if not _is_sha256(self.fingerprint):
            raise Qwen2MachineError(
                "derived-context fingerprint must be a lowercase SHA-256"
            )
        if self.fingerprint_scope == DERIVED_FINGERPRINT_CONTENT:
            if self.recipe is not None:
                raise Qwen2MachineError(
                    "full-content derived identity cannot carry a recipe"
                )
            if self.fingerprint != self.full_content_fingerprint():
                raise Qwen2MachineError(
                    "derived-context content fingerprint does not match tensors"
                )
            return
        if self.fingerprint_scope != DERIVED_FINGERPRINT_RECIPE:
            raise Qwen2MachineError(
                f"unsupported derived fingerprint scope {self.fingerprint_scope!r}"
            )
        if not isinstance(self.recipe, Mapping):
            raise Qwen2MachineError("derived recipe identity requires its recipe")
        normalized = _jsonable(self.recipe)
        object.__setattr__(self, "recipe", normalized)
        if self.fingerprint != _sha256_json(normalized):
            raise Qwen2MachineError(
                "derived-context fingerprint does not match its recipe"
            )

    @classmethod
    def from_recipe(
        cls,
        *,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
        causal_masks: Mapping[str, torch.Tensor | None],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        recipe: Mapping[str, Any],
    ) -> DerivedLayerContext:
        normalized = _jsonable(recipe)
        return cls(
            position_ids=position_ids,
            cache_position=cache_position,
            causal_masks=causal_masks,
            position_embeddings=position_embeddings,
            fingerprint=_sha256_json(normalized),
            fingerprint_scope=DERIVED_FINGERPRINT_RECIPE,
            recipe=normalized,
        )

    def full_content_fingerprint(self) -> str:
        cos, sin = self.position_embeddings
        return _sha256_json(
            {
                "position_ids": _tensor_sha256(self.position_ids),
                "cache_position": _tensor_sha256(self.cache_position),
                "masks": {
                    key: None if value is None else _tensor_sha256(value)
                    for key, value in self.causal_masks.items()
                },
                "rope": [_tensor_sha256(cos), _tensor_sha256(sin)],
            }
        )

    def verify_full_content(self, expected: str) -> str:
        if not _is_sha256(expected):
            raise Qwen2MachineError(
                "expected derived-context content fingerprint is invalid"
            )
        observed = self.full_content_fingerprint()
        if observed != expected:
            raise Qwen2MachineError("derived-context full-content verification failed")
        return observed

    def mask_for(self, layer_type: str) -> torch.Tensor | None:
        if layer_type in self.causal_masks:
            return self.causal_masks[layer_type]
        if "full_attention" in self.causal_masks:
            return self.causal_masks["full_attention"]
        raise Qwen2MachineError(f"derived context has no mask for {layer_type!r}")

    def __repr__(self) -> str:
        return (
            f"DerivedLayerContext(fingerprint={self.fingerprint}, "
            f"scope={self.fingerprint_scope})"
        )


@dataclass(frozen=True, slots=True)
class AutoregressiveState:
    """A committed prefix with one generated token pending outside the cache."""

    token_ids: torch.Tensor
    attention_mask: torch.Tensor
    kv_cache: NormalizedKVCache
    prompt_length: int
    cursor: int
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    model_identity: ModelIdentity
    frame_id: str
    finished: bool = False
    finish_reason: str | None = None
    sampler: str = SAMPLER

    def __post_init__(self) -> None:
        if self.sampler != SAMPLER:
            raise Qwen2MachineError("v1 supports only deterministic greedy sampling")
        if not isinstance(self.token_ids, torch.Tensor) or self.token_ids.dtype != torch.long:
            raise Qwen2MachineError("token_ids must be a torch.long tensor")
        if not isinstance(self.attention_mask, torch.Tensor):
            raise Qwen2MachineError("attention_mask must be a tensor")
        if self.token_ids.ndim != 2 or self.attention_mask.shape != self.token_ids.shape:
            raise Qwen2MachineError("v1 token_ids/attention_mask must have identical rank-2 shape")
        if self.token_ids.shape[0] != 1:
            raise Qwen2MachineError("v1 supports batch size one")
        prompt_length = _as_int(self.prompt_length, "prompt_length", minimum=1)
        cursor = _as_int(self.cursor, "cursor", minimum=1)
        maximum = _as_int(self.max_new_tokens, "max_new_tokens", minimum=1)
        if int(self.token_ids.shape[1]) != prompt_length + cursor:
            raise Qwen2MachineError("token length must equal prompt_length + cursor")
        if cursor > maximum:
            raise Qwen2MachineError("cursor cannot exceed max_new_tokens")
        if self.kv_cache.common_sequence_length() != int(self.token_ids.shape[1]) - 1:
            raise Qwen2MachineError(
                "committed prefix invariant requires KV length == token length - 1"
            )
        if self.finished != (self.finish_reason is not None):
            raise Qwen2MachineError("finished and finish_reason must agree")
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise Qwen2MachineError("committed prefix requires its final frame ID")
        if self.finish_reason not in {None, "eos", "max_new_tokens"}:
            raise Qwen2MachineError(f"unsupported finish reason {self.finish_reason!r}")
        object.__setattr__(self, "eos_token_ids", _normalise_eos(self.eos_token_ids))

    @property
    def pending_token_id(self) -> int:
        return int(self.token_ids[0, -1].item())

    @property
    def generated_ids(self) -> torch.Tensor:
        return self.token_ids[:, self.prompt_length :]

    def clone(self) -> AutoregressiveState:
        """Create an explicit deep payload copy.

        Normal branching should use :meth:`fork`; deep cloning is retained
        for callers that need independently mutable exported tensors.
        """

        return AutoregressiveState(
            token_ids=self.token_ids.detach().clone(),
            attention_mask=self.attention_mask.detach().clone(),
            kv_cache=self.kv_cache.clone(),
            prompt_length=self.prompt_length,
            cursor=self.cursor,
            max_new_tokens=self.max_new_tokens,
            eos_token_ids=self.eos_token_ids,
            model_identity=self.model_identity,
            frame_id=self.frame_id,
            finished=self.finished,
            finish_reason=self.finish_reason,
            sampler=self.sampler,
        )

    def fork(self) -> AutoregressiveState:
        """Fork immutable row state by reference; execution appends COW."""

        return replace(self, kv_cache=self.kv_cache.fork())

    def export_prefix_slots(self, *, clone: bool = False) -> dict[str, Any]:
        """Export the durable ``llm_prefix`` cut without transient logits."""

        result: dict[str, Any] = {
            "schema": PREFIX_SCHEMA,
            "state_schema_id": PREFIX_SCHEMA,
            "model_identity": self.model_identity.to_dict(),
            "frame_id": self.frame_id,
            "sampler": self.sampler,
            "token_ids": self.token_ids.detach().clone() if clone else self.token_ids,
            "attention_mask": (
                self.attention_mask.detach().clone() if clone else self.attention_mask
            ),
            "kv_cache_schema": self.kv_cache.schema,
            "kv_layer_count": len(self.kv_cache.layers),
            "prompt_length": self.prompt_length,
            "cursor": self.cursor,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_ids": tuple(self.eos_token_ids),
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "token_cut": self.cursor,
        }
        result.update(self.kv_cache.export_tensor_slots(clone=clone))
        return result


@dataclass(frozen=True, slots=True)
class AutoregressiveStepResult:
    """One committed macro-step plus debugger-visible micro-act evidence."""

    state: AutoregressiveState
    logits: torch.Tensor
    token_id: int
    frames: tuple[Frame, ...]
    evidence: tuple[ActEvidence, ...]
    cut_frames: Mapping[int, Frame]
    phase: str
    root_frame: Frame

    @property
    def final_frame(self) -> Frame:
        return self.frames[-1]

    @property
    def macro_evidence(self) -> ActEvidence:
        return self.evidence[-1]


@dataclass(frozen=True, slots=True)
class LayerReplayResult:
    cut: int
    logits: torch.Tensor
    reference_logits: torch.Tensor
    exact: bool
    frames: tuple[Frame, ...]
    evidence: tuple[ActEvidence, ...]


@dataclass(frozen=True, slots=True)
class AutoregressiveGenerationResult:
    state: AutoregressiveState
    steps: tuple[AutoregressiveStepResult, ...]

    @property
    def token_ids(self) -> torch.Tensor:
        return self.state.token_ids

    @property
    def generated_ids(self) -> torch.Tensor:
        return self.state.generated_ids


def embed_spec() -> ActSpec:
    return ActSpec(
        name="llm.embed",
        abi_version="v1",
        reads=("input_ids", "attention_mask", "kv_cache"),
        writes=("hidden_state", "derived_context"),
        numerical_contract="native-bit-exact",
        write_kinds={"hidden_state": "tensor", "derived_context": "derived_state"},
        write_abis={
            "hidden_state": {"coordinate_space": "qwen2.residual"},
            "derived_context": {"coordinate_space": "qwen2.position-mask-rope"},
        },
    )


def layer_spec() -> ActSpec:
    return ActSpec(
        name="llm.layer",
        abi_version="v1",
        reads=("hidden_state", "derived_context", "kv_cache"),
        writes=("hidden_state", "kv_cache"),
        numerical_contract="native-bit-exact",
        write_kinds={"hidden_state": "tensor", "kv_cache": "kv_cache"},
        write_abis={
            "hidden_state": {"coordinate_space": "qwen2.residual"},
            "kv_cache": {"schema": KV_CACHE_SCHEMA},
        },
    )


def readout_spec() -> ActSpec:
    return ActSpec(
        name="llm.readout",
        abi_version="v1",
        reads=("hidden_state",),
        writes=("logits",),
        numerical_contract="native-bit-exact",
        write_kinds={"logits": "tensor"},
        write_abis={"logits": {"coordinate_space": "qwen2.vocabulary"}},
    )


def sample_spec() -> ActSpec:
    return ActSpec(
        name="llm.sample",
        abi_version="v1",
        reads=("logits", "forced_token_id"),
        writes=("sampled_token_id", "sample_source"),
        numerical_contract="deterministic-greedy",
        write_kinds={"sampled_token_id": "token_id", "sample_source": "scalar"},
    )


def macro_step_spec(phase: str) -> ActSpec:
    if phase not in {"prefill", "decode_step"}:
        raise Qwen2MachineError(f"unsupported macro phase {phase!r}")
    return ActSpec(
        name=f"llm.{phase}",
        abi_version="v1",
        reads=(
            "token_ids",
            "attention_mask",
            "kv_cache",
            "cursor",
            "max_new_tokens",
            "eos_token_ids",
            "sampled_token_id",
            "phase",
        ),
        writes=(
            "token_ids",
            "attention_mask",
            "kv_cache",
            "cursor",
            "finished",
            "finish_reason",
            "phase",
        ),
        invalidates=(
            "input_ids",
            "hidden_state",
            "derived_context",
            "logits",
            "sampled_token_id",
            "sample_source",
            "forced_token_id",
        ),
        numerical_contract="native-bit-exact",
        write_kinds={
            "token_ids": "token_ids",
            "attention_mask": "attention_mask",
            "kv_cache": "kv_cache",
            "cursor": "scalar",
            "finished": "scalar",
            "finish_reason": "scalar",
            "phase": "scalar",
        },
        write_abis={"kv_cache": {"schema": KV_CACHE_SCHEMA}},
    )


class _Qwen2NativeAdapter:
    """Narrow adapter around resident Hugging Face Qwen2/Qwen3 modules."""

    def __init__(
        self,
        model: Any,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_fingerprint_policy: Any | None = None,
    ) -> None:
        self.model = model
        self.backbone = getattr(model, "model", None)
        if self.backbone is None:
            raise Qwen2MachineError("causal LM must expose model.embed_tokens/layers/norm")
        for name in ("embed_tokens", "layers", "norm"):
            if not hasattr(self.backbone, name):
                raise Qwen2MachineError(f"Qwen backbone is missing {name}")
        if embedding_provider is None:
            # Keep this import local: saturn's package initializer imports the
            # v1 adapter before the AR package initializer is complete.
            from mdb.ar.embedding import NativeEmbeddingProvider

            provider_kwargs = {}
            if embedding_fingerprint_policy is not None:
                provider_kwargs["fingerprint_policy"] = embedding_fingerprint_policy
            embedding_provider = NativeEmbeddingProvider(
                self.backbone.embed_tokens,
                **provider_kwargs,
            )
        elif embedding_fingerprint_policy is not None:
            raise Qwen2MachineError(
                "embedding_fingerprint_policy belongs to the supplied embedding provider"
            )
        self.embedding_provider = embedding_provider
        if not callable(getattr(self.embedding_provider, "embed", None)):
            raise Qwen2MachineError("embedding_provider must expose embed()")
        if not callable(getattr(model, "lm_head", None)):
            raise Qwen2MachineError("causal LM must expose lm_head")
        self.layers = tuple(self.backbone.layers)
        if not self.layers:
            raise Qwen2MachineError("Qwen backbone has no decoder layers")
        config = model.config
        model_type = str(getattr(config, "model_type", ""))
        if model_type == "qwen2":
            from transformers.models.qwen2 import modeling_qwen2 as modeling
        elif model_type == "qwen3":
            from transformers.models.qwen3 import modeling_qwen3 as modeling
        else:
            supported = ", ".join(sorted(SUPPORTED_NATIVE_MODEL_TYPES))
            raise Qwen2MachineError(
                f"native Qwen adapter supports {supported}; got model_type={model_type!r}"
            )
        self.model_type = model_type
        self.modeling = modeling
        configured = int(getattr(config, "num_hidden_layers", len(self.layers)))
        if configured != len(self.layers):
            raise Qwen2MachineError(
                f"configured/physical layer count mismatch: {configured} != {len(self.layers)}"
            )
        layer_types = getattr(config, "layer_types", None)
        self.layer_types = tuple(layer_types or ("full_attention",) * len(self.layers))
        if len(self.layer_types) != len(self.layers):
            raise Qwen2MachineError("config.layer_types does not match decoder layer count")
        unsupported = set(self.layer_types) - {"full_attention"}
        if unsupported:
            raise Qwen2MachineError(
                f"v1 normalized Qwen cache supports full attention only: {sorted(unsupported)}"
            )
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            attention_heads = getattr(config, "num_attention_heads", None)
            if hidden_size is not None and attention_heads:
                head_dim = int(hidden_size) // int(attention_heads)
        if not isinstance(num_kv_heads, int) or num_kv_heads < 1:
            raise Qwen2MachineError("Qwen config must declare num_key_value_heads")
        if not isinstance(head_dim, int) or head_dim < 1:
            raise Qwen2MachineError("Qwen config must declare a positive head_dim")
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self._derived_recipe_model_fingerprint = _sha256_json(
            {
                "model_type": self.model_type,
                "config": _config_payload(config),
                "layer_types": self.layer_types,
            }
        )

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def new_hf_cache(self, normalized: NormalizedKVCache):
        if len(normalized.layers) != self.layer_count:
            raise Qwen2MachineError("normalized KV layer count does not match model")
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise Qwen2MachineError("transformers DynamicCache is unavailable") from exc
        try:
            cache = DynamicCache(config=self.model.config)
        except (AttributeError, TypeError, ValueError):
            cache = DynamicCache()
        for index, state in enumerate(normalized.layers):
            if state is None:
                continue
            if hasattr(cache, "layers") and index < len(cache.layers):
                layer = cache.layers[index]
                if bool(getattr(layer, "is_sliding", False)):
                    raise Qwen2MachineError("sliding-window cache layers are deferred from v1")
                initialise = getattr(layer, "lazy_initialization", None)
                if callable(initialise) and not bool(getattr(layer, "is_initialized", False)):
                    initialise(state.key, state.value)
                layer.keys = state.key
                layer.values = state.value
                layer.is_initialized = True
            else:
                cache.update(state.key, state.value, index)
        return cache

    def bind_cache(self, normalized: NormalizedKVCache) -> ResidentKVCacheHandle:
        """Bind one native cache for the lifetime of a cycle."""

        return ResidentKVCacheHandle(
            cache=self.new_hf_cache(normalized),
            normalized=normalized,
            layer_count=self.layer_count,
        )

    def normalise_cache(
        self,
        cache: Any,
        *,
        prior: NormalizedKVCache | None = None,
    ) -> NormalizedKVCache:
        states: list[KVLayerState | None] = []
        if hasattr(cache, "layers"):
            source_layers = cache.layers
            for index in range(self.layer_count):
                if index >= len(source_layers):
                    states.append(None)
                    continue
                layer = source_layers[index]
                key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
                if (
                    not bool(getattr(layer, "is_initialized", False))
                    or not isinstance(key, torch.Tensor)
                    or key.numel() == 0
                ):
                    states.append(None)
                    continue
                previous = prior.layers[index] if prior is not None else None
                if previous is not None and previous.key is key and previous.value is value:
                    states.append(previous)
                else:
                    states.append(KVLayerState(key, value))
        elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for index in range(self.layer_count):
                if index >= len(cache.key_cache):
                    states.append(None)
                else:
                    states.append(KVLayerState(cache.key_cache[index], cache.value_cache[index]))
        else:  # compatibility with legacy tuple caches
            try:
                entries = tuple(cache)
            except TypeError as exc:
                raise Qwen2MachineError("unsupported Hugging Face cache object") from exc
            for index in range(self.layer_count):
                if index >= len(entries):
                    states.append(None)
                    continue
                entry = entries[index]
                if len(entry) < 2 or entry[0] is None:
                    states.append(None)
                else:
                    states.append(KVLayerState(entry[0], entry[1]))
        return NormalizedKVCache(tuple(states))

    def derive_context(
        self,
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        normalized: NormalizedKVCache,
        *,
        cache_handle: ResidentKVCacheHandle | None = None,
    ) -> DerivedLayerContext:
        past_length = normalized.common_sequence_length(complete=False)
        query_length = int(hidden_state.shape[1])
        cache_position = torch.arange(
            past_length,
            past_length + query_length,
            device=hidden_state.device,
            dtype=torch.long,
        )
        position_ids = cache_position.unsqueeze(0)
        cache = (
            cache_handle.cache
            if cache_handle is not None
            else self.new_hf_cache(normalized)
        )
        custom = getattr(self.backbone, "prepare_saturn_context", None)
        if callable(custom):
            value = custom(
                hidden_state,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
            )
            if isinstance(value, DerivedLayerContext):
                return value
            if not isinstance(value, Mapping):
                raise Qwen2MachineError("prepare_saturn_context must return a mapping")
            return DerivedLayerContext(
                position_ids=value.get("position_ids", position_ids),
                cache_position=value.get("cache_position", cache_position),
                causal_masks=value["causal_masks"],
                position_embeddings=tuple(value["position_embeddings"]),
            )
        try:
            from transformers.masking_utils import (
                create_causal_mask,
                create_sliding_window_causal_mask,
            )
        except ImportError as exc:  # pragma: no cover - pinned transformers has these
            raise Qwen2MachineError("pinned transformers masking helpers are unavailable") from exc
        mask_kwargs = {
            "config": self.model.config,
            "inputs_embeds": hidden_state,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        masks: dict[str, torch.Tensor | None] = {
            "full_attention": create_causal_mask(**mask_kwargs)
        }
        if set(self.layer_types) - {"full_attention"}:
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        rotary = getattr(self.backbone, "rotary_emb", None)
        if not callable(rotary):
            raise Qwen2MachineError("Qwen backbone is missing rotary_emb")
        position_embeddings = tuple(rotary(hidden_state, position_ids))
        cos, sin = position_embeddings
        return DerivedLayerContext.from_recipe(
            position_ids=position_ids,
            cache_position=cache_position,
            causal_masks=masks,
            position_embeddings=position_embeddings,
            recipe={
                "schema": "saturn-derived-context-recipe-v1",
                "model": self._derived_recipe_model_fingerprint,
                "past_length": past_length,
                "query_length": query_length,
                "attention_mask": _attention_mask_recipe(attention_mask),
                "position_range": [past_length, past_length + query_length],
                "causal_masks": {
                    name: _tensor_recipe(value) for name, value in masks.items()
                },
                "rope": [_tensor_recipe(cos), _tensor_recipe(sin)],
            },
        )

    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: NormalizedKVCache,
        *,
        inputs_embeds: torch.Tensor | None = None,
        cache_handle: ResidentKVCacheHandle | None = None,
    ) -> tuple[torch.Tensor, DerivedLayerContext]:
        hidden_state = self.embedding_provider.embed(
            input_ids,
            inputs_embeds=inputs_embeds,
        )
        return hidden_state, self.derive_context(
            hidden_state,
            attention_mask,
            kv_cache,
            cache_handle=cache_handle,
        )

    def embedding_descriptor(self) -> Mapping[str, Any]:
        describe = getattr(self.embedding_provider, "describe", None)
        if callable(describe):
            return dict(describe())
        return {
            "schema": "saturn-ar-embedding-boundary-v1",
            "provider": type(self.embedding_provider).__name__,
            "mode": "custom",
            "native_exact": False,
        }

    def embedding_receipt(self) -> Mapping[str, Any]:
        receipt = getattr(self.embedding_provider, "last_receipt", None)
        if receipt is not None:
            to_dict = getattr(receipt, "to_dict", None)
            if callable(to_dict):
                return dict(to_dict())
            if isinstance(receipt, Mapping):
                return dict(receipt)
        descriptor = self.embedding_descriptor()
        return {
            "schema": "saturn-ar-embedding-receipt-v1",
            "mode": str(descriptor.get("mode", "custom")),
            "provider": str(descriptor.get("provider", type(self.embedding_provider).__name__)),
            "exact": bool(descriptor.get("native_exact", False)),
            "verified": False,
            "cache_hit": False,
        }

    def run_layer(
        self,
        index: int,
        hidden_state: torch.Tensor,
        context: DerivedLayerContext,
        normalized: NormalizedKVCache,
    ) -> tuple[torch.Tensor, NormalizedKVCache]:
        cache = self.new_hf_cache(normalized)
        layer = self.layers[index]
        kwargs = {
            "attention_mask": context.mask_for(self.layer_types[index]),
            "position_ids": context.position_ids,
            "past_key_values": cache,
            "use_cache": True,
            "position_embeddings": context.position_embeddings,
            "cache_position": context.cache_position,
        }
        signature = inspect.signature(layer.forward if hasattr(layer, "forward") else layer)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        output = layer(hidden_state, **kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise Qwen2MachineError(f"decoder layer {index} did not return a tensor")
        next_cache = self.normalise_cache(cache, prior=normalized)
        previous = normalized.layers[index]
        expected_length = (0 if previous is None else previous.sequence_length) + int(
            hidden_state.shape[1]
        )
        observed = next_cache.layers[index]
        if observed is None or observed.sequence_length != expected_length:
            raise Qwen2MachineError(
                f"decoder layer {index} did not advance KV cache to {expected_length}"
            )
        return output, next_cache

    def run_layer_resident(
        self,
        index: int,
        hidden_state: torch.Tensor,
        context: DerivedLayerContext,
        handle: ResidentKVCacheHandle,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one native layer against an already-bound cache.

        Return the native merged key/value tensors without constructing a
        ``KVLayerState``.  The StateLedger owns normalization and can assign
        an append identity after hashing only the produced delta.  The old
        compatibility method above remains available for callers that
        explicitly want a complete normalized snapshot.
        """

        if handle.layer_count != self.layer_count:
            raise Qwen2MachineError("resident cache was bound to another model")
        if index in handle.touched_layers:
            raise Qwen2MachineError(
                f"resident cache layer {index} was executed twice in one cycle"
            )
        layer = self.layers[index]
        kwargs = {
            "attention_mask": context.mask_for(self.layer_types[index]),
            "position_ids": context.position_ids,
            "past_key_values": handle.cache,
            "use_cache": True,
            "position_embeddings": context.position_embeddings,
            "cache_position": context.cache_position,
        }
        signature = inspect.signature(layer.forward if hasattr(layer, "forward") else layer)
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        output = layer(hidden_state, **kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise Qwen2MachineError(f"decoder layer {index} did not return a tensor")
        native_layer = handle.layer(index)
        key = getattr(native_layer, "keys", None)
        value = getattr(native_layer, "values", None)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise Qwen2MachineError(f"decoder layer {index} produced no resident KV state")
        previous = handle.normalized.layers[index]
        expected_length = (0 if previous is None else previous.sequence_length) + int(
            hidden_state.shape[1]
        )
        if int(key.shape[-2]) != expected_length:
            raise Qwen2MachineError(
                f"decoder layer {index} did not advance resident KV cache to {expected_length}"
            )
        handle.mark_touched(index)
        return output, key, value

    def readout(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.backbone.norm(hidden_state))


class Qwen2AutoregressiveMachine:
    """Resident Qwen2 native backend with typed token and layer transitions."""

    # mrun may share frozen context tensors across transaction snapshots because
    # every hook below treats the source Mapping/tensors as read-only and emits
    # a new copy-on-write prefix tree.
    program_context_is_immutable = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        *,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        tokenizer_revision: str | None = None,
        strict_identity: bool = True,
    ) -> None:
        if strict_identity and (model_id != MODEL_ID or revision != MODEL_REVISION):
            raise Qwen2MachineError(
                f"v1 is pinned to {MODEL_ID}@{MODEL_REVISION}; got {model_id}@{revision}"
            )
        config = getattr(model, "config", None)
        if config is None:
            raise Qwen2MachineError("causal LM has no config")
        if str(getattr(config, "model_type", "")) != "qwen2":
            raise Qwen2MachineError("v1 accepts only Hugging Face model_type='qwen2'")
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.tokenizer_revision = str(tokenizer_revision or revision)
        self.identity = ModelIdentity(
            model_id=self.model_id,
            revision=self.revision,
            config_sha256=_sha256_json(_config_payload(config)),
            tokenizer_sha256=_sha256_json(
                {
                    "revision": self.tokenizer_revision,
                    "tokenizer": _tokenizer_payload(tokenizer),
                }
            ),
        )
        self._adapter = _Qwen2NativeAdapter(model)
        # Program contexts are immutable after publication. Cache their exact
        # content identity behind cheap tensor storage/version guards so mrun's
        # repeated transactional validation does not copy the full KV cache to
        # CPU several times per token.
        self._program_identity_cache: dict[
            tuple[Any, ...], tuple[tuple[Any, ...], str]
        ] = {}
        self.registry = ActRegistry()
        self._register_acts()
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> Qwen2AutoregressiveMachine:
        """Load the exact v1 model/tokenizer revision for a resident worker."""

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise Qwen2MachineError("transformers is required to load Qwen") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        return cls(model, tokenizer)

    @property
    def layer_count(self) -> int:
        return self._adapter.layer_count

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype | None:
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            return None

    @property
    def manifest(self) -> dict[str, Any]:
        specs = (
            embed_spec(),
            layer_spec(),
            readout_spec(),
            sample_spec(),
            macro_step_spec("prefill"),
            macro_step_spec("decode_step"),
        )
        return {
            "schema": MACHINE_SCHEMA,
            "mode": "autoregressive",
            "architecture": "dense",
            "family": "dense-llm",
            "machine": "qwen2.autoregressive",
            "model_identity": self.identity.to_dict(),
            "state_schema_id": PREFIX_SCHEMA,
            "kv_cache_schema": KV_CACHE_SCHEMA,
            "layer_count": self.layer_count,
            "vocabulary_size": int(self.model.config.vocab_size),
            "num_kv_heads": int(self._adapter.num_kv_heads),
            "head_dim": int(self._adapter.head_dim),
            "sampler": SAMPLER,
            "batch_size": 1,
            "acts": [spec.to_dict() for spec in specs],
            "termination": ["eos", "max_new_tokens"],
        }

    def _register_acts(self) -> None:
        def embed_runner(values: Mapping[str, Any]):
            hidden, context = self._adapter.embed(
                values["input_ids"], values["attention_mask"], values["kv_cache"]
            )
            return {
                "hidden_state": hidden,
                "derived_context": context,
            }, {
                "numerical_contract_ok": True,
                "use_cache": True,
                "query_length": int(values["input_ids"].shape[1]),
                "cache_position": context.cache_position.tolist(),
            }

        self.registry.register(embed_spec(), IMPLEMENTATION, embed_runner)

        for index in range(self.layer_count):
            implementation = f"{IMPLEMENTATION}.layer.{index}"

            def layer_runner(values: Mapping[str, Any], layer_index: int = index):
                hidden, cache = self._adapter.run_layer(
                    layer_index,
                    values["hidden_state"],
                    values["derived_context"],
                    values["kv_cache"],
                )
                layer = cache.layers[layer_index]
                return {"hidden_state": hidden, "kv_cache": cache}, {
                    "numerical_contract_ok": True,
                    "layer_index": layer_index,
                    "use_cache": True,
                    "kv_sequence_length": layer.sequence_length if layer is not None else 0,
                }

            self.registry.register(layer_spec(), implementation, layer_runner)

        def readout_runner(values: Mapping[str, Any]):
            logits = self._adapter.readout(values["hidden_state"])
            return {"logits": logits}, {
                "numerical_contract_ok": True,
                "logit_shape": list(logits.shape),
            }

        self.registry.register(readout_spec(), IMPLEMENTATION, readout_runner)

        def sample_runner(values: Mapping[str, Any]):
            logits = values["logits"]
            forced = values["forced_token_id"]
            if forced is None:
                sampled = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                source = SAMPLER
            else:
                forced = _as_int(forced, "forced_token_id")
                if forced >= int(logits.shape[-1]):
                    raise Qwen2MachineError("forced_token_id is outside the vocabulary")
                sampled = torch.tensor([[forced]], dtype=torch.long, device=logits.device)
                source = "debug-forced"
            return {"sampled_token_id": sampled, "sample_source": source}, {
                "numerical_contract_ok": True,
                "token_id": int(sampled.item()),
                "forced": source == "debug-forced",
            }

        self.registry.register(sample_spec(), IMPLEMENTATION, sample_runner)

        for macro_phase in ("prefill", "decode_step"):
            expected_phase = macro_phase

            def commit_runner(values: Mapping[str, Any], phase: str = expected_phase):
                if values["phase"] != phase:
                    raise Qwen2MachineError(f"{phase} commit received phase {values['phase']!r}")
                token = values["sampled_token_id"].to(
                    device=values["token_ids"].device, dtype=torch.long
                )
                tokens = torch.cat((values["token_ids"], token), dim=1)
                mask_tail = torch.ones(
                    (1, 1),
                    dtype=values["attention_mask"].dtype,
                    device=values["attention_mask"].device,
                )
                attention_mask = torch.cat((values["attention_mask"], mask_tail), dim=1)
                cursor = int(values["cursor"]) + 1
                eos = int(token.item()) in set(values["eos_token_ids"])
                maximum = cursor >= int(values["max_new_tokens"])
                finish_reason = "eos" if eos else ("max_new_tokens" if maximum else None)
                return {
                    "token_ids": tokens,
                    "attention_mask": attention_mask,
                    "kv_cache": values["kv_cache"],
                    "cursor": cursor,
                    "finished": finish_reason is not None,
                    "finish_reason": finish_reason,
                    "phase": "prefix",
                }, {
                    "numerical_contract_ok": True,
                    "token_id": int(token.item()),
                    "cursor": cursor,
                    "finished": finish_reason is not None,
                    "finish_reason": finish_reason,
                }

            self.registry.register(macro_step_spec(macro_phase), IMPLEMENTATION, commit_runner)

    def _encoded_inputs(
        self,
        prompt: str | None,
        input_ids: torch.Tensor | Sequence[int] | None,
        attention_mask: torch.Tensor | Sequence[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (prompt is None) == (input_ids is None):
            raise Qwen2MachineError("provide exactly one of prompt or input_ids")
        if prompt is not None:
            if self.tokenizer is None:
                raise Qwen2MachineError("prompt input requires a tokenizer")
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(device=self.device, dtype=torch.long)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise Qwen2MachineError("v1 requires one non-empty token row")
        if attention_mask.shape != input_ids.shape:
            raise Qwen2MachineError("attention_mask shape must match input_ids")
        return input_ids, attention_mask

    def _default_eos(self) -> tuple[int, ...]:
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        config_eos = getattr(self.model.config, "eos_token_id", None)
        return _normalise_eos(tokenizer_eos if tokenizer_eos is not None else config_eos)

    def _root_frame(
        self,
        *,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        kv_cache: NormalizedKVCache,
        prompt_length: int,
        cursor: int,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...],
        forced_token_id: int | None,
        phase: str,
        parent_frame_id: str | None = None,
    ) -> Frame:
        return Frame.from_values(
            {
                "token_ids": token_ids,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "kv_cache": kv_cache,
                "prompt_length": prompt_length,
                "cursor": cursor,
                "max_new_tokens": max_new_tokens,
                "eos_token_ids": eos_token_ids,
                "forced_token_id": forced_token_id,
                "phase": phase,
            },
            kinds={
                "token_ids": "token_ids",
                "attention_mask": "attention_mask",
                "input_ids": "token_ids",
                "kv_cache": "kv_cache",
            },
            abis={"kv_cache": {"schema": KV_CACHE_SCHEMA}},
            parent_frame_id=parent_frame_id,
            act="llm.root:v1",
            implementation=IMPLEMENTATION,
            program_fingerprint=self.identity.fingerprint,
            metadata={
                "schema": MACHINE_SCHEMA,
                "family": "dense-llm",
                "phase": phase,
                "model_identity": self.identity.to_dict(),
                "token_cut": int(token_ids.shape[1]),
            },
        )

    def _execute_macro(
        self,
        root: Frame,
        *,
        phase: str,
        prompt_length: int,
    ) -> AutoregressiveStepResult:
        started = time.perf_counter()
        frames: list[Frame] = []
        evidence: list[ActEvidence] = []
        frame, observed = self.registry.run("llm.embed:v1", IMPLEMENTATION, root)
        frames.append(frame)
        evidence.append(observed)
        cut_frames: dict[int, Frame] = {0: frame}
        for index in range(self.layer_count):
            frame, observed = self.registry.run(
                "llm.layer:v1", f"{IMPLEMENTATION}.layer.{index}", frame
            )
            frames.append(frame)
            evidence.append(observed)
            cut_frames[index + 1] = frame
        frame, observed = self.registry.run("llm.readout:v1", IMPLEMENTATION, frame)
        frames.append(frame)
        evidence.append(observed)
        logits = frame.values["logits"]
        frame, observed = self.registry.run("llm.sample:v1", IMPLEMENTATION, frame)
        frames.append(frame)
        evidence.append(observed)
        token_id = int(frame.values["sampled_token_id"].item())
        precommit_frame = frame
        frame, commit_evidence = self.registry.run(f"llm.{phase}:v1", IMPLEMENTATION, frame)
        frames.append(frame)
        macro_evidence = ActEvidence(
            act=f"llm.{phase}:v1",
            implementation=IMPLEMENTATION,
            input_frame_id=root.frame_id,
            output_frame_id=frame.frame_id,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            numerical_contract="native-bit-exact",
            metrics={
                **dict(commit_evidence.metrics),
                "micro_act_count": len(evidence),
                "precommit_frame_id": precommit_frame.frame_id,
                "transactional": True,
            },
        )
        evidence.append(macro_evidence)
        state = AutoregressiveState(
            token_ids=frame.values["token_ids"],
            attention_mask=frame.values["attention_mask"],
            kv_cache=frame.values["kv_cache"],
            prompt_length=prompt_length,
            cursor=int(frame.values["cursor"]),
            max_new_tokens=int(frame.values["max_new_tokens"]),
            eos_token_ids=tuple(frame.values["eos_token_ids"]),
            model_identity=self.identity,
            frame_id=frame.frame_id,
            finished=bool(frame.values["finished"]),
            finish_reason=frame.values["finish_reason"],
        )
        return AutoregressiveStepResult(
            state=state,
            logits=logits,
            token_id=token_id,
            frames=tuple(frames),
            evidence=tuple(evidence),
            cut_frames=cut_frames,
            phase=phase,
            root_frame=root,
        )

    @torch.inference_mode()
    def prefill(
        self,
        prompt: str | None = None,
        *,
        input_ids: torch.Tensor | Sequence[int] | None = None,
        attention_mask: torch.Tensor | Sequence[int] | None = None,
        max_new_tokens: int = 128,
        eos_token_ids: int | Sequence[int] | None = None,
        forced_token_id: int | None = None,
    ) -> AutoregressiveStepResult:
        tokens, mask = self._encoded_inputs(prompt, input_ids, attention_mask)
        maximum = _as_int(max_new_tokens, "max_new_tokens", minimum=1)
        eos = self._default_eos() if eos_token_ids is None else _normalise_eos(eos_token_ids)
        cache = NormalizedKVCache.empty(self.layer_count)
        root = self._root_frame(
            token_ids=tokens,
            attention_mask=mask,
            input_ids=tokens,
            kv_cache=cache,
            prompt_length=int(tokens.shape[1]),
            cursor=0,
            max_new_tokens=maximum,
            eos_token_ids=eos,
            forced_token_id=forced_token_id,
            phase="prefill",
        )
        return self._execute_macro(root, phase="prefill", prompt_length=int(tokens.shape[1]))

    @torch.inference_mode()
    def step(
        self,
        state: AutoregressiveState,
        *,
        forced_token_id: int | None = None,
    ) -> AutoregressiveStepResult:
        if state.model_identity != self.identity:
            raise Qwen2MachineError("prefix model identity does not match resident machine")
        if state.finished:
            raise Qwen2MachineError("cannot step a terminated prefix")
        root = self._root_frame(
            token_ids=state.token_ids,
            attention_mask=state.attention_mask,
            input_ids=state.token_ids[:, -1:],
            kv_cache=state.kv_cache,
            prompt_length=state.prompt_length,
            cursor=state.cursor,
            max_new_tokens=state.max_new_tokens,
            eos_token_ids=state.eos_token_ids,
            forced_token_id=forced_token_id,
            phase="decode_step",
            parent_frame_id=state.frame_id,
        )
        return self._execute_macro(root, phase="decode_step", prompt_length=state.prompt_length)

    decode_step = step

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None = None,
        *,
        input_ids: torch.Tensor | Sequence[int] | None = None,
        attention_mask: torch.Tensor | Sequence[int] | None = None,
        max_new_tokens: int = 128,
        eos_token_ids: int | Sequence[int] | None = None,
    ) -> AutoregressiveGenerationResult:
        first = self.prefill(
            prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        steps = [first]
        state = first.state
        while not state.finished:
            current = self.step(state)
            steps.append(current)
            state = current.state
        return AutoregressiveGenerationResult(state=state, steps=tuple(steps))

    def clone_state(self, state: AutoregressiveState) -> AutoregressiveState:
        return state.clone()

    def fork_state(self, state: AutoregressiveState) -> AutoregressiveState:
        return state.fork()

    def termination(self, state: AutoregressiveState) -> bool:
        return state.finished

    def compatibility_key(self) -> tuple[Any, ...]:
        return (
            "dense-llm",
            self.identity.fingerprint,
            str(self.dtype),
            str(self.device),
            KV_CACHE_SCHEMA,
            SAMPLER,
        )

    def _prefix_slot_names(self) -> set[str]:
        names = {
            "schema",
            "state_schema_id",
            "model_identity",
            "frame_id",
            "sampler",
            "token_ids",
            "attention_mask",
            "kv_cache_schema",
            "kv_layer_count",
            "prompt_length",
            "cursor",
            "max_new_tokens",
            "eos_token_ids",
            "finished",
            "finish_reason",
            "token_cut",
        }
        for index in range(self.layer_count):
            names.add(f"kv_cache.layer_{index:03d}.key")
            names.add(f"kv_cache.layer_{index:03d}.value")
        return names

    def prefix_slot_specs(self) -> dict[str, Any]:
        """Return the fixed table consumed by ``StateCutCheckpoint``."""

        from .statecut_checkpoint import StateCutSlotSpec

        json_shapes = {
            "schema": "schema_id",
            "state_schema_id": "schema_id",
            "model_identity": "model_identity",
            "frame_id": "frame_id",
            "sampler": "sampler_config",
            "kv_cache_schema": "schema_id",
            "kv_layer_count": "scalar",
            "prompt_length": "scalar",
            "cursor": "scalar",
            "max_new_tokens": "scalar",
            "eos_token_ids": "token_id_list",
            "finished": "scalar",
            "finish_reason": "scalar",
            "token_cut": "scalar",
        }
        specs = {
            name: StateCutSlotSpec(kind="json", shape_class=shape)
            for name, shape in json_shapes.items()
        }
        specs["token_ids"] = StateCutSlotSpec(
            kind="tensor", shape_class="batch_sequence", dtype=str(torch.int64)
        )
        specs["attention_mask"] = StateCutSlotSpec(
            kind="tensor", shape_class="batch_sequence", dtype=str(torch.int64)
        )
        if self.dtype is None:
            raise Qwen2MachineError("resident model has no parameter dtype")
        for index in range(self.layer_count):
            for side in ("key", "value"):
                specs[f"kv_cache.layer_{index:03d}.{side}"] = StateCutSlotSpec(
                    kind="tensor", shape_class="qwen2_kv_cache", dtype=str(self.dtype)
                )
        if set(specs) != self._prefix_slot_names():  # pragma: no cover - local invariant
            raise AssertionError("Qwen prefix slot table is incomplete")
        return specs

    def to_statecut_checkpoint(
        self,
        state: AutoregressiveState,
        *,
        checkpoint_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Build a durable token-boundary checkpoint from one committed prefix."""

        from .statecut_checkpoint import StateCutBoundary, StateCutCheckpoint

        return StateCutCheckpoint(
            checkpoint_id=checkpoint_id,
            family="dense-llm",
            state_schema_id=PREFIX_SCHEMA,
            model_identity=self.identity.to_dict(),
            boundary=StateCutBoundary(
                kind="token_cut", index=state.cursor, total=state.max_new_tokens
            ),
            slots=self.export_prefix_slots(state),
            slot_specs=self.prefix_slot_specs(),
            metadata={
                "derived_slots": ["position_ids", "cache_position", "rope_inputs"],
                **dict(metadata or {}),
            },
        )

    def import_prefix_slots(
        self, slots: Mapping[str, Any], *, clone: bool = False
    ) -> AutoregressiveState:
        observed_names = set(slots)
        expected_names = self._prefix_slot_names()
        if observed_names != expected_names:
            raise Qwen2MachineError(
                "prefix slots do not match the fixed Qwen schema: "
                f"missing={sorted(expected_names - observed_names)}, "
                f"extra={sorted(observed_names - expected_names)}"
            )
        if slots.get("schema") != PREFIX_SCHEMA:
            raise Qwen2MachineError(f"unsupported prefix schema {slots.get('schema')!r}")
        if slots.get("state_schema_id") != PREFIX_SCHEMA:
            raise Qwen2MachineError("prefix state_schema_id mismatch")
        identity_payload = slots.get("model_identity")
        if not isinstance(identity_payload, Mapping):
            raise Qwen2MachineError("prefix has no model_identity mapping")
        identity = ModelIdentity.from_mapping(identity_payload)
        if identity != self.identity:
            raise Qwen2MachineError("prefix model/config/tokenizer identity mismatch")
        if slots.get("kv_cache_schema") != KV_CACHE_SCHEMA:
            raise Qwen2MachineError("prefix KV cache schema mismatch")
        layer_count = _as_int(slots.get("kv_layer_count"), "kv_layer_count", minimum=1)
        if layer_count != self.layer_count:
            raise Qwen2MachineError("prefix KV layer count does not match resident model")
        token_ids, attention_mask = slots.get("token_ids"), slots.get("attention_mask")
        if not isinstance(token_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise Qwen2MachineError("prefix token_ids/attention_mask must be tensors")
        if token_ids.dtype != torch.long or attention_mask.dtype != torch.long:
            raise Qwen2MachineError("prefix token_ids/attention_mask dtype must be torch.int64")
        for index in range(layer_count):
            key = slots[f"kv_cache.layer_{index:03d}.key"]
            value = slots[f"kv_cache.layer_{index:03d}.value"]
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise Qwen2MachineError(f"prefix cache layer {index} must contain tensors")
            if key.shape != value.shape or key.ndim != 4:
                raise Qwen2MachineError(f"prefix cache layer {index} has invalid KV shape")
            if (
                int(key.shape[0]) != 1
                or int(key.shape[1]) != self._adapter.num_kv_heads
                or int(key.shape[-1]) != self._adapter.head_dim
            ):
                raise Qwen2MachineError(
                    f"prefix cache layer {index} geometry must be "
                    f"[1,{self._adapter.num_kv_heads},seq,{self._adapter.head_dim}], "
                    f"got {list(key.shape)}"
                )
        cache = NormalizedKVCache.from_tensor_slots(
            slots,
            layer_count=layer_count,
            clone=clone,
            device=self.device,
            expected_dtype=self.dtype,
        )
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=self.device)
        state = AutoregressiveState(
            token_ids=token_ids.detach().clone() if clone else token_ids,
            attention_mask=(attention_mask.detach().clone() if clone else attention_mask),
            kv_cache=cache,
            prompt_length=_as_int(slots.get("prompt_length"), "prompt_length", minimum=1),
            cursor=_as_int(slots.get("cursor"), "cursor", minimum=1),
            max_new_tokens=_as_int(slots.get("max_new_tokens"), "max_new_tokens", minimum=1),
            eos_token_ids=_normalise_eos(slots.get("eos_token_ids", ())),
            model_identity=identity,
            frame_id=str(slots.get("frame_id") or ""),
            finished=bool(slots.get("finished", False)),
            finish_reason=slots.get("finish_reason"),
            sampler=str(slots.get("sampler", "")),
        )
        if int(slots.get("token_cut", -1)) != state.cursor:
            raise Qwen2MachineError("prefix token_cut does not match cursor")
        return state

    def export_prefix_slots(
        self, state: AutoregressiveState, *, clone: bool = False
    ) -> dict[str, Any]:
        if state.model_identity != self.identity:
            raise Qwen2MachineError("cannot export a prefix from another machine")
        return state.export_prefix_slots(clone=clone)

    @torch.inference_mode()
    def replay_layer_cut(
        self,
        step: AutoregressiveStepResult,
        cut: int,
        *,
        intervention: Callable[[Frame], Frame] | None = None,
    ) -> LayerReplayResult:
        cut = _as_int(cut, "cut")
        if cut > self.layer_count or cut not in step.cut_frames:
            raise Qwen2MachineError(f"layer cut must be between 0 and {self.layer_count} inclusive")
        frame = step.cut_frames[cut]
        if intervention is not None:
            if not callable(intervention):
                raise Qwen2MachineError("layer-cut intervention must be callable")
            intervened = intervention(frame)
            if not isinstance(intervened, Frame):
                raise Qwen2MachineError("layer-cut intervention must return a Frame")
            if intervened.program_fingerprint != self.identity.fingerprint:
                raise Qwen2MachineError("intervention returned a frame from another program")
            intervened.require(layer_spec().reads)
            frame = intervened
        frames: list[Frame] = []
        evidence: list[ActEvidence] = []
        for index in range(cut, self.layer_count):
            frame, observed = self.registry.run(
                "llm.layer:v1", f"{IMPLEMENTATION}.layer.{index}", frame
            )
            frames.append(frame)
            evidence.append(observed)
        frame, observed = self.registry.run("llm.readout:v1", IMPLEMENTATION, frame)
        frames.append(frame)
        evidence.append(observed)
        logits = frame.values["logits"]
        return LayerReplayResult(
            cut=cut,
            logits=logits,
            reference_logits=step.logits,
            exact=torch.equal(logits, step.logits),
            frames=tuple(frames),
            evidence=tuple(evidence),
        )

    # mrun family lifecycle hooks. Payloads intentionally contain no model object.
    def compile_program_context(
        self,
        prompt: str | None = None,
        *,
        input_ids: torch.Tensor | Sequence[int] | None = None,
        attention_mask: torch.Tensor | Sequence[int] | None = None,
        prefix_slots: Mapping[str, Any] | None = None,
        max_new_tokens: int = 128,
        eos_token_ids: int | Sequence[int] | None = None,
        sampler: str = SAMPLER,
    ) -> dict[str, Any]:
        if sampler != SAMPLER:
            raise Qwen2MachineError("v1 supports only greedy sampling")
        if prefix_slots is not None:
            if input_ids is not None or attention_mask is not None:
                raise Qwen2MachineError(
                    "prefix_slots cannot be combined with input_ids or attention_mask"
                )
            state = self.import_prefix_slots(prefix_slots)
            if state.finished:
                raise Qwen2MachineError("cannot compile a terminated durable prefix")
            if prompt is not None:
                prompt_tokens, prompt_mask = self._encoded_inputs(prompt, None, None)
                prompt_slice = state.token_ids[:, : state.prompt_length]
                mask_slice = state.attention_mask[:, : state.prompt_length]
                if not torch.equal(prompt_tokens, prompt_slice) or not torch.equal(
                    prompt_mask, mask_slice
                ):
                    raise Qwen2MachineError(
                        "durable prefix does not begin with the supplied prompt"
                    )
            return {
                "schema": PROGRAM_CONTEXT_SCHEMA,
                "phase": "prefix",
                "prefix": state.export_prefix_slots(),
            }
        tokens, mask = self._encoded_inputs(prompt, input_ids, attention_mask)
        eos = self._default_eos() if eos_token_ids is None else _normalise_eos(eos_token_ids)
        return {
            "schema": PROGRAM_CONTEXT_SCHEMA,
            "phase": "prompt",
            "model_identity": self.identity.to_dict(),
            "token_ids": tokens,
            "attention_mask": mask,
            "max_new_tokens": _as_int(max_new_tokens, "max_new_tokens", minimum=1),
            "eos_token_ids": eos,
            "sampler": SAMPLER,
        }

    def step_program_context(
        self,
        payload: Mapping[str, Any],
        *,
        forced_token_id: int | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, Any]]:
        if payload.get("schema") != PROGRAM_CONTEXT_SCHEMA:
            raise Qwen2MachineError("unsupported mrun program context")
        phase = payload.get("phase")
        if phase == "prompt":
            result = self.prefill(
                input_ids=payload["token_ids"],
                attention_mask=payload["attention_mask"],
                max_new_tokens=int(payload["max_new_tokens"]),
                eos_token_ids=payload.get("eos_token_ids", ()),
                forced_token_id=forced_token_id,
            )
        elif phase == "prefix":
            prefix = payload.get("prefix")
            if not isinstance(prefix, Mapping):
                raise Qwen2MachineError("prefix context has no durable prefix mapping")
            result = self.step(self.import_prefix_slots(prefix), forced_token_id=forced_token_id)
        else:
            raise Qwen2MachineError(f"unsupported program context phase {phase!r}")
        next_payload = {
            "schema": PROGRAM_CONTEXT_SCHEMA,
            "phase": "prefix",
            "prefix": result.state.export_prefix_slots(),
        }
        metadata = {
            "family": "dense-llm",
            "act": result.macro_evidence.act,
            "token_id": result.token_id,
            "cursor": result.state.cursor,
            "finished": result.state.finished,
            "finish_reason": result.state.finish_reason,
            "frame": result.final_frame.to_dict(),
            "evidence": [item.to_dict() for item in result.evidence],
            "use_cache": True,
            # These two values are live-worker evidence and are intentionally
            # absent from ``next_payload`` / StateCut persistence.
            "logits": result.logits,
            "step_result": result,
        }
        return result.token_id, next_payload, metadata

    def _program_context_descriptor(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("schema") != PROGRAM_CONTEXT_SCHEMA:
            raise Qwen2MachineError("unsupported mrun program context")
        phase = str(payload.get("phase") or "")
        if phase == "prompt":
            identity_payload = payload.get("model_identity")
            if not isinstance(identity_payload, Mapping):
                raise Qwen2MachineError("prompt context has no model identity")
            identity = ModelIdentity.from_mapping(identity_payload)
            tokens, mask = payload.get("token_ids"), payload.get("attention_mask")
            if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise Qwen2MachineError("prompt context has no token/mask tensor")
            descriptor = {
                "schema": PROGRAM_CONTEXT_SCHEMA,
                "phase": phase,
                "model_identity": identity.to_dict(),
                "token_ids": _tensor_sha256(tokens),
                "attention_mask": _tensor_sha256(mask),
                "max_new_tokens": _as_int(
                    payload.get("max_new_tokens"), "max_new_tokens", minimum=1
                ),
                "eos_token_ids": _normalise_eos(payload.get("eos_token_ids", ())),
                "sampler": str(payload.get("sampler") or ""),
            }
        elif phase == "prefix":
            prefix = payload.get("prefix")
            if not isinstance(prefix, Mapping):
                raise Qwen2MachineError("prefix context has no durable prefix mapping")
            identity_payload = prefix.get("model_identity")
            if not isinstance(identity_payload, Mapping):
                raise Qwen2MachineError("prefix context has no model identity")
            identity = ModelIdentity.from_mapping(identity_payload)
            tokens, mask = prefix.get("token_ids"), prefix.get("attention_mask")
            if not isinstance(tokens, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise Qwen2MachineError("prefix context has no token/mask tensor")
            layer_count = _as_int(prefix.get("kv_layer_count"), "kv_layer_count", minimum=1)
            kv_layers: list[dict[str, Any]] = []
            for index in range(layer_count):
                key = prefix.get(f"kv_cache.layer_{index:03d}.key")
                value = prefix.get(f"kv_cache.layer_{index:03d}.value")
                if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                    raise Qwen2MachineError(f"prefix cache layer {index} is incomplete")
                kv_layers.append(
                    {
                        "key": _tensor_sha256(key),
                        "value": _tensor_sha256(value),
                        "geometry": _tensor_summary(key),
                    }
                )
            descriptor = {
                "schema": PROGRAM_CONTEXT_SCHEMA,
                "phase": phase,
                "prefix_schema": prefix.get("schema"),
                "state_schema_id": prefix.get("state_schema_id"),
                "model_identity": identity.to_dict(),
                "frame_id": str(prefix.get("frame_id") or ""),
                "token_ids": _tensor_sha256(tokens),
                "attention_mask": _tensor_sha256(mask),
                "prompt_length": _as_int(prefix.get("prompt_length"), "prompt_length", minimum=1),
                "cursor": _as_int(prefix.get("cursor"), "cursor", minimum=1),
                "max_new_tokens": _as_int(
                    prefix.get("max_new_tokens"), "max_new_tokens", minimum=1
                ),
                "eos_token_ids": _normalise_eos(prefix.get("eos_token_ids", ())),
                "sampler": str(prefix.get("sampler") or ""),
                "kv_cache_schema": str(prefix.get("kv_cache_schema") or ""),
                "kv_layer_count": layer_count,
                "kv_layers": kv_layers,
                "finished": bool(prefix.get("finished")),
                "finish_reason": prefix.get("finish_reason"),
                "token_cut": _as_int(prefix.get("token_cut"), "token_cut", minimum=1),
            }
        else:
            raise Qwen2MachineError(f"unsupported program context phase {phase!r}")
        if identity != self.identity:
            raise Qwen2MachineError("program context model identity mismatch")
        if descriptor["sampler"] != SAMPLER:
            raise Qwen2MachineError("program context sampler mismatch")
        return descriptor

    @staticmethod
    def _tensor_identity_guard(value: Any) -> tuple[Any, ...]:
        if not isinstance(value, torch.Tensor):
            return ("not-a-tensor", type(value).__name__)
        try:
            version = int(value._version)
        except RuntimeError:  # inference tensors intentionally omit version counters
            version = None
        return (
            tuple(int(item) for item in value.shape),
            str(value.dtype),
            str(value.device),
            int(value.data_ptr()),
            version,
        )

    def _program_context_guard(
        self, payload: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        """Return a cache key plus a mutation-sensitive, device-local guard."""

        phase = str(payload.get("phase") or "")
        if phase == "prompt":
            tokens, mask = payload.get("token_ids"), payload.get("attention_mask")
            cache_key = (phase, id(tokens), id(mask))
            guard = (
                payload.get("schema"),
                phase,
                _jsonable(payload.get("model_identity")),
                self._tensor_identity_guard(tokens),
                self._tensor_identity_guard(mask),
                payload.get("max_new_tokens"),
                tuple(payload.get("eos_token_ids") or ()),
                payload.get("sampler"),
            )
            return cache_key, guard
        if phase != "prefix":
            return ((phase, id(payload)), (payload.get("schema"), phase))
        prefix = payload.get("prefix")
        if not isinstance(prefix, Mapping):
            return ((phase, id(prefix)), (payload.get("schema"), phase, "invalid"))
        layer_count = prefix.get("kv_layer_count")
        tensor_guards = [
            self._tensor_identity_guard(prefix.get("token_ids")),
            self._tensor_identity_guard(prefix.get("attention_mask")),
        ]
        if isinstance(layer_count, int) and not isinstance(layer_count, bool):
            for index in range(max(0, layer_count)):
                tensor_guards.extend(
                    (
                        self._tensor_identity_guard(
                            prefix.get(f"kv_cache.layer_{index:03d}.key")
                        ),
                        self._tensor_identity_guard(
                            prefix.get(f"kv_cache.layer_{index:03d}.value")
                        ),
                    )
                )
        cache_key = (phase, str(prefix.get("frame_id") or ""))
        guard = (
            payload.get("schema"),
            phase,
            prefix.get("schema"),
            prefix.get("state_schema_id"),
            _jsonable(prefix.get("model_identity")),
            str(prefix.get("frame_id") or ""),
            prefix.get("sampler"),
            prefix.get("kv_cache_schema"),
            layer_count,
            prefix.get("prompt_length"),
            prefix.get("cursor"),
            prefix.get("max_new_tokens"),
            tuple(prefix.get("eos_token_ids") or ()),
            prefix.get("finished"),
            prefix.get("finish_reason"),
            prefix.get("token_cut"),
            tuple(tensor_guards),
        )
        return cache_key, guard

    def _cached_program_context_checksum(self, payload: Mapping[str, Any]) -> str:
        cache_key, guard = self._program_context_guard(payload)
        cached = self._program_identity_cache.get(cache_key)
        if cached is not None and cached[0] == guard:
            return cached[1]
        checksum = _sha256_json(self._program_context_descriptor(payload))
        if len(self._program_identity_cache) >= 4096:
            self._program_identity_cache.clear()
        self._program_identity_cache[cache_key] = (guard, checksum)
        return checksum

    def program_context_checksum(self, payload: Mapping[str, Any]) -> str:
        """Hash the complete immutable prompt/prefix context, including KV bytes."""

        return self._cached_program_context_checksum(payload)

    def program_context_key(self, payload: Mapping[str, Any]) -> str:
        return f"llm-prefix:{self._cached_program_context_checksum(payload)}"

    def program_compatibility_key(
        self,
        _session: Any,
        _state: Any,
        _payload: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        return self.compatibility_key()

    def program_step_terminal(
        self,
        _session: Any,
        _source_state: Any,
        next_payload: Mapping[str, Any],
        _output: Any,
        _metadata: Mapping[str, Any],
    ) -> bool:
        prefix = next_payload.get("prefix")
        if not isinstance(prefix, Mapping):
            raise Qwen2MachineError("terminal check requires a prefix payload")
        return bool(prefix.get("finished"))


@dataclass(slots=True)
class Qwen2AutoregressiveContext:
    """Convenient mutable row wrapper; resident weights remain on ``machine``."""

    machine: Qwen2AutoregressiveMachine
    state: AutoregressiveState | None = None
    last_result: AutoregressiveStepResult | None = None

    def prefill(self, prompt: str | None = None, **kwargs: Any) -> AutoregressiveStepResult:
        result = self.machine.prefill(prompt, **kwargs)
        self.state, self.last_result = result.state, result
        return result

    def step(self, *, forced_token_id: int | None = None) -> AutoregressiveStepResult:
        if self.state is None:
            raise Qwen2MachineError("context must be prefetched before step")
        result = self.machine.step(self.state, forced_token_id=forced_token_id)
        self.state, self.last_result = result.state, result
        return result

    def generate(self, prompt: str | None = None, **kwargs: Any) -> AutoregressiveGenerationResult:
        if self.state is None:
            first = self.prefill(prompt, **kwargs)
            steps = [first]
        else:
            if prompt is not None or kwargs:
                raise Qwen2MachineError("continuing a context does not accept a new prompt/config")
            steps = []
        while self.state is not None and not self.state.finished:
            steps.append(self.step())
        if self.state is None:
            raise Qwen2MachineError("generation produced no state")
        return AutoregressiveGenerationResult(self.state, tuple(steps))

    def fork(self) -> Qwen2AutoregressiveContext:
        return Qwen2AutoregressiveContext(
            self.machine,
            None if self.state is None else self.state.fork(),
            None,
        )

    def clone(self) -> Qwen2AutoregressiveContext:
        """Deep-copy live payloads for callers that require mutable isolation."""

        return Qwen2AutoregressiveContext(
            self.machine,
            None if self.state is None else self.state.clone(),
            None,
        )

    def export_prefix_slots(self, *, clone: bool = False) -> dict[str, Any]:
        if self.state is None:
            raise Qwen2MachineError("context has no prefix to export")
        return self.machine.export_prefix_slots(self.state, clone=clone)

    def import_prefix_slots(self, slots: Mapping[str, Any], *, clone: bool = False) -> None:
        self.state = self.machine.import_prefix_slots(slots, clone=clone)
        self.last_result = None

    def replay_layer_cut(
        self,
        cut: int,
        *,
        intervention: Callable[[Frame], Frame] | None = None,
    ) -> LayerReplayResult:
        if self.last_result is None:
            raise Qwen2MachineError("context has no live step trace to replay")
        return self.machine.replay_layer_cut(self.last_result, cut, intervention=intervention)


__all__ = [
    "DERIVED_FINGERPRINT_CONTENT",
    "DERIVED_FINGERPRINT_RECIPE",
    "IMPLEMENTATION",
    "KV_FINGERPRINT_APPEND",
    "KV_FINGERPRINT_CONTENT",
    "KV_CACHE_SCHEMA",
    "MACHINE_SCHEMA",
    "MODEL_ID",
    "MODEL_REVISION",
    "PREFIX_SCHEMA",
    "PROGRAM_CONTEXT_SCHEMA",
    "SAMPLER",
    "AutoregressiveGenerationResult",
    "AutoregressiveState",
    "AutoregressiveStepResult",
    "DerivedLayerContext",
    "KVLayerState",
    "LayerReplayResult",
    "ModelIdentity",
    "NormalizedKVCache",
    "Qwen2AutoregressiveContext",
    "Qwen2AutoregressiveMachine",
    "Qwen2MachineError",
    "embed_spec",
    "layer_spec",
    "macro_step_spec",
    "readout_spec",
    "sample_spec",
]

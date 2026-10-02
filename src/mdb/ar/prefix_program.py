"""Typed, family-neutral compiled prefix programs for cached AR decoders.

The program contains two related boundaries:

* ``full`` is the cache after the complete prompt plus the prompt's final
  next-token logits.  Replaying it skips prefill and is the exact compiler
  serving boundary.
* ``query`` is the cache immediately before the prompt's final token plus that
  pending token.  It is the causal microscope boundary: K/V interventions are
  consumed before the first generated token is selected.

The tensor coordinates remain family-local.  This module only standardizes the
role and custody contract shared by Hugging Face ``DynamicCache`` decoders.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch

from mdb.tensor_hash import tensor_bytes_sha256

AR_PREFIX_PROGRAM_SCHEMA = "saturn-ar-prefix-program-v1"


class ArPrefixProgramError(ValueError):
    """A compiled prefix violates its typed cache or identity contract."""


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def dynamic_cache_pairs(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return initialized K/V layers without copying or accepting legacy tuples."""

    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list) or not layers:
        raise ArPrefixProgramError("compiled prefix requires initialized DynamicCache layers")
    result = []
    for index, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ArPrefixProgramError(f"cache layer {index} is not initialized")
        if (key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3]
                or min(key.shape[-1], value.shape[-1]) < 1):
            raise ArPrefixProgramError(f"cache layer {index} has unsupported K/V geometry")
        result.append((key, value))
    return tuple(result)


def clone_dynamic_cache(cache: Any, *, config: Any) -> Any:
    """Deep-clone a modern DynamicCache and refuse aliased branch storage."""

    source = dynamic_cache_pairs(cache)
    data = [(key.clone(), value.clone()) for key, value in source]
    try:
        cloned = type(cache)(ddp_cache_data=data, config=config)
    except TypeError as exc:
        raise ArPrefixProgramError(
            "unsupported DynamicCache constructor; refusing aliased replay"
        ) from exc
    source_ptrs = {tensor.data_ptr() for pair in source for tensor in pair}
    clone_ptrs = {tensor.data_ptr() for pair in dynamic_cache_pairs(cloned) for tensor in pair}
    if source_ptrs & clone_ptrs:
        raise ArPrefixProgramError("replayed DynamicCache aliases compiler custody")
    return cloned


def _cpu_layers(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            key.detach().cpu().clone().contiguous(),
            value.detach().cpu().clone().contiguous(),
        )
        for key, value in dynamic_cache_pairs(cache)
    )


def _layer_manifest(
    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> list[dict[str, Any]]:
    return [
        {
            "layer": index,
            "key": {
                "shape": list(key.shape),
                "dtype": str(key.dtype),
                "sha256": tensor_bytes_sha256(key),
            },
            "value": {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_bytes_sha256(value),
            },
        }
        for index, (key, value) in enumerate(layers)
    ]


@dataclass(frozen=True, slots=True)
class ArPrefixProgram:
    """A prompt-static AR source object plus a pre-query causal StateCut."""

    model_id: str
    revision: str
    tokenizer_sha256: str
    source_token_ids: tuple[int, ...]
    full_cache_layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    full_next_logits: torch.Tensor
    query_cache_layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    pending_token_id: int
    cache_abi: str = "transformers-dynamic-cache-post-position-encoding-kv-v1"

    def __post_init__(self) -> None:
        if self.cache_abi not in {
            "transformers-dynamic-cache-post-position-encoding-kv-v1",
            "saturn-recipient-local-compact-value-kv-v1",
        }:
            raise ArPrefixProgramError("unsupported prefix cache ABI")
        if not self.model_id or not self.revision or not self.tokenizer_sha256:
            raise ArPrefixProgramError("prefix program identity fields must be non-empty")
        if len(self.tokenizer_sha256) != 64:
            raise ArPrefixProgramError("tokenizer identity must be a SHA-256")
        if len(self.source_token_ids) < 2 or any(token < 0 for token in self.source_token_ids):
            raise ArPrefixProgramError("prefix program requires at least two valid token IDs")
        if self.pending_token_id != self.source_token_ids[-1]:
            raise ArPrefixProgramError("pending token does not close the source token sequence")
        if not self.full_cache_layers or len(self.full_cache_layers) != len(
            self.query_cache_layers
        ):
            raise ArPrefixProgramError("full/query cache layer counts differ")
        full_length = len(self.source_token_ids)
        query_length = full_length - 1
        for boundary, layers, expected in (
            ("full", self.full_cache_layers, full_length),
            ("query", self.query_cache_layers, query_length),
        ):
            for index, (key, value) in enumerate(layers):
                compact = self.cache_abi == "saturn-recipient-local-compact-value-kv-v1"
                if (key.ndim != 4 or value.ndim != 4
                        or key.shape[:3] != value.shape[:3]
                        or min(key.shape[-1], value.shape[-1]) < 1
                        or (not compact and key.shape != value.shape)):
                    raise ArPrefixProgramError(
                        f"{boundary} layer {index} has unsupported K/V geometry"
                    )
                if int(key.shape[0]) != 1 or int(key.shape[-2]) != expected:
                    raise ArPrefixProgramError(
                        f"{boundary} layer {index} sequence geometry differs from tokens"
                    )
                if key.device.type != "cpu" or value.device.type != "cpu":
                    raise ArPrefixProgramError("durable prefix tensors must be CPU resident")
        if self.full_next_logits.ndim != 2 or int(self.full_next_logits.shape[0]) != 1:
            raise ArPrefixProgramError("next-token logits must have shape [1, vocab]")
        if self.full_next_logits.device.type != "cpu":
            raise ArPrefixProgramError("durable next-token logits must be CPU resident")

    @classmethod
    def capture(
        cls,
        *,
        model_id: str,
        revision: str,
        tokenizer_sha256: str,
        source_token_ids: tuple[int, ...],
        full_cache: Any,
        full_next_logits: torch.Tensor,
        query_cache: Any,
        cache_abi: str = "transformers-dynamic-cache-post-position-encoding-kv-v1",
    ) -> ArPrefixProgram:
        return cls(
            model_id=model_id,
            revision=revision,
            tokenizer_sha256=tokenizer_sha256,
            source_token_ids=tuple(int(token) for token in source_token_ids),
            full_cache_layers=_cpu_layers(full_cache),
            full_next_logits=full_next_logits.detach().cpu().clone().contiguous(),
            query_cache_layers=_cpu_layers(query_cache),
            pending_token_id=int(source_token_ids[-1]),
            cache_abi=cache_abi,
        )

    def _layers(self, boundary: str) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if boundary == "full":
            return self.full_cache_layers
        if boundary == "query":
            return self.query_cache_layers
        raise ArPrefixProgramError(f"unknown prefix boundary {boundary!r}")

    def rehydrate_cache(
        self,
        *,
        boundary: str,
        cache_type: type,
        config: Any,
        device: torch.device | str,
        dtype: torch.dtype,
        layers: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> Any:
        source = self._layers(boundary) if layers is None else layers
        data = [
            (
                key.to(device=device, dtype=dtype).clone(),
                value.to(device=device, dtype=dtype).clone(),
            )
            for key, value in source
        ]
        try:
            return cache_type(ddp_cache_data=data, config=config)
        except TypeError as exc:
            raise ArPrefixProgramError(
                "recipient DynamicCache cannot hydrate compiled prefix object code"
            ) from exc

    @property
    def tensor_bytes(self) -> int:
        tensors = [self.full_next_logits]
        for pair in (*self.full_cache_layers, *self.query_cache_layers):
            tensors.extend(pair)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def manifest(self) -> dict[str, Any]:
        body = {
            "schema": AR_PREFIX_PROGRAM_SCHEMA,
            "model_id": self.model_id,
            "revision": self.revision,
            "tokenizer_sha256": self.tokenizer_sha256,
            "cache_abi": self.cache_abi,
            "source_token_ids": list(self.source_token_ids),
            "source_token_count": len(self.source_token_ids),
            "pending_token_id": self.pending_token_id,
            "full_cache": _layer_manifest(self.full_cache_layers),
            "query_cache": _layer_manifest(self.query_cache_layers),
            "full_next_logits": {
                "shape": list(self.full_next_logits.shape),
                "dtype": str(self.full_next_logits.dtype),
                "sha256": tensor_bytes_sha256(self.full_next_logits),
            },
            "tensor_bytes": self.tensor_bytes,
            "semantic_contract": {
                "key": "content address with family-local positional encoding already applied",
                "value": "payload retrieved by the live query",
                "full_boundary": "exact prompt-prefill serving object",
                "query_boundary": "pre-final-prompt-token causal microscope StateCut",
            },
        }
        return {**body, "fingerprint": _digest(body)}

    def tensor_archive(self) -> dict[str, torch.Tensor]:
        result = {"full.next_logits": self.full_next_logits.contiguous()}
        for boundary, layers in (
            ("full", self.full_cache_layers),
            ("query", self.query_cache_layers),
        ):
            for index, (key, value) in enumerate(layers):
                result[f"{boundary}.layer.{index:03d}.key"] = key.contiguous()
                result[f"{boundary}.layer.{index:03d}.value"] = value.contiguous()
        return result

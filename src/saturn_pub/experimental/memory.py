"""Portable shared Qwen history consumed by segmented, globally normalized attention.

Keys retain their original post-RoPE positions. This exact-history example does not
relocate pages, concatenate independently compiled histories, or extend context limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AttentionInterface
from transformers.cache_utils import Cache
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

from ..values import describe, digest


@dataclass(frozen=True)
class Segments:
    ancestor: torch.Tensor
    delta: torch.Tensor


def segmented_attention(
    module: Any,
    query: torch.Tensor,
    key: Segments,
    value: Segments,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[torch.Tensor, None]:
    if not isinstance(key, Segments) or not isinstance(value, Segments):
        raise ValueError("segmented attention requires the paired shared-history cache")
    if query.shape[0] != 1 or query.shape[-2] != 1 or dropout:
        raise ValueError("shared memory supports batch-one one-token inference only")
    if attention_mask is not None and torch.count_nonzero(attention_mask):
        raise ValueError("masked or padded shared history is not supported")
    if query.shape[1] % key.ancestor.shape[1]:
        raise ValueError("GQA head geometry mismatch")
    repeats = query.shape[1] // key.ancestor.shape[1]
    q = query.float()
    maximum = torch.full((*q.shape[:-1], 1), -torch.inf, device=q.device)
    mass = torch.zeros_like(maximum)
    weighted = torch.zeros_like(q)
    for keys, values in ((key.ancestor, value.ancestor), (key.delta, value.delta)):
        if keys.shape[-2] == 0:
            continue
        keys = keys.repeat_interleave(repeats, dim=1).float()
        values = values.repeat_interleave(repeats, dim=1).float()
        scores = torch.matmul(q, keys.transpose(-1, -2)) * scaling
        new_maximum = torch.maximum(maximum, scores.max(-1, keepdim=True).values)
        old_scale = torch.exp(maximum - new_maximum)
        probabilities = torch.exp(scores - new_maximum)
        weighted = weighted * old_scale + torch.matmul(probabilities, values)
        mass = mass * old_scale + probabilities.sum(-1, keepdim=True)
        maximum = new_maximum
    result = (weighted / mass).to(query.dtype)
    return result.transpose(1, 2).contiguous(), None


_ATTENTION_NAME = "saturn_pub_segmented_reference"


class SharedCache(Cache):
    def __init__(
        self,
        keys: tuple[torch.Tensor, ...],
        values: tuple[torch.Tensor, ...],
        *,
        capacity: int = 64,
    ):
        super().__init__(layers=[])
        if not keys or len(keys) != len(values) or capacity < 1:
            raise ValueError("matching ancestor layers and positive capacity required")
        self._keys = keys
        self._values = values
        self._seal = digest(describe((keys, values)))
        self.capacity = capacity
        self._delta_keys: list[torch.Tensor | None] = [None] * len(keys)
        self._delta_values: list[torch.Tensor | None] = [None] * len(keys)
        self._lengths = [0] * len(keys)

    def verify(self) -> None:
        if digest(describe((self._keys, self._values))) != self._seal:
            raise ValueError("shared ancestor was mutated")

    def fork(self) -> SharedCache:
        self.verify()
        child = SharedCache(self._keys, self._values, capacity=self.capacity)
        for index, length in enumerate(self._lengths):
            if not length:
                continue
            child._delta_keys[index] = torch.empty_like(self._delta_keys[index])
            child._delta_values[index] = torch.empty_like(self._delta_values[index])
            child._delta_keys[index][:, :, :length].copy_(self._delta_keys[index][:, :, :length])
            child._delta_values[index][:, :, :length].copy_(
                self._delta_values[index][:, :, :length]
            )
            child._lengths[index] = length
        return child

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *_: Any,
        **__: Any,
    ) -> tuple[Segments, Segments]:
        if not 0 <= layer_idx < len(self._keys):
            raise ValueError("layer index out of range")
        expected = (*self._keys[layer_idx].shape[:2], 1, self._keys[layer_idx].shape[-1])
        if tuple(key_states.shape) != expected or value_states.shape != key_states.shape:
            raise ValueError("one K/V row per layer required")
        if key_states.dtype != self._keys[layer_idx].dtype:
            raise ValueError("shared cache dtype mismatch")
        length = self._lengths[layer_idx]
        if length >= self.capacity:
            raise ValueError("private delta capacity exceeded")
        if self._delta_keys[layer_idx] is None:
            key = self._keys[layer_idx]
            self._delta_keys[layer_idx] = torch.empty(
                (*key.shape[:2], self.capacity, key.shape[-1]), device=key.device, dtype=key.dtype
            )
            self._delta_values[layer_idx] = torch.empty_like(self._delta_keys[layer_idx])
        self._delta_keys[layer_idx][:, :, length : length + 1].copy_(key_states)
        self._delta_values[layer_idx][:, :, length : length + 1].copy_(value_states)
        self._lengths[layer_idx] += 1
        return (
            Segments(self._keys[layer_idx], self._delta_keys[layer_idx][:, :, : length + 1]),
            Segments(self._values[layer_idx], self._delta_values[layer_idx][:, :, : length + 1]),
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._keys[layer_idx].shape[-2] + self._lengths[layer_idx]

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + query_length, 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self._keys)

    @property
    def is_linear(self) -> list[bool]:
        return [False] * len(self._keys)

    @property
    def ancestor_pointers(self) -> tuple[int, ...]:
        return tuple(x.data_ptr() for x in (*self._keys, *self._values))


class SharedQwenMemory:
    """Capture one native prefix; every branch consumes its own generated-token tail."""

    @torch.inference_mode()
    def __init__(self, adapter: Any, prefix: list[int], *, capacity: int = 64):
        if not prefix or len(prefix) + capacity > adapter.model.config.max_position_embeddings:
            raise ValueError("prefix and private capacity must fit the configured context")
        self.adapter = adapter
        self.prefix = tuple(prefix)
        native = adapter.model(torch.tensor([prefix], device=adapter.device), use_cache=True)
        self.root = SharedCache(
            tuple(layer.keys.detach().clone() for layer in native.past_key_values.layers),
            tuple(layer.values.detach().clone() for layer in native.past_key_values.layers),
            capacity=capacity,
        )
        AttentionInterface.register(_ATTENTION_NAME, segmented_attention)
        AttentionMaskInterface.register(_ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])

    @torch.inference_mode()
    def step(self, cache: SharedCache, token: int) -> torch.Tensor:
        if cache.ancestor_pointers != self.root.ancestor_pointers:
            raise ValueError("branch belongs to another memory root")
        cache.verify()
        if not 0 <= token < self.adapter.model.config.vocab_size:
            raise ValueError("token outside vocabulary")
        if len(set(cache._lengths)) != 1 or cache._lengths[0] >= cache.capacity:
            raise ValueError("incomplete or full private tail")
        model = self.adapter.model
        original = model.config._attn_implementation
        lengths = list(cache._lengths)
        try:
            model.config._attn_implementation = _ATTENTION_NAME
            return model(
                torch.tensor([[token]], device=self.adapter.device),
                past_key_values=cache,
                use_cache=True,
            ).logits[:, -1]
        except Exception:
            cache._lengths = lengths
            raise
        finally:
            model.config._attn_implementation = original

    def accounting(self, branches: list[SharedCache]) -> dict[str, Any]:
        if not branches or len({id(b) for b in branches}) != len(branches):
            raise ValueError("account unique branch objects")
        if any(b.ancestor_pointers != self.root.ancestor_pointers for b in branches):
            raise ValueError("all branches must share this ancestor")

        def size(tensors: Any) -> int:
            return sum(x.numel() * x.element_size() for x in tensors if x is not None)

        ancestor = size((*self.root._keys, *self.root._values))
        allocated = sum(size((*b._delta_keys, *b._delta_values)) for b in branches)
        row_bytes = ancestor // len(self.prefix)
        committed = row_bytes * sum(b._lengths[0] for b in branches)
        return {
            "accounted_branches": len(branches),
            "shared_ancestor_bytes": ancestor,
            "private_committed_bytes": committed,
            "private_allocated_bytes": allocated,
            "physical_allocated_bytes": ancestor + allocated,
            "flat_committed_bytes": len(branches) * ancestor + committed,
            "saved_committed_bytes": (len(branches) - 1) * ancestor,
            "attention_rows": [b.get_seq_length() for b in branches],
            "ancestor_shared": True,
            "reduced_attention_work_claim": False,
            "temporary_attention_copies": "head-repeat and float32 casts; no full KV join",
        }

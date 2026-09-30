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

from ..contracts import ExecutionPoint, SlotSpec, SurfaceManifest
from ..core import Adapter, Session
from ..values import clone, describe, digest


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

    @classmethod
    def from_segments(
        cls,
        keys: tuple[torch.Tensor, ...],
        values: tuple[torch.Tensor, ...],
        delta_keys: tuple[torch.Tensor, ...],
        delta_values: tuple[torch.Tensor, ...],
        *,
        capacity: int,
    ) -> SharedCache:
        if len(delta_keys) != len(keys) or len(delta_values) != len(keys):
            raise ValueError("private delta must cover every ancestor layer")
        cache = cls(keys, values, capacity=capacity)
        lengths = {value.shape[-2] for value in (*delta_keys, *delta_values)}
        if len(lengths) != 1:
            raise ValueError("private K/V deltas must have one complete causal length")
        length = lengths.pop()
        if length > capacity:
            raise ValueError("private delta exceeds declared capacity")
        if length:
            for index, (key, value) in enumerate(zip(delta_keys, delta_values)):
                cache._delta_keys[index] = torch.empty(
                    (*key.shape[:2], capacity, key.shape[-1]),
                    device=key.device,
                    dtype=key.dtype,
                )
                cache._delta_values[index] = torch.empty_like(cache._delta_keys[index])
                cache._delta_keys[index][:, :, :length].copy_(key)
                cache._delta_values[index][:, :, :length].copy_(value)
                cache._lengths[index] = length
        return cache

    def segments(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        if len(set(self._lengths)) != 1:
            raise ValueError("cannot persist an incomplete layer update")
        self.verify()
        keys, values = [], []
        for index, length in enumerate(self._lengths):
            ancestor = self._keys[index]
            if not length:
                shape = (*ancestor.shape[:2], 0, ancestor.shape[-1])
                keys.append(torch.empty(shape, device=ancestor.device, dtype=ancestor.dtype))
                values.append(torch.empty(shape, device=ancestor.device, dtype=ancestor.dtype))
            else:
                keys.append(self._delta_keys[index][:, :, :length].detach().clone())
                values.append(self._delta_values[index][:, :, :length].detach().clone())
        return tuple(keys), tuple(values)

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


class SharedQwenAdapter(Adapter):
    """Durable exact-history cuts for the bounded segmented-attention reference ABI.

    The ancestor and private delta are both authoritative payload. In-process `SharedCache`
    forks pointer-share the ancestor; durable Session cuts clone it so they remain standalone.
    This adapter therefore claims exact history and continuation, not zero-copy durable forks.
    """

    def __init__(self, adapter: Any, *, capacity: int = 64):
        if capacity < 1:
            raise ValueError("private capacity must be positive")
        if adapter.model.config._attn_implementation != "eager":
            raise ValueError("shared history requires an eager Qwen2 source adapter")
        self.source = adapter
        self.model = adapter.model
        self.device = adapter.device
        self.dtype = adapter.dtype
        self.layers = adapter.layers
        self.capacity = capacity
        self._declared_capacity = capacity
        self.model_identity = digest(
            {
                "source_model_identity": adapter.model_identity,
                "shared_history": {"capacity": capacity, "attention": _ATTENTION_NAME},
            }
        )
        self._frozen_versions = tuple(
            (name, value._version)
            for name, value in (*self.model.named_parameters(), *self.model.named_buffers())
        )
        self.execution = {
            "family": "qwen2-shared-exact-history",
            "adapter": "segmented-history-v1",
            "granularity": "token",
            "environment_versions": dict(adapter.execution["environment_versions"]),
            "model_program": "native_prefill->segmented_global_softmax->native_suffix->greedy",
            "kernel": {
                "attention": _ATTENTION_NAME,
                "accumulation": "float32-segmented-online-softmax",
            },
            "batch": {"size": 1, "query_tokens": 1, "padding": False},
            "device": {"type": self.device.type, "index": self.device.index},
            "dtype": str(self.dtype),
            "numeric_scope": {
                "ancestor_positions": "native-post-rope-immutable",
                "phase_relocation": False,
                "private_capacity": capacity,
                "temporary_attention_copies": "head-repeat-and-float32-casts",
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "torch_num_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "tensor_layout": "state-slot-bound-stride",
            },
            "resource_scope": {
                "durable_storage": "content-addressed-ancestor-pages-dedup-eligible",
                "live_session_ram": "dense-clone-not-shared",
                "attention_compute": "all-ancestor-and-private-rows",
            },
            "sampler": "greedy",
            "parity": "bounded-logits-separate-numerical-abi",
        }
        AttentionInterface.register(_ATTENTION_NAME, segmented_attention)
        AttentionMaskInterface.register(_ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["eager"])

    def validate_execution(self) -> None:
        self.source.validate_execution()
        if self.capacity != self._declared_capacity:
            raise ValueError("shared-history configuration changed after identity freeze; rewrap")
        current = tuple(
            (name, value._version)
            for name, value in (*self.model.named_parameters(), *self.model.named_buffers())
        )
        if current != self._frozen_versions:
            raise ValueError(
                "model parameters or buffers changed after adapter identity was frozen; "
                "rewrap the model in a new adapter"
            )
        expected = self.execution["numeric_scope"]
        live = {
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if any(expected[name] != value for name, value in live.items()):
            raise ValueError("live numerical environment drifted from the execution ABI; rewrap")

    @torch.inference_mode()
    def session(self, prefix: list[int], pending_token: int) -> Session:
        self.validate_execution()
        if not prefix:
            raise ValueError("a nonempty native ancestor is required")
        if len(prefix) + self.capacity > self.model.config.max_position_embeddings:
            raise ValueError("ancestor and private capacity exceed the configured context")
        tokens = torch.tensor([prefix], dtype=torch.long, device=self.device)
        if bool((tokens < 0).any()) or bool((tokens >= self.model.config.vocab_size).any()):
            raise ValueError("ancestor token outside vocabulary")
        if not 0 <= pending_token < self.model.config.vocab_size:
            raise ValueError("pending token outside vocabulary")
        native = self.model(tokens, use_cache=True)
        ancestor_keys = tuple(
            layer.keys.detach().clone() for layer in native.past_key_values.layers
        )
        ancestor_values = tuple(
            layer.values.detach().clone() for layer in native.past_key_values.layers
        )
        shape = (*ancestor_keys[0].shape[:2], 0, ancestor_keys[0].shape[-1])
        empty = tuple(
            torch.empty(shape, device=self.device, dtype=self.dtype) for _ in range(self.layers)
        )
        all_tokens = torch.tensor([prefix + [pending_token]], dtype=torch.long, device=self.device)
        return Session(
            self,
            {
                "tokens": all_tokens,
                "ancestor_keys": ancestor_keys,
                "ancestor_values": ancestor_values,
                "delta_keys": empty,
                "delta_values": tuple(value.clone() for value in empty),
                "ancestor_length": len(prefix),
                "capacity": self.capacity,
                "logits": None,
            },
        )

    def boundary(self, state):
        return f"shared-token:{state['tokens'].shape[1] - 1}/before:decode"

    def point(self, state):
        return ExecutionPoint(
            family="qwen2-shared-exact-history",
            logical_step=state["tokens"].shape[1] - 1,
            phase="decode",
            operator="segmented_attention_native_suffix",
            edge="before",
            next_operation="decode_and_greedy_commit",
            local_clock=state["delta_keys"][0].shape[-2],
        )

    def surface(self, state):
        def sequence_schema(values):
            return {
                "kind": "tensor_sequence",
                "length": self.layers,
                "shapes": [list(value.shape) for value in values],
                "strides": [list(value.stride()) for value in values],
                "dtype": str(self.dtype),
                "device": str(self.device),
            }

        return SurfaceManifest(
            slots=(
                SlotSpec(
                    "tokens",
                    role="cursor",
                    producer="greedy_commit",
                    consumer="decode",
                    schema={
                        "kind": "tensor",
                        "shape": list(state["tokens"].shape),
                        "stride": list(state["tokens"].stride()),
                        "dtype": "torch.int64",
                        "device": str(self.device),
                    },
                ),
                SlotSpec(
                    "ancestor_keys",
                    role="immutable-cache",
                    producer="native_prefill",
                    consumer="segmented_attention",
                    schema=sequence_schema(state["ancestor_keys"]),
                ),
                SlotSpec(
                    "ancestor_values",
                    role="immutable-cache",
                    producer="native_prefill",
                    consumer="segmented_attention",
                    schema=sequence_schema(state["ancestor_values"]),
                ),
                SlotSpec(
                    "delta_keys",
                    role="private-cache",
                    producer="decode",
                    consumer="segmented_attention",
                    schema=sequence_schema(state["delta_keys"]),
                ),
                SlotSpec(
                    "delta_values",
                    role="private-cache",
                    producer="decode",
                    consumer="segmented_attention",
                    schema=sequence_schema(state["delta_values"]),
                ),
                SlotSpec(
                    "ancestor_length",
                    role="clock",
                    producer="native_prefill",
                    consumer="runtime",
                    schema={"kind": "integer", "minimum": 1},
                ),
                SlotSpec(
                    "capacity",
                    role="limit",
                    producer="adapter",
                    consumer="runtime",
                    schema={"kind": "integer", "minimum": 1},
                ),
                SlotSpec(
                    "logits",
                    role="observation",
                    producer="native_readout",
                    consumer="greedy",
                    persistence="evidence",
                    schema={"kind": "tensor", "optional": state["logits"] is None},
                ),
            ),
            state_schema="saturn-pub-qwen2-shared-history-v1",
            consumer="native-qwen2-segmented-suffix",
            horizon="bounded-native-autoregressive-suffix",
        )

    def addresses(self, state):
        return ("tokens",) + (("logits",) if state["logits"] is not None else ())

    def write(self, state, writes, invalidates):
        raise ValueError("shared exact-history cuts are read-only; fork from a different ancestor")

    def accounting(self, state):
        self.validate(state)

        def size(values):
            return sum(value.numel() * value.element_size() for value in values)

        ancestor = size((*state["ancestor_keys"], *state["ancestor_values"]))
        private = size((*state["delta_keys"], *state["delta_values"]))
        rows = state["ancestor_length"] + state["delta_keys"][0].shape[-2]
        return {
            "ancestor_payload_bytes": ancestor,
            "private_payload_bytes": private,
            "logical_payload_bytes": ancestor + private,
            "attention_rows": rows,
            "durable_ancestor_dedup_eligible": True,
            "live_session_ancestor_shared": False,
            "reduced_attention_work_claim": False,
            "temporary_attention_copies": "head-repeat and float32 casts; no full KV join",
        }

    def validate(self, state):
        self.validate_execution()
        required = {
            "tokens",
            "ancestor_keys",
            "ancestor_values",
            "delta_keys",
            "delta_values",
            "ancestor_length",
            "capacity",
            "logits",
        }
        if set(state) != required:
            raise ValueError("shared-history state schema keys differ from the contract")
        tokens = state["tokens"]
        if (
            tokens.ndim != 2
            or tuple(tokens.shape[:1]) != (1,)
            or tokens.dtype != torch.long
            or tokens.device != self.device
        ):
            raise ValueError("invalid shared-history token geometry/dtype/device")
        if bool((tokens < 0).any()) or bool((tokens >= self.model.config.vocab_size).any()):
            raise ValueError("shared-history token outside vocabulary")
        if type(state["ancestor_length"]) is not int or state["ancestor_length"] < 1:
            raise ValueError("invalid ancestor length")
        if state["capacity"] != self.capacity:
            raise ValueError("shared-history capacity differs from the execution contract")
        groups = (
            state["ancestor_keys"],
            state["ancestor_values"],
            state["delta_keys"],
            state["delta_values"],
        )
        if any(len(group) != self.layers for group in groups):
            raise ValueError("shared-history cache must cover every decoder layer")
        delta_lengths = set()
        for index in range(self.layers):
            key, value = state["ancestor_keys"][index], state["ancestor_values"][index]
            delta_key, delta_value = state["delta_keys"][index], state["delta_values"][index]
            base = (1, self.model.config.num_key_value_heads)
            width = self.model.config.hidden_size // self.model.config.num_attention_heads
            if (
                tuple(key.shape) != (*base, state["ancestor_length"], width)
                or value.shape != key.shape
            ):
                raise ValueError("invalid shared ancestor K/V geometry")
            if tuple(delta_key.shape[:2]) != base or delta_key.shape[-1] != width:
                raise ValueError("invalid private delta geometry")
            if delta_value.shape != delta_key.shape:
                raise ValueError("private K/V geometry mismatch")
            for tensor in (key, value, delta_key, delta_value):
                if tensor.dtype != self.dtype or tensor.device != self.device:
                    raise ValueError("shared-history cache dtype/device mismatch")
                if not torch.isfinite(tensor).all():
                    raise ValueError("non-finite shared-history cache")
            delta_lengths.add(delta_key.shape[-2])
        if len(delta_lengths) != 1:
            raise ValueError("private cache layers have different clocks")
        delta_length = delta_lengths.pop()
        if delta_length > self.capacity:
            raise ValueError("private delta exceeds capacity")
        if tokens.shape[1] != state["ancestor_length"] + delta_length + 1:
            raise ValueError("token and cache clocks disagree")
        logits = state["logits"]
        if logits is not None and (
            tuple(logits.shape) != (1, self.model.config.vocab_size)
            or logits.dtype != self.dtype
            or logits.device != self.device
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("invalid shared-history logits")

    @torch.inference_mode()
    def advance(self, state):
        self.validate(state)
        if state["delta_keys"][0].shape[-2] >= self.capacity:
            raise ValueError("private delta capacity reached")
        cache = SharedCache.from_segments(
            tuple(state["ancestor_keys"]),
            tuple(state["ancestor_values"]),
            tuple(state["delta_keys"]),
            tuple(state["delta_values"]),
            capacity=self.capacity,
        )
        original = self.model.config._attn_implementation
        try:
            self.model.config._attn_implementation = _ATTENTION_NAME
            logits = self.model(
                state["tokens"][:, -1:], past_key_values=cache, use_cache=True
            ).logits[:, -1]
        finally:
            self.model.config._attn_implementation = original
        delta_keys, delta_values = cache.segments()
        out = clone(state)
        out["delta_keys"], out["delta_values"] = delta_keys, delta_values
        out["logits"] = logits
        out["tokens"] = torch.cat((state["tokens"], logits.argmax(-1, keepdim=True)), dim=-1)
        return out

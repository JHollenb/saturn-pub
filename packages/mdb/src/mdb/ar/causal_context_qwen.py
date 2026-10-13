"""Exact-history Qwen2/Qwen3 continuation over verified radix K/V segments.

The durable radix store owns byte identity and ancestry. This runtime places one
shared ancestor and private deltas on device, then executes the Qwen2 decoder
one token at a time. No serving attention call builds a flat ancestor+delta K/V.
Its numerical contract is the direct segmented attention kernel, not HF SDPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch

from mdb.ar.qwen_shared_ancestor_attention import (
    execute_dense_shared_ancestor_attention,
    qwen_cache_layer_tensors,
)
from mdb.causal_radix_store import CausalRadixStore


class CausalContextQwenError(ValueError):
    """A Qwen context mount or continuation violates its declared ABI."""


LayerKV = tuple[torch.Tensor, torch.Tensor]


@dataclass(slots=True)
class QwenTokenExecution:
    """Provisional token state; its owner alone publishes visible history."""

    state: CausalContextQwenMount
    token_id: int
    hidden: torch.Tensor
    next_layer: int = 0
    logits: torch.Tensor | None = None
    sampled_token: int | None = None


@dataclass(slots=True)
class CausalContextQwenMount:
    mount_sha256: str
    page_id: str
    model_abi_sha256: str
    node_sha256s: tuple[str, ...]
    ancestor: tuple[LayerKV, ...]
    delta: tuple[LayerKV, ...]
    ancestor_rows: int
    delta_rows: int
    delta_capacity: int

    @property
    def length(self) -> int:
        return self.ancestor_rows + self.delta_rows

    @property
    def physical_kv_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for pair in self.delta for t in pair)

    def fork(self) -> CausalContextQwenMount:
        """Copy the private tail only; an aborted fork cannot mutate its parent."""
        return CausalContextQwenMount(
            self.mount_sha256,
            self.page_id,
            self.model_abi_sha256,
            self.node_sha256s,
            self.ancestor,
            tuple((key.clone(), value.clone()) for key, value in self.delta),
            self.ancestor_rows,
            self.delta_rows,
            self.delta_capacity,
        )

    def view(self) -> CausalContextQwenMount:
        """Create a length-isolated view for a linear, exclusive append."""
        return CausalContextQwenMount(
            self.mount_sha256, self.page_id, self.model_abi_sha256,
            self.node_sha256s, self.ancestor, self.delta,
            self.ancestor_rows, self.delta_rows, self.delta_capacity,
        )


class QwenKVRowCodec:
    """Serialize each token's all-layer K/V bits as one deterministic radix row."""

    def __init__(self, model: Any) -> None:
        config = model.config
        self.model_type = str(getattr(config, "model_type", ""))
        if self.model_type not in {"qwen2", "qwen3"}:
            raise CausalContextQwenError("only Qwen2/Qwen3 exact-history K/V is supported")
        self.layers = int(config.num_hidden_layers)
        self.groups = int(config.num_key_value_heads)
        configured_head_dim = getattr(config, "head_dim", None)
        if configured_head_dim is None:
            hidden_size = int(config.hidden_size)
            query_heads = int(config.num_attention_heads)
            if query_heads <= 0 or hidden_size % query_heads:
                raise CausalContextQwenError("Qwen hidden size is not divisible by query heads")
            configured_head_dim = hidden_size // query_heads
        self.head_dim = int(configured_head_dim)
        if self.head_dim != int(model.model.layers[0].self_attn.head_dim):
            raise CausalContextQwenError("Qwen config and attention head widths disagree")
        self.dtype = next(model.parameters()).dtype
        if self.dtype != torch.bfloat16:
            raise CausalContextQwenError("the current direct-read ABI requires BF16")
        self.row_bytes = 2 * self.layers * self.groups * self.head_dim * 2

    def encode_cache_rows(self, cache: Any) -> tuple[bytes, ...]:
        owned: list[LayerKV] = []
        for layer in range(self.layers):
            key, value = qwen_cache_layer_tensors(cache, layer)
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise CausalContextQwenError("cache dtype changed")
            if key.shape[0] != 1 or tuple(key.shape[1::2]) != (self.groups, self.head_dim):
                raise CausalContextQwenError("cache geometry changed")
            owned.append((key.detach().to("cpu"), value.detach().to("cpu")))
        tokens = int(owned[0][0].shape[2])
        if any(int(t.shape[2]) != tokens for pair in owned for t in pair):
            raise CausalContextQwenError("cache layer lengths disagree")
        rows = []
        for token in range(tokens):
            chunks = []
            for key, value in owned:
                for tensor in (key, value):
                    chunks.append(tensor[0, :, token, :].contiguous().view(torch.uint8).numpy().tobytes())
            row = b"".join(chunks)
            if len(row) != self.row_bytes:
                raise CausalContextQwenError("serialized row byte count changed")
            rows.append(row)
        return tuple(rows)

    def decode_segment(self, raw: bytes, device: torch.device) -> tuple[LayerKV, ...]:
        if not raw or len(raw) % self.row_bytes:
            raise CausalContextQwenError("segment is empty or has partial K/V rows")
        tokens = len(raw) // self.row_bytes
        # The mutable bytearray owns this CPU staging buffer. It is never a
        # concatenated logical page: each call decodes one physical segment.
        flat = torch.frombuffer(bytearray(raw), dtype=self.dtype)
        shaped = flat.reshape(tokens, self.layers, 2, self.groups, self.head_dim)
        return tuple(
            (
                shaped[:, layer, 0].permute(1, 0, 2).unsqueeze(0).contiguous().to(device),
                shaped[:, layer, 1].permute(1, 0, 2).unsqueeze(0).contiguous().to(device),
            )
            for layer in range(self.layers)
        )

    def encode_layers(self, layers: tuple[LayerKV, ...], rows: int) -> bytes:
        """Persist only visible rows, with the same all-layer row byte ABI."""
        if len(layers) != self.layers or rows < 0:
            raise CausalContextQwenError("checkpoint layer count or visible length changed")
        if rows == 0:
            return b""
        owned = []
        for pair in layers:
            if len(pair) != 2:
                raise CausalContextQwenError("checkpoint needs paired K/V")
            for tensor in pair:
                if (tensor.dtype != self.dtype or tensor.ndim != 4
                        or tensor.shape[0] != 1 or tensor.shape[1] != self.groups
                        or tensor.shape[3] != self.head_dim or tensor.shape[2] < rows):
                    raise CausalContextQwenError("checkpoint K/V geometry changed")
            owned.append(tuple(t[:, :, :rows, :].detach().cpu() for t in pair))
        return b"".join(
            tensor[0, :, row, :].contiguous().view(torch.uint8).numpy().tobytes()
            for row in range(rows) for pair in owned for tensor in pair
        )


class CausalContextQwenRuntime:
    """A resident exact-history consumer for one model and one radix store."""

    def __init__(
        self,
        model: Any,
        store: CausalRadixStore,
        *,
        model_abi_sha256: str,
        backend: str = "triton",
    ) -> None:
        if len(model_abi_sha256) != 64:
            raise CausalContextQwenError("model ABI must be a SHA-256 digest")
        self.model = model
        self.model.eval()
        self.store = store
        self.codec = QwenKVRowCodec(model)
        self.model_abi_sha256 = model_abi_sha256
        self.backend = backend
        if self.codec.model_type == "qwen3" and backend != "native-eager":
            raise CausalContextQwenError(
                "Qwen3 exact-history is qualified only for native-eager attention"
            )
        self.device = next(model.parameters()).device
        if getattr(model.config, "use_sliding_window", False):
            raise CausalContextQwenError("sliding-window Qwen needs a separate mask ABI")
        self._ancestors: dict[str, tuple[LayerKV, ...]] = {}
        self._parent_index = torch.zeros(1, dtype=torch.int32, device=self.device)

    def evict_unused_ancestors(self, retained: set[str]) -> None:
        """Release cache ownership after the session manager drops its mounts."""
        for key in tuple(self._ancestors):
            if key not in retained:
                del self._ancestors[key]

    def has_resident_ancestor(self, node_sha256: str) -> bool:
        return node_sha256 in self._ancestors

    def mount(self, mount_sha256: str, page_id: str, *, extra_rows: int) -> CausalContextQwenMount:
        if extra_rows <= 0:
            raise CausalContextQwenError("mount requires positive private-tail capacity")
        read = self.store.read(
            mount_sha256, page_id, model_abi_sha256=self.model_abi_sha256
        )
        if len(read.segments) not in {1, 2}:
            raise CausalContextQwenError("current consumer requires an ancestor and optional delta")
        ancestor_id = read.node_sha256s[0]
        ancestor = self._ancestors.get(ancestor_id)
        if ancestor is None:
            ancestor = self.codec.decode_segment(read.segments[0], self.device)
            self._ancestors[ancestor_id] = ancestor
        ancestor_rows = int(ancestor[0][0].shape[2])
        initial_delta = (self.codec.decode_segment(read.segments[1], self.device)
                         if len(read.segments) == 2 else None)
        delta_rows = int(initial_delta[0][0].shape[2]) if initial_delta else 0
        capacity = delta_rows + extra_rows
        delta = []
        for index, (key, value) in enumerate(ancestor):
            placed_key = torch.empty(
                (1, self.codec.groups, capacity, self.codec.head_dim),
                dtype=self.codec.dtype, device=self.device,
            )
            placed_value = torch.empty_like(placed_key)
            if initial_delta:
                placed_key[:, :, :delta_rows, :].copy_(initial_delta[index][0])
                placed_value[:, :, :delta_rows, :].copy_(initial_delta[index][1])
            delta.append((placed_key, placed_value))
        return CausalContextQwenMount(
            mount_sha256, page_id, self.model_abi_sha256,
            read.node_sha256s, ancestor, tuple(delta),
            ancestor_rows, delta_rows, capacity,
        )

    @torch.inference_mode()
    def step(
        self,
        state: CausalContextQwenMount,
        token_id: int,
        *,
        monitor: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Consume one token through all qualified Qwen layers and the frozen head."""
        if state.model_abi_sha256 != self.model_abi_sha256:
            raise CausalContextQwenError("mounted model ABI changed")
        if state.delta_rows >= state.delta_capacity:
            raise CausalContextQwenError("private-tail capacity exhausted")
        if token_id < 0 or token_id >= int(self.model.config.vocab_size):
            raise CausalContextQwenError("token is outside the model vocabulary")
        execution = self.begin_token(state, token_id)
        trace: dict[str, Any] = {"input_position": state.length, "layers": []}
        while execution.next_layer < self.codec.layers:
            index = execution.next_layer
            attention_output = self.advance_token(execution)
            if monitor:
                trace["layers"].append({"layer": index,
                    "attention_output": attention_output.detach().clone(),
                    "residual": execution.hidden.detach().clone()})
        logits = self.finish_token(execution)
        state.delta_rows += 1
        if monitor:
            trace["final_hidden"] = self.model.model.norm(execution.hidden).detach().clone()
        return logits, trace

    @torch.inference_mode()
    def begin_token(self, state: CausalContextQwenMount, token_id: int) -> QwenTokenExecution:
        if state.model_abi_sha256 != self.model_abi_sha256 or state.delta_rows >= state.delta_capacity:
            raise CausalContextQwenError("incompatible mount or exhausted private capacity")
        if isinstance(token_id, bool) or not 0 <= token_id < self.model.config.vocab_size:
            raise CausalContextQwenError("token is outside the model vocabulary")
        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        return QwenTokenExecution(state, token_id, self.model.model.embed_tokens(token))

    @torch.inference_mode()
    def advance_token(self, execution: QwenTokenExecution) -> torch.Tensor:
        if self.codec.model_type == "qwen2":
            from transformers.models.qwen2 import modeling_qwen2 as modeling
        else:
            from transformers.models.qwen3 import modeling_qwen3 as modeling
        state, hidden, index = execution.state, execution.hidden, execution.next_layer
        if execution.logits is not None or index >= self.codec.layers:
            raise CausalContextQwenError("token has no remaining decoder layer")
        position_ids = torch.tensor([[state.length]], dtype=torch.long, device=self.device)
        rope = self.model.model.rotary_emb(hidden, position_ids)
        layer = self.model.model.layers[index]
        residual = hidden
        normalized = layer.input_layernorm(hidden)
        attention = layer.self_attn
        shape = (1, 1, -1, attention.head_dim)
        query = attention.q_proj(normalized).view(shape)
        key = attention.k_proj(normalized).view(shape)
        if self.codec.model_type == "qwen3":
            query = attention.q_norm(query)
            key = attention.k_norm(key)
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2)
        value = attention.v_proj(normalized).view(shape).transpose(1, 2)
        query, key = modeling.apply_rotary_pos_emb(query, key, *rope)
        query = query.contiguous()
        delta_key, delta_value = state.delta[index]
        delta_key[:, :, state.delta_rows : state.delta_rows + 1, :].copy_(key)
        delta_value[:, :, state.delta_rows : state.delta_rows + 1, :].copy_(value)
        ancestor_key, ancestor_value = state.ancestor[index]
        if self.backend == "native-eager":
            keys = torch.cat((ancestor_key, delta_key[:, :, :state.delta_rows + 1]), dim=2)
            values = torch.cat((ancestor_value, delta_value[:, :, :state.delta_rows + 1]), dim=2)
            attended, _ = modeling.eager_attention_forward(attention, query, keys, values,
                attention_mask=None, scaling=attention.scaling, dropout=0.0)
            attended = attended.transpose(1, 2)
        else:
            attended = execute_dense_shared_ancestor_attention(
                query, ancestor_key, ancestor_value,
                delta_key, delta_value, self._parent_index,
                scaling=attention.scaling, backend=self.backend,
                check_finite=False, delta_length=state.delta_rows + 1,
            )
        attention_output = attention.o_proj(
            attended.transpose(1, 2).reshape(1, 1, -1).contiguous()
        )
        hidden = residual + attention_output
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        execution.hidden = hidden
        execution.next_layer += 1
        return attention_output

    @torch.inference_mode()
    def finish_token(self, execution: QwenTokenExecution) -> torch.Tensor:
        if execution.next_layer != self.codec.layers or execution.logits is not None:
            raise CausalContextQwenError("readout requires exactly one complete token")
        logits = self.model.lm_head(self.model.model.norm(execution.hidden))[:, -1, :]
        if not bool(torch.isfinite(logits).all()):
            raise CausalContextQwenError("non-finite readout cannot commit a token")
        execution.logits = logits
        return execution.logits

    def physical_bytes(self, states: tuple[CausalContextQwenMount, ...]) -> dict[str, int]:
        ancestors = {state.node_sha256s[0]: state.ancestor for state in states}
        ancestor_bytes = sum(
            tensor.numel() * tensor.element_size()
            for layers in ancestors.values() for pair in layers for tensor in pair
        )
        delta_bytes = sum(state.physical_kv_bytes for state in states)
        return {"ancestor_bytes": ancestor_bytes, "delta_capacity_bytes": delta_bytes,
                "total_kv_bytes": ancestor_bytes + delta_bytes}


def cache_prefix_digest(cache: Any, layers: int, prefix_rows: int) -> str:
    """Hash exact prefix K/V bytes without constructing a flat K/V tensor."""
    digest = sha256()
    for layer in range(layers):
        for tensor in qwen_cache_layer_tensors(cache, layer):
            raw = tensor[:, :, :prefix_rows, :].contiguous().view(torch.uint8).cpu().numpy()
            digest.update(raw.tobytes())
    return digest.hexdigest()

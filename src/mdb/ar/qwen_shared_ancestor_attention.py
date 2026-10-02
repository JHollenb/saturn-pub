"""Dense attention over one shared ancestor plus row-local child deltas.

The Triton path maps each logical page position to either the shared ancestor
or the selected child delta inside the kernel.  It never materializes a full
``[ancestor; delta]`` K/V page.  The caller remains responsible for exact
typed routing, model/RoPE ABI checks, PhaseLock query preparation, causal
eligibility, and merging this page with any live tail.

This is an attention-boundary primitive, not semantic authority.  The Torch
path deliberately materializes the page and serves as the numerical reference.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from mdb.ar.qwen_joint_value_output_fused import (
    _next_power_of_two,
    _triton_modules,
)


class QwenSharedAncestorAttentionError(ValueError):
    """Shared-ancestor attention inputs or backend are invalid."""


def qwen_cache_layer_tensors(cache: Any, index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one K/V layer across modern, transitional, and legacy HF cache ABIs."""

    if index < 0:
        raise QwenSharedAncestorAttentionError("cache layer index must be nonnegative")
    layers = getattr(cache, "layers", None)
    if layers is not None:
        if index >= len(layers):
            raise QwenSharedAncestorAttentionError("cache layer is absent")
        layer = layers[index]
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
    else:
        key_cache = getattr(cache, "key_cache", None)
        value_cache = getattr(cache, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            if index >= len(key_cache) or index >= len(value_cache):
                raise QwenSharedAncestorAttentionError("cache layer is absent")
            key, value = key_cache[index], value_cache[index]
        else:
            try:
                entry = tuple(cache)[index]
            except (IndexError, TypeError) as error:
                raise QwenSharedAncestorAttentionError("unsupported cache ABI") from error
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                raise QwenSharedAncestorAttentionError("unsupported cache layer ABI")
            key, value = entry[0], entry[1]
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise QwenSharedAncestorAttentionError("cache layer K/V tensors are absent")
    if key.ndim != 4 or value.shape != key.shape:
        raise QwenSharedAncestorAttentionError("cache layer K/V geometry changed")
    return key, value


def shared_ancestor_backend_available(device: torch.device | str) -> bool:
    resolved = torch.device(device)
    return resolved.type == "cuda" and torch.cuda.is_available() and _triton_modules() is not None


def shared_ancestor_backend_name(
    device: torch.device | str,
    *,
    requested: str = "auto",
) -> str:
    normalized = str(requested).lower()
    if normalized not in {"auto", "triton", "torch"}:
        raise QwenSharedAncestorAttentionError("backend must be auto, triton, or torch")
    if normalized == "torch":
        return "torch-materialized-shared-ancestor-reference"
    if shared_ancestor_backend_available(device):
        return "triton-dense-shared-ancestor-zero-page-copy"
    if normalized == "triton":
        return "triton-unavailable"
    return "torch-materialized-shared-ancestor-reference-fallback"


def _validate_inputs(
    query: torch.Tensor,
    ancestor_key: torch.Tensor,
    ancestor_value: torch.Tensor,
    delta_key: torch.Tensor,
    delta_value: torch.Tensor,
    parent_index: torch.Tensor,
    *,
    check_finite: bool,
) -> tuple[int, int, int, int, int, int, int]:
    if query.ndim != 4:
        raise QwenSharedAncestorAttentionError(
            "query must be [rows, query_heads, query_tokens, head_dim]"
        )
    tensors = (ancestor_key, ancestor_value, delta_key, delta_value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise QwenSharedAncestorAttentionError("ancestor and delta K/V must be rank-four tensors")
    if parent_index.ndim != 1 or parent_index.dtype not in {torch.int32, torch.int64}:
        raise QwenSharedAncestorAttentionError(
            "parent_index must be a rank-one int32 or int64 tensor"
        )
    rows, heads, query_tokens, head_dim = (int(value) for value in query.shape)
    parents, groups, ancestor_tokens, ancestor_dim = (int(value) for value in ancestor_key.shape)
    delta_rows, delta_groups, delta_tokens, delta_dim = (int(value) for value in delta_key.shape)
    if (
        min(
            rows,
            heads,
            query_tokens,
            head_dim,
            parents,
            groups,
            ancestor_tokens,
        )
        <= 0
    ):
        raise QwenSharedAncestorAttentionError("shared-ancestor geometry must be positive")
    if heads % groups:
        raise QwenSharedAncestorAttentionError("query heads must be divisible by K/V heads")
    if tuple(ancestor_value.shape) != tuple(ancestor_key.shape):
        raise QwenSharedAncestorAttentionError("ancestor K/V geometry changed")
    if tuple(delta_value.shape) != tuple(delta_key.shape):
        raise QwenSharedAncestorAttentionError("delta K/V geometry changed")
    if (delta_rows, delta_groups) != (rows, groups):
        raise QwenSharedAncestorAttentionError("delta row or K/V-head geometry changed")
    if ancestor_dim != head_dim or delta_dim != head_dim:
        raise QwenSharedAncestorAttentionError("K/V head dimension changed")
    if int(parent_index.shape[0]) != rows:
        raise QwenSharedAncestorAttentionError("parent_index row count changed")
    all_tensors = (query, *tensors, parent_index)
    if len({tensor.device for tensor in all_tensors}) != 1:
        raise QwenSharedAncestorAttentionError("all tensors must share one device")
    if len({tensor.dtype for tensor in (query, *tensors)}) != 1:
        raise QwenSharedAncestorAttentionError("query and K/V dtypes must match")
    # Raw-pointer CUDA kernels must validate indices even when the optional
    # finite-value scan is disabled.
    if bool((parent_index < 0).any()) or bool((parent_index >= parents).any()):
        raise QwenSharedAncestorAttentionError("parent_index is out of range")
    if check_finite:
        if any(not torch.isfinite(tensor).all() for tensor in (query, *tensors)):
            raise QwenSharedAncestorAttentionError("attention tensors must be finite")
    return rows, heads, query_tokens, head_dim, parents, groups, ancestor_tokens


def _torch_reference(
    query: torch.Tensor,
    ancestor_key: torch.Tensor,
    ancestor_value: torch.Tensor,
    delta_key: torch.Tensor,
    delta_value: torch.Tensor,
    parent_index: torch.Tensor,
    *,
    scaling: float,
    delta_length: int,
) -> torch.Tensor:
    selected_key = ancestor_key.index_select(0, parent_index)
    selected_value = ancestor_value.index_select(0, parent_index)
    key = torch.cat((selected_key, delta_key[:, :, :delta_length, :]), dim=2)
    value = torch.cat((selected_value, delta_value[:, :, :delta_length, :]), dim=2)
    groups = int(key.shape[1])
    heads = int(query.shape[1])
    head_to_group = torch.arange(heads, device=query.device) // (heads // groups)
    expanded_key = key.index_select(1, head_to_group)
    expanded_value = value.index_select(1, head_to_group)
    scores = torch.einsum("rhtd,rhsd->rhts", query.float(), expanded_key.float())
    probabilities = torch.softmax(scores * float(scaling), dim=-1)
    return torch.einsum("rhts,rhsd->rhtd", probabilities, expanded_value.float()).to(
        dtype=query.dtype
    )


@lru_cache(maxsize=1)
def _build_kernel() -> tuple[Any, Any] | None:
    modules = _triton_modules()
    if modules is None:
        return None
    triton, tl = modules

    @triton.jit
    def dense_shared_ancestor_attention_kernel(
        query_ptr,
        ancestor_key_ptr,
        ancestor_value_ptr,
        delta_key_ptr,
        delta_value_ptr,
        parent_index_ptr,
        output_ptr,
        heads,
        groups,
        query_tokens,
        ancestor_tokens: tl.constexpr,
        delta_capacity: tl.constexpr,
        delta_length,
        head_dim: tl.constexpr,
        scaling,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        token = program_id % query_tokens
        head = (program_id // query_tokens) % heads
        row = program_id // (query_tokens * heads)
        group = head // (heads // groups)
        parent = tl.load(parent_index_ptr + row).to(tl.int32)
        dimension = tl.arange(0, BLOCK_D)
        dimension_mask = dimension < head_dim
        query_offset = ((row * heads + head) * query_tokens + token) * head_dim
        query_values = tl.load(
            query_ptr + query_offset + dimension,
            mask=dimension_mask,
            other=0.0,
        ).to(tl.float32)

        maximum = tl.full((), -float("inf"), dtype=tl.float32)
        total_capacity = ancestor_tokens + delta_capacity
        total_tokens = ancestor_tokens + delta_length
        for start in range(0, total_capacity, BLOCK_S):
            logical = start + tl.arange(0, BLOCK_S)
            valid = logical < total_tokens
            from_ancestor = logical < ancestor_tokens
            ancestor_offset = (
                (parent * groups + group) * ancestor_tokens + logical[:, None]
            ) * head_dim + dimension[None, :]
            delta_logical = tl.maximum(logical - ancestor_tokens, 0)
            delta_offset = (
                (row * groups + group) * delta_capacity + delta_logical[:, None]
            ) * head_dim + dimension[None, :]
            ancestor_values = tl.load(
                ancestor_key_ptr + ancestor_offset,
                mask=valid[:, None] & from_ancestor[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            delta_values = tl.load(
                delta_key_ptr + delta_offset,
                mask=valid[:, None] & (~from_ancestor[:, None]) & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            key_values = tl.where(from_ancestor[:, None], ancestor_values, delta_values)
            score = tl.sum(key_values * query_values[None, :], axis=1) * scaling
            score = tl.where(valid, score, -float("inf"))
            maximum = tl.maximum(maximum, tl.max(score, axis=0))

        denominator = tl.zeros((), dtype=tl.float32)
        numerator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, total_capacity, BLOCK_S):
            logical = start + tl.arange(0, BLOCK_S)
            valid = logical < total_tokens
            from_ancestor = logical < ancestor_tokens
            ancestor_offset = (
                (parent * groups + group) * ancestor_tokens + logical[:, None]
            ) * head_dim + dimension[None, :]
            delta_logical = tl.maximum(logical - ancestor_tokens, 0)
            delta_offset = (
                (row * groups + group) * delta_capacity + delta_logical[:, None]
            ) * head_dim + dimension[None, :]
            ancestor_keys = tl.load(
                ancestor_key_ptr + ancestor_offset,
                mask=valid[:, None] & from_ancestor[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            delta_keys = tl.load(
                delta_key_ptr + delta_offset,
                mask=valid[:, None] & (~from_ancestor[:, None]) & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            key_values = tl.where(from_ancestor[:, None], ancestor_keys, delta_keys)
            score = tl.sum(key_values * query_values[None, :], axis=1) * scaling
            probability = tl.exp(score - maximum)
            probability = tl.where(valid, probability, 0.0)
            ancestor_values = tl.load(
                ancestor_value_ptr + ancestor_offset,
                mask=valid[:, None] & from_ancestor[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            delta_values = tl.load(
                delta_value_ptr + delta_offset,
                mask=valid[:, None] & (~from_ancestor[:, None]) & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            value_values = tl.where(from_ancestor[:, None], ancestor_values, delta_values)
            denominator += tl.sum(probability, axis=0)
            numerator += tl.sum(probability[:, None] * value_values, axis=0)
        output_offset = ((row * heads + head) * query_tokens + token) * head_dim
        tl.store(
            output_ptr + output_offset + dimension,
            numerator / denominator,
            mask=dimension_mask,
        )

    return dense_shared_ancestor_attention_kernel, triton


def _triton_execute(
    query: torch.Tensor,
    ancestor_key: torch.Tensor,
    ancestor_value: torch.Tensor,
    delta_key: torch.Tensor,
    delta_value: torch.Tensor,
    parent_index: torch.Tensor,
    *,
    scaling: float,
    shape: tuple[int, int, int, int, int, int, int],
    delta_length: int,
) -> torch.Tensor:
    kernels = _build_kernel()
    if kernels is None:
        raise QwenSharedAncestorAttentionError("Triton is unavailable")
    kernel, _triton = kernels
    rows, heads, query_tokens, head_dim, _parents, groups, ancestor_tokens = shape
    delta_capacity = int(delta_key.shape[2])
    block_dimension = _next_power_of_two(head_dim)
    block_tokens = 32
    output = torch.empty_like(query)
    kernel[(rows * heads * query_tokens,)](
        query,
        ancestor_key,
        ancestor_value,
        delta_key,
        delta_value,
        parent_index,
        output,
        heads,
        groups,
        query_tokens,
        ancestor_tokens,
        delta_capacity,
        delta_length,
        head_dim,
        float(scaling),
        BLOCK_S=block_tokens,
        BLOCK_D=block_dimension,
        num_warps=4,
    )
    return output


@torch.inference_mode()
def execute_dense_shared_ancestor_attention(
    query: torch.Tensor,
    ancestor_key: torch.Tensor,
    ancestor_value: torch.Tensor,
    delta_key: torch.Tensor,
    delta_value: torch.Tensor,
    parent_index: torch.Tensor,
    *,
    scaling: float,
    backend: str = "auto",
    check_finite: bool = True,
    delta_length: int | None = None,
) -> torch.Tensor:
    """Attend to shared and child storage as one logical ordered page."""

    shape = _validate_inputs(
        query,
        ancestor_key,
        ancestor_value,
        delta_key,
        delta_value,
        parent_index,
        check_finite=check_finite,
    )
    normalized = str(backend).lower()
    if normalized not in {"auto", "triton", "torch"}:
        raise QwenSharedAncestorAttentionError("backend must be auto, triton, or torch")
    resolved_delta_length = int(delta_key.shape[2]) if delta_length is None else int(delta_length)
    if resolved_delta_length < 0 or resolved_delta_length > int(delta_key.shape[2]):
        raise QwenSharedAncestorAttentionError("delta_length exceeds placed delta capacity")
    if normalized == "torch":
        return _torch_reference(
            query,
            ancestor_key,
            ancestor_value,
            delta_key,
            delta_value,
            parent_index,
            scaling=scaling,
            delta_length=resolved_delta_length,
        )
    available = shared_ancestor_backend_available(query.device)
    if normalized == "triton" and not available:
        raise QwenSharedAncestorAttentionError("requested Triton backend is unavailable")
    if not available:
        return _torch_reference(
            query,
            ancestor_key,
            ancestor_value,
            delta_key,
            delta_value,
            parent_index,
            scaling=scaling,
            delta_length=resolved_delta_length,
        )
    # A copy here would silently invalidate the zero-K/V-materialization ABI.
    if not all(
        tensor.is_contiguous()
        for tensor in (
            query,
            ancestor_key,
            ancestor_value,
            delta_key,
            delta_value,
            parent_index,
        )
    ):
        raise QwenSharedAncestorAttentionError(
            "Triton ABI requires contiguous inputs; no implicit copy"
        )
    return _triton_execute(
        query,
        ancestor_key,
        ancestor_value,
        delta_key,
        delta_value,
        parent_index,
        scaling=scaling,
        shape=shape,
        delta_length=resolved_delta_length,
    )


@torch.inference_mode()
def execute_dense_contiguous_canonical_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scaling: float,
    backend: str = "triton",
) -> torch.Tensor:
    """Paired dense arm of the same two-pass arithmetic graph.

    K/V are supplied as one resident contiguous page. This is a canonical
    kernel comparison, not a claim of bit parity with a legacy native reader.
    """
    if key.ndim != 4 or key.shape[0] != 1 or value.shape != key.shape:
        raise QwenSharedAncestorAttentionError("canonical page must be [1, groups, tokens, dim]")
    empty = key.new_empty((query.shape[0], key.shape[1], 0, key.shape[3]))
    parents = torch.zeros(query.shape[0], dtype=torch.int32, device=query.device)
    return execute_dense_shared_ancestor_attention(
        query,
        key,
        value,
        empty,
        empty,
        parents,
        scaling=scaling,
        backend=backend,
    )


__all__ = [
    "QwenSharedAncestorAttentionError",
    "execute_dense_shared_ancestor_attention",
    "execute_dense_contiguous_canonical_attention",
    "qwen_cache_layer_tensors",
    "shared_ancestor_backend_available",
    "shared_ancestor_backend_name",
]

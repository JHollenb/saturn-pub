"""Optional Triton hot path for compact Qwen V -> attention -> O execution.

The ordinary compact implementation in :mod:`qwen_joint_value_output` is a
useful algebraic reference, but it launches several small operations and
materializes both the current compact value and the compact attention mix.
This module keeps the same compact representation and live query while
combining the current V read with the attention reduction in one Triton
kernel, followed by a compact O epilogue that writes the model-width result.

The prefix V cache must already be projected into compact value coordinates.
That compilation is deliberately outside the hot path.  The Triton backend is
optional; CPU callers and unsupported CUDA installations use the public
unfused Torch reference path and report that fallback explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import torch

from mdb.ar.qwen_joint_value_output import (
    QwenCompactJointValueOutputProgram,
    QwenJointValueOutputError,
)


@dataclass(frozen=True, slots=True)
class FusedCompactStep:
    """Attention write plus the compact value that must be appended to cache."""

    attention_write: torch.Tensor
    current_compact_value: torch.Tensor


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@lru_cache(maxsize=1)
def _triton_modules() -> tuple[Any, Any] | None:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 - Triton is an optional CUDA dependency
        return None
    return triton, tl


def fused_backend_available(device: torch.device | str) -> bool:
    """Return whether the optional Triton backend can be selected."""

    resolved = torch.device(device)
    return resolved.type == "cuda" and torch.cuda.is_available() and _triton_modules() is not None


def prepare_fused_compact_program(
    program: QwenCompactJointValueOutputProgram,
) -> QwenCompactJointValueOutputProgram:
    """Materialize the compiler artifact's kernel layout once, outside execution timing."""

    return replace(
        program,
        input_weight=program.input_weight.contiguous(),
        input_bias=program.input_bias.contiguous(),
        value_basis=program.value_basis.contiguous(),
        output_basis=program.output_basis.contiguous(),
        head_cores=program.head_cores.contiguous(),
        output_bias=program.output_bias.contiguous(),
    )


def _validate_inputs(
    program: QwenCompactJointValueOutputProgram,
    attention_norm: torch.Tensor,
    query: torch.Tensor,
    current_key: torch.Tensor,
    prefix_key: torch.Tensor,
    compact_prefix_value: torch.Tensor,
    *,
    check_finite: bool,
) -> tuple[int, int, int, int]:
    if attention_norm.ndim != 3 or query.ndim != 4:
        raise QwenJointValueOutputError(
            "fused compact path expects attention_norm [B,T,D] and query [B,H,T,Dh]"
        )
    if current_key.ndim != 4 or prefix_key.ndim != 4 or compact_prefix_value.ndim != 4:
        raise QwenJointValueOutputError(
            "fused compact path expects rank-four key and compact-prefix tensors"
        )
    batch, tokens, hidden = (int(value) for value in attention_norm.shape)
    heads = int(query.shape[1])
    if hidden != int(program.input_weight.shape[-1]):
        raise QwenJointValueOutputError("fused attention hidden width changed")
    if tuple(query.shape[:3]) != (batch, heads, tokens):
        raise QwenJointValueOutputError("fused query geometry changed")
    if int(query.shape[-1]) != program.head_dim:
        raise QwenJointValueOutputError("fused query head dimension changed")
    if tuple(current_key.shape[:3]) != (
        batch,
        program.key_value_heads,
        tokens,
    ) or int(current_key.shape[-1]) != program.head_dim:
        raise QwenJointValueOutputError("fused current-key geometry changed")
    if tuple(prefix_key.shape[:2]) != (batch, program.key_value_heads):
        raise QwenJointValueOutputError("fused prefix-key group geometry changed")
    if int(prefix_key.shape[-1]) != program.head_dim:
        raise QwenJointValueOutputError("fused prefix-key head dimension changed")
    prefix_length = int(prefix_key.shape[-2])
    if tuple(compact_prefix_value.shape) != (
        batch,
        program.key_value_heads,
        prefix_length,
        program.value_rank,
    ):
        raise QwenJointValueOutputError("fused compact-prefix geometry changed")
    tensors = (attention_norm, query, current_key, prefix_key, compact_prefix_value)
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise QwenJointValueOutputError("fused compact tensors must share a device")
    if check_finite and any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise QwenJointValueOutputError("fused compact inputs must be finite")
    program_tensors = (
        program.input_weight,
        program.input_bias,
        program.head_cores,
        program.output_basis,
        program.output_bias,
    )
    if any(tensor.device != attention_norm.device for tensor in program_tensors):
        raise QwenJointValueOutputError("fused program tensors must share the input device")
    if any(tensor.dtype != attention_norm.dtype for tensor in program_tensors):
        raise QwenJointValueOutputError("fused program and input dtypes must match")
    return batch, heads, tokens, prefix_length


def _torch_reference(
    program: QwenCompactJointValueOutputProgram,
    attention_norm: torch.Tensor,
    query: torch.Tensor,
    current_key: torch.Tensor,
    prefix_key: torch.Tensor,
    compact_prefix_value: torch.Tensor,
    *,
    scaling: float,
) -> torch.Tensor:
    return program.execute_core(
        attention_norm,
        query,
        current_key,
        prefix_key,
        torch.zeros(
            (*prefix_key.shape[:-1], program.head_dim),
            dtype=attention_norm.dtype,
            device=attention_norm.device,
        ),
        scaling=scaling,
        compact_prefix_value=compact_prefix_value,
    ).attention_write


def _normalize_attention_mask(
    attention_mask: torch.Tensor | None,
    *,
    batch: int,
    tokens: int,
    total_length: int,
    device: torch.device,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 4 or int(attention_mask.shape[1]) != 1:
        raise QwenJointValueOutputError(
            "fused compact attention mask must have [batch,1,query,key] shape"
        )
    if int(attention_mask.shape[0]) != batch:
        raise QwenJointValueOutputError("fused compact attention-mask batch changed")
    if int(attention_mask.shape[-2]) < tokens or int(attention_mask.shape[-1]) < total_length:
        raise QwenJointValueOutputError("fused compact attention mask is shorter than the step")
    if attention_mask.device != device or not attention_mask.is_floating_point():
        raise QwenJointValueOutputError(
            "fused compact attention mask must be floating point on the input device"
        )
    return attention_mask[..., -tokens:, -total_length:].contiguous()


def _torch_step(
    program: QwenCompactJointValueOutputProgram,
    attention_norm: torch.Tensor,
    query: torch.Tensor,
    current_key: torch.Tensor,
    prefix_key: torch.Tensor,
    compact_prefix_value: torch.Tensor,
    *,
    scaling: float,
    attention_mask: torch.Tensor | None,
) -> FusedCompactStep:
    groups_per_kv = program.query_heads // program.key_value_heads
    current_compact = torch.einsum(
        "btd,gid->bgti", attention_norm, program.input_weight
    ) + program.input_bias[:, None, :]
    key = torch.cat((prefix_key, current_key), dim=-2)
    key_heads = key.repeat_interleave(groups_per_kv, dim=1)
    weights = torch.matmul(query, key_heads.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    compact_value = torch.cat((compact_prefix_value, current_compact), dim=-2)
    compact_heads = compact_value.repeat_interleave(groups_per_kv, dim=1)
    compact_mix = torch.matmul(weights, compact_heads)
    output_coordinates = torch.einsum(
        "bhti,hoi->bto", compact_mix, program.head_cores
    )
    attention_write = (
        torch.einsum("bto,do->btd", output_coordinates, program.output_basis)
        + program.output_bias
    )
    return FusedCompactStep(
        attention_write=attention_write,
        current_compact_value=current_compact,
    )


@lru_cache(maxsize=1)
def _build_triton_kernels() -> tuple[Any, Any, Any] | None:
    modules = _triton_modules()
    if modules is None:
        return None
    triton, tl = modules

    @triton.jit
    def compact_attention_kernel(
        norm_ptr,
        query_ptr,
        prefix_key_ptr,
        current_key_ptr,
        prefix_value_ptr,
        input_weight_ptr,
        input_bias_ptr,
        mix_ptr,
        current_value_ptr,
        attention_mask_ptr,
        batch,
        heads,
        groups,
        tokens,
        prefix_length: tl.constexpr,
        hidden: tl.constexpr,
        head_dim: tl.constexpr,
        value_rank: tl.constexpr,
        scaling,
        HAS_ATTENTION_MASK: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        BLOCK_RV: tl.constexpr,
    ):
        """Fuse current compact V construction with stable attention reduction."""

        program_id = tl.program_id(0)
        token = program_id % tokens
        head = (program_id // tokens) % heads
        batch_index = program_id // (tokens * heads)
        group = head // (heads // groups)

        rv = tl.arange(0, BLOCK_RV)
        rv_mask = rv < value_rank
        current = tl.zeros((BLOCK_RV,), dtype=tl.float32)
        for d_start in range(0, hidden, BLOCK_D):
            d = d_start + tl.arange(0, BLOCK_D)
            d_mask = d < hidden
            norm_offset = (batch_index * tokens + token) * hidden + d
            norm = tl.load(norm_ptr + norm_offset, mask=d_mask, other=0.0).to(tl.float32)
            weight_offset = (group * value_rank + rv[:, None]) * hidden + d[None, :]
            weights = tl.load(
                input_weight_ptr + weight_offset,
                mask=rv_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            current += tl.sum(weights * norm[None, :], axis=1)
        bias_offset = group * value_rank + rv
        current += tl.load(input_bias_ptr + bias_offset, mask=rv_mask, other=0.0).to(
            tl.float32
        )
        group_size = heads // groups
        current_offset = (
            (batch_index * groups + group) * tokens + token
        ) * value_rank + rv
        tl.store(
            current_value_ptr + current_offset,
            current,
            mask=rv_mask & (head == group * group_size),
        )

        query_offset = ((batch_index * heads + head) * tokens + token) * head_dim
        qd = tl.arange(0, BLOCK_DH)
        q_mask = qd < head_dim
        query_values = tl.load(query_ptr + query_offset + qd, mask=q_mask, other=0.0).to(
            tl.float32
        )
        current_key_offset = (
            (batch_index * groups + group) * tokens + token
        ) * head_dim + qd
        current_key_values = tl.load(
            current_key_ptr + current_key_offset, mask=q_mask, other=0.0
        ).to(tl.float32)
        current_score = tl.sum(current_key_values * query_values)
        key_group_stride = prefix_length * head_dim
        max_score = tl.full((), -float("inf"), dtype=tl.float32)
        for s_start in range(0, prefix_length + 1, BLOCK_S):
            s = s_start + tl.arange(0, BLOCK_S)
            prefix_mask = s < prefix_length
            current_mask = s == prefix_length
            score = tl.zeros((BLOCK_S,), dtype=tl.float32)
            prefix_key_offset = batch_index * groups * key_group_stride + group * key_group_stride
            key_offset = prefix_key_offset + s[:, None] * head_dim + qd[None, :]
            prefix_keys = tl.load(
                prefix_key_ptr + key_offset,
                mask=prefix_mask[:, None] & q_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(prefix_keys * query_values[None, :], axis=1)
            score = tl.where(current_mask, current_score, score)
            score = tl.where(prefix_mask | current_mask, score * scaling, -float("inf"))
            if HAS_ATTENTION_MASK:
                mask_offset = (
                    (batch_index * tokens + token) * (prefix_length + 1) + s
                )
                mask_value = tl.load(
                    attention_mask_ptr + mask_offset,
                    mask=prefix_mask | current_mask,
                    other=0.0,
                ).to(tl.float32)
                score += mask_value
            max_score = tl.maximum(max_score, tl.max(score, axis=0))

        denominator = tl.zeros((), dtype=tl.float32)
        mixed = tl.zeros((BLOCK_RV,), dtype=tl.float32)
        for s_start in range(0, prefix_length + 1, BLOCK_S):
            s = s_start + tl.arange(0, BLOCK_S)
            prefix_mask = s < prefix_length
            current_mask = s == prefix_length
            prefix_key_offset = batch_index * groups * key_group_stride + group * key_group_stride
            key_offset = prefix_key_offset + s[:, None] * head_dim + qd[None, :]
            prefix_keys = tl.load(
                prefix_key_ptr + key_offset,
                mask=prefix_mask[:, None] & q_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(prefix_keys * query_values[None, :], axis=1)
            score = tl.where(current_mask, current_score, score)
            score = tl.where(prefix_mask | current_mask, score * scaling, -float("inf"))
            if HAS_ATTENTION_MASK:
                mask_offset = (
                    (batch_index * tokens + token) * (prefix_length + 1) + s
                )
                mask_value = tl.load(
                    attention_mask_ptr + mask_offset,
                    mask=prefix_mask | current_mask,
                    other=0.0,
                ).to(tl.float32)
                score += mask_value
            probability = tl.exp(score - max_score)
            probability = tl.where(prefix_mask | current_mask, probability, 0.0)
            denominator += tl.sum(probability, axis=0)
            value_offset = (
                (batch_index * groups + group) * prefix_length * value_rank
                + s[:, None] * value_rank
                + rv[None, :]
            )
            prefix_values = tl.load(
                prefix_value_ptr + value_offset,
                mask=prefix_mask[:, None] & rv_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            values = tl.where(current_mask[:, None], current[None, :], prefix_values)
            mixed += tl.sum(probability[:, None] * values, axis=0)
        mixed = mixed / denominator
        mix_offset = ((batch_index * heads + head) * tokens + token) * value_rank + rv
        tl.store(mix_ptr + mix_offset, mixed, mask=rv_mask)

    @triton.jit
    def compact_output_kernel(
        mix_ptr,
        head_core_ptr,
        output_basis_ptr,
        output_bias_ptr,
        output_ptr,
        batch,
        heads: tl.constexpr,
        tokens,
        value_rank: tl.constexpr,
        output_rank: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_RV: tl.constexpr,
        BLOCK_RO: tl.constexpr,
    ):
        """Contract compact per-head attention with the compact O basis."""

        program_id = tl.program_id(0)
        d_block = tl.program_id(1)
        token_index = program_id % tokens
        batch_index = program_id // tokens
        d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d < hidden
        result = tl.load(output_bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        rv = tl.arange(0, BLOCK_RV)
        for head in range(heads):
            for output_start in range(0, output_rank, BLOCK_RO):
                output_index = output_start + tl.arange(0, BLOCK_RO)
                output_mask = output_index < output_rank
                coordinates = tl.zeros((BLOCK_RO,), dtype=tl.float32)
                for value_start in range(0, value_rank, BLOCK_RV):
                    value_index = value_start + rv
                    value_mask = value_index < value_rank
                    mix_offset = (
                        (batch_index * heads + head) * tokens + token_index
                    ) * value_rank + value_index
                    mix = tl.load(
                        mix_ptr + mix_offset, mask=value_mask, other=0.0
                    ).to(tl.float32)
                    core_offset = (head * output_rank + output_index[:, None]) * value_rank
                    core = tl.load(
                        head_core_ptr + core_offset + value_index[None, :],
                        mask=output_mask[:, None] & value_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    coordinates += tl.sum(core * mix[None, :], axis=1)
                basis_offset = d[:, None] * output_rank + output_index[None, :]
                basis = tl.load(
                    output_basis_ptr + basis_offset,
                    mask=d_mask[:, None] & output_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                result += tl.sum(basis * coordinates[None, :], axis=1)

        output_offset = (batch_index * tokens + token_index) * hidden + d
        tl.store(output_ptr + output_offset, result, mask=d_mask)

    return compact_attention_kernel, compact_output_kernel, triton


def _triton_execute(
    program: QwenCompactJointValueOutputProgram,
    attention_norm: torch.Tensor,
    query: torch.Tensor,
    current_key: torch.Tensor,
    prefix_key: torch.Tensor,
    compact_prefix_value: torch.Tensor,
    *,
    scaling: float,
    shape: tuple[int, int, int, int],
    attention_mask: torch.Tensor | None,
) -> FusedCompactStep:
    kernels = _build_triton_kernels()
    if kernels is None:
        raise QwenJointValueOutputError("Triton is unavailable")
    attention_kernel, output_kernel, _triton = kernels
    batch, heads, tokens, prefix_length = shape
    groups = program.key_value_heads
    if heads != program.query_heads or groups != int(program.input_weight.shape[0]):
        raise QwenJointValueOutputError("fused head/group geometry changed")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            attention_norm,
            query,
            current_key,
            prefix_key,
            compact_prefix_value,
            program.input_weight,
            program.input_bias,
            program.head_cores,
            program.output_basis,
            program.output_bias,
        )
    ):
        raise QwenJointValueOutputError(
            "fused Triton path requires contiguous inputs and program tensors"
        )

    value_rank = program.value_rank
    hidden = int(program.input_weight.shape[-1])
    # Keep the per-program key tile small enough for head_dim=128 on the
    # 4080.  The sequence reduction is streamed over these tiles, so a
    # smaller tile trades a few more iterations for substantially lower
    # register pressure than a 128-row x 128-column tile.
    block_s = 32
    block_d = 128
    block_dh = _next_power_of_two(program.head_dim)
    block_rv = _next_power_of_two(value_rank)
    compact_mix = torch.empty(
        (batch, heads, tokens, value_rank),
        dtype=attention_norm.dtype,
        device=attention_norm.device,
    )
    current_compact = torch.empty(
        (batch, groups, tokens, value_rank),
        dtype=attention_norm.dtype,
        device=attention_norm.device,
    )
    grid_attention = (batch * heads * tokens,)
    attention_kernel[grid_attention](
        attention_norm,
        query,
        prefix_key,
        current_key,
        compact_prefix_value,
        program.input_weight,
        program.input_bias,
        compact_mix,
        current_compact,
        attention_norm if attention_mask is None else attention_mask,
        batch,
        heads,
        groups,
        tokens,
        prefix_length,
        hidden,
        program.head_dim,
        value_rank,
        float(scaling),
        HAS_ATTENTION_MASK=attention_mask is not None,
        BLOCK_S=block_s,
        BLOCK_D=block_d,
        BLOCK_DH=block_dh,
        BLOCK_RV=block_rv,
        num_warps=4,
    )
    output = torch.empty(
        (batch, tokens, hidden),
        dtype=attention_norm.dtype,
        device=attention_norm.device,
    )
    grid_output = (batch * tokens, (hidden + block_d - 1) // block_d)
    output_kernel[grid_output](
        compact_mix,
        program.head_cores,
        program.output_basis,
        program.output_bias,
        output,
        batch,
        heads,
        tokens,
        value_rank,
        program.output_rank,
        hidden,
        BLOCK_D=block_d,
        BLOCK_RV=64,
        BLOCK_RO=64,
        num_warps=4,
    )
    return FusedCompactStep(
        attention_write=output,
        current_compact_value=current_compact,
    )


@torch.inference_mode()
def execute_fused_compact_step(
    program: QwenCompactJointValueOutputProgram,
    attention_norm: torch.Tensor,
    query: torch.Tensor,
    current_key: torch.Tensor,
    prefix_key: torch.Tensor,
    compact_prefix_value: torch.Tensor,
    *,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
    backend: str = "auto",
    check_finite: bool = True,
) -> FusedCompactStep:
    """Execute one compact attention step and return the value-cache append."""

    shape = _validate_inputs(
        program,
        attention_norm,
        query,
        current_key,
        prefix_key,
        compact_prefix_value,
        check_finite=check_finite,
    )
    batch, _heads, tokens, prefix_length = shape
    normalized_mask = _normalize_attention_mask(
        attention_mask,
        batch=batch,
        tokens=tokens,
        total_length=prefix_length + tokens,
        device=attention_norm.device,
    )
    normalized_backend = str(backend).lower()
    if normalized_backend not in {"auto", "triton", "torch"}:
        raise ValueError("fused compact backend must be auto, triton, or torch")
    use_triton = normalized_backend == "triton" or (
        normalized_backend == "auto" and fused_backend_available(attention_norm.device)
    )
    if normalized_backend == "triton" and not fused_backend_available(attention_norm.device):
        raise QwenJointValueOutputError("requested Triton backend is unavailable")
    if use_triton:
        if tokens != 1:
            raise QwenJointValueOutputError(
                "Triton compact step supports one decode token; use the Torch prefill path"
            )
        return _triton_execute(
            program,
            attention_norm.contiguous(),
            query.contiguous(),
            current_key.contiguous(),
            prefix_key.contiguous(),
            compact_prefix_value.contiguous(),
            scaling=scaling,
            shape=shape,
            attention_mask=normalized_mask,
        )
    return _torch_step(
        program,
        attention_norm,
        query,
        current_key,
        prefix_key,
        compact_prefix_value,
        scaling=scaling,
        attention_mask=normalized_mask,
    )


@torch.inference_mode()
def execute_fused_compact_write(
    program: QwenCompactJointValueOutputProgram,
    attention_norm: torch.Tensor,
    query: torch.Tensor,
    current_key: torch.Tensor,
    prefix_key: torch.Tensor,
    compact_prefix_value: torch.Tensor,
    *,
    scaling: float,
    backend: str = "auto",
    check_finite: bool = True,
) -> torch.Tensor:
    """Execute compact live attention and O write without native-width traces.

    ``backend="triton"`` is fail-closed when Triton cannot compile the
    declared geometry.  ``backend="auto"`` uses Triton on CUDA and the Torch
    reference elsewhere.  The fallback is intentionally named in
    :func:`fused_backend_name`; it is not a fused-performance claim.  Set
    ``check_finite=False`` only after an untimed diagnostic has validated the
    operands; this keeps device-wide finite reductions out of a hot timing arm.
    """

    return execute_fused_compact_step(
        program,
        attention_norm,
        query,
        current_key,
        prefix_key,
        compact_prefix_value,
        scaling=scaling,
        backend=backend,
        check_finite=check_finite,
    ).attention_write


def fused_backend_name(
    device: torch.device | str,
    *,
    requested: str = "auto",
) -> str:
    """Describe the backend that ``execute_fused_compact_write`` will use."""

    normalized = str(requested).lower()
    if normalized not in {"auto", "triton", "torch"}:
        raise ValueError("fused compact backend must be auto, triton, or torch")
    if normalized == "torch":
        return "torch-unfused-reference"
    if fused_backend_available(device):
        return "triton-two-kernel-compact"
    if normalized == "triton":
        return "triton-unavailable"
    return "torch-unfused-reference-fallback"


__all__ = [
    "FusedCompactStep",
    "execute_fused_compact_step",
    "execute_fused_compact_write",
    "fused_backend_available",
    "fused_backend_name",
]

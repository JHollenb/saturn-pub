"""Joint Qwen V -> attention -> O object code.

For query head ``h`` owned by KV group ``g(h)``, the residual write is

    sum_h O_h sum_j a_hj V_g x_j.

This module compiles that complete path. The exact program is an independently
owned copied-weight oracle. The compact program uses consumer-weighted value
bases and a shared residual-output basis while retaining the live attention
query and weights.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


class QwenJointValueOutputError(ValueError):
    pass


def _tensor_bytes(*values: torch.Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def _repeat_groups(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    batch, heads, length, width = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, groups, length, width)
        .reshape(batch, heads * groups, length, width)
    )


def _attention_weights(
    query: torch.Tensor,
    prefix_key: torch.Tensor,
    current_key: torch.Tensor,
    *,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    key = torch.cat((prefix_key, current_key), dim=-2)
    key_heads = _repeat_groups(key, groups)
    weights = torch.matmul(query, key_heads.transpose(2, 3)) * scaling
    return F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)


@dataclass(frozen=True, slots=True)
class JointAttentionWrite:
    current_value: torch.Tensor
    attention_mix: torch.Tensor
    attention_write: torch.Tensor


@dataclass(frozen=True, slots=True)
class CompactJointAttentionCore:
    """Native-width-free outputs of the compact hot path."""

    current_compact_value: torch.Tensor
    compact_attention_mix: torch.Tensor
    attention_write: torch.Tensor


@dataclass(frozen=True, slots=True)
class QwenExactJointValueOutputProgram:
    """Bit-parity oracle stored independently of the native modules."""

    value_weight: torch.Tensor
    value_bias: torch.Tensor | None
    output_weight: torch.Tensor
    output_bias: torch.Tensor | None
    query_heads: int
    key_value_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        hidden = int(self.output_weight.shape[0])
        if self.query_heads < 1 or self.key_value_heads < 1:
            raise QwenJointValueOutputError("head counts must be positive")
        if self.query_heads % self.key_value_heads:
            raise QwenJointValueOutputError("query heads must divide into KV groups")
        if self.value_weight.shape != (self.key_value_heads * self.head_dim, hidden):
            raise QwenJointValueOutputError("copied V weight geometry changed")
        if self.output_weight.shape != (hidden, self.query_heads * self.head_dim):
            raise QwenJointValueOutputError("copied O weight geometry changed")

    @property
    def tensor_bytes(self) -> int:
        values = [self.value_weight, self.output_weight]
        if self.value_bias is not None:
            values.append(self.value_bias)
        if self.output_bias is not None:
            values.append(self.output_bias)
        return _tensor_bytes(*values)

    def execute(
        self,
        attention_norm: torch.Tensor,
        query: torch.Tensor,
        current_key: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        *,
        scaling: float,
    ) -> JointAttentionWrite:
        groups = self.query_heads // self.key_value_heads
        current_flat = F.linear(attention_norm, self.value_weight, self.value_bias)
        current_value = current_flat.view(
            attention_norm.shape[0],
            attention_norm.shape[1],
            self.key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        weights = _attention_weights(
            query,
            prefix_key,
            current_key,
            groups=groups,
            scaling=scaling,
        )
        value = torch.cat((prefix_value, current_value), dim=-2)
        value_heads = _repeat_groups(value, groups)
        mixed = torch.matmul(weights, value_heads).transpose(1, 2).contiguous()
        attention_mix = mixed.reshape(*attention_norm.shape[:-1], -1).contiguous()
        attention_write = F.linear(attention_mix, self.output_weight, self.output_bias)
        return JointAttentionWrite(current_value, attention_mix, attention_write)


@dataclass(frozen=True, slots=True)
class QwenCompactJointValueOutputProgram:
    """Two-sided factorization of grouped V reads and head-specific O writes."""

    input_weight: torch.Tensor  # [G, r_v, d_model]
    input_bias: torch.Tensor  # [G, r_v]
    value_basis: torch.Tensor  # [G, d_head, r_v]
    output_basis: torch.Tensor  # [d_model, r_o]
    head_cores: torch.Tensor  # [H, r_o, r_v]
    output_bias: torch.Tensor  # [d_model]
    query_heads: int
    key_value_heads: int
    head_dim: int
    native_tensor_bytes: int
    value_weight_relative_l2: float
    joint_output_relative_l2: float

    def __post_init__(self) -> None:
        groups, value_rank, hidden = self.input_weight.shape
        output_hidden, output_rank = self.output_basis.shape
        if groups != self.key_value_heads or output_hidden != hidden:
            raise QwenJointValueOutputError("compact hidden or group geometry changed")
        if self.value_basis.shape != (groups, self.head_dim, value_rank):
            raise QwenJointValueOutputError("compact value basis geometry changed")
        if self.input_bias.shape != (groups, value_rank):
            raise QwenJointValueOutputError("compact input bias geometry changed")
        if self.head_cores.shape != (self.query_heads, output_rank, value_rank):
            raise QwenJointValueOutputError("compact head-core geometry changed")
        if self.output_bias.shape != (hidden,):
            raise QwenJointValueOutputError("compact output bias geometry changed")
        if self.query_heads % self.key_value_heads:
            raise QwenJointValueOutputError("query heads must divide into KV groups")
        if self.native_tensor_bytes < 1:
            raise QwenJointValueOutputError("native tensor byte count must be positive")
        for value in (
            self.input_weight,
            self.input_bias,
            self.value_basis,
            self.output_basis,
            self.head_cores,
            self.output_bias,
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise QwenJointValueOutputError("compact tensors must be finite floating point")

    @property
    def value_rank(self) -> int:
        return int(self.input_weight.shape[1])

    @property
    def output_rank(self) -> int:
        return int(self.output_basis.shape[1])

    @property
    def tensor_bytes(self) -> int:
        return _tensor_bytes(
            self.input_weight,
            self.input_bias,
            self.value_basis,
            self.output_basis,
            self.head_cores,
            self.output_bias,
        )

    def descriptor(self) -> dict[str, Any]:
        body = {
            "schema": "saturn-qwen-joint-value-output-program-v1",
            "compiler": "consumer-weighted-two-sided-hosvd-v1",
            "query_heads": self.query_heads,
            "key_value_heads": self.key_value_heads,
            "head_dim": self.head_dim,
            "hidden_size": int(self.input_weight.shape[-1]),
            "value_rank": self.value_rank,
            "output_rank": self.output_rank,
            "tensor_bytes": self.tensor_bytes,
            "native_tensor_bytes": self.native_tensor_bytes,
            "storage_ratio_vs_native_v_o": self.tensor_bytes / self.native_tensor_bytes,
            "value_weight_relative_l2": self.value_weight_relative_l2,
            "joint_output_relative_l2": self.joint_output_relative_l2,
            "uses_activation_chart": False,
            "uses_live_query": True,
            "prefix_value_representation": "projected-through-value-basis-at-runtime",
        }
        body["fingerprint"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return body

    def compile_prefix_value(self, prefix_value: torch.Tensor) -> torch.Tensor:
        """Lower native prefix V once into the compact cache representation."""

        if prefix_value.ndim != 4:
            raise QwenJointValueOutputError("prefix V must have [batch, group, row, head] shape")
        if prefix_value.shape[1] != self.key_value_heads:
            raise QwenJointValueOutputError("prefix V group count changed")
        if prefix_value.shape[-1] != self.head_dim:
            raise QwenJointValueOutputError("prefix V head dimension changed")
        return torch.einsum("bgld,gdi->bgli", prefix_value, self.value_basis)

    def execute_core(
        self,
        attention_norm: torch.Tensor,
        query: torch.Tensor,
        current_key: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        *,
        scaling: float,
        compact_prefix_value: torch.Tensor | None = None,
    ) -> CompactJointAttentionCore:
        """Execute without reconstructing native V or concatenated head values."""

        groups_per_kv = self.query_heads // self.key_value_heads
        weights = _attention_weights(
            query,
            prefix_key,
            current_key,
            groups=groups_per_kv,
            scaling=scaling,
        )
        current_compact = torch.einsum(
            "btd,gid->bgti", attention_norm, self.input_weight
        ) + self.input_bias[:, None, :]
        prefix_compact = (
            self.compile_prefix_value(prefix_value)
            if compact_prefix_value is None
            else compact_prefix_value
        )
        expected_prefix = (
            int(prefix_key.shape[0]),
            self.key_value_heads,
            int(prefix_key.shape[-2]),
            self.value_rank,
        )
        if prefix_compact.shape != expected_prefix:
            raise QwenJointValueOutputError(
                "compact prefix V geometry changed: "
                f"{tuple(prefix_compact.shape)} != {expected_prefix}"
            )
        if prefix_compact.device != attention_norm.device:
            raise QwenJointValueOutputError("compact prefix V device changed")
        compact_value = torch.cat((prefix_compact, current_compact), dim=-2)
        compact_heads = _repeat_groups(compact_value, groups_per_kv)
        compact_mix = torch.matmul(weights, compact_heads)
        output_coordinates = torch.einsum("bhti,hoi->bto", compact_mix, self.head_cores)
        attention_write = torch.einsum(
            "bto,do->btd", output_coordinates, self.output_basis
        ) + self.output_bias
        return CompactJointAttentionCore(
            current_compact_value=current_compact,
            compact_attention_mix=compact_mix,
            attention_write=attention_write,
        )

    def execute(
        self,
        attention_norm: torch.Tensor,
        query: torch.Tensor,
        current_key: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        *,
        scaling: float,
        compact_prefix_value: torch.Tensor | None = None,
    ) -> JointAttentionWrite:
        """Execute and materialize native-width debugger compatibility fields."""

        core = self.execute_core(
            attention_norm,
            query,
            current_key,
            prefix_key,
            prefix_value,
            scaling=scaling,
            compact_prefix_value=compact_prefix_value,
        )

        current_value = torch.einsum(
            "bgti,gdi->bgtd", core.current_compact_value, self.value_basis
        )
        groups_per_kv = self.query_heads // self.key_value_heads
        basis_heads = (
            self.value_basis[:, None, :, :]
            .expand(
                self.key_value_heads,
                groups_per_kv,
                self.head_dim,
                self.value_rank,
            )
            .reshape(self.query_heads, self.head_dim, self.value_rank)
        )
        mixed_full = torch.einsum(
            "bhti,hdi->bhtd", core.compact_attention_mix, basis_heads
        )
        attention_mix = (
            mixed_full.transpose(1, 2)
            .contiguous()
            .reshape(*attention_norm.shape[:-1], self.query_heads * self.head_dim)
        )
        return JointAttentionWrite(current_value, attention_mix, core.attention_write)


def compile_qwen_exact_joint_value_output(attention: Any) -> QwenExactJointValueOutputProgram:
    value = attention.v_proj
    output = attention.o_proj
    if not isinstance(getattr(value, "weight", None), torch.Tensor):
        raise QwenJointValueOutputError("native V weight is unavailable")
    if not isinstance(getattr(output, "weight", None), torch.Tensor):
        raise QwenJointValueOutputError("native O weight is unavailable")
    return QwenExactJointValueOutputProgram(
        value_weight=value.weight.detach().clone(),
        value_bias=None if value.bias is None else value.bias.detach().clone(),
        output_weight=output.weight.detach().clone(),
        output_bias=None if output.bias is None else output.bias.detach().clone(),
        query_heads=int(attention.config.num_attention_heads),
        key_value_heads=int(attention.config.num_key_value_heads),
        head_dim=int(attention.head_dim),
    )


@torch.inference_mode()
def compile_qwen_compact_joint_value_output(
    attention: Any,
    *,
    value_rank: int,
    output_rank: int,
    artifact_dtype: torch.dtype | None = None,
) -> QwenCompactJointValueOutputProgram:
    """Compile the joint path using consumer-weighted value/output bases."""

    value_weight = getattr(attention.v_proj, "weight", None)
    output_weight = getattr(attention.o_proj, "weight", None)
    if not isinstance(value_weight, torch.Tensor) or not isinstance(output_weight, torch.Tensor):
        raise QwenJointValueOutputError("native V/O weights must exist during compilation")
    query_heads = int(attention.config.num_attention_heads)
    key_value_heads = int(attention.config.num_key_value_heads)
    head_dim = int(attention.head_dim)
    if query_heads % key_value_heads:
        raise QwenJointValueOutputError("query heads must divide into KV groups")
    hidden = int(output_weight.shape[0])
    if not 1 <= value_rank <= head_dim:
        raise QwenJointValueOutputError("value rank lies outside head dimension")
    maximum_output_rank = min(hidden, query_heads * value_rank)
    if not 1 <= output_rank <= maximum_output_rank:
        raise QwenJointValueOutputError(
            f"output rank must be in [1, {maximum_output_rank}]"
        )

    device = value_weight.device
    target_dtype = value_weight.dtype if artifact_dtype is None else artifact_dtype
    value_native = value_weight.detach().float().reshape(key_value_heads, head_dim, hidden)
    output_native = (
        output_weight.detach().float().reshape(hidden, query_heads, head_dim).permute(1, 0, 2)
    )
    heads_per_group = query_heads // key_value_heads
    bases = []
    for group in range(key_value_heads):
        first = group * heads_per_group
        last = first + heads_per_group
        group_output = output_native[first:last]
        metric = torch.einsum("hdi,hdj->ij", group_output, group_output)
        eigenvalues, eigenvectors = torch.linalg.eigh(metric)
        order = torch.argsort(eigenvalues, descending=True)[:value_rank]
        bases.append(eigenvectors[:, order])
    value_basis = torch.stack(bases)

    projected_columns = []
    for head in range(query_heads):
        group = head // heads_per_group
        projected_columns.append(output_native[head] @ value_basis[group])
    output_column_matrix = torch.cat(projected_columns, dim=1)
    output_basis = torch.linalg.svd(output_column_matrix, full_matrices=False).U[:, :output_rank]
    head_basis = value_basis.repeat_interleave(heads_per_group, dim=0)
    head_cores = torch.einsum("do,hde,hei->hoi", output_basis, output_native, head_basis)
    input_weight = torch.einsum("gdi,gdk->gik", value_basis, value_native)
    native_value_bias = getattr(attention.v_proj, "bias", None)
    if native_value_bias is None:
        input_bias = torch.zeros(
            key_value_heads, value_rank, dtype=torch.float32, device=device
        )
    else:
        input_bias = torch.einsum(
            "gdi,gd->gi",
            value_basis,
            native_value_bias.detach().float().reshape(key_value_heads, head_dim),
        )
    native_output_bias = getattr(attention.o_proj, "bias", None)
    output_bias = (
        torch.zeros(hidden, dtype=torch.float32, device=device)
        if native_output_bias is None
        else native_output_bias.detach().float().clone()
    )

    reconstructed_value = torch.einsum("gdi,gik->gdk", value_basis, input_weight)
    value_denominator = max(float(value_native.norm().item()), 1.0e-12)
    value_error = float((reconstructed_value - value_native).norm().item() / value_denominator)
    reconstructed_output = torch.einsum(
        "do,hoi,hei->hde", output_basis, head_cores, head_basis
    )
    output_denominator = max(float(output_native.norm().item()), 1.0e-12)
    output_error = float(
        (reconstructed_output - output_native).norm().item() / output_denominator
    )
    native_values = [value_weight, output_weight]
    for optional in (native_value_bias, native_output_bias):
        if isinstance(optional, torch.Tensor):
            native_values.append(optional)

    return QwenCompactJointValueOutputProgram(
        input_weight=input_weight.to(dtype=target_dtype),
        input_bias=input_bias.to(dtype=target_dtype),
        value_basis=value_basis.to(dtype=target_dtype),
        output_basis=output_basis.to(dtype=target_dtype),
        head_cores=head_cores.to(dtype=target_dtype),
        output_bias=output_bias.to(dtype=target_dtype),
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        native_tensor_bytes=_tensor_bytes(*native_values),
        value_weight_relative_l2=value_error,
        joint_output_relative_l2=output_error,
    )


def joint_attention_writer(program: Any):
    """Adapt a compiled program to ``execute_qwen_one_row_algebra``."""

    def write(
        _layer: int,
        _residual_before: torch.Tensor,
        attention_norm: torch.Tensor,
        query: torch.Tensor,
        current_key: torch.Tensor,
        prefix_key: torch.Tensor,
        prefix_value: torch.Tensor,
        attention: Any,
    ) -> JointAttentionWrite:
        return program.execute(
            attention_norm,
            query,
            current_key,
            prefix_key,
            prefix_value,
            scaling=float(attention.scaling),
        )

    return write


__all__ = [
    "CompactJointAttentionCore",
    "JointAttentionWrite",
    "QwenCompactJointValueOutputProgram",
    "QwenExactJointValueOutputProgram",
    "QwenJointValueOutputError",
    "compile_qwen_compact_joint_value_output",
    "compile_qwen_exact_joint_value_output",
    "joint_attention_writer",
]

"""Typed compiled prefix state for recurrent Mamba-family decoders."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from mdb.tensor_hash import tensor_bytes_sha256


class MambaStateProgramError(RuntimeError):
    """Raised when a recurrent source program violates its ABI."""


def _cache_state(container: Any, *, label: str) -> torch.Tensor:
    """Read one Mamba state without assuming the Transformers container ABI.

    Transformers 5.13 stored the single Mamba state directly as a tensor;
    5.14 stores it in a state-indexed mapping. Older fixtures also used a
    one-element sequence. Indexing the tensor as though it were a container
    silently drops the batch dimension, so discriminate the representations
    before selecting state zero.
    """

    if isinstance(container, torch.Tensor):
        value = container
    elif isinstance(container, Mapping):
        value = container.get(0)
    else:
        value = container[0]
    if not isinstance(value, torch.Tensor):
        raise MambaStateProgramError(f"{label} is uninitialized")
    return value


def mamba_execution_abi(config: Any) -> dict[str, Any]:
    """Describe the native runtime that gives a recurrent state its meaning.

    Mamba cache tensors are not identified by weights and tokenizer alone. The
    scan implementation and numerical runtime can change the compiled state and
    its continuation basin. Keep this descriptor small, deterministic, and
    independent of the state payload itself.
    """

    import transformers
    from transformers.models.mamba import modeling_mamba

    config_payload = config.to_dict() if hasattr(config, "to_dict") else repr(config)
    kernels = {
        name: getattr(modeling_mamba, name, None) is not None
        for name in (
            "selective_state_update",
            "selective_scan_fn",
            "causal_conv1d_fn",
            "causal_conv1d_update",
            "mamba_inner_fn",
        )
    }
    body = {
        "schema": "saturn-mamba-execution-abi-v1",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "config_class": f"{type(config).__module__}.{type(config).__qualname__}",
        "config_sha256": hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "cache_class": "transformers.cache_utils.DynamicCache",
        "kernel_availability": kernels,
        "scan_backend": "optimized" if all(kernels.values()) else "sequential-fallback",
    }
    fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "fingerprint": fingerprint}


def _optional_sha256(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise MambaStateProgramError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class MambaLayerState:
    """One layer's bound local-convolution and recurrent source state."""

    conv: torch.Tensor
    recurrent: torch.Tensor

    def clone(self) -> MambaLayerState:
        return MambaLayerState(self.conv.clone(), self.recurrent.clone())


@dataclass(frozen=True)
class MambaStateProgram:
    """A sealed prefix compiled into the Mamba cache ABI."""

    model_id: str
    revision: str
    source_token_ids: tuple[int, ...]
    pending_token_id: int
    layers: tuple[MambaLayerState, ...]
    model_config_sha256: str | None = None
    tokenizer_sha256: str | None = None
    source_program_fingerprint: str | None = None
    execution_abi_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise MambaStateProgramError("state program identity is incomplete")
        if not isinstance(self.pending_token_id, int) or isinstance(self.pending_token_id, bool):
            raise MambaStateProgramError("pending token ID must be an integer")
        if not self.source_token_ids:
            raise MambaStateProgramError("source state has no source token IDs")
        if not self.layers:
            raise MambaStateProgramError("source program has no recurrent layers")
        object.__setattr__(
            self,
            "model_config_sha256",
            _optional_sha256(self.model_config_sha256, "model config hash"),
        )
        object.__setattr__(
            self,
            "tokenizer_sha256",
            _optional_sha256(self.tokenizer_sha256, "tokenizer hash"),
        )
        object.__setattr__(
            self,
            "source_program_fingerprint",
            _optional_sha256(self.source_program_fingerprint, "source program fingerprint"),
        )
        object.__setattr__(
            self,
            "execution_abi_sha256",
            _optional_sha256(self.execution_abi_sha256, "execution ABI hash"),
        )
        for index, layer in enumerate(self.layers):
            for operand in ("conv", "recurrent"):
                tensor = getattr(layer, operand)
                if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                    raise MambaStateProgramError(
                        f"layer {index} {operand} state is not a non-empty tensor"
                    )
                if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all():
                    raise MambaStateProgramError(
                        f"layer {index} {operand} state is not finite floating-point data"
                    )

    @classmethod
    def from_cache(
        cls,
        *,
        cache: Any,
        model_id: str,
        revision: str,
        source_token_ids: tuple[int, ...],
        pending_token_id: int,
        model_config_sha256: str | None = None,
        tokenizer_sha256: str | None = None,
        source_program_fingerprint: str | None = None,
        execution_abi_sha256: str | None = None,
        batch_index: int | None = None,
    ) -> MambaStateProgram:
        if batch_index is not None and (
            not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0
        ):
            raise MambaStateProgramError("batch index must be a non-negative integer")
        layers = []
        for index, layer in enumerate(cache.layers):
            if not hasattr(layer, "conv_states") or not hasattr(layer, "recurrent_states"):
                raise MambaStateProgramError(f"cache layer {index} is not recurrent")
            conv = _cache_state(layer.conv_states, label=f"cache layer {index} conv state")
            recurrent = _cache_state(
                layer.recurrent_states,
                label=f"cache layer {index} recurrent state",
            )
            if batch_index is not None:
                if conv.ndim == 0 or recurrent.ndim == 0:
                    raise MambaStateProgramError(
                        f"cache layer {index} states have no batch axis"
                    )
                if batch_index >= conv.shape[0] or batch_index >= recurrent.shape[0]:
                    raise MambaStateProgramError(
                        f"batch index {batch_index} is outside cache layer {index}"
                    )
                conv = conv[batch_index : batch_index + 1]
                recurrent = recurrent[batch_index : batch_index + 1]
            layers.append(MambaLayerState(conv.detach().clone(), recurrent.detach().clone()))
        if not layers:
            raise MambaStateProgramError("source program has no recurrent layers")
        return cls(
            model_id=model_id,
            revision=revision,
            source_token_ids=source_token_ids,
            pending_token_id=pending_token_id,
            layers=tuple(layers),
            model_config_sha256=model_config_sha256,
            tokenizer_sha256=tokenizer_sha256,
            source_program_fingerprint=source_program_fingerprint,
            execution_abi_sha256=execution_abi_sha256,
        )

    def relocated(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MambaStateProgram:
        """Return a non-aliasing copy at an explicit durable/execution placement."""

        return MambaStateProgram(
            model_id=self.model_id,
            revision=self.revision,
            source_token_ids=self.source_token_ids,
            pending_token_id=self.pending_token_id,
            layers=tuple(
                MambaLayerState(
                    layer.conv.to(device=device, dtype=dtype).clone(),
                    layer.recurrent.to(device=device, dtype=dtype).clone(),
                )
                for layer in self.layers
            ),
            model_config_sha256=self.model_config_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            source_program_fingerprint=self.source_program_fingerprint,
            execution_abi_sha256=self.execution_abi_sha256,
        )

    def transformed(
        self,
        *,
        admitted_layers: tuple[bool, ...] | None = None,
        keep_conv: bool = True,
        keep_recurrent: bool = True,
        layer_order: tuple[int, ...] | None = None,
    ) -> MambaStateProgram:
        count = len(self.layers)
        admitted = admitted_layers or (True,) * count
        order = layer_order or tuple(range(count))
        if len(admitted) != count or sorted(order) != list(range(count)):
            raise MambaStateProgramError("layer transform does not match source ABI")
        layers = []
        for recipient, donor in enumerate(order):
            source = self.layers[donor]
            conv = source.conv.clone()
            recurrent = source.recurrent.clone()
            if not admitted[recipient] or not keep_conv:
                conv.zero_()
            if not admitted[recipient] or not keep_recurrent:
                recurrent.zero_()
            layers.append(MambaLayerState(conv, recurrent))
        return MambaStateProgram(
            model_id=self.model_id,
            revision=self.revision,
            source_token_ids=self.source_token_ids,
            pending_token_id=self.pending_token_id,
            layers=tuple(layers),
            model_config_sha256=self.model_config_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            source_program_fingerprint=self.source_program_fingerprint,
            execution_abi_sha256=self.execution_abi_sha256,
        )

    def hydrate(
        self,
        config: Any,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        batch_size: int = 1,
        preserve_state_dtype: bool = False,
    ) -> Any:
        """Create a non-aliasing native cache at an explicit execution placement.

        Durable programs are normally CPU resident.  ``device`` and ``dtype``
        are therefore execution coordinates, not properties of the artifact.
        Omitting them preserves the source tensor placement for compatibility.
        ``preserve_state_dtype`` is used when replaying a native cache whose
        state slots have different dtypes (for example, Transformers Mamba's
        BF16 convolution state and FP32 recurrent state).
        """
        from transformers.cache_utils import DynamicCache

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise MambaStateProgramError("batch size must be a positive integer")

        current_abi = mamba_execution_abi(config)["fingerprint"]
        if self.execution_abi_sha256 is not None and self.execution_abi_sha256 != current_abi:
            raise MambaStateProgramError(
                "Mamba execution ABI changed; recurrent state requires recompilation"
            )

        cache = DynamicCache(config=config)
        if len(cache.layers) != len(self.layers):
            raise MambaStateProgramError(
                f"layer count mismatch: cache={len(cache.layers)} program={len(self.layers)}"
            )
        for index, (cache_layer, source) in enumerate(zip(cache.layers, self.layers, strict=True)):
            if not hasattr(cache_layer, "update_conv_state") or not hasattr(
                cache_layer, "update_recurrent_state"
            ):
                raise MambaStateProgramError(f"cache layer {index} lacks recurrent update ABI")
            if preserve_state_dtype:
                conv = source.conv.to(device=device)
                recurrent = source.recurrent.to(device=device)
            else:
                conv = source.conv.to(device=device, dtype=dtype)
                recurrent = source.recurrent.to(device=device, dtype=dtype)
            if conv.ndim == 0 or recurrent.ndim == 0:
                raise MambaStateProgramError(f"layer {index} state has no batch axis")
            if conv.shape[0] != 1 or recurrent.shape[0] != 1:
                raise MambaStateProgramError(
                    "MambaStateProgram hydration requires a single-source batch axis"
                )
            if batch_size != 1:
                conv = conv.expand((batch_size, *conv.shape[1:]))
                recurrent = recurrent.expand((batch_size, *recurrent.shape[1:]))
            conv = conv.clone()
            recurrent = recurrent.clone()
            cache_layer.update_conv_state(
                conv,
                state_idx=0,
                conv_kernel_size=int(conv.shape[-1]),
            )
            cache_layer.update_recurrent_state(recurrent, state_idx=0)
        return cache

    @property
    def tensor_bytes(self) -> int:
        return sum(
            item.numel() * item.element_size()
            for layer in self.layers
            for item in (layer.conv, layer.recurrent)
        )

    def manifest(self) -> dict[str, Any]:
        body = {
            "schema": "saturn-mamba-state-program-v1",
            "cache_abi": "transformers-dynamic-cache-linear-attention-v1",
            "model_id": self.model_id,
            "revision": self.revision,
            "source_token_ids": list(self.source_token_ids),
            "pending_token_id": self.pending_token_id,
            "model_config_sha256": self.model_config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "source_program_fingerprint": self.source_program_fingerprint,
            "tensor_bytes": self.tensor_bytes,
            "layers": [
                {
                    "layer": index,
                    "conv_shape": list(layer.conv.shape),
                    "conv_sha256": tensor_bytes_sha256(layer.conv),
                    "recurrent_shape": list(layer.recurrent.shape),
                    "recurrent_sha256": tensor_bytes_sha256(layer.recurrent),
                }
                for index, layer in enumerate(self.layers)
            ],
        }
        if self.execution_abi_sha256 is not None:
            body["execution_abi_sha256"] = self.execution_abi_sha256
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return {**body, "fingerprint": hashlib.sha256(canonical).hexdigest()}

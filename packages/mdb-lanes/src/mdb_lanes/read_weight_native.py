"""Native-profile functional Qwen2 source-to-read replay.

The caller supplies a native input at layer 19, the locally reconstructed
layer-19 attention output, committed KV history for layers 20--22, and the
current RoPE values.  This module applies the remaining Qwen2 equations with
ordinary functional operations over the resident BF16 weights.  It never calls
an original model module's ``forward`` method.

The arithmetic follows ``CausalContextQwenRuntime.advance_token`` on its
``native-eager`` lane.  It is a dense, per-state functional execution, not a
cheap replacement for the 24 linear maps it evaluates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Mapping
from typing import Any


class ReadWeightNativeError(ValueError):
    """A model, tensor, or native-profile contract is invalid."""


_SOURCE_LAYER = 19
_TARGET_LAYER = 22
_HISTORY_LAYERS = (20, 21, 22)
_NATIVE_PROFILE = "causal-context-qwen2-bf16-native-eager-v1"


def _torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - framework-free import contract
        raise ReadWeightNativeError("native read replay requires PyTorch") from exc
    try:
        modeling = importlib.import_module("transformers.models.qwen2.modeling_qwen2")
    except ImportError as exc:  # pragma: no cover - optional dependency contract
        raise ReadWeightNativeError("native read replay requires Transformers Qwen2 operations") from exc
    return torch, functional, modeling


def _finite_tensor(value: Any, label: str, *, torch: Any, shape: tuple[int, ...] | None = None,
                   dtype: Any | None = None, device: Any | None = None):
    if not isinstance(value, torch.Tensor):
        raise ReadWeightNativeError(f"{label} must be a tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ReadWeightNativeError(f"{label} shape differs")
    if dtype is not None and value.dtype != dtype:
        raise ReadWeightNativeError(f"{label} dtype differs")
    if device is not None and value.device != device:
        raise ReadWeightNativeError(f"{label} device differs")
    if value.numel() == 0 or not bool(torch.isfinite(value).all()):
        raise ReadWeightNativeError(f"{label} must be finite and nonempty")
    return value


def _config_identity(config: Any) -> tuple[Any, ...]:
    rope_scaling = json.dumps(getattr(config, "rope_scaling", None), sort_keys=True,
                              separators=(",", ":"), default=str)
    return (
        getattr(config, "model_type", None),
        getattr(config, "hidden_size", None),
        getattr(config, "num_hidden_layers", None),
        getattr(config, "num_attention_heads", None),
        getattr(config, "num_key_value_heads", None),
        getattr(config, "head_dim", None),
        getattr(config, "hidden_act", None),
        getattr(config, "rms_norm_eps", None),
        getattr(config, "use_sliding_window", False),
        getattr(config, "_attn_implementation", None),
        getattr(config, "max_position_embeddings", None),
        getattr(config, "rope_theta", None),
        rope_scaling,
    )


def _module_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    rows = []
    for index in range(_SOURCE_LAYER, _TARGET_LAYER + 1):
        layer = model.model.layers[index]
        for component, module in (
            ("input_layernorm", layer.input_layernorm),
            ("post_attention_layernorm", layer.post_attention_layernorm),
            ("attention", layer.self_attn),
            ("mlp", layer.mlp),
        ):
            for name, parameter in module.named_parameters(recurse=True):
                rows.append((f"{index}.{component}.{name}", parameter))
    return tuple(rows)


def _module_identity(module: Any) -> tuple[Any, ...] | None:
    if module is None:
        return None
    kind = type(module)
    return id(module), kind.__module__, kind.__qualname__


def _live_execution_signature(model: Any) -> tuple[Any, ...]:
    layers = model.model.layers
    rows: list[tuple[Any, ...]] = [
        ("model", _module_identity(model), bool(model.training)),
        ("body", _module_identity(model.model), id(layers)),
    ]
    for index in range(_SOURCE_LAYER, _TARGET_LAYER + 1):
        layer = layers[index]
        input_norm = layer.input_layernorm
        post_norm = layer.post_attention_layernorm
        attention = layer.self_attn
        mlp = layer.mlp
        linears = tuple(
            (name, _module_identity(getattr(attention, name)))
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        rows.append((
            "layer", index,
            _module_identity(layer), bool(layer.training),
            _module_identity(input_norm), repr(getattr(input_norm, "variance_epsilon", None)),
            bool(input_norm.training),
            _module_identity(post_norm), repr(getattr(post_norm, "variance_epsilon", None)),
            bool(post_norm.training),
            _module_identity(attention), bool(attention.training),
            repr(getattr(attention, "scaling", None)),
            repr(getattr(attention, "num_key_value_groups", None)),
            repr(getattr(attention, "head_dim", None)),
            repr(getattr(attention, "sliding_window", None)),
            repr(getattr(attention, "q_norm", None)),
            repr(getattr(attention, "k_norm", None)), linears,
            _module_identity(mlp), bool(mlp.training),
            _module_identity(getattr(mlp, "act_fn", None)),
        ))
    return tuple(rows)


def _parameter_signature(rows: tuple[tuple[str, Any], ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (name, id(value), value.data_ptr(), tuple(value.shape), str(value.dtype), int(value._version))
        for name, value in rows
    )


def _digest_identity(config_identity: tuple[Any, ...], signature: tuple[tuple[Any, ...], ...],
                     execution_signature: tuple[Any, ...]) -> str:
    # This identifies live, in-process parameters and execution policy.  It is
    # deliberately not a portable checkpoint digest or custody substitute.
    body = {"profile": _NATIVE_PROFILE, "config": config_identity,
            "parameters": [list(row) for row in signature],
            "execution_policy": execution_signature}
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _linear(functional: Any, module: Any, value: Any):
    return functional.linear(value, module.weight, module.bias)


def _rms_norm(value: Any, module: Any, *, torch: Any):
    # Same cast boundary as Qwen2RMSNorm.forward in the qualified runtime:
    # FP32 accumulation, return to the incoming dtype, then multiply gamma.
    input_dtype = value.dtype
    widened = value.to(torch.float32)
    variance = widened.pow(2).mean(-1, keepdim=True)
    normalized = widened * torch.rsqrt(variance + float(module.variance_epsilon))
    return module.weight * normalized.to(input_dtype)


def _mlp(value: Any, module: Any, *, torch: Any, functional: Any):
    gate = _linear(functional, module.gate_proj, value)
    up = _linear(functional, module.up_proj, value)
    activated = functional.silu(gate) * up
    return _linear(functional, module.down_proj, activated)


def _attention(
    query: Any,
    key_bank: Any,
    value_bank: Any,
    module: Any,
    *,
    torch: Any,
    functional: Any,
    modeling: Any,
):
    """Mirror Qwen2 eager_attention_forward while retaining its score tape."""
    key_states = modeling.repeat_kv(key_bank, module.num_key_value_groups)
    value_states = modeling.repeat_kv(value_bank, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * float(module.scaling)
    weights = functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = functional.dropout(weights, p=0.0, training=module.training)
    attended = torch.matmul(weights, value_states)
    # Qwen2 eager returns [batch, query, heads, head_dim].  The public
    # CausalContext runtime transposes to [batch, heads, query, head_dim], then
    # transposes back before flattening for o_proj; preserve those boundaries.
    attended = attended.transpose(1, 2).contiguous()
    attended = attended.transpose(1, 2)
    projection_input = attended.transpose(1, 2).reshape(1, 1, -1).contiguous()
    output = _linear(functional, module.o_proj, projection_input)
    return scores, weights, output


class NativeQwenReadTransport:
    """Evaluate the native-BF16 L19 attention write through the L22 block.

    ``history`` maps layers 20, 21, and 22 to committed ``(key, value)`` banks,
    each shaped ``[1, kv_heads, rows, head_dim]`` and excluding the current
    row.  ``incoming`` is the native-zero L19 input row. ``cos`` and ``sin``
    are the exact current-position RoPE values supplied by the caller.

    ``evaluate(attention19)`` returns the A path.  ``value_edit`` optionally
    scales one *historical* value row only at L22, which gives the direct AB
    block output without subtracting a projected value contribution from an
    observed A attention output.
    """

    logical_dense_linears = 24

    def __init__(
        self,
        model: Any,
        *,
        incoming: Any,
        history: Mapping[int, tuple[Any, Any]],
        cos: Any,
        sin: Any,
        source_layer: int = _SOURCE_LAYER,
        target_layer: int = _TARGET_LAYER,
    ) -> None:
        torch, _, _ = _torch()
        if source_layer != _SOURCE_LAYER or target_layer != _TARGET_LAYER:
            raise ReadWeightNativeError("only the qualified L19-to-L22 Qwen2 path is supported")
        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "qwen2":
            raise ReadWeightNativeError("model must use the Qwen2 model family")
        if getattr(config, "hidden_act", None) != "silu":
            raise ReadWeightNativeError("native functional MLP profile requires Qwen2 SiLU activation")
        if getattr(model, "training", True):
            raise ReadWeightNativeError("native-eager replay requires an eval-mode model")
        if getattr(config, "use_sliding_window", False):
            raise ReadWeightNativeError("sliding-window Qwen requires a separate mask profile")
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) <= _TARGET_LAYER:
            raise ReadWeightNativeError("model must expose Qwen2 layers 19 through 22")
        hidden_size = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", None) or hidden_size // heads)
        if heads <= 0 or kv_heads <= 0 or head_dim <= 0 or heads % kv_heads:
            raise ReadWeightNativeError("Qwen2 attention geometry is invalid")
        if float(config.rms_norm_eps) <= 0 or not math.isfinite(float(config.rms_norm_eps)):
            raise ReadWeightNativeError("Qwen2 RMSNorm epsilon is invalid")

        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        if model_dtype != torch.bfloat16:
            raise ReadWeightNativeError("native-eager replay requires BF16 model weights")
        incoming = _finite_tensor(incoming, "incoming", torch=torch, shape=(1, 1, hidden_size),
                                  dtype=torch.bfloat16, device=model_device).detach().clone()
        if not isinstance(history, Mapping) or set(history) != set(_HISTORY_LAYERS):
            raise ReadWeightNativeError("history must contain exactly layers 20, 21, and 22")
        checked_history: dict[int, tuple[Any, Any]] = {}
        row_count = None
        for index in _HISTORY_LAYERS:
            pair = history[index]
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ReadWeightNativeError(f"history layer {index} must contain key and value")
            key, value = pair
            if not isinstance(key, torch.Tensor) or key.ndim != 4 or key.shape[0] != 1:
                raise ReadWeightNativeError(f"history layer {index} key shape differs")
            expected_tail = (kv_heads, key.shape[-2], head_dim)
            if tuple(key.shape[1:]) != expected_tail:
                raise ReadWeightNativeError(f"history layer {index} key geometry differs")
            if row_count is None:
                row_count = int(key.shape[-2])
            if int(key.shape[-2]) != row_count:
                raise ReadWeightNativeError("history layers have different committed row counts")
            shape = (1, kv_heads, row_count, head_dim)
            checked_history[index] = (
                _finite_tensor(key, f"history layer {index} key", torch=torch, shape=shape,
                               dtype=torch.bfloat16, device=model_device).detach().clone(),
                _finite_tensor(value, f"history layer {index} value", torch=torch, shape=shape,
                               dtype=torch.bfloat16, device=model_device).detach().clone(),
            )

        if not row_count:
            raise ReadWeightNativeError("at least one committed KV row is required")
        if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
            raise ReadWeightNativeError("cos and sin must be tensors")
        rope_shape = (1, 1, head_dim)
        self.cos = _finite_tensor(cos, "cos", torch=torch, shape=rope_shape,
                                  device=model_device).detach().clone()
        self.sin = _finite_tensor(sin, "sin", torch=torch, shape=rope_shape,
                                  dtype=self.cos.dtype, device=model_device).detach().clone()
        if self.cos.dtype not in (torch.float32, torch.bfloat16):
            raise ReadWeightNativeError("RoPE values must use the qualified FP32 or BF16 dtype")

        self.model = model
        self.incoming = incoming
        self.history = checked_history
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.device = model_device
        self.hidden_size = hidden_size
        self.query_heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self._config_identity = _config_identity(config)
        self._config_object_id = id(config)
        self._model_object_id = id(model)
        self._parameters = _module_parameters(model)
        self._parameter_signature = _parameter_signature(self._parameters)
        self._validate_profile(torch)
        self._execution_signature = _live_execution_signature(model)
        self.identity = _digest_identity(
            self._config_identity, self._parameter_signature, self._execution_signature
        )

    @property
    def cost(self) -> dict[str, Any]:
        """Dense work performed by each evaluation; no speed claim is implied."""
        return {
            "profile": _NATIVE_PROFILE,
            "logical_dense_linears_per_evaluate": self.logical_dense_linears,
            "source_mlp_linears": 3,
            "intermediate_and_target_block_linears": 21,
            "attention_blocks": 3,
            "native_module_forwards": 0,
            "scope": "Per-state functional BF16 evaluation; weights, attention reductions, finite guards and tape writes are paid.",
        }

    def _validate_profile(self, torch: Any) -> None:
        for index in range(_SOURCE_LAYER, _TARGET_LAYER + 1):
            layer = self.model.model.layers[index]
            attention = layer.self_attn
            if layer.training or attention.training or layer.mlp.training:
                raise ReadWeightNativeError("native-eager replay requires eval-mode path modules")
            if getattr(attention, "sliding_window", None) is not None:
                raise ReadWeightNativeError("sliding-window attention is outside the native-eager profile")
            if getattr(attention, "q_norm", None) is not None or getattr(attention, "k_norm", None) is not None:
                raise ReadWeightNativeError("Qwen3 query/key normalization is outside the Qwen2 profile")
            if getattr(attention, "num_key_value_groups", None) != self.query_heads // self.kv_heads:
                raise ReadWeightNativeError(f"layer {index} grouped-query geometry differs")
            projections = (
                (attention.q_proj, self.query_heads * self.head_dim),
                (attention.k_proj, self.kv_heads * self.head_dim),
                (attention.v_proj, self.kv_heads * self.head_dim),
            )
            if (getattr(attention, "head_dim", None) != self.head_dim
                    or any(module.in_features != self.hidden_size or module.out_features != width
                           for module, width in projections[:3])
                    or attention.o_proj.in_features != self.query_heads * self.head_dim
                    or attention.o_proj.out_features != self.hidden_size):
                raise ReadWeightNativeError(f"layer {index} projection geometry differs from config")
            if not math.isfinite(float(attention.scaling)) or float(attention.scaling) <= 0:
                raise ReadWeightNativeError(f"layer {index} attention scale is invalid")
        for name, parameter in self._parameters:
            if parameter.dtype != torch.bfloat16 or parameter.device != self.device:
                raise ReadWeightNativeError(f"native parameter profile differs at {name}")
            if not bool(torch.isfinite(parameter).all()):
                raise ReadWeightNativeError(f"native parameter is non-finite at {name}")

    def _assert_identity(self) -> None:
        config = getattr(self.model, "config", None)
        if (id(self.model) != self._model_object_id or id(config) != self._config_object_id
                or _config_identity(config) != self._config_identity):
            raise ReadWeightNativeError("model identity or numerical config changed after construction")
        try:
            live_parameters = _module_parameters(self.model)
            live_parameter_signature = _parameter_signature(live_parameters)
            live_execution_signature = _live_execution_signature(self.model)
        except Exception as exc:
            raise ReadWeightNativeError("model module topology changed after construction") from exc
        if live_parameter_signature != self._parameter_signature:
            raise ReadWeightNativeError("model weights changed after construction")
        if live_execution_signature != self._execution_signature:
            raise ReadWeightNativeError("model module topology or execution policy changed after construction")

    def _linear_attention(self, index: int, hidden: Any, *, torch: Any, functional: Any,
                          modeling: Any, value_edit: Mapping[str, Any] | None = None):
        layer = self.model.model.layers[index]
        attention = layer.self_attn
        shape = (1, 1, -1, self.head_dim)
        normalized = _rms_norm(hidden, layer.input_layernorm, torch=torch)
        query = _linear(functional, attention.q_proj, normalized).view(shape).transpose(1, 2).contiguous()
        key = _linear(functional, attention.k_proj, normalized).view(shape).transpose(1, 2)
        value = _linear(functional, attention.v_proj, normalized).view(shape).transpose(1, 2)
        query, key = modeling.apply_rotary_pos_emb(query, key, self.cos, self.sin)
        query = query.contiguous()
        old_key, old_value = self.history[index]
        key_bank = torch.cat((old_key, key), dim=2)
        value_bank = torch.cat((old_value, value), dim=2)
        if value_edit is not None and index == _TARGET_LAYER:
            head, row, scale = self._parse_value_edit(value_edit, old_value.shape[2])
            value_bank = value_bank.clone()
            value_bank[:, head, row, :] = value_bank[:, head, row, :] * scale
        scores, weights, output = _attention(
            query, key_bank, value_bank, attention,
            torch=torch, functional=functional, modeling=modeling,
        )
        return query, key, value, key_bank, value_bank, scores, weights, output

    def _parse_value_edit(self, value_edit: Mapping[str, Any], old_rows: int):
        if not isinstance(value_edit, Mapping) or set(value_edit) != {"kv_head", "source_row", "scale"}:
            raise ReadWeightNativeError("value_edit must contain kv_head, source_row, and scale")
        head, row, scale = value_edit["kv_head"], value_edit["source_row"], value_edit["scale"]
        if type(head) is not int or not 0 <= head < self.kv_heads:
            raise ReadWeightNativeError("value_edit kv_head is outside the historical bank")
        if type(row) is not int or not 0 <= row < old_rows:
            raise ReadWeightNativeError("value_edit source_row must address an L22 historical row")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
            raise ReadWeightNativeError("value_edit scale must be finite")
        if not 0.0 <= float(scale) <= 1.0:
            raise ReadWeightNativeError("value_edit scale must be in [0, 1]")
        return head, row, float(scale)

    def evaluate(self, attention19: Any, *, value_edit: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run one native-profile L19-attention-to-L22 transition tape."""
        torch, functional, modeling = _torch()
        self._assert_identity()
        source_attention = _finite_tensor(
            attention19, "attention19", torch=torch, shape=(1, 1, self.hidden_size),
            dtype=torch.bfloat16, device=self.device,
        ).detach().clone()
        if value_edit is not None:
            # Validate before doing dense work, even though it is applied only
            # at the final target block.
            self._parse_value_edit(value_edit, int(self.history[_TARGET_LAYER][1].shape[2]))

        with torch.inference_mode():
            tape: dict[str, Any] = {
                "input19": self.incoming.clone(),
                "attention19": source_attention,
            }
            hidden = self.incoming + source_attention
            mlp19 = _mlp(
                _rms_norm(hidden, self.model.model.layers[_SOURCE_LAYER].post_attention_layernorm,
                          torch=torch),
                self.model.model.layers[_SOURCE_LAYER].mlp,
                torch=torch, functional=functional,
            )
            hidden = hidden + mlp19
            tape.update(mlp19=mlp19, exit19=hidden)
            for index in (20, 21, 22):
                tape[f"input{index}"] = hidden
                (query, current_key, current_value, key_bank, value_bank, scores, weights,
                 attention_output) = self._linear_attention(
                    index, hidden, torch=torch, functional=functional, modeling=modeling,
                    value_edit=value_edit,
                )
                residual = hidden + attention_output
                mlp_output = _mlp(
                    _rms_norm(residual, self.model.model.layers[index].post_attention_layernorm,
                              torch=torch),
                    self.model.model.layers[index].mlp,
                    torch=torch, functional=functional,
                )
                hidden = residual + mlp_output
                tape[f"attention{index}"] = attention_output
                tape[f"mlp{index}"] = mlp_output
                tape[f"exit{index}"] = hidden
                tape[f"current_k{index}"] = current_key
                tape[f"current_v{index}"] = current_value
                tape[f"q{index}"] = query
                tape[f"k{index}"] = key_bank
                tape[f"v{index}"] = value_bank
                if index == _TARGET_LAYER:
                    tape["scores22"] = scores
                    tape["weights22"] = weights

        for name, value in tape.items():
            if isinstance(value, torch.Tensor) and (value.numel() == 0 or not bool(torch.isfinite(value).all())):
                raise ReadWeightNativeError(f"native replay produced non-finite {name}")
        tape["cost"] = self.cost
        tape["identity"] = self.identity
        tape["native_profile"] = _NATIVE_PROFILE
        return tape


__all__ = ["NativeQwenReadTransport", "ReadWeightNativeError"]

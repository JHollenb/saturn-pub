"""Functional Qwen3 prefix arithmetic for a prospective FLUX prompt cohort.

The caller supplies token IDs, an explicit padding mask, a native source-bound
attention contract, and already-bound parameter tensors.  This module performs
the embedding and Qwen3 blocks 0..26 directly with Torch tensor primitives.  It
never tokenizes a prompt, reads a checkpoint, calls a model/module forward, or
refreshes a native hidden state.  The caller must run the returned forecast
before any native comparison and retain every miss.

The implementation follows the installed Qwen3 graph used by the prospective
cohort: FP32 RMS statistics with the native cast/weight order, per-head Q/K
RMSNorm, grouped-query attention, default Qwen RoPE (base from config), causal
plus padding masking, residuals, and SwiGLU.  Attention dispatch and GQA are
explicit inputs rather than assumptions.  In particular, a masked SDPA call
uses ``repeat_kv`` while an unmasked SDPA call may use native ``enable_gqa``;
the caller must bind whichever policy the native runtime actually selected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

DEFAULT_PREFIX_LAYERS = 27
DEFAULT_SELECTED_HIDDEN_STATES = (9, 18, 27)
QWEN_LAYER_PARAMETER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


class QwenPrefixError(ValueError):
    """Raised when a prospective Qwen prefix contract is incomplete."""


class ModuleForwardForbidden(RuntimeError):
    """Raised if a functional forecast enters any guarded native module."""


@dataclass(frozen=True)
class AttentionDispatch:
    """Native attention dispatch bound by the caller before a forecast.

    ``backend`` is the actual Transformers attention implementation (currently
    ``eager`` or ``sdpa`` in the qualified cohort).  ``gqa_policy`` records the
    actual policy selected for this mask/device: ``repeat_kv`` or
    ``enable_gqa``.  The source is required so a future execution receipt can
    identify the inspected native dispatch code instead of silently assuming a
    matching class name.
    """

    backend: str
    gqa_policy: str
    source: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AttentionDispatch":
        try:
            backend = str(value["backend"])
            gqa_policy = str(value["gqa_policy"])
            source = str(value["source"])
        except (KeyError, TypeError) as error:
            raise QwenPrefixError(
                "attention_dispatch must bind backend, gqa_policy, and source"
            ) from error
        if backend not in {"eager", "sdpa"}:
            raise QwenPrefixError(f"unsupported native Qwen attention backend: {backend!r}")
        if gqa_policy not in {"repeat_kv", "enable_gqa"}:
            raise QwenPrefixError(f"unsupported native Qwen GQA policy: {gqa_policy!r}")
        if not source:
            raise QwenPrefixError("attention dispatch source binding must be nonempty")
        if backend == "eager" and gqa_policy != "repeat_kv":
            raise QwenPrefixError("Qwen eager attention always uses repeat_kv")
        return cls(backend=backend, gqa_policy=gqa_policy, source=source)

    def to_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "gqa_policy": self.gqa_policy,
            "source": self.source,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QwenPrefixError(message)


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_int(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise QwenPrefixError(f"Qwen config field {name!r} must be an integer") from error
    if result <= 0:
        raise QwenPrefixError(f"Qwen config field {name!r} must be positive")
    return result


def _rope_parameters(config: Any) -> Mapping[str, Any]:
    parameters = _config_value(config, "rope_parameters", None)
    if parameters is None:
        parameters = _config_value(config, "rope_scaling", None)
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise QwenPrefixError("Qwen rope_parameters must be a mapping")
    return parameters


def _native_profile(config: Any, prefix_layers: int) -> dict[str, Any]:
    hidden_size = _as_int(_config_value(config, "hidden_size"), "hidden_size")
    intermediate_size = _as_int(_config_value(config, "intermediate_size"), "intermediate_size")
    num_heads = _as_int(_config_value(config, "num_attention_heads"), "num_attention_heads")
    num_kv_heads = _as_int(
        _config_value(config, "num_key_value_heads", num_heads), "num_key_value_heads"
    )
    head_dim = _as_int(
        _config_value(config, "head_dim", hidden_size // num_heads), "head_dim"
    )
    vocab_size = _as_int(_config_value(config, "vocab_size"), "vocab_size")
    max_positions = _as_int(
        _config_value(config, "max_position_embeddings"), "max_position_embeddings"
    )
    full_layers = _as_int(
        _config_value(config, "num_hidden_layers", prefix_layers), "num_hidden_layers"
    )
    _require(prefix_layers <= full_layers, "prefix_layers exceeds the native Qwen layer count")
    # Qwen permits the residual width and projected attention width to differ.
    # The FLUX encoder is 2560-wide with 32*128 = 4096 attention channels; an
    # equality check here would reject the actual checkpoint before any
    # parameter-shape validation.  Keep the two widths explicit instead.
    attention_width = num_heads * head_dim
    _require(num_heads % num_kv_heads == 0, "Qwen attention heads must divide KV heads")
    _require(head_dim % 2 == 0, "Qwen RoPE head_dim must be even")
    rope = _rope_parameters(config)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    _require(rope_type == "default", "only default Qwen RoPE is qualified by this helper")
    rope_theta = rope.get("rope_theta", _config_value(config, "rope_theta", None))
    _require(rope_theta is not None and float(rope_theta) > 0, "Qwen rope_theta must be positive")
    _require(
        float(rope_theta) == 1_000_000.0,
        "the qualified FLUX Qwen prefix requires default RoPE base 1e6",
    )
    rms_eps = float(_config_value(config, "rms_norm_eps", 1e-6))
    _require(rms_eps > 0, "Qwen rms_norm_eps must be positive")
    hidden_act = str(_config_value(config, "hidden_act", "silu"))
    _require(hidden_act == "silu", "only native Qwen SwiGLU/SILU is qualified")
    layer_types = _config_value(config, "layer_types", None)
    if layer_types is not None:
        _require(
            len(layer_types) >= prefix_layers
            and all(str(layer_types[index]) == "full_attention" for index in range(prefix_layers)),
            "Qwen prefix contains unsupported sliding-attention layers",
        )
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "num_key_value_groups": num_heads // num_kv_heads,
        "attention_width": attention_width,
        "head_dim": head_dim,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_positions,
        "num_hidden_layers": full_layers,
        "prefix_layers": prefix_layers,
        "rope_theta": float(rope_theta),
        "rms_norm_eps": rms_eps,
    }


def bind_prospective_tokenized_input(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    tokenizer_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind token IDs as a new tokenizer/parser stage without calling a tokenizer.

    A future raw-prompt cohort must provide this record after its explicitly
    pinned tokenizer and chat-template stage.  The helper deliberately rejects
    a historical binding so a newly pinned template cannot repair old evidence.
    """

    _require(isinstance(input_ids, Tensor) and input_ids.ndim == 2, "input_ids must be rank-2")
    _require(
        isinstance(attention_mask, Tensor)
        and attention_mask.ndim == 2
        and attention_mask.shape == input_ids.shape,
        "attention_mask must be rank-2 and match input_ids",
    )
    _require(
        isinstance(tokenizer_binding, Mapping), "tokenizer_binding must be an explicit mapping"
    )
    _require(
        str(tokenizer_binding.get("cohort", "")) == "prospective-new",
        "Qwen tokenizer binding is only valid for a new prospective cohort",
    )
    _require(
        str(tokenizer_binding.get("parser_stage", "")) == "tokenizer",
        "tokenizer binding must identify the tokenizer parser stage",
    )
    _require(
        bool(tokenizer_binding.get("implementation_pinned", False)),
        "tokenizer implementation must be pinned before a prospective forecast",
    )
    return {
        "input_ids": input_ids.detach().clone(),
        "attention_mask": attention_mask.detach().clone(),
        "tokenizer_binding": dict(tokenizer_binding),
        "historical_repair": False,
    }


@contextmanager
def module_entry_guard(modules: Iterable[torch.nn.Module]):
    """Reject module forwards while a functional forecast is being formed."""

    if isinstance(modules, torch.nn.Module):
        modules = (modules,)
    unique: list[torch.nn.Module] = []
    seen: set[int] = set()
    for component in modules:
        _require(isinstance(component, torch.nn.Module), "module_entry_guard requires nn.Module values")
        for module in component.modules():
            if id(module) not in seen:
                seen.add(id(module))
                unique.append(module)
    entries: list[dict[str, str]] = []

    def forbidden(module: torch.nn.Module, _args: tuple[Any, ...]) -> None:
        entry = {"type": type(module).__name__, "module_id": str(id(module))}
        entries.append(entry)
        raise ModuleForwardForbidden(
            "functional Qwen prefix entered native module " + entry["type"]
        )

    handles = [module.register_forward_pre_hook(forbidden, prepend=True) for module in unique]
    try:
        yield entries
    finally:
        for handle in handles:
            handle.remove()


def _mask_for_attention(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    backend: str,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor | None, bool, dict[str, Any]]:
    """Build the same causal/padding mask shape consumed by native Qwen."""

    padding = None
    if attention_mask is not None:
        _require(
            isinstance(attention_mask, Tensor)
            and attention_mask.shape == (batch_size, sequence_length),
            "attention_mask must be [batch, sequence]",
        )
        if attention_mask.dtype != torch.bool:
            _require(bool(torch.isfinite(attention_mask).all()), "attention_mask contains nonfinite values")
            _require(
                bool(((attention_mask == 0) | (attention_mask == 1)).all()),
                "attention_mask must contain only 0/1 values",
            )
        padding = attention_mask.to(device=device, dtype=torch.bool)

    causal = torch.tril(torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool))
    allowed = causal.view(1, 1, sequence_length, sequence_length).expand(batch_size, -1, -1, -1)
    if padding is not None:
        allowed = allowed & padding[:, None, None, :]
    has_padding = padding is not None and not bool(padding.all())

    # Transformers' SDPA mask builder intentionally returns None for an
    # unpadded full prefill so native SDPA can use is_causal=True.  Eager always
    # receives the additive 4-D mask.  This choice is part of the dispatch
    # contract and is recorded in the result profile.
    if backend == "sdpa" and not has_padding:
        return None, True, {
            "kind": "native-causal-flag",
            "shape": None,
            "dtype": None,
            "has_padding": False,
        }
    if backend == "sdpa":
        return allowed, False, {
            "kind": "boolean-causal-padding",
            "shape": list(allowed.shape),
            "dtype": str(allowed.dtype),
            "has_padding": has_padding,
        }
    minimum = torch.finfo(dtype).min
    additive = torch.where(allowed, torch.zeros((), device=device, dtype=dtype), minimum)
    return additive, False, {
        "kind": "additive-causal-padding",
        "shape": list(additive.shape),
        "dtype": str(additive.dtype),
        "has_padding": has_padding,
    }


def _rms_norm(value: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    input_dtype = value.dtype
    work = value.to(torch.float32)
    variance = work.pow(2).mean(-1, keepdim=True)
    work = work * torch.rsqrt(variance + epsilon)
    return weight * work.to(input_dtype)


def _repeat_kv(value: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return value
    batch, heads, sequence, head_dim = value.shape
    return value[:, :, None, :, :].expand(batch, heads, groups, sequence, head_dim).reshape(
        batch, heads * groups, sequence, head_dim
    )


def _attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None,
    *,
    is_causal: bool,
    scale: float,
    groups: int,
    dispatch: AttentionDispatch,
) -> Tensor:
    if dispatch.gqa_policy == "repeat_kv":
        key = _repeat_kv(key, groups)
        value = _repeat_kv(value, groups)
        gqa_kwargs: dict[str, Any] = {}
    else:
        _require(dispatch.backend == "sdpa", "enable_gqa is only an SDPA policy")
        _require(mask is None, "native SDPA enable_gqa is incompatible with a padding mask")
        gqa_kwargs = {"enable_gqa": True}

    if dispatch.backend == "eager":
        scores = torch.matmul(query, key.transpose(2, 3)) * scale
        if mask is not None:
            scores = scores + mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(probabilities, value).transpose(1, 2).contiguous()

    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        **gqa_kwargs,
    ).transpose(1, 2).contiguous()


def _linear(value: Tensor, weight: Tensor) -> Tensor:
    return F.linear(value, weight, bias=None)


def _check_parameter(parameters: Mapping[str, Tensor], name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    _require(name in parameters, f"missing bound Qwen parameter: {name}")
    value = parameters[name]
    _require(isinstance(value, Tensor), f"bound Qwen parameter is not a tensor: {name}")
    _require(tuple(value.shape) == shape, f"Qwen parameter shape differs for {name}: {tuple(value.shape)} != {shape}")
    _require(value.dtype == dtype, f"Qwen parameter dtype differs for {name}: {value.dtype} != {dtype}")
    return value


def functional_qwen_prefix(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    parameters: Mapping[str, Tensor],
    config: Any,
    *,
    attention_dispatch: Mapping[str, Any],
    rope_buffer_policy: Mapping[str, Any] | None = None,
    prefix_layers: int = DEFAULT_PREFIX_LAYERS,
    selected_hidden_states: Sequence[int] = DEFAULT_SELECTED_HIDDEN_STATES,
) -> dict[str, Any]:
    """Compute the Qwen3 embedding and prefix blocks without native forwards.

    ``parameters`` is a read-only mapping of bound tensor views keyed by the
    exact native names (``model.embed_tokens.weight`` and
    ``model.layers.<index>.*``).  The function accepts a full model config but
    consumes only layers ``0..prefix_layers-1``.  The returned hidden-state
    index is native-style: index 0 is the embedding and index ``n`` is the
    output after block ``n-1``.  The default selected 9/18/27 states therefore
    produce a final conditioning width of 3*hidden_size (7680 for the
    qualified 2560-wide Qwen encoder).
    """

    _require(isinstance(input_ids, Tensor) and input_ids.ndim == 2, "input_ids must be rank-2")
    _require(input_ids.shape[0] == 1, "the qualified Qwen prefix is batch-one")
    _require(input_ids.dtype in {torch.int32, torch.int64}, "input_ids must be integer IDs")
    _require(isinstance(parameters, Mapping), "parameters must be a read-only named tensor mapping")
    _require(prefix_layers > 0, "prefix_layers must be positive")
    profile = _native_profile(config, prefix_layers)
    selected = tuple(int(index) for index in selected_hidden_states)
    _require(selected and tuple(sorted(set(selected))) == selected, "selected hidden states must be sorted and unique")
    _require(all(1 <= index <= prefix_layers for index in selected), "selected hidden state is outside the prefix")
    dispatch = AttentionDispatch.from_mapping(attention_dispatch)
    batch_size, sequence_length = input_ids.shape
    _require(sequence_length > 0, "Qwen prefix requires a nonempty sequence")
    _require(sequence_length <= profile["max_position_embeddings"], "sequence exceeds Qwen max positions")
    embedding_name = "model.embed_tokens.weight"
    embedding_weight = parameters.get(embedding_name)
    _require(isinstance(embedding_weight, Tensor), f"missing bound Qwen parameter: {embedding_name}")
    _require(
        tuple(embedding_weight.shape) == (profile["vocab_size"], profile["hidden_size"]),
        "Qwen embedding shape differs from the bound config",
    )
    _require(
        embedding_weight.dtype == torch.bfloat16,
        "the qualified Qwen prefix requires BF16 bound weights",
    )
    dtype = embedding_weight.dtype
    _require(input_ids.device == embedding_weight.device, "input IDs and Qwen weights must share a device")
    _require(
        bool((input_ids >= 0).all()) and bool((input_ids < profile["vocab_size"]).all()),
        "input IDs escape the bound Qwen vocabulary",
    )
    if attention_mask is not None:
        _require(attention_mask.device == input_ids.device, "attention mask and input IDs must share a device")
    mask, is_causal, mask_profile = _mask_for_attention(
        attention_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        backend=dispatch.backend,
        dtype=dtype,
        device=input_ids.device,
    )
    if dispatch.gqa_policy == "enable_gqa":
        _require(mask is None and is_causal, "enable_gqa requires the native unmasked causal path")
    if (
        dispatch.backend == "sdpa"
        and mask is None
        and profile["num_key_value_groups"] > 1
    ):
        _require(
            dispatch.gqa_policy == "enable_gqa",
            "unmasked native SDPA GQA policy must explicitly be enable_gqa",
        )

    position_ids = torch.arange(sequence_length, device=input_ids.device, dtype=torch.long).unsqueeze(0)
    hidden = F.embedding(input_ids, embedding_weight)
    hidden_states: dict[int, Tensor] = {0: hidden}
    num_heads = profile["num_attention_heads"]
    num_kv_heads = profile["num_key_value_heads"]
    head_dim = profile["head_dim"]
    hidden_size = profile["hidden_size"]
    groups = profile["num_key_value_groups"]
    scale = head_dim**-0.5
    theta = profile["rope_theta"]
    # Qwen3 registers inv_freq as a non-persistent buffer.  The historical
    # ``from_pretrained(dtype=BF16).to(cuda)`` path constructs it on CPU as
    # float32 and moves it without a dtype cast; a separate BF16 placement is
    # retained as the v1 tiny-test option.  Bind this construction/placement
    # policy explicitly; never read observed native rotary values as forecast
    # input.  The optional argument preserves the v1 helper API while making
    # the selected source-bound policy visible in the returned profile.
    rope_policy = dict(
        {
            "init_device": "cpu",
            "init_dtype": "float32",
            "placement_dtype": "bfloat16",
            "source": "transformers-qwen3-rotary-embedding-v1",
        }
        if rope_buffer_policy is None
        else rope_buffer_policy
    )
    _require(
        rope_policy.get("init_device") == "cpu",
        "qualified Qwen RoPE requires the source-bound CPU initialization device",
    )
    _require(
        rope_policy.get("init_dtype") == "float32",
        "qualified Qwen RoPE requires float32 inverse-frequency initialization",
    )
    _require(rope_policy.get("placement_dtype") in {"float32", "bfloat16"}, "Qwen RoPE placement dtype is unsupported")
    _require(bool(rope_policy.get("source")), "Qwen RoPE policy source binding must be nonempty")
    _require(
        dtype == torch.bfloat16,
        "Qwen RoPE BF16 placement policy requires BF16 embedding weights",
    )
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device="cpu", dtype=torch.float32)
            / head_dim
        )
    )
    if rope_policy["placement_dtype"] == "bfloat16":
        inv_freq = inv_freq.to(dtype=dtype).float()
    else:
        inv_freq = inv_freq.to(dtype=torch.float32)
    inv_freq = inv_freq.to(device=input_ids.device)
    frequencies = (
        inv_freq[None, :, None].expand(batch_size, -1, -1)
        @ position_ids[:, None, :].to(torch.float32)
    ).transpose(1, 2)
    rotary = torch.cat((frequencies, frequencies), dim=-1)
    cos = rotary.cos().to(dtype)
    sin = rotary.sin().to(dtype)

    for layer_index in range(prefix_layers):
        root = f"model.layers.{layer_index}."
        input_norm = _check_parameter(
            parameters,
            root + "input_layernorm.weight",
            (hidden_size,),
            dtype,
        )
        post_norm = _check_parameter(
            parameters,
            root + "post_attention_layernorm.weight",
            (hidden_size,),
            dtype,
        )
        q_norm = _check_parameter(parameters, root + "self_attn.q_norm.weight", (head_dim,), dtype)
        k_norm = _check_parameter(parameters, root + "self_attn.k_norm.weight", (head_dim,), dtype)
        q_weight = _check_parameter(
            parameters, root + "self_attn.q_proj.weight", (num_heads * head_dim, hidden_size), dtype
        )
        k_weight = _check_parameter(
            parameters, root + "self_attn.k_proj.weight", (num_kv_heads * head_dim, hidden_size), dtype
        )
        v_weight = _check_parameter(
            parameters, root + "self_attn.v_proj.weight", (num_kv_heads * head_dim, hidden_size), dtype
        )
        o_weight = _check_parameter(
            parameters, root + "self_attn.o_proj.weight", (hidden_size, num_heads * head_dim), dtype
        )
        gate_weight = _check_parameter(
            parameters, root + "mlp.gate_proj.weight", (profile["intermediate_size"], hidden_size), dtype
        )
        up_weight = _check_parameter(
            parameters, root + "mlp.up_proj.weight", (profile["intermediate_size"], hidden_size), dtype
        )
        down_weight = _check_parameter(
            parameters,
            root + "mlp.down_proj.weight",
            (hidden_size, profile["intermediate_size"]),
            dtype,
        )

        residual = hidden
        normalized = _rms_norm(hidden, input_norm, profile["rms_norm_eps"])
        q = _linear(normalized, q_weight).view(batch_size, sequence_length, num_heads, head_dim)
        k = _linear(normalized, k_weight).view(batch_size, sequence_length, num_kv_heads, head_dim)
        v = _linear(normalized, v_weight).view(batch_size, sequence_length, num_kv_heads, head_dim)
        q = _rms_norm(q, q_norm, profile["rms_norm_eps"]).transpose(1, 2)
        k = _rms_norm(k, k_norm, profile["rms_norm_eps"]).transpose(1, 2)
        v = v.transpose(1, 2)
        q = (q * cos.unsqueeze(1)) + (_rotate_half(q) * sin.unsqueeze(1))
        k = (k * cos.unsqueeze(1)) + (_rotate_half(k) * sin.unsqueeze(1))
        attended = _attention(
            q,
            k,
            v,
            mask,
            is_causal=is_causal,
            scale=scale,
            groups=groups,
            dispatch=dispatch,
        )
        hidden = residual + _linear(attended.reshape(batch_size, sequence_length, -1), o_weight)

        residual = hidden
        normalized = _rms_norm(hidden, post_norm, profile["rms_norm_eps"])
        gate = _linear(normalized, gate_weight)
        up = _linear(normalized, up_weight)
        hidden = residual + _linear(F.silu(gate) * up, down_weight)
        hidden_states[layer_index + 1] = hidden

    selected_states = [hidden_states[index] for index in selected]
    conditioning = torch.cat(selected_states, dim=-1)
    return {
        "input_ids": input_ids.detach().clone(),
        "attention_mask": None if attention_mask is None else attention_mask.detach().clone(),
        "position_ids": position_ids,
        "hidden_states": hidden_states,
        "ports": {
            "embedding": hidden_states[0],
            "selected": {index: hidden_states[index] for index in selected},
            "conditioning": conditioning,
        },
        "conditioning": conditioning,
        "profile": {
            "model_family": "Qwen3",
            "prefix_layers": prefix_layers,
            "selected_hidden_states": list(selected),
            "hidden_size": hidden_size,
            "attention_width": profile["attention_width"],
            "conditioning_width": int(conditioning.shape[-1]),
            "dtype": str(dtype),
            "qualification_scope": "tiny synthetic CPU parity only; original CUDA pending",
            "position_source": "config-derived arange, no observed activation context",
            "rope_source": "config default base1e6 with source-bound CPU-float32 -> BF16 placement reconstructed",
            "rope_buffer_policy": rope_policy,
            "attention_dispatch": dispatch.to_dict(),
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "num_key_value_groups": groups,
            "gqa_policy": dispatch.gqa_policy,
            "mask": mask_profile,
            "native_module_forwards": 0,
        },
    }


def _rotate_half(value: Tensor) -> Tensor:
    midpoint = value.shape[-1] // 2
    return torch.cat((-value[..., midpoint:], value[..., :midpoint]), dim=-1)


__all__ = [
    "AttentionDispatch",
    "DEFAULT_PREFIX_LAYERS",
    "DEFAULT_SELECTED_HIDDEN_STATES",
    "ModuleForwardForbidden",
    "QwenPrefixError",
    "bind_prospective_tokenized_input",
    "functional_qwen_prefix",
    "module_entry_guard",
]

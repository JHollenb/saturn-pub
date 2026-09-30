"""Batch-one, greedy native decoder machine for registered HF autoregressive families.

One generic adapter drives many families (gpt2, gpt_neox/Pythia, phi, llama, mistral,
mixtral, gemma-v1, qwen2, qwen3) through their *native* HF decoder layers, one token and
one layer at a time, over an externally held key/value cache. The state grammar, surface
manifest, execution/numeric contract, frozen-model guards, ``validate``/``write`` rules,
``layer`` vs ``operation`` granularity, and ``generate``/``native_logits`` semantics match
``adapters/qwen.py``. Family differences (absolute vs rotary position, parallel residual,
Gemma's embedding normalizer, MoE layers, sliding-window attention, module paths) are
declared per family and refused fail-closed when a config is outside the registered rule.
"""

from __future__ import annotations

import inspect
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import transformers
from transformers import DynamicCache

from ..contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from ..core import Adapter, Session
from ..values import clone, digest, identity


def _stable_configuration(value: Any, *, key: str = "") -> Any:
    if isinstance(value, Mapping):
        return {name: _stable_configuration(item, key=name) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        items = [_stable_configuration(item) for item in value]
        return sorted(items) if key == "_use_default_values" else items
    return value


def _tensor_guards(module: Any, *, prefix: str = "") -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            f"{prefix}{name}",
            id(value),
            value._version,
            value.data_ptr(),
            value.untyped_storage().data_ptr(),
        )
        for name, value in (*module.named_parameters(), *module.named_buffers())
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_common(config: Any) -> None:
    """Fail-closed refusals shared by every registered dense family."""
    if getattr(config, "quantization_config", None):
        raise ValueError("quantized checkpoints are not supported; load a raw-float model")
    if getattr(config, "auto_map", None):
        raise ValueError("remote-code (auto_map) checkpoints are not supported")


def _rope_type(config: Any) -> str:
    """Transformers 5.x stores RoPE as a dict (``rope_type`` defaults to ``'default'``)."""
    rope = getattr(config, "rope_parameters", None)
    if rope is None:
        rope = getattr(config, "rope_scaling", None)
    if isinstance(rope, Mapping):
        return rope.get("rope_type") or rope.get("type") or "default"
    return "default" if rope is None else "unknown"


def _partial_rotary_factor(config: Any) -> float | None:
    rope = getattr(config, "rope_parameters", None)
    if isinstance(rope, Mapping) and rope.get("partial_rotary_factor") is not None:
        return float(rope["partial_rotary_factor"])
    factor = getattr(config, "partial_rotary_factor", None)
    return None if factor is None else float(factor)


# --- per-family config validators (ported semantic rules from the private registry) ----


def _validate_gpt2(config: Any) -> None:
    _reject_common(config)
    if config.activation_function != "gelu_new":
        raise ValueError("registered GPT-2 semantics require gelu_new (tanh-approximated) MLP")
    if not getattr(config, "scale_attn_weights", True):
        raise ValueError("registered GPT-2 attention requires head-dimension scaling")
    if getattr(config, "scale_attn_by_inverse_layer_idx", False):
        raise ValueError("inverse-layer GPT-2 attention scaling is not registered")
    if getattr(config, "reorder_and_upcast_attn", False):
        raise ValueError("reordered/upcast GPT-2 attention is not registered")
    if getattr(config, "add_cross_attention", False):
        raise ValueError("cross-attention GPT-2 topology is not registered")
    _require(config.n_embd % config.n_head == 0, "GPT-2 n_embd must be divisible by n_head")


def _validate_gpt_neox(config: Any) -> None:
    _reject_common(config)
    if config.hidden_act != "gelu":
        raise ValueError("registered GPT-NeoX semantics require exact erf gelu")
    kv = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    if kv != config.num_attention_heads:
        raise ValueError("registered GPT-NeoX fused QKV requires one K/V head per query head")
    if getattr(config, "tie_word_embeddings", False):
        raise ValueError("registered GPT-NeoX/Pythia topology requires untied embeddings")
    if _rope_type(config) != "default":
        raise ValueError("registered GPT-NeoX semantics certify default (unscaled) RoPE only")
    pct = getattr(config, "rotary_pct", 0.25)
    if not (0.0 < float(pct) <= 1.0):
        raise ValueError("GPT-NeoX rotary_pct must be in (0, 1]")


def _validate_phi(config: Any) -> None:
    _reject_common(config)
    if config.hidden_act != "gelu_new":
        raise ValueError("registered Phi semantics require gelu_new")
    if getattr(config, "tie_word_embeddings", False):
        raise ValueError("registered Phi topology requires untied lexical matrices")
    if getattr(config, "qk_layernorm", False):
        raise ValueError("Phi qk_layernorm is not registered")
    if _rope_type(config) != "default":
        raise ValueError("registered Phi semantics require unscaled default RoPE")
    factor = _partial_rotary_factor(config)
    factor = 0.5 if factor is None else factor
    if not (0.0 < factor <= 1.0):
        raise ValueError("Phi partial_rotary_factor must be in (0, 1]")
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None and head_dim != config.hidden_size // config.num_attention_heads:
        raise ValueError("Phi head_dim must equal hidden_size / num_attention_heads")


def _reject_scaled_rope(config: Any) -> None:
    if _rope_type(config) != "default":
        raise ValueError("this adapter certifies default (unscaled) RoPE only")
    factor = _partial_rotary_factor(config)
    if factor is not None and factor != 1.0:
        raise ValueError("partial rotary dimensions require a distinct semantic variant")


def _validate_llama(config: Any) -> None:
    _reject_common(config)
    _reject_scaled_rope(config)
    tp = getattr(config, "pretraining_tp", 1)
    if tp not in (None, 1):
        raise ValueError("llama pretraining_tp must be 1; partitioned arithmetic is not registered")
    if getattr(config, "rope_interleaved", False):
        raise ValueError("interleaved llama rotary coordinates are not registered")


def _validate_mistral(config: Any) -> None:
    _reject_common(config)
    _reject_scaled_rope(config)


def _validate_mixtral(config: Any) -> None:
    _reject_common(config)
    _reject_scaled_rope(config)
    if getattr(config, "attention_bias", False):
        raise ValueError("registered Mixtral attention is biasless")
    experts = config.num_local_experts
    top_k = config.num_experts_per_tok
    if not (isinstance(experts, int) and experts >= 2):
        raise ValueError("Mixtral requires at least two routed experts")
    if not (isinstance(top_k, int) and 0 < top_k < experts):
        raise ValueError("Mixtral top-k must be a positive count smaller than the expert count")
    if getattr(config, "output_router_logits", False):
        raise ValueError("router-logit outputs require a separate declared output space")


def _validate_gemma(config: Any) -> None:
    _reject_common(config)
    _reject_scaled_rope(config)
    act = getattr(config, "hidden_activation", None) or config.hidden_act
    if config.hidden_act != "gelu_pytorch_tanh" or act != "gelu_pytorch_tanh":
        raise ValueError("registered Gemma-1 semantics require gelu_pytorch_tanh")
    if getattr(config, "attention_bias", False):
        raise ValueError("registered Gemma-1 attention is biasless")
    if getattr(config, "mlp_bias", False):
        raise ValueError("registered Gemma-1 MLP is biasless")


def _validate_qwen2(config: Any) -> None:
    _reject_common(config)
    _reject_scaled_rope(config)
    if getattr(config, "use_sliding_window", False):
        raise ValueError("sliding-window Qwen2 attention is not registered")


def _validate_qwen3(config: Any) -> None:
    _reject_common(config)
    _reject_scaled_rope(config)
    if getattr(config, "use_sliding_window", False):
        raise ValueError("sliding-window Qwen3 attention is not registered")


@dataclass(frozen=True)
class _FamilySpec:
    model_type: str
    model_cls: type
    config_cls: type
    backbone: str  # attribute on the CausalLM giving the backbone Model
    layers: str  # attribute on the backbone giving the decoder ModuleList
    embed: str  # attribute on the backbone giving the token embedding module
    final_norm: str  # attribute on the backbone giving the final norm module
    position: str  # "rotary" | "absolute"
    cache_kwarg: str  # "past_key_values" | "layer_past"
    validate: Callable[[Any], None]
    abs_pos: str | None = None  # attribute for absolute position embedding (gpt2 wpe)
    moe: bool = False
    tiny_config: Mapping[str, Any] = field(default_factory=dict)


def _import_families() -> dict[str, _FamilySpec]:
    from transformers import (
        GemmaConfig,
        GemmaForCausalLM,
        GPT2Config,
        GPT2LMHeadModel,
        GPTNeoXConfig,
        GPTNeoXForCausalLM,
        LlamaConfig,
        LlamaForCausalLM,
        MistralConfig,
        MistralForCausalLM,
        MixtralConfig,
        MixtralForCausalLM,
        PhiConfig,
        PhiForCausalLM,
        Qwen2Config,
        Qwen2ForCausalLM,
        Qwen3Config,
        Qwen3ForCausalLM,
    )

    common = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        attn_implementation="eager",
    )
    specs = [
        _FamilySpec(
            "gpt2",
            GPT2LMHeadModel,
            GPT2Config,
            "transformer",
            "h",
            "wte",
            "ln_f",
            "absolute",
            "past_key_values",
            _validate_gpt2,
            abs_pos="wpe",
            tiny_config=dict(
                vocab_size=64,
                n_embd=32,
                n_layer=2,
                n_head=4,
                n_positions=128,
                n_inner=64,
                activation_function="gelu_new",
                attn_implementation="eager",
            ),
        ),
        _FamilySpec(
            "gpt_neox",
            GPTNeoXForCausalLM,
            GPTNeoXConfig,
            "gpt_neox",
            "layers",
            "embed_in",
            "final_layer_norm",
            "rotary",
            "layer_past",
            _validate_gpt_neox,
            tiny_config=dict(
                **common, rotary_pct=0.25, use_parallel_residual=True, hidden_act="gelu"
            ),
        ),
        _FamilySpec(
            "phi",
            PhiForCausalLM,
            PhiConfig,
            "model",
            "layers",
            "embed_tokens",
            "final_layernorm",
            "rotary",
            "past_key_values",
            _validate_phi,
            tiny_config=dict(**common, partial_rotary_factor=0.5, hidden_act="gelu_new"),
        ),
        _FamilySpec(
            "llama",
            LlamaForCausalLM,
            LlamaConfig,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_llama,
            tiny_config=dict(**common, num_key_value_heads=2, head_dim=8),
        ),
        _FamilySpec(
            "mistral",
            MistralForCausalLM,
            MistralConfig,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_mistral,
            tiny_config=dict(**common, num_key_value_heads=2, head_dim=8, sliding_window=128),
        ),
        _FamilySpec(
            "mixtral",
            MixtralForCausalLM,
            MixtralConfig,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_mixtral,
            moe=True,
            tiny_config=dict(
                **common,
                num_key_value_heads=2,
                head_dim=8,
                num_local_experts=4,
                num_experts_per_tok=2,
            ),
        ),
        _FamilySpec(
            "gemma",
            GemmaForCausalLM,
            GemmaConfig,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_gemma,
            tiny_config=dict(
                **common,
                num_key_value_heads=2,
                head_dim=8,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
            ),
        ),
        _FamilySpec(
            "qwen2",
            Qwen2ForCausalLM,
            Qwen2Config,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_qwen2,
            tiny_config=dict(**common, num_key_value_heads=2, head_dim=8),
        ),
        _FamilySpec(
            "qwen3",
            Qwen3ForCausalLM,
            Qwen3Config,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_qwen3,
            tiny_config=dict(**common, num_key_value_heads=2, head_dim=8),
        ),
    ]
    return {spec.model_type: spec for spec in specs}


_FAMILIES: dict[str, _FamilySpec] | None = None


def families() -> dict[str, _FamilySpec]:
    global _FAMILIES
    if _FAMILIES is None:
        _FAMILIES = _import_families()
    return _FAMILIES


SUPPORTED_FAMILIES = (
    "gpt2",
    "gpt_neox",
    "phi",
    "llama",
    "mistral",
    "mixtral",
    "gemma",
    "qwen2",
    "qwen3",
)


class DecoderAdapter(Adapter):
    """Generic native decoder adapter. Same session grammar as ``QwenAdapter``."""

    def __init__(self, model: Any, *, granularity: str = "layer"):
        model_type = getattr(model.config, "model_type", None)
        spec = families().get(model_type)
        if spec is None:
            raise ValueError(
                f"unregistered decoder model_type: {model_type!r}; "
                f"supported families are {', '.join(SUPPORTED_FAMILIES)}"
            )
        if not isinstance(model, spec.model_cls):
            raise ValueError(f"{model_type} adapter requires a {spec.model_cls.__name__}")
        spec.validate(model.config)
        if getattr(model.config, "_attn_implementation", "eager") != "eager":
            raise ValueError("DecoderAdapter requires attn_implementation='eager'")
        if granularity not in {"layer", "operation"}:
            raise ValueError("granularity must be 'layer' or 'operation'")
        self.spec = spec
        self.model = model.eval()
        self.granularity = granularity
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        config = model.config
        self.backbone = getattr(model, spec.backbone)
        self._layer_modules = getattr(self.backbone, spec.layers)
        self.layers = len(self._layer_modules)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_positions = config.max_position_embeddings
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", None) or self.num_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.sliding_window = (
            getattr(config, "sliding_window", None) if spec.model_type == "mistral" else None
        )
        self.rotary = getattr(self.backbone, "rotary_emb", None)
        if spec.position == "rotary" and self.rotary is None:
            raise ValueError(f"{spec.model_type} backbone is missing rotary_emb")
        if spec.model_type == "qwen3" and not hasattr(self._layer_modules[0].self_attn, "q_norm"):
            raise ValueError("registered Qwen3 semantics require attention qk-norm")
        stable = _stable_configuration(config.to_dict())
        self._configuration_fingerprint = digest(stable)
        self.model_identity = identity(model, stable)
        self._frozen_versions = _tensor_guards(model)
        self._layer_forward_kwargs = _accepts_kwargs(self._layer_modules[0].forward)
        parity = "bounded-logits-moe" if spec.moe else "bounded-logits"
        self.execution = {
            "family": spec.model_type,
            "adapter": "native-decoder-v1",
            "granularity": granularity,
            "attention": "eager",
            "position": spec.position,
            "dtype": str(self.dtype),
            "device": {"type": self.device.type, "index": self.device.index},
            "batch": {"size": 1, "padding": False},
            "environment_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
            },
            "model_program": self._model_program(),
            "kernel": {"attention": "transformers-eager", "cache": "DynamicCache"},
            "numeric_scope": {
                "inference_mode": True,
                "autocast": False,
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "torch_num_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "tensor_layout": "state-slot-bound-stride",
            },
            "sampler": "greedy",
            "parity": parity,
        }

    def _model_program(self) -> str:
        embed = "embed+abs_pos" if self.spec.position == "absolute" else "embed"
        return f"{embed}->decoder_layer[*]->final_norm->lm_head->greedy_argmax->token_commit"

    @classmethod
    def tiny(cls, family: str, seed: int = 7, *, granularity: str = "layer") -> DecoderAdapter:
        spec = families().get(family)
        if spec is None:
            raise ValueError(
                f"unknown family {family!r}; supported: {', '.join(SUPPORTED_FAMILIES)}"
            )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            config = spec.config_cls(**dict(spec.tiny_config))
            return cls(spec.model_cls(config), granularity=granularity)

    @classmethod
    def from_pretrained(
        cls, path: str, *, granularity: str = "layer", **kwargs: Any
    ) -> DecoderAdapter:
        from transformers import AutoConfig

        model_type = AutoConfig.from_pretrained(path).model_type
        spec = families().get(model_type)
        if spec is None:
            raise ValueError(
                f"unregistered decoder model_type: {model_type!r}; "
                f"supported families are {', '.join(SUPPORTED_FAMILIES)}"
            )
        model = spec.model_cls.from_pretrained(path, attn_implementation="eager", **kwargs)
        return cls(model, granularity=granularity)

    # --- frozen-model guards (identical policy to QwenAdapter) --------------------------

    def _verify_frozen_model(self) -> None:
        if _tensor_guards(self.model) != self._frozen_versions:
            raise ValueError(
                "model parameters or buffers changed after adapter identity was frozen; "
                "rewrap the model in a new adapter"
            )

    def validate_execution(self) -> None:
        self._verify_frozen_model()
        if (
            digest(_stable_configuration(self.model.config.to_dict()))
            != self._configuration_fingerprint
        ):
            raise ValueError(
                "model configuration changed after adapter identity was frozen; rewrap"
            )
        expected = self.execution["numeric_scope"]
        current = {
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if any(expected[name] != value for name, value in current.items()):
            raise ValueError("live numerical environment drifted from the execution ABI; rewrap")

    # --- external key/value cache -------------------------------------------------------

    def _cache(self, keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> DynamicCache:
        cache = DynamicCache(config=self.model.config)
        for index, (key, value) in enumerate(zip(keys, values)):
            if key.shape[-2]:
                cache.update(key.to(self.device), value.to(self.device), index)
        return cache

    def _kv_shape(self, length: int) -> tuple[int, int, int, int]:
        return (1, self.num_kv_heads, length, self.head_dim)

    def _cache_values(self, cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        empty = self._kv_shape(0)
        keys, values = [], []
        for layer in cache.layers:
            keys.append(
                clone(layer.keys)
                if getattr(layer, "keys", None) is not None
                else torch.empty(empty, device=self.device, dtype=self.dtype)
            )
            values.append(
                clone(layer.values)
                if getattr(layer, "values", None) is not None
                else torch.empty(empty, device=self.device, dtype=self.dtype)
            )
        return keys, values

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        embed_module = getattr(self.backbone, self.spec.embed)
        hidden = embed_module(tokens[:, -1:])
        if self.spec.position == "absolute":
            position = tokens.shape[1] - 1
            positions = torch.tensor([[position]], device=self.device)
            hidden = hidden + getattr(self.backbone, self.spec.abs_pos)(positions)
        return hidden

    def _run_layer(
        self, index: int, hidden: torch.Tensor, cache: DynamicCache, position: int
    ) -> torch.Tensor:
        positions = torch.tensor([[position]], device=self.device)
        kwargs: dict[str, Any] = {self.spec.cache_kwarg: cache, "use_cache": True}
        if self.spec.position == "rotary":
            kwargs["position_ids"] = positions
            kwargs["position_embeddings"] = self.rotary(hidden, positions)
            kwargs["attention_mask"] = None
        else:
            kwargs["attention_mask"] = None
        if not self._layer_forward_kwargs:
            signature = inspect.signature(self._layer_modules[index].forward)
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        result = self._layer_modules[index](hidden, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    # --- session / grammar --------------------------------------------------------------

    @torch.inference_mode()
    def session(self, token_ids: Sequence[int]) -> Session:
        self.validate_execution()
        tokens = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        if tokens.shape[-1] < 1:
            raise ValueError("at least one pending token is required")
        self._guard_window(tokens.shape[-1])
        cache = DynamicCache(config=self.model.config)
        if tokens.shape[-1] > 1:
            cache = self.model(tokens[:, :-1], use_cache=True).past_key_values
        keys, values = self._cache_values(cache)
        return Session(
            self,
            {
                "tokens": tokens,
                "keys": keys,
                "values": values,
                "layer": 0,
                "phase": "embed",
                "hidden": None,
                "logits": None,
                "sampled_token": None,
            },
        )

    def _guard_window(self, count: int) -> None:
        if self.sliding_window is not None and count > self.sliding_window:
            raise ValueError(
                "context exceeds the sliding-window horizon; windowed attention is not registered"
            )

    def boundary(self, state: Mapping[str, Any]) -> str:
        if self.granularity == "layer":
            if state["hidden"] is None:
                return "token"
            return f"layer:{state['layer']}"
        phase = state["phase"]
        if phase == "layer":
            return f"token:{state['tokens'].shape[1] - 1}/layer:{state['layer']}"
        return f"token:{state['tokens'].shape[1] - 1}/{phase}"

    def point(self, state: Mapping[str, Any]) -> ExecutionPoint:
        phase = state["phase"]
        index = state["layer"] if phase == "layer" else None
        next_operation = {
            "embed": f"{self.spec.backbone}.{self.spec.embed}",
            "normalization": f"{self.spec.backbone}.{self.spec.final_norm}",
            "readout": "lm_head",
            "sample": "greedy_argmax",
            "commit": "token_commit",
        }.get(
            phase,
            (
                f"decoder_layer.{state['layer']}"
                if state["layer"] < self.layers
                else "normalization_readout_sampler_commit"
            ),
        )
        return ExecutionPoint(
            family=self.spec.model_type,
            logical_step=state["tokens"].shape[1] - 1,
            phase=phase,
            operator=phase,
            index=index,
            edge="before",
            next_operation=next_operation,
            local_clock=state["layer"],
        )

    def transition(self, state: Mapping[str, Any]) -> TransitionSpec:
        phase = state["phase"]
        if state["hidden"] is None:
            reads, writes = ("tokens",), ("hidden", "logits", "phase")
        elif state["layer"] < self.layers:
            reads = ("tokens", "hidden", "keys", "values", "layer")
            writes = ("hidden", "keys", "values", "layer", "phase")
        elif self.granularity == "operation":
            reads, writes = {
                "normalization": (("hidden",), ("hidden", "phase")),
                "readout": (("hidden",), ("logits", "phase")),
                "sample": (("logits",), ("sampled_token", "phase")),
                "commit": (
                    ("tokens", "sampled_token"),
                    ("tokens", "hidden", "sampled_token", "layer", "phase"),
                ),
            }[phase]
        else:
            reads = ("hidden", "tokens")
            writes = ("logits", "tokens", "hidden", "layer", "phase")
        return TransitionSpec(
            self.point(state).next_operation,
            reads,
            writes,
            consumer=f"native-{self.spec.model_type}-suffix",
            footprint="declared",
        )

    def surface(self, state: Mapping[str, Any]) -> SurfaceManifest:
        def tensor_schema(value: torch.Tensor | None, *, optional: bool = False) -> dict[str, Any]:
            if value is None:
                return {"kind": "tensor", "optional": optional}
            return {
                "kind": "tensor",
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "optional": optional,
            }

        slots = (
            SlotSpec(
                "tokens",
                role="cursor",
                producer="token_commit",
                consumer="embed",
                schema=tensor_schema(state["tokens"]),
            ),
            SlotSpec(
                "keys",
                role="cache",
                producer="decoder_layer",
                consumer="attention",
                schema={"kind": "tensor_sequence", "length": self.layers},
            ),
            SlotSpec(
                "values",
                role="cache",
                producer="decoder_layer",
                consumer="attention",
                schema={"kind": "tensor_sequence", "length": self.layers},
            ),
            SlotSpec(
                "hidden",
                role="carrier",
                writable=state["hidden"] is not None,
                persistence="authoritative",
                producer="embed_or_layer",
                consumer="native_suffix",
                schema=tensor_schema(state["hidden"], optional=True),
            ),
            SlotSpec(
                "logits",
                role="observation",
                producer="lm_head",
                consumer="greedy_argmax",
                schema=tensor_schema(state["logits"], optional=True),
            ),
            SlotSpec(
                "sampled_token",
                role="decision",
                producer="greedy_argmax",
                consumer="token_commit",
                schema=tensor_schema(state["sampled_token"], optional=True),
            ),
            SlotSpec(
                "layer",
                role="clock",
                producer="decoder_layer",
                consumer="runtime",
                schema={"kind": "integer", "minimum": 0, "maximum": self.layers},
            ),
            SlotSpec(
                "phase",
                role="clock",
                producer="runtime",
                consumer="runtime",
                schema={
                    "kind": "enum",
                    "values": ["embed", "layer", "normalization", "readout", "sample", "commit"],
                },
            ),
        )
        return SurfaceManifest(
            slots=slots,
            state_schema=f"saturn-pub-decoder-{self.spec.model_type}-state-v1",
            consumer=f"native-{self.spec.model_type}-suffix",
            horizon="native-autoregressive-suffix",
        )

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        result = ("tokens",)
        if state["hidden"] is not None:
            result += ("hidden",)
        if state["logits"] is not None:
            result += ("logits",)
        return result

    def validate(self, state: Mapping[str, Any]) -> None:
        self._verify_frozen_model()
        expected_keys = {
            "tokens",
            "keys",
            "values",
            "layer",
            "phase",
            "hidden",
            "logits",
            "sampled_token",
        }
        if set(state) != expected_keys:
            raise ValueError("decoder state schema keys differ from the execution contract")
        tokens = state["tokens"]
        if (
            tokens.ndim != 2
            or tokens.shape[0] != 1
            or tokens.dtype != torch.long
            or tokens.device != self.device
        ):
            raise ValueError("decoder state requires batch-one integer tokens")
        if not 0 < tokens.shape[1] <= self.max_positions:
            raise ValueError("token count outside configured context")
        self._guard_window(tokens.shape[1])
        if (tokens < 0).any() or (tokens >= self.vocab_size).any():
            raise ValueError("token ID outside vocabulary")
        layer = state["layer"]
        if type(layer) is not int or not 0 <= layer <= self.layers:
            raise ValueError("invalid layer cursor")
        phase = state["phase"]
        allowed_phases = {"embed", "layer"}
        if self.granularity == "operation":
            allowed_phases |= {"normalization", "readout", "sample", "commit"}
        if phase not in allowed_phases:
            raise ValueError("invalid decoder operation phase")
        if phase == "embed" and (state["hidden"] is not None or layer != 0):
            raise ValueError("invalid embedding boundary closure")
        if phase != "embed" and state["hidden"] is None:
            raise ValueError("missing hidden carrier")
        if phase == "layer" and layer > self.layers:
            raise ValueError("layer phase exceeds the native layer program")
        if self.granularity == "operation" and phase == "layer" and layer == self.layers:
            raise ValueError("operation mode must expose the normalization boundary")
        if phase in {"normalization", "readout", "sample", "commit"} and layer != self.layers:
            raise ValueError("post-layer phase requires a complete decoder stack")
        expected = tokens.shape[1] - 1
        keys, values = state["keys"], state["values"]
        if len(keys) != self.layers or len(values) != self.layers:
            raise ValueError("missing layer K/V")
        for index, (key, value) in enumerate(zip(keys, values)):
            length = expected + (1 if state["hidden"] is not None and index < layer else 0)
            shape = self._kv_shape(length)
            if tuple(key.shape) != shape or tuple(value.shape) != shape:
                raise ValueError("K/V closure does not match token/layer cursor")
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError("K/V dtype mismatch")
            if key.device != self.device or value.device != self.device:
                raise ValueError("K/V device mismatch")
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise ValueError("non-finite K/V")
        hidden = state["hidden"]
        if hidden is not None and (
            tuple(hidden.shape) != (1, 1, self.hidden_size)
            or hidden.dtype != self.dtype
            or hidden.device != self.device
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("invalid hidden carrier")
        logits = state["logits"]
        if logits is not None and (
            tuple(logits.shape) != (1, self.vocab_size)
            or logits.dtype != self.dtype
            or logits.device != self.device
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("invalid logit observation")
        if phase in {"sample", "commit"} and logits is None:
            raise ValueError("sampling requires the native logit observation")
        sampled = state["sampled_token"]
        if sampled is not None and (
            tuple(sampled.shape) != (1, 1)
            or sampled.dtype != torch.long
            or sampled.device != self.device
            or bool((sampled < 0).any())
            or bool((sampled >= self.vocab_size).any())
        ):
            raise ValueError("invalid sampled token")
        if (phase == "commit") != (sampled is not None):
            raise ValueError("sampled token exists only at the commit boundary")

    def write(
        self, state: Mapping[str, Any], writes: Mapping[str, Any], invalidates: tuple[str, ...]
    ) -> Mapping[str, Any]:
        if "tokens" in writes:
            raise ValueError("token writes require a new prefill; use adapter.session(new_tokens)")
        if "logits" in writes:
            raise ValueError("logits are observations; intervene on hidden before readout")
        if "sampled_token" in writes:
            raise ValueError("sampled tokens are decisions; intervene on logits upstream")
        if invalidates:
            raise ValueError(
                "decoder derived state is recomputed automatically; no manual invalidation"
            )
        return super().write(state, writes, ())

    @torch.inference_mode()
    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate(state)
        out = clone(state)
        tokens = out["tokens"].to(self.device)
        if out["hidden"] is None:
            if tokens.shape[1] >= self.max_positions:
                raise ValueError("context capacity reached")
            out["hidden"] = self._embed(tokens)
            out["logits"] = None
            out["phase"] = "layer"
            return out
        hidden = out["hidden"].to(self.device)
        layer = out["layer"]
        if layer < self.layers:
            cache = self._cache(out["keys"], out["values"])
            out["hidden"] = self._run_layer(layer, hidden, cache, tokens.shape[1] - 1)
            out["keys"], out["values"] = self._cache_values(cache)
            out["layer"] = layer + 1
            if self.granularity == "operation" and out["layer"] == self.layers:
                out["phase"] = "normalization"
        elif self.granularity == "operation" and out["phase"] == "normalization":
            out["hidden"] = getattr(self.backbone, self.spec.final_norm)(hidden)
            out["phase"] = "readout"
        elif self.granularity == "operation" and out["phase"] == "readout":
            out["logits"] = self.model.lm_head(hidden)[:, -1]
            out["phase"] = "sample"
        elif self.granularity == "operation" and out["phase"] == "sample":
            out["sampled_token"] = out["logits"].argmax(-1, keepdim=True)
            out["phase"] = "commit"
        elif self.granularity == "operation" and out["phase"] == "commit":
            out["tokens"] = torch.cat((tokens, out["sampled_token"]), dim=-1)
            out["hidden"] = None
            out["sampled_token"] = None
            out["layer"] = 0
            out["phase"] = "embed"
        else:
            final_norm = getattr(self.backbone, self.spec.final_norm)
            logits = self.model.lm_head(final_norm(hidden))[:, -1]
            out["logits"] = logits
            out["tokens"] = torch.cat((tokens, logits.argmax(-1, keepdim=True)), dim=-1)
            out["hidden"] = None
            out["layer"] = 0
            out["phase"] = "embed"
        return out

    def generate(self, session: Session, tokens: int = 1) -> None:
        if tokens < 1:
            raise ValueError("generation budget must be positive")
        for _ in range(tokens):
            initial = session.read("tokens").shape[1]
            while session.read("tokens").shape[1] == initial:
                session.continue_()

    @torch.inference_mode()
    def native_logits(self, tokens: Sequence[int]) -> torch.Tensor:
        """Uninstrumented full forward used as the numerical comparator."""
        return self.model(torch.tensor([list(tokens)], device=self.device)).logits[:, -1]


def _accepts_kwargs(function: Callable[..., Any]) -> bool:
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(function).parameters.values()
    )

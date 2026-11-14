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
from ..core import Adapter, Session, seal_append
from ..numerics import (
    check_numeric_environment,
    live_numeric_environment,
    measure_accumulation,
    needs_accumulation_measurement,
)
from ..values import clone, digest, identity
from ._residency import BlockResidency


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


def _causal_mask_kwargs(
    create_fn: Callable[..., Any],
    *,
    config: Any,
    hidden: Any,
    cache: Any,
    position_ids: Any,
    cache_position: Any,
) -> dict[str, Any]:
    """Version-robust keyword set for ``transformers`` ``create_causal_mask``.

    The streamed backbone (unlike the resident path) builds the eager attention mask itself,
    and ``create_causal_mask``'s signature differs across transformers majors:

    * transformers 5.x: ``(config, inputs_embeds, attention_mask, past_key_values, position_ids)``
    * transformers 4.x: ``(config, input_embeds, attention_mask, cache_position, past_key_values,
      position_ids)`` -- the embeds keyword loses its ``s`` and ``cache_position`` is required.

    Inspecting the live signature keeps the streamed path bitwise-correct on either major (the
    resident path is unaffected -- it never calls ``create_causal_mask``).
    """
    params = inspect.signature(create_fn).parameters
    kwargs: dict[str, Any] = {
        "config": config,
        "attention_mask": None,
        "past_key_values": cache,
        "position_ids": position_ids,
    }
    kwargs["inputs_embeds" if "inputs_embeds" in params else "input_embeds"] = hidden
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    return kwargs


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


def _validate_gemma2(config: Any) -> None:
    """Gemma-2: GeGLU, soft-capped attention/logits, alternating local/global attention.

    Both logit-softcaps and the sliding window are read from the config and honored by the
    adapter (final-logit softcap applied at readout, attention softcap inside the native
    eager layer). Sliding-window attention alternates with full attention layer by layer;
    the adapter reproduces it exactly while the context stays within one window (full causal
    and windowed causal coincide there) and refuses a context longer than one window.
    """
    _reject_common(config)
    _reject_scaled_rope(config)
    act = getattr(config, "hidden_activation", None) or config.hidden_act
    if act != "gelu_pytorch_tanh":
        raise ValueError("registered Gemma-2 semantics require gelu_pytorch_tanh (GeGLU)")
    if getattr(config, "attention_bias", False):
        raise ValueError("registered Gemma-2 attention is biasless")
    if getattr(config, "mlp_bias", False):
        raise ValueError("registered Gemma-2 MLP is biasless")
    if not getattr(config, "sliding_window", None):
        raise ValueError(
            "registered Gemma-2 requires a sliding_window (alternating local/global attention)"
        )
    if getattr(config, "query_pre_attn_scalar", None) is None:
        raise ValueError("registered Gemma-2 requires query_pre_attn_scalar attention scaling")


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
        Gemma2Config,
        Gemma2ForCausalLM,
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
            "gemma2",
            Gemma2ForCausalLM,
            Gemma2Config,
            "model",
            "layers",
            "embed_tokens",
            "norm",
            "rotary",
            "past_key_values",
            _validate_gemma2,
            tiny_config=dict(
                **common,
                num_key_value_heads=2,
                head_dim=8,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                sliding_window=16,
                query_pre_attn_scalar=8,
                attn_logit_softcapping=4.0,
                final_logit_softcapping=2.0,
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
    "gemma2",
    "qwen2",
    "qwen3",
)


class DecoderAdapter(Adapter):
    """Generic native decoder adapter. Same session grammar as ``QwenAdapter``.

    ``shares_state``: ``advance`` never writes a tensor of its input state in place (the K/V
    tensors the native layer sees are copies made by ``_cache``; the hidden carrier and the
    logits are only read), and its result keeps the input's tensor for every K/V layer the
    transition did not change, so consecutive cuts of a Session share them. Native calls run
    under ``torch.no_grad`` (version counters kept) rather than inference mode.
    """

    shares_state = True

    def __init__(
        self,
        model: Any,
        *,
        granularity: str = "layer",
        residency: str = "resident",
        device: Any = None,
        pin_host: bool = False,
    ):
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
        # Block-streamed residency parks frozen weights in host memory and copies one native
        # module to the execution device at a time; resident keeps the historical zero-overhead
        # path. place() must run before the frozen guard because host pinning reassigns storage.
        if residency == "streamed":
            target = (
                device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._residency = BlockResidency(target, mode="streamed", pin_host=pin_host)
        else:
            target = device if device is not None else next(model.parameters()).device
            self._residency = BlockResidency(target, mode="resident")
        self._residency.place(self.model)
        self.device = self._residency.device
        self.dtype = next(model.parameters()).dtype
        config = model.config
        self.backbone = getattr(model, spec.backbone)
        self._layer_modules = getattr(self.backbone, spec.layers)
        self.layers = len(self._layer_modules)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and len(layer_types) != self.layers:
            raise ValueError(
                f"config.layer_types has {len(layer_types)} entries but the model has "
                f"{self.layers} decoder layers (num_hidden_layers = {config.num_hidden_layers}); "
                f"layer_types must have exactly {self.layers} entries, so trim it together with "
                "num_hidden_layers when truncating a config"
            )
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_positions = config.max_position_embeddings
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", None) or self.num_heads
        self.head_dim = getattr(config, "head_dim", None) or self.hidden_size // self.num_heads
        self.sliding_window = (
            getattr(config, "sliding_window", None)
            if spec.model_type in {"mistral", "gemma2"}
            else None
        )
        # Gemma-2 soft-caps the final logits inside the CausalLM (not the lm_head); reproduce it
        # at readout so the stepped logits match a native full forward. Attention-logit softcap
        # and query_pre_attn_scalar scaling live inside the native eager layer, unchanged.
        self.final_logit_softcapping = (
            getattr(config, "final_logit_softcapping", None)
            if spec.model_type == "gemma2"
            else None
        )
        # Gemma scales token embeddings by sqrt(hidden_size). transformers>=5 folds that scale
        # into a scaled-embedding module (``embed_scale``); older transformers apply it in
        # ``Model.forward`` with a plain ``nn.Embedding``. Detect which, and apply the scale in
        # ``_embed`` only when the embedding module does not already carry it, so the adapter
        # reproduces the native forward on either transformers version.
        self._embed_scale: float | None = None
        if spec.model_type in {"gemma", "gemma2"}:
            embed_module = getattr(self.backbone, spec.embed)
            if getattr(embed_module, "embed_scale", None) is None:
                self._embed_scale = float(config.hidden_size) ** 0.5
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
                # native calls run under torch.no_grad (not inference mode) so that state
                # tensors keep version counters: the shares_state guard needs them
                "inference_mode": False,
                "grad_enabled": False,
                "autocast": False,
                **live_numeric_environment(),
                "tensor_layout": "state-slot-bound-stride",
            },
            "sampler": "greedy",
            "parity": parity,
        }
        if self._residency.mode == "resident" and needs_accumulation_measurement(
            self.device.type, self.dtype
        ):
            # the accumulation precision of native 16-bit matmuls, measured on this device
            # (a handful of linear stages, no forward); bounds read it from the scope
            self.execution["numeric_scope"]["accumulation"] = measure_accumulation(
                self._layer_modules, getattr(model, "lm_head", None)
            )
        # Resident execution keeps the historical contract verbatim; only streamed residency
        # adds a field, so existing resident receipts stay byte-identical.
        if self._residency.mode == "streamed":
            self.execution["residency"] = self._residency.to_dict()
        if spec.model_type == "gemma2":
            self.execution["logit_softcap"] = {
                "final": self.final_logit_softcapping,
                "attention": getattr(config, "attn_logit_softcapping", None),
            }
            self.execution["attention_schedule"] = "alternating-local-global-sliding-window"
            self.execution["sliding_window"] = self.sliding_window

    def _model_program(self) -> str:
        embed = "embed+abs_pos" if self.spec.position == "absolute" else "embed"
        readout = "lm_head"
        if self.spec.model_type == "gemma2" and self.final_logit_softcapping is not None:
            readout = "lm_head->logit_softcap"
        return f"{embed}->decoder_layer[*]->final_norm->{readout}->greedy_argmax->token_commit"

    def _apply_logit_softcap(self, logits: torch.Tensor) -> torch.Tensor:
        """Gemma-2 final-logit soft-cap, applied after ``lm_head`` to match native logits."""
        cap = self.final_logit_softcapping
        if cap is None:
            return logits
        return torch.tanh(logits / cap) * cap

    @classmethod
    def tiny(
        cls,
        family: str,
        seed: int = 7,
        *,
        granularity: str = "layer",
        residency: str = "resident",
        device: Any = None,
        pin_host: bool = False,
    ) -> DecoderAdapter:
        spec = families().get(family)
        if spec is None:
            raise ValueError(
                f"unknown family {family!r}; supported: {', '.join(SUPPORTED_FAMILIES)}"
            )
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            config = spec.config_cls(**dict(spec.tiny_config))
            return cls(
                spec.model_cls(config),
                granularity=granularity,
                residency=residency,
                device=device,
                pin_host=pin_host,
            )

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        granularity: str = "layer",
        residency: str = "resident",
        device: Any = None,
        pin_host: bool = False,
        **kwargs: Any,
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
        return cls(
            model,
            granularity=granularity,
            residency=residency,
            device=device,
            pin_host=pin_host,
        )

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
        check_numeric_environment(self.execution["numeric_scope"])

    # --- external key/value cache -------------------------------------------------------

    def _cache(self, keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> DynamicCache:
        """A native cache holding copies of ``keys``/``values`` (``DynamicCache.update`` copies:
        the native layer and its hooks never see the state's own tensors)."""
        cache = DynamicCache(config=self.model.config)
        for index, (key, value) in enumerate(zip(keys, values)):
            if key.shape[-2]:
                cache.update(key.to(self.device), value.to(self.device), index)
        return cache

    @staticmethod
    def _populated(cache: DynamicCache) -> list[tuple[Any, ...] | None]:
        """Per layer: ``(keys, keys version, values, values version)`` of the tensors ``_cache``
        put in the cache, or None for a layer it left empty (or whose tensors keep no version
        counter, as under inference mode: a write to them could not be told)."""
        out: list[tuple[Any, ...] | None] = []
        for layer in cache.layers:
            key, value = getattr(layer, "keys", None), getattr(layer, "values", None)
            if key is None or value is None or key.is_inference() or value.is_inference():
                out.append(None)
            else:
                out.append((key, key._version, value, value._version))
        return out

    def _kv_shape(self, length: int) -> tuple[int, int, int, int]:
        return (1, self.num_kv_heads, length, self.head_dim)

    @staticmethod
    def _same_layout(a: torch.Tensor, b: torch.Tensor) -> bool:
        return (
            a.shape == b.shape
            and a.stride() == b.stride()
            and a.dtype == b.dtype
            and a.device == b.device
        )

    def _cache_values_shared(
        self,
        cache: DynamicCache,
        populated: Sequence[tuple[Any, ...] | None],
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """``_cache_values`` with the input's own tensors for every layer the transition left
        alone: the cache still holds the very copy ``_cache`` put there, unwritten (same object,
        same version counter), so it is bitwise the input, which is kept instead (same layout
        required). A layer the native code or a cell edit replaced or wrote is taken over."""
        out_keys, out_values = self._cache_values(cache)
        for index, entry in enumerate(populated):
            layer = cache.layers[index]
            current_key = getattr(layer, "keys", None)
            current_value = getattr(layer, "values", None)
            if entry is None:
                if current_key is None and current_value is None:
                    out_keys[index], out_values[index] = keys[index], values[index]
                continue
            key, key_version, value, value_version = entry
            if (
                current_key is key
                and current_value is value
                and key._version == key_version
                and value._version == value_version
                and self._same_layout(key, keys[index])
                and self._same_layout(value, values[index])
            ):
                out_keys[index], out_values[index] = keys[index], values[index]
        for index in range(len(out_keys)):
            for new, old in ((out_keys[index], keys[index]), (out_values[index], values[index])):
                self._declare_append(new, old)
        return out_keys, out_values

    @staticmethod
    def _declare_append(new: torch.Tensor, old: torch.Tensor) -> None:
        """Declare a layer tensor the transition grew by rows (``seal_append``) when its leading
        rows are bitwise the input's, so a cut holds only the appended rows of it."""
        if new is old or new.dim() != 4 or old.dim() != 4:
            return
        rows = old.shape[2]
        if new.shape[2] <= rows or new.shape[:2] != old.shape[:2] or new.shape[3] != old.shape[3]:
            return
        if new.dtype != old.dtype or new.device != old.device:
            return
        if rows and not torch.equal(new.narrow(2, 0, rows), old):
            return  # a cell edit rewrote cached rows: the tensor is held whole
        seal_append(new, old, 2)

    def _cache_values(self, cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The cache's per-layer tensors, taken over without a copy: every one is a fresh
        ``torch.cat`` of the cache update (``_cache`` copies the input layers in), contiguous,
        and the cache object is dropped by every caller."""
        empty = self._kv_shape(0)
        keys, values = [], []
        for layer in cache.layers:
            keys.append(
                layer.keys
                if getattr(layer, "keys", None) is not None
                else torch.empty(empty, device=self.device, dtype=self.dtype)
            )
            values.append(
                layer.values
                if getattr(layer, "values", None) is not None
                else torch.empty(empty, device=self.device, dtype=self.dtype)
            )
        return keys, values

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        run = self._residency.run
        embed_module = getattr(self.backbone, self.spec.embed)
        hidden = run(embed_module, tokens[:, -1:].clone())  # modules never see state tensors
        if self._embed_scale is not None:
            hidden = hidden * torch.tensor(self._embed_scale, dtype=hidden.dtype)
        if self.spec.position == "absolute":
            position = tokens.shape[1] - 1
            positions = torch.tensor([[position]], device=self.device)
            hidden = hidden + run(getattr(self.backbone, self.spec.abs_pos), positions)
        return hidden

    def _layer_kwargs(self, index: int, kwargs: dict[str, Any]) -> dict[str, Any]:
        if self._layer_forward_kwargs:
            return kwargs
        signature = inspect.signature(self._layer_modules[index].forward)
        return {k: v for k, v in kwargs.items() if k in signature.parameters}

    def _run_layer(
        self,
        index: int,
        hidden: torch.Tensor,
        cache: DynamicCache,
        position: int,
        ports: Any = None,
    ) -> torch.Tensor:
        positions = torch.tensor([[position]], device=self.device)
        kwargs: dict[str, Any] = {self.spec.cache_kwarg: cache, "use_cache": True}
        if self.spec.position == "rotary":
            kwargs["position_ids"] = positions
            kwargs["position_embeddings"] = self._residency.run(self.rotary, hidden, positions)
            kwargs["attention_mask"] = None
        else:
            kwargs["attention_mask"] = None
        kwargs = self._layer_kwargs(index, kwargs)
        module = self._layer_modules[index]
        if ports is None:
            result = self._residency.run(module, hidden, **kwargs)
        else:
            with ports.installed(self, index, module):
                result = self._residency.run(module, hidden, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    def _streamed_backbone(self, tokens: torch.Tensor) -> tuple[torch.Tensor, DynamicCache]:
        """Block-streamed replica of the native backbone forward over a full token span.

        Streams the embedding, rotary, and every decoder layer one module at a time while the
        model's own ``create_causal_mask`` builds the eager attention mask. Because the copies
        are byte-preserving and ``functional_call`` substitutes without mutating the module, the
        returned hidden state and key/value cache are bitwise-identical to a resident
        ``model(tokens)`` forward on the same device (proven by tests and measured on real
        weights). Used only in streamed residency, where the whole model cannot fit the device.
        """
        from transformers.masking_utils import create_causal_mask

        run = self._residency.run
        length = tokens.shape[1]
        hidden = run(getattr(self.backbone, self.spec.embed), tokens)
        if self._embed_scale is not None:
            hidden = hidden * torch.tensor(self._embed_scale, dtype=hidden.dtype)
        position_ids = torch.arange(length, device=self.device).unsqueeze(0)
        if self.spec.position == "absolute":
            hidden = hidden + run(getattr(self.backbone, self.spec.abs_pos), position_ids)
        cache = DynamicCache(config=self.model.config)
        cache_position = torch.arange(length, device=self.device)
        mask = create_causal_mask(
            **_causal_mask_kwargs(
                create_causal_mask,
                config=self.model.config,
                hidden=hidden,
                cache=cache,
                position_ids=position_ids,
                cache_position=cache_position,
            )
        )
        kwargs: dict[str, Any] = {
            self.spec.cache_kwarg: cache,
            "use_cache": True,
            "attention_mask": mask,
        }
        if self.spec.position == "rotary":
            kwargs["position_ids"] = position_ids
            kwargs["position_embeddings"] = run(self.rotary, hidden, position_ids)
        for index in range(self.layers):
            result = run(self._layer_modules[index], hidden, **self._layer_kwargs(index, kwargs))
            hidden = result[0] if isinstance(result, tuple) else result
        return hidden, cache

    # --- session / grammar --------------------------------------------------------------

    @torch.no_grad()
    def session(self, token_ids: Sequence[int], *, prefill: int | None = None) -> Session:
        """A session over the prompt ``token_ids``: the first ``prefill`` rows are computed in
        one batched native call (default: all but the last), the next row is pending.

        With ``prefill`` below ``len(token_ids) - 1`` the remaining prompt rows are stepped one
        row at a time by the same transitions as generated rows: the state then holds the whole
        prompt in the ``prompt`` slot and the ``sample`` transition of a prompt row writes the
        prompt's next token (``forced_token``) instead of the argmax. The default procedure has
        no ``prompt`` slot, so its states, cuts and fingerprints are those of before the slot
        existed. The split is part of the cut: a batched and a stepped row can differ in their
        cached K/V bits, and the root cut of each procedure is a different state."""
        self.validate_execution()
        tokens = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        if tokens.shape[-1] < 1:
            raise ValueError("at least one pending token is required")
        count = tokens.shape[-1]
        if prefill is None:
            prefill = count - 1
        if isinstance(prefill, bool) or not isinstance(prefill, int) or not 0 <= prefill < count:
            raise ValueError(f"prefill must be an integer in 0..{count - 1} (the prompt has "
                             f"{count} tokens; the row after the prefilled ones is pending)")
        self._guard_window(count)
        prompt = tokens
        tokens = tokens[:, : prefill + 1].clone()
        cache = DynamicCache(config=self.model.config)
        if prefill > 0:
            if self._residency.mode == "streamed":
                cache = self._streamed_backbone(tokens[:, :-1])[1]
            else:
                cache = self.model(tokens[:, :-1], use_cache=True).past_key_values
        keys, values = self._cache_values(cache)
        state = {
            "tokens": tokens,
            "keys": keys,
            "values": values,
            "layer": 0,
            "phase": "embed",
            "hidden": None,
            "logits": None,
            "sampled_token": None,
        }
        if prefill < count - 1:
            state["prompt"] = prompt
        return Session(self, state)

    @staticmethod
    def _forced(state: Mapping[str, Any]) -> torch.Tensor | None:
        """The prompt token the pending row's ``sample`` writes (``[1, 1]``), or None when the
        row is the prompt's last (or past it) and ``sample`` is the native argmax."""
        prompt = state.get("prompt")
        if prompt is None:
            return None
        row = state["tokens"].shape[1]
        if row >= prompt.shape[1]:
            return None
        return prompt[:, row : row + 1].clone()

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
            "sample": "greedy_argmax" if self._forced(state) is None else "forced_token",
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
                "sample": (("logits",) if self._forced(state) is None else ("tokens", "prompt"),
                           ("sampled_token", "phase")),
                "commit": (
                    ("tokens", "sampled_token"),
                    ("tokens", "hidden", "sampled_token", "layer", "phase"),
                ),
            }[phase]
        else:
            reads = ("hidden", "tokens") + (("prompt",) if "prompt" in state else ())
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
        if "prompt" in state:  # a stepped prompt: the tokens its sample transitions force
            slots += (
                SlotSpec(
                    "prompt",
                    role="decision",
                    producer="prefill",
                    consumer="forced_token",
                    schema=tensor_schema(state["prompt"]),
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
        self.validate_state(state)

    def _all_finite(self, tensors: Sequence[Any]) -> bool | None:
        """Whether every one of ``tensors`` is finite, with one host sync; None when one is not
        a tensor on the execution device (the per-tensor checks then decide, and name it)."""
        if not all(isinstance(t, torch.Tensor) and t.device == self.device for t in tensors):
            return None
        if not tensors:
            return True
        return bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())

    def validate_state(self, state: Mapping[str, Any]) -> None:
        """``validate`` without the frozen-model check, which ``validate_execution`` makes.

        Finiteness of the K/V, hidden and logits tensors is one fused check (one host sync);
        only when it fails do the per-tensor checks run, so the refusal names the same tensor
        in the same order as before.
        """
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
        if "prompt" in state:
            expected_keys.add("prompt")
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
        if bool(((tokens < 0) | (tokens >= self.vocab_size)).any()):
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
        finite = self._all_finite(
            [*keys, *values, *(state[n] for n in ("hidden", "logits") if state[n] is not None)]
        )
        for index, (key, value) in enumerate(zip(keys, values)):
            length = expected + (1 if state["hidden"] is not None and index < layer else 0)
            shape = self._kv_shape(length)
            if tuple(key.shape) != shape or tuple(value.shape) != shape:
                raise ValueError("K/V closure does not match token/layer cursor")
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError("K/V dtype mismatch")
            if key.device != self.device or value.device != self.device:
                raise ValueError("K/V device mismatch")
            if not finite and (not torch.isfinite(key).all() or not torch.isfinite(value).all()):
                raise ValueError("non-finite K/V")
        hidden = state["hidden"]
        if hidden is not None and (
            tuple(hidden.shape) != (1, 1, self.hidden_size)
            or hidden.dtype != self.dtype
            or hidden.device != self.device
            or (not finite and not torch.isfinite(hidden).all())
        ):
            raise ValueError("invalid hidden carrier")
        logits = state["logits"]
        if logits is not None and (
            tuple(logits.shape) != (1, self.vocab_size)
            or logits.dtype != self.dtype
            or logits.device != self.device
            or (not finite and not torch.isfinite(logits).all())
        ):
            raise ValueError("invalid logit observation")
        if phase in {"sample", "commit"} and logits is None:
            raise ValueError("sampling requires the native logit observation")
        sampled = state["sampled_token"]
        if sampled is not None and (
            tuple(sampled.shape) != (1, 1)
            or sampled.dtype != torch.long
            or sampled.device != self.device
            or bool(((sampled < 0) | (sampled >= self.vocab_size)).any())
        ):
            raise ValueError("invalid sampled token")
        if (phase == "commit") != (sampled is not None):
            raise ValueError("sampled token exists only at the commit boundary")
        prompt = state.get("prompt")
        if prompt is not None:
            if (
                prompt.ndim != 2
                or prompt.shape[0] != 1
                or prompt.dtype != torch.long
                or prompt.device != self.device
                or prompt.shape[1] < 2
            ):
                raise ValueError("a stepped prompt is batch-one integer tokens, two or more")
            shared = min(tokens.shape[1], prompt.shape[1])
            if not torch.equal(tokens[:, :shared], prompt[:, :shared]):
                raise ValueError("the tokens differ from the stepped prompt they follow")
            forced = self._forced(state)
            if sampled is not None and forced is not None and not torch.equal(sampled, forced):
                raise ValueError("a prompt row's sampled token must be the prompt's next token")

    def read(self, state: Mapping[str, Any], address: str) -> Any:
        """Read a listed address, or one of the cache/decision slots that exist in the state.

        ``keys`` and ``values`` (per-layer cache tensors) are always present; ``sampled_token``
        only at the operation-granularity commit boundary; ``prompt`` only in a session whose
        prompt rows are stepped (``session(..., prefill=K)``). They are readable here but are
        not in ``addresses()``, so frames, cuts and receipts are unchanged.
        """
        if address in ("keys", "values"):
            return clone(state[address])
        if address == "prompt":
            if state.get("prompt") is None:
                raise ValueError(f"unsupported address at {self.boundary(state)}: {address} "
                                 "(this session prefilled its prompt)")
            return clone(state[address])
        if address == "sampled_token":
            if state["sampled_token"] is None:
                raise ValueError(f"unsupported address at {self.boundary(state)}: {address}")
            return clone(state[address])
        return super().read(state, address)

    def read_part(self, state: Mapping[str, Any], address: str, index: Any) -> Any:
        """One layer of ``keys`` / ``values`` (``index``), copying that layer only."""
        if address in ("keys", "values"):
            return clone(state[address][index])
        return super().read_part(state, address, index)

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

    def ports(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        """Qualified module names a port controller may hook in the pending decoder layer."""
        if state["hidden"] is None or state["phase"] != "layer" or state["layer"] >= self.layers:
            return ()
        prefix = f"{self.spec.backbone}.{self.spec.layers}.{state['layer']}"
        return tuple(
            f"{prefix}.{name}" if name else prefix
            for name, _ in self._layer_modules[state["layer"]].named_modules()
        )

    def ports_ahead(self, state: Mapping[str, Any], steps: int) -> tuple[Any, ...]:
        """The decoder layer index each of the next ``steps`` transitions runs (``None`` for the
        embedding and the post-layer operations), derived from the state without running."""
        bare = state["hidden"] is None
        layer = state["layer"]
        post = ("normalization", "readout", "sample", "commit")
        phase = state["phase"] if state["phase"] in post else None
        out: list[int | None] = []
        for _ in range(steps):
            if bare:
                out.append(None)
                bare = False
            elif layer < self.layers:
                out.append(layer)
                layer += 1
                if layer == self.layers:
                    phase = post[0]
            else:
                out.append(None)
                if self.granularity == "operation" and phase != post[-1]:
                    phase = post[post.index(phase) + 1] if phase in post else post[1]
                else:
                    bare, layer, phase = True, 0, None
        return tuple(out)

    @torch.no_grad()
    def advance(
        self, state: Mapping[str, Any], *, ports: Any = None, validated: bool = False
    ) -> Mapping[str, Any]:
        """The next native transition of ``state``, which is left unchanged.

        The result shares the tensors of the slots the transition does not write with
        ``state`` (they are never written in place) and holds new tensors for the rest; in a
        layer transition every K/V layer other than the ones the layer (or a cell edit) wrote
        is the input's own tensor (``shares_state``).
        ``validated``: the caller validated ``state`` and the executor (``validate_execution``)
        in this same call, so the entry validation is not repeated.
        """
        if not validated:
            self.validate(state)
        out = {**state, "keys": list(state["keys"]), "values": list(state["values"])}
        tokens = out["tokens"].to(self.device)
        if out["hidden"] is None:
            if tokens.shape[1] >= self.max_positions:
                raise ValueError("context capacity reached")
            out["hidden"] = self._embed(tokens)
            out["logits"] = None
            out["phase"] = "layer"
            return out
        # native modules and their hooks get a private copy of the carrier (as they got a copy of
        # the whole state before shares_state): a hook writing its input cannot reach the state
        hidden = out["hidden"].to(self.device, copy=True)
        layer = out["layer"]
        if layer < self.layers:
            cache = self._cache(out["keys"], out["values"])
            populated = self._populated(cache)
            out["hidden"] = self._run_layer(layer, hidden, cache, tokens.shape[1] - 1, ports)
            out["keys"], out["values"] = self._cache_values_shared(
                cache, populated, out["keys"], out["values"]
            )
            out["layer"] = layer + 1
            if self.granularity == "operation" and out["layer"] == self.layers:
                out["phase"] = "normalization"
        elif self.granularity == "operation" and out["phase"] == "normalization":
            out["hidden"] = self._residency.run(
                getattr(self.backbone, self.spec.final_norm), hidden
            )
            out["phase"] = "readout"
        elif self.granularity == "operation" and out["phase"] == "readout":
            out["logits"] = self._apply_logit_softcap(
                self._residency.run(self.model.lm_head, hidden)[:, -1]
            )
            out["phase"] = "sample"
        elif self.granularity == "operation" and out["phase"] == "sample":
            forced = self._forced(out)  # a stepped prompt row: the prompt's next token
            out["sampled_token"] = (out["logits"].argmax(-1, keepdim=True) if forced is None
                                    else forced)
            out["phase"] = "commit"
        elif self.granularity == "operation" and out["phase"] == "commit":
            out["tokens"] = torch.cat((tokens, out["sampled_token"]), dim=-1)
            seal_append(out["tokens"], tokens, 1)  # cuts hold the new token only
            out["hidden"] = None
            out["sampled_token"] = None
            out["layer"] = 0
            out["phase"] = "embed"
        else:
            final_norm = getattr(self.backbone, self.spec.final_norm)
            run = self._residency.run
            logits = self._apply_logit_softcap(
                run(self.model.lm_head, run(final_norm, hidden))[:, -1]
            )
            out["logits"] = logits
            forced = self._forced(out)
            chosen = logits.argmax(-1, keepdim=True) if forced is None else forced
            out["tokens"] = torch.cat((tokens, chosen), dim=-1)
            seal_append(out["tokens"], tokens, 1)  # cuts hold the new token only
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
        """Uninstrumented full forward used as the numerical comparator.

        In streamed residency the full forward is itself block-streamed (the whole model may
        not fit the device), reproducing the resident ``model(tokens)`` logits bitwise.
        """
        sequence = torch.tensor([list(tokens)], device=self.device)
        if self._residency.mode == "streamed":
            hidden, _ = self._streamed_backbone(sequence)
            final_norm = getattr(self.backbone, self.spec.final_norm)
            run = self._residency.run
            return self._apply_logit_softcap(
                run(self.model.lm_head, run(final_norm, hidden))[:, -1]
            )
        return self.model(sequence).logits[:, -1]


def _accepts_kwargs(function: Callable[..., Any]) -> bool:
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(function).parameters.values()
    )

"""Batch-one, greedy Qwen2 token/layer machine using native HF decoder layers."""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import transformers
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

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


class QwenAdapter(Adapter):
    def __init__(self, model: Qwen2ForCausalLM, *, granularity: str = "layer"):
        if model.config.model_type != "qwen2":
            raise ValueError("v1 supports Qwen2/Qwen2.5 full-attention models")
        if getattr(model.config, "use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        if model.config._attn_implementation != "eager":
            raise ValueError("QwenAdapter requires attn_implementation='eager'")
        if granularity not in {"layer", "operation"}:
            raise ValueError("granularity must be 'layer' or 'operation'")
        self.model = model.eval()
        self.granularity = granularity
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.layers = len(model.model.layers)
        configuration = _stable_configuration(model.config.to_dict())
        self._configuration_fingerprint = digest(configuration)
        self.model_identity = identity(model, configuration)
        self._frozen_versions = _tensor_guards(model)
        self.execution = {
            "family": "qwen2",
            "adapter": "native-boundary-v2",
            "granularity": granularity,
            "attention": "eager",
            "dtype": str(self.dtype),
            "device": {"type": self.device.type, "index": self.device.index},
            "batch": {"size": 1, "padding": False},
            "environment_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
            },
            "model_program": (
                "embed->decoder_layer[*]->rms_norm->lm_head->greedy_argmax->token_commit"
            ),
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
            "parity": "bounded-logits",
        }

    @classmethod
    def tiny(cls, seed: int = 7, *, granularity: str = "layer") -> QwenAdapter:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            config = Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                max_position_embeddings=128,
                attn_implementation="eager",
            )
            return cls(Qwen2ForCausalLM(config), granularity=granularity)

    @classmethod
    def from_pretrained(
        cls, path: str, *, granularity: str = "layer", **kwargs: Any
    ) -> QwenAdapter:
        return cls(
            Qwen2ForCausalLM.from_pretrained(path, attn_implementation="eager", **kwargs),
            granularity=granularity,
        )

    def _verify_frozen_model(self) -> None:
        current = _tensor_guards(self.model)
        if current != self._frozen_versions:
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

    def _cache(self, keys: Sequence[torch.Tensor], values: Sequence[torch.Tensor]) -> DynamicCache:
        cache = DynamicCache(config=self.model.config)
        for index, (key, value) in enumerate(zip(keys, values)):
            if key.shape[-2]:
                cache.update(key.to(self.device), value.to(self.device), index)
        return cache

    def _cache_values(self, cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        shape = (
            1,
            self.model.config.num_key_value_heads,
            0,
            self.model.config.hidden_size // self.model.config.num_attention_heads,
        )
        keys, values = [], []
        for layer in cache.layers:
            keys.append(
                clone(layer.keys)
                if layer.keys is not None
                else torch.empty(shape, device=self.device, dtype=self.dtype)
            )
            values.append(
                clone(layer.values)
                if layer.values is not None
                else torch.empty(shape, device=self.device, dtype=self.dtype)
            )
        return keys, values

    @torch.inference_mode()
    def session(self, token_ids: Sequence[int]) -> Session:
        self.validate_execution()
        tokens = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        if tokens.shape[-1] < 1:
            raise ValueError("at least one pending token is required")
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
        operator = phase
        index = state["layer"] if phase == "layer" else None
        next_operation = {
            "embed": "model.embed_tokens",
            "normalization": "model.norm",
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
            family="qwen2",
            logical_step=state["tokens"].shape[1] - 1,
            phase=phase,
            operator=operator,
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
            consumer="native-qwen2-suffix",
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
            state_schema="saturn-pub-qwen2-state-v2",
            consumer="native-qwen2-suffix",
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
            raise ValueError("Qwen state schema keys differ from the execution contract")
        tokens = state["tokens"]
        if (
            tokens.ndim != 2
            or tokens.shape[0] != 1
            or tokens.dtype != torch.long
            or tokens.device != self.device
        ):
            raise ValueError("Qwen state requires batch-one integer tokens")
        if not 0 < tokens.shape[1] <= self.model.config.max_position_embeddings:
            raise ValueError("token count outside configured context")
        if (tokens < 0).any() or (tokens >= self.model.config.vocab_size).any():
            raise ValueError("token ID outside vocabulary")
        layer = state["layer"]
        if type(layer) is not int or not 0 <= layer <= self.layers:
            raise ValueError("invalid layer cursor")
        phase = state["phase"]
        allowed_phases = {"embed", "layer"}
        if self.granularity == "operation":
            allowed_phases |= {"normalization", "readout", "sample", "commit"}
        if phase not in allowed_phases:
            raise ValueError("invalid Qwen operation phase")
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
            shape = (
                1,
                self.model.config.num_key_value_heads,
                length,
                self.model.config.hidden_size // self.model.config.num_attention_heads,
            )
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
            tuple(hidden.shape) != (1, 1, self.model.config.hidden_size)
            or hidden.dtype != self.dtype
            or hidden.device != self.device
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("invalid hidden carrier")
        logits = state["logits"]
        if logits is not None and (
            tuple(logits.shape) != (1, self.model.config.vocab_size)
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
            or bool((sampled >= self.model.config.vocab_size).any())
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
                "Qwen derived state is recomputed automatically; no manual invalidation"
            )
        return super().write(state, writes, ())

    @torch.inference_mode()
    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate(state)
        out = clone(state)
        tokens = out["tokens"].to(self.device)
        if out["hidden"] is None:
            if tokens.shape[1] >= self.model.config.max_position_embeddings:
                raise ValueError("context capacity reached")
            out["hidden"] = self.model.model.embed_tokens(tokens[:, -1:])
            out["logits"] = None
            out["phase"] = "layer"
            return out
        hidden = out["hidden"].to(self.device)
        layer = out["layer"]
        if layer < self.layers:
            cache = self._cache(out["keys"], out["values"])
            positions = torch.tensor([[tokens.shape[1] - 1]], device=self.device)
            embeddings = self.model.model.rotary_emb(hidden, positions)
            out["hidden"] = self.model.model.layers[layer](
                hidden,
                attention_mask=None,
                position_ids=positions,
                past_key_values=cache,
                position_embeddings=embeddings,
                use_cache=True,
            )
            out["keys"], out["values"] = self._cache_values(cache)
            out["layer"] = layer + 1
            if self.granularity == "operation" and out["layer"] == self.layers:
                out["phase"] = "normalization"
        elif self.granularity == "operation" and out["phase"] == "normalization":
            out["hidden"] = self.model.model.norm(hidden)
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
            logits = self.model.lm_head(self.model.model.norm(hidden))[:, -1]
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

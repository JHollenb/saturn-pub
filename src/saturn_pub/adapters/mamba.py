"""Batch-one, greedy native Mamba-1 machine (HF ``MambaForCausalLM``).

Mamba has no attention or positional embedding. Its per-layer continuation state is a causal
convolution window (``conv``) and a selective-scan recurrent state (``recur``) instead of a
key/value cache. This adapter cuts the native ``MambaBlock`` stack one token and one layer at a
time over an externally held ``transformers`` linear-attention cache (transformers 5.14.1 folded
``MambaCache`` into ``DynamicCache`` with per-layer ``conv_states``/``recurrent_states``). The
session grammar, surface manifest, frozen-model guards, ``validate``/``write`` rules,
``layer``/``operation`` granularity, and ``generate``/``native_logits`` semantics match the
decoder and Qwen adapters. Parity against the native full forward is numeric, not bit-exact:
the cached single-step recurrence and the full-sequence scan agree up to floating-point order.
"""

from __future__ import annotations

import platform
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import transformers
from transformers import DynamicCache, MambaConfig, MambaForCausalLM

from ..contracts import ExecutionPoint, SlotSpec, SurfaceManifest, TransitionSpec
from ..core import Adapter, Session
from ..values import clone, digest, identity
from .decoder import _stable_configuration, _tensor_guards

_MAX_CONTEXT = 1 << 20


class MambaAdapter(Adapter):
    def __init__(self, model: MambaForCausalLM, *, granularity: str = "layer"):
        if model.config.model_type != "mamba":
            raise ValueError("MambaAdapter supports Mamba-1 (MambaForCausalLM) models")
        if type(model).__name__ != "MambaForCausalLM":
            raise ValueError("MambaAdapter requires a MambaForCausalLM causal-LM topology")
        if getattr(model.config, "hidden_act", "silu") != "silu":
            raise ValueError("registered Mamba semantics require the SiLU activation")
        if not getattr(model.config, "rms_norm", True):
            raise ValueError("registered Mamba semantics require RMS normalization")
        if getattr(model.config, "use_mambapy", False):
            raise ValueError("MambaPy scan arithmetic is not registered")
        if getattr(model.config, "quantization_config", None):
            raise ValueError("quantized checkpoints are not supported; load a raw-float model")
        if granularity not in {"layer", "operation"}:
            raise ValueError("granularity must be 'layer' or 'operation'")
        config = model.config
        self.model = model.eval()
        self.granularity = granularity
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.backbone = model.backbone
        self.layers = len(self.backbone.layers)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.d_conv = config.conv_kernel
        self.d_state = config.state_size
        self.d_inner = (
            getattr(config, "intermediate_size", None) or config.hidden_size * config.expand
        )
        stable = _stable_configuration(config.to_dict())
        self._configuration_fingerprint = digest(stable)
        self.model_identity = identity(model, stable)
        self._frozen_versions = _tensor_guards(model)
        self.execution = {
            "family": "mamba",
            "adapter": "native-mamba-v1",
            "granularity": granularity,
            "recurrence": "selective-scan",
            "dtype": str(self.dtype),
            "device": {"type": self.device.type, "index": self.device.index},
            "batch": {"size": 1, "padding": False},
            "environment_versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
            },
            "model_program": (
                "embed->mamba_block[*]->rms_norm->lm_head->greedy_argmax->token_commit"
            ),
            "kernel": {"mixer": "transformers-slow-forward", "cache": "DynamicCache"},
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
            "parity": "bounded-recurrence",
        }

    @classmethod
    def tiny(cls, seed: int = 7, *, granularity: str = "layer") -> MambaAdapter:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            config = MambaConfig(
                vocab_size=64,
                hidden_size=32,
                state_size=16,
                num_hidden_layers=2,
                conv_kernel=4,
                expand=2,
                time_step_rank=4,
            )
            return cls(MambaForCausalLM(config), granularity=granularity)

    @classmethod
    def from_pretrained(
        cls, path: str, *, granularity: str = "layer", **kwargs: Any
    ) -> MambaAdapter:
        return cls(MambaForCausalLM.from_pretrained(path, **kwargs), granularity=granularity)

    # --- frozen-model guards ------------------------------------------------------------

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

    # --- external recurrent state -------------------------------------------------------

    def _conv_shape(self) -> tuple[int, int, int]:
        return (1, self.d_inner, self.d_conv)

    def _recur_shape(self) -> tuple[int, int, int]:
        return (1, self.d_inner, self.d_state)

    def _zero_state(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        conv = [
            torch.zeros(self._conv_shape(), device=self.device, dtype=self.dtype)
            for _ in range(self.layers)
        ]
        recur = [
            torch.zeros(self._recur_shape(), device=self.device, dtype=self.dtype)
            for _ in range(self.layers)
        ]
        return conv, recur

    def _cache_values(self, cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        conv, recur = [], []
        for index in range(self.layers):
            layer = cache.layers[index]
            stored_conv = layer.conv_states.get(0) if layer.conv_states.get(0) is not None else None
            stored_recur = (
                layer.recurrent_states.get(0) if layer.recurrent_states.get(0) is not None else None
            )
            conv.append(
                clone(stored_conv.to(self.device, self.dtype))
                if stored_conv is not None
                else torch.zeros(self._conv_shape(), device=self.device, dtype=self.dtype)
            )
            recur.append(
                clone(stored_recur.to(self.device, self.dtype))
                if stored_recur is not None
                else torch.zeros(self._recur_shape(), device=self.device, dtype=self.dtype)
            )
        return conv, recur

    def _hydrate_layer(self, index: int, conv: torch.Tensor, recur: torch.Tensor) -> DynamicCache:
        cache = DynamicCache(config=self.model.config)
        layer = cache.layers[index]
        layer.lazy_initialization(
            conv_states=conv.to(self.device),
            recurrent_states=recur.to(self.device),
            conv_kernel_size=self.d_conv,
        )
        layer.conv_states[0].copy_(conv.to(self.device))
        layer.recurrent_states[0].copy_(recur.to(self.device))
        layer.has_previous_state[0] = True
        return cache

    # --- session / grammar --------------------------------------------------------------

    @torch.inference_mode()
    def session(self, token_ids: Sequence[int]) -> Session:
        self.validate_execution()
        tokens = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        if tokens.shape[-1] < 1:
            raise ValueError("at least one pending token is required")
        if tokens.shape[-1] > 1:
            cache = self.model(tokens[:, :-1], use_cache=True).cache_params
            conv, recur = self._cache_values(cache)
        else:
            conv, recur = self._zero_state()
        return Session(
            self,
            {
                "tokens": tokens,
                "conv": conv,
                "recur": recur,
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
        index = state["layer"] if phase == "layer" else None
        next_operation = {
            "embed": "backbone.embeddings",
            "normalization": "backbone.norm_f",
            "readout": "lm_head",
            "sample": "greedy_argmax",
            "commit": "token_commit",
        }.get(
            phase,
            (
                f"mamba_block.{state['layer']}"
                if state["layer"] < self.layers
                else "normalization_readout_sampler_commit"
            ),
        )
        return ExecutionPoint(
            family="mamba",
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
            reads = ("tokens", "hidden", "conv", "recur", "layer")
            writes = ("hidden", "conv", "recur", "layer", "phase")
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
            consumer="native-mamba-suffix",
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
                "conv",
                role="cache",
                producer="mamba_block",
                consumer="causal_conv",
                schema={"kind": "tensor_sequence", "length": self.layers},
            ),
            SlotSpec(
                "recur",
                role="cache",
                producer="mamba_block",
                consumer="selective_scan",
                schema={"kind": "tensor_sequence", "length": self.layers},
            ),
            SlotSpec(
                "hidden",
                role="carrier",
                writable=state["hidden"] is not None,
                persistence="authoritative",
                producer="embed_or_block",
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
                producer="mamba_block",
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
            state_schema="saturn-pub-mamba-state-v1",
            consumer="native-mamba-suffix",
            horizon="native-recurrent-suffix",
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
            "conv",
            "recur",
            "layer",
            "phase",
            "hidden",
            "logits",
            "sampled_token",
        }
        if set(state) != expected_keys:
            raise ValueError("mamba state schema keys differ from the execution contract")
        tokens = state["tokens"]
        if (
            tokens.ndim != 2
            or tokens.shape[0] != 1
            or tokens.dtype != torch.long
            or tokens.device != self.device
        ):
            raise ValueError("mamba state requires batch-one integer tokens")
        if not 0 < tokens.shape[1] <= _MAX_CONTEXT:
            raise ValueError("token count outside configured context")
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
            raise ValueError("invalid mamba operation phase")
        if phase == "embed" and (state["hidden"] is not None or layer != 0):
            raise ValueError("invalid embedding boundary closure")
        if phase != "embed" and state["hidden"] is None:
            raise ValueError("missing hidden carrier")
        if self.granularity == "operation" and phase == "layer" and layer == self.layers:
            raise ValueError("operation mode must expose the normalization boundary")
        if phase in {"normalization", "readout", "sample", "commit"} and layer != self.layers:
            raise ValueError("post-layer phase requires a complete block stack")
        conv, recur = state["conv"], state["recur"]
        if len(conv) != self.layers or len(recur) != self.layers:
            raise ValueError("missing layer recurrent state")
        for window, scan in zip(conv, recur):
            if (
                tuple(window.shape) != self._conv_shape()
                or tuple(scan.shape) != self._recur_shape()
            ):
                raise ValueError("conv/recur closure does not match the mixer geometry")
            if window.dtype != self.dtype or scan.dtype != self.dtype:
                raise ValueError("recurrent state dtype mismatch")
            if window.device != self.device or scan.device != self.device:
                raise ValueError("recurrent state device mismatch")
            if not torch.isfinite(window).all() or not torch.isfinite(scan).all():
                raise ValueError("non-finite recurrent state")
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
                "mamba derived state is recomputed automatically; no manual invalidation"
            )
        return super().write(state, writes, ())

    @torch.inference_mode()
    def advance(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate(state)
        out = clone(state)
        tokens = out["tokens"].to(self.device)
        if out["hidden"] is None:
            out["hidden"] = self.backbone.embeddings(tokens[:, -1:])
            out["logits"] = None
            out["phase"] = "layer"
            return out
        hidden = out["hidden"].to(self.device)
        layer = out["layer"]
        if layer < self.layers:
            cache = self._hydrate_layer(layer, out["conv"][layer], out["recur"][layer])
            block = self.backbone.layers[layer]
            result = block(hidden, cache_params=cache, attention_mask=None)
            out["hidden"] = result[0] if isinstance(result, tuple) else result
            layer_cache = cache.layers[layer]
            new_conv = list(out["conv"])
            new_recur = list(out["recur"])
            new_conv[layer] = clone(layer_cache.conv_states[0].to(self.device, self.dtype))
            new_recur[layer] = clone(layer_cache.recurrent_states[0].to(self.device, self.dtype))
            out["conv"], out["recur"] = new_conv, new_recur
            out["layer"] = layer + 1
            if self.granularity == "operation" and out["layer"] == self.layers:
                out["phase"] = "normalization"
        elif self.granularity == "operation" and out["phase"] == "normalization":
            out["hidden"] = self.backbone.norm_f(hidden)
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
            logits = self.model.lm_head(self.backbone.norm_f(hidden))[:, -1]
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

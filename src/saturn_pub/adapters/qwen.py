"""Batch-one, greedy Qwen2 token/layer machine using native HF decoder layers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from ..core import Adapter, Session
from ..values import clone, identity


class QwenAdapter(Adapter):
    def __init__(self, model: Qwen2ForCausalLM):
        if model.config.model_type != "qwen2":
            raise ValueError("v1 supports Qwen2/Qwen2.5 full-attention models")
        if getattr(model.config, "use_sliding_window", False):
            raise ValueError("sliding-window attention is not supported")
        if model.config._attn_implementation != "eager":
            raise ValueError("QwenAdapter requires attn_implementation='eager'")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.layers = len(model.model.layers)
        self.model_identity = identity(model, model.config.to_dict())
        self.execution = {
            "family": "qwen2",
            "adapter": "native-layer-v1",
            "attention": "eager",
            "dtype": str(self.dtype),
            "device_type": self.device.type,
            "sampler": "greedy",
            "parity": "bounded-logits",
        }

    @classmethod
    def tiny(cls, seed: int = 7) -> QwenAdapter:
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
            return cls(Qwen2ForCausalLM(config))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> QwenAdapter:
        return cls(Qwen2ForCausalLM.from_pretrained(path, attn_implementation="eager", **kwargs))

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
                "hidden": None,
                "logits": None,
            },
        )

    def boundary(self, state: Mapping[str, Any]) -> str:
        if state["hidden"] is None:
            return "token"
        return f"layer:{state['layer']}"

    def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
        result = ("tokens",)
        if state["hidden"] is not None:
            result += ("hidden",)
        if state["logits"] is not None:
            result += ("logits",)
        return result

    def validate(self, state: Mapping[str, Any]) -> None:
        tokens = state["tokens"]
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.long:
            raise ValueError("Qwen state requires batch-one integer tokens")
        if not 0 < tokens.shape[1] <= self.model.config.max_position_embeddings:
            raise ValueError("token count outside configured context")
        if (tokens < 0).any() or (tokens >= self.model.config.vocab_size).any():
            raise ValueError("token ID outside vocabulary")
        layer = state["layer"]
        if not isinstance(layer, int) or not 0 <= layer <= self.layers:
            raise ValueError("invalid layer cursor")
        if state["hidden"] is None and layer != 0:
            raise ValueError("missing hidden carrier")
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
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise ValueError("non-finite K/V")
        hidden = state["hidden"]
        if hidden is not None and (
            tuple(hidden.shape) != (1, 1, self.model.config.hidden_size)
            or hidden.dtype != self.dtype
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("invalid hidden carrier")
        logits = state["logits"]
        if logits is not None and (
            tuple(logits.shape) != (1, self.model.config.vocab_size)
            or logits.dtype != self.dtype
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("invalid logit observation")

    def write(
        self, state: Mapping[str, Any], writes: Mapping[str, Any], invalidates: tuple[str, ...]
    ) -> Mapping[str, Any]:
        if "tokens" in writes:
            raise ValueError("token writes require a new prefill; use adapter.session(new_tokens)")
        if "logits" in writes:
            raise ValueError("logits are observations; intervene on hidden before readout")
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
        else:
            logits = self.model.lm_head(self.model.model.norm(hidden))[:, -1]
            out["logits"] = logits
            out["tokens"] = torch.cat((tokens, logits.argmax(-1, keepdim=True)), dim=-1)
            out["hidden"] = None
            out["layer"] = 0
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

"""Causal next-byte comparators: a small Transformer and actual HF Mamba v1.

Both return [batch, sequence, vocabulary] logits, default to untied input/output
embeddings, and receive only byte IDs. Equal context/bytes/parameters does not
imply equal FLOPs or optimized-kernel throughput.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model, bias=False),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model, bias=False),
        )

    def forward(self, hidden):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(self.norm1(hidden)).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v)
        ]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        hidden = hidden + self.attn_out(attended.transpose(1, 2).reshape(batch, length, width))
        return hidden + self.ff(self.norm2(hidden))


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 24,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        tie_word_embeddings: bool = False,
        positional_scale: float = 1.0,
    ):
        super().__init__()
        if not math.isfinite(positional_scale) or positional_scale < 0:
            raise ValueError("positional_scale must be finite and nonnegative")
        self.d_model = d_model
        # Fixed, captured by Saturn's nonpersistent-buffer closure. Keeping this
        # out of state_dict lets existing weight-only checkpoints retain their
        # original scale=1 behavior without a missing-key migration.
        self.register_buffer(
            "positional_scale", torch.tensor(float(positional_scale)), persistent=False
        )
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [CausalTransformerBlock(d_model, n_heads, ff_mult) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.architecture = {
            "kind": "causal_transformer",
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ff_mult": ff_mult,
            "vocab_size": vocab_size,
            "positions": "fixed_sinusoidal",
            "positional_scale": float(positional_scale),
            "positional_scale_trainable": False,
            "tied_embeddings": tie_word_embeddings,
            "dropout": 0.0,
            "attention": "torch_sdpa_causal",
            "initializer_range": 0.02,
        }
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _positions(self, length, device, dtype):
        positions = torch.arange(length, device=device, dtype=torch.float32)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions[:, None] * frequencies[None, :]
        positional = torch.zeros(length, self.d_model, device=device)
        positional[:, 0::2] = torch.sin(angles)
        positional[:, 1::2] = torch.cos(angles[:, : self.d_model // 2])
        return positional.to(dtype=dtype)

    @torch.no_grad()
    def input_scale_observation(self, byte_ids):
        """Empirical input scales only; no model fitting or target information."""
        hidden = self.embedding(byte_ids)
        raw = self._positions(byte_ids.shape[1], hidden.device, hidden.dtype)
        scaled = raw * self.positional_scale
        embedding_rms = hidden.float().square().mean().sqrt()
        position_rms = scaled.float().square().mean().sqrt()
        return {
            "embedding_rms": float(embedding_rms),
            "unscaled_position_rms": float(raw.float().square().mean().sqrt()),
            "position_rms": float(position_rms),
            "position_to_embedding_rms_ratio": float(position_rms / embedding_rms)
            if bool(embedding_rms > 0)
            else None,
            "combined_input_rms": float((hidden + scaled[None]).float().square().mean().sqrt()),
            "positional_scale": float(self.positional_scale),
            "input_shape": list(byte_ids.shape),
            "targets_used": False,
        }

    def forward(self, byte_ids):
        hidden = self.embedding(byte_ids)
        positional = self._positions(byte_ids.shape[1], hidden.device, hidden.dtype)
        hidden = hidden + (positional * self.positional_scale)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


class MambaV1LM(nn.Module):
    """Transformers' actual Mamba v1, freshly initialized without weight downloads."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 24,
        n_layers: int = 2,
        state_size: int = 8,
        expand: int = 2,
        conv_kernel: int = 4,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()
        from transformers import MambaConfig, MambaForCausalLM

        config = MambaConfig(
            vocab_size=vocab_size,
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            state_size=state_size,
            expand=expand,
            conv_kernel=conv_kernel,
            use_cache=False,
            use_mambapy=False,
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )
        # Recent Transformers exposes a compile-only associative scan. Eager
        # reference execution keeps the official implementation's serial scan.
        if hasattr(config, "use_associative_scan"):
            config.use_associative_scan = False
        self.model = MambaForCausalLM(config)
        self.architecture = {
            "kind": "mamba_v1",
            "implementation": "transformers.MambaForCausalLM",
            "d_model": d_model,
            "n_layers": n_layers,
            "state_size": state_size,
            "expand": expand,
            "conv_kernel": conv_kernel,
            "time_step_rank": config.time_step_rank,
            "vocab_size": vocab_size,
            "tied_embeddings": tie_word_embeddings,
            "initializer_range": config.initializer_range,
            "cache": False,
            "use_mambapy": False,
            "optimized_kernels": "official implementation chooses available device kernels; record environment",
        }

    def forward(self, byte_ids):
        return self.model(input_ids=byte_ids, use_cache=False, return_dict=True).logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_baseline(
    kind: str,
    *,
    vocab_size: int = 256,
    d_model: int = 24,
    n_layers: int = 2,
    state_size: int = 8,
    seed: int = 0,
    n_heads: int = 4,
    ff_mult: int = 4,
    expand: int = 2,
    conv_kernel: int = 4,
    tie_word_embeddings: bool = False,
    positional_scale: float = 1.0,
) -> nn.Module:
    if min(vocab_size, d_model, n_layers, state_size, n_heads, ff_mult, expand, conv_kernel) <= 0:
        raise ValueError("model dimensions must be positive")
    # Model construction does not advance the caller's global RNG closure.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if kind in {"transformer", "causal_transformer"}:
            model = CausalTransformerLM(
                vocab_size,
                d_model,
                n_layers,
                n_heads,
                ff_mult,
                tie_word_embeddings,
                positional_scale,
            )
        elif kind in {"mamba", "mamba_v1", "mamba_hf"}:
            model = MambaV1LM(
                vocab_size, d_model, n_layers, state_size, expand, conv_kernel, tie_word_embeddings
            )
        else:
            raise ValueError(
                f"unknown causal baseline: {kind}; bidirectional Hydra needs a separate masked task"
            )
    model.architecture["parameters"] = parameter_count(model)
    model.architecture["seed"] = int(seed)
    return model

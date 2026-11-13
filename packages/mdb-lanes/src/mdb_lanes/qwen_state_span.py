"""Experimental recurring Qwen2 state producer and validated KV installation.

The producer reads the current branch, never an observed target exit. A packet
owns every current KV row in its contiguous span and the exit residual. Installing
it advances the layer cursor, not the token/history cursor. Callers retain a fork
for rollback; validation failures make no writes. This lane is BF16 native-eager.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F

from mdb_lanes.read_weight_native import _rms_norm


@dataclass(frozen=True)
class QwenSpanPrediction:
    binding: tuple
    entry_hidden: torch.Tensor
    start: int
    stop: int
    hidden: torch.Tensor
    rows: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    attention_output: torch.Tensor
    profile: str


def _version(value):
    try:
        return value._version
    except RuntimeError:  # inference tensors have no version counter
        return None


def _binding(execution):
    state = execution.state
    return (id(execution), id(state), state.model_abi_sha256, state.length,
            state.delta_rows, execution.token_id, execution.next_layer,
            id(execution.hidden), _version(execution.hidden),
            tuple((id(t), _version(t)) for pair in state.delta for t in pair))


@torch.inference_mode()
def install_span(runtime, execution, packet: QwenSpanPrediction):
    """Validate the whole packet before writing any private current-row bytes."""
    if (execution.state.model_abi_sha256 != runtime.model_abi_sha256
            or packet.binding != _binding(execution) or not torch.equal(packet.entry_hidden, execution.hidden)
            or execution.logits is not None
            or packet.start != execution.next_layer
            or not 0 <= packet.start < packet.stop <= runtime.codec.layers
            or len(packet.rows) != packet.stop - packet.start
            or execution.state.delta_rows >= execution.state.delta_capacity):
        raise ValueError("stale or incompatible span prediction")
    if packet.profile not in {"bf16-native-eager-functional", "bf16-native-eager-fp32-rank-gu"}:
        raise ValueError("unknown span arithmetic profile")
    hidden = execution.hidden
    targets = []
    for i, pair in enumerate(packet.rows, packet.start):
        if len(pair) != 2:
            raise ValueError("span needs paired KV rows")
        for value, bank in zip(pair, execution.state.delta[i]):
            target = bank[:, :, execution.state.delta_rows:execution.state.delta_rows+1]
            targets.append((value, target))
    targets.extend(((packet.hidden, hidden), (packet.attention_output, hidden)))
    for value, target in targets:
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or value.device != target.device
                or not bool(torch.isfinite(value).all())):
            raise ValueError("span tensor shape/dtype/device/finite contract differs")
    for value, target in targets[:-2]:
        target.copy_(value)
    execution.hidden = packet.hidden
    execution.next_layer = packet.stop
    return packet.attention_output


class QwenStateSpanProducer:
    """Original-equation control or source-frozen reduced G/U producer.

    Bases are rows E of an existing frozen source basis. Compilation pays for
    Wg @ E.T and Wu @ E.T once. Production pays E @ x and the two images;
    original gate/up maps are never evaluated in the reduced arm. Attention,
    down projections and the rest of the model retain their native operations.
    """
    @torch.inference_mode()
    def __init__(self, runtime: Any, start: int, stop: int, *, bases=None):
        if (runtime.codec.model_type != "qwen2" or runtime.backend != "native-eager"
                or runtime.codec.dtype != torch.bfloat16
                or not 0 <= start < stop <= runtime.codec.layers
                or runtime.model.config.hidden_act != "silu"):
            raise ValueError("span producer requires Qwen2 BF16 native-eager SiLU")
        self.runtime, self.start, self.stop = runtime, start, stop
        self.profile = "bf16-native-eager-functional" if bases is None else "bf16-native-eager-fp32-rank-gu"
        self.factors = {}
        self.compile_macs = 0
        if bases is not None:
            if set(bases) != set(range(start, stop)):
                raise ValueError("basis layer coverage differs")
            for index in range(start, stop):
                layer = runtime.model.model.layers[index]
                E = bases[index].detach().to(device=runtime.device, dtype=torch.float32).contiguous().clone()
                if (E.ndim != 2 or E.shape[1] != runtime.model.config.hidden_size
                        or not 0 < E.shape[0] < E.shape[1] or not bool(torch.isfinite(E).all())):
                    raise ValueError("invalid frozen receiver basis")
                if layer.mlp.gate_proj.bias is not None or layer.mlp.up_proj.bias is not None:
                    raise ValueError("reduced producer requires bias-free gate/up")
                self.factors[index] = (E, layer.mlp.gate_proj.weight.float() @ E.T,
                                      layer.mlp.up_proj.weight.float() @ E.T)
                self.compile_macs += E.shape[0] * (layer.mlp.gate_proj.weight.numel() + layer.mlp.up_proj.weight.numel())

    @torch.inference_mode()
    def predict(self, execution):
        from transformers.models.qwen2 import modeling_qwen2 as modeling
        if (execution.next_layer != self.start or execution.logits is not None
                or execution.state.model_abi_sha256 != self.runtime.model_abi_sha256):
            raise ValueError("prediction requires the current span entry")
        binding = _binding(execution)
        entry_hidden = execution.hidden.clone()
        state, hidden = execution.state, execution.hidden
        position = torch.tensor([[state.length]], dtype=torch.long, device=hidden.device)
        rope = self.runtime.model.model.rotary_emb(hidden, position)
        rows = []
        for index in range(self.start, self.stop):
            layer = self.runtime.model.model.layers[index]
            attn = layer.self_attn
            normalized = _rms_norm(hidden, layer.input_layernorm, torch=torch)
            shape = (1, 1, -1, attn.head_dim)
            query = F.linear(normalized, attn.q_proj.weight, attn.q_proj.bias).view(shape).transpose(1, 2).contiguous()
            key = F.linear(normalized, attn.k_proj.weight, attn.k_proj.bias).view(shape).transpose(1, 2)
            value = F.linear(normalized, attn.v_proj.weight, attn.v_proj.bias).view(shape).transpose(1, 2)
            query, key = modeling.apply_rotary_pos_emb(query, key, *rope)
            keys = torch.cat((state.ancestor[index][0], state.delta[index][0][:, :, :state.delta_rows], key), dim=2)
            values = torch.cat((state.ancestor[index][1], state.delta[index][1][:, :, :state.delta_rows], value), dim=2)
            attended, _ = modeling.eager_attention_forward(attn, query.contiguous(), keys, values,
                attention_mask=None, scaling=attn.scaling, dropout=0.0)
            attention_output = F.linear(attended.reshape(1, 1, -1).contiguous(), attn.o_proj.weight, attn.o_proj.bias)
            hidden = hidden + attention_output
            normalized = _rms_norm(hidden, layer.post_attention_layernorm, torch=torch)
            if index in self.factors:
                E, gate_image, up_image = self.factors[index]
                code = F.linear(normalized.float(), E)
                gate = F.linear(code, gate_image).to(hidden.dtype)
                up = F.linear(code, up_image).to(hidden.dtype)
            else:
                gate = F.linear(normalized, layer.mlp.gate_proj.weight, layer.mlp.gate_proj.bias)
                up = F.linear(normalized, layer.mlp.up_proj.weight, layer.mlp.up_proj.bias)
            hidden = hidden + F.linear(F.silu(gate) * up, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
            rows.append((key, value))
        return QwenSpanPrediction(binding, entry_hidden, self.start, self.stop, hidden, tuple(rows), attention_output, self.profile)


@contextmanager
def use_state_span(runtime, producer):
    """Exclusive synchronous dispatch before the span; use step(monitor=False).

    The caller is the sole writer of the owned branch during this context.
    Per-layer monitor traces are unsupported because one dispatch spans layers.
    """
    if producer.runtime is not runtime:
        raise ValueError("producer belongs to another runtime")
    original = runtime.advance_token
    original_step = runtime.step
    def advance(execution):
        if execution.next_layer == producer.start:
            return install_span(runtime, execution, producer.predict(execution))
        return original(execution)
    runtime.advance_token = advance
    def step(state, token_id, *, monitor=False):
        if monitor:
            raise ValueError("per-layer monitor is unsupported during whole-span dispatch")
        return original_step(state, token_id, monitor=False)
    runtime.step = step
    try:
        yield
    finally:
        runtime.advance_token = original
        runtime.step = original_step

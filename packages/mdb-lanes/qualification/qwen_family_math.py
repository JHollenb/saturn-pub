"""Original-parameter Qwen2 eager equations and private greedy/KV futures.

This is a bounded FP32/default-RoPE/all-attended full-causal adapter, not a native module
wrapper.  Attention interventions occur after receiver KV append.  They edit
consumed operands, never that layer's stored KV.  ``tail_only`` retains the
natural prefix head and replaces only the final query's head.  Later layers
and generated tokens consume their own resulting hidden states and caches.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


class QwenMathError(ValueError):
    """Unsupported profile or malformed functional operand."""


def require(condition, message):
    if not condition:
        raise QwenMathError(message)


def rms_norm(value, weight, epsilon):
    dtype = value.dtype
    wide = value.to(torch.float32)
    return weight * (wide * torch.rsqrt(wide.pow(2).mean(-1, keepdim=True) + epsilon)).to(dtype)


def rotate_half(value):
    left, right = value[..., : value.shape[-1] // 2], value[..., value.shape[-1] // 2 :]
    return torch.cat((-right, left), dim=-1)


def repeat_kv(value, groups):
    if groups == 1:
        return value
    batch, heads, length, width = value.shape
    return value[:, :, None, :, :].expand(batch, heads, groups, length, width).reshape(
        batch, heads * groups, length, width)


def eager_attention(q, k, v, mask, scaling, kv_groups):
    """Return BHTD head; operation order matches Qwen2 eager inference."""
    k, v = repeat_kv(k, kv_groups), repeat_kv(v, kv_groups)
    weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    if mask is not None:
        weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(weights, v)


def clone_cache(cache):
    """Clone optional per-layer (post-RoPE K, V), preserving private ownership."""
    return tuple(None if pair is None else tuple(value.detach().clone() for value in pair)
                 for pair in cache)


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


class FunctionalQwen:
    """Take original state-dict tensors/config, without retaining native modules.

    ``intervene(layer, stage, ports)`` returns None or overrides. Attention
    accepts q/k/v and optional tail_only; MLP accepts raw gate/up. Capture is
    called after each stage and the block. ``graph_kernel`` runs only for an
    overridden stage: attention returns BHTD, MLP returns SiLU(gate)*up.
    Callbacks must be read-only with respect to the supplied original tensors.
    ``retain_layers`` clones complete port records for only those layers.
    """

    def __init__(self, parameters, config, *, device_type="cpu"):
        get = config.get if isinstance(config, Mapping) else lambda key, default=None: getattr(config, key, default)
        self.layers = int(get("num_hidden_layers"))
        self.width = int(get("hidden_size"))
        self.heads = int(get("num_attention_heads"))
        self.kv_heads = int(get("num_key_value_heads", self.heads))
        self.head_dim = int(get("head_dim") or self.width // self.heads)
        self.intermediate = int(get("intermediate_size"))
        self.epsilon = float(get("rms_norm_eps", 1e-6))
        rope = get("rope_parameters") or {"rope_type": "default", "rope_theta": get("rope_theta", 10000.)}
        require(get("model_type", "qwen2") == "qwen2", "only Qwen2 geometry is supported")
        require(get("hidden_act", "silu") == "silu", "only SiLU MLP is supported")
        require(get("_attn_implementation") in (None, "eager"), "only eager attention is supported")
        require(rope.get("rope_type", "default") == "default", "only default RoPE is supported")
        require(not get("use_sliding_window", False) and all(
            item == "full_attention" for item in get("layer_types", ["full_attention"] * self.layers)),
            "sliding/local attention is unsupported")
        require(self.layers > 0 and self.head_dim % 2 == 0 and self.heads % self.kv_heads == 0,
                "invalid layer/head/GQA geometry")
        require(self.epsilon > 0 and float(rope["rope_theta"]) > 0, "invalid RMS/RoPE constants")
        self.parameters = {name: value.detach() for name, value in parameters.items()}
        embed = self.parameters["model.embed_tokens.weight"]
        self.device, self.dtype = embed.device, embed.dtype
        require(self.device.type == device_type and self.dtype == torch.float32,
                "declared FP32 device profile differs")
        self.groups, self.scaling = self.heads // self.kv_heads, self.head_dim**-0.5
        self.vocab = embed.shape[0]
        self._parameter("model.embed_tokens.weight", (self.vocab, self.width))
        self._parameter("model.norm.weight", (self.width,))
        if "lm_head.weight" not in self.parameters and get("tie_word_embeddings", False):
            self.parameters["lm_head.weight"] = embed
        self._parameter("lm_head.weight", (self.vocab, self.width))
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                self._parameter(prefix + name, (self.width,))
            for name, size in (("q", self.heads), ("k", self.kv_heads), ("v", self.kv_heads)):
                self._parameter(prefix + f"self_attn.{name}_proj.weight", (size * self.head_dim, self.width))
                self._parameter(prefix + f"self_attn.{name}_proj.bias", (size * self.head_dim,))
            self._parameter(prefix + "self_attn.o_proj.weight", (self.width, self.heads * self.head_dim))
            for name in ("gate", "up"):
                self._parameter(prefix + f"mlp.{name}_proj.weight", (self.intermediate, self.width))
            self._parameter(prefix + "mlp.down_proj.weight", (self.width, self.intermediate))
        self.inv_freq = 1. / (float(rope["rope_theta"]) ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=self.device).float() / self.head_dim))
        self.profile = {"model_type": "qwen2", "dtype": "torch.float32", "device": device_type,
                        "attention": "eager", "rope": "default", "cache": "private post-RoPE KV",
                        "all_attended": True,
                        "numeric_claim": "measured finite errors; no unconditional bit identity"}

    def _parameter(self, name, shape):
        require(name in self.parameters, "missing original parameter: " + name)
        value = self.parameters[name]
        require(tuple(value.shape) == shape and value.dtype == self.dtype and value.device == self.device,
                "parameter shape/dtype/device differs: " + name)
        return value

    def _rope(self, hidden, positions):
        frequency = self.inv_freq[None, :, None].float().expand(positions.shape[0], -1, 1).to(hidden.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            angles = (frequency.float() @ positions[:, None, :].float()).transpose(1, 2)
            angles = torch.cat((angles, angles), dim=-1)
            cos, sin = angles.cos(), angles.sin()
        return cos.to(hidden.dtype), sin.to(hidden.dtype)

    def _cache(self, cache):
        if cache is None:
            return [None] * self.layers
        require(len(cache) == self.layers, "cache layer count differs")
        for pair in cache:
            require(pair is None or len(pair) == 2, "cache requires K/V pairs")
            if pair is not None:
                require(pair[0].shape == pair[1].shape and tuple(pair[0].shape[:2]) == (1, self.kv_heads)
                        and pair[0].shape[-1] == self.head_dim, "cache geometry differs")
                for value in pair:
                    require(value.ndim == 4 and value.dtype == self.dtype and value.device == self.device,
                            "cache dtype/device differs")
        return list(clone_cache(cache))

    def _mask(self, value, length, past):
        total = past + length
        allowed = torch.arange(total, device=self.device)[None, :] <= (
            torch.arange(length, device=self.device)[:, None] + past)
        mask = torch.zeros((1, 1, length, total), device=self.device, dtype=self.dtype)
        mask = mask.masked_fill(~allowed[None, None, :, :], torch.finfo(self.dtype).min)
        if value is not None and value.ndim == 4:
            require(value.shape[0] == 1 and value.shape[1] in (1, self.heads)
                    and tuple(value.shape[-2:]) == (length, total)
                    and value.device == self.device and value.dtype == self.dtype, "4D mask profile differs")
            require(torch.equal(value, mask.expand_as(value)), "only all-attended causal masks are supported")
            return value
        if value is not None:
            require(tuple(value.shape) == (1, total) and value.device == self.device
                    and bool((value == 1).all()), "2D mask must be all-attended and cover the complete key history")
        return mask

    def _overrides(self, layer, stage, ports, intervene, names):
        changed = None if intervene is None else intervene(layer, stage, ports)
        if changed is None:
            return ports, False, False
        require(isinstance(changed, Mapping) and set(changed) <= set(names) | {"tail_only"},
                "unknown intervention port at " + stage)
        result = dict(ports)
        for name in names:
            if name in changed:
                value = changed[name]
                require(isinstance(value, torch.Tensor) and value.shape == ports[name].shape
                        and value.dtype == self.dtype and value.device == self.device,
                        "intervention shape/dtype/device differs: " + name)
                result[name] = value
        require(type(changed.get("tail_only", False)) is bool, "tail_only must be boolean")
        return result, True, changed.get("tail_only", False)

    @torch.no_grad()
    def forward(self, input_ids=None, *, hidden_states=None, start_layer=0, stop_layer=None,
                cache=None, attention_mask=None, position_ids=None, capture=None, intervene=None,
                graph_kernel=None, retain_layers=(), logits_to_keep=0):
        """Run a full prompt/token or resume at an explicitly supplied layer input.

        A stopped prefix returns logits=None. Lower-layer caches already filled
        by a genuine prefix may coexist with empty upper-layer caches. Positions
        and masking are derived from the first executed layer, not layer zero.
        Returned caches never alias caller caches. This method calls no modules.
        """
        require((input_ids is None) != (hidden_states is None), "supply exactly input_ids or hidden_states")
        require(type(start_layer) is int and 0 <= start_layer < self.layers, "invalid start layer")
        stop = self.layers if stop_layer is None else stop_layer
        require(type(stop) is int and start_layer < stop <= self.layers, "invalid stop layer")
        require(input_ids is None or start_layer == 0, "input IDs require layer zero")
        if input_ids is not None:
            require(input_ids.ndim == 2 and input_ids.shape[0] == 1 and input_ids.shape[1] > 0
                    and input_ids.dtype == torch.long and input_ids.device == self.device,
                    "input IDs require B1 nonempty int64 on the declared device")
            hidden = F.embedding(input_ids, self.parameters["model.embed_tokens.weight"])
        else:
            require(hidden_states.ndim == 3 and hidden_states.shape[0] == 1
                    and hidden_states.shape[1] > 0 and hidden_states.shape[-1] == self.width
                    and hidden_states.dtype == self.dtype and hidden_states.device == self.device,
                    "layer input shape/dtype/device differs")
            hidden = hidden_states.clone()
        local_cache = self._cache(cache)
        past = 0 if local_cache[start_layer] is None else local_cache[start_layer][0].shape[-2]
        length = hidden.shape[1]
        if position_ids is None:
            position_ids = (torch.arange(length, device=self.device) + past).unsqueeze(0)
        require(tuple(position_ids.shape) == (1, length) and position_ids.dtype == torch.long
                and position_ids.device == self.device, "position IDs differ")
        mask, (cos, sin) = self._mask(attention_mask, length, past), self._rope(hidden, position_ids)
        boundaries = {}
        for layer in range(start_layer, stop):
            prefix = f"model.layers.{layer}."
            pair = local_cache[layer]
            require((0 if pair is None else pair[0].shape[-2]) == past, "executed layers have inconsistent KV lengths")
            residual = hidden
            norm = rms_norm(hidden, self.parameters[prefix + "input_layernorm.weight"], self.epsilon)

            def project(name):
                key = prefix + "self_attn." + name + "_proj."
                return F.linear(norm, self.parameters[key + "weight"], self.parameters[key + "bias"]).view(
                    1, length, -1, self.head_dim).transpose(1, 2)

            q_raw, k_raw, v = (project(name) for name in ("q", "k", "v"))
            q = (q_raw * cos.unsqueeze(1)) + (rotate_half(q_raw) * sin.unsqueeze(1))
            k = (k_raw * cos.unsqueeze(1)) + (rotate_half(k_raw) * sin.unsqueeze(1))
            if pair is not None:
                k, v = torch.cat((pair[0], k), dim=-2), torch.cat((pair[1], v), dim=-2)
            else:
                # DynamicLayer.update also cats an empty first cache allocation.
                k, v = torch.cat((k[:, :, :0, :], k), dim=-2), torch.cat((v[:, :, :0, :], v), dim=-2)
            local_cache[layer] = (k.clone(), v.clone())
            original = {"q": q, "k": k, "v": v, "mask": mask, "scaling": self.scaling, "kv_groups": self.groups}
            operands, changed, tail = self._overrides(layer, "attention", original, intervene, ("q", "k", "v"))
            natural_head = eager_attention(q, k, v, mask, self.scaling, self.groups) if tail else None
            consumed = {name: operands[name] for name in original}
            if tail:
                consumed["q"], consumed["mask"] = operands["q"][:, :, -1:, :], mask[:, :, -1:, :]
            if changed and graph_kernel is not None:
                mixed_head = graph_kernel(layer, "attention", consumed)
            else:
                mixed_head = eager_attention(**consumed)
            require(mixed_head.shape == consumed["q"].shape and mixed_head.dtype == self.dtype
                    and mixed_head.device == self.device, "attention kernel return differs")
            head = mixed_head
            if tail:
                head = natural_head.clone()
                head[:, :, -1:, :] = mixed_head
            merged = head.transpose(1, 2).contiguous().reshape(1, length, -1).contiguous()
            attention_return = F.linear(merged, self.parameters[prefix + "self_attn.o_proj.weight"])
            hidden = residual + attention_return
            attn_ports = {"residual": residual, "normalized": norm, "q_raw": q_raw, "k_raw": k_raw,
                          "original_q": q, "original_k": k, "original_v": v,
                          "q": consumed["q"], "k": consumed["k"], "v": consumed["v"],
                          "mask": consumed["mask"], "cos": cos, "sin": sin, "position_ids": position_ids,
                          "scaling": self.scaling, "kv_groups": self.groups, "tail_only": tail,
                          "natural_head": natural_head, "primitive_return": mixed_head, "head": head,
                          "merged": merged, "attention_return": attention_return, "post_attention": hidden}
            if capture is not None:
                capture(layer, "attention", attn_ports)
            residual_mlp = hidden
            norm_mlp = rms_norm(hidden, self.parameters[prefix + "post_attention_layernorm.weight"], self.epsilon)
            gate = F.linear(norm_mlp, self.parameters[prefix + "mlp.gate_proj.weight"])
            up = F.linear(norm_mlp, self.parameters[prefix + "mlp.up_proj.weight"])
            operands, changed, tail = self._overrides(layer, "mlp", {"gate": gate, "up": up}, intervene, ("gate", "up"))
            require(not tail, "MLP row replacement must be explicit in gate/up arrays")
            product = graph_kernel(layer, "mlp", operands) if changed and graph_kernel is not None else (
                F.silu(operands["gate"]) * operands["up"])
            require(product.shape == gate.shape and product.dtype == self.dtype and product.device == self.device,
                    "MLP kernel return differs")
            mlp_return = F.linear(product, self.parameters[prefix + "mlp.down_proj.weight"])
            hidden = residual_mlp + mlp_return
            mlp_ports = {"residual": residual_mlp, "normalized": norm_mlp, "original_gate": gate,
                         "original_up": up, "gate": operands["gate"], "up": operands["up"],
                         "primitive_return": product, "mlp_return": mlp_return, "output": hidden}
            block_ports = {"input": residual, "post_attention": residual_mlp, "output": hidden}
            if capture is not None:
                capture(layer, "mlp", mlp_ports)
                capture(layer, "block", block_ports)
            if layer in retain_layers:
                boundaries[layer] = _snapshot({"attention": attn_ports, "mlp": mlp_ports, "block": block_ports})
        logits, normalized = None, None
        if stop == self.layers:
            normalized = rms_norm(hidden, self.parameters["model.norm.weight"], self.epsilon)
            require(type(logits_to_keep) is int and logits_to_keep >= 0, "invalid logits_to_keep")
            logits = F.linear(normalized[:, slice(-logits_to_keep, None), :], self.parameters["lm_head.weight"])
        return {"logits": logits, "hidden_states": hidden, "normalized": normalized,
                "cache": clone_cache(local_cache), "boundaries": boundaries,
                "position_ids": position_ids.clone(), "past_length": past, "executed_layers": (start_layer, stop)}

    @torch.no_grad()
    def greedy(self, input_ids=None, *, initial=None, max_new_tokens, **forward_kwargs):
        """Predict a fixed number of own argmax tokens, intervening at prefill only.

        The last predicted token has not yet been fed, just as native cached
        generation returns KV for prompt + generated_tokens[:-1]. No EOS stop,
        sampling, teacher token, or native module call is implicit here.
        """
        require(type(max_new_tokens) is int and 1 <= max_new_tokens <= 256, "greedy budget must be 1..256")
        require((input_ids is None) != (initial is None), "supply input IDs or an initial result")
        first = self.forward(input_ids, **forward_kwargs) if initial is None else initial
        require(first["logits"] is not None, "greedy needs a complete decoder result")
        current, predictions, rows = first, [], []
        for index in range(max_new_tokens):
            logits = current["logits"][:, -1, :].clone()
            token = logits.argmax(dim=-1, keepdim=True)
            predictions.append(token)
            rows.append({"index": index, "token": int(token.item()), "logits": logits,
                         "cache_length": current["cache"][0][0].shape[-2]})
            if index + 1 < max_new_tokens:
                current = self.forward(token, cache=current["cache"], logits_to_keep=1)
        return {"tokens": torch.cat(predictions, dim=1), "steps": rows,
                "cache": clone_cache(current["cache"]), "initial": first}


__all__ = ["FunctionalQwen", "QwenMathError", "clone_cache", "eager_attention", "rms_norm"]

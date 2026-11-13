"""Smooth, receiver-free transport for a Qwen read.

This module is deliberately a *producer* rather than another model backend.  It
copies the small collection of Qwen parameters needed by the L19-to-L22 path
into FP32 tensors and evaluates that path with functional PyTorch operations.
No module ``forward`` method is called.  The resulting smooth shadow is useful
for a baseline-only tangent calculation; a native A receiver may be opened
later by a caller as a held-out measurement.

The shadow keeps the old KV rows fixed and recomputes only the current row at
each intervening layer.  ``compile_score_program`` then reduces the resulting
query/current-key tangent to per-row attention-logit coefficients.  Once that
program has been compiled, ``evaluate_score_program`` does not perform dense
query/key products.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any


class ReadWeightTransportError(ValueError):
    """Raised when a transport input violates the fixed tensor contract."""


def _torch():
    """Import torch lazily so framework-free MDB commands remain importable."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - exercised in framework-free installs
        raise ReadWeightTransportError("read weight transport requires PyTorch") from exc
    return torch, functional


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - guarded by callers
        return False
    return isinstance(value, torch.Tensor)


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReadWeightTransportError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _float_parameter(value: Any, *, device: Any):
    torch, _ = _torch()
    if not isinstance(value, torch.Tensor):
        raise ReadWeightTransportError("Qwen parameter is not a tensor")
    # ``detach().clone()`` is important when the source was captured under
    # inference_mode: the transport's JVP must receive an ordinary tensor.
    with torch.inference_mode(False):
        return value.detach().to(device=device, dtype=torch.float32).clone().requires_grad_(False)


def _float_input(value: Any, *, device: Any, label: str):
    torch, _ = _torch()
    if not isinstance(value, torch.Tensor):
        raise ReadWeightTransportError(f"{label} must be a tensor")
    return value.to(device=device, dtype=torch.float32)


def _copy_linear(module: Any, *, device: Any, label: str) -> tuple[Any, Any | None]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ReadWeightTransportError(f"{label} has no weight")
    bias = getattr(module, "bias", None)
    return (
        _float_parameter(weight, device=device),
        None if bias is None else _float_parameter(bias, device=device),
    )


def _copy_norm(module: Any, *, device: Any, label: str) -> tuple[Any, float]:
    weight = getattr(module, "weight", None)
    if weight is None:
        raise ReadWeightTransportError(f"{label} has no weight")
    epsilon = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
    if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ReadWeightTransportError(f"{label} has an invalid RMSNorm epsilon")
    return _float_parameter(weight, device=device), float(epsilon)


def _module_layers(model: Any) -> tuple[Any, ...]:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None) if backbone is not None else None
    if layers is None:
        layers = getattr(model, "layers", None)
    if layers is None:
        raise ReadWeightTransportError("model must expose model.layers or layers")
    try:
        result = tuple(layers)
    except TypeError as exc:
        raise ReadWeightTransportError("model layers are not iterable") from exc
    if not result:
        raise ReadWeightTransportError("model has no decoder layers")
    return result


def _attention_geometry(attention: Any, model: Any, *, label: str) -> tuple[int, int, int, float]:
    torch, _ = _torch()
    q_weight = getattr(getattr(attention, "q_proj", None), "weight", None)
    k_weight = getattr(getattr(attention, "k_proj", None), "weight", None)
    if not isinstance(q_weight, torch.Tensor) or not isinstance(k_weight, torch.Tensor):
        raise ReadWeightTransportError(f"{label} has no q_proj/k_proj weights")
    config = getattr(model, "config", None)
    heads = getattr(attention, "num_heads", None)
    if heads is None and config is not None:
        heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(attention, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)
    if head_dim is None and heads:
        head_dim = int(q_weight.shape[0]) // int(heads)
    kv_heads = getattr(attention, "num_key_value_heads", None)
    if kv_heads is None and config is not None:
        kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is None and head_dim:
        kv_heads = int(k_weight.shape[0]) // int(head_dim)
    try:
        heads = _plain_int(heads, f"{label} num_heads", minimum=1)
        kv_heads = _plain_int(kv_heads, f"{label} num_key_value_heads", minimum=1)
        head_dim = _plain_int(head_dim, f"{label} head_dim", minimum=1)
    except ReadWeightTransportError:
        raise
    if int(q_weight.shape[0]) != heads * head_dim:
        raise ReadWeightTransportError(f"{label} q_proj width disagrees with head geometry")
    if int(k_weight.shape[0]) != kv_heads * head_dim:
        raise ReadWeightTransportError(f"{label} k_proj width disagrees with head geometry")
    if heads % kv_heads:
        raise ReadWeightTransportError(f"{label} query heads are not divisible by KV heads")
    scaling = getattr(attention, "scaling", head_dim ** -0.5)
    if isinstance(scaling, torch.Tensor):
        if scaling.numel() != 1:
            raise ReadWeightTransportError(f"{label} scaling must be scalar")
        scaling = float(scaling.detach().item())
    if not isinstance(scaling, (int, float)) or not math.isfinite(float(scaling)) or float(scaling) <= 0:
        raise ReadWeightTransportError(f"{label} has invalid attention scaling")
    return heads, kv_heads, head_dim, float(scaling)


def _linear(functional, spec, x):
    return functional.linear(x, spec[0], spec[1])


def _rms_norm(x, spec, functional):
    torch, _ = _torch()
    weight, epsilon = spec
    # The smooth producer intentionally keeps the whole derivative in FP32.
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + epsilon) * weight


def _swi_glu(x, mlp, functional):
    torch, _ = _torch()
    gate = _linear(functional, mlp["gate"], x)
    up = _linear(functional, mlp["up"], x)
    return _linear(functional, mlp["down"], functional.silu(gate) * up)


def _rotate_half(x):
    half = int(x.shape[-1]) // 2
    if half * 2 != int(x.shape[-1]):
        raise ReadWeightTransportError("RoPE head dimension must be even")
    return _torch()[0].cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(x, cos, sin):
    torch, _ = _torch()
    if cos.shape[-1] == int(x.shape[-1]) // 2:
        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
    if cos.shape[-1] != x.shape[-1] or sin.shape[-1] != x.shape[-1]:
        raise ReadWeightTransportError("cos/sin last dimension must match head dimension")
    try:
        return x * cos + _rotate_half(x) * sin
    except RuntimeError as exc:
        raise ReadWeightTransportError("cos/sin cannot broadcast over the current row") from exc


def _repeat_kv(value, repeats: int):
    if repeats == 1:
        return value
    batch, kv_heads, rows, width = value.shape
    return value[:, :, None, :, :].expand(batch, kv_heads, repeats, rows, width).reshape(
        batch, kv_heads * repeats, rows, width
    )


def _canonical_q(value, *, label: str):
    torch, _ = _torch()
    if not isinstance(value, torch.Tensor):
        raise ReadWeightTransportError(f"{label} must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != 1:
        raise ReadWeightTransportError(f"{label} must have shape [1, heads, 1, head_dim]")
    return value


def _canonical_k(value, *, label: str):
    torch, _ = _torch()
    if not isinstance(value, torch.Tensor):
        raise ReadWeightTransportError(f"{label} must be a tensor")
    if value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != 1:
        raise ReadWeightTransportError(f"{label} must have shape [1, heads, rows, head_dim]")
    return value


def _canonical_scores(value, *, label: str):
    torch, _ = _torch()
    if not isinstance(value, torch.Tensor):
        raise ReadWeightTransportError(f"{label} must be a tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 3:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[2] != 1:
        raise ReadWeightTransportError(f"{label} must have shape [1, heads, 1, rows]")
    return value


class SmoothQwenReadTransport:
    """FP32 functional shadow for a Qwen source-write to target read.

    ``incoming`` is the hidden row entering ``source_layer`` and
    ``baseline_attention`` is that layer's native attention output before its
    MLP.  ``history`` contains old, already committed KV rows for every full
    intermediate layer.  The target layer's current q/k are returned after its
    input RMSNorm and RoPE.
    """

    def __init__(
        self,
        model: Any,
        *,
        source_layer: int = 19,
        target_layer: int = 22,
        incoming: Any,
        baseline_attention: Any,
        history: Mapping[int, tuple[Any, Any]],
        cos: Any,
        sin: Any,
    ) -> None:
        torch, _ = _torch()
        self.source_layer = _plain_int(source_layer, "source_layer")
        self.target_layer = _plain_int(target_layer, "target_layer")
        if self.source_layer >= self.target_layer:
            raise ReadWeightTransportError("source_layer must precede target_layer")
        if not isinstance(history, Mapping):
            raise ReadWeightTransportError("history must map layer index to (K, V)")
        if not _is_tensor(incoming):
            raise ReadWeightTransportError("incoming must be a tensor")
        if incoming.ndim != 3 or tuple(incoming.shape[:2]) != (1, 1):
            raise ReadWeightTransportError("incoming must have shape [1, 1, hidden_size]")
        self.device = incoming.device
        self.incoming = _float_parameter(incoming, device=self.device)
        self.baseline_attention = _float_parameter(baseline_attention, device=self.device)
        if self.baseline_attention.shape != self.incoming.shape:
            raise ReadWeightTransportError("baseline_attention must match incoming shape")
        self.cos = _float_parameter(cos, device=self.device)
        self.sin = _float_parameter(sin, device=self.device)
        if self.cos.ndim < 2 or self.sin.ndim < 2 or self.cos.shape != self.sin.shape:
            raise ReadWeightTransportError("cos and sin must have identical row shapes")

        layers = _module_layers(model)
        if self.target_layer >= len(layers):
            raise ReadWeightTransportError("target_layer is outside model layers")
        if self.source_layer >= len(layers):
            raise ReadWeightTransportError("source_layer is outside model layers")

        source = layers[self.source_layer]
        source_norm = _copy_norm(
            getattr(source, "post_attention_layernorm", None),
            device=self.device,
            label=f"layer {self.source_layer} post_attention_layernorm",
        )
        source_mlp_module = getattr(source, "mlp", None)
        if source_mlp_module is None:
            raise ReadWeightTransportError(f"layer {self.source_layer} has no MLP")
        self._source_mlp = {
            "norm": source_norm,
            "gate": _copy_linear(source_mlp_module.gate_proj, device=self.device, label="source gate_proj"),
            "up": _copy_linear(source_mlp_module.up_proj, device=self.device, label="source up_proj"),
            "down": _copy_linear(source_mlp_module.down_proj, device=self.device, label="source down_proj"),
        }

        self._full: dict[int, dict[str, Any]] = {}
        for index in range(self.source_layer + 1, self.target_layer):
            layer = layers[index]
            attention = getattr(layer, "self_attn", None)
            mlp = getattr(layer, "mlp", None)
            if attention is None or mlp is None:
                raise ReadWeightTransportError(f"layer {index} is missing self_attn or mlp")
            geometry = _attention_geometry(attention, model, label=f"layer {index} attention")
            self._full[index] = {
                "input_norm": _copy_norm(
                    layer.input_layernorm, device=self.device, label=f"layer {index} input_layernorm"
                ),
                "q": _copy_linear(attention.q_proj, device=self.device, label=f"layer {index} q_proj"),
                "k": _copy_linear(attention.k_proj, device=self.device, label=f"layer {index} k_proj"),
                "v": _copy_linear(attention.v_proj, device=self.device, label=f"layer {index} v_proj"),
                "o": _copy_linear(attention.o_proj, device=self.device, label=f"layer {index} o_proj"),
                "post_norm": _copy_norm(
                    layer.post_attention_layernorm,
                    device=self.device,
                    label=f"layer {index} post_attention_layernorm",
                ),
                "gate": _copy_linear(mlp.gate_proj, device=self.device, label=f"layer {index} gate_proj"),
                "up": _copy_linear(mlp.up_proj, device=self.device, label=f"layer {index} up_proj"),
                "down": _copy_linear(mlp.down_proj, device=self.device, label=f"layer {index} down_proj"),
                "geometry": geometry,
            }

        target = layers[self.target_layer]
        target_attention = getattr(target, "self_attn", None)
        if target_attention is None:
            raise ReadWeightTransportError(f"layer {self.target_layer} has no self_attn")
        self._target_geometry = _attention_geometry(
            target_attention, model, label=f"layer {self.target_layer} attention"
        )
        target_heads, target_kv_heads, target_head_dim, _ = self._target_geometry
        self._target = {
            "norm": _copy_norm(
                target.input_layernorm, device=self.device, label=f"layer {self.target_layer} input_layernorm"
            ),
            "q": _copy_linear(target_attention.q_proj, device=self.device, label="target q_proj"),
            "k": _copy_linear(target_attention.k_proj, device=self.device, label="target k_proj"),
        }
        for index, block in self._full.items():
            heads, kv_heads, head_dim, _ = block["geometry"]
            if (heads, kv_heads, head_dim) != (target_heads, target_kv_heads, target_head_dim):
                raise ReadWeightTransportError("intermediate and target attention geometry differ")

        expected_layers = set(range(self.source_layer + 1, self.target_layer))
        if set(history) != expected_layers:
            raise ReadWeightTransportError(
                f"history must contain exactly intermediate layers {sorted(expected_layers)}"
            )
        self.history: dict[int, tuple[Any, Any]] = {}
        for index in sorted(expected_layers):
            key, value = history[index]
            key = _float_parameter(key, device=self.device)
            value = _float_parameter(value, device=self.device)
            if key.ndim == 3:
                key = key.unsqueeze(0)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if key.ndim != 4 or value.ndim != 4 or key.shape[0] != 1 or value.shape[0] != 1:
                raise ReadWeightTransportError(f"history layer {index} must be [1, kv_heads, rows, head_dim]")
            if key.shape != value.shape:
                raise ReadWeightTransportError(f"history layer {index} K/V shapes differ")
            _, kv_heads, head_dim, _ = self._full[index]["geometry"]
            if tuple(key.shape[1:]) != (kv_heads, key.shape[-2], head_dim):
                raise ReadWeightTransportError(f"history layer {index} has incompatible KV geometry")
            self.history[index] = (key, value)

        self.hidden_size = int(self.incoming.shape[-1])
        self.logical_linears = 3 + 7 * len(self._full) + 2

    def _full_block(self, index: int, hidden: Any) -> Any:
        torch, functional = _torch()
        block = self._full[index]
        heads, kv_heads, width, scaling = block["geometry"]
        normalized = _rms_norm(hidden, block["input_norm"], functional)
        query = _linear(functional, block["q"], normalized).view(1, 1, heads, width).transpose(1, 2)
        key = _linear(functional, block["k"], normalized).view(1, 1, kv_heads, width).transpose(1, 2)
        value = _linear(functional, block["v"], normalized).view(1, 1, kv_heads, width).transpose(1, 2)
        query = _rope(query, self.cos, self.sin)
        key = _rope(key, self.cos, self.sin)
        old_key, old_value = self.history[index]
        if old_key.shape[1] != kv_heads or old_key.shape[-1] != width:
            raise ReadWeightTransportError(f"history layer {index} has incompatible KV geometry")
        key_bank = torch.cat((old_key, key), dim=2)
        value_bank = torch.cat((old_value, value), dim=2)
        repeated_key = _repeat_kv(key_bank, heads // kv_heads)
        repeated_value = _repeat_kv(value_bank, heads // kv_heads)
        scores = torch.matmul(query, repeated_key.transpose(-2, -1)) * scaling
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, repeated_value).transpose(1, 2).reshape(1, 1, heads * width)
        attention_write = _linear(functional, block["o"], attended)
        residual = hidden + attention_write
        mlp_input = _rms_norm(residual, block["post_norm"], functional)
        mlp_write = _swi_glu(mlp_input, block, functional)
        return residual + mlp_write

    def _shadow_impl(self, attention19: Any) -> tuple[Any, Any]:
        torch, functional = _torch()
        residual = self.incoming + attention19
        normalized = _rms_norm(residual, self._source_mlp["norm"], functional)
        mlp_write = _swi_glu(normalized, self._source_mlp, functional)
        hidden = residual + mlp_write
        for index in sorted(self._full):
            hidden = self._full_block(index, hidden)
        target_norm = _rms_norm(hidden, self._target["norm"], functional)
        heads, kv_heads, width, _ = self._target_geometry
        query = _linear(functional, self._target["q"], target_norm).view(1, 1, heads, width).transpose(1, 2)
        key = _linear(functional, self._target["k"], target_norm).view(1, 1, kv_heads, width).transpose(1, 2)
        return _rope(query, self.cos, self.sin), _rope(key, self.cos, self.sin)

    def shadow(self, attention19: Any) -> tuple[Any, Any]:
        """Return smooth target q/current-k for an arbitrary source attention row."""

        torch, _ = _torch()
        value = _float_input(attention19, device=self.device, label="attention19")
        if value.shape != self.incoming.shape:
            raise ReadWeightTransportError("attention19 must match incoming shape")
        with torch.inference_mode(False):
            return self._shadow_impl(value)

    def compile_direction(self, write_delta: Any) -> dict[str, Any]:
        """Compile a baseline tangent with ``torch.func.jvp``.

        The returned tensors are detached FP32 artifacts.  The derivative is a
        smooth lifted derivative around ``baseline_attention.float()`` and is
        therefore diagnostic relative to the later native BF16 intervention.
        """

        torch, _ = _torch()
        direction = _float_input(write_delta, device=self.device, label="write_delta")
        if direction.shape != self.baseline_attention.shape:
            raise ReadWeightTransportError("write_delta must match baseline_attention shape")
        start = time.perf_counter()
        with torch.inference_mode(False):
            base = self.baseline_attention.detach().clone().requires_grad_(True)
            tangent = direction.detach().clone()
            try:
                primal, derivative = torch.func.jvp(self._shadow_impl, (base,), (tangent,))
            except AttributeError as exc:  # pragma: no cover - old torch diagnostic
                raise ReadWeightTransportError("torch.func.jvp is required for transport compilation") from exc
        q0, k0 = primal
        q_dot, k_dot = derivative
        return {
            "q_dot": q_dot.detach().clone(),
            "k_dot": k_dot.detach().clone(),
            "shadow_q0": q0.detach().clone(),
            "shadow_k0": k0.detach().clone(),
            "logical_linears": self.logical_linears,
            "logical_cost": {
                "source_mlp_linears": 3,
                "intermediate_full_block_linears": 7 * len(self._full),
                "target_qk_linears": 2,
                "total_dense_linears": self.logical_linears,
                "native_forward_calls": 0,
            },
            "host_enqueue_seconds": float(time.perf_counter() - start),
        }

    def compile_score_program(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return compile_score_program(*args, **kwargs)

    def evaluate_score_program(self, program: Mapping[str, Any], dose: float, *, method: str = "transport"):
        return evaluate_score_program(program, dose, method=method)


def compile_score_program(
    q0_native: Any,
    k_full_native: Any,
    native_score0: Any,
    q_dot: Any,
    k_dot: Any,
    *,
    kv_head: int,
    source_row: int,
    scaling: float,
) -> dict[str, Any]:
    """Compile per-row attention-logit coefficients from a baseline tape.

    ``native_score0`` is trusted as the widened result of the native BF16
    q/k matmul and scaling.  It is intentionally not recomputed from the
    smooth shadow tensors.
    """

    torch, _ = _torch()
    q0 = _canonical_q(q0_native, label="q0_native").float()
    k_full = _canonical_k(k_full_native, label="k_full_native").float()
    scores = _canonical_scores(native_score0, label="native_score0").float()
    q_tangent = _canonical_q(q_dot, label="q_dot").float()
    k_tangent = _canonical_k(k_dot, label="k_dot").float()
    if not all(t.device == q0.device for t in (k_full, scores, q_tangent, k_tangent)):
        raise ReadWeightTransportError("score-program tensors must share a device")
    if q_tangent.shape != q0.shape or k_tangent.shape != k_full.shape[:2] + (1, k_full.shape[-1]):
        raise ReadWeightTransportError("q_dot/k_dot shapes do not match native q/k geometry")
    if q0.shape[1] != scores.shape[1] or q0.shape[-1] != k_full.shape[-1]:
        raise ReadWeightTransportError("native q/k/score geometry is inconsistent")
    if scores.shape[-1] != k_full.shape[-2]:
        raise ReadWeightTransportError("native scores do not cover the full key bank")
    if not isinstance(scaling, (int, float)) or not math.isfinite(float(scaling)) or float(scaling) <= 0:
        raise ReadWeightTransportError("scaling must be a finite positive number")
    kv_head = _plain_int(kv_head, "kv_head")
    source_row = _plain_int(source_row, "source_row")
    kv_heads = int(k_full.shape[1])
    if kv_head >= kv_heads:
        raise ReadWeightTransportError("kv_head is outside k_full_native")
    rows = int(k_full.shape[-2])
    if source_row >= rows:
        raise ReadWeightTransportError("source_row is outside k_full_native")
    query_heads = int(q0.shape[1])
    if query_heads % kv_heads:
        raise ReadWeightTransportError("query heads are not divisible by KV heads")
    repeats = query_heads // kv_heads
    first = kv_head * repeats
    last = first + repeats
    q_group = q0[:, first:last, 0, :]
    q_dot_group = q_tangent[:, first:last, 0, :]
    key_group = k_full[:, kv_head, :, :]
    current_key_dot = k_tangent[:, kv_head, 0, :]
    query_coefficients = torch.einsum("bhd,btd->bht", q_dot_group, key_group) * float(scaling)
    current_key_coefficients = torch.einsum("bhd,bd->bh", q_group, current_key_dot) * float(scaling)
    cross_coefficients = torch.einsum("bhd,bd->bh", q_dot_group, current_key_dot) * float(scaling)
    current_row = rows - 1
    coefficients = query_coefficients.clone()
    coefficients[..., current_row] += current_key_coefficients
    logp0 = torch.log_softmax(scores[:, first:last, 0, :], dim=-1)
    return {
        "schema": "mdb-smooth-qwen-score-program-v1",
        "logp0": logp0.detach().clone(),
        "a": coefficients.detach().clone(),
        "a_query": query_coefficients.detach().clone(),
        "b": cross_coefficients.detach().clone(),
        "b_current": cross_coefficients.detach().clone(),
        "kv_head": kv_head,
        "query_heads": tuple(range(first, last)),
        "source_row": source_row,
        "current_row": current_row,
        "scaling": float(scaling),
    }


def evaluate_score_program(
    program: Mapping[str, Any], dose: float, *, method: str = "transport"
) -> dict[str, Any]:
    """Evaluate a compiled score program without dense q/k products."""

    torch, _ = _torch()
    if not isinstance(program, Mapping) or program.get("schema") != "mdb-smooth-qwen-score-program-v1":
        raise ReadWeightTransportError("program has an unsupported score-program schema")
    if method not in {"transport", "linear-logweight", "stale", "wrong-sign", "query-only"}:
        raise ReadWeightTransportError(f"unknown score-program method {method!r}")
    if isinstance(dose, torch.Tensor):
        if dose.numel() != 1:
            raise ReadWeightTransportError("dose must be scalar")
        dose = float(dose.detach().item())
    if not isinstance(dose, (int, float)) or isinstance(dose, bool) or not math.isfinite(float(dose)):
        raise ReadWeightTransportError("dose must be finite")
    dose = float(dose)
    logp0 = program["logp0"]
    a = program["a"]
    if not _is_tensor(logp0) or not _is_tensor(a):
        raise ReadWeightTransportError("program is missing tensor coefficients")
    logp0 = logp0.float()
    a = a.float()
    b = program["b"].float()
    source_row = _plain_int(program["source_row"], "program source_row")
    current_row = _plain_int(program["current_row"], "program current_row")
    if source_row >= logp0.shape[-1] or current_row >= logp0.shape[-1]:
        raise ReadWeightTransportError("program row is outside logp0")

    # Preserve the baseline exactly at the anchor; this also avoids a needless
    # log-softmax round trip changing the native-baseline receipt by one ulp.
    if dose == 0.0 or method == "stale":
        delta_logp = torch.zeros_like(logp0[..., source_row])
        pred_logp = logp0[..., source_row].clone()
    elif method == "linear-logweight":
        expectation = (logp0.exp() * a).sum(dim=-1)
        delta_logp = dose * (a[..., source_row] - expectation)
        pred_logp = logp0[..., source_row] + delta_logp
    else:
        coefficients = program["a_query"].float() if method == "query-only" else a
        alpha = -dose if method == "wrong-sign" else dose
        delta_logits = alpha * coefficients
        if method != "query-only":
            delta_logits = delta_logits.clone()
            delta_logits[..., current_row] += (alpha * alpha) * b
        logits = logp0 + delta_logits
        pred_logp_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        pred_logp = pred_logp_all[..., source_row]
        delta_logp = pred_logp - logp0[..., source_row]
    probability = pred_logp.exp()
    return {
        "method": method,
        "dose": dose,
        "pred_logpB": pred_logp,
        "logpB": pred_logp,
        "delta_logpB": delta_logp,
        "pB": probability,
    }


__all__ = [
    "ReadWeightTransportError",
    "SmoothQwenReadTransport",
    "compile_score_program",
    "evaluate_score_program",
]

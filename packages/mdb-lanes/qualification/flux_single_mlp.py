"""Entry-state-only sampled single6 MLP forecast and later native witnesses.

The numerical predictor imports only NumPy. The scoped native witness hooks are
a separate API and must be installed after the outer early-MLP forecast seal.
"""
from __future__ import annotations

from contextlib import contextmanager


def mlp_weight_row_indices(*, hidden_width, mlp_width, neuron_indices):
    if (type(hidden_width) is not int or hidden_width < 1 or type(mlp_width) is not int or mlp_width < 1
            or not neuron_indices or len(set(neuron_indices)) != len(neuron_indices)
            or any(type(index) is not int or not 0 <= index < mlp_width for index in neuron_indices)):
        raise ValueError("invalid native MLP row geometry/selection")
    return [3 * hidden_width + index for index in neuron_indices] + [3 * hidden_width + mlp_width + index for index in neuron_indices]


def predict_entry_samples(residual_image, single_mod, weight_rows, bias_rows, *, eps, query_indices, neuron_indices, hidden_width, mlp_width):
    """Predict selected MLP neurons from current pre6 image residual and weights.

    ``weight_rows`` contains first-half u rows followed by paired v rows from
    ``attn.to_qkv_mlp_proj``. Native BF16 values enter as exact FP32 widenings.
    No observed norm, modulated value, preactivation or output is an argument.
    """
    import numpy as np

    rows = mlp_weight_row_indices(hidden_width=hidden_width, mlp_width=mlp_width, neuron_indices=neuron_indices)
    if (residual_image.ndim != 3 or residual_image.shape[0] != 1 or residual_image.shape[2] != hidden_width
            or single_mod.shape not in {(1, 3 * hidden_width), (1, 1, 3 * hidden_width)}
            or weight_rows.shape != (len(rows), hidden_width) or (bias_rows is not None and bias_rows.shape != (len(rows),))
            or type(eps) not in {int, float} or not np.isfinite(eps) or eps <= 0
            or not query_indices or len(set(query_indices)) != len(query_indices)
            or any(type(index) is not int or not 0 <= index < residual_image.shape[1] for index in query_indices)
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (residual_image, single_mod, weight_rows))
            or (bias_rows is not None and (bias_rows.dtype != np.float32 or not np.isfinite(bias_rows).all()))):
        raise ValueError("invalid entry MLP source/weights/epsilon/query profile")
    def readonly(value):
        value = np.asarray(value)
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    def bf16(value):
        bits = np.asarray(value, dtype=np.float32).view(np.uint32)
        rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)
        return rounded.view(np.float32)
    def silu(value):
        # Stable both tails without clipping the source values.
        value = np.asarray(value, dtype=np.float64)
        decay = np.exp(-np.abs(value))
        return np.where(value >= 0, value / (1 + decay), value * decay / (1 + decay))
    x = np.array(residual_image[:, query_indices], dtype=np.float64, copy=True)
    shift, scale, _gate = np.split(np.asarray(single_mod, dtype=np.float64).reshape(1, 1, 3 * hidden_width), 3, axis=-1)
    centered = x - x.mean(axis=-1, keepdims=True)
    norm = centered / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + eps)
    modulated = (1 + scale) * norm + shift
    projection = modulated @ weight_rows.astype(np.float64).T
    if bias_rows is not None:
        projection += bias_rows.astype(np.float64)
    u, v = np.split(projection, 2, axis=-1)
    analytic = {"input_samples": x, "layernorm": norm, "modulated": modulated, "u": u, "v": v,
        "silu_u": silu(u), "activation": silu(u) * v}
    norm_b, scale_b = bf16(norm), bf16(1 + scale)
    product_b = bf16(scale_b * norm_b)
    modulated_b = bf16(product_b.astype(np.float64) + shift)
    projection_b = modulated_b.astype(np.float64) @ weight_rows.astype(np.float64).T
    if bias_rows is not None:
        projection_b += bias_rows.astype(np.float64)
    u_b, v_b = np.split(bf16(projection_b), 2, axis=-1)
    silu_b = bf16(silu(u_b))
    reference = {"layernorm": norm_b, "one_plus_scale": scale_b, "mod_product": product_b, "modulated": modulated_b,
        "u": u_b, "v": v_b, "silu_u": silu_b, "activation": bf16(silu_b * v_b)}
    if any(not np.isfinite(value).all() for family in (analytic, reference) for value in family.values()):
        raise ValueError("nonfinite sampled early MLP forecast")
    return {"query_indices": list(query_indices), "neuron_indices": list(neuron_indices), "weight_row_indices": rows,
        "eps": float(eps), "hidden_width": hidden_width, "mlp_width": mlp_width,
        "analytic": {key: readonly(value) for key, value in analytic.items()},
        "bf16_reference": {key: readonly(value) for key, value in reference.items()},
        "arithmetic": "FP64 biased-variance LayerNorm(no affine), modulation, paired fused linear rows and SiLU(u)*v. BF16 reference: B(norm64), B(1+scale), B(product), B(+shift), FP64 linear on quantized modulation then B(u/v), B(SiLU(Bu)), B(product). B is FP64→FP32→RNE-BF16 widened FP32; reference is not an exact CUDA LayerNorm/GEMM/SiLU emulator.",
        "scope": "Current pre6 entry residual, modulation and declared actual weight rows only; no native current-site norm, MLP or attention outputs enter prediction"}


@contextmanager
def capture_source_calls(block, *, text_rows, query_indices, neuron_indices, source_order=("zero", "text")):
    """Observe first two later native block calls; never substitute their values.

    Install only after source construction and the outer early forecast seal.
    The caller owns the fixed zero/text call order and binds all samples to the
    current-step single6 source cuts. Earlier-timestep module6 calls are allowed
    in the separately counted builder, while these hooks are not installed.
    """
    import torch

    width, mlp_width = block.attn.query_dim, block.attn.mlp_hidden_dim
    mlp_weight_row_indices(hidden_width=width, mlp_width=mlp_width, neuron_indices=neuron_indices)
    if (block.training or block.norm.elementwise_affine or not block.norm.eps > 0
            or type(text_rows) is not int or text_rows < 1 or tuple(source_order) != ("zero", "text")
            or not query_indices or len(set(query_indices)) != len(query_indices)
            or any(type(index) is not int or index < 0 for index in query_indices)):
        raise ValueError("unsupported scoped native MLP witness profile")
    state, active, handles = {"rows": {}, "calls": 0}, {"role": None}, []
    positions = [text_rows + index for index in query_indices]
    def before(module, args, kwargs):
        if active["role"] is not None:
            raise ValueError("overlapping native block calls in scoped MLP witness")
        index = state["calls"]
        state["calls"] += 1
        if index < 2:
            value = kwargs.get("hidden_states", args[0] if args else None)
            if (value is None or value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != width
                    or max(positions) >= value.shape[1] or kwargs.get("encoder_hidden_states") is not None):
                raise ValueError("native source block sample geometry differs")
            role = source_order[index]
            active["role"] = role
            state["rows"][role] = {"call_index": index, "query_indices": list(query_indices), "neuron_indices": list(neuron_indices),
                "layernorm_eps": float(block.norm.eps), "layernorm_affine": False}
    def sample(name, value):
        if active["role"] is not None:
            row = state["rows"][active["role"]]
            if name in row or not bool(torch.isfinite(value).all()):
                raise ValueError("repeated/nonfinite native MLP witness")
            row[name] = value.detach().clone()
    def normalized(module, args, value):
        if active["role"] is not None:
            sample("layernorm", value[:, positions])
    def modulated(module, args):
        if active["role"] is not None:
            sample("modulated_input", args[0][:, positions])
    def preactivation(module, args):
        if active["role"] is not None:
            value = args[0][:, positions]
            sample("u", value[:, :, neuron_indices])
            sample("v", value[:, :, [mlp_width + index for index in neuron_indices]])
    def activation(module, args, value):
        if active["role"] is not None:
            sample("activation", value[:, positions][:, :, neuron_indices])
    def after(module, args, kwargs, value):
        if active["role"] is not None and value is not None:
            if not {"layernorm", "modulated_input", "u", "v", "activation"} <= set(state["rows"][active["role"]]):
                raise ValueError("native MLP source witness is incomplete")
        active["role"] = None
    try:
        handles = [block.register_forward_pre_hook(before, with_kwargs=True),
            block.norm.register_forward_hook(normalized), block.attn.to_qkv_mlp_proj.register_forward_pre_hook(modulated),
            block.attn.mlp_act_fn.register_forward_pre_hook(preactivation), block.attn.mlp_act_fn.register_forward_hook(activation),
            block.register_forward_hook(after, with_kwargs=True, always_call=True)]
        yield state
    finally:
        for handle in handles:
            handle.remove()


__all__ = ["mlp_weight_row_indices", "predict_entry_samples", "capture_source_calls"]

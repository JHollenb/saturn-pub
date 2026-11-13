"""Independent NumPy FP64 equations for the frozen R3 final suffix.

This is a full-weight reference, not a compressed surrogate. The caller owns the
prefix observations, stages FP32 intervention payloads, and seals forecasts
before native consumers run. No model, Torch, or native consumer is imported.
The reference uses a sequential selective recurrence; native R3 uses doubling.
Their finite-precision discrepancy is a retained diagnostic, not an exactness
claim. Supplied overrides are widened to FP64 without another FP32 rounding.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

EPSILON = 1e-5
CHECKPOINT = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"
SOURCE_FORMULAS = {
    "direct_effect/model.py": "202317979d86d043d19a382a5f9f4b0120c7fee7333e53b1c29dd701f3861cc8",
    "isa_lab/ssm.py": "319f942903166039e3f6d25a82b38f87412054ea18bf31e19c77554af6998573",
}


def required_weight_names():
    names = ["mixers.2.a_log", "mixers.2.d", "mixers.2.conv.weight", "mixers.2.conv.bias",
             "mixers.2.select.weight", "pos", "post.2.0.weight", "post.2.0.bias"]
    for module in ("mixers.2.in_proj", "mixers.2.dt_proj", "mixers.2.out_proj",
                   "post.2.1", "post.2.3", "reg_out", "imm_out", "flag_query", "flags.0", "flags.2",
                   "dest", "src", "width_head", "length", "syscall", "store_flag", "store_out"):
        names.extend((module + ".weight", module + ".bias"))
    return tuple(sorted(names))


def _array(value, name, shape=None):
    original = np.asarray(value)
    if original.dtype.kind not in "fiu" or original.dtype.kind == "b":
        raise ValueError(name + " must be a real numeric array")
    array = np.asarray(original, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(name + " must be finite")
    if shape is not None and array.shape != tuple(shape):
        raise ValueError(f"{name} shape {array.shape} differs from {tuple(shape)}")
    return array


def weight_manifest(weights):
    """Describe all supplied original arrays without rewriting their dtype."""
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a state-dict mapping of owned NumPy arrays")
    missing = sorted(set(required_weight_names()) - set(weights))
    if missing:
        raise ValueError("missing suffix parameters: " + ", ".join(missing))
    result = {}
    for name in sorted(weights):
        value = np.asarray(weights[name])
        _array(value, "weight " + name)
        packed = np.ascontiguousarray(value)
        result[name] = {"shape": list(packed.shape), "dtype": packed.dtype.str,
                        "sha256": hashlib.sha256(packed.tobytes(order="C")).hexdigest()}
    return result


def formula_manifest():
    return {"schema": "mdb-explicit-r3-suffix-equations-v1", "checkpoint": CHECKPOINT,
        "source_formulas": dict(SOURCE_FORMULAS), "parameters": list(required_weight_names()),
        "arithmetic": "NumPy FP64; erf GELU; biased LayerNorm variance; epsilon1e-5",
        "recurrence": "sequential H[t]=exp(dt[t]*A)*H[t-1]+dt[t]*u[t]*B[t], H[-1]=0",
        "native_difference": "native Torch FP32 doubling scan and kernel reduction order",
        "payloads": "caller stages exact FP32 payloads; reference only widens supplied overrides",
        "heads": ["regs", "flags", "dest", "src", "width", "length", "syscall", "imm",
                  "store", "store_addr", "store_value"],
        "consumer_horizon": "one suffix; own-state successor requires a separately observed prefix",
        "semantic_authority": "supplied equations and weights; no ISA semantic name inferred"}


def prepare_weights(weights):
    """Own immutable FP64 suffix arrays; full original manifest is caller data."""
    if not isinstance(weights, Mapping):
        raise ValueError("weights must be a state-dict mapping")
    missing = sorted(set(required_weight_names()) - set(weights))
    if missing:
        raise ValueError("missing suffix parameters: " + ", ".join(missing))
    result = {}
    for name in required_weight_names():
        value = _array(weights[name], "weight " + name).copy()
        value.flags.writeable = False
        result[name] = value
    return result


def _weight(weights, name, shape=None):
    if name not in weights:
        raise ValueError("missing suffix parameter: " + name)
    return _array(weights[name], "weight " + name, shape)


def linear(value, weights, module):
    value = _array(value, module + " input")
    weight = _weight(weights, module + ".weight")
    if weight.ndim != 2 or value.ndim == 0 or value.shape[-1] != weight.shape[1]:
        raise ValueError("affine geometry differs: " + module)
    bias = _weight(weights, module + ".bias", (weight.shape[0],))
    return value @ weight.T + bias


def sigmoid(value):
    value = _array(value, "sigmoid input")
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1. / (1. + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    result[~positive] = exponent / (1. + exponent)
    return result


def silu(value):
    value = _array(value, "SiLU input")
    return value * sigmoid(value)


def gelu(value):
    value = _array(value, "GELU input")
    error = np.fromiter((math.erf(float(item) / math.sqrt(2.)) for item in value.flat),
                        dtype=np.float64, count=value.size).reshape(value.shape)
    return value * .5 * (1. + error)


def softmax(value):
    value = _array(value, "softmax input")
    shifted = value - value.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=-1, keepdims=True)


def causal_depthwise_conv(value, weight, bias):
    """Torch cross-correlation with left padding kernel-1, then truncate."""
    value, weight, bias = (_array(item, name) for item, name in (
        (value, "convolution input"), (weight, "convolution weight"), (bias, "convolution bias")))
    if value.ndim != 3 or weight.ndim != 3 or weight.shape[:2] != (value.shape[2], 1):
        raise ValueError("causal depthwise convolution geometry differs")
    channels, kernel = value.shape[2], weight.shape[2]
    if kernel < 1 or bias.shape != (channels,):
        raise ValueError("causal convolution kernel/bias differs")
    result = np.broadcast_to(bias, value.shape).copy()
    for tap in range(kernel):
        lag = kernel - 1 - tap
        if lag < value.shape[1]:
            result[:, lag:, :] += value[:, :value.shape[1] - lag, :] * weight[:, 0, tap]
    return result


def selective_recurrence(u, dt, a, b, c):
    """Return every independent sequential state and its native C read."""
    u, dt, a, b, c = (_array(item, name) for item, name in (
        (u, "scan u"), (dt, "scan dt"), (a, "scan A"), (b, "scan B"), (c, "scan C")))
    if (u.ndim != 3 or dt.shape != u.shape or a.ndim != 2 or a.shape[0] != u.shape[2]
            or b.shape != (u.shape[0], u.shape[1], a.shape[1]) or c.shape != b.shape):
        raise ValueError("selective recurrence geometry differs")
    decay = np.exp(dt[..., None] * a)
    writer = dt[..., None] * u[..., None] * b[:, :, None, :]
    states = np.empty_like(writer)
    previous = np.zeros((u.shape[0], u.shape[2], a.shape[1]), dtype=np.float64)
    for position in range(u.shape[1]):
        previous = decay[:, position] * previous + writer[:, position]
        states[:, position] = previous
    read = (states * c[:, :, None, :]).sum(axis=-1)
    if not all(np.isfinite(item).all() for item in (decay, writer, states, read)):
        raise ValueError("nonfinite explicit selective recurrence")
    return {"decay": decay, "writer": writer, "H": states, "weighted_read": read}


def post_write(carrier, weights, *, return_stages=False):
    carrier = _array(carrier, "post2 carrier")
    if carrier.ndim != 3:
        raise ValueError("post2 carrier must have batch/position/channel axes")
    width = carrier.shape[-1]
    mean = carrier.mean(axis=-1, keepdims=True)
    variance = ((carrier - mean) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1. / np.sqrt(variance + EPSILON)
    normalized = ((carrier - mean) * rstd * _weight(weights, "post.2.0.weight", (width,))
                  + _weight(weights, "post.2.0.bias", (width,)))
    affine = linear(normalized, weights, "post.2.1")
    activated = gelu(affine)
    write = linear(activated, weights, "post.2.3")
    if write.shape != carrier.shape:
        raise ValueError("post2 write geometry differs from carrier")
    stages = {"post2_mean": mean, "post2_variance": variance, "post2_rstd": rstd,
              "post2_norm": normalized, "post2_affine": affine, "post2_gelu": activated,
              "P2_computed": write}
    return stages if return_stages else write


def _head_results(h, z, reg_bits, flags, weights):
    batch, positions, width = h.shape
    if (z.shape != (batch, 256) or reg_bits.shape != (batch, 8, positions)
            or flags.shape != (batch, 4) or positions != 64 or width != 48):
        raise ValueError("R3 suffix/head ABI differs from batchx64x48, z256, eight64-bit registers, four flags")
    reg_write = linear(h, weights, "reg_out")
    prior = reg_bits * 8. - 4.
    query = linear(z, weights, "flag_query").reshape(batch, 4, 16)
    pos = _weight(weights, "pos", (positions, 16))
    attention_logits = np.einsum("bkd,id->bki", query, pos)
    attention = softmax(attention_logits)
    read = np.einsum("bki,biw->bkw", attention, h)
    pooled = h.mean(axis=1)
    flag_input = np.concatenate((z, flags, read.reshape(batch, 4 * width), pooled), axis=-1)
    flag_affine = linear(flag_input, weights, "flags.0")
    flag_hidden = gelu(flag_affine)
    flag_write = linear(flag_hidden, weights, "flags.2")
    stores = linear(h, weights, "store_out")
    heads = {"regs": prior + reg_write.transpose(0, 2, 1),
             "flags": flag_write + (flags * 8. - 4.),
             "dest": linear(z, weights, "dest"), "src": linear(z, weights, "src"),
             "width": linear(z, weights, "width_head"), "length": linear(z, weights, "length"),
             "syscall": linear(z, weights, "syscall")[..., 0],
             "imm": linear(h, weights, "imm_out")[..., 0],
             "store": linear(z, weights, "store_flag")[..., 0],
             "store_addr": stores[..., 0], "store_value": stores[..., 1]}
    stages = {"flag_query": query, "flag_attention_logits": attention_logits,
              "flag_attention": attention, "flag_read": read, "flag_mean": pooled,
              "flag_input": flag_input, "flag_affine": flag_affine,
              "flag_gelu": flag_hidden, "flag_write": flag_write}
    return heads, stages


def predict_heads(h, z, reg_bits, flags, weights):
    """Evaluate every supplied native readout equation from final h and z."""
    arrays = [_array(item, name) for item, name in ((h, "h"), (z, "z"),
              (reg_bits, "register roots"), (flags, "flag roots"))]
    return _head_results(*arrays, weights)[0]


def predict_suffix(state, z, reg_bits, flags, weights, *, projection_override=None,
                   carrier_override=None, post_override=None):
    """Predict the full final mixer/MLP/native heads with explicit substitutions.

    ``state`` is the final mixer's input S, not its residual output M. Overrides
    replace U, M, or P2 respectively; all equations after that boundary consume
    the exact supplied payload, widened to FP64. Inputs and weights are not
    mutated. Returned U/M/P2 are the effective values actually consumed.
    """
    state = _array(state, "S")
    if state.ndim != 3 or state.shape[1:] != (64, 48):
        raise ValueError("S must have shape [batch,64,48]")
    batch, _, width = state.shape
    z = _array(z, "z", (batch, 256))
    reg_bits = _array(reg_bits, "register roots", (batch, 8, 64))
    flags = _array(flags, "flag roots", (batch, 4))
    if not np.isin(reg_bits, (0., 1.)).all() or not np.isin(flags, (0., 1.)).all():
        raise ValueError("raw register and flag roots must contain binary bits")
    rms = state / np.sqrt((state ** 2).mean(axis=-1, keepdims=True) + EPSILON)
    projected = linear(rms, weights, "mixers.2.in_proj")
    if projected.shape[-1] % 2:
        raise ValueError("mixer input projection must split into u and gate equally")
    u_preconv, gate = np.split(projected, 2, axis=-1)
    inner = u_preconv.shape[-1]
    convolved = causal_depthwise_conv(u_preconv, _weight(weights, "mixers.2.conv.weight"),
                                     _weight(weights, "mixers.2.conv.bias"))
    u = silu(convolved)
    selection_weight = _weight(weights, "mixers.2.select.weight")
    if selection_weight.ndim != 2 or selection_weight.shape[1] != inner or (selection_weight.shape[0] - 1) % 2:
        raise ValueError("selective factor projection geometry differs")
    selected = u @ selection_weight.T
    state_width = (selected.shape[-1] - 1) // 2
    if state_width < 1:
        raise ValueError("selective state dimension must be positive")
    delta, b, c = np.split(selected, (1, 1 + state_width), axis=-1)
    dt_affine = linear(delta, weights, "mixers.2.dt_proj")
    if dt_affine.shape != u.shape:
        raise ValueError("dt projection differs from selective channels")
    dt = np.logaddexp(0., dt_affine)
    a = -np.exp(_weight(weights, "mixers.2.a_log", (inner, state_width)))
    recurrence = selective_recurrence(u, dt, a, b, c)
    local = u * _weight(weights, "mixers.2.d", (inner,))
    activated_gate = silu(gate)
    gated = (recurrence["weighted_read"] + local) * activated_gate
    computed_u = linear(gated, weights, "mixers.2.out_proj")
    if computed_u.shape != state.shape:
        raise ValueError("projected final mixer write differs from S geometry")
    effective_u = computed_u if projection_override is None else _array(projection_override, "U override", state.shape)
    computed_m = state + effective_u
    effective_m = computed_m if carrier_override is None else _array(carrier_override, "M override", state.shape)
    post_stages = post_write(effective_m, weights, return_stages=True)
    computed_p = post_stages["P2_computed"]
    effective_p = computed_p if post_override is None else _array(post_override, "P2 override", state.shape)
    h = effective_m + effective_p
    heads, head_stages = _head_results(h, z, reg_bits, flags, weights)
    reg_weight = _weight(weights, "reg_out.weight", (8, width))
    direct = lambda value: (value @ reg_weight.T).transpose(0, 2, 1)
    bias = np.broadcast_to(_weight(weights, "reg_out.bias", (8,))[None, :, None], (batch, 8, 64)).copy()
    # A carrier override need not equal S+U. Keep its signed correction explicit
    # rather than assigning that change to either historical SSA operand.
    correction = direct(effective_m - computed_m)
    terms = {"prior": reg_bits * 8. - 4., "skip": direct(state),
             "projection": direct(effective_u), "carrier_override": correction,
             "post": direct(effective_p), "bias": bias}
    stages = {"S": state.copy(), "rms": rms, "in_projected": projected,
              "u_preconv": u_preconv, "convolved": convolved, "u": u, "gate_raw": gate,
              "G": activated_gate, "selected": selected, "delta": delta, "B": b, "C": c,
              "dt_affine": dt_affine, "dt": dt, "A": a, **recurrence,
              "local_bypass": local, "gated": gated, "U_computed": computed_u, "U": effective_u.copy(),
              "M_computed": computed_m, "M": effective_m.copy(), **post_stages,
              "P2": effective_p.copy(), "h": h, **head_stages}
    for name, array in {**heads, **stages, **terms}.items():
        if not np.isfinite(array).all():
            raise ValueError("nonfinite explicit suffix result: " + name)
    return {"heads": heads, "stages": stages, "register_terms": terms,
            "overrides": {"projection": projection_override is not None,
                          "carrier": carrier_override is not None, "post": post_override is not None}}


def self_test():
    """Small hand-derived mechanics; no model, checkpoint, or native forward."""
    u = np.array([[[2.], [3.]]])
    dt = np.ones_like(u)
    scan = selective_recurrence(u, dt, np.array([[-math.log(2.)]]),
                               np.ones((1, 2, 1)), np.ones((1, 2, 1)))
    np.testing.assert_allclose(scan["H"].reshape(-1), [2., 4.], rtol=0., atol=1e-15)
    np.testing.assert_allclose(scan["weighted_read"].reshape(-1), [2., 4.], rtol=0., atol=1e-15)
    conv = causal_depthwise_conv(np.array([[[1.], [2.], [3.], [4.]]]),
                                np.array([[[1., 10., 100.]]]), np.array([0.]))
    np.testing.assert_array_equal(conv.reshape(-1), [100., 210., 321., 432.])
    np.testing.assert_allclose(softmax(np.array([[1000., 1000.]])), [[.5, .5]], rtol=0., atol=0.)
    np.testing.assert_array_equal(gelu(np.array([0.])), [0.])
    if not np.isfinite(silu(np.array([-1000., 1000.]))).all():
        raise AssertionError("stable SiLU mechanics differ")
    return {"status": "passed", "hand_scan_states": [2., 4.],
            "hand_causal_convolution": [100., 210., 321., 432.], "model_forward_calls": 0}


if __name__ == "__main__":
    print(json.dumps({"self_test": self_test(), "formula_manifest": formula_manifest()}, sort_keys=True))

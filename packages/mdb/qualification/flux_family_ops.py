"""Bounded attention contrasts and prefix-only proposal features.

Qualification helpers, not model execution or causal-rule induction. Torch and
Diffusers load only when a tensor helper is called. Native attention dispatch is
shared explicitly; input Q/K are already positioned and normalized native ports.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensors(values, *, ndim):
    import torch

    first = values[0]
    _require(isinstance(first, torch.Tensor) and first.ndim == ndim and first.is_floating_point()
             and all(size > 0 for size in first.shape), "expected nonempty floating tensor geometry")
    for value in values:
        _require(isinstance(value, torch.Tensor) and value.shape == first.shape and value.dtype == first.dtype
                 and value.device == first.device and bool(torch.isfinite(value).all()),
                 "tensor shape/dtype/device/finite-value mismatch")


def _indices(value, size, name):
    if value is None:
        return list(range(size))
    _require(type(value) is list and all(type(index) is int and 0 <= index < size for index in value)
             and len(value) == len(set(value)), name + " must contain unique in-bounds integer indices")
    return value


def attention_variant(*, base, changed, text_rows, recipe):
    """Select independent positioned Q/K/V sources, then native attention.

    base/changed map q,k,v to [B,text+image,H,D] tensors. Recipe source fields
    q,k_text,v_text,k_image,v_image default to 'changed'. Optional heads,
    text_positions and image_positions lists restrict replacements; positions
    are relative to the named partition. Missing lists select all, [] selects
    none. Q selection covers both partitions; only image query outputs are used
    by the declared consumer. Text query outputs are returned but may be dropped.
    Returns owned q/k/v tensors and full flattened [B,S,H*D] attention. Inputs
    remain unchanged; the returned attention shares native dispatch kernels.
    """
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    fields = {"q", "k_text", "v_text", "k_image", "v_image"}
    _require(isinstance(base, Mapping) and isinstance(changed, Mapping)
             and {"q", "k", "v"} <= base.keys() and {"q", "k", "v"} <= changed.keys(), "q/k/v operand mappings required")
    _require(isinstance(recipe, Mapping) and not set(recipe) - fields - {"heads", "text_positions", "image_positions"}, "unknown attention recipe fields")
    _require(all(type(recipe.get(name, "changed")) is str and recipe.get(name, "changed") in {"base", "changed"}
                 for name in fields), "attention sources must be base or changed")
    _tensors([source[key] for source in (base, changed) for key in ("q", "k", "v")], ndim=4)
    _, sequence, heads, _ = changed["q"].shape
    _require(type(text_rows) is int and 0 <= text_rows <= sequence, "invalid text row count")
    selected_heads = _indices(recipe.get("heads"), heads, "heads")
    text = _indices(recipe.get("text_positions"), text_rows, "text_positions")
    image = [text_rows + index for index in _indices(recipe.get("image_positions"), sequence - text_rows, "image_positions")]
    with torch.inference_mode():
        result = {key: changed[key].clone() for key in ("q", "k", "v")}
        for field in sorted(fields):
            if recipe.get(field, "changed") != "base":
                continue
            key = field[0]
            rows = text + image if field == "q" else text if field.endswith("text") else image
            if not rows or not selected_heads:
                continue
            row_index = torch.tensor(rows, device=result[key].device, dtype=torch.long)[:, None]
            head_index = torch.tensor(selected_heads, device=result[key].device, dtype=torch.long)[None, :]
            result[key][:, row_index, head_index, :] = base[key][:, row_index, head_index, :]
        result["attention"] = dispatch_attention_fn(result["q"], result["k"], result["v"],
            backend=None, parallel_config=None).flatten(2, 3).to(result["q"].dtype)
    return result


def _rms(value):
    import torch

    maximum = float(value.abs().max())
    return maximum * math.sqrt(float(torch.mean((value / maximum).square()))) if maximum else 0.


def _relationships(first, second):
    import torch

    first_rms, second_rms = _rms(first), _rms(second)
    dot = float(torch.mean(first * second))
    cosine = float(torch.mean((first / first_rms) * (second / second_rms))) if first_rms and second_rms else 0.
    _require(math.isfinite(dot) and math.isfinite(cosine), "nonfinite prefix dot/cosine feature")
    return {"mean_dot": dot, "cosine": max(-1., min(1., cosine)),
            "either_zero": float(not first_rms or not second_rms)}


def prefix_signature(base_residual, changed_residual, base_operands, changed_operands, attention_width, text_rows):
    """Flat JSON float features from native image-prefix ports only.

    R is residual [B,S,D]; operands are the actual to_out input [B,S,A+M].
    A comprises the first attention_width columns, M the remainder. All text
    rows are excluded. Symmetric relative RMS delta is ||change-base|| /
    (||base||+||change||), zero for two zero vectors; no epsilon or decision
    threshold. Dot features are mean elementwise inner products. Cross-variable
    R/A relationships appear only when their widths match, never via resizing.
    Nonfinite inputs/features refuse; zero-vector cosines use 0 with an explicit
    either_zero feature. These observations are not suffix or effect forecasts.
    """
    _tensors([base_residual, changed_residual], ndim=3)
    _tensors([base_operands, changed_operands], ndim=3)
    _require(base_residual.shape[:2] == base_operands.shape[:2]
             and base_residual.device == base_operands.device and base_residual.dtype == base_operands.dtype,
             "residual/operand prefix geometry or arithmetic profile differs")
    _require(type(attention_width) is int and 0 < attention_width < base_operands.shape[-1], "attention width must leave nonempty M columns")
    _require(type(text_rows) is int and 0 <= text_rows < base_residual.shape[1], "image rows required")
    pairs = {"R": (base_residual[:, text_rows:], changed_residual[:, text_rows:]),
             "A": (base_operands[:, text_rows:, :attention_width], changed_operands[:, text_rows:, :attention_width]),
             "M": (base_operands[:, text_rows:, attention_width:], changed_operands[:, text_rows:, attention_width:])}
    pairs = {name: tuple(value.detach().double() for value in pair) for name, pair in pairs.items()}
    features = {}
    for name, (base, changed) in pairs.items():
        baseline, after, delta = _rms(base), _rms(changed), _rms(changed - base)
        denominator = max(baseline, after)
        relative = (delta / denominator) / (baseline / denominator + after / denominator) if denominator else 0.
        features.update({name + "_base_rms": baseline, name + "_changed_rms": after,
                         name + "_delta_rms": delta, name + "_symmetric_relative_rms_delta": relative})
        features.update({name + "_base_changed_" + key: value for key, value in _relationships(base, changed).items()})
    features["R_A_widths_match"] = float(pairs["R"][0].shape[-1] == attention_width)
    if features["R_A_widths_match"]:
        for label, first, second in (("base_R_A", pairs["R"][0], pairs["A"][0]),
                                     ("changed_R_A", pairs["R"][1], pairs["A"][1]),
                                     ("delta_R_A", pairs["R"][1] - pairs["R"][0], pairs["A"][1] - pairs["A"][0])):
            features.update({label + "_" + key: value for key, value in _relationships(first, second).items()})
    _require(all(type(value) is float and math.isfinite(value) for value in features.values()), "nonfinite prefix feature")
    return features


def rank_supports(base_attention, changed_attention, text_rows, heads):
    """Rank native image-source contrast energies, ties by ascending index.

    Inputs are full flattened native [B,S,H*D] attention outputs, heads is H.
    Returns {'heads': head indices, 'image_positions': image-relative row indices}.
    Rankings propose observational supports; they do not establish causal utility.
    """
    import torch

    _tensors([base_attention, changed_attention], ndim=3)
    _require(type(heads) is int and heads > 0 and base_attention.shape[-1] % heads == 0, "invalid head count")
    _require(type(text_rows) is int and 0 <= text_rows < base_attention.shape[1], "image rows required")
    delta = changed_attention[:, text_rows:].detach().double() - base_attention[:, text_rows:].detach().double()
    delta = delta.unflatten(-1, (heads, -1))
    maximum = float(delta.abs().max())
    energy = (delta / maximum).square() if maximum else torch.zeros_like(delta)
    by_head, by_row = energy.sum(dim=(0, 1, 3)).cpu().tolist(), energy.sum(dim=(0, 2, 3)).cpu().tolist()
    _require(all(math.isfinite(value) for value in by_head + by_row), "nonfinite support contrast energy")
    return {"heads": sorted(range(heads), key=lambda index: (-by_head[index], index)),
            "image_positions": sorted(range(len(by_row)), key=lambda index: (-by_row[index], index))}

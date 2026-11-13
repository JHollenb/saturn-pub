"""Tiny native attention reconstruction and prefix-only support mechanics."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ops = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_family_ops.py"))


def operands():
    with torch.random.fork_rng():
        torch.manual_seed(19)
        base = {key: torch.randn(1, 5, 2, 4) for key in ("q", "k", "v")}
        changed = {key: value + torch.randn_like(value) * .3 for key, value in base.items()}
    return base, changed


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_single_block_reconstructs_with_independently_dispatched_attention(monkeypatch, dtype):
    from diffusers.models.transformers import transformer_flux2 as native

    with torch.random.fork_rng():
        torch.manual_seed(23)
        block = native.Flux2SingleTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4, mlp_ratio=2.).eval().to(dtype)
        residuals = [torch.randn(1, 5, 8).to(dtype), torch.randn(1, 5, 8).to(dtype)]
        modulation = torch.randn(1, 24).to(dtype)
        angles = torch.randn(5, 4)
    recorded, projection_inputs, mlps = [], [], []
    dispatch = native.dispatch_attention_fn

    def observing(q, k, v, **kwargs):
        recorded.append({key: value.detach().clone() for key, value in zip(("q", "k", "v"), (q, k, v))})
        return dispatch(q, k, v, **kwargs)

    monkeypatch.setattr(native, "dispatch_attention_fn", observing)
    hooks = [block.attn.to_out.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0].detach().clone())),
             block.attn.mlp_act_fn.register_forward_hook(lambda module, args, out: mlps.append(out.detach().clone()))]
    try:
        with torch.inference_mode():
            outputs = [block(value, None, modulation, (angles.cos(), angles.sin())) for value in residuals]
    finally:
        for hook in hooks:
            hook.remove()
    variant = ops["attention_variant"](base=recorded[0], changed=recorded[1], text_rows=2, recipe={})
    torch.testing.assert_close(variant["attention"], projection_inputs[1][..., :8], rtol=0., atol=0.)
    gate = modulation.unsqueeze(1).chunk(3, dim=-1)[2]
    with torch.inference_mode():
        rebuilt = residuals[1] + gate * torch.nn.functional.linear(
            torch.cat([variant["attention"], mlps[1]], dim=-1), block.attn.to_out.weight, block.attn.to_out.bias)
    torch.testing.assert_close(rebuilt[:, 2:], outputs[1][:, 2:], rtol=0., atol=0.)
    features = ops["prefix_signature"](residuals[0], residuals[1], projection_inputs[0], projection_inputs[1], 8, 2)
    assert features["R_A_widths_match"] == 1.
    assert features == ops["prefix_signature"](residuals[0], residuals[1], projection_inputs[0], projection_inputs[1], 8, 2)
    json.dumps(features, allow_nan=False)


@pytest.mark.parametrize("source", ["k", "v"])
def test_k_and_v_are_independently_selectable_with_native_attention(source):
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    base, changed = operands()
    result = ops["attention_variant"](base=base, changed=changed, text_rows=2,
                                     recipe={source + "_text": "base", source + "_image": "base"})
    expected = {key: base[key] if key == source else changed[key] for key in ("q", "k", "v")}
    for key, value in expected.items():
        torch.testing.assert_close(result[key], value, rtol=0., atol=0.)
        assert result[key].data_ptr() != changed[key].data_ptr()
    torch.testing.assert_close(result["attention"], dispatch_attention_fn(expected["q"], expected["k"], expected["v"],
        backend=None, parallel_config=None).flatten(2, 3), rtol=0., atol=0.)


def test_restricted_heads_and_partition_relative_rows_leave_other_coordinates_exact():
    base, changed = operands()
    copies = {label: {key: value.clone() for key, value in group.items()} for label, group in (("base", base), ("changed", changed))}
    recipe = {name: "base" for name in ("q", "k_text", "v_text", "k_image", "v_image")}
    recipe.update(heads=[1], text_positions=[0], image_positions=[1])
    result = ops["attention_variant"](base=base, changed=changed, text_rows=2, recipe=recipe)
    for key in ("q", "k", "v"):
        expected = changed[key].clone()
        expected[:, [0, 3], 1] = base[key][:, [0, 3], 1]
        torch.testing.assert_close(result[key], expected, rtol=0., atol=0.)
        torch.testing.assert_close(base[key], copies["base"][key], rtol=0., atol=0.)
        torch.testing.assert_close(changed[key], copies["changed"][key], rtol=0., atol=0.)
    empty = ops["attention_variant"](base=base, changed=changed, text_rows=2, recipe={"q": "base", "heads": []})
    torch.testing.assert_close(empty["q"], changed["q"], rtol=0., atol=0.)


@pytest.mark.parametrize("recipe", [{"unknown": "base"}, {"q": "future"}, {"q": []}, {"heads": [1, 1]},
    {"heads": [-1]}, {"heads": [True]}, {"text_positions": [2]}, {"image_positions": [3]}, {"heads": (0,)}])
def test_invalid_recipe_refuses_before_dispatch(recipe):
    base, changed = operands()
    with pytest.raises(ValueError):
        ops["attention_variant"](base=base, changed=changed, text_rows=2, recipe=recipe)


@pytest.mark.parametrize("problem", ["shape", "dtype", "nonfinite", "text_rows"])
def test_invalid_operand_profile_refuses(problem):
    base, changed = operands()
    text_rows = 2
    if problem == "shape":
        base["q"] = base["q"][:, :-1]
    elif problem == "dtype":
        base["k"] = base["k"].double()
    elif problem == "nonfinite":
        changed["v"][0, 0, 0, 0] = float("nan")
    else:
        text_rows = 6
    with pytest.raises(ValueError):
        ops["attention_variant"](base=base, changed=changed, text_rows=text_rows, recipe={})


def test_prefix_only_signature_excludes_text_and_handles_zero_vectors_without_nan():
    r = torch.zeros(1, 4, 2)
    projection = torch.zeros(1, 4, 6)
    changed_r, changed_projection = r.clone(), projection.clone()
    changed_r[:, :2] = 1000.
    changed_projection[:, :2] = -1000.
    untouched = ops["prefix_signature"](r, changed_r, projection, changed_projection, 2, 2)
    assert untouched["R_delta_rms"] == untouched["A_delta_rms"] == untouched["M_delta_rms"] == 0.
    assert untouched["R_base_changed_either_zero"] == 1.
    changed_r[:, 2:] = 2.
    changed_projection[:, 2:, :2] = -3.
    changed_projection[:, 2:, 2:] = 4.
    features = ops["prefix_signature"](r, changed_r, projection, changed_projection, 2, 2)
    assert features["R_delta_rms"] == 2. and features["A_delta_rms"] == 3. and features["M_delta_rms"] == 4.
    assert features["R_symmetric_relative_rms_delta"] == 1.
    assert features["delta_R_A_cosine"] == -1. and features["delta_R_A_mean_dot"] == -6.
    assert all(type(value) is float for value in features.values())
    json.dumps(features, allow_nan=False)


def test_support_energy_ranking_is_image_relative_deterministic_and_ties_by_index():
    base = torch.zeros(1, 5, 4)
    changed = base.clone()
    changed[:, :2] = 1e6
    changed[:, 2, :2] = 2.
    changed[:, 3, 2:] = 4.
    changed[:, 4, :2] = 2.
    expected = {"heads": [1, 0], "image_positions": [1, 0, 2]}
    assert ops["rank_supports"](base, changed, 2, 2) == expected
    assert ops["rank_supports"](base, changed, 2, 2) == expected
    assert ops["rank_supports"](base, base, 2, 2) == {"heads": [0, 1], "image_positions": [0, 1, 2]}
    with pytest.raises(ValueError, match="head count"):
        ops["rank_supports"](base, changed, 2, 3)


def test_loading_qualification_helper_does_not_import_frameworks():
    path = Path(__file__).parents[1] / "qualification/flux_family_ops.py"
    result = subprocess.run([sys.executable, "-I", "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1]); assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules",
        str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

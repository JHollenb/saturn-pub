"""Independent new math and saved-only instruments on tiny explicit operands."""
from __future__ import annotations

import ast
import runpy
from pathlib import Path

import numpy as np

HERE = Path(__file__).parents[1]
A = runpy.run_path(str(HERE / "qualification/flux_reentry_audit_math.py"))
R = runpy.run_path(str(HERE / "qualification/flux_reentry_reference.py"))
F = runpy.run_path(str(HERE / "tests/test_flux_reentry_reference.py"))


def test_independent_math_has_no_framework_or_predictor_imports():
    imports = ast.walk(ast.parse((HERE / "qualification/flux_reentry_audit_math.py").read_text()))
    names = [node.module for node in imports if isinstance(node, ast.ImportFrom)]
    assert names == ["__future__"]


def test_complete_independent_project_and_joint_feed_their_own_state():
    model = F["tiny_model"]()
    z, c, img, txt = F["operands"]()
    wide, parameters = F["wide"], F["parameters"]
    weights = {key: value for key, value in parameters(model).items()
        if any(key.startswith(name + ".") for name in R["PROJECT_LINEAR_NAMES"])}
    arguments = (wide(z), wide(c), img.numpy(), txt.numpy(), np.float32(166.4853), weights)
    profile = {"axes_dims_rope": [2, 2], "rope_theta": 2000, "timestep_channels": 8}
    predicted = R["predict_project"](*arguments, **profile)
    independent = A["independent_project"](*arguments, **profile)
    for key in ("image", "text"):
        np.testing.assert_array_equal(predicted[key], independent[key])
    for key in ("temb", "img_mod", "txt_mod", "single_mod"):
        np.testing.assert_array_equal(predicted["context"][key], independent["context"][key])
    for p, i in zip(predicted["context"]["rope"], independent["context"]["rope"]):
        np.testing.assert_array_equal(p, i)
    joint_weights = parameters(model.transformer_blocks[0])
    native_profile = {"heads": 2, "layernorm_eps": 1e-6, "qk_eps": 1e-6}
    p = R["predict_joint"](predicted["image"], predicted["text"], predicted["context"], joint_weights, **native_profile)
    i = A["independent_joint"](independent["image"], independent["text"], independent["context"], joint_weights, **native_profile)
    for key in ("image", "text"):
        np.testing.assert_array_equal(p[key], i[key])


def test_bf16_halfway_even_and_odd_rounding_are_explicit():
    x = np.asarray([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000], np.uint32).view(np.float32)
    expected = np.asarray([0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000], np.uint32)
    np.testing.assert_array_equal(A["B"](x).view(np.uint32), expected)


def test_generic_comparisons_and_ordered_identity_keep_signed_cancellation():
    analyzer = runpy.run_path(str(HERE / "qualification/analyze_flux_reentry.py"))
    native = np.zeros((1, 2, 2), np.float32)
    inherited = np.ones_like(native)
    projected = -np.ones_like(native)
    predicted = np.full_like(native, .25)
    decomposition = analyzer["finite_difference"]([native, inherited, projected, predicted])
    assert decomposition["max_abs_identity_residual"] == 0
    assert decomposition["inherited_latent"]["signed_mean"] == 1
    assert decomposition["project_addition"]["signed_mean"] == -2
    assert decomposition["joint_addition"]["signed_mean"] == 1.25
    assert decomposition["total"]["signed_mean"] == .25
    assert analyzer["compare"](np.ones((2, 3), np.float32), np.ones((2, 3), np.float32))["bitwise_exact"]


def test_nested_atomic_parameter_mapping_preserves_typed_context_and_tuple():
    import torch
    from saturn_pub.values import describe

    analyzer = runpy.run_path(str(HERE / "qualification/analyze_flux_reentry.py"))
    context = {"temb": torch.ones(1, 2, dtype=torch.bfloat16),
        "rope": (torch.ones(2, 2), torch.zeros(2, 2))}
    parameters = describe({"value": {"context": context, "image": torch.ones(1, 2, 2)}})
    fields = analyzer["described_mapping"](analyzer["described_mapping"](parameters)["value"])
    assert fields["context"] == describe(context)
    assert fields["image"] == describe(torch.ones(1, 2, 2))
    # Recursive unpacking loses the mapping and tuple nodes. The Act overlay
    # must compare those nodes too, not silently normalize list/tuple identity.
    assert fields["context"]["kind"] == "mapping"
    entries = {item["key"]["value"]: item["value"] for item in fields["context"]["items"]}
    assert entries["rope"]["kind"] == "tuple"

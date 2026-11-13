"""Independent algebra and source-aware installed mechanics; no learned forwards."""
from __future__ import annotations

import ast
import copy
import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from saturn_pub.core import Adapter, Session
from torch.nn import functional as F

ROOT = Path(__file__).parents[1]
FILE = ROOT / "qualification/flux_common_differential_campaign.py"
C = runpy.run_path(str(FILE))
S = runpy.run_path(str(FILE.with_name("stage_common_differential.py")))


def fixture():
    top = torch.tensor([[[.5, -.5, .5, -.5]]], dtype=torch.bfloat16)
    bottom = torch.tensor([[[.5, -.5, -.5, .5]]], dtype=torch.bfloat16)
    write = torch.tensor([[[3., -1., 5., -7.]]], dtype=torch.bfloat16)
    return write, dict(top=top, bottom=bottom, random=top.roll(1, -1))


def test_source_only_closed_form_factorization_and_native_differential_removal(monkeypatch):
    write, priors = fixture()
    monkeypatch.setattr(F, "linear", lambda *_a, **_k: pytest.fail("source projection must add no learned projection"))
    values, report = C["projection_payloads"](write, write, priors)
    expected = {"common-only": [-4., 4., -0., -0.], "differential-only": [-0., -0., -4., 4.],
        "both": [-4., 4., -4., 4.], "native-differential-remove": [-0., -0., -6., 6.]}
    for arm, row in expected.items():
        assert torch.equal(values[arm].view(torch.int16), torch.tensor([[row]], dtype=torch.bfloat16).view(torch.int16))
    geometry = report["common_differential"]
    assert geometry["cosine"]["values"] == [0.]
    assert geometry["orthogonality_max_abs"] < 1e-15 and geometry["ideal_recomposition_max_abs"] < 2e-15
    assert geometry["common_beta"]["values"][0] == pytest.approx(2*np.sqrt(2))
    assert geometry["differential_beta"]["values"][0] == pytest.approx(6*np.sqrt(2))
    assert abs(report["arms"]["native-differential-remove"]["ideal_differential_after"]["values"][0]) < 2e-15
    repeated, _ = C["projection_payloads"](write, write, {name: value*8 for name, value in priors.items()})
    assert all(torch.equal(values[name].view(torch.int16), repeated[name].view(torch.int16)) for name in values)


def test_explicit_fp64_to_fp32_to_bf16_cast_order_at_double_rounding_boundary(monkeypatch):
    # Exact nearest-BF16 rounding chooses up, whereas the declared FP32
    # intermediate hits a tie and rounds down. Torch's convenience FP64 cast
    # itself uses FP32 on this profile, so it is not an independent oracle.
    t = 1+2**-8+2**-28
    write = torch.tensor([[[-t, t, 0., 0.]]], dtype=torch.float64)
    priors = dict(top=torch.tensor([[[1., -1., 0., 0.]]], dtype=torch.bfloat16),
        bottom=torch.tensor([[[0., 0., 1., -1.]]], dtype=torch.bfloat16), random=torch.zeros(1, 1, 4, dtype=torch.bfloat16))
    original = torch.from_numpy; constructed_dtypes = []

    def observe(array):
        if array.shape == (1, 1, 4):
            constructed_dtypes.append(array.dtype)
        return original(array)

    monkeypatch.setattr(torch, "from_numpy", observe)
    values, report = C["projection_payloads"](write, write, priors)
    assert values["both"][0, 0, 0].item() == 1.
    assert min((1., 1+2**-7), key=lambda value: abs(t-value)) == 1+2**-7
    assert len(constructed_dtypes) == 4 and all(dtype == np.float32 for dtype in constructed_dtypes)
    assert report["arms"]["both"]["double_to_float32_to_bf16_cast_error_rms"] > 0


def test_parallel_antiparallel_missing_and_zero_rows_remain_finite_observations():
    _, priors = fixture(); top = priors["top"].repeat(1, 5, 1)
    bottom = torch.cat([top[:, :1], -top[:, :1], torch.zeros_like(top[:, :1]), priors["bottom"], torch.zeros_like(top[:, :1])], dim=1)
    top[:, 3:] = 0
    write = torch.tensor([[[3., -1., 5., -7.]]], dtype=torch.bfloat16).repeat(1, 5, 1)
    values, report = C["projection_payloads"](write, write, dict(top=top, bottom=bottom, random=torch.zeros_like(top)))
    geometry = report["common_differential"]
    assert geometry["zero_sum_rows"] == geometry["zero_difference_rows"] == 1
    assert geometry["incomplete_direction_rows"] == 3
    assert report["directions"]["top"]["zero_direction_rows"] == 2
    assert report["directions"]["bottom"]["zero_direction_rows"] == 2
    assert all(bool(torch.isfinite(value).all()) for value in values.values())
    assert torch.count_nonzero(values["common-only"][:, 2:]) == torch.count_nonzero(values["differential-only"][:, 2:]) == 0
    assert torch.count_nonzero(values["both"][:, 2]) > 0
    assert geometry["ideal_recomposition_max_abs"] > 0  # Missing selector is disclosed, never silently repaired.


class ArithmeticAdapter(Adapter):
    model_identity = "common-differential-arithmetic-only"
    execution = dict(family="finite-addition-mechanics")

    def addresses(self, state):
        return ("image",)

    def validate(self, state):
        assert set(state) == {"image", "clock"}

    def boundary(self, state):
        return str(state["clock"])

    def advance(self, state):
        return dict(state, clock=state["clock"]+1)


def test_separate_finite_payloads_and_real_public_endpoints_need_not_add():
    generator = torch.Generator().manual_seed(902)
    write = torch.randn(1, 8, 8, generator=generator).to(torch.bfloat16)
    priors = {name: torch.randn(1, 8, 8, generator=generator).to(torch.bfloat16) for name in ("top", "bottom", "random")}
    payloads, _ = C["projection_payloads"](write, write, priors)
    own = (torch.randn(1, 8, 8, generator=generator)*32).to(torch.bfloat16)
    base = C["previous"](); outputs = {}
    for arm, payload in payloads.items():
        branch = Session(ArithmeticAdapter(), dict(image=own.clone(), clock=0))
        before, after, receipt = base["apply_slot_edit"](branch, "image", payload, mode="add")
        assert receipt.to_dict()["parent"] == before.fingerprint and after.payload["clock"] == 0
        outputs[arm] = after.payload["image"]
    c, d, both = (payloads[name] for name in ("common-only", "differential-only", "both"))
    assert torch.count_nonzero((c.double()+d.double())-both.double()) > 0
    interaction = outputs["both"].double()-outputs["common-only"].double()-outputs["differential-only"].double()+own.double()
    assert torch.count_nonzero(interaction) > 0
    assert torch.count_nonzero(((own+c)+d).double()-outputs["both"].double()) > 0


@pytest.mark.parametrize("lane", ["forecast", "native"])
def test_actual_interaction_serializer_retains_every_signed_component_and_pixel(tmp_path, lane):
    base = C["previous"](); old = base["helpers"](); identifier = "toy"; paths, ports = {}, {}
    vectors = {"both": [0, 255, 15], "common-only": [255, 0, 10], "differential-only": [100, 100, 5], "baseline": [1, 1, 1]}
    for arm, values in vectors.items():
        key = identifier+"/"+arm; directory = tmp_path / "cases" / arm; directory.mkdir(parents=True)
        paths[key] = dict(root=str(directory.relative_to(tmp_path)))
        tensor = torch.tensor(values, dtype=torch.bfloat16)
        ports[key] = dict(x=old["_save_tensor"](tmp_path, "ports/"+arm+".npy", tensor))
        decoded = old["_save_tensor"](directory, "decoded.npy", tensor)
        pixels = old["_save_bytes"](directory, "pixels.npy", np.array(values, dtype=np.uint8))
        C["save"](directory / ("decoder-forecast.json" if lane == "forecast" else "decoder-native-reference.json"), dict(decoded=decoded, pixels=pixels))
    result = C["save_interactions"](tmp_path, {identifier: {}}, paths, ports, lane, base, old)
    expected = torch.tensor([-354., 156., 1.], dtype=torch.float64)
    for group in result[identifier].values():
        if isinstance(group, dict):
            for record in group.values():
                actual = base["load_tensor"](tmp_path, record, "cpu")
                assert actual.dtype == torch.float64 and torch.equal(actual, expected)
    assert C["read"](tmp_path / (lane+"-interactions.json"))[identifier]["formula"] == "Y_both - Y_common - Y_differential + Y_baseline"


def actual_selector_guard():
    """Execute the worker's own nested guard, without duplicating its predicate."""
    run = ast.parse(FILE.read_text())
    record = next(node for node in ast.walk(run) if isinstance(node, ast.FunctionDef) and node.name == "record_root")
    guard = next(node for node in record.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and any(isinstance(argument, ast.Constant) and argument.value == "selector transfer image layout/model/clock differs" for argument in node.value.args))
    function = ast.FunctionDef(name="check", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in ("cut", "selector_cut", "priors", "literal")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[copy.deepcopy(guard)], decorator_list=[], type_params=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])); namespace = dict(torch=torch, require=C["require"])
    exec(compile(module, str(FILE)+":actual-selector-guard", "exec"), namespace)
    return namespace["check"]


@pytest.mark.parametrize("wrong", ["model", "grid", "step", "phase", "cursor", "layout"])
def test_actual_fox_selector_transfer_guard_rejects_hostile_binding(wrong):
    image = torch.ones(1, 4, 8, dtype=torch.bfloat16); ids = torch.arange(16).reshape(1, 4, 4)
    payload = dict(image=image, img_ids=ids, phase="joint", step=0, cursor=1)
    cut = SimpleNamespace(model_identity="same-model", payload=copy.deepcopy(payload))
    selector = SimpleNamespace(model_identity="same-model", payload=copy.deepcopy(payload))
    priors = dict(top=image.clone(), bottom=image.clone(), random=image.clone()); check = actual_selector_guard()
    check(cut, selector, priors, image)
    if wrong == "model":
        cut.model_identity = "different-model"
    elif wrong == "grid":
        cut.payload["img_ids"] += 1
    elif wrong == "layout":
        priors["bottom"] = priors["bottom"][:, :2]
    else:
        cut.payload[wrong] = {"step": 1, "phase": "project", "cursor": 2}[wrong]
    with pytest.raises(ValueError, match="selector transfer"):
        check(cut, selector, priors, image)


def test_actual_source_guard_allows_declared_prefix_and_blocks_forecast_and_early_targets():
    base = C["previous"](); module = torch.nn.Identity(); phase, released, observed = dict(kind="source-qwen"), dict(native=False), {}
    linear, sdpa = F.linear, F.scaled_dot_product_attention
    with base["instrument"]([module], phase, released, observed):
        value = torch.tensor([1.]); assert module(value) is value
        phase["kind"] = "source-project"; assert module(value) is value
        phase["kind"] = "source-joint0"; assert module(value) is value
        phase["kind"] = "forecast-denoiser"
        with pytest.raises(ValueError, match="module forward forbidden"):
            module(value)
        phase["kind"] = "target-denoiser"
        with pytest.raises(ValueError, match="native target before global seal"):
            module(value)
        released["native"] = True; assert module(value) is value
    assert F.linear is linear and F.scaled_dot_product_attention is sdpa and not module._forward_pre_hooks
    assert set(observed) == {"source-qwen", "source-project", "source-joint0", "target-denoiser"}


def test_real_installed_adapter_is_rebound_after_same_value_storage_change_without_forward(monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel

    monkeypatch.setattr(torch.nn.Module, "__call__", lambda *_a, **_k: pytest.fail("adapter binding must execute no module"))
    model = Flux2Transformer2DModel(in_channels=8, num_layers=1, num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
        joint_attention_dim=24, timestep_guidance_channels=16, axes_dims_rope=(4, 4, 4, 4), guidance_embeds=False)
    scheduler = dict(FlowMatchEulerDiscreteScheduler().config)
    original = C["fresh_adapter"](model, scheduler, device="cpu"); original.validate_execution()
    parameter = next(model.parameters()); parameter.data = parameter.detach().clone()
    with pytest.raises(ValueError, match="rewrap the model"):
        original.validate_execution()
    rebound = C["fresh_adapter"](model, scheduler, device="cpu"); rebound.validate_execution()
    assert rebound is not original and rebound.model_identity == original.model_identity
    assert rebound.granularity == "operation"


def test_actual_staged_metadata_source_closure_and_fixed_new_context(tmp_path):
    staged = ROOT / "fixtures/common-differential-staged"
    request = C["read"](staged / "request.json")
    assert C["NEW_CONTEXT"]["id"] == "foxcat0s821" and C["NEW_CONTEXT"]["seed"] == 821
    assert "red fox" in C["NEW_CONTEXT"]["prompt"] and "gray tabby cat" in C["NEW_CONTEXT"]["prompt"]
    assert C["NEW_SELECTOR_CONTEXT"] == "fresh0s613"
    assert request["config"]["new_context"] == C["NEW_CONTEXT"]
    assert request["config"]["new_selector_context"] == C["NEW_SELECTOR_CONTEXT"]
    assert request["scheduling_contract"]["component_hydrations"] == C["contract"]()["component_hydrations"] == 3
    assert C["contract"]()["functional_transitions"] == C["contract"]()["native_target_transitions"] == 18*110
    assert C["contract"]()["native_qwen_source_calls"] == C["contract"]()["native_project_source_transitions"] == C["contract"]()["native_joint0_source_transitions"] == 1
    copied = tmp_path / "stage"; shutil.copytree(staged, copied)
    S["validate"](copied, request)
    (copied / "payload" / FILE.name).write_bytes(b"hostile worker bytes")
    with pytest.raises(ValueError, match="staged source/wheel changed"):
        S["validate"](copied, request)
    changed = copy.deepcopy(request); changed["config"]["new_context"]["seed"] += 1
    with pytest.raises(ValueError, match="request or launcher changed"):
        S["validate"](staged, changed)

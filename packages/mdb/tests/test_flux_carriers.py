"""Own-source casts/strides, typed project writes, and actual edited-Z3 rescue."""
from __future__ import annotations

import ast
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from PIL import Image
from saturn_pub import Session
from saturn_pub.adapters.flux2 import Flux2KleinAdapter
from saturn_pub.store import LocalStore
from saturn_pub.values import describe

from mdb.forward_recording import tensor_descriptor
from mdb.native_navigation import NativeCutNavigator

HERE = Path(__file__).parents[1] / "qualification"
q = runpy.run_path(str(HERE / "flux_carriers.py"))
campaign = runpy.run_path(str(HERE / "flux_carriers_campaign.py"))


def specimen():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(317)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=1, num_single_layers=1,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6, timestep_guidance_channels=8,
            axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(torch.bfloat16)
        adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
        session = adapter.session(latent=torch.randn(1, 4, 4, dtype=torch.bfloat16).transpose(1, 2),
            conditioning=torch.randn(1, 3, 6, dtype=torch.bfloat16), img_ids=torch.zeros(1, 4, 4),
            txt_ids=torch.zeros(1, 3, 4), timesteps=torch.tensor([1000., 800., 600., 300.]),
            sigmas=torch.tensor([1., .8, .6, .3, 0.]))
    initial = session.capture()
    nav = NativeCutNavigator(session, max_steps=30)
    nav.until(logical_step=2, phase="project")
    return adapter, initial, nav.capture()


def test_fp32_anchor_residual_one_bf16_cast_and_native_copy_strides():
    _, initial, parent = specimen()
    z0, z2, c = initial.payload["latent"], parent.payload["latent"], parent.payload["conditioning"]
    anchor = parent.payload["sigmas"][2].float() * z0.float()
    residual = z2.float() - anchor
    before = parent.fingerprint
    for alpha in (0., .5, 1.):
        for beta in (0., .5, 1.):
            result = q["carrier_payloads"](initial, parent, alpha=alpha, beta=beta)
            assert torch.equal(result["anchor"], anchor) and result["anchor"].dtype == torch.float32
            assert torch.equal(result["residual"], residual)
            expected = z2 if alpha == 1. else (anchor + alpha * residual).to(torch.bfloat16)
            assert torch.equal(result["latent"], expected) and result["latent"].stride() == z2.stride()
            assert torch.equal(result["conditioning"], c if beta == 1. else (beta * c.float()).to(torch.bfloat16))
            assert result["conditioning"].stride() == c.stride()
            assert result["recipe"]["native_latent_copy"] == (alpha == 1.)
            assert result["recipe"]["native_conditioning_copy"] == (beta == 1.)
            assert result["latent"].data_ptr() != z2.data_ptr() and result["conditioning"].data_ptr() != c.data_ptr()
    assert parent.fingerprint == before


def test_typed_sham_latent_and_c_are_exact_identity_and_close_future(tmp_path):
    adapter, initial, parent = specimen()
    baseline = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=12)
    sham = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=12)
    value = q["carrier_payloads"](initial, parent, alpha=1., beta=1.)
    store = LocalStore(tmp_path)
    actions = [q["install_carrier_port"](sham, slot, value[slot], name="sham-" + slot, store=store, expected_step=2)
               for slot in ("latent", "conditioning")]
    assert actions[0]["post_act"] == actions[1]["parent"]
    assert describe(sham.capture().payload) == describe(parent.payload)
    adapter.finish(baseline.session)
    adapter.finish(sham.session)
    assert torch.equal(baseline.session.read("latent"), sham.session.read("latent"))


def test_rescue_reuses_actual_edited_z3_and_changes_only_future_own_c(tmp_path):
    adapter, initial, parent = specimen()
    n = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=16)
    value = q["carrier_payloads"](initial, parent, alpha=0., beta=0.)
    store = LocalStore(tmp_path)
    for slot in ("latent", "conditioning"):
        q["install_carrier_port"](n, slot, value[slot], name="edited-" + slot, store=store, expected_step=2)
    n.until(logical_step=3, phase="project")
    actual_edited_z3 = n.capture()
    saved = store.save(actual_edited_z3)
    restored = store.load(saved)
    rescue = NativeCutNavigator(Session.from_cut(adapter, restored), max_steps=8)
    action = q["install_carrier_port"](rescue, "conditioning", parent.payload["conditioning"],
        name="own-C-rescue", store=store, expected_step=3)
    assert action["parent"] == actual_edited_z3.fingerprint
    expected = {**actual_edited_z3.payload, "conditioning": parent.payload["conditioning"]}
    assert describe(rescue.capture().payload) == describe(expected)
    assert torch.equal(rescue.session.read("latent"), n.session.read("latent"))
    assert not torch.equal(rescue.session.read("conditioning"), n.session.read("conditioning"))
    native = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=8)
    native.until(logical_step=3, phase="project")
    assert not torch.equal(native.session.read("latent"), actual_edited_z3.payload["latent"])
    # Both consume their own saved edited latent. Restoring C changes only the
    # future project input; no prefix replay or replacement with native Z3.
    rescue.step()
    assert torch.equal(rescue.session.read("conditioning"), parent.payload["conditioning"])
    rescue.until(logical_step=4, phase="project")
    n.until(logical_step=4, phase="project")
    assert rescue.steps == 5 and n.steps == 10  # tiny five-operation steps:5 shared+5 own.
    assert rescue.point["logical_step"] == n.point["logical_step"] == 4


def test_source_and_wrong_clock_refusals_leave_frontier_unchanged(tmp_path):
    adapter, initial, parent = specimen()
    n = NativeCutNavigator(Session.from_cut(adapter, parent), max_steps=8)
    before = n.capture().fingerprint
    with pytest.raises(ValueError, match="clock"):
        q["install_carrier_port"](n, "conditioning", parent.payload["conditioning"],
            name="too-early", store=LocalStore(tmp_path), expected_step=3)
    assert n.capture().fingerprint == before
    with pytest.raises(ValueError, match="closed project"):
        q["validate_sources"](parent, parent)
    with pytest.raises(ValueError, match="alpha/beta"):
        q["carrier_payloads"](initial, parent, alpha=True, beta=0.)
    n.step()
    mid = n.capture().fingerprint
    with pytest.raises(ValueError, match="clock"):
        q["install_carrier_port"](n, "latent", parent.payload["latent"],
            name="mid-block", store=LocalStore(tmp_path), expected_step=2)
    assert n.capture().fingerprint == mid


def test_fixed_eight_cells_sham_and_two_shared_rescues_config():
    branches = campaign["BRANCHES"]
    config = {"branches": branches, "preview_rows": campaign["PREVIEW_ROWS"], "preview_steps": [3],
        "input_cuts": {"Z0": "a" * 64, "Z2": "b" * 64}, "input_store": "input-store",
        "source_selection": "source-selection.json", "steps": 4, "height": 512, "width": 512, "seed": 47, "prompt": "own prompt",
        "max_adapter_steps": 650, "max_images": 15, "max_final_images": 11,
        "max_preview_images": 4, "max_vae_calls": 16, "max_wall_seconds": 600, "archive_compression_level": 1}
    campaign["validate_config"](config)
    assert len(branches) == 11 and sum(row["rescue_from"] is None for row in branches) == 9
    assert 9 * 56 + 2 * 28 == 560
    assert sum(2 for row in branches if row["rescue_from"] is None and row["id"] != "native") + 2 == 18
    for update in ({"branches": branches[:-1]}, {"max_adapter_steps": 559}, {"preview_steps": [2, 3]},
                   {"archive_compression_level": True}, {"max_images": 14}, {"seed": 17}, {"prompt": ""}):
        with pytest.raises(ValueError):
            campaign["validate_config"]({**config, **update})


def publication_fixture(tmp_path, *, mutate=False):
    """Exercise the worker's exact final publication path, with no VAE/model."""
    path = HERE / "flux_carriers_campaign.py"
    finish = next(node for node in ast.walk(ast.parse(path.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == "finish")
    own_c = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    final = SimpleNamespace(fingerprint="f" * 64, payload={"conditioning": own_c}, execution_point={"logical_step": 4})
    calls = []

    def capture():
        calls.append(True)
        if mutate and len(calls) > 1:
            return SimpleNamespace(fingerprint="e" * 64)
        return final

    n = SimpleNamespace(point={"logical_step": 4}, capture=capture)
    pixels = iter([Image.new("RGB", (2, 2), "white"), Image.new("RGB", (2, 2), "black"), Image.new("RGB", (2, 2), "white")])
    namespace = {**campaign, "np": np, "describe": describe, "tensor_descriptor": tensor_descriptor,
        "store": SimpleNamespace(save=lambda cut: cut.fingerprint), "decode": lambda n, **kw: next(pixels),
        "config": {"max_final_images": 11, "max_images": 15, "max_preview_images": 4}, "output": tmp_path,
        "counts": {"final_images": 0, "images": 0, "preview_images": 0}, "selection": {"expected_native_png_sha256": "a" * 64},
        "parity": {}, "controls": {}, "rows": [], "event": lambda *args, **kwargs: None}
    exec(compile(ast.Module(body=[finish], type_ignores=[]), str(path), "exec"), namespace)
    return namespace, n, own_c


def test_preview_numerical_disagreement_preserves_native_and_later_rows(tmp_path):
    namespace, nav, own_c = publication_fixture(tmp_path)
    for name in ("native", "sham"):
        namespace["finish"](nav, {"id": name, "rescue_from": None}, own_c)
    assert [row["id"] for row in namespace["rows"]] == ["native", "sham"]
    assert (tmp_path / "rows/native.json").is_file() and (tmp_path / "rows/sham.json").is_file()
    assert namespace["parity"]["pixel_exact"] is False and namespace["parity"]["png_exact"] is False
    assert namespace["controls"]["preview_helper_final_pixel_exact"] is False
    assert namespace["controls"]["preview_helper_final_png_exact"] is False
    assert namespace["counts"]["final_images"] == 2


def test_preview_state_mutation_remains_strict(tmp_path):
    namespace, nav, own_c = publication_fixture(tmp_path, mutate=True)
    with pytest.raises(ValueError, match="changed completed carrier state"):
        namespace["finish"](nav, {"id": "native", "rescue_from": None}, own_c)
    assert namespace["rows"] == [] and not (tmp_path / "rows/native.json").exists()

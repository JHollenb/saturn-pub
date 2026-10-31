"""Tiny random Diffusers 0.39 graph parity; never loads trained checkpoints."""
from __future__ import annotations

import ast
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from saturn_pub.adapters.flux2 import Flux2KleinAdapter

graph = runpy.run_path(str(Path(__file__).parents[1] / "qualification/flux_functional_denoiser.py"))


def tiny(dtype):
    with torch.random.fork_rng():
        torch.manual_seed(91)
        model = Flux2Transformer2DModel(in_channels=4, num_layers=2, num_single_layers=2,
            attention_head_dim=8, num_attention_heads=2, joint_attention_dim=6,
            timestep_guidance_channels=8, axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False).eval().to(dtype)
        state = {"latent": torch.randn(1, 4, 4).to(dtype), "conditioning": torch.randn(1, 3, 6).to(dtype),
            "img_ids": torch.tensor([[[0., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 1., 1., 0.]]]),
            "txt_ids": torch.arange(12).reshape(1, 3, 4).float(),
            "timesteps": torch.tensor([1000., 670., 330.]), "sigmas": torch.tensor([1., .67, .33, 0.]),
            "step": 0, "phase": "project", "cursor": 0, "context": None,
            "text": None, "image": None, "noise_prediction": None}
    return model, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("observed_project", [False, True])
def test_entire_ownstate_trajectory_matches_native_every_boundary_without_any_module_calls(dtype, observed_project, monkeypatch):
    torch.set_num_threads(1)
    model, state = tiny(dtype)
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    if observed_project:
        state = adapter.advance(state)
    original = graph["_clone_tree"](state)
    versions = [(id(p), p.data_ptr(), p._version) for p in model.parameters()]
    native = []
    current = state
    while current["step"] < len(current["timesteps"]):
        before = (current["step"], current["phase"], current["cursor"])
        current = adapter.advance(current)
        native.append((before, current))

    def forbidden(*args, **kwargs):
        raise AssertionError("functional predictor called a native module")

    for module in model.modules():
        monkeypatch.setattr(module, "forward", forbidden)
    calls = []

    def compare(event, actual):
        before, expected = native[len(calls)]
        assert (event["step"], event["phase"], event["index"]) == before
        assert (event["next_step"], event["next_phase"], event["next_index"]) == (expected["step"], expected["phase"], expected["cursor"])
        for key in ("image", "text", "latent", "noise_prediction"):
            if expected[key] is None:
                assert actual[key] is None
            else:
                torch.testing.assert_close(actual[key], expected[key], rtol=0., atol=0.)
        if expected["context"] is not None:
            for key in ("temb", "img_mod", "txt_mod", "single_mod"):
                torch.testing.assert_close(actual["context"][key], expected["context"][key], rtol=0., atol=0.)
            for a, e in zip(actual["context"]["rope"], expected["context"]["rope"]):
                torch.testing.assert_close(a, e, rtol=0., atol=0.)
        calls.append(event)

    result = graph["functional_trajectory"](model, state, compare)
    assert len(calls) == 3 * 7 - int(observed_project)
    torch.testing.assert_close(result["latent"], current["latent"], rtol=0., atol=0.)
    assert [(id(p), p.data_ptr(), p._version) for p in model.parameters()] == versions
    for key in ("latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas"):
        torch.testing.assert_close(state[key], original[key], rtol=0., atol=0.)
        assert result[key].data_ptr() != state[key].data_ptr()


def test_scheduler_preserves_bfloat16_product_rounding_before_float_addition():
    latent = torch.tensor([1.], dtype=torch.bfloat16)
    noise = torch.tensor([.104], dtype=torch.bfloat16)
    sigmas = torch.tensor([1., .67])
    actual = graph["functional_scheduler"](latent, noise, sigmas, 0)
    expected = (latent.float() + (sigmas[1] - sigmas[0]) * noise).to(noise.dtype)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)


def test_budget_and_unknown_frontier_fail_without_native_fallback():
    model, state = tiny(torch.float32)
    with pytest.raises(ValueError, match="budget exhausted"):
        graph["functional_trajectory"](model, state, max_transitions=1)
    with pytest.raises(ValueError, match="malformed"):
        graph["functional_trajectory"](model, dict(state, phase="unknown"))
    model.train()
    with pytest.raises(ValueError, match="eval"):
        graph["functional_trajectory"](model, state)


def test_worker_parent_manifest_pack_predict_seal_native_clocks_end_to_end(tmp_path, monkeypatch):
    """Exercise the actual worker's nested pack/unpack/tag helpers on CPU.

    Only unpack's literal CUDA destination is substituted with CPU for this tiny
    mechanics check; the production helper's tree/array/clock logic is executed.
    This does not qualify CUDA kernels or actual image geometry.
    """
    from saturn_pub.store import LocalStore

    from mdb.forecast_transaction import ForecastTransaction, verify_transaction

    worker = Path(__file__).parents[1] / "qualification/flux_trajectory_forecast_campaign.py"
    tree = ast.parse(worker.read_text())
    nodes = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
             and node.name in {"pack", "unpack", "arrays_at"}}
    assert set(nodes) == {"pack", "unpack", "arrays_at"}

    class CPUUnpack(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda" else node

    from collections.abc import Mapping

    namespace = {"torch": torch, "np": np, "Mapping": Mapping}
    for name, node in nodes.items():
        if name == "unpack":
            node = CPUUnpack().visit(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(worker), "exec"), namespace)
    model, initial = tiny(torch.bfloat16)
    adapter = Flux2KleinAdapter(model, FlowMatchEulerDiscreteScheduler(), granularity="operation")
    session = adapter.session(**{key: initial[key] for key in
                                ("latent", "conditioning", "img_ids", "txt_ids", "timesteps", "sigmas")})
    session.continue_(1)
    parent = session.capture()
    assert parent.execution_point["phase"] == "joint" and parent.execution_point["index"] == 0
    store = LocalStore(tmp_path / "typed-store")
    store.save(parent)
    tx = ForecastTransaction(tmp_path / "transaction", model_identity=adapter.model_identity,
        input_identity=parent.fingerprint, source_identity="tiny-worker-helper-test",
        frontier={"kind": "architectural"}, budgets={"max_native_calls": 1})
    expected_tags = []
    with tx:
        tx.frontier("initial", {"state": namespace["pack"](parent.payload), "manifest": parent.manifest()})

        def predict(context):
            state = namespace["unpack"](context.frontiers["initial"]["state"])
            assert state["context"]["rope"][0].dtype == parent.payload["context"]["rope"][0].dtype
            for name in ("image", "text", "latent"):
                torch.testing.assert_close(state[name], parent.payload[name], rtol=0., atol=0.)
            rows = []

            def record(info, current):
                tag, arrays = namespace["arrays_at"](info, current)
                expected_tags.append(tag)
                rows.append({"tag": tag, "arrays": {name: value.float().cpu().numpy().copy() for name, value in arrays.items()}})

            def forbidden(*args, **kwargs):
                raise AssertionError("prediction invoked native module")

            with monkeypatch.context() as patch:
                for module in model.modules():
                    patch.setattr(module, "forward", forbidden)
                final = graph["functional_trajectory"](model, state, record)
            return {"predictions": {"rows": rows, "final_latent": final["latent"].float().numpy().copy()}}

        tx.predict("trajectory", predict)
        assert session.capture(retain=False).fingerprint == parent.fingerprint
        seal = tx.seal()
        tx.release()

        def execute(context):
            rows = context.predictions["trajectory"]["rows"]
            tags = []
            while len(tags) < len(rows):
                before = session.capture(retain=False).payload
                receipt = session.continue_(1)
                store.receipt(receipt)
                tag, arrays = namespace["arrays_at"]({"step": before["step"], "phase": before["phase"], "index": before["cursor"]},
                                                     session.capture(retain=False).payload)
                row = rows[len(tags)]
                assert tag == row["tag"]
                for name, value in arrays.items():
                    np.testing.assert_array_equal(value.float().cpu().numpy(), row["arrays"][name])
                tags.append(tag)
            assert tags == expected_tags
            assert session.capture(retain=False).payload["step"] == len(initial["timesteps"])
            np.testing.assert_array_equal(session.read("latent").float().numpy(), context.predictions["trajectory"]["final_latent"])
            return {"tags": tags, "native_transitions": len(tags)}

        tx.execute("native", execute)
    assert tx.observation("native")["native_transitions"] == 20
    assert verify_transaction(tmp_path / "transaction", expected_seal_sha256=seal["seal_sha256"])["verified"]

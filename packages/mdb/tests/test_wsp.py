from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from mdb.forecast_transaction import ForecastTransactionError, verify_transaction
from mdb.wsp import WeightStudy, bind_patch_ports, main, verify_weight_study
from mdb.wsp_response import linear_response, receiver_factorial, swiglu_response
from mdb.wsp_weights import ParameterPatch, WeightSpaceError, parameter_inventory, weight_diff


def model():
    result = torch.nn.Linear(2, 2, bias=False).eval()
    with torch.no_grad():
        result.weight.copy_(torch.eye(2))
    return result


def patch(net):
    return ParameterPatch(net, {"weight": torch.tensor([[2., 0.], [0., 3.]])},
                          model_identity={"checkpoint": "tiny"})


def study(root, net=None):
    return WeightStudy(root, model=net, binding={"model": {"checkpoint": "tiny"},
        "checkpoint": "original-tiny-bytes", "execution": "CPU FP32", "consumer": "two steps",
        "horizon": "2 generated states"}, input_identity={"x": [1., 1.]},
        source_identity={"test": "tiny-mechanics"})


def predict(context):
    return {"predictions": {"states": np.array([[2., 3.], [4., 9.]])}}


def execute(context):
    x = torch.ones(2)
    states = []
    for _ in range(2):
        x = context.model(x)
        states.append(x)
    return {"observations": {"states": torch.stack(states)}, "work": {"forwards": 2},
            "state": {"policy": "restart", "initial": [1., 1.], "own_state_steps": 2}}


def test_diff_counts_actual_byte_changes_support_and_denominator():
    a = torch.zeros((3, 4), dtype=torch.bfloat16)
    b = a.clone()
    b[1, 2] = 3
    b[2, 1] = -0.0
    result = weight_diff({"D": a}, {"D": b}, chunk_elements=3, support_limit=1)
    assert result["changed_elements"] == 2
    assert result["denominator"]["elements"] == 12
    row = result["tensors"][0]
    assert row["numerically_changed_elements"] == 1
    assert row["columns"] == {"count": 2, "indices": [1], "truncated": True}
    assert row["delta_l2"] == 3
    assert weight_diff({"scalar": a[0, 0]}, {"scalar": b[0, 0]})["changed_elements"] == 0


def test_diff_refuses_missing_dtype_and_nonfinite_evidence():
    for before, after in [({"a": torch.ones(1)}, {}),
        ({"a": torch.ones(1)}, {"a": torch.ones(1, dtype=torch.float64)}),
        ({"a": torch.ones(1)}, {"a": torch.tensor([float("nan")])})]:
        with pytest.raises(WeightSpaceError):
            weight_diff(before, after)


def test_patch_restores_original_bytes_on_failure_and_can_be_reused():
    net = model()
    candidate = patch(net)
    before = parameter_inventory(net)
    with pytest.raises(RuntimeError, match="native failure"):
        with candidate.installed(net):
            torch.testing.assert_close(net(torch.ones(2)), torch.tensor([2., 3.]))
            raise RuntimeError("native failure")
    assert parameter_inventory(net) == before
    with candidate.installed(net):
        assert net.weight[0, 0] == 2
    assert parameter_inventory(net) == before


def test_tied_aliases_work_and_conflicts_and_storage_views_are_refused():
    net = model()
    net.tied = net.weight
    candidate = patch(net)
    assert candidate.manifest["parameters"][0]["aliases"] == ["tied", "weight"]
    with candidate.installed(net):
        assert net.weight is net.tied
        assert net.tied[0, 0] == 2
    with pytest.raises(WeightSpaceError, match="conflicting"):
        ParameterPatch(net, {"weight": torch.ones(2, 2), "tied": torch.zeros(2, 2)},
                       model_identity="tiny")
    net.view = torch.nn.Parameter(net.weight.detach().view(4))
    with pytest.raises(WeightSpaceError, match="share touched storage"):
        patch(net)


def test_patch_prevalidates_all_parameters_before_any_write():
    net = torch.nn.Sequential(model(), model())
    candidate = ParameterPatch(net, {"0.weight": torch.ones(2, 2), "1.weight": torch.ones(2, 2)},
                               model_identity="tiny")
    before = net[0].weight.detach().clone()
    with torch.no_grad():
        net[1].weight[0, 0] = 7
    with pytest.raises(WeightSpaceError, match="original parameter bytes"):
        with candidate.installed(net):
            pytest.fail("must not install")
    assert torch.equal(net[0].weight, before)


def test_dose_with_several_tied_aliases_uses_canonical_parameter_order():
    net = torch.nn.Module()
    net.a = torch.nn.Parameter(torch.tensor([1.]))
    net.z = net.a
    net.b = torch.nn.Parameter(torch.tensor([2.]))
    candidate = ParameterPatch(net, {"z": torch.tensor([3.]), "b": torch.tensor([4.])},
                               model_identity="tied")
    with candidate.scaled(net, .5).installed(net):
        assert net.a == 2
        assert net.b == 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_patch_save_load_dose_and_source_binding(tmp_path, dtype):
    net = model().to(dtype)
    replacement = torch.tensor([[2., 0.], [0., 3.]], dtype=dtype)
    candidate = ParameterPatch(net, {"weight": replacement}, model_identity="tiny")
    candidate.save(tmp_path / "patch")
    loaded = ParameterPatch.load(tmp_path / "patch", net, model_identity="tiny")
    assert loaded.manifest == candidate.manifest
    with loaded.scaled(net, 0).installed(net):
        assert torch.equal(net.weight, torch.eye(2, dtype=dtype))
    with loaded.scaled(net, -1).installed(net):
        assert torch.equal(net.weight, torch.tensor([[0., 0.], [0., -1.]], dtype=dtype))
    with pytest.raises(WeightSpaceError, match="identity differs"):
        ParameterPatch.load(tmp_path / "patch", net, model_identity="other")
    payload = tmp_path / "patch" / "weights" / "0.npy"
    payload.write_bytes(payload.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="digest"):
        ParameterPatch.load(tmp_path / "patch", net, model_identity="tiny")


def test_linear_and_swiglu_response_accounts_include_joint_changes():
    rng = np.random.default_rng(15)
    g, u, d = rng.normal(size=(3, 4)), rng.normal(size=(3, 4)), rng.normal(size=(4, 3))
    gp, up, dp = g + .2, u - .1, d + .3
    x = rng.normal(size=(2, 4))
    result = swiglu_response(g, u, d, gp, up, dp, x)
    effect = sum(result[k] for k in ("output_map", "gate", "payload", "gate_payload_interaction"))
    np.testing.assert_allclose(effect, result["edited"] - result["baseline"], atol=1e-14)
    linear = linear_response(d, dp, x[:, :3])
    np.testing.assert_allclose(linear["direct"], linear["edited"] - linear["baseline"], atol=1e-14)
    corners = receiver_factorial(result["baseline"], result["edited"],
                                 result["baseline"] * 2, result["edited"] * 3)
    np.testing.assert_allclose(corners["weight"] + corners["receiver"] + corners["interaction"],
                               corners["total"])
    with pytest.raises(ValueError, match="geometry"):
        swiglu_response(g, u, d, gp, up, dp, np.ones(7))


def test_sealed_study_predicts_own_state_and_restores_with_offline_audit(tmp_path):
    net = model()
    before = parameter_inventory(net)
    with study(tmp_path / "study", net) as experiment:
        experiment.add_arm("edited", role="edited", predictor=predict, patch=patch(net),
                           state_policy="restart", symbol_refs=[{"role": "hypothesis", "name": "write"}])
        experiment.seal()
        experiment.run("edited", execute)
        row = experiment.tx.observation("edited")
        np.testing.assert_array_equal(row["result"]["observations"]["states"], [[2, 3], [4, 9]])
        assert row["terminal_status"] == "not-evaluated"
        assert parameter_inventory(net) == before
    assert verify_transaction(tmp_path / "study")["verified"]
    verified = verify_weight_study(tmp_path / "study")
    assert verified["outcomes"] == 1
    assert verify_weight_study(tmp_path / "study", expected_manifest=verified["outcome_manifest"])["verified"]
    with pytest.raises(WeightSpaceError, match="manifest identity"):
        verify_weight_study(tmp_path / "study", expected_manifest="wrong")
    assert not net._forward_pre_hooks


def test_early_execution_and_caught_prediction_execution_latch(tmp_path):
    net = model()
    with study(tmp_path / "early", net) as experiment:
        def caught(context):
            with pytest.raises(WeightSpaceError, match="outside released"):
                net(torch.ones(2))
            return predict(context)
        with pytest.raises(WeightSpaceError, match="latched"):
            experiment.add_arm("bad", role="native", predictor=caught, state_policy="restart")
        with pytest.raises(WeightSpaceError, match="latched"):
            experiment.seal()
    assert not net._forward_pre_hooks
    with study(tmp_path / "unsealed", net) as experiment:
        experiment.add_arm("native", role="native", predictor=predict, state_policy="restart")
        with pytest.raises(ForecastTransactionError, match="phase"):
            experiment.run("native", execute)


def test_optional_failure_is_preserved_and_next_arm_still_executes(tmp_path):
    net = model()
    with study(tmp_path / "study", net) as experiment:
        for name in ("bad", "good"):
            experiment.add_arm(name, role="edited", predictor=predict, patch=patch(net),
                               state_policy="restart")
        experiment.seal()
        def failure(context):
            context.model(torch.ones(2))
            raise RuntimeError("consumer unavailable")
        experiment.run("bad", failure)
        experiment.run("good", execute)
        bad = experiment.tx.observation("bad")
        assert bad["instrument_status"] == "evidence-incomplete"
        assert bad["result"]["error"]["message"] == "consumer unavailable"
        assert experiment.tx.observation("good")["instrument_status"] == "observed"
    assert verify_transaction(tmp_path / "study")["verified"]


def test_unrelated_parameter_mutation_stops_study(tmp_path):
    net = model()
    with study(tmp_path / "study", net) as experiment:
        experiment.add_arm("native", role="native", predictor=predict, state_policy="restart")
        experiment.seal()
        def mutate(context):
            context.model.weight.add_(1)
            return execute(context)
        with pytest.raises(WeightSpaceError, match="outside the restored patch"):
            experiment.run("native", mutate)


def test_results_are_frozen_before_parameter_restore_including_bf16(tmp_path):
    net = model().to(torch.bfloat16)
    candidate = ParameterPatch(net, {"weight": torch.ones(2, 2, dtype=torch.bfloat16)},
                               model_identity={"checkpoint": "tiny"})
    with study(tmp_path / "study", net) as experiment:
        experiment.add_arm("edited", role="edited", predictor=predict, patch=candidate,
                           state_policy="restart")
        experiment.seal()
        experiment.run("edited", lambda context: {"observations": {"parameter": context.model.weight},
                                                  "work": {}, "state": {"policy": "restart"}})
        archived = experiment.tx.observation("edited")["result"]["observations"]["parameter"]
        assert archived["encoding"] == "bf16-bits-v1"
        np.testing.assert_array_equal(archived["bits"], np.full((2, 2), 0x3f80, dtype=np.uint16))
        assert torch.equal(net.weight, torch.eye(2, dtype=torch.bfloat16))


def test_static_reading_mode_and_no_framework_cli(tmp_path, capsys):
    with study(tmp_path / "static") as experiment:
        experiment.add_arm("read", role="static", state_policy="not-applicable", predictor=predict)
        experiment.seal()
    assert main(["verify", str(tmp_path / "static")]) == 0
    assert json.loads(capsys.readouterr().out)["verified"]
    source = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'transformers', 'numpy', 'saturn', 'common'}:
            raise RuntimeError('unexpected framework: ' + name)
sys.meta_path.insert(0, Block())
from mdb.cli import main
main(['wsp', '--help'])
"""
    process = subprocess.run([sys.executable, "-c", source], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert "inspect" in process.stdout
    process = subprocess.run([sys.executable, "-c", source.replace(
        "main(['wsp', '--help'])", "main(['wsp', 'verify', " + repr(str(tmp_path / "static")) + "])")],
        text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["verified"]


@pytest.mark.parametrize("tamper", ["observation", "array"])
def test_outcome_verifier_detects_postrelease_corruption(tmp_path, tamper):
    net = model()
    with study(tmp_path / "study", net) as experiment:
        experiment.add_arm("native", role="native", predictor=predict, state_policy="restart")
        experiment.seal()
        experiment.run("native", execute)
    if tamper == "observation":
        path = tmp_path / "study" / "observations" / "native.json"
    else:
        final = json.loads((tmp_path / "study" / "wsp-outcomes.json").read_text())
        path = tmp_path / "study" / final["outcomes"][0]["arrays"][0]["path"]
    path.write_bytes(path.read_bytes()[:-1] + b"X")
    with pytest.raises((WeightSpaceError, ForecastTransactionError), match="bytes differ|artifact changed"):
        verify_weight_study(tmp_path / "study")


def test_frozen_patch_artifact_tamper_blocks_native_release(tmp_path):
    net = model()
    with study(tmp_path / "study", net) as experiment:
        experiment.add_arm("edited", role="edited", predictor=predict, patch=patch(net),
                           state_policy="restart")
        experiment.seal()
        path = tmp_path / "study" / "patches" / "edited" / "weights" / "0.npy"
        path.write_bytes(path.read_bytes()[:-1] + b"X")
        with pytest.raises(ForecastTransactionError, match="artifact changed"):
            experiment.run("edited", execute)
        assert experiment.report()["instrumented_work"]["root_forward_calls"] == 0


def test_native_budget_is_latched_even_if_consumer_catches_error(tmp_path):
    net = model()
    with WeightStudy(tmp_path / "study", model=net, budgets={"max_native_calls": 1},
        binding={"model": "tiny", "checkpoint": "tiny", "execution": "CPU FP32", "consumer": "toy",
                 "horizon": "2 steps"}, input_identity="input", source_identity="source") as experiment:
        experiment.add_arm("native", role="native", predictor=predict, state_policy="restart")
        experiment.seal()
        def catches(context):
            with pytest.raises(WeightSpaceError, match="budget"):
                execute(context)
            return {"observations": {}, "work": {}, "state": {}}
        with pytest.raises(WeightSpaceError, match="latched"):
            experiment.run("native", catches)


def test_tiny_policy_demo_has_causal_restoration_and_own_action_forecasts(tmp_path):
    from mdb.wsp_demo import run_demo

    report = run_demo(tmp_path / "demo")
    assert report["rows"]["native"]["actions"] == [0, 0, 0]
    assert report["rows"]["edited"]["actions"] == [1]
    assert report["rows"]["zero"]["actions"] == [0, 0, 0]
    assert report["rows"]["restore"]["actions"] == [0, 0, 0]
    assert all(row["state_forecast_error"]["max_abs"] == 0 for row in report["rows"].values())
    assert report["parameters_restored"]
    assert report["verification"]["outcomes"] == 4


def test_single_file_checkpoint_diff_cli(tmp_path, capsys):
    from safetensors.torch import save_file

    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"weight": torch.eye(2)}, a)
    save_file({"weight": torch.ones(2, 2)}, b)
    assert main(["diff", str(a), str(b)]) == 0
    assert json.loads(capsys.readouterr().out)["changed_elements"] == 2


def test_patch_joins_real_parameter_ports_and_refuses_wrong_source_bytes():
    from mdb.forward_ports import NativePortCatalog
    from mdb.forward_recording import record_forward

    net = model()
    candidate = patch(net)
    def record():
        return record_forward(net, (torch.ones(1, 2),), capture_ports=True,
            model_identity={"checkpoint": "tiny"}, execution={"backend": "CPU FP32"},
            input_identity="declared-local-parent")
    catalog = NativePortCatalog(record().trace)
    joined = bind_patch_ports(candidate, catalog)
    assert joined["parameters"][0]["port"]["selector"] == "parameter:weight"
    assert joined["parameters"][0]["consumers"][0]["first_consumer"] is not None
    assert joined["input_identity"] == "declared-local-parent"
    with torch.no_grad():
        net.weight.add_(1)
    with pytest.raises(WeightSpaceError, match="source"):
        bind_patch_ports(candidate, NativePortCatalog(record().trace))

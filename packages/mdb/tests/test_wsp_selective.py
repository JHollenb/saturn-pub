import numpy as np
import pytest
import torch

from mdb.wsp_discovery import fit_output_delta
from mdb.wsp_selective import (
    fit_low_rank_delta,
    synthesize_low_rank_patch,
    virtual_low_rank_edit,
)
from mdb.wsp_weights import WeightSpaceError, parameter_inventory


def test_factored_fit_matches_dense_ridge_and_preserves_orthogonal_keys():
    rng = np.random.default_rng(7)
    h = np.concatenate((rng.normal(size=(8, 3)), np.zeros((8, 2))), axis=1)
    k = np.concatenate((np.zeros((6, 3)), rng.normal(size=(6, 2))), axis=1)
    r = rng.normal(size=(8, 4))
    a, b, report = fit_low_rank_delta(h, r, k, rank=4, ridge=.01)
    np.testing.assert_allclose(a @ b, fit_output_delta(h, r, k, ridge=.01), atol=1e-12)
    np.testing.assert_allclose(k @ b.T @ a.T, 0, atol=1e-12)
    assert report["task_rows"] == 8
    a, b, _ = fit_low_rank_delta(h, r, k, rank=1, ridge=.01)
    assert np.linalg.matrix_rank(a @ b) == 1


def test_preservation_penalty_changes_conflicting_response():
    h = np.eye(2)
    r = np.eye(2)
    weak = fit_low_rank_delta(h, r, h, rank=2, preserve_weight=0.)
    strong = fit_low_rank_delta(h, r, h, rank=2, preserve_weight=100.)
    assert np.linalg.norm(strong[0] @ strong[1]) < np.linalg.norm(weak[0] @ weak[1]) / 90
    with pytest.raises(ValueError):
        fit_low_rank_delta(h, r, h, rank=0)


def test_factored_response_compiles_restores_and_uses_current_state():
    torch.manual_seed(33)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Tanh()).double()
    a, b = np.ones((4, 2)) / 4, np.arange(10).reshape(2, 5) / 10
    x = torch.randn(3, 5, dtype=torch.float64)
    original = parameter_inventory(model)
    with virtual_low_rank_edit(model, "0", a, b, dose=.7):
        y = model(x)
    patch = synthesize_low_rank_patch(model, "0", a, b, dose=.7, model_identity="tiny")
    with patch.installed(model):
        torch.testing.assert_close(model(x), y, atol=1e-14, rtol=1e-14)
        assert not torch.equal(model(x + 1), y)
    assert parameter_inventory(model) == original
    with pytest.raises(WeightSpaceError):
        synthesize_low_rank_patch(model, "0", a, np.zeros((3, 5)), model_identity="tiny")


def test_activation_capture_is_durable_in_installed_public_packages(tmp_path):
    from mdb.forecast_transaction import load_forecast_tree
    from mdb.wsp import WeightStudy, verify_weight_study

    root = tmp_path / "capture"
    binding = dict(model="tiny", checkpoint="tiny", execution="cpu", consumer="capture", horizon=1)
    values = np.arange(1024, dtype=np.float32).reshape(32, 32)
    with WeightStudy(root, binding=binding, input_identity="x", source_identity="test") as study:
        study.add_arm("native", role="native", state_policy="restart",
                      predictor=lambda c: {"predictions": {"capture": "shape 32x32"}})
        study.seal()
        study.run("native", lambda c: {"observations": {"hidden": values}, "work": {}, "state": {}})
    import json

    record = json.loads((root / "observations/native.json").read_text())
    result = load_forecast_tree(root, record["value"])
    np.testing.assert_array_equal(result["result"]["observations"]["hidden"], values)
    assert verify_weight_study(root)["verified"]

from __future__ import annotations

import numpy as np
import pytest
import torch

from mdb.wsp_discovery import (
    AffineEdit,
    affine_sites,
    fit_output_delta,
    synthesize_affine_patch,
    virtual_edit,
)
from mdb.wsp_discovery_demo import run_discovery_demo
from mdb.wsp_weights import parameter_inventory


@pytest.mark.parametrize("mode,channels", [("write-gain", ()), ("channel-gain", (0, 2))])
def test_virtual_intervention_compiles_to_actual_affine_weights(mode, channels):
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2)).double().eval()
    x = torch.randn(5, 3).double()
    original = parameter_inventory(model)
    edit = AffineEdit("0", mode, -.3, channels)
    with virtual_edit(model, edit):
        virtual = model(x).detach()
    patch = synthesize_affine_patch(model, edit, model_identity="tiny")
    with patch.installed(model):
        compiled = model(x).detach()
    torch.testing.assert_close(virtual, compiled, atol=1e-14, rtol=1e-14)
    assert parameter_inventory(model) == original


def test_ridge_fit_uses_preservation_and_sample_space_solve():
    h = np.array([[1., 0., 0.], [0., 1., 0.]])
    r = np.array([[3., 4.], [5., 6.]])
    k = np.array([[0., 0., 1.]])
    delta = fit_output_delta(h, r, k, ridge=1e-8)
    np.testing.assert_allclose(h @ delta.T, r, atol=1e-7)
    np.testing.assert_array_equal(k @ delta.T, np.zeros((1, 2)))
    with pytest.raises(ValueError):
        fit_output_delta(h, r, ridge=-1)


def test_automatic_sites_exclude_declared_head():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = torch.nn.Linear(3, 3)
            self.head = torch.nn.Linear(3, 5)
        def get_output_embeddings(self):
            return self.head
    assert set(affine_sites(Model())) == {"hidden"}


def test_blind_fault_site_and_continuous_gain_are_discovered(tmp_path):
    report = run_discovery_demo(tmp_path / "blind", seed=731)
    assert report["site_discovered"]
    assert abs(report["selected"]["gain"] - 1 / 8) < 1e-8
    assert report["heldout_mse"]["compiled"] < 1e-20
    assert report["heldout_mse"]["native"] > 1e-5
    assert report["virtual_compiled_max_abs"] < 1e-12
    assert report["restored_native_max_abs"] == 0
    assert report["parameter_restoration"]

import pytest

from mdb.wsp_program import DigitRewrite


def test_rewrite_has_literal_operand_and_column_semantics():
    p = DigitRewrite("a", 0, 7, 3)
    assert p.apply(247, 182) == (243, 182)
    assert p.apply(274, 187) == (274, 187)
    assert DigitRewrite("b", 1, 7, 3).apply(247, 175) == (247, 135)
    with pytest.raises(ValueError):
        DigitRewrite("a", 0, 7, 7)


def test_lowering_consumes_bound_role_map_and_refuses_other_specimens():
    p = DigitRewrite("a", 0, 7, 3)
    account = {"checkpoint_sha256": "original", "execution": "fp32",
        "evidence": [{"path": "findings.md", "sha256": "pinned"}],
        "symbols": {p.symbol: {"writers": [{"module": "reader.v_proj"}], "consumer": "digit reader"}}}
    recipe = p.lower(account, checkpoint_sha256="original", execution="fp32")
    assert recipe["writers"] == account["symbols"][p.symbol]["writers"]
    assert recipe["symbol"] == p.symbol
    with pytest.raises(ValueError):
        p.lower(account, checkpoint_sha256="other", execution="fp32")
    with pytest.raises(ValueError):
        DigitRewrite("b", 0, 7, 3).lower(account, checkpoint_sha256="original", execution="fp32")


def test_frozen_response_rebuilds_exact_parameter_bytes_in_fresh_model():
    import numpy as np
    import torch

    from mdb.wsp_selective import fit_low_rank_delta, synthesize_low_rank_patch
    from mdb.wsp_weights import parameter_inventory

    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).double()
    fresh = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=False)).double()
    fresh.load_state_dict(model.state_dict())
    h, k, desired = np.eye(3)[:1], np.eye(3)[1:], np.array([[0., 2.]])
    left, right, _ = fit_low_rank_delta(h, desired, k, rank=1, ridge=1e-8)
    original = parameter_inventory(fresh)
    a = synthesize_low_rank_patch(model, "0", left, right, model_identity="frozen")
    b = synthesize_low_rank_patch(fresh, "0", left.copy(), right.copy(), model_identity="frozen")
    assert a.manifest == b.manifest
    source = torch.tensor(h)
    protected = torch.tensor(k)
    before, protected_before = fresh(source).detach(), fresh(protected).detach()
    with b.installed(fresh):
        np.testing.assert_allclose((fresh(source).detach()-before).numpy(), desired, atol=3e-8)
        torch.testing.assert_close(fresh(protected), protected_before, atol=1e-14, rtol=0)
    assert parameter_inventory(fresh) == original

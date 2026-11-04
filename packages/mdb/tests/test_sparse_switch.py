import pytest

torch = pytest.importorskip("torch")

from mdb.sparse_switch import (  # noqa: E402
    SCHEMA,
    SparseSwitchView,
    fisher_energy,
    fisher_neuron_weights,
    list_metrics,
    switch_scores,
    top_m_mask,
)


def setup(seed=0, n=32, d=6, items=40):
    gen = torch.Generator().manual_seed(seed)
    w_down = torch.randn(d, items, generator=gen, dtype=torch.float64)
    a = torch.randn(d, d, generator=gen, dtype=torch.float64)
    fisher = a @ a.T
    g = torch.randn(n, items, generator=gen)
    g_ref = torch.randn(items, generator=gen)
    u = torch.randn(n, items, generator=gen)
    return w_down, fisher, g, g_ref, u


def test_weights_match_explicit_quadratic_forms():
    w_down, fisher, *_ = setup()
    w = fisher_neuron_weights(fisher, w_down)
    explicit = torch.stack([(w_down[:, j] @ fisher @ w_down[:, j]).sqrt() for j in range(w_down.shape[1])])
    assert torch.allclose(w, explicit)


def test_top_m_mask_counts_and_identity_metrics():
    w_down, fisher, g, g_ref, u = setup()
    sc = switch_scores(g, g_ref, u, fisher_neuron_weights(fisher, w_down))
    mask = top_m_mask(sc, 5)
    assert (mask.sum(-1) == 5).all()
    met = list_metrics(sc, sc, 5)
    assert torch.allclose(met["recall"], torch.ones(len(sc)))
    assert torch.allclose(met["mass_recall"], torch.ones(len(sc)))
    assert (met["flip_in"] == 0).all() and (met["flip_out"] == 0).all() and (met["margin_ratio"] == 0).all()


def test_wrong_list_has_lower_recall_and_mass():
    w_down, fisher, g, g_ref, u = setup(1)
    sc = switch_scores(g, g_ref, u, fisher_neuron_weights(fisher, w_down))
    met = list_metrics(sc, -sc, 5)  # the bottom-5 list
    assert (met["recall"] == 0).all()
    assert (met["mass_recall"] < 1).all()
    assert met["flip_in"].mean() > 0.5


def test_view_energy_kept_and_report():
    w_down, fisher, g, g_ref, u = setup(2)
    s = ((g - g_ref) * u).double() @ w_down.T
    view = SparseSwitchView()
    view.add_switch_energy(3, fisher_energy(s, fisher))
    view.add_energy(3, "exact", fisher_energy(s - s, fisher))
    view.add_energy(3, "zero", fisher_energy(s, fisher))
    view.add_lists(3, "self", 4, s.abs().float(), s.abs().float())
    rep = view.report()
    assert rep["schema"] == SCHEMA
    assert rep["energy_kept"][3]["exact"] == pytest.approx(1.0)
    assert rep["energy_kept"][3]["zero"] == pytest.approx(0.0)
    assert rep["lists"][3]["self"][4]["recall"] == pytest.approx(1.0)
    assert rep["lists"][3]["self"][4]["margin_ratio_median"] == pytest.approx(0.0)

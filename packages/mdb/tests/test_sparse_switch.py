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


QWEN15 = {"d": 1536, "inter": 8960, "context": 128, "n_heads": 12, "n_kv": 2, "head_dim": 128}


def test_flop_fractions_match_the_derived_shapes():
    from mdb.sparse_switch import code_layer_macs, flop_fraction, layer_macs

    layer = layer_macs(**QWEN15)
    assert layer["mlp"] == 3 * 1536 * 8960
    assert 28 * layer["total"] == pytest.approx(1.32e9, rel=0.01)
    for p, f in ((512, 0.225), (1024, 0.275), (2048, 0.375)):
        assert flop_fraction([("pool", {"p": p})] * 28, QWEN15) == pytest.approx(f, abs=0.002)
    assert flop_fraction([("native", {})] * 28, QWEN15) == pytest.approx(1.0)
    assert code_layer_macs("global_base", QWEN15) < code_layer_macs("token_base", QWEN15)


def test_global_base_products_equal_the_base_matrix():
    from mdb.sparse_switch import code_products, global_base_matrix

    gen = torch.Generator().manual_seed(3)
    d, items = 5, 11
    w_down = torch.randn(d, items, generator=gen, dtype=torch.float64)
    w_up = torch.randn(items, d, generator=gen, dtype=torch.float64)
    n = torch.randn(4, d, generator=gen, dtype=torch.float64)
    g_glob = torch.rand(items, generator=gen, dtype=torch.float64)
    g = torch.randn(4, items, generator=gen, dtype=torch.float64)
    u = n @ w_up.T
    none = torch.zeros(items, dtype=torch.bool)
    write = code_products(g, u, none, g_glob) @ w_down.T
    assert torch.allclose(write, n @ global_base_matrix(w_down, g_glob, w_up).T)
    keep = torch.zeros(items, dtype=torch.bool)
    keep[:3] = True
    exact = code_products(g, u, keep, g_glob)
    assert torch.allclose(exact[:, :3], g[:, :3] * u[:, :3]) and torch.allclose(exact[:, 3:], g_glob[3:] * u[:, 3:])
    g_tok = torch.rand(items, generator=gen, dtype=torch.float64)
    hyb = code_products(g, u, none, g_glob, hybrid=(g_tok, u))
    assert torch.allclose(hyb, g_tok * u)


def test_pool_mass_and_spearman():
    from mdb.sparse_switch import pool_from_counts, pool_mass, spearman

    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    pool = pool_from_counts(torch.tensor([9, 0, 8, 0, 0]), 2)
    assert pool.tolist() == [True, False, True, False, False]
    assert float(pool_mass(scores, pool, 2)[0]) == pytest.approx(5 / 9)
    x = torch.arange(10.0)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)

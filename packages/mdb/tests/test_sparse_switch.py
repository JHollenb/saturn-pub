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


def _spd(gen, d, scale=1.0):
    a = torch.randn(d, d, generator=gen, dtype=torch.float64) * scale
    return a @ a.T + 1e-3 * torch.eye(d, dtype=torch.float64)


def test_metric_proxy_is_pca_under_the_identity_metric():
    from mdb.sparse_switch import metric_proxy

    gen = torch.Generator().manual_seed(4)
    d, k = 9, 3
    S = _spd(gen, d)
    pr = metric_proxy(torch.eye(d, dtype=torch.float64), S, k)
    ev, Q = torch.linalg.eigh(S)
    Pk = Q[:, -k:]
    assert torch.allclose(pr["decode"] @ pr["encode"].T, Pk @ Pk.T, atol=1e-8)
    assert pr["share"] == pytest.approx(float(ev[-k:].sum() / ev.sum()))


def test_metric_proxy_matches_the_reduced_rank_solution_and_beats_pca():
    from mdb.sparse_switch import metric_proxy, proxy_metric_share

    gen = torch.Generator().manual_seed(5)
    d, k = 10, 3
    S, M = _spd(gen, d), _spd(gen, d)
    pr = metric_proxy(M, S, k)
    lm, V = torch.linalg.eigh(M)
    Mh, Mmh = V @ torch.diag(lm.sqrt()) @ V.T, V @ torch.diag(lm.rsqrt()) @ V.T
    lam, U = torch.linalg.eigh(Mh @ S @ Mh)
    Uk = U[:, -k:]
    B = Mmh @ Uk @ Uk.T @ Mh
    assert torch.allclose(pr["decode"] @ pr["encode"].T, B, atol=1e-7)
    share = proxy_metric_share(M, S, pr["encode"], pr["decode"])
    assert share == pytest.approx(pr["share"], abs=1e-9)
    ev, Q = torch.linalg.eigh(S)
    assert proxy_metric_share(M, S, Q[:, -k:], Q[:, -k:]) <= share + 1e-12


def test_proxy_gate_inputs_with_centres():
    from mdb.sparse_switch import proxy_gate_inputs

    gen = torch.Generator().manual_seed(6)
    d, k, items = 6, 2, 7
    n, c = torch.randn(4, d, generator=gen), torch.randn(4, d, generator=gen)
    enc = torch.randn(d, k, generator=gen)
    wg, wu = torch.randn(items, d, generator=gen), torch.randn(items, d, generator=gen)
    dec = torch.randn(d, k, generator=gen)
    a, u = proxy_gate_inputs(n, enc, wg @ dec, wu @ dec, center_n=c, center_a=c @ wg.T, center_u=c @ wu.T)
    B = dec @ enc.T
    nhat = c + (n - c) @ B.T
    assert torch.allclose(a, nhat @ wg.T, atol=1e-5) and torch.allclose(u, nhat @ wu.T, atol=1e-5)


def test_hybrid_error_scores_are_the_hybrid_base_residual():
    from mdb.sparse_switch import code_products, hybrid_error_scores

    gen = torch.Generator().manual_seed(7)
    items = 8
    g, u = torch.randn(3, items, generator=gen), torch.randn(3, items, generator=gen)
    g_tok, u_tok, g_glob = torch.rand(items, generator=gen), torch.randn(3, items, generator=gen), torch.rand(items, generator=gen)
    none = torch.zeros(items, dtype=torch.bool)
    resid = g * u - code_products(g, u, none, g_glob, hybrid=(g_tok, u_tok))
    w = torch.rand(items, generator=gen)
    assert torch.allclose(hybrid_error_scores(g, u, g_tok, u_tok, g_glob, w), resid.abs() * w, atol=1e-6)


def test_serving_ledger_and_column_equivalents():
    from mdb.sparse_switch import (
        ServingLedger,
        column_equivalents,
        column_equivalents_decide_after,
        readout_fraction,
        top1_margin,
    )

    assert column_equivalents(0.331, 0.0) == pytest.approx(331.0)
    assert column_equivalents(0.331, 1.0) == pytest.approx(1000.0)
    assert column_equivalents_decide_after(0.331, 0.25) == pytest.approx(581.0)
    assert readout_fraction(QWEN15, 151936, 28) == pytest.approx(0.177, abs=0.002)
    logits = torch.tensor([[3.0, 1.0, 0.5], [2.0, 1.9, -1.0]])
    assert torch.allclose(top1_margin(logits), torch.tensor([2.0, 0.1]))
    led = ServingLedger()
    pos = torch.tensor([4, 20, 70, 100])
    led.add(pos, torch.tensor([True, False, False, True]), torch.tensor([True, True, False, True]), torch.tensor([0.0, 1.0, 2.0, 1.0]))
    rep = led.report(f=0.5)
    assert rep["all"]["n"] == 4 and rep["all"]["rho"] == pytest.approx(0.5) and rep["all"]["agree"] == pytest.approx(0.75)
    assert rep["all"]["column_equivalents"] == pytest.approx(750.0)
    assert rep["64-127"]["n"] == 2 and rep["64-127"]["kl"] == pytest.approx(1.5)


def test_serving_ledger_splits_served_and_native():
    from mdb.sparse_switch import ServingLedger, column_equivalents_early_abort

    led = ServingLedger()
    pos = torch.tensor([4, 5, 6, 7])
    led.add(pos, torch.tensor([True, True, False, False]), torch.tensor([True, True, True, False]), torch.tensor([0.0, 0.2, 1.0, 3.0]))
    rep = led.report(f=0.3, readout=0.177, f_pre=0.1)["all"]
    assert rep["agree_native"] == pytest.approx(1.0) and rep["agree_served"] == pytest.approx(0.5)
    assert rep["kl_served"] == pytest.approx(2.0) and rep["kl_native"] == pytest.approx(0.1)
    assert rep["decide_after_with_readout"] == pytest.approx(1000 * (0.3 + 0.5) + 177)
    assert rep["column_equivalents_early_abort"] == pytest.approx(column_equivalents_early_abort(0.3, 0.1, 0.5))
    assert column_equivalents_early_abort(0.3, 0.1, 0.5) == pytest.approx(1000 * (0.5 * 0.3 + 0.5 * 1.1))


def test_ridge_from_moments_matches_the_explicit_gcv_ridge():
    from mdb.sparse_switch import ridge_from_moments

    gen = torch.Generator().manual_seed(8)
    n, p, q = 60, 5, 3
    X = torch.randn(n, p, generator=gen, dtype=torch.float64)
    Y = X @ torch.randn(p, q, generator=gen, dtype=torch.float64) + 0.5 * torch.randn(n, q, generator=gen, dtype=torch.float64)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    fit = ridge_from_moments(Xc.T @ Xc, Xc.T @ Yc, float((Yc ** 2).sum()), n, grid)
    best = None
    for lam in grid:
        B = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(p, dtype=torch.float64), Xc.T @ Yc)
        H = Xc @ torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(p, dtype=torch.float64), Xc.T)
        rss = float(((Yc - Xc @ B) ** 2).sum())
        gcv = rss / (n - float(torch.trace(H))) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, lam, B, rss)
    assert fit["lam"] == best[1] and fit["rss"] == pytest.approx(best[3]) and torch.allclose(fit["beta"], best[2])


def test_shrinkage_tables_and_lambda():
    from mdb.sparse_switch import shrinkage_lambda, shrinkage_means

    counts = torch.tensor([0, 1, 9])
    emp = torch.tensor([[9.0], [2.0], [4.0]])
    prior = torch.tensor([[1.0], [0.0], [0.0]])
    cc = shrinkage_means(counts, emp, prior, 3.0)
    assert cc[:, 0].tolist() == pytest.approx([1.0, 0.5, 3.0])
    lam, between = shrinkage_lambda(2.0, torch.tensor([3.0, 1.5]), torch.tensor([1.0, 4.0]))
    assert between == pytest.approx(0.5 * ((3.0 - 2.0) + (1.5 - 0.5))) and lam == pytest.approx(2.0 / between)


def test_error_mass_and_gates():
    from mdb.sparse_switch import error_mass, kl_gate, quantile_threshold, z_gate

    sc = torch.tensor([[3.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    keep = torch.tensor([[True, False, False], [False, False, True]])
    assert error_mass(sc, keep).tolist() == pytest.approx([5.0, 2.0])
    stat = torch.arange(100.0)
    t = quantile_threshold(stat, 0.3)
    assert float((stat < t).float().mean()) == pytest.approx(0.3, abs=0.011)
    t2 = quantile_threshold(stat, 0.3, below=False)
    assert float(kl_gate(stat, t2).float().mean()) == pytest.approx(0.3, abs=0.011)
    assert z_gate(torch.tensor([1.0, 3.0]), torch.tensor([4.0, 4.0]), 1.0).tolist() == [True, False]

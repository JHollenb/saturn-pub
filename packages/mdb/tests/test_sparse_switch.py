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


def test_layer_bytes_match_the_derived_qwen_counts():
    from mdb.sparse_switch import bytes_per_token, layer_bytes

    nat = layer_bytes("native", QWEN15)
    assert nat["shared"] / 1e6 == pytest.approx(93.6, abs=0.3)
    a1 = layer_bytes("proxy", QWEN15, k=512, m=1024)
    assert a1["total"] / 1e6 == pytest.approx(45.1, abs=0.3)
    col = bytes_per_token([("native", {})] * 28, QWEN15, vocab=151936)
    prx = bytes_per_token([("proxy", {"k": 512, "m": 1024})] * 28, QWEN15, vocab=151936)
    assert prx["bytes_per_step"] / col["bytes_per_step"] == pytest.approx(0.56, abs=0.01)
    b256 = layer_bytes("proxy", QWEN15, k=512, m=1024, batch=256)
    assert b256["total"] == b256["shared"] + 256 * b256["per_row"]
    un = layer_bytes("proxy", QWEN15, k=512, batch=256, impl="union", union=8960)
    assert un["shared"] > nat["shared"]


def test_timed_record_adds_bandwidth_and_ratios():
    from mdb.sparse_switch import timed_record

    timing = {"batch": 1, "claim": "speedups reported",
              "arms": {"native": {"median_ms": 10.0, "p10_ms": 9.0, "p90_ms": 11.0},
                       "a1": {"median_ms": 8.0, "p10_ms": 7.0, "p90_ms": 9.0}}}
    rec = timed_record(timing, {"native": {"bytes_per_step": 3e9, "flop_fraction": 1.0},
                                "a1": {"bytes_per_step": 1.5e9, "flop_fraction": 0.5}})
    assert rec["arms"]["native"]["achieved_GBps"] == pytest.approx(300.0)
    assert rec["arms"]["a1"]["time_ratio_vs_baseline"] == pytest.approx(0.8)
    assert rec["arms"]["a1"]["byte_ratio_vs_baseline"] == pytest.approx(0.5)
    refused = timed_record({**timing, "claim": "refused: parity gate missing or failed"},
                           {"native": {"bytes_per_step": 3e9}, "a1": {"bytes_per_step": 1.5e9}})
    assert refused["arms"]["a1"]["time_ratio_vs_baseline"] is None


def test_pregate_table_shrinks_rare_keys_to_the_prior():
    from mdb.sparse_switch import pregate_table

    keys = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2])
    vals = torch.tensor([1.0, 1.2, 0.8, 1.0, 3.0, 3.2, 2.8, 3.0, 9.0])
    table, lam, st = pregate_table(keys, vals, 4)
    assert lam > 0 and st["keys_ge2"] == 2
    gm = float(vals.double().mean())
    assert float(table[3]) == pytest.approx(gm)
    assert gm < float(table[2]) < 9.0
    assert abs(float(table[0]) - 1.0) < abs(gm - 1.0)
    big, _, _ = pregate_table(keys, vals, 4, prior=torch.tensor([5.0, 5.0, 5.0, 5.0]))
    assert float(big[3]) == pytest.approx(5.0)


def test_quantize_rows_and_ledger_groups():
    from mdb.sparse_switch import ServingLedger, quantize_rows

    w = torch.tensor([[1.0, -0.5, 0.25], [10.0, 3.0, -7.0]])
    dq, sc = quantize_rows(w, 8)
    assert torch.allclose(dq, w, atol=float(sc.max()) / 2 + 1e-6)
    dq4, _ = quantize_rows(w, 4)
    assert float((dq4 - w).abs().max()) <= 10.0 / 7 / 2 + 1e-6
    led = ServingLedger()
    led.add(torch.tensor([4, 5, 6]), torch.tensor([False, False, True]), torch.tensor([True, False, True]),
            torch.tensor([0.0, 1.0, 0.0]), groups=torch.tensor([0, 1, 1]))
    rep = led.report()
    assert rep["group:0"]["agree"] == pytest.approx(1.0) and rep["group:1"]["n"] == 2


def test_spearman_ties_and_or_gate_thresholds():
    from mdb.sparse_switch import average_ranks, or_gate_thresholds, spearman

    assert average_ranks(torch.tensor([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]
    x = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert spearman(x, y, ties=True) == pytest.approx(0.8783, abs=1e-3)
    g = torch.Generator().manual_seed(0)
    m, e = torch.rand(20000, generator=g), torch.rand(20000, generator=g)
    th = or_gate_thresholds(m, e, 0.5)
    assert th["union_share"] == pytest.approx(0.5, abs=2e-3)
    assert th["marginal_share"] == pytest.approx(1 - 0.5 ** 0.5, abs=5e-3)  # independent: 1 - (1 - s)^2 = 0.5


def test_exact_select_cost_and_the_af_frontier_fractions():
    from mdb.sparse_switch import bytes_per_token, flop_fraction, layer_bytes

    def f(kind, **o):
        return flop_fraction([(kind, o)] * 28, QWEN15)
    assert f("exact_select", m=1024) == pytest.approx(0.792, abs=0.001)
    assert f("exact_select", m=2048) == pytest.approx(0.825, abs=0.001)
    assert f("proxy", k=512, m=2048) == pytest.approx(0.586, abs=0.001)
    assert f("proxy", k=768, m=1024) == pytest.approx(0.592, abs=0.001)
    assert f("proxy", k=1024, m=512) == pytest.approx(0.647, abs=0.001)
    x1 = layer_bytes("exact_select", QWEN15, m=1024)
    nat = layer_bytes("native", QWEN15)
    assert nat["shared"] > x1["shared"] > 2 * 1536 * 8960 * 2
    col = bytes_per_token([("native", {})] * 28, QWEN15, vocab=151936)
    xb = bytes_per_token([("exact_select", {"m": 1024})] * 28, QWEN15, vocab=151936)
    assert xb["bytes_per_step"] / col["bytes_per_step"] == pytest.approx(0.822, abs=0.002)


def test_kl_law_fit_and_ledger_columns():
    from mdb.sparse_switch import ServingLedger, fit_kl_law, kl_law

    kl = torch.tensor([0.18, 0.25, 0.42, 0.51, 1.3, 2.7], dtype=torch.float64)
    a = 0.967 - 0.403 * kl.sqrt()
    a0, beta, rms = fit_kl_law(a, kl)
    assert a0 == pytest.approx(0.967, abs=1e-9) and beta == pytest.approx(0.403, abs=1e-9) and rms < 1e-9
    assert kl_law(0.25) == pytest.approx(0.967 - 0.403 * 0.5)
    led = ServingLedger()
    led.add(torch.tensor([4, 5]), torch.tensor([False, False]), torch.tensor([True, False]), torch.tensor([0.25, 0.25]))
    r = led.report(law=(0.967, 0.403))["all"]
    assert r["agree_law"] == pytest.approx(0.7655) and r["law_excess"] == pytest.approx(0.5 - 0.7655)


def test_score_histogram_thresholds_hit_the_mean_count():
    from mdb.sparse_switch import ScoreHistogram

    g = torch.Generator().manual_seed(0)
    s0 = torch.randn(400, 2000, generator=g).exp()
    s1 = (torch.randn(400, 2000, generator=g) * 2).exp() * 3
    h = ScoreHistogram(2, bins=20000)
    h.add(0, s0, values=s0)
    h.add(1, s1, values=s1)
    for layer, s in ((0, s0), (1, s1)):
        th = h.threshold(layer, 250)
        assert float((s > th).sum(-1).float().mean()) == pytest.approx(250, rel=0.01)
        cnt, mass, _ = h.above(layer, th)
        assert cnt == pytest.approx(250, rel=0.01)
        assert mass == pytest.approx(float((s * (s > th)).sum(-1).mean()), rel=0.01)
    tg = h.global_threshold(500)
    tot = float(((s0 > tg).sum(-1) + (s1 > tg).sum(-1)).float().mean())
    assert tot == pytest.approx(500, rel=0.01)


def test_ramp_weights_and_products():
    from mdb.sparse_switch import code_products, ramp_products, ramp_weights

    s = torch.tensor([[0.5, 1.0, 1.1, 1.2, 2.0]])
    keep, om = ramp_weights(s, 1.0, 0.0)
    assert keep.tolist() == [[False, False, True, True, True]] and om.tolist() == [[0, 0, 1, 1, 1]]
    keep, om = ramp_weights(s, 1.0, 0.25)  # theta_hi = 4/3
    assert keep.tolist() == [[False, False, True, True, True]]
    assert om[0, 2] == pytest.approx(0.3, abs=1e-6) and om[0, 3] == pytest.approx(0.6, abs=1e-6) and om[0, 4] == 1.0
    g, u = torch.randn(3, 5), torch.randn(3, 5)
    gt, ut, gg = torch.randn(3, 5), torch.randn(3, 5), torch.randn(5)
    hard = torch.rand(3, 5) > 0.5
    assert torch.allclose(ramp_products(g, u, hard.float(), gg, hybrid=(gt, ut)),
                          code_products(g, u, hard, gg, hybrid=(gt, ut)), atol=1e-6)


def test_rank_waterfill_takes_the_largest_eigenvalues():
    from mdb.sparse_switch import rank_waterfill

    e = [torch.tensor([10.0, 1.0, 0.5, 0.1]), torch.tensor([5.0, 4.0, 3.0, 2.0])]
    r = rank_waterfill(e, 4)
    assert r["k"] == [1, 3] and r["theta"] == 3.0
    assert r["kept_share"] == pytest.approx(22.0 / 25.6) and r["uniform_kept_share"] == pytest.approx(20.0 / 25.6)
    rc = rank_waterfill(e, 4, cap=2)
    assert rc["k"] == [2, 2] and rc["cap_binds"]


def test_group_sets_and_jaccard():
    from mdb.sparse_switch import group_sets, set_jaccard

    g = torch.Generator().manual_seed(1)
    s = torch.rand(6, 50, generator=g)
    assert torch.equal(group_sets(s, 1, 10), top_m_mask(s, 10))
    gs = group_sets(s, 2, 10)
    assert torch.equal(gs[0], gs[1]) and int(gs[0].sum()) == 10
    assert torch.equal(gs[0], top_m_mask(s[0] + s[1], 10))
    j = set_jaccard(top_m_mask(s, 10), top_m_mask(s, 10))
    assert torch.allclose(j, torch.ones(6))
    tail = group_sets(s[:5], 2, 10)
    assert torch.equal(tail[4], top_m_mask(s[4], 10))


def test_integer_quantization_and_int4_packing():
    from mdb.sparse_switch import (
        dequantize_int,
        pack_int4,
        quantize_int,
        quantize_rows,
        unpack_int4,
    )

    g = torch.Generator().manual_seed(2)
    w = torch.randn(6, 256, generator=g)
    q, s = quantize_int(w, 8)
    assert q.dtype == torch.int8 and int(q.abs().max()) <= 127
    assert torch.allclose(dequantize_int(q, s), quantize_rows(w, 8)[0], atol=1e-6)
    q4, s4 = quantize_int(w, 4, group=128)
    assert s4.shape == (6, 2) and int(q4.abs().max()) <= 7
    err4 = (dequantize_int(q4, s4, 128) - w).abs().max()
    assert float(err4) <= float(s4.max()) / 2 + 1e-6
    assert torch.equal(unpack_int4(pack_int4(q4)), q4)
    assert pack_int4(q4).shape == (6, 128) and pack_int4(q4).dtype == torch.uint8


def test_quantized_layer_bytes():
    from mdb.sparse_switch import bytes_per_token, layer_bytes

    sh = {"d": 8, "inter": 32, "context": 4, "n_heads": 2, "n_kv": 1, "head_dim": 4}
    full = layer_bytes("exact_select", sh, m=4)
    q8 = layer_bytes("exact_select", sh, m=4, qbits=8)
    assert full["shared"] - q8["shared"] == 2 * 32 * 8 * 2 - (2 * 32 * 8 + 2 * 32 * 4)
    q4g = layer_bytes("proxy", sh, k=8, m=4, qbits=4, group=4)
    assert q4g["shared"] == layer_bytes("proxy", sh, k=8, m=4)["shared"] - 2 * 32 * 8 * 2 + 2 * 32 * 8 * 0.5 + 2 * 32 * 2 * 4
    b16 = bytes_per_token([("native", {})], sh, vocab=10)["bytes_per_step"]
    b8 = bytes_per_token([("native", {})], sh, vocab=10, readout_bits=8)["bytes_per_step"]
    assert b16 - b8 == 10 * 8 * 2 - (10 * 8 + 10 * 4)


def test_certified_candidates_contain_the_argmax_and_blockwise_recall():
    from mdb.sparse_switch import blockwise_topk_recall, certified_candidates, head_subspace

    g = torch.Generator().manual_seed(3)
    W = torch.randn(500, 16, generator=g) @ torch.diag(torch.linspace(3, 0.1, 16))
    h = torch.randn(40, 16, generator=g)
    basis, ev = head_subspace(W, 6)
    assert basis.shape == (16, 6) and bool((ev[:-1] >= ev[1:]).all())
    Pe, Ph = W @ basis, h @ basis
    lhat = Ph @ Pe.T
    e_perp = ((W * W).sum(-1) - (Pe * Pe).sum(-1)).clamp_min(0).sqrt()
    h_perp = ((h * h).sum(-1) - (Ph * Ph).sum(-1)).clamp_min(0).sqrt()
    mask, sizes = certified_candidates(lhat, h_perp, e_perp)
    top = (h @ W.T).argmax(-1)
    assert bool(mask[torch.arange(40), top].all()) and bool((sizes >= 1).all())
    assert blockwise_topk_recall(h @ W.T, top, 64, 1) == 1.0
    assert 0.0 <= blockwise_topk_recall(lhat, top, 64, 1) <= 1.0


def test_gate_frontier_and_dispersion():
    from mdb.sparse_switch import gate_frontier, kl_law, kl_log_dispersion, metric_eigen_share

    g = torch.Generator().manual_seed(4)
    P, L = 2000, 4
    kl = (torch.randn(P, generator=g) * 1.5 - 2).exp()
    counts = kl[:, None] * torch.tensor([1.0, 2.0, 3.0, 4.0]) + 0.01 * torch.rand(P, L, generator=g)
    agree = torch.rand(P, generator=g) > kl.clamp_max(1)
    fr = gate_frontier(counts, kl, agree, 0.5, [0.1, 0.2, 0.3, 0.5], depths=[1, 3], rhos=[0.0, 0.2, 0.4], budget=800)
    r0 = [r for r in fr["rows"] if r["rho"] == 0.0][0]
    assert r0["rho_realized"] == 0.0 and r0["predicted"] == pytest.approx(kl_law(float(kl.mean())))
    r2 = [r for r in fr["rows"] if r["depth"] == 3 and r["rho"] == 0.2][0]
    assert r2["rho_realized"] == pytest.approx(0.2, abs=0.01) and r2["kl_kept"] < float(kl.mean())
    assert r2["cost"] == pytest.approx(1000 * (0.8 * 0.5 + 0.2 * 1.5), rel=0.02)
    assert fr["best"] is None or fr["best"]["cost"] <= 800
    assert kl_log_dispersion(kl) == pytest.approx(float(kl.log().std()), rel=1e-6)
    S = torch.diag(torch.tensor([4.0, 1.0, 0.0]))
    sh = metric_eigen_share(S, torch.eye(3), [1, 2])
    assert sh[1] == pytest.approx(0.8) and sh[2] == pytest.approx(1.0)


def test_prior_art_keys_products_and_masks():
    from mdb.sparse_switch import (
        below_threshold_mass,
        cats_scores,
        gate_key_scores,
        magnitude_mask,
        zero_base_products,
        zero_base_scores,
    )

    g = torch.tensor([[0.5, -0.2, 2.0, 0.0]])
    u = torch.tensor([[1.0, 3.0, -1.0, 4.0]])
    w = torch.tensor([1.0, 2.0, 0.5, 1.0])
    assert torch.equal(cats_scores(g), g.abs())
    assert torch.allclose(gate_key_scores(g, torch.tensor([0.5, 0.0, 1.0, 1.0]), u, w), torch.tensor([[0.0, 1.2, 0.5, 4.0]]))
    assert torch.allclose(zero_base_scores(g, u, w), torch.tensor([[0.5, 1.2, 1.0, 0.0]]))
    keep = torch.tensor([[True, False, True, False]])
    assert torch.equal(zero_base_products(g, u, keep), torch.tensor([[0.5, 0.0, -2.0, 0.0]]))
    assert torch.equal(magnitude_mask(u, 2.0), torch.tensor([[0.0, 3.0, 0.0, 4.0]]))
    s = torch.tensor([[1.0, 2.0, 3.0]])
    assert float(below_threshold_mass(s, 2.0)) == pytest.approx(5.0)


def test_attention_cats_teal_macs_and_bytes():
    from mdb.sparse_switch import code_layer_macs, layer_bytes, layer_macs

    sh = {"d": 8, "inter": 32, "context": 4, "n_heads": 2, "n_kv": 1, "head_dim": 4}
    L = layer_macs(**sh)
    att = L["attention"]
    assert code_layer_macs("cats", sh, m=4) == att + 8 * 32 + 2 * 4 * 8
    assert code_layer_macs("cats_hybrid", sh, m=4) - code_layer_macs("cats", sh, m=4) == 64
    assert code_layer_macs("exact_select_zero", sh, m=4) == att + 2 * 8 * 32 + 4 * 8
    assert code_layer_macs("exact_select", sh, m=4, r=2) - code_layer_macs("exact_select", sh, m=4) == 2 * (32 + 8)
    assert code_layer_macs("exact_select", sh, m=4, regather=True) - code_layer_macs("exact_select", sh, m=4) == 2 * 4 * 8
    assert code_layer_macs("teal", sh, density=1.0) == pytest.approx(L["total"])
    sc = 2 * 4 * 2 * 4
    assert code_layer_macs("teal", sh, density=0.5) == pytest.approx(0.5 * (L["total"] - sc) + sc)
    assert code_layer_macs("attention_only", sh) == att
    c0 = layer_bytes("cats", sh, m=4, hybrid=False)
    c1 = layer_bytes("cats", sh, m=4, hybrid=True)
    assert c1["shared"] - c0["shared"] == 64 * 2 and c1["per_row"] - c0["per_row"] == (2 * 32 + 2 * 8) * 2
    t1 = layer_bytes("teal", sh, density=1.0)
    nat = layer_bytes("native", sh)
    assert t1["total"] == pytest.approx(nat["total"])
    assert layer_bytes("teal", sh, density=0.5)["shared"] < t1["shared"]
    rg = layer_bytes("exact_select", sh, m=4, regather=True)["per_row"] - layer_bytes("exact_select", sh, m=4)["per_row"]
    assert rg == 2 * 4 * 8 * 2


def test_layer_choice_frontier_is_greedy_and_monotone():
    from mdb.sparse_switch import layer_choice_frontier

    ea, eb = [4.0, 1.0, 3.0, 2.0], [1.0, 1.5, 2.0, 0.0]
    ta, tb = [1.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.5, 1.0]
    fr = layer_choice_frontier(ea, eb, ta, tb)
    assert fr["order"][0] == 3  # faster and better: swapped first
    assert 1 not in fr["order"]  # worse and slower: never swapped
    assert fr["order"][1:] == [0, 2]  # 3 / 1 = 3 before 1 / 0.5 = 2
    errs = [r["error"] for r in fr["rows"]]
    assert errs == sorted(errs, reverse=True) and fr["rows"][-1]["error"] == pytest.approx(1.0 + 1.0 + 2.0 + 0.0)


def test_tail_proxy_maps_project_in_the_metric():
    from mdb.sparse_switch import tail_proxy_maps

    g = torch.Generator().manual_seed(7)
    d, I = 6, 20
    Wd = torch.randn(d, I, generator=g, dtype=torch.float64)
    t = torch.randn(500, I, generator=g, dtype=torch.float64) * torch.linspace(2, 0.1, I, dtype=torch.float64)
    y = t @ Wd.T
    S = y.T @ y / len(y)
    A = torch.randn(d, d, generator=g, dtype=torch.float64)
    M = A @ A.T + 0.1 * torch.eye(d, dtype=torch.float64)
    E, D, share = tail_proxy_maps(S, M, Wd, d)
    assert share == pytest.approx(1.0)
    assert torch.allclose((t @ E.double().T) @ D.double().T, y, atol=1e-4)  # full rank: exact
    E2, D2, s2 = tail_proxy_maps(S, M, Wd, 2)
    yh = (t @ E2.double().T) @ D2.double().T
    err = (((y - yh) @ M) * (y - yh)).sum(-1).mean() / ((y @ M) * y).sum(-1).mean()
    assert float(err) == pytest.approx(1 - s2, rel=1e-3, abs=1e-6)


def test_readout_kl_ties_sigma_set_and_key_frontier():
    from mdb.sparse_switch import (
        bf16_tie_mask,
        certified_candidates,
        key_gate_frontier,
        kl_law,
        readout_kl_stats,
        sigma_candidates,
    )

    lg = torch.tensor([[2.0, 2.0, 1.0], [3.0, 1.0, 0.0]]).to(torch.bfloat16)
    assert bf16_tie_mask(lg).tolist() == [True, False]
    ref = torch.tensor([[1.0, 0.0, -1.0]])
    kl, ag = readout_kl_stats(ref, ref)
    assert float(kl) == pytest.approx(0.0, abs=1e-7) and bool(ag)
    lhat = torch.tensor([[1.0, 0.9, 0.0]])
    hp, ep = torch.tensor([0.1]), torch.tensor([1.0, 1.0, 1.0])
    m_cs, s_cs = certified_candidates(lhat, hp, ep)
    m_s, s_s = sigma_candidates(lhat, hp, ep, 1.0, 4)
    assert int(s_s) <= int(s_cs) and bool((m_s <= m_cs).all())
    g = torch.Generator().manual_seed(5)
    kl_ = (torch.randn(4000, generator=g) - 3).exp()
    key = kl_ * (1 + 0.1 * torch.rand(4000, generator=g))
    agree = torch.rand(4000, generator=g) > 0.1
    fr = key_gate_frontier(key, kl_, agree, 0.8, [0.0, 0.1], repair=0.0)
    assert fr["rows"][0]["predicted"] == pytest.approx(kl_law(float(kl_.mean())))
    assert fr["rows"][1]["kl_kept"] < float(kl_.mean()) and fr["rows"][1]["cost"] == pytest.approx(1000 * (0.8 + 0.1), rel=0.02)
    assert fr["spearman_key_kl"] > 0.9


def test_wina_weight_dispersion_and_svd_remainder():
    from mdb.sparse_switch import (
        column_norm_weights,
        fisher_neuron_weights,
        logit_gain_dispersion,
        svd_remainder_maps,
    )

    g = torch.Generator().manual_seed(11)
    d, I = 8, 40
    Wd = torch.randn(d, I, generator=g, dtype=torch.float64)
    wn = column_norm_weights(Wd)
    assert torch.allclose(wn.double(), torch.stack([Wd[:, j].norm() for j in range(I)]), rtol=1e-5)
    psi = fisher_neuron_weights(torch.eye(d, dtype=torch.float64), Wd)  # isotropic metric: psi = l2 norm, rho = 1
    disp = logit_gain_dispersion(psi, wn)
    assert disp["sd_log_rho"] < 1e-5 and disp["spearman_psi_wnorm"] == pytest.approx(1.0)
    A = torch.diag(torch.linspace(3, 0.1, d, dtype=torch.float64))
    disp2 = logit_gain_dispersion(fisher_neuron_weights(A @ A.T, Wd), wn)
    assert disp2["sd_log_rho"] > 0.05
    E, D, share = svd_remainder_maps(Wd, d)
    t = torch.randn(30, I, generator=g, dtype=torch.float64)
    assert share == pytest.approx(1.0) and torch.allclose((t @ E.double().T) @ D.double().T, t @ Wd.T, atol=1e-4)
    E2, D2, s2 = svd_remainder_maps(Wd, 3)
    U, S, Vh = torch.linalg.svd(Wd, full_matrices=False)
    assert torch.allclose(D2.double() @ E2.double(), (U[:, :3] * S[:3]) @ Vh[:3], atol=1e-4)
    assert s2 == pytest.approx(float((S[:3] ** 2).sum() / (S ** 2).sum()), rel=1e-6)


def test_teal_greedy_allocation_follows_the_cheapest_error_and_lands_on_target():
    from mdb.sparse_switch import (
        TEAL_MATRICES,
        code_layer_macs,
        teal_footprints,
        teal_greedy_allocation,
    )

    shapes = {"d": 16, "inter": 64, "context": 8, "n_heads": 4, "n_kv": 2, "head_dim": 4}
    foot = teal_footprints(shapes)
    assert tuple(foot) == TEAL_MATRICES
    cost = {"q": 1.0, "k": 5.0, "v": 5.0, "o": 2.0, "gate": 0.5, "up": 3.0, "down": 0.2}
    res = teal_greedy_allocation(lambda p: sum(cost[n_] * p[n_] ** 2 for n_ in p), foot, 0.5, alpha=0.05, cap=0.95)
    F = sum(foot.values())
    assert res["P"] == pytest.approx(0.5) and sum(res["sparsity"][n_] * foot[n_] for n_ in foot) / F == pytest.approx(0.5)
    assert res["path"][0]["matrix"] == "down"  # lowest error per step first
    assert all(0 <= v <= 0.95 + 1e-12 for v in res["sparsity"].values())
    dens = {n_: 1 - v for n_, v in res["sparsity"].items()}
    full = code_layer_macs("native", shapes)
    uni = code_layer_macs("teal", shapes, density=0.5)
    assert code_layer_macs("teal_alloc", shapes, densities=dens) == pytest.approx(uni)  # footprint-weighted = MAC-weighted
    assert code_layer_macs("teal_alloc", shapes, densities={n_: 1.0 for n_ in foot}) == pytest.approx(full)


def test_g2u_costs_rsp_remainder_and_honest_gate_cost():
    from mdb.sparse_switch import (
        ServingLedger,
        code_layer_macs,
        column_equivalents_decide_after_honest,
        key_gate_frontier,
        layer_bytes,
    )

    shapes = {"d": 1536, "inter": 8960, "context": 128, "n_heads": 12, "n_kv": 2, "head_dim": 128}
    full = code_layer_macs("native", shapes)
    g2u = code_layer_macs("gate_exact_up_proxy", shapes, k=256, m=3007) / full
    assert g2u == pytest.approx(0.719, abs=0.003)  # ADVISOR 16.3: G2U(3,007) = 0.719
    x8 = code_layer_macs("exact_select", shapes, m=3007) / full
    assert x8 == pytest.approx(0.856, abs=0.003)
    assert code_layer_macs("exact_select_zero", shapes, m=10, r=256) - code_layer_macs("exact_select_zero", shapes, m=10) == 256 * (8960 + 1536)
    b = layer_bytes("g2u", shapes, k=256, m=2255, qbits=8)["total"] / 1e6
    assert 38 < b < 46  # ADVISOR 16.3: about 42 MB at 2,255
    t8 = layer_bytes("exact_select", shapes, m=10, qbits=8, r=128, tail_qbits=8)["total"]
    t16 = layer_bytes("exact_select", shapes, m=10, qbits=8, r=128)["total"]
    assert t8 < t16
    assert column_equivalents_decide_after_honest(0.8317, 0.06, 0.177, needs_readout=True) == pytest.approx(902.3, abs=0.1)
    assert column_equivalents_decide_after_honest(0.8317, 0.06, 0.177, needs_readout=False) == pytest.approx(891.7, abs=0.1)
    led = ServingLedger()
    pos = torch.arange(4, 12)
    led.add(pos, pos < 6, torch.ones(8, dtype=torch.bool), torch.zeros(8))
    r = led.report(f=0.8, readout=0.177, needs_readout=True)["all"]
    assert r["decide_after_honest"] == pytest.approx(1000 * (0.8 + 0.25 * 1.177))
    g = torch.Generator().manual_seed(3)
    kl_ = torch.rand(1000, generator=g)
    fr = key_gate_frontier(kl_, kl_, kl_ > 0.5, 0.8, [0.1], needs_readout=True, readout=0.177)
    assert fr["rows"][0]["cost_honest"] == pytest.approx(fr["rows"][0]["cost"] + 1000 * 0.1 * 0.177, rel=1e-3)


def test_g2u_scores_reduce_to_exact_selection_at_full_rank():
    from mdb.sparse_switch import gate_exact_up_proxy_scores, hybrid_error_scores

    g_ = torch.Generator().manual_seed(4)
    d, I = 6, 30
    Wu = torch.randn(I, d, generator=g_, dtype=torch.float64)
    n = torch.randn(5, d, generator=g_, dtype=torch.float64)
    nt = torch.randn(5, d, generator=g_, dtype=torch.float64)
    gate = torch.randn(5, I, generator=g_, dtype=torch.float64)
    gt, gg, w = torch.randn(5, I, generator=g_), torch.randn(I, generator=g_), torch.rand(I, generator=g_)
    ut = nt @ Wu.T
    sc, uh = gate_exact_up_proxy_scores(gate, n, nt, torch.eye(d, dtype=torch.float64), Wu, ut, gt, gg, w)
    assert torch.allclose(uh.double(), n @ Wu.T, atol=1e-4)
    assert torch.allclose(sc, hybrid_error_scores(gate, n @ Wu.T, gt, ut, gg, w), atol=1e-4)

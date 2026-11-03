import json

import numpy as np
import pytest

from mdb.cli import main as cli_main
from mdb.compiled_state import (
    CompiledStateError,
    CompiledStateMap,
    CompilerFit,
    FeatureSet,
    PromptFamily,
    ResidualAccumulator,
    Symbol,
    agreement,
    columns,
    design,
    horizon,
    main,
    render,
    seal_report,
    verify_maps,
    verify_report,
    visibility,
    wilson,
)


def family(**kw):
    """Four positions x0 x1 op y: x0 at 0, x1 at 1, pair sum s (card 5) and gate g (card 2) at 1."""
    base = dict(name="toy", prefix="shots\n", positions=("x0", "x1", "op", "y"),
                symbols=(Symbol("x0", 3, 0), Symbol("x1", 3, 1), Symbol("s", 5, 1), Symbol("g", 2, 1)),
                feature_sets={"digits": FeatureSet(("x0", "x1")), "digits+s": FeatureSet(("x0", "x1", "s")),
                              "digits|g": FeatureSet(("x0", "x1"), switch="g")},
                score_positions=(2, 3))
    return PromptFamily(**{**base, **kw})


def values_for(x0, x1):
    s = x0 + x1
    return np.stack([x0, x1, s, (s >= 2).astype(int)], 1)


def grid():
    x0, x1 = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    return values_for(x0.ravel(), x1.ravel())


def test_family_layout_fingerprint_and_round_trip():
    fam = family()
    assert fam.width == 1 + 3 + 3 + 5 + 2
    assert fam.offsets() == {"x0": 1, "x1": 4, "s": 7, "g": 12}
    back = PromptFamily.from_dict(json.loads(json.dumps(fam.to_dict())))
    assert back.fingerprint == fam.fingerprint and back.feature_sets["digits|g"].switch == "g"
    tampered = {**fam.to_dict(), "prefix": "other"}
    with pytest.raises(CompiledStateError):
        PromptFamily.from_dict(tampered)
    with pytest.raises(CompiledStateError):
        family(symbols=(Symbol("x0", 3, 4),))
    with pytest.raises(CompiledStateError):
        family(feature_sets={"bad": FeatureSet(("nope",))})


def test_visibility_is_causal_and_columns_select_blocks():
    fam = family()
    V = visibility(fam)
    assert V[:, 0].all()
    assert V[0, 1:4].all() and not V[0, 4:].any()  # position 0 sees bias + x0 only
    assert V[1:, 4:].all()
    cols = columns(fam, "digits")
    assert cols[:7].all() and not cols[7:].any()


def test_design_is_one_hot_and_rejects_out_of_range():
    fam = family()
    X = design(fam, values_for(np.array([2]), np.array([1])))
    assert X.sum() == 5 and X[0, 0] == 1 and X[0, 1 + 2] == 1 and X[0, 4 + 1] == 1 and X[0, 7 + 3] == 1
    with pytest.raises(CompiledStateError):
        design(fam, np.array([[3, 0, 0, 0]]))
    with pytest.raises(CompiledStateError):
        design(fam, np.zeros((1, 3), int))


def linear_states(fam, values, rng, D=6):
    """States exactly linear in each position's visible digits (no s, no g)."""
    W = rng.normal(size=(len(fam.positions), fam.width, D))
    P = design(fam, values)[:, None, :] * visibility(fam)[None] * columns(fam, "digits")[None, None]
    return np.einsum("bpf,pfd->bpd", P, W)


def test_ridge_recovers_linear_states_and_streams_exactly():
    rng = np.random.default_rng(0)
    fam = family()
    vals = np.repeat(grid(), 20, 0)
    X = linear_states(fam, vals, rng)
    one = CompilerFit(fam, [3], X.shape[2])
    one.add(vals, {3: X})
    two = CompilerFit(fam, [3], X.shape[2])
    two.add(vals[:50], {3: X[:50]})
    two.add(vals[50:], {3: X[50:]})
    np.testing.assert_allclose(one.gram, two.gram)
    np.testing.assert_allclose(one.xty[3], two.xty[3])
    m = one.solve("digits", 3, lam=1e-6)
    np.testing.assert_allclose(m.compile(vals), X, atol=1e-4)
    # position 0 cannot see x1: its weights on the x1 block stay exactly zero
    assert not m.weights[0, 4:7].any()
    acc = ResidualAccumulator(4, X.shape[2])
    acc.add(X, m.compile(vals))
    assert all(r["r2"] > 0.999999 for r in acc.summary(fam.positions))


def test_ridge_matches_closed_form_with_bias_regularized():
    rng = np.random.default_rng(1)
    fam = family()
    vals = grid()[rng.integers(0, 9, 40)]
    X = rng.normal(size=(40, 4, 3))
    fit = CompilerFit(fam, [0], 3)
    fit.add(vals, {0: X})
    lam = 0.7
    m = fit.solve("digits+s", 0, lam)
    keep = visibility(fam)[2] & columns(fam, "digits+s")
    Phi = design(fam, vals)[:, keep]
    W = np.linalg.solve(Phi.T @ Phi + lam * np.eye(keep.sum()), Phi.T @ X[:, 2])
    np.testing.assert_allclose(m.weights[2][keep], W, rtol=1e-5, atol=1e-6)
    assert not m.weights[2][~keep].any()


def test_switched_map_captures_gate_interaction_from_its_position_on():
    rng = np.random.default_rng(2)
    fam = family()
    vals = np.repeat(grid(), 10, 0)
    X = linear_states(fam, vals, rng, D=4)
    g = vals[:, 3].astype(bool)
    X[g, 1:] *= -2.0  # gate flips the linear map from position 1 on
    fit = CompilerFit(fam, [5], 4)
    fit.add(vals, {5: X})
    pooled, switched = fit.solve("digits", 5, 1e-6), fit.solve("digits|g", 5, 1e-6)
    assert switched.weights.shape == (2, 4, fam.width, 4)
    np.testing.assert_array_equal(switched.weights[0, 0], pooled.weights[0])
    np.testing.assert_allclose(switched.compile(vals)[:, 1:], X[:, 1:], atol=1e-4)
    r_pool, r_sw = ResidualAccumulator(4, 4), ResidualAccumulator(4, 4)
    r_pool.add(X, pooled.compile(vals))
    r_sw.add(X, switched.compile(vals))
    assert r_sw.summary(fam.positions)[3]["r2"] > 0.9999 > r_pool.summary(fam.positions)[3]["r2"]
    assert switched.switch_rows == [int((~g).sum()), int(g.sum())]


def test_switch_moments_must_be_accumulated_and_lambda_positive():
    fam = family()
    fit = CompilerFit(fam, [1], 2, feature_sets=["digits"])
    fit.add(grid(), {1: np.zeros((9, 4, 2))})
    with pytest.raises(CompiledStateError):
        fit.solve("digits|g", 1)
    with pytest.raises(CompiledStateError):
        fit.solve("digits", 1, lam=0.0)
    with pytest.raises(CompiledStateError):
        fit.solve("digits", 2)


def test_residual_statistics_match_direct_formulas():
    rng = np.random.default_rng(3)
    nat, comp = rng.normal(size=(30, 4, 5)), rng.normal(size=(30, 4, 5))
    acc = ResidualAccumulator(4, 5)
    acc.add(nat[:11], comp[:11])
    acc.add(nat[11:], comp[11:])
    out = acc.summary(["a", "b", "c", "d"])
    for p in range(4):
        sse = ((nat[:, p] - comp[:, p]) ** 2).sum()
        sst = ((nat[:, p] - nat[:, p].mean(0)) ** 2).sum()
        assert out[p]["r2"] == pytest.approx(1 - sse / sst)
        assert out[p]["native_norm"] == pytest.approx(np.linalg.norm(nat[:, p], axis=1).mean())
        assert out[p]["residual_norm"] == pytest.approx(np.linalg.norm(nat[:, p] - comp[:, p], axis=1).mean())
    with pytest.raises(CompiledStateError):
        ResidualAccumulator(1, 1).summary(["a"])


def test_agreement_error_recall_and_precision():
    native = np.array([[1, 2], [1, 2], [3, 4], [5, 6]])
    compiled = np.array([[1, 2], [1, 9], [3, 4], [5, 7]])
    truth = np.array([[1, 2], [1, 2], [3, 5], [5, 7]])  # native wrong on rows 2, 3
    a = agreement(native, compiled, truth)
    assert a["row_agreement"] == 0.5 and a["position_agreement"] == [1.0, 0.5]
    assert a["n_native_wrong"] == 2 and a["error_recall"] == 0.5  # row 2 reproduced, row 3 not
    assert a["n_compiled_wrong"] == 2 and a["error_precision"] == 0.5  # rows 1, 2; row 2 equals native
    assert a["native_accuracy"] == 0.5 and a["compiled_accuracy"] == 0.5
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and wilson(0, 0) is None
    assert "error_recall" not in agreement(native, compiled)


def test_horizon_on_the_part_e_ladders():
    f1 = {4: 0.990, 8: 0.986, 12: 0.985, 16: 0.975, 20: 0.976, 21: 0.972, 22: 0.971, 23: 0.790, 24: 0.029,
          26: 0.010, 28: 0.009}
    f4 = {4: 0.990, 20: 0.984, 22: 0.981, 23: 0.982, 24: 0.981, 26: 0.692, 28: 0.403}
    assert horizon(f1, 0.95) == {"threshold": 0.95, "horizon": 22, "first_break": 23, "deepest_pass": 22,
                                 "monotone": True}
    assert horizon(f4, 0.95)["horizon"] == 24 and horizon(f4, 0.95)["first_break"] == 26
    bumpy = horizon({4: 0.99, 8: 0.5, 12: 0.99}, 0.95)
    assert bumpy["horizon"] == 4 and bumpy["deepest_pass"] == 12 and not bumpy["monotone"]
    assert horizon({4: 0.1, 8: 0.99}, 0.95)["horizon"] is None
    assert horizon({4: 0.99}, 0.95)["first_break"] is None


def test_maps_save_load_verify_and_detect_tampering(tmp_path):
    rng = np.random.default_rng(4)
    fam = family()
    vals = np.repeat(grid(), 3, 0)
    fit = CompilerFit(fam, [2], 3)
    fit.add(vals, {2: rng.normal(size=(27, 4, 3))})
    m = fit.solve("digits|g", 2)
    manifest = m.save(tmp_path)
    back = CompiledStateMap.load(tmp_path, "digits|g", 2)
    np.testing.assert_array_equal(back.compile(vals), m.compile(vals))
    assert [x["fingerprint"] for x in verify_maps(tmp_path)] == [manifest["fingerprint"]]
    npy = next(tmp_path.glob("*.npy"))
    w = np.load(npy)
    w[0, 0, 0, 0] += 1
    np.save(npy, w)
    with pytest.raises(CompiledStateError):
        verify_maps(tmp_path)


def synthetic_report(fam):
    positions = [{"position": n, "native_norm": 10.0, "residual_norm": 1.0, "relative_residual": 0.1,
                  "r2": 0.9} for n in fam.positions]
    layer = lambda agree: {"agreement": {"n": 8, "row_agreement": agree, "error_recall": 0.5},  # noqa: E731
                           "positions": positions}
    return seal_report({"family": {**fam.to_dict(), "width": fam.width}, "model": {"name": "toy"},
                        "lam": 1.0, "threshold": 0.95, "layers": [1, 2],
                        "rows": {"fit": 9, "eval": 8, "r2": 8},
                        "noise_ceiling": {"n": 8, "row_agreement": 1.0, "error_recall": 0.75},
                        "native_splice_control": {"rows": 8, "max_abs_dlogit": {"1": 0.0, "2": 0.0}},
                        "feature_sets": {"digits": {"horizon": horizon({1: 0.99, 2: 0.5}, 0.95),
                                                    "layers": {"1": layer(0.99), "2": layer(0.5)}, "maps": {}}}})


def test_report_seal_render_and_cli(tmp_path, capsys):
    fam = family()
    report = synthetic_report(fam)
    assert verify_report(report) == report
    with pytest.raises(CompiledStateError):
        verify_report({**report, "lam": 2.0})
    text = render(report)
    assert "[digits] horizon L1  first break L2" in text and "R2@op" in text
    assert "error recall 0.750" in text and "max |dlogit| 0 over 8 rows" in text
    assert "0.100" in render(report, position="y", layer=2)
    shuffled = {**report["feature_sets"]["digits"], "layers": {"2": report["feature_sets"]["digits"]["layers"]["2"],
                                                               "1": report["feature_sets"]["digits"]["layers"]["1"]}}
    rows = render({**report, "feature_sets": {"digits": shuffled}}).splitlines()
    assert [r.split()[0] for r in rows[-2:]] == ["1", "2"]  # numeric layer order regardless of key order
    with pytest.raises(CompiledStateError):
        render(report, position="nope")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert cli_main(["compiled-state", str(path), "--position", "op"]) == 0
    assert "|resid|" in capsys.readouterr().out
    path.write_text(json.dumps({**report, "lam": 3.0}))
    assert main([str(path)]) == 2

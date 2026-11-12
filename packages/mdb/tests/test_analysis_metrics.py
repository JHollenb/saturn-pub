"""Position-robust relative error, mean-write baselines, rounding-bound margins, correlations."""
import numpy as np
import pytest

from mdb.analysis_metrics import (
    Bounded,
    combine,
    correlation_report,
    fit_mean_writes,
    flag_effects,
    float_spacing,
    infer_step,
    margin,
    mean_write_baselines,
    mean_write_forecast,
    relative_error,
    rounding_bound,
    spearman,
)
from mdb.native_discovery import error_metrics


def sink_states(rng, docs=4, spans=2, positions=6, width=5, sink=200.0):
    """Residual-like states whose position 0 carries a massive, nearly fixed activation."""
    out = []
    for _ in range(docs):
        s = rng.normal(size=(spans + 1, positions, width))
        for b in range(1, spans + 1):
            s[b, 0] = s[b - 1, 0] + sink + rng.normal(scale=0.1, size=width)
            s[b, 1:] = s[b - 1, 1:] + rng.normal(size=(positions - 1, width))
        out.append(s)
    return out


def test_relative_error_keeps_historical_fields_and_exposes_position_zero():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(6, 5))
    target[0] += 300.0
    prediction = target.copy()
    prediction[0] += 0.5  # tiny miss at the sink
    prediction[1:] = 0.0  # nothing right elsewhere
    m = relative_error(prediction, target)
    legacy = error_metrics(prediction, target)
    for key in ("mse", "relative_mse", "max_abs", "scalars"):
        assert legacy[key] == pytest.approx(m[key], rel=1e-12)
    assert m["relative_mse"] < 0.01  # uncentred number looks excellent
    assert m["position_0_energy_fraction"] > 0.99
    assert m["relative_mse_excluding_position_0"] == pytest.approx(1.0)  # predicting zero elsewhere
    assert m["centered_relative_mse_excluding_position_0"] > 0.9
    for key in ("centered_relative_mse", "relative_mse_excluding_position_0",
                "centered_relative_mse_excluding_position_0", "position_0_energy_fraction"):
        assert legacy[key] == pytest.approx(m[key])
    one = relative_error(target[:1], target[:1])
    assert one["relative_mse_excluding_position_0"] is None
    with pytest.raises(ValueError):
        relative_error(target, target[:, :4])


def test_centered_relative_error_is_one_for_the_documents_own_constant():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(7, 3)) + 50.0
    m = relative_error(np.broadcast_to(target.mean(0), target.shape), target)
    assert m["centered_relative_mse"] == pytest.approx(1.0)
    assert m["relative_mse"] < 0.01


def test_position_split_mean_write_beats_global_mean_on_a_sink():
    rng = np.random.default_rng(2)
    discovery = sink_states(rng)
    held = sink_states(rng, docs=1)[0]
    fitted = fit_mean_writes(discovery)
    assert fitted["global"].shape == (2, 5) and fitted["documents"] == 4
    expected_rest = np.concatenate([d[1, 1:] - d[0, 1:] for d in discovery]).mean(0)
    np.testing.assert_allclose(fitted["rest"][0], expected_rest)
    split = mean_write_forecast(held[0], fitted, split=True)
    glob = mean_write_forecast(held[0], fitted)
    np.testing.assert_allclose(glob, held[0][None] + fitted["global"].cumsum(0)[:, None])
    np.testing.assert_allclose(split[:, 0], held[0, 0] + fitted["position_0"].cumsum(0))
    rows = mean_write_baselines(held[0], held, fitted)
    for g, s in zip(rows["mean_write"], rows["position_split_mean_write"], strict=True):
        assert s["relative_mse"] < 0.05 * g["relative_mse"]
        assert g["position_0_energy_fraction"] > 0.9


def test_spacing_and_rounding_bounds_follow_the_format():
    assert float_spacing([16.625, 21.375, 31.9], "bfloat16").tolist() == [0.125] * 3
    assert float_spacing([15.875], "torch.bfloat16")[0] == 0.0625
    assert float_spacing([1.0], "float16")[0] == 2 ** -10
    assert float_spacing([1.0], "float32")[0] == np.spacing(np.float32(1.0))
    assert rounding_bound([16.5], dtype="bfloat16")[0] == 0.0625
    assert rounding_bound([3.0, 4.0], step=0.25).tolist() == [0.125, 0.125]
    assert infer_step([16.625, 21.375, 17.0]) == 0.125
    with pytest.raises(ValueError):
        rounding_bound([1.0])
    with pytest.raises(ValueError):
        float_spacing([1.0], "int8")


def test_bounds_propagate_through_differences_and_interactions():
    # Four BF16 logit pairs for native, cell A, cell B, pair: margins and a two-cell interaction.
    logits = {"native": (16.625, 21.375), "a": (19.75, 20.625), "b": (17.0, 21.25), "ab": (20.5, 20.0)}
    m = {k: margin(np.array(v), 0, 1, dtype="bfloat16", source=k) for k, v in logits.items()}
    assert m["native"].value == -4.75 and m["native"].bound == 0.125
    single = m["a"] - m["native"]
    assert single.bound == 0.25 and single.resolved
    interaction = m["ab"] - m["a"] - m["b"] + m["native"]
    assert interaction.bound == 0.5
    assert interaction.value == pytest.approx(0.5 - (-0.875) - (-4.25) + (-4.75))
    small = Bounded(0.375, 0.0) + Bounded(0.0, 0.5)
    assert not small.resolved
    rows = flag_effects({"interaction": interaction, "tiny": small})
    assert rows[1]["status"] == "inside rounding bound"
    # A quantity reused with opposite signs cancels exactly instead of double-counting its bound.
    nonadditivity = (m["ab"] - m["native"]) - combine([(1, m["a"] - m["native"]), (1, m["b"] - m["native"])])
    assert nonadditivity.bound == pytest.approx(0.125 * (1 + 1 + 1 + 1))
    assert (m["a"] - m["a"]).bound == 0.0
    assert (2 * m["a"]).bound == 0.25 and (-m["a"]).bound == 0.125
    with pytest.raises(TypeError):
        m["a"] * m["b"]


def test_correlation_report_returns_pearson_spearman_and_leave_one_out():
    x = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.9])
    y = np.array([0.0, 0.1, -0.1, 0.05, 0.0, 4.0])  # one leverage point drives Pearson
    r = correlation_report(x, y, labels=list("abcdef"))
    assert r["n"] == 6
    assert r["pearson"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    rx, ry = np.argsort(np.argsort(x)) + 1, np.array([2.5, 5, 1, 4, 2.5, 6])
    assert r["spearman"] == pytest.approx(np.corrcoef(rx, ry)[0, 1])
    loo = r["leave_one_out"]
    assert loo["pearson"]["min_dropped"] == "f"
    assert loo["pearson"]["min"] < r["pearson"] - 0.3
    drop_f = [row for row in loo["rows"] if row["dropped"] == "f"][0]
    assert drop_f["pearson"] == pytest.approx(np.corrcoef(x[:5], y[:5])[0, 1])
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert correlation_report([1, 1, 1], [1, 2, 3])["pearson"] is None
    with pytest.raises(ValueError):
        correlation_report([1, 2], [1, 2])

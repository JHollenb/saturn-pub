import pytest

from mdb.timing import PROTOCOL, TimingError, time_arms, time_fn


def test_time_fn_reports_registered_statistics():
    calls = []
    r = time_fn(lambda: calls.append(1), reps=7, warmup=2)
    assert len(calls) == 9 and r["reps"] == 7 and r["warmup"] == 2
    assert 0 <= r["p10_ms"] <= r["median_ms"] <= r["p90_ms"]
    with pytest.raises(TimingError):
        time_fn(lambda: None, reps=0)
    with pytest.raises(TimingError):
        time_fn(lambda: None, warmup=-1)


def test_time_arms_reports_speedups_only_behind_a_passing_gate():
    arms = {"native": lambda: sum(range(20000)), "predictor": lambda: None}
    ok = time_arms(arms, baseline="native", gate={"passed": True, "exact": True}, reps=5, warmup=1, batch=256)
    assert ok["claim"] == "speedups reported" and ok["batch"] == 256 and ok["protocol"]["name"] == PROTOCOL["name"]
    assert ok["arms"]["native"]["speedup_vs_baseline"] == pytest.approx(1.0)
    assert ok["arms"]["predictor"]["speedup_vs_baseline"] > 1.0
    for gate in (None, {"passed": False}):
        refused = time_arms(arms, baseline="native", gate=gate, reps=3, warmup=0)
        assert refused["claim"].startswith("refused")
        assert all(a["speedup_vs_baseline"] is None for a in refused["arms"].values())
        assert refused["arms"]["predictor"]["median_ms"] >= 0
    with pytest.raises(TimingError, match="baseline"):
        time_arms(arms, baseline="missing", gate={"passed": True})

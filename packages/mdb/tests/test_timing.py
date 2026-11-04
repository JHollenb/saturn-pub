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


def test_time_arms_interleaved_reports_block_noise_pairs_and_status():
    from mdb.timing import time_arms_interleaved

    arms = {"native": lambda: sum(range(20000)), "a": lambda: sum(range(5000)), "b": lambda: None}
    r = time_arms_interleaved(arms, baseline="native", gate={"passed": True}, blocks=3, reps=4, warmup=1,
                              pairs=[("a", "b")], status={"a": "sealed", "b": "amended"})
    assert len(r["arms"]["a"]["block_median_ms"]) == 3 and r["arms"]["a"]["status"] == "sealed"
    assert r["arms"]["native"]["time_ratio_mean"] == pytest.approx(1.0)
    assert r["pairs"]["a-b"]["mean_ms"] == pytest.approx(sum(r["pairs"]["a-b"]["diff_ms_by_block"]) / 3)
    assert r["pooled_between_block_sd_ms"] >= 0 and r["pairs"]["a-b"]["status"] == ["sealed", "amended"]
    refused = time_arms_interleaved(arms, baseline="native", gate=None, blocks=2, reps=2, warmup=0)
    assert refused["claim"].startswith("refused") and refused["arms"]["a"]["time_ratio_by_block"] is None
    with pytest.raises(TimingError):
        time_arms_interleaved(arms, baseline="native", gate=None, blocks=1)


def test_op_breakdown_records_parts_and_byte_split():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("op_breakdown graphs need CUDA")
    from mdb.timing import op_breakdown

    x = torch.ones(1 << 20, device="cuda")
    ops = {"a": lambda: x.mul_(1.0), "b": lambda: x.add_(0.0), "total": lambda: (x.mul_(1.0), x.add_(0.0))}
    rec = op_breakdown(ops, reps=5, total="total", bytes_by_op={"a": 8e6, "b": 8e6}, bandwidth_gbs=500.0)
    assert rec["l2"] == "hot" and set(rec["us"]) == {"a", "b", "total"}
    assert rec["byte_us"]["a"] == pytest.approx(16.0) and "total_not_bytes_us" in rec

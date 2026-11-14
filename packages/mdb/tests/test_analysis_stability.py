"""Verdict stability (SC6): the floor stays the noise bound; a verdict within k floors is marked
kernel-sensitive in ``supplied.judge`` and ``forecast.verify``; ``calibrate`` measures k per model
from paired effects of two realisations and reproduces the exp6 audit table."""

import json
import math
from pathlib import Path

import pytest

from mdb.analysis import stability

EXP6 = Path(__file__).resolve().parents[1] / "results" / "showcase-20261114" / "exp6" / "out"


def test_calibrate_returns_the_smallest_margin_above_which_no_verdict_is_lost(tmp_path):
    floor = 0.25
    pairs = [
        (0.30, 0.20, floor),    # 1.2 floors, lost (falls to the floor)
        (0.45, -0.40, floor),   # 1.8 floors, lost by a sign reversal
        (0.80, 0.70, floor),    # 3.2 floors, kept
        (2.00, 1.90, floor),    # 8 floors, kept
        (0.10, 0.40, floor),    # below the floor, gained on the other path
    ]
    cal = stability.calibrate(pairs, model="tiny", paths=("mdb-stepped", "batched"))
    assert cal.k == pytest.approx(1.8)
    assert (cal.n, cal.above_floor, cal.lost, cal.sign_reversed, cal.gained) == (5, 4, 2, 1, 1)
    table = {row["bin"]: row for row in cal.table}
    assert table["1-1.5"]["lost"] == 1 and table["1.5-2"]["sign_reversed"] == 1
    assert table[">4"] == {"bin": ">4", "n": 1, "lost": 0, "sign_reversed": 0}
    path = cal.save(tmp_path / "tiny.json")
    assert stability.Calibration.load(path) == cal
    none_lost = stability.calibrate([(1.0, 1.0, floor)], model="tiny")
    assert none_lost.k == 1.0 and none_lost.lost == 0


def test_stability_marks_within_k_floors_kernel_sensitive_and_names_the_path():
    out = stability.stability(0.9, 0.25)
    assert out["stability"] == stability.SENSITIVE and out["kernel_path"] == "undeclared"
    assert out["k"]["value"] == 4.0 and out["k"]["authority"] == "supplied"
    assert out["floors"]["value"] == pytest.approx(3.6)
    assert stability.stability(1.1, 0.25, kernel_path="batched")["stability"] == stability.STABLE
    cal = stability.Calibration(model="m", k=2.0, paths=["mdb-stepped", "batched"], n=1,
                                above_floor=1, lost=1, sign_reversed=0)
    calibrated = stability.stability(0.6, 0.25, calibration=cal.to_json())
    assert calibrated["stability"] == stability.STABLE
    assert calibrated["k"]["authority"] == "measured" and "calibrated for m" in \
        calibrated["k_source"]
    assert stability.stability(0.6, 0.25, k=3)["stability"] == stability.SENSITIVE


def test_judge_keeps_its_verdict_and_adds_stability_and_kernel_path():
    torch = pytest.importorskip("torch")
    from mdb.analysis import supplied

    record = {"id": "r", "body": {"supplied": {"value": -0.75}}}
    # magnitude 10 in bf16: spacing 0.0625, floor 0.125
    near = supplied.judge(record, -0.3, magnitude=10.0, dtype=torch.bfloat16,
                          kernel_path="mdb-stepped")
    assert near["native_effect"] and near["sign_agrees"]
    assert near["stability"] == "kernel-sensitive" and near["kernel_path"] == "mdb-stepped"
    far = supplied.judge(record, -1.0, magnitude=10.0, dtype=torch.bfloat16)
    assert far["stability"] == "stable" and far["kernel_path"] == "undeclared"
    assert supplied.judge(record, -0.3, magnitude=10.0, dtype=torch.bfloat16,
                          stable_floors=2.0)["stability"] == "stable"


def test_verify_marks_a_margin_forecast_with_its_stability_and_kernel_path():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mdb.analysis import forecast
    from mdb.ops.debugger import open_tiny

    run = open_tiny("qwen3", "bf16").run([1, 5, 9, 3])
    run.until("t3.L1.h")
    margin = forecast.seal(run, forecast.Margin(3, 5, 9))
    vector = forecast.seal(run, "t3.L1.o")
    run.until("t3.logits")
    results = {r["id"]: r for r in forecast.verify(run, kernel_path="mdb-stepped")}
    m = results[margin["id"]]
    assert m["verdict"] == "confirmed" and m["kernel_path"] == "mdb-stepped"
    s = m["stability"]
    native = abs(m["native"]["value"])
    assert s["stability"] == ("stable" if native > 4 * s["floor"]["value"] else "kernel-sensitive")
    assert s["floors"]["value"] == pytest.approx(native / s["floor"]["value"])
    v = results[vector["id"]]
    assert v["kernel_path"] == "mdb-stepped" and "stability" not in v


@pytest.mark.skipif(not EXP6.is_dir(), reason="showcase exp6 outputs not in this checkout")
def test_calibration_on_exp6_reproduces_the_audit_loss_table():
    """AUDIT-SHOWCASE.md correction 1: kernel path (mdb stepped vs TL batched unprocessed) loses
    60/274 (Qwen3-0.6B-Base) and 16/228 (Qwen2.5-0.5B); nothing above 4 floors is lost."""

    def spacing(x):
        x = abs(float(x))
        return 2.0 ** (math.floor(math.log2(x)) - 7)

    want = {"Qwen3-0.6B-Base": (60, 274, [51, 8, 1, 0]),
            "Qwen2.5-0.5B": (16, 228, [14, 2, 0, 0])}
    for model, (n_lost, n_above, by_bin) in want.items():
        rows = [json.loads(line) for line in open(EXP6 / f"exp6-{model}.jsonl")]
        pairs = [(r["mdb"][k] - r["mdb"][0], r["U16"][k] - r["U16"][0],
                  2 * spacing(r["max_abs_logit"])) for r in rows for k in range(1, len(r["mdb"]))]
        cal = stability.calibrate(pairs, model=model, paths=("mdb-stepped", "tl-batched"))
        assert (cal.lost, cal.above_floor) == (n_lost, n_above)
        assert [row["lost"] for row in cal.table] == by_bin
        assert cal.k <= stability.STABLE_FLOORS

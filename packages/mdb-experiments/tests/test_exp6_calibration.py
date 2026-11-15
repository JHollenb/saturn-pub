"""Experiment-artifact test moved out of packages/mdb/tests/test_analysis_stability.py: the
stability calibration reproduces the showcase exp6 audit loss table from the saved outputs in
results/showcase-20261114/exp6/out. Run from packages/mdb with PYTHONPATH=src:../../src."""

import json
import math
from pathlib import Path

import pytest

from mdb.analysis import stability

EXP6 = Path(__file__).resolve().parents[1] / "results" / "showcase-20261114" / "exp6" / "out"


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

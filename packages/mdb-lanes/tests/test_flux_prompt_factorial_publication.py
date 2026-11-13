"""The larger counterexample must remain visible in the exported research plot."""
import runpy
from pathlib import Path

import numpy as np

P = runpy.run_path(str(Path(__file__).resolve().parents[1] / "qualification/publish_flux_prompt_factorial.py"))


def test_commit_plot_contains_every_series_after_first_case(tmp_path, monkeypatch):
    audit = {"numeric": {}}
    for case, maximum in zip(P["CELLS"], (.02, .01, .09, .04), strict=True):
        transitions = [None] * 112
        for step in range(4):
            transitions[step * 28 + 27] = {"logical_step": step, "local_index": 27, "kind": "commit",
                "fields": {"latent": {"full_native_array_retained": True, "native_cut": "fixture-cut",
                    "predicted_minus_native_full": {"rms": maximum * (step + 1) / 4,
                        "relative_rms": maximum * (step + 1) / 4}}}}
        audit["numeric"][case] = {"transitions": transitions}
    figures = []
    original_close = P["plt"].close
    monkeypatch.setattr(P["plt"], "close", lambda figure: figures.append(figure))
    P["plot_commit_curves"](audit, tmp_path / "curves.png")
    figure = figures[-1]
    try:
        for axis in figure.axes:
            assert len(axis.lines) == 4
            low, high = axis.get_ylim()
            for line in axis.lines:
                values = np.asarray(line.get_ydata())
                assert low <= values.min() <= values.max() <= high
    finally:
        original_close(figure)

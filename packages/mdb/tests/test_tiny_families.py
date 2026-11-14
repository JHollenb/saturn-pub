"""The tiny families are named once (``mdb.engine.models.TINY_FAMILIES``); the command line's
help repeats them as a literal (it loads no torch), and a tiny mistral context longer than its
sliding window is refused by the adapter, which is what the forecast's attention premise
("every cached source is visible") rests on."""
import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb import cli  # noqa: E402
from mdb.analysis import forecast  # noqa: E402
from mdb.engine.models import TINY_FAMILIES  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402


def test_the_cli_lists_exactly_the_engine_tiny_families():
    assert cli.TINY_FAMILIES == TINY_FAMILIES
    assert f"--tiny {'|'.join(TINY_FAMILIES)}" in cli.USAGE
    assert f"tiny:{'|'.join(TINY_FAMILIES)}" in cli.USAGE
    assert f"--tiny {'|'.join(TINY_FAMILIES)}" in cli.__doc__.replace("\n              ", " ")


def test_a_mistral_context_past_the_window_is_refused_and_the_window_holds_the_forecast_premise():
    debugger = open_tiny("mistral", "fp32")
    window = debugger.adapter.sliding_window
    assert window == 128 and debugger.adapter.max_positions > window
    with pytest.raises(ValueError, match="context exceeds the sliding-window horizon"):
        debugger.run([1] * (window + 1))
    run = debugger.run([1 + i % 7 for i in range(window)])  # exactly the window: accepted
    run.until("L0.kr")
    view = forecast.seal(run)  # the premise check passes because no source is out of window
    assert view["target"].endswith("L0.s"), view["target"]
    assert "every cached source is visible" in json.dumps(view), view

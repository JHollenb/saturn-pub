"""``print``/``x``/``display`` read a declared margin M[a,b] as the measured difference of the
row's two logits at the cursor (M45); ``watch`` refuses it with what to use instead."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import dispatch  # noqa: E402
from mdb.ops.errors import Refused  # noqa: E402
from mdb.repl.render import render  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    c = mdb.attach(tiny_model("qwen2", "bf16", layers=2)).ops()
    dispatch(c, "start", {"prompt": [1, 5, 9, 3]})
    return c


def test_margin_print_is_the_measured_logit_difference(ctx):
    with pytest.raises(Refused, match="not computed at .*until t3.logits"):
        dispatch(ctx, "print", {"address": "M[4,9]"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    logits = ctx.run.read("t3.logits").reshape(-1)
    want = float(logits[4].double() - logits[9].double())
    result = dispatch(ctx, "print", {"address": "M[4,9]"})
    value = result["margin"]["value"]
    assert value == {"value": want, "authority": "measured"} and result["state"] == "resolved"
    assert result["address"] == "t3.M[4,9]"
    assert "logits[4] - logits[9]" in render(result) and "measured" in render(result)
    shown = dispatch(ctx, "x", {"address": "t3.M[4,9]"})
    assert shown["values"] == {"values": [want], "authority": "measured"}
    dispatch(ctx, "display", {"address": "M[4,9]"})
    assert dispatch(ctx, "display", {})["displays"][0]["margin"]["value"]["value"] == want


def test_watch_of_a_margin_names_what_to_use(ctx):
    with pytest.raises(Refused, match="derived margin"):
        dispatch(ctx, "watch", {"address": "M[4,9]"})
    with pytest.raises(Refused, match="derived value"):
        dispatch(ctx, "set", {"target": "M[\"a\",\"b\"]", "value": 1.0})

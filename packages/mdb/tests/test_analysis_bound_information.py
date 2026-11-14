"""The worst-case rounding interval is information (M56/M57), and a what-if over several rows is
stage-checked (M55).

On Qwen2.5-1.5B the chain interval assumes every rounding error aligns: the L24 score interval is
12.9, softmax multiplies by exp(2 x 12.9) and the bound reached inf/nan, which made an exact
forecast read as unresolved. Here a tiny Qwen2 with its layer-0 queries and keys scaled by 500
has a score interval above 355 (exp overflows float64) and score gaps of thousands (the native
softmax underflows to 0 where the exact one is about 1e-54).
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.accumulation import PREMISE_FAILS  # noqa: E402
from mdb.analysis.forecast import (  # noqa: E402
    NOT_INFORMATIVE,
    EmulatedRowModel,
    Margin,
    RowModel,
    _known,
    stage_checks,
)
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.analysis.verdict import NOT_CONFIRMED  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach  # noqa: E402
from mdb.ops.registry import dispatch, has_unresolved  # noqa: E402
from mdb.repl.render import render  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2, 11, 4]
ROW = len(PROMPT) - 1


def _scaled(factor: float = 500.0):
    """Tiny bf16 Qwen2 whose layer-0 query and key maps (and biases) are scaled by ``factor``:
    the scores scale by ``factor**2`` and so does the worst-case score interval."""
    model = tiny_model("qwen2", "bf16", layers=2)
    attn = model.model.layers[0].self_attn
    with torch.no_grad():
        for linear in (attn.q_proj, attn.k_proj):
            linear.weight.mul_(factor)
            linear.bias.mul_(factor)
    return attach(model, None)


@pytest.fixture(scope="module")
def scaled():
    return _scaled()


def _row_model(debugger):
    run = debugger.run([1, 5, 9, 3])
    row = run.parse(run.where)[0]
    return RowModel(run, row, _known(run, row, run.parse(run.where)))


def test_the_softmax_bound_contains_a_native_underflow_to_zero():
    """Scores 1.5e4 apart: float32 exp underflows, the native p is 0 where the exact one is
    about 1e-6500; the bound's absolute underflow term contains it (it had none: M56, the one
    stage of 533 that failed on Qwen2.5-1.5B L0.p)."""
    model = _row_model(attach(tiny_model("qwen2", "bf16")))
    s = torch.tensor([[0.0, -64.0, -128.0, -192.0, -15000.0]], dtype=torch.float64)
    p, bound = model._softmax(s, torch.zeros_like(s))
    native = torch.softmax(s.to(torch.bfloat16), -1, dtype=torch.float32).to(torch.bfloat16)
    native = native.double()
    assert float(native[0, 2]) == 0.0 and float(p[0, 2]) > 0.0     # e^-128 underflows
    assert bool(((native - p).abs() <= bound).all()), ((native - p).abs() - bound).max()
    assert bool(torch.isfinite(bound).all())


def test_a_non_positive_softmax_denominator_is_a_stated_premise_failure():
    """A score shift of 2e7 makes 1 - worst - gamma negative: the bound is refused with the
    premise named (it was a negative or nan bound)."""
    model = _row_model(attach(tiny_model("qwen2", "bf16")))
    s = torch.tensor([[0.0, -2.0e7]], dtype=torch.float64)
    p, bound = model._softmax(s, torch.zeros_like(s))
    assert bound is None
    assert model._failed.startswith(PREMISE_FAILS) and "1 - worst - gamma" in model._failed


def test_an_overflowing_score_interval_is_named_and_the_forecast_is_resolved(scaled):
    """Score interval above 355: exp(2 x interval) overflows float64. The forecast seals the
    exact and the emulated prediction, names the stop where the interval stops being
    informative and why, carries no inf or nan, and the result is resolved (it was unresolved);
    verify confirms it and reports each stage within its own bound on native operands."""
    ctx = scaled.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    result = dispatch(ctx, "forecast", {"target": "M[4,9]"})
    json.dumps(result, allow_nan=False)
    assert result["state"] == "resolved" and not has_unresolved(result)
    bound = result["forecast"]["bound"]
    assert bound["informative"] is False and bound["from"] == f"t{ROW}.L0.p", bound
    assert "score interval" in bound["cause"] and "beyond float64" in bound["cause"], bound
    assert bound["note"].startswith(f"{NOT_INFORMATIVE} t{ROW}.L0.p (score interval")
    text = render(result)
    assert "predicted exact" in text and "emulated" in text and bound["note"] in text
    dispatch(ctx, "until", {"target": f"t{ROW}.logits"})
    checked = dispatch(ctx, "verify", {})
    json.dumps(checked, allow_nan=False)
    item = checked["forecasts"][0]
    assert (item["verdict"], checked["state"]) == ("confirmed", "resolved"), item
    stages = item["stages"]
    assert stages["checked"]["value"] == stages["within"]["value"] > 30, stages
    assert stages["text"].startswith("each stage within its own rounding bound on native "
                                     f"operands: {stages['within']['value']}/")
    assert item["bound_note"] == bound["note"]
    shown = render(checked)
    assert "stages [per-stage bound, not a chain bound]: each stage within" in shown
    assert bound["note"] in shown


def test_the_per_stage_bound_holds_where_the_native_softmax_underflows(scaled):
    """L0.p on native operands: native p = 0 where the exact p is tiny, contained by the
    absolute term (the stage check itself: no chain bound involved)."""
    run = scaled.run(PROMPT)
    run.until(f"t{ROW}.logits")
    native = run.read(f"t{ROW}.L0.p").double()
    assert int((native == 0).sum()) > 0
    (check,) = stage_checks(run, [(ROW, "L0.p")])
    assert check["within"] is True and check["ratio"] <= 1, check


def test_a_whatif_prints_the_note_instead_of_a_huge_bound(scaled):
    """M56/M57: no `within bound 1e36` and no silent line: the sealed exact change and the
    rounding effect say where the worst-case interval stopped being informative; the result
    is resolved when native equals emulated; each run's stages are reported."""
    ctx = scaled.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "until", {"target": f"t{ROW}.logits"})
    result = dispatch(ctx, "whatif", {"target": "L0.q=0.5", "op": "add", "consumer": "M[4,9]"})
    json.dumps(result, allow_nan=False)
    exact = result["prediction"]["exact"]
    assert "bound" not in exact and exact["bound_note"].startswith(
        f"{NOT_INFORMATIVE} t{ROW}.L0.p (score interval"), exact
    terms = {t["name"]: t for t in result["split"]["terms"]}
    assert "bound" not in terms["rounding effect"]
    assert terms["rounding effect"]["bound_note"].startswith(NOT_INFORMATIVE)
    assert result["effect"]["residual_spacings"]["value"] == 0.0
    assert result["state"] == "resolved" and not has_unresolved(result)
    text = render(result)
    assert "within bound" not in text and exact["bound_note"] in text
    assert f"rounding effect {_value(terms['rounding effect'])}" in text
    assert terms["rounding effect"]["bound_note"] in text
    assert "edited run stages [per-stage bound, not a chain bound]: each stage within" in text
    # rewind mode: the unedited run stands past the consumer and is checked there too
    assert set(result["stages"]) == {"edited", "base"}, result["stages"].keys()
    assert "base run stages [per-stage bound, not a chain bound]: each stage within" in text


def _value(term):
    from mdb.repl.render import _claimed

    return _claimed(term["value"])


def test_has_unresolved_skips_information():
    assert not has_unresolved({"bound": {"state": "unresolved", "reason": "x",
                                         "information": True}})
    assert has_unresolved({"bound": {"state": "unresolved", "reason": "x"}})


def test_a_whatif_over_two_rows_is_stage_checked(monkeypatch):
    """M55: a wrong RMSNorm in both chains (x 1.5) parts the emulated change from the native
    one on an edit at t6 with the consumer at t7; the stages are checked across both rows and
    the native norm outside the bound of the (wrong) identity is named (it was `unresolved,
    stages not checked`)."""
    def wrong(original):
        def rms(self, *args):
            value, bound = original(self, *args)
            return value * 1.5, bound
        return rms

    debugger = attach(tiny_model("qwen2", "bf16", layers=2))
    monkeypatch.setattr(RowModel, "_rms", wrong(RowModel._rms))
    monkeypatch.setattr(EmulatedRowModel, "_rms", wrong(EmulatedRowModel._rms))
    run = debugger.run(PROMPT, prefill=ROW - 1)
    result = whatif(run, Edit(f"t{ROW - 1}.L0.h_out", 4.0, "add"), Margin(ROW, 4, 9))
    effect = result.report["effect"]
    assert effect["residual_spacings"]["value"] > 1, effect
    assert (effect["verdict"], effect["state"]) == (NOT_CONFIRMED, "resolved"), effect
    assert "outside the rounding bound of its identity" in effect["reason"], effect["reason"]
    assert "stages not checked" not in effect["reason"]
    stages = result.report["stages"]["edited"]
    assert {stop.split(".")[0] for stop in [stages["worst_stage"]]} <= {f"t{ROW - 1}", f"t{ROW}"}
    assert stages["walked"]["value"] > stages["checked"]["value"] - 1 > 20

"""verify's order proof is "the sealed cut is an ancestor of the current cut", read from the run's
cut trail: it holds on models deeper than the session's receipt window, survives a ``whatif``
between ``forecast`` and ``verify``, still reports an edit committed after sealing, and refuses a
different path (M47)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import forecast, propagate  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.engine.run import _RECEIPTS  # noqa: E402
from mdb.ops import dispatch  # noqa: E402

PROMPT = [1, 5, 9, 3]


def _ctx(layers, project=None):
    dbg = mdb.attach(tiny_model("qwen2", "bf16", layers=layers), project=project)
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    return dbg, ctx


def _verdict(ctx):
    return dispatch(ctx, "verify", {})["forecasts"][0]


def test_verify_confirms_past_the_receipt_window():
    _, ctx = _ctx(70)
    dispatch(ctx, "until", {"target": "t3.L0.h"})
    dispatch(ctx, "forecast", {"target": "M[4,9]"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    assert len(ctx.run.session.receipts) <= _RECEIPTS < 70 + 4 * 72   # the window did drop them
    result = _verdict(ctx)
    assert result["verdict"] == "confirmed", result
    assert result["order"]["descends"]


@pytest.mark.parametrize("layers", [4, 40])
def test_verify_survives_a_whatif_between_forecast_and_verify(layers):
    _, ctx = _ctx(layers)
    dispatch(ctx, "until", {"target": "t3.L0.h"})
    dispatch(ctx, "forecast", {"target": "M[4,9]"})
    dispatch(ctx, "until", {"target": "t3.L2.q"})
    before = propagate.state_digest(ctx.run)
    for target in ("L3.m=0.5", "L0.o=0.5", "L2.q=0.5"):
        dispatch(ctx, "whatif", {"target": target, "op": "add", "consumer": "M[4,9]"})
    assert propagate.state_digest(ctx.run) == before
    dispatch(ctx, "until", {"target": "t3.logits"})
    assert _verdict(ctx)["verdict"] == "confirmed"


def test_an_edit_after_sealing_is_still_reported_as_edited():
    _, ctx = _ctx(4)
    dispatch(ctx, "until", {"target": "t3.L0.h"})
    dispatch(ctx, "forecast", {"target": "M[4,9]"})
    dispatch(ctx, "until", {"target": "t3.L2.q"})
    dispatch(ctx, "set", {"target": "L2.q", "value": 0.5, "op": "add"})
    dispatch(ctx, "commit", {})
    dispatch(ctx, "until", {"target": "t3.logits"})
    result = _verdict(ctx)
    assert result["verdict"] == "edited" and result["order"]["descends"], result


def test_a_different_path_is_open_not_confirmed():
    _, ctx = _ctx(4)
    dispatch(ctx, "until", {"target": "t3.L2.h"})
    dispatch(ctx, "forecast", {"target": "M[4,9]"})
    dispatch(ctx, "reverse-finish", {})               # back across layers 2, 1, 0 to t3.L0.h
    dispatch(ctx, "reverse-finish", {})
    dispatch(ctx, "reverse-finish", {})
    dispatch(ctx, "until", {"target": "t3.L0.q"})
    dispatch(ctx, "set", {"target": "L0.q", "value": 0.5, "op": "add"})   # a different path
    dispatch(ctx, "commit", {})
    dispatch(ctx, "until", {"target": "t3.logits"})
    result = _verdict(ctx)
    assert result["verdict"] == "open" and not result["order"]["descends"], result


def test_whatif_passes_the_project_store(tmp_path):
    dbg, ctx = _ctx(70, project=tmp_path / "p")
    dispatch(ctx, "until", {"target": "t3.L0.h"})
    result = dispatch(ctx, "whatif", {"target": "L69.m=0.5", "op": "add", "consumer": "M[4,9]"})
    record = result["sealed_record"]
    summary = record["body"]["operands_summary"]
    assert summary["pointer"]["state"] == "stored", summary["pointer"]
    full = propagate.load_operands(record, store=dbg.project.store)    # no run involved
    assert len(full) == summary["count"] > 64


def test_the_live_ledger_counts_every_step_not_the_listed_ones():
    from mdb.serve.views.live import _forecast_row

    _, ctx = _ctx(70)
    dispatch(ctx, "until", {"target": "t3.L0.h"})
    dispatch(ctx, "forecast", {"target": "M[4,9]"})
    row = _forecast_row(next(iter(forecast.ledger(ctx.run).values())))
    assert row.steps > 64

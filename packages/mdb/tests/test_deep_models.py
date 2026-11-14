"""Every analysis op on a 40-layer and a 70-layer tiny model (M40): none refused for record size.

Lists that grow with the layers (forecast chain, explain terms, what-if operands) are bounded to
the record's sequence bound and stored by pointer (``mdb.core.bounded``). One process, each op
once per depth; every result passes the record bound, the what-if operand pointer resolves to
the full list (in memory on the run, and from a project store) and its sha256 is checked.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import propagate  # noqa: E402
from mdb.analysis.forecast import Margin, ledger  # noqa: E402
from mdb.core.bounded import MAX_ITEMS  # noqa: E402
from mdb.core.pointers import check_bounded  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import dispatch  # noqa: E402

PROMPT = [1, 5, 9, 3]
OTHER = [1, 7, 9, 3]


def _ok(result):
    check_bounded(result)
    assert result.get("state") != "refused", result
    return result


@pytest.mark.parametrize("layers", [40, 70])
def test_every_analysis_op_fits_the_record_bound_at_depth(layers, tmp_path):
    model = tiny_model("qwen2", "bf16", layers=layers)
    dbg = mdb.attach(model, project=tmp_path / "p")
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": OTHER, "name": "other"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    dispatch(ctx, "start", {"prompt": PROMPT})

    # forecast (a margin) at t3.L0.h, verified once the cursor has passed it
    dispatch(ctx, "until", {"target": "t3.L0.h"})
    forecast = _ok(dispatch(ctx, "forecast", {"target": "M[4,9]"}))
    assert forecast["record"]["body"]["method"]["summary"]["count"] > MAX_ITEMS
    # M41: the op result's chain list obeys the bound as well (it was sealed, then refused)
    assert len(forecast["forecast"]["method"]) == MAX_ITEMS
    assert forecast["forecast"]["method_listed"]["count"] > MAX_ITEMS
    for target in (f"L{layers - 1}.m", "logits"):
        sealed = _ok(dispatch(ctx, "forecast", {"target": target}))
        assert sealed["forecast"]["method_listed"]["listed"] == MAX_ITEMS
    assert len(ledger(ctx.run)) == 3

    dispatch(ctx, "until", {"target": "t3.logits"})
    verified = _ok(dispatch(ctx, "verify", {}))
    verdict = verified["forecasts"][0]["verdict"]
    # Past 64 layers the session's receipt window (engine.run._RECEIPTS) no longer holds the
    # whole path from the sealed cut, so the lineage test answers "open" (new issue M42).
    assert verdict in ("confirmed", "edited", "not confirmed") + (("open",) if layers > 64 else ())

    # what-if at depth: a port edit, a cell edit, a value of another run
    before = propagate.state_digest(ctx.run)
    for args in ({"target": f"L{layers - 1}.m=0.5", "op": "add", "consumer": "M[4,9]"},
                 {"target": "cache.L1.v[row=1]", "value": 0.5, "op": "mix",
                  "consumer": "M[4,9]", "dose": 0.5},
                 {"target": "L0.o", "value": "other:L0.o", "consumer": "M[4,9]"}):
        result = _ok(dispatch(ctx, "whatif", args))
        assert result["prediction"]["state"] in ("resolved", "refused"), result["prediction"]
    assert propagate.state_digest(ctx.run) == before
    _ok(dispatch(ctx, "explain", {"address": "M[4,9]"}))
    _ok(dispatch(ctx, "explain", {"address": f"L{layers - 1}.o"}))
    _ok(dispatch(ctx, "reads", {"source": "t1", "consumer": "M[4,9]", "top": 3}))
    _ok(dispatch(ctx, "symbols", {"consumer": "M[4,9]"}))
    _ok(dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 1}))
    _ok(dispatch(ctx, "compare", {"a": "L1.m=0.5 add", "b": "L0.v=0.25 add",
                                  "consumer": "M[4,9]"}))


@pytest.mark.parametrize("layers", [40, 70])
@pytest.mark.parametrize("with_project", [False, True], ids=["memory", "project"])
def test_the_whatif_operands_pointer_resolves_to_the_full_list(layers, with_project, tmp_path):
    dbg = mdb.attach(tiny_model("qwen2", "bf16", layers=layers),
                     project=tmp_path / "p" if with_project else None)
    run = dbg.run(PROMPT)
    run.until("t3.L0.h")
    store = dbg.project.store if with_project else None
    edit = propagate.Edit(f"t3.L{layers - 1}.m", 0.5, "add")
    result = propagate.whatif(run, edit, Margin(3, 4, 9), store=store)
    body = result.record["body"]
    summary = body["operands_summary"]
    assert len(body["operands"]) == summary["listed"] == MAX_ITEMS
    assert summary["count"] > MAX_ITEMS and summary["rest"]["count"] == summary["count"] - MAX_ITEMS
    assert summary["pointer"]["state"] == ("stored" if with_project else "absent")
    full = propagate.load_operands(result.record, store=store, run=run)
    assert len(full) == summary["count"] and full[:MAX_ITEMS] == body["operands"]
    assert result.report["effect"]["state"] in ("resolved", "unresolved")
    check_bounded(result.report)
    if with_project:       # the stored blob is checked against its sha256
        broken = dict(summary["pointer"], sha256="0" * 64)
        record = {**result.record, "body": {**body, "operands_summary": {**summary, "pointer": broken}}}
        with pytest.raises(ValueError, match="sha256"):
            propagate.load_operands(record, store=store)

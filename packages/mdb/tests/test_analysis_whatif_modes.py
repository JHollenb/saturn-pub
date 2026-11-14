"""The same what-if at different cursors (M46): one exact change, one verdict.

``exact = F(x1) - F(x0)`` (P09 section 1, P20) is defined on the retained operands. Sealed with
the cursor before the edit (mode ``ahead``), the operands past the cursor exist only as the
chain's own float64 forecast, and the sealed value says so (``operands.source == "forecast"``).
After the native run the exact chain is computed again on the measured operands of the unedited
run, so the post-hoc exact change, the split and the verdict are the same at every cursor.

Tolerance: both cursors evaluate the same float64 identities on the same native operands, so the
only admissible difference is float64 reassociation, far below one step of the model dtype; the
tests allow ``spacing / 2**20`` (``spacing``: the model dtype's gap at the consumer's magnitude,
``verdict.spacing``). The old code differed by up to 0.4 spacings on these models (bf16), and by
0.086 (0.7 spacings) on Qwen2.5-1.5B (job-176bc5bdcee7 vs job-59cba8de26fc).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.analysis.propagate import Edit, whatif  # noqa: E402
from mdb.analysis.verdict import spacing  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach  # noqa: E402

CLEAN = [1, 7, 9, 3, 11, 2]
MAIN = [1, 5, 9, 3, 11, 2]
ROW = 5
CURSORS = (f"t{ROW}.L0.h", f"t{ROW}.L3.h", f"t{ROW}.logits")   # ahead, ahead, rewind
CONSUMER = Margin(ROW, 6, 7)


@pytest.fixture(scope="module", params=["bf16", "fp32"])
def debugger(request):
    return attach(tiny_model("qwen2", request.param, layers=6))


def _clean(debugger, address):
    run = debugger.run(CLEAN)
    run.until(f"t{ROW}.logits")
    return run.read(address).clone()


def _edits(debugger):
    head = torch.zeros(1, 1, debugger.adapter.hidden_size, dtype=torch.bool)
    head[..., debugger.adapter.head_dim:2 * debugger.adapter.head_dim] = True
    a = f"t{ROW}.L4.A"
    return {
        "port add": Edit(f"t{ROW}.L4.m", 0.5, "add"),
        "cross-run value": Edit(f"t{ROW}.L4.m", _clean(debugger, f"t{ROW}.L4.m")),
        "head-selected": Edit(a, _clean(debugger, a), mask=head, selection="h1"),
    }


def _at(debugger, edit, cursor):
    run = debugger.run(MAIN)
    run.until(cursor)
    return whatif(run, edit, CONSUMER).report


@pytest.mark.parametrize("name", ["port add", "cross-run value", "head-selected"])
def test_post_hoc_exact_and_verdict_agree_at_every_cursor(debugger, name):
    edit = _edits(debugger)[name]
    reports = {cursor: _at(debugger, edit, cursor) for cursor in CURSORS}
    rewind = reports[CURSORS[-1]]
    assert rewind["mode"] == "rewind" and reports[CURSORS[0]]["mode"] == "ahead"
    size = max(abs(rewind["measured"]["base"]["value"]), abs(rewind["measured"]["edited"]["value"]))
    tol = spacing(size, debugger.adapter.dtype) / 2 ** 20
    for cursor, report in reports.items():
        assert report["measured"]["delta"] == rewind["measured"]["delta"], cursor
        effect, other = report["effect"], rewind["effect"]
        for key in ("exact", "residual"):
            assert abs(effect[key]["value"] - other[key]["value"]) <= tol, (cursor, key)
        assert (effect["verdict"], effect["state"]) == (other["verdict"], other["state"]), cursor
        assert effect["floor"] == other["floor"], cursor
        for key in ("requested", "written"):
            assert report["exact"][key]["authority"] == "derived-exact"
            assert abs(report["exact"][key]["value"] - rewind["exact"][key]["value"]) <= tol
        assert effect["exact"]["value"] == report["exact"]["written"]["value"]


@pytest.mark.parametrize("name", ["port add", "head-selected"])
def test_the_split_closes_and_names_the_forecast_operands(debugger, name):
    edit = _edits(debugger)[name]
    for cursor in CURSORS:
        report = _at(debugger, edit, cursor)
        terms = {t["name"]: t["value"]["value"] for t in report["split"]["terms"]}
        native = report["measured"]["delta"]["value"]
        assert abs(sum(terms.values()) - native) <= 1e-9 * max(1.0, abs(native)), (cursor, terms)
        operands = report["prediction"]["exact"]["operands"]
        if cursor == CURSORS[-1]:      # every operand measured at the seal: no forecast term
            assert operands == {"source": "measured", "at": cursor} and "operands" not in terms
            assert (report["prediction"]["exact"]["value"]["value"]
                    == report["exact"]["requested"]["value"])
        else:
            assert operands["source"] == "forecast" and operands["at"] == cursor
            assert f"after {cursor}" in operands["premise"]
            sealed = report["prediction"]["exact"]["value"]["value"]
            assert terms["operands"] == pytest.approx(
                report["exact"]["requested"]["value"] - sealed, abs=1e-15)
        if name == "head-selected":   # the other heads are the native ones: nothing to realise
            assert terms["realisation"] == 0.0, cursor


@pytest.mark.parametrize("name", ["port add", "cross-run value", "head-selected"])
def test_rewind_exact_is_not_taken_against_the_native_base(debugger, name):
    """The sealed exact change with the cursor past the consumer is F(x1) - F(x0) on the
    captured operands, not F(x1) minus the native base value (which folds the base run's
    rounding into the exact change): it equals the sealed exact change with the cursor between
    the edited layer and the consumer, where every stop the chain reads past the cursor is one
    the edit reaches (recomputed either way)."""
    edit = _edits(debugger)[name]
    past = _at(debugger, edit, f"t{ROW}.logits")
    between = _at(debugger, edit, f"t{ROW}.L5.h")
    assert past["mode"] == between["mode"] == "rewind"
    size = max(abs(past["measured"]["base"]["value"]), abs(past["measured"]["edited"]["value"]))
    tol = spacing(size, debugger.adapter.dtype) / 2 ** 20
    assert abs(past["prediction"]["exact"]["value"]["value"]
               - between["prediction"]["exact"]["value"]["value"]) <= tol


@pytest.mark.parametrize("cursor", CURSORS)
def test_a_no_op_edit_has_exact_change_zero_at_every_cursor(debugger, cursor):
    """``x * 1`` writes x bit for bit: F(x1) - F(x0) is 0 exactly, sealed and post hoc. The old
    rewind value was -(native base - F(x0)) (the base run's rounding, -0.00037 on tiny bf16) and
    the old ahead verdict value the forecast-vs-native difference of x0."""
    report = _at(debugger, Edit(f"t{ROW}.L4.m", 1.0, "scale"), cursor)
    assert report["measured"]["delta"]["value"] == 0.0
    assert report["prediction"]["exact"]["value"]["value"] == 0.0
    assert report["exact"]["requested"]["value"] == report["exact"]["written"]["value"] == 0.0
    assert report["effect"]["exact"]["value"] == 0.0

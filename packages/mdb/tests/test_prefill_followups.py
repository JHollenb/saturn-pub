"""Prefill follow-ups (M54): every verb takes the run's first stepped row from ``run.root``
(``prefill=K``), ``each`` takes ``prefill=``, ``start`` builds its session once, records carry
the split, the cross-row what-if bound says what it covers, ``sweep`` plans its cost, refuses
over a cap and is a record, and staging an edit makes no native call.

Tiny Qwen2 bf16, 2 layers, clean ``1 7 9 3 6 2`` and corrupt ``1 5 9 3 6 2`` (row 1 differs),
both ``prefill=0`` unless a test says otherwise. Each test fails on the code before M54.
"""

import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.core import index  # noqa: E402
from mdb.core.bounded import load_json  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402
from mdb.ops.lines import parse  # noqa: E402
from mdb.ops.predict import session_digest  # noqa: E402
from mdb.repl.render import heat_table, render  # noqa: E402
from mdb.repl.shell import Shell  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "prompt_set_tiny.jsonl"
CLEAN, CORRUPT = [1, 7, 9, 3, 6, 2], [1, 5, 9, 3, 6, 2]
CONSUMER = "t5.M[4,9]"


@pytest.fixture(scope="module")
def tiny():
    return mdb.attach(tiny_model("qwen2", "bf16"))


def _ctx(debugger, prefill=0, at="t5.logits"):
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": CLEAN, "name": "clean", "prefill": prefill})
    dispatch(ctx, "until", {"target": "t5.logits"})
    dispatch(ctx, "start", {"prompt": CORRUPT, "prefill": prefill})
    dispatch(ctx, "until", {"target": at})
    return ctx


# --------------------------------------------------------------------------- (1) first row


def test_reads_reach_an_earlier_stepped_row(tiny):
    ctx = _ctx(tiny)
    out = dispatch(ctx, "reads", {"source": "t0", "rows": "1", "consumer": CONSUMER,
                                  "causal": False})
    assert out["rows"] == [1]
    assert out["reads"] and all(r["row"] == 1 for r in out["reads"])


def test_a_symbol_on_an_uncomputed_earlier_row_is_refused(tiny):
    ctx = _ctx(tiny, at="t1.L0.h")
    with pytest.raises(Refused, match="t1.L1.k has not been computed"):
        dispatch(ctx, "symbol", {"name": "x", "addresses": "t1.L1.k"})


def test_seek_moves_back_to_an_earlier_stepped_row(tiny):
    ctx = _ctx(tiny)
    dispatch(ctx, "seek", {"target": "t1.L0.m"})
    assert ctx.run.where == "t1.L0.m"
    default = tiny.ops()
    dispatch(default, "start", {"prompt": CORRUPT})                    # rows 0..4 prefilled
    dispatch(default, "until", {"target": "t5.logits"})
    with pytest.raises(Refused, match="prefill=K"):
        dispatch(default, "seek", {"target": "t1.L0.m"})


def test_reverse_continue_finds_a_hit_on_an_earlier_stepped_row(tiny):
    ctx = _ctx(tiny)
    dispatch(ctx, "break", {"pattern": "t1.L0.m"})
    out = dispatch(ctx, "reverse-continue", {})
    assert ctx.run.where == "t1.L0.m" and out["reason"] != "start-of-run"


# --------------------------------------------------------------------------- (2) each prefill=


def test_each_takes_prefill_by_role_for_the_item_and_its_pair(tiny):
    dispatch(tiny.ops(), "promptset", {"action": "load", "name": "recall", "arg": str(FIXTURE)})
    line = "sweep h rows=@subject layers=0:2 = @pair:h --consumer M[$answer,$foil]"
    args = parse("each", f"recall prefill=@subject: {line}")
    assert args["prefill"] == "@subject"
    out = tiny.each("recall", line, prefill="@subject")
    rows = {row["name"]: row for row in out.summary["items"]}
    assert rows["a"]["state"] == "resolved" and rows["a"]["prefill"] == 1
    assert out.summary["prefill"] == "@subject"
    assert rows["d"]["state"] == "refused" and "subject" in rows["d"]["reason"]
    ctx = tiny.ops()                                   # item a by hand: b is its pair
    dispatch(ctx, "start", {"prompt": [1, 7, 9, 3], "name": "clean", "prefill": 1})
    dispatch(ctx, "until", {"target": "t3.logits"})
    dispatch(ctx, "start", {"prompt": [1, 5, 9, 3], "name": "byhand", "prefill": 1})
    dispatch(ctx, "until", {"target": "t3.logits"})
    hand = dispatch(ctx, "sweep", {"port": "h", "rows": "1", "layers": "0:2", "origin": "clean",
                                   "consumer": "M[4,9]"})
    assert out.results["a"]["table"] == hand["table"]
    past = tiny.each("recall", "print t@last.logits", prefill="4")
    assert all(r["state"] == "refused" and "past the prompt" in r["reason"]
               for r in past.summary["items"] if r["name"] != "c")


# --------------------------------------------------------------------------- (3) one session


def test_start_with_prefill_builds_the_session_once(tiny, monkeypatch):
    made = []
    original = tiny.adapter.session

    def counted(tokens, **kw):
        made.append(kw.get("prefill"))
        return original(tokens, **kw)

    monkeypatch.setattr(tiny.adapter, "session", counted)
    ctx = tiny.ops()
    dispatch(ctx, "start", {"prompt": CORRUPT, "prefill": 2})
    assert made == [2] and ctx.run.where == "t2.L0.h"
    run = tiny.run(CORRUPT, prefill=0)
    assert made == [2, 0] and run.where == "t0.L0.h"


# --------------------------------------------------------------------------- (4) records


def test_forecast_whatif_and_symbol_records_carry_the_split(tmp_path):
    from mdb.analysis import forecast

    debugger = mdb.open_tiny("qwen2", "bf16", project=tmp_path / "p")
    ctx = _ctx(debugger, prefill=2, at="t5.L0.h")
    dispatch(ctx, "forecast", {"target": "M[4,9]"})
    dispatch(ctx, "until", {"target": "t5.logits"})
    entry = forecast.ledger(ctx.run)[max(forecast.ledger(ctx.run))]
    want = {"prefill": 2, "first_row": 2, "prompt_rows": 6}
    assert {k: entry.record["body"]["procedure"][k] for k in want} == want
    out = dispatch(ctx, "whatif", {"target": "t3.L1.h", "value": "clean:t3.L1.h",
                                   "consumer": CONSUMER})
    assert {k: out["sealed_record"]["body"]["procedure"][k] for k in want} == want
    symbols = dispatch(ctx, "symbols", {"consumer": "M[4,9]"})
    stored = symbols.get("stored")
    assert stored, symbols.keys()
    record = index.get_record(debugger.project, debugger.session_id, stored[0])
    assert record["body"]["procedure"]["prefill"] == 2


# --------------------------------------------------------------------------- (5) the bound


def test_a_cross_row_whatif_bound_says_it_is_a_worst_case_without_the_sign(tiny):
    ctx = _ctx(tiny, at="t2.L1.h")
    out = dispatch(ctx, "whatif", {"target": "t1.L1.h", "value": "clean:t1.L1.h",
                                   "consumer": CONSUMER})
    exact = out["prediction"]["exact"]
    assert exact["operands"]["source"] == "forecast" and exact["operands"]["stops"]["value"] > 0
    assert exact["bound"]["value"] > 1000 * abs(exact["value"]["value"])   # measured: 43197
    assert exact["wider_than_value"] and "forecast from the cursor" in exact["bound_scope"]
    assert "no sign" in render(out)
    # the measured change is what emulation predicted, bit for bit: the bound is loose, not wrong
    assert out["measured"]["delta"]["value"] == out["prediction"]["emulated"]["value"]["value"]


# --------------------------------------------------------------------------- (6, 7) sweep


def test_a_sweep_is_a_stored_record_with_its_table_by_pointer(tmp_path):
    debugger = mdb.open_tiny("qwen2", "bf16", project=tmp_path / "p")
    ctx = _ctx(debugger)
    out = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER})
    record = index.get_record(debugger.project, debugger.session_id, out["stored"])
    verify_record(record)
    body = record["kind"], record["body"]
    assert body[0] == "sweep" and out["record"] == record["id"]
    body = body[1]
    assert body["rows"] == "0:6" and body["layers"] == "0:2" and "seconds" not in body["cost"]
    assert body["procedure"]["prefill"] == 0 and body["origin_procedure"]["first_row"] == 0
    table = load_json({**body["table"], "tensor": record["pointers"][0]},
                      store=debugger.project.store)
    assert table == out["table"]


def test_sweep_plans_its_layer_calls_and_comes_back_without_a_replay(tiny):
    ctx = _ctx(tiny)
    run = ctx.run
    before = session_digest(ctx)
    out = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                  "rows": "2:5"})
    layers, c = tiny.adapter.layers, 5
    cells = sum((layers - l) + layers * (c - r) for r in range(2, 5) for l in range(layers))
    assert out["cost"]["planned"]["layer"]["value"] == cells
    assert out["cost"]["native"]["layer"]["value"] == cells       # no restore replay on top
    assert out["cost"]["native"]["probe"]["value"] <= 6            # origin reads only
    assert session_digest(ctx) == before and run.where == "t5.logits"


def test_sweep_over_its_cap_is_refused_and_max_calls_overrides(tiny):
    ctx = _ctx(tiny)
    with pytest.raises(Refused, match=r"78 layer calls planned.*max_calls=78"):
        dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                "max_calls": 50})
    assert parse("sweep", "h = clean --consumer t5.M[4,9] max_calls=78")["max_calls"] == 78
    out = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                  "max_calls": 78})
    assert out["cost"]["native"]["layer"]["value"] == 78


def test_the_repl_prints_the_plan_before_the_result(tiny):
    ctx = _ctx(tiny)
    buffer = io.StringIO()
    shell = Shell(ctx, out=buffer)
    shell.execute("sweep h rows=1:3 = clean --consumer t5.M[4,9]")
    text = buffer.getvalue().splitlines()
    assert text[0].startswith("sweep plan: 4 cells") and text[1].startswith("sweep h = clean")


def test_a_blob_table_still_draws_its_heat_table():
    dbg = mdb.attach(tiny_model("qwen2", "bf16", layers=12))
    ctx = _ctx(dbg)
    out = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                  "rows": "1:6"})
    assert "table" not in out and out["heat"]["rows"] == "1:6"
    text = render(out)
    assert "L11" in text and "cell: tenths of max" in text


def test_two_digit_row_labels_are_not_jammed():
    """Seen on Qwen2.5-1.5B rows 18:21: the header printed ``t18t19t20`` (3-column cells)."""
    result = {"table": {"rows": [18, 19, 20], "layers": [0, 1],
                        "delta": {"value": [[8.5, 0.0, 0.0], [0.1, 0.0, 0.0]],
                                  "authority": "measured"}}}
    lines = heat_table(result, 80)
    assert lines[0].split() == ["t18", "t19", "t20"]
    assert lines[1].split() == ["L0", "+9", "0", "0"]


# --------------------------------------------------------------------------- (9) staging


def test_staging_an_edit_makes_no_native_call(tiny):
    run = Run(tiny.adapter, CORRUPT, prefill=0)
    run.until("t1.L1.h")
    probes = run.calls["probe"]
    run.set("t1.L1.m", 0.0, "scale")             # past the cursor: the captures stay
    run.set("t1.L1.A", 0.5, "scale")
    run.set("t1.L1.h", 2.0, "scale")             # at the cursor: dropped, probed on a read
    assert run.calls["probe"] == probes
    seen = run.read("t1.L1.h")
    assert run.calls["probe"] == probes + 1
    eager = Run(tiny.adapter, CORRUPT, prefill=0)
    eager.until("t1.L1.h")
    eager.set("t1.L1.h", 2.0, "scale", check=True)
    assert torch.equal(seen, eager.read("t1.L1.h"))
    run.until("t5.logits")
    eager.set("t1.L1.m", 0.0, "scale")
    eager.set("t1.L1.A", 0.5, "scale")
    eager.until("t5.logits")
    assert torch.equal(run.read("t5.logits"), eager.read("t5.logits"))

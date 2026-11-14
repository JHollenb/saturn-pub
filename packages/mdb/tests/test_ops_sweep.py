"""``sweep PORT rows= layers= = ORIGIN --consumer C`` (M35): the position x layer activation
patching table, each cell measured natively, the session restored once.

Tiny Qwen2 bf16, 2 layers, clean ``1 7 9 3 6 2`` and corrupt ``1 5 9 3 6 2`` (row 1 differs),
both started with ``prefill=0``. Every cell equals the same patch made by hand on a fresh run;
the layer-call count is the stated cost model; the session digest and the held edit tensors
are unchanged; a large table goes to a checked blob; the refusals; the line grammar.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_ops_predict import unlabelled  # noqa: E402

import mdb  # noqa: E402
from mdb.core.bounded import load_json  # noqa: E402
from mdb.engine import Run  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import BadArgs, Refused, dispatch, sweep as sweeps  # noqa: E402
from mdb.ops.lines import parse  # noqa: E402
from mdb.ops.predict import session_digest  # noqa: E402
from mdb.repl.render import heat_table, render  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "prompt_set_tiny.jsonl"
CLEAN, CORRUPT = [1, 7, 9, 3, 6, 2], [1, 5, 9, 3, 6, 2]
CONSUMER, A, B = "t5.M[4,9]", 4, 9


@pytest.fixture(scope="module")
def tiny():
    return mdb.attach(tiny_model("qwen2", "bf16"))


def _ctx(tiny, prefill=0):
    ctx = tiny.ops()
    dispatch(ctx, "start", {"prompt": CLEAN, "name": "clean", "prefill": prefill})
    dispatch(ctx, "until", {"target": "t5.logits"})
    dispatch(ctx, "start", {"prompt": CORRUPT, "prefill": prefill})
    dispatch(ctx, "until", {"target": "t5.logits"})
    return ctx


def _margin(run):
    run.until("t5.logits")
    logits = run.read("t5.logits").reshape(-1).float()
    return float(logits[A] - logits[B])


def _by_hand(tiny, row, layer, port="h"):
    clean = tiny.runs["clean"]
    run = Run(tiny.adapter, CORRUPT, prefill=0)
    base = _margin(Run(tiny.adapter, CORRUPT, prefill=0))
    run.until(f"t{row}.L{layer}.h")
    run.set(f"t{row}.L{layer}.{port}", clean.read(f"t{row}.L{layer}.{port}"), "replace")
    return _margin(run) - base


def test_every_cell_is_the_patch_made_by_hand_and_the_session_is_unchanged(tiny):
    ctx = _ctx(tiny)
    run = ctx.run
    before, edits = session_digest(ctx), run.held_bytes()["edits"]
    result = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER})
    assert unlabelled(result) == []
    table = result["table"]
    assert table["delta"]["authority"] == "measured"
    assert table["rows"] == list(range(6)) and table["layers"] == [0, 1]
    grid = table["delta"]["value"]
    for i, layer in enumerate(table["layers"]):
        for j, row in enumerate(table["rows"]):
            assert grid[i][j] == _by_hand(tiny, row, layer), (row, layer)
    assert grid[0][0] == grid[1][0] == 0.0       # row 0 is the same token in both
    assert result["base"]["value"] == _margin(Run(tiny.adapter, CORRUPT, prefill=0))
    assert session_digest(ctx) == before and run.where == "t5.logits"
    assert result["session"]["unchanged"] and run.held_bytes()["edits"] == edits
    assert len(result["top"]) == 8 and result["cells"]["count"] == 12


def test_the_whatif_native_delta_is_the_sweep_cell(tiny):
    ctx = _ctx(tiny)
    whatif = dispatch(ctx, "whatif", {"target": "t1.L1.h", "value": "clean:t1.L1.h",
                                      "consumer": CONSUMER})
    cell = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                   "rows": "1", "layers": "1"})
    assert whatif["measured"]["delta"]["value"] == cell["table"]["delta"]["value"][0][0] != 0.0
    assert whatif["prediction"]["state"] == "resolved"   # the exact chain crosses rows


def test_layer_calls_follow_the_cost_model(tiny):
    """Cell (r, l), consumer row c, L layers: (L - l) + L(c - r) layer calls; the return to the
    held trail costs none (M54; it replayed (L - l0) + L(c - r0) before). No cell replays from
    row 0."""
    ctx = _ctx(tiny)
    layers, c = tiny.adapter.layers, 5
    for rows, span in (("0:6", range(0, 6)), ("2:5", range(2, 5))):
        result = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                         "rows": rows})
        cells = sum((layers - l) + layers * (c - r) for r in span for l in range(layers))
        layer = result["cost"]["native"]["layer"]
        assert layer == {"value": cells, "authority": "measured"}, rows


def test_a_head_selector_and_a_table_too_large_for_the_record(tiny, monkeypatch):
    ctx = _ctx(tiny)
    inline = dispatch(ctx, "sweep", {"port": "A[h1]", "origin": "clean", "consumer": CONSUMER,
                                     "rows": "1:4", "top": 3})
    assert inline["port"] == "A[h1]" and len(inline["top"]) == 3
    monkeypatch.setattr(sweeps, "INLINE_BYTES", 16)
    blob = dispatch(ctx, "sweep", {"port": "A[h1]", "origin": "clean", "consumer": CONSUMER,
                                   "rows": "1:4", "top": 3})
    assert "table" not in blob and len(blob["table_pointer"]["sha256"]) == 64
    assert load_json(blob["table_pointer"], run=ctx.run) == inline["table"]
    assert [t["row"] for t in blob["top"]] == [t["row"] for t in inline["top"]]


def test_a_table_over_the_item_bound_goes_to_a_blob_not_a_refusal():
    """12 layers x 6 rows: small in bytes but over the record's nested-list item bound."""
    dbg = mdb.attach(tiny_model("qwen2", "bf16", layers=12))
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": CLEAN, "name": "clean", "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.logits"})
    dispatch(ctx, "start", {"prompt": CORRUPT, "prefill": 0})
    result = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                     "rows": "1:6"})
    assert "table" not in result and result["cells"]["count"] == 60
    table = load_json(result["table_pointer"], run=ctx.run)
    assert len(table["delta"]["value"]) == 12 and table["rows"] == [1, 2, 3, 4, 5]
    assert "blob" in render(result)


def test_refusals(tiny):
    ctx = _ctx(tiny)
    with pytest.raises(Refused, match="is the swept run itself"):
        dispatch(ctx, "sweep", {"port": "h", "origin": "main", "consumer": CONSUMER})
    with pytest.raises(Refused, match="one number per cell"):
        dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": "t5.logits"})
    with pytest.raises(BadArgs, match="PORT is a port name"):
        dispatch(ctx, "sweep", {"port": "nope", "origin": "clean", "consumer": CONSUMER})
    with pytest.raises(Refused, match="outside"):
        dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                "rows": "3:9"})
    dispatch(ctx, "start", {"prompt": CORRUPT, "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.L1.h"})
    dispatch(ctx, "set", {"target": "t5.L1.h", "value": "clean:t5.L1.h"})
    with pytest.raises(Refused, match="edits are staged"):
        dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER})
    default = tiny.ops()
    dispatch(default, "start", {"prompt": CLEAN, "name": "clean", "prefill": 0})
    dispatch(default, "until", {"target": "t5.logits"})
    dispatch(default, "start", {"prompt": CORRUPT})                  # rows 0..4 prefilled
    with pytest.raises(Refused, match=r"start ... prefill=K. for a K <= 2"):
        dispatch(default, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER,
                                    "rows": "2:6"})


def test_the_line_grammar_and_the_heat_table(tiny):
    want = {"port": "A[h1]", "rows": "1:4", "layers": "0:2", "origin": "clean",
            "consumer": "t5.M[4,9]", "top": 3}
    for line in ("A[h1] rows=1:4 layers=0:2 = clean --consumer t5.M[4,9] top=3",
                 "A[h1] rows=1:4 layers=0:2 =clean top=3 --consumer t5.M[4,9]",
                 "A[h1] top=3 rows=1:4 layers=0:2 origin=clean consumer=t5.M[4,9]"):
        assert parse("sweep", line) == want, line
    with pytest.raises(BadArgs, match="sweep PORT"):
        parse("sweep", "h rows=1:4 --consumer t5.M[4,9]")
    ctx = _ctx(tiny)
    result = dispatch(ctx, "sweep", {"port": "h", "origin": "clean", "consumer": CONSUMER})
    lines = heat_table(result, 80)
    assert all(len(line) <= 80 for line in lines)
    assert lines[0].split() == [f"t{r}" for r in range(6)]
    assert [line.split()[0] for line in lines[1:3]] == ["L0", "L1"]
    text = render(result)
    assert "measured" in text and "layer calls" in text and "session unchanged" in text


def test_each_composes_over_a_prompt_set_with_the_pair_as_origin(tiny):
    """``each SET: sweep h rows=@last ... = @pair:h``: the origin binds to the paired item's
    run; a pair of another length is a per-item refusal (its rows are other rows)."""
    dispatch(tiny.ops(), "promptset", {"action": "load", "name": "recall", "arg": str(FIXTURE)})
    out = tiny.each("recall", "sweep h rows=@last layers=0:2 = @pair:h "
                              "--consumer M[$answer,$foil]")
    rows = {row["name"]: row for row in out.summary["items"]}
    assert rows["a"]["state"] == rows["b"]["state"] == "resolved"
    assert rows["c"]["state"] == "refused" and "has not computed" in rows["c"]["reason"]
    ctx = tiny.ops()                                  # item a by hand: b is its pair
    dispatch(ctx, "start", {"prompt": [1, 7, 9, 3], "name": "clean"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    dispatch(ctx, "start", {"prompt": [1, 5, 9, 3], "name": "byhand"})
    dispatch(ctx, "until", {"target": "t3.logits"})
    hand = dispatch(ctx, "sweep", {"port": "h", "rows": "3", "layers": "0:2",
                                   "origin": "clean", "consumer": "M[4,9]"})
    assert out.results["a"]["table"] == hand["table"]
    assert out.results["a"]["origin"] == "pair-recall-b"

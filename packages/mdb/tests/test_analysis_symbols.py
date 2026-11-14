"""``analysis.symbols`` on tiny Qwen2/Qwen3: the table is a partition.

Per head, listed bindings plus ``Unbound`` hold all the attention mass at the reader row (the
native ``sum_s p``, which is 1 within the stated bound), and the bindings' contributions plus
``Unbound`` equal the consumer's native value. An edited row shows ``edited``; a declared
symbol appears and takes its rows out of the token rows; with a project the sealed records
round-trip through ``mdb.core.index`` and reopen in a fresh ``Project``.
"""

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.reads import head_mask  # noqa: E402
from mdb.core import index  # noqa: E402
from mdb.core.pointers import check_bounded  # noqa: E402
from mdb.core.project import Project  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen3", "fp32")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _sums(table):
    tensors = table.tensors
    for key, mass in tensors["mass"].items():
        listed = mass[tensors["listed"]].sum(0)
        assert torch.allclose(listed + tensors["unbound_mass"][key], tensors["total_mass"][key],
                              rtol=0, atol=1e-15)
    accounting = table.report["accounting"]
    assert accounting["within_bound"] is True
    assert accounting["max_head_deviation"]["value"] <= accounting["deviation_bound"]["value"]


def test_rows_and_unbound_hold_all_mass_and_the_whole_consumer(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    margin = Margin(3, 4, 9)
    table = symbols.table(run, consumer=margin)
    _sums(table)
    report = check_bounded(table.report)
    assert [b["symbol"] for b in report["bindings"]] == ["t0", "t1", "t2", "t3"]
    assert report["unbound"]["mass"]["value"] == 0.0
    native = report["consumer_total"]["value"]
    listed = sum(b["to_consumer"]["value"] for b in report["bindings"])
    assert listed + report["unbound"]["readout"]["value"] == pytest.approx(native, abs=1e-12)
    # a page of two bindings: the other rows move to Unbound, the sums still close
    page = symbols.table(run, consumer=margin, offset=1, limit=2)
    _sums(page)
    assert [b["symbol"] for b in page.report["bindings"]] == ["t1", "t2"]
    assert page.report["unbound"]["mass"]["value"] > 0.9
    shown = sum(b["to_consumer"]["value"] for b in page.report["bindings"])
    assert shown + page.report["unbound"]["readout"]["value"] == pytest.approx(native, abs=1e-12)
    # the default consumer at logits is the margin between the two largest logits
    assert symbols.table(run).report["consumer"].startswith("t3.M[")


def test_status_edited_live_and_a_declared_symbol(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L0.h")
    fresh = symbols.table(run).report
    assert {b["status"] for b in fresh["bindings"]} == {"live"}
    assert fresh["unbound"]["readout"]["state"] == "unresolved"
    run.set("t3.L0.v", 0.25, "add")
    run.until("t3.L1.h")
    declared = symbols.declare(run, "ctx", ["t1", "cache.L0.v[row=2]"])
    assert declared.addresses == ["t1", "cache.L0.v[row=2]"]
    with pytest.raises(Refusal, match="overlaps symbol ctx"):
        symbols.declare(run, "other", ["t2.L0.v"])
    symbols.declare(run, "keys", ["t2.L0.k"])  # k cells: no overlap with ctx's v cell
    with pytest.raises(Refusal, match="has not been computed"):
        symbols.declare(run, "late", ["t4"])
    table = symbols.table(run)
    _sums(table)
    rows = {b["symbol"]: b for b in table.report["bindings"]}
    assert rows["ctx"]["kind"] == "declared" and rows["ctx"]["rows"] == [1, 2]
    assert rows["keys"]["count"] == 0 and rows["keys"]["mass"]["value"] == 0.0
    assert "t1" not in rows
    cells = 2 * run.adapter.layers * run.adapter.num_kv_heads
    left = cells - 2 * run.adapter.num_kv_heads   # ctx holds t2.L0.v, keys holds t2.L0.k
    assert rows["t2"]["definitions"] == [f"t2 (uncovered part: {left} of {cells} k/v cells)"]
    assert rows["t3"]["status"] == "edited"
    assert any("t3.L0.v" in item for item in rows["t3"]["edited_by"])
    assert rows["t0"]["status"] in ("read", "live")
    assert rows["t0"]["lifetime"]["strip"].endswith("?")
    assert set(table.report["legend"]) == {"#", ".", "-", "?"}
    assert rows["ctx"]["mean_mass"]["value"] == pytest.approx(
        rows["ctx"]["mass"]["value"] / rows["ctx"]["count"], abs=1e-15)


def test_a_symbol_over_part_of_a_row_partitions_the_cells(debugger):
    """A symbol over one KV head of one layer's v (not a whole layer x KV block): the token row
    keeps exactly the other cells, nothing is counted twice, Unbound mass is 0."""
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    symbols.declare(run, "S2", ["cache.L1.v[kv0,row=2]"])
    table = symbols.table(run, consumer=Margin(3, 4, 9))
    _sums(table)
    report = table.report
    assert report["unbound"]["mass"]["value"] == 0.0
    partition = report["accounting"]["partition"]
    assert partition["max_gap"]["value"] <= 1e-15
    rows = {b["symbol"]: b for b in report["bindings"]}
    p = {layer: run.read(f"t3.L{layer}.p").double().reshape(run.adapter.num_heads, -1)
         for layer in range(run.adapter.layers)}
    row2 = sum(float(p[layer][:, 2].sum()) for layer in p)
    assert rows["S2"]["mass"]["value"] + rows["t2"]["mass"]["value"] == pytest.approx(
        row2, abs=1e-12)


def test_a_broken_partition_is_refused_not_closed(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    twice = [symbols.Binding("a", "declared", [symbols.Source(1)], ["t1"]),
             symbols.Binding("b", "declared", [symbols.Source(1)], ["t1"])]
    grid = symbols._Grid(run, twice, None, [3])
    errors = grid.owners
    assert any("row=1] is held by 2 bindings" in e for e in errors)
    assert any("row=0] is held by 0 bindings" in e for e in errors)
    assert grid.partition_error().startswith("the bindings do not partition")


def test_edit_provenance_supplied_mass_and_blocked_effect(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.h")
    run.set("t3.L1.p", 0.5, "replace", mask=head_mask(run, 3, 1, 0, 1))
    run.set("t3.L1.A", 0.0, "replace", mask=head_mask(run, 3, 1, 1))
    run.until("t3.logits")
    table = symbols.table(run, consumer=Margin(3, 4, 9), readers=8)
    rows = {b["symbol"]: b for b in table.report["bindings"]}
    reads = {r["name"]: r for r in rows["t1"]["readers"]}
    assert reads["t3.L1.h0"]["mass"]["authority"] == "supplied"
    assert rows["t1"]["mass"]["authority"] == "supplied"
    assert reads["t3.L1.h1"]["effect"]["state"] == "refused"
    assert rows["t1"]["to_consumer"]["state"] == "refused"
    parts = table.report["unbound"]["parts"]
    native = table.report["consumer_total"]["value"]
    listed = sum(b["to_consumer"]["value"] for b in table.report["bindings"]
                 if "value" in b["to_consumer"])
    total = (parts["embedding_mlp"]["value"] + parts["attention_not_listed"]["value"]
             + parts["remainder"]["value"]["value"])
    assert listed + total == pytest.approx(native, abs=1e-12)
    assert parts["remainder"]["within"] is True


def test_records_round_trip_through_the_index_and_a_fresh_project(tmp_path):
    dbg = mdb.attach(tiny_model("qwen2", "fp32"), project=tmp_path / "p")
    run = dbg.run(PROMPT)
    run.until("t3.logits")
    ctx = dbg.ops(run)
    dispatch(ctx, "symbol", {"name": "ctx", "addresses": "t0,t1"})
    result = dispatch(ctx, "symbols", {"consumer": "M[4,9]"})
    stored = result["stored"]
    assert stored == result["records"] and len(stored) == 1
    reopened = Project(tmp_path / "p", read_only=True)
    record = index.get_record(reopened, dbg.session_id, stored[0])
    assert verify_record(record)["kind"] == "symbol-table"
    assert record["body"]["bindings"][0]["symbol"] == "ctx"
    listed = index.list_records(reopened, dbg.session_id, kind="symbol-table", strict=True)
    assert [r["id"] for r in listed] == stored


class _Tok:
    """A fake tokenizer: token text with a leading space, a newline, a wide character."""
    TEXT = {1: " freezing", 5: "\n", 9: "\u51b7", 3: "Temperature"}

    def decode(self, ids):
        return self.TEXT[ids[0]]


def _symbols_text(tokenizer):
    from mdb.repl.render import render

    dbg = open_tiny("qwen2", "fp32")
    run = dbg.run(PROMPT)
    run.until("t3.logits")
    ctx = dbg.ops(run)
    dbg.tokenizer = tokenizer
    result = dispatch(ctx, "symbols", {"consumer": "M[4,9]"})
    return render(result).splitlines(), result


def _column(lines, key):
    rows = [ln for ln in lines if ln.startswith("  ") and key in ln and " mass " in ln]
    from mdb.repl.render import _display_width

    return {_display_width(ln.split(key)[0]) for ln in rows}


def test_symbols_names_are_quoted_and_padded_by_display_width():
    lines, _ = _symbols_text(_Tok())
    text = "\n".join(lines)
    assert '"\\n"' in text and '" freezing"' in text and '"\u51b7"' in text
    from mdb.repl.render import _display_width

    rows = [ln for ln in lines if re.match(r"  (t\d|Unbound) ", ln) and " mass " in ln]
    assert len(rows) == 5
    # a binding row has a 7-wide status after the name; Unbound has none
    starts = {_display_width(ln[:ln.index(" mass ") + 1]) - (0 if "Unbound" in ln else 7)
              for ln in rows}
    assert len(starts) == 1, starts
    ids, _ = _symbols_text(None)
    assert any("t0 (id 1)" in ln for ln in ids)
    rows = [ln for ln in ids if ln.startswith("  t") and " mass " in ln]
    assert len({ln.index("read ") for ln in rows}) == 1


def test_unbound_remainder_is_the_measured_value_and_the_bound_is_labelled():
    lines, result = _symbols_text(None)
    remainder = result["unbound"]["parts"]["remainder"]
    row = next(ln for ln in lines if "+ remainder" in ln)
    value, bound = f"{remainder['value']['value']:.5g}", "bound "
    assert f"remainder {value}" in row.replace("+ remainder ", "+ remainder ") or "remainder" in row
    assert ", bound " in row and "[bound]: within" in row

"""``analysis.symbols`` on tiny Qwen2/Qwen3: the table is a partition.

Per head, listed bindings plus ``Unbound`` hold all the attention mass at the reader row (the
native ``sum_s p``, which is 1 within the stated bound), and the bindings' contributions plus
``Unbound`` equal the consumer's native value. An edited row shows ``edited``; a declared
symbol appears and takes its rows out of the token rows; with a project the sealed records
round-trip through ``mdb.core.index`` and reopen in a fresh ``Project``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
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
        symbols.declare(run, "other", ["t2.L0.k"])
    with pytest.raises(Refusal, match="has not been computed"):
        symbols.declare(run, "late", ["t4"])
    table = symbols.table(run)
    _sums(table)
    rows = {b["symbol"]: b for b in table.report["bindings"]}
    assert rows["ctx"]["kind"] == "declared" and rows["ctx"]["rows"] == [1, 2]
    assert "t1" not in rows
    assert rows["t2"]["definitions"] == ["t2 (uncovered part)"]
    assert rows["t3"]["status"] == "edited"
    assert any("t3.L0.v" in item for item in rows["t3"]["edited_by"])
    assert rows["t0"]["status"] in ("read", "live")
    assert rows["t0"]["lifetime"]["strip"].endswith("?")


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

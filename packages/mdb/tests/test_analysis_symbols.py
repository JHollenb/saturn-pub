"""``analysis.symbols`` on tiny Qwen2/Qwen3: the table is a partition.

Per head, listed bindings plus ``Unbound`` hold all the attention mass at the reader row (the
native ``sum_s p``, which is 1 within the stated bound), and the bindings' contributions plus
``Unbound`` equal the consumer's native value. An edited row shows ``edited``; a declared
symbol appears and takes its rows out of the token rows; with a project the sealed records
round-trip through ``mdb.core.index`` and reopen in a fresh ``Project``.
"""

import pytest
import torch

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import mdb  # noqa: E402
from mdb.analysis import symbols  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.reads import head_mask, readout  # noqa: E402
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


# --- content readouts, content lifetime, hop -------------------------------------------------


def test_content_readout_equals_a_hand_projection_of_the_bindings_write(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    symbols.declare(run, "cell", ["cache.L1.v[kv0,row=1]"])
    sets = [symbols.Candidates("small", (2, 4, 6, 8), truth=4),
            symbols.Candidates("large", (10, 11, 12))]
    table = symbols.table(run, consumer=Margin(3, 4, 9), sets=sets, content_layers=True)
    _sums(table)                                  # the content fields leave the partition intact
    view = next(b for b in table.report["bindings"] if b["symbol"] == "cell")
    # by hand: the cell's write is the reads of row 1 through KV head 0 at layer 1
    from mdb.analysis.reads import head_mix
    mix = head_mix(run, 3, 1, {})
    write = sum(mix.o_parts()[h, 1] for h in range(mix.p.shape[0]) if mix.kappa[h] == 0)
    rho, gamma = symbols.readout_basis(run, 3)
    W = run.adapter.model.lm_head.weight.double()
    for cand in sets:
        hand = rho * (W[list(cand.tokens)] * gamma) @ write.double()
        best = cand.tokens[int(hand.argmax())]
        got = view["content"][cand.name]
        assert got["best"] == best
        assert got["score"]["value"] == pytest.approx(float(hand.max()), abs=1e-12)
        assert got["score"]["authority"] == "derived-exact"
    assert "truth_rank" in view["content"]["small"] and "truth_rank" not in view["content"]["large"]
    assert len(view["content"]["vocab_top"]["tokens"]) == 5
    strip = view["content_lifetime"]["small"]
    assert len(strip) == run.adapter.layers and strip[0] == "-" and strip[1] in "C"
    check_bounded(table.report)


def test_content_is_absent_without_sets_and_hop_reports_the_indirect_share(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    plain = symbols.table(run, consumer=Margin(3, 4, 9))
    assert all("content" not in b for b in plain.report["bindings"])
    assert plain.tensors["write"] == {}
    hop = symbols.table(run, consumer=Margin(3, 4, 9),
                        hops={"t1": {"native_delta": -2.0, "direct_delta": -0.5}})
    view = next(b for b in hop.report["bindings"] if b["symbol"] == "t1")
    assert view["hop"]["indirect_share"]["value"] == pytest.approx(0.75)
    assert view["hop"]["native_delta"]["authority"] == "measured"
    with pytest.raises(Refusal, match="at least two"):
        symbols.table(run, consumer=Margin(3, 4, 9), sets=[symbols.Candidates("one", (2,))])


def test_hop_edge_ladder_closes_and_its_parts_match_hand_computations(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    reader = readout(run, Margin(3, 4, 9))
    donor = debugger.run([2, 7, 9, 3])
    value = donor.read_cache(0, "v", staged=False).clone()[:, 0:1, 1:2]
    edited = debugger.run(PROMPT)
    edited.set_cell(0, "v", value, rows=[1], kv=[0])
    edited.until("t3.logits")
    edge = symbols.hop_edge(run, edited, reader, cell_layer=0, top_neurons=3, spectral=True,
                            top_bins=2)
    fp32 = run.adapter.model.lm_head.weight.dtype == torch.float32
    tol = 1e-4 if fp32 else 5e-2
    assert edge["upstream"]["value"] == 0.0               # the cell layer's input row is untouched
    from mdb.analysis.reads import head_mix
    w = [sum(head_mix(r, 3, 0, {}).o_parts()[h, 1] for h in range(run.adapter.num_heads)
             if head_mix(r, 3, 0, {}).kappa[h] == 0) for r in (run, edited)]
    hand = float(reader.vector @ (w[1] - w[0]).double().reshape(-1))
    assert edge["direct"]["value"] == pytest.approx(hand, abs=tol)
    for layer in edge["mlp"]:
        listed = sum(n["effect"]["value"] for n in layer["neurons"])
        assert listed + layer["omitted"]["effect"]["value"] == pytest.approx(
            layer["exact"]["value"], abs=1e-9)
        for n in layer["neurons"]:
            assert n["b0_ds"]["value"] + n["s0_db"]["value"] + n["ds_db"]["value"] == \
                pytest.approx(n["df"]["value"], abs=1e-12)
        assert abs(layer["down_rounding"]["value"]) <= tol + 1e-3 * abs(layer["exact"]["value"])
        assert layer["spectral"]["parseval_max_error"]["value"] < 1e-9
    native = edge["consumer"]["native_delta"]["value"]
    parts = (edge["direct"]["value"] + edge["upstream"]["value"] + edge["totals"]["mlp"]["value"]
             + edge["totals"]["reread"]["value"] + edge["norm"]["value"]
             + edge["residual_add_rounding"]["value"] + edge["rounding"]["value"])
    assert parts == pytest.approx(native, abs=1e-9)
    if fp32:
        assert abs(edge["rounding"]["value"]) < 1e-3 * max(1.0, abs(native))
    table = symbols.table(run, consumer=Margin(3, 4, 9), hops={"t1": {
        "native_delta": native, "direct_delta": edge["direct"]["value"], "edge": edge}})
    view = next(b for b in table.report["bindings"] if b["symbol"] == "t1")
    assert view["hop"]["edge"]["mlp"][0]["layer"] == 0
    assert len(view["hop"]["edge"]["reread"]) == run.adapter.layers - 1
    check_bounded(table.report)

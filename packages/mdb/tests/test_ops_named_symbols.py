"""Named symbols from a transplant result (M23), on tiny Qwen2 (bf16).

``symbol NAME = whatif`` seals a ``symbol`` record (role, source row, readers, consumer,
measured effect, origin); ``symbol show``/``symbols`` list it; the name works where an address
is expected (``print``, ``whatif NAME = clean:NAME``, ``break``) with the same numbers as the
address; inside ``each`` it binds through each item's role (and ``@pair:NAME`` through the
pair's), so it gives what the explicit ``t@ROLE`` line gives; a use where the role does not
bind is refused with the reason; a result whose verdict is not ``effect`` (or that is not a
transplant) declares nothing; the record stays a bounded, verifiable record; the server's
``/op`` gives the REPL's result.
"""

import json

import pytest

import mdb
from mdb.core.pointers import MAX_RECORD_BYTES
from mdb.core.records import stored_bytes, verify_record
from mdb.ops import Refused, dispatch
from mdb.ops.lines import parse

CLEAN = [1, 7, 9, 3, 6, 2]
MAIN = [1, 5, 9, 3, 6, 2]
ITEMS = [
    {"name": "a", "tokens": MAIN, "roles": {"subject": 1}, "vars": {"answer": 4, "foil": 9},
     "pair": "b"},
    {"name": "b", "tokens": CLEAN, "roles": {"subject": 1}, "vars": {"answer": 4, "foil": 9},
     "pair": "a"},
    {"name": "c", "tokens": [2, 6, 10, 4, 8], "roles": {"subject": 2},
     "vars": {"answer": 6, "foil": 7}, "pair": "a"},
    {"name": "d", "tokens": [3, 11, 5, 4], "vars": {"answer": 4, "foil": 9}, "pair": "a"},
]
TRANSPLANT = {"target": "t1.L1.v[kv0]", "value": "clean:t1.L1.v[kv0]", "consumer": "M[4,9]"}


def _session(with_set=False):
    dbg = mdb.open_tiny("qwen2")
    ctx = dbg.ops()
    dispatch(ctx, "start", {"name": "clean", "prompt": CLEAN, "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.logits"})
    dispatch(ctx, "start", {"name": "main", "prompt": MAIN, "prefill": 0})
    dispatch(ctx, "until", {"target": "t5.logits"})
    if with_set:
        dispatch(ctx, "promptset", {"action": "define", "name": "pairs",
                                    "arg": json.dumps(ITEMS)})
    return dbg, ctx


def _timeless(node):
    """A result without its wall-clock fields (``seconds``)."""
    if isinstance(node, dict):
        return {k: _timeless(v) for k, v in node.items() if k != "seconds"}
    if isinstance(node, list):
        return [_timeless(v) for v in node]
    return node


def _declare(ctx, name="ctx1", **extra):
    found = dispatch(ctx, "whatif", dict(TRANSPLANT))
    out = dispatch(ctx, "symbol", {"name": name, "result": "whatif", **extra})
    return found, out


@pytest.fixture(scope="module")
def declared():
    dbg, ctx = _session()
    found, out = _declare(ctx)
    return dbg, ctx, found, out


def test_declare_from_a_whatif_transplant_seals_role_row_readers_and_the_measured_effect(declared):
    dbg, ctx, found, out = declared
    body = out["body"]
    assert out["action"] == "declare" and out["symbol"] == "ctx1"
    assert body["role"]["writer"] == {"layer": 1, "port": "v", "form": "occurrence",
                                      "address": "L1.v[kv0]"}
    row = body["row"]
    assert (row["row"], row["rule"], row["address"], row["token"]) == (1, "absolute",
                                                                      "t1.L1.v[kv0]", 5)
    readers = body["role"]["readers"]
    assert readers["state"] == "resolved" and readers["source"] == "cache.L1.v[kv0,row=1]"
    assert [r["name"] for r in readers["listed"]] == ["t5.L1.h0", "t5.L1.h1"]
    assert all(r["mass"]["authority"] == "measured" and "verdict" in r["causal"]
               for r in readers["listed"])
    evidence = body["evidence"]
    assert evidence["delta"] == {k: found["measured"]["delta"][k] for k in ("value", "authority")}
    assert evidence["delta"]["authority"] == "measured" and evidence["verdict"] == "effect"
    assert body["consumer"]["address"] == "t5.M[4,9]"
    assert body["origin"]["donor"]["address"] == "clean:t1.L1.v[kv0]"
    assert body["procedure"]["prefill"] == 0
    record = dispatch(ctx, "symbol", {"name": "ctx1", "action": "show"})["record"]
    assert record["kind"] == "symbol" and record["id"] == out["record"]


def test_the_record_verifies_and_stays_under_the_record_bound_with_eight_readers():
    from mdb.ops import named

    dbg, ctx = _session()
    _declare(ctx, "wide", readers=8)
    record = named.store(dbg)["wide"]
    assert verify_record(record)["id"] == record["id"]
    assert len(stored_bytes(record)) < MAX_RECORD_BYTES
    assert record["body"]["role"]["readers"]["count"] == 2   # tiny qwen2: 2 heads read kv0


def test_show_lists_and_delete_removes_the_named_symbol(declared):
    dbg, ctx, _, out = declared
    shown = dispatch(ctx, "symbol", parse("symbol", "show ctx1"))
    assert shown["here"] == {"address": "t1.L1.v[kv0]", "row": 1, "how": "its declaring prompt"}
    table = dispatch(ctx, "symbols", {"consumer": "M[4,9]", "limit": 1})
    assert [n["symbol"] for n in table["named"]] == ["ctx1"]
    assert table["named"][0]["delta"] == out["body"]["evidence"]["delta"]
    dbg2, ctx2 = _session()
    _declare(ctx2, "gone")
    assert dispatch(ctx2, "symbol", parse("symbol", "delete gone"))["action"] == "delete"
    assert "named" not in dispatch(ctx2, "symbols", {"limit": 1})
    with pytest.raises(Refused, match="no named symbol 'gone'"):
        dispatch(ctx2, "symbol", {"name": "gone", "action": "show"})


def test_the_name_is_an_address_for_print_whatif_and_break(declared):
    dbg, ctx, found, _ = declared
    by_name = dispatch(ctx, "print", {"address": "ctx1"})
    by_address = dispatch(ctx, "print", {"address": "t1.L1.v[kv0]"})
    assert by_name["stats"] == by_address["stats"]
    assert by_name["symbol_uses"][0]["address"] == "t1.L1.v[kv0]"
    again = dispatch(ctx, "whatif", parse("whatif", "ctx1 = clean:ctx1 --consumer M[4,9]"))
    assert again["measured"]["delta"] == found["measured"]["delta"]
    assert [u["address"] for u in again["symbol_uses"]] == ["t1.L1.v[kv0]", "clean:t1.L1.v[kv0]"]
    point = dispatch(ctx, "break", parse("break", "ctx1"))
    assert point["breakpoint"]["pattern"] == "t1.L1.v[kv0]"
    dispatch(ctx, "delete", {"ids": [point["breakpoint"]["id"]]})


def test_reuse_across_a_prompt_set_binds_each_items_role_and_its_pairs():
    dbg, ctx = _session(with_set=True)
    _, out = _declare(ctx, "subj")
    assert out["body"]["row"]["rule"] == "role" and out["body"]["row"]["role"] == "subject"
    assert out["body"]["row"]["address"] == "t@subject.L1.v[kv0]"
    line = "whatif subj = @pair:subj --consumer t@last.M[$answer,$foil]"
    explicit = ("whatif t@subject.L1.v[kv0] = @pair:t@subject.L1.v[kv0] --consumer "
                "t@last.M[$answer,$foil]")
    named_run = dispatch(ctx, "each", {"set": "pairs", "line": line, "prefill": "@subject"})
    plain_run = dispatch(ctx, "each", {"set": "pairs", "line": explicit, "prefill": "@subject"})
    rows = {r["name"]: r for r in named_run["items"]}
    plain = {r["name"]: r for r in plain_run["items"]}
    for item in ("a", "b", "c"):
        got, want = (_timeless(dispatch(ctx, "promptset", {"action": "result", "name": run["id"],
                                                          "arg": item})["result"])
                     for run in (named_run, plain_run))
        assert got == want, item
    assert rows["c"]["bound"]["target"] == "t2.L1.v[kv0]"
    assert rows["c"]["bound"]["value"] == "pair-pairs-a:t1.L1.v[kv0]"
    assert rows["a"]["key"] == plain["a"]["key"]


def test_a_use_where_the_role_does_not_bind_is_refused_with_the_reason():
    dbg, ctx = _session(with_set=True)
    _declare(ctx, "subj")
    found = dispatch(ctx, "each", {"set": "pairs", "line": "print subj"})
    d = next(r for r in found["items"] if r["name"] == "d")
    assert d["state"] == "refused" and "no role 'subject'" in d["reason"]
    dispatch(ctx, "start", {"name": "other", "prompt": [3, 11, 5, 4, 8], "prefill": 0})
    dispatch(ctx, "until", {"target": "t4.logits"})
    with pytest.raises(Refused, match=r"binds role @subject.*no loaded prompt-set item"):
        dispatch(ctx, "print", {"address": "subj"})
    dispatch(ctx, "frame", {"address": "main:"})
    with pytest.raises(Refused, match="nowhere is not a live run"):
        dispatch(ctx, "whatif", {"target": "subj", "value": "nowhere:subj",
                                 "consumer": "M[4,9]"})


def test_an_absolute_symbol_binds_by_prefix_and_refuses_another_prefix(declared):
    dbg, ctx, _, _ = declared
    dispatch(ctx, "start", {"name": "other2", "prompt": [1, 5, 2, 2], "prefill": 0})
    dispatch(ctx, "until", {"target": "t3.logits"})
    use = dispatch(ctx, "print", {"address": "ctx1"})["symbol_uses"][0]
    assert use["row"] == 1 and "same tokens" in use["how"]
    dispatch(ctx, "start", {"name": "other3", "prompt": [2, 5, 9, 3], "prefill": 0})
    dispatch(ctx, "until", {"target": "t3.logits"})
    with pytest.raises(Refused, match="absolute row t1"):
        dispatch(ctx, "print", {"address": "ctx1"})
    dispatch(ctx, "frame", {"address": "main:"})


def test_a_no_effect_or_scalar_result_declares_nothing():
    from mdb.ops import named

    dbg, ctx = _session()
    dispatch(ctx, "whatif", {"target": "t1.L1.v[kv0]", "value": "t1.L1.v[kv0]",
                             "consumer": "M[4,9]"})
    with pytest.raises(Refused, match="verdict 'no effect'.*no record is made"):
        dispatch(ctx, "symbol", {"name": "nothing", "result": "whatif"})
    dispatch(ctx, "whatif", {"target": "L1.m", "value": 1.0, "op": "scale",
                             "consumer": "M[4,9]"})
    with pytest.raises(Refused, match="not a transplant"):
        dispatch(ctx, "symbol", {"name": "nothing", "result": "whatif"})
    assert "nothing" not in named.store(dbg)


def test_compare_side_and_each_item_results_declare_too():
    dbg, ctx = _session(with_set=True)
    dispatch(ctx, "compare", parse("compare", "t1.L1.v[kv0]=clean:t1.L1.v[kv0] L1.m *= 1 "
                                              "--consumer M[4,9]"))
    a = dispatch(ctx, "symbol", parse("symbol", "sa = compare.a causal=false"))
    assert a["body"]["evidence"]["op"] == "compare" and a["body"]["row"]["role"] == "subject"
    with pytest.raises(Refused, match="not a transplant"):
        dispatch(ctx, "symbol", parse("symbol", "sb = compare.b"))
    found = dispatch(ctx, "each", {"set": "pairs", "prefill": "@subject", "line":
                                   "whatif t@subject.L1.v[kv0] = @pair:t@subject.L1.v[kv0] "
                                   "--consumer t@last.M[$answer,$foil]"})
    item = dispatch(ctx, "symbol", parse("symbol", f"sc = {found['id']}:c"))
    row = item["body"]["row"]
    assert (row["row"], row["role"], row["role_source"]) == (2, "subject", "set item c")
    assert item["body"]["role"]["readers"]["state"] == "unresolved"   # item run released


def test_the_server_op_gives_the_repl_result(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from mdb.serve.app import create_app
    from mdb.serve.live import LiveSession, open_debugger

    session = LiveSession(open_debugger(tmp_path / "p", tiny="qwen2"))
    try:
        c = TestClient(create_app(session.debugger.project.root, port=8799, live=session),
                       base_url="http://127.0.0.1:8799")
        lines = ["start name=clean prefill=0 1 7 9 3 6 2", "until t5.logits",
                 "start name=main prefill=0 1 5 9 3 6 2", "until t5.logits",
                 "whatif t1.L1.v[kv0] = clean:t1.L1.v[kv0] --consumer M[4,9]",
                 "symbol ctx1 = whatif", "print ctx1"]
        served = []
        for number, line in enumerate(lines):
            verb, _, rest = line.partition(" ")
            response = c.post(f"/api/v1/s/{session.sid}/op",
                              json={"op": verb, "args": parse(verb, rest),
                                    "request_id": f"r{number}"})
            assert response.status_code == 200, response.text
            served.append(response.json()["data"]["result"])
    finally:
        session.close()
    dbg, ctx = _session()
    _, local = _declare(ctx)
    assert served[5]["body"] == local["body"]
    assert served[5]["stored"] == [served[5]["record"]]      # the server holds a project
    assert served[6]["symbol_uses"] == dispatch(ctx, "print", {"address": "ctx1"})["symbol_uses"]

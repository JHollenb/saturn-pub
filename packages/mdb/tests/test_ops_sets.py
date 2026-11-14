"""Prompt sets and ``each`` (M22): one op line over many prompts, one run per prompt.

Roles and variables bind before dispatch (an undefined role refuses that item only); ``each``
over four tiny prompts gives, for ``print``, ``whatif`` and ``reads``, the numbers a user gets
by running each prompt by hand (bit for bit); ``@pair:`` takes a value from the paired item's
clean run as ``ORIGIN:ADDR`` does by hand; the summary's statistics are right and never merge
authority labels; the caller's run, cursor and session digest are unchanged afterwards; the
summary stays under the record bound with 40 items; the REPL grammar (``each SET: LINE``,
``set load``/``set show``).
"""

import json
from pathlib import Path

import pytest

from mdb.analysis import summary as stats
from mdb.core.pointers import MAX_RECORD_BYTES, check_bounded
from mdb.ops import BadArgs, Refused, dispatch
from mdb.ops.lines import parse, route
from mdb.ops.sets import Bound, Unbound, bind_text, make_set

FIXTURE = Path(__file__).parent / "fixtures" / "prompt_set_tiny.jsonl"
WHATIF = "whatif L1.A[h1] *= 0 --consumer M[$answer,$foil]"
READS = "reads t@subject --consumer M[$answer,$foil] top=2"
PRINT = "print t@last.logits[i=$answer]"


class _NoModel:
    tokenizer = None

    def tokens(self, prompt):
        return list(prompt)


# --------------------------------------------------------------------------- binding (no model)


def test_roles_and_vars_bind_before_dispatch_and_an_undefined_role_refuses_one_item():
    found = make_set("s", [
        {"name": "a", "tokens": [1, 5, 9, 3], "roles": {"subject": 1},
         "vars": {"answer": 4, "foil": 9}, "pair": "b"},
        {"name": "b", "tokens": [1, 7, 9, 3, 2], "roles": {"subject": 2},
         "vars": {"answer": 6, "foil": 7}},
        {"name": "c", "tokens": [3, 11], "vars": {"answer": "x"}},
    ])
    a, b, c = (Bound(_NoModel(), item) for item in found.items)
    assert bind_text("t@subject.L1.A[h5]", a) == "t1.L1.A[h5]"
    assert bind_text("t@last.M[$answer,$foil]", a) == "t3.M[4,9]"
    assert bind_text("t@last.M[$answer,$foil]", b) == "t4.M[6,7]"
    assert bind_text("cache.L0.v[row=@subject]", b) == "cache.L0.v[row=2]"
    assert bind_text("@subject", b) == "2"
    assert bind_text("@pair:t@last.L1.h", a, b, "pair-s-b") == "pair-s-b:t4.L1.h"
    assert (bind_text("cache.L0.v[row=@subject] = @pair:cache.L0.v[row=@subject]", a, b, "p")
            == "cache.L0.v[row=1] = p:cache.L0.v[row=2]")
    for text, reason in (("t@subject.L1.h", "no role 'subject'"),
                         ("M[$answer,$foil]", "no tokenizer"),
                         ("M[$nope,1]", "no variable 'nope'"),
                         ("@pair:L1.h", "has no pair")):
        with pytest.raises(Unbound, match=reason):
            bind_text(text, c)
    late = Bound(_NoModel(), make_set("t", [{"name": "x", "tokens": [1, 2],
                                             "roles": {"subject": 5}}]).items[0])
    with pytest.raises(Unbound, match="past the prompt"):
        bind_text("t@subject.L0.h", late)


def test_a_malformed_set_is_refused_whole_naming_the_item_and_key(tmp_path):
    with pytest.raises(BadArgs, match="item 2 \\(b\\): role subject must be a row"):
        make_set("s", [{"name": "a", "tokens": [1]},
                       {"name": "b", "tokens": [1], "roles": {"subject": -1}}])
    with pytest.raises(BadArgs, match="pairs with 'zz'"):
        make_set("s", [{"name": "a", "tokens": [1], "pair": "zz"}])
    with pytest.raises(BadArgs, match="repeats"):
        make_set("s", [{"name": "a", "tokens": [1]}, {"name": "a", "tokens": [2]}])
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"name": "a", "tokens": [1]}\nsecret-looking text\n')
    from mdb.ops.sets import read_jsonl

    with pytest.raises(BadArgs, match="line 2: not JSON") as error:
        read_jsonl(str(bad))
    assert "secret" not in str(error.value)


def test_set_define_routes_and_parses_its_json_with_spaces():
    text = '[{"name": "a", "tokens": [1, 2]}, {"name": "b", "tokens": [3]}]'
    verb, rest = route("set", f"define demo {text}")
    assert verb == "promptset"
    assert parse(verb, rest) == {"action": "define", "name": "demo", "arg": text}


def _row(name, read, mass, verdict="causal"):
    return {"name": name, "state": "resolved", "key": {
        "label": read, "verdict": verdict,
        "columns": {"mass": {"value": mass, "authority": "measured"}}}}


def _each_result(rows):
    return {"set": "s", "line": "reads", "total": len(rows), "at": "t@last.logits",
            "runs": "released", "id": "each1", "items": rows,
            "statistics": {"exclusion": "x", "numeric": [], "verdicts": [],
                           "omitted": {"count": 0}, "partial": {"count": 0}}}


def test_each_table_columns_are_as_wide_as_their_widest_cell_and_keep_addresses_whole():
    from mdb.repl.render import _each_lines

    long = "t21.L24.h40"
    rows = [_row("freezing", long, 0.38281234), _row("b", "t3.L1.h0", 0.9),
            {"name": "c", "state": "refused", "reason": "no role 'subject'"}]
    lines = _each_lines(_each_result(rows))
    table = [line for line in lines if "freezing" in line or line.strip().startswith("item")]
    head, first = table
    assert first.split()[1] == long and first.split()[2] == "0.38281"
    assert head.index("mass") == first.index("0.38281")
    assert [line for line in lines if line.startswith("  b ")][0].index("0.9") == \
        head.index("mass")
    assert any(line.startswith("  c") and "refused: no role" in line for line in lines)
    assert max(len(line) for line in table) <= 80


def test_each_table_shortens_numbers_not_addresses_to_fit_80_columns():
    from mdb.repl.render import _each_lines

    a = "t2121.L124.h4040.somewhere.very.long.address"
    rows = [_row("x" * 12, a, 0.123456789, "a long verdict text")]
    rows[0]["key"]["columns"]["effect"] = {"value": 0.987654321, "authority": "derived-exact"}
    line = [l for l in _each_lines(_each_result(rows)) if l.startswith("  xxxx")][0]
    assert a in line and "0.123456789" not in line and "0.12346" not in line
    assert "0.123" in line


def test_set_load_of_a_big_set_prints_the_count_and_the_first_five():
    from mdb.repl.render import _promptset_lines

    items = [{"name": f"p{i}", "ids": [1, 2], "prompt_tokens": 2, "roles": [], "vars": []}
             for i in range(12)]
    base = {"set": "big", "source": "f", "total": 12, "items": items}
    load = _promptset_lines({**base, "action": "load"})
    assert load[0].startswith("set big: 12 items") and len(load) == 7
    assert load[-1] == "  +7 more (set show big)"
    assert len(_promptset_lines({**base, "action": "show"})) == 13
    small = {**base, "total": 3, "items": items[:3]}
    assert len(_promptset_lines({**small, "action": "load"})) == 4


def test_the_repl_grammar_routes_set_load_and_parses_each():
    assert route("set", "load recall f.jsonl") == ("promptset", "load recall f.jsonl")
    assert route("set", "L1.m = 0 scale") == ("set", "L1.m = 0 scale")
    assert parse("promptset", "load recall f.jsonl") == {
        "action": "load", "name": "recall", "arg": "f.jsonl"}
    assert parse("each", f"recall at=t@last.L1.p keep: {WHATIF}") == {
        "set": "recall", "line": WHATIF, "at": "t@last.L1.p", "keep": True}
    with pytest.raises(BadArgs, match="each SET"):
        parse("each", "recall whatif L1.m = 0")


# --------------------------------------------------------------------------- statistics


def _claim(value, authority):
    return {"value": value, "authority": authority}


def test_summary_statistics_are_exact_and_mixed_labels_are_never_merged():
    results = [
        {"d": _claim(-1.0, "measured"), "m": _claim(1.0, "measured"),
         "effect": {"verdict": "effect"}, "only": _claim(3.0, "measured")},
        {"d": _claim(-2.0, "measured"), "m": _claim(2.0, "derived-exact"),
         "effect": {"verdict": "effect"}},
        {"d": _claim(0.5, "measured"), "m": _claim(4.0, "measured"),
         "effect": {"verdict": "no effect above the noise floor"}},
    ]
    out = stats.summarize(results)
    d, m = out["numeric"]
    assert d == {"path": "d", "n": 3, "mean": _claim(-2.5 / 3, "measured"),
                 "min": _claim(-2.0, "measured"), "max": _claim(0.5, "measured"),
                 "agree": {"sign": "-", "count": 2}}
    assert m["mixed"] is True and "mean" not in m and m["n"] == 3
    assert m["by_authority"]["measured"]["n"] == 2
    assert m["by_authority"]["measured"]["mean"] == _claim(2.5, "measured")
    assert m["by_authority"]["derived-exact"]["mean"] == _claim(2.0, "derived-exact")
    assert out["verdicts"] == [{"path": "effect.verdict", "counts": [
        {"label": "effect", "count": 2}, {"label": "no effect above the noise floor",
                                          "count": 1}]}]
    assert out["partial"] == 1   # "only" is in one item: counted, not summarised


def test_table_columns_follow_the_result_shape():
    margin = {"margin": {"value": _claim(0.5, "measured")}}
    value = {"stats": {"mean": _claim(0.25, "measured"), "abs_max": _claim(0.25, "measured")}}
    assert stats.key_columns("print", margin)["columns"] == {"margin": _claim(0.5, "measured")}
    assert list(stats.key_columns("print", value)["columns"]) == ["mean", "abs_max"]
    other = stats.key_columns("frobnicate", {"x": _claim(1.0, "bound"), "v": {"verdict": "ok"}})
    assert other["columns"] == {"x": _claim(1.0, "bound")} and other["verdict"] == "ok"


# --------------------------------------------------------------------------- live (tiny model)


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from mdb.ops.debugger import open_tiny

    debugger = open_tiny("qwen2", "bf16")
    dispatch(debugger.ops(), "promptset", {"action": "load", "name": "recall",
                                            "arg": str(FIXTURE)})
    return debugger


def _by_hand(debugger, tokens, name, args, *, donor=None):
    ctx = debugger.ops()
    if donor is not None:
        dispatch(ctx, "start", {"prompt": donor, "name": "clean"})
        dispatch(ctx, "until", {"target": f"t{len(donor) - 1}.logits"})
    dispatch(ctx, "start", {"prompt": tokens, "name": "byhand"})
    dispatch(ctx, "until", {"target": f"t{len(tokens) - 1}.logits"})
    return dispatch(ctx, name, args)


ITEMS = {"a": ([1, 5, 9, 3], 1, 4, 9), "b": ([1, 7, 9, 3], 1, 4, 9),
         "c": ([2, 6, 10, 4, 8], 2, 6, 7), "d": ([3, 11, 5], None, 4, 9)}


@pytest.mark.parametrize("line", [PRINT, WHATIF, READS], ids=["print", "whatif", "reads"])
def test_each_equals_running_every_prompt_by_hand_bit_for_bit(tiny, line):
    ctx = tiny.ops()
    out = tiny.each("recall", line, ctx=ctx)
    for name, (tokens, subject, answer, foil) in ITEMS.items():
        last = len(tokens) - 1
        if line == PRINT:
            hand = _by_hand(tiny, tokens, "print", {"address": f"t{last}.logits[i={answer}]"})
        elif line == WHATIF:
            hand = _by_hand(tiny, tokens, "whatif", {"target": "L1.A[h1]", "value": 0.0,
                                                     "op": "scale",
                                                     "consumer": f"M[{answer},{foil}]"})
        elif subject is None:
            row = next(r for r in out.summary["items"] if r["name"] == name)
            assert row["state"] == "refused" and "no role 'subject'" in row["reason"]
            assert name not in out.results
            continue
        else:
            hand = _by_hand(tiny, tokens, "reads", {"source": f"t{subject}", "top": 2,
                                                    "consumer": f"M[{answer},{foil}]"})
        mine, theirs = stats.claims(out.results[name]), stats.claims(hand)
        assert mine and mine == theirs, name           # every labelled number, exactly
        assert stats.verdicts(out.results[name]) == stats.verdicts(hand)
    refused = [r for r in out.summary["items"] if r["state"] == "refused"]
    assert out.summary["statistics"]["n"] == 4 - len(refused)


def test_pair_takes_the_value_from_the_paired_items_clean_run_as_by_hand(tiny):
    line = ("whatif cache.L0.v[row=@subject] = @pair:cache.L0.v[row=@subject] "
            "--consumer M[$answer,$foil]")
    out = tiny.each("recall", line)
    row = {r["name"]: r for r in out.summary["items"]}
    assert row["d"]["state"] == "refused"                     # no role, and no pair
    assert row["a"]["bound"]["value"] == "pair-recall-b:cache.L0.v[row=1]"
    hand = _by_hand(tiny, ITEMS["a"][0], "whatif", {
        "target": "cache.L0.v[row=1]", "value": "clean:cache.L0.v[row=1]",
        "consumer": "M[4,9]"}, donor=ITEMS["b"][0])
    assert stats.claims(out.results["a"]) == stats.claims(hand)
    assert not any(name.startswith("pair-") for name in tiny.runs)   # donors released


def test_pair_address_without_a_row_is_the_paired_items_last_row_for_other_lengths(tiny):
    items = [{"name": "p", "tokens": [1, 5, 9, 3], "roles": {"subject": 1},
              "vars": {"answer": 4, "foil": 9}, "pair": "q"},
             {"name": "q", "tokens": [1, 7, 9, 3, 2], "roles": {"subject": 2},
              "vars": {"answer": 4, "foil": 9}}]
    port = "whatif L1.A[h1] = @pair:L1.A[h1] --consumer M[$answer,$foil]"
    out = tiny.each(items, port)
    row = {r["name"]: r for r in out.summary["items"]}["p"]
    assert row["state"] != "refused", row
    assert row["bound"]["value"] == "pair-tmp-q:t4.L1.A[h1]" or row["bound"]["value"].endswith(
        ":t4.L1.A[h1]")
    assert row["donor"] == [row["bound"]["value"]]
    hand = _by_hand(tiny, items[0]["tokens"], "whatif", {
        "target": "L1.A[h1]", "value": "clean:t4.L1.A[h1]", "consumer": "M[4,9]"},
        donor=items[1]["tokens"])
    assert stats.claims(out.results["p"]) == stats.claims(hand)
    cell = "whatif cache.L0.v[row=@subject] = @pair:cache.L0.v[row=@subject] " \
           "--consumer M[$answer,$foil]"
    out = tiny.each(items, cell)
    row = {r["name"]: r for r in out.summary["items"]}["p"]
    assert row["state"] != "refused", row
    assert row["bound"]["value"].endswith(":cache.L0.v[row=2]")
    hand = _by_hand(tiny, items[0]["tokens"], "whatif", {
        "target": "cache.L0.v[row=1]", "value": "clean:cache.L0.v[row=2]",
        "consumer": "M[4,9]"}, donor=items[1]["tokens"])
    assert stats.claims(out.results["p"]) == stats.claims(hand)


def test_pair_address_binding_rows():
    found = make_set("s", [{"name": "a", "tokens": [1, 5, 9, 3], "pair": "b"},
                           {"name": "b", "tokens": [1, 7, 9, 3, 2], "roles": {"subject": 2}}])
    a, b = (Bound(_NoModel(), item) for item in found.items)
    assert bind_text("@pair:L20.A[h5]", a, b, "p") == "p:t4.L20.A[h5]"
    assert bind_text("@pair:t@subject.L20.A[h5]", a, b, "p") == "p:t2.L20.A[h5]"
    assert bind_text("@pair:t1.L20.A", a, b, "p") == "p:t1.L20.A"


def test_the_callers_run_cursor_and_digest_are_unchanged_and_keep_holds_runs(tiny):
    from mdb.ops.predict import session_digest

    ctx = tiny.ops()
    dispatch(ctx, "start", {"prompt": [1, 5, 9, 3]})
    dispatch(ctx, "until", {"target": "L1.p"})
    dispatch(ctx, "break", {"pattern": "L*.o"})
    run, where, digest = ctx.run, ctx.run.where, session_digest(ctx)
    runs, live = dict(tiny.runs), tiny.live
    result = dispatch(ctx, "each", {"set": "recall", "line": f"{WHATIF}; {PRINT}"})
    assert result["op"] == "each" and result["verb"] == "print"
    assert ctx.run is run and run.where == where and session_digest(ctx) == digest
    assert tiny.runs == runs and tiny.live is live
    dispatch(ctx, "each", {"set": "recall", "line": PRINT, "keep": True})
    assert {"recall-a", "recall-b", "recall-c", "recall-d"} <= set(tiny.runs)
    assert ctx.run is run and session_digest(ctx) == digest
    kept = tiny.runs["recall-c"]
    assert kept.where == "t4.logits"
    for name in ("recall-a", "recall-b", "recall-c", "recall-d"):
        tiny.runs.pop(name)


def test_a_bad_line_or_set_refuses_the_whole_each(tiny):
    ctx = tiny.ops()
    with pytest.raises(BadArgs, match="cannot be one of its lines"):
        dispatch(ctx, "each", {"set": "recall", "line": "start 1 2 3"})
    with pytest.raises(BadArgs):
        dispatch(ctx, "each", {"set": "recall", "line": "whatif L1.m"})
    with pytest.raises(Refused, match="no prompt set 'nope'"):
        dispatch(ctx, "each", {"set": "nope", "line": PRINT})


def test_the_summary_record_is_bounded_with_40_items(tiny):
    rows = [{"name": f"p{i}", "tokens": [1 + i % 7, 5, 9 + i % 3, 3], "roles": {"subject": 1},
             "vars": {"answer": 4, "foil": 9}} for i in range(40)]
    out = tiny.each(rows, READS)
    summary = out.summary
    assert len(summary["items"]) == 40 and summary["total"] == 40
    assert check_bounded(summary, "each") == summary
    import json

    assert len(json.dumps(summary, sort_keys=True, separators=(",", ":"))) <= MAX_RECORD_BYTES
    assert all("result" in row and len(row["result"]) == 64 for row in summary["items"])
    assert len(out.results) == 40                 # full results by pointer, not inline
    assert "reads" not in json.dumps(summary["items"][0]["key"])


def test_with_a_project_the_summary_is_stored_and_without_one_nothing(tmp_path):
    pytest.importorskip("torch")
    from mdb.core import index
    from mdb.ops.debugger import open_tiny

    debugger = open_tiny("qwen2", "bf16", project=tmp_path / "p")
    out = debugger.each(str(FIXTURE), PRINT)
    stored = out.summary["stored"]
    assert isinstance(stored, str)
    record = index.get_record(debugger.project, debugger.session_id, stored)
    assert record["kind"] == "prompt-set" and record["body"]["set"] == "file"


# --------------------------------------------------------------------------- durable results (M48)


def _tamper(project_root, ref):
    path = Path(project_root) / "store" / "tensors" / f"{ref}.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 0x01
    path.write_bytes(bytes(raw))


def test_item_results_are_blobs_read_back_in_a_new_debugger_and_a_tampered_one_is_refused(
        tmp_path):
    pytest.importorskip("torch")
    from mdb.ops.debugger import open_tiny

    debugger = open_tiny("qwen2", "bf16", project=tmp_path / "p")
    out = debugger.each(str(FIXTURE), WHATIF)
    stored = out.summary["stored"]
    assert out.results
    for row in out.summary["items"]:
        if row["state"] != "refused":
            assert row["blob"]["state"] == "stored" and row["blob"]["sha256"]
    # a new debugger on the same project (a new process): memory is empty, the blob answers
    fresh = open_tiny("qwen2", "bf16", project=tmp_path / "p")
    assert fresh.session_id == debugger.session_id
    name = next(iter(out.results))
    read = dispatch(fresh.ops(), "promptset", {"action": "result", "name": stored, "arg": name})
    assert read["result"] == out.results[name] and read["from"] == "stored blob"
    # the old process-local id is gone with the process
    with pytest.raises(Refused, match="no each result"):
        dispatch(fresh.ops(), "promptset", {"action": "result", "name": "each1", "arg": name})
    ref = next(r for r in out.summary["items"] if r["name"] == name)["blob"]["ref"]
    _tamper(tmp_path / "p", ref)
    with pytest.raises(Refused, match="refused"):
        dispatch(open_tiny("qwen2", "bf16", project=tmp_path / "p").ops(), "promptset",
                 {"action": "result", "name": stored, "arg": name})


def test_without_a_project_the_row_says_held_in_memory_only(tiny):
    out = tiny.each("recall", PRINT)
    blobs = [row["blob"] for row in out.summary["items"] if row["state"] != "refused"]
    assert blobs and all(b["state"] == "absent" and "memory only" in b["reason"] for b in blobs)


def test_the_summary_with_blob_pointers_stays_under_64_kib_at_40_items(tmp_path):
    pytest.importorskip("torch")
    import json

    from mdb.core import index
    from mdb.ops.debugger import open_tiny

    rows = [{"name": f"p{i}", "tokens": [1 + i % 7, 5, 9 + i % 3, 3], "roles": {"subject": 1},
             "vars": {"answer": 4, "foil": 9}} for i in range(40)]
    debugger = open_tiny("qwen2", "bf16", project=tmp_path / "p")
    out = debugger.each(rows, READS)
    assert len(json.dumps(out.summary, sort_keys=True, separators=(",", ":"))) <= MAX_RECORD_BYTES
    assert all(r["blob"]["state"] == "stored" for r in out.summary["items"])
    assert isinstance(out.summary["stored"], str)      # the sealed record accepted the pointers
    index.get_record(debugger.project, debugger.session_id, out.summary["stored"])


def test_promptset_define_takes_items_inline(tiny):
    items = [{"name": "a", "tokens": [1, 5, 9, 3]}, {"name": "b", "tokens": [2, 6, 10]}]
    for text in (json.dumps(items), "\n".join(json.dumps(i) for i in items)):
        out = dispatch(tiny.ops(), "promptset", {"action": "define", "name": "inl", "arg": text})
        assert out["total"] == 2 and out["source"] == "inline"
    with pytest.raises(BadArgs, match="ITEMS is a JSON list"):
        dispatch(tiny.ops(), "promptset", {"action": "define", "name": "inl"})

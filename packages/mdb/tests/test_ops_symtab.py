"""The symbol-table options reachable from op lines (review 2026-11-14): ``candidates`` declares
sets, ``symbols sets= truth= content_layers= co= switch= lexical_null= hops= sink= labels=
derive= foreign= causal=`` reaches every new column, and ``each`` runs over a set past 64 items
(rows past 64 in a blob) and in shards.
"""

import json

import pytest

pytest.importorskip("torch")

from mdb.core.pointers import MAX_RECORD_BYTES, check_bounded  # noqa: E402
from mdb.ops import BadArgs, Refused, dispatch  # noqa: E402
from mdb.ops.lines import parse  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]


@pytest.fixture(scope="module")
def tiny():
    return open_tiny("qwen3", "fp32")


def _line(ctx, text):
    verb, _, rest = text.partition(" ")
    return dispatch(ctx, verb, parse(verb, rest))


def test_candidates_line_grammar():
    assert parse("candidates", "countries = #4,#6,#8 rule=single") == {
        "name": "countries", "source": "#4,#6,#8", "rule": "single"}
    assert parse("candidates", "capitals = #2,#10,#3 paired=countries")["paired"] == "countries"
    assert parse("candidates", "show countries") == {"action": "show", "name": "countries"}
    assert parse("candidates", "list") == {"action": "list"}
    with pytest.raises(BadArgs):
        parse("candidates", "oops")


def test_symbols_options_reach_every_new_column(tiny):
    ctx = tiny.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "until", {"target": "t5.logits"})
    out = _line(ctx, "candidates countries = #4,#6,#8")
    assert out["size"] == 3 and len(out["sha256"]) == 64
    _line(ctx, "candidates capitals = #2,#10,#3 paired=countries")
    line = ("symbols consumer=M[4,9] sets=countries,capitals truth=countries:6,capitals:10 "
            "content_layers=true lexical_null=true co=capitals/countries switch=capitals:countries"
            " hops=t1:-2.0:-0.5 sink=0 labels=family:city,depth:2 "
            "derive=dec:content.countries.best foreign=true causal=true")
    report = _line(ctx, line)
    check_bounded(report)
    t1 = report["bindings"][1]
    content = t1["content"]
    assert content["countries"]["truth"] == 6 and "flags" in content["countries"]
    assert "lexical_null" in content["capitals"] and "countries" in content["capitals"]["co"]
    assert content["capitals"]["switched"]["on"] == "countries"
    assert "countries" in t1["content_lifetime"]
    assert t1["hop"]["indirect_share"]["value"] == pytest.approx(0.75)
    assert "mass_nonsink" in t1 and "contrast" in t1 and "mask_delta" in t1
    assert t1["labels"]["family"] == {"label": "city", "source": "supplied"}
    assert t1["labels"]["depth"]["value"] == 2
    assert t1["labels"]["dec"]["value"] == content["countries"]["best"]
    assert t1["foreign"]["cells"][0]["nnsight"]["index"] == "[:, 1, :]"
    assert t1["causal"]["status"] == "intervened"
    assert report["candidate_sets"][0]["sha256"]
    assert isinstance(report["records"][0], str)
    from mdb.repl.render import _symbols_lines

    text = "\n".join(_symbols_lines(report))
    assert "    causal " in text and "    content countries:" in text and "labels family=city" in text
    off = _line(ctx, "symbols consumer=M[4,9] sink=none")
    assert "mass_nonsink" not in off["bindings"][1]
    assert off["bindings"][1]["causal"]["status"] == "unknown"
    with pytest.raises(Refused, match="no candidate set"):
        _line(ctx, "symbols sets=nope")
    with pytest.raises(BadArgs, match="need sets"):
        _line(ctx, "symbols truth=countries:6")
    shown = _line(ctx, "candidates show countries")
    assert shown["words"][0] == {"word": "#4", "pieces": [4]}
    assert {s["name"] for s in _line(ctx, "candidates list")["sets"]} == {"countries",
                                                                           "capitals"}


def test_each_runs_past_64_items_and_in_shards(tiny, tmp_path):
    rows = [{"name": f"p{i}", "tokens": [1 + i % 7, 5, 9 + i % 3, 3], "vars": {"answer": 4}}
            for i in range(70)]
    out = tiny.each(rows, "print t@last.logits[i=$answer]")
    summary = out.summary
    assert summary["total"] == 70 and len(summary["items"]) == 64
    assert summary["items_omitted"] == 6 and summary["items_blob"]["state"] == "absent"
    assert len(out.results) == 70 and summary["statistics"]["n"] == 70
    assert len(json.dumps(summary, separators=(",", ":"))) <= MAX_RECORD_BYTES
    from mdb.ops.sets import load_item_result, parse_shard, shard_items, make_set

    assert parse_shard("2/3") == (2, 3)
    with pytest.raises(BadArgs):
        parse_shard("4/3")
    items = make_set("s", rows).items
    parts = [shard_items(items, (k, 3)) for k in (1, 2, 3)]
    assert sum(len(p) for p in parts) == 70 and parts[2][-1].name == "p69"
    with pytest.raises(Refused):            # held in memory only: no stored blob for the item
        load_item_result(tiny, summary, "p69")
    shard = tiny.each(rows, "print t@last.logits[i=$answer]", shard="2/3")
    assert shard.summary["shard"] == {"k": 2, "n": 3, "items": 24, "set_total": 70}
    assert shard.summary["total"] == 24
    stored = open_tiny("qwen3", "fp32", project=tmp_path / "p")
    big = stored.each(rows, "print t@last.logits[i=$answer]")
    assert big.summary["items_blob"]["state"] == "stored"
    assert isinstance(big.summary["stored"], str)
    assert load_item_result(stored, big.summary, "p69") == big.results["p69"]

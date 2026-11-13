"""The A6 verbs through ``dispatch``: ``reads``, ``symbols``, ``symbol``, ``circuit``,
``compare`` and ``seek`` on tiny Qwen2 (BF16).

Every number of every result carries one valid label (structural integers excepted) and every
result fits the bounds; ``compare`` leaves the debugger session digest unchanged; ``seek``
moves forward, backward, and by a stored recipe (a durable checkpoint name, an exported
occurrence pointer, another session's id) and refuses what it cannot reach. The shell's
parsers for the verbs are checked against the JSON arguments they produce.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from test_ops_predict import unlabelled  # noqa: E402

import mdb  # noqa: E402
from mdb.core.pointers import check_bounded  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops import BadArgs, Refused, dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.ops.predict import _session_digest  # noqa: E402
from mdb.repl.shell import parse  # noqa: E402

PROMPT = [1, 5, 9, 3]


@pytest.fixture(scope="module")
def debugger():
    return open_tiny("qwen2", "bf16")


def _ctx(debugger, where="t3.logits"):
    run = debugger.run(PROMPT)
    run.until(where)
    return debugger.ops(run)


def _clean(result):
    assert unlabelled(result) == []
    check_bounded(result)
    return result


def test_every_verb_returns_labelled_bounded_results(debugger):
    ctx = _ctx(debugger)
    reads = _clean(dispatch(ctx, "reads", {"source": "t1", "consumer": "M[4,9]", "top": 3}))
    assert len(reads["reads"]) == 3 and reads["omitted"]["count"] == 1
    assert {e["causal"]["verdict"] for e in reads["reads"]} <= {"causal",
                                                                 "not causal at this dose"}
    declared = _clean(dispatch(ctx, "symbol", {"name": "ctx", "addresses": "t1,t2"}))
    assert declared["symbol"] == "ctx" and declared["rows"] == [1, 2]
    table = _clean(dispatch(ctx, "symbols", {"consumer": "M[4,9]"}))
    assert table["bindings"][0]["symbol"] == "ctx" and "stored" not in table
    built = _clean(dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 2}))
    tested = _clean(dispatch(ctx, "circuit", {"target": "test", "id": built["id"]}))
    assert tested["test"]["verdict"] in ("sufficient at this dose", "not sufficient at this dose")
    shown = _clean(dispatch(ctx, "circuit", {"target": "show", "id": built["id"]}))
    assert shown["record"]["body"]["root"]["name"] == built["record"]["body"]["root"]["name"]
    before = _session_digest(ctx)
    compared = _clean(dispatch(ctx, "compare", {"a": "L1.m=0.5 add", "b": "L0.v=0.25 add",
                                                "consumer": "M[4,9]"}))
    assert _session_digest(ctx) == before
    assert compared["session"]["debugger_before"] == compared["session"]["debugger_after"]
    assert set(compared["measured"]) == {"base", "A", "B", "A+B"}
    with pytest.raises(Refused, match="has not been computed"):
        dispatch(ctx, "symbol", {"name": "late", "addresses": "t4"})
    with pytest.raises(BadArgs):
        dispatch(ctx, "compare", {"a": "L1.m=0.5 shift", "b": "L0.v=1"})
    with pytest.raises(Refused, match="no checkpoint 'nope'"):
        dispatch(ctx, "compare", {"a": "nope", "b": "L0.v=1"})


def test_compare_against_an_in_memory_checkpoint(debugger):
    ctx = _ctx(debugger, "t3.L1.h")
    dispatch(ctx, "set", {"target": "L1.m", "value": 0.5, "op": "add"})
    dispatch(ctx, "checkpoint", {"name": "edited"})
    dispatch(ctx, "seek", {"target": "t3.L0.h"})
    result = _clean(dispatch(ctx, "compare", {"a": "edited", "b": "L0.v=0.25 add",
                                              "consumer": "M[4,9]"}))
    assert result["interaction"]["state"] == "refused"
    assert result["where"] == "t3.L0.h"


def test_seek_forward_and_backward_within_the_run(debugger):
    ctx = _ctx(debugger, "t3.L0.h")
    ahead = _clean(dispatch(ctx, "seek", {"target": "t3.L1.p"}))
    assert ahead["via"] == "run" and ctx.run.where == "t3.L1.p"
    back = _clean(dispatch(ctx, "seek", {"target": "t3.L0.v"}))
    assert back["via"] == "run" and ctx.run.where == "t3.L0.v"
    assert any("reverse-step" in move for move in back["moves"])
    with pytest.raises(Refused, match="only through a stored recipe"):
        dispatch(ctx, "seek", {"target": "t1.L0.p"})
    assert ctx.run.where == "t3.L0.v"
    with pytest.raises(Refused, match="needs a stored recipe"):
        dispatch(ctx, "seek", {"target": "other:t3.L0.p"})
    with pytest.raises(Refused, match="no selector"):
        dispatch(ctx, "seek", {"target": "t3.L0.p[h0]"})


def test_seek_by_stored_recipe(tmp_path):
    project = tmp_path / "p"
    dbg = mdb.attach(tiny_model("qwen2", "bf16"), project=project)
    run = dbg.run(PROMPT)
    run.until("t3.L1.h")
    ctx = dbg.ops(run)
    dispatch(ctx, "set", {"target": "L1.m", "value": 0.5, "op": "add"})
    dispatch(ctx, "checkpoint", {"name": "pre"})
    pointer = dbg.export(run, "t3.L1.h", keep=True)
    dispatch(ctx, "until", {"target": "t3.logits"})
    edited = run.read("t3.logits").clone()
    # a durable checkpoint name as the origin: the recipe replays, then the cursor moves on
    named = _clean(dispatch(ctx, "seek", {"target": "pre:t3.logits"}))
    assert named["via"] == "recipe" and ctx.run is not run
    assert torch.equal(ctx.run.read("t3.logits"), edited)
    # an exported occurrence pointer
    by_pointer = _clean(dispatch(ctx, "seek", {"target": json.dumps(pointer)}))
    assert by_pointer["recipe"] == pointer["recipe"] and ctx.run.where == "t3.L1.h"
    # the session id as the origin, from a run that prefilled row 3: the recipe of the indexed
    # occurrence, else of the session cursor (the checkpoint ``pre``)
    longer = dbg.ops(dbg.run([*PROMPT, 7]))
    by_session = _clean(dispatch(longer, "seek", {"target": f"{dbg.session_id}:t3.L1.h"}))
    assert by_session["via"] == "recipe" and by_session["recipe"] == pointer["recipe"]
    assert longer.run.where == "t3.L1.h"
    longer = dbg.ops(dbg.run([*PROMPT, 7]))
    cursor = _clean(dispatch(longer, "seek", {"target": f"{dbg.session_id}:t3.L1.p"}))
    assert cursor["recipe"] == dbg.checkpoints()[0]["recipe"] and longer.run.where == "t3.L1.p"
    with pytest.raises(Refused, match="names no durable checkpoint"):
        dispatch(ctx, "seek", {"target": "nobody:t3.L0.p"})
    # the stored recipe reaches t3.L1.h; t1 was prefilled before it
    with pytest.raises(Refused, match="but not t1.L0.p"):
        dispatch(ctx, "seek", {"target": "pre:t1.L0.p"})
    # a recipe of another model is refused, not replayed
    qwen3 = mdb.attach(tiny_model("qwen3", "bf16"), project=project)
    foreign = qwen3.run(PROMPT)
    foreign.until("t3.L0.h")
    assert qwen3.session_id != dbg.session_id
    for target in ("pre:t3.L1.h", f"{dbg.session_id}:t3.L1.h"):
        with pytest.raises(Refused, match="cannot be replayed here"):
            dispatch(qwen3.ops(foreign), "seek", {"target": target})


def test_shell_parsers():
    assert parse("compare", "L1.m = 0.5 add L0.v=0.25 --consumer M[4,9]") == {
        "a": "L1.m=0.5 add", "b": "L0.v=0.25", "consumer": "M[4,9]"}
    assert parse("compare", "here L1.m=0.5 scale") == {"a": "here", "b": "L1.m=0.5 scale"}
    with pytest.raises(BadArgs):
        parse("compare", "L1.m=0.5")
    assert parse("symbol", "ctx = t1, cache.L1.v[row=2]") == {
        "name": "ctx", "addresses": "t1, cache.L1.v[row=2]"}
    with pytest.raises(BadArgs):
        parse("symbol", "ctx")
    assert parse("reads", "t1 --consumer=M[4,9] top=3") == parse("reads",
                                                                  "t1 top=3 consumer=M[4,9]")
    assert parse("circuit", "M[4,9] top=2 depth=3") == {"target": "M[4,9]", "top": 2,
                                                         "depth": 3}
    assert parse("symbols", "--consumer M[4,9]") == {"consumer": "M[4,9]"}

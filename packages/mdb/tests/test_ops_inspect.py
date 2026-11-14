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
    assert {e["causal"]["verdict"] for e in reads["reads"]} <= {
        "causal", "not causal at this dose", "unresolved"}
    declared = _clean(dispatch(ctx, "symbol", {"name": "ctx", "addresses": "t1,t2"}))
    assert declared["symbol"] == "ctx" and declared["rows"] == [1, 2]
    table = _clean(dispatch(ctx, "symbols", {"consumer": "M[4,9]"}))
    assert table["bindings"][0]["symbol"] == "ctx" and "stored" not in table
    built = _clean(dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 2}))
    with pytest.raises(Refused, match="circuit test is disabled.*issue M4"):
        dispatch(ctx, "circuit", {"target": "test", "id": built["id"]})
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
    # without a row the target is on the cursor's row, as for `until`
    near = _clean(dispatch(ctx, "seek", {"target": "L1.p"}))
    assert near["target"] == "t3.L1.p" and ctx.run.where == "t3.L1.p"
    assert _clean(dispatch(ctx, "seek", {"target": "L0.q"}))["target"] == "t3.L0.q"
    assert _clean(dispatch(ctx, "seek", {"target": "logits"}))["target"] == "t3.logits"
    with pytest.raises(BadArgs, match="bad address 'p'"):
        dispatch(ctx, "seek", {"target": "p"})
    with pytest.raises(Refused, match="no selector"):
        dispatch(ctx, "seek", {"target": "L0.p[h0]"})


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
    with pytest.raises(Refused, match="names no run .*no durable checkpoint"):
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


# --- an interrupt anywhere inside a verb leaves the session as it was (M1) ---------------------

INJECT = [
    ("reads", "t3.logits", "restore", {"source": "t1", "consumer": "M[4,9]", "top": 2}),
    ("compare", "t3.logits", "restore", {"a": "L1.m=0.5 add", "b": "L0.v=0.25 add",
                                         "consumer": "M[4,9]"}),
    ("seek", "t3.L0.q", "continue_", {"target": "t3.L1.m"}),
]


@pytest.mark.parametrize("case", INJECT, ids=lambda c: c[0])
def test_inject_an_interrupt_at_each_native_call_and_the_session_is_unchanged(debugger, case,
                                                                              monkeypatch):
    from saturn_pub.core import Session

    from mdb.analysis.propagate import state_digest

    verb, where, method, args = case
    original = getattr(Session, method)
    fired_any = False
    for nth in range(1, 25):
        ctx = _ctx(debugger, where)
        before, digest = ctx.run.checkpoint(), state_digest(ctx.run)
        calls = {"n": 0}

        def wrapped(self, *a, _nth=nth, _calls=calls, **k):
            _calls["n"] += 1
            out = original(self, *a, **k)
            if _calls["n"] == _nth:
                raise KeyboardInterrupt
            return out

        monkeypatch.setattr(Session, method, wrapped)
        try:
            dispatch(ctx, verb, args)
            fired = False
        except KeyboardInterrupt:
            fired = True
        monkeypatch.undo()
        if not fired:
            break  # past the verb's last such call: it completed
        fired_any = True
        now = ctx.run.checkpoint()
        assert state_digest(ctx.run) == digest, (verb, nth)
        for key in ("where", "cut", "edits", "path_edits"):
            assert now[key] == before[key], (verb, nth, key)
    assert fired_any


# --- the fix-round behaviours through dispatch ---------------------------------------------------


def test_consumer_ids_outside_the_vocabulary_are_refused(debugger):
    ctx = _ctx(debugger)
    for verb, args in (("reads", {"source": "t1"}), ("symbols", {}), ("circuit", {}),
                       ("compare", {"a": "L1.m=0.5 add", "b": "L0.v=1 add"})):
        for consumer in ("M[0,99999]", "logits[999]"):
            key = "target" if verb == "circuit" else "consumer"
            with pytest.raises(Refused, match=r"outside the vocabulary \(valid ids 0\.\.31\)"):
                dispatch(ctx, verb, {**args, key: consumer})


def test_seek_lists_the_edits_it_dropped_or_restaged(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.q")
    run.set("t3.L1.v", 0.5, "add")
    ctx = debugger.ops(run)
    back = _clean(dispatch(ctx, "seek", {"target": "t3.L0.p"}))
    assert back["dropped_edits"] == ["t3.L1.v add 0.5"] and back["restaged_edits"] == []
    run = debugger.run(PROMPT)
    run.until("t3.L0.h")
    run.set("t3.L0.v", 0.25, "add")
    run.until("t3.logits")
    ctx = debugger.ops(run)
    restaged = _clean(dispatch(ctx, "seek", {"target": "t3.L0.p"}))
    assert restaged["restaged_edits"] == ["t3.L0.v add 0.25"]
    assert restaged["dropped_edits"] == []


def test_declared_symbols_and_circuits_outlive_the_run(tmp_path):
    dbg = mdb.attach(tiny_model("qwen2", "bf16"), project=tmp_path / "p")
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "until", {"target": "t3.logits"})
    dispatch(ctx, "symbol", {"name": "ctx", "addresses": "t1"})
    dispatch(ctx, "symbol", {"name": "late", "addresses": "t3.L1.v"})
    built = dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 1})
    dispatch(ctx, "checkpoint", {"name": "here"})
    first = ctx.run
    # a recipe seek replays into a new run: the ledger and the symbols come along
    _clean(dispatch(ctx, "seek", {"target": "here:t3.logits"}))
    assert ctx.run is not first
    shown = _clean(dispatch(ctx, "circuit", {"target": "show", "id": built["id"]}))
    assert shown["record"]["id"] == built["record"]["id"]
    table = _clean(dispatch(ctx, "symbols", {}))
    assert [b["symbol"] for b in table["bindings"]][:2] == ["ctx", "late"]
    # after start the cursor is at t3.L0.h: t3.L1.v is not computed, so `late` is unresolved
    dispatch(ctx, "start", {"prompt": PROMPT})
    table = _clean(dispatch(ctx, "symbols", {}))
    pending = {item["symbol"]: item for item in table["unresolved_symbols"]}
    assert list(pending) == ["late"] and pending["late"]["status"] == "unresolved"
    assert "has not been computed" in pending["late"]["reason"]
    assert table["bindings"][0]["symbol"] == "ctx"


def test_circuit_test_op_is_refused_before_any_work(debugger):
    ctx = _ctx(debugger)
    built = dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 3})
    before = _session_digest(ctx)
    for ident in (built["id"], 999, None):  # refused first, even for an unknown id
        with pytest.raises(Refused, match=r"circuit test is disabled: zero-ablating the "
                                          r"complement says nothing on a real model \(issue M4\)"):
            dispatch(ctx, "circuit", {"target": "test", "id": ident})
    assert _session_digest(ctx) == before
    assert not circuits_tests(ctx, built["id"])


def circuits_tests(ctx, ident):
    from mdb.analysis import circuits

    return circuits.ledger(ctx.run)[ident].tests


def test_circuit_show_marks_what_the_last_test_ablated(debugger):
    ctx = _ctx(debugger)
    from mdb.analysis import circuits

    built = dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 3})
    # the op is disabled (M4); the analysis function still records tests that show reads back
    tested = {"test": circuits.test(ctx.run, circuits.ledger(ctx.run)[built["id"]])}
    shown = _clean(dispatch(ctx, "circuit", {"target": "show", "id": built["id"]}))
    assert shown["ablated"]["retained"] == tested["test"]["retained"]["nodes"]
    ids = set()
    stack = [shown["record"]["body"]["root"]]
    while stack:
        node = stack.pop()
        ids.add(node["id"])
        stack += node.get("children", [])
    assert set(shown["ablated"]["retained"]) | set(shown["ablated"]["complement"]) <= ids


def test_circuits_survive_start_and_are_marked_as_from_a_previous_run():
    ctx = open_tiny("qwen2", "bf16").ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "until", {"target": "t3.logits"})
    first = dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 1})
    own = _clean(dispatch(ctx, "circuit", {"target": "show", "id": first["id"]}))
    assert "from_previous_run" not in own
    before = ctx.run
    dispatch(ctx, "start", {"prompt": [1, 7, 9, 3]})
    assert ctx.run is not before
    dispatch(ctx, "until", {"target": "t3.logits"})
    shown = _clean(dispatch(ctx, "circuit", {"target": "show", "id": first["id"]}))
    assert shown["record"]["id"] == first["record"]["id"]
    assert shown["from_previous_run"] is True and "token_text" not in shown
    second = dispatch(ctx, "circuit", {"target": "M[4,9]", "top": 2, "depth": 1})
    assert second["id"] == first["id"] + 1          # ids continue, none reused
    fresh = _clean(dispatch(ctx, "circuit", {"target": "show", "id": second["id"]}))
    assert "from_previous_run" not in fresh

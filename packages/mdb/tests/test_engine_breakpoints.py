import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.core import address as addresses  # noqa: E402
from mdb.engine import breakpoints as bp  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops import BadArgs, dispatch  # noqa: E402

PROMPT = [1, 5, 9, 3]
ROW = len(PROMPT) - 1


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _ctx(debugger):
    ctx = debugger.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    return ctx


# --------------------------------------------------------------------------- conditions


@pytest.mark.parametrize("text", [
    "__import__('os')", "v.shape", "open('x')", "norm.real", "lambda: 1", "x > 1",
    "[norm]", "norm if max else min", "v[norm]", "v[0:2]", "norm(v, v)", "True",
    "'abc' == norm", "norm ** 2", "max(1)" + " " * 300,
])
def test_condition_whitelist_refuses(text):
    with pytest.raises(bp.ConditionError):
        bp.Condition(text)


def _env(values, axes=("batch", "row", "d_model")):
    full = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)
    selection = bp.select(full, axes, addresses.Selector())
    return bp.ConditionEnv(full, selection, axes)


def test_condition_evaluates_reductions_elements_and_selectors():
    env = _env([3.0, -4.0, 0.0])
    assert bp.Condition("norm == 5").evaluate(env)
    assert bp.Condition("abs_max == 4 and argmax == 0 and numel == 3").evaluate(env)
    assert bp.Condition("v[0, 0, 1] < -3.5").evaluate(env)
    assert bp.Condition("v[0, 0, -1] == 0").evaluate(env)
    assert bp.Condition('norm(v["d=0:2"]) == 5 and max(v["d=2"]) == 0').evaluate(env)
    assert bp.Condition("not (mean > 0) and -min > 3.9").evaluate(env)
    assert not bp.Condition("1 < norm < 4").evaluate(env)
    with pytest.raises(bp.ConditionError, match="not a number"):
        bp.Condition('v["d=0:2"] > 0').evaluate(env)
    with pytest.raises(bp.ConditionError, match="unknown name 'v'"):
        bp.Condition("v > 0")
    with pytest.raises(bp.ConditionError, match="out of range"):
        bp.Condition("v[0, 0, 7] > 0").evaluate(env)


def test_select_maps_keys_to_axes():
    value = torch.arange(3 * 4, dtype=torch.float32).reshape(1, 3, 1, 4)
    axes = ("batch", "head", "row", "src")
    sel = addresses.parse("t5.L0.p[h1,src=1:3]").selector
    picked = bp.select(value, axes, sel, row=5)
    assert torch.equal(picked.value, value[:, 1:2, :, 1:3])
    assert picked.original(1) == [0, 1, 0, 2]
    with pytest.raises(bp.SelectionError, match="only row 5"):
        bp.select(value, axes, addresses.parse("t5.L0.p[row=4]").selector, row=5)
    flat = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
    q = bp.select(flat, ("batch", "row", "head*head_dim"),
                  addresses.parse("t0.L0.q[h1,d=2]").selector, head_dim=8)
    assert q.value.reshape(-1).tolist() == [10.0]
    with pytest.raises(bp.SelectionError, match="out of range"):
        bp.select(value, axes, addresses.parse("t5.L0.p[src=9]").selector)


# --------------------------------------------------------------------------- continue


def test_continue_stops_at_each_layer_p_in_order(debugger):
    ctx = _ctx(debugger)
    layers = ctx.run.adapter.layers
    dispatch(ctx, "break", {"pattern": "L*.p"})
    seen = []
    for _ in range(layers + 1):
        result = dispatch(ctx, "continue", {})
        assert result["reason"] == "breakpoint"
        seen.append(result["where"])
    expected = [f"t{ROW}.L{i}.p" for i in range(layers)] + [f"t{ROW + 1}.L0.p"]
    assert seen == expected
    assert dispatch(ctx, "info", {"subject": "breakpoints"})["items"][0]["hits"] == layers + 1


def test_breakpoint_selector_does_not_constrain_the_stop(debugger):
    ctx = _ctx(debugger)
    # The condition sees the selection p[h0]: one head with row + 1 sources. It is false at row
    # ROW and true at ROW + 1, and the selector never stops the breakpoint from matching.
    dispatch(ctx, "break", {"pattern": "L1.p[h0]", "condition": f"numel == {ROW + 2}"})
    assert dispatch(ctx, "continue", {"budget": 2})["where"] == f"t{ROW + 1}.L1.p"


def test_conditional_breakpoint_holds_at_exactly_one_layer(debugger):
    run = debugger.run(PROMPT)
    run.until(f"t{ROW}.L1.h_out")
    norms = [float(run.read(f"t{ROW}.L{i}.h_out").double().norm()) for i in (0, 1)]
    assert norms[0] != norms[1]
    middle = sum(norms) / 2
    hit = 1 if norms[1] > norms[0] else 0
    op = ">" if hit == 1 else "<"
    ctx = _ctx(debugger)
    dispatch(ctx, "break", {"pattern": f"t{ROW}.L*.h_out", "condition": f"norm {op} {middle!r}"})
    result = dispatch(ctx, "continue", {"budget": 2})
    assert (result["where"], result["reason"]) == (f"t{ROW}.L{hit}.h_out", "breakpoint")
    assert dispatch(ctx, "continue", {"budget": 2})["reason"] == "budget"


def test_tbreak_fires_once(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "tbreak", {"pattern": "L1.o"})
    dispatch(ctx, "break", {"pattern": "L0.p"})
    first = dispatch(ctx, "continue", {})
    assert first["where"] == f"t{ROW}.L0.p"
    second = dispatch(ctx, "continue", {})
    assert second["where"] == f"t{ROW}.L1.o" and second["hits"][0]["kind"] == "tbreak"
    items = dispatch(ctx, "info", {"subject": "breakpoints"})["items"]
    assert [item["pattern"] for item in items] == ["L0.p"]
    assert dispatch(ctx, "continue", {})["where"] == f"t{ROW + 1}.L0.p"


def test_watch_change_fires_exactly_where_the_selected_value_changes(debugger):
    budget = 3
    run = debugger.run(PROMPT)
    layers = run.adapter.layers
    sequence = []
    for row in range(ROW, ROW + budget):
        for layer in range(layers):
            address = f"t{row}.L{layer}.h"
            run.until(address)
            sequence.append((address, tensor_digest(run.read()[..., :1].contiguous())))
    expected = [where for (where, digest), (_, before) in zip(sequence[1:], sequence)
                if digest != before]
    assert expected, "fixture should produce at least one change"
    ctx = _ctx(debugger)
    watch = dispatch(ctx, "watch", {"address": "L*.h[d=0]"})
    assert watch["baseline"] == f"t{ROW}.L0.h"
    fired, end = [], ROW + budget
    while True:
        row = int(ctx.run.where.split(".")[0][1:])
        result = dispatch(ctx, "continue", {"budget": end - row})
        if result["reason"] != "watch":
            assert (result["reason"], result["where"]) == ("budget", f"t{end}.start")
            break
        fired.append(result["where"])
        hit = result["hits"][0]
        assert hit["old"]["norm"]["authority"] == "measured"
        assert hit["new"]["numel"] == 1
    assert fired == expected


def test_watch_on_an_unchanging_cell_never_fires(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "watch", {"address": "L0.kr[src=0]"})
    result = dispatch(ctx, "continue", {"budget": 2})
    assert (result["reason"], result["where"]) == ("budget", f"t{ROW + 2}.start")


def test_level_and_edge_watches_and_bad_watches(debugger):
    ctx = _ctx(debugger)
    with pytest.raises(BadArgs, match="needs a predicate"):
        dispatch(ctx, "watch", {"address": "L0.h", "mode": "level"})
    with pytest.raises(BadArgs):
        dispatch(ctx, "watch", {"address": "L0.h", "mode": "edge", "predicate": "norm >"})
    dispatch(ctx, "watch", {"address": "L*.a", "mode": "level", "predicate": "norm > 0"})
    assert dispatch(ctx, "continue", {})["where"] == f"t{ROW}.L0.a"
    assert dispatch(ctx, "continue", {})["where"] == f"t{ROW}.L1.a"
    dispatch(ctx, "delete", {})
    dispatch(ctx, "watch", {"address": "L*.a", "mode": "edge", "predicate": "norm > 0"})
    assert dispatch(ctx, "continue", {"budget": 1})["reason"] == "budget"


def test_reverse_continue_returns_to_the_previous_hit(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "break", {"pattern": "L*.p"})
    first = dispatch(ctx, "continue", {})["where"]
    value = tensor_digest(ctx.run.read())
    second = dispatch(ctx, "continue", {})["where"]
    assert first != second
    back = dispatch(ctx, "reverse-continue", {})
    assert (back["where"], back["reason"]) == (first, "breakpoint")
    assert tensor_digest(ctx.run.read()) == value
    root = dispatch(ctx, "reverse-continue", {})
    assert (root["reason"], root["where"]) == ("start-of-run", f"t{ROW}.start")


def test_advance_and_until_are_exact(debugger):
    ctx = _ctx(debugger)
    assert dispatch(ctx, "until", {"target": "L1.sg"})["where"] == f"t{ROW}.L1.sg"
    assert dispatch(ctx, "advance", {"target": f"t{ROW + 1}.L0.k"})["where"] == f"t{ROW + 1}.L0.k"
    dispatch(ctx, "break", {"pattern": "fn"})
    result = dispatch(ctx, "advance", {"target": f"t{ROW + 2}.L0.k"})
    assert (result["where"], result["reason"]) == (f"t{ROW + 1}.fn", "breakpoint")
    result = dispatch(ctx, "advance", {"target": f"t{ROW + 2}.L0.k"})
    assert (result["where"], result["reason"]) == (f"t{ROW + 2}.L0.k", "advance")


# =========================================================================== fix round, part 1

import time  # noqa: E402

from mdb.ops import Refused  # noqa: E402


def _small_context_debugger(max_positions=7):
    """A random tiny Qwen3 whose context is only ``max_positions`` long (end-of-context)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from mdb.ops.debugger import attach

    torch.manual_seed(3)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=max_positions,
                         attn_implementation="eager")
    return attach(Qwen3ForCausalLM(config).eval())


# --- O2 bounded compute ------------------------------------------------------------------


def test_continue_budget_zero_is_one_row_not_forever(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "break", {"pattern": "L0.p", "condition": "norm < 0"})  # never true
    started = time.perf_counter()
    result = dispatch(ctx, "continue", {"budget": 0})
    assert time.perf_counter() - started < 20
    assert (result["reason"], result["where"], result["budget"]) == (
        "budget", f"t{ROW + 1}.start", 1)


def test_end_of_context_is_a_normal_stop_not_an_error():
    dbg = _small_context_debugger(max_positions=7)
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    dispatch(ctx, "break", {"pattern": "L0.p", "condition": "norm < 0"})
    result = dispatch(ctx, "continue", {"budget": 4096})
    assert result["reason"] == "end-of-context"
    assert result["where"] == ctx.run.where
    assert result["hits"] == [] and result["state"] == "resolved"
    row = int(result["where"].split(".")[0][1:])
    assert ROW < row <= 6  # moved forward, never past the 7 positions
    # the cursor is usable: it can be read and walked back
    assert dispatch(ctx, "where", {})["where"] == result["where"]
    assert dispatch(ctx, "reverse-step", {})["where"] != result["where"]


# --- O3 paging, hit caps, evaluation errors ------------------------------------------------


def test_seventy_breakpoints_page_delete_and_continue(debugger):
    ctx = _ctx(debugger)
    for _ in range(70):
        dispatch(ctx, "break", {"pattern": "L0.p"})
    listed = dispatch(ctx, "info", {"subject": "breakpoints"})
    assert (listed["total"], listed["offset"], listed["limit"], len(listed["items"])) == (
        70, 0, 32, 32)
    last = dispatch(ctx, "info", {"subject": "breakpoints", "offset": 64, "limit": 64})
    assert [item["id"] for item in last["items"]] == list(range(65, 71))
    assert dispatch(ctx, "info", {"subject": "breakpoints", "offset": 500})["items"] == []
    stop = dispatch(ctx, "continue", {})
    assert (stop["reason"], stop["hits_total"], len(stop["hits"]), stop["omitted"]) == (
        "breakpoint", 70, 32, 38)
    assert stop["where"] == f"t{ROW}.L0.p"
    counted = dispatch(ctx, "info", {"subject": "breakpoints", "limit": 64})["items"]
    assert all(item["hits"] == 1 for item in counted)
    deleted = dispatch(ctx, "delete", {})
    assert (deleted["deleted_total"], len(deleted["deleted"]), deleted["omitted"]) == (70, 32, 38)
    assert dispatch(ctx, "info", {"subject": "breakpoints"})["total"] == 0


def test_every_list_is_paged_never_refused_for_length(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.q"})
    for i in range(70):
        dispatch(ctx, "set", {"target": f"L0.q[d={i % 8}]", "value": 0.0, "op": "add"})
    staged = dispatch(ctx, "info", {"subject": "edits"})
    assert staged["staged_total"] == 70 and len(staged["staged"]) == 32
    tail = dispatch(ctx, "info", {"subject": "edits", "offset": 64, "limit": 64})
    assert len(tail["staged"]) == 6 and tail["committed"] == []
    for _ in range(70):
        dispatch(ctx, "checkpoint", {})
    page = dispatch(ctx, "checkpoints", {"offset": 60})
    assert (page["total"], len(page["checkpoints"])) == (70, 10)
    assert dispatch(ctx, "info", {"subject": "checkpoints"})["total"] == 70
    for i in range(8):
        dispatch(ctx, "display", {"address": f"L0.q[d={i}]"})
    shown = dispatch(ctx, "info", {"subject": "display", "limit": 3, "offset": 2})
    assert (shown["total"], len(shown["displays"])) == (8, 3)
    dispatch(ctx, "commit", {})  # 70 edits in one layer still committed and listed (capped)
    done = dispatch(ctx, "info", {"subject": "edits", "limit": 5})
    entry = done["committed"][0]
    assert entry["edits_total"] == 70 and len(entry["edits"]) == 5 and entry["omitted"] == 65


def test_condition_error_mid_continue_stops_with_reason_error(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "break", {"pattern": "L1.a", "condition": "norm / (max - max) > 0"})
    first = dispatch(ctx, "continue", {})
    assert (first["reason"], first["where"]) == ("error", f"t{ROW}.L1.a")
    assert first["error"]["id"] == 1 and "division by zero" in first["error"]["message"]
    assert first["hits"] == [] and ctx.run.where == f"t{ROW}.L1.a"
    again = dispatch(ctx, "continue", {})  # keeps going from the stop, errors at the next one
    assert (again["reason"], again["where"]) == ("error", f"t{ROW + 1}.L1.a")
    dispatch(ctx, "delete", {})
    dispatch(ctx, "break", {"pattern": "L*.p[h0]", "condition": "v[0,0,0,50] > 0"})
    idx = dispatch(ctx, "continue", {})
    assert idx["reason"] == "error" and "out of range" in idx["error"]["message"]
    assert idx["where"] == ctx.run.where and idx["state"] == "resolved"


def test_a_watch_that_cannot_be_evaluated_stops_the_motion(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "watch", {"address": "L*.a", "mode": "level",
                            "predicate": "norm / (max - max) > 0"})
    result = dispatch(ctx, "continue", {})
    assert (result["reason"], result["error"]["id"]) == ("error", 1)


# --- O4 non-finite --------------------------------------------------------------------------


def test_reduce_and_top_over_finite_elements_with_nonfinite_counts():
    value = torch.tensor([1.0, float("nan"), -3.0, float("inf"), float("-inf"), 2.0]).reshape(
        1, 1, 6)
    selection = bp.select(value, ("batch", "row", "d_model"), addresses.Selector())
    stats = bp.reduce(selection)
    assert stats["over"] == "finite" and stats["nonfinite"] == {"nan": 1, "posinf": 1, "neginf": 1}
    assert stats["max"] == 2.0 and stats["min"] == -3.0 and stats["abs_max"] == 3.0
    assert stats["argmax"] == 5 and stats["argmax_index"] == [0, 0, 5]
    assert stats["norm"] == pytest.approx(14.0 ** 0.5)
    assert [index for index, _ in bp.top(selection, 3)] == [[0, 0, 2], [0, 0, 5], [0, 0, 0]]
    empty = bp.select(torch.full((1, 1, 3), float("nan")), ("batch", "row", "d_model"),
                      addresses.Selector())
    assert bp.reduce(empty)["unresolved"] == "no finite element"
    assert bp.top(empty, 3) == []
    clean = bp.reduce(bp.select(torch.ones(1, 1, 2), ("batch", "row", "d_model"),
                                addresses.Selector()))
    assert "nonfinite" not in clean and "over" not in clean


def test_ops_label_statistics_over_finite_and_count_the_rest():
    from mdb.ops import builtin

    raw = bp.reduce(bp.select(torch.tensor([[[1.0, float("inf"), -2.0]]]),
                              ("batch", "row", "d_model"), addresses.Selector()))
    stats = builtin._stats(raw)
    assert stats["over"] == "finite" and stats["nonfinite"] == {"nan": 0, "posinf": 1, "neginf": 0}
    assert stats["max"] == {"value": 1.0, "authority": "measured"}
    none = builtin._stats(bp.reduce(bp.select(torch.full((1, 1, 2), float("nan")),
                                               ("batch", "row", "d_model"),
                                               addresses.Selector())))
    assert none["norm"] == {"state": "unresolved", "reason": "no finite element"}
    assert none["argmax_index"]["state"] == "unresolved"


def test_inspecting_and_committing_a_nonfinite_state(debugger):
    ctx = _ctx(debugger)
    if ctx.run.adapter.dtype != torch.bfloat16:
        pytest.skip("3e38 overflows only in bfloat16 with this model")
    dispatch(ctx, "until", {"target": "L0.q"})
    with pytest.raises(Refused, match="finite range of bfloat16"):
        dispatch(ctx, "set", {"target": "q", "value": 1e39})
    assert dispatch(ctx, "info", {"subject": "edits"})["staged"] == []  # nothing was staged
    dispatch(ctx, "set", {"target": "q", "value": 3e38})
    dispatch(ctx, "until", {"target": "L0.A"})
    printed = dispatch(ctx, "print", {"address": "s"})
    stats = printed["stats"]
    assert stats["over"] == "finite" and sum(stats["nonfinite"].values()) > 0
    assert printed["state"] in ("resolved", "unresolved")
    x = dispatch(ctx, "x", {"address": "s", "count": 8})
    assert set(x["values"]["values"]) <= {"inf", "-inf", "nan"} | {
        v for v in x["values"]["values"] if isinstance(v, float)}
    assert sum(x["nonfinite"].values()) > 0
    where = ctx.run.where
    with pytest.raises(Refused, match="abort"):
        dispatch(ctx, "commit", {})
    assert ctx.run.where == where and len(ctx.run.edits) == 1  # refused: unchanged
    assert dispatch(ctx, "abort", {})["count"] == 1


# --- O5 selector ranges --------------------------------------------------------------------


def test_huge_selector_ranges_are_refused_before_anything_is_expanded(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.a"})
    big = 10 ** 14  # the largest range end the address grammar accepts (15 digits)
    for op, args in (("print", {"address": f"L0.h[d=0:{big}]"}),
                     ("x", {"address": f"L0.h[d=0:{big}]"}),
                     ("set", {"target": f"L0.a[d=0:{big}]", "value": 0.0}),
                     ("display", {"address": f"L0.h[d=0:{big}]"})):
        started = time.perf_counter()
        with pytest.raises(Refused, match="out of range.*size"):
            dispatch(ctx, op, args)
        assert time.perf_counter() - started < 0.05, op
    started = time.perf_counter()
    with pytest.raises(BadArgs, match="more than 15 digits"):  # beyond the grammar: also quick
        dispatch(ctx, "print", {"address": f"L0.h[d=0:{10 ** 15}]"})
    assert time.perf_counter() - started < 0.05
    started = time.perf_counter()
    with pytest.raises(bp.SelectionRange, match="d_model"):
        bp.select(torch.zeros(1, 1, 8), ("batch", "row", "d_model"),
                  addresses.parse(f"t0.L0.h[d=0:{big}]").selector)
    assert time.perf_counter() - started < 0.05
    with pytest.raises(bp.SelectionRange, match="size 3"):
        bp.plan(("batch", "src"), [1, 3], addresses.parse("t0.L0.p[src=3|4|5]").selector)


def test_selector_strings_in_conditions_are_checked_when_the_breakpoint_is_made(debugger):
    ctx = _ctx(debugger)
    big = 10 ** 14
    started = time.perf_counter()
    with pytest.raises(Refused, match="out of range.*d_model"):
        dispatch(ctx, "break", {"pattern": "L0.a", "condition": f'norm(v["d=0:{big}"]) > 0'})
    assert time.perf_counter() - started < 0.05
    with pytest.raises(Refused, match="out of range"):
        dispatch(ctx, "break", {"pattern": "L0.h[d=999]"})
    with pytest.raises(BadArgs, match="bad selector"):
        dispatch(ctx, "break", {"pattern": "L0.a", "condition": 'norm(v["zz=1"]) > 0'})
    assert dispatch(ctx, "info", {"subject": "breakpoints"})["total"] == 0  # no id burned
    ok = dispatch(ctx, "break", {"pattern": "L0.a", "condition": 'norm(v["d=0:2"]) >= 0'})
    assert ok["breakpoint"]["id"] == 1
    # an unknowable size (src depends on the row) is checked at first evaluation
    dispatch(ctx, "break", {"pattern": "L0.p", "condition": 'max(v["src=90"]) > 0'})
    stop = dispatch(ctx, "continue", {})
    assert stop["reason"] == "breakpoint"  # the L0.a breakpoint, earlier
    stop = dispatch(ctx, "continue", {})
    assert stop["reason"] == "error" and "out of range" in stop["error"]["message"]


def test_a_concrete_layer_beyond_the_model_is_refused_with_the_valid_range(debugger):
    ctx = _ctx(debugger)
    layers = ctx.run.adapter.layers
    for op, args in (("break", {"pattern": f"L{layers}.p"}), ("tbreak", {"pattern": f"L{layers}.p"}),
                     ("watch", {"address": f"L{layers + 3}.h"})):
        with pytest.raises(Refused, match=rf"valid layers are 0\.\.{layers - 1}"):
            dispatch(ctx, op, args)
    assert dispatch(ctx, "info", {"subject": "breakpoints"})["total"] == 0


# --- O7 reverse motion ----------------------------------------------------------------------


class _LayerCounter:
    """Counts native decoder-layer executions (an independent forward hook)."""

    def __init__(self, model):
        self.count = 0
        self.handles = [layer.register_forward_pre_hook(self._hit)
                        for layer in model.model.layers]

    def _hit(self, *_):
        self.count += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


def test_reverse_continue_is_one_cursor_move_not_a_walk(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": f"t{ROW + 3}.L1.v"})
    dispatch(ctx, "break", {"pattern": f"t{ROW}.L0.k"})  # three tokens back, one match
    counter = _LayerCounter(debugger.model)
    try:
        result = dispatch(ctx, "reverse-continue", {})
    finally:
        counter.close()
    assert (result["reason"], result["where"]) == ("breakpoint", f"t{ROW}.L0.k")
    # one boundary restore and ONE re-probe of L0 (the auditor measured 143 layer passes
    # for this walk before; the target is one pass)
    assert counter.count <= 2, counter.count
    again = dispatch(ctx, "reverse-continue", {})
    assert again["reason"] == "start-of-run" and again["where"] == f"t{ROW}.start"


def _selected_digest(run, address):
    return tensor_digest(run.read(address)[..., :1].contiguous())


def test_reverse_watch_baselines_on_the_cursor_value_and_keeps_its_own_memory(debugger):
    ctx = _ctx(debugger)
    run = ctx.run
    dispatch(ctx, "watch", {"address": "L*.h[d=0]"})
    target = f"t{ROW + 1}.L1.h"
    run.until(target)
    cursor = _selected_digest(run, target)
    expected = None
    for row, layer in ((ROW + 1, 0), (ROW, 1), (ROW, 0)):
        address = f"t{row}.L{layer}.h"
        if _selected_digest(run, address) != cursor:
            expected = address
            break
    assert expected is not None, "fixture should change somewhere"
    forward_memory = dict(ctx.table.items[1].state)
    result = dispatch(ctx, "reverse-continue", {})
    # the change between the previous matching stop and the cursor itself is found
    assert (result["reason"], result["where"]) == ("watch", expected)
    assert result["hits"][0]["new"]["numel"] == 1
    assert ctx.table.items[1].state == forward_memory  # forward memory untouched
    assert ctx.table.items[1].hits == 1


def test_restart_resets_watch_memory(debugger):
    ctx = _ctx(debugger)
    watch = dispatch(ctx, "watch", {"address": "L*.h[d=0]"})
    assert watch["baseline"] == f"t{ROW}.L0.h"
    dispatch(ctx, "checkpoint", {"name": "early"})
    first = dispatch(ctx, "continue", {"budget": 3})
    assert first["reason"] == "watch"
    assert ctx.table.items[1].state  # forward memory exists
    dispatch(ctx, "restart", {"name": "early"})
    assert ctx.table.items[1].state == {}
    # after the restart the first evaluation is only a baseline again (no spurious hit)
    second = dispatch(ctx, "continue", {"budget": 1})
    assert second["hits"] == [] or second["where"] != f"t{ROW}.L0.h"
    ctx.table.items[1].state = {"digest": "stale"}
    dispatch(ctx, "start", {})
    assert ctx.table.items[1].state == {}


# --- O8 gdb fidelity -----------------------------------------------------------------------


def test_edge_fires_once_where_level_fires_at_every_stop(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "watch", {"address": "L*.a", "mode": "level", "predicate": "norm > 0"})
    seen = [dispatch(ctx, "continue", {"budget": 4})["where"] for _ in range(3)]
    assert seen == [f"t{ROW}.L0.a", f"t{ROW}.L1.a", f"t{ROW + 1}.L0.a"]
    ctx = _ctx(debugger)
    dispatch(ctx, "watch", {"address": "L*.a", "mode": "edge", "predicate": "norm > 0"})
    first = dispatch(ctx, "continue", {"budget": 4})
    assert (first["reason"], first["where"]) == ("watch", f"t{ROW}.L0.a")
    later = dispatch(ctx, "continue", {"budget": 4})  # holds at every later stop: no new edge
    assert later["reason"] == "budget" and later["where"] == f"t{ROW + 4}.start"


def test_enable_disable_condition_and_ignore(debugger):
    ctx = _ctx(debugger)
    layers = ctx.run.adapter.layers
    dispatch(ctx, "break", {"pattern": "L*.p"})
    dispatch(ctx, "break", {"pattern": "L0.o"})
    assert dispatch(ctx, "disable", {"ids": [1]})["ids"] == [1]
    listed = dispatch(ctx, "info", {"subject": "breakpoints"})["items"]
    assert [(i["id"], i["enabled"]) for i in listed] == [(1, False), (2, True)]
    assert dispatch(ctx, "continue", {})["where"] == f"t{ROW}.L0.o"
    assert dispatch(ctx, "enable", {})["ids"] == [1, 2]
    assert dispatch(ctx, "continue", {})["where"] == f"t{ROW}.L1.p"
    with pytest.raises(Refused, match="no breakpoint or watch 9"):
        dispatch(ctx, "disable", {"ids": [1, 9]})
    assert all(i["enabled"] for i in dispatch(ctx, "info", {"subject": "breakpoints"})["items"])
    # condition N EXPR, and an empty EXPR removes it
    cond = dispatch(ctx, "condition", {"id": 1, "expr": "norm < 0"})
    assert cond["breakpoint"]["condition"] == "norm < 0" and cond["removed"] is False
    nxt = dispatch(ctx, "continue", {"budget": 2})
    assert nxt["where"] != f"t{ROW + 1}.L0.p"  # L*.p is now never true
    removed = dispatch(ctx, "condition", {"id": 1})
    assert removed["breakpoint"]["condition"] is None and removed["removed"] is True
    with pytest.raises(BadArgs):
        dispatch(ctx, "condition", {"id": 1, "expr": "norm >"})
    # ignore N COUNT: the next COUNT hits are counted, not reported
    dispatch(ctx, "delete", {})
    ctx2 = _ctx(debugger)
    dispatch(ctx2, "break", {"pattern": "L*.p"})
    dispatch(ctx2, "ignore", {"id": 1, "count": layers})
    stop = dispatch(ctx2, "continue", {})
    assert stop["where"] == f"t{ROW + 1}.L0.p"
    item = dispatch(ctx2, "info", {"subject": "breakpoints"})["items"][0]
    assert (item["hits"], item["ignore"]) == (layers + 1, 0)


def test_continue_n_ignores_the_breakpoint_stopped_at(debugger):
    ctx = _ctx(debugger)
    layers = ctx.run.adapter.layers
    dispatch(ctx, "break", {"pattern": "L*.p"})
    assert dispatch(ctx, "continue", {})["where"] == f"t{ROW}.L0.p"
    result = dispatch(ctx, "continue", {"n": 3})  # ignore it twice more: stop at the 3rd next
    assert result["ignored"] == {"ids": [1], "count": 2}
    expected = [f"t{ROW}.L{i}.p" for i in range(layers)] + [f"t{ROW + 1}.L{i}.p"
                                                              for i in range(layers)]
    assert result["where"] == expected[3]
    item = dispatch(ctx, "info", {"subject": "breakpoints"})["items"][0]
    assert item["hits"] == 4 and item["ignore"] == 0
    other = dispatch(ctx, "until", {"target": f"t{ROW + 3}.L0.k"})  # stops at the breakpoint
    assert other["reason"] == "breakpoint"
    dispatch(ctx, "delete", {})
    note = dispatch(ctx, "continue", {"n": 2})
    assert "not stopped at a breakpoint" in note["note"]


def test_until_and_advance_both_honour_breakpoints(debugger):
    for verb in ("until", "advance"):
        ctx = _ctx(debugger)
        dispatch(ctx, "break", {"pattern": "L1.p"})
        result = dispatch(ctx, verb, {"target": f"t{ROW}.L1.o"})
        assert (result["reason"], result["where"]) == ("breakpoint", f"t{ROW}.L1.p"), verb
        again = dispatch(ctx, verb, {"target": f"t{ROW}.L1.o"})
        assert again["where"] == f"t{ROW}.L1.o", verb
    ctx = _ctx(debugger)
    reached = dispatch(ctx, "until", {"target": "L1.p"})  # nothing armed: exact, as before
    assert reached["where"] == f"t{ROW}.L1.p" and reached["hits"] == []
    with pytest.raises(Refused, match="past the model's context limit"):
        dispatch(ctx, "until", {"target": "t99999.L0.h"})


def test_print_without_an_argument_shows_the_breakpoint_selection(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "break", {"pattern": "L1.p[h0]"})
    stop = dispatch(ctx, "continue", {})
    assert stop["focus"] == f"t{ROW}.L1.p[h0]"
    shown = dispatch(ctx, "print", {})
    assert shown["address"] == f"t{ROW}.L1.p[h0]"
    assert shown["selected_shape"][1] == 1  # one head of the (batch, head, row, src) value
    dispatch(ctx, "step", {})
    assert dispatch(ctx, "print", {})["address"] == ctx.run.where  # the selection was the stop's

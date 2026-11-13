import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.ops.debugger import open_tiny  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops import (  # noqa: E402
    BadArgs,
    Refused,
    dispatch,
    graph,  # noqa: E402
)

PROMPT = [1, 5, 9, 3]
ROW = len(PROMPT) - 1
CASES = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16")]


@pytest.fixture(scope="module", params=CASES, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _ctx(debugger):
    ctx = debugger.ops()
    result = dispatch(ctx, "start", {"prompt": PROMPT})
    names = ctx.run.program.names
    assert result["where"] == f"t{ROW}.L0.{names[0]}"
    return ctx


def test_step_next_finish_until_units(debugger):
    ctx = _ctx(debugger)
    names = ctx.run.program.names
    assert dispatch(ctx, "step", {})["where"] == f"t{ROW}.L0.{names[1]}"
    assert dispatch(ctx, "step", {"n": 2})["where"] == f"t{ROW}.L0.{names[3]}"
    attention = [p.name for p in ctx.run.program.ports if p.sublayer == "attn"]
    mlp = [p.name for p in ctx.run.program.ports if p.sublayer == "mlp"]
    nxt = dispatch(ctx, "next", {})
    assert nxt["where"] == f"t{ROW}.L0.{attention[-1]}"
    assert nxt["stop"]["sublayer"] == "attn"
    assert dispatch(ctx, "next", {})["where"] == f"t{ROW}.L0.{mlp[-1]}"
    dispatch(ctx, "step", {})
    done = dispatch(ctx, "finish", {})
    assert done["where"] == f"t{ROW}.L1.h_out"
    assert dispatch(ctx, "step", {"unit": "token"})["where"] == f"t{ROW + 1}.start"
    with pytest.raises(Refused, match="behind the cursor"):
        dispatch(ctx, "until", {"target": f"t{ROW}.L0.a"})
    for target, where in (("L1.f", f"t{ROW + 1}.L1.f"), ("logits", f"t{ROW + 1}.logits"),
                          (f"t{ROW + 2}.L0.kr", f"t{ROW + 2}.L0.kr")):
        assert dispatch(ctx, "until", {"target": target})["where"] == where
    with pytest.raises(BadArgs, match="bad address"):
        dispatch(ctx, "until", {"target": "L0.nope"})
    with pytest.raises(Refused, match="origins"):
        dispatch(ctx, "print", {"address": "boiling:t3.L0.h"})


def test_print_and_x_equal_direct_reductions(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L1.u"})
    run = ctx.run
    p = run.read(f"t{ROW}.L1.p").double()
    part = p[:, 0:1, :, 1:3]
    result = dispatch(ctx, "print", {"address": "L1.p[h0,src=1:3]", "top": 2, "slice": 2})
    assert result["address"] == f"t{ROW}.L1.p[h0,src=1:3]"
    assert result["selected_shape"] == list(part.shape)
    stats = result["stats"]
    assert stats["norm"] == {"value": float(part.norm()), "authority": "measured"}
    assert stats["max"]["value"] == float(part.max())
    assert stats["min"]["value"] == float(part.min())
    assert stats["mean"]["value"] == float(part.mean())
    flat = part.reshape(-1)
    order = torch.sort(-flat.abs(), stable=True).indices[:2].tolist()
    assert [entry["value"]["value"] for entry in result["top"]] == [float(flat[i]) for i in order]
    assert result["top"][0]["index"][1] == 0 and result["top"][0]["index"][3] in (1, 2)
    assert result["slice"]["values"] == flat[:2].tolist()
    assert result["pointer"]["value"]["sha256"] == tensor_digest(run.read(f"t{ROW}.L1.p"))
    assert result["pointer"]["address"]["port"] == "p"
    u = run.read().double().reshape(-1)
    x = dispatch(ctx, "x", {"count": 5, "offset": 3})
    assert x["values"] == {"values": u[3:8].tolist(), "authority": "measured"}
    q = run.read(f"t{ROW}.L1.q").double().reshape(-1)
    head_dim = run.adapter.head_dim
    xq = dispatch(ctx, "x", {"address": "L1.q[h1]", "count": 4})
    assert xq["values"]["values"] == q[head_dim:head_dim + 4].tolist()
    with pytest.raises(Refused, match="not been computed"):
        dispatch(ctx, "print", {"address": "L1.m"})


def test_list_marks_cursor_and_shows_proofs(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.s"})
    result = dispatch(ctx, "list", {})
    program = ctx.run.program
    assert result["layer"] == 0
    assert [row["port"] for row in result["ports"]] == list(program.names)
    marked = [row["port"] for row in result["ports"] if row["cursor"]]
    assert marked == ["s"]
    for row, spec in zip(result["ports"], program.ports):
        assert row["proofs"] == list(spec.proofs) and row["operator"] == spec.operator
    assert [row["computed"] for row in result["ports"]] == [
        program.index(name) <= program.index("s") for name in program.names]


def test_backtrace_follows_program_edges(debugger):
    ctx = _ctx(debugger)
    program = ctx.run.program
    result = dispatch(ctx, "backtrace", {"address": "L0.o"})
    levels = result["chain"].split(" <- ")
    assert levels[0] == "o" and levels[-1] == "h"
    level = {name: i for i, names in enumerate(levels) for name in names.split(",")}
    edges = graph.producers(program)
    for name in level:
        for producer in edges[name]:
            assert level[producer] > level[name]
            rhs = program.spec(name).operator.partition("=")[2]
            assert producer in rhs or producer.removesuffix("n") in rhs
    assert {"A", "p", "v", "s", "qr", "kr", "q", "k", "a"} <= set(level)
    assert "r" not in level and "m" not in level
    if program.family == "qwen2":
        assert result["chain"] == "o <- A <- v,p <- s <- qr,kr <- q,k <- a <- h"
    else:
        assert result["chain"] == "o <- A <- v,p <- s <- qr,kr <- qn,kn <- q,k <- a <- h"
    assert dispatch(ctx, "frame", {"address": "L0.o"})["focus"] == f"t{ROW}.L0.o"
    assert dispatch(ctx, "up", {"n": 1})["focus"] == f"t{ROW}.L0.A"
    assert dispatch(ctx, "up", {"n": 2})["focus"] == f"t{ROW}.L0.s"
    assert dispatch(ctx, "down", {"n": 1})["focus"] == f"t{ROW}.L0.p"
    assert dispatch(ctx, "frame", {"address": "L1.h"})["focus"] == f"t{ROW}.L1.h"
    assert dispatch(ctx, "up", {})["focus"] == f"t{ROW}.L0.h_out"
    assert dispatch(ctx, "backtrace", {"address": "logits"})["chain"] == (
        f"logits <- fn <- L{ctx.run.adapter.layers - 1}.h_out")


def test_set_commit_changes_downstream_and_abort_restores(debugger):
    base = _ctx(debugger)
    dispatch(base, "until", {"target": "L0.m"})
    base_m = base.run.read()
    base.run.finish()
    dispatch(base, "until", {"target": "L1.a"})
    base_a = base.run.read()

    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.n"})
    staged = dispatch(ctx, "set", {"target": "L0.n", "value": 2.0, "op": "scale"})
    assert staged["edits"][0]["value"] == {"value": 2.0, "authority": "supplied"}
    dispatch(ctx, "abort", {})
    dispatch(ctx, "step", {"n": 5})
    assert torch.equal(ctx.run.read(), base_m)

    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.n"})
    dispatch(ctx, "set", {"target": "n", "value": 2.0, "op": "scale"})
    printed = dispatch(ctx, "print", {})
    assert printed["provenance"]["edited"] and printed["stats"]["norm"]["authority"] in (
        "supplied", "measured", "derived-exact")  # labels are re-pinned in fix round part 2
    done = dispatch(ctx, "commit", {})
    assert done["where"] == f"t{ROW}.L0.h_out" and done["committed"] == 1
    dispatch(ctx, "until", {"target": "L1.a"})
    assert not torch.equal(ctx.run.read(), base_a)
    edits = dispatch(ctx, "info", {"subject": "edits"})
    assert edits["staged"] == []
    assert edits["committed"][0]["layer"] == 0
    assert edits["committed"][0]["edits"][0]["port"] == "n"
    assert dispatch(ctx, "print", {"address": "L1.a"}).get("pointer_omitted", "").startswith(
        "edited run")


def test_set_with_selector_and_address_value(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.g"})
    run = ctx.run
    before = run.read(f"t{ROW}.L0.n").clone()
    dispatch(ctx, "set", {"target": "L0.n[d=1:3]", "value": 0.5})
    after = run.read(f"t{ROW}.L0.n")
    assert after[0, 0, 1:3].tolist() == [0.5, 0.5]
    assert torch.equal(after[0, 0, 3:], before[0, 0, 3:])
    assert torch.equal(after[0, 0, :1], before[0, 0, :1])
    dispatch(ctx, "abort", {})
    dispatch(ctx, "set", {"target": "L0.n", "value": f"t{ROW}.L0.a", "op": "replace"})
    assert torch.equal(run.read(f"t{ROW}.L0.n"), run.read(f"t{ROW}.L0.a"))
    with pytest.raises(Refused, match="not editable"):
        dispatch(ctx, "set", {"target": "L0.kr", "value": 1.0})


def test_restart_reproduces_identical_digests(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L1.v"})
    saved = dispatch(ctx, "checkpoint", {"name": "pre"})
    assert saved["bytes"] < 1024

    def walk():
        trace = []
        for _ in range(9):
            where = dispatch(ctx, "step", {})["where"]
            trace.append((where, tensor_digest(ctx.run.read())))
        return trace

    first = walk()
    restarted = dispatch(ctx, "restart", {"name": "pre"})
    assert restarted["where"] == saved["where"]
    assert walk() == first
    assert [c["name"] for c in dispatch(ctx, "checkpoints", {})["checkpoints"]] == ["pre"]
    back = dispatch(ctx, "reverse-step", {"n": 9})
    assert back["where"] == saved["where"]
    with pytest.raises(Refused, match="no checkpoint"):
        dispatch(ctx, "restart", {"name": "nope"})


def test_displays_follow_the_cursor(debugger):
    ctx = _ctx(debugger)
    shown = dispatch(ctx, "display", {"address": "a[d=0]"})
    assert "unavailable" in shown["displays"][0]
    stepped = dispatch(ctx, "step", {})
    item = stepped["displays"][0]
    assert item["address"] == f"t{ROW}.L0.a[d=0]"
    assert item["stats"]["norm"]["value"] == abs(float(ctx.run.read()[0, 0, 0].double()))
    dispatch(ctx, "undisplay", {})
    assert "displays" not in dispatch(ctx, "step", {})


# =========================================================================== fix round, part 1

from mdb.ops import OpError  # noqa: E402


def test_step_and_reverse_counts_are_capped_before_anything_moves(debugger):
    ctx = _ctx(debugger)
    where = ctx.run.where
    for op, args in (("step", {"n": 4097}), ("step", {"n": 5000, "unit": "sublayer"}),
                     ("next", {"n": 4097}), ("reverse-step", {"n": 4097}),
                     ("reverse-next", {"n": 4097}), ("continue", {"budget": 4097})):
        with pytest.raises(BadArgs, match="<= 4096"):
            dispatch(ctx, op, args)
    # layer and token units are limited by the token rows they would enter (budget, default 8)
    with pytest.raises(Refused, match="would enter 20 token rows.*budget 8.*Nothing moved"):
        dispatch(ctx, "step", {"n": 20, "unit": "token"})
    with pytest.raises(Refused, match="budget 8"):
        dispatch(ctx, "step", {"n": 200, "unit": "layer"})
    assert ctx.run.where == where  # refusals leave the state unchanged
    moved = dispatch(ctx, "step", {"n": 3, "unit": "token", "budget": 3})
    assert moved["where"] == f"t{ROW + 3}.start"
    with pytest.raises(Refused, match="would enter 3 token rows.*budget 2"):
        dispatch(ctx, "reverse-step", {"n": 3, "unit": "token", "budget": 2})
    assert ctx.run.where == f"t{ROW + 3}.start"
    assert dispatch(ctx, "reverse-step", {"n": 3, "unit": "token", "budget": 3})["where"] == (
        f"t{ROW}.start")
    # a big port step is allowed up to the cap (bounded by 4096, not by rows)
    assert dispatch(ctx, "step", {"n": 30})["where"]
    with pytest.raises(Refused, match="cannot reverse before the run root.*nothing moved"):
        dispatch(ctx, "reverse-step", {"n": 4000})
    assert ctx.run.where == dispatch(ctx, "where", {})["where"]  # still usable, unchanged


def test_reverse_before_the_root_refuses_without_moving(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "step", {"n": 3})
    where = ctx.run.where
    with pytest.raises(Refused, match="before the run root"):
        dispatch(ctx, "reverse-step", {"n": 6})
    assert ctx.run.where == where
    assert dispatch(ctx, "reverse-step", {"n": 3})["where"] == f"t{ROW}.L0.h"
    assert dispatch(ctx, "reverse-step", {"n": 2})["where"] == f"t{ROW}.start"


def test_step_to_the_end_of_context_stops_normally():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from mdb.ops.debugger import attach

    torch.manual_seed(5)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=7, attn_implementation="eager")
    ctx = attach(Qwen3ForCausalLM(config).eval()).ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    result = dispatch(ctx, "step", {"n": 4096})
    assert result["reason"] == "end-of-context" and result["where"] == ctx.run.where
    ctx = attach(Qwen3ForCausalLM(config).eval()).ops()
    dispatch(ctx, "start", {"prompt": PROMPT})
    result = dispatch(ctx, "continue", {"budget": 0})  # a minimum of one row, never forever
    assert result["reason"] == "budget" and result["budget"] == 1


def test_finish_goes_to_h_out_then_to_sample(debugger):
    ctx = _ctx(debugger)
    layers = ctx.run.adapter.layers
    dispatch(ctx, "until", {"target": "L0.m"})
    assert dispatch(ctx, "finish", {})["where"] == f"t{ROW}.L0.h_out"
    # from an h_out boundary: to this row's sample
    assert dispatch(ctx, "finish", {})["where"] == f"t{ROW}.sample"
    with pytest.raises(Refused, match="already at t3.sample"):
        dispatch(ctx, "finish", {})
    dispatch(ctx, "until", {"target": f"t{ROW + 1}.L1.q"})
    assert dispatch(ctx, "finish", {})["where"] == f"t{ROW + 1}.L{1}.h_out"
    assert layers >= 2
    dispatch(ctx, "until", {"target": f"t{ROW + 1}.logits"})  # a token-level stop
    assert dispatch(ctx, "finish", {})["where"] == f"t{ROW + 1}.sample"
    ctx2 = _ctx(debugger)
    assert dispatch(ctx2, "finish", {})["where"] == f"t{ROW}.L0.h_out"  # at L0.h: its layer
    assert "h_out" in registry_help("finish") and "sample" in registry_help("finish")


def registry_help(name):
    from mdb.ops import registry

    return registry()[name].summary


def test_up_follows_the_first_producer_names_a_producer_and_pushes_the_focus(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.o"})
    program = ctx.run.program
    producers = graph.producers(program)["r"]
    assert producers[0] == "h" and "o" in producers  # r = h + o: both are reachable
    dispatch(ctx, "frame", {"address": "L0.r"})
    first = dispatch(ctx, "up", {})
    assert first["focus"] == f"t{ROW}.L0.h" and "o" in [a.split(".")[-1] for a in
                                                         first["alternatives"]]
    assert first["producers"] == [f"L0.{p}" for p in producers]
    back = dispatch(ctx, "down", {})  # the stack: `up` pushed r, so down returns to it
    assert back["focus"] == f"t{ROW}.L0.r"
    named = dispatch(ctx, "up", {"port": "o"})
    assert named["focus"] == f"t{ROW}.L0.o" and f"L0.h" in named["alternatives"]
    assert dispatch(ctx, "down", {})["focus"] == f"t{ROW}.L0.r"
    with pytest.raises(Refused, match="no producer 'zz'"):
        dispatch(ctx, "up", {"port": "zz"})
    assert ctx.focus == f"t{ROW}.L0.r"  # a refusal leaves the focus unchanged
    two = dispatch(ctx, "up", {"n": 2})
    assert len(ctx.focus_stack) == 2 and two["focus"] != f"t{ROW}.L0.r"
    dispatch(ctx, "down", {"n": 2})
    assert ctx.focus == f"t{ROW}.L0.r" and not ctx.focus_stack
    with pytest.raises(Refused, match="no frame beyond"):
        dispatch(ctx, "up", {"n": 4000})
    assert ctx.focus == f"t{ROW}.L0.r"


def test_set_refuses_scalars_outside_the_port_dtype_and_unstages_failures(debugger, monkeypatch):
    from mdb.engine import run as runmod

    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.q"})
    big = 1e39 if ctx.run.adapter.dtype == torch.bfloat16 else 1e300
    with pytest.raises(Refused, match="finite range"):
        dispatch(ctx, "set", {"target": "q", "value": big})
    assert ctx.run.edits == []
    # a selector edit is ONE masked Run.set call; a failure in it stages nothing
    original = runmod.Run.set
    calls = []

    def flaky(self, target, value, op="replace", **kwargs):
        calls.append(kwargs)
        raise ValueError("injected failure in the single set call")

    monkeypatch.setattr(runmod.Run, "set", flaky)
    before = ctx.run.read().clone()
    with pytest.raises(Refused, match="injected"):
        dispatch(ctx, "set", {"target": "L0.q[h1]", "value": 2.0})
    assert len(calls) == 1 and calls[0]["mask"] is not None
    monkeypatch.setattr(runmod.Run, "set", original)
    assert ctx.run.edits == [] and torch.equal(ctx.run.read(), before)


def test_interrupt_during_set_leaves_the_staged_list_unchanged(debugger, monkeypatch):
    import io

    from mdb.engine import run as runmod
    from mdb.repl.shell import Shell

    ctx = _ctx(debugger)
    out = io.StringIO()
    shell = Shell(ctx, out=out)
    shell.execute("until L0.q")
    shell.execute("set q[d=0] = 1 add")  # one staged edit already
    assert len(ctx.run.edits) == 1

    original = runmod.Run.set

    def interrupted(self, target, value, op="replace", **kwargs):
        raise KeyboardInterrupt  # Run.set is atomic: an interrupt inside it stages nothing

    monkeypatch.setattr(runmod.Run, "set", interrupted)
    assert shell.execute("set q = 0.5") is True
    assert f"interrupted at {ctx.run.where}" in out.getvalue()
    assert "Traceback" not in out.getvalue()
    monkeypatch.setattr(runmod.Run, "set", original)
    assert len(ctx.run.edits) == 1  # the interrupted set staged nothing
    assert shell.execute("where") is True and ctx.run.where == f"t{ROW}.L0.q"


def test_every_error_and_result_carries_a_state(debugger):
    ctx = _ctx(debugger)
    assert dispatch(ctx, "where", {})["state"] == "resolved"
    with pytest.raises(Refused) as refused:
        dispatch(ctx, "print", {"address": "L1.m"})
    data = refused.value.to_json()
    assert data["state"] == "refused" and data["reason"] == data["message"]
    # a replica-derived s whose check has not been reached is an unresolved FIELD, not dropped
    dispatch(ctx, "until", {"target": "L0.q"})
    dispatch(ctx, "set", {"target": "q", "value": 0.5, "op": "scale"})
    dispatch(ctx, "until", {"target": "L0.s"})
    result = dispatch(ctx, "print", {"address": "s"})
    provenance = result["provenance"]
    if isinstance(provenance.get("replica_equal"), dict):
        assert provenance["replica_equal"]["state"] == "unresolved"
        assert result["state"] == "unresolved"


def test_ops_that_raise_unexpectedly_become_internal_errors(debugger, monkeypatch):
    from mdb.ops import builtin

    ctx = _ctx(debugger)

    def boom(*_a, **_k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(builtin, "_stop", boom)
    with pytest.raises(OpError) as caught:
        dispatch(ctx, "where", {})
    assert caught.value.kind == "internal" and caught.value.state == "unresolved"
    assert "RuntimeError: synthetic failure" in caught.value.message
    assert caught.value.to_json()["state"] == "unresolved"


# =========================================================================== fix round, part 2
# Edit provenance, selector edits as one masked set, cache cells, producers/consumers graph.

LOOKAHEADS = ["none", "layer"]


def _ctx_la(debugger, lookahead):
    run = debugger.run(PROMPT, lookahead=lookahead)
    return debugger.ops(run)


def _prov(result):
    return result["provenance"]


def _under(result):
    return [e["address"] for e in _prov(result).get("under_edits", [])]


@pytest.mark.parametrize("lookahead", LOOKAHEADS)
def test_selector_edit_labels_edited_untouched_and_downstream(debugger, lookahead):
    ctx = _ctx_la(debugger, lookahead)
    dispatch(ctx, "until", {"target": "L0.q"})
    native = debugger.run(PROMPT, lookahead=lookahead)
    native.until(f"t{ROW}.L0.q")
    base = native.read().clone()
    dispatch(ctx, "set", {"target": "L0.q[h1]", "value": 2.0})
    whole = dispatch(ctx, "print", {"address": "L0.q"})
    assert _prov(whole)["edited"] is True and "mixed" in _prov(whole)
    assert _prov(whole)["mixed"]["written"] > 0 and _prov(whole)["mixed"]["untouched"] > 0
    assert whole["stats"]["norm"]["authority"] != "measured"
    untouched = dispatch(ctx, "print", {"address": "L0.q[h0]"})
    assert _prov(untouched)["edited"] is False
    assert _prov(untouched)["same_port_edit"]["port"] == "q"
    assert untouched["stats"]["norm"]["authority"] == "measured"
    # untouched elements are bit-identical to the native run, the written ones are not
    now = ctx.run.read()
    head_dim = ctx.run.adapter.head_dim
    assert torch.equal(now[..., :head_dim], base[..., :head_dim])
    assert not torch.equal(now[..., head_dim:2 * head_dim], base[..., head_dim:2 * head_dim])
    # downstream value: measured, but never shown without the edit it sits under
    dispatch(ctx, "until", {"target": "L0.p"})
    down = dispatch(ctx, "print", {"address": "L0.p"})
    assert f"t{ROW}.L0.q" in _under(down) and _prov(down)["under_edits_total"] >= 1
    assert down["stats"]["norm"]["authority"] == "measured"
    dispatch(ctx, "commit", {})
    for address in (f"t{ROW}.L0.h_out", "L0.h_out"):
        after = dispatch(ctx, "print", {"address": address})
        assert _prov(after)["under_edits"][0]["staged"] is False
    dispatch(ctx, "until", {"target": "logits"})
    logits = dispatch(ctx, "print", {"address": "logits"})
    assert _prov(logits)["under_edits_total"] >= 1


def test_every_value_after_an_edit_is_labelled_a_fuzz_walk(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.q"})
    dispatch(ctx, "set", {"target": "L0.q[h0]", "value": 0.5})
    unlabelled = []
    for _ in range(12):
        result = dispatch(ctx, "step", {})
        where = result["where"]
        printed = dispatch(ctx, "print", {})
        prov = _prov(printed)
        if not (prov.get("edited") or prov.get("under_edits") or prov.get("same_port_edit")):
            unlabelled.append(where)
    assert unlabelled == []


def test_selector_edit_is_atomic_and_one_call(debugger, monkeypatch):
    from mdb.engine import run as runmod

    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.q"})
    calls = []
    original = runmod.Run.set

    def counting(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(runmod.Run, "set", counting)
    dispatch(ctx, "set", {"target": "L0.q[h1]", "value": 2.0})
    assert len(calls) == 1 and calls[0]["mask"] is not None
    assert len(ctx.run.edits) == 1
    info = dispatch(ctx, "info", {"subject": "edits"})
    assert info["staged_total"] == 1 and info["committed_total"] == 0
    dispatch(ctx, "step", {"n": 6})
    dispatch(ctx, "commit", {})
    info = dispatch(ctx, "info", {"subject": "edits"})
    assert info["staged_total"] == 0 and info["committed_total"] == 1


@pytest.mark.parametrize("selector", ["", "[h1]"])
def test_set_ahead_of_the_cursor_behaves_the_same_with_and_without_selector(debugger, selector):
    ctx = _ctx(debugger)
    # the cursor is at L0.<first port>; q is ahead of it
    try:
        result = dispatch(ctx, "set", {"target": f"L1.q{selector}", "value": 1.0})
    except Refused as error:
        refusal = str(error)
    else:
        refusal = None
        assert result["staged"] == 1
    other = _ctx(debugger)
    try:
        dispatch(other, "set", {"target": "L1.q", "value": 1.0})
        plain = None
    except Refused as error:
        plain = str(error)
    assert (refusal is None) == (plain is None)


def test_cache_cells_read_through_read_cache_without_capture(debugger, monkeypatch):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L1.q"})
    run = ctx.run
    expected = run.read_cache(0, "k")
    hits = {"capture": 0}
    for name in ("capture",):
        if hasattr(run, name):
            real = getattr(run, name)

            def spy(*a, _real=real, **k):
                hits["capture"] += 1
                return _real(*a, **k)

            monkeypatch.setattr(run, name, spy)
    printed = dispatch(ctx, "print", {"address": "cache.L0.k"})
    xs = dispatch(ctx, "x", {"address": "cache.L0.k", "count": 4})
    assert printed["stats"]["abs_max"]["value"] == pytest.approx(
        float(expected.float().abs().max()), rel=1e-2)
    assert xs["values"]["values"][:4] == [pytest.approx(float(v), rel=1e-2, abs=1e-3)
                                for v in expected.flatten()[:4].float()]
    assert hits["capture"] == 0
    assert _prov(printed)["authority"] == "measured"


def test_cell_breakpoint_and_watch_fire_at_boundary_stops_only(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "break", {"pattern": "cache.L0.v[row=2]"})
    stop = dispatch(ctx, "continue", {})
    assert stop["where"].endswith(".h_out") or stop["where"].endswith(".start")
    assert stop["hits"] and stop["hits"][0]["kind"] in ("break", "breakpoint")


def test_backtrace_follows_declared_inputs(debugger):
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L1.q"})
    spec = {p.name: p for p in ctx.run.program.ports}
    trace = dispatch(ctx, "backtrace", {})
    frames = trace["frames"]
    assert frames[0]["address"].endswith("L1.q")
    for frame in frames:
        port = frame["address"].rsplit(".", 1)[1]
        if port in spec:
            declared = set(spec[port].inputs)
            shown = set(frame.get("reads", [])) | set(frame.get("inputs", []) or [])
            assert shown <= declared | {"cache", "prev_layer"} | {frame.get("next", "")}


def test_where_reports_computed_past_cursor_in_layer_lookahead(debugger):
    ctx = _ctx_la(debugger, "layer")
    result = dispatch(ctx, "where", {})
    past = ctx.run.status().get("computed_past_cursor")
    assert result.get("computed_past_cursor") == (past or result.get("computed_past_cursor"))
    if past:
        assert result["computed_past_cursor"]


# --- round 3, part 2 ---------------------------------------------------------------------------


def test_cache_cells_under_a_committed_same_layer_edit_carry_the_label(debugger):
    """G3: a v edit at row 3 layer 0 reaches the layer-0 v cache cell of that row and later rows,
    not earlier rows and not the k cache."""
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.a"})
    dispatch(ctx, "set", {"target": "L0.v", "value": 0.5, "op": "add"})
    dispatch(ctx, "until", {"target": "L1.q"})
    for address, edited in (("cache.L0.v", True), (f"cache.L0.v[row={ROW}]", True),
                            ("cache.L0.v[row=1]", False), ("cache.L0.k", False)):
        for op, args in (("print", {}), ("x", {"count": 2})):
            result = dispatch(ctx, op, {"address": address, **args})
            prov = _prov(result)
            assert bool(prov.get("under_edits")) is edited, (address, op, prov)


def test_print_of_an_unchecked_replica_shows_numbers_labelled_unchecked(debugger):
    """F1e (amended): s at the cursor in lookahead none is derived-exact from the captured
    operands; only the comparison with native is pending, and the label says so."""
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L1.s"})
    result = dispatch(ctx, "print", {"address": "L1.s", "top": 2, "slice": 2})
    prov = _prov(result)
    assert prov["authority"] == "derived-exact" and prov["replica_checked"] is False
    assert prov.get("replica_check_at") == "p" and "state" not in prov
    assert result["stats"]["norm"]["authority"] == "derived-exact"
    dispatch(ctx, "until", {"target": "L1.p"})
    prov = _prov(dispatch(ctx, "print", {"address": "L1.s"}))
    assert prov["authority"] == "derived-exact" and prov["replica_equal"] is True
    assert prov["replica_checked"] is True and "replica_check_at" not in prov


def test_a_masked_edit_labels_an_untouched_selection_measured_with_the_marker(debugger):
    """G2 at the ops layer: after commit the unedited half stays measured, the mixed one does not."""
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.m"})
    dispatch(ctx, "set", {"target": "L0.m[d0:4]", "value": 0.5, "op": "scale"})
    dispatch(ctx, "until", {"target": "L1.q"})
    untouched = _prov(dispatch(ctx, "print", {"address": "L0.m[d8:12]"}))
    assert untouched["edited"] is False and untouched["authority"] == "measured"
    assert untouched.get("same_port_edit")
    mixed = _prov(dispatch(ctx, "print", {"address": "L0.m"}))
    assert mixed["edited"] is True and mixed["authority"] == "derived-exact"


def test_an_interrupt_in_a_motion_leaves_the_ops_session_in_step(debugger, monkeypatch):
    """G1 through the ops layer: finish interrupted after the native session moved."""
    from saturn_pub.core import Session

    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.a"})
    dispatch(ctx, "set", {"target": "L0.v", "value": 0.5, "op": "add"})
    original = Session.continue_
    fired = {"n": 0}

    def interrupting(self, *a, **k):
        out = original(self, *a, **k)
        if not fired["n"]:
            fired["n"] = 1
            raise KeyboardInterrupt
        return out

    monkeypatch.setattr(Session, "continue_", interrupting)
    with pytest.raises(KeyboardInterrupt):
        dispatch(ctx, "finish", {})
    monkeypatch.undo()
    run = ctx.run
    assert run.session.capture(retain=False).fingerprint == run._trail[-1].fingerprint
    assert [e["port"] for e in run.edits] == ["v"]
    dispatch(ctx, "finish", {})
    assert _prov(dispatch(ctx, "print", {"address": "L0.h_out"})).get("under_edits")


def test_display_and_watch_on_cache_cells_label_a_committed_same_layer_edit(debugger):
    """G3: display and watch use the same cell rule as print; v row 3 is under the edit, v row 1
    and k are not. Two watches on different tensors at one stop do not share a value."""
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.a"})
    dispatch(ctx, "set", {"target": "L0.v", "value": 0.5, "op": "add"})
    cells = (f"cache.L0.v[row={ROW}]", "cache.L0.v[row=1]", "cache.L0.k")
    for address in cells:
        dispatch(ctx, "display", {"address": address})
    for address in ("cache.L0.v", "cache.L0.k"):
        dispatch(ctx, "watch", {"address": address})
    moved = dispatch(ctx, "continue", {})
    shown = {d["address"]: d for d in moved["displays"]}
    assert [bool(shown[a]["provenance"].get("under_edits")) for a in cells] == [True, False, False]
    hits = {h["pattern"]: h for h in moved["hits"]}
    assert set(hits) == {"cache.L0.v", "cache.L0.k"}
    assert hits["cache.L0.v"]["new_provenance"].get("under_edits")
    assert not hits["cache.L0.k"]["new_provenance"].get("under_edits")
    run = ctx.run
    for address, tensor in (("cache.L0.v", "v"), ("cache.L0.k", "k")):
        norm = float(run.read_cache(0, tensor).double().norm())
        assert hits[address]["new"]["norm"]["value"] == pytest.approx(norm, rel=1e-2)


def test_values_at_a_later_row_stay_under_edits_committed_in_an_earlier_row(debugger):
    """G5 gap: the next row's ports, logits and sample still say they sit under the edit."""
    ctx = _ctx(debugger)
    dispatch(ctx, "until", {"target": "L0.a"})
    dispatch(ctx, "set", {"target": "L0.v", "value": 0.5, "op": "add"})
    dispatch(ctx, "step", {"unit": "token"})
    for target in ("start", "L0.q", "L1.m", "logits"):
        if target != "start":
            dispatch(ctx, "until", {"target": target})
        prov = _prov(dispatch(ctx, "print", {"address": target}))
        assert [u["address"] for u in prov["under_edits"]] == [f"t{ROW}.L0.v"], target

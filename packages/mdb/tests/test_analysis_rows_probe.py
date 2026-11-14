"""B2 verbs: Run.feed (+ feed op), breakpoints.replay over a Trace, rows.row_table and
probe.suffix_sweep, each checked against mdb's stepped native path on a tiny fp32 model."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis import probe, rows, symbols  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.engine.breakpoints import ConditionError, TraceCondition, replay  # noqa: E402
from mdb.ops import Refused, dispatch  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

IDS = [(i * 7) % 31 + 1 for i in range(28)]
LETTERS = [20, 21, 22, 23]
N_PROMPT = 14
SUFFIX = [25, 26, 27]
BINDS = {"sink": [0], "question": list(range(1, 6)), "hint": list(range(6, 10)),
         "template": list(range(10, N_PROMPT))}


@pytest.fixture(scope="module")
def dbg():
    torch.manual_seed(0)
    return open_tiny("qwen2", "fp32")


def _letters(run):
    return run.read("logits").reshape(-1)[LETTERS].double()


def _stepped(dbg, seq, zero_cols=()):
    """mdb native answer-row letters of ``seq``, optionally with the answer row's p on
    ``zero_cols`` zeroed at every layer (masked replace at each p port)."""
    run = dbg.run(seq)
    if zero_cols:
        for layer in range(run.adapter.layers):
            run.until(f"L{layer}.h")
            mask = torch.zeros(run.address_shape(f"L{layer}.p"), dtype=torch.bool)
            mask[..., list(zero_cols)] = True
            run.set(f"L{layer}.p", 0.0, "replace", mask=mask)
    run.until("logits")
    return _letters(run)


# --------------------------------------------------------------------------- feed


def test_feed_equals_prefill_and_labels_supplied(dbg):
    fed = dbg.run(IDS[:6])
    where = fed.feed(IDS[6:12])
    assert where == "t11.start"
    assert fed.supplied_tokens() == {5 + i: IDS[6 + i] for i in range(6)}
    fed.until("logits")
    plain = dbg.run(IDS[:12])
    plain.until("logits")
    assert torch.allclose(fed.read("logits"), plain.read("logits"), atol=1e-5, rtol=0)
    info = fed.read_info("t7.sample")
    assert info["authority"] == "supplied" and info["by"] == "feed"
    assert int(fed.read("t7.sample").reshape(-1)[0]) == IDS[8]
    assert fed.read_info("t11.logits")["authority"] == "measured"


def test_feed_survives_restore_and_replay(dbg):
    run = dbg.run(IDS[:4])
    ck = run.checkpoint()
    run.feed(IDS[4:9])
    run.until("logits")
    want = run.read("logits").clone()
    run.restore(ck)
    assert run.supplied_tokens() == {}
    run.feed(IDS[4:9])
    run.until("logits")
    assert torch.equal(run.read("logits"), want)
    # a past fed decision is re-derived by replay (cut pruned or not) with its supplied id
    assert int(run.read("t5.sample").reshape(-1)[0]) == IDS[6]


def test_feed_refuses_bad_ids_and_op(dbg):
    run = dbg.run(IDS[:4])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        run.feed([10**6])
    with pytest.raises(ValueError, match="at least one"):
        run.feed([])
    ctx = dbg.ops()
    dispatch(ctx, "start", {"prompt": IDS[:4]})
    out = dispatch(ctx, "feed", {"ids": IDS[4:7]})
    assert out["where"] == "t6.start"
    assert out["supplied"] == [f"t3.sample={IDS[4]}", f"t4.sample={IDS[5]}", f"t5.sample={IDS[6]}"]
    with pytest.raises(Refused, match="outside the vocabulary"):
        dispatch(ctx, "feed", {"ids": [10**6]})


# --------------------------------------------------------------------------- replay


def test_trace_condition_whitelist():
    TraceCondition('argmax_letter == X and ablated["hint"] < 0')
    TraceCondition('share["hint"] > share["chain_so_far"] and step >= 3')
    for bad in ("v > 0", 'norm(v) > 1', 'foo["x"] > 0', "share[0] > 1", "__import__('os')"):
        with pytest.raises(ConditionError):
            TraceCondition(bad)


def test_replay_edge_level_crossings_and_undefined():
    recs = []
    pattern = [1, 1, -1, 1, -1, -1]
    for k, sign in enumerate(pattern):
        recs.append({"step": 10 * k, "margin": 1.0, "argmax_letter": "B" if k != 5 else "A",
                     "ablated": {"hint": -0.5 * sign}, "share": {"hint": None if k == 4 else 0.3}})
    edge = replay('argmax_letter == X and ablated["hint"] < 0', recs, X="B", R="A", mode="edge")
    assert edge["holds"] == [True, True, False, True, False, False]
    assert edge["hits"] == [0, 30] and edge["fired_at"] == 0 and edge["crossings"] == 3
    assert edge["fired_fraction"] == pytest.approx(0.5)
    level = replay('share["hint"] > 0.2', recs, X="B", R="A", mode="level")
    assert level["undefined"] == [40] and level["hits"] == [0, 10, 20, 30, 50]
    change = replay('ablated["hint"] < 0', recs, X="B", R="A", mode="change")
    assert change["hits"] == [20, 30, 40]


# --------------------------------------------------------------------------- row_table


def test_row_table_matches_symbols_and_stepped_ablation(dbg):
    run = dbg.run(IDS)
    binds = dict(BINDS, chain=list(range(N_PROMPT, len(IDS))))
    trace = rows.row_table(run, binds, letters=LETTERS, X="B", R="C",
                           arms={"hint": BINDS["hint"]}, chunk=5, check=True)
    assert len(trace.records) == len(IDS)
    assert trace.meta["trace_check_max_abs"] < 1e-9
    for t in (12, 20, len(IDS) - 1):
        native = _stepped(dbg, IDS[: t + 1])
        abl = _stepped(dbg, IDS[: t + 1], BINDS["hint"])
        rec = trace.records[t]
        assert rec["margin"] == pytest.approx(float(native[1] - native[2]), abs=1e-5)
        assert rec["ablated"]["hint"] == pytest.approx(float(abl[1] - abl[2]), abs=1e-5)
        # the symbols table at a stepped row t agrees on every binding's to_consumer
        sub = dbg.run(IDS[: t + 1])
        sub.until("logits")
        for name, cols in binds.items():
            cols = [c for c in cols if c <= t]
            if cols:
                symbols.declare(sub, name, [f"t{c}" for c in cols])
        tab = symbols.table(sub, consumer=Margin(t, LETTERS[1], LETTERS[2]), limit=16)
        for b in tab.report["bindings"]:
            if b["symbol"] in rec["to_consumer"]:
                assert rec["to_consumer"][b["symbol"]] == pytest.approx(
                    b["to_consumer"]["value"], abs=1e-9)


def test_row_table_ablation_inside_the_arm_zeroes_the_rows_own_read(dbg):
    """A row inside the arm's columns reads its own (freshly appended) value cell too; the
    stepped p-zero reference zeroes that diagonal read, so row_table must as well
    (AUDIT-B2: off by 0.0135 at hint rows 6-9 before the fix)."""
    run = dbg.run(IDS[:N_PROMPT])
    trace = rows.row_table(run, dict(BINDS), letters=LETTERS, X="B", R="C",
                           arms={"hint": BINDS["hint"]}, chunk=3)
    for t in BINDS["hint"]:
        abl = _stepped(dbg, IDS[: t + 1], [c for c in BINDS["hint"] if c <= t])
        assert trace.records[t]["ablated"]["hint"] == pytest.approx(
            float(abl[1] - abl[2]), abs=1e-5), t
    # rows before the arm are untouched by it
    for t in range(BINDS["hint"][0]):
        assert trace.records[t]["ablated"]["hint"] == pytest.approx(
            trace.records[t]["margin"], abs=1e-5)


# --------------------------------------------------------------------------- suffix_sweep


def test_suffix_sweep_matches_stepped_arms(dbg):
    run = dbg.run(IDS)
    chain_len = len(IDS) - N_PROMPT
    steps = [0, 5, chain_len]
    trace = probe.suffix_sweep(run, steps, SUFFIX, n_prompt=N_PROMPT, bindings=BINDS,
                               letters=LETTERS, X="B", null_steps=tuple(steps))
    assert [r["step"] for r in trace.records] == steps
    assert trace.meta["null_vs_fresh_max_abs"] < 1e-5
    R = trace.meta["R"]
    ri = "ABCD".index(R)
    for rec in trace.records:
        k = rec["step"]
        seq = IDS[: N_PROMPT + k] + SUFFIX
        last = len(seq) - 1
        nat = _stepped(dbg, seq)
        assert rec["margin"] == pytest.approx(float(nat[1] - nat[ri]), abs=1e-5)
        m2 = _stepped(dbg, seq, BINDS["hint"])
        assert rec["ablated"]["hint"] == pytest.approx(float(m2[1] - m2[ri]), abs=1e-5)
        chain_cols = list(range(N_PROMPT, N_PROMPT + k))
        if chain_cols:
            cm = _stepped(dbg, seq, chain_cols)
            assert rec["ablated"]["chain_so_far"] == pytest.approx(float(cm[1] - cm[ri]),
                                                                   abs=1e-5)
        # M3: the hint rows' V zeroed for every suffix row = fed suffix after zeroing hint V
        base = run.adapter.model
        with torch.no_grad():
            cache = base.model(input_ids=torch.tensor([IDS[: N_PROMPT + k]]),
                               use_cache=True).past_key_values
            for layer in cache.layers:
                layer.values[:, :, BINDS["hint"]] = 0
            h = base(input_ids=torch.tensor([SUFFIX]), past_key_values=cache).logits[0, -1]
        m3 = h[LETTERS].double()
        assert rec["ablated"]["hint_suffix"] == pytest.approx(float(m3[1] - m3[ri]), abs=1e-5)
        # exact shares at the probe answer row equal the symbols table there
        sub = dbg.run(seq)
        sub.until("logits")
        parts = dict(BINDS, chain_so_far=chain_cols,
                     suffix=list(range(N_PROMPT + k, last + 1)))
        for name, cols in parts.items():
            if cols:
                symbols.declare(sub, name, [f"t{c}" for c in cols])
        tab = symbols.table(sub, consumer=Margin(last, LETTERS[1], LETTERS[ri]), limit=16)
        for b in tab.report["bindings"]:
            if b["symbol"] in rec["to_consumer"]:
                assert rec["to_consumer"][b["symbol"]] == pytest.approx(
                    b["to_consumer"]["value"], abs=1e-9)
                assert rec["mass"][b["symbol"]] == pytest.approx(b["mass"]["value"], abs=1e-9)


def test_suffix_sweep_feed_spot_check(dbg):
    """The sweep's native answer row equals mdb stepping the suffix with Run.feed."""
    run = dbg.run(IDS)
    k = 7
    trace = probe.suffix_sweep(run, [k], SUFFIX, n_prompt=N_PROMPT, bindings=BINDS,
                               letters=LETTERS, X="B", R="A", arms=("native",))
    fed = dbg.run(IDS[: N_PROMPT + k] + SUFFIX[:1])
    fed.feed(SUFFIX[1:])
    fed.until("logits")
    lg = _letters(fed)
    assert trace.records[0]["margin"] == pytest.approx(float(lg[1] - lg[0]), abs=1e-5)


def test_sweep_refusals(dbg):
    run = dbg.run(IDS)
    with pytest.raises(Exception, match="probe steps"):
        probe.suffix_sweep(run, [99], SUFFIX, n_prompt=N_PROMPT, bindings=BINDS,
                           letters=LETTERS, X="B")
    with pytest.raises(Exception, match="overlap"):
        probe.suffix_sweep(run, [1], SUFFIX, n_prompt=N_PROMPT,
                           bindings=dict(BINDS, extra=[6]), letters=LETTERS, X="B")


def test_feed_from_a_sample_stop_replaces_the_greedy_decision(dbg):
    run = dbg.run(IDS[:5])
    run.until("sample")
    greedy = int(run.read("sample").reshape(-1)[0])
    token = IDS[5] if IDS[5] != greedy else IDS[6]
    run.feed([token])
    assert run.where == "t5.start" and run.supplied_tokens() == {4: token}
    assert int(run.read("t4.sample").reshape(-1)[0]) == token
    ck = run.checkpoint()
    run.restore(run.checkpoint())
    run.reverse_step("token", 1)
    assert run.where == "t4.start" and run.supplied_tokens() == {}
    assert run.restore(ck) == "t5.start"

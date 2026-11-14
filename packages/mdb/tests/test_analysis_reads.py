"""``analysis.reads`` on tiny Qwen2/Qwen3 (BF16 and FP32).

Exactness: per head, the per-source parts sum to the head output and the head writes sum to
``o`` (float64 identities), and the native ``A`` and ``o`` sit within the stated rounding bound
of them. Causal verdicts equal a fresh run with the same ``p`` edit committed, bit for bit. A
read with small mass and large effect and one with large mass and no effect (both constructed
with edits: the tiny random models attend almost uniformly) are labelled correctly.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.analysis.propagate import state_digest  # noqa: E402
from mdb.analysis.reads import head_mask, head_mix, reads  # noqa: E402
from mdb.core.pointers import AUTHORITIES, check_bounded  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

PROMPT = [1, 5, 9, 3]
CONFIGS = [("qwen2", "bf16"), ("qwen2", "fp32"), ("qwen3", "bf16"), ("qwen3", "fp32")]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: "-".join(c))
def debugger(request):
    return open_tiny(*request.param)


def _at_logits(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.logits")
    return run


def test_value_mix_and_head_write_close_per_head_and_within_the_rounding_bound(debugger):
    run = _at_logits(debugger)
    for layer in range(run.adapter.layers):
        mix = head_mix(run, 3, layer)
        exact = mix.head_out()
        assert torch.allclose(mix.source_parts().sum(1), exact, rtol=0, atol=1e-15)
        heads, dh = exact.shape
        blocks = mix.weight.reshape(-1, heads, dh)
        writes = sum(blocks[:, h, :] @ exact[h] for h in range(heads))
        assert torch.allclose(mix.o_parts().sum((0, 1)), writes, rtol=0, atol=1e-14)
        checks = mix.checks()
        for name in ("value_mix", "head_write"):
            assert bool((checks[name]["eta"].abs() <= checks[name]["bound"]).all()), name
    report = reads(run, "t1", causal=False).report
    assert report["checks"]["value_mix"]["within_bound"] is True
    assert report["checks"]["head_write"]["within_bound"] is True


def _fresh_margin(debugger, edits, margin):
    run = debugger.run(PROMPT)
    for address, value, op, mask in edits:
        entry = address.rsplit(".", 1)[0] + ".h"
        if run.parse(run.where) < run.parse(entry):
            run.until(entry)
        run.set(address, value, op, mask=mask)
    run.until(f"t{margin.row}.logits")
    logits = run.read(f"t{margin.row}.logits").double().reshape(-1)
    return logits[margin.a] - logits[margin.b]


def _rule(delta, entry):
    """The P20 rule: floor = max(|residual|, 2 x spacing); causal when |native| > floor and
    exact has its sign with |exact| > |residual|; not causal when native and exact are both
    within the floor; else unresolved. ``delta`` is the fresh run's change."""
    effect = entry["causal"]["effect"]
    assert effect["native"]["value"] == delta and effect["native"]["authority"] == "measured"
    floor, exact = effect["floor"]["value"], effect["exact"]["value"]
    residual = effect["residual"]["value"]
    assert floor == max(abs(residual), 2 * effect["spacing"]["value"])
    assert "bound" not in str(effect["floor"]["authority"])
    if abs(delta) > floor:
        same = exact != 0 and (exact > 0) == (delta > 0)
        return "causal" if same and abs(exact) > abs(residual) else "unresolved"
    return "not causal at this dose" if abs(exact) <= floor else "unresolved"


def test_causal_verdict_equals_a_fresh_committed_run_bitwise(debugger):
    run = _at_logits(debugger)
    margin = Margin(3, 4, 9)
    before = state_digest(run)
    result = reads(run, "t2", consumer=margin, top=2)
    assert state_digest(run) == before
    base = _fresh_margin(debugger, [], margin)
    for entry in result.report["reads"]:
        layer, head = entry["layer"], int(entry["name"].rsplit(".h", 1)[1])
        mask = head_mask(run, 3, layer, head, 2)
        edited = _fresh_margin(debugger, [(f"t3.L{layer}.p", 0.0, "replace", mask)], margin)
        assert entry["causal"]["delta"]["value"] == float(edited - base)
        assert entry["causal"]["verdict"] == _rule(float(edited - base), entry)
        if entry["causal"]["verdict"] == "unresolved":
            assert entry["causal"]["reason"].startswith("changed, ")


def test_reads_at_other_rows_with_causal_verdicts_equal_fresh_runs(debugger):
    """rows=all with the default causal verdict (the p edit in an earlier row's frame)."""
    run = debugger.run(PROMPT)
    run.until("t5.logits")
    margin = Margin(5, 4, 9)
    result = reads(run, "t3", rows="all", consumer=margin, top=32)
    assert {e["row"] for e in result.report["reads"]} == {3, 4, 5}
    base = _fresh_margin(debugger, [], margin)
    for entry in result.report["reads"]:
        j, layer = entry["row"], entry["layer"]
        head = int(entry["name"].rsplit(".h", 1)[1])
        mask = head_mask(run, j, layer, head, 3)
        edited = _fresh_margin(debugger, [(f"t{j}.L{layer}.p", 0.0, "replace", mask)], margin)
        assert entry["causal"]["delta"]["value"] == float(edited - base), entry["name"]
        assert entry["causal"]["verdict"] == _rule(float(edited - base), entry)


def test_reads_rank_by_effect_sum_the_rest_and_are_bounded(debugger):
    run = _at_logits(debugger)
    result = reads(run, "t0", consumer=Margin(3, 4, 9), top=2, causal=False)
    report = check_bounded(result.report)
    assert report["total"] == run.adapter.layers * run.adapter.num_heads
    sizes = [abs(e["to_consumer"]["value"]) for e in report["reads"]]
    assert sizes == sorted(sizes, reverse=True)
    values = [e["_value"] for e in result.tensors["entries"]]
    total = float(sum(values[2:], values[0] * 0))
    assert report["omitted"]["count"] == len(values) - 2
    assert report["omitted"]["to_consumer"]["value"] == pytest.approx(total, abs=1e-15)
    assert all(e["causal"]["verdict"] == "unresolved" for e in report["reads"])
    # a cache-cell source restricts the reads to its layer
    cell = reads(run, "cache.L1.v[row=0]", causal=False).report
    assert {e["layer"] for e in cell["reads"]} == {1}


def test_small_mass_large_effect_and_large_mass_no_effect_are_labelled(debugger):
    """Constructed at L1 of row 3 (prompt [1, 5]: rows 1-3 stepped): score of (h0, src 3)
    lowered by 3 and row 3's own value scaled by 50 (small mass, large effect); score of
    (h1, src 1) raised by 8 with row 1's L1 value scaled to 0 earlier (large mass, no effect).
    No edit lies on the path of the p edits, so each verdict has its rounding bound."""
    run = debugger.run([1, 5])
    run.until("t1.L1.h")
    run.set("t1.L1.v", 0.0, "scale")
    run.until("t3.L1.h")
    s_shape = tuple(run.port_shape("s"))
    low = torch.zeros(s_shape, dtype=torch.bool)
    low[0, 0, 0, 3] = True
    high = torch.zeros(s_shape, dtype=torch.bool)
    high[0, 1, 0, 1] = True
    run.set("t3.L1.s", -3.0, "add", mask=low)
    run.set("t3.L1.s", 8.0, "add", mask=high)
    run.set("t3.L1.v", 50.0, "scale")
    run.until("t3.logits")
    entry = next(e for e in reads(run, "t3.L1.v", top=8).report["reads"]
                 if e["name"] == "t3.L1.h0")
    assert entry["mass"]["authority"] in AUTHORITIES
    assert entry["causal"]["verdict"] == "causal"
    assert entry["causal"]["effect"]["verdict"] == "effect"
    # a read of another row by the same head: more than twice the mass, a smaller effect
    normal = next(e for e in reads(run, "t2.L1.v", top=8).report["reads"]
                  if e["name"] == "t3.L1.h0")
    assert normal["mass"]["value"] > 2 * entry["mass"]["value"]
    assert (entry["causal"]["delta"]["norm"]["value"]
            > normal["causal"]["delta"]["norm"]["value"])
    big = reads(run, "t1.L1.v", top=8).report["reads"]
    head1 = next(e for e in big if e["name"] == "t3.L1.h1")
    assert head1["mass"]["value"] > 0.9
    assert head1["causal"]["verdict"] == "not causal at this dose"
    assert head1["causal"]["delta"]["max_abs"]["value"] == 0.0
    assert head1["to_consumer"]["max_abs"]["value"] == 0.0


def test_an_edit_that_blocks_the_value_mix_leaves_the_verdict_unmeasured(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.h")
    run.set("t3.L1.A", 0.0, "replace", mask=head_mask(run, 3, 1, 1))
    run.until("t3.logits")
    before = run.checkpoint()
    entry = next(e for e in reads(run, "t1", consumer=Margin(3, 4, 9), top=8).report["reads"]
                 if e["name"] == "t3.L1.h1")
    assert entry["to_consumer"]["state"] == "refused"
    assert entry["causal"]["verdict"] == "unresolved"
    assert "written by an edit" in entry["causal"]["reason"]
    assert run.checkpoint()["path_edits"] == before["path_edits"]


def test_mid_row_reads_list_uncomputed_layers_and_refusals(debugger):
    run = debugger.run(PROMPT)
    run.until("t3.L1.h")
    report = reads(run, "t1", causal=False).report
    assert {e["layer"] for e in report["reads"]} == {0}
    assert any("t3.L1" in item for item in report["not_computed"])
    with pytest.raises(Refusal, match="has not been computed"):
        reads(run, "t1", rows="4")
    with pytest.raises(Refusal, match="a read source"):
        reads(run, "t1.L0.q")
    with pytest.raises(Refusal, match="no layer"):
        reads(run, "t1.L9.v")
    late = reads(run, "t1", consumer=Margin(3, 4, 9), causal=False).report["reads"]
    assert all(e["to_consumer"]["state"] == "refused" for e in late)
    assert "has not been computed" in late[0]["to_consumer"]["reason"]


def test_causal_verdicts_share_one_excursion_and_equal_one_whatif_per_read():
    """top=6 reads on a 12-layer tiny model: each verdict (delta, effect, record id) is bit for
    bit what a solo ``whatif`` of that read gives, the session ends where it started, and the
    whole call makes at most 70 native layer calls (one whatif per read: 127; measured 62)."""
    from mdb.analysis.reads import layer_module, verdict
    from mdb.engine.models import tiny_model
    from mdb.ops.debugger import attach

    run = attach(tiny_model("qwen2", "bf16", layers=12)).run(PROMPT)
    run.until("t3.logits")
    margin = Margin(3, 4, 9)
    calls = [0]
    hooks = [layer_module(run.adapter, i).register_forward_hook(
        lambda *_: calls.__setitem__(0, calls[0] + 1)) for i in range(run.adapter.layers)]
    before = state_digest(run)
    batched = reads(run, "t1", consumer=margin, top=6)
    used = calls[0]
    for hook in hooks:
        hook.remove()
    assert state_digest(run) == before
    assert used <= 70, used
    listed = batched.tensors["entries"][:6]
    assert [e["name"] for e in batched.report["reads"]] == [e["name"] for e in listed]
    assert len({e["layer"] for e in listed}) >= 3 and len({e["name"] for e in listed}) == 6
    for entry in listed:
        j, layer, head, source = entry["_mask"]
        solo = verdict(run, f"t{j}.L{layer}.p", head_mask(run, j, layer, head, source),
                       f"h{head},src={source}", entry["_target"], 0.0)
        assert entry["causal"] == solo, entry["name"]
    assert state_digest(run) == before

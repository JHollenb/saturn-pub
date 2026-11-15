"""Branches (gap audit P1) on tiny Qwen2 (bf16) and Qwen3 (fp32).

Every arm is measured from one checkpoint against one floor; an arm equals the same edit made by
hand on a fresh run; the self-patch null is exactly 0; the random null's change has the arm's
norm; the answer null is refused off the residual stream; a held arm equals ``Run.hold`` by hand;
batched cell arms agree with their stepped re-check up to the measured kernel delta; the record
seals and verifies with custody; the run is back at the branch point afterwards."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from mdb.analysis.branches import Branches, edit  # noqa: E402
from mdb.analysis.finite import f64  # noqa: E402
from mdb.analysis.forecast import Margin, Refusal  # noqa: E402
from mdb.core.records import verify_record  # noqa: E402
from mdb.ops.debugger import open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7]
D = [2, 8, 4, 6, 3]
LAST = 4


@pytest.fixture(scope="module", params=[("qwen2", "bf16"), ("qwen3", "fp32")],
                ids=lambda c: "-".join(c))
def dbg(request):
    return open_tiny(*request.param)


@pytest.fixture(scope="module")
def donor(dbg):
    run = dbg.run(D, name="donor")
    run.until(f"t{LAST}.logits")
    return run


def _margin(logits, a=3, b=7):
    lg = f64(logits).reshape(-1)
    return float(lg[a] - lg[b])


def test_arms_match_hand_runs_and_nulls(dbg, donor):
    run = dbg.run(P, name="br")
    br = dbg.branches(run, Margin(LAST, 3, 7), at=f"t{LAST}.L0.h")
    assert run.where == f"t{LAST}.L0.h" and br.floor > 0
    br.add("M", [edit(f"t{LAST}.L0.m", donor.ref(f"t{LAST}.L0.m"))])
    br.add("A1", [edit(f"t{LAST}.L0.A[h1]", donor.ref(f"t{LAST}.L0.A[h1]"))])
    br.add_null("self", "self", like="A1")
    br.add_null("rnd", "random", like="M", seed=5)
    br.add_null("ans", "answer", like="M")
    with pytest.raises(Refusal):
        br.add_null("bad", "answer", like="A1")                 # A is not the residual stream
    assert "bad" not in br.arms
    rows = {r["name"]: r for r in br.run()}
    assert run.where == f"t{LAST}.L0.h" and run.edits == []    # back at the branch point
    hand = dbg.run(P, name="hand")
    hand.until(f"t{LAST}.L0.h")
    hand.set(f"t{LAST}.L0.m", donor.read(f"t{LAST}.L0.m"))
    hand.until(f"t{LAST}.logits")
    assert rows["M"]["M"]["value"] == _margin(hand.read(f"t{LAST}.logits"))
    assert rows["self"]["dM"]["value"] == 0.0
    assert {r["floor"]["value"] for r in rows.values()} == {br.floor}   # one floor
    assert rows["M"]["edits"][0]["origin"]["origin"] == "donor"
    # the random null's change has the M arm's norm (float64, before the one cast)
    change = br._change(br.arm("M").edits[0])["delta"]
    null = br.arm("rnd").edits[0].value
    assert abs(float(f64(null).norm()) - float(change.norm())) <= 1e-2 * float(change.norm())
    assert rows["ans"]["edits"][0]["origin"]["null"] == "answer"
    for name in ("M", "A1", "rnd", "ans"):
        assert rows[name]["stability"] in ("stable", "kernel-sensitive")
        assert rows[name]["kernel_path"] == "mdb-stepped"
    cmp = br.compare("M", "self")
    assert cmp["difference"]["value"] == rows["M"]["dM"]["value"]
    rec = br.seal()
    verify_record(rec)
    assert rec["body"]["custody"]["kernel_path"] == "mdb-stepped"
    assert rec["body"]["n_arms"] == 5
    assert [r["name"] for r in br.table()][0] == "base"


def test_held_arm_equals_hand_hold(dbg, donor):
    run = dbg.run(P, name="bh")
    br = Branches(run, Margin(LAST, 3, 7), at=f"t{LAST}.L0.h")
    e = edit(f"t{LAST}.L0.m", donor.ref(f"t{LAST}.L0.m"))
    br.add("held", [e], holds=["t@last.L>0.p", "t@last.L>0.norm_scale"])
    row = br.run()[0]
    assert row["holds"] and row["holds"][0]["edits_fired"] == 3 * (dbg.adapter.layers - 1)
    assert run.holds == []                                     # released after the arm
    base = dbg.run(P, name="bh-base")
    base.until(f"t{LAST}.logits")
    hand = dbg.run(P, name="bh-hand")
    hand.hold("t@last.L>0.p", source=base)
    hand.hold("t@last.L>0.norm_scale", source=base)
    hand.until(f"t{LAST}.L0.h")
    hand.set(f"t{LAST}.L0.m", donor.read(f"t{LAST}.L0.m"))
    hand.until(f"t{LAST}.logits")
    assert row["M"]["value"] == _margin(hand.read(f"t{LAST}.logits"))


def test_batched_cell_arms_with_stepped_recheck(dbg, donor):
    run = dbg.run(P, name="bb")
    br = Branches(run, Margin(LAST, 3, 7))                     # at t4.L0.h, the first port
    for r in range(3):
        cell = f"cache.L1.v[kv0,row={r}]"
        br.add(f"c{r}", [edit(cell, donor.ref(cell))])
    rows = br.run(mode="batched", recheck=2)
    assert all(r["kernel_path"] == "batched" for r in rows)
    rechecked = [r for r in rows if "stepped" in r]
    assert len(rechecked) == 2
    for r in rechecked:
        # measured 2026-11-14: bf16 0 exactly; fp32 <= 5.96e-8 = the floor (2 spacings + 2 x null)
        assert abs(r["kernel_delta"]["value"]) <= r["floor"]["value"]
    # the null delta over the margin's own tokens is part of the whole-vocabulary one
    for r in rows:
        own = r["kernel_delta_null_consumer_tokens"]["value"]
        assert 0.0 <= own <= r["kernel_delta_null"]["value"]
    with pytest.raises(Refusal):
        br.add("p", [edit(f"t{LAST}.L0.m", 0.0, "scale")])
        br.run(["p"], mode="batched")


def test_interchange_rides_on_branches(dbg):
    from mdb.analysis import subspace

    dh = dbg.adapter.head_dim
    full = subspace.subspace_binding("f", 1, 0, torch.eye(dh, dtype=torch.float64))
    src = torch.randn(dh, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    out = subspace.interchange(dbg, full, P, 2, src, Margin(LAST, 3, 7), keep=True)
    assert out["M"]["value"] == out["M_whole"]["value"]       # full basis = whole cell
    assert [r["name"] for r in out["arms"]] == ["base", "M", "M_whole"]
    br = out["branches"]
    assert out["floor"]["value"] == br.floor
    br.add_null("self", "self", like="M")
    assert br.run(["self"])[0]["dM"]["value"] == 0.0


def test_arm_magnitude_and_head_selector(dbg):
    """An arm row carries the consumer's |logit| size on that arm (a floor over an item's arms),
    and a head selector edit ``A[h1] = 0`` equals the hand-built head mask."""
    run = dbg.run(P, name="mag")
    br = dbg.branches(run, Margin(LAST, 3, 7), at=f"t{LAST}.L0.h")
    br.add("A1", [edit(f"t{LAST}.L0.A[h1]", 0.0)])
    row = br.run()[0]
    hand = dbg.run(P, name="hand-mag")
    hand.until(f"t{LAST}.L0.h")
    shape, dh = hand.port_shape("A"), dbg.adapter.head_dim
    mask = torch.zeros(shape, dtype=torch.bool)
    mask.reshape(-1)[dh:2 * dh] = True
    hand.set(f"t{LAST}.L0.A", torch.zeros(shape, dtype=dbg.adapter.dtype), mask=mask)
    hand.until(f"t{LAST}.logits")
    lg = hand.read(f"t{LAST}.logits").reshape(-1)
    assert row["M"]["value"] == _margin(lg)
    assert row["magnitude"]["value"] == max(abs(float(lg[3].float())), abs(float(lg[7].float())))
    assert row["edits"][0]["selection"] == "h1"

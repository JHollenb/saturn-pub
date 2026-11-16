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
    with pytest.raises(Refusal):                               # a cell arm replaces whole cells
        br.add("p", [edit("cache.L1.v[kv0,row=1]", 0.0, "scale")])
        br.run(["p"], mode="batched")
    with pytest.raises(Refusal):                               # h is not a native-module port
        br.add("h", [edit(f"t{LAST}.L1.h", 0.0, "scale")])
        br.run(["h"], mode="batched")


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


# --- gap fixes 2026-11-14: batched port arms, flat cell value (EX1), the recheck band ----------

def _port_arms(dbg, donor, br, at_layer):
    """f-port arms of every edit kind: a donor block ref, a selector scale, a masked add, two
    layers in one arm, and an m-port write."""
    dm, ff = dbg.adapter.model.config.hidden_size, dbg.adapter.model.config.intermediate_size
    g = torch.Generator().manual_seed(3)
    L = at_layer
    mask = torch.zeros(1, 1, ff, dtype=torch.bool)
    mask[..., ::3] = True
    br.add("ref", [edit(f"t{LAST}.L{L}.f[i=2:9]", donor.ref(f"t{LAST}.L{L}.f[i=2:9]"))])
    br.add("scale", [edit(f"t{LAST}.L{L}.f[i=3|5]", 0.0, "scale")])
    br.add("mask", [edit(f"t{LAST}.L{L}.f", torch.randn(1, 1, ff, generator=g) * 0.4, "add",
                         mask=mask)])
    br.add("two", [edit(f"t{LAST}.L{L}.f[i=1|4]", 0.5, "add"),
                   edit(f"t{LAST}.L1.f[i=7]", 0.0, "replace")])
    br.add("m", [edit(f"t{LAST}.L{L}.m", torch.randn(1, 1, dm, generator=g) * 0.5, "add")])


@pytest.mark.parametrize("at", [None, f"t{LAST}.L1.h"], ids=["at-L0", "at-L1"])
def test_batched_port_arms_equal_stepped(dbg, donor, at):
    """f (and m) port arms run batched equal the same arms stepped within the arm's floor, with
    each arm's own mask, at the first port and from a later branch point."""
    run = dbg.run(P, name="bp")
    br = Branches(run, Margin(LAST, 3, 7), at=at)
    _port_arms(dbg, donor, br, 1)
    rows = br.run(mode="batched", recheck=0)
    assert all(r["kernel_path"] == "batched" and not r["recheck"]["done"] for r in rows)
    batched = {r["name"]: r["dM"]["value"] for r in rows}
    assert run.where == br.at                                   # back at the branch point
    stepped = {r["name"]: r["dM"]["value"] for r in br.run_arms(again=True)}
    for r in rows:
        assert abs(batched[r["name"]] - stepped[r["name"]]) <= r["floor"]["value"], r["name"]
    assert any(abs(v) > 0 for v in stepped.values())
    # a masked arm touches only its mask: the batched arm differs from the unmasked add
    assert batched["mask"] != batched["m"]


def test_batched_port_arms_equal_nnsight_batch(dbg, donor):
    """The same f/m arms through nnsight's batched patch (interop.nn.batch_whatif) agree with
    the Branches batched path within the arm's floor."""
    pytest.importorskip("nnsight")
    import warnings

    from mdb.interop import nn

    warnings.filterwarnings("ignore")
    run = dbg.run(P, name="nnb")
    br = Branches(run, Margin(LAST, 3, 7))
    dm = dbg.adapter.model.config.hidden_size
    vec = torch.randn(dm, generator=torch.Generator().manual_seed(4)) * 0.5
    br.add("scale", [edit(f"t{LAST}.L1.f[i=3|5]", 0.0, "scale")])
    br.add("two", [edit(f"t{LAST}.L0.f[i=1|4]", 0.5, "add"),
                   edit(f"t{LAST}.L1.f[i=7]", 0.0, "replace")])
    br.add("m", [edit(f"t{LAST}.L1.m", vec.reshape(1, 1, dm).to(dbg.adapter.dtype), "add")])
    rows = {r["name"]: r for r in br.run(mode="batched", recheck=0)}
    arms = {"scale": [nn.Edit("L1.f[i=3|5]", 0.0, "scale")],
            "two": [nn.Edit("L0.f[i=1|4]", 0.5, "add"), nn.Edit("L1.f[i=7]", 0.0, "replace")],
            "m": [nn.Edit("L1.m", vec.reshape(1, dm).to(dbg.adapter.dtype), "add")]}
    res = nn.batch_whatif(dbg.adapter.model, [P], [(3, 7)], arms)
    for name, row in rows.items():
        assert abs(res["arms"][name]["delta"][0] - row["dM"]["value"]) <= row["floor"]["value"], name


def test_flat_cell_value_accepted_by_both_paths(dbg, donor):
    """EX1: a flat (d,) value for one cell is normalised to the block shape when the arm is
    created, so the batched path and the stepped recheck accept the same arm."""
    run = dbg.run(P, name="flat")
    br = Branches(run, Margin(LAST, 3, 7))
    d = dbg.adapter.head_dim
    cell = "cache.L1.v[kv0,row=1]"
    flat = donor.ref(cell).value.reshape(-1)
    assert flat.shape == (d,)
    br.add("flat", [edit(cell, flat)])
    br.add("ref", [edit(cell, donor.ref(cell))])
    br.add("flatref", [edit(cell, type(donor.ref(cell))(flat, donor.ref(cell).pointer))])
    assert tuple(br.arm("flat").edits[0].value.shape) == (1, 1, 1, d)
    assert br.arm("flatref").edits[0].value.pointer["shape"] == [1, 1, 1, d]
    rows = {r["name"]: r for r in br.run(mode="batched", recheck="all")}
    assert all(r["recheck"]["done"] and "stepped" in r for r in rows.values())
    assert rows["flat"]["stepped"]["dM"] == rows["ref"]["stepped"]["dM"]
    assert rows["flat"]["dM"] == rows["ref"]["dM"] == rows["flatref"]["dM"]
    stepped = br.run_arms(["flat"], again=True)[0]                # the stepped path alone
    assert stepped["dM"] == rows["flat"]["stepped"]["dM"]


def test_recheck_band_default_skips_stable_and_null(dbg, donor):
    """90 m-port writes of geometrically growing size (null, in-band and stable arms alike): by
    default exactly the arms with 1/k <= |dM|/floor <= k are re-measured stepped, each row says
    why, and the other options (0, 'all', int k) behave as documented."""
    run = dbg.run(P, name="band")
    br = Branches(run, Margin(LAST, 3, 7))
    dm = dbg.adapter.model.config.hidden_size
    v = torch.randn(dm, generator=torch.Generator().manual_seed(11), dtype=torch.float64)
    for i in range(90):
        value = (v * 1e-10 * 1.4 ** i).to(dbg.adapter.dtype).reshape(1, 1, dm)
        br.add(f"g{i}", [edit(f"t{LAST}.L1.m", value, "add")])
    rows = br.run(mode="batched")                                 # the default is "band"
    k = rows[0]["k"]["value"]
    inband = [1 / k <= r["effect_floors"]["value"] <= k for r in rows]
    assert [r["recheck"]["done"] for r in rows] == inband
    assert [("stepped" in r) for r in rows] == inband
    nulls = [r for r in rows if r["effect_floors"]["value"] < 1 / k]
    stable = [r for r in rows if r["effect_floors"]["value"] > k]
    assert nulls and stable and any(inband)
    assert all("clearly null" in r["recheck"]["why"] for r in nulls)
    assert all("above the band" in r["recheck"]["why"] for r in stable)
    assert all("kernel-sensitive band" in r["recheck"]["why"] for r in rows if r["recheck"]["done"])
    for r in rows:
        if "stepped" in r:                                       # the recheck is still within floor
            assert abs(r["kernel_delta"]["value"]) <= r["floor"]["value"]
    names = [r["name"] for r in rows]
    for option, count in ((0, 0), (None, 0), ("all", len(rows)), (3, 3)):
        got = br.run(mode="batched", recheck=option)
        assert [r["name"] for r in got] == names
        assert sum(r["recheck"]["done"] for r in got) == count, option
        assert all(r["recheck"]["why"] for r in got)
    top = br.run(mode="batched", recheck=3)
    largest = sorted(top, key=lambda r: -abs(r["dM"]["value"]))[:3]
    assert all(r["recheck"]["done"] for r in largest)             # int k: the k largest |dM|
    with pytest.raises(Refusal):
        br.run(mode="batched", recheck="sometimes")


def test_probes_read_each_arm_before_restore(dbg, donor):
    """A probe reads a value other than the consumer on every stepped arm (amplitude programme):
    the self-patch null probes the base value; an upstream h edit changes a later cache cell and
    the probe equals the same read on a hand run."""
    def cell(run):
        return f64(run.read_cache(1, "v", staged=False)[0, 0, 2]).reshape(-1).tolist()

    src = dbg.run(D, name="pr-donor", prefill=2)
    src.until(f"t{LAST}.logits")
    run = dbg.run(P, name="pr", prefill=2)
    br = Branches(run, Margin(LAST, 3, 7), at="t2.L0.h", probes={"v": cell, "n": lambda r: 1.5})
    br.add("H", [edit("t2.L0.h", src.ref("t2.L0.h"))])
    br.add_null("self", "self", like="H")
    rows = {r["name"]: r for r in br.run()}
    base = [x["value"] for x in br.base_probes["v"]]
    assert [x["value"] for x in rows["self"]["probes"]["v"]] == base
    hand = dbg.run(P, name="pr-hand", prefill=2)
    hand.until("t2.L0.h")
    hand.set("t2.L0.h", src.read("t2.L0.h"))
    hand.until(f"t{LAST}.logits")
    assert [x["value"] for x in rows["H"]["probes"]["v"]] == cell(hand) != base
    assert rows["H"]["probes"]["n"]["value"] == 1.5
    rec = br.seal()
    verify_record(rec)
    assert rec["body"]["probes"] == ["n", "v"] and "probes" in rec["body"]["base"]


def test_seal_large_branches_by_pointer(dbg, donor):
    """Many arms with probes exceed one record: the record keeps a compact arm table and the full
    rows by pointer (amplitude programme, arm B had 33 arms x 6 probes; under the 1 MiB v3 bound
    that fits, so this uses 25 arms x 16 probes x 64 values)."""
    from mdb.core.bounded import load_json

    run = dbg.run(P, name="big", prefill=2)
    br = Branches(run, Margin(LAST, 3, 7), at="t2.L0.h",
                  probes={f"p{i}": (lambda r: [0.125] * 64) for i in range(16)})
    for i in range(24):
        br.add_null(f"n{i}", "self", like=br.add("H", []).name) if i == 0 else \
            br.add_null(f"n{i}", "random", like="H", seed=i)
    br.run()
    rec = br.seal()
    verify_record(rec)
    assert rec["body"].get("arms_compact") is True
    full = load_json(rec["body"]["arms_payload"], run=run)
    assert len(full) == len(rec["body"]["arms"]) == 25 and "probes" in full[0]


def test_compact_branches_payload_sealed_in_sink(dbg, donor, tmp_path):
    """Amplitude audit 6a: a compact branches record's ``arms_payload`` is ``state: absent`` (no
    store), so per-arm probe values were in no sealed file. With a sink open the payload bytes
    are written next to the record and read back sha256-checked."""
    from mdb.core import custody

    run = dbg.run(P, name="big-sink", prefill=2)
    br = Branches(run, Margin(LAST, 3, 7), at="t2.L0.h",
                  probes={f"p{i}": (lambda r: [0.125] * 64) for i in range(16)})
    br.add("H", [])
    for i in range(24):
        br.add_null(f"n{i}", "random", like="H", seed=i)
    br.run()
    with custody.records_to(tmp_path / "records-{run}.jsonl") as sink:
        rec = br.seal()
    assert rec["body"]["arms_compact"] is True
    assert rec["body"]["arms_payload"]["state"] == "absent"
    assert [r["id"] for r in sink.records()] == [rec["id"]]
    full = sink.load_json(rec["body"]["arms_payload"])
    import json

    measured = [a.result for a in br.arms.values() if a.result is not None]
    assert len(full) == 25 and sorted(full[0]["probes"]) == sorted(f"p{i}" for i in range(16))
    assert full == json.loads(json.dumps(measured))              # every arm's probe values

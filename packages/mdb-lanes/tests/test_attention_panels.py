"""One panel generator for attention experiments (framework-free; tiny tensors only for the blend)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mdb_lanes import attention_panels as P
from mdb_lanes import attention_port as ap

FLUX = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
HEADFREE = FLUX / "runs/headfree/plan.json"
HEADFREE_CAPTURE = FLUX / "runs/headfree/result/capture.cat.4242.npz"
HEADFREE_METRICS = FLUX / "runs/headfree/result/metrics.json"

UNIVERSE = [(b, h) for b in range(4) for h in range(3)]          # 4 blocks x 3 heads
STRATA = tuple("joint" if b < 2 else "single" for b, _ in UNIVERSE)


def tiny_sets():
    return {"red": {"own2": [(0, 0), (2, 1)], "own4": [(0, 0), (2, 1), (1, 2), (3, 0)]},
            "ball": {"own2": [(1, 1), (3, 2)], "own4": [(1, 1), (3, 2), (0, 2), (2, 0)]}}


def tiny_spec(**kw):
    base = dict(items=("red", "ball"), universe=UNIVERSE, sets=tiny_sets(), contexts=("p|4242", "p|9001"),
                cut_sets=("own2",), keep_sets=("own2",), random=P.RandomControls(n=1, strata=STRATA, seed=3),
                nested=("own2", "own4"))
    base.update(kw)
    return P.ReadPanelSpec(**base)


# ------------------------------------------------------------------------------------- the record

def test_module_import_loads_no_framework():
    code = ("import sys; import mdb_lanes.attention_panels; "
            "print(sorted(m for m in ('torch', 'numpy', 'diffusers') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_arm_record_round_trips_through_json_and_refuses_unknown_kinds():
    panel = P.read_panel(tiny_spec())
    for arm in panel.arms:
        assert P.Arm.from_dict(json.loads(json.dumps(arm.to_dict()))) == arm
    assert P.Panel.from_json(json.loads(panel.canonical())).sha256() == panel.sha256()
    with pytest.raises(P.PanelError):
        P.Arm(arm="x", family="read", kind="transfer", role="test")
    with pytest.raises(P.PanelError):
        P.Arm(arm="x", family="read", kind="cut", role="driver")


# ------------------------------------------------------------------------------ read panel content

def test_full_read_panel_emits_both_cut_types_floors_keep_random_other_and_nested_arms():
    panel = P.read_panel(tiny_spec())
    ids = panel.by_id()
    # both cut types for test sets, twins linked
    assert ids["z:own2:red"].twin == "m:own2:red" and ids["m:own2:red"].twin == "z:own2:red"
    assert ids["z:own2:red"].units == ((0, 0), (2, 1)) and ids["z:own2:red"].claim == "necessary"
    # keep = complement of the set over the universe
    keep = ids["z:keep-own2:red"]
    assert set(keep.units) == set(UNIVERSE) - {(0, 0), (2, 1)} and keep.claim == "sufficient"
    # every no-renormalisation cut has recompute sham + recompute twin; every renormalising cut its sham
    for a in panel.arms:
        if a.kind in ("cut", "keep", "all_cut"):
            kinds = sorted(f.kind for f in panel.arms if f.floor_of == a.arm)
            assert kinds == (["floor_sham"] if a.cut_type == "renorm"
                             else ["floor_cut_recompute", "floor_sham_recompute"]), a.arm
    # random controls for both cut and keep, controlling the same-type test arm
    assert ids["z:rand0-own2:red"].control_for == ("z:own2:red",)
    assert ids["m:keep-rand0-own2:red"].control_for == ("m:keep-own2:red",)
    assert set(ids["z:rand0-own2:red"].units) == {tuple(u) for u in panel.controls["red:own2"]["sets"][0]}
    keep_draw = {tuple(u) for u in panel.controls["red:keep-own2"]["sets"][0]}   # its own draw (own seed)
    assert set(ids["z:keep-rand0-own2:red"].units) == set(UNIVERSE) - keep_draw
    report = panel.controls["red:own2"]["reports"][0]
    assert report["strata_counts"]["matched"] and report["distinct_blocks"]["matched"]
    # other item, same units: rows of the other word, units of this word's set
    other = ids["z:own2@rows-ball:red"]
    assert other.units == ids["z:own2:red"].units and other.rows_item == "ball"
    assert other.sources == ("own:ball",) and other.condition == "iso:red"
    # it rules out "these units damage any word"; it does not hold mass fixed
    assert other.separates_identity is False and other.rules_out == ("generic_damage",)
    assert ids["z:keep-own2@rows-ball:red"].rules_out == ()             # keep: reference only
    # nested series: added units alone
    added = ids["z:own4-minus-own2:red"]
    assert set(added.units) == {(1, 2), (3, 0)} and added.role == "test"
    # references, including the unmodified native run that iso/del are scored against
    assert {"iso:red", "del:red", "z:none:red", "m:none:red", "native"} <= set(ids)
    assert ids["iso:red"].reference == "native" and ids["native"].kind == "native"
    assert ids["z:none:red"].units is None


def test_panel_records_when_random_sets_cannot_separate_identity_from_mass():
    mass = tuple(10.0 if u in {(0, 0), (2, 1)} else 0.01 for u in UNIVERSE)   # the target holds the mass
    panel = P.read_panel(tiny_spec(random=P.RandomControls(n=2, strata=STRATA, mass=mass, mass_tolerance=0.5,
                                                           disjoint=True, seed=1)))
    info = panel.identity["red:own2"]["random_cut"]
    assert info["separable"] is False
    assert "reach the target mass" in info["why"] and info["max_reachable_mass_bound"] < info["target_mass"]
    assert panel.by_id()["z:rand0-own2:red"].separates_identity is False
    assert panel.by_id()["z:rand0-own2:red"].mass_relation == "less"
    assert panel.by_id()["z:rand0-own2:red"].rules_out == ("set_size",)
    other = panel.identity["red:own2"]["other_item_cut"]
    assert other["separable"] is False and other["rules_out"] == ["generic_damage"]


def test_mass_matched_sets_that_reuse_the_target_are_not_identity_controls():
    big = [(b, h) for b in range(10) for h in range(6)]
    strata = tuple("joint" if b < 3 else "single" for b, _ in big)
    target = [(0, 0), (1, 2), (4, 1), (6, 3), (8, 5)]
    mass = tuple(5.0 if u in target else 0.01 for u in big)
    sets = {"red": {"s": target}, "ball": {"s": [(9, 0)]}}
    panel = P.read_panel(P.ReadPanelSpec(items=("red", "ball"), universe=big, sets=sets, cut_sets=("s",),
                                         random=P.RandomControls(n=1, strata=strata, mass=mass,
                                                                 mass_tolerance=0.5, seed=0)))
    info = panel.identity["red:s"]["random_cut"]
    assert panel.controls["red:s"]["reports"][0]["mass"]["matched"]          # mass matched ...
    assert info["overlap_with_target"][0] == 5 and info["separable"] is False  # ... by re-using the target
    assert "re-using target units" in info["why"]


def test_structurally_impossible_random_control_is_recorded_as_missing():
    big = {"red": {"own2": UNIVERSE[:-1]}, "ball": {"own2": UNIVERSE[1:]}}
    panel = P.read_panel(tiny_spec(sets=big, nested=(), random=P.RandomControls(n=1, strata=None, disjoint=True)))
    assert panel.controls["red:own2"]["status"] == "control_missing"
    assert panel.identity["red:own2"]["random_cut"]["status"] == "control_missing"
    assert not any(a.control == "random" for a in panel.arms)


def test_mandatory_controls_refuse_without_written_waiver():
    with pytest.raises(P.PanelError, match="both cut types"):
        P.read_panel(tiny_spec(cut_types=("zero",)))
    with pytest.raises(P.PanelError, match="floors"):
        P.read_panel(tiny_spec(floors=False))
    with pytest.raises(P.PanelError, match="random"):
        P.read_panel(tiny_spec(random=None))
    with pytest.raises(P.PanelError, match="other-item"):
        P.read_panel(tiny_spec(other_items={}))
    with pytest.raises(P.PanelError, match="reason"):
        P.read_panel(tiny_spec(floors=False, waive={"floors": " "}))
    waived = P.read_panel(tiny_spec(floors=False, waive={"floors": "pilot"}))
    assert waived.waivers == {"floors": "pilot"} and not any(a.role == "floor" for a in waived.arms)


def test_budget_cap_and_deterministic_lease_split():
    panel = P.read_panel(tiny_spec())
    n = panel.branch_count
    assert n == len(panel.arms) * 2
    with pytest.raises(P.PanelError, match="over the cap"):
        panel.check_budget(n - 1)
    assert panel.check_budget(n) == n
    leases = panel.leases(40)
    assert leases == panel.leases(40)
    copies = {tuple(b) for lease in leases for b in lease["replicated"]}
    flat = [tuple(b) for lease in leases for b in lease["branches"] if b not in lease["replicated"]]
    assert sorted(flat) == sorted((c, a.arm) for c in panel.contexts for a in panel.arms)
    assert copies <= set(flat)
    assert all(len(lease["branches"]) <= 40 and lease["plan_sha256"] == panel.sha256() for lease in leases)
    # a test arm, its floors and its controls never straddle a lease
    for lease in leases:
        names = {a for _, a in lease["branches"]}
        if "z:own2:red" in names:
            assert {"z:own2:red/sham_recompute", "z:own2:red/cut_recompute", "z:rand0-own2:red"} <= names
    biggest = max(sum(1 for a in panel.arms if a.bundle == b) for b in {a.bundle for a in panel.arms})
    with pytest.raises(P.PanelError, match="over the lease cap"):
        panel.leases(biggest - 1)
    with pytest.raises(P.PanelError):
        panel.leases(256, max_total=n - 1)


# ------------------------------------------------------------------------------- arm -> port call

def test_port_mapping_matches_attention_port_cut_and_floor_arms():
    panel = P.read_panel(tiny_spec())
    ids = panel.by_id()
    z = ids["z:own2:red"]
    base = ap.cut(["image", "others:red"], ["own:red"], heads=[(0, 0), (2, 1)], renormalize=False)
    assert P.port_cuts(z) == (base,)
    assert P.port_cuts(ids["z:own2:red/sham_recompute"]) == (ap.floor_arms(base)["sham_recompute"],)
    assert P.port_cuts(ids["z:own2:red/cut_recompute"]) == (ap.floor_arms(base)["cut_recompute"],)
    m = ap.cut(["image", "others:red"], ["own:red"], heads=[(0, 0), (2, 1)], renormalize=True)
    assert P.port_cuts(ids["m:own2:red"]) == (m,)
    assert P.port_cuts(ids["m:own2:red/sham"]) == (ap.floor_arms(m)["sham"],)
    assert P.port_cuts(ids["z:none:red"])[0].heads is None
    assert P.port_cuts(ids["iso:red"]) == () and P.port_cuts(ids["del:red"]) == ()
    other = P.port_cuts(ids["z:own2@rows-ball:red"])[0]
    assert other.sources == ("own:ball",) and other.heads == base.heads
    rows = P.item_rows({"red": [3], "ball": [5, 6]}, text_rows=8, total_rows=12)
    assert rows["own:ball"] == [5, 6] and rows["others:red"] == [0, 1, 2, 4, 5, 6, 7] and rows["image"] == [8, 9, 10, 11]


def test_sets_from_ranking_reads_attention_ranking_order():
    np = pytest.importorskip("numpy")
    from mdb_lanes.attention_ranking import contrast_ranking
    prof = np.random.default_rng(0).random((3, 4, 3)) + 0.01
    ranking = contrast_ranking(prof, 0)
    sets = P.sets_from_ranking(ranking, (2, 4))
    assert set(sets["own2"]) == set(ranking["top"][:2]) and set(sets["own2"]) < set(sets["own4"])


# ------------------------------------------------------------------ reproduction: FLUX headfree plan

def legacy_headcut_arms(spec, word, blocks, heads):
    """Transcription of read_graph_worker.py: per-word `del:` arm (line 557) + `headcut` block (624-646)."""
    every = [[b, h] for b in range(blocks) for h in range(heads)]
    named = spec["head_sets"][word]
    plan_arms = spec["head_arms"]
    cond = f"iso:{word}"
    rows, srcs = ("image", f"others:{word}"), (f"own:{word}",)       # (["image", others], [own])
    out = {f"del:{word}": (None, (), (), None, f"delete:{word}"),
           f"iso:{word}": (None, (), (), None, cond),
           f"z:none:{word}": ("zero", rows, srcs, None, cond)}
    if plan_arms.get("sham"):
        out[f"z:sham-iso:{word}"] = ("zero", (), (), None, cond)
    for s in plan_arms["cut"]:
        out[f"z:{s}:{word}"] = ("zero", rows, srcs, frozenset(map(tuple, named[s])), cond)
    for s in plan_arms["keep"]:
        inside = {tuple(x) for x in named[s]}
        out[f"z:keep-{s}:{word}"] = ("zero", rows, srcs, frozenset(tuple(x) for x in every if tuple(x) not in inside), cond)
    for s in plan_arms["mask"]:
        out[f"m:{s}:{word}"] = ("renorm", rows, srcs, frozenset(map(tuple, named[s])), cond)
    return out


def headfree_spec(spec, universe, **kw):
    arms = spec["head_arms"]
    controls = {s: "own" + s.removeprefix("rand") for s in arms["cut"] if s.startswith("rand")}
    base = dict(items=tuple(spec["cut_words"]), universe=universe, sets=spec["head_sets"], contexts=("4242", "9001"),
                cut_sets=tuple(s for s in arms["cut"] if s not in controls), keep_sets=tuple(arms["keep"]),
                external_controls=controls)
    base.update(kw)
    return P.ReadPanelSpec(**base)


def legacy_view(panel, word):
    return {a.arm: (a.cut_type, a.receivers, a.sources, None if a.units is None else frozenset(a.units), a.condition)
            for a in panel.arms if a.item == word}


@pytest.mark.skipif(not HEADFREE.exists() or not HEADFREE_CAPTURE.exists(), reason="FLUX headfree run not on disk")
def test_headfree_plan_arm_list_is_reproduced_exactly_per_word():
    np = pytest.importorskip("numpy")
    blocks, heads = np.load(HEADFREE_CAPTURE)["tok_w"].shape[1:3]      # measured model geometry (25, 24)
    universe = [(b, h) for b in range(blocks) for h in range(heads)]
    plan = json.loads(HEADFREE.read_text())
    checked = 0
    for spec in plan["prompts"]:
        arms = spec["head_arms"]
        panel = P.read_panel(headfree_spec(
            spec, universe, cut_types=("zero",), cut_type_overrides={s: ("zero", "renorm") for s in arms["mask"]},
            random=None, other_items={}, reference_types=("zero",), floors=False,
            waive={k: "reproduce the 2026-11-11 headfree run as executed" for k in P.WAIVABLE},
            port={"precision": "legacy", "zero_method": "legacy FP32 recompute of every head in a touched block"}))
        for word in spec["cut_words"]:
            assert legacy_view(panel, word) == legacy_headcut_arms(spec, word, blocks, heads)
            checked += 1
        # the executed metrics carry exactly these per-word arms (plus prompt-level sham / img<-alltext);
        # the panel adds the unmodified `native` run that iso/del are scored against
        metrics = json.loads(HEADFREE_METRICS.read_text())
        ran = {m["arm"] for m in metrics if m["prompt"] == spec["id"]}
        assert {a.arm for a in panel.arms} - {"native"} == ran - {"sham", "img<-alltext"}
        assert panel.execution["precision"] == "legacy"
    assert checked == 6


@pytest.mark.skipif(not HEADFREE.exists() or not HEADFREE_CAPTURE.exists(), reason="FLUX headfree run not on disk")
def test_full_headfree_panel_contains_every_executed_arm_and_adds_the_missing_controls():
    np = pytest.importorskip("numpy")
    blocks, heads = np.load(HEADFREE_CAPTURE)["tok_w"].shape[1:3]
    universe = [(b, h) for b in range(blocks) for h in range(heads)]
    spec = json.loads(HEADFREE.read_text())["prompts"][0]
    strata = tuple("joint" if b < 5 else "single" for b, _ in universe)
    panel = P.read_panel(headfree_spec(spec, universe, random=P.RandomControls(n=1, strata=strata),
                                       nested=("own60", "own120", "own240")))
    word = "red"
    new = legacy_view(panel, word)
    for name, sig in legacy_headcut_arms(spec, word, blocks, heads).items():
        assert new[name] == sig, name
    ids = panel.by_id()
    assert "z:keep-rand0-own60:red" in ids and "m:keep-own60:red" in ids
    assert "z:own240-minus-own60:red" in ids and "z:own60@rows-black:red" in ids
    leases = panel.leases(P.FLUX_LANE_CAP)
    assert all(len(lease["branches"]) <= 256 for lease in leases)
    assert sum(len(lease["branches"]) - len(lease["replicated"]) for lease in leases) == panel.branch_count


# ------------------------------------------------------------------ reproduction: value panels

def _content(rows):
    return sorted((r["kind"], tuple(sorted(tuple(c) for c in r["cells"])), float(r["dose"])) for r in rows)


def test_value_coalition_plan_is_reproduced():
    from mdb_lanes.value_coalition import LATE_CELLS, plan
    groups = {f"KV{kv}": [c for c in LATE_CELLS if c[1] == kv] for kv in range(2)}
    groups.update({f"L{layer}-both": [(layer, 0), (layer, 1)] for layer in range(19, 28)})
    panel = P.value_panel(LATE_CELLS, label="late", groups=groups, doses=(0.5,), reference_dose=0)
    assert len(panel.arms) == len(plan()) == 56
    assert _content(P.to_legacy(panel)) == _content(plan())
    assert panel.aliases == {}


def test_frozen_value_pair_and_resolver_arms_are_reproduced():
    from mdb_lanes import attention_source_resolver, frozen_value_pair
    full = {"full-late": [(layer, kv) for layer in range(19, 28) for kv in range(2)]}
    pair = P.value_panel(frozen_value_pair.PAIR, label="pair", doses=(), references=("native", "donor_native"),
                         comparators=full, reference_dose=1)
    assert _content(P.to_legacy(pair)) == _content(frozen_value_pair.arms())
    assert set(pair.aliases) == {"without-L20-KV0", "without-L20-KV1"}       # equal to the singletons
    resolver = P.value_panel(attention_source_resolver.PAIR, label="pair", singletons=False, complements=False,
                             doses=(), references=("native", "donor_native"), comparators=full,
                             fixed_distance=True, reference_dose=1)
    assert _content(P.to_legacy(resolver)) == _content(attention_source_resolver.arms())


def legacy_worker_mapping(kind):
    """Transcription of value_coalition_worker.py 139-147 / resolver_execution_worker.py 79-89 (symbolic rows)."""
    side = "donor" if kind in {"donor-native", "reciprocal"} else "recipient"
    payload_side = "recipient" if kind in {"sham", "reciprocal"} else "donor"
    target = "wrong" if kind == "wrong-target" else "source"
    payload = "wrong" if kind == "wrong-payload" else "source"
    if kind == "fixed-distance":
        target = payload = "fixed-distance"
    return side, payload_side, target, payload, kind not in {"native", "donor-native"}


def test_value_spec_is_the_worker_mapping():
    panel = P.value_panel([(20, 0), (20, 1)], fixed_distance=True)
    seen = set()
    for arm in panel.arms:
        legacy = P.LEGACY_VALUE_KIND[arm.kind]
        side, payload_side, target, payload, replaces = legacy_worker_mapping(legacy)
        spec = P.value_spec(arm)
        assert P.run_side(arm) == side
        assert (spec is not None) == replaces
        if spec:
            assert (spec["side"], spec["payload_side"], spec["target_row"], spec["payload_row"]) == (
                side, payload_side, target, payload)
        seen.add(legacy)
    assert seen == set(P.LEGACY_VALUE_KIND.values())


def test_blend_is_bit_identical_to_value_coalition_replacement():
    torch = pytest.importorskip("torch")
    from mdb_lanes.value_coalition import replacement
    g = torch.Generator().manual_seed(0)
    current = [torch.randn(1, 2, 3, 4, generator=g).bfloat16() for _ in range(28)]
    donor = [torch.randn(1, 2, 3, 4, generator=g).bfloat16() for _ in range(28)]
    for dose in (0.5, 1):
        ref = replacement(current, donor, [(20, 1)], 2, 0, dose)
        mine = P.blend(current[20][:, 1, 2], donor[20][:, 1, 0], dose)
        assert mine.dtype == torch.bfloat16 and torch.equal(mine, ref[20][:, 1, 2])
    assert torch.equal(P.blend(current[3], donor[3], 0), current[3])
    with pytest.raises(P.PanelError):
        P.blend(current[3], donor[3], float("nan"))


# ----------------------------------------------------------------------------------------- sealing

def test_seal_hash_is_canonical_and_results_refuse_a_different_plan():
    a, b = P.read_panel(tiny_spec()), P.read_panel(tiny_spec())
    assert a.sha256() == b.sha256() and a.canonical() == b.canonical()
    seal = a.seal(predictions={"z:own2:red": 0.1})
    assert seal["plan_sha256"] == a.sha256() and len(seal["seal_sha256"]) == 64
    other = P.read_panel(tiny_spec(random=P.RandomControls(n=1, strata=STRATA, seed=4)))
    assert other.sha256() != a.sha256()
    rec = P.results_record(a, {"p|4242": {"z:own2:red": 0.2}}, seal=seal)
    assert P.attach_results(a, rec, seal=seal) is rec
    with pytest.raises(P.PanelError, match="different plan"):
        P.attach_results(other, rec)
    with pytest.raises(P.PanelError, match="seal"):
        P.attach_results(a, rec, seal=a.seal(predictions={"z:own2:red": 0.3}))
    bad = P.results_record(a, {"p|4242": {"z:nope:red": 0.2}})
    with pytest.raises(P.PanelError, match="unknown arm"):
        P.attach_results(a, bad)


def test_predictions_seal_through_forecast_transaction(tmp_path):
    pytest.importorskip("numpy")
    panel = P.read_panel(tiny_spec())
    tx, seal = P.seal_with_forecast_transaction(panel, tmp_path / "tx", {"z:own2:red": 0.1, "red:own2": "supported"},
                                                model_identity="flux2-klein-4b", input_identity="cat prompt")
    try:
        manifest = tx.verify()
        assert seal["plan_sha256"] == panel.sha256()
        assert manifest["declarations"]["source_identity"] == {"attention_panel_sha256": panel.sha256()}
        assert tx.prediction("panel_predictions") == {"z:own2:red": 0.1, "red:own2": "supported"}
    finally:
        tx.close()
    with pytest.raises(P.PanelError, match="unknown"):
        P.seal_with_forecast_transaction(panel, tmp_path / "tx2", {"z:nope:red": 1.0},
                                         model_identity="m", input_identity="i")


# ---------------------------------------------------------------------------------------- analysis

def small_panel(**kw):
    sets = {"red": {"s": [(0, 0)]}, "ball": {"s": [(1, 1)]}}
    base = dict(items=("red", "ball"), universe=UNIVERSE, sets=sets, contexts=("c0",), cut_sets=("s",),
                random=P.RandomControls(n=1, seed=0), references=("condition", "all_cut"))
    base.update(kw)
    return P.read_panel(P.ReadPanelSpec(**base))


def fill(panel, item, effect, control, floor=0.01, other=None, allcut=1.0, keep=None, keep_random=None,
         keep_other=None):
    """Outcomes: test arms `effect` (keep tests `keep`), random controls `control` (`keep_random`),
    other-item `other` (`keep_other`), all-cut `allcut`, floor shams `floor`, twin = its cut."""
    out = {}
    for a in panel.arms:
        if a.item != item or a.role == "floor":
            continue
        is_keep = a.kind == "keep"
        if a.role == "test":
            out[a.arm] = keep if (is_keep and keep is not None) else effect
        elif a.control == "random":
            out[a.arm] = keep_random if (is_keep and keep_random is not None) else control
        elif a.control == "other_item":
            value = control if other is None else other
            out[a.arm] = keep_other if (is_keep and keep_other is not None) else value
        elif a.kind == "all_cut":
            out[a.arm] = allcut
    for a in panel.arms:
        if a.item == item and a.role == "floor" and a.floor_of in out:
            out[a.arm] = out[a.floor_of] if a.kind == "floor_cut_recompute" else floor
    return out


def pairs(panel, outcomes, context="c0"):
    """Direct delta-vs-recompute pair distances (here equal to the scalar gap)."""
    rows = {}
    for key in P.twin_pairs(panel):
        a, b = key.split("|")
        if a in outcomes and b in outcomes:
            rows[key] = abs(outcomes[a] - outcomes[b])
    return {context: rows}


# Per-item mass for the other-item relation: equal mass everywhere, so an other-item control cuts
# as much of the other word's mass as the test cuts of its own (share 1). Since the 2026-11-13 fix
# an other-item control of UNKNOWN mass share rules nothing out (test_other_item_share_...).
EQUAL_ITEM_MASS = {"red": {u: 1.0 for u in UNIVERSE}, "ball": {u: 1.0 for u in UNIVERSE}}


def verdict(panel, outcomes, *, measured_twins=True, item_mass=EQUAL_ITEM_MASS, **kw):
    rec = P.results_record(panel, {"c0": outcomes}, pair_distances=pairs(panel, outcomes) if measured_twins else None)
    return {(v["item"], v["unit_set"]): v for v in P.verdicts(panel, rec, item_mass=item_mass, **kw)}


def test_verdict_scope_not_supported_unresolved():
    panel = small_panel()                                   # random control without mass: rules out set size only
    v = verdict(panel, fill(panel, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "set_only" and v["claim_scope"] == "needed_set" and v["cut_types_agree"]
    assert v["ruled_out"] == ["generic_damage", "set_size"] and v["remaining_alternatives"] == ["mass"]
    assert v["claim_statement"].startswith("WEAKER CLAIM") and "mass" in v["claim_statement"]
    assert set(v["cut_types"]) == {"zero", "renorm"}
    v = verdict(panel, fill(panel, "red", effect=0.5, control=0.05, other=0.495))[("red", "s")]
    assert v["status"] == "not_supported"                  # other item, same units: same effect
    assert {r["control"] for r in v["refuted_by"]} == {"z:s@rows-ball:red", "m:s@rows-ball:red"}
    v = verdict(panel, fill(panel, "red", effect=0.005, control=0.0))[("red", "s")]
    assert v["status"] == "unresolved" and v["claim_scope"] is None
    assert v["cut_types"]["zero"]["contexts"][0]["why"] == "test effect inside its floor"


def test_finding1_random_control_refutes_necessity_whatever_its_mass_confound():
    panel = small_panel()                                   # mass unknown
    v = verdict(panel, fill(panel, "red", effect=0.5, control=0.9, other=0.05))[("red", "s")]
    assert v["status"] == "not_supported"
    assert {(r["cut_type"], r["control"]) for r in v["refuted_by"]} == {("zero", "z:rand0-s:red"),
                                                                       ("renorm", "m:rand0-s:red")}
    # random set with LESS mass than the target (mass-confounded): still refutes when it does as much damage
    light = tuple(5.0 if u == (0, 0) else 0.01 for u in UNIVERSE)
    pm = small_panel(random=P.RandomControls(n=1, mass=light, mass_tolerance=0.1, disjoint=True, seed=0))
    assert pm.by_id()["z:rand0-s:red"].mass_relation == "less"
    v = verdict(pm, fill(pm, "red", effect=0.5, control=0.9, other=0.05))[("red", "s")]
    assert v["status"] == "not_supported" and v["refuted_by"]
    # ... and when beaten it rules out set size but not mass (confounded_with_mass)
    v = verdict(pm, fill(pm, "red", effect=0.5, control=0.05, other=0.05))[("red", "s")]
    assert v["status"] == "set_only" and v["remaining_alternatives"] == ["mass"]
    comp = v["cut_types"]["zero"]["contexts"][0]["controls"]
    assert {c["control"]: c["confounded_with_mass"] for c in comp if c["kind"] == "random"} == {"z:rand0-s:red": True}
    # a HEAVIER random set doing as much damage is explained by its mass: it neither refutes nor supports
    heavy = tuple(0.01 if u == (0, 0) else 1.0 for u in UNIVERSE)
    ph = small_panel(random=P.RandomControls(n=1, mass=heavy, seed=0))
    assert ph.by_id()["z:rand0-s:red"].mass_relation == "more"
    v = verdict(ph, fill(ph, "red", effect=0.5, control=0.9, other=0.05))[("red", "s")]
    assert v["status"] == "set_only" and not v["refuted_by"] and v["ruled_out"] == ["generic_damage"]
    # mass-matched random + other item beaten: the full (identity) claim
    flat = tuple(1.0 for _ in UNIVERSE)
    pi = small_panel(random=P.RandomControls(n=1, mass=flat, mass_tolerance=0.1, disjoint=True, seed=0))
    assert pi.by_id()["z:rand0-s:red"].mass_relation == "matched" and pi.by_id()["z:rand0-s:red"].separates_identity
    v = verdict(pi, fill(pi, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "supported" and v["claim_scope"] == "identity" and v["remaining_alternatives"] == []
    # without an other-item control, "these units damage any word" stays open
    alone = small_panel(random=P.RandomControls(n=1, mass=flat, mass_tolerance=0.1, disjoint=True, seed=0),
                        other_items={}, waive={"other_item": "single word"})
    v = verdict(alone, fill(alone, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "set_only" and v["remaining_alternatives"] == ["generic_damage"]


def test_floor_uses_the_larger_of_recompute_sham_and_twin_gap():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    out["z:s:red/cut_recompute"] = 0.1                                       # |delta - recompute| >= 0.4
    rec = P.results_record(panel, {"c0": out})
    row = {r["arm"]: r for r in P.arm_effects(panel, rec)}["z:s:red"]
    assert row["floor"] == pytest.approx(0.4) and row["floor_parts"]["sham_recompute"] == 0.01
    assert "lower bound" in row["floor_parts"]["twin_gap_source"] and row["floor_incomplete"]
    rec = P.results_record(panel, {"c0": out}, pair_distances={"c0": {"z:s:red|z:s:red/cut_recompute": 0.45}})
    row = {r["arm"]: r for r in P.arm_effects(panel, rec)}["z:s:red"]
    assert row["floor"] == pytest.approx(0.45) and row["floor_parts"]["twin_gap_source"] == "pair distance"
    # finding 3: without the pair distance the scalar gap is only a lower bound -> never supported
    v = verdict(panel, out, measured_twins=False)[("red", "s")]
    zero = v["cut_types"]["zero"]
    assert zero["status"] == "unresolved" and zero["contexts"][0]["reason"] == "twin_gap_not_measured"
    assert v["status"] == "unresolved" and v["floor_incomplete"] is True


def test_finding3_missing_twin_gap_blocks_support_unless_waived_in_writing():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    v = verdict(panel, out, measured_twins=False)[("red", "s")]
    assert v["status"] == "unresolved" and v["floor_incomplete"] and v["twin_gap_waiver"] is None
    assert v["cut_types"]["renorm"]["status"] == "set_only"          # renormalising cuts have no twin
    w = verdict(panel, out, measured_twins=False, twin_gap_waiver="pilot: twin pairs were not stored")[("red", "s")]
    assert w["status"] == "set_only" and w["floor_incomplete"] and w["twin_gap_waiver"].startswith("pilot")
    with pytest.raises(P.PanelError, match="written reason"):
        verdict(panel, out, measured_twins=False, twin_gap_waiver=" ")
    full = verdict(panel, out)[("red", "s")]
    assert full["status"] == "set_only" and full["floor_incomplete"] is False


def test_cut_type_disagreement_does_not_count():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    for a in panel.arms:
        if a.item == "red" and a.cut_type == "renorm" and a.role == "control":
            out[a.arm] = 0.5                                                   # renorm: effect == controls
    v = verdict(panel, out)[("red", "s")]
    assert v["cut_types"]["zero"]["status"] == "set_only"
    assert v["cut_types"]["renorm"]["status"] == "not_supported"
    assert v["status"] == "not_supported" and not v["cut_types_agree"] and "disagree" in v["why"]


def test_control_missing_for_absent_floor_waived_type_or_no_generic_damage_control():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    del out["z:s:red/sham_recompute"]
    assert verdict(panel, out)[("red", "s")]["status"] == "control_missing"
    one = small_panel(cut_types=("zero",), reference_types=("zero",), waive={"both_cut_types": "pilot"})
    v = verdict(one, fill(one, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "control_missing" and v["missing_cut_types"] == ["renorm"]
    assert v["waivers"] == {"both_cut_types": "pilot"}
    # a random control of unknown mass rules out set size only; with no other item, two alternatives remain
    alone = small_panel(other_items={}, waive={"other_item": "single word"})
    v = verdict(alone, fill(alone, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "set_only" and v["remaining_alternatives"] == ["generic_damage", "mass"]
    assert v["cut_types"]["zero"]["mass_confounded_controls"] == ["z:rand0-s:red"]
    # nothing rules out a generic-damage explanation
    bare = small_panel(random=None, other_items={}, waive={"random_cut": "pilot", "other_item": "single word"})
    v = verdict(bare, fill(bare, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "control_missing" and "generic-damage" in v["cut_types"]["zero"]["contexts"][0]["why"]


def keep_panel(**kw):
    sets = {"red": {"s": [(0, 0), (1, 1)]}, "ball": {"s": [(2, 2), (3, 1)]}}
    return small_panel(sets=sets, cut_sets=(), keep_sets=("s",), **kw)


def test_finding2_keep_claim_needs_recovery_and_a_beaten_random_keep():
    panel = keep_panel()
    v = verdict(panel, fill(panel, "red", 0, 0, keep=0.1, keep_random=0.8, allcut=1.0))[("red", "keep-s")]
    assert v["claim"] == "sufficient" and v["status"] == "set_only" and v["claim_scope"] == "sufficient_set"
    assert v["remaining_alternatives"] == ["mass"] and v["thresholds"]["recovery_threshold"] == 0.7
    assert v["cut_types"]["zero"]["contexts"][0]["recovery"] == pytest.approx(0.9)
    # keep effect .55 vs all-cut .60: keeping the set recovers ~8% -> not supported (was `supported`)
    v = verdict(panel, fill(panel, "red", 0, 0, keep=0.55, keep_random=0.9, allcut=0.6))[("red", "keep-s")]
    assert v["status"] == "not_supported" and v["findings"] == ["low_recovery"]
    assert v["cut_types"]["zero"]["contexts"][0]["recovery"] == pytest.approx(1 - 0.55 / 0.6)
    # the threshold is a recorded parameter
    v = verdict(panel, fill(panel, "red", 0, 0, keep=0.55, keep_random=0.9, allcut=0.6),
                recovery_threshold=0.05)[("red", "keep-s")]
    assert v["status"] == "set_only" and v["thresholds"]["recovery_threshold"] == 0.05
    # a same-size random keep recovers as much: any same-size set suffices (a finding), not this set
    v = verdict(panel, fill(panel, "red", 0, 0, keep=0.1, keep_random=0.1, allcut=1.0))[("red", "keep-s")]
    assert v["status"] == "not_supported" and v["findings"] == ["any_same_size_set_suffices"]
    assert {r["control"] for r in v["refuted_by"]} == {"z:keep-rand0-s:red", "m:keep-rand0-s:red"}
    # the other-item keep is a reference only: it can neither support nor stand in for the random keep
    out = fill(panel, "red", 0, 0, keep=0.1, keep_random=0.8, keep_other=0.9, allcut=1.0)
    for name in [k for k in out if "keep-rand0" in k]:
        del out[name]
    v = verdict(panel, out)[("red", "keep-s")]
    assert v["status"] == "control_missing"
    refs = v["cut_types"]["zero"]["contexts"][0]["references"]
    assert refs and refs[0]["kind"] == "other_item" and "never counted" in refs[0]["note"]
    # nothing to recover
    v = verdict(panel, fill(panel, "red", 0, 0, keep=0.001, keep_random=0.8, allcut=0.005))[("red", "keep-s")]
    assert v["status"] == "unresolved"
    # no random keep at all (waived): control_missing
    bare = keep_panel(random=None, waive={"random_keep": "pilot"})
    v = verdict(bare, fill(bare, "red", 0, 0, keep=0.1, allcut=1.0))[("red", "keep-s")]
    assert v["status"] == "control_missing"


def test_finding4_picture_judgement_cannot_repair_controls_and_unknown_keys_raise():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    key = ("red", "s", "necessary")
    v = verdict(panel, out, judgements={key: {"status": "not_supported", "by": "auditor", "note": "duplicated object"}})
    v = v[("red", "s")]
    assert v["scalar_status"] == "set_only" and v["status"] == "not_supported" and v["overridden_by_picture"]
    assert v["claim_scope"] is None
    # a picture cannot widen the scope the controls allow
    v = verdict(panel, out, judgements={key: {"status": "supported", "by": "auditor"}})[("red", "s")]
    assert v["status"] == "set_only" and "picture_capped" in v and not v["overridden_by_picture"]
    # ... but may say a pixel refutation was not real (capped by what the beaten controls rule out)
    refuted = fill(panel, "red", effect=0.5, control=0.9, other=0.05)
    v = verdict(panel, refuted, judgements={key: {"status": "supported", "by": "auditor"}})[("red", "s")]
    assert v["scalar_status"] == "not_supported" and v["status"] == "set_only" and v["overridden_by_picture"]
    # cannot repair a missing cut type / control
    one = small_panel(cut_types=("zero",), reference_types=("zero",), waive={"both_cut_types": "pilot"})
    v = verdict(one, fill(one, "red", effect=0.5, control=0.05),
                judgements={key: {"status": "supported", "by": "auditor"}})[("red", "s")]
    assert v["status"] == "control_missing" and "missing_cut_type" in v["picture_blocked_by"]
    bare = small_panel(random=None, other_items={}, waive={"random_cut": "pilot", "other_item": "single word"})
    v = verdict(bare, fill(bare, "red", effect=0.5, control=0.05),
                judgements={key: {"status": "supported", "by": "auditor"}})[("red", "s")]
    assert v["status"] == "control_missing" and "control_missing" in v["picture_blocked_by"]
    # cannot repair an incomplete floor
    v = verdict(panel, out, measured_twins=False, judgements={key: {"status": "supported", "by": "a"}})[("red", "s")]
    assert v["status"] == "unresolved" and "floor_incomplete" in v["picture_blocked_by"]
    with pytest.raises(P.PanelError, match="unknown"):
        verdict(panel, out, judgements={("red", "typo", "necessary"): {"status": "not_supported", "by": "a"}})
    with pytest.raises(P.PanelError):
        verdict(panel, out, judgements={key: {"status": "maybe", "by": "x"}})


def test_finding5_seal_is_recomputed_and_leases_carry_it():
    a = P.read_panel(tiny_spec())
    seal = a.seal(predictions={"z:own2:red": 0.1})
    assert a.verify_seal(seal) is seal
    forged = dict(seal, predictions={"z:own2:red": 0.9})              # predictions edited after sealing
    with pytest.raises(P.PanelError, match="altered"):
        P.verify_seal(a, forged)
    rec = P.results_record(a, {"p|4242": {"z:own2:red": 0.2}}, seal=forged)
    with pytest.raises(P.PanelError, match="altered"):
        P.attach_results(a, rec, seal=forged)
    with pytest.raises(P.PanelError, match="altered"):
        P.verify_seal(a, dict(seal, note="added later"))
    other = P.read_panel(tiny_spec(random=P.RandomControls(n=1, strata=STRATA, seed=4)))
    with pytest.raises(P.PanelError, match="different plan"):
        P.verify_seal(other, seal)
    leases = a.leases(40, seal=seal)
    assert all(lease["seal_sha256"] == seal["seal_sha256"] and lease["plan_sha256"] == a.sha256() for lease in leases)
    assert leases[0]["lease_sha256"] != a.leases(40)[0]["lease_sha256"] and a.leases(40)[0]["seal_sha256"] is None
    with pytest.raises(P.PanelError, match="altered"):
        a.leases(40, seal=forged)
    # references run in another lease are listed, so scoring waits for them
    for lease in leases:
        inside = {tuple(b) for b in lease["branches"]}
        assert all(tuple(d) not in inside for d in lease["depends_on"])
    assert any(d[1] == "native" for lease in leases for d in lease["depends_on"])


def test_plans_written_before_the_new_fields_keep_their_hash_and_profiled_controls():
    panel = small_panel()
    d = json.loads(panel.canonical())
    d.pop("execution")
    for a in d["arms"]:                                   # what a 5e09d95 plan looks like
        a.pop("rules_out", None)
        a.pop("mass_relation", None)
        if a["control"] == "other_item":
            a["separates_identity"] = True
    legacy = P.Panel.from_json(d)
    assert legacy.canonical() == P.canonical_json(d)     # hash of the saved plan is reproduced
    out = fill(legacy, "red", effect=0.5, control=0.05)
    old, new = verdict(legacy, out)[("red", "s")], verdict(panel, out)[("red", "s")]
    assert (old["status"], old["ruled_out"]) == (new["status"], new["ruled_out"]) == ("set_only", ["generic_damage",
                                                                                               "set_size"])
    assert verdict(legacy, fill(legacy, "red", effect=0.5, control=0.9))[("red", "s")]["status"] == "not_supported"


def test_finding6_universe_geometry_is_recorded_and_checked():
    panel = P.read_panel(tiny_spec())
    geo = panel.execution["universe"]
    assert (geo["blocks"], geo["heads"], geo["full_grid"], geo["n_units"]) == (4, 3, True, 12)
    assert geo["strata_counts"] == {"joint": 6, "single": 6}
    assert panel.check_universe(4, 3)
    with pytest.raises(P.PanelError, match="5 blocks x 3 heads"):
        panel.check_universe(5, 3)
    part = [u for u in UNIVERSE if u != (3, 1)]
    strata = tuple(s for u, s in zip(UNIVERSE, STRATA) if u != (3, 1))
    with pytest.raises(P.PanelError, match="full"):
        P.read_panel(tiny_spec(universe=part, random=P.RandomControls(n=1, strata=strata, seed=3)))
    ok = P.read_panel(tiny_spec(universe=part, random=P.RandomControls(n=1, strata=strata, seed=3),
                                waive={"partial_universe": "only captured heads"}))
    assert ok.check_universe(4, 3) and not ok.execution["universe"]["full_grid"]
    with pytest.raises(P.PanelError):
        ok.check_universe(4, 4)
    assert ok.sha256() != panel.sha256()


def test_finding7_degenerate_inputs_are_refused():
    ball = tiny_sets()["ball"]
    cases = [({"red": {"own2": [], "own4": [(0, 0)]}, "ball": ball}, "empty"),
             ({"red": {"own2": UNIVERSE, "own4": [(0, 0)]}, "ball": ball}, "whole universe"),
             ({"red": {"own2": [(0, 0), (0, 0), (2, 1)], "own4": [(0, 0)]}, "ball": ball}, "more than once")]
    for sets, match in cases:
        with pytest.raises(P.PanelError, match=match):
            P.read_panel(tiny_spec(sets=sets, nested=()))
    with pytest.raises(P.PanelError, match="n=0"):
        P.RandomControls(n=0)
    with pytest.raises(P.PanelError, match="itself"):
        P.read_panel(tiny_spec(other_items={"red": ("red",), "ball": ("red",)}))
    with pytest.raises(P.PanelError, match="not an item"):
        P.read_panel(tiny_spec(other_items={"red": ("zebra",), "ball": ("red",)}))


def test_finding8_references_are_consistent_and_native_is_a_branch():
    with pytest.raises(P.PanelError, match="condition"):
        P.read_panel(tiny_spec(references=("all_cut", "deletion")))
    with pytest.raises(P.PanelError, match="all-cut"):
        P.read_panel(tiny_spec(references=("condition",)))
    plain = P.read_panel(tiny_spec(condition=None, references=("all_cut",)))
    assert plain.by_id()["z:own2:red"].reference == "native" and "native" in plain.by_id()
    panel = P.read_panel(tiny_spec())
    assert panel.by_id()["native"].condition is None and panel.branch_count == len(panel.arms) * 2
    flat = {tuple(b) for lease in panel.leases(40) for b in lease["branches"]}
    assert {("p|4242", "native"), ("p|9001", "native")} <= flat


def test_finding9_value_panel_controls_doses_and_bundles():
    with pytest.raises(P.PanelError, match="sham"):
        P.value_panel([(1, 0), (1, 1)], controls=(), references=())
    with pytest.raises(P.PanelError, match="native"):
        P.value_panel([(1, 0)], controls=("sham",), references=())
    waived = P.value_panel([(1, 0)], controls=(), references=(),
                           waive={"sham": "pilot", "native_reference": "scored elsewhere"})
    assert waived.waivers == {"sham": "pilot", "native_reference": "scored elsewhere"}
    for d in (7.0, -3.0):
        with pytest.raises(P.PanelError, match="outside"):
            P.value_panel([(1, 0)], doses=(d,))
    extra = P.value_panel([(1, 0)], doses=(1.5,), allow_extrapolation=True)
    assert [a.dose for a in extra.arms if a.kind == "dose"] == [1.5] and extra.execution["allow_extrapolation"]
    vp = P.value_panel([(1, 0), (1, 1)], fixed_distance=True)
    size = sum(1 for a in vp.arms if a.bundle == "coalition:all")
    controls = {"sham-all", "wrong-payload-all", "wrong-target-all", "reciprocal-all", "fixed-distance-all"}
    for cap in range(size, len(vp.arms) + 1):
        for lease in vp.leases(cap):
            names = {a for _, a in lease["branches"]}
            if "coalition-all" in names:
                assert controls <= names, cap
    with pytest.raises(P.PanelError, match="over the lease cap"):
        vp.leases(size - 1)


def test_finding10_execution_block_is_hashed_and_checked_against_the_port():
    base = P.read_panel(tiny_spec())
    ex = base.execution
    assert ex["precision"] == "native" and ex["zero_method"]["zero"] == "delta" and ex["zero_method"]["renorm"] is None
    assert set(ex["unbound"]) == {"row_table", "forwards_per_image"}
    assert P.read_panel(tiny_spec(precision="legacy")).sha256() != base.sha256()
    assert P.read_panel(tiny_spec(port_zero_method="recompute")).sha256() != base.sha256()
    rows = P.item_rows({"red": [3], "ball": [5, 6]}, text_rows=8, total_rows=12)
    bound = P.read_panel(tiny_spec(row_table=rows, forwards_per_image=4))
    assert bound.sha256() != base.sha256() and bound.execution["unbound"] == []
    kw = P.port_kwargs(bound, rows)
    assert {k: kw[k] for k in ("zero_method", "precision", "empty")} == {"zero_method": "delta", "precision": "native",
                                                                         "empty": "raise"}
    port = ap.AttentionPort(None, **kw)
    assert P.check_port(bound, port, forwards=4)
    z = bound.by_id()["z:own2:red"]
    assert P.port_cuts(z, panel=bound, port=port) == P.port_cuts(z)
    with pytest.raises(P.PanelError, match="precision"):
        P.check_port(bound, ap.AttentionPort(None, rows=rows, precision="legacy"))
    with pytest.raises(P.PanelError, match="precision"):
        P.port_cuts(z, panel=bound, port=ap.AttentionPort(None, rows=rows, precision="legacy"))
    with pytest.raises(P.PanelError, match="zero_method"):
        P.check_port(bound, ap.AttentionPort(None, rows=rows, zero_method="recompute"))
    with pytest.raises(P.PanelError, match="row table"):
        P.port_kwargs(bound, {**rows, "image": [8, 9]})
    with pytest.raises(P.PanelError, match="row table"):
        P.check_port(bound, ap.AttentionPort(None, rows={**rows, "image": [8]}))
    with pytest.raises(P.PanelError, match="forwards"):
        P.check_port(bound, port, forwards=5)
    with pytest.raises(P.PanelError, match="contradicts"):
        P.read_panel(tiny_spec(precision="native", port={"precision": "legacy"}))


def test_finding11_thresholds_recorded_inputs_validated():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    v = verdict(panel, out, floor_factor=2.0)[("red", "s")]
    assert v["thresholds"]["floor_factor"] == 2.0 and v["thresholds"]["overlap_sd"] == 3.0
    with pytest.raises(P.PanelError, match="floor_factor"):
        verdict(panel, out, floor_factor=0)
    v0 = verdict(panel, out, floor_factor=0, floor_factor_waiver="exact backend")[("red", "s")]
    assert v0["thresholds"]["floor_factor_waiver"] == "exact backend"
    # the overlap threshold is a parameter (recorded in the plan)
    big = [(b, h) for b in range(10) for h in range(6)]
    strata = tuple("joint" if b < 3 else "single" for b, _ in big)
    target = [(0, 0), (1, 2), (4, 1), (6, 3), (8, 5)]
    mass = tuple(5.0 if u in target else 0.01 for u in big)
    sets = {"red": {"s": target}, "ball": {"s": [(9, 0)]}}
    for sd, relation in ((3.0, "matched_by_reuse"), (1e6, "matched")):
        p = P.read_panel(P.ReadPanelSpec(items=("red", "ball"), universe=big, sets=sets, cut_sets=("s",),
                                         random=P.RandomControls(n=1, strata=strata, mass=mass, mass_tolerance=0.5,
                                                                 seed=0, overlap_sd=sd)))
        assert p.by_id()["z:rand0-s:red"].mass_relation == relation and p.controls["red:s"]["overlap_sd"] == sd
    # numpy scalars are converted at construction
    np = pytest.importorskip("numpy")
    a = P.read_panel(tiny_spec(random=P.RandomControls(n=np.int64(1), strata=np.array(STRATA), seed=np.int64(3),
                                                       mass=np.ones(12), mass_tolerance=np.float32(0.5))))
    b = P.read_panel(tiny_spec(random=P.RandomControls(n=1, strata=STRATA, seed=3, mass=(1.0,) * 12,
                                                       mass_tolerance=0.5)))
    assert a.sha256() == b.sha256()
    # results validation
    with pytest.raises(P.PanelError, match="unknown context"):
        P.attach_results(panel, P.results_record(panel, {"c0": {}}, pair_distances={"nope": {}}))
    with pytest.raises(P.PanelError, match="negative"):
        P.attach_results(panel, P.results_record(panel, {"c0": {"z:s:red": -5.0}}))
    with pytest.raises(P.PanelError, match="negative"):
        P.attach_results(panel, P.results_record(panel, {"c0": {}}, pair_distances={
            "c0": {"z:s:red|z:s:red/cut_recompute": -0.1}}))
    with pytest.raises(P.PanelError, match="pair distance key"):
        P.attach_results(panel, P.results_record(panel, {"c0": {}}, pair_distances={"c0": {"z:s:red": 0.1}}))
    # aliased value arms
    vp = P.value_panel([(1, 0), (1, 1)])
    alias = "without-L1-KV0"
    with pytest.raises(P.PanelError, match="resolve_aliases"):
        P.attach_results(vp, P.results_record(vp, {"default": {alias: 0.1}}))
    resolved = P.resolve_aliases(vp, {"default": {alias: 0.1}})
    assert resolved == {"default": {vp.aliases[alias]: 0.1}}
    assert P.attach_results(vp, P.results_record(vp, resolved))
    with pytest.raises(P.PanelError, match="alias"):
        P.resolve_aliases(vp, {"default": {alias: 0.1, vp.aliases[alias]: 0.2}})


def test_finding12_value_spec_documents_worker_details_and_blend_dtypes():
    doc = P.value_spec.__doc__
    assert "len(tokens) - 4" in doc and "FULL dose" in doc and "PAYLOAD side" in doc
    torch = pytest.importorskip("torch")
    before, payload = torch.zeros(3, dtype=torch.bfloat16), torch.ones(3, dtype=torch.float32)
    assert P.blend(before, payload, 1).dtype == torch.float32          # exact payload copy
    assert P.blend(before, payload, 0.5).dtype == torch.bfloat16       # lerp cast to before's dtype


# ------------------------------------------------------------------ leftovers (2026-11-12 fix pass)

def test_random_draws_are_independent_per_item_set_and_kind_yet_deterministic():
    """Same seed, different items with same-shaped sets: draws used to be identical (seed reused)."""
    big = [(b, h) for b in range(6) for h in range(8)]
    sets = {"red": {"s": [(0, 0), (1, 1), (2, 2)]}, "ball": {"s": [(3, 3), (4, 4), (5, 5)]}}
    spec = P.ReadPanelSpec(items=("red", "ball"), universe=big, sets=sets, cut_sets=("s",), keep_sets=("s",),
                           random=P.RandomControls(n=1, seed=7, match_blocks=False))
    a, b = P.read_panel(spec), P.read_panel(spec)
    assert a.sha256() == b.sha256()                                            # deterministic
    red, ball = a.controls["red:s"], a.controls["ball:s"]
    assert red["seed"] == ball["seed"] == 7 and red["draw_seed"] != ball["draw_seed"]
    assert red["draw_seed"] == P.draw_seed(7, "red", "s", "cut") and red["seed_derivation"] == P.SEED_DERIVATION
    # identical target shapes: the old shared seed gave the same draw (as a pattern of flat indices)
    from mdb_lanes.attention_ranking import matched_control_sets
    idx = {u: i for i, u in enumerate(big)}
    old = [sorted(int(i) for i in matched_control_sets([idx[u] for u in sets[w]["s"]], len(big), seed=7,
                                                       n_sets=1)[0]["units"]) for w in ("red", "ball")]
    assert old[0] == old[1]                                                    # the bug: one draw for both items
    new = [sorted(idx[tuple(u)] for u in a.controls[f"{w}:s"]["sets"][0]) for w in ("red", "ball")]
    assert new[0] != new[1]
    assert a.controls["red:keep-s"]["draw_seed"] != red["draw_seed"]          # cut and keep draws differ too
    assert P.draw_seed(7, "red", "s", "cut") != P.draw_seed(8, "red", "s", "cut")


def test_keep_arms_share_a_lease_with_their_all_cut_reference():
    panel = P.read_panel(tiny_spec())
    needs = panel.shared_needs()
    assert needs["z:keep-own2:red"] == ["z:none:red", "z:none:red/sham_recompute", "z:none:red/cut_recompute"]
    assert needs["m:keep-rand0-own2:red"] == ["m:none:red", "m:none:red/sham"]
    keep_bundle = sum(1 for a in panel.arms if a.bundle == "keep-own2:red")
    extra = 5                                                                  # z all-cut + 2 floors, m all-cut + sham
    for cap in range(keep_bundle + extra, panel.branch_count + 1, 3):
        leases = panel.leases(cap)
        for lease in leases:
            names = {(c, a) for c, a in lease["branches"]}
            assert len(lease["branches"]) <= cap                               # copies count against the cap
            for c, a in names:
                for n in needs.get(a, ()):
                    assert (c, n) in names, (cap, a, n)
        primary = [tuple(b) for lease in leases for b in lease["branches"] if b not in lease["replicated"]]
        assert sorted(primary) == sorted((c, a.arm) for c in panel.contexts for a in panel.arms)
    small = panel.leases(keep_bundle + extra)
    assert any(lease["replicated"] for lease in small)
    with pytest.raises(P.PanelError, match="shared references"):
        panel.leases(keep_bundle + extra - 1)


def test_references_only_panel_is_refused_unless_waived():
    with pytest.raises(P.PanelError, match="references only"):
        P.read_panel(tiny_spec(cut_sets=(), keep_sets=(), nested=()))
    refs = P.read_panel(tiny_spec(cut_sets=(), keep_sets=(), nested=(),
                                  waive={"references_only": "measure the reference arms alone"}))
    assert not any(a.role == "test" for a in refs.arms)
    assert P.read_panel(tiny_spec(cut_sets=(), keep_sets=())).by_id()["z:own4-minus-own2:red"].role == "test"


def test_random_mass_per_item_and_value_panel_port_block():
    """Items that read different sources draw against their own mass; value plans can bind a lane."""
    heavy_red = tuple(5.0 if u in {(0, 0), (2, 1)} else 0.01 for u in UNIVERSE)
    heavy_ball = tuple(5.0 if u in {(1, 1), (3, 2)} else 0.01 for u in UNIVERSE)
    panel = P.read_panel(tiny_spec(random=P.RandomControls(n=1, strata=STRATA, seed=3,
                                                           mass={"red": heavy_red, "ball": heavy_ball})))
    assert panel.controls["red:own2"]["reports"][0]["mass"]["target"] == pytest.approx(10.0)
    assert panel.controls["ball:own2"]["reports"][0]["mass"]["target"] == pytest.approx(10.0)
    with pytest.raises(P.PanelError, match="no mass for 'ball'"):
        P.read_panel(tiny_spec(random=P.RandomControls(n=1, strata=STRATA, mass={"red": heavy_red})))
    with pytest.raises(P.PanelError, match="one value per universe unit"):
        P.read_panel(tiny_spec(random=P.RandomControls(n=1, strata=STRATA, mass=(1.0,) * 5)))
    from mdb_lanes.value_coalition import LATE_CELLS
    plain = P.value_panel(LATE_CELLS, label="late")
    assert P.value_panel(LATE_CELLS, label="late", port=None).sha256() == plain.sha256()
    lane = P.value_panel(LATE_CELLS, label="late", port={"dtype": "float32"})
    assert lane.port == {"dtype": "float32"} and lane.sha256() != plain.sha256()
    assert P.Panel.from_json(json.loads(lane.canonical())).sha256() == lane.sha256()


# ------------------------------------------------------- audit 2026-11-13 fix pass (findings 1-3, 7)

def shared_panel(*, exclude_named=True, n=2, rel=0.2, abs_tol=None, sets=None):
    """Two items reading different rows with the SAME head sets (the Qwen layout): set a = (0,0),
    set b = (0,1). Strata = block, so a's random draw must come from block 0, and mass matching
    pulls it to the heaviest free head of block 0, which is b's head (0,1) unless named units are
    excluded."""
    mass = {u: 0.01 for u in UNIVERSE}
    mass.update({(0, 0): 1.0, (0, 1): 0.9})
    vec = tuple(mass[u] for u in UNIVERSE)
    sets = sets or {"a": [(0, 0)], "b": [(0, 1)]}
    rc = (P.RandomControls(n=n, strata=tuple(b for b, _ in UNIVERSE), mass=vec, mass_tolerance=abs_tol, disjoint=True,
                           seed=0, exclude_named=exclude_named) if abs_tol is not None else
          P.RandomControls(n=n, strata=tuple(b for b, _ in UNIVERSE), mass=vec, mass_rel_tolerance=rel,
                           disjoint=True, seed=0, exclude_named=exclude_named))
    return P.read_panel(P.ReadPanelSpec(items=("red", "ball"), universe=UNIVERSE, sets={"red": sets, "ball": sets},
                                        contexts=("c0",), cut_sets=tuple(sets), random=rc,
                                        references=("condition", "all_cut")))


def test_duplicate_draws_are_flagged_and_count_once():
    from mdb_lanes.attention_ranking import matched_control_sets
    target, n = [0], 12
    mass = [1.0, 0.9] + [0.01] * 10
    strata = [b for b, _ in UNIVERSE]
    drawn = matched_control_sets(target, n, strata=strata, mass=mass, mass_rel_tolerance=0.2, disjoint=True,
                                 seed=0, n_sets=3)
    assert [d["report"]["duplicate_of"] for d in drawn] == [None, 0, 0]    # mass pulls every draw to unit 1
    panel = shared_panel(exclude_named=False, n=3)
    entry = panel.controls["red:a"]
    assert entry["independent_draws"] == 1 and [d["duplicate_of"] for d in entry["duplicates"]] == [0, 0]
    ids = panel.by_id()
    assert "z:rand0-a:red" in ids and "z:rand1-a:red" not in ids and "z:rand2-a:red" not in ids
    # an OLD plan that emitted both identical draws: the second is excluded at verdict time
    d = json.loads(panel.canonical())
    rand0 = [a for a in d["arms"] if a["unit_set"] in ("rand0-a",) and a["item"] == "red"]
    for a in rand0:
        twin = dict(a, arm=a["arm"].replace("rand0", "rand1"), unit_set="rand1-a",
                    twin=None if a["twin"] is None else a["twin"].replace("rand0", "rand1"),
                    floor_of=None if a.get("floor_of") is None else a["floor_of"].replace("rand0", "rand1"),
                    bundle=a["bundle"])
        d["arms"].insert(d["arms"].index(a) + 1, twin)
    old = P.Panel.from_json(d)
    out = fill(old, "red", effect=0.5, control=0.05)
    v = verdict(old, out, exclude_named_draws=False)[("red", "a")]
    ctl = {c["control"]: c for c in v["cut_types"]["zero"]["contexts"][0]["controls"]}
    assert ctl["z:rand1-a:red"]["duplicate_of"] == 0 and "counts once" in ctl["z:rand1-a:red"]["excluded"]
    assert not ctl["z:rand1-a:red"]["refutes"] and ctl["z:rand1-a:red"]["rules_out"] == []


def test_random_draws_exclude_every_named_test_unit_and_old_draws_with_named_units_are_excluded():
    # generation: with exclusion the draw for a avoids b's head (0,1) and takes (0,2)
    panel = shared_panel()
    assert panel.controls["red:a"]["sets"][0] == [[0, 2]] and panel.controls["red:a"]["excluded_named_units"] == 1
    assert panel.by_id()["z:rand0-a:red"].mass_relation == "less"           # 0.01 vs 1.0: not matched
    # without exclusion (what the sealed Qwen plan did) the "random" control for a IS set b
    old = shared_panel(exclude_named=False)
    assert old.controls["red:a"]["sets"][0] == [[0, 1]]
    assert old.by_id()["z:rand0-a:red"].mass_relation == "matched"           # 0.9 vs 1.0, within 20%
    item_mass = {"red": {u: (1.0 if u == (0, 0) else 0.9 if u == (0, 1) else 0.01) for u in UNIVERSE},
                 "ball": {u: (0.6 if u == (0, 0) else 0.5 if u == (0, 1) else 0.01) for u in UNIVERSE}}
    out = fill(old, "red", effect=0.5, control=0.05)
    v = verdict(old, out, item_mass=item_mass)[("red", "a")]
    zero = v["cut_types"]["zero"]
    assert "z:rand0-a:red" in zero["excluded_controls"] and "named test sets" in zero["excluded_controls"]["z:rand0-a:red"]
    # the excluded random control rules nothing out; the other item (share 0.6) rules out generic damage only
    assert v["status"] == "set_only" and v["ruled_out"] == ["generic_damage"]
    assert v["remaining_alternatives"] == ["mass", "set_size"]
    # ... and it cannot refute either, even when it does as much damage as the test
    w = verdict(old, fill(old, "red", effect=0.5, control=0.9, other=0.05), item_mass=item_mass)[("red", "a")]
    assert w["status"] == "set_only" and not w["refuted_by"]
    # the named-unit check can be turned into a flag only (sensitivity reporting)
    f = verdict(old, out, item_mass=item_mass, exclude_named_draws=False)[("red", "a")]
    c = f["cut_types"]["zero"]["contexts"][0]["controls"]
    assert f["status"] == "supported" and any(x.get("named_overlap") == [[0, 1]] for x in c)


def test_no_valid_control_means_control_missing_never_set_only():
    old = shared_panel(exclude_named=False)
    # other item carries a near-empty share of the target's mass through head (0,0): 0.0005 / 1.0
    item_mass = {"red": {u: (1.0 if u == (0, 0) else 0.01) for u in UNIVERSE},
                 "ball": {u: (0.0005 if u == (0, 0) else 0.01) for u in UNIVERSE}}
    v = verdict(old, fill(old, "red", effect=0.5, control=0.05), item_mass=item_mass)[("red", "a")]
    assert v["status"] == "control_missing" and v["ruled_out"] == []
    row = v["cut_types"]["zero"]["contexts"][0]
    other = next(c for c in row["controls"] if c["kind"] == "other_item")
    assert other["other_item_share"] == pytest.approx(0.0005) and other["rules_out"] == []
    assert v["thresholds"]["other_item_min_share"] == P.OTHER_ITEM_MIN_SHARE
    # unknown other-item mass (no item_mass, sets not stored per item): rules nothing out either
    panel = small_panel(random=None, waive={"random_cut": "pilot"})
    u = verdict(panel, fill(panel, "red", effect=0.5, control=0.05), item_mass=None)[("red", "s")]
    assert u["status"] == "control_missing"
    unk = next(c for c in u["cut_types"]["zero"]["contexts"][0]["controls"] if c["kind"] == "other_item")
    assert unk["other_item_share"] is None and unk["mass_relation"] == "unknown"
    # a heavier other-item cut (share 40x) that does as much damage cannot refute; beaten, it rules out
    heavy = {"red": {u: (0.01 if u == (0, 0) else 0.01) for u in UNIVERSE},
             "ball": {u: (0.4 if u == (0, 0) else 0.01) for u in UNIVERSE}}
    h = verdict(old, fill(old, "red", effect=0.5, control=0.05, other=0.9), item_mass=heavy)[("red", "a")]
    assert not h["refuted_by"] and h["status"] == "control_missing"     # random excluded, other not beaten
    h = verdict(old, fill(old, "red", effect=0.5, control=0.05, other=0.05), item_mass=heavy)[("red", "a")]
    assert h["status"] == "set_only" and h["ruled_out"] == ["generic_damage"]


def test_mass_relation_is_relative_with_an_absolute_floor():
    assert P.mass_relation(0.0169, 0.0015) == "more"                 # the FLUX own240-minus-own60 case
    assert P.mass_relation(1.09, 1.0) == "matched" and P.mass_relation(1.11, 1.0) == "more"
    assert P.mass_relation(0.89, 1.0) == "less"
    assert P.mass_relation(0.0009, 0.0) == "matched"                 # floor: near-zero target
    assert P.mass_relation(0.002, 0.0) == "more"
    assert P.mass_relation(1.2, 1.0, rel_tolerance=0.25) == "matched"
    assert P.mass_relation(None, 1.0) == "unknown"
    # generation: an absolute search tolerance of 0.02 no longer labels an 11x control "matched"
    big = [(b, h) for b in range(4) for h in range(3)]
    mass = tuple(0.0015 if u == (0, 0) else 0.0169 for u in big)
    sets = {"red": {"s": [(0, 0)]}, "ball": {"s": [(1, 1)]}}
    p = P.read_panel(P.ReadPanelSpec(items=("red", "ball"), universe=big, sets=sets, contexts=("c0",), cut_sets=("s",),
                                     random=P.RandomControls(n=1, mass=mass, mass_tolerance=0.02, disjoint=True, seed=0),
                                     references=("condition", "all_cut")))
    report = p.controls["red:s"]["reports"][0]["mass"]
    assert report["matched"] is True                                    # the absolute search said matched ...
    assert p.by_id()["z:rand0-s:red"].mass_relation == "more"           # ... the label says more
    assert p.identity["red:s"]["random_cut"]["separable"] is False
    # an old plan that stored "matched": re-labelled at verdict time, so it cannot refute as matched
    d = json.loads(p.canonical())
    for a in d["arms"]:
        if a.get("control") == "random":
            a["mass_relation"], a["rules_out"], a["separates_identity"] = "matched", ["mass", "set_size"], True
    old = P.Panel.from_json(d)
    rec = P.results_record(old, {"c0": fill(old, "red", effect=0.5, control=0.9, other=0.05)})
    rec["pair_distances"] = pairs(old, rec["outcomes"]["c0"])
    v = {(x["item"], x["unit_set"]): x for x in P.verdicts(old, rec, item_mass=EQUAL_ITEM_MASS_BIG)}[("red", "s")]
    c = next(x for x in v["cut_types"]["zero"]["contexts"][0]["controls"] if x["kind"] == "random")
    assert c["mass_relation_stored"] == "matched" and c["mass_relation"] == "more" and not c["refutes"]
    assert "mass" not in v["ruled_out"]


EQUAL_ITEM_MASS_BIG = {"red": {(b, h): 1.0 for b in range(4) for h in range(3)},
                       "ball": {(b, h): 1.0 for b in range(4) for h in range(3)}}


def test_effect_fraction_is_reported_and_min_fraction_is_a_named_parameter():
    panel = small_panel()
    v = verdict(panel, fill(panel, "red", effect=0.5, control=0.05, allcut=1.0))[("red", "s")]
    assert v["status"] == "set_only" and v["effect_fraction"]["min"] == pytest.approx(0.5)
    assert v["thresholds"]["min_effect_fraction"] == P.DEFAULT_MIN_EFFECT_FRACTION
    assert v["thresholds"]["parameter_sources"]["min_effect_fraction"] == P.DEFAULT_SOURCE
    # beaten controls, but 2% of the all-cut effect: not reported as a bare needed set
    small = fill(panel, "red", effect=0.02, control=0.001, floor=0.001, allcut=1.0)
    s = verdict(panel, small)[("red", "s")]
    assert s["status"] == "unresolved" and "effect_below_min_fraction" in s["findings"]
    assert s["effect_fraction"]["min"] == pytest.approx(0.02) and "below min_effect_fraction" in s["why"]
    assert s["claim_scope"] is None
    # the bar is a parameter: per call ...
    a = verdict(panel, small, min_effect_fraction=0.01)[("red", "s")]
    assert a["status"] == "set_only" and a["thresholds"]["parameter_sources"]["min_effect_fraction"] == "argument"
    # ... or per plan (hashed only when set, so plans without it keep their hash)
    assert "verdict_parameters" not in panel.execution
    planned = small_panel(min_effect_fraction=0.01, recovery_threshold=0.5)
    assert planned.execution["verdict_parameters"] == {"min_effect_fraction": 0.01, "recovery_threshold": 0.5}
    assert planned.sha256() != panel.sha256()
    p = verdict(planned, fill(planned, "red", effect=0.02, control=0.001, floor=0.001, allcut=1.0))[("red", "s")]
    assert p["status"] == "set_only" and p["thresholds"]["parameter_sources"]["min_effect_fraction"] == "plan"
    # a picture judgement cannot repair a small effect
    j = verdict(panel, small, judgements={("red", "s", "necessary"): {"status": "supported", "by": "a"}})[("red", "s")]
    assert "effect_below_min_fraction" in j["picture_blocked_by"] and j["status"] == "unresolved"
    # keep claims: the fraction is the recovery
    kp = keep_panel()
    k = verdict(kp, fill(kp, "red", 0, 0, keep=0.1, keep_random=0.8, allcut=1.0))[("red", "keep-s")]
    assert k["effect_fraction"]["min"] == pytest.approx(0.9) and "recovery" in k["effect_fraction"]["definition"]
    assert k["thresholds"]["parameter_sources"]["recovery_threshold"] == P.DEFAULT_SOURCE
    with pytest.raises(P.PanelError):
        small_panel(recovery_threshold=1.5)

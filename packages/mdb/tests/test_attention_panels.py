"""One panel generator for attention experiments (framework-free; tiny tensors only for the blend)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mdb import attention_panels as P
from mdb import attention_port as ap

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
    code = ("import sys; import mdb.attention_panels; "
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
    rand = set(ids["z:rand0-own2:red"].units)
    assert set(ids["z:keep-rand0-own2:red"].units) == set(UNIVERSE) - rand
    report = panel.controls["red:own2"]["reports"][0]
    assert report["strata_counts"]["matched"] and report["distinct_blocks"]["matched"]
    # other item, same units: rows of the other word, units of this word's set
    other = ids["z:own2@rows-ball:red"]
    assert other.units == ids["z:own2:red"].units and other.rows_item == "ball"
    assert other.sources == ("own:ball",) and other.condition == "iso:red" and other.separates_identity
    # nested series: added units alone
    added = ids["z:own4-minus-own2:red"]
    assert set(added.units) == {(1, 2), (3, 0)} and added.role == "test"
    # references
    assert {"iso:red", "del:red", "z:none:red", "m:none:red"} <= set(ids)
    assert ids["z:none:red"].units is None


def test_panel_records_when_random_sets_cannot_separate_identity_from_mass():
    mass = tuple(10.0 if u in {(0, 0), (2, 1)} else 0.01 for u in UNIVERSE)   # the target holds the mass
    panel = P.read_panel(tiny_spec(random=P.RandomControls(n=2, strata=STRATA, mass=mass, mass_tolerance=0.5,
                                                           disjoint=True, seed=1)))
    info = panel.identity["red:own2"]["random_cut"]
    assert info["separable"] is False
    assert "reach the target mass" in info["why"] and info["max_reachable_mass_bound"] < info["target_mass"]
    assert panel.by_id()["z:rand0-own2:red"].separates_identity is False
    assert panel.identity["red:own2"]["other_item_cut"]["separable"] is True


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
    flat = [tuple(b) for lease in leases for b in lease["branches"]]
    assert sorted(flat) == sorted((c, a.arm) for c in panel.contexts for a in panel.arms)
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
    from mdb.attention_ranking import contrast_ranking
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
        # the executed metrics carry exactly these per-word arms (plus prompt-level sham / img<-alltext)
        metrics = json.loads(HEADFREE_METRICS.read_text())
        ran = {m["arm"] for m in metrics if m["prompt"] == spec["id"]}
        assert {a.arm for a in panel.arms} == ran - {"sham", "img<-alltext"}
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
    assert sum(len(lease["branches"]) for lease in leases) == panel.branch_count


# ------------------------------------------------------------------ reproduction: value panels

def _content(rows):
    return sorted((r["kind"], tuple(sorted(tuple(c) for c in r["cells"])), float(r["dose"])) for r in rows)


def test_value_coalition_plan_is_reproduced():
    from mdb.value_coalition import LATE_CELLS, plan
    groups = {f"KV{kv}": [c for c in LATE_CELLS if c[1] == kv] for kv in range(2)}
    groups.update({f"L{layer}-both": [(layer, 0), (layer, 1)] for layer in range(19, 28)})
    panel = P.value_panel(LATE_CELLS, label="late", groups=groups, doses=(0.5,), reference_dose=0)
    assert len(panel.arms) == len(plan()) == 56
    assert _content(P.to_legacy(panel)) == _content(plan())
    assert panel.aliases == {}


def test_frozen_value_pair_and_resolver_arms_are_reproduced():
    from mdb import attention_source_resolver, frozen_value_pair
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
    from mdb.value_coalition import replacement
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


def fill(panel, item, effect, control, floor=0.01, other=None):
    """Outcomes: test arms `effect`, random controls `control`, other-item `other`, floor arms `floor`."""
    out = {}
    for a in panel.arms:
        if a.item != item:
            continue
        if a.role == "floor":
            out[a.arm] = effect if a.kind == "floor_cut_recompute" and a.floor_of in out else floor
        elif a.role == "test":
            out[a.arm] = effect
        elif a.control == "random":
            out[a.arm] = control
        elif a.control == "other_item":
            out[a.arm] = control if other is None else other
        elif a.kind == "all_cut":
            out[a.arm] = 1.0
    # twin recompute equals the cut (gap 0) unless overridden
    for a in panel.arms:
        if a.item == item and a.kind == "floor_cut_recompute":
            out[a.arm] = out[a.floor_of]
    return out


def verdict(panel, outcomes, **kw):
    rec = P.results_record(panel, {"c0": outcomes})
    return {(v["item"], v["unit_set"]): v for v in P.verdicts(panel, rec, **kw)}


def test_verdict_supported_not_supported_unresolved():
    panel = small_panel()
    v = verdict(panel, fill(panel, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "supported" and v["cut_types_agree"]
    assert set(v["cut_types"]) == {"zero", "renorm"}
    v = verdict(panel, fill(panel, "red", effect=0.5, control=0.05, other=0.495))[("red", "s")]
    assert v["status"] == "not_supported"                                      # other item, same units: same effect
    v = verdict(panel, fill(panel, "red", effect=0.005, control=0.0))[("red", "s")]
    assert v["status"] == "unresolved"
    assert v["cut_types"]["zero"]["contexts"][0]["why"] == "test effect inside its floor"


def test_floor_uses_the_larger_of_recompute_sham_and_twin_gap():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    out["z:s:red/cut_recompute"] = 0.1                                       # |delta - recompute| = 0.4
    rec = P.results_record(panel, {"c0": out})
    row = {r["arm"]: r for r in P.arm_effects(panel, rec)}["z:s:red"]
    assert row["floor"] == pytest.approx(0.4) and row["floor_parts"]["sham_recompute"] == 0.01
    assert "lower bound" in row["floor_parts"]["twin_gap_source"]
    rec = P.results_record(panel, {"c0": out}, pair_distances={"c0": {"z:s:red|z:s:red/cut_recompute": 0.45}})
    row = {r["arm"]: r for r in P.arm_effects(panel, rec)}["z:s:red"]
    assert row["floor"] == pytest.approx(0.45) and row["floor_parts"]["twin_gap_source"] == "pair distance"
    # without the pair distance the scalar gap (.4) is the floor and the .45 margin clears it
    v = verdict(panel, out)[("red", "s")]
    assert v["cut_types"]["zero"]["status"] == "supported"          # scalar gap .4 < margin .45


def test_cut_type_disagreement_does_not_count():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    for a in panel.arms:
        if a.item == "red" and a.cut_type == "renorm" and a.role == "control":
            out[a.arm] = 0.5                                                   # renorm: effect == controls
    v = verdict(panel, out)[("red", "s")]
    assert v["cut_types"]["zero"]["status"] == "supported"
    assert v["cut_types"]["renorm"]["status"] == "not_supported"
    assert v["status"] == "not_supported" and not v["cut_types_agree"] and "disagree" in v["why"]


def test_control_missing_for_absent_floor_waived_type_or_mass_confounded_controls():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    del out["z:s:red/sham_recompute"]
    assert verdict(panel, out)[("red", "s")]["status"] == "control_missing"
    one = small_panel(cut_types=("zero",), reference_types=("zero",), waive={"both_cut_types": "pilot"})
    v = verdict(one, fill(one, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "control_missing" and v["missing_cut_types"] == ["renorm"]
    assert v["waivers"] == {"both_cut_types": "pilot"}
    # random controls without mass are not identity-separating; with no other item there is no separator
    alone = small_panel(other_items={}, waive={"other_item": "single word"})
    v = verdict(alone, fill(alone, "red", effect=0.5, control=0.05))[("red", "s")]
    assert v["status"] == "control_missing"
    assert v["cut_types"]["zero"]["mass_confounded_controls"] == ["z:rand0-s:red"]


def test_keep_claim_compares_against_random_keep_and_all_cut():
    sets = {"red": {"s": [(0, 0), (1, 1)]}, "ball": {"s": [(2, 2), (3, 1)]}}
    panel = small_panel(sets=sets, cut_sets=(), keep_sets=("s",))
    out = fill(panel, "red", effect=0.1, control=0.8)                          # keep-s barely moves; random keep does
    v = verdict(panel, out)[("red", "keep-s")]
    assert v["claim"] == "sufficient" and v["status"] == "supported"
    out = fill(panel, "red", effect=0.8, control=0.1)
    assert verdict(panel, out)[("red", "keep-s")]["status"] == "not_supported"


def test_picture_judgement_overrides_scalar_and_is_recorded():
    panel = small_panel()
    out = fill(panel, "red", effect=0.5, control=0.05)
    j = {("red", "s", "necessary"): {"status": "not_supported", "by": "auditor", "note": "duplicated object"}}
    v = verdict(panel, out, judgements=j)[("red", "s")]
    assert v["scalar_status"] == "supported" and v["status"] == "not_supported" and v["overridden_by_picture"]
    j = {("red", "s", "necessary"): {"status": "supported", "by": "auditor"}}
    v = verdict(panel, out, judgements=j)[("red", "s")]
    assert v["status"] == "supported" and not v["overridden_by_picture"]
    with pytest.raises(P.PanelError):
        verdict(panel, out, judgements={("red", "s", "necessary"): {"status": "maybe", "by": "x"}})

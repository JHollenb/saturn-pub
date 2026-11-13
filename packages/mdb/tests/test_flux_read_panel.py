"""Pure-Python checks for the FLUX read-panel lane (mdb.flux_read_panels): tiny inputs, no model."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import mdb.attention_panels as P
import mdb.attention_port as ap
import mdb.flux_read_panels as F

FLUX = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
PLANS = Path(__file__).resolve().parents[1] / "qualification/flux_read_panels_20261112/plans"


def tiny_panel(contexts=("1", "2")):
    universe = [(b, h) for b in range(3) for h in range(2)]
    sets = {w: {"s2": [(0, 0), (1, 1)]} for w in ("a", "b")}
    return P.read_panel(P.ReadPanelSpec(items=("a", "b"), universe=universe, sets=sets, contexts=contexts,
                                        cut_sets=("s2",), keep_sets=("s2",),
                                        random=P.RandomControls(n=1, strata=tuple("x" for _ in universe)),
                                        other_items={"a": ("b",), "b": ("a",)}))


ITEMS = {"a": {"prompt": "p", "word": "a"}, "b": {"prompt": "p", "word": "b"}, "p": {"prompt": "p", "word": None}}


def test_module_import_loads_no_framework():
    code = ("import sys, mdb.flux_read_panels; "
            "print(int('torch' in sys.modules), int('numpy' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["0", "0"]


def test_delete_word_and_hybrid_row_plan():
    prompt = "a black cat beside a red ball"
    assert F.delete_word(prompt, "red") == "a black cat beside a ball"
    assert F.delete_word("red, blue", "red") == ", blue"
    rec_o = {"words": [{"word": "red", "rows": [5]}], "n_real": 9, "real_rows": list(range(9)),
             "input_ids": [1, 2, 3, 4, 5, 50, 6, 7, 8]}
    rec_d = {"words": [], "n_real": 8, "real_rows": list(range(8)), "input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}
    plan = F.hybrid_own_plan(rec_o, rec_d, "red")
    assert plan["own_rows"] == [5] and plan["shift"] == 1
    assert plan["later_rows"] == [6, 7, 8] and plan["later_source_rows"] == [5, 6, 7]
    assert plan["pad_from"] == 9 and plan["before_rows"] == [0, 1, 2, 3, 4] and plan["later_id_mismatch"] == 0
    assert F.hybrid_own_plan(rec_o, dict(rec_d, n_real=9, real_rows=list(range(9))), "red") is None   # no shift


def test_plan_hash_refusal_and_lease_checks():
    panel = tiny_panel()
    plan = panel.to_json()
    assert F.load_plan(plan, panel.sha256()).sha256() == panel.sha256()
    with pytest.raises(P.PanelError, match="differs from the sealed"):
        F.load_plan(plan, "0" * 64)
    edited = json.loads(json.dumps(plan))
    edited["arms"][0]["notes"] = ["edited"]
    with pytest.raises(P.PanelError, match="refusing to run"):
        F.load_plan(edited, panel.sha256())
    lease = panel.leases(64)[0]
    assert F.check_lease(panel, lease) is lease
    with pytest.raises(P.PanelError, match="different plan"):
        F.check_lease(panel, dict(lease, plan_sha256="1" * 64))
    with pytest.raises(P.PanelError, match="seal differs"):
        F.check_lease(panel, dict(lease, branches=lease["branches"][1:]))
    other = tiny_panel(contexts=("1", "3"))
    with pytest.raises(P.PanelError):
        F.check_lease(other, lease)


def test_roots_work_items_and_lease_cap():
    panel = tiny_panel()
    lease = panel.leases(1000)[0]
    roots = F.lease_roots(panel, lease, ITEMS)
    assert set(roots) == {(c, k, "p", w) for c in ("1", "2") for w in ("a", "b") for k in ("del", "iso")} | {
        (c, "native", "p", None) for c in ("1", "2")}
    assert F.work_items(panel, lease, ITEMS) == panel.branch_count + 10
    cap, leases = F.leases_within(panel, ITEMS, limit=40)
    assert all(F.work_items(panel, x, ITEMS) <= 40 for x in leases)
    assert sum(len(x["branches"]) for x in leases) == panel.branch_count


def test_cut_dict_round_trip():
    cuts = [ap.cut(["image", [3, 1, 2]], ["own:a"], heads=[(0, 1), (2, 0)], renormalize=False),
            ap.cut(["image"], ["text"], heads=None, renormalize=True),
            ap.cut([[5]], [[1, 2]], heads=[1, (0, 0)], blocks=[0, 2], steps=[1], renormalize=False,
                   zero_method="recompute")]
    cuts += [ap.sham(cuts[0]), ap.sham(renormalize=False), *ap.floor_arms(cuts[0]).values(), *ap.floor_arms(cuts[1]).values()]
    for c in cuts:
        back = F.cut_from_dict(json.loads(json.dumps(c.to_dict())))
        assert back == c
        assert back.to_dict() == c.to_dict()


def _recipe(**over):
    body = dict(schema=F.RECIPE_SCHEMA,
                arm=dict(context="4242", arm="z:own60:ball", pass_tag="t"),
                model=dict(model_identity="m" * 64),
                software=dict(torch="2.13.0", diffusers="0.39.0"),
                input=dict(prompt="a red ball", conditioning=dict(kind="hybrid_own", word="ball")),
                noise=dict(seed=4242, height=256, width=256, steps=4),
                port=dict(precision="native", zero_method="delta", forwards=4,
                          cuts=[ap.cut(["image"], ["own:ball"], heads=[(0, 1)], renormalize=False).to_dict()]),
                expected=dict(latent_sha256="a" * 64, image_rgb_sha256="b" * 64, conditioning_sha256="c" * 64))
    body.update(over)
    return body


def test_recipe_seal_validate_and_refusals():
    sealed = F.seal_recipe(_recipe())
    assert F.validate_recipe(sealed)["fingerprint"] == sealed["fingerprint"]
    tampered = json.loads(json.dumps(sealed))
    tampered["noise"]["seed"] = 9001
    with pytest.raises(P.PanelError, match="seal differs"):
        F.validate_recipe(tampered)
    with pytest.raises(P.PanelError, match="already sealed"):
        F.seal_recipe(sealed)
    bad = [_recipe(extra=1), _recipe(port=dict(_recipe()["port"], forwards=8)),
           _recipe(input=dict(prompt="x", conditioning=dict(kind="deletion"))),
           _recipe(expected=dict(latent_sha256="a" * 64)), _recipe(noise=dict(seed=-1, height=1, width=1, steps=4)),
           _recipe(port=dict(_recipe()["port"], precision="fast"))]
    for body in bad:
        with pytest.raises(P.PanelError):
            F.seal_recipe(body)
    assert F.seal_recipe(_recipe(port=None, input=dict(prompt="x", conditioning=dict(kind="native"))))


def test_compare_regeneration_reports_hash_equality_and_lane_drift():
    sealed = F.seal_recipe(_recipe())
    same = {"latent_sha256": "a" * 64, "image_rgb_sha256": "b" * 64, "conditioning_sha256": "c" * 64}
    row = F.compare_regeneration(sealed, same, lane={"torch": "2.13.0", "diffusers": "0.39.0"})
    assert row["match"] and row["same_lane"] and row["lane_drift"] == {}
    row = F.compare_regeneration(sealed, dict(same, latent_sha256="d" * 64), lane={"torch": "2.14.0"})
    assert not row["match"] and not row["hashes"]["latent_sha256"]["equal"] and row["hashes"]["image_rgb_sha256"]["equal"]
    assert row["lane_drift"] == {"torch": {"recipe": "2.13.0", "now": "2.14.0"}} and not row["same_lane"]


def test_gone_score_matches_hand_computation():
    # MAD(native, del) = 0.2; d = 1 - MAD(arm, del) / 0.2
    px = {("1", "iso:w"): 0.16, ("1", "z:none:w"): 0.04, ("1", "z:s:w"): 0.10,
          ("2", "iso:w"): 0.10, ("2", "z:none:w"): 0.02, ("2", "z:s:w"): 0.10}
    table = F.gone_table(px, {("1", "w"): 0.2, ("2", "w"): 0.2}, ["w"], ["z:s:{w}"], contexts=["1", "2"],
                         floors={("1", "z:s:w"): 0.012})
    (row,) = table
    c1, c2 = row["contexts"]
    assert c1["span"] == pytest.approx((1 - 0.04 / 0.2) - (1 - 0.16 / 0.2))
    assert c1["gone"]["z:s:{w}"] == pytest.approx(0.5) and c2["gone"]["z:s:{w}"] == pytest.approx(0.0)
    assert row["mean"]["z:s:{w}"] == pytest.approx(0.25)
    assert c1["floor_gone_bound"]["z:s:{w}"] == pytest.approx(0.012 / (0.2 * 0.6))
    assert F.d_score(0.05, 0.2) == pytest.approx(0.75)
    with pytest.raises(P.PanelError):
        F.gone(0.5, 0.2, 0.2)


@pytest.mark.skipif(not (FLUX / "runs/headfree/result/metrics.json").exists(), reason="FLUX headfree run not on disk")
def test_gone_table_reproduces_the_saved_headfree_table():
    metrics = json.loads((FLUX / "runs/headfree/result/metrics.json").read_text())
    old = json.loads((FLUX / "runs/headfree/headfree_table.json").read_text())
    px = {(str(m["seed"]), m["arm"]): m["px_mad_del"] for m in metrics if "px_mad_del" in m}
    base = {(str(m["seed"]), m["arm"][4:]): m["px_mad"] for m in metrics if m["arm"].startswith("del:")}
    cols = {c: (f"{c}:{{w}}" if c.startswith("m:") else f"z:{c}:{{w}}") for c in old[0]["mean"]}
    for row in old:
        (new,) = F.gone_table(px, base, [row["word"]], list(cols.values()), contexts=["4242", "9001"])
        for i, seed_row in enumerate(row["seeds"]):
            for c, template in cols.items():
                assert new["contexts"][i]["gone"][template] == pytest.approx(seed_row[c], abs=1e-12)


def test_results_from_rows_records_twin_gap_and_merge_refuses_duplicates():
    panel = tiny_panel(contexts=("1",))
    ids = panel.by_id()
    rows = [{"context": "1", "arm": a, "vs_reference": {"px_mad": 0.01 * i}} for i, a in enumerate(ids)]
    twin = next(a for a in panel.arms if a.kind == "floor_cut_recompute")
    for r in rows:
        if r["arm"] == twin.arm:
            r["vs_floor_of"] = {"px_mad": 0.003}
    rows.append({"context": "1", "arm": "not-in-plan", "vs_reference": {"px_mad": 1.0}})
    rec = F.results_from_rows(panel, rows)
    assert P.attach_results(panel, rec) is rec
    assert rec["pair_distances"]["1"] == {f"{twin.floor_of}|{twin.arm}": 0.003}
    half = len(rows) // 2
    merged = F.merge_results(panel, [F.results_from_rows(panel, rows[:half]), F.results_from_rows(panel, rows[half:])])
    assert merged["outcomes"] == rec["outcomes"]
    with pytest.raises(P.PanelError, match="two results"):
        F.merge_results(panel, [rec, rec])


def test_prompt_level_arms_map_to_port_cuts_and_legacy_names():
    arms = {a.arm: a for a in F.prompt_level_arms("cat")}
    assert set(arms) == {"native:cat", "z:img<-alltext:cat", "z:img<-alltext:cat/sham_recompute",
                         "z:img<-alltext:cat/cut_recompute", "m:img<-alltext:cat", "m:img<-alltext:cat/sham"}
    assert P.port_cuts(arms["native:cat"]) == ()
    (m,) = P.port_cuts(arms["m:img<-alltext:cat"])
    assert m == ap.cut(["image"], ["text"], renormalize=True)
    (f,) = P.port_cuts(arms["z:img<-alltext:cat/cut_recompute"])
    assert f == ap.floor_arms(ap.cut(["image"], ["text"], renormalize=False))["cut_recompute"]
    assert [a.arm for a in F.prompt_level_arms("cat", cut_types=("renorm",), floors=False)] == [
        "native:cat", "m:img<-alltext:cat"]
    assert F.legacy_file_name("cat", 4242, "native:cat") == "native.cat.4242.png"
    assert F.legacy_file_name("cat", 4242, "m:img<-alltext:cat") == "arm.cat.4242.img_alltext.png"
    assert F.legacy_file_name("cat", 9001, "z:keep-own60:red") == "arm.cat.9001.z_keep_own60_red.png"
    assert F.legacy_file_name("cat", 9001, "z:own60:red/sham_recompute") is None


def test_rows_runs_and_row_table():
    assert F.runs([5, 1, 2, 3, 9, 10]) == [[1, 4], [5, 6], [9, 11]]
    assert F.unruns(F.runs([5, 1, 2, 3, 9, 10])) == [1, 2, 3, 5, 9, 10]
    table = F.row_table({"red": [2]}, 4, 7)
    assert table["image"] == [4, 5, 6] and table["text"] == [0, 1, 2, 3]
    assert table["own:red"] == [2] and table["others:red"] == [0, 1, 3]


def test_merge_panels_keeps_contexts_and_refuses_mismatch():
    a, b = tiny_panel(), tiny_panel(contexts=("1", "3"))
    merged = F.merge_panels([a], extra_arms=F.prompt_level_arms("p", floors=False, cut_types=("renorm",)))
    assert merged.contexts == a.contexts and len(merged.arms) == len(a.arms) + 2
    with pytest.raises(P.PanelError):
        F.merge_panels([a, b])


@pytest.mark.skipif(not (PLANS / "legacy.plan.json").exists() or not (FLUX / "runs/headfree/result/metrics.json").exists(),
                    reason="sealed plans or FLUX headfree run not on disk")
def test_sealed_plans_contain_every_executed_headfree_arm():
    metrics = json.loads((FLUX / "runs/headfree/result/metrics.json").read_text())
    seal = json.loads((PLANS / "legacy.seal.json").read_text())
    legacy = F.load_plan(json.loads((PLANS / "legacy.plan.json").read_text()), seal["plan_sha256"])
    full = F.load_plan(json.loads((PLANS / "full.plan.json").read_text()),
                       json.loads((PLANS / "full.seal.json").read_text())["plan_sha256"])
    rename = {"sham": "native:{p}", "img<-alltext": "m:img<-alltext:{p}"}
    executed = {(str(m["seed"]), rename.get(m["arm"], m["arm"]).format(p=m["prompt"])) for m in metrics}
    planned = {(c, a.arm) for a in legacy.arms for c in legacy.contexts}
    assert planned == executed and len(executed) == 224
    sig = lambda a: (a.kind, a.cut_type, a.receivers, a.sources, a.units, a.condition)
    new = full.by_id()
    for a in legacy.arms:
        assert sig(new[a.arm]) == sig(a), a.arm

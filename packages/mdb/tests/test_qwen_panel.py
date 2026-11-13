"""Qwen value / read-cut panels: pure logic, plus a CPU dry run of the worker on a tiny random model."""
import importlib.util
import json
from pathlib import Path

import pytest

from mdb import attention_panels as P
from mdb import qwen_panels as Q

QUAL = Path(__file__).resolve().parents[1] / "qualification"
MAIN02 = Path("/Users/jakeholl/domains/artifacts/mdb-value-coalition-20261112/main-02")


def load(name):
    spec = importlib.util.spec_from_file_location(name, QUAL / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_attention(strong=Q.NAMED_HEADS.values()):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    a = rng.random((Q.LAYERS, Q.HEADS, Q.RECEIVER_ROW + 1)) * 0.1
    a[:, :, 0] += 2.0                                              # attention sink at row 0
    for layer, head in strong:
        a[layer, head, Q.SOURCE_ROW] += 3.0
    return a / a.sum(-1, keepdims=True)


# ------------------------------------------------------------------------------------ plans

def test_value_plan_is_value_coalition_plan_and_names_map_to_the_saved_run():
    from mdb.value_coalition import plan
    for lane in Q.LANES:
        panel = Q.value_plan(lane)
        assert len(panel.arms) == 56 and panel.aliases == {} and panel.port["lane"] == lane
        assert {Q.legacy_name(a.arm) for a in panel.arms} == {r["arm"] for r in plan()}
        legacy = {r["arm"]: r for r in plan()}
        for a in panel.arms:
            old = legacy[Q.legacy_name(a.arm)]
            assert sorted(map(tuple, old["cells"])) == sorted(a.units) and float(old["dose"]) == a.dose
    assert Q.value_plan("float32").sha256() != Q.value_plan("bfloat16").sha256()
    with pytest.raises(Q.QwenPanelError):
        Q.value_plan("float16")


def test_value_specs_resolve_rows_and_geometry_is_checked():
    panel = Q.value_plan("float32")
    ids = panel.by_id()
    assert Q.resolve_value(ids["native-recipient"]) is None
    wt, wp = Q.resolve_value(ids["wrong-target-late"]), Q.resolve_value(ids["wrong-payload-late"])
    assert (wt["target_row"], wt["payload_row"]) == (17, 18) and (wp["target_row"], wp["payload_row"]) == (18, 17)
    rec = Q.resolve_value(ids["reciprocal-late"])
    assert P.run_side(ids["reciprocal-late"]) == "donor" and rec["payload_side"] == "recipient"
    assert Q.resolve_value(ids["dose0.5-late"])["dose"] == 0.5
    assert Q.check_value_geometry(panel, 28, 2)
    with pytest.raises(Q.QwenPanelError, match="live model"):
        Q.check_value_geometry(panel, 24, 2)


def test_native_recipes_cover_copy_arms_and_name_what_is_not_expressible():
    from mdb.native_replay_recipe import seal, validate
    panel = Q.value_plan("float32")
    expressible, not_expressible = [], {}
    for arm in panel.arms:
        body, why = Q.native_recipe(arm, model_identity="m", execution={"dtype": "x"})
        if body is None:
            not_expressible[arm.kind] = why
        else:
            validate(seal(body))
            expressible.append(arm.kind)
            edits = [e for e in body["events"] if e["kind"] == "copy-values"]
            assert len(edits) == (0 if arm.kind in ("native", "replay", "donor_native") else 1)
    assert not_expressible == {"dose": "dose 0.5: copy-values copies whole rows; no blend event"}
    assert set(expressible) >= {"native", "replay", "donor_native", "sham", "coalition", "singleton", "complement",
                                "wrong_target", "wrong_payload", "reciprocal"}
    body, _ = Q.native_recipe(panel.by_id()["wrong-target-late"], model_identity="m", execution={})
    edit = next(e for e in body["events"] if e["kind"] == "copy-values")
    assert (edit["target_row"], edit["source_row"], edit["source"]) == (17, 18, "donor-parent")
    stages = [e["token_id"] for e in body["events"] if e["session"] == "recipient" and e["kind"] == "prompt-token"]
    assert stages == list(Q.RECIPIENT_IDS[19:]) and body["roots"][0]["token_ids"] == list(Q.RECIPIENT_IDS[:19])
    read = Q.read_plan("float32", fake_attention())
    assert Q.native_recipe(read.by_id()["z:L20H5:freezing"], model_identity="m", execution={})[0] is None
    native_read, _ = Q.native_recipe(read.by_id()["native"], model_identity="m", execution={})
    assert native_read is not None and not any(e["kind"] == "copy-values" for e in native_read["events"])


def test_label_free_ranking_recovers_planted_heads_and_carries_trivial_baselines():
    from mdb.attention_ranking import require_trivial_baselines
    sets, summary = Q.label_free_sets(fake_attention())
    assert {tuple(u) for u in sets["rank3"]} == set(Q.NAMED_HEADS.values())
    assert summary["recovered"]["rank3"] == sorted(Q.NAMED_HEADS)
    for record in summary["top_source_baselines"].values():
        require_trivial_baselines(record)
    assert summary["top_source_baselines"]["named3"]["nominator"]["top_source_agreement"] == 1.0
    assert summary["top_source_baselines"]["all_heads"]["best_trivial"]["name"] == "position_0"
    with pytest.raises(Q.QwenPanelError, match="receiver attention"):
        Q.label_free_sets(fake_attention()[:, :, :10])


def test_read_plan_has_named_and_label_free_sets_same_layer_random_heads_and_other_item():
    panel = Q.read_plan("float32", fake_attention())
    ids = panel.by_id()
    assert panel.check_universe(28, 12) and panel.port["lane"] == "float32"
    for name in ("L20H5", "L21H0", "L20H11", "named3", "rank3", "rank12"):
        for t in ("z", "m"):
            assert f"{t}:{name}:freezing" in ids and f"{t}:keep-{name}:freezing" in ids
            assert f"{t}:{name}@rows-icy:freezing" in ids
    assert ids["z:named3:freezing"].units == ((20, 5), (20, 11), (21, 0))
    assert ids["z:named3:freezing"].receivers == ("receiver",) and ids["z:named3:freezing"].sources == ("own:freezing",)
    assert ids["z:named3@rows-icy:freezing"].sources == ("own:icy",)
    keep = ids["z:keep-L20H5:freezing"]
    assert len(keep.units) == 335 and (20, 5) not in keep.units
    for k in range(2):
        rand = ids[f"z:rand{k}-named3:freezing"]
        assert sorted(layer for layer, _ in rand.units) == [20, 20, 21]          # same layers
        assert not set(rand.units) & {(20, 5), (20, 11), (21, 0)}               # disjoint
    assert P.port_kwargs(panel, Q.row_table())["rows"] == {"receiver": [21], "own:freezing": [18], "own:icy": [2]}
    # per-item mass: freezing draws are matched against attention to row 18, icy against row 2
    assert panel.controls["freezing:L20H5"]["reports"][0]["mass"]["target"] > 0.4
    assert panel.controls["icy:L20H5"]["reports"][0]["mass"]["target"] < 0.1
    smoke = Q.smoke_read_plan("bfloat16")
    assert {a.kind for a in smoke.arms} >= {"cut", "floor_sham", "floor_sham_recompute", "floor_cut_recompute",
                                            "all_cut", "native"}


# --------------------------------------------------------------------------------- analysis

def margins_from(values):
    return {arm: Q.bounded_margin(arm, {"hot": v, "cold": 0.0}, "float32") for arm, v in values.items()}


def test_l20_fit_coefficients_and_bounds():
    panel = Q.value_plan("float32")
    values = {a.arm: 0.0 for a in panel.arms}
    values.update({"native-recipient": -4.75, "only-L20-KV0": -0.875, "only-L20-KV1": -3.875,
                   "coalition-L20-both": 0.375, "coalition-late": 3.375, "coalition-KV0": 3.875, "coalition-KV1": -4.75})
    m = margins_from(values)
    fit = Q.l20_fit(m)
    assert fit["c0"].value == -4.75 and fit["ca (L20/KV0)"].value == 3.875 and fit["cb (L20/KV1)"].value == 0.875
    assert fit["cab"].value == pytest.approx(0.375)
    coeffs = Q.coefficients(m)
    assert coeffs["interaction L20"].value == pytest.approx(0.375)
    assert coeffs["KV_group_interaction"].value == pytest.approx(3.375 - 3.875 + 4.75 + -4.75)
    # BF16-produced logits carry a bound; an interaction smaller than it is unresolved
    b = {arm: Q.bounded_margin(arm, {"hot": 20.0 + v, "cold": 20.0}, "bfloat16") for arm, v in values.items()}
    assert not Q.l20_fit(b)["cab"].resolved and Q.l20_fit(b)["cab"].bound > 0


@pytest.mark.skipif(not (MAIN02 / "analysis/arms.json").exists(), reason="main-02 artifacts not on this machine")
def test_coefficients_match_value_coalition_bounds_on_the_saved_run():
    bounds = load("value_coalition_bounds")
    saved = json.loads((MAIN02 / "analysis/arms.json").read_text())
    from mdb.analysis_metrics import Bounded
    panel = Q.value_plan("bfloat16")
    rows = []
    for a in panel.arms:
        old = next(s for s in saved if s["arm"] == Q.legacy_name(a.arm))
        rows.append({"arm": a.arm, "tracked": {w: old["words"][w]["logit"] for w in ("hot", "cold")}})
    mine = Q.coefficients(Q.value_margins(rows, "bfloat16"))
    legacy_margin = {}
    for s in saved:
        hot = Bounded(s["words"]["hot"]["logit"], 0.0)
        legacy_margin[s["arm"]] = hot - s["words"]["cold"]["logit"]
    theirs = bounds.coefficients(legacy_margin)
    assert set(mine) == set(theirs)
    assert all(mine[k].value == pytest.approx(theirs[k].value, abs=1e-12) for k in mine)
    # the saved +0.375 L20 interaction does not exceed its BF16 readout bound
    assert mine["interaction L20"].value == 0.375 and not mine["interaction L20"].resolved
    by_name = {s["arm"]: s for s in saved}
    same = [{"arm": r["arm"], "tokens_text": [by_name[Q.legacy_name(r["arm"])]["own_text"]],
             "margin": by_name[Q.legacy_name(r["arm"])]["hot_minus_cold"]} for r in rows]
    agreement = Q.saved_agreement(saved, same)
    assert agreement["arms"] == 56 and agreement["margin_equal"] == 56 and not agreement["differences"]
    same[0] = dict(same[0], margin=same[0]["margin"] + 0.125)
    assert [d["arm"] for d in Q.saved_agreement(saved, same)["differences"]] == [same[0]["arm"]]


def test_read_results_outcomes_and_twin_pairs():
    panel = Q.smoke_read_plan("float32")
    rows = [{"arm": a.arm, "margin": -4.75} for a in panel.arms]
    by = {r["arm"]: r for r in rows}
    by["z:L20H5:freezing"]["margin"] = -2.75
    by["z:L20H5:freezing/cut_recompute"]["margin"] = -2.5
    rec = Q.read_results(panel, rows)
    out = rec["outcomes"][Q.CONTEXT]
    assert out["z:L20H5:freezing"] == 2.0 and out["m:L20H5:freezing"] == 0.0 and "native" not in out
    assert rec["pair_distances"][Q.CONTEXT]["z:L20H5:freezing|z:L20H5:freezing/cut_recompute"] == 0.25
    assert P.attach_results(panel, rec) is rec
    assert Q.signed_shift(rows, "z:L20H5:freezing") == 2.0
    with pytest.raises(Q.QwenPanelError, match="reference"):
        Q.read_results(panel, [r for r in rows if r["arm"] != "native"])


def test_max_source_attention_and_correlations():
    a = fake_attention()
    cells = Q.max_source_attention(a)
    assert cells["L20/KV0"] == pytest.approx(a[20, :6, 18].max()) and cells["L20/KV1"] == pytest.approx(a[20, 6:, 18].max())
    panel = Q.value_plan("float32")
    values = {arm.arm: 0.1 * i for i, arm in enumerate(panel.arms)}
    report = Q.cell_correlations(cells, Q.coefficients(margins_from(values)))
    assert set(report) == {"singleton", "complement_loss"}
    assert report["singleton"]["n"] == 18 and "spearman" in report["singleton"]
    assert report["singleton"]["leave_one_out"]["pearson"]["min"] is not None


# --------------------------------------------------------------------- worker dry run (CPU)

class TinyTokenizer:
    texts = {Q.RECIPIENT_TEXT: list(Q.RECIPIENT_IDS), Q.DONOR_TEXT: list(Q.DONOR_IDS),
             **{" " + w: [t] for w, t in Q.TRACKED.items()}}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(self.texts[text])}

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import mdb.ar.model as armodel
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=65905, hidden_size=48, intermediate_size=32, num_hidden_layers=28,
                      num_attention_heads=12, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True)
    cfg._attn_implementation = "eager"
    base = Qwen2ForCausalLM(cfg).eval()
    state = {k: v.clone() for k, v in base.state_dict().items()}

    def fake_load(model, *, device, torch_dtype, attn_implementation):
        assert attn_implementation == "eager" and torch_dtype == torch.float32
        m = Qwen2ForCausalLM(cfg)
        m.load_state_dict(state)
        m = m.to(torch_dtype).eval()
        from types import SimpleNamespace
        return SimpleNamespace(family="qwen2", registry_name=model), m, TinyTokenizer()

    monkeypatch.setattr(armodel, "load_ar_model", fake_load)
    stage = load("qwen_panel_stage")
    plans = tmp_path / "plans"
    stage.build_plans(plans, "smoke")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"tiny")
    payload = tmp_path / "payload"
    payload.mkdir()
    for p in plans.iterdir():
        (payload / p.name).write_bytes(p.read_bytes())
    (payload / "qwen_panel_worker.py").write_bytes((QUAL / "qwen_panel_worker.py").read_bytes())
    config = stage.run_config(plans, device="cpu", dry_run=True, output_scratch=str(tmp_path / "out"),
                              model_path=str(model_dir))
    config["expected_weight_file_sha256"] = stage.sha(model_dir / "model.safetensors")
    monkeypatch.chdir(payload)
    monkeypatch.setenv("MRUN_JOB_ID", "dry")

    def run(cfg_edit=None):
        c = json.loads(json.dumps(config))
        if cfg_edit:
            cfg_edit(c)
        (payload / "saturn_run_config.json").write_text(json.dumps(c))
        (payload / "saturn_debug_request.json").write_text(json.dumps({"request_sha256": "x"}))
        monkeypatch.setenv("SATURN_RUN_CONFIG", str(payload / "saturn_run_config.json"))
        monkeypatch.setenv("SATURN_DEBUG_REQUEST", str(payload / "saturn_debug_request.json"))
        load("qwen_panel_worker").worker()
        return tmp_path / "out" / "dry" / "qwen-panels"
    return run, tmp_path


def test_worker_dry_run_on_a_tiny_model(tiny_run):
    run, tmp_path = tiny_run
    root = run()
    summary = json.loads((root / "summary.json").read_text())
    assert set(summary["lanes"]) == {"float32", "bfloat16"}
    assert {m: summary["modules"][m]["sha256"] for m in summary["modules"]}
    for lane in ("float32", "bfloat16"):
        lane_root = root / lane
        value = [json.loads(line) for line in (lane_root / f"value-{lane}" / "rows.jsonl").read_text().splitlines()]
        assert [r["arm"] for r in value] == ["native-recipient", "replay-recipient", "native-donor", "sham-late",
                                             "coalition-L20-both"]
        by = {r["arm"]: r for r in value}
        assert by["replay-recipient"]["margin"] == by["native-recipient"]["margin"]       # determinism
        assert by["sham-late"]["margin"] == by["native-recipient"]["margin"]              # exact copy of itself
        assert by["native-recipient"]["logits_dtype"] == lane and len(by["native-recipient"]["tokens"]) == 3
        assert by["coalition-L20-both"]["to_reference"]["max_abs_logit"] >= 0
        recipes = [json.loads(line) for line in (lane_root / f"value-{lane}" / "recipes.jsonl").read_text().splitlines()]
        assert all("native_replay_recipe" in r for r in recipes)
        read = [json.loads(line) for line in (lane_root / f"read-smoke-{lane}" / "rows.jsonl").read_text().splitlines()]
        rb = {r["arm"]: r for r in read}
        assert rb["z:L20H5:freezing"]["port"]["removed_reads"] == 1           # one head, one receiver, one source
        assert rb["z:none:freezing"]["port"]["removed_reads"] == 28            # port log: (receiver, source) cells per block
        assert rb["z:L20H5:freezing/sham_recompute"]["port"]["removed_reads"] == 0
        results = json.loads((lane_root / f"read-smoke-{lane}" / "results.json").read_text())
        assert results["plan_sha256"] == json.loads(
            (lane_root / f"read-smoke-{lane}" / "plan-identity.json").read_text())["plan_sha256"]
        # the native replay recipe regenerates the worker's logits (same model, same row order)
        for check in summary["lanes"][lane]["replay_check"]:
            assert check["tokens_native_replay"] and check["max_abs_logit_vs_worker"] >= 0
    fp32 = summary["lanes"]["float32"]["replay_check"]
    assert all(c["max_abs_logit_vs_worker"] < 1e-4 for c in fp32), fp32


def test_worker_refuses_a_changed_plan_hash_or_lane(tiny_run):
    run, _ = tiny_run

    def wrong_hash(c):
        c["lanes"][0]["plans"][0]["plan_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="differs from the config"):
        run(wrong_hash)


def test_worker_refuses_a_plan_from_another_lane(tiny_run):
    run, _ = tiny_run

    def swap(c):
        c["lanes"][0]["plans"][0], c["lanes"][1]["plans"][0] = c["lanes"][1]["plans"][0], c["lanes"][0]["plans"][0]

    with pytest.raises(RuntimeError, match="built for lane"):
        run(swap)

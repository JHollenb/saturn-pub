"""Contrast ranking, matched control sets and mandatory trivial source baselines (numpy only)."""
import json
from pathlib import Path

import numpy as np
import pytest

from mdb.attention_ranking import (
    contrast_ranking,
    contrast_score,
    evaluate_route_forecast,
    evaluate_top_source,
    group_contrast_ranking,
    matched_control_sets,
    require_trivial_baselines,
    shares,
    trivial_route_baselines,
)
from mdb.attention_source_resolver import evaluate_top_source as resolver_evaluate
from mdb.attention_source_resolver import nominate

FLUX = Path("/Users/jakeholl/domains/saturn/experiments/2026-11-11-flux-read-graph")
CAPTURE = FLUX / "runs/full-a/result/capture.cat.4242.npz"
TOKENS = FLUX / "runs/full-a/result/tokens.json"
SETS = FLUX / "head_sets3.json"


def test_contrast_score_matches_the_reference_formula_and_ranks_specific_units_first():
    rng = np.random.default_rng(0)
    profiles = rng.random((5, 3, 4))
    profiles[2, 1, 3] += 5.0  # unit (1, 3) is specific to item 2
    s = shares(profiles)
    rest = s[[0, 1, 3, 4]].mean(0)
    expected = s[2] * np.log((s[2] + 1e-9) / (rest + 1e-9))
    result = contrast_ranking(profiles, 2, top=3)
    np.testing.assert_allclose(result["scores"], expected.ravel())
    np.testing.assert_allclose(contrast_score(s[2], rest), expected)
    assert result["top"][0] == (1, 3)
    assert result["order"][0] == np.ravel_multi_index((1, 3), (3, 4))
    # Scaling an item's raw mass does not change its shares or its ranking.
    scaled = profiles.copy()
    scaled[2] *= 17.0
    assert np.array_equal(contrast_ranking(scaled, 2)["order"], result["order"])


def test_rest_items_baseline_and_tie_order_are_explicit():
    profiles = np.array([[1.0, 1.0, 2.0], [1.0, 1.0, 1.0], [5.0, 1.0, 1.0]])
    by_rest = contrast_ranking(profiles, 0, rest_items=[1])
    by_baseline = contrast_ranking(profiles, 0, baseline=[2.0, 2.0, 2.0])
    np.testing.assert_allclose(by_rest["scores"], by_baseline["scores"])
    assert by_rest["order"].tolist() == [2, 0, 1]  # equal scores at units 0 and 1: lower index first
    with pytest.raises(ValueError):
        contrast_ranking(profiles, 0, rest_items=[0, 1])
    with pytest.raises(ValueError):
        contrast_ranking(profiles, 0, rest_items=[1], baseline=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        contrast_ranking(-profiles, 0)
    with pytest.raises(ValueError):
        contrast_ranking(np.zeros((2, 3)), 0)


def test_group_contrast_matches_leave_one_prompt_out_reference():
    rng = np.random.default_rng(3)
    profiles = rng.random((8, 2, 5))
    labels = ["color", "object", None, "color", "object", "color", "scene", "object"]
    prompt = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    include = prompt != 0
    result = group_contrast_ranking(profiles, labels, "color", include=include)
    s = shares(profiles)
    train = [i for i in range(8) if include[i] and labels[i] is not None]
    mean = s[[i for i in train if labels[i] == "color"]].mean(0)
    base = s[train].mean(0)
    reference = np.argsort((mean * np.log((mean + 1e-9) / (base + 1e-9))).ravel())[::-1]
    assert result["order"].tolist() == reference.tolist()
    assert result["members"] == 2
    with pytest.raises(ValueError):
        group_contrast_ranking(profiles, labels, "light")


@pytest.mark.skipif(not (CAPTURE.exists() and TOKENS.exists() and SETS.exists()),
                    reason="FLUX read-graph capture not present on this machine")
def test_reproduces_flux_head_sets3_own_sets_from_the_saved_capture():
    tokens = json.loads(TOKENS.read_text())
    tok_w = np.load(CAPTURE)["tok_w"].astype(np.float64)
    words = [(w["word"].strip(",.").lower(), w["rows"]) for w in tokens["cat"]["words"] if w["rows"]]
    profiles = np.stack([tok_w[..., rows].sum((0, -1)) for _, rows in words])
    reference = json.loads(SETS.read_text())["sets"]
    for word in ("ball", "red", "black", "snow"):
        (item,) = [i for i, (w, _) in enumerate(words) if w == word]
        rest = [i for i, (w, _) in enumerate(words) if w != word]
        ranking = contrast_ranking(profiles, item, rest_items=rest, top=240)
        for n in (60, 120, 240):
            assert {tuple(u) for u in ranking["top"][:n]} == {
                tuple(u) for u in reference[f"cat:{word}"][f"own{n}"]}, (word, n)


def flux_like_units(blocks=10, heads=6, joint=3):
    n = blocks * heads
    block = np.arange(n) // heads
    stratum = (block >= joint).astype(int)
    return n, block, stratum


def test_control_sets_match_strata_and_blocks_and_are_deterministic():
    n, block, stratum = flux_like_units()
    target = np.array([0, 1, 2, 7, 20, 21, 33, 40, 41, 59])
    a = matched_control_sets(target, n, strata=stratum, blocks=block, seed=5, n_sets=3)
    b = matched_control_sets(target, n, strata=stratum, blocks=block, seed=5, n_sets=3)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x["units"], y["units"])
        report = x["report"]
        assert report["strata_counts"]["matched"]
        assert report["distinct_blocks"]["matched"]
        assert report["distinct_blocks"]["mode"] == "per-stratum"
        assert len(x["units"]) == len(target)
        assert report["unmatched"] == []
    other = matched_control_sets(target, n, strata=stratum, blocks=block, seed=6)[0]
    assert not np.array_equal(other["units"], a[0]["units"])


def test_control_report_exposes_unmatched_mass_and_disjointness():
    n, block, stratum = flux_like_units()
    mass = np.full(n, 0.1)
    target = np.arange(0, 12, 2)
    mass[target] = 5.0  # the target holds nearly all of the mass
    plain = matched_control_sets(target, n, strata=stratum, mass=mass, seed=1)[0]["report"]
    assert plain["mass"]["matched"] is None
    assert "mass" in plain["unmatched"]
    assert plain["mass"]["target"] > plain["mass"]["control"]
    disjoint = matched_control_sets(target, n, strata=stratum, blocks=block, mass=mass,
                                    mass_tolerance=0.5, disjoint=True, seed=1)[0]
    assert disjoint["report"]["overlap_with_target"] == 0
    assert disjoint["report"]["disjoint"]["matched"]
    assert disjoint["report"]["mass"]["matched"] is False  # infeasible without the target units
    assert "mass" in disjoint["report"]["unmatched"]


def test_mass_matching_swaps_preserve_structure():
    n, block, stratum = flux_like_units()
    rng = np.random.default_rng(2)
    mass = rng.random(n)
    target = np.sort(np.argsort(mass)[::-1][:8])  # heaviest units
    result = matched_control_sets(target, n, strata=stratum, blocks=block, mass=mass,
                                  mass_tolerance=0.05, seed=4)[0]
    report = result["report"]
    assert report["strata_counts"]["matched"] and report["distinct_blocks"]["matched"]
    assert report["mass"]["matched"], report
    assert abs(report["mass"]["control"] - mass[result["units"]].sum()) < 1e-12


def test_mixed_strata_blocks_fall_back_to_total_block_matching():
    n = 24
    block = np.arange(n) // 4
    stratum = np.arange(n) % 2  # each block holds both strata
    target = np.array([0, 1, 5, 9, 14, 22])
    result = matched_control_sets(target, n, strata=stratum, blocks=block, seed=9)[0]["report"]
    assert result["distinct_blocks"]["mode"] == "total"
    assert result["distinct_blocks"]["matched"] and result["strata_counts"]["matched"]


def test_trivial_baselines_are_always_reported_and_can_win():
    # Native top source is position 0 on most rows (an attention sink).
    target = np.array([0, 0, 0, 2, 0, 0])
    visible = np.array([3, 4, 5, 6, 7, 8])
    nominated = np.array([2, 3, 4, 2, 6, 7])  # attends to self: right once
    record = evaluate_top_source(nominated, target, visible=visible)
    require_trivial_baselines(record)
    b = record["baselines"]
    assert b["position_0"]["top_source_agreement"] == pytest.approx(5 / 6)
    assert b["offset_0_from_end"]["top_source_agreement"] == pytest.approx(0)
    assert b["previous_token"]["top_source_agreement"] == pytest.approx(0)
    assert b["uniform"]["top_source_agreement"] == pytest.approx(np.mean(1 / visible))
    assert record["nominator"]["top_source_agreement"] == pytest.approx(1 / 6)
    assert record["best_trivial"]["name"] == "position_0"
    assert not record["beats_every_trivial_baseline"]
    assert record["margin_over_best_trivial"] == pytest.approx(1 / 6 - 5 / 6)
    with pytest.raises(ValueError):
        require_trivial_baselines({"nominator": record["nominator"]})
    with pytest.raises(ValueError):
        evaluate_top_source(nominated, np.array([0, 0, 0, 9, 0, 0]), visible=visible)
    assert resolver_evaluate(nominated, target, visible=visible) == record


def test_route_evaluation_uses_causal_visibility_and_weights():
    t, s = 4, 4
    native = np.zeros((2, t, s))
    native[:, np.arange(t), 0] = 0.6  # sink at position 0
    native[:, np.arange(t), np.arange(t)] += 0.4
    native[:, 0, 0] = 1.0
    predicted = np.tril(np.ones((2, t, s)))
    predicted[:, np.arange(t), np.arange(t)] += 1.0  # predicts self
    predicted /= predicted.sum(-1, keepdims=True)
    record = evaluate_route_forecast(predicted, native)
    assert record["rows"] == 8
    assert record["baselines"]["position_0"]["top_source_agreement"] == 1.0
    assert record["nominator"]["top_source_agreement"] == pytest.approx(0.25)
    assert record["baselines"]["position_0"]["native_weight_at_choice"] == pytest.approx((1 + 3 * 0.6) / 4)
    expected = {k: dict(v) for k, v in record["baselines"].items()}
    expected["previous_token"].pop("nominator_top_source_agreement_same_rows")  # nominator-specific
    assert trivial_route_baselines(native) == expected
    with pytest.raises(ValueError):
        evaluate_route_forecast(predicted[:, :, :3], native)


def test_resolver_nomination_scored_against_baselines_on_rows():
    rows = [([.7, .1, .1, .1], 3), ([.1, .1, .1, .6, .1], 4), ([.9, .05, .05], 2)]
    nominated = [nominate(w, cached)["predicted_row"] for w, cached in rows]
    native = [int(np.argmax(w)) for w, _ in rows]
    record = evaluate_top_source(nominated, native, visible=[len(w) for w, _ in rows])
    assert record["nominator"]["top_source_agreement"] == 1.0
    assert record["baselines"]["position_0"]["top_source_agreement"] == pytest.approx(2 / 3)


def test_previous_token_baseline_excludes_query_position_zero():
    # Row 0 is query 0 (visible 1): its "previous token" clamps to position 0, which would score
    # correct whenever the native top source is 0. It must be left out and counted.
    record = evaluate_top_source([0, 0, 0], [0, 0, 1], visible=[1, 3, 3])
    prev = record["baselines"]["previous_token"]
    assert prev["rows"] == 2 and prev["excluded_rows"] == 1
    assert prev["top_source_agreement"] == pytest.approx(0.5)  # queries 2 -> source 1: right on row 2 only
    assert prev["nominator_top_source_agreement_same_rows"] == pytest.approx(0.5)
    assert record["baselines"]["position_0"]["rows"] == 3
    only_zero = evaluate_top_source([0], [0], visible=[1])
    assert only_zero["baselines"]["previous_token"]["rows"] == 0
    assert only_zero["baselines"]["previous_token"]["top_source_agreement"] is None
    assert only_zero["best_trivial"]["name"] != "previous_token"


def test_query_positions_are_validated_against_visible():
    with pytest.raises(ValueError, match="query_positions"):
        evaluate_top_source([0, 0], [0, 0], visible=[2, 2], query_positions=[1, 3])
    with pytest.raises(ValueError, match="query_positions"):
        evaluate_top_source([0, 0], [0, 0], visible=[2, 2], query_positions=[-1, 1])
    ok = evaluate_top_source([0, 0], [0, 0], visible=[2, 2], query_positions=[1, 2])
    assert ok["baselines"]["previous_token"]["rows"] == 2


def test_empty_offsets_still_report_the_current_token_baseline():
    record = evaluate_top_source([1, 2], [1, 2], visible=[2, 3], offsets_from_end=())
    assert record["baselines"]["offset_0_from_end"]["top_source_agreement"] == 1.0
    with pytest.raises(ValueError):
        evaluate_top_source([1, 2], [1, 2], visible=[2, 3], offsets_from_end=(-1,))


def test_control_set_failure_names_the_failed_constraint():
    strata = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match=r"stratum 1: needs 3 units, only 0 allowed"):
        matched_control_sets([3, 4, 5], 6, strata=strata, disjoint=True, max_tries=3)
    blocks = np.array([0, 1, 2, 2, 2, 2])
    with pytest.raises(ValueError, match="needs 3 distinct blocks, only 1 allowed"):
        matched_control_sets([0, 1, 2], 6, blocks=blocks, disjoint=True, max_tries=3)

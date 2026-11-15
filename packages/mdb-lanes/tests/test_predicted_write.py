"""CPU tests for mdb_lanes.predicted_write (tiny numpy arrays; no model, no torch)."""
import itertools

import numpy as np
import pytest

from mdb_lanes import predicted_write as pw


def _case(seed=0, h=3, r=5, sk=9, d=4):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(h, r, sk)) * 2.0
    v = rng.normal(size=(h, sk, d))
    m = np.zeros(sk)
    m[[2, 5]] = 1.0
    return scores, v, m


def test_zero_cut_identity_matches_direct_recomputation():
    scores, v, m = _case()
    p = pw.softmax_rows(scores)
    a = p @ v
    direct = (p * (1 - m)) @ v            # contribution removed, other weights unchanged
    assert np.allclose(pw.zero_cut_delta(p, v, m), direct - a, atol=1e-13)


def test_mask_cut_identity_matches_direct_recomputation():
    scores, v, m = _case(1)
    p = pw.softmax_rows(scores)
    a = p @ v
    masked = np.where(m > 0, -np.inf, scores)
    direct = pw.softmax_rows(masked) @ v   # boolean mask, softmax over the remaining sources
    assert np.allclose(pw.mask_cut_delta(p, v, m), direct - a, atol=1e-12)
    assert np.allclose(pw.mask_cut_delta(p, v, m, a), direct - a, atol=1e-12)


def test_mask_identity_is_mass_times_value_contrast_for_one_source():
    scores, v, _ = _case(2)
    p = pw.softmax_rows(scores)
    a = p @ v
    m = np.zeros(scores.shape[-1])
    m[4] = 1.0
    pj = p[..., 4][..., None]
    expect = pj / (1 - pj) * (a - v[:, None, 4, :])
    assert np.allclose(pw.mask_cut_delta(p, v, m), expect, atol=1e-12)


def test_write_through_head_maps_matches_full_projection():
    rng = np.random.default_rng(3)
    h, r, d, c = 3, 4, 2, 5
    delta = rng.normal(size=(h, r, d))
    weight = rng.normal(size=(c, h * d))
    gate = rng.normal(size=c)
    maps = pw.head_maps(weight, h)
    per_head = pw.head_writes(delta, maps, gate)
    flat = delta.transpose(1, 0, 2).reshape(r, h * d)          # heads concatenated per row
    full = (flat @ weight.T) * gate
    assert np.allclose(per_head.sum(0), full, atol=1e-12)
    assert np.allclose(pw.block_write(delta, maps, gate), full, atol=1e-12)
    assert np.allclose(pw.write_norms(delta, maps, gate), np.linalg.norm(per_head, axis=-1), atol=1e-12)


def test_split3_closes_and_routing_ignores_common_values():
    rng = np.random.default_rng(4)
    h, r, sk, d = 2, 3, 7, 4
    pn = pw.softmax_rows(rng.normal(size=(h, r, sk)))
    pd = pw.softmax_rows(rng.normal(size=(h, r, sk)))
    vn, vd = rng.normal(size=(h, sk, d)), rng.normal(size=(h, sk, d))
    parts = pw.split3(pn, vn, pd, vd)
    assert np.allclose(parts["routing"] + parts["payload"] + parts["interaction"], parts["total"], atol=1e-13)
    vbar = rng.normal(size=(h, d))
    assert np.allclose(pw.routing_centered(pn, pd, vn, vbar), parts["routing"], atol=1e-13)


def test_payload_groups_and_common_part_close():
    rng = np.random.default_rng(5)
    h, r, sk, d = 2, 3, 8, 3
    pn = pw.softmax_rows(rng.normal(size=(h, r, sk)))
    dv = rng.normal(size=(h, sk, d))
    word, real, pad, img = (np.zeros(sk) for _ in range(4))
    word[1] = 1
    real[[0, 2]] = 1
    pad[[3, 4]] = 1
    img[[5, 6, 7]] = 1
    parts = pw.payload_parts(pn, dv, {"word": word, "real": real, "pad": pad, "img": img})
    assert np.allclose(sum(parts.values()), pn @ dv, atol=1e-13)
    text = word + real + pad
    common, specific = pw.common_payload(pn, dv, text)
    assert np.allclose(common + specific, (pn * text) @ dv, atol=1e-13)
    # a payload change shared identically by every text source is all common
    same = np.repeat(rng.normal(size=(h, 1, d)), sk, axis=1)
    c2, s2 = pw.common_payload(pn, same, text)
    assert np.allclose(s2, 0, atol=1e-13)


def test_shares_sum_to_one_for_an_exact_decomposition():
    rng = np.random.default_rng(6)
    x, y, z = rng.normal(size=(3, 4, 5))
    t = x + y + z
    raw = {k: {"ip": pw.inner(v, t), "nn": pw.inner(v, v)} for k, v in (("x", x), ("y", y), ("z", z), ("total", t))}
    s = pw.shares(raw)
    assert abs(s["x"]["projection_share"] + s["y"]["projection_share"] + s["z"]["projection_share"] - 1) < 1e-12


def test_deletion_alignment_is_a_bijection_with_the_hybrid_shift():
    t = 12
    rec_o = {"n_real": 8, "real_rows": list(range(8)), "input_ids": [1, 2, 3, 9, 4, 5, 6, 7],
             "words": [{"word": "red", "rows": [3]}, {"word": "ball", "rows": [4]}]}
    rec_d = {"n_real": 7, "real_rows": list(range(7)), "input_ids": [1, 2, 3, 4, 5, 6, 7], "words": []}
    al = pw.deletion_alignment(rec_o, rec_d, "red", t)
    assert sorted(al["pi"]) == list(range(t))
    assert al["pi"][:3] == [0, 1, 2]                    # before the word: same row
    assert al["pi"][4] == 3 and al["pi"][7] == 6        # later real rows shift by one
    assert al["pi"][8] == 7 and al["pi"][11] == 10      # padding shifts by one
    assert al["word"] == [3] and al["pi"][3] == 11      # the word maps to the freed trailing pad row
    assert al["real"] == [0, 1, 2, 4, 5, 6, 7] and al["pad"] == list(range(8, 12))


def test_top_units_breaks_ties_by_index():
    assert pw.top_units([1, 3, 3, 2, 3], 2) == [1, 2]


def _brute(c, o, k, target, tol, exclude=()):
    best = None
    for s in itertools.combinations([i for i in range(len(c)) if i not in exclude], k):
        s = list(s)
        if abs(c[s].sum() - target) <= tol * target and (best is None or o[s].sum() < o[best].sum() - 1e-15):
            best = s
    return best


@pytest.mark.parametrize("seed", range(6))
def test_matched_min_stays_in_band_and_finds_the_small_case_optimum(seed):
    rng = np.random.default_rng(seed)
    n, k = 11, 4
    c = rng.random(n)
    o = c * rng.uniform(0.2, 2.0, n)                     # correlated, as mass and predicted write are
    target = float(np.sort(c)[::-1][:k].sum()) * 0.8
    units, rep = pw.matched_min(c, o, k, target, 0.10)
    assert len(units) == k and abs(c[units].sum() - target) <= 0.10 * target
    best = _brute(c, o, k, target, 0.10)
    assert o[units].sum() <= o[best].sum() * 1.05 + 1e-12      # 1-swap local optimum near the optimum here
    assert rep["local_optimum"]


def test_matched_min_respects_exclusions_and_reports_unreachable():
    c = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
    o = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    units, _ = pw.matched_min(c, o, 2, 6.0, 0.10, exclude=[3])
    assert 3 not in units and units == [0, 4]
    with pytest.raises(ValueError):
        pw.matched_min(c, o, 2, 100.0, 0.10)


def test_spearman_matches_rank_definition():
    assert pw.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert pw.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pw.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254, abs=1e-6)


# ------------------------------------------------------------- promoted from the experiment


def test_receiver_write_scores_are_the_gated_cut_write_norms():
    scores, v, m = _case(seed=3, h=3, r=6, sk=9, d=4)
    p = pw.softmax_rows(scores)
    rng = np.random.default_rng(4)
    maps_i, maps_t = rng.normal(size=(3, 4, 7)), rng.normal(size=(3, 4, 7))
    g_i, g_t = rng.normal(size=7), np.ones(7)
    img, oth = np.array([3, 4, 5]), np.array([0, 2])
    got = pw.receiver_write_scores(p, v, m, {"img": (img, maps_i, g_i), "oth": (oth, maps_t, g_t)})
    assert set(got) == {"zero_img", "zero_oth", "mask_img", "mask_oth"}
    a = p @ v
    for kind, delta in (("zero", (p * (1 - m)) @ v - a),
                        ("mask", (p * (1 - m) / (p * (1 - m)).sum(-1, keepdims=True)) @ v - a)):
        for name, idx, maps, gate in (("img", img, maps_i, g_i), ("oth", oth, maps_t, g_t)):
            write = np.einsum("hrd,hdc->hrc", delta[:, idx], maps) * gate
            assert np.allclose(got[f"{kind}_{name}"], np.linalg.norm(write, axis=-1).sum(-1), atol=1e-12)


def test_pw_shares_parts_and_normalisation():
    rng = np.random.default_rng(5)
    zi, zo, mi, mo = (rng.random((4, 3, 5)) for _ in range(4))
    s = pw.pw_shares(zi, zo, mi, mo)
    assert s["pw_both"].shape == (15,)
    assert s["pw_both"].sum() == pytest.approx(2.0) and s["pw_img_only"].sum() == pytest.approx(2.0)
    assert np.allclose(s["pw_img"] + s["pw_txt"], s["pw_both"], atol=1e-15)
    flat = zi.sum(0).ravel()
    assert s["pw_img_only"][7] == pytest.approx(flat[7] / flat.sum() + mi.sum(0).ravel()[7] / mi.sum(0).ravel().sum())


def test_clean_random_draws_exclude_named_previous_and_each_other():
    n = 60
    strata = np.array([0] * 15 + [1] * 45)
    blocks = np.arange(n) // 6
    reference = [0, 7, 20, 21, 33, 40]
    named = [[1, 2, 3], [50, 51]]
    previous = [[10, 11, 12, 13, 14, 15]]
    draws, reports = pw.clean_random_draws(reference, n, named=named, previous=previous, seed=7, item="red",
                                           names=("rand_c", "rand_d"), strata=strata, blocks=blocks)
    again, _ = pw.clean_random_draws(reference, n, named=named, previous=previous, seed=7, item="red",
                                     names=("rand_c", "rand_d"), strata=strata, blocks=blocks)
    assert draws == again                                    # deterministic
    banned = set(reference) | {1, 2, 3, 50, 51} | set(previous[0])
    c, d = set(draws["rand_c"]), set(draws["rand_d"])
    assert not (c & banned) and not (d & banned) and not (c & d)
    for units in (c, d):
        assert len(units) == len(reference)
        assert sorted(strata[list(units)]) == sorted(strata[reference])
    other, _ = pw.clean_random_draws(reference, n, named=named, previous=previous, seed=7, item="ball",
                                     names=("rand_c",), strata=strata, blocks=blocks)
    assert set(reports) == {"rand_c", "rand_d"} and isinstance(other["rand_c"], list)

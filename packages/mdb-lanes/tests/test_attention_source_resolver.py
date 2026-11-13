import pytest

from mdb_lanes.attention_source_resolver import arms, nominate


def test_numeric_nomination_tracks_moving_rows_and_keeps_first_tie():
    for source in (1, 5, 13, 23):
        weights = [.01] * (source + 3)
        weights[source] = .9
        found = nominate(weights, len(weights) - 1)
        assert found["source_row"] == source
        assert found["status"] == "addressable-cached-source"
    assert nominate([.5, .5, 0.], 2)["source_row"] == 0


def test_does_not_force_an_uncached_nomination_into_the_prefix():
    found = nominate([.1, .2, .7], 2)
    assert found["predicted_row"] == 2
    assert found["source_row"] is None
    assert found["status"] == "outside-retained-prefix"
    for weights in ([], [0., 0.], [.3, float("nan")], [-.1, 1.1], [True, .5]):
        with pytest.raises(ValueError):
            nominate(weights, 0)


def test_controls_keep_the_frozen_cell_pair_and_broader_comparator():
    plan = {a["arm"]: a for a in arms()}
    assert len(plan) == 9
    assert len(plan["auto-full-late"]["cells"]) == 18
    assert all(len(row["cells"]) == 2 for name, row in plan.items() if name != "auto-full-late")

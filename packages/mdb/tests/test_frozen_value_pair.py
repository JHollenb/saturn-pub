from mdb.frozen_value_pair import PAIR, arms, cases


def test_contexts_change_only_the_declared_source_within_each_pair():
    rows = cases()
    assert len(rows) == 5
    assert len({r["recipient"]["text"] for r in rows}) == 5
    for row in rows:
        normalized = []
        for side, word in (("recipient", "freezing"), ("donor", "boiling")):
            text = row[side]["text"]
            a, b = row[side]["source_span"]
            assert text[a:b] == word
            normalized.append(text[:a] + "ARGUMENT" + text[b:])
        assert normalized[0] == normalized[1]


def test_routes_are_frozen_with_matched_controls_and_broader_comparator():
    rows = {row["arm"]: row for row in arms()}
    assert len(rows) == 10
    assert len(rows["full-late"]["cells"]) == 18
    for name in ("pair", "sham-pair", "reciprocal-pair", "wrong-payload-pair", "wrong-target-pair"):
        assert rows[name]["cells"] == PAIR
    assert rows["only-KV0"]["cells"] + rows["only-KV1"]["cells"] == PAIR

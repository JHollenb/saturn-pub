"""Resolver adversaries and semantic/provenance boundaries, without a model."""
import json

import pytest

from mdb.live_scaffold import invocation_graph, render_html, resolve_addition, scaffold


def resolve(text, step=0):
    return resolve_addition(text, list(range(len(text))), [(i, i + 1) for i in range(len(text))], step)


def test_column_roles_advance_and_survive_prefix_and_width_changes():
    a = resolve("Examples\n347+125=", 0)["bindings"]
    b = resolve("Compute: 3471+1256=", 1)["bindings"]
    assert [x["column_from_right"] for x in a] == [2, 1, 0]
    assert [x["column_from_right"] for x in b] == [2, 1, 0]
    assert [x["heads"] for x in a] == [x["heads"] for x in b]
    assert a[0]["arguments"][1]["source_row"] - a[0]["arguments"][0]["source_row"] == 4
    assert b[0]["arguments"][1]["source_row"] - b[0]["arguments"][0]["source_row"] == 5


def test_last_expression_is_resolved_without_reading_answer():
    first = resolve("12+35=47; 347+125=4", 1)
    second = resolve("12+35=47; 347+125=9", 1)
    assert first["bindings"] == second["bindings"]
    assert first["input_identity"] != second["input_identity"]


def test_missing_lower_column_is_explicit_not_a_fabricated_address():
    result = resolve("347+125=47", 2)
    assert [b["status"] for b in result["bindings"]] == ["resolved", "not-applicable", "not-applicable"]


def test_merged_digit_tokens_refuse_instead_of_guessing():
    with pytest.raises(ValueError, match="one exact token"):
        resolve_addition("12+34=", [1, 2, 3, 4], [(0, 2), (2, 3), (3, 5), (5, 6)], 0)


@pytest.mark.parametrize("text", ["347 minus 125", "3+125=", "no operands"])
def test_unsupported_grammar_stays_unsupported(text):
    with pytest.raises(ValueError):
        resolve(text)


def test_schema_is_stable_but_occurrences_keep_distinct_history_and_output(tmp_path):
    schema = scaffold("checkpoint")
    occurrences = []
    for i, history in enumerate(("history-a", "history-b")):
        occurrences.append({"case": "case", "answer_step": i, "history_sha256": history,
                            "bindings": resolve("347+125=", i)["bindings"],
                            "next_token_id": i, "next_token_text": "</script>" if i == 0 else "7",
                            "logits_sha256": f"logits-{i}"})
    graph = invocation_graph(schema, occurrences)
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids))
    assert all(e["source"] in ids and e["target"] in ids for e in graph["edges"])
    assert {e["kind"] for e in graph["edges"]} == {"observed-return", "proposed-semantic-binding", "native-suffix-dataflow"}
    target = tmp_path / "view.html"
    render_html(graph, target)
    text = target.read_text()
    assert "&lt;/script&gt;" in text
    embedded = text.split('id="graph">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["fingerprint"] == graph["fingerprint"]

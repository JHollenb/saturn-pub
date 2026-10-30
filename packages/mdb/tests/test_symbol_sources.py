import copy

import pytest

from mdb.symbol_sources import (
    SymbolSourceError,
    make_vocabulary,
    nominate_history,
    nominate_prompt,
    nominate_scores,
    verify_candidate,
)


@pytest.fixture
def vocabulary():
    return make_vocabulary(["MOV", "ADD", "SUB"], namespace="isa.operation",
                           evidence_refs=["checkpoint://frozen-label-order"])


def source():
    return dict(model_identity="model", execution="execution", source_cut="cut",
                context="context", evidence_refs=["cas://cut"])


def test_prompt_reads_observed_tokens_and_preserves_ambiguity(vocabulary):
    result = nominate_prompt(["MOV", "r0", "7"], vocabulary, **source())
    assert result["selected_symbol"] == "isa.operation.MOV"
    assert verify_candidate(result)["passed"]
    both = nominate_prompt(["MOV", "ADD"], vocabulary, **source())
    assert both["selected_symbol"] is None
    assert both["abstain_reason"] == "multiple-literal-symbols"
    assert nominate_prompt(["unknown"], vocabulary, **source())["selected_symbol"] is None


def test_scores_use_actual_frozen_order_and_preserve_all_logits(vocabulary):
    result = nominate_scores([1.0, 4.0, 2.0], vocabulary, producer_id="native-head", **source())
    assert result["selected_label"] == "ADD"
    assert result["proposals"][0]["index"] == 1
    assert result["source"]["logits"] == [1.0, 4.0, 2.0]
    assert result["semantic_qualified"] is False
    assert result["causal_dependency_tested"] is False
    assert verify_candidate(result)["passed"]
    tied = nominate_scores([1.0, 1.0, 0.0], vocabulary, producer_id="native-head", **source())
    assert tied["selected_label"] is None


@pytest.mark.parametrize("scores", [[0.0], [0.0, float("nan"), 1.0], [0.0, float("inf"), 1.0]])
def test_invalid_native_scores_refuse_as_instrument_errors(vocabulary, scores):
    with pytest.raises(SymbolSourceError, match="finite scores"):
        nominate_scores(scores, vocabulary, producer_id="native-head", **source())


def test_history_is_own_output_and_ordered_not_teacher_state(vocabulary):
    events = [dict(index=i, symbol=label, source_cut="cut", receipt="receipt",
                   authority="model-predicted", commit_status="committed")
              for i, label in enumerate(["MOV", "ADD"])]
    result = nominate_history(events, vocabulary, **source())
    assert result["selected_label"] == "ADD"
    assert [row["label"] for row in result["proposals"]] == ["MOV", "ADD"]
    assert result["source"]["textual_cot"] is False
    assert verify_candidate(result)["passed"]
    events[0]["authority"] = "teacher"
    with pytest.raises(SymbolSourceError, match="model-predicted"):
        nominate_history(events, vocabulary, **source())


def test_source_readers_have_no_evaluator_argument_and_reject_tampering(vocabulary):
    with pytest.raises(TypeError, match="expected_answer"):
        nominate_prompt(["MOV"], vocabulary, expected_answer="ADD", **source())
    result = nominate_scores([1, 0, -1], vocabulary, producer_id="native-head", **source())
    altered = copy.deepcopy(result)
    altered["selected_label"] = "SUB"
    with pytest.raises(SymbolSourceError, match="fingerprint"):
        verify_candidate(altered)


def test_resealed_selection_and_nested_evaluator_metadata_are_rejected(vocabulary):
    from saturn_pub.values import digest

    result = nominate_scores([1, 0, -1], vocabulary, producer_id="native-head", **source())
    altered = {key: value for key, value in result.items() if key != "fingerprint"}
    altered["selected_label"] = "SUB"
    altered["selected_symbol"] = "isa.operation.SUB"
    altered["fingerprint"] = digest(altered)
    with pytest.raises(SymbolSourceError, match="recomputation"):
        verify_candidate(altered)
    with pytest.raises(SymbolSourceError, match="evaluator fields"):
        nominate_prompt(["MOV"], vocabulary,
                        source_metadata={"nested": {"expected_answer": "ADD"}}, **source())


def test_rejected_proposal_does_not_become_committed_history(vocabulary):
    events = [dict(index=0, symbol="MOV", source_cut="cut", receipt="receipt",
                   authority="model-predicted", commit_status="committed"),
              dict(index=1, symbol="ADD", source_cut="cut", receipt="receipt",
                   authority="model-predicted", commit_status="rejected")]
    result = nominate_history(events, vocabulary, **source())
    assert result["selected_label"] == "MOV"
    assert result["proposals"][1]["commit_status"] == "rejected"
    assert verify_candidate(result)["passed"]

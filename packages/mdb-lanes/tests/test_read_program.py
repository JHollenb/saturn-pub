from __future__ import annotations

from copy import deepcopy

import pytest

from mdb_lanes.read_program import (
    FrozenReadProgram,
    ReadProgramError,
    make_two_read_program,
)


def test_standard_program_declares_frozen_roles_and_round_trips_digest():
    program = make_two_read_program()

    assert program.ports["A"].layer == 19
    assert program.ports["A"].kv_head == 0
    assert program.ports["A"].row_role == "preceding-answer-first-generated"
    assert program.ports["A"].provenance == "supplied"
    assert program.ports["B"].layer == 22
    assert program.ports["B"].kv_head == 0
    assert program.ports["B"].row_role == "current-followup-copula"
    assert program.ports["B"].provenance == "supplied"
    assert program.prediction_assistance == "native-A-receiver"

    payload = program.to_dict()
    restored = FrozenReadProgram.from_dict(payload)
    assert restored == program
    assert restored.to_dict() == payload


def test_resolution_uses_fresh_spans_when_history_length_changes():
    program = FrozenReadProgram.standard()

    first_history = [10, 11, 20, 21, 30, 31, 32]
    first = program.resolve(
        first_history,
        preceding_answer_span=(2, 4),
        followup_span=(4, 7),
        copula_token_id=31,
    )
    assert first[0]["port"] == "A"
    assert first[0]["row"] == 2
    assert first[0]["source_token_id"] == 20
    assert first[1]["port"] == "B"
    assert first[1]["row"] == 5

    # The extra prompt token shifts both spans.  No fixed global row is reused.
    second_history = [99] + first_history
    second = program.resolve(
        second_history,
        preceding_answer_span=(3, 5),
        followup_span=(5, 8),
        copula_token_id=31,
    )
    assert second[0]["row"] == 3
    assert second[0]["source_token_id"] == 20
    assert second[1]["row"] == 6
    assert second[1]["source_token_id"] == 31


def test_copula_lookup_is_last_occurrence_inside_current_followup_only():
    program = FrozenReadProgram.standard()
    history = [5, 50, 7, 9, 7, 8, 7, 99]

    result = program.resolve(
        history,
        preceding_answer_span=(1, 2),
        followup_span=(2, 6),
        copula_token_id=7,
    )

    assert result[1]["row"] == 4
    assert result[1]["source_token_id"] == 7
    assert result[1]["lineage"]["span"] == [2, 6]
    assert result[1]["lineage"]["occurrence"] == "last-copula-in-span"


def test_first_generated_row_tracks_answer_start_on_each_invocation():
    program = FrozenReadProgram.standard()
    history = [1, 2, 40, 41, 8, 9]

    initial = program.resolve(
        history,
        preceding_answer_span=(2, 4),
        followup_span=(4, 6),
        copula_token_id=8,
    )
    changed = program.resolve(
        [1, 2, 55, 56, 8, 9],
        preceding_answer_span=(2, 4),
        followup_span=(4, 6),
        copula_token_id=8,
    )

    assert initial[0]["row"] == changed[0]["row"] == 2
    assert initial[0]["source_token_id"] == 40
    assert changed[0]["source_token_id"] == 55
    assert initial[1]["row"] == changed[1]["row"] == 4


def test_missing_copula_abstains_without_inventing_a_global_row():
    program = FrozenReadProgram.standard()

    result = program.resolve(
        [1, 2, 3, 4, 99, 5],
        preceding_answer_span=(1, 2),
        followup_span=(2, 4),
        copula_token_id=99,
    )

    assert result[0]["row"] == 1
    assert result[1] is None


def test_digest_tampering_is_rejected():
    payload = FrozenReadProgram.standard().to_dict()
    tampered = deepcopy(payload)
    tampered["ports"]["A"]["layer"] = 20

    with pytest.raises(ReadProgramError, match="fingerprint"):
        FrozenReadProgram.from_dict(tampered)

    tampered = deepcopy(payload)
    tampered["fingerprint"] = "0" * 64
    with pytest.raises(ReadProgramError, match="fingerprint"):
        FrozenReadProgram.from_dict(tampered)


def test_unknown_role_and_prediction_assistance_fail_closed():
    payload = FrozenReadProgram.standard().to_dict()
    payload["ports"]["B"]["row_role"] = "unknown"
    with pytest.raises(ReadProgramError, match="fingerprint"):
        FrozenReadProgram.from_dict(payload)

    with pytest.raises(ReadProgramError, match="prediction_assistance"):
        FrozenReadProgram(prediction_assistance="arbitrary-callback")


@pytest.mark.parametrize(
    ("history", "answer", "followup", "copula"),
    [
        ([1, 2, 3], (1, 1), (1, 2), 2),  # empty answer span
        ([1, 2, 3], (1, 2), (2, 4), 2),  # follow-up out of history
        ([1, 2, 3], (2, 3), (1, 2), 2),  # reversed span order
        ([1, 2, 3], (1, 3), (2, 3), 2),  # overlapping spans
        ([1, 2, 3], (True, 2), (2, 3), 2),  # bool is not a token index
        ([1, "2", 3], (0, 1), (1, 2), 2),  # non-integer token id
    ],
)
def test_invalid_spans_and_ids_fail(history, answer, followup, copula):
    with pytest.raises(ReadProgramError):
        FrozenReadProgram.standard().resolve(
            history,
            preceding_answer_span=answer,
            followup_span=followup,
            copula_token_id=copula,
        )


def test_declared_layer_and_head_bounds_are_checked_without_native_imports():
    payload = FrozenReadProgram.standard().to_dict()
    payload["ports"]["A"]["layer"] = -1
    with pytest.raises(ReadProgramError, match="layer"):
        FrozenReadProgram(ports=payload["ports"])

    payload["ports"]["A"]["layer"] = 19
    payload["ports"]["B"]["kv_head"] = 256
    with pytest.raises(ReadProgramError, match="kv_head"):
        FrozenReadProgram(ports=payload["ports"])

"""Tests for the runtime-agnostic bisect verbs and their hash-chained ledger."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from saturn_pub.evidence._canonical import GENESIS_SHA256
from saturn_pub.evidence.bisect import (
    BISECT_SCHEMA,
    CutHandle,
    first_divergence,
    first_divergence_pairs,
    first_true,
    main,
    make_cuts,
    probe_economics,
    splice_bisect,
    verify_ledger,
)

SIZES = [1, 2, 3, 4, 5, 7, 8, 13, 16, 31, 33, 64]


def _canonical_sha256(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CountingPredicate:
    """Boolean lattice predicate that counts callback invocations."""

    def __init__(self, lattice):
        self.lattice = list(lattice)
        self.calls = 0

    def __call__(self, cut: CutHandle):
        self.calls += 1
        value = bool(self.lattice[cut.index])
        return value, {"lattice_value": value}


def step_lattice(size: int, boundary: int) -> list[bool]:
    return [position >= boundary for position in range(size)]


@pytest.mark.parametrize("size", SIZES)
def test_first_true_finds_every_boundary(size):
    for boundary in range(size + 1):
        predicate = CountingPredicate(step_lattice(size, boundary))
        result = first_true(make_cuts(size), predicate)
        if boundary == size:
            assert result.first_true_index is None
            assert result.first_true_cut_id is None
        else:
            assert result.first_true_index == boundary
            assert result.first_true_cut_id == f"cut/{boundary}"
        assert not result.monotonicity_violation
        assert result.lattice_size == size
        assert result.probe_count == len(result.probes) == predicate.calls
        assert len({probe.index for probe in result.probes}) == result.probe_count


@pytest.mark.parametrize("size", SIZES)
def test_first_true_probe_budget(size):
    for boundary in range(size + 1):
        predicate = CountingPredicate(step_lattice(size, boundary))
        result = first_true(make_cuts(size), predicate)
        if size == 1:
            assert result.probe_count == 1
        else:
            assert result.probe_count <= 2 + math.ceil(math.log2(size))


@pytest.mark.parametrize("size", SIZES)
def test_first_true_without_endpoint_verification(size):
    for boundary in range(size + 1):
        predicate = CountingPredicate(step_lattice(size, boundary))
        result = first_true(make_cuts(size), predicate, verify_endpoints=False)
        expected = None if boundary == size else boundary
        assert result.first_true_index == expected
        assert not result.monotonicity_violation
        assert result.probe_count <= math.ceil(math.log2(max(size, 2))) + 1


def test_first_true_all_false_short_circuits():
    result = first_true(make_cuts(64), CountingPredicate([False] * 64))
    assert result.first_true_index is None
    assert result.probe_count == 2


def test_first_true_all_true_short_circuits():
    result = first_true(make_cuts(64), CountingPredicate([True] * 64))
    assert result.first_true_index == 0
    assert result.probe_count == 2


def test_first_true_size_one():
    assert first_true(make_cuts(1), CountingPredicate([True])).first_true_index == 0
    assert first_true(make_cuts(1), CountingPredicate([False])).first_true_index is None


def test_first_true_empty_lattice():
    result = first_true([], CountingPredicate([]))
    assert result.first_true_index is None
    assert result.probes == []
    assert result.probe_count == 0
    assert result.lattice_size == 0
    assert result.ledger_sha256 is None
    assert not result.monotonicity_violation


def test_first_true_probe_order_endpoints_then_mids():
    result = first_true(make_cuts(4), CountingPredicate([False, False, True, True]))
    assert result.first_true_index == 2
    assert [probe.index for probe in result.probes] == [0, 3, 2, 1]


def test_first_true_respects_caller_coordinates():
    cuts = [CutHandle(cut_id=f"step/{index}", index=index) for index in (10, 20, 30, 40)]
    lattice = {10: False, 20: False, 30: True, 40: True}
    result = first_true(cuts, lambda cut: (lattice[cut.index], {"step": cut.index}))
    assert result.first_true_index == 30
    assert result.first_true_cut_id == "step/30"


def test_first_true_rejects_unordered_cuts():
    cuts = [CutHandle(cut_id="a", index=1), CutHandle(cut_id="b", index=1)]
    with pytest.raises(ValueError, match="strictly increasing"):
        first_true(cuts, lambda cut: (True, {}))


def test_non_monotone_lattice_sets_violation_flag():
    result = first_true(make_cuts(4), CountingPredicate([True, False, False, False]))
    assert result.monotonicity_violation
    assert result.first_true_index is None
    assert result.probe_count == 2


def test_non_monotone_lattice_larger():
    predicate = CountingPredicate([True, True, True, False, True, False, True, False])
    result = first_true(make_cuts(8), predicate)
    assert result.monotonicity_violation
    assert result.first_true_index is None


def test_consistent_probes_do_not_flag_violation():
    result = first_true(make_cuts(4), CountingPredicate([False, True, False, True]))
    assert not result.monotonicity_violation
    assert result.first_true_index == 3


def _digest_fns(size: int, diverge_at: int):
    arm_a = [f"state-{position}" for position in range(size)]
    arm_b = [
        arm_a[position] if position < diverge_at else f"forked-{position}"
        for position in range(size)
    ]
    return (lambda cut: arm_a[cut.index]), (lambda cut: arm_b[cut.index])


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 17, 33])
def test_first_divergence_finds_every_fork_point(size):
    for diverge_at in range(size + 1):
        digest_a, digest_b = _digest_fns(size, diverge_at)
        result = first_divergence(make_cuts(size), digest_a, digest_b)
        if diverge_at == size:
            assert result.first_true_index is None
        else:
            assert result.first_true_index == diverge_at
        assert not result.monotonicity_violation
        assert result.probe_count <= 2 + math.ceil(math.log2(max(size, 2)))


def test_first_divergence_probe_records_carry_digests_not_verdicts():
    digest_a, digest_b = _digest_fns(8, 5)
    result = first_divergence(make_cuts(8), digest_a, digest_b)
    for probe in result.probes:
        assert probe.verdict is None
        assert probe.digest_a is not None and probe.digest_b is not None
    boundary = next(probe for probe in result.probes if probe.index == 5)
    assert boundary.digest_a != boundary.digest_b


def test_first_divergence_healed_fork_flags_violation():
    arm_a = ["x0", "same-1", "same-2", "same-3"]
    arm_b = ["y0", "same-1", "same-2", "same-3"]
    result = first_divergence(
        make_cuts(4), lambda cut: arm_a[cut.index], lambda cut: arm_b[cut.index]
    )
    assert result.first_true_index is None
    assert result.monotonicity_violation


def test_first_divergence_pairs_probes_each_cut_once():
    """The saturn_pub integration form calls the pair callback at most per cut."""

    calls: list[int] = []

    def digests(cut: CutHandle) -> tuple[str, str]:
        calls.append(cut.index)
        left = f"s{cut.index}"
        right = left if cut.index < 6 else f"f{cut.index}"
        return left, right

    result = first_divergence_pairs(make_cuts(16), digests)
    assert result.first_true_index == 6
    assert len(calls) == len(set(calls)) == result.probe_count
    assert result.probe_count < result.lattice_size
    assert verify_ledger(result.probes)


def test_probe_economics_reports_savings_vs_linear():
    result = first_true(make_cuts(64), CountingPredicate(step_lattice(64, 40)))
    econ = probe_economics(result)
    assert econ["linear_probe_count"] == 64
    assert econ["probe_count"] == result.probe_count
    assert econ["saved_probes"] == 64 - result.probe_count
    assert econ["speedup"] == 64 / result.probe_count
    assert result.to_json_dict()["linear_probe_count"] == 64


def test_splice_bisect_finds_threshold_crossing():
    scores = [0.0, 0.1, 0.2, 0.6, 0.8, 0.9]
    result = splice_bisect(
        make_cuts(len(scores)),
        lambda cut: (scores[cut.index], {"trial": cut.cut_id}),
        0.5,
    )
    assert result.first_true_index == 3
    assert not result.monotonicity_violation


def test_splice_bisect_threshold_edge_equality_counts_as_crossed():
    scores = [0.0, 0.2, 0.5, 0.5, 0.9]
    result = splice_bisect(make_cuts(len(scores)), lambda cut: (scores[cut.index], {}), 0.5)
    assert result.first_true_index == 2


def test_splice_bisect_smaller_transfers_direction():
    scores = [0.9, 0.7, 0.5, 0.1]
    result = splice_bisect(
        make_cuts(len(scores)),
        lambda cut: (scores[cut.index], {}),
        0.5,
        larger_transfers=False,
    )
    assert result.first_true_index == 2


def test_splice_bisect_never_crossing():
    scores = [0.0, 0.1, 0.2]
    result = splice_bisect(make_cuts(len(scores)), lambda cut: (scores[cut.index], {}), 0.5)
    assert result.first_true_index is None
    assert result.probe_count == 2


def test_non_finite_values_sanitized_before_hashing():
    scores = [0.0, float("nan"), 0.9]
    result = splice_bisect(make_cuts(3), lambda cut: (scores[cut.index], {}), 0.5)
    assert result.first_true_index == 2
    assert verify_ledger(result.probes)
    nan_probe = next(probe for probe in result.probes if probe.index == 1)
    assert nan_probe.verdict is False
    assert nan_probe.evidence_sha256 == _canonical_sha256(
        {"score": "NaN", "threshold": 0.5, "larger_transfers": True, "evidence": {}}
    )

    inf_result = first_true(make_cuts(2), lambda cut: (cut.index >= 1, {"score": math.inf}))
    assert inf_result.first_true_index == 1
    assert verify_ledger(inf_result.probes)
    assert inf_result.probes[0].evidence_sha256 == _canonical_sha256({"score": "Infinity"})


def test_splice_bisect_evidence_hash_covers_score_and_threshold():
    result = splice_bisect(make_cuts(2), lambda cut: (float(cut.index), {"note": "n"}), 1.0)
    probe = result.probes[0]
    expected = _canonical_sha256(
        {"score": 0.0, "threshold": 1.0, "larger_transfers": True, "evidence": {"note": "n"}}
    )
    assert probe.evidence_sha256 == expected


def test_ledger_chain_verifies_and_head_matches():
    result = first_true(make_cuts(16), CountingPredicate(step_lattice(16, 9)))
    assert verify_ledger(result.probes)
    assert result.ledger_sha256 == result.probes[-1].sha256
    assert result.probes[0].prev_sha256 == GENESIS_SHA256
    for previous, current in zip(result.probes, result.probes[1:], strict=False):
        assert current.prev_sha256 == previous.sha256


def test_ledger_records_are_independently_recomputable():
    predicate = CountingPredicate([False, True])
    result = first_true(make_cuts(2), predicate)
    probe = result.probes[0]
    body = {key: value for key, value in probe.to_json_dict().items() if key != "sha256"}
    assert probe.sha256 == _canonical_sha256(body)
    assert probe.evidence_sha256 == _canonical_sha256({"lattice_value": False})


def test_verify_ledger_detects_tampering():
    result = first_true(make_cuts(16), CountingPredicate(step_lattice(16, 9)))
    records = [probe.to_json_dict() for probe in result.probes]
    assert verify_ledger(records)

    flipped = [dict(record) for record in records]
    flipped[1]["verdict"] = not flipped[1]["verdict"]
    assert not verify_ledger(flipped)

    reindexed = [dict(record) for record in records]
    reindexed[2]["index"] = 999
    assert not verify_ledger(reindexed)

    assert not verify_ledger(records[1:])
    dropped = records[:1] + records[2:]
    assert not verify_ledger(dropped)
    assert verify_ledger([])


def test_result_to_json_dict_round_trips():
    result = first_true(make_cuts(4), CountingPredicate([False, False, True, True]))
    payload = json.loads(json.dumps(result.to_json_dict()))
    assert payload["schema"] == BISECT_SCHEMA
    assert payload["verb"] == "first_true"
    assert payload["first_true_index"] == 2
    assert payload["first_true_cut_id"] == "cut/2"
    assert payload["probe_count"] == len(payload["probes"]) == 4
    assert payload["lattice_size"] == 4
    assert payload["linear_probe_count"] == 4
    assert payload["monotonicity_violation"] is False
    assert payload["ledger_sha256"] == payload["probes"][-1]["sha256"]
    assert verify_ledger(payload["probes"])


def test_cli_demo_and_verify(tmp_path, capsys):
    lattice_path = tmp_path / "lattice.json"
    lattice_path.write_text(json.dumps({"lattice": [False, False, True, True]}), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"

    assert main(["demo", str(lattice_path), "--out", str(receipt_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["first_true_index"] == 2
    assert printed["verb"] == "first_true"
    assert verify_ledger(printed["probes"])

    assert main(["verify", str(receipt_path)]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verified"] is True
    assert verdict["chain_valid"] is True
    assert verdict["head_matches"] is True


def test_cli_verify_rejects_tampered_receipt(tmp_path, capsys):
    lattice_path = tmp_path / "lattice.json"
    lattice_path.write_text(json.dumps({"lattice": [False, True, True]}), encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    assert main(["demo", str(lattice_path), "--out", str(receipt_path)]) == 0
    capsys.readouterr()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["probes"][0]["verdict"] = not receipt["probes"][0]["verdict"]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert main(["verify", str(receipt_path)]) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verified"] is False
    assert verdict["chain_valid"] is False


def test_verify_ledger_non_mapping_probe_false_and_cli_exits_1(tmp_path, capsys):
    assert verify_ledger(["not-a-dict"]) is False
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"probes": ["not-a-dict"]}), encoding="utf-8")
    assert main(["verify", str(receipt_path)]) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verified"] is False
    assert verdict["chain_valid"] is False


def test_cli_demo_rejects_malformed_lattice(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"lattice": [0, 1, 2]}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["demo", str(bad_path)])

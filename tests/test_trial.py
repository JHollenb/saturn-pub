"""Offline tests for the Instrument Trial port: bundle verifier and arbitration.

Tiny, torch-free, and well under 30 s. The verifier runs against the packaged
hash-pinned bundle; arbitration runs against a parameter-free integer redundant-writer
store that reproduces the distributed-store pattern from bundle case 1.
"""

import json
import shutil

import pytest

from saturn_pub import Act, Adapter, Session
from saturn_pub.trial import (
    DecisionRule,
    Reading,
    Trap,
    TrialRow,
    arbitrate,
    traps,
    verify_bundle,
)


class StoreCircuit(Adapter):
    """Two redundant writers each reset the carrier to the source, then a consumer reads.

    Ablating one writer's result is repaired by the next; ablating across both writers
    reaches the consumer with a zero carrier. No parameters are learned.
    """

    model_identity = "test-store-circuit-v1"
    execution = {"family": "store-circuit", "parity": "exact-integer"}

    def boundary(self, state):
        cursor = state["cursor"]
        if cursor == 3:
            return "halted"
        return f"writer:{cursor}" if cursor < 2 else "consumer"

    def validate(self, state):
        if set(state) != {"source", "carrier", "out", "cursor"}:
            raise ValueError("incomplete store-circuit closure")
        for name in ("source", "carrier", "out", "cursor"):
            if type(state[name]) is not int:
                raise ValueError(f"integer slot required: {name}")
        if not 0 <= state["cursor"] <= 3:
            raise ValueError("invalid cursor")

    def addresses(self, state):
        return ("source", "carrier", "out", "cursor")

    def advance(self, state):
        cursor = state["cursor"]
        if cursor == 3:
            raise ValueError("circuit halted")
        result = dict(state)
        if cursor < 2:
            result["carrier"] = state["source"]
        else:
            result["out"] = 1 if state["carrier"] >= 1 else 0
        result["cursor"] = cursor + 1
        return result

    def session(self):
        return Session(self, {"source": 1, "carrier": 0, "out": 0, "cursor": 0})


def _effect(branches):
    native = branches["native"].read("out")
    candidate = branches["candidate"].read("out")
    metrics = {"effect": float(native - candidate)}
    if "sham" in branches:
        metrics["sham_effect"] = float(native - branches["sham"].read("out"))
    return metrics


def _delete_both(branch):
    branch.continue_()
    branch.apply(Act.zero("carrier"))
    branch.continue_()
    branch.apply(Act.zero("carrier"))
    branch.continue_()


def _noop(branch):
    branch.continue_(3)


NECESSITY = DecisionRule("consumer-necessity", "1", "effect", 0.5, 0.15)


# --- bundle verifier ---------------------------------------------------------------


def test_packaged_bundle_reproduces_all_verdicts():
    result = verify_bundle()
    assert result["ok"] is True
    assert result["hash_ok"] is True
    assert result["verdict_ok"] is True
    # Six cases derive eleven arm verdicts; every one matches the published expectation.
    assert len(result["verdicts"]) == 11
    assert all(row["ok"] for row in result["verdicts"].values())
    assert all(row["ok"] for row in result["hashes"])
    assert result["verdicts"]["case-4/community"]["derived"] == "state recovered"
    assert result["verdicts"]["case-4/saturn"]["derived"].startswith("state not recovered")


def test_verify_detects_a_tampered_receipt(tmp_path):
    bundle = tmp_path / "bundle"
    shutil.copytree(verify_bundle()["bundle"], bundle)
    victim = bundle / "case-1" / "circuit-panel-ledger.json"
    victim.write_bytes(victim.read_bytes() + b" ")
    result = verify_bundle(bundle=bundle)
    assert result["ok"] is False
    assert result["hash_ok"] is False
    bad = [row for row in result["hashes"] if not row["ok"]]
    assert [row["file"] for row in bad] == ["case-1/circuit-panel-ledger.json"]


def test_verify_detects_a_wrong_expected_verdict(tmp_path):
    packaged = verify_bundle()
    tampered = tmp_path / "expected.json"
    data = {case: {} for case in {k.split("/")[0] for k in packaged["verdicts"]}}
    for key, row in packaged["verdicts"].items():
        case, arm = key.split("/", 1)
        data[case][arm] = row["derived"]
    data["case-4"]["community"] = "this is not the derived verdict"
    tampered.write_text(json.dumps(data))
    result = verify_bundle(expected=tampered)
    assert result["ok"] is False
    assert result["verdict_ok"] is False
    assert result["verdicts"]["case-4/community"]["ok"] is False


# --- arbitration -------------------------------------------------------------------


def test_naive_single_site_reading_inverts():
    reading = Reading("single-site activation patching", "no necessary component", False)
    row = arbitrate(
        StoreCircuit().session(), reading, NECESSITY, effect=_effect, candidate=_delete_both, steps=3
    )
    assert isinstance(row, TrialRow)
    assert row.verdict == "invert"
    assert row.classification == "present"
    assert row.consumer_effect == 1.0
    assert row.controls["exact_gate"] is True
    assert row.receipt and row.parent


def test_consumer_gated_reading_agrees():
    reading = Reading("consumer-gated all-step ablation", "carrier store is load-bearing", True)
    row = arbitrate(
        StoreCircuit().session(), reading, NECESSITY, effect=_effect, candidate=_delete_both, steps=3
    )
    assert row.verdict == "agree"
    assert row.classification == "present"


def test_declined_reading_is_inconclusive():
    reading = Reading("single-step mediation", "holds at the measured step", True, declined=True)
    row = arbitrate(
        StoreCircuit().session(), reading, NECESSITY, effect=_effect, candidate=_delete_both, steps=3
    )
    assert row.verdict == "inconclusive"
    assert "declined" in row.reason


def test_absent_effect_agrees_with_a_no_effect_reading():
    reading = Reading("single-site activation patching", "no effect at this site", False)
    row = arbitrate(
        StoreCircuit().session(), reading, NECESSITY, effect=_effect, candidate=_noop, steps=3
    )
    assert row.classification == "absent"
    assert row.verdict == "agree"


def test_ambiguity_band_is_inconclusive():
    rule = DecisionRule("wide-band", "1", "effect", 5.0, -5.0)
    reading = Reading("probe", "no decodable content", False)
    row = arbitrate(
        StoreCircuit().session(), reading, rule, effect=_effect, candidate=_delete_both, steps=3
    )
    assert row.classification == "ambiguous"
    assert row.verdict == "inconclusive"


def test_sham_control_is_recorded():
    rule = DecisionRule(
        "necessity-sham", "1", "effect", 0.5, 0.15,
        require_sham_flat=True, sham_metric="sham_effect", sham_tolerance=0.1,
    )
    reading = Reading("single-site activation patching", "no necessary component", False)
    row = arbitrate(
        StoreCircuit().session(), reading, rule, effect=_effect,
        candidate=_delete_both, sham=_noop, steps=3,
    )
    assert row.controls["sham_flat"] is True
    assert row.controls["sham_value"] == 0.0
    assert row.verdict == "invert"


def test_arbitrate_from_adapter_and_parent():
    session = StoreCircuit().session()
    parent = session.capture()
    reading = Reading("consumer-gated all-step ablation", "load-bearing", True)
    row = arbitrate(
        StoreCircuit(), reading, NECESSITY, effect=_effect,
        candidate=_delete_both, steps=3, parent=parent,
    )
    assert row.verdict == "agree"


def test_arbitrate_from_adapter_without_parent_refuses():
    reading = Reading("probe", "no content", False)
    with pytest.raises(ValueError, match="requires a parent"):
        arbitrate(StoreCircuit(), reading, NECESSITY, effect=_effect, candidate=_delete_both, steps=3)


def test_single_act_intervention_is_accepted():
    reading = Reading("ablation", "a single carrier zero has no effect (it is repaired)", False)
    row = arbitrate(
        StoreCircuit().session(), reading, NECESSITY, effect=_effect,
        candidate=Act.zero("carrier"), steps=3,
    )
    # One zero at the parent is reconstructed by the redundant writers; the consumer never changes.
    assert row.classification == "absent"
    assert row.consumer_effect == 0.0
    assert row.verdict == "agree"


# --- reading / rule / traps contracts ----------------------------------------------


def test_reading_external_seam_tags_provenance():
    reading = Reading.from_external("circuit-tracer", "edge is not load-bearing", False)
    assert reading.source == "circuit-tracer"
    assert reading.instrument == "external:circuit-tracer"
    assert reading.to_dict()["source"] == "circuit-tracer"


def test_reading_and_rule_fingerprints_are_content_addressed():
    a = Reading("probe", "no decodable content", False)
    b = Reading("probe", "no decodable content", False)
    c = Reading("probe", "decodable: wolf", True)
    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint

    r1 = DecisionRule("rule", "1", "effect", 0.5, 0.15)
    r2 = DecisionRule("rule", "1", "effect", 0.9, 0.15)
    assert r1.fingerprint != r2.fingerprint
    assert r1.to_dict()["fingerprint"] == r1.fingerprint


def test_decision_rule_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="present_threshold must be >= absent_threshold"):
        DecisionRule("bad", "1", "effect", 0.1, 0.9)


def test_traps_checklist_is_complete():
    rows = traps()
    assert len(rows) == 7
    assert all(isinstance(trap, Trap) for trap in rows)
    assert [trap.number for trap in rows] == list(range(1, 8))
    assert all(trap.gate and trap.failure for trap in rows)

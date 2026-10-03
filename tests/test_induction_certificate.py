"""Offline tests for the induction-certificate gate, workload, and custody binding.

Stdlib-only and well under 30 s: no torch, no model framework. Outcome vectors are
synthesized directly so the frozen decision rule is exercised in isolation.
"""

import json

import pytest

from saturn_pub.certificate import (
    INDUCTION_CERTIFICATE_SCHEMA,
    INDUCTION_PANEL_SCHEMA,
    CertificateError,
    ConsumerContract,
    InductionCircuitPolicy,
    InductionPanelSpec,
    build_relational_induction_workload,
    certificate_receipt,
    certify_induction_route,
    register_certificate_claim,
    relational_induction_positions,
    write_certificate,
    wrong_donor_indices,
)
from saturn_pub.evidence import ClaimsRegistry, verify


def _branch(correct, margin):
    return {"correct": correct, "margin": [margin] * len(correct)}


def _execution(**overrides):
    base = {
        "consumer": "native_final_norm_lm_head",
        "backend": "native-decoder-v1",
        "device": "cpu",
        "dtype": "torch.float32",
    }
    base.update(overrides)
    return base


def _panel(panel_id, *, execution=None):
    total = 192
    clean = [1.0] * 190 + [0.0] * 2
    return {
        "schema": INDUCTION_PANEL_SCHEMA,
        "panel_id": panel_id,
        "examples": total,
        "classes": 48,
        "branches": {
            "clean": _branch(clean, 6.0),
            "source_deletion": _branch([0.0] * total, -8.0),
            "match_deletion": _branch([1.0] * 188 + [0.0] * 4, 5.0),
            "correct_repair": _branch(clean, 6.0),
            "wrong_repair": _branch([0.0] * total, -7.0),
        },
        "execution": execution or _execution(),
    }


# --- gate ------------------------------------------------------------------------------


def test_two_clean_panels_certify_a_bounded_route():
    result = certify_induction_route([_panel("a"), _panel("b")])
    assert result["certified"] is True
    assert result["schema"] == INDUCTION_CERTIFICATE_SCHEMA
    assert result["gates"] == {"replication": True, "all_panels_certified": True}
    assert result["panels"][0]["metrics"]["source_deletion_accuracy"] == 0.0
    assert len(result["content_sha256"]) == 64
    # every per-panel gate passed: two panels x seven gates = fourteen here
    assert all(all(p["gates"].values()) for p in result["panels"])


def test_content_hash_is_stable_and_covers_the_body():
    first = certify_induction_route([_panel("a"), _panel("b")])
    again = certify_induction_route([_panel("a"), _panel("b")])
    assert first["content_sha256"] == again["content_sha256"]
    # a one-example change to an outcome vector moves the seal
    changed = _panel("a")
    changed["branches"]["clean"]["correct"][0] = 0.0
    moved = certify_induction_route([changed, _panel("b")])
    assert moved["content_sha256"] != first["content_sha256"]


def test_single_panel_fails_replication_by_default_policy():
    result = certify_induction_route([_panel("solo")])
    assert result["gates"]["replication"] is False
    assert result["gates"]["all_panels_certified"] is True
    assert result["certified"] is False


def test_wrong_donor_success_fails_specificity_without_erasing_other_gates():
    panel = _panel("failed-specificity")
    panel["branches"]["wrong_repair"] = _branch([1.0] * 150 + [0.0] * 42, 2.0)
    result = certify_induction_route([panel], InductionCircuitPolicy(min_panels=1))
    measured = result["panels"][0]
    assert measured["gates"]["wrong_donor_specificity"] is False
    assert measured["gates"]["source_necessity"] is True
    assert result["certified"] is False


def test_repair_must_replay_the_clean_decision_and_margin():
    panel = _panel("failed-replay")
    panel["branches"]["correct_repair"] = _branch([0.0] * 2 + [1.0] * 190, 4.0)
    result = certify_induction_route([panel], InductionCircuitPolicy(min_panels=1))
    gates = result["panels"][0]["gates"]
    assert gates["correct_repair_sufficiency"] is True
    assert gates["repair_replay_fidelity"] is False
    assert result["certified"] is False


def test_source_necessity_uses_a_wilson_upper_bound():
    panel = _panel("necessity")
    # 10/192 successes: point estimate is below chance+slack but the 1.96z upper bound is not
    panel["branches"]["source_deletion"] = _branch([1.0] * 10 + [0.0] * 182, -4.0)
    result = certify_induction_route([panel], InductionCircuitPolicy(min_panels=1))
    assert result["panels"][0]["gates"]["source_necessity"] is False


def test_surrogate_consumer_fails_native_continuation_gate():
    panel = _panel("surrogate", execution=_execution(consumer="learned_linear_probe"))
    result = certify_induction_route([panel], InductionCircuitPolicy(min_panels=1))
    assert result["panels"][0]["gates"]["native_consumer_continuation"] is False
    assert result["certified"] is False


def test_pinned_backend_and_device_are_enforced():
    policy = InductionCircuitPolicy(
        min_panels=1,
        consumer=ConsumerContract(backends=("native-decoder-v1",), device_prefixes=("cuda",)),
    )
    cpu_panel = _panel("on-cpu")
    assert certify_induction_route([cpu_panel], policy)["panels"][0]["gates"][
        "native_consumer_continuation"
    ] is False
    gpu_panel = _panel("on-gpu", execution=_execution(device="cuda:0"))
    assert certify_induction_route([gpu_panel], policy)["panels"][0]["gates"][
        "native_consumer_continuation"
    ] is True


def test_duplicate_panel_ids_are_refused():
    with pytest.raises(CertificateError):
        certify_induction_route([_panel("dup"), _panel("dup")])


def test_malformed_panels_fail_closed():
    with pytest.raises(CertificateError):
        certify_induction_route([{"schema": "wrong"}], InductionCircuitPolicy(min_panels=1))
    short = _panel("short", execution=_execution())
    short["branches"]["clean"]["correct"].append(1.0)  # misaligned row count
    with pytest.raises(CertificateError):
        certify_induction_route([short], InductionCircuitPolicy(min_panels=1))


def test_policy_rejects_out_of_range_thresholds():
    with pytest.raises(CertificateError):
        InductionCircuitPolicy(clean_min_accuracy=1.5)
    with pytest.raises(CertificateError):
        InductionCircuitPolicy(min_panels=0)


def test_policy_round_trips_through_its_dict():
    policy = InductionCircuitPolicy(
        min_panels=3, consumer=ConsumerContract(device_prefixes=("cuda",))
    )
    body = policy.to_dict()
    assert body["min_panels"] == 3
    assert body["consumer"]["device_prefixes"] == ["cuda"]
    assert json.dumps(body)  # fully JSON-serializable for the content seal


# --- workload --------------------------------------------------------------------------


def test_workload_positions_are_consistent_and_deterministic():
    spec = InductionPanelSpec("wl", seed=2026, examples=64, length=12, classes=48)
    items, classes = build_relational_induction_workload(spec)
    again, _ = build_relational_induction_workload(spec)
    assert classes == 48
    assert len(items) == 64
    assert [i["tokens"] for i in items] == [i["tokens"] for i in again]  # seed determinism
    for item in items:
        match, source = relational_induction_positions(item)
        assert (match, source) == (item["match"], item["source"])
        # the answer token sits at the earlier source position and is the labelled candidate
        assert item["tokens"][source] == item["cands"][item["label"]]
        assert item["query"] == len(item["tokens"]) - 1


def test_workload_rejects_impossible_shapes():
    with pytest.raises(ValueError):
        InductionPanelSpec("x", seed=1, length=1)
    with pytest.raises(ValueError):
        InductionPanelSpec("x", seed=1, length=50, classes=48)


def test_wrong_donor_indices_pick_a_different_label():
    labels = [0, 0, 1, 1]
    donors = wrong_donor_indices(labels)
    assert all(labels[donors[row]] != labels[row] for row in range(len(labels)))
    with pytest.raises(ValueError):
        wrong_donor_indices([3, 3, 3])


# --- custody binding -------------------------------------------------------------------


def test_certificate_registers_as_a_claim_and_drift_detects(tmp_path):
    cert = certify_induction_route([_panel("a"), _panel("b")])
    receipt = certificate_receipt(cert)
    assert len(receipt.fingerprint) == 64

    cert_path = tmp_path / "certificate.json"
    registry = ClaimsRegistry(tmp_path / "claims.jsonl")
    row = register_certificate_claim(
        registry,
        cert,
        "ind-cert-demo",
        "bounded induction source circuit certified on two sealed panels",
        cert_path,
        worker="induction-certificate",
    )
    assert row.status == "measured"
    assert row.receipt_content_sha256 == cert["content_sha256"]
    assert row.gates == {"replication": True, "all_panels_certified": True}
    assert registry.verify_chain() is True

    report = verify(registry)
    assert report.newly_stale == 0

    # tampering with the on-disk certificate drifts the claim stale
    write_certificate({**cert, "content_sha256": "0" * 64}, cert_path)
    assert registry.verify_chain() is True
    report_after = verify(registry)
    assert report_after.newly_stale == 1


def test_failed_certificate_registers_as_refuted(tmp_path):
    panel = _panel("fail", execution=_execution(consumer="surrogate"))
    cert = certify_induction_route([panel, _panel("ok")])
    assert cert["certified"] is False
    registry = ClaimsRegistry(tmp_path / "claims.jsonl")
    row = register_certificate_claim(
        registry, cert, "ind-cert-fail", "induction circuit did not certify", tmp_path / "c.json"
    )
    assert row.status == "refuted"

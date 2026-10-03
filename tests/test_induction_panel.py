"""Measurement-helper mechanics on tiny random decoders (torch, CPU, well under 30 s).

Random-weight tiny models do not perform induction, so these tests check the *mechanics*
and invariants of the causal battery, not certification:

* the five branches run and feed the gate;
* the full-parent repair replays the clean decision and margin bit-exactly
  (``repair_correctness_mismatch_fraction == 0``), which is a mechanical identity that
  must hold for any weights;
* deleting the source edge actually perturbs the forward (a non-trivial intervention);
* every branch terminates at the native consumer, and a device-pinned policy fails on CPU;
* the Session/fork consumer cross-check agrees with the batched clean decision;
* an unsupported attention family is refused fail-closed.
"""

import pytest

from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.certificate import (
    ConsumerContract,
    InductionCircuitPolicy,
    InductionPanelSpec,
    certify_induction_route,
)
from saturn_pub.certificate.panel import certify_decoder_induction, measure_induction_panel

# small, fast alphabet; length >= 2, classes >= length, vocab is 64 on the tiny fixtures
_SPEC_A = InductionPanelSpec("tiny-a", seed=1, examples=24, length=4, classes=16, token_low=4, token_high=60)
_SPEC_B = InductionPanelSpec("tiny-b", seed=2, examples=24, length=5, classes=16, token_low=4, token_high=60)


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_panel_mechanics_and_repair_identity(family):
    adapter = DecoderAdapter.tiny(family, seed=5)
    panel = measure_induction_panel(adapter, _SPEC_A, batch_size=8)

    assert panel["schema"] == "saturn-pub-induction-panel-v1"
    assert panel["examples"] == 24
    for name in ("clean", "source_deletion", "match_deletion", "correct_repair", "wrong_repair"):
        assert len(panel["branches"][name]["correct"]) == 24
        assert len(panel["branches"][name]["margin"]) == 24
        assert all(c in (0.0, 1.0) for c in panel["branches"][name]["correct"])

    # the gate parses the measured panel and computes per-panel metrics
    result = certify_induction_route([panel], InductionCircuitPolicy(min_panels=1))
    metrics = result["panels"][0]["metrics"]

    # mechanical identity: full-parent repair reproduces the clean decision and margin exactly
    assert metrics["repair_correctness_mismatch_fraction"] == 0.0
    assert metrics["repair_margin_mae"] == pytest.approx(0.0, abs=1e-6)
    assert result["panels"][0]["gates"]["correct_repair_sufficiency"] is True
    assert result["panels"][0]["gates"]["repair_replay_fidelity"] is True

    # the source deletion is a real intervention: it moves the margin for some example
    moved = any(
        abs(a - b) > 1e-6
        for a, b in zip(
            panel["branches"]["clean"]["margin"], panel["branches"]["source_deletion"]["margin"]
        )
    )
    assert moved

    # native-consumer gate holds with the default contract; the route is the full parent
    assert result["panels"][0]["gates"]["native_consumer_continuation"] is True
    assert panel["route_edge_count"] == adapter.layers * adapter.num_heads


@pytest.mark.parametrize("family", ["qwen2", "llama"])
def test_session_fork_consumer_crosscheck_agrees(family):
    adapter = DecoderAdapter.tiny(family, seed=6)
    panel = measure_induction_panel(adapter, _SPEC_A, batch_size=8)
    crosscheck = panel["execution"]["session_consumer_crosscheck"]
    assert crosscheck["agrees"] is True


def test_device_pinned_policy_fails_on_cpu_specimen():
    adapter = DecoderAdapter.tiny("qwen2", seed=7)
    policy = InductionCircuitPolicy(
        min_panels=2, consumer=ConsumerContract(device_prefixes=("cuda",))
    )
    bundle = certify_decoder_induction(adapter, [_SPEC_A, _SPEC_B], policy, batch_size=8)
    # the CPU specimen cannot satisfy a cuda-pinned consumer contract: a legitimate FAIL
    for measured in bundle["certificate"]["panels"]:
        assert measured["gates"]["native_consumer_continuation"] is False
    assert bundle["certificate"]["certified"] is False


def test_certify_decoder_induction_round_trip():
    adapter = DecoderAdapter.tiny("llama", seed=8)
    bundle = certify_decoder_induction(
        adapter, [_SPEC_A, _SPEC_B], InductionCircuitPolicy(), batch_size=8
    )
    cert = bundle["certificate"]
    assert cert["panel_count"] == 2
    assert cert["gates"]["replication"] is True  # two distinct panels supplied
    assert len(cert["content_sha256"]) == 64
    # mechanical repair identity holds on both panels regardless of (random) certification
    assert all(
        p["metrics"]["repair_correctness_mismatch_fraction"] == 0.0 for p in cert["panels"]
    )


def test_unsupported_attention_family_is_refused():
    adapter = DecoderAdapter.tiny("gpt2", seed=9)
    with pytest.raises(ValueError, match="llama-style attention"):
        measure_induction_panel(adapter, _SPEC_A, batch_size=8)


def test_candidate_token_outside_vocab_is_refused():
    adapter = DecoderAdapter.tiny("qwen2", seed=10)  # tiny vocab is 64
    spec = InductionPanelSpec("oob", seed=1, examples=8, length=4, classes=16, token_low=60, token_high=200)
    with pytest.raises(ValueError, match="vocabulary"):
        measure_induction_panel(adapter, spec, batch_size=8)

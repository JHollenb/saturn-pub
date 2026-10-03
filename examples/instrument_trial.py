"""Arbitrate standard-instrument readings against the native consumer, and verify the bundle.

Run: python examples/instrument_trial.py

Offline, tiny, no downloads. Two parts:

1. verify_bundle() re-derives the six shipped Instrument Trial case verdicts from the
   hash-pinned receipt bundle (stdlib, no torch).

2. On the supplied two-writer redundant circuit (examples/repair_circuit.py), the same
   distributed-store ablation is arbitrated under two different declared readings:
     * a naive single-site patching reading ("no necessary component") INVERTS, because
       the unchanged native consumer shows the carrier store is load-bearing across steps;
     * a consumer-gated all-step reading ("the store is load-bearing") AGREES.
   The arbiter is the consumer, not an internal metric; the verdict depends only on what
   the instrument claimed about the same measured effect.
"""

from __future__ import annotations

import json
from pathlib import Path

from repair_circuit import RepairCircuit

from saturn_pub import Act
from saturn_pub.trial import DecisionRule, Reading, arbitrate, traps, verify_bundle


def consumer_effect(branches):
    """Consumer closure: how much native target contact the intervention removed, in [0, 1]."""
    native = branches["native"].read("output").tolist()[0]
    candidate = branches["candidate"].read("output").tolist()[0]
    metrics = {"target_removed": float(native) - float(candidate)}
    if "sham" in branches:
        sham = branches["sham"].read("output").tolist()[0]
        metrics["sham_removed"] = float(native) - float(sham)
    return metrics


def delete_across_steps(branch):
    """The distributed store: zero the carrier after each redundant writer, then let it read."""
    branch.continue_()  # writer.first  -> carrier reconstructed
    branch.apply(Act.zero("carrier"))
    branch.continue_()  # writer.repair -> carrier reconstructed again (self-repair)
    branch.apply(Act.zero("carrier"))
    branch.continue_()  # consumer.threshold reads the zeroed carrier


def single_site_control(branch):
    """What single-site patching actually sees: one ablation, repaired downstream."""
    branch.continue_()
    branch.apply(Act.zero("carrier"))
    branch.continue_(2)


def main(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    # Part 1: offline bundle verifier.
    verified = verify_bundle()
    print("Bundle verifier:", "all hashes + verdicts reproduce" if verified["ok"] else "FAILED")
    for key, row in sorted(verified["verdicts"].items()):
        print(f"  {key:26s} {row['derived']}")

    # Part 2: consumer-gated arbitration on the supplied redundant-writer circuit.
    rule = DecisionRule(
        name="consumer-necessity",
        version="1",
        metric="target_removed",
        present_threshold=0.5,
        absent_threshold=0.15,
        require_sham_flat=True,
        sham_metric="sham_removed",
        sham_tolerance=0.1,
        description="carrier store is load-bearing iff the native consumer loses most target "
        "contact under all-step ablation, with a flat sham",
    )

    # Show what a single-site instrument measures: the ablation is repaired, output unchanged.
    probe = RepairCircuit().session()
    single_site_control(probe)
    single_site_target = probe.read("output").tolist()[0]

    readings = {
        "naive_single_site": Reading(
            instrument="single-site activation patching",
            claim="no necessary component: ablating any one writer leaves the output intact",
            asserts_effect=False,
            detail={"single_site_target_contact": single_site_target},
        ),
        "consumer_gated": Reading(
            instrument="consumer-gated all-step route ablation",
            claim="the carrier store is jointly load-bearing across steps",
            asserts_effect=True,
        ),
    }

    rows = {}
    for name, reading in readings.items():
        session = RepairCircuit().session()
        row = arbitrate(
            session,
            reading,
            rule,
            effect=consumer_effect,
            candidate=delete_across_steps,
            sham=single_site_control,
            steps=3,
        )
        rows[name] = row
        print(f"\n{name}: {row.verdict.upper()}  ({row.reason})")
        print(f"  consumer effect (target removed): {row.consumer_effect}")
        print(f"  controls: {dict(row.controls)}")

    assert single_site_target == 1.0, "single-site ablation should be repaired (output intact)"
    assert rows["naive_single_site"].verdict == "invert"
    assert rows["consumer_gated"].verdict == "agree"
    assert rows["naive_single_site"].controls["sham_flat"] is True

    report = {
        "schema": "saturn-pub-instrument-trial-demo-v1",
        "bundle_verifier": {
            "ok": verified["ok"],
            "verdicts": {k: v["derived"] for k, v in verified["verdicts"].items()},
        },
        "specimen": "supplied two-writer redundant circuit; CPU float32; no training",
        "decision_rule": rule.to_dict(),
        "rows": {name: row.to_dict() for name, row in rows.items()},
        "traps": [trap.to_dict() for trap in traps()],
        "note": "A reading is a candidate, not a verdict; the unchanged native consumer decides.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport: {output / 'report.json'}")
    print(
        "\nSame measured effect, two readings: the naive single-site claim inverts, the "
        "consumer-gated claim agrees. The consumer is the arbiter."
    )


if __name__ == "__main__":
    main(Path("outputs/instrument-trial"))

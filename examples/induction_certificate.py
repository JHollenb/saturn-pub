"""Declare an induction-circuit certificate in advance, measure it, and read the gates.

This is a *mechanics* demo on a tiny randomly initialized Qwen2 model, so it runs in a
second on CPU with no checkpoint download. Random weights do not perform induction, so
the ``clean_behavior`` gate fails and the certificate is honestly reported as NOT
certified -- a failing gate is a legitimate result, named rather than tuned away. The
real-weight certificate on cached small decoders lives in
``experiments/induction_certificate/``.

The pattern is the point:

1. Freeze the policy (thresholds + native-consumer contract) and the held-out panels
   BEFORE any measurement.
2. Run the causal battery on a native ``saturn_pub`` decoder adapter -- every branch
   terminates at the model's own final norm + lm head.
3. Hand the measured outcomes to the frozen gate and read the signed verdict.
4. Register the certificate in the evidence plane so it drift-detects like any claim.

    python examples/induction_certificate.py
"""

from __future__ import annotations

import json
from pathlib import Path

from saturn_pub.adapters.decoder import DecoderAdapter
from saturn_pub.certificate import (
    InductionCircuitPolicy,
    InductionPanelSpec,
    register_certificate_claim,
)
from saturn_pub.certificate.panel import certify_decoder_induction
from saturn_pub.evidence import ClaimsRegistry

OUTPUT = Path("outputs/induction-certificate")


def main() -> None:
    # 1. Freeze the decision rule and the sealed panels in advance.
    policy = InductionCircuitPolicy()  # default thresholds; native-consumer contract
    specs = [
        InductionPanelSpec(
            "demo-len4-seed1", seed=1, examples=48, length=4, classes=16, token_low=4, token_high=60
        ),
        InductionPanelSpec(
            "demo-len5-seed2", seed=2, examples=48, length=5, classes=16, token_low=4, token_high=60
        ),
    ]

    # 2+3. Measure the battery on a native decoder adapter and apply the frozen gate.
    adapter = DecoderAdapter.tiny("qwen2", seed=5)
    bundle = certify_decoder_induction(adapter, specs, policy, batch_size=16)
    certificate = bundle["certificate"]

    # Gate table.
    columns = [
        "clean_behavior",
        "source_necessity",
        "position_specificity",
        "correct_repair_sufficiency",
        "wrong_donor_specificity",
        "repair_replay_fidelity",
        "native_consumer_continuation",
    ]
    print(
        f"certificate certified: {certificate['certified']}  (tiny random weights -> expected FAIL)"
    )
    print(f"content sha256: {certificate['content_sha256']}")
    header = "panel".ljust(22) + "  " + "  ".join(c[:10].rjust(10) for c in columns)
    print(header)
    for panel in certificate["panels"]:
        cells = "  ".join(("yes" if panel["gates"][c] else "no").rjust(10) for c in columns)
        print(panel["panel_id"].ljust(22) + "  " + cells)
        m = panel["metrics"]
        print(
            f"  clean={m['clean_accuracy']:.3f} source_cut={m['source_deletion_accuracy']:.3f} "
            f"repair_mismatch={m['repair_correctness_mismatch_fraction']:.3f} "
            f"repair_margin_mae={m['repair_margin_mae']:.4f}"
        )
    print(f"top-level gates: {certificate['gates']}")

    # 4. Register the certificate in the evidence plane (drift-detectable claim).
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = ClaimsRegistry(OUTPUT / "claims.jsonl")
    row = register_certificate_claim(
        registry,
        certificate,
        "induction-certificate-demo",
        "bounded relational-induction source circuit on a tiny random Qwen2 specimen",
        OUTPUT / "certificate.json",
        worker="examples/induction_certificate.py",
    )
    print(
        f"registered claim {row.claim_id!r} with status {row.status!r}; chain ok: "
        f"{registry.verify_chain()}"
    )

    report = {
        "schema": "saturn-pub-induction-certificate-demo-v1",
        "certified": certificate["certified"],
        "content_sha256": certificate["content_sha256"],
        "policy": certificate["policy"],
        "panels": [
            {"panel_id": p["panel_id"], "gates": p["gates"], "metrics": p["metrics"]}
            for p in certificate["panels"]
        ],
        "claim_status": row.status,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT / 'report.json'}")


if __name__ == "__main__":
    main()

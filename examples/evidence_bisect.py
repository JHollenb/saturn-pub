"""Offline evidence plane: bisect two branches, cite the receipt, verify the claim.

A tiny random Qwen2 (no download) is stepped to a shared parent cut. A native
arm continues unchanged; a perturbed arm zeros one carrier and continues. The
two arms are retained ``StateCut`` branches, so the only thing the bisect sees
is a caller callback that replays each arm to a cut and returns its fingerprint.
``first_divergence_pairs`` then locates the first divergent decode step in a
logarithmic number of replays instead of scanning every step.

The bisect receipt is written to disk, registered as a claim citing that
receipt, and verified; the claim auto-stales if the receipt later changes. The
address of the divergent cut is cited into the receipt and indexed with the
cross-reference index, so all four evidence verbs appear in one run. Everything
runs offline in a few seconds.
"""

import json
from pathlib import Path

from saturn_pub import Act
from saturn_pub.adapters.qwen import QwenAdapter
from saturn_pub.evidence import (
    ClaimsRegistry,
    build_index,
    first_divergence_pairs,
    ingest_receipt,
    make_citation,
    make_cuts,
    probe_economics,
    query_refs,
    verify,
    verify_ledger,
)

OUT = Path("outputs/evidence-bisect")
OUT.mkdir(parents=True, exist_ok=True)

adapter = QwenAdapter.tiny()
session = adapter.session([5, 7, 11])
session.continue_(2)  # token embedding boundary, then native decoder layer 0
parent = session.capture()  # the one cut both arms share at step 0

STEPS = 24


def native_digest(offset: int) -> str:
    """Native arm replayed to ``offset`` micro-transitions from the parent."""
    if offset == 0:
        return parent.fingerprint
    arm = session.fork(parent)
    arm.continue_(offset)
    return arm.capture(retain=False).fingerprint


def perturbed_digest(offset: int) -> str:
    """Perturbed arm: zero one carrier at the parent, then replay to ``offset``."""
    if offset == 0:
        return parent.fingerprint  # the intervention has not happened yet
    arm = session.fork(parent)
    arm.apply(Act.zero("hidden"))
    arm.continue_(offset)
    return arm.capture(retain=False).fingerprint


lattice = make_cuts(STEPS, prefix="decode-step")
result = first_divergence_pairs(
    lattice, lambda cut: (native_digest(cut.index), perturbed_digest(cut.index))
)
assert verify_ledger(result.probes)
assert not result.monotonicity_violation

divergent_step = result.first_true_index
address = f"ar://decode/step/{divergent_step}/site/hidden"
citation = make_citation(address, model_id=adapter.model_identity, quantity="first-divergence")

# Write the bisect receipt (the evidence this claim will rest on), with the
# divergent cut cited so the cross-reference index can attribute it.
receipt = result.to_json_dict()
receipt["experiment"] = "evidence-bisect-example"
receipt["divergent_address"] = citation
receipt_path = OUT / "report.json"
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

# Register a claim citing that receipt, then verify it (the receipt file is
# fingerprinted; the claim auto-stales if it ever changes).
registry = ClaimsRegistry(OUT / "claims.jsonl")
row = ingest_receipt(
    registry,
    receipt_path,
    "first-divergent-decode-step",
    f"the perturbed arm first diverges from native at decode step {divergent_step}",
)
clean = verify(registry)

# Index the addresses the receipts cite, and look the divergent one back up.
build_index(OUT, OUT / "xref.sqlite")
indexed = query_refs(OUT / "xref.sqlite", address)

summary = {
    "first_divergent_step": divergent_step,
    "economics": probe_economics(result),
    "claim_id": row.claim_id,
    "claim_sha256": row.sha256,
    "chain_ok": registry.verify_chain(),
    "verify": clean.to_dict(),
    "cited_address": address,
    "indexed_rows": [
        {"address": r["address"], "model_id": r["model_id"], "path": r["path"]} for r in indexed
    ],
}
print(json.dumps(summary, indent=2, sort_keys=True))
assert clean.still_valid == 1 and clean.chain_ok
assert indexed and indexed[0]["model_id"] == adapter.model_identity

"""The evidence plane: make the experimental record queryable and self-auditing.

Four stdlib-only verbs, sharing one custody idiom (canonical JSON + sha256 hash
chains, see :mod:`saturn_pub.evidence._canonical`):

* :mod:`~saturn_pub.evidence.bisect` -- first-divergence (and ``first_true`` /
  ``splice_bisect``) over an ordered cut lattice with a caller-supplied probe;
* :mod:`~saturn_pub.evidence.claims` -- an executable claim registry that
  hash-chains claims to their evidence receipts and auto-stales drifted ones;
* :mod:`~saturn_pub.evidence.xref` -- a SQLite cross-reference index over the
  addresses receipts cite;
* :mod:`~saturn_pub.evidence.citations` -- writer-side index-ready citations.

Nothing here imports ``torch`` or the rest of ``saturn_pub``; the bisect verbs
reach a live run only through caller callbacks. The subpackage is intentionally
self-contained so it can later move into a neutral base package.
"""

from __future__ import annotations

from ._canonical import canonical_json_bytes, content_fingerprint, sha256_hex
from .bisect import (
    BISECT_SCHEMA,
    ArmDigests,
    BisectResult,
    CutHandle,
    ProbeRecord,
    first_divergence,
    first_divergence_pairs,
    first_true,
    make_cuts,
    probe_economics,
    splice_bisect,
    verify_ledger,
)
from .citations import (
    ADDRESS_SCHEME,
    CITATION_KEY,
    CITATION_VERSION,
    citation_fields,
    is_citation,
    make_citation,
    validate_address,
)
from .claims import (
    CLAIM_STATUSES,
    CLAIMS_REGISTRY_SCHEMA,
    ClaimRow,
    ClaimsRegistry,
    ClaimsRegistryError,
    DependencyFingerprint,
    ReplayRecipe,
    VerifyReport,
    fingerprint_paths,
    gates_match,
    harvest_declared_refs,
    ingest_receipt,
    verify,
)
from .xref import (
    XREF_SCHEMA,
    BuildReport,
    EmptySweepError,
    Reference,
    SchemaVersionError,
    build_index,
    extract_addresses,
    extract_references,
    index_stats,
    query_refs,
)

__all__ = [
    # canonical custody helpers
    "canonical_json_bytes",
    "content_fingerprint",
    "sha256_hex",
    # bisect
    "BISECT_SCHEMA",
    "ArmDigests",
    "BisectResult",
    "CutHandle",
    "ProbeRecord",
    "first_divergence",
    "first_divergence_pairs",
    "first_true",
    "make_cuts",
    "probe_economics",
    "splice_bisect",
    "verify_ledger",
    # citations
    "ADDRESS_SCHEME",
    "CITATION_KEY",
    "CITATION_VERSION",
    "citation_fields",
    "is_citation",
    "make_citation",
    "validate_address",
    # claims
    "CLAIM_STATUSES",
    "CLAIMS_REGISTRY_SCHEMA",
    "ClaimRow",
    "ClaimsRegistry",
    "ClaimsRegistryError",
    "DependencyFingerprint",
    "ReplayRecipe",
    "VerifyReport",
    "fingerprint_paths",
    "gates_match",
    "harvest_declared_refs",
    "ingest_receipt",
    "verify",
    # xref
    "XREF_SCHEMA",
    "BuildReport",
    "EmptySweepError",
    "Reference",
    "SchemaVersionError",
    "build_index",
    "extract_addresses",
    "extract_references",
    "index_stats",
    "query_refs",
]

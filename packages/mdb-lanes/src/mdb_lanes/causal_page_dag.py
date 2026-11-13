"""Content-addressed exact-prefix factoring for immutable causal pages.

This module plans storage; it does not interpret model state or authorize a
semantic route.  Callers provide per-row cryptographic identities produced by
their tensor custody layer.  A shared ancestor is admitted only when every
page has the same model ABI and an identical leading sequence of row digests
and byte counts.  Children retain an exact dependency on that ancestor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

CAUSAL_PAGE_DAG_SCHEMA = "saturn-causal-page-dag-plan-v1"


class CausalPageDAGError(ValueError):
    """A page manifest or exact-prefix factorization is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: str, *, label: str) -> str:
    normalized = value.casefold()
    invalid_character = any(
        character not in "0123456789abcdef" for character in normalized
    )
    if len(normalized) != 64 or invalid_character:
        raise CausalPageDAGError(f"{label} must be a SHA-256 hex digest")
    return normalized


@dataclass(frozen=True, slots=True)
class CausalPageRow:
    """One immutable logical token row across every stored K/V layer."""

    payload_sha256: str
    payload_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload_sha256",
            _sha256(self.payload_sha256, label="row payload_sha256"),
        )
        if isinstance(self.payload_bytes, bool) or int(self.payload_bytes) <= 0:
            raise CausalPageDAGError("row payload_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_sha256": self.payload_sha256,
            "payload_bytes": int(self.payload_bytes),
        }


@dataclass(frozen=True, slots=True)
class CausalPageManifest:
    """Exact ordered row identity for one independently addressable page."""

    page_id: str
    model_abi_sha256: str
    payload_sha256: str
    rows: tuple[CausalPageRow, ...]

    def __post_init__(self) -> None:
        if not self.page_id.strip():
            raise CausalPageDAGError("page_id must be non-empty")
        object.__setattr__(
            self,
            "model_abi_sha256",
            _sha256(self.model_abi_sha256, label="model_abi_sha256"),
        )
        object.__setattr__(
            self,
            "payload_sha256",
            _sha256(self.payload_sha256, label="page payload_sha256"),
        )
        if not self.rows:
            raise CausalPageDAGError("page manifest must contain at least one row")

    @property
    def payload_bytes(self) -> int:
        return sum(row.payload_bytes for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "model_abi_sha256": self.model_abi_sha256,
            "payload_sha256": self.payload_sha256,
            "rows": [row.to_dict() for row in self.rows],
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class CausalPageChild:
    page_id: str
    source_payload_sha256: str
    required_ancestor_sha256: str
    delta_rows: tuple[CausalPageRow, ...]

    @property
    def payload_bytes(self) -> int:
        return sum(row.payload_bytes for row in self.delta_rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "source_payload_sha256": self.source_payload_sha256,
            "required_ancestor_sha256": self.required_ancestor_sha256,
            "delta_rows": [row.to_dict() for row in self.delta_rows],
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class CausalPageDAGPlan:
    model_abi_sha256: str
    ancestor_rows: tuple[CausalPageRow, ...]
    ancestor_sha256: str
    children: tuple[CausalPageChild, ...]
    exact_backing_bytes: int
    factored_backing_bytes: int
    fingerprint: str

    @property
    def saved_bytes(self) -> int:
        return self.exact_backing_bytes - self.factored_backing_bytes

    @property
    def saved_fraction(self) -> float:
        return self.saved_bytes / self.exact_backing_bytes

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": CAUSAL_PAGE_DAG_SCHEMA,
            "factorization": "longest-byte-identical-causal-prefix",
            "model_abi_sha256": self.model_abi_sha256,
            "ancestor_rows": [row.to_dict() for row in self.ancestor_rows],
            "ancestor_sha256": self.ancestor_sha256,
            "children": [child.to_dict() for child in self.children],
            "exact_backing_bytes": self.exact_backing_bytes,
            "factored_backing_bytes": self.factored_backing_bytes,
            "saved_bytes": self.saved_bytes,
            "saved_fraction": self.saved_fraction,
        }
        return {**body, "fingerprint": self.fingerprint}


def compile_common_prefix_dag(
    pages: Sequence[CausalPageManifest],
    *,
    minimum_shared_rows: int = 1,
) -> CausalPageDAGPlan:
    """Factor the longest exact common causal prefix from two or more pages."""

    ordered = tuple(pages)
    if len(ordered) < 2:
        raise CausalPageDAGError("common-prefix factoring requires at least two pages")
    if isinstance(minimum_shared_rows, bool) or int(minimum_shared_rows) <= 0:
        raise CausalPageDAGError("minimum_shared_rows must be positive")
    page_ids = [page.page_id for page in ordered]
    if len(page_ids) != len(set(page_ids)):
        raise CausalPageDAGError("page IDs must be unique")
    model_abis = {page.model_abi_sha256 for page in ordered}
    if len(model_abis) != 1:
        raise CausalPageDAGError("causal pages do not share one model ABI")

    shared_count = 0
    maximum = min(len(page.rows) for page in ordered)
    while shared_count < maximum:
        candidate = ordered[0].rows[shared_count]
        if not all(page.rows[shared_count] == candidate for page in ordered[1:]):
            break
        shared_count += 1
    if shared_count < int(minimum_shared_rows):
        raise CausalPageDAGError(
            f"only {shared_count} exact common rows; minimum is {minimum_shared_rows}"
        )
    if any(len(page.rows) == shared_count for page in ordered):
        raise CausalPageDAGError("factoring would leave an empty child delta")

    ancestor_rows = ordered[0].rows[:shared_count]
    ancestor_sha256 = _digest(
        {
            "model_abi_sha256": ordered[0].model_abi_sha256,
            "rows": [row.to_dict() for row in ancestor_rows],
        }
    )
    children = tuple(
        CausalPageChild(
            page_id=page.page_id,
            source_payload_sha256=page.payload_sha256,
            required_ancestor_sha256=ancestor_sha256,
            delta_rows=page.rows[shared_count:],
        )
        for page in ordered
    )
    exact_backing_bytes = sum(page.payload_bytes for page in ordered)
    factored_backing_bytes = sum(row.payload_bytes for row in ancestor_rows) + sum(
        child.payload_bytes for child in children
    )
    body = {
        "schema": CAUSAL_PAGE_DAG_SCHEMA,
        "model_abi_sha256": ordered[0].model_abi_sha256,
        "ancestor_rows": [row.to_dict() for row in ancestor_rows],
        "ancestor_sha256": ancestor_sha256,
        "children": [child.to_dict() for child in children],
        "exact_backing_bytes": exact_backing_bytes,
        "factored_backing_bytes": factored_backing_bytes,
    }
    return CausalPageDAGPlan(
        model_abi_sha256=ordered[0].model_abi_sha256,
        ancestor_rows=ancestor_rows,
        ancestor_sha256=ancestor_sha256,
        children=children,
        exact_backing_bytes=exact_backing_bytes,
        factored_backing_bytes=factored_backing_bytes,
        fingerprint=_digest(body),
    )


def verify_causal_page_dag(
    plan: CausalPageDAGPlan,
    sources: Sequence[CausalPageManifest],
) -> None:
    """Verify a plan fingerprint, dependencies, row closure, and byte accounting."""

    source_by_id = {page.page_id: page for page in sources}
    if len(source_by_id) != len(tuple(sources)):
        raise CausalPageDAGError("source page IDs must be unique")
    if set(source_by_id) != {child.page_id for child in plan.children}:
        raise CausalPageDAGError("plan and source page identities differ")
    if any(page.model_abi_sha256 != plan.model_abi_sha256 for page in source_by_id.values()):
        raise CausalPageDAGError("plan and source model ABIs differ")
    expected_ancestor_sha256 = _digest(
        {
            "model_abi_sha256": plan.model_abi_sha256,
            "rows": [row.to_dict() for row in plan.ancestor_rows],
        }
    )
    if expected_ancestor_sha256 != plan.ancestor_sha256:
        raise CausalPageDAGError("ancestor fingerprint changed")
    for child in plan.children:
        source = source_by_id[child.page_id]
        if child.source_payload_sha256 != source.payload_sha256:
            raise CausalPageDAGError(f"source payload changed for {child.page_id}")
        if child.required_ancestor_sha256 != plan.ancestor_sha256:
            raise CausalPageDAGError(f"ancestor dependency changed for {child.page_id}")
        if (*plan.ancestor_rows, *child.delta_rows) != source.rows:
            raise CausalPageDAGError(f"row closure changed for {child.page_id}")
    exact_bytes = sum(page.payload_bytes for page in source_by_id.values())
    factored_bytes = sum(row.payload_bytes for row in plan.ancestor_rows) + sum(
        child.payload_bytes for child in plan.children
    )
    if exact_bytes != plan.exact_backing_bytes or factored_bytes != plan.factored_backing_bytes:
        raise CausalPageDAGError("page DAG byte accounting changed")
    body = {
        "schema": CAUSAL_PAGE_DAG_SCHEMA,
        "model_abi_sha256": plan.model_abi_sha256,
        "ancestor_rows": [row.to_dict() for row in plan.ancestor_rows],
        "ancestor_sha256": plan.ancestor_sha256,
        "children": [child.to_dict() for child in plan.children],
        "exact_backing_bytes": plan.exact_backing_bytes,
        "factored_backing_bytes": plan.factored_backing_bytes,
    }
    if _digest(body) != plan.fingerprint:
        raise CausalPageDAGError("causal page DAG plan fingerprint changed")


__all__ = [
    "CAUSAL_PAGE_DAG_SCHEMA",
    "CausalPageChild",
    "CausalPageDAGError",
    "CausalPageDAGPlan",
    "CausalPageManifest",
    "CausalPageRow",
    "compile_common_prefix_dag",
    "verify_causal_page_dag",
]

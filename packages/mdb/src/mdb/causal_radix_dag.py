"""Compressed radix-DAG planning for immutable causal page families.

Unlike a single common-prefix factorization, a radix plan can preserve nested
sharing: all pages may share a system prefix, one subgroup may share a policy,
and a smaller subgroup may share an entity scaffold.  Nodes are path-bound to
their exact parent; physical segment payloads remain eligible for ordinary CAS
deduplication when identical bytes appear under different parents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from mdb.causal_page_dag import CausalPageDAGError, CausalPageManifest, CausalPageRow

CAUSAL_RADIX_DAG_SCHEMA = "saturn-causal-radix-dag-plan-v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class CausalRadixSegment:
    """One compressed trie edge bound to its exact parent node."""

    node_sha256: str
    required_parent_sha256: str
    start_row: int
    end_row: int
    rows: tuple[CausalPageRow, ...]
    terminal_page_ids: tuple[str, ...] = ()

    @property
    def payload_bytes(self) -> int:
        return sum(row.payload_bytes for row in self.rows)

    @property
    def payload_identity(self) -> tuple[tuple[str, int], ...]:
        return tuple((row.payload_sha256, row.payload_bytes) for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_sha256": self.node_sha256,
            "required_parent_sha256": self.required_parent_sha256,
            "start_row": self.start_row,
            "end_row": self.end_row,
            "rows": [row.to_dict() for row in self.rows],
            "terminal_page_ids": list(self.terminal_page_ids),
            "payload_bytes": self.payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class CausalRadixPagePath:
    """Ordered node chain reconstructing one exact source page."""

    page_id: str
    source_payload_sha256: str
    node_sha256s: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "source_payload_sha256": self.source_payload_sha256,
            "node_sha256s": list(self.node_sha256s),
        }


@dataclass(frozen=True, slots=True)
class CausalRadixDAGPlan:
    model_abi_sha256: str
    root_sha256: str
    segments: tuple[CausalRadixSegment, ...]
    paths: tuple[CausalRadixPagePath, ...]
    exact_backing_bytes: int
    logical_segment_bytes: int
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
            "schema": CAUSAL_RADIX_DAG_SCHEMA,
            "factorization": "compressed-exact-causal-prefix-radix-dag",
            "model_abi_sha256": self.model_abi_sha256,
            "root_sha256": self.root_sha256,
            "segments": [segment.to_dict() for segment in self.segments],
            "paths": [path.to_dict() for path in self.paths],
            "exact_backing_bytes": self.exact_backing_bytes,
            "logical_segment_bytes": self.logical_segment_bytes,
            "factored_backing_bytes": self.factored_backing_bytes,
            "saved_bytes": self.saved_bytes,
            "saved_fraction": self.saved_fraction,
        }
        return {**body, "fingerprint": self.fingerprint}


@dataclass(slots=True)
class _TrieNode:
    row: CausalPageRow | None
    children: dict[CausalPageRow, _TrieNode] = field(default_factory=dict)
    page_ids: set[str] = field(default_factory=set)
    terminal_page_ids: set[str] = field(default_factory=set)


def compile_causal_radix_dag(
    pages: tuple[CausalPageManifest, ...] | list[CausalPageManifest],
) -> CausalRadixDAGPlan:
    """Compile every exact nested prefix shared by two or more causal pages."""

    ordered = tuple(sorted(pages, key=lambda page: page.page_id))
    if not ordered:
        raise CausalPageDAGError("causal radix factoring requires at least one page")
    page_ids = [page.page_id for page in ordered]
    if len(page_ids) != len(set(page_ids)):
        raise CausalPageDAGError("causal radix page IDs must be unique")
    model_abis = {page.model_abi_sha256 for page in ordered}
    if len(model_abis) != 1:
        raise CausalPageDAGError("causal radix pages do not share one model ABI")
    model_abi = ordered[0].model_abi_sha256
    root_sha256 = _digest(
        {
            "schema": CAUSAL_RADIX_DAG_SCHEMA,
            "model_abi_sha256": model_abi,
            "root": True,
        }
    )
    root = _TrieNode(row=None)
    for page in ordered:
        node = root
        node.page_ids.add(page.page_id)
        for row in page.rows:
            node = node.children.setdefault(row, _TrieNode(row=row))
            node.page_ids.add(page.page_id)
        node.terminal_page_ids.add(page.page_id)

    segments: list[CausalRadixSegment] = []
    path_nodes: dict[str, list[str]] = {page.page_id: [] for page in ordered}

    def walk(first: _TrieNode, *, parent_sha256: str, start_row: int) -> None:
        node = first
        rows: list[CausalPageRow] = []
        while True:
            if node.row is None:  # pragma: no cover - root never enters walk
                raise CausalPageDAGError("causal radix segment contains the virtual root")
            rows.append(node.row)
            if node.terminal_page_ids or len(node.children) != 1:
                break
            node = next(iter(node.children.values()))
        end_row = start_row + len(rows)
        node_body = {
            "schema": CAUSAL_RADIX_DAG_SCHEMA,
            "model_abi_sha256": model_abi,
            "required_parent_sha256": parent_sha256,
            "start_row": start_row,
            "end_row": end_row,
            "rows": [row.to_dict() for row in rows],
        }
        node_sha256 = _digest(node_body)
        segment = CausalRadixSegment(
            node_sha256=node_sha256,
            required_parent_sha256=parent_sha256,
            start_row=start_row,
            end_row=end_row,
            rows=tuple(rows),
            terminal_page_ids=tuple(sorted(node.terminal_page_ids)),
        )
        segments.append(segment)
        for page_id in sorted(node.page_ids):
            path_nodes[page_id].append(node_sha256)
        for child in sorted(
            node.children.values(),
            key=lambda item: (
                item.row.payload_sha256 if item.row is not None else "",
                item.row.payload_bytes if item.row is not None else 0,
            ),
        ):
            walk(child, parent_sha256=node_sha256, start_row=end_row)

    for child in sorted(
        root.children.values(),
        key=lambda item: (
            item.row.payload_sha256 if item.row is not None else "",
            item.row.payload_bytes if item.row is not None else 0,
        ),
    ):
        walk(child, parent_sha256=root_sha256, start_row=0)

    segments_tuple = tuple(sorted(segments, key=lambda segment: segment.node_sha256))
    paths = tuple(
        CausalRadixPagePath(
            page_id=page.page_id,
            source_payload_sha256=page.payload_sha256,
            node_sha256s=tuple(path_nodes[page.page_id]),
        )
        for page in ordered
    )
    exact_backing_bytes = sum(page.payload_bytes for page in ordered)
    logical_segment_bytes = sum(segment.payload_bytes for segment in segments_tuple)
    unique_payloads: dict[tuple[tuple[str, int], ...], int] = {}
    for segment in segments_tuple:
        unique_payloads.setdefault(segment.payload_identity, segment.payload_bytes)
    factored_backing_bytes = sum(unique_payloads.values())
    body = {
        "schema": CAUSAL_RADIX_DAG_SCHEMA,
        "model_abi_sha256": model_abi,
        "root_sha256": root_sha256,
        "segments": [segment.to_dict() for segment in segments_tuple],
        "paths": [path.to_dict() for path in paths],
        "exact_backing_bytes": exact_backing_bytes,
        "logical_segment_bytes": logical_segment_bytes,
        "factored_backing_bytes": factored_backing_bytes,
    }
    plan = CausalRadixDAGPlan(
        model_abi_sha256=model_abi,
        root_sha256=root_sha256,
        segments=segments_tuple,
        paths=paths,
        exact_backing_bytes=exact_backing_bytes,
        logical_segment_bytes=logical_segment_bytes,
        factored_backing_bytes=factored_backing_bytes,
        fingerprint=_digest(body),
    )
    verify_causal_radix_dag(plan, ordered, recompile=False)
    return plan


def verify_causal_radix_dag(
    plan: CausalRadixDAGPlan,
    sources: tuple[CausalPageManifest, ...] | list[CausalPageManifest],
    *,
    recompile: bool = True,
) -> None:
    """Verify path closure, ancestry, accounting, fingerprint, and determinism."""

    ordered = tuple(sorted(sources, key=lambda page: page.page_id))
    source_by_id = {page.page_id: page for page in ordered}
    if len(source_by_id) != len(ordered):
        raise CausalPageDAGError("causal radix source page IDs must be unique")
    if set(source_by_id) != {path.page_id for path in plan.paths}:
        raise CausalPageDAGError("causal radix plan and source page identities differ")
    if any(page.model_abi_sha256 != plan.model_abi_sha256 for page in ordered):
        raise CausalPageDAGError("causal radix model ABI changed")
    expected_root = _digest(
        {
            "schema": CAUSAL_RADIX_DAG_SCHEMA,
            "model_abi_sha256": plan.model_abi_sha256,
            "root": True,
        }
    )
    if expected_root != plan.root_sha256:
        raise CausalPageDAGError("causal radix root fingerprint changed")
    segment_by_id = {segment.node_sha256: segment for segment in plan.segments}
    if len(segment_by_id) != len(plan.segments):
        raise CausalPageDAGError("causal radix node identities are not unique")
    for segment in plan.segments:
        expected_node = _digest(
            {
                "schema": CAUSAL_RADIX_DAG_SCHEMA,
                "model_abi_sha256": plan.model_abi_sha256,
                "required_parent_sha256": segment.required_parent_sha256,
                "start_row": segment.start_row,
                "end_row": segment.end_row,
                "rows": [row.to_dict() for row in segment.rows],
            }
        )
        if expected_node != segment.node_sha256:
            raise CausalPageDAGError("causal radix node fingerprint changed")
        if segment.end_row - segment.start_row != len(segment.rows):
            raise CausalPageDAGError("causal radix segment row bounds changed")
        if not segment.rows:
            raise CausalPageDAGError("causal radix segment is empty")
    for path in plan.paths:
        source = source_by_id[path.page_id]
        if path.source_payload_sha256 != source.payload_sha256:
            raise CausalPageDAGError(f"source payload changed for {path.page_id}")
        parent = plan.root_sha256
        reconstructed: list[CausalPageRow] = []
        for node_sha256 in path.node_sha256s:
            segment = segment_by_id.get(node_sha256)
            if segment is None:
                raise CausalPageDAGError(f"missing causal radix node for {path.page_id}")
            if segment.required_parent_sha256 != parent:
                raise CausalPageDAGError(f"causal radix ancestry changed for {path.page_id}")
            if segment.start_row != len(reconstructed):
                raise CausalPageDAGError(f"causal radix row order changed for {path.page_id}")
            reconstructed.extend(segment.rows)
            parent = segment.node_sha256
        if tuple(reconstructed) != source.rows:
            raise CausalPageDAGError(f"causal radix row closure changed for {path.page_id}")
        if not path.node_sha256s:
            raise CausalPageDAGError(f"causal radix path is empty for {path.page_id}")
        if path.page_id not in segment_by_id[path.node_sha256s[-1]].terminal_page_ids:
            raise CausalPageDAGError(f"causal radix terminal changed for {path.page_id}")
    exact_bytes = sum(page.payload_bytes for page in ordered)
    logical_bytes = sum(segment.payload_bytes for segment in plan.segments)
    unique_payload_bytes = {
        segment.payload_identity: segment.payload_bytes for segment in plan.segments
    }
    factored_bytes = sum(unique_payload_bytes.values())
    if (
        exact_bytes != plan.exact_backing_bytes
        or logical_bytes != plan.logical_segment_bytes
        or factored_bytes != plan.factored_backing_bytes
    ):
        raise CausalPageDAGError("causal radix byte accounting changed")
    body = {
        "schema": CAUSAL_RADIX_DAG_SCHEMA,
        "model_abi_sha256": plan.model_abi_sha256,
        "root_sha256": plan.root_sha256,
        "segments": [segment.to_dict() for segment in plan.segments],
        "paths": [path.to_dict() for path in plan.paths],
        "exact_backing_bytes": plan.exact_backing_bytes,
        "logical_segment_bytes": plan.logical_segment_bytes,
        "factored_backing_bytes": plan.factored_backing_bytes,
    }
    if _digest(body) != plan.fingerprint:
        raise CausalPageDAGError("causal radix plan fingerprint changed")
    if recompile:
        expected = compile_causal_radix_dag(ordered)
        if expected.to_dict() != plan.to_dict():
            raise CausalPageDAGError("causal radix plan is not the deterministic factorization")


__all__ = [
    "CAUSAL_RADIX_DAG_SCHEMA",
    "CausalRadixDAGPlan",
    "CausalRadixPagePath",
    "CausalRadixSegment",
    "compile_causal_radix_dag",
    "verify_causal_radix_dag",
]

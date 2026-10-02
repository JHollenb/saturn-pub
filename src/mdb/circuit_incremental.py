"""Exactness-first dependency-cone planning for discovered circuit graphs.

This module is the build-system half of the generalized circuit router.  It
does not assume that a measured route is complete.  A full native execution
publishes content-addressed cell results.  A later edit invalidates the cells
that own the edited address and every downstream dependent cell.  Reuse is
execution-authorized only after an independent full-build parity receipt has
certified this exact graph/specimen/numerical contract.

If incremental and full native outputs disagree, the plan becomes a skeleton
falsifier and names the reused frontier that may hide a missing edge.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mdb.execution_skeleton import ExecutionSkeleton, content_digest

CIRCUIT_CELL_SCHEMA = "saturn-circuit-cell-v1"
CIRCUIT_BUILD_GRAPH_SCHEMA = "saturn-circuit-build-graph-v2"
CIRCUIT_CELL_RESULT_SCHEMA = "saturn-circuit-cell-result-v1"
CIRCUIT_BUILD_STATE_SCHEMA = "saturn-circuit-build-state-v1"
CIRCUIT_INCREMENTAL_PLAN_SCHEMA = "saturn-circuit-incremental-plan-v2"
CIRCUIT_PARITY_RECEIPT_SCHEMA = "saturn-circuit-incremental-parity-v2"
CIRCUIT_CERTIFICATION_SCOPE_SCHEMA = "saturn-circuit-certification-scope-v2"
CIRCUIT_INCREMENTAL_PLAN_V3_SCHEMA = "saturn-circuit-incremental-plan-v3"
CIRCUIT_PARITY_RECEIPT_V3_SCHEMA = "saturn-circuit-incremental-parity-v3"
CIRCUIT_CERTIFICATION_SCOPE_V3_SCHEMA = "saturn-circuit-certification-scope-v3"
CIRCUIT_INCREMENTAL_PLAN_V4_SCHEMA = "saturn-circuit-incremental-plan-v4"
CIRCUIT_PARITY_RECEIPT_V4_SCHEMA = "saturn-circuit-incremental-parity-v4"
CIRCUIT_CERTIFICATION_SCOPE_V4_SCHEMA = "saturn-circuit-certification-scope-v4"
CIRCUIT_CANDIDATE_REQUEST_SCHEMA = "saturn-circuit-candidate-request-v1"
CIRCUIT_EXECUTABLE_BINDING_SCHEMA = "saturn-circuit-executable-binding-v1"
CIRCUIT_NATIVE_CONSUMER_BINDING_SCHEMA = "saturn-circuit-native-consumer-binding-v1"
CIRCUIT_EXECUTABLE_MANIFEST_SCHEMA = "saturn-circuit-executable-manifest-v1"


class CircuitIncrementalError(ValueError):
    """Raised when incremental reuse cannot preserve its exact contract."""


@dataclass(frozen=True, slots=True)
class CircuitExecutableBinding:
    """Build-custody identity for one executor entrypoint."""

    executor_key: str
    artifact_sha256: str
    artifact_ref: str
    entrypoint: str
    abi: str
    runtime_lane: str
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_EXECUTABLE_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "executor_key", _text(self.executor_key, "executor key"))
        object.__setattr__(
            self, "artifact_sha256", _sha(self.artifact_sha256, "executable artifact digest")
        )
        for name in ("artifact_ref", "entrypoint", "abi", "runtime_lane"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "executor_key": self.executor_key,
            "artifact_sha256": self.artifact_sha256,
            "artifact_ref": self.artifact_ref,
            "entrypoint": self.entrypoint,
            "abi": self.abi,
            "runtime_lane": self.runtime_lane,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitExecutableBinding:
        result = cls(
            executor_key=value.get("executor_key"),
            artifact_sha256=value.get("artifact_sha256"),
            artifact_ref=value.get("artifact_ref"),
            entrypoint=value.get("entrypoint"),
            abi=value.get("abi"),
            runtime_lane=value.get("runtime_lane"),
        )
        if value.get("schema") != cls.schema or value.get("fingerprint") != result.fingerprint:
            raise CircuitIncrementalError("executable binding schema or fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CircuitNativeConsumerBinding:
    """Build-custody identity for the native output consumer."""

    artifact_sha256: str
    artifact_ref: str
    entrypoint: str
    abi: str
    runtime_lane: str
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_NATIVE_CONSUMER_BINDING_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_sha256", _sha(self.artifact_sha256, "consumer artifact digest")
        )
        for name in ("artifact_ref", "entrypoint", "abi", "runtime_lane"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_sha256": self.artifact_sha256,
            "artifact_ref": self.artifact_ref,
            "entrypoint": self.entrypoint,
            "abi": self.abi,
            "runtime_lane": self.runtime_lane,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitNativeConsumerBinding:
        result = cls(
            artifact_sha256=value.get("artifact_sha256"),
            artifact_ref=value.get("artifact_ref"),
            entrypoint=value.get("entrypoint"),
            abi=value.get("abi"),
            runtime_lane=value.get("runtime_lane"),
        )
        if value.get("schema") != cls.schema or value.get("fingerprint") != result.fingerprint:
            raise CircuitIncrementalError("native consumer binding schema or fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CircuitExecutableManifest:
    """Immutable ordered executable bundle supplied by build/deploy custody."""

    executors: tuple[CircuitExecutableBinding, ...]
    native_consumer: CircuitNativeConsumerBinding
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_EXECUTABLE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        executors = tuple(self.executors)
        if not executors or any(not isinstance(row, CircuitExecutableBinding) for row in executors):
            raise CircuitIncrementalError("executable manifest requires executor bindings")
        keys = [row.executor_key for row in executors]
        if len(keys) != len(set(keys)):
            raise CircuitIncrementalError("executable manifest executor keys must be unique")
        if not isinstance(self.native_consumer, CircuitNativeConsumerBinding):
            raise CircuitIncrementalError("executable manifest requires a native consumer binding")
        object.__setattr__(self, "executors", executors)
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "executors": [row.to_dict() for row in self.executors],
            "native_consumer": self.native_consumer.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitExecutableManifest:
        result = cls(
            executors=tuple(
                CircuitExecutableBinding.from_dict(row) for row in value.get("executors", ())
            ),
            native_consumer=CircuitNativeConsumerBinding.from_dict(
                value.get("native_consumer", {})
            ),
        )
        if value.get("schema") != cls.schema or value.get("fingerprint") != result.fingerprint:
            raise CircuitIncrementalError("executable manifest schema or fingerprint mismatch")
        return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CircuitIncrementalError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    result = _text(value, label).casefold()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise CircuitIncrementalError(f"{label} must be a SHA-256 digest")
    return result


def _strings(value: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(_text(item, label) for item in value)
    if len(result) != len(set(result)):
        raise CircuitIncrementalError(f"{label} contains duplicates")
    return tuple(sorted(result))


def _json(value: Any, label: str) -> Any:
    def finite(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            raise CircuitIncrementalError(f"{label} contains a non-finite number")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise CircuitIncrementalError(f"{label} has a non-string key")
                finite(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                finite(child)

    finite(value)
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise CircuitIncrementalError(f"{label} must be finite JSON data") from exc


@dataclass(frozen=True, slots=True)
class CircuitCell:
    cell_id: str
    role: str
    dependencies: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    physical_forward_units: int = 0
    coordinate: Mapping[str, Any] = field(default_factory=dict)
    executor_key: str = "native"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_CELL_SCHEMA

    def __post_init__(self) -> None:
        for name in ("cell_id", "role", "executor_key"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("dependencies", "reads", "writes", "invalidates"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if (
            isinstance(self.physical_forward_units, bool)
            or not isinstance(self.physical_forward_units, int)
            or self.physical_forward_units < 0
        ):
            raise CircuitIncrementalError("physical_forward_units must be a non-negative integer")
        object.__setattr__(self, "coordinate", _json(dict(self.coordinate), "cell coordinate"))
        object.__setattr__(self, "metadata", _json(dict(self.metadata), "cell metadata"))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cell_id": self.cell_id,
            "role": self.role,
            "dependencies": list(self.dependencies),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "physical_forward_units": self.physical_forward_units,
            "coordinate": dict(self.coordinate),
            "executor_key": self.executor_key,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitCell:
        result = cls(
            cell_id=value.get("cell_id"),
            role=value.get("role"),
            dependencies=tuple(value.get("dependencies", ())),
            reads=tuple(value.get("reads", ())),
            writes=tuple(value.get("writes", ())),
            invalidates=tuple(value.get("invalidates", ())),
            physical_forward_units=value.get("physical_forward_units", 0),
            coordinate=value.get("coordinate", {}),
            executor_key=value.get("executor_key", "native"),
            metadata=value.get("metadata", {}),
        )
        if (
            value.get("schema") not in (None, cls.schema)
            or value.get("fingerprint", result.fingerprint) != result.fingerprint
        ):
            raise CircuitIncrementalError("circuit cell schema or fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CircuitBuildGraph:
    graph_id: str
    skeleton_fingerprint: str
    specimen_fingerprint: str
    cells: tuple[CircuitCell, ...]
    outputs: tuple[str, ...]
    exact_parity_receipts: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    numerical_contract: str = "exact"
    native_consumer_contract: str = "native-output-sha256-exact-v1"
    native_consumer_identity: str | None = None
    executable_manifest: CircuitExecutableManifest | None = None
    resource_provider_fingerprint: str | None = None
    certification_subject_fingerprint: str = field(init=False)
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_BUILD_GRAPH_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _text(self.graph_id, "graph_id"))
        object.__setattr__(
            self, "skeleton_fingerprint", _sha(self.skeleton_fingerprint, "skeleton fingerprint")
        )
        object.__setattr__(
            self, "specimen_fingerprint", _sha(self.specimen_fingerprint, "specimen fingerprint")
        )
        cells = tuple(self.cells)
        if not cells or any(not isinstance(cell, CircuitCell) for cell in cells):
            raise CircuitIncrementalError("build graph requires CircuitCell entries")
        by_id = {cell.cell_id: cell for cell in cells}
        if len(by_id) != len(cells):
            raise CircuitIncrementalError("build graph cell IDs must be unique")
        for cell in cells:
            missing = set(cell.dependencies) - set(by_id)
            if missing:
                raise CircuitIncrementalError(
                    f"cell {cell.cell_id!r} has missing dependencies {sorted(missing)}"
                )
        outputs = _strings(self.outputs, "graph output")
        if not outputs or set(outputs) - set(by_id):
            raise CircuitIncrementalError("build graph outputs must name existing cells")
        # Kahn validation doubles as deterministic topological ordering.
        indegree = {cell.cell_id: len(cell.dependencies) for cell in cells}
        children = {cell.cell_id: [] for cell in cells}
        for cell in cells:
            for parent in cell.dependencies:
                children[parent].append(cell.cell_id)
        queue = deque(sorted(key for key, degree in indegree.items() if degree == 0))
        ordered = []
        while queue:
            node = queue.popleft()
            ordered.append(by_id[node])
            for child in sorted(children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(cells):
            raise CircuitIncrementalError("build graph must be acyclic")
        object.__setattr__(self, "cells", tuple(ordered))
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "metadata", _json(dict(self.metadata), "graph metadata"))
        object.__setattr__(
            self, "numerical_contract", _text(self.numerical_contract, "numerical contract")
        )
        # ``exact`` here describes equality between the fresh full and incremental
        # executions *within this named implementation lane*.  A lane such as
        # ``int8_weight_only`` may be approximate relative to a source checkpoint
        # while still supporting byte-exact incremental certification against a
        # fresh full execution through that same lane.
        object.__setattr__(
            self,
            "native_consumer_contract",
            _text(self.native_consumer_contract, "native consumer contract"),
        )
        consumer_identity = self.native_consumer_identity
        if consumer_identity is None:
            consumer_identity = content_digest(
                {
                    "schema": "saturn-derived-native-consumer-identity-v1",
                    "outputs": [by_id[cell_id].to_dict() for cell_id in outputs],
                }
            )
        object.__setattr__(
            self,
            "native_consumer_identity",
            _sha(consumer_identity, "native consumer identity"),
        )
        manifest = self.executable_manifest
        if manifest is not None:
            if not isinstance(manifest, CircuitExecutableManifest):
                raise CircuitIncrementalError("executable_manifest must be content-addressed")
            required = {cell.executor_key for cell in cells}
            declared = {row.executor_key for row in manifest.executors}
            if required != declared:
                raise CircuitIncrementalError(
                    "executable manifest must bind exactly the graph executor keys"
                )
        provider_fingerprint = self.resource_provider_fingerprint
        if provider_fingerprint is not None:
            object.__setattr__(
                self,
                "resource_provider_fingerprint",
                _sha(provider_fingerprint, "resource provider fingerprint"),
            )
        receipts = tuple(self.exact_parity_receipts)
        normalized_receipts: list[Mapping[str, Any]] = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                # V1 accepted any bare digest as authority. Preserve the ability
                # to read that intent, but deliberately drop its authority.
                legacy_digest = _sha(receipt, "legacy parity receipt")
                normalized_receipts.append({"legacy_digest": legacy_digest})
                continue
            normalized_receipts.append(_json(dict(receipt), "parity receipt"))
        object.__setattr__(self, "exact_parity_receipts", tuple(normalized_receipts))
        subject = content_digest(self._subject_body())
        object.__setattr__(self, "certification_subject_fingerprint", subject)
        # A receipt is an attestation about the graph, not part of graph identity.
        # Keeping this alias stable allows a pre-certification build state to be
        # replayed after its receipt is attached.
        object.__setattr__(self, "fingerprint", subject)

    def _subject_body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph_id": self.graph_id,
            "skeleton_fingerprint": self.skeleton_fingerprint,
            "specimen_fingerprint": self.specimen_fingerprint,
            "cells": [cell.to_dict() for cell in self.cells],
            "outputs": list(self.outputs),
            "metadata": dict(self.metadata),
            "numerical_contract": self.numerical_contract,
            "native_consumer_contract": self.native_consumer_contract,
            "native_consumer_identity": self.native_consumer_identity,
            **(
                {"executable_manifest": self.executable_manifest.to_dict()}
                if self.executable_manifest is not None
                else {}
            ),
            # This is the portable, content-addressed identity declared by the
            # physical provider. Run-local paths and lifecycle receipts remain
            # outside graph identity and do not widen this authority boundary.
            **(
                {"resource_provider_fingerprint": self.resource_provider_fingerprint}
                if self.resource_provider_fingerprint is not None
                else {}
            ),
        }

    def _body(self) -> dict[str, Any]:
        return {
            **self._subject_body(),
            "exact_parity_receipts": [dict(receipt) for receipt in self.exact_parity_receipts],
            "certification_subject_fingerprint": self.certification_subject_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitBuildGraph:
        result = cls(
            graph_id=value.get("graph_id"),
            skeleton_fingerprint=value.get("skeleton_fingerprint"),
            specimen_fingerprint=value.get("specimen_fingerprint"),
            cells=tuple(CircuitCell.from_dict(row) for row in value.get("cells", ())),
            outputs=tuple(value.get("outputs", ())),
            exact_parity_receipts=tuple(value.get("exact_parity_receipts", ())),
            metadata=value.get("metadata", {}),
            numerical_contract=value.get("numerical_contract", "exact"),
            native_consumer_contract=value.get(
                "native_consumer_contract", "native-output-sha256-exact-v1"
            ),
            native_consumer_identity=value.get("native_consumer_identity"),
            executable_manifest=(
                CircuitExecutableManifest.from_dict(value["executable_manifest"])
                if "executable_manifest" in value
                else None
            ),
            resource_provider_fingerprint=value.get("resource_provider_fingerprint"),
        )
        if (
            value.get("schema") != cls.schema
            or value.get("fingerprint") != result.fingerprint
            or value.get(
                "certification_subject_fingerprint", result.certification_subject_fingerprint
            )
            != result.certification_subject_fingerprint
        ):
            raise CircuitIncrementalError("build graph schema or fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CircuitCellResult:
    cell_id: str
    cell_fingerprint: str
    input_fingerprint: str
    output_sha256: str
    artifact_ref: str
    numerical_contract: str = "exact"
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_CELL_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_id", _text(self.cell_id, "cell_id"))
        for name in ("cell_fingerprint", "input_fingerprint", "output_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "artifact_ref", _text(self.artifact_ref, "artifact_ref"))
        object.__setattr__(
            self, "numerical_contract", _text(self.numerical_contract, "numerical_contract")
        )
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cell_id": self.cell_id,
            "cell_fingerprint": self.cell_fingerprint,
            "input_fingerprint": self.input_fingerprint,
            "output_sha256": self.output_sha256,
            "artifact_ref": self.artifact_ref,
            "numerical_contract": self.numerical_contract,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitCellResult:
        result = cls(
            cell_id=value.get("cell_id"),
            cell_fingerprint=value.get("cell_fingerprint"),
            input_fingerprint=value.get("input_fingerprint"),
            output_sha256=value.get("output_sha256"),
            artifact_ref=value.get("artifact_ref"),
            numerical_contract=value.get("numerical_contract", "exact"),
        )
        if (
            value.get("schema") not in (None, cls.schema)
            or value.get("fingerprint", result.fingerprint) != result.fingerprint
        ):
            raise CircuitIncrementalError("cell result schema or fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class CircuitBuildState:
    graph_fingerprint: str
    support_fingerprint: str
    results: tuple[CircuitCellResult, ...]
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_BUILD_STATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "graph_fingerprint", _sha(self.graph_fingerprint, "graph fingerprint")
        )
        object.__setattr__(
            self, "support_fingerprint", _sha(self.support_fingerprint, "support fingerprint")
        )
        results = tuple(self.results)
        ids = [row.cell_id for row in results]
        if len(ids) != len(set(ids)):
            raise CircuitIncrementalError("build state contains duplicate cell results")
        object.__setattr__(self, "results", tuple(sorted(results, key=lambda row: row.cell_id)))
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph_fingerprint": self.graph_fingerprint,
            "support_fingerprint": self.support_fingerprint,
            "results": [row.to_dict() for row in self.results],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CircuitBuildState:
        result = cls(
            graph_fingerprint=value.get("graph_fingerprint"),
            support_fingerprint=value.get("support_fingerprint"),
            results=tuple(CircuitCellResult.from_dict(row) for row in value.get("results", ())),
        )
        if value.get("schema") != cls.schema or value.get("fingerprint") != result.fingerprint:
            raise CircuitIncrementalError("build state schema or fingerprint mismatch")
        return result


def graph_from_skeleton(
    skeleton: ExecutionSkeleton, *, forward_roles: Sequence[str] = ("writer", "consumer")
) -> CircuitBuildGraph:
    """Lower an observed skeleton to a conservative cell graph.

    Every observed edge becomes a dependency. Nodes whose kind/role occurs in
    ``forward_roles`` receive one physical forward unit; callers can replace
    this coarse accounting with a family-specific graph after decompilation.
    """
    dependencies: dict[str, list[str]] = {node.node_id: [] for node in skeleton.nodes}
    for edge in skeleton.edges:
        dependencies[edge.target_node_id].append(edge.source_node_id)
    sources = {edge.source_node_id for edge in skeleton.edges}
    outputs = tuple(node.node_id for node in skeleton.nodes if node.node_id not in sources)
    cells = tuple(
        CircuitCell(
            cell_id=node.node_id,
            role=node.kind,
            dependencies=tuple(dependencies[node.node_id]),
            reads=tuple(str(item) for item in node.abi.get("reads", ())),
            writes=tuple(str(item) for item in node.abi.get("writes", ())),
            invalidates=tuple(str(item) for item in node.metadata.get("invalidates", ())),
            physical_forward_units=1 if node.kind in set(forward_roles) else 0,
            coordinate=node.coordinate,
            executor_key=f"native:{node.kind}",
        )
        for node in skeleton.nodes
    )
    return CircuitBuildGraph(
        graph_id=f"{skeleton.skeleton_id}-build-graph",
        skeleton_fingerprint=skeleton.fingerprint,
        specimen_fingerprint=skeleton.specimen.fingerprint,
        cells=cells,
        outputs=outputs or (skeleton.nodes[-1].node_id,),
        metadata={
            "lowering": "conservative-one-cell-per-observed-node",
            "parity_status": "unproved",
        },
    )


def _certification_scope(
    graph: CircuitBuildGraph,
    previous: CircuitBuildState,
    *,
    changed_cells: Sequence[str],
    changed_addresses: Sequence[str],
    recompute_cells: Sequence[str],
    reused_cells: Sequence[str],
    missing_cell_results: Sequence[str],
    candidate_request_fingerprint: str,
) -> dict[str, Any]:
    """Return the immutable subject of one exact incremental certification."""

    manifest = graph.executable_manifest
    authority_refusals: list[str] = []
    if manifest is None:
        authority_refusals.append("executable-manifest-required")
    if (
        any("weight_set" in cell.metadata for cell in graph.cells)
        and graph.resource_provider_fingerprint is None
    ):
        authority_refusals.append("resource-provider-fingerprint-required")
    return {
        "schema": CIRCUIT_CERTIFICATION_SCOPE_V4_SCHEMA,
        "certification_subject_fingerprint": graph.certification_subject_fingerprint,
        "specimen_fingerprint": graph.specimen_fingerprint,
        "numerical_contract": graph.numerical_contract,
        "native_consumer_contract": graph.native_consumer_contract,
        "native_consumer_identity": graph.native_consumer_identity,
        **(
            {
                "executable_manifest_fingerprint": manifest.fingerprint,
                "resolved_executable_bindings": [
                    row.to_dict() for row in manifest.executors
                ],
                "resolved_native_consumer_binding": manifest.native_consumer.to_dict(),
            }
            if manifest is not None
            else {}
        ),
        **(
            {"resource_provider_fingerprint": graph.resource_provider_fingerprint}
            if graph.resource_provider_fingerprint is not None
            else {}
        ),
        "previous_state_fingerprint": previous.fingerprint,
        "candidate_request_fingerprint": candidate_request_fingerprint,
        "changed_cells": sorted(set(changed_cells)),
        "changed_addresses": sorted(set(changed_addresses)),
        "recompute_cells": list(recompute_cells),
        "reused_cells": list(reused_cells),
        "missing_cell_results": sorted(set(missing_cell_results)),
        **(
            {"authority_refusals": authority_refusals}
            if authority_refusals
            else {}
        ),
    }


def _receipt_authorizes_scope(receipt: Mapping[str, Any], scope: Mapping[str, Any]) -> bool:
    """Fail-closed validation for a same-generation content-addressed receipt."""

    try:
        body = {key: value for key, value in receipt.items() if key != "fingerprint"}
        if (
            scope.get("schema") != CIRCUIT_CERTIFICATION_SCOPE_V4_SCHEMA
            or receipt.get("schema") != CIRCUIT_PARITY_RECEIPT_V4_SCHEMA
        ):
            return False
        if receipt.get("fingerprint") != content_digest(body):
            return False
        if receipt.get("certification_scope") != scope:
            return False
        if receipt.get("certification_scope_fingerprint") != content_digest(scope):
            return False
        # Historical unsealed receipts remain readable but can never authorize
        # v4 incremental execution. Conditional refusal markers also preserve
        # byte identity for fully bound graphs while failing unsafe graphs shut.
        if (
            scope.get("executable_manifest_fingerprint") is None
            or scope.get("authority_refusals")
        ):
            return False
        bound_fields = (
            "certification_subject_fingerprint",
            "specimen_fingerprint",
            "numerical_contract",
            "native_consumer_contract",
            "native_consumer_identity",
            "candidate_request_fingerprint",
            *(
                (
                    "executable_manifest_fingerprint",
                    "resolved_executable_bindings",
                    "resolved_native_consumer_binding",
                )
                if scope.get("executable_manifest_fingerprint") is not None
                else ()
            ),
            *(
                ("resource_provider_fingerprint",)
                if scope.get("resource_provider_fingerprint") is not None
                else ()
            ),
        )
        if any(receipt.get(key) != scope.get(key) for key in bound_fields):
            return False
        full = _sha(receipt.get("full_output_sha256"), "full output hash")
        incremental = _sha(receipt.get("incremental_output_sha256"), "incremental output hash")
        if scope.get("executable_manifest_fingerprint") is not None:
            _sha(
                receipt.get("full_execution_receipt_fingerprint"),
                "full execution receipt fingerprint",
            )
            _sha(
                receipt.get("incremental_execution_receipt_fingerprint"),
                "incremental execution receipt fingerprint",
            )
            _sha(
                receipt.get("full_executable_bundle_fingerprint"),
                "full executable bundle fingerprint",
            )
            _sha(
                receipt.get("incremental_executable_bundle_fingerprint"),
                "incremental executable bundle fingerprint",
            )
        full_state = _sha(
            receipt.get("full_result_state_fingerprint"),
            "full result state fingerprint",
        )
        incremental_state = _sha(
            receipt.get("incremental_result_state_fingerprint"),
            "incremental result state fingerprint",
        )
        return (
            full == incremental
            and full_state == incremental_state
            and receipt.get("exact") is True
            and receipt.get("status") == "certified"
            and receipt.get("authority") == {"admit_plan_scope": True, "widen_tolerance": False}
        )
    except (CircuitIncrementalError, TypeError, ValueError):
        return False


def plan_incremental_rebuild(
    graph: CircuitBuildGraph,
    previous: CircuitBuildState,
    *,
    changed_cells: Sequence[str] = (),
    changed_addresses: Sequence[str] = (),
    candidate_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if previous.graph_fingerprint != graph.fingerprint:
        raise CircuitIncrementalError("previous state belongs to another build graph")
    by_id = {cell.cell_id: cell for cell in graph.cells}
    results = {row.cell_id: row for row in previous.results}
    missing_results = set(by_id) - set(results)
    requested_changed_cells = _strings(tuple(changed_cells), "changed cell")
    dirty = set(requested_changed_cells)
    if dirty - set(by_id):
        raise CircuitIncrementalError("changed_cells names an unknown cell")
    addresses = set(_strings(tuple(changed_addresses), "changed address"))
    normalized_candidate_inputs = _json(
        dict(candidate_inputs or {}), "candidate inputs"
    )
    unknown_candidate_inputs = set(normalized_candidate_inputs) - set(by_id)
    if unknown_candidate_inputs:
        raise CircuitIncrementalError(
            f"candidate_inputs names unknown cells {sorted(unknown_candidate_inputs)}"
        )
    candidate_request = {
        "schema": CIRCUIT_CANDIDATE_REQUEST_SCHEMA,
        "graph_fingerprint": graph.fingerprint,
        "previous_state_fingerprint": previous.fingerprint,
        "changed_cells": list(requested_changed_cells),
        "changed_addresses": sorted(addresses),
        "cell_inputs": normalized_candidate_inputs,
    }
    candidate_request_fingerprint = content_digest(candidate_request)
    for cell in graph.cells:
        if addresses.intersection(set(cell.reads) | set(cell.writes) | set(cell.invalidates)):
            dirty.add(cell.cell_id)
    dirty.update(missing_results)
    children = {cell.cell_id: [] for cell in graph.cells}
    for cell in graph.cells:
        for parent in cell.dependencies:
            children[parent].append(cell.cell_id)
    queue = deque(sorted(dirty))
    while queue:
        parent = queue.popleft()
        for child in children[parent]:
            if child not in dirty:
                dirty.add(child)
                queue.append(child)
    recompute = tuple(cell.cell_id for cell in graph.cells if cell.cell_id in dirty)
    reused = tuple(cell.cell_id for cell in graph.cells if cell.cell_id not in dirty)
    baseline_units = sum(cell.physical_forward_units for cell in graph.cells)
    planned_units = sum(
        cell.physical_forward_units for cell in graph.cells if cell.cell_id in dirty
    )
    scope = _certification_scope(
        graph,
        previous,
        changed_cells=tuple(changed_cells),
        changed_addresses=tuple(addresses),
        recompute_cells=recompute,
        reused_cells=reused,
        missing_cell_results=tuple(missing_results),
        candidate_request_fingerprint=candidate_request_fingerprint,
    )
    scope_fingerprint = content_digest(scope)
    matching_receipts = [
        receipt["fingerprint"]
        for receipt in graph.exact_parity_receipts
        if _receipt_authorizes_scope(receipt, scope)
    ]
    authority = bool(matching_receipts) and not missing_results
    body = {
        "schema": CIRCUIT_INCREMENTAL_PLAN_V4_SCHEMA,
        "graph_fingerprint": graph.fingerprint,
        "certification_subject_fingerprint": graph.certification_subject_fingerprint,
        "specimen_fingerprint": graph.specimen_fingerprint,
        "numerical_contract": graph.numerical_contract,
        "native_consumer_contract": graph.native_consumer_contract,
        "native_consumer_identity": graph.native_consumer_identity,
        **(
            {
                "executable_manifest_fingerprint": graph.executable_manifest.fingerprint,
                "resolved_executable_bindings": [
                    row.to_dict() for row in graph.executable_manifest.executors
                ],
                "resolved_native_consumer_binding": (
                    graph.executable_manifest.native_consumer.to_dict()
                ),
            }
            if graph.executable_manifest is not None
            else {}
        ),
        **(
            {"resource_provider_fingerprint": graph.resource_provider_fingerprint}
            if graph.resource_provider_fingerprint is not None
            else {}
        ),
        "previous_state_fingerprint": previous.fingerprint,
        "candidate_request": candidate_request,
        "candidate_request_fingerprint": candidate_request_fingerprint,
        "changed_cells": sorted(set(changed_cells)),
        "changed_addresses": sorted(addresses),
        "recompute_cells": list(recompute),
        "reused_cells": list(reused),
        "missing_cell_results": sorted(missing_results),
        "forward_accounting": {
            "baseline_units": baseline_units,
            "planned_units": planned_units,
            "avoided_units": baseline_units - planned_units,
        },
        "execution_authorized": authority,
        "authorizing_receipts": sorted(matching_receipts),
        "reason": "exact-parity-certified"
        if authority
        else "discovery-only-until-full-build-parity",
        "native_consumer_required": True,
        "certification_scope": scope,
        "certification_scope_fingerprint": scope_fingerprint,
    }
    return {**body, "fingerprint": content_digest(body)}


def _validate_v4_execution_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    plan: Mapping[str, Any],
    output_sha256: str,
    label: str,
) -> tuple[str, str | None, str]:
    if not isinstance(receipt, Mapping):
        raise CircuitIncrementalError(f"v4 parity requires the {label} execution receipt")
    body = {key: value for key, value in receipt.items() if key != "fingerprint"}
    fingerprint = receipt.get("fingerprint")
    if (
        receipt.get("schema") != "saturn-circuit-execution-receipt-v4"
        or fingerprint != content_digest(body)
    ):
        raise CircuitIncrementalError(f"{label} execution receipt is not content-addressed v4")
    if receipt.get("output_sha256") != output_sha256:
        raise CircuitIncrementalError(f"{label} execution receipt output mismatch")
    if receipt.get("candidate_request_fingerprint") != plan.get(
        "candidate_request_fingerprint"
    ):
        raise CircuitIncrementalError(f"{label} candidate request binding mismatch")
    result_state = _sha(
        receipt.get("result_state_fingerprint"),
        f"{label} result state fingerprint",
    )
    if plan.get("executable_manifest_fingerprint") is None:
        return _sha(fingerprint, f"{label} execution receipt fingerprint"), None, result_state
    if receipt.get("executable_authority") is not True:
        raise CircuitIncrementalError(f"{label} execution used an unsealed executable bundle")
    if receipt.get("executable_manifest_fingerprint") != plan.get(
        "executable_manifest_fingerprint"
    ):
        raise CircuitIncrementalError(f"{label} execution manifest mismatch")
    if receipt.get("resolved_resource_provider_fingerprint") != plan.get(
        "resource_provider_fingerprint"
    ):
        raise CircuitIncrementalError(f"{label} resource provider binding mismatch")
    expected_executors = plan.get("resolved_executable_bindings", ())
    observed_executors = receipt.get("resolved_executable_bindings", ())
    if len(expected_executors) != len(observed_executors) or any(
        observed != {**expected, "resolution": "artifact-load-verified"}
        for expected, observed in zip(expected_executors, observed_executors, strict=True)
    ):
        raise CircuitIncrementalError(f"{label} resolved executor bindings mismatch")
    expected_consumer = plan.get("resolved_native_consumer_binding")
    if receipt.get("resolved_native_consumer_binding") != {
        **expected_consumer,
        "resolution": "artifact-load-verified",
    }:
        raise CircuitIncrementalError(f"{label} resolved native consumer binding mismatch")
    loads = receipt.get("executable_load_receipts")
    expected_binding_fingerprints = [
        row.get("fingerprint") for row in expected_executors
    ] + [expected_consumer.get("fingerprint")]
    if not isinstance(loads, list) or len(loads) != len(expected_binding_fingerprints):
        raise CircuitIncrementalError(f"{label} executable load receipt set mismatch")
    load_fingerprints: list[str] = []
    loader_fingerprints: set[str] = set()
    for expected_binding, load in zip(expected_binding_fingerprints, loads, strict=True):
        if not isinstance(load, Mapping):
            raise CircuitIncrementalError(f"{label} executable load receipt is malformed")
        load_body = {key: value for key, value in load.items() if key != "fingerprint"}
        if (
            load.get("schema") != "saturn-circuit-executable-load-receipt-v1"
            or load.get("fingerprint") != content_digest(load_body)
            or load.get("binding_fingerprint") != expected_binding
            or load.get("expected_sha256") != load.get("observed_sha256")
            or load.get("authority_scope")
            != "top-level-python-artifact-bytes-and-entrypoint-v1"
        ):
            raise CircuitIncrementalError(f"{label} executable load receipt mismatch")
        load_fingerprints.append(
            _sha(load.get("fingerprint"), f"{label} executable load receipt fingerprint")
        )
        loader_fingerprints.add(
            _sha(load.get("loader_fingerprint"), f"{label} executable loader fingerprint")
        )
    if len(loader_fingerprints) != 1:
        raise CircuitIncrementalError(f"{label} executable loads used multiple loaders")
    bundle = _sha(
        receipt.get("executable_bundle_fingerprint"),
        f"{label} executable bundle fingerprint",
    )
    expected_bundle = content_digest(
        {
            "schema": "saturn-circuit-executable-bundle-v1",
            "manifest_fingerprint": plan.get("executable_manifest_fingerprint"),
            "loader_fingerprint": next(iter(loader_fingerprints)),
            "load_receipt_fingerprints": load_fingerprints,
        }
    )
    if bundle != expected_bundle:
        raise CircuitIncrementalError(f"{label} executable bundle fingerprint mismatch")
    return _sha(fingerprint, f"{label} execution receipt fingerprint"), bundle, result_state


def record_incremental_parity(
    plan: Mapping[str, Any],
    *,
    full_output_sha256: str,
    incremental_output_sha256: str,
    full_execution_receipt: Mapping[str, Any] | None = None,
    incremental_execution_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if plan.get("schema") != CIRCUIT_INCREMENTAL_PLAN_V4_SCHEMA:
        raise CircuitIncrementalError("unsupported incremental plan schema")
    if plan.get("fingerprint") != content_digest(
        {key: value for key, value in plan.items() if key != "fingerprint"}
    ):
        raise CircuitIncrementalError("incremental plan fingerprint mismatch")
    full = _sha(full_output_sha256, "full output hash")
    incremental = _sha(incremental_output_sha256, "incremental output hash")
    scope = plan.get("certification_scope")
    if not isinstance(scope, Mapping):
        raise CircuitIncrementalError("incremental plan lacks a certification scope")
    scope = _json(dict(scope), "certification scope")
    if scope.get("schema") != CIRCUIT_CERTIFICATION_SCOPE_V4_SCHEMA:
        raise CircuitIncrementalError("unsupported certification scope schema")
    scope_fingerprint = content_digest(scope)
    if plan.get("certification_scope_fingerprint") != scope_fingerprint:
        raise CircuitIncrementalError("certification scope fingerprint mismatch")
    bound_fields = (
        "certification_subject_fingerprint",
        "specimen_fingerprint",
        "numerical_contract",
        "native_consumer_contract",
        "native_consumer_identity",
        "candidate_request_fingerprint",
        *(
            (
                "executable_manifest_fingerprint",
                "resolved_executable_bindings",
                "resolved_native_consumer_binding",
            )
            if scope.get("executable_manifest_fingerprint") is not None
            else ()
        ),
        *(
            ("resource_provider_fingerprint",)
            if scope.get("resource_provider_fingerprint") is not None
            else ()
        ),
    )
    if any(plan.get(key) != scope.get(key) for key in bound_fields):
        raise CircuitIncrementalError("incremental plan certification binding mismatch")
    exact = full == incremental
    authority_eligible = (
        scope.get("executable_manifest_fingerprint") is not None
        and not scope.get("authority_refusals")
    )
    execution_receipts: dict[str, str] = {}
    full_execution, full_bundle, full_state = _validate_v4_execution_receipt(
            full_execution_receipt,
            plan=plan,
            output_sha256=full,
            label="full",
        )
    incremental_execution, incremental_bundle, incremental_state = _validate_v4_execution_receipt(
            incremental_execution_receipt,
            plan=plan,
            output_sha256=incremental,
            label="incremental",
        )
    execution_receipts = {
            "full_execution_receipt_fingerprint": full_execution,
            "incremental_execution_receipt_fingerprint": incremental_execution,
            "full_result_state_fingerprint": full_state,
            "incremental_result_state_fingerprint": incremental_state,
            **(
                {
                    "full_executable_bundle_fingerprint": full_bundle,
                    "incremental_executable_bundle_fingerprint": incremental_bundle,
                }
                if full_bundle is not None and incremental_bundle is not None
                else {}
            ),
        }
    body = {
        "schema": CIRCUIT_PARITY_RECEIPT_V4_SCHEMA,
        "plan_fingerprint": plan["fingerprint"],
        "certification_scope": scope,
        "certification_scope_fingerprint": scope_fingerprint,
        **{key: scope[key] for key in bound_fields},
        **execution_receipts,
        "full_output_sha256": full,
        "incremental_output_sha256": incremental,
        "exact": exact and full_state == incremental_state,
        "status": (
            "certified"
            if exact and full_state == incremental_state and authority_eligible
            else (
                "exact-non-authoritative"
                if exact and full_state == incremental_state
                else "skeleton-falsified"
            )
        ),
        "missing_edge_candidates": [] if exact else list(plan.get("reused_cells", ())),
        "authority": {
            "admit_plan_scope": (
                exact and full_state == incremental_state and authority_eligible
            ),
            "widen_tolerance": False,
        },
    }
    return {**body, "fingerprint": content_digest(body)}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise CircuitIncrementalError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="verify an incremental build graph")
    verify.add_argument("graph", type=Path)
    plan = sub.add_parser("plan", help="plan a dependency-cone rebuild")
    plan.add_argument("graph", type=Path)
    plan.add_argument("state", type=Path)
    plan.add_argument("--changed-cell", action="append", default=[])
    plan.add_argument("--changed-address", action="append", default=[])
    plan.add_argument("--output", type=Path)
    parity = sub.add_parser("parity", help="record exact full-versus-incremental parity")
    parity.add_argument("plan", type=Path)
    parity.add_argument("--full", required=True)
    parity.add_argument("--incremental", required=True)
    parity.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "verify":
        graph = CircuitBuildGraph.from_dict(_read(args.graph))
        print(
            json.dumps({"status": "valid", "graph_fingerprint": graph.fingerprint}, sort_keys=True)
        )
        return 0
    if args.command == "plan":
        graph = CircuitBuildGraph.from_dict(_read(args.graph))
        state = CircuitBuildState.from_dict(_read(args.state))
        result = plan_incremental_rebuild(
            graph, state, changed_cells=args.changed_cell, changed_addresses=args.changed_address
        )
    else:
        result = record_incremental_parity(
            _read(args.plan),
            full_output_sha256=args.full,
            incremental_output_sha256=args.incremental,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


__all__ = [
    "CIRCUIT_BUILD_GRAPH_SCHEMA",
    "CIRCUIT_BUILD_STATE_SCHEMA",
    "CIRCUIT_CERTIFICATION_SCOPE_SCHEMA",
    "CIRCUIT_CELL_RESULT_SCHEMA",
    "CIRCUIT_CELL_SCHEMA",
    "CIRCUIT_INCREMENTAL_PLAN_SCHEMA",
    "CIRCUIT_PARITY_RECEIPT_SCHEMA",
    "CIRCUIT_CERTIFICATION_SCOPE_V3_SCHEMA",
    "CIRCUIT_CERTIFICATION_SCOPE_V4_SCHEMA",
    "CIRCUIT_CANDIDATE_REQUEST_SCHEMA",
    "CIRCUIT_EXECUTABLE_BINDING_SCHEMA",
    "CIRCUIT_EXECUTABLE_MANIFEST_SCHEMA",
    "CIRCUIT_INCREMENTAL_PLAN_V3_SCHEMA",
    "CIRCUIT_INCREMENTAL_PLAN_V4_SCHEMA",
    "CIRCUIT_NATIVE_CONSUMER_BINDING_SCHEMA",
    "CIRCUIT_PARITY_RECEIPT_V3_SCHEMA",
    "CIRCUIT_PARITY_RECEIPT_V4_SCHEMA",
    "CircuitBuildGraph",
    "CircuitBuildState",
    "CircuitCell",
    "CircuitCellResult",
    "CircuitIncrementalError",
    "CircuitExecutableBinding",
    "CircuitExecutableManifest",
    "CircuitNativeConsumerBinding",
    "graph_from_skeleton",
    "plan_incremental_rebuild",
    "record_incremental_parity",
]


if __name__ == "__main__":
    raise SystemExit(main())

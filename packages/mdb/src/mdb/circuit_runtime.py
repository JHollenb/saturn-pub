"""Model-free exact runtime for :mod:`saturn.circuit_incremental` graphs.

The runtime is intentionally small.  Executors consume finite JSON values,
every cell output is stored in a content-addressed artifact store, and reuse is
admitted only when the planner, prior state, declared inputs, and stored bytes
all verify exactly.  A native-consumer callback always runs over the graph's
declared outputs; it is never cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Protocol

from mdb.authoritative_dependency import (
    AuthoritativeDependencyError,
    BranchPlan,
    BranchRequest,
    DependencyContract,
    DependencyDeclaration,
    WeightPageAdmission,
    admit_weight_pages,
    compute_dirty_closure,
)
from mdb.authoritative_dependency import (
    compile_branch_plan as compile_authoritative_branch_plan,
)
from mdb.circuit_incremental import (
    CIRCUIT_CANDIDATE_REQUEST_SCHEMA,
    CIRCUIT_INCREMENTAL_PLAN_SCHEMA,
    CIRCUIT_INCREMENTAL_PLAN_V3_SCHEMA,
    CIRCUIT_INCREMENTAL_PLAN_V4_SCHEMA,
    CircuitBuildGraph,
    CircuitBuildState,
    CircuitCell,
    CircuitCellResult,
    CircuitExecutableBinding,
    CircuitIncrementalError,
    CircuitNativeConsumerBinding,
    plan_incremental_rebuild,
)
from mdb.circuit_residency import CircuitWeightLease, CircuitWeightSet
from mdb.execution_skeleton import canonical_bytes, content_digest
from mdb.vm_cache import (
    VM_CACHE_BUNDLE_REF_PREFIX,
    VMCacheEmitter,
    VMCacheError,
    VMCacheExecutionBundle,
)

CIRCUIT_EXECUTION_RECEIPT_SCHEMA = "saturn-circuit-execution-receipt-v2"
CIRCUIT_EXECUTION_RECEIPT_V3_SCHEMA = "saturn-circuit-execution-receipt-v3"
CIRCUIT_EXECUTION_RECEIPT_V4_SCHEMA = "saturn-circuit-execution-receipt-v4"
CIRCUIT_EXECUTION_RESULT_SCHEMA = "saturn-circuit-execution-result-v1"
CIRCUIT_SKIPPED_VALUE_SCHEMA = "saturn-circuit-skipped-value-v1"
CIRCUIT_EXECUTABLE_LOAD_RECEIPT_SCHEMA = "saturn-circuit-executable-load-receipt-v1"
CIRCUIT_EXECUTABLE_BUNDLE_SCHEMA = "saturn-circuit-executable-bundle-v1"
EXECUTABLE_AUTHORITY_SCOPE = "top-level-python-artifact-bytes-and-entrypoint-v1"


class CircuitRuntimeError(CircuitIncrementalError):
    """Raised when exact circuit execution or reuse cannot be established."""


@dataclass(frozen=True, slots=True)
class ExecutableLoadReceipt:
    binding_fingerprint: str
    artifact_ref: str
    expected_sha256: str
    observed_sha256: str
    entrypoint: str
    loader_fingerprint: str
    authority_scope: str = EXECUTABLE_AUTHORITY_SCOPE
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_EXECUTABLE_LOAD_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "binding_fingerprint",
            "expected_sha256",
            "observed_sha256",
            "loader_fingerprint",
        ):
            value = str(getattr(self, name)).casefold()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise CircuitRuntimeError(f"{name} must be a SHA-256 digest")
            object.__setattr__(self, name, value)
        if self.expected_sha256 != self.observed_sha256:
            raise CircuitRuntimeError("loaded executable bytes do not match the manifest")
        if self.authority_scope != EXECUTABLE_AUTHORITY_SCOPE:
            raise CircuitRuntimeError("unsupported executable authority scope")
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding_fingerprint": self.binding_fingerprint,
            "artifact_ref": self.artifact_ref,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "entrypoint": self.entrypoint,
            "loader_fingerprint": self.loader_fingerprint,
            "authority_scope": self.authority_scope,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}


class LocalExecutableArtifactLoader:
    """Load exact Python artifact bytes beneath one confined custody root.

    V1 authority seals only each top-level Python source artifact and declared
    entrypoint.  It does not seal the Python interpreter or imported dependency
    closure; those require a future wheel, environment, or container loader.
    """

    def __init__(self, artifact_root: str | Path) -> None:
        self.artifact_root = Path(artifact_root).resolve(strict=True)
        if not self.artifact_root.is_dir():
            raise CircuitRuntimeError("executable artifact root must be a directory")
        self.fingerprint = content_digest(
            {
                "schema": "saturn-local-executable-artifact-loader-v1",
                "implementation": "saturn-local-python-source-file-loader-v1",
                "runtime_lane": "python-source-v1",
                "authority_scope": EXECUTABLE_AUTHORITY_SCOPE,
            }
        )

    def load(
        self, binding: CircuitExecutableBinding | CircuitNativeConsumerBinding
    ) -> tuple[Callable[..., Any], ExecutableLoadReceipt]:
        if binding.runtime_lane != "python-source-v1":
            raise CircuitRuntimeError(
                f"unsupported local executable runtime lane {binding.runtime_lane!r}"
            )
        relative = PurePosixPath(binding.artifact_ref)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in ("", ".", "..") for part in relative.parts)
            or "\\" in binding.artifact_ref
        ):
            raise CircuitRuntimeError("executable artifact_ref must be a safe relative path")
        try:
            target = (self.artifact_root / Path(*relative.parts)).resolve(strict=True)
        except FileNotFoundError as exc:
            raise CircuitRuntimeError(
                f"missing executable artifact {binding.artifact_ref!r}"
            ) from exc
        try:
            target.relative_to(self.artifact_root)
        except ValueError as exc:
            raise CircuitRuntimeError("executable artifact_ref escapes the artifact root") from exc
        if not target.is_file() or target.suffix != ".py":
            raise CircuitRuntimeError("local executable artifact must be a Python source file")
        payload = target.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if observed != binding.artifact_sha256:
            raise CircuitRuntimeError("executable artifact SHA-256 does not match the manifest")
        namespace: dict[str, Any] = {
            "__builtins__": __builtins__,
            "__file__": str(target),
            "__name__": f"saturn_artifact_{observed}",
        }
        try:
            code = compile(payload, str(target), "exec", dont_inherit=True)
            exec(code, namespace, namespace)
        except Exception as exc:
            raise CircuitRuntimeError(
                f"failed to load executable artifact {binding.artifact_ref!r}"
            ) from exc
        if ":" in binding.entrypoint:
            module_name, entrypoint = binding.entrypoint.split(":", 1)
            if module_name != target.stem:
                raise CircuitRuntimeError("executable entrypoint module does not match artifact")
        else:
            entrypoint = binding.entrypoint
        value: Any = namespace
        for component in entrypoint.split("."):
            if not component or component.startswith("_"):
                raise CircuitRuntimeError("executable entrypoint is invalid")
            if isinstance(value, Mapping):
                if component not in value:
                    raise CircuitRuntimeError(
                        f"missing executable entrypoint {binding.entrypoint!r}"
                    )
                value = value[component]
            else:
                if not hasattr(value, component):
                    raise CircuitRuntimeError(
                        f"missing executable entrypoint {binding.entrypoint!r}"
                    )
                value = getattr(value, component)
        if not callable(value):
            raise CircuitRuntimeError(
                f"executable entrypoint {binding.entrypoint!r} is not callable"
            )
        receipt = ExecutableLoadReceipt(
            binding_fingerprint=binding.fingerprint,
            artifact_ref=binding.artifact_ref,
            expected_sha256=binding.artifact_sha256,
            observed_sha256=observed,
            entrypoint=binding.entrypoint,
            loader_fingerprint=self.fingerprint,
        )
        return value, receipt


def _json(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_bytes(value))
    except (TypeError, ValueError) as exc:
        raise CircuitRuntimeError(f"{label} must be finite JSON data") from exc


class ArtifactStore(Protocol):
    """Narrow content-addressed artifact-store contract used by the runtime."""

    def put(self, value: Any) -> str: ...

    def get(self, artifact_ref: str) -> Any: ...

    def expected_ref(self, output_sha256: str) -> str: ...


class MemoryArtifactStore:
    """A model-free in-memory CAS suitable for tests and small mechanics runs."""

    def __init__(self) -> None:
        self._artifacts: dict[str, bytes] = {}

    def put(self, value: Any) -> str:
        payload = canonical_bytes(value)
        digest = content_digest(value)
        ref = self.expected_ref(digest)
        self._artifacts.setdefault(ref, payload)
        return ref

    def get(self, artifact_ref: str) -> Any:
        try:
            payload = self._artifacts[artifact_ref]
        except KeyError as exc:
            raise CircuitRuntimeError(f"missing cell artifact {artifact_ref!r}") from exc
        try:
            return json.loads(payload)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise CircuitRuntimeError(f"corrupt cell artifact {artifact_ref!r}") from exc

    @staticmethod
    def expected_ref(output_sha256: str) -> str:
        return f"cas://sha256/{output_sha256}"


class DirectoryArtifactStore:
    """Atomic directory-backed CAS that can be reopened by another process."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def expected_ref(output_sha256: str) -> str:
        return f"cas://sha256/{output_sha256}"

    def _path(self, artifact_ref: str) -> Path:
        prefix = "cas://sha256/"
        if not artifact_ref.startswith(prefix):
            raise CircuitRuntimeError(f"unsupported cell artifact reference {artifact_ref!r}")
        digest = artifact_ref[len(prefix) :].casefold()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise CircuitRuntimeError(f"invalid cell artifact digest {digest!r}")
        return self.root / digest[:2] / f"{digest}.json"

    def put(self, value: Any) -> str:
        payload = canonical_bytes(value)
        digest = content_digest(value)
        ref = self.expected_ref(digest)
        target = self._path(ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != payload:
                raise CircuitRuntimeError(f"content collision for cell artifact {ref!r}")
            return ref
        from .bench_publication import stage_write
        if stage_write(target, payload): return ref
        descriptor, temporary = tempfile.mkstemp(prefix=f".{digest}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ref

    def get(self, artifact_ref: str) -> Any:
        target = self._path(artifact_ref)
        try:
            payload = target.read_bytes()
        except FileNotFoundError as exc:
            raise CircuitRuntimeError(f"missing cell artifact {artifact_ref!r}") from exc
        digest = artifact_ref.rsplit("/", 1)[-1]
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise CircuitRuntimeError(f"corrupt cell artifact {artifact_ref!r}") from exc
        if content_digest(value) != digest or canonical_bytes(value) != payload:
            raise CircuitRuntimeError(f"artifact hash mismatch for {artifact_ref!r}")
        return value


@dataclass(frozen=True, slots=True)
class CircuitExecutionReceipt:
    graph_fingerprint: str
    mode: str
    authoritative: bool
    recomputed_cells: tuple[str, ...]
    reused_cells: tuple[str, ...]
    skipped_cells: tuple[str, ...]
    baseline_forward_units: int
    executed_forward_units: int
    reused_forward_units: int
    skipped_forward_units: int
    output_sha256: str
    candidate_request_fingerprint: str
    result_state_fingerprint: str
    plan_fingerprint: str | None = None
    resource_demand_telemetry: tuple[Mapping[str, Any], ...] = ()
    provider_receipt_fingerprint: str | None = None
    resolved_resource_provider_fingerprint: str | None = None
    dependency_contract_fingerprint: str | None = None
    parent_state_cut_fingerprint: str | None = None
    branch_plan_fingerprint: str | None = None
    page_admission_fingerprint: str | None = None
    admitted_page_ids: tuple[str, ...] = ()
    vm_cache_bundle_reference: str | None = None
    vm_cache_bundle_fingerprint: str | None = None
    fingerprint: str = field(init=False)

    schema: ClassVar[str] = CIRCUIT_EXECUTION_RECEIPT_V4_SCHEMA

    def __post_init__(self) -> None:
        for name in ("candidate_request_fingerprint", "result_state_fingerprint"):
            digest = str(getattr(self, name)).casefold()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise CircuitRuntimeError(f"{name} must be a SHA-256 digest")
            object.__setattr__(self, name, digest)
        if self.baseline_forward_units != (
            self.executed_forward_units + self.reused_forward_units + self.skipped_forward_units
        ):
            raise CircuitRuntimeError("forward-unit accounting does not reconcile")
        telemetry = tuple(
            _json(dict(row), "resource demand telemetry")
            for row in self.resource_demand_telemetry
        )
        object.__setattr__(self, "resource_demand_telemetry", telemetry)
        if self.provider_receipt_fingerprint is not None:
            digest = str(self.provider_receipt_fingerprint).casefold()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise CircuitRuntimeError("provider receipt fingerprint must be a SHA-256 digest")
            object.__setattr__(self, "provider_receipt_fingerprint", digest)
        if self.resolved_resource_provider_fingerprint is not None:
            provider = str(self.resolved_resource_provider_fingerprint).casefold()
            if len(provider) != 64 or any(
                char not in "0123456789abcdef" for char in provider
            ):
                raise CircuitRuntimeError(
                    "resolved resource provider fingerprint must be a SHA-256 digest"
                )
            object.__setattr__(self, "resolved_resource_provider_fingerprint", provider)
        for name in (
            "dependency_contract_fingerprint",
            "parent_state_cut_fingerprint",
            "branch_plan_fingerprint",
            "page_admission_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                digest = str(value).casefold()
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise CircuitRuntimeError(f"{name} must be a SHA-256 digest")
                object.__setattr__(self, name, digest)
        page_ids = tuple(str(page_id) for page_id in self.admitted_page_ids)
        if len(page_ids) != len(set(page_ids)):
            raise CircuitRuntimeError("admitted page IDs must be unique")
        object.__setattr__(self, "admitted_page_ids", page_ids)
        cache_reference = self.vm_cache_bundle_reference
        cache_fingerprint = self.vm_cache_bundle_fingerprint
        if (cache_reference is None) != (cache_fingerprint is None):
            raise CircuitRuntimeError(
                "VM cache bundle reference and fingerprint must be supplied together"
            )
        if cache_reference is not None:
            digest = str(cache_fingerprint).casefold()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise CircuitRuntimeError(
                    "vm_cache_bundle_fingerprint must be a SHA-256 digest"
                )
            expected_reference = VM_CACHE_BUNDLE_REF_PREFIX + digest
            if str(cache_reference) != expected_reference:
                raise CircuitRuntimeError("VM cache bundle reference/fingerprint mismatch")
            object.__setattr__(self, "vm_cache_bundle_reference", expected_reference)
            object.__setattr__(self, "vm_cache_bundle_fingerprint", digest)
        object.__setattr__(self, "fingerprint", content_digest(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph_fingerprint": self.graph_fingerprint,
            "mode": self.mode,
            "authoritative": self.authoritative,
            "recomputed_cells": list(self.recomputed_cells),
            "reused_cells": list(self.reused_cells),
            "skipped_cells": list(self.skipped_cells),
            "forward_units": {
                "baseline": self.baseline_forward_units,
                "executed": self.executed_forward_units,
                "reused": self.reused_forward_units,
                "skipped": self.skipped_forward_units,
            },
            "output_sha256": self.output_sha256,
            "candidate_request_fingerprint": self.candidate_request_fingerprint,
            "result_state_fingerprint": self.result_state_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "resource_demand_telemetry": [dict(row) for row in self.resource_demand_telemetry],
            "provider_receipt_fingerprint": self.provider_receipt_fingerprint,
            **(
                {
                    "dependency_contract_fingerprint": self.dependency_contract_fingerprint,
                    "parent_state_cut_fingerprint": self.parent_state_cut_fingerprint,
                    "branch_plan_fingerprint": self.branch_plan_fingerprint,
                    "page_admission_fingerprint": self.page_admission_fingerprint,
                    "admitted_page_ids": list(self.admitted_page_ids),
                }
                if self.dependency_contract_fingerprint is not None
                else {}
            ),
            **(
                {
                    "resolved_resource_provider_fingerprint": (
                        self.resolved_resource_provider_fingerprint
                    )
                }
                if self.resolved_resource_provider_fingerprint is not None
                else {}
            ),
            **(
                {
                    "vm_cache_bundle_reference": self.vm_cache_bundle_reference,
                    "vm_cache_bundle_fingerprint": self.vm_cache_bundle_fingerprint,
                }
                if self.vm_cache_bundle_reference is not None
                else {}
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class CircuitExecutionReceiptV3(CircuitExecutionReceipt):
    """Execution receipt that binds the deployed executable bundle."""

    executable_manifest_fingerprint: str = ""
    resolved_executable_bindings: tuple[Mapping[str, Any], ...] = ()
    resolved_native_consumer_binding: Mapping[str, Any] = field(default_factory=dict)
    executable_load_receipts: tuple[Mapping[str, Any], ...] = ()
    executable_bundle_fingerprint: str | None = None
    executable_authority: bool = False

    schema: ClassVar[str] = CIRCUIT_EXECUTION_RECEIPT_V4_SCHEMA

    def __post_init__(self) -> None:
        manifest = str(self.executable_manifest_fingerprint).casefold()
        if len(manifest) != 64 or any(char not in "0123456789abcdef" for char in manifest):
            raise CircuitRuntimeError("executable manifest fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "executable_manifest_fingerprint", manifest)
        object.__setattr__(
            self,
            "resolved_executable_bindings",
            tuple(
                _json(dict(row), "resolved executable binding")
                for row in self.resolved_executable_bindings
            ),
        )
        object.__setattr__(
            self,
            "resolved_native_consumer_binding",
            _json(dict(self.resolved_native_consumer_binding), "resolved native consumer binding"),
        )
        object.__setattr__(
            self,
            "executable_load_receipts",
            tuple(
                _json(dict(row), "executable load receipt")
                for row in self.executable_load_receipts
            ),
        )
        if self.executable_bundle_fingerprint is not None:
            bundle = str(self.executable_bundle_fingerprint).casefold()
            if len(bundle) != 64 or any(char not in "0123456789abcdef" for char in bundle):
                raise CircuitRuntimeError(
                    "executable bundle fingerprint must be a SHA-256 digest"
                )
            object.__setattr__(self, "executable_bundle_fingerprint", bundle)
        if self.executable_authority and (
            not self.executable_load_receipts or self.executable_bundle_fingerprint is None
        ):
            raise CircuitRuntimeError("executable authority requires verified artifact loads")
        super(CircuitExecutionReceiptV3, self).__post_init__()

    def _body(self) -> dict[str, Any]:
        return {
            **super(CircuitExecutionReceiptV3, self)._body(),
            "executable_manifest_fingerprint": self.executable_manifest_fingerprint,
            "resolved_executable_bindings": [
                dict(row) for row in self.resolved_executable_bindings
            ],
            "resolved_native_consumer_binding": dict(
                self.resolved_native_consumer_binding
            ),
            "executable_load_receipts": [
                dict(row) for row in self.executable_load_receipts
            ],
            "executable_bundle_fingerprint": self.executable_bundle_fingerprint,
            "executable_authority": self.executable_authority,
        }


@dataclass(frozen=True, slots=True)
class CircuitExecutionResult:
    state: CircuitBuildState
    output: Any
    receipt: CircuitExecutionReceipt

    schema: ClassVar[str] = CIRCUIT_EXECUTION_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state": self.state.to_dict(),
            "output": _json(self.output, "execution result output"),
            "receipt": self.receipt.to_dict(),
        }


Executor = Callable[..., Any]
NativeConsumer = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class _AuthoritativeRuntimeBinding:
    """Metadata-only dependency binding used by the existing circuit runtime."""

    contract: DependencyContract
    branch_plan: BranchPlan
    admission: WeightPageAdmission
    parent_state_cut_fingerprint: str
    branch_id: str

    @property
    def admitted_page_ids(self) -> tuple[str, ...]:
        return self.admission.admitted_page_ids


def _coerce_dependency_contract(value: Any) -> DependencyContract:
    if isinstance(value, DependencyContract):
        return value
    if isinstance(value, Mapping):
        try:
            return DependencyContract.from_dict(value)
        except AuthoritativeDependencyError as exc:
            raise CircuitRuntimeError("invalid dependency contract") from exc
    raise CircuitRuntimeError("dependency_contract must be a DependencyContract or mapping")


def _coerce_branch_plan(value: Any) -> BranchPlan:
    if isinstance(value, BranchPlan):
        return value
    if isinstance(value, Mapping):
        try:
            return BranchPlan.from_dict(value)
        except AuthoritativeDependencyError as exc:
            raise CircuitRuntimeError("invalid authoritative branch plan") from exc
    raise CircuitRuntimeError("branch_plan must be a BranchPlan or mapping")


def _state_cut_fingerprint(value: Any) -> str:
    if isinstance(value, str):
        raw = value
    elif isinstance(value, Mapping):
        raw = value.get("fingerprint")
    else:
        raw = getattr(value, "fingerprint", None)
    if not isinstance(raw, str):
        raise CircuitRuntimeError(
            "parent_state_cut must expose a StateCut fingerprint; state payloads are not accepted"
        )
    digest = raw.casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise CircuitRuntimeError("parent_state_cut fingerprint must be a SHA-256 digest")
    return digest


def _declarations_match_graph(
    contract: DependencyContract, graph: CircuitBuildGraph
) -> None:
    """Check that an explicit contract names this exact graph declaration set."""

    if contract.graph_fingerprint != graph.fingerprint:
        raise CircuitRuntimeError("dependency contract does not match the build graph")
    if tuple(contract.node_ids) != tuple(cell.cell_id for cell in graph.cells):
        raise CircuitRuntimeError("dependency contract nodes do not match the build graph")
    if tuple(contract.outputs) != tuple(graph.outputs):
        raise CircuitRuntimeError("dependency contract outputs do not match the build graph")
    try:
        declarations = tuple(DependencyDeclaration.from_cell(cell) for cell in graph.cells)
    except AuthoritativeDependencyError as exc:
        raise CircuitRuntimeError(
            "build graph lacks complete dependency declarations for authoritative execution"
        ) from exc
    if tuple(row.fingerprint for row in contract.declarations) != tuple(
        row.fingerprint for row in declarations
    ):
        raise CircuitRuntimeError("dependency contract declarations do not match the build graph")


def _branch_for_dirty_cone(
    contract: DependencyContract,
    branch_plan: BranchPlan,
    *,
    changed_cells: Sequence[str],
    changed_addresses: Sequence[str],
) -> tuple[str, ...]:
    """Return the branch IDs whose frozen closure matches this execution."""

    try:
        closure = compute_dirty_closure(
            contract,
            changed_cells=changed_cells,
            changed_addresses=changed_addresses,
        )
    except AuthoritativeDependencyError as exc:
        raise CircuitRuntimeError("authoritative dirty-cone request is invalid") from exc
    positions = {node_id: index for index, node_id in enumerate(contract.node_ids)}
    first_dirty = min(
        (positions[node_id] for node_id in closure.dirty_cells),
        default=len(contract.nodes),
    )
    expected_suffix = contract.node_ids[first_dirty:]
    matches: list[str] = []
    for branch in branch_plan.branches:
        if (
            branch.shared_prefix_cells != branch_plan.shared_prefix_cells
            or branch.branch_suffix_cells != expected_suffix
            or branch.dirty_cells != closure.dirty_cells
            or branch.reused_cells != closure.reused_cells
            or branch.dirty_page_ids != closure.dirty_page_ids
            or branch.dirty_work_units != closure.dirty_work_units
            or branch.closure_fingerprint != closure.fingerprint
            or branch_plan.branch_page_ids.get(branch.branch_id) != closure.dirty_page_ids
            or branch_plan.branch_work_units.get(branch.branch_id) != closure.dirty_work_units
        ):
            continue
        matches.append(branch.branch_id)
    if not matches:
        raise CircuitRuntimeError(
            "authoritative branch plan does not match the requested dirty cone"
        )
    return tuple(matches)


@dataclass(frozen=True, slots=True)
class SealedExecutor:
    """Legacy caller wrapper; never grants v3 executable authority."""

    function: Executor
    binding: CircuitExecutableBinding

    def __post_init__(self) -> None:
        if not callable(self.function) or not isinstance(self.binding, CircuitExecutableBinding):
            raise CircuitRuntimeError("sealed executor requires a callable and executable binding")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class SealedNativeConsumer:
    """Legacy caller wrapper; never grants v3 executable authority."""

    function: NativeConsumer
    binding: CircuitNativeConsumerBinding

    def __post_init__(self) -> None:
        if not callable(self.function) or not isinstance(
            self.binding, CircuitNativeConsumerBinding
        ):
            raise CircuitRuntimeError(
                "sealed native consumer requires a callable and consumer binding"
            )

    def __call__(self, outputs: Mapping[str, Any]) -> Any:
        return self.function(outputs)


class CircuitRuntime:
    """Execute a validated build graph through registered cell executors.

    Executors are called as ``executor(cell=..., inputs=..., lease=...)``.
    ``inputs`` contains ``dependencies`` (cell-id to decoded artifact) and
    ``external`` (the caller's JSON input for this cell).

    ``weight_set_adapter`` may lower the temporary tuple-of-resource-names
    representation into a richer resident ``CircuitWeightSet``.  This keeps
    the runtime independent of any residency implementation while preserving
    the provider contract ``provider.acquire(weight_set, cell_id=...)``.
    """

    def __init__(
        self,
        executors: Mapping[str, Executor] | None = None,
        *,
        artifact_store: ArtifactStore | None = None,
        resource_provider: Any | None = None,
        weight_set_adapter: Callable[[CircuitCell, tuple[str, ...]], Any] | None = None,
        executable_artifact_loader: LocalExecutableArtifactLoader | None = None,
        vm_cache_emitter: VMCacheEmitter | None = None,
    ) -> None:
        registry = dict(executors or {})
        if any(not callable(value) for value in registry.values()):
            raise CircuitRuntimeError("executors must contain only callables")
        if not registry and executable_artifact_loader is None:
            raise CircuitRuntimeError("runtime requires executors or an executable artifact loader")
        if (
            executable_artifact_loader is not None
            and type(executable_artifact_loader) is not LocalExecutableArtifactLoader
        ):
            raise CircuitRuntimeError(
                "only the concrete local executable artifact loader can grant authority"
            )
        self.executors = registry
        self.artifact_store = artifact_store or MemoryArtifactStore()
        self.resource_provider = resource_provider
        self.weight_set_adapter = weight_set_adapter
        self.executable_artifact_loader = executable_artifact_loader
        if vm_cache_emitter is not None and not isinstance(vm_cache_emitter, VMCacheEmitter):
            raise CircuitRuntimeError("vm_cache_emitter must be a VMCacheEmitter")
        self.vm_cache_emitter = vm_cache_emitter

    def plan_authoritative_branch(
        self,
        graph: CircuitBuildGraph,
        *,
        parent_state_cut: Any,
        branch_id: str = "branch",
        changed_cells: Sequence[str] = (),
        changed_addresses: Sequence[str] = (),
        dependency_contract: DependencyContract | Mapping[str, Any] | None = None,
        available_pages: Mapping[str, Any] | None = None,
        byte_budget: int | None = None,
    ) -> BranchPlan:
        """Compile one authoritative branch at the runtime planning boundary.

        This is a metadata-only helper.  It does not execute a cell or open a
        provider lease; the returned ``BranchPlan`` is consumed by
        :meth:`incremental_build`, where native-consumer parity still controls
        execution authority.
        """

        if not isinstance(graph, CircuitBuildGraph):
            raise CircuitRuntimeError("plan_authoritative_branch requires a CircuitBuildGraph")
        if dependency_contract is None:
            try:
                contract = DependencyContract.from_graph(
                    graph,
                    provider=self.resource_provider,
                )
            except AuthoritativeDependencyError as exc:
                raise CircuitRuntimeError(
                    "build graph lacks complete dependency declarations for authoritative planning"
                ) from exc
        else:
            contract = _coerce_dependency_contract(dependency_contract)
        _declarations_match_graph(contract, graph)
        parent_fingerprint = _state_cut_fingerprint(parent_state_cut)
        try:
            return compile_authoritative_branch_plan(
                contract,
                (
                    BranchRequest(
                        branch_id=branch_id,
                        parent_state_cut=parent_fingerprint,
                        changed_cells=changed_cells,
                        changed_addresses=changed_addresses,
                    ),
                ),
                parent_state_cut=parent_fingerprint,
                provider=self.resource_provider,
                available_pages=available_pages,
                byte_budget=byte_budget,
            )
        except AuthoritativeDependencyError as exc:
            raise CircuitRuntimeError("failed to compile authoritative branch plan") from exc

    # The longer name is useful to callers that distinguish this from the
    # legacy incremental planner while keeping one implementation/runtime.
    compile_authoritative_branch_plan = plan_authoritative_branch

    def full_build(
        self,
        graph: CircuitBuildGraph,
        *,
        cell_inputs: Mapping[str, Any] | None = None,
        native_consumer: NativeConsumer | None = None,
        native_consumer_identity: str | None = None,
        candidate_request: Mapping[str, Any] | None = None,
        dependency_contract: DependencyContract | Mapping[str, Any] | None = None,
        branch_plan: BranchPlan | Mapping[str, Any] | None = None,
        parent_state_cut: Any | None = None,
        authoritative_branch_id: str | None = None,
        vm_cache_emission: Mapping[str, Any] | None = None,
    ) -> CircuitExecutionResult:
        """Execute every cell and the unchanged declared-output consumer."""
        normalized_inputs = _json(dict(cell_inputs or {}), "candidate inputs")
        if candidate_request is None:
            request = {
                "schema": "saturn-circuit-full-build-request-v1",
                "graph_fingerprint": graph.fingerprint,
                "cell_inputs": normalized_inputs,
            }
        else:
            request = _json(dict(candidate_request), "candidate request")
            if request.get("schema") != CIRCUIT_CANDIDATE_REQUEST_SCHEMA:
                raise CircuitRuntimeError("unsupported candidate request schema")
            if request.get("graph_fingerprint") != graph.fingerprint:
                raise CircuitRuntimeError("candidate request belongs to another build graph")
            if request.get("cell_inputs") != normalized_inputs:
                raise CircuitRuntimeError(
                    "candidate request inputs do not match the full-build inputs"
                )
        request_fingerprint = content_digest(request)
        return self._execute(
            graph,
            previous=None,
            recompute=tuple(cell.cell_id for cell in graph.cells),
            reused=(),
            cell_inputs=cell_inputs,
            native_consumer=native_consumer,
            native_consumer_identity=native_consumer_identity,
            mode="full",
            authoritative=True,
            plan_fingerprint=None,
            candidate_request_fingerprint=request_fingerprint,
            dependency_contract=dependency_contract,
            branch_plan=branch_plan,
            parent_state_cut=parent_state_cut,
            changed_cells=(),
            changed_addresses=(),
            authoritative_branch_id=authoritative_branch_id,
            full_build=True,
            vm_cache_emission=vm_cache_emission,
        )

    def incremental_build(
        self,
        graph: CircuitBuildGraph,
        previous: CircuitBuildState,
        *,
        changed_cells: Sequence[str] = (),
        changed_addresses: Sequence[str] = (),
        cell_inputs: Mapping[str, Any] | None = None,
        native_consumer: NativeConsumer | None = None,
        native_consumer_identity: str | None = None,
        plan: Mapping[str, Any] | None = None,
        allow_uncertified: bool = False,
        dependency_contract: DependencyContract | Mapping[str, Any] | None = None,
        branch_plan: BranchPlan | Mapping[str, Any] | None = None,
        parent_state_cut: Any | None = None,
        authoritative_branch_id: str | None = None,
        vm_cache_emission: Mapping[str, Any] | None = None,
    ) -> CircuitExecutionResult:
        """Execute the planned dirty cone and verify every reused artifact.

        Uncertified plans are refused by default.  ``allow_uncertified=True``
        is the explicit discovery/certification bootstrap lane: execution is
        permitted, but its receipt is non-authoritative.
        """
        expected = plan_incremental_rebuild(
            graph,
            previous,
            changed_cells=changed_cells,
            changed_addresses=changed_addresses,
            candidate_inputs=cell_inputs,
        )
        selected = expected if plan is None else dict(plan)
        self._verify_plan(selected, expected)
        authorized = selected.get("execution_authorized") is True
        if not authorized and not allow_uncertified:
            raise CircuitRuntimeError("incremental plan is not execution_authorized")
        return self._execute(
            graph,
            previous=previous,
            recompute=tuple(selected["recompute_cells"]),
            reused=tuple(selected["reused_cells"]),
            cell_inputs=cell_inputs,
            native_consumer=native_consumer,
            native_consumer_identity=native_consumer_identity,
            mode="incremental" if authorized else "incremental-certification",
            authoritative=authorized,
            plan_fingerprint=selected["fingerprint"],
            candidate_request_fingerprint=selected["candidate_request_fingerprint"],
            dependency_contract=dependency_contract,
            branch_plan=branch_plan,
            parent_state_cut=parent_state_cut,
            changed_cells=changed_cells,
            changed_addresses=changed_addresses,
            authoritative_branch_id=authoritative_branch_id,
            full_build=False,
            vm_cache_emission=vm_cache_emission,
        )

    @staticmethod
    def _verify_plan(selected: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
        if selected.get("schema") not in (
            CIRCUIT_INCREMENTAL_PLAN_SCHEMA,
            CIRCUIT_INCREMENTAL_PLAN_V3_SCHEMA,
            CIRCUIT_INCREMENTAL_PLAN_V4_SCHEMA,
        ):
            raise CircuitRuntimeError("unsupported incremental plan schema")
        body = {key: value for key, value in selected.items() if key != "fingerprint"}
        if selected.get("fingerprint") != content_digest(body):
            raise CircuitRuntimeError("incremental plan fingerprint mismatch")
        if dict(selected) != dict(expected):
            raise CircuitRuntimeError("incremental plan does not match the requested dirty cone")

    def _execute(
        self,
        graph: CircuitBuildGraph,
        *,
        previous: CircuitBuildState | None,
        recompute: tuple[str, ...],
        reused: tuple[str, ...],
        cell_inputs: Mapping[str, Any] | None,
        native_consumer: NativeConsumer | None,
        native_consumer_identity: str | None,
        mode: str,
        authoritative: bool,
        plan_fingerprint: str | None,
        candidate_request_fingerprint: str,
        dependency_contract: DependencyContract | Mapping[str, Any] | None,
        branch_plan: BranchPlan | Mapping[str, Any] | None,
        parent_state_cut: Any | None,
        changed_cells: Sequence[str],
        changed_addresses: Sequence[str],
        authoritative_branch_id: str | None,
        full_build: bool,
        vm_cache_emission: Mapping[str, Any] | None,
    ) -> CircuitExecutionResult:
        if vm_cache_emission is not None and self.vm_cache_emitter is None:
            raise CircuitRuntimeError(
                "vm_cache_emission requires a VMCacheEmitter bound to the runtime"
            )
        resolved_provider_fingerprint = self._resolve_resource_provider_binding(graph)
        (
            executable_authority,
            resolved_executors,
            resolved_consumer,
            load_receipts,
            bundle_fingerprint,
            execution_executors,
            execution_consumer,
        ) = self._resolve_executable_bundle(graph, native_consumer)
        if graph.executable_manifest is not None:
            if (
                native_consumer_identity is not None
                and str(native_consumer_identity).casefold() != graph.native_consumer_identity
            ):
                raise CircuitRuntimeError("native consumer identity does not match the build graph")
        elif native_consumer is None:
            if native_consumer_identity is not None:
                raise CircuitRuntimeError(
                    "native_consumer_identity is invalid without a native_consumer"
                )
            if graph.native_consumer_identity != self._default_native_consumer_identity(graph):
                raise CircuitRuntimeError(
                    "build graph requires its bound native_consumer callback"
                )
        else:
            if native_consumer_identity is None:
                raise CircuitRuntimeError(
                    "a native_consumer callback requires its bound identity"
                )
            identity = str(native_consumer_identity).casefold()
            if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
                raise CircuitRuntimeError("native consumer identity must be a SHA-256 digest")
            if identity != graph.native_consumer_identity:
                raise CircuitRuntimeError("native consumer identity does not match the build graph")
        external = dict(cell_inputs or {})
        missing_executors = sorted(
            {cell.executor_key for cell in graph.cells} - set(execution_executors)
        )
        if missing_executors:
            raise CircuitRuntimeError(f"missing graph executors {missing_executors}")
        unknown_inputs = set(external) - {cell.cell_id for cell in graph.cells}
        if unknown_inputs:
            raise CircuitRuntimeError(f"cell_inputs names unknown cells {sorted(unknown_inputs)}")
        external = {key: _json(value, f"input for cell {key!r}") for key, value in external.items()}
        previous_rows = {} if previous is None else {row.cell_id: row for row in previous.results}
        if previous is not None:
            self._verify_previous_state(graph, previous)

        authoritative_binding = self._bind_authoritative_plan(
            graph,
            dependency_contract=dependency_contract,
            branch_plan=branch_plan,
            parent_state_cut=parent_state_cut,
            changed_cells=changed_cells,
            changed_addresses=changed_addresses,
            authoritative_branch_id=authoritative_branch_id,
            full_build=full_build,
        )
        if authoritative_binding is not None:
            self._preflight_authoritative_pages(graph, authoritative_binding, recompute)

        values: dict[str, Any] = {}
        rows: dict[str, CircuitCellResult] = {}
        telemetry: list[Mapping[str, Any]] = []
        executed_cells: list[str] = []
        skipped_cells: list[str] = []
        recompute_set = set(recompute)
        reused_set = set(reused)
        for cell in graph.cells:  # CircuitBuildGraph stores deterministic topological order.
            self._verify_resource_provider_binding(
                graph, resolved_provider_fingerprint
            )
            dependencies = {parent: values[parent] for parent in cell.dependencies}
            dependency_hashes = {parent: rows[parent].output_sha256 for parent in cell.dependencies}
            invocation = {
                "dependencies": dependency_hashes,
                "external": {
                    "present": cell.cell_id in external,
                    "value": external.get(cell.cell_id),
                },
            }
            input_fingerprint = content_digest(invocation)
            weight_set = self._weight_set_for_execution(
                cell,
                authoritative_binding,
                graph.numerical_contract,
            )
            if (
                isinstance(weight_set, CircuitWeightSet)
                and weight_set.numerical_contract != graph.numerical_contract
            ):
                raise CircuitRuntimeError(
                    f"cell {cell.cell_id!r} weight-set numerical contract does not match the graph"
                )
            if cell.cell_id in reused_set:
                row = previous_rows.get(cell.cell_id)
                if row is None:
                    raise CircuitRuntimeError(f"missing reusable result for cell {cell.cell_id!r}")
                self._verify_reused_cell(
                    cell,
                    row,
                    input_fingerprint,
                    numerical_contract=graph.numerical_contract,
                )
                value = self.artifact_store.get(row.artifact_ref)
                if content_digest(value) != row.output_sha256:
                    raise CircuitRuntimeError(f"artifact hash mismatch for cell {cell.cell_id!r}")
                if self._is_skipped_value(value, cell.cell_id):
                    skipped_cells.append(cell.cell_id)
            elif cell.cell_id in recompute_set:
                skipped = self._route_skip_value(cell, dependencies)
                if skipped is not None:
                    value = skipped
                    skipped_cells.append(cell.cell_id)
                else:
                    executor = execution_executors[cell.executor_key]
                    context = nullcontext(None)
                    if weight_set is not None and self.resource_provider is None:
                        raise CircuitRuntimeError(
                            f"cell {cell.cell_id!r} declares a weight_set but no provider is bound"
                        )
                    if self.resource_provider is not None and weight_set is not None:
                        try:
                            prefetch = getattr(self.resource_provider, "prefetch", None)
                            if callable(prefetch):
                                prefetch(weight_set, cell_id=cell.cell_id)
                            context = self.resource_provider.acquire(
                                weight_set, cell_id=cell.cell_id
                            )
                        except Exception as exc:
                            raise CircuitRuntimeError(
                                f"resource acquisition failed for cell {cell.cell_id!r}"
                            ) from exc
                    with context as lease:
                        self._verify_resource_provider_binding(
                            graph, resolved_provider_fingerprint
                        )
                        value = executor(
                            cell=cell,
                            inputs={
                                "dependencies": dependencies,
                                "external": external.get(cell.cell_id),
                            },
                            lease=lease,
                        )
                        observed = self._lease_telemetry(lease, cell.cell_id)
                    self._verify_resource_provider_binding(
                        graph, resolved_provider_fingerprint
                    )
                    if observed is not None:
                        telemetry.append({**observed, "lease_scope_closed": True})
                    executed_cells.append(cell.cell_id)
                value = _json(value, f"output for cell {cell.cell_id!r}")
                output_sha256 = content_digest(value)
                artifact_ref = self.artifact_store.put(value)
                stored = self.artifact_store.get(artifact_ref)
                if content_digest(stored) != output_sha256:
                    raise CircuitRuntimeError(
                        f"artifact store failed exact verification for cell {cell.cell_id!r}"
                    )
                row = CircuitCellResult(
                    cell_id=cell.cell_id,
                    cell_fingerprint=cell.fingerprint,
                    input_fingerprint=input_fingerprint,
                    output_sha256=output_sha256,
                    artifact_ref=artifact_ref,
                    numerical_contract=graph.numerical_contract,
                )
            else:
                raise CircuitRuntimeError(f"execution plan omits cell {cell.cell_id!r}")
            values[cell.cell_id] = value
            rows[cell.cell_id] = row

        graph_outputs = {cell_id: values[cell_id] for cell_id in graph.outputs}
        self._verify_resource_provider_binding(graph, resolved_provider_fingerprint)
        output = graph_outputs if execution_consumer is None else execution_consumer(graph_outputs)
        self._verify_resource_provider_binding(graph, resolved_provider_fingerprint)
        output = _json(output, "native consumer output")
        output_sha256 = content_digest(output)
        results = tuple(rows[cell.cell_id] for cell in graph.cells)
        support = content_digest(
            {
                row.cell_id: {
                    "input": row.input_fingerprint,
                    "output": row.output_sha256,
                }
                for row in results
            }
        )
        state = CircuitBuildState(
            graph_fingerprint=graph.fingerprint,
            support_fingerprint=support,
            results=results,
        )
        by_id = {cell.cell_id: cell for cell in graph.cells}
        baseline_units = sum(cell.physical_forward_units for cell in graph.cells)
        executed_units = sum(by_id[cell_id].physical_forward_units for cell_id in executed_cells)
        skipped_units = sum(by_id[cell_id].physical_forward_units for cell_id in skipped_cells)
        reused_units = baseline_units - executed_units - skipped_units
        cache_bundle = self._emit_vm_cache_bundle(
            request=vm_cache_emission,
            graph=graph,
            previous=previous,
            state=state,
            mode=mode,
            authoritative=authoritative and executable_authority,
            candidate_request_fingerprint=candidate_request_fingerprint,
            plan_fingerprint=plan_fingerprint,
            authoritative_binding=authoritative_binding,
            recompute=recompute,
            reused=reused,
            skipped=tuple(skipped_cells),
            output_sha256=output_sha256,
            native_records=(
                execution_consumer.vm_cache_records(output)
                if vm_cache_emission is not None
                and callable(getattr(execution_consumer, "vm_cache_records", None))
                else None
            ),
        )
        receipt_type = (
            CircuitExecutionReceiptV3
            if graph.executable_manifest is not None
            else CircuitExecutionReceipt
        )
        receipt_kwargs: dict[str, Any] = {}
        if graph.executable_manifest is not None:
            receipt_kwargs = {
                "executable_manifest_fingerprint": graph.executable_manifest.fingerprint,
                "resolved_executable_bindings": resolved_executors,
                "resolved_native_consumer_binding": resolved_consumer,
                "executable_load_receipts": load_receipts,
                "executable_bundle_fingerprint": bundle_fingerprint,
                "executable_authority": executable_authority,
            }
        provider_receipt_fingerprint = self._provider_receipt_fingerprint()
        self._verify_resource_provider_binding(graph, resolved_provider_fingerprint)
        receipt = receipt_type(
            graph_fingerprint=graph.fingerprint,
            mode=mode,
            authoritative=authoritative and executable_authority,
            recomputed_cells=recompute,
            reused_cells=reused,
            skipped_cells=tuple(skipped_cells),
            baseline_forward_units=baseline_units,
            executed_forward_units=executed_units,
            reused_forward_units=reused_units,
            skipped_forward_units=skipped_units,
            output_sha256=output_sha256,
            candidate_request_fingerprint=candidate_request_fingerprint,
            result_state_fingerprint=state.fingerprint,
            plan_fingerprint=plan_fingerprint,
            resource_demand_telemetry=tuple(telemetry),
            provider_receipt_fingerprint=provider_receipt_fingerprint,
            resolved_resource_provider_fingerprint=resolved_provider_fingerprint,
            **(
                {
                    "vm_cache_bundle_reference": cache_bundle.reference,
                    "vm_cache_bundle_fingerprint": cache_bundle.fingerprint,
                }
                if cache_bundle is not None
                else {}
            ),
            **(
                {
                    "dependency_contract_fingerprint": authoritative_binding.contract.fingerprint,
                    "parent_state_cut_fingerprint": (
                        authoritative_binding.parent_state_cut_fingerprint
                    ),
                    "branch_plan_fingerprint": authoritative_binding.branch_plan.fingerprint,
                    "page_admission_fingerprint": authoritative_binding.admission.fingerprint,
                    "admitted_page_ids": authoritative_binding.admitted_page_ids,
                }
                if authoritative_binding is not None
                else {}
            ),
            **receipt_kwargs,
        )
        return CircuitExecutionResult(state=state, output=output, receipt=receipt)

    def _emit_vm_cache_bundle(
        self,
        *,
        request: Mapping[str, Any] | None,
        graph: CircuitBuildGraph,
        previous: CircuitBuildState | None,
        state: CircuitBuildState,
        mode: str,
        authoritative: bool,
        candidate_request_fingerprint: str,
        plan_fingerprint: str | None,
        authoritative_binding: _AuthoritativeRuntimeBinding | None,
        recompute: Sequence[str],
        reused: Sequence[str],
        skipped: Sequence[str],
        output_sha256: str,
        native_records: Mapping[str, Any] | None = None,
    ) -> VMCacheExecutionBundle | None:
        """Emit pointer-only cache records from an actual execution boundary.

        The caller declares the four typed cache records, but cannot override
        execution lineage, dirty-cone membership, plan authority, or output
        identity.  No tensor or output value is passed to the cache emitter.
        """

        if request is None:
            return None
        if self.vm_cache_emitter is None:
            raise CircuitRuntimeError(
                "vm_cache_emission requires a VMCacheEmitter bound to the runtime"
            )
        if not isinstance(request, Mapping):
            raise CircuitRuntimeError("vm_cache_emission must be a mapping")
        required = {"run_id", "model_identity", "input_identity", "abi_identity", "entries"}
        missing = sorted(required - set(request))
        if missing:
            raise CircuitRuntimeError(f"vm_cache_emission is missing fields {missing}")
        allowed = required | {
            "parent_lineage",
            "output_pointer",
            "metadata",
            "manifest_id_prefix",
        }
        unknown = sorted(set(request) - allowed)
        if unknown:
            raise CircuitRuntimeError(f"vm_cache_emission has unknown fields {unknown}")

        if authoritative_binding is not None:
            parent_identity: Mapping[str, Any] = {
                "kind": "state-cut",
                "fingerprint": authoritative_binding.parent_state_cut_fingerprint,
            }
            selected_plan_fingerprint = authoritative_binding.branch_plan.fingerprint
            admitted_plan: Mapping[str, Any] | None = {
                "fingerprint": selected_plan_fingerprint,
                "pointer": f"ar://saturn/authoritative-branch-plan/{selected_plan_fingerprint}",
                "execution_authorized": authoritative,
                "authorizing_receipts": [],
                "reason": "authoritative-dependency-plan",
            }
        elif previous is not None:
            parent_identity = {
                "kind": "circuit-build-state",
                "fingerprint": previous.fingerprint,
            }
            admitted_plan = (
                None
                if plan_fingerprint is None
                else {
                    "fingerprint": plan_fingerprint,
                    "pointer": f"ar://saturn/circuit-plan/{plan_fingerprint}",
                    "execution_authorized": authoritative,
                    "authorizing_receipts": [],
                    "reason": mode,
                }
            )
        else:
            parent_identity = {
                "kind": "candidate-request",
                "fingerprint": candidate_request_fingerprint,
            }
            admitted_plan = None

        supplied_metadata = request.get("metadata", {})
        if not isinstance(supplied_metadata, Mapping):
            raise CircuitRuntimeError("vm_cache_emission.metadata must be a mapping")
        metadata = {
            **dict(supplied_metadata),
            "execution": {
                "mode": mode,
                "authoritative": authoritative,
                "candidate_request_fingerprint": candidate_request_fingerprint,
                "result_state_fingerprint": state.fingerprint,
                "output_sha256": output_sha256,
            },
        }
        entries = dict(request["entries"])
        output_pointer = request.get("output_pointer")
        if native_records is not None:
            if not isinstance(native_records, Mapping) or set(native_records) - {
                "entries", "metadata", "output_pointer"
            }:
                raise CircuitRuntimeError("native VM cache records have unsupported fields")
            records = native_records.get("entries", {})
            if not isinstance(records, Mapping) or set(records) - {
                "source", "address", "operator", "renderer"
            }:
                raise CircuitRuntimeError("native VM cache records have invalid planes")
            for plane, rows in records.items():
                entries[plane] = tuple(entries.get(plane, ())) + tuple(rows)
            metadata["native_execution"] = native_records.get("metadata", {})
            output_pointer = native_records.get("output_pointer", output_pointer)
        try:
            return self.vm_cache_emitter.emit(
                run_id=request["run_id"],
                parent_identity=parent_identity,
                graph_fingerprint=graph.fingerprint,
                model_identity=request["model_identity"],
                input_identity=request["input_identity"],
                abi_identity=request["abi_identity"],
                entries=entries,
                parent_lineage=request.get("parent_lineage", ()),
                dirty={
                    "mode": mode,
                    "recomputed_cells": list(recompute),
                    "reused_cells": list(reused),
                    "skipped_cells": list(skipped),
                },
                admitted_plan=admitted_plan,
                output_pointer=output_pointer,
                metadata=metadata,
                manifest_id_prefix=request.get("manifest_id_prefix"),
            )
        except VMCacheError as exc:
            raise CircuitRuntimeError("VM cache emission failed") from exc

    def _bind_authoritative_plan(
        self,
        graph: CircuitBuildGraph,
        *,
        dependency_contract: DependencyContract | Mapping[str, Any] | None,
        branch_plan: BranchPlan | Mapping[str, Any] | None,
        parent_state_cut: Any | None,
        changed_cells: Sequence[str],
        changed_addresses: Sequence[str],
        authoritative_branch_id: str | None,
        full_build: bool,
    ) -> _AuthoritativeRuntimeBinding | None:
        supplied = (dependency_contract, branch_plan, parent_state_cut)
        if all(value is None for value in supplied):
            return None
        if dependency_contract is None:
            raise CircuitRuntimeError(
                "authoritative execution requires a dependency contract"
            )
        if branch_plan is None:
            raise CircuitRuntimeError("authoritative execution requires a branch plan")
        if parent_state_cut is None:
            raise CircuitRuntimeError("authoritative execution requires a parent StateCut")
        contract = _coerce_dependency_contract(dependency_contract)
        _declarations_match_graph(contract, graph)
        plan = _coerce_branch_plan(branch_plan)
        parent_fingerprint = _state_cut_fingerprint(parent_state_cut)
        if plan.contract_fingerprint != contract.fingerprint:
            raise CircuitRuntimeError("branch plan contract fingerprint does not match the graph")
        if plan.parent_state_cut_fingerprint != parent_fingerprint:
            raise CircuitRuntimeError("branch plan parent StateCut does not match the request")
        if plan.native_consumer_required is not True:
            raise CircuitRuntimeError("authoritative branch plan requires the native consumer")
        admission = plan.admission
        if not isinstance(admission, WeightPageAdmission):
            raise CircuitRuntimeError("branch plan lacks a typed page admission")
        if admission.contract_fingerprint != contract.fingerprint:
            raise CircuitRuntimeError(
                "page admission contract fingerprint does not match the graph"
            )
        if admission.admitted_page_ids != plan.admitted_page_ids:
            raise CircuitRuntimeError("branch plan and page admission sets do not match")
        try:
            refreshed_admission = admit_weight_pages(
                contract,
                plan.admitted_page_ids,
                provider=self.resource_provider,
                byte_budget=admission.byte_budget,
            )
        except AuthoritativeDependencyError as exc:
            raise CircuitRuntimeError(
                "authoritative page admission failed before provider acquisition"
            ) from exc
        if (
            refreshed_admission.total_bytes != admission.total_bytes
            or refreshed_admission.weight_ids != admission.weight_ids
        ):
            raise CircuitRuntimeError("page admission changed during runtime binding")
        if admission.provider_fingerprint is not None and (
            refreshed_admission.provider_fingerprint != admission.provider_fingerprint
        ):
            raise CircuitRuntimeError("page admission provider fingerprint changed")
        if full_build:
            branch_id = authoritative_branch_id or "full"
        else:
            matches = _branch_for_dirty_cone(
                contract,
                plan,
                changed_cells=changed_cells,
                changed_addresses=changed_addresses,
            )
            if authoritative_branch_id is not None:
                if authoritative_branch_id not in matches:
                    raise CircuitRuntimeError(
                        "authoritative_branch_id does not match the requested dirty cone"
                    )
                branch_id = authoritative_branch_id
            elif len(matches) == 1:
                branch_id = matches[0]
            else:
                raise CircuitRuntimeError(
                    "authoritative branch plan matches multiple branches; specify "
                    "authoritative_branch_id"
                )
        # The re-admission above is a live mechanics check.  Keep the sealed
        # plan admission as the receipt identity so a provider that does not
        # expose an inventory cannot cause a false custody change merely by
        # moving from ``inventory-bound`` to ``declared-only`` status.
        return _AuthoritativeRuntimeBinding(
            contract=contract,
            branch_plan=plan,
            admission=admission,
            parent_state_cut_fingerprint=parent_fingerprint,
            branch_id=branch_id,
        )

    @staticmethod
    def _preflight_authoritative_pages(
        graph: CircuitBuildGraph,
        binding: _AuthoritativeRuntimeBinding,
        recompute: Sequence[str],
    ) -> None:
        declarations = {node.node_id: node for node in binding.contract.declarations}
        admitted = set(binding.admitted_page_ids)
        graph_cells = {cell.cell_id: cell for cell in graph.cells}
        unknown_recompute = set(recompute) - set(graph_cells)
        if unknown_recompute:
            raise CircuitRuntimeError(
                f"authoritative execution names unknown cells {sorted(unknown_recompute)}"
            )
        for cell_id in recompute:
            declared = {page.page_id for page in declarations[cell_id].weight_pages}
            missing = sorted(declared - admitted)
            if missing:
                raise CircuitRuntimeError(
                    f"authoritative page admission omits pages for cell {cell_id!r}: {missing}"
                )

    def _weight_set_for_execution(
        self,
        cell: CircuitCell,
        binding: _AuthoritativeRuntimeBinding | None,
        numerical_contract: str,
    ) -> Any | None:
        if binding is None:
            return self._weight_set(cell)
        declaration = next(
            (node for node in binding.contract.declarations if node.node_id == cell.cell_id),
            None,
        )
        if declaration is None:  # pragma: no cover - declarations are checked at binding
            raise CircuitRuntimeError(f"dependency contract omits cell {cell.cell_id!r}")
        raw = self._weight_set(cell)
        pages = tuple((page.page_id, page.byte_size) for page in declaration.weight_pages)
        if not pages:
            if raw is not None:
                raise CircuitRuntimeError(
                    f"cell {cell.cell_id!r} has provider demand outside its dependency contract"
                )
            return None
        expected = CircuitWeightSet(
            weight_set_id=f"authoritative:{cell.cell_id}",
            resources=pages,
            demand_class=(raw.demand_class if isinstance(raw, CircuitWeightSet) else "static"),
            routed_demand_source=(
                raw.routed_demand_source if isinstance(raw, CircuitWeightSet) else None
            ),
            numerical_contract=(
                raw.numerical_contract if isinstance(raw, CircuitWeightSet) else numerical_contract
            ),
        )
        # The graph numerical contract is checked by the caller after this
        # helper returns.  Use the typed declaration as the source of page
        # identity and preserve the existing weight-set adapter only for the
        # legacy/non-authoritative lane.
        if isinstance(raw, CircuitWeightSet):
            if raw.resource_keys != expected.resource_keys or raw.resources != expected.resources:
                raise CircuitRuntimeError(
                    f"cell {cell.cell_id!r} weight demand disagrees with dependency pages"
                )
            expected = CircuitWeightSet(
                weight_set_id=raw.weight_set_id,
                resources=pages,
                demand_class=raw.demand_class,
                routed_demand_source=raw.routed_demand_source,
                numerical_contract=raw.numerical_contract,
            )
        return expected

    def _resolve_resource_provider_binding(
        self, graph: CircuitBuildGraph
    ) -> str | None:
        """Resolve only the graph-declared physical-provider authority boundary.

        The fingerprint is supplied by the provider and is expected to bind its
        portable logical/content identity. Saturn does not infer stronger
        content custody from a class name, path, lease, or lifecycle receipt.
        """

        expected = graph.resource_provider_fingerprint
        if expected is None:
            return None
        if self.resource_provider is None:
            raise CircuitRuntimeError(
                "build graph binds a resource provider but no provider is present"
            )
        observed = getattr(self.resource_provider, "provider_fingerprint", None)
        if not isinstance(observed, str):
            raise CircuitRuntimeError(
                "bound resource provider lacks a SHA-256 provider_fingerprint"
            )
        observed = observed.casefold()
        if len(observed) != 64 or any(
            char not in "0123456789abcdef" for char in observed
        ):
            raise CircuitRuntimeError(
                "bound resource provider lacks a SHA-256 provider_fingerprint"
            )
        if observed != expected:
            raise CircuitRuntimeError("resource provider fingerprint does not match the graph")
        return observed

    def _verify_resource_provider_binding(
        self,
        graph: CircuitBuildGraph,
        resolved_fingerprint: str | None,
    ) -> None:
        if resolved_fingerprint is None:
            return
        observed = self._resolve_resource_provider_binding(graph)
        if observed != resolved_fingerprint:
            raise CircuitRuntimeError("resource provider changed during circuit execution")

    def _resolve_executable_bundle(
        self,
        graph: CircuitBuildGraph,
        native_consumer: NativeConsumer | None,
    ) -> tuple[
        bool,
        tuple[Mapping[str, Any], ...],
        Mapping[str, Any],
        tuple[Mapping[str, Any], ...],
        str | None,
        Mapping[str, Executor],
        NativeConsumer | None,
    ]:
        """Resolve artifact-loaded callables against deploy custody before execution."""

        manifest = graph.executable_manifest
        if manifest is None:
            # Preserve historical v2 execution behavior. It is not executable
            # authority in the v3 path because no v3 manifest/receipt is emitted.
            return True, (), {}, (), None, self.executors, native_consumer
        declared = {row.executor_key: row for row in manifest.executors}
        resolved: list[Mapping[str, Any]] = []
        if set(declared) != {cell.executor_key for cell in graph.cells}:
            raise CircuitRuntimeError("executable manifest does not cover the graph")
        expected_consumer = manifest.native_consumer
        loader = self.executable_artifact_loader
        if loader is not None:
            if native_consumer is not None:
                raise CircuitRuntimeError(
                    "native_consumer must be resolved by the executable artifact loader"
                )
            loaded_executors: dict[str, Executor] = {}
            loads: list[Mapping[str, Any]] = []
            for binding in manifest.executors:
                function, receipt = loader.load(binding)
                loaded_executors[binding.executor_key] = function
                loads.append(receipt.to_dict())
                resolved.append({**binding.to_dict(), "resolution": "artifact-load-verified"})
            loaded_consumer, consumer_receipt = loader.load(expected_consumer)
            loads.append(consumer_receipt.to_dict())
            consumer = {
                **expected_consumer.to_dict(),
                "resolution": "artifact-load-verified",
            }
            bundle = content_digest(
                {
                    "schema": CIRCUIT_EXECUTABLE_BUNDLE_SCHEMA,
                    "manifest_fingerprint": manifest.fingerprint,
                    "loader_fingerprint": loader.fingerprint,
                    "load_receipt_fingerprints": [row["fingerprint"] for row in loads],
                }
            )
            return (
                True,
                tuple(resolved),
                consumer,
                tuple(loads),
                bundle,
                loaded_executors,
                loaded_consumer,
            )

        for binding in manifest.executors:
            candidate = self.executors.get(binding.executor_key)
            if isinstance(candidate, SealedExecutor) and candidate.binding != binding:
                raise CircuitRuntimeError(
                    f"caller-wrapped executor binding mismatch for {binding.executor_key!r}"
                )
            resolution = (
                "caller-wrapped-discovery"
                if isinstance(candidate, SealedExecutor)
                else "unsealed-discovery"
            )
            resolved.append({**binding.to_dict(), "resolution": resolution})
        if (
            isinstance(native_consumer, SealedNativeConsumer)
            and native_consumer.binding != expected_consumer
        ):
            raise CircuitRuntimeError("caller-wrapped native consumer binding mismatch")
        consumer_resolution = (
            "caller-wrapped-discovery"
            if isinstance(native_consumer, SealedNativeConsumer)
            else "unsealed-discovery"
        )
        consumer = {**expected_consumer.to_dict(), "resolution": consumer_resolution}
        return (
            False,
            tuple(resolved),
            consumer,
            (),
            None,
            self.executors,
            native_consumer,
        )

    @staticmethod
    def _default_native_consumer_identity(graph: CircuitBuildGraph) -> str:
        by_id = {cell.cell_id: cell for cell in graph.cells}
        return content_digest(
            {
                "schema": "saturn-derived-native-consumer-identity-v1",
                "outputs": [by_id[cell_id].to_dict() for cell_id in graph.outputs],
            }
        )

    @staticmethod
    def _route_skip_value(
        cell: CircuitCell, dependencies: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        if "route_condition" not in cell.metadata:
            return None
        condition = cell.metadata["route_condition"]
        if not isinstance(condition, Mapping) or set(condition) != {"cell_id", "equals"}:
            raise CircuitRuntimeError(
                f"cell {cell.cell_id!r} route_condition must contain cell_id and equals"
            )
        source = condition["cell_id"]
        if not isinstance(source, str) or source not in cell.dependencies:
            raise CircuitRuntimeError(
                f"cell {cell.cell_id!r} route source must be a direct dependency"
            )
        expected = _json(condition["equals"], "route condition value")
        actual = dependencies[source]
        if actual == expected:
            return None
        return {
            "schema": CIRCUIT_SKIPPED_VALUE_SCHEMA,
            "cell_id": cell.cell_id,
            "route_source": source,
            "actual_sha256": content_digest(actual),
            "expected_sha256": content_digest(expected),
        }

    @staticmethod
    def _is_skipped_value(value: Any, cell_id: str) -> bool:
        if not isinstance(value, Mapping) or value.get("schema") != CIRCUIT_SKIPPED_VALUE_SCHEMA:
            return False
        if value.get("cell_id") != cell_id:
            raise CircuitRuntimeError(f"skipped artifact cell mismatch for {cell_id!r}")
        return True

    @staticmethod
    def _verify_previous_state(graph: CircuitBuildGraph, previous: CircuitBuildState) -> None:
        if previous.graph_fingerprint != graph.fingerprint:
            raise CircuitRuntimeError("previous state belongs to another build graph")
        expected_support = content_digest(
            {
                row.cell_id: {"input": row.input_fingerprint, "output": row.output_sha256}
                for row in previous.results
            }
        )
        if previous.support_fingerprint != expected_support:
            raise CircuitRuntimeError("previous state support fingerprint mismatch")
        graph_ids = {cell.cell_id for cell in graph.cells}
        if {row.cell_id for row in previous.results} - graph_ids:
            raise CircuitRuntimeError("previous state contains results outside the graph")
        if any(
            row.numerical_contract != graph.numerical_contract for row in previous.results
        ):
            raise CircuitRuntimeError(
                "previous state cell-result numerical contract does not match the graph"
            )

    def _verify_reused_cell(
        self,
        cell: CircuitCell,
        row: CircuitCellResult,
        input_fingerprint: str,
        *,
        numerical_contract: str,
    ) -> None:
        if row.cell_id != cell.cell_id or row.cell_fingerprint != cell.fingerprint:
            raise CircuitRuntimeError(f"cell fingerprint mismatch for reuse of {cell.cell_id!r}")
        if row.input_fingerprint != input_fingerprint:
            raise CircuitRuntimeError(f"input fingerprint mismatch for reuse of {cell.cell_id!r}")
        if row.numerical_contract != numerical_contract:
            raise CircuitRuntimeError(
                f"numerical contract mismatch for reuse of {cell.cell_id!r}"
            )
        value = self.artifact_store.get(row.artifact_ref)
        if content_digest(value) != row.output_sha256:
            raise CircuitRuntimeError(f"artifact hash mismatch for cell {cell.cell_id!r}")
        expected_ref = self.artifact_store.expected_ref(row.output_sha256)
        if row.artifact_ref != expected_ref:
            raise CircuitRuntimeError(f"artifact reference mismatch for cell {cell.cell_id!r}")

    def _weight_set(self, cell: CircuitCell) -> Any | None:
        if "weight_set" not in cell.metadata:
            return None
        raw = cell.metadata["weight_set"]
        if isinstance(raw, Mapping):
            try:
                if raw.get("schema") == CircuitWeightSet.schema:
                    return CircuitWeightSet.from_dict(raw)
                resources = raw.get("resources", ())
                if isinstance(resources, list) and resources and isinstance(resources[0], Mapping):
                    resources = tuple((row.get("key"), row.get("byte_size")) for row in resources)
                return CircuitWeightSet(
                    weight_set_id=raw.get("weight_set_id", cell.cell_id),
                    resources=resources,
                    demand_class=raw.get("demand_class", "static"),
                    routed_demand_source=raw.get("routed_demand_source"),
                    numerical_contract=raw.get("numerical_contract", "native"),
                )
            except Exception as exc:
                raise CircuitRuntimeError(
                    f"invalid typed weight_set metadata for cell {cell.cell_id!r}"
                ) from exc
        if not isinstance(raw, list) or not raw:
            raise CircuitRuntimeError(
                f"cell {cell.cell_id!r} weight_set metadata must be a non-empty list"
            )
        if any(not isinstance(item, str) or not item.strip() for item in raw):
            raise CircuitRuntimeError(
                f"cell {cell.cell_id!r} weight_set entries must be non-empty text"
            )
        resources = tuple(item.strip() for item in raw)
        if len(resources) != len(set(resources)):
            raise CircuitRuntimeError(f"cell {cell.cell_id!r} weight_set contains duplicates")
        if self.weight_set_adapter is None:
            return resources
        try:
            return self.weight_set_adapter(cell, resources)
        except Exception as exc:
            raise CircuitRuntimeError(
                f"weight_set adaptation failed for cell {cell.cell_id!r}"
            ) from exc

    def _lease_telemetry(self, lease: Any, cell_id: str) -> Mapping[str, Any] | None:
        provider_kind = (
            None
            if self.resource_provider is None
            else (
                f"{type(self.resource_provider).__module__}."
                f"{type(self.resource_provider).__qualname__}"
            )
        )
        if isinstance(lease, CircuitWeightLease):
            return {
                "cell_id": cell_id,
                "lease_id": lease.lease_id,
                "provider_fingerprint": lease.provider_fingerprint,
                "provider_kind": provider_kind,
                "weight_set_fingerprint": lease.weight_set_fingerprint,
                "resource_keys": sorted(str(key) for key in lease.resources),
                "declared_demand_bytes": lease.resident_bytes,
                "accounting_basis": "declared-weight-set-bytes",
                "physical_residency_proven": False,
            }
        if lease is None or not hasattr(lease, "telemetry"):
            return None
        value = lease.telemetry() if callable(lease.telemetry) else lease.telemetry
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise CircuitRuntimeError(f"resource telemetry for cell {cell_id!r} must be a mapping")
        return {
            "cell_id": cell_id,
            "provider_kind": provider_kind,
            "accounting_basis": "provider-reported-unverified",
            "physical_residency_proven": False,
            "provider_reported": _json(dict(value), "resource provider telemetry"),
        }

    def _provider_receipt_fingerprint(self) -> str | None:
        if self.resource_provider is None:
            return None
        receipt_fn = getattr(self.resource_provider, "receipt", None)
        if not callable(receipt_fn):
            return None
        receipt = receipt_fn()
        to_dict = getattr(receipt, "to_dict", None)
        fingerprint = getattr(receipt, "fingerprint", None)
        if not callable(to_dict) or not isinstance(fingerprint, str):
            raise CircuitRuntimeError("resource provider receipt is not content-addressable")
        value = to_dict()
        if not isinstance(value, Mapping):
            raise CircuitRuntimeError("resource provider receipt must serialize to a mapping")
        body = {key: item for key, item in value.items() if key != "fingerprint"}
        if value.get("fingerprint") != fingerprint or content_digest(body) != fingerprint:
            raise CircuitRuntimeError("resource provider receipt fingerprint mismatch")
        return fingerprint


__all__ = [
    "ArtifactStore",
    "CIRCUIT_EXECUTABLE_BUNDLE_SCHEMA",
    "CIRCUIT_EXECUTABLE_LOAD_RECEIPT_SCHEMA",
    "CIRCUIT_EXECUTION_RECEIPT_SCHEMA",
    "CIRCUIT_EXECUTION_RESULT_SCHEMA",
    "CIRCUIT_EXECUTION_RECEIPT_V3_SCHEMA",
    "CIRCUIT_SKIPPED_VALUE_SCHEMA",
    "CircuitExecutionReceipt",
    "CircuitExecutionReceiptV3",
    "CircuitExecutionResult",
    "CircuitRuntime",
    "CircuitRuntimeError",
    "DirectoryArtifactStore",
    "EXECUTABLE_AUTHORITY_SCOPE",
    "ExecutableLoadReceipt",
    "LocalExecutableArtifactLoader",
    "MemoryArtifactStore",
    "SealedExecutor",
    "SealedNativeConsumer",
]

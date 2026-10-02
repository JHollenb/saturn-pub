"""Authoritative dependency, dirty-cone, and page-admission metadata.

This module is deliberately smaller than :mod:`saturn.circuit_runtime`.  It
does not execute cells, open a weight lease, or retain tensors.  It validates
the declarations that an executor may use, computes a conservative weighted
dirty closure, and produces a branch plan whose accounting is explicitly
*not* a timing or speedup measurement.

The contract is intentionally fail-closed.  A graph with a missing read,
write, invalidation, edge, or page declaration cannot silently become an
authoritative plan.  This is useful at the boundary between a compiled graph
and a demand-paged provider: a provider can consume the admitted page IDs,
while the native consumer remains responsible for proving numerical parity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

AUTHORITATIVE_DEPENDENCY_SCHEMA = "saturn-authoritative-dependency-v1"
AUTHORITATIVE_BRANCH_PLAN_SCHEMA = "saturn-authoritative-branch-plan-v1"
AUTHORITATIVE_ADMISSION_SCHEMA = "saturn-authoritative-page-admission-v1"
MAX_METADATA_BYTES = 64 * 1024

_MISSING = object()
_FORBIDDEN_TIMING_KEYS = frozenset(
    {
        "speedup",
        "speedup_ratio",
        "latency",
        "latency_seconds",
        "throughput",
        "wall_time",
        "wall_time_seconds",
    }
)


class AuthoritativeDependencyError(ValueError):
    """Raised when an authoritative dependency declaration is incomplete."""


def _canonical(value: Any, *, label: str, max_bytes: int = MAX_METADATA_BYTES) -> str:
    """Return bounded canonical JSON, rejecting payload-like values."""

    def validate(item: Any, path: str) -> Any:
        if isinstance(item, (bytes, bytearray, memoryview)):
            raise AuthoritativeDependencyError(
                f"{label}{path} contains binary payload; only metadata is allowed"
            )
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise AuthoritativeDependencyError(f"{label}{path} contains a non-finite number")
            return item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise AuthoritativeDependencyError(
                        f"{label}{path} has a non-string or empty metadata key"
                    )
                result[key] = validate(child, f"{path}.{key}")
            return result
        if isinstance(item, (list, tuple)):
            return [validate(child, f"{path}[{index}]") for index, child in enumerate(item)]
        raise AuthoritativeDependencyError(
            f"{label}{path} contains unsupported payload type {type(item).__name__}"
        )

    checked = validate(value, "")
    try:
        encoded = json.dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - validate is stricter
        raise AuthoritativeDependencyError(f"{label} is not canonical metadata") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise AuthoritativeDependencyError(f"{label} exceeds the {max_bytes}-byte metadata bound")
    return encoded


def _metadata(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise AuthoritativeDependencyError(f"{label} must be a metadata mapping")
    encoded = _canonical(value, label=label)
    return json.loads(encoded)


def _digest(value: Any, *, label: str) -> str:
    encoded = _canonical(value, label=label, max_bytes=MAX_METADATA_BYTES)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AuthoritativeDependencyError(f"{label} must be a string")
    if not allow_empty and not value:
        raise AuthoritativeDependencyError(f"{label} must not be empty")
    return value


def _strings(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise AuthoritativeDependencyError(f"{label} must be a sequence of strings")
    if not isinstance(value, Sequence):
        raise AuthoritativeDependencyError(f"{label} must be a sequence of strings")
    result = tuple(_text(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise AuthoritativeDependencyError(f"{label} contains duplicate entries")
    return result


def _optional_sha(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    text = _text(value, label=label).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AuthoritativeDependencyError(f"{label} must be a SHA-256 fingerprint")
    return text


def _positive_int(value: Any, *, label: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthoritativeDependencyError(f"{label} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        bound = "non-negative" if allow_zero else "positive"
        raise AuthoritativeDependencyError(f"{label} must be {bound}")
    return value


def _unique_tuple(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise AuthoritativeDependencyError(f"{label} contains duplicate entries")
    return result


def _forbidden_timing_metadata(value: Mapping[str, Any], *, label: str) -> None:
    def walk(item: Any, path: str) -> None:
        if not isinstance(item, Mapping):
            return
        for key, child in item.items():
            child_path = f"{path}.{key}"
            if key.lower() in _FORBIDDEN_TIMING_KEYS:
                raise AuthoritativeDependencyError(
                    f"{child_path} is a measured timing claim; timing is outside this contract"
                )
            walk(child, child_path)

    walk(value, label)


def _fingerprint_from_mapping(value: Mapping[str, Any], *, label: str) -> str | None:
    raw = value.get("fingerprint")
    if raw is None:
        return None
    return _optional_sha(raw, label=f"{label}.fingerprint")


@dataclass(frozen=True, slots=True)
class WeightPageRef:
    """A logical weight/page identity and its declared byte extent.

    ``byte_size`` is accounting metadata.  It is not a proof that bytes are
    resident, mapped, or fetched from a provider.
    """

    page_id: str
    byte_size: int
    weight_id: str | None = None
    storage_class: str = "weight-page"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        page_id = _text(self.page_id, label="weight page page_id")
        byte_size = _positive_int(self.byte_size, label=f"weight page {page_id}.byte_size")
        weight_id = (
            page_id
            if self.weight_id is None
            else _text(self.weight_id, label=f"weight page {page_id}.weight_id")
        )
        storage_class = _text(self.storage_class, label=f"weight page {page_id}.storage_class")
        metadata = _metadata(self.metadata, label=f"weight page {page_id}.metadata")
        _forbidden_timing_metadata(metadata, label=f"weight page {page_id}.metadata")
        object.__setattr__(self, "page_id", page_id)
        object.__setattr__(self, "byte_size", byte_size)
        object.__setattr__(self, "weight_id", weight_id)
        object.__setattr__(self, "storage_class", storage_class)
        object.__setattr__(self, "metadata", metadata)
        body = {
            "page_id": page_id,
            "byte_size": byte_size,
            "weight_id": weight_id,
            "storage_class": storage_class,
            "metadata": metadata,
        }
        object.__setattr__(self, "fingerprint", _digest(body, label="weight page"))

    @classmethod
    def from_value(cls, value: Any, *, label: str = "weight page") -> WeightPageRef:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise AuthoritativeDependencyError(f"{label} must be a mapping")
        page_id = value.get("page_id", value.get("resource_key", value.get("key", _MISSING)))
        if page_id is _MISSING:
            raise AuthoritativeDependencyError(f"{label} is missing page_id")
        byte_size = value.get("byte_size", value.get("bytes", value.get("size", _MISSING)))
        if byte_size is _MISSING:
            raise AuthoritativeDependencyError(
                f"{label} is missing byte_size; page admission cannot be inferred"
            )
        return cls(
            page_id=page_id,
            byte_size=byte_size,
            weight_id=value.get("weight_id"),
            storage_class=value.get("storage_class", "weight-page"),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "saturn-weight-page-ref-v1",
            "page_id": self.page_id,
            "byte_size": self.byte_size,
            "weight_id": self.weight_id,
            "storage_class": self.storage_class,
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WeightPageRef:
        if value.get("schema") not in (None, "saturn-weight-page-ref-v1"):
            raise AuthoritativeDependencyError("unsupported weight page schema")
        page = cls.from_value(value)
        declared = value.get("fingerprint")
        if (
            declared is not None
            and _optional_sha(declared, label="weight page fingerprint") != page.fingerprint
        ):
            raise AuthoritativeDependencyError("weight page fingerprint mismatch")
        return page


# A short alias is useful for callers that use the phrase used by the mrun
# compiler rather than the phrase used by Saturn's residency module.
WeightPage = WeightPageRef


def _weight_pages_from_weight_set(value: Any, *, label: str) -> tuple[WeightPageRef, ...]:
    """Adapt a CircuitWeightSet-shaped value without retaining its payload."""

    if hasattr(value, "to_dict") and not isinstance(value, Mapping):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            raise AuthoritativeDependencyError(
                f"{label} lists resource names without byte sizes; page admission is refused"
            )
        raise AuthoritativeDependencyError(f"{label} must be a weight-set mapping")
    resources = value.get("resources", _MISSING)
    if resources is _MISSING:
        raise AuthoritativeDependencyError(f"{label} is missing resources")
    weight_id = value.get("weight_set_id", value.get("weight_id"))
    result: list[WeightPageRef] = []
    if isinstance(resources, Mapping):
        items = list(resources.items())
        for key, descriptor in items:
            if isinstance(descriptor, Mapping):
                descriptor = dict(descriptor)
                descriptor.setdefault("page_id", key)
                if weight_id is not None:
                    descriptor.setdefault("weight_id", weight_id)
                result.append(
                    WeightPageRef.from_value(descriptor, label=f"{label}.resources[{key}]")
                )
            else:
                result.append(
                    WeightPageRef(
                        page_id=key,
                        byte_size=descriptor,
                        weight_id=weight_id,
                        storage_class="weight",
                    )
                )
    elif isinstance(resources, Sequence) and not isinstance(resources, (str, bytes, bytearray)):
        for index, descriptor in enumerate(resources):
            if isinstance(descriptor, Mapping):
                descriptor = dict(descriptor)
                if weight_id is not None:
                    descriptor.setdefault("weight_id", weight_id)
                result.append(
                    WeightPageRef.from_value(descriptor, label=f"{label}.resources[{index}]")
                )
            else:
                raise AuthoritativeDependencyError(
                    f"{label}.resources[{index}] is missing page byte_size"
                )
    else:
        raise AuthoritativeDependencyError(f"{label}.resources must be a mapping or sequence")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class DependencyDeclaration:
    """One node's authoritative reads, writes, invalidations, and cost."""

    node_id: str
    dependencies: Sequence[str] | None = None
    reads: Sequence[str] | None = None
    writes: Sequence[str] | None = None
    invalidates: Sequence[str] | None = None
    weight_pages: Sequence[WeightPageRef | Mapping[str, Any]] = field(default_factory=tuple)
    work_units: int = 0
    role: str = "cell"
    control_dependencies: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        node_id = _text(self.node_id, label="dependency node_id")
        if self.dependencies is None:
            raise AuthoritativeDependencyError(f"dependency {node_id} is missing dependencies")
        if self.reads is None:
            raise AuthoritativeDependencyError(f"dependency {node_id} is missing reads")
        if self.writes is None:
            raise AuthoritativeDependencyError(f"dependency {node_id} is missing writes")
        if self.invalidates is None:
            raise AuthoritativeDependencyError(f"dependency {node_id} is missing invalidates")
        dependencies = _strings(self.dependencies, label=f"dependency {node_id}.dependencies")
        reads = _strings(self.reads, label=f"dependency {node_id}.reads")
        writes = _strings(self.writes, label=f"dependency {node_id}.writes")
        invalidates = _strings(self.invalidates, label=f"dependency {node_id}.invalidates")
        controls = _strings(
            self.control_dependencies,
            label=f"dependency {node_id}.control_dependencies",
        )
        if self.weight_pages is None:
            raise AuthoritativeDependencyError(
                f"dependency {node_id} has an absent weight_pages declaration"
            )
        pages = tuple(
            WeightPageRef.from_value(page, label=f"dependency {node_id}.weight_pages[{index}]")
            for index, page in enumerate(self.weight_pages)
        )
        page_ids = tuple(page.page_id for page in pages)
        if len(set(page_ids)) != len(page_ids):
            raise AuthoritativeDependencyError(
                f"dependency {node_id}.weight_pages contains duplicate page IDs"
            )
        work_units = _positive_int(
            self.work_units,
            label=f"dependency {node_id}.work_units",
            allow_zero=True,
        )
        role = _text(self.role, label=f"dependency {node_id}.role")
        metadata = _metadata(self.metadata, label=f"dependency {node_id}.metadata")
        _forbidden_timing_metadata(metadata, label=f"dependency {node_id}.metadata")
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "invalidates", invalidates)
        object.__setattr__(self, "weight_pages", pages)
        object.__setattr__(self, "work_units", work_units)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "control_dependencies", controls)
        object.__setattr__(self, "metadata", metadata)
        body = self._body()
        object.__setattr__(self, "fingerprint", _digest(body, label=f"dependency {node_id}"))

    def _body(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "dependencies": list(self.dependencies),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "invalidates": list(self.invalidates),
            "weight_pages": [page.to_dict() for page in self.weight_pages],
            "work_units": self.work_units,
            "role": self.role,
            "control_dependencies": list(self.control_dependencies),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DependencyDeclaration:
        if not isinstance(value, Mapping):
            raise AuthoritativeDependencyError("dependency declaration must be a mapping")
        required = ("node_id", "dependencies", "reads", "writes", "invalidates")
        missing = [key for key in required if key not in value]
        if missing:
            raise AuthoritativeDependencyError(
                f"dependency declaration missing {', '.join(missing)}"
            )
        declaration = cls(
            node_id=value["node_id"],
            dependencies=value["dependencies"],
            reads=value["reads"],
            writes=value["writes"],
            invalidates=value["invalidates"],
            weight_pages=value.get("weight_pages", ()),
            work_units=value.get("work_units", value.get("physical_forward_units", 0)),
            role=value.get("role", "cell"),
            control_dependencies=value.get("control_dependencies", ()),
            metadata=value.get("metadata", {}),
        )
        declared = value.get("fingerprint")
        if (
            declared is not None
            and _optional_sha(declared, label="dependency fingerprint") != declaration.fingerprint
        ):
            raise AuthoritativeDependencyError(
                f"dependency {declaration.node_id} fingerprint mismatch"
            )
        return declaration

    @classmethod
    def from_cell(cls, cell: Any) -> DependencyDeclaration:
        """Read a CircuitCell-shaped object without importing its module.

        The attribute check is intentional: a cell that predates the
        declaration contract cannot become authoritative by implication.
        """

        if isinstance(cell, Mapping):
            raw = dict(cell)
            if "node_id" not in raw:
                if "cell_id" in raw:
                    raw["node_id"] = raw["cell_id"]
                else:
                    raise AuthoritativeDependencyError("cell mapping is missing cell_id/node_id")
            raw.setdefault("role", "cell")
            raw.setdefault("metadata", {})
            # A namespace proxy lets mapping-shaped cells use the same
            # embedded-declaration and weight-set conflict checks as objects.
            return cls.from_cell(SimpleNamespace(**raw))
        node_id = getattr(cell, "cell_id", getattr(cell, "node_id", _MISSING))
        if node_id is _MISSING:
            raise AuthoritativeDependencyError("cell is missing cell_id/node_id")
        node_id = _text(node_id, label="cell_id")
        required = ("dependencies", "reads", "writes", "invalidates")
        for key in required:
            if not hasattr(cell, key) or getattr(cell, key) is None:
                raise AuthoritativeDependencyError(f"cell {node_id} is missing {key}")
        cell_metadata = getattr(cell, "metadata", {})
        cell_metadata = _metadata(cell_metadata, label=f"cell {node_id}.metadata")
        declaration_metadata = dict(cell_metadata)
        declaration_metadata.pop("dependency_declaration", None)
        direct = {
            "node_id": node_id,
            "dependencies": cell.dependencies,
            "reads": cell.reads,
            "writes": cell.writes,
            "invalidates": cell.invalidates,
            "work_units": getattr(cell, "physical_forward_units", getattr(cell, "work_units", 0)),
            "role": getattr(cell, "role", "cell"),
            "control_dependencies": cell_metadata.get("control_dependencies", ()),
            "metadata": declaration_metadata,
        }
        raw_pages: Any = getattr(cell, "weight_pages", _MISSING)
        if raw_pages is _MISSING:
            raw_pages = cell_metadata.get("weight_pages", _MISSING)
        if raw_pages is not _MISSING:
            direct["weight_pages"] = raw_pages
        raw_weight_set = cell_metadata.get("weight_set", _MISSING)
        if raw_weight_set is not _MISSING:
            weight_pages = _weight_pages_from_weight_set(
                raw_weight_set, label=f"cell {node_id}.metadata.weight_set"
            )
            if "weight_pages" in direct:
                explicit = tuple(
                    WeightPageRef.from_value(page, label=f"cell {node_id}.weight_pages[{index}]")
                    for index, page in enumerate(direct["weight_pages"])
                )
                if explicit != weight_pages:
                    raise AuthoritativeDependencyError(
                        f"cell {node_id} has conflicting weight_pages and weight_set declarations"
                    )
            direct["weight_pages"] = weight_pages
        embedded = cell_metadata.get("dependency_declaration", _MISSING)
        declaration = cls(**direct)
        if embedded is not _MISSING:
            if not isinstance(embedded, Mapping):
                raise AuthoritativeDependencyError(
                    f"cell {node_id}.metadata.dependency_declaration must be a mapping"
                )
            embedded_declaration = cls.from_dict(embedded)
            direct_core = declaration._body().copy()
            embedded_core = embedded_declaration._body().copy()
            # The wrapper and typed weight-set adapters are not part of the
            # declaration they wrap. Compare all authoritative fields and the
            # remaining metadata, so a conflicting declaration cannot hide
            # behind a cell-local adapter key.
            adapter_keys = {
                "control_dependencies",
                "dependency_declaration",
                "weight_set",
                "weight_pages",
            }
            direct_core["metadata"] = {
                key: value
                for key, value in direct_core["metadata"].items()
                if key not in adapter_keys
            }
            if embedded_core != direct_core:
                raise AuthoritativeDependencyError(
                    f"cell {node_id} has conflicting dependency declarations"
                )
        return declaration

    def to_dict(self) -> dict[str, Any]:
        value = self._body()
        value["schema"] = "saturn-dependency-declaration-v1"
        value["fingerprint"] = self.fingerprint
        return value


DependencyNode = DependencyDeclaration


@dataclass(frozen=True, slots=True)
class DependencyContract:
    """A validated, metadata-only graph dependency contract."""

    contract_id: str
    nodes: Sequence[DependencyDeclaration]
    outputs: Sequence[str] = field(default_factory=tuple)
    graph_fingerprint: str | None = None
    skeleton_fingerprint: str | None = None
    specimen_fingerprint: str | None = None
    provider_fingerprint: str | None = None
    input_addresses: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)
    _node_map: dict[str, DependencyDeclaration] = field(init=False, repr=False, compare=False)
    _descendants: dict[str, tuple[str, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        contract_id = _text(self.contract_id, label="dependency contract_id")
        nodes = tuple(self.nodes)
        if not nodes:
            raise AuthoritativeDependencyError("dependency contract must contain at least one node")
        if any(not isinstance(node, DependencyDeclaration) for node in nodes):
            raise AuthoritativeDependencyError("dependency contract nodes must be declarations")
        node_ids = tuple(node.node_id for node in nodes)
        if len(set(node_ids)) != len(node_ids):
            raise AuthoritativeDependencyError("dependency contract contains duplicate node IDs")
        node_map = {node.node_id: node for node in nodes}
        outputs = _strings(self.outputs, label="dependency contract.outputs")
        unknown_outputs = sorted(set(outputs) - set(node_ids))
        if unknown_outputs:
            raise AuthoritativeDependencyError(
                f"dependency contract outputs reference unknown nodes: {unknown_outputs}"
            )
        input_addresses = _strings(
            self.input_addresses,
            label="dependency contract.input_addresses",
        )
        graph_fingerprint = _optional_sha(
            self.graph_fingerprint, label="dependency contract.graph_fingerprint"
        )
        skeleton_fingerprint = _optional_sha(
            self.skeleton_fingerprint, label="dependency contract.skeleton_fingerprint"
        )
        specimen_fingerprint = _optional_sha(
            self.specimen_fingerprint, label="dependency contract.specimen_fingerprint"
        )
        provider_fingerprint = _optional_sha(
            self.provider_fingerprint, label="dependency contract.provider_fingerprint"
        )
        metadata = _metadata(self.metadata, label="dependency contract.metadata")
        _forbidden_timing_metadata(metadata, label="dependency contract.metadata")
        _validate_topology(nodes, node_map)
        descendants = _build_descendants(nodes)
        _validate_addresses(
            nodes,
            node_map=node_map,
            input_addresses=set(input_addresses),
        )
        page_declarations: dict[str, WeightPageRef] = {}
        for node in nodes:
            for page in node.weight_pages:
                previous = page_declarations.get(page.page_id)
                if previous is not None and previous != page:
                    raise AuthoritativeDependencyError(
                        f"page {page.page_id} has conflicting declarations"
                    )
                page_declarations[page.page_id] = page
        object.__setattr__(self, "contract_id", contract_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "graph_fingerprint", graph_fingerprint)
        object.__setattr__(self, "skeleton_fingerprint", skeleton_fingerprint)
        object.__setattr__(self, "specimen_fingerprint", specimen_fingerprint)
        object.__setattr__(self, "provider_fingerprint", provider_fingerprint)
        object.__setattr__(self, "input_addresses", input_addresses)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_node_map", node_map)
        object.__setattr__(self, "_descendants", descendants)
        object.__setattr__(self, "fingerprint", _digest(self._body(), label="dependency contract"))

    def _body(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "outputs": list(self.outputs),
            "graph_fingerprint": self.graph_fingerprint,
            "skeleton_fingerprint": self.skeleton_fingerprint,
            "specimen_fingerprint": self.specimen_fingerprint,
            "provider_fingerprint": self.provider_fingerprint,
            "input_addresses": list(self.input_addresses),
            "metadata": dict(self.metadata),
        }

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    @property
    def declarations(self) -> tuple[DependencyDeclaration, ...]:
        return tuple(self.nodes)

    @property
    def page_map(self) -> dict[str, WeightPageRef]:
        result: dict[str, WeightPageRef] = {}
        for node in self.nodes:
            for page in node.weight_pages:
                result[page.page_id] = page
        return result

    @property
    def address_set(self) -> frozenset[str]:
        values: set[str] = set(self.input_addresses)
        for node in self.nodes:
            values.update(node.reads)
            values.update(node.writes)
            values.update(node.invalidates)
        return frozenset(values)

    def descendants_of(self, node_id: str) -> tuple[str, ...]:
        node_id = _text(node_id, label="node_id")
        if node_id not in self._node_map:
            raise AuthoritativeDependencyError(f"unknown node {node_id}")
        return self._descendants[node_id]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DependencyContract:
        if not isinstance(value, Mapping):
            raise AuthoritativeDependencyError("dependency contract must be a mapping")
        schema = value.get("schema")
        if schema not in (None, AUTHORITATIVE_DEPENDENCY_SCHEMA):
            raise AuthoritativeDependencyError(f"unsupported dependency contract schema {schema}")
        raw_nodes = value.get("nodes", _MISSING)
        if raw_nodes is _MISSING:
            raise AuthoritativeDependencyError("dependency contract is missing nodes")
        contract = cls(
            contract_id=value.get("contract_id", _MISSING),
            nodes=tuple(DependencyDeclaration.from_dict(node) for node in raw_nodes),
            outputs=value.get("outputs", ()),
            graph_fingerprint=value.get("graph_fingerprint"),
            skeleton_fingerprint=value.get("skeleton_fingerprint"),
            specimen_fingerprint=value.get("specimen_fingerprint"),
            provider_fingerprint=value.get("provider_fingerprint"),
            input_addresses=value.get("input_addresses", ()),
            metadata=value.get("metadata", {}),
        )
        declared = value.get("fingerprint")
        if (
            declared is not None
            and _optional_sha(declared, label="dependency contract fingerprint")
            != contract.fingerprint
        ):
            raise AuthoritativeDependencyError("dependency contract fingerprint mismatch")
        return contract

    @classmethod
    def from_graph(
        cls,
        graph: Any,
        *,
        provider: Any | None = None,
        input_addresses: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> DependencyContract:
        """Build a contract from a ``CircuitBuildGraph``-shaped object."""

        cells = getattr(graph, "cells", _MISSING)
        if cells is _MISSING and isinstance(graph, Mapping):
            cells = graph.get("cells", _MISSING)
        if cells is _MISSING:
            raise AuthoritativeDependencyError("graph is missing cells")
        declarations = tuple(DependencyDeclaration.from_cell(cell) for cell in cells)
        graph_id = getattr(graph, "graph_id", _MISSING)
        if graph_id is _MISSING and isinstance(graph, Mapping):
            graph_id = graph.get("graph_id", graph.get("contract_id", _MISSING))
        if graph_id is _MISSING:
            raise AuthoritativeDependencyError("graph is missing graph_id")
        outputs = getattr(graph, "outputs", _MISSING)
        if outputs is _MISSING and isinstance(graph, Mapping):
            outputs = graph.get("outputs", ())
        if outputs is _MISSING:
            outputs = ()
        graph_fp = getattr(graph, "fingerprint", _MISSING)
        if graph_fp is _MISSING and isinstance(graph, Mapping):
            graph_fp = graph.get("fingerprint", graph.get("graph_fingerprint"))
        if graph_fp is _MISSING:
            graph_fp = None
        skeleton_fp = getattr(graph, "skeleton_fingerprint", _MISSING)
        if skeleton_fp is _MISSING and isinstance(graph, Mapping):
            skeleton_fp = graph.get("skeleton_fingerprint")
        if skeleton_fp is _MISSING:
            skeleton_fp = None
        specimen_fp = getattr(graph, "specimen_fingerprint", _MISSING)
        if specimen_fp is _MISSING and isinstance(graph, Mapping):
            specimen_fp = graph.get("specimen_fingerprint")
        provider_fp = getattr(graph, "resource_provider_fingerprint", _MISSING)
        if provider_fp is _MISSING and isinstance(graph, Mapping):
            provider_fp = graph.get(
                "resource_provider_fingerprint", graph.get("provider_fingerprint")
            )
        if provider_fp is _MISSING:
            provider_fp = None
        provider_observed = _provider_fingerprint(provider)
        if (
            provider_fp is not None
            and provider_observed is not None
            and provider_fp != provider_observed
        ):
            raise AuthoritativeDependencyError(
                "graph/provider resource provider fingerprints conflict"
            )
        if provider_fp is None:
            provider_fp = provider_observed
        graph_metadata = getattr(graph, "metadata", {})
        if graph_metadata is _MISSING and isinstance(graph, Mapping):
            graph_metadata = graph.get("metadata", {})
        merged_metadata = dict(_metadata(graph_metadata, label="graph.metadata"))
        if metadata:
            merged_metadata.update(_metadata(metadata, label="dependency contract.metadata"))
        declared_inputs = input_addresses
        if not declared_inputs:
            declared_inputs = getattr(graph, "input_addresses", _MISSING)
            if declared_inputs is _MISSING and isinstance(graph, Mapping):
                declared_inputs = graph.get("input_addresses", _MISSING)
            if declared_inputs is _MISSING:
                declared_inputs = merged_metadata.get("input_addresses", ())
        return cls(
            contract_id=graph_id,
            nodes=declarations,
            outputs=outputs,
            graph_fingerprint=graph_fp,
            skeleton_fingerprint=skeleton_fp,
            specimen_fingerprint=specimen_fp,
            provider_fingerprint=provider_fp,
            input_addresses=declared_inputs,
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        value = self._body()
        value["schema"] = AUTHORITATIVE_DEPENDENCY_SCHEMA
        value["fingerprint"] = self.fingerprint
        return value


AuthoritativeDependencyContract = DependencyContract
AuthoritativeDependencyGraph = DependencyContract


def _validate_topology(
    nodes: Sequence[DependencyDeclaration],
    node_map: Mapping[str, DependencyDeclaration],
) -> None:
    positions = {node.node_id: index for index, node in enumerate(nodes)}
    for node in nodes:
        for dependency in node.dependencies:
            if dependency not in node_map:
                raise AuthoritativeDependencyError(
                    f"dependency {node.node_id} references unknown node {dependency}"
                )
            if positions[dependency] >= positions[node.node_id]:
                raise AuthoritativeDependencyError(
                    f"dependency {node.node_id} is not topologically ordered after {dependency}"
                )
        unknown_controls = set(node.control_dependencies) - set(node.dependencies)
        if unknown_controls:
            raise AuthoritativeDependencyError(
                f"dependency {node.node_id}.control_dependencies are not dependencies: "
                f"{sorted(unknown_controls)}"
            )
        for dependency in node.dependencies:
            parent = node_map[dependency]
            data_edge = bool(set(parent.writes) & set(node.reads))
            control_edge = dependency in node.control_dependencies
            if not data_edge and not control_edge:
                raise AuthoritativeDependencyError(
                    f"dependency edge {dependency}->{node.node_id} has no declared data or "
                    "control reason"
                )


def _build_descendants(nodes: Sequence[DependencyDeclaration]) -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dependency in node.dependencies:
            children[dependency].append(node.node_id)
    result: dict[str, tuple[str, ...]] = {}
    for node in nodes:
        seen: set[str] = set()
        queue = deque(children.get(node.node_id, ()))
        while queue:
            child = queue.popleft()
            if child in seen:
                continue
            seen.add(child)
            queue.extend(children.get(child, ()))
        order = {candidate.node_id: index for index, candidate in enumerate(nodes)}
        result[node.node_id] = tuple(sorted(seen, key=order.__getitem__))
    return result


def _has_path_to(
    node_map: Mapping[str, DependencyDeclaration], ancestor: str, node_id: str
) -> bool:
    if ancestor == node_id:
        return True
    queue = deque(node_map[node_id].dependencies)
    seen: set[str] = set()
    while queue:
        candidate = queue.popleft()
        if candidate == ancestor:
            return True
        if candidate in seen:
            continue
        seen.add(candidate)
        queue.extend(node_map[candidate].dependencies)
    return False


def _validate_addresses(
    nodes: Sequence[DependencyDeclaration],
    *,
    node_map: Mapping[str, DependencyDeclaration],
    input_addresses: set[str],
) -> None:
    address_users: dict[str, set[str]] = defaultdict(set)
    writers: dict[str, str] = {}
    declared_data_addresses: set[str] = set(input_addresses)
    for node in nodes:
        declared_data_addresses.update(node.reads)
        declared_data_addresses.update(node.writes)
    for node in nodes:
        for address in node.reads:
            address_users[address].add(node.node_id)
        for address in node.writes:
            address_users[address].add(node.node_id)
            if address in input_addresses:
                raise AuthoritativeDependencyError(
                    f"address {address} is both an external input and a write"
                )
            previous = writers.get(address)
            if previous is not None and previous != node.node_id:
                raise AuthoritativeDependencyError(
                    f"address {address} has conflicting writers {previous} and {node.node_id}"
                )
            writers[address] = node.node_id
        for address in node.invalidates:
            address_users[address].add(node.node_id)
            if address not in declared_data_addresses:
                raise AuthoritativeDependencyError(
                    f"invalidated address {address} has no read/write or external-input declaration"
                )
            if address not in node.writes and not node.metadata.get(
                "invalidation_authority", False
            ):
                raise AuthoritativeDependencyError(
                    f"node {node.node_id} invalidates {address} without writing it or declaring "
                    "invalidation_authority"
                )
    for node in nodes:
        raw_read_modify_write = node.metadata.get("read_modify_write", ())
        read_modify_write = set(
            _strings(
                raw_read_modify_write,
                label=f"node {node.node_id}.metadata.read_modify_write",
            )
        )
        if not read_modify_write.issubset(set(node.reads) & set(node.writes)):
            raise AuthoritativeDependencyError(
                f"node {node.node_id}.read_modify_write must name read/write addresses"
            )
        for address in node.reads:
            writer = writers.get(address)
            if writer is None:
                if address not in input_addresses:
                    raise AuthoritativeDependencyError(
                        f"read address {address} has no writer or declared external input"
                    )
            elif writer == node.node_id:
                if address not in read_modify_write:
                    raise AuthoritativeDependencyError(
                        f"node {node.node_id} reads its own write {address} without "
                        "read_modify_write"
                    )
            elif not _has_path_to(node_map, writer, node.node_id):
                raise AuthoritativeDependencyError(
                    f"read address {address} on {node.node_id} is not ordered after writer {writer}"
                )
    for node in nodes:
        for address in node.invalidates:
            if address not in address_users:
                raise AuthoritativeDependencyError(
                    f"invalidated address {address} has no declaration"
                )


def _provider_fingerprint(provider: Any | None) -> str | None:
    if provider is None:
        return None
    value = getattr(provider, "provider_fingerprint", None)
    if value is None:
        value = getattr(provider, "resource_provider_fingerprint", None)
    return _optional_sha(value, label="provider fingerprint") if value is not None else None


def _state_cut_fingerprint(value: Any, *, label: str) -> str:
    if isinstance(value, str):
        result = _optional_sha(value, label=label)
        if result is None:  # pragma: no cover - str branch cannot be None
            raise AuthoritativeDependencyError(f"{label} is required")
        return result
    if isinstance(value, Mapping):
        raw = value.get("fingerprint")
    else:
        raw = getattr(value, "fingerprint", None)
    result = _optional_sha(raw, label=label)
    if result is None:
        raise AuthoritativeDependencyError(
            f"{label} must be a StateCut fingerprint; tensor/state payload is not accepted"
        )
    return result


@dataclass(frozen=True, slots=True)
class DirtyClosure:
    """Conservative weighted closure for one change set."""

    contract_fingerprint: str
    changed_cells: tuple[str, ...]
    changed_addresses: tuple[str, ...]
    dirty_cells: tuple[str, ...]
    reused_cells: tuple[str, ...]
    invalidated_addresses: tuple[str, ...]
    dirty_page_ids: tuple[str, ...]
    baseline_page_ids: tuple[str, ...]
    dirty_weight_bytes: int
    baseline_weight_bytes: int
    reused_weight_bytes: int
    avoided_weight_bytes: int
    dirty_work_units: int
    baseline_work_units: int
    reused_work_units: int
    avoided_work_units: int
    mechanics_status: str = "valid"
    instrument_status: str = "metadata-only"
    trend_status: str = "not-assessed"
    terminal_status: str = "not-established"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "contract_fingerprint",
            "dirty_page_ids",
            "baseline_page_ids",
            "changed_cells",
            "changed_addresses",
            "dirty_cells",
            "reused_cells",
            "invalidated_addresses",
        ):
            values = getattr(self, field_name)
            if field_name == "contract_fingerprint":
                _optional_sha(values, label="dirty closure.contract_fingerprint")
            else:
                _unique_tuple(values, label=f"dirty closure.{field_name}")
        for field_name in (
            "dirty_weight_bytes",
            "baseline_weight_bytes",
            "reused_weight_bytes",
            "avoided_weight_bytes",
            "dirty_work_units",
            "baseline_work_units",
            "reused_work_units",
            "avoided_work_units",
        ):
            _positive_int(
                getattr(self, field_name), label=f"dirty closure.{field_name}", allow_zero=True
            )
        if self.dirty_work_units + self.reused_work_units != self.baseline_work_units:
            raise AuthoritativeDependencyError(
                "dirty closure work accounting does not partition the baseline"
            )
        if self.dirty_weight_bytes + self.reused_weight_bytes != self.baseline_weight_bytes:
            raise AuthoritativeDependencyError(
                "dirty closure page accounting does not partition the baseline"
            )
        if self.avoided_work_units != self.baseline_work_units - self.dirty_work_units:
            raise AuthoritativeDependencyError("dirty closure avoided work accounting mismatch")
        if self.avoided_weight_bytes != self.baseline_weight_bytes - self.dirty_weight_bytes:
            raise AuthoritativeDependencyError("dirty closure avoided page accounting mismatch")
        body = self.to_dict(include_fingerprint=False)
        object.__setattr__(self, "fingerprint", _digest(body, label="dirty closure"))

    @property
    def weighted_dirty_units(self) -> int:
        return self.dirty_work_units

    @property
    def weight_page_set(self) -> tuple[str, ...]:
        return self.dirty_page_ids

    @property
    def dirty_weight_pages(self) -> tuple[str, ...]:
        return self.dirty_page_ids

    @property
    def admitted_page_ids(self) -> tuple[str, ...]:
        return self.dirty_page_ids

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": "saturn-dirty-closure-v1",
            "contract_fingerprint": self.contract_fingerprint,
            "changed_cells": list(self.changed_cells),
            "changed_addresses": list(self.changed_addresses),
            "dirty_cells": list(self.dirty_cells),
            "reused_cells": list(self.reused_cells),
            "invalidated_addresses": list(self.invalidated_addresses),
            "dirty_page_ids": list(self.dirty_page_ids),
            "baseline_page_ids": list(self.baseline_page_ids),
            "accounting": {
                "dirty_weight_bytes": self.dirty_weight_bytes,
                "baseline_weight_bytes": self.baseline_weight_bytes,
                "reused_weight_bytes": self.reused_weight_bytes,
                "avoided_weight_bytes": self.avoided_weight_bytes,
                "dirty_work_units": self.dirty_work_units,
                "baseline_work_units": self.baseline_work_units,
                "reused_work_units": self.reused_work_units,
                "avoided_work_units": self.avoided_work_units,
                "basis": "declared-logical-work-and-page-bytes",
            },
            "statuses": {
                "mechanics_status": self.mechanics_status,
                "instrument_status": self.instrument_status,
                "trend_status": self.trend_status,
                "terminal_status": self.terminal_status,
            },
            "speedup_claim": "not-claimed",
            "physical_residency_proven": False,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


DirtyCone = DirtyClosure


def compute_dirty_closure(
    contract: DependencyContract,
    *,
    changed_cells: Sequence[str] = (),
    changed_addresses: Sequence[str] = (),
) -> DirtyClosure:
    """Compute direct changes plus all authoritative descendants."""

    if not isinstance(contract, DependencyContract):
        raise AuthoritativeDependencyError("compute_dirty_closure requires a DependencyContract")
    changed_cells = _strings(changed_cells, label="changed_cells")
    changed_addresses = _strings(changed_addresses, label="changed_addresses")
    unknown_cells = sorted(set(changed_cells) - set(contract.node_ids))
    if unknown_cells:
        raise AuthoritativeDependencyError(
            f"changed_cells reference unknown nodes: {unknown_cells}"
        )
    unknown_addresses = sorted(set(changed_addresses) - set(contract.address_set))
    if unknown_addresses:
        raise AuthoritativeDependencyError(
            f"changed_addresses lack dependency declarations: {unknown_addresses}"
        )
    dirty: set[str] = set(changed_cells)
    invalidated: set[str] = set()
    for node in contract.nodes:
        touched = set(node.reads) | set(node.writes) | set(node.invalidates)
        hit = touched & set(changed_addresses)
        if hit:
            dirty.add(node.node_id)
            invalidated.update(hit & (set(node.invalidates) | set(node.writes)))
    queue = deque(dirty)
    while queue:
        node_id = queue.popleft()
        for descendant in contract.descendants_of(node_id):
            if descendant not in dirty:
                dirty.add(descendant)
                queue.append(descendant)
    for node in contract.nodes:
        if node.node_id in dirty:
            invalidated.update(node.invalidates)
    dirty_cells = tuple(node.node_id for node in contract.nodes if node.node_id in dirty)
    reused_cells = tuple(node.node_id for node in contract.nodes if node.node_id not in dirty)
    pages = contract.page_map
    dirty_page_ids = tuple(
        page_id
        for page_id in pages
        if any(
            page_id == page.page_id
            for node in contract.nodes
            if node.node_id in dirty
            for page in node.weight_pages
        )
    )
    baseline_page_ids = tuple(pages)
    dirty_weight_bytes = sum(pages[page_id].byte_size for page_id in dirty_page_ids)
    baseline_weight_bytes = sum(page.byte_size for page in pages.values())
    reused_weight_bytes = baseline_weight_bytes - dirty_weight_bytes
    dirty_work_units = sum(node.work_units for node in contract.nodes if node.node_id in dirty)
    baseline_work_units = sum(node.work_units for node in contract.nodes)
    reused_work_units = baseline_work_units - dirty_work_units
    return DirtyClosure(
        contract_fingerprint=contract.fingerprint,
        changed_cells=changed_cells,
        changed_addresses=changed_addresses,
        dirty_cells=dirty_cells,
        reused_cells=reused_cells,
        invalidated_addresses=tuple(sorted(invalidated)),
        dirty_page_ids=dirty_page_ids,
        baseline_page_ids=baseline_page_ids,
        dirty_weight_bytes=dirty_weight_bytes,
        baseline_weight_bytes=baseline_weight_bytes,
        reused_weight_bytes=reused_weight_bytes,
        avoided_weight_bytes=reused_weight_bytes,
        dirty_work_units=dirty_work_units,
        baseline_work_units=baseline_work_units,
        reused_work_units=reused_work_units,
        avoided_work_units=reused_work_units,
    )


plan_dirty_cone = compute_dirty_closure


@dataclass(frozen=True, slots=True)
class WeightPageAdmission:
    """A declared page admission, independent of physical residency."""

    contract_fingerprint: str
    requested_page_ids: tuple[str, ...]
    admitted_page_ids: tuple[str, ...]
    weight_ids: tuple[str, ...]
    total_bytes: int
    byte_budget: int | None
    provider_fingerprint: str | None
    availability_status: str
    mechanics_status: str
    instrument_status: str
    trend_status: str
    terminal_status: str
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _optional_sha(self.contract_fingerprint, label="page admission.contract_fingerprint")
        _unique_tuple(self.requested_page_ids, label="page admission.requested_page_ids")
        _unique_tuple(self.admitted_page_ids, label="page admission.admitted_page_ids")
        _unique_tuple(self.weight_ids, label="page admission.weight_ids")
        if self.requested_page_ids != self.admitted_page_ids:
            raise AuthoritativeDependencyError(
                "page admission must fail closed instead of partially admitting pages"
            )
        _positive_int(self.total_bytes, label="page admission.total_bytes", allow_zero=True)
        if self.byte_budget is not None:
            _positive_int(self.byte_budget, label="page admission.byte_budget", allow_zero=True)
            if self.total_bytes > self.byte_budget:
                raise AuthoritativeDependencyError("page admission exceeds byte budget")
        _optional_sha(self.provider_fingerprint, label="page admission.provider_fingerprint")
        body = self.to_dict(include_fingerprint=False)
        object.__setattr__(self, "fingerprint", _digest(body, label="page admission"))

    @property
    def page_ids(self) -> tuple[str, ...]:
        return self.admitted_page_ids

    @property
    def admitted_weight_pages(self) -> tuple[str, ...]:
        return self.admitted_page_ids

    @property
    def admitted_pages(self) -> tuple[str, ...]:
        return self.admitted_page_ids

    @property
    def admitted_weights(self) -> tuple[str, ...]:
        return self.weight_ids

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": AUTHORITATIVE_ADMISSION_SCHEMA,
            "contract_fingerprint": self.contract_fingerprint,
            "requested_page_ids": list(self.requested_page_ids),
            "admitted_page_ids": list(self.admitted_page_ids),
            "weight_ids": list(self.weight_ids),
            "total_bytes": self.total_bytes,
            "byte_budget": self.byte_budget,
            "provider_fingerprint": self.provider_fingerprint,
            "availability_status": self.availability_status,
            "statuses": {
                "mechanics_status": self.mechanics_status,
                "instrument_status": self.instrument_status,
                "trend_status": self.trend_status,
                "terminal_status": self.terminal_status,
            },
            "physical_residency_proven": False,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WeightPageAdmission:
        """Hydrate one content-addressed page admission.

        Runtime adapters receive plans from either an in-process compiler or a
        serialized custody artifact.  Keep the latter on the same fail-closed
        path as the former instead of trusting a page list merely because it
        has the right shape.
        """

        if not isinstance(value, Mapping):
            raise AuthoritativeDependencyError("page admission must be a mapping")
        statuses = value.get("statuses", {})
        if not isinstance(statuses, Mapping):
            raise AuthoritativeDependencyError("page admission.statuses must be a mapping")
        result = cls(
            contract_fingerprint=value.get("contract_fingerprint"),
            requested_page_ids=tuple(value.get("requested_page_ids", ())),
            admitted_page_ids=tuple(value.get("admitted_page_ids", ())),
            weight_ids=tuple(value.get("weight_ids", ())),
            total_bytes=value.get("total_bytes"),
            byte_budget=value.get("byte_budget"),
            provider_fingerprint=value.get("provider_fingerprint"),
            availability_status=value.get("availability_status", "declared-only"),
            mechanics_status=statuses.get("mechanics_status", "valid"),
            instrument_status=statuses.get("instrument_status", "metadata-only"),
            trend_status=statuses.get("trend_status", "not-assessed"),
            terminal_status=statuses.get("terminal_status", "not-established"),
        )
        if value.get("schema") not in (None, AUTHORITATIVE_ADMISSION_SCHEMA):
            raise AuthoritativeDependencyError("unsupported page admission schema")
        declared = value.get("fingerprint")
        if (
            declared is not None
            and _optional_sha(declared, label="page admission fingerprint") != result.fingerprint
        ):
            raise AuthoritativeDependencyError("page admission fingerprint mismatch")
        return result


def _normalise_available_pages(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise AuthoritativeDependencyError(f"{label} must be a page inventory mapping")
    result: dict[str, int] = {}
    for page_id, descriptor in value.items():
        page_id = _text(page_id, label=f"{label} page_id")
        if isinstance(descriptor, WeightPageRef):
            size = descriptor.byte_size
            if descriptor.page_id != page_id:
                raise AuthoritativeDependencyError(f"{label} key/page_id conflict for {page_id}")
        elif isinstance(descriptor, Mapping):
            page = WeightPageRef.from_value(
                dict(descriptor, page_id=descriptor.get("page_id", page_id)),
                label=f"{label}[{page_id}]",
            )
            if page.page_id != page_id:
                raise AuthoritativeDependencyError(f"{label} key/page_id conflict for {page_id}")
            size = page.byte_size
        else:
            size = _positive_int(descriptor, label=f"{label}[{page_id}]")
        result[page_id] = size
    return result


def _provider_page_inventory(provider: Any | None) -> Mapping[str, Any] | None:
    if provider is None:
        return None
    for name in ("page_inventory", "resource_inventory", "pages", "manifest"):
        value = getattr(provider, name, None)
        if isinstance(value, Mapping):
            return value
    return None


def admit_weight_pages(
    contract: DependencyContract,
    page_ids: Sequence[str],
    *,
    provider: Any | None = None,
    available_pages: Mapping[str, Any] | None = None,
    byte_budget: int | None = None,
) -> WeightPageAdmission:
    """Admit a complete declared page set, or raise without partial admission."""

    if not isinstance(contract, DependencyContract):
        raise AuthoritativeDependencyError("admit_weight_pages requires a DependencyContract")
    requested = _strings(page_ids, label="page_ids")
    page_map = contract.page_map
    unknown = sorted(set(requested) - set(page_map))
    if unknown:
        raise AuthoritativeDependencyError(
            f"requested pages lack authoritative declarations: {unknown}"
        )
    expected_provider = contract.provider_fingerprint
    observed_provider = _provider_fingerprint(provider)
    if expected_provider is not None and provider is None:
        raise AuthoritativeDependencyError(
            "contract binds a provider fingerprint but no provider was supplied"
        )
    if expected_provider is not None and observed_provider != expected_provider:
        raise AuthoritativeDependencyError("provider fingerprint does not match contract")
    if expected_provider is None and observed_provider is not None:
        expected_provider = observed_provider
    inventory = (
        _normalise_available_pages(available_pages, label="available_pages")
        if available_pages is not None
        else _provider_page_inventory(provider)
    )
    availability_status = "declared-only"
    if inventory is not None:
        inventory = _normalise_available_pages(inventory, label="provider page inventory")
        missing = sorted(set(requested) - set(inventory))
        if missing:
            raise AuthoritativeDependencyError(
                f"provider inventory is missing requested pages: {missing}"
            )
        conflicts = [
            page_id for page_id in requested if inventory[page_id] != page_map[page_id].byte_size
        ]
        if conflicts:
            raise AuthoritativeDependencyError(
                f"provider/page byte declarations conflict for: {sorted(conflicts)}"
            )
        availability_status = "inventory-bound"
    total_bytes = sum(page_map[page_id].byte_size for page_id in requested)
    if byte_budget is not None:
        _positive_int(byte_budget, label="byte_budget", allow_zero=True)
        if total_bytes > byte_budget:
            raise AuthoritativeDependencyError(
                f"requested page set is {total_bytes} bytes, over budget {byte_budget}"
            )
    weight_ids = tuple(dict.fromkeys(page_map[page_id].weight_id for page_id in requested))
    return WeightPageAdmission(
        contract_fingerprint=contract.fingerprint,
        requested_page_ids=requested,
        admitted_page_ids=requested,
        weight_ids=weight_ids,
        total_bytes=total_bytes,
        byte_budget=byte_budget,
        provider_fingerprint=expected_provider,
        availability_status=availability_status,
        mechanics_status="valid",
        instrument_status="inventory-bound" if inventory is not None else "metadata-only",
        trend_status="not-assessed",
        terminal_status="not-established",
    )


@dataclass(frozen=True, slots=True)
class BranchRequest:
    """A branch request carrying only a parent StateCut identity."""

    branch_id: str
    parent_state_cut: Any | None = None
    changed_cells: Sequence[str] = field(default_factory=tuple)
    changed_addresses: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parent_state_cut_fingerprint: str | None = None

    def __post_init__(self) -> None:
        branch_id = _text(self.branch_id, label="branch_id")
        parent_fp = (
            _state_cut_fingerprint(
                self.parent_state_cut,
                label=f"branch {branch_id}.parent_state_cut",
            )
            if self.parent_state_cut is not None
            else None
        )
        declared_parent_fp = _optional_sha(
            self.parent_state_cut_fingerprint,
            label=f"branch {branch_id}.parent_state_cut_fingerprint",
        )
        if (
            parent_fp is not None
            and declared_parent_fp is not None
            and parent_fp != declared_parent_fp
        ):
            raise AuthoritativeDependencyError(
                f"branch {branch_id} has conflicting StateCut identities"
            )
        parent_fp = parent_fp or declared_parent_fp
        if parent_fp is None:
            raise AuthoritativeDependencyError(
                f"branch {branch_id} requires a parent StateCut fingerprint"
            )
        changed_cells = _strings(self.changed_cells, label=f"branch {branch_id}.changed_cells")
        changed_addresses = _strings(
            self.changed_addresses,
            label=f"branch {branch_id}.changed_addresses",
        )
        metadata = _metadata(self.metadata, label=f"branch {branch_id}.metadata")
        _forbidden_timing_metadata(metadata, label=f"branch {branch_id}.metadata")
        object.__setattr__(self, "branch_id", branch_id)
        object.__setattr__(self, "parent_state_cut_fingerprint", parent_fp)
        object.__setattr__(self, "changed_cells", changed_cells)
        object.__setattr__(self, "changed_addresses", changed_addresses)
        object.__setattr__(self, "metadata", metadata)
        # Do not retain parent_state_cut: it may contain tensors.  The
        # dataclass field is replaced by the identity string below.
        object.__setattr__(self, "parent_state_cut", parent_fp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "saturn-branch-request-v1",
            "branch_id": self.branch_id,
            "parent_state_cut_fingerprint": self.parent_state_cut_fingerprint,
            "changed_cells": list(self.changed_cells),
            "changed_addresses": list(self.changed_addresses),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CompiledBranch:
    """One branch's dirty cone and conservative execution suffix."""

    branch_id: str
    parent_state_cut_fingerprint: str
    shared_prefix_cells: tuple[str, ...]
    branch_suffix_cells: tuple[str, ...]
    dirty_cells: tuple[str, ...]
    reused_cells: tuple[str, ...]
    dirty_page_ids: tuple[str, ...]
    dirty_work_units: int
    closure_fingerprint: str
    mechanics_status: str = "valid"
    instrument_status: str = "metadata-only"
    trend_status: str = "not-assessed"
    terminal_status: str = "not-established"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.branch_id, label="compiled branch.branch_id")
        _optional_sha(
            self.parent_state_cut_fingerprint,
            label="compiled branch.parent_state_cut_fingerprint",
        )
        _optional_sha(self.closure_fingerprint, label="compiled branch.closure_fingerprint")
        for field_name in (
            "shared_prefix_cells",
            "branch_suffix_cells",
            "dirty_cells",
            "reused_cells",
            "dirty_page_ids",
        ):
            _unique_tuple(getattr(self, field_name), label=f"compiled branch.{field_name}")
        _positive_int(
            self.dirty_work_units, label="compiled branch.dirty_work_units", allow_zero=True
        )
        object.__setattr__(
            self,
            "fingerprint",
            _digest(self.to_dict(include_fingerprint=False), label="compiled branch"),
        )

    @property
    def branch_suffix(self) -> tuple[str, ...]:
        return self.branch_suffix_cells

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": "saturn-compiled-branch-v1",
            "branch_id": self.branch_id,
            "parent_state_cut_fingerprint": self.parent_state_cut_fingerprint,
            "shared_prefix_cells": list(self.shared_prefix_cells),
            "branch_suffix_cells": list(self.branch_suffix_cells),
            "dirty_cells": list(self.dirty_cells),
            "reused_cells": list(self.reused_cells),
            "dirty_page_ids": list(self.dirty_page_ids),
            "dirty_work_units": self.dirty_work_units,
            "closure_fingerprint": self.closure_fingerprint,
            "statuses": {
                "mechanics_status": self.mechanics_status,
                "instrument_status": self.instrument_status,
                "trend_status": self.trend_status,
                "terminal_status": self.terminal_status,
            },
            "speedup_claim": "not-claimed",
            "physical_residency_proven": False,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompiledBranch:
        """Hydrate one content-addressed compiled branch."""

        if not isinstance(value, Mapping):
            raise AuthoritativeDependencyError("compiled branch must be a mapping")
        statuses = value.get("statuses", {})
        if not isinstance(statuses, Mapping):
            raise AuthoritativeDependencyError("compiled branch.statuses must be a mapping")
        result = cls(
            branch_id=value.get("branch_id"),
            parent_state_cut_fingerprint=value.get("parent_state_cut_fingerprint"),
            shared_prefix_cells=tuple(value.get("shared_prefix_cells", ())),
            branch_suffix_cells=tuple(value.get("branch_suffix_cells", ())),
            dirty_cells=tuple(value.get("dirty_cells", ())),
            reused_cells=tuple(value.get("reused_cells", ())),
            dirty_page_ids=tuple(value.get("dirty_page_ids", ())),
            dirty_work_units=value.get("dirty_work_units"),
            closure_fingerprint=value.get("closure_fingerprint"),
            mechanics_status=statuses.get("mechanics_status", "valid"),
            instrument_status=statuses.get("instrument_status", "metadata-only"),
            trend_status=statuses.get("trend_status", "not-assessed"),
            terminal_status=statuses.get("terminal_status", "not-established"),
        )
        if value.get("schema") not in (None, "saturn-compiled-branch-v1"):
            raise AuthoritativeDependencyError("unsupported compiled branch schema")
        declared = value.get("fingerprint")
        if (
            declared is not None
            and _optional_sha(declared, label="compiled branch fingerprint") != result.fingerprint
        ):
            raise AuthoritativeDependencyError("compiled branch fingerprint mismatch")
        return result


@dataclass(frozen=True, slots=True)
class BranchPlan:
    """A compiled multi-branch plan with shared-prefix metadata."""

    contract_fingerprint: str
    parent_state_cut_fingerprint: str
    shared_prefix_cells: tuple[str, ...]
    branches: tuple[CompiledBranch, ...]
    admitted_page_ids: tuple[str, ...]
    shared_page_ids: tuple[str, ...]
    branch_page_ids: Mapping[str, tuple[str, ...]]
    shared_work_units: int
    branch_work_units: Mapping[str, int]
    admission: WeightPageAdmission
    native_consumer_required: bool = True
    mechanics_status: str = "valid"
    instrument_status: str = "metadata-only"
    trend_status: str = "not-assessed"
    terminal_status: str = "not-established"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        _optional_sha(self.contract_fingerprint, label="branch plan.contract_fingerprint")
        _optional_sha(
            self.parent_state_cut_fingerprint, label="branch plan.parent_state_cut_fingerprint"
        )
        _unique_tuple(self.shared_prefix_cells, label="branch plan.shared_prefix_cells")
        if not self.branches:
            raise AuthoritativeDependencyError("branch plan must contain at least one branch")
        branch_ids = tuple(branch.branch_id for branch in self.branches)
        if len(set(branch_ids)) != len(branch_ids):
            raise AuthoritativeDependencyError("branch plan contains duplicate branch IDs")
        if any(
            branch.parent_state_cut_fingerprint != self.parent_state_cut_fingerprint
            for branch in self.branches
        ):
            raise AuthoritativeDependencyError(
                "branch plan contains a branch from a different parent StateCut"
            )
        _unique_tuple(self.admitted_page_ids, label="branch plan.admitted_page_ids")
        _unique_tuple(self.shared_page_ids, label="branch plan.shared_page_ids")
        branch_page_ids = {
            _text(key, label="branch plan.branch_page_ids key"): _unique_tuple(
                value, label=f"branch plan.branch_page_ids[{key}]"
            )
            for key, value in self.branch_page_ids.items()
        }
        branch_work_units = {
            _text(key, label="branch plan.branch_work_units key"): _positive_int(
                value, label=f"branch plan.branch_work_units[{key}]", allow_zero=True
            )
            for key, value in self.branch_work_units.items()
        }
        if set(branch_page_ids) != set(branch_ids) or set(branch_work_units) != set(branch_ids):
            raise AuthoritativeDependencyError(
                "branch plan maps must cover each branch exactly once"
            )
        if self.admission.admitted_page_ids != self.admitted_page_ids:
            raise AuthoritativeDependencyError("branch plan/admission page sets conflict")
        _positive_int(
            self.shared_work_units, label="branch plan.shared_work_units", allow_zero=True
        )
        object.__setattr__(self, "branch_page_ids", branch_page_ids)
        object.__setattr__(self, "branch_work_units", branch_work_units)
        object.__setattr__(
            self,
            "fingerprint",
            _digest(self.to_dict(include_fingerprint=False), label="branch plan"),
        )

    @property
    def page_ids(self) -> tuple[str, ...]:
        return self.admitted_page_ids

    @property
    def shared_prefix(self) -> tuple[str, ...]:
        return self.shared_prefix_cells

    @property
    def admitted_pages(self) -> tuple[str, ...]:
        return self.admitted_page_ids

    @property
    def admitted_weights(self) -> tuple[str, ...]:
        return self.admission.weight_ids

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "schema": AUTHORITATIVE_BRANCH_PLAN_SCHEMA,
            "contract_fingerprint": self.contract_fingerprint,
            "parent_state_cut_fingerprint": self.parent_state_cut_fingerprint,
            "shared_prefix_cells": list(self.shared_prefix_cells),
            "branches": [branch.to_dict() for branch in self.branches],
            "admitted_page_ids": list(self.admitted_page_ids),
            "shared_page_ids": list(self.shared_page_ids),
            "branch_page_ids": {key: list(value) for key, value in self.branch_page_ids.items()},
            "work_accounting": {
                "shared_work_units": self.shared_work_units,
                "branch_work_units": dict(self.branch_work_units),
                "basis": "declared-logical-work-only",
            },
            "admission": self.admission.to_dict(),
            "native_consumer_required": self.native_consumer_required,
            "statuses": {
                "mechanics_status": self.mechanics_status,
                "instrument_status": self.instrument_status,
                "trend_status": self.trend_status,
                "terminal_status": self.terminal_status,
            },
            "speedup_claim": "not-claimed",
            "physical_residency_proven": False,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BranchPlan:
        """Hydrate a serialized branch plan and verify every nested seal."""

        if not isinstance(value, Mapping):
            raise AuthoritativeDependencyError("branch plan must be a mapping")
        raw_branches = value.get("branches", ())
        if not isinstance(raw_branches, Sequence) or isinstance(
            raw_branches, (str, bytes, bytearray)
        ):
            raise AuthoritativeDependencyError("branch plan.branches must be a sequence")
        raw_branch_page_ids = value.get("branch_page_ids", {})
        raw_work = value.get("work_accounting", {})
        if not isinstance(raw_branch_page_ids, Mapping):
            raise AuthoritativeDependencyError("branch plan.branch_page_ids must be a mapping")
        if not isinstance(raw_work, Mapping):
            raise AuthoritativeDependencyError("branch plan.work_accounting must be a mapping")
        raw_branch_work = raw_work.get("branch_work_units", {})
        if not isinstance(raw_branch_work, Mapping):
            raise AuthoritativeDependencyError(
                "branch plan.work_accounting.branch_work_units must be a mapping"
            )
        statuses = value.get("statuses", {})
        if not isinstance(statuses, Mapping):
            raise AuthoritativeDependencyError("branch plan.statuses must be a mapping")
        admission = WeightPageAdmission.from_dict(value.get("admission", {}))
        result = cls(
            contract_fingerprint=value.get("contract_fingerprint"),
            parent_state_cut_fingerprint=value.get("parent_state_cut_fingerprint"),
            shared_prefix_cells=tuple(value.get("shared_prefix_cells", ())),
            branches=tuple(CompiledBranch.from_dict(branch) for branch in raw_branches),
            admitted_page_ids=tuple(value.get("admitted_page_ids", ())),
            shared_page_ids=tuple(value.get("shared_page_ids", ())),
            branch_page_ids={
                str(key): tuple(page_ids) for key, page_ids in raw_branch_page_ids.items()
            },
            shared_work_units=raw_work.get("shared_work_units"),
            branch_work_units={str(key): units for key, units in raw_branch_work.items()},
            admission=admission,
            native_consumer_required=value.get("native_consumer_required", True),
            mechanics_status=statuses.get("mechanics_status", "valid"),
            instrument_status=statuses.get("instrument_status", "metadata-only"),
            trend_status=statuses.get("trend_status", "not-assessed"),
            terminal_status=statuses.get("terminal_status", "not-established"),
        )
        if value.get("schema") not in (None, AUTHORITATIVE_BRANCH_PLAN_SCHEMA):
            raise AuthoritativeDependencyError("unsupported branch plan schema")
        declared = value.get("fingerprint")
        if (
            declared is not None
            and _optional_sha(declared, label="branch plan fingerprint") != result.fingerprint
        ):
            raise AuthoritativeDependencyError("branch plan fingerprint mismatch")
        return result


CompiledBranchPlan = BranchPlan


def compile_branch_plan(
    contract: DependencyContract,
    requests: Sequence[BranchRequest],
    *,
    parent_state_cut: Any | None = None,
    provider: Any | None = None,
    available_pages: Mapping[str, Any] | None = None,
    byte_budget: int | None = None,
) -> BranchPlan:
    """Compile conservative branch suffixes and one complete page admission."""

    if not isinstance(contract, DependencyContract):
        raise AuthoritativeDependencyError("compile_branch_plan requires a DependencyContract")
    requests = tuple(requests)
    if not requests:
        raise AuthoritativeDependencyError("compile_branch_plan requires at least one branch")
    if any(not isinstance(request, BranchRequest) for request in requests):
        raise AuthoritativeDependencyError("branch requests must be BranchRequest values")
    branch_ids = tuple(request.branch_id for request in requests)
    if len(set(branch_ids)) != len(branch_ids):
        raise AuthoritativeDependencyError("branch requests contain duplicate branch IDs")
    parent_fp = (
        _state_cut_fingerprint(parent_state_cut, label="parent_state_cut")
        if parent_state_cut is not None
        else requests[0].parent_state_cut_fingerprint
    )
    if any(request.parent_state_cut_fingerprint != parent_fp for request in requests):
        raise AuthoritativeDependencyError("branches do not share one parent StateCut identity")
    closures = {
        request.branch_id: compute_dirty_closure(
            contract,
            changed_cells=request.changed_cells,
            changed_addresses=request.changed_addresses,
        )
        for request in requests
    }
    positions = {node_id: index for index, node_id in enumerate(contract.node_ids)}
    first_dirty = {
        branch_id: min(
            (positions[node_id] for node_id in closure.dirty_cells),
            default=len(contract.nodes),
        )
        for branch_id, closure in closures.items()
    }
    divergence = min(first_dirty.values())
    shared_prefix = contract.node_ids[:divergence]
    shared_page_ids = tuple(
        page_id
        for page_id in contract.page_map
        if any(
            page_id == page.page_id
            for node in contract.nodes[:divergence]
            for page in node.weight_pages
        )
    )
    branch_values: list[CompiledBranch] = []
    branch_page_ids: dict[str, tuple[str, ...]] = {}
    branch_work_units: dict[str, int] = {}
    for request in requests:
        closure = closures[request.branch_id]
        # The suffix starts at the first branch-local invalidation.  Cells in
        # the gap after the common prefix remain explicitly listed as reused
        # cells in the closure; an executor must not infer they are dirty.
        suffix = contract.node_ids[first_dirty[request.branch_id] :]
        branch_values.append(
            CompiledBranch(
                branch_id=request.branch_id,
                parent_state_cut_fingerprint=parent_fp,
                shared_prefix_cells=shared_prefix,
                branch_suffix_cells=suffix,
                dirty_cells=closure.dirty_cells,
                reused_cells=closure.reused_cells,
                dirty_page_ids=closure.dirty_page_ids,
                dirty_work_units=closure.dirty_work_units,
                closure_fingerprint=closure.fingerprint,
            )
        )
        branch_page_ids[request.branch_id] = closure.dirty_page_ids
        branch_work_units[request.branch_id] = closure.dirty_work_units
    admitted_page_ids = tuple(
        dict.fromkeys(
            page_id
            for page_id in (
                *shared_page_ids,
                *(page_id for closure in closures.values() for page_id in closure.dirty_page_ids),
            )
        )
    )
    admission = admit_weight_pages(
        contract,
        admitted_page_ids,
        provider=provider,
        available_pages=available_pages,
        byte_budget=byte_budget,
    )
    shared_work_units = sum(node.work_units for node in contract.nodes[:divergence])
    return BranchPlan(
        contract_fingerprint=contract.fingerprint,
        parent_state_cut_fingerprint=parent_fp,
        shared_prefix_cells=shared_prefix,
        branches=tuple(branch_values),
        admitted_page_ids=admitted_page_ids,
        shared_page_ids=shared_page_ids,
        branch_page_ids=branch_page_ids,
        shared_work_units=shared_work_units,
        branch_work_units=branch_work_units,
        admission=admission,
        native_consumer_required=True,
    )


plan_compiled_branches = compile_branch_plan
build_dependency_contract = DependencyContract.from_graph
dependency_contract_from_graph = DependencyContract.from_graph


__all__ = [
    "AUTHORITATIVE_ADMISSION_SCHEMA",
    "AUTHORITATIVE_BRANCH_PLAN_SCHEMA",
    "AUTHORITATIVE_DEPENDENCY_SCHEMA",
    "AuthoritativeDependencyError",
    "WeightPageRef",
    "WeightPage",
    "DependencyDeclaration",
    "DependencyNode",
    "DependencyContract",
    "AuthoritativeDependencyContract",
    "AuthoritativeDependencyGraph",
    "DirtyClosure",
    "DirtyCone",
    "compute_dirty_closure",
    "plan_dirty_cone",
    "WeightPageAdmission",
    "admit_weight_pages",
    "BranchRequest",
    "CompiledBranch",
    "BranchPlan",
    "CompiledBranchPlan",
    "compile_branch_plan",
    "plan_compiled_branches",
    "build_dependency_contract",
    "dependency_contract_from_graph",
]

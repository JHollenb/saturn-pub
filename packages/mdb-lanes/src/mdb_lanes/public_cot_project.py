"""Export and verify public CoT cuts in an MDB :class:`Project`.

The helper is deliberately metadata-first.  Export receives real
``saturn_pub.StateCut`` objects and lets ``Project.save_session`` publish their
CAS payloads.  Verification checks the cut metadata, page/blob references,
receipts, and point/schema-qualified ``SymbolBinding`` records without calling
``Project.load_cut`` or hydrating a tensor page.

This module does not import a model framework and does not turn a public cut
into a semantic or causal claim.  Lifecycle definitions and uses remain the
existing ``mdb.symbol_lifecycle`` records and retain their native authority
boundaries.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from saturn_pub import Session, StateCut
from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import canonical, digest

from .symbol_lifecycle import SymbolLifecycle

PUBLIC_COT_PROJECT_SCHEMA = "mdb-public-cot-project-v1"


class PublicCotProjectError(ValueError):
    """Raised when a public CoT project export cannot be sealed safely."""


def _sealed_id(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PublicCotProjectError(f"{label} is not a SHA-256 object id")
    return value


def _native_session(value: Any) -> Session:
    session = value if isinstance(value, Session) else getattr(value, "session", None)
    if not isinstance(session, Session):
        raise PublicCotProjectError("expected a public saturn_pub Session or wrapper")
    return session


def _binding(value: Any) -> SymbolBinding:
    if isinstance(value, SymbolBinding):
        return value
    if isinstance(value, Mapping):
        try:
            return SymbolBinding.from_dict(dict(value))
        except (TypeError, ValueError) as exc:
            raise PublicCotProjectError("invalid sealed SymbolBinding") from exc
    raise PublicCotProjectError("bindings must be public SymbolBinding records")


def _lifecycle(value: Any) -> SymbolLifecycle:
    if isinstance(value, SymbolLifecycle):
        return value
    if isinstance(value, Mapping):
        try:
            return SymbolLifecycle.from_dict(value)
        except (TypeError, ValueError) as exc:
            raise PublicCotProjectError("invalid sealed SymbolLifecycle") from exc
    raise PublicCotProjectError("lifecycles must be public SymbolLifecycle records")


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PublicCotProjectError("debug_info must be a JSON object")
    body = dict(value)
    try:
        canonical(body)
    except (TypeError, ValueError) as exc:
        raise PublicCotProjectError("debug_info must be JSON-sealable metadata") from exc
    return body


def _cut_file_inventory(project: Any, cut_id: str) -> dict[str, Any]:
    """Check CAS refs and hashes without decoding slot pages or tensor blobs."""
    cut_id = _sealed_id(cut_id, label="cut id")
    root = Path(project.store.root)
    path = root / "cuts" / f"{cut_id}.json"
    try:
        data = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicCotProjectError(f"cannot read stored cut metadata: {cut_id}") from exc
    if data.get("schema") != "saturn-pub-local-cut-v2":
        raise PublicCotProjectError(f"cut {cut_id} is not an authoritative v2 CAS cut")
    manifest = data.get("manifest")
    if data.get("fingerprint") != cut_id or digest(manifest) != cut_id:
        raise PublicCotProjectError(f"cut {cut_id} metadata seal mismatch")
    if manifest.get("schema") != "saturn-pub-statecut-v2" or data.get("logical_parent") != manifest.get(
        "parent"
    ):
        raise PublicCotProjectError(f"cut {cut_id} ancestry metadata mismatch")
    slots = data.get("slots")
    if not isinstance(slots, dict):
        raise PublicCotProjectError(f"cut {cut_id} has no slot inventory")
    page_blobs: set[str] = set()
    for page_id in slots.values():
        page_id = _sealed_id(page_id, label="slot page id")
        if not isinstance(page_id, str):
            raise PublicCotProjectError(f"cut {cut_id} has an invalid slot page id")
        page_path = root / "pages" / f"{page_id}.json"
        try:
            raw = page_path.read_bytes()
            page = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublicCotProjectError(f"cut {cut_id} slot page is unavailable") from exc
        if hashlib.sha256(raw).hexdigest() != page_id or page.get("schema") != "saturn-pub-slot-page-v1":
            raise PublicCotProjectError(f"cut {cut_id} slot page seal mismatch")
        blob_id = page.get("tensor_blob")
        if blob_id is not None:
            if not isinstance(blob_id, str):
                raise PublicCotProjectError(f"cut {cut_id} slot page has an invalid tensor blob id")
            page_blobs.add(blob_id)
    blobs = data.get("physical_blobs", [])
    if not isinstance(blobs, list):
        raise PublicCotProjectError(f"cut {cut_id} has an invalid blob inventory")
    if page_blobs != set(blobs):
        raise PublicCotProjectError(f"cut {cut_id} tensor blob inventory mismatch")
    for blob_id in blobs:
        blob_id = _sealed_id(blob_id, label="tensor blob id")
        try:
            blob = (root / "blobs" / f"{blob_id}.safetensors").read_bytes()
        except OSError as exc:
            raise PublicCotProjectError(f"cut {cut_id} tensor blob is unavailable") from exc
        if hashlib.sha256(blob).hexdigest() != blob_id:
            raise PublicCotProjectError(f"cut {cut_id} tensor blob seal mismatch")
    return {
        "cut": cut_id,
        "slot_pages": len(slots),
        "physical_blobs": len(blobs),
        "payload_hydrated": False,
    }


def _cut_points(
    project: Any, session_id: str, cut_ids: Sequence[str]
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    points: dict[str, list[str]] = {}
    schemas: dict[str, Any] = {}
    for cut_id in cut_ids:
        descriptor = project.cut_descriptor(session_id, cut_id)
        point = descriptor.get("execution_point")
        surface = descriptor.get("surface") or {}
        if not isinstance(point, Mapping) or not isinstance(surface, Mapping):
            raise PublicCotProjectError(f"cut {cut_id} lacks public point/surface metadata")
        point_id = digest(dict(point))
        points.setdefault(point_id, []).append(cut_id)
        schemas[cut_id] = surface.get("state_schema")
    return schemas, {point_id: tuple(cuts) for point_id, cuts in points.items()}


def _binding_cut_joins(
    value: Mapping[str, Sequence[str]] | None,
    *,
    bindings: Sequence[SymbolBinding],
    cut_ids: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PublicCotProjectError("binding_cut_ids must be a mapping")
    known = {row.to_dict()["fingerprint"] for row in bindings}
    retained = set(cut_ids)
    result: dict[str, tuple[str, ...]] = {}
    for identifier, rows in value.items():
        if identifier not in known:
            raise PublicCotProjectError("binding_cut_ids contains an unknown binding")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise PublicCotProjectError("binding_cut_ids values must be nonempty cut sequences")
        selected = tuple(dict.fromkeys(rows))
        if any(cut not in retained for cut in selected):
            raise PublicCotProjectError("binding_cut_ids contains an unretained cut")
        result[identifier] = selected
    return result


def _lifecycle_cut_joins(lifecycles: Sequence[SymbolLifecycle]) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = {}
    for lifecycle in lifecycles:
        cuts = result.setdefault(lifecycle.binding_fingerprint, set())
        for event in (*lifecycle.definitions, *lifecycle.uses):
            source_cut = getattr(event, "source_cut", None)
            if source_cut:
                cuts.add(source_cut)
    return {identifier: tuple(sorted(cuts)) for identifier, cuts in result.items() if cuts}


def _validate_binding(
    binding: SymbolBinding,
    *,
    identity: str,
    execution: Mapping[str, Any],
    schemas: Mapping[str, Any],
    points: Mapping[str, Sequence[str]],
    binding_cut_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if binding.model_identity != identity:
        raise PublicCotProjectError("symbol binding model identity differs from project session")
    if binding.execution != digest(dict(execution)):
        raise PublicCotProjectError("symbol binding execution differs from project session")
    locations = []
    selected_scope = set(binding_cut_ids)
    location_candidates = {
        cut_id
        for location in binding.locations
        for cut_id in points.get(location.execution_point, ())
    }
    if selected_scope - location_candidates:
        raise PublicCotProjectError("binding cut scope contains an unrelated native cut")
    for location in binding.locations:
        candidates = tuple(points.get(location.execution_point, ()))
        if not candidates:
            raise PublicCotProjectError(
                f"symbol location {location.slot} has no retained native execution cut"
            )
        if selected_scope:
            selected = tuple(cut for cut in candidates if cut in selected_scope)
            if not selected:
                raise PublicCotProjectError("binding cut scope does not match its location point")
        else:
            selected = candidates
        for cut_id in selected:
            if location.state_schema != schemas[cut_id]:
                raise PublicCotProjectError(
                    "symbol location state schema differs from its native cut"
                )
        row = {"slot": location.slot, "selector": list(location.selector)}
        if len(selected) == 1:
            row["cut"] = selected[0]
        else:
            if not selected_scope:
                raise PublicCotProjectError(
                    "symbol location has multiple native cuts; supply an exact cut join"
                )
            row["cuts"] = list(selected)
        locations.append(row)
    return {
        "symbol": binding.symbol,
        "fingerprint": binding.to_dict()["fingerprint"],
        "status": binding.status,
        "locations": locations,
        "authority": "read-only-debug-info",
    }


def _receipt_body(value: Any) -> dict[str, Any]:
    if not callable(getattr(value, "to_dict", None)) or not isinstance(
        getattr(value, "fingerprint", None), str
    ):
        raise PublicCotProjectError("receipts must be sealed public saturn_pub Receipt objects")
    body = value.to_dict()
    if not isinstance(body, dict) or body.get("fingerprint") != value.fingerprint:
        raise PublicCotProjectError("receipt seal is not self-consistent")
    return body


def _check_receipt_contract(body: Mapping[str, Any], *, identity: str, execution: Mapping[str, Any]) -> None:
    declared_identity = body.get("model_identity")
    if declared_identity is not None and declared_identity != identity:
        raise PublicCotProjectError("receipt model identity differs from project session")
    declared_execution = body.get("execution")
    if declared_execution is not None:
        try:
            matches = digest(declared_execution) == digest(dict(execution))
        except (TypeError, ValueError) as exc:
            raise PublicCotProjectError("receipt execution is not sealable") from exc
        if not matches:
            raise PublicCotProjectError("receipt execution differs from project session")
    if body.get("status") != "refused" and (
        declared_identity is None or declared_execution is None
    ):
        raise PublicCotProjectError("completed receipt lacks model/execution identity")


def _receipt_refs(
    session: Session,
    cuts: set[str],
    extra: Sequence[Any],
    *,
    identity: str,
    execution: Mapping[str, Any],
) -> None:
    receipts = [*session.receipts, *session.attempts, *extra]
    for receipt in receipts:
        body = _receipt_body(receipt)
        _check_receipt_contract(body, identity=identity, execution=execution)
        operation = body.get("operation")
        if operation in {"restore", "abort"}:
            # These records revisit or reject an existing state; all explicit
            # refs are still required if present.
            refs = (body.get("parent"), body.get("result"), body.get("candidate"))
        else:
            refs = (body.get("parent"), body.get("result"))
        missing = [ref for ref in refs if ref and ref not in cuts]
        if missing:
            raise PublicCotProjectError(
                f"receipt {body.get('fingerprint', '<unsealed>')} lacks retained cut payloads: {missing}"
            )


def export_public_cot_project(
    project: Any,
    session_id: str,
    session: Session | Any,
    cuts: Sequence[StateCut],
    *,
    bindings: Sequence[SymbolBinding | Mapping[str, Any]] = (),
    lifecycles: Sequence[SymbolLifecycle | Mapping[str, Any]] = (),
    binding_cut_ids: Mapping[str, Sequence[str]] | None = None,
    debug_info: Mapping[str, Any] | None = None,
    receipts: Sequence[Any] = (),
    expected_revision: int = 0,
) -> dict[str, Any]:
    """Publish actual public cuts, bindings, and optional lifecycle records."""
    native = _native_session(session)
    if not isinstance(session_id, str) or not session_id.strip():
        raise PublicCotProjectError("session_id must be nonempty text")
    if isinstance(cuts, (str, bytes)) or not isinstance(cuts, Sequence):
        raise PublicCotProjectError("cuts must be a sequence of StateCut objects")
    selected = native.capture(retain=False)
    supplied_cuts = list(cuts) + [selected]
    unique_cuts: dict[str, StateCut] = {}
    for cut in supplied_cuts:
        if not isinstance(cut, StateCut):
            raise PublicCotProjectError("cuts must contain public StateCut objects")
        cut.verify()
        if cut.model_identity != native.adapter.model_identity or dict(cut.execution) != dict(
            native.adapter.execution
        ):
            raise PublicCotProjectError("cut belongs to another model or execution contract")
        unique_cuts[cut.fingerprint] = cut

    binding_rows = [_binding(value) for value in bindings]
    lifecycle_rows = [_lifecycle(value) for value in lifecycles]
    for lifecycle in lifecycle_rows:
        binding_rows.append(lifecycle.binding)
    binding_map: dict[str, SymbolBinding] = {}
    for row in binding_rows:
        fingerprint = row.to_dict()["fingerprint"]
        existing = binding_map.get(fingerprint)
        if existing is not None and existing.to_dict() != row.to_dict():
            raise PublicCotProjectError("binding fingerprint has conflicting records")
        binding_map[fingerprint] = row

    # Existing sessions may already contain older cut points needed by a
    # later binding; inspect their metadata, never hydrate their payloads.
    existing_ids: list[str] = []
    try:
        existing_ids = list(project.cut_ids(session_id))
    except ValueError:
        pass
    all_ids = list(dict.fromkeys([*existing_ids, *unique_cuts]))
    schemas, points = _cut_points(project, session_id, existing_ids) if existing_ids else ({}, {})
    points = {point_id: list(cuts) for point_id, cuts in points.items()}
    for cut_id, cut in unique_cuts.items():
        point = dict(cut.execution_point or {})
        point_id = digest(point)
        points.setdefault(point_id, []).append(cut_id)
        schemas[cut_id] = (cut.surface or {}).get("state_schema")
    points = {point_id: tuple(cuts) for point_id, cuts in points.items()}
    identity = native.adapter.model_identity
    execution = dict(native.adapter.execution)
    info = _json_mapping(debug_info)
    existing_project_info = info.get("public_cot_project")
    if existing_project_info is not None and not isinstance(existing_project_info, Mapping):
        raise PublicCotProjectError("debug_info.public_cot_project must be a mapping")
    stored_joins = (
        None
        if existing_project_info is None
        else existing_project_info.get("binding_cut_joins")
    )
    supplied_joins = binding_cut_ids if binding_cut_ids is not None else stored_joins
    joins = _binding_cut_joins(
        supplied_joins,
        bindings=tuple(binding_map.values()),
        cut_ids=all_ids,
    )
    for identifier, cuts in _lifecycle_cut_joins(lifecycle_rows).items():
        if any(cut not in all_ids for cut in cuts):
            raise PublicCotProjectError("lifecycle binding join refers to an unretained cut")
        existing = joins.get(identifier)
        if existing is not None and set(existing) != set(cuts):
            raise PublicCotProjectError("binding cut join differs from lifecycle source cuts")
        joins[identifier] = cuts
    binding_reports = [
        _validate_binding(
            binding,
            identity=identity,
            execution=execution,
            schemas=schemas,
            points=points,
            binding_cut_ids=joins.get(binding.to_dict()["fingerprint"], ()),
        )
        for binding in binding_map.values()
    ]
    _receipt_refs(
        native,
        set(all_ids) | set(unique_cuts),
        receipts,
        identity=identity,
        execution=execution,
    )

    existing_symbols = info.get("symbols", [])
    if not isinstance(existing_symbols, list):
        raise PublicCotProjectError("debug_info.symbols must be a list")
    info["symbols"] = [*existing_symbols, *(row.to_dict() for row in binding_map.values())]
    if existing_project_info is None:
        existing_project_info = {
            "schema": PUBLIC_COT_PROJECT_SCHEMA,
            "state_schema": (native.inspect().surface or {}).get("state_schema"),
            "authority": "native-public-cuts; read-only-symbol-bindings",
        }
    if existing_project_info.get("schema") != PUBLIC_COT_PROJECT_SCHEMA:
        raise PublicCotProjectError("debug_info.public_cot_project schema mismatch")
    existing_project_info["binding_cut_joins"] = {
        identifier: list(cuts) for identifier, cuts in sorted(joins.items())
    }
    info["public_cot_project"] = existing_project_info

    revision = project.save_session(
        session_id,
        native,
        list(unique_cuts.values()),
        debug_info=info,
        receipts=tuple(receipts),
        expected_revision=expected_revision,
    )
    lifecycle_reports = []
    for lifecycle in lifecycle_rows:
        saved = project.save_symbol_lifecycle(session_id, lifecycle)
        lifecycle_reports.append(
            {
                "lifecycle_id": saved["lifecycle_id"],
                "symbol": saved["lifecycle"]["symbol"],
                "definitions": len(saved["lifecycle"]["definitions"]),
                "uses": len(saved["lifecycle"]["uses"]),
            }
        )
    proof = verify_public_cot_project(
        project,
        session_id,
        lifecycle_ids=tuple(row["lifecycle_id"] for row in lifecycle_reports),
    )
    proof.update(
        {
            "schema": PUBLIC_COT_PROJECT_SCHEMA,
            "revision": revision,
            "exported_cuts": sorted(unique_cuts),
            "bindings": binding_reports,
            "binding_cut_joins": {
                identifier: list(cuts) for identifier, cuts in sorted(joins.items())
            },
            "lifecycles": lifecycle_reports,
        }
    )
    return proof


def verify_public_cot_project(
    project: Any,
    session_id: str,
    *,
    lifecycle_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Verify project custody and qualified metadata without tensor hydration."""
    row = project.session(session_id)
    identity = json.loads(row["identity"])
    execution = json.loads(row["execution"])
    cut_ids = list(project.cut_ids(session_id))
    if not cut_ids or row["selected"] not in cut_ids:
        raise PublicCotProjectError("project session has no retained selected cut")
    storage = [_cut_file_inventory(project, cut_id) for cut_id in cut_ids]
    schemas, points = _cut_points(project, session_id, cut_ids)
    for cut_id in cut_ids:
        descriptor = project.cut_descriptor(session_id, cut_id)
        if descriptor["model_identity"] != identity or descriptor["manifest"]["execution"] != execution:
            raise PublicCotProjectError("cut differs from project model/execution identity")
    raw_project_info = json.loads(row["debug_info"]).get("public_cot_project", {})
    if not isinstance(raw_project_info, Mapping):
        raise PublicCotProjectError("stored public CoT project metadata is not a mapping")
    binding_objects = tuple(_binding(value) for value in project.symbols(session_id))
    joins = _binding_cut_joins(
        raw_project_info.get("binding_cut_joins"),
        bindings=binding_objects,
        cut_ids=cut_ids,
    )
    bindings = [
        _validate_binding(
            binding,
            identity=identity,
            execution=execution,
            schemas=schemas,
            points=points,
            binding_cut_ids=joins.get(binding.to_dict()["fingerprint"], ()),
        )
        for binding in binding_objects
    ]
    receipt_rows = []
    for receipt in project.receipts(session_id):
        body = _receipt_body(receipt)
        _check_receipt_contract(body, identity=identity, execution=execution)
        if body.get("operation") in {"restore", "abort"}:
            refs = [body.get("parent"), body.get("result"), body.get("candidate")]
        else:
            refs = [body.get("parent"), body.get("result")]
        if any(ref and ref not in cut_ids for ref in refs):
            raise PublicCotProjectError("project receipt references a non-retained cut")
        receipt_rows.append(
            {
                "fingerprint": body["fingerprint"],
                "operation": body.get("operation"),
                "parent": body.get("parent"),
                "result": body.get("result"),
            }
        )
    lifecycle_rows = []
    for identifier in lifecycle_ids:
        record = project.symbol_lifecycle(session_id, identifier)
        lifecycle = _lifecycle(record["lifecycle"])
        lifecycle.verify()
        if lifecycle.binding_fingerprint not in {row["fingerprint"] for row in bindings}:
            raise PublicCotProjectError("lifecycle binding is absent from the project symbol table")
        event_rows = [*lifecycle.definitions, *lifecycle.uses]
        refs = {
            cut
            for event in event_rows
            for cut in (
                getattr(event, "source_cut", None),
                getattr(event, "parent_cut", None),
                getattr(event, "result_cut", None),
            )
            if cut
        }
        if not refs.issubset(set(cut_ids)):
            raise PublicCotProjectError("lifecycle event refers to a non-retained cut")
        lifecycle_rows.append(
            {
                "lifecycle_id": identifier,
                "symbol": lifecycle.symbol,
                "definitions": len(lifecycle.definitions),
                "uses": len(lifecycle.uses),
            }
        )
    return {
        "schema": PUBLIC_COT_PROJECT_SCHEMA,
        "session_id": session_id,
        "selected_cut": row["selected"],
        "cut_ids": cut_ids,
        "cut_storage": storage,
        "payload_hydrated": False,
        "bindings": bindings,
        "binding_cut_joins": {
            identifier: list(cuts) for identifier, cuts in sorted(joins.items())
        },
        "receipts": receipt_rows,
        "lifecycles": lifecycle_rows,
        "native_payload_reads": 0,
        "authority": {
            "cuts": "public-saturn-statecut-cas",
            "bindings": "read-only-debug-info",
            "causal_effect": "not-established",
        },
    }


__all__ = [
    "PUBLIC_COT_PROJECT_SCHEMA",
    "PublicCotProjectError",
    "export_public_cot_project",
    "verify_public_cot_project",
]

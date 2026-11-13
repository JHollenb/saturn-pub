"""Join sealed producer observations to an existing native symbol binding.

This module is deliberately a join layer.  A producer can describe a native
consumer response, a semantic manifest candidate, a CoT register observation,
or a future ISA source hypothesis, but it cannot manufacture a
``saturn_pub.SymbolBinding``.  The caller must provide the exact binding and
the exact retained public StateCut on which the producer was observed.

The adapter only reads Project metadata and writes a sealed
``SymbolLifecycle``.  It never loads native tensor pages, imports model
frameworks, translates Bench cuts into public cuts, or promotes a producer's
name/status into semantic truth.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from saturn_pub.symbols import SymbolBinding
from saturn_pub.values import digest as native_digest

from .hyperion import (
    HYPERION_REPORT_SCHEMA,
    HyperionError,
    verify_hyperion_report,
)
from .semantic_binding import SEMANTIC_BINDING_SCHEMA, SemanticBinding, SemanticBindingError
from .symbol_lifecycle import SymbolLifecycle, SymbolLifecycleError, SymbolOccurrence, SymbolUse

PRODUCER_JOIN_SCHEMA = "mdb-symbol-producer-join-v1"
ISA_SOURCE_CANDIDATE_SCHEMA = "mdb-isa-source-candidate-v1"
SYMBOL_SOURCE_CANDIDATE_SCHEMA = "mdb-symbol-source-candidate-v1"
NATIVE_OBSERVATION_SCHEMAS = frozenset(
    {
        "saturn-bench-observation-v1",
        "saturn-bench-family-observation-v2",
        "saturn-bench-consumer-observation-v1",
    }
)
_COT_OBSERVATION_SCHEMA = "mdb-cot-register-observation-v1"
_SHA256 = frozenset("0123456789abcdef")


class SymbolProducerError(ValueError):
    """A producer record cannot be joined to the requested public binding."""


def _text(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise SymbolProducerError(f"{label} must be non-empty text")
    return value.strip()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _json_safe(value: Any, *, label: str = "producer", depth: int = 0) -> Any:
    """Make a lossless JSON representation without touching tensor payloads."""

    if depth > 12:
        raise SymbolProducerError(f"{label} metadata is too deeply nested")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SymbolProducerError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "__mdb_bytes__": base64.b64encode(raw).decode("ascii"),
            "encoding": "base64",
            "length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if any(hasattr(value, name) for name in ("detach", "numpy", "ndim", "shape")):
        raise SymbolProducerError(f"{label} contains a tensor-like payload")
    if isinstance(value, Mapping):
        if len(value) > 512:
            raise SymbolProducerError(f"{label} has too many mapping entries")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise SymbolProducerError(f"{label} contains a non-text key")
            result[key] = _json_safe(item, label=f"{label}.{key}", depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 512:
            raise SymbolProducerError(f"{label} has too many sequence entries")
        return [
            _json_safe(item, label=f"{label}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    raise SymbolProducerError(f"{label} contains unsupported {type(value).__name__}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise SymbolProducerError(f"{label} must be a mapping or sealed record")


def _as_sha(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in _SHA256 for char in result):
        raise SymbolProducerError(f"{label} must be a lowercase SHA-256")
    return result


def _execution_digest(value: Any, label: str) -> str:
    if isinstance(value, Mapping):
        return native_digest(dict(value))
    return _as_sha(value, label)


def _record_payload(value: Any) -> dict[str, Any]:
    payload = _mapping(value, "producer record")
    try:
        safe = _json_safe(payload)
    except SymbolProducerError:
        raise
    if not isinstance(safe, dict):  # defensive; _mapping already guarantees this
        raise SymbolProducerError("producer record must be a JSON object")
    return safe


def seal_isa_source_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a future ISA source candidate without assigning it native meaning.

    This helper only supplies the format needed for a later producer.  It does
    not derive addresses or a binding.  The candidate must still be wrapped
    with a retained public ``source_cut`` when joined.
    """

    body = _record_payload(candidate)
    schema = body.get("schema", ISA_SOURCE_CANDIDATE_SCHEMA)
    if schema != ISA_SOURCE_CANDIDATE_SCHEMA:
        raise SymbolProducerError("unsupported ISA source-candidate schema")
    body["schema"] = schema
    if not _text(body.get("candidate_id"), "ISA candidate_id"):
        raise SymbolProducerError("ISA source candidate needs candidate_id")
    declared = body.pop("fingerprint", None)
    fingerprint = _sha(body)
    if declared is not None and declared != fingerprint:
        raise SymbolProducerError("ISA source-candidate fingerprint mismatch")
    return {**body, "fingerprint": fingerprint}


def _verify_isa_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("schema") != ISA_SOURCE_CANDIDATE_SCHEMA:
        raise SymbolProducerError("unsupported ISA source-candidate schema")
    declared = _as_sha(record.get("fingerprint"), "ISA candidate fingerprint")
    body = {key: value for key, value in record.items() if key != "fingerprint"}
    if _sha(body) != declared:
        raise SymbolProducerError("ISA source-candidate fingerprint mismatch")
    _text(record.get("candidate_id"), "ISA candidate_id")
    return {"kind": "isa-source-candidate", "verification": "sealed"}


def _verify_symbol_source_candidate(record: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the model-neutral source-label envelope from ``symbol_sources``."""

    if record.get("schema") != SYMBOL_SOURCE_CANDIDATE_SCHEMA:
        raise SymbolProducerError("unsupported symbol-source candidate schema")
    # Delegate to the producer's public verifier.  A fingerprint alone is
    # insufficient: a caller could re-sign altered logits/proposals and make
    # a false winner look like a source observation.  verify_candidate()
    # recomputes the proposal rows and selected label from the frozen source.
    from .symbol_sources import SymbolSourceError, verify_candidate

    try:
        result = verify_candidate(record)
    except (KeyError, TypeError, ValueError, SymbolSourceError) as exc:
        raise SymbolProducerError(
            f"symbol-source candidate recomputation failed: {exc}"
        ) from exc
    return {
        "kind": "symbol-source-candidate",
        "verification": "source-recomputed",
        "source_trust": record.get("source_trust"),
        "candidate_verification": result,
    }


def _artifact_get(artifacts: Any, reference: str) -> Any:
    if artifacts is None:
        raise KeyError(reference)
    getter = getattr(artifacts, "get", None)
    if callable(getter):
        value = getter(reference)
    else:
        value = artifacts[reference]
    if value is None:
        raise KeyError(reference)
    return value


def _verify_cot(
    record: Mapping[str, Any], *, artifacts: Any = None, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    if record.get("schema") != _COT_OBSERVATION_SCHEMA:
        raise SymbolProducerError("unsupported CoT register schema")
    required = ("config_ref", "observation_ref", "source_cut", "lens_ref", "session_id")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise SymbolProducerError(f"CoT observation missing {', '.join(missing)}")
    if record.get("interpretation") != "decoded-candidates":
        raise SymbolProducerError("CoT observation interpretation is not source-bound")
    for key in ("causal_dependency_tested", "native_mutation", "semantic_cot_qualified"):
        if record.get(key) is not False:
            raise SymbolProducerError(f"CoT observation {key} cannot be promoted")
    if artifacts is not None and manifest is not None:
        from .bench_cot import verify_artifact

        try:
            verify_artifact(record, artifacts, manifest)
        except Exception as exc:  # normalize producer-specific verifier errors
            raise SymbolProducerError(f"CoT observation seal verification failed: {exc}") from exc
        return {"kind": "cot-register-observation", "verification": "artifact-verified"}
    return {
        "kind": "cot-register-observation",
        "verification": "schema-verified-fields-only",
        "verification_status": "evidence-incomplete",
    }


def _verify_native_observation(
    record: Mapping[str, Any], *, artifacts: Any = None, reference: str | None = None
) -> dict[str, Any]:
    schema = record.get("schema")
    if schema not in NATIVE_OBSERVATION_SCHEMAS:
        raise SymbolProducerError("unsupported native observation schema")
    if not record.get("model_identity"):
        raise SymbolProducerError("native observation needs model_identity")
    if artifacts is not None and reference:
        try:
            stored = _record_payload(_artifact_get(artifacts, reference))
        except Exception as exc:
            raise SymbolProducerError("native observation artifact reference is unavailable") from exc
        if stored != dict(record):
            raise SymbolProducerError("native observation artifact bytes differ")
        return {"kind": "native-observation", "verification": "artifact-verified"}
    return {
        "kind": "native-observation",
        "verification": "schema-verified-fields-only",
        "verification_status": "evidence-incomplete",
    }


def _verify_producer(
    record: Mapping[str, Any], *, artifacts: Any = None, manifest: Mapping[str, Any] | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    schema = record.get("schema")
    if schema == HYPERION_REPORT_SCHEMA:
        try:
            result = verify_hyperion_report(record)
        except (HyperionError, ValueError, KeyError, TypeError) as exc:
            raise SymbolProducerError(f"Hyperion report seal verification failed: {exc}") from exc
        return {"kind": "hyperion-report", "verification": "sealed", "compiler": result}
    if schema == SEMANTIC_BINDING_SCHEMA:
        try:
            SemanticBinding.from_dict(record)
        except (SemanticBindingError, ValueError, KeyError, TypeError) as exc:
            raise SymbolProducerError(f"semantic binding seal verification failed: {exc}") from exc
        return {"kind": "semantic-binding", "verification": "sealed"}
    if schema == _COT_OBSERVATION_SCHEMA:
        return _verify_cot(record, artifacts=artifacts, manifest=manifest)
    if schema in NATIVE_OBSERVATION_SCHEMAS:
        return _verify_native_observation(record, artifacts=artifacts, reference=reference)
    if schema == ISA_SOURCE_CANDIDATE_SCHEMA:
        return _verify_isa_candidate(record)
    if schema == SYMBOL_SOURCE_CANDIDATE_SCHEMA:
        return _verify_symbol_source_candidate(record)
    raise SymbolProducerError(f"unsupported producer schema: {schema!r}")


def _coalesce(envelope: Mapping[str, Any], record: Mapping[str, Any], key: str) -> Any:
    left = envelope.get(key)
    right = record.get(key)
    if left is not None and right is not None and left != right:
        raise SymbolProducerError(f"producer {key} disagrees with its join envelope")
    return left if left is not None else right


def _registered_binding(project: Any, session_id: str, binding: Any) -> SymbolBinding:
    try:
        resolved = binding if isinstance(binding, SymbolBinding) else SymbolBinding.from_dict(dict(binding))
    except (TypeError, KeyError, ValueError) as exc:
        raise SymbolProducerError("existing native binding is not sealed") from exc
    session = project.session(session_id)
    identity = json.loads(session["identity"])
    execution = json.loads(session["execution"])
    if resolved.model_identity != identity or resolved.execution != native_digest(execution):
        raise SymbolProducerError("native binding belongs to another project model or execution")
    fingerprint = resolved.to_dict()["fingerprint"]
    try:
        registered = project.symbols(session_id)
    except Exception as exc:
        raise SymbolProducerError("project symbol registry could not be verified") from exc
    if fingerprint not in {row.get("fingerprint") for row in registered}:
        raise SymbolProducerError("producer join requires an already registered native binding")
    return resolved


def _check_declared_identity(
    record: Mapping[str, Any], envelope: Mapping[str, Any], *, project_identity: Mapping[str, Any],
    project_execution: Mapping[str, Any], binding: SymbolBinding, session_id: str,
) -> None:
    declared_session = _coalesce(envelope, record, "session_id")
    if declared_session is not None and declared_session != session_id:
        raise SymbolProducerError("producer session differs from the public Project session")
    declared_model = _coalesce(envelope, record, "model_identity")
    if declared_model is None and record.get("schema") == HYPERION_REPORT_SCHEMA:
        symbol = record.get("symbol")
        if isinstance(symbol, Mapping):
            declared_model = symbol.get("model_id")
    if declared_model is None and record.get("schema") == SEMANTIC_BINDING_SCHEMA:
        context = record.get("context")
        if isinstance(context, Mapping):
            declared_model = context.get("model_id")
    if declared_model is not None:
        if isinstance(declared_model, Mapping):
            if isinstance(project_identity, Mapping):
                if dict(declared_model) != dict(project_identity):
                    raise SymbolProducerError("producer model identity differs from the Project session")
            elif declared_model.get("model_id") != project_identity:
                raise SymbolProducerError("producer model_id differs from the Project session")
        elif declared_model != (
            project_identity.get("model_id")
            if isinstance(project_identity, Mapping)
            else project_identity
        ):
            raise SymbolProducerError("producer model_id differs from the Project session")
    declared_execution = _coalesce(envelope, record, "execution")
    if declared_execution is None:
        declared_execution = _coalesce(envelope, record, "execution_digest")
    if declared_execution is not None and _execution_digest(declared_execution, "producer execution") != native_digest(project_execution):
        raise SymbolProducerError("producer execution differs from the Project session")
    declared_binding_context = envelope.get("binding_context")
    if declared_binding_context is None:
        declared_binding_context = record.get("binding_context")
    if declared_binding_context is not None and declared_binding_context != binding.context:
        raise SymbolProducerError("producer binding context differs from the native binding")
    if record.get("schema") == SYMBOL_SOURCE_CANDIDATE_SCHEMA:
        selected_symbol = _text(record.get("selected_symbol"), "symbol-source selected_symbol")
        if selected_symbol != binding.symbol:
            raise SymbolProducerError("symbol-source selected_symbol differs from the native binding")
        declared_context = _coalesce(envelope, record, "context")
        if declared_context is not None and declared_context != binding.context:
            raise SymbolProducerError("symbol-source context differs from the native binding")


def _cut_refs(project: Any, session_id: str, envelope: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, str | None]:
    refs: dict[str, str | None] = {}
    for key in ("source_cut", "parent_cut", "result_cut"):
        value = _coalesce(envelope, record, key)
        if value is None:
            refs[key] = None
            continue
        ref = _text(value, f"producer {key}")
        try:
            project.cut_descriptor(session_id, ref)
        except Exception as exc:
            raise SymbolProducerError(
                f"producer {key} is not a retained public Project cut; Bench cuts are not interchangeable"
            ) from exc
        refs[key] = ref
    if refs["source_cut"] is None:
        raise SymbolProducerError("producer join requires an exact retained source_cut")
    return refs


def _evidence_refs(record: Mapping[str, Any], envelope: Mapping[str, Any], fingerprint: str) -> tuple[str, ...]:
    values: list[Any] = []
    for container in (envelope, record):
        for key in ("evidence_refs", "evidence_ref", "producer_ref", "report_ref", "register_ref"):
            value = container.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values.extend(item for item in value if isinstance(item, str))
    values = list(dict.fromkeys(values))
    return tuple(values or (f"producer://sha256/{fingerprint}",))


def _locations(value: Any, binding: SymbolBinding) -> tuple[Any, ...]:
    if value is None:
        return binding.locations
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SymbolProducerError("producer use locations must be a sequence")
    try:
        rows = tuple(
            item if hasattr(item, "slot") else type(binding.locations[0])(
                item["slot"], tuple(item["selector"]), item["role"],
                item["execution_point"], item["state_schema"],
            )
            for item in value
        )
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise SymbolProducerError("producer use locations are not public StateAddress records") from exc
    allowed = {location.to_dict().__repr__() for location in binding.locations}
    if any(location.to_dict().__repr__() not in allowed for location in rows):
        raise SymbolProducerError("producer use locations exceed the existing native binding")
    return rows


def _use_from_record(record: Any, binding: SymbolBinding, session_id: str) -> SymbolUse:
    if isinstance(record, SymbolUse):
        if record.binding_fingerprint != binding.to_dict()["fingerprint"] or record.session_id != session_id:
            raise SymbolProducerError("supplied use belongs to another binding or session")
        return record
    raw = _record_payload(record)
    binding_fp = binding.to_dict()["fingerprint"]
    if raw.get("binding_fingerprint") not in (None, binding_fp):
        raise SymbolProducerError("supplied use belongs to another binding")
    source_cut = raw.get("source_cut")
    parent_cut = raw.get("parent_cut")
    result_cut = raw.get("result_cut")
    locations = _locations(raw.get("locations"), binding)
    metadata = dict(raw.get("metadata") or {})
    metadata.setdefault("producer_join", {"schema": PRODUCER_JOIN_SCHEMA, "raw_use": raw})
    return SymbolUse(
        use_id=_text(raw.get("use_id"), "use_id"),
        binding_fingerprint=binding_fp,
        session_id=session_id,
        consumer=_text(raw.get("consumer"), "use consumer"),
        source_cut=source_cut,
        parent_cut=parent_cut,
        result_cut=result_cut,
        branch=raw.get("branch"),
        execution=raw.get("execution", binding.execution),
        receipt=raw.get("receipt"),
        locations=locations,
        clock=raw.get("clock"),
        signed_observations=tuple(raw.get("signed_observations") or ()),
        evidence_refs=tuple(raw.get("evidence_refs") or ()),
        status=raw.get("status", "unknown"),
        validity_status=raw.get("validity_status", "unknown"),
        validity_support=tuple(raw.get("validity_support") or ()),
        metadata=metadata,
    )


def _validate_use_refs(project: Any, session_id: str, use: SymbolUse) -> None:
    cuts = project.cut_ids(session_id)
    retained = set(cuts)
    for name, reference in use.cuts.items():
        if reference not in retained:
            raise SymbolProducerError(f"use {name}_cut is not retained in the public Project")
        project.cut_descriptor(session_id, reference)
    if use.receipt is None:
        return
    receipts = {item.fingerprint: item.to_dict() for item in project.receipts(session_id)}
    body = receipts.get(use.receipt)
    if body is None:
        raise SymbolProducerError("use receipt is not retained in the public Project")
    if use.parent_cut is not None and body.get("parent") != use.parent_cut:
        raise SymbolProducerError("use parent_cut disagrees with its native receipt")
    if use.result_cut is not None and body.get("result") != use.result_cut:
        raise SymbolProducerError("use result_cut disagrees with its native receipt")


def build_symbol_lifecycle(
    project: Any,
    session_id: str,
    binding: SymbolBinding | Mapping[str, Any],
    producer_records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    uses: Sequence[SymbolUse | Mapping[str, Any]] = (),
    producer_artifacts: Any = None,
    producer_manifest: Mapping[str, Any] | None = None,
) -> SymbolLifecycle:
    """Build a lifecycle from sealed producer records without saving it."""

    binding = _registered_binding(project, session_id, binding)
    if isinstance(producer_records, Mapping) or hasattr(producer_records, "to_dict"):
        producer_records = (producer_records,)  # type: ignore[assignment]
    if isinstance(producer_records, (str, bytes)) or not isinstance(producer_records, Sequence):
        raise SymbolProducerError("producer_records must be a record or sequence of records")
    if not producer_records:
        raise SymbolProducerError("producer join requires at least one producer record")
    session = project.session(session_id)
    project_identity = json.loads(session["identity"])
    project_execution = json.loads(session["execution"])
    definitions: list[SymbolOccurrence] = []
    summaries: list[dict[str, Any]] = []
    for index, supplied in enumerate(producer_records):
        envelope = _record_payload(supplied)
        raw_value = envelope.pop("producer", supplied)
        if raw_value is supplied:
            raw = _record_payload(supplied)
            envelope = {}
            for key in ("source_cut", "parent_cut", "result_cut", "branch", "receipt", "session_id", "model_identity", "execution", "execution_digest", "binding_context", "evidence_refs", "producer_ref", "occurrence_id", "lifecycle_status"):
                if key in raw:
                    envelope[key] = raw[key]
        else:
            raw = _record_payload(raw_value)
        schema = raw.get("schema")
        if not isinstance(schema, str):
            raise SymbolProducerError("producer record needs a known schema")
        reference = envelope.get("producer_ref") or raw.get("producer_ref")
        reference = reference if isinstance(reference, str) else None
        verification = _verify_producer(
            raw, artifacts=producer_artifacts or envelope.get("producer_artifacts"),
            manifest=producer_manifest or envelope.get("producer_manifest"), reference=reference,
        )
        _check_declared_identity(
            raw, envelope, project_identity=project_identity, project_execution=project_execution,
            binding=binding, session_id=session_id,
        )
        cuts = _cut_refs(project, session_id, envelope, raw)
        receipt = _coalesce(envelope, raw, "receipt")
        if receipt is not None:
            receipt = _text(receipt, "producer receipt")
            receipts = {item.fingerprint: item.to_dict() for item in project.receipts(session_id)}
            if receipt not in receipts:
                raise SymbolProducerError("producer receipt is not retained in the public Project")
            body = receipts[receipt]
            for key, expected in (("parent_cut", cuts["parent_cut"]), ("result_cut", cuts["result_cut"])):
                if expected is not None and body.get("parent" if key == "parent_cut" else "result") != expected:
                    raise SymbolProducerError("producer receipt disagrees with its public cut reference")
        raw_copy = _json_safe(raw)
        producer_fingerprint = raw.get("fingerprint") if isinstance(raw.get("fingerprint"), str) else _sha(raw_copy)
        evidence_refs = _evidence_refs(raw, envelope, producer_fingerprint)
        occurrence_id = envelope.get("occurrence_id") or f"producer-{index}-{producer_fingerprint[:16]}"
        metadata = {
            "producer_join": {
                "schema": PRODUCER_JOIN_SCHEMA,
                "producer_schema": schema,
                "producer_fingerprint": producer_fingerprint,
                "verification": verification,
                "source_cut_namespace": "saturn-pub-project",
                "source_cut": cuts["source_cut"],
                "parent_cut": cuts["parent_cut"],
                "result_cut": cuts["result_cut"],
                "raw_record": raw_copy,
                "producer_status": raw.get("status"),
                "producer_terminal_status": raw.get("terminal_status"),
                "producer_semantic_status": raw.get("semantic_cot_qualified"),
            }
        }
        definitions.append(
            SymbolOccurrence.for_binding(
                binding,
                occurrence_id=_text(occurrence_id, "occurrence_id"),
                session_id=session_id,
                source_cut=cuts["source_cut"],
                result_cut=cuts["result_cut"],
                branch=envelope.get("branch") or raw.get("branch"),
                execution=binding.execution,
                receipt=receipt,
                locations=binding.locations,
                clock=_coalesce(envelope, raw, "clock"),
                spectral_address=_coalesce(envelope, raw, "spectral_address"),
                vm_bundle=_coalesce(envelope, raw, "vm_bundle"),
                cot_refs=(raw,) if schema == _COT_OBSERVATION_SCHEMA else (),
                evidence_refs=evidence_refs,
                status=envelope.get("lifecycle_status", "observed"),
                validity_status="unknown",
                metadata=metadata,
            )
        )
        summaries.append(
            {
                "schema": schema,
                "fingerprint": producer_fingerprint,
                "verification": verification,
                "source_cut": cuts["source_cut"],
                "status": raw.get("status"),
                "semantic_status": raw.get("semantic_cot_qualified"),
            }
        )
    lifecycle_uses = tuple(_use_from_record(item, binding, session_id) for item in uses)
    for use in lifecycle_uses:
        _validate_use_refs(project, session_id, use)
    return SymbolLifecycle.define(
        binding,
        definitions=tuple(definitions),
        uses=lifecycle_uses,
        mechanics_status="valid",
        instrument_status=(
            "valid"
            if all(
                row["verification"].get("verification")
                in {"sealed", "artifact-verified", "source-recomputed"}
                for row in summaries
            )
            else "evidence-incomplete"
        ),
        trend_status="observations-only",
        terminal_status="not-assessed",
        claim_boundary=(
            "This lifecycle joins sealed producer observations to an exact native binding. "
            "Producer names, semantic statuses, and signed effects remain descriptive; "
            "the join establishes neither semantic equivalence nor causal necessity."
        ),
        metadata={
            "schema": PRODUCER_JOIN_SCHEMA,
            "producer_records": summaries,
            "authority": "read-only-producer-evidence-join",
            "source_cut_namespace": "saturn-pub-project",
            "semantic_equivalence": "not-established",
            "causal_effect": "not-established",
        },
    )


def join_symbol_producers(
    project: Any,
    session_id: str,
    binding: SymbolBinding | Mapping[str, Any],
    producer_records: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    uses: Sequence[SymbolUse | Mapping[str, Any]] = (),
    producer_artifacts: Any = None,
    producer_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and persist a producer-qualified, read-only symbol lifecycle."""

    lifecycle = build_symbol_lifecycle(
        project,
        session_id,
        binding,
        producer_records,
        uses=uses,
        producer_artifacts=producer_artifacts,
        producer_manifest=producer_manifest,
    )
    try:
        return project.save_symbol_lifecycle(session_id, lifecycle)
    except SymbolLifecycleError as exc:
        raise SymbolProducerError(f"native lifecycle validation failed: {exc}") from exc


integrate_symbol_producers = join_symbol_producers


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SymbolProducerError(f"cannot read {label} JSON {path}: {exc}") from exc


def _resolve_session(project: Any, requested: str | None, parser: argparse.ArgumentParser) -> str:
    if requested is not None:
        try:
            project.session(requested)
        except Exception as exc:
            parser.error(f"unknown --session-id {requested!r}: {exc}")
        return requested
    sessions = project.sessions()
    if len(sessions) != 1:
        parser.error("select --session-id for a project with zero or multiple sessions")
    return sessions[0]["session_id"]


def _resolve_binding(
    project: Any, session_id: str, fingerprint: str, parser: argparse.ArgumentParser
) -> SymbolBinding:
    try:
        rows = project.symbols(session_id)
    except Exception as exc:
        parser.error(f"cannot read registered symbol bindings: {exc}")
    matches = [row for row in rows if row.get("fingerprint") == fingerprint]
    if len(matches) != 1:
        parser.error(
            f"--binding must name one exact registered binding fingerprint; found {len(matches)}"
        )
    try:
        return SymbolBinding.from_dict(matches[0])
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(f"registered binding is not sealed: {exc}")


def _producer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdb symbol-producers",
        description=(
            "Join producer JSON observations to one exact registered native binding. "
            "CoT rows without retained verifier artifacts remain schema-verified evidence-incomplete."
        ),
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("producer_paths", type=Path, nargs="+")
    parser.add_argument("--session-id")
    parser.add_argument("--binding", required=True, help="exact registered SymbolBinding fingerprint")
    parser.add_argument("--uses", type=Path, help="JSON object or array of supplied native SymbolUse records")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Join JSON producer records without resolving names into native addresses."""

    parser = _producer_parser()
    args = parser.parse_args(argv)
    try:
        from .project import Project

        project = Project(args.root)
        session_id = _resolve_session(project, args.session_id, parser)
        binding = _resolve_binding(project, session_id, args.binding, parser)
        producer_records = []
        for path in args.producer_paths:
            value = _read_json(path, "producer")
            if not isinstance(value, Mapping):
                raise SymbolProducerError(f"producer JSON {path} must contain one object")
            producer_records.append(value)
        uses: tuple[Mapping[str, Any], ...] = ()
        if args.uses is not None:
            value = _read_json(args.uses, "uses")
            if isinstance(value, Mapping):
                uses = (value,)
            elif isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
                uses = tuple(value)
            else:
                raise SymbolProducerError("uses JSON must contain an object or array of objects")
        result = join_symbol_producers(
            project,
            session_id,
            binding,
            producer_records,
            uses=uses,
        )
    except (OSError, SymbolProducerError, ValueError) as exc:
        print(f"mdb symbol-producers: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


__all__ = [
    "HYPERION_REPORT_SCHEMA",
    "ISA_SOURCE_CANDIDATE_SCHEMA",
    "NATIVE_OBSERVATION_SCHEMAS",
    "PRODUCER_JOIN_SCHEMA",
    "SYMBOL_SOURCE_CANDIDATE_SCHEMA",
    "SymbolProducerError",
    "build_symbol_lifecycle",
    "integrate_symbol_producers",
    "join_symbol_producers",
    "main",
    "seal_isa_source_candidate",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI smoke path
    raise SystemExit(main())

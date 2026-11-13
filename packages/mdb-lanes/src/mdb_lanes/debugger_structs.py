"""Adapters for importing metadata-only Saturn object manifests as debugger structs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .debugger_protocol import (
    DebuggerAddressSpec,
    DebuggerProtocolError,
    DebuggerStructSpec,
    digest,
)
from .semantic_binding import SemanticBinding, SemanticBindingError

OBJECT_MANIFEST_SCHEMA = "saturn-object-manifest-v1"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value)).strip("._:-")
    return slug or "object"


def _int_descriptor(maximum: int | None = None) -> dict[str, Any]:
    descriptor: dict[str, Any] = {"type": "index", "minimum": 0}
    if maximum is not None:
        descriptor["maximum"] = int(maximum)
    return descriptor


def _object_entry_fields(entry: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    lexical = entry.get("lexical")
    if isinstance(lexical, Mapping):
        fields["lexical"] = dict(lexical)
    if isinstance(entry.get("route"), Sequence) and not isinstance(entry["route"], (str, bytes)):
        fields["route"] = [str(site) for site in entry["route"]]
    if isinstance(entry.get("spatial"), Mapping):
        fields["spatial"] = dict(entry["spatial"])
    for name in ("properties", "evidence"):
        if isinstance(entry.get(name), Mapping):
            fields[name] = dict(entry[name])
    return fields


def object_manifest_structs(manifest: Mapping[str, Any]) -> tuple[DebuggerStructSpec, ...]:
    """Project a validated ``saturn-object-manifest-v1`` into read-only structs.

    The imported addresses are metadata/evidence addresses.  They intentionally
    do not become writable tensor aliases.  A native worker can separately add a
    writable SourceWrite address with :func:`source_write_address_spec` when it
    has a resolver and consumer-owned writer for that coordinate.
    """

    if not isinstance(manifest, Mapping) or manifest.get("schema") != OBJECT_MANIFEST_SCHEMA:
        raise DebuggerProtocolError("struct load expected saturn-object-manifest-v1")
    declared = str(manifest.get("fingerprint") or "")
    if declared:
        body = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if digest(body) != declared:
            raise DebuggerProtocolError("object manifest fingerprint mismatch")
    raw_objects = manifest.get("objects")
    if not isinstance(raw_objects, Mapping) or not raw_objects:
        raise DebuggerProtocolError("object manifest has no objects")
    text_positions = manifest.get("text_positions")
    text_maximum = int(text_positions) - 1 if isinstance(text_positions, int) else None
    structs: list[DebuggerStructSpec] = []
    for object_name, raw_entry in raw_objects.items():
        if not isinstance(raw_entry, Mapping):
            raise DebuggerProtocolError(f"object {object_name!r} is not a struct")
        object_name = str(object_name)
        slug = _slug(object_name)
        struct_name = f"object.{slug}"
        lexical = raw_entry.get("lexical")
        attribute_rows = {}
        if isinstance(lexical, Mapping) and isinstance(lexical.get("attribute_rows"), Mapping):
            attribute_rows = lexical["attribute_rows"]
        address_specs = [
            DebuggerAddressSpec(
                name=f"object-lexical-row-{slug}",
                kind="object-lexical-row",
                path_template="object[{object}][lexical][row={row}]",
                fields={
                    "object": {"type": "string", "choices": [object_name]},
                    "row": _int_descriptor(text_maximum),
                },
                struct_name=struct_name,
                invalidates=("native-conditioner-suffix",),
                numerical_contract="metadata-only",
            )
        ]
        if attribute_rows:
            attributes = sorted(str(name) for name in attribute_rows)
            address_specs.append(
                DebuggerAddressSpec(
                    name=f"object-attribute-row-{slug}",
                    kind="object-attribute-row",
                    path_template="object[{object}][attribute={attribute}][row={row}]",
                    fields={
                        "object": {"type": "string", "choices": [object_name]},
                        "attribute": {"type": "string", "choices": attributes},
                        "row": _int_descriptor(text_maximum),
                    },
                    struct_name=struct_name,
                    invalidates=("native-conditioner-suffix",),
                    numerical_contract="metadata-only",
                )
            )
        structs.append(
            DebuggerStructSpec(
                name=struct_name,
                schema=OBJECT_MANIFEST_SCHEMA,
                fields={
                    "object": object_name,
                    **_object_entry_fields(raw_entry),
                },
                address_specs=tuple(address_specs),
                metadata={
                    "manifest_fingerprint": declared or None,
                    "model_id": manifest.get("model_id"),
                    "scene_prompt": manifest.get("scene_prompt"),
                    "claim_boundary": manifest.get("claim_boundary"),
                },
            )
        )
    return tuple(structs)


def semantic_binding_path(symbol: str, boundary: str, index: int) -> str:
    """Return the stable logical path for one semantic binding location."""

    if int(index) < 0:
        raise DebuggerProtocolError("semantic binding index must be non-negative")
    return f"semantic[{symbol}][boundary={boundary}][index={int(index)}]"


def semantic_binding_struct(
    value: SemanticBinding | Mapping[str, Any],
    *,
    writable: bool = False,
) -> DebuggerStructSpec:
    """Project a semantic location list into one debugger-owned struct.

    The logical path contains no physical row.  The native adapter resolves
    ``(symbol, boundary, index)`` through the fingerprinted binding and retains
    ownership of each declared physical component.  Asking for a writable
    projection fails unless every component closes its native writer,
    consumer, invalidation, and numerical contract.
    """

    try:
        binding = value if isinstance(value, SemanticBinding) else SemanticBinding.from_dict(value)
    except SemanticBindingError as exc:
        raise DebuggerProtocolError(str(exc)) from exc
    if writable and not binding.writable:
        raise DebuggerProtocolError(
            f"semantic binding {binding.symbol!r} has no closed native write contract"
        )
    # SemanticBinding already restricts symbols to DebuggerStructSpec-safe
    # identifiers.  Preserve the exact symbol here: slug normalization can
    # collapse distinct names such as ``object.a`` and ``object.a-``.
    slug = binding.symbol
    struct_name = f"semantic.{slug}"
    addresses: tuple[DebuggerAddressSpec, ...] = ()
    if not binding.abstain:
        contracts = binding.numerical_contracts
        numerical_contract = (
            contracts[0] if len(contracts) == 1 else "semantic-binding-composite-v1"
        )
        addresses = (
            DebuggerAddressSpec(
                name=f"semantic-binding-{slug}",
                kind="semantic-binding",
                path_template="semantic[{symbol}][boundary={boundary}][index={index}]",
                fields={
                    "symbol": {"type": "name", "choices": [binding.symbol]},
                    "boundary": {"type": "name", "choices": list(binding.boundaries)},
                    "index": {
                        "type": "index",
                        "minimum": min(binding.indices),
                        "maximum": max(binding.indices),
                        "choices": list(binding.indices),
                    },
                },
                readable=True,
                writable=writable,
                struct_name=struct_name,
                source_write={
                    "schema": "saturn-debugger-semantic-binding-v1",
                    "symbol": binding.symbol,
                    "binding_fingerprint": binding.fingerprint,
                    "manifest_fingerprint": binding.manifest_fingerprint,
                    "context_fingerprint": binding.context_fingerprint,
                    "locations": [location.to_dict() for location in binding.locations],
                    "tensor_payloads_in_control_plane": False,
                },
                boundaries=binding.boundaries,
                invalidates=binding.invalidates,
                numerical_contract=numerical_contract,
            ),
        )
    return DebuggerStructSpec(
        name=struct_name,
        schema="saturn-semantic-binding-struct-v1",
        fields={
            "symbol": binding.symbol,
            "semantic_type": binding.semantic_type,
            "context": dict(binding.context),
            "locations": [location.to_dict() for location in binding.locations],
            "confidence": binding.confidence,
            "status": binding.status,
            "abstain": binding.abstain,
            "abstain_reason": binding.abstain_reason,
        },
        address_specs=addresses,
        metadata={
            "binding_fingerprint": binding.fingerprint,
            "manifest_fingerprint": binding.manifest_fingerprint,
            "context_fingerprint": binding.context_fingerprint,
            "claim_boundary": binding.claim_boundary,
            "evidence_refs": list(binding.evidence_refs),
        },
    )


__all__ = [
    "OBJECT_MANIFEST_SCHEMA",
    "object_manifest_structs",
    "semantic_binding_path",
    "semantic_binding_struct",
]

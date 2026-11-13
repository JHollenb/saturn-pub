"""Per-object SourceAddress structs as first-class debugger IO.

An object manifest turns an experiment's per-object registry (lexical token
rows, certified-route sites, spatial latent blocks, controllability evidence)
into a typed, fingerprinted receipt that can be published beside a durable
trajectory checkpoint and read back through ``saturn-debug objects``.

The manifest is metadata-only: it names coordinates and evidence, it does not
carry tensors. Addresses are prompt- and seed-local coordinates on one scene,
not universal concept labels; the schema records that boundary explicitly so a
reader cannot silently promote a registry row into a semantic claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .runtime_telemetry import _digest, _jsonable

OBJECT_MANIFEST_SCHEMA = "saturn-object-manifest-v1"

_DEFAULT_CLAIM_BOUNDARY = (
    "Addresses are prompt- and seed-local coordinates measured on one scene; "
    "lexical rows are this prompt's tokenizer layout and spatial blocks are "
    "this seed's rendering. Neither is a universal concept address, a semantic "
    "classifier, nor a certified-circuit claim."
)


class ObjectManifestError(ValueError):
    """Raised when an object manifest cannot be made unambiguous."""


def _int_list(value: Any, *, label: str, upper: int) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ObjectManifestError(f"{label} must be a sequence of integers")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ObjectManifestError(f"{label} must contain plain integers")
        if item < 0 or item >= upper:
            raise ObjectManifestError(f"{label} entry {item} outside [0, {upper})")
        out.append(int(item))
    if len(set(out)) != len(out):
        raise ObjectManifestError(f"{label} contains duplicate positions")
    return out


def _pixel_bbox_lookup(coordinate_manifest: Mapping[str, Any] | None) -> dict[int, list[int]]:
    image = (coordinate_manifest or {}).get("image")
    if not isinstance(image, Mapping):
        return {}
    tokens = image.get("tokens")
    if not isinstance(tokens, Sequence):
        return {}
    lookup: dict[int, list[int]] = {}
    for token in tokens:
        if not isinstance(token, Mapping):
            continue
        flat = token.get("flat_index")
        bbox = token.get("pixel_bbox")
        if isinstance(flat, int) and isinstance(bbox, Sequence) and len(bbox) == 4:
            lookup[int(flat)] = [int(v) for v in bbox]
    return lookup


def _spatial_entry(
    spatial: Mapping[str, Any],
    *,
    name: str,
    grid_positions: int,
    bbox_lookup: Mapping[int, list[int]],
) -> dict[str, Any]:
    blocks_by_seed = spatial.get("blocks_by_seed")
    if spatial.get("status") in {"not-measured", "not-applicable"}:
        payload = {
            "status": str(spatial["status"]),
            "instrument": str(
                spatial.get("instrument") or "no-spatial-measurement"
            ),
        }
        if spatial.get("reason") is not None:
            payload["reason"] = str(spatial["reason"])
        return payload
    if not isinstance(blocks_by_seed, Mapping) or not blocks_by_seed:
        raise ObjectManifestError(f"object {name!r} spatial address needs blocks_by_seed")
    resolved: dict[str, Any] = {}
    for seed, blocks in blocks_by_seed.items():
        positions = _int_list(
            blocks, label=f"object {name!r} spatial blocks (seed {seed})", upper=grid_positions
        )
        entry: dict[str, Any] = {"blocks": positions}
        if bbox_lookup:
            entry["pixel_bboxes"] = {
                str(block): bbox_lookup[block] for block in positions if block in bbox_lookup
            }
        resolved[str(seed)] = entry
    payload: dict[str, Any] = {
        "blocks_by_seed": resolved,
        "instrument": str(spatial.get("instrument") or "attribute_contrast_pixel_diff"),
    }
    for optional in ("instrument_agreement", "anchor"):
        if spatial.get(optional) is not None:
            payload[optional] = _jsonable(spatial[optional])
    return payload


def build_object_manifest(
    objects: Mapping[str, Mapping[str, Any]],
    *,
    scene_prompt: str,
    model_id: str,
    route: Sequence[str],
    seeds: Sequence[Any],
    text_positions: int = 512,
    grid_positions: int = 256,
    coordinate_manifest: Mapping[str, Any] | None = None,
    claim_boundary: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join per-object SourceAddress structs into one debugger receipt.

    ``objects`` maps object name -> struct with:

    * ``lexical``: ``{"rows": [...], "tokens": [...]}`` plus optional
      ``attribute_rows`` mapping property name -> row list;
    * ``spatial``: ``{"blocks_by_seed": {seed: [...]}}`` plus optional
      ``instrument``/``instrument_agreement``/``anchor``; a lexical-only
      compiler may instead declare ``{"status": "not-measured"}`` and must
      not later imply a spatial address from that receipt;
    * optional ``evidence``: JSON-able controllability summary (edit battery
      scores, stress results, isolation results);
    * optional ``properties``: JSON-able per-property back-map (route sites,
      conditioner onset depth, dose ceiling, ...).
    """

    if not objects:
        raise ObjectManifestError("object manifest requires at least one object")
    if not str(scene_prompt):
        raise ObjectManifestError("object manifest requires the scene prompt")
    route_sites = [str(site) for site in route]
    if not route_sites:
        raise ObjectManifestError("object manifest requires a non-empty route")
    bbox_lookup = _pixel_bbox_lookup(coordinate_manifest)

    entries: dict[str, Any] = {}
    for name, struct in objects.items():
        if not isinstance(struct, Mapping):
            raise ObjectManifestError(f"object {name!r} struct must be a mapping")
        lexical = struct.get("lexical")
        if not isinstance(lexical, Mapping):
            raise ObjectManifestError(f"object {name!r} needs a lexical address")
        rows = _int_list(
            lexical.get("rows"), label=f"object {name!r} lexical rows", upper=text_positions
        )
        tokens = lexical.get("tokens")
        if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
            raise ObjectManifestError(f"object {name!r} lexical tokens must be a sequence")
        if len(tokens) != len(rows):
            raise ObjectManifestError(
                f"object {name!r} lexical rows/tokens length mismatch "
                f"({len(rows)} rows, {len(tokens)} tokens)"
            )
        attribute_rows: dict[str, list[int]] = {}
        raw_attributes = lexical.get("attribute_rows") or {}
        if not isinstance(raw_attributes, Mapping):
            raise ObjectManifestError(f"object {name!r} attribute_rows must be a mapping")
        for prop, prop_rows in raw_attributes.items():
            resolved_rows = _int_list(
                prop_rows,
                label=f"object {name!r} attribute {prop!r} rows",
                upper=text_positions,
            )
            missing = [row for row in resolved_rows if row not in rows]
            if missing:
                raise ObjectManifestError(
                    f"object {name!r} attribute {prop!r} rows {missing} are outside "
                    "the object's lexical rows"
                )
            attribute_rows[str(prop)] = resolved_rows

        spatial = struct.get("spatial")
        if not isinstance(spatial, Mapping):
            raise ObjectManifestError(f"object {name!r} needs a spatial address")

        entry: dict[str, Any] = {
            "lexical": {
                "rows": rows,
                "tokens": [str(token) for token in tokens],
                "attribute_rows": attribute_rows,
            },
            "route": route_sites,
            "spatial": _spatial_entry(
                spatial, name=name, grid_positions=grid_positions, bbox_lookup=bbox_lookup
            ),
        }
        for optional in ("evidence", "properties"):
            if struct.get(optional) is not None:
                entry[optional] = _jsonable(struct[optional])
        entries[str(name)] = entry

    payload: dict[str, Any] = {
        "schema": OBJECT_MANIFEST_SCHEMA,
        "scene_prompt": str(scene_prompt),
        "model_id": str(model_id),
        "seeds": [str(seed) for seed in seeds],
        "text_positions": int(text_positions),
        "grid_positions": int(grid_positions),
        "objects": entries,
        "claim_boundary": str(claim_boundary or _DEFAULT_CLAIM_BOUNDARY),
        "join_semantics": (
            "lexical rows are tokenizer coordinates on the scene prompt; spatial blocks "
            "are packed latent-grid coordinates joined to pixel bboxes through the "
            "checkpoint's coordinate manifest; evidence entries are measured on the "
            "recorded jobs only"
        ),
    }
    if provenance is not None:
        payload["provenance"] = _jsonable(provenance)
    coordinate_fingerprint = (coordinate_manifest or {}).get("fingerprint")
    if coordinate_fingerprint is not None:
        payload["coordinate_manifest_fingerprint"] = str(coordinate_fingerprint)
    payload["fingerprint"] = _digest(
        {key: value for key, value in payload.items() if key != "fingerprint"}
    )
    return payload


def validate_object_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Check schema + fingerprint of an existing manifest and return it as dict."""

    if not isinstance(manifest, Mapping):
        raise ObjectManifestError("object manifest must be a mapping")
    if manifest.get("schema") != OBJECT_MANIFEST_SCHEMA:
        raise ObjectManifestError(
            f"unsupported object manifest schema {manifest.get('schema')!r}"
        )
    payload = {str(key): value for key, value in manifest.items()}
    declared = str(payload.get("fingerprint") or "")
    recomputed = _digest({key: value for key, value in payload.items() if key != "fingerprint"})
    if declared != recomputed:
        raise ObjectManifestError("object manifest fingerprint mismatch")
    if not isinstance(payload.get("objects"), Mapping) or not payload["objects"]:
        raise ObjectManifestError("object manifest has no objects")
    return payload


__all__ = [
    "OBJECT_MANIFEST_SCHEMA",
    "ObjectManifestError",
    "build_object_manifest",
    "validate_object_manifest",
]

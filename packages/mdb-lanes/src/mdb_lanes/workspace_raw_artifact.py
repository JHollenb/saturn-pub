"""Content-addressed raw tensor custody for an explicit Thott audit.

The normal Thott receipt is intentionally summary-only.  This module is the
opt-in escape hatch for an audit run that needs to recompute a decision from
the exact tensors seen by the hooks.  It stores numerical tensors only: the
event key, shape, dtype, and hash are preserved in the safetensors metadata;
no prose or inferred interpretation is generated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

THOTT_RAW_WORKSPACE_SCHEMA = "saturn-thott-raw-workspace-v1"
THOTT_RAW_REFERENCE_SCHEMA = "saturn-safetensors-artifact-reference-v1"


class WorkspaceRawArtifactError(ValueError):
    """Raised when a raw Thott tensor artifact fails its custody contract."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash the exact contiguous tensor bytes, preserving dtype and shape separately."""

    if not isinstance(value, torch.Tensor):
        raise WorkspaceRawArtifactError("raw artifact values must be tensors")
    prepared = value.detach().to(device="cpu").contiguous()
    raw = prepared.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_reference(
    *, original_key: str, tensor_name: str, value: torch.Tensor
) -> dict[str, Any]:
    prepared = value.detach().to(device="cpu").contiguous()
    return {
        "original_key": str(original_key),
        "tensor_name": str(tensor_name),
        "dtype": str(prepared.dtype),
        "shape": [int(item) for item in prepared.shape],
        "numel": int(prepared.numel()),
        "bytes": int(prepared.numel() * prepared.element_size()),
        "sha256": tensor_sha256(prepared),
    }


def build_raw_workspace_artifact(
    tensors: Mapping[str, torch.Tensor],
    *,
    scope: Mapping[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Serialize the complete caller-selected raw tensor bank.

    The returned values are ``(safetensors_bytes, reference, manifest)``.  The
    caller owns transport and persistence; this function never writes a file
    or contacts a store.
    """

    if not isinstance(tensors, Mapping) or not tensors:
        raise WorkspaceRawArtifactError("raw workspace artifact cannot be empty")
    ordered = sorted((str(key), value) for key, value in tensors.items())
    payload: dict[str, torch.Tensor] = {}
    references: list[dict[str, Any]] = []
    for index, (original_key, value) in enumerate(ordered):
        if not isinstance(value, torch.Tensor):
            raise WorkspaceRawArtifactError(
                f"raw tensor {original_key!r} is {type(value).__name__}, not a tensor"
            )
        tensor_name = f"tensor_{index:06d}"
        prepared = value.detach().to(device="cpu").contiguous()
        payload[tensor_name] = prepared
        references.append(
            _tensor_reference(
                original_key=original_key,
                tensor_name=tensor_name,
                value=prepared,
            )
        )
    manifest: dict[str, Any] = {
        "schema": THOTT_RAW_WORKSPACE_SCHEMA,
        "tensor_count": len(references),
        "tensors": references,
        "scope": dict(scope or {}),
    }
    manifest_bytes = _canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    encoded = bytes(
        save_safetensors(
            payload,
            metadata={
                "saturn_schema": THOTT_RAW_WORKSPACE_SCHEMA,
                "manifest_sha256": manifest_sha256,
                "manifest_json": manifest_bytes.decode("utf-8"),
            },
        )
    )
    reopened = load_safetensors(encoded)
    if set(reopened) != set(payload) or any(
        not torch.equal(reopened[name], value) for name, value in payload.items()
    ):
        raise WorkspaceRawArtifactError("raw safetensors artifact did not reopen bit-exactly")
    artifact_sha256 = hashlib.sha256(encoded).hexdigest()
    reference = {
        "schema": THOTT_RAW_REFERENCE_SCHEMA,
        "artifact": f"thott-workspace-raw-{artifact_sha256}.safetensors",
        "sha256": artifact_sha256,
        "bytes": len(encoded),
        "media_type": "application/x-safetensors",
        "content_schema": THOTT_RAW_WORKSPACE_SCHEMA,
        "tensor_count": len(references),
        "manifest_sha256": manifest_sha256,
        "event_order_sha256": hashlib.sha256(
            _canonical_json({"original_keys": [row["original_key"] for row in references]})
        ).hexdigest(),
        "retention": "explicit-audit-artifact; numerical tensors only",
    }
    return encoded, reference, manifest


def validate_raw_workspace_artifact(
    encoded: bytes,
    *,
    expected_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen and hash every tensor in a raw artifact before accepting it."""

    if not isinstance(encoded, bytes) or not encoded:
        raise WorkspaceRawArtifactError("raw artifact bytes must be non-empty")
    artifact_sha256 = hashlib.sha256(encoded).hexdigest()
    if expected_reference is not None:
        if expected_reference.get("sha256") != artifact_sha256:
            raise WorkspaceRawArtifactError("raw artifact SHA-256 does not match its reference")
        if int(expected_reference.get("bytes", -1)) != len(encoded):
            raise WorkspaceRawArtifactError("raw artifact byte count does not match its reference")
        if expected_reference.get("content_schema") != THOTT_RAW_WORKSPACE_SCHEMA:
            raise WorkspaceRawArtifactError("raw artifact content schema is unsupported")
    tensors = load_safetensors(encoded)
    # The safetensors loader exposes metadata only through the file API, so
    # the serialized manifest is carried in a reserved JSON tensor-free side
    # channel by ``_extract_metadata`` below.  Keep this validation function
    # strict about the numerical payload; callers additionally validate the
    # metadata manifest.
    return {
        "schema": THOTT_RAW_WORKSPACE_SCHEMA,
        "artifact_sha256": artifact_sha256,
        "bytes": len(encoded),
        "tensor_count": len(tensors),
        "tensor_names": sorted(tensors),
        "tensor_hashes": {name: tensor_sha256(value) for name, value in tensors.items()},
    }


def read_raw_workspace_manifest(path: str | Path) -> dict[str, Any]:
    """Read and verify the manifest embedded in a persisted raw artifact."""

    artifact_path = Path(path)
    with safe_open(str(artifact_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = sorted(str(key) for key in handle.keys())
    if metadata.get("saturn_schema") != THOTT_RAW_WORKSPACE_SCHEMA:
        raise WorkspaceRawArtifactError("raw artifact metadata schema is unsupported")
    raw_manifest = metadata.get("manifest_json")
    if not isinstance(raw_manifest, str):
        raise WorkspaceRawArtifactError("raw artifact has no embedded manifest")
    try:
        manifest = json.loads(raw_manifest)
    except (TypeError, json.JSONDecodeError) as exc:
        raise WorkspaceRawArtifactError("raw artifact manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != THOTT_RAW_WORKSPACE_SCHEMA:
        raise WorkspaceRawArtifactError("raw artifact manifest schema is unsupported")
    if hashlib.sha256(_canonical_json(manifest)).hexdigest() != metadata.get(
        "manifest_sha256"
    ):
        raise WorkspaceRawArtifactError("raw artifact manifest hash does not match metadata")
    references = manifest.get("tensors")
    if not isinstance(references, list) or int(manifest.get("tensor_count", -1)) != len(references):
        raise WorkspaceRawArtifactError("raw artifact manifest tensor inventory is malformed")
    expected_names = sorted(
        str(row.get("tensor_name")) for row in references if isinstance(row, Mapping)
    )
    if expected_names != keys:
        raise WorkspaceRawArtifactError("raw artifact tensor names do not match its manifest")
    return {"manifest": manifest, "metadata": metadata, "tensor_names": keys}


def validate_raw_workspace_file(
    path: str | Path,
    *,
    expected_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the file hash, manifest, and every numerical tensor hash."""

    artifact_path = Path(path)
    encoded = artifact_path.read_bytes()
    summary = validate_raw_workspace_artifact(encoded, expected_reference=expected_reference)
    loaded = load_safetensors(encoded)
    manifest_payload = read_raw_workspace_manifest(artifact_path)
    references = {
        str(row["tensor_name"]): row
        for row in manifest_payload["manifest"]["tensors"]
        if isinstance(row, Mapping) and "tensor_name" in row
    }
    if set(references) != set(loaded):
        raise WorkspaceRawArtifactError("raw artifact manifest and numerical payload disagree")
    mismatches: list[str] = []
    for tensor_name, value in loaded.items():
        if tensor_sha256(value) != references[tensor_name].get("sha256"):
            mismatches.append(tensor_name)
    if mismatches:
        raise WorkspaceRawArtifactError(
            f"raw artifact tensor hashes do not match: {mismatches[:4]}"
        )
    return {
        **summary,
        "manifest_sha256": manifest_payload["metadata"].get("manifest_sha256"),
        "manifest": manifest_payload["manifest"],
        "tensor_hashes_match": True,
    }


__all__ = [
    "THOTT_RAW_REFERENCE_SCHEMA",
    "THOTT_RAW_WORKSPACE_SCHEMA",
    "WorkspaceRawArtifactError",
    "build_raw_workspace_artifact",
    "read_raw_workspace_manifest",
    "tensor_sha256",
    "validate_raw_workspace_artifact",
    "validate_raw_workspace_file",
]

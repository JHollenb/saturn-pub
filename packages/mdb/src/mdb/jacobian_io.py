"""Standalone custody and I/O layer for frozen Jacobian workspace lenses.

The classes in this module own no model runtime.  They bind caller supplied
tensors to strict-JSON identities, keep private snapshots, and persist only
JSON plus safetensors numerical payloads.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from .jacobian_workspace import (
    JacobianWorkspaceError,
    JObservation,
    derive_dictionary_rows,
    metadata_sha256,
    normalized_atom_steering,
    observe_j,
    span_coordinate_edit,
    tensor_payload_identity,
    validate_lens_metadata,
)
from .tensor_hash import prefixed_tensor_sha256, tensor_bytes_sha256

JACOBIAN_LENS_ARTIFACT_SCHEMA = "saturn-jacobian-lens-artifact-v1"
JACOBIAN_PREPARED_WRITE_SCHEMA = "saturn-j-prepared-write-v1"
JACOBIAN_PREPARED_WRITE_ARTIFACT_SCHEMA = "saturn-j-prepared-write-artifact-v1"


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise JacobianWorkspaceError(f"{label} must be strict JSON") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JacobianWorkspaceError(f"{label} must be a mapping")
    copied = _json_copy(dict(value), label)
    if not isinstance(copied, dict):  # pragma: no cover - JSON round trip guarantees this
        raise JacobianWorkspaceError(f"{label} must be a mapping")
    return copied


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JacobianWorkspaceError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise JacobianWorkspaceError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise JacobianWorkspaceError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _tensor_snapshot(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise JacobianWorkspaceError(f"{label} must be a torch.Tensor")
    if not value.is_floating_point():
        raise JacobianWorkspaceError(f"{label} must have floating dtype")
    snapshot = value.detach().to(device="cpu").contiguous().clone()
    if not bool(torch.isfinite(snapshot).all().item()):
        raise JacobianWorkspaceError(f"{label} contains non-finite values")
    return snapshot


def _tensor_reference(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise JacobianWorkspaceError(f"{label} must be a torch.Tensor")
    if not value.is_floating_point():
        raise JacobianWorkspaceError(f"{label} must have floating dtype")
    reference = value.detach()
    if not bool(torch.isfinite(reference).all().item()):
        raise JacobianWorkspaceError(f"{label} contains non-finite values")
    return reference


def _header_tensor_sha256(value: torch.Tensor) -> str:
    prefix = f"{value.dtype}|{tuple(value.shape)}|".encode("utf-8")
    return prefixed_tensor_sha256(prefix, value)


def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.storage_offset() == right.storage_offset()
        and left.data_ptr() == right.data_ptr()
    )


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise JacobianWorkspaceError(f"artifact path contains different bytes: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise JacobianWorkspaceError(
                    f"artifact path contains different bytes: {path}"
                ) from None
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _artifact_save(
    root: str | Path,
    *,
    artifact_schema: str,
    metadata: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    target = Path(root).expanduser().resolve()
    prepared = {
        name: tensor.detach().to(device="cpu").contiguous() for name, tensor in tensors.items()
    }
    encoded = bytes(save_safetensors(prepared))
    tensor_rows = {name: tensor_payload_identity(value) for name, value in prepared.items()}
    body = {
        "schema": artifact_schema,
        "metadata": _mapping(metadata, "artifact metadata"),
        "tensor_file": "tensors.safetensors",
        "tensor_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "tensors": tensor_rows,
    }
    envelope = {**body, "fingerprint": metadata_sha256(body)}
    manifest_bytes = (
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    _write_immutable(target / "tensors.safetensors", encoded)
    _write_immutable(target / "manifest.json", manifest_bytes)


def _artifact_open(
    root: str | Path, *, artifact_schema: str, tensor_names: set[str]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    target = Path(root).expanduser().resolve()
    try:
        envelope = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        encoded = (target / "tensors.safetensors").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JacobianWorkspaceError("cannot read Jacobian artifact") from exc
    if not isinstance(envelope, dict) or envelope.get("schema") != artifact_schema:
        raise JacobianWorkspaceError("unsupported Jacobian artifact schema")
    fingerprint = envelope.get("fingerprint")
    body = {key: value for key, value in envelope.items() if key != "fingerprint"}
    if fingerprint != metadata_sha256(body):
        raise JacobianWorkspaceError("Jacobian artifact manifest fingerprint mismatch")
    if envelope.get("tensor_file") != "tensors.safetensors":
        raise JacobianWorkspaceError("Jacobian artifact tensor path is unsupported")
    if envelope.get("tensor_file_sha256") != hashlib.sha256(encoded).hexdigest():
        raise JacobianWorkspaceError("Jacobian artifact tensor file hash mismatch")
    try:
        tensors = load_safetensors(encoded)
    except Exception as exc:
        raise JacobianWorkspaceError("Jacobian artifact is not valid safetensors") from exc
    if set(tensors) != tensor_names:
        raise JacobianWorkspaceError("Jacobian artifact tensor inventory mismatch")
    rows = envelope.get("tensors")
    if not isinstance(rows, dict) or any(
        rows.get(name) != tensor_payload_identity(value) for name, value in tensors.items()
    ):
        raise JacobianWorkspaceError("Jacobian artifact tensor identity mismatch")
    metadata = _mapping(envelope.get("metadata"), "artifact metadata")
    return metadata, {name: value.clone() for name, value in tensors.items()}


class FrozenJacobianLens:
    """Verified immutable snapshot of a fitted dense Jacobian transport."""

    def __init__(self, manifest: Mapping[str, Any], transport: torch.Tensor) -> None:
        snapshot = _tensor_snapshot(transport, "transport")
        manifest_payload = _mapping(manifest, "lens manifest")
        if "fingerprint" not in manifest_payload:
            manifest_payload["fingerprint"] = metadata_sha256(manifest_payload)
        validated = validate_lens_metadata(
            manifest_payload, matrix_shape=snapshot.shape, matrix_dtype=snapshot.dtype
        )
        declared = validated["transport"]["sha256"]
        if tensor_bytes_sha256(snapshot) != declared:
            raise JacobianWorkspaceError("transport payload SHA-256 mismatch")
        self._manifest = _mapping(validated, "lens manifest")
        self._transport = snapshot

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def transport_for(
        self, device: torch.device | str, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if dtype is not None and dtype != self._transport.dtype:
            raise JacobianWorkspaceError("requested dtype differs from frozen lens dtype")
        return self._transport.to(device=torch.device(device), copy=True).detach()

    def save(self, root: str | Path) -> None:
        _artifact_save(
            root,
            artifact_schema=JACOBIAN_LENS_ARTIFACT_SCHEMA,
            metadata=self._manifest,
            tensors={"transport": self._transport},
        )

    @classmethod
    def open(cls, root: str | Path) -> "FrozenJacobianLens":
        metadata, tensors = _artifact_open(
            root,
            artifact_schema=JACOBIAN_LENS_ARTIFACT_SCHEMA,
            tensor_names={"transport"},
        )
        return cls(metadata, tensors["transport"])


def _validate_source(
    source: Mapping[str, Any],
    residual: torch.Tensor,
    *,
    model_identity: Mapping[str, Any],
    completed_layer_boundary: int,
    execution_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = _mapping(source, "source")
    for key in ("model_abi_sha256",):
        _digest(payload.get(key), f"source.{key}")
    for key in ("cut_ref", "branch"):
        _text(payload.get(key), f"source.{key}")
    _integer(payload.get("token_position"), "source.token_position")
    boundary = _integer(
        payload.get("completed_layer_boundary"),
        "source.completed_layer_boundary",
        1,
    )
    if boundary != completed_layer_boundary:
        raise JacobianWorkspaceError("source completed layer boundary mismatch")
    expected_abi = model_identity.get("model_abi_sha256")
    if expected_abi is not None and payload["model_abi_sha256"] != expected_abi:
        raise JacobianWorkspaceError("source model ABI mismatch")
    if execution_binding is not None:
        if payload["model_abi_sha256"] != execution_binding["model_abi_sha256"]:
            raise JacobianWorkspaceError("source execution-binding model ABI mismatch")
        numerical_contract = _text(payload.get("numerical_contract"), "source.numerical_contract")
        if numerical_contract != execution_binding["numerical_contract"]:
            raise JacobianWorkspaceError("source execution-binding numerical contract mismatch")
        if "session_id" in execution_binding:
            session_id = _text(payload.get("session_id"), "source.session_id")
            if session_id != execution_binding["session_id"]:
                raise JacobianWorkspaceError("source execution-binding session ID mismatch")
    if "input_tokens_sha256" in payload:
        _digest(payload["input_tokens_sha256"], "source.input_tokens_sha256")
    activation = tensor_payload_identity(residual)
    declared_activation = payload.get("activation")
    if declared_activation is not None and declared_activation != activation:
        raise JacobianWorkspaceError("source activation identity mismatch")
    payload["activation"] = activation
    return _mapping(payload, "source")


def _validate_consumer(consumer: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(consumer, "consumer")
    _text(payload.get("name"), "consumer.name")
    _integer(payload.get("horizon"), "consumer.horizon")
    return payload


def _validate_execution_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(value, "execution_binding")
    _digest(payload.get("model_abi_sha256"), "execution_binding.model_abi_sha256")
    _text(payload.get("numerical_contract"), "execution_binding.numerical_contract")
    if "session_id" in payload:
        _text(payload["session_id"], "execution_binding.session_id")
    return payload


def _logical_address(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if value is None else _mapping(value, "logical_address")


class JacobianReader:
    """Identity-bound reader over a frozen lens and hash-bound live decoder views."""

    def __init__(
        self,
        lens: FrozenJacobianLens,
        decoder_weight: torch.Tensor,
        *,
        model_identity: Mapping[str, Any],
        application_lane: str,
        rms_weight: torch.Tensor | None = None,
        eps: float = 1e-6,
        final_norm: Callable[[torch.Tensor], torch.Tensor] | None = None,
        output_head: Callable[[torch.Tensor], torch.Tensor] | None = None,
        execution_binding: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(lens, FrozenJacobianLens):
            raise JacobianWorkspaceError("lens must be a FrozenJacobianLens")
        identity = _mapping(model_identity, "model_identity")
        lane = _text(application_lane, "application_lane")
        manifest = validate_lens_metadata(
            lens.manifest,
            matrix_shape=lens._transport.shape,
            matrix_dtype=lens._transport.dtype,
            application_lane=lane,
            expected_model_identity=identity,
        )
        if identity != manifest["model_identity"]:
            raise JacobianWorkspaceError("model identity must exactly match the lens")
        binding = (
            None if execution_binding is None else _validate_execution_binding(execution_binding)
        )
        if (
            binding is not None
            and identity.get("model_abi_sha256") is not None
            and identity["model_abi_sha256"] != binding["model_abi_sha256"]
        ):
            raise JacobianWorkspaceError(
                "execution binding model ABI does not match lens model identity"
            )
        # Decoder/output-head tensors can be several GiB.  Keep a detached
        # caller-owned view and bind it by the manifest hash instead of making
        # a permanent duplicate.  Every operation rechecks that binding.
        decoder = _tensor_reference(decoder_weight, "decoder_weight")
        width = int(manifest["coordinate_map"]["residual_width"])
        if tuple(decoder.shape) != (int(manifest["decoder"]["vocab_size"]), width):
            raise JacobianWorkspaceError("decoder geometry does not match lens manifest")
        if decoder.dtype != lens._transport.dtype:
            raise JacobianWorkspaceError("decoder dtype does not match lens transport")
        decoder_meta = manifest["decoder"]
        decoder_digests = {
            decoder_meta[key]
            for key in (
                "weight_sha256",
                "weights_sha256",
                "decoder_weight_sha256",
                "unembedding_weight_sha256",
            )
            if key in decoder_meta
        }
        if len(decoder_digests) > 1:
            raise JacobianWorkspaceError("decoder manifest contains conflicting weight hashes")
        self._decoder_sha256 = next(iter(decoder_digests), None)
        if self._decoder_sha256 is not None:
            _digest(self._decoder_sha256, "decoder weight SHA-256")
            if self._decoder_sha256 != _header_tensor_sha256(decoder):
                raise JacobianWorkspaceError("decoder weight SHA-256 mismatch")
        rms = None if rms_weight is None else _tensor_reference(rms_weight, "rms_weight")
        if rms is not None and (tuple(rms.shape) != (width,) or rms.dtype != decoder.dtype):
            raise JacobianWorkspaceError("RMS weight geometry/dtype does not match lens")
        rms_digests = {
            decoder_meta[key]
            for key in ("rms_weight_sha256", "final_norm_weight_sha256")
            if key in decoder_meta
        }
        if len(rms_digests) > 1:
            raise JacobianWorkspaceError("decoder manifest contains conflicting RMS hashes")
        self._rms_sha256 = next(iter(rms_digests), None)
        native_norm_weight = (
            None
            if final_norm is None or not hasattr(final_norm, "weight")
            else _tensor_reference(getattr(final_norm, "weight"), "final_norm.weight")
        )
        if self._rms_sha256 is not None:
            _digest(self._rms_sha256, "RMS weight SHA-256")
            norm_candidates = [value for value in (rms, native_norm_weight) if value is not None]
            if not norm_candidates or any(
                self._rms_sha256 != _header_tensor_sha256(value) for value in norm_candidates
            ):
                raise JacobianWorkspaceError("RMS weight SHA-256 mismatch")
        if (
            isinstance(eps, bool)
            or not isinstance(eps, (int, float))
            or not math.isfinite(eps)
            or eps <= 0
        ):
            raise JacobianWorkspaceError("eps must be a finite positive number")
        if (final_norm is None) != (output_head is None):
            raise JacobianWorkspaceError("final_norm and output_head must be supplied together")
        if final_norm is None and rms is None:
            raise JacobianWorkspaceError("supply native final_norm/output_head or rms_weight")
        native_head_weight = (
            None
            if output_head is None or not hasattr(output_head, "weight")
            else _tensor_reference(getattr(output_head, "weight"), "output_head.weight")
        )
        self._native_head_alias = native_head_weight is not None and _same_tensor_view(
            native_head_weight, decoder
        )
        if (
            native_head_weight is not None
            and not self._native_head_alias
            and _header_tensor_sha256(native_head_weight) != _header_tensor_sha256(decoder)
        ):
            raise JacobianWorkspaceError("native output head weight does not match decoder weight")
        self.lens = lens
        self._manifest = manifest
        self._decoder_weight = decoder
        self._rms_weight = rms
        self._native_norm_weight = native_norm_weight
        self._native_head_weight = native_head_weight
        self._model_identity = identity
        self._execution_binding = binding
        self.application_lane = lane
        self.eps = float(eps)
        self.final_norm = final_norm
        self.output_head = output_head

    @property
    def model_identity(self) -> dict[str, Any]:
        return copy.deepcopy(self._model_identity)

    @property
    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    @property
    def execution_binding(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._execution_binding)

    def _verify_consumer_tensors(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._native_norm_weight is not None:
            current_norm_weight = getattr(self.final_norm, "weight", None)
            if not isinstance(current_norm_weight, torch.Tensor) or not _same_tensor_view(
                current_norm_weight.detach(), self._native_norm_weight
            ):
                raise JacobianWorkspaceError(
                    "native final norm weight was rebound after reader construction"
                )
        if self._native_head_weight is not None:
            current_head_weight = getattr(self.output_head, "weight", None)
            if not isinstance(current_head_weight, torch.Tensor) or not _same_tensor_view(
                current_head_weight.detach(), self._native_head_weight
            ):
                raise JacobianWorkspaceError(
                    "native output head weight was rebound after reader construction"
                )
        if self._decoder_weight.device != device:
            raise JacobianWorkspaceError("decoder and residual must share a device")
        if self._decoder_weight.dtype != dtype:
            raise JacobianWorkspaceError("decoder and residual must share a dtype")
        decoder_digest = (
            _header_tensor_sha256(self._decoder_weight)
            if self._decoder_sha256 is not None or self._native_head_weight is not None
            else None
        )
        if self._decoder_sha256 is not None and self._decoder_sha256 != decoder_digest:
            raise JacobianWorkspaceError("decoder weight changed after reader construction")
        if self._rms_weight is not None:
            if self._rms_weight.device != device or self._rms_weight.dtype != dtype:
                raise JacobianWorkspaceError("RMS weight and residual must share device/dtype")
            if self._rms_sha256 is not None and self._rms_sha256 != _header_tensor_sha256(
                self._rms_weight
            ):
                raise JacobianWorkspaceError("RMS weight changed after reader construction")
        if self._native_norm_weight is not None:
            if self._native_norm_weight.device != device or self._native_norm_weight.dtype != dtype:
                raise JacobianWorkspaceError(
                    "native final norm weight and residual must share device/dtype"
                )
            if self._rms_sha256 is not None and self._rms_sha256 != _header_tensor_sha256(
                self._native_norm_weight
            ):
                raise JacobianWorkspaceError(
                    "native final norm weight changed after reader construction"
                )
        if self._native_head_weight is not None:
            if self._native_head_weight.device != device or self._native_head_weight.dtype != dtype:
                raise JacobianWorkspaceError(
                    "native output head weight and residual must share device/dtype"
                )
            if (
                not self._native_head_alias
                and _header_tensor_sha256(self._native_head_weight) != decoder_digest
            ):
                raise JacobianWorkspaceError(
                    "native output head weight changed relative to decoder weight"
                )

    def _source(self, source: Mapping[str, Any], residual: torch.Tensor) -> dict[str, Any]:
        return _validate_source(
            source,
            residual,
            model_identity=self._model_identity,
            completed_layer_boundary=int(
                self._manifest["coordinate_map"]["completed_layer_boundary"]
            ),
            execution_binding=self._execution_binding,
        )

    def read(
        self,
        residual: torch.Tensor,
        *,
        source: Mapping[str, Any],
        token_subset: Sequence[int] | None = None,
        top_k: int = 10,
        logical_address: Mapping[str, Any] | None = None,
    ) -> JObservation:
        source_payload = self._source(source, residual)
        self._verify_consumer_tensors(residual.device, residual.dtype)
        transport = self.lens.transport_for(residual.device, residual.dtype)
        decoder = self._decoder_weight
        rms = self._rms_weight
        observation = observe_j(
            residual,
            transport,
            decoder,
            rms,
            eps=self.eps,
            token_subset=token_subset,
            top_k=top_k,
            final_norm=self.final_norm,
            output_head=self.output_head,
        )
        metadata = {
            **observation.metadata,
            "source": source_payload,
            "lens": {
                "fingerprint": self._manifest["fingerprint"],
                "transport": copy.deepcopy(self._manifest["transport"]),
            },
            "address": _logical_address(logical_address),
            "execution_binding": self.execution_binding,
            "readout_policy": {
                "application_lane": self.application_lane,
                "token_subset": None if token_subset is None else list(token_subset),
                "top_k": top_k,
            },
        }
        metadata["fingerprint"] = metadata_sha256(
            {key: value for key, value in metadata.items() if key != "fingerprint"}
        )
        return JObservation(
            transported=observation.transported.clone(),
            raw_scores=observation.raw_scores.clone(),
            normalized_scores=observation.normalized_scores.clone(),
            selected_token_ids=observation.selected_token_ids.clone(),
            selected_raw_scores=observation.selected_raw_scores.clone(),
            selected_normalized_scores=observation.selected_normalized_scores.clone(),
            metadata=metadata,
        )


@dataclass(frozen=True)
class PreparedJWrite:
    candidate: torch.Tensor
    delta: torch.Tensor
    metadata: dict[str, Any]

    def save(self, root: str | Path) -> None:
        _artifact_save(
            root,
            artifact_schema=JACOBIAN_PREPARED_WRITE_ARTIFACT_SCHEMA,
            metadata=self.metadata,
            tensors={"candidate": self.candidate, "delta": self.delta},
        )

    @classmethod
    def open(cls, root: str | Path) -> "PreparedJWrite":
        metadata, tensors = _artifact_open(
            root,
            artifact_schema=JACOBIAN_PREPARED_WRITE_ARTIFACT_SCHEMA,
            tensor_names={"candidate", "delta"},
        )
        _validate_prepared_metadata(metadata, tensors["candidate"], tensors["delta"])
        return cls(
            candidate=tensors["candidate"].clone(),
            delta=tensors["delta"].clone(),
            metadata=metadata,
        )


def _native_vector(residual: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not isinstance(residual, torch.Tensor) or residual.ndim < 1:
        raise JacobianWorkspaceError("residual must be a tensor with a width dimension")
    if any(size != 1 for size in residual.shape[:-1]):
        raise JacobianWorkspaceError("a write must retain exactly one token")
    if residual.dtype not in {torch.float32, torch.float64, torch.bfloat16}:
        raise JacobianWorkspaceError("writes support float32, float64, or bfloat16 residuals")
    if not bool(torch.isfinite(residual.detach()).all().item()):
        raise JacobianWorkspaceError("residual contains non-finite values")
    return residual.reshape(-1), tuple(residual.shape)


def _validate_prepared_metadata(
    metadata: Mapping[str, Any], candidate: torch.Tensor, delta: torch.Tensor
) -> dict[str, Any]:
    payload = _mapping(metadata, "prepared write metadata")
    if payload.get("schema") != JACOBIAN_PREPARED_WRITE_SCHEMA:
        raise JacobianWorkspaceError("unsupported prepared write schema")
    fingerprint = payload.get("fingerprint")
    body = {key: value for key, value in payload.items() if key != "fingerprint"}
    if fingerprint != metadata_sha256(body):
        raise JacobianWorkspaceError("prepared write metadata fingerprint mismatch")
    if payload.get("candidate") != tensor_payload_identity(candidate):
        raise JacobianWorkspaceError("prepared candidate identity mismatch")
    if payload.get("delta") != tensor_payload_identity(delta):
        raise JacobianWorkspaceError("prepared delta identity mismatch")
    _validate_consumer(payload.get("consumer"))
    return payload


class JacobianWriter:
    """Prepare and validate source-bound residual candidates."""

    def __init__(self, reader: JacobianReader) -> None:
        if not isinstance(reader, JacobianReader):
            raise JacobianWorkspaceError("reader must be a JacobianReader")
        self.reader = reader

    def _prepared(
        self,
        *,
        residual: torch.Tensor,
        source: Mapping[str, Any],
        candidate: torch.Tensor,
        delta: torch.Tensor,
        dictionary: Mapping[str, Any],
        recipe: Mapping[str, Any],
        consumer: Mapping[str, Any],
        logical_address: Mapping[str, Any] | None,
        native_shape: tuple[int, ...],
    ) -> PreparedJWrite:
        source_payload = self.reader._source(source, residual)
        candidate = candidate.reshape(native_shape).detach().clone()
        delta = delta.reshape(native_shape).detach().clone()
        body = {
            "schema": JACOBIAN_PREPARED_WRITE_SCHEMA,
            "source": source_payload,
            "lens": {
                "fingerprint": self.reader._manifest["fingerprint"],
                "application_lane": self.reader.application_lane,
            },
            "address": _logical_address(logical_address),
            "execution_binding": self.reader.execution_binding,
            "dictionary": _mapping(dictionary, "dictionary recipe"),
            "recipe": _mapping(recipe, "write recipe"),
            "consumer": _validate_consumer(consumer),
            "candidate": tensor_payload_identity(candidate),
            "delta": tensor_payload_identity(delta),
        }
        metadata = {**body, "fingerprint": metadata_sha256(body)}
        return PreparedJWrite(candidate=candidate, delta=delta, metadata=metadata)

    def _atoms(
        self, token_ids: Sequence[int], device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.reader._verify_consumer_tensors(device, dtype)
        transport = self.reader.lens.transport_for(device, dtype)
        decoder = self.reader._decoder_weight
        rows = derive_dictionary_rows(transport, decoder, token_ids, normalize=True)
        return rows.rows, rows.metadata

    def prepare_span(
        self,
        residual: torch.Tensor,
        *,
        source: Mapping[str, Any],
        token_ids: Sequence[int],
        target_coordinates: Sequence[float] | torch.Tensor,
        dose: float = 1,
        consumer: Mapping[str, Any],
        logical_address: Mapping[str, Any] | None = None,
    ) -> PreparedJWrite:
        vector, native_shape = _native_vector(residual)
        self.reader._source(source, residual)
        atoms, dictionary = self._atoms(token_ids, residual.device, residual.dtype)
        work_dtype = torch.float32 if residual.dtype == torch.bfloat16 else residual.dtype
        if isinstance(target_coordinates, torch.Tensor):
            target = target_coordinates.detach().to(device=residual.device, dtype=work_dtype)
        else:
            target = torch.tensor(target_coordinates, device=residual.device, dtype=work_dtype)
        edit = span_coordinate_edit(vector.to(work_dtype), atoms.to(work_dtype), target, dose=dose)
        candidate = edit.candidate.to(dtype=residual.dtype)
        if float(dose) == 0.0:
            candidate = vector.clone()
        delta = candidate - vector
        return self._prepared(
            residual=residual,
            source=source,
            candidate=candidate,
            delta=delta,
            dictionary=dictionary,
            recipe={
                "operation": "signed-span-coordinate-edit",
                "dose": float(dose),
                "token_ids": list(token_ids),
                "target_coordinates": [float(item) for item in target.detach().cpu().tolist()],
                "work_dtype": str(work_dtype),
                "native_dtype": str(residual.dtype),
                "native_cast_applied": residual.dtype == torch.bfloat16,
                "noop_exact": float(dose) == 0.0,
                "algebra": edit.metadata,
            },
            consumer=consumer,
            logical_address=logical_address,
            native_shape=native_shape,
        )

    def prepare_steer(
        self,
        residual: torch.Tensor,
        *,
        source: Mapping[str, Any],
        token_id: int,
        dose: float,
        consumer: Mapping[str, Any],
        logical_address: Mapping[str, Any] | None = None,
    ) -> PreparedJWrite:
        vector, native_shape = _native_vector(residual)
        self.reader._source(source, residual)
        atoms, dictionary = self._atoms([token_id], residual.device, residual.dtype)
        work_dtype = torch.float32 if residual.dtype == torch.bfloat16 else residual.dtype
        edit = normalized_atom_steering(vector.to(work_dtype), atoms[0].to(work_dtype), dose=dose)
        candidate = edit.candidate.to(dtype=residual.dtype)
        if float(dose) == 0.0:
            candidate = vector.clone()
        delta = candidate - vector
        return self._prepared(
            residual=residual,
            source=source,
            candidate=candidate,
            delta=delta,
            dictionary=dictionary,
            recipe={
                "operation": "normalized-dictionary-atom-steering",
                "dose": float(dose),
                "token_id": int(token_id),
                "work_dtype": str(work_dtype),
                "native_dtype": str(residual.dtype),
                "native_cast_applied": residual.dtype == torch.bfloat16,
                "noop_exact": float(dose) == 0.0,
                "algebra": edit.metadata,
            },
            consumer=consumer,
            logical_address=logical_address,
            native_shape=native_shape,
        )

    def validate(
        self,
        prepared: PreparedJWrite,
        *,
        residual: torch.Tensor,
        source: Mapping[str, Any],
    ) -> torch.Tensor:
        if not isinstance(prepared, PreparedJWrite):
            raise JacobianWorkspaceError("prepared must be a PreparedJWrite")
        metadata = _validate_prepared_metadata(
            prepared.metadata, prepared.candidate, prepared.delta
        )
        current_source = self.reader._source(source, residual)
        if metadata["source"] != current_source:
            raise JacobianWorkspaceError("prepared write source is stale")
        if metadata["lens"] != {
            "fingerprint": self.reader._manifest["fingerprint"],
            "application_lane": self.reader.application_lane,
        }:
            raise JacobianWorkspaceError("prepared write lens or lane mismatch")
        if metadata.get("execution_binding") != self.reader.execution_binding:
            raise JacobianWorkspaceError("prepared write execution binding mismatch")
        if prepared.candidate.shape != residual.shape or prepared.candidate.dtype != residual.dtype:
            raise JacobianWorkspaceError("prepared candidate geometry/dtype mismatch")
        return prepared.candidate.detach().clone()


__all__ = [
    "FrozenJacobianLens",
    "JACOBIAN_LENS_ARTIFACT_SCHEMA",
    "JACOBIAN_PREPARED_WRITE_ARTIFACT_SCHEMA",
    "JACOBIAN_PREPARED_WRITE_SCHEMA",
    "JacobianReader",
    "JacobianWriter",
    "PreparedJWrite",
]

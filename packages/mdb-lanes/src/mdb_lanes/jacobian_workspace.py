"""Bounded Jacobian-workspace observation and residual-edit algebra.

This module reads an already fitted, frozen dense transport.  It neither fits a
lens nor owns a model: callers supply tensors and, for exact readout, the native
final-normalization and output-head operations.  Returned metadata is strict
JSON and deliberately describes readouts and coordinates rather than sparse
codes or semantic capability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .tensor_hash import tensor_bytes_sha256

JACOBIAN_LENS_SCHEMA = "saturn-jacobian-lens-v1"
JACOBIAN_OBSERVATION_SCHEMA = "saturn-j-workspace-observation-v1"
JACOBIAN_DICTIONARY_ROWS_SCHEMA = "saturn-j-dictionary-rows-v1"
JACOBIAN_EDIT_SCHEMA = "saturn-j-workspace-intervention-v1"

_HEX = frozenset("0123456789abcdef")
_REQUIRED_MODEL_DIGESTS = ("weight_sha256", "config_sha256", "tokenizer_sha256")


class JacobianWorkspaceError(ValueError):
    """Raised when a J-space payload or operation violates its declared contract."""


@dataclass(frozen=True)
class JObservation:
    """Tensor-valued observation with a separately serializable manifest."""

    transported: torch.Tensor
    raw_scores: torch.Tensor
    normalized_scores: torch.Tensor
    selected_token_ids: torch.Tensor
    selected_raw_scores: torch.Tensor
    selected_normalized_scores: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DictionaryRows:
    """Selected ``W_U @ J`` rows and their declared scale convention."""

    rows: torch.Tensor
    metadata: dict[str, Any]


@dataclass(frozen=True)
class JEdit:
    """A residual candidate, its actual delta, and JSON-safe diagnostics."""

    candidate: torch.Tensor
    delta: torch.Tensor
    metadata: dict[str, Any]


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise JacobianWorkspaceError("metadata must be strict JSON without tensors") from exc


def metadata_sha256(value: Mapping[str, Any]) -> str:
    """Return the stable canonical-JSON identity of a metadata payload."""

    if not isinstance(value, Mapping):
        raise JacobianWorkspaceError("metadata must be a mapping")
    return hashlib.sha256(_canonical_json(dict(value))).hexdigest()


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise JacobianWorkspaceError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JacobianWorkspaceError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JacobianWorkspaceError(f"{label} must be a non-empty string")
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise JacobianWorkspaceError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _dtype_name(value: Any) -> str:
    text = str(value)
    return text if text.startswith("torch.") else "torch." + text


def _shape(value: Sequence[Any], label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise JacobianWorkspaceError(f"{label} must be a shape sequence")
    shape = tuple(_plain_int(item, label, minimum=1) for item in value)
    if not shape:
        raise JacobianWorkspaceError(f"{label} cannot be empty")
    return shape


def validate_lens_metadata(
    metadata: Mapping[str, Any],
    *,
    matrix_shape: Sequence[int],
    matrix_dtype: Any,
    application_lane: str | None = None,
    expected_model_identity: Mapping[str, Any] | None = None,
    expected_coordinate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a frozen dense lens manifest and optional runtime compatibility.

    The function checks metadata only.  Payload bytes remain caller-owned and
    are bound by the transport SHA-256, shape, and dtype in the manifest.
    """

    payload = _mapping(metadata, "lens metadata")
    # Round-trip first: this rejects tensors and host-specific metadata objects.
    payload = json.loads(_canonical_json(payload))
    if payload.get("schema") != JACOBIAN_LENS_SCHEMA:
        raise JacobianWorkspaceError(
            f"unsupported lens schema {payload.get('schema')!r}"
        )

    model = _mapping(payload.get("model_identity"), "model_identity")
    _text(model.get("model_id"), "model_identity.model_id")
    for key in _REQUIRED_MODEL_DIGESTS:
        _digest(model.get(key), f"model_identity.{key}")

    compatibility = _mapping(
        payload.get("execution_compatibility"), "execution_compatibility"
    )
    _text(compatibility.get("fitting_lane"), "execution_compatibility.fitting_lane")
    lanes = compatibility.get("permitted_application_lanes")
    if (
        not isinstance(lanes, list)
        or not lanes
        or any(not isinstance(lane, str) or not lane for lane in lanes)
    ):
        raise JacobianWorkspaceError(
            "execution_compatibility.permitted_application_lanes must be non-empty strings"
        )
    if len(set(lanes)) != len(lanes):
        raise JacobianWorkspaceError("permitted application lanes contain duplicates")
    if application_lane is not None and application_lane not in lanes:
        raise JacobianWorkspaceError(
            f"application lane {application_lane!r} is not permitted by this lens"
        )

    coordinate = _mapping(payload.get("coordinate_map"), "coordinate_map")
    width = _plain_int(coordinate.get("residual_width"), "coordinate_map.residual_width", minimum=1)
    source_layer = _plain_int(coordinate.get("source_layer"), "coordinate_map.source_layer")
    boundary = _plain_int(
        coordinate.get("completed_layer_boundary"),
        "coordinate_map.completed_layer_boundary",
        minimum=1,
    )
    if boundary != source_layer + 1:
        raise JacobianWorkspaceError(
            "completed_layer_boundary must equal source_layer + 1 for a block-output lens"
        )
    _text(coordinate.get("hook_convention"), "coordinate_map.hook_convention")
    _text(
        coordinate.get("source_position_semantics"),
        "coordinate_map.source_position_semantics",
    )

    transport = _mapping(payload.get("transport"), "transport")
    if transport.get("frozen") is not True:
        raise JacobianWorkspaceError("transport must declare frozen=true")
    declared_shape = _shape(transport.get("shape"), "transport.shape")
    actual_shape = _shape(matrix_shape, "matrix_shape")
    if declared_shape != actual_shape:
        raise JacobianWorkspaceError(
            f"transport shape mismatch: manifest {declared_shape}, payload {actual_shape}"
        )
    if len(actual_shape) != 2 or actual_shape != (width, width):
        raise JacobianWorkspaceError("transport must be square at the declared residual width")
    declared_dtype = _dtype_name(transport.get("dtype"))
    actual_dtype = _dtype_name(matrix_dtype)
    if declared_dtype != actual_dtype:
        raise JacobianWorkspaceError(
            f"transport dtype mismatch: manifest {declared_dtype}, payload {actual_dtype}"
        )
    _digest(transport.get("sha256"), "transport.sha256")

    decoder = _mapping(payload.get("decoder"), "decoder")
    if _plain_int(decoder.get("residual_width"), "decoder.residual_width", minimum=1) != width:
        raise JacobianWorkspaceError("decoder and coordinate residual widths differ")
    _plain_int(decoder.get("vocab_size"), "decoder.vocab_size", minimum=1)
    _text(decoder.get("final_norm"), "decoder.final_norm")
    _text(decoder.get("output_head"), "decoder.output_head")

    estimator = _mapping(payload.get("estimator"), "estimator")
    if estimator.get("name") != "target-sum/source-mean":
        raise JacobianWorkspaceError(
            "estimator.name must be exactly 'target-sum/source-mean'"
        )
    dictionary = _mapping(payload.get("dictionary"), "dictionary")
    if dictionary.get("atom_derivation") != "decoder-weight-times-transport":
        raise JacobianWorkspaceError(
            "dictionary.atom_derivation must declare decoder-weight-times-transport"
        )
    if dictionary.get("norm_scale_convention") not in {
        "none",
        "unit-l2-after-transport",
    }:
        raise JacobianWorkspaceError("unsupported dictionary norm scale convention")
    if dictionary.get("sparse_interpretation", "unperformed") != "unperformed":
        raise JacobianWorkspaceError(
            "this bounded observer cannot claim a sparse interpretation"
        )

    if expected_model_identity is not None:
        for key, expected in expected_model_identity.items():
            if model.get(key) != expected:
                raise JacobianWorkspaceError(f"model identity mismatch at {key!r}")
    if expected_coordinate is not None:
        for key, expected in expected_coordinate.items():
            if coordinate.get(key) != expected:
                raise JacobianWorkspaceError(f"coordinate mismatch at {key!r}")

    declared_fingerprint = payload.get("fingerprint")
    if declared_fingerprint is not None:
        body = {key: value for key, value in payload.items() if key != "fingerprint"}
        if _digest(declared_fingerprint, "fingerprint") != metadata_sha256(body):
            raise JacobianWorkspaceError("lens metadata fingerprint mismatch")
    return payload


def tensor_payload_identity(value: torch.Tensor) -> dict[str, Any]:
    """Describe exact tensor bytes with shape/dtype kept explicit."""

    _require_tensor(value, "value")
    descriptor = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "sha256": tensor_bytes_sha256(value),
    }
    descriptor["identity_sha256"] = metadata_sha256(descriptor)
    return descriptor


def _require_tensor(value: Any, label: str, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise JacobianWorkspaceError(f"{label} must be a torch.Tensor")
    if ndim is not None and value.ndim != ndim:
        raise JacobianWorkspaceError(f"{label} must have rank {ndim}, got {value.ndim}")
    if not value.is_floating_point():
        raise JacobianWorkspaceError(f"{label} must have floating dtype")
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise JacobianWorkspaceError(f"{label} contains non-finite values")
    return value


def _validate_observer_tensors(
    residual: torch.Tensor,
    transport: torch.Tensor,
    decoder_weight: torch.Tensor,
    *, verified_static: bool = False,
) -> int:
    _require_tensor(residual, "residual")
    if not verified_static:
        _require_tensor(transport, "transport", ndim=2)
        _require_tensor(decoder_weight, "decoder_weight", ndim=2)
    width = int(residual.shape[-1]) if residual.ndim else 0
    if residual.ndim < 1 or transport.shape != (width, width):
        raise JacobianWorkspaceError(
            "transport must be [residual_width, residual_width]"
        )
    if decoder_weight.shape[1] != width:
        raise JacobianWorkspaceError("decoder width does not match residual width")
    if transport.device != residual.device or decoder_weight.device != residual.device:
        raise JacobianWorkspaceError("residual, transport, and decoder must share a device")
    if transport.dtype != residual.dtype or decoder_weight.dtype != residual.dtype:
        raise JacobianWorkspaceError("residual, transport, and decoder must share a dtype")
    return width


def _token_ids(token_subset: Sequence[int] | None, vocab_size: int) -> list[int]:
    if token_subset is None:
        return list(range(vocab_size))
    if isinstance(token_subset, (str, bytes)) or not isinstance(token_subset, Sequence):
        raise JacobianWorkspaceError("token_subset must be a sequence of token IDs")
    result: list[int] = []
    for token_id in token_subset:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise JacobianWorkspaceError("token_subset must contain plain integers")
        if token_id < 0 or token_id >= vocab_size:
            raise JacobianWorkspaceError(f"token ID {token_id} is outside the decoder vocabulary")
        result.append(int(token_id))
    if not result or len(set(result)) != len(result):
        raise JacobianWorkspaceError("token_subset must be non-empty and contain no duplicates")
    return result


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    _require_tensor(weight, "rms_weight", ndim=1)
    if weight.shape != (value.shape[-1],) or weight.device != value.device:
        raise JacobianWorkspaceError("rms_weight geometry/device does not match transported state")
    if weight.dtype != value.dtype:
        raise JacobianWorkspaceError("rms_weight dtype does not match transported state")
    # Match the common native RMSNorm equation: accumulate in fp32, then cast
    # before applying the learned scale.
    variance = value.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normalized = (value.to(torch.float32) * torch.rsqrt(variance + eps)).to(value.dtype)
    return normalized * weight


def observe_j(
    residual: torch.Tensor,
    transport: torch.Tensor,
    decoder_weight: torch.Tensor,
    rms_weight: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    token_subset: Sequence[int] | None = None,
    top_k: int = 10,
    final_norm: Callable[[torch.Tensor], torch.Tensor] | None = None,
    output_head: Callable[[torch.Tensor], torch.Tensor] | None = None,
    _verified_identities: Mapping[str, Any] | None = None,
) -> JObservation:
    """Apply a frozen transport and publish raw and native-normalized readouts.

    Pass ``final_norm`` and ``output_head`` for the exact native decoder path.
    For dependency-light fixtures, ``rms_weight`` selects the declared RMSNorm
    equation and the supplied decoder weight is the head.  The function never
    calls ``train()``, ``eval()``, or changes gradient flags on either callable.
    """

    # Private fast path: only the managed frozen reader supplies these identities.
    # Public mutable callers retain full finite-value and payload verification.
    _validate_observer_tensors(residual, transport, decoder_weight,
                              verified_static=_verified_identities is not None)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise JacobianWorkspaceError("top_k must be a positive integer")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise JacobianWorkspaceError("eps must be a finite positive number")
    if (final_norm is None) != (output_head is None):
        raise JacobianWorkspaceError("final_norm and output_head must be supplied together")
    if final_norm is None and rms_weight is None:
        raise JacobianWorkspaceError("supply native final_norm/output_head or rms_weight")

    ids = None if token_subset is None else _token_ids(token_subset, int(decoder_weight.shape[0]))
    with torch.inference_mode():
        transported = torch.matmul(residual, transport.transpose(0, 1))
        raw_scores = torch.matmul(transported, decoder_weight.transpose(0, 1))
        if final_norm is not None:
            normalized_state = final_norm(transported)
            _require_tensor(normalized_state, "native final norm output")
            if normalized_state.shape != transported.shape:
                raise JacobianWorkspaceError("native final norm changed residual geometry")
            normalized_scores = output_head(normalized_state)  # type: ignore[misc]
            _require_tensor(normalized_scores, "native output head result")
            expected = (*transported.shape[:-1], decoder_weight.shape[0])
            if normalized_scores.shape != expected:
                raise JacobianWorkspaceError("native output head returned incompatible logits")
            decoder_policy = "caller-native-final-norm-and-output-head"
        else:
            normalized_state = _rms_norm(transported, rms_weight, float(eps))  # type: ignore[arg-type]
            normalized_scores = torch.matmul(
                normalized_state, decoder_weight.transpose(0, 1)
            )
            decoder_policy = "declared-rmsnorm-weight-and-decoder-matmul"
        index = None if ids is None else torch.tensor(ids, dtype=torch.long, device=residual.device)
        subset_normalized = normalized_scores if index is None else normalized_scores.index_select(-1, index)
        subset_raw = raw_scores if index is None else raw_scores.index_select(-1, index)
        candidate_count = int(decoder_weight.shape[0]) if ids is None else len(ids)
        count = min(top_k, candidate_count)
        selected_values, selected_offsets = torch.topk(
            subset_normalized, k=count, dim=-1, largest=True, sorted=True
        )
        selected_ids = selected_offsets if index is None else index[selected_offsets]
        selected_raw = subset_raw.gather(-1, selected_offsets)

    metadata = {
        "schema": JACOBIAN_OBSERVATION_SCHEMA,
        "activation": tensor_payload_identity(residual),
        "transport": tensor_payload_identity(transport) if _verified_identities is None else dict(_verified_identities["transport"]),
        "decoder_weight": tensor_payload_identity(decoder_weight) if _verified_identities is None else dict(_verified_identities["decoder_weight"]),
        "transport_policy": "left-apply-J; transported = residual @ J.T",
        "raw_score_policy": "decoder_weight @ transported; no final normalization",
        "normalized_score_policy": decoder_policy,
        "readout_coverage": "full-vocabulary" if token_subset is None else "declared-subset",
        "ranked_token_count": count,
        "candidate_token_count": candidate_count,
        "vocab_size": int(decoder_weight.shape[0]),
        "sparse_interpretation": "unperformed",
    }
    metadata["fingerprint"] = metadata_sha256(metadata)
    return JObservation(
        transported=transported,
        raw_scores=raw_scores,
        normalized_scores=normalized_scores,
        selected_token_ids=selected_ids,
        selected_raw_scores=selected_raw,
        selected_normalized_scores=selected_values,
        metadata=metadata,
    )


def derive_dictionary_rows(
    transport: torch.Tensor,
    decoder_weight: torch.Tensor,
    token_ids: Sequence[int],
    *,
    normalize: bool = True,
) -> DictionaryRows:
    """Derive selected ``W_U @ J`` rows without materializing the full dictionary."""

    _require_tensor(transport, "transport", ndim=2)
    _require_tensor(decoder_weight, "decoder_weight", ndim=2)
    if transport.shape[0] != transport.shape[1] or decoder_weight.shape[1] != transport.shape[0]:
        raise JacobianWorkspaceError("decoder and square transport geometry are incompatible")
    if transport.device != decoder_weight.device or transport.dtype != decoder_weight.dtype:
        raise JacobianWorkspaceError("decoder and transport must share device and dtype")
    ids = _token_ids(token_ids, int(decoder_weight.shape[0]))
    index = torch.tensor(ids, dtype=torch.long, device=transport.device)
    with torch.inference_mode():
        raw_rows = decoder_weight.index_select(0, index) @ transport
        raw_norms = torch.linalg.vector_norm(raw_rows, dim=-1)
        if normalize:
            if bool((raw_norms == 0).any().item()):
                raise JacobianWorkspaceError("cannot normalize a zero dictionary row")
            rows = raw_rows / raw_norms.unsqueeze(-1)
            convention = "unit-l2-after-transport"
        else:
            rows = raw_rows
            convention = "none"
    metadata = {
        "schema": JACOBIAN_DICTIONARY_ROWS_SCHEMA,
        "token_ids": ids,
        "atom_derivation": "decoder-weight-times-transport",
        "norm_scale_convention": convention,
        "pre_normalization_l2": [float(value) for value in raw_norms.cpu().tolist()],
        "transport": tensor_payload_identity(transport),
        "decoder_weight": tensor_payload_identity(decoder_weight),
        "rows": tensor_payload_identity(rows),
        "sparse_interpretation": "unperformed",
    }
    metadata["fingerprint"] = metadata_sha256(metadata)
    return DictionaryRows(rows=rows, metadata=metadata)


def normalized_atom_steering(
    residual: torch.Tensor,
    atom: torch.Tensor,
    *,
    dose: float,
) -> JEdit:
    """Add ``dose`` times a unit-L2 atom, preserving exact bytes at zero dose."""

    _require_tensor(residual, "residual", ndim=1)
    _require_tensor(atom, "atom", ndim=1)
    if atom.shape != residual.shape or atom.device != residual.device or atom.dtype != residual.dtype:
        raise JacobianWorkspaceError("atom geometry/device/dtype must match residual")
    dose_value = _finite_dose(dose)
    with torch.inference_mode():
        norm = torch.linalg.vector_norm(atom)
        if float(norm.item()) == 0.0:
            raise JacobianWorkspaceError("steering atom cannot be zero")
        if dose_value == 0.0:
            candidate = residual.clone()
            delta = torch.zeros_like(residual)
            noop_exact = True
        else:
            delta = atom / norm * dose_value
            candidate = residual + delta
            noop_exact = False
    return _edit_result(
        candidate,
        delta,
        operation="normalized-atom-steering",
        dose=dose_value,
        diagnostics={"atom_l2_before_normalization": float(norm.item())},
        noop_exact=noop_exact,
    )


def _finite_dose(dose: Any) -> float:
    if isinstance(dose, bool) or not isinstance(dose, (int, float)) or not math.isfinite(dose):
        raise JacobianWorkspaceError("dose must be a finite number")
    return float(dose)


def span_coordinate_edit(
    residual: torch.Tensor,
    atoms: torch.Tensor,
    target_coordinates: torch.Tensor,
    *,
    dose: float = 1.0,
) -> JEdit:
    """Move selected pseudoinverse span coordinates toward explicit targets.

    ``atoms`` has shape ``[coordinates, residual_width]``.  Coordinates are
    signed least-squares values; this is a span operation, not a sparse code.
    """

    _require_tensor(residual, "residual", ndim=1)
    _require_tensor(atoms, "atoms", ndim=2)
    _require_tensor(target_coordinates, "target_coordinates", ndim=1)
    if atoms.shape[1] != residual.shape[0] or target_coordinates.shape[0] != atoms.shape[0]:
        raise JacobianWorkspaceError("span coordinate geometry is incompatible")
    if (
        atoms.device != residual.device
        or target_coordinates.device != residual.device
        or atoms.dtype != residual.dtype
        or target_coordinates.dtype != residual.dtype
    ):
        raise JacobianWorkspaceError("span tensors must share device and dtype")
    dose_value = _finite_dose(dose)
    vectors = atoms.transpose(0, 1)
    with torch.inference_mode():
        source = torch.linalg.lstsq(vectors, residual.unsqueeze(-1)).solution.squeeze(-1)
        rank = int(torch.linalg.matrix_rank(vectors).item())
        singular_values = torch.linalg.svdvals(vectors)
        if singular_values.numel() and float(singular_values[-1].item()) > 0.0:
            condition_value = float(
                (singular_values[0] / singular_values[-1]).item()
            )
            condition: float | None = (
                condition_value if math.isfinite(condition_value) else None
            )
        else:
            condition = None
        exact_target = torch.equal(target_coordinates, source)
        if dose_value == 0.0 or exact_target:
            candidate = residual.clone()
            delta = torch.zeros_like(residual)
            noop_exact = True
        else:
            delta = vectors @ ((target_coordinates - source) * dose_value)
            candidate = residual + delta
            noop_exact = False
    return _edit_result(
        candidate,
        delta,
        operation="span-coordinate-edit",
        dose=dose_value,
        diagnostics={
            "source_coordinates": [float(value) for value in source.cpu().tolist()],
            "target_coordinates": [
                float(value) for value in target_coordinates.detach().cpu().tolist()
            ],
            "rank": rank,
            "coordinate_count": int(atoms.shape[0]),
            "condition_number": condition,
            "coordinate_semantics": "signed-pseudoinverse-span-coordinates; not sparse codes",
        },
        noop_exact=noop_exact,
    )


def _edit_result(
    candidate: torch.Tensor,
    delta: torch.Tensor,
    *,
    operation: str,
    dose: float,
    diagnostics: Mapping[str, Any],
    noop_exact: bool,
) -> JEdit:
    metadata = {
        "schema": JACOBIAN_EDIT_SCHEMA,
        "operation": operation,
        "dose": dose,
        "noop_exact": noop_exact,
        "actual_delta_norm": float(torch.linalg.vector_norm(delta).item()),
        "candidate": tensor_payload_identity(candidate),
        "delta": tensor_payload_identity(delta),
        "diagnostics": dict(diagnostics),
        "sparse_interpretation": "unperformed",
    }
    metadata["fingerprint"] = metadata_sha256(metadata)
    return JEdit(candidate=candidate, delta=delta, metadata=metadata)


__all__ = [
    "DictionaryRows",
    "JACOBIAN_DICTIONARY_ROWS_SCHEMA",
    "JACOBIAN_EDIT_SCHEMA",
    "JACOBIAN_LENS_SCHEMA",
    "JACOBIAN_OBSERVATION_SCHEMA",
    "JEdit",
    "JObservation",
    "JacobianWorkspaceError",
    "derive_dictionary_rows",
    "metadata_sha256",
    "normalized_atom_steering",
    "observe_j",
    "span_coordinate_edit",
    "tensor_payload_identity",
    "validate_lens_metadata",
]

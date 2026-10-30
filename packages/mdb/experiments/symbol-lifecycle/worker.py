"""Small native symbol-lifecycle assay for the public MDB surface.

The experiment-local decoder adapter admits one typed, registered K/V row
replacement at the committed pre-embedding boundary.  Public ``Session``
owns the native suffix consumer and exact restore; public ``Project`` retains
sealed cuts, receipts, and the read-only lifecycle sidecar.  Token labels are
only descriptive nominations, and the one-specimen causal trends remain
unqualified.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import gzip
import json
import math
import tarfile
import traceback
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

REPORT_MARKER = "SYMBOL_LIFECYCLE_REPORT_V1 "
ARCHIVE_MARKER = "SYMBOL_LIFECYCLE_PROJECT_ARCHIVE_V1 "
PROMPT = "France Paris The capital of France is"
QUERY = " Paris or Rome?"
CORRECT_DONOR_PROMPT = "Germany Berlin The capital of Germany is"
WRONG_DONOR_PROMPT = "Italy Rome The capital of Italy is"
MAX_SEQUENCE_LENGTH = 32
MAX_NATIVE_OPERATIONS = 256


def _digest(value: Any) -> str:
    body = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(body.encode("utf-8")).hexdigest()


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert receipts to bounded, finite JSON; never stringify unknown state."""

    if depth > 16:
        raise TypeError("report metadata exceeds nesting limit")
    # Torch exposes its version as a ``str`` subclass; retain such scalar
    # metadata without falling back to an opaque repr.
    if value is None or isinstance(value, str) or type(value) in (int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("report contains a non-finite number")
        return value
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict(), depth=depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    if isinstance(value, set):
        return [_jsonable(item, depth=depth + 1) for item in sorted(value, key=str)]
    # A tensor in a report is a worker bug: callers must retain a typed digest,
    # never a repr or an unbounded tolist payload.
    if any(hasattr(value, attribute) for attribute in ("detach", "numpy", "shape")):
        raise TypeError("tensor-like payload must be reduced to a typed descriptor")
    raise TypeError(f"report contains unsupported {type(value).__name__}")


def _write_report(report: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload = gzip.compress(json.dumps(_jsonable(report), sort_keys=True).encode("utf-8"))
    print(REPORT_MARKER + base64.b64encode(payload).decode("ascii"), flush=True)
    archive_info = report.get("native_project_archive")
    if isinstance(archive_info, Mapping) and archive_info.get("path"):
        archive_path = Path(str(archive_info["path"]))
        if archive_path.is_file():
            archive_bytes = archive_path.read_bytes()
            if sha256(archive_bytes).hexdigest() != archive_info.get("sha256"):
                raise RuntimeError("native project archive changed before log custody emission")
            print(ARCHIVE_MARKER + base64.b64encode(archive_bytes).decode("ascii"), flush=True)


def _row_text(row: Mapping[str, Any], tokenizer: Any | None = None) -> str:
    text = row.get("token_text") or row.get("text")
    if text is not None:
        return str(text)
    token_id = row.get("token_id")
    if tokenizer is not None and isinstance(token_id, int):
        try:
            return str(tokenizer.decode([token_id]))
        except Exception:
            pass
    return f"token:{token_id}"


def _select_source_rows(
    rows: list[Mapping[str, Any]], tokenizer: Any | None = None
) -> tuple[list[int], str]:
    """Choose two lexical anchors, retaining a truthful fallback if tokenization differs."""

    anchors: list[int] = []
    for index, row in enumerate(rows):
        text = _row_text(row, tokenizer).lower().replace(" ", "")
        if "france" in text or "paris" in text:
            anchors.append(index)
    if len(anchors) >= 2:
        # Prefer a contiguous pair so a native half-open row selector names the
        # coalition without silently including a large unrelated span.
        for left, right in zip(anchors, anchors[1:]):
            if right == left + 1:
                return [left, right], "lexical-anchors-contiguous"
        return anchors[:2], "lexical-anchors-span"
    if len(rows) >= 2:
        return [len(rows) - 2, len(rows) - 1], "fallback-final-two-rows"
    raise RuntimeError("instrument-broken: committed prompt has fewer than two readable K/V rows")


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    result = [int(value) for value in values]
    if not result:
        raise RuntimeError(f"instrument-broken: tokenizer produced no tokens for {text!r}")
    return result


def _answer_token_ids(tokenizer: Any) -> dict[str, int]:
    """Require one-token answer labels before claiming a graded margin."""

    result: dict[str, int] = {}
    for label in ("Paris", "Rome", "Berlin"):
        values = _token_ids(tokenizer, f" {label}")
        if len(values) != 1:
            raise RuntimeError(
                f"instrument-broken: answer label {label!r} is not one token; "
                "the graded margin would be underspecified"
            )
        result[label] = values[0]
    if len(set(result.values())) != len(result):
        raise RuntimeError("instrument-broken: answer labels share one token")
    return result


def _answer_observation(logits: Any, answer_tokens: Mapping[str, int]) -> dict[str, Any]:
    """Record the native one-token readout without treating labels as semantics."""

    values = logits.detach().float().reshape(-1)
    labels = {label: float(values[int(token)].item()) for label, token in answer_tokens.items()}
    if not all(math.isfinite(value) for value in labels.values()):
        raise RuntimeError("instrument-broken: native answer logits are non-finite")
    return {
        "status": "completed",
        "answer_token_ids": dict(answer_tokens),
        "answer_logits": labels,
        "margin_paris_minus_rome": labels["Paris"] - labels["Rome"],
        "margin_berlin_minus_paris": labels["Berlin"] - labels["Paris"],
        "margin_berlin_minus_rome": labels["Berlin"] - labels["Rome"],
        "argmax_token_id": int(values.argmax().item()),
        "argmax_logit": float(values.max().item()),
    }


def _native_token_probe(adapter: Any, cut: Any, *, answer_tokens: Mapping[str, int]) -> dict[str, Any]:
    """Run one native token and read its retained logits from the same suffix."""

    from saturn_pub.core import Session

    session = Session.from_cut(adapter, cut)
    initial = session.capture(retain=False)
    receipt = session.step(granularity="token", max_steps=96)
    result = session.capture(retain=False)
    logits = session.read("logits")
    if logits is None:
        raise RuntimeError("instrument-broken: token suffix did not retain native logits")
    receipt_body = receipt.to_dict()
    micro_receipts = receipt_body.get("micro_receipts") or ()
    native_transitions = sum(int(item.get("steps", 1)) for item in micro_receipts)
    if native_transitions < 1:
        raise RuntimeError("instrument-broken: token receipt has no native transitions")
    return {
        "initial_cut": initial,
        "result_cut": result,
        "receipt_object": receipt,
        "receipt": receipt_body,
        "tokens": session.read("tokens").detach().cpu().tolist(),
        "margin": _answer_observation(logits, answer_tokens),
        "native_transition_count": native_transitions,
    }


def _spectral_native_record(*, symbol: str, scene_id: str, instance_id: str,
                            parent_cut: Any, binding_fingerprint: str,
                            operator_reference: str, consumer: str) -> tuple[Any, Any, dict[str, Any]]:
    """Allocate and register one typed SAS address for this exact native binding."""

    from mdb.spectral_address import (
        AddressAllocator,
        SpectralAddressSpaceSpec,
        SpectralBinding,
        SpectralFrame,
        SpectralInstanceKey,
        SpectralLogicalAddress,
        SpectralQuantizationProfile,
        SpectralRoleTransformSpec,
    )

    dimensions = 8
    profile = SpectralQuantizationProfile(
        "native-symbol-lifecycle-phase-v1", phase_bits=16, frequency_layout="flat-v1"
    )
    space = SpectralAddressSpaceSpec(
        address_space_id="saturn.native.qwen.symbols.v1",
        dimensions=dimensions,
        namespaces=("saturn.native.qwen",),
        supported_operations=("resolve", "read", "replace"),
        quantization_profile=profile,
        read_kernel_ids=("native-exact-identity-v1",),
        frame_ids=("native-decoder-parent",),
        gauge_policy="consistent-global-phase",
        boundary_conditions="toroidal-core",
    )
    allocator = AddressAllocator(f"{space.address_space_id}:{binding_fingerprint}", dimensions)
    key = allocator.allocate(symbol, "saturn.native.qwen", "cache-coalition")
    frame = SpectralFrame(parent_cut.fingerprint)
    instance = SpectralInstanceKey.deterministic(
        scene_id, instance_id, dimensions, edge_id="source"
    )
    role = SpectralRoleTransformSpec(
        "source", (0.0,) * dimensions, 1, 0,
        "source", "source", inverse_role_id="source", exact_unbinding=True,
        spf_conformance="supplied-native-binding",
    )
    logical = SpectralLogicalAddress(key, instance, role, frame)
    registered = SpectralBinding(
        edge_id=f"edge:{logical.fingerprint}",
        operator_id="native.decoder.registered-row-replacement",
        arguments=((role.role_id, logical),),
        payload_reference=operator_reference,
        frame=frame,
    )
    record = {
        "schema": space.schema,
        "address_space": {
            "address_space_id": space.address_space_id,
            "dimensions": space.dimensions,
            "namespaces": list(space.namespaces),
            "supported_operations": list(space.supported_operations),
            "quantization_profile": {
                "profile_id": profile.profile_id,
                "phase_bits": profile.phase_bits,
                "frequency_layout": profile.frequency_layout,
                "profile_version": profile.profile_version,
            },
            "read_kernel_ids": list(space.read_kernel_ids),
            "frame_ids": list(space.frame_ids),
            "gauge_policy": space.gauge_policy,
            "boundary_conditions": space.boundary_conditions,
        },
        "canonical_key": key.to_dict(),
        "canonical_key_fingerprint": key.fingerprint,
        "instance": {
            "scene_id": instance.scene_id,
            "instance_id": instance.instance_id,
            "instance_phases": list(instance.instance_phases),
            "edge_id": instance.edge_id,
            "execution_scope": instance.execution_scope,
        },
        "role": {
            "role_id": role.role_id,
            "arity": role.arity,
            "argument_index": role.argument_index,
            "input_type": role.input_type,
            "output_type": role.output_type,
            "exact_unbinding": role.exact_unbinding,
            "spf_conformance": role.spf_conformance,
        },
        "frame": {"frame_id": frame.frame_id, "parent_frame_id": frame.parent_frame_id},
        "logical_fingerprint": logical.fingerprint,
        "typed_logical_address": {
            "schema": "saturn-spectral-address-abi-v1",
            "canonical_key": key.to_dict(),
            "instance_key": dataclasses.asdict(instance),
            "role_transform": dataclasses.asdict(role),
            "frame": dataclasses.asdict(frame),
            "fingerprint": logical.fingerprint,
        },
        "registered_binding": {
            "edge_id": registered.edge_id,
            "operator_id": registered.operator_id,
            "payload_reference": registered.payload_reference,
            "frame_id": registered.frame.frame_id,
        },
        "native_source_cut": f"cas://sha256/{parent_cut.fingerprint}",
        "native_operator_reference": operator_reference,
        "native_consumer": consumer,
        "mapping_status": "supplied-native-exact-address; semantic-resolution-unqualified",
        "registration_authority": "host-supplied-typed-registration",
    }
    return space, logical, record


def _vm_execution_bundle(*, run_id: str, parent_cut: str, candidate_cut: str,
                         result_cut: str, model_identity: Any, execution: Any,
                         context: str, operator_manifest: Any, consumer: str,
                         receipt: str) -> tuple[Any, dict[str, Any]]:
    from mdb.vm_cache import VMCacheExecutionBundle

    plane_descriptor_payloads = {
        "source": {"source_cut": parent_cut},
        "address": {"context": context, "source_cut": parent_cut},
        "operator": operator_manifest,
        "renderer": {"consumer": consumer, "result_cut": result_cut},
    }
    address_digest = _digest(plane_descriptor_payloads["address"])
    operator_digest = _digest(plane_descriptor_payloads["operator"])
    renderer_digest = _digest(plane_descriptor_payloads["renderer"])
    refs = {
        "source": f"cas://sha256/{_digest(plane_descriptor_payloads['source'])}",
        "address": f"cas://sha256/{address_digest}",
        "operator": f"cas://sha256/{operator_digest}",
        "renderer": f"cas://sha256/{renderer_digest}",
    }
    bundle = VMCacheExecutionBundle.create(
        run_id=run_id,
        parent_identity={"source_cut": parent_cut, "candidate_cut": candidate_cut},
        graph_fingerprint=parent_cut,
        model_identity={"id": model_identity},
        input_identity={"context": context, "source_cut": parent_cut},
        abi_identity={"execution": execution, "consumer": consumer},
        plane_references=refs,
        lineage={"source_cut": parent_cut, "candidate_cut": candidate_cut, "result_cut": result_cut},
        admitted_plan={"native_consumer": consumer, "receipt": receipt},
        metadata={
            "semantic_equivalence": "not_assessed",
            "authority": "native-receipt-and-pointer-only",
            "plane_descriptor_refs": refs,
        },
    )
    return bundle, plane_descriptor_payloads


def _write_vm_plane_descriptors(
    project_root: Path, bundles: list[tuple[Any, Mapping[str, Any]]]
) -> dict[str, Any]:
    """Retain the exact small descriptor bytes named by every VM plane ref."""

    destination = project_root / "vm-cache" / "descriptors"
    destination.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bundle, payloads in bundles:
        for plane, reference in bundle.plane_references.items():
            digest = str(reference).rsplit("/", 1)[-1]
            payload = payloads.get(plane)
            if payload is None or _digest(payload) != digest:
                raise RuntimeError(f"VM plane descriptor digest mismatch for {plane}")
            if digest in seen:
                continue
            seen.add(digest)
            encoded = json.dumps(
                _jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            if sha256(encoded).hexdigest() != digest:
                raise RuntimeError(f"VM plane descriptor bytes mismatch for {plane}")
            path = destination / f"{digest}.json"
            path.write_bytes(encoded)
            inventory.append({
                "plane": plane,
                "reference": reference,
                "path": str(path.relative_to(project_root)),
                "bytes": len(encoded),
                "sha256": digest,
            })
    return {"schema": "symbol-lifecycle-vm-plane-custody-v1", "files": inventory}


def _archive_project(project_root: Path, export_root: Path, destination: Path) -> dict[str, Any]:
    """Seal the native Project bytes for direct collection from the worker log."""

    with tarfile.open(destination, "w:gz") as archive:
        archive.add(project_root, arcname="project")
        archive.add(export_root, arcname="export")
    size = destination.stat().st_size
    if size > 64 * 1024 * 1024:
        raise RuntimeError(f"native project archive exceeds the 64 MiB log-artifact budget: {size}")
    return {"schema": "symbol-lifecycle-native-project-archive-v1", "path": str(destination),
            "bytes": size, "sha256": sha256(destination.read_bytes()).hexdigest(),
            "archive_marker": "SYMBOL_LIFECYCLE_PROJECT_ARCHIVE_V1"}


def _memory_adapter_class() -> type:
    """Build the explicit experiment-local cache-edit adapter.

    The base public decoder deliberately exposes K/V as read-only.  This
    narrow subclass changes that contract only at the committed pre-embedding
    boundary and admits one typed row replacement Act.  It never mutates the
    resident module or reconstructs history from arbitrary tensors.
    """

    from saturn_pub.adapters.decoder import DecoderAdapter
    from saturn_pub.contracts import SlotSpec, SurfaceManifest
    from saturn_pub.values import clone

    class DecoderMemoryAdapter(DecoderAdapter):
        def __init__(self, model: Any, **kwargs: Any) -> None:
            super().__init__(model, **kwargs)
            self.execution = {
                **self.execution,
                "adapter": "native-decoder-memory-v1",
                "memory_contract": {
                    "boundary": "committed-cache-before-embedding",
                    "operation": "registered-row-replacement",
                    "scope": "keys-and-values-only",
                    "historical_reconstruction": False,
                },
            }

        @staticmethod
        def _cache_editable(state: Mapping[str, Any]) -> bool:
            return (
                state.get("phase") == "embed"
                and state.get("layer") == 0
                and state.get("hidden") is None
                and state.get("logits") is None
                and state.get("sampled_token") is None
            )

        def surface(self, state: Mapping[str, Any]) -> SurfaceManifest:
            base = super().surface(state)
            editable = self._cache_editable(state)
            slots = tuple(
                SlotSpec(
                    slot.name,
                    role=slot.role,
                    writable=editable and slot.name in {"keys", "values"},
                    persistence=slot.persistence,
                    producer=slot.producer,
                    consumer=slot.consumer,
                    invalidates=slot.invalidates,
                    schema=slot.schema,
                )
                for slot in base.slots
            )
            return SurfaceManifest(
                slots,
                f"saturn-pub-decoder-{self.spec.model_type}-memory-state-v1",
                base.consumer,
                base.horizon,
            )

        def addresses(self, state: Mapping[str, Any]) -> tuple[str, ...]:
            # The base decoder keeps cache ports out of its read surface.  This
            # experiment explicitly admits typed cache inspection so the
            # binding and Act can carry the same public closure; it does not
            # make the cache writable outside ``_cache_editable``.
            return tuple(dict.fromkeys((*super().addresses(state), "keys", "values")))

        def write(
            self,
            state: Mapping[str, Any],
            writes: Mapping[str, Any],
            invalidates: tuple[str, ...],
        ) -> Mapping[str, Any]:
            import torch

            if not self._cache_editable(state):
                raise ValueError("cache replacement requires the committed pre-embedding boundary")
            if set(writes) != {"keys", "values"} or invalidates:
                raise ValueError("memory Act must replace both keys and values without invalidation")
            result = clone(state)
            for name in ("keys", "values"):
                current = state[name]
                replacement = writes[name]
                if not isinstance(replacement, (list, tuple)) or len(replacement) != self.layers:
                    raise ValueError(f"{name} replacement must contain every decoder layer")
                checked = []
                for index, (before, after) in enumerate(zip(current, replacement)):
                    if not isinstance(after, torch.Tensor):
                        raise ValueError(f"{name}[{index}] replacement is not a tensor")
                    if tuple(after.shape) != tuple(before.shape) or after.dtype != before.dtype:
                        raise ValueError(f"{name}[{index}] shape/dtype changed")
                    if after.device != before.device or not bool(torch.isfinite(after).all()):
                        raise ValueError(f"{name}[{index}] device or finite-value contract failed")
                    checked.append(after.detach().clone())
                result[name] = checked
            self.validate(result)
            self.surface(result).validate(result)
            return result

    return DecoderMemoryAdapter


class _RowReplacementOperation:
    """Pure typed operation that changes only selected K/V rows."""

    def __init__(self, layer: int, rows: tuple[int, ...], key_rows: tuple[Any, ...], value_rows: tuple[Any, ...]):
        self.layer = int(layer)
        self.rows = tuple(int(row) for row in rows)
        self.key_rows = tuple(row.detach().clone() for row in key_rows)
        self.value_rows = tuple(row.detach().clone() for row in value_rows)

    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        import torch

        keys = [value.detach().clone() for value in inputs["keys"]]
        values = [value.detach().clone() for value in inputs["values"]]
        if len(keys) != len(values) or not 0 <= self.layer < len(keys):
            raise ValueError("typed donor layer is outside the complete K/V closure")
        before_keys = [value.detach().clone() for value in keys]
        before_values = [value.detach().clone() for value in values]
        for row, key, value in zip(self.rows, self.key_rows, self.value_rows):
            if not 0 <= row < keys[self.layer].shape[-2]:
                raise ValueError("typed donor row is outside the committed cache")
            target_key = keys[self.layer][..., row : row + 1, :]
            target_value = values[self.layer][..., row : row + 1, :]
            if tuple(key.shape) != tuple(target_key.shape):
                raise ValueError("typed donor key row geometry changed")
            if tuple(value.shape) != tuple(target_value.shape):
                raise ValueError("typed donor value row geometry changed")
            if key.dtype != target_key.dtype or value.dtype != target_value.dtype:
                raise ValueError("typed donor row dtype changed")
            if key.device != target_key.device or value.device != target_value.device:
                raise ValueError("typed donor row device changed")
            if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
                raise ValueError("typed donor row contains a non-finite value")
            target_key.copy_(key)
            target_value.copy_(value)
        if not bool(torch.isfinite(keys[self.layer]).all()) or not bool(torch.isfinite(values[self.layer]).all()):
            raise ValueError("typed donor row contains a non-finite value")
        selected = set(self.rows)
        for index, (before, after) in enumerate(zip(before_keys, keys)):
            if index != self.layer and not torch.equal(before, after):
                raise ValueError("typed K/V operation changed an unselected layer")
            if index == self.layer:
                for row in range(after.shape[-2]):
                    if row not in selected and not torch.equal(before[..., row : row + 1, :], after[..., row : row + 1, :]):
                        raise ValueError("typed K/V operation changed an unselected key row")
        for index, (before, after) in enumerate(zip(before_values, values)):
            if index != self.layer and not torch.equal(before, after):
                raise ValueError("typed K/V operation changed an unselected value layer")
            if index == self.layer:
                for row in range(after.shape[-2]):
                    if row not in selected and not torch.equal(before[..., row : row + 1, :], after[..., row : row + 1, :]):
                        raise ValueError("typed K/V operation changed an unselected value row")
        return {"keys": keys, "values": values}


def _memory_act(session: Any, layer: int, rows: list[int], *, mode: str, scale: float = 1.0,
                donor: tuple[list[Any], list[Any]] | None = None, name: str) -> Any:
    """Create one registered, full-closure K/V row replacement Act."""

    import torch
    from saturn_pub.core import Act
    from saturn_pub.values import describe, digest

    keys = session.read("keys")
    values = session.read("values")
    rows_tuple = tuple(sorted(set(int(row) for row in rows)))
    if not rows_tuple:
        raise ValueError("memory Act needs at least one row")
    if type(layer) is not int or not 0 <= layer < len(keys):
        raise ValueError("memory Act layer is outside the cache")
    length = int(keys[layer].shape[-2])
    if any(row < 0 or row >= length for row in rows_tuple):
        raise ValueError("memory Act row is outside the committed cache")
    if mode not in {"copy", "zero", "scale"}:
        raise ValueError("memory Act mode must be copy, zero or scale")
    if not math.isfinite(float(scale)):
        raise ValueError("memory Act scale must be finite")

    if donor is None:
        donor_keys, donor_values = keys, values
    else:
        donor_keys, donor_values = donor
        if len(donor_keys) != len(keys) or len(donor_values) != len(values):
            raise ValueError("typed donor must contain every decoder layer")
        if tuple(donor_keys[layer].shape) != tuple(keys[layer].shape) or tuple(donor_values[layer].shape) != tuple(values[layer].shape):
            raise ValueError("typed donor K/V geometry changed")
        if donor_keys[layer].dtype != keys[layer].dtype or donor_values[layer].dtype != values[layer].dtype:
            raise ValueError("typed donor K/V dtype changed")
        if donor_keys[layer].device != keys[layer].device or donor_values[layer].device != values[layer].device:
            raise ValueError("typed donor K/V device changed")
        if not bool(torch.isfinite(donor_keys[layer]).all()) or not bool(torch.isfinite(donor_values[layer]).all()):
            raise ValueError("typed donor K/V contains a non-finite value")
    key_rows = []
    value_rows = []
    for row in rows_tuple:
        key_row = donor_keys[layer][..., row : row + 1, :].detach().clone()
        value_row = donor_values[layer][..., row : row + 1, :].detach().clone()
        if mode == "zero":
            key_row = torch.zeros_like(key_row)
            value_row = torch.zeros_like(value_row)
        elif mode == "scale":
            key_row = key_row * float(scale)
            value_row = value_row * float(scale)
        key_rows.append(key_row)
        value_rows.append(value_row)
    operation = _RowReplacementOperation(layer, rows_tuple, tuple(key_rows), tuple(value_rows))
    operation_key = _digest({
        "layer": layer,
        "rows": list(rows_tuple),
        "keys": [describe(row) for row in key_rows],
        "values": [describe(row) for row in value_rows],
    })
    operation_id = f"saturn.experimental.decoder-memory.row-replace.{operation_key}"
    Act.register_operation(operation_id, "1", operation, pure=True)
    return Act.registered(
        name,
        ("keys", "values"),
        ("keys", "values"),
        operation_id=operation_id,
        operation_version="1",
        implementation=operation,
        numerical_contract="same-program-exact-row-replacement",
        parameters={"layer": layer, "rows": list(rows_tuple), "head_scope": "all-kv-heads",
                    "mode": mode, "scale": scale, "operation_key": operation_key},
        preconditions={"keys": digest(describe(keys)), "values": digest(describe(values))},
        # At the committed pre-embedding boundary hidden/logits are absent
        # from the public read surface.  The adapter's authoritative-slot
        # check still preserves their ``None`` closure; tokens is the only
        # explicit readable cursor that this Act must witness.
        preserve=("tokens",),
    )


def _public_probe(adapter: Any, parent_cut: Any, candidate_cut: Any, *, name: str,
                  consumer: str, answer_tokens: Mapping[str, int],
                  baseline_probe: Mapping[str, Any]) -> dict[str, Any]:
    """Run one native public suffix from a candidate and reuse one shared baseline."""

    candidate_probe = _native_token_probe(adapter, candidate_cut, answer_tokens=answer_tokens)
    baseline_result = baseline_probe["result_cut"]
    candidate_result = candidate_probe["result_cut"]
    baseline_receipt = baseline_probe["receipt_object"]
    candidate_receipt = candidate_probe["receipt_object"]
    baseline_margin = baseline_probe["margin"]
    candidate_margin = candidate_probe["margin"]
    return {
        "name": name,
        "status": "completed",
        "consumer": consumer,
        "parent_cut": parent_cut.fingerprint,
        "candidate_cut": candidate_cut.fingerprint,
        "baseline_initial_cut": baseline_probe["initial_cut"].fingerprint,
        "candidate_initial_cut": candidate_probe["initial_cut"].fingerprint,
        "baseline_result_cut": baseline_result.fingerprint,
        "candidate_result_cut": candidate_result.fingerprint,
        "baseline_tokens": baseline_probe["tokens"],
        "candidate_tokens": candidate_probe["tokens"],
        "tokens_equal": baseline_probe["tokens"] == candidate_probe["tokens"],
        "baseline_receipts": [baseline_receipt.to_dict()],
        "candidate_receipts": [candidate_receipt.to_dict()],
        "receipt": candidate_receipt.to_dict(),
        "candidate_receipt_object": candidate_receipt,
        "baseline_receipt_object": baseline_receipt,
        "candidate_result": candidate_result,
        "baseline_result": baseline_result,
        "baseline_step_cuts": (baseline_result,),
        "candidate_step_cuts": (candidate_result,),
        "candidate_receipt_parent_cut": candidate_cut.fingerprint,
        "baseline_receipt_parent_cut": parent_cut.fingerprint,
        "baseline_margin": dict(baseline_margin),
        "candidate_margin": dict(candidate_margin),
        "baseline_margin_result": baseline_result,
        "candidate_margin_result": candidate_result,
        "baseline_margin_receipt": baseline_receipt,
        "candidate_margin_receipt": candidate_receipt,
        "baseline_margin_receipts": (baseline_receipt,),
        "candidate_margin_receipts": (candidate_receipt,),
        "baseline_margin_cuts": (baseline_result,),
        "candidate_margin_cuts": (candidate_result,),
        "native_transition_count": int(baseline_probe["native_transition_count"])
        + int(candidate_probe["native_transition_count"]),
        "candidate_native_transition_count": int(candidate_probe["native_transition_count"]),
        "shared_baseline_native_transition_count": int(baseline_probe["native_transition_count"]),
        "native_transition_breakdown": {
            "baseline": int(baseline_probe["native_transition_count"]),
            "candidate": int(candidate_probe["native_transition_count"]),
        },
        "exact_parent_input": baseline_probe["initial_cut"].fingerprint == parent_cut.fingerprint,
        "exact_candidate_input": candidate_probe["initial_cut"].fingerprint == candidate_cut.fingerprint,
    }


def _clock_witness(cut: Any, *, provenance: tuple[str, ...], execution: str,
                   consumer: str, reset: str) -> Any:
    from mdb.symbol_lifecycle import ClockWitness

    source_ref = cut.fingerprint
    return ClockWitness(
        clock_id=f"execution-point:{cut.fingerprint}",
        kind="execution-point",
        domain="saturn-pub.decoder",
        reading=dict(cut.execution_point or {}),
        provenance=provenance,
        validity_status="supported",
        validity_support=provenance,
        status="observed",
        metadata={"source_cut": source_ref, "model_execution": execution,
                  "consumer": consumer, "reset": reset},
    )


def _save_and_reload_lifecycle(project: Any, session_id: str, lifecycle: Any, *, port: str,
                               selector: list[int], cut_id: str, branch: str) -> dict[str, Any]:
    """Use only the public Project lifecycle seam; fail closed before launch if absent."""

    save = getattr(project, "save_symbol_lifecycle", None)
    list_rows = getattr(project, "symbol_lifecycles", None)
    show = getattr(project, "symbol_lifecycle", None)
    at = getattr(project, "symbols_at", None)
    if not all(callable(item) for item in (save, list_rows, show, at)):
        raise RuntimeError("public Project symbol lifecycle API is not staged")
    saved = save(session_id, lifecycle)
    reloaded = show(session_id, lifecycle.fingerprint)
    listed = list_rows(session_id, symbol=lifecycle.symbol, cut_id=cut_id, branch=branch)
    located = at(session_id, port, cut_id=cut_id, selector=selector, branch=branch)
    return {"saved": saved, "reloaded": reloaded, "listed": listed, "located": located}


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    """Run the public Session/Project lifecycle assay on one resident Qwen."""

    from saturn_pub.contracts import StateAddress
    from saturn_pub.core import Session
    from saturn_pub.symbols import SymbolBinding

    from mdb import Project
    from mdb.bench_loading import load_bench_model, resolve_bench_model
    from mdb.spectral_address import SpectralQuery
    from mdb.symbol_lifecycle import SymbolLifecycle, SymbolOccurrence, SymbolUse

    root = Path(args.root).expanduser().resolve()
    export_root = Path(args.export_root).expanduser().resolve() if args.export_root else root.parent / "export"
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else root.parent / "project"
    for destination in (root, export_root, project_root):
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite existing destination: {destination}")
    root.mkdir(parents=True, exist_ok=False)

    spec = resolve_bench_model(args.model)
    device = str(getattr(args, "device", "cuda"))
    if device != "cuda" and not bool(getattr(args, "tiny_fixture", False)):
        raise RuntimeError("native real-model assay requires CUDA; CPU is reserved for explicit tiny fixtures")
    loaded_spec, model, tokenizer = load_bench_model(args.model, device=device)
    loaded_identity = loaded_spec.to_dict()
    if {key: loaded_identity.get(key) for key in spec.to_dict()} != spec.to_dict():
        raise RuntimeError("model registry identity changed between resolution and load")
    if tokenizer is None:
        raise RuntimeError("instrument-broken: Qwen lifecycle requires a tokenizer")

    MemoryAdapter = _memory_adapter_class()
    adapter = MemoryAdapter(model, granularity="operation", device=device)
    token_ids = _token_ids(tokenizer, PROMPT)
    answer_tokens = _answer_token_ids(tokenizer)
    if len(token_ids) < 3 or len(token_ids) > MAX_SEQUENCE_LENGTH:
        raise RuntimeError(
            f"instrument-broken: prompt token count {len(token_ids)} is outside [3, {MAX_SEQUENCE_LENGTH}]"
        )
    session = adapter.session(token_ids)
    parent_cut = session.capture(retain=True)
    session_id = "public-symbol-lifecycle"
    frame = session.inspect()
    consumer = str(frame.surface["consumer"])
    source_layer = min(6, adapter.layers - 1)
    committed_rows = [{"token_id": token_id} for token_id in token_ids[:-1]]
    members, selection_mode = _select_source_rows(committed_rows, tokenizer)
    source_heads = list(range(adapter.num_kv_heads))
    wrong_row = next(
        (row for row in range(len(token_ids) - 1) if row not in members), None
    )
    if wrong_row is None:
        raise RuntimeError("instrument-broken: no committed non-source row for wrong-row control")
    late_layers = list(range(max(0, adapter.layers - 8), adapter.layers))
    if source_layer in late_layers:
        late_layers.remove(source_layer)
    binding_layers = [source_layer, *late_layers]

    def locations_for(layers: list[int]) -> tuple[Any, ...]:
        return tuple(
            StateAddress("keys", (layer, 0, head, row), "cache")
            for layer in layers for row in members for head in source_heads
        ) + tuple(
            StateAddress("values", (layer, 0, head, row), "cache")
            for layer in layers for row in members for head in source_heads
        )

    source_locations = locations_for([source_layer])
    late_locations = locations_for(late_layers)
    binding_locations = locations_for(binding_layers)
    context_digest = _digest({"token_ids": token_ids, "model": spec.to_dict()})
    context = f"prompt:{context_digest}"
    binding = SymbolBinding.bind(
        session,
        "cot.source.coalition",
        binding_locations,
        context=context,
        context_slots=("tokens", "keys", "values"),
        evidence=(f"cas://sha256/{parent_cut.fingerprint}",),
        claim="A supplied committed-cache K/V row coalition for one native decoder context.",
        status="supplied",
    )
    # ``SymbolBinding.bind`` qualifies bare addresses with the exact public
    # execution point and state schema.  Lifecycle records must use those
    # sealed instances, not fresh unqualified StateAddress values.
    def qualified_locations(layers: list[int]) -> tuple[Any, ...]:
        selected_layers = set(layers)
        return tuple(location for location in binding.locations if location.selector[0] in selected_layers)

    source_locations = qualified_locations([source_layer])
    late_locations = qualified_locations(late_layers)
    binding_locations = tuple(binding.locations)
    binding_fingerprint = binding.to_dict()["fingerprint"]
    source_rows = [
        {
            "row_index": row,
            "token_id": int(token_ids[row]),
            "token_text": _row_text(committed_rows[row], tokenizer),
            "layer": source_layer,
            "components": ["key", "value"],
            "heads": source_heads,
            # keys/values are a list of tensors shaped [batch, head, row, dim].
            # The leading list coordinate is the layer; the tensor batch axis
            # must remain explicit. Leave dim unindexed to name the entire
            # vector actually replaced by the row Act, rather than dim 0.
            "selector": [source_layer, 0, 0, row],
        }
        for row in members
    ]
    source_clock = _clock_witness(
        parent_cut,
        provenance=(f"cas://sha256/{parent_cut.fingerprint}",),
        execution=binding.execution,
        consumer=consumer,
        reset="public-session-parent-cut",
    )

    donor_specs = [
        ("zero_member_0", [members[0]], "zero", 1.0),
        ("zero_member_1", [members[-1]], "zero", 1.0),
        ("zero_coalition", members, "zero", 1.0),
        ("half_coalition", members, "scale", 0.5),
    ]
    donor_sessions: dict[str, Session] = {}
    donors: list[dict[str, Any]] = []
    all_cuts = [parent_cut]
    all_receipts = []

    # The matched donor supplies both the source-row intervention and the
    # wrong-target-row control.  The geometry check is deliberately strict: a
    # donor with a different committed length would make a row address
    # ambiguous rather than proving anything about the symbol.
    correct_token_ids = _token_ids(tokenizer, CORRECT_DONOR_PROMPT)
    if len(correct_token_ids) != len(token_ids):
        raise RuntimeError(
            "instrument-broken: source-swap prompts must have the same token geometry as the recipient"
        )
    correct_session = adapter.session(correct_token_ids)
    correct_cut = correct_session.capture(retain=True)
    donor_sessions.update({"source_correct": correct_session})
    donors.extend((
        {
            "name": "source_correct",
            "kind": "context-donor",
            "prompt": CORRECT_DONOR_PROMPT,
            "token_ids": correct_token_ids,
            "mode": "copy",
            "scale": 1.0,
            "rows": list(members),
            "cut_id": correct_cut.fingerprint,
            "parent_cut": None,
            "source_authority": "independent-native-prefill-context",
        },
    ))
    all_cuts.extend((correct_cut,))

    for donor_name, donor_rows, mode, scale in donor_specs:
        donor = session.fork(parent_cut)
        act = _memory_act(donor, source_layer, donor_rows, mode=mode, scale=scale, name=donor_name)
        receipt = donor.apply(act)
        donor_cut = donor.capture(retain=False)
        donor_sessions[donor_name] = donor
        donors.append({
            "name": donor_name,
            "mode": mode,
            "scale": scale,
            "rows": list(donor_rows),
            "act": act.manifest(),
            "receipt": receipt.to_dict(),
            "cut_id": donor_cut.fingerprint,
            "parent_cut": parent_cut.fingerprint,
        })
        all_cuts.extend((donor_cut,))
        all_receipts.append(receipt)

    definition_operator = next(row["act"] for row in donors if row["name"] == "zero_coalition")
    definition_operator_ref = f"cas://sha256/{_digest(definition_operator)}"
    spectral_space, spectral_address, spectral_record = _spectral_native_record(
        symbol=binding.symbol,
        scene_id=session_id,
        instance_id="main/source",
        parent_cut=parent_cut,
        binding_fingerprint=binding_fingerprint,
        operator_reference=definition_operator_ref,
        consumer=consumer,
    )

    arm_specs = [
        {"name": "parent_copy", "donor": "parent_copy", "rows": members,
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source"},
        {"name": "matched_source_donor", "donor": "source_correct", "rows": members,
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source"},
        {"name": "wrong_row_control", "donor": "source_correct", "rows": [wrong_row],
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source",
         "control_kind": "same-donor-payload-wrong-target-row"},
        {"name": "zero_member_0", "donor": "zero_member_0", "rows": [members[0]],
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source"},
        {"name": "zero_member_1", "donor": "zero_member_1", "rows": [members[-1]],
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source"},
        {"name": "zero_coalition", "donor": "zero_coalition", "rows": members,
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source"},
        # The donor already contains the one permitted 0.5 scaling.  Install
        # it as an exact copy so the recorded arm dose cannot silently become
        # a second 0.5 multiplication.
        {"name": "half_coalition", "donor": "half_coalition", "rows": members,
         "layers": [source_layer], "mode": "copy", "scale": 1.0, "location_set": "source"},
    ]
    planned_native_transitions = (len(arm_specs) + 1) * (int(adapter.layers) + 5)
    planned_act_operations = len(donor_specs) + len(arm_specs)
    # These are independent helper calls around the raw native suffix/Act
    # count.  The recipient and matched donor each perform one public prefill;
    # each candidate arm restores the recipient parent once, followed by one
    # Project reopen restore and one exported-project restore.  Keep these
    # categories separate so the shared baseline is never charged twice.
    planned_prefill_forward_calls = 2
    planned_native_restore_operations = len(arm_specs)
    planned_project_reopen_restore_operations = 1
    planned_export_reopen_restore_operations = 1
    planned_restore_operations = (
        planned_native_restore_operations
        + planned_project_reopen_restore_operations
        + planned_export_reopen_restore_operations
    )
    planned_bounded_operations = (
        planned_native_transitions
        + planned_act_operations
        + planned_prefill_forward_calls
        + planned_restore_operations
    )
    if planned_bounded_operations > MAX_NATIVE_OPERATIONS:
        raise RuntimeError(
            "instrument-broken: native assay plan exceeds the 256-operation budget "
            f"(bounded={planned_bounded_operations}; suffix={planned_native_transitions}, "
            f"Acts={planned_act_operations}, prefills={planned_prefill_forward_calls}, "
            f"restores={planned_restore_operations})"
        )
    baseline_probe = _native_token_probe(adapter, parent_cut, answer_tokens=answer_tokens)
    all_cuts.append(baseline_probe["result_cut"])
    all_receipts.append(baseline_probe["receipt_object"])
    uses = []
    interventions: list[dict[str, Any]] = []
    vm_bundles: list[tuple[Any, Mapping[str, Any]]] = []
    for arm in arm_specs:
        arm_name = arm["name"]
        donor_name = arm["donor"]
        arm_rows = list(arm["rows"])
        arm_layers = list(arm["layers"])
        mode = arm["mode"]
        scale = float(arm["scale"])
        candidate = session.fork(parent_cut)
        act_receipts = []
        act_manifests = []
        act_parent_cuts = []
        if donor_name == "parent_copy":
            donor_keys, donor_values = session.read("keys"), session.read("values")
            donor_cut = parent_cut
        else:
            donor = donor_sessions[donor_name]
            donor_keys, donor_values = donor.read("keys"), donor.read("values")
            donor_cut = next(row["cut_id"] for row in donors if row["name"] == donor_name)
        donor_cut_id = donor_cut.fingerprint if hasattr(donor_cut, "fingerprint") else str(donor_cut)
        for layer in arm_layers:
            act_parent_cuts.append(candidate.capture(retain=False))
            act = _memory_act(
                candidate,
                layer,
                arm_rows,
                mode=mode,
                scale=scale,
                donor=None if mode == "zero" else (donor_keys, donor_values),
                name=f"install_{arm_name}_layer_{layer}",
            )
            act_receipts.append(candidate.apply(act))
            act_manifests.append(act.manifest())
        candidate_cut = candidate.capture(retain=False)
        probe = _public_probe(
            adapter,
            parent_cut,
            candidate_cut,
            name=arm_name,
            consumer=consumer,
            answer_tokens=answer_tokens,
            baseline_probe=baseline_probe,
        )
        restore_receipt = session.restore(parent_cut)
        all_cuts.extend(act_parent_cuts)
        all_cuts.extend((candidate_cut, probe["candidate_result"]))
        all_receipts.extend(act_receipts)
        all_receipts.extend((probe["candidate_receipt_object"], restore_receipt))
        restore_exact = restore_receipt.to_dict().get("verified_exact") is True and session.capture(retain=False).fingerprint == parent_cut.fingerprint
        candidate_receipt = probe["candidate_receipt_object"]
        candidate_parent_ref = f"cas://sha256/{probe['candidate_receipt_parent_cut']}"
        result_ref = f"cas://sha256/{probe['candidate_result_cut']}"
        receipt_ref = f"cas://sha256/{candidate_receipt.fingerprint}"
        restore_ref = f"cas://sha256/{restore_receipt.fingerprint}"
        vm_bundle, vm_payloads = _vm_execution_bundle(
            run_id=f"symbol-lifecycle:{arm_name}",
            parent_cut=parent_cut.fingerprint,
            candidate_cut=candidate_cut.fingerprint,
            result_cut=probe["candidate_result_cut"],
            model_identity=adapter.model_identity,
            execution=binding.execution,
            context=context,
            operator_manifest=act_manifests,
            consumer=consumer,
            receipt=receipt_ref,
        )
        vm_bundles.append((vm_bundle, vm_payloads))
        observations = ({
            "arm": arm_name,
            "control_kind": arm.get("control_kind", "intervention"),
            "donor_cut": donor_cut_id,
            "candidate_cut": candidate_cut.fingerprint,
            "candidate_receipt_parent_cut": probe["candidate_receipt_parent_cut"],
            "baseline_result_cut": probe["baseline_result_cut"],
            "candidate_result_cut": probe["candidate_result_cut"],
            "tokens_equal": probe["tokens_equal"],
            "baseline_tokens": probe["baseline_tokens"],
            "candidate_tokens": probe["candidate_tokens"],
            "native_consumer": consumer,
            "baseline_margin": probe["baseline_margin"],
            "candidate_margin": probe["candidate_margin"],
            "margin_delta_paris_minus_rome": (
                probe["candidate_margin"]["margin_paris_minus_rome"]
                - probe["baseline_margin"]["margin_paris_minus_rome"]
            ),
            "margin_delta_berlin_minus_paris": (
                probe["candidate_margin"]["margin_berlin_minus_paris"]
                - probe["baseline_margin"]["margin_berlin_minus_paris"]
            ),
            "native_transition_count": probe["native_transition_count"],
            "baseline_argmax_token_id": probe["baseline_margin"]["argmax_token_id"],
            "candidate_argmax_token_id": probe["candidate_margin"]["argmax_token_id"],
            "exact_parent_input": probe["exact_parent_input"],
            "exact_candidate_input": probe["exact_candidate_input"],
            "act_receipts": [receipt.fingerprint for receipt in act_receipts],
            "candidate_receipt": candidate_receipt.fingerprint,
            "restore_receipt": restore_receipt.fingerprint,
            "selected_parent_restored_exact": restore_exact,
            "effect_authority": "native-public-decoder-suffix-observation",
        },)
        use_locations = late_locations if arm["location_set"] == "late" else source_locations
        use = SymbolUse(
            use_id=f"use:{arm_name}",
            binding_fingerprint=binding_fingerprint,
            session_id=session_id,
            consumer=consumer,
            source_cut=parent_cut.fingerprint,
            # The native one-token receipt's parent is the exact candidate
            # fork cut; the lifecycle join requires that immediate parent.
            parent_cut=probe["candidate_receipt_parent_cut"],
            result_cut=probe["candidate_result_cut"],
            branch=f"arm:{arm_name}",
            execution=binding.execution,
            receipt=receipt_ref,
            locations=use_locations,
            clock=_clock_witness(
                probe["candidate_result"],
                provenance=(f"cas://sha256/{parent_cut.fingerprint}", candidate_parent_ref, result_ref, receipt_ref),
                execution=binding.execution,
                consumer=consumer,
                reset="session-from-parent-cut",
            ),
            signed_observations=observations,
            evidence_refs=(f"cas://sha256/{parent_cut.fingerprint}", candidate_parent_ref, receipt_ref, result_ref, restore_ref),
            status="observed",
            validity_status="supported",
            validity_support=(receipt_ref, restore_ref),
            metadata={
                "arm": arm_name,
                "control_kind": arm.get("control_kind", "intervention"),
                "donor_cut": donor_cut_id,
                "rows": list(arm_rows),
                "layers": arm_layers,
                "intervention_mode": mode,
                "install_scale": scale,
                "act_receipts": [receipt.fingerprint for receipt in act_receipts],
                "spectral_address_ref": spectral_address.fingerprint,
                "vm_bundle_ref": vm_bundle.reference,
                "vm_bundle_summary": vm_bundle.summary(),
                "consumer_authority": "native-public-session",
                "causal_necessity": "not-established-by-one-specimen",
            },
        )
        uses.append(use)
        interventions.append({
            "name": arm_name,
            "control_kind": arm.get("control_kind", "intervention"),
            "rows": list(arm_rows),
            "layers": arm_layers,
            "mode": mode,
            "scale": scale,
            "donor_cut": donor_cut_id,
            "candidate_cut": candidate_cut.fingerprint,
            "act": act_manifests,
            "act_receipts": [receipt.to_dict() for receipt in act_receipts],
            "vm_bundle": vm_bundle.to_dict(),
            "probe": {key: value for key, value in probe.items()
                      if key not in {"candidate_receipt_object", "baseline_receipt_object", "candidate_result", "baseline_result",
                                     "baseline_step_cuts", "candidate_step_cuts",
                                     "baseline_margin_result", "candidate_margin_result", "baseline_margin_receipt", "candidate_margin_receipt",
                                     "baseline_margin_receipts", "candidate_margin_receipts", "baseline_margin_cuts", "candidate_margin_cuts"}},
            "restore": restore_receipt.to_dict(),
            "restore_exact": restore_exact,
        })

    actual_native_transitions = int(baseline_probe["native_transition_count"]) + sum(
        int(row["probe"]["candidate_native_transition_count"]) for row in interventions
    )
    actual_act_operations = sum(
        1 for receipt in all_receipts if receipt.to_dict().get("operation") == "apply"
    )
    actual_restore_operations = sum(
        1 for receipt in all_receipts if receipt.to_dict().get("operation") == "restore"
    )
    # ``reopen_restore`` and ``exported_restore`` are deliberately outside
    # ``all_receipts``: they validate persisted Project custody after the
    # native arms finish.  Report them separately rather than hiding them or
    # charging the shared baseline again.
    project_reopen_restore_operations = 1
    export_reopen_restore_operations = 1
    total_restore_operations = (
        actual_restore_operations
        + project_reopen_restore_operations
        + export_reopen_restore_operations
    )
    prefill_forward_calls = planned_prefill_forward_calls
    bounded_operations = (
        actual_native_transitions
        + actual_act_operations
        + prefill_forward_calls
        + total_restore_operations
    )
    native_budget = {
        "max_operations": MAX_NATIVE_OPERATIONS,
        "planned_native_transitions": planned_native_transitions,
        "actual_native_transitions": actual_native_transitions,
        "planned_act_operations": planned_act_operations,
        "actual_act_operations": actual_act_operations,
        "planned_prefill_forward_calls": planned_prefill_forward_calls,
        "prefill_forward_calls": prefill_forward_calls,
        "planned_native_restore_operations": planned_native_restore_operations,
        "restore_operations": actual_restore_operations,
        "project_reopen_restore_operations": project_reopen_restore_operations,
        "export_reopen_restore_operations": export_reopen_restore_operations,
        "planned_restore_operations": planned_restore_operations,
        "total_restore_operations": total_restore_operations,
        "counted_operations": actual_native_transitions + actual_act_operations,
        "planned_bounded_operations": planned_bounded_operations,
        "bounded_operations": bounded_operations,
        "arms": len(arm_specs),
        "shared_baseline": True,
        "native_scope": "one-token-step-receipts-and-registered-Acts; shared baseline counted once",
        "helper_scope": "independent recipient+donor prefills and native/project/export restores reported separately",
    }
    if native_budget["bounded_operations"] > MAX_NATIVE_OPERATIONS:
        raise RuntimeError(
            "instrument-broken: observed native assay exceeded the 256-operation budget "
            f"(bounded={native_budget['bounded_operations']}; raw={native_budget['counted_operations']})"
        )

    definition_vm_bundle, definition_vm_payloads = _vm_execution_bundle(
        run_id="symbol-lifecycle:definition",
        parent_cut=parent_cut.fingerprint,
        candidate_cut=parent_cut.fingerprint,
        result_cut=parent_cut.fingerprint,
        model_identity=adapter.model_identity,
        execution=binding.execution,
        context=context,
        operator_manifest=definition_operator,
        consumer=consumer,
        receipt=f"cas://sha256/{parent_cut.fingerprint}",
    )
    vm_bundles.append((definition_vm_bundle, definition_vm_payloads))
    occurrence = SymbolOccurrence.for_binding(
        binding,
        occurrence_id="definition:cot.source.coalition:parent",
        session_id=session_id,
        source_cut=parent_cut.fingerprint,
        locations=source_locations,
        branch="main",
        execution=binding.execution,
        coalition=tuple(f"{component}:layer={source_layer}:row={row}" for component in ("keys", "values") for row in members),
        location_sets={
            "source_rows": [f"row:{row}" for row in members],
            "source_layer": [f"layer:{source_layer}"],
            "late_layers": [f"layer:{layer}" for layer in late_layers],
            "kv_heads": [f"head:{head}" for head in source_heads],
            "binding_layers": [f"layer:{layer}" for layer in binding_layers],
        },
        clock=source_clock,
        spectral_address=spectral_address,
        vm_bundle=definition_vm_bundle,
        cot_refs=({
            "kind": "locator-annotation",
            "interpretation": "supplied-row-nomination",
            "causal_dependency_tested": False,
            "semantic_cot_qualified": False,
            "source": "public-native-cache-coalition",
        },),
        definition_refs=(f"cas://sha256/{parent_cut.fingerprint}", definition_vm_bundle.reference,
                         spectral_address.fingerprint),
        evidence_refs=(f"cas://sha256/{parent_cut.fingerprint}", definition_vm_bundle.reference),
        status="observed",
        validity_status="supported",
        validity_support=(f"cas://sha256/{parent_cut.fingerprint}", definition_vm_bundle.reference),
        metadata={
            "selection_mode": selection_mode,
            "token_labels_are_descriptive": True,
            "state_schema": frame.surface["state_schema"],
            "consumer": consumer,
            "native_authority": "public-session-decoder-memory-act",
            "spectral_mapping_status": spectral_record["mapping_status"],
            "spectral_address_fingerprint": spectral_address.fingerprint,
            "vm_bundle_reference": definition_vm_bundle.reference,
            "native_source_cut": parent_cut.fingerprint,
            "native_operator_reference": definition_operator_ref,
            "native_consumer": consumer,
        },
    )

    project = Project(project_root)
    project.save_session(
        session_id,
        session,
        all_cuts,
        debug_info={"symbols": [binding.to_dict()], "symbol_context": context},
        receipts=all_receipts,
    )
    reopened_project = Project(project_root)
    loaded_parent = reopened_project.load_cut(session_id, parent_cut.fingerprint, device=str(adapter.device))
    reopen_session = Session.from_cut(adapter, loaded_parent)
    reopen_restore = reopen_session.restore(loaded_parent)
    export_reopen = {
        "project_root": str(project_root),
        "parent_cut": parent_cut.fingerprint,
        "loaded_parent_cut": loaded_parent.fingerprint,
        "exact_loaded_parent": loaded_parent.fingerprint == parent_cut.fingerprint,
        "reopen_restore": reopen_restore.to_dict(),
        "reopen_selected_cut": reopen_session.capture(retain=False).fingerprint,
    }
    mechanics_status = "valid" if all(
        row["restore_exact"]
        and row["probe"]["exact_parent_input"]
        and row["probe"]["exact_candidate_input"]
        and row["probe"]["status"] == "completed"
        for row in interventions
    ) and export_reopen["exact_loaded_parent"] else "invalid"
    instrument_status = (
        "valid"
        if mechanics_status == "valid"
        and interventions
        and all(
            row["probe"]["baseline_margin"]["status"] == "completed"
            and row["probe"]["candidate_margin"]["status"] == "completed"
            and row["act_receipts"]
            for row in interventions
        )
        else "instrument-broken"
    )
    trend_status = (
        "observations-only"
        if interventions
        and all(
            row["probe"]["baseline_margin"]["status"] == "completed"
            and row["probe"]["candidate_margin"]["status"] == "completed"
            for row in interventions
        )
        else "evidence-incomplete"
    )
    lifecycle = SymbolLifecycle.define(
        binding,
        definitions=(occurrence,),
        uses=tuple(uses),
        mechanics_status=mechanics_status,
        instrument_status=instrument_status,
        trend_status=trend_status,
        terminal_status="not-assessed",
        metadata={
            "assay": "public-session-decoder-memory-v1",
            "native_consumer": consumer,
            "source_layer": source_layer,
            "source_members": members,
            "late_layers": late_layers,
            "selection_mode": selection_mode,
            "spectral_address": spectral_record,
            "spectral_address_fingerprint": spectral_address.fingerprint,
            "vm_bundle": definition_vm_bundle.summary(),
            "native_budget": native_budget,
            "vm_authority": "pointer-only VM metadata; native Session/Project receipts remain authoritative",
            "cot_authority": "candidate locator only; native swap receipts decide use evidence",
            "semantic_resolution": "unqualified",
        },
    )
    lifecycle_join_full = _save_and_reload_lifecycle(
        reopened_project,
        session_id,
        lifecycle,
        port="keys",
        selector=[source_layer, 0, 0, members[0]],
        cut_id=parent_cut.fingerprint,
        branch="main",
    )
    lifecycle_join = {
        "saved_lifecycle_id": lifecycle_join_full["saved"].get("lifecycle_id"),
        "reloaded_lifecycle_id": lifecycle_join_full["reloaded"].get("lifecycle_id"),
        "listed_count": len(lifecycle_join_full["listed"]),
        "located_count": len(lifecycle_join_full["located"].get("matches", [])),
        "located_ids": [row.get("occurrence_id") for row in lifecycle_join_full["located"].get("matches", [])],
    }
    query_kwargs = {
        "phases": spectral_address.phases,
        "namespace": spectral_address.canonical_key.namespace,
        "symbol_type": spectral_address.canonical_key.symbol_type,
        "frame_id": spectral_address.frame.frame_id,
        "exact_symbol_id": spectral_address.canonical_key.symbol_id,
        "exact_scene_id": spectral_address.instance_key.scene_id,
        "exact_instance_id": spectral_address.instance_key.instance_id,
        "exact_role_id": spectral_address.role_transform.role_id,
    }
    sas_queries = {
        "correct": SpectralQuery(
            query_id="source-exact", transaction_id="read", **query_kwargs
        ),
        "wrong_instance": dataclasses.replace(
            SpectralQuery(query_id="source-wrong-instance", transaction_id="read", **query_kwargs),
            exact_instance_id="wrong-instance",
        ),
        "wrong_role": dataclasses.replace(
            SpectralQuery(query_id="source-wrong-role", transaction_id="read", **query_kwargs),
            exact_role_id="wrong-role",
        ),
    }
    spectral_queries = {
        name: {
            "query": dataclasses.asdict(query),
            "result": reopened_project.resolve_symbol_address(
                session_id, query, cut_id=parent_cut.fingerprint, branch="main"
            ),
        }
        for name, query in sas_queries.items()
    }
    vm_descriptor_custody = _write_vm_plane_descriptors(project_root, vm_bundles)
    import shutil

    shutil.copytree(project_root, export_root)
    reopened_export = Project(export_root)
    exported_lifecycle = reopened_export.symbol_lifecycle(session_id, lifecycle.fingerprint)
    exported_parent = reopened_export.load_cut(
        session_id, parent_cut.fingerprint, device=str(adapter.device)
    )
    exported_session = Session.from_cut(adapter, exported_parent)
    exported_restore = exported_session.restore(exported_parent)
    archive_info = _archive_project(
        project_root, export_root, root / "native-project-archive.tar.gz"
    )
    (root / "native-session-manifest.json").write_text(
        json.dumps(
            {
                "schema": "symbol-lifecycle-native-session-manifest-v1",
                "model": spec.to_dict(),
                "session_id": session_id,
                "parent_cut": parent_cut.fingerprint,
                "lifecycle": lifecycle.fingerprint,
                "execution": adapter.execution,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "symbol-lifecycle-native-assay-v2",
        "model": spec.to_dict(),
        "loaded_model_spec": loaded_identity,
        "config": {"model": args.model, "prompt": PROMPT, "query": QUERY, "device": device,
                    "session_id": session_id, "worker_root": str(root)},
        "native_consumer": consumer,
        "mechanics_status": mechanics_status if exported_lifecycle["lifecycle_id"] == lifecycle.fingerprint else "invalid",
        "instrument_status": instrument_status,
        "instrument": "public-decoder-memory-act-v1",
        "trend_status": trend_status,
        "terminal_status": "not-assessed",
        "model_identity": adapter.model_identity,
        "execution": adapter.execution,
        "native_budget": native_budget,
        "source_context": {
            "prompt": PROMPT,
            "query": QUERY,
            "matched_donor_prompt": CORRECT_DONOR_PROMPT,
            "unrelated_content_control_prompt": WRONG_DONOR_PROMPT,
            "token_ids": token_ids,
            "answer_token_ids": answer_tokens,
            "committed_token_count": len(token_ids) - 1,
            "source_parent_cut": parent_cut.fingerprint,
            "branch": "main",
            "execution_point": dict(frame.execution_point),
            "state_schema": frame.surface["state_schema"],
            "consumer": consumer,
        },
        "source_coalition": {
            "selection_mode": selection_mode,
            "layer": source_layer,
            "members": source_rows,
            "locations": [location.to_dict() for location in source_locations],
            "late_locations": [location.to_dict() for location in late_locations],
            "selector_span": {"layer": source_layer, "rows": members, "heads": source_heads,
                               "components": ["key", "value"]},
        },
        "symbol_binding": binding.to_dict(),
        "symbol_occurrence": occurrence.to_dict(),
        "spectral_address": spectral_record,
        "spectral_queries": spectral_queries,
        "vm_bundle": definition_vm_bundle.to_dict(),
        "vm_descriptor_custody": vm_descriptor_custody,
        "donors": donors,
        "interventions": interventions,
        "symbol_uses": [use.to_dict() for use in uses],
        "lifecycle": {
            "fingerprint": lifecycle.fingerprint,
            "join": lifecycle_join,
            "reloaded": {
                "lifecycle_id": lifecycle_join["reloaded_lifecycle_id"],
                "listed_count": lifecycle_join["listed_count"],
                "located_count": lifecycle_join["located_count"],
            },
        },
        "export_reopen": export_reopen,
        "exported_project": {
            "root": str(export_root),
            "lifecycle": {
                "lifecycle_id": exported_lifecycle["lifecycle_id"],
                "definitions": len(exported_lifecycle["lifecycle"]["definitions"]),
                "uses": len(exported_lifecycle["lifecycle"]["uses"]),
            },
            "exact_lifecycle": exported_lifecycle["lifecycle_id"] == lifecycle.fingerprint,
            "parent_cut": exported_parent.fingerprint,
            "restore": exported_restore.to_dict(),
            "selected_cut": exported_session.capture(retain=False).fingerprint,
        },
        "native_project_archive": archive_info,
        "project": {"root": str(project_root), "sessions": reopened_project.sessions(),
                    "symbols": reopened_project.symbols(session_id)},
        "limitations": [
            "One pretrained Qwen specimen and one short prompt; no semantic-generalization claim.",
            "Token text nominates source rows but does not prove a unique semantic symbol.",
            "The typed edit is a present committed-cache replacement; it does not reconstruct historical execution.",
            "The native consumer is the public decoder suffix; Project lifecycle is read-only metadata over retained public cuts.",
            "The Germany/Berlin donor is a matched payload control, not proof of Paris/Rome correctness; Berlin is reported separately.",
            "The Italy/Rome prompt is retained only as unrelated content metadata, not as a wrong-source geometry control; wrong_row_control uses the matched donor payload at a different target row.",
            "The spectral address and VM bundle are typed native metadata registrations; semantic resolution and VM payload equivalence remain unqualified.",
        ],
    }


def run_mechanics(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise public Session/Project reopen mechanics without model download or CUDA."""

    from mdb import BenchSession, Project

    project_root = (
        Path(args.project_root).expanduser().resolve()
        if args.project_root
        else Path(args.root).expanduser().resolve().parent / "mechanics-project"
    )
    project = Project(project_root)
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        # The base local checkout intentionally does not carry Torch.  Keep a
        # pure-Python Bench mechanics smoke available; the native CUDA worker
        # still requires the staged autoregressive wheels.
        from mdb.counter_debugger import CounterDebuggerAdapter

        adapter = CounterDebuggerAdapter()
        with BenchSession.from_backend(adapter, identity={"model_id": "counter-plus-one"}) as owner:
            debugger = owner.debugger()
            debugger.wait_for_pause(timeout_s=1.0)
            parent = debugger.checkpoint("parent")
            stepped = debugger.step(1)
            restored = debugger.restore(parent["cut_id"])
        reopened = Project(project_root)
        sessions = reopened.sessions()
        return {
            "schema": "symbol-lifecycle-mechanics-v1",
            "mechanics_status": "valid" if restored.get("cut_id") == parent["cut_id"] else "invalid",
            "instrument_status": "not-applicable",
            "instrument": "public-mdb-counter-debugger",
            "trend_status": "not-assessed",
            "terminal_status": "not-assessed",
            "parent": parent,
            "stepped": stepped,
            "restored": restored,
            "project_sessions": sessions,
            "project_status": "initialized-only; autoregressive public Session unavailable: " + str(exc),
        }
    MemoryAdapter = _memory_adapter_class()
    adapter = MemoryAdapter.tiny("qwen2", seed=113, device="cpu", granularity="operation")
    native = adapter.session([1, 2, 3])
    parent = native.capture(retain=True)
    act = _memory_act(native, 0, [0], mode="zero", name="mechanics_zero_row")
    applied = native.apply(act)
    changed = native.capture(retain=False)
    restored = native.restore(parent)
    project.save_session(
        "mechanics",
        native,
        [parent, changed],
        debug_info={},
        receipts=native.receipts,
    )
    sessions = project.sessions()
    reopened = Project(project_root)
    graph = reopened.graph("mechanics")
    verification = graph.verify(raise_on_error=False)
    loaded = reopened.load_cut("mechanics", parent.fingerprint, device="cpu")
    return {
        "schema": "symbol-lifecycle-mechanics-v1",
        "mechanics_status": "valid" if verification.ok and restored.to_dict().get("result") == parent.fingerprint and loaded.fingerprint == parent.fingerprint else "invalid",
        "instrument_status": "valid",
        "instrument": "public-decoder-memory-act-v1",
        "trend_status": "not-assessed",
        "terminal_status": "not-assessed",
        "parent": {"fingerprint": parent.fingerprint, **parent.manifest()},
        "applied": applied.to_dict(),
        "changed_cut": changed.fingerprint,
        "restored": restored.to_dict(),
        "project_sessions": sessions,
        "project_graph": graph.inspect(),
        "project_graph_verify": {"ok": verification.ok, "nodes": verification.node_count, "edges": verification.edge_count, "errors": list(verification.errors)},
        "reopened_parent_cut": loaded.fingerprint,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "mechanics"), default="native")
    parser.add_argument("--model", default="qwen2.5-0.5b-instruct")
    parser.add_argument("--root", default="./symbol-lifecycle-session")
    parser.add_argument("--export-root")
    parser.add_argument("--project-root")
    parser.add_argument("--output", default="./symbol-lifecycle-report.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tiny-fixture", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    try:
        report = run_mechanics(args) if args.mode == "mechanics" else run_native(args)
    except Exception as exc:
        report = {
            "schema": "symbol-lifecycle-native-assay-v1" if args.mode == "native" else "symbol-lifecycle-mechanics-v1",
            "mechanics_status": "invalid",
            "instrument_status": "instrument-broken",
            "trend_status": "evidence-incomplete",
            "terminal_status": "not-assessed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }
        _write_report(report, output)
        return 1
    _write_report(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

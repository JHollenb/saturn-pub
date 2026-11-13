"""Counterfactual branches over a sealed native ATen forward trace.

Every branch starts from the declared tensor roots. Interventions are applied
to produced SSA handles at their producer boundary, using Saturn-pub's typed
``Act`` identity; the captured native ATen operations remain the suffix
consumer. Values captured by the original forward are observations only and
are never replay roots. Replacing a view-valued SSA result clones and rebinds
that handle only; it does not mutate the root or any other alias.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import torch
from saturn_pub import Act, Receipt, StateCut
from saturn_pub.values import canonical, clone, digest

from .forward_recording import ATenStepper, ForwardRecordingError, _references, tensor_descriptor
from .forward_symbols import verify_trace


class ForwardInterventionError(ForwardRecordingError):
    """A native replay or typed SSA replacement is outside the declared contract."""


@dataclass(frozen=True)
class ForwardInterventionResult:
    """One same-parent branch, its native output observations, and sealed receipt."""

    output: Any
    output_differences: tuple[dict[str, Any], ...]
    receipt: Receipt
    descendant_observations: tuple[dict[str, Any], ...]
    step_observations: tuple[dict[str, Any], ...]
    selected_values: dict[str, torch.Tensor] = field(default_factory=dict)


def _raise(label: str, error: Exception) -> ForwardInterventionError:
    return ForwardInterventionError(f"{label}: {error}")


def _is_finite(value: torch.Tensor) -> bool:
    try:
        return bool(torch.isfinite(value).all().item())
    except (RuntimeError, TypeError) as exc:
        raise ForwardInterventionError("replacement tensor must support a finite-value check") from exc


def _root_fingerprint(trace: Mapping[str, Any], roots: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    root_rows: dict[str, Any] = {}
    if any(not isinstance(key, str) or not key for key in roots):
        raise ForwardInterventionError("replay root IDs must be non-empty text")
    for value_id in sorted(roots):
        value = roots[value_id]
        if not isinstance(value, torch.Tensor):
            raise ForwardInterventionError(f"replay root {value_id} is not a tensor")
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ForwardInterventionError(f"replay root {value_id} differs from the captured CPU tensor contract")
        if not value.is_contiguous() or value.storage_offset() != 0:
            raise ForwardInterventionError(f"replay root {value_id} differs from the captured dense root contract")
        try:
            descriptor = tensor_descriptor(value)
        except Exception as exc:  # tensor adapters can fail for opaque layouts
            raise _raise(f"cannot fingerprint replay root {value_id}", exc) from exc
        root_rows[value_id] = {
            "sha256": descriptor["sha256"],
            "shape": descriptor["shape"],
            "dtype": descriptor["dtype"],
            "device": str(value.device),
        }
    body = {
        "schema": "mdb-forward-intervention-parent-v1",
        "trace_fingerprint": trace["fingerprint"],
        "model_identity": trace["model_identity"],
        "execution": trace["execution"],
        "input_identity": trace["input_identity"],
        "roots": root_rows,
    }
    return digest(body), root_rows


def _stepper(trace: Mapping[str, Any], roots: Mapping[str, Any]) -> ATenStepper:
    try:
        return ATenStepper(trace, roots)
    except ForwardRecordingError as exc:
        raise ForwardInterventionError(str(exc)) from exc


def _validate_replacement(parent: Any, replacement: Any, value_id: str) -> None:
    if not isinstance(parent, torch.Tensor) or not isinstance(replacement, torch.Tensor):
        raise ForwardInterventionError(f"SSA value {value_id} replacement must be a tensor")
    if parent.layout != torch.strided or replacement.layout != torch.strided:
        raise ForwardInterventionError("sparse or opaque tensor layouts are unsupported")
    if parent.device.type != "cpu" or replacement.device != parent.device:
        raise ForwardInterventionError("native branching requires a replacement on the captured CPU device")
    if tuple(replacement.shape) != tuple(parent.shape):
        raise ForwardInterventionError(f"replacement shape differs at SSA value {value_id}")
    if replacement.dtype != parent.dtype:
        raise ForwardInterventionError(f"replacement dtype differs at SSA value {value_id}")
    if not _is_finite(parent) or not _is_finite(replacement):
        raise ForwardInterventionError(f"replacement or native parent is non-finite at SSA value {value_id}")


def _dose_value(value: Any, value_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForwardInterventionError(f"dose for SSA value {value_id} must be a finite number")
    dose = float(value)
    if not math.isfinite(dose):
        raise ForwardInterventionError(f"dose for SSA value {value_id} must be finite")
    return dose


def _blend(parent: torch.Tensor, replacement: torch.Tensor, dose: float, value_id: str) -> torch.Tensor:
    if dose == 0:
        result = parent.detach().clone(memory_format=torch.preserve_format)
    elif dose == 1:
        result = replacement.detach().clone(memory_format=torch.preserve_format)
    elif parent.is_floating_point() or parent.is_complex():
        result = parent + dose * (replacement - parent)
    else:
        raise ForwardInterventionError(
            f"non-binary dose at SSA value {value_id} requires a floating or complex tensor"
        )
    if not _is_finite(result):
        raise ForwardInterventionError(f"dose produced non-finite replacement at SSA value {value_id}")
    return result.detach().clone(memory_format=torch.preserve_format)


def _safe_number(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _scalar_digest(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        value = {"kind": "float", "value": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    elif isinstance(value, torch.dtype):
        value = {"kind": "dtype", "value": str(value)}
    elif isinstance(value, torch.device):
        value = {"kind": "device", "value": str(value)}
    elif isinstance(value, (torch.layout, torch.memory_format)):
        value = {"kind": "torch", "value": str(value)}
    else:
        value = {"kind": type(value).__name__, "value": value}
    return digest(value)


def _tensor_difference(native: torch.Tensor, candidate: torch.Tensor, path: str) -> dict[str, Any]:
    native_desc, candidate_desc = tensor_descriptor(native), tensor_descriptor(candidate)
    geometry_matches = (
        tuple(native.shape) == tuple(candidate.shape)
        and native.dtype == candidate.dtype
        and native.device == candidate.device
    )
    row: dict[str, Any] = {
        "path": path,
        "kind": "tensor",
        "native_digest": native_desc["sha256"],
        "candidate_digest": candidate_desc["sha256"],
        "native_shape": native_desc["shape"],
        "candidate_shape": candidate_desc["shape"],
        "native_dtype": native_desc["dtype"],
        "candidate_dtype": candidate_desc["dtype"],
        "native_device": str(native.device),
        "candidate_device": str(candidate.device),
        "geometry_matches": geometry_matches,
        "changed": native_desc["sha256"] != candidate_desc["sha256"],
    }
    if not geometry_matches:
        return row
    if native.numel() == 0:
        row.update({"mean_delta": 0.0, "mean_abs_delta": 0.0, "max_abs_delta": 0.0})
        return row
    if native.dtype == torch.bool:
        changed = torch.count_nonzero(native != candidate).item()
        row["changed_elements"] = int(changed)
        return row
    try:
        dtype = torch.complex128 if native.is_complex() else torch.float64
        delta = candidate.to(dtype=dtype) - native.to(dtype=dtype)
        absolute = delta.abs()
        mean = delta.mean().item()
        if isinstance(mean, complex):
            mean_delta: Any = {"real": _safe_number(mean.real), "imag": _safe_number(mean.imag)}
        else:
            mean_delta = _safe_number(mean)
        row.update({
            "mean_delta": mean_delta,
            "mean_abs_delta": _safe_number(absolute.mean().item()),
            "max_abs_delta": _safe_number(absolute.max().item()),
        })
    except (RuntimeError, TypeError, OverflowError):
        # Content hashes remain useful for unusual numeric dtypes with no common
        # accumulator type; do not turn a diagnostic limit into a branch failure.
        pass
    return row


def _output_differences(native: Any, candidate: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(native, torch.Tensor) and isinstance(candidate, torch.Tensor):
        return [_tensor_difference(native, candidate, path)]
    if isinstance(native, Mapping) and isinstance(candidate, Mapping):
        rows = []
        keys = list(dict.fromkeys([*native.keys(), *candidate.keys()]))
        for key in keys:
            child = f"{path}.{key}"
            if key not in native or key not in candidate:
                left, right = native.get(key), candidate.get(key)
                rows.append({"path": child, "kind": "structure", "changed": True,
                             "native_present": key in native, "candidate_present": key in candidate,
                             "native_digest": None if key not in native else _tree_digest(left),
                             "candidate_digest": None if key not in candidate else _tree_digest(right)})
            else:
                rows.extend(_output_differences(native[key], candidate[key], child))
        return rows
    if isinstance(native, (tuple, list)) and isinstance(candidate, (tuple, list)):
        rows = []
        if type(native) is not type(candidate) or len(native) != len(candidate):
            return [{"path": path, "kind": "structure", "changed": True,
                     "native_type": type(native).__name__, "candidate_type": type(candidate).__name__,
                     "native_length": len(native), "candidate_length": len(candidate)}]
        for index, (left, right) in enumerate(zip(native, candidate, strict=True)):
            rows.extend(_output_differences(left, right, f"{path}[{index}]"))
        return rows
    return [{
        "path": path,
        "kind": "scalar",
        "native_digest": _scalar_digest(native),
        "candidate_digest": _scalar_digest(candidate),
        "changed": _scalar_digest(native) != _scalar_digest(candidate),
    }]


def _tree_digest(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return tensor_descriptor(value)["sha256"]
    if isinstance(value, Mapping):
        return digest({str(key): _tree_digest(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return digest([_tree_digest(item) for item in value])
    return _scalar_digest(value)


def _check_native_output(encoded: Any, actual: Any, values: Mapping[str, Any], path: str = "$") -> None:
    """Bind an optional observed output to the sealed trace; never seed replay with it."""
    if isinstance(encoded, dict) and set(encoded) == {"$value"}:
        value_id = encoded["$value"]
        if not isinstance(actual, torch.Tensor):
            raise ForwardInterventionError(f"native output {path} is not the captured tensor")
        descriptor = tensor_descriptor(actual)
        expected = values[value_id]
        if any(descriptor[key] != expected[key] for key in ("sha256", "shape", "dtype")):
            raise ForwardInterventionError(f"native output {path} differs from the sealed trace")
        return
    if isinstance(encoded, dict) and set(encoded) == {"$tuple"}:
        if not isinstance(actual, tuple) or len(actual) != len(encoded["$tuple"]):
            raise ForwardInterventionError(f"native output {path} tuple structure differs from the trace")
        for index, (item, value) in enumerate(zip(encoded["$tuple"], actual, strict=True)):
            _check_native_output(item, value, values, f"{path}[{index}]")
        return
    if isinstance(encoded, list):
        if not isinstance(actual, list) or len(actual) != len(encoded):
            raise ForwardInterventionError(f"native output {path} list structure differs from the trace")
        for index, (item, value) in enumerate(zip(encoded, actual, strict=True)):
            _check_native_output(item, value, values, f"{path}[{index}]")
        return
    if isinstance(encoded, dict) and set(encoded) == {"$device"}:
        expected = torch.device(encoded["$device"])
        if actual != expected:
            raise ForwardInterventionError(f"native output {path} differs from the sealed trace")
        return
    if isinstance(encoded, dict) and set(encoded) == {"$dtype"}:
        if actual != getattr(torch, encoded["$dtype"]):
            raise ForwardInterventionError(f"native output {path} differs from the sealed trace")
        return
    if isinstance(encoded, dict) and set(encoded) == {"$float"}:
        expected = float(encoded["$float"])
        matches = (
            isinstance(actual, float)
            and ((math.isnan(actual) and math.isnan(expected)) or actual == expected)
        )
        if not matches:
            raise ForwardInterventionError(f"native output {path} differs from the sealed trace")
        return
    if isinstance(encoded, dict) and set(encoded) == {"$torch"}:
        expected = getattr(torch, encoded["$torch"])
        if actual != expected:
            raise ForwardInterventionError(f"native output {path} differs from the sealed trace")
        return
    if isinstance(encoded, dict):
        if not isinstance(actual, Mapping) or set(actual) != set(encoded):
            raise ForwardInterventionError(f"native output {path} mapping structure differs from the trace")
        for key, item in encoded.items():
            _check_native_output(item, actual[key], values, f"{path}.{key}")
        return
    if type(actual) is not type(encoded) or (isinstance(actual, float) and not math.isfinite(actual)) or actual != encoded:
        raise ForwardInterventionError(f"native output {path} differs from the sealed trace")


def _descendant_ids(trace: Mapping[str, Any], targets: set[str]) -> set[str]:
    visible = set(targets)
    for event in trace["events"]:
        if set(event["inputs"]) & visible:
            visible.update(event["outputs"])
    return visible - targets


def _resolve_doses(targets: Mapping[str, Any], dose: float | Mapping[str, float]) -> dict[str, float]:
    if isinstance(dose, Mapping):
        if set(dose) != set(targets):
            raise ForwardInterventionError("dose mapping must name every replacement SSA value exactly once")
        return {key: _dose_value(dose[key], key) for key in targets}
    return {key: _dose_value(dose, key) for key in targets}


def _source_identity(value: Mapping[str, Any] | None, trace: Mapping[str, Any]) -> dict[str, Any]:
    if value is None:
        result = {
            "model_identity": trace["model_identity"],
            "input_identity": trace["input_identity"],
        }
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise ForwardInterventionError("source_identity must be a serializable mapping")
    try:
        canonical(result)
    except (TypeError, ValueError) as exc:
        raise ForwardInterventionError("source_identity must contain finite JSON values") from exc
    return result


def branch_forward_many(
    trace: Mapping[str, Any],
    roots: Mapping[str, torch.Tensor],
    replacements: Mapping[str, torch.Tensor],
    *,
    dose: float | Mapping[str, float] = 1.0,
    native_output: Any = None,
    source_identity: Mapping[str, Any] | None = None,
    observe_values: tuple[str, ...] = (),
    _prepared_parent: "PreparedForwardParent | None" = None,
) -> ForwardInterventionResult:
    """Replay from roots, replace one or more produced SSA values, and run the suffix.

    ``native_output`` is an optional observation-only copy of the original
    forward's output. It is checked against the trace seal and is never used as
    a continuation state or root. Supplying it avoids a second full native
    replay when the recorder's output is already available.
    """
    if _prepared_parent is None:
        verify_trace(trace)
    elif type(_prepared_parent) is not PreparedForwardParent:
        raise ForwardInterventionError("prepared replay requires a native PreparedForwardParent")
    if not isinstance(roots, Mapping):
        raise ForwardInterventionError("roots must be the exact declared tensor-root mapping")
    if not isinstance(replacements, Mapping) or not replacements:
        raise ForwardInterventionError("at least one produced SSA value replacement is required")
    if (not isinstance(observe_values, tuple)
            or any(not isinstance(value_id, str) or not value_id for value_id in observe_values)
            or len(set(observe_values)) != len(observe_values)):
        raise ForwardInterventionError("observe_values must be a tuple of unique non-empty SSA value IDs")
    for value_id in observe_values:
        descriptor = trace["values"].get(value_id)
        if descriptor is None:
            raise ForwardInterventionError(f"unknown observed SSA value: {value_id}")
        if not isinstance(descriptor.get("shape"), list) or not isinstance(descriptor.get("dtype"), str):
            raise ForwardInterventionError(f"observed SSA value {value_id} is not a tensor")
    trace_roots = {key for key, row in trace["values"].items() if row["producer"] is None}
    if set(roots) != trace_roots:
        raise ForwardInterventionError("replay admits exactly declared roots; intermediate oracles are forbidden")
    if _prepared_parent is None:
        root_fp, root_rows = _root_fingerprint(trace, roots)
        machine, retained_prefix = None, []
    else:
        machine, retained_prefix, root_fp, root_rows = _prepared_parent._resume(
            trace, roots, observe_values)

    sites: list[tuple[int, str, torch.Tensor]] = []
    for value_id, replacement in replacements.items():
        row = trace["values"].get(value_id)
        if row is None:
            raise ForwardInterventionError(f"unknown SSA value: {value_id}")
        if row["producer"] is None:
            raise ForwardInterventionError("declared roots are read-only; only produced SSA values can be replaced")
        if not isinstance(replacement, torch.Tensor):
            raise ForwardInterventionError(f"replacement for SSA value {value_id} must be a tensor")
        sites.append((row["producer"], value_id, replacement))
    sites.sort(key=lambda item: (item[0], item[1]))
    doses = _resolve_doses(replacements, dose)
    source = _source_identity(source_identity, trace)

    if _prepared_parent is not None:
        native_output = _prepared_parent._native_reference(trace, roots)
    elif native_output is None:
        baseline_machine = _stepper(trace, roots)
        native_output = baseline_machine.run(require_exact=True)
    else:
        _check_native_output(trace["outputs"], native_output, trace["values"])

    if machine is None:
        machine = _stepper(trace, roots)
    first_producer = sites[0][0]
    if machine.pc > first_producer + 1:
        raise ForwardInterventionError("intervention precedes the prepared native boundary")
    prefix_rows: list[dict[str, Any]] = retained_prefix
    while machine.pc <= first_producer:
        row = machine.step()
        prefix_rows.append({"pc": row["pc"], "operator": row["event"]["operator"],
                            "exact_outputs": row["exact_outputs"]})
        if not row["exact_outputs"]:
            raise ForwardInterventionError(f"native prefix replay differs at step {row['pc']}")

    step_observations = list(prefix_rows)
    site_rows: list[dict[str, Any]] = []

    def apply_site(value_id: str, replacement: torch.Tensor) -> None:
        parent_value = machine.environment[value_id]
        _validate_replacement(parent_value, replacement, value_id)
        applied = _blend(parent_value, replacement, doses[value_id], value_id)
        act = Act.replace(value_id, applied, name="native-aten-ssa-value-replacement")
        act.verify_implementation()
        writes = dict(act.implementation({value_id: parent_value}))
        if set(writes) != {value_id} or not isinstance(writes[value_id], torch.Tensor):
            raise ForwardInterventionError("public replacement Act returned an invalid write set")
        machine.environment[value_id] = writes[value_id].detach().clone(memory_format=torch.preserve_format)
        current_descriptor = tensor_descriptor(parent_value)
        applied_descriptor = tensor_descriptor(machine.environment[value_id])
        replacement_descriptor = tensor_descriptor(replacement)
        site_rows.append({
            "value_id": value_id,
            "producer_event": trace["values"][value_id]["producer"],
            "dose": doses[value_id],
            "parent_value_digest": current_descriptor["sha256"],
            "replacement_digest": replacement_descriptor["sha256"],
            "applied_value_digest": applied_descriptor["sha256"],
            "effective": current_descriptor["sha256"] != applied_descriptor["sha256"],
            "act": act.manifest(),
            "act_fingerprint": digest(act.manifest()),
        })

    for producer, value_id, replacement in sites:
        if producer == first_producer:
            apply_site(value_id, replacement)

    while machine.pc < len(trace["events"]):
        row = machine.step()
        step_observations.append({"pc": row["pc"], "operator": row["event"]["operator"],
                                  "exact_outputs": row["exact_outputs"]})
        for producer, value_id, replacement in sites:
            if producer == row["pc"]:
                apply_site(value_id, replacement)

    branch_output = machine.run(require_exact=False)
    output_differences = _output_differences(native_output, branch_output)
    target_ids = {value_id for _, value_id, _ in sites}
    descendants = _descendant_ids(trace, target_ids)
    observed_values = []
    changed_descendants = []
    changed_values = []
    for value_id in sorted(target_ids | descendants):
        candidate_value = machine.environment[value_id]
        candidate_desc = tensor_descriptor(candidate_value)
        native_desc = trace["values"][value_id]
        matches = candidate_desc["sha256"] == native_desc["sha256"]
        observation = {
            "value_id": value_id,
            "producer_event": native_desc["producer"],
            "native_digest": native_desc["sha256"],
            "candidate_digest": candidate_desc["sha256"],
            "matches_native": matches,
            "is_intervention_site": value_id in target_ids,
        }
        observed_values.append(observation)
        if not matches:
            changed_values.append(value_id)
            if value_id not in target_ids:
                changed_descendants.append(value_id)

    selected_values: dict[str, torch.Tensor] = {}
    selected_value_observations = []
    for value_id in observe_values:
        candidate_value = machine.environment.get(value_id)
        if not isinstance(candidate_value, torch.Tensor):
            raise ForwardInterventionError(f"observed SSA value {value_id} did not resolve to a tensor")
        candidate_desc = tensor_descriptor(candidate_value)
        native_desc = trace["values"][value_id]
        if any(candidate_desc[key] != native_desc[key] for key in ("shape", "dtype")):
            raise ForwardInterventionError(f"observed SSA value {value_id} differs from the sealed tensor geometry")
        selected_values[value_id] = candidate_value.detach().clone(memory_format=torch.preserve_format)
        selected_value_observations.append({
            "value_id": value_id,
            "native_digest": native_desc["sha256"],
            "candidate_digest": candidate_desc["sha256"],
            "shape": candidate_desc["shape"],
            "dtype": candidate_desc["dtype"],
            "matches_native": candidate_desc["sha256"] == native_desc["sha256"],
        })

    reuse = {} if _prepared_parent is None else {
        "replay_reuse": _prepared_parent._reuse_receipt(len(trace["events"])),
    }
    receipt = Receipt.make(
        operation="mdb.native-aten.ssa-replacement-branch",
        status="observed",
        model_identity=trace["model_identity"],
        execution=trace["execution"],
        input_identity=trace["input_identity"],
        source_identity=source,
        trace_fingerprint=trace["fingerprint"],
        parent_fingerprint=root_fp,
        root_fingerprints=root_rows,
        native_prefix={"through_event": first_producer, "steps": prefix_rows, "exact": True},
        intervention_sites=site_rows,
        changed_descendants=changed_descendants,
        changed_values=changed_values,
        descendant_observations=observed_values,
        selected_value_observations=selected_value_observations,
        output_differences=output_differences,
        suffix_observations=step_observations[len(prefix_rows):],
        authority={"intervention": "declared-ssa-value-replacement", "causal_effect": "not-established"},
        alias_scope="selected-SSA-handle-only; other aliases and original roots remain independently bound",
        **reuse,
    )
    if _prepared_parent is None and _root_fingerprint(trace, roots)[0] != root_fp:
        raise ForwardInterventionError("branch changed caller-owned replay roots")
    return ForwardInterventionResult(
        output=clone(branch_output),
        output_differences=tuple(output_differences),
        receipt=receipt,
        descendant_observations=tuple(observed_values),
        step_observations=tuple(step_observations),
        selected_values=selected_values,
    )


def _frontier_identity(environment: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Seal complete backing storage and view geometry, including alias groups."""
    groups, storages, values = {}, {}, {}
    for key, value in sorted(environment.items()):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ForwardInterventionError("prepared frontier requires dense CPU tensors")
        storage = value.untyped_storage()
        identity = storage._cdata
        if identity not in groups:
            group = f"storage-{len(groups)}"
            groups[identity] = group
            raw = torch.empty(0, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),), (1,))
            storages[group] = {"bytes": storage.nbytes(),
                               "sha256": tensor_descriptor(raw)["sha256"]}
        values[key] = {"storage": groups[identity], "shape": list(value.shape),
                       "stride": list(value.stride()), "offset": value.storage_offset(),
                       "dtype": str(value.dtype)}
    return {"values": values, "storages": storages}


class PreparedForwardParent:
    """Owned, root-produced native prefix for repeated exact scalar branches.

    Preparation performs one exact root replay. Only its live frontier and
    explicitly requested observations can become branch state. Deep copying the
    frontier as one object preserves backing-storage aliases and view geometry.
    The public StateCut seals the closure descriptor; this first version is a
    live-process owner, not a durable StateCut hydration adapter.
    """

    def __init__(self, trace, roots, *, through_value: str,
                 native_output=None, observe_values: tuple[str, ...] = ()):
        started = time.perf_counter()
        verify_trace(trace)
        row = trace["values"].get(through_value)
        if row is None or row["producer"] is None:
            raise ForwardInterventionError("prepared boundary must be a produced SSA value")
        if not isinstance(observe_values, tuple) or len(set(observe_values)) != len(observe_values):
            raise ForwardInterventionError("prepared observations must be unique SSA IDs")
        if any(key not in trace["values"] for key in observe_values):
            raise ForwardInterventionError("unknown prepared observation")
        self.__trace = copy.deepcopy(trace)
        machine = _stepper(self.__trace, roots)
        self.__roots = machine.environment.copy()
        self.__root_fp, self.__root_rows = _root_fingerprint(self.__trace, self.__roots)
        self.__through = row["producer"]
        prefix = []
        while machine.pc <= self.__through:
            step = machine.step()
            if not step["exact_outputs"]:
                raise ForwardInterventionError("prepared native prefix differs from sealed trace")
            prefix.append({"pc": step["pc"], "operator": step["event"]["operator"],
                           "exact_outputs": True})
        required = set(observe_values) | set(self.__trace["events"][self.__through]["outputs"])
        required.update(_references(self.__trace["outputs"]))
        for event in self.__trace["events"][machine.pc:]:
            required.update(event["inputs"])
        self.__environment = copy.deepcopy({key:value for key,value in machine.environment.items()
                                            if key in required})
        self.__prefix = prefix
        self.__closure = _frontier_identity(self.__environment)
        self.__cut = StateCut(model_identity=self.__trace["model_identity"],
            execution=self.__trace["execution"], boundary="mdb.native-aten.produced-prefix-v1",
            parent=self.__root_fp,
            _payload={"frontier": self.__closure, "trace_fingerprint": self.__trace["fingerprint"],
                      "next_event": machine.pc, "root_fingerprints": self.__root_rows},
            execution_point={"through_event": self.__through},
            surface={"ownership":"private live native frontier", "durable_hydration":False})
        self.__native = clone(machine.run(require_exact=True))
        if native_output is not None:
            _check_native_output(self.__trace["outputs"], native_output, self.__trace["values"])
            if _tree_digest(native_output) != _tree_digest(self.__native):
                raise ForwardInterventionError("prepared native reference differs")
        self.__lock = RLock()
        self.__closed = False
        self.__branches = 0
        self.__seconds = time.perf_counter() - started

    @property
    def fingerprint(self):
        return self.__cut.fingerprint

    @property
    def through_event(self):
        return self.__through

    def manifest(self):
        return copy.deepcopy({"schema":"mdb-prepared-forward-parent-v1", "fingerprint":self.fingerprint,
            "state_cut":self.__cut.manifest(), "trace_fingerprint":self.__trace["fingerprint"],
            "parent_fingerprint":self.__root_fp, "root_fingerprints":copy.deepcopy(self.__root_rows),
            "through_event":self.__through, "prefix_steps":copy.deepcopy(self.__prefix),
            "work":{"root_replay_operators":len(self.__trace["events"]),
                    "reused_prefix_operators_per_branch":self.__through+1,
                    "candidate_operators_per_branch":len(self.__trace["events"])-self.__through-1,
                    "frontier_values":len(self.__closure["values"]),
                    "frontier_storage_bytes":sum(r["bytes"] for r in self.__closure["storages"].values())},
            "preparation_seconds":self.__seconds, "completed_branches":self.__branches,
            "closed":self.__closed})

    def value_descriptors(self):
        return copy.deepcopy(self.__trace["values"])

    def _resume(self, trace, roots, observations):
        if self.__closed:
            raise ForwardInterventionError("prepared parent is closed")
        if trace is not self.__trace or roots is not self.__roots:
            raise ForwardInterventionError("prepared parent belongs to a different trace/root owner")
        self.__cut.verify()
        if _frontier_identity(self.__environment) != self.__closure:
            raise ForwardInterventionError("prepared native frontier changed")
        if any((self.__trace["values"][key]["producer"] is None
                or self.__trace["values"][key]["producer"] <= self.__through)
               and key not in self.__environment for key in observations):
            raise ForwardInterventionError("observation was not retained in prepared frontier")
        machine = object.__new__(ATenStepper)
        machine.trace, machine.pc, machine.steps = self.__trace, self.__through+1, []
        machine.environment = copy.deepcopy(self.__environment)
        return machine, copy.deepcopy(self.__prefix), self.__root_fp, copy.deepcopy(self.__root_rows)

    def _native_reference(self, trace, roots):
        if trace is not self.__trace or roots is not self.__roots:
            raise ForwardInterventionError("prepared native reference has a different owner")
        return self.__native

    def _reuse_receipt(self, event_count):
        return {"schema":"mdb-native-prefix-reuse-v1", "state_cut_fingerprint":self.fingerprint,
                "state_cut_manifest":copy.deepcopy(self.__cut.manifest()), "through_event":self.__through,
                "prefix_operators_reused":self.__through+1,
                "operators_executed":event_count-self.__through-1,
                "prefix_evidence":"exact root-produced preparation replay",
                "verification":"full step/output descriptors; scalar native ATen continuation"}

    def branch(self, replacements, *, dose=1., source_identity=None, observe_values=()):
        with self.__lock:
            if self.__closed:
                raise ForwardInterventionError("prepared parent is closed")
            result = branch_forward_many(self.__trace, self.__roots, replacements, dose=dose,
                source_identity=source_identity, observe_values=observe_values,
                _prepared_parent=self)
            self.__branches += 1
            return result

    def close(self):
        with self.__lock:
            self.__environment.clear()
            self.__roots.clear()
            self.__native = None
            self.__closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def branch_forward(
    trace: Mapping[str, Any],
    roots: Mapping[str, torch.Tensor],
    value_id: str,
    replacement: torch.Tensor,
    *,
    dose: float = 1.0,
    native_output: Any = None,
    source_identity: Mapping[str, Any] | None = None,
    observe_values: tuple[str, ...] = (),
) -> ForwardInterventionResult:
    """Single-value convenience API for :func:`branch_forward_many`."""
    return branch_forward_many(
        trace,
        roots,
        {value_id: replacement},
        dose=dose,
        native_output=native_output,
        source_identity=source_identity,
        observe_values=observe_values,
    )


# Short alias for callers building small intervention loops.
branch = branch_forward


__all__ = [
    "ForwardInterventionError",
    "ForwardInterventionResult",
    "branch",
    "branch_forward",
    "branch_forward_many",
]

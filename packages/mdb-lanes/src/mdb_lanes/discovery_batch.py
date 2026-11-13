"""Explicit native forward screening, separate from scalar ATen debugger replay.

This adapter batches original B=1 inputs through the actual frozen model.  It
does not resize recorded graphs, publish architectural state, or execute past
component stops.  The controller owns publication and must compare each result's
immutable lane/epoch/branch/revision binding before using it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field

import torch

from .forward_recording import tensor_descriptor


class BatchError(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_batch_attempt(attempt):
    """Verify the sealed row ownership/work record before journal publication."""
    value = copy.deepcopy(attempt)
    fingerprint = value.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or _digest(value) != fingerprint:
        raise BatchError("native batch attempt record changed")
    return fingerprint


@dataclass(frozen=True)
class OutputAct:
    """A row-local edit at an explicitly admitted native module output.

    Occurrences are one-based. Tensor replacements/additions retain a B=1 axis.
    No Python callbacks or selectors inferred from semantic instruction names
    are accepted.
    """
    module: str
    kind: str
    value: object = field(default=None, repr=False, compare=False)
    occurrence: int = 1

    def manifest(self):
        if self.kind in {"replace", "add"}:
            value = tensor_descriptor(self.value)
        else:
            value = self.value
        return {"module": self.module, "kind": self.kind, "value": value,
                "occurrence": self.occurrence}


@dataclass(frozen=True)
class BatchProfile:
    model_identity: str
    program_identity: str
    input_shapes: tuple
    head_shapes: dict
    max_batch_size: int = 4
    allowed_modules: tuple = ()
    backend: str = "cpu"
    dtype: str = "torch.float32"
    cpu_threads: int = 1
    atol: float = 1e-5
    rtol: float = 1e-5
    qualification: str = "unqualified-native-screen"

    def manifest(self):
        return {"schema": "mdb-native-batch-profile-v1", "adapter": "frozen-native-inputs",
                "model_identity": self.model_identity, "program_identity": self.program_identity,
                "frontier_abi": "native-B1-roots-v1", "stage": "full-forward-screen",
                "input_shapes": [list(shape) for shape in self.input_shapes],
                "head_shapes": {key: list(shape) for key, shape in self.head_shapes.items()},
                "max_batch_size": self.max_batch_size, "allowed_modules": list(self.allowed_modules),
                "backend": self.backend, "dtype": self.dtype, "cpu_threads": self.cpu_threads,
                "numerical_contract": {"kind": "bounded-abs-relative", "atol": self.atol,
                                       "rtol": self.rtol}, "qualification": self.qualification,
                "component_debug": "scalar-replay-required", "native_eager_pause": False}


@dataclass(frozen=True)
class BatchRow:
    lane_id: str
    epoch_id: str
    branch_id: str
    expected_revision: int
    roots: tuple = field(repr=False, compare=False)
    input_identity: str = ""
    acts: tuple = ()
    status: str = "ready"
    frontier: str = "native-inputs"
    component_stops: bool = False

    def binding(self):
        return {"lane_id": self.lane_id, "epoch_id": self.epoch_id,
                "branch_id": self.branch_id, "expected_revision": self.expected_revision,
                "input_identity": self.input_identity}


@dataclass(frozen=True)
class ScreenRow:
    binding: dict
    roots_digest: str
    acts: tuple
    outputs: dict = field(repr=False, compare=False)
    attempt_id: str
    physical_row: int

    def matches(self, *, lane_id, epoch_id, branch_id, revision):
        return all(self.binding[key] == value for key, value in
                   (("lane_id", lane_id), ("epoch_id", epoch_id),
                    ("branch_id", branch_id), ("expected_revision", revision)))


@dataclass
class ScreenResult:
    rows: list = field(default_factory=list)
    attempts: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    scalar_required: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class FrozenNativeBatch:
    """One serialized owner of a shared, frozen native model and bounded work.

    Native hooks modify only freshly computed per-row activations. All hooked
    sites must execute the same invocation schedule across a physical batch;
    differing schedules form separate groups. Paused lanes occupy no slot.
    Failed calls remain charged and produce no results for their group.
    """

    def __init__(self, model, profile, *, max_physical_calls=128, max_logical_rows=512):
        if not isinstance(profile, BatchProfile):
            raise BatchError("an explicit native batch profile is required")
        self.profile = copy.deepcopy(profile)
        manifest = self.profile.manifest()
        if not profile.model_identity or not profile.program_identity:
            raise BatchError("model and program identities are required")
        if profile.backend != "cpu" or profile.dtype != "torch.float32":
            raise BatchError("this profile supports only CPU FP32")
        if type(profile.max_batch_size) is not int or profile.max_batch_size < 1:
            raise BatchError("batch size must be a positive integer")
        if type(profile.cpu_threads) is not int or profile.cpu_threads < 1:
            raise BatchError("CPU threads must be a positive integer")
        if not profile.input_shapes or not profile.head_shapes:
            raise BatchError("input geometry and all output heads must be declared")
        for shape in (*profile.input_shapes, *profile.head_shapes.values()):
            if not shape or shape[0] != 1 or any(type(d) is not int or d < 1 for d in shape):
                raise BatchError("profile geometry requires dense B=1 shapes")
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
               for v in (profile.atol, profile.rtol)):
            raise BatchError("numerical tolerances must be finite nonnegative numbers")
        if any(type(v) is not int or v < 1 for v in (max_physical_calls, max_logical_rows)):
            raise BatchError("finite positive physical and logical budgets are required")
        self.model = model
        self.profile_id = _digest(manifest)
        self.max_physical_calls, self.max_logical_rows = max_physical_calls, max_logical_rows
        self.work = {"physical_model_calls": 0, "physical_batches": 0, "logical_forward_rows": 0,
                     "failed_physical_calls": 0, "padded_rows": 0}
        self.attempts = []
        self._lock = threading.Lock()
        self._modules = dict(model.named_modules())
        if any(name not in self._modules for name in profile.allowed_modules):
            raise BatchError("admitted module output site does not exist")
        self._state = self._model_state()
        self._check_model()

    def _model_state(self):
        return {key: (id(value), value._version, value.data_ptr(), tuple(value.shape),
                      str(value.dtype), str(value.device))
                for key, value in (*self.model.named_parameters(), *self.model.named_buffers())}

    def _check_model(self):
        if _digest(self.profile.manifest()) != self.profile_id:
            raise BatchError("admitted native batch profile changed")
        if any(module.training for module in self.model.modules()):
            raise BatchError("native batch model must be in evaluation mode")
        if any(value.requires_grad for value in self.model.parameters()):
            raise BatchError("native batch parameters must already be frozen")
        if any(value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32)
               for value in (*self.model.parameters(), *self.model.buffers())):
            raise BatchError("model tensors violate declared CPU FP32 profile")
        if self._model_state() != self._state:
            raise BatchError("shared frozen model tensor identity/version changed")
        if torch.get_num_threads() != self.profile.cpu_threads:
            raise BatchError("CPU thread count differs from declared profile")

    def _prepare(self, row):
        if not isinstance(row, BatchRow):
            raise BatchError("screen requests must be BatchRow objects")
        if any(not isinstance(v, str) or not v for v in (row.lane_id, row.epoch_id, row.branch_id)):
            raise BatchError("lane, epoch and branch identities are required")
        if type(row.expected_revision) is not int or row.expected_revision < 0:
            raise BatchError("expected revision must be a nonnegative integer")
        if len(row.roots) != len(self.profile.input_shapes):
            raise BatchError("root count differs from admitted profile")
        roots = []
        for value, shape in zip(row.roots, self.profile.input_shapes):
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                    or value.device.type != "cpu" or str(value.dtype) != self.profile.dtype
                    or value.layout != torch.strided or not bool(torch.isfinite(value).all())):
                raise BatchError("root violates admitted geometry/device/dtype/finite profile")
            roots.append(value.detach().clone())
        acts, seen = [], set()
        for act in row.acts:
            if not isinstance(act, OutputAct) or act.module not in self.profile.allowed_modules:
                raise BatchError("Act requires an explicitly admitted module output")
            if type(act.occurrence) is not int or act.occurrence < 1:
                raise BatchError("Act occurrence must be positive")
            site = (act.module, act.occurrence)
            if site in seen:
                raise BatchError("only one Act per output occurrence and row is supported")
            seen.add(site)
            if act.kind in {"replace", "add"}:
                value = act.value
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or str(value.dtype) != self.profile.dtype or value.ndim < 1
                        or value.shape[0] != 1 or not bool(torch.isfinite(value).all())):
                    raise BatchError("Act payload requires a finite B=1 CPU FP32 tensor")
                value = value.detach().clone()
            elif act.kind == "scale":
                value = act.value
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise BatchError("scale requires a finite numeric factor")
            elif act.kind == "zeros_like" and act.value is None:
                value = None
            else:
                raise BatchError("unsupported native output Act")
            acts.append(OutputAct(act.module, act.kind, value, act.occurrence))
        descriptors = [tensor_descriptor(value) for value in roots]
        root_digest = _digest(descriptors)
        binding = row.binding()
        if not binding["input_identity"]:
            binding["input_identity"] = root_digest
        return {"binding": binding, "roots": tuple(roots), "roots_digest": root_digest,
                "acts": tuple(acts)}

    def screen(self, rows, *, batch_size=None):
        """Compute only eligible native-input screens; never mutate lane state.

        A size of one is an explicit native scalar fallback. Incompatible hook
        schedules naturally become separate groups, possibly scalar groups.
        Component stepping/breakpoints return ``scalar_required`` without work.
        """
        size = self.profile.max_batch_size if batch_size is None else batch_size
        if type(size) is not int or not 1 <= size <= self.profile.max_batch_size:
            raise BatchError("requested batch size is outside the admitted profile")
        if not self._lock.acquire(blocking=False):
            raise BatchError("native batch owner already has an in-flight call")
        try:
            self._check_model()
            result, groups, owners = ScreenResult(), {}, set()
            for row in rows:
                if not isinstance(row, BatchRow):
                    raise BatchError("screen requests must be BatchRow objects")
                if row.status != "ready":
                    result.skipped.append({**row.binding(), "reason": row.status})
                    continue
                if row.frontier != "native-inputs" or row.component_stops:
                    result.scalar_required.append({**row.binding(),
                        "reason": "component-frontier-or-stop-requires-scalar-replay"})
                    continue
                owner = (row.lane_id, row.epoch_id, row.branch_id)
                if owner in owners:
                    raise BatchError("duplicate lane/epoch/branch request")
                owners.add(owner)
                prepared = self._prepare(row)
                # The native program and all tensor geometry are profile-bound.
                # Hook invocation schedules are a further compatibility class.
                key = tuple(sorted((act.module, act.occurrence) for act in prepared["acts"]))
                groups.setdefault(key, []).append(prepared)
            for group in groups.values():
                for start in range(0, len(group), size):
                    batch = group[start:start + size]
                    if (self.work["physical_model_calls"] >= self.max_physical_calls
                            or self.work["logical_forward_rows"] + len(batch) > self.max_logical_rows):
                        result.errors.append({"kind": "budget", "rows": [r["binding"] for r in batch]})
                        continue
                    self._execute(batch, result)
            return result
        finally:
            self._lock.release()

    def _execute(self, rows, result):
        self._check_model()
        self.work["physical_model_calls"] += 1
        self.work["physical_batches"] += int(len(rows) > 1)
        self.work["logical_forward_rows"] += len(rows)
        attempt_id = f"native_batch_{self.work['physical_model_calls']:06d}"
        attempt = {"schema": "mdb-native-batch-attempt-v1", "attempt_id": attempt_id,
                   "profile_id": self.profile_id, "physical_batch_size": len(rows),
                   "mode": "native-physical-batch" if len(rows) > 1 else "native-scalar",
                   "rows": [{"physical_row": index, **r["binding"],
                             "roots_digest": r["roots_digest"],
                             "acts": [act.manifest() for act in r["acts"]]}
                            for index, r in enumerate(rows)], "padded_rows": 0,
                   "mechanics_status": "attempted"}
        self.attempts.append(attempt)
        result.attempts.append(attempt)
        handles, counts, applied = [], {}, set()
        started = time.perf_counter()
        try:
            def hook_for(name):
                def hook(module, args, output):
                    counts[name] = counts.get(name, 0) + 1
                    occurrence = counts[name]
                    edits = {index: act for index, r in enumerate(rows) for act in r["acts"]
                             if act.module == name and act.occurrence == occurrence}
                    if not edits:
                        return output
                    if (not isinstance(output, torch.Tensor) or output.ndim < 1
                            or output.shape[0] != len(rows)):
                        raise BatchError("native module output has no aligned batch axis")
                    pieces = []
                    for index in range(len(rows)):
                        value, act = output[index:index + 1], edits.get(index)
                        if act is not None:
                            if act.kind in {"replace", "add"} and (
                                    act.value.shape != value.shape or act.value.dtype != value.dtype
                                    or act.value.device != value.device):
                                raise BatchError("Act tensor differs from actual module-output geometry")
                            if act.kind == "zeros_like":
                                value = torch.zeros_like(value)
                            elif act.kind == "scale":
                                value = value * act.value
                            elif act.kind == "add":
                                value = value + act.value
                            else:
                                value = act.value.clone()
                            if not bool(torch.isfinite(value).all()):
                                raise BatchError("Act produced nonfinite activation")
                            applied.add((index, name, occurrence))
                        pieces.append(value)
                    return torch.cat(pieces, dim=0)
                return hook

            sites = sorted({act.module for r in rows for act in r["acts"]})
            for name in sites:
                handles.append(self._modules[name].register_forward_hook(hook_for(name)))
            roots = tuple(torch.cat([r["roots"][index] for r in rows], dim=0)
                          for index in range(len(self.profile.input_shapes)))
            attempt["input_tensor_bytes"] = sum(v.numel() * v.element_size() for v in roots)
            with torch.no_grad():
                outputs = self.model(*roots)
            self._check_model()
            expected_acts = {(index, act.module, act.occurrence)
                             for index, r in enumerate(rows) for act in r["acts"]}
            if applied != expected_acts:
                raise BatchError("native forward did not execute every declared output Act")
            if not isinstance(outputs, dict) or set(outputs) != set(self.profile.head_shapes):
                raise BatchError("native output head set differs from admitted profile")
            for name, shape in self.profile.head_shapes.items():
                value = outputs[name]
                expected = (len(rows), *shape[1:])
                if (not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
                        or value.device.type != "cpu" or str(value.dtype) != self.profile.dtype
                        or not bool(torch.isfinite(value).all())):
                    raise BatchError(f"native head violates geometry/device/dtype/finite contract: {name}")
            attempt["output_tensor_bytes"] = sum(v.numel() * v.element_size() for v in outputs.values())
            # Validate the complete result before publishing any row from this call.
            completed = [ScreenRow(copy.deepcopy(r["binding"]), r["roots_digest"],
                         tuple(act.manifest() for act in r["acts"]),
                         {key: value[index:index + 1].detach().clone() for key, value in outputs.items()},
                         attempt_id, index) for index, r in enumerate(rows)]
            result.rows.extend(completed)
            attempt["mechanics_status"] = "valid"
        except Exception as error:
            self.work["failed_physical_calls"] += 1
            attempt["mechanics_status"] = "errored"
            attempt["error"] = f"{type(error).__name__}: {error}"
            result.errors.append({"kind": "native_execution", "attempt_id": attempt_id,
                                  "error": attempt["error"], "rows": attempt["rows"]})
        finally:
            for handle in handles:
                handle.remove()
            attempt["elapsed_seconds"] = time.perf_counter() - started
            attempt["fingerprint"] = _digest(attempt)


def compare_screen_rows(reference, candidate, profile):
    """All-head numerical observations; no scientific presence/absence verdict."""
    if reference.binding != candidate.binding or reference.roots_digest != candidate.roots_digest:
        raise BatchError("cannot compare different lane roots or ownership")
    if reference.acts != candidate.acts:
        raise BatchError("cannot compare different intervention specifications")
    if set(reference.outputs) != set(candidate.outputs) or set(reference.outputs) != set(profile.head_shapes):
        raise BatchError("comparison must retain every declared output head")
    report = {}
    for key in profile.head_shapes:
        left, right = reference.outputs[key], candidate.outputs[key]
        if tuple(left.shape) != tuple(right.shape):
            raise BatchError("comparison geometry differs")
        report[key] = {"exact": torch.equal(left, right),
                       "max_abs": float((left - right).abs().max()),
                       "within_contract": bool(torch.allclose(left, right,
                           atol=profile.atol, rtol=profile.rtol))}
    return report


__all__ = ["BatchError", "BatchProfile", "BatchRow", "OutputAct", "ScreenRow",
           "ScreenResult", "FrozenNativeBatch", "compare_screen_rows", "verify_batch_attempt"]

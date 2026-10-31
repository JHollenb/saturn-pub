"""A bounded, trusted-Python capture/predict/seal/execute transaction.

Prediction callbacks receive owned immutable copies, never a model handle. Model
module-call hooks enforce the declared exclusion window and latch attempted
violations even when a callback catches the exception. This is instrumentation,
not an OS sandbox: direct ``.forward`` calls, other model objects, arbitrary
Python, and external side effects are outside its isolation contract.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

SCHEMA = "mdb-forecast-transaction-v1"
DEFAULT_ARRAY_STORAGE = "bundle-v1"
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class ForecastTransactionError(RuntimeError):
    """A refused phase, identity, native entry, or evidence publication."""


@dataclass(frozen=True)
class ForecastPayload:
    value: Any
    dtype: str | None = "float32"
    provenance: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionContext:
    frontiers: Mapping
    weights: Mapping
    predictions: Mapping
    declarations: Mapping


@dataclass(frozen=True)
class ExecutionContext:
    model: Any
    frontiers: Mapping
    predictions: Mapping
    payloads: Mapping
    declarations: Mapping


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in (".", ".."):
        raise ForecastTransactionError("record names must be safe explicit identifiers")
    return value


def _json(value):
    return json.loads(_canonical(value))


def _readonly(value):
    """Immutable byte-backed arrays cannot be made writable via setflags()."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ForecastTransactionError("only finite real numeric arrays are admitted")
        return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _readonly(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_readonly(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ForecastTransactionError(f"unsupported callback value: {type(value).__name__}")


def _mkdir_durable(path, *, exist_ok=True):
    """Persist every new directory entry before publishing files beneath it."""
    path = Path(path)
    if path.exists():
        if not exist_ok:
            raise FileExistsError(str(path))
        if path.is_symlink() or not path.is_dir():
            raise ForecastTransactionError("artifact directory is not a plain directory")
        return
    if not path.parent.exists():
        _mkdir_durable(path.parent)
    path.mkdir(mode=0o700, exist_ok=exist_ok)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ForecastTransaction:
    """One process-local transaction with immutable published evidence.

    ``allowed_prefix`` is ``{'stop_before': PATH, 'observe': [PATH, ...]}``.
    Without prefix capture, explicitly declare ``frontier={'kind': 'empty'}``,
    ``'architectural'`` or ``'raw-input'`` and publish inputs with ``frontier()``.
    No native execution is used to prepare these independent forecasts.

    Scripts contain already-bound trusted Python callables, not import names or
    serialized executable code. Wall limits are cooperative checks at operations
    and instrumented module entries; callbacks are not forcibly preempted.

    Numerical publications use immutable bundles by default. Select
    ``array_storage='files-v1'`` for the legacy individual-NPY layout.
    """

    DEFAULT_BUDGETS = {"max_actions": 512, "max_captures": 64, "max_predictions": 128,
                       "max_executions": 128, "max_native_calls": 256,
                       "max_capture_operations": 8192, "max_bytes": 256 * 1024 * 1024,
                       "max_files": 4096, "max_wall_ms": 120000}

    def __init__(self, root, *, model=None, model_identity, input_identity, source_identity,
                 allowed_prefix=None, denied_modules=(), frontier=None, weights=None,
                 budgets=None, array_storage=DEFAULT_ARRAY_STORAGE):
        self.root = Path(root)
        _mkdir_durable(self.root, exist_ok=False)
        self.model = model
        self.phase = "open"
        self.budgets = {**self.DEFAULT_BUDGETS, **(budgets or {})}
        self.work = {"actions": 0, "captures": 0, "predictions": 0, "executions": 0,
                     "native_calls": 0, "denied_attempts": 0, "retained_bytes": 0}
        self._started = time.monotonic()
        self._owner = threading.get_ident()
        self._op_lock, self._guard_lock, self._event_lock = (threading.RLock() for _ in range(3))
        self._busy, self._native_window = False, False
        self._hooks, self._violations, self._events = [], [], []
        self._frontiers, self._predictions, self._observations = {}, {}, {}
        self._event_head = None
        self._seal_digest = None
        self._failure = None
        self._weights = {}
        self._model_signature = None
        self.array_storage = array_storage
        self._array_descriptors, self._pending_arrays = {}, {}
        self._pending_array_bytes = 0
        self.allowed_prefix = _json(allowed_prefix) if allowed_prefix is not None else None
        self.denied_modules = tuple(denied_modules)
        self.declarations = {}
        try:
            if array_storage not in ("files-v1", "bundle-v1"):
                raise ForecastTransactionError("unsupported array storage format")
            for key, value in self.budgets.items():
                if key not in self.DEFAULT_BUDGETS or type(value) is not int or value < 1:
                    raise ForecastTransactionError(f"invalid positive finite budget: {key}")
            if not model_identity or not input_identity or not source_identity:
                raise ForecastTransactionError("model, input and source identities must be explicitly declared")
            if self.allowed_prefix is None:
                if not isinstance(frontier, Mapping) or frontier.get("kind") not in (
                        "empty", "architectural", "raw-input"):
                    raise ForecastTransactionError("capture-optional execution requires an explicit frontier kind")
            elif (model is None or not isinstance(self.allowed_prefix.get("stop_before"), str)
                  or not self.allowed_prefix["stop_before"]):
                raise ForecastTransactionError("prefix capture requires a bound model and exact stop_before")
            if model is not None:
                self._validate_model()
                self._model_signature = self._signature()
                self._weights = _readonly(dict(model.state_dict()))
                if weights is not None:
                    raise ForecastTransactionError("model-bound weights are read from its actual state_dict")
                modules = dict(model.named_modules())
                if any(path not in modules for path in self.denied_modules):
                    raise ForecastTransactionError("unknown denied module path")
                if self.allowed_prefix:
                    paths = (self.allowed_prefix["stop_before"], *self.allowed_prefix.get("observe", ()))
                    if any(path not in modules for path in paths):
                        raise ForecastTransactionError("unknown prefix boundary or observation module")
                for path, module in modules.items():
                    self._hooks.append(module.register_forward_pre_hook(self._guard(path), prepend=True))
            else:
                if self.denied_modules:
                    raise ForecastTransactionError("denied module guards require a bound model")
                self._weights = _readonly(weights or {})
            weight_manifest = self._encode(self._weights)
            self._flush_arrays()
            self.declarations = _json({"schema": SCHEMA, "model_identity": model_identity,
                "input_identity": input_identity, "source_identity": source_identity,
                "allowed_prefix": self.allowed_prefix, "frontier": frontier,
                "denied_modules": list(self.denied_modules), "budgets": self.budgets,
                "execution_scope": "CPU FP32 eval; genuine prefix wrapper or explicitly declared frontier",
                "callback_isolation": "trusted Python; instrumented bound-model module calls only; not an OS sandbox",
                "guard_limits": ["direct .forward calls may bypass hooks", "other model objects are not instrumented",
                    "arbitrary Python and external side effects are not sandboxed",
                    "callbacks must join background work before returning; wall checks are cooperative"],
                "actual_weight_manifest": weight_manifest})
            if array_storage != "files-v1":
                self.declarations["array_storage"] = array_storage
            self._write_json("declarations.json", self.declarations)
            self._event("opened")
        except BaseException as error:
            self._fail(error)
            raise

    def _validate_model(self):
        import torch

        if any(module.training for module in self.model.modules()):
            raise ForecastTransactionError("bound model must already be in evaluation mode")
        for value in (*self.model.parameters(), *self.model.buffers()):
            if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32):
                raise ForecastTransactionError("initial transaction scope is CPU FP32 model state")
            if not bool(torch.isfinite(value).all()):
                raise ForecastTransactionError("model state must be finite")

    def _signature(self):
        return {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "sha256": _sha(value.detach().cpu().numpy().tobytes(order="C"))}
                for name, value in self.model.state_dict().items()}

    def _check(self):
        if (time.monotonic() - self._started) * 1000 >= self.budgets["max_wall_ms"]:
            raise ForecastTransactionError("transaction wall budget exhausted")
        if self._violations:
            raise ForecastTransactionError("latched native-entry or transaction violation")
        if self.model is not None:
            self._validate_model()
            if self._signature() != self._model_signature:
                raise ForecastTransactionError("bound model state changed during transaction")

    def _violate(self, kind, **detail):
        with self._guard_lock:
            self.work["denied_attempts"] += 1
            if len(self._violations) < 16:
                row = {"kind": kind, "thread": threading.get_ident(), **detail}
                self._violations.append(row)
                self._event("violation", detail=row)
        raise ForecastTransactionError(kind)

    def _guard(self, path):
        def guard(module, args):
            owner = threading.get_ident() == self._owner
            with self._guard_lock:
                if self._violations:
                    self._violate("native entry after latched violation", module=path or "<root>")
                allowed = (owner and self._native_window and self.phase in ("capture", "execute")
                           and not (self.phase == "capture" and path in self.denied_modules))
                if not allowed:
                    self._violate("native module entry forbidden", module=path or "<root>", phase=self.phase)
                if (time.monotonic() - self._started) * 1000 >= self.budgets["max_wall_ms"]:
                    self._violate("native entry after wall budget", module=path or "<root>")
                if path == "":
                    if self.work["native_calls"] >= self.budgets["max_native_calls"]:
                        self._violate("native-call budget exhausted")
                    self.work["native_calls"] += 1
        return guard

    def _publish(self, relative, raw, *, deduplicate=False, emergency=False):
        path = self.root / relative
        current = self.root
        for component in Path(relative).parts:
            current = current / component
            if current.is_symlink():
                raise ForecastTransactionError("artifact path contains a symbolic link")
        if path.exists():
            if deduplicate and path.read_bytes() == raw:
                return
            raise ForecastTransactionError("immutable artifact already exists: " + relative)
        if not emergency and self.work["retained_bytes"] + len(raw) > self.budgets["max_bytes"]:
            raise ForecastTransactionError("retained-byte budget exhausted")
        _mkdir_durable(path.parent)
        temporary = path.parent / (".publish-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.work["retained_bytes"] += len(raw)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_json(self, relative, value, *, emergency=False):
        self._flush_arrays()
        self._publish(relative, _canonical(value), emergency=emergency)

    def _flush_arrays(self):
        """Publish one closed pack before any metadata can reference it."""
        if not self._pending_arrays:
            return
        from mdb.forecast_storage import BUNDLE_MAGIC

        pieces, offset = [BUNDLE_MAGIC], len(BUNDLE_MAGIC)
        for identifier, (raw, descriptor) in sorted(self._pending_arrays.items()):
            descriptor["offset"], descriptor["nbytes"] = offset, len(raw)
            offset += len(raw)
            pieces.append(raw)
        bundle = b"".join(pieces)
        identifier = _sha(bundle)
        relative = "arrays/" + identifier + ".npyb"
        self._publish(relative, bundle, deduplicate=True)
        for _, descriptor in self._pending_arrays.values():
            descriptor["path"] = relative
            descriptor["bundle_sha256"] = identifier
        self._pending_arrays.clear()
        self._pending_array_bytes = 0

    def _event(self, kind, *, detail=None):
        from saturn_pub import Receipt
        from saturn_pub.store import LocalStore

        with self._event_lock:
            event = {"schema": "mdb-forecast-phase-event-v1", "sequence": len(self._events),
                     "phase": self.phase, "kind": kind, "previous": self._event_head,
                     "elapsed_ms": (time.monotonic() - self._started) * 1000,
                     "detail": _json(detail or {}), "work": dict(self.work)}
            receipt = Receipt.make(operation="mdb.forecast." + kind, status="observed", phase_event=event)
            _mkdir_durable(self.root / "custody" / "receipts")
            event["receipt_id"] = LocalStore(self.root / "custody").receipt(receipt)
            raw = _canonical(event)
            self._publish(f"events/{len(self._events):06d}.json", raw, emergency=True)
            self._event_head = _sha(raw)
            self._events.append(event)
            return event

    def _cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._native_window = False

    def _fail(self, error):
        self.phase = "failed"
        self._failure = {"error_type": type(error).__name__, "error": str(error)}
        try:
            self._event("failed", detail={"error_type": type(error).__name__, "error": str(error),
                                          "violations": list(self._violations)})
        finally:
            self._cleanup()

    @contextmanager
    def _operation(self, name, phases):
        if threading.get_ident() != self._owner:
            self._violate("background transaction operation", operation=name)
        with self._op_lock:
            if self._busy:
                self._violate("background or reentrant transaction operation", operation=name)
            if self.phase not in phases:
                error = ForecastTransactionError(f"{name} unavailable in phase {self.phase}")
                self._fail(error)
                raise error
            self._busy = True
            try:
                self._check()
                if self.work["actions"] >= self.budgets["max_actions"]:
                    raise ForecastTransactionError("action budget exhausted")
                self.work["actions"] += 1
                yield
                self._check()
            except BaseException as error:
                self._fail(error)
                raise
            finally:
                self._busy = False
                self._native_window = False

    def _encode(self, value):
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            array = _readonly(value)
            content = array.tobytes(order="C")
            content_sha = _sha(content)
            key = (array.dtype.str, array.shape, content_sha)
            # Snapshot and hash every supplied array. Object-identity caching
            # would miss a callback mutating and reusing the same ndarray.
            if key in self._array_descriptors:
                return self._array_descriptors[key]
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            raw = buffer.getvalue()
            identifier = _sha(raw)
            relative = "arrays/" + identifier + ".npy"
            descriptor = {"kind": "array", "path": relative, "sha256": identifier,
                          "shape": list(array.shape), "dtype": str(array.dtype),
                          "content_sha256": content_sha}
            if self.array_storage == "bundle-v1":
                from mdb.forecast_storage import BUNDLE_MAGIC

                if self.work["retained_bytes"] + len(BUNDLE_MAGIC) + self._pending_array_bytes + len(raw) > self.budgets["max_bytes"]:
                    raise ForecastTransactionError("retained-byte budget exhausted")
                descriptor.update(storage="bundle-v1")
                self._pending_arrays[identifier] = (raw, descriptor)
                self._pending_array_bytes += len(raw)
            else:
                self._publish(relative, raw, deduplicate=True)
            self._array_descriptors[key] = descriptor
            return descriptor
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ForecastTransactionError("stored mappings require string keys")
            return {"kind": "mapping", "items": {key: self._encode(item) for key, item in value.items()}}
        if isinstance(value, (list, tuple)):
            return {"kind": "tuple" if isinstance(value, tuple) else "list",
                    "items": [self._encode(item) for item in value]}
        if isinstance(value, str):
            # Framework metadata can use a str subclass, such as TorchVersion.
            # Store its JSON string value, without invoking arbitrary reprs.
            return {"kind": "scalar", "value": str(value)}
        if value is None or type(value) in (bool, int) or (type(value) is float and math.isfinite(value)):
            return {"kind": "scalar", "value": value}
        raise ForecastTransactionError(f"unsupported stored value: {type(value).__name__}")

    def _decode(self, encoded):
        return load_forecast_tree(self.root, encoded)

    @staticmethod
    def _select(value, path):
        for component in path:
            value = value[component]
        return value

    def frontier(self, name, value, *, input_identity=None):
        """Publish explicitly supplied raw/architectural roots with no native call."""
        with self._operation("frontier", ("open",)):
            name = _name(name)
            if name in self._frontiers:
                raise ForecastTransactionError("frontier name already exists")
            if self.allowed_prefix is not None:
                raise ForecastTransactionError("supplied frontier requires the capture-optional declaration")
            record = {"schema": "mdb-forecast-frontier-v1", "name": name,
                      "origin": "explicitly-supplied-frontier", "input_identity": input_identity,
                      "value": self._encode(value)}
            self._write_json("frontiers/" + name + ".json", record)
            self._frontiers[name] = record
            self._event("frontier_published", detail={"name": name, "native_calls": 0})
            return _json(record)

    def capture(self, name, args, *, kwargs=None, input_identity=None):
        """Capture the declared genuine prefix; never complete a hidden reference."""
        with self._operation("capture", ("open",)):
            from .forward_prefix import record_prefix

            name = _name(name)
            if self.allowed_prefix is None or name in self._frontiers:
                raise ForecastTransactionError("capture requires a declared prefix and a new frontier name")
            if self.work["captures"] >= self.budgets["max_captures"]:
                raise ForecastTransactionError("capture budget exhausted")
            roots = _readonly({"args": args, "kwargs": kwargs or {}})
            # Capture uses independently owned CPU roots. Prediction callbacks
            # receive immutable NumPy copies of the resulting actual frontier.
            def native(value):
                import numpy as np
                import torch
                if isinstance(value, np.ndarray):
                    if value.dtype.kind == "f" and value.dtype != np.float32:
                        raise ForecastTransactionError("native roots must be CPU FP32")
                    return torch.from_numpy(value.copy())
                if isinstance(value, Mapping):
                    return {key: native(item) for key, item in value.items()}
                if isinstance(value, tuple):
                    return tuple(native(item) for item in value)
                return value
            actual = native(roots)
            self.phase, self._native_window = "capture", True
            self.work["captures"] += 1
            capture = record_prefix(self.model, actual["args"], kwargs=actual["kwargs"],
                stop_before=self.allowed_prefix["stop_before"], observe=tuple(self.allowed_prefix.get("observe", ())),
                model_identity=self.declarations["model_identity"],
                input_identity=input_identity or self.declarations["input_identity"],
                execution={"backend": "cpu", "dtype": "float32", "mode": "genuine-prefix"},
                max_operations=self.budgets["max_capture_operations"])
            self._native_window, self.phase = False, "open"
            self._check()
            record = {"schema": "mdb-forecast-frontier-v1", "name": name, "origin": "genuine-native-prefix",
                      "input_identity": input_identity or self.declarations["input_identity"],
                      "roots": self._encode(roots), "value": self._encode(capture.output),
                      "trace_fingerprint": capture.trace["fingerprint"],
                      "prefix_contract": capture.trace["metadata"]["prefix_contract"]}
            self._write_json("captures/" + name + "-trace.json", capture.trace)
            self._write_json("frontiers/" + name + ".json", record)
            self._frontiers[name] = record
            self._event("prefix_captured", detail={"name": name, "trace_fingerprint": capture.trace["fingerprint"],
                                                   "native_suffix_computed": False})
            return _json(record)

    def predict(self, name, callback, *, captures=None, payload_casts=None, payload_provenance=None):
        with self._operation("predict", ("open",)):
            import numpy as np

            name = _name(name)
            if name in self._predictions or not callable(callback):
                raise ForecastTransactionError("predict requires a new name and trusted bound callback")
            if self.work["predictions"] >= self.budgets["max_predictions"]:
                raise ForecastTransactionError("prediction budget exhausted")
            selected = list(self._frontiers) if captures is None else list(captures)
            if len(selected) != len(set(selected)) or any(item not in self._frontiers for item in selected):
                raise ForecastTransactionError("prediction frontier selection is unknown or duplicated")
            self.phase = "predict"
            context = PredictionContext(
                MappingProxyType({key: self._decode(self._frontiers[key]["value"]) for key in selected}),
                _readonly(self._weights), MappingProxyType({key: self.prediction(key) for key in self._predictions}),
                _readonly(self.declarations))
            self.work["predictions"] += 1
            result = callback(context)
            self._check()
            if not isinstance(result, Mapping) or "predictions" not in result or set(result) - {"predictions", "payloads"}:
                raise ForecastTransactionError("predict callback returns predictions and optional named payloads")
            payloads = result.get("payloads", {})
            if not isinstance(payloads, Mapping):
                raise ForecastTransactionError("payloads must be a named mapping")
            casts, provenance = dict(payload_casts or {}), dict(payload_provenance or {})
            if (set(casts) | set(provenance)) - set(payloads):
                raise ForecastTransactionError("cast/provenance names must refer to returned payloads")
            encoded_payloads = {}
            for key, supplied in payloads.items():
                _name(key)
                payload = supplied if isinstance(supplied, ForecastPayload) else ForecastPayload(supplied)
                original = _readonly(payload.value)
                if not isinstance(original, np.ndarray):
                    raise ForecastTransactionError("intervention payloads must be numeric arrays")
                dtype = casts.get(key, payload.dtype)
                if dtype not in (None, "float32", "float64", "int64", "uint8", "bool"):
                    raise ForecastTransactionError("unsupported explicit payload cast")
                staged = np.asarray(original, dtype=dtype) if dtype else original
                encoded_payloads[key] = {"value": self._encode(staged), "original": self._encode(original),
                    "cast": {"from": str(original.dtype), "to": str(staged.dtype), "policy": "explicit-astype"},
                    "provenance": _json({**dict(payload.provenance), **dict(provenance.get(key, {})),
                                          "forecast": name, "frontiers": selected})}
            record = {"schema": "mdb-sealed-forecast-v1", "name": name, "frontiers": selected,
                      "predictions": self._encode(result["predictions"]), "payloads": encoded_payloads,
                      "callback_scope": "trusted Python bound callable; no OS isolation"}
            self._write_json("forecasts/" + name + ".json", record)
            self._predictions[name] = record
            self.phase = "open"
            self._event("forecast_persisted", detail={"name": name, "frontiers": selected})
            return _json(record)

    def prediction(self, name, *, path=()):
        return self._select(self._decode(self._predictions[name]["predictions"]), path)

    def payload(self, forecast, name, *, as_tensor=False):
        value = self._decode(self._predictions[forecast]["payloads"][name]["value"])
        if as_tensor:
            import torch
            return torch.from_numpy(value.copy())
        return value

    def observation(self, name, *, path=()):
        return self._select(self._decode(self._observations[name]["value"]), path)

    def _files(self):
        result, total = {}, 0
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise ForecastTransactionError("transaction artifact contains a symbolic link")
            if path.is_file():
                raw = path.read_bytes()
                result[str(path.relative_to(self.root))] = {"sha256": _sha(raw), "bytes": len(raw)}
                total += len(raw)
        if len(result) > self.budgets["max_files"] or total > self.budgets["max_bytes"]:
            raise ForecastTransactionError("retained artifact geometry exceeds transaction budget")
        return result

    def seal(self):
        with self._operation("seal", ("open",)):
            if not self._predictions:
                raise ForecastTransactionError("a global seal requires at least one persisted forecast")
            self.phase = "sealing"
            self._check()
            _verify_numerical_records(self.root, self.declarations,
                                      list(self._frontiers.values()), list(self._predictions.values()))
            manifest = {"schema": "mdb-global-forecast-seal-v1", "transaction_schema": SCHEMA,
                        "declarations": self.declarations, "files": self._files(),
                        "frontiers": list(self._frontiers), "forecasts": list(self._predictions),
                        "pre_release_work": dict(self.work), "violations": list(self._violations),
                        "event_head_before_seal": self._event_head}
            self._write_json("seal.json", manifest)
            self._seal_digest = _sha((self.root / "seal.json").read_bytes())
            self.phase = "sealed"
            self._event("globally_sealed", detail={"seal_sha256": self._seal_digest})
            return {"seal_sha256": self._seal_digest, "path": str(self.root / "seal.json"),
                    "forecasts": list(self._predictions)}

    def verify(self):
        if self._seal_digest is None:
            raise ForecastTransactionError("transaction has no global seal")
        path = self.root / "seal.json"
        if path.is_symlink():
            raise ForecastTransactionError("seal is a symbolic link")
        raw = path.read_bytes()
        if _sha(raw) != self._seal_digest:
            raise ForecastTransactionError("global forecast seal changed")
        manifest = json.loads(raw)
        for relative, reference in manifest["files"].items():
            path = self.root / relative
            if path.is_symlink():
                raise ForecastTransactionError("sealed artifact is a symbolic link")
            actual = path.read_bytes()
            if len(actual) != reference["bytes"] or _sha(actual) != reference["sha256"]:
                raise ForecastTransactionError("sealed artifact changed: " + relative)
        self._check()
        return manifest

    def _release(self):
        self.verify()
        if self.phase == "sealed":
            self.phase = "released"
            self._event("native_execution_released", detail={"seal_sha256": self._seal_digest})

    def release(self):
        with self._operation("release", ("sealed", "released")):
            self._release()
            return {"phase": self.phase, "seal_sha256": self._seal_digest}

    def execute(self, name, callback, *, analysis=None):
        with self._operation("execute", ("sealed", "released")):
            name = _name(name)
            if name in self._observations or not callable(callback):
                raise ForecastTransactionError("execute requires a new name and trusted bound callback")
            if self.work["executions"] >= self.budgets["max_executions"]:
                raise ForecastTransactionError("execution budget exhausted")
            self._release()
            context = ExecutionContext(self.model,
                MappingProxyType({key: self._decode(value["value"]) for key, value in self._frontiers.items()}),
                MappingProxyType({key: self.prediction(key) for key in self._predictions}),
                MappingProxyType({key: MappingProxyType({name: self.payload(key, name)
                    for name in value["payloads"]}) for key, value in self._predictions.items()}),
                _readonly(self.declarations))
            self.phase, self._native_window = "execute", True
            self.work["executions"] += 1
            result = callback(context)
            self._native_window = False
            record = {"schema": "mdb-forecast-observation-v1", "name": name,
                      "seal_sha256": self._seal_digest, "value": self._encode(result)}
            self._write_json("observations/" + name + ".json", record)
            self._observations[name] = record
            self._event("observation_persisted", detail={"name": name, "seal_sha256": self._seal_digest})
            self._check()
            if analysis is not None:
                if not callable(analysis):
                    raise ForecastTransactionError("analysis must be a trusted bound callback")
                compared = analysis(self.observation(name))
                self._write_json("analysis/" + name + ".json", {"value": self._encode(compared)})
                self._event("analysis_persisted", detail={"name": name})
            self.phase = "released"
            return _json(record)

    def run_script(self, actions):
        """Execute a finite list of Python-bound actions through these same methods."""
        methods = {"frontier": self.frontier, "capture": self.capture, "predict": self.predict,
                   "seal": self.seal, "release": self.release, "execute": self.execute}
        if not isinstance(actions, (list, tuple)) or len(actions) > self.budgets["max_actions"]:
            raise ForecastTransactionError("script must be an explicitly bounded list of actions")
        prepared = []
        for action in actions:
            if not isinstance(action, Mapping) or action.get("op") not in methods:
                raise ForecastTransactionError("unknown scripted transaction action")
            arguments = dict(action)
            operation = arguments.pop("op")
            # Bind signatures before the first action; serialized import/eval
            # strings never resolve to executable callbacks.
            import inspect
            try:
                inspect.signature(methods[operation]).bind(**arguments)
            except TypeError as error:
                raise ForecastTransactionError("script action arguments do not match the typed operation") from error
            if operation in ("predict", "execute") and not callable(arguments.get("callback")):
                raise ForecastTransactionError("script callbacks must be trusted Python callables")
            prepared.append((methods[operation], arguments))
        return [method(**arguments) for method, arguments in prepared]

    def report(self):
        return {"schema": SCHEMA, "phase": self.phase, "seal_sha256": self._seal_digest,
                "failure": _json(self._failure),
                "work": dict(self.work), "violations": _json(self._violations),
                "frontiers": list(self._frontiers), "forecasts": list(self._predictions),
                "observations": list(self._observations), "event_head": self._event_head,
                "events": _json(self._events)}

    def close(self):
        if threading.get_ident() != self._owner:
            self._violate("background transaction close")
        with self._op_lock:
            if self._busy:
                self._violate("cannot close a running callback")
            if self.phase != "closed":
                prior = self.phase
                self._cleanup()
                self.phase = "closed"
                self._event("closed", detail={"prior_phase": prior})

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None and self.phase not in ("failed", "closed"):
            self._fail(error)
        self.close()


def load_forecast_array(root, descriptor):
    """Read one sealed NPY leaf in either storage format, returning an owned view.

    The byte-backed NumPy result is immutable. This validates content integrity;
    authenticate its descriptor using an externally pinned transaction or archive.
    """
    from mdb.forecast_storage import ArrayReader

    try:
        return ArrayReader(root).array(descriptor)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError) as error:
        raise ForecastTransactionError(str(error)) from error


def load_forecast_tree(root, descriptor):
    """Decode a numerical tree with one per-call verified-bundle byte cache."""
    from mdb.forecast_storage import ArrayReader

    try:
        return ArrayReader(root).tree(descriptor)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError) as error:
        raise ForecastTransactionError(str(error)) from error


def _verify_numerical_records(root, declarations, frontiers, forecasts):
    """Check every referenced numerical leaf with stdlib-only storage validation."""
    from mdb.forecast_storage import ArrayReader

    reader = ArrayReader(root)

    def visit(value):
        if isinstance(value, Mapping):
            if value.get("kind") == "array":
                reader.raw(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    try:
        visit(declarations["actual_weight_manifest"])
        for record in frontiers:
            visit(record["value"])
            if "roots" in record:
                visit(record["roots"])
        for record in forecasts:
            visit(record["predictions"])
            for payload in record["payloads"].values():
                visit(payload["value"])
                visit(payload["original"])
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RuntimeError) as error:
        raise ForecastTransactionError(str(error)) from error


def verify_transaction(root, *, expected_seal_sha256=None, expected_event_head=None,
                       max_files=8192, max_bytes=1024 * 1024 * 1024):
    """Verify metadata/bytes without importing NumPy, Torch, or model code.

    An external expected seal/head authenticates the identity being checked.
    Without those pins this establishes local hash consistency, not provenance
    against an actor who can replace the entire directory.
    """
    root = Path(root).resolve()
    if type(max_files) is not int or not 1 <= max_files <= 65536:
        raise ForecastTransactionError("invalid offline file bound")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 1024 * 1024 * 1024:
        raise ForecastTransactionError("invalid offline byte bound")

    def path_for(relative):
        if not isinstance(relative, str):
            raise ForecastTransactionError("artifact path must be text")
        relative_path = Path(relative)
        if relative_path.is_absolute() or any(part in (".", "..") for part in relative_path.parts):
            raise ForecastTransactionError("unsafe artifact path")
        path = root / relative_path
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ForecastTransactionError("artifact escapes transaction directory")
        return path

    files = [path for path in root.rglob("*") if path.is_file()]
    if len(files) > max_files or sum(path.stat().st_size for path in files) > max_bytes:
        raise ForecastTransactionError("offline transaction geometry exceeds bound")
    raw = path_for("seal.json").read_bytes()
    identifier = _sha(raw)
    if expected_seal_sha256 is not None and expected_seal_sha256 != identifier:
        raise ForecastTransactionError("global seal differs from expected identity")
    manifest = json.loads(raw)
    if manifest.get("schema") != "mdb-global-forecast-seal-v1":
        raise ForecastTransactionError("unsupported forecast seal")
    for relative, reference in manifest["files"].items():
        actual = path_for(relative).read_bytes()
        if len(actual) != reference["bytes"] or _sha(actual) != reference["sha256"]:
            raise ForecastTransactionError("sealed artifact changed: " + relative)
    frontiers, forecasts = [], []
    for relative in manifest["files"]:
        kind = relative.split("/", 1)[0]
        if relative.endswith(".json") and kind in ("frontiers", "forecasts"):
            (frontiers if kind == "frontiers" else forecasts).append(json.loads(path_for(relative).read_bytes()))
    _verify_numerical_records(root, manifest["declarations"], frontiers, forecasts)
    previous, events = None, []
    for index, path in enumerate(sorted((root / "events").glob("*.json"))):
        raw = path_for(str(path.relative_to(root))).read_bytes()
        event = json.loads(raw)
        if path.name != f"{index:06d}.json" or event["sequence"] != index or event["previous"] != previous:
            raise ForecastTransactionError("phase event chain differs")
        receipt_id = event["receipt_id"]
        if not isinstance(receipt_id, str) or not re.fullmatch("[0-9a-f]{64}", receipt_id):
            raise ForecastTransactionError("invalid public receipt identity")
        receipt = json.loads(path_for("custody/receipts/" + receipt_id + ".json").read_bytes())
        declared_fingerprint = receipt.pop("fingerprint", None)
        body = {key: value for key, value in event.items() if key != "receipt_id"}
        if declared_fingerprint != receipt_id or _sha(_canonical(receipt)) != receipt_id or receipt.get("phase_event") != body:
            raise ForecastTransactionError("phase event/public receipt binding differs")
        previous = _sha(raw)
        events.append(event)
    if expected_event_head is not None and previous != expected_event_head:
        raise ForecastTransactionError("event head differs from expected identity")
    return {"schema": "mdb-forecast-offline-verification-v1", "verified": True,
            "seal_sha256": identifier, "event_head": previous, "events": len(events),
            "phase": events[-1]["phase"] if events else None,
            "identity_scope": "externally-pinned" if expected_seal_sha256 else "local-hash-consistency",
            "frontiers": manifest["frontiers"], "forecasts": manifest["forecasts"]}


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="mdb forecast", description="Verify or show sealed forecast metadata without loading a model.")
    parser.add_argument("action", choices=("verify", "show"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-seal")
    args = parser.parse_args(argv)
    result = verify_transaction(args.path, expected_seal_sha256=args.expected_seal)
    if args.action == "show":
        result["manifest"] = json.loads((args.path / "seal.json").read_bytes())
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0

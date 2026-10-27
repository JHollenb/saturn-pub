"""Public runner construction and local use of the existing debugger protocol."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from threading import RLock
from uuid import uuid4

from .backends import ForwardResult, RunnerCapabilities, create_backend
from .debugger_adapters import CallbackDebuggerAdapter, DebuggerAdapter
from .errors import BenchError


class _SessionDebuggerAdapter(CallbackDebuggerAdapter):
    """Reuse debugger semantics while sharing runner serialization/lifetime."""

    def __init__(self, session):
        self.session = session
        names = (
            "status",
            "inspect",
            "checkpoint",
            "restore",
            "preview_act",
            "commit_act",
            "abort_act",
            "replay",
            "step",
            "continue_run",
            "resolve_value",
        )
        callbacks = {name: partial(session._state_call, name) for name in names}
        for name in session.adapter.optional_operations():
            callbacks[name] = partial(self._optional, name)
        super().__init__(session.adapter.capabilities, **callbacks)

    def _optional(self, name, *args, **kwargs):
        with self.session._lock:
            self.session._require_open()
            return self.session.adapter.optional_operation(name, *args, **kwargs)

    def address_specs(self):
        return self.session.adapter.address_specs()

    def structs(self):
        return self.session.adapter.structs()

    def validate_read(self, path):
        # Native adapters can resolve qualified symbols beyond static slots.
        with self.session._lock:
            self.session._require_open()
            return self.session.adapter.validate_read(path)

    def optional_operations(self):
        return self.session.adapter.optional_operations()


class BackendBenchSession:
    """A caller-owned executor, optionally bound to a native DebuggerAdapter.

    This construction path makes no model-family, tensor-framework or scheduler
    assumptions. It never synthesizes checkpoints from captured activations.
    """

    def __init__(
        self, backend, *, identity, debugger_adapter=None, owns_backend=False, backend_options=None
    ):
        if not isinstance(identity, Mapping) or not identity.get("model_id"):
            raise BenchError("a caller-declared model_id identity is required")
        self.identity = json.loads(json.dumps(dict(identity), allow_nan=False))
        self.session_id = "backend-" + uuid4().hex
        self._lock = RLock()
        self.closed = False
        options = dict(backend_options or {})
        created = isinstance(backend, str)
        if options and not created:
            raise BenchError("backend options require a registered factory name")
        self.backend = create_backend(backend, **options) if created else backend
        self.owns_backend = owns_backend or created
        self.adapter = debugger_adapter
        self._controller = self._debug_session = None
        self._last_result = None
        self._calls = 0
        try:
            if isinstance(self.backend, DebuggerAdapter):
                if debugger_adapter is not None:
                    raise BenchError("provide one debugger adapter")
                self.adapter = self.backend
                self.runner = None
            else:
                self.runner = self.backend
                if not callable(getattr(self.runner, "forward", None)):
                    raise BenchError("backend must implement forward or be a DebuggerAdapter")
                if not callable(getattr(self.runner, "capabilities", None)):
                    raise BenchError("runner must expose capabilities")
                if not isinstance(self.runner.capabilities(), RunnerCapabilities):
                    raise BenchError("runner must return RunnerCapabilities")
                if self.adapter is None:
                    self.adapter = getattr(self.runner, "debugger_adapter", None)
            if self.adapter is not None and not isinstance(self.adapter, DebuggerAdapter):
                raise BenchError("debugger_adapter must implement the existing DebuggerAdapter ABC")
            bound_identity = getattr(self.adapter, "identity", None)
            if bound_identity is not None and bound_identity != self.identity:
                raise BenchError("session identity must match the stateful adapter's bound identity")
        except BaseException:
            if self.owns_backend:
                close = getattr(self.backend, "close", None)
                if callable(close):
                    close()
            raise

    def _require_open(self):
        if self.closed:
            raise BenchError("backend session is closed")

    def capabilities(self):
        with self._lock:
            self._require_open()
            runner = self.runner.capabilities() if self.runner else None
            state = self.adapter.capabilities_payload() if self.adapter else None
            operations = ["inspect", "close"]
            if runner:
                operations.append("forward")
            if state:
                operations += [
                    "debugger",
                    "checkpoint",
                    "restore",
                    "step",
                    "continue_run",
                    "preview_act",
                    "commit_act",
                    "abort_act",
                    "replay",
                    "resolve_value",
                ]
            return {
                "schema": "mdb-backend-session-v1",
                "session_id": self.session_id,
                "model_identity": deepcopy(self.identity),
                "identity_authority": "caller-declared",
                "backend": runner.backend if runner else state["backend"],
                "capture_points": list(runner.capture_points) if runner else [],
                "intervention_points": list(runner.intervention_points) if runner else [],
                "intervention_callbacks": runner.intervention_callbacks if runner else False,
                "intervention_specs": runner.intervention_specs if runner else False,
                "operations": operations,
                "stateful_debugger": state,
                "activation_capture_is_checkpoint": False,
            }

    def forward(self, inputs, *, capture=(), interventions=None, **kwargs):
        with self._lock:
            self._require_open()
            if self.runner is None:
                raise BenchError("this adapter supplies native debugger operations, not forward")
            capture, interventions = self.runner.capabilities().validate(capture, interventions)
            result = self.runner.forward(
                inputs, capture=capture, interventions=interventions, **kwargs
            )
            if not isinstance(result, ForwardResult) or not isinstance(result.captured, Mapping):
                raise BenchError("runner must return ForwardResult with a captured mapping")
            missing = set(capture) - set(result.captured)
            if missing:
                raise BenchError(f"requested hooks did not fire: {sorted(missing)}")
            self._last_result = result
            self._calls += 1
            return result

    def inspect(self, path=None):
        with self._lock:
            self._require_open()
            if path is None:
                return {
                    "session_id": self.session_id,
                    "forward_calls": self._calls,
                    "capability": self.capabilities(),
                }
            if self._last_result is not None and path in self._last_result.captured:
                return self._last_result.captured[path]
            if self.adapter is not None:
                return self.adapter.inspect(path)
            raise BenchError(f"no captured value: {path}")

    def _state_call(self, operation, *args, **kwargs):
        with self._lock:
            self._require_open()
            if self.adapter is None:
                raise BenchError(f"backend does not supply stateful operation: {operation}")
            return getattr(self.adapter, operation)(*args, **kwargs)

    def checkpoint(self, name=None):
        return self._state_call("checkpoint", name)

    def restore(self, cut_id):
        return self._state_call("restore", cut_id)

    def step(self, count=1):
        if type(count) is not int or count <= 0:
            raise BenchError("step count must be a positive integer")
        return self._state_call("step", count)

    def continue_run(self):
        return self._state_call("continue_run")

    def preview_act(self, act):
        with self._lock:
            self._require_open()
            if self.adapter is None:
                raise BenchError("backend does not supply stateful operation: preview_act")
            self.adapter.validate_write(act)
            return self._state_call("preview_act", act)

    def commit_act(self, act_id):
        return self._state_call("commit_act", act_id)

    def abort_act(self, act_id):
        return self._state_call("abort_act", act_id)

    def replay(self, cut_id):
        return self._state_call("replay", cut_id)

    def resolve_value(self, path):
        if self.adapter is not None:
            return self._state_call("resolve_value", path)
        return self.inspect(path)

    def debugger(self):
        """Connect the existing debugger locally, without a broker or mrun."""
        with self._lock:
            self._require_open()
            if self.adapter is None:
                raise BenchError("backend does not supply a stateful debugger adapter")
            if self._debug_session is None:
                from .debugger_protocol import DebugCapability, InMemoryDebugTransport
                from .debugger_session import DebugSession
                from .debugger_worker import DebugWorkerController

                caps = self.adapter.capabilities
                capability = DebugCapability(
                    job_id=self.session_id,
                    session_id=self.session_id,
                    family=caps.family,
                    model_identity=deepcopy(self.identity),
                    boundaries=caps.boundaries,
                    inspectable_slots=caps.inspectable_slots,
                    writable_slots=caps.writable_slots,
                    exact_replay=caps.exact_replay,
                    state_schema_id=caps.state_schema_id,
                    backend=caps.backend,
                    consumer=caps.consumer,
                    boundary_catalog=caps.boundary_catalog,
                    address_specs=caps.address_specs,
                    structs=caps.structs,
                    algebra_ops=caps.algebra_ops,
                    checkpoint_requires_pause=caps.checkpoint_requires_pause,
                    optional_operations=self.adapter.optional_operations(),
                )
                self._controller = DebugWorkerController(
                    capability, _SessionDebuggerAdapter(self), execution_lock=self._lock
                )
                transport = InMemoryDebugTransport(self._controller)
                binder = getattr(self.adapter, "bind_debugger", None)
                if callable(binder):
                    transport = binder(self._controller)
                self._debug_session = DebugSession(transport, job_id=self.session_id)
                if callable(binder):
                    self._debug_session.pause()
            return self._debug_session

    def safe_point(self, boundary, index, metadata=None):
        """The native runtime calls this at its own safe execution boundaries."""
        with self._lock:
            self.debugger()
            return self._controller.safe_point(boundary, index, metadata)

    def close(self):
        with self._lock:
            if not self.closed:
                if self.owns_backend:
                    close = getattr(self.backend, "close", None)
                    if callable(close):
                        close()
                self._last_result = None
                self.closed = True

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *args):
        self.close()

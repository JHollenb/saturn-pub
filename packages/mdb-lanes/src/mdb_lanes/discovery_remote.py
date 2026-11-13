"""Discovery over the existing debugger transport and finite resident worker.

No second HTTP service is introduced. The mrun broker retains authentication
and lease ownership; only JSON control metadata crosses the control channel.
"""
from __future__ import annotations

import copy
import math
import os
import time
import uuid
from collections.abc import Mapping

from .debugger_adapters import AdapterCapabilities, DebuggerAdapter
from .debugger_protocol import (
    DISCOVERY_RELAY_BINDING_SCHEMA,
    DebugCapability,
    DebuggerProtocolError,
    DebugResponse,
    HttpDebugTransport,
    MrunDebugWorkerPoller,
    bind_discovery_relay_request,
    digest,
)
from .debugger_session import DebugSession
from .debugger_worker import DebugWorkerController
from .scheduling_contract import (
    BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA,
    BOUNDED_DEBUGGER_LEASE_SCOPE,
    DIRECT_COLLECTION_POLICY,
    FINITE_DEBUGGER_EXECUTION_MODEL,
    LOCAL_CHECKPOINT_REPLAY_POLICY,
    NO_AUTOMATIC_RETRY_POLICY,
    OUTER_MRUN_CONTROL_PLANE,
    RESIDENT_WORKER_DATA_PLANE,
    validate_bounded_debugger_lease_contract,
)

DISCOVERY_OPERATIONS = tuple("discovery." + name for name in (
    "capabilities", "status", "lane", "catalog", "alias", "break", "watch", "motion",
    "inspect", "checkpoint", "branch", "act", "events", "close"))


def bounded_debugger_scheduling_contract(*, max_debug_commands, max_wall_ms,
                                         max_native_operations, max_model_forwards,
                                         max_branches=16, extra=None):
    """Declare finite interactive admission honestly under its own schema."""
    contract = {
        "schema": BOUNDED_DEBUGGER_LEASE_CONTRACT_SCHEMA,
        "lease_scope": BOUNDED_DEBUGGER_LEASE_SCOPE,
        "execution_model": FINITE_DEBUGGER_EXECUTION_MODEL,
        "task_family": "discovery-debugger", "interactive": True,
        "control_plane": OUTER_MRUN_CONTROL_PLANE, "data_plane": RESIDENT_WORKER_DATA_PLANE,
        "max_mrun_submissions": 1, "per_branch_mrun_submissions": 0,
        "per_transition_mrun_submissions": 0, "additional_scheduler_submissions": 0,
        "nested_mrun_submissions": 0, "model_loads": 1, "resident_model_concurrency": 1,
        "branch_execution": "in_process", "replay_policy": LOCAL_CHECKPOINT_REPLAY_POLICY,
        "collection_policy": DIRECT_COLLECTION_POLICY,
        "mrun_retry_policy": NO_AUTOMATIC_RETRY_POLICY, "retry_on_kill": False,
        "checkpoint_policy": "one finite resident debugger lease; native safe-point restore and branches stay local",
        "max_debug_commands": max_debug_commands, "max_wall_ms": max_wall_ms,
        "max_native_operations": max_native_operations, "max_model_forwards": max_model_forwards,
        "max_branches": max_branches,
    }
    if extra:
        overlap = set(extra).intersection(contract)
        if overlap:
            raise ValueError("bounded debugger contract fields are owned by the helper: " +
                             ", ".join(sorted(overlap)))
        contract.update(extra)
    return validate_bounded_debugger_lease_contract(contract)


class DiscoveryDebuggerAdapter(DebuggerAdapter):
    """Legacy status/capability bridge; mutation requires the typed dialect."""

    def __init__(self, session):
        self.session = session
        self.capabilities = AdapterCapabilities(
            family="mdb-discovery-aten", boundaries=("native-safe-point",),
            inspectable_slots=("discovery-v1:live-ports",),
            writable_slots=("discovery-v1:unconsumed-produced-ports",),
            mutable_closure=("lane-frontier", "architectural-driver"),
            exact_replay=True, state_schema_id="mdb-discovery-session-v1",
            backend="aten-cpu-fp32", consumer="native-ATen-record-replay",
            algebra_ops=(), checkpoint_requires_pause=True)

    def status(self):
        return self.session.call("status")

    def optional_operations(self):
        return DISCOVERY_OPERATIONS

    @staticmethod
    def _typed_only(*args, **kwargs):
        raise DebuggerProtocolError("use negotiated discovery-v1 operations for this resident session")

    inspect = _typed_only
    checkpoint = _typed_only
    restore = _typed_only
    preview_act = _typed_only
    commit_act = _typed_only
    abort_act = _typed_only
    replay = _typed_only
    step = _typed_only
    continue_run = _typed_only


def make_discovery_worker(session, *, job_id, session_id=None, endpoint=None):
    """Expose an already bound resident session at its actual stopped heads."""
    adapter = DiscoveryDebuggerAdapter(session)
    bindings = sorted({lane.branch.runtime.catalog.program_id
                       for lane in session.lanes.values() if lane.branch.runtime is not None})
    capabilities = adapter.capabilities
    capability = DebugCapability(
        job_id=job_id, session_id=session_id or "discovery-" + uuid.uuid4().hex,
        family=capabilities.family, model_identity={"program_bindings": bindings,
                                                   "binding_digest": digest(bindings)},
        boundaries=capabilities.boundaries,
        inspectable_slots=capabilities.inspectable_slots,
        writable_slots=capabilities.writable_slots, exact_replay=capabilities.exact_replay,
        endpoint=endpoint, state_schema_id=capabilities.state_schema_id,
        backend=capabilities.backend, consumer=capabilities.consumer,
        source_fingerprint=digest(bindings), optional_operations=DISCOVERY_OPERATIONS,
        checkpoint_requires_pause=True)
    return DebugWorkerController(capability, adapter, discovery_session=session)


def attach_discovery_worker(session, *, job_id, session_id=None, endpoint, token=None,
                            agent_headers=None, debug_credential=None,
                            debug_credential_id=None, timeout=30.0):
    """Register with the existing mrun job broker; no compute is advanced."""
    worker = make_discovery_worker(session, job_id=job_id, session_id=session_id,
                                   endpoint=endpoint)
    poller = MrunDebugWorkerPoller(endpoint, token=token, agent_headers=agent_headers,
                                  debug_credential=debug_credential,
                                  debug_credential_id=debug_credential_id, timeout=timeout)
    poller.register(worker.capability)
    return worker, poller


def attach_discovery_from_environment(session, *, environment=None, session_id=None,
                                      timeout=30.0):
    """Use job-scoped credentials injected by an admitted mrun lease."""
    environment = os.environ if environment is None else environment
    job_id = environment.get("MRUN_JOB_ID")
    server = environment.get("MRUN_SERVER_URL")
    if not job_id or not server:
        raise DebuggerProtocolError("discovery worker requires MRUN_JOB_ID and MRUN_SERVER_URL")
    credential = environment.get("MRUN_DEBUG_CREDENTIAL")
    return attach_discovery_worker(session, job_id=job_id, session_id=session_id,
        endpoint=f"{server.rstrip('/')}/api/jobs/{job_id}/debug",
        token=None if credential else environment.get("MRUN_SERVER_TOKEN"),
        debug_credential=credential,
        debug_credential_id=environment.get("MRUN_DEBUG_CREDENTIAL_ID"), timeout=timeout)


def serve_discovery(worker, poller, *, max_commands=2048, max_wall_ms=120000,
                    poll_s=0.25, register=False, scheduling_contract=None):
    """Service one finite admitted residency; motion runs only on typed requests.

    The poller waits only at safe resident frontiers. Each accepted motion uses
    DiscoverySession's own primitive safe points and stricter work budgets. The
    remaining residency time caps every remote motion's wall budget; it cannot
    preempt a native kernel. ``discovery.close`` ends service without committing
    heads, resolving effects, clearing blockers, or extending the lease.
    """
    if worker.discovery_session is None:
        raise DebuggerProtocolError("worker has no discovery session")
    if type(max_commands) is not int or max_commands < 1:
        raise ValueError("max_commands must be a positive finite integer")
    if type(max_wall_ms) is not int or max_wall_ms < 1:
        raise ValueError("max_wall_ms must be a positive finite integer")
    if type(poll_s) not in (int, float) or not math.isfinite(poll_s) or not 0 < poll_s <= 1:
        raise ValueError("poll_s must be finite in (0, 1]")
    if register:
        poller.register(worker.capability)
    session = worker.discovery_session
    if scheduling_contract is not None:
        contract = validate_bounded_debugger_lease_contract(scheduling_contract)
        max_commands = min(max_commands, contract["max_debug_commands"])
        max_wall_ms = min(max_wall_ms, contract["max_wall_ms"])
        for budget, ceiling in (("max_native_operations", contract["max_native_operations"]),
                                ("max_model_forwards", contract["max_model_forwards"]),
                                ("max_branches", contract["max_branches"]),
                                ("max_commands", contract["max_debug_commands"])):
            session.budgets[budget] = min(session.budgets[budget], ceiling)
    remaining_session_ms = session.budgets["max_wall_ms"] - int(
        (time.monotonic() - session._started) * 1000)
    residency_ms = min(max_wall_ms, max(0, remaining_session_ms))
    deadline = time.monotonic() + residency_ms / 1000
    # This narrows the logical session allowance; it never extends an existing
    # plan or admitted lease. All native motion consults that same ceiling.
    session.budgets["max_wall_ms"] = min(session.budgets["max_wall_ms"],
        int((deadline - session._started) * 1000))
    served, control_error = 0, None
    while served < max_commands and not worker.discovery_close_requested:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        original_timeout = getattr(poller, "timeout", None)
        try:
            if original_timeout is not None:
                poller.timeout = min(original_timeout, max(0.001, remaining))
            if poller._serve_once(worker, wait_s=min(float(poll_s), remaining)):
                served += 1
        except DebuggerProtocolError as error:
            # An idle long poll may expire at the residency ceiling. Other
            # control-channel failures return the retained head and an explicit
            # service error, without fabricating a completed motion.
            if time.monotonic() < deadline:
                control_error = str(error)
            break
        finally:
            if original_timeout is not None:
                poller.timeout = original_timeout
    reason = ("control_error" if control_error is not None else
              "closed" if worker.discovery_close_requested else
              "command_budget" if served >= max_commands else "wall_budget")
    return {"schema": "mdb-discovery-residency-v1", "reason": reason,
            "served_commands": served, "max_commands": max_commands,
            "max_wall_ms": residency_ms, "state_advanced_on_exit": False,
            "control_error": control_error,
            "status": session.call("status")}


class RemoteDiscoverySession:
    """The same controller call shape through a standard DebugSession."""

    def __init__(self, session: DebugSession):
        self.session = session
        if not isinstance(session.transport, DiscoveryRelayTransport):
            session.transport = DiscoveryRelayTransport(session.transport)
        self._closed_snapshot = None

    @classmethod
    def connect(cls, job_id, *, endpoint=None, token=None, timeout=30.0):
        resolved = endpoint or os.environ.get("SATURN_DEBUG_ENDPOINT")
        if not resolved:
            resolved = os.environ.get("MRUN_URL", "http://127.0.0.1:9025")
        resolved = resolved.rstrip("/")
        if "/api/jobs/" not in resolved and not resolved.endswith("/debug"):
            resolved += f"/api/jobs/{job_id}/debug"
        return cls(DebugSession(HttpDebugTransport(resolved,
                                token=token if token is not None else os.environ.get("MRUN_TOKEN"),
                                timeout=timeout),
                                job_id=job_id))

    def call(self, operation, payload=None, *, request_id=None, expected_revision=None):
        if not isinstance(operation, str):
            raise DebuggerProtocolError("discovery operation must be text")
        if payload is not None and not isinstance(payload, Mapping):
            raise DebuggerProtocolError("discovery payload must be an object")
        command = dict(payload or {})
        if expected_revision is not None:
            command["expected_revision"] = expected_revision
        operation = operation if operation.startswith("discovery.") else "discovery." + operation
        if self._closed_snapshot is not None:
            if operation == "discovery.status":
                return {**copy.deepcopy(self._closed_snapshot), "source": "closed-worker-snapshot"}
            raise DebuggerProtocolError("remote discovery residency is closed")
        result = self.session.call(operation, command, request_id=request_id)
        if operation == "discovery.close" and result.get("ok"):
            self._closed_snapshot = copy.deepcopy(result["final_status"])
        return result

    def status(self):
        return self.call("status")

    def capabilities(self):
        return self.call("capabilities")

    def close(self):
        return self.call("close")


class DiscoveryRelayTransport:
    """Metadata binding around the existing transport, including older brokers.

    The original transport still owns authentication, HTTP, errors and envelope
    serialization. The worker verifies and unwraps these logical IDs before its
    own content guard and revision checks.
    """

    schema = DISCOVERY_RELAY_BINDING_SCHEMA

    def __init__(self, transport):
        self.transport = transport

    def request(self, request):
        wire_request = bind_discovery_relay_request(request)
        response = self.transport.request(wire_request)
        if response.request_id != wire_request.request_id:
            raise DebuggerProtocolError("debugger response request ID differs from submitted request")
        return DebugResponse(request.request_id, response.sequence, response.ok,
                             response.result, response.error)

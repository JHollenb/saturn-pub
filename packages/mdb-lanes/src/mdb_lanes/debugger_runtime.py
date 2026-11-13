"""Runtime-side assembly helpers for mrun-instrumented Saturn workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .debugger_adapters import DebuggerAdapter
from .debugger_protocol import DebugCapability, MrunDebugWorkerPoller
from .debugger_worker import DebugWorkerController


def attach_worker_debugger(
    *,
    job_id: str,
    session_id: str,
    adapter: DebuggerAdapter,
    model_identity: Mapping[str, Any],
    endpoint: str,
    source_fingerprint: str | None = None,
    token: str | None = None,
    agent_headers: Mapping[str, str] | None = None,
    debug_credential: str | None = None,
    debug_credential_id: str | None = None,
    source_frame_stack: Any | None = None,
) -> tuple[DebugWorkerController, MrunDebugWorkerPoller]:
    """Register one native runtime and return its safe-point control pair."""
    capabilities = adapter.capabilities
    capability = DebugCapability(
        job_id=job_id,
        session_id=session_id,
        family=capabilities.family,
        model_identity=dict(model_identity),
        boundaries=capabilities.boundaries,
        inspectable_slots=capabilities.inspectable_slots,
        writable_slots=capabilities.writable_slots,
        exact_replay=capabilities.exact_replay,
        endpoint=endpoint,
        state_schema_id=capabilities.state_schema_id,
        backend=capabilities.backend,
        consumer=capabilities.consumer,
        source_fingerprint=source_fingerprint,
        boundary_catalog=capabilities.boundary_catalog,
        address_specs=capabilities.address_specs,
        structs=capabilities.structs,
        algebra_ops=capabilities.algebra_ops,
        optional_operations=tuple(dict.fromkeys((*adapter.optional_operations(),
            *(("frames",) if source_frame_stack is not None else ())))),
        # A live mrun worker may only materialize an explicit checkpoint at a
        # typed safe point.  In-memory mechanics fixtures can opt out by
        # constructing their own DebugCapability with the legacy default.
        checkpoint_requires_pause=True,
    )
    controller = DebugWorkerController(capability, adapter, source_frame_stack=source_frame_stack)
    poller = MrunDebugWorkerPoller(
        endpoint,
        token=token,
        agent_headers=dict(agent_headers or {}),
        debug_credential=debug_credential,
        debug_credential_id=debug_credential_id,
    )
    poller.register(capability)
    return controller, poller

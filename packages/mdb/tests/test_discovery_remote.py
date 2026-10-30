from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
import torch

from mdb.debugger_protocol import (
    DEBUGGER_RESPONSE_SCHEMA,
    DebuggerProtocolError,
    DebugRequest,
    InMemoryDebugTransport,
)
from mdb.debugger_session import DebugSession
from mdb.discovery import DiscoverySession
from mdb.discovery_remote import (
    RemoteDiscoverySession,
    attach_discovery_from_environment,
    attach_discovery_worker,
    bounded_debugger_scheduling_contract,
    make_discovery_worker,
    serve_discovery,
)
from mdb.discovery_runtime import ReplayForward
from mdb.discovery_script import ScriptRunner
from mdb.forward_recording import record_forward


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.LayerNorm(3), torch.nn.Linear(3, 5),
                                         torch.nn.GELU(), torch.nn.Linear(5, 3))

    def forward(self, value):
        return value + self.block(value)


def resident():
    torch.manual_seed(2)
    native = record_forward(Residual().eval(), (torch.ones(2, 3),),
        model_identity="remote-tiny", execution={"backend": "cpu-fp32"},
        input_identity={"case": "remote-test"})
    session = DiscoverySession()
    session.add_lane("a", lambda: ReplayForward(native))
    return session


def connected(session=None):
    session = resident() if session is None else session
    worker = make_discovery_worker(session, job_id="job-remote", session_id="remote-session")
    remote = RemoteDiscoverySession(DebugSession(InMemoryDebugTransport(worker), job_id="job-remote"))
    return session, worker, remote


def test_same_envelopes_expose_negotiated_extension_without_legacy_motion_bypass():
    session, worker, remote = connected()
    capability = remote.session.capabilities()
    assert capability["schema"] == "saturn-debugger-capability-v1"
    assert capability["discovery"]["interface"] == "discovery-v1"
    assert "discovery.motion" in capability["optional_operations"]
    response = worker.handle(DebugRequest("discovery.status"))
    assert response.to_dict()["schema"] == DEBUGGER_RESPONSE_SCHEMA
    assert response.result["lanes"][0]["pc"] == 0
    with pytest.raises(DebuggerProtocolError, match="discovery-v1"):
        remote.session.call("continue")
    assert session.lanes["a"].branch.runtime.pc == 0
    assert remote.capabilities()["execution_modes"] == ["record-replay"]


def test_remote_retry_content_guard_and_expected_revision_match_local_controller():
    session, worker, remote = connected()
    first = remote.call("motion", {"unit": "op"}, request_id="motion-1", expected_revision=0)
    assert first["completed_unit"] and first["lane"]["pc"] == 1
    assert remote.call("motion", {"unit": "op"}, request_id="motion-1", expected_revision=0) == first
    assert session.work["replay_native_operations"] == 1
    with pytest.raises(DebuggerProtocolError, match="different command content"):
        remote.call("motion", {"unit": "law"}, request_id="motion-1", expected_revision=0)
    stale = remote.call("motion", {"unit": "op"}, expected_revision=0)
    assert not stale["ok"] and stale["status"] == "revision_conflict"
    assert session.work["replay_native_operations"] == 1
    # Sequence rejection is the existing outer transport rule; no execution.
    response = worker.handle(DebugRequest("discovery.motion", {"unit": "op"},
                                         request_id="stale-sequence", sequence=1))
    assert not response.ok and "out-of-order" in response.error


def test_discovery_request_id_cannot_escape_guard_through_legacy_operation():
    session, worker, _ = connected()
    request = DebugRequest("discovery.motion", {"unit": "op"}, request_id="bound")
    assert worker.handle(request).ok
    response = worker.handle(DebugRequest("status", request_id="bound"))
    assert not response.ok and "different command content" in response.error
    assert session.lanes["a"].branch.runtime.pc == 1
    worker.handle(DebugRequest("status", request_id="legacy-id"))
    refused = worker.handle(DebugRequest("discovery.motion", {"unit": "op"}, request_id="legacy-id"))
    assert not refused.ok and session.lanes["a"].branch.runtime.pc == 1


def test_fresh_client_resynchronizes_stable_id_without_reissuing_native_motion():
    session, worker, remote = connected()
    for _ in range(3):
        remote.status()
    fresh = RemoteDiscoverySession(DebugSession(InMemoryDebugTransport(worker), job_id="job-remote"))
    result = fresh.call("motion", {"unit": "op"}, request_id="fresh-client-id", expected_revision=0)
    assert result["ok"] and result["lane"]["pc"] == 1
    assert session.work["replay_native_operations"] == 1
    assert fresh.call("motion", {"unit": "op"}, request_id="fresh-client-id", expected_revision=0) == result
    assert len(fresh.session._receipt) == 3  # refused sequence, success, cached retry


def test_wire_binding_bypasses_old_broker_cache_and_reconciles_lost_response_once():
    session, worker, _ = connected()

    class OldBroker:
        def __init__(self):
            self.responses, self.seen = {}, []
            self.drop = False

        def request(self, request):
            self.seen.append(request)
            if request.request_id not in self.responses:
                self.responses[request.request_id] = worker.handle(request)
            response = self.responses[request.request_id]
            if self.drop and request.operation == "discovery.motion":
                self.drop = False
                raise OSError("lost reply after successful native execution")
            return response

    broker = OldBroker()
    remote = RemoteDiscoverySession(DebugSession(broker, job_id="job-remote"))
    for _ in range(3):
        remote.status()
    fresh = RemoteDiscoverySession(DebugSession(broker, job_id="job-remote"))
    broker.drop = True
    # The first fresh request is sequence-refused, then its reply is lost. No
    # model work happened. Resending the same logical ID resolves against the
    # worker while using a different old-broker wire cache key.
    with pytest.raises(OSError, match="lost reply"):
        fresh.call("motion", {"unit": "op"}, request_id="stable", expected_revision=0)
    first = fresh.call("motion", {"unit": "op"}, request_id="stable", expected_revision=0)
    assert first["ok"] and session.work["replay_native_operations"] == 1
    assert fresh.call("motion", {"unit": "op"}, request_id="stable", expected_revision=0) == first
    with pytest.raises(DebuggerProtocolError, match="different command content"):
        fresh.call("motion", {"unit": "law"}, request_id="stable", expected_revision=0)
    assert session.work["replay_native_operations"] == 1
    broker.drop = True
    with pytest.raises(OSError, match="lost reply"):
        fresh.call("motion", {"unit": "op"}, request_id="lost-success", expected_revision=1)
    assert session.work["replay_native_operations"] == 2
    reconciled = fresh.call("motion", {"unit": "op"}, request_id="lost-success", expected_revision=1)
    assert reconciled["ok"] and reconciled["lane"]["pc"] == 2
    assert session.work["replay_native_operations"] == 2
    mutation_wires = [request.request_id for request in broker.seen
                      if request.operation == "discovery.motion"]
    assert len(set(mutation_wires)) == len(mutation_wires)


def test_relay_metadata_tampering_is_refused_before_any_native_work():
    from mdb.debugger_protocol import bind_discovery_relay_request

    session, worker, _ = connected()
    request = bind_discovery_relay_request(DebugRequest("discovery.motion", {"unit": "op"},
                                                       request_id="logical", sequence=1))
    request.payload["unit"] = "law"
    response = worker.handle(request)
    assert not response.ok and "binding differs" in response.error
    assert session.lanes["a"].branch.runtime.pc == 0


def test_malformed_discovery_request_is_a_protocol_refusal_not_worker_failure():
    session, worker, _ = connected()
    for request in (DebugRequest("discovery.motion", ["op"]),
                    DebugRequest("discovery.motion", {"unit": "op"}, sequence=True),
                    DebugRequest(123, {"unit": "op"})):
        response = worker.handle(request)
        assert not response.ok
    assert session.lanes["a"].branch.runtime.pc == 0


def test_remote_script_captures_branches_and_native_act_keep_hit_cut_immutable():
    session, _, remote = connected()
    script = """interface mdb-discovery-v1
alias focus = mlp_001
break focus exit --id hit
on-hit hit
  capture before
  call {"operation":"act","payload":{"selector":"focus:output","value_spec":{"kind":"zeros_like"}}}
end
step law
"""
    result = ScriptRunner(remote).run(script)
    assert result["ok"] and result["handler_hits"] == 1
    commands = [row for row in result["records"] if "response" in row]
    hit = next(row["response"] for row in commands if row["operation"] == "motion")
    stopped = session.cuts[hit["hit_cut"]]["runtime"].read("mlp_001:output")
    actual = session.lanes["a"].branch.runtime.read("mlp_001:output")
    assert torch.equal(actual, torch.zeros_like(actual))
    assert not torch.equal(stopped, actual)


def test_remote_pending_effect_budget_and_group_blocks_are_not_cleared_by_reattach():
    session, _, remote = connected()
    session.lanes["a"].branch.blocked = [
        {"kind": "predicted_syscall", "blocking": True},
        {"kind": "group_pause", "blocking": True, "owner": "other"}]
    first = remote.call("motion", {"unit": "op"})
    another = RemoteDiscoverySession(remote.session)
    again = another.call("motion", {"unit": "forward"})
    assert first["stop_causes"] == again["stop_causes"]
    assert session.lanes["a"].branch.runtime.pc == 0
    # A finite work ceiling also remains unchanged by client reconnect.
    session.lanes["a"].branch.blocked = []
    session.budgets["max_native_operations"] = 0
    blocked = another.call("motion", {"unit": "op"})
    assert any(row["kind"] == "budget" for row in blocked["stop_causes"])
    assert session.lanes["a"].branch.runtime.pc == 0


@contextmanager
def broker():
    """Actual HTTP envelopes using the existing mrun broker endpoint shape."""
    requests = queue.Queue()
    responses, events, registrations, headers = {}, {}, [], []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, body):
            raw = json.dumps(body, allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            parsed = urlsplit(self.path)
            wait_s = float(parse_qs(parsed.query).get("wait_s", [0])[0])
            try:
                request = requests.get(timeout=max(0.001, wait_s))
            except queue.Empty:
                request = None
            self.send_json({"request": request})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            headers.append(dict(self.headers))
            if self.path.endswith("/register"):
                registrations.append(body)
                self.send_json({"registered": True})
            elif self.path.endswith("/respond"):
                with lock:
                    responses[body["request_id"]] = body
                    events[body["request_id"]].set()
                self.send_json({"accepted": True})
            else:
                # Match the production broker's old cache-before-dispatch
                # behavior, which the content-bound wire metadata must survive.
                with lock:
                    cached = responses.get(body["request_id"])
                if cached is not None:
                    self.send_json(cached)
                    return
                event = threading.Event()
                with lock:
                    events[body["request_id"]] = event
                requests.put(body)
                if not event.wait(3):
                    self.send_error(504)
                    return
                self.send_json(responses[body["request_id"]])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {"base": f"http://127.0.0.1:{server.server_port}",
               "registrations": registrations, "headers": headers}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_actual_http_attach_register_command_poll_response_and_finite_close():
    session = resident()
    with broker() as service:
        endpoint = service["base"] + "/api/jobs/job-remote/debug"
        worker, poller = attach_discovery_worker(session, job_id="job-remote", endpoint=endpoint,
            debug_credential="narrow-credential", debug_credential_id="credential-1")
        reports = []
        thread = threading.Thread(target=lambda: reports.append(serve_discovery(worker, poller,
            max_commands=8, max_wall_ms=5000, poll_s=0.01)), daemon=True)
        thread.start()
        remote = RemoteDiscoverySession.connect("job-remote", endpoint=service["base"], token="client-token")
        assert remote.capabilities()["interface"] == "discovery-v1"
        assert remote.call("break", {"selector": "mlp_001", "edge": "enter", "id": "entry"})["ok"]
        hit = remote.call("motion", {"unit": "law"})
        assert hit["lane"]["pc"] == 0 and hit["stop_causes"][0]["kind"] == "breakpoint"
        assert remote.call("inspect", {"selector": "state"})["value"]["pc"] == 0
        first = remote.call("motion", {"unit": "op"}, request_id="http-retry")
        assert first["lane"]["pc"] == 1
        assert remote.call("motion", {"unit": "op"}, request_id="http-retry") == first
        with pytest.raises(DebuggerProtocolError, match="different command content"):
            remote.call("motion", {"unit": "law"}, request_id="http-retry")
        assert remote.close()["state_advanced"] is False
        assert remote.status()["source"] == "closed-worker-snapshot"
        with pytest.raises(DebuggerProtocolError, match="closed"):
            remote.call("motion", {"unit": "op"})
        thread.join(3)
        assert not thread.is_alive()
        assert reports[0]["reason"] == "closed" and reports[0]["served_commands"] == 8
        registration = service["registrations"][0]["capability"]
        assert registration["protocol"] == "saturn-debugger-protocol-v1"
        assert "discovery.motion" in registration["optional_operations"]
        assert any(row.get("X-Mrun-Token") == "client-token" for row in service["headers"])
        assert any(row.get("X-Mrun-Debug-Credential") == "narrow-credential"
                   for row in service["headers"])
        assert session.work["replay_native_operations"] == 1


def test_residency_command_and_wall_limits_stop_without_hidden_motion():
    session, worker, _ = connected()

    class Poller:
        def __init__(self):
            self.count = 0

        def _serve_once(self, worker, *, wait_s):
            self.count += 1
            worker.handle(DebugRequest("discovery.status"))
            return True

    poller = Poller()
    result = serve_discovery(worker, poller, max_commands=2, max_wall_ms=1000)
    assert result["reason"] == "command_budget" and result["served_commands"] == 2
    assert poller.count == 2 and session.lanes["a"].branch.runtime.pc == 0

    class Idle:
        def _serve_once(self, worker, *, wait_s):
            time.sleep(wait_s)
            return False

    result = serve_discovery(worker, Idle(), max_commands=2, max_wall_ms=5, poll_s=0.001)
    assert result["reason"] == "wall_budget" and result["served_commands"] == 0
    assert session.lanes["a"].branch.runtime.pc == 0


def test_scoped_environment_reuses_mrun_endpoint_and_does_not_send_broad_token(monkeypatch):
    called = []

    def attach(session, **kwargs):
        called.append(kwargs)
        return "worker", "poller"

    monkeypatch.setattr("mdb.discovery_remote.attach_discovery_worker", attach)
    assert attach_discovery_from_environment(None, environment={
        "MRUN_JOB_ID": "job-remote", "MRUN_SERVER_URL": "http://scheduler/",
        "MRUN_SERVER_TOKEN": "broad", "MRUN_DEBUG_CREDENTIAL": "scoped",
        "MRUN_DEBUG_CREDENTIAL_ID": "credential-1"}) == ("worker", "poller")
    assert called[0]["token"] is None
    assert called[0]["endpoint"] == "http://scheduler/api/jobs/job-remote/debug"
    with pytest.raises(DebuggerProtocolError, match="MRUN_JOB_ID"):
        attach_discovery_from_environment(None, environment={})


def test_older_agent_requires_public_endpoint_explicitly_injected_into_worker_command(monkeypatch):
    calls = []

    def attach(session, **kwargs):
        calls.append(kwargs)
        return "worker", "poller"

    monkeypatch.setattr("mdb.discovery_remote.attach_discovery_worker", attach)
    # The deployed guarded agent can inject scoped credentials without the
    # newer public server URL field. A client-side MRUN_URL is not authority to
    # invent an unsealed worker endpoint.
    environment = {"MRUN_JOB_ID": "old-agent-job", "MRUN_URL": "http://client-only",
                   "MRUN_DEBUG_CREDENTIAL": "scoped-test-value",
                   "MRUN_DEBUG_CREDENTIAL_ID": "scoped-test-id"}
    with pytest.raises(DebuggerProtocolError, match="MRUN_SERVER_URL"):
        attach_discovery_from_environment(None, environment=environment)
    assert calls == []
    environment["MRUN_SERVER_URL"] = "http://sealed-scheduler/"
    assert attach_discovery_from_environment(None, environment=environment) == ("worker", "poller")
    assert calls[0]["endpoint"] == "http://sealed-scheduler/api/jobs/old-agent-job/debug"
    assert calls[0]["debug_credential"] == environment["MRUN_DEBUG_CREDENTIAL"]
    assert calls[0]["token"] is None


def test_service_enforces_declared_debugger_ceilings_without_extending_session():
    session, worker, _ = connected()
    session.budgets["max_model_forwards"] = 1
    contract = bounded_debugger_scheduling_contract(max_debug_commands=1, max_wall_ms=1000,
        max_native_operations=1, max_model_forwards=2, max_branches=1)

    class Poller:
        def _serve_once(self, worker, *, wait_s):
            worker.handle(DebugRequest("discovery.motion", {"unit": "law"}))
            return True

    result = serve_discovery(worker, Poller(), max_commands=10, max_wall_ms=10000,
                             scheduling_contract=contract)
    assert result["reason"] == "command_budget" and result["served_commands"] == 1
    assert session.budgets["max_model_forwards"] == 1
    assert session.budgets["max_native_operations"] == 1
    assert session.budgets["max_branches"] == 1
    assert session.work["replay_native_operations"] == 1
    assert session.lanes["a"].branch.runtime.pc == 1


def test_control_channel_failure_returns_retained_safe_head_and_explicit_error():
    session, worker, _ = connected()

    class Broken:
        def _serve_once(self, worker, *, wait_s):
            raise DebuggerProtocolError("broker disconnected")

    result = serve_discovery(worker, Broken(), max_commands=10, max_wall_ms=1000)
    assert result["reason"] == "control_error" and result["control_error"] == "broker disconnected"
    assert result["status"]["lanes"][0]["pc"] == 0
    assert session.work["replay_native_operations"] == 0

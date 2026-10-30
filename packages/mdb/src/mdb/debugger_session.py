"""Client-side debugger session, command semantics, lineage, and receipts."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .debugger_protocol import (
    ActSpec,
    BreakpointSpec,
    DebuggerAddressSpec,
    DebuggerProtocolError,
    DebuggerStructSpec,
    DebugRequest,
    DebugTransport,
    WatchSpec,
    conditioner_row_path,
    digest,
)

_SEMANTIC_SOURCE_WRITE_SCHEMA = "saturn-debugger-semantic-binding-v1"
_SEMANTIC_ACT_IDENTITY_FIELDS = (
    "binding_fingerprint",
    "manifest_fingerprint",
    "context_fingerprint",
)


class DebugSession:
    def __init__(
        self, transport: DebugTransport, *, job_id: str, session_id: str | None = None
    ) -> None:
        self.transport = transport
        self.job_id = job_id
        self.session_id = session_id
        self._receipt: list[dict[str, Any]] = []
        self._sequence = 0
        self._pending_act: ActSpec | None = None
        self._current_cut: str | None = None
        self._cuts: dict[str, dict[str, Any]] = {}
        self._cut_names: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._session_dir: Path | None = None
        self._capability_snapshot: dict[str, Any] = {}

    @property
    def pending_act(self) -> bool:
        return self._pending_act is not None

    def call(self, operation: str, payload: dict[str, Any] | None = None, *, request_id: str | None = None) -> dict[str, Any]:
        request_payload = dict(payload or {})
        for attempt in range(2):
            self._sequence += 1
            request = DebugRequest(operation, request_payload, sequence=self._sequence,
                                   **({"request_id": request_id} if request_id is not None else {}))
            try:
                response = self.transport.request(request)
            except Exception as exc:  # noqa: BLE001 - preserve disconnects in custody
                self._append_receipt(request, None, error=str(exc))
                raise
            if not response.ok:
                error = response.error or "debugger operation refused"
                self._append_receipt(request, response, error=error)
                expected = self._expected_sequence(error)
                if (expected is not None and attempt == 0 and
                        (request_id is None or operation.startswith("discovery."))):
                    # A newly opened shell can safely resynchronize against a
                    # resident worker after a session file load or reconnect.
                    self._sequence = expected - 1
                    continue
                raise DebuggerProtocolError(error)
            result = response.result
            self._append_receipt(request, response, result=result)
            self._remember_result(result)
            return result
        raise DebuggerProtocolError("debugger request could not be resynchronized")

    @staticmethod
    def _expected_sequence(error: str) -> int | None:
        match = re.search(r"expected next sequence (\d+)", error)
        return int(match.group(1)) if match else None

    def _append_receipt(
        self,
        request: DebugRequest,
        response: Any | None,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "request": request.to_dict(),
            "response": response.to_dict() if response is not None else None,
        }
        if result is not None:
            entry["result_digest"] = digest(result)
        if error is not None:
            entry["error"] = error
        self._receipt.append(entry)

    def record_refusal(self, command: str, error: str | Exception) -> None:
        """Record a shell-side refusal that never reached the worker."""

        message = str(error)
        self._receipt.append(
            {
                "timestamp": time.time(),
                "local": True,
                "command": command,
                "response": {"ok": False, "error": message},
                "error": message,
            }
        )

    def _remember_result(self, result: dict[str, Any]) -> None:
        capability = result.get("capability")
        if isinstance(capability, dict):
            self._capability_snapshot = dict(capability)
            self.session_id = capability.get("session_id") or self.session_id
        elif result.get("schema") == "saturn-debugger-capability-v1":
            self._capability_snapshot = dict(result)
            self.session_id = result.get("session_id") or self.session_id
        cut_id = result.get("cut_id") or result.get("handle")
        if cut_id:
            cut = dict(result)
            cut["cut_id"] = str(cut_id)
            self._cuts[str(cut_id)] = cut
            name = result.get("name")
            if name:
                self._cut_names[str(name)] = str(cut_id)
            self._current_cut = str(cut_id)
        for cut in result.get("cuts", ()) if isinstance(result.get("cuts"), list) else ():
            if isinstance(cut, dict) and cut.get("cut_id"):
                self._cuts[str(cut["cut_id"])] = dict(cut)
        if isinstance(result.get("names"), dict):
            self._cut_names.update({str(k): str(v) for k, v in result["names"].items()})

    def _act_address(
        self,
        path: str,
        supplied: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Bind a semantic Act to the exact declaration the client inspected."""

        if not self._capability_snapshot:
            self.capabilities()
        declared_identity: dict[str, Any] = {}
        for raw in self._capability_snapshot.get("address_specs", ()):
            if not isinstance(raw, dict):
                continue
            spec = DebuggerAddressSpec.from_dict(raw)
            if spec.match(path) is None:
                continue
            source_write = spec.source_write
            if source_write.get("schema") != _SEMANTIC_SOURCE_WRITE_SCHEMA:
                break
            declared_identity["semantic_symbol"] = source_write.get("symbol")
            for field in _SEMANTIC_ACT_IDENTITY_FIELDS:
                declared_identity[field] = source_write.get(field)
            break
        # Preserve caller-supplied values so the worker—not the client—records
        # and rejects any stale or forged identity in the durable transcript.
        return {**declared_identity, **dict(supplied or {})}

    @property
    def current_cut(self) -> str | None:
        return self._current_cut

    @property
    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def set_alias(self, name: str, command: str) -> dict[str, Any]:
        if not name.isidentifier() or name.startswith("__"):
            raise ValueError("alias name must be an ordinary identifier")
        if not command.strip():
            raise ValueError("alias command must not be empty")
        self._aliases[name] = command.strip()
        return {"alias": name, "command": self._aliases[name]}

    def resolve_cut(self, cut_id: str) -> str:
        aliases = {
            "@current": self._current_cut,
            "@root": self._root_cut(),
            "@parent": self._parent_cut(),
        }
        resolved = aliases.get(cut_id, self._cut_names.get(cut_id, cut_id))
        if not resolved:
            raise RuntimeError(f"no current StateCut for {cut_id}")
        return str(resolved)

    def _root_cut(self) -> str | None:
        if not self._cuts:
            return None
        current = next(iter(self._cuts))
        seen: set[str] = set()
        while current in self._cuts and current not in seen:
            seen.add(current)
            parent = self._cuts[current].get("parent_cut")
            if not parent or str(parent) not in self._cuts:
                return current
            current = str(parent)
        return current

    def _parent_cut(self) -> str | None:
        if not self._current_cut:
            return None
        cut = self._cuts.get(self._current_cut, {})
        parent = cut.get("parent_cut")
        return str(parent) if parent else None

    def status(self) -> dict[str, Any]:
        result = self.call("status")
        pause_cut = (result.get("pause_reason") or {}).get("cut") or {}
        cut_id = pause_cut.get("cut_id") or (result.get("runtime") or {}).get("cut_id")
        if cut_id:
            self._current_cut = str(cut_id)
            if isinstance(pause_cut, dict):
                self._cuts[str(cut_id)] = dict(pause_cut)
        return result

    def info(self) -> dict[str, Any]:
        return self.call("info")

    def capabilities(self) -> dict[str, Any]:
        # ``capability`` is the full identity snapshot used for persistence,
        # custody, and replay validation. ``breakable`` remains the compact
        # interactive catalog.
        result = self.call("capability")
        self.session_id = result.get("session_id") or self.session_id
        self._capability_snapshot = dict(result)
        return result

    def breakable(self) -> dict[str, Any]:
        return self.call("breakable")

    def wait_for_pause(
        self, *, timeout_s: float = 5.0, poll_s: float = 0.05
    ) -> dict[str, Any]:
        """Turn a pause request into a real StateCut at the next safe point."""

        self.pause()
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            status = self.status()
            if isinstance(status.get("capability"), dict):
                self._capability_snapshot = dict(status["capability"])
                self.session_id = status["capability"].get("session_id") or self.session_id
            if status.get("state") == "paused":
                cut = (status.get("pause_reason") or {}).get("cut") or {}
                cut_id = cut.get("cut_id") or self.current_cut
                if cut_id:
                    self._cut_names.setdefault("@attach", str(cut_id))
                    self._current_cut = str(cut_id)
                status["cut_name"] = "@attach"
                return status
            # Local mechanics transports have no runtime loop to deliver the
            # requested pause.  Advance one declared boundary so guided mode
            # remains useful and deterministic in tests/toy organisms.
            worker = getattr(self.transport, "worker", None)
            if worker is not None and status.get("state") == "pause-pending":
                boundaries = (status.get("capability") or {}).get("boundaries") or ["step"]
                worker.safe_point(str(boundaries[0]), 0, {"guided": True})
                continue
            if time.monotonic() >= deadline:
                return {**status, "timed_out": True, "timeout_s": timeout_s}
            time.sleep(min(max(0.0, poll_s), max(0.0, deadline - time.monotonic())))

    def pause(self) -> dict[str, Any]:
        return self.call("pause")

    def continue_run(self) -> dict[str, Any]:
        return self.call("continue")

    def step(self, count: int = 1, *, kind: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"count": count}
        if kind is not None:
            payload["kind"] = kind
        return self.call("step", payload)

    def inspect(self, path: str | None = None) -> dict[str, Any]:
        return self.call("inspect", {"path": path})

    def checkpoint(self, name: str | None = None) -> dict[str, Any]:
        result = self.call("checkpoint", {"name": name})
        if name and result.get("cut_id"):
            self._cut_names[str(name)] = str(result["cut_id"])
        return result

    def fork(self, name: str | None = None) -> dict[str, Any]:
        result = self.call("fork", {"name": name or "fork"})
        if name and result.get("cut_id"):
            self._cut_names[str(name)] = str(result["cut_id"])
        return result

    def restore(self, cut_id: str) -> dict[str, Any]:
        resolved = self.resolve_cut(cut_id)
        result = self.call("restore", {"cut_id": resolved})
        self._current_cut = result.get("cut_id") or resolved
        return result

    def replay(self, cut_id: str) -> dict[str, Any]:
        resolved = self.resolve_cut(cut_id)
        result = self.call("replay", {"cut_id": resolved})
        self._current_cut = result.get("cut_id") or self._current_cut
        return result

    def measure_circuit(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Measure one BenchSession-owned AR circuit at the paused cut."""

        if not isinstance(spec, dict):
            raise ValueError("measure_circuit spec must be a JSON object")
        return self.call("measure_circuit", {"spec": spec})

    def replay_circuit(self, evidence_ref: str) -> dict[str, Any]:
        """Replay one BenchSession circuit evidence reference exactly."""

        if not isinstance(evidence_ref, str) or not evidence_ref.strip():
            raise ValueError("replay_circuit evidence_ref must be a non-empty string")
        return self.call("replay_circuit", {"evidence_ref": evidence_ref.strip()})

    def _resolve_memory_spec(self, spec: dict[str, Any] | None) -> dict[str, Any]:
        if spec is None:
            return {}
        if not isinstance(spec, dict):
            raise ValueError("memory spec must be a JSON object")
        resolved = dict(spec)
        for field in ("cut_id", "donor_cut", "left_cut", "right_cut"):
            value = resolved.get(field)
            if isinstance(value, str):
                resolved[field] = self.resolve_cut(value)
        intervention = resolved.get("intervention")
        if isinstance(intervention, dict):
            resolved["intervention"] = self._resolve_memory_spec(intervention)
        return resolved

    def memory_read(self, spec: dict[str, Any] | None = None) -> dict[str, Any]:
        selected = self._current_cut
        result = self.call("memory_read", {"spec": self._resolve_memory_spec(spec)})
        self._current_cut = selected
        return result

    def memory_write(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self.call("memory_write", {"spec": self._resolve_memory_spec(spec)})

    def memory_diff(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self.call("memory_diff", {"spec": self._resolve_memory_spec(spec)})

    def memory_probe(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self.call("memory_probe", {"spec": self._resolve_memory_spec(spec)})

    def add_breakpoint(self, spec: BreakpointSpec) -> dict[str, Any]:
        return self.call("breakpoint.add", spec.to_dict())

    def list_breakpoints(self) -> dict[str, Any]:
        return self.call("breakpoint.list")

    def add_watch(self, spec: WatchSpec) -> dict[str, Any]:
        return self.call("watch.add", spec.to_dict())

    def list_watches(self) -> dict[str, Any]:
        return self.call("watch.list")

    def clear_watch(self, watch_id: str) -> dict[str, Any]:
        return self.call("watch.clear", {"watch_id": watch_id})

    def add_struct(self, spec: DebuggerStructSpec) -> dict[str, Any]:
        return self.call("struct.add", {"struct": spec.to_dict()})

    def list_structs(self) -> dict[str, Any]:
        return self.call("struct.list")

    def add_address(self, spec: DebuggerAddressSpec) -> dict[str, Any]:
        return self.call("address.add", {"address": spec.to_dict()})

    def list_addresses(self) -> dict[str, Any]:
        return self.call("address.list")

    def algebra(self) -> dict[str, Any]:
        return self.call("algebra.describe")

    def tree(self, prefix: str | None = None) -> dict[str, Any]:
        return self.call("address.tree", {"prefix": prefix})

    def describe(self, path: str) -> dict[str, Any]:
        return self.call("address.describe", {"path": path})

    def find(self, query: str) -> dict[str, Any]:
        return self.call("address.find", {"query": query})

    def inspect_stats(self, path: str) -> dict[str, Any]:
        return self.call("inspect", {"path": path, "stats": True})

    def diff(self, path: str, left_cut: str, right_cut: str) -> dict[str, Any]:
        return self.call(
            "inspect.diff",
            {
                "path": path,
                "left_cut": self.resolve_cut(left_cut),
                "right_cut": self.resolve_cut(right_cut),
            },
        )

    def cuts(self) -> dict[str, Any]:
        result = self.call("cuts")
        return result

    def frames(self, node_id: str | None = None) -> dict[str, Any]:
        """Inspect attached source scopes and their executable StateCut links."""
        return self.call("frames", {"node_id": node_id})

    def why(self, path: str) -> dict[str, Any]:
        return self.call("why", {"path": path})

    def xref(self, path: str) -> dict[str, Any]:
        return self.call("xref", {"path": path})

    def reverse_continue(self, predicate: str) -> dict[str, Any]:
        return self.call("reverse-continue", {"predicate": predicate})

    def first_divergence(self, left_cut: str, right_cut: str) -> dict[str, Any]:
        return self.call(
            "first-divergence",
            {"left_cut": self.resolve_cut(left_cut), "right_cut": self.resolve_cut(right_cut)},
        )

    def attribute(self, path: str) -> dict[str, Any]:
        return self.call("attribute", {"path": path})

    def promote(self, cut_id: str) -> dict[str, Any]:
        return self.call("promote", {"cut_id": self.resolve_cut(cut_id)})

    def doctor(self) -> dict[str, Any]:
        return self.call("doctor")

    def abort_all(self) -> dict[str, Any]:
        # Always ask the resident worker.  The local shell can be stale after
        # a session reload, while the worker remains the authority on pending
        # Acts and lease-safe continuation.
        result = self.call("act.abort_all")
        self._pending_act = None
        return result

    def reconnect(self) -> dict[str, Any]:
        previous = dict(self._capability_snapshot)
        capability = self.capabilities()
        for field in ("session_id", "family", "backend", "state_schema_id", "source_fingerprint"):
            expected = previous.get(field)
            observed = capability.get(field)
            if expected not in {None, "", "unknown"} and observed not in {None, "", "unknown"}:
                if expected != observed:
                    raise ValueError(f"reconnected worker {field} does not match the session")
        result = self.status()
        result["capability"] = capability
        result["reconnected"] = True
        return result

    def preview(self) -> dict[str, Any]:
        if self._pending_act is None:
            return {"pending_act": False, "reason": "no pending Act"}
        return {"pending_act": True, "act": self._pending_act.to_dict()}

    def set_value(
        self,
        path: str,
        value: Any,
        *,
        sign: str = "replace",
        address: dict[str, Any] | None = None,
        struct_name: str | None = None,
    ) -> dict[str, Any]:
        if self._pending_act is not None:
            raise RuntimeError("an Act is already pending; commit or abort it first")
        status = self.status()
        pause_reason = status.get("pause_reason") or {}
        runtime = status.get("runtime") or {}
        self._current_cut = (
            (pause_reason.get("cut") or {}).get("cut_id")
            or runtime.get("cut_id")
            or self._current_cut
        )
        if not self._current_cut:
            raise RuntimeError("set requires a current StateCut; pause or checkpoint first")
        act = ActSpec(
            path=path,
            value=value,
            parent_cut=self._current_cut,
            sign=sign,
            address=self._act_address(path, address),
            struct_name=struct_name,
        )
        self._pending_act = act
        try:
            return self.call("act.preview", {"act": act.to_dict()})
        except Exception:
            if self._pending_act is act:
                self._pending_act = None
            raise

    def set_expression(
        self,
        path: str,
        expression: str,
        *,
        sign: str = "replace",
        address: dict[str, Any] | None = None,
        struct_name: str | None = None,
    ) -> dict[str, Any]:
        if self._pending_act is not None:
            raise RuntimeError("an Act is already pending; commit or abort it first")
        status = self.status()
        pause_reason = status.get("pause_reason") or {}
        runtime = status.get("runtime") or {}
        self._current_cut = (
            (pause_reason.get("cut") or {}).get("cut_id")
            or runtime.get("cut_id")
            or self._current_cut
        )
        if not self._current_cut:
            raise RuntimeError("set-expr requires a current StateCut; pause or checkpoint first")
        act = ActSpec(
            path=path,
            value=None,
            parent_cut=self._current_cut,
            sign=sign,
            expression=expression,
            address=self._act_address(path, address),
            struct_name=struct_name,
        )
        self._pending_act = act
        try:
            return self.call("act.preview", {"act": act.to_dict()})
        except Exception:
            if self._pending_act is act:
                self._pending_act = None
            raise

    def set_conditioner_row(
        self,
        stream: str,
        site: str,
        step: int,
        row: int,
        value: Any,
        *,
        sign: str = "replace",
        expression: str | None = None,
    ) -> dict[str, Any]:
        path = conditioner_row_path(stream, site, step, row)
        address = {
            "kind": "conditioner-row",
            "stream": stream,
            "site": site,
            "step": int(step),
            "row": int(row),
        }
        if expression is not None:
            return self.set_expression(path, expression, sign=sign, address=address)
        return self.set_value(path, value, sign=sign, address=address)

    def commit(self) -> dict[str, Any]:
        if self._pending_act is None:
            raise RuntimeError("no pending Act")
        result = self.call("act.commit", {"act_id": self._pending_act.act_id})
        self._current_cut = result.get("cut_id") or self._current_cut
        self._pending_act = None
        return result

    def abort(self) -> dict[str, Any]:
        if self._pending_act is None:
            return {"aborted": False, "reason": "no pending Act"}
        result = self.call("act.abort", {"act_id": self._pending_act.act_id})
        self._pending_act = None
        return result

    def write_receipt(self, path: str | Path | None = None) -> dict[str, Any]:
        previous: str | None = None
        chain: list[dict[str, Any]] = []
        for entry in self._receipt:
            material = {"previous": previous, "entry": entry}
            entry_copy = dict(entry)
            entry_copy["chain_previous"] = previous
            entry_copy["chain_sha256"] = hashlib.sha256(
                json.dumps(
                    material, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest()
            previous = entry_copy["chain_sha256"]
            chain.append(entry_copy)
        receipt = {
            "schema": "saturn-debugger-session-receipt-v1",
            "job_id": self.job_id,
            "session_id": self.session_id,
            "current_cut": self._current_cut,
            "commands": chain,
            "cuts": list(self._cuts.values()),
            "cut_names": dict(self._cut_names),
            "receipt_sha256": digest(
                {"job_id": self.job_id, "session_id": self.session_id, "commands": chain}
            ),
        }
        if path is not None:
            destination = Path(path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return receipt

    def save_session(
        self,
        path: str | Path,
        *,
        shell_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        capability = dict(self._capability_snapshot)
        if not capability:
            try:
                capability = dict(self.capabilities())
            except Exception:  # noqa: BLE001 - retain a usable local transcript
                capability = {}
        receipt = self.write_receipt()
        payload = {
            "schema": "saturn-debugger-session-v2",
            "job_id": self.job_id,
            "session_id": self.session_id,
            "current_cut": self._current_cut,
            "cuts": list(self._cuts.values()),
            "cut_names": dict(self._cut_names),
            "aliases": dict(self._aliases),
            "capability": capability,
            "pending_act": self._pending_act.to_dict() if self._pending_act else None,
            "shell": dict(shell_state or {}),
            "receipt": receipt,
        }
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._session_dir = destination.parent
        return {"saved": True, "path": str(destination), "sha256": digest(payload)}

    def load_session(self, path: str | Path) -> dict[str, Any]:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") not in {
            "saturn-debugger-session-v1",
            "saturn-debugger-session-v2",
        }:
            raise ValueError("unsupported debugger session schema")
        if str(payload.get("job_id")) != self.job_id:
            raise ValueError("saved session belongs to a different job")
        saved_capability = dict(payload.get("capability") or {})
        current_capability = dict(self.capabilities()) if saved_capability else {}
        saved_session_id = payload.get("session_id")
        if (
            saved_session_id
            and current_capability.get("session_id")
            and str(saved_session_id) != str(current_capability["session_id"])
        ):
            raise ValueError("saved session belongs to a different debugger session")
        for field in (
            "family",
            "backend",
            "state_schema_id",
            "source_fingerprint",
            "consumer",
        ):
            expected = saved_capability.get(field)
            observed = current_capability.get(field)
            if expected not in {None, "", "unknown"} and observed not in {None, "", "unknown"}:
                if expected != observed:
                    raise ValueError(
                        f"saved session {field} does not match the attached worker"
                    )
        saved_cuts = {
            str(item["cut_id"]): dict(item)
            for item in payload.get("cuts", ())
            if isinstance(item, dict) and item.get("cut_id")
        }
        remote_cuts = self.cuts() if payload.get("current_cut") else {"cuts": []}
        remote_ids = {
            str(item.get("cut_id"))
            for item in remote_cuts.get("cuts", ())
            if isinstance(item, dict) and item.get("cut_id")
        }
        current_cut = payload.get("current_cut")
        if current_cut and str(current_cut) not in remote_ids:
            raise ValueError(
                f"saved current StateCut is not present on the attached worker: {current_cut}"
            )
        self.session_id = current_capability.get("session_id") or payload.get("session_id")
        self._capability_snapshot = current_capability or saved_capability
        self._current_cut = current_cut
        self._cuts = saved_cuts
        self._cut_names = {str(k): str(v) for k, v in (payload.get("cut_names") or {}).items()}
        self._aliases = {str(k): str(v) for k, v in (payload.get("aliases") or {}).items()}
        raw_pending = payload.get("pending_act")
        self._pending_act = ActSpec(**dict(raw_pending)) if isinstance(raw_pending, dict) else None
        self._session_dir = source.parent
        return {
            "loaded": True,
            "path": str(source),
            "current_cut": self._current_cut,
            "validated_against": {
                "session_id": self.session_id,
                "capability": bool(current_capability or saved_capability),
                "remote_cut_count": len(remote_ids),
            },
            "pending_act": self._pending_act.to_dict() if self._pending_act else None,
            "shell": dict(payload.get("shell") or {}),
        }

    def finish(self, directory: str | Path | None = None) -> dict[str, Any]:
        if self._pending_act is not None:
            raise RuntimeError("cannot finish with a pending Act; commit or abort it first")
        try:
            capability = self.capabilities()
        except Exception:  # noqa: BLE001 - preserve transcript if the worker disconnects
            capability = {}
        root = (
            Path(directory).expanduser().resolve()
            if directory
            else (self._session_dir or Path.cwd() / ".saturn" / "debug" / self.job_id)
        )
        root.mkdir(parents=True, exist_ok=True)
        receipt_path = root / "session-receipt.json"
        receipt = self.write_receipt(receipt_path)
        custody = {
            "schema": "saturn-debugger-custody-manifest-v1",
            "job_id": self.job_id,
            "session_id": self.session_id,
            "artifacts": [{"path": str(receipt_path), "sha256": digest(receipt)}],
            "statecut_lineage": list(self._cuts.values()),
            "capability": {
                "digest": digest(capability),
                "model_identity": capability.get("model_identity", {}),
                "state_schema_id": capability.get("state_schema_id"),
                "backend": capability.get("backend"),
                "consumer": capability.get("consumer"),
                "source_fingerprint": capability.get("source_fingerprint"),
                "exact_replay": capability.get("exact_replay"),
                "optional_operations": capability.get("optional_operations", []),
            },
            "claim_boundary": (
                "debugger session transcript and declared lineage; not semantic proof"
            ),
        }
        custody["manifest_sha256"] = digest(custody)
        custody_path = root / "custody-manifest.json"
        custody_path.write_text(
            json.dumps(custody, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence = {
            "schema": "saturn-debugger-evidence-sweep-v1",
            "status": "complete",
            "transcript": str(receipt_path),
            "custody": str(custody_path),
            "commands": len(self._receipt),
            "cuts": len(self._cuts),
            "receipt_sha256": receipt["receipt_sha256"],
            "custody_sha256": custody["manifest_sha256"],
        }
        evidence_path = root / "evidence-summary.json"
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {
            "finished": True,
            "directory": str(root),
            "artifacts": [str(receipt_path), str(custody_path), str(evidence_path)],
        }

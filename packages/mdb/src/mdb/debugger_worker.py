"""Worker-side safe-point controller for instrumented mrun jobs."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .debugger_adapters import DebuggerAdapter
from .debugger_features import (
    DebuggerFeatureError,
    attach_stats,
    evaluate_predicate,
    provenance_for_path,
    search_capabilities,
    template_tree,
)
from .debugger_protocol import (
    ActSpec,
    BreakpointSpec,
    DebugCapability,
    DebuggerAddressSpec,
    DebuggerProtocolError,
    DebuggerStructSpec,
    DebugRequest,
    DebugResponse,
    WatchSpec,
    now_utc,
)


def _watch_signature(value: Any) -> str:
    """Make a bounded, deterministic comparison key for a watched value."""

    candidate = value
    for method in ("detach", "cpu"):
        callback = getattr(candidate, method, None)
        if callable(callback):
            try:
                candidate = callback()
            except Exception:  # noqa: BLE001 - a native value may not expose both steps
                break
    for method in ("tolist", "as_list"):
        callback = getattr(candidate, method, None)
        if callable(callback):
            try:
                candidate = callback()
                break
            except Exception:  # noqa: BLE001 - fall back to repr below
                pass
    try:
        return json.dumps(candidate, sort_keys=True, default=repr, allow_nan=False)
    except (TypeError, ValueError):
        return repr(candidate)


class DebugWorkerController:
    """Serialize debugger commands and expose a safe-point polling hook.

    Runtime loops call :meth:`safe_point` at typed boundaries.  Commands never
    interrupt a native kernel; a pause is acknowledged only after the adapter
    has captured a StateCut at the requested boundary.
    """

    def __init__(self, capability: DebugCapability, adapter: DebuggerAdapter, *, source_frame_stack: Any | None = None) -> None:
        self.capability = capability
        self.adapter = adapter
        self.source_frame_stack = source_frame_stack
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_client_sequence = 0
        self._last_request: dict[str, DebugResponse] = {}
        self._breakpoints: dict[str, BreakpointSpec] = {}
        self._watches: dict[str, WatchSpec] = {}
        self._paused = False
        self._pause_reason: dict[str, Any] | None = None
        self._pending_pause = False
        self._pending_act_id: str | None = None
        self._last_heartbeat = now_utc()
        self._cut_history: dict[str, dict[str, Any]] = {}
        self._cut_names: dict[str, str] = {}
        self._events: list[dict[str, Any]] = []
        self._watch_previous: dict[str, Any] = {}

    def handle(self, request: DebugRequest) -> DebugResponse:
        with self._lock:
            cached = self._last_request.get(request.request_id)
            if cached is not None:
                return cached
            self._sequence += 1
            try:
                if request.sequence is not None and request.sequence <= self._last_client_sequence:
                    raise DebuggerProtocolError(
                        "stale or out-of-order debugger request sequence; "
                        f"expected next sequence {self._last_client_sequence + 1}; "
                        "reconnect and resync"
                    )
                if request.sequence is not None:
                    self._last_client_sequence = request.sequence
                result = self._dispatch(request.operation, request.payload)
                response = DebugResponse(request.request_id, self._sequence, True, dict(result))
            except (DebuggerProtocolError, KeyError, TypeError, ValueError) as exc:
                response = DebugResponse(request.request_id, self._sequence, False, error=str(exc))
            except RuntimeError as exc:
                # Native BenchSession circuit methods use RuntimeError
                # subclasses for refused or incompatible measurements. Keep
                # legacy adapter error behavior while converting these two
                # new control operations into protocol refusals.
                serializable_runtime_errors = {
                    "measure_circuit", "replay_circuit", "memory_read", "memory_write",
                    "memory_diff", "memory_probe",
                }
                if request.operation not in serializable_runtime_errors:
                    raise
                response = DebugResponse(request.request_id, self._sequence, False, error=str(exc))
            self._last_request[request.request_id] = response
            return response

    def safe_point(
        self, boundary: str, index: int, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        with self._lock:
            self._last_heartbeat = now_utc()
            event = {"boundary": boundary, "index": index, "metadata": dict(metadata or {})}
            self._events.append(event)
            fired = [
                spec.to_dict()
                for spec in self._breakpoints.values()
                if spec.enabled and self._matches(spec, boundary, index, event)
            ]
            watch_fired = self._fired_watches(event)
            if fired or watch_fired or self._pending_pause:
                cut_name = "watchpoint" if watch_fired and not fired else "breakpoint"
                cut = self.adapter.checkpoint(cut_name)
                self._remember_cut(cut, name=cut_name, event=event)
                self._paused = True
                self._pending_pause = False
                self._pause_reason = {
                    "event": event,
                    "breakpoints": fired,
                    "watches": watch_fired,
                    "cut": cut,
                }
            return self.status()

    def _fired_watches(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        fired: list[dict[str, Any]] = []
        for spec in self._watches.values():
            try:
                self.adapter.validate_read(spec.path)
                value = self.adapter.resolve_value(spec.path)
            except (DebuggerProtocolError, KeyError, TypeError, ValueError):
                try:
                    inspection = self.adapter.inspect(spec.path)
                    value = inspection.get("value") if isinstance(inspection, Mapping) else None
                except Exception:  # noqa: BLE001 - an unavailable watch is not a worker crash
                    continue
            if value is None:
                continue
            previous = self._watch_previous.get(f"{spec.watch_id}:value")
            previous_key = self._watch_previous.get(spec.watch_id)
            try:
                matched = evaluate_predicate(
                    spec.predicate,
                    value,
                    {"previous": previous, **dict(event.get("metadata") or {})},
                )
            except DebuggerFeatureError:
                matched = False
            current_key = _watch_signature(value)
            was_active = bool(self._watch_previous.get(f"{spec.watch_id}:active", False))
            changed = previous_key is None or current_key != previous_key
            if spec.mode == "level":
                should_fire = matched
            elif spec.mode == "edge":
                should_fire = matched and not was_active
            else:
                should_fire = matched and changed
            self._watch_previous[f"{spec.watch_id}:value"] = value
            self._watch_previous[spec.watch_id] = current_key
            self._watch_previous[f"{spec.watch_id}:active"] = matched
            if should_fire:
                fired.append(
                    {
                        "watch_id": spec.watch_id,
                        "path": spec.path,
                        "predicate": spec.predicate,
                        "mode": spec.mode,
                        "boundary": event.get("boundary"),
                        "index": event.get("index"),
                    }
                )
        return fired

    def _remember_cut(
        self,
        result: Mapping[str, Any],
        *,
        name: str | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> None:
        cut_id = result.get("cut_id") or result.get("handle")
        if not cut_id:
            return
        cut = {
            **self._cut_history.get(str(cut_id), {}),
            **dict(result),
        }
        cut["cut_id"] = str(cut_id)
        if event is not None:
            cut.setdefault("event", dict(event))
        self._cut_history[str(cut_id)] = cut
        if name:
            self._cut_names[str(name)] = str(cut_id)
        if self._paused:
            self._pause_reason = {
                **dict(self._pause_reason or {}),
                "cut": cut,
            }

    def _cut_id(self, raw: str) -> str:
        aliases = {
            "@current": self._current_cut(),
            "@parent": self._parent_cut(),
            "@root": self._root_cut(),
        }
        resolved = aliases.get(raw, self._cut_names.get(raw, raw))
        if not resolved:
            raise DebuggerProtocolError(f"no current StateCut for {raw}")
        return resolved

    def _current_cut(self) -> str | None:
        return (
            str((self._pause_reason or {}).get("cut", {}).get("cut_id"))
            if self._pause_reason
            else None
        )

    def _parent_cut(self) -> str | None:
        current = self._current_cut()
        if current and current in self._cut_history:
            parent = self._cut_history[current].get("parent_cut")
            return str(parent) if parent else None
        return None

    def _root_cut(self) -> str | None:
        if not self._cut_history:
            return None
        current = next(iter(self._cut_history))
        seen: set[str] = set()
        while current in self._cut_history and current not in seen:
            seen.add(current)
            parent = self._cut_history[current].get("parent_cut")
            if not parent or parent not in self._cut_history:
                return current
            current = str(parent)
        return current

    def service_safe_point(
        self,
        boundary: str,
        index: int,
        metadata: Mapping[str, Any] | None = None,
        *,
        poller: Any | None = None,
        poll_s: float = 0.25,
    ) -> dict[str, Any]:
        """Poll mrun, evaluate breakpoints, and block safely while paused.

        This is the single hook runtime loops should call after completing a
        typed boundary and before entering the next native operation.
        """
        if poller is not None:
            poller.serve_once(self)
        status = self.safe_point(boundary, index, metadata)
        while status["state"] in {"paused", "pause-pending"}:
            if poller is None:
                time.sleep(poll_s)
            else:
                poller.serve_until_resumed(self, poll_s=poll_s)
            status = self.status()
        return status

    def _matches(
        self, spec: BreakpointSpec, boundary: str, index: int, event: Mapping[str, Any]
    ) -> bool:
        if spec.kind == "if":
            # Conditions are intentionally restricted to event metadata; the
            # worker must not eval arbitrary code from an attach client.
            expression = spec.expression or ""
            if "==" in expression:
                key, expected = (part.strip() for part in expression.split("==", 1))
                return str(event.get("metadata", {}).get(key, event.get(key))) == expected.strip(
                    "'\""
                )
            metadata = event.get("metadata", {})
            if isinstance(metadata, Mapping):
                for key, value in metadata.items():
                    prefix = f"{key} "
                    if expression.startswith(prefix):
                        try:
                            return evaluate_predicate(
                                expression[len(prefix) :],
                                value,
                                metadata,
                            )
                        except DebuggerFeatureError:
                            return False
            return False
        return spec.kind == boundary and spec.index == index

    def _dispatch(self, operation: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if operation == "capability":
            return self._capability_payload()
        if operation in {"capabilities", "breakable"}:
            capability = self._capability_payload()
            return {
                "family": capability["family"],
                "boundaries": capability["boundaries"],
                "boundary_catalog": capability.get("boundary_catalog", []),
                "inspectable_slots": capability["inspectable_slots"],
                "writable_slots": capability["writable_slots"],
                "address_specs": capability.get("address_specs", []),
                "structs": capability.get("structs", []),
                "algebra_ops": capability.get("algebra_ops", []),
                "optional_operations": capability.get("optional_operations", []),
                "checkpoint_requires_pause": capability.get(
                    "checkpoint_requires_pause", False
                ),
            }
        if operation in {"status", "info"}:
            return self.status()
        if operation == "heartbeat":
            self._last_heartbeat = now_utc()
            return {"alive": True, "last_heartbeat": self._last_heartbeat}
        if operation == "pause":
            self._pending_pause = True
            return {"state": "pause-pending"}
        if operation in {"continue", "resume"}:
            if self._pending_act_id is not None:
                raise DebuggerProtocolError(
                    "cannot continue with a pending Act; commit or abort it first"
                )
            self._paused = False
            self._pause_reason = None
            return self.adapter.continue_run()
        if operation == "step":
            if not self._paused:
                raise DebuggerProtocolError("step requires a paused worker")
            count = int(payload.get("count", 1))
            kind = payload.get("kind")
            if kind:
                try:
                    result = self.adapter.optional_operation("step_kind", str(kind), count)
                except DebuggerProtocolError:
                    result = self.adapter.step(count)
                    result = {**dict(result), "requested_boundary": str(kind)}
            else:
                result = self.adapter.step(count)
            self._paused = True
            return result
        if operation == "breakpoint.add":
            spec = BreakpointSpec(**dict(payload))
            self.adapter.validate_breakpoint(spec)
            self._breakpoints[spec.breakpoint_id] = spec
            return {"breakpoint": spec.to_dict()}
        if operation == "breakpoint.list":
            return {"breakpoints": [x.to_dict() for x in self._breakpoints.values()]}
        if operation == "breakpoint.clear":
            self._breakpoints.pop(str(payload["breakpoint_id"]), None)
            return {"cleared": str(payload["breakpoint_id"])}
        if operation == "watch.add":
            spec = WatchSpec(**dict(payload))
            self.adapter.validate_watch(spec)
            self._watches[spec.watch_id] = spec
            return {"watch": spec.to_dict()}
        if operation == "watch.list":
            return {"watches": [x.to_dict() for x in self._watches.values()]}
        if operation == "watch.clear":
            watch_id = str(payload["watch_id"])
            self._watches.pop(watch_id, None)
            self._watch_previous.pop(f"{watch_id}:value", None)
            self._watch_previous.pop(watch_id, None)
            self._watch_previous.pop(f"{watch_id}:active", None)
            return {"cleared": watch_id}
        if operation == "struct.add":
            raw = payload.get("struct") or payload
            spec = DebuggerStructSpec.from_dict(raw)
            return dict(self.adapter.register_struct(spec))
        if operation == "struct.list":
            return {"structs": [item.to_dict() for item in self.adapter.structs()]}
        if operation == "address.add":
            raw = payload.get("address") or payload
            spec = DebuggerAddressSpec.from_dict(raw)
            return dict(self.adapter.register_address(spec))
        if operation == "address.list":
            return {"address_specs": [item.to_dict() for item in self.adapter.address_specs()]}
        if operation == "algebra.describe":
            return {
                "operations": list(self.adapter.capabilities.algebra_ops),
                "evaluation": "worker-side safe AST; no imports, attributes, subscripts, or eval",
            }
        if operation == "inspect":
            path = payload.get("path")
            self.adapter.validate_read(path)
            return attach_stats(
                self.adapter.inspect(path),
                enabled=bool(payload.get("stats", False)),
            )
        if operation == "inspect.diff":
            path = str(payload.get("path") or "")
            self.adapter.validate_read(path)
            left = str(payload.get("left_cut") or "")
            right = str(payload.get("right_cut") or "")
            return dict(
                self.adapter.optional_operation(
                    "diff", path, self._cut_id(left), self._cut_id(right)
                )
            )
        if operation == "address.tree":
            return {
                "prefix": payload.get("prefix"),
                "addresses": template_tree(self._capability_payload(), payload.get("prefix")),
            }
        if operation == "address.describe":
            path = str(payload.get("path") or "")
            self.adapter.validate_read(path)
            info = self.adapter.address_info(path)
            if info is None:
                return {"path": path, "slot": path, "declared": False}
            spec, fields = info
            return {
                "path": path,
                "declared": True,
                "address": spec.to_dict(),
                "fields": fields,
                "inspection": dict(self.adapter.inspect(path)),
            }
        if operation == "address.find":
            return {
                "query": payload.get("query"),
                "results": search_capabilities(
                    str(payload.get("query") or ""), self._capability_payload()
                ),
            }
        if operation in {"why", "xref"}:
            path = str(payload.get("path") or "")
            self.adapter.validate_read(path)
            hook = getattr(self.adapter, "optional_operation", None)
            if callable(hook):
                try:
                    return dict(hook(operation.replace("-", "_"), path))
                except DebuggerProtocolError:
                    pass
            return provenance_for_path(path, self._capability_payload())
        if operation in {"reverse-continue", "first-divergence", "bisect", "attribute", "promote"}:
            hook_args: list[Any] = []
            if operation == "reverse-continue":
                hook_args = [str(payload.get("predicate") or "")]
            elif operation in {"first-divergence", "bisect"}:
                hook_args = [
                    str(payload.get("left_cut") or ""),
                    str(payload.get("right_cut") or ""),
                ]
            elif operation == "attribute":
                hook_args = [str(payload.get("path") or "")]
            elif operation == "promote":
                hook_args = [str(payload.get("cut_id") or "")]
            return dict(self.adapter.optional_operation(operation.replace("-", "_"), *hook_args))
        if operation in {"measure_circuit", "replay_circuit"}:
            if operation not in self.adapter.optional_operations():
                raise DebuggerProtocolError(
                    "adapter "
                    f"{self.capability.family} does not expose debugger operation: {operation}"
                )
            if not self._paused:
                raise DebuggerProtocolError(
                    f"{operation} requires a paused worker at a typed StateCut"
                )
            if self._pending_act_id is not None:
                raise DebuggerProtocolError(
                    f"cannot {operation} with a pending Act; commit or abort it first"
                )
            if operation == "measure_circuit":
                spec = payload.get("spec")
                if not isinstance(spec, Mapping):
                    raise DebuggerProtocolError("measure_circuit requires a JSON object spec")
                try:
                    spec = json.loads(json.dumps(dict(spec), allow_nan=False))
                except (TypeError, ValueError) as exc:
                    raise DebuggerProtocolError(
                        "measure_circuit spec must contain JSON values only"
                    ) from exc
                result = dict(self.adapter.optional_operation(operation, spec))
                self._remember_cut(result)
                return result
            evidence_ref = payload.get("evidence_ref")
            if not isinstance(evidence_ref, str) or not evidence_ref.strip():
                raise DebuggerProtocolError(
                    "replay_circuit requires a non-empty evidence_ref string"
                )
            result = dict(self.adapter.optional_operation(operation, evidence_ref.strip()))
            self._remember_cut(result)
            return result
        if operation in {"memory_read", "memory_write", "memory_diff", "memory_probe"}:
            if operation not in self.adapter.optional_operations():
                raise DebuggerProtocolError(
                    f"adapter {self.capability.family} does not expose debugger operation: "
                    f"{operation}"
                )
            if operation in {"memory_write", "memory_probe"}:
                if not self._paused:
                    raise DebuggerProtocolError(
                        f"{operation} requires a paused worker at a typed StateCut"
                    )
                if self._pending_act_id is not None:
                    raise DebuggerProtocolError(
                        f"cannot {operation} with a pending Act; commit or abort it first"
                    )
            spec = payload.get("spec", {})
            if not isinstance(spec, Mapping):
                raise DebuggerProtocolError(f"{operation} requires a JSON object spec")
            try:
                spec = json.loads(json.dumps(dict(spec), allow_nan=False))
            except (TypeError, ValueError) as exc:
                raise DebuggerProtocolError(
                    f"{operation} spec must contain JSON values only"
                ) from exc
            def resolve_cuts(value: Any) -> Any:
                if not isinstance(value, dict):
                    return value
                output = dict(value)
                for field in ("cut_id", "donor_cut", "left_cut", "right_cut"):
                    if isinstance(output.get(field), str):
                        output[field] = self._cut_id(output[field])
                if isinstance(output.get("intervention"), dict):
                    output["intervention"] = resolve_cuts(output["intervention"])
                return output
            result = dict(self.adapter.optional_operation(operation, resolve_cuts(spec)))
            if operation in {"memory_write", "memory_probe"}:
                self._remember_cut(result)
            return result
        if operation == "frames":
            if self.source_frame_stack is None:
                raise DebuggerProtocolError("this worker has no attached source frame stack")
            return self.source_frame_stack.debug_view(payload.get("node_id"))
        if operation == "cuts":
            result = {"cuts": list(self._cut_history.values()), "names": dict(self._cut_names)}
            if self.source_frame_stack is not None:
                result["source_frames"] = self.source_frame_stack.debug_view()
            return result
        if operation == "doctor":
            capability = self._capability_payload()
            checks = {
                "protocol": True,
                "instrumented": bool(capability.get("schema")),
                "typed_boundaries": bool(capability.get("boundaries")),
                "state_schema": capability.get("state_schema_id") not in {None, "unknown"},
                "exact_replay": bool(capability.get("exact_replay")),
                "native_consumer": capability.get("consumer") not in {None, "", "unknown"},
            }
            return {"ok": all(checks.values()), "checks": checks, "capability": capability}
        if operation == "checkpoint":
            if self.capability.checkpoint_requires_pause and not self._paused:
                raise DebuggerProtocolError(
                    "checkpoint requires a paused typed safe point; use pause, guided, or break"
                )
            result = dict(self.adapter.checkpoint(payload.get("name")))
            self._remember_cut(result, name=payload.get("name"))
            return result
        if operation == "fork":
            if self.capability.checkpoint_requires_pause and not self._paused:
                raise DebuggerProtocolError(
                    "fork requires a paused typed safe point; use pause, guided, or break"
                )
            if "fork_branch" in self.adapter.optional_operations():
                result = dict(self.adapter.optional_operation("fork_branch", payload.get("name") or "fork"))
            else:
                result = dict(self.adapter.checkpoint(payload.get("name") or "fork"))
            self._remember_cut(result, name=payload.get("name") or "fork")
            result["branch"] = True
            return result
        if operation == "restore":
            if not self._paused:
                raise DebuggerProtocolError("restore requires a paused worker")
            if self._pending_act_id is not None:
                raise DebuggerProtocolError(
                    "cannot restore with a pending Act; commit or abort it first"
                )
            cut_id = self._cut_id(str(payload["cut_id"]))
            result = dict(self.adapter.restore(cut_id))
            self._remember_cut(result)
            return result
        if operation == "replay":
            if not self._paused:
                raise DebuggerProtocolError("replay requires a paused worker")
            if self._pending_act_id is not None:
                raise DebuggerProtocolError(
                    "cannot replay with a pending Act; commit or abort it first"
                )
            if not self.capability.exact_replay:
                raise DebuggerProtocolError(
                    "exact replay is unavailable for this worker capability"
                )
            cut_id = self._cut_id(str(payload["cut_id"]))
            result = dict(self.adapter.replay(cut_id))
            self._remember_cut(result)
            if result.get("exact") is False:
                raise DebuggerProtocolError("native suffix refused exact replay")
            return result
        if operation == "act.preview":
            if self._pending_act_id is not None:
                raise DebuggerProtocolError(
                    "an Act preview is already pending; commit or abort it first"
                )
            act = ActSpec(**dict(payload["act"]))
            self.adapter.validate_write(act)
            if not self._paused:
                raise DebuggerProtocolError(
                    "Act preview requires a paused worker at a typed StateCut"
                )
            current_cut = self._current_cut()
            if current_cut is None:
                raise DebuggerProtocolError(
                    "Act preview requires an identified current StateCut"
                )
            if str(act.parent_cut) != current_cut:
                raise DebuggerProtocolError(
                    "Act parent_cut is stale or does not match the current paused StateCut: "
                    f"expected {current_cut!r}, got {act.parent_cut!r}"
                )
            effective = act
            algebra: dict[str, Any] | None = None
            if act.expression is not None:
                if not self.adapter.capabilities.algebra_ops:
                    raise DebuggerProtocolError("worker does not advertise debugger algebra")
                value = self.adapter.evaluate_expression(act.expression)
                effective = replace(act, value=value, expression=None)
                algebra = {
                    "expression": act.expression,
                    "evaluated_in_worker": True,
                    "operations": list(self.adapter.capabilities.algebra_ops),
                }
            result = dict(self.adapter.preview_act(effective))
            self._pending_act_id = act.act_id
            if algebra is not None:
                result["algebra"] = algebra
            return result
        if operation == "act.commit":
            if not self._paused:
                raise DebuggerProtocolError("Act commit requires a paused worker")
            act_id = str(payload["act_id"])
            if self._pending_act_id != act_id:
                raise DebuggerProtocolError("Act commit does not match the pending preview")
            result = dict(self.adapter.commit_act(act_id))
            self._remember_cut(result, name="act")
            self._pending_act_id = None
            return result
        if operation == "act.abort":
            if not self._paused:
                raise DebuggerProtocolError("Act abort requires a paused worker")
            act_id = str(payload["act_id"])
            if self._pending_act_id != act_id:
                raise DebuggerProtocolError("Act abort does not match the pending preview")
            result = dict(self.adapter.abort_act(act_id))
            self._remember_cut(result, name="aborted-act-parent")
            self._pending_act_id = None
            return result
        if operation == "act.abort_all":
            if self._pending_act_id is None:
                return {"aborted": False, "reason": "no pending Act"}
            result = dict(self.adapter.abort_act(self._pending_act_id))
            self._remember_cut(result, name="aborted-act-parent")
            self._pending_act_id = None
            return {**result, "aborted_all": True}
        if operation == "act.pending":
            return {
                "pending_act": self._pending_act_id is not None,
                "act_id": self._pending_act_id,
            }
        raise DebuggerProtocolError(f"unsupported debugger operation: {operation}")

    def status(self) -> dict[str, Any]:
        return {
            "state": "paused"
            if self._paused
            else ("pause-pending" if self._pending_pause else "running"),
            "capability": self._capability_payload(),
            "pause_reason": self._pause_reason,
            "runtime": dict(self.adapter.status()),
            "breakpoints": [x.to_dict() for x in self._breakpoints.values()],
            "watches": [x.to_dict() for x in self._watches.values()],
            "last_heartbeat": self._last_heartbeat,
            "pending_act": self._pending_act_id is not None,
            "cuts": list(self._cut_history.values()),
            "cut_names": dict(self._cut_names),
        }

    def _capability_payload(self) -> dict[str, Any]:
        """Return static protocol identity plus live adapter declarations."""

        payload = self.capability.to_dict()
        adapter_payload = self.adapter.capabilities_payload()
        payload["boundary_catalog"] = adapter_payload.get("boundary_catalog", [])
        payload["address_specs"] = adapter_payload.get("address_specs", [])
        payload["structs"] = adapter_payload.get("structs", [])
        payload["algebra_ops"] = adapter_payload.get("algebra_ops", [])
        payload["optional_operations"] = list(
            self.adapter.optional_operations()
            or payload.get("optional_operations", [])
        )
        if self.source_frame_stack is not None and "frames" not in payload["optional_operations"]:
            payload["optional_operations"].append("frames")
        payload["checkpoint_requires_pause"] = bool(
            payload.get("checkpoint_requires_pause", False)
            or adapter_payload.get("checkpoint_requires_pause", False)
        )
        return payload

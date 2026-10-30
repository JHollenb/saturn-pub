"""Resident scalar discovery controller over the native ATen replay adapter.

This module is framework-free until a bound adapter is invoked.  All front doors
use :meth:`DiscoverySession.call`; retained cuts are explicitly live-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any


class DiscoveryError(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _summary(value):
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        data = tensor.tolist()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "digest": _digest(data), "value": data}
    if isinstance(value, dict):
        return {key: _summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary(item) for item in value]
    return value


@dataclass
class _Branch:
    id: str
    runtime: Any
    driver: Any = None
    phase: str = "forward_active"
    forward_index: int = 0
    parent_cut: str | None = None
    head: str | None = None
    blocked: list = field(default_factory=list)
    suppressed: set = field(default_factory=set)


@dataclass
class _Epoch:
    id: str
    factory: Any
    reference_operations: int
    reference_forwards: int
    branches: dict = field(default_factory=dict)


@dataclass
class _Lane:
    id: str
    epochs: dict
    epoch_id: str
    branch_id: str = "main"
    revision: int = 0
    status: str = "ready"

    @property
    def epoch(self):
        return self.epochs[self.epoch_id]

    @property
    def branch(self):
        return self.epoch.branches[self.branch_id]


class DiscoverySession:
    """One finite resident lease; no physical batching or eager pause is implied.

    ``factory`` returns a fresh ReplayForward.  For ISA lanes it receives the
    driver's three fresh roots.  ``reference_operations`` is a declared upper
    bound charged/admitted *before* a native recording factory is called;
    ``reference_forwards`` is normally one for capture-first and zero for an
    archive-only replay.  The factory must not perform undisclosed native work.
    """

    DEFAULT_BUDGETS = {"max_native_operations": 1_000_000,
                       "max_model_forwards": 1_000, "max_branches": 256,
                       "max_commands": 10_000, "max_instructions": 1_000,
                       "max_wall_ms": 3_600_000}
    READS = {"capabilities", "status", "catalog", "inspect", "events"}

    def __init__(self, budgets=None):
        self.budgets = dict(self.DEFAULT_BUDGETS)
        self.budgets.update(budgets or {})
        for key, value in self.budgets.items():
            if key not in self.DEFAULT_BUDGETS or type(value) is not int or value < 0:
                raise DiscoveryError(f"invalid finite budget: {key}")
        self.work = {"native_operations": 0, "reference_native_operations": 0,
                     "replay_native_operations": 0, "model_forwards": 0,
                     "reference_forwards": 0, "logical_forward_rows": 0,
                     "branches": 0, "commands": 0, "architectural_commits": 0,
                     "replay_attempts": 0, "physical_batches": 0,
                     "preparation_operations": 0, "act_commits": 0}
        self.lanes = {}
        self.focus = None
        self.aliases = {}
        self.breakpoints = {}
        self.watches = {}
        self.cuts = {}
        self.events = []
        self._cache = {}
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self._sequence = 0

    def _event(self, kind, **fields):
        row = {"sequence": len(self.events), "kind": kind, **fields}
        self.events.append(row)
        return row

    def _limit(self, key, cost=1):
        counter = {"max_native_operations": "native_operations",
                   "max_model_forwards": "model_forwards", "max_branches": "branches",
                   "max_commands": "commands", "max_instructions": "architectural_commits"}[key]
        return self.work[counter] + cost > self.budgets[key]

    def _wall_exhausted(self):
        return (time.monotonic() - self._started) * 1000 >= self.budgets["max_wall_ms"]

    def _budget_cause(self, name, **fields):
        return {"kind": "budget", "budget": name, "blocking": True, **fields}

    def _new_runtime(self, epoch, driver):
        if driver is not None and driver.termination:
            return None
        if self._wall_exhausted():
            raise DiscoveryError("budget exhausted: max_wall_ms")
        for key, cost in (("max_native_operations", epoch.reference_operations),
                          ("max_model_forwards", epoch.reference_forwards)):
            if self._limit(key, cost):
                raise DiscoveryError(f"budget exhausted before reference capture: {key}")
        # Charge before invoking code, including failed physical attempts.  The
        # operation bound is conservative when capture ends in an exception.
        self.work["native_operations"] += epoch.reference_operations
        self.work["reference_native_operations"] += epoch.reference_operations
        self.work["model_forwards"] += epoch.reference_forwards
        self.work["reference_forwards"] += epoch.reference_forwards
        self.work["logical_forward_rows"] += 1
        attempt = self._event("physical_attempt", mode="record-replay" if
                              epoch.reference_forwards else "archive-replay",
                              admitted_reference_operations=epoch.reference_operations)
        runtime = epoch.factory(*driver.inputs()) if driver is not None else epoch.factory()
        # The bound is an execution admission contract, not a scientific gate.
        actual = len(runtime.trace["events"]) if epoch.reference_forwards else 0
        if actual > epoch.reference_operations:
            self._event("integrity_error", attempt=attempt["sequence"],
                        error="reference capture exceeded declared operation bound", actual=actual)
            self.work["native_operations"] += actual - epoch.reference_operations
            self.work["reference_native_operations"] += actual - epoch.reference_operations
            raise DiscoveryError("reference capture exceeded declared operation bound")
        if epoch.reference_forwards:
            # Release the unspent reservation, retaining every actually executed
            # primitive.  This is admission accounting, never execution rollback.
            self.work["native_operations"] -= epoch.reference_operations - actual
            self.work["reference_native_operations"] -= epoch.reference_operations - actual
        self._event("forward_created", attempt=attempt["sequence"],
                    pc=runtime.pc, reference_operations=actual)
        return runtime

    def add_lane(self, lane_id, factory, *, driver=None, reference_operations=0,
                 reference_forwards=0):
        with self._lock:
            if lane_id in self.lanes:
                raise DiscoveryError(f"lane already exists: {lane_id}")
            if not isinstance(lane_id, str) or not lane_id:
                raise DiscoveryError("lane_id must be a nonempty string")
            if any(type(v) is not int or v < 0 for v in
                   (reference_operations, reference_forwards)):
                raise DiscoveryError("reference work declarations must be nonnegative integers")
            if reference_forwards and not reference_operations:
                raise DiscoveryError("capture-first factories require a native operation bound")
            if self._limit("max_branches"):
                raise DiscoveryError("budget exhausted: max_branches")
            epoch = _Epoch("input_00", factory, reference_operations, reference_forwards)
            bound_driver = driver.fork() if driver is not None else None
            runtime = self._new_runtime(epoch, bound_driver)
            branch = _Branch("main", runtime, bound_driver,
                             phase="driver_terminated" if runtime is None else "forward_active")
            epoch.branches["main"] = branch
            lane = _Lane(lane_id, {epoch.id: epoch}, epoch.id,
                         status="completed" if runtime is None else "ready")
            self.lanes[lane_id] = lane
            self.work["branches"] += 1
            self.focus = self.focus or lane_id
            self._capture(lane, name="root")
            self._event("lane_created", target=self._target(lane))
            return self._frame(lane)

    def _target(self, lane):
        return {"lane_id": lane.id, "epoch_id": lane.epoch_id,
                "branch_id": lane.branch_id, "revision": lane.revision}

    def _frame(self, lane):
        branch = lane.branch
        return {**self._target(lane), "status": lane.status, "phase": branch.phase,
                "pc": branch.runtime.pc if branch.runtime is not None else None,
                "forward_index": branch.forward_index, "head_cut": branch.head,
                "clock": {"forward": branch.forward_index,
                          "aten_next": branch.runtime.pc if branch.runtime is not None else None,
                          "instruction": branch.driver.instruction_index if branch.driver else None},
                "driver": branch.driver.state() if branch.driver is not None else None,
                "blocking_causes": copy.deepcopy(branch.blocked),
                "restore_support": "live", "pending_act": None}

    def _capture(self, lane, name=None):
        branch = lane.branch
        self._sequence += 1
        cut_id = f"cut_{self._sequence:06d}"
        runtime = branch.runtime.fork() if branch.runtime is not None else None
        cut = {"id": cut_id, "name": name, "target": self._target(lane),
               "runtime": runtime, "driver": branch.driver.fork() if branch.driver else None,
               "phase": branch.phase, "forward_index": branch.forward_index,
               "blocked": copy.deepcopy(branch.blocked), "restore_support": "live"}
        self.cuts[cut_id] = cut
        branch.head = cut_id
        return cut_id

    def _cut_manifest(self, cut):
        return {key: copy.deepcopy(value) for key, value in cut.items()
                if key not in ("runtime", "driver")} | {
                    "pc": cut["runtime"].pc if cut["runtime"] is not None else None,
                    "driver": cut["driver"].state() if cut["driver"] else None}

    def _get_cut(self, selector, lane):
        if selector in self.cuts:
            cut = self.cuts[selector]
        else:
            matches = [cut for cut in self.cuts.values() if cut["name"] == selector
                       and cut["target"]["lane_id"] == lane.id
                       and cut["target"]["epoch_id"] == lane.epoch_id]
            if len(matches) != 1:
                raise DiscoveryError(f"cut is unknown or ambiguous: {selector}")
            cut = matches[0]
        if (cut["target"]["lane_id"], cut["target"]["epoch_id"]) != (lane.id, lane.epoch_id):
            raise DiscoveryError("cut belongs to a different lane/input epoch")
        return cut

    def _restore(self, lane, cut):
        branch = lane.branch
        branch.runtime = cut["runtime"].fork() if cut["runtime"] is not None else None
        branch.driver = cut["driver"].fork() if cut["driver"] is not None else None
        branch.phase, branch.forward_index = cut["phase"], cut["forward_index"]
        branch.blocked = copy.deepcopy(cut["blocked"])
        branch.suppressed.clear()
        lane.status = "completed" if branch.phase == "driver_terminated" else "paused"
        self._prime_watches(lane, reason="restore")

    def _lane(self, payload):
        target = payload.get("target", {})
        lane_id = target.get("lane_id", payload.get("lane_id", payload.get("lane", self.focus)))
        if lane_id not in self.lanes:
            raise DiscoveryError(f"unknown lane: {lane_id}")
        return self.lanes[lane_id]

    def _resolve(self, lane, selector):
        suffix = ""
        base_selector = selector
        if isinstance(selector, str):
            for candidate in (":input", ":output"):
                if selector.endswith(candidate):
                    base_selector, suffix = selector[:-len(candidate)], candidate
                    break
        alias = self.aliases.get(base_selector)
        if alias:
            if alias["program_binding"] != self._program_binding(lane):
                raise DiscoveryError("alias belongs to a different program binding")
            if suffix:
                return lane.branch.runtime.catalog.resolve(alias["target"]["id"] + suffix)
            return copy.deepcopy(alias["target"])
        if lane.branch.runtime is None:
            raise DiscoveryError("no active forward catalog")
        return lane.branch.runtime.catalog.resolve(selector)

    def _program_binding(self, lane):
        catalog = lane.branch.runtime.catalog if lane.branch.runtime else None
        return getattr(catalog, "program_id", None)

    def _watch_value(self, lane, watch):
        runtime = lane.branch.runtime
        if runtime is None:
            raise DiscoveryError("no active forward")
        if watch["program_binding"] != self._program_binding(lane):
            raise DiscoveryError("watch is inapplicable to this program binding")
        return _summary(runtime.read(watch["value_id"]))

    def _prime_watches(self, lane, reason):
        for watch in self.watches.values():
            if watch.get("lanes") and lane.id not in watch["lanes"]:
                continue
            key = (lane.id, lane.epoch_id, lane.branch_id)
            try:
                watch["baseline"][key] = self._watch_value(lane, watch)
            except (ValueError, KeyError):
                watch["baseline"].pop(key, None)
        self._event("watches_primed", target=self._target(lane), reason=reason)

    def _safe_causes(self, lane):
        branch, causes = lane.branch, []
        if branch.runtime is None:
            return causes
        pc = branch.runtime.pc
        boundaries = branch.runtime.catalog.boundaries_at(pc)
        for breakpoint in self.breakpoints.values():
            if not breakpoint["enabled"] or (breakpoint.get("lanes") and
                                               lane.id not in breakpoint["lanes"]):
                continue
            for boundary in boundaries:
                site = boundary.get("site_id", boundary.get("id"))
                occurrence = (breakpoint["id"], lane.epoch_id, lane.branch_id,
                              branch.forward_index, site, boundary["edge"], pc)
                edge = "entry" if breakpoint["edge"] == "enter" else breakpoint["edge"]
                if (site == breakpoint["site_id"] and boundary["edge"] in
                        (edge, "enter" if edge == "entry" else edge)
                        and occurrence not in branch.suppressed):
                    causes.append({"kind": "breakpoint", "id": breakpoint["id"],
                                   "boundary": copy.deepcopy(boundary),
                                   "occurrence": {**self._target(lane),
                                                  "forward_index": branch.forward_index,
                                                  "pc": pc, "site_id": site,
                                                  "edge": boundary["edge"]}})
                    branch.suppressed.add(occurrence)
        for watch in self.watches.values():
            if not watch["enabled"] or (watch.get("lanes") and lane.id not in watch["lanes"]):
                continue
            at = watch.get("at")
            if at and not any(b.get("site_id") == at["site_id"] and
                              b["edge"] == at["edge"] for b in boundaries):
                continue
            key = (lane.id, lane.epoch_id, lane.branch_id)
            try:
                value = self._watch_value(lane, watch)
            except (ValueError, KeyError):
                continue  # A future value is unavailable, never a reference read.
            old = watch["baseline"].get(key)
            watch["baseline"][key] = value
            if old is not None and old != value:
                causes.append({"kind": "watchpoint", "id": watch["id"],
                               "value_id": watch["value_id"], "before": old, "after": value})
        return causes

    def capabilities(self):
        return {"schema": "mdb-discovery-capabilities-v1", "interface": "discovery-v1",
                "execution_modes": ["record-replay"], "default_unit": "law",
                "motion_units": ["op", "math", "law", "update", "forward", "instruction"],
                "clock_domains": ["aten-event", "forward", "instruction"],
                "checkpoint_support": ["live"], "resident_scalar_lanes": True,
                "physical_batch_profiles": [], "direct_eager_pause": False,
                "instruction_drivers": ["r3-predicted-length"],
                "durable_retry": False, "arbitrary_python_scripts": False,
                "scalar_scheduler": "ordered-targets-safe-point-parking",
                "act_value_specs": ["zeros_like", "scale"],
                "observation_profiles": {"selected": "live replay values and native outputs"},
                "budgets": dict(self.budgets)}

    def call(self, operation, payload=None, *, request_id=None, expected_revision=None):
        payload = copy.deepcopy(payload or {})
        with self._lock:
            try:
                if not isinstance(operation, str) or not isinstance(payload, dict):
                    raise DiscoveryError("operation must be text and payload must be an object")
                if request_id is not None and not isinstance(request_id, str):
                    raise DiscoveryError("request_id must be text")
                digest = _digest({"operation": operation, "payload": payload,
                                  "expected_revision": expected_revision})
            except (ValueError, TypeError) as error:
                return {"ok": False, "status": "invalid_request", "error": str(error)}
            if request_id is not None and request_id in self._cache:
                previous_digest, response = self._cache[request_id]
                if digest != previous_digest:
                    return {"ok": False, "status": "request_conflict",
                            "error": "request ID reused with different command content"}
                return copy.deepcopy(response)
            # Cached retries perform no command work.  Fresh read-only commands
            # remain available after exhaustion so a stopped head is inspectable,
            # but they still count toward the finite command allowance.
            self.work["commands"] += 1
            try:
                response = self._dispatch(operation, payload, expected_revision)
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                response = {"ok": False, "status": "error", "error": str(error),
                            "work": dict(self.work)}
            journal_response = (dict(ok=response.get("ok"), cursor=response.get("cursor"),
                                     event_count=len(response.get("events", [])))
                                if operation.removeprefix("discovery.") == "events" else
                                copy.deepcopy(response))
            self._event("command", operation=operation, payload=payload, request_id=request_id,
                        digest=digest, response=journal_response)
            if request_id is not None:
                self._cache[request_id] = (digest, copy.deepcopy(response))
            return response

    def _dispatch(self, operation, payload, expected_revision):
        op = operation.removeprefix("discovery.")
        shortcuts = {"capture": ("checkpoint", "capture"), "restore": ("checkpoint", "restore"),
                     "fork": ("branch", "fork"), "select": ("branch", "select"),
                     "apply": ("act", "commit")}
        if op in shortcuts:
            op, action = shortcuts[op]
            payload.setdefault("action", action)
        if op == "capabilities":
            return {"ok": True, **self.capabilities()}
        if op == "status" or (op == "lane" and payload.get("action", "list") == "list"):
            return {"ok": True, "focus": self.focus,
                    "lanes": [self._frame(lane) for lane in self.lanes.values()],
                    "work": dict(self.work), "budgets": dict(self.budgets)}
        if op == "events":
            after = payload.get("after", -1)
            return {"ok": True, "events": copy.deepcopy(self.events[after + 1:]),
                    "cursor": len(self.events) - 1}
        if op == "motion" and (payload.get("all") or payload.get("lanes")):
            return self._motion_group(payload)
        lane = self._lane(payload)
        action = payload.get("action", ("add" if any(key in payload for key in
                                                   ("selector", "port", "name")) else "list")
                             if op in ("alias", "break", "watch") else "")
        readonly = op in self.READS or (op in ("alias", "break", "watch", "checkpoint", "branch")
                                       and action == "list")
        if op == "lane" and action == "select":
            self.focus = lane.id
            return {"ok": True, "lane": self._frame(lane), "work": dict(self.work)}
        if not readonly:
            target = payload.get("target", {})
            expected = target.get("expected_revision", payload.get("expected_revision", expected_revision))
            if expected is not None and expected != lane.revision:
                return {"ok": False, "status": "revision_conflict", "lane": self._frame(lane),
                        "current_revision": lane.revision, "head_cut": lane.branch.head}
            for key, actual in (("epoch_id", lane.epoch_id), ("branch_id", lane.branch_id)):
                supplied = target.get(key, payload.get(key))
                if supplied is not None and supplied != actual:
                    return {"ok": False, "status": "target_conflict", "lane": self._frame(lane)}
            if (self.work["commands"] > self.budgets["max_commands"]
                    and not (op == "checkpoint" and action == "capture")):
                return self._result(lane, [self._budget_cause("max_commands")], status="stopped")
            if lane.status == "closed":
                raise DiscoveryError("lane is closed")
        if op == "catalog":
            if payload.get("selector", payload.get("target_name")):
                result = self._resolve(lane, payload.get("selector", payload.get("target_name")))
            else:
                result = lane.branch.runtime.catalog.sites if lane.branch.runtime else []
                if payload.get("kind"):
                    result = [site for site in result if site["kind"] == payload["kind"]]
            return {"ok": True, "catalog": copy.deepcopy(result),
                    "program_binding": self._program_binding(lane)}
        if op == "inspect":
            selector = payload.get("selector", payload.get("port", "state"))
            source = payload.get("source", "live")
            if source != "live":
                raise DiscoveryError("recorded-history inspection is not supported by this controller")
            if selector in ("state", "lane"):
                result = self._frame(lane)
            elif selector == "outputs":
                result = _summary(lane.branch.runtime.outputs())
            elif selector in ("live", "frontier"):
                result = lane.branch.runtime.inspect()
            else:
                resolved = self._resolve(lane, selector)
                value_id = resolved.get("value_id", resolved.get("id"))
                result = _summary(lane.branch.runtime.read(value_id))
            return {"ok": True, "source": "live", "value": result, "lane": self._frame(lane)}
        if op == "alias":
            if action == "list":
                return {"ok": True, "aliases": copy.deepcopy(list(self.aliases.values()))}
            name = payload["name"]
            if action in ("remove", "delete"):
                del self.aliases[name]
            else:
                old = self.aliases.get(name)
                resolved = self._resolve(lane, payload.get("selector", payload.get("target_name")))
                self.aliases[name] = {"name": name, "namespace": "session",
                                      "target": resolved, "revision": old["revision"] + 1 if old else 1,
                                      "program_binding": self._program_binding(lane),
                                      "provenance": payload.get("provenance", "typed-command")}
            lane.revision += 1
            return self._result(lane, [], alias=copy.deepcopy(self.aliases.get(name)))
        if op in ("break", "watch"):
            table = self.breakpoints if op == "break" else self.watches
            if action == "list":
                rows = [{k: copy.deepcopy(v) for k, v in row.items() if k != "baseline"}
                        for row in table.values()]
                return {"ok": True, op + "points": rows}
            identifier = payload.get("id", payload.get("name", f"{op}_{len(table):03d}"))
            if action in ("remove", "delete"):
                del table[identifier]
            elif action in ("enable", "disable"):
                table[identifier]["enabled"] = action == "enable"
            else:
                selector = payload.get("selector", payload.get("port"))
                resolved = self._resolve(lane, selector)
                row = {"id": identifier, "selector": selector, "enabled": True,
                       "lanes": payload.get("lanes", [lane.id] if any(
                           key in payload for key in ("lane", "lane_id", "target")) else None),
                       "site_id": resolved.get("site_id", resolved["id"]),
                       "edge": payload.get("edge", "exit"), "resolved": resolved,
                       "alias_revision": self.aliases.get(selector, {}).get("revision")}
                if op == "watch":
                    if payload.get("predicate", "change") != "change":
                        raise DiscoveryError("only bounded change watches are supported")
                    row["value_id"] = resolved.get("value_id", resolved["id"])
                    row["program_binding"] = self._program_binding(lane)
                    row["baseline"] = {}
                    at = payload.get("at")
                    if at:
                        boundary = self._resolve(lane, at if isinstance(at, str) else at["selector"])
                        row["at"] = {"site_id": boundary.get("site_id", boundary["id"]),
                                     "edge": "exit" if isinstance(at, str) else at.get("edge", "exit")}
                table[identifier] = row
                if op == "watch":
                    self._prime_watches(lane, "armed")
            lane.revision += 1
            return self._result(lane, [], installed_id=identifier)
        if op == "checkpoint":
            if action == "list":
                return {"ok": True, "cuts": [self._cut_manifest(c) for c in self.cuts.values()
                                             if c["target"]["lane_id"] == lane.id]}
            if action == "restore":
                cut = self._get_cut(payload.get("cut", payload.get("name")), lane)
                self._restore(lane, cut)
            elif action not in ("capture", ""):
                raise DiscoveryError(f"unsupported checkpoint action: {action}")
            lane.revision += 1
            cut_id = self._capture(lane, payload.get("name") if action != "restore" else None)
            return self._result(lane, [], cut=cut_id)
        if op == "branch":
            if action == "list":
                return {"ok": True, "branches": [{"id": b.id, "head_cut": b.head,
                                                    "parent_cut": b.parent_cut}
                                                   for b in lane.epoch.branches.values()]}
            name = payload.get("name", payload.get("branch"))
            if action == "fork":
                if self._limit("max_branches"):
                    return self._result(lane, [self._budget_cause("max_branches")])
                if name in lane.epoch.branches or not isinstance(name, str) or not name:
                    raise DiscoveryError("new branch name must be unique and nonempty")
                cut = self._get_cut(payload.get("cut", payload.get("from", lane.branch.head)), lane)
                branch = _Branch(name, cut["runtime"].fork() if cut["runtime"] else None,
                                 cut["driver"].fork() if cut["driver"] else None,
                                 phase=cut["phase"], forward_index=cut["forward_index"],
                                 parent_cut=cut["id"], head=cut["id"], blocked=copy.deepcopy(cut["blocked"]))
                lane.epoch.branches[name] = branch
                self.work["branches"] += 1
            elif action == "select":
                if name not in lane.epoch.branches:
                    raise DiscoveryError(f"unknown branch: {name}")
                lane.branch_id = name
                lane.status = "completed" if lane.branch.phase == "driver_terminated" else "paused"
                self._prime_watches(lane, "branch-select")
            else:
                raise DiscoveryError(f"unsupported branch action: {action}")
            lane.revision += 1
            return self._result(lane, [], branch=name)
        if op == "lane":
            if action == "close":
                lane.status = "closed"
            elif action == "checkout-epoch":
                epoch_id = payload.get("epoch", payload.get("checkout_epoch"))
                branch_id = payload.get("branch", "main")
                if epoch_id not in lane.epochs or branch_id not in lane.epochs[epoch_id].branches:
                    raise DiscoveryError("unknown epoch/branch")
                lane.epoch_id, lane.branch_id = epoch_id, branch_id
                lane.status = "completed" if lane.branch.phase == "driver_terminated" else "paused"
                self._prime_watches(lane, "checkout-epoch")
            else:
                raise DiscoveryError("input replacement uses replace_input() with an admitted bound factory")
            lane.revision += 1
            return self._result(lane, [])
        if op == "act":
            if action not in ("commit", "apply", ""):
                raise DiscoveryError("only immediate adapter-owned atomic Acts are available")
            selector = payload.get("selector", payload.get("port"))
            resolved = self._resolve(lane, selector)
            value_id = resolved.get("value_id", resolved["id"])
            current = lane.branch.runtime.read(value_id)
            # JSON supplies a value, never a memory pointer or Python expression.
            spec = payload.get("value_spec")
            if spec is not None:
                if "value" in payload or not isinstance(spec, dict):
                    raise DiscoveryError("value_spec must be an object and excludes value")
                kind = spec.get("kind")
                if kind == "zeros_like" and set(spec) == {"kind"}:
                    self.work["preparation_operations"] += 1
                    value = current.new_zeros(current.shape)
                elif kind == "scale" and set(spec) == {"kind", "factor"}:
                    factor = spec["factor"]
                    if type(factor) not in (int, float):
                        raise DiscoveryError("scale factor must be a finite number")
                    try:
                        factor = float(factor)
                    except OverflowError as error:
                        raise DiscoveryError("scale factor must be a finite number") from error
                    if not math.isfinite(factor):
                        raise DiscoveryError("scale factor must be a finite number")
                    self.work["preparation_operations"] += 1
                    value = current * factor
                else:
                    raise DiscoveryError("unsupported value_spec; use zeros_like or finite scale")
            else:
                value = payload.get("value")
                if hasattr(current, "new_tensor"):
                    value = current.new_tensor(value)
            receipt = lane.branch.runtime.apply(value_id, value, sign=payload.get("sign", "replace"))
            self.work["act_commits"] += 1
            lane.revision += 1
            self._capture(lane)
            return self._result(lane, [], receipt=receipt, value_spec=copy.deepcopy(spec))
        if op == "motion":
            return self._motion(lane, payload)
        raise DiscoveryError(f"unsupported discovery operation: {op}")

    def _result(self, lane, causes, *, status="stopped", completed_units=0,
                completed_unit=False, hit_cut=None, **fields):
        return {"ok": status != "errored", "status": status, "lane": self._frame(lane),
                "target": self._target(lane), "completed_units": completed_units,
                "completed_unit": completed_unit, "stop_causes": copy.deepcopy(causes),
                "hit_cut": hit_cut, "final_head": lane.branch.head,
                "work": dict(self.work), **fields}

    def _motion(self, lane, payload):
        branch = lane.branch
        action, unit = payload.get("action", "step"), payload.get("unit", "law")
        if action == "pause":
            lane.status = "paused"
            lane.revision += 1
            return self._result(lane, [{"kind": "pause", "blocking": True}])
        if action not in ("step", "next", "finish", "into", "continue", "until"):
            raise DiscoveryError(f"unsupported motion action: {action}")
        if unit not in ("op", "math", "law", "update", "forward", "instruction"):
            raise DiscoveryError(f"unsupported motion unit: {unit}")
        count = payload.get("count", 1)
        if type(count) is not int or count < 1:
            raise DiscoveryError("motion count must be a positive finite integer")
        if branch.blocked:
            return self._result(lane, branch.blocked,
                                status="errored" if lane.status == "errored" else "stopped")
        if branch.phase == "driver_terminated":
            return self._result(lane, [{"kind": branch.driver.termination}], status="completed")
        if unit == "instruction" and branch.driver is None:
            raise DiscoveryError("instruction motion requires a bound ISA driver")
        command_budget = payload.get("budget", {})
        max_ops = command_budget.get("max_native_operations", payload.get("max_ops", 1_000_000))
        max_wall = command_budget.get("max_wall_ms", self.budgets["max_wall_ms"])
        if type(max_ops) is not int or max_ops < 0 or type(max_wall) is not int or max_wall < 0:
            raise DiscoveryError("motion budgets must be nonnegative finite integers")
        started, initial_work = time.monotonic(), self.work["native_operations"]
        completed, receipts, commits, causes = 0, [], [], []
        lane.status = "running"

        def budgets():
            result = []
            if self.work["native_operations"] - initial_work >= max_ops:
                result.append(self._budget_cause("command.max_native_operations"))
            if self._limit("max_native_operations"):
                result.append(self._budget_cause("max_native_operations"))
            if self._wall_exhausted() or (time.monotonic() - started) * 1000 >= max_wall:
                result.append(self._budget_cause("max_wall_ms"))
            return result

        target_pc, last_completed_pc, target_available = None, None, True
        try:
            while True:
                if branch.runtime is None or branch.phase == "input_ready":
                    remaining = max_ops - (self.work["native_operations"] - initial_work)
                    if lane.epoch.reference_operations > remaining:
                        causes = [self._budget_cause("command.max_native_operations",
                                                   required_capture_bound=lane.epoch.reference_operations)]
                        break
                    # Admit and construct first; failure keeps the previous
                    # completed heads and logical forward clock unchanged.
                    next_runtime = self._new_runtime(lane.epoch, branch.driver)
                    if branch.phase == "input_ready":
                        branch.forward_index += 1
                    branch.runtime = next_runtime
                    branch.phase = "forward_active"
                    self._prime_watches(lane, "new-forward")
                runtime = branch.runtime
                causes = self._safe_causes(lane)
                causes.extend(budgets())
                if runtime.complete and unit != "instruction" and (not receipts or target_pc is None):
                    branch.phase = "forward_complete"
                    causes.extend([{"kind": "commit_required", "blocking": True},
                                   {"kind": "forward_complete"}])
                    break
                if target_pc is None and action == "until":
                    site = self._resolve(lane, payload.get("selector", payload.get("until")))
                    edge = payload.get("edge", "exit")
                    target_pc = site["entry" if edge in ("entry", "enter") else "exit"]
                    if target_pc < runtime.pc:
                        raise DiscoveryError("until target is before the current frontier")
                elif target_pc is None and action != "continue":
                    try:
                        target_pc = (runtime.pc if runtime.complete else
                                     runtime.catalog.target(unit if unit != "instruction" else "forward",
                                                            runtime.pc, motion=action,
                                                            scope=(self._resolve(lane, payload["scope"])["id"]
                                                                   if payload.get("scope") else None)))
                        target_available = True
                    except ValueError as error:
                        if not str(error).startswith("no remaining "):
                            raise
                        # Transparent native operations can remain after the
                        # final selected component; finish them without calling
                        # forward completion another law or math occurrence.
                        target_pc = len(runtime.trace["events"])
                        target_available = False
                at_target = (target_pc is not None and runtime.pc >= target_pc
                             and runtime.pc != last_completed_pc and target_available)
                if at_target and unit != "instruction":
                    last_completed_pc = runtime.pc
                    completed += 1
                    if completed >= count or action == "until":
                        causes.append({"kind": "motion_target", "unit": unit})
                    else:
                        target_pc = None
                if runtime.complete:
                    branch.phase = "forward_complete"
                    if unit == "instruction" and not causes:
                        if self._limit("max_instructions"):
                            causes.append(self._budget_cause("max_instructions"))
                        else:
                            commit = branch.driver.commit(runtime.outputs())
                            commits.append(commit)
                            if not commit["committed"]:
                                causes.extend(commit["stop_causes"])
                                branch.phase = "effect_blocked"
                                branch.blocked = copy.deepcopy(causes)
                            else:
                                self.work["architectural_commits"] += 1
                                completed += 1
                                if branch.driver.termination:
                                    branch.phase = "driver_terminated"
                                    causes.append({"kind": branch.driver.termination})
                                else:
                                    branch.phase = "input_ready"
                                if completed >= count:
                                    causes.append({"kind": "motion_target", "unit": "instruction"})
                                if not causes:
                                    target_pc = None
                                    continue
                    elif unit != "instruction":
                        if completed < count or action == "continue":
                            causes.append({"kind": "commit_required", "blocking": True})
                        causes.append({"kind": "forward_complete"})
                    if causes:
                        break
                if causes:
                    break
                if branch.phase == "input_ready":
                    target_pc = None
                    continue
                self.work["native_operations"] += 1
                self.work["replay_native_operations"] += 1
                self.work["replay_attempts"] += 1
                self._event("native_attempt", target=self._target(lane),
                            forward_index=branch.forward_index, pc=runtime.pc)
                receipt = runtime.advance_one()
                receipts.append(receipt)
                branch.suppressed.clear()
                if runtime.complete:
                    branch.phase = "forward_complete"
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            if isinstance(error, DiscoveryError) and str(error).startswith("budget exhausted"):
                causes.append(self._budget_cause(str(error).rsplit(": ", 1)[-1]))
            else:
                causes.append({"kind": "execution_error", "error": str(error), "blocking": True})
                branch.blocked = copy.deepcopy(causes)
                lane.status = "errored"
        if lane.status != "errored":
            lane.status = "completed" if branch.phase == "driver_terminated" else "paused"
        lane.revision += 1
        hit_cut = self._capture(lane)
        self._event("stop", target=self._target(lane), hit_cut=hit_cut,
                    stop_causes=copy.deepcopy(causes), completed_units=completed,
                    completed_unit=completed >= count, work=dict(self.work))
        return self._result(lane, causes, status="errored" if lane.status == "errored" else
                            "completed" if completed >= count else "stopped",
                            completed_units=completed, completed_unit=completed >= count,
                            hit_cut=hit_cut, receipts=receipts, commits=commits)

    def replace_input(self, lane_id, factory, *, driver=None, expected_revision=None,
                      reference_operations=0, reference_forwards=0):
        """Trusted binding operation; old epochs and cuts remain retained."""
        with self._lock:
            lane = self.lanes[lane_id]
            if expected_revision is not None and lane.revision != expected_revision:
                return {"ok": False, "status": "revision_conflict", "lane": self._frame(lane)}
            if lane.status in ("running", "closed"):
                raise DiscoveryError("input replacement requires a stopped open lane")
            if self._limit("max_branches"):
                raise DiscoveryError("budget exhausted: max_branches")
            epoch_id = f"input_{len(lane.epochs):02d}"
            epoch = _Epoch(epoch_id, factory, reference_operations, reference_forwards)
            bound_driver = driver.fork() if driver else None
            runtime = self._new_runtime(epoch, bound_driver)
            epoch.branches["main"] = _Branch("main", runtime, bound_driver,
                                             phase="driver_terminated" if runtime is None
                                             else "forward_active")
            lane.epochs[epoch_id] = epoch
            lane.epoch_id, lane.branch_id = epoch_id, "main"
            lane.status = "completed" if runtime is None else "ready"
            lane.revision += 1
            self.work["branches"] += 1
            self._prime_watches(lane, "input-replacement")
            self._capture(lane, "root")
            self._event("input_replaced", target=self._target(lane))
            return self._result(lane, [])

    def _motion_group(self, payload):
        """Freeze logical ownership before dispatch; scalar lanes park separately."""
        lane_ids = list(self.lanes) if payload.get("all") else list(payload["lanes"])
        if not lane_ids or len(lane_ids) != len(set(lane_ids)):
            raise DiscoveryError("motion target set must be nonempty and unique")
        if any(identifier not in self.lanes for identifier in lane_ids):
            raise DiscoveryError("motion target set contains an unknown lane")
        policy = payload.get("stop_policy", "group")
        if policy not in ("group", "lane"):
            raise DiscoveryError("stop_policy must be group or lane")
        targets = [self._target(self.lanes[identifier]) for identifier in lane_ids]
        self._event("group_accepted", targets=targets, stop_policy=policy)
        results, group_hit = [], None
        for target in targets:
            lane = self.lanes[target["lane_id"]]
            if lane.status in ("closed", "completed", "errored"):
                results.append(self._result(lane, [], status=lane.status))
                continue
            if group_hit is not None:
                lane.status = "paused"
                lane.revision += 1
                cause = {"kind": "group_pause", "blocking": True,
                         "owner": group_hit, "target_set": lane_ids}
                lane.branch.blocked = [cause]
                cut = self._capture(lane)
                results.append(self._result(lane, [cause], hit_cut=cut))
                continue
            command = {k: copy.deepcopy(v) for k, v in payload.items()
                       if k not in ("all", "lanes", "stop_policy")}
            command["target"] = {**target, "expected_revision": target["revision"]}
            command.pop("lane", None)
            command.pop("lane_id", None)
            result = self._dispatch("motion", command, target["revision"])
            results.append(result)
            stops = result.get("stop_causes", [])
            if policy == "group" and (not result["ok"] or any(
                    row.get("kind") not in ("motion_target", "forward_complete", "program_end",
                                             "instruction_limit") for row in stops)):
                group_hit = target["lane_id"]
        return {"ok": all(result["ok"] for result in results), "status": "stopped",
                "results": results, "lanes": [self._frame(self.lanes[i]) for i in lane_ids],
                "stop_causes": [dict(cause, lane_id=result["lane"]["lane_id"])
                                for result in results for cause in result.get("stop_causes", [])],
                "group_pause_owner": group_hit, "work": dict(self.work)}

"""Bounded discovery scripts over the same typed calls as the Python client.

This module is framework free. Scripts are data, never Python expressions.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import uuid
from pathlib import Path


class DiscoveryScriptError(ValueError):
    pass


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def parse_script(text, *, max_commands=2048):
    """Parse the explicit versioned dialect, with finite repeat and hit blocks."""
    lines = text.splitlines()
    meaningful = [(i + 1, line.strip()) for i, line in enumerate(lines)
                  if line.strip() and not line.lstrip().startswith("#")]
    if not meaningful or meaningful[0][1] != "interface mdb-discovery-v1":
        raise DiscoveryScriptError("script must begin with interface mdb-discovery-v1")

    def block(index, nested=False):
        commands = []
        while index < len(meaningful):
            line_no, line = meaningful[index]
            words = shlex.split(line)
            index += 1
            if words == ["end"]:
                if not nested:
                    raise DiscoveryScriptError(f"unexpected end at line {line_no}")
                return commands, index
            if words[0] in {"repeat", "on-hit"}:
                if len(words) != 2:
                    raise DiscoveryScriptError(f"invalid block at line {line_no}")
                body, index = block(index, True)
                if words[0] == "repeat":
                    try:
                        count = int(words[1])
                    except ValueError as error:
                        raise DiscoveryScriptError("repeat needs a finite integer") from error
                    if not 0 <= count <= max_commands or len(body) * count > max_commands:
                        raise DiscoveryScriptError("repeat exceeds script command bound")
                    commands.extend(body * count)
                else:
                    if nested:
                        raise DiscoveryScriptError("on-hit blocks cannot be nested")
                    commands.append({"handler": words[1], "commands": body, "line": line_no})
            else:
                commands.append({"command": line, "line": line_no})
            if len(commands) > max_commands:
                raise DiscoveryScriptError("script exceeds command bound")
        if nested:
            raise DiscoveryScriptError("unterminated block")
        return commands, index

    commands, _ = block(1)
    return {"schema": "mdb-discovery-script-v1", "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "commands": commands}


def compile_command(line):
    """Console sugar; ``call`` exposes every typed operation without new syntax."""
    if line.startswith("call "):
        command = json.loads(line[5:])
        if not isinstance(command, dict) or not isinstance(command.get("operation"), str):
            raise DiscoveryScriptError("call requires an operation and optional payload object")
        return command["operation"], dict(command.get("payload", {}))
    words = shlex.split(line)
    if not words:
        raise DiscoveryScriptError("empty command")
    verb, args = words[0], words[1:]
    if verb in {"status", "capabilities", "events", "catalog"} and not args:
        return verb, {}
    if verb in {"step", "next", "into", "finish", "continue", "run", "until"}:
        payload = {"action": "continue" if verb in {"continue", "run"} else verb}
        index = 0
        if args and not args[0].startswith("--"):
            payload["selector" if verb == "until" else "unit"] = args[0]
            index = 1
        while index < len(args):
            option = args[index]
            if option == "--defer":
                payload["defer"] = True
                index += 1
            elif option == "--all":
                payload["all"] = True
                index += 1
            elif option in {"--lane", "--unit", "--edge", "--count", "--max-ops", "--stop-policy", "--scope"}:
                if index + 1 == len(args):
                    raise DiscoveryScriptError(f"missing value for {option}")
                value = args[index + 1]
                key = {"--lane": "lane", "--unit": "unit", "--edge": "edge", "--count": "count",
                       "--max-ops": "max_ops", "--stop-policy": "stop_policy", "--scope": "scope"}[option]
                payload[key] = int(value) if key in {"count", "max_ops"} else value
                index += 2
            else:
                raise DiscoveryScriptError(f"unknown motion option: {option}")
        return "motion", payload
    if verb == "lane" and len(args) == 2 and args[0] == "focus":
        return "lane", {"action": "select", "lane_id": args[1]}
    if verb == "alias" and len(args) == 3 and args[1] == "=":
        return "alias", {"action": "bind", "name": args[0], "selector": args[2]}
    if verb == "break" and len(args) >= 2:
        payload = {"action": "add", "selector": args[0], "edge": args[1]}
        for i in range(2, len(args), 2):
            if i + 1 >= len(args) or args[i] not in {"--id", "--lane"}:
                raise DiscoveryScriptError("break accepts --id and --lane")
            payload[{"--id": "id", "--lane": "lane"}[args[i]]] = args[i + 1]
        return "break", payload
    if verb in {"capture", "restore", "fork", "select"} and len(args) == 1:
        return {"capture": "checkpoint", "restore": "checkpoint", "fork": "branch", "select": "branch"}[verb], {
            "action": verb, "cut" if verb == "restore" else "name": args[0]}
    if verb == "save-frontier" and len(args) == 1:
        return "checkpoint", {"action": "save", "store": args[0]}
    if verb == "inspect" and len(args) == 1:
        return "inspect", {"selector": args[0]}
    raise DiscoveryScriptError(f"unknown command: {line}; use call with a typed JSON payload")


class ScriptRunner:
    """One controller, deterministic handlers, bounded deferred continuation."""

    def __init__(self, session, *, journal=None, max_commands=2048, max_handler_hits=128):
        self.session, self.journal = session, journal
        self.max_commands, self.max_handler_hits = max_commands, max_handler_hits
        self.records, self.handlers = [], []
        self.run_id, self.calls, self.hits = uuid.uuid4().hex, 0, 0

    def _record(self, value):
        value = json.loads(json.dumps(value, allow_nan=False))
        self.records.append(value)
        if self.journal:
            self.journal.append(value)
        return value

    def _execute(self, item, *, target=None):
        if self.calls >= self.max_commands:
            raise DiscoveryScriptError("script command budget exhausted")
        operation, payload = compile_command(item["command"])
        payload.pop("defer", None)
        normalized = operation.removeprefix("discovery.")
        focus_change = normalized == "lane" and payload.get("action") in {"select", "focus"}
        explicit_lane = payload.get("lane_id", payload.get("lane"))
        if target and normalized not in {"status", "events", "capabilities"} and not focus_change and not explicit_lane:
            payload.setdefault("target", {k: target[k] for k in
                                          ("lane_id", "epoch_id", "branch_id", "expected_revision") if k in target})
        self.calls += 1
        identifier = f"script:{self.run_id}:{self.calls}"
        response = self.session.call(operation, payload, request_id=identifier)
        self._record({"schema": "mdb-discovery-command-record-v1", "line": item["line"],
                      "command": item["command"], "request_id": identifier,
                      "operation": operation, "payload": payload, "response": response})
        return response

    @staticmethod
    def _blocked(response):
        causes = response.get("stop_causes", [])
        blocking = {"budget", "cancellation", "cancelled", "effect_blocked", "unhandled_effect",
                    "integrity_error", "error", "group_pause", "program_end", "instruction_limit"}
        return not response.get("ok", False) or any(
            cause.get("blocking") or cause.get("kind") in blocking for cause in causes)

    @staticmethod
    def _target(response):
        target = dict(response.get("target", {}))
        lane = response.get("lane")
        if isinstance(lane, dict):
            target.update({k: lane[k] for k in ("lane_id", "epoch_id", "branch_id", "revision") if k in lane})
        target.update({k: response[k] for k in ("lane_id", "epoch_id", "branch_id", "revision") if k in response})
        if "revision" in target:
            target["expected_revision"] = target.pop("revision")
        return target

    def _handle(self, response):
        """Handlers run once per stop; only the outer loop may continue."""
        if "results" in response:
            outcomes = [self._handle(result) for result in response["results"]]
            return {**response, "ok": all(outcome.get("ok") for outcome in outcomes),
                    "results": outcomes}
        while response.get("ok"):
            causes = response.get("stop_causes", [])
            if not causes:
                return response
            ids = {c.get("id", c.get("breakpoint_id", c.get("watch_id"))) for c in causes}
            matched = [h for h in self.handlers if h["handler"] == "*" or h["handler"] in ids]
            if not matched:
                return response
            self.hits += 1
            if self.hits > self.max_handler_hits:
                raise DiscoveryScriptError("script handler-hit budget exhausted")
            hit = response
            target = self._target(hit)
            final = response
            deferred = False
            handler_failed = False
            for handler in matched:
                for item in handler["commands"]:
                    op, payload = compile_command(item["command"])
                    op = op.removeprefix("discovery.")
                    if op == "motion" and payload.get("defer"):
                        if payload.get("action") != "continue":
                            raise DiscoveryScriptError("handlers may defer only continue")
                        deferred = True
                        continue
                    if op == "motion":
                        raise DiscoveryScriptError("handler motion must use continue --defer")
                    action = self._execute(item, target=target)
                    final = action
                    if not action.get("ok"):
                        handler_failed = True
                        break
                    returned_target = self._target(action)
                    if returned_target.get("lane_id") == target.get("lane_id"):
                        target.update(returned_target)
                if handler_failed:
                    break
            if handler_failed:
                return final
            if not deferred or self._blocked(hit) or self._blocked(final):
                return {**hit, "target": target, "lane": final.get("lane", hit.get("lane")),
                        "final_head": final.get("final_head", hit.get("final_head")),
                        "hit_cut": hit.get("hit_cut")}
            response = self._execute({"line": 0, "command": "continue"}, target=target)
        return response

    def run(self, script):
        parsed = parse_script(script, max_commands=self.max_commands) if isinstance(script, str) else script
        if parsed.get("schema") != "mdb-discovery-script-v1":
            raise DiscoveryScriptError("unsupported script schema")
        self._record({"schema": "mdb-discovery-script-start-v1", "script_sha256": parsed["sha256"],
                      "run_id": self.run_id, "max_commands": self.max_commands})
        # Resolve syntax for the complete script before executing its first call.
        for item in parsed["commands"]:
            for command in item.get("commands", [item]):
                operation, payload = compile_command(command["command"])
                operation = operation.removeprefix("discovery.")
                if "handler" in item and operation == "motion" and not (
                    payload.get("action") == "continue" and payload.get("defer")
                ):
                    raise DiscoveryScriptError("handler motion must use continue --defer")
        for item in parsed["commands"]:
            if "handler" in item:
                self.handlers.append(item)
            else:
                response = self._execute(item)
                if not response.get("ok"):
                    return {"ok": False, "records": self.records, "last_response": response}
                response = self._handle(response)
                if not response.get("ok"):
                    return {"ok": False, "records": self.records, "last_response": response}
        return {"ok": True, "records": self.records, "commands_executed": self.calls,
                "handler_hits": self.hits, "status": self.session.call("status", {})}


class DiscoveryJournal:
    """Immutable metadata objects plus a verified chain; no frontier recovery claim."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "objects").mkdir(exist_ok=True)
        self.head_path = self.root / "head.json"
        self.head = json.loads(self.head_path.read_text()) if self.head_path.exists() else None
        self.verify()

    def append(self, record):
        body = {"schema": "mdb-discovery-journal-object-v1", "previous": self.head,
                "record": record}
        raw = _json(body)
        identifier = hashlib.sha256(raw).hexdigest()
        target = self.root / "objects" / (identifier + ".json")
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError("journal content-address collision")
        else:
            with target.open("xb") as stream:
                stream.write(raw)
        # Publish the referenced object before the head index.
        temporary = self.root / (".head-" + uuid.uuid4().hex)
        temporary.write_bytes(_json(identifier))
        temporary.replace(self.head_path)
        self.head = identifier
        return identifier

    def verify(self):
        current, seen, records = self.head, set(), []
        while current:
            if (not isinstance(current, str) or len(current) != 64
                    or any(c not in "0123456789abcdef" for c in current) or current in seen):
                raise ValueError("invalid journal chain")
            seen.add(current)
            raw = (self.root / "objects" / (current + ".json")).read_bytes()
            if hashlib.sha256(raw).hexdigest() != current:
                raise ValueError("journal object digest differs")
            body = json.loads(raw)
            if body.get("schema") != "mdb-discovery-journal-object-v1":
                raise ValueError("unsupported journal object")
            records.append(body["record"])
            current = body["previous"]
        return list(reversed(records))

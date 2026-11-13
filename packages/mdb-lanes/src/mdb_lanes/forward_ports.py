"""Authoritative native module ports, separate from inferred ATen math scopes.

The opt-in recorder binds invocation inputs/returns and declared state names to
anonymous SSA values. Shared module paths are declaration aliases: hooks cannot
identify which Python attribute selected a shared physical module. This module
reads sealed JSON only and never imports a model framework.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


class NativePortError(ValueError):
    """A native port is absent, ambiguous, or inconsistent with its trace."""


_TAGS = {"$value", "$tuple", "$dtype", "$device", "$torch", "$float"}


def pack_port_tree(value):
    """Protect literal native dict keys from semantic-annotation metadata keys."""
    if isinstance(value, list):
        return [pack_port_tree(item) for item in value]
    if isinstance(value, dict):
        if len(value) == 1 and next(iter(value)) in _TAGS:
            return {key: pack_port_tree(item) for key, item in value.items()}
        return {"$dict": [[key, pack_port_tree(item)] for key, item in value.items()]}
    return value


def unpack_port_tree(value):
    if isinstance(value, list):
        return [unpack_port_tree(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"$dict"}:
            pairs = value["$dict"]
            if not isinstance(pairs, list) or any(not isinstance(pair, list) or len(pair) != 2
                                                  or not isinstance(pair[0], str) for pair in pairs):
                raise NativePortError("malformed native dictionary encoding")
            if len({pair[0] for pair in pairs}) != len(pairs):
                raise NativePortError("duplicate native dictionary key")
            return {key: unpack_port_tree(item) for key, item in pairs}
        return {key: unpack_port_tree(item) for key, item in value.items()}
    return value


def _references(value):
    result = []
    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"$value"}:
                if item["$value"] not in result:
                    result.append(item["$value"])
            else:
                for nested in item.values():
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
    visit(value)
    return result


def _select(value, path):
    for part in path:
        if isinstance(value, dict) and set(value) == {"$tuple"}:
            value = value["$tuple"]
        try:
            value = value[part]
        except (KeyError, IndexError, TypeError) as error:
            raise NativePortError("native port path does not exist") from error
    return value


class NativePortCatalog:
    """Inspect captured ports and select the first real SSA consumers.

    Examples: ``resolve('module:layer:return', occurrence=1)``;
    ``resolve('module:layer:input', path=['kwargs', 'mask'])``;
    ``resolve('parameter:layer.weight')``; ``first_consumers(selector)``.
    Repeated physical calls require an occurrence or explicit call ID. A single
    unique tensor value has ``value_id`` for runtime.read/apply; trees always
    retain their complete encoded arguments and ordered unique ``values``.
    """

    def __init__(self, trace):
        from .forward_symbols import verify_trace

        verify_trace(trace)
        self.trace = copy.deepcopy(trace)
        self.trace_fingerprint = trace["fingerprint"]
        self._ports = copy.deepcopy(trace.get("metadata", {}).get("native_ports"))
        if not isinstance(self._ports, dict):
            raise NativePortError("trace has no native ports; record with capture_ports=True")
        if (self._ports.get("schema") != "mdb-native-ports-v1"
                or self._ports.get("encoding") != "trace-tags-with-ordered-dict-pairs-v1"):
            raise NativePortError("unsupported native port schema/encoding")
        self._modules = {row["id"]: row for row in self._ports.get("modules", [])}
        self._calls = {row["id"]: row for row in self._ports.get("calls", [])}
        if (len(self._modules) != len(self._ports.get("modules", []))
                or len(self._calls) != len(self._ports.get("calls", []))):
            raise NativePortError("duplicate native module/call identity")
        self._state = {}
        self._validate()

    def _check_refs(self, tree, pc):
        from .forward_symbols import _refs

        values = set(self.trace["values"])
        try:
            references = _refs(tree, values, "native port")
        except ValueError as error:
            raise NativePortError(str(error)) from error
        if any(self.trace["values"][key]["producer"] is not None
               and self.trace["values"][key]["producer"] >= pc for key in references):
            raise NativePortError("native port references a future SSA value")

    def _validate(self):
        paths, orders, occurrences = {}, set(), {}
        for key, module in self._modules.items():
            if (not isinstance(key, str) or not isinstance(module.get("paths"), list) or not module["paths"]
                    or module.get("canonical_path") != module["paths"][0]
                    or not isinstance(module.get("type"), str)):
                raise NativePortError("malformed declared native module")
            for path in module["paths"]:
                if not isinstance(path, str) or not path or path in paths:
                    raise NativePortError("ambiguous declared module path")
                paths[path] = key
        self._paths = paths
        for call in self._ports["calls"]:
            module_id = call.get("module_id")
            if module_id not in self._modules:
                raise NativePortError("native call names an unknown module")
            expected = occurrences.get(module_id, 0)
            if type(call.get("occurrence")) is not int or call["occurrence"] != expected:
                raise NativePortError("native call occurrence order differs")
            occurrences[module_id] = expected + 1
            entry, exit_ = call.get("entry_pc"), call.get("exit_pc")
            if (type(entry) is not int or type(exit_) is not int
                    or not 0 <= entry <= exit_ <= len(self.trace["events"])):
                raise NativePortError("native invocation PC interval is invalid")
            before, after = call.get("entry_order"), call.get("exit_order")
            if (type(before) is not int or type(after) is not int or not 0 <= before < after
                    or before in orders or after in orders):
                raise NativePortError("native invocation hook ordering is invalid")
            orders.update((before, after))
            parent = call.get("parent")
            if parent is not None:
                if parent not in self._calls:
                    raise NativePortError("native invocation parent is unknown")
                owner = self._calls[parent]
                if not (owner["entry_order"] < before < after < owner["exit_order"]
                        and owner["entry_pc"] <= entry <= exit_ <= owner["exit_pc"]):
                    raise NativePortError("native invocation is not inside its parent")
            if call.get("status") not in {"returned", "aborted"}:
                raise NativePortError("native invocation has no completed/aborted observation")
            if ("return" in call) != (call["status"] == "returned"):
                raise NativePortError("aborted invocation cannot claim a native return")
            for edge in ("args", "kwargs"):
                if edge not in call:
                    raise NativePortError("native invocation has no observed arguments")
                self._check_refs(unpack_port_tree(call[edge]), entry)
            if "return" in call:
                self._check_refs(unpack_port_tree(call["return"]), exit_)
        if orders != set(range(2 * len(self._calls))):
            raise NativePortError("native invocation hook clock is not contiguous")
        state = self._ports.get("state")
        if not isinstance(state, dict) or set(state) != {"parameters", "buffers"}:
            raise NativePortError("native state declaration inventory differs")
        for plural, kind in (("parameters", "parameter"), ("buffers", "buffer")):
            for row in state[plural]:
                name, value = row.get("name"), row.get("value_id")
                if not isinstance(name, str) or not name or (kind, name) in self._state:
                    raise NativePortError("duplicate or malformed declared state name")
                if value is not None:
                    descriptor = self.trace["values"].get(value)
                    if descriptor is None or descriptor["producer"] is not None or descriptor["origin"] != "parameter":
                        raise NativePortError("named state is not a declared read-only SSA root")
                self._state[kind, name] = value

    def _call(self, namespace, value, occurrence):
        if namespace == "call":
            if occurrence is not None or value not in self._calls:
                raise NativePortError("unknown call or occurrence supplied with an exact call ID")
            return self._calls[value]
        module_id = self._paths.get(value)
        if module_id is None:
            raise NativePortError("unknown declared native module: " + value)
        calls = [row for row in self._ports["calls"] if row["module_id"] == module_id]
        if occurrence is not None:
            if type(occurrence) is not int or occurrence < 0:
                raise NativePortError("occurrence must be a nonnegative integer")
            calls = [row for row in calls if row["occurrence"] == occurrence]
        if len(calls) != 1:
            raise NativePortError("native module has " + str(len(calls)) + " matching invocations; select an occurrence or call ID")
        return calls[0]

    def resolve(self, selector, *, occurrence=None, path=None):
        if not isinstance(selector, str):
            raise NativePortError("native port selector must be text")
        prefix, separator, rest = selector.partition(":")
        if not separator:
            raise NativePortError("native port selector requires module:, call:, parameter:, or buffer:")
        if prefix in {"parameter", "buffer"}:
            if occurrence is not None or path is not None or (prefix, rest) not in self._state:
                raise NativePortError("unknown named state or incompatible occurrence/path")
            value = self._state[prefix, rest]
            names = ([rest] if value is None else
                     [name for (kind, name), key in self._state.items() if kind == prefix and key == value])
            result = {"selector": selector, "kind": prefix, "name": rest, "available_pc": 0,
                      "provenance": self._ports["state_provenance"], "declared_aliases": names,
                      "encoded": None if value is None else {"$value": value}, "values": [] if value is None else [value],
                      "absent": value is None, "editable": False, "trace_fingerprint": self.trace_fingerprint}
        elif prefix in {"module", "call"}:
            target, separator, edge = rest.rpartition(":")
            if not separator or edge not in {"args", "kwargs", "input", "return", "output"}:
                raise NativePortError("native invocation selector requires :input, :args, :kwargs, :return, or :output")
            call = self._call(prefix, target, occurrence)
            edge = "return" if edge == "output" else edge
            if edge == "return" and call["status"] != "returned":
                raise NativePortError("native invocation aborted and has no observed return")
            encoded = ({"args": unpack_port_tree(call["args"]), "kwargs": unpack_port_tree(call["kwargs"])}
                       if edge == "input" else unpack_port_tree(call[edge]))
            if path is not None:
                if not isinstance(path, (list, tuple)):
                    raise NativePortError("native port path must be an explicit list/tuple")
                encoded = _select(encoded, path)
            module = self._modules[call["module_id"]]
            result = {"selector": selector, "kind": "native-module-port", "edge": edge,
                "call_id": call["id"], "module_id": call["module_id"], "module_path": module["canonical_path"],
                "declared_module_aliases": list(module["paths"]), "occurrence": call["occurrence"],
                "entry_pc": call["entry_pc"], "exit_pc": call["exit_pc"], "status": call["status"],
                "available_pc": call["exit_pc"] if edge == "return" else call["entry_pc"],
                "encoded": encoded, "values": _references(encoded), "trace_fingerprint": self.trace_fingerprint,
                "provenance": "actual native module invocation hook; separate from inferred ATen math scopes"}
        else:
            raise NativePortError("unknown native port namespace")
        if len(result["values"]) == 1:
            result["value_id"] = result["values"][0]
        return copy.deepcopy(result)

    def list(self, *, kind=None):
        modules = copy.deepcopy(self._ports["modules"])
        calls = copy.deepcopy(self._ports["calls"])
        for call in calls:
            for edge in ("args", "kwargs", "return"):
                if edge in call:
                    call[edge] = unpack_port_tree(call[edge])
        result = {"schema": "mdb-native-port-catalog-v1", "trace_fingerprint": self.trace_fingerprint,
                  "modules": modules, "calls": calls,
                  "parameters": [self.resolve("parameter:" + name) for kind, name in self._state if kind == "parameter"],
                  "buffers": [self.resolve("buffer:" + name) for kind, name in self._state if kind == "buffer"]}
        if kind is None:
            return result
        names = {"module": "modules", "call": "calls", "parameter": "parameters", "buffer": "buffers"}
        if kind not in names:
            raise NativePortError("unknown native catalog list kind")
        return result[names[kind]]

    def first_consumers(self, selector, *, occurrence=None, path=None, after=None):
        port = self.resolve(selector, occurrence=occurrence, path=path)
        boundary = port["available_pc"] if after is None else after
        if type(boundary) is not int or not 0 <= boundary <= len(self.trace["events"]):
            raise NativePortError("consumer boundary PC is invalid")
        result = []
        for value in port["values"]:
            uses = [event for event in self.trace["events"] if value in event["inputs"]]
            later = [event for event in uses if event["sequence"] >= boundary]
            def describe(event):
                return None if event is None else {"pc": event["sequence"], "operator": event["operator"],
                                                  "module_path": event["module_path"]}
            result.append({"value_id": value, "producer_pc": self.trace["values"][value]["producer"],
                "port_pc": port["available_pc"], "search_from_pc": boundary,
                "first_consumer": describe(uses[0] if uses else None),
                "first_after_port": describe(later[0] if later else None),
                "already_consumed": bool(uses and uses[0]["sequence"] < port["available_pc"]),
                "authority": "direct SSA input edges; alias/view consumers remain separate SSA edges"})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--select")
    parser.add_argument("--occurrence", type=int)
    parser.add_argument("--path", help="JSON array selecting a nested actual port")
    parser.add_argument("--kind", choices=("module", "call", "parameter", "buffer"))
    parser.add_argument("--first-consumers", action="store_true")
    args = parser.parse_args(argv)
    catalog = NativePortCatalog(json.loads(args.trace.read_text()))
    if args.select:
        method = catalog.first_consumers if args.first_consumers else catalog.resolve
        result = method(args.select, occurrence=args.occurrence,
                        path=None if args.path is None else json.loads(args.path))
    else:
        if args.first_consumers or args.occurrence is not None or args.path is not None:
            parser.error("--first-consumers, --occurrence and --path require --select")
        result = catalog.list(kind=args.kind)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Unified interactive and scriptable Saturn debugger front end."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .debugger_console import DebuggerConsoleError, TypedDebuggerConsole
from .debugger_protocol import (
    BreakpointSpec,
    DebuggerAddressSpec,
    DebuggerProtocolError,
    DebuggerStructSpec,
    HttpDebugTransport,
    WatchSpec,
    conditioner_row_path,
)
from .debugger_session import DebugSession
from .debugger_structs import object_manifest_structs

HELP = """Saturn debugger commands:
  guided | info | status | doctor | capabilities | breakable | where | frame [ID] | list
  tree [PREFIX] | describe PATH | find TEXT | print PATH [--stats] | inspect PATH
  diff|compare PATH CUT_A CUT_B | display PATH | why PATH | xref PATH | attribute PATH
  break token|layer|step|sequence|expert INDEX | break if EXPRESSION
  watch PATH [PREDICATE] [change|level|edge] | until PREDICATE
  continue | pause | step [KIND] [COUNT] | next | reverse-continue PREDICATE
  checkpoint [NAME] | restore CUT_ID | checkout CUT_ID | fork [NAME] | cuts
  frames [NODE_ID]   inspect source scopes, spectral bindings, and StateCut links
  set PATH VALUE [replace|add|scale]
  edit [PATH VALUE|PATH = EXPRESSION] | python STATEMENT
  set-expr PATH EXPRESSION [replace|add|scale]
  calc PATH = EXPRESSION
  conditioner inspect|set|calc STREAM SITE STEP ROW [VALUE|EXPRESSION]
  struct list | struct add NAME JSON | struct load FILE
  address list | address add NAME JSON | address load FILE
  algebra
  preview | commit | abort | abort-all
  replay CUT_ID | reverse-step | reverse-continue PREDICATE
  measure-circuit SPEC|@FILE | replay-circuit EVIDENCE_REF
  history | repeat [N] | alias NAME COMMAND | let NAME VALUE
  save-session FILE | load-session FILE | reconnect | finish [DIR]
  recipe list | recipe run NAME | complete PREFIX
  source FILE | record [FILE] | quit
"""


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _json_payload(raw: str, *, label: str) -> dict[str, Any]:
    source = raw
    if raw.startswith("@"):
        source = Path(raw[1:]).expanduser().read_text(encoding="utf-8")
    try:
        value = json.loads(source)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"{label} requires JSON or @FILE") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} requires a JSON object")
    return value


def _sign(args: list[str], *, default: str = "replace") -> tuple[list[str], str]:
    if args and args[-1] in {"replace", "add", "scale"}:
        return args[:-1], args[-1]
    return args, default


def _session(job_id: str, endpoint: str | None, token: str | None) -> DebugSession:
    resolved = endpoint or os.environ.get("SATURN_DEBUG_ENDPOINT")
    if not resolved:
        base = os.environ.get("MRUN_URL", "http://127.0.0.1:9025").rstrip("/")
        resolved = f"{base}/api/jobs/{job_id}/debug"
    elif (
        "/api/jobs/" not in resolved.rstrip("/")
        and not resolved.rstrip("/").endswith("/debug")
    ):
        # ``--endpoint`` is commonly the mrun base URL when using
        # ``--latest``/``jobs``. Accept both forms without making the user
        # copy a job-specific debug URL by hand.
        resolved = resolved.rstrip("/") + f"/api/jobs/{job_id}/debug"
    return DebugSession(HttpDebugTransport(resolved, token=token), job_id=job_id)


class DebugShell:
    def __init__(self, session: DebugSession, *, json_output: bool = False) -> None:
        self.session = session
        self.json_output = json_output
        self.running = True
        self.history: list[str] = []
        self.variables: dict[str, str] = {}
        self.display_paths: list[str] = []
        self.last_command: str | None = None
        self.console = TypedDebuggerConsole(
            edit=lambda path, expression: self.session.set_expression(path, str(expression)),
            preview=self.session.preview,
            commit=self.session.commit,
            abort=self.session.abort,
            inspect=self.session.inspect,
        )

    def emit(self, value: Any) -> None:
        if self.json_output:
            print(json.dumps(value, sort_keys=True))
        elif isinstance(value, str):
            print(value)
        else:
            print(json.dumps(value, indent=2, sort_keys=True))

    def execute(self, line: str) -> bool:
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        try:
            if line == "!!":
                if not self.last_command:
                    raise ValueError("no previous command")
                line = self.last_command
            elif line.startswith("!") and len(line) > 1 and line[1:].isdigit():
                index = int(line[1:])
                if index < 1 or index > len(self.history):
                    raise ValueError(f"history entry out of range: {index}")
                line = self.history[index - 1]
            line = Template(line).safe_substitute(self.variables)
            raw_command, separator, raw_remainder = line.partition(" ")
            if raw_command in {"measure-circuit", "measure_circuit"}:
                # Keep JSON punctuation and its quoted object keys intact;
                # shlex would strip the inner quotes from an unquoted object.
                remainder = raw_remainder.strip() if separator else ""
                if remainder and remainder[0] in {"'", '"'}:
                    try:
                        quoted = shlex.split(remainder)
                    except ValueError:
                        quoted = []
                    if len(quoted) == 1:
                        remainder = quoted[0]
                argv = [raw_command] + ([remainder] if remainder else [])
            elif raw_command in {"python", "py", "console"}:
                # shlex is correct for debugger arguments, but it would strip
                # the quotes that belong to read("logical.address") in the
                # typed console language. Preserve the console source verbatim.
                remainder = raw_remainder.strip() if separator else ""
                if remainder and remainder[0] in {"'", '"'}:
                    try:
                        quoted = shlex.split(remainder)
                    except ValueError:
                        quoted = []
                    if len(quoted) == 1:
                        remainder = quoted[0]
                argv = [raw_command] + ([remainder] if remainder else [])
            else:
                argv = shlex.split(line)
            if not argv:
                return True
            command = argv.pop(0)
            if command in self.session.aliases:
                expanded = self.session.aliases[command]
                line = expanded + (
                    " " + " ".join(shlex.quote(item) for item in argv) if argv else ""
                )
                return self.execute(line)
            self.history.append(line)
            if command != "repeat":
                self.last_command = line
            result = self._dispatch(command, argv)
            if result is not None:
                if command in {
                    "continue",
                    "c",
                    "resume",
                    "step",
                    "next",
                    "s",
                    "restore",
                    "checkout",
                    "replay",
                    "reverse-step",
                    "commit",
                }:
                    result = self._with_displays(result)
                self.emit(result)
            # Script commands can return a structured failure without raising
            # (notably ``source FILE`` aggregates failures from its child
            # commands). Preserve that status for batch callers.
            return not (isinstance(result, dict) and result.get("ok") is False)
        except (
            DebuggerProtocolError,
            DebuggerConsoleError,
            RuntimeError,
            ValueError,
            IndexError,
            SyntaxError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            self.session.record_refusal(line, exc)
            self.emit({"ok": False, "error": str(exc)})
            return False

    def _with_displays(self, result: Any) -> Any:
        if not self.display_paths:
            return result
        displays = [self.session.inspect(path) for path in self.display_paths]
        return {"result": result, "display": displays}

    def _dispatch(self, command: str, args: list[str]) -> Any:
        if command in {"help", "?"}:
            return HELP
        if command in {"quit", "q", "exit"}:
            self.running = False
            return {"state": "detached"}
        if command == "guided":
            status = self.session.status()
            capabilities = self.session.breakable()
            pause = (
                status
                if status.get("state") == "paused"
                else self.session.wait_for_pause()
            )
            return {
                "guided": True,
                "status": status,
                "capabilities": capabilities,
                "pause": pause,
                "next": ["where", "tree", "breakable", "help"],
            }
        if command in {"info", "status"}:
            return self.session.status()
        if command == "doctor":
            return self.session.doctor()
        if command in {"capabilities", "caps", "breakable"}:
            return self.session.breakable()
        if command in {"where", "frame", "list"}:
            return self.session.inspect(args[0] if args else None)
        if command == "tree":
            return self.session.tree(args[0] if args else None)
        if command == "describe":
            if not args:
                raise ValueError("describe requires PATH")
            return self.session.describe(args[0])
        if command == "find":
            if not args:
                raise ValueError("find requires TEXT")
            return self.session.find(" ".join(args))
        if command in {"print", "inspect", "p", "x"}:
            if not args:
                raise ValueError("print requires PATH")
            stats = "--stats" in args
            path = next((arg for arg in args if arg != "--stats"), None)
            if not path:
                raise ValueError("print requires PATH")
            return self.session.inspect_stats(path) if stats else self.session.inspect(path)
        if command in {"diff", "compare"}:
            if len(args) != 3:
                raise ValueError(f"{command} requires PATH CUT_A CUT_B")
            return self.session.diff(args[0], args[1], args[2])
        if command == "display":
            if not args:
                return {"display": list(self.display_paths)}
            if args[0] == "clear":
                self.display_paths.clear()
                return {"display": []}
            path = args[0]
            self.session.inspect(path)
            if path not in self.display_paths:
                self.display_paths.append(path)
            return {"displaying": list(self.display_paths)}
        if command in {"why", "xref", "attribute"}:
            if not args:
                raise ValueError(f"{command} requires PATH")
            return getattr(self.session, command)(" ".join(args))
        if command == "break":
            if not args:
                return self.session.list_breakpoints()
            if args[0] == "clear":
                if len(args) != 2:
                    raise ValueError("break clear requires BREAKPOINT_ID")
                return self.session.call("breakpoint.clear", {"breakpoint_id": args[1]})
            if args[0] == "if":
                return self.session.add_breakpoint(
                    BreakpointSpec(kind="if", expression=" ".join(args[1:]))
                )
            if len(args) != 2:
                raise ValueError("break requires KIND INDEX")
            return self.session.add_breakpoint(BreakpointSpec(kind=args[0], index=int(args[1])))
        if command == "watch":
            if not args:
                return self.session.list_watches()
            if args[0] == "clear":
                if len(args) != 2:
                    raise ValueError("watch clear requires WATCH_ID")
                return self.session.clear_watch(args[1])
            mode = "change"
            if args[-1] in {"change", "level", "edge"} and len(args) > 1:
                mode = args[-1]
                args = args[:-1]
            return self.session.add_watch(
                WatchSpec(path=args[0], predicate=" ".join(args[1:]) or None, mode=mode)
            )
        if command == "until":
            if not args:
                raise ValueError("until requires a typed predicate")
            breakpoint = self.session.add_breakpoint(
                BreakpointSpec(kind="if", expression=" ".join(args))
            )
            try:
                result = self.session.continue_run()
            finally:
                breakpoint_id = breakpoint.get("breakpoint", {}).get("breakpoint_id")
                if breakpoint_id:
                    self.session.call("breakpoint.clear", {"breakpoint_id": breakpoint_id})
            return {"until": " ".join(args), "result": result}
        if command in {"continue", "c", "resume"}:
            return self.session.continue_run()
        if command == "pause":
            return self.session.pause()
        if command in {"step", "next", "s"}:
            kind = None
            count_args = list(args)
            if count_args:
                try: int(count_args[0])
                except ValueError: kind = count_args.pop(0)
            if len(count_args) > 1:
                raise ValueError("step accepts [KIND] [COUNT]")
            try:
                count = int(count_args[0]) if count_args else 1
            except ValueError as exc:
                raise ValueError("step COUNT must be an integer") from exc
            if count < 1:
                raise ValueError("step COUNT must be positive")
            return self.session.step(count, kind=kind)
        if command == "checkpoint":
            return self.session.checkpoint(args[0] if args else None)
        if command in {"restore", "checkout"}:
            if not args:
                raise ValueError(f"{command} requires CUT_ID")
            return self.session.restore(args[0])
        if command == "fork":
            return self.session.fork(args[0] if args else "fork")
        if command == "cuts":
            return self.session.cuts()
        if command == "frames":
            if len(args) > 1:
                raise ValueError("frames accepts [NODE_ID]")
            return self.session.frames(args[0] if args else None)
        if command == "set":
            if len(args) < 2:
                raise ValueError("set requires PATH VALUE [replace|add|scale]")
            value_args, sign = _sign(args[1:])
            if len(value_args) != 1:
                raise ValueError("set VALUE must be one JSON/scalar token")
            return self.session.set_value(args[0], _value(value_args[0]), sign=sign)
        if command == "edit":
            if not args:
                return self._edit_wizard()
            if "=" in args:
                equals = args.index("=")
                if equals != 1 or equals == len(args) - 1:
                    raise ValueError("edit expression syntax is: edit PATH = EXPRESSION [sign]")
                expression_args, sign = _sign(args[equals + 1 :])
                return self.session.set_expression(args[0], " ".join(expression_args), sign=sign)
            if len(args) < 2:
                raise ValueError("edit requires PATH VALUE [replace|add|scale]")
            value_args, sign = _sign(args[1:])
            if len(value_args) != 1:
                raise ValueError("edit VALUE must be one JSON/scalar token")
            return self.session.set_value(args[0], _value(value_args[0]), sign=sign)
        if command in {"set-expr", "expr"}:
            if len(args) < 2:
                raise ValueError("set-expr requires PATH EXPRESSION [replace|add|scale]")
            expression_args, sign = _sign(args[1:])
            expression = " ".join(expression_args)
            return self.session.set_expression(args[0], expression, sign=sign)
        if command in {"calc", "algebra-set"}:
            if len(args) < 3:
                raise ValueError("calc requires PATH = EXPRESSION [replace|add|scale]")
            if args[1] != "=":
                raise ValueError("calc syntax is: calc PATH = EXPRESSION [replace|add|scale]")
            expression_args, sign = _sign(args[2:])
            return self.session.set_expression(args[0], " ".join(expression_args), sign=sign)
        if command in {"conditioner", "conditioner-row", "row"}:
            if not args:
                return self.session.breakable()
            action = args.pop(0)
            if action == "inspect":
                if len(args) != 4:
                    raise ValueError("conditioner inspect requires STREAM SITE STEP ROW")
                path = conditioner_row_path(args[0], args[1], int(args[2]), int(args[3]))
                return self.session.inspect(path)
            if action in {"set", "write"}:
                if len(args) < 5:
                    raise ValueError(
                        "conditioner set requires STREAM SITE STEP ROW VALUE [replace|add|scale]"
                    )
                value_args, sign = _sign(args[4:])
                if len(value_args) != 1:
                    raise ValueError("conditioner row VALUE must be one JSON/scalar token")
                return self.session.set_conditioner_row(
                    args[0], args[1], int(args[2]), int(args[3]), _value(value_args[0]), sign=sign
                )
            if action in {"calc", "expr"}:
                if len(args) < 5:
                    raise ValueError(
                        "conditioner calc requires STREAM SITE STEP ROW EXPRESSION [sign]"
                    )
                expression_args, sign = _sign(args[4:])
                if expression_args and expression_args[0] == "=":
                    expression_args = expression_args[1:]
                if not expression_args:
                    raise ValueError("conditioner calc requires a non-empty expression")
                return self.session.set_conditioner_row(
                    args[0],
                    args[1],
                    int(args[2]),
                    int(args[3]),
                    None,
                    sign=sign,
                    expression=" ".join(expression_args),
                )
            raise ValueError("conditioner requires inspect, set, or calc")
        if command == "struct":
            if not args or args[0] == "list":
                return self.session.list_structs()
            action = args.pop(0)
            if action == "load":
                if len(args) != 1:
                    raise ValueError("struct load requires FILE")
                payload = json.loads(Path(args[0]).expanduser().read_text(encoding="utf-8"))
                if (
                    isinstance(payload, dict)
                    and payload.get("schema") == "saturn-object-manifest-v1"
                ):
                    results = [
                        self.session.add_struct(spec) for spec in object_manifest_structs(payload)
                    ]
                    return {"loaded": len(results), "results": results}
                return self.session.add_struct(DebuggerStructSpec.from_dict(payload))
            if action == "add":
                if len(args) < 2:
                    raise ValueError("struct add requires NAME JSON")
                payload = _json_payload(" ".join(args[1:]), label="struct add")
                payload["name"] = args[0]
                return self.session.add_struct(DebuggerStructSpec.from_dict(payload))
            raise ValueError("struct requires list, add, or load")
        if command == "address":
            if not args or args[0] == "list":
                return self.session.list_addresses()
            action = args.pop(0)
            if action == "load":
                if len(args) != 1:
                    raise ValueError("address load requires FILE")
                spec = DebuggerAddressSpec.from_dict(
                    json.loads(Path(args[0]).expanduser().read_text(encoding="utf-8"))
                )
                return self.session.add_address(spec)
            if action == "add":
                if len(args) < 2:
                    raise ValueError("address add requires NAME JSON")
                payload = _json_payload(" ".join(args[1:]), label="address add")
                payload["name"] = args[0]
                return self.session.add_address(DebuggerAddressSpec.from_dict(payload))
            raise ValueError("address requires list, add, or load")
        if command in {"algebra", "algebra-help"}:
            return self.session.algebra()
        if command == "preview":
            return self.session.preview()
        if command == "commit":
            return self.session.commit()
        if command == "abort":
            return self.session.abort()
        if command == "abort-all":
            return self.session.abort_all()
        if command in {"python", "py", "console"}:
            if args:
                return self.console.execute(" ".join(args))
            return self._python_repl()
        if command == "replay":
            if not args:
                raise ValueError("replay requires CUT_ID")
            return self.session.replay(args[0])
        if command in {"measure-circuit", "measure_circuit"}:
            if not args:
                raise ValueError("measure-circuit requires a JSON object spec or @FILE")
            return self.session.measure_circuit(
                _json_payload(" ".join(args), label="measure-circuit")
            )
        if command in {"replay-circuit", "replay_circuit"}:
            if len(args) != 1:
                raise ValueError("replay-circuit requires EVIDENCE_REF")
            return self.session.replay_circuit(args[0])
        if command == "reverse-step":
            parent = self.session.resolve_cut("@parent")
            if not parent:
                raise ValueError("current StateCut has no durable parent")
            return self.session.restore(parent)
        if command == "reverse-continue":
            if not args:
                raise ValueError("reverse-continue requires a typed predicate")
            return self.session.reverse_continue(" ".join(args))
        if command in {"first-divergence", "bisect"}:
            if len(args) != 2:
                raise ValueError(f"{command} requires CUT_A CUT_B")
            return self.session.first_divergence(args[0], args[1])
        if command == "promote":
            if not args:
                raise ValueError("promote requires CUT_ID")
            return self.session.promote(args[0])
        if command == "history":
            limit = int(args[0]) if args else len(self.history)
            if limit < 0:
                raise ValueError("history limit must be non-negative")
            entries = [] if limit == 0 else self.history[-limit:]
            offset = len(self.history) - len(entries)
            return {
                "history": [
                    {"index": offset + i + 1, "command": item}
                    for i, item in enumerate(entries)
                ]
            }
        if command == "repeat":
            if not self.last_command:
                raise ValueError("no previous command")
            if args and args[0].isdigit():
                count = max(1, int(args[0]))
                command_line = self.last_command
            else:
                count = 1
                command_line = " ".join(args)
            for _ in range(count):
                if not self.execute(command_line):
                    break
            return {"repeated": command_line, "count": count}
        if command == "alias":
            if len(args) < 2:
                raise ValueError("alias requires NAME COMMAND")
            return self.session.set_alias(args[0], " ".join(args[1:]))
        if command in {"let", "path"}:
            if len(args) < 2:
                raise ValueError(f"{command} requires NAME VALUE")
            self.variables[args[0]] = " ".join(args[1:])
            return {"variable": args[0], "value": self.variables[args[0]]}
        if command == "save-session":
            if len(args) != 1:
                raise ValueError("save-session requires FILE")
            return self.session.save_session(
                args[0],
                shell_state={
                    "history": list(self.history),
                    "variables": dict(self.variables),
                    "display_paths": list(self.display_paths),
                    "last_command": self.last_command,
                },
            )
        if command == "load-session":
            if len(args) != 1:
                raise ValueError("load-session requires FILE")
            result = self.session.load_session(args[0])
            shell = result.get("shell") or {}
            self.history = [str(item) for item in shell.get("history", ())]
            self.variables = {str(k): str(v) for k, v in (shell.get("variables") or {}).items()}
            self.display_paths = [str(item) for item in shell.get("display_paths", ())]
            self.last_command = shell.get("last_command")
            return result
        if command == "reconnect":
            return self.session.reconnect()
        if command == "finish":
            return self.session.finish(args[0] if args else None)
        if command == "recipe":
            return self._recipe(args)
        if command == "complete":
            return self._complete(" ".join(args))
        if command == "source":
            if not args:
                raise ValueError("source requires FILE")
            return self.run_script(Path(args[0]))
        if command == "record":
            return self.session.write_receipt(args[0] if args else None)
        raise ValueError(f"unknown debugger command: {command}")

    def _edit_wizard(self) -> dict[str, Any]:
        if not sys.stdin.isatty():
            raise ValueError(
                "edit requires PATH and VALUE in scripts; interactive prompting needs a terminal"
            )
        path = input("address: ").strip()
        operation = input("operation [replace|add|scale|expression]: ").strip() or "replace"
        value = input("value/expression: ").strip()
        if operation == "expression":
            return self.session.set_expression(path, value)
        return self.session.set_value(path, _value(value), sign=operation)

    def _python_repl(self) -> dict[str, Any]:
        if not sys.stdin.isatty():
            raise ValueError("python requires a quoted statement in scripts")
        results: list[Any] = []
        while True:
            try:
                line = input("(mdb.py) ")
            except EOFError:
                break
            if not line.strip() or line.strip() in {"quit", "exit"}:
                break
            results.append(self.console.execute(line))
        return {"console": results}

    def _recipe(self, args: list[str]) -> dict[str, Any]:
        recipes = {
            "semantic-object": ["capabilities", "struct list", "address list", "where"],
            "conditioner-row": ["capabilities", "breakable"],
            "causal-path": ["where", "tree", "breakable"],
            "evidence-close": ["status", "cuts", "record"],
        }
        if not args or args[0] == "list":
            return {
                "recipes": sorted(recipes),
                "descriptions": {name: commands for name, commands in recipes.items()},
            }
        if args[0] != "run" or len(args) < 2:
            raise ValueError("recipe syntax is: recipe list | recipe run NAME")
        name = args[1]
        if name not in recipes:
            raise ValueError(f"unknown recipe {name}; choose from {', '.join(sorted(recipes))}")
        results: list[Any] = []
        for recipe_command in recipes[name]:
            if recipe_command == "record":
                results.append(self.session.write_receipt())
            else:
                command_argv = shlex.split(recipe_command)
                results.append(self._dispatch(command_argv[0], command_argv[1:]))
        return {"recipe": name, "results": results}

    def _complete(self, prefix: str) -> dict[str, Any]:
        commands = [
            "help",
            "info",
            "status",
            "doctor",
            "capabilities",
            "breakable",
            "where",
            "frame",
            "list",
            "tree",
            "describe",
            "find",
            "print",
            "inspect",
            "diff",
            "compare",
            "display",
            "why",
            "xref",
            "attribute",
            "break",
            "watch",
            "until",
            "continue",
            "pause",
            "step",
            "next",
            "checkpoint",
            "restore",
            "checkout",
            "fork",
            "cuts",
            "frames",
            "set",
            "edit",
            "set-expr",
            "calc",
            "conditioner",
            "struct",
            "address",
            "algebra",
            "preview",
            "commit",
            "abort",
            "abort-all",
            "python",
            "replay",
            "measure-circuit",
            "replay-circuit",
            "reverse-step",
            "reverse-continue",
            "first-divergence",
            "bisect",
            "promote",
            "history",
            "repeat",
            "alias",
            "let",
            "save-session",
            "load-session",
            "reconnect",
            "finish",
            "recipe",
            "complete",
            "source",
            "record",
            "quit",
        ]
        prefix = prefix.lstrip()
        try:
            tokens = shlex.split(prefix)
        except ValueError:
            tokens = prefix.split()
        trailing_space = prefix.endswith(" ")
        if not tokens or (len(tokens) == 1 and not trailing_space):
            partial = tokens[0] if tokens else prefix
            return {
                "prefix": prefix,
                "matches": sorted(item for item in commands if item.startswith(partial)),
            }

        command = tokens[0]
        partial = "" if trailing_space else tokens[-1]
        if not trailing_space and len(tokens) == 1:
            partial = command
        try:
            capability = self.session.breakable()
        except Exception:  # noqa: BLE001 - completion must never interrupt editing
            capability = {}
        address_candidates = list(capability.get("inspectable_slots") or ())
        address_candidates.extend(
            str(item.get("path_template"))
            for item in (capability.get("address_specs") or ())
            if isinstance(item, dict) and item.get("path_template")
        )
        boundary_candidates = list(capability.get("boundaries") or ())
        cut_candidates = ["@current", "@parent", "@root"]
        try:
            cuts = self.session.cuts()
            cut_candidates.extend(
                str(item.get("cut_id"))
                for item in cuts.get("cuts", ())
                if item.get("cut_id")
            )
            cut_candidates.extend(str(name) for name in (cuts.get("names") or {}))
        except Exception:  # noqa: BLE001 - static symbolic cuts remain useful offline
            pass
        context: dict[str, list[str]] = {
            "print": address_candidates,
            "inspect": address_candidates,
            "describe": address_candidates,
            "display": address_candidates + ["clear"],
            "watch": address_candidates + ["change", "level", "edge"],
            "set": address_candidates,
            "edit": address_candidates,
            "set-expr": address_candidates,
            "calc": address_candidates,
            "why": address_candidates,
            "xref": address_candidates,
            "attribute": address_candidates,
            "break": boundary_candidates + ["if", "clear"],
            "restore": cut_candidates,
            "checkout": cut_candidates,
            "replay": cut_candidates,
            "reverse-step": cut_candidates,
            "diff": address_candidates + cut_candidates,
            "compare": address_candidates + cut_candidates,
            "first-divergence": cut_candidates,
            "bisect": cut_candidates,
            "promote": cut_candidates,
            "conditioner": ["inspect", "set", "write", "calc", "expr"],
            "struct": ["list", "add", "load"],
            "address": ["list", "add", "load"],
            "python": list(capability.get("algebra_ops") or ())
            + ["read", "edit", "preview", "commit", "abort", "inspect"],
            "py": list(capability.get("algebra_ops") or ())
            + ["read", "edit", "preview", "commit", "abort", "inspect"],
            "console": list(capability.get("algebra_ops") or ())
            + ["read", "edit", "preview", "commit", "abort", "inspect"],
        }
        candidates = context.get(command, [])
        return {
            "prefix": prefix,
            "matches": sorted({item for item in candidates if item.startswith(partial)}),
        }

    def run_script(self, path: Path) -> dict[str, Any]:
        failures = 0
        commands = 0
        for raw_line in path.expanduser().read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("for ") and " do " in line:
                header, body = line[4:].split(" do ", 1)
                name, values = header.split(" in ", 1)
                values = values.strip().strip("[]")
                for value in [item.strip() for item in values.split(",") if item.strip()]:
                    self.variables[name.strip()] = value.strip("'\"")
                    commands += 1
                    if not self.execute(body):
                        failures += 1
                continue
            if line.startswith("if ") and " then " in line:
                condition, body = line[3:].split(" then ", 1)
                if not self._script_condition(condition):
                    continue
                line = body
            commands += 1
            if not self.execute(line):
                failures += 1
            if not self.running:
                break
        return {"commands": commands, "failures": failures, "ok": failures == 0}

    def _script_condition(self, expression: str) -> bool:
        parts = expression.split("==", 1)
        if len(parts) != 2:
            raise ValueError("script if syntax is: if NAME == VALUE then COMMAND")
        left, right = (part.strip() for part in parts)
        if left == "state":
            observed = self.session.status().get("state")
        else:
            observed = self.variables.get(left)
        return str(observed) == right.strip("'\"")

    def repl(self) -> int:
        if not self.json_output:
            print("Saturn debugger; type 'help' for commands.")
        try:
            import readline

            readline.parse_and_bind("tab: complete")
            readline.set_completer(lambda text, _state: self._completion_value(text))
        except (ImportError, AttributeError):
            pass
        while self.running:
            try:
                line = input("(mdb) ")
            except EOFError:
                break
            self.execute(line)
        return 0

    def _completion_value(self, text: str) -> str | None:
        matches = self._complete(text)["matches"]
        try:
            import readline

            index = readline.get_completion_state()
        except (ImportError, AttributeError):
            index = 0
        return matches[index] if index < len(matches) else None


def _discover_jobs(endpoint: str | None = None) -> list[dict[str, Any]]:
    base = (endpoint or os.environ.get("MRUN_URL", "http://127.0.0.1:9025")).rstrip("/")
    if "/api/jobs/" in base:
        base = base.split("/api/jobs/", 1)[0].rstrip("/")
    url = base + "/api/jobs?" + urlencode({"debuggable": "true"})
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=10.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - CLI turns discovery failures into one clear error
        raise DebuggerProtocolError(f"cannot discover debugger-capable mrun jobs: {exc}") from exc
    if isinstance(payload, dict):
        jobs = payload.get("jobs") or payload.get("items") or []
    else:
        jobs = payload
    if not isinstance(jobs, list):
        raise DebuggerProtocolError("mrun job discovery returned an invalid payload")
    return [dict(job) for job in jobs if isinstance(job, dict)]


def _resolve_job_id(args: argparse.Namespace) -> str:
    job_id = getattr(args, "job_id", None)
    if job_id:
        return str(job_id)
    if not getattr(args, "latest", False):
        raise DebuggerProtocolError("attach/debug requires JOB_ID or --latest")
    jobs = _discover_jobs(args.endpoint)
    if not jobs:
        raise DebuggerProtocolError("no debugger-capable mrun jobs were discovered")
    jobs.sort(
        key=lambda job: str(
            job.get("created_at") or job.get("submitted_at") or job.get("job_id", "")
        )
    )
    latest = jobs[-1]
    resolved = latest.get("job_id") or latest.get("id")
    if not resolved:
        raise DebuggerProtocolError("mrun returned a debuggable job without a job ID")
    return str(resolved)


def _attach(args: argparse.Namespace) -> int:
    job_id = _resolve_job_id(args)
    session = _session(job_id, args.endpoint, args.token)
    if getattr(args, "session_dir", None):
        session._session_dir = Path(args.session_dir).expanduser().resolve()  # noqa: SLF001 - CLI owns session placement
    shell = DebugShell(session, json_output=args.json)
    if getattr(args, "guided", False):
        ok = shell.execute("guided")
        if not ok:
            return 1
    if args.command_line:
        ok = shell.execute(args.command_line)
        if args.record:
            shell.session.write_receipt(args.record)
        return 0 if ok else 1
    if args.script:
        result = shell.run_script(Path(args.script))
        if args.record:
            shell.session.write_receipt(args.record)
        return 0 if result["ok"] else 1
    return shell.repl()


def _jobs(args: argparse.Namespace) -> int:
    jobs = _discover_jobs(args.endpoint)
    payload = {"jobs": jobs, "count": len(jobs), "debuggable": True}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run(args: argparse.Namespace) -> int:
    if not args.command:
        raise ValueError("saturn run --debug requires a command after --")
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("saturn run --debug requires a command after --")
    environment = dict(os.environ)
    environment["SATURN_DEBUG_ENABLED"] = "1"
    environment["SATURN_DEBUG_PROTOCOL"] = "saturn-debugger-protocol-v1"
    if args.endpoint:
        environment["SATURN_DEBUG_ENDPOINT"] = args.endpoint
    return subprocess.call(command, env=environment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdb", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("attach", "debug"):
        command = sub.add_parser(name, help="attach to an instrumented mrun job")
        command.add_argument("job_id", nargs="?")
        command.add_argument(
            "--latest", action="store_true", help="attach to the newest debuggable job"
        )
        command.add_argument("--endpoint")
        command.add_argument("--token", default=os.environ.get("MRUN_TOKEN"))
        command.add_argument("--command", dest="command_line", help="execute one debugger command")
        command.add_argument("--script", type=Path)
        command.add_argument("--record", type=Path)
        command.add_argument("--session-dir", type=Path)
        command.add_argument(
            "--guided", action="store_true", help="request a first safe pause and show capabilities"
        )
        command.add_argument("--json", action="store_true")
        command.set_defaults(handler=_attach)
    script = sub.add_parser("script", help="run a debugger command script")
    script.add_argument("file", type=Path)
    script.add_argument("--job", required=True)
    script.add_argument("--endpoint")
    script.add_argument("--token", default=os.environ.get("MRUN_TOKEN"))
    script.add_argument("--record", type=Path)
    script.add_argument("--json", action="store_true")
    script.set_defaults(
        handler=lambda args: _attach(
            argparse.Namespace(
                job_id=args.job,
                endpoint=args.endpoint,
                token=args.token,
                command_line=None,
                script=args.file,
                record=args.record,
                session_dir=None,
                guided=False,
                latest=False,
                json=args.json,
            )
        )
    )
    run = sub.add_parser("run", help="launch a command with debugger instrumentation enabled")
    run.add_argument("--debug", action="store_true", required=True)
    run.add_argument("--endpoint")
    run.add_argument("command", nargs=argparse.REMAINDER)
    run.set_defaults(handler=_run)
    replay = sub.add_parser("replay", help="attach and replay an exact StateCut")
    replay.add_argument("target", help="StateCut or session identity")
    replay.add_argument("--job", help="mrun job ID when target is a StateCut")
    replay.add_argument("--endpoint")
    replay.add_argument("--token", default=os.environ.get("MRUN_TOKEN"))
    replay.add_argument("--json", action="store_true")
    replay.set_defaults(
        handler=lambda args: _attach(
            argparse.Namespace(
                job_id=args.job or os.environ.get("SATURN_DEBUG_JOB") or args.target,
                endpoint=args.endpoint,
                token=args.token,
                command_line=f"replay {shlex.quote(args.target)}",
                script=None,
                record=None,
                session_dir=None,
                guided=False,
                latest=False,
                json=args.json,
            )
        )
    )
    jobs = sub.add_parser(
        "jobs", help="list mrun jobs that registered Saturn debugger capabilities"
    )
    jobs.add_argument("--endpoint")
    jobs.add_argument("--json", action="store_true")
    jobs.set_defaults(handler=_jobs)
    doctor = sub.add_parser("doctor", help="check debugger instrumentation and replay readiness")
    doctor.add_argument("job_id")
    doctor.add_argument("--endpoint")
    doctor.add_argument("--token", default=os.environ.get("MRUN_TOKEN"))
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(
        handler=lambda args: _attach(
            argparse.Namespace(
                job_id=args.job_id,
                endpoint=args.endpoint,
                token=args.token,
                command_line="doctor",
                script=None,
                record=None,
                session_dir=None,
                guided=False,
                latest=False,
                json=args.json,
            )
        )
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = argv if argv is not None else sys.argv[1:]
    if raw and raw[0] == "bench":
        from mdb.bench_cli import main as bench_main
        return bench_main(raw[1:])
    # `saturn debug vm ...` is the model-virtual-memory page-op group, implemented
    # in saturn.debug.  The `debug` leaf here still attaches to a job when the next
    # token is a job id (never the literal "vm"), so this shim is non-breaking.
    if len(raw) >= 2 and raw[0] in ("debug", "attach") and raw[1] == "vm":
        from mdb.vm_cli import main as debug_main

        return debug_main(raw[2:])
    args = build_parser().parse_args(raw)
    try:
        return int(args.handler(args) or 0)
    except (
        DebuggerProtocolError,
        DebuggerConsoleError,
        RuntimeError,
        ValueError,
        OSError,
        SyntaxError,
        json.JSONDecodeError,
    ) as exc:
        payload = {"ok": False, "error": str(exc)}
        if getattr(args, "json", False):
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"mdb: {exc}", file=sys.stderr)
        return 1

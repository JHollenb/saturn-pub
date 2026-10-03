"""An in-process debugger; every operation delegates to the public Session API."""

from __future__ import annotations

import argparse
import json
import shlex
from fnmatch import fnmatchcase
from typing import Any

from .core import Act, Session
from .observers import ObservationError, Observer, WatchEvent
from .paths import Trajectory, compare_trajectories
from .store import LocalStore
from .symbols import SymbolTable, backtrace, structural_symbols


class Debugger:
    def __init__(self, session: Session):
        self.branches = {"root": session}
        self.active = "root"
        self.cuts = {}
        self.breakpoints: list[str] = []
        self.observer = Observer(reader=self._read_observed)
        self.watch_events: list[WatchEvent] = []
        self.watch_errors: list[dict[str, Any]] = []
        self.max_continue_steps = 1000
        self.symbols = SymbolTable()
        self.symbol_context = ""
        self.trajectories: dict[str, Trajectory] = {}

    @property
    def session(self) -> Session:
        return self.branches[self.active]

    def _read_observed(self, session: Session, address: str) -> Any:
        if address.startswith("symbol:"):
            binding = self.symbols.resolve(session, address[7:], context=self.symbol_context)
            return tuple(session.read_port(location) for location in binding.locations)
        return session.read(address)

    def _frame(self) -> dict[str, Any]:
        frame = self.session.inspect()
        result = {"branch": self.active, "boundary": frame.boundary, "slots": dict(frame.slots)}
        point = getattr(frame, "execution_point", None)
        surface = getattr(frame, "surface", None)
        if point is not None:
            result["execution_point"] = dict(point)
        if surface is not None:
            result["surface"] = dict(surface)
        return result

    def _observe(self, receipt: Any) -> tuple[WatchEvent, ...]:
        try:
            events = self.observer.evaluate(self.session, self.active, receipt=receipt)
        except ObservationError as exc:
            self.cuts[exc.cut.fingerprint] = exc.cut
            self.watch_events.extend(exc.events)
            self.watch_errors.append(
                {
                    "branch": self.active,
                    "cut": exc.cut.fingerprint,
                    "receipt": receipt.fingerprint,
                    "errors": list(exc.errors),
                }
            )
            raise
        for event in events:
            self.cuts[event.cut.fingerprint] = event.cut
        self.watch_events.extend(events)
        return events

    def _step(self) -> Any:
        step = getattr(self.session, "step", None)
        return step() if step is not None else self.session.continue_()

    def advance(
        self,
        steps: int = 1,
        *,
        patterns: list[str] | None = None,
        stop_on_breakpoint: bool = False,
        trace: bool = False,
    ) -> dict[str, Any]:
        """Advance one micro-transition at a time so watches see every safe point."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("continue limit must be positive")
        patterns = patterns or []
        rows = []
        receipts = []
        for completed in range(1, steps + 1):
            receipt = self._step()
            receipts.append(receipt.to_dict())
            events = self._observe(receipt)
            frame = self._frame()
            if trace:
                rows.append(
                    {
                        "step": completed,
                        "frame": frame,
                        "receipt": receipt.to_dict(),
                        "watch_events": [event.to_dict() for event in events],
                    }
                )
            if events:
                return {
                    "stop": "watchpoint",
                    "steps": completed,
                    "events": [event.to_dict() for event in events],
                    "frame": frame,
                    "receipts": receipts,
                    **({"trace": rows} if trace else {}),
                }
            matches = [pattern for pattern in patterns if fnmatchcase(frame["boundary"], pattern)]
            if matches and stop_on_breakpoint:
                return {
                    "stop": "breakpoint",
                    "steps": completed,
                    "matched": matches,
                    "frame": frame,
                    "receipts": receipts,
                    **({"trace": rows} if trace else {}),
                }
        return {
            "stop": "step-limit",
            "steps": steps,
            "frame": self._frame(),
            "receipts": receipts,
            **({"trace": rows} if trace else {}),
        }

    def _until(self, patterns: list[str], limit: int) -> dict[str, Any]:
        # Advance first: continuing from a breakpoint must make forward progress.
        return self.advance(limit, patterns=patterns, stop_on_breakpoint=True)

    def execute(self, line: str) -> Any:
        args = shlex.split(line)
        if not args:
            return None
        command, *rest = args
        if command == "info" and rest == ["steps"]:
            cut = self.session.capture(retain=False)
            state, _ = self.session._cut_state(cut)
            return {
                "point": dict(cut.execution_point or {}),
                "next_transition": self.session.adapter.transition(state).to_dict(),
            }
        if command == "info" and rest == ["symbols"]:
            return {
                "structural": structural_symbols(self.session),
                "qualified": [row.to_dict() for row in self.symbols.bindings],
            }
        if command == "resolve" and len(rest) == 1:
            binding = self.symbols.resolve(self.session, rest[0], context=self.symbol_context)
            return binding.to_dict()
        if command == "backtrace" and len(rest) == 1:
            if rest[0].startswith("symbol:"):
                binding = self.symbols.resolve(
                    self.session, rest[0][7:], context=self.symbol_context
                )
                return {
                    "binding": binding.to_dict(),
                    "provenance": [backtrace(self.session, loc.slot) for loc in binding.locations],
                    "granularity": "slot-level-dependencies",
                }
            return backtrace(self.session, rest[0])
        if command == "diff" and len(rest) == 3 and rest[2] == "--first-divergence":
            return compare_trajectories(self.trajectories[rest[0]], self.trajectories[rest[1]])
        if command in ("inspect", "where") and not rest:
            return self._frame()
        if command == "read" and len(rest) == 1:
            if rest[0].startswith("symbol:"):
                from .values import describe

                return describe(self._read_observed(self.session, rest[0]))
            value = self.session.read(rest[0])
            return value.tolist() if hasattr(value, "tolist") else value
        if command == "break" and len(rest) == 1:
            if rest[0] not in self.breakpoints:
                self.breakpoints.append(rest[0])
            return {"breakpoints": list(self.breakpoints)}
        if command == "breakpoints" and not rest:
            return {"breakpoints": list(self.breakpoints)}
        if command == "delete" and len(rest) == 1:
            if rest[0] == "all":
                self.breakpoints.clear()
            else:
                self.breakpoints.remove(rest[0])
            return {"breakpoints": list(self.breakpoints)}
        if command == "until" and len(rest) in (1, 2):
            return self._until(
                [rest[0]], int(rest[1]) if len(rest) == 2 else self.max_continue_steps
            )
        if command == "continue" and not rest and self.breakpoints:
            return self._until(self.breakpoints, self.max_continue_steps)
        if command == "continue" and not rest and self.observer.watchpoints:
            return self.advance(self.max_continue_steps)
        if command in ("step", "next", "continue") and len(rest) <= 1:
            steps = int(rest[0]) if rest else 1
            # An explicit budget preserves the legacy exact-N breakpoint behavior. Watches
            # still stop at a retained safe point; bare continue handles installed breaks above.
            return self.advance(steps)
        if command == "trace" and len(rest) <= 1:
            return self.advance(int(rest[0]) if rest else 1, trace=True)
        if command == "watch" and len(rest) in (2, 3):
            threshold = float(rest[2]) if len(rest) == 3 else None
            watch = self.observer.add(rest[0], rest[1], threshold)
            # Establish change-watch semantics at the command's current branch and safe point.
            try:
                self.observer.sync(self.session, self.active)
            except Exception:
                self.observer.remove(watch.identifier)
                raise
            return watch.to_dict()
        if command == "watchpoints" and not rest:
            return {"watchpoints": [watch.to_dict() for watch in self.observer.watchpoints]}
        if command == "unwatch" and len(rest) == 1:
            if rest[0] == "all":
                self.observer.clear()
            else:
                self.observer.remove(rest[0])
            return {"watchpoints": [watch.to_dict() for watch in self.observer.watchpoints]}
        if command == "capture" and len(rest) == 1:
            self.cuts[rest[0]] = self.session.capture()
            return {"cut": rest[0], "fingerprint": self.cuts[rest[0]].fingerprint}
        if command == "fork" and len(rest) in (1, 2):
            if rest[0] in self.branches:
                raise ValueError("branch name already exists")
            self.branches[rest[0]] = self.session.fork(
                self.cuts[rest[1]] if len(rest) == 2 else None
            )
            return {"branch": rest[0]}
        if command == "use" and len(rest) == 1:
            if rest[0] not in self.branches:
                raise ValueError("unknown branch")
            self.active = rest[0]
            # A new branch gets a baseline at its selected cursor. Existing branches retain theirs.
            if not self.observer.has_baseline(self.active):
                self.observer.sync(self.session, self.active)
            return {"active": self.active}
        if command == "zero" and len(rest) == 1:
            receipt = self.session.apply(Act.zero(rest[0]))
            events = self._observe(receipt)
            result = receipt.to_dict()
            if events:
                result["watch_events"] = [event.to_dict() for event in events]
            return result
        if command in ("add", "replace") and len(rest) == 2:
            value = json.loads(rest[1])
            current = self.session.read(rest[0])
            if hasattr(current, "shape"):
                import torch

                value = torch.as_tensor(value, device=current.device, dtype=current.dtype)
                if command == "add" and value.ndim == 0:
                    value = torch.ones_like(current) * value
            act = Act.add(rest[0], value) if command == "add" else Act.replace(rest[0], value)
            receipt = self.session.apply(act)
            events = self._observe(receipt)
            result = receipt.to_dict()
            if events:
                result["watch_events"] = [event.to_dict() for event in events]
            return result
        if command == "restore" and len(rest) == 1:
            result = self.session.restore(self.cuts[rest[0]]).to_dict()
            self.observer.sync(self.session, self.active)
            return result
        if command in ("compare", "diff") and len(rest) == 1:
            if rest[0] in self.branches:
                other = self.branches[rest[0]]
            elif rest[0] in self.cuts:
                other = self.session.fork(self.cuts[rest[0]])
            else:
                raise ValueError("unknown branch or cut")
            return self.session.compare(other)
        if command == "replay" and 1 <= len(rest) <= 3:
            cut_name = rest[0]
            steps = int(rest[1]) if len(rest) >= 2 else 1
            name = rest[2] if len(rest) == 3 else f"replay-{len(self.branches)}"
            if name in self.branches:
                raise ValueError("branch name already exists")
            replay = getattr(self.session, "replay", None)
            branch = (
                replay(self.cuts[cut_name], steps=steps)
                if replay
                else self.session.fork(self.cuts[cut_name])
            )
            if replay is None:
                branch.continue_(steps)
            self.branches[name] = branch
            return {"branch": name, "steps": steps, "frame": self._frame_for(name)}
        if command == "commit" and len(rest) == 1:
            return self.session.commit(self.branches[rest[0]]).to_dict()
        if command == "abort" and len(rest) == 1:
            return self.session.abort(self.branches[rest[0]]).to_dict()
        if command == "save" and len(rest) == 2:
            return {"saved": LocalStore(rest[0]).save(self.cuts[rest[1]])}
        if command == "load" and len(rest) == 3:
            device = getattr(self.session.adapter, "device", None)
            cut = (
                LocalStore(rest[0]).load(rest[1], device=str(device))
                if device is not None
                else LocalStore(rest[0]).load(rest[1])
            )
            self.session._compatible(cut)
            self.cuts[rest[2]] = cut
            return {"cut": rest[2], "fingerprint": cut.fingerprint}
        if command == "help":
            return (
                "inspect / where | read ADDRESS | step / next [N] | break BOUNDARY_PATTERN | "
                "info steps / symbols | resolve SYMBOL | backtrace SLOT | "
                "breakpoints | delete PATTERN / all | continue [N] | until PATTERN [MAX_STEPS] | "
                "watch ADDRESS change / nonfinite | watch ADDRESS gt / lt THRESHOLD | "
                "watchpoints | unwatch ID / all | trace [N] | capture NAME | fork BRANCH [CUT] | "
                "use BRANCH | zero ADDRESS | add ADDRESS JSON | replace ADDRESS JSON | "
                "compare / diff BRANCH_OR_CUT | replay CUT [STEPS] [BRANCH] | commit BRANCH | "
                "diff NATIVE_TRACE CANDIDATE_TRACE --first-divergence | "
                "abort BRANCH | restore CUT | save DIRECTORY CUT | load DIRECTORY DIGEST NAME | quit"
            )
        raise ValueError("unknown command or wrong arguments; use help")

    def _frame_for(self, branch: str) -> dict[str, Any]:
        active = self.active
        self.active = branch
        try:
            return self._frame()
        finally:
            self.active = active


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, branch, and replay native model state")
    sub = parser.add_subparsers(dest="command", required=True)
    debug = sub.add_parser("debug", help="start an offline in-process model debugger")
    debug.add_argument("--family", choices=("ar", "diffusion"), default="ar")
    debug.add_argument("--script", help="read debugger commands from a text file")
    inspect = sub.add_parser("inspect", help="verify a saved metadata descriptor without torch")
    inspect.add_argument("directory")
    inspect.add_argument("digest")
    evidence = sub.add_parser(
        "evidence", help="offline evidence plane: bisect | claims | xref | cite"
    )
    evidence.add_argument("evidence_args", nargs=argparse.REMAINDER)
    route = sub.add_parser("route", help="offline metadata-delta route: validate | select | show")
    route.add_argument("route_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "evidence":
        from .evidence.cli import main as evidence_main

        return evidence_main(args.evidence_args)
    if args.command == "route":
        from .route import main as route_main

        return route_main(args.route_args)
    if args.command == "inspect":
        print(json.dumps(LocalStore(args.directory).inspect(args.digest), indent=2))
        return 0
    if args.family == "ar":
        from .adapters.qwen import QwenAdapter

        session = QwenAdapter.tiny().session([5, 7, 11])
    else:
        from .adapters.diffusion import DiffusionAdapter

        session = DiffusionAdapter.tiny().session()
    debugger = Debugger(session)
    if args.script:
        from pathlib import Path

        for line in Path(args.script).read_text().splitlines():
            if line.strip() in ("quit", "exit"):
                break
            print(json.dumps(debugger.execute(line), indent=2))
        return 0
    print("Saturn local debugger. Random offline model. Type help or quit.")
    while True:
        try:
            line = input(f"{debugger.active}> ")
            if line.strip() in ("quit", "exit"):
                return 0
            print(json.dumps(debugger.execute(line), indent=2))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

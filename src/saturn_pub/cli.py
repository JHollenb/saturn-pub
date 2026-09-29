"""An in-process debugger; every operation delegates to the public Session API."""

from __future__ import annotations

import argparse
import json
import shlex
from fnmatch import fnmatchcase
from typing import Any

from .core import Act, Session
from .store import LocalStore
from .values import describe


class Debugger:
    def __init__(self, session: Session):
        self.branches = {"root": session}
        self.active = "root"
        self.cuts = {}
        self.breakpoints: list[str] = []
        self.max_continue_steps = 1000

    def _until(self, patterns: list[str], limit: int) -> dict[str, Any]:
        if limit < 1:
            raise ValueError("continue limit must be positive")
        # Advance first: continuing from a breakpoint must make forward progress.
        for steps in range(1, limit + 1):
            self.session.continue_()
            boundary = self.session.inspect().boundary
            matches = [pattern for pattern in patterns if fnmatchcase(boundary, pattern)]
            if matches:
                return {
                    "stop": "breakpoint",
                    "steps": steps,
                    "matched": matches,
                    "frame": self.execute("inspect"),
                }
        return {"stop": "step-limit", "steps": limit, "frame": self.execute("inspect")}

    @property
    def session(self) -> Session:
        return self.branches[self.active]

    def execute(self, line: str) -> Any:
        args = shlex.split(line)
        if not args:
            return None
        command, *rest = args
        if command in ("inspect", "where") and not rest:
            frame = self.session.inspect()
            return {"branch": self.active, "boundary": frame.boundary, "slots": dict(frame.slots)}
        if command == "read" and len(rest) == 1:
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
        if command in ("step", "next", "continue") and len(rest) <= 1:
            return self.session.continue_(int(rest[0]) if rest else 1).to_dict()
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
            return {"active": self.active}
        if command == "zero" and len(rest) == 1:
            return self.session.apply(Act.zero(rest[0])).to_dict()
        if command in ("add", "replace") and len(rest) == 2:
            value = json.loads(rest[1])
            current = self.session.read(rest[0])
            if hasattr(current, "shape"):
                import torch

                value = torch.as_tensor(value, device=current.device, dtype=current.dtype)
                if command == "add" and value.ndim == 0:
                    value = torch.ones_like(current) * value
            act = Act.add(rest[0], value) if command == "add" else Act.replace(rest[0], value)
            return self.session.apply(act).to_dict()
        if command == "restore" and len(rest) == 1:
            return self.session.restore(self.cuts[rest[0]]).to_dict()
        if command == "compare" and len(rest) == 1:
            return self.session.compare(self.branches[rest[0]])
        if command == "commit" and len(rest) == 1:
            return self.session.commit(self.branches[rest[0]]).to_dict()
        if command == "abort" and len(rest) == 1:
            return self.session.abort(self.branches[rest[0]]).to_dict()
        if command == "save" and len(rest) == 2:
            return {"saved": LocalStore(rest[0]).save(self.cuts[rest[1]])}
        if command == "load" and len(rest) == 3:
            cut = LocalStore(rest[0]).load(rest[1], device=str(self.session.adapter.device))
            self.session._compatible(cut)
            self.cuts[rest[2]] = cut
            return {"cut": rest[2], "fingerprint": cut.fingerprint}
        if command == "help":
            return (
                "inspect / where | read ADDRESS | step / next [N] | break BOUNDARY_PATTERN | "
                "breakpoints | delete PATTERN / all | continue [N] | until PATTERN [MAX_STEPS] | "
                "capture NAME | fork BRANCH [CUT] | "
                "use BRANCH | zero ADDRESS | add ADDRESS JSON | replace ADDRESS JSON | "
                "compare BRANCH | commit BRANCH | abort BRANCH | restore CUT | save DIRECTORY CUT | "
                "load DIRECTORY DIGEST NAME | quit"
            )
        raise ValueError("unknown command or wrong arguments; use help")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, branch, and replay native model state")
    sub = parser.add_subparsers(dest="command", required=True)
    debug = sub.add_parser("debug", help="start an offline in-process model debugger")
    debug.add_argument("--family", choices=("ar", "diffusion"), default="ar")
    debug.add_argument("--script", help="read debugger commands from a text file")
    inspect = sub.add_parser("inspect", help="verify a saved metadata descriptor without torch")
    inspect.add_argument("directory")
    inspect.add_argument("digest")
    args = parser.parse_args(argv)
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
            print(json.dumps(describe(debugger.execute(line)), indent=2))
        return 0
    print("Saturn local debugger. Random offline model. Type help or quit.")
    while True:
        try:
            line = input(f"{debugger.active}> ")
            if line.strip() in ("quit", "exit"):
                return 0
            print(json.dumps(describe(debugger.execute(line)), indent=2))
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        except Exception as exc:
            print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())

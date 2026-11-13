"""Discovery debugger entry points; metadata browsing stays framework free."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .discovery_script import DiscoveryJournal, ScriptRunner


def load_capture(root):
    """Rehydrate a *root replay recipe*, never a paused native frontier."""
    import torch

    from .discovery_runtime import ReplayForward
    from .forward_recording import ForwardCapture

    root = Path(root)
    manifest = json.loads((root / "capture.json").read_text())
    if manifest.get("schema") != "mdb-discovery-root-recipe-v1":
        raise ValueError("unsupported replay recipe")
    files = {}
    for name in ("trace.json", "roots.pt"):
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]:
            raise ValueError("replay recipe file differs: " + name)
        files[name] = path
    trace = json.loads(files["trace.json"].read_text())
    roots = torch.load(files["roots.pt"], map_location="cpu", weights_only=True)
    return ReplayForward(ForwardCapture(trace, roots, {}, None))


def save_capture(capture, root):
    """Publish root-only replay inputs, with no future intermediate activations."""
    import torch

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "trace.json").write_text(json.dumps(capture.trace, sort_keys=True, allow_nan=False))
    torch.save(capture.initial_values, root / "roots.pt")
    manifest = {"schema": "mdb-discovery-root-recipe-v1", "restore_mode": "replay-recipe",
                "native_frontier_restorable": False,
                "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("trace.json", "roots.pt")}}
    (root / "capture.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb discovery", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    browse = sub.add_parser("inspect", help="Verify and browse a command journal without Torch")
    browse.add_argument("journal", type=Path)
    browse.add_argument("--last", type=int, default=10)
    browse.add_argument("--cut", help="Inspect a durable frontier ID in the supplied store")
    replay = sub.add_parser("replay", help="Run a script over a verified root-only captured forward")
    replay.add_argument("capture", type=Path)
    resume = sub.add_parser("resume", help="Restore a durable replay frontier without executing its prefix")
    resume.add_argument("store", type=Path)
    resume.add_argument("cut_id")
    resume.add_argument("--admission-id", help="Explicit identity of a newly admitted continuation lease")
    resume.add_argument("--budget-json", type=Path, help="All absolute finite ceilings for that continuation")
    attach = sub.add_parser("attach", help="Control a finite resident mrun discovery worker")
    attach.add_argument("job_id")
    attach.add_argument("--endpoint")
    attach.add_argument("--token-env", help="Environment variable containing the broker token")
    for command in (replay, resume, attach):
        input_mode = command.add_mutually_exclusive_group(required=True)
        input_mode.add_argument("--script", type=Path)
        input_mode.add_argument("--command", dest="commands", action="append", help="Repeat for sequential console commands")
        input_mode.add_argument("--interactive", action="store_true", help="Keep live cuts and branches in this console")
        command.add_argument("--output", type=Path, required=True)
    replay.add_argument("--lane", default="lane_00")
    replay.add_argument("--max-ops", type=int, default=131072)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        if args.cut:
            from saturn_pub.store import LocalStore

            from .discovery_storage import inspect_frontier, inspect_session_frontier

            store = LocalStore(args.journal)
            manifest = store.inspect(args.cut)
            if manifest.get("boundary") == "mdb-discovery-durable-session-v1":
                result = inspect_session_frontier(store, args.cut)
            else:
                result = inspect_frontier(store, args.cut)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        rows = DiscoveryJournal(args.journal).verify()
        print(json.dumps({"schema": "mdb-discovery-offline-inspection-v1", "verified_records": len(rows),
                          "restore_mode": "metadata-only", "records": rows[-max(0, args.last):] if args.last else []},
                         indent=2, sort_keys=True, allow_nan=False))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == "attach":
        import os

        from .discovery_remote import RemoteDiscoverySession

        token = os.environ.get(args.token_env) if args.token_env else None
        if args.token_env and not token:
            parser.error("the specified broker token environment variable is empty")
        session = RemoteDiscoverySession.connect(args.job_id, endpoint=args.endpoint, token=token)
    elif args.command == "resume":
        from .discovery_storage import resume_session_frontier

        if bool(args.admission_id) != bool(args.budget_json):
            parser.error("new admission requires both --admission-id and --budget-json")
        options = {} if not args.admission_id else {
            "admission": {"id": args.admission_id},
            "continuation_budget": json.loads(args.budget_json.read_text())}
        session = resume_session_frontier(args.store, args.cut_id, **options)
    else:
        from .discovery import DiscoverySession

        session = DiscoverySession(budgets={"max_native_operations": args.max_ops})
        session.add_lane(args.lane, lambda: load_capture(args.capture))
    runner = ScriptRunner(session, journal=DiscoveryJournal(args.output / "journal"))
    if args.interactive:
        result = {"ok": True}
        while True:
            try:
                line = input(f"(discovery-v1 | {getattr(session, 'focus', args.command)} | unit=law) ")
            except EOFError:
                break
            if line.strip() in {"quit", "exit"}:
                break
            if not line.strip():
                continue
            if line.strip().startswith(("on-hit ", "repeat ")):
                while True:
                    next_line = input("... ")
                    line += "\n" + next_line
                    if next_line.strip() == "end":
                        break
            result = runner.run("interface mdb-discovery-v1\n" + line)
            latest = runner.records[-1] if runner.records else result
            print(json.dumps(latest.get("response", latest), indent=2, sort_keys=True))
    else:
        script = args.script.read_text() if args.script else "interface mdb-discovery-v1\n" + "\n".join(args.commands)
        result = runner.run(script)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"ok": result["ok"], "commands_executed": runner.calls,
                      "handler_hits": runner.hits, "output": str(args.output)}, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

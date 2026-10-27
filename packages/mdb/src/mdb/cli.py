"""One command for standalone sessions, debugging, VM and evidence."""
import argparse
import importlib
import json
import sys

TOOLS = {"cache": "vm_cache", "hyperion": "hyperion", "context": "context_cli",
    "bisect": "bisect_adapters", "xref": "xref_index", "claims": "claims_registry",
    "watch": "receipt_watch", "custody": "artifact_custody", "cite": "evidence_citations",
    "evidence": "evidence_pipeline", "states": "state_annotations", "graph": "state_graph",
    "vm": "vm_cli", "public": "saturn_pub_adapter", "project": "project"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "help":
        from .debugger_cli import HELP
        print(HELP)
        return 0
    if argv and argv[0] in {"open", "resume"} and any(
        item in {"--local", "--tiny"} or item.startswith(("--local=", "--tiny=")) for item in argv):
        from .debugger_cli import main as debug_main
        return debug_main(["debug", *argv[1:], *(["--resume"] if argv[0] == "resume" else [])])
    if argv and argv[0] == "version":
        from . import __version__
        print(__version__)
        return 0
    if argv and argv[0] in {"open", "resume", "inspect", "verify", "bench"}:
        from .bench_cli import main as bench_main
        return bench_main(argv[1:] if argv[0] == "bench" else argv)
    if argv and argv[0] in TOOLS:
        return importlib.import_module("." + TOOLS[argv[0]], __package__).main(argv[1:])
    if argv and argv[0] in {"attach", "jobs", "run", "debug", "script", "doctor", "replay"}:
        if argv[:2] == ["debug", "vm"]:
            from .vm_cli import main as vm_main
            return vm_main(argv[2:])
        from .debugger_cli import main as debug_main
        return debug_main(argv)
    if argv and argv[0] == "capabilities":
        from .capabilities import inventory
        print(json.dumps(inventory(), indent=2, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(prog="mdb", description="Standalone model virtual memory and reversible model debugger")
    parser.add_argument("command", nargs="?", choices=["open", "resume", "inspect", "verify", "attach", "debug", "doctor", "help", "jobs", "run", "script", "replay", "bench", "vm", "cache", "hyperion", "context", "capabilities", "version", *[k for k in TOOLS if k not in {"vm", "cache", "hyperion", "context"}]])
    parser.parse_args(argv)
    parser.print_help()
    return 0

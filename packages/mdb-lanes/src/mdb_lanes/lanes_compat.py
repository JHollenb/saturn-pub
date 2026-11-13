"""COMPATIBILITY SHIM -- the lane side of the engine/lanes plug-in boundary.

The engine (``mdb.core``/``engine``/``analysis``/``ops``/``repl``/``serve``, ``mdb/__init__.py``,
``mdb/cli.py``) never imports a lane module by name. It finds lane verbs and lane exports through
two entry-point groups (``mdb.core.plugins``):

* ``mdb.lanes``: every verb in ``VERBS`` -> ``<this module>:main``;
* ``mdb.lanes.exports``: every name in ``EXPORTS`` -> the lane ``module:attribute`` it names.

This module is lane code. ``tools/split_lanes.py apply`` moves it to ``mdb_lanes.lanes_compat``
with the other lane modules and writes both groups from the two tables below into
``packages/mdb-lanes/pyproject.toml``. Until the split, and for source-tree runs where nothing
is installed (``PYTHONPATH=.../src``), the groups come from the stub distribution
``mdb_lanes-0.1.0.dist-info`` next to the package; ``python -m mdb_lanes.lanes_compat --check``
verifies that stub matches these tables, ``--write`` regenerates it. Module references below are
relative to this module's package, so they stay correct after the move.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

STUB = "mdb_lanes-0.1.0.dist-info"

# verb -> lane module whose ``main(argv[1:])`` runs it
TOOLS = {
    "cache": "vm_cache", "hyperion": "hyperion", "context": "context_cli",
    "analyze": "native_discovery", "bisect": "bisect_adapters", "xref": "xref_index",
    "claims": "claims_registry", "watch": "receipt_watch", "custody": "artifact_custody",
    "cite": "evidence_citations", "evidence": "evidence_pipeline", "states": "state_annotations",
    "graph": "state_graph", "vm": "vm_cli", "public": "saturn_pub_adapter", "project": "project",
    "symbols": "symbol_lifecycle_project", "symbol-trace": "symbol_trace",
    "symbol-producers": "symbol_producers", "forward-symbols": "forward_symbols",
    "circuits": "circuit_explorer", "discovery": "discovery_cli", "ports": "forward_ports",
    "forecast": "forecast_transaction", "equations": "equation_comparison",
    "compiled-state": "compiled_state", "storage": "storage_cli", "edit-search": "edit_search",
    "view": "evidence_viewer", "family": "computation_protocol", "atlas": "computation_atlas",
    "wsp": "wsp",
}
BENCH_VERBS = ("open", "resume", "inspect", "verify", "bench")
DEBUGGER_VERBS = ("attach", "jobs", "run", "debug", "script", "doctor", "replay")
VERBS = sorted({*TOOLS, *BENCH_VERBS, *DEBUGGER_VERBS, "help", "capabilities"})

# attribute of the ``mdb`` package -> "lane_module:attribute"
EXPORTS = {
    "BenchSession": "bench:BenchSession",
    "SaturnPubDebuggerAdapter": "saturn_pub_adapter:SaturnPubDebuggerAdapter",
    "Project": "project:Project",
    "BenchError": "errors:BenchError",
    "ModelRunner": "backends:ModelRunner",
    "RunnerCapabilities": "backends:RunnerCapabilities",
    "ForwardResult": "backends:ForwardResult",
    "register_backend": "backends:register_backend",
    "available_backends": "backends:available_backends",
    "TransformerLensRunner": "backend_torch:TransformerLensRunner",
    "TransformerLensDebuggerAdapter": "lens_debugger:TransformerLensDebuggerAdapter",
    "TransformersRunner": "backend_torch:TransformersRunner",
    "CausalContextVM": "causal_context_vm:CausalContextVM",
    "MambaStateVM": "mamba_state_vm:MambaStateVM",
    **{name: "state_graph:" + name for name in (
        "StateGraph", "GraphNode", "GraphEdge", "GraphIntegrityError", "GraphSnapshot",
        "GraphVerification", "NodeBinding", "Evidence", "StateGraphError", "ExecutionBinding",
        "GraphBinding", "SourceNode", "StateNode", "ActionNode", "OutputNode", "evidence")},
}


def _module(name):
    return importlib.import_module("." + name, __package__)


def main(argv):
    """Run lane verb ``argv[0]`` with its arguments (the old ``mdb`` dispatch, unchanged)."""
    argv = list(argv)
    verb = argv[0] if argv else ""
    if verb == "help":
        print(_module("debugger_cli").HELP)
        return 0
    if verb in {"open", "resume"} and any(
            item in {"--local", "--tiny"} or item.startswith(("--local=", "--tiny="))
            for item in argv):
        return _module("debugger_cli").main(
            ["debug", *argv[1:], *(["--resume"] if verb == "resume" else [])])
    if verb in BENCH_VERBS:
        return _module("bench_cli").main(argv[1:] if verb == "bench" else argv)
    if verb in TOOLS:
        return _module(TOOLS[verb]).main(argv[1:])
    if verb in DEBUGGER_VERBS:
        if argv[:2] == ["debug", "vm"]:
            return _module("vm_cli").main(argv[2:])
        return _module("debugger_cli").main(argv)
    if verb == "capabilities":
        print(json.dumps(_module("capabilities").inventory(), indent=2, sort_keys=True))
        return 0
    print(f"mdb lanes: unknown verb {verb!r}", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- stub distribution

def entry_points_text(package=None):
    """``entry_points.txt`` declaring both groups for the package this module lives in."""
    package = package or __package__
    lines = ["[mdb.lanes]"]
    lines += [f"{verb} = {package}.lanes_compat:main" for verb in VERBS]
    lines += ["", "[mdb.lanes.exports]"]
    lines += [f"{name} = {package}.{target}" for name, target in sorted(EXPORTS.items())]
    return "\n".join(lines) + "\n"


def metadata_text():
    return ("Metadata-Version: 2.1\nName: mdb-lanes\nVersion: 0.1.0\n"
            "Summary: Source-tree metadata of mdb-lanes: with this src directory on sys.path the "
            "lane entry points and version resolve as for an editable install (see "
            "lanes_compat.py). A built wheel carries its own metadata.\n")


def stub_dir():
    """The stub's directory: the ``src`` directory holding this module's top package."""
    return Path(__file__).resolve().parents[len(__package__.split("."))] / STUB


def stub_files():
    return {"METADATA": metadata_text(), "entry_points.txt": entry_points_text()}


if __name__ == "__main__":
    target = stub_dir()
    if sys.argv[1:] == ["--write"]:
        target.mkdir(exist_ok=True)
        for name, text in stub_files().items():
            (target / name).write_text(text)
        print(f"wrote {target}")
    elif sys.argv[1:] == ["--check"]:
        stale = [n for n, t in stub_files().items()
                 if not (target / n).is_file() or (target / n).read_text() != t]
        print("stale: " + ", ".join(stale) if stale else f"{target} is current")
        sys.exit(1 if stale else 0)
    else:
        print("usage: python -m mdb_lanes.lanes_compat --check | --write", file=sys.stderr)
        sys.exit(2)

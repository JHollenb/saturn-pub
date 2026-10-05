"""Extract only the existing finite Mamba page-op CLI from the research CLI."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root.parent / "saturn/src/saturn/debug.py"
tree = ast.parse(source.read_text())
names = {"_vm_build_config", "_vm_request", "_vm_collect", "_vm_submit", "_vm_add_common", "_add_vm_group"}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
header = '''"""Finite Mamba VM page assays, distinct from a live bench session."""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any
from .job_support import terminal_job, decode_marker, worker_command, stage_worker, guarded_mrun_launch, one_lease_scheduling_contract
from .paths import data_root
ROOT = data_root()
_VM_WORKER = Path(__file__).parent / "workers/run_saturn_mvm_op.py"
_VM_MARKER = "SATURN_MVM_OP_GZIP_B64="
_VM_MODES = ("read-formed-state", "residual-write", "forced-formed-install")
'''
text = header + "\n\n".join(ast.unparse(node) for node in nodes)
text = text.replace("default=10000.0", "default=None").replace("default=2000.0", "default=None")
text += '''

def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb vm", description=__doc__)
    # Reuse the complete original VM group and its parameter contracts.
    subs = parser.add_subparsers(dest="group", required=True)
    _add_vm_group(subs)
    args = parser.parse_args(["vm", *(argv or [])])
    return args.handler(args)
'''
(root / "src/mdb/vm_cli.py").write_text(text)

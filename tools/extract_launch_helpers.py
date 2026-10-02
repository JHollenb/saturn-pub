"""Keep existing admission/collection contracts without workspace staging."""
import ast
import hashlib
import json
from pathlib import Path

source = Path(__file__).resolve().parents[2] / "saturn/src/saturn/job_support.py"
destination = Path(__file__).resolve().parents[1] / "src/mdb/job_support.py"
names = {
    "_reject_nested_submission", "_mrun_launch", "reject_staged_target_collisions",
    "source_hash_manifest", "guarded_mrun_launch", "one_lease_scheduling_contract",
    "one_lease_restart_proof_scheduling_contract", "one_lease_sequential_models_scheduling_contract",
    "_request_sha256", "request_sha256", "terminal_job", "guarded_job_receipt", "decode_marker",
}
tree = ast.parse(source.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "scheduling_contract"]
nodes += [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
header = '''"""Package-independent mrun admission and direct collection contracts."""
from __future__ import annotations
import base64
import gzip
import hashlib
import importlib
import inspect
import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from .local_data_plane import reject_external_operation
from .tools import configure_workspace, scheduler_api
from .payload import stage_worker, worker_command
_TERMINAL_STATES = {"succeeded", "failed", "killed_ram", "killed_vram", "timeout", "cancelled", "lost"}
RESIDENT_WORKER_ENV = "SATURN_RESIDENT_WORKER"
'''
destination.write_text(header + "\n\n".join(ast.unparse(node) for node in nodes) + "\n")
manifest = destination.parents[2] / "PROVENANCE.json"
value = json.loads(manifest.read_text())
value["launch_helpers"] = {"original_path": "src/saturn/job_support.py",
    "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "functions": sorted(names),
    "extracted_path": "src/mdb/job_support.py", "extracted_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
manifest.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

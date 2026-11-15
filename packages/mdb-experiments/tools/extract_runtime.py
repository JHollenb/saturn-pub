"""Reproducible extraction of the VM's closed Python source graph.

This is a maintainer tool, not an MDB runtime dependency. The input checkout
is explicit; installed MDB never discovers or imports that checkout.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOTS = (
    "bench", "bench_cli", "bench_debugger", "bench_verify", "causal_context_vm",
    "mamba_state_vm", "mvm_worker_support", "vm_cache", "vm_payload_cache",
    "debugger_cli", "debugger_runtime", "evidence_pipeline", "bisect_adapters",
    "evidence_citations", "hyperion", "model_os_kernel", "source_frame_stack",
    "context_register", "future_forest", "receipt_watch", "claims_registry",
    "xref_index", "evidence_specs", "divergence_bisect", "artifact_custody",
    "scheduling_contract",
)
# These workflow modules are replaced by package-aware implementations. Pulling
# in their research command trees would defeat the extraction boundary.
REPLACED = {"saturn", "saturn.tools", "saturn.job_support", "saturn.debug"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def rewrite(text):
    text = re.sub(r"(?m)^(\s*(?:from|import)\s+)saturn(?=[.\s])", r"\1mdb", text)
    text = re.sub(r"(import_module\([\"'])saturn(?=[.\"'])", r"\1mdb", text)
    text = re.sub(r"([\"']module[\"']\s*:\s*[\"'])saturn(?=\.)", r"\1mdb", text)
    return text


def graph(source):
    paths = {}
    for path in (source / "src/saturn").rglob("*.py"):
        parts = path.relative_to(source / "src").with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        paths[module] = path
    edges = {}
    for module, path in paths.items():
        base = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        targets = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = ".".join(base[:len(base) - node.level + 1]) if node.level else ""
                name = ".".join(value for value in (prefix, node.module) if value)
                names = [name, *[name + "." + alias.name for alias in node.names]]
            else:
                continue
            targets.update(name for name in names if name in paths and name not in REPLACED)
        edges[module] = targets
    return paths, edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    paths, edges = graph(source)
    pending = ["saturn." + name for name in ROOTS]
    # Bench modules contain lazy family/discovery imports; include every member
    # of this deliberately bounded subsystem, then close their actual imports.
    pending += [module for module in paths if module.startswith("saturn.bench")]
    seen = set()
    while pending:
        module = pending.pop()
        if module in seen or module in REPLACED:
            continue
        seen.add(module)
        pending.extend(edges[module] - seen)
    records = []
    for module in sorted(seen):
        original = paths[module]
        relative = original.relative_to(source / "src/saturn")
        target = destination / "src/mdb" / relative
        raw = original.read_bytes()
        transformed = rewrite(raw.decode()).encode()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(transformed)
        records.append({"module": module, "original_path": original.relative_to(source).as_posix(),
            "original_sha256": digest(raw), "extracted_path": target.relative_to(destination).as_posix(),
            "extracted_sha256": digest(transformed)})
    for path in (destination / "src/mdb").rglob("*.py"):
        if path.parent != destination / "src/mdb":
            path.parent.joinpath("__init__.py").touch(exist_ok=True)
    workers = []
    for name in ("run_saturn_bench.py", "run_saturn_mvm_op.py"):
        original = source / "workers" / name
        target = destination / "src/mdb/workers" / name
        target.parent.mkdir(exist_ok=True)
        raw = original.read_bytes()
        transformed = rewrite(raw.decode()).encode()
        target.write_bytes(transformed)
        workers.append({"original_path": "workers/" + name, "original_sha256": digest(raw),
            "extracted_path": target.relative_to(destination).as_posix(), "extracted_sha256": digest(transformed)})
    (destination / "src/mdb/workers/__init__.py").touch()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"schema": "mdb-extraction-provenance-v1", "source_revision": revision,
        "source_kind": "working-tree-including-required-untracked-files", "source_roots": list(ROOTS),
        "replaced_workflow_modules": sorted(REPLACED), "files": records + workers}
    (destination / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"modules": len(seen), "source_bytes": sum(paths[m].stat().st_size for m in seen)}))


if __name__ == "__main__":
    main()

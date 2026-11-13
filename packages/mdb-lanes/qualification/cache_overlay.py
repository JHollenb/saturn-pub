"""Prepare writable uv metadata on ext4 while retaining existing cached payloads.

Operational maintenance only: no model execution, package changes or cache prune.
The original archive payloads must remain available for the worker lifetime.
"""
from hashlib import sha256
import json
from pathlib import Path
import shutil


def main():
    source = Path("/mnt/big/llm-models/.uv-cache")
    destination = Path("/mnt/hdd8tb/mdb/uv-cache")
    staging = destination.with_name("uv-cache-staging")
    if destination.exists() or staging.exists():
        raise RuntimeError("cache overlay destination must be absent")
    staging.mkdir(parents=True)
    archive = staging/"archive-v0"; archive.mkdir()
    linked = []
    for entry in sorted((source/"archive-v0").iterdir()):
        if not entry.is_dir():
            continue
        (archive/entry.name).symlink_to(entry, target_is_directory=True)
        linked.append(entry.name)
    copied = []
    for name in ("wheels-v6", "simple-v21", "simple-v24", "interpreter-v4"):
        if (source/name).is_dir():
            shutil.copytree(source/name, staging/name, symlinks=True)
            copied.append(name)
    for name in ("CACHEDIR.TAG", ".gitignore"):
        if (source/name).is_file():
            shutil.copy2(source/name, staging/name)
    metadata = {p.relative_to(staging).as_posix(): sha256(p.read_bytes()).hexdigest()
        for name in copied for p in sorted((staging/name).rglob("*")) if p.is_file() and not p.is_symlink()}
    report = {"schema": "mdb-uv-cache-overlay-v1", "source": str(source), "destination": str(destination),
        "archive_links": linked, "metadata_sha256": metadata, "copied_categories": copied,
        "boundary": "original archives required; new environments and extractions write to ext4; original cache unchanged"}
    staging.rename(destination)
    output = destination.parent/"cache-overlay.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"destination": str(destination), "archive_links": len(linked),
        "metadata_files": len(metadata), "report": str(output)}), flush=True)


if __name__ == "__main__": main()

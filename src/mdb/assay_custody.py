"""Deletion evidence restricted to the current assay's private source stores."""
from hashlib import sha256
from pathlib import Path
import re
import shutil


def remove_source_store(path, job_root):
    root = Path(job_root)
    source = Path(path)
    if root.parent != Path("/mnt/big/mdb/qualification") or not re.fullmatch(r"job-[0-9a-f]{12}", root.name):
        raise ValueError("source deletion requires a canonical assay job root")
    if source.is_symlink() or source.resolve() == root.resolve() or not source.resolve().is_relative_to(root.resolve()):
        raise ValueError("source deletion is outside the private assay store")
    selected = source / "selected.json"
    before = sha256(selected.read_bytes()).hexdigest()
    shutil.rmtree(source)
    return {"root": str(root), "source": str(source), "selected_sha256": before,
            "removed": not source.exists()}

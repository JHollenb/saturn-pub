"""Find a model checkpoint on this machine by name (standard library only; never downloads).

``locate_model("qwen2.5-1.5b")`` returns the directory of a local Hugging Face checkpoint:
a directory path as given, or a snapshot in the local Hugging Face cache whose repository id
matches the name. Matching is case-insensitive and tried in order: the full id
(``Qwen/Qwen2.5-1.5B-Instruct``), the id without its organisation, then a substring that
only one cached repository contains. A name that matches none, or several, is refused with the
candidates listed, so a short name never silently picks a model.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import BadArgs


def cache_dirs() -> list[Path]:
    """The Hugging Face hub cache directories that exist, in lookup order."""
    found = []
    if os.environ.get("HF_HUB_CACHE"):
        found.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        found.append(Path(os.environ["HF_HOME"]) / "hub")
    found.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen, out = set(), []
    for path in found:
        path = path.expanduser()
        if path not in seen and path.is_dir():
            seen.add(path)
            out.append(path)
    return out


def _snapshot(repo: Path) -> Path | None:
    """The snapshot ``refs/main`` names, else the newest one; only one that has a config."""
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = repo / "refs" / "main"
    if ref.is_file():
        named = snapshots / ref.read_text().strip()
        if (named / "config.json").is_file():
            return named
    usable = [path for path in snapshots.iterdir() if (path / "config.json").is_file()]
    return max(usable, key=lambda path: path.stat().st_mtime) if usable else None


def cached_models() -> dict[str, Path]:
    """``repository id -> snapshot directory`` of every cached model that has a config."""
    out: dict[str, Path] = {}
    for cache in cache_dirs():
        for repo in sorted(cache.glob("models--*")):
            identifier = repo.name[len("models--"):].replace("--", "/")
            snapshot = _snapshot(repo)
            if snapshot is not None:
                out.setdefault(identifier, snapshot)
    return out


def locate_model(name: str) -> Path:
    """The checkpoint directory for ``name`` (a directory, or a cached repository's name)."""
    text = str(name).strip()
    if not text:
        raise BadArgs("no model given: pass a checkpoint directory or a cached model's name")
    path = Path(text).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise BadArgs(f"{path} is not a Hugging Face checkpoint directory "
                          "(config.json missing)")
        return path.resolve()
    models = cached_models()
    wanted = text.lower()
    tiers = (
        [identifier for identifier in models if identifier.lower() == wanted],
        [identifier for identifier in models if identifier.lower().split("/")[-1] == wanted],
        [identifier for identifier in models if wanted in identifier.lower()],
    )
    for found in tiers:
        if len(found) == 1:
            return models[found[0]]
        if found:
            raise BadArgs(f"model name {text!r} matches {len(found)} cached models: "
                          f"{', '.join(sorted(found))}; give more of the name")
    where = ", ".join(str(cache) for cache in cache_dirs()) or "no Hugging Face cache found"
    known = ", ".join(sorted(models)[:12]) + (" ..." if len(models) > 12 else "")
    raise BadArgs(f"no directory and no cached model named {text!r} ({where}); nothing is "
                  f"downloaded. Cached: {known or 'none'}")

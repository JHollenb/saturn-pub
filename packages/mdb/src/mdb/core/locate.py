"""Find a model checkpoint on this machine by name (standard library only; never downloads).

``locate_model("qwen2.5-1.5b")`` returns the directory of a local Hugging Face checkpoint:
a directory path as given, or a snapshot in the local Hugging Face cache whose repository id
matches the name. Matching is case-insensitive and tried in order: the full id
(``Qwen/Qwen2.5-1.5B-Instruct``), the id without its organisation, then a substring that
only one cached repository contains. A name that matches none, or several, is refused with the
candidates listed, so a short name never silently picks a model.

Settings (``mdb.core.settings``): ``[model] aliases`` maps a short name to a directory or a
cached name (tried first); ``[model] roots`` (or ``MDB_MODEL_ROOTS``) lists directories whose
checkpoints (``ROOT/NAME`` or ``ROOT/ORG/NAME`` with a config.json) are searched with the cache,
the cache winning a tie on the same id.

``parse_target`` is the one reading of "which model" every front end uses: ``tiny:FAMILY``,
an ``http(s)://`` address of a running ``mdb serve``, or a directory or name for
``locate_model``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .errors import BadArgs
from .settings import TINY_FAMILIES


def parse_target(text: str) -> tuple[str, str]:
    """``("tiny", FAMILY)``, ``("connect", URL)`` or ``("model", NAME_OR_DIR)``; a configured
    alias is replaced by its target first (so an alias may name ``tiny:qwen2``)."""
    from . import settings

    text = str(text).strip()
    text = settings.current().get("model.aliases").get(text, text)
    if text.startswith(("http://", "https://")):
        return "connect", text
    if text.startswith("tiny:"):
        family = text[len("tiny:"):]
        if family not in TINY_FAMILIES:
            raise BadArgs("tiny models are " + ", ".join(f"tiny:{name}" for name in TINY_FAMILIES))
        return "tiny", family
    return "model", text


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


def model_roots() -> list[Path]:
    """The configured model roots that exist (``[model] roots``, ``MDB_MODEL_ROOTS``)."""
    from . import settings

    return [Path(root).expanduser() for root in settings.current().get("model.roots")
            if Path(root).expanduser().is_dir()]


def root_models() -> dict[str, Path]:
    """``NAME`` or ``ORG/NAME`` -> directory of every checkpoint (with a config.json) one or
    two levels under a model root."""
    out: dict[str, Path] = {}
    for root in model_roots():
        for config in sorted([*root.glob("*/config.json"), *root.glob("*/*/config.json")]):
            out.setdefault(config.parent.relative_to(root).as_posix(), config.parent)
    return out


def known_models() -> dict[str, Path]:
    """The cached models, then the model roots' (the cache wins a tie on one id)."""
    out = cached_models()
    for identifier, path in root_models().items():
        out.setdefault(identifier, path)
    return out


def locate_model(name: str) -> Path:
    """The checkpoint directory for ``name`` (a directory, or a cached repository's name)."""
    text = str(name).strip()
    if not text:
        raise BadArgs("no model given: pass a checkpoint directory or a cached model's name")
    from . import settings

    aliases = settings.current().get("model.aliases")
    if text in aliases and aliases[text] != text:
        target = aliases[text]
        if target in aliases:
            raise BadArgs(f"model alias {text!r} names another alias {target!r}; aliases are "
                          "one step (a directory or a model name)")
        return locate_model(target)
    path = Path(text).expanduser()
    if path.is_dir():
        if not (path / "config.json").is_file():
            raise BadArgs(f"{path} is not a Hugging Face checkpoint directory "
                          "(config.json missing)")
        return path.resolve()
    models = known_models()
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
            raise BadArgs(f"model name {text!r} matches {len(found)} local models: "
                          f"{', '.join(sorted(found))}; give more of the name")
    where = ", ".join(str(place) for place in [*cache_dirs(), *model_roots()]) or \
        "no Hugging Face cache or model root found"
    known = ", ".join(sorted(models)[:12]) + (" ..." if len(models) > 12 else "")
    raise BadArgs(f"no directory and no cached model named {text!r} ({where}); nothing is "
                  f"downloaded. Cached: {known or 'none'}")

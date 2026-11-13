"""Explicit maintainer extraction of ordinary source documents, without annotations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mdb.native_discovery import digest, file_hash, prepare_corpus, save_json


def build(source_root, out, *, count=20, skip=0, seed=20261112, exclude_corpus=(), one_per_basename=False):
    """``one_per_basename`` keeps one file per content hash and per file name, and drops file names
    used by any excluded corpus: versions of one script under several paths are near-duplicates
    that content hashing alone does not group."""
    source_root = source_root.resolve()
    # Only an explicitly supplied public source tree is accessed by this extractor.
    paths = [p for directory in (source_root / "packages/mdb/docs", source_root / "src/saturn_pub")
             for p in directory.rglob("*") if p.suffix in {".md", ".py"} and p.stat().st_size >= 512]
    paths.sort(key=lambda p: digest([seed, str(p.relative_to(source_root))]))
    # Old corpora grouped by relative path; new ones group by file content hash. Exclude both.
    excluded = {value for corpus in exclude_corpus
                for row in json.loads(corpus.read_text())["documents"]
                for value in (row["group"], row.get("source"))}
    for corpus in exclude_corpus:  # sidecars record content hashes for path-grouped corpora
        sidecar = corpus.with_suffix(".sources.json")
        if sidecar.exists():
            excluded |= {"sha256:" + s["sha256"] for s in json.loads(sidecar.read_text())["sources"]}
    paths = [p for p in paths if p.relative_to(source_root).as_posix() not in excluded
             and "sha256:" + file_hash(p) not in excluded]
    dropped_by_name = 0
    if one_per_basename:
        names = {Path(v).name for v in excluded if v and not v.startswith("sha256:")}
        hashes, kept = set(), []
        for p in paths:
            h = file_hash(p)
            if p.name in names or h in hashes:
                dropped_by_name += 1
                continue
            names.add(p.name)
            hashes.add(h)
            kept.append(p)
        paths = kept
    paths = paths[skip:skip + count]
    if len(paths) != count:
        raise ValueError("not enough source documents")
    rows, sources = [], []
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        text = path.read_text()
        # Byte-identical files under different paths share one group, so they share one split.
        group = "sha256:" + file_hash(path)
        # Seeded character window is a collection budget, not a semantic address.
        start = int(hashlib.sha256((str(seed) + relative).encode()).hexdigest(), 16) % max(1, len(text) - 4096)
        start = text.find("\n", start) + 1 if start else 0
        window = text[start:start + 4096]
        rows.append({"id": digest(relative)[:20], "text": window, "source": relative, "group": group})
        sources.append({"source": relative, "sha256": file_hash(path), "characters": [start, start + len(window)]})
    save_json(out, prepare_corpus(rows, seed=seed))
    save_json(out.with_suffix(".sources.json"), {"seed": seed, "skip": skip, "count": count,
              "selection": "hash-ranked ordinary markdown/python files; deterministic character windows",
              "split_group": "file content sha256 (byte-identical files cannot cross splits)",
              "source_root": str(source_root), "sources": sources,
              "one_per_basename": bool(one_per_basename), "dropped_by_name_or_content": dropped_by_name,
              "excluded_corpora": {str(p): file_hash(p) for p in exclude_corpus}})
    return {"corpus": str(out), "documents": count, "source_groups": len(sources)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--exclude-corpus", type=Path, nargs="*", default=())
    p.add_argument("--one-per-basename", action="store_true")
    print(json.dumps(build(**vars(p.parse_args())), sort_keys=True))

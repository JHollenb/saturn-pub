"""Automatic analysis of ordinary native executions; optional numeric imports are lazy."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = "mdb-native-discovery-v1"
DOCUMENT_FIELDS = {"id", "text", "source", "group"}


class DiscoveryError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def sealed(value):
    return {**value, "fingerprint": digest(value)}


def checked(value):
    body = dict(value)
    fingerprint = body.pop("fingerprint", None)
    if digest(body) != fingerprint:
        raise DiscoveryError("record fingerprint differs")
    return body


def document(value):
    if set(value) - DOCUMENT_FIELDS:
        raise DiscoveryError("documents accept only id, text, source and group; no labels or spans")
    if any(not isinstance(value.get(k), str) or not value[k].strip() for k in ("id", "text")):
        raise DiscoveryError("document needs nonempty id and text")
    for k in ("source", "group"):
        if k in value and (not isinstance(value[k], str) or not value[k].strip()):
            raise DiscoveryError("source and group must be nonempty text")
    return {**value, "source": value.get("source", value["id"]),
            "group": value.get("group", value.get("source", value["id"]))}


def prepare_corpus(documents, *, seed=0, near_duplicate=0.8):
    """Deduplicate before group-level splitting; assignments are frozen in the manifest."""
    if not 0 < near_duplicate <= 1:
        raise DiscoveryError("near_duplicate must be in (0, 1]")
    rows = [document(row) for row in documents]
    if len({row["id"] for row in rows}) != len(rows):
        raise DiscoveryError("duplicate document ID")
    kept, rejected, seen = [], [], []
    for row in sorted(rows, key=lambda r: r["id"]):
        words = re.findall(r"\w+", row["text"].casefold())
        normalized = " ".join(words)
        if not normalized:
            raise DiscoveryError("document contains no words")
        shingles = {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}
        match = next((old_id for old_id, old_text, old in seen
                      if normalized == old_text or
                      len(shingles & old) / max(1, len(shingles | old)) >= near_duplicate), None)
        if match is not None:
            rejected.append({"id": row["id"], "duplicate_of": match})
            continue
        seen.append((row["id"], normalized, shingles))
        kept.append({**row, "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()})
    groups = sorted({r["group"] for r in kept}, key=lambda g: digest([seed, g]))
    if len(groups) < 5:
        raise DiscoveryError("at least five distinct document groups are needed after deduplication")
    n_dev = max(1, len(groups) // 5)
    assignment = {g: ("development" if i < n_dev else
                     "heldout" if i < 2 * n_dev else "discovery")
                  for i, g in enumerate(groups)}
    result = {"schema": SCHEMA + "/corpus", "seed": seed,
              "near_duplicate_jaccard": near_duplicate, "deduplicated": rejected,
              "documents": [{**r, "split": assignment[r["group"]]} for r in kept],
              "split_unit": "source document group; exact/word-shingle dedup before allocation"}
    return sealed(result)


def load_corpus(path):
    corpus = json.loads(Path(path).read_text())
    body = checked(corpus)
    if body.get("schema") != SCHEMA + "/corpus":
        raise DiscoveryError("unsupported corpus schema")
    ids, groups, hashes = set(), {}, set()
    for row in corpus["documents"]:
        document({k: row[k] for k in DOCUMENT_FIELDS if k in row})
        if row["id"] in ids or row["text_sha256"] in hashes:
            raise DiscoveryError("duplicate corpus row")
        if row["split"] not in {"discovery", "development", "heldout"}:
            raise DiscoveryError("invalid split")
        if groups.setdefault(row["group"], row["split"]) != row["split"]:
            raise DiscoveryError("a source group crosses splits")
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise DiscoveryError("document bytes changed")
        ids.add(row["id"])
        hashes.add(row["text_sha256"])
    if set(groups.values()) != {"discovery", "development", "heldout"}:
        raise DiscoveryError("corpus must contain discovery, development and heldout groups")
    return corpus


SHARED_PREFIX_REFUSAL = 8


def cross_split_prefixes(windows, *, refuse_at=SHARED_PREFIX_REFUSAL):
    """Longest shared leading token run between visible windows of different splits.

    ``windows``: list of ``(split, token_ids)``. A causal model gives identical states at every
    position of a shared leading run, so a held-out window that begins with a discovery window's
    first ``k`` tokens leaks ``k`` positions. Raises when any cross-split pair shares
    ``min(refuse_at, len(shorter window))`` or more leading tokens (identical windows and whole
    short windows that prefix a longer one are always refused). Returns the per-window longest
    cross-split shared prefix length.
    """
    if type(refuse_at) is not int or refuse_at < 1:
        raise DiscoveryError("shared-prefix refusal length must be a positive integer")
    rows = [(split, tuple(int(t) for t in tokens)) for split, tokens in windows]
    longest = [0] * len(rows)
    for i, (split_i, a) in enumerate(rows):
        for j in range(i + 1, len(rows)):
            split_j, b = rows[j]
            if split_i == split_j:
                continue
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            longest[i], longest[j] = max(longest[i], n), max(longest[j], n)
            if n >= min(refuse_at, len(a), len(b)):
                raise DiscoveryError(
                    f"visible token windows from {split_i} and {split_j} share their first {n} tokens")
    return longest


def save_arrays(path, **arrays):
    import numpy as np
    for key, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise DiscoveryError("only finite non-object numerical arrays are permitted: " + key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": path.name, "sha256": file_hash(path),
            "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in arrays.items()}}


def read_arrays(root, descriptor):
    import numpy as np
    path = Path(root) / descriptor["path"]
    if Path(descriptor["path"]).is_absolute() or ".." in Path(descriptor["path"]).parts:
        raise DiscoveryError("array path escapes artifact")
    if file_hash(path) != descriptor["sha256"]:
        raise DiscoveryError("array artifact changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(descriptor["arrays"]):
            raise DiscoveryError("array inventory differs")
        arrays = {k: archive[k] for k in archive.files}
    for k, a in arrays.items():
        if (a.dtype.kind not in "biuf" or not np.isfinite(a).all()
                or list(a.shape) != descriptor["arrays"][k]["shape"]
                or str(a.dtype) != descriptor["arrays"][k]["dtype"]):
            raise DiscoveryError("array contract differs: " + k)
    return arrays


def error_metrics(prediction, target):
    import numpy as np
    a, b = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise DiscoveryError("metric requires matching finite arrays")
    mse = float(np.square(a - b).mean())
    energy = float(np.square(b).mean())
    result = {"mse": mse, "relative_mse": mse / max(energy, 1e-20),
              "max_abs": float(np.abs(a - b).max()), "scalars": int(b.size)}
    # Additive companions (positions on axis 0): the uncentred ratio above can be a pure
    # position-0 score when the first token carries a massive activation.
    from .analysis_metrics import relative_error
    extra = relative_error(a, b)
    for key in ("centered_relative_mse", "relative_mse_excluding_position_0",
                "centered_relative_mse_excluding_position_0", "position_0_energy_fraction"):
        result[key] = extra[key]
    return result


def verify_run(root):
    """Framework-free artifact verification; forecasts retain the public verifier."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    checked(manifest)
    if manifest.get("schema") != SCHEMA + "/run":
        raise DiscoveryError("unsupported run schema")
    for relative, expected in manifest["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise DiscoveryError("manifest path escapes run")
        if (root / relative).is_symlink() or not (root / relative).resolve().is_relative_to(root.resolve()):
            raise DiscoveryError("artifact escapes run: " + relative)
        if file_hash(root / relative) != expected:
            raise DiscoveryError("artifact changed: " + relative)
    corpus = load_corpus(root / "corpus.json")
    model = json.loads((root / "predictor" / "model.json").read_text())
    checked(model)
    if manifest["corpus"] != corpus["fingerprint"] or manifest["predictor"] != model["fingerprint"]:
        raise DiscoveryError("run manifest identities differ")
    custody = root / "parameter-custody.json"
    if custody.exists():
        record = json.loads(custody.read_text())
        checked(record)
        if record["run_manifest_sha256"] != file_hash(root / "manifest.json"):
            raise DiscoveryError("parameter custody run identity differs")
        if set(record["files"]) != {"parameter-artifacts.json", "retrieve_weights.py"}:
            raise DiscoveryError("parameter custody inventory differs")
        for name, sha in record["files"].items():
            if file_hash(root / name) != sha:
                raise DiscoveryError("parameter custody artifact changed")
        checked(json.loads((root / "parameter-artifacts.json").read_text()))
    discovery_ids = {r["id"] for r in corpus["documents"] if r["split"] == "discovery"}
    development_ids = {r["id"] for r in corpus["documents"] if r["split"] == "development"}
    if set(model["fitted_document_ids"]) != discovery_ids:
        raise DiscoveryError("predictor fitting split differs")
    if set(model["selection_document_ids"]) != development_ids:
        raise DiscoveryError("predictor selection split differs")
    from .native_discovery_export import load_catalog_bank
    if (root / "knowledge-bank.json").exists():
        load_catalog_bank(root / "knowledge-bank.json")
    from .forecast_transaction import verify_transaction
    forecasts = []
    for path in sorted((root / "forecasts").glob("*")):
        if path.is_dir():
            report = verify_transaction(path)
            events = [json.loads(p.read_text())["kind"] for p in sorted((path / "events").glob("*.json"))]
            if events.index("globally_sealed") >= events.index("observation_persisted"):
                raise DiscoveryError("native observation preceded forecast seal")
            forecasts.append(report)
    if len(forecasts) != sum(r["split"] == "heldout" for r in corpus["documents"]):
        raise DiscoveryError("heldout forecast coverage differs")
    return {"schema": SCHEMA + "/verification", "files": len(manifest["files"]),
            "forecasts": forecasts, "mechanics_status": "valid",
            "terminal_status": "not-requested"}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb analyze", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="deduplicate and split label-free JSONL documents")
    prep.add_argument("input", type=Path)
    prep.add_argument("--out", required=True, type=Path)
    prep.add_argument("--seed", type=int, default=0)
    run = sub.add_parser("run", help="one admitted ordinary-execution discovery job")
    run.add_argument("--corpus", required=True, type=Path)
    run.add_argument("--model", required=True)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--stages", type=int, default=3)
    run.add_argument("--features", type=int, default=32)
    run.add_argument("--steps", type=int, default=60)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for name in ("inspect", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("root", type=Path)
    index = sub.add_parser("index", help="project verified observations for standard WATCH/XREF")
    index.add_argument("root", type=Path)
    index.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
        result = prepare_corpus(rows, seed=args.seed)
        save_json(args.out, result)
        print(json.dumps({"corpus": str(args.out), "fingerprint": result["fingerprint"],
                          "documents": len(result["documents"])}))
    elif args.action == "inspect":
        print((args.root / "report.json").read_text())
    elif args.action == "verify":
        print(json.dumps(verify_run(args.root), sort_keys=True))
    elif args.action == "index":
        from .native_discovery_export import index_observations
        print(json.dumps(index_observations(args.root, args.out), sort_keys=True))
    else:
        import os
        if not os.environ.get("MRUN_JOB_ID"):
            raise DiscoveryError("real-model discovery requires an admitted mrun worker")
        from .native_discovery_run import run_pretrained
        result = run_pretrained(**{k: v for k, v in vars(args).items() if k != "action"})
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    main()

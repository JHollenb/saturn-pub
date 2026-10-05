"""Adapters that make ``first_divergence`` a standard verb over real artifacts.

:mod:`saturn.divergence_bisect` owns the search mechanics and the hash-chained
probe ledger, but it only speaks :class:`CutHandle` lattices and digest
callbacks.  This module builds those lattices from the two artifact families
SATURN keeps bisecting by hand:

* paired JSONL streams (validation streams, replay logs): records are grouped
  along one integer axis (a decode step, a layer, a token position) and each
  group is digested through canonical JSON of caller-selected fields;
* paired checkpoint directories (training branches, forest arms): files are
  ordered by natural name sort or mtime and digested as streaming sha256 of
  their raw bytes, hashed lazily so a bisect touches O(log n) files.

Positions present in only one arm digest to a distinct missing-position
sentinel, so an arm that simply stops early reads as a divergence at the first
missing cut instead of an error.

Dependency is one-way: this module imports the core; the core never imports
this module.

Claim boundary: adapters select and hash caller-named state.  A boundary they
localize says where the selected fields stop agreeing, not why.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mdb import divergence_bisect
from mdb.divergence_bisect import (
    BisectResult,
    CutHandle,
    canonical_json_bytes,
    content_fingerprint,
    first_divergence,
)

ADAPTER_SCHEMA = "saturn-bisect-adapters-v1"

# Version marker bound into every group/file digest payload (and recorded in
# cut meta / ledger receipts).  v2 domain-separates real content from the
# missing-position sentinel and tags non-finite floats so a float ``nan``
# never digests like the string ``"NaN"``.
ADAPTER_DIGEST_DOMAIN = "saturn-bisect-adapter-digest-v2"

DigestFn = Callable[[CutHandle], str]

_CORE_COMMANDS = ("demo", "verify")
_HASH_CHUNK_BYTES = 1 << 20


def _missing_digest(axis_key: str, position: int) -> str:
    """Sentinel digest for a position absent from one arm.

    Group and file digests always hash a payload carrying the
    ``digest_domain`` key and never a ``{"missing": ...}`` payload, so the
    sentinel cannot collide with real content and always reads as divergence
    against a present position.
    """

    return content_fingerprint({"axis": [axis_key, position], "missing": True})


def _tag_non_finite(value: Any) -> Any:
    """Replace non-finite floats with a tagged dict before canonical hashing.

    :func:`canonical_json_bytes` maps ``nan`` to the string ``"NaN"`` (its
    encoding must stay stable for ledger hashing), which would make a float
    ``nan`` and the literal string ``"NaN"`` digest identically here.  The
    adapter digest path pre-transforms non-finite floats to
    ``{"__nonfinite__": ...}`` so genuinely different values diverge.
    """

    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite__": "NaN"}
        if math.isinf(value):
            return {"__nonfinite__": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, dict):
        return {key: _tag_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tag_non_finite(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# JSONL stream adapter
# ---------------------------------------------------------------------------


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: record must be a JSON object, "
                    f"got {type(record).__name__}"
                )
            records.append(record)
    return records


def _axis_position(record: dict[str, Any], axis_key: str, path: Path) -> int | None:
    """The record's integer axis coordinate, or ``None`` when it has none.

    Records without the axis key are silently skipped (streams mix kinds);
    records that carry it with a non-integer value fail closed.
    """

    if axis_key not in record:
        return None
    value = record[axis_key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{path}: axis key {axis_key!r} must be an integer, "
            f"got {value!r} ({type(value).__name__})"
        )
    return value


def _group_digests(
    records: Sequence[dict[str, Any]],
    *,
    axis_key: str,
    digest_keys: Sequence[str],
    match_keys: Sequence[str] | None,
    path: Path,
) -> tuple[dict[int, str], dict[int, int]]:
    """Per-position content digests plus per-position record counts.

    Each grouped record is projected to its ``digest_keys`` (absent keys stay
    absent, which is distinct from an explicit ``null``).  With ``match_keys``
    the match identity is bound into the hashed entry, so state swapped
    between two identities changes the digest even when the state multiset
    does not.  Entries are sorted by their canonical bytes — match identity
    first — so record order in the file never changes a digest.
    """

    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        position = _axis_position(record, axis_key, path)
        if position is None:
            continue
        groups.setdefault(position, []).append(record)

    digests: dict[int, str] = {}
    counts: dict[int, int] = {}
    for position, members in groups.items():
        entries: list[dict[str, Any]] = []
        for record in members:
            entry: dict[str, Any] = {
                "state": {key: record[key] for key in digest_keys if key in record}
            }
            if match_keys is not None:
                entry["match"] = {key: record.get(key) for key in match_keys}
            entries.append(_tag_non_finite(entry))
        entries.sort(key=canonical_json_bytes)
        digests[position] = content_fingerprint(
            {
                "axis": [axis_key, position],
                "digest_domain": ADAPTER_DIGEST_DOMAIN,
                "entries": entries,
            }
        )
        counts[position] = len(members)
    return digests, counts


def _clean_keys(keys: Sequence[str], *, name: str) -> list[str]:
    cleaned = [str(key).strip() for key in keys if str(key).strip()]
    if not cleaned:
        raise ValueError(f"{name} must name at least one field")
    return cleaned


def cuts_from_jsonl_streams(
    path_a: str | Path,
    path_b: str | Path,
    *,
    axis_key: str,
    digest_keys: Sequence[str],
    match_keys: Sequence[str] | None = None,
) -> tuple[list[CutHandle], DigestFn, DigestFn]:
    """Build a divergence lattice from two JSONL streams along one axis.

    Records are grouped by the integer value of ``axis_key`` (records without
    the key are skipped); each group digests to canonical-JSON sha256 over the
    ``digest_keys`` projection of its records, sorted by ``match_keys`` when
    given.  The lattice is the sorted union of positions from both streams;
    positions present in only one stream digest to a missing-position sentinel
    on the other side.  Raises ``ValueError`` when neither stream carries the
    axis key — a silent empty lattice would read as a false "clean".
    """

    path_a = Path(path_a)
    path_b = Path(path_b)
    axis_key = str(axis_key)
    digest_keys = _clean_keys(digest_keys, name="digest_keys")
    if match_keys is not None:
        match_keys = _clean_keys(match_keys, name="match_keys")

    digests_a, counts_a = _group_digests(
        _load_jsonl_records(path_a),
        axis_key=axis_key,
        digest_keys=digest_keys,
        match_keys=match_keys,
        path=path_a,
    )
    digests_b, counts_b = _group_digests(
        _load_jsonl_records(path_b),
        axis_key=axis_key,
        digest_keys=digest_keys,
        match_keys=match_keys,
        path=path_b,
    )

    positions = sorted(set(digests_a) | set(digests_b))
    if not positions:
        raise ValueError(
            f"no records in {path_a} or {path_b} carry integer axis key {axis_key!r}"
        )
    cuts = [
        CutHandle(
            cut_id=f"{axis_key}/{position}",
            index=position,
            meta={
                "records_a": counts_a.get(position, 0),
                "records_b": counts_b.get(position, 0),
            },
        )
        for position in positions
    ]

    def _make(digests: dict[int, str]) -> DigestFn:
        def digest(cut: CutHandle) -> str:
            return digests.get(cut.index, _missing_digest(axis_key, cut.index))

        return digest

    return cuts, _make(digests_a), _make(digests_b)


# ---------------------------------------------------------------------------
# Checkpoint directory adapter
# ---------------------------------------------------------------------------


_NATURAL_SPLIT = re.compile(r"(\d+)")


def _natural_key(name: str) -> tuple[tuple[int, int] | tuple[int, str], ...]:
    """Sort key that orders embedded integers numerically: ckpt-2 < ckpt-10."""

    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in _NATURAL_SPLIT.split(name)
    )


def _ordered_files(directory: Path, *, pattern: str, order: str) -> list[Path]:
    if order not in ("name", "mtime"):
        raise ValueError(f"order must be 'name' or 'mtime', got {order!r}")
    if not directory.is_dir():
        raise ValueError(f"checkpoint directory not found: {directory}")
    files = [path for path in directory.glob(pattern) if path.is_file()]
    if order == "name":
        files.sort(key=lambda path: (_natural_key(path.name), path.name))
    else:
        files.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return files


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cuts_from_checkpoint_dirs(
    dir_a: str | Path,
    dir_b: str | Path,
    *,
    pattern: str = "*.safetensors",
    order: str = "name",
) -> tuple[list[CutHandle], DigestFn, DigestFn]:
    """Build a divergence lattice from two checkpoint directories.

    Files matching ``pattern`` in each directory are ordered (``name`` =
    natural sort with embedded integers compared numerically, ``mtime`` =
    modification time) and paired by ordinal position; the cut index is the
    ordinal.  Digests wrap the streaming sha256 of raw file bytes in a
    domain-separated payload (``ADAPTER_DIGEST_DOMAIN``), computed lazily
    and cached, so a bisect hashes only the O(log n) files it probes.
    Positions beyond the shorter sequence digest to a missing-position
    sentinel.  Raises ``ValueError`` when neither directory has a match — a
    silent empty lattice would read as a false "clean".
    """

    files_a = _ordered_files(Path(dir_a), pattern=pattern, order=order)
    files_b = _ordered_files(Path(dir_b), pattern=pattern, order=order)
    if not files_a and not files_b:
        raise ValueError(f"no files match {pattern!r} in {dir_a} or {dir_b}")

    cuts = [
        CutHandle(
            cut_id=f"checkpoint/{position}",
            index=position,
            meta={
                "file_a": files_a[position].name if position < len(files_a) else None,
                "file_b": files_b[position].name if position < len(files_b) else None,
                "digest_domain": ADAPTER_DIGEST_DOMAIN,
            },
        )
        for position in range(max(len(files_a), len(files_b)))
    ]

    def _make(files: list[Path]) -> DigestFn:
        cache: dict[int, str] = {}

        def digest(cut: CutHandle) -> str:
            if cut.index not in cache:
                if 0 <= cut.index < len(files):
                    # Domain-separated: hash a structured payload around the
                    # raw file sha256 so a file whose bytes spell the missing
                    # sentinel's canonical JSON cannot digest to the sentinel.
                    cache[cut.index] = content_fingerprint(
                        {
                            "digest_domain": ADAPTER_DIGEST_DOMAIN,
                            "file_sha256": _file_sha256(files[cut.index]),
                        }
                    )
                else:
                    cache[cut.index] = _missing_digest("checkpoint", cut.index)
            return cache[cut.index]

        return digest

    return cuts, _make(files_a), _make(files_b)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_csv(text: str, *, name: str) -> list[str]:
    return _clean_keys(text.split(","), name=name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="saturn-bisect",
        description=(
            "Locate the first divergence between two real artifacts: paired JSONL "
            "streams along an integer axis, or paired checkpoint directories."
        ),
        epilog=(
            "The core subcommands 'demo' and 'verify' are delegated unchanged to "
            "saturn.divergence_bisect; 'verify' re-checks any --ledger receipt "
            "written here."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    streams = commands.add_parser(
        "streams", help="first divergence between two JSONL streams along an integer axis"
    )
    streams.add_argument("stream_a", type=Path, help="arm-A JSONL stream")
    streams.add_argument("stream_b", type=Path, help="arm-B JSONL stream")
    streams.add_argument(
        "--axis-key", required=True, help="integer record field that orders the lattice"
    )
    streams.add_argument(
        "--digest-keys",
        required=True,
        help="comma-separated record fields hashed into each cut digest",
    )
    streams.add_argument(
        "--match-keys",
        help="comma-separated fields that identify a record within a cut across arms",
    )

    checkpoints = commands.add_parser(
        "checkpoints", help="first divergence between two checkpoint directories"
    )
    checkpoints.add_argument("dir_a", type=Path, help="arm-A checkpoint directory")
    checkpoints.add_argument("dir_b", type=Path, help="arm-B checkpoint directory")
    checkpoints.add_argument(
        "--pattern", default="*.safetensors", help="glob for checkpoint files"
    )
    checkpoints.add_argument(
        "--order",
        choices=("name", "mtime"),
        default="name",
        help="pairing order: natural name sort or modification time",
    )

    for sub in (streams, checkpoints):
        sub.add_argument("--ledger", type=Path, help="write the full bisect receipt here")
        sub.add_argument(
            "--json", action="store_true", help="print the full receipt as one JSON object"
        )
    return parser


def _print_human(result: BisectResult, ledger_path: Path | None) -> None:
    print(f"verb: {result.verb}")
    print(f"lattice: {result.lattice_size} cuts")
    if result.first_true_index is None:
        print("boundary: none (arms agree at every probed cut)")
    else:
        print(f"boundary: index={result.first_true_index} cut_id={result.first_true_cut_id}")
    print(f"monotonicity_violation: {result.monotonicity_violation}")
    print(f"probes: {result.probe_count}")
    for probe in result.probes:
        reading = "diverged" if probe.observed_true else "agree"
        left = (probe.digest_a or "")[:16]
        right = (probe.digest_b or "")[:16]
        print(
            f"  probe index={probe.index} cut_id={probe.cut_id} {reading} "
            f"digest_a={left} digest_b={right}"
        )
    print(f"ledger: {ledger_path if ledger_path is not None else 'not written'}")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if argv and argv[0] in _CORE_COMMANDS:
        return divergence_bisect.main(argv)
    args = _parser().parse_args(argv)

    try:
        if args.command == "streams":
            digest_keys = _split_csv(args.digest_keys, name="digest_keys")
            match_keys = (
                _split_csv(args.match_keys, name="match_keys")
                if args.match_keys is not None
                else None
            )
            cuts, digest_a, digest_b = cuts_from_jsonl_streams(
                args.stream_a,
                args.stream_b,
                axis_key=args.axis_key,
                digest_keys=digest_keys,
                match_keys=match_keys,
            )
            adapter = {
                "adapter": "jsonl-streams",
                "schema": ADAPTER_SCHEMA,
                "digest_domain": ADAPTER_DIGEST_DOMAIN,
                "stream_a": str(args.stream_a),
                "stream_b": str(args.stream_b),
                "axis_key": args.axis_key,
                "digest_keys": digest_keys,
                "match_keys": match_keys,
            }
        else:
            cuts, digest_a, digest_b = cuts_from_checkpoint_dirs(
                args.dir_a, args.dir_b, pattern=args.pattern, order=args.order
            )
            adapter = {
                "adapter": "checkpoint-dirs",
                "schema": ADAPTER_SCHEMA,
                "digest_domain": ADAPTER_DIGEST_DOMAIN,
                "dir_a": str(args.dir_a),
                "dir_b": str(args.dir_b),
                "pattern": args.pattern,
                "order": args.order,
            }
        result = first_divergence(cuts, digest_a, digest_b)
        payload = result.to_json_dict()
        payload["adapter"] = adapter
        if args.ledger is not None:
            args.ledger.parent.mkdir(parents=True, exist_ok=True)
            args.ledger.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.json:
        payload["ledger_path"] = str(args.ledger) if args.ledger is not None else None
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        _print_human(result, args.ledger)
    return 0


__all__ = [
    "ADAPTER_DIGEST_DOMAIN",
    "ADAPTER_SCHEMA",
    "cuts_from_checkpoint_dirs",
    "cuts_from_jsonl_streams",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

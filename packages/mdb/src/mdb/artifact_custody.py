"""Content-addressed custody of run artifacts into a durable job directory.

Saturn receipts historically pinned bare absolute scratch paths (mostly
``/tmp/saturn-ar-*``) next to a sha256; a replay audit found 97.7% of those
(path, sha256) pairs already rotted.  This module is the repair primitive:
:func:`take_custody` copies a file into ``<job_dir>/<category>/`` under a
content-addressed name and returns a :class:`CustodyRecord` whose
``relative_path`` is stable relative to ``job_dir``, so a receipt can carry
``{relative_path, sha256, bytes, origin_path}`` instead of a doomed absolute
path.  Every custody event is appended to a hash-chained JSONL manifest at
``<job_dir>/artifacts/custody-manifest.jsonl`` and :func:`verify_custody`
re-checks files and chain later.

Mechanics that matter to callers:

* ``take_custody`` is fail-closed: a missing/symlinked source, an unsafe
  category or filename, a hash collision on the content-addressed target, or
  a source file that changes mid-copy all raise :class:`ArtifactCustodyError`.
  It is idempotent: when the target already exists with matching bytes the
  copy is skipped and the record's ``action`` is ``"reused"``.
* The manifest chain follows the shared ledger convention: each row carries
  ``prev_sha256`` and its own ``sha256`` over canonical JSON
  (``sort_keys=True``, ``separators=(",", ":")``) of the row minus its own
  ``sha256``; the first row has ``prev_sha256`` ``None``.  The ledger is
  append-only and single-chain per job directory; verification detects edits
  and reordering but not truncation of the tail (an external head pointer is
  the caller's job).
* :func:`is_ephemeral_path` is the shared heuristic wire-in sites use to tag
  receipt refs ``ephemeral: true`` when a path is intentionally scratch.
* Stdlib only.  Writers under ``saturn.ar`` may import this module freely; it
  must never grow a torch/mrun dependency.

CLI exit codes: 0 success (and verify passed), 1 verify found custody broken,
2 unusable input (missing source, unsafe names, malformed manifest on take).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ARTIFACT_CUSTODY_SCHEMA = "saturn-artifact-custody-v1"
CUSTODY_MANIFEST_CATEGORY = "artifacts"
CUSTODY_MANIFEST_NAME = "custody-manifest.jsonl"
DEFAULT_CATEGORY = "artifacts"

_CHUNK_BYTES = 1024 * 1024
_SHA256_CHARS = frozenset("0123456789abcdef")
_PORTABLE_NAME_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_EPHEMERAL_PREFIXES = (
    "/tmp/",
    "/var/tmp/",
    "/var/folders/",
    "/private/tmp/",
    "/private/var/tmp/",
    "/private/var/folders/",
    "/dev/shm/",
    "/run/",
)


class ArtifactCustodyError(ValueError):
    """Raised when custody cannot be taken or a manifest cannot be trusted."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sanitize_non_finite(value: Any) -> Any:
    """Recursively replace non-finite floats with strict-JSON-safe strings."""

    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {key: _sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(item) for item in value]
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(_sanitize_non_finite(value))).hexdigest()


def row_sha256(row: Mapping[str, Any]) -> str:
    """Hash a manifest row over canonical JSON of everything but ``sha256``."""

    body = {key: value for key, value in row.items() if key != "sha256"}
    return _sha256_json(body)


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file's bytes; raises on unreadable input."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def is_ephemeral_path(path: str | Path) -> bool:
    """True when ``path`` is absolute and under a known scratch/temp root.

    This is a heuristic for tagging receipt refs ``ephemeral: true`` so replay
    audits can excuse them; a relative path is never called ephemeral here.
    """

    text = str(path)
    if not text.startswith("/"):
        return False
    probe = text if text.endswith("/") else text + "/"
    if any(probe.startswith(prefix) for prefix in _EPHEMERAL_PREFIXES):
        return True
    tempdir = tempfile.gettempdir().rstrip("/") + "/"
    return tempdir.startswith("/") and probe.startswith(tempdir)


def _require_safe_component(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in (".", "..")
        or any(character not in _PORTABLE_NAME_CHARS for character in value)
    ):
        raise ArtifactCustodyError(
            f"{label} must be one non-empty portable path component "
            f"([A-Za-z0-9._-], not '.'/'..'), got {value!r}"
        )
    return value


@dataclass(frozen=True)
class CustodyRecord:
    """One durable, content-addressed custody of a single file.

    ``relative_path`` is POSIX-style and relative to the job directory the
    custody was taken into; ``origin_path`` preserves where the bytes came
    from (frequently ephemeral scratch) for provenance only.
    """

    relative_path: str
    sha256: str
    bytes: int
    origin_path: str
    category: str = DEFAULT_CATEGORY
    action: str = "copied"

    def __post_init__(self) -> None:
        if not isinstance(self.relative_path, str) or not self.relative_path:
            raise ArtifactCustodyError("custody relative_path cannot be empty")
        posix = PurePosixPath(self.relative_path)
        if posix.is_absolute() or ".." in posix.parts:
            raise ArtifactCustodyError(
                "custody relative_path must be relative and stay inside the job directory"
            )
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in _SHA256_CHARS for character in self.sha256)
        ):
            raise ArtifactCustodyError("custody sha256 must be one lowercase SHA-256")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ArtifactCustodyError("custody bytes must be a non-negative integer")
        if not isinstance(self.origin_path, str) or not self.origin_path:
            raise ArtifactCustodyError("custody origin_path cannot be empty")
        _require_safe_component(self.category, label="custody category")
        if self.action not in ("copied", "cloned", "reused"):
            raise ArtifactCustodyError("custody action must be 'copied' or 'reused'")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "origin_path": self.origin_path,
            "category": self.category,
            "action": self.action,
        }


def manifest_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / CUSTODY_MANIFEST_CATEGORY / CUSTODY_MANIFEST_NAME


class CustodyManifest:
    """Hash-chained append-only JSONL ledger of custody events for one job."""

    def __init__(self, job_dir: str | Path) -> None:
        self.job_dir = Path(job_dir)
        self.path = manifest_path(self.job_dir)

    @property
    def lock_path(self) -> Path:
        """Sidecar lock file serializing read-tail+verify+append."""

        return self.path.with_name(self.path.name + ".lock")

    def rows(self) -> list[dict[str, Any]]:
        """Parse every manifest row; fail closed on any malformed line."""

        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        text = self.path.read_bytes().decode("utf-8", errors="strict")
        for index, raw in enumerate(text.split("\n")):
            line = raw[:-1] if raw.endswith("\r") else raw
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactCustodyError(
                    f"custody manifest line {index + 1} is not JSON: {self.path}"
                ) from exc
            if not isinstance(row, dict):
                raise ArtifactCustodyError(
                    f"custody manifest line {index + 1} is not an object: {self.path}"
                )
            rows.append(row)
        return rows

    def verify_chain(self, rows: Sequence[Mapping[str, Any]] | None = None) -> str | None:
        """Return None when the chain verifies, else a human-readable reason."""

        if rows is None:
            try:
                rows = self.rows()
            except (ArtifactCustodyError, OSError, UnicodeDecodeError) as exc:
                return str(exc)
        previous: str | None = None
        for index, row in enumerate(rows):
            declared = row.get("sha256")
            if not isinstance(declared, str) or row_sha256(row) != declared:
                return f"row {index} sha256 does not match its content"
            if row.get("prev_sha256") != previous:
                return f"row {index} prev_sha256 breaks the chain"
            if row.get("schema") != ARTIFACT_CUSTODY_SCHEMA:
                return f"row {index} has an unexpected schema"
            previous = declared
        return None

    def head_sha256(self) -> str | None:
        rows = self.rows()
        if not rows:
            return None
        head = rows[-1].get("sha256")
        if not isinstance(head, str):
            raise ArtifactCustodyError("custody manifest head row has no sha256")
        return head

    def append(self, record: CustodyRecord, *, timestamp: float | None = None) -> dict[str, Any]:
        """Append one hash-chained row for ``record``; returns the row written.

        The row's ``sha256``/``prev_sha256`` pair is the ledger chain (same
        convention as the other Saturn ledgers); the custodied file's digest
        is stored separately as ``file_sha256``.

        The read-verify-append critical section runs under an exclusive
        ``flock`` on ``<manifest>.lock`` (same pattern as the claims
        registry), so concurrent writers serialize instead of forking the
        chain.
        """

        if not isinstance(record, CustodyRecord):
            raise ArtifactCustodyError("custody manifest accepts CustodyRecord rows only")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                rows = self.rows()
                error = self.verify_chain(rows)
                if error is not None:
                    raise ArtifactCustodyError(
                        f"custody manifest chain broken before append: {error}"
                    )
                payload = record.to_json_dict()
                payload["file_sha256"] = payload.pop("sha256")
                row = {
                    "schema": ARTIFACT_CUSTODY_SCHEMA,
                    "timestamp": time.time() if timestamp is None else float(timestamp),
                    **payload,
                    "prev_sha256": rows[-1]["sha256"] if rows else None,
                }
                row["sha256"] = row_sha256(row)
                line = json.dumps(_sanitize_non_finite(row), sort_keys=True, allow_nan=False)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return row


def custody_manifest(job_dir: str | Path) -> CustodyManifest:
    """The custody ledger for ``job_dir`` (``artifacts/custody-manifest.jsonl``)."""

    return CustodyManifest(job_dir)


def _clone_file(source: Path, destination: Path) -> bool:
    """Copy-on-write clone (APFS clonefile / Linux FICLONE); False when unsupported.

    A clone is a separate inode — later writes to either side never touch the other —
    but shares data blocks, so custody of a large artifact costs no extra disk.
    """

    import platform

    try:
        if platform.system() == "Darwin":
            import ctypes

            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            return libc.clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0
        if platform.system() == "Linux":
            import fcntl as _fcntl

            ficlone = 0x40049409
            with source.open("rb") as reader, destination.open("xb") as writer:
                try:
                    _fcntl.ioctl(writer.fileno(), ficlone, reader.fileno())
                    return True
                except OSError:
                    pass
            destination.unlink(missing_ok=True)
    except (OSError, AttributeError):
        destination.unlink(missing_ok=True)
    return False


def take_custody(
    src_path: str | Path,
    job_dir: str | Path,
    *,
    category: str = DEFAULT_CATEGORY,
) -> CustodyRecord:
    """Copy ``src_path`` into ``<job_dir>/<category>/<sha256[:16]>-<basename>``.

    Content-addressed and idempotent: an existing target with matching bytes
    is reused without copying.  Fail-closed on a missing or symlinked source,
    unsafe names, a differing file already at the target address, or a source
    that changes while being copied.  Every call appends one manifest row.
    """

    source = Path(src_path)
    if source.is_symlink():
        raise ArtifactCustodyError(f"custody refuses symlinked source: {source}")
    if not source.is_file():
        raise ArtifactCustodyError(f"custody source is not a readable file: {source}")
    _require_safe_component(category, label="custody category")
    basename = _require_safe_component(source.name, label="custody source filename")

    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    sha256 = digest.hexdigest()

    job_root = Path(job_dir)
    target_dir = job_root / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target_name = f"{sha256[:16]}-{basename}"
    target = target_dir / target_name
    relative_path = str(PurePosixPath(category) / target_name)

    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise ArtifactCustodyError(f"custody target is not a regular file: {target}")
        if target.stat().st_size != size or sha256_file(target) != sha256:
            raise ArtifactCustodyError(
                f"custody target exists with different content: {target}"
            )
        action = "reused"
    else:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target_name}.", dir=target_dir)
        temporary = Path(temporary_name)
        try:
            os.close(descriptor)
            temporary.unlink()
            if _clone_file(source, temporary):
                # copy-on-write clone: separate inode, shared blocks, no extra disk
                action = "cloned"
                if temporary.stat().st_size != size or sha256_file(temporary) != sha256:
                    raise ArtifactCustodyError(
                        f"custody source changed while being cloned: {source}"
                    )
            else:
                action = "copied"
                copied = hashlib.sha256()
                copied_size = 0
                with temporary.open("xb") as writer, source.open("rb") as reader:
                    while chunk := reader.read(_CHUNK_BYTES):
                        copied.update(chunk)
                        copied_size += len(chunk)
                        writer.write(chunk)
                    writer.flush()
                    os.fsync(writer.fileno())
                if copied.hexdigest() != sha256 or copied_size != size:
                    raise ArtifactCustodyError(
                        f"custody source changed while being copied: {source}"
                    )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    record = CustodyRecord(
        relative_path=relative_path,
        sha256=sha256,
        bytes=size,
        origin_path=str(source),
        category=category,
        action=action,
    )
    custody_manifest(job_root).append(record)
    return record


def _escapes_job_root(
    job_root: Path, resolved_root: Path, posix: PurePosixPath
) -> bool:
    """True when ``posix`` under ``job_root`` crosses a symlinked parent or
    resolves outside the resolved job root.

    Custody verification must only ever trust bytes physically contained in
    the job directory; a ``<job>/artifacts`` swapped for a symlink pointing
    elsewhere would otherwise still verify.
    """

    parent = job_root
    for part in posix.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            return True
    try:
        resolved = (job_root / posix).resolve(strict=False)
    except OSError:
        return True
    return not resolved.is_relative_to(resolved_root)


def verify_custody(job_dir: str | Path) -> dict[str, Any]:
    """Re-check the manifest chain and every custodied file under ``job_dir``.

    Fail-closed report semantics: an absent or unreadable manifest makes the
    report not-ok (there is nothing trustworthy to verify against).  For each
    ``relative_path`` only the latest manifest row is authoritative.
    """

    job_root = Path(job_dir)
    ledger = custody_manifest(job_root)
    report: dict[str, Any] = {
        "schema": ARTIFACT_CUSTODY_SCHEMA,
        "job_dir": str(job_root),
        "manifest_path": str(ledger.path),
        "manifest_present": ledger.path.is_file(),
        "chain_ok": False,
        "chain_error": None,
        "row_count": 0,
        "files": [],
        "files_expected": 0,
        "files_present": 0,
        "files_ok": 0,
        "ok": False,
    }
    if not report["manifest_present"]:
        report["chain_error"] = "custody manifest is absent"
        return report
    try:
        rows = ledger.rows()
    except (ArtifactCustodyError, OSError, UnicodeDecodeError) as exc:
        report["chain_error"] = str(exc)
        return report
    report["row_count"] = len(rows)
    chain_error = ledger.verify_chain(rows)
    report["chain_ok"] = chain_error is None
    report["chain_error"] = chain_error

    try:
        resolved_root = job_root.resolve(strict=True)
    except OSError:
        resolved_root = None
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        relative = row.get("relative_path")
        if isinstance(relative, str) and relative:
            latest[relative] = row
    files = []
    for relative in sorted(latest):
        row = latest[relative]
        entry: dict[str, Any] = {
            "relative_path": relative,
            "expected_sha256": row.get("file_sha256"),
            "present": False,
            "bytes_ok": False,
            "sha256_ok": False,
            "ok": False,
        }
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts:
            # A manifest row must never address outside the job directory.
            files.append(entry)
            continue
        if resolved_root is None or _escapes_job_root(job_root, resolved_root, posix):
            # A symlinked parent (or a path resolving outside the resolved
            # job root) can silently swap custodied bytes; fail closed.
            files.append(entry)
            continue
        path = job_root / posix
        expected_bytes = row.get("bytes")
        if path.is_file() and not path.is_symlink():
            entry["present"] = True
            entry["bytes_ok"] = (
                isinstance(expected_bytes, int)
                and not isinstance(expected_bytes, bool)
                and path.stat().st_size == expected_bytes
            )
            try:
                entry["sha256_ok"] = (
                    isinstance(entry["expected_sha256"], str)
                    and sha256_file(path) == entry["expected_sha256"]
                )
            except OSError:
                entry["sha256_ok"] = False
        entry["ok"] = entry["present"] and entry["bytes_ok"] and entry["sha256_ok"]
        files.append(entry)
    report["files"] = files
    report["files_expected"] = len(files)
    report["files_present"] = sum(1 for entry in files if entry["present"])
    report["files_ok"] = sum(1 for entry in files if entry["ok"])
    report["ok"] = bool(report["chain_ok"]) and report["files_ok"] == report["files_expected"]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Take or verify content-addressed custody of job artifacts"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    take = commands.add_parser("take", help="copy a file into durable job custody")
    take.add_argument("src", type=Path, help="source file to take custody of")
    take.add_argument("--job-dir", required=True, type=Path, help="durable job directory")
    take.add_argument("--category", default=DEFAULT_CATEGORY, help="subdirectory under job dir")
    take.add_argument("--json", action="store_true", dest="as_json")
    verify = commands.add_parser("verify", help="verify manifest chain and custodied files")
    verify.add_argument("--job-dir", required=True, type=Path, help="durable job directory")
    verify.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "take":
        try:
            record = take_custody(args.src, args.job_dir, category=args.category)
        except (ArtifactCustodyError, OSError) as exc:
            print(f"custody refused: {exc}", file=sys.stderr)
            return 2
        if args.as_json:
            print(json.dumps({"schema": ARTIFACT_CUSTODY_SCHEMA, **record.to_json_dict()},
                             sort_keys=True))
        else:
            print(
                f"CUSTODY {record.action} {record.relative_path} "
                f"sha256={record.sha256[:12]} bytes={record.bytes} "
                f"origin={record.origin_path}"
            )
        return 0
    report = verify_custody(args.job_dir)
    if args.as_json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(
            f"custody {'OK' if report['ok'] else 'BROKEN'} "
            f"rows={report['row_count']} files={report['files_ok']}/"
            f"{report['files_expected']} chain_ok={report['chain_ok']}"
            + (f" error={report['chain_error']}" if report["chain_error"] else "")
        )
    return 0 if report["ok"] else 1


__all__ = [
    "ARTIFACT_CUSTODY_SCHEMA",
    "CUSTODY_MANIFEST_NAME",
    "ArtifactCustodyError",
    "CustodyManifest",
    "CustodyRecord",
    "custody_manifest",
    "is_ephemeral_path",
    "main",
    "manifest_path",
    "row_sha256",
    "sha256_file",
    "take_custody",
    "verify_custody",
]


if __name__ == "__main__":
    raise SystemExit(main())

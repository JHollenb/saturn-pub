"""Cross-reference index over receipts, in the spirit of Ghidra XREFs.

Receipts mention ``ar://`` state addresses in free text, config blobs, and
structured payloads. This module sweeps a tree of JSON receipts, extracts every
concrete address occurrence together with its surrounding snippet, and stores
the references in a small SQLite database so that "who mentions this address?"
becomes a single indexed query instead of a repository grep.

Every reference row records the file's ``path_class`` (report / run-receipt /
observation / other, classified from the basename), plus best-effort model
attribution (``model_id``, ``revision``) discovered during a structural JSON
walk -- the nearest enclosing object carrying a model descriptor (a ``model``
dict/string, ``model_id``, or ``registry_name`` key) claims the references
beneath it, and ``NULL`` means "no descriptor found", never a guess. Citation
dicts produced by :func:`saturn_pub.evidence.citations.make_citation` (marker
key ``saturn_pub_citation``) are recognized during the walk and their
``model_id`` / ``revision`` / ``input_sha256`` fields take precedence over
structural attribution. Files larger than ``MAX_FILE_BYTES`` (256 MiB) are
recorded in the ``skipped`` table instead of being silently dropped. Opening a
database with any other ``user_version`` fails closed on reads and triggers a
drop-and-rebuild on writes.

Staleness contract: unchanged-file detection is an exact match on the stored
``(mtime, size)`` pair. A rewrite that keeps the same size and restores the
original mtime is therefore skipped as "unchanged" even though its bytes differ;
the ``files.sha256`` column records content identity so an external audit can
catch exactly that case.

Honest claim boundary: this is mechanics-only infrastructure. It indexes where
addresses appear and which model descriptor enclosed them; it makes no
scientific claim about what any address, attribution, or receipt means.

CLI exit codes: 0 = success, 2 = missing database or schema-version mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .citations import ADDRESS_SCHEME, citation_fields

XREF_SCHEMA = "saturn-pub-xref-index-v1"
XREF_USER_VERSION = 1
MAX_FILE_BYTES = 256 * 1024 * 1024
SNIPPET_CHARS = 160
_CONTEXT_CHARS = 60
_CLAIM_BOUNDARY_CHARS = 200

_ADDRESS_RE = re.compile(rf"{ADDRESS_SCHEME}://[A-Za-z0-9_./{{}}+-]+")
_SCHEME_PREFIX = f"{ADDRESS_SCHEME}://"
_JOB_DIR_RE = re.compile(r"^job-[0-9a-f]+$")
_TRAILING_PUNCTUATION = ".,;:}/"

_PATH_CLASSES = ("report", "run-receipt", "observation", "other")
_MODEL_IDENTITY_KEYS = ("model_id", "registry_name", "name")


class SchemaVersionError(RuntimeError):
    """An existing index was built with a different ``user_version``."""


class EmptySweepError(FileNotFoundError):
    """A full build saw zero files on disk while the index holds rows.

    Purging in that state would silently wipe the whole index (a typo'd or
    unmounted root reads exactly like "every receipt was deleted"), so the build
    fails closed instead. Subclasses ``FileNotFoundError`` so CLI handling maps
    it to exit code 2.
    """


@dataclass
class BuildReport:
    """Counters for one incremental sweep of a receipt tree."""

    files_scanned: int = 0
    files_skipped_unchanged: int = 0
    files_unparseable: int = 0
    files_too_large: int = 0
    refs_added: int = 0
    refs_with_model_id: int = 0
    refs_from_citations: int = 0
    distinct_addresses: int = 0


@dataclass
class Reference:
    """One concrete address occurrence extracted from a receipt."""

    address: str
    snippet: str
    model_id: str | None = None
    revision: str | None = None
    input_sha256: str | None = None
    from_citation: bool = False


def classify_path(basename: str) -> str:
    """Classify a receipt basename into one of ``_PATH_CLASSES``.

    Exact names only for the report / run-receipt mirror pair, so ``--dedup``
    never collapses a variant receipt against an unrelated ``report.json``.
    """

    name = basename.lower()
    if name == "report.json":
        return "report"
    if name == "run-receipt.json":
        return "run-receipt"
    if "observation" in name:
        return "observation"
    return "other"


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files(
    path TEXT PRIMARY KEY,
    mtime REAL,
    size INTEGER,
    sha256 TEXT,
    experiment TEXT,
    job_id TEXT
);
CREATE TABLE IF NOT EXISTS refs(
    address TEXT,
    path TEXT,
    experiment TEXT,
    job_id TEXT,
    snippet TEXT,
    path_class TEXT,
    model_id TEXT,
    revision TEXT,
    input_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS receipts(
    path TEXT PRIMARY KEY,
    experiment TEXT,
    config_sha256 TEXT,
    content_sha256 TEXT,
    claim_boundary TEXT,
    gates_json TEXT
);
CREATE TABLE IF NOT EXISTS skipped(
    path TEXT PRIMARY KEY,
    reason TEXT,
    size INTEGER
);
CREATE INDEX IF NOT EXISTS idx_refs_address ON refs(address);
CREATE INDEX IF NOT EXISTS idx_refs_experiment ON refs(experiment);
"""

_TABLES = ("files", "refs", "receipts", "skipped")


def _connect(db: Path) -> sqlite3.Connection:
    """Open (creating or migrating) a writable index at the current schema.

    Any other ``user_version`` -- including 0, which fresh databases and any
    older schema report -- drops the known tables and recreates them, so the
    next full build repopulates from scratch instead of mixing schemas.
    """

    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != XREF_USER_VERSION:
        connection.executescript("".join(f"DROP TABLE IF EXISTS {table};" for table in _TABLES))
    connection.executescript(_SCHEMA_SQL)
    connection.execute(f"PRAGMA user_version = {XREF_USER_VERSION}")
    return connection


def _path_coordinates(relative: Path) -> tuple[str | None, str | None]:
    """Derive (experiment_dir, job_id) from a path relative to the tree root."""

    parts = relative.parts
    experiment = parts[0] if len(parts) > 1 else None
    job_id = next((part for part in parts if _JOB_DIR_RE.match(part)), None)
    return experiment, job_id


def _snippet(text: str, start: int, end: int) -> str:
    window = text[max(0, start - _CONTEXT_CHARS) : end + _CONTEXT_CHARS]
    return " ".join(window.split())[:SNIPPET_CHARS]


def _iter_concrete(text: str):
    """Yield (address, start, end) for each concrete ``ar://`` match in text."""

    for match in _ADDRESS_RE.finditer(text):
        address = match.group(0).rstrip(_TRAILING_PUNCTUATION)
        if not address.startswith(_SCHEME_PREFIX) or len(address) <= len(_SCHEME_PREFIX):
            continue
        if "{" in address:
            continue
        yield address, match.start(), match.end()


def extract_addresses(text: str) -> list[tuple[str, str]]:
    """Return (address, snippet) pairs for every concrete ``ar://`` occurrence.

    Trailing punctuation is stripped from each match. Template strings that
    still contain ``{`` after stripping are not concrete addresses and are
    excluded, as are degenerate matches that strip down to a bare scheme.
    """

    return [(address, _snippet(text, start, end)) for address, start, end in _iter_concrete(text)]


def _text_references(text: str) -> list[Reference]:
    return [Reference(address, snippet) for address, snippet in extract_addresses(text)]


def _model_descriptor(node: dict[str, Any]) -> tuple[str, str | None] | None:
    """Best-effort model identity carried by ``node``, or ``None`` -- never a guess.

    Recognized shapes, in preference order: ``model`` mapping to a dict with
    ``model_id``/``registry_name``/``name`` (optional sibling ``revision``
    inside that dict), ``model`` mapping to a non-empty string, then a direct
    ``model_id`` or ``registry_name`` key with an optional sibling ``revision``.
    A bare ``name`` key outside a ``model`` dict is too generic and never counts.
    """

    def _string(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    model = node.get("model")
    if isinstance(model, dict):
        for key in _MODEL_IDENTITY_KEYS:
            identity = _string(model.get(key))
            if identity is not None:
                return identity, _string(model.get("revision"))
    identity = _string(model)
    if identity is not None:
        return identity, _string(node.get("revision"))
    for key in ("model_id", "registry_name"):
        identity = _string(node.get(key))
        if identity is not None:
            return identity, _string(node.get("revision"))
    return None


def _scan_string(
    text: str,
    pointer: str,
    model_id: str | None,
    revision: str | None,
    out: list[Reference],
) -> None:
    for address, start, end in _iter_concrete(text):
        window = " ".join(text[max(0, start - _CONTEXT_CHARS) : end + _CONTEXT_CHARS].split())
        snippet = f"{pointer or '$'}: {window}"[:SNIPPET_CHARS]
        out.append(Reference(address, snippet, model_id, revision))


def _citation_snippet(pointer: str, fields: dict[str, Any]) -> str:
    parts = [f"{pointer or '$'}: citation", fields["address"]]
    if fields["quantity"]:
        parts.append(f"quantity={fields['quantity']}")
    if fields["note"]:
        parts.append(f"note={fields['note']}")
    return " ".join(" ".join(parts).split())[:SNIPPET_CHARS]


def _visit(
    node: Any,
    pointer: str,
    model_id: str | None,
    revision: str | None,
    out: list[Reference],
) -> None:
    if isinstance(node, dict):
        fields = citation_fields(node)
        if fields is not None:
            if fields["model_id"] is not None or fields["revision"] is not None:
                # The citation's attribution pair wins as a unit: a citation
                # that names only a revision must not inherit a structural
                # model_id's revision (or vice versa).
                cited_model, cited_revision = fields["model_id"], fields["revision"]
            else:
                cited_model, cited_revision = model_id, revision
            out.append(
                Reference(
                    address=fields["address"],
                    snippet=_citation_snippet(pointer, fields),
                    model_id=cited_model,
                    revision=cited_revision,
                    input_sha256=fields["input_sha256"],
                    from_citation=True,
                )
            )
            for key, value in node.items():
                if key == "address":
                    continue
                _visit(value, f"{pointer}/{key}", model_id, revision, out)
            return
        descriptor = _model_descriptor(node)
        if descriptor is not None:
            model_id, revision = descriptor
        for key, value in node.items():
            if _SCHEME_PREFIX in key:
                _scan_string(key, pointer, model_id, revision, out)
            _visit(value, f"{pointer}/{key}", model_id, revision, out)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _visit(item, f"{pointer}/{index}", model_id, revision, out)
    elif isinstance(node, str) and _SCHEME_PREFIX in node:
        _scan_string(node, pointer, model_id, revision, out)


def _walk_references(payload: Any) -> list[Reference]:
    references: list[Reference] = []
    _visit(payload, "", None, None, references)
    return references


def _references_for(payload: Any, text: str) -> list[Reference]:
    """References for a parsed payload; falls back to the flat text scan."""

    if _SCHEME_PREFIX not in text:
        return []
    try:
        return _walk_references(payload)
    except RecursionError:
        return _text_references(text)


def extract_references(text: str) -> tuple[list[Reference], bool]:
    """Extract references with attribution; second element flags unparseable JSON.

    Parseable JSON is walked structurally (model descriptors and citation dicts
    populate attribution); malformed JSON falls back to the flat regex scan with
    ``NULL`` attribution, never a hard failure.
    """

    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return _text_references(text), True
    return _references_for(payload, text), False


def _receipt_identity(payload: Any) -> dict[str, Any] | None:
    """Extract top-level receipt identity fields from a parsed payload."""

    if not isinstance(payload, dict):
        return None
    keys = ("experiment", "config_sha256", "content_sha256", "claim_boundary", "gates")
    if not any(key in payload for key in keys):
        return None
    gates = payload.get("gates")
    gates_json = (
        json.dumps(gates, sort_keys=True, separators=(",", ":"), default=str)
        if isinstance(gates, dict)
        else None
    )
    claim_boundary = payload.get("claim_boundary")
    return {
        "experiment": str(payload["experiment"]) if "experiment" in payload else None,
        "config_sha256": str(payload["config_sha256"]) if "config_sha256" in payload else None,
        "content_sha256": str(payload["content_sha256"]) if "content_sha256" in payload else None,
        "claim_boundary": (
            str(claim_boundary)[:_CLAIM_BOUNDARY_CHARS] if claim_boundary is not None else None
        ),
        "gates_json": gates_json,
    }


def build_index(
    root: Path | str,
    db: Path | str,
    limit: int | None = None,
) -> BuildReport:
    """Incrementally index every ``*.json`` receipt under ``root`` into ``db``.

    Files whose stored (mtime, size) match are skipped without re-reading. Files
    above ``MAX_FILE_BYTES`` are recorded in the ``skipped`` table. Malformed
    JSON is still regex-scanned for addresses and counted as unparseable, never
    fatal. ``limit`` caps how many files are read this run (for smokes).

    A full build (``limit=None``) also purges files/refs/receipts/skipped rows
    whose path was not seen on disk during this sweep. Limited builds purge
    nothing. Fail-closed guards: a ``root`` that is not a directory raises
    ``FileNotFoundError``; a full build that finds zero files while the index
    already holds rows raises :class:`EmptySweepError` instead of purging.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"xref build root is not a directory: {root}")
    report = BuildReport()
    seen: set[str] = set()
    connection = _connect(Path(db))
    try:
        for path in sorted(root.rglob("*.json")):
            if not path.is_file():
                continue
            if limit is not None and report.files_scanned >= limit:
                break
            relative = path.relative_to(root)
            seen.add(relative.as_posix())
            stat = path.stat()
            if stat.st_size > MAX_FILE_BYTES:
                report.files_too_large += 1
                connection.execute(
                    "INSERT OR REPLACE INTO skipped(path, reason, size) VALUES (?, ?, ?)",
                    (
                        relative.as_posix(),
                        f"file exceeds MAX_FILE_BYTES ({MAX_FILE_BYTES} bytes)",
                        stat.st_size,
                    ),
                )
                for table in ("files", "refs", "receipts"):
                    connection.execute(f"DELETE FROM {table} WHERE path = ?", (relative.as_posix(),))
                continue
            stored = connection.execute(
                "SELECT mtime, size FROM files WHERE path = ?", (relative.as_posix(),)
            ).fetchone()
            if stored is not None and stored[0] == stat.st_mtime and stored[1] == stat.st_size:
                report.files_skipped_unchanged += 1
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            experiment, job_id = _path_coordinates(relative)
            path_class = classify_path(relative.name)
            try:
                payload: Any = json.loads(text)
                parsed = True
            except (json.JSONDecodeError, RecursionError):
                parsed = False
            if parsed:
                identity = _receipt_identity(payload)
                references = _references_for(payload, text)
            else:
                report.files_unparseable += 1
                identity = None
                references = _text_references(text)
            connection.execute("DELETE FROM refs WHERE path = ?", (relative.as_posix(),))
            connection.execute("DELETE FROM receipts WHERE path = ?", (relative.as_posix(),))
            connection.execute("DELETE FROM skipped WHERE path = ?", (relative.as_posix(),))
            connection.execute(
                "INSERT OR REPLACE INTO files(path, mtime, size, sha256, experiment, job_id)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    relative.as_posix(),
                    stat.st_mtime,
                    stat.st_size,
                    hashlib.sha256(raw).hexdigest(),
                    experiment,
                    job_id,
                ),
            )
            connection.executemany(
                "INSERT INTO refs(address, path, experiment, job_id, snippet,"
                " path_class, model_id, revision, input_sha256)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        reference.address,
                        relative.as_posix(),
                        experiment,
                        job_id,
                        reference.snippet,
                        path_class,
                        reference.model_id,
                        reference.revision,
                        reference.input_sha256,
                    )
                    for reference in references
                ],
            )
            if identity is not None:
                connection.execute(
                    "INSERT INTO receipts"
                    "(path, experiment, config_sha256, content_sha256,"
                    " claim_boundary, gates_json)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        relative.as_posix(),
                        identity["experiment"],
                        identity["config_sha256"],
                        identity["content_sha256"],
                        identity["claim_boundary"],
                        identity["gates_json"],
                    ),
                )
            report.refs_added += len(references)
            report.refs_with_model_id += sum(
                1 for reference in references if reference.model_id is not None
            )
            report.refs_from_citations += sum(
                1 for reference in references if reference.from_citation
            )
            report.files_scanned += 1
        if limit is None:
            if not seen:
                stored_rows = sum(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in _TABLES
                )
                if stored_rows:
                    raise EmptySweepError(
                        f"refusing to purge: no *.json files found under {root} but the"
                        f" index at {db} holds {stored_rows} row(s); delete the index"
                        " explicitly if the corpus is really gone"
                    )
            for table in _TABLES:
                stored_paths = [
                    row[0] for row in connection.execute(f"SELECT DISTINCT path FROM {table}")
                ]
                connection.executemany(
                    f"DELETE FROM {table} WHERE path = ?",
                    [(stored,) for stored in stored_paths if stored not in seen],
                )
        connection.commit()
        report.distinct_addresses = connection.execute(
            "SELECT COUNT(DISTINCT address) FROM refs"
        ).fetchone()[0]
    finally:
        connection.close()
    return report


def _connect_readonly(db: Path | str) -> sqlite3.Connection:
    """Open an existing index read-only; never create a 0-byte database.

    Raises ``SchemaVersionError`` when the file exists but was built with a
    different ``user_version`` (fail-closed; run ``build`` to migrate).
    """

    db = Path(db)
    if not db.is_file():
        raise FileNotFoundError(f"xref index does not exist: {db} (run 'build' first)")
    connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != XREF_USER_VERSION:
        connection.close()
        raise SchemaVersionError(
            f"xref index {db} has schema user_version {version},"
            f" expected {XREF_USER_VERSION} (run 'build' to migrate)"
        )
    return connection


def _job_dir(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def dedup_refs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse report / run-receipt mirror pairs, preferring the report row.

    Within each ``(address, containing directory)`` group, ``run-receipt`` rows
    are dropped when a ``report`` row exists; every other row survives (an
    address cited only by a run-receipt is never lost).
    """

    reported = {
        (row["address"], _job_dir(row["path"]))
        for row in rows
        if row["path_class"] == "report"
    }
    return [
        row
        for row in rows
        if not (
            row["path_class"] == "run-receipt"
            and (row["address"], _job_dir(row["path"])) in reported
        )
    ]


def query_refs(db: Path | str, prefix: str, dedup: bool = False) -> list[dict[str, Any]]:
    """Return reference rows whose address starts with ``prefix``, as dicts.

    ``dedup=True`` collapses report / run-receipt mirror pairs. Raises
    ``FileNotFoundError`` when the database file does not exist and
    ``SchemaVersionError`` on a version mismatch.
    """

    connection = _connect_readonly(db)
    try:
        cursor = connection.execute(
            "SELECT address, path, experiment, job_id, snippet,"
            " path_class, model_id, revision, input_sha256 FROM refs"
            " WHERE substr(address, 1, ?) = ?"
            " ORDER BY address, experiment, path",
            (len(prefix), prefix),
        )
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()
    return dedup_refs(rows) if dedup else rows


def index_stats(db: Path | str, dedup: bool = False) -> dict[str, Any]:
    """Summarize an existing index: file, ref, skip, and address counts.

    ``dedup=True`` computes ref/address/top-address figures over the
    mirror-collapsed rows. Raises ``FileNotFoundError`` when the database file
    does not exist and ``SchemaVersionError`` on a version mismatch.
    """

    connection = _connect_readonly(db)
    try:
        files = connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        receipts = connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        skipped = connection.execute("SELECT COUNT(*) FROM skipped").fetchone()[0]
        cursor = connection.execute("SELECT address, path, path_class, model_id FROM refs")
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    finally:
        connection.close()
    deduped = dedup_refs(rows)
    pool = deduped if dedup else rows
    counts = Counter(row["address"] for row in pool)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "schema": XREF_SCHEMA,
        "dedup": dedup,
        "files_indexed": files,
        "refs": len(pool),
        "refs_total": len(rows),
        "refs_deduped": len(deduped),
        "refs_with_model_id": sum(1 for row in pool if row["model_id"] is not None),
        "refs_with_model_id_total": sum(1 for row in rows if row["model_id"] is not None),
        "receipts": receipts,
        "files_skipped_too_large": skipped,
        "distinct_addresses": len(counts),
        "top_addresses": [{"address": address, "refs": count} for address, count in top],
    }


def _cmd_build(args: argparse.Namespace) -> int:
    report = build_index(args.root, args.db, limit=args.limit)
    if report.files_too_large:
        print(
            f"warning: skipped {report.files_too_large} file(s) over"
            f" {MAX_FILE_BYTES} bytes (recorded in the 'skipped' table)",
            file=sys.stderr,
        )
    print(json.dumps({"schema": XREF_SCHEMA, "db": str(args.db), **asdict(report)}, sort_keys=True))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    rows = query_refs(args.db, args.address, dedup=args.dedup)
    if args.json:
        print(json.dumps(rows, sort_keys=True))
        return 0
    if not rows:
        print(f"no references match prefix {args.address!r}")
        return 0
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(row["experiment"] or "(no experiment)", []).append(row)
    for experiment in sorted(by_experiment):
        print(f"experiment {experiment} ({len(by_experiment[experiment])} refs)")
        for row in by_experiment[experiment]:
            job = f" [{row['job_id']}]" if row["job_id"] else ""
            print(f"  {row['address']}{job} <- {row['path']}")
            print(f"    {row['snippet']}")
    by_address: dict[str, Counter[str | None]] = {}
    for row in rows:
        by_address.setdefault(row["address"], Counter())[row["model_id"]] += 1
    print("model_id breakdown:")
    for address in sorted(by_address):
        print(f"  {address}")
        models = sorted(
            by_address[address].items(), key=lambda item: (item[0] is None, item[0] or "")
        )
        for model, count in models:
            print(f"    {model or '(unattributed)'}: {count}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    stats = index_stats(args.db, dedup=args.dedup)
    if args.json:
        print(json.dumps(stats, sort_keys=True))
        return 0
    print(f"files indexed:      {stats['files_indexed']}")
    print(f"refs:               {stats['refs']}")
    print(f"refs total:         {stats['refs_total']}")
    print(f"refs deduped:       {stats['refs_deduped']}")
    print(f"refs with model_id: {stats['refs_with_model_id']}")
    print(f"refs with model_id (total): {stats['refs_with_model_id_total']}")
    print(f"receipts:           {stats['receipts']}")
    print(f"skipped too large:  {stats['files_skipped_too_large']}")
    print(f"distinct addresses: {stats['distinct_addresses']}")
    print("top addresses:")
    for row in stats["top_addresses"]:
        print(f"  {row['refs']:>6}  {row['address']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-reference index over receipts (Ghidra-style XREFs)"
    )
    commands = parser.add_subparsers(dest="xref_command", required=True)

    build = commands.add_parser("build", help="sweep a receipt tree into the index")
    build.add_argument("--root", type=Path, required=True, help="tree of *.json receipts")
    build.add_argument("--db", type=Path, required=True, help="SQLite index path")
    build.add_argument("--limit", type=int, default=None, help="max files scanned this run")
    build.set_defaults(handler=_cmd_build)

    query = commands.add_parser("query", help="look up references by address or prefix")
    query.add_argument("address")
    query.add_argument("--db", type=Path, required=True)
    query.add_argument("--json", action="store_true")
    query.add_argument(
        "--dedup",
        action="store_true",
        help="collapse report/run-receipt mirror pairs (report row wins)",
    )
    query.set_defaults(handler=_cmd_query)

    stats = commands.add_parser("stats", help="summarize the index")
    stats.add_argument("--db", type=Path, required=True)
    stats.add_argument("--json", action="store_true")
    stats.add_argument(
        "--dedup",
        action="store_true",
        help="report ref/address counts over mirror-collapsed rows",
    )
    stats.set_defaults(handler=_cmd_stats)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, SchemaVersionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        # Downstream pager/head closed the pipe; that is not an indexing error.
        sys.stderr.close()
        return 0


__all__ = [
    "MAX_FILE_BYTES",
    "XREF_SCHEMA",
    "XREF_USER_VERSION",
    "BuildReport",
    "EmptySweepError",
    "Reference",
    "SchemaVersionError",
    "build_index",
    "classify_path",
    "dedup_refs",
    "extract_addresses",
    "extract_references",
    "index_stats",
    "query_refs",
]


if __name__ == "__main__":
    raise SystemExit(main())

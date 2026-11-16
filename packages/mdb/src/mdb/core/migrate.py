"""``mdb records migrate PATH``: write the schema-v3 twin of a JSONL file of v2 records.

A record file (a ``records_to`` sink, or any file whose every line is a stored record, an
``mdb_payload`` line or an ``mdb_execution`` line) is migrated to a NEW file beside it
(``NAME.v3.jsonl``, or under ``--out``), never in place, with a map file ``NAME.v3.map.json``:
``{old id: new id}`` plus the source and target sha256. Each v2 record is verified, re-sealed
as v3 with the same content (``records.migrate_record``) and the target file is then read back
and every record compared with its source, field by field (canonical JSON of every field but
``schema``, ``id`` and ``digest``: every number bit-identical). Payload lines are copied byte
for byte; a v3 record already in the source is copied unchanged (its id maps to itself).

References inside a body to another record's id (a watch-hits record's ``trace``, a value's
``origin.record``) are content and keep the v2 id; the map file resolves them, and the v2 file
stays readable. Legacy ``mdb-circuit-investigation-v1`` records, project indexes and core files
are not rewritten: readers accept their v2 records as they are.

Torch-free. ``verify_file`` checks a record file without writing anything.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from . import records as R
from .pointers import RecordError

MAP_SCHEMA = "mdb-record-migration-v1"
SUFFIX = ".v3.jsonl"


def _declare_kinds() -> None:
    """Import the modules that declare record kinds a reader may meet (all torch-free)."""
    from . import kinds  # noqa: F401
    try:
        from . import recipe  # noqa: F401  (``checkpoint``)
    except ImportError:          # pragma: no cover - saturn_pub missing
        pass


def _lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        big = text.startswith('{"bytes":') and '"mdb_payload"' in text[:200]
        value = R.parse_line(text, f"{path}:{number}", limit=None if big else R.MAX_LINE_BYTES)
        if not isinstance(value, dict):
            raise RecordError(f"{path}:{number}: not a record, payload or execution line")
        out.append((text, value))
    return out


def _kind_of(line: dict[str, Any]) -> str:
    if "mdb_payload" in line:
        return "payload"
    if "mdb_execution" in line:
        return "execution"
    if line.get("schema") in R.SCHEMAS:
        return "record"
    return "other"


def is_record_file(path: str | Path) -> bool:
    """Every non-empty line is a stored record, payload or execution line (and one record)."""
    try:
        kinds = [_kind_of(v) for _, v in _lines(Path(path))]
    except (RecordError, OSError, UnicodeDecodeError):
        return False
    return "record" in kinds and "other" not in kinds


def _executions(lines: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    from .custody import _executions as checked

    return checked([line for _, line in lines])


def read_file(path: str | Path, *, verify: bool = True) -> list[dict[str, Any]]:
    """The logical records of a record file (v3 and v2 lines), verified unless asked not to."""
    _declare_kinds()
    lines = _lines(Path(path))
    if any(_kind_of(v) == "other" for _, v in lines):
        raise RecordError(f"{path} holds lines that are not records, payloads or executions "
                          "(an unknown or altered schema)")
    executions = _executions(lines)
    return [R.load_record(v, executions=executions, verify=verify)
            for _, v in lines if _kind_of(v) == "record"]


def verify_file(path: str | Path) -> dict[str, Any]:
    """Verify every record of a file; returns counts by schema and kind."""
    records = read_file(path)
    by_schema: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for record in records:
        by_schema[record["schema"]] = by_schema.get(record["schema"], 0) + 1
        by_kind[record["kind"]] = by_kind.get(record["kind"], 0) + 1
    return {"path": str(path), "records": len(records), "schemas": by_schema, "kinds": by_kind}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_of(path: Path, out: Path | None = None, root: Path | None = None) -> Path:
    name = path.name[:-len(".jsonl")] if path.name.endswith(".jsonl") else path.name
    if out is None:
        return path.with_name(name + SUFFIX)
    relative = path.parent.relative_to(root) if root is not None else Path()
    return out / relative / (name + SUFFIX)


def migrate_file(path: str | Path, target: str | Path | None = None, *,
                 force: bool = False) -> dict[str, Any]:
    """Write the v3 twin of one record file and its map; returns the map (also written).

    Nothing is written unless every record migrates and the written file reads back with the
    same content; ``target`` (default ``NAME.v3.jsonl`` beside ``path``) must not exist unless
    ``force``."""
    _declare_kinds()
    path = Path(path)
    target = Path(target) if target is not None else target_of(path)
    if not target.name.endswith(SUFFIX):
        raise RecordError(f"the target {target} must end in {SUFFIX}")
    if target.resolve() == path.resolve() or (target.exists() and path.exists()
                                              and target.samefile(path)):
        raise RecordError(f"the target {target} is the source; a migration never writes in place")
    map_path = target.with_name(target.name[:-len(SUFFIX)] + ".v3.map.json")
    if map_path.resolve() == path.resolve():
        raise RecordError(f"the map {map_path} is the source")
    if not force and target.exists():
        raise FileExistsError(f"{target} exists; pass force=True to replace it")
    lines = _lines(path)
    if any(_kind_of(v) == "other" for _, v in lines):
        raise RecordError(f"{path} holds lines that are not records, payloads or executions")
    executions = _executions(lines)
    out: list[str] = []
    written: set[str] = set()
    mapping: dict[str, str] = {}
    sources: list[dict[str, Any]] = []
    counts = {"records": 0, "migrated": 0, "already_v3": 0, "payloads": 0}
    for text, line in lines:
        kind = _kind_of(line)
        if kind == "payload":
            out.append(text)
            counts["payloads"] += 1
            continue
        if kind == "execution":
            continue                      # re-emitted before the first record that needs it
        old = R.load_record(line, executions=executions)
        new = R.migrate_record(old)
        counts["records"] += 1
        counts["migrated" if old["schema"] == R.SCHEMA_V2 else "already_v3"] += 1
        if old["id"] in mapping and mapping[old["id"]] != new["id"]:
            raise RecordError(f"record {old['id'][:12]} maps to two ids")
        mapping[old["id"]] = new["id"]
        sources.append(old)
        stored = R.stored_record(new)
        if stored["execution_sha256"] not in written:
            out.append(R._canonical({"mdb_execution": R.EXECUTION_SCHEMA,
                                     "sha256": stored["execution_sha256"],
                                     "execution": new["execution"]}))
            written.add(stored["execution_sha256"])
        out.append(R._canonical(stored))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text("".join(t + "\n" for t in out), encoding="utf-8")
    try:
        back = read_file(partial)
        if len(back) != len(sources):
            raise RecordError(f"read back {len(back)} records, wrote {len(sources)}")
        for old, new in zip(sources, back):
            R.same_content(old, new)
            if mapping[old["id"]] != new["id"] or new["schema"] != R.SCHEMA:
                raise RecordError(f"record {old['id'][:12]} read back under another id")
        before = {v["sha256"] for _, v in lines if _kind_of(v) == "payload"}
        after = {v["sha256"] for _, v in _lines(partial) if _kind_of(v) == "payload"}
        if before != after:
            raise RecordError("payload lines differ after migration")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    report = {"schema": MAP_SCHEMA, "source": str(path), "source_sha256": _sha(path),
              "target": str(target), "target_sha256": _sha(partial), **counts,
              "verified": len(back),
              "check": "every record read back from the target equals its source in every "
                       "field but schema, id and digest (canonical JSON)",
              "map": mapping}
    # the map first (atomically), then the target: a crash between leaves a map and no target,
    # which a rerun replaces; never a target without its map
    map_partial = map_path.with_name(map_path.name + ".partial")
    map_partial.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    map_partial.replace(map_path)
    partial.replace(target)
    report["map_path"] = str(map_path)
    return report


_LOOKS = re.compile(r'"schema"\s*:\s*"mdb-record-|"mdb_execution"\s*:')


def looks_like_record_file(path: str | Path) -> bool:
    """True when the file's text mentions a record schema or an execution line, even if it
    is truncated, altered or unreadable (so a scan reports it instead of skipping it)."""
    try:
        return bool(_LOOKS.search(Path(path).read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return True


def record_files(path: str | Path, *, include_targets: bool = False) -> list[Path]:
    """Record files at ``path``: the file itself, or every ``*.jsonl`` under a directory that
    looks like a record file (``looks_like_record_file``, so a damaged one is listed and then
    refused by the reader), not a ``.v3.jsonl`` target unless ``include_targets``."""
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.jsonl")
                  if (include_targets or not p.name.endswith(SUFFIX)) and looks_like_record_file(p))


USAGE = ("usage: mdb records migrate PATH [PATH ...] [--out DIR] [--force]\n"
         "       mdb records verify PATH [PATH ...]\n"
         "  migrate  write NAME.v3.jsonl and NAME.v3.map.json for every record file at PATH (a\n"
         "           JSONL file, or a directory searched for record files); never in place\n"
         "  verify   read every record of the files at PATH and check its seal")


def main(argv: list[str]) -> int:
    import sys

    if not argv or argv[0] in ("-h", "--help") or argv[0] not in ("migrate", "verify"):
        print(USAGE, file=sys.stderr if argv and argv[0] not in ("-h", "--help") else sys.stdout)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    verb, rest = argv[0], argv[1:]
    out = force = None
    paths = []
    i = 0
    while i < len(rest):
        if rest[i] == "--out" and i + 1 < len(rest):
            out, i = Path(rest[i + 1]), i + 2
        elif rest[i] == "--force":
            force, i = True, i + 1
        else:
            paths.append(Path(rest[i]))
            i += 1
    if not paths:
        print(USAGE, file=sys.stderr)
        return 2
    status = 0
    for root in paths:
        files = record_files(root, include_targets=verb == "verify")
        if not files:
            print(f"{root}: no record files", file=sys.stderr)
            status = 1
        for file in files:
            try:
                if verb == "verify":
                    report = verify_file(file)
                    print(f"{file}: {report['records']} records verified {report['schemas']}")
                    continue
                target = target_of(file, out, root if root.is_dir() else file.parent)
                report = migrate_file(file, target, force=bool(force))
                print(f"{file} -> {report['target']}: {report['records']} records "
                      f"({report['migrated']} migrated, {report['already_v3']} already v3), "
                      f"{report['verified']} read back equal; map {report['map_path']}")
            except (RecordError, OSError, ValueError) as error:
                print(f"{file}: refused: {error}", file=sys.stderr)
                status = 1
    return status


__all__ = ["MAP_SCHEMA", "SUFFIX", "is_record_file", "looks_like_record_file", "main", "migrate_file", "read_file",
           "record_files", "target_of", "verify_file"]

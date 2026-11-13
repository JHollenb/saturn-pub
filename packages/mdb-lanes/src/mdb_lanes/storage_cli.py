"""Plan, compact and verify immutable numerical evidence. No model imports."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from saturn_pub.shared_store import SharedStore, TENSOR_SUFFIXES, configure, file_identity, safe_path


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def plan(scope, pool, pattern):
    scope, pool = safe_path(scope), safe_path(pool)
    paths = []
    for root in sorted(scope.glob(pattern)):
        if root.is_symlink() or not root.is_dir() or root == pool:
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix in TENSOR_SUFFIXES and path.is_file():
                safe_path(path)
                paths.append(path)
    rows, inodes, contents = [], {}, {}
    for i, path in enumerate(paths):
        digest, st = file_identity(path)
        if st.st_dev != scope.stat().st_dev:
            raise ValueError("source crosses filesystem boundary")
        rows.append({"path": str(path), "sha256": digest, "bytes": st.st_size})
        inodes[(st.st_dev, st.st_ino)] = (st.st_size, st.st_blocks * 512)
        contents[digest] = st.st_size
        if (i + 1) % 500 == 0:
            print(f"hashed {i + 1}/{len(paths)} tensor files", file=sys.stderr, flush=True)
    return {"schema": "mdb-shared-storage-plan-v1", "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": str(scope), "store": str(pool), "pattern": pattern,
            "files": rows, "file_count": len(rows),
            "logical_bytes": sum(r["bytes"] for r in rows),
            "unique_content_bytes": sum(contents.values()),
            "unique_content_files": len(contents),
            "source_unique_inode_bytes": sum(v[0] for v in inodes.values()),
            "source_unique_inode_stat_allocated_bytes": sum(v[1] for v in inodes.values()),
            "scope_free_bytes": os.statvfs(scope).f_bavail * os.statvfs(scope).f_frsize}


def _read_plan(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "mdb-shared-storage-plan-v1":
        raise ValueError("unsupported shared storage plan")
    scope, pool = safe_path(value["scope"]), safe_path(value["store"])
    seen = set()
    for row in value["files"]:
        path = safe_path(row["path"])
        if (not path.is_relative_to(scope) or path.is_relative_to(pool)
                or path in seen or path.suffix not in TENSOR_SUFFIXES):
            raise ValueError("plan path escapes evidence scope or repeats")
        seen.add(path)
    return value


def compact(plan_path, journal):
    value = _read_plan(plan_path)
    pool = SharedStore(value["store"])
    totals = {"files": 0, "relinked": 0, "released_inode_logical_bytes": 0,
              "released_inode_stat_allocated_bytes": 0}
    # Append-only intent/completion journal permits verification after interruption.
    journal = Path(journal)
    if journal.exists():
        raise ValueError("journal already exists; resume with a new journal path")
    if journal.absolute() in {Path(r["path"]) for r in value["files"]}:
        raise ValueError("journal would overwrite scientific evidence")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("x") as stream:
        def event(row):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        for i, row in enumerate(value["files"]):
            if Path(row["path"]).stat().st_size != row["bytes"]:
                raise ValueError("source size changed from plan")
            event({"stage": "intent", **row})
            result = pool.intern(row["path"], expected_sha256=row["sha256"])
            event({"stage": "complete", **result})
            totals["files"] += 1
            for k in ["relinked", "released_inode_logical_bytes", "released_inode_stat_allocated_bytes"]:
                totals[k] += result[k]
            if (i + 1) % 500 == 0:
                print(f"shared {i + 1}/{len(value['files'])} tensor files", file=sys.stderr, flush=True)
    return {"schema": "mdb-shared-storage-compaction-v1", "plan": str(Path(plan_path).absolute()),
            "journal": str(journal.absolute()), **totals,
            "scope_free_bytes_after": os.statvfs(value["scope"]).f_bavail * os.statvfs(value["scope"]).f_frsize}


def verify(plan_path):
    value = _read_plan(plan_path)
    pool, checked, allocated = SharedStore(value["store"]), set(), 0
    for row in value["files"]:
        path = safe_path(row["path"])
        obj = pool.object_path(row["sha256"])
        st = path.stat()
        if st.st_size != row["bytes"] or st.st_mode & 0o222 or not os.path.samefile(path, obj):
            raise ValueError("evidence is not a sealed shared link: " + str(path))
        if row["sha256"] not in checked:
            digest, st = file_identity(path)
            if digest != row["sha256"] or st.st_size != row["bytes"]:
                raise ValueError("shared evidence bytes changed")
            checked.add(digest)
            allocated += st.st_blocks * 512
    return {"schema": "mdb-shared-storage-verification-v1", "mechanics_status": "verified",
            "files": len(value["files"]), "unique_objects": len(checked),
            "unique_object_stat_allocated_bytes": allocated,
            "logical_bytes": value["logical_bytes"],
            "scope": "Byte-identical sharing; original paths/manifests unchanged. Allocation counters do not resolve APFS clones."}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mdb storage")
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("configure")
    c.add_argument("--scope", required=True); c.add_argument("--store", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--scope", required=True); p.add_argument("--store", required=True)
    p.add_argument("--pattern", default="*"); p.add_argument("--report", required=True)
    c = commands.add_parser("compact")
    c.add_argument("--plan", required=True); c.add_argument("--journal", required=True)
    c.add_argument("--report", required=True)
    v = commands.add_parser("verify")
    v.add_argument("--plan", required=True); v.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "report") and Path(args.report).exists():
            raise ValueError("report already exists; use a new report path")
        if args.command == "configure":
            result = {"config": str(configure(args.scope, args.store))}
        elif args.command == "plan":
            result = plan(args.scope, args.store, args.pattern)
        elif args.command == "compact":
            result = compact(args.plan, args.journal)
        else:
            result = verify(args.plan)
        if hasattr(args, "report"):
            _save(args.report, result)
        print(json.dumps({k: v for k, v in result.items() if k != "files" or not isinstance(v, list)}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

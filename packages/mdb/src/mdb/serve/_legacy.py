"""The ONLY place ``mdb serve`` touches the old flat ``mdb.*`` modules.

Serve must not import flat modules at top level (``mdb.project``, ``mdb.symbol_lifecycle`` and
``mdb.symbol_reads`` pull ~126 modules through an import cycle). Each use is one small function
with a lazy import and a comment naming the flat module and exactly what is read from it, so
this file can be replaced by slim ``mdb.core`` readers without touching the views.

Every function here is a read; none writes to the project or creates a table. The ``Project``
members used: ``__init__(read_only=True)``, ``store``, ``sessions``, ``session``, ``cut_ids``,
``cut_descriptor``, ``receipts``, ``symbols``, ``slice_ids``, ``slice_record``,
``symbol_lifecycles``, ``findings``, ``graph`` and ``_db`` (overridden, see ``open_project``).
"""
from __future__ import annotations

# --------------------------------------------------------------------------- mdb.project

def open_project(root):
    """A ``mdb.project.Project`` over ``root`` whose every sqlite connection is read-only.

    ``Project(root, read_only=True)`` already opens ``mode=ro``, but SQLite may still create side
    files next to the database (``-shm``/``-wal`` for a WAL database, a rollback journal). This
    subclass replaces ``Project._db`` (the one place the base class connects; every ``Project``
    read and every ``mdb.core.index`` read goes through it) so that:

    * no ``-wal`` file next to the database: ``file:...?mode=ro&immutable=1`` (SQLite takes no
      locks and creates nothing; the file is assumed not to change while served);
    * a ``-wal`` file exists (a project converted to WAL elsewhere): ``mode=ro``, which needs the
      directory to be writable for SQLite's ``-shm`` wal-index; when it is not, this raises
      ``ValueError`` (one clear message) so the server fails at startup instead of per request.

    ``PRAGMA query_only=ON`` is set on every connection. The ``Project`` methods this module
    relies on are listed below; none could be avoided (``Project`` takes no connection or URI),
    so only ``_db`` is overridden and nothing in ``mdb/project.py`` is touched.
    """
    import os
    import sqlite3
    from contextlib import contextmanager
    from pathlib import Path

    from ..project import Project  # flat module: mdb.project.Project

    root = Path(root).expanduser().resolve()
    if (root / "project.sqlite-wal").exists() and not os.access(root, os.W_OK):
        raise ValueError(
            f"{root} is a WAL-mode project (it has a project.sqlite-wal file) in a directory "
            "that is not writable; SQLite needs to create its -shm wal-index there to read it. "
            "Make the directory writable, or serve a rollback-journal copy of the project "
            "(sqlite3 project.sqlite 'PRAGMA journal_mode=DELETE').")

    class ReadOnlyProject(Project):
        @contextmanager
        def _db(self):
            path = self.root / "project.sqlite"
            mode = "mode=ro" if path.with_name(path.name + "-wal").exists() else \
                "mode=ro&immutable=1"
            db = sqlite3.connect(path.as_uri() + "?" + mode, uri=True, timeout=30)
            db.row_factory = sqlite3.Row
            try:
                db.execute("PRAGMA query_only=ON")
                with db:
                    yield db
            finally:
                db.close()

    return ReadOnlyProject(root, read_only=True)


def sessions(project):
    """The session ids, ``SELECT id FROM sessions`` over ``Project._db`` (``Project.sessions()``
    would parse every session's JSON, so one corrupt row would hide the healthy sessions)."""
    with project._db() as db:
        return [row["id"] for row in db.execute("SELECT id FROM sessions ORDER BY id")]


def session_row(project, sid):
    """Project.session(sid) -> row with identity, execution, selected, debug_info (JSON text)."""
    return project.session(sid)


def cut_ids(project, sid):
    """Project.cut_ids(sid)."""
    return project.cut_ids(sid)


def cut_descriptor(project, sid, cut):
    """Project.cut_descriptor(sid, cut) -> dict with execution_point, surface, parent, ..."""
    return project.cut_descriptor(sid, cut)


def receipts(project, sid):
    """Project.receipts(sid) -> NativeReceipt objects; read as ``to_dict()``."""
    return [receipt.to_dict() for receipt in project.receipts(sid)]


def symbols(project, sid):
    """Project.symbols(sid) -> symbol binding rows."""
    return project.symbols(sid)


def slice_ids(project, sid):
    """Project.slice_ids(sid)."""
    return project.slice_ids(sid)


def slice_record(project, sid, slice_id):
    """Project.slice_record(sid, slice_id) -> sealed slice dict."""
    return project.slice_record(sid, slice_id)


def symbol_lifecycles(project, sid):
    """Project.symbol_lifecycles(sid) -> lifecycle rows (``lifecycle``, ``joins``, ids)."""
    return project.symbol_lifecycles(sid)


def findings(project, sid):
    """Project.findings(sid) -> state annotations (``annotation_id``, ``details``, ...)."""
    return project.findings(sid)


def session_graph(project, sid):
    """Project.graph(sid).seal().to_dict(): the sealed session state graph."""
    return project.graph(sid).seal().to_dict()


def store_root(project):
    """Project.store.root: the saturn_pub LocalStore directory (read by ``serve.blobs``)."""
    return project.store.root


def tensor_info(project, ref):
    """Project.store.tensor_info(ref): shape/dtype of a stored tensor (no bytes read)."""
    return project.store.tensor_info(ref)


def db(project):
    """Project._db(): sqlite connection context (serve only runs SELECTs through it)."""
    return project._db()


# --------------------------------------------------------------------------- symbol lifecycles

def lifecycle_reads(body):
    """Attention reads recorded in a symbol lifecycle body, as ``SymbolRead.to_dict()`` dicts.

    Reads: mdb.symbol_lifecycle.SymbolLifecycle.from_dict (``symbol``, ``binding_fingerprint``)
    and mdb.symbol_reads.ReadSource / reads_from_lifecycle / SymbolReadError. The read source is
    the project binding (rows from the lifecycle's ``uses[].metadata.source_row``), class
    ``supplied``. Empty when the lifecycle carries no attention trace or no uses.
    """
    trace = (body.get("metadata") or {}).get("attention_trace")
    if not trace or not body.get("uses"):
        return []
    from ..symbol_lifecycle import SymbolLifecycle  # flat module: mdb.symbol_lifecycle
    from ..symbol_reads import ReadSource, SymbolReadError, reads_from_lifecycle  # mdb.symbol_reads

    lifecycle = SymbolLifecycle.from_dict(body)
    rows = sorted({int(u["metadata"].get("source_row", 0)) for u in body["uses"]
                   if isinstance(u.get("metadata"), dict) and "source_row" in u["metadata"]})
    fingerprint = str(lifecycle.binding_fingerprint)[:12]
    source = ReadSource(lifecycle.symbol, tuple(rows or [0]), "supplied",
                        {"by": "project binding " + fingerprint})
    try:
        return [read.to_dict() for read in reads_from_lifecycle(lifecycle, source=source)]
    except (SymbolReadError, KeyError, ValueError):
        return []

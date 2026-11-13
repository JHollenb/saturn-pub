"""The engine's own durable project: a schema-3 ``project.sqlite`` plus a saturn_pub ``store/``.

The directory is the one the lane ``mdb_lanes.project.Project`` writes and
``mdb.core.project_read.ProjectReader`` (``mdb serve``) reads: the same seven schema-3 tables,
``PRAGMA user_version = 3``, the same rollback-journal SQLite discipline (one connection per
operation, ``timeout=30``, ``BEGIN IMMEDIATE`` around read-modify-write) and the same session,
cut and store layout. Nothing here needs the lane package, a model or a framework.

* Creation is atomic: the database and ``store/`` are built in a hidden sibling directory and
  renamed into place, so a crash leaves either no project or a whole one. When another process
  wins the race the loser opens the winner's project.
* ``read_only=True`` opens SQLite with ``mode=ro`` and never creates a file or a directory, so a
  read-only open (and every read through it) changes no byte of the tree.
* The index tables of ``mdb.core.index`` (records, cursor, recipes, occurrences) are additive
  and created lazily by that module on its first write; this module creates only the schema-3
  tables. ``db()``, ``json()``, ``read_only``, ``session()`` and ``store`` are what
  ``mdb.core.index`` uses.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

from saturn_pub.store import LocalStore
from saturn_pub.values import canonical

SCHEMA_VERSION = 3
DATABASE = "project.sqlite"
STORE = "store"

# The lane project's schema-3 tables, verbatim, so either writer opens what the other made.
_SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, identity TEXT NOT NULL, execution TEXT NOT NULL,
        selected TEXT NOT NULL, debug_info TEXT NOT NULL, revision INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS cuts (
        session_id TEXT NOT NULL, id TEXT NOT NULL, manifest TEXT NOT NULL,
        PRIMARY KEY (session_id, id));
    CREATE TABLE IF NOT EXISTS receipts (
        session_id TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
        PRIMARY KEY (session_id, id));
    CREATE TABLE IF NOT EXISTS imports (id TEXT PRIMARY KEY, body TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, analyzer TEXT NOT NULL,
        version TEXT NOT NULL, input_digest TEXT NOT NULL, body TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS analysis_inputs
        ON analyses(session_id, analyzer, version, input_digest);
    CREATE TABLE IF NOT EXISTS symbols (
        session_id TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
        PRIMARY KEY(session_id, id));
    CREATE TABLE IF NOT EXISTS slices (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, body TEXT NOT NULL);
    PRAGMA user_version = 3;
"""
_TABLES = ("sessions", "cuts", "receipts", "imports", "analyses", "symbols", "slices")


class ProjectError(ValueError):
    """The directory is not a usable mdb project, or a write would break a session binding."""


def _connect(path, read_only):
    if read_only:
        return sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True, timeout=30)
    return sqlite3.connect(path, timeout=30)


def _fsync_tree(root):
    for path in (root / DATABASE, root / STORE, root):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        except OSError:  # some filesystems cannot fsync a directory; the files are flushed
            pass
        finally:
            os.close(descriptor)


class Project:
    """A durable mdb project directory, opened for writing (created when absent) or read-only."""

    def __init__(self, root, *, read_only=False):
        self.root = Path(root).expanduser().resolve()
        self.read_only = bool(read_only)
        database = self.root / DATABASE
        if read_only:
            if not database.is_file() or not (self.root / STORE).is_dir():
                raise ProjectError(f"{self.root} is not an mdb project (read-only open needs "
                                   f"{DATABASE} and {STORE}/)")
        elif not database.is_file():
            self._create()
        self.store = LocalStore(self.root / STORE)
        with self.db() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if read_only and version != SCHEMA_VERSION:
                raise ProjectError(f"read-only project requires schema {SCHEMA_VERSION}, found "
                                   f"{version}; open a writable copy first")
            if not read_only:
                if version not in (0, 1, 2, SCHEMA_VERSION):
                    raise ProjectError(f"unsupported mdb project schema {version}")
                present = {r[0] for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                if version != SCHEMA_VERSION or not set(_TABLES) <= present:
                    db.executescript(_SCHEMA)   # the lane's own migration to schema 3
        if not read_only:
            (self.root / STORE).mkdir(exist_ok=True)

    def _create(self):
        """Build the project beside its final path and rename it into place (all or nothing)."""
        if self.root.exists() and (not self.root.is_dir() or any(self.root.iterdir())):
            raise ProjectError(f"{self.root} exists and is not an mdb project (no {DATABASE})")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{self.root.name}.", suffix=".creating",
                                        dir=self.root.parent))
        try:
            (staging / STORE).mkdir()
            db = sqlite3.connect(staging / DATABASE)
            try:
                db.executescript(_SCHEMA)
                db.commit()
            finally:
                db.close()
            _fsync_tree(staging)
            try:
                os.rename(staging, self.root)   # replaces only an absent or empty directory
            except OSError:
                if not (self.root / DATABASE).is_file():
                    raise
                # another process created the project first: open theirs
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @contextmanager
    def db(self):
        """One SQLite connection (``Row`` rows), committed on success; read-only when opened so."""
        db = _connect(self.root / DATABASE, self.read_only)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def json(value):
        """The canonical JSON text every project row is stored as (the lane's ``_json``)."""
        return canonical(value).decode("utf-8")

    def _writable(self):
        if self.read_only:
            raise ProjectError("this project was opened read-only")

    # ----------------------------------------------------------------------- sessions

    def sessions(self):
        """Every session: ``{session_id, model_identity, execution, selected_cut}`` by id."""
        with self.db() as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY id").fetchall()
        return [{"session_id": r["id"], "model_identity": json.loads(r["identity"]),
                 "execution": json.loads(r["execution"]), "selected_cut": r["selected"]}
                for r in rows]

    def session(self, session_id):
        """The raw session row (``id, identity, execution, selected, debug_info, revision``)."""
        with self.db() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown project session: {session_id}")
        return dict(row)

    def open_session(self, session_id, *, model_identity, execution):
        """Create the session bound to this model identity and execution, or reopen it.

        A new session selects no cut yet (``selected`` is empty) and is at revision 0, the
        revision the lane writer expects of a session it has not saved. Reopening a session
        bound to another model identity or execution is refused.
        """
        self._writable()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ProjectError("a project session needs a non-empty id")
        identity, profile = self.json(model_identity), self.json(dict(execution))
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                           (session_id, identity, profile, "", self.json({}), 0))
            elif row["identity"] != identity or row["execution"] != profile:
                raise ProjectError(f"project session {session_id!r} is bound to another model "
                                   "or execution")
        return self.session(session_id)

    def select(self, session_id, cut_id):
        """Make an indexed cut the session's selected cut (one revision step)."""
        self._writable()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM cuts WHERE session_id=? AND id=?",
                          (session_id, cut_id)).fetchone() is None:
                raise ProjectError(f"cut {cut_id} is not retained in session {session_id!r}")
            db.execute("UPDATE sessions SET selected=?, revision=revision+1 WHERE id=?",
                       (cut_id, session_id))

    # ----------------------------------------------------------------------- cuts

    def save_cut(self, session_id, cut):
        """Publish a native cut to the store, then index it under the session (the lane order:
        immutable bytes first, so an index row never names missing bytes). Returns its id."""
        self._writable()
        row = self.session(session_id)
        if (self.json(cut.model_identity) != row["identity"]
                or self.json(dict(cut.execution)) != row["execution"]):
            raise ProjectError("the cut belongs to another model or execution than the session")
        self.store.save(cut)
        body = self.json(cut.manifest())
        with self.db() as db:
            db.execute("INSERT OR IGNORE INTO cuts VALUES (?, ?, ?)",
                       (session_id, cut.fingerprint, body))
            stored = db.execute("SELECT manifest FROM cuts WHERE session_id=? AND id=?",
                                (session_id, cut.fingerprint)).fetchone()[0]
        if stored != body:
            raise ProjectError("project cut index identity collision")
        return cut.fingerprint

    def cut_ids(self, session_id):
        """Cut ids indexed under the session, in insertion order."""
        self.session(session_id)
        with self.db() as db:
            return [r[0] for r in db.execute(
                "SELECT id FROM cuts WHERE session_id=? ORDER BY rowid", (session_id,))]

    def load_cut(self, session_id, cut_id, *, device="cpu"):
        """A retained native cut, checked against its index row and the session binding."""
        with self.db() as db:
            row = db.execute("SELECT manifest FROM cuts WHERE session_id=? AND id=?",
                             (session_id, cut_id)).fetchone()
        if row is None:
            raise ProjectError(f"cut {cut_id} is not retained in session {session_id!r}")
        cut = self.store.load(cut_id, device=device)
        if self.json(cut.manifest()) != row[0]:
            raise ProjectError("project cut index differs from the native cut")
        return cut

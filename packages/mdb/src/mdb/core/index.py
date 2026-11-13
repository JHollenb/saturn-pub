"""Project index: sealed records, the cursor, replay recipes and occurrence locations.

Functions take a project, duck-typed: ``mdb.core.project.Project`` (the engine's own), the lane
``mdb_lanes.project.Project`` or ``mdb.core.project_read.ProjectReader``. What is used is
``read_only``, ``session(sid)``, ``store`` and a connection: the public ``db()`` of the engine
project, else the ``_db()`` the two other classes offer (the one duck-typed fallback, so this
module imports neither class nor a framework). Row text is ``saturn_pub.values.canonical``.

The index is purely additive to a schema-3 project (``user_version`` stays 3, so every other
checkout opens the same directory)::

    records(id PRIMARY KEY, session_id, kind, schema, body)   # mdb-record-v2, stored_form
    cursor(session_id PRIMARY KEY, record_id)
    recipes(id PRIMARY KEY, session_id, body)
    occurrences(session_id, address, recipe_id, event_id, cut_id, blob_id,
                PRIMARY KEY(session_id, address))

The four tables are created with ``CREATE TABLE IF NOT EXISTS`` in ONE transaction, lazily, by
the first write through this module (``ensure_tables``); never on a read and never on a
read-only connection. A reader treats a missing table as empty. ``circuit_records`` and
``circuit_focus`` are the tables ``circuit_investigation`` creates and writes, untouched:
``list_records``/``get_record`` pass legacy ``mdb-circuit-investigation-v1`` rows through
(read-only, marked by their own schema string) and return v2 rows from ``records``. Writes of
v2 records go only to ``records``; legacy writes stay with ``circuit_investigation``.
"""
from __future__ import annotations

import json

from saturn_pub.values import canonical

from .pointers import RecordError, _sha
from .records import LEGACY_SCHEMAS, SCHEMA, kind_spec, stored_form, verify_legacy, verify_record

LEGACY_CIRCUIT = LEGACY_SCHEMAS[0]
MAX_RECIPE_BYTES = 256 * 1024

TABLES = ("records", "cursor", "recipes", "occurrences")
_TABLE_STATEMENTS = (
    """CREATE TABLE IF NOT EXISTS records (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, kind TEXT NOT NULL,
        schema TEXT NOT NULL, body TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS records_session ON records(session_id, kind)",
    """CREATE TABLE IF NOT EXISTS cursor (
        session_id TEXT PRIMARY KEY, record_id TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS recipes (
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL, body TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS occurrences (
        session_id TEXT NOT NULL, address TEXT NOT NULL, recipe_id TEXT,
        event_id INTEGER, cut_id TEXT, blob_id TEXT,
        PRIMARY KEY (session_id, address))""",
)


class NotIndexed(LookupError):
    """The project index holds no such record, recipe or cursor."""


def _connect(project):
    """A connection context of ``project``: its public ``db()``, else its ``_db()``."""
    opener = getattr(project, "db", None)
    return opener() if callable(opener) else project._db()


def _text(value):
    """The canonical JSON text a row is stored as (what every project writer's ``_json`` is)."""
    return canonical(value).decode("utf-8")


def _tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def migrate(db):
    """Ensure the additive tables on an open writable connection (idempotent, one transaction).

    Kept for callers of the earlier schema-4 name; it changes nothing else (no data moves, no
    ``user_version``).
    """
    if set(TABLES) <= _tables(db):
        return
    if not db.in_transaction:
        db.execute("BEGIN IMMEDIATE")
    for statement in _TABLE_STATEMENTS:
        db.execute(statement)


def ensure_tables(project):
    """Create the four index tables if absent, atomically. Called by every write below."""
    _writable(project)
    with _connect(project) as db:
        migrate(db)


def _exists(db, table):
    return table in _tables(db)


def _writable(project):
    if project.read_only:
        raise RecordError("writing the project index requires a writable project")


def _binding(project, sid):
    row = project.session(sid)
    return json.loads(row["identity"]), json.loads(row["execution"])


# --------------------------------------------------------------------------- records

def put_record(project, record):
    """Index a sealed v2 record under its session; returns its id. Idempotent."""
    _writable(project)
    record = verify_record(record)
    identity, execution = _binding(project, record["session_id"])
    if record["model_identity"] != identity or record["execution"] != execution:
        raise RecordError("record belongs to another model or execution than its session")
    body = stored_form(record)   # the bytes the size bound, the digest and the row all share
    ensure_tables(project)
    with _connect(project) as db:
        db.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?,?)",
                   (record["id"], record["session_id"], record["kind"], SCHEMA, body))
        existing = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()
        if existing[0] != body:
            raise RecordError("record identity collision")
    return record["id"]


def _row(db, sid, identifier):
    """(kind, schema, body) of one indexed record: a v2 row first, then a legacy circuit row."""
    if _exists(db, "records"):
        row = db.execute("SELECT kind, schema, body FROM records WHERE session_id=? AND id=?",
                         (sid, identifier)).fetchone()
        if row is not None:
            return row["kind"], row["schema"], row["body"]
    if _exists(db, "circuit_records"):
        row = db.execute("SELECT kind, body FROM circuit_records WHERE session_id=? AND id=?",
                         (sid, identifier)).fetchone()
        if row is not None:
            return row["kind"], LEGACY_CIRCUIT, row["body"]
    return None


def _verified(project, sid, identifier, kind, schema, body):
    identity, execution = _binding(project, sid)
    try:
        value = json.loads(body)
    except (ValueError, RecursionError) as error:   # not JSON, an integer past the digit limit,
        # or nesting deeper than the interpreter's recursion limit
        raise RecordError(f"record index row is not readable JSON: {error}") from error
    if schema == SCHEMA:
        value = verify_record(value)
        if value["id"] != identifier or value["kind"] != kind or value["session_id"] != sid:
            raise RecordError("record index row differs from its seal")
    elif schema in LEGACY_SCHEMAS:
        value = verify_legacy(value, identifier)
        if value.get("kind") != kind or value.get("session_id") != sid:
            raise RecordError("legacy record index row differs from its seal")
    else:
        raise RecordError(f"unknown record schema {schema!r}")
    if value.get("model_identity") != identity or value.get("execution") != execution:
        raise RecordError("record seal or session binding changed")
    return value


def get_record(project, sid, identifier):
    """A verified record: a v2 envelope, or a legacy circuit record ``{record_id, **body}``."""
    with _connect(project) as db:
        row = _row(db, sid, identifier)
    if row is None:
        raise NotIndexed(f"record {identifier} is not indexed in session {sid}")
    return _verified(project, sid, identifier, *row)


def list_records(project, sid, *, kind=None, schema=None, strict=False):
    """Verified records of a session: legacy circuit records (their rowid order), then v2.

    The two kinds live in different tables, so their relative insertion order is not kept:
    legacy rows come first, each group in insertion order. Missing tables read as empty.

    One bad row never fails the whole listing. A row that cannot be returned verified is a
    placeholder in its position, ``{id, kind, schema, verified: False, state, reason}``:
    ``state`` ``"unresolved"`` when its kind is not declared in this process (another process
    may declare it), ``"refused"`` when it fails verification for any other reason (a tampered
    seal, a changed session binding). ``strict=True`` raises on the first such row instead, as
    ``get_record`` does.
    """
    project.session(sid)
    rows = []
    with _connect(project) as db:
        if _exists(db, "circuit_records"):
            rows += [(r[0], r[1], LEGACY_CIRCUIT, r[2]) for r in db.execute(
                "SELECT id, kind, body FROM circuit_records WHERE session_id=? ORDER BY rowid",
                (sid,))]
        if _exists(db, "records"):
            rows += [tuple(r) for r in db.execute(
                "SELECT id, kind, schema, body FROM records WHERE session_id=? ORDER BY rowid",
                (sid,))]
    out = []
    for i, k, s, b in rows:
        if (kind is not None and k != kind) or (schema is not None and s != schema):
            continue
        if not strict and s == SCHEMA and kind_spec(k) is None:
            out.append({"id": i, "kind": k, "schema": s, "verified": False, "state": "unresolved",
                        "reason": f"record kind {k!r} is not declared in this process"})
            continue
        try:
            out.append(_verified(project, sid, i, k, s, b))
        except (RecordError, ValueError, KeyError, TypeError, RecursionError) as error:
            if strict:
                raise
            out.append({"id": i, "kind": k, "schema": s, "verified": False, "state": "refused",
                        "reason": str(error)})
    return out


# --------------------------------------------------------------------------- cursor

def set_cursor(project, sid, record_id):
    """Point the session cursor at an indexed record."""
    _writable(project)
    get_record(project, sid, record_id)
    ensure_tables(project)
    with _connect(project) as db:
        db.execute("INSERT INTO cursor VALUES (?,?) ON CONFLICT(session_id) "
                   "DO UPDATE SET record_id=excluded.record_id", (sid, record_id))
    return record_id


def get_cursor(project, sid):
    """The record id under the session cursor, or None when no cursor is set."""
    project.session(sid)
    with _connect(project) as db:
        if not _exists(db, "cursor"):
            return None
        row = db.execute("SELECT record_id FROM cursor WHERE session_id=?", (sid,)).fetchone()
    return None if row is None else row[0]


# --------------------------------------------------------------------------- recipes

def recipe_fingerprint(body):
    """The ``native_replay_recipe.fingerprint`` convention (sha256 of ASCII canonical JSON)."""
    import hashlib

    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def put_recipe(project, sid, recipe):
    """Index a recipe (sealed with ``fingerprint`` or sealed here); returns its id.

    Recipes are content-addressed and global to the project: the same recipe saved from two
    sessions is stored once, under the first session.
    """
    _writable(project)
    project.session(sid)
    value = json.loads(json.dumps(recipe, allow_nan=False))
    if not isinstance(value, dict):
        raise RecordError("a recipe is a JSON object")
    expected = value.pop("fingerprint", None)
    identifier = recipe_fingerprint(value)
    if expected is not None and expected != identifier:
        raise RecordError("recipe seal differs from its content")
    sealed = {**value, "fingerprint": identifier}
    body = _text(sealed)
    if len(body.encode()) > MAX_RECIPE_BYTES:
        raise RecordError(f"recipe exceeds {MAX_RECIPE_BYTES} bytes")
    ensure_tables(project)
    with _connect(project) as db:
        db.execute("INSERT OR IGNORE INTO recipes VALUES (?,?,?)", (identifier, sid, body))
        existing = db.execute("SELECT body FROM recipes WHERE id=?", (identifier,)).fetchone()
        if existing[0] != body:
            raise RecordError("recipe identity collision")
    return identifier


def recipe_ids(project, sid):
    """Ids of the recipes indexed under a session, in insertion order."""
    project.session(sid)
    with _connect(project) as db:
        if not _exists(db, "recipes"):
            return []
        return [r[0] for r in db.execute(
            "SELECT id FROM recipes WHERE session_id=? ORDER BY rowid", (sid,))]


def get_recipe(project, identifier):
    """A verified recipe by id."""
    with _connect(project) as db:
        row = (db.execute("SELECT body FROM recipes WHERE id=?", (identifier,)).fetchone()
               if _exists(db, "recipes") else None)
    if row is None:
        raise NotIndexed(f"recipe {identifier} is not indexed")
    value = json.loads(row[0])
    body = {k: v for k, v in value.items() if k != "fingerprint"}
    if value.get("fingerprint") != identifier or recipe_fingerprint(body) != identifier:
        raise RecordError("recipe seal differs from its index id")
    return value


# --------------------------------------------------------------------------- occurrences

_OCCURRENCE_COLUMNS = ("recipe_id", "event_id", "cut_id", "blob_id")


def _occurrence_key(address, ports, global_ports, layers):
    from . import address as addresses
    from .pointers import _occurrence_address

    occurrence = _occurrence_address(address, ports, global_ports)
    if not isinstance(occurrence, addresses.Occurrence):
        raise RecordError("only port occurrences are indexed (a state cell has no clock)")
    try:
        occurrence = addresses.value_key(occurrence, layers=layers)
    except addresses.AddressError as error:
        raise RecordError(f"occurrence address: {error}") from error
    return addresses.format_address(occurrence)


def index_occurrence(project, sid, address, *, layers, recipe_id=None, event_id=None, cut_id=None,
                     blob_id=None, ports=None, global_ports=None):
    """Record where an occurrence value can be found or rebuilt.

    Keyed by the canonical address text. Re-indexing may fill fields that were null; a field
    that is already set must not change. ``blob_id`` must name a tensor in the project store.
    Address text is parsed against ``ports``/``global_ports`` (default: the Qwen2 list).
    ``layers`` (the model's layer count) is required: the row is keyed by ``address.value_key``,
    so ``tN.emb`` and ``tN.L0.h`` (or ``tN.Lk.h_out`` and ``tN.L{k+1}.h``) are one row, equal
    selected sets (``h=1|2``, ``h=1:3``) are one row, and a resolved logits token is ``i``.
    """
    _writable(project)
    project.session(sid)
    key = _occurrence_key(address, ports, global_ports, layers)
    if event_id is not None and (isinstance(event_id, bool) or not isinstance(event_id, int)
                                 or event_id < 0):
        raise RecordError("event_id must be a non-negative recipe event index")
    if cut_id is not None:
        cut_id = _sha(cut_id, "cut id")
    if blob_id is not None:
        blob_id = _sha(blob_id, "blob id")
        if not project.store.has_tensor(blob_id, verify=True):
            raise RecordError(f"tensor {blob_id} is not in the project store (or is corrupt)")
    if recipe_id is not None:
        with _connect(project) as db:
            if not _exists(db, "recipes") or db.execute(
                    "SELECT 1 FROM recipes WHERE id=?", (recipe_id,)).fetchone() is None:
                raise RecordError(f"recipe {recipe_id} is not indexed")
    ensure_tables(project)
    new = dict(zip(_OCCURRENCE_COLUMNS, (recipe_id, event_id, cut_id, blob_id)))
    with _connect(project) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM occurrences WHERE session_id=? AND address=?",
                         (sid, key)).fetchone()
        merged = dict(new)
        if row is not None:
            for column in _OCCURRENCE_COLUMNS:
                old = row[column]
                if old is not None and new[column] is not None and old != new[column]:
                    raise RecordError(f"occurrence {key} already has a different {column}")
                merged[column] = old if new[column] is None else new[column]
        db.execute("INSERT INTO occurrences VALUES (?,?,?,?,?,?) "
                   "ON CONFLICT(session_id, address) DO UPDATE SET recipe_id=excluded.recipe_id, "
                   "event_id=excluded.event_id, cut_id=excluded.cut_id, blob_id=excluded.blob_id",
                   (sid, key, *(merged[c] for c in _OCCURRENCE_COLUMNS)))
    return {"session_id": sid, "address": key, **merged}


def find_occurrence(project, sid, address, *, layers, ports=None, global_ports=None):
    """The index row for an occurrence, or None when it was never indexed.

    ``layers`` (required): look up under ``value_key`` (see ``index_occurrence``)."""
    key = _occurrence_key(address, ports, global_ports, layers)
    project.session(sid)
    with _connect(project) as db:
        if not _exists(db, "occurrences"):
            return None
        row = db.execute("SELECT * FROM occurrences WHERE session_id=? AND address=?",
                         (sid, key)).fetchone()
    return None if row is None else dict(row)

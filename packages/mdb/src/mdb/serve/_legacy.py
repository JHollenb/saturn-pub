"""The one place ``mdb serve`` reads a recorded project: thin calls into ``mdb.core.project_read``.

Historically this module wrapped the flat ``mdb.project`` lane modules; it now wraps the engine
reader (standard library + ``saturn_pub`` + ``mdb.core``) and keeps its function names, so views
and tests that patch these names are unchanged. Every function is a read; none writes to the
project or creates a table.

Listing functions take ``refuse(kind, identifier, error)``: a bad row is reported through it and
skipped (strict when it is omitted). Cuts and receipts stay strict: a session whose execution
history does not verify is refused as a whole by ``Recorded``.
"""
from __future__ import annotations

from ..core.project_read import ProjectReader
from ..core.project_read import lifecycle_reads as _lifecycle_reads


def open_project(root):
    """A ``ProjectReader`` over ``root``: every sqlite connection is read-only (``mode=ro`` with
    ``immutable=1`` unless a ``-wal`` file exists; ``PRAGMA query_only=ON``). A WAL project in an
    unwritable directory raises ``ValueError`` at open."""
    return ProjectReader(root)


def sessions(project):
    """Session ids (``SELECT id FROM sessions``): a corrupt session row hides no other session."""
    return project.sessions()


def session_row(project, sid):
    """Session row: identity, execution, selected, debug_info (JSON text), revision."""
    return project.session(sid)


def cut_ids(project, sid):
    return project.cut_ids(sid)


def cut_descriptor(project, sid, cut):
    """Verified cut metadata: execution_point, surface, parent, boundary, manifest."""
    return project.cut_descriptor(sid, cut)


def receipts(project, sid):
    """Verified native receipts as dicts."""
    return project.receipts(sid)


def symbols(project, sid, refuse=None):
    return project.symbols(sid) if refuse is None else project.symbols(sid, refuse=refuse)


def slice_ids(project, sid):
    return project.slice_ids(sid)


def slice_record(project, sid, slice_id):
    """Verified sealed slice dict."""
    return project.slice_record(sid, slice_id)


def symbol_lifecycles(project, sid, refuse=None):
    """Verified lifecycle rows (``lifecycle_id``, ``lifecycle``, ``joins``)."""
    return (project.symbol_lifecycles(sid) if refuse is None
            else project.symbol_lifecycles(sid, refuse=refuse))


def findings(project, sid, refuse=None):
    """State annotations tagged ``finding`` (``annotation_id``, ``details``, ...)."""
    return project.findings(sid) if refuse is None else project.findings(sid, refuse=refuse)


def session_graph(project, sid, refuse=None):
    """The sealed session state graph (snapshot dict); rows that cannot be placed are left out
    and reported through ``refuse``."""
    return (project.session_graph(sid) if refuse is None
            else project.session_graph(sid, refuse=refuse))


def store_root(project):
    """The saturn_pub LocalStore directory (read by ``serve.blobs``)."""
    return project.store.root


def tensor_info(project, ref):
    """Shape/dtype of a stored tensor (no bytes materialised)."""
    return project.tensor_info(ref)


def db(project):
    """Read-only sqlite connection context (serve only runs SELECTs through it)."""
    return project._db()


def lifecycle_reads(body):
    """Attention reads recorded in a symbol lifecycle body, as sealed read dicts."""
    return _lifecycle_reads(body)

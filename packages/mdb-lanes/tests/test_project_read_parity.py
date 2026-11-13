"""Parity: the engine reader (``mdb.core.project_read``) returns exactly what the project writer's
own readers (the ``Project`` lane) return for a well-formed project, key order included, and the
session graph has the same snapshot identity. Skips without the serve fixture."""
from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path

import pytest

FIXTURE = os.environ.get("MDB_SERVE_FIXTURE")
pytestmark = pytest.mark.skipif(
    not FIXTURE or not (Path(FIXTURE) / "project.sqlite").is_file(),
    reason="set MDB_SERVE_FIXTURE to a project from tests/fixtures/serve_fixture_project.py")


def test_reader_matches_the_project_writers_readers_row_for_row(tmp_path):
    from mdb.core.project_read import ProjectReader, lifecycle_reads
    from mdb_lanes.project import Project
    from mdb_lanes.symbol_lifecycle import SymbolLifecycle
    from mdb_lanes.symbol_reads import ReadSource, reads_from_lifecycle

    root = tmp_path / "project"
    shutil.copytree(FIXTURE, root)
    reader, project = ProjectReader(root), Project(root, read_only=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sid in reader.sessions():
            pairs = {
                "cuts": ([reader.cut_descriptor(sid, c) for c in reader.cut_ids(sid)],
                         [project.cut_descriptor(sid, c) for c in project.cut_ids(sid)]),
                "receipts": (reader.receipts(sid), [r.to_dict() for r in project.receipts(sid)]),
                "symbols": (reader.symbols(sid), project.symbols(sid)),
                "slices": (reader.slices(sid),
                           [project.slice_record(sid, i) for i in project.slice_ids(sid)]),
                "lifecycles": (reader.symbol_lifecycles(sid), project.symbol_lifecycles(sid)),
                "findings": (reader.findings(sid), project.findings(sid)),
                "graph": (reader.session_graph(sid), project.graph(sid).seal().to_dict()),
            }
            for name, (mine, theirs) in pairs.items():
                assert json.dumps(mine) == json.dumps(theirs), (sid, name)
            for record in reader.symbol_lifecycles(sid):
                body = record["lifecycle"]
                lifecycle = SymbolLifecycle.from_dict(body)
                rows = sorted({int(u["metadata"].get("source_row", 0)) for u in body["uses"]
                               if "source_row" in (u.get("metadata") or {})})
                source = ReadSource(lifecycle.symbol, tuple(rows or [0]), "supplied",
                                    {"by": "project binding " + lifecycle.binding_fingerprint[:12]})
                theirs = [r.to_dict() for r in reads_from_lifecycle(lifecycle, source=source)]
                assert theirs and json.dumps(lifecycle_reads(body)) == json.dumps(theirs)

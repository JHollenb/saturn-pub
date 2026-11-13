"""mdb.core.project_read and mdb.core.graph_seal: the engine-set read-only project reader.

Fixture tests use the serve fixture project (``MDB_SERVE_FIXTURE``, built by
``tests/fixtures/serve_fixture_project.py``) and skip without it. Every test works on a copy.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from mdb.core import graph_seal as gs
from mdb.core.project_read import ProjectReader, lifecycle_digest, lifecycle_reads

FIXTURE = os.environ.get("MDB_SERVE_FIXTURE")
needs_fixture = pytest.mark.skipif(
    not FIXTURE or not (Path(FIXTURE) / "project.sqlite").is_file(),
    reason="set MDB_SERVE_FIXTURE to a project from tests/fixtures/serve_fixture_project.py")


@pytest.fixture()
def project(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(FIXTURE, root)
    return root


def _sql(root, statement, *args):
    db = sqlite3.connect(root / "project.sqlite")
    db.execute(statement, args)
    db.commit()
    db.close()


class Refusals(list):
    def __call__(self, kind, ident, error):
        self.append((kind, ident, type(error).__name__))


# --------------------------------------------------------------------------- graph_seal (pure)

def _small_graph():
    g = gs.GraphBuilder(metadata={"source": "test"})
    a = g.add_node(g.state_node("cas://sha256/" + "a" * 64, "a" * 64))
    b = g.add_node(g.state_node("cas://sha256/" + "b" * 64, "b" * 64))
    g.add_edge(a, b, "continues", evidence_=gs.evidence("observed"))
    g.add_action("cas://sha256/" + "c" * 64, inputs=[a], output=b, evidence_class="observed")
    note = g.add_node(g.source_node("note://1", labels=("note",), metadata={"x": [1, 2.5]}))
    g.add_edge(note, a, "annotates", graph="knowledge")
    return g


def test_a_sealed_graph_verifies_and_every_identity_is_content_addressed():
    snap = _small_graph().seal()
    out = gs.verify_snapshot(json.loads(json.dumps(snap)))
    assert out["snapshot_id"] == snap["snapshot_id"]
    assert [n["node_id"] for n in snap["nodes"]] == sorted(n["node_id"] for n in snap["nodes"])
    assert len(snap["events"]) == len(snap["nodes"]) + len(snap["edges"])
    assert snap["nodes"][0]["node_id"] == gs.digest(
        {k: v for k, v in snap["nodes"][0].items() if k not in ("node_id", "fingerprint")})


@pytest.mark.parametrize("tamper", ["node", "edge", "event", "snapshot_meta", "order"])
def test_a_tampered_snapshot_is_refused(tamper):
    snap = json.loads(json.dumps(_small_graph().seal()))
    if tamper == "node":
        snap["nodes"][0]["labels"] = ["forged"]
    elif tamper == "edge":
        snap["edges"][0]["relation"] = "supports"
    elif tamper == "event":
        snap["events"][1]["sequence"] = 7
    elif tamper == "snapshot_meta":
        snap["metadata"] = {"source": "elsewhere"}
    else:
        snap["events"] = list(reversed(snap["events"]))
    with pytest.raises(gs.GraphSealError):
        gs.verify_snapshot(snap)


def test_the_builder_refuses_a_second_parent_and_unknown_nodes():
    g = gs.GraphBuilder()
    a = g.add_node(g.state_node("cas://sha256/" + "a" * 64, "a" * 64))
    b = g.add_node(g.state_node("cas://sha256/" + "b" * 64, "b" * 64))
    c = g.add_node(g.state_node("cas://sha256/" + "c" * 64, "c" * 64))
    g.add_edge(a, c, "continues")
    g.add_edge(b, c, "continues")
    with pytest.raises(gs.GraphSealError, match="parent/continues"):
        g.seal()
    with pytest.raises(gs.GraphSealError, match="unknown"):
        g.add_edge(a, "f" * 64, "continues")
    with pytest.raises(gs.GraphSealError):
        gs.plain({"x": float("nan")})


# --------------------------------------------------------------------------- reader (fixture)

@needs_fixture
def test_opening_is_strictly_read_only(project):
    reader = ProjectReader(project)
    with pytest.raises(sqlite3.OperationalError):
        with reader._db() as db:
            db.execute("CREATE TABLE sneaky (x)")
    assert not list(project.glob("project.sqlite-*"))


@needs_fixture
def test_opening_refuses_a_missing_store_and_another_schema(project, tmp_path):
    _sql(project, "PRAGMA user_version = 2")
    with pytest.raises(ValueError, match="schema 3"):
        ProjectReader(project)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="existing index and store"):
        ProjectReader(empty)


@needs_fixture
def test_a_wal_project_in_an_unwritable_directory_is_refused_at_open(project):
    (project / "project.sqlite-wal").write_bytes(b"")
    project.chmod(0o555)
    try:
        with pytest.raises(ValueError, match="not writable"):
            ProjectReader(project)
    finally:
        project.chmod(0o755)


@needs_fixture
def test_every_row_of_the_fixture_verifies(project):
    reader = ProjectReader(project)
    assert reader.sessions() == ["qwen", "tiny"]
    with pytest.raises(ValueError, match="unknown project session"):
        reader.session("nope")
    counts = {sid: (len(reader.cut_ids(sid)), len(reader.receipts(sid)), len(reader.symbols(sid)),
                    len(reader.slices(sid)), len(reader.symbol_lifecycles(sid)),
                    len(reader.findings(sid)), len(reader.circuit_records(sid)))
              for sid in reader.sessions()}
    assert counts["tiny"][:2] == (18, 16) and counts["qwen"][:2] == (2, 1)
    assert counts["qwen"][4] == 1 and counts["tiny"][5] == 1
    for sid in reader.sessions():
        for cut in reader.cut_ids(sid):
            desc = reader.cut_descriptor(sid, cut)
            assert desc["fingerprint"] == cut and isinstance(desc["execution_point"], dict)


@needs_fixture
def test_the_session_graph_seals_and_verifies_from_its_own_snapshot(project):
    reader = ProjectReader(project)
    for sid in reader.sessions():
        snap = reader.session_graph(sid)
        assert gs.verify_snapshot(json.loads(json.dumps(snap)))["snapshot_id"] == \
            snap["snapshot_id"]
        kinds = {n["kind"] for n in snap["nodes"]}
        assert {"state", "action"} <= kinds


@needs_fixture
def test_a_record_the_graph_cannot_place_is_left_out_and_reported(project):
    from saturn_pub.values import digest

    reader = ProjectReader(project)
    before = reader.session_graph("tiny")
    db = sqlite3.connect(project / "project.sqlite")
    arm = json.loads(db.execute("select body from circuit_records where kind='comparison-arm' "
                                "limit 1").fetchone()[0])
    db.close()
    bad = {**arm, "focus_id": {"x": 1}}
    _sql(project, "insert into circuit_records values (?,?,?,?)", digest(bad), "tiny",
         bad["kind"], json.dumps(bad))
    with pytest.raises(ValueError, match="unresolved"):  # strict: the writer's graph raises too
        reader.session_graph("tiny")
    refused = Refusals()
    after = reader.session_graph("tiny", refuse=refused)
    assert after["snapshot_id"] == before["snapshot_id"]
    assert [(kind, ident) for kind, ident, _ in refused] == [("circuit-record", digest(bad))]


@needs_fixture
def test_a_record_referring_to_a_left_out_record_is_left_out_in_turn(project):
    from saturn_pub.values import digest

    db = sqlite3.connect(project / "project.sqlite")
    arm = json.loads(db.execute("select body from circuit_records where kind='comparison-arm' "
                                "limit 1").fetchone()[0])
    cmp_ = json.loads(db.execute("select body from circuit_records where kind='comparison' "
                                 "limit 1").fetchone()[0])
    db.close()
    bad_arm = {**arm, "focus_id": "0" * 64}
    dependent = {**cmp_, "arms": [digest(bad_arm)]}
    for body in (bad_arm, dependent):
        _sql(project, "insert into circuit_records values (?,?,?,?)", digest(body), "tiny",
             body["kind"], json.dumps(body))
    refused = Refusals()
    ProjectReader(project).session_graph("tiny", refuse=refused)
    assert sorted(r[1] for r in refused) == sorted([digest(bad_arm), digest(dependent)])


@needs_fixture
def test_a_broken_symbol_seal_is_refused_per_row_and_its_slice_leaves_the_graph(project):
    _sql(project, "update symbols set body = replace(body, 'Supplied', 'Xupplied') "
                  "where session_id='tiny'")
    reader = ProjectReader(project)
    with pytest.raises(ValueError):
        reader.symbols("tiny")
    refused = Refusals()
    assert reader.symbols("tiny", refuse=refused) == []
    assert [r[0] for r in refused] == ["symbol"]
    refused = Refusals()
    snap = reader.session_graph("tiny", refuse=refused)
    assert {r[0] for r in refused} == {"symbol", "slice"}
    assert not any(str(n["content"].get("source_ref", "")).startswith("symbol://")
                   for n in snap["nodes"])


@needs_fixture
def test_a_tampered_lifecycle_is_refused_and_reads_follow_the_verified_body(project):
    reader = ProjectReader(project)
    (record,) = reader.symbol_lifecycles("qwen")
    body = record["lifecycle"]
    assert body["fingerprint"] == lifecycle_digest(
        {k: v for k, v in body.items() if k != "fingerprint"})
    reads = lifecycle_reads(body)
    assert reads and len(reads) == len(body["uses"])
    for read in reads:
        assert read["schema"] == "mdb-symbol-read-v2" and read["causal"]["status"] == "unknown"
        assert read["lifecycle_fingerprint"] == body["fingerprint"]
    assert lifecycle_reads({**body, "metadata": {}}) == []
    _sql(project, "update symbol_lifecycles set body = replace(body, '\"source_row\":', "
                  "'\"source_row\":\"x\",\"_o\":') where session_id='qwen'")
    refused = Refusals()
    assert reader.symbol_lifecycles("qwen", refuse=refused) == []
    assert [r[0] for r in refused] == ["lifecycle"]


@needs_fixture
def test_an_annotation_object_that_changed_is_refused_per_row(project):
    reader = ProjectReader(project)
    (finding,) = reader.findings("tiny")
    objects = list((project / "sessions").rglob("objects/*.json"))
    assert objects
    for path in objects:
        path.write_bytes(path.read_bytes().replace(b"finding", b"fiNding"))
    refused = Refusals()
    assert reader.findings("tiny", refuse=refused) == []
    assert refused and {r[0] for r in refused} == {"finding"}
    assert finding["annotation_id"].startswith("state-annotation://sha256/")


def test_the_reader_imports_no_framework_and_no_lane_module():
    code = ("import sys, json; import mdb.core.project_read, mdb.core.graph_seal; "
            "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('torch', 'numpy', 'fastapi', 'transformers') or (m.startswith('mdb.') and "
            "m.split('.')[1] not in ('core',)))))")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                         check=True)
    assert json.loads(out.stdout) == []

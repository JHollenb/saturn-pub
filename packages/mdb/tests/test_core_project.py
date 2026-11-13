"""The engine-native project: schema-3 layout shared with the lane writer and ``ProjectReader``,
atomic creation, byte-stable read-only opens, session binding and the index duck interface."""
import hashlib
import os
import sqlite3

import pytest

from mdb.core import index as I
from mdb.core import project as project_module
from mdb.core import records as R
from mdb.core.pointers import RecordError
from mdb.core.project import Project, ProjectError
from mdb.core.project_read import ProjectReader

IDENTITY = "model-x"
EXECUTION = {"dtype": "bfloat16", "kernel": "eager"}
R.declare_kind("test-project", numeric=("margin",), required=("margin",))
LANE_TABLES = {"sessions", "cuts", "receipts", "imports", "analyses", "symbols", "slices"}


def _tree(root):
    """sha256 over every file's relative path and bytes (directories included by name)."""
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        h.update(str(path.relative_to(root)).encode())
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()


def _objects(project):
    with project.db() as db:
        return {r[0]: r[1] for r in db.execute("SELECT name, type FROM sqlite_master")}


def _record(sid="s1", **body):
    return R.seal_record(session_id=sid, kind="test-project", model_identity=IDENTITY,
                         execution=EXECUTION, authority="measured",
                         body={"margin": R.num(-4.75, "measured"), **body})


@pytest.fixture
def project(tmp_path):
    project = Project(tmp_path / "runs" / "t")
    project.open_session("s1", model_identity=IDENTITY, execution=EXECUTION)
    return project


def test_a_new_project_is_the_lane_schema_3_layout_and_the_reader_opens_it(tmp_path):
    project = Project(tmp_path / "runs" / "t")
    assert (project.root / "project.sqlite").is_file() and (project.root / "store").is_dir()
    objects = _objects(project)
    assert LANE_TABLES <= set(objects)
    for table in I.TABLES:   # the additive index tables wait for the first index write
        assert table not in objects
    with project.db() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"   # as the lane
    project.open_session("s1", model_identity=IDENTITY, execution=EXECUTION)
    reader = ProjectReader(project.root)
    assert reader.sessions() == ["s1"]
    assert reader.session("s1")["selected"] == "" and reader.cut_ids("s1") == []
    assert [p.name for p in tmp_path.joinpath("runs").iterdir()] == ["t"]   # no staging left


def test_creation_is_all_or_nothing(tmp_path, monkeypatch):
    def crash(*_):
        raise OSError("disk went away")

    monkeypatch.setattr(project_module.os, "rename", crash)
    with pytest.raises(OSError, match="disk went away"):
        Project(tmp_path / "t")
    assert list(tmp_path.iterdir()) == []   # neither the project nor a half-made staging dir


def test_losing_a_creation_race_opens_the_winners_project(tmp_path, monkeypatch):
    winner = Project(tmp_path / "won")
    winner.open_session("s1", model_identity=IDENTITY, execution=EXECUTION)
    rename = os.rename

    def race(source, target):   # another process renamed its project into place first
        rename(winner.root, target)
        rename(source, target)

    monkeypatch.setattr(project_module.os, "rename", race)
    loser = Project(tmp_path / "t")
    assert [s["session_id"] for s in loser.sessions()] == ["s1"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t"]


def test_an_existing_directory_that_is_not_a_project_is_refused(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "notes.txt").write_text("mine")
    with pytest.raises(ProjectError, match="not an mdb project"):
        Project(tmp_path / "t")
    with pytest.raises(ProjectError, match="not an mdb project"):
        Project(tmp_path / "absent", read_only=True)
    assert not (tmp_path / "absent").exists()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert Project(empty).root == empty.resolve()   # an empty directory is taken over


def test_read_only_open_and_reads_change_no_byte(project):
    I.put_record(project, _record())
    before = _tree(project.root)
    view = Project(project.root, read_only=True)
    assert [s["session_id"] for s in view.sessions()] == ["s1"]
    assert view.cut_ids("s1") == [] and len(I.list_records(view, "s1")) == 1
    assert I.get_cursor(view, "s1") is None and I.recipe_ids(view, "s1") == []
    for write in (lambda: I.put_record(view, _record(extra="x")),
                  lambda: view.open_session("s2", model_identity=IDENTITY, execution=EXECUTION)):
        with pytest.raises((RecordError, ProjectError), match="read-only|writable"):
            write()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        with view.db() as db:
            db.execute("DELETE FROM sessions")
    reader = ProjectReader(project.root)
    assert reader.sessions() == ["s1"]
    assert _tree(project.root) == before


def test_a_session_is_bound_to_its_model_and_execution(project):
    again = project.open_session("s1", model_identity=IDENTITY, execution=EXECUTION)
    assert again["revision"] == 0 and again["selected"] == ""
    with pytest.raises(ProjectError, match="bound to another model or execution"):
        project.open_session("s1", model_identity=IDENTITY, execution={**EXECUTION, "x": 1})
    with pytest.raises(ProjectError, match="not retained"):
        project.select("s1", "a" * 64)
    with pytest.raises(ValueError, match="unknown project session"):
        project.session("nope")


def test_the_index_uses_the_public_interface(project):
    record = _record()
    assert I.put_record(project, record) == record["id"]
    assert I.get_record(project, "s1", record["id"]) == record
    I.set_cursor(project, "s1", record["id"])
    assert I.get_cursor(project, "s1") == record["id"]
    identifier = I.put_recipe(project, "s1", {"schema": "x", "steps": []})
    assert I.recipe_ids(project, "s1") == [identifier]
    assert set(I.TABLES) <= set(_objects(project))
    reader = ProjectReader(project.root)   # the reader reads what the engine project wrote
    assert I.list_records(reader, "s1") == [record]


def test_a_record_over_64_kib_or_holding_an_array_is_refused_at_write(project):
    np = pytest.importorskip("numpy")
    with pytest.raises(RecordError, match="bytes"):
        _record(note=["x" * 4000] * 17)
    with pytest.raises(RecordError, match="array or tensor"):
        _record(values=np.zeros(3))
    forged = {**_record(), "body": {"margin": R.num(1.0, "measured"), "values": [0.5] * 3}}
    with pytest.raises(RecordError):
        I.put_record(project, forged)
    assert I.list_records(project, "s1") == []


def test_two_writers_on_one_project_serialise_and_never_corrupt(project):
    other = Project(project.root)   # a second open of the same directory
    with project.db() as db:
        db.execute("BEGIN IMMEDIATE")   # one writer holds the write lock
        locked = sqlite3.connect(project.root / "project.sqlite", timeout=0.05)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            locked.execute("BEGIN IMMEDIATE")
        locked.close()
        db.execute("UPDATE sessions SET debug_info=? WHERE id='s1'", ('{"a":1}',))
    other.open_session("s2", model_identity=IDENTITY, execution=EXECUTION)
    I.put_record(other, _record("s2"))
    I.put_record(project, _record())
    with project.db() as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert ProjectReader(project.root).sessions() == ["s1", "s2"]
    assert project.session("s1")["debug_info"] == '{"a":1}'

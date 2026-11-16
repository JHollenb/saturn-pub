"""Record custody and sinks (gap audit P8).

Undeclared, a sealed record is byte-identical to before (the goldens' ids stay); declared, every
record carries script and prereg sha256, library versions and the kernel path; a sink appends
every sealed record (any kind) and reads them back verified; MDB_SCRIPT declares from the
environment."""

import json

import pytest

from mdb.core import custody
from mdb.core.records import seal_record, verify_record

IDENTITY = {"family": "qwen2", "parameters_sha256": "0" * 64}


def _record_lines(path):
    """The record lines of a sink file (v3 sinks also hold one ``mdb_execution`` line per
    execution profile)."""
    return [line for line in path.read_text().splitlines() if '"mdb_execution"' not in line]


def _seal(body=None):
    return seal_record(session_id="s", kind="supplied-forecast", model_identity=IDENTITY,
                       execution={"device": "cpu"}, authority="supplied",
                       body=body or {"schema": "x", "target": {"row": 1}, "edit": {},
                                     "supplied": {"value": 1.0, "authority": "supplied"},
                                     "library": {"name": "t"}, "address_map": {},
                                     "sealed_at": {"where": "t1.logits"}})


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("MDB_SCRIPT", raising=False)
    monkeypatch.delenv("MDB_PREREG", raising=False)
    monkeypatch.delenv("MDB_RECORDS", raising=False)
    custody.clear()
    for sink in custody.sinks():
        sink.close()
    yield
    custody.clear()
    for sink in custody.sinks():
        sink.close()


def test_undeclared_is_unchanged_and_declared_carries_custody(tmp_path):
    plain = _seal()
    assert "custody" not in plain["body"]
    assert _seal()["id"] == plain["id"]
    script, prereg = tmp_path / "driver.py", tmp_path / "prereg.json"
    script.write_text("print(1)\n")
    prereg.write_text("{}\n")
    block = custody.declare(script=script, prereg=prereg, kernel_path="mdb-stepped")
    assert block["script_sha256"] == custody.file_sha256(script)
    record = _seal()
    held = record["body"]["custody"]
    assert held["script_sha256"] == custody.file_sha256(script)
    assert held["prereg_sha256"] == custody.file_sha256(prereg)
    assert held["kernel_path"] == "mdb-stepped" and "mdb" in held["libraries"]
    assert record["id"] != plain["id"]
    verify_record(record)
    own = _seal({**plain["body"], "custody": {"schema": "mine"}})
    assert own["body"]["custody"] == {"schema": "mine"}       # a record's own custody is kept


def test_sink_persists_every_record(tmp_path):
    path = tmp_path / "records.jsonl"
    with custody.records_to(path) as sink:
        a = _seal()
        custody.declare(note="second")
        b = _seal()
    _seal()                                                   # closed: not written
    lines = _record_lines(path)
    assert len(lines) == 2 and sink.count == 2
    assert len(path.read_text().splitlines()) == 3          # one execution line, written once
    back = sink.records()
    assert [r["id"] for r in back] == [a["id"], b["id"]]
    tampered = json.loads(lines[0])
    tampered["body"]["schema"] = "y"
    path.write_text(path.read_text().splitlines()[0] + "\n" + json.dumps(tampered) + "\n")
    with pytest.raises(Exception):
        sink.records()


def test_environment_declares(tmp_path, monkeypatch):
    script = tmp_path / "job.py"
    script.write_text("x = 1\n")
    sink_path = tmp_path / "env.jsonl"
    monkeypatch.setenv("MDB_SCRIPT", str(script))
    monkeypatch.setenv("MDB_RECORDS", str(sink_path))
    custody._STATE["env_read"] = False
    record = _seal()
    assert record["body"]["custody"]["script_sha256"] == custody.file_sha256(script)
    assert len(_record_lines(sink_path)) == 1


def test_sink_names_are_unique_per_run(tmp_path, monkeypatch):
    """Pre-release custody fix: shard jobs that opened one file name lost all but the last
    shard's records at collection. No path, a directory or ``{run}`` names the file per run."""
    monkeypatch.setattr(custody, "_RUN_TAG", [])
    monkeypatch.setenv("MRUN_JOB_ID", "job-abc123")
    tag = custody.run_tag()
    assert tag.startswith("job-abc123-") and tag.endswith(f"pid{__import__('os').getpid()}")
    assert custody.sink_path(tmp_path) == tmp_path / f"records-{tag}.jsonl"
    assert custody.sink_path(str(tmp_path / "shard-{run}.jsonl")) == tmp_path / f"shard-{tag}.jsonl"
    monkeypatch.chdir(tmp_path)
    with custody.records_to() as sink:
        a = _seal()
    assert sink.path == (tmp_path / f"records-{tag}.jsonl").absolute()
    assert [r["id"] for r in sink.records()] == [a["id"]]
    monkeypatch.setattr(custody, "_RUN_TAG", [])                 # another job: another file
    monkeypatch.setenv("MRUN_JOB_ID", "job-def456")
    with custody.records_to(tmp_path) as other:
        _seal()
    assert other.path != sink.path and len(sink.records()) == 1


def test_sink_refuses_a_file_another_run_wrote(tmp_path, monkeypatch):
    path = tmp_path / "records.jsonl"
    with custody.records_to(path):
        first = _seal()
    with custody.records_to(path):                               # this process's own file: ok
        second = _seal()
    foreign = tmp_path / "foreign.jsonl"
    foreign.write_text(path.read_text())                         # written by "another run"
    with pytest.raises(FileExistsError):
        custody.records_to(foreign)
    with custody.records_to(foreign, append=True) as sink:
        third = _seal()
    assert [r["id"] for r in sink.records()] == [first["id"], second["id"], third["id"]]
    monkeypatch.setenv("MDB_RECORDS", str(tmp_path / "env-foreign.jsonl"))
    (tmp_path / "env-foreign.jsonl").write_text(path.read_text())
    custody._STATE["env_read"] = False
    with pytest.raises(FileExistsError):
        _seal()
    custody._STATE["env_read"] = False
    monkeypatch.setenv("MDB_RECORDS_APPEND", "1")
    _seal()
    assert len(_record_lines(tmp_path / "env-foreign.jsonl")) == 3


def test_absent_payload_is_sealed_in_the_sink(tmp_path):
    """A pointer with no store (``state: absent``) still has its bytes in the sink file."""
    from mdb.core.bounded import put_json

    obj = [{"arm": "H", "probes": {"v": [0.5, 0.25]}}]
    bare = put_json(obj)
    with custody.records_to(tmp_path / "r.jsonl") as sink:
        pointer = put_json(obj)
        record = _seal()
    assert pointer == bare                                       # a sink never changes a pointer
    assert sink.load_json(pointer) == obj and sink.payload_count == 1
    assert [r["id"] for r in sink.records()] == [record["id"]]   # payload lines are not records
    with custody.records_to(tmp_path / "plain.jsonl", payloads=False) as plain:
        put_json(obj)
    assert plain.payloads() == {} and not plain.path.exists()
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    bad = json.loads(lines[0])
    bad["raw"] = bad["raw"].replace("0.5", "0.6")
    (tmp_path / "r.jsonl").write_text(json.dumps(bad) + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError):
        sink.load_json(pointer)


def test_records_to_same_path_twice_is_one_sink(tmp_path):
    path = tmp_path / "again.jsonl"
    with custody.records_to(path) as first:
        second = custody.records_to(path)
        assert second is first
        _seal()
    assert len(_record_lines(path)) == 1 and len(first.records()) == 1


def test_suffixless_missing_path_is_a_directory(tmp_path):
    new_dir = tmp_path / "new-dir"
    got = custody.sink_path(new_dir)
    assert got.parent == new_dir and got.name.startswith("records-")
    assert custody.sink_path(tmp_path / "plain.jsonl") == tmp_path / "plain.jsonl"
    with custody.records_to(new_dir) as sink:
        _seal()
    assert sink.path.parent == new_dir.absolute() and new_dir.is_dir()

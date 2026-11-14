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
    lines = path.read_text().splitlines()
    assert len(lines) == 2 and sink.count == 2
    back = sink.records()
    assert [r["id"] for r in back] == [a["id"], b["id"]]
    tampered = json.loads(lines[0])
    tampered["body"]["schema"] = "y"
    path.write_text(json.dumps(tampered) + "\n")
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
    assert len(sink_path.read_text().splitlines()) == 1

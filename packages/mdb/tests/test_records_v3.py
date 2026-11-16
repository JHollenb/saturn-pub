"""Record schema v3: round trip, column claims, v2 reading, migration, bounds, corrupt input.

v3 changes only the encoding and the id: the stored form keeps the execution profile by sha256
and a column claim's values as base64 bytes. Every logical field, so every number, is the same
after a round trip and after migrating a v2 record."""
import base64
import json
import struct
from pathlib import Path

import pytest

from mdb.core import custody
from mdb.core import index as I
from mdb.core import migrate as M
from mdb.core import records as R
from mdb.core.pointers import RecordError, seal_digest
from mdb.core.project import Project

IDENTITY = "model-x"
EXECUTION = {"dtype": "bfloat16", "kernel": "eager"}
R.declare_kind("test-v3", numeric=("margin",), required=("margin",), plain_numeric=("n",))
EDGE = [0.0, -0.0, 5e-324, -1.7976931348623157e308, 0.1, 1 / 3, 2.0 ** -1074, 123456.789e-30]
REPO = Path(__file__).resolve().parents[2]


def _bits(values):
    return [struct.pack("<d", v) for v in values]


def _body(**more):
    return {"margin": R.num(-4.75, "measured", unit="logit"), "n": 3,
            "grid": R.nums([[0.5, -0.25, 1e-300], [3.0, -0.0, 7.25]], "derived-exact",
                           proof="P15"),
            "edge": R.nums(EDGE, "measured"),
            "f32": R.nums([1.5, -2.25, 3.0517578125e-05], "measured", dtype="float32"),
            "ids": R.nums([[1, 2], [3, -4]], "supplied"), **more}


def _seal(body=None, execution=EXECUTION, session="s1"):
    return R.seal_record(session_id=session, kind="test-v3", model_identity=IDENTITY,
                         execution=execution, authority="measured",
                         body=_body() if body is None else body)


def _v2(body, session="s1"):
    """A v2 record as mdb 0.2.1 sealed it (canonical JSON of the record is the digest's form)."""
    envelope = {"schema": R.SCHEMA_V2, "session_id": session, "kind": "test-v3",
                "model_identity": IDENTITY, "execution": EXECUTION, "authority": "measured",
                "pointers": [], "body": body}
    digest = seal_digest(envelope)
    return {**envelope, "id": digest, "digest": digest}


V2_BODY = {"margin": R.num(-4.75, "measured", unit="logit"), "n": 3,
           "effects": {"a": R.num(0.1, "derived-exact", proof="P20"), "b": R.num(-0.0, "bound")},
           "note": "∑ text"}


# --------------------------------------------------------------------------- round trip

def test_v3_round_trip_is_bit_identical_through_the_stored_form():
    record = _seal()
    assert record["schema"] == R.SCHEMA == "mdb-record-v3" and set(record) == set(R.FIELDS)
    line = R.stored_form(record)
    stored = json.loads(line)
    assert set(stored) == set(R.STORED_FIELDS) and "execution" not in stored
    assert stored["execution_sha256"] == seal_digest(EXECUTION)
    assert stored["body"]["edge"]["values"] == base64.b64encode(
        b"".join(_bits(EDGE))).decode()
    assert stored["id"] == record["id"] == seal_digest(
        {k: v for k, v in stored.items() if k not in ("id", "digest")})
    back = R.load_record(line, execution=EXECUTION)
    assert back == record
    assert _bits(back["body"]["edge"]["values"]) == _bits(EDGE)          # -0.0, subnormals
    assert back["body"]["ids"]["values"] == [1, 2, 3, -4] and back["body"]["ids"]["dtype"] == "int64"
    assert R.verify_record(json.loads(json.dumps(record))) == record      # logical JSON too
    assert R.load_record(line.encode(), executions={stored["execution_sha256"]: EXECUTION}) == record


def test_column_values_read_back_as_arrays_of_their_shape_and_dtype():
    np = pytest.importorskip("numpy")
    grid = np.random.default_rng(0).standard_normal((28, 32))
    claim = R.nums(grid, "derived-exact", proof="P15")
    assert claim["shape"] == [28, 32] and claim["dtype"] == "float64"
    record = _seal(_body(grid=claim))
    out = R.nums_values(R.load_record(R.stored_form(record), execution=EXECUTION)["body"]["grid"])
    assert out.dtype == np.float64 and out.shape == (28, 32) and (out == grid).all()
    assert R.nums(np.arange(6, dtype=np.int32).reshape(2, 3), "measured")["dtype"] == "int64"
    assert R.nums(np.ones(3, dtype=np.float32), "measured")["dtype"] == "float32"


def test_a_grid_costs_about_eight_bytes_a_number_not_seventy():
    grid = [[(i * 32 + j) * 1e-3 + 1 / 7 for j in range(32)] for i in range(28)]
    as_column = len(R.stored_bytes(_seal(_body(grid=R.nums(grid, "measured")))))
    base = len(R.stored_bytes(_seal(_body(grid=R.nums([0.0], "measured")))))
    per_number = (as_column - base) / 896
    assert 10 <= per_number <= 11                  # base64 of 8 bytes: 10.7 characters
    one_num = len(json.dumps(R.num(grid[3][5], "measured"), separators=(",", ":")))
    assert one_num > 4 * per_number


def test_column_claims_are_checked():
    with pytest.raises(RecordError, match="finite"):
        R.nums([1.0, float("nan")], "measured")
    with pytest.raises(RecordError, match="authority"):
        R.nums([1.0], "guess")
    with pytest.raises(RecordError, match="float32"):
        R.nums([0.1], "measured", dtype="float32")
    with pytest.raises(RecordError, match="shape"):
        R.nums([1.0, 2.0, 3.0], "measured", shape=[2, 2])
    with pytest.raises(RecordError, match="integers"):
        R.nums([1.5], "measured", dtype="int64")
    with pytest.raises(RecordError, match="numbers"):
        R.nums([True, 1.0], "measured")
    bad = {**R.nums([1.0, 2.0], "measured"), "extra": 1}
    with pytest.raises(RecordError, match="column fields"):
        _seal(_body(grid=bad))
    with pytest.raises(RecordError, match="holding integers"):
        _seal(_body(grid={**R.nums([1.0, 2.0], "measured"), "values": [1, 2]}))
    unlabelled = {k: v for k, v in R.nums([1.0], "measured").items() if k != "authority"}
    with pytest.raises(RecordError):
        _seal(_body(grid=unlabelled))


# --------------------------------------------------------------------------- bounds

def test_a_column_over_the_inline_bound_needs_a_store_and_is_a_checked_blob(tmp_path):
    np = pytest.importorskip("numpy")
    from saturn_pub.store import LocalStore

    big = np.arange(R.MAX_NUMS_INLINE + 1, dtype=np.float64) / 3
    with pytest.raises(RecordError, match="pass store="):
        R.nums(big, "measured")
    store = LocalStore(tmp_path / "store")
    claim = R.nums(big, "measured", store=store)
    assert claim["values"]["kind"] == "tensor" and claim["values"]["shape"] == [len(big)]
    record = _seal(_body(grid=claim))
    assert len(R.stored_bytes(record)) < 4096
    assert (R.nums_values(record["body"]["grid"], store=store) == big).all()
    small = R.nums([1.0, 2.0], "measured", store=store, inline=False)
    assert small["values"]["kind"] == "tensor"
    path = tmp_path / "store" / "tensors" / f"{claim['values']['ref']}.safetensors"
    raw = bytearray(path.read_bytes())
    raw[-3] ^= 0x01
    path.write_bytes(bytes(raw))
    from mdb.core.pointers import Unresolved
    with pytest.raises((Unresolved, RecordError, ValueError)):
        R.nums_values(record["body"]["grid"], store=store)
    with pytest.raises(RecordError, match="inline"):
        R.check_nums({**R.nums([0.0], "measured"), "shape": [R.MAX_NUMS_INLINE + 1],
                      "values": [0.0] * (R.MAX_NUMS_INLINE + 1)})


def test_the_v3_bound_is_1_mib_and_v2_stays_64_kib():
    assert R.MAX_RECORD_BYTES_V3 == 1024 * 1024 and R.MAX_RECORD_ITEMS_V3 == 65536
    from mdb.core.pointers import MAX_RECORD_BYTES
    assert MAX_RECORD_BYTES == 64 * 1024
    # a summary-like body of 2000 labelled numbers: over both v2 bounds (64 KiB, 4096
    # values), well inside v3
    rows = {f"r{i:03d}": [R.num(i / 7 + j, "measured") for j in range(10)] for i in range(200)}
    record = _seal(_body(rows=rows))
    assert MAX_RECORD_BYTES < len(R.stored_bytes(record)) < R.MAX_RECORD_BYTES_V3
    assert R.verify_record(record) == record
    with pytest.raises(RecordError, match="bytes|values in one record"):
        R.verify_record(_v2({**V2_BODY, "rows": rows}))


# --------------------------------------------------------------------------- v2 records

def test_v2_records_still_verify_and_are_refused_when_changed():
    old = _v2(V2_BODY)
    assert R.verify_record(old) == old
    assert R.load_record(R.stored_form(old)) == old                       # no execution needed
    assert R.stored_form(old) == json.dumps(old, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False)
    changed = json.loads(json.dumps(old))
    changed["body"]["effects"]["a"]["value"] = 0.2
    with pytest.raises(RecordError, match="seal differs"):
        R.verify_record(changed)
    with_column = _v2({**V2_BODY, "grid": R.nums([1.0, 2.0], "measured")})
    with pytest.raises(RecordError):
        R.verify_record(with_column)                                      # v2 has no columns


def test_a_sink_holding_v2_and_v3_lines_reads_both(tmp_path):
    path = tmp_path / "mixed.jsonl"
    old = _v2(V2_BODY)
    path.write_text(R.stored_form(old) + "\n")
    with custody.records_to(path, append=True) as sink:
        new = _seal()
    assert [r["schema"] for r in sink.records()] == [R.SCHEMA_V2, R.SCHEMA]
    assert sink.records() == [old, new]
    assert sink.executions() == {seal_digest(EXECUTION): EXECUTION}
    assert M.verify_file(path)["schemas"] == {R.SCHEMA_V2: 1, R.SCHEMA: 1}


def test_the_index_reads_v2_rows_and_stores_v3_without_the_execution(tmp_path):
    import sqlite3

    project = Project(tmp_path / "p")
    project.open_session("s1", model_identity=IDENTITY, execution=EXECUTION)
    old, new = _v2(V2_BODY), _seal()
    assert I.put_record(project, old) == old["id"] and I.put_record(project, new) == new["id"]
    assert I.get_record(project, "s1", old["id"]) == old
    assert I.get_record(project, "s1", new["id"]) == new
    assert I.list_records(project, "s1") == [old, new]
    assert I.list_records(project, "s1", schema=R.SCHEMA_V2) == [old]
    with project.db() as db:
        rows = dict(db.execute("SELECT id, schema FROM records").fetchall())
        body = db.execute("SELECT body FROM records WHERE id=?", (new["id"],)).fetchone()[0]
    assert rows == {old["id"]: R.SCHEMA_V2, new["id"]: R.SCHEMA}
    assert body == R.stored_form(new) and '"execution_sha256"' in body
    other = _seal(execution={"dtype": "float32"})
    with pytest.raises(RecordError, match="another model or execution"):
        I.put_record(project, other)
    with sqlite3.connect(project.root / "project.sqlite") as db:     # a row bound elsewhere
        db.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                   (other["id"], "s1", "test-v3", R.SCHEMA, R.stored_form(other)))
    with pytest.raises(RecordError, match="execution"):
        I.get_record(project, "s1", other["id"])


def test_the_repository_v2_corpus_reads():
    found = sorted(REPO.glob("mdb-experiments/results/step-speedups-20261115/beast/ex04/*/records.jsonl"))
    if not found:
        pytest.skip("mdb-experiments results are not in this checkout")
    for path in found:
        report = M.verify_file(path)
        assert report["schemas"] == {R.SCHEMA_V2: report["records"]} and report["records"] >= 1


# --------------------------------------------------------------------------- migration

def test_migrate_record_keeps_every_field_but_the_id():
    old = _v2(V2_BODY)
    new = R.migrate_record(old)
    assert new["schema"] == R.SCHEMA and new["id"] != old["id"]
    assert R.same_content(old, new) == []
    for field in R.CONTENT_FIELDS:
        assert json.dumps(new[field], sort_keys=True) == json.dumps(old[field], sort_keys=True)
    assert _bits([new["body"]["effects"]["b"]["value"]]) == _bits([-0.0])
    assert R.migrate_record(new) == new                                    # v3 is unchanged
    assert R.migrate_record(old) == new                                    # deterministic
    with pytest.raises(RecordError, match="differ in body"):
        R.same_content(old, {**new, "body": {**new["body"], "n": 4}})


def test_migration_adds_no_custody_and_writes_no_sink(tmp_path):
    custody.declare(note="would be attached by seal_record")
    try:
        with custody.records_to(tmp_path / "open.jsonl") as sink:
            new = R.migrate_record(_v2(V2_BODY))
        assert "custody" not in new["body"] and sink.count == 0
    finally:
        custody.clear()


def _v2_file(path, records, payload=None):
    lines = [R.stored_form(r) for r in records]
    if payload is not None:
        lines.insert(1, payload)
    path.write_text("".join(line + "\n" for line in lines))
    return path


PAYLOAD = json.dumps({"mdb_payload": "mdb-payload-v1", "sha256": "ab" * 32, "bytes": 2,
                      "raw": "[]"}, sort_keys=True, separators=(",", ":"))


def test_migrate_file_writes_a_new_file_and_a_verified_map(tmp_path):
    a, b = _v2(V2_BODY), _v2({**V2_BODY, "note": "second"})
    source = _v2_file(tmp_path / "records.jsonl", [a, b], PAYLOAD)
    before = source.read_bytes()
    report = M.migrate_file(source)
    assert source.read_bytes() == before                                  # never in place
    target = tmp_path / "records.v3.jsonl"
    assert report["target"] == str(target) and report["records"] == report["verified"] == 2
    assert report["migrated"] == 2 and report["already_v3"] == 0 and report["payloads"] == 1
    saved = json.loads((tmp_path / "records.v3.map.json").read_text())
    assert saved["map"] == report["map"] and set(saved["map"]) == {a["id"], b["id"]}
    assert saved["source_sha256"] and saved["target_sha256"]
    back = M.read_file(target)
    assert [saved["map"][a["id"]], saved["map"][b["id"]]] == [r["id"] for r in back]
    for old, new in zip((a, b), back):
        assert R.same_content(old, new) == [] and new["schema"] == R.SCHEMA
    lines = target.read_text().splitlines()
    assert PAYLOAD in lines and sum('"mdb_execution"' in line for line in lines) == 1
    with pytest.raises(FileExistsError):
        M.migrate_file(source)
    again = M.migrate_file(target, tmp_path / "again.v3.jsonl")            # v3 maps to itself
    assert again["already_v3"] == 2 and all(k == v for k, v in again["map"].items())


def test_migrate_refuses_a_corrupt_file_and_writes_nothing(tmp_path):
    good, bad = _v2(V2_BODY), _v2({**V2_BODY, "note": "x"})
    bad["body"]["note"] = "tampered"
    source = _v2_file(tmp_path / "r.jsonl", [good, bad])
    with pytest.raises(RecordError, match="seal differs"):
        M.migrate_file(source)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["r.jsonl"]
    truncated = tmp_path / "t.jsonl"
    truncated.write_text(R.stored_form(good) + "\n" + R.stored_form(good)[:-40] + "\n")
    with pytest.raises(RecordError, match="truncated or corrupt"):
        M.migrate_file(truncated)
    other = tmp_path / "items.jsonl"
    other.write_text('{"name": "a", "tokens": [1, 2]}\n')
    assert not M.is_record_file(other)


def test_the_cli_migrates_a_directory_and_verifies(tmp_path, capsys):
    from mdb.cli import main

    root = tmp_path / "results"
    (root / "job").mkdir(parents=True)
    _v2_file(root / "job" / "records.jsonl", [_v2(V2_BODY)])
    (root / "job" / "items.jsonl").write_text('{"name": "a"}\n')          # not a record file
    assert main(["records", "migrate", str(root), "--out", str(tmp_path / "v3")]) == 0
    target = tmp_path / "v3" / "job" / "records.v3.jsonl"
    assert target.is_file() and (tmp_path / "v3" / "job" / "records.v3.map.json").is_file()
    assert not (tmp_path / "v3" / "job" / "items.v3.jsonl").exists()
    assert main(["records", "verify", str(target)]) == 0
    assert "1 records verified" in capsys.readouterr().out
    assert main(["records", "migrate", str(root), "--out", str(tmp_path / "v3")]) == 1  # exists
    assert main(["records"]) == 2


# --------------------------------------------------------------------------- corrupt input

def test_corrupt_or_truncated_stored_records_are_refused():
    record = _seal()
    stored = json.loads(R.stored_form(record))
    with pytest.raises(RecordError, match="truncated or corrupt"):
        R.load_record(R.stored_form(record)[:-25], execution=EXECUTION)
    with pytest.raises(RecordError, match="not UTF-8"):
        R.load_record(b"\xff\xfe", execution=EXECUTION)
    raw = base64.b64decode(stored["body"]["edge"]["values"])
    short = json.loads(json.dumps(stored))
    short["body"]["edge"]["values"] = base64.b64encode(raw[:-8]).decode()
    with pytest.raises(RecordError, match="truncated or corrupt"):
        R.load_record(short, execution=EXECUTION)
    flipped = json.loads(json.dumps(stored))
    flipped["body"]["edge"]["values"] = base64.b64encode(
        raw[:8] + bytes([raw[8] ^ 1]) + raw[9:]).decode()
    with pytest.raises(RecordError, match="seal differs"):
        R.load_record(flipped, execution=EXECUTION)
    junk = json.loads(json.dumps(stored))
    junk["body"]["edge"]["values"] = "not base64!"
    with pytest.raises(RecordError, match="base64"):
        R.load_record(junk, execution=EXECUTION)
    nan = json.loads(json.dumps(stored))
    nan["body"]["edge"]["values"] = base64.b64encode(
        struct.pack("<d", float("nan")) + raw[8:]).decode()
    with pytest.raises(RecordError, match="finite"):
        R.load_record(nan, execution=EXECUTION)
    as_list = json.loads(json.dumps(stored))
    as_list["body"]["edge"]["values"] = EDGE
    with pytest.raises(RecordError):
        R.load_record(as_list, execution=EXECUTION)
    with pytest.raises(RecordError, match="execution_sha256"):
        R.load_record(stored, execution={"dtype": "float32"})
    with pytest.raises(RecordError, match="no execution profile"):
        R.load_record(stored)
    with pytest.raises(RecordError, match="load_record"):
        R.verify_record(stored)
    changed = json.loads(json.dumps(stored))
    changed["execution_sha256"] = seal_digest({"dtype": "float32"})
    with pytest.raises(RecordError, match="seal differs"):
        R.load_record(changed, execution={"dtype": "float32"})


def test_a_sink_with_a_tampered_execution_line_is_refused(tmp_path):
    with custody.records_to(tmp_path / "s.jsonl") as sink:
        _seal()
    lines = sink.path.read_text().splitlines()
    line = json.loads(lines[0])
    assert "mdb_execution" in line
    line["execution"]["dtype"] = "float32"
    sink.path.write_text(json.dumps(line) + "\n" + lines[1] + "\n")
    with pytest.raises(ValueError, match="differs from its sha256"):
        sink.records()
    sink.path.write_text(lines[1] + "\n")                      # the profile line is missing
    with pytest.raises(RecordError, match="no execution profile"):
        sink.records()


# --------------------------------------------------------------------------- audit fixes

def _sink_dir(tmp_path):
    """good.jsonl, tampered.jsonl (last record's schema and body changed), truncated.jsonl."""
    for name in ("good", "tampered", "truncated"):
        with custody.records_to(tmp_path / f"{name}.jsonl"):
            _seal(session="a")
            _seal(session="b")
    lines = (tmp_path / "tampered.jsonl").read_text().splitlines()
    bad = json.loads(lines[-1])
    bad["schema"] = "mdb-record-v9"
    bad["body"]["grid"] = "x"
    (tmp_path / "tampered.jsonl").write_text("\n".join(lines[:-1] + [json.dumps(bad)]) + "\n")
    text = (tmp_path / "truncated.jsonl").read_text()
    (tmp_path / "truncated.jsonl").write_text(text[:-40] + "\n")


def test_verify_fails_closed_on_tampered_and_truncated_files(tmp_path, capsys):
    from mdb.cli import main

    _sink_dir(tmp_path)
    for name in ("tampered", "truncated"):
        assert main(["records", "verify", str(tmp_path / f"{name}.jsonl")]) == 1
    capsys.readouterr()
    assert main(["records", "verify", str(tmp_path / "good.jsonl")]) == 0
    assert main(["records", "verify", str(tmp_path)]) == 1                 # mixed directory
    out = capsys.readouterr()
    assert "tampered.jsonl: refused" in out.err and "truncated.jsonl: refused" in out.err
    assert "good.jsonl: 2 records verified" in out.out
    with pytest.raises(RecordError):
        M.verify_file(tmp_path / "tampered.jsonl")
    with pytest.raises(RecordError):
        custody.Sink(tmp_path / "tampered.jsonl").records()


def test_a_directory_verify_includes_v3_twins_and_skips_unrelated_files(tmp_path, capsys):
    from mdb.cli import main

    with custody.records_to(tmp_path / "r.jsonl"):
        _seal()
    (tmp_path / "items.jsonl").write_text('{"name": "a"}\n')
    assert main(["records", "migrate", str(tmp_path)]) == 0
    (tmp_path / "r.v3.jsonl").write_text((tmp_path / "r.v3.jsonl").read_text()[:-30] + "\n")
    capsys.readouterr()
    assert main(["records", "verify", str(tmp_path)]) == 1
    assert "r.v3.jsonl: refused" in capsys.readouterr().err


def test_duplicate_keys_and_oversize_lines_are_refused_before_they_verify(tmp_path):
    record = _seal()
    stored = R.stored_form(record)
    execution = EXECUTION
    assert R.load_record(stored, execution=execution) == record
    duplicated = stored[:-1] + ',"session_id":"s1"}'
    with pytest.raises(RecordError, match="duplicate key"):
        R.load_record(duplicated, execution=execution)
    with pytest.raises(RecordError, match="duplicate key"):
        R.load_record(duplicated.encode(), execution=execution)
    huge = '{"schema":"mdb-record-v3","pad":"' + "x" * (R.MAX_LINE_BYTES + 1) + '"}'
    with pytest.raises(RecordError, match="refused before parsing"):
        R.load_record(huge)
    with pytest.raises(RecordError, match="refused before parsing"):
        R.load_record(huge.encode())
    path = tmp_path / "s.jsonl"
    path.write_text(huge + "\n")
    with pytest.raises(RecordError, match="refused before parsing"):
        custody.Sink(path).records()
    with pytest.raises(RecordError, match="refused before parsing"):
        M.read_file(path)
    path.write_text(stored[:-1] + ',"id":"0"}\n')
    with pytest.raises(RecordError, match="duplicate key"):
        custody.Sink(path).records()
    project = Project(tmp_path / "p")
    project.open_session("s1", model_identity=IDENTITY, execution=EXECUTION)
    I.put_record(project, record)
    for body, match in ((duplicated, "duplicate key"), (huge, "refused before parsing")):
        with project.db() as db:
            db.execute("UPDATE records SET body=?", (body,))
        with pytest.raises(RecordError, match=match):
            I.get_record(project, "s1", record["id"])


def test_a_profile_that_differs_only_by_1_against_1_0_is_not_the_sessions(tmp_path):
    project = Project(tmp_path / "p")
    project.open_session("s1", model_identity=IDENTITY, execution={"dtype": "bf16", "guidance": 1})
    same = _seal(execution={"dtype": "bf16", "guidance": 1})
    assert I.put_record(project, same) == same["id"]
    other = _seal(execution={"dtype": "bf16", "guidance": 1.0})
    assert other["execution"] == same["execution"] and other["id"] != same["id"]
    with pytest.raises(RecordError, match="another model or execution"):
        I.put_record(project, other)
    with project.db() as db:           # a row bound elsewhere, written behind the API
        db.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                   (other["id"], "s1", "test-v3", R.SCHEMA, R.stored_form(other)))
    with pytest.raises(RecordError, match="execution"):
        I.get_record(project, "s1", other["id"])


def test_sink_and_file_readers_raise_record_error_and_nums_names_store_errors(tmp_path):
    with custody.records_to(tmp_path / "s.jsonl") as sink:
        _seal()
    lines = sink.path.read_text().splitlines()
    execution = json.loads(lines[0])
    for edit in (lambda e: e.update(execution={"dtype": "x"}),
                 lambda e: e.pop("execution"),
                 lambda e: e.update(execution={"v": float("nan")}),
                 lambda e: e.pop("sha256")):
        changed = json.loads(lines[0])
        edit(changed)
        sink.path.write_text(json.dumps(changed) + "\n" + lines[1] + "\n")
        with pytest.raises(RecordError):
            sink.records()
        with pytest.raises(RecordError):
            M.read_file(sink.path)
    assert execution["mdb_execution"]
    sink.path.write_text(lines[0] + "\n" + lines[1][:-5] + "\n")          # truncated last line
    with pytest.raises(RecordError, match="truncated or corrupt"):
        sink.records()
    sink.path.write_text("[1, 2]\n")
    with pytest.raises(RecordError):
        sink.records()

    class Refusing:
        def put_tensor(self, data):
            raise ValueError("shared storage refuses symbolic links")

    with pytest.raises(RecordError) as error:
        R.nums([0.5] * (R.MAX_NUMS_INLINE + 1), "measured", store=Refusing())
    assert "symbolic links" in str(error.value) and "cannot be sealed" not in str(error.value)


def test_migrate_never_writes_its_source_and_writes_the_map_before_the_target(tmp_path):
    source = _v2_file(tmp_path / "r.jsonl", [_v2(V2_BODY)])
    before = source.read_bytes()
    for target in (source, tmp_path / "r.jsonl.v3.jsonl.x", tmp_path / "other.jsonl"):
        with pytest.raises(RecordError):
            M.migrate_file(source, target, force=True)
    assert source.read_bytes() == before
    link = tmp_path / "link.v3.jsonl"
    link.symlink_to(source)
    with pytest.raises(RecordError, match="source"):
        M.migrate_file(source, link, force=True)
    assert source.read_bytes() == before
    report = M.migrate_file(source)
    target = Path(report["target"])
    assert json.loads(Path(report["map_path"]).read_text())["target_sha256"] == M._sha(target)
    assert not list(tmp_path.glob("*.partial"))
    target.unlink()                                  # a crash after the map: the rerun recovers
    assert M.migrate_file(source)["records"] == 1


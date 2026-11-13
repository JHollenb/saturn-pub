import hashlib
import json
import shutil
import sqlite3

import numpy as np
import pytest
from saturn_pub.values import canonical, digest

from mdb import circuit_investigation as ci
from mdb import native_replay_recipe
from mdb.core import index as I
from mdb.core import records as R
from mdb.core.pointers import RecordError, occurrence_pointer
from mdb.project import Project

IDENTITY = "model-x"
EXECUTION = {"dtype": "bfloat16", "kernel": "eager"}
R.declare_kind("test-index", numeric=("margin",), required=("margin",))

def _session_row(sid):
    return (sid, canonical(IDENTITY).decode(), canonical(EXECUTION).decode(), "0" * 64, "{}", 1)


def _add_session(project, sid):
    with project._db() as db:
        db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)", _session_row(sid))


def _legacy_body(sid, kind, **fields):
    return {"schema": ci.SCHEMA, "session_id": sid, "kind": kind, "model_identity": IDENTITY,
            "execution": EXECUTION, **fields}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(sid="s1", margin=-4.75, identity=IDENTITY, **extra):
    return R.seal_record(session_id=sid, kind="test-index", model_identity=identity,
                         execution=EXECUTION, authority="measured",
                         body={"margin": R.num(margin, "measured"), **extra})


def _objects(project):
    with project._db() as db:
        return {r[0]: r[1] for r in db.execute("SELECT name, type FROM sqlite_master")}


def _version(project):
    with project._db() as db:
        return db.execute("PRAGMA user_version").fetchone()[0]


def _new(root):
    project = Project(root)
    # A project that never saved an object has no store directory; read-only open needs one.
    (project.root / "store").mkdir(exist_ok=True)
    return project


@pytest.fixture
def project(tmp_path):
    project = _new(tmp_path / "p")
    _add_session(project, "s1")
    _add_session(project, "s2")
    return project


def test_fresh_project_is_schema_3_and_the_index_tables_wait_for_the_first_write(tmp_path):
    project = _new(tmp_path / "p")
    objects = _objects(project)
    assert _version(project) == 3
    for table in ("records", "cursor", "recipes", "occurrences", "circuit_records",
                  "circuit_focus"):
        assert table not in objects
    assert objects["sessions"] == "table" and objects["slices"] == "table"
    _add_session(project, "s1")
    # reads on a writable project create nothing
    assert I.list_records(project, "s1") == [] and I.get_cursor(project, "s1") is None
    assert I.find_occurrence(project, "s1", "t1.L0.v", layers=32) is None
    with pytest.raises(I.NotIndexed):
        I.get_recipe(project, "0" * 64)
    assert "records" not in _objects(project)
    view = Project(tmp_path / "p", read_only=True)
    assert _version(view) == 3 and I.list_records(view, "s1") == []
    assert "records" not in _objects(project)
    I.put_record(project, _record())
    objects = _objects(project)
    for table in ("records", "cursor", "recipes", "occurrences"):
        assert objects[table] == "table"
    assert "circuit_records" not in objects      # circuit_investigation's table, not ours
    assert _version(project) == 3
    Project(tmp_path / "p")                       # reopening is idempotent
    assert _version(Project(tmp_path / "p", read_only=True)) == 3


def test_records_put_get_list(project):
    first, second = _record(margin=-4.75), _record(margin=3.875)
    assert I.put_record(project, first) == first["id"]
    assert I.put_record(project, first) == first["id"]  # idempotent
    I.put_record(project, second)
    other = _record(sid="s2")
    I.put_record(project, other)
    assert I.get_record(project, "s1", first["id"]) == first
    assert I.list_records(project, "s1") == [first, second]
    assert I.list_records(project, "s1", kind="focus") == []
    assert I.list_records(project, "s2") == [other]
    with pytest.raises(I.NotIndexed):
        I.get_record(project, "s2", first["id"])
    with pytest.raises(ValueError, match="unknown project session"):
        I.list_records(project, "nope")


def test_record_must_match_its_session(project):
    with pytest.raises(RecordError, match="another model"):
        I.put_record(project, _record(identity="model-y"))
    with pytest.raises(ValueError, match="unknown project session"):
        I.put_record(project, _record(sid="nope"))
    tampered = {**_record(), "authority": "supplied"}
    with pytest.raises(RecordError, match="seal differs"):
        I.put_record(project, tampered)


def test_stored_record_tamper_detected(project):
    record = _record()
    I.put_record(project, record)
    changed = {**record, "body": {"margin": R.num(-1, "measured")}}
    with project._db() as db:
        db.execute("UPDATE records SET body=? WHERE id=?", (project._json(changed), record["id"]))
    with pytest.raises(RecordError, match="seal differs"):
        I.get_record(project, "s1", record["id"])


def test_a_deeply_nested_row_is_a_malformed_row_not_a_recursion_error(project):
    """Audit round 3: a body of '[' * 200000 escaped json.loads as a raw RecursionError."""
    record = _record()
    I.put_record(project, record)
    with project._db() as db:
        db.execute("UPDATE records SET body=? WHERE id=?", ("[" * 200000, record["id"]))
    with pytest.raises(RecordError, match="not readable JSON"):
        I.get_record(project, "s1", record["id"])
    with pytest.raises(RecordError, match="not readable JSON"):
        I.list_records(project, "s1", strict=True)
    (placeholder,) = I.list_records(project, "s1")
    assert placeholder["verified"] is False and placeholder["state"] == "refused"
    assert placeholder["id"] == record["id"]


def test_cursor(project):
    record = _record()
    assert I.get_cursor(project, "s1") is None
    with pytest.raises(I.NotIndexed):
        I.set_cursor(project, "s1", record["id"])
    I.put_record(project, record)
    I.set_cursor(project, "s1", record["id"])
    second = _record(margin=1)
    I.put_record(project, second)
    I.set_cursor(project, "s1", second["id"])
    assert I.get_cursor(project, "s1") == second["id"]
    assert I.get_cursor(project, "s2") is None


def test_read_only_refuses_writes(project):
    I.put_record(project, _record())
    view = Project(project.root, read_only=True)
    assert I.list_records(view, "s1") == [_record()]
    for call in (lambda: I.put_record(view, _record(margin=2)),
                 lambda: I.set_cursor(view, "s1", _record()["id"]),
                 lambda: I.put_recipe(view, "s1", {"a": 1}),
                 lambda: I.index_occurrence(view, "s1", "t1.L0.v", layers=32)):
        with pytest.raises(RecordError, match="writable"):
            call()


def _recipe():
    return native_replay_recipe.seal({
        "schema": native_replay_recipe.SCHEMA, "model_identity": IDENTITY, "execution": EXECUTION,
        "roots": [{"id": "r", "token_ids": [1, 2, 3]}],
        "events": [{"id": "e0", "session": "r", "kind": "continue", "steps": 2}]})


def test_recipes(project):
    recipe = _recipe()
    native_replay_recipe.validate(recipe)
    identifier = I.put_recipe(project, "s1", recipe)
    assert identifier == recipe["fingerprint"]
    unsealed = {k: v for k, v in recipe.items() if k != "fingerprint"}
    assert I.recipe_fingerprint(unsealed) == native_replay_recipe.fingerprint(unsealed)
    assert I.put_recipe(project, "s1", unsealed) == identifier
    assert I.put_recipe(project, "s2", recipe) == identifier  # content-addressed, stored once
    assert I.get_recipe(project, identifier) == recipe
    with pytest.raises(RecordError, match="seal differs"):
        I.put_recipe(project, "s1", {**recipe, "fingerprint": "0" * 64})
    with pytest.raises(I.NotIndexed):
        I.get_recipe(project, "0" * 64)
    with project._db() as db:
        db.execute("UPDATE recipes SET body=? WHERE id=?",
                   (project._json({**unsealed, "roots": [], "fingerprint": identifier}),
                    identifier))
    with pytest.raises(RecordError, match="seal differs"):
        I.get_recipe(project, identifier)


def test_occurrences(project):
    recipe = I.put_recipe(project, "s1", _recipe())
    blob = project.store.put_tensor(np.arange(4, dtype=np.float32))["ref"]
    assert I.find_occurrence(project, "s1", "t18.L20.v[kv0]", layers=32) is None
    row = I.index_occurrence(project, "s1", "t18.L20.v[kv=0]", recipe_id=recipe, event_id=0, layers=32)
    assert row == {"session_id": "s1", "address": "t18.L20.v[kv0]", "recipe_id": recipe,
                   "event_id": 0, "cut_id": None, "blob_id": None}
    filled = I.index_occurrence(project, "s1", "t18.L20.v[kv0]", blob_id=blob, cut_id="ab" * 32, layers=32)
    assert filled == {**row, "blob_id": blob, "cut_id": "ab" * 32}
    assert I.find_occurrence(project, "s1", "t18.L20.v[kv0]", layers=32) == filled
    assert I.find_occurrence(project, "s2", "t18.L20.v[kv0]", layers=32) is None
    with pytest.raises(RecordError, match="different cut_id"):
        I.index_occurrence(project, "s1", "t18.L20.v[kv0]", cut_id="cd" * 32, layers=32)
    with pytest.raises(RecordError, match="state cell"):
        I.index_occurrence(project, "s1", "cache.L20.v[row=18]", layers=32)
    with pytest.raises(RecordError, match="recipe .* is not indexed"):
        I.index_occurrence(project, "s1", "t1.L0.v", recipe_id="0" * 64, layers=32)
    with pytest.raises(RecordError, match="not in the project store"):
        I.index_occurrence(project, "s1", "t1.L0.v", blob_id="0" * 64, layers=32)
    with pytest.raises(RecordError, match="event index"):
        I.index_occurrence(project, "s1", "t1.L0.v", event_id=-1, layers=32)
    assert I.index_occurrence(project, "s1", "t1.L0.v~prefill", layers=32)["address"] == "t1.L0.v~prefill"
    custom = I.index_occurrence(project, "s1", "t2.L0.q_norm", ports=("q_norm",), layers=32)
    assert I.find_occurrence(project, "s1", "t2.L0.q_norm", ports=("q_norm",), layers=32) == custom
    with pytest.raises(RecordError, match="unknown port"):
        I.find_occurrence(project, "s1", "t2.L0.q_norm", layers=32)
    # the pointer and the index row name the same occurrence
    pointer = occurrence_pointer("t18.L20.v[kv=0]", program="stepped", shape=[4],
                                 dtype="float32", recipe=recipe, event=0)
    from mdb.core.address import format_address, from_json
    assert format_address(from_json(pointer["address"])) == filled["address"]




def test_occurrences_are_keyed_by_value_when_layers_is_given(project):
    blob = project.store.put_tensor(np.arange(4, dtype=np.float32))["ref"]
    row = I.index_occurrence(project, "s1", "t5.emb[d=3]", layers=4, cut_id="ab" * 32)
    assert row["address"] == "t5.L0.h[d=3]"
    for spelling in ("t5.emb[d=3]", "t5.L0.h[d=3]"):
        assert I.find_occurrence(project, "s1", spelling, layers=4) == row
    filled = I.index_occurrence(project, "s1", "t5.L0.h[d=3]", layers=4, blob_id=blob)
    assert filled == {**row, "blob_id": blob}
    last = I.index_occurrence(project, "s1", "t5.L1.h_out", layers=4)
    assert last["address"] == "t5.L2.h"
    assert I.find_occurrence(project, "s1", "t5.L2.h", layers=4) == last
    assert I.find_occurrence(project, "s1", "t5.L1.h_out", layers=4) == last
    tail = I.index_occurrence(project, "s1", "t5.L3.h_out", layers=4)
    assert tail["address"] == "t5.L3.h_out" == I.find_occurrence(
        project, "s1", "t5.L3.h_out", layers=4)["address"]
    with pytest.raises(RecordError, match="does not exist"):
        I.index_occurrence(project, "s1", "t5.L4.h", layers=4)
    # layers is a required keyword: there is no "key as written" path that splits one value
    with pytest.raises(TypeError, match="layers"):
        I.index_occurrence(project, "s1", "t5.emb[d=3]")
    with pytest.raises(TypeError, match="layers"):
        I.find_occurrence(project, "s1", "t5.emb[d=3]")
    with project._db() as db:
        assert db.execute("SELECT count(*) FROM occurrences").fetchone()[0] == 3


def test_c8_equal_values_share_one_occurrence_row(project):
    first = I.index_occurrence(project, "s1", 't9.logits["x"=5]', layers=4, cut_id="ab" * 32)
    assert first["address"] == "t9.logits[i=5]"
    for text in ('t9.logits["y"=5]', "t9.logits[i=5]"):
        assert I.find_occurrence(project, "s1", text, layers=4) == first
    q = I.index_occurrence(project, "s1", "t9.L1.q[h=1|2]", layers=4, cut_id="ab" * 32)
    assert q["address"] == "t9.L1.q[h=1:3]"
    assert I.find_occurrence(project, "s1", "t9.L1.q[h=1:3]", layers=4) == q
    with project._db() as db:
        assert db.execute("SELECT count(*) FROM occurrences").fetchone()[0] == 2


def test_a_consecutive_m_set_is_indexed_and_found(project):
    """Audit round 3: t3.M[i=14|15] is valid; its canonical form must be constructible."""
    first = I.index_occurrence(project, "s1", "t3.M[i=14|15]", layers=4, cut_id="ab" * 32)
    assert first["address"] == "t3.M[i=14|15]"
    assert I.find_occurrence(project, "s1", "t3.M[i=14|15]", layers=4) == first


def test_a_corrupt_blob_is_not_indexed(project):
    blob = project.store.put_tensor(np.arange(4, dtype=np.float32))["ref"]
    path = project.root / "store" / "tensors" / f"{blob}.safetensors"
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    assert project.store.has_tensor(blob) and not project.store.has_tensor(blob, verify=True)
    with pytest.raises(RecordError, match="not in the project store"):
        I.index_occurrence(project, "s1", "t1.L0.v", blob_id=blob, layers=32)
    project.store.put_tensor(np.arange(4, dtype=np.float32))   # re-putting repairs the blob
    assert I.index_occurrence(project, "s1", "t1.L0.v", blob_id=blob, layers=32)["blob_id"] == blob


def test_non_ascii_record_is_stored_in_the_bounded_form(project):
    from mdb.core.pointers import MAX_RECORD_BYTES

    R.declare_kind("test-wide")

    def sized(extra):
        def text(nbytes):
            return "∑" * (nbytes // 3) + "x" * (nbytes % 3)

        def build(last):
            return {"filler": {**{f"f{i:03d}": text(4000) for i in range(16)},
                               "f016": text(last)}}

        def make(last):
            return R.seal_record(session_id="s1", kind="test-wide", model_identity=IDENTITY,
                                 execution=EXECUTION, authority="measured", body=build(last))

        last = 1 + MAX_RECORD_BYTES + extra - len(R.stored_bytes(make(1)))
        return make(last)

    record = sized(0)
    assert len(R.stored_bytes(record)) == MAX_RECORD_BYTES
    I.put_record(project, record)
    with project._db() as db:
        stored = db.execute("SELECT body FROM records WHERE id=?", (record["id"],)).fetchone()[0]
    assert stored == R.stored_form(record) and len(stored.encode("utf-8")) == MAX_RECORD_BYTES
    assert I.get_record(project, "s1", record["id"]) == record
    with pytest.raises(RecordError, match=f"{MAX_RECORD_BYTES + 1} bytes"):
        sized(1)


def test_undeclared_record_kinds_cannot_be_indexed(project):
    envelope = {"schema": R.SCHEMA, "session_id": "s1", "kind": "never-declared",
                "model_identity": IDENTITY, "execution": EXECUTION, "authority": "measured",
                "pointers": [], "body": {"note": "text"}}
    from mdb.core.pointers import seal_digest

    digest_ = seal_digest(envelope)
    with pytest.raises(RecordError, match="not declared"):
        I.put_record(project, {**envelope, "id": digest_, "digest": digest_})
    assert "records" not in _objects(project)


# ------------------------------------------------------------------ schema-3 compatibility
#
# Other branches use the same project directories with the code at BASE (schema 3). A project
# touched by this code must stay fully usable by that code. The BASE code is extracted with
# `git archive` into a temporary directory and driven in separate processes.

import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

BASE = "bf9d31d13c4e5a03c1b2ed67231ef5ad3e942cd8"
REPO = Path(__file__).resolve().parents[3]
TESTS = Path(__file__).resolve().parent

BUILD = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from test_saturn_pub_adapter import opened
from mdb.circuit_investigation import finding

project, live, owner, debug = opened(Path(sys.argv[1]))
debug.checkpoint("parent")
project.focus("experiment", "port:value")
plan = project.plan_comparison(
    "experiment", [{"port": "value", "value": 2}, {"port": "value", "mode": "scale", "value": 3}])
output = project.run_comparison("experiment", plan["record_id"], live=live)
finding(project, "experiment", "ordered cooperation", record_id=output["record_id"])
live.session.continue_(3)
live.checkpoint()
sliced = project.slice("experiment", port="value")
node = sliced["graph"]["nodes"][0]["node_id"]
project.focus("experiment", slice_id=sliced["slice_id"], node_id=node)
owner.close()
print(project.root)
'''

# One persistent BASE process: commands arrive on stdin, one JSON answer per line.
BASE_SERVER = '''
import json
import sys

from mdb import circuit_investigation as ci
from mdb.project import Project

path, mode = sys.argv[1], sys.argv[2]
project = Project(path, read_only=(mode == "ro"))


def state():
    with project._db() as db:
        return {"user_version": db.execute("PRAGMA user_version").fetchone()[0],
                "tables": sorted(r[0] for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"))}


for line in sys.stdin:
    command = line.strip()
    if command == "quit":
        break
    out = {}
    try:
        if command == "reopen":
            project = Project(path, read_only=(mode == "ro"))
        if command == "save":
            out["saved"] = ci.save(project, "experiment", "note", text="from BASE")["record_id"]
        out["records"] = [r["record_id"] for r in ci.records(project, "experiment")]
        out["project_records"] = [r["record_id"] for r in project.circuit_records("experiment")]
        out["current"] = ci.current(project, "experiment")["record_id"]
        out.update(state())
    except Exception as error:  # noqa: BLE001
        out["error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out), flush=True)
'''


@pytest.fixture(scope="module")
def base_code(tmp_path_factory):
    root = tmp_path_factory.mktemp("base")
    try:
        archive = subprocess.run(
            ["git", "-C", str(REPO), "archive", BASE, "packages/mdb/src", "src/saturn_pub"],
            check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(root)], input=archive, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"base commit {BASE[:8]} is not available in this checkout")
    env = dict(os.environ, PYTHONPATH=f"{root / 'src'}:{root / 'packages/mdb/src'}",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


@pytest.fixture(scope="module")
def v3_project(base_code, tmp_path_factory):
    """A schema-3 project written by the base code: sessions, cuts, symbols, a slice, a circuit
    focus and eight circuit records."""
    parent = tmp_path_factory.mktemp("v3")
    done = subprocess.run([sys.executable, "-c", BUILD, str(parent), str(TESTS)], env=base_code,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    root = Path(done.stdout.strip().splitlines()[-1])
    with sqlite3.connect(root / "project.sqlite") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("SELECT count(*) FROM circuit_records").fetchone()[0] >= 8
        assert db.execute("SELECT count(*) FROM circuit_focus").fetchone()[0] >= 1
        assert db.execute("SELECT count(*) FROM symbols").fetchone()[0] >= 0
        assert db.execute("SELECT count(*) FROM cuts").fetchone()[0] >= 1
        assert db.execute("SELECT count(*) FROM slices").fetchone()[0] >= 1
    return root


def _copy(v3_project, tmp_path, name="copy"):
    destination = tmp_path / name
    shutil.copytree(v3_project, destination)
    return destination


def _tree(root):
    """{relative path: sha256} of every file under root."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _session_binding(project, sid="experiment"):
    row = project.session(sid)
    return json.loads(row["identity"]), json.loads(row["execution"])


def _v2(project, margin=-4.75, sid="experiment"):
    identity, execution = _session_binding(project, sid)
    return R.seal_record(session_id=sid, kind="test-index", model_identity=identity,
                         execution=execution, authority="measured",
                         body={"margin": R.num(margin, "measured")})


def _base_server(base_code, path, mode="rw"):
    process = subprocess.Popen([sys.executable, "-c", BASE_SERVER, str(path), mode],
                               env=base_code, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)

    def ask(command):
        process.stdin.write(command + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        assert line, process.stderr.read()[-2000:]
        return json.loads(line)

    ask.process = process
    return ask


def _stop(ask):
    ask.process.stdin.write("quit\n")
    ask.process.stdin.flush()
    ask.process.wait(timeout=60)


def test_v3_project_read_only_with_the_new_code_changes_no_byte(v3_project, tmp_path):
    root = _copy(v3_project, tmp_path)
    before = _tree(root)
    project = Project(root, read_only=True)
    legacy = ci.records(project, "experiment")
    assert len(legacy) >= 8
    listed = I.list_records(project, "experiment")
    assert [r["record_id"] for r in listed] == [r["record_id"] for r in legacy]
    assert listed == legacy
    assert {r["schema"] for r in listed} == {ci.SCHEMA}            # marked by their own schema
    assert I.list_records(project, "experiment", schema=R.SCHEMA) == []
    assert I.list_records(project, "experiment", schema=ci.SCHEMA) == legacy
    assert I.get_record(project, "experiment", legacy[0]["record_id"]) == legacy[0]
    assert I.get_cursor(project, "experiment") is None
    assert I.find_occurrence(project, "experiment", "t1.L0.v", layers=32) is None
    with pytest.raises(I.NotIndexed):
        I.get_recipe(project, "0" * 64)
    with pytest.raises(I.NotIndexed):
        I.get_record(project, "experiment", "0" * 64)
    assert project.focus("experiment")
    assert project.graph().verify().ok
    for call in (lambda: I.put_record(project, _v2(project)),
                 lambda: I.put_recipe(project, "experiment", {"a": 1}),
                 lambda: I.set_cursor(project, "experiment", legacy[0]["record_id"]),
                 lambda: I.ensure_tables(project)):
        with pytest.raises(RecordError, match="writable"):
            call()
    assert _tree(root) == before
    with sqlite3.connect(root / "project.sqlite") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    assert not names & {"records", "cursor", "recipes", "occurrences"}


def _write_index(project, record_margin=-4.75):
    legacy = ci.records(project, "experiment")
    first, second = _v2(project, record_margin), _v2(project, record_margin + 1)
    I.put_record(project, first)
    I.put_record(project, second)
    I.set_cursor(project, "experiment", first["id"])
    recipe = I.put_recipe(project, "experiment", _recipe())
    blob = project.store.put_tensor(np.arange(3, dtype=np.float32))["ref"]
    row = I.index_occurrence(project, "experiment", "t4.L1.v", recipe_id=recipe, event_id=0,
                             blob_id=blob, layers=32)
    return legacy, first, second, recipe, row


def test_new_code_writes_the_additive_index_into_a_copy_and_base_code_still_works(
        v3_project, base_code, tmp_path):
    original = _tree(v3_project)
    root = _copy(v3_project, tmp_path)
    project = Project(root)                      # new code, writable
    legacy, first, second, recipe, row = _write_index(project)
    with sqlite3.connect(root / "project.sqlite") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3       # never changed
        types = {r[0]: r[1] for r in db.execute("SELECT name, type FROM sqlite_master")}
        assert types["circuit_records"] == "table" and types["circuit_focus"] == "table"
        assert all(types[t] == "table" for t in ("records", "cursor", "recipes", "occurrences"))
        assert not {n for n, t in types.items() if t in ("view", "trigger")}
        assert db.execute("SELECT count(*) FROM circuit_records").fetchone()[0] == len(legacy)
        assert [r[0] for r in db.execute("SELECT id FROM records ORDER BY rowid")] == [
            first["id"], second["id"]]
    assert I.get_cursor(project, "experiment") == first["id"]
    assert _tree(v3_project) == original         # the source project was never touched
    ids = [r["record_id"] for r in legacy]
    # (iii) the BASE code opens the written copy read-only and writable
    ro = _base_server(base_code, root, "ro")
    answer = ro("list")
    assert answer.get("error") is None, answer
    assert answer["records"] == ids and answer["project_records"] == ids
    assert answer["user_version"] == 3
    _stop(ro)
    rw = _base_server(base_code, root, "rw")
    answer = rw("list")
    assert answer["records"] == ids and answer["user_version"] == 3, answer
    saved = rw("save")
    assert saved["saved"] and saved["records"] == [*ids, saved["saved"]] and saved["user_version"] == 3
    assert saved["current"] == ci.current(Project(root, read_only=True), "experiment")["record_id"]
    again = rw("reopen")
    assert again["records"] == [*ids, saved["saved"]] and again["user_version"] == 3
    _stop(rw)
    # (iv) the new code sees the BASE code's new record and its own v2 records
    reopened = Project(root)
    listed = I.list_records(reopened, "experiment")
    assert [r.get("record_id", r.get("id")) for r in listed] == [*ids, saved["saved"],
                                                                 first["id"], second["id"]]
    assert I.list_records(reopened, "experiment", schema=R.SCHEMA) == [first, second]
    assert len(ci.records(reopened, "experiment")) == len(ids) + 1
    assert I.get_cursor(reopened, "experiment") == first["id"]
    assert I.find_occurrence(reopened, "experiment", "t4.L1.v", layers=32) == row
    assert I.get_recipe(reopened, recipe)["fingerprint"] == recipe
    assert _tree(v3_project) == original


def test_a_base_process_holding_the_project_open_keeps_working(v3_project, base_code, tmp_path):
    root = _copy(v3_project, tmp_path)
    base = _base_server(base_code, root, "rw")
    ids = base("list")["records"]
    assert len(ids) >= 8
    new = Project(root)                          # the new code, a second process-equivalent
    assert base("list")["records"] == ids        # Project() alone added nothing
    first = _v2(new)
    I.put_record(new, first)                     # creates the four tables under the BASE process
    answer = base("list")
    assert answer["records"] == ids and answer["user_version"] == 3 and "error" not in answer
    saved = base("save")
    assert saved["records"] == [*ids, saved["saved"]] and "error" not in saved
    second = _v2(new, 1.5)
    I.put_record(new, second)
    I.set_cursor(new, "experiment", second["id"])
    assert base("list")["records"] == [*ids, saved["saved"]]
    assert [r.get("record_id", r.get("id")) for r in I.list_records(new, "experiment")] == [
        *ids, saved["saved"], first["id"], second["id"]]
    again = base("reopen")
    assert again["records"] == [*ids, saved["saved"]] and again["user_version"] == 3
    _stop(base)


def test_failure_while_creating_the_additive_tables_leaves_the_project_unchanged(
        v3_project, tmp_path, monkeypatch):
    root = _copy(v3_project, tmp_path)
    project = Project(root)
    record = _v2(project)

    def snapshot():
        with sqlite3.connect(root / "project.sqlite") as db:
            return (sorted(tuple(r) for r in db.execute(
                "SELECT type, name, sql FROM sqlite_master")),
                    db.execute("PRAGMA user_version").fetchone()[0],
                    list(db.iterdump()))

    before = snapshot()
    broken = (*I._TABLE_STATEMENTS[:3], "CREATE TABLE this is not sql (")
    monkeypatch.setattr(I, "_TABLE_STATEMENTS", broken)
    with pytest.raises(sqlite3.OperationalError):
        I.put_record(project, record)
    assert snapshot() == before                  # the first three CREATEs were rolled back too
    for call in (lambda: I.set_cursor(project, "experiment", ci.records(project, "experiment")[0]["record_id"]),
                 lambda: I.put_recipe(project, "experiment", {"a": 1}),
                 lambda: I.index_occurrence(project, "experiment", "t1.L0.v", layers=32)):
        with pytest.raises(sqlite3.OperationalError):
            call()
        assert snapshot() == before
    monkeypatch.undo()
    assert I.put_record(project, record) == record["id"]
    assert snapshot() != before


def test_every_table_is_created_in_one_transaction(v3_project, tmp_path):
    root = _copy(v3_project, tmp_path)
    project = Project(root)
    statements = []
    real = I._TABLE_STATEMENTS
    assert len(real) == 5 and all("IF NOT EXISTS" in s for s in real)
    I.ensure_tables(project)
    with project._db() as db:
        db.set_trace_callback(statements.append)
        I.migrate(db)                            # tables exist: no transaction, no statement
    assert not [s for s in statements if "CREATE" in s]


def test_legacy_rows_are_read_verified_and_tampering_is_refused(v3_project, tmp_path):
    root = _copy(v3_project, tmp_path)
    project = Project(root)
    legacy = ci.records(project, "experiment")
    victim = legacy[0]["record_id"]
    with project._db() as db:
        body = json.loads(db.execute("SELECT body FROM circuit_records WHERE id=?",
                                     (victim,)).fetchone()[0])
        db.execute("UPDATE circuit_records SET body=? WHERE id=?",
                   (canonical({**body, "note": "changed"}).decode(), victim))
    with pytest.raises(RecordError, match="seal differs"):
        I.get_record(project, "experiment", victim)
    with pytest.raises(RecordError, match="seal differs"):
        I.list_records(project, "experiment", strict=True)
    listed = I.list_records(project, "experiment")           # one bad row does not fail the list
    bad = [r for r in listed if not r.get("verified", True)]
    assert len(bad) == 1 and bad[0]["id"] == victim and bad[0]["state"] == "refused"
    assert "seal differs" in bad[0]["reason"] and bad[0]["schema"] == ci.SCHEMA
    assert len(listed) == len(legacy) and all(r.get("verified", True) for r in listed if r is not bad[0])


def test_unsupported_versions_are_still_refused_like_the_base_code(tmp_path):
    root = tmp_path / "p"
    _new(root)
    with sqlite3.connect(root / "project.sqlite") as db:
        db.execute("PRAGMA user_version = 4")
    with pytest.raises(ValueError, match="unsupported MDB project schema"):
        Project(root)
    with pytest.raises(ValueError, match="requires schema 3"):
        Project(root, read_only=True)


# --------------------------------------------------------------------------- C3: one bad row

def test_c3_list_records_survives_a_kind_this_process_never_declared(project):
    """Process A declares a kind and indexes a record; this process (kind undeclared) lists."""
    import subprocess
    import sys
    import textwrap

    mine = I.put_record(project, _record("s1", margin=1.5))
    script = textwrap.dedent(f"""
        import sys
        from mdb.core import index as I, records as R
        from mdb.project import Project
        R.declare_kind("other-process-kind", numeric=("m",), required=("m",))
        project = Project({str(project.root)!r})
        record = R.seal_record(session_id="s1", kind="other-process-kind",
                               model_identity={IDENTITY!r}, execution={EXECUTION!r},
                               authority="measured", body={{"m": R.num(2.5, "measured")}})
        print(I.put_record(project, record))
        """)
    import os
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    theirs = done.stdout.strip()
    third = I.put_record(project, _record("s1", margin=3.5))
    assert R.kind_spec("other-process-kind") is None

    listed = I.list_records(project, "s1")
    assert [r["id"] for r in listed] == [mine, theirs, third]       # positions kept
    assert listed[0]["body"]["margin"]["value"] == 1.5 and "verified" not in listed[2]
    assert listed[1] == {"id": theirs, "kind": "other-process-kind", "schema": R.SCHEMA,
                         "verified": False, "state": "unresolved",
                         "reason": "record kind 'other-process-kind' is not declared in this "
                                   "process"}
    assert [r["id"] for r in I.list_records(project, "s1", kind="other-process-kind")] == [theirs]
    with pytest.raises(RecordError, match="not declared"):
        I.get_record(project, "s1", theirs)                        # get stays strict
    with pytest.raises(RecordError, match="not declared"):
        I.list_records(project, "s1", strict=True)                 # strict restores raise-first
    # a tampered v2 row is "refused" with its reason, still in place
    with project._db() as db:
        row = json.loads(db.execute("SELECT body FROM records WHERE id=?", (mine,)).fetchone()[0])
        row["body"]["margin"]["value"] = 9.5
        db.execute("UPDATE records SET body=? WHERE id=?", (canonical(row).decode(), mine))
    listed = I.list_records(project, "s1")
    assert [r["id"] for r in listed] == [mine, theirs, third]
    assert listed[0]["state"] == "refused" and listed[0]["verified"] is False
    assert "seal differs" in listed[0]["reason"] and "verified" not in listed[2]
    with pytest.raises(RecordError, match="seal differs"):
        I.list_records(project, "s1", strict=True)
    # an unreadable row (huge integer text, C7) is refused, not a crash
    with project._db() as db:
        db.execute("UPDATE records SET body=? WHERE id=?",
                   ('{"x":' + "9" * 5000 + "}", third))
    listed = I.list_records(project, "s1")
    assert listed[2]["state"] == "refused" and listed[2]["verified"] is False

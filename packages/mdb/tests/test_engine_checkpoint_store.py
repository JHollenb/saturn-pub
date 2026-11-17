"""Checkpoint stores (#517, ``mdb.core.checkpoint_store``): a checkpoint written in one process
is published to a store (a directory, or an S3 bucket through boto3 against moto) and a run
from its record id in another process with an empty local project fetches it, blobs checked by
sha256; Branches run from it; blobs a store holds are not sent again; a changed blob or record
is refused and a missing blob is named by its sha256. Tiny Qwen3 fp32, CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import saturn_pub  # noqa: E402

import mdb  # noqa: E402
from mdb.analysis.branches import edit  # noqa: E402
from mdb.analysis.forecast import Margin  # noqa: E402
from mdb.core import checkpoint_store as stores  # noqa: E402
from mdb.core import settings  # noqa: E402
from mdb.engine import checkpoint as states  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402

CASE = ("qwen3", "fp32")
P = [1, 5, 9, 3, 7, 2, 11]
LAST = len(P) - 1
A, B = 4, 9
CUT = "t4.L1.p"


@pytest.fixture(scope="module")
def dbg():
    return open_tiny(*CASE)


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    """The cache project and S3 settings of this test only (no real endpoint or credential)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for name in ("AWS_ENDPOINT_URL", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def _source(dbg, name="src"):
    run = dbg.run(P, name=name, prefill=2)
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.25, "add")
    run.until(CUT)
    return run


def _arms(dbg, run):
    br = dbg.branches(run, Margin(LAST, A, B), at=f"t{LAST}.L1.h")
    br.add("zero", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
    br.add("half", [edit(f"t{LAST}.L1.h", 0.5, "scale")])
    return json.loads(json.dumps([[r["name"], r["dM"]] for r in br.run()]))


def _blobs(root):
    return sorted((Path(root) / "tensors").iterdir())


# ------------------------------------------------------------------ the store itself

def test_directory_store_checks_every_blob(tmp_path):
    store = stores.open_store(tmp_path / "s")
    data = b"bytes of a blob"
    import hashlib

    sha = hashlib.sha256(data).hexdigest()
    assert not store.has(sha)
    with pytest.raises(stores.MissingBlob, match=sha):
        store.get(sha)
    with pytest.raises(ValueError, match="differ from their sha256"):
        store.put("0" * 64, data)
    store.put(sha, data)
    assert store.has(sha) and store.get(sha) == data
    (tmp_path / "s" / "tensors" / f"{sha}.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="differs from its sha256"):
        store.get(sha)
    assert stores.open_store(f"file://{tmp_path}/s").uri == store.uri
    with pytest.raises(ValueError, match="no credentials"):
        stores.open_store("s3://key:secret@bucket/p")
    with pytest.raises(ValueError, match="unknown checkpoint store"):
        stores.open_store("gs://bucket/p")


def test_the_config_names_the_store(tmp_path, monkeypatch):
    monkeypatch.setenv("MDB_STORAGE_CHECKPOINT_STORE", f"file://{tmp_path}/env")
    settings.reset()
    assert stores.configured().uri == (tmp_path / "env").as_uri()
    assert stores.configured(f"file://{tmp_path}/arg").uri == (tmp_path / "arg").as_uri()
    monkeypatch.delenv("MDB_STORAGE_CHECKPOINT_STORE")
    user = Path(os.environ["XDG_CONFIG_HOME"]) / "mdb" / "config.toml"
    user.parent.mkdir(parents=True)
    user.write_text('[storage]\ncheckpoint_store = "s3://experiments/mdb-checkpoints"\n'
                    'endpoint = "http://127.0.0.1:1"\n')
    settings.reset()
    store = stores.configured()
    assert (store.uri, store.endpoint) == ("s3://experiments/mdb-checkpoints",
                                            "http://127.0.0.1:1")
    user.write_text('[storage]\naws_secret_access_key = "x"\n')
    settings.reset()
    with pytest.raises(settings.ConfigError):
        settings.current()


# ------------------------------------------------------------------ publish and fetch

def test_publish_dedups_and_the_record_names_the_store(dbg, tmp_path):
    remote = stores.open_store(tmp_path / "remote")
    pdbg = attach(tiny_model(*CASE), project=tmp_path / "a")
    run = _source(pdbg, "pub")
    first = states.save(run, project=pdbg.project, name="one", store=remote)
    sent = states._LAST_WRITE["published"]
    assert first["body"]["store"] == {"uri": remote.uri}
    assert sent["sent"] > 0 and sent["store"] == remote.uri
    # the same state again: every blob is in the store already (another job, same prefix)
    states.save(run, project=pdbg.project, name="two", store=remote)
    again = states._LAST_WRITE["published"]
    assert again["present"] >= sent["sent"] - 1 and again["sent"] <= 1   # the new document
    # one row later: only the new row's parts and the document are sent
    run.until(f"t{LAST}.L0.h")
    states.save(run, project=pdbg.project, name="three", store=remote)
    later = states._LAST_WRITE["published"]
    assert later["present"] > 0 and later["sent"] < sent["sent"] + later["present"]
    # a project save without a store names none (record ids unchanged)
    plain = states.save(run, project=pdbg.project, name="four")
    assert "store" not in plain["body"] and states._LAST_WRITE["published"] is None
    # a file holds its own tensors: no store
    with pytest.raises(ValueError, match="holds its own tensors"):
        states.save(run, tmp_path / "x.pt", store=remote)


def test_run_from_a_record_id_fetches_into_an_empty_project(dbg, tmp_path):
    remote = stores.open_store(tmp_path / "remote")
    src = _source(dbg, "fetch-src")
    record = src.dump(remote.uri)                       # through the local cache project
    assert record["body"]["store"] == {"uri": remote.uri}
    want = _arms(dbg, _source(dbg, "fetch-ref"))
    other = attach(tiny_model(*CASE), project=tmp_path / "empty")
    run = other.run_from(record["id"][:12], store=remote)
    assert run.where == CUT and run.core["exact"]
    assert _arms(other, run) == want
    # the project now holds the record and every blob: a run needs no store any more
    again = attach(tiny_model(*CASE), project=tmp_path / "empty").run_from(record["id"][:12])
    assert again.where == CUT
    # Run.checkpoint(durable=<store>) names the store; run_from of that result fetches from it
    durable = _source(dbg, "durable").checkpoint(durable=remote.uri)["durable"]
    assert durable == {"record": record["id"], "path": None, "store": remote.uri}
    third = attach(tiny_model(*CASE), project=tmp_path / "third")
    assert third.run_from({"durable": durable}).where == CUT


def test_a_changed_or_missing_blob_is_refused(dbg, tmp_path):
    remote = stores.open_store(tmp_path / "remote")
    record = _source(dbg, "tamper").dump(remote.uri)
    ckpt = states.load(record["id"], store=remote)
    part = ckpt.doc["parts"][-1]["ref"]
    blob = tmp_path / "remote" / "tensors" / f"{part}.safetensors"
    good = blob.read_bytes()
    fresh = iter(range(100))

    def other_project():
        return attach(tiny_model(*CASE), project=tmp_path / f"p{next(fresh)}")

    blob.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(ValueError, match=f"{part} in .* differs from its sha256"):
        other_project().run_from(record["id"], store=remote)
    blob.unlink()
    with pytest.raises(stores.MissingBlob, match=part):
        other_project().run_from(record["id"], store=remote)
    blob.write_bytes(good)
    # a changed record does not verify
    path = tmp_path / "remote" / "records" / f"{record['id']}.json"
    data = json.loads(path.read_text())
    data["body"]["where"] = "t0.start"
    path.write_text(json.dumps(data))
    with pytest.raises(Exception, match="seal|digest|differs|record"):
        other_project().run_from(record["id"], store=remote)
    path.unlink()
    with pytest.raises(ValueError, match="no checkpoint record"):
        other_project().run_from(record["id"], store=remote)


_CHILD = """
import json, os, sys
import torch
args = json.loads(sys.argv[1])
scope = args["scope"]   # a cut binds the numeric scope: match the parent's
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
sys.path.insert(0, args["tests"])
import test_engine_checkpoint_store as T
from mdb.engine.models import tiny_model
import mdb
dbg = mdb.attach(tiny_model(*T.CASE))          # no project: the cache project, the store from
run = dbg.run_from(args["record"][:10])        # MDB_STORAGE_CHECKPOINT_STORE
print(json.dumps({"where": run.where, "exact": run.core["exact"], "arms": T._arms(dbg, run),
                  "cache": sorted(os.listdir(os.environ["XDG_CACHE_HOME"]))}))
"""


def _child(dbg, record, env):
    paths = [str(Path(m.__file__).parents[1]) for m in (saturn_pub, mdb)]
    env = {**os.environ, **env,
           "PYTHONPATH": os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")])}
    payload = json.dumps({"record": record["id"], "tests": str(Path(__file__).parent),
                          "scope": dbg.adapter.execution["numeric_scope"]})
    result = subprocess.run([sys.executable, "-c", _CHILD, payload], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_another_process_runs_branches_from_a_published_record(dbg, tmp_path):
    remote = tmp_path / "remote"
    record = _source(dbg, "xp").dump(remote.as_uri())
    want = _arms(dbg, _source(dbg, "xp-ref"))
    child = _child(dbg, record, {"MDB_STORAGE_CHECKPOINT_STORE": remote.as_uri(),
                                 "XDG_CACHE_HOME": str(tmp_path / "child-cache")})
    assert child == {"where": CUT, "exact": True, "arms": want, "cache": ["mdb"]}


# ------------------------------------------------------------------ S3 (boto3 against moto)

@pytest.fixture
def s3(monkeypatch):
    pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    for name, value in (("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing"),
                        ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    with moto.mock_aws():
        import boto3

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="experiments")
        yield "s3://experiments/mdb-checkpoints"


def test_s3_store_round_trip_through_boto3(dbg, tmp_path, s3):
    store = stores.open_store(s3)
    assert isinstance(store, stores.S3Store) and store.uri == s3
    record = _source(dbg, "s3").dump(s3)
    assert record["body"]["store"] == {"uri": s3}
    keys = store._keys("")
    assert f"records/{record['id']}.json" in keys
    assert all(k.startswith(("records/", "tensors/")) for k in keys)
    assert stores.S3Store(s3, client=store.client).find_record(record["id"][:10])["id"] == \
        record["id"]
    want = _arms(dbg, _source(dbg, "s3-ref"))
    run = attach(tiny_model(*CASE), project=tmp_path / "empty").run_from(record["id"][:12],
                                                                          store=s3)
    assert _arms(dbg, run) == want
    # dedup: the same state again sends only what is new
    count = len(keys)
    _source(dbg, "s3").dump(s3)
    assert len(store._keys("tensors/")) == len([k for k in keys if k.startswith("tensors/")])
    assert len(store._keys("")) <= count + 2
    part = states.load(record["id"], store=s3).doc["parts"][0]["ref"]
    store.client.delete_object(Bucket="experiments",
                               Key=f"mdb-checkpoints/tensors/{part}.safetensors")
    with pytest.raises(stores.MissingBlob, match=part):
        attach(tiny_model(*CASE), project=tmp_path / "e2").run_from(record["id"], store=s3)


def test_another_process_reads_s3_over_http(dbg, tmp_path, monkeypatch):
    """The boto3 path over HTTP across processes: a moto server stands for MinIO."""
    pytest.importorskip("boto3")
    server = pytest.importorskip("moto.server")
    pytest.importorskip("flask")
    for name, value in (("AWS_ACCESS_KEY_ID", "testing"), ("AWS_SECRET_ACCESS_KEY", "testing"),
                        ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    moto = server.ThreadedMotoServer(port=0, verbose=False)
    moto.start()
    try:
        host, port = moto.get_host_and_port()
        endpoint = f"http://{host}:{port}"
        monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
        store = stores.open_store("s3://experiments/mdb-checkpoints")
        store.client.create_bucket(Bucket="experiments")
        record = _source(dbg, "http").dump(store)
        child = _child(dbg, record, {"MDB_STORAGE_CHECKPOINT_STORE": store.uri,
                                     "MDB_STORAGE_ENDPOINT": endpoint,
                                     "XDG_CACHE_HOME": str(tmp_path / "child-cache")})
        assert child["where"] == CUT and child["arms"] == _arms(dbg, _source(dbg, "http-ref"))
    finally:
        moto.stop()

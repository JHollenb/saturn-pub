"""Durable debugging on tiny models: ``attach(project=)``, ``checkpoint``/``restart`` across
processes, the replay refusals, ``export`` + ``mdb.resolve`` through all four tiers."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import saturn_pub  # noqa: E402

import mdb  # noqa: E402
from mdb.core import index as I  # noqa: E402
from mdb.core import recipe as recipes  # noqa: E402
from mdb.core.address import format_address, from_json  # noqa: E402
from mdb.core.errors import Refused  # noqa: E402
from mdb.core.project import Project  # noqa: E402
from mdb.core.project_read import ProjectReader  # noqa: E402
from mdb.engine.models import tiny_model  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402

TOKENS = [1, 5, 9, 3]
CASES = [("qwen2", "fp32"), ("qwen3", "bf16")]


def _tree(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        h.update(str(path.relative_to(root)).encode())
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()


def _edited(dbg):
    """Scalar, tensor and masked edits committed on layer 0, a scalar staged on layer 1."""
    run = dbg.run(TOKENS)
    run.until("L0.p")
    run.set("L0.a", 0.5, "scale")
    shape = run.port_shape("u")
    run.set("L0.u", torch.linspace(-0.5, 0.5, shape[-1]).reshape(shape), "add")
    hidden = run.port_shape("h_out")
    mask = torch.zeros(hidden, dtype=torch.bool)
    mask[..., :4] = True
    run.set("L0.h_out", torch.full(hidden, 1.5), "replace", mask=mask)
    run.finish()
    run.until("t3.L1.sg")
    run.set("L1.u", 0.125, "add")
    return run


def _walk(run, count=7):
    return [[run.step("port"), tensor_digest(run.read())] for _ in range(count)]


@pytest.fixture(scope="module", params=CASES, ids=lambda case: "-".join(case))
def saved(request, tmp_path_factory):
    family, dtype = request.param
    root = tmp_path_factory.mktemp("persist") / "runs" / "t"
    dbg = mdb.attach(tiny_model(family, dtype), project=root)
    run = _edited(dbg)
    info = dbg.checkpoint(run, "pre")
    expected = [info["cut"], run.where] + _walk(run)
    return {"family": family, "dtype": dtype, "root": root, "dbg": dbg, "info": info,
            "expected": expected}


_CHILD = """
import json, sys
import torch
args = json.loads(sys.argv[1])
scope = args["scope"]   # a cut binds the numeric scope: match the parent's before attaching
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
torch.backends.cuda.matmul.allow_tf32 = scope["tf32_matmul"]
import mdb
from mdb.engine.models import tiny_model
from mdb.engine.ports import tensor_digest
dbg = mdb.attach(tiny_model(args["family"], args["dtype"]), project=args["project"])
run = dbg.restart("pre")
trace = [run.checkpoint()["cut"], run.where]
for _ in range(7):
    run.step("port")
    trace.append([run.where, tensor_digest(run.read())])
print(json.dumps({"session": dbg.session_id, "trace": trace}))
"""


def test_a_checkpoint_restarts_in_a_fresh_process_at_the_same_cut(saved):
    dbg = saved["dbg"]
    assert saved["info"]["blobs"] == 3   # the tensor edit, its masked partner and the mask
    paths = [str(Path(module.__file__).parents[1]) for module in (saturn_pub, mdb)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    payload = json.dumps({"family": saved["family"], "dtype": saved["dtype"],
                          "project": str(saved["root"]),
                          "scope": dbg.adapter.execution["numeric_scope"]})
    result = subprocess.run([sys.executable, "-c", _CHILD, payload], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    child = json.loads(result.stdout.strip().splitlines()[-1])
    assert child["session"] == dbg.session_id
    assert child["trace"] == saved["expected"]


def test_restart_in_process_and_the_project_records(saved):
    dbg, info = saved["dbg"], saved["info"]
    run = dbg.restart("pre")
    assert [run.checkpoint()["cut"], run.where] + _walk(run) == saved["expected"]
    assert dbg.checkpoints() == [{"name": "pre", "where": "t3.L1.sg", "cut": info["cut"],
                                  "recipe": info["recipe"]}]
    reader = ProjectReader(saved["root"])
    assert dbg.session_id in reader.sessions()
    assert reader.session(dbg.session_id)["selected"] == info["cut"]
    assert reader.cut_ids(dbg.session_id) == [info["cut"]]
    assert reader.cut_descriptor(dbg.session_id, info["cut"])["fingerprint"] == info["cut"]
    records = I.list_records(reader, dbg.session_id)
    assert [(r["kind"], r["body"]["name"]) for r in records] == [("checkpoint", "pre")]
    assert I.get_cursor(reader, dbg.session_id) == info["record"]
    recipe = recipes.load(reader, info["recipe"])
    assert recipes.to_checkpoint(recipe)["cut"] == info["cut"]
    with pytest.raises(Refused, match="no durable checkpoint 'nope'"):
        dbg.restart("nope")


def test_replay_refusals_name_their_reason(saved, tmp_path):
    family, dtype, root = saved["family"], saved["dtype"], saved["root"]
    other = mdb.attach(tiny_model(family, dtype, seed=7), project=root)
    with pytest.raises(recipes.RecipeError, match="wrong model identity") as caught:
        other.restart("pre")
    assert caught.value.code == "model-identity"
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(2 if before != 2 else 3)
        if torch.get_num_threads() == before:
            pytest.skip("this torch build does not change its thread count")
        scoped = mdb.attach(tiny_model(family, dtype), project=root)
        with pytest.raises(recipes.RecipeError, match="torch_num_threads: recipe") as caught:
            scoped.restart("pre")
        assert caught.value.code == "numeric-scope"
    finally:
        torch.set_num_threads(before)
    recipe = recipes.load(saved["dbg"].project, saved["info"]["recipe"])
    forged = recipes.seal({**recipe, "cut": "0" * 64})
    with pytest.raises(recipes.RecipeError, match="fingerprint mismatch") as caught:
        saved["dbg"].replay(forged)
    assert caught.value.code == "fingerprint"
    victim = next(p for p in recipes.tensor_pointers(recipe) if p["dtype"] == "bool")["ref"]
    for code, damage in (("corrupt-blob", lambda p: p.write_bytes(p.read_bytes()[:-1] + b"\1")),
                         ("missing-blob", lambda p: p.unlink())):
        copy = tmp_path / code
        shutil.copytree(root, copy)
        damage(copy / "store" / "tensors" / f"{victim}.safetensors")
        copied = mdb.attach(tiny_model(family, dtype), project=copy)
        with pytest.raises(recipes.RecipeError, match=f"blob {victim}") as caught:
            copied.restart("pre")
        assert caught.value.code == code


def test_all_four_tiers_return_the_same_bytes(saved):
    dbg = saved["dbg"]
    run = dbg.restart("pre")
    run.until("L1.u")
    pointer = dbg.export(run, "t3.L1.u", keep=True)
    assert pointer["authority"] == "derived-exact"   # the staged add wrote it
    assert format_address(from_json(pointer["address"])) == f"{dbg.session_id}:t3.L1.u"
    answers = {}
    for tier in ("live", "blob", "cut", "replay"):
        result = mdb.resolve(pointer, session=dbg, tiers=(tier,))
        assert (result.state, result.tier) == ("resolved", tier), result.reason
        assert result.authority == "derived-exact"
        answers[tier] = result.value
    first = answers["live"]
    for value in answers.values():
        assert value.dtype == first.dtype and tensor_digest(value) == tensor_digest(first)
    assert tensor_digest(first) == pointer["value"]["sha256"]
    assert mdb.resolve(pointer, session=dbg).tier == "live"
    boundary = dbg.export(run, "t3.L0.h_out")   # the pending layer's boundary: every tier
    for tier in ("live", "cut", "replay"):
        result = mdb.resolve(boundary, session=dbg, tiers=(tier,))
        assert (result.state, result.sha256) == ("resolved", boundary["value"]["sha256"])
    committed = dbg.export(run, "t3.L0.f")   # a committed layer's port: its cut is not stored
    states = {tier: mdb.resolve(committed, session=dbg, tiers=(tier,)).state
              for tier in ("live", "cut", "replay")}
    assert states == {"live": "unresolved", "cut": "unresolved", "replay": "resolved"}
    assert tensor_digest(run.read("t3.L0.f")) == committed["value"]["sha256"]


def test_resolver_refusals_and_the_no_model_case(saved):
    dbg = saved["dbg"]
    run = dbg.restart("pre")
    pointer = dbg.export(run, "t3.L0.h")
    foreign = {**pointer, "address": {**pointer["address"], "origin": {"branch": "b"}}}
    result = mdb.resolve(foreign, session=dbg)
    assert result.state == "refused" and "origin b: is not the attached session" in result.reason
    assert result.value is None
    tampered = {**pointer, "value": {**pointer["value"], "sha256": "0" * 64}}
    result = mdb.resolve(tampered, session=dbg)
    assert result.state == "refused" and "digest mismatch" in result.reason
    result = mdb.resolve(pointer, project=saved["root"])   # no model, no blob
    assert result.state == "unresolved" and "attach the model" in result.needs
    assert mdb.resolve(pointer, session=dbg, tiers=("replay",)).authority == "measured"


def test_read_only_access_changes_no_byte(saved):
    dbg = saved["dbg"]
    run = dbg.restart("pre")
    run.until("L1.u")
    pointer = dbg.export(run, "t3.L1.u", keep=True)
    before = _tree(saved["root"])
    view = Project(saved["root"], read_only=True)
    assert dbg.session_id in [s["session_id"] for s in view.sessions()]
    assert recipes.checkpoints(view, dbg.session_id)[0]["body"]["name"] == "pre"
    assert mdb.resolve(pointer, project=saved["root"]).tier == "blob"
    reader = ProjectReader(saved["root"])
    assert reader.session_graph(dbg.session_id)
    assert _tree(saved["root"]) == before


def test_without_a_project_nothing_changes():
    dbg = mdb.attach(tiny_model("qwen2"))
    assert dbg.project is None and dbg.session_id is None
    run = dbg.run(TOKENS)
    with pytest.raises(Refused, match="needs a project"):
        dbg.checkpoint(run, "x")
    with pytest.raises(Refused, match="pass project= too"):
        mdb.attach(tiny_model("qwen2"), session="s")
    assert run.restore(run.checkpoint("x")) == "t3.L0.h"

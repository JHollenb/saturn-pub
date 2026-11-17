"""Checkpoints as content-addressed tensor pointers (#517, ``mdb.engine.checkpoint``) on tiny
Qwen2 (bf16) and Qwen3 (fp32), CPU.

Every verb on a run rehydrated from a checkpoint gives what it gives on the source run, in the
same process and in a fresh one (file and project record id); a cut inside a row continues bit
for bit at every port phase; Branches, Pool and each run from a checkpoint; checkpoints move
between the source run and a reloaded one (both ways, and across an earlier checkpoint of the
same lineage) and are refused off the lineage; durable checkpoints live in the project store
and are addressed by record id; tampering, another model and another dtype are refused (a
replica only when asked); legacy v1 core files still load."""

import copy
import io
import json
import os
import re
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
from mdb.engine import checkpoint as states  # noqa: E402
from mdb.engine import core_file  # noqa: E402
from mdb.engine.models import cast_weights, tiny_model  # noqa: E402
from mdb.engine.ports import tensor_digest  # noqa: E402
from mdb.ops.debugger import attach, open_tiny  # noqa: E402

P = [1, 5, 9, 3, 7, 2, 11]
LAST = len(P) - 1
A, B = 4, 9
CASES = [("qwen2", "bf16"), ("qwen3", "fp32")]
# the cut of the battery: inside row 4 (layer 0 committed, cursor at L1.p), after an edit at t3
CUT = "t4.L1.p"


@pytest.fixture(scope="module", params=CASES, ids=lambda c: "-".join(c))
def case(request):
    return request.param


@pytest.fixture(scope="module")
def dbg(case):
    return open_tiny(*case)


def _source(dbg, name="src"):
    run = dbg.run(P, name=name, prefill=2)
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.25, "add")
    run.until(CUT)
    return run


@pytest.fixture(scope="module")
def saved(dbg, tmp_path_factory):
    path = tmp_path_factory.mktemp("ck") / "cut.pt"
    record = _source(dbg, "saver").dump(path)
    return path, record


# ------------------------------------------------------------------ the verb battery

# record ids differ by custody (the script); a circuit's ledger number by what the debugger
# built before (the battery runs twice on one debugger)
_IDS = [(re.compile(r"record [0-9a-f]{16}"), "record <id>"),
        (re.compile(r"^circuit \d+ ", re.M), "circuit N "),
        (re.compile(r"\b[0-9a-f]{16,64}\b"), "<hex>")]


def _san(x):
    if isinstance(x, torch.Tensor):
        return tensor_digest(x)[:16]
    if isinstance(x, dict):
        return {str(k): _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_san(v) for v in x]
    if isinstance(x, str):
        for rx, rep in _IDS:
            x = rx.sub(rep, x)
        return x
    return x


def _shell(dbg, run, *lines):
    from mdb.repl.shell import Shell

    buf = io.StringIO()
    sh = Shell(dbg.ops(run), out=buf)
    for line in lines:
        buf.write(f">> {line}\n")
        sh.execute(line)
    return buf.getvalue()


def battery(dbg, make):
    """Every verb from the cut ``make(name)`` returns (``make`` builds a fresh run each time)."""
    out = {}
    run = make("v1")
    out["where"] = run.where
    out["steps"] = [run.step() for _ in range(3)]
    run.until(f"t{LAST}.logits")
    out["logits"] = run.read()
    run = make("v2")
    run.set("L1.m", 0.5, "add")
    run.until(f"t{LAST}.logits")
    out["edit"] = [run.read(), run.read("t4.L1.m")]
    run = make("v3")
    run.set("cache.L0.v[kv0,row=1]", 0.0)
    run.until(f"t{LAST}.logits")
    out["cell"] = run.read()
    out["forecast"] = _shell(dbg, make("v4"), f"until t{LAST}.L1.p", f"forecast M[{A},{B}]",
                             f"until t{LAST}.logits", "verify")
    out["whatif"] = _shell(dbg, make("v5"), f"until t{LAST}.L0.h",
                           f"whatif L1.A[h1] *= 0 --consumer M[{A},{B}]", "where")
    out["symbols"] = _shell(dbg, make("v6"), f"until t{LAST}.logits",
                            f"symbols consumer=M[{A},{B}] limit=3")
    out["reads"] = _shell(dbg, make("v7"), f"until t{LAST}.logits",
                          f"reads t5 --consumer M[{A},{B}] top=3")
    out["readout"] = _shell(dbg, make("v8"), f"until t{LAST}.logits", "print logits top=3",
                            f"explain M[{A},{B}]")
    out["circuit"] = _shell(dbg, make("v9"), f"until t{LAST}.logits",
                            f"circuit M[{A},{B}] top=3 depth=1")
    run = make("v10")
    br = dbg.branches(run, Margin(LAST, A, B), at=f"t{LAST}.L1.h")
    br.add("zero-m", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
    br.add("add-m", [edit(f"t{LAST}.L1.m", 0.25, "add")])
    br.add_null("self", "self", like="zero-m")
    out["branches"] = [[r["name"], r["dM"]] for r in br.run()]
    run = make("v11")
    run.until(f"t{LAST}.L1.p")
    point = run.checkpoint("pre")
    run.set(f"t{LAST}.L1.m", 0.0, "scale")
    run.until(f"t{LAST}.logits")
    edited = run.read()
    out["restore"] = [edited, run.restore(point), (run.until(f"t{LAST}.logits"), run.read())[1]]
    base = dbg.run(P, name="base", prefill=2)
    base.until(f"t{LAST}.logits")
    run = make("v12")
    run.hold("t@last.L>0.p", source=base)
    run.until(f"t{LAST}.logits")
    out["hold"] = [run.holds, run.read()]
    dbg.runs.clear()
    return _san(out)


def test_every_verb_on_a_rehydrated_run_matches_the_source(dbg, saved):
    path, _ = saved
    source = battery(dbg, lambda name: _source(dbg, name))
    loaded = battery(dbg, lambda name: dbg.run_from(path, name=name))
    assert loaded == source


_CHILD = """
import json, sys
import torch
args = json.loads(sys.argv[1])
scope = args["scope"]   # a cut binds the numeric scope: match the parent's
torch.set_num_threads(scope["torch_num_threads"])
torch.use_deterministic_algorithms(scope["deterministic_algorithms"])
torch.set_float32_matmul_precision(scope["float32_matmul_precision"])
sys.path.insert(0, args["tests"])
import test_engine_checkpoint as T
from mdb.engine.models import tiny_model
import mdb
# the battery runs on a debugger with no project (a project stores records and blobs, which
# changes the printed lines); the record id is resolved through the project's debugger
dbg = mdb.attach(tiny_model(*args["case"]))
durable = mdb.attach(tiny_model(*args["case"]), project=args["project"])
out = {"file": T.battery(dbg, lambda name: dbg.run_from(args["path"], name=name))}
by_id = durable.load_checkpoint(args["record"][:12])
out["record"] = T.battery(dbg, lambda name: dbg.run_from(by_id, name=name))
run = durable.restart("cut")
out["restart"] = [run.where, run.core["exact"]]
print(json.dumps(out))
"""


def test_every_verb_in_a_fresh_process(dbg, case, saved, tmp_path):
    path, _ = saved
    project = tmp_path / "proj"
    pdbg = attach(tiny_model(*case), project=project)
    info = pdbg.checkpoint(_source(pdbg, "p"), "cut")
    source = battery(dbg, lambda name: _source(dbg, name))
    paths = [str(Path(m.__file__).parents[1]) for m in (saturn_pub, mdb)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}
    payload = json.dumps({"case": list(case), "path": str(path), "project": str(project),
                          "record": info["state"], "tests": str(Path(__file__).parent),
                          "scope": dbg.adapter.execution["numeric_scope"]})
    result = subprocess.run([sys.executable, "-c", _CHILD, payload], env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    child = json.loads(result.stdout.strip().splitlines()[-1])
    assert child["file"] == source
    assert child["record"] == source
    assert child["restart"][:2] == [CUT, True]


# ------------------------------------------------------------------ cuts inside a row

STOPS = ["t4.start", "t4.emb", "t4.L0.p", "t4.L0.m", "t4.L0.h_out", "t4.L1.sg", "t4.fn",
         "t4.logits", "t4.sample"]


@pytest.mark.parametrize("stop", STOPS)
def test_a_cut_at_every_port_phase_continues_bit_for_bit(dbg, stop, tmp_path):
    src = _source(dbg, "phase")
    if src.parse(stop) < src.parse(src.where):
        src.rewind(stop)
    else:
        src.until(stop)
    if src.pending_layer is not None and src._port is not None:
        src.set(f"L{src.pending_layer}.h_out", 0.125, "add")       # staged at the cursor
    record = src.dump(tmp_path / "c.pt")
    assert record["body"]["where"] == src.where
    run = dbg.run_from(tmp_path / "c.pt", name="phase-r")
    assert run.where == src.where and run.edits == src.edits
    assert run._trail[-1].fingerprint == src._trail[-1].fingerprint
    src.until(f"t{LAST}.logits")
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())


def test_a_staged_cell_edit_at_the_cursor_is_kept(dbg, tmp_path):
    src = _source(dbg, "cellsrc")
    src.set("cache.L1.k[kv0,row=2]", 0.0)
    src.dump(tmp_path / "cell.pt")
    run = dbg.run_from(tmp_path / "cell.pt", name="cell-r")
    assert run.cells == src.cells
    src.until(f"t{LAST}.logits")
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())


def test_a_cut_before_an_edit_is_refused_never_truncated(dbg, tmp_path):
    src = _source(dbg, "trunc")            # edit at t3.L0, cursor in t4
    with pytest.raises(ValueError, match="after t3.start"):
        src.dump(tmp_path / "x.pt", row=3)
    src.dump(tmp_path / "x.pt", row=4)      # no edit after t4.start: fine
    src.set("L1.m", 1.0, "add")
    with pytest.raises(ValueError, match="staged"):
        src.dump(tmp_path / "y.pt", row=4)


def test_history_sets_how_far_back_the_reloaded_run_rewinds(dbg, tmp_path):
    src = _source(dbg, "hist")
    for history, earliest in (("cursor", "t4.L0.h_out"), ("row", "t4.start"),
                              ("root", "t2.start")):
        src.dump(tmp_path / f"{history}.pt", history=history)
        run = dbg.run_from(tmp_path / f"{history}.pt", name=f"h-{history}")
        assert run.root == earliest
        with pytest.raises(ValueError, match="rehydrated from a checkpoint"):
            run.rewind(f"t{run.first_row - 1}.L0.m" if run.first_row > 0 else "t0.start")
    run = dbg.run_from(tmp_path / "root.pt", name="h-root2")
    run.rewind("t2.L0.m")                   # before the t3 edit: the source's own trail
    clean = dbg.run(P, name="clean", prefill=2)
    clean.until("t3.L0.h")
    assert torch.equal(run.read("t2.L0.m"), clean.read("t2.L0.m"))
    run.until("t3.L0.h")
    run.set("t3.L0.m", 0.25, "add")
    run.until(f"t{LAST}.logits")
    src.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())


# ------------------------------------------------------------------ branches, pool, each

def test_branches_pool_and_each_from_a_checkpoint(dbg, saved):
    path, record = saved
    src, run = _source(dbg, "bp-src"), dbg.run_from(path, name="bp-r")
    rows = []
    for r in (src, run):
        br = dbg.branches(r, Margin(LAST, A, B), at=f"t{LAST}.L1.h")
        br.add("zero", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
        br.add_null("rnd", "random", like="zero", seed=0)
        rows.append([[x["name"], x["dM"], x.get("effect_floors")] for x in br.run()])
        assert torch.equal(br.reference.read(f"t{LAST}.logits"), br.base_logits)
    assert rows[0] == rows[1]
    assert br.reference.lineage["root"] == "prompt" and br.reference._origin is not None
    # pool: a port and a cache cell, from the file and from the loaded object
    src.until(f"t{LAST}.L1.h")
    pool = dbg.pool([{"id": "a", "core": str(path)},
                     {"id": "b", "from": dbg.load_checkpoint(path)}], f"t{LAST}.L1.h")
    assert torch.equal(pool.values[0], src.read(f"t{LAST}.L1.h"))
    assert torch.equal(pool.values[0], pool.values[1])
    assert pool.items[0]["origin"]["checkpoint"] == record["id"]
    cells = dbg.pool([{"id": "a", "core": str(path)}], "cache.L1.v[kv0,row=3]")
    assert torch.equal(cells.values[0], src.read_cache(1, "v")[:, :1, 3:4])
    # each: the item key (core / from) and the Python argument agree with a start of the prompt
    line = f"whatif L1.A[h1] *= 0 --consumer M[{A},{B}]"
    by_key = dbg.each([{"name": "x", "core": str(path)}], line).results
    by_arg = dbg.each([{"name": "x"}], line, core=str(path)).results
    plain = dbg.each([{"name": "x", "tokens": P}], line, prefill=2).results
    assert _san(by_key) == _san(by_arg)
    assert _san(by_key)["x"]["prediction"] != _san(plain)["x"]["prediction"]  # the t3 edit
    with pytest.raises(Exception, match="core or from"):
        dbg.each([{"name": "x", "core": str(path), "from": str(path)}], line)


# ------------------------------------------------------------------ lineage

def test_checkpoints_move_between_source_and_reloaded_runs(dbg, saved, tmp_path):
    path, _ = saved
    src, run = _source(dbg, "lin-src"), dbg.run_from(path, name="lin-r")
    assert run._path_manifest(0) == src._path_manifest(len(src._trail) - 1)[:1]
    src.until(f"t{LAST}.L0.p")
    src.set("L0.u", 0.5, "add")
    later = src.checkpoint("later")
    run.restore(later)                                        # source -> reloaded
    assert run.where == src.where and run.edits == src.edits
    run.until(f"t{LAST}.logits")
    mine = run.checkpoint("mine")
    src.rewind("t5.L0.h")
    src.restore(mine)                                         # reloaded -> source
    assert torch.equal(src.read(), run.read())
    # a run reloaded from an EARLIER checkpoint of the same lineage restores the later one
    early = _source(dbg, "lin-early")
    early.rewind("t4.start")
    early.dump(tmp_path / "early.pt", history="cursor")
    from_early = dbg.run_from(tmp_path / "early.pt", name="lin-e")
    from_early.restore(mine)
    assert torch.equal(from_early.read(), run.read())
    # refused: before the root, and another path before the root (not this lineage)
    before = _source(dbg, "lin-y")
    before.rewind("t3.L0.h")
    with pytest.raises(ValueError, match="before this run's root"):
        from_early.restore(before.checkpoint())
    other = dbg.run(P, name="lin-o", prefill=2)
    other.until("t5.start")
    with pytest.raises(ValueError, match="another path before this run's root"):
        from_early.restore(other.checkpoint())


def test_a_fingerprint_binds_content_and_ancestry(dbg, saved, tmp_path):
    path, _ = saved
    src = _source(dbg, "fp")
    run = dbg.run_from(path, name="fp-r")
    assert [m.fingerprint for m in run._trail] == [
        m.fingerprint for m in src._trail[-len(run._trail):]]
    verified = dbg.load_checkpoint(path).verify(dbg.adapter)
    assert verified["ok"] and verified["reached"] == src._trail[-1].fingerprint
    # a forged parent (same tensors, other claimed ancestry) is refused at rehydration
    data = torch.load(path, weights_only=True)
    doc = json.loads(data["document"])
    doc["marks"][-1]["cut"]["parent"] = "0" * 64
    data["document"] = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    torch.save(data, tmp_path / "forged.pt")
    with pytest.raises(ValueError, match="document differs"):
        states.load_file(tmp_path / "forged.pt")
    loaded = states.load_file(path)
    loaded.doc["marks"][-1]["cut"]["parent"] = "0" * 64       # past the seal, in memory
    with pytest.raises(ValueError, match="differs from its fingerprint"):
        loaded.run(dbg.adapter)


# ------------------------------------------------------------------ project store

def test_project_store_addressing_and_durable_checkpoints(case, tmp_path):
    root = tmp_path / "proj"
    dbg = attach(tiny_model(*case), project=root)
    src = _source(dbg)
    info = dbg.checkpoint(src, "cut")
    durable = src.checkpoint("again", durable=dbg.project)["durable"]
    src.until(f"t{LAST}.logits")
    want = src.read()
    other = attach(tiny_model(*case), project=root)            # another debugger, same project
    for ref in (info["state"], info["state"][:10], durable["record"]):
        run = other.run_from(ref, name="pr")
        assert run.where == CUT and run.core["record"].startswith(ref[:10])
        run.until(f"t{LAST}.logits")
        assert torch.equal(run.read(), want)
    restarted = other.restart("cut", verify=True)
    assert restarted.core["exact"] and restarted.core["verified"]["ok"]
    replayed = other.restart("cut", replay=True)
    assert replayed._trail[-1].fingerprint == restarted._trail[-1].fingerprint
    body = states.find_record(other.project, info["state"])["body"]
    shas = states._unpacked(body["tensors"]["sha256"])
    assert len(shas) == body["tensors"]["count"] > 0
    assert all(other.project.store.has_tensor(p["ref"])
               for p in states.load(info["state"], project=other.project).doc["parts"])
    with pytest.raises(Exception, match="record id"):
        other.run_from("zz-not-an-id")
    with pytest.raises(Exception, match="ambiguous|no checkpoint record"):
        other.run_from("0000000000")


# ------------------------------------------------------------------ refusals

def test_tamper_model_and_dtype_refusals(case, dbg, saved, tmp_path):
    path, _ = saved
    data = torch.load(path, weights_only=True)
    key = sorted(data["tensors"])[0]
    data["tensors"][key] = data["tensors"][key].clone()
    data["tensors"][key].view(-1)[0] += 1
    torch.save(data, tmp_path / "t.pt")
    with pytest.raises(ValueError, match="differs from its sha256"):
        states.load_file(tmp_path / "t.pt")
    other = attach(tiny_model(case[0], case[1], seed=7))       # same shape, other weights
    with pytest.raises(ValueError, match="another model"):
        other.run_from(path)
    replica = other.run_from(path, replica=True)
    assert replica.core["replica"] and replica.lineage["root"] == "checkpoint"
    with pytest.raises(ValueError, match="another model shape"):
        open_tiny("llama", "fp32").run_from(path)
    model = tiny_model(case[0], case[1])
    wide = attach(cast_weights(copy.deepcopy(model), torch.float64
                               if case[1] == "fp32" else torch.float32))
    with pytest.raises(ValueError, match="another (model|execution)"):   # the identity binds
        wide.run_from(path)                                            # the dtype too
    run = wide.run_from(path, replica=True)
    assert run.core["replica"] and run.where == CUT and run.core_check is None  # nothing captured
    # saved at an earlier boundary after running on, the logits the source computed are captured
    # and a replica is null-checked against them
    src = _source(dbg, "cap")
    src.until(f"t{LAST}.logits")
    record = src.dump(tmp_path / "cap.pt", at="t4.start")
    assert record["body"]["captured"] == [f"t{LAST}.logits"]
    check = wide.run_from(tmp_path / "cap.pt", replica=True).core_check
    assert check["same_dtype_and_identity"] is False and "argmax_equal" in check["checks"][
        f"t{LAST}.logits"]
    exact = dbg.run_from(tmp_path / "cap.pt").core_check
    assert exact["ok"] and exact["checks"][f"t{LAST}.logits"]["max_abs_diff"] == 0.0
    # a store blob changed on disk is refused when the checkpoint is used
    root = tmp_path / "proj"
    pdbg = attach(tiny_model(*case), project=root)
    info = pdbg.checkpoint(_source(pdbg, "tp"), "cut")
    part = states.load(info["state"], project=pdbg.project).doc["parts"][0]
    blob = root / "store" / "tensors" / f"{part['ref']}.safetensors"
    blob.write_bytes(blob.read_bytes()[:-1] + b"\1")
    with pytest.raises(ValueError, match="corrupt|differs"):
        attach(tiny_model(*case), project=root).run_from(info["state"])


def test_legacy_v1_core_files_still_load(dbg, tmp_path):
    src = dbg.run(P, name="v1src", prefill=2)
    src.until(f"t{LAST}.logits")
    core_file.dump_v1(src, tmp_path / "v1.pt", row=4)
    loaded = core_file.load(tmp_path / "v1.pt")
    assert isinstance(loaded, core_file.CoreFile)
    run = dbg.run_from(tmp_path / "v1.pt", name="v1")
    assert run.where == "t4.start" and run.core_check["ok"]
    assert run.lineage["root"] == "checkpoint"                # no parent recorded: foreign
    run.until(f"t{LAST}.logits")
    assert torch.equal(run.read(), src.read())
    br = dbg.branches(run, Margin(LAST, A, B), at=f"t{LAST}.L1.h")   # fresh() from the v1 root
    br.add("zero", [edit(f"t{LAST}.L1.m", 0.0, "scale")])
    assert len(br.run()) == 1
